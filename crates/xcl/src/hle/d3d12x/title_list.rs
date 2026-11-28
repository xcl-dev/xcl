//! Command lists of console code.
//!
//! The D3D12.X code compiled into a title (the SDK's inline functions) writes command records
//! straight into the list object when the list's buffer has room – +0x18 write pointer, +0x30
//! end, +0x28 the limit for dispatch records – and otherwise calls the list method the record
//! stands for, with the method's own arguments (HANDOFF: "inline writers"; e.g. 0xC000B600 +
//! a hardware topology, else #20 IASetPrimitiveTopology). A PC list object has other data at
//! those offsets. So console code gets a list object of the console's shape whose buffer never
//! has room: every inline write becomes the method call, and each method runs on the
//! (adopted) PC list behind it – with all of XCL's console slot handling.
//!
//! The object: the console fields, the PC list at `F_BACKING`; its vtable is a thunk per slot
//! (`mov rcx, [rcx + F_BACKING]` and on to the PC list's own vtable entry). QueryInterface
//! hands out the console object again, Release frees it with the PC list, and
//! ExecuteCommandLists runs the PC lists (`backing`).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows_sys::core::GUID;

use crate::util::HRESULT;

const LIST_SLOTS: usize = 600;
/// the object: console fields up to `ROOM`, no-room "buffer" at `ROOM` (with slack after it
/// for code that would write before checking), primary PC list at `F_PRIMARY` (identity /
/// refcount), currently active recording PC list at `F_BACKING` (swapped on mid-list sync split).
const OBJECT_BYTES: usize = 0x3000;
const ROOM: usize = 0x1000;
const F_PRIMARY: usize = 0x2FF0;
const F_BACKING: usize = 0x2FF8;
const F_WRITE: usize = 0x18;
const F_DISPATCH_END: usize = 0x28;
const F_END: usize = 0x30;
/// `XCL_LIST_TRACE`: the last `RING` slots called (u16 each) and a counter
const F_RING: usize = 0x2000;
const F_RING_AT: usize = 0x2100;
const RING: usize = 64;
/// Maximum number of mid-list split segments per command list recording.
const MAX_EXTRA_SEGMENTS: usize = 1024;

#[derive(Default)]
struct SegmentBook {
    /// Additional PC command lists recorded after `primary` in this recording (in order).
    extra_active: Vec<usize>,
    /// Closed extra PC command lists from earlier recordings, kept for reuse on future splits.
    spares: Vec<usize>,
    /// A PC command list recording in place of `primary`: at the title's Reset the GPU had not
    /// finished primary's last submit, and the PC does not let a list be submitted again before
    /// that (#1452, debug layer id 553). A console does.
    first: Option<usize>,
    /// Closed object that receives calls after an unrecoverable split failure, until Reset.
    failed: Option<usize>,
    /// The identity-bearing primary returned a failed Close and can never be Reset again.
    primary_broken: bool,
    /// Non-primary lists whose failed Close permanently forbids Reset/reuse.
    unresettable: Vec<usize>,
}

/// Wrapper address -> split segment book (populated lazily only for lists that split).
static SEGMENTS: Mutex<Option<HashMap<usize, SegmentBook>>> = Mutex::new(None);
/// Active recording PC command list address -> `title_list` wrapper address.
static WRAPPER_OF: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

struct Vtable(usize);

/// The console list vtable: QueryInterface / AddRef / Release / Close / Reset of their own,
/// every other slot a thunk to the currently active PC list's entry (`F_BACKING`).
fn vtable() -> usize {
    static V: OnceLock<Vtable> = OnceLock::new();
    V.get_or_init(|| {
        let mut v = vec![0usize; LIST_SLOTS];
        let trace = std::env::var_os("XCL_LIST_TRACE").is_some();
        for (slot, e) in v.iter_mut().enumerate() {
            *e = if trace { forward_traced(slot) } else { forward(slot) };
        }
        v[0] = qi as usize;
        v[1] = add_ref as usize;
        v[2] = release as usize;
        v[9] = close as usize;
        v[10] = reset as usize;
        v[139] = slot_139(v[139], &super::real::QUEUE_SHIFTED);
        Vtable(Box::leak(v.into_boxed_slice()).as_ptr() as usize)
    })
    .0
}

/// `mov rcx, [rcx + F_BACKING]; mov rax, [rcx]; jmp [rax + slot * 8]`
fn forward(slot: usize) -> usize {
    let mut c = vec![0x48, 0x8B, 0x89];
    c.extend_from_slice(&(F_BACKING as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0x8B, 0x01, 0xFF, 0xA0]);
    c.extend_from_slice(&((slot * 8) as u32).to_le_bytes());
    let p = crate::stubs::alloc_code(c.len());
    unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
    }
    p as usize
}

/// `forward` after `record(this, slot)` (argument registers, xmm0–3 kept)
fn forward_traced(slot: usize) -> usize {
    let mut c: Vec<u8> = vec![0x51, 0x52, 0x41, 0x50, 0x41, 0x51]; // push rcx, rdx, r8, r9
    c.extend_from_slice(&[0x48, 0x83, 0xEC, 0x68]); // sub rsp, 0x68
    for (reg, off) in [(0x44u8, 0x20u8), (0x4C, 0x30), (0x54, 0x40), (0x5C, 0x50)] {
        c.extend_from_slice(&[0xF3, 0x0F, 0x7F, reg, 0x24, off]); // movdqu [rsp+off], xmmN
    }
    c.push(0xBA); // mov edx, slot
    c.extend_from_slice(&(slot as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0xB8]); // mov rax, record
    c.extend_from_slice(&(record as usize as u64).to_le_bytes());
    c.extend_from_slice(&[0xFF, 0xD0]); // call rax
    for (reg, off) in [(0x44u8, 0x20u8), (0x4C, 0x30), (0x54, 0x40), (0x5C, 0x50)] {
        c.extend_from_slice(&[0xF3, 0x0F, 0x6F, reg, 0x24, off]); // movdqu xmmN, [rsp+off]
    }
    c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x68]); // add rsp, 0x68
    c.extend_from_slice(&[0x41, 0x59, 0x41, 0x58, 0x5A, 0x59]); // pop r9, r8, rdx, rcx
    c.extend_from_slice(&[0x48, 0x8B, 0x89]);
    c.extend_from_slice(&(F_BACKING as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0x8B, 0x01, 0xFF, 0xA0]);
    c.extend_from_slice(&((slot * 8) as u32).to_le_bytes());
    let p = crate::stubs::alloc_code(c.len());
    unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
    }
    p as usize
}

unsafe extern "system" fn record(this: *mut c_void, slot: u32) {
    let at = (this as usize + F_RING_AT) as *mut usize;
    let k = *at % RING;
    *((this as usize + F_RING + 2 * k) as *mut u16) = slot as u16;
    *at += 1;
}

/// #9 Close: closes the active recording segment (`F_BACKING`); a failure is logged with
/// the last methods called (`XCL_LIST_TRACE`). Prior split segments were closed on split.
unsafe extern "system" fn close(this: *mut c_void) -> HRESULT {
    let pc = *((this as usize + F_BACKING) as *const *mut c_void);
    let f: unsafe extern "system" fn(*mut c_void) -> HRESULT = std::mem::transmute(pc_slot(pc, 9));
    let hr = f(pc);
    super::capture::list_end(this as usize);
    if hr >= 0 {
        super::title_textures::note_list_closed(this as usize);
    }
    let n = *((this as usize + F_RING_AT) as *const usize);
    if hr < 0 && n > 0 {
        let first = n.saturating_sub(RING);
        let slots: Vec<String> = (first..n).map(|k| (*((this as usize + F_RING + 2 * (k % RING)) as *const u16)).to_string()).collect();
        crate::xlog_warn!("D3D12.X: console list Close → {:#x}; console slots called since Reset (last {}): {}", hr as u32, slots.len(), slots.join(" "));
    }
    // a new recording starts at Reset
    *((this as usize + F_RING_AT) as *mut usize) = 0;
    hr
}

fn recover_failed_with<T>(failed: usize, in_flight: bool, reuse: impl FnOnce(usize) -> Option<T>, replace: impl FnOnce() -> Option<T>) -> Option<T> {
    if in_flight { replace() } else { reuse(failed) }
}

/// #10 Reset(allocator, initial_pso): resets the primary PC list (`F_PRIMARY`) via `list_x::reset_pc`
/// (calling directly rather than through the dual-dispatch vtable thunk so console state always
/// resets even though this wrapper sits in `xcl.exe`), restores `F_BACKING = F_PRIMARY`, and moves
/// any extra split segments into the per-list spare pool. While the GPU has not finished
/// primary's last submit, the new recording goes onto a spare instead (`first`). The newer console
/// layout's Reset (#139) reaches it through `list_x::reset_x` on the active PC list.
pub(super) unsafe extern "system" fn reset(this: *mut c_void, alloc: usize, pso: usize) -> HRESULT {
    let primary = *((this as usize + F_PRIMARY) as *const *mut c_void);
    let adopted = super::real::is_adopted(primary);
    let (failed, primary_broken, first) = SEGMENTS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(this as usize)))
        .map_or((None, false, None), |b| (b.failed, b.primary_broken, b.first));
    let stand_in = if let Some(failed) = failed {
        recover_failed_with(
            failed,
            super::alloc_pool::in_flight(failed),
            |failed| (super::list_x::reset_pc(failed as *mut c_void, alloc, pso) >= 0).then_some(failed),
            || super::list_x::reset_stand_in(this, primary, alloc, pso),
        )
    } else if primary_broken {
        match first.filter(|&p| !super::alloc_pool::in_flight(p)) {
            Some(first) => (super::list_x::reset_pc(first as *mut c_void, alloc, pso) >= 0).then_some(first),
            None => super::list_x::reset_stand_in(this, primary, alloc, pso),
        }
    } else if adopted && super::alloc_pool::in_flight(primary as usize) {
        super::list_x::reset_stand_in(this, primary, alloc, pso)
    } else {
        None
    };
    let hr = if stand_in.is_some() {
        0
    } else if failed.is_some() || primary_broken {
        0x8000_4005u32 as HRESULT
    } else if adopted {
        super::list_x::reset_pc(primary, alloc, pso)
    } else {
        let f: unsafe extern "system" fn(*mut c_void, usize, usize) -> HRESULT = std::mem::transmute(pc_slot(primary, 10));
        f(primary, alloc, pso)
    };
    if hr >= 0 {
        super::title_textures::forget_list_closed(this as usize);
        let active = stand_in.unwrap_or(primary as usize);
        *((this as usize + F_BACKING) as *mut usize) = active;
        let moved = {
            let mut sg = SEGMENTS.lock().unwrap();
            let m = sg.get_or_insert_with(HashMap::new);
            if stand_in.is_some() || m.contains_key(&(this as usize)) {
                let b = m.entry(this as usize).or_default();
                let mut old = std::mem::take(&mut b.extra_active);
                old.extend(std::mem::replace(&mut b.first, stand_in).filter(|first| Some(*first) != stand_in));
                old.extend(b.failed.take().filter(|failed| Some(*failed) != stand_in));
                b.spares.extend(old.iter().copied());
                old
            } else {
                Vec::new()
            }
        };
        if !moved.is_empty() || stand_in.is_some() {
            let mut wg = WRAPPER_OF.lock().unwrap();
            let m = wg.get_or_insert_with(HashMap::new);
            for &seg in moved.iter().chain([primary as usize].iter()) {
                m.remove(&seg);
            }
            m.insert(active, this as usize);
            drop(wg);
            for seg in moved {
                super::bind::reset(seg as *mut c_void, 0, 0);
                super::list_x::clear_segment_state(seg);
                // a spare records again only after its next Reset: it does not hold its
                // allocator meanwhile
                super::alloc_pool::retire_recording(seg);
            }
            // nor does the previous recording a stand-in now replaces: it is not reset here either
            if stand_in.is_some() {
                super::bind::reset(primary, 0, 0);
                super::alloc_pool::retire_recording(primary as usize);
            }
        }
        *((this as usize + F_RING_AT) as *mut usize) = 0;
        super::capture::list_begin(this as usize, pso);
    }
    hr
}

/// #139: a title on the newer console list layout (`newer_layout`) calls Reset(allocator,
/// initial_pso, flags) there, which goes through wrapper `reset` so `F_PRIMARY` is reset and any
/// mid-list split segments are reclaimed (forwarded, it reset only the last continuation segment
/// and left `F_PRIMARY` on an old recording/allocator). A title on the older layout and the
/// runtime keep `forwarder`, the tail jump to `F_BACKING`: the backing list's own layout guard at
/// #139 tells a title by its return address, which a call from here would hide.
fn slot_139(forwarder: usize, newer_layout: &'static std::sync::atomic::AtomicBool) -> usize {
    super::guard::make_layout_split(super::guard::LayoutTargets { older: forwarder, newer: reset_139 as usize, pc: forwarder, newer_layout })
}

unsafe extern "system" fn reset_139(this: *mut c_void, alloc: usize, pso: usize, _flags: usize) -> HRESULT {
    reset(this, alloc, pso)
}

/// The console object for the PC list `pc` (adopted; one reference moves to it).
pub unsafe fn wrap(pc: *mut c_void) -> *mut c_void {
    if pc.is_null() {
        return pc;
    }
    let layout = std::alloc::Layout::from_size_align(OBJECT_BYTES, 64).unwrap();
    let o = std::alloc::alloc_zeroed(layout);
    if o.is_null() {
        return pc;
    }
    *(o as *mut usize) = vtable();
    let room = o as usize + ROOM;
    for f in [F_WRITE, F_DISPATCH_END, F_END] {
        *(o.add(f) as *mut usize) = room;
    }
    *(o.add(F_PRIMARY) as *mut usize) = pc as usize;
    *(o.add(F_BACKING) as *mut usize) = pc as usize;
    WRAPPER_OF.lock().unwrap().get_or_insert_with(HashMap::new).insert(pc as usize, o as usize);
    o as *mut c_void
}

/// The primary PC list behind a console list object (None: not one).
pub fn backing(list: *mut c_void) -> Option<*mut c_void> {
    if list.is_null() || !crate::gpu::shader::readable(list as usize, 8) {
        return None;
    }
    unsafe { (*(list as *const usize) == vtable()).then(|| *((list as usize + F_PRIMARY) as *const *mut c_void)) }
}

/// Expand a console list object into all of its recorded PC command list segments (in exact
/// recording order). Returns `false` if `list` is not a `title_list` wrapper.
pub fn push_segments(list: *mut c_void, out: &mut Vec<*mut c_void>) -> bool {
    let Some(primary) = backing(list) else {
        return false;
    };
    let sg = SEGMENTS.lock().unwrap();
    let book = sg.as_ref().and_then(|m| m.get(&(list as usize)));
    if let Some(failed) = book.and_then(|b| b.failed) {
        out.push(failed as *mut c_void);
        return true;
    }
    out.push(book.and_then(|b| b.first).map_or(primary, |f| f as *mut c_void));
    if let Some(book) = book {
        for &seg in &book.extra_active {
            out.push(seg as *mut c_void);
        }
    }
    true
}

/// Look up the `title_list` wrapper currently recording into `pc`.
pub(super) fn wrapper_of(pc: *mut c_void) -> Option<*mut c_void> {
    if pc.is_null() {
        return None;
    }
    WRAPPER_OF.lock().unwrap().as_ref()?.get(&(pc as usize)).map(|&w| w as *mut c_void)
}

/// Whether `wrapper` may allocate/activate `n` more split segments for this recording.
pub(super) fn can_split(wrapper: *mut c_void, n: usize) -> bool {
    let sg = SEGMENTS.lock().unwrap();
    sg.as_ref()
        .and_then(|m| m.get(&(wrapper as usize)))
        .is_none_or(|b| b.failed.is_none() && b.extra_active.len() + n <= MAX_EXTRA_SEGMENTS)
}

/// Reserve every bookkeeping slot a split will publish, before its producer is closed.
pub(super) fn prepare_split(wrapper: *mut c_void, n: usize) -> bool {
    let mut sg = SEGMENTS.lock().unwrap();
    let m = sg.get_or_insert_with(HashMap::new);
    if m.try_reserve(1).is_err() {
        return false;
    }
    let b = m.entry(wrapper as usize).or_default();
    if b.failed.is_some()
        || b.extra_active.len() + n > MAX_EXTRA_SEGMENTS
        || b.extra_active.try_reserve(n).is_err()
        || b.spares.try_reserve(n).is_err()
        // (the producer, and a stand-in for it whose empty recording failed: `replace_failed`)
        || b.unresettable.try_reserve(2).is_err()
    {
        return false;
    }
    let mut wg = WRAPPER_OF.lock().unwrap();
    wg.get_or_insert_with(HashMap::new).try_reserve(1).is_ok()
}

/// Reserve only the bookkeeping a fail-closed transition or successful standby return needs.
/// Unlike a recording split, this does not consume physical-segment capacity.
pub(super) fn prepare_fail_closed(wrapper: *mut c_void) -> bool {
    let mut sg = SEGMENTS.lock().unwrap();
    let m = sg.get_or_insert_with(HashMap::new);
    if m.try_reserve(1).is_err() {
        return false;
    }
    let b = m.entry(wrapper as usize).or_default();
    if b.failed.is_some() || b.spares.try_reserve(1).is_err() || b.unresettable.try_reserve(1).is_err() {
        return false;
    }
    let mut wg = WRAPPER_OF.lock().unwrap();
    wg.get_or_insert_with(HashMap::new).try_reserve(1).is_ok()
}

/// Pop a previously created closed extra PC command list from `wrapper`'s spare pool for reuse
/// (the last one whose submits the GPU has finished).
pub(super) fn pop_spare(wrapper: *mut c_void) -> Option<*mut c_void> {
    let mut sg = SEGMENTS.lock().unwrap();
    let spares = &mut sg.as_mut()?.get_mut(&(wrapper as usize))?.spares;
    let k = spares.iter().rposition(|&p| !super::alloc_pool::in_flight(p))?;
    Some(spares.remove(k) as *mut c_void)
}

/// Give back to `wrapper`'s spare pool a closed PC command list that did not become a segment.
pub(super) fn put_spare(wrapper: *mut c_void, pc: *mut c_void) {
    super::alloc_pool::retire_recording(pc as usize);
    let mut sg = SEGMENTS.lock().unwrap();
    sg.get_or_insert_with(HashMap::new).entry(wrapper as usize).or_default().spares.push(pc as usize);
}

/// Take segment `pc` out of `wrapper`'s recording, so it is not submitted. Its failed Close left
/// it open, so quarantine it until wrapper destruction rather than treating it as a closed spare.
pub(super) fn drop_segment(wrapper: *mut c_void, pc: *mut c_void) {
    let mut sg = SEGMENTS.lock().unwrap();
    if let Some(b) = sg.as_mut().and_then(|m| m.get_mut(&(wrapper as usize))) {
        b.extra_active.retain(|&p| p != pc as usize);
        b.unresettable.push(pc as usize);
    }
}

/// The closed PC command lists `wrapper` keeps for reuse.
#[cfg(test)]
pub(super) fn spares_for_test(wrapper: *mut c_void) -> Vec<usize> {
    SEGMENTS.lock().unwrap().as_ref().and_then(|m| m.get(&(wrapper as usize))).map_or(Vec::new(), |b| b.spares.clone())
}

/// The closed list `stand` takes the place of `failed`, a list of `wrapper`'s recording whose
/// Close failed: submitted, `failed` would remove the device, and no Reset opens it again. It is
/// kept only to be released with the wrapper (the primary is that anyway); a failed primary makes
/// the title's next Reset record on another list.
pub(super) fn replace_failed(wrapper: *mut c_void, failed: *mut c_void, stand: *mut c_void) {
    let primary = unsafe { *((wrapper as usize + F_PRIMARY) as *const usize) };
    let (failed, stand) = (failed as usize, stand as usize);
    let mut sg = SEGMENTS.lock().unwrap();
    let b = sg.get_or_insert_with(HashMap::new).entry(wrapper as usize).or_default();
    if b.first == Some(failed) || (b.first.is_none() && failed == primary) {
        b.first = Some(stand);
    } else if let Some(k) = b.extra_active.iter().position(|&p| p == failed) {
        b.extra_active[k] = stand;
    } else {
        return;
    }
    if failed == primary {
        b.primary_broken = true;
    } else {
        b.unresettable.push(failed);
    }
}

/// Keep `pc`, a list of `wrapper` in no known state, only to release it with the wrapper.
pub(super) fn quarantine(wrapper: *mut c_void, pc: *mut c_void) {
    let mut sg = SEGMENTS.lock().unwrap();
    sg.get_or_insert_with(HashMap::new).entry(wrapper as usize).or_default().unresettable.push(pc as usize);
}

/// Add the closed segment `pc` to `wrapper`'s recording, after the ones it has.
pub(super) fn push_segment(wrapper: *mut c_void, pc: *mut c_void) {
    let mut sg = SEGMENTS.lock().unwrap();
    let extra = &mut sg.get_or_insert_with(HashMap::new).entry(wrapper as usize).or_default().extra_active;
    extra.push(pc as usize);
    SEGMENTS_PEAK.fetch_max(extra.len(), std::sync::atomic::Ordering::Relaxed);
}

/// the most extra segments a recording has had (of `MAX_EXTRA_SEGMENTS`)
static SEGMENTS_PEAK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The most extra segments a title list's recording has had so far (the put-off dispatches'
/// line: `pso_async::put_off_line`).
pub(super) fn segments_peak() -> usize {
    SEGMENTS_PEAK.load(std::sync::atomic::Ordering::Relaxed)
}

/// Switch `wrapper` from closed segment `old_pc` to newly opened segment `new_pc`.
pub(super) unsafe fn activate_segment(wrapper: *mut c_void, old_pc: *mut c_void, new_pc: *mut c_void) {
    *((wrapper as usize + F_BACKING) as *mut usize) = new_pc as usize;
    {
        let mut sg = SEGMENTS.lock().unwrap();
        let extra = &mut sg.get_or_insert_with(HashMap::new).entry(wrapper as usize).or_default().extra_active;
        extra.push(new_pc as usize);
        SEGMENTS_PEAK.fetch_max(extra.len(), std::sync::atomic::Ordering::Relaxed);
    }
    let mut wg = WRAPPER_OF.lock().unwrap();
    let m = wg.get_or_insert_with(HashMap::new);
    m.remove(&(old_pc as usize));
    m.insert(new_pc as usize, wrapper as usize);
}

/// Route a terminally failed recording to a known-closed object until the title Resets it.
pub(super) unsafe fn fail_closed(wrapper: *mut c_void, old_pc: *mut c_void, closed_pc: *mut c_void, producer_broken: bool) {
    *((wrapper as usize + F_BACKING) as *mut usize) = closed_pc as usize;
    let mut sg = SEGMENTS.lock().unwrap();
    let b = sg.get_or_insert_with(HashMap::new).entry(wrapper as usize).or_default();
    b.failed = Some(closed_pc as usize);
    if producer_broken {
        if old_pc as usize == *((wrapper as usize + F_PRIMARY) as *const usize) {
            b.primary_broken = true;
        } else {
            b.extra_active.retain(|&p| p != old_pc as usize);
            if b.first == Some(old_pc as usize) {
                b.first = None;
            }
            b.unresettable.push(old_pc as usize);
        }
    }
    drop(sg);
    let mut wg = WRAPPER_OF.lock().unwrap();
    let m = wg.get_or_insert_with(HashMap::new);
    m.remove(&(old_pc as usize));
    m.insert(closed_pc as usize, wrapper as usize);
}

/// For `KickoffX` on an already-split list: returns `(primary, prior_closed_segments, extra_to_reclaim)`
/// when `active_pc != primary`, switches `F_BACKING` back to `primary`, and moves all extra
/// segments (including `active_pc`) into `spares` so recording resumes cleanly on `primary`.
/// (`primary` here is the recording's first segment: `first` when one stands in.)
pub(super) unsafe fn reclaim_for_kickoff(active_pc: *mut c_void) -> Option<(*mut c_void, Vec<*mut c_void>, Vec<usize>)> {
    let wrapper = wrapper_of(active_pc)?;
    let mut sg = SEGMENTS.lock().unwrap();
    let book = sg.as_mut().and_then(|m| m.get_mut(&(wrapper as usize)))?;
    if book.failed.is_some() {
        return None;
    }
    let primary = book.first.map_or_else(|| *((wrapper as usize + F_PRIMARY) as *const *mut c_void), |f| f as *mut c_void);
    if primary == active_pc {
        return None;
    }
    let (prior, reclaimed) = {
        let extra = std::mem::take(&mut book.extra_active);
        let mut prior = Vec::with_capacity(extra.len());
        prior.push(primary);
        for &seg in &extra {
            if seg != active_pc as usize {
                prior.push(seg as *mut c_void);
            }
        }
        book.spares.extend(extra.iter().copied());
        (prior, extra)
    };
    drop(sg);
    *((wrapper as usize + F_BACKING) as *mut usize) = primary as usize;
    {
        let mut wg = WRAPPER_OF.lock().unwrap();
        if let Some(m) = wg.as_mut() {
            for &seg in &reclaimed {
                m.remove(&seg);
            }
            m.insert(primary as usize, wrapper as usize);
        }
    }
    Some((primary, prior, reclaimed))
}

unsafe fn pc_slot(pc: *mut c_void, slot: usize) -> usize {
    *(*(pc as *const *const usize)).add(slot)
}

/// QueryInterface: the primary PC list answers; where it hands out itself, the console object goes out.
/// Command list interfaces of the console's headers that the PC does not know: the console's
/// ID3D12GraphicsCommandList5 / 6 (the PC's data1–data3, another tail – titles ask for them
/// and use the answer without a check). The console object has the console's whole vtable,
/// so it is the answer.
const CONSOLE_LIST_IIDS: [(u32, u16, u16, [u8; 8]); 2] = [
    (0x5505_0859, 0x4024, 0x474c, [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74]),
    (0xc382_7890, 0x4024, 0x474c, [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74]),
];

unsafe extern "system" fn qi(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let pc = *((this as usize + F_PRIMARY) as *const *mut c_void);
    let f: unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(pc_slot(pc, 0));
    if !riid.is_null() && !ppv.is_null() {
        let g = &*riid;
        if CONSOLE_LIST_IIDS.contains(&(g.data1, g.data2, g.data3, g.data4)) {
            let add_ref_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(pc, 1));
            add_ref_fn(pc);
            *ppv = this;
            return 0;
        }
    }
    let hr = f(pc, riid, ppv);
    if hr >= 0 && !ppv.is_null() && *ppv == pc {
        *ppv = this;
    }
    hr
}

/// AddRef: always references the primary PC list (`F_PRIMARY`).
unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    let pc = *((this as usize + F_PRIMARY) as *const *mut c_void);
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(pc, 1));
    f(pc)
}

/// Release: the console object (and any extra split segment PC lists) goes with the primary PC
/// list's last reference.
unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    let pc = *((this as usize + F_PRIMARY) as *const *mut c_void);
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(pc, 2));
    let n = f(pc);
    if n == 0 {
        super::capture::forget(this as usize);
        super::title_textures::forget_list_closed(this as usize);
        let book = SEGMENTS.lock().unwrap().as_mut().and_then(|m| m.remove(&(this as usize)));
        {
            let mut wg = WRAPPER_OF.lock().unwrap();
            if let Some(m) = wg.as_mut() {
                if m.get(&(pc as usize)) == Some(&(this as usize)) {
                    m.remove(&(pc as usize));
                }
                if let Some(ref b) = book {
                    for &seg in b.extra_active.iter().chain(b.spares.iter()).chain(b.first.iter()).chain(b.failed.iter()).chain(b.unresettable.iter()) {
                        if m.get(&seg) == Some(&(this as usize)) {
                            m.remove(&seg);
                        }
                    }
                }
            }
        }
        if let Some(b) = book {
            for seg in b.extra_active.into_iter().chain(b.spares.into_iter()).chain(b.first).chain(b.failed).chain(b.unresettable) {
                let raw = seg as *mut c_void;
                super::alloc_pool::forget_list(seg);
                super::bind::forget(raw);
                let rel: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(raw, 2));
                rel(raw);
            }
        }
        // no forget of `pc` here: the adopted primary forgot its state in its own final Release
        // (`real::cl_release`), and after `f(pc)` the address may already be a new list's
        std::alloc::dealloc(this as *mut u8, std::alloc::Layout::from_size_align(OBJECT_BYTES, 64).unwrap());
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in PC object: slot 5 returns `this`, slots 0 / 2 count.
    #[repr(C)]
    struct Fake {
        vt: *const usize,
        refs: u32,
    }

    unsafe extern "system" fn fake_qi(this: *mut c_void, _riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
        (*(this as *mut Fake)).refs += 1;
        *ppv = this;
        0
    }
    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *mut Fake)).refs += 1;
        (*(this as *mut Fake)).refs
    }
    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *mut Fake)).refs -= 1;
        (*(this as *mut Fake)).refs
    }
    unsafe extern "system" fn fake_this(this: *mut c_void, a: u64, b: u64, c: u64, d: u64) -> u64 {
        this as u64 + a + 10 * b + 100 * c + 1000 * d
    }

    const REUSED_PSO: usize = 0x7E57_D102;

    unsafe extern "system" fn reuse_release(this: *mut c_void) -> u32 {
        let fake = &mut *(this as *mut Fake);
        fake.refs -= 1;
        if fake.refs == 0 {
            // The adopted list forgets before the native runtime frees it. Model an allocator
            // immediately handing the same address to a new list before Release returns.
            super::super::bind::forget(this);
            super::super::bind::reset(this, 0, REUSED_PSO);
        }
        fake.refs
    }

    #[test]
    fn wrapper_release_does_not_forget_a_new_list_at_the_reused_primary_address() {
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[0] = fake_qi as usize;
        vt[1] = fake_add_ref as usize;
        vt[2] = reuse_release as usize;
        let primary = Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        unsafe {
            let wrapper = wrap(primary as *mut c_void);
            super::super::bind::reset(primary as *mut c_void, 0, REUSED_PSO + 1);
            let rel: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(*(*(wrapper as *const *const usize)).add(2));
            assert_eq!(rel(wrapper), 0);
            assert_eq!(super::super::bind::tracked_pso(primary as *mut c_void), REUSED_PSO);
            super::super::bind::forget(primary as *mut c_void);
            drop(Box::from_raw(primary));
        }
    }

    #[test]
    fn split_limit_counts_every_physical_extra_segment() {
        let wrapper = 0x7ffe_3001usize as *mut c_void;
        {
            let mut sg = SEGMENTS.lock().unwrap();
            sg.get_or_insert_with(HashMap::new).insert(
                wrapper as usize,
                SegmentBook { extra_active: (0..MAX_EXTRA_SEGMENTS - 1).map(|i| 0x8000 + i).collect(), ..Default::default() },
            );
        }
        assert!(can_split(wrapper, 1));
        assert!(!can_split(wrapper, 2), "a deferred dispatch needs two physical segments");
        assert!(!prepare_split(wrapper, 2));
        assert_eq!(SEGMENTS.lock().unwrap().as_ref().unwrap()[&(wrapper as usize)].extra_active.len(), MAX_EXTRA_SEGMENTS - 1);
        assert!(prepare_split(wrapper, 1));
        push_segment(wrapper, 0x9000usize as *mut c_void);
        assert!(!can_split(wrapper, 1));
        assert!(can_split(wrapper, 0));
        assert!(prepare_fail_closed(wrapper), "Kickoff standby does not consume segment capacity");
        SEGMENTS.lock().unwrap().as_mut().unwrap().get_mut(&(wrapper as usize)).unwrap().failed = Some(0x9100);
        assert!(!prepare_fail_closed(wrapper), "an already failed recording cannot preflight again");
        SEGMENTS.lock().unwrap().as_mut().unwrap().remove(&(wrapper as usize));
    }

    /// A list whose Close failed leaves the recording for the closed list in its place (the
    /// primary as the recording's first list too), a quarantined list is not submitted, and all
    /// of them are released once, with the wrapper.
    #[test]
    fn a_list_whose_close_failed_gives_its_place_and_goes_with_the_wrapper() {
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[0] = fake_qi as usize;
        vt[1] = fake_add_ref as usize;
        vt[2] = fake_release as usize;
        let fake = || Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        let [primary, seg, stand, stand2, odd] = [fake(), fake(), fake(), fake(), fake()];
        let p = |f: *mut Fake| f as *mut c_void;
        unsafe {
            let o = wrap(p(primary));
            let segs = |o| {
                let mut s = Vec::new();
                assert!(push_segments(o, &mut s));
                s
            };
            replace_failed(o, p(primary), p(stand));
            assert_eq!(segs(o), vec![p(stand)]);
            activate_segment(o, p(primary), p(seg));
            assert_eq!(segs(o), vec![p(stand), p(seg)]);
            replace_failed(o, p(seg), p(stand2));
            assert_eq!(segs(o), vec![p(stand), p(stand2)]);
            quarantine(o, p(odd));
            assert_eq!(segs(o), vec![p(stand), p(stand2)], "a quarantined list is not submitted");
            {
                let books = SEGMENTS.lock().unwrap();
                let b = &books.as_ref().unwrap()[&(o as usize)];
                assert!(b.primary_broken);
                assert_eq!(b.unresettable, vec![seg as usize, odd as usize]);
            }
            let rel: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(o as *const *const usize)).add(2));
            assert_eq!(rel(o), 0);
            for f in [primary, seg, stand, stand2, odd] {
                assert_eq!((*f).refs, 0, "released once");
                drop(Box::from_raw(f));
            }
        }
    }

    #[test]
    fn failed_primary_routes_submission_and_calls_to_closed_standby() {
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[0] = fake_qi as usize;
        vt[1] = fake_add_ref as usize;
        vt[2] = fake_release as usize;
        let primary = Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        let standby = Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        unsafe {
            let wrapper = wrap(primary as *mut c_void);
            fail_closed(wrapper, primary as *mut c_void, standby as *mut c_void, true);
            assert_eq!(*((wrapper as usize + F_BACKING) as *const usize), standby as usize);
            assert!(!can_split(wrapper, 1));
            let mut submitted = Vec::new();
            assert!(push_segments(wrapper, &mut submitted));
            assert_eq!(submitted, vec![standby as *mut c_void]);
            {
                let books = SEGMENTS.lock().unwrap();
                let book = &books.as_ref().unwrap()[&(wrapper as usize)];
                assert!(book.primary_broken);
                assert_eq!(book.failed, Some(standby as usize));
            }

            let rel: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(wrapper as *const *const usize)).add(2));
            assert_eq!(rel(wrapper), 0);
            drop(Box::from_raw(primary));
            drop(Box::from_raw(standby));
        }
    }

    #[test]
    fn failed_continuation_reset_keeps_the_closed_producer_reusable() {
        let producer = 0x7ffe_4002usize as *mut c_void;
        let standby = 0x7ffe_4003usize as *mut c_void;
        let mut object = vec![0usize; OBJECT_BYTES / std::mem::size_of::<usize>()];
        let wrapper = object.as_mut_ptr() as *mut c_void;
        unsafe {
            *((wrapper as usize + F_PRIMARY) as *mut usize) = producer as usize;
            fail_closed(wrapper, producer, standby, false);
        }
        let books = SEGMENTS.lock().unwrap();
        let book = &books.as_ref().unwrap()[&(wrapper as usize)];
        assert_eq!(book.failed, Some(standby as usize));
        assert!(!book.primary_broken);
        assert!(book.unresettable.is_empty());
        drop(books);
        SEGMENTS.lock().unwrap().as_mut().unwrap().remove(&(wrapper as usize));
        WRAPPER_OF.lock().unwrap().as_mut().unwrap().remove(&(standby as usize));
    }

    #[test]
    fn in_flight_failed_standby_is_replaced_instead_of_reused() {
        use std::cell::Cell;
        let reused = Cell::new(false);
        let replaced = Cell::new(false);
        let selected = recover_failed_with(
            0x7100,
            true,
            |_| {
                reused.set(true);
                Some(0x7100)
            },
            || {
                replaced.set(true);
                Some(0x7200)
            },
        );
        assert_eq!(selected, Some(0x7200));
        assert!(!reused.get() && replaced.get());
    }

    /// The SDK's inline writers check `write < end` (or `write + n <= end`): never room here.
    #[test]
    fn console_list_objects_have_no_room_and_forward_to_the_pc_list() {
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[0] = fake_qi as usize;
        vt[1] = fake_add_ref as usize;
        vt[2] = fake_release as usize;
        vt[20] = fake_this as usize;
        let fake = Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        unsafe {
            let o = wrap(fake as *mut c_void);
            assert_eq!(backing(o), Some(fake as *mut c_void));
            assert_eq!(backing(fake as *mut c_void), None);
            let (w, e) = (*((o as usize + F_WRITE) as *const usize), *((o as usize + F_END) as *const usize));
            assert!(w >= e && *((o as usize + F_DISPATCH_END) as *const usize) == w);
            let slot = |i: usize| *(*(o as *const *const usize)).add(i);
            // a method: the PC list's, with the PC list as `this` and the arguments kept
            let m: unsafe extern "system" fn(*mut c_void, u64, u64, u64, u64) -> u64 = std::mem::transmute(slot(20));
            assert_eq!(m(o, 1, 2, 3, 4), fake as u64 + 4321);
            // QueryInterface hands out the console object – also for the console-only list IIDs
            let q: unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(slot(0));
            let mut out = std::ptr::null_mut();
            assert_eq!(q(o, std::ptr::null(), &mut out), 0);
            assert_eq!(out, o);
            let (d1, d2, d3, d4) = CONSOLE_LIST_IIDS[0];
            let gcl5 = GUID { data1: d1, data2: d2, data3: d3, data4: d4 };
            let mut out = std::ptr::null_mut();
            assert_eq!(q(o, &gcl5, &mut out), 0);
            assert_eq!(out, o);
            let r: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(slot(2));
            assert_eq!(r(o), 2);
            assert_eq!(r(o), 1);
            assert_eq!(r(o), 0); // the object is freed with it
            drop(Box::from_raw(fake));
        }
    }

    #[test]
    fn reset_releases_retired_split_segment_from_alloc_pool() {
        unsafe extern "system" fn fake_reset(_this: *mut c_void, _alloc: usize, _pso: usize) -> HRESULT {
            0
        }
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[0] = fake_qi as usize;
        vt[1] = fake_add_ref as usize;
        vt[2] = fake_release as usize;
        vt[10] = fake_reset as usize;
        let primary = Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        let seg = Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        let alloc1 = 0x7E57_A110usize;
        let alloc2 = 0x7E57_A120usize;
        unsafe {
            let w = wrap(primary as *mut c_void);
            super::super::alloc_pool::created_on(seg as usize, alloc1, 0);
            assert_eq!(super::super::alloc_pool::recording_backing_for_test(seg as usize), Some(alloc1));
            activate_segment(w, primary as *mut c_void, seg as *mut c_void);
            assert_eq!(reset(w, alloc2, 0), 0);
            assert_eq!(
                super::super::alloc_pool::recording_backing_for_test(seg as usize),
                None,
                "retired split segment must be forgotten by alloc_pool so its allocator can be reused"
            );
            assert_eq!(release(w), 0);
            drop(Box::from_raw(primary));
            drop(Box::from_raw(seg));
        }
    }

    #[test]
    fn release_does_not_remove_reused_pc_from_wrapper_map() {
        static REUSED_WRAPPER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        unsafe extern "system" fn fake_release_reusing_pc(this: *mut c_void) -> u32 {
            let f = &mut *(this as *mut Fake);
            f.refs -= 1;
            if f.refs == 0 {
                // Simulate another thread immediately creating a new wrapper on the same PC address.
                f.refs = 1;
                let w2 = wrap(this);
                REUSED_WRAPPER.store(w2 as usize, std::sync::atomic::Ordering::Release);
                return 0;
            }
            f.refs
        }
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[0] = fake_qi as usize;
        vt[1] = fake_add_ref as usize;
        vt[2] = fake_release_reusing_pc as usize;
        let fake = Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        REUSED_WRAPPER.store(0, std::sync::atomic::Ordering::Release);
        unsafe {
            let w1 = wrap(fake as *mut c_void);
            assert_eq!(release(w1), 0);
            let w2 = REUSED_WRAPPER.load(std::sync::atomic::Ordering::Acquire) as *mut c_void;
            assert!(!w2.is_null());
            assert_eq!(
                wrapper_of(fake as *mut c_void),
                Some(w2),
                "old wrapper release must not erase the new wrapper registered for a reused pc address"
            );
            // Switch vtable to normal fake_release so w2 cleans up without recursing.
            let mut vt2 = vt.clone();
            vt2[2] = fake_release as usize;
            (*fake).vt = vt2.as_ptr();
            assert_eq!(release(w2), 0);
            drop(Box::from_raw(fake));
        }
    }

    #[repr(C)]
    struct ResetFake {
        vt: *const usize,
        refs: u32,
        reset_calls: u32,
        slot139_calls: u32,
        last_alloc: usize,
        last_pso: usize,
        last_arg3: usize,
    }

    unsafe extern "system" fn rfake_reset(this: *mut c_void, alloc: usize, pso: usize) -> HRESULT {
        let f = &mut *(this as *mut ResetFake);
        f.reset_calls += 1;
        f.last_alloc = alloc;
        f.last_pso = pso;
        0
    }
    unsafe extern "system" fn rfake_slot139(this: *mut c_void, a1: usize, a2: usize, a3: usize) -> HRESULT {
        let f = &mut *(this as *mut ResetFake);
        f.slot139_calls += 1;
        f.last_alloc = a1;
        f.last_pso = a2;
        f.last_arg3 = a3;
        0
    }
    unsafe extern "system" fn rfake_release(this: *mut c_void) -> u32 {
        let f = &mut *(this as *mut ResetFake);
        f.refs -= 1;
        f.refs
    }

    /// Newer-layout slot #139 after a mid-list split resets `F_PRIMARY`, restores `F_BACKING`,
    /// and moves split segments to `spares` so stale primary lists are never resubmitted; on the
    /// older layout slot #139 still forwards unchanged to `F_BACKING`.
    #[test]
    fn newer_layout_slot_139_resets_primary_and_reclaims_split_segments() {
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[2] = rfake_release as usize;
        vt[10] = rfake_reset as usize;
        vt[139] = rfake_slot139 as usize;
        let primary = Box::into_raw(Box::new(ResetFake { vt: vt.as_ptr(), refs: 1, reset_calls: 0, slot139_calls: 0, last_alloc: 0, last_pso: 0, last_arg3: 0 }));
        let seg1 = Box::into_raw(Box::new(ResetFake { vt: vt.as_ptr(), refs: 1, reset_calls: 0, slot139_calls: 0, last_alloc: 0, last_pso: 0, last_arg3: 0 }));
        unsafe {
            let o = wrap(primary as *mut c_void);
            activate_segment(o, primary as *mut c_void, seg1 as *mut c_void);
            assert_eq!(*((o as usize + F_BACKING) as *const usize), seg1 as usize);

            // Not a title caller: forwards slot #139 to the active backing segment without resetting.
            static NEWER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
            let s: unsafe extern "system" fn(*mut c_void, usize, usize, usize) -> HRESULT = std::mem::transmute(slot_139(forward(139), &NEWER));
            assert_eq!(s(o, 11, 22, 33), 0);
            assert_eq!((*seg1).slot139_calls, 1);
            assert_eq!(((*seg1).last_alloc, (*seg1).last_pso, (*seg1).last_arg3), (11, 22, 33));
            assert_eq!((*primary).reset_calls, 0);

            // Newer layout: runs wrapper Reset on F_PRIMARY and reclaims split segments.
            assert_eq!(reset_139(o, 0xA110_C002, 0x5555, 0), 0);
            assert_eq!((*primary).reset_calls, 1, "primary must receive Reset");
            assert_eq!((*primary).last_alloc, 0xA110_C002);
            assert_eq!((*primary).last_pso, 0x5555);
            assert_eq!((*seg1).reset_calls, 0, "split continuation must not receive #139 Reset");
            assert_eq!(*((o as usize + F_BACKING) as *const usize), primary as usize);
            assert_eq!(wrapper_of(primary as *mut c_void), Some(o));
            assert_eq!(wrapper_of(seg1 as *mut c_void), None);
            let mut segs = Vec::new();
            assert!(push_segments(o, &mut segs));
            assert_eq!(segs, vec![primary as *mut c_void]);
            assert_eq!(pop_spare(o), Some(seg1 as *mut c_void));

            let rel: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(o as *const *const usize)).add(2));
            assert_eq!(rel(o), 0);
            drop(Box::from_raw(primary));
            drop(Box::from_raw(seg1));
        }
    }

    unsafe extern "system" fn older_139(this: *mut c_void, a1: usize, a2: usize, a3: usize) -> HRESULT {
        rfake_slot139(this, a1, a2, a3);
        1
    }
    unsafe extern "system" fn newer_139(_: *mut c_void, _: usize, _: usize, _: usize) -> HRESULT {
        2
    }
    unsafe extern "system" fn pc_139(_: *mut c_void, _: usize, _: usize, _: usize) -> HRESULT {
        3
    }

    /// A title calling #139 reaches the backing list's layout guard (as `real.rs` installs it)
    /// with its own return address: on the older layout the older handler runs with the title's
    /// arguments, not the PC method; on the newer layout the primary is reset. The layout flag is
    /// the test's own, so no other test sees it change.
    #[test]
    fn slot_139_keeps_the_title_caller_for_the_backing_layout_guard() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE};
        static NEWER: AtomicBool = AtomicBool::new(false);
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[2] = rfake_release as usize;
        vt[10] = rfake_reset as usize;
        vt[139] = super::super::guard::make_layout_split(super::super::guard::LayoutTargets {
            older: older_139 as usize,
            newer: newer_139 as usize,
            pc: pc_139 as usize,
            newer_layout: &NEWER,
        });
        let primary = Box::into_raw(Box::new(ResetFake { vt: vt.as_ptr(), refs: 1, reset_calls: 0, slot139_calls: 0, last_alloc: 0, last_pso: 0, last_arg3: 0 }));
        let seg1 = Box::into_raw(Box::new(ResetFake { vt: vt.as_ptr(), refs: 1, reset_calls: 0, slot139_calls: 0, last_alloc: 0, last_pso: 0, last_arg3: 0 }));
        let s = slot_139(forward(139), &NEWER);
        // the title: `sub rsp, 0x28; mov rax, s; call rax; add rsp, 0x28; ret` in a mapped image
        let code = unsafe { VirtualAlloc(std::ptr::null(), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) as usize };
        assert_ne!(code, 0);
        let mut c = vec![0x48, 0x83, 0xEC, 0x28, 0x48, 0xB8];
        c.extend_from_slice(&(s as u64).to_le_bytes());
        c.extend_from_slice(&[0xFF, 0xD0, 0x48, 0x83, 0xC4, 0x28, 0xC3]);
        crate::modules::register_mapped(Arc::new(crate::modules::MappedModule {
            name: "title_list_139.exe".into(), path: "title_list_139.exe".into(), base: code, size: 0x1000, entry_rva: 0, is_dll: false,
            exports: Default::default(), tls_callbacks: Mutex::new(Vec::new()), no_thread_calls: AtomicBool::new(false),
            initialized: AtomicBool::new(true), stack_reserve: 0,
        }));
        unsafe {
            std::ptr::copy_nonoverlapping(c.as_ptr(), code as *mut u8, c.len());
            let title: unsafe extern "system" fn(*mut c_void, usize, usize, usize) -> HRESULT = std::mem::transmute(code);
            let runtime: unsafe extern "system" fn(*mut c_void, usize, usize, usize) -> HRESULT = std::mem::transmute(s);
            let o = wrap(primary as *mut c_void);
            activate_segment(o, primary as *mut c_void, seg1 as *mut c_void);

            // older layout: the title's call reaches the older handler on the active segment
            assert_eq!(title(o, 11, 22, 33), 1, "older layout: the older handler, not the PC method");
            assert_eq!(((*seg1).slot139_calls, (*seg1).last_alloc, (*seg1).last_pso, (*seg1).last_arg3), (1, 11, 22, 33));
            assert_eq!((*primary).reset_calls, 0);
            // the runtime still reaches the PC method
            assert_eq!(runtime(o, 0, 0, 0), 3);

            // newer layout: the title's call resets the primary and reclaims the split segment
            NEWER.store(true, Ordering::Release);
            assert_eq!(title(o, 0xA110_C002, 0x5555, 0), 0);
            assert_eq!(((*primary).reset_calls, (*primary).last_alloc, (*primary).last_pso), (1, 0xA110_C002, 0x5555));
            assert_eq!((*seg1).slot139_calls, 1);
            assert_eq!(*((o as usize + F_BACKING) as *const usize), primary as usize);
            assert_eq!(pop_spare(o), Some(seg1 as *mut c_void));

            crate::modules::unregister_mapped(code);
            VirtualFree(code as *mut _, 0, MEM_RELEASE);
            let rel: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(o as *const *const usize)).add(2));
            assert_eq!(rel(o), 0);
            drop(Box::from_raw(primary));
            drop(Box::from_raw(seg1));
        }
    }

    /// End-to-end `title_list` lifecycle for `title_textures::CLOSED`:
    /// failed Close does not record, successful Close records the wrapper pointer, failed Reset
    /// keeps it, successful Reset clears it, non-final Release keeps it, and final Release clears
    /// it before freeing the wrapper address.
    #[test]
    fn title_list_close_reset_and_release_maintain_closed_tracking() {
        #[repr(C)]
        struct LifecycleFake {
            vt: *const usize,
            refs: u32,
            close_hr: HRESULT,
            reset_hr: HRESULT,
        }
        unsafe extern "system" fn lf_add_ref(this: *mut c_void) -> u32 {
            let f = &mut *(this as *mut LifecycleFake);
            f.refs += 1;
            f.refs
        }
        unsafe extern "system" fn lf_release(this: *mut c_void) -> u32 {
            let f = &mut *(this as *mut LifecycleFake);
            f.refs -= 1;
            f.refs
        }
        unsafe extern "system" fn lf_close(this: *mut c_void) -> HRESULT {
            (*(this as *mut LifecycleFake)).close_hr
        }
        unsafe extern "system" fn lf_reset(this: *mut c_void, _alloc: usize, _pso: usize) -> HRESULT {
            (*(this as *mut LifecycleFake)).reset_hr
        }
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[1] = lf_add_ref as usize;
        vt[2] = lf_release as usize;
        vt[9] = lf_close as usize;
        vt[10] = lf_reset as usize;
        let fake = Box::into_raw(Box::new(LifecycleFake { vt: vt.as_ptr(), refs: 1, close_hr: -1, reset_hr: -1 }));
        unsafe {
            let w = wrap(fake as *mut c_void);
            assert!(!super::super::title_textures::is_list_closed_for_test(w as usize));
            // Failed Close must not mark the list closed.
            assert_eq!(close(w), -1);
            assert!(!super::super::title_textures::is_list_closed_for_test(w as usize));
            // Successful Close marks the wrapper pointer closed.
            (*fake).close_hr = 0;
            assert_eq!(close(w), 0);
            assert!(super::super::title_textures::is_list_closed_for_test(w as usize));
            // Failed Reset leaves the old closed recording intact.
            assert_eq!(reset(w, 0, 0), -1);
            assert!(super::super::title_textures::is_list_closed_for_test(w as usize));
            // Successful Reset clears CLOSED so the reopened list does not keep a stale timestamp.
            (*fake).reset_hr = 0;
            assert_eq!(reset(w, 0, 0), 0);
            assert!(!super::super::title_textures::is_list_closed_for_test(w as usize));
            // Re-close, AddRef, non-final Release keeps CLOSED; final Release clears it.
            assert_eq!(close(w), 0);
            assert!(super::super::title_textures::is_list_closed_for_test(w as usize));
            assert_eq!(add_ref(w), 2);
            assert_eq!(release(w), 1);
            assert!(super::super::title_textures::is_list_closed_for_test(w as usize));
            assert_eq!(release(w), 0);
            assert!(!super::super::title_textures::is_list_closed_for_test(w as usize));
            drop(Box::from_raw(fake));
        }
    }
}
