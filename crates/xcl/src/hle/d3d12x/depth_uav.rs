//! Depth buffers that the console also lets shaders write (ALLOW_DEPTH_STENCIL |
//! ALLOW_UNORDERED_ACCESS). The PC refuses the pair: `device_x::pc_resource_desc` keeps the depth
//! buffer and drops ALLOW_UNORDERED_ACCESS, so a UAV of such a buffer is a null view and what a
//! compute program stores through it is lost.
//!
//! This module keeps what XCL knows of these buffers: each object by its address, the objects at
//! one title address and shape together in a `Cell` (on the PC they share texels, `vram`), and the
//! DSVs that name them. A UAV of the stencil plane of such a buffer is made on a twin of the
//! plane: an R8 texture with unordered access, made at the first such UAV (`resolve_uav`). The
//! store, the load (the slot's companion SRV) and the T# of the UAV slot then use the twin.
//!
//! The depth plane of a D32, D16 or D32S8 buffer has a twin too (R32 or R16), kept in step with
//! the plane around the title's transitions (`plan_barriers`): the PC buffer cannot be in
//! UNORDERED_ACCESS, so where the title has it there it rests in `PARK`; the title's transition
//! into UNORDERED_ACCESS is followed by a copy of the plane into the twin (what was drawn reaches
//! the shaders), its transition out of it is preceded by a copy of the twin into the plane (what
//! the shaders wrote reaches the next depth test or sampling). A transition names the state it
//! leaves and the one it enters, so each copy starts and ends in states the title gave.
//!
//! Behind `XCL_DEPTH_UAV_TWIN=1`; with the switch off nothing is registered, a UAV of a depth
//! buffer stays a null view and every hook stops at one atomic load.
//!
//! (After the depth+UAV twins of xcl-dev/xcl#1989 by raylitebull: registering at creation, DSVs,
//! forgetting at the last release, the UAV on a twin and its formats; and #1063 by IvyNotFound:
//! the PC state a buffer was created in.)

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::sharded::Sharded;

/// The switch's value when `XCL_DEPTH_UAV_TWIN` is not set.
const DEFAULT_ON: bool = false;

/// Whether depth buffers with unordered access are kept (`XCL_DEPTH_UAV_TWIN`).
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on(std::env::var("XCL_DEPTH_UAV_TWIN").ok().as_deref(), DEFAULT_ON))
}

/// "0" off, "1" on, anything else (or nothing) `default`.
fn on(v: Option<&str>, default: bool) -> bool {
    match v.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => default,
    }
}

/// What makes objects one buffer: on the PC, objects at one title address share texels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
    /// `vram::canonical_va` of the title address, or the object's address when it has none
    at: usize,
    /// `at` is a title address
    placed: bool,
    width: u32,
    height: u32,
    array: u16,
    mips: u16,
    /// the PC resource format: 19, 39, 44 or 53
    format: u32,
}

struct Cell {
    key: Key,
    /// its number: what a list notes of it (`plan_barriers`) is not taken for a later Cell at the
    /// same address
    id: u64,
    /// the format of the console's description
    console_format: u32,
    /// a twin was made
    twinned: AtomicBool,
    /// the last transition into UNORDERED_ACCESS recorded put the depth plane's twin in step with
    /// the plane (a copy, or the two known to hold the same): only then is the twin copied back
    /// (`plan_barriers`)
    entered: AtomicBool,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// registered objects of this buffer
    members: u32,
    /// the PC state of the first object a title's creation call made (not the object of a view
    /// by address, which is made in COMMON: `by_address`)
    made_in: Option<u32>,
    /// [depth, stencil]
    twins: [Option<Twin>; 2],
}

/// A plane's twin: a colour texture of the plane's size with unordered access, resting in
/// UNORDERED_ACCESS.
struct Twin {
    res: ID3D12Resource,
    /// its number in the log (`#n`)
    n: u64,
}

// D3D12 resources are free-threaded
unsafe impl Send for Twin {}
unsafe impl Sync for Twin {}

impl Drop for Twin {
    fn drop(&mut self) {
        TWINS.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Clone)]
struct Object {
    cell: Arc<Cell>,
    /// the PC state the object was created in
    state: u32,
}

#[derive(Clone)]
#[allow(dead_code)] // (read once DSV binds are followed)
struct Dsv {
    res: usize,
    /// D3D12_DSV_FLAGS of the view: 1 read-only depth, 2 read-only stencil
    flags: u32,
    cell: Weak<Cell>,
}

/// Every Cell by its key: taken at creation and release only.
static CELLS: Mutex<Option<HashMap<Key, Arc<Cell>>>> = Mutex::new(None);
/// Registered objects by their address.
static OBJECTS: Sharded<Object> = Sharded::new();
/// DSV handles that name a registered object.
static DSVS: Sharded<Dsv> = Sharded::new();
/// The gates: entries in OBJECTS and DSVS.
static REGISTERED: AtomicUsize = AtomicUsize::new(0);
static DSV_ENTRIES: AtomicUsize = AtomicUsize::new(0);
/// Twins alive.
static TWINS: AtomicUsize = AtomicUsize::new(0);
/// Twins made, UAVs made on a twin, UAVs of depth buffers left null views.
static MADE: AtomicU64 = AtomicU64::new(0);
static REDIRECTED: AtomicU64 = AtomicU64::new(0);
static NULL_VIEWS: AtomicU64 = AtomicU64::new(0);
/// ClearUnorderedAccessView* done on a twin, and not done because what the view names is not known.
static UAV_CLEARS: AtomicU64 = AtomicU64::new(0);
static CLEARS_NOT_DONE: AtomicU64 = AtomicU64::new(0);
/// Cell numbers given.
static CELL_IDS: AtomicU64 = AtomicU64::new(1);
/// Copies of a depth plane into its twin, of a twin into its plane, and copies into the twin not
/// done since the two were in step (`plan_barriers`).
static TO_TWIN: AtomicU64 = AtomicU64::new(0);
static TO_PLANE: AtomicU64 = AtomicU64::new(0);
static IN_STEP: AtomicU64 = AtomicU64::new(0);

/// Whether a twin is alive: the gate of the paths that record.
pub(super) fn any() -> bool {
    TWINS.load(Ordering::Relaxed) != 0
}

/// Whether a buffer is kept: the gate of ResourceBarrier (`plan_barriers`).
pub(super) fn any_kept() -> bool {
    REGISTERED.load(Ordering::Relaxed) != 0
}

fn u32_at(x: &[u8; 0x38], o: usize) -> u32 {
    u32::from_le_bytes(x[o..o + 4].try_into().unwrap())
}

/// The PC object `res` was made for the console description `console_desc` in PC state
/// `pc_state` (the state of the call that succeeded), at the title address `va` (None: memory of
/// its own). Kept when it is a single-sample 2D depth buffer with unordered access.
pub(super) fn note_created(res: *mut c_void, console_desc: &[u8; 0x38], va: Option<usize>, pc_state: u32) {
    if !enabled() || res.is_null() {
        return;
    }
    register(res as usize, console_desc, va.map(super::vram::canonical_va), pc_state);
}

/// `note_created` without the switch, `at` already canonical. True: kept.
fn register(res: usize, x: &[u8; 0x38], at: Option<usize>, pc_state: u32) -> bool {
    let format = u32_at(&super::device_x::pc_resource_desc(x), 0x20);
    if u32_at(x, 0) != D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32 || u32_at(x, 0x30) & 6 != 6 || u32_at(x, 0x24) > 1 || !matches!(format, 19 | 39 | 44 | 53) {
        return false;
    }
    let key = Key {
        at: at.unwrap_or(res),
        placed: at.is_some(),
        width: u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()) as u32,
        height: u32_at(x, 0x18),
        array: u16::from_le_bytes(x[0x1C..0x1E].try_into().unwrap()),
        mips: u16::from_le_bytes(x[0x1E..0x20].try_into().unwrap()),
        format,
    };
    let mut g = CELLS.lock().unwrap();
    let cells = g.get_or_insert_with(HashMap::new);
    let cell = cells
        .entry(key)
        .or_insert_with(|| Arc::new(Cell { key, id: CELL_IDS.fetch_add(1, Ordering::Relaxed), console_format: u32_at(x, 0x20), twinned: AtomicBool::new(false), entered: AtomicBool::new(false), inner: Mutex::new(Inner::default()) }))
        .clone();
    {
        let mut inner = cell.inner.lock().unwrap();
        inner.members += 1;
        if inner.made_in.is_none() && !BY_ADDRESS.with(std::cell::Cell::get) {
            inner.made_in = Some(pc_state);
        }
    }
    // `XCL_TRACE_FRAME`: the big ones, with where they are – two buffers at one title address
    // are one memory on the console
    if key.width >= 1920 && super::present::trace_frame() {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 96 {
            crate::xlog_info!(
                "frame trace: depth buffer with unordered access {res:#x}: {}x{}x{} mips {}, PC format {}, console format {}, {} {:#x}, made in PC state {pc_state:#x}{}",
                key.width,
                key.height,
                key.array,
                key.mips,
                key.format,
                u32_at(x, 0x20),
                if key.placed { "at title address" } else { "memory of its own, object" },
                key.at,
                if BY_ADDRESS.with(std::cell::Cell::get) { " (for a view by address)" } else { "" }
            );
        }
    }
    let gone = match OBJECTS.insert(res, Object { cell, state: pc_state }) {
        None => {
            REGISTERED.fetch_add(1, Ordering::Relaxed);
            Default::default()
        }
        // (an object XCL did not see released: its membership goes back)
        Some(old) => leave(cells, &old.cell),
    };
    drop(g);
    drop(gone);
    true
}

thread_local! {
    /// this thread is making the object of a view by address (`by_address`)
    static BY_ADDRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `f` makes the object of a view by address (`real::resource_at`): XCL makes it in COMMON, so
/// its state is not the one the title made the buffer in.
pub(super) fn by_address<R>(f: impl FnOnce() -> R) -> R {
    BY_ADDRESS.with(|b| b.set(true));
    let r = f();
    BY_ADDRESS.with(|b| b.set(false));
    r
}

/// One object of `cell` is gone; the last one takes the Cell with it, and its twins, which the
/// caller drops outside every lock (their Release goes through `real::res_release`).
#[must_use]
fn leave(cells: &mut HashMap<Key, Arc<Cell>>, cell: &Arc<Cell>) -> [Option<Twin>; 2] {
    let mut inner = cell.inner.lock().unwrap();
    inner.members = inner.members.saturating_sub(1);
    if inner.members == 0 && cells.get(&cell.key).is_some_and(|c| Arc::ptr_eq(c, cell)) {
        cells.remove(&cell.key);
        return std::mem::take(&mut inner.twins);
    }
    Default::default()
}

/// The last reference of `res` goes (`real::res_release`).
pub(super) fn forget(res: usize) {
    if REGISTERED.load(Ordering::Relaxed) == 0 {
        return;
    }
    let Some(o) = OBJECTS.remove(res) else { return };
    REGISTERED.fetch_sub(1, Ordering::Relaxed);
    let gone = CELLS.lock().unwrap().as_mut().map(|cells| leave(cells, &o.cell));
    drop(gone);
}

/// What a UAV of a resource is made on (`resolve_uav`).
pub(super) enum Resolve {
    /// the twin, and the view of it
    Twin(*mut c_void, D3D12_UNORDERED_ACCESS_VIEW_DESC),
    /// a depth buffer whose UAV stays a null view, and why
    Null(&'static str),
    /// not a depth buffer with unordered access
    NotOurs,
}

/// CreateUnorderedAccessView of `res` (description `rd`) with the title's view `view` as it came:
/// a UAV of the stencil plane of a kept two-plane depth buffer, or of the depth plane of a kept
/// D32, D16 or D32S8 one, is made on the plane's twin (made now when there is none yet).
pub(super) unsafe fn resolve_uav(dev: *mut c_void, res: *mut c_void, rd: Option<&D3D12_RESOURCE_DESC>, view: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>) -> Resolve {
    let Some(rd) = rd.filter(|d| d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 && d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0) else {
        return Resolve::NotOurs;
    };
    if !enabled() {
        static SAID: AtomicBool = AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            crate::xlog_info!("D3D12.X: depth UAV twins are off (XCL_DEPTH_UAV_TWIN): a UAV of a depth buffer is a null view");
        }
        return Resolve::Null("XCL_DEPTH_UAV_TWIN is off");
    }
    if rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || rd.SampleDesc.Count > 1 {
        return Resolve::Null("not a single-sample 2D depth buffer");
    }
    if view.is_some_and(|v| !matches!(v.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2D | D3D12_UAV_DIMENSION_TEXTURE2DARRAY)) {
        return Resolve::Null("not a 2D view");
    }
    // (no twin on a guessed state)
    let Some(o) = OBJECTS.get(res as usize) else { return Resolve::Null("XCL did not see the buffer created") };
    let key = o.cell.key;
    // `XCL_TRACE_FRAME`: which of the big buffers shaders get a UAV of, and of which plane
    if key.width >= 1920 && super::present::trace_frame() {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 96 {
            crate::xlog_info!(
                "frame trace: UAV of depth buffer {:#x} ({}x{} PC format {}, console format {}, at {:#x}): view format {}, its {} plane",
                res as usize,
                key.width,
                key.height,
                key.format,
                o.cell.console_format,
                key.at,
                view.map_or(-1, |v| v.Format.0),
                if stencil_view(view) { "stencil" } else { "depth" }
            );
        }
    }
    let stencil = stencil_view(view);
    // (a D24S8's depth plane: no UAV format holds it as a shader reads it)
    if if stencil { !matches!(key.format, 19 | 44) } else { !matches!(key.format, 19 | 39 | 53) } {
        return Resolve::Null("its depth plane has no twin yet");
    }
    // (its states stay the title's: the twin would never be copied)
    if !stencil && (o.cell.inner.lock().unwrap().made_in.unwrap_or(o.state) | o.state) & UNORDERED_ACCESS != 0 {
        return Resolve::Null("the buffer was made in UNORDERED_ACCESS");
    }
    // (the Cell holds the twin)
    let Some((twin, _)) = plane_twin(&o, stencil as usize, dev) else { return Resolve::Null("the twin could not be made") };
    REDIRECTED.fetch_add(1, Ordering::Relaxed);
    Resolve::Twin(twin.as_raw(), if stencil { twin_view(view, key.array) } else { depth_twin_view(view, key.array, key.format == 53) })
}

/// The twin of `plane` (0 depth, 1 stencil) of the buffer `o` is in, and its number: the one it
/// has, or one made now on the device `dev` (None: it could not be made).
unsafe fn plane_twin(o: &Object, plane: usize, dev: *mut c_void) -> Option<(ID3D12Resource, u64)> {
    let key = o.cell.key;
    let mut inner = o.cell.inner.lock().unwrap();
    if let Some(t) = &inner.twins[plane] {
        return Some((t.res.clone(), t.n));
    }
    let t = make_twin(dev, &key, twin_format(key.format, plane))?;
    TWINS.fetch_add(1, Ordering::Relaxed);
    let n = MADE.fetch_add(1, Ordering::Relaxed) + 1;
    inner.twins[plane] = Some(Twin { res: t.clone(), n });
    o.cell.twinned.store(true, Ordering::Relaxed);
    // (the state of the title's object; that of the UAV's own when the title made none)
    let state = inner.made_in.unwrap_or(o.state);
    if n <= 8 || n.is_power_of_two() {
        if plane == 1 {
            crate::xlog_info!(
                "D3D12.X: depth UAV twin #{n}: the stencil plane of a {}x{}x{} depth buffer (PC format {}, console format {}) at {:#x} is an R8 texture; the buffer was made in state {:#x}",
                key.width,
                key.height,
                key.array,
                key.format,
                o.cell.console_format,
                key.at,
                state
            );
        } else {
            crate::xlog_info!(
                "D3D12.X: depth UAV twin #{n}: the depth plane of a {}x{}x{} depth buffer (PC format {}, console format {}) at {:#x} is an R{} texture, copied from the plane when the title moves the buffer into UNORDERED_ACCESS and back when it moves it out; the buffer was made in state {:#x}",
                key.width,
                key.height,
                key.array,
                key.format,
                o.cell.console_format,
                key.at,
                if key.format == 53 { 16 } else { 32 },
                state
            );
        }
    }
    Some((t, n))
}

/// The format of the twin of `plane` of a buffer of the PC format `format`: R8 for the stencil
/// plane, R16 for D16's depth, R32 for D32's and D32S8's (the plane copies as R32_TYPELESS).
fn twin_format(format: u32, plane: usize) -> DXGI_FORMAT {
    match (plane, format) {
        (1, _) => DXGI_FORMAT_R8_TYPELESS,
        (_, 53) => DXGI_FORMAT_R16_TYPELESS,
        _ => DXGI_FORMAT_R32_TYPELESS,
    }
}

/// Where a buffer whose depth plane can have a twin rests while the title has it in
/// UNORDERED_ACCESS: COPY_SOURCE, the state the plane is copied into the twin from.
pub(super) const PARK: u32 = 0x800;
/// D3D12_RESOURCE_STATE_UNORDERED_ACCESS, COPY_DEST
const UNORDERED_ACCESS: u32 = 0x8;
const COPY_DEST: u32 = 0x400;
/// The states in which nothing writes a depth plane: DEPTH_READ, NON_PIXEL_SHADER_RESOURCE,
/// PIXEL_SHADER_RESOURCE, INDIRECT_ARGUMENT, COPY_SOURCE, RESOLVE_SOURCE (and their sums).
const READ_ONLY: u32 = 0x20 | 0x40 | 0x80 | 0x200 | 0x800 | 0x2000;
/// D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY, END_ONLY
const BEGIN_ONLY: u32 = 1;
const END_ONLY: u32 = 2;

/// A copy between depth plane subresources of a title's object and the same ones of the twin.
pub(super) struct PlaneCopy {
    /// the object the title's transition names
    pub depth: usize,
    /// the subresource the transition names (or D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
    pub sub: u32,
    /// the depth plane's subresources it copies (the twin's have the same numbers)
    pub subs: Vec<u32>,
    pub twin: ID3D12Resource,
}

/// What goes on the PC list with a title's barriers (`plan_barriers`).
#[derive(Default)]
pub(super) struct Plan {
    /// before them: twins into depth planes (the buffer goes PARK → COPY_DEST, the title's
    /// transition takes it on from COPY_DEST; the twin UNORDERED_ACCESS → COPY_SOURCE and back)
    pub to_plane: Vec<PlaneCopy>,
    /// after them: depth planes into twins (the buffer rests in PARK; the twin
    /// UNORDERED_ACCESS → COPY_DEST and back)
    pub to_twin: Vec<PlaneCopy>,
}

/// Whether the depth plane of `o`'s buffer can have a twin: a D32, D16 or D32S8 buffer, and the
/// object was not made in UNORDERED_ACCESS (that object's states stay the title's).
fn plane_twinnable(o: &Object) -> bool {
    matches!(o.cell.key.format, 19 | 39 | 53) && o.state & UNORDERED_ACCESS == 0
}

/// The title's barriers `barriers` (PC D3D12_RESOURCE_BARRIERs as words, the console's bits
/// dropped) on a list (`direct`: a DIRECT one) as they go to the PC, and the copies around
/// them. A transition of an object whose depth plane can have a twin into or out of
/// UNORDERED_ACCESS goes into or out of `PARK` instead (also on lists that copy nothing, so the
/// buffer's states agree wherever the title moves it; a split one at its end, whole); on a
/// DIRECT list the twin is copied from the plane after a transition into it (the twin made
/// then, when there is none yet) and into the plane before a transition out of it.
///
/// `in_step` holds the Cells whose planes and twins are known to hold the same since a copy
/// earlier on this recording, with no transition to a state that writes the plane after it:
/// their copy into the twin is not done. It is the list's, and goes with its recording – what
/// another list did in between is not seen from here.
pub(super) fn plan_barriers(barriers: &mut Vec<[u32; 8]>, direct: bool, in_step: &mut Vec<u64>) -> Plan {
    plan_with(barriers, direct, in_step, &mut |res, o| unsafe {
        let r = res as *mut c_void;
        let mut dev: Option<ID3D12Device> = None;
        ID3D12Resource::from_raw_borrowed(&r)?.GetDevice(&mut dev).ok()?;
        plane_twin(o, 0, dev?.as_raw())
    })
}

/// `plan_barriers` with `twin` giving the depth twin of an object (made when there is none).
fn plan_with(barriers: &mut Vec<[u32; 8]>, direct: bool, in_step: &mut Vec<u64>, twin: &mut dyn FnMut(usize, &Object) -> Option<(ID3D12Resource, u64)>) -> Plan {
    let mut plan = Plan::default();
    if !any_kept() {
        return plan;
    }
    let mut out = Vec::with_capacity(barriers.len());
    for mut b in barriers.drain(..) {
        let res = (b[2] as u64 | (b[3] as u64) << 32) as usize;
        let Some(o) = (b[0] == 0).then(|| OBJECTS.get(res)).flatten().filter(plane_twinnable) else {
            out.push(b);
            continue;
        };
        let id = o.cell.id;
        let (flags, sub, mut before, mut after) = (b[1], b[4], b[5], b[6]);
        let (leaving, entering) = (before == UNORDERED_ACCESS, after == UNORDERED_ACCESS);
        let writes = |s: u32| s == 0 || s & !READ_ONLY != 0;
        if !leaving && !entering {
            if writes(after) {
                in_step.retain(|&c| c != id);
            }
            out.push(b);
            continue;
        }
        // a split transition is done whole at its end: the copies go where the buffer is in the
        // state the title left
        if flags & BEGIN_ONLY != 0 {
            continue;
        }
        b[1] = flags & !END_ONLY;
        // the depth plane's subresources the transition names (the stencil plane's come after them)
        let n = o.cell.key.mips.max(1) as u32 * o.cell.key.array.max(1) as u32;
        let subs: Vec<u32> = if sub == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES { (0..n).collect() } else { (sub < n).then_some(sub).into_iter().collect() };
        let entered = &o.cell.entered;
        if !direct && o.cell.inner.lock().unwrap().twins[0].is_some() {
            static SAID: AtomicBool = AtomicBool::new(false);
            if !SAID.swap(true, Ordering::Relaxed) {
                crate::xlog_warn!("D3D12.X: a buffer with a depth UAV twin goes into or out of UNORDERED_ACCESS on a compute or copy list: the twin is not copied there");
            }
        }
        if leaving {
            before = PARK;
            // (a twin whose last way in was not copied holds something else than the plane: not
            // copied back – what the shaders wrote into it is lost, as without a twin)
            let t = (direct && !subs.is_empty()).then(|| twin_if_made(&o)).flatten();
            if let Some((t, k)) = t.filter(|_| entered.load(Ordering::Relaxed)) {
                before = COPY_DEST;
                let c = TO_PLANE.fetch_add(1, Ordering::Relaxed) + 1;
                if c <= 4 || c.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: depth UAV twin #{k}: the title moves depth buffer {res:#x} out of UNORDERED_ACCESS into state {after:#x}: the twin is copied into its depth plane first ({c} such copies so far)");
                }
                if subs.len() as u32 == n && !in_step.contains(&id) {
                    in_step.push(id);
                }
                plan.to_plane.push(PlaneCopy { depth: res, sub, subs: subs.clone(), twin: t });
            }
            if writes(after) {
                in_step.retain(|&c| c != id);
            }
        }
        if entering {
            after = PARK;
            if !direct {
                entered.store(false, Ordering::Relaxed);
            } else if !subs.is_empty() {
                // (a plane written since the copy that put the two in step: again)
                if in_step.contains(&id) && !writes(before) {
                    IN_STEP.fetch_add(1, Ordering::Relaxed);
                    entered.store(true, Ordering::Relaxed);
                } else if let Some((t, k)) = twin(res, &o) {
                    let c = TO_TWIN.fetch_add(1, Ordering::Relaxed) + 1;
                    if c <= 4 || c.is_power_of_two() {
                        crate::xlog_info!(
                            "D3D12.X: depth UAV twin #{k}: the title moves depth buffer {res:#x} from state {before:#x} into UNORDERED_ACCESS: its depth plane is copied into the twin ({c} such copies so far, {} not needed)",
                            IN_STEP.load(Ordering::Relaxed)
                        );
                    }
                    if subs.len() as u32 == n && !in_step.contains(&id) {
                        in_step.push(id);
                    }
                    entered.store(true, Ordering::Relaxed);
                    plan.to_twin.push(PlaneCopy { depth: res, sub, subs, twin: t });
                } else {
                    entered.store(false, Ordering::Relaxed);
                }
            }
        }
        (b[5], b[6]) = (before, after);
        if before != after {
            out.push(b);
        }
    }
    *barriers = out;
    plan
}

/// The depth twin of `o`'s buffer when it has one, and its number.
fn twin_if_made(o: &Object) -> Option<(ID3D12Resource, u64)> {
    o.cell.inner.lock().unwrap().twins[0].as_ref().map(|t| (t.res.clone(), t.n))
}

/// A UAV of a depth buffer stays a null view (`Resolve::Null`).
pub(super) fn null_view(why: &'static str) {
    #[cfg(test)]
    NULL_REASONS.lock().unwrap().push(why);
    let n = NULL_VIEWS.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a UAV of a depth buffer stays a null view ({why}): {n} so far");
    }
}

#[cfg(test)]
static NULL_REASONS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

/// What a ClearUnorderedAccessView* goes to the PC with (`clear_target`).
#[derive(Debug, PartialEq)]
pub(super) enum Clear {
    /// the resource the title gave
    AsIs,
    /// the twin the slot's view names
    Twin(usize),
    /// nothing: what the slot's view names is not known, or it is a null view
    NotDone,
}

/// ClearUnorderedAccessView* of `res` through a GPU slot whose UAV in the shared heap names
/// `named` (`console_heap::names_of`; None: the slot is not in the shared heap). The PC clear of
/// a view of a twin must name the twin: the depth buffer with the twin's view is an invalid pair.
/// A buffer with a twin whose slot names 0 or `GONE` is not cleared: 0 is also what a copy from a
/// slot outside the shared heap notes, and that copy may hold the twin's view.
pub(super) fn clear_target(res: usize, named: Option<usize>) -> Clear {
    if !any() {
        return Clear::AsIs;
    }
    let Some(o) = OBJECTS.get(res) else { return Clear::AsIs };
    let twins = o.cell.inner.lock().unwrap().twins.each_ref().map(|t| t.as_ref().map(|t| t.res.as_raw() as usize));
    let c = clear_choice(twins, named);
    match c {
        Clear::Twin(_) => UAV_CLEARS.fetch_add(1, Ordering::Relaxed),
        Clear::NotDone => CLEARS_NOT_DONE.fetch_add(1, Ordering::Relaxed),
        Clear::AsIs => 0,
    };
    c
}

/// `clear_target` for a buffer with the twins `twins`: the twin when the view names one; not
/// done when what it names is not known (a null view clears nothing either); else as it came
/// (a view of another resource).
fn clear_choice(twins: [Option<usize>; 2], named: Option<usize>) -> Clear {
    if twins.iter().all(Option::is_none) {
        return Clear::AsIs;
    }
    match named {
        None | Some(0) | Some(super::console_heap::GONE) => Clear::NotDone,
        Some(n) if twins.contains(&Some(n)) => Clear::Twin(n),
        Some(_) => Clear::AsIs,
    }
}

/// `XCL_DUMP_TARGETS`: every twin alive, with its tag in the dump ("twin<n> <plane>"). Each
/// rests in UNORDERED_ACCESS.
pub(super) fn twins_to_dump() -> Vec<(String, ID3D12Resource)> {
    if !any() {
        return Vec::new();
    }
    let cells: Vec<Arc<Cell>> = CELLS.lock().unwrap().as_ref().map_or(Vec::new(), |m| m.values().filter(|c| c.twinned.load(Ordering::Relaxed)).cloned().collect());
    let mut out = Vec::new();
    for c in cells {
        let inner = c.inner.lock().unwrap();
        for (t, plane) in inner.twins.iter().zip(["depth", "stencil"]) {
            if let Some(t) = t {
                out.push((format!("twin{} {plane}", t.n), t.res.clone()));
            }
        }
    }
    out
}

/// `XCL_DUMP_TARGETS`: what the twin dumped as `tag` (`twins_to_dump`) into the readback buffer
/// `rb` at `fp` holds: its most frequent values, how many differ and the share of zeros.
pub(super) unsafe fn log_values(rb: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, tag: &str) {
    let mut p: *mut c_void = std::ptr::null_mut();
    if rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
        return;
    }
    let f = &fp.Footprint;
    let len = (rb.GetDesc().Width as usize).saturating_sub(fp.Offset as usize);
    let data = std::slice::from_raw_parts((p as *const u8).add(fp.Offset as usize), len);
    let counts = byte_counts(data, f.Width as usize, f.Height as usize, f.RowPitch as usize);
    rb.Unmap(0, None);
    let (top, distinct, zero) = most_frequent(&counts);
    let (n, plane) = tag.strip_prefix("twin").and_then(|t| t.split_once(' ')).unwrap_or((tag, ""));
    let values = top.iter().map(|(v, c)| format!("{v} ×{c}")).collect::<Vec<_>>().join(", ");
    crate::xlog_info!("D3D12.X: target dump: depth UAV twin #{n} ({}x{}, {plane}): {values} ({distinct} distinct values, {zero:.1} % zero)", f.Width, f.Height);
}

/// Texels by byte value of a `w`x`h` 8-bit image whose rows start `pitch` bytes apart (the
/// padding after each row is not counted; a short buffer ends the count).
fn byte_counts(data: &[u8], w: usize, h: usize, pitch: usize) -> [u64; 256] {
    let mut c = [0u64; 256];
    for y in 0..h {
        let Some(row) = data.get(y * pitch..y * pitch + w) else { break };
        for &v in row {
            c[v as usize] += 1;
        }
    }
    c
}

/// The eight most frequent values with their counts (most first; equal counts: the smaller
/// value first), the number of distinct values and the share of zeros in %.
fn most_frequent(c: &[u64; 256]) -> (Vec<(u8, u64)>, usize, f64) {
    let mut v: Vec<(u8, u64)> = (0..256).filter(|&i| c[i] != 0).map(|i| (i as u8, c[i])).collect();
    let distinct = v.len();
    let total: u64 = c.iter().sum();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(8);
    (v, distinct, if total == 0 { 0.0 } else { c[0] as f64 * 100.0 / total as f64 })
}

/// A plane's twin of `key`: `format` (`twin_format`) with unordered access, of the buffer's
/// size, array and mips, in UNORDERED_ACCESS. Made through `real::dev_create_committed_resource`:
/// a PC resource, adopted, never in title memory.
unsafe fn make_twin(dev: *mut c_void, key: &Key, format: DXGI_FORMAT) -> Option<ID3D12Resource> {
    if dev.is_null() {
        return None;
    }
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: key.width as u64,
        Height: key.height,
        DepthOrArraySize: key.array.max(1),
        MipLevels: key.mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut out = std::ptr::null_mut();
    let hr = super::real::dev_create_committed_resource(
        dev,
        &heap as *const _ as usize,
        0,
        &desc as *const _ as usize,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as usize,
        0,
        &ID3D12Resource::IID as *const _ as *const windows_sys::core::GUID,
        &mut out,
    );
    (hr >= 0 && !out.is_null()).then(|| ID3D12Resource::from_raw(out))
}

/// The view names the stencil plane: plane slice 1, or a stencil or 8-bit format.
fn stencil_view(view: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>) -> bool {
    let Some(v) = view else { return false };
    let plane = unsafe {
        match v.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => v.Anonymous.Texture2D.PlaneSlice,
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => v.Anonymous.Texture2DArray.PlaneSlice,
            _ => 0,
        }
    };
    plane == 1 || matches!(v.Format.0, 22 | 47 | 60..=64 | 120)
}

/// The twin's view: the title's (a Texture2D or Texture2DArray one when it gave none) on plane 0,
/// R8_UINT or the 8-bit class the title asked for (UNORM, SNORM, SINT).
fn twin_view(view: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>, array: u16) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut v = plane_view(view, array);
    v.Format = match v.Format {
        f @ (DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_R8_SNORM | DXGI_FORMAT_R8_SINT) => f,
        _ => DXGI_FORMAT_R8_UINT,
    };
    v
}

/// The depth twin's view (`r16`: of a D16's R16 twin, else of an R32 one): the title's on plane
/// 0, as `twin_view`'s; R32_FLOAT or R16_UNORM – the depth as a shader reads it – unless the
/// title asked for another class of the twin's size (R32_UINT, R16_UINT …).
fn depth_twin_view(view: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>, array: u16, r16: bool) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut v = plane_view(view, array);
    v.Format = match (r16, v.Format) {
        (false, f @ (DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32_SINT)) => f,
        (false, _) => DXGI_FORMAT_R32_FLOAT,
        (true, f @ (DXGI_FORMAT_R16_UNORM | DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_SNORM | DXGI_FORMAT_R16_SINT | DXGI_FORMAT_R16_FLOAT)) => f,
        (true, _) => DXGI_FORMAT_R16_UNORM,
    };
    v
}

/// The title's view (a Texture2D or Texture2DArray one of the whole buffer when it gave none)
/// on plane 0, its format as it came.
fn plane_view(view: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>, array: u16) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut v = match view {
        Some(v) => *v,
        None if array > 1 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: array as u32, PlaneSlice: 0 } },
            ..Default::default()
        },
        None => D3D12_UNORDERED_ACCESS_VIEW_DESC { ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() },
    };
    match v.ViewDimension {
        D3D12_UAV_DIMENSION_TEXTURE2D => v.Anonymous.Texture2D.PlaneSlice = 0,
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => v.Anonymous.Texture2DArray.PlaneSlice = 0,
        _ => {}
    }
    v
}

/// (tests) Whether `res` is kept.
#[cfg(test)]
pub(super) fn kept(res: usize) -> bool {
    OBJECTS.contains(res)
}

/// (tests) The state line L2 gives for the buffer `res` is in, and whether that buffer has a
/// title address.
#[cfg(test)]
pub(super) fn made_in(res: usize) -> Option<(Option<u32>, Option<usize>)> {
    let o = OBJECTS.get(res)?;
    let made_in = o.cell.inner.lock().unwrap().made_in;
    Some((made_in, o.cell.key.placed.then_some(o.cell.key.at)))
}

/// (tests) The stencil twin of the buffer `res` is in.
#[cfg(test)]
pub(super) fn twin_of(res: usize) -> Option<usize> {
    OBJECTS.get(res)?.cell.inner.lock().unwrap().twins[1].as_ref().map(|t| t.res.as_raw() as usize)
}

/// (tests) The depth twin of the buffer `res` is in.
#[cfg(test)]
pub(super) fn depth_twin_of(res: usize) -> Option<usize> {
    OBJECTS.get(res)?.cell.inner.lock().unwrap().twins[0].as_ref().map(|t| t.res.as_raw() as usize)
}

/// (tests) Copies of depth planes into their twins, of twins into their planes, and copies into
/// twins not done since the two were in step.
#[cfg(test)]
pub(super) fn plane_copies() -> (u64, u64, u64) {
    (TO_TWIN.load(Ordering::Relaxed), TO_PLANE.load(Ordering::Relaxed), IN_STEP.load(Ordering::Relaxed))
}

/// (tests) The object a DSV handle names.
#[cfg(test)]
pub(super) fn dsv_names(handle: usize) -> Option<usize> {
    DSVS.get(handle).map(|d| d.res)
}

/// (tests) Twins made, twins alive, UAVs made on a twin, and the reasons of the null views so far.
#[cfg(test)]
pub(super) fn counts() -> (u64, usize, u64, Vec<&'static str>) {
    (MADE.load(Ordering::Relaxed), TWINS.load(Ordering::Relaxed), REDIRECTED.load(Ordering::Relaxed), NULL_REASONS.lock().unwrap().clone())
}

/// (tests) UAV clears done on a twin, and not done.
#[cfg(test)]
pub(super) fn clears() -> (u64, u64) {
    (UAV_CLEARS.load(Ordering::Relaxed), CLEARS_NOT_DONE.load(Ordering::Relaxed))
}

/// CreateDepthStencilView of `res` at `handle` with D3D12_DSV_FLAGS `flags`.
pub(super) fn note_dsv(handle: usize, res: usize, flags: u32) {
    if REGISTERED.load(Ordering::Relaxed) == 0 && DSV_ENTRIES.load(Ordering::Relaxed) == 0 {
        return;
    }
    match OBJECTS.get(res) {
        Some(o) => {
            if DSVS.insert(handle, Dsv { res, flags, cell: Arc::downgrade(&o.cell) }).is_none() {
                DSV_ENTRIES.fetch_add(1, Ordering::Relaxed);
            }
        }
        None => forget_dsv(handle),
    }
}

/// The D3D12_DSV_FLAGS of the DSV at `handle` when it names a kept buffer (`XCL_TRACE_FRAME`).
pub(super) fn dsv_flags(handle: usize) -> Option<u32> {
    if DSV_ENTRIES.load(Ordering::Relaxed) == 0 {
        return None;
    }
    DSVS.get(handle).map(|d| d.flags)
}

fn forget_dsv(handle: usize) {
    if DSV_ENTRIES.load(Ordering::Relaxed) != 0 && DSVS.remove(handle).is_some() {
        DSV_ENTRIES.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The DSV handle increment of `dev`, read once.
unsafe fn dsv_increment(dev: *mut c_void) -> u64 {
    static INC: AtomicU32 = AtomicU32::new(0);
    let mut inc = INC.load(Ordering::Relaxed);
    if inc == 0 {
        let Some(d) = ID3D12Device::from_raw_borrowed(&dev) else { return 0 };
        inc = d.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        INC.store(inc, Ordering::Relaxed);
    }
    inc as u64
}

/// CopyDescriptorsSimple of `n` DSVs from `src` to `dst`: each destination names what its source
/// names, or nothing.
pub(super) unsafe fn dsvs_copied(dev: *mut c_void, n: u32, dst: u64, src: u64) {
    if DSV_ENTRIES.load(Ordering::Relaxed) == 0 {
        return;
    }
    copy_dsvs(n, dst, src, dsv_increment(dev));
}

/// CopyDescriptors of DSVs: destination and source ranges (start, length; no lengths: 1 each)
/// walked in step.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn dsv_ranges_copied(dev: *mut c_void, nd: u32, dsts: *const u64, dsizes: *const u32, ns: u32, srcs: *const u64, ssizes: *const u32) {
    if DSV_ENTRIES.load(Ordering::Relaxed) == 0 || dsts.is_null() || srcs.is_null() {
        return;
    }
    let ranges = |starts: *const u64, sizes: *const u32, n: u32| -> Vec<(u64, u32)> {
        (0..n as usize).map(|r| (*starts.add(r), if sizes.is_null() { 1 } else { *sizes.add(r) })).collect()
    };
    let inc = dsv_increment(dev);
    for (dst, src, n) in pieces(&ranges(dsts, dsizes, nd), &ranges(srcs, ssizes, ns), inc) {
        copy_dsvs(n, dst, src, inc);
    }
}

/// Destination and source ranges cut where either one ends: (dst, src, n) runs.
pub(super) fn pieces(dsts: &[(u64, u32)], srcs: &[(u64, u32)], inc: u64) -> Vec<(u64, u64, u32)> {
    let (mut out, mut d, mut s, mut dk, mut sk) = (Vec::new(), 0, 0, 0u32, 0u32);
    while d < dsts.len() && s < srcs.len() {
        let n = (dsts[d].1 - dk).min(srcs[s].1 - sk);
        if n > 0 {
            out.push((dsts[d].0 + dk as u64 * inc, srcs[s].0 + sk as u64 * inc, n));
        }
        dk += n;
        sk += n;
        if dk == dsts[d].1 {
            (d, dk) = (d + 1, 0);
        }
        if sk == srcs[s].1 {
            (s, sk) = (s + 1, 0);
        }
    }
    out
}

fn copy_dsvs(n: u32, dst: u64, src: u64, inc: u64) {
    for k in 0..n as u64 {
        let (to, from) = ((dst + k * inc) as usize, (src + k * inc) as usize);
        match DSVS.get(from) {
            Some(v) => {
                if DSVS.insert(to, v).is_none() {
                    DSV_ENTRIES.fetch_add(1, Ordering::Relaxed);
                }
            }
            None => forget_dsv(to),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A console resource description: 2D, `w`x`h`, one slice and mip, `samples`, `flags`.
    fn desc(dim: u32, w: u64, h: u32, format: u32, samples: u32, flags: u32) -> [u8; 0x38] {
        let mut x = [0u8; 0x38];
        x[0..4].copy_from_slice(&dim.to_le_bytes());
        x[0x10..0x18].copy_from_slice(&w.to_le_bytes());
        x[0x18..0x1C].copy_from_slice(&h.to_le_bytes());
        x[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
        x[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
        x[0x20..0x24].copy_from_slice(&format.to_le_bytes());
        x[0x24..0x28].copy_from_slice(&samples.to_le_bytes());
        x[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
        x
    }

    fn cell_of(res: usize) -> Option<Arc<Cell>> {
        OBJECTS.get(res).map(|o| o.cell)
    }

    #[test]
    fn the_switch_reads_0_and_1() {
        assert!(!on(Some("0"), true));
        assert!(on(Some("1"), false));
        assert!(on(Some(" 1\n"), false), "trimmed");
        assert!(on(Some("yes"), true) && !on(Some("yes"), false), "anything else: the default");
        assert!(on(None, true) && !on(None, false));
    }

    #[test]
    fn only_depth_buffers_that_allow_unordered_access_are_kept() {
        let base = 0x7A00_0000_0000usize;
        // console D24S8 (45), D16S8 (118), D32 (40), D16 (55), D32S8 (20) with flags 6
        for (k, f) in [45u32, 118, 40, 55, 20].into_iter().enumerate() {
            let res = base + k * 0x100;
            assert!(register(res, &desc(3, 64, 32, f, 1, 6), Some(res), 0x10), "format {f}");
            assert_eq!(cell_of(res).unwrap().key.format, [44, 44, 39, 53, 19][k]);
            forget(res);
            assert!(cell_of(res).is_none());
        }
        let refused = [
            ("depth only", desc(3, 64, 32, 45, 1, 2)),
            ("unordered access only", desc(3, 64, 32, 41, 1, 4)),
            ("4 samples", desc(3, 64, 32, 45, 4, 6)),
            ("a 3D texture", desc(4, 64, 32, 45, 1, 6)),
            ("a colour format", desc(3, 64, 32, 28, 1, 6)),
        ];
        for (k, (what, x)) in refused.iter().enumerate() {
            let res = base + 0x1_0000 + k * 0x100;
            assert!(!register(res, x, None, 0x10), "{what}");
            assert!(cell_of(res).is_none(), "{what}");
        }
    }

    #[test]
    fn depth_buffers_at_one_title_address_share_a_cell_until_the_last_is_released() {
        let (a, b, c, at) = (0x7B00_0000_1000usize, 0x7B00_0000_2000usize, 0x7B00_0000_3000usize, 0x2_0000_0000usize);
        let x = desc(3, 2560, 1440, 45, 1, 6);
        assert!(register(a, &x, Some(at), 0x10));
        assert!(register(b, &x, Some(at), 0));
        let (ca, cb) = (cell_of(a).unwrap(), cell_of(b).unwrap());
        assert!(Arc::ptr_eq(&ca, &cb), "one address, one shape: one Cell");
        assert_eq!(ca.inner.lock().unwrap().members, 2);
        assert_eq!((OBJECTS.get(a).unwrap().state, OBJECTS.get(b).unwrap().state), (0x10, 0), "each object keeps its own state");
        // another shape at the address, and memory of its own: Cells of their own
        assert!(register(c, &desc(3, 1280, 720, 45, 1, 6), Some(at), 0x10));
        assert!(!Arc::ptr_eq(&ca, &cell_of(c).unwrap()));
        let own = 0x7B00_0000_4000usize;
        assert!(register(own, &x, None, 0x10));
        let co = cell_of(own).unwrap();
        assert!(!co.key.placed && co.key.at == own && !Arc::ptr_eq(&ca, &co));

        let in_cells = |k: &Key| CELLS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(k));
        forget(a);
        assert!(in_cells(&ca.key) && ca.inner.lock().unwrap().members == 1, "b still holds it");
        forget(b);
        assert!(!in_cells(&ca.key), "the last release takes the Cell");
        // a second registration of one address without a release gives the first one's membership back
        assert!(register(own, &x, None, 0x10));
        assert_eq!(cell_of(own).unwrap().inner.lock().unwrap().members, 1);
        forget(own);
        forget(c);
        assert!(!in_cells(&co.key) && cell_of(c).is_none());
    }

    /// The library's tests run with the switch as the process has it (by default as DEFAULT_ON).
    #[test]
    fn the_switch_decides_whether_a_buffer_is_kept_and_its_uavs_redirected() {
        let res = 0x7B20_0000_1000usize;
        note_created(res as *mut c_void, &desc(3, 64, 64, 45, 1, 6), Some(0x2_2000_0000), 0x10);
        assert_eq!(kept(res), enabled(), "registered only with the switch on");
        let rd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R24G8_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        // (no device: with the switch on the twin cannot be made, it is never a twin here)
        let why = match unsafe { resolve_uav(std::ptr::null_mut(), res as *mut c_void, Some(&rd), Some(&uav2d(DXGI_FORMAT_X24_TYPELESS_G8_UINT, 1))) } {
            Resolve::Null(why) => Some(why),
            Resolve::Twin(..) | Resolve::NotOurs => None,
        };
        if enabled() {
            assert_eq!(why, Some("the twin could not be made"));
        } else {
            assert_eq!(why, Some("XCL_DEPTH_UAV_TWIN is off"), "off: a null view, as before the twins");
        }
        forget(res);
        assert!(!kept(res));
    }

    #[test]
    fn the_state_a_buffer_was_made_in_is_that_of_the_titles_object() {
        let (view, title, at) = (0x7B10_0000_1000usize, 0x7B10_0000_2000usize, 0x2_1000_0000usize);
        let x = desc(3, 2560, 1440, 45, 1, 6);
        // the object of a view by address comes first and is made in COMMON: not the buffer's state
        assert!(by_address(|| register(view, &x, Some(at), 0)));
        assert_eq!(made_in(view), Some((None, Some(at))));
        assert!(!BY_ADDRESS.with(std::cell::Cell::get), "the mark ends with the call");
        assert!(register(title, &x, Some(at), 0x10));
        assert_eq!(made_in(view), Some((Some(0x10), Some(at))), "the title's object gives it");
        // a later object, or one made by address again, does not change it
        assert!(by_address(|| register(view, &x, Some(at), 0)));
        assert_eq!(made_in(title), Some((Some(0x10), Some(at))));
        forget(view);
        forget(title);
        assert!(cell_of(title).is_none());
    }

    #[test]
    fn a_copied_dsv_names_what_its_source_names() {
        let (res, inc) = (0x7C00_0000_1000usize, 0x20u64);
        let (heap, other) = (0x5_0000_0000u64, 0x6_0000_0000u64);
        assert!(register(res, &desc(3, 64, 64, 45, 1, 6), None, 0x10));
        note_dsv(heap as usize, res, 2);
        assert!(DSVS.get(heap as usize).is_some_and(|d| d.res == res && d.flags == 2));
        // a DSV of a resource that is not kept: not known (and an older entry there goes)
        note_dsv(heap as usize + 0x20, 0x7C00_0000_9000, 0);
        assert!(DSVS.get(heap as usize + 0x20).is_none());

        // two descriptors from `heap`: the first names `res`, the second nothing
        note_dsv(other as usize + 0x20, res, 0);
        copy_dsvs(2, other, heap, inc);
        assert!(DSVS.get(other as usize).is_some_and(|d| d.res == res && d.flags == 2 && d.cell.upgrade().is_some()));
        assert!(DSVS.get(other as usize + 0x20).is_none(), "a copy from an unknown handle forgets the destination");

        // ranges are cut where either side ends
        assert_eq!(pieces(&[(0x100, 3)], &[(0x500, 1), (0x900, 2)], inc), vec![(0x100, 0x500, 1), (0x120, 0x900, 2)]);
        assert_eq!(pieces(&[(0x100, 1), (0x300, 2)], &[(0x500, 2)], inc), vec![(0x100, 0x500, 1), (0x300, 0x520, 1)]);
        assert_eq!(pieces(&[(0x100, 0), (0x300, 1)], &[(0x500, 1)], inc), vec![(0x300, 0x500, 1)]);

        forget(res);
        assert!(DSVS.get(other as usize).is_some_and(|d| d.cell.upgrade().is_none()), "a DSV of a buffer that went names no Cell");
        forget_dsv(heap as usize);
        forget_dsv(other as usize);
    }

    fn uav2d(format: DXGI_FORMAT, plane: u32) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: plane } },
        }
    }

    fn uav2d_array(format: DXGI_FORMAT, mip: u32, first: u32, n: u32, plane: u32) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_UAV { MipSlice: mip, FirstArraySlice: first, ArraySize: n, PlaneSlice: plane } },
        }
    }

    #[test]
    fn a_view_names_the_stencil_plane_by_format_or_plane_slice() {
        // X32_TYPELESS_G8X24_UINT, X24_TYPELESS_G8_UINT, the R8 formats, the console's X16_TYPELESS_G8_UINT
        for f in [22, 47, 60, 61, 62, 63, 64, 120] {
            assert!(stencil_view(Some(&uav2d(DXGI_FORMAT(f), 0))), "format {f}");
        }
        assert!(stencil_view(Some(&uav2d(DXGI_FORMAT_R32_UINT, 1))), "plane slice 1");
        assert!(stencil_view(Some(&uav2d_array(DXGI_FORMAT_UNKNOWN, 0, 0, 2, 1))), "plane slice 1 of an array view");
        // R24_UNORM_X8_TYPELESS, R32_FLOAT, R16_UNORM, R32_UINT, R32_FLOAT_X8X24_TYPELESS on plane 0; no view
        for f in [46, 41, 56, 42, 21] {
            assert!(!stencil_view(Some(&uav2d(DXGI_FORMAT(f), 0))), "format {f}");
        }
        assert!(!stencil_view(None));
    }

    #[test]
    fn the_twins_view_keeps_the_slices_and_the_r8_class_asked_for() {
        let v = twin_view(Some(&uav2d_array(DXGI_FORMAT_X24_TYPELESS_G8_UINT, 1, 2, 3, 1)), 6);
        assert_eq!((v.Format, v.ViewDimension), (DXGI_FORMAT_R8_UINT, D3D12_UAV_DIMENSION_TEXTURE2DARRAY));
        let a = unsafe { v.Anonymous.Texture2DArray };
        assert_eq!((a.MipSlice, a.FirstArraySlice, a.ArraySize, a.PlaneSlice), (1, 2, 3, 0));
        let v = twin_view(Some(&uav2d(DXGI_FORMAT(120), 1)), 1);
        assert_eq!((v.Format, v.ViewDimension, unsafe { v.Anonymous.Texture2D.PlaneSlice }), (DXGI_FORMAT_R8_UINT, D3D12_UAV_DIMENSION_TEXTURE2D, 0));
        for f in [DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R8_UINT] {
            assert_eq!(twin_view(Some(&uav2d(f, 0)), 1).Format, f, "{f:?}");
        }
        // no view: the whole buffer
        let v = twin_view(None, 4);
        assert_eq!((v.Format, v.ViewDimension, unsafe { v.Anonymous.Texture2DArray.ArraySize }), (DXGI_FORMAT_R8_UINT, D3D12_UAV_DIMENSION_TEXTURE2DARRAY, 4));
        let v = twin_view(None, 1);
        assert_eq!((v.Format, v.ViewDimension, unsafe { v.Anonymous.Texture2D.PlaneSlice }), (DXGI_FORMAT_R8_UINT, D3D12_UAV_DIMENSION_TEXTURE2D, 0));
    }

    #[test]
    fn a_depth_view_is_a_view_of_a_twin_of_the_planes_size() {
        assert_eq!(
            [twin_format(39, 0), twin_format(19, 0), twin_format(53, 0), twin_format(44, 1), twin_format(19, 1)],
            [DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R8_TYPELESS],
            "D32, D32S8, D16 depth; stencil"
        );
        // an R32 twin: the depth as a float unless the title asked for another R32 class
        for (asked, got) in [
            (DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_UINT),
        ] {
            assert_eq!(depth_twin_view(Some(&uav2d(asked, 0)), 1, false).Format, got, "{asked:?}");
        }
        // an R16 one: UNORM unless another R16 class was asked for
        for (asked, got) in [
            (DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_UNORM),
            (DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_UNORM),
            (DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM),
            (DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16_UNORM),
            (DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_UINT),
        ] {
            assert_eq!(depth_twin_view(Some(&uav2d(asked, 0)), 1, true).Format, got, "{asked:?}");
        }
        let v = depth_twin_view(Some(&uav2d_array(DXGI_FORMAT_R32_FLOAT, 1, 2, 3, 0)), 6, false);
        let a = unsafe { v.Anonymous.Texture2DArray };
        assert_eq!((v.ViewDimension, a.MipSlice, a.FirstArraySlice, a.ArraySize, a.PlaneSlice), (D3D12_UAV_DIMENSION_TEXTURE2DARRAY, 1, 2, 3, 0));
        // no view: the whole buffer
        let v = depth_twin_view(None, 4, true);
        assert_eq!((v.Format, v.ViewDimension, unsafe { v.Anonymous.Texture2DArray.ArraySize }), (DXGI_FORMAT_R16_UNORM, D3D12_UAV_DIMENSION_TEXTURE2DARRAY, 4));
        // (the stencil twin's view is as it was)
        assert_eq!(twin_view(None, 1).Format, DXGI_FORMAT_R8_UINT);
    }

    /// The library's tests run with the switch as the process has it (by default as DEFAULT_ON).
    #[test]
    fn a_depth_uav_of_a_d32_or_d16_buffer_asks_for_a_twin_and_a_d24s8_one_stays_null() {
        let rd = |format| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let base = 0x7B30_0000_1000usize;
        // console D32, D16, D32S8, D24S8; the PC formats of the buffers; the views of their depth
        for (k, (console, pc, view, on)) in [
            (40u32, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT, "the twin could not be made"),
            (55, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM, "the twin could not be made"),
            (20, DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, "the twin could not be made"),
            (45, DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, "its depth plane has no twin yet"),
        ]
        .into_iter()
        .enumerate()
        {
            let res = base + k * 0x100;
            note_created(res as *mut c_void, &desc(3, 64, 64, console, 1, 6), None, 0x10);
            // (no device: with the switch on a twin is asked for and cannot be made)
            let why = match unsafe { resolve_uav(std::ptr::null_mut(), res as *mut c_void, Some(&rd(pc)), Some(&uav2d(view, 0))) } {
                Resolve::Null(why) => Some(why),
                Resolve::Twin(..) | Resolve::NotOurs => None,
            };
            assert_eq!(why, Some(if enabled() { on } else { "XCL_DEPTH_UAV_TWIN is off" }), "console format {console}");
            forget(res);
        }
    }

    #[test]
    fn the_resource_of_a_uav_clear_is_what_the_view_names() {
        let (twin, buffer, other) = (0x7D00_0000_1000usize, 0x7D00_0000_2000usize, 0x7D00_0000_3000usize);
        let twinned = [None, Some(twin)];
        assert_eq!(clear_choice(twinned, Some(twin)), Clear::Twin(twin), "the view names the twin");
        assert_eq!(clear_choice(twinned, Some(other)), Clear::AsIs, "a view of another resource");
        assert_eq!(clear_choice(twinned, None), Clear::NotDone, "a slot outside the shared heap: not known");
        // 0: a null view, or a copy of a slot outside the shared heap (which may hold the twin's
        // view); GONE: the null view of a destroyed resource
        assert_eq!(clear_choice(twinned, Some(0)), Clear::NotDone, "0: not known");
        assert_eq!(clear_choice(twinned, Some(super::super::console_heap::GONE)), Clear::NotDone, "GONE");
        // a buffer without a twin, whatever the slot
        for named in [None, Some(0), Some(buffer)] {
            assert_eq!(clear_choice([None, None], named), Clear::AsIs, "{named:?}");
        }

        // the lookup: a kept buffer with a stand-in twin (a COM object that only counts
        // references), one without a twin, and one XCL does not keep
        unsafe extern "system" fn add_ref(_this: *mut c_void) -> u32 {
            1
        }
        let vt: &'static [usize; 3] = Box::leak(Box::new([0, add_ref as usize, add_ref as usize]));
        let fake: &'static mut *const usize = Box::leak(Box::new(vt.as_ptr()));
        let fake = fake as *mut *const usize as *mut c_void;
        let untwinned = 0x7D00_0000_4000usize;
        assert!(register(buffer, &desc(3, 64, 64, 45, 1, 6), None, 0x10));
        assert!(register(untwinned, &desc(3, 32, 32, 45, 1, 6), None, 0x10));
        {
            let o = OBJECTS.get(buffer).unwrap();
            TWINS.fetch_add(1, Ordering::Relaxed);
            o.cell.inner.lock().unwrap().twins[1] = Some(Twin { res: unsafe { ID3D12Resource::from_raw(fake) }, n: 0 });
        }
        let t = fake as usize;
        let before = clears();
        assert_eq!(clear_target(buffer, Some(t)), Clear::Twin(t), "the slot names the twin");
        assert_eq!(clear_target(buffer, None), Clear::NotDone, "not known");
        assert_eq!(clear_target(buffer, Some(0)), Clear::NotDone, "0");
        assert_eq!(clear_target(buffer, Some(other)), Clear::AsIs, "a view of another resource");
        assert_eq!(clear_target(untwinned, None), Clear::AsIs, "a kept buffer without a twin");
        assert_eq!(clear_target(other, None), Clear::AsIs, "a buffer XCL does not keep");
        let after = clears();
        assert_eq!((after.0 - before.0, after.1 - before.1), (1, 2), "one clear on the twin, two not done");
        // (the last release takes the stand-in twin with the Cell)
        forget(buffer);
        forget(untwinned);
    }

    /// A stand-in for a twin: a COM object that only counts references.
    fn stand_in() -> ID3D12Resource {
        unsafe extern "system" fn add_ref(_this: *mut c_void) -> u32 {
            1
        }
        let vt: &'static [usize; 3] = Box::leak(Box::new([0, add_ref as usize, add_ref as usize]));
        let fake: &'static mut *const usize = Box::leak(Box::new(vt.as_ptr()));
        unsafe { ID3D12Resource::from_raw(fake as *mut *const usize as *mut c_void) }
    }

    /// `res` (kept) gets the stand-in twin `t` for its depth plane.
    fn plant_depth_twin(res: usize, t: &ID3D12Resource) {
        let o = OBJECTS.get(res).unwrap();
        TWINS.fetch_add(1, Ordering::Relaxed);
        o.cell.inner.lock().unwrap().twins[0] = Some(Twin { res: t.clone(), n: 0 });
    }

    fn tr(res: usize, sub: u32, before: u32, after: u32) -> [u32; 8] {
        [0, 0, res as u32, (res as u64 >> 32) as u32, sub, before, after, 0]
    }

    const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    const UA: u32 = UNORDERED_ACCESS;
    const DEPTH_WRITE: u32 = 0x10;
    const DEPTH_READ: u32 = 0x20;

    /// (the copies of a plan: (object, subresource of the transition, subresources copied, twin))
    fn copies(c: &[PlaneCopy]) -> Vec<(usize, u32, Vec<u32>, usize)> {
        c.iter().map(|c| (c.depth, c.sub, c.subs.clone(), c.twin.as_raw() as usize)).collect()
    }

    #[test]
    fn the_titles_unordered_access_of_a_depth_buffer_is_parked_and_its_twin_copied_around_it() {
        let res = 0x7E00_0000_1000usize;
        // a console D32 with unordered access, made in DEPTH_WRITE, with a depth twin
        assert!(register(res, &desc(3, 64, 64, 40, 1, 6), None, DEPTH_WRITE));
        let t = stand_in();
        plant_depth_twin(res, &t);
        let tw = t.as_raw() as usize;
        assert_eq!(depth_twin_of(res), Some(tw));
        let id = cell_of(res).unwrap().id;
        let asked = std::cell::Cell::new(0);
        let mut twin = |_r: usize, o: &Object| {
            asked.set(asked.get() + 1);
            twin_if_made(o)
        };
        let mut in_step = Vec::new();
        let mut plan = |b: [u32; 8], direct: bool, in_step: &mut Vec<u64>| {
            let mut v = vec![b];
            let p = plan_with(&mut v, direct, in_step, &mut twin);
            (v, copies(&p.to_plane), copies(&p.to_twin))
        };
        let one = |sub: u32| vec![(res, sub, vec![0], tw)];

        // into UNORDERED_ACCESS: the buffer goes to PARK, its plane is copied into the twin after
        // the barriers; the two are in step
        assert_eq!(plan(tr(res, ALL, DEPTH_WRITE, UA), true, &mut in_step), (vec![tr(res, ALL, DEPTH_WRITE, PARK)], vec![], one(ALL)));
        assert_eq!(in_step, [id]);
        // out of it into DEPTH_READ: the twin is copied into the plane first, the transition goes
        // on from COPY_DEST; still in step (nothing writes the plane in DEPTH_READ)
        assert_eq!(plan(tr(res, ALL, UA, DEPTH_READ), true, &mut in_step), (vec![tr(res, ALL, COPY_DEST, DEPTH_READ)], one(ALL), vec![]));
        assert_eq!(in_step, [id]);
        // into it again from DEPTH_READ: the twin holds the plane, no copy
        let (_, _, skipped) = plane_copies();
        assert_eq!(plan(tr(res, ALL, DEPTH_READ, UA), true, &mut in_step), (vec![tr(res, ALL, DEPTH_READ, PARK)], vec![], vec![]));
        assert!(plane_copies().2 > skipped, "counted as not needed");
        // out into DEPTH_WRITE: copied back; the plane is drawn into from then on
        assert_eq!(plan(tr(res, ALL, UA, DEPTH_WRITE), true, &mut in_step), (vec![tr(res, ALL, COPY_DEST, DEPTH_WRITE)], one(ALL), vec![]));
        assert!(in_step.is_empty());
        // so into it again is a copy again
        assert_eq!(plan(tr(res, ALL, DEPTH_WRITE, UA), true, &mut in_step), (vec![tr(res, ALL, DEPTH_WRITE, PARK)], vec![], one(ALL)));
        // a split transition out: its beginning goes, its end is done whole, with the copy
        let mut begin = tr(res, ALL, UA, DEPTH_WRITE);
        begin[1] = BEGIN_ONLY;
        assert_eq!(plan(begin, true, &mut in_step), (vec![], vec![], vec![]));
        let mut end = tr(res, ALL, UA, DEPTH_WRITE);
        end[1] = END_ONLY;
        assert_eq!(plan(end, true, &mut in_step), (vec![tr(res, ALL, COPY_DEST, DEPTH_WRITE)], one(ALL), vec![]));
        // a transition that leaves nothing to do (COPY_SOURCE is PARK) still copies
        assert_eq!(plan(tr(res, ALL, PARK, UA), true, &mut in_step).2, one(ALL));
        assert_eq!(plan(tr(res, ALL, UA, PARK), true, &mut in_step), (vec![tr(res, ALL, COPY_DEST, PARK)], one(ALL), vec![]));
        // on a compute list: parked, nothing copied – and a twin whose way in was not copied is
        // not copied back on the next DIRECT list either (it does not hold the plane)
        let before = asked.get();
        assert_eq!(plan(tr(res, ALL, DEPTH_READ, UA), false, &mut Vec::new()), (vec![tr(res, ALL, DEPTH_READ, PARK)], vec![], vec![]));
        assert_eq!(plan(tr(res, ALL, UA, DEPTH_READ), true, &mut Vec::new()), (vec![tr(res, ALL, PARK, DEPTH_READ)], vec![], vec![]));
        assert_eq!(asked.get(), before, "no twin asked for on a compute list or on the way out");
        // other transitions go as they came
        assert_eq!(plan(tr(res, ALL, DEPTH_READ, DEPTH_WRITE), true, &mut in_step), (vec![tr(res, ALL, DEPTH_READ, DEPTH_WRITE)], vec![], vec![]));
        forget(res);
    }

    #[test]
    fn only_depth_plane_subresources_are_copied_and_other_buffers_go_as_they_came() {
        let (res, d24, in_ua, unknown) = (0x7E10_0000_1000usize, 0x7E10_0000_2000usize, 0x7E10_0000_3000usize, 0x7E10_0000_4000usize);
        // a console D32S8 of 2 mips: depth plane subresources 0 and 1, stencil 2 and 3
        let mut x = desc(3, 64, 64, 20, 1, 6);
        x[0x1E..0x20].copy_from_slice(&2u16.to_le_bytes());
        assert!(register(res, &x, None, DEPTH_WRITE));
        let t = stand_in();
        plant_depth_twin(res, &t);
        let tw = t.as_raw() as usize;
        let mut twin = |_r: usize, o: &Object| twin_if_made(o);
        let mut plan = |b: [u32; 8]| {
            let mut v = vec![b];
            let p = plan_with(&mut v, true, &mut Vec::new(), &mut twin);
            (v, copies(&p.to_twin))
        };
        assert_eq!(plan(tr(res, ALL, DEPTH_WRITE, UA)), (vec![tr(res, ALL, DEPTH_WRITE, PARK)], vec![(res, ALL, vec![0, 1], tw)]));
        assert_eq!(plan(tr(res, 1, DEPTH_WRITE, UA)), (vec![tr(res, 1, DEPTH_WRITE, PARK)], vec![(res, 1, vec![1], tw)]));
        assert_eq!(plan(tr(res, 2, DEPTH_WRITE, UA)), (vec![tr(res, 2, DEPTH_WRITE, PARK)], vec![]), "a stencil subresource: parked, not copied");

        // a D24S8 (its depth plane has no twin), an object made in UNORDERED_ACCESS, one XCL does
        // not keep, a UAV barrier: as they came
        assert!(register(d24, &desc(3, 64, 64, 45, 1, 6), None, DEPTH_WRITE));
        assert!(register(in_ua, &desc(3, 64, 64, 40, 1, 6), None, UA));
        let uav_barrier = [2, 0, res as u32, (res as u64 >> 32) as u32, 0, 0, 0, 0];
        for b in [tr(d24, ALL, DEPTH_WRITE, UA), tr(in_ua, ALL, UA, DEPTH_WRITE), tr(unknown, ALL, DEPTH_WRITE, UA), uav_barrier] {
            assert_eq!(plan(b), (vec![b], vec![]), "{b:x?}");
        }
        forget(res);
        forget(d24);
        forget(in_ua);
    }

    /// The library's tests run with the switch as the process has it (by default as DEFAULT_ON).
    #[test]
    fn with_the_switch_off_the_titles_barriers_go_as_they_came() {
        let res = 0x7E20_0000_1000usize;
        note_created(res as *mut c_void, &desc(3, 64, 64, 40, 1, 6), None, DEPTH_WRITE);
        let mut b = vec![tr(res, ALL, DEPTH_WRITE, UA)];
        // (no device: no twin is made here)
        let p = plan_with(&mut b, true, &mut Vec::new(), &mut |_, _| None);
        if enabled() {
            assert_eq!(b, [tr(res, ALL, DEPTH_WRITE, PARK)], "on: parked");
        } else {
            assert_eq!(b, [tr(res, ALL, DEPTH_WRITE, UA)], "off: as it came");
        }
        assert!(p.to_plane.is_empty() && p.to_twin.is_empty());
        forget(res);
    }

    #[test]
    fn the_values_of_a_twin_are_counted_by_row_pitch() {
        // 3x2 texels, rows 8 bytes apart: the padding (0xEE) is not counted
        let data = [0, 8, 8, 0xEE, 0xEE, 0xEE, 0xEE, 0xEE, 7, 8, 0, 0xEE];
        let c = byte_counts(&data, 3, 2, 8);
        assert_eq!((c[0], c[7], c[8], c[0xEE]), (2, 1, 3, 0));
        let (top, distinct, zero) = most_frequent(&c);
        assert_eq!((top, distinct), (vec![(8, 3), (0, 2), (7, 1)], 3));
        assert!((zero - 100.0 * 2.0 / 6.0).abs() < 1e-9, "{zero}");
        // a buffer shorter than the rows it should hold: the rows that are there
        assert_eq!(byte_counts(&data[..10], 3, 2, 8).iter().sum::<u64>(), 3);
        // ten values once each and 200 three times: 200 first, then the smaller values, eight in all
        let mut c = [0u64; 256];
        (1..=10).for_each(|v| c[v] = 1);
        c[200] = 3;
        let (top, distinct, zero) = most_frequent(&c);
        assert_eq!((top, distinct, zero), (vec![(200, 3), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], 11, 0.0));
        assert_eq!(most_frequent(&[0; 256]), (vec![], 0, 0.0), "nothing counted");
    }
}
