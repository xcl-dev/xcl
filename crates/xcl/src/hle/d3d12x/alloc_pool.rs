//! A title's command allocator Reset does not take recordings the title still submits.
//!
//! On the PC a list's recording lives in its allocator: after `allocator.Reset()` a list that
//! recorded on it must not be submitted again ("The command allocator was reset after the
//! command list was recorded", debug layer id 541 – the runtime removes the device,
//! 0x887a0001). A retail title does exactly that, which its console lets it: one recording,
//! submitted at 61.6 s, at 63.2 s on a second queue and at 67.8 s again, with the allocator
//! it had been given reset at 67.2 s.
//!
//! So the title's allocator is a name for a PC allocator that changes: at the title's Reset,
//! when lists still hold recordings on the PC allocator in use (or the GPU has not finished
//! them), that one is put aside as it is and the title's allocator goes on with another – one
//! put aside earlier whose lists have all been reset since and whose submits are done, or a new
//! one. A list reset on the title's allocator records on the PC allocator in use. A list holds
//! one recording, so no more allocators are put aside than there are lists.
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

#[derive(Default)]
struct Backing {
    /// Raw-address generation: a completed probe must not free a replacement at the same key.
    generation: u64,
    /// PC lists whose recording is on this allocator
    users: HashSet<usize>,
    /// queue → (fence, value after the last submit of one of them)
    marks: HashMap<usize, (ID3D12Fence, u64)>,
    /// Submits reserved before ExecuteCommandLists whose completion is not marked yet.
    submit_pins: usize,
    /// Probe reference transferred from the title's Release while this backing is busy.
    release_owner: Option<ID3D12CommandAllocator>,
    /// Its allocator namespace is gone; release its retained/raw owner once it becomes free.
    retire_when_free: bool,
    /// The driver Reset this backing while idle, and no recording has used it since.
    already_reset: bool,
}

#[derive(Default)]
struct State {
    /// title allocator → the PC allocator in use (absent: itself)
    cur: HashMap<usize, usize>,
    /// title allocator → PC allocators put aside
    aside: HashMap<usize, Vec<usize>>,
    backings: HashMap<usize, Backing>,
    next_backing_generation: u64,
    /// Title allocator -> exact reset operation currently outside this lock.
    resetting: HashMap<usize, u64>,
    next_reset_generation: u64,
    of_list: HashMap<usize, usize>,
    /// title allocator → the type of the lists on it
    types: HashMap<usize, i32>,
    fences: HashMap<usize, Arc<QueueFence>>,
    made: u64,
    swaps: u64,
    /// PC list → its last submit (queue, fence, value after it, recording number then)
    last_submit: HashMap<usize, LastSubmit>,
    /// PC list → recordings so far (a Reset or creation starts the next one)
    recordings: HashMap<usize, u64>,
    /// PC list → object lifetime, so a late reservation cannot mark a reused address.
    list_lifetimes: HashMap<usize, u64>,
    next_list_lifetime: u64,
    resubmits: u64,
    /// (PC list, object lifetime) → reserved submits not yet associated with a queue fence.
    submit_pins: HashMap<(usize, u64), usize>,
}

enum ResetWork {
    Reset { owner: ID3D12CommandAllocator, swap: bool, needed: bool },
    Create { owner: ID3D12CommandAllocator, ty: i32 },
}

struct ResetClaim {
    alloc: usize,
    current: usize,
    token: u64,
    mine: Vec<usize>,
    probes: Vec<BackingProbe>,
    work: ResetWork,
}

struct IdleResetClaim {
    alloc: usize,
    token: u64,
    mine: Vec<usize>,
    probes: Vec<BackingProbe>,
    work: Vec<(BackingProbe, ID3D12CommandAllocator)>,
}

#[derive(Clone)]
struct LastSubmit {
    queue: usize,
    fence: ID3D12Fence,
    value: u64,
    recording: u64,
}

struct QueueFence {
    fence: ID3D12Fence,
    next_value: AtomicU64,
}

impl QueueFence {
    fn new(fence: ID3D12Fence) -> Self {
        Self { fence, next_value: AtomicU64::new(0) }
    }
}

fn same_submit(a: &LastSubmit, b: &LastSubmit) -> bool {
    a.queue == b.queue
        && a.fence.as_raw() == b.fence.as_raw()
        && a.value == b.value
        && a.recording == b.recording
}

unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

#[derive(Clone)]
struct BackingProbe {
    raw: usize,
    generation: u64,
    users: HashSet<usize>,
    marks: Vec<(usize, ID3D12Fence, u64)>,
    submit_pins: usize,
    release_owner: Option<usize>,
    retire_when_free: bool,
    already_reset: bool,
}

impl BackingProbe {
    fn new(raw: usize, backing: &Backing) -> Self {
        Self {
            raw,
            generation: backing.generation,
            users: backing.users.clone(),
            marks: backing.marks.iter().map(|(&queue, (fence, value))| (queue, fence.clone(), *value)).collect(),
            submit_pins: backing.submit_pins,
            release_owner: backing.release_owner.as_ref().map(|owner| owner.as_raw() as usize),
            retire_when_free: backing.retire_when_free,
            already_reset: backing.already_reset,
        }
    }

    unsafe fn free(&self) -> bool {
        self.users.is_empty() && self.submit_pins == 0 && self.marks.iter().all(|(_, fence, value)| fence.GetCompletedValue() >= *value)
    }

    unsafe fn in_flight(&self) -> bool {
        self.submit_pins != 0 || self.marks.iter().any(|(_, fence, value)| fence.GetCompletedValue() < *value)
    }

    fn matches(&self, backing: &Backing) -> bool {
        self.generation == backing.generation
            && self.users == backing.users
            && self.submit_pins == backing.submit_pins
            && self.release_owner == backing.release_owner.as_ref().map(|owner| owner.as_raw() as usize)
            && self.retire_when_free == backing.retire_when_free
            && self.already_reset == backing.already_reset
            && self.marks.len() == backing.marks.len()
            && self.marks.iter().all(|(queue, fence, value)| {
                backing.marks.get(queue).is_some_and(|(current, current_value)| current.as_raw() == fence.as_raw() && current_value == value)
            })
    }
}

fn backing_mut(state: &mut State, raw: usize) -> &mut Backing {
    if !state.backings.contains_key(&raw) {
        state.next_backing_generation = state.next_backing_generation.wrapping_add(1).max(1);
        state.backings.insert(raw, Backing { generation: state.next_backing_generation, ..Default::default() });
    }
    state.backings.get_mut(&raw).unwrap()
}

fn namespace(state: &State, alloc: usize) -> Vec<usize> {
    let mut mine: Vec<_> = state.cur.get(&alloc).into_iter().chain(state.aside.get(&alloc).into_iter().flatten()).chain([&alloc]).copied().collect();
    mine.sort_unstable();
    mine.dedup();
    mine
}

fn namespace_probes(state: &State, alloc: usize) -> (Vec<usize>, Vec<BackingProbe>) {
    let mine = namespace(state, alloc);
    let probes = mine.iter().filter_map(|raw| state.backings.get(raw).map(|backing| BackingProbe::new(*raw, backing))).collect();
    (mine, probes)
}

fn namespace_matches(state: &State, alloc: usize, mine: &[usize], probes: &[BackingProbe]) -> bool {
    namespace(state, alloc) == mine
        && mine.iter().all(|raw| match (state.backings.get(raw), probes.iter().find(|probe| probe.raw == *raw)) {
            (None, None) => true,
            (Some(backing), Some(probe)) => probe.matches(backing),
            _ => false,
        })
}

fn reserve_reset(state: &mut State, alloc: usize) -> Option<u64> {
    if state.resetting.contains_key(&alloc) {
        return None;
    }
    state.next_reset_generation = state.next_reset_generation.wrapping_add(1).max(1);
    let token = state.next_reset_generation;
    state.resetting.insert(alloc, token);
    Some(token)
}

fn cancel_reset(state: &mut State, alloc: usize, token: u64) {
    if state.resetting.get(&alloc) == Some(&token) {
        state.resetting.remove(&alloc);
    }
}

fn run_reset(raw: *mut c_void, reset: impl FnOnce(*mut c_void) -> HRESULT) -> HRESULT {
    reset(raw)
}

#[cfg(test)]
unsafe fn backing_free_for_test(backing: &Backing) -> bool { BackingProbe::new(0, backing).free() }

/// Remove completion caches for a retired queue before its raw address can be reused. This is
/// called only after its retirement mark completes; every removed COM fence drops unlocked.
pub(super) fn forget_queue(id: super::present::QueueId) {
    let queue = id.raw();
    let (marker, gone) = {
        let mut state = STATE.lock().unwrap();
        let Some(state) = state.as_mut() else { return };
        let mut gone = Vec::new();
        let marker = state.fences.remove(&queue);
        for backing in state.backings.values_mut() {
            if let Some((fence, _)) = backing.marks.remove(&queue) {
                gone.push(fence);
            }
        }
        let lists: Vec<_> = state.last_submit.iter().filter_map(|(&list, submit)| (submit.queue == queue).then_some(list)).collect();
        for list in lists {
            if let Some(submit) = state.last_submit.remove(&list) {
                gone.push(submit.fence);
            }
        }
        (marker, gone)
    };
    drop(marker);
    drop(gone);
}

pub(super) fn release_with_probe(
    add_ref: impl FnOnce(),
    mut release: impl FnMut() -> u32,
    retain_probe: impl FnOnce() -> bool,
) -> (u32, bool) {
    add_ref();
    let after_caller = release();
    if after_caller == 1 && retain_probe() { (0, true) } else { (release(), false) }
}

/// Transfer the caller's temporary AddRef probe into pool ownership when releasing `raw` would
/// otherwise destroy an allocator that still owns recordings or unretired GPU work.
pub(super) unsafe fn retain_release_probe(raw: *mut c_void) -> bool {
    let key = raw as usize;
    loop {
        let probe = {
            let g = STATE.lock().unwrap();
            let Some(backing) = g.as_ref().and_then(|s| s.backings.get(&key)) else { return false };
            if backing.release_owner.is_some() { return false }
            BackingProbe::new(key, backing)
        };
        let free = probe.free();
        let mut g = STATE.lock().unwrap();
        let Some(backing) = g.as_mut().and_then(|s| s.backings.get_mut(&key)) else { return false };
        if backing.release_owner.is_some() {
            return false;
        }
        if backing.generation != probe.generation || !probe.matches(backing) {
            drop(g);
            continue;
        }
        if free {
            return false;
        }
        backing.release_owner = Some(ID3D12CommandAllocator::from_raw(raw));
        return true;
    }
}

/// Drop transferred Release probes only after leaving STATE: their hooked Release re-enters here.
pub(super) fn sweep_release_owners() {
    sweep_release_owners_with(super::real::release_retained_allocator, drop);
}

fn sweep_release_owners_with(
    mut release_transferred: impl FnMut(ID3D12CommandAllocator),
    mut release_made: impl FnMut(ID3D12CommandAllocator),
) {
    loop {
        let probes: Vec<_> = {
            let g = STATE.lock().unwrap();
            let Some(s) = g.as_ref() else { return };
            s.backings.iter().filter_map(|(&raw, backing)| backing.retire_when_free.then(|| BackingProbe::new(raw, backing))).collect()
        };
        let ready: Vec<_> = probes.into_iter().filter(|probe| unsafe { probe.free() }).collect();
        if ready.is_empty() { return }
        let removed = {
            let mut g = STATE.lock().unwrap();
            let Some(s) = g.as_mut() else { return };
            ready.iter().find_map(|probe| {
                s.backings.get(&probe.raw).is_some_and(|backing| probe.matches(backing)).then(|| probe.raw)
            }).and_then(|raw| s.backings.remove(&raw).map(|backing| (raw, backing)))
        };
        let Some((raw, mut backing)) = removed else { continue };
        let owner = backing.release_owner.take();
        drop(backing);
        super::console_list::allocator_reset(raw as *mut c_void);
        match owner {
            Some(owner) => release_transferred(owner),
            None => release_made(unsafe { ID3D12CommandAllocator::from_raw(raw as *mut c_void) }),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct AllocatorSubmitStats {
    pub outer_execute_calls: u64,
    pub chunks: u64,
    pub multi_chunk_calls: u64,
    pub forced_wait_boundaries: u64,
    pub lists_marked: u64,
    /// Backing mark-map insertions, including overwrites when lists share a backing.
    pub backing_mark_writes: u64,
    pub signal_attempts: u64,
    pub signal_failures: u64,
    pub state_lock_wait_ns: u64,
    pub state_lock_hold_ns: u64,
}

struct SubmitCounters {
    outer_execute_calls: AtomicU64,
    chunks: AtomicU64,
    multi_chunk_calls: AtomicU64,
    forced_wait_boundaries: AtomicU64,
    lists_marked: AtomicU64,
    backing_mark_writes: AtomicU64,
    signal_attempts: AtomicU64,
    signal_failures: AtomicU64,
    state_lock_wait_ns: AtomicU64,
    state_lock_hold_ns: AtomicU64,
}

macro_rules! submit_counters {
    ($($name:ident),+ $(,)?) => {
        static SUBMIT: SubmitCounters = SubmitCounters { $($name: AtomicU64::new(0)),+ };

        pub fn submit_stats() -> AllocatorSubmitStats {
            AllocatorSubmitStats { $($name: SUBMIT.$name.load(Ordering::Relaxed)),+ }
        }
    };
}

submit_counters!(
    outer_execute_calls, chunks, multi_chunk_calls, forced_wait_boundaries,
    lists_marked, backing_mark_writes, signal_attempts, signal_failures,
    state_lock_wait_ns, state_lock_hold_ns,
);

fn add_duration(counter: &AtomicU64, elapsed: std::time::Duration) {
    counter.fetch_add(elapsed.as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
}

struct SubmitStateGuard<'a> {
    guard: MutexGuard<'a, Option<State>>,
    held: std::time::Instant,
}

impl Deref for SubmitStateGuard<'_> {
    type Target = Option<State>;
    fn deref(&self) -> &Self::Target { &self.guard }
}

impl DerefMut for SubmitStateGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.guard }
}

impl Drop for SubmitStateGuard<'_> {
    fn drop(&mut self) {
        add_duration(&SUBMIT.state_lock_hold_ns, self.held.elapsed());
    }
}

fn lock_for_submit() -> SubmitStateGuard<'static> {
    let waiting = std::time::Instant::now();
    let guard = STATE.lock().unwrap();
    add_duration(&SUBMIT.state_lock_wait_ns, waiting.elapsed());
    SubmitStateGuard { guard, held: std::time::Instant::now() }
}

pub fn note_execute_call(chunks: usize) {
    let (multi_chunk, boundaries) = execute_topology(chunks);
    SUBMIT.outer_execute_calls.fetch_add(1, Ordering::Relaxed);
    SUBMIT.chunks.fetch_add(chunks as u64, Ordering::Relaxed);
    SUBMIT.multi_chunk_calls.fetch_add(multi_chunk, Ordering::Relaxed);
    SUBMIT.forced_wait_boundaries.fetch_add(boundaries, Ordering::Relaxed);
}

fn execute_topology(chunks: usize) -> (u64, u64) {
    ((chunks > 1) as u64, chunks.saturating_sub(1) as u64)
}

#[derive(Clone, Copy)]
struct SubmitEntry {
    list: usize,
    backing: Option<usize>,
    recording: u64,
    lifetime: u64,
}

/// A fail-closed reservation made before native ExecuteCommandLists. Dropping it before
/// `execution_possible` releases the reservation because no GPU work can refer to it yet.
#[must_use]
pub(super) struct BeginSubmit {
    entries: Vec<SubmitEntry>,
}

/// A reservation armed immediately before native execution. Once armed, execution may begin at
/// any point, so dropping it before `finish` deliberately leaves its allocator and lists pinned.
#[must_use]
pub(super) struct PendingSubmit {
    entries: Vec<SubmitEntry>,
}

/// Exact queue progress after a native submission. Keeping the fence owner makes this receipt
/// independent of a command queue raw address being released and reused for another generation.
#[derive(Clone)]
#[allow(dead_code)] // consumed by the later opt-in generation-pin stage
pub(super) struct SubmissionReceipt {
    marker: Arc<QueueFence>,
    value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SubmissionReceiptStatus {
    Pending,
    Complete,
    DeviceLost,
}

#[allow(dead_code)] // consumed by the later opt-in generation-pin stage
impl SubmissionReceipt {
    /// Exact queue-marker generation. The retained `Arc` prevents pointer reuse while a
    /// retirement sweep observes this identity.
    pub(super) fn fence_identity(&self) -> usize {
        Arc::as_ptr(&self.marker) as usize
    }

    pub(super) fn completed_value(&self) -> u64 {
        unsafe { self.marker.fence.GetCompletedValue() }
    }

    pub(super) fn target_value(&self) -> u64 {
        self.value
    }

    /// Exact state of the native submission represented by this receipt. D3D12 fences return
    /// `u64::MAX` after device removal; that sentinel is not successful completion.
    pub(super) fn status(&self) -> SubmissionReceiptStatus {
        match self.completed_value() {
            u64::MAX => SubmissionReceiptStatus::DeviceLost,
            completed if completed >= self.value => SubmissionReceiptStatus::Complete,
            _ => SubmissionReceiptStatus::Pending,
        }
    }

    /// Whether resources protected by this receipt may be retired. Device loss ends their GPU
    /// use even though diagnostics must not report it as successful submission completion.
    pub(super) fn is_complete(&self) -> bool {
        self.completed_value() >= self.value
    }
}

/// Best-effort exact queue progress for diagnostics that own no submitted resources. The caller
/// must already hold the queue's ordered turn. Failure leaves the sample untracked and does not
/// poison a [`SubmissionGate`] or change later title submission behavior.
pub(super) unsafe fn try_timing_receipt(queue: *mut c_void) -> Option<SubmissionReceipt> {
    PendingSubmit { entries: Vec::new() }.finish_receipted(queue)
}

#[cfg(test)]
pub(super) fn submission_receipt_for_test(completed: u64, value: u64) -> (SubmissionReceipt, &'static AtomicU64) {
    use std::mem::{offset_of, size_of};
    use std::sync::atomic::AtomicU32;

    #[repr(C)]
    struct Fence { vtable: *const usize, completed: AtomicU64, refs: AtomicU32 }
    unsafe extern "system" fn get_completed(this: *mut c_void) -> u64 {
        (*(this as *mut Fence)).completed.load(Ordering::Acquire)
    }
    unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
        (*(this as *mut Fence)).refs.fetch_add(1, Ordering::Relaxed) + 1
    }
    unsafe extern "system" fn release(this: *mut c_void) -> u32 {
        (*(this as *mut Fence)).refs.fetch_sub(1, Ordering::Relaxed) - 1
    }
    extern "system" fn unexpected() { panic!("unexpected test fence call") }

    let mut vtable = vec![unexpected as usize; size_of::<ID3D12Fence_Vtbl>() / size_of::<usize>()].into_boxed_slice();
    vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / size_of::<usize>()] = get_completed as usize;
    vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.AddRef) / size_of::<usize>()] = add_ref as usize;
    vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / size_of::<usize>()] = release as usize;
    let vtable = Box::leak(vtable);
    let object = Box::leak(Box::new(Fence {
        vtable: vtable.as_ptr(), completed: AtomicU64::new(completed), refs: AtomicU32::new(1),
    }));
    let object = object as *mut Fence;
    let completed = unsafe { &(*object).completed };
    let fence = unsafe { ID3D12Fence::from_raw(object.cast()) };
    (SubmissionReceipt { marker: Arc::new(QueueFence::new(fence)), value }, completed)
}

/// Result of a receipt-requesting submission. Once execution begins, failure to mark it cannot
/// be turned back into `NotExecuted`: callers must retain dependent objects indefinitely.
#[allow(dead_code)] // consumed by the later opt-in generation-pin stage
pub(super) enum ReceiptedSubmission {
    NotExecuted,
    Executed(SubmissionReceipt),
    ExecutedUntracked,
}

fn release_submit_pins(s: &mut State, entries: &[SubmitEntry]) {
    for entry in entries {
        let key = (entry.list, entry.lifetime);
        if let Some(n) = s.submit_pins.get_mut(&key) {
            *n -= 1;
            if *n == 0 {
                s.submit_pins.remove(&key);
            }
        }
        if let Some(backing) = entry.backing.and_then(|b| s.backings.get_mut(&b)) {
            backing.submit_pins = backing.submit_pins.saturating_sub(1);
        }
    }
}

fn record_submit(s: &mut State, entries: &[SubmitEntry], queue: usize, fence: &ID3D12Fence, value: u64) -> (u64, Vec<ID3D12Fence>) {
    let mut backing_mark_writes = 0;
    let mut replaced = Vec::new();
    for entry in entries {
        if let Some(backing) = entry.backing {
            backing_mark_writes += 1;
            let backing = backing_mut(s, backing);
            backing.already_reset = false;
            if let Some((old, _)) = backing.marks.insert(queue, (fence.clone(), value)) {
                replaced.push(old);
            }
        }
        if s.list_lifetimes.get(&entry.list) == Some(&entry.lifetime) {
            if let Some(old) = s.last_submit.insert(
                entry.list,
                LastSubmit { queue, fence: fence.clone(), value, recording: entry.recording },
            ) {
                replaced.push(old.fence);
            }
        }
    }
    release_submit_pins(s, entries);
    (backing_mark_writes, replaced)
}

/// Capture and pin the exact backing and recording that a native submit will execute. The caller
/// must keep this token until it knows ExecuteCommandLists cannot have run, or convert it with
/// `execution_possible` immediately before making the native call.
pub(super) fn begin_submit(lists: &[*mut c_void]) -> Option<BeginSubmit> {
    sweep_release_owners();
    if !enabled() {
        return Some(BeginSubmit { entries: Vec::new() });
    }
    let mut entries = Vec::new();
    entries.try_reserve(lists.len()).ok()?;
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    s.list_lifetimes.try_reserve(lists.len()).ok()?;
    s.submit_pins.try_reserve(lists.len()).ok()?;
    for &list in lists {
        let list = list as usize;
        if entries.iter().any(|entry: &SubmitEntry| entry.list == list) {
            continue;
        }
        let lifetime = match s.list_lifetimes.get(&list) {
            Some(&lifetime) => lifetime,
            None => {
                s.next_list_lifetime = s.next_list_lifetime.wrapping_add(1).max(1);
                s.list_lifetimes.insert(list, s.next_list_lifetime);
                s.next_list_lifetime
            }
        };
        entries.push(SubmitEntry {
            list,
            backing: s.of_list.get(&list).copied(),
            recording: s.recordings.get(&list).copied().unwrap_or(0),
            lifetime,
        });
    }
    for entry in &entries {
        *s.submit_pins.entry((entry.list, entry.lifetime)).or_default() += 1;
        if let Some(backing) = entry.backing {
            s.backings.get_mut(&backing).unwrap().submit_pins += 1;
        }
    }
    Some(BeginSubmit { entries })
}

impl BeginSubmit {
    /// Arm immediately before calling native ExecuteCommandLists. From this point onward, any
    /// unwind, drop, or completion-mark failure must retain the reservation permanently.
    pub(super) fn execution_possible(mut self) -> PendingSubmit {
        PendingSubmit { entries: std::mem::take(&mut self.entries) }
    }
}

impl Drop for BeginSubmit {
    fn drop(&mut self) {
        if !self.entries.is_empty() {
            let mut g = STATE.lock().unwrap();
            if let Some(s) = g.as_mut() {
                release_submit_pins(s, &self.entries);
            }
            drop(g);
            sweep_release_owners();
        }
    }
}

impl PendingSubmit {
    /// Associate the captured submit with the queue's existing retirement fence. `false` leaves
    /// all reservations permanently pinned: execution may have happened but cannot be retired.
    pub(super) unsafe fn finish(self, queue: *mut c_void) -> bool {
        if self.entries.is_empty() {
            return true;
        }
        let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return false };
        let key = queue as usize;
        super::queue_submit::with(key, || self.finish_with_queue_ops(
            key,
            || {
                let mut dev: Option<ID3D12Device> = None;
                q.GetDevice(&mut dev).ok()?;
                dev?.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok()
            },
            |fence, value| super::real::native_queue_signal(q, fence, value).is_ok(),
        ))
    }

    /// Mark a submission and return its exact queue fence/value. Unlike `finish`, this also makes
    /// a marker when allocator swapping is disabled and therefore `entries` is empty.
    #[allow(dead_code)] // consumed through `submit_receipted` by the later generation-pin stage
    unsafe fn finish_receipted(self, queue: *mut c_void) -> Option<SubmissionReceipt> {
        let q = ID3D12CommandQueue::from_raw_borrowed(&queue)?;
        let key = queue as usize;
        super::queue_submit::with(key, || self.finish_with_queue_ops_receipted(
            key,
            || {
                let mut dev: Option<ID3D12Device> = None;
                q.GetDevice(&mut dev).ok()?;
                dev?.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok()
            },
            |fence, value| super::real::native_queue_signal(q, fence, value).is_ok(),
        ))
    }

    /// The queue turn excludes another thread while the state lock is released around driver
    /// calls. An `Arc` keeps the exact marker alive; replacement means the queue generation was
    /// invalidated, so the submitted work deliberately remains pinned.
    fn finish_with_queue_ops(
        self,
        key: usize,
        create: impl FnOnce() -> Option<ID3D12Fence>,
        signal: impl FnOnce(&ID3D12Fence, u64) -> bool,
    ) -> bool {
        if self.entries.is_empty() {
            return true;
        }
        self.finish_with_queue_ops_inner(key, create, signal).is_some()
    }

    #[allow(dead_code)] // production consumer lands with the generation-pin stage
    fn finish_with_queue_ops_receipted(
        self,
        key: usize,
        create: impl FnOnce() -> Option<ID3D12Fence>,
        signal: impl FnOnce(&ID3D12Fence, u64) -> bool,
    ) -> Option<SubmissionReceipt> {
        // `begin_submit` intentionally does not initialize allocator state when swapping is off;
        // a requested receipt still needs the shared queue-generation marker namespace.
        if self.entries.is_empty() {
            STATE.lock().unwrap().get_or_insert_with(State::default);
        }
        self.finish_with_queue_ops_inner(key, create, signal)
    }

    fn finish_with_queue_ops_inner(
        mut self,
        key: usize,
        create: impl FnOnce() -> Option<ID3D12Fence>,
        signal: impl FnOnce(&ID3D12Fence, u64) -> bool,
    ) -> Option<SubmissionReceipt> {
        let marker = {
            let g = lock_for_submit();
            let s = g.as_ref()?;
            s.fences.get(&key).cloned()
        };
        let marker = match marker {
            Some(marker) => marker,
            None => {
                let fence = create()?;
                let made = Arc::new(QueueFence::new(fence));
                let mut g = lock_for_submit();
                let s = g.as_mut()?;
                let (marker, unused) = match s.fences.entry(key) {
                    Entry::Occupied(entry) => (Arc::clone(entry.get()), Some(made)),
                    Entry::Vacant(entry) => {
                        let marker = Arc::clone(&made);
                        entry.insert(made);
                        (marker, None)
                    }
                };
                drop(g);
                drop(unused);
                marker
            }
        };
        // Reserve before entering the driver. The queue lease excludes another thread, while an
        // atomic reservation lets intentional same-thread re-entry take a distinct later value.
        // A failed Signal burns its value; its submit remains pinned and a gap is harmless.
        let Ok(value) = marker.next_value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1)) else {
            return None;
        };
        let value = value + 1;
        SUBMIT.signal_attempts.fetch_add(1, Ordering::Relaxed);
        if !signal(&marker.fence, value) {
            SUBMIT.signal_failures.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let fence = marker.fence.clone();
        let mut g = lock_for_submit();
        let s = g.as_mut()?;
        if s.fences.get(&key).is_none_or(|current| !Arc::ptr_eq(current, &marker)) {
            return None;
        }
        let (backing_mark_writes, replaced) = record_submit(s, &self.entries, key, &fence, value);
        SUBMIT.lists_marked.fetch_add(self.entries.len() as u64, Ordering::Relaxed);
        SUBMIT.backing_mark_writes.fetch_add(backing_mark_writes, Ordering::Relaxed);
        self.entries.clear();
        drop(g);
        drop(replaced);
        sweep_release_owners();
        Some(SubmissionReceipt { marker, value })
    }

    #[cfg(test)]
    fn finish_with_mark(mut self, queue: usize, fence: &ID3D12Fence, value: u64) {
        let mut g = STATE.lock().unwrap();
        let (_, replaced) = record_submit(g.as_mut().unwrap(), &self.entries, queue, fence, value);
        self.entries.clear();
        drop(g);
        drop(replaced);
    }
}

fn reserve_and_execute_with<B, P>(
    failed: &mut bool,
    begin: impl FnOnce() -> Option<B>,
    prepare: impl FnOnce(),
    arm: impl FnOnce(B) -> P,
    execute: impl FnOnce(),
    finish: impl FnOnce(P) -> bool,
) -> bool {
    if *failed {
        return false;
    }
    let Some(begin) = begin() else {
        *failed = true;
        return false;
    };
    prepare();
    let pending = arm(begin);
    execute();
    let _ = finish(pending);
    true
}

#[derive(Default)]
pub(super) struct SubmissionGate {
    failed: bool,
}

impl SubmissionGate {
    /// `prepare` runs before the reservation is armed; `execute` must contain only the native call.
    pub(super) unsafe fn submit(
        &mut self,
        queue: *mut c_void,
        lists: &[*mut c_void],
        prepare: impl FnOnce(),
        execute: impl FnOnce(),
    ) -> bool {
        reserve_and_execute_with(
            &mut self.failed,
            || begin_submit(lists),
            prepare,
            BeginSubmit::execution_possible,
            execute,
            |pending| pending.finish(queue),
        )
    }

    /// Submit and request an exact retirement receipt. The existing `submit` path remains the
    /// zero-extra-marker path; this variant is for future resources that need queue-fenced pins.
    #[allow(dead_code)] // production consumer lands with the generation-pin stage
    pub(super) unsafe fn submit_receipted(
        &mut self,
        queue: *mut c_void,
        lists: &[*mut c_void],
        prepare: impl FnOnce(),
        execute: impl FnOnce(),
    ) -> ReceiptedSubmission {
        if self.failed {
            return ReceiptedSubmission::NotExecuted;
        }
        let Some(begin) = begin_submit(lists) else {
            self.failed = true;
            return ReceiptedSubmission::NotExecuted;
        };
        prepare();
        let pending = begin.execution_possible();
        execute();
        match pending.finish_receipted(queue) {
            Some(receipt) => ReceiptedSubmission::Executed(receipt),
            None => {
                self.failed = true;
                ReceiptedSubmission::ExecutedUntracked
            }
        }
    }

    pub(super) fn failed(&self) -> bool {
        self.failed
    }
}

/// `XCL_ALLOC_SWAP=0` turns it off.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_ALLOC_SWAP").is_ok_and(|v| v.trim() == "0"))
}

fn move_list(s: &mut State, list: usize, backing: usize) {
    if let Some(old) = s.of_list.insert(list, backing) {
        if old != backing {
            if let Some(b) = s.backings.get_mut(&old) {
                b.users.remove(&list);
            }
        }
    }
    let backing = backing_mut(s, backing);
    backing.users.insert(list);
    backing.already_reset = false;
}

/// Detach a recording that is no longer in the title list's submission graph. Its object
/// lifetime, last submit and submit pins remain, so address reuse and GPU completion stay exact.
pub(super) fn retire_recording(list: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    retire_recording_in(s, list);
    drop(g);
    sweep_release_owners();
}

fn retire_recording_in(s: &mut State, list: usize) {
    if let Some(backing) = s.of_list.remove(&list) {
        if let Some(backing) = s.backings.get_mut(&backing) {
            backing.users.remove(&list);
        }
    }
}

/// The PC list's final Release ended its recording. Submitted work still holds the backing
/// through `marks`; removing the list must not remove those fences or the other users.
pub(super) fn forget_list(list: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let last_submit = s.last_submit.remove(&list);
    s.recordings.remove(&list);
    if let Some(lifetime) = s.list_lifetimes.remove(&list) {
        s.submit_pins.remove(&(list, lifetime));
    }
    if let Some(backing) = s.of_list.remove(&list) {
        if let Some(b) = s.backings.get_mut(&backing) {
            b.users.remove(&list);
        }
    }
    drop(g);
    drop(last_submit);
    sweep_release_owners();
}

/// Reset `list` on the title's allocator `alloc`, committing its new recording association
/// only after native Reset succeeds. No pool lock is held across the driver call.
pub fn reset_list(list: usize, alloc: usize, ty: i32, reset: impl FnOnce(usize) -> HRESULT) -> HRESULT {
    let backing = current(alloc);
    let hr = reset(backing);
    if hr < 0 || !enabled() || alloc == 0 {
        return hr;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    if ty >= 0 {
        s.types.insert(alloc, ty);
    }
    move_list(s, list, backing);
    *s.recordings.entry(list).or_default() += 1;
    hr
}

#[cfg(test)]
pub(super) use recording_backing as recording_backing_for_test;

pub(super) fn recording_backing(list: usize) -> Option<usize> {
    STATE.lock().unwrap().as_ref()?.of_list.get(&list).copied()
}

/// The PC allocator the title's allocator `alloc` stands for now.
pub fn current(alloc: usize) -> usize {
    if !enabled() {
        return alloc;
    }
    STATE.lock().unwrap().as_ref().and_then(|s| s.cur.get(&alloc).copied()).unwrap_or(alloc)
}

/// `list` (of type `ty`) was made recording on the PC allocator `raw` itself
/// (CreateCommandList).
pub fn created_on(list: usize, raw: usize, ty: i32) {
    if !enabled() || raw == 0 {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    s.types.entry(raw).or_insert(ty);
    move_list(s, list, raw);
    *s.recordings.entry(list).or_default() += 1;
}

/// Diagnostic (#1452): `lists` are about to be submitted to `queue`. A PC list whose previous
/// submit the GPU has not finished must not be submitted again on the PC (debug layer id 553,
/// "the command queue fence has not advanced past previous executions of the command list");
/// a console lets a title do it. Logged (the first 8, then at powers of two) with whether it is
/// the same recording and the same queue. Every submitted PC list is seen (`executed` marks them all).
pub unsafe fn note_resubmits(queue: *mut c_void, lists: &[*mut c_void]) -> usize {
    if !enabled() {
        return 0;
    }
    // A driver query may block or call back into XCL. Keep strong fence snapshots alive, but
    // never make GetCompletedValue while the allocator pool's global state is locked.
    let mut probes = Vec::with_capacity(lists.len());
    {
        let g = lock_for_submit();
        let Some(s) = g.as_ref() else { return 0 };
        for &list in lists {
            let list = list as usize;
            if let (Some(&lifetime), Some(last)) = (s.list_lifetimes.get(&list), s.last_submit.get(&list)) {
                probes.push((list, lifetime, last.clone()));
            }
        }
    }
    let incomplete: Vec<_> = probes
        .into_iter()
        .filter(|(_, _, last)| last.fence.GetCompletedValue() < last.value)
        .collect();
    if incomplete.is_empty() {
        return 0;
    }
    let mut warnings = Vec::with_capacity(incomplete.len());
    let found = {
        let mut g = lock_for_submit();
        let Some(s) = g.as_mut() else { return 0 };
        let mut found = 0;
        for (list, lifetime, observed) in incomplete {
            if s.list_lifetimes.get(&list) != Some(&lifetime)
                || s.last_submit.get(&list).is_none_or(|current| !same_submit(current, &observed))
            {
                continue;
            }
            let same_recording = s.recordings.get(&list).copied().unwrap_or(0) == observed.recording;
            let same_queue = observed.queue == queue as usize;
            found += 1;
            s.resubmits += 1;
            let n = s.resubmits;
            if n <= 8 || n.is_power_of_two() {
                warnings.push((same_recording, same_queue, n));
            }
        }
        found
    };
    for (same_recording, same_queue, n) in warnings {
        crate::xlog_warn!(
            "D3D12.X: a command list is submitted while its previous submit is still on the GPU ({}, {}; #{n} so far) – the PC does not allow it (debug layer id 553)",
            if same_recording { "the same recording" } else { "a new recording" },
            if same_queue { "the same queue" } else { "another queue" },
        );
    }
    found
}

/// Whether the GPU has not finished the last submit of the PC list `list` (#1452: it must not be
/// submitted again before that; `title_list` records the title's next recording elsewhere).
pub fn in_flight(list: usize) -> bool {
    let (lifetime, last) = {
        let g = STATE.lock().unwrap();
        let Some(s) = g.as_ref() else { return false };
        let Some(&lifetime) = s.list_lifetimes.get(&list) else { return false };
        if s.submit_pins.get(&(list, lifetime)).copied().unwrap_or(0) != 0 {
            return true;
        }
        let Some(last) = s.last_submit.get(&list).cloned() else { return false };
        (lifetime, last)
    };
    if unsafe { last.fence.GetCompletedValue() } < last.value {
        return true;
    }
    // If the address changed generation, or another submit was reserved/published while the
    // driver call ran, fail closed. The caller may spend another list/allocator but cannot reset
    // storage that newly became busy.
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return false };
    s.list_lifetimes.get(&list) != Some(&lifetime)
        || s.submit_pins.get(&(list, lifetime)).copied().unwrap_or(0) != 0
        || s.last_submit.get(&list).is_none_or(|current| !same_submit(current, &last))
}

/// Compatibility helper for tests that submit directly instead of using `SubmissionGate`.
#[cfg(test)]
pub unsafe fn executed(queue: *mut c_void, lists: &[*mut c_void]) {
    let Some(begin) = begin_submit(lists) else { return };
    let _ = begin.execution_possible().finish(queue);
}

/// Whether the GPU may still run a recording made on the title's allocator `alloc`.
pub(super) unsafe fn allocator_in_flight(alloc: usize) -> bool {
    let (mine, probes) = {
        let g = STATE.lock().unwrap();
        let Some(s) = g.as_ref() else { return false };
        namespace_probes(s, alloc)
    };
    let busy = probes.iter().any(|probe| probe.in_flight());
    let stable = {
        let g = STATE.lock().unwrap();
        let Some(s) = g.as_ref() else { return false };
        namespace_matches(s, alloc, &mine, &probes)
    };
    sweep_release_owners();
    busy || !stable
}

/// The title's allocator `alloc` reached its logical final Release. Detach its namespace now;
/// backings still used by recordings/submits retain their native owner until a later sweep.
unsafe fn forget_in(s: &mut State, alloc: usize, free: &HashSet<usize>) -> (Vec<usize>, Vec<Backing>) {
    s.types.remove(&alloc);
    s.resetting.remove(&alloc);
    let mut mine: Vec<usize> = s.aside.remove(&alloc).unwrap_or_default();
    mine.extend(s.cur.remove(&alloc));
    mine.push(alloc);
    mine.sort_unstable();
    mine.dedup();
    let mut made = Vec::new();
    let mut removed = Vec::new();
    for b in mine {
        if s.backings.contains_key(&b) && !free.contains(&b) {
            if let Some(backing) = s.backings.get_mut(&b) {
                backing.retire_when_free = true;
            }
            continue;
        }
        if let Some(mut gone) = s.backings.remove(&b) {
            for l in gone.users.drain() {
                s.of_list.remove(&l);
            }
            if gone.release_owner.is_none() && b != alloc {
                made.push(b);
            }
            removed.push(gone);
        } else if b != alloc {
            made.push(b);
        }
    }
    (made, removed)
}

pub unsafe fn forget(alloc: usize) {
    let (made, removed, free) = loop {
        let (mine, probes) = {
            let g = STATE.lock().unwrap();
            let Some(s) = g.as_ref() else {
                drop(g);
                super::console_list::allocator_reset(alloc as *mut c_void);
                return;
            };
            namespace_probes(s, alloc)
        };
        let free: HashSet<_> = probes.iter().filter(|probe| unsafe { probe.free() }).map(|probe| probe.raw).collect();
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        if namespace_matches(s, alloc, &mine, &probes) {
            let (made, removed) = forget_in(s, alloc, &free);
            break (made, removed, free);
        }
    };
    // Native backing ownership also keys console UMA allocations. Free only backings whose
    // exact generation was proved idle, and do it outside STATE because freeing can re-enter.
    for raw in free {
        super::console_list::allocator_reset(raw as *mut c_void);
    }
    for mut backing in removed {
        let owner = backing.release_owner.take();
        drop(backing);
        if let Some(owner) = owner {
            super::real::release_retained_allocator(owner);
        }
    }
    // (outside the lock: their Release comes back here)
    for b in made {
        drop(ID3D12CommandAllocator::from_raw(b as *mut c_void));
    }
}

/// Reset every idle aside backing once. The namespace token and backing probes make a driver
/// callback, final Release, or newly recorded list invalidate the whole unlocked batch.
unsafe fn reclaim_idle_aside(alloc: *mut c_void) {
    let a = alloc as usize;
    let claim = loop {
        let (mine, probes, free) = {
            let g = STATE.lock().unwrap();
            let Some(s) = g.as_ref() else { return };
            if s.resetting.contains_key(&a) { return }
            let (mine, probes) = namespace_probes(s, a);
            let free = probes.iter().filter(|probe| probe.free()).map(|probe| probe.raw).collect::<HashSet<_>>();
            (mine, probes, free)
        };
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        if !namespace_matches(s, a, &mine, &probes) { continue }
        if s.resetting.contains_key(&a) { return }
        let mut work = Vec::new();
        for raw in s.aside.get(&a).into_iter().flatten().copied() {
            let Some(probe) = probes.iter().find(|probe| probe.raw == raw && free.contains(&raw) && !probe.already_reset) else { continue };
            let raw_ptr = raw as *mut c_void;
            if let Some(owner) = ID3D12CommandAllocator::from_raw_borrowed(&raw_ptr).cloned() {
                work.push((probe.clone(), owner));
            }
        }
        if work.is_empty() { return }
        let token = reserve_reset(s, a).unwrap();
        break IdleResetClaim { alloc: a, token, mine, probes, work };
    };

    let IdleResetClaim { alloc, token, mine, probes, work } = claim;
    let results = work.into_iter().map(|(probe, owner)| {
        let ok = run_reset(owner.as_raw(), |raw| super::real::reset_pc_allocator(raw)) >= 0;
        (probe, owner, ok)
    }).collect::<Vec<_>>();
    let successful = results.iter().filter(|(_, _, ok)| *ok).map(|(probe, _, _)| probe.clone()).collect::<Vec<_>>();
    let mut retired = Vec::new();
    {
        let mut g = STATE.lock().unwrap();
        if let Some(s) = g.as_mut() {
            commit_idle_resets(s, alloc, token, &mine, &probes, &successful, &mut retired);
        }
    }
    drop(retired);
    drop(results);
}

fn commit_idle_resets(
    s: &mut State,
    alloc: usize,
    token: u64,
    mine: &[usize],
    probes: &[BackingProbe],
    successful: &[BackingProbe],
    retired: &mut Vec<ID3D12Fence>,
) -> bool {
    let valid = s.resetting.get(&alloc) == Some(&token) && namespace_matches(s, alloc, mine, probes);
    if valid {
        for probe in successful {
            if s.aside.get(&alloc).is_some_and(|aside| aside.contains(&probe.raw)) {
                if let Some(backing) = s.backings.get_mut(&probe.raw) {
                    retired.extend(backing.marks.drain().map(|(_, (fence, _))| fence));
                    backing.already_reset = true;
                }
            }
        }
    }
    cancel_reset(s, alloc, token);
    valid
}

/// Free spares owned by another title allocator of the same type. The backing snapshot lets the
/// caller ask the driver fences outside STATE, then reject a candidate whose use changed meanwhile.
fn shared_aside_probes(s: &State, alloc: usize, ty: i32) -> Vec<(usize, usize, Option<BackingProbe>)> {
    s.aside.iter().filter(|(owner, _)| {
        **owner != alloc && s.types.get(owner) == Some(&ty) && !s.resetting.contains_key(owner)
    }).flat_map(|(&owner, aside)| {
        aside.iter().copied().filter(move |&raw| raw != owner).map(move |raw| {
            (owner, raw, s.backings.get(&raw).map(|backing| BackingProbe::new(raw, backing)))
        })
    }).collect()
}

/// Transfer one still-free shared spare into `alloc`'s namespace. A donor with a reset in progress
/// is skipped: its unlocked driver call may already target the candidate.
fn take_shared_aside(
    s: &mut State,
    alloc: usize,
    ty: i32,
    candidates: &[(usize, usize, Option<BackingProbe>)],
    free: &HashSet<usize>,
) -> bool {
    for (owner, raw, observed) in candidates {
        if !free.contains(raw)
            || s.types.get(owner) != Some(&ty)
            || s.resetting.contains_key(owner)
            || s.aside.get(owner).is_none_or(|aside| !aside.contains(raw))
        {
            continue;
        }
        let stable = match (observed, s.backings.get(raw)) {
            (Some(probe), Some(backing)) => probe.matches(backing),
            (None, None) => true,
            _ => false,
        };
        if !stable {
            continue;
        }
        let aside = s.aside.get_mut(owner).unwrap();
        aside.remove(aside.iter().position(|candidate| candidate == raw).unwrap());
        s.aside.entry(alloc).or_default().push(*raw);
        return true;
    }
    false
}

/// The title resets its allocator `alloc`. `None`: reset it as it is (nothing holds a
/// recording on it, or this is off). `Some(hr)`: done here – the PC allocator in use was put
/// aside with its recordings and another one took its place.
pub unsafe fn title_reset(alloc: *mut c_void) -> Option<HRESULT> {
    if !enabled() {
        return None;
    }
    let a = alloc as usize;
    let claim = loop {
        let (mine, probes, shared) = {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            if s.resetting.contains_key(&a) {
                return Some(crate::util::E_FAIL);
            }
            let (mine, probes) = namespace_probes(s, a);
            let shared = s.types.get(&a).map(|&ty| shared_aside_probes(s, a, ty)).unwrap_or_default();
            (mine, probes, shared)
        };
        let mut free: HashSet<_> = probes.iter().filter(|probe| unsafe { probe.free() }).map(|probe| probe.raw).collect();
        free.extend(shared.iter().filter_map(|(_, raw, probe)| {
            probe.as_ref().is_none_or(|probe| unsafe { probe.free() }).then_some(*raw)
        }));
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return None };
        if !namespace_matches(s, a, &mine, &probes) {
            continue;
        }
        if s.resetting.contains_key(&a) {
            return Some(crate::util::E_FAIL);
        }
        let current = s.cur.get(&a).copied().unwrap_or(a);
        if !s.backings.contains_key(&current) || free.contains(&current) {
            if current == a {
                let retired = s.backings.get_mut(&current)
                    .map(|backing| backing.marks.drain().map(|(_, (fence, _))| fence).collect::<Vec<_>>())
                    .unwrap_or_default();
                drop(g);
                drop(retired);
                drop(probes);
                reclaim_idle_aside(alloc);
                return None;
            }
            let raw = current as *mut c_void;
            let Some(owner) = ID3D12CommandAllocator::from_raw_borrowed(&raw).cloned() else { return Some(crate::util::E_FAIL) };
            let token = reserve_reset(s, a).unwrap();
            let needed = s.backings.get(&current).is_none_or(|backing| !backing.already_reset);
            break ResetClaim { alloc: a, current, token, mine, probes, work: ResetWork::Reset { owner, swap: false, needed } };
        }
        let Some(&ty) = s.types.get(&a) else { return Some(crate::util::E_FAIL) };
        if let Some(candidate) = s.aside.get(&a).and_then(|values| values.iter().copied().find(|raw| !s.backings.contains_key(raw) || free.contains(raw))) {
            let raw = candidate as *mut c_void;
            let Some(owner) = ID3D12CommandAllocator::from_raw_borrowed(&raw).cloned() else { return Some(crate::util::E_FAIL) };
            let token = reserve_reset(s, a).unwrap();
            let needed = s.backings.get(&candidate).is_none_or(|backing| !backing.already_reset);
            break ResetClaim { alloc: a, current, token, mine, probes, work: ResetWork::Reset { owner, swap: true, needed } };
        }
        if take_shared_aside(s, a, ty, &shared, &free) {
            // Restart so the transferred candidate participates in this namespace's snapshot and
            // reset token exactly like a spare it had always owned.
            continue;
        }
        let Some(owner) = ID3D12CommandAllocator::from_raw_borrowed(&alloc).cloned() else { return Some(crate::util::E_FAIL) };
        let token = reserve_reset(s, a).unwrap();
        break ResetClaim { alloc: a, current, token, mine, probes, work: ResetWork::Create { owner, ty } };
    };

    let ResetClaim { alloc, current, token, mine, probes, work } = claim;
    match work {
        ResetWork::Reset { owner, swap, needed } => {
            let candidate = owner.as_raw() as usize;
            let hr = if needed { run_reset(owner.as_raw(), |raw| super::real::reset_pc_allocator(raw)) } else { 0 };
            let mut retired = Vec::new();
            let result = {
                let mut g = STATE.lock().unwrap();
                let Some(s) = g.as_mut() else { return Some(crate::util::E_FAIL) };
                if hr >= 0 && commit_reset(s, alloc, current, candidate, token, &mine, &probes, swap, false, &mut retired) {
                    hr
                } else {
                    cancel_reset(s, alloc, token);
                    if hr < 0 { hr } else { crate::util::E_FAIL }
                }
            };
            drop(retired);
            drop(owner);
            if result >= 0 {
                reclaim_idle_aside(alloc as *mut c_void);
            }
            Some(result)
        }
        ResetWork::Create { owner, ty } => {
            let mut dev: Option<ID3D12Device> = None;
            let made: Result<ID3D12CommandAllocator, HRESULT> = match owner.GetDevice(&mut dev) {
                Err(error) => Err(error.code().0),
                Ok(()) => match dev {
                    Some(dev) => dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE(ty)).map_err(|error| error.code().0),
                    None => Err(crate::util::E_FAIL),
                },
            };
            let made = match made {
                Ok(made) => made,
                Err(error) => {
                    let mut g = STATE.lock().unwrap();
                    if let Some(s) = g.as_mut() { cancel_reset(s, alloc, token); }
                    return Some(error);
                }
            };
            let raw = made.into_raw();
            if !super::real::is_adopted(raw) {
                super::real::adopt(raw, super::real::Kind::CommandAllocator);
            }
            let mut retired = Vec::new();
            let committed = {
                let mut g = STATE.lock().unwrap();
                g.as_mut().is_some_and(|s| commit_reset(s, alloc, current, raw as usize, token, &mine, &probes, true, true, &mut retired))
            };
            drop(retired);
            drop(owner);
            if committed {
                reclaim_idle_aside(alloc as *mut c_void);
                Some(0)
            } else {
                drop(ID3D12CommandAllocator::from_raw(raw));
                Some(crate::util::E_FAIL)
            }
        }
    }
}

fn commit_reset(
    s: &mut State,
    alloc: usize,
    current: usize,
    candidate: usize,
    token: u64,
    mine: &[usize],
    probes: &[BackingProbe],
    swap: bool,
    created: bool,
    retired: &mut Vec<ID3D12Fence>,
) -> bool {
    if s.resetting.get(&alloc) != Some(&token) || !namespace_matches(s, alloc, mine, probes) {
        cancel_reset(s, alloc, token);
        return false;
    }
    if swap && !created {
        let Some(position) = s.aside.get(&alloc).and_then(|values| values.iter().position(|raw| *raw == candidate)) else {
            cancel_reset(s, alloc, token);
            return false;
        };
        s.aside.get_mut(&alloc).unwrap().remove(position);
    }
    if let Some(backing) = s.backings.get_mut(&candidate) {
        retired.extend(backing.marks.drain().map(|(_, (fence, _))| fence));
        backing.already_reset = true;
    }
    if !swap {
        cancel_reset(s, alloc, token);
        return true;
    }
    s.aside.entry(alloc).or_default().push(current);
    s.cur.insert(alloc, candidate);
    if created { s.made += 1; }
    s.swaps += 1;
    cancel_reset(s, alloc, token);
    if s.swaps.is_power_of_two() {
        let aside: usize = s.aside.values().map(|v| v.len()).sum();
        crate::xlog_info!(
            "D3D12.X: {} allocator resets so far with recordings still held or on the GPU – another PC allocator took its place ({} made, {aside} put aside now)",
            s.swaps,
            s.made
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// STATE is process-wide and other tests running in parallel take it briefly, so a single
    /// `try_lock` can fail without the caller holding it. Retry until a generous deadline: when
    /// this thread holds STATE, `try_lock` keeps returning `WouldBlock` and this returns false.
    fn state_not_held_by_caller() -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if STATE.try_lock().is_ok() { return true; }
            if std::time::Instant::now() >= deadline { return false; }
            std::thread::yield_now();
        }
    }

    #[test]
    fn release_probe_transfers_only_the_logical_final_reference() {
        use std::cell::Cell;
        let refs = Cell::new(1u32);
        let retained = Cell::new(0u32);
        let (remaining, transferred) = release_with_probe(
            || refs.set(refs.get() + 1),
            || {
                refs.set(refs.get() - 1);
                refs.get()
            },
            || {
                retained.set(retained.get() + 1);
                true
            },
        );
        assert_eq!((remaining, transferred, refs.get(), retained.get()), (0, true, 1, 1));
        refs.set(refs.get() - 1); // the out-of-lock sweep releases the transferred probe
        assert_eq!(refs.get(), 0);

        refs.set(2);
        retained.set(0);
        let (remaining, transferred) = release_with_probe(
            || refs.set(refs.get() + 1),
            || {
                refs.set(refs.get() - 1);
                refs.get()
            },
            || {
                retained.set(retained.get() + 1);
                true
            },
        );
        assert_eq!((remaining, transferred, refs.get(), retained.get()), (1, false, 1, 0));
    }

    #[test]
    fn release_probe_retries_a_same_address_backing_replacement() {
        use std::mem::{offset_of, size_of};
        use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
        static REPLACE: AtomicBool = AtomicBool::new(false);
        static KEY: AtomicUsize = AtomicUsize::new(0);
        #[repr(C)]
        struct Object { vtable: *const usize, value: AtomicU64, refs: AtomicU32 }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            if REPLACE.swap(false, Ordering::Relaxed) {
                let gone = {
                    let mut g = STATE.lock().unwrap();
                    let s = g.as_mut().unwrap();
                    let key = KEY.load(Ordering::Relaxed);
                    let gone = s.backings.remove(&key).unwrap();
                    backing_mut(s, key);
                    gone
                };
                drop(gone);
            }
            (*this.cast::<Object>()).value.load(Ordering::Acquire)
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            let o = &*this.cast::<Object>();
            o.refs.fetch_add(1, Ordering::AcqRel) + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let o = &*this.cast::<Object>();
            o.refs.fetch_sub(1, Ordering::AcqRel) - 1
        }
        extern "system" fn unexpected_call() { std::process::abort(); }
        const SLOT: usize = size_of::<usize>();
        let fence_vtable = Box::leak(Box::new([unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT]));
        fence_vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        fence_vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.AddRef) / SLOT] = add_ref as usize;
        fence_vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let allocator_vtable = Box::leak(Box::new([unexpected_call as usize, add_ref as usize, release as usize]));
        let fence_object = Box::leak(Box::new(Object {
            vtable: fence_vtable.as_ptr(), value: AtomicU64::new(1), refs: AtomicU32::new(1),
        }));
        let allocator_object = Box::leak(Box::new(Object {
            vtable: allocator_vtable.as_ptr(), value: AtomicU64::new(0), refs: AtomicU32::new(2),
        }));
        let fence = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut *fence_object).cast()) };
        let root = unsafe { ID3D12CommandAllocator::from_raw(std::ptr::from_mut(&mut *allocator_object).cast()) };
        let caller_probe = unsafe { ID3D12CommandAllocator::from_raw(root.as_raw()) };
        let key = root.as_raw() as usize;
        let first_generation = {
            let mut g = STATE.lock().unwrap();
            let backing = backing_mut(g.get_or_insert_with(State::default), key);
            backing.marks.insert(7, (fence.clone(), 2));
            backing.generation
        };
        KEY.store(key, Ordering::Relaxed);
        REPLACE.store(true, Ordering::Relaxed);
        let retained = unsafe { retain_release_probe(caller_probe.as_raw()) };
        if retained { std::mem::forget(caller_probe) } else { drop(caller_probe) }
        let replacement = {
            let mut g = STATE.lock().unwrap();
            g.as_mut().unwrap().backings.remove(&key).unwrap()
        };
        assert!(!replacement.release_owner.is_some());
        assert_ne!(replacement.generation, first_generation);
        drop(replacement);
        drop(root);
        drop(fence);
        assert!(!retained, "the replacement must be probed before accepting the caller's reference");
        assert_eq!(
            (allocator_object.refs.load(Ordering::Acquire), fence_object.refs.load(Ordering::Acquire)),
            (0, 0),
        );
    }

    #[test]
    fn completion_queries_run_outside_allocator_state() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        static OUTSIDE: AtomicBool = AtomicBool::new(false);
        static MUTATE: AtomicBool = AtomicBool::new(false);
        static MUTATE_BACKING: AtomicUsize = AtomicUsize::new(0);
        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: Cell<u64>,
            refs: Cell<u32>,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            let outside = state_not_held_by_caller();
            OUTSIDE.store(outside, Ordering::Relaxed);
            if outside && MUTATE.swap(false, Ordering::Relaxed) {
                let mut g = STATE.lock().unwrap();
                if let Some(backing) = g
                    .as_mut()
                    .and_then(|s| s.backings.get_mut(&MUTATE_BACKING.load(Ordering::Relaxed)))
                {
                    backing.submit_pins += 1;
                }
            }
            (*this.cast::<Fence>()).value.get()
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.refs.set(f.refs.get() + 1);
            f.refs.get()
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.refs.set(f.refs.get() - 1);
            f.refs.get()
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let mut vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.AddRef) / SLOT] = add_ref as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let mut object = Fence { vtable: vtable.as_ptr(), value: Cell::new(0), refs: Cell::new(1) };
        let fence = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut object).cast()) };
        let (list, lifetime) = (0x7ffe_6a01usize, 17u64);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.list_lifetimes.insert(list, lifetime);
            s.recordings.insert(list, 3);
            s.last_submit.insert(list, LastSubmit { queue: 0x81, fence: fence.clone(), value: 1, recording: 3 });
        }

        OUTSIDE.store(false, Ordering::Relaxed);
        assert_eq!(unsafe { note_resubmits(0x81usize as *mut c_void, &[list as *mut c_void]) }, 1);
        assert!(OUTSIDE.load(Ordering::Relaxed), "resubmit polling must not hold STATE");
        OUTSIDE.store(false, Ordering::Relaxed);
        assert!(in_flight(list));
        assert!(OUTSIDE.load(Ordering::Relaxed), "in-flight polling must not hold STATE");
        object.value.set(1);
        OUTSIDE.store(false, Ordering::Relaxed);
        assert!(!in_flight(list));
        assert!(OUTSIDE.load(Ordering::Relaxed), "completed polling must not hold STATE");

        let alloc = 0x7ffe_6a02usize;
        {
            let mut g = STATE.lock().unwrap();
            backing_mut(g.as_mut().unwrap(), alloc).marks.insert(0x81, (fence.clone(), 2));
        }
        OUTSIDE.store(false, Ordering::Relaxed);
        assert!(unsafe { allocator_in_flight(alloc) });
        assert!(OUTSIDE.load(Ordering::Relaxed), "allocator polling must not hold STATE");

        object.value.set(2);
        MUTATE_BACKING.store(alloc, Ordering::Relaxed);
        MUTATE.store(true, Ordering::Relaxed);
        OUTSIDE.store(false, Ordering::Relaxed);
        assert!(
            unsafe { allocator_in_flight(alloc) },
            "a state change during the unlocked probe must fail closed",
        );
        assert!(OUTSIDE.load(Ordering::Relaxed), "allocator callback must be able to reenter STATE");

        let backing = {
            let mut g = STATE.lock().unwrap();
            g.as_mut().unwrap().backings.remove(&alloc).unwrap()
        };
        drop(backing);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            s.last_submit.remove(&list);
            s.recordings.remove(&list);
            s.list_lifetimes.remove(&list);
        }
        drop(fence);
        assert_eq!(object.refs.get(), 0);
    }

    #[test]
    fn submit_marking_calls_out_unlocked_and_revalidates_the_queue_marker() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            refs: Cell<u32>,
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.refs.set(f.refs.get() + 1);
            f.refs.get()
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.refs.set(f.refs.get() - 1);
            f.refs.get()
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let mut vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.AddRef) / SLOT] = add_ref as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let mut first_object = Fence { vtable: vtable.as_ptr(), refs: Cell::new(1) };
        let mut old_object = Fence { vtable: vtable.as_ptr(), refs: Cell::new(1) };
        let mut new_object = Fence { vtable: vtable.as_ptr(), refs: Cell::new(1) };
        let first_fence = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut first_object).cast()) };
        let old_fence = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut old_object).cast()) };
        let new_fence = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut new_object).cast()) };

        let (first_list, first_queue) = (0x7ffe_6b01usize, 0x7ffe_6b02usize);
        let pending = begin_submit(&[first_list as *mut c_void]).unwrap().execution_possible();
        assert!(pending.finish_with_queue_ops(
            first_queue,
            || {
                assert!(state_not_held_by_caller(), "fence creation must run outside STATE");
                Some(first_fence.clone())
            },
            |_, value| {
                assert!(state_not_held_by_caller(), "queue Signal must run outside STATE");
                assert_eq!(value, 1);
                true
            },
        ));
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.last_submit[&first_list].value, 1);
            assert!(!s.submit_pins.contains_key(&(first_list, s.list_lifetimes[&first_list])));
        }

        let failed_list = 0x7ffe_6b05usize;
        let failed = begin_submit(&[failed_list as *mut c_void]).unwrap().execution_possible();
        assert!(!failed.finish_with_queue_ops(first_queue, || unreachable!(), |_, value| {
            assert_eq!(value, 2);
            false
        }));
        let retried = begin_submit(&[failed_list as *mut c_void]).unwrap().execution_possible();
        assert!(retried.finish_with_queue_ops(first_queue, || unreachable!(), |_, value| {
            assert_eq!(value, 3, "a failed Signal leaves a gap instead of colliding with a retry");
            true
        }));
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            let lifetime = s.list_lifetimes[&failed_list];
            assert_eq!(s.submit_pins.get(&(failed_list, lifetime)), Some(&1), "the failed submit stays pinned");
            assert_eq!(s.last_submit[&failed_list].value, 3);
        }

        let (outer_list, inner_list) = (0x7ffe_6b06usize, 0x7ffe_6b07usize);
        let outer = begin_submit(&[outer_list as *mut c_void]).unwrap().execution_possible();
        let inner = begin_submit(&[inner_list as *mut c_void]).unwrap().execution_possible();
        assert!(super::super::queue_submit::with(first_queue, || {
            outer.finish_with_queue_ops(first_queue, || unreachable!(), |_, outer_value| {
                assert_eq!(outer_value, 4);
                assert!(super::super::queue_submit::with(first_queue, || {
                    inner.finish_with_queue_ops(first_queue, || unreachable!(), |_, inner_value| {
                        assert_eq!(inner_value, 5, "same-thread re-entry reserves a distinct value");
                        true
                    })
                }));
                true
            })
        }));
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!((s.last_submit[&outer_list].value, s.last_submit[&inner_list].value), (4, 5));
        }

        let (reused_list, reused_queue) = (0x7ffe_6b03usize, 0x7ffe_6b04usize);
        let pending = begin_submit(&[reused_list as *mut c_void]).unwrap().execution_possible();
        assert!(!pending.finish_with_queue_ops(
            reused_queue,
            || Some(old_fence.clone()),
            |_, value| {
                assert!(state_not_held_by_caller(), "queue invalidation can run while Signal is outside STATE");
                assert_eq!(value, 1);
                STATE.lock().unwrap().as_mut().unwrap().fences.insert(
                    reused_queue,
                    Arc::new(QueueFence::new(new_fence.clone())),
                );
                true
            },
        ), "a replacement raw-address generation must not receive the old submit's mark");
        let (markers, submits) = {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(!s.last_submit.contains_key(&reused_list));
            let lifetime = s.list_lifetimes[&reused_list];
            assert_eq!(s.submit_pins.get(&(reused_list, lifetime)), Some(&1));
            assert_eq!(s.fences[&reused_queue].fence.as_raw(), new_fence.as_raw());
            s.submit_pins.remove(&(reused_list, lifetime));
            s.list_lifetimes.remove(&reused_list);
            let markers = [s.fences.remove(&first_queue), s.fences.remove(&reused_queue)];
            let submits = [
                s.last_submit.remove(&first_list),
                s.last_submit.remove(&failed_list),
                s.last_submit.remove(&outer_list),
                s.last_submit.remove(&inner_list),
            ];
            let failed_lifetime = s.list_lifetimes.remove(&failed_list).unwrap();
            s.submit_pins.remove(&(failed_list, failed_lifetime));
            s.list_lifetimes.remove(&first_list);
            s.list_lifetimes.remove(&outer_list);
            s.list_lifetimes.remove(&inner_list);
            (markers, submits)
        };
        drop(markers);
        drop(submits);
        drop(first_fence);
        drop(old_fence);
        drop(new_fence);
        assert_eq!((first_object.refs.get(), old_object.refs.get(), new_object.refs.get()), (0, 0, 0));
    }

    #[test]
    fn submission_receipt_keeps_exact_marker_and_failure_is_untracked() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: Cell<u64>,
            refs: Cell<u32>,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            (*this.cast::<Fence>()).value.get()
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.refs.set(f.refs.get() + 1);
            f.refs.get()
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.refs.set(f.refs.get() - 1);
            f.refs.get()
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let mut vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.AddRef) / SLOT] = add_ref as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let mut old_object = Fence { vtable: vtable.as_ptr(), value: Cell::new(0), refs: Cell::new(1) };
        let mut new_object = Fence { vtable: vtable.as_ptr(), value: Cell::new(1), refs: Cell::new(1) };
        let old_fence = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut old_object).cast()) };
        let new_fence = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut new_object).cast()) };
        let (list, second_list, failed_list, queue) =
            (0x7ffe_6c01usize, 0x7ffe_6c02usize, 0x7ffe_6c03usize, 0x7ffe_6c04usize);

        let receipt = begin_submit(&[list as *mut c_void])
            .unwrap()
            .execution_possible()
            .finish_with_queue_ops_receipted(queue, || Some(old_fence.clone()), |_, value| value == 1)
            .expect("the successful Signal has an exact receipt");
        assert!(!receipt.is_complete());
        let receipt_refs = old_object.refs.get();
        let receipt_clone = receipt.clone();
        assert_eq!(old_object.refs.get(), receipt_refs, "receipt clone called COM AddRef");
        drop(receipt_clone);
        assert_eq!(old_object.refs.get(), receipt_refs, "receipt clone called COM Release");
        let second_receipt = begin_submit(&[second_list as *mut c_void])
            .unwrap()
            .execution_possible()
            .finish_with_queue_ops_receipted(queue, || unreachable!(), |_, value| value == 2)
            .expect("the same queue marker makes another receipt");
        assert_eq!(receipt.fence_identity(), second_receipt.fence_identity());

        let old_marker = STATE.lock().unwrap().as_mut().unwrap().fences.insert(
            queue,
            Arc::new(QueueFence::new(new_fence.clone())),
        );
        drop(old_marker);
        assert!(!receipt.is_complete(), "a complete replacement marker cannot satisfy an old receipt");
        assert!(!second_receipt.is_complete());
        old_object.value.set(2);
        assert!(receipt.is_complete());
        assert!(second_receipt.is_complete());

        let failed = begin_submit(&[failed_list as *mut c_void]).unwrap().execution_possible();
        assert!(failed
            .finish_with_queue_ops_receipted(queue, || unreachable!(), |_, value| {
                assert_eq!(value, 1);
                false
            })
            .is_none());
        let (marker, submit, second_submit) = {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            let lifetime = s.list_lifetimes[&failed_list];
            assert_eq!(s.submit_pins.get(&(failed_list, lifetime)), Some(&1), "untracked execution remains pinned");
            s.submit_pins.remove(&(failed_list, lifetime));
            s.list_lifetimes.remove(&failed_list);
            s.list_lifetimes.remove(&list);
            s.list_lifetimes.remove(&second_list);
            (s.fences.remove(&queue), s.last_submit.remove(&list), s.last_submit.remove(&second_list))
        };
        drop(marker);
        drop(submit);
        drop(second_submit);
        drop(receipt);
        drop(second_receipt);
        drop(old_fence);
        drop(new_fence);
        assert_eq!((old_object.refs.get(), new_object.refs.get()), (0, 0));
    }

    #[test]
    fn released_namespace_keeps_busy_made_backing_until_retirement() {
        let (alloc, made) = (0x7ffe_6100usize, 0x7ffe_6200usize);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.cur.insert(alloc, made);
            s.backings.insert(made, Backing { submit_pins: 1, ..Default::default() });
        }
        assert!(unsafe { allocator_in_flight(alloc) });
        unsafe { forget(alloc) };
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(s.backings[&made].retire_when_free);
            assert_eq!(s.backings[&made].submit_pins, 1);
            assert!(!s.cur.contains_key(&alloc));
            s.backings.remove(&made); // synthetic raw has no COM reference to sweep
        }
    }

    #[test]
    fn absent_free_made_backing_is_scheduled_for_release_once() {
        let (alloc, made) = (0x7ffe_6300usize, 0x7ffe_6400usize);
        let mut state = State::default();
        state.cur.insert(alloc, made);
        state.aside.insert(alloc, vec![made, alloc, made]);
        let (release, owners) = unsafe { forget_in(&mut state, alloc, &HashSet::new()) };
        assert_eq!(release, vec![made]);
        assert!(owners.is_empty());
        assert!(!state.cur.contains_key(&alloc) && !state.aside.contains_key(&alloc));
    }

    #[test]
    fn transferred_owner_is_released_once_outside_state_after_logical_forget() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

        #[repr(C)]
        struct FakeAllocator {
            vtable: *const usize,
            refs: AtomicU32,
            releases: AtomicU32,
            release_outside_state: AtomicBool,
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            let object = &*this.cast::<FakeAllocator>();
            object.refs.fetch_add(1, Ordering::AcqRel) + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let object = &*this.cast::<FakeAllocator>();
            for _ in 0..10_000 {
                if STATE.try_lock().is_ok() {
                    object.release_outside_state.store(true, Ordering::Release);
                    break;
                }
                std::thread::yield_now();
            }
            object.releases.fetch_add(1, Ordering::AcqRel);
            object.refs.fetch_sub(1, Ordering::AcqRel) - 1
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        let vtable = Box::leak(Box::new([unexpected_call as usize, add_ref as usize, release as usize]));
        let object = Box::leak(Box::new(FakeAllocator {
            vtable: vtable.as_ptr(),
            refs: AtomicU32::new(1),
            releases: AtomicU32::new(0),
            release_outside_state: AtomicBool::new(false),
        }));
        let raw = std::ptr::from_mut(&mut *object).cast::<c_void>();
        let owner = unsafe { ID3D12CommandAllocator::from_raw(raw) };
        {
            let mut g = STATE.lock().unwrap();
            g.get_or_insert_with(State::default).backings.insert(
                raw as usize,
                Backing { submit_pins: 1, release_owner: Some(owner), ..Default::default() },
            );
        }
        unsafe { forget(raw as usize) };
        {
            let mut g = STATE.lock().unwrap();
            let backing = g.as_mut().unwrap().backings.get_mut(&(raw as usize)).unwrap();
            assert!(backing.retire_when_free);
            backing.submit_pins = 0;
        }
        sweep_release_owners_with(drop, drop);
        // Tests running in parallel sweep the same STATE (e.g. through `forget`), so another thread
        // can take this retired backing first and release it just after our sweep returns.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while object.releases.load(Ordering::Acquire) == 0 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(
            (object.refs.load(Ordering::Acquire), object.releases.load(Ordering::Acquire)),
            (0, 1),
        );
        assert!(object.release_outside_state.load(Ordering::Acquire));
        assert!(!STATE.lock().unwrap().as_ref().unwrap().backings.contains_key(&(raw as usize)));
    }

    #[test]
    fn execute_topology_counts_only_split_boundaries() {
        assert_eq!(execute_topology(1), (0, 0));
        assert_eq!(execute_topology(4), (1, 3));
    }

    #[test]
    fn reserved_submit_orders_prepare_arm_execute_and_finish() {
        use std::cell::RefCell;
        let calls = RefCell::new(Vec::new());
        let mut failed = false;
        let submitted = reserve_and_execute_with(
            &mut failed,
            || {
                calls.borrow_mut().push("begin");
                Some(1)
            },
            || calls.borrow_mut().push("prepare"),
            |begin| {
                calls.borrow_mut().push("arm");
                begin + 1
            },
            || calls.borrow_mut().push("execute"),
            |pending| {
                calls.borrow_mut().push("finish-failed");
                assert_eq!(pending, 2);
                false
            },
        );
        assert!(submitted && !failed, "marker failure cannot undo a native execution");
        assert_eq!(*calls.borrow(), ["begin", "prepare", "arm", "execute", "finish-failed"]);
    }

    #[test]
    fn begin_failure_stops_later_chunks_and_publication() {
        use std::cell::Cell;
        let mut failed = false;
        let executed = Cell::new(0);
        let published = Cell::new(0);
        for begin_succeeds in [true, false, true] {
            let submitted = reserve_and_execute_with(
                &mut failed,
                || begin_succeeds.then_some(()),
                || {},
                |()| (),
                || executed.set(executed.get() + 1),
                |_| true,
            );
            if submitted {
                published.set(published.get() + 1);
            }
        }
        assert!(failed);
        assert_eq!((executed.get(), published.get()), (1, 1));
    }

    #[test]
    fn submit_reservation_keeps_pre_reset_attribution_across_queues() {
        use std::mem::{offset_of, size_of};
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: AtomicU64,
            refs: AtomicU32,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            (*this.cast::<Fence>()).value.load(Ordering::Acquire)
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            let fence = &*this.cast::<Fence>();
            fence.refs.fetch_add(1, Ordering::AcqRel) + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let fence = &*this.cast::<Fence>();
            fence.refs.fetch_sub(1, Ordering::AcqRel) - 1
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let vtable = Box::leak(Box::new([unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT]));
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.AddRef) / SLOT] = add_ref as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let fence_object = Box::leak(Box::new(Fence {
            vtable: vtable.as_ptr(),
            value: AtomicU64::new(0),
            refs: AtomicU32::new(1),
        }));
        let fence = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut *fence_object).cast()) };

        let (alloc, next, list) = (0x7ffe_5100usize, 0x7ffe_5200usize, 0x7ffe_5300usize);
        created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let first = begin_submit(&[list as *mut c_void, list as *mut c_void]).unwrap().execution_possible();
        let cross_queue = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();
        assert!(in_flight(list));
        let reset = std::thread::spawn(move || {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            s.aside.insert(alloc, vec![next]);
            s.backings.insert(next, Backing::default());
            let (mine, probes) = namespace_probes(s, alloc);
            let token = reserve_reset(s, alloc).unwrap();
            let mut retired = Vec::new();
            commit_reset(s, alloc, alloc, next, token, &mine, &probes, true, false, &mut retired)
        })
        .join()
        .unwrap();
        assert!(reset, "the reserved backing is put aside instead of Reset");
        assert_eq!(reset_list(list, alloc, -1, |backing| {
            assert_eq!(backing, next);
            0
        }), 0);
        let second = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();

        first.finish_with_mark(0x71, &fence, 3);
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.backings[&alloc].marks.get(&0x71).unwrap().1, 3);
            assert!(!s.backings[&next].marks.contains_key(&0x71));
            assert_eq!(s.submit_pins.get(&(list, s.list_lifetimes[&list])), Some(&2));
            assert_eq!(s.last_submit[&list].recording, 1);
        }
        cross_queue.finish_with_mark(0x72, &fence, 4);
        second.finish_with_mark(0x73, &fence, 5);
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.backings[&alloc].marks.get(&0x71).unwrap().1, 3);
            assert_eq!(s.backings[&alloc].marks.get(&0x72).unwrap().1, 4);
            assert_eq!(s.backings[&next].marks.get(&0x73).unwrap().1, 5);
            assert!(!s.submit_pins.contains_key(&(list, s.list_lifetimes[&list])));
            assert_eq!(s.last_submit[&list].recording, 2);
        }
        let released = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();
        forget_list(list);
        created_on(list, next, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let reused_address = begin_submit(&[list as *mut c_void]).unwrap();
        released.finish_with_mark(0x74, &fence, 6);
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.backings[&next].marks.get(&0x74).unwrap().1, 6);
            assert!(!s.last_submit.contains_key(&list), "a late mark cannot resurrect a released list address");
            assert_eq!(s.submit_pins.get(&(list, s.list_lifetimes[&list])), Some(&1));
        }
        drop(reused_address);
        forget_list(list);
        fence_object.value.store(6, Ordering::Release);
        let retired = {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            s.cur.remove(&alloc);
            s.aside.remove(&alloc);
            s.types.remove(&alloc);
            [s.backings.remove(&alloc), s.backings.remove(&next)]
        };
        drop(retired);
        drop(fence);
        assert_eq!(fence_object.refs.load(Ordering::Acquire), 0);
    }

    #[test]
    fn submit_reservation_drop_and_failure_are_fail_closed() {
        let (alloc, list) = (0x7ffe_5400usize, 0x7ffe_5500usize);
        created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);

        let before_execute = begin_submit(&[list as *mut c_void, list as *mut c_void]).unwrap();
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.submit_pins.get(&(list, s.list_lifetimes[&list])), Some(&1), "duplicates reserve once");
            assert_eq!(s.backings[&alloc].submit_pins, 1);
        }
        drop(before_execute);
        assert!(!in_flight(list), "a reservation abandoned before Execute is released");

        let armed_for_execute = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();
        drop(armed_for_execute);
        assert!(in_flight(list), "an armed execution stays pinned without a completion mark");
        let failed_finish = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();
        assert!(!unsafe { failed_finish.finish(std::ptr::null_mut()) });
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.submit_pins.get(&(list, s.list_lifetimes[&list])), Some(&2));
            assert_eq!(s.backings[&alloc].submit_pins, 2);
        }

        // Final list cleanup removes its recording hold but cannot make the backing reusable
        // after execution became possible without a completion mark.
        forget_list(list);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(!s.backings[&alloc].users.contains(&list));
            assert!(!s.submit_pins.keys().any(|&(p, _)| p == list), "a released address is not pinned against a later object");
            assert!(!unsafe { backing_free_for_test(&s.backings[&alloc]) });
            s.backings.get_mut(&alloc).unwrap().submit_pins = 0;
        }
        unsafe { forget(alloc) };
    }

    #[test]
    fn reset_reservation_runs_unlocked_and_failed_attempt_keeps_the_spare() {
        let (alloc, spare, list) = (0x1000usize, 0x2000usize, 0x3000usize);
        let mut s = State::default();
        s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        move_list(&mut s, list, alloc);
        s.aside.insert(alloc, vec![spare]);
        s.backings.insert(spare, Backing::default());
        let (mine, probes) = namespace_probes(&s, alloc);
        let token = reserve_reset(&mut s, alloc).unwrap();
        let hr = run_reset(spare as *mut c_void, |raw| {
            assert_eq!(raw as usize, spare);
            crate::util::E_FAIL
        });
        assert_eq!(hr, crate::util::E_FAIL);
        cancel_reset(&mut s, alloc, token);
        assert_eq!(s.aside.get(&alloc), Some(&vec![spare]), "the failed candidate is still owned and reusable");
        assert!(!s.cur.contains_key(&alloc), "the current backing did not change");

        let token = reserve_reset(&mut s, alloc).unwrap();
        let mut retired = Vec::new();
        assert!(commit_reset(&mut s, alloc, alloc, spare, token, &mine, &probes, true, false, &mut retired));
        assert_eq!(s.cur.get(&alloc), Some(&spare));
        assert_eq!(s.aside.get(&alloc), Some(&vec![alloc]), "only the successful reset switches backings");
        assert_eq!(s.of_list.get(&list), Some(&alloc), "the earlier recording stays on its original backing");
        assert_eq!((s.made, s.swaps), (0, 1));
    }

    #[test]
    fn a_free_spare_from_another_allocator_is_transferred_safely() {
        let (requester, donor, busy_owner, other_owner) = (0x1100usize, 0x2200usize, 0x3300usize, 0x4400usize);
        let (spare, busy, other, list) = (0x2210usize, 0x3310usize, 0x4410usize, 0x9900usize);
        let mut s = State::default();
        for alloc in [requester, donor, busy_owner] {
            s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        }
        s.types.insert(other_owner, D3D12_COMMAND_LIST_TYPE_COMPUTE.0);
        s.aside.insert(donor, vec![donor, spare]);
        s.aside.insert(busy_owner, vec![busy]);
        s.aside.insert(other_owner, vec![other]);
        backing_mut(&mut s, spare);
        move_list(&mut s, list, busy);

        let candidates = shared_aside_probes(&s, requester, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        assert!(candidates.iter().any(|&(owner, raw, _)| owner == donor && raw == spare));
        assert!(!candidates.iter().any(|&(_, raw, _)| raw == donor || raw == other));
        let free = candidates.iter().filter_map(|(_, raw, probe)| {
            probe.as_ref().is_none_or(|probe| unsafe { probe.free() }).then_some(*raw)
        }).collect();
        assert!(take_shared_aside(&mut s, requester, D3D12_COMMAND_LIST_TYPE_DIRECT.0, &candidates, &free));
        assert_eq!(s.aside.get(&donor), Some(&vec![donor]), "the donor's own allocator stays in its namespace");
        assert_eq!(s.aside.get(&requester), Some(&vec![spare]));

        // A snapshot that was free cannot transfer a backing after a recording starts using it.
        let raced = 0x2220usize;
        s.aside.get_mut(&donor).unwrap().push(raced);
        backing_mut(&mut s, raced);
        let candidates = shared_aside_probes(&s, requester, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let free = HashSet::from([raced]);
        move_list(&mut s, list + 1, raced);
        assert!(!take_shared_aside(&mut s, requester, D3D12_COMMAND_LIST_TYPE_DIRECT.0, &candidates, &free));
        assert!(s.aside[&donor].contains(&raced));

        // Nor can a candidate be taken while its donor may already be resetting it unlocked.
        let token = reserve_reset(&mut s, donor).unwrap();
        assert!(shared_aside_probes(&s, requester, D3D12_COMMAND_LIST_TYPE_DIRECT.0)
            .iter().all(|(owner, _, _)| *owner != donor));
        cancel_reset(&mut s, donor, token);
    }

    #[test]
    fn retired_recording_keeps_lifetime_and_submit_pin_until_exact_reuse() {
        let (list, old, next) = (0x7ffe_7601usize, 0x7ffe_7602usize, 0x7ffe_7603usize);
        let mut s = State::default();
        move_list(&mut s, list, old);
        s.recordings.insert(list, 4);
        s.list_lifetimes.insert(list, 9);
        s.submit_pins.insert((list, 9), 1);
        s.backings.get_mut(&old).unwrap().submit_pins = 1;
        let old_generation = s.backings[&old].generation;

        retire_recording_in(&mut s, list);
        assert!(!s.of_list.contains_key(&list));
        assert!(!s.backings[&old].users.contains(&list));
        assert_eq!(s.recordings.get(&list), Some(&4));
        assert_eq!(s.list_lifetimes.get(&list), Some(&9));
        assert_eq!(s.submit_pins.get(&(list, 9)), Some(&1));
        assert_eq!(s.backings[&old].submit_pins, 1, "an in-flight submit still owns the old backing");

        move_list(&mut s, list, next);
        *s.recordings.get_mut(&list).unwrap() += 1;
        assert_eq!(s.of_list.get(&list), Some(&next));
        assert!(s.backings[&next].users.contains(&list));
        assert_eq!(s.backings[&old].generation, old_generation, "reuse cannot replace the old backing generation");
        assert_eq!(s.recordings.get(&list), Some(&5));
    }

    #[test]
    fn retired_list_stays_in_flight_and_old_token_cannot_unpin_reused_address() {
        let (list, old, next) = (0x7ffe_7681usize, 0x7ffe_7682usize, 0x7ffe_7683usize);
        created_on(list, old, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let old_submit = begin_submit(&[list as *mut c_void]).unwrap();
        retire_recording(list);
        assert!(in_flight(list), "retirement keeps the exact pending submit alive");

        forget_list(list);
        created_on(list, next, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let reused_submit = begin_submit(&[list as *mut c_void]).unwrap();
        drop(old_submit);
        assert!(in_flight(list), "an old lifetime token cannot unpin the reused raw address");
        drop(reused_submit);
        assert!(!in_flight(list));
        forget_list(list);
        let mut g = STATE.lock().unwrap();
        let s = g.as_mut().unwrap();
        s.backings.remove(&old);
        s.backings.remove(&next);
        s.types.remove(&old);
        s.types.remove(&next);
    }

    #[test]
    fn idle_reset_commit_is_once_only_and_rejects_concurrent_recording() {
        let (alloc, spare, failed, list) = (0x7ffe_7701usize, 0x7ffe_7702usize, 0x7ffe_7703usize, 0x7ffe_7704usize);
        let mut s = State::default();
        s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        s.aside.insert(alloc, vec![spare]);
        backing_mut(&mut s, spare);

        let (mine, probes) = namespace_probes(&s, alloc);
        let successful = vec![probes.iter().find(|probe| probe.raw == spare).unwrap().clone()];
        let token = reserve_reset(&mut s, alloc).unwrap();
        assert!(commit_idle_resets(&mut s, alloc, token, &mine, &probes, &successful, &mut Vec::new()));
        assert!(s.backings[&spare].already_reset);
        assert!(namespace_probes(&s, alloc).1.iter().find(|probe| probe.raw == spare).unwrap().already_reset,
            "the next reclaim pass skips an unchanged backing");

        s.aside.get_mut(&alloc).unwrap().push(failed);
        backing_mut(&mut s, failed);
        let (mine, probes) = namespace_probes(&s, alloc);
        let token = reserve_reset(&mut s, alloc).unwrap();
        assert!(commit_idle_resets(&mut s, alloc, token, &mine, &probes, &[], &mut Vec::new()));
        assert!(!s.backings[&failed].already_reset, "a failed driver Reset remains retryable");
        assert!(!s.resetting.contains_key(&alloc), "a failed idle batch releases its token");

        let (mine, probes) = namespace_probes(&s, alloc);
        let token = reserve_reset(&mut s, alloc).unwrap();
        move_list(&mut s, list, spare);
        assert!(!commit_idle_resets(&mut s, alloc, token, &mine, &probes, &successful, &mut Vec::new()),
            "a recording created during the unlocked Reset invalidates its exact claim");
        assert!(!s.backings[&spare].already_reset);
        assert!(!s.resetting.contains_key(&alloc));
    }

    #[test]
    fn reset_commit_rejects_reentrant_backing_mutation() {
        let (alloc, spare, list) = (0x7ffe_7a10usize, 0x7ffe_7a20usize, 0x7ffe_7a30usize);
        let (mine, probes, token) = {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            move_list(s, list, alloc);
            s.aside.insert(alloc, vec![spare]);
            backing_mut(s, spare);
            let (mine, probes) = namespace_probes(s, alloc);
            let token = reserve_reset(s, alloc).unwrap();
            (mine, probes, token)
        };
        assert_eq!(run_reset(spare as *mut c_void, |_| {
            let started = std::time::Instant::now();
            let mut state = loop {
                if let Ok(state) = STATE.try_lock() { break state; }
                assert!(started.elapsed() < std::time::Duration::from_secs(1), "driver Reset callback could not reenter allocator state");
                std::thread::yield_now();
            };
            state.as_mut().unwrap().backings.get_mut(&spare).unwrap().submit_pins += 1;
            0
        }), 0);
        let (committed, token_released) = {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            let committed = commit_reset(s, alloc, alloc, spare, token, &mine, &probes, true, false, &mut Vec::new());
            let token_released = !s.resetting.contains_key(&alloc);
            s.of_list.remove(&list);
            s.backings.remove(&alloc);
            s.backings.remove(&spare);
            s.aside.remove(&alloc);
            s.types.remove(&alloc);
            (committed, token_released)
        };
        assert!(!committed, "state changed during the unlocked driver call must fail closed");
        assert!(token_released, "a rejected reset commit must release its operation token");
    }

    #[test]
    fn final_release_cancels_reset_before_raw_namespace_reuse() {
        let (alloc, spare, list) = (0x7ffe_7b10usize, 0x7ffe_7b20usize, 0x7ffe_7b30usize);
        let mut s = State::default();
        s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        move_list(&mut s, list, alloc);
        s.aside.insert(alloc, vec![spare]);
        backing_mut(&mut s, spare);
        let (mine, probes) = namespace_probes(&s, alloc);
        let old_token = reserve_reset(&mut s, alloc).unwrap();

        let _ = unsafe { forget_in(&mut s, alloc, &HashSet::new()) };
        s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        s.aside.insert(alloc, vec![spare]);
        let new_token = reserve_reset(&mut s, alloc).unwrap();
        assert_ne!(old_token, new_token);
        assert!(!commit_reset(&mut s, alloc, alloc, spare, old_token, &mine, &probes, true, false, &mut Vec::new()));
        assert_eq!(s.resetting.get(&alloc), Some(&new_token), "a stale completion cannot cancel the replacement's claim");
    }

    /// Final-release cleanup drops only the recording's hold: another live list and a pending
    /// GPU fence each still prevent reuse, then completion makes the backing reusable.
    #[test]
    fn released_lists_leave_only_live_recordings_and_gpu_work_holding_the_backing() {
        use std::mem::{offset_of, size_of};
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: AtomicU64,
            refs: AtomicU32,
            final_releases: AtomicU32,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            (*this.cast::<Fence>()).value.load(Ordering::Acquire)
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.refs.fetch_add(1, Ordering::AcqRel) + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            let refs = f.refs.fetch_sub(1, Ordering::AcqRel) - 1;
            if refs == 0 {
                f.final_releases.fetch_add(1, Ordering::AcqRel);
            }
            refs
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let vtable = Box::leak(Box::new([unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT]));
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.AddRef) / SLOT] = add_ref as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let fence = Box::leak(Box::new(Fence {
            vtable: vtable.as_ptr(),
            value: AtomicU64::new(0),
            refs: AtomicU32::new(1),
            final_releases: AtomicU32::new(0),
        }));
        let mut ids = Box::new([0u8; 3]);
        let alloc = ids.as_mut_ptr() as usize;
        let (first, second) = (alloc + 1, alloc + 2);
        created_on(first, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        created_on(second, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        {
            let mut g = STATE.lock().unwrap();
            let b = g.as_mut().unwrap().backings.get_mut(&alloc).unwrap();
            // The map owns one synthetic fence reference, dropped by forget below.
            let f = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut *fence).cast()) };
            b.marks.insert(1, (f, 7));
        }

        super::super::list_x::on_release(first);
        let first_removed = {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            !s.of_list.contains_key(&first) && s.of_list.get(&second) == Some(&alloc)
                && s.backings[&alloc].users == HashSet::from([second])
                && !unsafe { backing_free_for_test(&s.backings[&alloc]) }
        };
        super::super::list_x::on_release(second);
        super::super::list_x::on_release(first); // duplicate cleanup is harmless
        let pending_still_held = {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            !s.of_list.contains_key(&second) && s.backings[&alloc].users.is_empty()
                && s.backings[&alloc].marks.len() == 1 && !unsafe { backing_free_for_test(&s.backings[&alloc]) }
        };
        fence.value.store(7, Ordering::Release);
        let reusable = unsafe { backing_free_for_test(&STATE.lock().unwrap().as_ref().unwrap().backings[&alloc]) };
        // Clean up the synthetic COM pointer before assertions, including on the old code.
        unsafe { forget(alloc) };
        assert!(first_removed, "another live recording must remain while the released list goes");
        assert!(pending_still_held, "final list Release must retain the submitted-work fence");
        assert!(reusable, "a released list cannot keep a completed backing busy");
        assert_eq!(
            (fence.refs.load(Ordering::Acquire), fence.final_releases.load(Ordering::Acquire)),
            (0, 1),
        );
    }

    /// A list recorded once and submitted again after the title reset its allocator: the
    /// recording is still there (the device is not removed), the title's allocator goes on with
    /// another PC allocator, and the first one is used again once the list has moved on.
    #[test]
    fn a_recording_outlives_the_reset_of_the_titles_allocator() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let a = alloc.as_raw();
            if !super::super::real::is_adopted(a) {
                super::super::real::adopt(a, super::super::real::Kind::CommandAllocator);
            }
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let l = list.as_raw();
            created_on(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            list.Close().unwrap();
            let lists = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done = |v: u64| {
                queue.Signal(&fence, v).unwrap();
                let start = std::time::Instant::now();
                while fence.GetCompletedValue() < v && start.elapsed().as_secs() < 10 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert!(fence.GetCompletedValue() >= v, "the queue finishes");
            };

            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(1);

            // the title resets its allocator (through the hook: `real::alloc_reset`) …
            alloc.Reset().unwrap();
            let second = current(a as usize);
            assert_ne!(second, a as usize, "the list still holds its recording: another PC allocator takes the place");
            // … and submits the recording again
            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(2);
            assert!(dev.GetDeviceRemovedReason().is_ok(), "the recording was still there");

            // the list is reset on the title's allocator: it records on the PC allocator in use
            let second_alloc = ID3D12CommandAllocator::from_raw_borrowed(&(second as *mut c_void)).unwrap().clone();
            assert_eq!(reset_list(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc| {
                assert_eq!(pc, second);
                list.Reset(&second_alloc, None).map_or_else(|e| e.code().0, |_| 0)
            }), 0);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(3);

            // the next reset: the first PC allocator holds nothing now and is used again
            alloc.Reset().unwrap();
            assert_eq!(current(a as usize), a as usize, "the allocator put aside is free again");
            assert!(dev.GetDeviceRemovedReason().is_ok());

            // the title's allocator goes while another PC allocator stands for it: nothing of
            // it stays (the next allocator at its address is its own)
            assert_eq!(reset_list(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc| {
                assert_eq!(pc, a as usize);
                list.Reset(&alloc, None).map_or_else(|e| e.code().0, |_| 0)
            }), 0);
            list.Close().unwrap();
            alloc.Reset().unwrap();
            assert_ne!(current(a as usize), a as usize);
            drop(lists);
            drop(list);
            drop(alloc);
            assert_eq!(current(a as usize), a as usize, "a released allocator leaves nothing behind");
            assert!(!STATE.lock().unwrap().as_ref().unwrap().types.contains_key(&(a as usize)));
        }
    }

    /// A list submitted again while its previous submit is still on the GPU is counted (and
    /// logged); once that submit is done, or after the list is forgotten, it is not.
    #[test]
    fn a_resubmit_while_the_previous_submit_runs_is_noticed() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            // a fence the GPU has not reached stands for a submit still running
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut ids = Box::new([0u8; 2]);
            let (alloc, list) = (ids.as_mut_ptr() as usize, ids.as_mut_ptr() as usize + 1);
            let queue = 0x10usize as *mut c_void;
            created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            // A real first submission captures the list's exact address lifetime before it
            // publishes LastSubmit. Seed that identity before constructing the synthetic submit.
            drop(begin_submit(&[list as *mut c_void]).unwrap());
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                let recording = s.recordings[&list];
                s.last_submit.insert(list, LastSubmit { queue: queue as usize, fence: fence.clone(), value: 1, recording });
            }
            let while_running = note_resubmits(queue, &[list as *mut c_void]);
            fence.Signal(1).unwrap();
            let after_done = note_resubmits(queue, &[list as *mut c_void]);
            {
                // running again, then the list goes: nothing is left to compare with
                let mut g = STATE.lock().unwrap();
                g.as_mut().unwrap().last_submit.get_mut(&list).unwrap().value = 2;
            }
            forget_list(list);
            let after_forget = note_resubmits(queue, &[list as *mut c_void]);
            forget(alloc);
            assert_eq!(while_running, 1, "a submit while the previous one runs must be noticed");
            assert_eq!(after_done, 0, "a finished previous submit is not a resubmit");
            assert_eq!(after_forget, 0, "a forgotten list has no previous submit");
        }
    }
}
