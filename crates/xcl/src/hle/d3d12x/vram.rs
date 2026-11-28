//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (32 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels). The CPU never sees these textures
//! (on the console it could, but GPU-owned targets are not CPU-written).
//!
//! Arenas cost video memory for whole ranges, and a placement straddling two made a third over
//! both while the two lived on with their resources: a retail game's streaming targets, packed
//! back to back over 3.5 GB of its addresses, made 15 GB of arenas in two seconds – past the
//! card's memory and the system's commit, and the device was removed. So the first choice is
//! the console's own model (`Pages`): a 64 KiB page of video memory for each 64 KiB page of
//! the title's address a target covers, and each target a reserved (tiled) resource whose tiles
//! are mapped onto the pages of its addresses. Targets at one address share pages, a page
//! exists once however many targets cover it, and what no target covers takes nothing.
//! Arenas remain for what cannot be tiled (an address not 64 KiB aligned, a resource the
//! driver does not make reserved: D32S8, MSAA): heaps of the placement's size, shared by
//! placements inside them, let go 2 s after their last resource (256 MiB ones: 2.8 GB of D32S8).

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// A heap of video memory for placements at title addresses `[base, base + size)`.
struct Arena<H> {
    dev: usize,
    base: usize,
    size: usize,
    /// the heap's alignment (a placement that needs more does not go in)
    align: usize,
    heap: H,
    /// resources placed in it
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
    id: u64,
}

/// The arenas, apart from D3D12 (`H`: the heap; numbers in the tests).
struct Arenas<H> {
    list: Vec<Arena<H>>,
    next_id: u64,
}

impl<H: Clone> Arenas<H> {
    const fn new() -> Self {
        Arenas { list: Vec::new(), next_id: 0 }
    }

    /// Return expired heaps for `dev` (preserving insertion order) to retire/drop outside `ARENAS.lock()`.
    fn sweep_dropped_for(&mut self, dev: usize, now: std::time::Instant) -> Vec<H> {
        let is_expired = |a: &Arena<H>| a.dev == dev && a.users == 0 && a.idle_since.is_some_and(|t| now.duration_since(t) >= IDLE);
        if !self.list.iter().any(is_expired) {
            return Vec::new();
        }
        let mut dropped = Vec::new();
        let mut kept = Vec::with_capacity(self.list.len());
        for a in self.list.drain(..) {
            if is_expired(&a) {
                dropped.push(a.heap);
            } else {
                kept.push(a);
            }
        }
        self.list = kept;
        dropped
    }

    /// Place `bytes` at `va` (`align`) on `dev`; returns `(placement, expired_heaps)` without dropping inside `ARENAS.lock()`.
    fn take_for(&mut self, dev: usize, va: usize, bytes: usize, align: usize, now: std::time::Instant, make: impl FnOnce(usize, usize) -> Option<H>) -> (Option<(H, usize, u64)>, Vec<H>) {
        let expired = self.sweep_dropped_for(dev, now);
        let fits = |a: &Arena<H>| a.dev == dev && a.align >= align && a.base <= va && va + bytes <= a.base + a.size && (va - a.base) % align == 0;
        let i = match self.list.iter().rposition(fits) {
            Some(i) => i,
            None => {
                let size = bytes.div_ceil(align) * align;
                let Some(heap) = make(size, align) else { return (None, expired) };
                self.next_id += 1;
                self.list.push(Arena { dev, base: va, size, align, heap, users: 0, idle_since: None, id: self.next_id });
                self.list.len() - 1
            }
        };
        let a = &mut self.list[i];
        a.users += 1;
        a.idle_since = None;
        (Some((a.heap.clone(), va - a.base, a.id)), expired)
    }

    /// A resource in arena `id` released.
    fn release(&mut self, id: u64, now: std::time::Instant) {
        if let Some(a) = self.list.iter_mut().find(|a| a.id == id) {
            a.users = a.users.saturating_sub(1);
            if a.users == 0 {
                a.idle_since = Some(now);
            }
        }
    }

    fn bytes(&self) -> usize { self.list.iter().map(|a| a.size).sum() }
    fn bytes_for(&self, dev: usize) -> usize {
        self.list.iter().filter(|a| a.dev == dev).fold(0, |bytes, a| bytes.saturating_add(a.size))
    }
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arenas<ID3D12Heap> {}

static ARENAS: Mutex<Arenas<ID3D12Heap>> = Mutex::new(Arenas::new());
/// resource → its arena's id
static ARENA_OF: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
/// bytes of every arena made
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (bytes of the arenas in use, bytes of all arenas made) – `XCL_ACTIVITY`
pub fn arena_bytes() -> (usize, usize) {
    (ARENAS.lock().unwrap().bytes(), MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed))
}

pub(super) fn arena_bytes_for(device: usize) -> usize {
    ARENAS.lock().unwrap().bytes_for(device)
}
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// `XCL_TEXTURE_BUDGET_EVICT=1`: retain CPU-side identities for live texture SRVs. This is
/// preparation only: recorded command lists can still name older descriptor contents, so these
/// records do not make descriptor replacement or texture eviction GPU-safe.
#[inline]
pub(super) fn srv_tracking_enabled() -> bool {
    #[cfg(test)]
    if SRV_TRACKING_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_TEXTURE_BUDGET_EVICT").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!(
                "D3D12.X: XCL_TEXTURE_BUDGET_EVICT=1 - evicting oldest top texture mips above {}% of local VRAM budget (safe-pinned packed tails stay resident)",
                budget_evict_pct()
            );
        }
        on
    })
}

#[cfg(test)]
static SRV_TRACKING_FOR_TEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SharedViewEpoch(u64);

enum EpochRetirement<R> {
    Reserved,
    Receipt(R),
    Poisoned,
}

struct SharedViewEpochs<R> {
    next: u64,
    poisoned: bool,
    pending: Vec<(SharedViewEpoch, EpochRetirement<R>)>,
}

impl<R> SharedViewEpochs<R> {
    const fn new() -> Self {
        Self { next: 0, poisoned: false, pending: Vec::new() }
    }

    fn reserve(&mut self) -> SharedViewEpoch {
        self.next = self.next.checked_add(1).expect("shared view heap epoch overflow");
        let epoch = SharedViewEpoch(self.next);
        self.pending.push((epoch, EpochRetirement::Reserved));
        epoch
    }

    fn cancel(&mut self, epoch: SharedViewEpoch) {
        match self.pending.iter().position(|entry| entry.0 == epoch) {
            Some(i) if matches!(self.pending[i].1, EpochRetirement::Reserved) => {
                self.pending.remove(i);
            }
            _ => self.poisoned = true,
        }
    }

    fn attach(&mut self, epoch: SharedViewEpoch, receipt: R) {
        match self.pending.iter_mut().find(|entry| entry.0 == epoch) {
            Some((_, state @ EpochRetirement::Reserved)) => *state = EpochRetirement::Receipt(receipt),
            _ => self.poisoned = true,
        }
    }

    fn poison(&mut self, epoch: SharedViewEpoch) {
        match self.pending.iter_mut().find(|entry| entry.0 == epoch) {
            Some((_, state @ EpochRetirement::Reserved)) => *state = EpochRetirement::Poisoned,
            _ => self.poisoned = true,
        }
    }

    fn sweep(&mut self, complete: impl Fn(&R) -> bool) {
        self.pending.retain(|(_, state)| !matches!(state, EpochRetirement::Receipt(receipt) if complete(receipt)));
    }

    fn quiescent(&mut self, complete: impl Fn(&R) -> bool) -> bool {
        self.sweep(complete);
        !self.poisoned && self.pending.is_empty()
    }
}

fn with_quiescent_epochs<R, T>(
    tracking_enabled: bool,
    epochs: &Mutex<SharedViewEpochs<R>>,
    complete: impl Fn(&R) -> bool,
    mutation: impl FnOnce() -> T,
) -> Option<T> {
    if !tracking_enabled {
        return None;
    }
    let mut epochs = epochs.lock().unwrap();
    if !epochs.quiescent(complete) {
        return None;
    }
    let result = mutation();
    drop(epochs);
    Some(result)
}

static SHARED_VIEW_EPOCHS: Mutex<SharedViewEpochs<super::alloc_pool::SubmissionReceipt>> =
    Mutex::new(SharedViewEpochs::new());

pub(super) fn reserve_shared_view_epoch() -> SharedViewEpoch {
    let mut epochs = SHARED_VIEW_EPOCHS.lock().unwrap();
    epochs.sweep(super::alloc_pool::SubmissionReceipt::is_complete);
    epochs.reserve()
}

pub(super) fn cancel_shared_view_epoch(epoch: SharedViewEpoch) {
    SHARED_VIEW_EPOCHS.lock().unwrap().cancel(epoch);
}

pub(super) fn attach_shared_view_epoch(epoch: SharedViewEpoch, receipt: super::alloc_pool::SubmissionReceipt) {
    SHARED_VIEW_EPOCHS.lock().unwrap().attach(epoch, receipt);
}

pub(super) fn poison_shared_view_epoch(epoch: SharedViewEpoch) {
    SHARED_VIEW_EPOCHS.lock().unwrap().poison(epoch);
}

/// Run a descriptor mutation only while every tracked shared-view submission is retired, holding
/// the epoch lock until the mutation ends so a new reservation cannot cross the check. This lock
/// is outermost: the closure may take descriptor backing/tracker locks, but must not submit GPU
/// work or call anything that reserves another shared-view epoch.
///
/// A missing post-execute receipt permanently returns `None`, so an untracked submission (or a
/// reused queue address) can never be mistaken for quiescence.
pub(super) fn with_quiescent_shared_view_heap<T>(mutation: impl FnOnce() -> T) -> Option<T> {
    with_quiescent_epochs(
        srv_tracking_enabled(), &SHARED_VIEW_EPOCHS, super::alloc_pool::SubmissionReceipt::is_complete, mutation,
    )
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct SrvSlot {
    heap_generation: u64,
    slot: u32,
}

impl SrvSlot {
    fn of(slot: &super::console_heap::Slot) -> Self {
        Self { heap_generation: slot.generation, slot: slot.k }
    }
}

/// Numeric COM identities are never dereferenced here. Resource and heap lifetime hooks remove
/// them; a later descriptor-generation design must still prove queue-fence safety before use.
#[derive(Clone, Copy)]
struct TrackedSrv {
    generation: u64,
    device: usize,
    resource: usize,
    pc_srv: u64,
    shared_srv: u64,
    index: u32,
    desc: Option<D3D12_SHADER_RESOURCE_VIEW_DESC>,
    /// Exact XCL ABI descriptor bytes written for this logical generation. Only shared-heap slots
    /// are tracked: their global word-0 index makes an unhooked raw overwrite observable.
    words: [u32; 8],
}

#[derive(Default)]
struct TrackedSrvs {
    next_generation: u64,
    by_slot: HashMap<SrvSlot, TrackedSrv>,
}

impl TrackedSrvs {
    fn write(&mut self, slot: SrvSlot, entry: Option<TrackedSrv>) {
        self.by_slot.remove(&slot);
        if let Some(mut entry) = entry {
            self.next_generation = self.next_generation.wrapping_add(1).max(1);
            entry.generation = self.next_generation;
            self.by_slot.insert(slot, entry);
        }
    }

    fn copy_range(
        &mut self,
        device: usize,
        copies: &[(SrvSlot, u64, u64, u32, SrvSlot, bool, [u32; 8], [u32; 8])],
    ) {
        let sources: Vec<_> = copies.iter().map(|(src, _, _, _, _, track_source, ..)|
            track_source.then(|| self.by_slot.get(src).copied()).flatten()).collect();
        for ((_, pc_srv, shared_srv, index, dst, _, source_words, destination_words), source) in copies.iter().zip(sources) {
            let entry = source.filter(|entry| entry.words == *source_words).map(|mut entry| {
                entry.device = device;
                entry.pc_srv = *pc_srv;
                entry.shared_srv = *shared_srv;
                entry.index = *index;
                entry.words = *destination_words;
                entry
            });
            self.write(*dst, entry);
        }
    }

    fn forget_heap(&mut self, generation: u64) {
        self.by_slot.retain(|slot, _| slot.heap_generation != generation);
    }

    fn forget_resource(&mut self, resource: usize) {
        self.by_slot.retain(|_, entry| entry.resource != resource);
    }

    /// Apply raw-memory observations made without this lock. A concurrent tracked overwrite has
    /// another logical generation and is omitted from this snapshot rather than judged against
    /// words observed for its predecessor.
    fn reconcile_observed(&mut self, observed: &[(SrvSlot, u64, Option<[u32; 8]>)]) -> Vec<(SrvSlot, TrackedSrv)> {
        let mut snapshot = Vec::new();
        for &(slot, generation, words) in observed {
            let Some(entry) = self.by_slot.get(&slot).copied().filter(|entry| entry.generation == generation) else { continue };
            if words == Some(entry.words) {
                snapshot.push((slot, entry));
            } else {
                self.by_slot.remove(&slot);
            }
        }
        snapshot
    }
}

static TRACKED_SRVS: Mutex<Option<TrackedSrvs>> = Mutex::new(None);
static TRACKED_SRV_WRITES: Mutex<()> = Mutex::new(());
static TRACKED_SRV_LIFETIMES: Mutex<()> = Mutex::new(());

pub(super) type TrackedSrvWriteGuard = std::sync::MutexGuard<'static, ()>;
pub(super) type TrackedSrvLifetimeGuard = std::sync::MutexGuard<'static, ()>;

/// Serialize an entire hooked descriptor mutation while tracking is enabled. Operation tokens
/// alone order hook starts, not native/console writes that complete in the opposite order.
pub(super) fn lock_tracked_srv_write() -> Option<TrackedSrvWriteGuard> {
    srv_tracking_enabled().then(|| TRACKED_SRV_WRITES.lock().unwrap())
}

/// Exclude final native resource Release while a tracked entry is turned into a strong owner.
/// This is deliberately distinct from `TRACKED_SRV_WRITES`: descriptor hooks hold the write gate
/// across driver calls which may make a temporary, non-final resource Release.
pub(super) fn lock_tracked_srv_lifetime() -> Option<TrackedSrvLifetimeGuard> {
    srv_tracking_enabled().then(|| TRACKED_SRV_LIFETIMES.lock().unwrap())
}

/// Record the descriptor currently in `slot`. `None` clears an SRV overwritten by another view.
pub(super) fn track_srv(
    device: usize,
    resource: usize,
    slot: &super::console_heap::Slot,
    desc: Option<D3D12_SHADER_RESOURCE_VIEW_DESC>,
    words: [u32; 8],
) {
    if !srv_tracking_enabled() || slot.shared_srv == 0 {
        return;
    }
    let texture_view = desc.is_none_or(|v| {
        v.ViewDimension != D3D12_SRV_DIMENSION_BUFFER && v.ViewDimension != D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE
    });
    let entry = (resource != 0 && texture_view && is_placed_texture(resource)).then_some(TrackedSrv {
        generation: 0,
        device,
        resource,
        pc_srv: slot.pc_srv,
        shared_srv: slot.shared_srv,
        index: slot.index,
        desc,
        words,
    });
    TRACKED_SRVS.lock().unwrap().get_or_insert_with(Default::default).write(SrvSlot::of(slot), entry);
}

pub(super) fn clear_tracked_srv(slot: &super::console_heap::Slot) {
    if srv_tracking_enabled() {
        TRACKED_SRVS.lock().unwrap().get_or_insert_with(Default::default).write(SrvSlot::of(slot), None);
    }
}

pub(super) fn copy_tracked_srvs(
    device: usize,
    copies: &[(super::console_heap::Slot, super::console_heap::Slot, [u32; 8], [u32; 8])],
) {
    if !srv_tracking_enabled() || copies.is_empty() {
        return;
    }
    let copies: Vec<_> = copies
        .iter()
        .map(|(src, dst, source_words, destination_words)|
            (SrvSlot::of(src), dst.pc_srv, dst.shared_srv, dst.index, SrvSlot::of(dst),
                src.shared_srv != 0 && dst.shared_srv != 0, *source_words, *destination_words))
        .collect();
    TRACKED_SRVS.lock().unwrap().get_or_insert_with(Default::default).copy_range(device, &copies);
}

pub(super) fn forget_srv_heap(generation: u64) {
    if !srv_tracking_enabled() {
        return;
    }
    let _write = lock_tracked_srv_write();
    if let Some(tracked) = TRACKED_SRVS.lock().unwrap().as_mut() {
        tracked.forget_heap(generation);
    }
}

/// Remove candidates for a descriptor heap that reached final native `Release`, while the heap
/// lifetime gate still excludes snapshots. Do not take `TRACKED_SRV_WRITES`: a descriptor hook
/// may hold it across a driver call which reenters a temporary heap Release.
pub(super) fn forget_srv_heap_after_release(
    generation: u64, lifetime: Option<&super::console_heap::HeapLifetimeGuard>,
) {
    debug_assert!(!srv_tracking_enabled() || lifetime.is_some());
    if lifetime.is_some() {
        if let Some(tracked) = TRACKED_SRVS.lock().unwrap().as_mut() {
            tracked.forget_heap(generation);
        }
    }
}

fn forget_tracked_srvs_locked(resource: usize) {
    if let Some(tracked) = TRACKED_SRVS.lock().unwrap().as_mut() {
        tracked.forget_resource(resource);
    }
}

#[cfg(test)]
fn forget_tracked_srvs(resource: usize) {
    if !srv_tracking_enabled() {
        return;
    }
    let _write = lock_tracked_srv_write();
    forget_tracked_srvs_locked(resource);
}

/// Clone an owner for one exact tracked resource while final `Release` is excluded. The tracker
/// mutex makes the snapshot coherent; this does not take the descriptor-write gate. A later
/// mutator must take that gate and revalidate the logical generation, as well as lease the exact
/// heap generation, before using `pc_srv` or `shared_srv`.
fn lease_tracked_srv_resource(slot: SrvSlot, generation: u64) -> Option<(TrackedSrv, ID3D12Resource, super::console_heap::ViewSlotLease)> {
    let (entry, resource) = lease_tracked_srv_resource_with(slot, generation, |resource| {
        let raw = resource as *mut std::ffi::c_void;
        unsafe { ID3D12Resource::from_raw_borrowed(&raw).cloned() }
    })?;
    let heap = super::console_heap::lease_view_slot(
        slot.heap_generation, slot.slot, entry.pc_srv, entry.shared_srv, entry.index,
    )?;
    Some((entry, resource, heap))
}

fn lease_tracked_srv_resource_with<T>(
    slot: SrvSlot, generation: u64, lease: impl FnOnce(usize) -> Option<T>,
) -> Option<(TrackedSrv, T)> {
    if !srv_tracking_enabled() {
        return None;
    }
    let _lifetime = TRACKED_SRV_LIFETIMES.lock().unwrap();
    let entry = TRACKED_SRVS
        .lock()
        .unwrap()
        .as_ref()?
        .by_slot
        .get(&slot)
        .copied()
        .filter(|entry| entry.generation == generation)?;
    let owner = lease(entry.resource)?;
    Some((entry, owner))
}

/// Execute one point-in-time mutation of an exact tracked SRV generation. Queue quiescence is
/// held outermost so no new shared-view submission can cross the mutation. Strong resource and
/// heap owners are acquired before `WRITE`, then the tracker generation and all eight console
/// words are revalidated immediately before `mutate` runs. The tracker mutex is not held across
/// `mutate`; `WRITE` is, so the closure must call only unhooked descriptor operations and must not
/// submit work or reserve another shared-view epoch.
///
/// Title code can write console descriptor memory without using a hook. Such a write can race
/// after the point-in-time word check, so this primitive deliberately promises no stronger
/// atomicity than the title's descriptor-lifetime rules. A mismatch fails closed and forgets only
/// the stale generation. Strong owners are dropped after `WRITE` and the tracker lock are gone.
#[allow(dead_code)] // consumed by the later descriptor-rewrite policy
fn with_exact_tracked_srv_mutation<T>(
    slot: SrvSlot,
    generation: u64,
    mutate: impl FnOnce(&TrackedSrv, &ID3D12Resource, &super::console_heap::ViewSlotLease) -> T,
) -> Option<T> {
    with_exact_tracked_srv_mutation_on(
        slot,
        generation,
        lease_tracked_srv_resource,
        |slot| super::console_heap::exact_view_slot_words(slot.heap_generation, slot.slot),
        mutate,
    )
}

fn with_exact_tracked_srv_mutation_on<T, Resource, Heap>(
    slot: SrvSlot,
    generation: u64,
    lease: impl FnOnce(SrvSlot, u64) -> Option<(TrackedSrv, Resource, Heap)>,
    observe_words: impl FnOnce(SrvSlot) -> Option<[u32; 8]>,
    mutate: impl FnOnce(&TrackedSrv, &Resource, &Heap) -> T,
) -> Option<T> {
    with_quiescent_shared_view_heap(|| {
        let (snapshot, resource, heap) = lease(slot, generation)?;
        let result = {
            let _write = lock_tracked_srv_write()?;
            let valid = {
                let mut tracked = TRACKED_SRVS.lock().unwrap();
                let current = tracked.as_mut()?.by_slot.get(&slot).copied()?;
                if current.generation != generation {
                    return None;
                }
                // This is TRACKED_SRVS -> HEAPS. Final heap cleanup drops HEAPS before taking
                // TRACKED_SRVS, and descriptor-hook lookups also take them sequentially, so there
                // is no nested inverse while the exact words are observed.
                if observe_words(slot) != Some(current.words) {
                    tracked.as_mut()?.by_slot.remove(&slot);
                    return None;
                }
                // The generation identifies the complete immutable tracker snapshot. Drop this
                // mutex before the supplied operation; WRITE keeps hooked mutations excluded.
                current
            };
            debug_assert_eq!(valid.generation, snapshot.generation);
            Some(mutate(&valid, &resource, &heap))
        };
        // `resource` and `heap` drop here, after the tracker and WRITE guards above.
        result
    })
    .flatten()
}

/// One exact tracked descriptor and the native owners which make its resource and heap safe to
/// use during a batch mutation.
struct ExactSrvMutation<'a, Resource, Heap> {
    entry: TrackedSrv,
    resource: &'a Resource,
    heap: &'a Heap,
}

/// Execute one point-in-time mutation of a batch of tracked SRV generations. One quiescent epoch
/// and one descriptor-write exclusion cover the whole batch; missing leases and stale candidates
/// fail closed independently, in the caller's deterministic order. The operation is not called if
/// no candidate survives revalidation. The supplied operation has the same restrictions as
/// `with_exact_tracked_srv_mutation`: it must use only unhooked descriptor operations and must not
/// submit work or reserve another shared-view epoch.
///
/// Raw title descriptor writes remain outside XCL's exclusion. All eight words are observed under
/// `WRITE` immediately before the operation, but title code can still race after that point; a
/// future policy must obey the title's descriptor-lifetime rules.
fn with_exact_tracked_srv_mutations<T>(
    candidates: &[(SrvSlot, u64)],
    mutate: impl FnOnce(&[ExactSrvMutation<'_, ID3D12Resource, super::console_heap::ViewSlotLease>]) -> T,
) -> Option<T> {
    with_exact_tracked_srv_mutations_on(
        candidates,
        lease_tracked_srv_resource,
        |slots| super::console_heap::view_slot_words(slots),
        mutate,
    )
}

fn with_exact_tracked_srv_mutations_on<T, Resource, Heap>(
    candidates: &[(SrvSlot, u64)],
    mut lease: impl FnMut(SrvSlot, u64) -> Option<(TrackedSrv, Resource, Heap)>,
    observe_words: impl FnOnce(&[(u64, u32)]) -> Vec<Option<[u32; 8]>>,
    mutate: impl FnOnce(&[ExactSrvMutation<Resource, Heap>]) -> T,
) -> Option<T> {
    if !srv_tracking_enabled() {
        return None;
    }
    let mut seen = HashSet::new();
    let candidates: Vec<_> = candidates.iter().copied().filter(|candidate| seen.insert(*candidate)).collect();
    if candidates.is_empty() {
        return None;
    }
    with_quiescent_shared_view_heap(|| {
        let leased: Vec<_> = candidates
            .into_iter()
            .filter_map(|(slot, generation)| lease(slot, generation).map(|(snapshot, resource, heap)| {
                (slot, generation, snapshot, resource, heap)
            }))
            .collect();
        if leased.is_empty() {
            return None;
        }
        let slots: Vec<_> = leased.iter().map(|(slot, ..)| (slot.heap_generation, slot.slot)).collect();

        // `leased` owns every native object and was declared before `write`: unwind therefore
        // drops the later-declared guard first, even for candidates rejected below.
        let mut valid = Vec::with_capacity(leased.len());
        let write = lock_tracked_srv_write()?;
        let observed = observe_words(&slots);
        let mut validated = Vec::with_capacity(leased.len());
        {
            let mut tracked = TRACKED_SRVS.lock().unwrap();
            let tracked = tracked.as_mut()?;
            for (i, (slot, generation, snapshot, ..)) in leased.iter().enumerate() {
                let Some(current) = tracked.by_slot.get(slot).copied() else {
                    validated.push(None);
                    continue;
                };
                if current.generation != *generation {
                    validated.push(None);
                    continue;
                }
                let words = observed.get(i).copied().flatten();
                if words != Some(current.words) {
                    // `current` was read under this mutex and still names the requested exact
                    // generation, so removal cannot erase a concurrent replacement.
                    tracked.by_slot.remove(slot);
                    validated.push(None);
                    continue;
                }
                debug_assert_eq!(snapshot.generation, current.generation);
                validated.push(Some(current));
            }
        }
        for ((_, _, _, resource, heap), entry) in leased.iter().zip(validated) {
            if let Some(entry) = entry {
                valid.push(ExactSrvMutation { entry, resource, heap });
            }
        }
        if valid.is_empty() {
            drop(write);
            drop(valid);
            drop(leased);
            return None;
        }
        let result = mutate(&valid);
        drop(write);
        drop(valid);
        drop(leased);
        Some(result)
    })
    .flatten()
}

/// An exact, on-demand snapshot for a future residency consumer. It is deliberately not on the
/// submit path: scanning every live SRV there would turn descriptor count into foreground cost.
/// A consumer must still capture descriptor generations with recorded lists and queue fences;
/// title code can raw-write descriptor memory immediately after this point-in-time snapshot.
#[allow(dead_code)]
fn reconciled_tracked_srvs() -> Vec<(SrvSlot, TrackedSrv)> {
    if !srv_tracking_enabled() {
        return Vec::new();
    }
    let candidates: Vec<_> = TRACKED_SRVS
        .lock()
        .unwrap()
        .as_ref()
        .map(|tracked| tracked.by_slot.iter().map(|(&slot, entry)| (slot, entry.generation)).collect())
        .unwrap_or_default();
    let slots: Vec<_> = candidates.iter().map(|(slot, _)| (slot.heap_generation, slot.slot)).collect();
    let words = super::console_heap::view_slot_words(&slots);
    let observed: Vec<_> = candidates
        .into_iter()
        .zip(words)
        .map(|((slot, generation), words)| (slot, generation, words))
        .collect();
    TRACKED_SRVS.lock().unwrap().as_mut().map_or_else(Vec::new, |tracked| tracked.reconcile_observed(&observed))
}

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains(&resource))
}

#[cfg(test)]
pub(super) fn forget(resource: usize) {
    forget_tracked_srvs(resource);
    forget_resource_backing(resource);
}

/// Remove descriptor records after native final `Release`, while the caller still holds the
/// lifetime gate acquired before that call. Do not acquire the descriptor-write gate here: an
/// unrelated hook may hold it across a driver call whose temporary resource Release waits on the
/// lifetime gate. The tracker mutex makes removal atomic with lease snapshots. A future descriptor
/// transaction must take the write gate and revalidate the exact generation and descriptor words;
/// a same-resource descriptor hook concurrent with final Release is invalid object lifetime use.
/// Once this returns no new resource lease can start, so the caller may drop the lifetime gate
/// before doing unrelated sparse/page/arena cleanup.
pub(super) fn forget_tracked_after_resource_release(
    resource: usize, lifetime: Option<&TrackedSrvLifetimeGuard>,
) {
    debug_assert!(!srv_tracking_enabled() || lifetime.is_some());
    if lifetime.is_some() {
        forget_tracked_srvs_locked(resource);
    }
}

pub(super) fn forget_resource_backing_after_release(resource: usize) {
    forget_resource_backing(resource);
}

fn forget_resource_backing(resource: usize) {
    let placed = PLACED_TEXTURES.lock().unwrap().as_mut().is_some_and(|textures| textures.remove(&resource));
    if placed {
        forget_pages(resource);
        if let Some(id) = ARENA_OF.lock().unwrap().as_mut().and_then(|m| m.remove(&resource)) {
            ARENAS.lock().unwrap().release(id, std::time::Instant::now());
        }
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// A new arena's heap: `size` bytes of video memory for title addresses from `va`.
unsafe fn arena_heap(dev: &ID3D12Device, va: usize, size: usize, align: usize) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // 64 KiB, or 4 MiB for MSAA targets
        Alignment: align as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: video-memory arena for {va:#x}: CreateHeap: {e}");
        return None;
    }
    let made = MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: video-memory arena #{n} for title addresses {va:#x}..{:#x} ({made} bytes of arenas made so far)", va + size);
    }
    heap
}

/// bytes of a page – of the title's addresses and of a tile
const PAGE: usize = 0x1_0000;
/// pages of a video-memory pool heap (64 MiB)
const POOL_PAGES: u32 = 1024;
/// a page no target uses any more is kept this long for the next target at its address (with
/// what it holds – a title may make a target again over memory it rendered to)
const IDLE: std::time::Duration = std::time::Duration::from_secs(2);
/// a pool tile given back is not reused before this (the GPU may still be at work on what the
/// title released)
const COOL: std::time::Duration = std::time::Duration::from_secs(2);

/// A pool tile: (pool heap, tile in it).
type Tile = (u32, u32);

struct Page {
    tile: Tile,
    /// targets using the page
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
}

/// A page of `Pages::pages` (its title address) or of `Pages::extra_pages` (its key there).
#[derive(Clone, Copy)]
enum PageKey {
    Own(usize),
    Extra(usize, usize),
}

/// The pages of `idle` – those whose last user went, oldest first, each with when – that no
/// resource has used for `IDLE` go back to the pools: out of their map, their tile to `cooling`.
/// An entry whose page was used again since (in use now, or idle again from a later moment, with
/// a later entry) is dropped. Only the queue is looked at: every page of both maps, looked at at
/// every placement, grew with the square of what a title streams.
fn sweep_idle(
    idle: &mut std::collections::VecDeque<(PageKey, std::time::Instant)>,
    pages: &mut std::collections::HashMap<usize, Page>,
    extra_pages: &mut std::collections::HashMap<(usize, usize), Page>,
    cooling: &mut std::collections::VecDeque<(Tile, std::time::Instant)>,
    now: std::time::Instant,
) {
    while idle.front().is_some_and(|&(_, since)| now.duration_since(since) >= IDLE) {
        let Some((key, since)) = idle.pop_front() else { break };
        let still_idle = |p: &Page| p.idle_since == Some(since);
        let page = match key {
            PageKey::Own(a) if pages.get(&a).is_some_and(still_idle) => pages.remove(&a),
            PageKey::Extra(va, i) if extra_pages.get(&(va, i)).is_some_and(still_idle) => extra_pages.remove(&(va, i)),
            _ => None,
        };
        if let Some(p) = page {
            cooling.push_back((p.tile, now));
        }
    }
}

struct Target {
    va: usize,
    console_bytes: Option<u64>,
    total: u32,
    mapped: bool,
    used: Vec<usize>,
    used_extra: Vec<(usize, usize)>,
    /// its levels get their pages one by one (`map_levels`); None: all of it has pages
    levels: Option<Levels>,
    /// virtual, resident, and SRV-sampled texel accounting for sampled placed textures (`None` for GPU-owned targets)
    texels: Option<TexelMeta>,
    /// Last touch (placement, level map, or SRV create) for coldest-first budget eviction.
    last_touch: std::time::Instant,
    /// Lowest mip level whose tiles remain resident (raised when top mips are evicted under budget).
    min_resident_mip: u32,
}

/// The tiles of a texture whose levels get video memory when an upload first has texels for
/// them (`level_pages`): a group for each level with tiles of its own, in order, then one for
/// the packed levels.
#[derive(Clone, Debug, PartialEq)]
struct Levels {
    /// (the group's first subresource, its first tile in the resource, its tiles, mapped)
    groups: Vec<(u32, u32, u32, bool)>,
    /// Physical-page keys owned by each mapped group.
    group_tiles: Vec<(Vec<usize>, Vec<(usize, usize)>)>,
    /// the texture's levels (the last group's go up to here)
    mips: u32,
}

impl Levels {
    /// From the resource's tiling: `standard` (first tile, tiles) of each level with tiles of
    /// its own, `packed` the same of the packed levels, `total` tiles in all. None: they do not
    /// add up to the resource (it then gets all its pages at once).
    fn new(standard: &[(u32, u32)], packed: Option<(u32, u32)>, total: u32, mips: u32) -> Option<Levels> {
        let mut groups: Vec<(u32, u32, u32, bool)> = standard.iter().enumerate().map(|(i, &(first, n))| (i as u32, first, n, false)).collect();
        if let Some((first, n)) = packed {
            groups.push((standard.len() as u32, first, n, false));
        }
        let mut next = 0u32;
        let mut by_tile = groups.clone();
        by_tile.sort_by_key(|g| g.1);
        for g in &by_tile {
            if g.2 == 0 || g.1 != next {
                return None;
            }
            next = g.1.checked_add(g.2)?;
        }
        let group_tiles = vec![(Vec::new(), Vec::new()); groups.len()];
        (next == total && !groups.is_empty()).then_some(Levels { groups, group_tiles, mips })
    }

    /// The levels of group `k`.
    fn levels_of(&self, k: usize) -> std::ops::Range<u32> {
        self.groups[k].0..self.groups.get(k + 1).map_or(self.mips, |g| g.0)
    }

    /// The group level `level` is in (the packed levels share the last one).
    fn group_of(&self, level: u32) -> usize {
        self.groups.partition_point(|g| g.0 <= level).saturating_sub(1)
    }

    /// Total tiles and packed-tail start mip if at least one evictable top group is mapped.
    fn evictable_chain(&self) -> Option<(u32, u32)> {
        let last = self.groups.len().checked_sub(1)?;
        let mut tiles = 0u32;
        for g in &self.groups[..last] {
            if g.3 {
                tiles = tiles.saturating_add(g.2);
            }
        }
        (tiles > 0).then_some((tiles, self.groups[last].0))
    }
}

fn take_level_tiles(levels: Option<&mut Levels>) -> Vec<(Vec<usize>, Vec<(usize, usize)>)> {
    levels
        .map(|levels| levels.group_tiles.iter_mut().map(std::mem::take).collect())
        .unwrap_or_default()
}

/// `XCL_TEXTURE_LEVEL_PAGES=1` (off by default): a sampled 2D texture of one slice on the
/// shared pages gets the pages of a level when an upload first has texels for it. A retail game
/// places its streamed textures at full size and maps memory under their levels as it loads
/// them, the small ones first; with a page for every 64 KiB a texture covers most of the pages
/// are for levels that are not loaded.
pub(super) fn level_pages() -> bool {
    #[cfg(test)]
    if LEVEL_PAGES_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_TEXTURE_LEVEL_PAGES").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("D3D12.X: XCL_TEXTURE_LEVEL_PAGES=1 - a sampled texture's levels get their video memory pages when an upload first has texels for them");
        }
        on
    })
}

/// `level_pages` for a test (under `VRAM_TEST_LOCK`)
#[cfg(test)]
pub(super) static LEVEL_PAGES_FOR_TEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Of the textures with lazy levels there are now: how many, the tiles they stand for, and the
/// tiles of them that have pages; and the level groups given pages so far.
static LEVEL_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// (textures with lazy levels, the tiles they stand for, the tiles of them with pages, level
/// groups given pages so far)
fn level_stats() -> (u64, u64, u64, u64) {
    let l = |k: usize| LEVEL_STATS[k].load(std::sync::atomic::Ordering::Relaxed);
    (l(0), l(1), l(2), l(3))
}

/// Snapshot of placed sampled-texture residency and SRV mip coverage (`XCL_ACTIVITY`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TexelStats {
    pub sampled_textures: u64,
    pub srv_textures: u64,
    pub total_texels: u64,
    pub resident_texels: u64,
    pub srv_sampled_texels: u64,
    pub evicted_mips: u64,
    pub evicted_tiles: u64,
}

static TEXEL_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
static EVICT_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub fn texel_stats() -> TexelStats {
    let l = |k: usize| TEXEL_STATS[k].load(std::sync::atomic::Ordering::Relaxed);
    TexelStats {
        sampled_textures: l(0),
        srv_textures: l(1),
        total_texels: l(2),
        resident_texels: l(3),
        srv_sampled_texels: l(4),
        evicted_mips: EVICT_STATS[0].load(std::sync::atomic::Ordering::Relaxed),
        evicted_tiles: EVICT_STATS[1].load(std::sync::atomic::Ordering::Relaxed),
    }
}

fn budget_evict_pct() -> u64 {
    static PCT: OnceLock<u64> = OnceLock::new();
    *PCT.get_or_init(|| {
        std::env::var("XCL_TEXTURE_BUDGET_PCT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(85, |p| p.clamp(50, 98))
    })
}

fn budget_override_bytes() -> Option<u64> {
    static BYTES: OnceLock<Option<u64>> = OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("XCL_TEXTURE_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&mb| mb > 0)
            .map(|mb| mb << 20)
    })
}

fn mip_texels(width: u64, height: u32, depth: u16, dim: i32, mip: u32) -> u64 {
    let shift = mip.min(63);
    let w = (width >> shift).max(1);
    let slices = (depth as u64).max(1);
    if dim == D3D12_RESOURCE_DIMENSION_TEXTURE3D.0 {
        let h = ((height as u64) >> shift).max(1);
        let d = ((depth as u64) >> shift).max(1);
        w.saturating_mul(h).saturating_mul(d)
    } else if dim == D3D12_RESOURCE_DIMENSION_TEXTURE1D.0 {
        w.saturating_mul(slices)
    } else {
        let h = ((height as u64) >> shift).max(1);
        w.saturating_mul(h).saturating_mul(slices)
    }
}

fn range_texels(width: u64, height: u32, depth: u16, dim: i32, mips: std::ops::Range<u32>) -> u64 {
    mips.fold(0u64, |acc, m| acc.saturating_add(mip_texels(width, height, depth, dim, m)))
}

fn srv_mip_range(res_mips: u32, desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> std::ops::Range<u32> {
    let Some(v) = desc else { return 0..res_mips };
    // SAFETY: each branch reads only the union field named by `v.ViewDimension`.
    let (most_detailed, mip_levels, min_lod_clamp) = unsafe {
        match v.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE1D => {
                let s = &v.Anonymous.Texture1D;
                (s.MostDetailedMip, s.MipLevels, s.ResourceMinLODClamp)
            }
            D3D12_SRV_DIMENSION_TEXTURE1DARRAY => {
                let s = &v.Anonymous.Texture1DArray;
                (s.MostDetailedMip, s.MipLevels, s.ResourceMinLODClamp)
            }
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                let s = &v.Anonymous.Texture2D;
                (s.MostDetailedMip, s.MipLevels, s.ResourceMinLODClamp)
            }
            D3D12_SRV_DIMENSION_TEXTURE2DMS | D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => (0, 1, 0.0),
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                let s = &v.Anonymous.Texture2DArray;
                (s.MostDetailedMip, s.MipLevels, s.ResourceMinLODClamp)
            }
            D3D12_SRV_DIMENSION_TEXTURE3D => {
                let s = &v.Anonymous.Texture3D;
                (s.MostDetailedMip, s.MipLevels, s.ResourceMinLODClamp)
            }
            D3D12_SRV_DIMENSION_TEXTURECUBE => {
                let s = &v.Anonymous.TextureCube;
                (s.MostDetailedMip, s.MipLevels, s.ResourceMinLODClamp)
            }
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                let s = &v.Anonymous.TextureCubeArray;
                (s.MostDetailedMip, s.MipLevels, s.ResourceMinLODClamp)
            }
            _ => return 0..0,
        }
    };
    let end = if mip_levels == 0 || mip_levels == u32::MAX {
        res_mips
    } else {
        most_detailed.saturating_add(mip_levels).min(res_mips)
    };
    let clamp_mip = if min_lod_clamp.is_finite() && min_lod_clamp > 0.0 {
        min_lod_clamp.floor() as u32
    } else {
        0
    };
    let start = most_detailed.max(clamp_mip).min(end);
    start..end
}

fn srv_slice_range(res_slices: u32, dim: i32, desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> std::ops::Range<u32> {
    if dim == D3D12_RESOURCE_DIMENSION_TEXTURE3D.0 {
        return 0..1;
    }
    let limit = res_slices.max(1);
    let Some(v) = desc else { return 0..limit };
    // SAFETY: each branch reads only the union field named by `v.ViewDimension`.
    let (first, count) = unsafe {
        match v.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE1D | D3D12_SRV_DIMENSION_TEXTURE2D | D3D12_SRV_DIMENSION_TEXTURE2DMS => (0, 1),
            D3D12_SRV_DIMENSION_TEXTURE1DARRAY => {
                let s = &v.Anonymous.Texture1DArray;
                (s.FirstArraySlice, s.ArraySize)
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                let s = &v.Anonymous.Texture2DArray;
                (s.FirstArraySlice, s.ArraySize)
            }
            D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => {
                let s = &v.Anonymous.Texture2DMSArray;
                (s.FirstArraySlice, s.ArraySize)
            }
            D3D12_SRV_DIMENSION_TEXTURECUBE => (0, 6),
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                let s = &v.Anonymous.TextureCubeArray;
                (s.First2DArrayFace, s.NumCubes.saturating_mul(6))
            }
            D3D12_SRV_DIMENSION_TEXTURE3D => (0, 1),
            _ => return 0..0,
        }
    };
    let start = first.min(limit);
    start..first.saturating_add(count).min(limit)
}

/// Add `new` to sorted, disjoint half-open ranges and return how many indices were new.
fn union_range(ranges: &mut Vec<std::ops::Range<u32>>, new: std::ops::Range<u32>) -> u32 {
    if new.is_empty() {
        return 0;
    }
    let before = ranges.iter().map(|r| r.end - r.start).sum::<u32>();
    let first = ranges.partition_point(|range| range.end < new.start);
    let mut end = first;
    let mut merged = new;
    while end < ranges.len() && ranges[end].start <= merged.end {
        merged.start = merged.start.min(ranges[end].start);
        merged.end = merged.end.max(ranges[end].end);
        end += 1;
    }
    if first == end {
        ranges.insert(first, merged);
    } else {
        ranges[first] = merged;
        ranges.drain(first + 1..end);
    }
    let after = ranges.iter().map(|r| r.end - r.start).sum::<u32>();
    after - before
}

struct TexelMeta {
    width: u64,
    height: u32,
    depth: u16,
    mips: u16,
    dim: i32,
    total_texels: u64,
    resident_texels: u64,
    srv_slices: Vec<Vec<std::ops::Range<u32>>>,
    srv_sampled_texels: u64,
}

impl TexelMeta {
    fn from_desc(desc: &D3D12_RESOURCE_DESC, resident: bool) -> Self {
        let mips = desc.MipLevels.max(1);
        let total_texels = range_texels(desc.Width, desc.Height, desc.DepthOrArraySize, desc.Dimension.0, 0..mips as u32);
        TexelMeta {
            width: desc.Width,
            height: desc.Height,
            depth: desc.DepthOrArraySize,
            mips,
            dim: desc.Dimension.0,
            total_texels,
            resident_texels: if resident { total_texels } else { 0 },
            srv_slices: vec![Vec::new(); mips as usize],
            srv_sampled_texels: 0,
        }
    }

    fn record_srv(&mut self, desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> (bool, u64) {
        let first_srv = self.srv_sampled_texels == 0;
        let mips = srv_mip_range(self.mips as u32, desc);
        let slices = srv_slice_range(self.depth as u32, self.dim, desc);
        let mut added = 0u64;
        for mip in mips {
            let new_slices = union_range(&mut self.srv_slices[mip as usize], slices.clone()) as u64;
            if new_slices == 0 {
                continue;
            }
            let per_slice = if self.dim == D3D12_RESOURCE_DIMENSION_TEXTURE3D.0 {
                mip_texels(self.width, self.height, self.depth, self.dim, mip)
            } else {
                mip_texels(self.width, self.height, 1, self.dim, mip)
            };
            added = added.saturating_add(per_slice.saturating_mul(new_slices));
        }
        self.srv_sampled_texels = self.srv_sampled_texels.saturating_add(added);
        (first_srv && added != 0, added)
    }
}

fn record_texel_insert(texels: &Option<TexelMeta>) {
    let Some(tm) = texels.as_ref() else { return };
    TEXEL_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if tm.total_texels > 0 {
        TEXEL_STATS[2].fetch_add(tm.total_texels, std::sync::atomic::Ordering::Relaxed);
    }
    if tm.resident_texels > 0 {
        TEXEL_STATS[3].fetch_add(tm.resident_texels, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Levels {
    /// Its tiles that have pages.
    fn mapped_tiles(&self) -> u64 {
        self.groups.iter().filter(|g| g.3).map(|g| g.2 as u64).sum()
    }
}

/// The tiling of reserved texture `res` of `mips` levels as `Levels`, and its tiles in all.
unsafe fn level_tiling(dev: &ID3D12Device, res: &ID3D12Resource, mips: u32) -> (Option<Levels>, u32) {
    let mut total = 0u32;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut n = mips;
    let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); mips as usize];
    dev.GetResourceTiling(res, Some(&mut total), Some(&mut packed), None, Some(&mut n), 0, tilings.as_mut_ptr());
    let standard: Vec<(u32, u32)> = tilings
        .iter()
        .take((packed.NumStandardMips as usize).min(n as usize))
        .map(|t| (t.StartTileIndexInOverallResource, t.WidthInTiles * t.HeightInTiles as u32 * t.DepthInTiles as u32))
        .collect();
    let tail = (packed.NumPackedMips > 0).then_some((packed.StartTileIndexInOverallResource, packed.NumTilesForPackedMips));
    let whole = standard.len() == packed.NumStandardMips as usize && packed.NumStandardMips as u32 + packed.NumPackedMips as u32 == mips;
    (if whole { Levels::new(&standard, tail, total, mips) } else { None }, total)
}

/// Per-heap free-tile allocator for `Pages`.
///
/// Keeping free tiles partitioned by pool heap (sorted descending so `.pop()` yields ascending
/// contiguous tile indices) prevents streaming churn across 100+ 64 MiB pools from scattering a
/// single texture across dozens of heaps — which otherwise explodes `Pages::map` into dozens of
/// `UpdateTileMappings` calls per texture.
struct TilePool {
    /// Free tile indices for each pool heap (descending when `!dirty_sort[h]`).
    free_by_heap: Vec<Vec<u32>>,
    /// Whether `free_by_heap[h]` must be sorted descending before popping.
    dirty_sort: Vec<bool>,
    /// Fresh-allocation packing class for heap `h`. Shared address pages may give either class
    /// live owners here, so this is never proof that a heap is safe to evict.
    heap_evictable: Vec<bool>,
    /// Total free tiles currently available across all heaps.
    free_total: usize,
    /// Tiles given back, reusable after `COOL`.
    cooling: std::collections::VecDeque<(Tile, std::time::Instant)>,
}

impl TilePool {
    fn new() -> Self {
        TilePool {
            free_by_heap: Vec::new(),
            dirty_sort: Vec::new(),
            heap_evictable: Vec::new(),
            free_total: 0,
            cooling: std::collections::VecDeque::new(),
        }
    }

    fn add_heap_at(&mut self, slot: u32, pool_pages: u32, evictable: bool) {
        let idx = slot as usize;
        if idx >= self.free_by_heap.len() {
            self.free_by_heap.resize_with(idx + 1, Vec::new);
            self.dirty_sort.resize(idx + 1, false);
            self.heap_evictable.resize(idx + 1, false);
        }
        debug_assert!(self.free_by_heap[idx].is_empty());
        self.free_by_heap[idx] = (0..pool_pages).rev().collect();
        self.dirty_sort[idx] = false;
        self.heap_evictable[idx] = evictable;
        self.free_total += pool_pages as usize;
    }

    /// Reclaim completely empty cooled heaps (`free_by_heap[h].len() == pool_pages`), keeping up
    /// to `keep_spare` empty heaps of each class warm so steady-state streaming never thrashes `CreateHeap`.
    /// Returns the reclaimed heap indices so the caller can drop their `ID3D12Heap`s.
    fn reclaim_empty_heaps(&mut self, keep_spare: usize, pool_pages: u32) -> Vec<u32> {
        let full = pool_pages as usize;
        let mut kept = [0usize; 2];
        let mut reclaimed = Vec::new();
        for h in 0..self.free_by_heap.len() {
            if self.free_by_heap[h].len() != full {
                continue;
            }
            let class = self.heap_evictable[h] as usize;
            if kept[class] < keep_spare {
                kept[class] += 1;
                continue;
            }
            self.free_by_heap[h].clear();
            self.free_by_heap[h].shrink_to_fit();
            self.dirty_sort[h] = false;
            self.heap_evictable[h] = false;
            self.free_total -= full;
            reclaimed.push(h as u32);
        }
        reclaimed
    }

    fn push_cooling(&mut self, tile: Tile, now: std::time::Instant) {
        self.cooling.push_back((tile, now));
    }

    /// Move every cooled tile (`>= COOL`) into its pool heap's free list so best-fit allocation
    /// sees all ready tiles at once.
    fn drain_cooling(&mut self, now: std::time::Instant) {
        while self.cooling.front().is_some_and(|&(_, at)| now.duration_since(at) >= COOL) {
            let Some(((h, t), _)) = self.cooling.pop_front() else { break };
            let idx = h as usize;
            if idx < self.free_by_heap.len() {
                self.free_by_heap[idx].push(t);
                self.dirty_sort[idx] = true;
                self.free_total += 1;
            }
        }
    }

    fn return_tiles(&mut self, tiles: &[Tile]) {
        for &(h, t) in tiles {
            let idx = h as usize;
            if idx < self.free_by_heap.len() {
                self.free_by_heap[idx].push(t);
                self.dirty_sort[idx] = true;
                self.free_total += 1;
            }
        }
    }

    /// Pick a pool heap in the requested allocation class to take up to `rem` tiles from:
    /// 1. `preferred` if it fits all `rem` tiles (keeps partial-overlap placements on 1 heap),
    /// 2. otherwise best-fit heap with `len >= rem` (fits the whole texture in 1 heap without
    ///    splitting fuller pools),
    /// 3. otherwise the heap with the most free tiles (minimizing distinct heaps touched).
    fn pick_heap(&self, rem: usize, preferred: Option<u32>, evictable: bool) -> Option<usize> {
        if self.free_total == 0 {
            return None;
        }
        if let Some(p) = preferred
            .map(|h| h as usize)
            .filter(|&h| h < self.free_by_heap.len() && self.heap_evictable.get(h).copied() == Some(evictable))
        {
            if self.free_by_heap[p].len() >= rem {
                return Some(p);
            }
        }
        let mut best_fit: Option<(usize, usize)> = None;
        let mut largest: Option<(usize, usize)> = None;
        for (h, tiles) in self.free_by_heap.iter().enumerate() {
            if self.heap_evictable.get(h).copied() != Some(evictable) {
                continue;
            }
            let len = tiles.len();
            if len == 0 {
                continue;
            }
            if len >= rem {
                if best_fit.is_none_or(|(_, blen)| len < blen) {
                    best_fit = Some((h, len));
                }
            } else if largest.is_none_or(|(lh, llen)| len > llen || (len == llen && preferred == Some(h as u32) && preferred != Some(lh as u32))) {
                largest = Some((h, len));
            }
        }
        best_fit.or(largest).map(|(h, _)| h)
    }

    /// Allocate `count` tiles in one allocation class, preferring a single heap and ascending
    /// contiguous offsets. `make_heap` allocates a new backing pool heap when the class has no
    /// free tiles. On failure, any tiles already taken are returned untouched.
    fn take_batch(
        &mut self,
        count: usize,
        preferred: Option<u32>,
        evictable: bool,
        now: std::time::Instant,
        pool_pages: u32,
        mut make_heap: impl FnMut() -> Option<u32>,
    ) -> Option<Vec<Tile>> {
        if count == 0 {
            return Some(Vec::new());
        }
        self.drain_cooling(now);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let rem = count - out.len();
            let h = match self.pick_heap(rem, preferred, evictable) {
                Some(h) => h,
                None => {
                    let Some(new_h) = make_heap() else {
                        self.return_tiles(&out);
                        return None;
                    };
                    self.add_heap_at(new_h, pool_pages, evictable);
                    new_h as usize
                }
            };
            if self.dirty_sort[h] {
                self.free_by_heap[h].sort_unstable_by(|a, b| b.cmp(a));
                self.dirty_sort[h] = false;
            }
            let free = &mut self.free_by_heap[h];
            let take = rem.min(free.len());
            for _ in 0..take {
                let t = free.pop().unwrap();
                out.push((h as u32, t));
            }
            self.free_total -= take;
        }
        Some(out)
    }

    fn mark_heap_pinned(&mut self, slot: u32) {
        if let Some(flag) = self.heap_evictable.get_mut(slot as usize) {
            *flag = false;
        }
    }
}

/// Allocate one 64 MiB default heap for `Pages`.
unsafe fn alloc_pool_heap(dev: &ID3D12Device) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        Alignment: 0,
        // every kind of texture (resource heap tier 2: the only flags for both)
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&desc, &mut heap)) {
        crate::xlog_warn!("D3D12.X: no video memory for more pages of title textures ({e})");
        return None;
    }
    heap
}

/// Backing 64 MiB default heaps for `Pages`, supporting slot reuse when completely empty heaps
/// are reclaimed back to local VRAM. `len()` reports the count of live heaps.
struct PoolHeaps {
    slots: Vec<Option<ID3D12Heap>>,
    free_slots: Vec<u32>,
    live: usize,
}

impl PoolHeaps {
    fn new() -> Self {
        Self { slots: Vec::new(), free_slots: Vec::new(), live: 0 }
    }

    fn len(&self) -> usize {
        self.live
    }

    fn push_heap(&mut self, heap: ID3D12Heap) -> u32 {
        self.live += 1;
        if let Some(idx) = self.free_slots.pop() {
            debug_assert!(self.slots[idx as usize].is_none());
            self.slots[idx as usize] = Some(heap);
            idx
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Some(heap));
            idx
        }
    }

    fn take_heap(&mut self, idx: u32) -> Option<ID3D12Heap> {
        let h = self.slots.get_mut(idx as usize)?.take()?;
        self.live -= 1;
        self.free_slots.push(idx);
        Some(h)
    }
}

impl std::ops::Index<usize> for PoolHeaps {
    type Output = ID3D12Heap;
    fn index(&self, idx: usize) -> &Self::Output {
        self.slots[idx].as_ref().expect("live VRAM pool heap")
    }
}

fn keep_spare_pools() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("XCL_VRAM_KEEP_SPARE_POOLS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1)
    })
}

/// A superseded texture whose NULL mapping is held until work submitted before it has finished.
struct HeldUnmap {
    res: ID3D12Resource,
    total: u32,
    used: Vec<usize>,
    used_extra: Vec<(usize, usize)>,
    level_tiles: Vec<(Vec<usize>, Vec<(usize, usize)>)>,
    /// Completion marks for every relevant title or upload queue.
    marks: Vec<(ID3D12Fence, u64)>,
}

fn all_marks_complete<F>(marks: &[(F, u64)], mut completed: impl FnMut(&F) -> u64) -> bool {
    marks.iter().all(|(fence, value)| completed(fence) >= *value)
}

/// The title's pages of video memory for tiled textures (see the module's comment).
struct Pages {
    queue: ID3D12CommandQueue,
    /// signalled after each mapping; title queues wait for it before their next work
    fence: ID3D12Fence,
    value: u64,
    /// per-device timestamp of the last periodic reclaim check
    last_reclaim: std::time::Instant,
    pools: PoolHeaps,
    pool: TilePool,
    /// title page address → its video memory
    pages: std::collections::HashMap<usize, Page>,
    /// extra / sub-64 KiB tiles keyed by `(va, tile_index)` so repeated placements at the same
    /// address share them without aliasing unrelated addresses
    extra_pages: std::collections::HashMap<(usize, usize), Page>,
    /// pages of both maps whose last user went, oldest first, with when (`sweep_idle`)
    idle: std::collections::VecDeque<(PageKey, std::time::Instant)>,
    /// resource → its mapping metadata and currently held pages / extra keys
    targets: std::collections::HashMap<usize, Target>,
    /// queue → the fence value it last waited for
    waited: std::collections::HashMap<usize, u64>,
    /// queue → reusable completion fence and its last signalled value
    unmap_fences: std::collections::HashMap<usize, (ID3D12Fence, u64)>,
    /// Superseded mappings waiting for all pre-existing queue work to finish.
    held_unmaps: Vec<HeldUnmap>,
    /// cached `reservable()` answers for `(DXGI_FORMAT, SampleCount)` on this device
    reservable_cache: std::collections::HashMap<(i32, u32), bool>,
    /// reusable scratch buffers for `map()`
    scratch_heaps: Vec<u32>,
    scratch_flags: Vec<D3D12_TILE_RANGE_FLAGS>,
    scratch_offsets: Vec<u32>,
    scratch_counts: Vec<u32>,
}

// D3D12 objects are free-threaded
unsafe impl Send for Pages {}

/// device raw pointer → its pages (None inside: tiled tier < 2 on this device)
static PAGES: Mutex<std::collections::BTreeMap<usize, Option<Pages>>> = Mutex::new(std::collections::BTreeMap::new());

struct BeforeExecuteQueue<D> {
    serial: Mutex<()>,
    /// A queue's device is immutable. Cache only a successful, non-null lookup; failures retry.
    device: OnceLock<D>,
}

impl<D> Default for BeforeExecuteQueue<D> {
    fn default() -> Self {
        Self { serial: Mutex::new(()), device: OnceLock::new() }
    }
}

impl<D> BeforeExecuteQueue<D> {
    /// Called with `serial` held, so the first successful lookup is also the only successful lookup.
    fn cached_or_try_device(&self, lookup: impl FnOnce() -> Option<D>) -> Option<&D> {
        if let Some(device) = self.device.get() {
            return Some(device);
        }
        let device = lookup()?;
        let _ = self.device.set(device);
        self.device.get()
    }
}

type BeforeExecuteGate = BeforeExecuteQueue<ID3D12Device>;

fn wait_gates() -> &'static Mutex<HashMap<usize, Arc<BeforeExecuteGate>>> {
    static GATES: OnceLock<Mutex<HashMap<usize, Arc<BeforeExecuteGate>>>> = OnceLock::new();
    GATES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn remove_wait_gate<D>(gates: &Mutex<HashMap<usize, Arc<BeforeExecuteQueue<D>>>>, queue: usize) -> Option<Arc<BeforeExecuteQueue<D>>> {
    gates.lock().unwrap().remove(&queue)
}

/// CPU time inside the residency work one `XCL_ACTIVITY` look caused. The look owns this
/// thread-local accumulator; internal uploads on the same thread join the same sample without
/// adding shared counters to the submission path.
#[derive(Default)]
pub(super) struct BeforeExecuteTiming {
    pub gate_wait: std::time::Duration,
    pub device: std::time::Duration,
    pub unmap_poll: std::time::Duration,
    pub state: std::time::Duration,
    pub queue_wait: std::time::Duration,
    pub retirement: std::time::Duration,
    pub calls: usize,
    pub device_cache_hits: usize,
    pub device_queries: usize,
    pub mapping_waits: usize,
    pub reclaim_passes: usize,
    pub retirement_passes: usize,
}

thread_local! {
    static BEFORE_EXECUTE_TIMING: std::cell::RefCell<Option<BeforeExecuteTiming>> = const { std::cell::RefCell::new(None) };
}

/// Begin an activity look's residency measurement. Retried sparse preparation reaches this more
/// than once, so an already-active accumulator is preserved until `take_before_execute_timing`.
pub(super) fn begin_before_execute_timing() {
    BEFORE_EXECUTE_TIMING.with(|timing| {
        let mut timing = timing.borrow_mut();
        if timing.is_none() {
            *timing = Some(BeforeExecuteTiming::default());
        }
    });
}

pub(super) fn take_before_execute_timing() -> Option<BeforeExecuteTiming> {
    BEFORE_EXECUTE_TIMING.with(|timing| timing.borrow_mut().take())
}

fn before_execute_timing_active() -> bool {
    BEFORE_EXECUTE_TIMING.with(|timing| timing.borrow().is_some())
}

fn record_before_execute_timing(measured: BeforeExecuteTiming) {
    BEFORE_EXECUTE_TIMING.with(|timing| {
        let mut timing = timing.borrow_mut();
        let Some(total) = timing.as_mut() else { return };
        total.gate_wait += measured.gate_wait;
        total.device += measured.device;
        total.unmap_poll += measured.unmap_poll;
        total.state += measured.state;
        total.queue_wait += measured.queue_wait;
        total.retirement += measured.retirement;
        total.calls = total.calls.saturating_add(measured.calls);
        total.device_cache_hits = total.device_cache_hits.saturating_add(measured.device_cache_hits);
        total.device_queries = total.device_queries.saturating_add(measured.device_queries);
        total.mapping_waits = total.mapping_waits.saturating_add(measured.mapping_waits);
        total.reclaim_passes = total.reclaim_passes.saturating_add(measured.reclaim_passes);
        total.retirement_passes = total.retirement_passes.saturating_add(measured.retirement_passes);
    });
}

fn reserve_page_wait(waited: &mut HashMap<usize, u64>, queue: usize, value: u64) -> bool {
    if value == 0 || waited.get(&queue).copied().unwrap_or(0) >= value {
        return false;
    }
    waited.insert(queue, value);
    true
}

impl Pages {
    unsafe fn new(dev: &ID3D12Device) -> Option<Pages> {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        // tier 2: unmapped tiles read as zero, writes to them go nowhere
        if o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            crate::xlog_info!("D3D12.X: tiled resources tier {}: GPU-owned textures in arenas", o.TiledResourcesTier.0);
            return None;
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        let _ = queue.SetName(windows::core::w!("xcl vram tile-mapping queue"));
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        crate::xlog_info!("D3D12.X: textures in title memory: tiled resources on 64 KiB pages of video memory, one per page of the title's");
        Some(Pages {
            queue,
            fence,
            value: 0,
            last_reclaim: std::time::Instant::now(),
            pools: PoolHeaps::new(),
            pool: TilePool::new(),
            pages: std::collections::HashMap::new(),
            extra_pages: std::collections::HashMap::new(),
            idle: std::collections::VecDeque::new(),
            targets: std::collections::HashMap::new(),
            waited: std::collections::HashMap::new(),
            unmap_fences: std::collections::HashMap::new(),
            held_unmaps: Vec::new(),
            reservable_cache: std::collections::HashMap::new(),
            scratch_heaps: Vec::new(),
            scratch_flags: Vec::new(),
            scratch_offsets: Vec::new(),
            scratch_counts: Vec::new(),
        })
    }

    /// Pages no resource used for `IDLE` go back to the pools.
    fn sweep(&mut self, now: std::time::Instant) {
        sweep_idle(&mut self.idle, &mut self.pages, &mut self.extra_pages, &mut self.pool.cooling, now);
    }

    /// Sweep expired idle pages, drain cooled tiles, and reclaim any completely empty 64 MiB
    /// pool heaps (above the warm spare count), returning their `ID3D12Heap`s to be dropped
    /// outside `PAGES.lock()`.
    fn sweep_and_reclaim(&mut self, now: std::time::Instant) -> Vec<ID3D12Heap> {
        self.sweep(now);
        self.pool.drain_cooling(now);
        let reclaimed_ids = self.pool.reclaim_empty_heaps(keep_spare_pools(), POOL_PAGES);
        let mut dropped = Vec::with_capacity(reclaimed_ids.len());
        for idx in reclaimed_ids {
            if let Some(h) = self.pools.take_heap(idx) {
                dropped.push(h);
            }
        }
        if !dropped.is_empty() {
            let mb = (dropped.len() * POOL_PAGES as usize * PAGE) >> 20;
            crate::xlog_info!(
                "D3D12.X: reclaimed {} empty 64 MiB VRAM pool heap(s) ({mb} MiB returned to GPU; {} pools active)",
                dropped.len(),
                self.pools.len()
            );
        }
        dropped
    }

    /// Map every tile of `res` (`tiles[i]`: overall tile i) – one call per pool heap.
    /// `tiles` for the tiles of `res` from the first one of subresource `sub` on.
    unsafe fn map(&mut self, res: &ID3D12Resource, sub: u32, tiles: &[Tile]) {
        if tiles.is_empty() {
            return;
        }
        self.scratch_heaps.clear();
        self.scratch_heaps.extend(tiles.iter().map(|t| t.0));
        self.scratch_heaps.sort_unstable();
        self.scratch_heaps.dedup();
        for i in 0..self.scratch_heaps.len() {
            let h = self.scratch_heaps[i];
            let ranges = heap_ranges_trimmed(h, tiles);
            if ranges.is_empty() {
                continue;
            }
            self.scratch_flags.clear();
            self.scratch_offsets.clear();
            self.scratch_counts.clear();
            let mut num_tiles = 0u32;
            for &(mapped, off, count) in &ranges {
                self.scratch_flags.push(if mapped { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP });
                self.scratch_offsets.push(off);
                self.scratch_counts.push(count);
                num_tiles += count;
            }
            let coord = D3D12_TILED_RESOURCE_COORDINATE { Subresource: sub, ..Default::default() };
            let region = D3D12_TILE_REGION_SIZE { NumTiles: num_tiles, UseBox: false.into(), ..Default::default() };
            self.queue.UpdateTileMappings(
                res,
                1,
                Some(&coord),
                Some(&region),
                &self.pools[h as usize],
                ranges.len() as u32,
                Some(self.scratch_flags.as_ptr()),
                Some(self.scratch_offsets.as_ptr()),
                Some(self.scratch_counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        self.value += 1;
        let _ = super::real::native_queue_signal(&self.queue, &self.fence, self.value);
    }

    /// Unmap every tile of `res` (`D3D12_TILE_RANGE_FLAG_NULL`) so its unshared tiles can return
    /// to the pool even while the title keeps the COM object alive.
    unsafe fn unmap(&mut self, res: &ID3D12Resource, total: u32) {
        if total == 0 {
            return;
        }
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let region = D3D12_TILE_REGION_SIZE { NumTiles: total, UseBox: false.into(), ..Default::default() };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        self.queue.UpdateTileMappings(
            res,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&flag),
            None,
            Some(&total),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        self.value += 1;
        let _ = super::real::native_queue_signal(&self.queue, &self.fence, self.value);
    }

    /// Unmap `num_tiles` starting at subresource `sub` (`D3D12_TILE_RANGE_FLAG_NULL`) for top-mip eviction.
    unsafe fn unmap_range(&mut self, res: &ID3D12Resource, sub: u32, num_tiles: u32) {
        if num_tiles == 0 {
            return;
        }
        let coord = D3D12_TILED_RESOURCE_COORDINATE { Subresource: sub, ..Default::default() };
        let region = D3D12_TILE_REGION_SIZE { NumTiles: num_tiles, UseBox: false.into(), ..Default::default() };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        self.queue.UpdateTileMappings(
            res,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&flag),
            None,
            Some(&num_tiles),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        self.value += 1;
        let _ = super::real::native_queue_signal(&self.queue, &self.fence, self.value);
    }

    /// Mark work submitted so far on every active queue. Retirement marks cover queues which
    /// stopped accepting signals before this snapshot. Failure is closed: the mapping stays live.
    unsafe fn mark_unmap(
        &mut self,
        dev: &ID3D12Device,
        queues: &[ID3D12CommandQueue],
        retiring: &[(ID3D12Fence, u64)],
        blocked: bool,
    ) -> Option<(Vec<(ID3D12Fence, u64)>, Vec<(ID3D12CommandQueue, ID3D12Fence, u64)>)> {
        if blocked {
            return None;
        }
        let mut marks = Vec::with_capacity(retiring.len() + queues.len());
        let mut signals = Vec::with_capacity(queues.len());
        for (fence, value) in retiring {
            let mut fence_dev: Option<ID3D12Device> = None;
            fence.GetDevice(&mut fence_dev).ok()?;
            if fence_dev.as_ref().is_some_and(|owner| owner.as_raw() == dev.as_raw()) {
                marks.push((fence.clone(), *value));
            }
        }
        let mut seen = HashSet::new();
        for queue in queues {
            let key = queue.as_raw() as usize;
            if !seen.insert(key) {
                continue;
            }
            let mut queue_dev: Option<ID3D12Device> = None;
            if queue.GetDevice(&mut queue_dev).is_err() || queue_dev.as_ref().is_none_or(|d| d.as_raw() != dev.as_raw()) {
                continue;
            }
            let entry = match self.unmap_fences.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let fence = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok()?;
                    entry.insert((fence, 0))
                }
            };
            let next = entry.1 + 1;
            entry.1 = next;
            marks.push((entry.0.clone(), next));
            signals.push((queue.clone(), entry.0.clone(), next));
        }
        Some((marks, signals))
    }

    /// Strong fence owners to poll without holding `PAGES`.
    fn held_unmap_fences(&self) -> Vec<ID3D12Fence> {
        self.held_unmaps.iter().flat_map(|held| held.marks.iter().map(|(fence, _)| fence.clone())).collect()
    }

    /// Submit completed NULL mappings and return their resource owners for release after unlock.
    unsafe fn release_held_unmaps(
        &mut self,
        now: std::time::Instant,
        mut completed: impl FnMut(&ID3D12Fence) -> u64,
    ) -> Vec<ID3D12Resource> {
        let mut done = Vec::new();
        for held in std::mem::take(&mut self.held_unmaps) {
            if all_marks_complete(&held.marks, &mut completed) {
                let key = held.res.as_raw() as usize;
                if let Some(levels) = self.targets.get_mut(&key).and_then(|target| target.levels.as_mut()) {
                    LEVEL_STATS[2].fetch_sub(levels.mapped_tiles(), std::sync::atomic::Ordering::Relaxed);
                    levels.groups.iter_mut().for_each(|group| group.3 = false);
                }
                if let Some(tm) = self.targets.get_mut(&key).and_then(|target| target.texels.as_mut()) {
                    if tm.resident_texels > 0 {
                        TEXEL_STATS[3].fetch_sub(tm.resident_texels, std::sync::atomic::Ordering::Relaxed);
                        tm.resident_texels = 0;
                    }
                }
                self.unmap(&held.res, held.total);
                self.release_tiles(held.used, held.used_extra, now, true);
                for (used, used_extra) in held.level_tiles {
                    self.release_tiles(used, used_extra, now, true);
                }
                done.push(held.res);
            } else {
                self.held_unmaps.push(held);
            }
        }
        done
    }

    fn release_tiles(&mut self, used: Vec<usize>, used_extra: Vec<(usize, usize)>, now: std::time::Instant, immediate_cool: bool) {
        for a in used {
            if let Some(p) = self.pages.get_mut(&a) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.pages.remove(&a) {
                            self.pool.push_cooling(p.tile, now);
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Own(a), now));
                    }
                }
            }
        }
        for k in used_extra {
            if let Some(p) = self.extra_pages.get_mut(&k) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.extra_pages.remove(&k) {
                            self.pool.push_cooling(p.tile, now);
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Extra(k.0, k.1), now));
                    }
                }
            }
        }
    }

    /// Acquire/share all `total` tiles for `res` at `va` and map them on the `Pages` queue.
    unsafe fn bind(&mut self, dev: &ID3D12Device, res: &ID3D12Resource, va: usize, console_bytes: Option<u64>, total: u32) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        self.bind_range(dev, res, va, console_bytes, total, 0, 0..total as usize, false)
    }

    /// `bind` for the tiles `range` of the resource's `total`, which start at subresource `sub`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn bind_range(
        &mut self, dev: &ID3D12Device, res: &ID3D12Resource, va: usize,
        console_bytes: Option<u64>, total: u32, sub: u32, range: std::ops::Range<usize>, evictable: bool,
    ) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        let (first, own) = (total > 0).then(|| owned_pages(va, console_bytes, total as usize)).flatten()?;
        let now = std::time::Instant::now();
        self.sweep(now);

        let mut missing = 0usize;
        let mut preferred: Option<u32> = None;
        for i in range.clone() {
            let existing = if i < own {
                self.pages.get(&(first + i * PAGE)).map(|p| p.tile.0)
            } else {
                self.extra_pages.get(&(va, i)).map(|p| p.tile.0)
            };
            match existing {
                Some(h) => {
                    if !evictable {
                        self.pool.mark_heap_pinned(h);
                    }
                    if preferred.is_none() {
                        preferred = Some(h);
                    }
                }
                None => missing += 1,
            }
        }

        let pools = &mut self.pools;
        let pool = &mut self.pool;
        let fresh = pool.take_batch(missing, preferred, evictable, now, POOL_PAGES, || {
            let heap = alloc_pool_heap(dev)?;
            Some(pools.push_heap(heap))
        })?;
        let mut fresh_iter = fresh.into_iter();

        let mut tiles = Vec::with_capacity(range.len());
        let mut used = Vec::with_capacity(own.min(range.len()));
        let mut used_extra = Vec::new();
        for i in range {
            let tile = if i < own {
                let a = first + i * PAGE;
                let t = match self.pages.get_mut(&a) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        p.tile
                    }
                    None => {
                        let t = fresh_iter.next().unwrap();
                        self.pages.insert(a, Page { tile: t, users: 1, idle_since: None });
                        t
                    }
                };
                used.push(a);
                t
            } else {
                let k = (va, i);
                let t = match self.extra_pages.get_mut(&k) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        p.tile
                    }
                    None => {
                        let t = fresh_iter.next().unwrap();
                        self.extra_pages.insert(k, Page { tile: t, users: 1, idle_since: None });
                        t
                    }
                };
                used_extra.push(k);
                t
            };
            tiles.push(tile);
        }
        self.map(res, sub, &tiles);
        Some((used, used_extra))
    }
}

/// Poll a snapshot without `PAGES`; a held unmap added meanwhile has no observation and stays held.
unsafe fn held_unmap_completions(dev_key: usize) -> Vec<(ID3D12Fence, u64)> {
    let mut fences = {
        let pages = PAGES.lock().unwrap();
        pages.get(&dev_key).and_then(Option::as_ref).map(Pages::held_unmap_fences).unwrap_or_default()
    };
    fences.sort_unstable_by_key(|fence| fence.as_raw() as usize);
    fences.dedup_by_key(|fence| fence.as_raw() as usize);
    fences.into_iter().map(|fence| {
        let value = fence.GetCompletedValue();
        (fence, value)
    }).collect()
}

/// The ranges of one UpdateTileMappings call for pool heap `heap` over a resource whose overall
/// tile i goes to `tiles[i]`: (mapped, tile offset in the heap, count) – runs of consecutive
/// tiles of the heap, and skipped runs for tiles of other heaps.
fn heap_ranges(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = Vec::new();
    for &(h, t) in tiles {
        let mine = h == heap;
        match out.last_mut() {
            Some(r) if mine && r.0 && r.1 + r.2 == t => r.2 += 1,
            Some(r) if !mine && !r.0 => r.2 += 1,
            _ => out.push((mine, if mine { t } else { 0 }, 1)),
        }
    }
    out
}

/// Like `heap_ranges`, with any trailing `SKIP` range trimmed so `UpdateTileMappings` only walks
/// up to the last tile belonging to `heap`.
fn heap_ranges_trimmed(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out = heap_ranges(heap, tiles);
    while out.last().is_some_and(|r| !r.0) {
        out.pop();
    }
    out
}

/// 64 KiB page boundaries `P` with `va <= P < va + bytes`, capped at `total` tiles:
/// `(first_page, count)`. Because every `P` lies strictly inside `[va, va + bytes)`, two
/// non-overlapping ranges never share a page here even when `va` is only 4 KiB aligned.
fn owned_pages(va: usize, console_bytes: Option<u64>, total: usize) -> Option<(usize, usize)> {
    match console_bytes {
        Some(b) => {
            let bytes = usize::try_from(b).ok()?;
            let first = va.checked_next_multiple_of(PAGE)?;
            let end = va.saturating_add(bytes);
            let n = if first < end { (end - first).div_ceil(PAGE) } else { 0 };
            Some((first, n.min(total)))
        }
        None => (va % PAGE == 0).then_some((va, total)),
    }
}

/// A reserved texture needs TILED in the format's support (`support2`) and, multisampled, a
/// tiled quality level at its sample count.
fn can_reserve(support2: i32, samples: u32, tiled_msaa_levels: u32) -> bool {
    support2 & D3D12_FORMAT_SUPPORT2_TILED.0 != 0 && (samples <= 1 || tiled_msaa_levels > 0)
}

/// Whether the device can make `desc` a reserved texture, asked before CreateReservedResource:
/// refused, it is an error in the debug layer (#940: R32G8X24_TYPELESS depth, 1 and 4 samples).
/// A query that fails says yes, and creation decides.
unsafe fn reservable(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    let mut f = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: desc.Format, ..Default::default() };
    if dev.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &mut f as *mut _ as *mut c_void, std::mem::size_of_val(&f) as u32).is_err() {
        return true;
    }
    // a format without TILED is not reservable whatever the sample count's query says
    if !can_reserve(f.Support2.0, 1, 0) {
        return false;
    }
    let samples = desc.SampleDesc.Count;
    let mut m = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: desc.Format, SampleCount: samples, Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE, NumQualityLevels: 0 };
    if samples > 1 && dev.CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &mut m as *mut _ as *mut c_void, std::mem::size_of_val(&m) as u32).is_err() {
        return true;
    }
    can_reserve(f.Support2.0, samples, m.NumQualityLevels)
}

/// Canonical physical-backing address for `va`: when `va` lies inside an unmapped GPU mirror view
/// of title CPU memory (`kernelx::mirrors::source`), map both views to the same backing address so
/// shared 64 KiB VRAM pages and arenas alias instead of duplicating video memory. If distinct
/// physical or committed pages are mapped directly at `va`, keep `va` unaliased (matching
/// `title_textures::mirror_source`).
fn canonical_va(va: usize) -> usize {
    let Some(src) = crate::hle::kernelx::mirrors::source(va) else { return va };
    if crate::hle::kernelx::physical_page_at(va) {
        return va;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let mapped = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0 && m.State == MEM_COMMIT;
    if mapped { va } else { src }
}

/// A texture at `va` as a reserved resource on the title's shared 64 KiB pages
/// (`console_bytes`: its size on the console; tiles past the owned page boundaries are shared by
/// `(va, tile_index)` so repeated placements at the same address do not leak). None: it cannot be.
#[allow(clippy::too_many_arguments)]
unsafe fn place_on_pages(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    require_aligned: bool,
    lazy: bool,
    track_levels: bool,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let va = canonical_va(va);
    if desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D {
        return None;
    }
    if (require_aligned || console_bytes.is_none()) && va % PAGE != 0 {
        return None;
    }
    // reserved textures have the PC's 64 KiB tiles
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let log = |why: &str| {
        if require_aligned && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_info!("D3D12.X: GPU-owned texture at {va:#x} ({}x{}, format {}, {} samples) not reserved ({why}): in an arena", desc.Width, desc.Height, desc.Format.0, desc.SampleDesc.Count);
        }
    };
    let dev_key = dev.as_raw() as usize;
    let is_reservable = {
        let mut g = PAGES.lock().unwrap();
        let pages = g.entry(dev_key).or_insert_with(|| Pages::new(dev)).as_mut()?;
        *pages.reservable_cache.entry((tiled.Format.0, tiled.SampleDesc.Count)).or_insert_with(|| reservable(dev, &tiled))
    };
    if !is_reservable {
        log("no tiled support for its format or sample count");
        return None;
    }
    // CreateReservedResource and GetResourceTiling do not touch `PAGES`; keep them outside the
    // lock so concurrent streaming threads, `before_execute`, and `page_bytes` do not serialize.
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        log(&format!("{hr:#x}"));
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mut total = 0u32;
    let levels = if lazy || track_levels {
        let (levels, n) = level_tiling(dev, res, desc.MipLevels as u32);
        total = n;
        levels.filter(|_| total > 0 && owned_pages(va, console_bytes, total as usize).is_some())
    } else {
        dev.GetResourceTiling(res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
        None
    };
    let eager_levels = levels.is_some() && !lazy;
    let texels = (!require_aligned && super::memory_report::enabled()).then(|| TexelMeta::from_desc(desc, levels.is_none() || eager_levels));
    let now = std::time::Instant::now();
    let bound = {
        let mut g = PAGES.lock().unwrap();
        if let Some(Some(pages)) = g.get_mut(&dev_key) {
            if let Some(mut lv) = levels {
                if eager_levels {
                    let mut ok = true;
                    let last_group = lv.groups.len().saturating_sub(1);
                    for k in 0..lv.groups.len() {
                        let (sub, first, n, _) = lv.groups[k];
                        let evictable = srv_tracking_enabled() && k < last_group;
                        if let Some((used, used_extra)) = pages.bind_range(dev, res, va, console_bytes, total, sub, first as usize..(first + n) as usize, evictable) {
                            lv.groups[k].3 = true;
                            lv.group_tiles[k] = (used, used_extra);
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        record_texel_insert(&texels);
                        pages.targets.insert(raw as usize, Target { va, console_bytes, total, mapped: true, used: Vec::new(), used_extra: Vec::new(), levels: Some(lv), texels, last_touch: now, min_resident_mip: 0 });
                        true
                    } else {
                        for (u, ue) in take_level_tiles(Some(&mut lv)) {
                            if !u.is_empty() || !ue.is_empty() {
                                pages.release_tiles(u, ue, now, true);
                            }
                        }
                        false
                    }
                } else {
                    // nothing mapped yet: a level gets its pages with its first texels (`map_levels`)
                    LEVEL_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    LEVEL_STATS[1].fetch_add(total as u64, std::sync::atomic::Ordering::Relaxed);
                    record_texel_insert(&texels);
                    pages.targets.insert(raw as usize, Target { va, console_bytes, total, mapped: true, used: Vec::new(), used_extra: Vec::new(), levels: Some(lv), texels, last_touch: now, min_resident_mip: 0 });
                    true
                }
            } else if let Some((used, used_extra)) = pages.bind(dev, res, va, console_bytes, total) {
                record_texel_insert(&texels);
                pages.targets.insert(raw as usize, Target { va, console_bytes, total, mapped: true, used, used_extra, levels: None, texels, last_touch: now, min_resident_mip: 0 });
                true
            } else {
                false
            }
        } else {
            false
        }
    };
    if !bound {
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    Some(hr)
}

/// Hold a superseded texture's mapping until work submitted before this call has finished.
/// Polling completion on the CPU avoids a dependency from the mapping queue back to a title queue,
/// which may itself already be waiting for later mapping work.
pub(super) unsafe fn unmap_superseded(res: &ID3D12Resource) {
    let mut dev: Option<ID3D12Device> = None;
    if res.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let dev_key = dev.as_raw() as usize;
    let observed = held_unmap_completions(dev_key);
    let mut completion = super::present::queue_completion_snapshot();
    completion.active.extend(super::title_textures::upload_queues());
    let key = res.as_raw() as usize;
    let (done, signals) = {
        let mut g = PAGES.lock().unwrap();
        let Some(Some(pages)) = g.get_mut(&dev_key) else { return };
        if !pages.targets.get(&key).is_some_and(|target| target.mapped) {
            return;
        }
        let Some((marks, signals)) = pages.mark_unmap(&dev, &completion.active, &completion.retiring, completion.blocked) else {
            crate::xlog_warn!("D3D12.X: could not guard a superseded texture unmap; keeping its mapping");
            return;
        };
        let Some(target) = pages.targets.get_mut(&key) else { return };
        let held = HeldUnmap {
            res: res.clone(),
            total: target.total,
            used: std::mem::take(&mut target.used),
            used_extra: std::mem::take(&mut target.used_extra),
            level_tiles: take_level_tiles(target.levels.as_mut()),
            marks,
        };
        target.mapped = false;
        pages.held_unmaps.push(held);
        let done = pages.release_held_unmaps(std::time::Instant::now(), |fence| {
            observed_completion(&observed, fence, |known, current| known.as_raw() == current.as_raw())
        });
        (done, signals)
    };
    for (queue, fence, value) in signals {
        if super::real::native_queue_signal(&queue, &fence, value).is_err() {
            crate::xlog_warn!("D3D12.X: could not signal a superseded texture unmap marker; keeping its mapping");
        }
    }
    drop(done);
}

/// Re-map a previously unmapped placed texture when the newer placement shadowing it is released.
pub(super) unsafe fn remap_if_unmapped(res: &ID3D12Resource) -> bool {
    let mut dev: Option<ID3D12Device> = None;
    if res.GetDevice(&mut dev).is_err() {
        return false;
    }
    let Some(dev) = dev else { return false };
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return false };
    let Some((va, console_bytes, total, mapped)) = pages.targets.get(&key).map(|t| (t.va, t.console_bytes, t.total, t.mapped)) else {
        return false;
    };
    if mapped {
        return true;
    }
    // A deferred unmap never changed the resource's tiles. Cancel it and restore their ownership.
    if let Some(index) = pages.held_unmaps.iter().position(|held| held.res.as_raw() as usize == key) {
        let held = pages.held_unmaps.swap_remove(index);
        if let Some(target) = pages.targets.get_mut(&key) {
            target.used = held.used;
            target.used_extra = held.used_extra;
            if let Some(levels) = target.levels.as_mut().filter(|levels| levels.group_tiles.len() == held.level_tiles.len()) {
                levels.group_tiles = held.level_tiles;
            } else {
                for (used, used_extra) in held.level_tiles {
                    target.used.extend(used);
                    target.used_extra.extend(used_extra);
                }
            }
            target.mapped = true;
        }
        drop(g);
        drop(held.res);
        return true;
    }
    // (lazy levels: they get their pages again with their next texels)
    if let Some(t) = pages.targets.get_mut(&key).filter(|t| t.levels.is_some()) {
        t.mapped = true;
        return true;
    }
    let Some((used, used_extra)) = pages.bind(&dev, res, va, console_bytes, total) else {
        return false;
    };
    if let Some(t) = pages.targets.get_mut(&key) {
        t.used = used;
        t.used_extra = used_extra;
        t.mapped = true;
        if let Some(tm) = t.texels.as_mut() {
            if tm.resident_texels == 0 && tm.total_texels > 0 {
                tm.resident_texels = tm.total_texels;
                TEXEL_STATS[3].fetch_add(tm.total_texels, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    true
}

/// A resource released: its pages are no longer its.
fn forget_pages(resource: usize) {
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(mut t) = pages.targets.remove(&resource) else { continue };
        if let Some(l) = &t.levels {
            LEVEL_STATS[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            LEVEL_STATS[1].fetch_sub(t.total as u64, std::sync::atomic::Ordering::Relaxed);
            LEVEL_STATS[2].fetch_sub(l.mapped_tiles(), std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(tm) = &t.texels {
            TEXEL_STATS[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            if tm.srv_sampled_texels != 0 {
                TEXEL_STATS[1].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                TEXEL_STATS[4].fetch_sub(tm.srv_sampled_texels, std::sync::atomic::Ordering::Relaxed);
            }
            if tm.total_texels > 0 {
                TEXEL_STATS[2].fetch_sub(tm.total_texels, std::sync::atomic::Ordering::Relaxed);
            }
            if tm.resident_texels > 0 {
                TEXEL_STATS[3].fetch_sub(tm.resident_texels, std::sync::atomic::Ordering::Relaxed);
            }
        }
        if t.mapped {
            pages.release_tiles(t.used, t.used_extra, now, false);
            for (used, used_extra) in take_level_tiles(t.levels.as_mut()) {
                pages.release_tiles(used, used_extra, now, false);
            }
        }
        return;
    }
}

/// Reclaimed heaps waiting for `unmap_mark` (if any) and every live queue's fence mark on one device.
struct RetiringBatch<H, F> { heaps: Vec<H>, unmap_mark: Option<(F, u64)>, marks: Vec<(F, u64)> }

/// Per-device multi-queue retirement state (`H` = heap, `F` = fence; pure-CPU types in tests).
struct RetireQueue<H, F> { fences: std::collections::HashMap<usize, (F, u64)>, batches: Vec<RetiringBatch<H, F>> }

impl<H, F: Clone> RetireQueue<H, F> {
    fn new() -> Self { Self { fences: std::collections::HashMap::new(), batches: Vec::new() } }

    /// Seal `heaps` behind `unmap_mark` and one fence signal per queue (signaled BEFORE queueing).
    /// Returns `heaps` immediately when zero marks were signaled and `unmap_mark` is `None`.
    fn retire_with<Q>(&mut self, heaps: Vec<H>, unmap_mark: Option<(F, u64)>, queues: &[(usize, Q)], mut make_fence: impl FnMut() -> Option<F>, mut signal: impl FnMut(&Q, &F, u64) -> bool) -> Vec<H> {
        if heaps.is_empty() { return Vec::new(); }
        let mut marks = Vec::with_capacity(queues.len());
        for (k, q) in queues {
            let (f, v) = match self.fences.entry(*k) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => { let Some(f) = make_fence() else { continue }; e.insert((f, 0)) }
            };
            if signal(q, f, *v + 1) { *v += 1; marks.push((f.clone(), *v)); }
        }
        if marks.is_empty() && unmap_mark.is_none() { return heaps; }
        self.batches.push(RetiringBatch { heaps, unmap_mark, marks });
        Vec::new()
    }

    /// Strong owners of every fence whose completion can change a sweep decision. The caller
    /// polls them after dropping the retirement-state lock, then rechecks the current batches.
    fn completion_fences(&self) -> Vec<F> {
        let mut fences: Vec<_> = self.fences.values().map(|(fence, _)| fence.clone()).collect();
        for batch in &self.batches {
            fences.extend(batch.unmap_mark.as_ref().map(|(fence, _)| fence.clone()));
            fences.extend(batch.marks.iter().map(|(fence, _)| fence.clone()));
        }
        fences
    }

    /// Reap completed batches and prune dead-queue fences once completed; caller drops outside locks.
    fn sweep_with(&mut self, live_queues: &HashSet<usize>, mut completed: impl FnMut(&F) -> u64) -> (Vec<RetiringBatch<H, F>>, Vec<F>) {
        let (mut done, mut kept) = (Vec::new(), Vec::new());
        for b in std::mem::take(&mut self.batches) {
            if b.unmap_mark.as_ref().is_none_or(|(f, v)| completed(f) >= *v) && b.marks.iter().all(|(f, v)| completed(f) >= *v) { done.push(b); } else { kept.push(b); }
        }
        self.batches = kept;
        let stale: Vec<usize> = self.fences.iter().filter(|(k, (f, v))| !live_queues.contains(k) && completed(f) >= *v).map(|(&k, _)| k).collect();
        let pruned = stale.into_iter().filter_map(|k| self.fences.remove(&k).map(|(f, _)| f)).collect();
        (done, pruned)
    }

    /// In-flight batches own their marks; only the address-keyed reusable fence is removed.
    fn forget_queue(&mut self, queue: usize) -> Option<F> {
        self.fences.remove(&queue).map(|(fence, _)| fence)
    }
}

fn observed_completion<F>(observed: &[(F, u64)], fence: &F, mut same: impl FnMut(&F, &F) -> bool) -> u64 {
    observed.iter().find_map(|(known, value)| same(known, fence).then_some(*value)).unwrap_or(0)
}

unsafe impl Send for RetireQueue<ID3D12Heap, ID3D12Fence> {}

type DeviceRetire = std::sync::Arc<Mutex<RetireQueue<ID3D12Heap, ID3D12Fence>>>;

/// Per-device retirement states. The registry is held only while acquiring a strong state lease;
/// fence creation, queue signaling, polling, and COM destruction never hold it.
static RETIRE: Mutex<Option<std::collections::HashMap<usize, DeviceRetire>>> = Mutex::new(None);
static RETIRING_BATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn device_retire_state(dev_key: usize) -> DeviceRetire {
    RETIRE
        .lock()
        .unwrap()
        .get_or_insert_with(std::collections::HashMap::new)
        .entry(dev_key)
        .or_insert_with(|| std::sync::Arc::new(Mutex::new(RetireQueue::new())))
        .clone()
}

/// Remove raw queue-address state before a retired queue's final owner can be released. Fence and
/// cached device objects are extracted from their locks and dropped afterward; batches keep clones.
pub(super) fn forget_queue(id: super::present::QueueId) {
    let queue = id.raw();
    let page_fences = {
        let mut pages = PAGES.lock().unwrap();
        pages.values_mut().flatten().filter_map(|pages| {
            pages.waited.remove(&queue);
            pages.unmap_fences.remove(&queue)
        }).collect::<Vec<_>>()
    };
    let wait_gate = remove_wait_gate(wait_gates(), queue);
    let states: Vec<_> = RETIRE.lock().unwrap().as_ref().into_iter().flat_map(|devices| devices.values().cloned()).collect();
    let gone: Vec<_> = states
        .iter()
        .filter_map(|state| state.lock().unwrap().forget_queue(queue))
        .collect();
    drop((page_fences, gone, wait_gate));
}

/// Collect all known live command queues belonging to `dev_key` (deduplicated by raw ptr).
unsafe fn collect_device_queues(dev_key: usize, extra: Option<&ID3D12CommandQueue>) -> Vec<(usize, ID3D12CommandQueue)> {
    let (mut seen, mut out) = (HashSet::new(), Vec::new());
    let mut add = |q: ID3D12CommandQueue| {
        let key = q.as_raw() as usize;
        let mut q_dev: Option<ID3D12Device> = None;
        if seen.insert(key) && q.GetDevice(&mut q_dev).is_ok() && q_dev.is_some_and(|d| d.as_raw() as usize == dev_key) {
            out.push((key, q));
        }
    };
    for q in super::present::active_queues_held() {
        add(q);
    }
    for q in extra.into_iter().cloned().chain(super::title_textures::upload_queues()).chain(super::sparse::queue()) {
        add(q);
    }
    out
}

/// Retire `newly_reclaimed` across all live queues on `dev_key`, sweep completed batches, and prune
/// dead-queue fences; drops every finished `ID3D12Heap` and `ID3D12Fence` outside all locks.
unsafe fn retire_heaps_and_sweep(dev: &ID3D12Device, dev_key: usize, newly_reclaimed: Vec<ID3D12Heap>, unmap_mark: Option<(ID3D12Fence, u64)>, extra_queue: Option<&ID3D12CommandQueue>) {
    let queues = collect_device_queues(dev_key, extra_queue);
    let live_keys: HashSet<usize> = queues.iter().map(|&(k, _)| k).collect();
    let state = if newly_reclaimed.is_empty() {
        let registry = RETIRE.lock().unwrap();
        let Some(state) = registry.as_ref().and_then(|map| map.get(&dev_key)).cloned() else { return };
        state
    } else {
        device_retire_state(dev_key)
    };
    let (immediate, mut probes) = {
        let mut rq = state.lock().unwrap();
        let before = rq.batches.len();
        let immediate = rq.retire_with(newly_reclaimed, unmap_mark, &queues, || dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok(), |q, f, v| super::real::native_queue_signal(q, f, v).is_ok());
        RETIRING_BATCHES.fetch_add(rq.batches.len() - before, std::sync::atomic::Ordering::AcqRel);
        (immediate, rq.completion_fences())
    };
    // Deduplication and every COM call/drop happen without RETIRE held. The strong snapshots keep
    // raw addresses from being reused while results are identified by their exact fence object.
    probes.sort_unstable_by_key(|fence| fence.as_raw() as usize);
    probes.dedup_by_key(|fence| fence.as_raw() as usize);
    let observed: Vec<_> = probes.into_iter().map(|fence| {
        let completed = fence.GetCompletedValue();
        (fence, completed)
    }).collect();
    let (done, pruned) = {
        let mut rq = state.lock().unwrap();
        rq.sweep_with(&live_keys, |fence| {
            observed_completion(&observed, fence, |known, current| known.as_raw() == current.as_raw())
        })
    };
    RETIRING_BATCHES.fetch_sub(done.len(), std::sync::atomic::Ordering::AcqRel);
    drop((immediate, done, pruned, observed));
}

/// Raise `ResourceMinLODClamp` in `v` to at least `min_mip` so sampling skips unmapped top mips.
pub(super) fn clamp_srv_min_lod(v: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, min_mip: u32) {
    let lod = min_mip as f32;
    // SAFETY: each branch reads only the union field named by `v.ViewDimension`.
    unsafe {
        let clamp = match v.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE1D => &mut v.Anonymous.Texture1D.ResourceMinLODClamp,
            D3D12_SRV_DIMENSION_TEXTURE1DARRAY => &mut v.Anonymous.Texture1DArray.ResourceMinLODClamp,
            D3D12_SRV_DIMENSION_TEXTURE2D => &mut v.Anonymous.Texture2D.ResourceMinLODClamp,
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => &mut v.Anonymous.Texture2DArray.ResourceMinLODClamp,
            D3D12_SRV_DIMENSION_TEXTURE3D => &mut v.Anonymous.Texture3D.ResourceMinLODClamp,
            D3D12_SRV_DIMENSION_TEXTURECUBE => &mut v.Anonymous.TextureCube.ResourceMinLODClamp,
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => &mut v.Anonymous.TextureCubeArray.ResourceMinLODClamp,
            _ => return,
        };
        *clamp = clamp.max(lod);
    }
}

/// Default SRV description for a texture created with `CreateShaderResourceView(res, NULL, slot)`.
pub(super) fn default_srv_desc(rd: &D3D12_RESOURCE_DESC) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    let mips = u32::from(rd.MipLevels.max(1));
    let mut v = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: rd.Format,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    match rd.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            if rd.DepthOrArraySize > 1 {
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                v.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV { MostDetailedMip: 0, MipLevels: mips, FirstArraySlice: 0, ArraySize: u32::from(rd.DepthOrArraySize), ResourceMinLODClamp: 0.0 };
            } else {
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                v.Anonymous.Texture1D = D3D12_TEX1D_SRV { MostDetailedMip: 0, MipLevels: mips, ResourceMinLODClamp: 0.0 };
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
            if rd.SampleDesc.Count > 1 {
                return None;
            }
            if rd.DepthOrArraySize > 1 {
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MostDetailedMip: 0, MipLevels: mips, FirstArraySlice: 0, ArraySize: u32::from(rd.DepthOrArraySize), PlaneSlice: 0, ResourceMinLODClamp: 0.0 };
            } else {
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                v.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: mips, PlaneSlice: 0, ResourceMinLODClamp: 0.0 };
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            v.Anonymous.Texture3D = D3D12_TEX3D_SRV { MostDetailedMip: 0, MipLevels: mips, ResourceMinLODClamp: 0.0 };
        }
        _ => return None,
    }
    Some(v)
}

/// Minimum resident mip level currently mapped for `resource` (0 when full mip chain is resident).
pub(super) fn resident_min_mip(resource: usize) -> u32 {
    if !srv_tracking_enabled() || resource == 0 || !is_placed_texture(resource) {
        return 0;
    }
    let g = PAGES.lock().unwrap();
    for pages in g.values().flatten() {
        if let Some(t) = pages.targets.get(&resource) {
            return t.min_resident_mip;
        }
    }
    0
}

/// Device local video memory `(CurrentUsage, Budget)` queried at most every 100 ms.
unsafe fn local_vram_usage_and_budget(dev: &ID3D12Device) -> Option<(u64, u64)> {
    use windows::Win32::Graphics::Dxgi::*;
    struct Cached {
        luid: (u32, i32),
        adapter: IDXGIAdapter3,
        at: std::time::Instant,
        usage: u64,
        budget: u64,
    }
    // SAFETY: DXGI adapters are free-threaded.
    unsafe impl Send for Cached {}
    static CACHE: Mutex<Option<Cached>> = Mutex::new(None);
    let luid = dev.GetAdapterLuid();
    let luid_key = (luid.LowPart, luid.HighPart);
    let mut g = CACHE.lock().unwrap();
    if g.as_ref().is_none_or(|c| c.luid != luid_key) {
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(luid).ok()?;
        *g = Some(Cached {
            luid: luid_key,
            adapter,
            at: std::time::Instant::now() - std::time::Duration::from_secs(1),
            usage: 0,
            budget: 0,
        });
    }
    let c = g.as_mut()?;
    if c.at.elapsed() >= std::time::Duration::from_millis(100) {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        c.adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).ok()?;
        c.usage = info.CurrentUsage;
        c.budget = budget_override_bytes().unwrap_or(info.Budget);
        c.at = std::time::Instant::now();
    }
    Some((c.usage, c.budget.max(1)))
}

#[derive(Clone, Copy)]
struct HeapEvictCandidate {
    res_key: usize,
    heap: u32,
    heap_evictable: bool,
    heap_free: u32,
    last_touch: std::time::Instant,
    tiles: u32,
}

/// Choose eviction targets grouped by pool heap so whole 64 MiB `evictable` heaps drain and can
/// be reclaimed to DXGI instead of leaving scattered pinned/evictable fragments.
fn pick_evict_candidates_by_heap(entries: &[HeapEvictCandidate], target_tiles: u64) -> Vec<usize> {
    if target_tiles == 0 || entries.is_empty() {
        return Vec::new();
    }
    let mut by_heap: std::collections::HashMap<u32, (bool, u32, std::time::Instant, Vec<&HeapEvictCandidate>)> = std::collections::HashMap::new();
    for e in entries {
        by_heap
            .entry(e.heap)
            .and_modify(|(_, _, newest, list)| {
                if e.last_touch > *newest {
                    *newest = e.last_touch;
                }
                list.push(e);
            })
            .or_insert_with(|| (e.heap_evictable, e.heap_free, e.last_touch, vec![e]));
    }
    let mut heaps: Vec<(u32, bool, u32, std::time::Instant, Vec<&HeapEvictCandidate>)> = by_heap
        .into_iter()
        .map(|(h, (ev, free, newest, mut list))| {
            list.sort_unstable_by_key(|e| (e.last_touch, e.res_key));
            (h, ev, free, newest, list)
        })
        .collect();
    heaps.sort_unstable_by_key(|&(h, ev, free, newest, _)| {
        (!ev, free == 0, std::cmp::Reverse(if free > 0 { free } else { 0 }), newest, h)
    });

    let max_drain = target_tiles.saturating_add(u64::from(POOL_PAGES));
    let mut chosen = Vec::new();
    let mut freed = 0u64;
    for (_, ev, _, _, list) in heaps {
        if freed >= target_tiles {
            break;
        }
        for e in list {
            chosen.push(e.res_key);
            freed += u64::from(e.tiles);
            if freed >= target_tiles && (!ev || freed >= max_drain) {
                break;
            }
        }
    }
    chosen
}

fn evictable_heap(pages: &Pages, lv: &Levels) -> Option<u32> {
    let last = lv.groups.len().checked_sub(1)?;
    for k in 0..last {
        if !lv.groups[k].3 {
            continue;
        }
        if let Some(&a) = lv.group_tiles[k].0.first() {
            if let Some(p) = pages.pages.get(&a) {
                return Some(p.tile.0);
            }
        }
        if let Some(key) = lv.group_tiles[k].1.first() {
            if let Some(p) = pages.extra_pages.get(key) {
                return Some(p.tile.0);
            }
        }
    }
    None
}

unsafe fn evict_under_budget(dev: &ID3D12Device, dev_key: usize, now: std::time::Instant) {
    let Some((usage, budget)) = local_vram_usage_and_budget(dev) else { return };
    let pct = budget_evict_pct();
    if usage.saturating_mul(100) < budget.saturating_mul(pct) {
        return;
    }
    let low_pct = pct.saturating_sub(5);
    let want_bytes = usage.saturating_sub(budget.saturating_mul(low_pct) / 100);
    let target_tiles = (want_bytes / PAGE as u64).clamp(64, 8192);
    let mut raw_cands: Vec<HeapEvictCandidate> = Vec::new();
    let mut tail_by_res: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    {
        let g = PAGES.lock().unwrap();
        let Some(Some(pages)) = g.get(&dev_key) else { return };
        for (&res_key, t) in &pages.targets {
            if !t.mapped {
                continue;
            }
            let Some(lv) = t.levels.as_ref() else { continue };
            let Some((tiles, tail_mip)) = lv.evictable_chain() else { continue };
            let Some(heap) = evictable_heap(pages, lv) else { continue };
            let heap_evictable = pages.pool.heap_evictable.get(heap as usize).copied().unwrap_or(false);
            let heap_free = pages.pool.free_by_heap.get(heap as usize).map_or(0, |v| v.len() as u32);
            raw_cands.push(HeapEvictCandidate {
                res_key,
                heap,
                heap_evictable,
                heap_free,
                last_touch: t.last_touch,
                tiles,
            });
            tail_by_res.insert(res_key, tail_mip);
        }
    }
    if raw_cands.is_empty() {
        return;
    }
    let chosen = pick_evict_candidates_by_heap(&raw_cands, target_tiles);
    if chosen.is_empty() {
        return;
    }
    let chosen_set: HashSet<usize> = chosen.iter().copied().collect();
    let srv_candidates: Vec<(SrvSlot, u64)> = {
        let g = TRACKED_SRVS.lock().unwrap();
        let Some(tracked) = g.as_ref() else { return };
        tracked
            .by_slot
            .iter()
            .filter(|&(_, entry)| entry.device == dev_key && chosen_set.contains(&entry.resource))
            .map(|(&slot, entry)| (slot, entry.generation))
            .collect()
    };
    if srv_candidates.is_empty() {
        return;
    }
    let clamped_resources = with_exact_tracked_srv_mutations(&srv_candidates, |mutations| {
        let mut live_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        if let Some(tracked) = TRACKED_SRVS.lock().unwrap().as_ref() {
            for entry in tracked.by_slot.values() {
                if entry.device == dev_key && chosen_set.contains(&entry.resource) {
                    *live_counts.entry(entry.resource).or_insert(0) += 1;
                }
            }
        }
        let mut surviving: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut all_clampable: std::collections::HashMap<usize, bool> = std::collections::HashMap::new();
        for m in mutations {
            *surviving.entry(m.entry.resource).or_insert(0) += 1;
            let clampable = tail_by_res.contains_key(&m.entry.resource)
                && m.entry.desc.or_else(|| default_srv_desc(&m.resource.GetDesc())).is_some();
            all_clampable
                .entry(m.entry.resource)
                .and_modify(|ok| *ok &= clampable)
                .or_insert(clampable);
        }
        let mut clamped: std::collections::HashMap<usize, (ID3D12Resource, u32)> = std::collections::HashMap::new();
        for m in mutations {
            let res_key = m.entry.resource;
            if !all_clampable.get(&res_key).copied().unwrap_or(false) {
                continue;
            }
            if surviving.get(&res_key).copied() != live_counts.get(&res_key).copied() {
                continue;
            }
            let Some(&tail_mip) = tail_by_res.get(&res_key) else { continue };
            let Some(mut d) = m.entry.desc.or_else(|| default_srv_desc(&m.resource.GetDesc())) else { continue };
            clamp_srv_min_lod(&mut d, tail_mip);
            super::real::native_create_srv(m.heap.device(), m.resource, &d, m.heap.pc_srv);
            if m.heap.shared_srv != 0 {
                super::real::native_copy_descriptors_simple(
                    m.heap.device(),
                    1,
                    m.heap.shared_srv,
                    m.heap.pc_srv,
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                );
            }
            clamped.entry(res_key).or_insert_with(|| (m.resource.clone(), tail_mip));
        }
        if !clamped.is_empty() {
            let mut g = PAGES.lock().unwrap();
            if let Some(Some(pages)) = g.get_mut(&dev_key) {
                for (&res_key, &(_, tail_mip)) in &clamped {
                    if let Some(t) = pages.targets.get_mut(&res_key) {
                        t.min_resident_mip = t.min_resident_mip.max(tail_mip);
                    }
                }
            }
        }
        clamped
    });
    let Some(clamped_resources) = clamped_resources else { return };
    if clamped_resources.is_empty() {
        return;
    }

    let mut evicted_groups = 0u64;
    let mut evicted_tiles = 0u64;
    {
        let mut g = PAGES.lock().unwrap();
        if let Some(Some(pages)) = g.get_mut(&dev_key) {
            for res_key in chosen {
                let Some((res, expected_tail)) = clamped_resources.get(&res_key) else { continue };
                if !is_placed_texture(res_key) {
                    continue;
                }
                let mut ops = Vec::new();
                {
                    let Some(t) = pages.targets.get_mut(&res_key) else { continue };
                    if !t.mapped {
                        continue;
                    }
                    let Some(lv) = t.levels.as_mut() else { continue };
                    let last = lv.groups.len().saturating_sub(1);
                    if lv.groups.get(last).map(|g| g.0) != Some(*expected_tail) {
                        continue;
                    }
                    let mut dropped_texels = 0u64;
                    let mut unmapped_tiles = 0u64;
                    for k in 0..last {
                        if !lv.groups[k].3 {
                            continue;
                        }
                        let (sub, _, tiles, _) = lv.groups[k];
                        let (used, used_extra) = std::mem::take(&mut lv.group_tiles[k]);
                        lv.groups[k].3 = false;
                        if let Some(tm) = t.texels.as_ref() {
                            dropped_texels = dropped_texels.saturating_add(range_texels(tm.width, tm.height, tm.depth, tm.dim, lv.levels_of(k)));
                        }
                        unmapped_tiles += u64::from(tiles);
                        evicted_groups += 1;
                        ops.push((sub, tiles, used, used_extra));
                    }
                    if ops.is_empty() {
                        continue;
                    }
                    t.min_resident_mip = t.min_resident_mip.max(*expected_tail);
                    if level_pages() && unmapped_tiles > 0 {
                        LEVEL_STATS[2].fetch_sub(unmapped_tiles, std::sync::atomic::Ordering::Relaxed);
                    }
                    if let Some(tm) = t.texels.as_mut() {
                        let removed = dropped_texels.min(tm.resident_texels);
                        if removed > 0 {
                            tm.resident_texels -= removed;
                            TEXEL_STATS[3].fetch_sub(removed, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    evicted_tiles += unmapped_tiles;
                }
                for (sub, tiles, used, used_extra) in ops {
                    pages.unmap_range(res, sub, tiles);
                    pages.release_tiles(used, used_extra, now, true);
                }
            }
        }
    }
    drop(clamped_resources);
    if evicted_groups > 0 {
        let total_groups = EVICT_STATS[0].fetch_add(evicted_groups, std::sync::atomic::Ordering::Relaxed) + evicted_groups;
        let total_tiles = EVICT_STATS[1].fetch_add(evicted_tiles, std::sync::atomic::Ordering::Relaxed) + evicted_tiles;
        if total_groups == evicted_groups || (total_groups.is_power_of_two() && total_groups >= 16) {
            crate::xlog_info!(
                "D3D12.X: texture budget eviction: evicted {} mip groups ({} MB) at {} / {} MB local VRAM",
                total_groups,
                (total_tiles * PAGE as u64) >> 20,
                usage >> 20,
                budget >> 20
            );
        }
    }
}

/// Before GPU work on `queue`: have it wait for every tile mapping made so far on its device,
/// release any previously retired heaps whose GPU fences have completed, and periodically (once
/// per second per device) retire completely empty 64 MiB VRAM pool heaps and idle arenas behind
/// per-queue GPU fence signals across all live queues on the device.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let measure = before_execute_timing_active();
    let mut measured = BeforeExecuteTiming { calls: measure as usize, ..Default::default() };
    let gate_started = measure.then(std::time::Instant::now);
    let queue_key = queue.as_raw() as usize;
    let wait_gate = wait_gates().lock().unwrap().entry(queue_key).or_default().clone();
    let _queue_serial = wait_gate.serial.lock().unwrap();
    if let Some(started) = gate_started {
        measured.gate_wait = started.elapsed();
    }
    if measure {
        let cached = wait_gate.device.get().is_some();
        measured.device_cache_hits = cached as usize;
        measured.device_queries = (!cached) as usize;
    }
    let device_started = measure.then(std::time::Instant::now);
    let dev = wait_gate.cached_or_try_device(|| {
        let mut dev: Option<ID3D12Device> = None;
        queue.GetDevice(&mut dev).ok()?;
        dev
    });
    let Some(dev) = dev else {
        if let Some(started) = device_started {
            measured.device = started.elapsed();
        }
        if measure {
            record_before_execute_timing(measured);
        }
        return;
    };
    if let Some(started) = device_started {
        measured.device = started.elapsed();
    }
    let dev_key = dev.as_raw() as usize;
    let now = std::time::Instant::now();
    let poll_started = measure.then(std::time::Instant::now);
    let unmap_completions = held_unmap_completions(dev_key);
    if let Some(started) = poll_started {
        measured.unmap_poll = started.elapsed();
    }
    let periodic = {
        let mut g = PAGES.lock().unwrap();
        g.get_mut(&dev_key).and_then(Option::as_mut).is_some_and(|pages| {
            if now.duration_since(pages.last_reclaim) >= std::time::Duration::from_secs(1) {
                pages.last_reclaim = now;
                true
            } else {
                false
            }
        })
    };
    if periodic && srv_tracking_enabled() {
        evict_under_budget(dev, dev_key, now);
    }
    let mut newly_reclaimed = Vec::new();
    let mut unmap_mark: Option<(ID3D12Fence, u64)> = None;
    let mut completed_unmaps = Vec::new();
    let mut wait = None;
    let state_started = measure.then(std::time::Instant::now);
    {
        let mut g = PAGES.lock().unwrap();
        if let Some(Some(pages)) = g.get_mut(&dev_key) {
            completed_unmaps = pages.release_held_unmaps(now, |fence| {
                observed_completion(&unmap_completions, fence, |known, current| known.as_raw() == current.as_raw())
            });
            let value = pages.value;
            if reserve_page_wait(&mut pages.waited, queue_key, value) {
                wait = Some((pages.fence.clone(), value));
            }
            if periodic {
                newly_reclaimed = pages.sweep_and_reclaim(now);
                if !newly_reclaimed.is_empty() {
                    unmap_mark = Some((pages.fence.clone(), pages.value));
                }
            }
        }
    }
    if let Some(started) = state_started {
        measured.state = started.elapsed();
        measured.mapping_waits = wait.is_some() as usize;
        measured.reclaim_passes = periodic as usize;
    }
    let queue_wait_started = measure.then(std::time::Instant::now);
    if let Some((fence, value)) = wait {
        let _ = super::real::native_queue_wait(queue, &fence, value);
        super::list_x::note_internal_wait(queue_key, "texture page mappings", &fence, value);
    }
    if let Some(started) = queue_wait_started {
        measured.queue_wait = started.elapsed();
    }
    let retirement_started = measure.then(std::time::Instant::now);
    if periodic {
        newly_reclaimed.extend(ARENAS.lock().unwrap().sweep_dropped_for(dev_key, now));
    }
    let retire = periodic || !newly_reclaimed.is_empty() || RETIRING_BATCHES.load(std::sync::atomic::Ordering::Acquire) > 0;
    if retire {
        retire_heaps_and_sweep(dev, dev_key, newly_reclaimed, unmap_mark, Some(queue));
    }
    drop(completed_unmaps);
    if let Some(started) = retirement_started {
        measured.retirement = started.elapsed();
        measured.retirement_passes = retire as usize;
    }
    if measure {
        record_before_execute_timing(measured);
    }
}

#[cfg(test)]
pub(super) static VRAM_TEST_LOCK: Mutex<()> = Mutex::new(());

/// A tile mapping on `dev`'s page queue, as `map` leaves it (the fence advanced and signalled
/// there): the value a queue must wait for before work that uses the pages. None: no pages on
/// this device (no tiled resources).
#[cfg(test)]
pub(super) unsafe fn note_mapping_for_test(dev: &ID3D12Device) -> Option<u64> {
    let mut g = PAGES.lock().unwrap();
    let pages = g.entry(dev.as_raw() as usize).or_insert_with(|| Pages::new(dev)).as_mut()?;
    pages.value += 1;
    super::real::native_queue_signal(&pages.queue, &pages.fence, pages.value).ok()?;
    Some(pages.value)
}

/// The page mapping value `queue` was last made to wait for on `dev` (`before_execute`).
#[cfg(test)]
pub(super) fn waited_for_test(dev: usize, queue: usize) -> u64 {
    let g = PAGES.lock().unwrap();
    g.get(&dev).and_then(|p| p.as_ref()).and_then(|p| p.waited.get(&queue).copied()).unwrap_or(0)
}

/// Tile identities of one mapped resource: title pages, then extra pages.
/// A device-wide byte count also falls when unrelated idle pages are swept.
#[cfg(test)]
fn target_page_keys(target: &Target) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut used = target.used.clone();
    let mut extra = target.used_extra.clone();
    if let Some(levels) = &target.levels {
        for (level_used, level_extra) in &levels.group_tiles {
            used.extend(level_used);
            extra.extend(level_extra);
        }
    }
    (used, extra)
}

#[cfg(test)]
fn target_tiles(pages: &Pages, target: &Target) -> Option<(Vec<Tile>, Vec<Tile>)> {
    let (used, extra) = target_page_keys(target);
    let used = used
        .iter()
        .map(|key| pages.pages.get(key).map(|page| page.tile))
        .collect::<Option<Vec<_>>>()?;
    let extra = extra
        .iter()
        .map(|key| pages.extra_pages.get(key).map(|page| page.tile))
        .collect::<Option<Vec<_>>>()?;
    (used.len() + extra.len() == target.total as usize).then_some((used, extra))
}

#[cfg(test)]
pub(super) fn mapped_tiles_for(device: usize, resource: usize) -> Option<(Vec<(u32, u32)>, Vec<(u32, u32)>)> {
    let g = PAGES.lock().unwrap();
    let p = g.get(&device)?.as_ref()?;
    let target = p.targets.get(&resource)?;
    if !target.mapped {
        return None;
    }
    target_tiles(p, target)
}

pub(super) fn page_bytes_for(device: usize) -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.get(&device) else { return (0, 0) };
    let pages = p.pages.len().saturating_add(p.extra_pages.len()).saturating_mul(PAGE);
    let pools = p.pools.len().saturating_mul(POOL_PAGES as usize).saturating_mul(PAGE);
    (pages, pools)
}

#[cfg(test)]
fn held_unmaps_for(device: usize) -> usize {
    let g = PAGES.lock().unwrap();
    g.get(&device).and_then(Option::as_ref).map_or(0, |pages| pages.held_unmaps.len())
}

/// Pool/tile identities used by one mapped resource: title pages, then extra pages.
/// Unlike device-wide byte counts, these cannot shrink when unrelated idle targets expire.
#[cfg(test)]
pub(super) fn tile_mappings_for(device: usize, resource: usize) -> Option<(Vec<(u32, u32)>, Vec<(u32, u32)>)> {
    let g = PAGES.lock().unwrap();
    let p = g.get(&device)?.as_ref()?;
    let target = p.targets.get(&resource)?;
    if !target.mapped { return None; }
    target_tiles(p, target)
}

/// (bytes of pages in use by tiled textures, bytes of pool heaps) – `XCL_ACTIVITY`
pub fn page_bytes() -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    g.values().flatten().fold((0, 0), |(used, pools), p| {
        (used + (p.pages.len() + p.extra_pages.len()) * PAGE, pools + p.pools.len() * POOL_PAGES as usize * PAGE)
    })
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

/// A sampled texture of the title at `va` (`desc`: the PC description, `console_bytes` its size on
/// the console) backed by shared 64 KiB video-memory pages; None when reserved placement is not
/// available (the caller falls back to a committed resource).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place_pages(
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) || std::env::var_os("XCL_VRAM_ARENAS").is_some() {
        return None;
    }
    if console_has_packed_3d_mips(desc) {
        return None;
    }
    let single_2d = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.DepthOrArraySize == 1
        && desc.SampleDesc.Count == 1
        && desc.Flags == D3D12_RESOURCE_FLAG_NONE;
    let lazy = level_pages() && single_2d;
    let track_levels = srv_tracking_enabled() && single_2d && desc.MipLevels >= 2;
    place_on_pages(dev, this, va, desc, state, clear, console_bytes, false, lazy, track_levels, riid, ppv)
}

/// The levels `levels` of `res` (None: all of them) get their pages if they have none yet: a
/// texture placed with lazy levels (any other: nothing to do). Returns the levels that got pages
/// by this call - the queue the texture is used on next has to wait for that (`before_execute`),
/// and what the pages held before is another texture's.
pub(super) unsafe fn map_levels(res: &ID3D12Resource, levels: Option<&[u32]>) -> Vec<u32> {
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    for (&dev_key, pages) in g.iter_mut() {
        let Some(pages) = pages.as_mut() else { continue };
        let Some(t) = pages.targets.get(&key) else { continue };
        let Some(lv) = t.levels.as_ref().filter(|_| t.mapped) else { return Vec::new() };
        let d = dev_key as *mut c_void;
        let Some(dev) = ID3D12Device::from_raw_borrowed(&d) else { return Vec::new() };
        let min_mip = if srv_tracking_enabled()
            && local_vram_usage_and_budget(dev).is_some_and(|(u, b)| u.saturating_mul(100) >= b.saturating_mul(budget_evict_pct()))
        {
            t.min_resident_mip
        } else {
            0
        };
        let mut wanted: Vec<usize> = match levels {
            Some(l) => l.iter().map(|&x| lv.group_of(x)).collect(),
            None => (0..lv.groups.len()).collect(),
        };
        wanted.sort_unstable();
        wanted.dedup();
        let todo: Vec<(usize, u32, u32, u32, std::ops::Range<u32>)> = wanted
            .into_iter()
            .filter(|&k| !lv.groups[k].3 && lv.levels_of(k).start >= min_mip)
            .map(|k| (k, lv.groups[k].0, lv.groups[k].1, lv.groups[k].2, lv.levels_of(k)))
            .collect();
        if todo.is_empty() {
            if let Some(t) = pages.targets.get_mut(&key) {
                t.last_touch = std::time::Instant::now();
            }
            return Vec::new();
        }
        let (va, console_bytes, total) = (t.va, t.console_bytes, t.total);
        let last_group = lv.groups.len().saturating_sub(1);
        let mut fresh = Vec::new();
        for (k, sub, first, n, levels) in todo {
            let evictable = srv_tracking_enabled() && k < last_group;
            let Some((used, used_extra)) = pages.bind_range(dev, res, va, console_bytes, total, sub, first as usize..(first + n) as usize, evictable) else { continue };
            let Some(t) = pages.targets.get_mut(&key) else { return fresh };
            t.last_touch = std::time::Instant::now();
            if let Some(l) = t.levels.as_mut() {
                l.groups[k].3 = true;
                l.group_tiles[k].0.extend(used);
                l.group_tiles[k].1.extend(used_extra);
                if let Some(lowest) = l.groups.iter().find(|g| g.3).map(|g| g.0) {
                    t.min_resident_mip = lowest;
                }
            }
            if let Some(tm) = t.texels.as_mut() {
                let added = range_texels(tm.width, tm.height, tm.depth, tm.dim, levels.clone());
                let delta = tm.total_texels.saturating_sub(tm.resident_texels).min(added);
                if delta > 0 {
                    tm.resident_texels += delta;
                    TEXEL_STATS[3].fetch_add(delta, std::sync::atomic::Ordering::Relaxed);
                }
            }
            LEVEL_STATS[2].fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
            let groups = LEVEL_STATS[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if groups >= 1024 && groups.is_power_of_two() {
                let (textures, stand, mapped, _) = level_stats();
                crate::xlog_info!("D3D12.X: level pages: {textures} textures standing for {} MB have pages for {} MB ({groups} levels given pages so far)", stand >> 4, mapped >> 4);
            }
            fresh.extend(levels);
        }
        return fresh;
    }
    Vec::new()
}

/// Record which mip levels of placed sampled texture `resource` an SRV can sample.
pub(super) fn note_srv(resource: usize, desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) {
    if (!super::memory_report::enabled() && !srv_tracking_enabled()) || resource == 0 || !is_placed_texture(resource) {
        return;
    }
    let mut g = PAGES.lock().unwrap();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.get_mut(&resource) else { continue };
        t.last_touch = std::time::Instant::now();
        let Some(tm) = t.texels.as_mut() else { return };
        let (first_srv, added) = tm.record_srv(desc);
        if first_srv {
            TEXEL_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if added > 0 {
            TEXEL_STATS[4].fetch_add(added, std::sync::atomic::Ordering::Relaxed);
        }
        return;
    }
}

/// Whether `resource` is a texture whose levels get their pages one by one (`map_levels`).
pub(super) fn has_lazy_levels(resource: usize) -> bool {
    PAGES.lock().unwrap().values().flatten().any(|p| p.targets.get(&resource).is_some_and(|t| t.levels.is_some()))
}

/// XG and the PC driver pack volume mip tails differently. Keep these textures on the
/// detile/upload path rather than mapping title pages directly onto the driver's packed tail.
fn console_has_packed_3d_mips(desc: &D3D12_RESOURCE_DESC) -> bool {
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || desc.MipLevels <= 1 {
        return false;
    }
    let mut console_desc = [0u8; 0x38];
    unsafe {
        std::ptr::copy_nonoverlapping(
            desc as *const _ as *const u8,
            console_desc.as_mut_ptr(),
            console_desc.len(),
        )
    };
    crate::hle::xg::Texture::new(&console_desc)
        .ok()
        .and_then(|t| t.texels().map(|s| s.depth > 0 && s.first_tail < s.chain.levels.len()))
        .unwrap_or(false)
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description, `console_bytes` its
/// size on the console): on the title's pages there, else in the arena for that address (None:
/// neither – the caller makes a resource of its own).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let va = canonical_va(va);
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        if let Some(hr) = place_on_pages(dev, this, va, desc, state, clear, console_bytes, true, false, false, riid, ppv) {
            return Some(hr);
        }
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || va as u64 % info.Alignment.max(1) != 0 {
        return None;
    }
    let align = (info.Alignment as usize).max(PAGE);
    let now = std::time::Instant::now();
    let dev_key = dev.as_raw() as usize;
    let (taken, expired) = ARENAS.lock().unwrap().take_for(dev_key, va, info.SizeInBytes as usize, align, now, |size, align| arena_heap(dev, va, size, align));
    if !expired.is_empty() {
        retire_heaps_and_sweep(dev, dev_key, expired, None, None);
    }
    let (heap, offset, id) = taken?;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        ARENAS.lock().unwrap().release(id, now);
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    ARENA_OF.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(*ppv as usize, id);
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn before_execute_queue_caches_only_a_successful_device_lookup() {
        let state = BeforeExecuteQueue::<u32>::default();
        let calls = std::cell::Cell::new(0);
        for _ in 0..2 {
            let _serial = state.serial.lock().unwrap();
            assert_eq!(state.cached_or_try_device(|| {
                calls.set(calls.get() + 1);
                Some(17)
            }).copied(), Some(17));
        }
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn before_execute_queue_retries_failed_and_null_device_lookups() {
        let state = BeforeExecuteQueue::<u32>::default();
        let calls = std::cell::Cell::new(0);
        let lookup = || {
            calls.set(calls.get() + 1);
            (calls.get() == 3).then_some(23)
        };
        for expected in [None, None, Some(23), Some(23)] {
            let _serial = state.serial.lock().unwrap();
            assert_eq!(state.cached_or_try_device(lookup).copied(), expected);
        }
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn removing_before_execute_queue_makes_reused_address_fresh() {
        let gates = Mutex::new(HashMap::<usize, Arc<BeforeExecuteQueue<u32>>>::new());
        let old = Arc::new(BeforeExecuteQueue::default());
        {
            let _serial = old.serial.lock().unwrap();
            assert_eq!(old.cached_or_try_device(|| Some(29)).copied(), Some(29));
        }
        gates.lock().unwrap().insert(5, old);
        drop(remove_wait_gate(&gates, 5));

        let fresh = gates.lock().unwrap().entry(5).or_default().clone();
        assert!(fresh.device.get().is_none());
        let _serial = fresh.serial.lock().unwrap();
        assert_eq!(fresh.cached_or_try_device(|| Some(31)).copied(), Some(31));
    }

    struct RegistryLockCheckingOwner {
        id: usize,
        gates: Arc<Mutex<HashMap<usize, Arc<BeforeExecuteQueue<RegistryLockCheckingOwner>>>>>,
        dropped_outside_lock: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for RegistryLockCheckingOwner {
        fn drop(&mut self) {
            self.dropped_outside_lock.store(self.gates.try_lock().is_ok(), std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    fn removed_before_execute_owner_drops_outside_registry_lock() {
        let gates: Arc<Mutex<HashMap<usize, Arc<BeforeExecuteQueue<RegistryLockCheckingOwner>>>>> = Arc::new(Mutex::new(HashMap::new()));
        let dropped_outside_lock = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let state = Arc::new(BeforeExecuteQueue::default());
        {
            let _serial = state.serial.lock().unwrap();
            state.cached_or_try_device(|| Some(RegistryLockCheckingOwner {
                id: 37,
                gates: gates.clone(),
                dropped_outside_lock: dropped_outside_lock.clone(),
            }));
        }
        gates.lock().unwrap().insert(7, state);
        drop(remove_wait_gate(&gates, 7));
        assert!(dropped_outside_lock.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn retained_before_execute_queue_survives_concurrent_key_reuse() {
        let gates: Arc<Mutex<HashMap<usize, Arc<BeforeExecuteQueue<RegistryLockCheckingOwner>>>>> = Arc::new(Mutex::new(HashMap::new()));
        let old_dropped_outside_lock = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let old = Arc::new(BeforeExecuteQueue::default());
        {
            let _serial = old.serial.lock().unwrap();
            old.cached_or_try_device(|| Some(RegistryLockCheckingOwner {
                id: 41,
                gates: gates.clone(),
                dropped_outside_lock: old_dropped_outside_lock.clone(),
            }));
        }
        gates.lock().unwrap().insert(9, old.clone());

        let (retained_tx, retained_rx) = std::sync::mpsc::channel();
        let (reused_tx, reused_rx) = std::sync::mpsc::channel();
        let retained = std::thread::spawn(move || {
            let _serial = old.serial.lock().unwrap();
            let owner = old.cached_or_try_device(|| None).unwrap();
            retained_tx.send(owner.id).unwrap();
            reused_rx.recv().unwrap();
            assert_eq!(owner.id, 41);
        });

        assert_eq!(retained_rx.recv().unwrap(), 41);
        drop(remove_wait_gate(&gates, 9));
        assert!(!old_dropped_outside_lock.load(std::sync::atomic::Ordering::Relaxed));

        let fresh_dropped_outside_lock = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fresh = Arc::new(BeforeExecuteQueue::default());
        {
            let _serial = fresh.serial.lock().unwrap();
            let owner = fresh.cached_or_try_device(|| Some(RegistryLockCheckingOwner {
                id: 43,
                gates: gates.clone(),
                dropped_outside_lock: fresh_dropped_outside_lock.clone(),
            })).unwrap();
            assert_eq!(owner.id, 43);
        }
        gates.lock().unwrap().insert(9, fresh.clone());
        reused_tx.send(()).unwrap();
        retained.join().unwrap();
        assert!(old_dropped_outside_lock.load(std::sync::atomic::Ordering::Relaxed));

        drop(remove_wait_gate(&gates, 9));
        drop(fresh);
        assert!(fresh_dropped_outside_lock.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[derive(Clone)]
    struct FakeReceipt {
        fence: Arc<std::sync::atomic::AtomicU64>,
        value: u64,
    }

    fn fake_complete(receipt: &FakeReceipt) -> bool {
        receipt.fence.load(std::sync::atomic::Ordering::Relaxed) >= receipt.value
    }

    #[test]
    fn before_execute_timing_is_one_thread_local_activity_sample() {
        drop(take_before_execute_timing());
        assert!(!before_execute_timing_active());
        begin_before_execute_timing();
        begin_before_execute_timing();
        record_before_execute_timing(BeforeExecuteTiming {
            gate_wait: std::time::Duration::from_micros(3),
            calls: 1,
            device_cache_hits: 1,
            mapping_waits: 1,
            ..Default::default()
        });
        record_before_execute_timing(BeforeExecuteTiming {
            state: std::time::Duration::from_micros(7),
            calls: 1,
            device_queries: 1,
            reclaim_passes: 1,
            ..Default::default()
        });
        let timing = take_before_execute_timing().unwrap();
        assert_eq!(timing.gate_wait, std::time::Duration::from_micros(3));
        assert_eq!(timing.state, std::time::Duration::from_micros(7));
        assert_eq!((timing.calls, timing.device_cache_hits, timing.device_queries), (2, 1, 1));
        assert_eq!((timing.mapping_waits, timing.reclaim_passes), (1, 1));
        assert!(!before_execute_timing_active());
    }

    #[test]
    fn shared_view_epochs_cancel_retire_and_remain_monotonic() {
        let mut epochs = SharedViewEpochs::new();
        let cancelled = epochs.reserve();
        let retiring = epochs.reserve();
        assert!(cancelled.0 < retiring.0);
        epochs.cancel(cancelled);

        let fence = Arc::new(std::sync::atomic::AtomicU64::new(0));
        epochs.attach(retiring, FakeReceipt { fence: Arc::clone(&fence), value: 3 });
        assert!(!epochs.quiescent(fake_complete));
        fence.store(3, std::sync::atomic::Ordering::Relaxed);
        assert!(epochs.quiescent(fake_complete));
        assert!(epochs.reserve().0 > retiring.0, "retirement never reuses an epoch identity");
    }

    #[test]
    fn reused_queue_address_cannot_retire_an_old_epoch() {
        let mut epochs = SharedViewEpochs::new();
        let old_epoch = epochs.reserve();
        let old_generation_fence = Arc::new(std::sync::atomic::AtomicU64::new(0));
        epochs.attach(old_epoch, FakeReceipt { fence: Arc::clone(&old_generation_fence), value: 7 });

        // A new queue generation at the same numeric address owns another fence. Its progress
        // cannot satisfy the strong fence owner retained by the old epoch's receipt.
        let replacement_generation_fence = Arc::new(std::sync::atomic::AtomicU64::new(99));
        assert_eq!(replacement_generation_fence.load(std::sync::atomic::Ordering::Relaxed), 99);
        assert!(!epochs.quiescent(fake_complete));
        old_generation_fence.store(7, std::sync::atomic::Ordering::Relaxed);
        assert!(epochs.quiescent(fake_complete));
    }

    #[test]
    fn untracked_shared_view_epoch_never_becomes_quiescent() {
        let mut epochs = SharedViewEpochs::<FakeReceipt>::new();
        let epoch = epochs.reserve();
        epochs.poison(epoch);
        assert!(!epochs.quiescent(fake_complete));
    }

    #[test]
    fn disabled_tracking_cannot_authorize_descriptor_mutation() {
        let epochs = Mutex::new(SharedViewEpochs::<FakeReceipt>::new());
        let mut ran = false;
        assert!(with_quiescent_epochs(false, &epochs, fake_complete, || ran = true).is_none());
        assert!(!ran);
    }

    #[test]
    fn quiescent_mutation_excludes_new_reservations() {
        let epochs = Arc::new(Mutex::new(SharedViewEpochs::<FakeReceipt>::new()));
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mutation_epochs = Arc::clone(&epochs);
        let mutation = std::thread::spawn(move || {
            with_quiescent_epochs(true, &mutation_epochs, fake_complete, || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
        });
        entered_rx.recv().unwrap();

        let (attempted_tx, attempted_rx) = std::sync::mpsc::channel();
        let (reserved_tx, reserved_rx) = std::sync::mpsc::channel();
        let reserve_epochs = Arc::clone(&epochs);
        let reserve = std::thread::spawn(move || {
            attempted_tx.send(()).unwrap();
            let epoch = reserve_epochs.lock().unwrap().reserve();
            reserved_tx.send(epoch).unwrap();
        });
        attempted_rx.recv().unwrap();
        assert!(reserved_rx.recv_timeout(std::time::Duration::from_millis(50)).is_err());
        release_tx.send(()).unwrap();
        assert!(mutation.join().unwrap().is_some());
        assert_eq!(reserved_rx.recv().unwrap(), SharedViewEpoch(1));
        reserve.join().unwrap();
    }

    fn tracked_entry(resource: usize, pc_srv: u64) -> TrackedSrv {
        let mut words = [0u32; 8];
        words[0] = pc_srv as u32;
        words[1] = resource as u32;
        words[3] = 9 << 28;
        TrackedSrv {
            generation: 0, device: 7, resource, pc_srv, shared_srv: pc_srv + 1,
            index: pc_srv as u32, desc: Some(D3D12_SHADER_RESOURCE_VIEW_DESC::default()), words,
        }
    }

    fn srv_slot(heap_generation: u64, slot: u32) -> SrvSlot {
        SrvSlot { heap_generation, slot }
    }

    struct MutationOwner(Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for MutationOwner {
        fn drop(&mut self) {
            assert!(TRACKED_SRV_WRITES.try_lock().is_ok(), "strong owner dropped under WRITE");
            assert!(TRACKED_SRVS.try_lock().is_ok(), "strong owner dropped under tracker lock");
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    struct UnwindMutationOwner {
        drops: Arc<std::sync::atomic::AtomicUsize>,
        unlocked: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Drop for UnwindMutationOwner {
        fn drop(&mut self) {
            use std::sync::TryLockError;
            let available = |mutex: &'static Mutex<()>| match mutex.try_lock() {
                Ok(guard) => { drop(guard); true }
                Err(TryLockError::Poisoned(error)) => { drop(error.into_inner()); true }
                Err(TryLockError::WouldBlock) => false,
            };
            let write = available(&TRACKED_SRV_WRITES) as usize;
            let tracker = match TRACKED_SRVS.try_lock() {
                Ok(guard) => { drop(guard); true }
                Err(TryLockError::Poisoned(error)) => { drop(error.into_inner()); true }
                Err(TryLockError::WouldBlock) => false,
            } as usize;
            self.unlocked.fetch_or(write | tracker << 1, std::sync::atomic::Ordering::Relaxed);
            self.drops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn begin_mutation_test(entry: TrackedSrv, slot: SrvSlot) -> (u64, [u32; 8]) {
        SRV_TRACKING_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
        *SHARED_VIEW_EPOCHS.lock().unwrap() = SharedViewEpochs::new();
        let mut tracked = TrackedSrvs::default();
        tracked.write(slot, Some(entry));
        let current = tracked.by_slot[&slot];
        *TRACKED_SRVS.lock().unwrap() = Some(tracked);
        (current.generation, current.words)
    }

    fn end_mutation_test() {
        TRACKED_SRVS.lock().unwrap().take();
        *SHARED_VIEW_EPOCHS.lock().unwrap() = SharedViewEpochs::new();
        SRV_TRACKING_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn exact_srv_mutation_revalidates_then_drops_owners_outside_inner_locks() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let slot = srv_slot(60, 2);
        let (generation, words) = begin_mutation_test(tracked_entry(0x6010, 0xA0), slot);
        let snapshot = TRACKED_SRVS.lock().unwrap().as_ref().unwrap().by_slot[&slot];
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let result = with_exact_tracked_srv_mutation_on(
            slot,
            generation,
            |_, _| Some((snapshot, MutationOwner(Arc::clone(&drops)), MutationOwner(Arc::clone(&drops)))),
            |_| Some(words),
            |entry, _, _| {
                assert_eq!(entry.resource, 0x6010);
                assert!(SHARED_VIEW_EPOCHS.try_lock().is_err(), "epoch was not held outermost");
                assert!(TRACKED_SRV_WRITES.try_lock().is_err(), "WRITE was not held for mutation");
                assert!(TRACKED_SRVS.try_lock().is_ok(), "tracker lock crossed mutation");
                17
            },
        );
        assert_eq!(result, Some(17));
        assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 2);
        end_mutation_test();
    }

    #[test]
    fn exact_srv_mutation_rejects_a_changed_generation_without_removing_it() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let slot = srv_slot(61, 1);
        let (generation, words) = begin_mutation_test(tracked_entry(0x6110, 0xA4), slot);
        let old = TRACKED_SRVS.lock().unwrap().as_ref().unwrap().by_slot[&slot];
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let result = with_exact_tracked_srv_mutation_on(
            slot,
            generation,
            |_, _| {
                // Model a hooked overwrite after the owners were acquired but before WRITE.
                TRACKED_SRVS.lock().unwrap().as_mut().unwrap().write(slot, Some(tracked_entry(0x6120, 0xA4)));
                Some((old, MutationOwner(Arc::clone(&drops)), MutationOwner(Arc::clone(&drops))))
            },
            |_| Some(words),
            |_, _, _| panic!("stale mutation ran"),
        );
        assert_eq!(result, None);
        let newer = TRACKED_SRVS.lock().unwrap().as_ref().unwrap().by_slot[&slot];
        assert_ne!(newer.generation, generation);
        assert_eq!(newer.resource, 0x6120);
        assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 2);
        end_mutation_test();
    }

    #[test]
    fn exact_srv_mutation_forgets_raw_mismatch_or_missing_heap_backing() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for missing in [false, true] {
            let slot = srv_slot(62 + missing as u64, 0);
            let (generation, words) = begin_mutation_test(tracked_entry(0x6210, 0xA8), slot);
            let snapshot = TRACKED_SRVS.lock().unwrap().as_ref().unwrap().by_slot[&slot];
            let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut changed = words;
            changed[5] ^= 1;
            let result = with_exact_tracked_srv_mutation_on(
                slot,
                generation,
                |_, _| Some((snapshot, MutationOwner(Arc::clone(&drops)), MutationOwner(Arc::clone(&drops)))),
                |_| (!missing).then_some(changed),
                |_, _, _| panic!("invalid raw descriptor mutation ran"),
            );
            assert_eq!(result, None);
            assert!(!TRACKED_SRVS.lock().unwrap().as_ref().unwrap().by_slot.contains_key(&slot));
            assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 2);
            end_mutation_test();
        }
    }

    #[test]
    fn exact_srv_mutation_fails_closed_when_resource_or_heap_lease_is_gone() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let slot = srv_slot(64, 0);
        let (generation, _) = begin_mutation_test(tracked_entry(0x6410, 0xAC), slot);
        let result = with_exact_tracked_srv_mutation_on(
            slot,
            generation,
            |_, _| None::<(TrackedSrv, (), ())>,
            |_| panic!("observed words without strong owners"),
            |_, _, _| panic!("mutation ran without strong owners"),
        );
        assert_eq!(result, None);
        assert!(TRACKED_SRVS.lock().unwrap().as_ref().unwrap().by_slot.contains_key(&slot));
        end_mutation_test();
    }

    #[test]
    fn exact_srv_batch_deduplicates_and_revalidates_mixed_candidates_once() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SRV_TRACKING_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
        *SHARED_VIEW_EPOCHS.lock().unwrap() = SharedViewEpochs::new();
        let (a, b, c, d, e) = (
            srv_slot(70, 0), srv_slot(70, 1), srv_slot(70, 2), srv_slot(70, 3), srv_slot(70, 4),
        );
        let mut tracked = TrackedSrvs::default();
        for (slot, resource, pc) in [(a, 0xA010, 0xB0), (b, 0xB010, 0xB4), (c, 0xC010, 0xB8), (d, 0xD010, 0xBC), (e, 0xE010, 0xC0)] {
            tracked.write(slot, Some(tracked_entry(resource, pc)));
        }
        let old_b = tracked.by_slot[&b];
        let snapshots = tracked.by_slot.clone();
        tracked.write(b, Some(tracked_entry(0xB020, 0xB4)));
        let newer_b = tracked.by_slot[&b];
        *TRACKED_SRVS.lock().unwrap() = Some(tracked);

        let candidates = [
            (a, snapshots[&a].generation),
            (a, snapshots[&a].generation),
            (b, old_b.generation),
            (c, snapshots[&c].generation),
            (d, snapshots[&d].generation),
            (e, snapshots[&e].generation),
        ];
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut lease_calls = Vec::new();
        let mut observations = 0;
        let result = with_exact_tracked_srv_mutations_on(
            &candidates,
            |slot, generation| {
                assert!(SHARED_VIEW_EPOCHS.try_lock().is_err(), "lease escaped the batch epoch");
                assert!(TRACKED_SRV_WRITES.try_lock().is_ok(), "WRITE was taken while leasing owners");
                lease_calls.push((slot, generation));
                (slot != d).then(|| {
                    let snapshot = snapshots[&slot];
                    (snapshot, MutationOwner(Arc::clone(&drops)), MutationOwner(Arc::clone(&drops)))
                })
            },
            |slots| {
                observations += 1;
                assert!(TRACKED_SRV_WRITES.try_lock().is_err(), "raw words were observed outside WRITE");
                assert!(TRACKED_SRVS.try_lock().is_ok(), "tracker lock crossed raw observation");
                assert_eq!(slots, &[(70, 0), (70, 1), (70, 2), (70, 4)]);
                let mut changed_c = snapshots[&c].words;
                changed_c[6] ^= 1;
                vec![Some(snapshots[&a].words), Some(old_b.words), Some(changed_c), Some(snapshots[&e].words)]
            },
            |valid| {
                assert!(SHARED_VIEW_EPOCHS.try_lock().is_err(), "batch mutation escaped its epoch");
                assert!(TRACKED_SRV_WRITES.try_lock().is_err(), "batch mutation escaped WRITE");
                assert!(TRACKED_SRVS.try_lock().is_ok(), "tracker lock crossed batch mutation");
                assert_eq!(valid.iter().map(|entry| entry.entry.resource).collect::<Vec<_>>(), vec![0xA010, 0xE010]);
                // Touch both strong-owner references: the batch exposes only validated owners.
                assert!(valid.iter().all(|entry| Arc::ptr_eq(&entry.resource.0, &entry.heap.0)));
                29
            },
        );
        assert_eq!(result, Some(29));
        assert_eq!(lease_calls, vec![
            (a, snapshots[&a].generation), (b, old_b.generation), (c, snapshots[&c].generation),
            (d, snapshots[&d].generation), (e, snapshots[&e].generation),
        ]);
        assert_eq!(observations, 1, "batch observed raw words more than once");
        assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 8, "strong owners did not drop after the batch");
        {
            let tracked = TRACKED_SRVS.lock().unwrap();
            let tracked = tracked.as_ref().unwrap();
            assert_eq!(tracked.by_slot[&b].generation, newer_b.generation, "stale generation removed its replacement");
            assert!(!tracked.by_slot.contains_key(&c), "raw mismatch remained tracked");
            assert!(tracked.by_slot.contains_key(&d), "missing lease removed a still-current candidate");
        }
        end_mutation_test();
    }

    #[test]
    fn exact_srv_batch_empty_or_tracking_disabled_does_no_work() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SRV_TRACKING_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
        if !srv_tracking_enabled() {
            let candidate = [(srv_slot(80, 0), 1)];
            assert_eq!(with_exact_tracked_srv_mutations_on(
                &candidate,
                |_, _| panic!("tracking-disabled batch leased an owner"),
                |_| panic!("tracking-disabled batch observed words"),
                |_: &[ExactSrvMutation<'_, (), ()>]| panic!("tracking-disabled batch mutated"),
            ), None);
        }
        SRV_TRACKING_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(with_exact_tracked_srv_mutations_on(
            &[],
            |_, _| panic!("empty batch leased an owner"),
            |_| panic!("empty batch observed words"),
            |_: &[ExactSrvMutation<'_, (), ()>]| panic!("empty batch mutated"),
        ), None);
        let slot = srv_slot(81, 0);
        let (generation, _) = begin_mutation_test(tracked_entry(0x8110, 0xD0), slot);
        let snapshot = TRACKED_SRVS.lock().unwrap().as_ref().unwrap().by_slot[&slot];
        assert_eq!(with_exact_tracked_srv_mutations_on(
            &[(slot, generation)],
            |_, _| Some((snapshot, (), ())),
            |_| vec![None],
            |_: &[ExactSrvMutation<'_, (), ()>]| panic!("all-stale batch mutated"),
        ), None);
        assert!(!TRACKED_SRVS.lock().unwrap().as_ref().unwrap().by_slot.contains_key(&slot));
        end_mutation_test();
    }

    #[test]
    fn exact_srv_batch_unwind_drops_guards_before_strong_owners() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let slot = srv_slot(82, 0);
        let (generation, words) = begin_mutation_test(tracked_entry(0x8210, 0xD4), slot);
        let snapshot = TRACKED_SRVS.lock().unwrap().as_ref().unwrap().by_slot[&slot];
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let unlocked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_exact_tracked_srv_mutations_on(
                &[(slot, generation)],
                |_, _| Some((
                    snapshot,
                    UnwindMutationOwner { drops: Arc::clone(&drops), unlocked: Arc::clone(&unlocked) },
                    UnwindMutationOwner { drops: Arc::clone(&drops), unlocked: Arc::clone(&unlocked) },
                )),
                |_| vec![Some(words)],
                |_| -> () { panic!("test batch unwind") },
            )
        }));
        assert!(outcome.is_err());
        assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(unlocked.load(std::sync::atomic::Ordering::Relaxed), 3,
            "strong owners unwound before WRITE or tracker lock");
        TRACKED_SRV_WRITES.clear_poison();
        SHARED_VIEW_EPOCHS.clear_poison();
        end_mutation_test();
    }

    #[test]
    fn tracked_srv_copy_requires_exact_source_words() {
        let mut tracked = TrackedSrvs::default();
        let (source, copied, raw_changed) = (srv_slot(1, 0), srv_slot(2, 0), srv_slot(2, 1));
        tracked.write(source, Some(tracked_entry(10, 100)));
        let source_words = tracked.by_slot[&source].words;
        let mut copied_words = source_words;
        copied_words[0] = 21;
        tracked.copy_range(8, &[(source, 201, 202, 21, copied, true, source_words, copied_words)]);
        assert_eq!(tracked.by_slot[&copied].resource, 10);

        let mut changed_words = source_words;
        changed_words[1] ^= 1;
        tracked.copy_range(8, &[(source, 301, 302, 31, raw_changed, true, changed_words, copied_words)]);
        assert!(!tracked.by_slot.contains_key(&raw_changed));
        tracked.write(raw_changed, Some(tracked_entry(30, 300)));
        tracked.copy_range(8, &[(source, 301, 302, 31, raw_changed, false, source_words, copied_words)]);
        assert!(!tracked.by_slot.contains_key(&raw_changed), "an untrackable source clears the destination");
    }

    #[test]
    fn tracked_srv_reconciliation_invalidates_same_index_word_change() {
        let mut tracked = TrackedSrvs::default();
        let slot = srv_slot(1, 0);
        tracked.write(slot, Some(tracked_entry(10, 100)));
        let generation = tracked.by_slot[&slot].generation;
        let mut changed = tracked.by_slot[&slot].words;
        changed[1] ^= 1;
        assert_eq!(changed[0], tracked.by_slot[&slot].words[0], "the PC descriptor index is unchanged");
        assert!(tracked.reconcile_observed(&[(slot, generation, Some(changed))]).is_empty());
        assert!(tracked.by_slot.is_empty(), "all eight console words identify the logical descriptor");
    }

    #[test]
    fn tracked_srv_reconciliation_rejects_cross_heap_local_index_collision() {
        let mut tracked = TrackedSrvs::default();
        let (source, destination) = (srv_slot(1, 3), srv_slot(2, 3));
        tracked.write(source, Some(tracked_entry(10, 6)));
        tracked.write(destination, Some(tracked_entry(20, 6)));
        let generation = tracked.by_slot[&destination].generation;
        let copied_words = tracked.by_slot[&source].words;
        assert_eq!(copied_words[0], tracked.by_slot[&destination].words[0], "fallback heaps repeat local index 2*k");
        assert!(tracked.reconcile_observed(&[(destination, generation, Some(copied_words))]).is_empty());
        assert!(!tracked.by_slot.contains_key(&destination));
        assert!(tracked.by_slot.contains_key(&source));
    }

    #[test]
    fn tracked_srv_reconciliation_does_not_remove_a_newer_generation() {
        let mut tracked = TrackedSrvs::default();
        let slot = srv_slot(1, 4);
        tracked.write(slot, Some(tracked_entry(10, 8)));
        let observed_generation = tracked.by_slot[&slot].generation;
        tracked.write(slot, Some(tracked_entry(20, 8)));
        let stale_words = tracked_entry(10, 8).words;
        assert!(tracked.reconcile_observed(&[(slot, observed_generation, Some(stale_words))]).is_empty());
        assert_eq!(tracked.by_slot[&slot].resource, 20);
    }

    #[test]
    fn tracked_srv_global_cleanup_hooks_are_gated_and_exact() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SRV_TRACKING_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
        PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).extend([10, 20]);
        let (heap_40, heap_41) = (srv_slot(40, 0), srv_slot(41, 0));
        let mut tracked = TrackedSrvs::default();
        tracked.write(heap_40, Some(tracked_entry(10, 100)));
        tracked.write(heap_41, Some(tracked_entry(20, 200)));
        *TRACKED_SRVS.lock().unwrap() = Some(tracked);
        forget_srv_heap(40);
        forget(20);
        assert!(TRACKED_SRVS.lock().unwrap().as_ref().is_some_and(|tracked| tracked.by_slot.is_empty()));
        PLACED_TEXTURES.lock().unwrap().as_mut().unwrap().remove(&10);
        TRACKED_SRVS.lock().unwrap().take();
        SRV_TRACKING_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn tracked_srv_lease_excludes_final_release_and_locked_cleanup() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SRV_TRACKING_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
        let slot = srv_slot(52, 3);
        let mut tracked = TrackedSrvs::default();
        tracked.write(slot, Some(tracked_entry(0x5210, 0x80)));
        let generation = tracked.by_slot[&slot].generation;
        *TRACKED_SRVS.lock().unwrap() = Some(tracked);

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (continue_tx, continue_rx) = std::sync::mpsc::channel();
        let lease = std::thread::spawn(move || {
            lease_tracked_srv_resource_with(slot, generation, |resource| {
                entered_tx.send(()).unwrap();
                continue_rx.recv().unwrap();
                Some(resource)
            })
            .map(|(entry, owner)| (entry.resource, owner))
        });
        entered_rx.recv().unwrap();

        let (attempted_tx, attempted_rx) = std::sync::mpsc::channel();
        let (released_tx, released_rx) = std::sync::mpsc::channel();
        let release = std::thread::spawn(move || {
            attempted_tx.send(()).unwrap();
            let _lifetime = lock_tracked_srv_lifetime();
            released_tx.send(()).unwrap();
        });
        attempted_rx.recv().unwrap();
        assert!(released_rx.recv_timeout(std::time::Duration::from_millis(50)).is_err(),
            "final Release crossed a resource lease in progress");
        continue_tx.send(()).unwrap();
        assert_eq!(lease.join().unwrap(), Some((0x5210, 0x5210)));
        released_rx.recv().unwrap();
        release.join().unwrap();

        let lifetime = lock_tracked_srv_lifetime();
        forget_tracked_after_resource_release(0x5210, lifetime.as_ref());
        assert!(TRACKED_SRVS.lock().unwrap().as_ref().is_some_and(|tracked| tracked.by_slot.is_empty()));
        drop(lifetime);
        TRACKED_SRVS.lock().unwrap().take();
        SRV_TRACKING_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn descriptor_write_allows_nonfinal_release_before_concurrent_lease() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SRV_TRACKING_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
        let slot = srv_slot(53, 2);
        let mut tracked = TrackedSrvs::default();
        tracked.write(slot, Some(tracked_entry(0x5310, 0x84)));
        let generation = tracked.by_slot[&slot].generation;
        *TRACKED_SRVS.lock().unwrap() = Some(tracked);

        let (release_entered_tx, release_entered_rx) = std::sync::mpsc::channel();
        let (finish_release_tx, finish_release_rx) = std::sync::mpsc::channel();
        let (drop_write_tx, drop_write_rx) = std::sync::mpsc::channel();
        let hook = std::thread::spawn(move || {
            let _write = lock_tracked_srv_write();
            // Model a driver temporary Release reentering while a descriptor hook holds WRITE.
            // A non-final Release takes only LIFETIME, then returns to the hook.
            let lifetime = lock_tracked_srv_lifetime();
            release_entered_tx.send(()).unwrap();
            finish_release_rx.recv().unwrap();
            drop(lifetime);
            drop_write_rx.recv().unwrap();
        });
        release_entered_rx.recv().unwrap();

        let (attempted_tx, attempted_rx) = std::sync::mpsc::channel();
        let (leased_tx, leased_rx) = std::sync::mpsc::channel();
        let lease = std::thread::spawn(move || {
            attempted_tx.send(()).unwrap();
            let leased = lease_tracked_srv_resource_with(slot, generation, Some)
                .map(|(entry, owner)| (entry.resource, owner));
            leased_tx.send(leased).unwrap();
        });
        attempted_rx.recv().unwrap();
        assert!(leased_rx.recv_timeout(std::time::Duration::from_millis(50)).is_err(),
            "a lease crossed the resource Release lifetime gate");
        finish_release_tx.send(()).unwrap();
        assert_eq!(leased_rx.recv().unwrap(), Some((0x5310, 0x5310)),
            "lease acquisition waited for the descriptor-write gate and deadlocked the hook");
        lease.join().unwrap();
        drop_write_tx.send(()).unwrap();
        hook.join().unwrap();

        let lifetime = lock_tracked_srv_lifetime();
        forget_tracked_after_resource_release(0x5310, lifetime.as_ref());
        drop(lifetime);
        TRACKED_SRVS.lock().unwrap().take();
        SRV_TRACKING_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn descriptor_heap_final_cleanup_does_not_invert_reentrant_write_order() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SRV_TRACKING_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
        let slot = srv_slot(55, 0);
        let mut tracked = TrackedSrvs::default();
        tracked.write(slot, Some(tracked_entry(0x5510, 0x90)));
        *TRACKED_SRVS.lock().unwrap() = Some(tracked);

        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (clean_tx, clean_rx) = std::sync::mpsc::channel();
        let (cleaned_tx, cleaned_rx) = std::sync::mpsc::channel();
        let final_release = std::thread::spawn(move || {
            let lifetime = super::super::console_heap::lock_lifetime();
            held_tx.send(()).unwrap();
            clean_rx.recv().unwrap();
            forget_srv_heap_after_release(55, lifetime.as_ref());
            cleaned_tx.send(()).unwrap();
            drop(lifetime);
        });
        held_rx.recv().unwrap();
        let (write_tx, write_rx) = std::sync::mpsc::channel();
        let (drop_tx, drop_rx) = std::sync::mpsc::channel();
        let hook = std::thread::spawn(move || {
            let _write = lock_tracked_srv_write();
            write_tx.send(()).unwrap();
            let lifetime = super::super::console_heap::lock_lifetime();
            drop(lifetime);
            drop_rx.recv().unwrap();
        });
        write_rx.recv().unwrap();
        clean_tx.send(()).unwrap();
        cleaned_rx.recv_timeout(std::time::Duration::from_secs(1)).expect("final cleanup must not wait for WRITE");
        final_release.join().unwrap();
        drop_tx.send(()).unwrap();
        hook.join().unwrap();
        assert!(TRACKED_SRVS.lock().unwrap().as_ref().is_some_and(|tracked| tracked.by_slot.is_empty()));
        TRACKED_SRVS.lock().unwrap().take();
        SRV_TRACKING_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn final_release_cleanup_does_not_wait_for_unrelated_descriptor_write() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SRV_TRACKING_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
        let (slot_a, slot_b) = (srv_slot(54, 0), srv_slot(54, 1));
        let mut tracked = TrackedSrvs::default();
        tracked.write(slot_a, Some(tracked_entry(0x5410, 0x88)));
        tracked.write(slot_b, Some(tracked_entry(0x5420, 0x8c)));
        *TRACKED_SRVS.lock().unwrap() = Some(tracked);

        let (lifetime_held_tx, lifetime_held_rx) = std::sync::mpsc::channel();
        let (clean_tx, clean_rx) = std::sync::mpsc::channel();
        let (cleaned_tx, cleaned_rx) = std::sync::mpsc::channel();
        let final_release = std::thread::spawn(move || {
            let lifetime = lock_tracked_srv_lifetime();
            lifetime_held_tx.send(()).unwrap();
            clean_rx.recv().unwrap();
            forget_tracked_after_resource_release(0x5410, lifetime.as_ref());
            cleaned_tx.send(()).unwrap();
            drop(lifetime);
        });
        lifetime_held_rx.recv().unwrap();

        let (write_held_tx, write_held_rx) = std::sync::mpsc::channel();
        let (release_attempted_tx, release_attempted_rx) = std::sync::mpsc::channel();
        let (release_returned_tx, release_returned_rx) = std::sync::mpsc::channel();
        let (drop_write_tx, drop_write_rx) = std::sync::mpsc::channel();
        let unrelated_hook = std::thread::spawn(move || {
            let _write = lock_tracked_srv_write();
            write_held_tx.send(()).unwrap();
            // Model resource B's driver call making a temporary Release while WRITE is held.
            release_attempted_tx.send(()).unwrap();
            let lifetime = lock_tracked_srv_lifetime();
            release_returned_tx.send(()).unwrap();
            drop(lifetime);
            drop_write_rx.recv().unwrap();
        });
        write_held_rx.recv().unwrap();
        release_attempted_rx.recv().unwrap();

        clean_tx.send(()).unwrap();
        cleaned_rx.recv_timeout(std::time::Duration::from_secs(1)).expect(
            "resource A final cleanup waited for WRITE while resource B Release waited for LIFETIME",
        );
        final_release.join().unwrap();
        release_returned_rx.recv().unwrap();
        drop_write_tx.send(()).unwrap();
        unrelated_hook.join().unwrap();

        {
            let tracked = TRACKED_SRVS.lock().unwrap();
            let tracked = tracked.as_ref().unwrap();
            assert!(!tracked.by_slot.contains_key(&slot_a));
            assert_eq!(tracked.by_slot[&slot_b].resource, 0x5420);
        }
        TRACKED_SRVS.lock().unwrap().take();
        SRV_TRACKING_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn title_queue_mapping_wait_reservation_is_monotonic_and_per_queue() {
        let mut waited = HashMap::new();
        assert!(!reserve_page_wait(&mut waited, 1, 0));
        assert!(reserve_page_wait(&mut waited, 1, 4));
        assert!(!reserve_page_wait(&mut waited, 1, 4));
        assert!(!reserve_page_wait(&mut waited, 1, 3));
        assert!(reserve_page_wait(&mut waited, 2, 3));
        assert!(reserve_page_wait(&mut waited, 1, 5));
        assert_eq!(waited, HashMap::from([(1, 5), (2, 3)]));
    }
    use std::time::{Duration, Instant};

    const MIB: usize = 1 << 20;

    /// The levels of a texture with lazy levels: a group of tiles for each level with tiles of its
    /// own and one for the packed levels; a level past the last standard one is in the packed
    /// group; a tiling that does not add up to the resource's tiles gives none.
    #[test]
    fn the_tiles_of_a_texture_are_grouped_by_its_levels() {
        // 2048x2048 BC7, 12 levels: 64, 16, 4, 1 tiles and 1 for the 8 packed levels
        let l = Levels::new(&[(0, 64), (64, 16), (80, 4), (84, 1)], Some((85, 1)), 86, 12).unwrap();
        assert_eq!(l.groups, vec![(0, 0, 64, false), (1, 64, 16, false), (2, 80, 4, false), (3, 84, 1, false), (4, 85, 1, false)]);
        assert_eq!((0..12).map(|i| l.group_of(i)).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4]);
        assert_eq!((l.levels_of(0), l.levels_of(3), l.levels_of(4)), (0..1, 3..4, 4..12));
        // all of it packed: one group
        let small = Levels::new(&[], Some((0, 1)), 1, 5).unwrap();
        assert_eq!((small.group_of(0), small.group_of(4), small.levels_of(0)), (0, 0, 0..5));
        // no packed levels: the last level's group ends at the texture's levels
        let plain = Levels::new(&[(0, 4), (4, 1)], None, 5, 2).unwrap();
        assert_eq!((plain.group_of(1), plain.levels_of(1)), (1, 1..2));
        // a gap, an overlap, a tile count that is not the resource's, nothing at all
        assert_eq!(Levels::new(&[(0, 4), (5, 1)], None, 6, 2), None);
        assert_eq!(Levels::new(&[(0, 4), (3, 2)], None, 5, 2), None);
        assert_eq!(Levels::new(&[(0, 4), (4, 1)], None, 6, 2), None);
        assert_eq!(Levels::new(&[(0, 4), (4, 0)], None, 4, 2), None);
        assert_eq!(Levels::new(&[], None, 0, 1), None);
    }

    #[test]
    fn level_tile_owners_round_trip_through_a_deferred_unmap() {
        let mut levels = Levels::new(&[(0, 2), (2, 1)], None, 3, 2).unwrap();
        levels.group_tiles[0] = (vec![0x1000, 0x2000], vec![(0x3000, 0)]);
        levels.group_tiles[1] = (vec![0x4000], Vec::new());
        let expected = levels.group_tiles.clone();

        let held = take_level_tiles(Some(&mut levels));
        assert_eq!(held, expected);
        assert!(levels.group_tiles.iter().all(|(used, extra)| used.is_empty() && extra.is_empty()));

        levels.group_tiles = held;
        assert_eq!(levels.group_tiles, expected, "cancelling a deferred unmap restores every group owner");
    }

    #[test]
    fn resource_release_takes_every_level_group_owner() {
        let mut levels = Levels::new(&[(0, 1), (1, 1)], Some((2, 1)), 3, 4).unwrap();
        levels.group_tiles[0].0.push(0x1000);
        levels.group_tiles[1].1.push((0x2000, 1));
        levels.group_tiles[2].0.push(0x3000);

        let released = take_level_tiles(Some(&mut levels));
        let used: Vec<_> = released.iter().flat_map(|owners| owners.0.iter().copied()).collect();
        let extra: Vec<_> = released.iter().flat_map(|owners| owners.1.iter().copied()).collect();
        assert_eq!(used, [0x1000, 0x3000]);
        assert_eq!(extra, [(0x2000, 1)]);
        assert!(levels.group_tiles.iter().all(|(used, extra)| used.is_empty() && extra.is_empty()));
    }

    #[test]
    fn test_mapping_observation_includes_level_group_owners() {
        let mut levels = Levels::new(&[(0, 1)], Some((1, 1)), 2, 2).unwrap();
        levels.group_tiles[0].0.push(0x1000);
        levels.group_tiles[1].1.push((0x2000, 1));
        let target = Target {
            va: 0x1000,
            console_bytes: None,
            total: 2,
            mapped: true,
            used: Vec::new(),
            used_extra: Vec::new(),
            levels: Some(levels),
            texels: None,
            last_touch: Instant::now(),
            min_resident_mip: 0,
        };

        assert_eq!(target_page_keys(&target), (vec![0x1000], vec![(0x2000, 1)]));
    }

    #[test]
    fn shared_page_identity_is_the_title_address_not_allocation_class() {
        let mut pages = HashMap::new();
        for _evictable in [false, true] {
            pages
                .entry(0x1000)
                .and_modify(|page: &mut Page| page.users += 1)
                .or_insert(Page { tile: (0, 1), users: 1, idle_since: None });
        }

        // Both policies share one page. `heap_evictable` is only a packing hint for fresh tiles;
        // future eviction must inspect live page owners rather than trusting the heap class.
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[&0x1000].users, 2);
        assert_eq!(pages[&0x1000].tile, (0, 1));
    }

    #[test]
    fn packed_console_3d_mip_tails_use_the_detile_upload_path() {
        fn desc(
            dimension: u32,
            width: u64,
            height: u32,
            depth: u16,
            mips: u16,
            format: u32,
            layout: u32,
        ) -> D3D12_RESOURCE_DESC {
            let mut bytes = [0u8; 0x38];
            bytes[0..4].copy_from_slice(&dimension.to_le_bytes());
            bytes[0x10..0x18].copy_from_slice(&width.to_le_bytes());
            bytes[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
            bytes[0x1C..0x1E].copy_from_slice(&depth.to_le_bytes());
            bytes[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
            bytes[0x20..0x24].copy_from_slice(&format.to_le_bytes());
            bytes[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            bytes[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
            unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const D3D12_RESOURCE_DESC) }
        }

        let packed_3d = desc(4, 64, 64, 64, 4, 28, 0x105);
        assert!(console_has_packed_3d_mips(&packed_3d));
        let single_mip_3d = desc(4, 64, 64, 64, 1, 28, 0x105);
        assert!(!console_has_packed_3d_mips(&single_mip_3d));
        let packed_2d = desc(3, 64, 64, 1, 7, 28, 0x105);
        assert!(!console_has_packed_3d_mips(&packed_2d));
    }

    /// One call per heap: its tiles in runs, the other heaps' skipped.
    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)];
        assert_eq!(heap_ranges(0, &tiles), [(true, 5, 2), (false, 0, 2), (true, 9, 3)]);
        assert_eq!(heap_ranges(1, &tiles), [(false, 0, 2), (true, 0, 2), (false, 0, 3)]);
        assert_eq!(heap_ranges_trimmed(1, &tiles), [(false, 0, 2), (true, 0, 2)]);
        // shared pages out of order: runs break where the tiles do
        assert_eq!(heap_ranges(0, &[(0, 3), (0, 1), (0, 2)]), [(true, 3, 1), (true, 1, 2)]);
    }

    /// Streaming churn across 100+ 64 MiB pool heaps: cooling tiles returned in interleaved
    /// cross-heap order are regrouped per heap and sorted into contiguous ascending runs, so new
    /// textures stay inside 1 pool heap (1 UpdateTileMappings call, 1 contiguous range) instead of
    /// scattering across dozens of heaps.
    #[test]
    fn tile_pool_keeps_churned_allocations_in_single_heap_and_contiguous() {
        let t0 = Instant::now();
        let mut pool = TilePool::new();
        let heaps_made = std::cell::Cell::new(0u32);
        let mut make = || {
            let h = heaps_made.get();
            heaps_made.set(h + 1);
            Some(h)
        };

        // Fill 100 pools (64 tiles each for a fast unit test, same multi-heap topology as 1024).
        const P: u32 = 64;
        let mut live: Vec<Vec<Tile>> = Vec::new();
        for _ in 0..100 {
            live.push(pool.take_batch(P as usize, None, false, t0, P, &mut make).unwrap());
        }
        assert_eq!(heaps_made.get(), 100);
        assert_eq!(pool.free_total, 0);

        // Free 32 tiles from every pool in interleaved round-robin order (worst-case churn across
        // all 100 pools in `cooling`).
        for tile_idx in 0..32usize {
            for h in 0..100usize {
                pool.push_cooling(live[h][tile_idx], t0);
            }
        }
        assert_eq!(pool.cooling.len(), 3200);

        // After COOL, allocating 100 textures of 32 tiles each must:
        // - allocate ZERO new heaps,
        // - place EVERY texture into exactly 1 single pool heap (1 UpdateTileMappings call),
        // - yield 1 single contiguous tile range (0..32) per texture!
        let t1 = t0 + COOL + Duration::from_millis(1);
        for _ in 0..100 {
            let alloc = pool.take_batch(32, None, false, t1, P, &mut make).unwrap();
            assert_eq!(alloc.len(), 32);
            let h0 = alloc[0].0;
            assert!(alloc.iter().all(|&(h, _)| h == h0), "texture scattered across multiple heaps");
            let ranges = heap_ranges_trimmed(h0, &alloc);
            assert_eq!(ranges, [(true, 0, 32)], "expected 1 contiguous ascending range");
        }
        assert_eq!(heaps_made.get(), 100, "must reuse cooled tiles without making new heaps");
        assert_eq!(pool.free_total, 0);

        // Partial overlap preference: if heap 7 has free tiles and preferred=Some(7), use heap 7.
        for t in 10..18u32 {
            pool.push_cooling((7, t), t1);
            pool.push_cooling((3, t), t1);
        }
        let t2 = t1 + COOL + Duration::from_millis(1);
        let pref = pool.take_batch(8, Some(7), false, t2, P, &mut make).unwrap();
        assert!(pref.iter().all(|&(h, _)| h == 7), "preferred heap should be chosen");
        assert_eq!(heap_ranges_trimmed(7, &pref), [(true, 10, 8)]);

        // Rollback on OOM: requesting more tiles than available when make_heap fails returns every
        // popped tile cleanly so free_total is preserved.
        assert_eq!(pool.free_total, 8);
        let fail = pool.take_batch(16, None, false, t2, P, || None);
        assert!(fail.is_none());
        assert_eq!(pool.free_total, 8, "OOM rollback must return all popped tiles");
    }

    #[test]
    fn tile_pool_keeps_evictable_and_pinned_tiles_in_separate_heaps() {
        let t0 = Instant::now();
        let mut pool = TilePool::new();
        let next_slot = std::cell::Cell::new(0u32);
        let mut alloc_slot = || {
            let slot = next_slot.get();
            next_slot.set(slot + 1);
            Some(slot)
        };

        const P: u32 = 64;
        let evictable = pool.take_batch(32, None, true, t0, P, &mut alloc_slot).unwrap();
        let pinned = pool.take_batch(32, None, false, t0, P, &mut alloc_slot).unwrap();
        let evictable_again = pool.take_batch(32, None, true, t0, P, &mut alloc_slot).unwrap();

        assert!(evictable.iter().chain(&evictable_again).all(|&(h, _)| h == evictable[0].0));
        assert!(pinned.iter().all(|&(h, _)| h == pinned[0].0));
        assert_ne!(evictable[0].0, pinned[0].0, "allocation classes must not share a pool heap");
    }

    #[test]
    fn tile_pool_keeps_one_warm_spare_for_each_allocation_class() {
        const P: u32 = 64;
        let mut pool = TilePool::new();
        pool.add_heap_at(0, P, false);
        pool.add_heap_at(1, P, false);
        pool.add_heap_at(2, P, true);
        pool.add_heap_at(3, P, true);

        assert_eq!(pool.reclaim_empty_heaps(1, P), [1, 3]);
        assert_eq!(pool.free_total, 2 * P as usize);
        assert_eq!(pool.pick_heap(P as usize, None, false), Some(0));
        assert_eq!(pool.pick_heap(P as usize, None, true), Some(2));
    }

    /// Empty cooled pool heaps (above the warm spare count) are reclaimed cleanly back to VRAM and
    /// their slot indices can be reused without growing the heap table.
    #[test]
    fn tile_pool_reclaims_empty_heaps_keeping_warm_spare_and_reuses_slots() {
        let t0 = Instant::now();
        let mut pool = TilePool::new();
        let free_slots: std::cell::RefCell<Vec<u32>> = std::cell::RefCell::new(Vec::new());
        let next_slot = std::cell::Cell::new(0u32);
        let mut alloc_slot = || {
            free_slots.borrow_mut().pop().or_else(|| {
                let h = next_slot.get();
                next_slot.set(h + 1);
                Some(h)
            })
        };

        const P: u32 = 64;
        let h0 = pool.take_batch(P as usize, None, false, t0, P, &mut alloc_slot).unwrap();
        let h1 = pool.take_batch(P as usize, None, false, t0, P, &mut alloc_slot).unwrap();
        let h2 = pool.take_batch(P as usize, None, false, t0, P, &mut alloc_slot).unwrap();
        let h3 = pool.take_batch(P as usize, None, false, t0, P, &mut alloc_slot).unwrap();
        assert_eq!(next_slot.get(), 4);
        assert_eq!(pool.free_total, 0);

        // Keep 1 tile live in heap 0; free all tiles in heaps 1, 2, 3.
        for &tile in &h0[1..] {
            pool.push_cooling(tile, t0);
        }
        for batch in [&h1, &h2, &h3] {
            for &tile in batch {
                pool.push_cooling(tile, t0);
            }
        }
        let t1 = t0 + COOL + Duration::from_millis(1);
        pool.drain_cooling(t1);
        assert_eq!(pool.free_total, 63 + 64 * 3);

        // Reclaim empty heaps keeping 1 warm spare: heaps 2 and 3 reclaim, heap 1 stays warm,
        // heap 0 still has 1 live tile so it is untouched.
        let reclaimed = pool.reclaim_empty_heaps(1, P);
        assert_eq!(reclaimed, vec![2, 3]);
        assert_eq!(pool.free_total, 63 + 64);
        free_slots.borrow_mut().extend(reclaimed);

        // First 64-tile allocation reuses warm spare heap 1 (no new heap allocated).
        let reuse_spare = pool.take_batch(P as usize, None, false, t1, P, &mut alloc_slot).unwrap();
        assert!(reuse_spare.iter().all(|&(h, _)| h == 1));
        assert_eq!(free_slots.borrow().len(), 2);

        // Next 64-tile allocation consumes heap 0's 63 free tiles plus 1 tile from reclaimed slot 3
        // without advancing `next_slot`.
        let reuse_slot = pool.take_batch(P as usize, None, false, t1, P, &mut alloc_slot).unwrap();
        assert_eq!(reuse_slot.iter().filter(|&&(h, _)| h == 0).count(), 63);
        assert_eq!(reuse_slot.iter().filter(|&&(h, _)| h == 3).count(), 1);
        assert_eq!(next_slot.get(), 4);
        assert_eq!(free_slots.borrow().len(), 1);
    }

    #[test]
    fn arenas_sweep_dropped_returns_expired_heaps() {
        let t0 = Instant::now();
        let mut arenas: Arenas<u32> = Arenas::new();
        let (_, _, id1) = arenas.take_for(1, 0x10_0000, 0x1_0000, PAGE, t0, |_, _| Some(101)).0.unwrap();
        let (_, _, id2) = arenas.take_for(1, 0x20_0000, 0x1_0000, PAGE, t0, |_, _| Some(202)).0.unwrap();
        arenas.release(id1, t0);
        arenas.release(id2, t0 + Duration::from_secs(1));
        assert!(arenas.sweep_dropped_for(1, t0 + Duration::from_millis(1900)).is_empty());
        assert_eq!(arenas.sweep_dropped_for(1, t0 + IDLE + Duration::from_millis(10)), vec![101]);
        assert_eq!(arenas.bytes(), 0x1_0000);
        assert_eq!(arenas.sweep_dropped_for(1, t0 + Duration::from_secs(1) + IDLE + Duration::from_millis(10)), vec![202]);
        assert_eq!(arenas.bytes(), 0);
    }

    #[test]
    fn arenas_sweep_dropped_preserves_newest_fitting_order() {
        let t0 = Instant::now();
        let mut arenas: Arenas<u32> = Arenas::new();
        let id1 = arenas.take_for(1, 0x10_0000, PAGE, PAGE, t0, |_, _| Some(101)).0.unwrap().2;
        let id2 = arenas.take_for(1, 0x20_0000, PAGE, PAGE, t0, |_, _| Some(202)).0.unwrap().2;
        let id3 = arenas.take_for(1, 0x20_0000, 2 * PAGE, PAGE, t0, |_, _| Some(303)).0.unwrap().2;
        arenas.release(id1, t0);
        let t1 = t0 + IDLE + Duration::from_millis(10);
        assert_eq!(arenas.sweep_dropped_for(1, t1), vec![101]);
        let (heap, off, picked_id) = arenas.take_for(1, 0x20_0000, PAGE, PAGE, t1, |_, _| panic!("must reuse existing arena")).0.unwrap();
        assert_eq!((heap, off, picked_id), (303, 0, id3), "sweep must preserve insertion order so rposition selects the newest fitting arena");
        assert_ne!(picked_id, id2);
    }

    #[test]
    fn arenas_isolate_devices_and_take_for_returns_expired_heaps() {
        use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
        struct DropFlag(Arc<AtomicUsize>);
        impl Drop for DropFlag { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }
        let (t0, d1, d2) = (Instant::now(), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let mut arenas: Arenas<Arc<DropFlag>> = Arenas::new();
        let (h1, _, id1) = arenas.take_for(0x1000, 0x10_0000, PAGE, PAGE, t0, |_, _| Some(Arc::new(DropFlag(d1.clone())))).0.unwrap();
        let (h2, _, id2) = arenas.take_for(0x2000, 0x10_0000, PAGE, PAGE, t0, |_, _| Some(Arc::new(DropFlag(d2.clone())))).0.unwrap();
        assert_ne!(id1, id2, "distinct devices at the same VA must never alias arenas");
        drop((h1, h2));
        arenas.release(id1, t0);
        arenas.release(id2, t0);
        let t1 = t0 + IDLE + Duration::from_millis(10);
        let exp1 = arenas.sweep_dropped_for(0x1000, t1);
        assert_eq!((exp1.len(), arenas.bytes(), d1.load(Ordering::SeqCst), d2.load(Ordering::SeqCst)), (1, PAGE, 0, 0));
        let (_, exp2) = arenas.take_for(0x2000, 0x20_0000, PAGE, PAGE, t1, |_, _| Some(Arc::new(DropFlag(Arc::new(AtomicUsize::new(0))))));
        assert_eq!((exp2.len(), d2.load(Ordering::SeqCst)), (1, 0), "take_for must not drop expired heaps inside ARENAS");
        drop((exp1, exp2));
        assert_eq!((d1.load(Ordering::SeqCst), d2.load(Ordering::SeqCst)), (1, 1));
    }

    #[test]
    fn retire_queue_waits_for_all_queues_and_prunes_dead_queue_fences() {
        use std::sync::{atomic::{AtomicU64, Ordering}, Arc};
        let (mut rq, unmap_f) = (RetireQueue::<u32, Arc<AtomicU64>>::new(), Arc::new(AtomicU64::new(0)));
        let mut qf: std::collections::HashMap<usize, Arc<AtomicU64>> = std::collections::HashMap::new();
        let mk = || Some(Arc::new(AtomicU64::new(0)));
        assert!(rq.retire_with(vec![101], Some((unmap_f.clone(), 1)), &[(1, 1usize), (2, 2), (3, 3)], mk, |q, f, _| { qf.entry(*q).or_insert_with(|| f.clone()); true }).is_empty());
        assert!(rq.retire_with(vec![202], None, &[(1, 1usize), (2, 2), (4, 4)], mk, |q, f, _| { qf.entry(*q).or_insert_with(|| f.clone()); true }).is_empty());
        qf[&2].store(2, Ordering::SeqCst);
        qf[&3].store(1, Ordering::SeqCst);
        unmap_f.store(1, Ordering::SeqCst);
        let live: HashSet<usize> = [1, 2].into_iter().collect();
        let (done, pruned) = rq.sweep_with(&live, |f| f.load(Ordering::SeqCst));
        assert!(done.is_empty() && pruned.len() == 1 && Arc::ptr_eq(&pruned[0], &qf[&3]) && rq.fences.contains_key(&4));
        drop(pruned);
        qf[&1].store(1, Ordering::SeqCst);
        let (done, pruned) = rq.sweep_with(&live, |f| f.load(Ordering::SeqCst));
        assert_eq!((done.len(), done[0].heaps.as_slice(), pruned.len()), (1, &[101][..], 0));
        assert!(Arc::strong_count(&qf[&3]) > 1, "completed batch fences must drop outside lock");
        drop(done);
        qf[&1].store(2, Ordering::SeqCst);
        qf[&4].store(1, Ordering::SeqCst);
        let (done, pruned) = rq.sweep_with(&live, |f| f.load(Ordering::SeqCst));
        assert_eq!((done[0].heaps.as_slice(), pruned.len(), rq.batches.is_empty()), (&[202][..], 1, true));
        assert!(Arc::ptr_eq(&pruned[0], &qf[&4]));
        assert_eq!(rq.retire_with(vec![303], None, &[] as &[(usize, usize)], mk, |_, _, _| true), vec![303]);
    }

    #[test]
    fn deferred_unmap_requires_every_queue_mark() {
        use std::sync::{atomic::{AtomicU64, Ordering}, Arc};
        let first = Arc::new(AtomicU64::new(0));
        let second = Arc::new(AtomicU64::new(0));
        let marks = [(first.clone(), 2), (second.clone(), 1)];
        assert!(!all_marks_complete(&marks, |fence| fence.load(Ordering::SeqCst)));
        first.store(2, Ordering::SeqCst);
        assert!(!all_marks_complete(&marks, |fence| fence.load(Ordering::SeqCst)), "one completed queue must not release the mapping");
        second.store(1, Ordering::SeqCst);
        assert!(all_marks_complete(&marks, |fence| fence.load(Ordering::SeqCst)));
    }

    #[test]
    fn retirement_completion_poll_is_unlocked_and_new_marks_fail_closed() {
        use std::sync::{atomic::{AtomicU64, Ordering}, Arc};
        let first = Arc::new(AtomicU64::new(1));
        let late = Arc::new(AtomicU64::new(0));
        let state = Mutex::new(RetireQueue::<u32, Arc<AtomicU64>>::new());
        {
            let mut rq = state.lock().unwrap();
            rq.fences.insert(1, (first.clone(), 1));
            rq.batches.push(RetiringBatch { heaps: vec![101], unmap_mark: None, marks: vec![(first.clone(), 1)] });
        }

        let probes = state.lock().unwrap().completion_fences();
        let mut inserted = false;
        let observed: Vec<_> = probes.into_iter().map(|fence| {
            assert!(state.try_lock().is_ok(), "completion polling held retirement state");
            if !inserted {
                inserted = true;
                state.lock().unwrap().batches.extend([
                    RetiringBatch { heaps: vec![202], unmap_mark: None, marks: vec![(first.clone(), 2)] },
                    RetiringBatch { heaps: vec![303], unmap_mark: None, marks: vec![(late.clone(), 1)] },
                ]);
            }
            let completed = fence.load(Ordering::SeqCst);
            (fence, completed)
        }).collect();

        let (done, pruned) = state.lock().unwrap().sweep_with(&HashSet::from([1]), |fence| {
            observed_completion(&observed, fence, Arc::ptr_eq)
        });
        assert_eq!(done.iter().flat_map(|batch| &batch.heaps).copied().collect::<Vec<_>>(), [101]);
        assert!(pruned.is_empty());
        let rq = state.lock().unwrap();
        assert_eq!(rq.batches.len(), 2);
        assert_eq!(rq.batches[0].heaps, [202], "a higher requirement added during the poll must be tested against the observed value");
        assert_eq!(rq.batches[1].heaps, [303], "a new fence added during the poll had no observation and must remain retired");
    }

    #[test]
    fn device_retirement_state_is_leased_outside_the_registry() {
        let _guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (first_key, second_key) = (0x7ffe_7600usize, 0x7ffe_7700usize);
        {
            let mut registry = RETIRE.lock().unwrap();
            let map = registry.get_or_insert_with(std::collections::HashMap::new);
            map.remove(&first_key);
            map.remove(&second_key);
        }

        let first = device_retire_state(first_key);
        let second = device_retire_state(second_key);
        let first_guard = first.lock().unwrap();
        assert!(RETIRE.try_lock().is_ok(), "one device's driver turn must not hold the retirement registry");
        assert!(second.try_lock().is_ok(), "one device must not serialize another device's retirement work");

        let removed = RETIRE.lock().unwrap().as_mut().unwrap().remove(&first_key).unwrap();
        assert!(std::sync::Arc::ptr_eq(&first, &removed));
        let replacement = device_retire_state(first_key);
        assert!(!std::sync::Arc::ptr_eq(&first, &replacement), "a removed device key gets a fresh state while old work retains its lease");

        drop(first_guard);
        drop(removed);
        let mut registry = RETIRE.lock().unwrap();
        registry.as_mut().unwrap().remove(&first_key);
        registry.as_mut().unwrap().remove(&second_key);
    }

    #[test]
    fn forgetting_queue_cache_preserves_inflight_batch_marks() {
        use std::sync::{atomic::AtomicU64, Arc};
        let mut rq = RetireQueue::<u32, Arc<AtomicU64>>::new();
        assert!(rq
            .retire_with(vec![101], None, &[(7, 7usize)], || Some(Arc::new(AtomicU64::new(0))), |_, _, _| true)
            .is_empty());
        let cached = rq.fences[&7].0.clone();
        let gone = rq.forget_queue(7).expect("cached queue fence");
        assert!(Arc::ptr_eq(&gone, &cached) && !rq.fences.contains_key(&7));
        assert_eq!(rq.batches.len(), 1);
        assert!(Arc::strong_count(&cached) >= 3, "the in-flight batch lost its own fence clone");
    }

    #[test]
    fn gpu_retire_queue_holds_heap_until_blocked_queue_passes_mark() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
        struct GuardedHeap(#[allow(dead_code)] ID3D12Heap, Arc<AtomicBool>);
        impl Drop for GuardedHeap { fn drop(&mut self) { self.1.store(true, Ordering::SeqCst); } }
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let q1: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let q2: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let probe: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = D3D12_HEAP_DESC { SizeInBytes: PAGE as u64, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: PAGE as u64, Flags: D3D12_HEAP_FLAG_NONE };
            let mut heap: Option<ID3D12Heap> = None;
            dev.CreateHeap(&desc, &mut heap).unwrap();
            let dropped = Arc::new(AtomicBool::new(false));
            q1.Wait(&gate, 1).unwrap();
            let (k1, k2) = (q1.as_raw() as usize, q2.as_raw() as usize);
            let mut rq: RetireQueue<GuardedHeap, ID3D12Fence> = RetireQueue::new();
            assert!(rq.retire_with(vec![GuardedHeap(heap.unwrap(), dropped.clone())], None, &[(k1, q1.clone()), (k2, q2.clone())], || dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok(), |q, f, v| q.Signal(f, v).is_ok()).is_empty());
            q2.Signal(&probe, 1).unwrap();
            probe.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let (done, pruned) = rq.sweep_with(&[k1, k2].into_iter().collect(), |f| f.GetCompletedValue());
            assert!(done.is_empty() && pruned.is_empty() && !dropped.load(Ordering::SeqCst), "heap dropped while Q1 was still blocked");
            gate.Signal(1).unwrap();
            q1.Signal(&probe, 2).unwrap();
            probe.SetEventOnCompletion(2, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let (done, pruned) = rq.sweep_with(&[k1].into_iter().collect(), |f| f.GetCompletedValue());
            assert_eq!((done.len(), pruned.len(), dropped.load(Ordering::SeqCst)), (1, 1, false));
            drop((done, pruned));
            assert!(dropped.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn superseded_texture_waits_for_every_title_queue() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::core::Interface;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let first: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let second: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let polling: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            super::super::present::note_execute(first.as_raw(), &[]);
            super::super::present::note_execute(second.as_raw(), &[]);
            let gate_first: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let gate_second: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            first.Wait(&gate_first, 1).unwrap();
            second.Wait(&gate_second, 1).unwrap();

            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 512,
                Height: 512,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let mut raw = std::ptr::null_mut();
            let hr = place_pages(dev.as_raw(), 0x7_1000_0000, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), None, riid, &mut raw);
            if hr.is_none_or(|hr| hr < 0) || raw.is_null() {
                return crate::gpu::recomp::runner::skip_test("reserved texture unavailable");
            }
            let resource = ID3D12Resource::from_raw(raw);
            let dev_key = dev.as_raw() as usize;
            let bytes = page_bytes_for(dev_key).0;
            assert!(bytes >= 16 * PAGE);

            unmap_superseded(&resource);
            before_execute(&polling);
            assert_eq!(held_unmaps_for(dev_key), 1);
            assert_eq!(page_bytes_for(dev_key).0, bytes, "blocked queues must keep the physical tiles live");

            gate_first.Signal(1).unwrap();
            let first_done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            first.Signal(&first_done, 1).unwrap();
            first_done.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            before_execute(&polling);
            assert_eq!(held_unmaps_for(dev_key), 1, "the second queue still owns prior work");

            gate_second.Signal(1).unwrap();
            let second_done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            second.Signal(&second_done, 1).unwrap();
            second_done.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            before_execute(&polling);
            assert_eq!(held_unmaps_for(dev_key), 0);
            assert!(page_bytes_for(dev_key).0 < bytes, "tiles return only after every queue completes");

            let mut remap_raw = std::ptr::null_mut();
            let remap_hr = place_pages(dev.as_raw(), 0x7_2000_0000, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), None, riid, &mut remap_raw);
            assert!(remap_hr.is_some_and(|hr| hr >= 0) && !remap_raw.is_null());
            let remapped = ID3D12Resource::from_raw(remap_raw);
            let remap_bytes = page_bytes_for(dev_key).0;
            first.Wait(&gate_first, 2).unwrap();
            second.Wait(&gate_second, 2).unwrap();
            unmap_superseded(&remapped);
            assert_eq!(held_unmaps_for(dev_key), 1);
            assert!(remap_if_unmapped(&remapped));
            assert_eq!(held_unmaps_for(dev_key), 0, "remapping cancels the deferred unmap");
            assert_eq!(page_bytes_for(dev_key).0, remap_bytes, "cancellation preserves the original tiles");
            gate_first.Signal(2).unwrap();
            gate_second.Signal(2).unwrap();
            drop(remapped);
            drop(resource);
        }
    }

    #[test]
    fn canonical_va_aliases_unmapped_gpu_view_but_preserves_committed_override() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGES_64K: u32 = 0x2040_0000;
        const GRAPHICS_NOACCESS: u32 = 0x800;
        unsafe {
            let cpu = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            let gpu_unmapped = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(cpu != 0 && gpu_unmapped != 0);
            crate::hle::kernelx::mirrors::reserved(cpu, PAGE, PAGES_64K, 0x04);
            crate::hle::kernelx::mirrors::reserved(gpu_unmapped, PAGE, PAGES_64K, GRAPHICS_NOACCESS | 0x01);

            // Unmapped GPU view aliases to its CPU view, and reverse lookup finds the GPU view.
            assert_eq!(canonical_va(gpu_unmapped), cpu);
            assert!(crate::hle::kernelx::mirrors::gpu_views_of(cpu).contains(&gpu_unmapped));

            // Once distinct committed memory is mapped directly at the GPU view, canonical_va must
            // NOT alias it back to the CPU view.
            assert!(!VirtualAlloc(gpu_unmapped as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert_eq!(canonical_va(gpu_unmapped), gpu_unmapped);

            crate::hle::kernelx::mirrors::released(gpu_unmapped);
            crate::hle::kernelx::mirrors::released(cpu);
            VirtualFree(gpu_unmapped as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(cpu as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// Idle pages go back after `IDLE`, and a sweep looks at the queue of idle pages alone: one
    /// used again stays, one idle again later goes when that is `IDLE` ago, and the sweeps of
    /// 20 000 placements beside 200 000 pages in use take no time to speak of (a look at every
    /// page at each of them: 4 000 000 000 looks).
    #[test]
    fn idle_pages_are_swept_from_their_queue() {
        use std::collections::{HashMap, VecDeque};
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut idle: VecDeque<(PageKey, Instant)> = VecDeque::new();
        let mut pages: HashMap<usize, Page> = (0..200_000usize).map(|i| (i * PAGE, Page { tile: (0, i as u32), users: 1, idle_since: None })).collect();
        let mut extra: HashMap<(usize, usize), Page> = HashMap::from([((7 * PAGE, 3), Page { tile: (1, 0), users: 1, idle_since: None })]);
        let mut cooling: VecDeque<(Tile, Instant)> = VecDeque::new();
        // what `release_tiles` and `bind` do to a page
        fn release(p: &mut Page, idle: &mut VecDeque<(PageKey, Instant)>, key: PageKey, when: Instant) {
            (p.users, p.idle_since) = (0, Some(when));
            idle.push_back((key, when));
        }
        fn use_again(p: &mut Page) {
            (p.users, p.idle_since) = (1, None);
        }
        // pages 1, 2, 3 idle at 0 ms and an extra page at 100 ms; page 2 is used again at
        // 500 ms; page 3 too, and is idle again at 1500 ms
        for a in [PAGE, 2 * PAGE, 3 * PAGE] {
            release(pages.get_mut(&a).unwrap(), &mut idle, PageKey::Own(a), at(0));
        }
        release(extra.get_mut(&(7 * PAGE, 3)).unwrap(), &mut idle, PageKey::Extra(7 * PAGE, 3), at(100));
        use_again(pages.get_mut(&(2 * PAGE)).unwrap());
        use_again(pages.get_mut(&(3 * PAGE)).unwrap());
        release(pages.get_mut(&(3 * PAGE)).unwrap(), &mut idle, PageKey::Own(3 * PAGE), at(1500));

        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(1900));
        assert!(cooling.is_empty() && pages.len() == 200_000 && idle.len() == 5, "nothing is idle for IDLE yet");
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2050));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1)], "page 1 alone: 2 is in use, 3 idle since 1500 ms");
        assert!(!pages.contains_key(&PAGE) && pages.contains_key(&(2 * PAGE)) && pages.contains_key(&(3 * PAGE)) && extra.len() == 1);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2150));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0)], "the extra page, idle since 100 ms");
        assert!(extra.is_empty() && pages.len() == 199_999);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3499));
        assert!(pages.contains_key(&(3 * PAGE)));
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3500));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0), (0, 3)]);
        assert!(idle.is_empty() && pages.len() == 199_998 && pages[&(2 * PAGE)].users == 1);

        // every placement sweeps: with nothing idle that is nothing, whatever is in use
        let t = Instant::now();
        for k in 0..20_000u64 {
            sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(4000 + k));
        }
        assert!(t.elapsed() < Duration::from_secs(2), "{:?} for 20 000 sweeps", t.elapsed());
        assert_eq!(pages.len(), 199_998);
    }

    /// Any 64 KiB page boundary inside `[va, va + bytes)` belongs to that range alone, so
    /// non-overlapping ranges never share an owned page even when only 4 KiB aligned.
    #[test]
    fn owned_pages_never_alias_disjoint_ranges() {
        assert_eq!(owned_pages(0x1_0000, Some(0x1_0000), 4), Some((0x1_0000, 1)));
        assert_eq!(owned_pages(0x1_0000, Some(0x1_4000), 4), Some((0x1_0000, 2)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4000), 4), Some((0x2_0000, 0)));
        assert_eq!(owned_pages(0x1_4000, Some(0x1_0000), 4), Some((0x2_0000, 1)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4_0000), 4), Some((0x2_0000, 4)));
        assert_eq!(owned_pages(0x1_4000, None, 4), None);
        // Exhaustively check 4 KiB-aligned disjoint pairs over 256 KiB.
        const K4: usize = 0x1000;
        for a in (0..16).map(|i| 0x10_0000 + i * K4) {
            for sa in [K4, 4 * K4, 16 * K4, 20 * K4] {
                let (fa, na) = owned_pages(a, Some(sa as u64), 8).unwrap();
                let pa: Vec<usize> = (0..na).map(|i| fa + i * PAGE).collect();
                for b in (0..16).map(|i| 0x10_0000 + i * K4) {
                    for sb in [K4, 4 * K4, 16 * K4, 20 * K4] {
                        if a + sa <= b || b + sb <= a {
                            let (fb, nb) = owned_pages(b, Some(sb as u64), 8).unwrap();
                            for j in 0..nb {
                                assert!(!pa.contains(&(fb + j * PAGE)), "disjoint [{a:#x}..{:#x}) and [{b:#x}..{:#x}) shared page", a + sa, b + sb);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Reserved only with TILED support, and for MSAA a tiled sample count (#940: the driver
    /// refused R32G8X24_TYPELESS reserved, at 1 and 4 samples, and the debug layer said so).
    #[test]
    fn reserved_needs_tiled_format_and_tiled_msaa() {
        let tiled = D3D12_FORMAT_SUPPORT2_TILED.0;
        assert!(can_reserve(tiled, 1, 0));
        assert!(!can_reserve(0, 1, 0), "format without TILED");
        assert!(can_reserve(tiled, 4, 1));
        assert!(!can_reserve(tiled, 4, 0), "no tiled quality level at 4 samples");
        assert!(!can_reserve(0, 4, 1));
    }

    /// The device's answer agrees with CreateReservedResource: what `reservable` turns away the
    /// driver refuses, what it lets through the driver makes (WARP or a hardware adapter).
    #[test]
    fn reservable_agrees_with_the_driver() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: windows::Win32::Graphics::Dxgi::IDXGIFactory4 = windows::Win32::Graphics::Dxgi::CreateDXGIFactory1().expect("DXGI factory");
                let Ok(warp) = factory.EnumWarpAdapter::<windows::Win32::Graphics::Dxgi::IDXGIAdapter>() else { return };
                let _ = D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev);
            }
            let Some(dev) = dev else { return };
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_err() || o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
                return;
            }
            for (format, samples) in [(DXGI_FORMAT_R32G8X24_TYPELESS, 1), (DXGI_FORMAT_R32G8X24_TYPELESS, 4), (DXGI_FORMAT_R32_TYPELESS, 1), (DXGI_FORMAT_R8G8B8A8_UNORM, 1)] {
                let flags = if format == DXGI_FORMAT_R8G8B8A8_UNORM { D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET } else { D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 128, Height: 128, DepthOrArraySize: 1, MipLevels: 1, Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, Flags: flags, ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                let made = dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).is_ok() && r.is_some();
                assert_eq!(reservable(&dev, &desc), made, "format {} x{samples}: reservable vs CreateReservedResource", format.0);
            }
        }
    }

    /// A placement's arena id, with heaps numbered as they are made.
    fn take(a: &mut Arenas<u32>, va: usize, bytes: usize, align: usize, now: Instant) -> u64 {
        let n = a.list.len() as u32;
        a.take_for(1, va, bytes, align, now, |_, _| Some(n)).0.unwrap().2
    }

    #[test]
    fn an_arena_is_shared_at_its_addresses_and_otherwise_its_placements_size() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let first = take(&mut a, a0, 70 * MIB, PAGE, t);
        assert_eq!(a.bytes(), 70 * MIB, "the placement's size, not 256 MiB");
        // the same address, and a smaller placement inside: the same memory
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t), first);
        assert_eq!(take(&mut a, a0 + 4 * MIB, 8 * MIB, PAGE, t), first);
        // one straddling its end gets an arena of its own size; so does an MSAA target that needs 4 MiB
        let straddle = take(&mut a, a0 + 64 * MIB, 70 * MIB, PAGE, t);
        let msaa = take(&mut a, a0 + 8 * MIB, 8 * MIB, 4 * MIB, t);
        assert!(straddle != first && msaa != first && msaa != straddle);
        assert_eq!(a.bytes(), (70 + 70 + 8) * MIB);
        // the newest arena holding an address is the one placements there share
        assert_eq!(take(&mut a, a0 + 64 * MIB, 4 * MIB, PAGE, t), straddle);
    }

    #[test]
    fn an_arena_no_resource_uses_goes_after_two_seconds() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let id = take(&mut a, a0, 70 * MIB, PAGE, t);
        a.release(id, t);
        // placed again within 2 s: the same memory (with what it holds)
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t + Duration::from_secs(1)), id);
        a.release(id, t + Duration::from_secs(1));
        take(&mut a, a0 + 512 * MIB, MIB, PAGE, t + Duration::from_secs(3) + Duration::from_millis(1));
        assert_eq!(a.bytes(), MIB, "the idle arena went");
    }

    /// The retail game's pattern (HANDOFF): targets packed back to back over 3.5 GB of its
    /// addresses and placed again and again, each round at the last round's addresses or a third
    /// of a target further on, so placements straddle earlier ones. Main's 256 MiB arenas merged
    /// straddled arenas into growing unions while the old ones lived on (15 GB in 2 s).
    #[test]
    fn streamed_targets_stay_bounded() {
        const S: usize = 64 * MIB; // a 3840x2160 D32S8, rounded
        const K: usize = 8; // targets a round
        const SPAN: usize = 3584 * MIB;
        let (t0, a0) = (Instant::now(), 0x1_0000_0000usize);
        let mut a = Arenas::new();
        let mut live: std::collections::VecDeque<Vec<u64>> = std::collections::VecDeque::new();
        let mut bytes = Vec::new();
        for r in 0..400usize {
            let now = t0 + Duration::from_millis(500 * r as u64);
            let start = ((r / 2) * S / 3) % (SPAN - K * S) & !(PAGE - 1);
            live.push_back((0..K).map(|k| take(&mut a, a0 + start + k * S, S, PAGE, now)).collect());
            // a target lives three rounds (1.5 s)
            if live.len() > 3 {
                for id in live.pop_front().unwrap() {
                    a.release(id, now);
                }
            }
            bytes.push(a.bytes());
        }
        // live: 3 rounds of targets; idle and kept (< 2 s): 4 more
        let most = *bytes.iter().max().unwrap();
        assert!(most <= 7 * K * S, "{most} bytes of arenas");
        let early = *bytes[20..120].iter().max().unwrap();
        let late = *bytes[300..].iter().max().unwrap();
        assert!(late <= early, "arenas grow over time: {early} then {late} bytes");
    }

    /// On the GPU: D32S8 targets 256 MiB apart take their own size of video memory, not an arena
    /// of 256 MiB each; depth and stencil survive a clear, CopyResource and a read of both planes;
    /// arenas no target uses go. Needs a hardware adapter (skips on WARP).
    #[test]
    fn d32s8_targets_take_their_own_size_of_video_memory() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R32G8X24_TYPELESS, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default() };
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC| {
                let mut p = std::ptr::null_mut();
                let hr = place(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32, &clear as *const _ as *const c_void, None, riid, &mut p);
                assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "D32S8 target at {va:#x}");
                ID3D12Resource::from_raw(p)
            };

            // 14 4K targets over 3.5 GB of addresses
            let big = tex(3840, 2160);
            let size = dev.GetResourceAllocationInfo(0, &[big]).SizeInBytes as usize;
            let (used0, arenas0) = (local(), arena_bytes().0);
            let targets: Vec<ID3D12Resource> = (0..14).map(|k| place_at(0x40_0000_0000 + k * 256 * MIB, &big)).collect();
            let (used, arenas) = (local().saturating_sub(used0) as usize, arena_bytes().0 - arenas0);
            println!("14 D32S8 3840x2160 targets ({size} bytes each): {arenas} bytes of arenas, {used} bytes more local video memory");
            assert!(arenas <= 14 * size.next_multiple_of(PAGE), "{arenas} bytes of arenas");
            assert!(used < 14 * 256 * MIB, "{used} bytes of video memory");
            drop(targets);

            // clear one, copy it to another, read both planes back
            let small = tex(64, 64);
            let (a, b) = (place_at(0x48_0000_0000, &small), place_at(0x48_1000_0000, &small));
            let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&a, Some(&view), dsv);
            let mut layouts = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 2, 0, Some(layouts.as_mut_ptr()), None, None, Some(&mut total));
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25, 0x5A, None);
            list.ResourceBarrier(&[barrier(&a, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE), barrier(&b, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyResource(&b, &a);
            list.ResourceBarrier(&[barrier(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (plane, layout) in layouts.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *layout } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: plane as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            // no event: returns when the fence has the value
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let bytes = std::slice::from_raw_parts(p as *const u8, total as usize);
            let row = |plane: usize, y: usize| layouts[plane].Offset as usize + y * layouts[plane].Footprint.RowPitch as usize;
            for y in [0, 31, 63] {
                let depth = f32::from_le_bytes(bytes[row(0, y)..row(0, y) + 4].try_into().unwrap());
                assert_eq!((depth, bytes[row(1, y)]), (0.25, 0x5A), "row {y}: depth, stencil");
            }
            readback.Unmap(0, None);
            drop((a, b));

            // 2 s later the arenas no target uses are let go: one new target's is all that is left
            std::thread::sleep(IDLE + Duration::from_millis(100));
            let last = place_at(0x50_0000_0000, &small);
            let small_size = dev.GetResourceAllocationInfo(0, &[small]).SizeInBytes as usize;
            assert_eq!(arena_bytes().0 - arenas0, small_size.next_multiple_of(PAGE));
            drop(last);
        }
    }

    /// Reproduces the multi-pool streaming churn pattern on the active D3D12 driver (comparing
    /// scattered cross-heap `UpdateTileMappings` against single-heap contiguous
    /// `UpdateTileMappings`, both with `FLAG_NONE` and per-texture `Signal`):
    /// `cargo test -p xcl --lib -- --ignored --exact hle::d3d12x::vram::tests::churn_update_tile_mappings_timing --nocapture`
    #[test]
    #[ignore]
    fn churn_update_tile_mappings_timing() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("D3D12 device");
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            const HEAPS: usize = 64;
            const TILES_PER_TEX: usize = 32; // 2 MiB tiled texture
            const TEX_COUNT: usize = 256;
            let heap_desc = D3D12_HEAP_DESC {
                SizeInBytes: 64 * PAGE as u64,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_NONE,
            };
            let mut heaps = Vec::with_capacity(HEAPS);
            for _ in 0..HEAPS {
                let mut h: Option<ID3D12Heap> = None;
                dev.CreateHeap(&heap_desc, &mut h).unwrap();
                heaps.push(h.unwrap());
            }
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 1024,
                Height: 512,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let make_textures = || -> Vec<ID3D12Resource> {
                (0..TEX_COUNT)
                    .map(|_| {
                        let mut r: Option<ID3D12Resource> = None;
                        dev.CreateReservedResource(&tex_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                        r.unwrap()
                    })
                    .collect()
            };

            // BEFORE pattern: flat-FIFO cooling scatters each 32-tile texture across 32 heaps,
            // calling UpdateTileMappings(FLAG_NONE) per heap + Signal per texture.
            let old_res = make_textures();
            let mut fval = 0u64;
            let t0 = Instant::now();
            let mut old_calls = 0usize;
            for (k, r) in old_res.iter().enumerate() {
                let tiles: Vec<Tile> = (0..TILES_PER_TEX).map(|i| (((k + i) % HEAPS) as u32, (i % 64) as u32)).collect();
                let mut hs: Vec<u32> = tiles.iter().map(|t| t.0).collect();
                hs.sort_unstable();
                hs.dedup();
                for h in hs {
                    let ranges = heap_ranges(h, &tiles);
                    let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|x| if x.0 { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP }).collect();
                    let offsets: Vec<u32> = ranges.iter().map(|x| x.1).collect();
                    let counts: Vec<u32> = ranges.iter().map(|x| x.2).collect();
                    let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                    let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles.len() as u32, UseBox: false.into(), ..Default::default() };
                    queue.UpdateTileMappings(r, 1, Some(&coord), Some(&region), &heaps[h as usize], ranges.len() as u32, Some(flags.as_ptr()), Some(offsets.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
                    old_calls += 1;
                }
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
            }
            let old_submit = t0.elapsed();
            fence.SetEventOnCompletion(fval, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let old_total = t0.elapsed();

            // AFTER pattern: per-heap contiguous TilePool -> 1 heap, 1 contiguous range,
            // D3D12_TILE_MAPPING_FLAG_NONE, per-texture Signal.
            let new_res = make_textures();
            let t1 = Instant::now();
            let mut new_calls = 0usize;
            for (k, r) in new_res.iter().enumerate() {
                let h = k % HEAPS;
                let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                let region = D3D12_TILE_REGION_SIZE { NumTiles: TILES_PER_TEX as u32, UseBox: false.into(), ..Default::default() };
                let flag = D3D12_TILE_RANGE_FLAG_NONE;
                let off = 0u32;
                let cnt = TILES_PER_TEX as u32;
                queue.UpdateTileMappings(r, 1, Some(&coord), Some(&region), &heaps[h], 1, Some(&flag), Some(&off), Some(&cnt), D3D12_TILE_MAPPING_FLAG_NONE);
                new_calls += 1;
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
            }
            let new_submit = t1.elapsed();
            fence.SetEventOnCompletion(fval, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let new_total = t1.elapsed();

            println!(
                "BEFORE ({TEX_COUNT} x 2MiB textures, {old_calls} UpdateTileMappings + {TEX_COUNT} Signals): submit={old_submit:?} gpu_done={old_total:?}"
            );
            println!(
                "AFTER  ({TEX_COUNT} x 2MiB textures, {new_calls} UpdateTileMappings + {TEX_COUNT} Signals): submit={new_submit:?} gpu_done={new_total:?}"
            );
            assert!(new_calls < old_calls);
        }
    }

    #[test]
    fn mip_texels_and_range_for_2d_array_3d_and_1d() {
        assert_eq!(mip_texels(64, 32, 1, D3D12_RESOURCE_DIMENSION_TEXTURE2D.0, 0), 64 * 32);
        assert_eq!(mip_texels(64, 32, 1, D3D12_RESOURCE_DIMENSION_TEXTURE2D.0, 1), 32 * 16);
        assert_eq!(mip_texels(64, 32, 6, D3D12_RESOURCE_DIMENSION_TEXTURE2D.0, 2), 16 * 8 * 6);
        assert_eq!(mip_texels(64, 32, 8, D3D12_RESOURCE_DIMENSION_TEXTURE3D.0, 2), 16 * 8 * 2);
        assert_eq!(mip_texels(64, 1, 4, D3D12_RESOURCE_DIMENSION_TEXTURE1D.0, 3), 8 * 4);
        assert_eq!(range_texels(4, 4, 1, D3D12_RESOURCE_DIMENSION_TEXTURE2D.0, 0..3), 16 + 4 + 1);
    }

    #[test]
    fn srv_mip_range_respects_most_detailed_mip_levels_and_min_lod_clamp() {
        assert_eq!(srv_mip_range(5, None), 0..5);
        let srv_2d = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 1,
                    MipLevels: 3,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 2.75,
                },
            },
            ..Default::default()
        };
        assert_eq!(srv_mip_range(6, Some(&srv_2d)), 2..4);

        let srv_all = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 2,
                    MipLevels: u32::MAX,
                    FirstArraySlice: 0,
                    ArraySize: 2,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
            ..Default::default()
        };
        assert_eq!(srv_mip_range(5, Some(&srv_all)), 2..5);
    }

    #[test]
    fn texel_meta_unions_overlapping_srv_ranges_without_double_counting() {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 8,
            Height: 8,
            DepthOrArraySize: 1,
            MipLevels: 4,
            ..Default::default()
        };
        let mut tm = TexelMeta::from_desc(&desc, false);
        assert_eq!(tm.total_texels, 64 + 16 + 4 + 1);
        assert_eq!(tm.resident_texels, 0);

        let srv_low = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 1, MipLevels: 2, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
            },
            ..Default::default()
        };
        assert_eq!(tm.record_srv(Some(&srv_low)), (true, 16 + 4));

        let srv_all = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 4, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
            },
            ..Default::default()
        };
        assert_eq!(tm.record_srv(Some(&srv_all)), (false, 64 + 1));
        assert_eq!(tm.record_srv(Some(&srv_all)), (false, 0));
        assert_eq!(tm.srv_sampled_texels, tm.total_texels);
    }

    #[test]
    fn texel_meta_unions_array_slices_without_counting_the_whole_mip() {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 6,
            MipLevels: 1,
            ..Default::default()
        };
        let mut tm = TexelMeta::from_desc(&desc, false);
        let view = |first| D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    FirstArraySlice: first,
                    ArraySize: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
            ..Default::default()
        };
        assert_eq!(tm.record_srv(Some(&view(0))), (true, 4096));
        assert_eq!(tm.record_srv(Some(&view(0))), (false, 0));
        assert_eq!(tm.record_srv(Some(&view(1))), (false, 4096));
        assert_eq!(tm.srv_sampled_texels, 8192);
        assert_eq!(tm.total_texels, 6 * 4096);

        let ms = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DMSArray: D3D12_TEX2DMS_ARRAY_SRV { FirstArraySlice: 4, ArraySize: 1 },
            },
            ..Default::default()
        };
        assert_eq!(tm.record_srv(Some(&ms)), (false, 4096));
        assert_eq!(tm.srv_sampled_texels, 3 * 4096);
    }

    #[test]
    fn clamp_srv_min_lod_raises_clamp_without_lowering_existing() {
        let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: u32::MAX,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.5,
                },
            },
            ..Default::default()
        };
        clamp_srv_min_lod(&mut srv, 2);
        unsafe { assert_eq!(srv.Anonymous.Texture2D.ResourceMinLODClamp, 2.0) };
        clamp_srv_min_lod(&mut srv, 1);
        unsafe { assert_eq!(srv.Anonymous.Texture2D.ResourceMinLODClamp, 2.0) };
    }

    #[test]
    fn heap_draining_eviction_empties_coldest_heaps_first_and_skips_pinned() {
        let t0 = Instant::now();
        let cands = vec![
            HeapEvictCandidate { res_key: 1, heap: 0, heap_evictable: false, heap_free: 900, last_touch: t0 + Duration::from_secs(1), tiles: 64 },
            HeapEvictCandidate { res_key: 10, heap: 3, heap_evictable: true, heap_free: 0, last_touch: t0 + Duration::from_secs(20), tiles: 64 },
            HeapEvictCandidate { res_key: 20, heap: 2, heap_evictable: true, heap_free: 0, last_touch: t0 + Duration::from_secs(5), tiles: 64 },
            HeapEvictCandidate { res_key: 21, heap: 2, heap_evictable: true, heap_free: 0, last_touch: t0 + Duration::from_secs(2), tiles: 64 },
            HeapEvictCandidate { res_key: 30, heap: 1, heap_evictable: true, heap_free: 400, last_touch: t0 + Duration::from_secs(10), tiles: 64 },
            HeapEvictCandidate { res_key: 31, heap: 1, heap_evictable: true, heap_free: 400, last_touch: t0 + Duration::from_secs(8), tiles: 64 },
        ];
        assert_eq!(pick_evict_candidates_by_heap(&cands, 100), vec![31, 30]);
        assert_eq!(pick_evict_candidates_by_heap(&cands, 150), vec![31, 30, 21, 20]);
    }

    #[test]
    fn gpu_unmap_top_mip_with_srv_min_lod_clamp_samples_resident_mip() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if crate::gpu::recomp::dxc::location().is_err() {
            crate::gpu::recomp::runner::skip_test("no DXC");
            return;
        }
        use std::mem::{transmute_copy, ManuallyDrop};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no D3D12 device");
                return;
            }
            let dev = dev.unwrap();
            if !tier2(&dev) {
                crate::gpu::recomp::runner::skip_test("no tiled tier 2");
                return;
            }

            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 512,
                Height: 512,
                DepthOrArraySize: 1,
                MipLevels: 2,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let mut tex: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut tex).unwrap();
            let tex = tex.unwrap();
            let (lv, total) = level_tiling(&dev, &tex, 2);
            let mut lv = lv.expect("512x512 RGBA8 with 2 mips has separate tile groups");
            assert!(lv.groups.len() >= 2 && total >= 2);

            let mut pages = Pages::new(&dev).unwrap();
            let va = 0x10_0000_0000usize;
            let last_group = lv.groups.len().saturating_sub(1);
            for k in 0..lv.groups.len() {
                let (sub, first, n, _) = lv.groups[k];
                let (u, ue) = pages.bind_range(&dev, &tex, va, None, total, sub, first as usize..(first + n) as usize, k < last_group).unwrap();
                lv.groups[k].3 = true;
                lv.group_tiles[k] = (u, ue);
            }

            let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let mut rows = [0u32; 2];
            let mut row_bytes = [0u64; 2];
            let mut upload_bytes = 0u64;
            dev.GetCopyableFootprints(&desc, 0, 2, 0, Some(layouts.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(row_bytes.as_mut_ptr()), Some(&mut upload_bytes));

            let buffer = |size: u64, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, uav: bool| {
                let d = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
                    ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, state, None, &mut r).unwrap();
                r.unwrap()
            };

            let upload = buffer(upload_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
            let mut p = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut p)).unwrap();
            let mip_colors = [0xFF00_00FFu32, 0xFF00_FF00u32]; // mip 0 = red, mip 1 = green
            for (m, l) in layouts.iter().enumerate() {
                for r in 0..rows[m] as usize {
                    let at = (p as *mut u8).add(l.Offset as usize + r * l.Footprint.RowPitch as usize) as *mut u32;
                    for x in 0..l.Footprint.Width as usize {
                        at.add(x).write_unaligned(mip_colors[m]);
                    }
                }
            }
            upload.Unmap(0, None);

            let src = r#"
Texture2D<float4> tex0 : register(t0);
SamplerState samp0 : register(s0);
RWByteAddressBuffer out0 : register(u0);
[RootSignature("DescriptorTable(SRV(t0, numDescriptors=1)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_POINT), UAV(u0)")]
[numthreads(1, 1, 1)]
void main() {
    float4 c = tex0.SampleLevel(samp0, float2(0.5, 0.5), 0.0);
    uint r = (uint)round(saturate(c.r) * 255.0);
    uint g = (uint)round(saturate(c.g) * 255.0);
    uint b = (uint)round(saturate(c.b) * 255.0);
    uint a = (uint)round(saturate(c.a) * 255.0);
    out0.Store(0, r | (g << 8) | (b << 16) | (a << 24));
}
"#;
            let dxil = crate::gpu::recomp::dxc::compile(src, "main", "cs_6_0", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let pso: ID3D12PipelineState = dev
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: transmute_copy(&rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                    ..Default::default()
                })
                .unwrap();

            let dh: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let slot_cpu = dh.GetCPUDescriptorHandleForHeapStart();
            let slot_gpu = dh.GetGPUDescriptorHandleForHeapStart();
            let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 2, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
                },
            };
            super::super::real::native_create_srv(&dev, &tex, &srv, slot_cpu.ptr as u64);

            let out = buffer(16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
            let rb = buffer(16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);

            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&pso)).unwrap();
            let barrier = |r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: transmute_copy(r),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: before,
                        StateAfter: after,
                    }),
                },
            };
            for (m, l) in layouts.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: transmute_copy(&tex),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: m as u32 },
                };
                let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: transmute_copy(&upload),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *l },
                };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src_loc, None);
            }
            list.ResourceBarrier(&[barrier(&tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
            list.SetDescriptorHeaps(&[Some(dh.clone())]);
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootDescriptorTable(0, slot_gpu);
            list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            list.ResourceBarrier(&[barrier(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyBufferRegion(&rb, 0, &out, 0, 16);
            list.Close().unwrap();

            queue.Wait(&pages.fence, pages.value).unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            queue.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, &dev, None, 20_000).map_err(|e| format!("{e}")).unwrap();

            let mut q = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut q)).unwrap();
            let got_before = (q as *const u32).read_unaligned();
            rb.Unmap(0, None);
            assert_eq!(got_before, mip_colors[0], "with mip 0 resident and unclamped, sampling LOD 0 reads red");

            let (tiles_evict, tail_mip) = lv.evictable_chain().unwrap();
            assert_eq!(tail_mip, 1);
            assert!(tiles_evict > 0);
            clamp_srv_min_lod(&mut srv, tail_mip);
            super::super::real::native_create_srv(&dev, &tex, &srv, slot_cpu.ptr as u64);
            let (sub0, _, tiles0, _) = lv.groups[0];
            let (u0, ue0) = std::mem::take(&mut lv.group_tiles[0]);
            lv.groups[0].3 = false;
            pages.unmap_range(&tex, sub0, tiles0);
            pages.release_tiles(u0, ue0, Instant::now(), false);

            alloc.Reset().unwrap();
            list.Reset(&alloc, Some(&pso)).unwrap();
            list.ResourceBarrier(&[barrier(&out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            list.SetDescriptorHeaps(&[Some(dh.clone())]);
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootDescriptorTable(0, slot_gpu);
            list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            list.ResourceBarrier(&[barrier(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyBufferRegion(&rb, 0, &out, 0, 16);
            list.Close().unwrap();

            queue.Wait(&pages.fence, pages.value).unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            queue.Signal(&fence, 2).unwrap();
            crate::gpu::fence::wait(&fence, 2, &dev, None, 20_000).map_err(|e| format!("{e}")).unwrap();

            let mut q2 = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut q2)).unwrap();
            let got_after = (q2 as *const u32).read_unaligned();
            rb.Unmap(0, None);
            assert_eq!(got_after, mip_colors[1], "with mip 0 unmapped and SRV clamped to mip 1, sampling LOD 0 reads green");

            for (u, ue) in lv.group_tiles {
                if !u.is_empty() || !ue.is_empty() {
                    pages.release_tiles(u, ue, Instant::now(), false);
                }
            }
        }
    }
}
