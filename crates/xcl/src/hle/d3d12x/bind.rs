//! Root arguments of console code, bound when it dispatches with a translated pipeline.
//!
//! A translated console shader reads its root arguments the console's way (user SGPRs s4…,
//! the rest from a table at s[0:1]: `console_list::user_data`) and memory through XCL's
//! shader ABI (u0 io, u1 res, u2… unified memory). So on a command list of console code the
//! compute root signature and root arguments are kept here – packed by the root signature's
//! layout – instead of going to the PC list, and each Dispatch with a translated pipeline
//! binds the ABI root signature, writes the user SGPRs (`io`) and the argument table to
//! unified memory and dispatches. That memory stays until the list's allocator is reset.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{self, Layout, Param};

#[derive(Default)]
struct State {
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    /// the graphics root signature's layout and arguments
    layout_gfx: Option<Arc<Layout>>,
    args_gfx: Vec<u8>,
    /// the title's graphics root signature (a PC object), bound before a draw with an
    /// untranslated pipeline – created with it – since SetGraphicsRootSignature binds nothing
    rs_gfx: usize,
    pso: usize,
    /// Retained nonblocking decision for `pso`.
    resolution: Option<super::pso_async::PipelineResolution>,
    allocator: usize,
    /// unified memory of this recording (io, argument tables)
    uma: Vec<usize>,
    /// where the next draw's io and argument table go (blocks in `uma`)
    arena: crate::gpu::uma::Arena,
    /// the console's quad sample mask (#142, `quad_mask`) as the io words of the next draws
    /// (`recomp::IO_QUAD_KILL`: the samples it takes away; [0, 0] = all ones)
    quad_kill: [u32; 2],
}

const STATE_SHARDS: usize = 64;

struct BindEntry {
    live: AtomicBool,
    state: Mutex<State>,
}

impl Default for BindEntry {
    fn default() -> Self {
        Self { live: AtomicBool::new(true), state: Mutex::new(State::default()) }
    }
}

const STATE_CACHE_ENTRIES: usize = 4;

struct BindStateCache {
    entries: [Option<(usize, Weak<BindEntry>)>; STATE_CACHE_ENTRIES],
    next: usize,
}

impl BindStateCache {
    fn get(&mut self, key: usize) -> Option<Arc<BindEntry>> {
        for slot in &mut self.entries {
            let Some((cached_key, cached)) = slot else { continue };
            if *cached_key != key {
                continue;
            }
            let entry = cached.upgrade();
            if entry.as_ref().is_some_and(|entry| entry.live.load(Ordering::Acquire)) {
                return entry;
            }
            *slot = None;
            return None;
        }
        None
    }

    fn insert(&mut self, key: usize, entry: &Arc<BindEntry>) {
        if let Some(slot) = self.entries.iter_mut().find(|slot| slot.as_ref().is_some_and(|(cached, _)| *cached == key)) {
            *slot = Some((key, Arc::downgrade(entry)));
            return;
        }
        if let Some(slot) = self.entries.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some((key, Arc::downgrade(entry)));
            return;
        }
        self.entries[self.next] = Some((key, Arc::downgrade(entry)));
        self.next = (self.next + 1) % STATE_CACHE_ENTRIES;
    }
}

thread_local! {
    static STATE_CACHE: RefCell<BindStateCache> = const { RefCell::new(BindStateCache {
        entries: [const { None }; STATE_CACHE_ENTRIES],
        next: 0,
    }) };
}

struct BindStateRegistry {
    shards: [Mutex<Option<HashMap<usize, Arc<BindEntry>>>>; STATE_SHARDS],
}

impl BindStateRegistry {
    const fn new() -> Self {
        Self { shards: [const { Mutex::new(None) }; STATE_SHARDS] }
    }

    fn get(&self, key: usize) -> Option<Arc<BindEntry>> {
        self.shards[state_shard_index(key)].lock().unwrap().as_ref()?.get(&key).cloned()
    }

    fn get_or_insert(&self, key: usize) -> Arc<BindEntry> {
        self.shards[state_shard_index(key)]
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(key)
            .or_default()
            .clone()
    }

    #[cfg(feature = "pipeline-resolution-bench")]
    fn insert(&self, key: usize, entry: Arc<BindEntry>) -> Option<Arc<BindEntry>> {
        let mut shard = self.shards[state_shard_index(key)].lock().unwrap();
        let replaced = shard.get_or_insert_with(HashMap::new).insert(key, entry);
        if let Some(entry) = &replaced {
            entry.live.store(false, Ordering::Release);
        }
        replaced
    }

    fn remove(&self, key: usize) -> Option<Arc<BindEntry>> {
        let mut shard = self.shards[state_shard_index(key)].lock().unwrap();
        let removed = shard.as_mut()?.remove(&key);
        if let Some(entry) = &removed {
            entry.live.store(false, Ordering::Release);
        }
        removed
    }
}

#[inline]
fn state_shard_index(key: usize) -> usize {
    let mut mixed = (key >> 4).wrapping_mul(0x9E37_79B1);
    mixed ^= mixed >> 16;
    mixed & (STATE_SHARDS - 1)
}

static STATES: BindStateRegistry = BindStateRegistry::new();

fn state_entry(key: usize) -> Arc<BindEntry> {
    if let Some(entry) = STATE_CACHE.with(|cache| cache.borrow_mut().get(key)) {
        return entry;
    }
    let entry = STATES.get_or_insert(key);
    STATE_CACHE.with(|cache| cache.borrow_mut().insert(key, &entry));
    entry
}

/// draws / dispatches of console code: with a translated pipeline, skipped, with a PC one
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// draws and dispatches of console code so far (`XCL_ACTIVITY`)
pub fn activity() -> (u64, u64) {
    (COUNTS[DRAWS].load(Ordering::Relaxed), COUNTS[DISPATCHES].load(Ordering::Relaxed))
}
const DRAWS: usize = 0;
const DRAWS_SKIPPED: usize = 1;
const DRAWS_PC: usize = 2;
const DISPATCHES: usize = 3;
const DISPATCHES_SKIPPED: usize = 4;

fn count(k: usize) {
    let n = COUNTS[k].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 {
        let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let (allocators, blocks) = console_list::pending_blocks();
        let (coherent_used, coherent_total, sparse_used) = crate::gpu::uma::get()
            .map_or((0, 0, 0), |u| (u.used() >> 20, u.size() >> 20, u.sparse_used() >> 20));
        crate::xlog_info!(
            "D3D12.X: console code so far: {} draws with translated pipelines ({} skipped, {} with PC pipelines), {} dispatches ({} skipped); coherent {coherent_used}/{coherent_total} MB, sparse {sparse_used} MB, {blocks} blocks held for {allocators} allocators",
            c[DRAWS], c[DRAWS_SKIPPED], c[DRAWS_PC], c[DISPATCHES], c[DISPATCHES_SKIPPED],
        );
    }
}

fn with<R>(list: *mut c_void, f: impl FnOnce(&mut State) -> R) -> R {
    loop {
        let entry = state_entry(list as usize);
        let mut state = entry.state.lock().unwrap();
        if !entry.live.load(Ordering::Acquire) {
            drop(state);
            continue;
        }
        return f(&mut state);
    }
}

/// Opaque fixture for measuring the actual retained-pipeline state snapshot without a device.
#[cfg(feature = "pipeline-resolution-bench")]
#[doc(hidden)]
pub struct BindStateBench {
    _resolutions: Vec<super::pso_async::PipelineResolutionBench>,
    lists: Vec<usize>,
    _keys: Vec<Box<u8>>,
}

#[cfg(feature = "pipeline-resolution-bench")]
impl BindStateBench {
    pub fn new(workers: usize) -> Self {
        assert!(workers != 0);
        let resolutions = (0..workers)
            .map(|_| super::pso_async::PipelineResolutionBench::new())
            .collect::<Vec<_>>();
        let keys = (0..workers * STATE_SHARDS).map(|index| Box::new(index as u8)).collect::<Vec<_>>();
        let mut used_shards = [false; STATE_SHARDS];
        let lists = keys
            .iter()
            .map(|key| (&**key as *const u8) as usize)
            .filter(|&list| {
                let shard = state_shard_index(list);
                !std::mem::replace(&mut used_shards[shard], true)
            })
            .take(workers)
            .collect::<Vec<_>>();
        assert_eq!(lists.len(), workers, "benchmark keys did not cover enough bind shards");
        for (worker, &list) in lists.iter().enumerate() {
            let entry = Arc::new(BindEntry {
                live: AtomicBool::new(true),
                state: Mutex::new(State {
                    pso: worker + 1,
                    resolution: Some(resolutions[worker].retained_resolution()),
                    ..Default::default()
                }),
            });
            assert!(STATES.insert(list, entry).is_none());
        }
        Self { _resolutions: resolutions, lists, _keys: keys }
    }

    #[inline(never)]
    pub fn snapshot_worker(&self, worker: usize) -> usize {
        let (pso, resolution) = with(self.lists[worker] as *mut c_void, |state| {
            (state.pso, state.resolution.clone())
        });
        std::hint::black_box(resolution);
        pso
    }

    #[inline(never)]
    pub fn snapshot_workers_serial(&self, workers: usize, batch: usize) -> usize {
        let mut checksum = 0usize;
        for worker in 0..workers {
            for _ in 0..batch {
                checksum = checksum.wrapping_add(self.snapshot_worker(worker).rotate_left(worker as u32));
            }
        }
        checksum
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
impl Drop for BindStateBench {
    fn drop(&mut self) {
        let removed = self.lists.iter().filter_map(|&list| STATES.remove(list)).collect::<Vec<_>>();
        drop(removed);
    }
}

fn native_backing(list: *mut c_void, allocator: usize) -> usize {
    if allocator == 0 {
        0
    } else {
        super::alloc_pool::recording_backing(list as usize)
            .unwrap_or_else(|| super::alloc_pool::current(allocator))
    }
}

/// The list was created or reset onto `allocator` with `initial_pso`: the previous
/// recording's memory is the previous native backing's until that backing is reset, and
/// pipeline and root signature state reset per D3D12 `CreateCommandList` / `Reset` semantics.
pub(super) fn reset(list: *mut c_void, allocator: usize, initial_pso: usize) {
    let resolution = super::pso_async::begin_pipeline_resolution(initial_pso);
    install_reset(list, allocator, initial_pso, resolution);
}

fn install_reset(
    list: *mut c_void,
    allocator: usize,
    initial_pso: usize,
    resolution: Option<super::pso_async::PipelineResolution>,
) {
    let backing = native_backing(list, allocator);
    let (old, blocks, old_resolution) = with(list, |s| {
        let old = std::mem::replace(&mut s.allocator, backing);
        s.arena = Default::default();
        s.pso = initial_pso;
        let old_resolution = std::mem::replace(&mut s.resolution, resolution);
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.quad_kill = [0; 2];
        (old, std::mem::take(&mut s.uma), old_resolution)
    });
    console_list::hold_until_reset(old, blocks);
    drop(old_resolution);
}

pub(super) struct PreparedNativePipeline {
    original: usize,
    raw: usize,
    resolution: Option<super::pso_async::PipelineResolution>,
    generation: Option<super::pso_async::NativeGeneration>,
    poison_on_commit: bool,
}

impl PreparedNativePipeline {
    pub(super) fn new(original: usize) -> Self {
        Self::from_resolution(original, super::pso_async::begin_pipeline_resolution(original))
    }

    fn from_resolution(
        original: usize,
        resolution: Option<super::pso_async::PipelineResolution>,
    ) -> Self {
        use super::pso_async::{NativePipelineBindMode, NativePipelineChoice};
        let choice = resolution.as_ref().map(|r| r.select_for_native_bind(NativePipelineBindMode::ForBind));
        let (raw, generation, poison_on_commit) = match choice {
            Some(NativePipelineChoice::Exact { raw, generation }) => (raw, Some(generation), false),
            Some(NativePipelineChoice::InitialFallback { raw, .. }) => (raw, None, false),
            Some(NativePipelineChoice::Unavailable) => {
                (0, None, !resolution.as_ref().unwrap().unavailable_is_emission_guarded())
            }
            None => (original, None, false),
        };
        Self { original, raw, resolution, generation, poison_on_commit }
    }

    pub(super) fn raw(&self) -> usize { self.raw }

    pub(super) fn record_generation(self, list: usize) {
        super::list_x::record_native_generation(list, self.generation);
        if self.poison_on_commit {
            super::list_x::poison_recording(list);
        }
    }

    pub(super) fn commit(self, list: *mut c_void, allocator: usize) {
        super::list_x::record_native_generation(list as usize, self.generation);
        install_reset(list, allocator, self.original, self.resolution);
        if self.poison_on_commit {
            super::list_x::poison_recording(list as usize);
        }
    }
}

/// `ClearState(pso)`: resets pipeline and root-argument state back to creation defaults
/// with `pso` bound, keeping the current recording's allocator and UMA arena intact.
#[cfg(test)]
pub(super) fn clear_state(list: *mut c_void, pso: usize) {
    let resolution = super::pso_async::begin_pipeline_resolution(pso);
    install_clear_state(list, pso, resolution);
}

fn install_clear_state(
    list: *mut c_void,
    pso: usize,
    resolution: Option<super::pso_async::PipelineResolution>,
) {
    let old_resolution = with(list, |s| {
        s.pso = pso;
        let old_resolution = std::mem::replace(&mut s.resolution, resolution);
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.quad_kill = [0; 2];
        old_resolution
    });
    drop(old_resolution);
}

/// #142 `SetQuadSampleMaskX`: the io words of the list's next draws (`quad_mask::kill_words`).
pub(super) fn set_quad_kill(list: *mut c_void, words: [u32; 2]) {
    with(list, |s| s.quad_kill = words);
}

/// Allocation-bearing root state captured before a mid-list split commits. Preparation also
/// transfers the reserved destination's old UMA ownership, so installation cannot allocate.
pub(super) struct PreparedSplit {
    to: usize,
    allocator: usize,
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    layout_gfx: Option<Arc<Layout>>,
    args_gfx: Vec<u8>,
    rs_gfx: usize,
    pso: usize,
    resolution: Option<super::pso_async::PipelineResolution>,
    quad_kill: [u32; 2],
}

impl PreparedSplit {
    /// Native Reset may borrow only an already-owned exact generation. A retained asynchronous
    /// pipeline which is not ready starts the continuation with no PSO and binds at emission.
    pub(super) fn initial_pipeline(&self) -> (usize, Option<super::pso_async::NativeGeneration>) {
        initial_pipeline_for_reset(self.pso, self.resolution.as_ref())
    }
}

fn initial_pipeline_for_reset(
    pso: usize,
    resolution: Option<&super::pso_async::PipelineResolution>,
) -> (usize, Option<super::pso_async::NativeGeneration>) {
    match resolution {
        Some(resolution) => resolution.ready_for_reset().map_or((0, None), |(raw, pin)| (raw, Some(pin))),
        None => (pso, None),
    }
}

/// Exact retained initial pipeline for Reset after a recording is submitted. This snapshots the
/// bind identity before waiting on no work and never resolves the title's reusable address.
pub(super) fn initial_pipeline(list: *mut c_void) -> (usize, Option<super::pso_async::NativeGeneration>) {
    let (pso, resolution) = with(list, |s| (s.pso, s.resolution.clone()));
    initial_pipeline_for_reset(pso, resolution.as_ref())
}

pub(super) fn prepare_split(from: *mut c_void, to: *mut c_void, allocator: usize) -> PreparedSplit {
    let backing = native_backing(to, allocator);
    let (layout, args, layout_gfx, args_gfx, rs_gfx, pso, resolution, quad_kill) = with(from, |s| {
        (s.layout.clone(), s.args.clone(), s.layout_gfx.clone(), s.args_gfx.clone(), s.rs_gfx, s.pso,
            s.resolution.clone(), s.quad_kill)
    });
    let (old, blocks) = with(to, |s| {
        let old = std::mem::replace(&mut s.allocator, 0);
        s.arena = Default::default();
        (old, std::mem::take(&mut s.uma))
    });
    console_list::hold_until_reset(old, blocks);
    PreparedSplit { to: to as usize, allocator: backing, layout, args, layout_gfx, args_gfx, rs_gfx, pso, resolution, quad_kill }
}

/// Install a preflighted split state without cloning or growing the state map.
pub(super) fn install_prepared_split(prepared: PreparedSplit) {
    let PreparedSplit { to, allocator, layout, args, layout_gfx, args_gfx, rs_gfx, pso, resolution, quad_kill } = prepared;
    let entry = STATES.get(to).expect("prepared split destination state");
    let old_resolution = {
        let mut s = entry.state.lock().unwrap();
        debug_assert_eq!(s.allocator, 0);
        debug_assert!(s.uma.is_empty());
        s.allocator = allocator;
        s.layout = layout;
        s.args = args;
        s.layout_gfx = layout_gfx;
        s.args_gfx = args_gfx;
        s.rs_gfx = rs_gfx;
        s.pso = pso;
        s.quad_kill = quad_kill;
        std::mem::replace(&mut s.resolution, resolution)
    };
    drop(old_resolution);
}

pub(super) fn inherit_for_split(from: *mut c_void, to: *mut c_void, allocator: usize) {
    install_prepared_split(prepare_split(from, to, allocator));
}

/// A put-off dispatch that could not be recorded.
pub(super) fn dispatch_dropped() {
    count(DISPATCHES_SKIPPED);
}

/// `to` goes on in the arena `from` took its pieces from (`from` takes no more).
pub(super) fn pass_arena(from: *mut c_void, to: *mut c_void) {
    let arena = with(from, |s| std::mem::take(&mut s.arena));
    with(to, |s| s.arena = arena);
}

/// The list is gone.
pub(super) fn forget(list: *mut c_void) {
    if let Some(s) = take_state(list as usize) {
        console_list::hold_until_reset(s.allocator, s.uma);
    }
}

fn take_state(list: usize) -> Option<State> {
    let entry = STATES.remove(list)?;
    let state = std::mem::take(&mut *entry.state.lock().unwrap());
    Some(state)
}

#[cfg(test)]
pub(crate) fn seed_reset_state_for_test(list: *mut c_void, allocator: usize, pso: usize, uma: Vec<usize>) {
    with(list, |s| {
        s.allocator = allocator;
        s.pso = pso;
        s.layout = Some(Arc::new(Layout::default()));
        s.args = vec![1, 2, 3, 4];
        s.layout_gfx = Some(Arc::new(Layout::default()));
        s.args_gfx = vec![5, 6, 7, 8];
        s.uma = uma;
    });
}

#[cfg(test)]
pub(crate) fn reset_state_for_test(list: *mut c_void) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<usize>) {
    with(list, |s| (s.allocator, s.pso, s.args.clone(), s.args_gfx.clone(), s.uma.clone()))
}

#[cfg(test)]
pub(crate) fn remove_state_for_test(list: *mut c_void) {
    drop(take_state(list as usize));
}

/// bytes of parameter `i` at constant `offset` (of the compute or the graphics arguments)
fn put_in(list: *mut c_void, gfx: bool, i: u32, offset: u32, bytes: &[u8]) {
    with(list, |s| {
        let (layout, args) = if gfx { (&s.layout_gfx, &mut s.args_gfx) } else { (&s.layout, &mut s.args) };
        let Some(at) = layout.as_ref().and_then(|l| l.params.get(i as usize)).map(|p| match *p {
            Param::Constants { at, .. } => at as usize + 4 * offset as usize,
            Param::Descriptor { at } | Param::Table { at } => at as usize,
        }) else {
            return;
        };
        if let Some(t) = args.get_mut(at..at + bytes.len()) {
            t.copy_from_slice(bytes);
        }
    })
}

fn put(list: *mut c_void, i: u32, offset: u32, bytes: &[u8]) {
    put_in(list, false, i, offset, bytes)
}

/// #34 SetGraphicsRootSignature (PC #30)
pub(super) unsafe extern "system" fn set_graphics_root_signature(this: *mut c_void, rs: *mut c_void) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args_gfx = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout_gfx = layout;
        s.rs_gfx = rs as usize;
    });
}

/// #36 SetGraphicsRootDescriptorTable (PC #32)
pub(super) unsafe extern "system" fn set_graphics_root_table(this: *mut c_void, i: u32, handle: u64) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    put_in(this, true, i, 0, &(handle as u32).to_le_bytes());
}

/// #38 SetGraphicsRoot32BitConstant (PC #34)
pub(super) unsafe extern "system" fn set_graphics_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    put_in(this, true, i, offset, &value.to_le_bytes());
}

/// #40 SetGraphicsRoot32BitConstants (PC #36)
pub(super) unsafe extern "system" fn set_graphics_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put_in(this, true, i, offset, &bytes);
    }
}

/// #42 / #44 / #46 SetGraphicsRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #38 /
/// #40 / #42)
pub(super) unsafe extern "system" fn set_graphics_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    note_address(va);
    put_in(this, true, i, 0, &va.to_le_bytes());
}

/// `XCL_TRACE_ADDRESSES=1`: root descriptor addresses of console code – in unified memory
/// (translated shaders see them) or not – counted and logged
fn note_address(va: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static IN: AtomicU64 = AtomicU64::new(0);
    static OUT: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("XCL_TRACE_ADDRESSES").is_some()) {
        return;
    }
    let inside = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    let (i, o) = if inside { (IN.fetch_add(1, Ordering::Relaxed) + 1, OUT.load(Ordering::Relaxed)) } else { (IN.load(Ordering::Relaxed), OUT.fetch_add(1, Ordering::Relaxed) + 1) };
    if (i + o).is_power_of_two() {
        crate::xlog_info!("D3D12.X: root descriptor addresses: {i} in unified memory, {o} outside (latest {va:#x})");
    }
}

/// Immutable classification of one successfully created command signature. Replacing this value
/// in one map operation prevents address reuse or concurrent execution from mixing generations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct IndirectSignatureInfo {
    dispatch_only_stride: u32,
    flags: u8,
}

impl IndirectSignatureInfo {
    const DISPATCHES: u8 = 1;
    const ROOT_CHANGES: u8 = 2;

    pub(super) fn new(dispatches: bool, root_changes: bool, dispatch_only_stride: Option<u32>) -> Self {
        Self {
            dispatch_only_stride: dispatch_only_stride.unwrap_or(0),
            flags: (dispatches as u8) * Self::DISPATCHES | (root_changes as u8) * Self::ROOT_CHANGES,
        }
    }

    fn dispatches(self) -> bool { self.flags & Self::DISPATCHES != 0 }
    fn root_changes(self) -> bool { self.flags & Self::ROOT_CHANGES != 0 }
    fn dispatch_only(self) -> Option<u32> { (self.dispatch_only_stride != 0).then_some(self.dispatch_only_stride) }
}

static INDIRECT_SIGNATURES: Mutex<Option<HashMap<usize, IndirectSignatureInfo>>> = Mutex::new(None);

pub(super) fn note_indirect_signature(sig: usize, classification: IndirectSignatureInfo) {
    INDIRECT_SIGNATURES.lock().unwrap().get_or_insert_with(HashMap::new).insert(sig, classification);
}

fn indirect_signature(sig: usize) -> IndirectSignatureInfo {
    INDIRECT_SIGNATURES.lock().unwrap().as_ref()
        .and_then(|signatures| signatures.get(&sig).copied()).unwrap_or_default()
}

/// #59 ExecuteIndirect(signature, max count, arguments, offset, count buffer, offset): with a
/// translated pipeline, on XCL's ABI with the kept root arguments (signatures that change root
/// arguments themselves cannot be – skipped, logged)
pub(super) unsafe extern "system" fn execute_indirect(this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) {
    execute_indirect_classified(this, sig, max, args, args_off, count_buf, count_off, indirect_signature(sig))
}

pub(super) unsafe fn execute_indirect_classified(
    this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64,
    count_buf: usize, count_off: u64, signature: IndirectSignatureInfo,
) {
    let (pso, resolution) = with(this, |state| (state.pso, state.resolution.clone()));
    execute_indirect_snapshot(
        this, sig, max, args, args_off, count_buf, count_off, signature, pso, resolution, true,
    )
}

pub(super) unsafe fn execute_indirect_retained(
    this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64,
    count_buf: usize, count_off: u64, signature: IndirectSignatureInfo,
) {
    let (pso, resolution) = with(this, |state| (state.pso, state.resolution.clone()));
    execute_indirect_snapshot(
        this, sig, max, args, args_off, count_buf, count_off, signature, pso, resolution, false,
    )
}

unsafe fn execute_indirect_snapshot(
    this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64,
    count_buf: usize, count_off: u64, signature: IndirectSignatureInfo, raw_pso: usize,
    resolution: Option<super::pso_async::PipelineResolution>, allow_defer: bool,
) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
    let pc: F = std::mem::transmute(super::real::orig_fn(this, 59));
    let dispatches = signature.dispatches();
    let _detail = crate::stall::detail_span(if dispatches { crate::stall::StallKind::DetailDispatch } else { crate::stall::StallKind::DetailDraw });
    let retained_compute = resolution.as_ref().map(|resolution| resolution.is_compute());
    if allow_defer && dispatches && retained_compute == Some(true) {
        let resolution = resolution.as_ref().unwrap();
        if super::list_x::defer_retained(this, resolution, super::list_x::Deferred::Indirect {
            sig, max, args, args_off, count_buf, count_off, signature,
        }) { return }
    }
    let (pso, bound_unbuilt) = match resolution {
        Some(resolution) => match resolution.resolve_for_indirect() {
            super::pso_async::IndirectPipelineChoice::Exact { raw, generation } => {
                super::list_x::record_native_generation(this as usize, Some(generation));
                (raw, false)
            }
            super::pso_async::IndirectPipelineChoice::InitialFallback { raw, bound_unbuilt } => {
                (raw, bound_unbuilt)
            }
            super::pso_async::IndirectPipelineChoice::Unavailable => {
                count_indirect_skip(dispatches);
                return;
            }
        },
        None => (raw_pso, false),
    };
    let (gfx, compute) = (super::real::is_translated_graphics(pso), super::real::is_translated_compute(pso));
    let args_off_title = args_off;
    let (args_off, count_off) = (args_off + super::real::buffer_delta(args), count_off + super::real::buffer_delta(count_buf));
    if !gfx && !compute {
        // A pipeline still being built (or whose build failed): its stand-in draws nothing, and
        // the list may hold XCL's ABI root signature from a translated draw before it – the PC
        // call with that pair is invalid (debug layer 953, "the currently set Root Signature
        // doesn't match the currently set Pipeline State Object"; a retail title's world then
        // hung the GPU, 0x887a0006). Skipped, like a draw or dispatch of such a pipeline.
        if bound_unbuilt {
            count_indirect_skip(dispatches);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a pipeline that is still being built skipped");
            }
            return;
        }
        // A finished two-pass MS/GS pipeline is a PC mesh shader pipeline (MS+PS) registered in
        // TWO_PASS rather than TRANSLATED_GRAPHICS. Falling through to raw PC ExecuteIndirect
        // with the title's graphics root signature (e.g. a non-DispatchMesh indirect draw or a
        // root-changing signature) invalidates the command list (Close → E_INVALIDARG).
        if let Some(tp) = super::real::two_pass_of_exact(pso) {
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!(
                    "D3D12.X: ExecuteIndirect with two-pass {} pipeline ({pso:#x}) skipped – PC pipeline is a mesh shader",
                    tp.kind
                );
            }
            return;
        }
        // a console compute program without a translation (its build failed): nothing to run,
        // as for Dispatch. The PC call would run it with the list's compute root signature, which
        // can still be XCL's ABI one from a translated dispatch before it (debug layer 953).
        // A native PC compute pipeline is not one of these and still goes to the PC call.
        // (`CONSOLE_COMPUTE` holds the title's object, which for a shared stand-in differs from
        // the driver pipeline `resolve_wait` returned.)
        if dispatches && retained_compute.unwrap_or_else(|| super::real::is_console_compute(raw_pso)) {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect dispatch with a pipeline that has no translated console shader ({raw_pso:#x}) skipped");
            }
            return;
        }
        // a PC pipeline: the title's root signature, as for a draw (`graphics_bound`)
        let rs = with(this, |s| s.rs_gfx) as *mut c_void;
        if let (Some(list), Some(r)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), ID3D12RootSignature::from_raw_borrowed(&rs)) {
            list.SetGraphicsRootSignature(r);
        }
        return fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
    }
    if signature.root_changes() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect that changes root arguments with a translated pipeline skipped");
        }
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if compute && max == 1 && count_buf == 0 && signature.dispatch_only().is_some() {
        let chained = indirect_address(args, args_off_title, args_off).map_or(Ok(false), |address| {
            super::real::dispatch_chain::execute_indirect(list, raw_pso, pso, with(this, |s| s.allocator), address,
                |root| bind_abi_with_root_resolved(this, list, raw_pso, pso, false, root, [0; 3]).map(|_| ()))
        });
        match chained {
            Ok(true) => {
                count(DISPATCHES);
                return;
            }
            Ok(false) => {}
            Err(e) => {
                count(DISPATCHES_SKIPPED);
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 4 {
                    crate::xlog_warn!("D3D12.X: split ExecuteIndirect skipped: {e}");
                }
                return;
            }
        }
    }
    match bind_abi_resolved(this, list, raw_pso, pso, gfx) {
        Ok(()) => {
            if compute {
                if super::real::pso_dispatch_remap_exact(pso).is_some() {
                    match super::real::remap_fallback::whole_exact(pso) {
                        Ok(fallback) => super::real::bind_pipeline_exact(
                            this, raw_pso, fallback.as_raw() as usize,
                        ),
                        Err(e) => {
                            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                                crate::xlog_warn!("D3D12.X: ExecuteIndirect remap fallback skipped: {e}");
                            }
                            count(DISPATCHES_SKIPPED);
                            return;
                        }
                    }
                }
                match super::real::dispatch_chain::whole(pso) {
                    Ok(Some(whole)) => super::real::bind_pipeline_exact(
                        this, raw_pso, whole.as_raw() as usize,
                    ),
                    Ok(None) => {}
                    Err(e) => {
                        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if N.fetch_add(1, Ordering::Relaxed) < 4 {
                            crate::xlog_warn!("D3D12.X: ExecuteIndirect whole compute PSO: {e}");
                        }
                        count(DISPATCHES_SKIPPED);
                        return;
                    }
                }
            }
            count(if gfx { DRAWS } else { DISPATCHES });
            fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off))
        }
        Err(e) => {
            count(if gfx { DRAWS_SKIPPED } else { DISPATCHES_SKIPPED });
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a translated pipeline skipped: {e}");
            }
        }
    }
}

fn count_indirect_skip(dispatches: bool) {
    count(if dispatches { DISPATCHES_SKIPPED } else { DRAWS_SKIPPED });
    #[cfg(test)]
    INDIRECT_SKIP_COUNTS.with(|counts| {
        let mut next = counts.get();
        next[dispatches as usize] += 1;
        counts.set(next);
    });
}

#[cfg(test)]
thread_local! {
    static INDIRECT_SKIP_COUNTS: std::cell::Cell<[u32; 2]> = const { std::cell::Cell::new([0; 2]) };
}

#[cfg(test)]
fn take_indirect_skip_counts() -> [u32; 2] {
    INDIRECT_SKIP_COUNTS.with(|counts| counts.replace([0; 2]))
}

unsafe fn indirect_address(resource: usize, title_offset: u64, pc_offset: u64) -> Option<u64> {
    match super::real::console_buffer(resource) {
        Some((base, size)) if title_offset.checked_add(12).is_some_and(|end| end <= size as u64) => {
            super::real::uma_va_of(base + title_offset as usize)
        }
        Some(_) => None,
        None => Some(ID3D12Resource::from_raw_borrowed(&(resource as *mut c_void))?.GetGPUVirtualAddress())
            .filter(|&address| address != 0).and_then(|address| address.checked_add(pc_offset)),
    }
}

/// XCL's own buffer for `fresh_indirect_arguments` (device, buffer): COMMON between its uses
static ARGUMENT_ORDER: Mutex<Option<(usize, ID3D12Resource)>> = Mutex::new(None);

/// `op` (an ExecuteIndirect on `list`) reading the arguments and count the list wrote before it.
///
/// The console orders a dispatch that writes indirect arguments before the ExecuteIndirect that
/// reads them with a barrier that reaches the PC as a UAV barrier (`list_x::resource_barrier_x`).
/// NVIDIA's driver (RTX 5090) fetches an ExecuteIndirect's arguments without waiting for that
/// barrier: it ran the count that was there before the list in all but a few of 100 tries, also
/// when the dispatch wrote them through another resource over the same memory, as translated
/// shaders do through the memory window buffers (`execute_indirect_reads_the_arguments_written_before_it`).
/// A retail title's ExecuteIndirect after the dispatch that writes its group count then runs
/// whatever count the memory held before. A transition to INDIRECT_ARGUMENT makes the driver
/// wait for the writes – of any buffer, as measured there. The title's buffer is in the title's
/// state (its transitions reach the PC list), so a buffer of XCL's own takes it, COMMON ->
/// INDIRECT_ARGUMENT before and back after. Only on a list XCL made (`list_x::on_create`) and
/// not a bundle, which cannot hold barriers; its type comes from XCL's state, not a call to it.
unsafe fn fresh_indirect_arguments(list: *mut c_void, op: impl FnOnce()) {
    let tracked = super::list_x::tracked_list_type(list).is_some_and(|t| t != D3D12_COMMAND_LIST_TYPE_BUNDLE);
    let buffer = if tracked { ID3D12GraphicsCommandList::from_raw_borrowed(&list).and_then(|l| argument_order_buffer(l)) } else { None };
    let Some(buffer) = buffer else { return op() };
    let b = buffer.as_raw() as usize;
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    op();
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
}

/// The device's buffer for `fresh_indirect_arguments` (made on first use)
unsafe fn argument_order_buffer(list: &ID3D12GraphicsCommandList) -> Option<ID3D12Resource> {
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let mut g = ARGUMENT_ORDER.lock().unwrap();
    if let Some((_, b)) = g.as_ref().filter(|(d, _)| *d == dev.as_raw() as usize) {
        return Some(b.clone());
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut b) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: no buffer to order its arguments after the writes before it ({e}) – it may read old ones");
        }
        return None;
    }
    let b = b?;
    *g = Some((dev.as_raw() as usize, b.clone()));
    Some(b)
}

/// A DrawInstanced of the console's rect list (three corners of each rectangle, the fourth
/// implied) as two triangles a rectangle; false: not a rect list, or one this cannot expand
/// (the caller draws the three vertices as a triangle – half of each rectangle).
///
/// The PC has no rect lists. The fourth corner is made here, on the CPU, from the title's
/// vertex data: v3 = v0 + v2 − v1 (seen in a retail title's full-screen passes: (−1,1) (−1,−1)
/// (1,−1) with texture coordinates (0,0) (0,1) (1,1) – the missing corner is (1,1), (1,0)).
/// The vertex layout is not looked at: each 4-byte word that is the same in all three vertices
/// is kept (a packed colour), any other is taken as a float. Vertices a shader makes from the
/// vertex index alone (no vertex buffer) are not expanded.
unsafe fn draw_rects(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) -> bool {
    if !cpu_rect_expansion(pso) {
        return false;
    }
    let Some(vbs) = super::list_x::rect_list_state(this) else { return false };
    let rects = (vertices / 3) as usize;
    if vbs.is_empty() || rects == 0 || rects > 4096 {
        return false;
    }
    let mut views = Vec::new();
    for &(slot, (va, bytes, stride), _) in &vbs {
        let (s, at) = (stride as usize, va as usize + first_vertex as usize * stride as usize);
        let len = rects * 3 * s;
        if va == 0 || s == 0 || s % 4 != 0 || first_vertex as usize * s + len > bytes as usize || !crate::gpu::shader::readable(at, len) {
            return false;
        }
        let p = allocate_upload_memory(this, rects * 6 * s, 16);
        if p.is_null() {
            return false;
        }
        let word = |v: usize, w: usize| (at as *const u8).add(v * s + w * 4).cast::<u32>().read_unaligned();
        for r in 0..rects {
            let out = p.add(r * 6 * s);
            let src = (at as *const u8).add(r * 3 * s);
            // v0 v1 v2, then v0 v2 v3
            std::ptr::copy_nonoverlapping(src, out, 3 * s);
            std::ptr::copy_nonoverlapping(src, out.add(3 * s), s);
            std::ptr::copy_nonoverlapping(src.add(2 * s), out.add(4 * s), s);
            for w in 0..s / 4 {
                let (a, b, c) = (word(r * 3, w), word(r * 3 + 1, w), word(r * 3 + 2, w));
                let d = if a == b && b == c { a } else { (f32::from_bits(a) + f32::from_bits(c) - f32::from_bits(b)).to_bits() };
                out.add(5 * s + w * 4).cast::<u32>().write_unaligned(d);
            }
        }
        views.push((slot, D3D12_VERTEX_BUFFER_VIEW { BufferLocation: super::list_x::pc_va_of(p as u64), SizeInBytes: (rects * 6 * s) as u32, StrideInBytes: stride }));
    }
    for (slot, v) in &views {
        list.IASetVertexBuffers(*slot, Some(&[*v]));
    }
    list.DrawInstanced(rects as u32 * 6, instances, 0, first_instance);
    for &(slot, _, pc) in &vbs {
        if let Some(v) = pc {
            list.IASetVertexBuffers(slot, Some(&[v]));
        }
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: rect lists are drawn as two triangles a rectangle, the fourth corner made on the CPU");
    }
    true
}

/// Whether a rect list drawn with the bound pipeline `pso` is expanded on the CPU (`draw_rects`):
/// not with the rect list GS (`real::has_rect_list_gs`), which completes each rect itself – both
/// would draw every rect twice (each CPU half is three corners of the rect).
fn cpu_rect_expansion(pso: usize) -> bool {
    !super::real::has_rect_list_gs(pso)
}

/// #12 DrawInstanced
pub(super) unsafe extern "system" fn draw_instanced(this: *mut c_void, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailDraw);
    DRAW_ARGS.with(|d| d.set((false, vertices, instances, first_vertex, 0)));
    if let Some((list, pso)) = graphics_bound(this) {
        if draw_rects(this, &list, pso, vertices, instances, first_vertex, first_instance) {
            return;
        }
        super::list_x::sync_title_buffers(this, None, first_vertex, vertices, if first_instance != 0 { instances.max(2) } else { instances });
        list.DrawInstanced(vertices, instances, first_vertex, first_instance);
    }
}

thread_local! {
    /// the draw being bound: (indexed, count, instances, first index / vertex, base vertex) – for the frame trace
    static DRAW_ARGS: std::cell::Cell<(bool, u32, u32, u32, i32)> = const { std::cell::Cell::new((false, 0, 0, 0, 0)) };
}

#[cfg(test)]
static LAST_ABI_PSO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static STOP_ABI_LIST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static STOP_ABI_AFTER_BIND_LIST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static STOP_ABI_TEST: Mutex<()> = Mutex::new(());

/// #13 DrawIndexedInstanced
pub(super) unsafe extern "system" fn draw_indexed_instanced(this: *mut c_void, indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailDraw);
    if let Some(list) = prepare_indexed_draw(this, indices, instances, first_index, base_vertex) {
        super::list_x::sync_title_buffers(this, Some((first_index, indices, base_vertex)), 0, 0, if first_instance != 0 { instances.max(2) } else { instances });
        emit_prepared_indexed_draw(&list, indices, instances, first_index, base_vertex, first_instance);
    }
}

/// Prepare an indexed draw once, before callers make temporary native state changes.
pub(super) unsafe fn prepare_indexed_draw(
    this: *mut c_void,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
) -> Option<ID3D12GraphicsCommandList> {
    DRAW_ARGS.with(|d| d.set((true, indices, instances, first_index, base_vertex)));
    graphics_bound(this).map(|(list, _)| list)
}

/// Submit an indexed draw whose translated graphics state has already been prepared.
pub(super) unsafe fn emit_prepared_indexed_draw(
    list: &ID3D12GraphicsCommandList,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    list.DrawIndexedInstanced(indices, instances, first_index, base_vertex, first_instance);
}

/// Resolve one retained bind for correctness-critical draw or dispatch emission and record its
/// exact generation before native commands can consume it. None is fail-closed.
fn resolve_pipeline(
    list: usize,
    pso: usize,
    resolution: Option<super::pso_async::PipelineResolution>,
) -> Option<usize> {
    match resolution.map(|resolution| resolution.resolve_for_emission()) {
        Some(super::pso_async::PipelineChoice::Available { raw, generation }) => {
            super::list_x::record_native_generation(list, Some(generation));
            Some(raw)
        }
        Some(super::pso_async::PipelineChoice::Unavailable) => None,
        None => Some(pso),
    }
}

fn resolve_bound_graphics_pipeline(this: *mut c_void, list: usize) -> Option<(usize, usize)> {
    let (pso, resolution) = with(this, |s| (s.pso, s.resolution.clone()));
    resolve_pipeline(list, pso, resolution).map(|real| (pso, real))
}

unsafe fn graphics_bound(this: *mut c_void) -> Option<(ID3D12GraphicsCommandList, usize)> {
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?.clone();
    let Some((pso, real)) = resolve_bound_graphics_pipeline(this, list.as_raw() as usize) else {
        super::list_x::note_dropped_draw(this);
        count(DRAWS_SKIPPED);
        return None;
    };
    if !super::real::is_translated_graphics(real) {
        // A finished two-pass MS/GS pipeline is a PC mesh shader pipeline (MS+PS) registered in
        // TWO_PASS rather than TRANSLATED_GRAPHICS. Binding the title's graphics RS and issuing
        // DrawInstanced / DrawIndexedInstanced against a mesh PSO invalidates the command list
        // (Close → E_INVALIDARG).
        if let Some(tp) = super::real::two_pass_of_exact(real) {
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!(
                    "D3D12.X: draw with two-pass {} pipeline ({real:#x}) skipped – PC pipeline is a mesh shader",
                    tp.kind
                );
            }
            return None;
        }
        count(DRAWS_PC);
        if super::present::trace_frame() {
            // which pipelines draw without a translation (first 12 distinct)
            static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
            let mut g = SEEN.lock().unwrap();
            if !g.contains(&pso) && g.len() < 12 {
                g.push(pso);
                crate::xlog_info!(
                    "frame trace: draw with an untranslated pipeline {pso:#x} (resolves to {real:#x}; pending {}, two-pass {}, RT0 {:?})",
                    super::pso_async::is_pending(pso),
                    super::real::two_pass_of_exact(real).is_some(),
                    super::present::rt0_resource(this as usize).map(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0) }))
                );
            }
        }
        // a PC pipeline (also a stand-in or fallback: made with the title's root signature):
        // the title's root signature, which nothing else binds on the PC list – a draw without
        // one is invalid (the debug layer reports it on every GPU; WARP faults on it)
        if super::pso_async::is_pending(pso) {
            super::list_x::note_dropped_draw(this);
        }
        let rs = with(this, |s| s.rs_gfx) as *mut c_void;
        if let Some(r) = ID3D12RootSignature::from_raw_borrowed(&rs) {
            list.SetGraphicsRootSignature(r);
        }
        return Some((list, real));
    }
    if super::present::trace_frame() {
        trace_ui_draw(this, pso, real);
    }
    match bind_abi_resolved(this, &list, pso, real, true) {
        Ok(()) => {
            count(DRAWS);
            Some((list, real))
        }
        Err(e) => {
            super::list_x::note_dropped_draw(this);
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: draw with a translated pipeline skipped: {e}");
            }
            None
        }
    }
}

/// `XCL_TRACE_FRAME`: draws into a presented buffer (the UI) once the title has presented for a
/// while – their pipeline, root arguments and the resources their descriptor tables start with
/// (format, size, kept in step with title memory or not). The first 24.
/// `XCL_TRACE_FRAME`, after 300 presents: one line per distinct (VS, PS, render target) drawn –
/// the shaders' dump hashes, the target and the textures of the first descriptor table.
unsafe fn trace_draw_kinds(this: *mut c_void, pso: usize) {
    // keyed by the target's shape, not the object: a UI library's transient targets (Renoir's
    // layers) are new objects every frame and would fill the list at once
    static SEEN: Mutex<Vec<(u64, u64, Option<(u64, u32, i32, u32)>)>> = Mutex::new(Vec::new());
    if super::present::activity().0 < 300 {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(pso) else { return };
    let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
    let desc = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.SampleDesc.Count) });
    {
        let mut g = SEEN.lock().unwrap();
        let key = (vs, ps, desc(rt));
        if g.len() >= 160 || g.contains(&key) {
            return;
        }
        g.push(key);
    }
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut tex = Vec::new();
    if let (Some(layout), Some(u)) = (layout, crate::gpu::uma::get()) {
        for p in layout.params.iter() {
            if let Param::Table { at } = *p {
                let t = word(at as usize) as usize;
                for k in 0..4usize {
                    if let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) {
                        tex.push(format!("{:?}{}", desc(r), super::title_textures::address(r).map_or(String::new(), |a| format!(" at {a:#x}"))));
                    }
                }
            }
            if tex.len() >= 4 {
                break;
            }
        }
    }
    crate::xlog_info!(
        "frame trace: draws vs {vs:016x} ps {ps:016x} into {:?}{} – textures {}; {}",
        desc(rt),
        if super::present::is_presented(rt) { " (presented)" } else { "" },
        tex.join(", "),
        super::list_x::state_summary(this)
    );
}

/// `XCL_TRACE_SHADERS=hash,…` (dump hashes, hex): the draws traced with their root arguments
/// are those whose vertex or pixel shader is listed – into any target, at any time (the first
/// 24); without it, draws into a presented target (or any – `XCL_TRACE_UI_ANY`) after 300
/// presents.
fn traced_shaders() -> &'static [u64] {
    static S: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("XCL_TRACE_SHADERS").unwrap_or_default().split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect())
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): draws whose descriptor
/// tables name it – that object, or another at its title address (the first 16).
unsafe fn trace_reads_of_watched(this: *mut c_void, pso: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let w_addr = super::real::title_address(w).or_else(|| super::title_textures::address(w));
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..16usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let addr = super::real::title_address(r).or_else(|| super::title_textures::address(r));
            if super::present::is_watched(r) || (addr.is_some() && addr == w_addr) {
                let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
                // each (source, target, shaders) once
                static SEEN: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
                {
                    let mut g = SEEN.lock().unwrap();
                    if g.len() >= 32 || g.contains(&(r, rt, pso)) {
                        return;
                    }
                    g.push((r, rt, pso));
                }
                if !super::present::is_presented(rt) {
                    super::present::follow(rt);
                }
                if let Some(dsv) = super::list_x::bound_dsv(this) {
                    super::list_x::watch_dsv(dsv);
                }
                let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.Flags.0) });
                crate::xlog_info!(
                    "frame trace: a draw (shaders {:x?}, {}) reads a watched target {r:#x} {:?} at {addr:x?} (slot {k}), drawing into {rt:#x} {:?}{} – {}",
                    super::real::pso_shaders(pso),
                    super::list_x::state_summary(this),
                    d(r),
                    d(rt),
                    if super::present::is_presented(rt) { " (presented)" } else { "" },
                    root_args_summary(this, 32)
                );
                let args = DRAW_ARGS.with(|d| d.get());
                crate::xlog_info!(
                    "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
                    super::real::pso_layout(pso).unwrap_or_default(),
                    super::list_x::geometry_summary(this, args)
                );
                return;
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: render targets a draw into the watched target samples (a UI library's
/// layers) are dumped with it (`present::follow`).
unsafe fn follow_layer_sources(this: *mut c_void) {
    let (Some(w), Some(rt)) = (super::present::watched_rt(), super::present::rt0_resource(this as usize)) else { return };
    if rt != w {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..4usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let layer = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| {
                let d = x.GetDesc();
                d.Flags.0 & 1 != 0 && d.Width >= 64 && d.Height >= 32 && d.SampleDesc.Count == 1
            });
            if layer && r != w {
                super::present::follow(r);
            }
        }
    }
}

unsafe fn trace_ui_draw(this: *mut c_void, original: usize, pso: usize) {
    trace_draw_kinds(this, pso);
    trace_reads_of_watched(this, pso);
    follow_layer_sources(this);
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // `XCL_TRACE_MANY=n`: n draws instead of 24 (a whole frame of a UI)
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| std::env::var("XCL_TRACE_MANY").ok().and_then(|v| v.parse().ok()).unwrap_or(24));
    if N.load(std::sync::atomic::Ordering::Relaxed) >= cap {
        return;
    }
    let Some(rt) = super::present::rt0_resource(this as usize) else { return };
    let wanted = traced_shaders();
    if wanted.is_empty() {
        // draws into a presented target, or – `XCL_TRACE_UI_ANY` – into any target
        if super::present::activity().0 < 300 || (!super::present::is_presented(rt) && std::env::var_os("XCL_TRACE_UI_ANY").is_none()) {
            return;
        }
    } else if !super::real::pso_shaders(pso).is_some_and(|(v, p)| wanted.contains(&v) || wanted.contains(&p)) {
        return;
    } else {
        super::present::watch_rt(rt);
    }
    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sh = super::real::pso_shaders(pso).map(|(v, p)| format!("vs {v:016x} ps {p:016x}"));
    crate::xlog_info!("frame trace: UI draw (pipeline {original:#x}, {}): {}", sh.unwrap_or_default(), root_args_summary(this, 16));
    if !wanted.is_empty() {
        // the picked shaders' geometry too (vertices dumped at the next present)
        let args = DRAW_ARGS.with(|d| d.get());
        crate::xlog_info!(
            "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
            super::real::pso_layout(pso).unwrap_or_default(),
            super::list_x::geometry_summary(this, args)
        );
    }
}

/// A draw's root arguments for the frame trace: constants, root buffers (their first words, as
/// the CPU sees them), the first slots of descriptor tables (the PC resources they name).
unsafe fn root_args_summary(this: *mut c_void, buffer_words: usize) -> String {
    let Some(u) = crate::gpu::uma::get() else { return String::new() };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return String::new() };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut what = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => what.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
            Param::Descriptor { at } => {
                let va = (word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32) as usize;
                let w: Vec<u32> = if va != 0 && crate::gpu::shader::readable(va, 4 * buffer_words) { (0..buffer_words).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect() } else { Vec::new() };
                what.push(format!("p{i} buffer {va:#x} {w:x?}"));
            }
            Param::Table { at } => {
                let t = word(at as usize) as usize;
                let slots: Vec<String> = (0..4usize)
                    .map(|k| {
                        let at = u.base() + t + 32 * k;
                        let w = super::console_heap::words_at((t + 32 * k) as u32);
                        match super::console_heap::resource_at(at) {
                            Some(r) => {
                                if super::title_textures::address(r).is_some() {
                                    super::title_textures::probe_gpu_content(r);
                                }
                                let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
                                format!("[{k}] {r:#x} {:?} title {:?} T# {:x?}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), super::title_textures::address(r).map(|a| format!("{a:#x}")), w.map(|w| w[..4].to_vec()))
                            }
                            None => format!("[{k}] - T# {:x?}", w.map(|w| w[..4].to_vec())),
                        }
                    })
                    .collect();
                what.push(format!("p{i} table {t:#x}: {}", slots.join(", ")));
            }
        }
    }
    what.join("; ")
}

/// #33 SetComputeRootSignature (PC #29)
pub(super) unsafe extern "system" fn set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout = layout;
    });
}

/// #35 SetComputeRootDescriptorTable (PC #31): the table's console address, 32 bits
pub(super) unsafe extern "system" fn set_compute_root_table(this: *mut c_void, i: u32, handle: u64) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    put(this, i, 0, &(handle as u32).to_le_bytes());
}

/// #37 SetComputeRoot32BitConstant (PC #33)
pub(super) unsafe extern "system" fn set_compute_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    put(this, i, offset, &value.to_le_bytes());
}

/// #39 SetComputeRoot32BitConstants (PC #35)
pub(super) unsafe extern "system" fn set_compute_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put(this, i, offset, &bytes);
    }
}

/// #41 / #43 / #45 SetComputeRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #37 /
/// #39 / #41): the 64-bit GPU address
pub(super) unsafe extern "system" fn set_compute_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    note_address(va);
    put(this, i, 0, &va.to_le_bytes());
}

/// #25 SetPipelineState: retain one title identity for the native bind and later emission.
pub(super) unsafe extern "system" fn set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    let pso = pso as usize;
    let resolution = super::pso_async::begin_pipeline_resolution(pso);
    if resolution.is_none() {
        super::pso_manifest::note_ready_compute(pso);
    }
    set_pipeline_state_retained(this, pso, resolution);
}

#[cfg(test)]
fn set_tracked_pipeline_state(this: *mut c_void, pso: usize) {
    let resolution = super::pso_async::begin_pipeline_resolution(pso);
    install_tracked_pipeline_state(this, pso, resolution);
}

fn install_tracked_pipeline_state(
    this: *mut c_void,
    pso: usize,
    resolution: Option<super::pso_async::PipelineResolution>,
) {
    let old_resolution = with(this, |s| {
        s.pso = pso;
        std::mem::replace(&mut s.resolution, resolution)
    });
    drop(old_resolution);
}

unsafe fn set_pipeline_state_retained(
    this: *mut c_void,
    pso: usize,
    resolution: Option<super::pso_async::PipelineResolution>,
) {
    use super::pso_async::{NativePipelineBindMode, NativePipelineChoice};
    let choice = resolution.as_ref().map(|r| r.select_for_native_bind(NativePipelineBindMode::Lazy));
    let (raw, generation) = match choice {
        Some(NativePipelineChoice::Exact { raw, generation }) => (Some(raw), Some(generation)),
        Some(NativePipelineChoice::InitialFallback { raw, .. }) => (Some(raw), None),
        Some(NativePipelineChoice::Unavailable) if resolution.as_ref().unwrap().unavailable_is_emission_guarded() => {
            (None, None)
        }
        Some(NativePipelineChoice::Unavailable) => {
            // Mesh/raw native consumers can bypass retained draw resolution. Preserve root and
            // dynamic state, but make Close/submission reject this whole physical recording.
            super::list_x::poison_recording(this as usize);
            (None, None)
        }
        None => (Some(pso), None),
    };
    super::list_x::record_native_generation(this as usize, generation);
    install_tracked_pipeline_state(this, pso, resolution);
    if let Some(raw) = raw {
        super::real::bind_pipeline_exact(this, pso, raw);
    }
    super::list_x::on_set_pipeline_state(this, pso);
}

pub(super) unsafe fn clear_pipeline_state(this: *mut c_void, pso: usize) {
    let resolution = super::pso_async::begin_pipeline_resolution(pso);
    clear_pipeline_state_retained(this, pso, resolution);
}

unsafe fn clear_pipeline_state_retained(
    this: *mut c_void,
    pso: usize,
    resolution: Option<super::pso_async::PipelineResolution>,
) {
    use super::pso_async::{NativePipelineBindMode, NativePipelineChoice};
    let choice = resolution.as_ref().map(|r| r.select_for_native_bind(NativePipelineBindMode::ForBind));
    let (raw, generation) = match choice {
        Some(NativePipelineChoice::Exact { raw, generation }) => (raw, Some(generation)),
        Some(NativePipelineChoice::InitialFallback { raw, .. }) => (raw, None),
        Some(NativePipelineChoice::Unavailable) => (0, None),
        None => (pso, None),
    };
    super::list_x::record_native_generation(this as usize, generation);
    super::real::clear_pipeline_exact(this, pso, raw);
    super::list_x::on_clear_state(this, pso);
    install_clear_state(this, pso, resolution);
}

/// #14 Dispatch: with a translated pipeline, on XCL's ABI with the kept root arguments
pub(super) unsafe extern "system" fn dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailDispatch);
    dispatch_flags(this, x, y, z, 0);
}

/// `DispatchX` flag: x, y, z are thread counts, not group counts; the console launches whole
/// groups and leaves the lanes past the count out of EXEC (0x8000, wave32, needs nothing here)
const DISPATCH_THREAD_DIMENSIONS: u32 = 0x20;
const MAX_DISPATCH_GROUPS: u32 = 65_535;

fn valid_dispatch_grid(groups: [u32; 3]) -> bool {
    groups.into_iter().all(|n| n <= MAX_DISPATCH_GROUPS)
}

/// What a dispatch of `dims` with `flags` launches on pipeline `pso`: its groups and the thread
/// limits for `IO_THREAD_LIMIT` (0: that axis ends on a whole group). None: it cannot be run
/// as the console would (thread dimensions with a group shape not known).
fn dispatch_groups(pso: usize, dims: [u32; 3], flags: u32) -> Option<([u32; 3], [u32; 3])> {
    if flags & DISPATCH_THREAD_DIMENSIONS == 0 {
        return Some((dims, [0; 3]));
    }
    let threads = super::real::pso_threads_exact(pso)?.map(|t| t.max(1));
    let groups = std::array::from_fn(|k| dims[k].div_ceil(threads[k]));
    let limits = std::array::from_fn(|k| if dims[k] % threads[k] != 0 { dims[k] } else { 0 });
    Some((groups, limits))
}

fn skip_untranslated_dispatch(pso: usize) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
        crate::xlog_warn!("D3D12.X: Dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped");
    }
    count(DISPATCHES_SKIPPED);
}

/// `Dispatch` / `DispatchX(x, y, z, flags)`.
pub(super) unsafe fn dispatch_flags(this: *mut c_void, x: u32, y: u32, z: u32, flags: u32) {
    let (bound_pso, resolution) = with(this, |s| (s.pso, s.resolution.clone()));
    if let Some(resolution) = &resolution {
        if super::list_x::defer_retained(this, resolution, super::list_x::Deferred::Dispatch(x, y, z, flags)) {
            return;
        }
    }
    let Some(pso) = resolve_pipeline(this as usize, bound_pso, resolution) else {
        skip_untranslated_dispatch(bound_pso);
        return;
    };
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !super::real::is_translated_compute(pso) {
        // a console pipeline without a translation (including a background build that
        // failed): nothing to run (its root arguments were not given to the PC list either)
        skip_untranslated_dispatch(pso);
        return;
    }
    let Some((groups, limits)) = dispatch_groups(pso, [x, y, z], flags) else {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: DispatchX of {x}x{y}x{z} threads (flags {flags:#x}) with a pipeline whose group shape is not known ({pso:#x}) skipped");
        }
        count(DISPATCHES_SKIPPED);
        return;
    };
    if !valid_dispatch_grid(groups) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: Dispatch of {}x{}x{} groups exceeds the PC limit of {MAX_DISPATCH_GROUPS} on one axis and was skipped", groups[0], groups[1], groups[2]);
        }
        count(DISPATCHES_SKIPPED);
        return;
    }
    if flags != 0 {
        // each kind of flags once: what titles pass, and what it launches
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let first = { let mut g = SEEN.lock().unwrap(); !g.contains(&flags) && { g.push(flags); true } };
        if first {
            crate::xlog_info!("D3D12.X: DispatchX with flags {flags:#x}: {x}x{y}x{z} launched as {}x{}x{} groups (thread limits {limits:?})", groups[0], groups[1], groups[2]);
        }
    }
    let done = translated_dispatch(this, list, bound_pso, pso, groups[0], groups[1], groups[2], limits);
    match done {
        Ok(()) => count(DISPATCHES),
        Err(e) => {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: Dispatch with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: a full-screen dispatch (enough groups to cover a 4K target in tiles of up
/// to 128 pixels) – what its root arguments say. A bindless title passes the indices of its
/// descriptors in root constants and constant buffers: every small value there is looked up
/// in the dispatch's descriptor tables, and the resources found (the pass's inputs and
/// outputs) are logged.
unsafe fn trace_fullscreen(this: *mut c_void, pso: usize, groups: [u32; 3]) {
    if groups[0] < 30 || groups[1] < 17 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&pso) || g.len() >= 12 {
            return;
        }
        g.push(pso);
    }
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut what: Vec<String> = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).filter_map(|k| word(at as usize + 4 * k)).collect();
                what.push(format!("p{i} constants {w:x?}"));
                values.extend(&w);
            }
            Param::Descriptor { at } => {
                let lo = word(at as usize).unwrap_or(0) as u64;
                let hi = word(at as usize + 4).unwrap_or(0) as u64;
                let va = (lo | hi << 32) as usize;
                let n = 64usize;
                if va != 0 && crate::gpu::shader::readable(va, 4 * n) {
                    let w: Vec<u32> = (0..n).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect();
                    what.push(format!("p{i} buffer {va:#x}: {w:x?}"));
                    values.extend(&w);
                } else {
                    what.push(format!("p{i} buffer {va:#x} (not readable)"));
                }
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                what.push(format!("p{i} table {v:#x}"));
                tables.push(v);
            }
        }
    }
    let mut found: Vec<(u32, usize)> = Vec::new();
    for &v in &values {
        if v >= 1 << 20 {
            continue;
        }
        for &t in &tables {
            let slot = u.base() + t as usize + 32 * v as usize;
            if let Some(r) = super::console_heap::resource_at(slot) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((v, r));
                }
            }
        }
    }
    let res: Vec<String> = found
        .iter()
        .map(|&(v, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
            format!("[{v}] {r:#x} {:?}{}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), if super::present::is_presented(r) { " PRESENTED" } else { "" })
        })
        .collect();
    crate::xlog_info!("frame trace: full-screen dispatch {groups:?}, pipeline {pso:#x}: {}", what.join("; "));
    crate::xlog_info!("frame trace: … its indices name: {}", if res.is_empty() { "nothing".to_string() } else { res.join("; ") });
}

/// #148 AllocateUploadMemory(size, alignment) → CPU/GPU pointer in unified memory (used by
/// `XboxBvhBuilder.dll` and other console middleware on default title command lists).
/// Freed automatically when the list's command allocator is reset.
pub(super) unsafe extern "system" fn allocate_upload_memory(this: *mut c_void, size: usize, align: usize) -> *mut u8 {
    let Some(u) = crate::gpu::uma::get() else {
        return std::ptr::null_mut();
    };
    let size = size.max(16);
    let align = align.max(16);
    if size <= crate::gpu::uma::ARENA_BLOCK / 2 {
        if let Some(p) = with(this, |s| s.arena.take(u, size, align, &mut s.uma)) {
            return p;
        }
    }
    if u.has_sparse() {
        if let Some((p, _)) = super::sparse::alloc(size, align) {
            with(this, |s| s.uma.push(p));
            return p as *mut u8;
        }
    }
    match u.alloc(size, align) {
        Some(p) => {
            with(this, |s| s.uma.push(p as usize));
            p
        }
        None => std::ptr::null_mut(),
    }
}

unsafe fn translated_dispatch(
    this: *mut c_void,
    list: &ID3D12GraphicsCommandList,
    original: usize,
    pso: usize,
    x: u32,
    y: u32,
    z: u32,
    limits: [u32; 3],
) -> Result<(), String> {
    if super::present::trace_frame() {
        trace_fullscreen(this, pso, [x, y, z]);
    }
    let allocator = bind_compute_abi_resolved(this, list, original, pso, limits)?;
    super::real::dispatch_chain::dispatch(list, original, pso, allocator, [x, y, z])
}

/// XCL's ABI for a translated pipeline: its root signature, the user SGPRs (`io`, the kept
/// arguments the console's way – compute or graphics) and the argument table in unified
/// memory, u1 res, u2… the memory windows; then the pipeline.
struct PreparedAbiState {
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    inline: usize,
    table_at: usize,
    io_at: usize,
    res_at: usize,
    allocator: usize,
    /// the quad sample mask's io words (graphics only)
    quad_kill: [u32; 2],
}

fn prepare_abi_state(
    this: *mut c_void,
    u: &crate::gpu::uma::Uma,
    gfx: bool,
    user_sgprs: Option<u32>,
) -> Result<PreparedAbiState, String> {
    with(this, |state| {
        let (layout, args) = if gfx {
            (state.layout_gfx.clone(), state.args_gfx.clone())
        } else {
            (state.layout.clone(), state.args.clone())
        };
        let inline = layout
            .as_ref()
            .map_or(0, |layout| if gfx { layout.inline_gfx } else { console_list::inline_for(layout, user_sgprs) } as usize)
            .min(args.len());
        let table_at = state.arena.take(u, (args.len() - inline).max(16), 256, &mut state.uma)
            .ok_or("unified memory is full")? as usize;
        let io_at = state.arena.take(
            u,
            crate::gpu::recomp::KERNEL_SGPRS * std::mem::size_of::<u32>(),
            256,
            &mut state.uma,
        ).ok_or("unified memory is full")? as usize;
        let res_at = state.arena.take(u, 256, 256, &mut state.uma).ok_or("unified memory is full")? as usize;
        let quad_kill = if gfx { state.quad_kill } else { [0; 2] };
        Ok(PreparedAbiState { layout, args, inline, table_at, io_at, res_at, allocator: state.allocator, quad_kill })
    })
}

fn prepared_abi_io(
    layout: Option<&Layout>, args: &[u8], table_at: usize, gfx: bool,
    user_sgprs: Option<u32>, uma_base: usize, limits: [u32; 3],
) -> Vec<u32> {
    let mut io = match layout {
        Some(layout) if gfx => console_list::user_data_gfx(layout, args, table_at as u64),
        Some(layout) => console_list::user_data_for(layout, user_sgprs, args, table_at as u64),
        None => vec![table_at as u32, (table_at >> 32) as u32],
    };
    io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (uma_base >> 32) as u32;
    if !gfx {
        for (k, limit) in limits.into_iter().enumerate() {
            io[crate::gpu::recomp::IO_THREAD_LIMIT[k]] = limit;
        }
    }
    io
}

/// A draw's quad sample mask into its pixel shader's io (`recomp::IO_QUAD_KILL`; nothing to
/// write when it takes nothing away: the slots are 0).
fn quad_kill_io(io: &mut [u32], words: [u32; 2]) {
    if words == [0; 2] {
        return;
    }
    for (k, w) in words.into_iter().enumerate() {
        io[crate::gpu::recomp::IO_QUAD_KILL[k]] = w;
    }
    super::quad_mask::note_masked_draw();
}

unsafe fn bind_abi_resolved(
    this: *mut c_void, list: &ID3D12GraphicsCommandList, original: usize, pso: usize, gfx: bool,
) -> Result<(), String> {
    bind_abi_with_root_resolved(this, list, original, pso, gfx, 0, [0; 3]).map(|_| ())
}

unsafe fn bind_compute_abi_resolved(
    this: *mut c_void,
    list: &ID3D12GraphicsCommandList,
    original: usize,
    pso: usize,
    limits: [u32; 3],
) -> Result<usize, String> {
    bind_abi_with_root_resolved(this, list, original, pso, false, 0, limits)
}

unsafe fn bind_abi_with_root_resolved(
    this: *mut c_void, list: &ID3D12GraphicsCommandList, original: usize, pso: usize, gfx: bool,
    root: usize, limits: [u32; 3],
) -> Result<usize, String> {
    #[cfg(test)]
    {
        if STOP_ABI_LIST.load(Ordering::Acquire) == this as usize {
            LAST_ABI_PSO.store(pso, Ordering::Release);
            return Err("stopped after resolved PSO for test".into());
        }
    }
    #[cfg(test)]
    if STOP_ABI_AFTER_BIND_LIST.load(Ordering::Acquire) == this as usize && root == 0 {
        super::real::bind_pipeline_exact(this, original, pso);
        LAST_ABI_PSO.store(pso, Ordering::Release);
        return Err("stopped after exact native bind for test".into());
    }
    let shared_view_bound = super::list_x::ensure_view_heap(this);
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let windows = super::real::uma_window_vas();
    let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
    // a compute shader takes no more arguments from registers than it has (`inline_for`)
    let user_sgprs = if gfx { None } else { super::real::pso_user_sgprs_exact(pso) };
    let prepared = prepare_abi_state(this, u, gfx, user_sgprs)?;
    let PreparedAbiState { layout, args, inline, table_at, io_at, res_at, allocator, quad_kill } = prepared;
    std::ptr::copy_nonoverlapping(args[inline..].as_ptr(), table_at as *mut u8, args.len() - inline);
    let mut io = prepared_abi_io(layout.as_deref(), &args, table_at, gfx, user_sgprs, u.base(), limits);
    // the quad sample mask for the pixel shader (graphics only; [0, 0] when all ones)
    quad_kill_io(&mut io, quad_kill);
    let io_bytes = io.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    std::ptr::copy_nonoverlapping(io_bytes.as_ptr(), io_at as *mut u8, io_bytes.len());
    std::ptr::write_bytes(res_at as *mut u8, 0, 256);
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    let rs = (root != 0).then_some(root)
        .or_else(|| super::real::pso_root_signature_exact(pso))
        .or_else(|| super::real::abi_rs_cached().map(|p| p as usize))
        .ok_or("no XCL ABI root signature")?;
    let rs = rs as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs).ok_or("root signature")?;
    let views = [va(io_at), va(res_at)].into_iter().chain((0..4).map(|k| windows.get(k).copied().filter(|&v| v != 0).unwrap_or(first)));
    if gfx {
        list.SetGraphicsRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetGraphicsRootUnorderedAccessView(k as u32, v);
        }
    } else {
        list.SetComputeRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, v);
        }
    }
    if root == 0 {
        super::real::bind_pipeline_exact(this, original, pso);
    }
    if shared_view_bound && super::vram::srv_tracking_enabled() {
        super::list_x::note_shared_view_heap_use(this);
    }
    Ok(allocator)
}

#[cfg(test)]
pub(super) fn last_draw_args() -> (bool, u32, u32, u32, i32) {
    DRAW_ARGS.with(|d| d.get())
}

#[cfg(test)]
pub(super) fn counts_snapshot() -> [u64; 5] {
    std::array::from_fn(|k| COUNTS[k].load(Ordering::Relaxed))
}

#[cfg(test)]
pub(super) fn tracked_pso(list: *mut c_void) -> usize {
    let Some(entry) = STATES.get(list as usize) else { return 0 };
    let pso = entry.state.lock().unwrap().pso;
    pso
}

#[cfg(test)]
fn state_locks_available_for_test(list: usize) -> bool {
    let shard = match STATES.shards[state_shard_index(list)].try_lock() {
        Ok(shard) => shard,
        Err(_) => return false,
    };
    let entry = shard.as_ref().and_then(|states| states.get(&list)).cloned();
    drop(shard);
    let Some(entry) = entry else { return true };
    let available = entry.state.try_lock().is_ok();
    available
}

#[cfg(test)]
mod tests {
    use super::super::{device_x, list_x, pso_async, real, title_list};
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn aligned_list_addresses_reach_many_bind_shards() {
        let shards = (0..256).map(|index| state_shard_index(0x7100_0000 + index * 16)).collect::<HashSet<_>>();
        assert!(shards.len() >= STATE_SHARDS / 2, "aligned list addresses reached only {} shards", shards.len());
    }

    #[test]
    fn busy_bind_shard_does_not_block_an_unrelated_shard() {
        let first = 0x7E57_BA00usize;
        let second = (1..STATE_SHARDS)
            .map(|index| first + index * 16)
            .find(|&key| state_shard_index(key) != state_shard_index(first))
            .expect("shard mixer did not distribute aligned keys");
        let shard = STATES.shards[state_shard_index(first)].lock().unwrap();
        let (sent, received) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            drop(STATES.get_or_insert(second));
            sent.send(()).unwrap();
        });
        received.recv_timeout(Duration::from_secs(1)).expect("unrelated bind shard blocked");
        worker.join().unwrap();
        drop(shard);
        drop(STATES.remove(second));
    }

    #[test]
    fn busy_list_state_does_not_block_another_list_in_the_same_shard() {
        let keys = (0..256).map(|index| Box::new(index as u8)).collect::<Vec<_>>();
        let mut seen = HashMap::new();
        let (first, second) = keys
            .iter()
            .map(|key| (&**key as *const u8) as usize)
            .find_map(|key| seen.insert(state_shard_index(key), key).map(|first| (first, key)))
            .expect("allocator did not provide two keys in one shard");
        let first_entry = STATES.get_or_insert(first);
        drop(STATES.get_or_insert(second));
        let state = first_entry.state.lock().unwrap();
        let (sent, received) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            with(second as *mut c_void, |state| state.pso = 0x7E57_D201);
            sent.send(()).unwrap();
        });
        received.recv_timeout(Duration::from_secs(1)).expect("unrelated list state blocked");
        worker.join().unwrap();
        drop(state);
        drop(STATES.remove(first));
        drop(STATES.remove(second));
    }

    #[test]
    fn forget_then_same_address_create_uses_a_fresh_entry() {
        let key = Box::new(0u8);
        let list = (&*key as *const u8) as *mut c_void;
        reset(list, 0, 0x7E57_D211);
        let old = STATES.get(list as usize).unwrap();
        forget(list);
        reset(list, 0, 0x7E57_D212);
        let new = STATES.get(list as usize).unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        old.state.lock().unwrap().pso = 0x7E57_D213;
        assert_eq!(tracked_pso(list), 0x7E57_D212, "orphaned state changed the replacement");
        forget(list);
    }

    #[test]
    fn warm_bind_cache_does_not_take_the_registry_shard() {
        let key = Box::new(0u8);
        let list = (&*key as *const u8) as usize;
        reset(list as *mut c_void, 0, 0x7E57_D221);
        let (ready_tx, ready_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            assert_eq!(with(list as *mut c_void, |state| state.pso), 0x7E57_D221);
            ready_tx.send(()).unwrap();
            continue_rx.recv().unwrap();
            done_tx.send(with(list as *mut c_void, |state| state.pso)).unwrap();
        });
        ready_rx.recv_timeout(Duration::from_secs(1)).expect("cache worker did not warm its entry");
        let shard = STATES.shards[state_shard_index(list)].lock().unwrap();
        continue_tx.send(()).unwrap();
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(1)).expect("warm cache took its registry shard"), 0x7E57_D221);
        drop(shard);
        worker.join().unwrap();
        forget(list as *mut c_void);
    }

    #[test]
    fn tls_cache_keeps_four_entries() {
        let mut cache = BindStateCache { entries: [const { None }; STATE_CACHE_ENTRIES], next: 0 };
        let entries = (0..=STATE_CACHE_ENTRIES).map(|_| Arc::new(BindEntry::default())).collect::<Vec<_>>();
        for (key, entry) in entries.iter().take(STATE_CACHE_ENTRIES).enumerate() {
            cache.insert(key, entry);
        }
        for key in 0..STATE_CACHE_ENTRIES {
            assert!(Arc::ptr_eq(&cache.get(key).unwrap(), &entries[key]));
        }
        cache.insert(STATE_CACHE_ENTRIES, &entries[STATE_CACHE_ENTRIES]);
        assert!(cache.get(0).is_none(), "FIFO cache did not evict its oldest entry");
        for key in 1..=STATE_CACHE_ENTRIES {
            assert!(Arc::ptr_eq(&cache.get(key).unwrap(), &entries[key]));
        }
    }

    #[test]
    fn cross_thread_cache_rejects_a_reused_address() {
        let key = Box::new(0u8);
        let list = (&*key as *const u8) as usize;
        reset(list as *mut c_void, 0, 0x7E57_D231);
        let old = STATES.get(list).unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            assert_eq!(with(list as *mut c_void, |state| state.pso), 0x7E57_D231);
            ready_tx.send(()).unwrap();
            continue_rx.recv().unwrap();
            with(list as *mut c_void, |state| state.pso)
        });
        ready_rx.recv_timeout(Duration::from_secs(1)).expect("cache worker did not warm its entry");
        forget(list as *mut c_void);
        reset(list as *mut c_void, 0, 0x7E57_D232);
        continue_tx.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), 0x7E57_D232, "worker reused its orphaned cache entry");
        assert!(!old.live.load(Ordering::Acquire));
        forget(list as *mut c_void);
    }

    #[test]
    fn cross_thread_cache_observes_reset_on_the_live_entry() {
        let key = Box::new(0u8);
        let list = (&*key as *const u8) as usize;
        reset(list as *mut c_void, 0, 0x7E57_D241);
        let (ready_tx, ready_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            assert_eq!(with(list as *mut c_void, |state| state.pso), 0x7E57_D241);
            ready_tx.send(()).unwrap();
            continue_rx.recv().unwrap();
            with(list as *mut c_void, |state| state.pso)
        });
        ready_rx.recv_timeout(Duration::from_secs(1)).expect("cache worker did not warm its entry");
        reset(list as *mut c_void, 0, 0x7E57_D242);
        continue_tx.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), 0x7E57_D242);
        forget(list as *mut c_void);
    }

    /// A pipeline with the rect list GS draws its rect lists whole: draw_rects leaves them alone.
    #[test]
    fn rect_lists_with_the_rect_list_gs_are_not_expanded_on_the_cpu() {
        let (with_gs, without) = (0x7E57_4000usize, 0x7E57_5000usize);
        real::register_rect_list_gs_for_test(with_gs);
        assert!(!cpu_rect_expansion(with_gs));
        assert!(cpu_rect_expansion(without) && cpu_rect_expansion(0));
        real::forget_pso_for_test(with_gs);
        assert!(cpu_rect_expansion(with_gs));
    }

    /// An ExecuteIndirect whose bound pipeline is still being built is skipped (its stand-in
    /// with XCL's ABI root signature on the list is the debug layer's 953 and a hung GPU);
    /// a built or failed one goes to the PC call. The gate `execute_indirect` uses.
    #[test]
    fn execute_indirect_with_a_pipeline_still_building_is_skipped() {
        let (building, built, failed) = (0x7E57_6001usize, 0x7E57_6002usize, 0x7E57_6003usize);
        // lazy submits: no worker starts them, so the states stay under the test's control
        pso_async::submit_lazy(building, false, Box::new(|| 0x1234), 0);
        pso_async::submit_lazy(built, false, Box::new(|| 0x1235), 0);
        pso_async::submit_lazy(failed, false, Box::new(|| 0), 0);
        assert!(pso_async::is_pending(building));
        assert_eq!(pso_async::resolve_wait(built), 0x1235);
        assert_eq!(pso_async::resolve_wait(failed), failed);
        assert!(!pso_async::is_pending(built) && !pso_async::is_pending(failed));
        // never resolved: still pending (the skip case), then forgotten
        assert!(pso_async::is_pending(building));
        pso_async::released(building);
        pso_async::released(built);
        pso_async::released(failed);
        assert!(!pso_async::is_pending(building));
    }

    /// A dispatch writes an ExecuteIndirect's group count through another buffer over the same
    /// memory (as translated shaders write through the memory windows), a UAV barrier (what a
    /// console barrier becomes), then the ExecuteIndirect: it runs the count just written, not
    /// the one the memory held before the list (`fresh_indirect_arguments`; without it an RTX
    /// 5090 ran the old count in nearly every try).
    #[test]
    fn execute_indirect_reads_the_arguments_written_before_it() {
        use crate::gpu::{fence, recomp::runner::skip_test};
        use std::mem::{transmute_copy, ManuallyDrop};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, IDXGIAdapter1, IDXGIFactory6, DXGI_CREATE_FACTORY_FLAGS};

        const TRIES: u32 = 40;
        // mode 0: write (x, 1, 1) at a[0..12]; mode 1: count the groups
        let src = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint mode; uint x; };
RWByteAddressBuffer a : register(u0);
RWByteAddressBuffer c : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint t : SV_GroupIndex) {
    if (t != 0u) return;
    if (mode == 0u) a.Store3(0, uint3(x, 1u, 1u)); else c.InterlockedAdd(0, 1u);
}
"#;
        if crate::gpu::recomp::dxc::location().is_err() {
            skip_test("no DXC");
            return;
        }
        unsafe {
            let setup = || -> Result<(ID3D12Device, String), String> {
                let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| format!("DXGI factory unavailable: {e}"))?;
                let sel = crate::gpu::adapter::Selector::from_env_or("auto");
                let (adapter, name): (IDXGIAdapter1, String) = if sel == crate::gpu::adapter::Selector::Warp {
                    (factory.EnumWarpAdapter().map_err(|e| format!("WARP unavailable: {e}"))?, "WARP (software renderer)".into())
                } else {
                    let (adapter, desc) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next()
                        .ok_or_else(|| crate::gpu::adapter::no_match(&factory, &sel))?;
                    (adapter, crate::gpu::adapter::name(&desc))
                };
                let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_11_0).map_err(|e| format!("{name}: D3D12 device unavailable ({e})"))?;
                let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_6 };
                dev.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as _, std::mem::size_of_val(&sm) as u32)
                    .map_err(|e| format!("{name}: cannot query SM 6.6 ({e})"))?;
                if sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
                    return Err(crate::gpu::agility::format_no_sm66_message(&name, crate::gpu::agility::state()));
                }
                Ok((dev, name))
            };
            let (dev, name) = match setup() {
                Ok(selected) => selected,
                Err(reason) => { skip_test(&reason); return; }
            };
            println!("ExecuteIndirect adapter: {name}");
            let dxil = crate::gpu::recomp::dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let cs = D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: cs, ..Default::default() }).unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut sig).unwrap();
            let sig = sig.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = |width| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            // the arguments and the buffer the dispatch writes them through: one memory
            let mut heap: Option<ID3D12Heap> = None;
            let heap_desc = D3D12_HEAP_DESC { SizeInBytes: 1 << 16, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
            dev.CreateHeap(&heap_desc, &mut heap).unwrap();
            let heap = heap.unwrap();
            let placed = || {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreatePlacedResource(&heap, 0, &desc(1 << 16), D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (args, written) = (placed(), placed());
            // the group count, read on the CPU
            let cpu_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            let mut groups: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &desc(256), D3D12_RESOURCE_STATE_COMMON, None, &mut groups).unwrap();
            let groups = groups.unwrap();
            let mut count = std::ptr::null_mut();
            groups.Map(0, None, Some(&mut count)).unwrap();
            let count = count as *mut u32;
            let uav_barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
            };
            let run = |list: &ID3D12GraphicsCommandList, close: &dyn Fn()| {
                close();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let v = done.GetCompletedValue() + 1;
                queue.Signal(&done, v).unwrap();
                fence::wait(&done, v, &dev, None, 10_000).unwrap();
            };
            let (mut old, mut wrong) = (0u32, Vec::new());
            for k in 0..TRIES {
                // each try writes a count the memory does not hold yet
                let new = if k % 2 == 0 { 4096 } else { 0 };
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let record = |x: u32| -> ID3D12GraphicsCommandList {
                    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
                    list.SetComputeRootSignature(&rs);
                    list.SetComputeRootUnorderedAccessView(1, written.GetGPUVirtualAddress());
                    list.SetComputeRootUnorderedAccessView(2, groups.GetGPUVirtualAddress());
                    list.SetComputeRoot32BitConstants(0, 2, [0u32, x].as_ptr() as _, 0);
                    list.Dispatch(1, 1, 1);
                    list
                };
                let before = record(old);
                run(&before, &|| before.Close().unwrap());
                alloc.Reset().unwrap();
                *count = 0;
                let list = record(new);
                list.ResourceBarrier(&[uav_barrier.clone()]);
                list.SetComputeRoot32BitConstants(0, 2, [1u32, 0].as_ptr() as _, 0);
                // from here as XCL records a title's ExecuteIndirect: on the adopted list
                let raw = list.as_raw();
                real::adopt(raw, real::Kind::CommandList);
                list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
                type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
                type Close = unsafe extern "system" fn(*mut c_void) -> windows::core::HRESULT;
                let ei: Ei = std::mem::transmute(real::orig_fn(raw, 59));
                fresh_indirect_arguments(raw, || ei(raw, sig.as_raw() as usize, 1, args.as_raw() as usize, 0, 0, 0));
                let close: Close = std::mem::transmute(real::orig_fn(raw, 9));
                run(&list, &|| close(raw).unwrap());
                let ran = std::ptr::read_volatile(count);
                if ran != new {
                    wrong.push((k, ran));
                }
                old = new;
            }
            groups.Unmap(0, None);
            assert!(wrong.is_empty(), "{} of {TRIES} ExecuteIndirects ran another group count than the one written before them (try, groups): {wrong:?}", wrong.len());
        }
    }

    /// Minimal RTS0 v2 root signature blob with one 4-constant parameter.
    fn test_rts0_blob() -> Vec<u8> {
        let words: [u32; 12] = [
            2, 1, 24, 0, 0, 0, // version=2, num_params=1, params_offset=24
            1, 0, 36,          // param 0: type=Constants(1), visibility=0, payload_offset=36
            0, 0, 4,           // shader_reg=0, reg_space=0, num_32bit_values=4
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn reset_and_clear_state_track_initial_pso_and_clear_root_args() {
        let list = 0x7100_0001usize as *mut c_void;
        let rs_compute = 0x7100_0010usize;
        let rs_gfx = 0x7100_0020usize;
        let alloc1 = 0x7100_0100usize;
        let alloc2 = 0x7100_0200usize;
        let pso_init = 0x7100_1000usize;
        let pso_mid = 0x7100_2000usize;
        let pso_cleared = 0x7100_3000usize;
        let pso_reset = 0x7100_4000usize;

        let blob = test_rts0_blob();
        console_list::note_root_signature(rs_compute, &blob);
        console_list::note_root_signature(rs_gfx, &blob);

        // Initial CreateCommandList / reset seeds PSO and allocator with clean root state.
        reset(list, alloc1, pso_init);
        with(list, |s| {
            assert_eq!(s.pso, pso_init);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Populate compute + graphics root signatures, constants, and a different PSO.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0xDEAD_BEEF, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0xCAFE_BABE, 0);
        }
        with(list, |s| {
            s.pso = pso_mid;
            assert!(s.layout.is_some() && !s.args.is_empty());
            assert!(s.layout_gfx.is_some() && !s.args_gfx.is_empty());
        });

        // ClearState(pso_cleared) updates PSO and clears root signatures/args while keeping allocator.
        clear_state(list, pso_cleared);
        with(list, |s| {
            assert_eq!(s.pso, pso_cleared);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // ClearState(null) clears PSO to 0.
        clear_state(list, 0);
        assert_eq!(tracked_pso(list), 0);

        // Re-populate root args, then Reset(alloc2, pso_reset) switches allocator/PSO and clears root args.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0x1234_5678, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0x8765_4321, 0);
        }
        reset(list, alloc2, pso_reset);
        with(list, |s| {
            assert_eq!(s.pso, pso_reset);
            assert_eq!(s.allocator, alloc2);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Reset with null initial PSO clears stale PSO back to 0.
        reset(list, alloc2, 0);
        assert_eq!(tracked_pso(list), 0);

        forget(list);
    }

    fn probed_resolution(
        list: usize,
        original: usize,
        entry: std::sync::Weak<BindEntry>,
        dropped: Arc<AtomicBool>,
    )
        -> pso_async::PipelineResolution {
        pso_async::pipeline_resolution_for_test(original, move || {
            assert!(state_locks_available_for_test(list), "pipeline resolution dropped under bind-state locks");
            let entry = entry.upgrade().expect("old bind entry disappeared before its resolution");
            assert!(entry.state.try_lock().is_ok(), "pipeline resolution dropped under its old entry lock");
            dropped.store(true, Ordering::Release);
        })
    }

    #[test]
    fn pipeline_changes_retain_the_pending_they_classified() {
        let list = 0x7100_4800usize as *mut c_void;
        let psos = [0x7100_4810usize, 0x7100_4820, 0x7100_4830];
        for pso in psos {
            pso_async::submit_lazy(pso, false, Box::new(|| {
                panic!("bind-time classification ran pipeline work")
            }), 0);
        }
        let identity = |pso| with(list, |s| {
            assert_eq!(s.pso, pso);
            s.resolution.as_ref().expect("registered pipeline was not classified").identity_for_test()
        });

        set_tracked_pipeline_state(list, psos[0]);
        assert_eq!(identity(psos[0]), pso_async::pending_identity_for_test(psos[0]).unwrap());
        clear_state(list, psos[1]);
        assert_eq!(identity(psos[1]), pso_async::pending_identity_for_test(psos[1]).unwrap());
        reset(list, 0, psos[2]);
        assert_eq!(identity(psos[2]), pso_async::pending_identity_for_test(psos[2]).unwrap());

        set_tracked_pipeline_state(list, 0x7100_4840);
        with(list, |s| assert!(s.resolution.is_none(), "unregistered pipeline got a decision"));
        clear_state(list, 0);
        with(list, |s| assert!(s.resolution.is_none(), "null pipeline got a decision"));

        forget(list);
        for pso in psos {
            assert_eq!(pso_async::released(pso), 0);
        }
    }

    #[test]
    fn pipeline_bind_retains_pending_across_real_address_reuse() {
        let old_list = 0x7100_4850usize as *mut c_void;
        let new_list = 0x7100_4860usize as *mut c_void;
        let pso = 0x7100_4870usize;
        pso_async::submit_lazy(pso, false, Box::new(|| {
            panic!("old retained pipeline work ran")
        }), 0);
        set_tracked_pipeline_state(old_list, pso);
        let old_identity = with(old_list, |s| {
            s.resolution.as_ref().unwrap().identity_for_test()
        });
        assert_eq!(Some(old_identity), pso_async::pending_identity_for_test(pso));

        assert_eq!(pso_async::released(pso), 0);
        assert!(pso_async::pending_identity_for_test(pso).is_none());
        pso_async::submit_lazy(pso, false, Box::new(|| {
            panic!("replacement pipeline work ran")
        }), 0);
        let replacement_identity = pso_async::pending_identity_for_test(pso).unwrap();
        assert_ne!(replacement_identity.1, old_identity.1);
        set_tracked_pipeline_state(new_list, pso);

        assert_eq!(with(old_list, |s| s.resolution.as_ref().unwrap().identity_for_test()), old_identity);
        assert_eq!(with(new_list, |s| s.resolution.as_ref().unwrap().identity_for_test()), replacement_identity);
        forget(old_list);
        forget(new_list);
        assert_eq!(pso_async::released(pso), 0);
    }

    #[repr(C)]
    struct ResolutionPso {
        vtable: *const usize,
        refs: AtomicUsize,
    }

    unsafe extern "system" fn resolution_add_ref(this: *mut c_void) -> u32 {
        (*(this as *mut ResolutionPso)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn resolution_release(this: *mut c_void) -> u32 {
        (*(this as *mut ResolutionPso)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    #[test]
    fn graphics_resolution_uses_retained_identity_and_records_its_pin() {
        let vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut old = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let old_raw = &mut old as *mut ResolutionPso as usize;
        let replacement_raw = &mut replacement as *mut ResolutionPso as usize;
        let standin = 0x7100_4880usize;
        let list = 0x7100_4890usize;
        pso_async::submit_lazy(standin, false, Box::new(move || old_raw), 0);
        assert_eq!(pso_async::resolve_wait(standin), old_raw);
        let retained = pso_async::begin_pipeline_resolution(standin).unwrap();
        assert_eq!(pso_async::released(standin), old_raw);
        pso_async::submit_lazy(standin, false, Box::new(move || replacement_raw), 0);

        assert_eq!(resolve_pipeline(list, standin, Some(retained)), Some(old_raw));
        assert_eq!(list_x::native_generation_count_for_test(list), 1);
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 1, "the reused registry address must not be resolved");

        list_x::clear_segment_state(list);
        assert_eq!(pso_async::released(standin), 0);
        assert_eq!(unsafe { resolution_release(old_raw as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(replacement_raw as *mut c_void) }, 0);
    }

    #[test]
    fn graphics_entrypoints_pass_the_retained_raw_to_the_abi() {
        let _stop_abi = STOP_ABI_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[25] = fake_set_pso as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) }, refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0), set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0), set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0), execute_indirect_calls: AtomicUsize::new(0),
        });
        let list = &*fake as *const FakeAdoptedList as *mut c_void;
        let _class = real::test_command_list_class(
            orig_vt.as_ptr() as usize,
            unsafe { ext_vt.as_ptr().add(1) } as usize,
        );
        let vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut old = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut selected_alias = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let old_raw = &mut old as *mut ResolutionPso as usize;
        let replacement_raw = &mut replacement as *mut ResolutionPso as usize;
        let selected_alias_raw = &mut selected_alias as *mut ResolutionPso as usize;
        let standin = 0x7100_4898usize;
        pso_async::submit_lazy(standin, false, Box::new(move || old_raw), 0);
        assert_eq!(pso_async::resolve_wait(standin), old_raw);
        real::register_translated_graphics_for_test(old_raw);
        unsafe { list_x::on_create(list, 0, standin, D3D12_COMMAND_LIST_TYPE_DIRECT) };
        set_tracked_pipeline_state(list, standin);
        assert_eq!(pso_async::released(standin), old_raw);
        pso_async::submit_lazy(standin, false, Box::new(move || replacement_raw), 0);
        let alias_calls = Arc::new(AtomicUsize::new(0));
        let selected_calls = alias_calls.clone();
        pso_async::submit_lazy(old_raw, false, Box::new(move || {
            selected_calls.fetch_add(1, Ordering::Relaxed);
            selected_alias_raw
        }), 0);
        STOP_ABI_AFTER_BIND_LIST.store(list as usize, Ordering::Release);

        LAST_ABI_PSO.store(0, Ordering::Release);
        assert!(unsafe { graphics_bound(list) }.is_none());
        assert_eq!(LAST_ABI_PSO.load(Ordering::Acquire), old_raw);
        LAST_ABI_PSO.store(0, Ordering::Release);
        assert!(unsafe { prepare_indexed_draw(list, 3, 1, 0, 0) }.is_none());
        assert_eq!(LAST_ABI_PSO.load(Ordering::Acquire), old_raw,
            "indexed draw must reuse the retained generation, not the title or replacement address");
        assert_eq!(fake.last_pso.load(Ordering::Acquire), old_raw);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 2);
        assert_eq!(alias_calls.load(Ordering::Relaxed), 0, "graphics ABI bind ran selected raw's alias build");
        assert_eq!(list_x::bound_pipeline_for_test(list as usize), standin,
            "graphics ABI bind replaced title-visible pipeline state");

        STOP_ABI_AFTER_BIND_LIST.store(0, Ordering::Release);
        forget(list);
        list_x::clear_segment_state(list as usize);
        list_x::on_release(list as usize);
        assert_eq!(pso_async::released(standin), 0);
        assert_eq!(pso_async::released(old_raw), 0);
        real::forget_pso_for_test(old_raw);
        assert_eq!(unsafe { resolution_release(old_raw as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(replacement_raw as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(selected_alias_raw as *mut c_void) }, 0);
    }

    #[test]
    fn dispatch_entrypoint_uses_the_retained_generation_after_address_reuse() {
        let _stop_abi = STOP_ABI_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let list_object = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let list = &list_object as *const ResolutionPso as *mut c_void;
        let mut original = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let original_raw = &mut original as *mut ResolutionPso as usize;
        let replacement_raw = &mut replacement as *mut ResolutionPso as usize;
        let standin = 0x7100_489Cusize;

        pso_async::submit_lazy(standin, true, Box::new(move || original_raw), 0);
        assert_eq!(pso_async::resolve_wait(standin), original_raw);
        real::register_translated_compute_for_test(original_raw);
        set_tracked_pipeline_state(list, standin);
        let original_owner = pso_async::released(standin);
        assert_eq!(original_owner, original_raw);

        let replacement_calls = Arc::new(AtomicUsize::new(0));
        let replacement_build_calls = replacement_calls.clone();
        pso_async::submit_lazy(standin, true, Box::new(move || {
            replacement_build_calls.fetch_add(1, Ordering::Relaxed);
            replacement_raw
        }), 0);
        STOP_ABI_LIST.store(list as usize, Ordering::Release);
        for _ in 0..2 {
            LAST_ABI_PSO.store(0, Ordering::Release);
            unsafe { dispatch_flags(list, 1, 1, 1, 0) };
            assert_eq!(LAST_ABI_PSO.load(Ordering::Acquire), original_raw);
        }
        assert_eq!(replacement_calls.load(Ordering::Relaxed), 0, "Dispatch ran the replacement build");
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 1, "Dispatch retained the replacement generation");
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 1, "repeated Dispatch duplicated its exact pin");

        STOP_ABI_LIST.store(0, Ordering::Release);
        forget(list);
        list_x::clear_segment_state(list as usize);
        assert_eq!(pso_async::released(standin), 0);
        real::forget_pso_for_test(original_raw);
        assert_eq!(unsafe { resolution_release(original_owner as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(replacement_raw as *mut c_void) }, 0);
    }

    #[test]
    fn execute_indirect_uses_retained_generation_after_address_reuse() {
        let _stop_abi = STOP_ABI_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[59] = fake_execute_indirect as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) }, refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0), set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0), set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0), execute_indirect_calls: AtomicUsize::new(0),
        });
        let list = &*fake as *const FakeAdoptedList as *mut c_void;
        let _class = real::test_command_list_class(
            orig_vt.as_ptr() as usize,
            unsafe { ext_vt.as_ptr().add(1) } as usize,
        );
        let pso_vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut original = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut selected_alias = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let original_raw = &mut original as *mut ResolutionPso as usize;
        let replacement_raw = &mut replacement as *mut ResolutionPso as usize;
        let selected_alias_raw = &mut selected_alias as *mut ResolutionPso as usize;
        let standin = 0x7100_48B0usize;
        let signature = IndirectSignatureInfo::new(true, false, None);

        pso_async::submit_lazy(standin, true, Box::new(move || original_raw), 0);
        assert_eq!(pso_async::resolve_wait(standin), original_raw);
        real::register_translated_compute_for_test(original_raw);
        unsafe { list_x::on_create(list, 0, standin, D3D12_COMMAND_LIST_TYPE_DIRECT) };
        set_tracked_pipeline_state(list, standin);
        let original_owner = pso_async::released(standin);
        let replacement_calls = Arc::new(AtomicUsize::new(0));
        let calls = replacement_calls.clone();
        pso_async::submit_lazy(standin, true, Box::new(move || {
            calls.fetch_add(1, Ordering::Relaxed);
            replacement_raw
        }), 0);
        let alias_calls = Arc::new(AtomicUsize::new(0));
        let selected_calls = alias_calls.clone();
        pso_async::submit_lazy(original_raw, true, Box::new(move || {
            selected_calls.fetch_add(1, Ordering::Relaxed);
            selected_alias_raw
        }), 0);
        STOP_ABI_AFTER_BIND_LIST.store(list as usize, Ordering::Release);
        for _ in 0..2 {
            LAST_ABI_PSO.store(0, Ordering::Release);
            unsafe { execute_indirect_classified(list, 1, 1, 0, 0, 0, 0, signature) };
            assert_eq!(LAST_ABI_PSO.load(Ordering::Acquire), original_raw);
            assert_eq!(fake.last_pso.load(Ordering::Acquire), original_raw,
                "selected raw was redirected through its registry alias at native bind");
        }
        assert_eq!(replacement_calls.load(Ordering::Relaxed), 0, "ExecuteIndirect ran a replacement build at the reused title address");
        assert_eq!(alias_calls.load(Ordering::Relaxed), 0, "ExecuteIndirect ran the selected raw's alias build");
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 2);
        assert_eq!(list_x::bound_pipeline_for_test(list as usize), standin,
            "exact internal bind replaced title-visible pipeline state");
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 1, "repeated ExecuteIndirect duplicated its exact pin");

        STOP_ABI_AFTER_BIND_LIST.store(0, Ordering::Release);
        forget(list);
        list_x::clear_segment_state(list as usize);
        list_x::on_release(list as usize);
        assert_eq!(pso_async::released(standin), 0);
        assert_eq!(pso_async::released(original_raw), 0);
        real::forget_pso_for_test(original_raw);
        assert_eq!(unsafe { resolution_release(original_owner as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(replacement_raw as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(selected_alias_raw as *mut c_void) }, 0);
    }

    #[test]
    fn execute_indirect_native_snapshot_ignores_a_later_registry_alias() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[59] = fake_execute_indirect as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) }, refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0), set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0), set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0), execute_indirect_calls: AtomicUsize::new(0),
        });
        let list = &*fake as *const FakeAdoptedList as *mut c_void;
        let native = 0x7100_48B2usize;
        reset(list, 0, native);
        let build_calls = Arc::new(AtomicUsize::new(0));
        let calls = build_calls.clone();
        pso_async::submit_lazy(native, true, Box::new(move || {
            calls.fetch_add(1, Ordering::Relaxed);
            0x7100_48B3
        }), 0);

        unsafe { execute_indirect_classified(list, 1, 1, 0, 0, 0, 0, IndirectSignatureInfo::default()) };
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 1);
        assert_eq!(build_calls.load(Ordering::Relaxed), 0, "native ExecuteIndirect re-resolved its raw address through the registry");
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 0);

        forget(list);
        let _ = pso_async::released(native);
    }

    #[test]
    fn deferred_execute_indirect_replays_its_inherited_resolution_once() {
        let _stop_abi = STOP_ABI_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[59] = fake_execute_indirect as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let make_list = || Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) }, refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0), set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0), set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0), execute_indirect_calls: AtomicUsize::new(0),
        });
        let source_object = make_list();
        let replay_object = make_list();
        let source = &*source_object as *const FakeAdoptedList as *mut c_void;
        let replay = &*replay_object as *const FakeAdoptedList as *mut c_void;
        let pso_vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut old = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let old_raw = &mut old as *mut ResolutionPso as usize;
        let replacement_raw = &mut replacement as *mut ResolutionPso as usize;
        let standin = 0x7100_48B4usize;

        pso_async::submit_lazy(standin, true, Box::new(move || old_raw), 0);
        assert_eq!(pso_async::resolve_wait(standin), old_raw);
        real::register_translated_compute_for_test(old_raw);
        set_tracked_pipeline_state(source, standin);
        let (inherited_pso, inherited_resolution) = with(source, |state| {
            (state.pso, state.resolution.clone())
        });
        with(replay, |state| {
            state.pso = inherited_pso;
            state.resolution = inherited_resolution;
        });

        let old_owner = pso_async::released(standin);
        pso_async::submit_lazy(standin, true, Box::new(move || replacement_raw), 0);
        set_tracked_pipeline_state(source, standin);
        STOP_ABI_LIST.store(replay as usize, Ordering::Release);
        LAST_ABI_PSO.store(0, Ordering::Release);
        unsafe {
            execute_indirect_retained(
                replay, 1, 1, 0, 0, 0, 0, IndirectSignatureInfo::new(true, false, None),
            )
        };
        assert_eq!(LAST_ABI_PSO.load(Ordering::Acquire), old_raw);
        assert_eq!(list_x::native_generation_count_for_test(replay as usize), 1);
        assert_eq!(list_x::native_generation_count_for_test(source as usize), 0,
            "deferred replay recursively split or pinned the source segment");
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 1,
            "deferred replay resolved the replacement generation");

        STOP_ABI_LIST.store(0, Ordering::Release);
        forget(source);
        forget(replay);
        list_x::clear_segment_state(replay as usize);
        assert_eq!(pso_async::released(standin), 0);
        real::forget_pso_for_test(old_raw);
        assert_eq!(unsafe { resolution_release(old_owner as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(replacement_raw as *mut c_void) }, 0);
    }

    #[test]
    fn unavailable_execute_indirect_counts_one_signature_selected_skip() {
        let _stop_abi = STOP_ABI_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[59] = fake_execute_indirect as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) }, refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0), set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0), set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0), execute_indirect_calls: AtomicUsize::new(0),
        });
        let list = &*fake as *const FakeAdoptedList as *mut c_void;
        STOP_ABI_LIST.store(list as usize, Ordering::Release);
        LAST_ABI_PSO.store(0, Ordering::Release);
        let _ = take_indirect_skip_counts();

        let compute = pso_async::pipeline_resolution_for_compute_test(0x7100_48B6, || {});
        with(list, |state| {
            state.pso = 0x7100_48B6;
            state.resolution = Some(compute.unavailable_for_test());
        });
        unsafe {
            execute_indirect_retained(
                list, 1, 1, 0, 0, 0, 0, IndirectSignatureInfo::new(true, false, None),
            )
        };
        assert_eq!(take_indirect_skip_counts(), [0, 1]);

        let graphics = pso_async::pipeline_resolution_for_test(0x7100_48B7, || {});
        with(list, |state| {
            state.pso = 0x7100_48B7;
            state.resolution = Some(graphics.unavailable_for_test());
        });
        unsafe {
            execute_indirect_retained(list, 1, 1, 0, 0, 0, 0, IndirectSignatureInfo::default())
        };
        assert_eq!(take_indirect_skip_counts(), [1, 0]);
        assert_eq!(LAST_ABI_PSO.load(Ordering::Acquire), 0, "unavailable resolution reached ABI binding");
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0, "unavailable resolution reached PC ExecuteIndirect");
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 0);

        STOP_ABI_LIST.store(0, Ordering::Release);
        forget(list);
    }

    #[test]
    fn dispatch_waits_outside_states_and_selects_the_build_once() {
        let _stop_abi = STOP_ABI_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let list_object = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let list = &list_object as *const ResolutionPso as *mut c_void;
        let mut native = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut ResolutionPso as usize;
        let standin = 0x7100_489Dusize;
        let list_key = list as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        pso_async::submit_lazy(standin, true, Box::new(move || {
            assert!(state_locks_available_for_test(list_key), "Dispatch waited for its build under bind-state locks");
            build_calls.fetch_add(1, Ordering::Relaxed);
            real::register_translated_compute_for_test(raw);
            raw
        }), 0);
        set_tracked_pipeline_state(list, standin);
        STOP_ABI_LIST.store(list as usize, Ordering::Release);

        LAST_ABI_PSO.store(0, Ordering::Release);
        unsafe { dispatch_flags(list, 1, 1, 1, 0) };
        assert_eq!(LAST_ABI_PSO.load(Ordering::Acquire), raw);
        LAST_ABI_PSO.store(0, Ordering::Release);
        unsafe { dispatch_flags(list, 1, 1, 1, 0) };
        assert_eq!(LAST_ABI_PSO.load(Ordering::Acquire), raw);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "repeated Dispatch consumed the build more than once");
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 1);

        STOP_ABI_LIST.store(0, Ordering::Release);
        forget(list);
        list_x::clear_segment_state(list as usize);
        let raw_owner = pso_async::released(standin);
        real::forget_pso_for_test(raw);
        assert_eq!(unsafe { resolution_release(raw_owner as *mut c_void) }, 0);
    }

    #[test]
    fn exact_compute_bind_bypasses_a_registry_alias_and_preserves_title_state() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[25] = fake_set_pso as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list = &*fake as *const FakeAdoptedList as *mut c_void;
        let _class = real::test_command_list_class(
            orig_vt.as_ptr() as usize,
            unsafe { ext_vt.as_ptr().add(1) } as usize,
        );
        let original = 0x7100_489Eusize;
        let exact_raw = 0x7100_489Fusize;
        let replacement = 0x7100_48A1usize;
        unsafe { list_x::on_set_pipeline_state(list, original) };
        assert_eq!(list_x::bound_pipeline_for_test(list as usize), original);
        pso_async::submit_lazy(exact_raw, false, Box::new(move || replacement), 0);
        assert_eq!(pso_async::resolve_wait(exact_raw), replacement);

        unsafe { real::bind_pipeline_exact(list, original, exact_raw) };
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 1);
        assert_eq!(fake.last_pso.load(Ordering::Acquire), exact_raw, "exact raw was re-resolved through its alias");
        assert_eq!(list_x::bound_pipeline_for_test(list as usize), original, "internal compute bind replaced title-visible state");
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 0, "internal exact bind pinned the alias generation");

        assert_eq!(pso_async::released(exact_raw), replacement);
        list_x::on_release(list as usize);
    }

    #[test]
    fn graphics_resolution_waits_outside_states_and_selects_only_once() {
        let vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut real = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut real as *mut ResolutionPso as usize;
        let standin = 0x7100_48A0usize;
        let list = 0x7100_48B0usize as *mut c_void;
        let list_key = list as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        pso_async::submit_recipe_keyed(standin, false, pso_async::Recipe::new(move || {
            assert!(state_locks_available_for_test(list_key), "pipeline build waited under bind-state locks");
            build_calls.fetch_add(1, Ordering::Relaxed);
            raw
        }), false, 0, None);
        set_tracked_pipeline_state(list, standin);

        assert_eq!(resolve_bound_graphics_pipeline(list, list as usize), Some((standin, raw)));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(cpu_rect_expansion(raw));
        assert_eq!(calls.load(Ordering::Relaxed), 1, "downstream draw decisions must reuse the selected raw PSO");

        forget(list);
        list_x::clear_segment_state(list as usize);
        assert_eq!(pso_async::released(standin), raw);
        assert_eq!(unsafe { resolution_release(raw as *mut c_void) }, 0);
    }

    #[test]
    fn graphics_resolution_pins_one_shot_and_fails_closed_when_unavailable() {
        let one_shot = 0x7100_48C0usize;
        let vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut real = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let one_shot_real = &mut real as *mut ResolutionPso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        pso_async::submit_lazy(one_shot, false, Box::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            one_shot_real
        }), 0);
        let retained = pso_async::begin_pipeline_resolution(one_shot).unwrap();
        let list = 0x7100_48E0;
        assert_eq!(resolve_pipeline(list, one_shot, Some(retained.clone())), Some(one_shot_real));
        assert_eq!(resolve_pipeline(list, one_shot, Some(retained)), Some(one_shot_real));
        assert_eq!(calls.load(Ordering::Relaxed), 1, "one-shot build must be consumed once");
        assert_eq!(list_x::native_generation_count_for_test(list), 1);
        assert_eq!(real.refs.load(Ordering::Relaxed), 2, "repeated draws retained more than one COM owner");
        assert_eq!(pso_async::released(one_shot), one_shot_real);
        list_x::clear_segment_state(list);
        assert_eq!(unsafe { resolution_release(one_shot_real as *mut c_void) }, 0);

        let failed = 0x7100_48E8usize;
        pso_async::submit_lazy(failed, false, Box::new(|| 0), 0);
        let retained = pso_async::begin_pipeline_resolution(failed).unwrap();
        assert_eq!(resolve_pipeline(0x7100_48E9, failed, Some(retained)), None);
        assert_eq!(pso_async::released(failed), 0);

        let unavailable = 0x7100_48F0usize;
        pso_async::submit_recipe_keyed(unavailable, false, pso_async::Recipe::new(|| panic!("orphan recipe ran")), false, 0, None);
        let retained = pso_async::begin_pipeline_resolution(unavailable).unwrap();
        assert_eq!(pso_async::released(unavailable), 0);
        assert_eq!(resolve_pipeline(0x7100_4900, unavailable, Some(retained)), None);

        let replacement = 0x7100_4910usize;
        pso_async::submit_lazy(replacement, false, Box::new(|| panic!("unregistered bind adopted later registry entry")), 0);
        assert_eq!(resolve_pipeline(0, replacement, None), Some(replacement));
        assert_eq!(pso_async::released(replacement), 0);
    }

    #[test]
    fn pipeline_resolution_replacements_drop_outside_state_lock() {
        let cases: [fn(*mut c_void); 4] = [
            |list| clear_state(list, 0x7100_5001),
            |list| reset(list, 0, 0x7100_5002),
            |list| set_tracked_pipeline_state(list, 0x7100_5003),
            forget,
        ];
        for (n, replace) in cases.into_iter().enumerate() {
            let list = (0x7100_5100usize + n) as *mut c_void;
            let dropped = Arc::new(AtomicBool::new(false));
            with(list, |_| {});
            let old_entry = STATES.get(list as usize).unwrap();
            let resolution = probed_resolution(
                list as usize,
                0x7100_5200 + n,
                Arc::downgrade(&old_entry),
                dropped.clone(),
            );
            with(list, |s| s.resolution = Some(resolution));
            replace(list);
            assert!(dropped.load(Ordering::Acquire), "case {n} retained the replaced resolution");
            drop(old_entry);
            if n + 1 != cases.len() {
                forget(list);
            }
        }
    }

    #[test]
    fn split_retains_the_exact_resolution_across_address_reuse() {
        let from = 0x7100_6001usize as *mut c_void;
        let to = 0x7100_6002usize as *mut c_void;
        let original = 0x7100_6100usize;
        let old = pso_async::pipeline_resolution_for_test(original, || {});
        let old_identity = old.identity_for_test();
        with(from, |s| {
            s.pso = original;
            s.resolution = Some(old);
        });
        reset(to, 0, 0);

        let prepared = prepare_split(from, to, 0);
        let replacement = pso_async::pipeline_resolution_for_test(original, || {});
        let replacement_identity = replacement.identity_for_test();
        assert_ne!(old_identity.1, replacement_identity.1);
        with(from, |s| s.resolution = Some(replacement));
        install_prepared_split(prepared);

        assert_eq!(with(to, |s| s.resolution.as_ref().unwrap().identity_for_test()), old_identity,
            "split state was redirected to the Pending now stored at the reused title address");
        assert_eq!(with(from, |s| s.resolution.as_ref().unwrap().identity_for_test()), replacement_identity);
        forget(from);
        forget(to);
    }

    #[test]
    fn split_reset_uses_only_an_owned_ready_generation() {
        let vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut native = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut ResolutionPso as usize;
        let standin = 0x7100_6200usize;
        let from = 0x7100_6201usize as *mut c_void;
        let to = 0x7100_6202usize as *mut c_void;
        pso_async::submit_lazy(standin, false, Box::new(move || raw), 0);
        assert_eq!(pso_async::resolve_wait(standin), raw);
        set_tracked_pipeline_state(from, standin);
        reset(to, 0, 0);
        let prepared = prepare_split(from, to, 0);
        let (initial, pin) = prepared.initial_pipeline();
        assert_eq!(initial, raw);
        assert!(pin.is_some(), "ready split initial pipeline has no exact owner");
        drop(pin); drop(prepared); forget(from); forget(to);
        let raw_owner = pso_async::released(standin);
        assert_eq!(raw_owner, raw);
        assert_eq!(unsafe { resolution_release(raw_owner as *mut c_void) }, 0);

        let pending = 0x7100_6210usize;
        let from = 0x7100_6211usize as *mut c_void;
        let to = 0x7100_6212usize as *mut c_void;
        pso_async::submit_lazy(pending, false, Box::new(|| panic!("split Reset started pending build")), 0);
        set_tracked_pipeline_state(from, pending);
        reset(to, 0, 0);
        let prepared = prepare_split(from, to, 0);
        assert!(matches!(prepared.initial_pipeline(), (0, None)), "pending split consulted its reusable address");
        drop(prepared); forget(from); forget(to);
        assert_eq!(pso_async::released(pending), 0);
    }

    #[test]
    fn kickoff_reset_uses_the_retained_generation_after_address_reuse() {
        let vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut original = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = ResolutionPso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let original_raw = &mut original as *mut ResolutionPso as usize;
        let replacement_raw = &mut replacement as *mut ResolutionPso as usize;
        let standin = 0x7100_6220usize;
        let list = 0x7100_6221usize as *mut c_void;

        pso_async::submit_lazy(standin, false, Box::new(move || original_raw), 0);
        assert_eq!(pso_async::resolve_wait(standin), original_raw);
        set_tracked_pipeline_state(list, standin);
        let original_owner = pso_async::released(standin);
        assert_eq!(original_owner, original_raw);

        pso_async::submit_lazy(standin, false, Box::new(move || replacement_raw), 0);
        assert_eq!(pso_async::resolve_wait(standin), replacement_raw);
        let (initial, pin) = initial_pipeline(list);
        assert_eq!(initial, original_raw, "Kickoff Reset followed the reused title address");
        assert!(pin.is_some(), "Kickoff Reset did not own the retained generation");
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 1, "Kickoff Reset retained the replacement generation");

        let replacement_owner = pso_async::released(standin);
        assert_eq!(replacement_owner, replacement_raw);
        drop(pin);
        forget(list);
        assert_eq!(unsafe { resolution_release(original_owner as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(replacement_owner as *mut c_void) }, 0);
    }

    #[test]
    fn split_preflight_transfers_destination_uma_before_install() {
        let from = 0x7101_0001usize as *mut c_void;
        let to = 0x7101_0002usize as *mut c_void;
        let (old_alloc, new_alloc) = (0x7101_0100usize, 0x7101_0200usize);
        let _ = console_list::take_pending_for_test(old_alloc);
        seed_reset_state_for_test(from, 0x7101_0300, 0x7101_1000, vec![0x7101_4000]);
        seed_reset_state_for_test(to, old_alloc, 0x7101_2000, vec![0x7101_5000, 0x7101_6000]);

        let prepared = prepare_split(from, to, new_alloc);
        assert_eq!(reset_state_for_test(from), (0x7101_0300, 0x7101_1000, vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![0x7101_4000]));
        assert_eq!(reset_state_for_test(to), (0, 0x7101_2000, vec![1, 2, 3, 4], vec![5, 6, 7, 8], Vec::new()));

        install_prepared_split(prepared);
        assert_eq!(reset_state_for_test(to), (new_alloc, 0x7101_1000, vec![1, 2, 3, 4], vec![5, 6, 7, 8], Vec::new()));
        assert_eq!(console_list::take_pending_for_test(old_alloc), vec![0x7101_5000, 0x7101_6000]);
        remove_state_for_test(from);
        remove_state_for_test(to);
    }

    #[test]
    fn reset_keys_uma_to_the_lists_native_allocator_backing() {
        let list = 0x7101_7001usize as *mut c_void;
        let (logical, native, old_native, block) = (0x7101_7100usize, 0x7101_7200usize, 0x7101_7300usize, 0x7101_7400usize);
        let _ = console_list::take_pending_for_test(old_native);
        super::super::alloc_pool::created_on(list as usize, native, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        seed_reset_state_for_test(list, old_native, 0, vec![block]);

        reset(list, logical, 0);

        assert_eq!(reset_state_for_test(list).0, native,
            "UMA ownership follows the recording's native backing, not the title allocator");
        assert_eq!(console_list::take_pending_for_test(old_native), vec![block]);
        remove_state_for_test(list);
        super::super::alloc_pool::forget_list(list as usize);
        unsafe { super::super::alloc_pool::forget(native); }
    }

    #[test]
    fn abandoned_split_preflight_keeps_old_uma_under_its_allocator() {
        let from = 0x7102_0001usize as *mut c_void;
        let to = 0x7102_0002usize as *mut c_void;
        let old_alloc = 0x7102_0100usize;
        let _ = console_list::take_pending_for_test(old_alloc);
        seed_reset_state_for_test(from, 0x7102_0200, 0x7102_1000, Vec::new());
        seed_reset_state_for_test(to, old_alloc, 0x7102_2000, vec![0x7102_5000]);

        drop(prepare_split(from, to, 0x7102_0300));
        forget(to);

        assert_eq!(console_list::take_pending_for_test(old_alloc), vec![0x7102_5000]);
        assert_eq!(reset_state_for_test(from).0, 0x7102_0200, "the active producer must remain untouched");
        remove_state_for_test(from);
    }

    #[repr(C)]
    struct FakeAdoptedList {
        vt: *const usize,
        refs: AtomicUsize,
        dispatch_calls: AtomicUsize,
        set_compute_rs_calls: AtomicUsize,
        set_compute_uav_calls: AtomicUsize,
        set_pso_calls: AtomicUsize,
        last_pso: AtomicUsize,
        execute_indirect_calls: AtomicUsize,
    }

    static FAKE_CLEAR_CALLS: AtomicUsize = AtomicUsize::new(0);
    static FAKE_CLOSE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static FAKE_CLOSED: AtomicBool = AtomicBool::new(false);
    static FAKE_RESET_FAIL: AtomicBool = AtomicBool::new(false);
    static FAKE_LAST_RESET_PSO: AtomicUsize = AtomicUsize::new(0);
    static FAKE_LAST_CLEAR_PSO: AtomicUsize = AtomicUsize::new(usize::MAX);
    static FAKE_PIN_COUNT_AT_SET: AtomicUsize = AtomicUsize::new(0);
    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    unsafe extern "system" fn fake_dispatch(this: *mut c_void, _x: u32, _y: u32, _z: u32) {
        (*(this as *const FakeAdoptedList)).dispatch_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_pso(this: *mut c_void, pso: *mut c_void) {
        let list = &*(this as *const FakeAdoptedList);
        FAKE_PIN_COUNT_AT_SET.store(list_x::native_generation_count_for_test(this as usize), Ordering::Release);
        list.set_pso_calls.fetch_add(1, Ordering::Relaxed);
        list.last_pso.store(pso as usize, Ordering::Release);
    }

    unsafe extern "system" fn fake_clear_state(_this: *mut c_void, pso: *mut c_void) {
        FAKE_CLEAR_CALLS.fetch_add(1, Ordering::Relaxed);
        FAKE_LAST_CLEAR_PSO.store(pso as usize, Ordering::Release);
    }
    unsafe extern "system" fn fake_close(_this: *mut c_void) -> windows::core::HRESULT {
        FAKE_CLOSE_CALLS.fetch_add(1, Ordering::Relaxed);
        FAKE_CLOSED.store(true, Ordering::Release);
        windows::core::HRESULT(0)
    }
    unsafe extern "system" fn fake_reset(_this: *mut c_void, _alloc: usize, pso: usize) -> i32 {
        FAKE_LAST_RESET_PSO.store(pso, Ordering::Release);
        if FAKE_RESET_FAIL.load(Ordering::Acquire) || !FAKE_CLOSED.swap(false, Ordering::AcqRel) {
            crate::util::E_FAIL
        } else { 0 }
    }
    unsafe extern "system" fn fake_set_compute_rs(this: *mut c_void, _rs: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_compute_rs_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_uav(this: *mut c_void, _slot: u32, _va: u64) {
        (*(this as *const FakeAdoptedList)).set_compute_uav_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_execute_indirect(
        this: *mut c_void,
        _sig: usize,
        _max: u32,
        _args: usize,
        _args_off: u64,
        _count_buf: usize,
        _count_off: u64,
    ) {
        (*(this as *const FakeAdoptedList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn title_pipeline_binds_use_one_retained_identity_and_fail_closed() {
        let _stop_abi = STOP_ABI_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize; orig_vt[2] = fake_release as usize;
        orig_vt[9] = fake_close as usize; orig_vt[11] = fake_clear_state as usize;
        orig_vt[10] = fake_reset as usize; orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize; orig_vt[59] = fake_execute_indirect as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) }, refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0), set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0), set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0), execute_indirect_calls: AtomicUsize::new(0),
        });
        let list = &*fake as *const FakeAdoptedList as *mut c_void;
        let _class = real::test_command_list_class(
            orig_vt.as_ptr() as usize,
            unsafe { ext_vt.as_ptr().add(1) } as usize,
        );
        unsafe { list_x::on_create(list, 0, 0, D3D12_COMMAND_LIST_TYPE_DIRECT) };
        let pso_vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut old = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let old_raw = &mut old as *mut ResolutionPso as usize;
        let replacement_raw = &mut replacement as *mut ResolutionPso as usize;
        let original = 0x7100_48F8usize;
        pso_async::submit_lazy(original, true, Box::new(move || old_raw), 0);
        assert_eq!(pso_async::resolve_wait(original), old_raw);
        let retained = pso_async::begin_pipeline_resolution(original).unwrap();
        let retained_identity = retained.identity_for_test();
        let old_owner = pso_async::released(original);
        let replacement_calls = Arc::new(AtomicUsize::new(0)); let calls = replacement_calls.clone();
        pso_async::submit_lazy(original, true, Box::new(move || {
            calls.fetch_add(1, Ordering::Relaxed);
            replacement_raw
        }), 0);
        let alias_calls = Arc::new(AtomicUsize::new(0)); let calls = alias_calls.clone();
        pso_async::submit_lazy(old_raw, true, Box::new(move || {
            calls.fetch_add(1, Ordering::Relaxed);
            replacement_raw
        }), 0);
        FAKE_PIN_COUNT_AT_SET.store(0, Ordering::Release);
        unsafe { set_pipeline_state_retained(list, original, Some(retained)) };
        assert_eq!(fake.last_pso.load(Ordering::Acquire), old_raw);
        assert_eq!(FAKE_PIN_COUNT_AT_SET.load(Ordering::Acquire), 1, "exact generation was not pinned before native SetPipelineState");
        assert_eq!(replacement_calls.load(Ordering::Relaxed), 0);
        assert_eq!(alias_calls.load(Ordering::Relaxed), 0);
        assert_eq!(list_x::bound_pipeline_for_test(list as usize), original);
        assert_eq!(with(list, |s| s.resolution.as_ref().unwrap().identity_for_test()), retained_identity);
        let clear_resolution = with(list, |s| s.resolution.clone());
        FAKE_CLEAR_CALLS.store(0, Ordering::Release);
        unsafe { clear_pipeline_state_retained(list, original, clear_resolution) };
        assert_eq!(FAKE_CLEAR_CALLS.load(Ordering::Acquire), 1);
        assert_eq!(FAKE_LAST_CLEAR_PSO.load(Ordering::Acquire), old_raw);
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 1, "Set and Clear duplicated the exact generation pin");
        assert_eq!(replacement_calls.load(Ordering::Relaxed), 0);
        assert_eq!(alias_calls.load(Ordering::Relaxed), 0);
        let unavailable = pso_async::pipeline_resolution_for_compute_test(original, || {})
            .unavailable_for_test();
        let (before_sets, before_clears) = (fake.set_pso_calls.load(Ordering::Relaxed), FAKE_CLEAR_CALLS.load(Ordering::Relaxed));
        unsafe { set_pipeline_state_retained(list, original, Some(unavailable)) };
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), before_sets, "unavailable SetPipelineState rebound the prior native pipeline");
        assert_eq!(FAKE_CLEAR_CALLS.load(Ordering::Relaxed), before_clears, "eligible omission reset logical root state");
        assert!(!list_x::recording_poisoned(list as usize));
        unsafe {
            dispatch_flags(list, 1, 1, 1, 0);
            execute_indirect_retained(
                list, 1, 1, 0, 0, 0, 0, IndirectSignatureInfo::new(true, false, None),
            );
        }
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0, "unavailable Dispatch executed the preceding PSO");
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0, "unavailable ExecuteIndirect executed the preceding PSO");

        let unavailable = pso_async::pipeline_resolution_for_test(original, || {}).unavailable_for_test();
        let (before_sets, before_clears) = (fake.set_pso_calls.load(Ordering::Relaxed), FAKE_CLEAR_CALLS.load(Ordering::Relaxed));
        unsafe { set_pipeline_state_retained(list, original, Some(unavailable)) };
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), before_sets);
        assert_eq!(FAKE_CLEAR_CALLS.load(Ordering::Relaxed), before_clears, "poisoning reset physical state");
        assert!(list_x::recording_poisoned(list as usize));
        FAKE_CLOSE_CALLS.store(0, Ordering::Release);
        FAKE_CLOSED.store(false, Ordering::Release);
        assert!(unsafe { real::close_for_test(list) } < 0);
        assert_eq!(FAKE_CLOSE_CALLS.load(Ordering::Acquire), 1, "poisoned Close did not close natively");
        assert!(real::submission_rejected_for_test(&[list]), "poisoned recording passed submission guard");

        let (parent, bundle) = (0x7100_48FAusize, 0x7100_48FBusize);
        unsafe {
            list_x::on_create(parent as *mut c_void, 0, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            list_x::on_create(bundle as *mut c_void, 0, 0, D3D12_COMMAND_LIST_TYPE_BUNDLE);
        }
        list_x::poison_recording(bundle);
        list_x::on_execute_bundle(parent, bundle); list_x::on_execute_bundle(parent, bundle);
        assert!(real::submission_rejected_for_test(&[parent as *mut c_void]));
        list_x::on_release(parent); list_x::on_release(bundle);

        let unavailable = pso_async::pipeline_resolution_for_test(original, || {}).unavailable_for_test();
        FAKE_LAST_CLEAR_PSO.store(usize::MAX, Ordering::Release);
        unsafe { clear_pipeline_state_retained(list, original, Some(unavailable)) };
        assert_eq!(FAKE_LAST_CLEAR_PSO.load(Ordering::Acquire), 0, "unavailable ClearState did not send NULL");
        assert_eq!(with(list, |s| s.pso), original);
        assert_eq!(list_x::bound_pipeline_for_test(list as usize), original);
        assert!(list_x::recording_poisoned(list as usize), "ClearState must not unpoison a recording");
        assert_eq!(unsafe { list_x::reset_pc(list, 0, 0) }, 0);
        assert!(!list_x::recording_poisoned(list as usize), "successful Reset must clear poison");

        forget(list);
        list_x::clear_segment_state(list as usize);
        list_x::on_release(list as usize);
        assert_eq!(pso_async::released(original), 0);
        assert_eq!(pso_async::released(old_raw), 0);
        assert_eq!(unsafe { resolution_release(old_owner as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(replacement_raw as *mut c_void) }, 0);
    }

    #[test]
    fn prepared_initial_pipeline_commits_only_after_create_or_reset_success() {
        let _stop_abi = STOP_ABI_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize; orig_vt[2] = fake_release as usize;
        orig_vt[10] = fake_reset as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize; ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) }, refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0), set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0), set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0), execute_indirect_calls: AtomicUsize::new(0),
        });
        let list = &*fake as *const FakeAdoptedList as *mut c_void;
        let _class = real::test_command_list_class(
            orig_vt.as_ptr() as usize,
            unsafe { ext_vt.as_ptr().add(1) } as usize,
        );
        unsafe { list_x::on_create(list, 0, 0, D3D12_COMMAND_LIST_TYPE_DIRECT) };
        let fallback = PreparedNativePipeline::from_resolution(
            0x7100_4900, Some(pso_async::pipeline_resolution_for_test(0x7100_4900, || {})),
        );
        assert_eq!(fallback.raw(), 0x7100_4900); drop(fallback);
        let pso_vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut exact = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let (exact_raw, replacement_raw) = (&mut exact as *mut ResolutionPso as usize, &mut replacement as *mut ResolutionPso as usize);
        let original = 0x7100_4901usize;
        pso_async::submit_lazy(original, false, Box::new(move || exact_raw), 0);
        assert_eq!(pso_async::resolve_wait(original), exact_raw);
        let failed_create = PreparedNativePipeline::from_resolution(
            original, pso_async::begin_pipeline_resolution(original),
        );
        assert_eq!(failed_create.raw(), exact_raw); drop(failed_create);
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 0);
        let retained = pso_async::begin_pipeline_resolution(original).unwrap();
        let identity = retained.identity_for_test();
        let prepared = PreparedNativePipeline::from_resolution(original, Some(retained));
        assert_eq!(prepared.raw(), exact_raw);
        let exact_owner = pso_async::released(original);
        let replacement_calls = Arc::new(AtomicUsize::new(0)); let calls = replacement_calls.clone();
        pso_async::submit_lazy(original, false, Box::new(move || { calls.fetch_add(1, Ordering::Relaxed); replacement_raw }), 0);
        prepared.commit(list, 0);
        assert_eq!(with(list, |s| s.resolution.as_ref().unwrap().identity_for_test()), identity);
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 1);
        assert_eq!(replacement_calls.load(Ordering::Relaxed), 0);

        let unavailable = pso_async::pipeline_resolution_for_compute_test(0x7100_4902, || {}).unavailable_for_test();
        let prepared = PreparedNativePipeline::from_resolution(0x7100_4902, Some(unavailable));
        assert_eq!(prepared.raw(), 0); prepared.commit(list, 0);
        assert!(!list_x::recording_poisoned(list as usize));
        let unavailable = pso_async::pipeline_resolution_for_test(0x7100_4903, || {}).unavailable_for_test();
        PreparedNativePipeline::from_resolution(0x7100_4903, Some(unavailable)).commit(list, 0);
        assert!(list_x::recording_poisoned(list as usize));
        let old_identity = with(list, |s| s.resolution.as_ref().unwrap().identity_for_test());
        drop(PreparedNativePipeline::from_resolution(
            0x7100_4904, Some(pso_async::pipeline_resolution_for_test(0x7100_4904, || {})),
        ));
        assert_eq!(with(list, |s| s.resolution.as_ref().unwrap().identity_for_test()), old_identity);
        assert!(list_x::recording_poisoned(list as usize));

        let mut reset_exact = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let reset_raw = &mut reset_exact as *mut ResolutionPso as usize;
        let reset_original = 0x7100_4906usize;
        pso_async::submit_lazy(reset_original, false, Box::new(move || reset_raw), 0);
        assert_eq!(pso_async::resolve_wait(reset_original), reset_raw);
        FAKE_CLOSED.store(true, Ordering::Release); FAKE_RESET_FAIL.store(true, Ordering::Release);
        assert!(unsafe { real::reset_for_test(list, std::ptr::null_mut(), reset_original as *mut c_void) } < 0);
        assert_eq!(FAKE_LAST_RESET_PSO.load(Ordering::Acquire), reset_raw);
        assert!(list_x::recording_poisoned(list as usize));
        assert_eq!(with(list, |s| s.resolution.as_ref().unwrap().identity_for_test()), old_identity);
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 1);
        FAKE_RESET_FAIL.store(false, Ordering::Release);
        FAKE_CLOSED.store(true, Ordering::Release);
        assert_eq!(unsafe { real::reset_for_test(list, std::ptr::null_mut(), std::ptr::null_mut()) }, 0);
        assert_eq!(FAKE_LAST_RESET_PSO.load(Ordering::Acquire), 0);
        assert!(!list_x::recording_poisoned(list as usize));

        FAKE_CLOSED.store(true, Ordering::Release);
        assert_eq!(unsafe { list_x::reset_pc(list, 0, reset_original) }, 0);
        assert_eq!(FAKE_LAST_RESET_PSO.load(Ordering::Acquire), reset_raw);
        assert_eq!(tracked_pso(list), reset_original);
        assert_eq!(list_x::native_generation_count_for_test(list as usize), 1);
        let reset_owner = pso_async::released(reset_original);

        forget(list); list_x::clear_segment_state(list as usize); list_x::on_release(list as usize);
        assert_eq!(pso_async::released(original), 0);
        assert_eq!(unsafe { resolution_release(exact_owner as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(replacement_raw as *mut c_void) }, 0);
        assert_eq!(unsafe { resolution_release(reset_owner as *mut c_void) }, 0);
    }

    /// `DispatchX` with thread dimensions (flag 0x20, as the BVH builder's rehydration passes
    /// use it): the console launches whole groups of the pipeline's `numthreads` and leaves
    /// the lanes past the count out; without the flag the counts are groups. With the flag
    /// and a group shape not known, nothing is launched (the counts as groups was the bug).
    #[test]
    fn dispatch_x_with_thread_dimensions_launches_whole_groups() {
        let pso = 0x7300_0001usize;
        real::register_threads_for_test(pso, [64, 1, 1]);
        // 640 threads in groups of 64: 10 groups, the last one whole
        assert_eq!(dispatch_groups(pso, [640, 1, 1], 0x20), Some(([10, 1, 1], [0, 0, 0])));
        // 100 threads: 2 groups, and the lanes past 100 to leave out
        assert_eq!(dispatch_groups(pso, [100, 1, 1], 0x20), Some(([2, 1, 1], [100, 0, 0])));
        // the wave32 bit (0x8000) changes nothing here
        assert_eq!(dispatch_groups(pso, [100, 1, 1], 0x8020), Some(([2, 1, 1], [100, 0, 0])));
        // a 2-D group shape
        let pso2 = 0x7300_0002usize;
        real::register_threads_for_test(pso2, [8, 8, 1]);
        assert_eq!(dispatch_groups(pso2, [20, 16, 3], 0x20), Some(([3, 2, 3], [20, 0, 0])));
        // group counts (Dispatch, DispatchX without the flag) are launched as they are
        assert_eq!(dispatch_groups(pso, [7, 2, 1], 0), Some(([7, 2, 1], [0, 0, 0])));
        assert_eq!(dispatch_groups(pso, [7, 2, 1], 0x8000), Some(([7, 2, 1], [0, 0, 0])));
        // no threads: no groups
        assert_eq!(dispatch_groups(pso, [0, 1, 1], 0x20), Some(([0, 1, 1], [0, 0, 0])));
        // thread dimensions on a pipeline whose group shape is not known: not launched
        assert_eq!(dispatch_groups(0x7300_00FF, [100, 1, 1], 0x20), None);
        assert_eq!(dispatch_groups(0x7300_00FF, [100, 1, 1], 0), Some(([100, 1, 1], [0, 0, 0])));

        let last = MAX_DISPATCH_GROUPS * 64;
        let at_limit = dispatch_groups(pso, [last, 1, 1], 0x20).unwrap().0;
        let past_limit = dispatch_groups(pso, [last + 1, 1, 1], 0x20).unwrap().0;
        assert!(valid_dispatch_grid(at_limit));
        assert!(!valid_dispatch_grid(past_limit));
        for k in 0..3 {
            let mut grid = [1; 3];
            grid[k] = MAX_DISPATCH_GROUPS;
            assert!(valid_dispatch_grid(grid), "axis {k} at the D3D12 limit");
            grid[k] += 1;
            assert!(!valid_dispatch_grid(grid), "axis {k} past the D3D12 limit");
        }
    }

    #[test]
    fn abi_preparation_snapshots_state_and_reserves_disjoint_regions() {
        let list = 0x7301_0001usize as *mut c_void;
        let u = crate::gpu::uma::Uma::with_windows(1, 8 << 20).expect("test UMA");
        let layout = Arc::new(Layout { size: 64, inline: 4, ..Default::default() });
        let args = (0..64u8).collect::<Vec<_>>();
        with(list, |state| {
            state.layout = Some(layout.clone());
            state.args = args.clone();
            state.allocator = 0x7301_1000;
        });

        let prepared = prepare_abi_state(list, &u, false, None).unwrap();
        assert!(Arc::ptr_eq(prepared.layout.as_ref().unwrap(), &layout));
        assert_eq!(prepared.args, args);
        assert_eq!(prepared.inline, 4);
        assert_eq!(prepared.allocator, 0x7301_1000);
        assert_eq!((prepared.table_at | prepared.io_at | prepared.res_at) & 255, 0);
        assert!(prepared.table_at + 60 <= prepared.io_at);
        assert!(prepared.io_at + crate::gpu::recomp::KERNEL_SGPRS * std::mem::size_of::<u32>() <= prepared.res_at);

        with(list, |state| {
            state.layout = None;
            state.args.clear();
            state.allocator = 0x7301_2000;
        });
        assert!(Arc::ptr_eq(prepared.layout.as_ref().unwrap(), &layout));
        assert_eq!(prepared.args, args);
        assert_eq!(prepared.allocator, 0x7301_1000);

        let state = take_state(list as usize).unwrap();
        assert_eq!(state.uma.len(), 1, "the three reservations should share one arena block");
        for block in state.uma {
            assert!(u.free(block));
        }
    }

    /// The quad sample mask (#142): a list's draws get its io words until the list's reset or
    /// ClearState, and a mid-list split carries them; an all-ones mask writes nothing (the io
    /// slots stay 0).
    #[test]
    fn quad_sample_mask_reaches_the_io_of_the_lists_draws() {
        let (list, split) = (0x5151_0000usize as *mut c_void, 0x5151_1000usize as *mut c_void);
        let words = super::super::quad_mask::kill_words(0x0000_FFFF_FFFF_0000);
        assert_eq!(words, [0xFFFF, 0xFFFF_0000]);
        set_quad_kill(list, words);
        assert_eq!(with(list, |s| s.quad_kill), words);
        let mut io = vec![0u32; crate::gpu::recomp::KERNEL_SGPRS];
        quad_kill_io(&mut io, words);
        assert_eq!((io[117], io[118]), (0xFFFF, 0xFFFF_0000));
        assert!(io.iter().enumerate().all(|(k, &w)| w == 0 || k == 117 || k == 118));
        let taken = [crate::gpu::recomp::IO_UMA_BASE_HI, crate::gpu::recomp::IO_THREAD_LIMIT[0], crate::gpu::recomp::IO_THREAD_LIMIT[1], crate::gpu::recomp::IO_THREAD_LIMIT[2]];
        assert!(!crate::gpu::recomp::IO_QUAD_KILL.iter().any(|&k| taken.contains(&k) || k >= crate::gpu::recomp::KERNEL_SCC_SLOT));
        let mut io = vec![0u32; crate::gpu::recomp::KERNEL_SGPRS];
        quad_kill_io(&mut io, super::super::quad_mask::kill_words(u64::MAX));
        assert!(io.iter().all(|&w| w == 0));
        inherit_for_split(list, split, 0);
        assert_eq!(with(split, |s| s.quad_kill), words);
        clear_state(list, 0);
        assert_eq!(with(list, |s| s.quad_kill), [0; 2]);
        set_quad_kill(list, words);
        reset(list, 0, 0);
        assert_eq!(with(list, |s| s.quad_kill), [0; 2]);
        forget(list);
        forget(split);
    }

    #[test]
    fn dispatch_limits_are_explicit_and_graphics_ignores_them() {
        let slots = crate::gpu::recomp::IO_THREAD_LIMIT;
        let compute = prepared_abi_io(None, &[], 0x1234_0000, false, None, 0, [100, 17, 0]);
        assert_eq!([compute[slots[0]], compute[slots[1]], compute[slots[2]]], [100, 17, 0]);

        let ordinary = prepared_abi_io(None, &[], 0x1234_1000, false, None, 0, [0; 3]);
        assert_eq!([ordinary[slots[0]], ordinary[slots[1]], ordinary[slots[2]]], [0; 3]);

        let graphics = prepared_abi_io(None, &[], 0x1234_2000, true, None, 0, [9, 8, 7]);
        assert_eq!([graphics[slots[0]], graphics[slots[1]], graphics[slots[2]]], [0; 3]);
    }

    #[test]
    fn command_signature_recreation_replaces_every_classification_bit() {
        assert_eq!(std::mem::size_of::<IndirectSignatureInfo>(), 8);
        let root_reuse = 0x7301_2800usize;
        note_indirect_signature(root_reuse, IndirectSignatureInfo::new(true, true, Some(12)));
        note_indirect_signature(root_reuse, IndirectSignatureInfo::default());
        assert_eq!(indirect_signature(root_reuse), IndirectSignatureInfo::default(),
            "a plain signature inherited root-changing or dispatch state");

        let dispatch_reuse = 0x7301_2810usize;
        note_indirect_signature(dispatch_reuse, IndirectSignatureInfo::new(true, false, Some(16)));
        let replacement = IndirectSignatureInfo::new(true, true, None);
        note_indirect_signature(dispatch_reuse, replacement);
        assert_eq!(indirect_signature(dispatch_reuse), replacement,
            "a non-dispatch-only replacement inherited the old stride");
    }

    #[test]
    fn command_signature_snapshot_never_mixes_generations() {
        let sig = 0x7301_2820usize;
        let a = IndirectSignatureInfo::new(true, false, Some(12));
        let b = IndirectSignatureInfo::new(false, true, None);
        note_indirect_signature(sig, a);
        let writer = std::thread::spawn(move || {
            for n in 0..20_000 {
                note_indirect_signature(sig, if n & 1 == 0 { b } else { a });
            }
        });
        for _ in 0..20_000 {
            let seen = indirect_signature(sig);
            assert!(seen == a || seen == b, "mixed command-signature snapshot: {seen:?}");
        }
        writer.join().unwrap();
    }

    #[test]
    fn deferred_indirect_owns_its_command_signature_snapshot() {
        let sig = 0x7301_2830usize;
        let original = IndirectSignatureInfo::new(true, false, Some(12));
        note_indirect_signature(sig, original);
        let deferred = list_x::Deferred::Indirect {
            sig: 1, max: 2, args: 3, args_off: 4, count_buf: 5, count_off: 6,
            signature: indirect_signature(sig),
        };
        note_indirect_signature(sig, IndirectSignatureInfo::new(false, true, None));

        let list_x::Deferred::Indirect { signature: retained, .. } = deferred else { unreachable!() };
        assert_eq!(retained, original,
            "deferred replay followed the classification now stored at a reused signature address");
    }

    #[test]
    fn resolved_compute_metadata_does_not_follow_a_reused_native_address() {
        let exact = 0x7301_3000usize;
        let replacement = 0x7301_4000usize;
        real::register_compute_metadata_for_test(exact, 6, [4, 2, 1], (2, 0), 0x7301_5000);
        real::register_compute_metadata_for_test(replacement, 9, [8, 1, 1], (3, 1), 0x7301_6000);
        pso_async::submit_lazy(exact, true, Box::new(move || replacement), 0);
        assert_eq!(pso_async::resolve_wait(exact), replacement);

        assert_eq!(real::pso_threads(exact), Some([8, 1, 1]),
            "the title-address accessor should still resolve an active stand-in");
        assert_eq!(real::pso_threads_exact(exact), Some([4, 2, 1]));
        assert_eq!(real::pso_user_sgprs_exact(exact), Some(6));
        assert_eq!(real::pso_dispatch_remap_exact(exact), Some((2, 0)));
        assert_eq!(real::pso_root_signature_exact(exact), Some(0x7301_5000));
        assert_eq!(dispatch_groups(exact, [8, 4, 1], DISPATCH_THREAD_DIMENSIONS),
            Some(([2, 2, 1], [0; 3])),
            "an already-selected native generation was redirected through PENDING");

        assert_eq!(pso_async::released(exact), replacement);
        real::forget_pso_for_test(exact);
        real::forget_pso_for_test(replacement);
    }

    #[test]
    fn async_compute_pso_success_and_failure_in_dispatch_and_execute_indirect() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        // Extended vtable layout expected by `real::orig_fn`: entry -1 holds `orig_vt` pointer,
        // and entries 0..64 hold the COM method pointers used via `ID3D12GraphicsCommandList`.
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            // SAFETY: `ext_vt` has 65 elements so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // 1. Background compute build SUCCEEDS: real PSO R is registered in TRANSLATED_COMPUTE,
        // stand-in S is NOT. ExecuteIndirect resolves S -> R and treats it as translated.
        let standin_ok = 0x7200_0001usize;
        let pso_vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut real_ok_pso = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let real_ok = &mut real_ok_pso as *mut ResolutionPso as usize;
        let sig_root_changing = 0x7200_00F0usize;
        note_indirect_signature(sig_root_changing, IndirectSignatureInfo::new(false, true, None));

        pso_async::submit(
            standin_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_ok), real_ok);
        assert!(!real::is_translated_compute(standin_ok));
        assert!(real::is_translated_compute(real_ok));

        reset(list_ptr, 0x7200_0100, standin_ok);
        unsafe {
            // With `sig_root_changing`, a translated compute PSO is caught by the root-changing
            // guard instead of falling through to `pc` (which happened when `R` wasn't registered).
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "ExecuteIndirect must recognize resolved async compute PSO R as translated"
        );

        // 2. Background compute build FAILS (returns 0): stand-in S_fail is NOT in
        // TRANSLATED_COMPUTE. Dispatch must skip cleanly without binding XCL's ABI RS/UAVs
        // onto S_fail, and ExecuteIndirect must not enter bind_abi either.
        let standin_fail = 0x7200_0003usize;
        pso_async::submit(standin_fail, true, Box::new(|| 0));
        assert_eq!(pso_async::resolve_wait(standin_fail), standin_fail);
        assert!(!real::is_translated_compute(standin_fail));

        reset(list_ptr, 0x7200_0100, standin_fail);
        let before = counts_snapshot();
        unsafe {
            dispatch(list_ptr, 1, 1, 1);
        }
        let after = counts_snapshot();
        assert_eq!(after[DISPATCHES], before[DISPATCHES]);
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);

        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "failed async compute stand-in must not enter translated ExecuteIndirect path"
        );

        // 3. The same failed stand-in (made from a console shader) with a Dispatch signature:
        // nothing to run, as for Dispatch – the PC call would pair the stand-in with the list's
        // compute root signature (XCL's ABI one after a translated dispatch: debug layer 953)
        let sig_dispatch = 0x7200_00F8usize;
        real::register_console_compute_for_test(standin_fail);
        note_indirect_signature(sig_dispatch, IndirectSignatureInfo::new(true, false, None));
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 1, "a dispatch of an untranslated console compute pipeline must not reach the PC ExecuteIndirect");
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 4. A native PC compute pipeline (not made from a console shader) with the same Dispatch
        // signature still goes to the PC call, as on main
        let native = 0x7200_0004usize;
        reset(list_ptr, 0x7200_0100, native);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 2, "a native PC compute pipeline must still reach the PC ExecuteIndirect");

        // a signature made later at the same address without a Dispatch argument goes to the PC
        reset(list_ptr, 0x7200_0100, standin_fail);
        note_indirect_signature(sig_dispatch, IndirectSignatureInfo::default());
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 3);

        forget(list_ptr);
        list_x::on_release(list_ptr as usize);
        let real_ok_owner = pso_async::released(standin_ok);
        let _ = pso_async::released(standin_fail);
        real::forget_pso_for_test(real_ok);
        real::forget_pso_for_test(standin_fail);
        assert_eq!(real_ok_owner, real_ok);
        assert_eq!(unsafe { resolution_release(real_ok_owner as *mut c_void) }, 0);
        assert!(!real::is_console_compute(standin_fail), "a released pipeline's address does not stay a console one");
    }

    #[test]
    fn execute_indirect_skips_failed_shared_console_compute_standin() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            last_pso: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // With shared stand-ins (`standin::shell`), `dev_create_compute_pso` registers the
        // title-facing shell in `CONSOLE_COMPUTE`, while `pso_async::resolve_wait` returns the
        // shared driver pipeline when the background build fails.
        let shell_fail = 0x7210_0001usize;
        let shell_ok = 0x7210_0002usize;
        let shared_driver = 0x7210_0003usize;
        let pso_vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut real_ok_pso = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let real_ok = &mut real_ok_pso as *mut ResolutionPso as usize;
        let sig_dispatch = 0x7210_00F8usize;
        let sig_root_changing = 0x7210_00F0usize;
        note_indirect_signature(sig_dispatch, IndirectSignatureInfo::new(true, false, None));
        note_indirect_signature(sig_root_changing, IndirectSignatureInfo::new(false, true, None));

        pso_async::submit_lazy(shell_fail, true, Box::new(|| 0), 0);
        pso_async::shared(shell_fail, shared_driver);
        real::register_console_compute_for_test(shell_fail);

        pso_async::submit_lazy(
            shell_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
            0,
        );
        pso_async::shared(shell_ok, shared_driver);
        real::register_console_compute_for_test(shell_ok);

        assert_eq!(pso_async::resolve_wait(shell_fail), shared_driver);
        assert!(!pso_async::is_pending(shell_fail));
        assert!(real::is_console_compute(shell_fail));
        assert!(!real::is_console_compute(shared_driver));

        // 1. Failed shared console compute stand-in with a Dispatch signature must be skipped,
        // never forwarded to PC ExecuteIndirect with the shared driver pipeline.
        reset(list_ptr, 0x7210_0100, shell_fail);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "failed shared console compute stand-in must not reach PC ExecuteIndirect"
        );
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 2. Another shell sharing the same driver pipeline whose build succeeds still takes the
        // translated path (caught here by the root-changing guard rather than falling through to PC).
        reset(list_ptr, 0x7210_0100, shell_ok);
        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0);

        // 3. Releasing `shell_fail` forgets its `CONSOLE_COMPUTE` entry so a native PC compute
        // pipeline later allocated at the same address still reaches PC ExecuteIndirect.
        let _ = pso_async::released(shell_fail);
        real::forget_pso_for_test(shell_fail);
        assert!(!real::is_console_compute(shell_fail));
        reset(list_ptr, 0x7210_0100, shell_fail);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "native PC pipeline at a released shell's address must still reach PC ExecuteIndirect"
        );

        forget(list_ptr);
        note_indirect_signature(sig_dispatch, IndirectSignatureInfo::default());
        list_x::on_release(list_ptr as usize);
        let real_ok_owner = pso_async::released(shell_ok);
        real::forget_pso_for_test(shell_ok);
        real::forget_pso_for_test(real_ok);
        assert_eq!(real_ok_owner, real_ok);
        assert_eq!(unsafe { resolution_release(real_ok_owner as *mut c_void) }, 0);
    }

    #[test]
    fn two_pass_pso_skips_direct_draw_and_unmatched_execute_indirect() {
        #[repr(C)]
        struct FakeTwoPassList {
            vt: *const usize,
            refs: AtomicUsize,
            set_graphics_rs_calls: AtomicUsize,
            execute_indirect_calls: AtomicUsize,
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            (*(this as *const FakeTwoPassList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            (*(this as *const FakeTwoPassList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }
        unsafe extern "system" fn set_graphics_rs(this: *mut c_void, _rs: *mut c_void) {
            (*(this as *const FakeTwoPassList)).set_graphics_rs_calls.fetch_add(1, Ordering::Relaxed);
        }
        unsafe extern "system" fn exec_indirect(
            this: *mut c_void,
            _sig: usize,
            _max: u32,
            _args: usize,
            _args_off: u64,
            _count_buf: usize,
            _count_off: u64,
        ) {
            (*(this as *const FakeTwoPassList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[1] = add_ref as usize;
        orig_vt[2] = release as usize;
        orig_vt[30] = set_graphics_rs as usize;
        orig_vt[59] = exec_indirect as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeTwoPassList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            set_graphics_rs_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeTwoPassList as *mut c_void;

        // Register a finished two-pass GS/MS PSO behind an async stand-in (registered in TWO_PASS,
        // not TRANSLATED_GRAPHICS).
        let standin_tp = 0x7250_0001usize;
        let pso_vtable = [0usize, resolution_add_ref as usize, resolution_release as usize];
        let mut real_tp_pso = ResolutionPso { vtable: pso_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let real_tp = &mut real_tp_pso as *mut ResolutionPso as usize;
        pso_async::submit(
            standin_tp,
            false,
            Box::new(move || {
                real::register_two_pass_for_test(real_tp, "GS");
                real_tp
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_tp), real_tp);
        assert!(real::two_pass_of(real_tp).is_some());
        assert!(!real::is_translated_graphics(real_tp));

        reset(list_ptr, 0x7250_0100, standin_tp);
        unsafe {
            // Seed a non-null title graphics RS so a fallthrough would visibly call SetGraphicsRootSignature.
            set_graphics_root_signature(list_ptr, list_ptr);
        }

        // 1. Direct draw (`graphics_bound`) must return None without binding the title's RS.
        let before = counts_snapshot();
        let bound = unsafe { graphics_bound(list_ptr) };
        let after = counts_snapshot();
        assert!(bound.is_none(), "two-pass MS/GS PSO must not fall through to PC DrawInstanced");
        assert!(after[DRAWS_SKIPPED] > before[DRAWS_SKIPPED]);
        assert_eq!(after[DRAWS_PC], before[DRAWS_PC]);
        assert_eq!(fake.set_graphics_rs_calls.load(Ordering::Relaxed), 0);

        // 2. Unmatched `ExecuteIndirect` must skip without binding the title's RS or calling PC ExecuteIndirect.
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, 0x7250_00F0, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert!(after[DRAWS_SKIPPED] > before[DRAWS_SKIPPED]);
        assert_eq!(fake.set_graphics_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0);

        forget(list_ptr);
        list_x::clear_segment_state(list_ptr as usize);
        let real_tp_owner = pso_async::released(standin_tp);
        real::forget_pso_for_test(real_tp);
        assert_eq!(real_tp_owner, real_tp);
        assert_eq!(unsafe { resolution_release(real_tp_owner as *mut c_void) }, 0);
        assert!(real::two_pass_of(real_tp).is_none());
    }

    #[test]
    fn create_list_reset_and_clear_state_track_initial_pso_on_d3d12() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        if console_list::enabled() {
            return;
        }
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            real::adopt(dev.as_raw(), real::Kind::Device);

            // Build two real PC compute PSOs from the inbox dummy CS bytecode.
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .expect("CreateRootSignature");
            let cs = real::dummy_cs_bytecode();
            let make_pso = || -> ID3D12PipelineState {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() },
                    ..Default::default()
                };
                let pso = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
                std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
                pso
            };
            let pso1 = make_pso();
            let pso2 = make_pso();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");

            // Stand-in synthetic key backed by real `pso1` in pso_async: CreateCommandList must
            // resolve it via resolve_for_bind for PC D3D12 while tracking `standin` in `bind`.
            let standin = 0x7300_0001usize;
            let pso1_raw = pso1.as_raw() as usize;
            pso_async::submit(standin, true, Box::new(move || pso1_raw));
            assert_eq!(pso_async::resolve_wait(standin), pso1_raw);

            let mut raw_list: *mut c_void = std::ptr::null_mut();
            let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
            let hr = device_x::create_command_list_pc(
                dev.as_raw(),
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32,
                alloc.as_raw() as usize,
                standin,
                &iid,
                &mut raw_list,
            );
            assert!(hr >= 0 && !raw_list.is_null(), "create_command_list_pc failed: {hr:#x}");
            let backing = title_list::backing(raw_list).expect("wrapped in title_list");
            assert_eq!(tracked_pso(backing), standin, "CreateCommandList must seed bind::State.pso");

            // ClearState(pso2) updates bind::State.pso to pso2 and clears root args.
            let rs_key = 0x7300_0010usize;
            console_list::note_root_signature(rs_key, &test_rts0_blob());
            set_compute_root_signature(backing, rs_key as *mut c_void);
            set_compute_root_constant(backing, 0, 0xABCD, 0);
            let cl = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).unwrap();
            cl.ClearState(&pso2);
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "ClearState must update bind::State.pso");
            with(backing, |s| assert!(s.layout.is_none() && s.args.is_empty()));

            // ClearState(None) clears bind::State.pso to 0.
            cl.ClearState(None);
            assert_eq!(tracked_pso(backing), 0, "ClearState(None) must clear bind::State.pso");

            // Close + reset_x(alloc, pso2) updates bind::State.pso to pso2 and clears root args.
            set_graphics_root_signature(backing, rs_key as *mut c_void);
            set_graphics_root_constant(backing, 0, 0x1111, 0);
            cl.Close().expect("Close");
            let hr = list_x::reset_x(backing, alloc.as_raw() as usize, pso2.as_raw() as usize);
            assert!(hr >= 0, "reset_x failed: {hr:#x}");
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "Reset must update bind::State.pso");
            with(backing, |s| assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty()));

            // Release the wrapped title list cleanly (calls bind::forget + list_x::on_release).
            let release_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(raw_list as *const *const usize)).add(2));
            assert_eq!(release_fn(raw_list), 0);
            let _ = pso_async::released(standin);
        }
    }
}
