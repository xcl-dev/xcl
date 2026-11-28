//! Console descriptor heaps (CBV/SRV/UAV and samplers).
//!
//! On the console a descriptor is GPU-readable memory: 32 bytes per CBV/SRV/UAV slot (T# /
//! V#), 16 per sampler (S#), and a descriptor table in a root argument is a 32-bit GPU address
//! (shaders load it with the high half 0 – amd_fidelityfx_upscaler_xbox.dll, HANDOFF "FSR4").
//! XCL gives console code heaps whose slots live in unified memory (so the table addresses
//! console code computes from the heap start work as they are in the recompiler's `mem`
//! model) and keeps PC descriptors beside each: view slot k → an SRV and, next to it, a UAV;
//! sampler slot k → PC sampler k. A slot holds the descriptor in the XCL shader ABI
//! (`desc::xcl_image_words`: T# word 0 = the SRV's index in the shader-visible PC heap;
//! `desc::xcl_sampler_words`: S# word 3 [11:0] = k; V# with its unified-memory address).
//!
//! View descriptors are values on the console: titles create them in a CPU-only heap and copy
//! them into the shader-visible one as memory (the console header's inline
//! CopyDescriptorsSimple; XDefiant copies every texture that way), and a descriptor table may
//! be in any heap. So all console view heaps share one shader-visible PC heap (`Global`):
//! each gets a range there, T# word 0 is an index into it – a T# copied anywhere still names
//! its PC descriptor – and that heap is what lists bind. The title's own PC heap stays as a
//! CPU-only mirror (where views are made, copies come from and UAV clears take their CPU
//! handle). A heap gets its place in the shared heap a page of `PAGE_SLOTS` slots at a time,
//! when a slot of that page is first used: titles make large heaps and fill few slots
//! (Minecraft: eight view heaps of 73 727 slots – 147 454 PC descriptors each, more in all
//! than the 1 000 000 an NVIDIA shared heap holds – with a few hundred views in them). A heap
//! left out of the shared heap would be lost: lists bind only the shared heap for views.
//! Without the shared heap (it could not be made) a heap keeps its own PC heap: SRV at 2k,
//! UAV at 2k + 1.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::{self, Acquire, Relaxed, Release}};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// bytes of a CBV/SRV/UAV slot (T# / V#) and of a sampler slot (S#)
pub const SLOT: u32 = 32;
pub const SAMPLER_SLOT: u32 = 16;

/// Whose memory a heap's slots are in.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Memory {
    /// a sparse VRAM allocation in window 0
    Sparse,
    /// committed system RAM
    Uma,
    /// the title's (its allocation hook gave it: `kernelx::runtime_alloc`)
    Title,
}

struct Heap {
    generation: u64,
    obj: usize,
    shadow: usize,
    n: u32,
    pc_cpu: u64,
    /// 0: not shader visible
    pc_gpu: u64,
    inc: u32,
    sampler: bool,
    /// views in the shared heap: the index there of each page's first SRV (`UNMAPPED`: the
    /// page has no place yet); empty: the heap's views are not in the shared heap
    pages: Vec<u32>,
    /// shader visible as the title made it
    visible: bool,
    memory: Memory,
}

impl Heap {
    fn slot_bytes(&self) -> u32 {
        if self.sampler {
            SAMPLER_SLOT
        } else {
            SLOT
        }
    }
}

static HEAPS: Mutex<Vec<Heap>> = Mutex::new(Vec::new());
static NEXT_HEAP_GENERATION: AtomicU64 = AtomicU64::new(0);
static HEAP_LIFETIMES: Mutex<()> = Mutex::new(());

pub(super) type HeapLifetimeGuard = std::sync::MutexGuard<'static, ()>;

/// Exclude final descriptor-heap `Release` while an exact heap generation is turned into
/// strong native owners. This is separate from the descriptor-write gate: hooks keep that gate
/// across driver calls which may temporarily release a heap.
pub(super) fn lock_lifetime() -> Option<HeapLifetimeGuard> {
    super::vram::srv_tracking_enabled().then(|| HEAP_LIFETIMES.lock().unwrap())
}

/// Shared ranges being written by lock-free single-descriptor publication, and releases which
/// cannot return to the allocator until every overlapping write has finished.
#[derive(Default)]
struct PublishLeases {
    active: Vec<(u32, u32)>,
    deferred: Vec<(u32, u32)>,
}

impl PublishLeases {
    fn acquire(&mut self, first: u32, count: u32) {
        self.active.push((first, count));
    }

    fn release_range(&mut self, free: &mut Vec<(u32, u32)>, first: u32, count: u32) {
        let range = (first, count);
        if self.active.iter().any(|&active| ranges_overlap(active, range)) {
            self.deferred.push(range);
        } else {
            merge_release(free, first, count);
        }
    }

    fn finish(&mut self, free: &mut Vec<(u32, u32)>, range: (u32, u32)) {
        self.finish_ranges(free, &[range]);
    }

    fn finish_ranges(&mut self, free: &mut Vec<(u32, u32)>, ranges: &[(u32, u32)]) {
        for &range in ranges {
            if let Some(i) = self.active.iter().position(|&active| active == range) {
                self.active.swap_remove(i);
            }
        }
        let mut i = 0;
        while i < self.deferred.len() {
            let deferred = self.deferred[i];
            if self.active.iter().any(|&active| ranges_overlap(active, deferred)) {
                i += 1;
            } else {
                self.deferred.swap_remove(i);
                merge_release(free, deferred.0, deferred.1);
            }
        }
    }
}

fn ranges_overlap(a: (u32, u32), b: (u32, u32)) -> bool {
    a.0 < b.0.saturating_add(b.1) && b.0 < a.0.saturating_add(a.1)
}

struct PublishLease {
    sampler: bool,
    range: (u32, u32),
}

struct PublishRangeLease {
    sampler: bool,
    ranges: Vec<(u32, u32)>,
}

impl Drop for PublishRangeLease {
    fn drop(&mut self) {
        if self.sampler {
            if let Some(global) = GLOBAL_SAMPLER.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                global.leases.finish_ranges(&mut global.free, &self.ranges);
            }
        } else if let Some(global) = GLOBAL.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            global.leases.finish_ranges(&mut global.free, &self.ranges);
        }
    }
}

impl Drop for PublishLease {
    fn drop(&mut self) {
        if self.sampler {
            if let Some(global) = GLOBAL_SAMPLER.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                global.leases.finish(&mut global.free, self.range);
            }
        } else if let Some(global) = GLOBAL.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            global.leases.finish(&mut global.free, self.range);
        }
    }
}

/// Released descriptor range in `GLOBAL` or `GLOBAL_SAMPLER` awaiting active GPU queues before
/// being returned to `free` (prevents overwriting shader-visible descriptors still read in flight).
struct RetiredRange<F> {
    first: u32,
    count: u32,
    /// Fence marks across all live queues on the device (`None` = signal failed, keep held).
    marks: Option<Vec<(F, u64)>>,
}

/// Pop retired ranges whose queue fences have all completed. A `None` entry (failed signal) stays
/// quarantined without wedging subsequent healthy ranges.
fn pop_ready_ranges<F>(
    retired: &mut VecDeque<RetiredRange<F>>,
    mut completed: impl FnMut(&F) -> u64,
) -> Vec<(u32, u32)> {
    let mut ready = Vec::new();
    let mut kept = VecDeque::new();
    while let Some(r) = retired.pop_front() {
        match r.marks.as_ref() {
            Some(m) if m.iter().all(|(f, v)| completed(f) >= *v) => ready.push((r.first, r.count)),
            None => kept.push_back(r),
            Some(_) => {
                kept.push_back(r);
                break;
            }
        }
    }
    while let Some(r) = kept.pop_back() {
        retired.push_front(r);
    }
    ready
}

struct QueueFences(HashMap<usize, (ID3D12Fence, u64)>);

// D3D12 fences are free-threaded; accessed only under HEAP_FENCES lock.
unsafe impl Send for QueueFences {}

static HEAP_FENCES: Mutex<Option<QueueFences>> = Mutex::new(None);
static GLOBAL_RETIRED: AtomicBool = AtomicBool::new(false);
/// Serialises `drain_global_retired` across `fill_nulls` and `merge_release` so a concurrent
/// `slot()` cannot miss ranges popped from `GLOBAL.retired` before they reach `GLOBAL.free`.
/// Lock order: `GLOBAL_DRAIN` -> `SLOT_WRITES` -> `GLOBAL` (never held alongside `HEAPS`).
static GLOBAL_DRAIN: Mutex<()> = Mutex::new(());
/// Incremented under `GLOBAL` whenever descriptors are merged back into `GLOBAL.free`.
static GLOBAL_DRAIN_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Signal every known live title queue belonging to `dev` (called outside `GLOBAL`,
/// `GLOBAL_SAMPLER`, `HEAPS`, and `SLOT_WRITES`). Returns `Some(vec![])` when no live queues
/// exist for `dev`, or `None` if fence creation or signaling fails so released descriptor ranges
/// stay safely quarantined.
unsafe fn signal_live_queues_for(dev: &ID3D12Device) -> Option<Vec<(ID3D12Fence, u64)>> {
    let dev_raw = dev.as_raw();
    let mut queues: Vec<(usize, ID3D12CommandQueue)> = Vec::new();
    for q in super::present::known_queues_held() {
        let mut qdev: Option<ID3D12Device> = None;
        if q.GetDevice(&mut qdev).is_ok() && qdev.as_ref().is_some_and(|d| d.as_raw() == dev_raw) {
            queues.push((q.as_raw() as usize, q));
        }
    }
    if queues.is_empty() {
        return Some(Vec::new());
    }
    let mut guard = HEAP_FENCES.lock().unwrap();
    let map = &mut guard.get_or_insert_with(|| QueueFences(HashMap::new())).0;
    let mut marks = Vec::with_capacity(queues.len());
    for (key, q) in queues {
        let (fence, val) = match map.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let f: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                e.insert((f, 0))
            }
        };
        let next = *val + 1;
        if super::real::native_queue_signal(&q, &*fence, next).is_err() {
            return None;
        }
        *val = next;
        marks.push((fence.clone(), next));
    }
    Some(marks)
}

/// The shader-visible PC heap of all console view heaps.
struct Global {
    dev: ID3D12Device,
    heap: ID3D12DescriptorHeap,
    cpu: u64,
    gpu: u64,
    inc: u32,
    cap: u32,
    next: u32,
    /// released ranges (first, count) proven GPU-idle
    free: Vec<(u32, u32)>,
    /// released ranges awaiting live GPU queue completion before nulling and entering `free`
    retired: VecDeque<RetiredRange<ID3D12Fence>>,
    /// a CPU-only heap of `NULL_BLOCK` null views (even SRV, odd UAV) and its start: copied over
    /// every range of the shared heap that is not handed out (`fill_nulls`)
    nulls: Option<(ID3D12DescriptorHeap, u64)>,
    leases: PublishLeases,
}

// D3D12 descriptor heaps and devices are free-threaded
unsafe impl Send for Global {}

static GLOBAL: Mutex<Option<Global>> = Mutex::new(None);
static GLOBAL_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// descriptors of the shared heap at most (the GPU may take more: OPTIONS19)
pub const GLOBAL_MAX: u32 = 1_000_000;
// (one million: what every resource binding tier 2+ GPU takes. The translated shaders accept a
// T# index only below it (`gpu::dxil::lower_image::VIEW_HEAP_MAX_SRV`), so the heap must never be
// smaller than that bound: an index past the heap's end hangs NVIDIA GPUs, which took 1 000 000
// while the bound was 2^21 – a raw console T# of a texture at 256–512 MB passed it.)

/// null views in the CPU-only heap the shared heap's free ranges are filled from
const NULL_BLOCK: u32 = 4096;

/// Copy null views (`nulls`, a CPU-only heap of `NULL_BLOCK` of them, even SRV, odd UAV) over
/// descriptors `from..to` of the heap at `cpu`, a block at a time; an odd `from` starts from
/// the block's first UAV, so every even descriptor stays an SRV and every odd one a UAV, as the
/// SRV / UAV pair of a console slot.
unsafe fn fill_nulls(dev: &ID3D12Device, cpu: u64, inc: u32, from: u32, to: u32, nulls: u64) {
    let mut at = from;
    while at < to {
        let skew = at % 2;
        let n = (NULL_BLOCK - skew).min(to - at);
        copy_views(dev, n, cpu + at as u64 * inc as u64, nulls + skew as u64 * inc as u64);
        at += n;
    }
}

/// CopyDescriptorsSimple of `n` view descriptors, past XCL's hook of an adopted device (which
/// looks up the console heaps for each call: `HEAPS`, then `GLOBAL`), so that it can be made
/// with `SLOT_WRITES` held.
unsafe fn copy_views(dev: &ID3D12Device, n: u32, dst: u64, src: u64) {
    let raw = dev.as_raw();
    if super::real::is_adopted(raw) {
        let f: unsafe extern "system" fn(*mut std::ffi::c_void, u32, u64, u64, u32) = std::mem::transmute(super::real::orig_fn(raw, 24));
        return f(raw, n, dst, src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
    }
    dev.CopyDescriptorsSimple(n, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: dst as usize }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: src as usize }, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

/// Which PC resource the descriptor at each index of the shared heap names, and the other way
/// round – so that the views of a resource that is destroyed can be made null views.
///
/// A title keeps T#s (each with an index in the shared heap) in its own memory, and its shaders
/// read them from there. On the console a T# of a released texture still names memory (leftover
/// bytes, nothing faults), and a title can sample a texture it has released (a material table
/// not updated yet, a sample of weight zero). On the PC the slot would keep a descriptor of a
/// destroyed resource: a shader that samples it touches an unmapped address.
struct Names {
    /// index → the resource its descriptor names (0: none, a null view, or not known; `GONE`: the
    /// null view `resource_gone` wrote there); the truth
    slots: Vec<AtomicUsize>,
    /// resource → indexes that named it (some may name another since: `slots` says)
    by: super::sharded::Sharded<Vec<u32>>,
}

impl Names {
    fn new(n: usize) -> Names {
        Names { slots: (0..n).map(|_| AtomicUsize::new(0)).collect(), by: super::sharded::Sharded::new() }
    }

    fn name(&self, i: u32) -> usize {
        self.slots.get(i as usize).map_or(0, |s| s.load(Relaxed))
    }

    /// The descriptor at `i` names `res` now (0: none; `GONE`: it was made the null view of a
    /// destroyed resource's). With `SLOT_WRITES` held, along with the write of the descriptor.
    fn note(&self, i: u32, res: usize) {
        let Some(s) = self.slots.get(i as usize) else { return };
        if s.swap(res, Relaxed) == res || res == 0 || res == GONE {
            return;
        }
        self.by.with(res, |v| {
            v.push(i);
            // a resource whose views are made again and again: only the indexes that still name it
            if v.len() >= 64 && v.len().is_power_of_two() {
                v.retain(|&j| self.name(j) == res);
                v.sort_unstable();
                v.dedup();
            }
        });
    }

    fn clear(&self, from: u32, to: u32) {
        for i in from..to {
            if let Some(s) = self.slots.get(i as usize) {
                s.store(0, Relaxed);
            }
        }
    }

    /// The indexes whose descriptor may name `res` (None: none ever did – the common case, one
    /// lookup), forgotten here.
    fn candidates(&self, res: usize) -> Option<Vec<u32>> {
        self.by.remove(res)
    }

    /// Whether the descriptor at `i` still names `res`: then it is `GONE` from now on (the caller
    /// writes a null view there, with `SLOT_WRITES` held). One overwritten since names another
    /// resource and stays.
    fn claim(&self, i: u32, res: usize) -> bool {
        self.slots.get(i as usize).is_some_and(|s| s.compare_exchange(res, GONE, Relaxed, Relaxed).is_ok())
    }
}

/// What `Names` (and `names_of`) give for a descriptor `resource_gone` made a null view. The
/// title's own CPU-only mirror slot still holds the destroyed resource's view there, so a copy
/// of that slot writes the null view in the shared heap instead of the mirror's (`publish_range`).
pub const GONE: usize = usize::MAX;

/// (made with the first view written to the shared heap: 8 bytes an index)
static NAMES: std::sync::OnceLock<Names> = std::sync::OnceLock::new();

/// Writes of descriptors into the shared heap and of what they name (`NAMES`), one at a time:
/// a null view for a destroyed resource must not come over a view another thread just wrote.
/// Taken before `GLOBAL` or `HEAPS`, never with them held.
static SLOT_WRITES: Mutex<()> = Mutex::new(());

fn names() -> &'static Names {
    NAMES.get_or_init(|| Names::new(GLOBAL_MAX as usize))
}

/// What null views are copied from: the device, the shared heap's start and increment, the
/// null views' start (None: no shared heap, or no null views).
fn null_source() -> Option<(ID3D12Device, u64, u32, u64)> {
    GLOBAL.lock().unwrap().as_ref().and_then(|s| s.nulls.as_ref().map(|n| (s.dev.clone(), s.cpu, s.inc, n.1)))
}

/// The PC resource `res` is destroyed next: every descriptor of the shared heap that still names
/// it becomes the null view of its kind (SRV at even indexes, UAV at odd). The number of them.
/// A resource that never had a view in the shared heap costs one lookup.
pub fn resource_gone(res: usize) -> usize {
    let Some(n) = NAMES.get() else { return 0 };
    let Some(candidates) = n.candidates(res) else { return 0 };
    let Some((dev, cpu, inc, nulls)) = null_source() else { return 0 };
    let _w = SLOT_WRITES.lock().unwrap();
    let mut nulled = 0;
    for i in candidates {
        if n.claim(i, res) {
            unsafe { fill_nulls(&dev, cpu, inc, i, i + 1, nulls) };
            nulled += 1;
        }
    }
    nulled
}

/// view slots per page of a heap's place in the shared heap (2 PC descriptors each)
pub const PAGE_SLOTS: u32 = 256;
/// a page with no place in the shared heap yet
const UNMAPPED: u32 = u32::MAX;

/// PC descriptors of page `p` of a heap of `n` slots: 2 per slot, and the last page only as
/// long as the heap (a heap of a few slots takes a few descriptors, not a whole page's).
fn page_descriptors(n: u32, p: u32) -> u32 {
    2 * n.saturating_sub(p * PAGE_SLOTS).clamp(1, PAGE_SLOTS)
}

/// A first-fit range of `count` in `free` (released ranges), else at `next` below `cap`. A freed
/// range ending at the cursor is contiguous with the tail: take it and bump over into the tail.
fn take_range(free: &mut Vec<(u32, u32)>, next: &mut u32, cap: u32, count: u32) -> Option<u32> {
    if let Some(i) = free.iter().position(|&(_, n)| n >= count) {
        let (f, n) = free[i];
        if n == count {
            free.swap_remove(i);
        } else {
            free[i] = (f + count, n - count);
        }
        return Some(f);
    }
    // no single freed range is big enough: one ending at the cursor continues into the tail
    if let Some(i) = free.iter().position(|&(f, n)| f + n == *next && (n as u64 + (cap - *next) as u64) >= count as u64) {
        let (f, n) = free[i];
        free.swap_remove(i);
        *next += count - n;
        return Some(f);
    }
    if cap - *next < count {
        return None;
    }
    *next += count;
    Some(*next - count)
}

/// Descriptors of the shared heap on `dev`: what it takes (OPTIONS19; 1 000 000 without it), at
/// most `GLOBAL_MAX`.
unsafe fn global_cap(dev: &ID3D12Device) -> u32 {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
    let max = if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &mut o as *mut _ as *mut std::ffi::c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.MaxViewDescriptorHeapSize > 0 {
        o.MaxViewDescriptorHeapSize
    } else {
        1_000_000
    };
    max.min(GLOBAL_MAX)
}

/// The shared shader-visible heap, made on first use. False: there is none (it could not be
/// made) – view heaps then keep their own.
pub unsafe fn shared_heap(dev: &ID3D12Device) -> bool {
    {
        let g = GLOBAL.lock().unwrap();
        if g.is_some() {
            return true;
        }
        if GLOBAL_FAILED.load(Relaxed) {
            return false;
        }
    }
    // Made and filled without `GLOBAL` held: the device may be adopted, and its descriptor hooks
    // look up the console heaps (`HEAPS`, then `GLOBAL`). Two threads that get here at once both
    // make one; the first to store it wins.
    let cap = global_cap(dev);
    let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: cap, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
    let heap = match dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&d) {
        Ok(h) => h,
        Err(e) => {
            GLOBAL_FAILED.store(true, Relaxed);
            crate::xlog_warn!("console descriptor heaps: no shared shader-visible heap of {cap} ({e}) – each keeps its own");
            return false;
        }
    };
    let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
    let start = std::time::Instant::now();
    null_views(dev, cpu, inc);
    null_window_views(dev, cpu, inc);
    // D3D12 leaves a new heap's descriptors undefined: every one XCL has not handed out holds a
    // null view, so an index a shader takes from a stale or garbage T# reads zeros, never a
    // descriptor of whatever the memory held (the memory windows' views come over theirs next)
    let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: NULL_BLOCK, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 };
    let nulls = dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&d).ok().map(|h| {
        let at = h.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        null_views_in(dev, at, inc, NULL_BLOCK as u64);
        fill_nulls(dev, cpu, inc, crate::gpu::recomp::MEM_HEAP_BASE, cap, at);
        (h, at)
    });
    crate::xlog_info!(
        "console descriptor heaps: one shader-visible PC heap of {cap} view descriptors for all of them ({})",
        if nulls.is_some() { format!("null views in the rest, {} ms", start.elapsed().as_millis()) } else { "no null views past the first: no CPU-only heap for them".into() }
    );
    if cap < GLOBAL_MAX {
        crate::xlog_warn!(
            "console descriptor heaps: the shared view heap has {cap} descriptors, fewer than the {GLOBAL_MAX} translated shaders allow for: a texture whose T# index is from {cap} to {} reads past the heap's end (their guard does not cover that)",
            GLOBAL_MAX - 1
        );
    }
    let mut g = GLOBAL.lock().unwrap();
    if g.is_none() {
        *g = Some(Global {
            dev: dev.clone(),
            cpu,
            gpu: heap.GetGPUDescriptorHandleForHeapStart().ptr,
            inc,
            heap,
            cap,
            // the memory windows and their two out-of-range null UAVs come first
            next: crate::gpu::recomp::MEM_NULL_STORE + 1,
            free: Vec::new(),
            retired: VecDeque::new(),
            nulls,
            leases: PublishLeases::default(),
        });
    }
    true
}

/// Null views at descriptors `0..MEM_HEAP_BASE` of the shared heap (an SRV / UAV pair per
/// console slot: even SRV, odd UAV), where shaders land for a T# of zeros (word 0, the PC index,
/// is 0): they read zeros there rather than whatever an unwritten descriptor holds.
unsafe fn null_views(dev: &ID3D12Device, cpu: u64, inc: u32) {
    null_views_in(dev, cpu, inc, crate::gpu::recomp::MEM_HEAP_BASE as u64)
}

/// `n` null views from `cpu`: even SRV, odd UAV (`null_views`).
unsafe fn null_views_in(dev: &ID3D12Device, cpu: u64, inc: u32, n: u64) {
    for k in 0..n {
        let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + k * inc as u64) as usize };
        if k % 2 == 0 {
            let d = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
            };
            dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&d), at);
        } else {
            let d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV::default() },
            };
            dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&d), at);
        }
    }
}

/// Null raw UAVs used by heap-window accessors for out-of-layout loads and stores.
unsafe fn null_window_views(dev: &ID3D12Device, cpu: u64, inc: u32) {
    for idx in [crate::gpu::recomp::MEM_NULL_LOAD, crate::gpu::recomp::MEM_NULL_STORE] {
        let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + idx as u64 * inc as u64) as usize };
        dev.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&crate::gpu::recomp::runner::raw_window_uav(4)),
            at,
        );
    }
}

static WINDOW_VIEWS_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the memory windows could not be put in the shared heap (`set_window_views`):
/// translated shaders then switch over root UAVs (`pso::heap_windows`).
pub fn window_views_failed() -> bool {
    WINDOW_VIEWS_FAILED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The memory windows' buffers `(resource, bytes)` as raw UAVs at `MEM_HEAP_BASE + k` of the
/// shared heap, made now if need be (`Options::heap_windows`: shaders take window k from there).
/// Active windows (`bytes > 0`) in wider layouts (8 / 16) must all have a valid buffer; a slot
/// without one (`(0, 0)`: a window with no memory) gets a null raw UAV – shaders read 0 there and
/// their stores go nowhere. (It once aliased window 0: what was at such an address read window
/// 0's bytes, and GPU writes there overwrote its descriptor heaps.)
pub unsafe fn set_window_views(dev: &ID3D12Device, windows: &[(usize, usize)]) -> bool {
    let count = windows.len().max(crate::gpu::uma::MAX_WINDOWS);
    let complete = count <= crate::gpu::uma::MAX_WINDOWS || windows.iter().all(|&(raw, bytes)| raw != 0 || bytes == 0);
    let ok = count <= crate::gpu::uma::MAX_HEAP_WINDOWS && complete && windows.first().is_some_and(|w| w.0 != 0) && shared_heap(dev) && {
        let (device, cpu, inc) = {
            let g = GLOBAL.lock().unwrap();
            let s = g.as_ref().unwrap();
            (s.dev.clone(), s.cpu, s.inc)
        };
        // The device may be adopted: its UAV hook re-enters descriptor bookkeeping.
        for k in 0..count {
            let (raw, bytes) = windows.get(k).copied().unwrap_or((0, 0));
            let raw = raw as *mut std::ffi::c_void;
            let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + (crate::gpu::recomp::MEM_HEAP_BASE as u64 + k as u64) * inc as u64) as usize };
            match ID3D12Resource::from_raw_borrowed(&raw) {
                Some(r) => device.CreateUnorderedAccessView(r, None::<&ID3D12Resource>, Some(&crate::gpu::recomp::runner::raw_window_uav(bytes as u64)), at),
                None => device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&crate::gpu::recomp::runner::raw_window_uav(crate::gpu::uma::MAX_SIZE as u64)),
                    at,
                ),
            }
        }
        true
    };
    if !ok {
        WINDOW_VIEWS_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
        crate::xlog_warn!("memory windows: could not bind {count} window(s) in the shared descriptor heap");
    }
    ok
}

/// `count` descriptors of the shared heap: the index of the first. None: no shared heap, or
/// no room left in it.
fn reserve_global(count: u32) -> Option<u32> {
    let mut g = GLOBAL.lock().unwrap();
    let s = g.as_mut()?;
    let (cap, mut next) = (s.cap, s.next);
    let r = take_range(&mut s.free, &mut next, cap, count);
    s.next = next;
    let used = in_use(s.next, &s.free);
    if r.is_some() {
        let placed = PAGES_PLACED.fetch_add(1, Relaxed) + 1;
        let marks = marks_passed(used, cap);
        if marks > FILL_SAID.fetch_max(marks, Relaxed) {
            crate::xlog_info!("console descriptor heaps: the shared heap is past {} % full: {used} of its {cap} descriptors in use, {placed} pages placed", FILL_MARKS[marks - 1]);
        }
    } else {
        let n = NO_ROOM.fetch_add(1, Relaxed) + 1;
        if n.is_power_of_two() {
            crate::xlog_warn!(
                "console descriptor heaps: the shared heap ({cap}) is full – a page of slots found no place there ({n} times so far; {used} descriptors in use, {} pages placed): those slots name the null views at 0 / 1, shaders see none of their views",
                PAGES_PLACED.load(Relaxed)
            );
        }
    }
    r
}

/// Pages that asked for a place in the shared heap and found no room (each try counts).
static NO_ROOM: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Pages of console heaps with a place in the shared heap now.
static PAGES_PLACED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// How many of `FILL_MARKS` the log has said the shared heap passed.
static FILL_SAID: AtomicUsize = AtomicUsize::new(0);

/// How full the shared heap is said to be in the log (percent), each once, the first time.
const FILL_MARKS: [u32; 3] = [50, 75, 90];

/// Descriptors of the shared heap handed out: those below the cursor `next` but the released
/// ranges in `free` (the null views and the memory windows at its start count).
fn in_use(next: u32, free: &[(u32, u32)]) -> u32 {
    next.saturating_sub(free.iter().map(|&(_, n)| n).sum())
}

/// How many of `FILL_MARKS` `used` descriptors of a shared heap of `cap` are past.
fn marks_passed(used: u32, cap: u32) -> usize {
    FILL_MARKS.iter().filter(|&&m| used as u64 * 100 > m as u64 * cap as u64).count()
}

/// The place in the shared heap of the page slot `k` of `h` is in – (index of the page's
/// first SRV there, the page's first slot) – given now if the page has none yet. None: the
/// heap is not in the shared heap, or that is full.
fn page_of(h: &mut Heap, k: u32) -> Option<(u32, u32)> {
    let p = (k / PAGE_SLOTS) as usize;
    let mut first = *h.pages.get(p)?;
    if first == UNMAPPED {
        first = reserve_global(page_descriptors(h.n, p as u32))?;
        h.pages[p] = first;
    }
    Some((first, p as u32 * PAGE_SLOTS))
}

/// Move completed `GLOBAL.retired` ranges to `GLOBAL.free`, nulling their view descriptors and
/// clearing `NAMES` under `SLOT_WRITES` before another heap can allocate them. Must be called
/// outside `SLOT_WRITES`, `HEAPS`, and `GLOBAL`.
fn drain_global_retired() -> bool {
    drain_global_retired_since(GLOBAL_DRAIN_SEQ.load(Acquire))
}

fn drain_global_retired_since(before_seq: u64) -> bool {
    drain_global_retired_with(before_seq, |dev, cpu, inc, nulls, ready| {
        let _w = SLOT_WRITES.lock().unwrap();
        for &(first, count) in ready {
            unsafe { fill_nulls(dev, cpu, inc, first, first + count, nulls) };
            if let Some(n) = NAMES.get() {
                n.clear(first, first + count);
            }
        }
    })
}

fn drain_global_retired_with(
    before_seq: u64,
    on_nulls: impl FnOnce(&ID3D12Device, u64, u32, u64, &[(u32, u32)]),
) -> bool {
    if !GLOBAL_RETIRED.load(Acquire) {
        return GLOBAL_DRAIN_SEQ.load(Acquire) != before_seq;
    }
    let _drain = GLOBAL_DRAIN.lock().unwrap();
    let (ready, nulls_info) = {
        let mut g = GLOBAL.lock().unwrap();
        let Some(s) = g.as_mut() else {
            GLOBAL_RETIRED.store(false, Release);
            return GLOBAL_DRAIN_SEQ.load(Acquire) != before_seq;
        };
        let ready = pop_ready_ranges(&mut s.retired, |f| unsafe { super::real::fence_completed(f) });
        if ready.is_empty() {
            GLOBAL_RETIRED.store(!s.retired.is_empty(), Release);
            return GLOBAL_DRAIN_SEQ.load(Acquire) != before_seq;
        }
        // Keep `GLOBAL_RETIRED` true across `fill_nulls` until `merge_release` completes.
        let nulls_info = s.nulls.as_ref().map(|n| (s.dev.clone(), s.cpu, s.inc, n.1));
        (ready, nulls_info)
    };
    if let Some((dev, cpu, inc, nulls)) = nulls_info {
        on_nulls(&dev, cpu, inc, nulls, &ready);
    }
    if let Some(s) = GLOBAL.lock().unwrap().as_mut() {
        for (first, count) in ready {
            s.leases.release_range(&mut s.free, first, count);
        }
        GLOBAL_DRAIN_SEQ.fetch_add(1, Release);
        GLOBAL_RETIRED.store(!s.retired.is_empty(), Release);
    } else {
        GLOBAL_RETIRED.store(false, Release);
    }
    true
}

/// The shared range starting at `first` (`count` descriptors) is quarantined behind active GPU
/// queues before its views are made null views and it returns to `free`, preventing overwrites
/// of live shader-visible descriptors.
pub fn release_global(first: u32, count: u32) {
    if count == 0 {
        return;
    }
    let dev = GLOBAL.lock().unwrap().as_ref().map(|s| s.dev.clone());
    let Some(dev) = dev else { return };
    let marks = unsafe { signal_live_queues_for(&dev) };
    if marks.as_ref().is_some_and(Vec::is_empty) {
        if let Some((dev, cpu, inc, nulls)) = null_source() {
            // (the range is still the released heap's)
            let _w = SLOT_WRITES.lock().unwrap();
            unsafe { fill_nulls(&dev, cpu, inc, first, first + count, nulls) };
            if let Some(n) = NAMES.get() {
                n.clear(first, first + count);
            }
        }
        if let Some(s) = GLOBAL.lock().unwrap().as_mut() {
            s.leases.release_range(&mut s.free, first, count);
            GLOBAL_DRAIN_SEQ.fetch_add(1, Release);
        }
        drain_global_retired();
        return;
    }
    if let Some(s) = GLOBAL.lock().unwrap().as_mut() {
        s.retired.push_back(RetiredRange { first, count, marks });
        GLOBAL_RETIRED.store(true, Release);
    }
    drain_global_retired();
}

/// Add a released range to `free`, merging the ranges it touches: adjacent releases become
/// one range, so first-fit (and the cursor-straddle in `take_range`) sees the whole run.
fn merge_release(free: &mut Vec<(u32, u32)>, first: u32, count: u32) {
    free.push((first, count));
    free.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(free.len());
    for (f, n) in free.drain(..) {
        match out.last_mut() {
            Some(last) if f <= last.0 + last.1 => last.1 = last.1.max(f + n - last.0),
            _ => out.push((f, n)),
        }
    }
    *free = out;
}

/// The shader-visible PC heap lists bind for view descriptors (None: none shared).
pub fn global_heap() -> Option<ID3D12DescriptorHeap> {
    GLOBAL.lock().unwrap().as_ref().map(|s| s.heap.clone())
}

/// Whether `obj` (a title's view heap) has its descriptors in the shared heap.
pub fn in_global(obj: usize) -> bool {
    HEAPS.lock().unwrap().iter().any(|h| h.obj == obj && !h.sampler && !h.pages.is_empty())
}

/// Memory for a heap of `bytes` from the title's allocation hook – where the title expects the
/// heap then. Only what descriptor tables can address (window 0) and is committed; anything else
/// goes back to the title.
unsafe fn title_memory(u: &crate::gpu::uma::Uma, bytes: usize) -> Option<usize> {
    use crate::hle::kernelx::{runtime_alloc, runtime_free, D3D_DESCRIPTOR_HEAP};
    let p = runtime_alloc(bytes, D3D_DESCRIPTOR_HEAP)?;
    let readable = crate::gpu::shader::readable(p, bytes);
    if title_memory_fits(u, p, bytes, readable) {
        return Some(p);
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
        let (free, largest) = u.sparse_free_in(0);
        crate::xlog_warn!(
            "console descriptor heap: the title's memory {p:#x} ({bytes:#x} bytes) not used (window {:?}, committed {readable}; window 0 has {} MB free, {} MB at most in one run)",
            u.window_of(p),
            free >> 20,
            largest >> 20
        );
    }
    runtime_free(p, D3D_DESCRIPTOR_HEAP);
    None
}

/// Whether the title's `[p, p + bytes)` can hold a heap's slots: committed (`readable`) and all
/// in window 0. A descriptor table is a 32-bit address, high half 0, which shaders and
/// `words_at` read in window 0 – at `base + (p as u32)`. In another window the tables would name
/// other bytes, and the substitute heap the title then gets does not start at its memory (a
/// retail game asserts that it does).
fn title_memory_fits(u: &crate::gpu::uma::Uma, p: usize, bytes: usize, readable: bool) -> bool {
    readable && bytes > 0 && u.window_of(p) == Some(0) && p.checked_add(bytes - 1).is_some_and(|e| u.window_of(e) == Some(0))
}

/// PC samplers of a console sampler heap of `n` slots: a shader-visible one gets the 2 048 a
/// shader-visible PC sampler heap holds at most, whatever `n` is. Translated shaders index the
/// bound sampler heap with S# word 3 [10:0] (0..2047), and a title builds S#s itself (a border
/// color pointer there): with only `n` PC samplers an index from `n` on read past the heap's end.
/// The samplers past `n` are the zero sampler; the console heap keeps its `n` slots.
pub fn pc_samplers(n: u32, visible: bool) -> u32 {
    if visible {
        n.max(D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE)
    } else {
        n
    }
}

/// `obj` is a PC heap from `pc_cpu` / `pc_gpu` (increment `inc`; `pc_gpu` 0 when not shader
/// visible) with 2·`n` view descriptors or `n` samplers: give it `n` console slots in
/// unified memory. `shared`: its views go to the shared heap (`shared_heap`), a page at a
/// time as slots are used; `visible`: what the title asked for.
pub fn register(obj: usize, n: u32, pc_cpu: u64, pc_gpu: u64, inc: u32, sampler: bool, shared: bool, visible: bool) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    let bytes = n.max(1) * if sampler { SAMPLER_SLOT } else { SLOT };
    // Descriptor tables are 32-bit console addresses (high dword 0): must live in window 0.
    // On the sparse-window path, `sparse::alloc_in(0, ...)` commits fresh zeroed pages so we
    // avoid touching unused pages with `write_bytes` (which would mark them dirty for write-watch).
    let (p, memory) = if let Some(p) = unsafe { title_memory(u, bytes as usize) } {
        (p as *mut u8, Memory::Title)
    } else if u.is_sparse_window(0) {
        let Some((addr, _)) = (unsafe { super::sparse::alloc_in(0, bytes as usize, 256) }) else {
            return false;
        };
        (addr as *mut u8, Memory::Sparse)
    } else {
        let Some(p) = u.alloc_in(0, bytes as usize, 256) else { return false };
        unsafe { std::ptr::write_bytes(p, 0, bytes as usize) };
        (p, Memory::Uma)
    };
    {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
            crate::xlog_info!(
                "console descriptor heap {obj:#x}: {n} {} slots at {} {:#x}, {}",
                if sampler { "sampler" } else { "view" },
                match memory {
                    Memory::Sparse => "sparse VRAM",
                    Memory::Uma => "UMA",
                    Memory::Title => "the title's memory",
                },
                p as usize - u.base(),
                if visible { "shader visible" } else { "CPU only" }
            );
        }
    }
    let pages = if shared && !sampler && GLOBAL.lock().unwrap().is_some() {
        vec![UNMAPPED; n.div_ceil(PAGE_SLOTS).max(1) as usize]
    } else if shared && sampler && GLOBAL_SAMPLER.lock().unwrap().is_some() {
        vec![UNMAPPED; n.div_ceil(SAMPLER_PAGE_SLOTS).max(1) as usize]
    } else {
        Vec::new()
    };
    let generation = NEXT_HEAP_GENERATION.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    HEAPS.lock().unwrap().push(Heap { generation, obj, shadow: p as usize, n, pc_cpu, pc_gpu, inc, sampler, pages, visible, memory });
    true
}

/// The heap object was released for good: its slots go back to unified memory.
pub fn unregister(obj: usize) {
    if let Some(h) = take(obj) {
        free(h);
    }
}

/// A heap removed by `take`; its backing storage remains live until `free`.
pub struct Taken(Heap);

/// Stop finding a heap by object address without freeing memory the object may still use.
pub fn take(obj: usize) -> Option<Taken> {
    let heap = detach(obj)?;
    super::vram::forget_srv_heap(heap.generation);
    Some(Taken(heap))
}

fn detach(obj: usize) -> Option<Heap> {
    let mut heaps = HEAPS.lock().unwrap();
    heaps.iter().position(|h| h.obj == obj).map(|i| heaps.swap_remove(i))
}

pub(super) fn generation(obj: usize) -> Option<u64> {
    HEAPS.lock().unwrap().iter().find(|heap| heap.obj == obj).map(|heap| heap.generation)
}

/// Detach bookkeeping before the native object can die. The caller holds `HEAP_LIFETIMES`, so
/// cleanup must not wait for the descriptor-write gate (a reentrant temporary Release may hold
/// that gate while waiting here).
pub(super) fn take_after_release(obj: usize, generation: Option<u64>, lifetime: Option<&HeapLifetimeGuard>) -> Option<Taken> {
    let mut heaps = HEAPS.lock().unwrap();
    let heap = heaps.iter().position(|h| h.obj == obj && Some(h.generation) == generation).map(|i| heaps.swap_remove(i))?;
    drop(heaps);
    super::vram::forget_srv_heap_after_release(heap.generation, lifetime);
    Some(Taken(heap))
}

/// Return a taken heap's slots and shadow memory after its native object is gone.
pub fn free(Taken(h): Taken) {
    for (p, &f) in h.pages.iter().enumerate().filter(|&(_, &f)| f != UNMAPPED) {
        if h.sampler {
            release_global_sampler(f, sampler_page_descriptors(h.n, p as u32));
        } else {
            release_global(f, page_descriptors(h.n, p as u32));
            PAGES_PLACED.fetch_sub(1, Relaxed);
        }
    }
    match h.memory {
        Memory::Sparse => {
            unsafe { super::sparse::free(h.shadow) };
        }
        Memory::Uma => {
            if let Some(u) = crate::gpu::uma::get() {
                u.free(h.shadow);
            }
        }
        Memory::Title => unsafe { crate::hle::kernelx::runtime_free(h.shadow, crate::hle::kernelx::D3D_DESCRIPTOR_HEAP) },
    }
}

/// Whether heap `obj` holds samplers, and the 32-bit console address of its slots.
pub fn base(obj: usize) -> Option<(bool, u32)> {
    HEAPS.lock().unwrap().iter().find(|h| h.obj == obj).map(|h| (h.sampler, h.shadow as u32))
}

/// Start of the console slots of heap `obj` (CPU handle = GPU handle on the console).
pub fn start(obj: usize) -> Option<u64> {
    HEAPS.lock().unwrap().iter().find(|h| h.obj == obj).map(|h| h.shadow as u64)
}

/// A console slot: where its descriptor goes and its PC descriptors.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub(super) generation: u64,
    pub at: usize,
    /// views: PC heap index of the SRV (the UAV is at + 1); samplers: of the sampler
    pub index: u32,
    /// PC CPU handles of the SRV / UAV where views are made (the title's heap)
    pub pc_srv: u64,
    pub pc_uav: u64,
    /// PC GPU handle of the UAV (0: heap not shader visible)
    pub pc_gpu_uav: u64,
    pub sampler: bool,
    /// CPU handle of the SRV in the shared heap (0: the heap is not in it)
    pub shared_srv: u64,
    /// the slot's number in its heap
    pub k: u32,
    /// slots from this one through the end of its heap
    pub remaining: u32,
}

/// Strong ownership and the exact handles of one view slot. Holding this value only proves CPU
/// object lifetime; a consumer must separately prove queue quiescence, take the descriptor-write
/// gate, and revalidate the tracked logical generation and words before mutating descriptors.
#[allow(dead_code)]
pub(super) struct ViewSlotLease {
    pub pc_srv: u64,
    pub shared_srv: u64,
    pub index: u32,
    _device: ID3D12Device,
    _source_heap: ID3D12DescriptorHeap,
    _shared_heap: Option<ID3D12DescriptorHeap>,
}

/// Lease the native objects behind one exact heap generation and slot. All arithmetic and both
/// CPU handles are checked while `HEAPS` (and, for a shared slot, `GLOBAL`) still identify that
/// generation. Driver calls happen only after allocator locks have been dropped.
#[allow(dead_code)]
pub(super) fn lease_view_slot(generation: u64, k: u32, pc_srv: u64, shared_srv: u64, index: u32) -> Option<ViewSlotLease> {
    let _lifetime = lock_lifetime()?;
    let (source_raw, shared_index) = {
        let heaps = HEAPS.lock().unwrap();
        let h = heaps.iter().find(|h| h.generation == generation && !h.sampler && k < h.n)?;
        let local_index = k.checked_mul(2)?;
        let expected_pc = h.pc_cpu.checked_add(local_index as u64 * h.inc as u64)?;
        if expected_pc != pc_srv {
            return None;
        }
        let shared_index = if h.pages.is_empty() {
            if shared_srv != 0 || index != local_index {
                return None;
            }
            None
        } else {
            let first = *h.pages.get((k / PAGE_SLOTS) as usize)?;
            if first == UNMAPPED {
                return None;
            }
            Some(first.checked_add(2 * (k % PAGE_SLOTS))?)
        };
        (h.obj as *mut std::ffi::c_void, shared_index)
    };
    let shared_raw = if let Some(expected_index) = shared_index {
        let global = GLOBAL.lock().unwrap();
        let global = global.as_ref()?;
        if expected_index.checked_add(1)? >= global.cap
            || index != expected_index
            || shared_srv != global.cpu.checked_add(expected_index as u64 * global.inc as u64)?
        {
            return None;
        }
        Some((global.dev.as_raw(), global.heap.as_raw()))
    } else {
        None
    };
    // The lifetime gate prevents source final Release, and process-global heaps are never
    // replaced. Invoke COM only after allocator mutexes have been dropped.
    let source_heap = unsafe { ID3D12DescriptorHeap::from_raw_borrowed(&source_raw)?.clone() };
    // The strong source owner now prevents native final Release. Leave the non-reentrant gate
    // before any fallible COM work so an error or unwind can drop that owner through our hooked
    // Release without trying to reacquire a gate this thread still holds.
    drop(_lifetime);
    let (device, shared_heap) = if let Some((device_raw, heap_raw)) = shared_raw {
        let device = unsafe { ID3D12Device::from_raw_borrowed(&device_raw)?.clone() };
        let heap = unsafe { ID3D12DescriptorHeap::from_raw_borrowed(&heap_raw)?.clone() };
        (device, Some(heap))
    } else {
        let mut device = None;
        unsafe { source_heap.GetDevice(&mut device).ok()? };
        (device?, None)
    };
    Some(ViewSlotLease { pc_srv, shared_srv, index, _device: device, _source_heap: source_heap, _shared_heap: shared_heap })
}

fn finish_view_slot(h: &Heap, k: u32, page: Option<(u32, u32)>) -> Slot {
    let uav = ((2 * k + 1) * h.inc) as u64;
    let shared = page.and_then(|(first, k0)| {
        let g = GLOBAL.lock().unwrap();
        let g = g.as_ref()?;
        let i = first + 2 * (k - k0);
        Some((i, g.gpu + (i as u64 + 1) * g.inc as u64, g.cpu + i as u64 * g.inc as u64))
    });
    // A heap meant for the shared heap whose page found no room there: lists bind only the
    // shared heap, where 2k is another heap's view, a memory window's raw UAV (k from 2 to 9) or
    // past the heap's end; its T#s name the null views at 0 / 1 instead, and nothing is copied.
    let own = if h.pages.is_empty() { (2 * k, if h.pc_gpu == 0 { 0 } else { h.pc_gpu + uav }, 0) } else { (0, 0, 0) };
    let (index, pc_gpu_uav, shared_srv) = shared.unwrap_or(own);
    Slot { generation: h.generation, at: h.shadow + (k * SLOT) as usize, index, pc_srv: h.pc_cpu + (2 * k * h.inc) as u64, pc_uav: h.pc_cpu + uav, pc_gpu_uav, sampler: false, shared_srv, k, remaining: h.n - k }
}

/// The slot behind a console CPU handle (None: not one of XCL's console heaps). A view slot
/// of a heap in the shared heap gets its page's place there now if the page has none yet; one
/// whose page finds no room there gets index 0, the null SRV / UAV pair.
pub fn slot(cpu: u64) -> Option<Slot> {
    drain_global_retired();
    let mut heaps = HEAPS.lock().unwrap();
    let idx = heaps.iter().position(|h| cpu >= h.shadow as u64 && cpu < (h.shadow + (h.n * h.slot_bytes()) as usize) as u64)?;
    let k = ((cpu - heaps[idx].shadow as u64) / heaps[idx].slot_bytes() as u64) as u32;
    if heaps[idx].sampler {
        let h = &mut heaps[idx];
        let shared = sampler_page_of(h, k).and_then(|(first, k0)| {
            let g = GLOBAL_SAMPLER.lock().unwrap();
            let g = g.as_ref()?;
            let i = first + (k - k0);
            Some((i, g.cpu + i as u64 * g.inc as u64))
        });
        // If this sampler heap is hosted in `GLOBAL_SAMPLER` (`!h.pages.is_empty()`), an exhausted
        // shared heap must map overflowed slots to reserved zero-sampler slot 0 rather than
        // aliasing unrelated heap entries at local index `k`.
        let (index, shared_srv) = shared.unwrap_or_else(|| (sampler_unmapped_index(!h.pages.is_empty(), k), 0));
        return Some(Slot { generation: h.generation, at: h.shadow + (k * SAMPLER_SLOT) as usize, index, pc_srv: h.pc_cpu + (k * h.inc) as u64, pc_uav: 0, pc_gpu_uav: 0, sampler: true, shared_srv, k, remaining: h.n - k });
    }
    let drain_seq = GLOBAL_DRAIN_SEQ.load(Acquire);
    let page = page_of(&mut heaps[idx], k);
    if page.is_none() && !heaps[idx].pages.is_empty() {
        // Check `GLOBAL_RETIRED` before `GLOBAL_DRAIN_SEQ` (reverse of the Release order in
        // `drain_global_retired_with`): observing `GLOBAL_RETIRED == false` guarantees the
        // preceding `GLOBAL_DRAIN_SEQ` increment is visible.
        if GLOBAL_RETIRED.load(Acquire) {
            drop(heaps);
            let drained = drain_global_retired_since(drain_seq);
            heaps = HEAPS.lock().unwrap();
            let h = heaps.iter_mut().find(|h| cpu >= h.shadow as u64 && cpu < (h.shadow + (h.n * h.slot_bytes()) as usize) as u64)?;
            let page = if drained || GLOBAL_DRAIN_SEQ.load(Acquire) != drain_seq { page_of(h, k) } else { None };
            return Some(finish_view_slot(h, k, page));
        }
        if GLOBAL_DRAIN_SEQ.load(Acquire) != drain_seq {
            let page = page_of(&mut heaps[idx], k);
            return Some(finish_view_slot(&heaps[idx], k, page));
        }
    }
    Some(finish_view_slot(&heaps[idx], k, page))
}

/// The view (or sampler) just made in slot `s` (its SRV, or with `uav` its UAV) of the PC
/// resource `res` (0: a null view; samplers: 0) goes to the shared heap.
pub unsafe fn publish(s: &Slot, uav: bool, res: usize) {
    if s.shared_srv == 0 {
        return;
    }
    // Keep the exact source heap generation alive and lease its shared destination before
    // dropping HEAPS. A concurrent final Release can then detach its books, but cannot free the
    // source object or recycle the shared index until CopyDescriptorsSimple returns.
    let snapshot = (|| {
        let heaps = HEAPS.lock().unwrap();
        let h = heaps.iter().find(|h| {
            h.generation == s.generation
                && h.sampler == s.sampler
                && s.k < h.n
                && s.at == h.shadow + (s.k * h.slot_bytes()) as usize
        })?;
        let raw = h.obj as *mut std::ffi::c_void;
        let source_owner = ID3D12DescriptorHeap::from_raw_borrowed(&raw)?.clone();
        if h.sampler {
            let first = *h.pages.get((s.k / SAMPLER_PAGE_SLOTS) as usize)?;
            if first == UNMAPPED || h.pc_cpu + (s.k * h.inc) as u64 != s.pc_srv {
                return None;
            }
            let idx = first + s.k % SAMPLER_PAGE_SLOTS;
            let mut globals = GLOBAL_SAMPLER.lock().unwrap();
            let global = globals.as_mut()?;
            let dst = global.cpu + idx as u64 * global.inc as u64;
            if dst != s.shared_srv {
                return None;
            }
            global.leases.acquire(idx, 1);
            Some((global.dev.clone(), source_owner, PublishLease { sampler: true, range: (idx, 1) }, dst, s.pc_srv, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER))
        } else {
            let first = *h.pages.get((s.k / PAGE_SLOTS) as usize)?;
            let src = if uav { s.pc_uav } else { s.pc_srv };
            let expected_src = h.pc_cpu + (2 * s.k + uav as u32) as u64 * h.inc as u64;
            if first == UNMAPPED || expected_src != src {
                return None;
            }
            let idx = first + 2 * (s.k % PAGE_SLOTS) + uav as u32;
            let mut globals = GLOBAL.lock().unwrap();
            let global = globals.as_mut()?;
            let dst = global.cpu + idx as u64 * global.inc as u64;
            if dst != s.shared_srv + uav as u64 * global.inc as u64 {
                return None;
            }
            global.leases.acquire(idx, 1);
            Some((global.dev.clone(), source_owner, PublishLease { sampler: false, range: (idx, 1) }, dst, src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV))
        }
    })();
    let Some((dev, _source_owner, _lease, dst, src, ty)) = snapshot else {
        return;
    };
    if ty == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER {
        dev.CopyDescriptorsSimple(1, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: dst as usize }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: src as usize }, ty);
        return;
    }
    // (past the device hook: `SLOT_WRITES` is held, and the lease keeps the index this heap's)
    let _w = SLOT_WRITES.lock().unwrap();
    copy_views(&dev, 1, dst, src);
    names().note(s.index + uav as u32, res);
}

/// The PC resources the SRV and the UAV of view slot `s` name in the shared heap (0: not known,
/// a sampler, or the slot has no place there): what `publish_range` takes for a copy of it.
pub fn names_of(s: &Slot) -> (usize, usize) {
    if s.sampler {
        return (0, 0);
    }
    match (s.shared_srv, NAMES.get()) {
        (0, _) | (_, None) => (0, 0),
        (_, Some(n)) => (n.name(s.index), n.name(s.index + 1)),
    }
}

/// `n` view or sampler slots from `first` were copied (in the title's heap): they go to the shared heap,
/// a run per page (a heap's pages lie apart there), stopping at the title heap's end. `names`: what
/// the source view slots' SRV and UAV name (`names_of`), one pair a slot (none for samplers). A
/// source descriptor that is `GONE` gets the null view in the shared heap, not the copy of the
/// destroyed resource's view the title's heap still holds.
pub unsafe fn publish_range(first: &Slot, n: u32, names: &[(usize, usize)]) {
    if n == 0 || first.shared_srv == 0 {
        return;
    }
    // Resolve every page and lease every destination while HEAPS prevents unregister from
    // detaching this exact generation. The device calls then need none of the allocator locks.
    let snapshot = (|| {
        let mut heaps = HEAPS.lock().unwrap();
        let h = heaps.iter_mut().find(|h| {
            h.generation == first.generation
                && h.sampler == first.sampler
                && first.k < h.n
                && first.at == h.shadow + (first.k * h.slot_bytes()) as usize
        })?;
        let mul = if h.sampler { 1u32 } else { 2u32 };
        if h.pc_cpu + mul as u64 * first.k as u64 * h.inc as u64 != first.pc_srv {
            return None;
        }
        let raw = h.obj as *mut std::ffi::c_void;
        let source_owner = ID3D12DescriptorHeap::from_raw_borrowed(&raw)?.clone();
        let page_slots = if h.sampler { SAMPLER_PAGE_SLOTS } else { PAGE_SLOTS };
        let limit = n.min(first.remaining).min(h.n - first.k);
        let mut runs = Vec::with_capacity(limit.div_ceil(page_slots) as usize);
        let mut done = 0;
        while done < limit {
            let k = first.k + done;
            let run = (page_slots - k % page_slots).min(limit - done);
            let (page, k0) = if h.sampler { sampler_page_of(h, k)? } else { page_of(h, k)? };
            runs.push((page + mul * (k - k0), h.pc_cpu + mul as u64 * k as u64 * h.inc as u64, mul * run));
            done += run;
        }
        if h.sampler {
            let mut globals = GLOBAL_SAMPLER.lock().unwrap();
            let global = globals.as_mut()?;
            if global.cpu + runs.first()?.0 as u64 * global.inc as u64 != first.shared_srv {
                return None;
            }
            for &(at, _, count) in &runs {
                global.leases.acquire(at, count);
            }
            Some((
                global.dev.clone(),
                source_owner,
                PublishRangeLease { sampler: true, ranges: runs.iter().map(|&(at, _, count)| (at, count)).collect() },
                global.cpu,
                global.inc,
                runs,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            ))
        } else {
            let mut globals = GLOBAL.lock().unwrap();
            let global = globals.as_mut()?;
            if global.cpu + runs.first()?.0 as u64 * global.inc as u64 != first.shared_srv {
                return None;
            }
            for &(at, _, count) in &runs {
                global.leases.acquire(at, count);
            }
            Some((
                global.dev.clone(),
                source_owner,
                PublishRangeLease { sampler: false, ranges: runs.iter().map(|&(at, _, count)| (at, count)).collect() },
                global.cpu,
                global.inc,
                runs,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            ))
        }
    })();
    let Some((dev, _source_owner, _lease, cpu, inc, runs, ty)) = snapshot else {
        return;
    };
    if ty == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER {
        for (at, src, count) in runs {
            dev.CopyDescriptorsSimple(
                count,
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + at as u64 * inc as u64) as usize },
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: src as usize },
                ty,
            );
        }
        return;
    }
    // the null views, taken before `SLOT_WRITES` when a source descriptor is `GONE`
    let nulls = if names.iter().any(|&(srv, uav)| srv == GONE || uav == GONE) { null_source() } else { None };
    // (past the device hook: `SLOT_WRITES` is held, and the leases keep the indexes this heap's)
    let _w = SLOT_WRITES.lock().unwrap();
    let m = self::names();
    let mut done = 0usize;
    for (at, src, count) in runs {
        copy_views(&dev, count, cpu + at as u64 * inc as u64, src);
        for j in 0..count / 2 {
            let (srv, uav) = names.get(done + j as usize).copied().unwrap_or((0, 0));
            for (i, res) in [(at + 2 * j, srv), (at + 2 * j + 1, uav)] {
                if let Some((ndev, ncpu, ninc, nat)) = nulls.as_ref().filter(|_| res == GONE) {
                    fill_nulls(ndev, *ncpu, *ninc, i, i + 1, *nat);
                }
                m.note(i, res);
            }
        }
        done += (count / 2) as usize;
    }
}

/// view slot address → the resource its view names (`XCL_TRACE_FRAME`, `XCL_DUMP_TARGETS`)
static RES: Mutex<Option<std::collections::HashMap<usize, usize>>> = Mutex::new(None);

/// The view in slot `s` names `res` (0: none).
pub fn note_resource(s: &Slot, res: usize) {
    if s.sampler || !(super::present::trace_frame() || super::targets::enabled() || super::capture::enabled()) {
        return;
    }
    let mut g = RES.lock().unwrap();
    let m = g.get_or_insert_with(Default::default);
    if res == 0 {
        m.remove(&s.at);
    } else {
        m.insert(s.at, res);
    }
}

/// The resource of the view slot at UMA address `at`.
pub fn resource_at(at: usize) -> Option<usize> {
    RES.lock().unwrap().as_ref()?.get(&at).copied()
}

/// The PC resource the descriptor at index `i` of the shared heap names (0: none or not known;
/// `GONE`: the null view of a destroyed resource) – what a T# with that index in word 0 samples,
/// whichever slot the T# lies in (`XCL_TRACE_FRAME` diagnostics).
pub fn named_by_index(i: u32) -> usize {
    NAMES.get().map_or(0, |n| n.name(i))
}

pub fn write(s: &Slot, words: &[u32]) {
    let n = words.len().min(if s.sampler { 4 } else { 8 });
    if !s.sampler && super::present::trace_frame() {
        let mut old = [0u32; 8];
        unsafe { std::ptr::copy_nonoverlapping(s.at as *const u32, old.as_mut_ptr(), 8) };
        if rewrites_image(&old, &words[..n]) {
            *REWRITES.lock().unwrap().get_or_insert_with(Default::default).entry(s.generation).or_default() += 1;
        }
    }
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), s.at as *mut u32, n) };
}

/// `XCL_TRACE_FRAME`: per view heap (generation), the views written over an image view with
/// other words. The slot's PC descriptor changes with it, and so does every copy of the slot
/// made as memory (its T# names that PC descriptor), where the console's copy keeps the old view.
static REWRITES: Mutex<Option<std::collections::HashMap<u64, u64>>> = Mutex::new(None);
/// `XCL_TRACE_FRAME`: stale copies seen in the first slots of descriptor tables at draws
static STALE_AT_BIND: AtomicU64 = AtomicU64::new(0);

/// Whether writing `new` over `old` changes an image view (a buffer V# carries its address,
/// so copies of it keep their own value).
fn rewrites_image(old: &[u32; 8], new: &[u32]) -> bool {
    old[3] >> 28 >= 8 && old[..new.len()] != *new
}

/// The PC index that a view slot `k` of `h` owns.
fn own_view_index(h: &Heap, k: u32) -> u32 {
    match h.pages.get((k / PAGE_SLOTS) as usize) {
        Some(&f) if f != UNMAPPED => f + 2 * (k % PAGE_SLOTS),
        Some(_) => 0,
        None => 2 * k,
    }
}

unsafe fn view_words(h: &Heap, k: u32) -> [u32; 8] {
    let mut w = [0u32; 8];
    std::ptr::copy_nonoverlapping((h.shadow + (k * SLOT) as usize) as *const u32, w.as_mut_ptr(), 8);
    w
}

/// The image now in the slot that owns PC index `index`, as a T# in a slot of `h` names it: an
/// index in the shared heap for a heap in it, in `h`'s own PC heap otherwise. None: no slot
/// owns it, or that slot holds something other than its own image (its PC descriptor is unknown).
fn owner_image(heaps: &[Heap], h: &Heap, index: u32) -> Option<[u32; 8]> {
    let (o, k) = if h.pages.is_empty() {
        if index % 2 != 0 {
            return None;
        }
        (h, index / 2)
    } else {
        heaps.iter().filter(|o| !o.sampler).find_map(|o| {
            o.pages.iter().enumerate().find_map(|(p, &f)| {
                (f != UNMAPPED && index >= f && index - f < 2 * PAGE_SLOTS && (index - f) % 2 == 0).then(|| (o, p as u32 * PAGE_SLOTS + (index - f) / 2))
            })
        })?
    };
    if k >= o.n {
        return None;
    }
    let w = unsafe { view_words(o, k) };
    (w[3] >> 28 >= 8 && w[0] == index).then_some(w)
}

/// An image T# in slot `k` of `h` that names another slot's PC descriptor (a copy made as
/// memory) whose view has changed since: the copy's other words are no longer that view's.
fn stale_copy(heaps: &[Heap], h: &Heap, k: u32, w: &[u32; 8]) -> bool {
    w[3] >> 28 >= 8 && w[0] != own_view_index(h, k) && owner_image(heaps, h, w[0]).is_some_and(|o| o[1..] != w[1..])
}

/// `XCL_TRACE_FRAME`, at a draw: whether the view slot at console GPU address `addr32` holds a
/// stale copy (counted for the census).
pub fn stale_copy_at(addr32: u32) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    let cpu = (u.base() + addr32 as usize) as u64;
    let heaps = HEAPS.lock().unwrap();
    let Some(h) = heaps.iter().find(|h| !h.sampler && cpu >= h.shadow as u64 && cpu < (h.shadow + (h.n * SLOT) as usize) as u64) else { return false };
    let k = ((cpu - h.shadow as u64) / SLOT as u64) as u32;
    let stale = stale_copy(&heaps, h, k, &unsafe { view_words(h, k) });
    if stale {
        STALE_AT_BIND.fetch_add(1, Ordering::Relaxed);
    }
    stale
}

/// What the slots of a view heap hold (`census`).
#[derive(Default, Debug, PartialEq)]
struct ViewCensus {
    empty: u32,
    own: u32,
    other: u32,
    buffers: u32,
    stale: u32,
    /// (slot, index) of the first copies and of the first stale ones
    examples: Vec<(u32, u32)>,
    stale_examples: Vec<(u32, u32)>,
}

fn view_census(heaps: &[Heap], h: &Heap) -> ViewCensus {
    let mut c = ViewCensus::default();
    for k in 0..h.n {
        let w = unsafe { view_words(h, k) };
        if w.iter().all(|&x| x == 0) {
            c.empty += 1;
        } else if w[3] >> 28 >= 8 {
            if w[0] == own_view_index(h, k) {
                c.own += 1;
            } else {
                c.other += 1;
                if c.examples.len() < 8 {
                    c.examples.push((k, w[0]));
                }
                if stale_copy(heaps, h, k, &w) {
                    c.stale += 1;
                    if c.stale_examples.len() < 8 {
                        c.stale_examples.push((k, w[0]));
                    }
                }
            }
        } else {
            c.buffers += 1;
        }
    }
    c
}

/// Copy a console slot's descriptor to another slot of the same kind, the PC heap index
/// in it (T# word 0, S# word 3 [11:0]) renumbered for the destination. Buffer V# carry
/// their address and stay as they are.
pub fn copy_words(src: &Slot, dst: &Slot) {
    let n = if src.sampler { 4 } else { 8 };
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(src.at as *const u32, w.as_mut_ptr(), n) };
    if src.sampler {
        w[3] = (w[3] & !0xFFF) | (dst.index & 0xFFF);
    } else if w[3] >> 28 >= 8 {
        w[0] = dst.index;
    }
    write(dst, &w[..n]);
    if !src.sampler {
        note_resource(dst, resource_at(src.at).unwrap_or(0));
    }
}

/// The tracking-only form of `copy_words`, retaining the exact source and destination bytes.
pub fn copy_words_captured(src: &Slot, dst: &Slot) -> ([u32; 8], [u32; 8]) {
    let mut source = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(src.at as *const u32, source.as_mut_ptr(), 8) };
    let mut destination = source;
    if destination[3] >> 28 >= 8 {
        destination[0] = dst.index;
    }
    write(dst, &destination);
    note_resource(dst, resource_at(src.at).unwrap_or(0));
    (source, destination)
}

/// `XCL_TRACE_FRAME`: what the slots of each shader-visible view heap hold – images whose
/// PC index is the slot's own (made or copied through the device) or another (copied by the
/// title itself, as memory), buffers, empty.
pub fn census() {
    let heaps = HEAPS.lock().unwrap();
    // samplers: S# word 3 [11:0] is the PC index in the sampler heap the slot belongs to
    for h in heaps.iter().filter(|h| h.sampler) {
        let (mut empty, mut own, mut other) = (0u32, 0u32, 0u32);
        let mut examples: Vec<(u32, u32)> = Vec::new();
        for k in 0..h.n {
            let w = unsafe { std::slice::from_raw_parts((h.shadow + (k * SAMPLER_SLOT) as usize) as *const u32, 4) };
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[3] & 0xFFF == sampler_own_index(&h.pages, k) {
                own += 1;
            } else {
                other += 1;
                if examples.len() < 8 {
                    examples.push((k, w[3] & 0xFFF));
                }
            }
        }
        crate::xlog_info!(
            "frame trace: sampler heap {:#x} ({} slots, {}): {own} with their own index, {other} with another (slot, index: {examples:?}), {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" }
        );
    }
    let rewrites = REWRITES.lock().unwrap().clone().unwrap_or_default();
    for h in heaps.iter().filter(|h| !h.sampler) {
        let ViewCensus { empty, own, other, buffers, stale, examples, stale_examples } = view_census(&heaps, h);
        crate::xlog_info!(
            "frame trace: view heap {:#x} ({} slots, {}): {own} images with their own PC index, {other} with another (slot, index: {examples:?}), {stale} of them stale (slot, index: {stale_examples:?}), {} image views rewritten, {buffers} buffers, {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" },
            rewrites.get(&h.generation).copied().unwrap_or(0)
        );
    }
    crate::xlog_info!("frame trace: {} stale copies seen in descriptor tables at draws", STALE_AT_BIND.load(Ordering::Relaxed));
}

/// The descriptor at a console GPU address (low 32 bits, as in a descriptor table), if it
/// is in one of the console heaps.
pub fn words_at(addr32: u32) -> Option<[u32; 8]> {
    let u = crate::gpu::uma::get()?;
    let s = slot((u.base() + addr32 as usize) as u64)?;
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(s.at as *const u32, w.as_mut_ptr(), if s.sampler { 4 } else { 8 }) };
    Some(w)
}

/// Exact console words currently in view slots identified by heap generation and slot number.
/// Build the generation index once so validating S tracked slots across H heaps is O(S + H), and
/// keep `HEAPS` held while reading so the backing storage cannot be freed underneath us.
pub(super) fn view_slot_words(slots: &[(u64, u32)]) -> Vec<Option<[u32; 8]>> {
    let heaps = HEAPS.lock().unwrap();
    let by_generation: std::collections::HashMap<_, _> = heaps
        .iter()
        .filter(|heap| !heap.sampler)
        .map(|heap| (heap.generation, heap))
        .collect();
    slots
        .iter()
        .map(|&(generation, k)| {
            let h = by_generation.get(&generation).filter(|heap| k < heap.n)?;
            let mut words = [0u32; 8];
            unsafe { std::ptr::copy_nonoverlapping((h.shadow + (k * SLOT) as usize) as *const u32, words.as_mut_ptr(), 8) };
            Some(words)
        })
        .collect()
}

/// Exact console words currently in one view slot. The exact-generation heap lease used by a
/// caller keeps the native object and its backing allocation alive; `HEAPS` still verifies that
/// the allocation has not been detached before the point-in-time read.
pub(super) fn exact_view_slot_words(generation: u64, k: u32) -> Option<[u32; 8]> {
    let heaps = HEAPS.lock().unwrap();
    let h = heaps.iter().find(|heap| heap.generation == generation && !heap.sampler && k < heap.n)?;
    let mut words = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping((h.shadow + (k * SLOT) as usize) as *const u32, words.as_mut_ptr(), 8) };
    Some(words)
}

// ---------------------------------------------------------------------------------------
// Shared shader-visible PC sampler heap (`GLOBAL_SAMPLER`, up to D3D12's 2048 limit):
// allows 16-byte S# descriptors copied via raw CPU `memcpy` between CPU-only and
// shader-visible sampler heaps to keep valid PC sampler indices without CopyDescriptors*.
// ---------------------------------------------------------------------------------------

/// D3D12 maximum shader-visible sampler heap size.
const GLOBAL_SAMPLER_MAX: u32 = 2048;
/// Sampler slots allocated per page from `GLOBAL_SAMPLER` (small page size minimizes waste across
/// many small sampler heaps while staying within D3D12's 2048-sampler hardware limit).
const SAMPLER_PAGE_SLOTS: u32 = 8;

/// Fallback index for a sampler slot when no `GLOBAL_SAMPLER` page is mapped:
/// hosted heaps (`hosted == true`, downgraded to CPU-only) use reserved slot `0` (immutable zero
/// sampler) so exhaustion never aliases another heap's sampler at `k`; unhosted heaps keep `k`.
fn sampler_unmapped_index(hosted: bool, k: u32) -> u32 {
    if hosted { 0 } else { k }
}

/// Expected PC sampler index for slot `k` of a sampler heap with shared-page table `pages`.
fn sampler_own_index(pages: &[u32], k: u32) -> u32 {
    match pages.get((k / SAMPLER_PAGE_SLOTS) as usize) {
        Some(&f) if f != UNMAPPED => f + (k % SAMPLER_PAGE_SLOTS),
        _ => sampler_unmapped_index(!pages.is_empty(), k),
    }
}

struct GlobalSampler {
    dev: ID3D12Device,
    heap: ID3D12DescriptorHeap,
    cpu: u64,
    inc: u32,
    cap: u32,
    next: u32,
    free: Vec<(u32, u32)>,
    retired: VecDeque<RetiredRange<ID3D12Fence>>,
    leases: PublishLeases,
}

unsafe impl Send for GlobalSampler {}

static GLOBAL_SAMPLER: Mutex<Option<GlobalSampler>> = Mutex::new(None);
static GLOBAL_SAMPLER_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn sampler_page_descriptors(n: u32, p: u32) -> u32 {
    n.saturating_sub(p * SAMPLER_PAGE_SLOTS).clamp(1, SAMPLER_PAGE_SLOTS)
}

/// Shared shader-visible PC sampler heap (2048 descriptors), created lazily on first sampler heap.
pub unsafe fn shared_sampler_heap(dev: &ID3D12Device) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if std::env::var("XCL_SHARED_SAMPLER_HEAP").is_ok_and(|v| v.trim() == "0") {
        return false;
    }
    let mut g = GLOBAL_SAMPLER.lock().unwrap();
    if g.is_none() {
        if GLOBAL_SAMPLER_FAILED.load(Relaxed) {
            return false;
        }
        let d = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: GLOBAL_SAMPLER_MAX,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        let heap: ID3D12DescriptorHeap = match dev.CreateDescriptorHeap(&d) {
            Ok(h) => h,
            Err(e) => {
                GLOBAL_SAMPLER_FAILED.store(true, Relaxed);
                crate::xlog_warn!("console descriptor heaps: no shared shader-visible sampler heap ({e}) – each keeps its own");
                return false;
            }
        };
        let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        let zero = zero_sampler();
        for k in 0..GLOBAL_SAMPLER_MAX as u64 {
            dev.CreateSampler(&zero, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + k * inc as u64) as usize });
        }
        crate::xlog_info!("console descriptor heaps: one shader-visible PC sampler heap of {GLOBAL_SAMPLER_MAX} descriptors for all sampler heaps");
        *g = Some(GlobalSampler {
            dev: dev.clone(),
            heap,
            cpu,
            inc,
            cap: GLOBAL_SAMPLER_MAX,
            // Reserve slot 0 as the immutable point-wrap zero sampler for empty/zeroed S# entries.
            next: 1,
            free: Vec::new(),
            retired: VecDeque::new(),
            leases: PublishLeases::default(),
        });
    }
    true
}

fn reserve_global_sampler(count: u32) -> Option<u32> {
    let mut g = GLOBAL_SAMPLER.lock().unwrap();
    let s = g.as_mut()?;
    for (first, n) in pop_ready_ranges(&mut s.retired, |f| unsafe { super::real::fence_completed(f) }) {
        s.leases.release_range(&mut s.free, first, n);
    }
    let (cap, mut next) = (s.cap, s.next);
    let r = take_range(&mut s.free, &mut next, cap, count);
    s.next = next;
    if r.is_none() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            crate::xlog_warn!("console descriptor heaps: the shared sampler heap ({cap}) is full – overflowed sampler slots map to reserved slot 0");
        }
    }
    r
}

fn sampler_page_of(h: &mut Heap, k: u32) -> Option<(u32, u32)> {
    let p = (k / SAMPLER_PAGE_SLOTS) as usize;
    let mut first = *h.pages.get(p)?;
    if first == UNMAPPED {
        first = reserve_global_sampler(sampler_page_descriptors(h.n, p as u32))?;
        h.pages[p] = first;
    }
    Some((first, p as u32 * SAMPLER_PAGE_SLOTS))
}

fn release_global_sampler(first: u32, count: u32) {
    if count == 0 {
        return;
    }
    let dev = GLOBAL_SAMPLER.lock().unwrap().as_ref().map(|s| s.dev.clone());
    let Some(dev) = dev else { return };
    let marks = unsafe { signal_live_queues_for(&dev) };
    if let Some(s) = GLOBAL_SAMPLER.lock().unwrap().as_mut() {
        if marks.as_ref().is_some_and(Vec::is_empty) {
            s.leases.release_range(&mut s.free, first, count);
        } else {
            s.retired.push_back(RetiredRange { first, count, marks });
        }
        for (f, n) in pop_ready_ranges(&mut s.retired, |f| unsafe { super::real::fence_completed(f) }) {
            s.leases.release_range(&mut s.free, f, n);
        }
    }
}

/// The shared shader-visible PC sampler heap lists bind (None: no shared sampler heap).
pub fn global_sampler_heap() -> Option<ID3D12DescriptorHeap> {
    GLOBAL_SAMPLER.lock().unwrap().as_ref().map(|s| s.heap.clone())
}

/// Whether `obj` (a title's sampler heap) has its descriptors in the shared sampler heap.
pub fn in_global_sampler(obj: usize) -> bool {
    HEAPS.lock().unwrap().iter().any(|h| h.obj == obj && h.sampler && !h.pages.is_empty())
}

/// The zero sampler: what an S# of zeros is on the console (point filtering, wrap, mip 0 only),
/// and what a shader gets whose S# XCL takes for a heap entry it is not (entry 0).
/// `XCL_ZERO_SAMPLER_LINEAR=1` (a diagnostic): linear filtering, clamp, every mip instead; a
/// picture that changes with it samples through the zero sampler.
pub(super) fn zero_sampler() -> D3D12_SAMPLER_DESC {
    static LINEAR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let linear = *LINEAR.get_or_init(|| {
        let on = std::env::var("XCL_ZERO_SAMPLER_LINEAR").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("D3D12.X: XCL_ZERO_SAMPLER_LINEAR=1 - the zero sampler is linear, clamp, all mips (a diagnostic)");
        }
        on
    });
    let address = if linear { D3D12_TEXTURE_ADDRESS_MODE_CLAMP } else { D3D12_TEXTURE_ADDRESS_MODE_WRAP };
    D3D12_SAMPLER_DESC {
        Filter: if linear { D3D12_FILTER_MIN_MAG_MIP_LINEAR } else { D3D12_FILTER_MIN_MAG_MIP_POINT },
        AddressU: address,
        AddressV: address,
        AddressW: address,
        MipLODBias: 0.0,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        BorderColor: [0.0; 4],
        MinLOD: 0.0,
        MaxLOD: if linear { 1000.0 } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every T# index the translated shaders accept, and its UAV at `tw0 + 1`, is inside the
    /// shared heap this GPU gets: an index past the heap's end hangs NVIDIA GPUs (whose heap is
    /// 1 000 000, while the shaders accepted indices up to 2^21 - 2).
    #[test]
    fn the_shaders_t_sharp_index_bound_fits_the_shared_heap() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let cap = unsafe { global_cap(&dev.unwrap()) };
        let bound = crate::gpu::dxil::lower_image::VIEW_HEAP_MAX_SRV;
        // accepted: tw0 < bound, so the highest UAV index is bound - 1 + 1 = bound
        assert!(bound < cap, "shaders accept indices up to {bound}, the shared heap holds {cap}");
    }

    /// The title's memory holds a heap only when committed and all in window 0: where a Series X
    /// run's view heap was (0xf2e00000 into window 0), not where a Series S run's was (window 1,
    /// 0x1400000 in), nor across the end of window 0, nor outside unified memory.
    #[test]
    fn title_memory_holds_a_heap_only_in_window_0() {
        use crate::gpu::uma::{Uma, WINDOW};
        let u = Uma::for_runtime_windows(16 << 20, 512 << 20, 16).expect("16-window aperture");
        let b = u.base();
        let view_heap = (1_245_184 * SLOT) as usize; // 0x2600000
        assert!(title_memory_fits(&u, b + 0xf2e0_0000, view_heap, true));
        assert!(!title_memory_fits(&u, b + 0xf2e0_0000, view_heap, false), "not committed");
        assert!(!title_memory_fits(&u, b + WINDOW + 0x140_0000, view_heap, true));
        assert!(!title_memory_fits(&u, b + WINDOW - 0x1_0000, view_heap, true), "into window 1");
        assert!(title_memory_fits(&u, b + WINDOW - view_heap, view_heap, true), "the end of window 0");
        // the coherent window, a view window, plain memory
        for p in [b + 3 * WINDOW, b + 4 * WINDOW, 0xf2e0_0000] {
            assert!(!title_memory_fits(&u, p, 0x1_0000, true), "{p:#x}");
        }
        assert!(!title_memory_fits(&u, b + 0x2001_0000, 0, true));
    }

    /// Ranges of the shared heap: released ones first (first fit, split), then new ones, and
    /// none past its size.
    #[test]
    fn shared_heap_ranges() {
        let (mut free, mut next) = (Vec::new(), 0u32);
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(0));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(40));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), None, "20 left");
        free.push((0, 40));
        assert_eq!(take_range(&mut free, &mut next, 100, 30), Some(0));
        assert_eq!(free, vec![(30, 10)]);
        assert_eq!(take_range(&mut free, &mut next, 100, 20), Some(80));
        assert_eq!(take_range(&mut free, &mut next, 100, 10), Some(30));
        assert!(free.is_empty());
    }

    /// How full the shared heap is: descriptors below the cursor but the released ones, and the
    /// marks (50, 75, 90 %) that use is past.
    #[test]
    fn shared_heap_use_and_the_marks_it_passes() {
        assert_eq!(in_use(100, &[]), 100);
        assert_eq!(in_use(100, &[(20, 10), (50, 6)]), 84);
        for (used, marks) in [(0, 0), (500, 0), (501, 1), (750, 1), (751, 2), (900, 2), (901, 3), (1000, 3)] {
            assert_eq!(marks_passed(used, 1000), marks, "{used} of 1000");
        }
        assert_eq!(marks_passed(1 << 21, 1 << 21), 3, "no overflow at the largest heap");
    }

    #[test]
    fn exact_view_slot_lease_survives_detach_and_rejects_address_reuse() {
        const CHILD: &str = "XCL_TEST_VIEW_SLOT_LEASE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::exact_view_slot_lease_survives_detach_and_rejects_address_reuse", "--nocapture"])
                .env(CHILD, "1").env("XCL_TEXTURE_BUDGET_EVICT", "1").output().unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        static RELEASE_REENTERED: AtomicBool = AtomicBool::new(false);
        #[repr(C)]
        struct Object { vt: *const usize, refs: AtomicU32 }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            (*(this as *mut Object)).refs.fetch_add(1, Ordering::Relaxed) + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            RELEASE_REENTERED.fetch_or(HEAP_LIFETIMES.try_lock().is_ok(), Ordering::Relaxed);
            (*(this as *mut Object)).refs.fetch_sub(1, Ordering::Relaxed) - 1
        }
        unsafe extern "system" fn get_device(_this: *mut c_void, _iid: *const windows::core::GUID, _out: *mut *mut c_void) -> windows::core::HRESULT {
            windows::core::HRESULT(0x8000_4002u32 as i32)
        }
        let mut vt = [0usize; 8];
        vt[1] = add_ref as usize;
        vt[2] = release as usize;
        vt[7] = get_device as usize;
        let mut device = Object { vt: vt.as_ptr(), refs: AtomicU32::new(1) };
        let mut global_heap = Object { vt: vt.as_ptr(), refs: AtomicU32::new(1) };
        let mut source_heap = Object { vt: vt.as_ptr(), refs: AtomicU32::new(1) };
        unsafe {
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: ID3D12Device::from_raw((&mut device as *mut Object).cast()),
                heap: ID3D12DescriptorHeap::from_raw((&mut global_heap as *mut Object).cast()),
                cpu: 0x10_0000, gpu: 0x20_0000, inc: 32, cap: 128, next: 80,
                free: Vec::new(), leases: PublishLeases::default(),
                nulls: None,
                retired: VecDeque::new(),
            });
            HEAPS.lock().unwrap().push(Heap {
                generation: 70, obj: &mut source_heap as *mut Object as usize, shadow: 0x5000,
                n: 4, pc_cpu: 0x30_0000, pc_gpu: 0, inc: 32, sampler: false,
                pages: vec![64], visible: true, memory: Memory::Uma,
            });
            let lease = lease_view_slot(70, 2, 0x30_0080, 0x10_0880, 68).expect("exact slot");
            assert_eq!((source_heap.refs.load(Ordering::Relaxed), global_heap.refs.load(Ordering::Relaxed), device.refs.load(Ordering::Relaxed)), (2, 2, 2));
            assert!(lease_view_slot(70, 4, 0x30_0100, 0, 0).is_none(), "slot bound");
            assert!(lease_view_slot(70, 2, 0x30_0080, 0x10_0880, 69).is_none(), "shared index");

            let lifetime = lock_lifetime();
            let Taken(old) = take_after_release(&mut source_heap as *mut Object as usize, Some(70), lifetime.as_ref()).unwrap();
            drop(lifetime);
            HEAPS.lock().unwrap().push(Heap { generation: 71, pc_cpu: 0x40_0000, pages: vec![80], ..old });
            assert!(lease_view_slot(70, 2, 0x30_0080, 0x10_0880, 68).is_none(), "old generation at reused object address");
            let lifetime = lock_lifetime();
            assert!(take_after_release(&mut source_heap as *mut Object as usize, Some(70), lifetime.as_ref()).is_none());
            drop(lifetime);
            assert!(HEAPS.lock().unwrap().iter().any(|heap| heap.generation == 71), "late cleanup must keep the replacement");
            assert_eq!((lease.pc_srv, lease.shared_srv, lease.index), (0x30_0080, 0x10_0880, 68));
            drop(lease);
            assert_eq!((source_heap.refs.load(Ordering::Relaxed), global_heap.refs.load(Ordering::Relaxed), device.refs.load(Ordering::Relaxed)), (1, 1, 1));
            HEAPS.lock().unwrap().clear();
            HEAPS.lock().unwrap().push(Heap {
                generation: 72, obj: &mut source_heap as *mut Object as usize, shadow: 0x6000,
                n: 1, pc_cpu: 0x50_0000, pc_gpu: 0, inc: 32, sampler: false,
                pages: Vec::new(), visible: true, memory: Memory::Uma,
            });
            RELEASE_REENTERED.store(false, Ordering::Relaxed);
            assert!(lease_view_slot(72, 0, 0x50_0000, 0, 0).is_none(), "GetDevice fails");
            assert!(RELEASE_REENTERED.load(Ordering::Relaxed), "failed lease must drop its owner outside the lifetime gate");
            HEAPS.lock().unwrap().clear();
            *GLOBAL.lock().unwrap() = None;
        }
    }

    #[test]
    fn a_freed_range_at_the_cursor_continues_into_the_never_used_tail() {
        // the false-full shape: four freed slots end at the cursor and four more were never
        // used; a request for eight is contiguous across the cursor, not exhaustion
        let (mut free, mut next) = (vec![(8u32, 4u32)], 12u32);
        assert_eq!(take_range(&mut free, &mut next, 16, 8), Some(8), "freed range plus tail is one run");
        assert_eq!(next, 16);
        assert!(free.is_empty());
    }

    #[test]
    fn released_ranges_touching_one_another_merge_into_one() {
        let mut free = vec![(0u32, 4), (16, 4)];
        merge_release(&mut free, 4, 8);
        assert_eq!(free, vec![(0, 12), (16, 4)]);
        merge_release(&mut free, 12, 4);
        assert_eq!(free, vec![(0, 20)], "the merge chains across the whole run");
        merge_release(&mut free, 30, 2);
        assert_eq!(free, vec![(0, 20), (30, 2)]);
    }

    #[test]
    fn window_views_do_not_hold_global_during_device_calls() {
        const CHILD: &str = "XCL_TEST_WINDOW_VIEWS_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::window_views_do_not_hold_global_during_device_calls", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        static UNLOCKED: AtomicBool = AtomicBool::new(true);
        static CALLS: AtomicU32 = AtomicU32::new(0);
        // slot k (bit k) got a null resource
        static NULLS: AtomicU32 = AtomicU32::new(0);
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        unsafe extern "system" fn create_uav(_this: *mut c_void, resource: *mut c_void, _counter: *mut c_void, _desc: *const D3D12_UNORDERED_ACCESS_VIEW_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
            // The adopted device hook can re-enter descriptor bookkeeping.
            UNLOCKED.fetch_and(GLOBAL.try_lock().is_ok(), Ordering::Relaxed);
            CALLS.fetch_add(1, Ordering::Relaxed);
            if resource.is_null() {
                let k = (at.ptr - 0x1000) / 32 - crate::gpu::recomp::MEM_HEAP_BASE as usize;
                NULLS.fetch_or(1 << k, Ordering::Relaxed);
            }
        }
        let mut vt = [0usize; 20];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        vt[19] = create_uav as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let mut heap = Object { vt: vt.as_ptr() };
        let mut resource = Object { vt: vt.as_ptr() };
        unsafe {
            let dev = ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void);
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: dev.clone(),
                heap: ID3D12DescriptorHeap::from_raw(&mut heap as *mut _ as *mut c_void),
                cpu: 0x1000, gpu: 0x2000, inc: 32, cap: 1024,
                next: crate::gpu::recomp::MEM_HEAP_BASE + crate::gpu::uma::MAX_HEAP_WINDOWS as u32, free: Vec::new(), retired: VecDeque::new(), nulls: None, leases: PublishLeases::default(),
            });
            let window = (&mut resource as *mut _ as usize, 0x10000);
            // the root-UAV slots past the windows there are: null views, not window 0 again
            assert!(set_window_views(&dev, &[window]));
            assert_eq!(CALLS.swap(0, Ordering::Relaxed), crate::gpu::uma::MAX_WINDOWS as u32);
            assert_eq!(NULLS.swap(0, Ordering::Relaxed), 0b1110);
            let mut windows = [window; crate::gpu::uma::MAX_HEAP_WINDOWS];
            assert!(set_window_views(&dev, &windows));
            assert_eq!(CALLS.swap(0, Ordering::Relaxed), windows.len() as u32);
            assert_eq!(NULLS.swap(0, Ordering::Relaxed), 0);
            // A window without memory `(0, 0)` (16 windows: the ones after the views) gets a null
            // view; an active window that failed to open `(0, bytes > 0)` is rejected.
            windows[7] = (0, 0);
            windows[15] = (0, 0);
            assert!(set_window_views(&dev, &windows));
            assert_eq!(CALLS.swap(0, Ordering::Relaxed), windows.len() as u32);
            assert_eq!(NULLS.swap(0, Ordering::Relaxed), 1 << 7 | 1 << 15);
            windows[7] = (0, 0x10000);
            assert!(!set_window_views(&dev, &windows));
            assert_eq!(CALLS.load(Ordering::Relaxed), 0);
            assert!(UNLOCKED.load(Ordering::Relaxed), "device callback ran while GLOBAL was locked");
            *GLOBAL.lock().unwrap() = None;
        }
    }

    /// A T# of zeros (an empty slot) names descriptors 0 (SRV) and 1 (UAV): those must be
    /// null views, never a memory window's raw buffer UAV (a texture read of that hangs the GPU).
    #[test]
    fn zero_tsharp_does_not_reach_window_views() {
        use std::ffi::c_void;
        use std::sync::Mutex as M;
        static SEEN: M<Vec<(bool, usize, usize)>> = M::new(Vec::new());
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn srv(_this: *mut c_void, r: *mut c_void, _d: *const D3D12_SHADER_RESOURCE_VIEW_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
            SEEN.lock().unwrap().push((false, r as usize, at.ptr));
        }
        unsafe extern "system" fn uav(_this: *mut c_void, r: *mut c_void, _c: *mut c_void, _d: *const D3D12_UNORDERED_ACCESS_VIEW_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
            SEEN.lock().unwrap().push((true, r as usize, at.ptr));
        }
        let base = crate::gpu::recomp::MEM_HEAP_BASE;
        assert!(base >= 2, "window views overlap the SRV / UAV pair of a zero T#");
        assert_eq!(base % 2, 0, "window views must not split a console slot's SRV / UAV pair");
        let mut vt = [0usize; 20];
        vt[18] = srv as usize;
        vt[19] = uav as usize;
        let mut device = Object { vt: vt.as_ptr() };
        unsafe {
            let dev = std::mem::ManuallyDrop::new(ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void));
            null_views(&dev, 0x1000, 32);
        }
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!(seen.len(), base as usize);
        for (k, &(is_uav, r, at)) in seen.iter().enumerate() {
            assert_eq!((is_uav, r, at), (k % 2 == 1, 0, 0x1000 + 32 * k), "descriptor {k}");
        }
    }

    /// A shader-visible sampler heap has a PC sampler at every index a translated shader forms
    /// (S# word 3 & 0x7FF), whatever the title asked for (a retail title: 128); a CPU-only one
    /// as many as its slots. D3D12 makes a shader-visible sampler heap of that size.
    #[test]
    fn every_sampler_index_a_shader_forms_is_in_the_pc_heap() {
        for n in [1, 16, 128, 2047, 2048] {
            assert_eq!(pc_samplers(n, true), 2048, "{n} slots");
            assert!(u32::MAX & 0x7FF < pc_samplers(n, true), "{n} slots: the highest index a shader forms");
            assert_eq!(pc_samplers(n, false), n, "{n} slots, CPU only");
        }
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let dev = unsafe {
            let mut dev: Option<ID3D12Device> = None;
            match CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) {
                Ok(warp) if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() => dev.unwrap(),
                _ => return crate::gpu::recomp::runner::skip_test("no WARP device"),
            }
        };
        let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, NumDescriptors: pc_samplers(128, true), Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
        let made = unsafe { dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&d) };
        assert!(made.is_ok(), "a shader-visible sampler heap of {}: {made:?}", d.NumDescriptors);
    }

    /// Every descriptor of a range that is not handed out gets a null view, an SRV at even and a
    /// UAV at odd indices (a range starting at an odd one too), a block at a time.
    #[test]
    fn free_ranges_of_the_shared_heap_get_null_views() {
        use std::ffi::c_void;
        use std::sync::Mutex as M;
        static COPIES: M<Vec<(u32, usize, usize)>> = M::new(Vec::new());
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn copy(_this: *mut c_void, n: u32, dst: D3D12_CPU_DESCRIPTOR_HANDLE, src: D3D12_CPU_DESCRIPTOR_HANDLE, ty: D3D12_DESCRIPTOR_HEAP_TYPE) {
            assert_eq!(ty, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            COPIES.lock().unwrap().push((n, dst.ptr, src.ptr));
        }
        let mut vt = [0usize; 25];
        vt[24] = copy as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let (heap, nulls, inc) = (0x10_0000u64, 0x90_0000u64, 32u32);
        for (from, to) in [(16u32, 3 * NULL_BLOCK + 5), (7, 9), (2 * NULL_BLOCK - 1, 2 * NULL_BLOCK + 1)] {
            COPIES.lock().unwrap().clear();
            unsafe {
                let dev = std::mem::ManuallyDrop::new(ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void));
                fill_nulls(&dev, heap, inc, from, to, nulls);
            }
            let mut at = from;
            for &(n, dst, src) in COPIES.lock().unwrap().iter() {
                assert_eq!(dst as u64, heap + at as u64 * inc as u64, "{from}..{to}: contiguous");
                let k = (src as u64 - nulls) / inc as u64;
                assert_eq!(k % 2, at as u64 % 2, "{from}..{to}: an SRV over an SRV, a UAV over a UAV");
                assert!(k + n as u64 <= NULL_BLOCK as u64, "{from}..{to}: within the null block");
                at += n;
            }
            assert_eq!(at, to, "{from}..{to}: all of it");
        }
    }

    /// What each index of the shared heap names: a destroyed resource's views (also copies of
    /// them) are claimed for null views, an index overwritten since by another resource's view
    /// is not, and a resource that never had a view costs one lookup and claims nothing.
    #[test]
    fn a_destroyed_resources_views_are_claimed_and_no_other() {
        let (a, b, c) = (0xA000usize, 0xB000usize, 0xC000usize);
        let n = Names::new(64);
        n.note(4, a); // SRV of slot 2
        n.note(5, a); // its UAV
        n.note(8, a);
        n.note(8, b); // overwritten by a view of b
        // a copy of slot 2 to slot 6 (`names_of` of the source, then `publish_range`)
        let (srv, uav) = (n.name(4), n.name(5));
        n.note(12, srv);
        n.note(13, uav);
        n.note(20, a);
        n.clear(20, 22); // a released range: null views there already
        assert!(n.candidates(c).is_none(), "a resource without views");
        let mut claimed: Vec<u32> = n.candidates(a).unwrap().into_iter().filter(|&i| n.claim(i, a)).collect();
        claimed.sort_unstable();
        assert_eq!(claimed, vec![4, 5, 12, 13], "a's views and their copies; not 8 (b's now), not 20 (nulled)");
        assert_eq!((n.name(4), n.name(8), n.name(12)), (GONE, b, GONE));
        assert!(n.candidates(a).is_none(), "forgotten");
        // b goes: its one view
        let claimed: Vec<u32> = n.candidates(b).unwrap().into_iter().filter(|&i| n.claim(i, b)).collect();
        assert_eq!(claimed, vec![8]);
    }

    /// A resource whose views are made again and again into the same few slots keeps a list of
    /// those slots, not one entry for each view.
    #[test]
    fn views_made_again_do_not_grow_a_resources_list() {
        let (a, b) = (0xA000usize, 0xB000usize);
        let n = Names::new(64);
        for k in 0..10_000u32 {
            n.note(k % 4, a);
            n.note(k % 4, b);
        }
        n.note(1, a);
        let v = n.candidates(a).unwrap();
        assert!(v.len() < 128, "{} entries", v.len());
        assert_eq!(v.iter().filter(|&&i| n.claim(i, a)).count(), 1, "only slot 1 is a's now");
        assert!(n.candidates(b).unwrap().len() < 128);
    }

    /// A copy made as memory names its source's PC descriptor: making another view in the source
    /// slot changes what the copy shows, where the console's copy keeps the old view.
    #[test]
    fn a_memory_copy_whose_source_slot_is_rewritten_is_stale() {
        let Some(u) = crate::gpu::uma::get() else { return };
        assert!(register(0xA7, 4, 0x1000, 0x5000, 32, false, false, true));
        let v = start(0xA7).unwrap();
        let (s0, s2) = (slot(v).unwrap(), slot(v + 64).unwrap());
        let a = [s0.index, 0x0470_0000, 0x0001_C03F, 9 << 28, 0, 0, 0, 0];
        write(&s0, &a);
        // the title's own copy: the words as they are, the source's index included
        unsafe { std::ptr::copy_nonoverlapping(s0.at as *const u32, s2.at as *mut u32, 8) };
        let count = || {
            let heaps = HEAPS.lock().unwrap();
            let h = heaps.iter().find(|h| h.obj == 0xA7).unwrap();
            view_census(&heaps, h)
        };
        let c = count();
        assert_eq!((c.own, c.other, c.stale), (1, 1, 0));
        assert!(!stale_copy_at((v + 64 - u.base() as u64) as u32));
        let b = [s0.index, 0x0470_0000, 0x0003_C07F, 9 << 28, 0, 0, 0, 0];
        assert!(rewrites_image(&a, &b) && !rewrites_image(&a, &a));
        assert!(!rewrites_image(&[0x1234, 0, 64, 0x3101_6FAC, 0, 0, 0, 0], &b), "a buffer V# is copied by value");
        write(&s0, &b);
        let c = count();
        assert_eq!((c.other, c.stale, c.stale_examples), (1, 1, vec![(2, s0.index)]));
        assert!(stale_copy_at((v + 64 - u.base() as u64) as u32));
        // the source slot holding something else: its PC descriptor is unknown, not counted
        write(&s0, &[0x1234, 0, 64, 0x3101_6FAC, 0, 0, 0, 0]);
        assert_eq!(count().stale, 0);
        unregister(0xA7);
    }

    #[test]
    fn slots_copy_and_release() {
        let Some(u) = crate::gpu::uma::get() else { return };
        let rel = |p: u64| (p as usize - u.base()) as u32;
        // views: 4 slots over a PC heap at 0x1000 (increment 32, GPU 0x5000); samplers: 2 at 0x9000
        assert!(register(0xA1, 4, 0x1000, 0x5000, 32, false, false, true));
        assert!(register(0xA2, 2, 0x9000, 0, 16, true, false, true));
        let (v, sm) = (start(0xA1).unwrap(), start(0xA2).unwrap());
        assert_eq!(u.window_of(v as usize), Some(0), "view heap must live in window 0 (32-bit table addressing)");
        assert_eq!(u.window_of(sm as usize), Some(0), "sampler heap must live in window 0");
        assert!(u.is_sparse(v as usize) && u.is_sparse(sm as usize), "descriptor heaps backed by sparse VRAM");
        let s1 = slot(v + 32).unwrap();
        assert_eq!((s1.index, s1.pc_srv, s1.pc_uav, s1.pc_gpu_uav), (2, 0x1040, 0x1060, 0x5060));
        // a 2D T# from slot 1 to slot 3: word 0 = the destination's SRV index
        write(&s1, &[2, 0x0470_0000, 0, 9 << 28, 0, 0, 0, 0]);
        let s3 = slot(v + 96).unwrap();
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], [6, 0x0470_0000, 0, 9 << 28]);
        // a buffer V# keeps its address
        write(&s1, &[0x1234, 0, 64, 0x3101_6FAC, 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[0], 0x1234);
        // a raytracing BVH SRSRC keeps its base address and box-sort flag verbatim
        let bvh = crate::gpu::desc::bvh_words(0x12_3456_7800, true);
        write(&s1, &[bvh[0], bvh[1], bvh[2], bvh[3], 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], bvh);
        // samplers: 16-byte slots, S# word 3 [11:0] = sampler index
        let (q0, q1) = (slot(sm).unwrap(), slot(sm + 16).unwrap());
        assert!(q1.sampler && q1.index == 1 && q1.pc_srv == 0x9010);
        write(&q0, &[1, 2, 3, 0xABC0_0000]);
        copy_words(&q0, &q1);
        assert_eq!(words_at(rel(sm + 16)).unwrap()[..4], [1, 2, 3, 0xABC0_0001]);
        assert!(slot(sm + 32).is_none(), "past the last sampler slot");
        unregister(0xA1);
        unregister(0xA2);
        assert!(start(0xA1).is_none() && slot(v).is_none() && slot(sm).is_none());
    }

    /// The shared heap full: the slots of a page that finds no place there name the null views
    /// at 0 / 1, never 2k (in the shared heap, the only view heap lists bind, that is another
    /// heap's view, a memory window's raw UAV for k from 2 to 9, or past its end); a T# copied
    /// there too. A page placed before keeps its place, and one placed once room is free again
    /// gets its own. Every page that found no room is counted.
    #[test]
    fn a_slot_whose_page_finds_no_room_names_the_null_views() {
        const CHILD: &str = "XCL_TEST_SHARED_HEAP_FULL_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::a_slot_whose_page_finds_no_room_names_the_null_views", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        let Some(u) = crate::gpu::uma::get() else { return };
        use std::ffi::c_void;
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        let mut vt = [0usize; 4];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let mut heap = Object { vt: vt.as_ptr() };
        // room for one page of 256 slots past the null views and the memory windows
        let base = crate::gpu::recomp::MEM_HEAP_BASE + crate::gpu::uma::MAX_HEAP_WINDOWS as u32;
        unsafe {
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void),
                heap: ID3D12DescriptorHeap::from_raw(&mut heap as *mut _ as *mut c_void),
                cpu: 0x10_0000, gpu: 0x20_0000, inc: 32, cap: base + 2 * PAGE_SLOTS,
                next: base, free: Vec::new(), retired: VecDeque::new(), nulls: None,
                leases: PublishLeases::default(),
            });
        }
        let rel = |p: u64| (p as usize - u.base()) as u32;
        assert!(register(0xC1, 2 * PAGE_SLOTS, 0x30_0000, 0, 32, false, true, true));
        assert!(register(0xC2, 16, 0x40_0000, 0, 32, false, true, true));
        let (a, b) = (start(0xC1).unwrap(), start(0xC2).unwrap());
        let placed = slot(a + 2 * SLOT as u64).unwrap();
        assert_eq!(placed.index, base + 4, "page 0 of the first heap takes the room there is");
        assert_ne!(placed.shared_srv, 0);
        let k = PAGE_SLOTS + 2;
        let s = slot(a + (k * SLOT) as u64).unwrap();
        assert_eq!((s.index, s.shared_srv, s.pc_gpu_uav), (0, 0, 0), "slot {k} of the first heap: its page found no room");
        assert_eq!(s.pc_srv, 0x30_0000 + (2 * k * 32) as u64, "views are still made in the title's heap");
        for k in 0..16 {
            let s = slot(b + (k * SLOT) as u64).unwrap();
            assert_eq!((s.index, s.shared_srv), (0, 0), "slot {k} of the second heap: no room for its page");
        }
        assert_eq!(NO_ROOM.load(Relaxed), 17, "every page that found no room, each time");
        assert_eq!((PAGES_PLACED.load(Relaxed), FILL_SAID.load(Relaxed)), (1, FILL_MARKS.len()), "one page placed; the heap is past every mark");
        // a 2D T# copied into a slot without a place names the null views
        write(&placed, &[base + 4, 0x0470_0000, 0, 9 << 28, 0, 0, 0, 0]);
        copy_words(&placed, &s);
        assert_eq!(words_at(rel(a + (k * SLOT) as u64)).unwrap()[..4], [0, 0x0470_0000, 0, 9 << 28]);
        assert_eq!(slot(a + 2 * SLOT as u64).unwrap().index, base + 4, "a placed page keeps its place");
        // the first heap goes: the second heap's page gets the room it left
        unregister(0xC1);
        let s = slot(b + 2 * SLOT as u64).unwrap();
        assert_eq!(s.index, base + 4);
        assert_ne!(s.shared_srv, 0);
        assert_eq!(PAGES_PLACED.load(Relaxed), 1, "the first heap's page went with it");
        unregister(0xC2);
        assert_eq!(PAGES_PLACED.load(Relaxed), 0);
        *GLOBAL.lock().unwrap() = None;
    }

    /// Minecraft's view heaps: eight of 73 727 slots, 147 454 PC descriptors each – more in all
    /// than an NVIDIA shared heap holds (1 000 000). Each gets a place there for the pages it
    /// uses; before, the seventh and eighth got none and their views were lost (flicker).
    #[test]
    fn large_heaps_get_shared_pages_as_used() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        if crate::gpu::uma::get().is_none() {
            return;
        }
        let dev = unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            dev.expect("D3D12 device")
        };
        assert!(unsafe { shared_heap(&dev) });
        const N: u32 = 73_727;
        let objs: Vec<usize> = (0..8).map(|i| 0xB000 + i).collect();
        for &o in &objs {
            assert!(register(o, N, 0x10_0000, 0, 32, false, true, true));
            assert!(in_global(o));
        }
        let mut seen = Vec::new();
        for &o in &objs {
            let v = start(o).unwrap();
            // slot 0, the last of page 0, the first of page 1 and the heap's last slot
            let s: Vec<Slot> = [0, PAGE_SLOTS - 1, PAGE_SLOTS, N - 1].iter().map(|&k| slot(v + (k * SLOT) as u64).unwrap()).collect();
            assert!(s.iter().all(|x| x.shared_srv != 0), "heap {o:#x}: a slot without a place in the shared heap");
            assert_eq!(s[1].index, s[0].index + 2 * (PAGE_SLOTS - 1), "a page is contiguous");
            assert_eq!(s[0].pc_srv, 0x10_0000, "views are still made in the title's heap");
            assert_eq!(s[3].pc_srv, 0x10_0000 + (2 * (N - 1) * 32) as u64);
            for x in &s {
                assert!(!seen.contains(&x.index), "two slots share the PC index {}", x.index);
                seen.push(x.index);
            }
            // the same slot again: the same place
            assert_eq!(slot(v).unwrap().index, s[0].index);
        }
        for &o in &objs {
            unregister(o);
        }
        // a heap smaller than a page takes 2 descriptors per slot, as it did when a heap's
        // whole range was reserved (a title with many small heaps must not fill the shared heap)
        assert_eq!((page_descriptors(4, 0), page_descriptors(N, 0), page_descriptors(N, N / PAGE_SLOTS)), (8, 2 * PAGE_SLOTS, 2 * (N % PAGE_SLOTS)));
        assert!(register(0xB100, 4, 0x10_0000, 0, 32, false, true, true) && register(0xB101, 4, 0x20_0000, 0, 32, false, true, true));
        let (a, b) = (slot(start(0xB100).unwrap()).unwrap(), slot(start(0xB101).unwrap()).unwrap());
        assert!(a.shared_srv != 0 && b.shared_srv != 0);
        assert_eq!(a.index.abs_diff(b.index), 8, "two heaps of 4 slots lie 8 descriptors apart");
        unregister(0xB100);
        unregister(0xB101);
    }

    #[test]
    fn shared_sampler_pages_preserve_zero_slot_and_clamp_tails() {
        assert_eq!(sampler_page_descriptors(8, 0), 8);
        assert_eq!(sampler_page_descriptors(SAMPLER_PAGE_SLOTS, 0), SAMPLER_PAGE_SLOTS);
        assert_eq!(sampler_page_descriptors(SAMPLER_PAGE_SLOTS + 5, 0), SAMPLER_PAGE_SLOTS);
        assert_eq!(sampler_page_descriptors(SAMPLER_PAGE_SLOTS + 5, 1), 5);
        // Simulate GLOBAL_SAMPLER reserving slot 0 for the immutable zero sampler.
        let (mut free, mut next) = (Vec::new(), 1u32);
        let a = take_range(&mut free, &mut next, GLOBAL_SAMPLER_MAX, sampler_page_descriptors(64, 0)).unwrap();
        let b = take_range(&mut free, &mut next, GLOBAL_SAMPLER_MAX, sampler_page_descriptors(64, 1)).unwrap();
        assert_eq!(a, 1);
        assert_eq!(b, 1 + SAMPLER_PAGE_SLOTS);
        assert!(b + SAMPLER_PAGE_SLOTS <= GLOBAL_SAMPLER_MAX);
        // Exhaustion fallback: hosted sampler heaps map overflowed slots to reserved slot 0; unhosted keep local k.
        assert_eq!(sampler_unmapped_index(true, 42), 0);
        assert_eq!(sampler_unmapped_index(false, 42), 42);
    }

    #[test]
    fn release_global_sampler_merges_adjacent_ranges_and_ignores_zero_count() {
        const CHILD: &str = "XCL_TEST_RELEASE_GLOBAL_SAMPLER_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::release_global_sampler_merges_adjacent_ranges_and_ignores_zero_count", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        let mut vt = [0usize; 4];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let mut heap = Object { vt: vt.as_ptr() };
        unsafe {
            *GLOBAL_SAMPLER.lock().unwrap() = Some(GlobalSampler {
                dev: ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void),
                heap: ID3D12DescriptorHeap::from_raw(&mut heap as *mut _ as *mut c_void),
                cpu: 0x1000, inc: 32, cap: 17, next: 1, free: Vec::new(), retired: VecDeque::new(), leases: PublishLeases::default(),
            });
            assert_eq!(reserve_global_sampler(4), Some(1));
            assert_eq!(reserve_global_sampler(4), Some(5));
            assert_eq!(reserve_global_sampler(4), Some(9));
            assert_eq!(reserve_global_sampler(4), Some(13));
            assert_eq!(reserve_global_sampler(1), None);
            // Adjacent out-of-order releases coalesce and zero-count releases are ignored.
            release_global_sampler(5, 4);
            release_global_sampler(1, 4);
            release_global_sampler(1, 0);
            assert_eq!(GLOBAL_SAMPLER.lock().unwrap().as_ref().unwrap().free, vec![(1, 8)]);
            assert_eq!(reserve_global_sampler(8), Some(1));
            // Coalesced range ending at cursor straddles cleanly into the never-used tail.
            {
                let mut g = GLOBAL_SAMPLER.lock().unwrap();
                let s = g.as_mut().unwrap();
                s.next = 13;
                s.free.clear();
            }
            release_global_sampler(9, 2);
            release_global_sampler(11, 2);
            assert_eq!(reserve_global_sampler(8), Some(9));
            let g = GLOBAL_SAMPLER.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.next, 17);
            assert!(s.free.is_empty());
            drop(g);
            *GLOBAL_SAMPLER.lock().unwrap() = None;
        }
    }

    #[test]
    fn in_global_excludes_shared_sampler_heaps() {
        const CHILD: &str = "XCL_TEST_IN_GLOBAL_SAMPLER_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::in_global_excludes_shared_sampler_heaps", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        let mut vt = [0usize; 4];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        let mut device = Object { vt: vt.as_ptr() };
        let mut view_heap = Object { vt: vt.as_ptr() };
        let mut sampler_heap = Object { vt: vt.as_ptr() };
        unsafe {
            let dev = ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void);
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: dev.clone(),
                heap: ID3D12DescriptorHeap::from_raw(&mut view_heap as *mut _ as *mut c_void),
                cpu: 0x1000, gpu: 0x2000, inc: 32, cap: 1024, next: 64, free: Vec::new(), retired: VecDeque::new(), nulls: None, leases: PublishLeases::default(),
            });
            *GLOBAL_SAMPLER.lock().unwrap() = Some(GlobalSampler {
                dev,
                heap: ID3D12DescriptorHeap::from_raw(&mut sampler_heap as *mut _ as *mut c_void),
                cpu: 0x4000, inc: 32, cap: 17, next: 1, free: Vec::new(), retired: VecDeque::new(), leases: PublishLeases::default(),
            });
            if crate::gpu::uma::get().is_some() {
                assert!(register(0xC100, 16, 0x9000, 0, 16, true, true, true));
                assert!(register(0xC101, 16, 0x1000, 0, 32, false, true, true));
                assert!(register(0xC102, 4, 0xA000, 0, 16, true, false, true));
                assert!(!in_global(0xC100) && in_global_sampler(0xC100));
                assert!(in_global(0xC101) && !in_global_sampler(0xC101));
                assert!(!in_global(0xC102) && !in_global_sampler(0xC102));
                unregister(0xC100);
                unregister(0xC101);
                unregister(0xC102);
                // Adjacent 4-slot sampler heaps coalesce in GLOBAL_SAMPLER on unregister.
                assert!(register(0xC110, 4, 0x9000, 0, 16, true, true, true));
                assert!(register(0xC111, 4, 0x9100, 0, 16, true, true, true));
                assert_eq!(slot(start(0xC110).unwrap()).unwrap().index, 1);
                assert_eq!(slot(start(0xC111).unwrap()).unwrap().index, 5);
                unregister(0xC110);
                unregister(0xC111);
                assert_eq!(GLOBAL_SAMPLER.lock().unwrap().as_ref().unwrap().free, vec![(1, 8)]);
            } else {
                let mut heaps = HEAPS.lock().unwrap();
                heaps.push(Heap { generation: 1, obj: 0xC100, shadow: 0, n: 16, pc_cpu: 0, pc_gpu: 0, inc: 16, sampler: true, pages: vec![UNMAPPED; 2], visible: true, memory: Memory::Uma });
                heaps.push(Heap { generation: 2, obj: 0xC101, shadow: 0, n: 16, pc_cpu: 0, pc_gpu: 0, inc: 32, sampler: false, pages: vec![UNMAPPED], visible: true, memory: Memory::Uma });
                drop(heaps);
                assert!(!in_global(0xC100) && in_global_sampler(0xC100));
                assert!(in_global(0xC101) && !in_global_sampler(0xC101));
                HEAPS.lock().unwrap().clear();
            }
            *GLOBAL_SAMPLER.lock().unwrap() = None;
            *GLOBAL.lock().unwrap() = None;
        }
    }

    #[test]
    fn sampler_own_index_tracks_mapped_pages_and_fallbacks() {
        // Hosted shared sampler heap with two mapped pages at GLOBAL_SAMPLER indices 1 and 17.
        let pages = [1u32, 17u32];
        assert_eq!(sampler_own_index(&pages, 0), 1);
        assert_eq!(sampler_own_index(&pages, SAMPLER_PAGE_SLOTS - 1), SAMPLER_PAGE_SLOTS);
        assert_eq!(sampler_own_index(&pages, SAMPLER_PAGE_SLOTS), 17);
        assert_eq!(sampler_own_index(&pages, SAMPLER_PAGE_SLOTS + 5), 22);
        // Unmapped or out-of-range page in a hosted shared sampler heap maps to reserved slot 0.
        assert_eq!(sampler_own_index(&[UNMAPPED], 0), 0);
        assert_eq!(sampler_own_index(&[1, UNMAPPED], SAMPLER_PAGE_SLOTS + 1), 0);
        assert_eq!(sampler_own_index(&[1], SAMPLER_PAGE_SLOTS * 2), 0);
        // Unhosted sampler heap (empty pages) keeps local slot index k.
        assert_eq!(sampler_own_index(&[], 0), 0);
        assert_eq!(sampler_own_index(&[], 42), 42);
    }

    /// Released ranges in `GLOBAL` and `GLOBAL_SAMPLER` stay quarantined (and view descriptors
    /// are not overwritten with null views) while a live queue on the device still has work in
    /// flight, and are nulled and reclaimed once that queue's fence completes.
    #[test]
    fn released_shared_ranges_stay_quarantined_until_queue_completes() {
        const CHILD: &str = "XCL_TEST_SHARED_QUARANTINE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::released_shared_ranges_stay_quarantined_until_queue_completes", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        if crate::gpu::uma::get().is_none() {
            return;
        }
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            return crate::gpu::recomp::runner::skip_test("no GPU");
        };
        unsafe {
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            super::super::present::note_execute(queue.as_raw(), &[]);

            let base = crate::gpu::recomp::MEM_HEAP_BASE + crate::gpu::uma::MAX_HEAP_WINDOWS as u32;
            let view_cap = base + 2 * PAGE_SLOTS;
            let view_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: view_cap,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            }).unwrap();
            let null_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: NULL_BLOCK,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            }).unwrap();
            let view_inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let nulls_at = null_heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
            null_views_in(&dev, nulls_at, view_inc, NULL_BLOCK as u64);

            let sampler_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                NumDescriptors: 5,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            }).unwrap();
            let sampler_inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

            *GLOBAL.lock().unwrap() = Some(Global {
                dev: dev.clone(),
                cpu: view_heap.GetCPUDescriptorHandleForHeapStart().ptr as u64,
                gpu: view_heap.GetGPUDescriptorHandleForHeapStart().ptr,
                heap: view_heap,
                inc: view_inc,
                cap: view_cap,
                next: base,
                free: Vec::new(),
                retired: VecDeque::new(),
                nulls: Some((null_heap, nulls_at)),
                leases: PublishLeases::default(),
            });
            *GLOBAL_SAMPLER.lock().unwrap() = Some(GlobalSampler {
                dev: dev.clone(),
                cpu: sampler_heap.GetCPUDescriptorHandleForHeapStart().ptr as u64,
                heap: sampler_heap,
                inc: sampler_inc,
                cap: 5,
                next: 1,
                free: Vec::new(),
                retired: VecDeque::new(),
                leases: PublishLeases::default(),
            });

            assert!(register(0xD100, PAGE_SLOTS, 0x10_0000, 0, 32, false, true, true));
            assert!(register(0xD200, 4, 0x20_0000, 0, 16, true, true, true));
            let s_view = slot(start(0xD100).unwrap()).unwrap();
            let s_samp = slot(start(0xD200).unwrap()).unwrap();
            assert_eq!(s_view.index, base);
            assert_eq!(s_samp.index, 1);
            names().note(s_view.index, 0xCAFE_0001);
            assert_eq!(named_by_index(s_view.index), 0xCAFE_0001);

            unregister(0xD100);
            unregister(0xD200);

            // Queue is still waiting on `gate`: ranges must stay quarantined and live view names intact.
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(GLOBAL.lock().unwrap().as_ref().unwrap().free.is_empty(), "view range returned to free while queue in flight");
            assert!(GLOBAL_SAMPLER.lock().unwrap().as_ref().unwrap().free.is_empty(), "sampler range returned to free while queue in flight");
            assert_eq!(named_by_index(base), 0xCAFE_0001, "view range overwritten while queue in flight");

            assert!(register(0xD101, PAGE_SLOTS, 0x30_0000, 0, 32, false, true, true));
            assert!(register(0xD201, 4, 0x40_0000, 0, 16, true, true, true));
            assert_eq!(slot(start(0xD101).unwrap()).unwrap().shared_srv, 0, "quarantined view range must not be reused yet");
            assert_eq!(slot(start(0xD201).unwrap()).unwrap().shared_srv, 0, "quarantined sampler range must not be reused yet");

            // Complete the queue fence: ranges are nulled/cleared and reclaimed on subsequent allocation.
            gate.Signal(1).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut got_view = 0;
            let mut got_samp = 0;
            while (got_view == 0 || got_samp == 0) && std::time::Instant::now() < deadline {
                if got_view == 0 {
                    let s = slot(start(0xD101).unwrap()).unwrap();
                    if s.shared_srv != 0 {
                        got_view = s.index;
                    }
                }
                if got_samp == 0 {
                    let s = slot(start(0xD201).unwrap()).unwrap();
                    if s.shared_srv != 0 {
                        got_samp = s.index;
                    }
                }
                if got_view == 0 || got_samp == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
            assert_eq!(got_view, base, "view range reclaimed after fence completion");
            assert_eq!(got_samp, 1, "sampler range reclaimed after fence completion");
            assert_eq!(named_by_index(base), 0, "reclaimed view range had NAMES cleared before reuse");

            unregister(0xD101);
            unregister(0xD201);
            *GLOBAL_SAMPLER.lock().unwrap() = None;
            *GLOBAL.lock().unwrap() = None;
        }
    }

    #[test]
    fn retired_descriptor_ranges_stay_quarantined_until_all_queues_complete() {
        use std::sync::{atomic::{AtomicU64, Ordering}, Arc};
        let f1 = Arc::new(AtomicU64::new(0));
        let f2 = Arc::new(AtomicU64::new(0));
        let mut retired: VecDeque<RetiredRange<Arc<AtomicU64>>> = VecDeque::new();
        let mut free: Vec<(u32, u32)> = Vec::new();
        retired.push_back(RetiredRange { first: 64, count: 16, marks: Some(vec![(f1.clone(), 1), (f2.clone(), 1)]) });
        retired.push_back(RetiredRange { first: 80, count: 16, marks: Some(vec![(f1.clone(), 2), (f2.clone(), 1)]) });
        retired.push_back(RetiredRange { first: 96, count: 16, marks: None });

        let drain = |r: &mut VecDeque<RetiredRange<Arc<AtomicU64>>>, free: &mut Vec<(u32, u32)>| {
            for (first, count) in pop_ready_ranges(r, |f| f.load(Ordering::SeqCst)) {
                merge_release(free, first, count);
            }
        };

        // While f2 is still pending, neither range may re-enter `free`.
        f1.store(1, Ordering::SeqCst);
        drain(&mut retired, &mut free);
        assert!(free.is_empty() && retired.len() == 3);

        // Once both queues pass mark 1, only range 1 returns; range 2 waits for f1 == 2.
        f2.store(1, Ordering::SeqCst);
        drain(&mut retired, &mut free);
        assert_eq!(free, vec![(64, 16)]);
        assert_eq!(retired.len(), 2);

        // Passing mark 2 coalesces [80, 96) with [64, 80); failed-signal range stays quarantined.
        f1.store(2, Ordering::SeqCst);
        drain(&mut retired, &mut free);
        assert_eq!(free, vec![(64, 32)]);
        assert_eq!(retired.len(), 1);

        // A failed-signal (`None`) range at the front stays quarantined without wedging later healthy ranges.
        retired.push_back(RetiredRange { first: 112, count: 16, marks: Some(vec![(f1.clone(), 3), (f2.clone(), 1)]) });
        drain(&mut retired, &mut free);
        assert_eq!(free, vec![(64, 32)]);
        assert_eq!(retired.len(), 2);
        f1.store(3, Ordering::SeqCst);
        drain(&mut retired, &mut free);
        assert_eq!(free, vec![(64, 32), (112, 16)]);
        assert_eq!(retired.len(), 1);
    }

    /// While `drain_global_retired` is running `fill_nulls` outside `GLOBAL`, `GLOBAL_RETIRED`
    /// stays set and a concurrent `slot()` waits for the in-flight drain to finish merging into
    /// `GLOBAL.free` instead of returning the unmapped null slot (`shared_srv == 0`).
    #[test]
    fn slot_waits_for_in_flight_drain_before_returning_unmapped() {
        const CHILD: &str = "XCL_TEST_DRAIN_WINDOW_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::slot_waits_for_in_flight_drain_before_returning_unmapped", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        let mut vt = [0usize; 25];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        let mut dev_obj = Object { vt: vt.as_ptr() };
        let mut heap_obj = Object { vt: vt.as_ptr() };
        let mut null_obj = Object { vt: vt.as_ptr() };

        let base = crate::gpu::recomp::MEM_HEAP_BASE + crate::gpu::uma::MAX_HEAP_WINDOWS as u32;
        let range = 2 * PAGE_SLOTS;
        unsafe {
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: ID3D12Device::from_raw(&mut dev_obj as *mut _ as *mut c_void),
                heap: ID3D12DescriptorHeap::from_raw(&mut heap_obj as *mut _ as *mut c_void),
                cpu: 0x10_0000,
                gpu: 0x20_0000,
                inc: 32,
                cap: base + range,
                next: base + range,
                free: Vec::new(),
                retired: VecDeque::from([RetiredRange { first: base, count: range, marks: Some(Vec::new()) }]),
                nulls: Some((ID3D12DescriptorHeap::from_raw(&mut null_obj as *mut _ as *mut c_void), 0x90_0000)),
                leases: PublishLeases::default(),
            });
            GLOBAL_RETIRED.store(true, Release);
            HEAPS.lock().unwrap().push(Heap {
                obj: 0xE100,
                shadow: 0x50_0000,
                n: PAGE_SLOTS,
                pc_cpu: 0x60_0000,
                pc_gpu: 0,
                inc: 32,
                sampler: false,
                pages: vec![UNMAPPED],
                visible: true,
                memory: Memory::Uma,
                generation: NEXT_HEAP_GENERATION.fetch_add(1, Ordering::Relaxed).wrapping_add(1),
            });
        }

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let before_seq = GLOBAL_DRAIN_SEQ.load(Acquire);
        let drainer = std::thread::spawn(move || {
            drain_global_retired_with(before_seq, |_, _, _, _, _| {
                entered_tx.send(()).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(80));
            })
        });

        // Wait until `drainer` has popped `s.retired`, released `GLOBAL`, and is inside `on_nulls`.
        entered_rx.recv().unwrap();
        let retired_during_nulls = GLOBAL_RETIRED.load(Acquire);

        let s = slot(0x50_0000).unwrap();
        let drained = drainer.join().unwrap();
        assert!(retired_during_nulls, "GLOBAL_RETIRED dropped before merge_release completed");
        assert!(drained);
        assert_ne!(s.shared_srv, 0, "concurrent slot() got unmapped null slot during in-flight drain");
        assert_eq!(s.index, base, "concurrent slot() must receive the reclaimed descriptor range");
        assert!(!GLOBAL_RETIRED.load(Acquire));

        // Second path: `slot()` already past its entry drain (holding `HEAPS` when the drain starts)
        // hits the post-`page_of` fallback (`GLOBAL_RETIRED` -> `drain_global_retired_since`).
        {
            let mut g = GLOBAL.lock().unwrap();
            let gs = g.as_mut().unwrap();
            gs.free.clear();
            gs.retired.push_back(RetiredRange { first: base, count: range, marks: Some(Vec::new()) });
            GLOBAL_RETIRED.store(true, Release);
        }
        HEAPS.lock().unwrap()[0].pages[0] = UNMAPPED;
        let (entered2_tx, entered2_rx) = std::sync::mpsc::channel();
        let before_seq2 = GLOBAL_DRAIN_SEQ.load(Acquire);
        let drainer2 = std::thread::spawn(move || {
            drain_global_retired_with(before_seq2, |_, _, _, _, _| {
                entered2_tx.send(()).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(80));
            })
        });
        entered2_rx.recv().unwrap();
        let retired_during_nulls2 = GLOBAL_RETIRED.load(Acquire);
        let fallback_drained = drain_global_retired_since(before_seq2);
        let drained2 = drainer2.join().unwrap();
        assert!(retired_during_nulls2);
        assert!(drained2 && fallback_drained, "drain_global_retired_since must wait for and observe in-flight merge");
        assert_eq!(reserve_global(range), Some(base));

        HEAPS.lock().unwrap().clear();
        *GLOBAL.lock().unwrap() = None;
    }

    #[test]
    fn publish_range_stops_at_small_and_tail_heap_bounds() {
        const CHILD: &str = "XCL_TEST_PUBLISH_BOUNDS_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::publish_range_stops_at_small_and_tail_heap_bounds", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        use std::sync::atomic::{AtomicBool, Ordering};

        static LOCKS_FREE: AtomicBool = AtomicBool::new(true);
        static CALLS: Mutex<Vec<(u32, u64, u64, i32)>> = Mutex::new(Vec::new());
        #[repr(C)]
        struct Object { vt: *const usize }
        unsafe extern "system" fn retain(_this: *mut c_void) -> u32 { 1 }
        unsafe extern "system" fn copy_simple(
            _this: *mut c_void,
            n: u32,
            dst: D3D12_CPU_DESCRIPTOR_HANDLE,
            src: D3D12_CPU_DESCRIPTOR_HANDLE,
            ty: D3D12_DESCRIPTOR_HEAP_TYPE,
        ) {
            LOCKS_FREE.fetch_and(
                GLOBAL.try_lock().is_ok() && GLOBAL_SAMPLER.try_lock().is_ok() && HEAPS.try_lock().is_ok(),
                Ordering::Relaxed,
            );
            CALLS.lock().unwrap().push((n, dst.ptr as u64, src.ptr as u64, ty.0));
        }

        let mut vt = [0usize; 25];
        vt[1] = retain as usize;
        vt[2] = retain as usize;
        vt[24] = copy_simple as usize;
        let (mut device, mut view_heap, mut sampler_heap) =
            (Object { vt: vt.as_ptr() }, Object { vt: vt.as_ptr() }, Object { vt: vt.as_ptr() });
        let (mut source4, mut source3, mut source260) =
            (Object { vt: vt.as_ptr() }, Object { vt: vt.as_ptr() }, Object { vt: vt.as_ptr() });
        let view4 = vec![0u8; (4 * SLOT) as usize];
        let sampler3 = vec![0u8; (3 * SAMPLER_SLOT) as usize];
        let view260 = vec![0u8; (260 * SLOT) as usize];
        unsafe {
            let dev = ID3D12Device::from_raw(&mut device as *mut _ as *mut c_void);
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: dev.clone(),
                heap: ID3D12DescriptorHeap::from_raw(&mut view_heap as *mut _ as *mut c_void),
                cpu: 0x10_0000,
                gpu: 0x18_0000,
                inc: 32,
                cap: 4096,
                next: 64,
                free: Vec::new(),
                leases: PublishLeases::default(), retired: VecDeque::new(),
                nulls: None,
            });
            *GLOBAL_SAMPLER.lock().unwrap() = Some(GlobalSampler {
                dev,
                heap: ID3D12DescriptorHeap::from_raw(&mut sampler_heap as *mut _ as *mut c_void),
                cpu: 0x20_0000,
                inc: 16,
                cap: 256,
                next: 1,
                free: Vec::new(),
                leases: PublishLeases::default(), retired: VecDeque::new(),
            });
            {
                let mut heaps = HEAPS.lock().unwrap();
                heaps.push(Heap { generation: 1, obj: &mut source4 as *mut Object as usize, shadow: view4.as_ptr() as usize, n: 4, pc_cpu: 0x30_0000, pc_gpu: 0, inc: 32, sampler: false, pages: vec![UNMAPPED], visible: true, memory: Memory::Uma });
                heaps.push(Heap { generation: 2, obj: &mut source3 as *mut Object as usize, shadow: sampler3.as_ptr() as usize, n: 3, pc_cpu: 0x40_0000, pc_gpu: 0, inc: 16, sampler: true, pages: vec![UNMAPPED], visible: true, memory: Memory::Uma });
                heaps.push(Heap { generation: 3, obj: &mut source260 as *mut Object as usize, shadow: view260.as_ptr() as usize, n: 260, pc_cpu: 0x50_0000, pc_gpu: 0, inc: 32, sampler: false, pages: vec![UNMAPPED; 2], visible: true, memory: Memory::Uma });
            }
            let v1 = slot(view4.as_ptr() as u64 + SLOT as u64).unwrap();
            publish_range(&v1, 10, &[]);
            let s1 = slot(sampler3.as_ptr() as u64 + SAMPLER_SLOT as u64).unwrap();
            publish_range(&s1, 10, &[]);
            let v250 = slot(view260.as_ptr() as u64 + (250 * SLOT) as u64).unwrap();
            publish_range(&v250, 20, &[]);
            HEAPS.lock().unwrap().clear();
            *GLOBAL_SAMPLER.lock().unwrap() = None;
            *GLOBAL.lock().unwrap() = None;
        }
        assert!(LOCKS_FREE.load(Ordering::Relaxed), "descriptor copies must run without heap locks");
        assert_eq!(
            *CALLS.lock().unwrap(),
            vec![
                (6, 0x10_0000 + 66 * 32, 0x30_0000 + 2 * 32, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0),
                (2, 0x20_0000 + 2 * 16, 0x40_0000 + 16, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0),
                (12, 0x10_0000 + (72 + 500) * 32, 0x50_0000 + 500 * 32, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0),
                (8, 0x10_0000 + 584 * 32, 0x50_0000 + 512 * 32, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0),
            ]
        );
    }

    #[test]
    fn publish_calls_run_unlocked_and_hold_generation_and_ranges() {
        const CHILD: &str = "XCL_TEST_SINGLE_PUBLISH_LEASE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::console_heap::tests::publish_calls_run_unlocked_and_hold_generation_and_ranges", "--nocapture"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        }
        use std::ffi::c_void;
        use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier, OnceLock};

        static GATE: OnceLock<(Arc<Barrier>, Arc<Barrier>)> = OnceLock::new();
        static LOCKS_FREE: AtomicBool = AtomicBool::new(false);
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        #[repr(C)]
        struct Object { vt: *const usize, refs: AtomicU32 }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            (*this.cast::<Object>()).refs.fetch_add(1, Ordering::Relaxed) + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            (*this.cast::<Object>()).refs.fetch_sub(1, Ordering::Relaxed) - 1
        }
        unsafe extern "system" fn copy_simple(_this: *mut c_void, n: u32, dst: D3D12_CPU_DESCRIPTOR_HANDLE, src: D3D12_CPU_DESCRIPTOR_HANDLE, ty: D3D12_DESCRIPTOR_HEAP_TYPE) {
            LOCKS_FREE.store(GLOBAL.try_lock().is_ok() && GLOBAL_SAMPLER.try_lock().is_ok() && HEAPS.try_lock().is_ok(), Ordering::Relaxed);
            let call = CALLS.fetch_add(1, Ordering::Relaxed);
            assert_eq!(
                (n, dst.ptr, src.ptr, ty),
                if call == 0 {
                    (1, 0x10_0000 + 64 * 32, 0x30_0000, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                } else {
                    (8, 0x10_0000 + 80 * 32, 0x30_0000, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                },
            );
            let (entered, leave) = GATE.get().unwrap();
            entered.wait();
            leave.wait();
        }
        let entered = Arc::new(Barrier::new(2));
        let leave = Arc::new(Barrier::new(2));
        GATE.set((entered.clone(), leave.clone())).ok();
        let mut vt = [0usize; 25];
        vt[1] = add_ref as usize;
        vt[2] = release as usize;
        vt[24] = copy_simple as usize;
        let mut device = Object { vt: vt.as_ptr(), refs: AtomicU32::new(1) };
        let mut global_heap = Object { vt: vt.as_ptr(), refs: AtomicU32::new(1) };
        let mut source_heap = Object { vt: vt.as_ptr(), refs: AtomicU32::new(1) };
        let shadow = vec![0u8; (4 * SLOT) as usize];
        unsafe {
            *GLOBAL.lock().unwrap() = Some(Global {
                dev: ID3D12Device::from_raw((&mut device as *mut Object).cast()),
                heap: ID3D12DescriptorHeap::from_raw((&mut global_heap as *mut Object).cast()),
                cpu: 0x10_0000, gpu: 0x18_0000, inc: 32, cap: 4096, next: 72,
                free: Vec::new(), leases: PublishLeases::default(), retired: VecDeque::new(),
                nulls: None,
            });
            HEAPS.lock().unwrap().push(Heap {
                generation: 77, obj: &mut source_heap as *mut Object as usize, shadow: shadow.as_ptr() as usize,
                n: 4, pc_cpu: 0x30_0000, pc_gpu: 0, inc: 32, sampler: false, pages: vec![64], visible: true, memory: Memory::Uma,
            });
            let stale = slot(shadow.as_ptr() as u64).unwrap();
            let worker = std::thread::spawn(move || publish(&stale, false, 0));
            entered.wait();
            assert_eq!(source_heap.refs.load(Ordering::Relaxed), 2, "source heap must stay owned during the device call");
            let Taken(old) = take(&mut source_heap as *mut Object as usize).unwrap();
            release_global(old.pages[0], page_descriptors(old.n, 0));
            assert_ne!(reserve_global(8), Some(64), "an in-flight destination range must not be recycled");
            HEAPS.lock().unwrap().push(Heap { generation: 78, pages: vec![80], ..old });
            leave.wait();
            worker.join().unwrap();
            assert!(LOCKS_FREE.load(Ordering::Relaxed));
            assert_eq!(source_heap.refs.load(Ordering::Relaxed), 1);
            assert_eq!(reserve_global(8), Some(64), "deferred range must return when publication finishes");
            publish(&stale, false, 0);
            assert_eq!(CALLS.load(Ordering::Relaxed), 1, "a stale generation must not publish into its replacement");

            GLOBAL.lock().unwrap().as_mut().unwrap().next = 88;
            let stale_range = slot(shadow.as_ptr() as u64).unwrap();
            let worker = std::thread::spawn(move || publish_range(&stale_range, 4, &[]));
            entered.wait();
            assert_eq!(source_heap.refs.load(Ordering::Relaxed), 2, "range source heap must stay owned during the device call");
            let Taken(old) = take(&mut source_heap as *mut Object as usize).unwrap();
            release_global(old.pages[0], page_descriptors(old.n, 0));
            assert_ne!(reserve_global(8), Some(80), "an in-flight range destination must not be recycled");
            HEAPS.lock().unwrap().push(Heap { generation: 79, pages: vec![96], ..old });
            leave.wait();
            worker.join().unwrap();
            assert_eq!(source_heap.refs.load(Ordering::Relaxed), 1);
            assert_eq!(reserve_global(8), Some(80), "deferred range must return when range publication finishes");
            publish_range(&stale_range, 4, &[]);
            assert_eq!(CALLS.load(Ordering::Relaxed), 2, "a stale range generation must not publish into its replacement");
            HEAPS.lock().unwrap().clear();
            *GLOBAL.lock().unwrap() = None;
        }
        assert_eq!((device.refs.load(Ordering::Relaxed), global_heap.refs.load(Ordering::Relaxed), source_heap.refs.load(Ordering::Relaxed)), (0, 0, 1));
    }
}
