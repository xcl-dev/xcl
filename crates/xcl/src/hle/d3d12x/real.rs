//! Real D3D12.X backend: the title gets genuine D3D12 objects from the PC driver.
//!
//! Every object handed to the title is "adopted": its vtable pointer is replaced with
//! an extended copy:
//!
//! ```text
//! ext[-1]                 original vtable (for calling originals)
//! ext[0 .. public)        original PC methods (a few overridden, see below)
//! ext[public .. internal) guard thunks: runtime/DXGI calls → original internal virtual,
//!                         calls from the game image → logged as console-only method
//! ext[internal .. +64)    logging stubs (console-only methods past the end)
//! ```
//!
//! Overrides: QueryInterface (logs unknown IIDs), every Create* on the device (adopts
//! the new object, retries with the PC IID when the console one is unknown, captures
//! shaders, builds fallback pipeline states when console shaders are rejected),
//! CreateRenderTargetView / OMSetRenderTargets / ExecuteCommandLists (presentation).

#![allow(non_snake_case)]

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use serde::Serialize;
use windows::core::{Interface, PCSTR};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows_sys::core::GUID;

use super::layout::{self, Iface};
use crate::stubs;
use crate::util::{HRESULT, E_INVALIDARG, E_NOINTERFACE, E_OUTOFMEMORY, S_FALSE, S_OK};

const DLL: &str = "d3d12_xs.dll";
const EXTRA_SLOTS: usize = 64;
const MAX_SCAN: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kind {
    Device,
    CommandQueue,
    CommandAllocator,
    CommandList,
    /// the console's DMA (copy) list, ID3D12XboxDmaCommandList: a PC list with the console's
    /// shorter vtable (`console_remap`)
    DmaList,
    Fence,
    DescriptorHeap,
    Resource,
    Heap,
    PipelineState,
    RootSignature,
    QueryHeap,
    CommandSignature,
    PipelineLibrary,
    StateObject,
}

#[cfg(test)]
mod slot_cache_tests {
    use super::*;

    #[test]
    fn cached_hot_methods_keep_one_nonzero_slot_across_interface_versions() {
        let families: [(&str, &[&str]); 5] = [
            (
                "ID3D12GraphicsCommandList",
                &[
                    "Close",
                    "Reset",
                    "ClearState",
                    "DrawInstanced",
                    "DrawIndexedInstanced",
                    "SetPipelineState",
                    "OMSetRenderTargets",
                    "ClearRenderTargetView",
                    "ExecuteIndirect",
                ],
            ),
            ("ID3D12Resource", &["Map", "Unmap", "WriteToSubresource"]),
            ("ID3D12CommandQueue", &["ExecuteCommandLists"]),
            ("ID3D12CommandAllocator", &["Reset"]),
            ("ID3D12Fence", &["Signal"]),
        ];
        for (base, methods) in families {
            let versions = layout::chain(base);
            assert!(!versions.is_empty(), "{base}");
            for method in methods {
                let slots: BTreeSet<_> = versions.iter().filter_map(|iface| layout::slot(iface, method)).collect();
                assert_eq!(slots.len(), 1, "{base}::{method} at {slots:?}");
                assert_ne!(slots.first(), Some(&0), "slot zero is the cache's empty value");
            }
        }
    }
}

impl Kind {
    pub fn base(self) -> &'static str {
        match self {
            Kind::Device => "ID3D12Device",
            Kind::CommandQueue => "ID3D12CommandQueue",
            Kind::CommandAllocator => "ID3D12CommandAllocator",
            Kind::CommandList | Kind::DmaList => "ID3D12GraphicsCommandList",
            Kind::Fence => "ID3D12Fence",
            Kind::DescriptorHeap => "ID3D12DescriptorHeap",
            Kind::Resource => "ID3D12Resource",
            Kind::Heap => "ID3D12Heap",
            Kind::PipelineState => "ID3D12PipelineState",
            Kind::RootSignature => "ID3D12RootSignature",
            Kind::QueryHeap => "ID3D12QueryHeap",
            Kind::CommandSignature => "ID3D12CommandSignature",
            Kind::PipelineLibrary => "ID3D12PipelineLibrary",
            Kind::StateObject => "ID3D12StateObject",
        }
    }
}

// ---------------------------------------------------------------------------------------
// statistics for the report
// ---------------------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize)]
pub struct ClassStat {
    pub interface: String,
    pub public_methods: usize,
    pub internal_slots: usize,
}

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize)]
pub struct D3d12Stats {
    pub backend: String,
    pub adapter: String,
    pub classes: Vec<ClassStat>,
    pub unknown_iids: Vec<String>,
    pub fallback_pipelines: u64,
    pub failed_pipelines: u64,
    pub fallback_root_signatures: u64,
    pub presents: u64,
    pub frame_boundaries: u64,
    /// Optional content evidence from the existing frame-dump readback; no new GPU readback.
    #[serde(default)]
    pub last_frame_dump: Option<super::present::FrameDumpSample>,
    /// unified memory region opened as a GPU heap (`gpu::uma`): size, bytes in use, buffer VA
    #[serde(default)]
    pub uma_mb: u64,
    #[serde(default)]
    pub uma_used_kb: u64,
    #[serde(default)]
    pub uma_gpu_va: String,
    #[serde(default)]
    pub uma_error: String,
    /// whether the GPU sees CPU writes to the coherent windows (`check_uma_import`)
    #[serde(default)]
    pub uma_import_check: String,
    /// graphics pipelines built from recompiled console shaders (XCL shader ABI root signature)
    #[serde(default)]
    pub translated_pipelines: u64,
    /// why console shaders could not be used (first messages)
    #[serde(default)]
    pub translation_errors: Vec<String>,
    /// pipelines built in the background behind a stand-in: built, failed, still building
    #[serde(default)]
    pub background_pipelines: (usize, usize, usize),
    /// Opt-in compute-pipeline pressure lifecycle totals.
    #[serde(default)]
    pub pipeline_pressure: super::pso_async::PipelinePressureStats,
    /// DXIL cache and compiler statistics.
    #[serde(default)]
    pub dxil_cache: crate::gpu::recomp::dxc::CacheStats,
    /// Native title `ExecuteCommandLists` chunks timed while `XCL_GPU_TIMING=1`. These are CPU
    /// durations inside the native call, not GPU execution or completion time.
    #[serde(default)]
    pub submissions: u64,
    #[serde(default)]
    pub slowest_submission_ms: f64,
    #[serde(default)]
    pub slowest_submission_lists: usize,
    #[serde(default)]
    pub submission_total_ns: u64,
    #[serde(default)]
    pub submission_mean_ms: f64,
    #[serde(default)]
    pub submission_p95_upper_ms: f64,
    #[serde(default)]
    pub submission_p99_upper_ms: f64,
    /// Counts in base-2 nanosecond buckets. Bucket zero includes zero and one nanosecond.
    #[serde(default)]
    pub submission_histogram_log2_ns: Vec<u64>,
    /// Opt-in title submissions correlated with an exact queue-generation fence receipt. These
    /// are completion observations, not GPU timestamp durations.
    #[serde(default)]
    pub gpu_completion_receipts_tracked: u64,
    #[serde(default)]
    pub gpu_completion_completed: u64,
    /// Receipts retired because their fence reported the D3D12 device-removed sentinel.
    #[serde(default)]
    pub gpu_completion_device_lost: u64,
    #[serde(default)]
    pub gpu_completion_pending: usize,
    /// Tracked receipts evicted because the bounded pending queue was full.
    #[serde(default)]
    pub gpu_completion_capacity_evictions: u64,
    /// Submitted chunks for which an exact queue marker could not be created.
    #[serde(default)]
    pub gpu_completion_receipt_failures: u64,
    /// Exact receipts discarded rather than blocking a submitter on report collection.
    #[serde(default)]
    pub gpu_completion_tracker_contentions: u64,
}

static ADAPTER: Mutex<String> = Mutex::new(String::new());
static ADAPTER_VENDOR: AtomicU32 = AtomicU32::new(0x1414);
static UNKNOWN_IIDS: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
static FALLBACK_PSO: AtomicU64 = AtomicU64::new(0);
static FAILED_PSO: AtomicU64 = AtomicU64::new(0);
static FALLBACK_RS: AtomicU64 = AtomicU64::new(0);
static TRANSLATED_PSO: AtomicU64 = AtomicU64::new(0);
static TRANSLATE_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

#[derive(Clone, Copy)]
struct SubmissionTiming {
    submissions: u64,
    total_ns: u64,
    slowest_ns: u64,
    slowest_lists: usize,
    histogram: [u64; crate::stall::HISTOGRAM_BUCKETS],
}

impl Default for SubmissionTiming {
    fn default() -> Self {
        Self {
            submissions: 0,
            total_ns: 0,
            slowest_ns: 0,
            slowest_lists: 0,
            histogram: [0; crate::stall::HISTOGRAM_BUCKETS],
        }
    }
}

impl SubmissionTiming {
    fn record(&mut self, ns: u64, lists: usize) {
        self.submissions = self.submissions.saturating_add(1);
        self.total_ns = self.total_ns.saturating_add(ns);
        let bucket = submission_histogram_bucket(ns);
        self.histogram[bucket] = self.histogram[bucket].saturating_add(1);
        if ns > self.slowest_ns {
            self.slowest_ns = ns;
            self.slowest_lists = lists;
        }
    }

    fn percentile_upper_ns(&self, percentile: u64) -> u64 {
        if self.submissions == 0 { return 0 }
        let target = ((self.submissions as u128 * percentile as u128 + 99) / 100) as u64;
        let mut cumulative = 0u64;
        let bucket = self.histogram.iter().position(|count| {
            cumulative = cumulative.saturating_add(*count);
            cumulative >= target
        }).unwrap_or(self.histogram.len() - 1);
        if bucket + 1 >= u64::BITS as usize { u64::MAX } else { (1u64 << (bucket + 1)) - 1 }
    }
}

const fn submission_histogram_bucket(ns: u64) -> usize {
    if ns == 0 { 0 } else { (u64::BITS - 1 - ns.leading_zeros()) as usize }
}

static SUBMISSION_TIMING: Mutex<SubmissionTiming> = Mutex::new(SubmissionTiming {
    submissions: 0,
    total_ns: 0,
    slowest_ns: 0,
    slowest_lists: 0,
    histogram: [0; crate::stall::HISTOGRAM_BUCKETS],
});

#[allow(dead_code)] // sequence metadata is consumed by the GPU timestamp-query stage
struct PendingGpuCompletion {
    receipt: super::alloc_pool::SubmissionReceipt,
    sequence: u64,
    lists: usize,
    queue: usize,
}

#[derive(Default)]
struct GpuCompletionTiming {
    tracked: u64,
    completed: u64,
    device_lost: u64,
    capacity_evictions: u64,
    pending: VecDeque<PendingGpuCompletion>,
}

impl GpuCompletionTiming {
    const MAX_PENDING: usize = 16 * 1024;

    fn push(
        &mut self,
        receipt: super::alloc_pool::SubmissionReceipt,
        lists: usize,
        queue: usize,
    ) {
        self.push_with_limit(receipt, lists, queue, Self::MAX_PENDING);
    }

    fn push_with_limit(
        &mut self,
        receipt: super::alloc_pool::SubmissionReceipt,
        lists: usize,
        queue: usize,
        max_pending: usize,
    ) {
        debug_assert!(max_pending > 0);
        self.tracked = self.tracked.saturating_add(1);
        if self.pending.len() == max_pending {
            self.pending.pop_front();
            self.capacity_evictions = self.capacity_evictions.saturating_add(1);
        }
        self.pending.push_back(PendingGpuCompletion { receipt, sequence: self.tracked, lists, queue });
    }

    /// Poll exact queue-generation receipts from report collection, never from title submission.
    /// `GetCompletedValue` does not wait; incomplete samples stay owned for a later report poll.
    fn retire_completed(&mut self) {
        let mut pending = VecDeque::with_capacity(self.pending.len());
        for sample in self.pending.drain(..) {
            match sample.receipt.status() {
                super::alloc_pool::SubmissionReceiptStatus::Complete => {
                    self.completed = self.completed.saturating_add(1);
                }
                super::alloc_pool::SubmissionReceiptStatus::DeviceLost => {
                    self.device_lost = self.device_lost.saturating_add(1);
                }
                super::alloc_pool::SubmissionReceiptStatus::Pending => pending.push_back(sample),
            }
        }
        self.pending = pending;
    }
}

static GPU_COMPLETION_TIMING: Mutex<GpuCompletionTiming> = Mutex::new(GpuCompletionTiming {
    tracked: 0,
    completed: 0,
    device_lost: 0,
    capacity_evictions: 0,
    pending: VecDeque::new(),
});
static GPU_COMPLETION_RECEIPT_FAILURES: AtomicU64 = AtomicU64::new(0);
static GPU_COMPLETION_TRACKER_CONTENTIONS: AtomicU64 = AtomicU64::new(0);

fn gpu_timing_on_value(value: Option<&str>) -> bool {
    value == Some("1")
}

pub(super) fn gpu_timing_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gpu_timing_on_value(std::env::var("XCL_GPU_TIMING").ok().as_deref()))
}

pub(super) fn note_gpu_completion_receipt(
    receipt: super::alloc_pool::SubmissionReceipt,
    lists: usize,
    queue: usize,
) {
    if !gpu_timing_on() {
        return;
    }
    // Instrumentation must never serialize title submitters behind report collection.
    try_note_gpu_completion_receipt(
        &GPU_COMPLETION_TIMING,
        &GPU_COMPLETION_TRACKER_CONTENTIONS,
        receipt,
        lists,
        queue,
    );
}

fn try_note_gpu_completion_receipt(
    timing: &Mutex<GpuCompletionTiming>,
    contentions: &AtomicU64,
    receipt: super::alloc_pool::SubmissionReceipt,
    lists: usize,
    queue: usize,
) -> bool {
    if let Ok(mut timing) = timing.try_lock() {
        timing.push(receipt, lists, queue);
        true
    } else {
        contentions.fetch_add(1, Ordering::Relaxed);
        false
    }
}

pub(super) fn note_gpu_completion_receipt_failure() {
    GPU_COMPLETION_RECEIPT_FAILURES.fetch_add(1, Ordering::Relaxed);
}

fn slow_submission_us_value(value: Option<&str>) -> u64 {
    value.and_then(|value| value.parse::<u64>().ok()).unwrap_or(50).saturating_mul(1_000)
}

fn slow_submission_us() -> u64 {
    static US: OnceLock<u64> = OnceLock::new();
    *US.get_or_init(|| slow_submission_us_value(std::env::var("XCL_GPU_TIMING_MS").ok().as_deref()))
}

fn note_submission_timing(elapsed: std::time::Duration, lists: usize, queue: *mut c_void) {
    let ns = elapsed.as_nanos().min(u64::MAX as u128) as u64;
    if let Ok(mut timing) = SUBMISSION_TIMING.lock() {
        timing.record(ns, lists);
    }
    let us = ns / 1_000;
    if us >= slow_submission_us() {
        crate::xlog_warn!("ExecuteCommandLists: {us} µs of CPU time for {lists} list(s), queue {queue:p}");
    }
}

/// XCL shader ABI root signature on the primary backend device (raw, kept for the process lifetime)
static ABI_RS: AtomicUsize = AtomicUsize::new(0);
static ABI_RS_DEV: AtomicUsize = AtomicUsize::new(0);
static ABI_RS_BY_DEV: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// XCL's shader ABI root signature, once a translated pipeline created it.
pub fn abi_rs_cached() -> Option<*mut c_void> {
    let dev = device_raw() as usize;
    let primary = ABI_RS.load(Ordering::Acquire);
    if primary != 0 && (dev == 0 || ABI_RS_DEV.load(Ordering::Acquire) == dev) {
        return Some(primary as *mut c_void);
    }
    ABI_RS_BY_DEV.lock().unwrap().as_ref()?.get(&dev).map(|&p| p as *mut c_void)
}

thread_local! {
    /// the last compute pipeline created on this thread runs a translated console shader
    static LAST_COMPUTE_TRANSLATED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// … with this root signature (static samplers; 0 = the plain ABI one)
    static LAST_COMPUTE_RS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Root signature of the pipelines built from translated console shaders (bind it, then the
/// XCL ABI root UAVs io / res / mem, to draw with them).
pub fn abi_root_signature() -> Option<ID3D12RootSignature> {
    let raw = abi_rs_cached()?;
    unsafe { ID3D12RootSignature::from_raw_borrowed(&raw).cloned() }
}
/// Each UMA window as one D3D12 buffer (placed in a heap opened over the process memory):
/// `mem`, `mem1`.. of recompiled shaders. (buffer, GPU VA) per window, kept for the process
/// lifetime.
static UMA_BUFFER: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
/// the committed UMA windows opened as D3D12 heaps (indexed by window `k`; `None` for sparse windows)
static UMA_HEAPS: Mutex<Vec<Option<usize>>> = Mutex::new(Vec::new());
/// console buffers placed in unified memory: resource → (address, bytes)
/// address, bytes, whether XCL allocated the memory (not: the title's own, CreatePlacedResourceX)
type UmaBuffer = (usize, usize, bool);
static UMA_BUFFERS: Mutex<Option<HashMap<usize, UmaBuffer>>> = Mutex::new(None);

/// Where a console-created buffer lives in unified memory (its console GPU address).
pub fn console_buffer(res: usize) -> Option<(usize, usize)> {
    UMA_BUFFERS.lock().unwrap().as_ref()?.get(&res).map(|&(p, n, _)| (p, n))
}

/// Number of live console buffers in unified memory.
pub fn console_buffers() -> usize {
    UMA_BUFFERS.lock().unwrap().as_ref().map_or(0, |m| m.len())
}
static UMA_ERROR: Mutex<String> = Mutex::new(String::new());

/// GPU virtual address of the unified memory buffer of window 0 (0: not available).
pub fn uma_gpu_va() -> u64 {
    uma_window_vas().first().copied().unwrap_or(0)
}

/// GPU virtual addresses of the UMA window buffers (`u2`, `u3`.. of the shader ABI).
pub fn uma_window_vas() -> Vec<u64> {
    UMA_BUFFER.lock().map(|g| g.iter().map(|b| b.1).collect()).unwrap_or_default()
}

/// The window buffer holding `len` bytes at UMA address `p`, and their offset in it.
pub(super) fn uma_window_buffer(p: usize, len: usize) -> Option<(usize, u64)> {
    let u = crate::gpu::uma::get()?;
    let k = u.window_of(p)?;
    let (base, size) = u.window(k);
    if p - base + len > size {
        return None;
    }
    Some((UMA_BUFFER.lock().ok()?.get(k)?.0, (p - base) as u64))
}

/// GPU virtual address of a UMA address (None: not in UMA or not open on the GPU).
pub fn uma_va_of(p: usize) -> Option<u64> {
    let u = crate::gpu::uma::get()?;
    let k = u.window_of(p)?;
    Some(*uma_window_vas().get(k)? + (p - u.window(k).0) as u64)
}

/// What the start-up check of the coherent UMA windows found (`check_uma_import`): empty – not
/// run, "ok …", "off (XCL_UMA_CHECK=0)", or why the GPU does not see the CPU's writes.
static UMA_IMPORT_CHECK: Mutex<String> = Mutex::new(String::new());

/// A word the CPU writes into each coherent window, read back by a GPU copy. The windows are
/// imported as heaps (`OpenExistingHeapFromAddress`), and a runtime that cannot import them may
/// still succeed with memory of its own: vkd3d-proton under a Wine without `userfaultfd` write
/// watches (#175). The GPU then reads zeros where the title wrote, and nothing else says so.
/// Off with `XCL_UMA_CHECK=0`.
unsafe fn check_uma_import(device: &ID3D12Device, u: &crate::gpu::uma::Uma, bufs: &[(usize, u64)]) -> Result<String, String> {
    let t0 = std::time::Instant::now();
    let wins: Vec<usize> = (0..bufs.len()).filter(|&k| u.is_committed_window(k) && bufs[k].0 != 0).collect();
    if wins.is_empty() {
        return Ok("no coherent window".into());
    }
    let queue: ID3D12CommandQueue =
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| format!("check not run: queue: {e}"))?;
    let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| format!("check not run: allocator: {e}"))?;
    let list: ID3D12GraphicsCommandList =
        device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| format!("check not run: list: {e}"))?;
    let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| format!("check not run: fence: {e}"))?;
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
    let mut readback: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback)
        .map_err(|e| format!("check not run: readback buffer: {e}"))?;
    let readback = readback.ok_or("check not run: readback buffer")?;
    // a block of the window's own (title data is not touched), one word per window
    let mut probes = Vec::new();
    for (i, &k) in wins.iter().take(64).enumerate() {
        let Some(p) = u.alloc_in(k, 4, 0) else { continue };
        let canary = 0x5843_4C00 | k as u32; // "XCL" + the window
        std::ptr::write_volatile(p as *mut u32, canary);
        if let Some(src) = ID3D12Resource::from_raw_borrowed(&(bufs[k].0 as *mut c_void)) {
            list.CopyBufferRegion(&readback, 4 * i as u64, src, (p as usize - u.window(k).0) as u64, 4);
        }
        probes.push((k, i, p as usize, canary));
    }
    let run = || -> Result<Vec<u32>, String> {
        list.Close().map_err(|e| format!("check not run: Close: {e}"))?;
        let list = list.cast::<ID3D12CommandList>().map_err(|e| e.to_string())?;
        native_queue_execute(&queue, &list);
        native_queue_signal(&queue, &fence, 1).map_err(|e| format!("check not run: Signal: {e}"))?;
        crate::gpu::fence::wait(&fence, 1, device, None, 2000).map_err(|e| format!("check not run: {e}"))?;
        let mut m = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut m)).map_err(|e| format!("check not run: Map: {e}"))?;
        let got = std::slice::from_raw_parts(m as *const u32, 64).to_vec();
        readback.Unmap(0, None);
        Ok(got)
    };
    let got = match run() {
        Ok(got) => got,
        Err(e) => {
            // The copy may still be running: what it uses stays (a few KiB, once). Released under
            // the GPU, vkd3d-proton frees it at once and the GPU faults (RADV: page faults, then a
            // ring timeout) – a slow first use of the imported window is enough to get here.
            std::mem::forget((queue, alloc, list, fence, readback));
            return Err(e);
        }
    };
    for &(_, _, p, _) in &probes {
        u.free(p);
    }
    let wrong: Vec<String> =
        probes.iter().filter(|&&(_, i, _, c)| got[i] != c).map(|&(k, i, _, c)| format!("window {k}: wrote {c:#x}, the GPU read {:#x}", got[i])).collect();
    if !wrong.is_empty() {
        return Err(format!(
            "the GPU does not see what the CPU writes to unified memory ({}): the host-memory import fell back to memory of its own (Wine without userfaultfd write watches?); shaders will read zeros or stale data where the title wrote",
            wrong.join("; ")
        ));
    }
    Ok(format!("ok, {} window(s), {} µs", probes.len(), t0.elapsed().as_micros()))
}

unsafe fn open_uma(device: &ID3D12Device, allow_driver_remap: bool) {
    let Some(u) = crate::gpu::uma::get() else {
        *UMA_ERROR.lock().unwrap() = "no UMA region".into();
        return;
    };
    let open = |k: usize| -> Result<(usize, u64, usize), String> {
        let (base, size) = u.window(k);
        if allow_driver_remap && super::uma_driver::wanted(device) {
            match super::uma_driver::remake(device, base, size) {
                Ok((heap, buf)) => {
                    u.set_driver_window(k);
                    return Ok(super::uma_driver::raw(heap, buf));
                }
                Err(e) => crate::xlog_warn!("D3D12.X: coherent window {k} stays imported: {e}"),
            }
        }
        let dev3: ID3D12Device3 = device.cast().map_err(|e| format!("ID3D12Device3: {e}"))?;
        let heap: ID3D12Heap = dev3.OpenExistingHeapFromAddress(base as *const c_void).map_err(|e| format!("OpenExistingHeapFromAddress (window {k}): {e}"))?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        device.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).map_err(|e| format!("CreatePlacedResource: {e}"))?;
        let buf = buf.ok_or("CreatePlacedResource")?;
        let va = buf.GetGPUVirtualAddress();
        Ok((buf.into_raw() as usize, va, heap.into_raw() as usize))
    };
    let n = u.shader_windows();
    let mut bufs: Vec<(usize, u64)> = vec![(0, 0); n];
    let mut heaps: Vec<Option<usize>> = vec![None; n];
    for k in 0..n {
        if !u.is_committed_window(k) {
            continue;
        }
        match open(k) {
            Ok((buf_raw, va, heap_raw)) => {
                bufs[k] = (buf_raw, va);
                heaps[k] = Some(heap_raw);
            }
            Err(e) => {
                crate::xlog_warn!("D3D12.X: UMA not available to the GPU: {e}");
                *UMA_ERROR.lock().unwrap() = e;
                return;
            }
        }
    }
    let check = if std::env::var("XCL_UMA_CHECK").is_ok_and(|v| v == "0") {
        Ok("off (XCL_UMA_CHECK=0)".into())
    } else {
        check_uma_import(device, u, &bufs)
    };
    match &check {
        Ok(s) => crate::xlog_info!("D3D12.X: UMA import check: {s}"),
        Err(e) => crate::xlog_error!("D3D12.X: UMA import check: {e}"),
    }
    *UMA_IMPORT_CHECK.lock().unwrap() = check.unwrap_or_else(|e| e);
    let sparse = super::sparse::init(device);
    for (k, buf_raw, va) in &sparse {
        if let Some(slot) = bufs.get_mut(*k) {
            *slot = (*buf_raw, *va);
        }
    }
    let w0_va = bufs.first().map_or(0, |b| b.1);
    crate::xlog_info!(
        "D3D12.X: UMA opened ({} sparse VRAM window(s), {} MB coherent window), window 0 buffer at VA {:#x}",
        sparse.len(),
        u.size() >> 20,
        w0_va
    );
    *UMA_HEAPS.lock().unwrap() = heaps;
    // shaders take the windows from the shared descriptor heap (`pso::heap_windows`)
    if n > 1 && crate::gpu::recomp::pso::heap_windows() {
        let views: Vec<(usize, usize)> = bufs.iter().enumerate().map(|(k, b)| (b.0, u.window(k).1)).collect();
        if super::console_heap::set_window_views(device, &views) {
            crate::xlog_info!("D3D12.X: memory windows from the shared descriptor heap (no per-access switch)");
        }
    }
    *UMA_BUFFER.lock().unwrap() = bufs;
}

pub fn stats() -> D3d12Stats {
    let classes = CLASSES
        .lock()
        .map(|g| {
            g.iter()
                .map(|c| ClassStat { interface: c.iface.name.to_string(), public_methods: c.public, internal_slots: c.scanned - c.public })
                .collect()
        })
        .unwrap_or_default();
    let (presents, boundaries) = super::present::counters();
    let submission_timing = SUBMISSION_TIMING.lock().map(|timing| *timing).unwrap_or_default();
    let (gpu_completion_receipts_tracked, gpu_completion_completed, gpu_completion_device_lost, gpu_completion_pending, gpu_completion_capacity_evictions) =
        GPU_COMPLETION_TIMING.lock().map(|mut timing| {
            if gpu_timing_on() {
                timing.retire_completed();
            }
            (timing.tracked, timing.completed, timing.device_lost, timing.pending.len(), timing.capacity_evictions)
        }).unwrap_or_default();
    D3d12Stats {
        backend: format!("{:?}", super::backend()).to_lowercase(),
        adapter: ADAPTER.lock().map(|a| a.clone()).unwrap_or_default(),
        classes,
        unknown_iids: UNKNOWN_IIDS.lock().map(|s| s.iter().cloned().collect()).unwrap_or_default(),
        fallback_pipelines: FALLBACK_PSO.load(Ordering::Relaxed),
        failed_pipelines: FAILED_PSO.load(Ordering::Relaxed),
        fallback_root_signatures: FALLBACK_RS.load(Ordering::Relaxed),
        presents,
        frame_boundaries: boundaries,
        last_frame_dump: super::present::last_frame_dump(),
        uma_mb: crate::gpu::uma::get().map(|u| (u.size() >> 20) as u64).unwrap_or(0),
        uma_used_kb: crate::gpu::uma::get().map(|u| (u.used() >> 10) as u64).unwrap_or(0),
        uma_gpu_va: match uma_gpu_va() {
            0 => String::new(),
            v => format!("{v:#x}"),
        },
        uma_error: UMA_ERROR.lock().map(|e| e.clone()).unwrap_or_default(),
        uma_import_check: UMA_IMPORT_CHECK.lock().map(|e| e.clone()).unwrap_or_default(),
        translated_pipelines: TRANSLATED_PSO.load(Ordering::Relaxed),
        translation_errors: TRANSLATE_ERRORS.lock().map(|e| e.clone()).unwrap_or_default(),
        background_pipelines: super::pso_async::counts(),
        pipeline_pressure: super::pso_async::pipeline_pressure_stats(),
        dxil_cache: crate::gpu::recomp::dxc::cache_stats(),
        submissions: submission_timing.submissions,
        slowest_submission_ms: submission_timing.slowest_ns as f64 / 1_000_000.0,
        slowest_submission_lists: submission_timing.slowest_lists,
        submission_total_ns: submission_timing.total_ns,
        submission_mean_ms: if submission_timing.submissions == 0 { 0.0 } else {
            submission_timing.total_ns as f64 / submission_timing.submissions as f64 / 1_000_000.0
        },
        submission_p95_upper_ms: submission_timing.percentile_upper_ns(95) as f64 / 1_000_000.0,
        submission_p99_upper_ms: submission_timing.percentile_upper_ns(99) as f64 / 1_000_000.0,
        submission_histogram_log2_ns: submission_timing.histogram.into(),
        gpu_completion_receipts_tracked,
        gpu_completion_completed,
        gpu_completion_device_lost,
        gpu_completion_pending,
        gpu_completion_capacity_evictions,
        gpu_completion_receipt_failures: GPU_COMPLETION_RECEIPT_FAILURES.load(Ordering::Relaxed),
        gpu_completion_tracker_contentions: GPU_COMPLETION_TRACKER_CONTENTIONS.load(Ordering::Relaxed),
    }
}

// ---------------------------------------------------------------------------------------
// device creation
// ---------------------------------------------------------------------------------------

static DEVICE: AtomicUsize = AtomicUsize::new(0);

pub fn device_raw() -> *mut c_void {
    DEVICE.load(Ordering::Acquire) as *mut c_void
}

/// The process's video memory on the device's adapter: (local, non-local) bytes in use.
pub fn video_memory() -> Option<(u64, u64)> {
    use windows::Win32::Graphics::Dxgi::*;
    let raw = device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?;
    unsafe {
        let luid = dev.GetAdapterLuid();
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(luid).ok()?;
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut non_local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local).ok()?;
        adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut non_local).ok()?;
        Some((local.CurrentUsage, non_local.CurrentUsage))
    }
}

/// The device runs only 32-lane waves: wave64 compute, mesh and geometry programs need the
/// split form (`Options::split_wave64`). Unknown device: assume so.
fn wave32_only(device: *mut c_void) -> bool {
    if device.is_null() {
        return true;
    }
    unsafe {
        let Some(dev) = ID3D12Device::from_raw_borrowed(&device) else { return true };
        crate::gpu::recomp::pso::wave32_only_for_device(dev)
    }
}

/// A wave64 VS / PS may run in 32-lane waves here (`pso::graphics_wave32`: the driver picks
/// the wave size of graphics shaders): the split flag for VS / PS translation. Unknown
/// device: assume so.
fn graphics_wave32(device: *mut c_void) -> bool {
    if device.is_null() {
        return true;
    }
    unsafe {
        let Some(dev) = ID3D12Device::from_raw_borrowed(&device) else { return true };
        crate::gpu::recomp::pso::graphics_wave32_for_device(dev)
    }
}

/// Create the process-wide D3D12 device (the adapter `d3d12_adapter` / `XCL_ADAPTER` chooses,
/// by default the high-performance GPU → WARP); once – later calls
/// return the existing device's adapter.
pub fn init() -> Result<String, String> {
    init_with_remap(false)
}

pub(super) fn init_with_remap(allow_driver_remap: bool) -> Result<String, String> {
    if !device_raw().is_null() {
        return Ok(ADAPTER.lock().map(|a| a.clone()).unwrap_or_default());
    }
    unsafe {
        let cfg = crate::state::config();
        match crate::gpu::agility::state() {
            crate::gpu::agility::State::Loaded { version, dir, .. } => {
                crate::xlog_info!("D3D12: Agility SDK v{version} ({})", dir.display());
            }
            crate::gpu::agility::State::Unavailable(reason) => {
                crate::xlog_warn!("D3D12: Agility SDK unavailable ({reason}) – using inbox D3D12");
            }
            crate::gpu::agility::State::NotFound => {}
        }
        let debug = cfg.d3d12_debug || std::env::var_os("XCL_D3D12_DEBUG").is_some();
        if debug {
            match crate::gpu::agility::enable_debug_layer() {
                Ok(()) => {
                    crate::xlog_info!("D3D12: debug layer enabled");
                }
                Err(e) => {
                    crate::xlog_warn!("D3D12: debug layer requested but unavailable: {e}");
                }
            }
        }
        // `XCL_DRED=1`: on a device removal the faulting address and the last GPU operations
        // (logged by `log_device_removed`); no debug layer needed. Opt-in: with it XDefiant
        // stalled in its first seconds twice.
        if std::env::var_os("XCL_DRED").is_some() {
            match crate::gpu::agility::configuration_interface::<ID3D12DeviceRemovedExtendedDataSettings>(
                &CLSID_D3D12DeviceRemovedExtendedData,
            ) {
                Ok(d) => {
                    d.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    d.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                }
                Err(e) => {
                    crate::xlog_warn!("D3D12: DRED requested but unavailable: {e}");
                }
            }
        }
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| format!("DXGI: {e}"))?;
        let mut device: Option<ID3D12Device> = None;
        let mut name = String::new();
        let mut vendor = 0x1414; // WARP is Microsoft's adapter
        let sel = crate::gpu::adapter::Selector::from_env_or(&cfg.d3d12_adapter);
        if sel != crate::gpu::adapter::Selector::Warp {
            let mut list = crate::gpu::adapter::candidates(&factory, &sel);
            if list.is_empty() {
                crate::xlog_warn!("D3D12: {} – using auto", crate::gpu::adapter::no_match(&factory, &sel));
                list = crate::gpu::adapter::candidates(&factory, &crate::gpu::adapter::Selector::Auto);
            }
            for (adapter, desc) in list {
                if let Ok(dev) = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0) {
                    device = Some(dev);
                    vendor = desc.VendorId;
                    name = format!("{} ({} MiB VRAM)", crate::gpu::adapter::name(&desc), desc.DedicatedVideoMemory >> 20);
                    break;
                }
            }
        }
        let device = match device {
            Some(d) => d,
            None => {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().map_err(|e| format!("WARP: {e}"))?;
                let d = crate::gpu::agility::create_device(&warp, D3D_FEATURE_LEVEL_11_0)
                    .map_err(|e| format!("D3D12 device (WARP): {e}"))?;
                name = "WARP (software renderer)".into();
                d
            }
        };
        driver_cache::note_vendor(vendor);
        ADAPTER_VENDOR.store(vendor, Ordering::Release);
        if debug {
            // the debug layer's messages into XCL's log (they otherwise go to a debugger only)
            if let Ok(q) = device.cast::<ID3D12InfoQueue1>() {
                let mut cookie = 0u32;
                let _ = q.RegisterMessageCallback(Some(debug_message), D3D12_MESSAGE_CALLBACK_FLAG_NONE, std::ptr::null_mut(), &mut cookie);
            }
        }
        open_uma(&device, allow_driver_remap);
        super::device_x::probe_dynamic_depth_bias(&device);
        crate::shaders::pipeline_library::activate(&device);
        let raw = device.into_raw(); // one reference kept for the process lifetime
        adopt(raw, Kind::Device);
        DEVICE.store(raw as usize, Ordering::Release);
        *ADAPTER.lock().unwrap() = name.clone();
        Ok(name)
    }
}

/// A debug layer message: errors and warnings into the log – the first 8 of each message id
/// (3 000 in all). One cap for all of them (200) was used up in seconds by two per-frame state
/// errors of a retail game, long before the message that came with its device removal.
unsafe extern "system" fn debug_message(_category: D3D12_MESSAGE_CATEGORY, severity: D3D12_MESSAGE_SEVERITY, id: D3D12_MESSAGE_ID, text: windows::core::PCSTR, _context: *mut c_void) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static PER_ID: Mutex<Option<HashMap<i32, u32>>> = Mutex::new(None);
    if severity.0 > D3D12_MESSAGE_SEVERITY_WARNING.0 {
        return;
    }
    {
        let mut g = PER_ID.lock().unwrap();
        let n = g.get_or_insert_with(HashMap::new).entry(id.0).or_insert(0);
        *n += 1;
        if *n > 8 {
            return;
        }
    }
    if N.fetch_add(1, Ordering::Relaxed) >= 3000 {
        return;
    }
    let t = if text.is_null() { String::new() } else { String::from_utf8_lossy(text.as_bytes()).into_owned() };
    crate::xlog_warn!("D3D12 debug layer ({}, id {}): {t}", if severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 { "error" } else { "warning" }, id.0);
}

type QiFn = unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT;
type RefFn = unsafe extern "system" fn(*mut c_void) -> u32;

pub unsafe fn create_device(riid: *const c_void, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() {
        return S_FALSE;
    }
    let dev = device_raw();
    if dev.is_null() {
        return E_INVALIDARG;
    }
    if !riid.is_null() {
        let qi: QiFn = std::mem::transmute(*vt(dev));
        let hr = qi(dev, riid as *const GUID, ppv);
        if hr >= 0 {
            crate::xlog_info!("D3D12.X: device {:p} for the title", *ppv);
            if std::env::var_os("XCL_WATCH_DEVICE").is_some() {
                crate::eh::watch_writes(*ppv as usize);
            }
            return hr;
        }
    }
    let add: RefFn = std::mem::transmute(*vt(dev).add(1));
    add(dev);
    *ppv = dev;
    S_OK
}

// ---------------------------------------------------------------------------------------
// vtable adoption
// ---------------------------------------------------------------------------------------

struct Class {
    orig: usize,
    kind: Kind,
    ext: usize,
    iface: &'static Iface,
    public: usize,
    scanned: usize,
}

static CLASSES: Mutex<Vec<Class>> = Mutex::new(Vec::new());
static EXTS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

#[cfg(test)]
pub(super) struct TestCommandListClass {
    ext: usize,
}

#[cfg(test)]
impl Drop for TestCommandListClass {
    fn drop(&mut self) {
        CLASSES.lock().unwrap().retain(|c| c.ext != self.ext);
    }
}

/// Register the synthetic adopted command-list vtable used by unit tests that call helpers whose
/// method slots are normally populated by `adopt`.
#[cfg(test)]
pub(super) fn test_command_list_class(orig: usize, ext: usize) -> TestCommandListClass {
    let iface = layout::find("ID3D12GraphicsCommandList").expect("command-list layout");
    CLASSES.lock().unwrap().push(Class {
        orig,
        kind: Kind::CommandList,
        ext,
        iface,
        public: iface.methods.len(),
        scanned: iface.methods.len(),
    });
    TestCommandListClass { ext }
}

unsafe fn vt(obj: *mut c_void) -> *const usize {
    *(obj as *const *const usize)
}

/// Original function of `slot` for an adopted object.
pub unsafe fn orig_fn(this: *mut c_void, slot: usize) -> usize {
    let ext = vt(this);
    let orig = *ext.sub(1) as *const usize;
    *orig.add(slot)
}

/// `slot_of(this, method)`, looked up once for this call site. COM interface versions append
/// methods, so one method keeps its slot throughout the interface family served by a hook.
macro_rules! slot {
    ($this:expr, $method:literal) => {{
        static SLOT: AtomicUsize = AtomicUsize::new(0);
        cached_slot(&SLOT, $this, $method)
    }};
}

unsafe fn cached_slot(cache: &AtomicUsize, this: *mut c_void, method: &str) -> usize {
    match cache.load(Ordering::Relaxed) {
        0 => {
            let slot = slot_of(this, method);
            cache.store(slot, Ordering::Relaxed);
            slot
        }
        slot => slot,
    }
}

unsafe fn native_queue_fn(queue: *mut c_void, slot: usize) -> usize {
    let table = vt(queue);
    if is_ext(table as usize) { orig_fn(queue, slot) } else { *table.add(slot) }
}

/// Submit XCL's own list through the PC queue slot, never through the title-facing wrapper.
/// The queue lease makes this one ordered operation, and is re-entrant when a title submit is
/// already publishing its effects.
pub(crate) unsafe fn native_queue_execute(queue: &ID3D12CommandQueue, list: &ID3D12CommandList) {
    let raw = queue.as_raw();
    type F = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let f = native_queue_fn(raw, 10);
    super::queue_submit::with(raw as usize, || {
        let f: F = std::mem::transmute(f);
        let list = list.as_raw();
        f(raw, 1, &list);
    });
}

/// Signal through the PC queue slot. Console queue layouts remap public slots 14+, so an
/// internal COM call on an adopted queue is not a reliable way to reach native Signal.
pub(crate) unsafe fn native_queue_signal(queue: &ID3D12CommandQueue, fence: &ID3D12Fence, value: u64) -> windows::core::Result<()> {
    let raw = queue.as_raw();
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT;
    let f = native_queue_fn(raw, 14);
    super::queue_submit::with(raw as usize, || {
        let f: F = std::mem::transmute(f);
        windows::core::HRESULT(f(raw, fence.as_raw(), value)).ok()
    })
}

/// Wait through the PC queue slot, bypassing the title's shifted queue layout.
pub(crate) unsafe fn native_queue_wait(queue: &ID3D12CommandQueue, fence: &ID3D12Fence, value: u64) -> windows::core::Result<()> {
    let raw = queue.as_raw();
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT;
    let f = native_queue_fn(raw, 15);
    super::queue_submit::with(raw as usize, || {
        let f: F = std::mem::transmute(f);
        windows::core::HRESULT(f(raw, fence.as_raw(), value)).ok()
    })
}

pub(crate) unsafe fn native_fence_completed(fence: &ID3D12Fence) -> u64 {
    let raw = fence.as_raw();
    let slot = std::mem::offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / std::mem::size_of::<usize>();
    type F = unsafe extern "system" fn(*mut c_void) -> u64;
    let f: F = std::mem::transmute(if is_ext(vt(raw) as usize) { orig_fn(raw, slot) } else { *vt(raw).add(slot) });
    f(raw)
}

#[cfg(test)]
pub(super) fn register_test_queue_vtable(ext: usize, add: bool) {
    let mut tables = EXTS.lock().unwrap();
    let tables = tables.get_or_insert_with(HashSet::new);
    if add { tables.insert(ext); } else { tables.remove(&ext); }
}

fn module_of(addr: usize) -> usize {
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };
    let mut h = std::ptr::null_mut();
    unsafe {
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, addr as *const u16, &mut h);
    }
    h as usize
}

/// Whether `obj` is an object XCL adopted (its vtable is one of XCL's).
pub(super) unsafe fn is_adopted(obj: *mut c_void) -> bool {
    !obj.is_null() && crate::gpu::shader::readable(obj as usize, 8) && is_ext(*(obj as *const usize))
}

/// A struct-returning method's first two arguments: true when `a` is the object (the C++ way:
/// this, return buffer, …), false when `b` is (the C way: return buffer, this, …). A return
/// buffer on the stack can hold a stale copy of an object's vtable pointer and look adopted
/// itself: a retail game's call of GetResourceAllocationInfo the C way then had its result
/// written over the device's vtable pointer (the next call read slot 25 of "vtable" 0x1000). So
/// when both look adopted, the one on this thread's stack – or the one that is not the device –
/// is the buffer.
pub(super) unsafe fn this_first(a: *mut c_void, b: *mut c_void) -> bool {
    match (is_adopted(a), is_adopted(b)) {
        (true, false) => true,
        (false, true) => false,
        (true, true) => {
            let (mut lo, mut hi) = (0usize, 0usize);
            windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut lo, &mut hi);
            let on_stack = |p: *mut c_void| (lo..hi).contains(&(p as usize));
            let dev = device_raw();
            if on_stack(a) != on_stack(b) {
                on_stack(b)
            } else if a == dev || b == dev {
                a == dev
            } else {
                true
            }
        }
        (false, false) => false,
    }
}

fn is_ext(v: usize) -> bool {
    EXTS.lock().map(|g| g.as_ref().map(|s| s.contains(&v)).unwrap_or(false)).unwrap_or(false)
}

fn kind_of(ext: usize) -> Option<Kind> {
    CLASSES.lock().unwrap().iter().find(|c| c.ext == ext).map(|c| c.kind)
}

/// If a native D3D12 method (such as `Close` or `Reset` on Agility SDK 1.620+, or copy-list
/// `Reset`) replaced `this`'s vtable pointer with another native runtime table, adopt the
/// new table under the same `Kind` so XCL's hooks stay installed while forwarding to the
/// runtime's current state table.
pub(super) unsafe fn preserve_vtable(this: *mut c_void, prev_ext: usize) {
    let cur = *(this as *const usize);
    if cur == prev_ext || is_ext(cur) {
        return;
    }
    if let Some(kind) = kind_of(prev_ext) {
        adopt(this, kind);
    } else {
        *(this as *mut usize) = prev_ext;
    }
}

/// An adopted object gets the extended vtable of another kind (a DMA list: created as a PC
/// list, it needs the console's DMA layout).
pub(super) unsafe fn readopt(obj: *mut c_void, kind: Kind) {
    if obj.is_null() || !is_ext(vt(obj) as usize) {
        return adopt(obj, kind);
    }
    let orig = *vt(obj).sub(1);
    *(obj as *mut usize) = class_for(obj, orig, kind);
}

/// Give a genuine D3D12 object the extended vtable for `kind`.
pub unsafe fn adopt(obj: *mut c_void, kind: Kind) {
    if obj.is_null() {
        return;
    }
    let v = vt(obj) as usize;
    if is_ext(v) {
        return;
    }
    let ext = class_for(obj, v, kind);
    *(obj as *mut usize) = ext;
}

unsafe fn class_for(obj: *mut c_void, orig: usize, kind: Kind) -> usize {
    if let Some(c) = CLASSES.lock().unwrap().iter().find(|c| c.orig == orig && c.kind == kind) {
        return c.ext;
    }
    let ot = orig as *const usize;
    // most derived PC interface this object implements
    let chain = layout::chain(kind.base());
    let qi: QiFn = std::mem::transmute(*ot);
    let release: RefFn = std::mem::transmute(*ot.add(2));
    let mut iface = *chain.last().expect("layout chain");
    for i in &chain {
        let iid = layout::parse_guid(i.iid);
        let mut out = std::ptr::null_mut();
        if qi(obj, &iid, &mut out) >= 0 && !out.is_null() {
            release(out);
            iface = i;
            break;
        }
    }
    let public = iface.methods.len();
    let m0 = module_of(*ot);
    let mut scanned = 0;
    while scanned < public + MAX_SCAN && module_of(*ot.add(scanned)) == m0 {
        scanned += 1;
    }
    let scanned = scanned.max(public);
    let max_console = console_slots(kind)
        .iter()
        .map(|e| e.0 + 1)
        .chain(console_remap(kind).iter().map(|e| e.0 + 1))
        .max()
        .unwrap_or(0);
    let total = (scanned + EXTRA_SLOTS).max(max_console);
    let mut v: Vec<usize> = Vec::with_capacity(total + 1);
    v.push(orig);
    for i in 0..scanned {
        v.push(*ot.add(i));
    }
    let hook = if kind == Kind::CommandQueue { super::present::on_queue_console_method as usize } else { 0 };
    for i in public..scanned {
        let info = stubs::new_info(DLL, &format!("{}::#{i} (not on PC)", iface.name), 0);
        info.hook.store(hook, Ordering::Release);
        v[1 + i] = super::guard::make_guard(v[1 + i], info);
    }
    for i in scanned..total {
        let (addr, info) = stubs::make_stub_info(DLL, &format!("{}::#{i} (not on PC)", iface.name), 0);
        info.hook.store(hook, Ordering::Release);
        v.push(addr);
    }
    for (method, f) in overrides(kind) {
        if let Some(s) = layout::slot(iface, method) {
            if s < public {
                v[1 + s] = f;
            }
        }
    }
    // the console's layout of the same PC methods (title calls only; the runtime keeps PC's)
    let pc = v.clone();
    for (x, p) in console_remap(kind) {
        if x < public {
            v[1 + x] = super::guard::make_split(pc[1 + p], pc[1 + x]);
        } else if x < scanned {
            v[1 + x] = super::guard::make_split(pc[1 + p], *ot.add(x));
        } else if x < total {
            v[1 + x] = pc[1 + p];
        }
    }
    // slots where the console's method differs from the PC one at the same index
    for (slot, f) in console_slots(kind) {
        if slot < public {
            v[1 + slot] = super::guard::make_split(f, v[1 + slot]);
        } else if slot < scanned {
            // an internal method of the runtime there (e.g. queue #30 = PresentX): the title
            // gets the console method, the runtime keeps its own
            v[1 + slot] = super::guard::make_split(f, *ot.add(slot));
        } else if slot < total {
            v[1 + slot] = f;
        }
    }
    // the other console extensions (device, command list): a title call is logged and answered
    // S_OK, not run as the PC method at that index
    if let Some(first) = console_extensions(kind) {
        let own: HashSet<usize> = console_slots(kind).iter().map(|e| e.0).chain(console_remap(kind).iter().map(|e| e.0)).collect();
        for i in (first..public).filter(|i| !own.contains(i)) {
            let (stub, _) = stubs::make_stub_info(DLL, &format!("{}::#{i} (console extension)", iface.name), 0);
            v[1 + i] = super::guard::make_split(stub, v[1 + i]);
        }
    }
    // methods the newer console layout keeps at other slots: picked at each call
    if kind == Kind::CommandList {
        let older_handlers = console_slots(kind);
        for (newer, older) in NEWER_LIST_SLOTS {
            let Some(&(_, f)) = older_handlers.iter().find(|e| e.0 == older) else { continue };
            if newer >= total {
                continue;
            }
            let pc = if newer < scanned { *ot.add(newer) } else { v[1 + newer] };
            v[1 + newer] = super::guard::make_layout_split(super::guard::LayoutTargets { older: v[1 + newer], newer: f, pc, newer_layout: &QUEUE_SHIFTED });
        }
    }
    // XCL_D3D12_TRACE: every public method called from console code is logged
    if std::env::var_os("XCL_D3D12_TRACE").is_some() {
        for i in 0..public {
            let name = format!("{}::{} (#{i})", iface.name, iface.methods.get(i).copied().unwrap_or("?"));
            v[1 + i] = super::guard::make_tracer(v[1 + i], name);
        }
    }
    let leaked: &'static [usize] = Box::leak(v.into_boxed_slice());
    let ext = leaked.as_ptr() as usize + 8;
    EXTS.lock().unwrap().get_or_insert_with(HashSet::new).insert(ext);
    crate::xlog_info!(
        "D3D12.X: {} – {public} PC methods, {} internal slots under the guard, +{EXTRA_SLOTS} stubs ({kind:?}, PC vtable {})",
        iface.name,
        scanned - public,
        crate::modules::describe_address(orig)
    );
    CLASSES.lock().unwrap().push(Class { orig, kind, ext, iface, public, scanned });
    ext
}

/// (console slot, PC slot) of PC methods that sit elsewhere in the console's vtable.
/// ID3D12GraphicsCommandList (from the Fortnite RHI call sites, matching XCL's own evidence
/// – FSR4 calls ClearUnorderedAccessViewUint at #53): 0–27 and 56–59 as on PC; the query
/// methods come right after ExecuteBundle (28–31 = PC 52–55: BeginQuery, EndQuery,
/// ResolveQueryData, SetPredication), so SetDescriptorHeaps … DiscardResource move up by 4
/// (32–55 = PC 28–51); ID3D12GraphicsCommandList1 is at 104–109 (PC 60–65), the shading-rate
/// methods at 124 / 125 (PC 77 / 78). The console-only methods in between: `list_x`.
fn console_remap(kind: Kind) -> Vec<(usize, usize)> {
    match kind {
        Kind::CommandList => (28..=31).map(|x| (x, x + 24)).chain((32..=55).map(|x| (x, x - 4))).chain((104..=109).map(|x| (x, x - 44))).chain([(124, 77), (125, 78)]).collect(),
        // the DMA list: Close / Reset, then the copy methods (XDefiant calls #12 with
        // CopyTextureRegion's arguments): 11–14 = PC 15–18 (CopyBufferRegion, CopyTextureRegion,
        // CopyResource, CopyTiles); what follows is not known yet (`console_extensions`)
        Kind::DmaList => (11..=14).map(|x| (x, x + 4)).collect(),
        _ => Vec::new(),
    }
}

/// console D3D12_DESCRIPTOR_HEAP_TYPE also used for CBV / SRV / UAV heaps
const XBOX_EXTRA_CBV_SRV_UAV: u32 = 4;

/// Map a console `D3D12_DESCRIPTOR_HEAP_TYPE` to its PC equivalent.
#[inline]
fn pc_heap_type(ty: u32) -> u32 {
    if ty == XBOX_EXTRA_CBV_SRV_UAV {
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32
    } else {
        ty
    }
}

/// Where a class's console extensions start: the device's from #44 (`device_x`), the command
/// list's from #60 (`list_x`).
fn console_extensions(kind: Kind) -> Option<usize> {
    match kind {
        Kind::Device => Some(44),
        Kind::CommandList => Some(60),
        Kind::DmaList => Some(15),
        _ => None,
    }
}

/// Console vtable slots that differ from the PC layout (found on console code calling them):
/// * device #62: `CreateComputePipelineState` with two extra arguments (XboxBvhBuilder.dll
///   passes a D3D12_COMPUTE_PIPELINE_STATE_DESC, NULL, NULL, riid, ppv); PC #62 = CreateStateObject.
/// * device #76: `CreateCommittedResource` (heap properties, flags, desc, state, clear value,
///   riid, ppv) as on PC #27 (PC #76 = CreateCommittedResource3); the console resource desc
///   carries more after the 52 PC bytes (amd_fidelityfx_upscaler_xbox.dll writes 3 at +52).
/// * descriptor heap #9 / #10: the heap start handles come back in rax on the console.
/// * resource #11 GetGPUVirtualAddress: console buffers live in unified memory; their console
///   address (= CPU address, low 32 bits = offset in `mem`) is what console code gets.
/// * device #14 / #15 / #17 / #18 / #19 / #22 / #23 / #24 (CreateDescriptorHeap,
///   GetDescriptorHandleIncrementSize, CreateConstantBufferView, CreateShaderResourceView,
///   CreateUnorderedAccessView, CreateSampler, CopyDescriptors, CopyDescriptorsSimple):
///   CBV/SRV/UAV and sampler heaps of console code get 32 / 16-byte slots in unified memory
///   (`console_heap`). #14–#19 seen from console code with the PC layout; #17 and #22–#24
///   assumed to follow it (each checks its arguments and otherwise goes to the PC method).
/// * device #12 `CreateCommandList`: console list types; with console lists on, console code
///   gets a console-shaped list (D3D12.X inline code writes into it), XCL's own code a PC one.
/// * the device's own methods (`device_x`, `frame`), the command list's (`list_x`), queue #30
///   PresentX (`present`).
fn console_slots(kind: Kind) -> Vec<(usize, usize)> {
    use super::device_x as x;
    match kind {
        Kind::Device => {
            vec![
                (0, dev_qi_x as usize),
                (8, x::create_command_queue_x as usize),
                (25, x::get_resource_allocation_info_x as usize),
                (26, x::get_custom_heap_properties_x as usize),
                (29, dev_create_placed_resource_x as usize),
                // CreateCommittedResource at its PC index too (XDefiant): console code uses the
                // buffer's GPU address as a CPU pointer – it goes into unified memory
                (27, dev_create_committed_resource_x as usize),
                (9, x::create_command_allocator_x as usize),
                (10, x::create_graphics_pso_pc as usize),
                (12, x::create_command_list_pc as usize),
                (14, dev_create_descriptor_heap_x as usize),
                (15, dev_increment_size_x as usize),
                (17, dev_create_cbv_x as usize),
                (18, dev_create_srv_x as usize),
                (21, dev_create_dsv_x as usize),
                (19, dev_create_uav_checked_x as usize),
                (22, dev_create_sampler_x as usize),
                (23, dev_copy_descriptors_x as usize),
                (24, dev_copy_descriptors_simple_x as usize),
                (38, x::get_copyable_footprints_x as usize),
                (41, x::create_command_signature_x as usize),
                (44, x::create_placed_resource_x as usize),
                (48, x::create_command_queue_x as usize),
                (52, x::set_debug_flags_x as usize),
                (55, x::create_derived_pso_x as usize),
                (56, x::serialize_pso_x as usize),
                (57, x::deserialize_pso_x as usize),
                (60, x::create_graphics_pso_x as usize),
                (61, x::create_sampler_x as usize),
                (62, dev_create_compute_pso_x as usize),
                (66, x::serialize_pso_x as usize),
                (67, x::deserialize_pso_x as usize),
                (70, dev_create_srv_at_x as usize),
                (71, dev_create_uav_at_x as usize),
                (73, x::create_command_list_x as usize),
                (74, x::register_page_pool_x as usize),
                (75, x::unregister_page_pool_x as usize),
                (76, dev_create_committed_resource_x as usize),
                // frame pacing (SetFrameIntervalX, ScheduleFrameEventX, WaitFrameEventX,
                // GetFrameStatisticsX)
                (78, super::frame::set_frame_interval_x as usize),
                (79, super::frame::schedule_frame_event_x as usize),
                (80, super::frame::wait_frame_event_x as usize),
                (81, super::frame::frame_statistics_x as usize),
                (86, x::create_pipeline_state_x as usize),
                (108, x::get_resource_allocation_info2_x as usize),
                (127, dev_create_committed_resource_layout_x as usize),
                (133, x::create_placed_resource_layout_x as usize),
            ]
        }
        Kind::DescriptorHeap => vec![(9, heap_cpu_start_x as usize), (10, heap_gpu_start_x as usize)],
        Kind::Resource => vec![(11, res_gpu_va_x as usize)],
        Kind::Fence => vec![(12, super::fence_x::completed_value_address as usize)],
        // the title's calls from #14 on: Signal / Wait / … / SignalX / PresentX at the slots of
        // its runtime's layout (`queue_dispatch`)
        Kind::CommandQueue => QUEUE_SLOTS.iter().enumerate().map(|(k, &f)| (QUEUE_FIRST + k, f as usize)).collect(),
        Kind::DmaList => {
            use super::list_x as l;
            vec![(9, l::dma_close_x as usize), (10, l::reset_x as usize), (11, l::copy_buffer_region_x as usize), (12, l::copy_texture_region_x as usize), (13, l::copy_resource_x as usize)]
        }
        Kind::CommandList => {
            use super::bind as b;
            use super::list_x as l;
            vec![
                // compute root arguments of console code: kept for translated dispatches
                (12, b::draw_instanced as usize),
                (15, l::copy_buffer_region_x as usize),
                (16, l::copy_texture_region_x as usize),
                (17, l::copy_resource_x as usize),
                (19, l::resolve_subresource_x as usize),
                (51, l::clear_depth_stencil_x as usize),
                (53, l::clear_uav_uint_x as usize),
                (54, l::clear_uav_float_x as usize),
                (108, l::resolve_subresource_region_x as usize),
                (13, b::draw_indexed_instanced as usize),
                (14, b::dispatch as usize),
                (48, l::ia_set_vertex_buffers_x as usize),
                // state a PC Reset clears, kept for KickoffX
                (20, l::ia_set_topology_x as usize),
                (21, l::rs_set_viewports_x as usize),
                (22, l::rs_set_scissors_x as usize),
                (23, l::om_set_blend_factor_x as usize),
                (24, l::om_set_stencil_ref_x as usize),
                (32, l::set_descriptor_heaps_x as usize),
                (59, b::execute_indirect as usize),
                (83, l::copy_texture_region_flags_x as usize),
                (34, b::set_graphics_root_signature as usize),
                (36, b::set_graphics_root_table as usize),
                (38, b::set_graphics_root_constant as usize),
                (40, b::set_graphics_root_constants as usize),
                (42, b::set_graphics_root_descriptor as usize),
                (44, b::set_graphics_root_descriptor as usize),
                (46, b::set_graphics_root_descriptor as usize),
                (26, l::resource_barrier_x as usize),
                (27, l::execute_bundle as usize),
                (25, b::set_pipeline_state as usize),
                (33, b::set_compute_root_signature as usize),
                (35, b::set_compute_root_table as usize),
                (37, b::set_compute_root_constant as usize),
                (39, b::set_compute_root_constants as usize),
                (41, b::set_compute_root_descriptor as usize),
                (43, b::set_compute_root_descriptor as usize),
                (45, b::set_compute_root_descriptor as usize),
                (10, l::reset_x as usize),
                (28, l::begin_query_x as usize),
                (29, l::end_query_x as usize),
                (30, l::resolve_query_data_x as usize),
                (47, l::ia_set_index_buffer_x as usize),
                (60, l::copy_memory_x as usize),
                (61, l::fill32_x as usize),
                (62, l::fill64_x as usize),
                (63, l::fill128_x as usize),
                (68, l::set_depth_bounds_x as usize),
                (69, l::write32_bottom_x as usize),
                (70, l::write64_bottom_x as usize),
                (71, l::wait32_x as usize),
                (72, l::wait64_x as usize),
                (78, l::dispatch_x as usize),
                (81, l::draw_indexed_x as usize),
                (84, l::copy_resource_flags_x as usize),
                (85, l::set_predication_buffer_x as usize),
                (88, l::write32_top_x as usize),
                (89, l::write64_top_x as usize),
                (90, l::write32_bottom_of_shader_x as usize),
                (93, l::kickoff_x as usize),
                (94, l::set_depth_bias_x as usize),
                // console compute/BVH builder extensions (`XboxBvhBuilder.dll`)
                (146, l::uav_barrier_x as usize),
                (148, b::allocate_upload_memory as usize),
            ]
        }
        _ => vec![],
    }
}

/// descriptor heap #9 / #10 GetCPU / GetGPUDescriptorHandleForHeapStart: console code
/// takes the handle in rax (only `this` passed); the PC methods return it through a hidden
/// pointer
unsafe extern "system" fn heap_cpu_start_x(this: *mut c_void) -> u64 {
    if let Some(p) = super::console_heap::start(this as usize) {
        return p;
    }
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64 = std::mem::transmute(orig_fn(this, 9));
    let mut h = 0u64;
    f(this, &mut h);
    h
}

unsafe extern "system" fn heap_gpu_start_x(this: *mut c_void) -> u64 {
    if let Some(p) = super::console_heap::start(this as usize) {
        return p;
    }
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64 = std::mem::transmute(orig_fn(this, 10));
    let mut h = 0u64;
    f(this, &mut h);
    h
}

unsafe extern "system" fn dev_create_descriptor_heap_x(this: *mut c_void, desc: *const [u32; 4], riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let mut d = if desc.is_null() { [u32::MAX; 4] } else { *desc };
    d[0] = pc_heap_type(d[0]);
    let desc = if desc.is_null() { desc } else { &d as *const [u32; 4] };
    let sampler = d[0] == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32;
    if d[0] != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32 && !sampler {
        return dev_create_descriptor_heap(this, desc as usize, riid, ppv);
    }
    // views: an SRV / UAV pair per console slot (in the shared shader-visible heap, the
    // title's own a CPU-only mirror – `console_heap`); samplers: one per slot
    let pc_n = if sampler { d[1] } else { d[1].saturating_mul(2) };
    let visible = d[2] & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32 != 0;
    let shared = match ID3D12Device::from_raw_borrowed(&this) {
        Some(dev) if !sampler && !desc.is_null() => super::console_heap::shared_heap(dev),
        Some(dev) if sampler && !desc.is_null() => super::console_heap::shared_sampler_heap(dev),
        _ => false,
    };
    let flags = if shared { d[2] & !(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32) } else { d[2] };
    let pc = [d[0], pc_n, flags, d[3]];
    let hr = dev_create_descriptor_heap(this, pc.as_ptr() as usize, riid, ppv);
    if hr >= 0 {
        let raw = *ppv;
        let heap = ID3D12DescriptorHeap::from_raw_borrowed(&raw).unwrap();
        let dev = ID3D12Device::from_raw_borrowed(&this).unwrap();
        let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        let gpu = if flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32 != 0 { heap.GetGPUDescriptorHandleForHeapStart().ptr } else { 0 };
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE(d[0] as i32));
        if sampler {
            // an S# of zeros – a slot nothing was written to – is a sampler on the console: point
            // filtering, wrap, mip 0 only. Every PC slot starts as that one, so a table entry the
            // title left empty samples as there (an unwritten PC descriptor is undefined).
            let zero = D3D12_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: 0.0,
            };
            for k in 0..d[1] as u64 {
                dev.CreateSampler(&zero, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + k * inc as u64) as usize });
            }
        }
        if !super::console_heap::register(raw as usize, d[1], cpu, gpu, inc, sampler, shared, visible) {
            crate::xlog_warn!("console descriptor heap: no room in UMA for {} slots", d[1]);
        }
    }
    hr
}

unsafe extern "system" fn dev_increment_size_x(this: *mut c_void, ty: u32) -> u32 {
    let ty = pc_heap_type(ty);
    if ty == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32 {
        return super::console_heap::SLOT;
    }
    if ty == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32 {
        return super::console_heap::SAMPLER_SLOT;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32) -> u32 = std::mem::transmute(orig_fn(this, 15));
    f(this, ty)
}

fn view_dim_to_image_type(dim: i32) -> Option<crate::gpu::desc::ImageType> {
    use crate::gpu::desc::ImageType as T;
    match dim {
        2 => Some(T::D1),
        3 => Some(T::D1Array),
        4 => Some(T::D2),
        5 => Some(T::D2Array),
        6 => Some(T::D2Msaa),
        7 => Some(T::D2MsaaArray),
        8 => Some(T::D3),
        9 | 10 => Some(T::Cube),
        _ => None,
    }
}

/// What a console view slot describes: (GFX10 format, image type, width, height, depth, mips)
/// from the view (or the resource when the view is NULL); None for buffers.
unsafe fn view_shape(res: &ID3D12Resource, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, dim: i32, array: u32, mips: u32) -> Option<(u32, crate::gpu::desc::ImageType, u32, u32, u32, u32)> {
    let r = res.GetDesc();
    let (fmt, ty, w, h, depth, _, last_level, _) = view_shape_desc(&r, format, dim, array, 0, mips)?;
    Some((fmt, ty, w, h, depth, last_level + 1))
}

/// Pure-desc shape and mip/sample-level breakdown for a console texture view:
/// `(gfx10_format, image_type, width, height, depth, base_level, last_level, max_mip)`.
/// For MSAA views, `base_level = 0`, `last_level = log2(SampleDesc.Count)` (GFX10 T# ABI),
/// and `max_mip = 0`.
fn view_shape_desc(
    r: &D3D12_RESOURCE_DESC,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    dim: i32,
    array: u32,
    base_mip: u32,
    mips: u32,
) -> Option<(u32, crate::gpu::desc::ImageType, u32, u32, u32, u32, u32, u32)> {
    use crate::gpu::desc::ImageType as T;
    let f = if format == windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN { r.Format } else { format };
    let fmt = crate::gpu::desc::gfx10_format(f).unwrap_or(0);
    let (w, h, d) = (r.Width as u32, r.Height, r.DepthOrArraySize as u32);
    let ty = view_dim_to_image_type(dim)?;
    let depth = match ty {
        T::D3 => d,
        T::D1Array | T::D2Array | T::D2MsaaArray => array.max(1),
        // a cube view: its faces (6 per cube)
        T::Cube => array.max(6),
        _ => 1,
    };
    let (base_level, last_level, max_mip) = if matches!(ty, T::D2Msaa | T::D2MsaaArray) {
        (0, r.SampleDesc.Count.max(1).trailing_zeros().min(4), 0)
    } else {
        let res_max_mip = (r.MipLevels as u32).max(1) - 1;
        let base = base_mip.min(res_max_mip).min(15);
        let last = if mips == 0 || mips == u32::MAX {
            res_max_mip
        } else {
            base.saturating_add(mips.saturating_sub(1)).min(res_max_mip)
        }
        .min(15);
        (base, last, res_max_mip.min(15))
    };
    Some((fmt, ty, w, h, depth, base_level, last_level, max_mip))
}

/// Build the XCL shader-ABI T# for an SRV, populating GFX10 `BASE_LEVEL` (`t[3][15:12]`),
/// `LAST_LEVEL` (`t[3][19:16]`, or `log2(SampleDesc.Count)` for MSAA), and `MAX_MIP` (`t[5][7:4]`).
fn srv_tsharp_from_desc(slot_index: u32, r: &D3D12_RESOURCE_DESC, desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> Option<[u32; 8]> {
    let (format, dim, array, base_mip, mips) = match desc {
        Some(v) => {
            // SAFETY: each branch reads the union member selected by `v.ViewDimension`.
            let (array, base_mip, mips) = unsafe {
                let a = &v.Anonymous;
                match v.ViewDimension.0 {
                    2 => (1, a.Texture1D.MostDetailedMip, a.Texture1D.MipLevels),
                    3 => (a.Texture1DArray.ArraySize, a.Texture1DArray.MostDetailedMip, a.Texture1DArray.MipLevels),
                    4 => (1, a.Texture2D.MostDetailedMip, a.Texture2D.MipLevels),
                    5 => (a.Texture2DArray.ArraySize, a.Texture2DArray.MostDetailedMip, a.Texture2DArray.MipLevels),
                    6 => (1, 0, 1),
                    7 => (a.Texture2DMSArray.ArraySize, 0, 1),
                    8 => (1, a.Texture3D.MostDetailedMip, a.Texture3D.MipLevels),
                    9 => (6, a.TextureCube.MostDetailedMip, a.TextureCube.MipLevels),
                    10 => (a.TextureCubeArray.NumCubes.max(1) * 6, a.TextureCubeArray.MostDetailedMip, a.TextureCubeArray.MipLevels),
                    _ => (1, 0, 1),
                }
            };
            (v.Format, v.ViewDimension.0, array, base_mip, mips)
        }
        None => (r.Format, resource_dim_to_view(r), r.DepthOrArraySize as u32, 0, r.MipLevels as u32),
    };
    let (fmt, ty, w, h, d, base_level, last_level, max_mip) = view_shape_desc(r, format, dim, array, base_mip, mips)?;
    let mut t = crate::gpu::desc::xcl_image_for_view(slot_index, fmt, ty, w, h, d, 1);
    t[3] = (t[3] & !0x000F_F000) | ((base_level & 0xF) << 12) | ((last_level & 0xF) << 16);
    t[5] = (t[5] & !0x0000_00F0) | ((max_mip & 0xF) << 4);
    Some(t)
}

fn resource_dim_to_view(r: &D3D12_RESOURCE_DESC) -> i32 {
    match r.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            if r.DepthOrArraySize > 1 {
                3
            } else {
                2
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => match (r.SampleDesc.Count > 1, r.DepthOrArraySize > 1) {
            (true, true) => 7,
            (true, false) => 6,
            (false, true) => 5,
            (false, false) => 4,
        },
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => 8,
        _ => 1,
    }
}

static NON_UMA_BUFFER_VIEWS: AtomicU64 = AtomicU64::new(0);

/// V# of a buffer view (raw: 4-byte elements, num_records in bytes; structured / typed:
/// stride = element size, num_records in elements); empty when the buffer is not in UMA.
fn buffer_view(res: usize, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, first: u64, n: u32, stride: u32, raw: bool) -> [u32; 4] {
    let Some((at, bytes)) = console_buffer(res) else {
        note_buffer_view();
        return [0; 4];
    };
    let fmt = crate::gpu::desc::gfx10_format(format).unwrap_or(22);
    let (stride, elem) = if raw {
        (0, 4)
    } else if stride > 0 {
        (stride, stride)
    } else {
        let e = crate::gpu::desc::format_bytes(fmt);
        (e, e)
    };
    let base = at as u64 + first * elem as u64;
    let room = (bytes as u64).saturating_sub(first * elem as u64);
    let records = if stride == 0 { (n as u64 * 4).min(room) } else { (n as u64).min(room / stride as u64) };
    crate::gpu::desc::buffer_words(base, stride, records as u32, fmt)
}

/// Console buffer → a placed resource in the unified memory heap, so its console address is
/// what the recompiled shaders read through `mem`. None: not placed (fall back to committed).
/// A committed texture of console code: on the console its memory is unified memory too –
/// titles write it through its GPU address (XDefiant clears one with memset). It gets memory
/// of the console's size (XG; the PC's where XG has no model) and is created as a texture at
/// that address (`device_x::create_placed_resource_x`: kept in step with the memory where XG
/// lays it out); GetGPUVirtualAddress returns the address. The memory is the coherent
/// window's, as long as it keeps `COHERENT_KEPT` free; beyond that the texture is a plain PC
/// one (None). Unicorn Overlord's 2.2 GB of committed textures filled the 512 MB window in its
/// first second, leaving no room for readback heaps. (In the sparse windows instead, XDefiant
/// rendered black – not understood yet.)
/// bytes of the coherent window committed textures leave free: readback / write-back heaps
/// and buffers, XCL's blocks outside its arena pool
const COHERENT_KEPT: usize = 128 << 20;

unsafe fn place_console_texture(this: *mut c_void, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let u = crate::gpu::uma::get()?;
    if ppv.is_null() || !crate::gpu::shader::readable(desc, 0x38) {
        return None;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    if !(2..=4).contains(&u32::from_le_bytes(x[0..4].try_into().unwrap())) {
        return None;
    }
    let (size, align) = match crate::hle::xg::Texture::new(&x) {
        Ok(t) if t.size() > 0 => (t.size(), t.alignment()),
        _ => {
            type F = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
            let pc: F = std::mem::transmute(orig_fn(this, 25));
            let mut r = [0u64; 2];
            pc(this, &mut r, 0, 1, &super::device_x::pc_resource_desc_for_device(this, &x));
            (r[0], r[1])
        }
    };
    if size == 0 || size == u64::MAX {
        return None;
    }
    let align = (align as usize).max(0x1_0000);
    let bytes = (size as usize).div_ceil(align) * align;
    if u.size().saturating_sub(u.used()) < bytes + COHERENT_KEPT {
        return None;
    }
    let p = u.alloc(bytes, align)? as usize;
    let hr = super::device_x::create_placed_resource_x(this, p as u64, desc as *const u8, state as u32, clear as *const c_void, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        u.free(p);
        return None;
    }
    UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (p, size as usize, true));
    Some(hr)
}

/// Largest single `D3D12_HEAP_TYPE_READBACK` allocation kept on the coherent window; larger
/// ones spill to sparse VRAM so one big readback heap/buffer cannot exhaust coherent headroom.
const COHERENT_READBACK_MAX: usize = 64 << 20;
/// Largest single `D3D12_HEAP_TYPE_CUSTOM` + `WRITE_BACK` `COPY_DEST` buffer kept on coherent RAM.
const COHERENT_CUSTOM_WB_MAX: usize = 16 << 20;

/// bytes of the coherent window readback heaps and buffers leave free (XCL's blocks outside its
/// arena pool)
const COHERENT_FLOOR: usize = 16 << 20;

/// Whether a readback heap / buffer of `rounded` bytes may take coherent memory. The room
/// committed textures leave (`COHERENT_KEPT`) is kept for these: with the textures at that
/// limit (Unicorn Overlord's), a second 128 MB of headroom sent every readback buffer to the
/// sparse windows.
fn has_coherent_headroom(u: &crate::gpu::uma::Uma, rounded: usize) -> bool {
    u.size().saturating_sub(u.used()) >= COHERENT_FLOOR.min(u.size() / 8).saturating_add(rounded)
}

/// Sparse-backed buffers whose `Map(0, NULL, ...)` should read back their full range from VRAM
/// (readback buffers that spilled to sparse VRAM or `COPY_DEST` buffers in `CUSTOM + WRITE_BACK`).
static SPARSE_READBACK: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

fn note_sparse_readback(res: usize) {
    SPARSE_READBACK.lock().unwrap().get_or_insert_with(HashSet::new).insert(res);
}

/// Whether `props` asks for CPU-cached readback memory. On Xbox UMA, `D3D12_HEAP_TYPE_CUSTOM`
/// with `D3D12_CPU_PAGE_PROPERTY_WRITE_BACK` is the general CPU-cached graphics pool (used for
/// hundreds of MB of VBs/IBs/CBs/UAVs), so only explicit `D3D12_HEAP_TYPE_READBACK` heaps — or
/// small non-UAV `COPY_DEST` buffers — stay in the 512 MB coherent window.
fn is_coherent_readback_buffer(props: &D3D12_HEAP_PROPERTIES, state: u32, flags: D3D12_RESOURCE_FLAGS, bytes: usize) -> bool {
    if props.Type == D3D12_HEAP_TYPE_READBACK {
        return bytes <= COHERENT_READBACK_MAX;
    }
    props.Type == D3D12_HEAP_TYPE_CUSTOM
        && props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
        && state == D3D12_RESOURCE_STATE_COPY_DEST.0 as u32
        && (flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0
        && bytes <= COHERENT_CUSTOM_WB_MAX
}

fn is_cpu_readback_buffer(props: &D3D12_HEAP_PROPERTIES, state: u32, flags: D3D12_RESOURCE_FLAGS) -> bool {
    props.Type == D3D12_HEAP_TYPE_READBACK
        || (props.Type == D3D12_HEAP_TYPE_CUSTOM
            && props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
            && state == D3D12_RESOURCE_STATE_COPY_DEST.0 as u32
            && (flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0)
}

unsafe fn place_console_buffer_sparse(this: *mut c_void, d: &D3D12_RESOURCE_DESC, bytes: usize, state: u32, readback_on_null_map: bool, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let (p, _) = super::sparse::alloc(bytes, 0x1_0000)?;
    if let Some((hr, delta)) = super::sparse::place_buffer(this, p, d, state, riid, ppv) {
        UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (p, bytes, true));
        if delta != 0 {
            BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, delta as u64);
        }
        if readback_on_null_map {
            note_sparse_readback(*ppv as usize);
        }
        return Some(hr);
    }
    super::sparse::free(p);
    None
}

/// Sanitize a console buffer description and initial state for placement in a coherent
/// cross-adapter UMA heap (`OpenExistingHeapFromAddress`).
fn sanitize_coherent_buffer(desc: &D3D12_RESOURCE_DESC, state: usize, cross_adapter: bool) -> (D3D12_RESOURCE_DESC, u32) {
    let mut d = *desc;
    d.Flags = D3D12_RESOURCE_FLAGS(d.Flags.0 & 0xFF);
    if cross_adapter {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    }
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Alignment = 0;
    (d, state as u32 & PC_RESOURCE_STATES)
}

unsafe fn place_console_buffer(this: *mut c_void, props: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let u = crate::gpu::uma::get()?;
    if UMA_BUFFER.lock().unwrap().is_empty() || ppv.is_null() {
        return None;
    }
    let d: D3D12_RESOURCE_DESC = std::ptr::read(desc as *const D3D12_RESOURCE_DESC);
    if d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || d.Width == 0 {
        return None;
    }
    let bytes = d.Width as usize;
    let st = state as u32 & PC_RESOURCE_STATES;
    let heap_props = (props != 0 && crate::gpu::shader::readable(props, std::mem::size_of::<D3D12_HEAP_PROPERTIES>()))
        .then(|| std::ptr::read_unaligned(props as *const D3D12_HEAP_PROPERTIES));
    let rounded = (bytes + 0xFFFF) & !0xFFFF;
    let prefer_coherent = heap_props.as_ref().is_some_and(|hp| is_coherent_readback_buffer(hp, st, d.Flags, bytes))
        && has_coherent_headroom(u, rounded);
    let readback_on_null_map = heap_props.as_ref().is_some_and(|hp| is_cpu_readback_buffer(hp, st, d.Flags));
    // Non-readback (and general CUSTOM+WRITE_BACK) committed console buffers live in sparse VRAM
    // windows; small genuine readback buffers prefer coherent committed RAM (spilling to sparse if full).
    if !prefer_coherent && u.has_sparse() {
        if let Some(hr) = place_console_buffer_sparse(this, &d, bytes, st, readback_on_null_map, riid, ppv) {
            return Some(hr);
        }
    }
    let Some(p) = u.alloc(rounded, 0x1_0000).map(|a| a as usize) else {
        if prefer_coherent && u.has_sparse() {
            return place_console_buffer_sparse(this, &d, bytes, st, readback_on_null_map, riid, ppv);
        }
        return None;
    };
    // the heap from OpenExistingHeapFromAddress is cross-adapter: row-major + sanitized flags & state
    let k = u.window_of(p)?;
    let (d, st) = sanitize_coherent_buffer(&d, state, !u.is_driver_window(k));
    let heap = (*UMA_HEAPS.lock().unwrap().get(k)?)?;
    let hr = dev_create_placed_resource(this, heap, p - u.window(k).0, &d as *const _ as usize, st as usize, clear, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        u.free(p);
        static ONCE: AtomicU64 = AtomicU64::new(0);
        if ONCE.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_warn!("console buffer in UMA: CreatePlacedResource {hr:#x} – keeping a separate resource (the shader will not see it)");
        }
        return None;
    }
    UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (p, bytes, true));
    // Map must hand out the same memory the shaders read (checked once)
    static CHECKED: AtomicU64 = AtomicU64::new(0);
    if CHECKED.fetch_add(1, Ordering::Relaxed) == 0 {
        let raw = *ppv;
        if let Some(r) = ID3D12Resource::from_raw_borrowed(&raw) {
            let mut m: *mut c_void = std::ptr::null_mut();
            if r.Map(0, None, Some(&mut m)).is_ok() {
                r.Unmap(0, None);
                if m as usize != p {
                    crate::xlog_warn!("console buffer in UMA: Map returns {m:?}, not {p:#x} – CPU writes will not reach the console address");
                } else {
                    crate::xlog_info!("console buffers in UMA: Map = console address ({p:#x})");
                }
            }
        }
    }
    Some(hr)
}

/// A buffer the title places at its own address (CreatePlacedResourceX): at a placement
/// boundary (64 KiB) in unified memory it is that memory, placed in the window's heap; the
/// title's allocation stays the title's.
pub(super) unsafe fn place_at_title_address(this: *mut c_void, va: usize, desc: &[u8; 0x38], state: u32, clear: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let u = crate::gpu::uma::get()?;
    let k = u.window_of(va)?;
    let (base, size) = u.window(k);
    let mut d: D3D12_RESOURCE_DESC = std::ptr::read_unaligned(desc.as_ptr() as *const D3D12_RESOURCE_DESC);
    let bytes = d.Width as usize;
    // a placed PC buffer starts at 64 KiB: one at another address starts below it and XCL adds
    // the difference where PC methods take an offset into it (`buffer_delta`)
    let start = va & !0xFFFF;
    let delta = va - start;
    if bytes == 0 || va - base + bytes > size {
        return None;
    }
    // the title's own memory (a sparse window): a resource over its video-memory tiles
    if u.is_sparse(va) {
        let (hr, delta) = super::sparse::place_buffer(this, va, &d, state, riid, ppv)?;
        UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (va, bytes, false));
        if delta != 0 {
            BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, delta as u64);
        }
        return Some(hr);
    }
    let heap = (*UMA_HEAPS.lock().unwrap().get(k)?)?;
    if !u.is_driver_window(k) {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    }
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Alignment = 0;
    d.Width = (delta + bytes) as u64;
    let hr = dev_create_placed_resource(this, heap, start - base, &d as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        return None;
    }
    UMA_BUFFERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (va, bytes, false));
    if delta != 0 {
        BUFFER_DELTA.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, delta as u64);
    }
    Some(hr)
}

/// buffers placed below their console address: resource → bytes before it
static BUFFER_DELTA: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// Where offset 0 of a console buffer is in its PC resource (0 for most).
pub fn buffer_delta(res: usize) -> u64 {
    BUFFER_DELTA.lock().unwrap().as_ref().and_then(|m| m.get(&res).copied()).unwrap_or(0)
}

#[cfg(test)]
pub(super) fn set_test_buffer_delta(res: usize, delta: Option<u64>) {
    let mut g = BUFFER_DELTA.lock().unwrap();
    match delta {
        Some(d) => {
            g.get_or_insert_with(HashMap::new).insert(res, d);
        }
        None => {
            if let Some(m) = g.as_mut() {
                m.remove(&res);
            }
        }
    }
}

/// Whether the PC's Map works on `res`: not a reserved resource, and its heap has CPU pages.
unsafe fn pc_can_map(res: *mut c_void) -> bool {
    !is_reserved(res) && pc_mappable(heap_properties(res))
}

/// The heap `res` lives in (not to be asked of a reserved resource).
unsafe fn heap_properties(res: *mut c_void) -> Option<D3D12_HEAP_PROPERTIES> {
    let r = ID3D12Resource::from_raw_borrowed(&res)?;
    let mut p = D3D12_HEAP_PROPERTIES::default();
    r.GetHeapProperties(Some(&mut p), None).ok().map(|_| p)
}

/// Whether the PC can map a resource in a heap with `props`: not a reserved one (no heap), not
/// one in a DEFAULT heap, not one in a custom heap without CPU pages.
fn pc_mappable(props: Option<D3D12_HEAP_PROPERTIES>) -> bool {
    match props {
        None => false,
        Some(p) if p.Type == D3D12_HEAP_TYPE_DEFAULT => false,
        Some(p) if p.Type == D3D12_HEAP_TYPE_CUSTOM => p.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
        Some(_) => true,
    }
}

/// resource Release (every caller): a console buffer's unified memory goes with it
/// Map: on the console every resource is CPU-visible memory. One in unified memory that the
/// PC cannot map (a texture XCL keeps in step with it) gives its unified-memory address – what
/// the console gives; the CPU's writes there reach the texture (`title_textures`).
unsafe extern "system" fn res_map(this: *mut c_void, sub: u32, range: *const c_void, data: *mut *mut c_void) -> HRESULT {
    // a resource in unified memory: its console address (no PC mapping – the PC's would be
    // the start of the PC resource, before the address for a buffer placed below it)
    if let Some((p, bytes)) = console_buffer(this as usize) {
        if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(p)) {
            if !range.is_null() {
                let r = *(range as *const D3D12_RANGE);
                if r.End > r.Begin {
                    let begin = r.Begin.min(bytes);
                    let end = r.End.min(bytes);
                    if end > begin {
                        super::sparse::readback_now(p + begin, end - begin);
                    }
                }
            } else if SPARSE_READBACK.lock().unwrap().as_ref().is_some_and(|s| s.contains(&(this as usize))) {
                super::sparse::readback_now(p, bytes);
            }
        }
        if !data.is_null() {
            *data = p as *mut c_void;
        }
        UNMAPPABLE.lock().unwrap().get_or_insert_with(HashSet::new).insert(this as usize);
        return S_OK;
    }
    // the PC's Map only on a resource it can map: on a reserved one or one without CPU pages it is
    // an invalid call (debug layer 824 / 822) that the runtime answers with E_INVALIDARG
    let hr = if pc_can_map(this) {
        let f: unsafe extern "system" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> HRESULT = std::mem::transmute(orig_fn(this, slot!(this, "Map")));
        f(this, sub, range, data)
    } else {
        E_INVALIDARG
    };
    if hr >= 0 {
        return hr;
    }
    match super::title_textures::address(this as usize).or_else(|| TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).copied())) {
        Some(p) => {
            if !data.is_null() {
                *data = p as *mut c_void;
            }
            UNMAPPABLE.lock().unwrap().get_or_insert_with(HashSet::new).insert(this as usize);
            S_OK
        }
        None => {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                let d = ID3D12Resource::from_raw_borrowed(&this).map(|r| r.GetDesc());
                crate::xlog_warn!("D3D12.X: Map of a resource not in unified memory → {:#x} ({:?})", hr as u32, d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Flags.0)));
            }
            hr
        }
    }
}

/// WriteToSubresource(subresource, box, data, row pitch, depth pitch): the CPU writing texels.
/// Every console resource is CPU-visible memory, a PC texture only in a CPU-visible heap.
unsafe extern "system" fn res_write_to_subresource(this: *mut c_void, sub: u32, dst_box: *const D3D12_BOX, data: *const c_void, row_pitch: u32, depth_pitch: u32) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_BOX, *const c_void, u32, u32) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "WriteToSubresource")));
    let hr = f(this, sub, dst_box, data, row_pitch, depth_pitch);
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static FAILED: AtomicU64 = AtomicU64::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed);
    if hr < 0 && FAILED.fetch_add(1, Ordering::Relaxed) < 8 {
        let d = ID3D12Resource::from_raw_borrowed(&this).map(|r| r.GetDesc());
        let b = dst_box.as_ref().map(|b| (b.left, b.top, b.right, b.bottom));
        crate::xlog_warn!(
            "D3D12.X: WriteToSubresource({sub}, box {b:?}, row pitch {row_pitch}) of {:?} → {:#x} (call #{n})",
            d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Layout.0, d.Flags.0)),
            hr as u32
        );
    }
    hr
}

/// resources the title placed at its own address that got a PC resource of their own: on the
/// console Map gives that address (the CPU's view of the resource)
static TITLE_ADDRESS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// The title address a resource stands for (buffers in unified memory, placed resources).
pub(super) fn title_address(res: usize) -> Option<usize> {
    console_buffer(res).map(|b| b.0).or_else(|| TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&res).copied()))
}

pub(super) fn note_title_address(res: usize, va: usize) {
    TITLE_ADDRESS.lock().unwrap().get_or_insert_with(HashMap::new).insert(res, va);
}

/// resources whose Map gave their unified-memory address (no PC mapping to undo)
static UNMAPPABLE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

unsafe extern "system" fn res_unmap(this: *mut c_void, sub: u32, range: *const c_void) {
    if UNMAPPABLE.lock().unwrap().as_ref().is_some_and(|s| s.contains(&(this as usize))) {
        return;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32, *const c_void) = std::mem::transmute(orig_fn(this, slot!(this, "Unmap")));
    f(this, sub, range)
}

unsafe extern "system" fn res_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    // When opt-in SRV tracking is active, exclude a future resource lease before the native
    // object can die. This is not the descriptor-write gate: descriptor hooks keep that across
    // driver calls which may make a temporary, non-final Release here.
    let mut tracked_srv_lifetime = super::vram::lock_tracked_srv_lifetime();
    // The last reference: kept until the queues are past the work in flight (`late_release`).
    // (Two last-but-one releases at once can miss each other here; the object then goes at
    // once, as before.)
    let mut held = false;
    let n = if super::late_release::enabled() {
        let add: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 1));
        if add(this) == 2 {
            f(this);
            held = true;
            0
        } else {
            f(this);
            f(this)
        }
    } else {
        f(this)
    };
    if n == 0 {
        // Remove lease candidates while Release is still serialized, then leave the lifetime
        // gate before the independent sparse/page/arena cleanup locks below. Do not wait for the
        // global descriptor-write gate here: an unrelated hook may hold it while a driver call
        // makes a temporary Release that waits for this lifetime gate. Any lease already returned
        // owns a strong resource reference and therefore prevents this logical zero.
        super::vram::forget_tracked_after_resource_release(
            this as usize, tracked_srv_lifetime.as_ref(),
        );
        drop(tracked_srv_lifetime.take());
        if let Some(s) = UNMAPPABLE.lock().unwrap().as_mut() {
            s.remove(&(this as usize));
        }
        if let Some(s) = SPARSE_READBACK.lock().unwrap().as_mut() {
            s.remove(&(this as usize));
        }
        if let Some(s) = RESERVED.lock().unwrap().as_mut() {
            s.remove(&(this as usize));
        }
        if let Some(m) = BUFFER_DELTA.lock().unwrap().as_mut() {
            m.remove(&(this as usize));
        }
        if let Some(m) = TITLE_ADDRESS.lock().unwrap().as_mut() {
            m.remove(&(this as usize));
        }
        super::sparse::forget_buffer(this as usize);
        super::vram::forget_resource_backing_after_release(this as usize);
        super::title_textures::forget(this as usize);
        let gone = UMA_BUFFERS.lock().unwrap().as_mut().and_then(|m| m.remove(&(this as usize)));
        if let (Some((p, _, true)), Some(u)) = (gone, crate::gpu::uma::get()) {
            if u.is_sparse(p) {
                super::sparse::free(p);
            } else {
                u.free(p);
            }
        }
    }
    drop(tracked_srv_lifetime.take());
    if held {
        super::late_release::hold(this as usize);
    }
    n
}

/// A stand-in that is no driver object (`standin`) was released for good.
pub(super) fn standin_released(pso: usize) {
    forget_pso(pso);
    super::pso_async::release(super::pso_async::released(pso));
}

/// A stand-in pipeline released for good takes its background-built real pipeline with it.
unsafe extern "system" fn pso_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let pso = this as usize;
    // The driver may reuse the address as soon as Release returns zero.
    let generation = if super::pso_manifest::enabled() {
        super::pso_manifest::ready_compute_generation(pso)
    } else {
        None
    };
    let n = f(this);
    if n == 0 {
        if let Some(generation) = generation {
            super::pso_manifest::forget_ready_compute(pso, generation);
        }
        forget_pso(pso);
        super::pso_async::release(super::pso_async::released(pso));
    }
    n
}

/// A pipeline state is gone: drop what XCL keeps by its address (a later pipeline may get the
/// same address and must not inherit its root signature or compute pass).
/// compute pipelines that run a translated console shader
static TRANSLATED_COMPUTE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// graphics pipelines built from translated console shaders
static TRANSLATED_GRAPHICS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Whether `pso` is a graphics pipeline of translated console shaders.
/// `XCL_TRACE_FRAME`: the console VS and PS (their dump names' hashes) of a translated PSO.
static PSO_SHADERS: Mutex<Option<HashMap<usize, (u64, u64)>>> = Mutex::new(None);

/// translated pipeline → its input layout as text (`XCL_TRACE_FRAME`)
static PSO_LAYOUTS: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);

pub fn pso_layout(pso: usize) -> Option<String> {
    PSO_LAYOUTS.lock().unwrap().as_ref()?.get(&pso).cloned()
}

pub fn pso_shaders(pso: usize) -> Option<(u64, u64)> {
    PSO_SHADERS.lock().unwrap().as_ref()?.get(&pso).copied()
}

pub fn is_translated_graphics(pso: usize) -> bool {
    pso != 0 && TRANSLATED_GRAPHICS.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

/// graphics pipelines given the generated rect list GS (`rect_list_gs`)
static RECT_LIST_GS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Whether `pso` (as `is_translated_graphics` takes it: a stand-in resolved) draws rect lists
/// whole with the generated GS. Its draws must not be expanded to two triangles a rect as well:
/// each half is three corners of the rect and the GS would complete both.
pub fn has_rect_list_gs(pso: usize) -> bool {
    pso != 0 && RECT_LIST_GS.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

/// Note `pso`, made from `copy`, if it kept the generated rect list GS (`rect_gs`; a pipeline
/// made again without it, `retry_without_rect_gs`, did not).
fn note_rect_list_gs(pso: usize, copy: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, rect_gs: bool) {
    if pso != 0 && rect_gs && !copy.GS.pShaderBytecode.is_null() {
        RECT_LIST_GS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

/// Whether `pso` runs a translated console compute shader.
pub fn is_translated_compute(pso: usize) -> bool {
    pso != 0 && TRANSLATED_COMPUTE.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

/// compute pipelines the title made from a console shader that runs untranslated: a background
/// build's stand-in (until it is resolved) and the fallback of a console shader that failed
static CONSOLE_COMPUTE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

unsafe fn note_console_compute(hr: HRESULT, ppv: *mut *mut c_void) {
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        CONSOLE_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    }
}

/// Whether `pso` was made from a console compute shader (translated or not); a native PC
/// compute pipeline is not.
pub fn is_console_compute(pso: usize) -> bool {
    pso != 0 && CONSOLE_COMPUTE.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

#[cfg(test)]
pub(super) fn register_console_compute_for_test(pso: usize) {
    if pso != 0 {
        CONSOLE_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn register_translated_compute_for_test(pso: usize) {
    if pso != 0 {
        TRANSLATED_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn register_rect_list_gs_for_test(pso: usize) {
    if pso != 0 {
        RECT_LIST_GS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn register_translated_graphics_for_test(pso: usize) {
    if pso != 0 {
        TRANSLATED_GRAPHICS.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

#[cfg(test)]
pub(super) fn forget_pso_for_test(pso: usize) {
    forget_pso(pso);
}

fn forget_pso(pso: usize) {
    super::dred::forget_pso(pso);
    dispatch_chain::forget(pso);
    if let Some(s) = TRANSLATED_COMPUTE.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(s) = CONSOLE_COMPUTE.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(s) = TRANSLATED_GRAPHICS.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(s) = RECT_LIST_GS.lock().unwrap().as_mut() {
        s.remove(&pso);
    }
    if let Some(m) = PSO_RS.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    if let Some(m) = PSO_USER_SGPRS.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    if let Some(m) = PSO_DISPATCH_REMAP.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
    remap_fallback::forget(pso);
    super::device_x::forget_base(pso);
    // the compute pass goes with the last user of its entry (`Drop for TwoPassPso`)
    let gone = TWO_PASS.lock().unwrap().as_mut().and_then(|m| m.remove(&pso));
    drop(gone);
}

/// resource #11 GetGPUVirtualAddress for console code
unsafe extern "system" fn res_gpu_va_x(this: *mut c_void) -> u64 {
    if let Some((p, _)) = console_buffer(this as usize) {
        return p as u64;
    }
    let f: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(orig_fn(this, 11));
    // a texture has no GPU address on the PC (0) – on the console every resource has one, and
    // titles use a texture's as the CPU's pointer to it (a retail game clears a 256x256 R32F one
    // through it: with 0 it wrote 256 KB at address 0): the title's address it was placed at
    match f(this) {
        0 => {
            let placed = TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).copied());
            placed.or_else(|| texture_memory(this)).unwrap_or(0) as u64
        }
        va => va,
    }
}

/// Memory for a texture the title did not place itself (a committed one) asking for its
/// address – on the console every texture has one (see `res_gpu_va_x`). Address space for the
/// texels of the PC description (+ a third for mips and 64 KiB of padding), its pages committed
/// as the title writes them (`kernelx::lazy`): the clears of GPU-written targets seen so far
/// cost nothing until done. What the title writes there is not uploaded to the texture, and the
/// memory stays with the address (not freed with the resource).
unsafe fn texture_memory(res: *mut c_void) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESERVE, PAGE_READWRITE};
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap();
    if let Some(p) = TITLE_ADDRESS.lock().unwrap().as_ref().and_then(|m| m.get(&(res as usize)).copied()) {
        return Some(p);
    }
    let d = ID3D12Resource::from_raw_borrowed(&res)?.GetDesc();
    if d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return None;
    }
    let bits = crate::gpu::desc::gfx10_format(d.Format).map_or(128, |f| crate::gpu::desc::format_bytes(f).max(1) as u64 * 8);
    let texels = d.Width.checked_mul(d.Height as u64)?.checked_mul(d.DepthOrArraySize.max(1) as u64)?;
    let bytes = (texels.checked_mul(bits)? / 8 * 4 / 3 + 0x1_0000).next_multiple_of(0x1_0000);
    if bytes > 1 << 30 {
        return None;
    }
    let p = VirtualAlloc(std::ptr::null(), bytes as usize, MEM_RESERVE, PAGE_READWRITE) as usize;
    if p == 0 {
        return None;
    }
    for page in (p..p + bytes as usize).step_by(0x1_0000) {
        crate::hle::kernelx::lazy::mark(page);
    }
    note_title_address(res as usize, p);
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_info!(
            "D3D12.X: GetGPUVirtualAddress of a committed {}x{} texture (format {}, flags {:#x}): {bytes:#x} bytes at {p:#x} for the title to write through (not uploaded)",
            d.Width,
            d.Height,
            d.Format.0,
            d.Flags.0
        );
    }
    Some(p)
}

fn note_buffer_view() {
    if NON_UMA_BUFFER_VIEWS.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_warn!("buffer view in the console heap: the buffer is not in UMA – the shader gets an empty V# (num_records 0)");
    }
}

/// #17 CreateConstantBufferView: a V# of float4 rows at the buffer's console address
/// (`desc::cbv_words`; unified memory, anything else gets an empty V#). No PC CBV: translated
/// shaders read buffers through `mem`.
unsafe extern "system" fn dev_create_cbv_x(this: *mut c_void, desc: *const D3D12_CONSTANT_BUFFER_VIEW_DESC, cpu: u64) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailViewDescriptor);
    let Some(slot) = super::console_heap::slot(cpu).filter(|s| !s.sampler) else {
        let f: unsafe extern "system" fn(*mut c_void, *const D3D12_CONSTANT_BUFFER_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 17));
        return f(this, desc, cpu);
    };
    let _tracked_write_lock = super::vram::lock_tracked_srv_write();
    let (va, size) = desc.as_ref().map_or((0, 0), |d| (d.BufferLocation, d.SizeInBytes));
    let in_uma = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    if !in_uma {
        note_buffer_view();
    }
    let (va, size) = if in_uma { (va, size) } else { (0, 0) };
    super::vram::clear_tracked_srv(&slot);
    super::console_heap::write(&slot, &crate::gpu::desc::cbv_words(va, size));
}

/// #22 CreateSampler: the PC sampler at the slot's index, the S# in XCL ABI form (word 3
/// [11:0] = sampler heap index; the state lives in the PC sampler).
pub(super) unsafe extern "system" fn dev_create_sampler_x(this: *mut c_void, desc: *const D3D12_SAMPLER_DESC, cpu: u64) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailViewDescriptor);
    let f: unsafe extern "system" fn(*mut c_void, *const D3D12_SAMPLER_DESC, u64) = std::mem::transmute(orig_fn(this, 22));
    let Some(slot) = super::console_heap::slot(cpu).filter(|s| s.sampler) else { return f(this, desc, cpu) };
    f(this, desc, slot.pc_srv);
    super::console_heap::publish(&slot, false);
    super::console_heap::write(&slot, &[0, 0, 0, slot.index & 0xFFF]);
}

/// Copy `n` console slots from `src` to `dst` (both in console heaps of one kind): the PC
/// descriptors (2 per view slot) and the slots. False: not two console ranges.
unsafe fn copy_console_slots(this: *mut c_void, n: u32, dst: u64, src: u64, ty: u32) -> bool {
    use super::console_heap::{copy_words, copy_words_captured, slot, SAMPLER_SLOT, SLOT};
    let (Some(d0), Some(s0)) = (slot(dst), slot(src)) else { return false };
    if d0.sampler != s0.sampler {
        return false;
    }
    let _tracked_write_lock = (!d0.sampler).then(super::vram::lock_tracked_srv_write).flatten();
    let step = if d0.sampler { SAMPLER_SLOT } else { SLOT } as u64;
    let ty = pc_heap_type(ty);
    let f: unsafe extern "system" fn(*mut c_void, u32, u64, u64, u32) = std::mem::transmute(orig_fn(this, 24));
    f(this, if d0.sampler { n } else { 2 * n }, d0.pc_srv, s0.pc_srv, ty);
    super::console_heap::publish_range(&d0, n);
    let mut tracked_copies = (!d0.sampler && super::vram::srv_tracking_enabled()).then(Vec::new);
    for k in 0..n as u64 {
        if let (Some(d), Some(s)) = (slot(dst + k * step), slot(src + k * step)) {
            if let Some(copies) = tracked_copies.as_mut() {
                let (source_words, destination_words) = copy_words_captured(&s, &d);
                copies.push((s, d, source_words, destination_words));
            } else {
                copy_words(&s, &d);
            }
        }
    }
    if let Some(copies) = tracked_copies {
        super::vram::copy_tracked_srvs(this as usize, &copies);
    }
    true
}

/// `XCL_TRACE_FRAME`: the device's descriptor copies (first few, and a count)
fn note_copy_call(what: &str, n: u32, dst: u64, src: u64) {
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if super::present::trace_frame() && (k < 4 || k.is_power_of_two()) {
        crate::xlog_info!("frame trace: {what} #{k}: {n} from {src:#x} to {dst:#x}");
    }
}

static MIXED_COPIES: AtomicU64 = AtomicU64::new(0);

fn note_mixed_copy() {
    if MIXED_COPIES.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_warn!("CopyDescriptors (console): range outside the XCL console heaps – copying on the PC only");
    }
}

/// #24 CopyDescriptorsSimple(n, dst, src, type)
unsafe extern "system" fn dev_copy_descriptors_simple_x(this: *mut c_void, n: u32, dst: u64, src: u64, ty: u32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailViewDescriptor);
    let ty = pc_heap_type(ty);
    note_copy_call("CopyDescriptorsSimple", n, dst, src);
    if copy_console_slots(this, n, dst, src, ty) {
        return;
    }
    if super::console_heap::slot(dst).is_some() || super::console_heap::slot(src).is_some() {
        note_mixed_copy();
        return;
    }
    let f: unsafe extern "system" fn(*mut c_void, u32, u64, u64, u32) = std::mem::transmute(orig_fn(this, 24));
    f(this, n, dst, src, ty);
}

/// #23 CopyDescriptors(dst ranges, src ranges, type): range by range over console slots.
#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn dev_copy_descriptors_x(this: *mut c_void, nd: u32, dsts: *const u64, dsizes: *const u32, ns: u32, srcs: *const u64, ssizes: *const u32, ty: u32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailViewDescriptor);
    let ty = pc_heap_type(ty);
    note_copy_call("CopyDescriptors", nd, if dsts.is_null() { 0 } else { *dsts }, if srcs.is_null() { 0 } else { *srcs });
    use super::console_heap::{slot, SAMPLER_SLOT, SLOT};
    let console = !dsts.is_null() && !srcs.is_null() && nd > 0 && slot(*dsts).is_some();
    if !console {
        if !srcs.is_null() && ns > 0 && slot(*srcs).is_some() {
            note_mixed_copy();
            return;
        }
        type F = unsafe extern "system" fn(*mut c_void, u32, *const u64, *const u32, u32, *const u64, *const u32, u32);
        let f: F = std::mem::transmute(orig_fn(this, 23));
        return f(this, nd, dsts, dsizes, ns, srcs, ssizes, ty);
    }
    // every descriptor as (dst, src) handle, ranges walked in step
    let step = if ty == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32 { SAMPLER_SLOT } else { SLOT } as u64;
    let expand = |starts: *const u64, sizes: *const u32, n: u32| -> Vec<u64> {
        (0..n as usize).flat_map(|r| {
            let (h, len) = (*starts.add(r), if sizes.is_null() { 1 } else { *sizes.add(r) });
            (0..len as u64).map(move |k| h + k * step)
        })
        .collect()
    };
    let (d, s) = (expand(dsts, dsizes, nd), expand(srcs, ssizes, ns));
    for (a, b) in d.iter().zip(s.iter()) {
        if !copy_console_slots(this, 1, *a, *b, ty) {
            note_mixed_copy();
        }
    }
}

/// Remove address-keyed state before a last native Release can expose that address for reuse.
/// A simultaneous Release can make the probe miss the last reference; the post-free `take`
/// prevents stale state from leaking but cannot protect a replacement already at the address.
unsafe fn release_taking_first<T>(this: *mut c_void, mut take: impl FnMut() -> T, after: impl FnOnce(T)) -> u32 {
    let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 1));
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let take_first = add_ref(this) == 2;
    let taken = take_first.then(&mut take);
    // Drop the temporary probe reference, then the caller's reference.
    f(this);
    let n = f(this);
    if n == 0 {
        after(match taken {
            Some(taken) => taken,
            None => take(),
        });
    }
    n
}

/// descriptor heap Release (every caller): a console heap's slots go with it
unsafe extern "system" fn heap_release(this: *mut c_void) -> u32 {
    let lifetime = super::console_heap::lock_lifetime();
    if lifetime.is_none() {
        return release_taking_first(this, || super::console_heap::take(this as usize), |h| h.into_iter().for_each(super::console_heap::free));
    }
    let generation = lifetime.as_ref().and_then(|_| super::console_heap::generation(this as usize));
    let mut taken = None;
    let n = release_taking_first(
        this,
        || super::console_heap::take_after_release(this as usize, generation, lifetime.as_ref()),
        |h| taken = h,
    );
    drop(lifetime);
    taken.into_iter().for_each(super::console_heap::free);
    n
}

/// Default UAV dimension for a texture resource when `CreateUnorderedAccessView` receives a NULL desc.
fn default_uav_dimension(rd: &D3D12_RESOURCE_DESC) -> D3D12_UAV_DIMENSION {
    match rd.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            if rd.DepthOrArraySize > 1 { D3D12_UAV_DIMENSION_TEXTURE1DARRAY } else { D3D12_UAV_DIMENSION_TEXTURE1D }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => D3D12_UAV_DIMENSION_TEXTURE3D,
        _ => {
            if rd.DepthOrArraySize > 1 { D3D12_UAV_DIMENSION_TEXTURE2DARRAY } else { D3D12_UAV_DIMENSION_TEXTURE2D }
        }
    }
}

/// Build an explicit default `D3D12_UNORDERED_ACCESS_VIEW_DESC` matching `rd`'s dimension and
/// array/depth extent with format `fmt` (used when `desc` was NULL and `rd.Format` needed linear/view mapping).
fn default_uav_desc(rd: &D3D12_RESOURCE_DESC, fmt: u32) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let dim = default_uav_dimension(rd);
    let mut d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(fmt as i32),
        ViewDimension: dim,
        ..Default::default()
    };
    match dim {
        D3D12_UAV_DIMENSION_TEXTURE1DARRAY => {
            d.Anonymous.Texture1DArray.ArraySize = rd.DepthOrArraySize as u32;
        }
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
            d.Anonymous.Texture2DArray.ArraySize = rd.DepthOrArraySize as u32;
        }
        D3D12_UAV_DIMENSION_TEXTURE3D => {
            d.Anonymous.Texture3D.WSize = rd.DepthOrArraySize as u32;
        }
        _ => {}
    }
    d
}

/// Device #19 wrapper: guards non-depth texture UAVs against sRGB or host-unsupported
/// `D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW` formats (preventing `0x887A0001`)
/// before delegating to `dev_create_uav_x`.
unsafe extern "system" fn dev_create_uav_checked_x(
    this: *mut c_void,
    res: *mut c_void,
    counter: *mut c_void,
    desc: *const D3D12_UNORDERED_ACCESS_VIEW_DESC,
    cpu: u64,
) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailViewDescriptor);
    if let Some(r) = ID3D12Resource::from_raw_borrowed(&res) {
        let rd = r.GetDesc();
        let is_tex = rd.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER;
        let has_uav_flag = rd.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0;
        if is_tex && has_uav_flag {
            let mut patched = desc.as_ref().copied();
            let raw_fmt = patched.as_ref().map_or(0, |d| d.Format.0 as u32);
            let res_fmt = rd.Format.0 as u32;
            let mapped_fmt = super::device_x::pc_uav_linear_format(if raw_fmt == 0 { res_fmt } else { raw_fmt });
            let eff_fmt = super::device_x::pc_shader_view_format(res_fmt, mapped_fmt);
            if let Some(c) = patched.as_mut() {
                if eff_fmt != 0 && (eff_fmt != raw_fmt && (raw_fmt != 0 || eff_fmt != res_fmt)) {
                    c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(eff_fmt as i32);
                }
            } else if eff_fmt != 0 && eff_fmt != res_fmt {
                patched = Some(default_uav_desc(&rd, eff_fmt));
            }
            if eff_fmt != 0 && !super::device_x::pc_typed_uav_supported(this, eff_fmt as i32) {
                static N: AtomicU64 = AtomicU64::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!(
                        "D3D12.X: CreateUnorderedAccessView format {} (resource fmt {}) lacks TYPED_UNORDERED_ACCESS_VIEW support on this GPU – binding null view",
                        eff_fmt,
                        rd.Format.0
                    );
                }
                let null_dim = desc.as_ref().map_or_else(|| default_uav_dimension(&rd), |d| d.ViewDimension);
                let null_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT,
                    ViewDimension: null_dim,
                    ..Default::default()
                };
                return dev_create_uav_x(this, std::ptr::null_mut(), std::ptr::null_mut(), &null_desc, cpu);
            }
            if let Some(ref c) = patched {
                return dev_create_uav_x(this, res, counter, c as *const _, cpu);
            }
        }
    }
    dev_create_uav_x(this, res, counter, desc, cpu)
}

/// #21 CreateDepthStencilView: the console's view formats as the PC's (`pc_view_format`).
unsafe extern "system" fn dev_create_dsv_x(this: *mut c_void, res: *mut c_void, desc: *const D3D12_DEPTH_STENCIL_VIEW_DESC, cpu: u64) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailViewDescriptor);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_DEPTH_STENCIL_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 21));
    let mut copy = desc.as_ref().copied();
    if let Some(c) = copy.as_mut() {
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_target_format(c.Format.0 as u32) as i32);
    }
    // no description: the resource's format – which a depth buffer made typeless
    // (`device_x::pc_resource_desc`) is not one a view can have
    if copy.is_none() {
        copy = ID3D12Resource::from_raw_borrowed(&res).and_then(|r| default_dsv(&r.GetDesc()));
    }
    f(this, res, copy.as_ref().map_or(desc, |c| c as *const _), cpu)
}

/// The view a DSV without a description has, for a depth buffer `pc_resource_desc` made
/// typeless (None: the resource's own format is a depth one).
fn default_dsv(d: &D3D12_RESOURCE_DESC) -> Option<D3D12_DEPTH_STENCIL_VIEW_DESC> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    let format = match d.Format.0 {
        19 => 20,
        39 => 40,
        44 => 45,
        53 => 55,
        _ => return None,
    };
    let mut v = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT(format), ..Default::default() };
    if d.DepthOrArraySize > 1 {
        v.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_DSV { MipSlice: 0, FirstArraySlice: 0, ArraySize: d.DepthOrArraySize as u32 };
    } else {
        v.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    }
    Some(v)
}

/// A shader view `v` (None: the resource's default one) of `rd` with the PC's format names
/// (`device_x::pc_shader_view_format`) – None: as it is.
fn pc_srv(rd: &D3D12_RESOURCE_DESC, v: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    if rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return None;
    }
    let rf = rd.Format.0 as u32;
    let mut out = match v {
        Some(v) => *v,
        None => {
            // only a typeless depth buffer needs a description of its own
            if !matches!(rf, 19 | 39 | 44 | 53) {
                return None;
            }
            let mut v = D3D12_SHADER_RESOURCE_VIEW_DESC { Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
            let slices = u32::from(rd.DepthOrArraySize);
            // a multisampled depth buffer has only the MS view dimensions
            match (rd.SampleDesc.Count > 1, slices > 1) {
                (true, true) => {
                    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                    v.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV { FirstArraySlice: 0, ArraySize: slices };
                }
                (true, false) => v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS,
                (false, true) => {
                    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                    v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MipLevels: u32::MAX, ArraySize: slices, ..Default::default() };
                }
                (false, false) => {
                    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                    v.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: u32::MAX, ..Default::default() };
                }
            }
            v
        }
    };
    if out.ViewDimension == D3D12_SRV_DIMENSION_BUFFER || out.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE {
        return None;
    }
    let f = super::device_x::pc_shader_view_format(rf, out.Format.0 as u32);
    let stencil = super::device_x::stencil_view(rf, f);
    if v.is_some() && f == out.Format.0 as u32 && !stencil {
        return None;
    }
    out.Format = DXGI_FORMAT(f as i32);
    // the PC reads stencil from plane 1; the console's view of it is plane 0 in a stencil format
    if stencil {
        if out.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D {
            out.Anonymous.Texture2D.PlaneSlice = 1;
        } else if out.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
            out.Anonymous.Texture2DArray.PlaneSlice = 1;
        }
    }
    Some(out)
}

unsafe extern "system" fn dev_create_srv_x(this: *mut c_void, res: *mut c_void, desc: *const D3D12_SHADER_RESOURCE_VIEW_DESC, cpu: u64) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailViewDescriptor);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 18));
    let mut copy = desc.as_ref().copied();
    if let Some(c) = copy.as_mut() {
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_view_format(c.Format.0 as u32) as i32);
        if res.is_null()
            && c.Format == windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN
            && view_dim_to_image_type(c.ViewDimension.0).is_some()
        {
            c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;
        }
    }
    // a depth buffer's view in the formats the PC has for it (a D24S8 format itself, R16_UNORM of
    // a D16S8 buffer …: an invalid call, the device removed)
    if let Some(v) = ID3D12Resource::from_raw_borrowed(&res).and_then(|r| pc_srv(&r.GetDesc(), copy.as_ref())) {
        copy = Some(v);
    }
    let desc = copy.as_ref().map_or(desc, |c| c as *const _);
    let Some(slot) = super::console_heap::slot(cpu) else { return f(this, res, desc, cpu) };
    let _tracked_write_lock = super::vram::lock_tracked_srv_write();
    super::vram::clear_tracked_srv(&slot);
    // Raytracing acceleration structure SRV (pResource == NULL, Location == console GPU VA):
    // translated shaders read BVH nodes from UMA (`bvh_intersect`), never from a PC DXR SRV,
    // so write the RDNA2 BVH descriptor directly and skip forwarding the console VA to PC.
    if let Some(v) = desc.as_ref().filter(|v| v.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE) {
        let va = v.Anonymous.RaytracingAccelerationStructure.Location;
        let in_uma = va != 0 && crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
        if in_uma {
            let w = crate::gpu::desc::bvh_words(va, true);
            return super::console_heap::write(&slot, &[w[0], w[1], w[2], w[3], 0, 0, 0, 0]);
        }
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: CreateShaderResourceView(RAYTRACING_ACCELERATION_STRUCTURE) at {va:#x} is not in unified memory – writing empty BVH descriptor");
        }
        return super::console_heap::write(&slot, &[0; 8]);
    }
    // Placed buffers with `buffer_delta != 0` can have unaligned structured offsets on the PC
    // backing; write a valid null PC buffer SRV so `slot.pc_srv` and `GLOBAL` never keep a stale
    // descriptor while translated shaders read the exact UMA address via `buffer_view` below.
    let buf_delta_null = desc
        .as_ref()
        .is_some_and(|v| v.ViewDimension == D3D12_SRV_DIMENSION_BUFFER && buffer_delta(res as usize) != 0);
    // translated shaders declare every cube T# as a TextureCubeArray: a title's TEXTURECUBE
    // view becomes the cube array of that one cube (a dimension mismatch is undefined in D3D12)
    let cube = desc.as_ref().and_then(cube_as_cube_array);
    let desc = cube.as_ref().map_or(desc, |v| v as *const _);
    if buf_delta_null {
        let null_buf = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        f(this, std::ptr::null_mut(), &null_buf, slot.pc_srv);
    } else {
        f(this, res, desc, slot.pc_srv);
    }
    super::console_heap::publish(&slot, false);
    super::console_heap::note_resource(&slot, res as usize);
    if !res.is_null() && super::memory_report::enabled() {
        super::vram::note_srv(res as usize, desc.as_ref());
    }
    if super::present::trace_frame() && super::title_textures::address(res as usize).is_some() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 24 {
            let rd = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc());
            let v = desc.as_ref().map(|v| (v.Format.0, v.ViewDimension.0, v.Shader4ComponentMapping, v.Anonymous.Texture2D.MostDetailedMip, v.Anonymous.Texture2D.MipLevels, v.Anonymous.Texture2D.ResourceMinLODClamp));
            crate::xlog_info!(
                "frame trace: SRV of title texture {res:p} {:?}: view (format, dimension, mapping, mip, mips, clamp) {v:x?}",
                rd.map(|d| (d.Width, d.Height, d.Format.0, d.MipLevels, d.Flags.0))
            );
        }
    }
    let Some(r) = ID3D12Resource::from_raw_borrowed(&res) else {
        // A null image SRV was already created and published at `slot.index`. Writing `[0; 8]`
        // would point shaders at `ResourceDescriptorHeap[0]` (memory window 0 / uninitialized),
        // so write a valid T# referencing `slot.index` with the view's format class and type.
        if let Some(v) = desc.as_ref() {
            if let Some(ty) = view_dim_to_image_type(v.ViewDimension.0) {
                let fmt = crate::gpu::desc::gfx10_format(v.Format).unwrap_or(0);
                let depth = if ty == crate::gpu::desc::ImageType::Cube { 6 } else { 1 };
                return super::console_heap::write(&slot, &crate::gpu::desc::xcl_image_for_view(slot.index, fmt, ty, 1, 1, depth, 1));
            }
        }
        return super::console_heap::write(&slot, &[0; 8]);
    };
    if let Some(v) = desc.as_ref().filter(|v| v.ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        return super::console_heap::write(&slot, &buffer_view(res as usize, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw));
    }
    let rdesc = r.GetDesc();
    match srv_tsharp_from_desc(slot.index, &rdesc, desc.as_ref()) {
        Some(t) => {
            super::console_heap::write(&slot, &t);
            super::vram::track_srv(this as usize, res as usize, &slot, desc.as_ref().copied(), t);
        }
        None => {
            note_buffer_view();
            super::console_heap::write(&slot, &[0; 8]);
        }
    }
}

/// A TEXTURECUBE view as the TEXTURECUBEARRAY of its one cube (None: any other view).
fn cube_as_cube_array(v: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    if v.ViewDimension != D3D12_SRV_DIMENSION_TEXTURECUBE {
        return None;
    }
    // SAFETY: the union member the view dimension names
    let c = unsafe { v.Anonymous.TextureCube };
    Some(D3D12_SHADER_RESOURCE_VIEW_DESC {
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV {
                MostDetailedMip: c.MostDetailedMip,
                MipLevels: c.MipLevels,
                First2DArrayFace: 0,
                NumCubes: 1,
                ResourceMinLODClamp: c.ResourceMinLODClamp,
            },
        },
        ..*v
    })
}

/// The texture a view by address describes: `desc` (a resource desc) placed at `va`, made once
/// per address and desc – titles make views of the same memory again and again.
unsafe fn resource_at(this: *mut c_void, desc: *const u8, va: u64) -> *mut c_void {
    type Key = (u64, [u8; 0x38]);
    static MADE: Mutex<Option<HashMap<Key, usize>>> = Mutex::new(None);
    if desc.is_null() || va == 0 || !crate::gpu::shader::readable(desc as usize, 0x38) {
        return std::ptr::null_mut();
    }
    // the PC part of the desc; the flags (@0x30) are its last dword, padding follows
    let mut x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    x[0x34..].fill(0);
    if let Some(&r) = MADE.lock().unwrap().get_or_insert_with(HashMap::new).get(&(va, x)) {
        return r as *mut c_void;
    }
    let mut r = std::ptr::null_mut();
    let hr = super::device_x::create_placed_resource_x(this, va, x.as_ptr(), 0, std::ptr::null(), &ID3D12Resource::IID as *const _ as *const GUID, &mut r);
    if hr < 0 || r.is_null() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            crate::xlog_warn!("D3D12.X: view at {va:#x}: the resource cannot be made ({hr:#x}); desc {x:02x?}");
        }
        r = std::ptr::null_mut();
    }
    MADE.lock().unwrap().get_or_insert_with(HashMap::new).insert((va, x), r as usize);
    r
}

/// device #70 (a retail game): a shader resource view of memory, without a resource object:
/// (resource desc, { the PC's SRV desc; GPU address @0x28; … }, destination).
unsafe extern "system" fn dev_create_srv_at_x(this: *mut c_void, rdesc: *const u8, view: *const u8, cpu: u64) {
    if view.is_null() || !crate::gpu::shader::readable(view as usize, 0x40) {
        return;
    }
    let va = (view.add(0x28) as *const u64).read_unaligned();
    let v = (view as *const D3D12_SHADER_RESOURCE_VIEW_DESC).read_unaligned();
    if v.ViewDimension == D3D12_SRV_DIMENSION_BUFFER {
        let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailViewDescriptor);
        let Some(slot) = super::console_heap::slot(cpu) else { return };
        let _tracked_write_lock = super::vram::lock_tracked_srv_write();
        super::vram::clear_tracked_srv(&slot);
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        let words = buffer_words_at(va, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw);
        return super::console_heap::write(&slot, &words);
    }
    dev_create_srv_x(this, resource_at(this, rdesc, va), &v, cpu);
}

/// device #71: the same for an unordered access view ({ the PC's UAV desc; GPU address @0x28;
/// the counter's address @0x30; … }).
unsafe extern "system" fn dev_create_uav_at_x(this: *mut c_void, rdesc: *const u8, view: *const u8, cpu: u64) {
    if view.is_null() || !crate::gpu::shader::readable(view as usize, 0x40) {
        return;
    }
    let va = (view.add(0x28) as *const u64).read_unaligned();
    let v = (view as *const D3D12_UNORDERED_ACCESS_VIEW_DESC).read_unaligned();
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailViewDescriptor);
    if v.ViewDimension == D3D12_UAV_DIMENSION_BUFFER {
        let Some(slot) = super::console_heap::slot(cpu) else { return };
        let _tracked_write_lock = super::vram::lock_tracked_srv_write();
        super::vram::clear_tracked_srv(&slot);
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        let words = buffer_words_at(va, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw);
        return super::console_heap::write(&slot, &words);
    }
    dev_create_uav_x(this, resource_at(this, rdesc, va), std::ptr::null_mut(), &v, cpu);
}

/// V# of `n` elements from element `first` of the buffer at `va`.
fn buffer_words_at(va: u64, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, first: u64, n: u32, stride: u32, raw: bool) -> [u32; 4] {
    let fmt = crate::gpu::desc::gfx10_format(format).unwrap_or(22);
    let (stride, elem) = match (raw, stride) {
        (true, _) => (0, 4),
        (false, 0) => {
            let e = crate::gpu::desc::format_bytes(fmt);
            (e, e)
        }
        (false, s) => (s, s),
    };
    let records = if stride == 0 { n.saturating_mul(4) } else { n };
    crate::gpu::desc::buffer_words(va + first * elem as u64, stride, records, fmt)
}

fn resolve_uav_array_size(array_size: u32, total_slices: u32, first_slice: u32) -> u32 {
    if array_size == 0 || array_size == u32::MAX {
        total_slices.saturating_sub(first_slice).max(1)
    } else {
        array_size
    }
}

/// Null-resource UAV descriptor matching `desc`'s (or `rd`'s) image dimension so PC D3D12
/// accepts a null UAV fallback when a resource lacks `ALLOW_UNORDERED_ACCESS`.
fn null_uav_desc(
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
    rd: Option<&D3D12_RESOURCE_DESC>,
) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;
    let dim = desc
        .map(|v| v.ViewDimension.0)
        .or_else(|| rd.map(resource_dim_to_view))
        .unwrap_or(D3D12_UAV_DIMENSION_TEXTURE2D.0);
    match dim {
        2 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1D,
            ..Default::default()
        },
        3 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture1DArray: D3D12_TEX1D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: 1 },
            },
        },
        5 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: 1, PlaneSlice: 0 },
            },
        },
        8 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_UAV { MipSlice: 0, FirstWSlice: 0, WSize: 1 },
            },
        },
        _ => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        },
    }
}

/// Companion SRV for an image UAV slot so translated shaders can read from the UAV's T#
/// (`ResourceDescriptorHeap[slot.index]`). Returns `None` for buffer or unsupported dimensions.
/// When `null_resource` is true (either `pResource == NULL` or `DENY_SHADER_RESOURCE`),
/// returns a valid null-resource SRV descriptor of the matching dimension and concrete format.
fn uav_companion_srv(
    rd: Option<&D3D12_RESOURCE_DESC>,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
    null_resource: bool,
) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_UNKNOWN};

    let (dim, raw_fmt, mip, first_slice, array_size, plane) = match desc {
        Some(v) => {
            // SAFETY: each branch reads the union field named by `v.ViewDimension`.
            let (mip, first, array, plane) = unsafe {
                match v.ViewDimension.0 {
                    2 => (v.Anonymous.Texture1D.MipSlice, 0, 1, 0),
                    3 => (
                        v.Anonymous.Texture1DArray.MipSlice,
                        v.Anonymous.Texture1DArray.FirstArraySlice,
                        v.Anonymous.Texture1DArray.ArraySize,
                        0,
                    ),
                    4 => (v.Anonymous.Texture2D.MipSlice, 0, 1, v.Anonymous.Texture2D.PlaneSlice),
                    5 => (
                        v.Anonymous.Texture2DArray.MipSlice,
                        v.Anonymous.Texture2DArray.FirstArraySlice,
                        v.Anonymous.Texture2DArray.ArraySize,
                        v.Anonymous.Texture2DArray.PlaneSlice,
                    ),
                    8 => (v.Anonymous.Texture3D.MipSlice, 0, 1, 0),
                    _ => return None,
                }
            };
            (v.ViewDimension.0, v.Format, mip, first, array, plane)
        }
        None => {
            let r = rd?;
            let dim = resource_dim_to_view(r);
            if !matches!(dim, 2 | 3 | 4 | 5 | 8) {
                return None;
            }
            (dim, r.Format, 0, 0, r.DepthOrArraySize as u32, 0)
        }
    };

    let fmt = if null_resource {
        // Avoid inheriting a potentially typeless or depth resource format for a null SRV.
        match desc.map(|v| v.Format).filter(|&f| f != DXGI_FORMAT_UNKNOWN) {
            Some(f) => f,
            None => DXGI_FORMAT_R32_FLOAT,
        }
    } else if raw_fmt == DXGI_FORMAT_UNKNOWN {
        rd.map_or(DXGI_FORMAT_R32_FLOAT, |r| r.Format)
    } else {
        raw_fmt
    };

    let (mip, first_slice, array_size, plane) = if null_resource {
        (0, 0, 1, 0)
    } else {
        let total = rd.map_or(1, |r| r.DepthOrArraySize as u32);
        let array = resolve_uav_array_size(array_size, total, first_slice);
        (mip, first_slice, array, plane)
    };

    let mut srv = match dim {
        2 => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE1D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture1D: D3D12_TEX1D_SRV { MostDetailedMip: mip, MipLevels: 1, ResourceMinLODClamp: 0.0 },
            },
        },
        3 => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE1DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture1DArray: D3D12_TEX1D_ARRAY_SRV {
                    MostDetailedMip: mip,
                    MipLevels: 1,
                    FirstArraySlice: first_slice,
                    ArraySize: array_size,
                    ResourceMinLODClamp: 0.0,
                },
            },
        },
        4 => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: mip, MipLevels: 1, PlaneSlice: plane, ResourceMinLODClamp: 0.0 },
            },
        },
        5 => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: mip,
                    MipLevels: 1,
                    FirstArraySlice: first_slice,
                    ArraySize: array_size,
                    PlaneSlice: plane,
                    ResourceMinLODClamp: 0.0,
                },
            },
        },
        8 => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_SRV { MostDetailedMip: mip, MipLevels: 1, ResourceMinLODClamp: 0.0 },
            },
        },
        _ => return None,
    };

    if let Some(r) = rd.filter(|_| !null_resource) {
        if let Some(v) = pc_srv(r, Some(&srv)) {
            srv = v;
        }
    }
    Some(srv)
}

unsafe extern "system" fn dev_create_uav_x(this: *mut c_void, res: *mut c_void, counter: *mut c_void, desc: *const D3D12_UNORDERED_ACCESS_VIEW_DESC, cpu: u64) {
    // the view's format as the PC's for the resource (B5G5R5A1 of a target made R8G8B8A8,
    // `device_x::pc_resource_desc`: an invalid call, the device removed)
    let mut copy = desc.as_ref().copied();
    if let Some(c) = copy.as_mut() {
        if let Some(r) = ID3D12Resource::from_raw_borrowed(&res) {
            let rf = r.GetDesc().Format.0 as u32;
            c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_shader_view_format(rf, c.Format.0 as u32) as i32);
        } else {
            c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_view_format(c.Format.0 as u32) as i32);
            if c.Format == windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN && view_dim_to_image_type(c.ViewDimension.0).is_some() {
                c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;
            }
        }
    }
    let desc = copy.as_ref().map_or(desc, |c| c as *const _);
    super::present::note_use(res as usize, super::present::USE_UAV);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const D3D12_UNORDERED_ACCESS_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 19));
    let rd_opt = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc());
    if super::present::trace_frame() {
        if rd_opt.is_some_and(|d| d.Width >= 1920 && d.Flags.0 & 1 != 0) {
            let s = super::console_heap::slot(cpu);
            crate::xlog_info!(
                "frame trace: UAV of render target {res:p} {:?} at {cpu:#x}: console slot {:?}, view {:?}",
                rd_opt.map(|d| (d.Width, d.Height, d.Format.0)),
                s.as_ref().map(|s| (s.index, s.pc_uav, s.at)),
                desc.as_ref().map(|v| (v.Format.0, v.ViewDimension.0))
            );
        }
    }
    // a resource without ALLOW_UNORDERED_ACCESS (a console depth buffer that also had it: the PC
    // refuses the pair, `device_x::pc_resource_desc`): a null view – a UAV of it is an invalid
    // call, and the device is removed (0x887A0001)
    let no_uav = rd_opt.is_some_and(|d| d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0);
    let null_desc = null_uav_desc(desc.as_ref(), rd_opt.as_ref());
    let pc_uav = |at: u64| {
        if no_uav {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: a UAV of a resource that cannot have one on the PC (a depth buffer) – a null view");
            }
            f(this, std::ptr::null_mut(), std::ptr::null_mut(), &null_desc, at)
        } else {
            f(this, res, counter, desc, at)
        }
    };
    let Some(slot) = super::console_heap::slot(cpu) else { return pc_uav(cpu) };
    let _tracked_write_lock = super::vram::lock_tracked_srv_write();
    super::vram::clear_tracked_srv(&slot);
    pc_uav(slot.pc_uav);
    super::console_heap::publish(&slot, true);
    super::console_heap::note_resource(&slot, res as usize);
    let Some(r) = ID3D12Resource::from_raw_borrowed(&res) else {
        // A null image UAV was already created and published at `slot.index + 1`. Create a
        // matching null companion SRV at `slot.index` and write a valid T# referencing
        // `slot.index` so shaders never fall back to `ResourceDescriptorHeap[0]` / `[1]`.
        if let Some(v) = desc.as_ref() {
            if let Some(ty) = view_dim_to_image_type(v.ViewDimension.0) {
                if let Some(srv) = uav_companion_srv(None, Some(v), true) {
                    let g: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 18));
                    g(this, std::ptr::null_mut(), &srv, slot.pc_srv);
                    super::console_heap::publish(&slot, false);
                }
                let fmt = crate::gpu::desc::gfx10_format(v.Format).unwrap_or(0);
                return super::console_heap::write(&slot, &crate::gpu::desc::xcl_image_for_view(slot.index, fmt, ty, 1, 1, 1, 1));
            }
        }
        return super::console_heap::write(&slot, &[0; 8]);
    };
    let rd = rd_opt.unwrap();
    if let Some(v) = desc.as_ref().filter(|v| v.ViewDimension == D3D12_UAV_DIMENSION_BUFFER) {
        let b = &v.Anonymous.Buffer;
        let raw = b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        return super::console_heap::write(&slot, &buffer_view(res as usize, v.Format, b.FirstElement, b.NumElements, b.StructureByteStride, raw));
    }
    // UAV dimensions: 1 buffer, 2 tex1d, 3 tex1d array, 4 tex2d, 5 tex2d array, 8 tex3d
    let (format, dim, array, mip) = match desc.as_ref() {
        Some(v) => {
            let a = &v.Anonymous;
            let (array, mip) = match v.ViewDimension.0 {
                2 => (1, a.Texture1D.MipSlice),
                3 => (resolve_uav_array_size(a.Texture1DArray.ArraySize, rd.DepthOrArraySize as u32, a.Texture1DArray.FirstArraySlice), a.Texture1DArray.MipSlice),
                4 => (1, a.Texture2D.MipSlice),
                5 => (resolve_uav_array_size(a.Texture2DArray.ArraySize, rd.DepthOrArraySize as u32, a.Texture2DArray.FirstArraySlice), a.Texture2DArray.MipSlice),
                8 => (a.Texture3D.WSize, a.Texture3D.MipSlice),
                _ => (1, 0),
            };
            (v.Format, v.ViewDimension.0, array, mip)
        }
        None => (rd.Format, resource_dim_to_view(&rd), rd.DepthOrArraySize as u32, 0),
    };
    let Some((fmt, ty, w, h, d, _)) = view_shape(r, format, dim, array, 1) else {
        note_buffer_view();
        return super::console_heap::write(&slot, &[0; 8]);
    };
    // RDNA images have no SRV / UAV distinction: console shaders may read any image UAV slot
    // (`image_load`, `image_sample_*`, `image_get_resinfo` all index `ResourceDescriptorHeap[slot.index]`).
    let deny_srv = rd.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 != 0;
    if let Some(srv) = uav_companion_srv(Some(&rd), desc.as_ref(), deny_srv) {
        let srv_res = if deny_srv { std::ptr::null_mut() } else { res };
        let g: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, u64) = std::mem::transmute(orig_fn(this, 18));
        g(this, srv_res, &srv, slot.pc_srv);
        super::console_heap::publish(&slot, false);
    }
    let (w, h) = ((w >> mip).max(1), (h >> mip).max(1));
    let d = if dim == 8 { (d >> mip).max(1) } else { d };
    super::console_heap::write(&slot, &crate::gpu::desc::xcl_image_for_view(slot.index, fmt, ty, w, h, d, 1));
}

/// device #127 (a retail game): CreateCommittedResource with a barrier layout in the initial state's place
/// – the arguments of the PC's CreateCommittedResource3 (heap properties, flags, desc, layout,
/// clear value, protected session, castable formats, riid, resource).
#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn dev_create_committed_resource_layout_x(
    this: *mut c_void,
    props: usize,
    flags: u32,
    desc: usize,
    layout: u32,
    clear: usize,
    _protected: usize,
    _castable: u32,
    _formats: usize,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    note_newer_device_method(127);
    let state = super::device_x::layout_state(layout) as usize;
    if flags & D3D12_HEAP_FLAG_ALLOW_DISPLAY.0 as u32 != 0 && desc != 0 && crate::gpu::shader::readable(desc, 0x38) {
        return create_display_target(this, props, desc, state, clear, riid, ppv);
    }
    dev_create_committed_resource_x(this, props, flags as usize, desc, state, clear, riid, ppv)
}

/// A display target: its console memory comes from the title's allocation hook, as the console
/// runtime's does (a retail game gives its back buffers one block and asserts that it is used up); on the
/// PC it is a resource of its own.
unsafe fn create_display_target(this: *mut c_void, props: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let d = super::device_x::pc_resource_desc(&x);
    let hr = dev_create_committed_resource(this, props, 0, d.as_ptr() as usize, state, clear, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        crate::xlog_warn!("D3D12.X: display target: {hr:#x}; desc {x:02x?}");
        return hr;
    }
    let size = crate::hle::xg::Texture::new(&x).map_or(0, |t| t.size()) as usize;
    let at = if size > 0 { crate::hle::kernelx::runtime_alloc(size, crate::hle::kernelx::D3D_DISPLAY_TARGET) } else { None };
    if let Some(p) = at {
        note_title_address(*ppv as usize, p);
    }
    crate::xlog_info!(
        "D3D12.X: display target {}x{} format {}: {size:#x} bytes of the title's memory at {:#x}",
        u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()),
        u32::from_le_bytes(x[0x18..0x1C].try_into().unwrap()),
        u32::from_le_bytes(x[0x20..0x24].try_into().unwrap()),
        at.unwrap_or(0)
    );
    hr
}

unsafe extern "system" fn dev_create_committed_resource_x(this: *mut c_void, props: usize, flags: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static SEEN: AtomicU64 = AtomicU64::new(0);
    if desc != 0 && std::env::var_os("XCL_D3D12_TRACE").is_some() && SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
        let w = |o: usize| *((desc + o) as *const u32);
        crate::xlog_info!(
            "CreateCommittedResource (console, slot 76): dimension {} {}x{} format {} flags {:#x}; console fields after the PC desc: +52 {:#x} +56 {:#x}",
            w(0),
            *((desc + 16) as *const u64),
            w(24),
            w(32),
            w(48),
            w(52),
            w(56)
        );
        crate::xlog_info!("  desc (dwords): {:x?}", (0..16).map(|k| w(4 * k)).collect::<Vec<_>>());
    }
    // console texture layouts (seen: 0x101, with 3 at +52) are not PC ones: the driver picks
    // its own (UNKNOWN); the console layout matters only where console code reads the memory
    let mut pc_desc = [0u8; 56];
    let console_desc = desc;
    let mut desc = desc;
    if desc != 0 {
        std::ptr::copy_nonoverlapping(desc as *const u8, pc_desc.as_mut_ptr(), 56);
        let layout = u32::from_le_bytes(pc_desc[44..48].try_into().unwrap());
        if layout > 3 {
            pc_desc[44..48].copy_from_slice(&0u32.to_le_bytes());
            pc_desc[52..56].copy_from_slice(&0u32.to_le_bytes());
            desc = pc_desc.as_ptr() as usize;
        }
    }
    if desc != 0 {
        if let Some(hr) = place_console_buffer(this, props, desc, state, clear, riid, ppv) {
            return hr;
        }
        if let Some(hr) = place_console_texture(this, console_desc, state, clear, riid, ppv) {
            return hr;
        }
    }
    // not placed in unified memory (e.g. its coherent window kept for readback heaps): a PC
    // resource of the PC's description – the console's flag bits (0x1000000, 0x300000 …) and
    // alignments made the PC refuse Fortnite's 3264x1836 scene targets (E_INVALIDARG, a crash)
    let pc_full = (console_desc != 0 && crate::gpu::shader::readable(console_desc, 0x38))
        .then(|| super::device_x::pc_resource_desc_for_device(this, &(console_desc as *const [u8; 0x38]).read_unaligned()));
    let desc = pc_full.as_ref().map_or(desc, |d| d.as_ptr() as usize);
    let mut clear_buf = [0u8; 24];
    let clear = match pc_full.as_ref() {
        Some(d) => super::device_x::pc_clear_for_desc(clear as *const c_void, u32::from_le_bytes(d[0x20..0x24].try_into().unwrap()), &mut clear_buf),
        None => super::device_x::pc_clear(clear as *const c_void, &mut clear_buf),
    } as usize;
    // console-only state bits (e.g. 0x100000 on a display target) are not the PC's; a resource
    // the PC description took ALLOW_UNORDERED_ACCESS from does not start as a UAV
    let st = pc_full.as_ref().map_or(state as u32 & PC_RESOURCE_STATES, |d| {
        let orig_flags = u32::from_le_bytes((console_desc as *const [u8; 0x38]).read_unaligned()[0x30..0x34].try_into().unwrap());
        let pc_flags = u32::from_le_bytes(d[0x30..0x34].try_into().unwrap());
        super::device_x::pc_initial_state_for_desc(orig_flags, pc_flags, state as u32)
    });
    let hr = dev_create_committed_resource(this, props, flags, desc, st as usize, clear, riid, ppv);
    if hr < 0 {
        let pw = |o: usize| if props == 0 { 0 } else { *((props + o) as *const u32) };
        crate::xlog_warn!(
            "CreateCommittedResource (console): {hr:#x}; heap type {} cpu {} pool {}, heap flags {flags:#x}, state {state:#x}",
            pw(0),
            pw(4),
            pw(8)
        );
    }
    hr
}

pub(super) unsafe extern "system" fn dev_create_command_list_x(this: *mut c_void, mask: usize, ty: usize, alloc: usize, pso: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let (pc_pso, generation) = super::pso_async::resolve_and_pin_if_enabled_identity(super::console_list::pc_pso(pso as *mut c_void) as usize);
    let hr = dev_create_command_list(this, mask, ty, alloc, pc_pso, riid, ppv);
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        super::list_x::on_console_list_create(*ppv, alloc, pc_pso, D3D12_COMMAND_LIST_TYPE(ty as i32));
        super::list_x::record_native_generation(*ppv as usize, generation);
        *ppv = super::console_list::create_with_allocator(*ppv, alloc as *mut c_void);
    }
    hr
}

unsafe extern "system" fn dev_create_compute_pso_x(this: *mut c_void, desc: *const c_void, a: usize, b: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if a != 0 || b != 0 {
        crate::xlog_warn!("CreateComputePipelineState (console, slot 62): unknown parameters {a:#x} {b:#x} ignored");
    }
    let hr = dev_create_compute_pso(this, desc, riid, ppv);
    // console code writes into pipeline objects (D3D12.X inline code): give it a console-shaped one
    if hr >= 0 && super::console_list::enabled() && !ppv.is_null() && !(*ppv).is_null() {
        *ppv = super::console_list::wrap_pso(*ppv, LAST_COMPUTE_TRANSLATED.with(|c| c.get()), LAST_COMPUTE_RS.with(|c| c.replace(0)));
    }
    hr
}

unsafe extern "system" fn dev_make_resident(this: *mut c_void, n: u32, objects: *const *mut c_void) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "MakeResident")));
    let _stall = crate::stall::span(crate::stall::StallKind::Residency);
    f(this, n, objects)
}

unsafe extern "system" fn dev_evict(this: *mut c_void, n: u32, objects: *const *mut c_void) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "Evict")));
    let _stall = crate::stall::span(crate::stall::StallKind::Residency);
    f(this, n, objects)
}

unsafe extern "system" fn dev_set_residency_priority(this: *mut c_void, n: u32, objects: *const *mut c_void, priorities: *const D3D12_RESIDENCY_PRIORITY) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *const D3D12_RESIDENCY_PRIORITY) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "SetResidencyPriority")));
    let _stall = crate::stall::span(crate::stall::StallKind::Residency);
    f(this, n, objects, priorities)
}

unsafe extern "system" fn dev_enqueue_make_resident(
    this: *mut c_void,
    flags: D3D12_RESIDENCY_FLAGS,
    n: u32,
    objects: *const *mut c_void,
    fence: *mut c_void,
    value: u64,
) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void, D3D12_RESIDENCY_FLAGS, u32, *const *mut c_void, *mut c_void, u64) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "EnqueueMakeResident")));
    let _stall = crate::stall::span(crate::stall::StallKind::Residency);
    f(this, flags, n, objects, fence, value)
}

fn overrides(kind: Kind) -> Vec<(&'static str, usize)> {
    let mut v: Vec<(&'static str, usize)> = vec![("QueryInterface", qi_override as usize)];
    match kind {
        Kind::Device => v.extend_from_slice(&[
            ("CreateCommandQueue", dev_create_command_queue as usize),
            ("CheckFeatureSupport", dev_check_feature_support as usize),
            ("CreateCommandAllocator", dev_create_command_allocator as usize),
            ("CreateGraphicsPipelineState", dev_create_graphics_pso as usize),
            ("CreateComputePipelineState", dev_create_compute_pso as usize),
            ("CreateCommandList", dev_create_command_list as usize),
            ("CreateDescriptorHeap", dev_create_descriptor_heap as usize),
            ("CreateRootSignature", dev_create_root_signature as usize),
            ("CreateRenderTargetView", dev_create_rtv as usize),
            ("CreateCommittedResource", dev_create_committed_resource as usize),
            ("CreateHeap", dev_create_heap as usize),
            ("CreatePlacedResource", dev_create_placed_resource as usize),
            ("CreateReservedResource", dev_create_reserved_resource as usize),
            ("MakeResident", dev_make_resident as usize),
            ("Evict", dev_evict as usize),
            ("CreateFence", dev_create_fence as usize),
            ("CreateQueryHeap", dev_create_query_heap as usize),
            ("CreateCommandSignature", dev_create_command_signature as usize),
            ("CreatePipelineLibrary", dev_create_pipeline_library as usize),
            ("SetResidencyPriority", dev_set_residency_priority as usize),
            ("CreatePipelineState", dev_create_pipeline_state as usize),
            ("EnqueueMakeResident", dev_enqueue_make_resident as usize),
            ("CreateCommandList1", dev_create_command_list1 as usize),
            ("CreateCommittedResource1", dev_create_committed_resource1 as usize),
            ("CreateHeap1", dev_create_heap1 as usize),
            ("CreateReservedResource1", dev_create_reserved_resource1 as usize),
            ("CreateStateObject", dev_create_state_object as usize),
            ("CreateCommittedResource2", dev_create_committed_resource2 as usize),
            ("CreatePlacedResource1", dev_create_placed_resource1 as usize),
            ("CreateShaderCacheSession", dev_create_shader_cache_session as usize),
            ("CreateCommandQueue1", dev_create_command_queue1 as usize),
        ]),
        Kind::CommandList => v.extend_from_slice(&[
            ("Release", cl_release as usize),
            ("Close", cl_close as usize),
            ("Reset", cl_reset as usize),
            ("ClearState", cl_clear_state as usize),
            ("DrawInstanced", cl_draw_instanced as usize),
            ("DrawIndexedInstanced", cl_draw_indexed_instanced as usize),
            ("SetPipelineState", cl_set_pipeline_state as usize),
            ("ExecuteBundle", cl_execute_bundle as usize),
            ("SetMarker", cl_set_marker as usize),
            ("OMSetRenderTargets", cl_om_set_render_targets as usize),
            ("ClearRenderTargetView", cl_clear_rtv as usize),
            ("ExecuteIndirect", cl_execute_indirect as usize),
        ]),
        Kind::DmaList => v.extend_from_slice(&[
            ("Release", cl_release as usize),
            ("Close", cl_close as usize),
            ("Reset", cl_reset as usize),
        ]),
        Kind::CommandQueue => v.extend_from_slice(&[("Release", q_release as usize), ("ExecuteCommandLists", q_execute_command_lists as usize)]),
        Kind::CommandAllocator => v.extend_from_slice(&[("Release", alloc_release as usize), ("Reset", alloc_reset as usize)]),
        Kind::Resource => v.extend_from_slice(&[
            ("Release", res_release as usize),
            ("Map", res_map as usize),
            ("Unmap", res_unmap as usize),
            ("WriteToSubresource", res_write_to_subresource as usize),
        ]),
        Kind::PipelineState => v.push(("Release", pso_release as usize)),
        Kind::DescriptorHeap => v.push(("Release", heap_release as usize)),
        Kind::Heap => v.push(("Release", mem_heap_release as usize)),
        Kind::Fence => v.extend_from_slice(&[("GetCompletedValue", fence_completed as usize), ("Signal", fence_signal as usize)]),
        _ => {}
    }
    v
}

/// Slot of `method` in the class of an adopted object.
unsafe fn slot_of(this: *mut c_void, method: &str) -> usize {
    let ext = vt(this) as usize;
    let g = CLASSES.lock().unwrap();
    let c = g.iter().find(|c| c.ext == ext).expect("adopted class");
    layout::slot(c.iface, method).expect("method in layout")
}

unsafe extern "system" fn qi_override(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let f: QiFn = std::mem::transmute(orig_fn(this, 0));
    let hr = f(this, riid, ppv);
    if hr == E_NOINTERFACE && !riid.is_null() {
        let s = layout::guid_string(&*riid);
        if UNKNOWN_IIDS.lock().unwrap().insert(s.clone()) {
            crate::xlog_warn!("D3D12.X: QueryInterface({s}) – interface unknown on the PC (console-only?)");
        }
    }
    hr
}

/// device #0 QueryInterface for console code: the console device is also an IDXGIDevice
/// (`dxgi_device`). Only the title gets it – DXGI asks a D3D12 device the same while creating
/// a swap chain and must keep getting the PC answer.
unsafe extern "system" fn dev_qi_x(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if !riid.is_null() && !ppv.is_null() && super::dxgi_device::answers(&std::mem::transmute::<GUID, windows::core::GUID>(*riid)) {
        *ppv = super::dxgi_device::create(this);
        return S_OK;
    }
    qi_override(this, riid, ppv)
}

fn pc_iid(kind: Kind) -> GUID {
    layout::parse_guid(layout::find(kind.base()).map(|i| i.iid).unwrap_or(""))
}

macro_rules! as_usize {
    ($x:ident) => {
        usize
    };
}

/// Close: a failure is logged with what was recorded (titles stop on it; the PC runtime
/// refuses some calls only here). Newer D3D12 runtimes (e.g. Agility SDK 1.620+) swap the
/// command list's vtable pointer on Close; adopt the runtime's new state vtable so
/// subsequent Reset and console calls dispatch through the current native table.
unsafe extern "system" fn cl_close(this: *mut c_void) -> HRESULT {
    let _detail = crate::stall::detail_span_if(!super::title_textures::is_own_work(), crate::stall::StallKind::DetailListLifecycle);
    let poisoned = super::list_x::recording_poisoned(this as usize);
    super::list_x::before_close(this);
    let ext = *(this as *const usize);
    let f: RefFn = std::mem::transmute(orig_fn(this, slot!(this, "Close")));
    let hr = f(this) as HRESULT;
    preserve_vtable(this, ext);
    if hr < 0 {
        let ty: RefFn = std::mem::transmute(orig_fn(this, slot_of(this, "GetType")));
        crate::xlog_warn!("D3D12.X: command list {this:p} (type {}) Close → {:#x}", ty(this), hr as u32);
    }
    if poisoned {
        crate::xlog_warn!("D3D12.X: command list {this:p} poisoned by an unavailable pipeline bind");
        crate::util::E_FAIL
    } else {
        hr
    }
}

#[cfg(test)]
pub(super) unsafe fn close_for_test(list: *mut c_void) -> HRESULT { cl_close(list) }

#[cfg(test)]
pub(super) fn submission_rejected_for_test(lists: &[*mut c_void]) -> bool {
    super::list_x::any_recording_poisoned(lists)
}

/// Reset (PC path): re-adopt the runtime's new open-state vtable if Reset replaced it.
unsafe extern "system" fn cl_reset(this: *mut c_void, allocator: *mut c_void, initial_state: *mut c_void) -> HRESULT {
    let _detail = crate::stall::detail_span_if(!super::title_textures::is_own_work(), crate::stall::StallKind::DetailListLifecycle);
    let ext = *(this as *const usize);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> HRESULT =
        std::mem::transmute(orig_fn(this, slot!(this, "Reset")));
    let prepared = super::bind::PreparedNativePipeline::new(initial_state as usize);
    let resolved = prepared.raw();
    let hr = f(this, allocator, resolved as *mut c_void);
    preserve_vtable(this, ext);
    if hr >= 0 {
        super::list_x::on_pc_reset(this);
        prepared.record_generation(this as usize);
        super::dred::reset(this);
        if !initial_state.is_null() && resolved != 0 {
            super::dred::mark_pipeline(this, initial_state as usize, resolved);
        }
    }
    hr
}

#[cfg(test)]
pub(super) unsafe fn reset_for_test(
    list: *mut c_void, allocator: *mut c_void, initial_state: *mut c_void,
) -> HRESULT {
    cl_reset(list, allocator, initial_state)
}

/// The PC size of a CheckFeatureSupport structure.
fn pc_feature_size(feature: i32) -> Option<usize> {
    use std::mem::size_of;
    Some(match D3D12_FEATURE(feature) {
        D3D12_FEATURE_D3D12_OPTIONS => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>(),
        D3D12_FEATURE_ARCHITECTURE => size_of::<D3D12_FEATURE_DATA_ARCHITECTURE>(),
        D3D12_FEATURE_FEATURE_LEVELS => size_of::<D3D12_FEATURE_DATA_FEATURE_LEVELS>(),
        D3D12_FEATURE_FORMAT_SUPPORT => size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>(),
        D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS => size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>(),
        D3D12_FEATURE_FORMAT_INFO => size_of::<D3D12_FEATURE_DATA_FORMAT_INFO>(),
        D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT => size_of::<D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT>(),
        D3D12_FEATURE_SHADER_MODEL => size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>(),
        D3D12_FEATURE_D3D12_OPTIONS1 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>(),
        D3D12_FEATURE_ROOT_SIGNATURE => size_of::<D3D12_FEATURE_DATA_ROOT_SIGNATURE>(),
        D3D12_FEATURE_ARCHITECTURE1 => size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>(),
        D3D12_FEATURE_D3D12_OPTIONS2 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS2>(),
        D3D12_FEATURE_SHADER_CACHE => size_of::<D3D12_FEATURE_DATA_SHADER_CACHE>(),
        D3D12_FEATURE_D3D12_OPTIONS3 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS3>(),
        D3D12_FEATURE_EXISTING_HEAPS => size_of::<D3D12_FEATURE_DATA_EXISTING_HEAPS>(),
        D3D12_FEATURE_D3D12_OPTIONS4 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS4>(),
        D3D12_FEATURE_D3D12_OPTIONS5 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>(),
        D3D12_FEATURE_D3D12_OPTIONS6 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS6>(),
        D3D12_FEATURE_D3D12_OPTIONS7 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>(),
        D3D12_FEATURE_D3D12_OPTIONS8 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS8>(),
        D3D12_FEATURE_D3D12_OPTIONS9 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS9>(),
        D3D12_FEATURE_D3D12_OPTIONS10 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS10>(),
        D3D12_FEATURE_D3D12_OPTIONS11 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS11>(),
        D3D12_FEATURE_D3D12_OPTIONS12 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>(),
        D3D12_FEATURE_D3D12_OPTIONS13 => size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS13>(),
        _ => return None,
    })
}

/// Apply `config.raytracing` / `XCL_RAYTRACING` override (if any) to a successful
/// `CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5)` result and log the reported tier once.
unsafe fn adjust_raytracing_options5(feature: i32, data: *mut u8, size: u32) {
    if D3D12_FEATURE(feature) != D3D12_FEATURE_D3D12_OPTIONS5 || data.is_null() || (size as usize) < std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() {
        return;
    }
    let opts = &mut *(data as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS5);
    let host = opts.RaytracingTier.0;
    if let Some(tier) = crate::state::config().raytracing_tier_override() {
        opts.RaytracingTier = D3D12_RAYTRACING_TIER(tier as i32);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: CheckFeatureSupport(D3D12_OPTIONS5) → RaytracingTier {} (host {})", opts.RaytracingTier.0, host);
    }
}

/// CheckFeatureSupport(feature, data, size): console structures can be larger than the PC's
/// (XDefiant: D3D12_OPTIONS with 0x40 bytes, 0x3C on the PC – and the title stops on the
/// error). A size the PC refuses is retried with the PC's, the console's extra bytes zero.
///
/// A query about one of the console's own formats (116 – 120, `device_x::pc_view_format`) asks
/// the PC about the format the resource gets there, and the caller's format field is put back.
/// A retail title asked FORMAT_SUPPORT of D16_UNORM_S8_UINT (118), the PC called it unsupported,
/// and the title stopped on its assertion.
unsafe extern "system" fn dev_check_feature_support(this: *mut c_void, feature: i32, data: *mut u8, size: u32) -> HRESULT {
    let f: CheckFeatureSupportFn = std::mem::transmute(orig_fn(this, slot_of(this, "CheckFeatureSupport")));
    check_feature_support_x(f, this, feature, data, size)
}

type CheckFeatureSupportFn = unsafe extern "system" fn(*mut c_void, i32, *mut u8, u32) -> HRESULT;

/// `dev_check_feature_support` with the PC device's own CheckFeatureSupport `f`.
pub(super) unsafe fn check_feature_support_x(f: CheckFeatureSupportFn, this: *mut c_void, feature: i32, data: *mut u8, size: u32) -> HRESULT {
    let console = console_format_query(feature, data, size);
    let hr = check_feature_support_pc(f, this, feature, data, size);
    if let Some(format) = console {
        (data as *mut u32).write_unaligned(format);
    }
    hr
}

/// The features whose data starts with a DXGI_FORMAT the caller asks about.
fn feature_takes_format(feature: i32) -> bool {
    matches!(
        D3D12_FEATURE(feature),
        D3D12_FEATURE_FORMAT_SUPPORT | D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS | D3D12_FEATURE_FORMAT_INFO | D3D12_FEATURE_PLACED_RESOURCE_SUPPORT_INFO
    )
}

/// For a format query (`feature_takes_format`) about a console-only format: its PC format written
/// into `data` in its place, and the console's returned (to be put back after the call).
unsafe fn console_format_query(feature: i32, data: *mut u8, size: u32) -> Option<u32> {
    if !feature_takes_format(feature) || data.is_null() || size < 4 {
        return None;
    }
    let format = (data as *const u32).read_unaligned();
    let pc = super::device_x::pc_view_format(format);
    (pc != format).then(|| {
        (data as *mut u32).write_unaligned(pc);
        format
    })
}

unsafe fn check_feature_support_pc(f: CheckFeatureSupportFn, this: *mut c_void, feature: i32, data: *mut u8, size: u32) -> HRESULT {
    let hr = f(this, feature, data, size);
    if hr >= 0 {
        adjust_raytracing_options5(feature, data, size);
        return hr;
    }
    let Some(pc) = pc_feature_size(feature).filter(|&pc| !data.is_null() && pc != size as usize) else { return hr };
    let n = pc.min(size as usize);
    let mut tmp = vec![0u8; pc];
    std::ptr::copy_nonoverlapping(data, tmp.as_mut_ptr(), n); // input fields (format, sample count …)
    let hr2 = f(this, feature, tmp.as_mut_ptr(), pc as u32);
    if hr2 >= 0 {
        std::ptr::write_bytes(data, 0, size as usize);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), data, n);
        adjust_raytracing_options5(feature, data, size);
    }
    static SEEN: Mutex<Vec<i32>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if !seen.contains(&feature) {
        seen.push(feature);
        crate::xlog_info!("D3D12.X: CheckFeatureSupport({feature}) with the console's {size} bytes, the PC's {pc} → {:#x}", hr2 as u32);
    }
    hr2
}

thread_local! {
    /// A `creator!` method is inside the runtime's own call: what it creates through the device
    /// (D3D12Core's DRED breadcrumb buffers, made with the debug layer on) is the runtime's, not
    /// the title's
    static IN_CREATOR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Device creation methods ending in (REFIID riid, void** ppv).
macro_rules! creator {
    ($name:ident, $method:literal, $kind:expr, [$($a:ident),*]) => {
        pub(super) unsafe extern "system" fn $name(this: *mut c_void, $($a: usize,)* riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
            type F = unsafe extern "system" fn(*mut c_void, $(as_usize!($a),)* *const GUID, *mut *mut c_void) -> HRESULT;
            let f: F = std::mem::transmute(orig_fn(this, slot_of(this, $method)));
            // the runtime calls the device's public methods from inside another one (a command
            // list's DRED breadcrumb buffer, made through CreatePlacedResource): that object
            // is still being made – asking it for interfaces to adopt it crashes D3D12Core
            if IN_CREATOR.with(|c| c.replace(true)) {
                return f(this, $($a,)* riid, ppv);
            }
            let mut hr = f(this, $($a,)* riid, ppv);
            IN_CREATOR.with(|c| c.set(false));
            if hr == E_NOINTERFACE {
                let iid = pc_iid($kind);
                hr = f(this, $($a,)* &iid, ppv);
                // an interface the PC does not have (a console one): which, once per method
                static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
                let asked = riid.as_ref().map(super::layout::guid_string).unwrap_or_default();
                let new = { let mut g = SEEN.lock().unwrap(); let n = !g.contains(&asked); if n { g.push(asked.clone()); } n };
                if new {
                    crate::xlog_info!("D3D12.X: {} for interface {asked} (not on the PC) – made as {}", $method, super::layout::guid_string(&iid));
                }
            }
            if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
                adopt(*ppv, $kind);
                if $kind == Kind::CommandQueue {
                    super::queue_submit::created(*ppv as usize);
                }
            }
            hr
        }
    };
}

creator!(dev_create_command_queue, "CreateCommandQueue", Kind::CommandQueue, [a1]);
creator!(dev_create_command_queue1, "CreateCommandQueue1", Kind::CommandQueue, [a1, a2]);
creator!(dev_create_command_allocator, "CreateCommandAllocator", Kind::CommandAllocator, [a1]);
creator!(dev_create_command_list, "CreateCommandList", Kind::CommandList, [a1, a2, a3, a4]);
creator!(dev_create_command_list1, "CreateCommandList1", Kind::CommandList, [a1, a2, a3]);

/// Create a raw adopted PC `ID3D12GraphicsCommandList` without wrapping it in `title_list`
/// (used for deferred mid-list split segments on the same allocator).
pub(super) unsafe fn create_raw_pc_command_list(
    dev: *mut c_void,
    ty: D3D12_COMMAND_LIST_TYPE,
    alloc: usize,
    init_pso: usize,
) -> Option<*mut c_void> {
    if dev.is_null() || alloc == 0 {
        return None;
    }
    let iid = layout::parse_guid(layout::find("ID3D12GraphicsCommandList")?.iid);
    let mut out: *mut c_void = std::ptr::null_mut();
    let hr = if is_ext(vt(dev) as usize) {
        dev_create_command_list(dev, 0, ty.0 as usize, alloc, init_pso, &iid, &mut out)
    } else {
        type F = unsafe extern "system" fn(*mut c_void, u32, D3D12_COMMAND_LIST_TYPE, usize, usize, *const GUID, *mut *mut c_void) -> HRESULT;
        let slot = layout::slot(layout::find("ID3D12Device")?, "CreateCommandList")?;
        let f: F = std::mem::transmute(*vt(dev).add(slot));
        let hr = f(dev, 0, ty, alloc, init_pso, &iid, &mut out);
        if hr >= 0 && !out.is_null() {
            adopt(out, Kind::CommandList);
        }
        hr
    };
    (hr >= 0 && !out.is_null()).then_some(out)
}

/// Create a raw adopted PC command list already closed and without an allocator. Split
/// preparation can reserve this while the producer still records; it is Reset onto the title's
/// allocator only after that producer has closed.
pub(super) unsafe fn create_closed_pc_command_list(dev: *mut c_void, ty: D3D12_COMMAND_LIST_TYPE) -> Option<*mut c_void> {
    if dev.is_null() {
        return None;
    }
    let iid = layout::parse_guid(layout::find("ID3D12GraphicsCommandList")?.iid);
    let mut out = std::ptr::null_mut();
    let hr = if is_ext(vt(dev) as usize) {
        dev_create_command_list1(dev, 0, ty.0 as usize, D3D12_COMMAND_LIST_FLAG_NONE.0 as usize, &iid, &mut out)
    } else {
        let dev4 = ID3D12Device::from_raw_borrowed(&dev)?.cast::<ID3D12Device4>().ok()?;
        match dev4.CreateCommandList1::<ID3D12GraphicsCommandList>(0, ty, D3D12_COMMAND_LIST_FLAG_NONE) {
            Ok(list) => {
                out = list.into_raw();
                adopt(out, Kind::CommandList);
                0
            }
            Err(e) => e.code().0,
        }
    };
    (hr >= 0 && !out.is_null()).then_some(out)
}
creator!(dev_create_descriptor_heap, "CreateDescriptorHeap", Kind::DescriptorHeap, [a1]);
creator!(dev_create_committed_resource, "CreateCommittedResource", Kind::Resource, [a1, a2, a3, a4, a5]);
creator!(dev_create_committed_resource1, "CreateCommittedResource1", Kind::Resource, [a1, a2, a3, a4, a5, a6]);
creator!(dev_create_committed_resource2, "CreateCommittedResource2", Kind::Resource, [a1, a2, a3, a4, a5, a6]);
creator!(dev_create_placed_resource, "CreatePlacedResource", Kind::Resource, [a1, a2, a3, a4, a5]);

/// D3D12_RESOURCE_STATES the PC knows (as `list_x::PC_STATES`)
pub(super) const PC_RESOURCE_STATES: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0xF_0000 | 0x20_0000 | 0x40_0000 | 0x80_0000 | 0x100_0000;

/// device #29 CreatePlacedResource(heap, offset, desc, state, clear, riid, resource) for console
/// code: a description the PC refuses (console layouts and flags – Fortnite's buffer in a
/// buffers-only heap) is made the PC's way (`device_x::pc_resource_desc`, the console's state
/// bits dropped). Logged with the console's values.
pub(super) unsafe extern "system" fn dev_create_placed_resource_x(this: *mut c_void, heap: usize, offset: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    // the clear value in the PC's format of the resource (D3D12 wants the two to match)
    let mut clear_buf = [0u8; 24];
    let clear = super::device_x::pc_clear(clear as *const c_void, &mut clear_buf) as usize;
    // a buffer in a heap that has unified memory (`note_heap_uma`): that memory at its offset
    if desc != 0 && crate::gpu::shader::readable(desc, 0x38) {
        let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
        let is_buffer = u32::from_le_bytes(x[0..4].try_into().unwrap()) == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32;
        if let Some((block, cpu_readback)) = is_buffer.then(|| heap_block(heap)).flatten() {
            let d = super::device_x::pc_resource_desc(&x);
            let st = state as u32 & PC_RESOURCE_STATES;
            if let Some(hr) = place_at_title_address(this, block + offset, &d, st, clear as *const c_void, riid, ppv) {
                if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() && cpu_readback && crate::gpu::uma::get().is_some_and(|u| u.is_sparse(block + offset)) {
                    let flags = u32::from_le_bytes(d[0x30..0x34].try_into().unwrap());
                    if st == D3D12_RESOURCE_STATE_COPY_DEST.0 as u32 && (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32) == 0 {
                        note_sparse_readback(*ppv as usize);
                    }
                }
                return hr;
            }
        }
    }
    if let Some(hr) = own_memory_if_larger_on_the_pc(this, heap, offset, desc, state, clear, riid, ppv) {
        return hr;
    }
    let hr = dev_create_placed_resource(this, heap, offset, desc, state, clear, riid, ppv);
    if hr >= 0 || desc == 0 || !crate::gpu::shader::readable(desc, 0x38) {
        return hr;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let d = super::device_x::pc_resource_desc(&x);
    let st = super::device_x::pc_initial_state(u32::from_le_bytes(x[0x30..0x34].try_into().unwrap()), state as u32);
    let mut clear_buf2 = [0u8; 24];
    let clear_d = super::device_x::pc_clear_for_desc(clear as *const c_void, u32::from_le_bytes(d[0x20..0x24].try_into().unwrap()), &mut clear_buf2) as usize;
    let hr2 = dev_create_placed_resource(this, heap, offset, d.as_ptr() as usize, st as usize, clear_d, riid, ppv);
    let heap_desc = pc_heap_desc(heap);
    let pc_size = pc_allocation_size(this, &d);
    // A texture the title placed by the console's size (`get_resource_allocation_info_x`
    // answers with XG's) can be larger on the PC and run past the end of its heap, which the
    // PC refuses – Minecraft then ended (0xC0000409). It gets memory of its own, of the
    // heap's kind; the console's aliasing of that one placement is lost.
    let is_texture = u32::from_le_bytes(x[0..4].try_into().unwrap()) != D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32;
    let hr3 = match heap_desc {
        Some(h) if hr2 < 0 && is_texture => Some(dev_create_committed_resource(this, &h.Properties as *const _ as usize, 0, d.as_ptr() as usize, st as usize, clear_d, riid, ppv)),
        _ => None,
    };
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
        let sizes = format!(
            "heap {:#x} bytes, PC size {:#x}{}",
            heap_desc.map_or(0, |h| h.SizeInBytes),
            pc_size,
            if heap_desc.is_some_and(|h| offset as u64 + pc_size > h.SizeInBytes) { " – past the heap's end" } else { "" }
        );
        let own = hr3.map_or(String::new(), |h| format!("; memory of its own: {h:#x}"));
        crate::xlog_info!(
            "D3D12.X: CreatePlacedResource(dimension {}, {}x{}, format {}, layout {:#x}, flags {:#x}, state {state:#x}) at {offset:#x} refused ({hr:#x}); the PC's way: {hr2:#x} ({sizes}){own}",
            u(0), u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x2C), u(0x30)
        );
    }
    hr3.unwrap_or(hr2)
}

/// A texture the title placed in a heap by the console's size (`get_resource_allocation_info_x`
/// answers with XG's), when the PC's is larger: placed at the console's offset it would cover
/// the start of whatever the title placed after it, and each would write over the other.
/// Minecraft's logo (1937x333, 0x2A0000 bytes on the console, 0x2E0000 on the PC) ran into the
/// text and 16x16 textures packed after it – magenta / black squares and a white speck along
/// its bottom edge. Such a texture gets memory of its own, of the heap's kind; the console's
/// aliasing of that one placement is lost. None: placed as the title asked.
#[allow(clippy::too_many_arguments)]
unsafe fn own_memory_if_larger_on_the_pc(this: *mut c_void, heap: usize, offset: usize, desc: usize, state: usize, clear: usize, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    if desc == 0 || !crate::gpu::shader::readable(desc, 0x38) {
        return None;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    if u32::from_le_bytes(x[0..4].try_into().unwrap()) == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32 {
        return None;
    }
    let console = crate::hle::xg::Texture::new(&x).ok()?.size();
    let d = super::device_x::pc_resource_desc(&x);
    let pc = pc_allocation_size(this, &d);
    if console == u64::MAX || pc == u64::MAX || pc <= console {
        return None;
    }
    let h = pc_heap_desc(heap)?;
    let st = super::device_x::pc_initial_state(u32::from_le_bytes(x[0x30..0x34].try_into().unwrap()), state as u32);
    let mut clear_buf = [0u8; 24];
    let clear = super::device_x::pc_clear_for_desc(clear as *const c_void, u32::from_le_bytes(d[0x20..0x24].try_into().unwrap()), &mut clear_buf) as usize;
    let mut hr = dev_create_committed_resource(this, &h.Properties as *const _ as usize, 0, d.as_ptr() as usize, st as usize, clear, riid, ppv);
    if hr < 0 && st != 0 {
        // as in the past-the-end fallback: a state the PC refuses for this desc (e.g. UAV
        // without the UAV flag) – COMMON rather than falling back to the overlapping placement
        hr = dev_create_committed_resource(this, &h.Properties as *const _ as usize, 0, d.as_ptr() as usize, 0, clear, riid, ppv);
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
        crate::xlog_info!(
            "D3D12.X: CreatePlacedResource({}x{}, format {}, flags {:#x}) at {offset:#x}: {pc:#x} bytes on the PC, {console:#x} on the console – memory of its own ({hr:#x})",
            u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x30)
        );
    }
    (hr >= 0).then_some(hr)
}

/// `XCL_DEBUG_NO_STENCIL` / `XCL_DEBUG_NO_CULL` `=hash,…` (pixel shader dump hashes, hex):
/// pipelines with those pixel shaders draw without their stencil test / culling – diagnostics
/// for draws the fixed-function stages reject.
fn debug_pso_switch(var: &str, ps: u64) -> bool {
    ps != 0 && std::env::var(var).is_ok_and(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).any(|h| h == ps))
}

/// The PC description of a heap XCL adopted (ID3D12Heap::GetDesc, slot 8).
unsafe fn pc_heap_desc(heap: usize) -> Option<D3D12_HEAP_DESC> {
    if !is_adopted(heap as *mut c_void) {
        return None;
    }
    type F = unsafe extern "system" fn(*mut c_void, *mut D3D12_HEAP_DESC) -> *mut D3D12_HEAP_DESC;
    let f: F = std::mem::transmute(orig_fn(heap as *mut c_void, 8));
    let mut h = D3D12_HEAP_DESC::default();
    f(heap as *mut c_void, &mut h);
    Some(h)
}

/// The PC driver's size for a (PC) resource description (GetResourceAllocationInfo, slot 25).
unsafe fn pc_allocation_size(device: *mut c_void, d: &[u8; 0x38]) -> u64 {
    type F = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
    let f: F = std::mem::transmute(orig_fn(device, 25));
    let safe = super::device_x::strip_unsupported_uav_for_device(device, *d);
    let mut r = [0u64; 2];
    f(device, &mut r, 0, 1, &safe);
    r[0]
}
creator!(dev_create_placed_resource1, "CreatePlacedResource1", Kind::Resource, [a1, a2, a3, a4, a5]);
creator!(pc_create_reserved_resource, "CreateReservedResource", Kind::Resource, [a1, a2, a3]);
creator!(pc_create_reserved_resource1, "CreateReservedResource1", Kind::Resource, [a1, a2, a3, a4]);

/// Resources made with CreateReservedResource(1), with their description: they have no heap, so
/// `res_map` asks none of them (GetHeapProperties on one is an invalid call too, debug layer
/// 901). The description guards against an address a later resource reuses without the last
/// Release having been seen.
static RESERVED: Mutex<Option<HashMap<usize, D3D12_RESOURCE_DESC>>> = Mutex::new(None);

unsafe fn desc_of(res: *mut c_void) -> Option<D3D12_RESOURCE_DESC> {
    ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc())
}

unsafe fn note_reserved(hr: HRESULT, ppv: *mut *mut c_void) -> HRESULT {
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        if let Some(d) = desc_of(*ppv) {
            RESERVED.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, d);
        }
    }
    hr
}

unsafe fn is_reserved(res: *mut c_void) -> bool {
    let kept = RESERVED.lock().unwrap().as_ref().and_then(|m| m.get(&(res as usize)).copied());
    kept.is_some_and(|k| desc_of(res) == Some(k))
}

/// CreateReservedResource
pub(super) unsafe extern "system" fn dev_create_reserved_resource(this: *mut c_void, a1: usize, a2: usize, a3: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    note_reserved(pc_create_reserved_resource(this, a1, a2, a3, riid, ppv), ppv)
}

/// CreateReservedResource1
pub(super) unsafe extern "system" fn dev_create_reserved_resource1(this: *mut c_void, a1: usize, a2: usize, a3: usize, a4: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    note_reserved(pc_create_reserved_resource1(this, a1, a2, a3, a4, riid, ppv), ppv)
}
creator!(dev_create_heap_pc, "CreateHeap", Kind::Heap, [a1]);

/// D3D12_HEAP_FLAGS the PC knows: SHARED 0x1, DENY_BUFFERS 0x4, ALLOW_DISPLAY 0x8,
/// SHARED_CROSS_ADAPTER 0x20, DENY_RT_DS_TEXTURES 0x40, DENY_NON_RT_DS_TEXTURES 0x80, then
/// 0x100 … 0x2000 (HARDWARE_PROTECTED … TOOLS_USE_MANUAL_WRITE_TRACKING); 0x2 and 0x10 are
/// the console's (Fortnite: 0xC2)
const PC_HEAP_FLAGS: u32 = 0x3FED;

/// CreateHeap(desc, riid, heap): a console description the PC refuses (console flags in the
/// high bits, alignments other than 64 KiB / 4 MiB, console heap types – Fortnite's distance
/// field atlas heap) is made the PC's way: known flags only, the alignment the PC has that
/// covers it, an unknown type as DEFAULT. Logged with the console's values.
pub(super) unsafe extern "system" fn dev_create_heap(this: *mut c_void, desc: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = dev_create_heap_pc(this, desc, riid, ppv);
    if hr >= 0 {
        note_heap_uma(desc, ppv);
    }
    if hr >= 0 || desc == 0 || !crate::gpu::shader::readable(desc, std::mem::size_of::<D3D12_HEAP_DESC>()) {
        return hr;
    }
    let mut d: D3D12_HEAP_DESC = std::ptr::read_unaligned(desc as *const D3D12_HEAP_DESC);
    let orig = (d.SizeInBytes, d.Properties.Type.0, d.Alignment, d.Flags.0);
    d.Flags = D3D12_HEAP_FLAGS(d.Flags.0 & PC_HEAP_FLAGS as i32);
    d.Alignment = if d.Alignment <= 0x1_0000 { 0 } else { D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT as u64 };
    if !(1..=5).contains(&d.Properties.Type.0) {
        d.Properties = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    }
    d.SizeInBytes = d.SizeInBytes.div_ceil(0x1_0000) * 0x1_0000;
    let hr2 = dev_create_heap_pc(this, &d as *const _ as usize, riid, ppv);
    if hr2 >= 0 {
        note_heap_uma(desc, ppv);
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_info!(
            "D3D12.X: CreateHeap({:#x} bytes, type {}, alignment {:#x}, flags {:#x}) refused ({hr:#x}); as type {}, alignment {:#x}, flags {:#x}: {hr2:#x}",
            orig.0, orig.1, orig.2, orig.3, d.Properties.Type.0, d.Alignment, d.Flags.0
        );
    }
    hr2
}
creator!(dev_create_heap1_pc, "CreateHeap1", Kind::Heap, [a1, a2]);

pub(super) unsafe extern "system" fn dev_create_heap1(this: *mut c_void, desc: usize, session: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = dev_create_heap1_pc(this, desc, session, riid, ppv);
    if hr >= 0 {
        note_heap_uma(desc, ppv);
    }
    hr
}

struct HeapUma {
    size: usize,
    align: usize,
    /// Small explicit `D3D12_HEAP_TYPE_READBACK` heap: prefers coherent Window 3 RAM.
    prefer_coherent: bool,
    /// `READBACK` or `CUSTOM + WRITE_BACK`: if backed by sparse VRAM, `COPY_DEST` buffers placed
    /// in it read back on `Map(0, NULL, ...)`.
    cpu_readback: bool,
    block: Option<usize>,
}

/// heap → its backing block (allocated lazily on the first placed buffer; sparse VRAM for
/// DEFAULT/UPLOAD/CUSTOM heaps, coherent committed window for small explicit READBACK heaps)
static HEAP_UMA: Mutex<Option<HashMap<usize, HeapUma>>> = Mutex::new(None);

/// A heap the title made that may hold buffers: a block of unified memory as large stands for
/// it. On the console every heap is memory the CPU and the GPU address alike, and console code
/// reads buffers (root descriptors, V#s) from that memory – a PC heap's buffer has a GPU
/// address only (Minecraft's constant and vertex buffers: a heap of its own; its UI drew
/// nothing – the constants read as zero). Buffers placed in the heap are placed in the block
/// (`dev_create_placed_resource_x`); textures stay in the PC heap. The block is taken when the
/// first buffer is placed (most heaps hold textures only: a block for each filled the 2 GiB of
/// unified memory, and draws then found no room for their arguments – the image flickered);
/// none when unified memory is full (the PC heap as before).
unsafe fn note_heap_uma(desc: usize, ppv: *mut *mut c_void) {
    const DENY_BUFFERS: i32 = 0x4;
    if super::title_textures::is_own_work() || desc == 0 || ppv.is_null() || (*ppv).is_null() || !crate::gpu::shader::readable(desc, std::mem::size_of::<D3D12_HEAP_DESC>()) {
        return;
    }
    let d: D3D12_HEAP_DESC = std::ptr::read_unaligned(desc as *const D3D12_HEAP_DESC);
    if d.Flags.0 & DENY_BUFFERS != 0 || d.SizeInBytes == 0 || UMA_BUFFER.lock().unwrap().is_empty() {
        return;
    }
    let size = d.SizeInBytes as usize;
    let is_rb = d.Properties.Type == D3D12_HEAP_TYPE_READBACK;
    let is_custom_wb = d.Properties.Type == D3D12_HEAP_TYPE_CUSTOM && d.Properties.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    HEAP_UMA.lock().unwrap().get_or_insert_with(HashMap::new).insert(
        *ppv as usize,
        HeapUma {
            size,
            align: (d.Alignment as usize).max(0x1_0000),
            prefer_coherent: is_rb && size <= COHERENT_READBACK_MAX,
            cpu_readback: is_rb || is_custom_wb,
            block: None,
        },
    );
}

/// The block of unified memory standing for `heap` and whether it may hold CPU-readback buffers
/// (taken now if it has none yet; None: not a heap that takes one, or unified memory is full).
fn heap_block(heap: usize) -> Option<(usize, bool)> {
    let (size, align, prefer_coherent, cpu_readback) = {
        let g = HEAP_UMA.lock().unwrap();
        let e = g.as_ref()?.get(&heap)?;
        if let Some(p) = e.block {
            return Some((p, e.cpu_readback));
        }
        (e.size, e.align, e.prefer_coherent, e.cpu_readback)
    };
    let u = crate::gpu::uma::get()?;
    let rounded = size.div_ceil(0x1_0000) * 0x1_0000;
    let use_coherent = prefer_coherent && has_coherent_headroom(u, rounded);
    let p = if !use_coherent && u.has_sparse() {
        unsafe { super::sparse::alloc(size, align) }
            .map(|(a, _)| a)
            .or_else(|| u.alloc(rounded, align).map(|a| a as usize))
    } else {
        u.alloc(rounded, align)
            .map(|a| a as usize)
            .or_else(|| u.has_sparse().then(|| unsafe { super::sparse::alloc(size, align) }.map(|(a, _)| a)).flatten())
    };
    let Some(p) = p else {
        static FULL: AtomicU64 = AtomicU64::new(0);
        if FULL.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_warn!(
                "D3D12.X: heap of {size:#x} bytes: unified memory is full ({} MB sparse, {} MB coherent used) – its buffers stay PC-only",
                u.sparse_used() >> 20,
                u.used() >> 20
            );
        }
        return None;
    };
    let mut g = HEAP_UMA.lock().unwrap();
    let Some(e) = g.as_mut().and_then(|m| m.get_mut(&heap)) else {
        drop(g);
        if u.is_sparse(p) {
            unsafe { super::sparse::free(p) };
        } else {
            u.free(p);
        }
        return None;
    };
    if let Some(existing) = e.block {
        let cr = e.cpu_readback;
        drop(g);
        if u.is_sparse(p) {
            unsafe { super::sparse::free(p) };
        } else {
            u.free(p);
        }
        return Some((existing, cr));
    }
    e.block = Some(p);
    drop(g);
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 4 {
        let kind = if u.is_sparse(p) { "sparse VRAM" } else { "coherent RAM" };
        crate::xlog_info!("D3D12.X: heap {heap:#x} ({size:#x} bytes) holds buffers: in {kind} at {p:#x}");
    }
    Some((p, cpu_readback))
}

/// ID3D12Heap::Release: its block of unified memory goes with it.
unsafe extern "system" fn mem_heap_release(this: *mut c_void) -> u32 {
    release_taking_first(
        this,
        || HEAP_UMA.lock().unwrap().as_mut().and_then(|m| m.remove(&(this as usize))).and_then(|e| e.block),
        |gone| {
            if let (Some(p), Some(u)) = (gone, crate::gpu::uma::get()) {
                if u.is_sparse(p) {
                    super::sparse::free(p);
                } else {
                    u.free(p);
                }
            }
        },
    )
}
creator!(dev_create_fence_pc, "CreateFence", Kind::Fence, [a1, a2]);

/// fences created through the device (AddRef'd: they are released at the end only)
/// (a read-write lock: `is_fence` reads it on every queue call, and `end_title` reads it with
/// the title's threads suspended – a reader that holds it then blocks no other reader)
static FENCES: std::sync::RwLock<Vec<usize>> = std::sync::RwLock::new(Vec::new());

/// CreateFence: as on the PC; the fence is remembered for `end_title`
unsafe extern "system" fn dev_create_fence(this: *mut c_void, value: usize, flags: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = dev_create_fence_pc(this, value, flags, riid, ppv);
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        let add: RefFn = std::mem::transmute(*vt(*ppv).add(1));
        add(*ppv);
        let n = {
            let mut g = FENCES.write().unwrap();
            g.push(*ppv as usize);
            g.len()
        };
        // `XCL_WATCH_FENCE=n`: writes over the n-th fence's first 8 bytes (its vtable) are logged
        if std::env::var("XCL_WATCH_FENCE").ok().and_then(|v| v.parse::<usize>().ok()) == Some(n) {
            crate::xlog_info!("D3D12.X: fence #{n} {:p} (initial value {value:#x}, flags {flags:#x}) watched", *ppv);
            crate::eh::watch_writes(*ppv as usize);
        }
    }
    hr
}

/// End the title on purpose without leaving the process stuck in the driver: every fence it
/// created is signalled to the largest value (GPU waits on values the title would have signalled
/// later complete), the queues drain, and the process ends without running DLL detach code.
pub fn end_title(code: u32) -> ! {
    unsafe {
        // a last resort: the process ends within 15 s whatever happens below (a suspended
        // thread may hold a lock the drain needs)
        let (tx, rx) = std::sync::mpsc::channel();
        let _ = std::thread::Builder::new().name("xcl-end-watchdog".into()).spawn(move || {
            let _ = tx.send(windows_sys::Win32::System::Threading::GetCurrentThreadId());
            std::thread::sleep(std::time::Duration::from_secs(15));
            crate::eh::terminate_unhooked(code);
        });
        let watchdog = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap_or(0);
        // what the NtTerminateProcess hook writes out, while no thread is suspended: the process
        // ends past the hook (a suspended thread may hold stderr, the log file or the heap)
        crate::hle::d3d12x::pso_async::flush_used_bounded(std::time::Duration::from_millis(100));
        flush_indirect_split_records(std::time::Duration::from_millis(100));
        crate::log::flush();
        // the title's threads stop first: with its fences at the maximum a running title thinks
        // the GPU is done and resets command allocators the GPU still runs (Fortnite: a device
        // hang at the end of a run)
        suspend_other_threads(watchdog);
        for &f in FENCES.read().unwrap_or_else(|e| e.into_inner()).iter() {
            let fr = f as *mut c_void;
            if let Some(f) = ID3D12Fence::from_raw_borrowed(&fr) {
                let _ = f.Signal(u64::MAX);
            }
        }
        drain_gpu();
        crate::log::flush();
        crate::eh::terminate_unhooked(code)
    }
}

/// Every thread of the process but this one and `keep` (XCL's end watchdog), suspended – twice
/// over: the title may start threads meanwhile.
unsafe fn suspend_other_threads(keep: u32) {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, SuspendThread, THREAD_SUSPEND_RESUME};
    let (pid, me) = (GetCurrentProcessId(), GetCurrentThreadId());
    let mut done: Vec<u32> = Vec::with_capacity(1024);
    for _ in 0..2 {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return;
        }
        let mut te: THREADENTRY32 = std::mem::zeroed();
        te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = Thread32First(snap, &mut te) != 0;
        while ok {
            let tid = te.th32ThreadID;
            if te.th32OwnerProcessID == pid && tid != me && tid != keep && !done.contains(&tid) {
                let h = OpenThread(THREAD_SUSPEND_RESUME, 0, tid);
                if !h.is_null() {
                    SuspendThread(h);
                    CloseHandle(h);
                }
                if done.len() < done.capacity() {
                    done.push(tid);
                }
            }
            ok = Thread32Next(snap, &mut te) != 0;
        }
        CloseHandle(snap);
    }
}
creator!(dev_create_query_heap, "CreateQueryHeap", Kind::QueryHeap, [a1]);
creator!(dev_create_command_signature, "CreateCommandSignature", Kind::CommandSignature, [a1, a2]);
creator!(dev_create_pipeline_library, "CreatePipelineLibrary", Kind::PipelineLibrary, [a1, a2]);
creator!(dev_create_state_object, "CreateStateObject", Kind::StateObject, [a1]);

unsafe extern "system" fn dev_create_shader_cache_session(
    this: *mut c_void,
    desc: *const D3D12_SHADER_CACHE_SESSION_DESC,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void, *const D3D12_SHADER_CACHE_SESSION_DESC, *const GUID, *mut *mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "CreateShaderCacheSession")));
    let hr = f(this, desc, riid, ppv);
    if hr >= 0 {
        return hr;
    }
    let fb = super::shader_cache_session::create_fallback(this, desc, riid as *const windows::core::GUID, ppv);
    if fb >= 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            crate::xlog_info!("D3D12.X: CreateShaderCacheSession failed on the PC runtime ({hr:#x}) – using an in-memory session");
        });
        fb
    } else {
        hr
    }
}

// ---------------------------------------------------------------------------------------
// pipeline states: capture + fallback
// ---------------------------------------------------------------------------------------

struct Dummy {
    vs: Vec<u8>,
    ps: Vec<u8>,
    cs: Vec<u8>,
}

/// Stand-in mesh shader and pixel shader (DXIL, as mesh pipelines need): one degenerate
/// triangle (DXC wants every output written), nothing drawn.
fn dummy_mesh() -> Option<&'static (Vec<u8>, Vec<u8>)> {
    static D: OnceLock<Option<(Vec<u8>, Vec<u8>)>> = OnceLock::new();
    D.get_or_init(|| {
        use crate::gpu::recomp::dxc::compile;
        let ms = "struct V { float4 p : SV_Position; };\n[outputtopology(\"triangle\")]\n[numthreads(1, 1, 1)]\nvoid main(out vertices V v[1], out indices uint3 t[1]) { SetMeshOutputCounts(1, 1); v[0].p = float4(0, 0, 0, 1); t[0] = uint3(0, 0, 0); }\n";
        let ps = "float4 main() : SV_Target { return float4(1, 0, 1, 1); }\n";
        match (compile(ms, "main", "ms_6_5", &[]), compile(ps, "main", "ps_6_0", &[])) {
            (Ok(m), Ok(p)) => Some((m, p)),
            (m, p) => {
                crate::xlog_warn!("D3D12.X: stand-in mesh shaders: {:?} {:?}", m.err(), p.err());
                None
            }
        }
    })
    .as_ref()
}

fn compile(src: &str, target: &str) -> Vec<u8> {
    unsafe {
        let mut blob = None;
        let t = format!("{target}\0");
        let r = D3DCompile(
            src.as_ptr() as *const c_void,
            src.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(b"main\0".as_ptr()),
            PCSTR(t.as_ptr()),
            0,
            0,
            &mut blob,
            None,
        );
        match (r, blob) {
            (Ok(()), Some(b)) => std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec(),
            _ => Vec::new(),
        }
    }
}

/// Stand-in shaders used when the PC driver rejects a console shader.
fn dummy() -> &'static Dummy {
    static D: OnceLock<Dummy> = OnceLock::new();
    D.get_or_init(|| Dummy {
        vs: compile("float4 main(uint id : SV_VertexID) : SV_Position { return float4(0, 0, 0, 1); }", "vs_5_0"),
        ps: compile("float4 main() : SV_Target { return float4(1, 0, 1, 1); }", "ps_5_0"),
        cs: compile("[numthreads(1, 1, 1)] void main() {}", "cs_5_0"),
    })
}

#[cfg(test)]
pub(super) fn dummy_cs_bytecode() -> &'static [u8] {
    &dummy().cs
}

fn bytecode(b: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as *const c_void, BytecodeLength: b.len() }
}

fn note_fallback(what: &str, hr: HRESULT, ok: bool) {
    let n = if ok { FALLBACK_PSO.fetch_add(1, Ordering::Relaxed) } else { FAILED_PSO.fetch_add(1, Ordering::Relaxed) };
    if n < 5 || n % 1000 == 0 {
        if ok {
            crate::xlog_warn!("D3D12.X: {what} rejected by the driver ({hr:#x}) – fallback shaders used (#{})", n + 1);
        } else {
            crate::xlog_error!("D3D12.X: {what} rejected ({hr:#x}), the fallback version too (#{})", n + 1);
        }
    }
}

type PsoFn = unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;

unsafe fn finish_pso(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    // built in the call on one of the title's threads (no stand-in): the driver's time is not
    // the title's (`pso_async::title_held`); a worker's build holds nobody
    let _in_pc = (!super::pso_async::building_here()).then(super::pso_async::pc_call);
    let mut hr = driver_pso(f, this, desc, riid, ppv);
    if hr == E_NOINTERFACE {
        let iid = pc_iid(Kind::PipelineState);
        hr = driver_pso(f, this, desc, &iid, ppv);
    }
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        adopt(*ppv, Kind::PipelineState);
    }
    hr
}

thread_local! {
    /// the driver ran out of memory on a pipeline with heap windows (`with_switch_fallback`)
    static HEAP_WINDOW_OOM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// heap-window driver calls still to fail with E_OUTOFMEMORY (`inject_heap_window_oom`)
static INJECTED_OOM: AtomicU64 = AtomicU64::new(0);
/// pipelines built in the switch form after the driver refused them with heap windows
static SWITCH_FORM_RETRIES: AtomicU64 = AtomicU64::new(0);

/// Tests: the next `n` driver pipeline calls with heap windows fail with E_OUTOFMEMORY without
/// reaching the driver (0 stops it), as NVIDIA 616.92 refuses some big retail programs (#156).
#[doc(hidden)]
pub fn inject_heap_window_oom(n: u64) {
    INJECTED_OOM.store(n, Ordering::SeqCst);
}

/// Pipelines built in the switch form because the driver refused them with heap windows.
pub fn switch_form_retries() -> u64 {
    SWITCH_FORM_RETRIES.load(Ordering::Relaxed)
}

/// Driver pipeline builds for the window title: every `Create*PipelineState` call XCL makes
/// (`driver_pso`, on a worker or on a title thread) while it runs and once done, and the huge ones
/// waiting for their slot (`driver_compile_identified`). A call given a driver blob of an earlier
/// build (`BLOB_SUPPLIED`) is a load, not a build, and is not counted.
struct DriverBuilds {
    building: AtomicUsize,
    waiting: AtomicUsize,
    done: AtomicUsize,
}

static DRIVER_BUILDS: DriverBuilds = DriverBuilds::new();

thread_local! {
    /// this thread's driver call is given a driver blob (`driver_compile_identified`)
    static BLOB_SUPPLIED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct BlobSupplied(bool);

impl BlobSupplied {
    fn set(supplied: bool) -> Self {
        Self(BLOB_SUPPLIED.with(|b| b.replace(supplied)))
    }
}

impl Drop for BlobSupplied {
    fn drop(&mut self) {
        BLOB_SUPPLIED.with(|b| b.set(self.0));
    }
}

/// One counted build or wait: undone when dropped (a build also counts as done then).
struct Counted<'a>(&'a AtomicUsize, Option<&'a AtomicUsize>);

impl Drop for Counted<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
        if let Some(done) = self.1 {
            done.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl DriverBuilds {
    const fn new() -> Self {
        Self { building: AtomicUsize::new(0), waiting: AtomicUsize::new(0), done: AtomicUsize::new(0) }
    }

    fn building(&self) -> Counted<'_> {
        self.building.fetch_add(1, Ordering::Relaxed);
        Counted(&self.building, Some(&self.done))
    }

    fn waiting(&self) -> Counted<'_> {
        self.waiting.fetch_add(1, Ordering::Relaxed);
        Counted(&self.waiting, None)
    }

    fn counts(&self) -> (usize, usize, usize) {
        (self.building.load(Ordering::Relaxed), self.waiting.load(Ordering::Relaxed), self.done.load(Ordering::Relaxed))
    }
}

/// Driver pipeline builds (running now, waiting for a huge-compile slot, done so far), for the
/// window title.
pub fn driver_build_counts() -> (usize, usize, usize) {
    DRIVER_BUILDS.counts()
}

/// A driver pipeline call; E_OUTOFMEMORY while this thread translates with heap windows is
/// noted for `with_switch_fallback`.
unsafe fn driver_pso(f: PsoFn, this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let _build = (!BLOB_SUPPLIED.with(|b| b.get())).then(|| DRIVER_BUILDS.building());
    let heap = crate::gpu::recomp::pso::mem_windows() & crate::gpu::recomp::HEAP_WINDOWS != 0;
    let injected = heap && INJECTED_OOM.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1)).is_ok();
    let hr = if injected {
        E_OUTOFMEMORY
    } else {
        let _stall = crate::stall::span(crate::stall::StallKind::DriverCompile);
        f(this, desc, riid, ppv)
    };
    if hr == E_OUTOFMEMORY && heap {
        HEAP_WINDOW_OOM.with(|o| o.set(true));
    }
    hr
}

/// Build a translated pipeline (`build`: a `translated_*` path); when the driver ran out of
/// memory on it with heap windows, translate it again in the switch form and build that.
/// NVIDIA 616.92 refuses some big programs whose heap index depends on the window with
/// E_OUTOFMEMORY in under a second, and builds the same programs in the switch form (#156).
fn with_switch_fallback<R>(mut build: impl FnMut() -> R) -> R {
    HEAP_WINDOW_OOM.with(|o| o.set(false));
    let r = build();
    if !HEAP_WINDOW_OOM.with(|o| o.replace(false)) {
        return r;
    }
    if crate::gpu::uma::get().is_some_and(|u| u.shader_windows() > crate::gpu::uma::MAX_WINDOWS) {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            crate::xlog_warn!("D3D12.X: the driver refused an expanded-window pipeline; switch-form retry requires XCL_UMA_WINDOWS=4");
        });
        return r;
    }
    let n = SWITCH_FORM_RETRIES.fetch_add(1, Ordering::Relaxed);
    if n < 5 || n % 100 == 0 {
        crate::xlog_warn!("D3D12.X: the driver ran out of memory on a pipeline with heap windows – building it in the switch form ({} so far)", n + 1);
    }
    crate::gpu::recomp::pso::in_switch_form(build)
}

thread_local! {
    /// translation errors noted during a cached-only attempt (`attempt`)
    static ATTEMPT_NOTES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run a translated-pipeline path without compiling (`dxc::cached_only`): (its result, whether
/// a shader was not compiled yet). Errors it noted count only when nothing was missing.
fn attempt<R>(f: impl FnOnce() -> R) -> (R, bool) {
    let (r, missed) = crate::gpu::recomp::dxc::cached_only(f);
    let notes = ATTEMPT_NOTES.with(|n| std::mem::take(&mut *n.borrow_mut()));
    if !missed {
        notes.into_iter().for_each(note_translation_error);
    }
    (r, missed)
}

fn note_translation_error(e: String) {
    // a refusal `with_switch_fallback` builds again in the switch form (it says so itself)
    if HEAP_WINDOW_OOM.with(|o| o.get()) {
        return;
    }
    if crate::gpu::recomp::dxc::in_cached_only() {
        ATTEMPT_NOTES.with(|n| n.borrow_mut().push(e));
        return;
    }
    // each different reason once (the first five only hid a retail game's "mesh pipeline with
    // AS" behind five earlier ones)
    let mut g = TRANSLATE_ERRORS.lock().unwrap();
    if g.len() < 40 && !g.contains(&e) {
        crate::xlog_warn!("D3D12.X: console shaders in the PSO without a translation: {e}");
        g.push(e);
    }
}

unsafe fn shader_bytes<'a>(b: &D3D12_SHADER_BYTECODE) -> Option<&'a [u8]> {
    let (p, n) = (b.pShaderBytecode as usize, b.BytecodeLength);
    (p != 0 && n > 0 && crate::gpu::shader::readable(p, n)).then(|| std::slice::from_raw_parts(p as *const u8, n))
}

/// Root signature of every pipeline built from translated console shaders (static samplers
/// make it per pipeline): the draw binds it instead of the title's.
static PSO_RS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

fn note_pso_rs(pso: *mut c_void, rs: *mut c_void) {
    if !pso.is_null() {
        PSO_RS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso as usize, rs as usize);
    }
}

/// user SGPRs and the console's `numthreads` of the console CS of a translated compute pipeline
static PSO_USER_SGPRS: Mutex<Option<HashMap<usize, (u32, [u32; 3])>>> = Mutex::new(None);
static PSO_DISPATCH_REMAP: Mutex<Option<HashMap<usize, (u32, u32)>>> = Mutex::new(None);

/// How many user SGPRs the console compute shader of a translated pipeline has (None: not
/// known). What is above the driver's four holds root arguments (`console_list::inline_for`).
pub fn pso_user_sgprs(pso: usize) -> Option<u32> {
    let pso = super::pso_async::resolve(pso);
    pso_user_sgprs_exact(pso)
}

pub(super) fn pso_user_sgprs_exact(pso: usize) -> Option<u32> {
    PSO_USER_SGPRS.lock().unwrap().as_ref()?.get(&pso).map(|e| e.0)
}

/// The console's group shape (`numthreads`) of a translated compute pipeline (None: not known).
pub fn pso_threads(pso: usize) -> Option<[u32; 3]> {
    let pso = super::pso_async::resolve(pso);
    pso_threads_exact(pso)
}

pub(super) fn pso_threads_exact(pso: usize) -> Option<[u32; 3]> {
    PSO_USER_SGPRS.lock().unwrap().as_ref()?.get(&pso).map(|e| e.1)
}

/// (tests) A translated compute pipeline with this console group shape.
#[cfg(test)]
pub(crate) fn register_threads_for_test(pso: usize, threads: [u32; 3]) {
    PSO_USER_SGPRS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, (0, threads));
}

#[cfg(test)]
pub(super) fn register_compute_metadata_for_test(
    pso: usize, user_sgprs: u32, threads: [u32; 3], remap: (u32, u32), root: usize,
) {
    PSO_USER_SGPRS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, (user_sgprs, threads));
    PSO_DISPATCH_REMAP.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, remap);
    PSO_RS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, root);
}

pub fn pso_dispatch_remap(pso: usize) -> Option<(u32, u32)> {
    let pso = super::pso_async::resolve(pso);
    pso_dispatch_remap_exact(pso)
}

pub(super) fn pso_dispatch_remap_exact(pso: usize) -> Option<(u32, u32)> {
    PSO_DISPATCH_REMAP.lock().unwrap().as_ref()?.get(&pso).copied()
}

fn note_pso_dispatch_remap(pso: *mut c_void, remap: Option<(u32, u32)>) {
    if pso.is_null() {
        return;
    }
    let mut guard = PSO_DISPATCH_REMAP.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(remap) = remap {
        map.insert(pso as usize, remap);
    } else {
        map.remove(&(pso as usize));
    }
}

/// The root signature a translated pipeline state was created with.
pub fn pso_root_signature(pso: usize) -> Option<usize> {
    let pso = super::pso_async::resolve(pso);
    pso_root_signature_exact(pso)
}

pub(super) fn pso_root_signature_exact(pso: usize) -> Option<usize> {
    PSO_RS.lock().unwrap().as_ref()?.get(&pso).copied()
}

/// ABI root signatures with static samplers, by device and sampler set (owned here, never released).
static ABI_RS_SAMPLERS: Mutex<Option<HashMap<(usize, Vec<u32>), usize>>> = Mutex::new(None);

/// The XCL ABI root signature with a pipeline's static samplers (the plain one without).
unsafe fn abi_rs_for(this: *mut c_void, samplers: &[crate::gpu::recomp::pso::StaticSampler]) -> Option<*mut c_void> {
    if samplers.is_empty() {
        return abi_rs_raw(this);
    }
    let key = (this as usize, samplers.iter().flat_map(|(w, c)| w.iter().copied().chain([*c as u32])).collect::<Vec<_>>());
    let mut g = ABI_RS_SAMPLERS.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(&rs) = map.get(&key) {
        return Some(rs as *mut c_void);
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    match crate::gpu::recomp::pso::abi_root_signature_with(dev, true, samplers) {
        Ok(rs) => {
            let raw = rs.into_raw();
            map.insert(key, raw as usize);
            Some(raw)
        }
        Err(e) => {
            note_translation_error(format!("XCL ABI root signature with static samplers: {e}"));
            None
        }
    }
}

/// XCL ABI root signature, created once per backend device.
unsafe fn abi_rs_raw(this: *mut c_void) -> Option<*mut c_void> {
    let dev_raw = this as usize;
    let cur = ABI_RS.load(Ordering::Acquire);
    if cur != 0 && ABI_RS_DEV.load(Ordering::Acquire) == dev_raw {
        return Some(cur as *mut c_void);
    }
    let mut roots = ABI_RS_BY_DEV.lock().unwrap();
    if let Some(&cached) = roots.as_ref().and_then(|m| m.get(&dev_raw)) {
        return Some(cached as *mut c_void);
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    match crate::gpu::recomp::pso::abi_root_signature(dev, true) {
        Ok(rs) => {
            let raw = rs.into_raw();
            if ABI_RS.load(Ordering::Acquire) == 0 {
                ABI_RS_DEV.store(dev_raw, Ordering::Release);
                ABI_RS.store(raw as usize, Ordering::Release);
            } else {
                roots.get_or_insert_with(HashMap::new).insert(dev_raw, raw as usize);
            }
            Some(raw)
        }
        Err(e) => {
            note_translation_error(format!("XCL ABI root signature: {e}"));
            None
        }
    }
}

/// A graphics pipeline whose VS (and PS) are console shaders: recompiled VS/PS + the XCL
/// shader ABI root signature. None: not console shaders, or not translatable (the caller
/// goes on with the title's description and the stand-in fallback).
/// A compute pipeline with a console CS: `None` = not a console shader, `Some(Err)` = console
/// shader the recompiler could not take, `Some(Ok(hr))` = translated pipeline created (or the
/// driver's verdict on it).
/// DXIL size from which a pipeline's driver compile is "huge" (default 256 KB, `XCL_HUGE_PSO_BYTES`):
/// a compute one goes to the background, and any of them built on an `xcl-pso*` worker waits
/// for one of the `XCL_HUGE_PSO_CONCURRENCY` slots. NVIDIA's compiler (nvgpucomp64) takes
/// minutes and ~1 GB for a 1 – 2 MB program; ~20 of them at once (one per worker) were slower
/// in total than 1 – 2 at a time. Small programs (ms each) never wait.
fn huge_pso_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_count(std::env::var("XCL_HUGE_PSO_BYTES").ok().as_deref(), 256 << 10, 1, usize::MAX))
}

/// Huge driver compiles at once on the workers (default 1, `XCL_HUGE_PSO_CONCURRENCY`).
fn huge_pso_concurrency() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_count(std::env::var("XCL_HUGE_PSO_CONCURRENCY").ok().as_deref(), 1, 1, 64))
}

/// Hard cap on concurrent huge (`>= 256 KB`) driver compiles even when wanted by a title bind
/// (`XCL_HUGE_PSO_WANTED_CONCURRENCY`, default 2, always `>= huge_pso_concurrency()`). Prevents a
/// burst of lazy compute binds from launching 10+ huge driver compiles simultaneously and spiking
/// process commit into OOM, while still letting a wanted build overtake 1 background build.
fn huge_pso_wanted_concurrency() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let base = huge_pso_concurrency();
        env_count(std::env::var("XCL_HUGE_PSO_WANTED_CONCURRENCY").ok().as_deref(), base.max(2), base, 64)
    })
}

/// A count from the environment: `default` when unset or not a number, clamped to the
/// inclusive range `min..=max`.
fn env_count(v: Option<&str>, default: usize, min: usize, max: usize) -> usize {
    v.and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(default).clamp(min, max)
}

/// A counting semaphore: background driver compiles of big programs at once (`most` 0: the
/// `huge_pso_concurrency` of the environment; `wanted_most` 0: `huge_pso_wanted_concurrency`).
struct Slots {
    n: Mutex<usize>,
    cv: std::sync::Condvar,
    most: usize,
    wanted_most: usize,
}

impl Slots {
    fn most(&self) -> usize {
        if self.most == 0 { huge_pso_concurrency() } else { self.most }
    }

    fn wanted_most(&self) -> usize {
        if self.wanted_most == 0 { huge_pso_wanted_concurrency() } else { self.wanted_most }
    }
}

struct SlotGuard(&'static Slots);

impl Drop for SlotGuard {
    fn drop(&mut self) {
        *self.0.n.lock().unwrap() -= 1;
        self.0.cv.notify_all();
    }
}

static BIG_COMPILES: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 0, wanted_most: 0 };
const AMD_MEDIUM_COMPUTE: usize = 128 << 10;
const MEDIUM_COMPUTE_END: usize = 256 << 10;
static MEDIUM_COMPILES: LazyLock<Slots> = LazyLock::new(|| {
    let (default_most, default_wanted_most) = medium_pso_defaults(total_physical_ram());
    let most = env_count(std::env::var("XCL_MEDIUM_PSO_CONCURRENCY").ok().as_deref(), default_most, 1, 64);
    Slots {
        n: Mutex::new(0),
        cv: std::sync::Condvar::new(),
        most,
        wanted_most: env_count(
            std::env::var("XCL_MEDIUM_PSO_WANTED_CONCURRENCY").ok().as_deref(),
            default_wanted_most.max(most),
            most,
            64,
        ),
    }
});
const GIANT_DRIVER_COMPILE: usize = 4 << 20;
static GIANT_COMPILES: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 1, wanted_most: 1 };

fn compile_slots(bytes: usize) -> &'static Slots {
    if bytes >= GIANT_DRIVER_COMPILE { &GIANT_COMPILES } else { &BIG_COMPILES }
}

fn total_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some(m.ullTotalPhys)
}

fn medium_pso_defaults(total_phys: Option<u64>) -> (usize, usize) {
    if total_phys.is_some_and(|n| n < 32u64 << 30) { (2, 3) } else { (3, 4) }
}

/// AMD's compiler takes tens of seconds and over 1 GiB per cold 128–256 KiB compute PSO.
/// Keep that AMD-only class separate from the existing >=256 KiB gate: graphics and other
/// vendors retain their current parallelism, while a multi-minute huge compile cannot block it.
fn medium_compute_compile(kind: &str, bytes: usize, huge_bytes: usize, vendor: u32) -> bool {
    vendor == 0x1002
        && (kind == "compute" || kind == "compute retry")
        && bytes >= AMD_MEDIUM_COMPUTE
        && bytes < huge_bytes.min(MEDIUM_COMPUTE_END)
}

/// A driver compile of `bytes` of DXIL through `build`: a huge one on an `xcl-pso*` worker
/// waits for its slot first (`big_compile_turn`; not a title thread's, nor one the title waits
/// for). Logs its time, size and whether it was gated.
#[cfg(test)]
fn driver_compile<R>(kind: &str, bytes: usize, gate: bool, build: impl FnOnce() -> R) -> R {
    driver_compile_identified(kind, bytes, gate, None, false, build).0
}

/// `XCL_PSO_TRACE=1`: every driver compile is logged, not only huge ones.
fn pso_trace() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("XCL_PSO_TRACE").is_ok_and(|v| v.trim() == "1"))
}

/// A timed driver compile with the console record's hash (`fnv64` of the shader container, as in
/// `[SHADER]` lines and `xcl shaders --measure-pso`) and whether a driver blob was supplied.
/// Also returns the time spent in the driver alone (not waiting for a slot).
fn driver_compile_identified<R>(kind: &str, bytes: usize, gate: bool, record: Option<u64>, cached: bool, build: impl FnOnce() -> R) -> (R, std::time::Duration) {
    let huge_bytes = huge_pso_bytes();
    let worker = std::thread::current().name().is_some_and(|n| n.starts_with("xcl-pso"));
    let medium = gate && worker && medium_compute_compile(kind, bytes, huge_bytes, ADAPTER_VENDOR.load(Ordering::Acquire));
    let huge = gate && worker && bytes >= huge_bytes;
    let gated = medium || huge;
    let waited = std::time::Instant::now();
    let gate_stall = gated.then(|| crate::stall::span(crate::stall::StallKind::DriverCompileGate)).flatten();
    let _slot = gated.then(|| {
        let _waiting = DRIVER_BUILDS.waiting();
        let slots = if medium { &*MEDIUM_COMPILES } else { compile_slots(bytes) };
        big_compile_turn(slots, || big_compile_commit_free(big_compile_headroom(bytes)), super::pso_async::build_is_wanted)
    });
    drop(gate_stall);
    let waited = waited.elapsed();
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    let log = pso_trace() || bytes >= huge_bytes || medium;
    if log {
        let record = record.map_or_else(
            || if cached { ", driver blob supplied".to_string() } else { String::new() },
            |hash| format!(", record {hash:016x}, driver blob {}", if cached { "supplied" } else { "absent" }),
        );
        crate::xlog_info!("D3D12.X: driver compile start #{sequence}: {kind} PSO, {bytes} B DXIL{record}");
    }
    let started = std::time::Instant::now();
    let _blob = BlobSupplied::set(cached);
    let r = build();
    let took = started.elapsed();
    super::pso_manifest::note(super::pso_manifest::Event::Build {
        kind, bytes, record, cached, driver_ms: took.as_millis(), waited_ms: waited.as_millis(),
    });
    if log {
        crate::xlog_info!("D3D12.X: driver compile finish #{sequence}: {kind} PSO, {bytes} B DXIL, {} ms{}", took.as_millis(), gate_note(bytes >= huge_bytes || medium, gated, waited));
    } else if took >= std::time::Duration::from_secs(1) {
        // a slow one below the huge size, logged without XCL_PSO_TRACE too (a title thread's
        // graphics pipelines took 2 - 5 s each, unlogged)
        let record = record.map_or_else(
            || if cached { ", driver blob supplied".to_string() } else { String::new() },
            |hash| format!(", record {hash:016x}, driver blob {}", if cached { "supplied" } else { "absent" }),
        );
        let thread = std::thread::current().name().unwrap_or("title").to_string();
        crate::xlog_info!("D3D12.X: slow driver compile #{sequence}: {kind} PSO, {bytes} B DXIL{record}, {} ms on {thread}", took.as_millis());
    }
    (r, took)
}

/// How a logged driver compile went through the huge-pipeline gate.
fn gate_note(huge_size: bool, gated: bool, waited: std::time::Duration) -> String {
    match (huge_size, gated) {
        (false, _) => String::new(),
        (true, true) => format!(", gated (waited {} ms)", waited.as_millis()),
        (true, false) => ", huge, ungated here (a title thread's, cached, or gated by its caller)".to_string(),
    }
}

/// A cached-blob attempt should be quick and must not take a cold-compile slot. Its retry
/// without the rejected blob is a cold compile and does take one.
fn gate_compute_driver_compile(has_cached_blob: bool) -> bool {
    !has_cached_blob
}

/// Before a big driver compile in the background: its turn – unwanted builds wait for both
/// `slots.most()` and commit headroom (`commit_free`, from `big_compile_headroom`) before taking a
/// slot so an unwanted commit-blocked compile never starves a wanted build. When a title thread
/// waits for the pipeline (`wanted`), it bypasses the commit wait and can use up to
/// `slots.wanted_most()` (default 2 for `>= 256 KB`, 1 for `>= 4 MiB` giant shaders) so a wanted
/// build is not stuck behind 1 background build while a burst of lazy binds still cannot spike
/// driver compiler commit into OOM.
fn big_compile_turn(slots: &'static Slots, commit_free: impl Fn() -> bool, wanted: impl Fn() -> bool) -> SlotGuard {
    const LOOK: std::time::Duration = std::time::Duration::from_millis(50);
    let mut n = slots.n.lock().unwrap();
    let mut parked = None;
    loop {
        let is_wanted = wanted();
        let cap = if is_wanted { slots.wanted_most() } else { slots.most() };
        if *n < cap && (is_wanted || commit_free()) {
            *n += 1;
            return SlotGuard(slots);
        }
        if parked.is_none() {
            parked = Some(super::pso_async::parked_for_big_compile());
        }
        n = slots.cv.wait_timeout(n, LOOK).unwrap().0;
    }
}

/// Commit headroom a big driver compile waits for: 3 GB for the class that starts at
/// `huge_pso_bytes()` (each takes ~1.5 GB in the driver), and ~512× the DXIL size above 6 MiB
/// of it — the driver's compile of a 29.8 MB program grew past 14 GB free before it was stopped
/// (#1269), and programs of ~1 MB took several GB each on an 8 GB card (same issue). The floor
/// leaves every program up to 6 MiB of DXIL waiting for exactly what it waited for before.
fn big_compile_headroom(dxil_len: usize) -> u64 {
    (3u64 << 30).max(dxil_len as u64 * 512)
}

/// That much commit is free (true when that is not known).
fn big_compile_commit_free(need: u64) -> bool {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    unsafe { GlobalMemoryStatusEx(&mut m) == 0 || m.ullAvailPageFile >= need }
}

#[cfg(test)]
mod big_compile_tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn wanted_builds_bypass_commit_and_unwanted_slot_but_respect_hard_cap() {
        static S: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 1, wanted_most: 2 };
        let held = big_compile_turn(&S, || true, || false);
        assert_eq!(*S.n.lock().unwrap(), 1);
        // Unwanted slot is full, but a wanted build immediately takes the 2nd (wanted) slot and
        // bypasses the commit-free check.
        let held_wanted = big_compile_turn(&S, || false, || true);
        assert_eq!(*S.n.lock().unwrap(), 2);
        // A 3rd wanted build MUST wait while both slots are occupied (hard cap = 2).
        let t = std::thread::spawn(|| {
            let _g = big_compile_turn(&S, || false, || true);
        });
        std::thread::sleep(Duration::from_millis(120));
        assert!(!t.is_finished(), "3rd concurrent wanted huge compile must wait for hard cap");
        drop(held_wanted);
        t.join().unwrap();
        drop(held);
        assert_eq!(*S.n.lock().unwrap(), 0);
        // Without commit free: an unwanted build waits without holding a slot until it becomes
        // wanted, then acquires a slot and proceeds.
        static WANTED2: AtomicBool = AtomicBool::new(false);
        let t0 = Instant::now();
        let t = std::thread::spawn(|| {
            let _g = big_compile_turn(&S, || false, || WANTED2.load(Ordering::Acquire));
        });
        std::thread::sleep(Duration::from_millis(120));
        assert!(!t.is_finished(), "no commit free: unwanted build waits");
        assert_eq!(*S.n.lock().unwrap(), 0, "unwanted commit-blocked build must not hold a slot");
        WANTED2.store(true, Ordering::Release);
        t.join().unwrap();
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert_eq!(*S.n.lock().unwrap(), 0, "the slot went back");
    }

    #[test]
    fn unwanted_commit_waiter_does_not_starve_wanted_compile() {
        // Mirrors GIANT_COMPILES (`most == 1, wanted_most == 1`) under low commit (< 3 GiB):
        // an unwanted compile waiting on commit headroom must never hold the slot and deadlock a
        // wanted compile.
        static S: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 1, wanted_most: 1 };
        static FREE: AtomicBool = AtomicBool::new(false);
        let bg = std::thread::spawn(|| {
            let _g = big_compile_turn(&S, || FREE.load(Ordering::Acquire), || false);
        });
        std::thread::sleep(Duration::from_millis(120));
        assert!(!bg.is_finished(), "unwanted compile waits for commit");
        assert_eq!(*S.n.lock().unwrap(), 0, "unwanted compile must not hold slot while commit-blocked");
        let t0 = Instant::now();
        let wanted = big_compile_turn(&S, || false, || true);
        assert!(t0.elapsed() < Duration::from_secs(1), "wanted compile must acquire slot immediately");
        assert_eq!(*S.n.lock().unwrap(), 1);
        drop(wanted);
        FREE.store(true, Ordering::Release);
        bg.join().unwrap();
        assert_eq!(*S.n.lock().unwrap(), 0);
    }

    #[test]
    fn wanted_burst_is_hard_capped_to_wanted_most() {
        use std::sync::atomic::AtomicUsize;
        static S: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 1, wanted_most: 2 };
        static IN: AtomicUsize = AtomicUsize::new(0);
        static PEAK: AtomicUsize = AtomicUsize::new(0);
        let t: Vec<_> = (0..12)
            .map(|_| {
                std::thread::spawn(|| {
                    let _g = big_compile_turn(&S, || false, || true);
                    let now = IN.fetch_add(1, Ordering::SeqCst) + 1;
                    PEAK.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(15));
                    IN.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        t.into_iter().for_each(|h| h.join().unwrap());
        assert_eq!(PEAK.load(Ordering::SeqCst), 2, "12 concurrent wanted huge compiles must cap at 2");
        assert_eq!(*S.n.lock().unwrap(), 0);
    }

    #[test]
    fn env_counts_parse() {
        assert_eq!(env_count(None, 2, 1, 64), 2);
        assert_eq!(env_count(Some(" 4 "), 2, 1, 64), 4);
        assert_eq!(env_count(Some("0"), 2, 1, 64), 1, "at least one slot: no deadlock");
        assert_eq!(env_count(Some("lots"), 2, 1, 64), 2);
        assert_eq!(env_count(Some("1000"), 2, 1, 64), 64, "an accidental value cannot defeat the cap");
        assert_eq!(env_count(Some("524288"), 256 << 10, 1, usize::MAX), 512 << 10);
    }

    #[test]
    fn only_huge_compiles_get_a_gate_note() {
        let ms = Duration::from_millis(5);
        assert_eq!(gate_note(false, false, ms), "", "a small compile (XCL_PSO_TRACE) is not called huge");
        assert_eq!(gate_note(true, true, ms), ", gated (waited 5 ms)");
        assert!(gate_note(true, false, ms).contains("ungated here"));
    }

    #[test]
    fn driver_builds_count_while_running_and_once_done() {
        let b = DriverBuilds::new();
        {
            let _one = b.building();
            let _two = b.building();
            let _wait = b.waiting();
            assert_eq!(b.counts(), (2, 1, 0));
        }
        assert_eq!(b.counts(), (0, 0, 2), "a wait is not a build");
    }

    unsafe extern "system" fn counting_driver(_: *mut c_void, _: *const c_void, _: *const GUID, _: *mut *mut c_void) -> HRESULT {
        // the global counter: other tests' builds can run too, so at least this one
        if DRIVER_BUILDS.counts().0 >= 1 { S_OK } else { S_FALSE }
    }

    unsafe extern "system" fn blob_driver(_: *mut c_void, _: *const c_void, _: *const GUID, _: *mut *mut c_void) -> HRESULT {
        if BLOB_SUPPLIED.with(|b| b.get()) { S_OK } else { S_FALSE }
    }

    #[test]
    fn every_driver_call_is_counted_and_a_blob_load_is_not() {
        let done = DRIVER_BUILDS.counts().2;
        let call = |f: PsoFn| unsafe { driver_pso(f, std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), std::ptr::null_mut()) };
        assert_eq!(call(counting_driver), S_OK, "counted while in the driver");
        assert!(DRIVER_BUILDS.counts().2 > done);
        // a compile given a driver blob reaches the driver as a load; a cold one does not
        assert_eq!(driver_compile_identified("test", 1, false, Some(1), true, || call(blob_driver)).0, S_OK);
        assert_eq!(driver_compile_identified("test", 1, false, Some(1), false, || call(blob_driver)).0, S_FALSE);
        assert_eq!(driver_compile_identified("two-pass test", 1, false, None, true, || call(blob_driver)).0, S_OK,
            "a cache hit without a console record is still a driver-blob load");
        let _ = std::panic::catch_unwind(|| driver_compile_identified("panic test", 1, false, None, true, || panic!("driver panic")));
        assert!(!BLOB_SUPPLIED.with(|b| b.get()), "reset after the call");
    }

    #[test]
    fn giant_driver_compiles_have_their_own_lane() {
        assert!(std::ptr::eq(compile_slots(GIANT_DRIVER_COMPILE - 1), &BIG_COMPILES));
        assert!(std::ptr::eq(compile_slots(GIANT_DRIVER_COMPILE), &GIANT_COMPILES));
    }

    #[test]
    fn medium_compute_class_is_amd_only_and_respects_the_huge_override() {
        let amd = 0x1002;
        let nvidia = 0x10de;
        assert!(!medium_compute_compile("compute", AMD_MEDIUM_COMPUTE - 1, 256 << 10, amd));
        assert!(medium_compute_compile("compute", AMD_MEDIUM_COMPUTE, 256 << 10, amd));
        assert!(medium_compute_compile("compute retry", 217 << 10, 256 << 10, amd));
        assert!(!medium_compute_compile("compute", MEDIUM_COMPUTE_END, 256 << 10, amd));
        assert!(!medium_compute_compile("graphics", 180 << 10, 256 << 10, amd));
        assert!(!medium_compute_compile("compute", 180 << 10, 256 << 10, nvidia));
        assert!(!medium_compute_compile("compute", 180 << 10, 128 << 10, amd),
            "a lower XCL_HUGE_PSO_BYTES routes this size through the existing huge lane");
        assert!(!std::ptr::eq(&*MEDIUM_COMPILES, &BIG_COMPILES),
            "medium compute must not occupy the existing huge slot");
    }

    #[test]
    fn medium_compute_concurrency_scales_down_below_32_gib() {
        assert_eq!(medium_pso_defaults(Some(16u64 << 30)), (2, 3));
        assert_eq!(medium_pso_defaults(Some((32u64 << 30) - 1)), (2, 3));
        assert_eq!(medium_pso_defaults(Some(32u64 << 30)), (3, 4));
        assert_eq!(medium_pso_defaults(Some(64u64 << 30)), (3, 4));
        assert_eq!(medium_pso_defaults(None), (3, 4));
    }

    #[test]
    fn only_a_cold_compute_compile_takes_a_slot() {
        assert!(!gate_compute_driver_compile(true), "the cached-blob attempt bypasses the gate");
        assert!(gate_compute_driver_compile(false), "a cold attempt, including a retry, is gated");
    }

    #[test]
    fn the_concurrency_limit_holds_and_all_finish() {
        use std::sync::atomic::AtomicUsize;
        static S: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 2, wanted_most: 2 };
        static IN: AtomicUsize = AtomicUsize::new(0);
        static PEAK: AtomicUsize = AtomicUsize::new(0);
        // the two holders of the slots meet here: both are inside at once, deterministically
        static BOTH: std::sync::Barrier = std::sync::Barrier::new(2);
        let t: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    let _g = big_compile_turn(&S, || true, || false);
                    let now = IN.fetch_add(1, Ordering::SeqCst) + 1;
                    PEAK.fetch_max(now, Ordering::SeqCst);
                    BOTH.wait();
                    IN.fetch_sub(1, Ordering::SeqCst);
                    BOTH.wait();
                })
            })
            .collect();
        let t0 = Instant::now();
        t.into_iter().for_each(|t| t.join().unwrap());
        assert!(t0.elapsed() < Duration::from_secs(10), "no deadlock");
        assert_eq!(PEAK.load(Ordering::SeqCst), 2);
        assert_eq!(*S.n.lock().unwrap(), 0);
    }

    #[test]
    fn small_and_title_thread_compiles_bypass_the_gate() {
        // every slot taken: a small compile on a worker, and a huge one on an unnamed (title)
        // thread, both go on at once
        // (both lanes: a huge compile of any size would block on a gate it should bypass)
        let mut held: Vec<_> = (0..huge_pso_concurrency()).map(|_| big_compile_turn(&BIG_COMPILES, || true, || false)).collect();
        held.push(big_compile_turn(&GIANT_COMPILES, || true, || false));
        let small = std::thread::Builder::new()
            .name("xcl-pso-test".into())
            .spawn(|| driver_compile("test", huge_pso_bytes() - 1, true, || 7))
            .unwrap();
        let t0 = Instant::now();
        assert_eq!(small.join().unwrap(), 7);
        let big = huge_pso_bytes().max(GIANT_DRIVER_COMPILE - 1);
        assert_eq!(std::thread::spawn(move || driver_compile("test", big, true, || 8)).join().unwrap(), 8);
        assert_eq!(std::thread::spawn(|| driver_compile("test", usize::MAX / 2, true, || 8)).join().unwrap(), 8);
        assert!(t0.elapsed() < Duration::from_secs(2));
        drop(held);
    }

    #[test]
    fn headroom_is_the_3_gb_floor_until_512_times_the_dxil_exceeds_it() {
        // the class from huge_pso_bytes() up to 6 MiB of DXIL: the floor, unchanged
        assert_eq!(big_compile_headroom(huge_pso_bytes()), 3 << 30);
        assert_eq!(big_compile_headroom(6 << 20), 3 << 30);
        // above it the wait scales with the program: the 29.8 MB one that grew past 14 GB of
        // free commit (#1269) waits for 15 GB
        assert_eq!(big_compile_headroom(8 << 20), 4 << 30);
        assert_eq!(big_compile_headroom(29_800_000), 29_800_000u64 * 512);
    }

    #[test]
    fn unwanted_builds_go_one_at_a_time_when_the_commit_is_there() {
        static S: Slots = Slots { n: Mutex::new(0), cv: std::sync::Condvar::new(), most: 1, wanted_most: 2 };
        let first = big_compile_turn(&S, || true, || false);
        let t = std::thread::spawn(|| {
            let _g = big_compile_turn(&S, || true, || false);
        });
        std::thread::sleep(Duration::from_millis(120));
        assert!(!t.is_finished(), "one at a time");
        drop(first);
        t.join().unwrap();
        assert_eq!(*S.n.lock().unwrap(), 0);
    }
}

unsafe fn translated_compute_pso(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<Result<HRESULT, ()>> {
    with_switch_fallback(|| translated_compute_pso_once(this, f, desc, riid, ppv))
}

unsafe fn translated_compute_pso_once(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<Result<HRESULT, ()>> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    if !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return None;
    }
    let d = &*(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let cs = shader_bytes(&d.CS)?;
    if !is_console(cs) {
        return None;
    }
    // made when first bound (`pso_async::lazy`): not even translated now
    if super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() {
        return Some(Err(()));
    }
    let compiled = match console_to_dxil_for(cs, Stage::Compute, 0, wave32_only(this)) {
        Ok(b) => b,
        Err(e) => {
            note_translation_error(format!("CS: {e}"));
            return Some(Err(()));
        }
    };
    let dxil = compiled.dxil;
    // a big program: the driver's compile takes it seconds to a minute (a retail game's 1 MB
    // ray-tracing resolve programs ~30 s each, ~600 of them): built in the background like one
    // whose DXIL is missing – the title's loading does not wait for it, its first bind does
    let big = dxil.len() >= huge_pso_bytes();
    if big && crate::gpu::recomp::dxc::defer() {
        return Some(Err(()));
    }
    let record = crate::gpu::shader::fnv64(cs);
    let warm = compiled.dispatch_chain.as_ref().and_then(|_| {
        let rs = abi_rs_for(this, &compiled.samplers)?;
        let result = compute_container(this, f, d, rs, &dxil, &compiled.samplers, record, riid, ppv, ComputeBuild::CachedWhole);
        (result.0 >= 0).then_some((rs, result))
    });
    if let Some(plan) = compiled.dispatch_chain.filter(|_| warm.is_none()) {
        if let Some(rs) = dispatch_chain::build(this, f, d, riid, ppv, record, &dxil, &compiled.samplers, plan) {
            LAST_COMPUTE_RS.with(|c| c.set(rs));
            note_pso_rs(*ppv, rs as *mut c_void);
            super::dred::label(*ppv as usize, &[("cs", Some(cs))]);
            if let Some(i) = crate::gpu::recomp::pso::compute_interface(cs) {
                PSO_USER_SGPRS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (i.user_sgprs, i.threads));
            }
            note_translated_compute(0, ppv);
            TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
            return Some(Ok(0));
        }
    }
    let (rs, (hr, in_driver, reused)) = match warm {
        Some(hit) => hit,
        None => {
            let Some(rs) = abi_rs_for(this, &compiled.samplers) else { return Some(Err(())) };
            (rs, compute_container(this, f, d, rs, &dxil, &compiled.samplers, record, riid, ppv, ComputeBuild::Whole))
        }
    };
    LAST_COMPUTE_RS.with(|c| c.set(rs as usize));
    if hr < 0 {
        // A huge program's verdict in the log itself: `note_translation_error` goes to the
        // report, or to the notes of a cached-only attempt – a retail title's 13.8 MB program
        // took an AMD driver 15 – 23 s and ~15 GB to refuse with E_OUTOFMEMORY in every run, and
        // the log showed only its `driver compile finish` and the dispatches skipped after it.
        if dxil.len() >= huge_pso_bytes() {
            // (the record hash is the console program's, as in `driver compile start` and
            // `xcl shaders --measure-pso`; the driver-cache name is the translated DXIL's)
            crate::xlog_warn!(
                "D3D12.X: the driver refused the compute PSO of record {record:016x} ({} B DXIL, driver cache {}) with {hr:#x} after {} ms in the driver: its dispatches are skipped",
                dxil.len(),
                driver_cache::name_for_root_node(&dxil, d.Flags.0, false, &compiled.samplers, d.NodeMask),
                in_driver.as_millis()
            );
        }
        note_translation_error(format!("the driver rejected the compute PSO with the translated CS ({hr:#x})"));
        return Some(Err(()));
    }
    TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
    note_pso_rs(*ppv, rs);
    super::dred::label(*ppv as usize, &[("cs", Some(cs))]);
    if let Some(i) = crate::gpu::recomp::pso::compute_interface(cs) {
        PSO_USER_SGPRS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (i.user_sgprs, i.threads));
    }
    note_pso_dispatch_remap(*ppv, compiled.dispatch_remap);
    remap_fallback::note(*ppv as usize, this, f, rs as usize, &compiled.samplers, d.Flags, d.NodeMask, record, cs,
        wave32_only(this), crate::gpu::recomp::pso::mem_windows(), compiled.dispatch_remap.is_some());
    note_translated_compute(hr, ppv);
    crate::xlog_info!(
        "D3D12.X: compute PSO with the translated console CS ({} B DXIL, {} ms{})",
        dxil.len(),
        in_driver.as_millis(),
        if reused { ", prepared native pipeline or driver's blob" } else { "" }
    );
    Some(Ok(hr))
}

pub(super) mod dispatch_chain;
pub(crate) use dispatch_chain::log_counts as log_dispatch_chain_counts;
pub(crate) use dispatch_chain::earlier_indirect_used as earlier_indirect_split_records;
pub(crate) use dispatch_chain::flush_indirect_used as flush_indirect_split_records;
pub(crate) use dispatch_chain::shutdown_indirect_used as shutdown_indirect_split_records;

pub(super) mod remap_fallback {
    use super::*;
    use std::sync::{Arc, LazyLock};

    struct Fallback {
        device: ID3D12Device,
        create: PsoFn,
        signature: ID3D12RootSignature,
        samplers: Vec<crate::gpu::recomp::pso::StaticSampler>,
        flags: D3D12_PIPELINE_STATE_FLAGS,
        node: u32,
        record: u64,
        cs: Vec<u8>,
        split_wave64: bool,
        windows: u32,
        whole: dispatch_chain::WholeBuild<ID3D12PipelineState>,
    }

    static FALLBACKS: LazyLock<Mutex<HashMap<usize, Arc<Fallback>>>> = LazyLock::new(Default::default);

    pub(super) fn compile_error_hr(err: &str) -> HRESULT {
        if crate::gpu::recomp::dxc::is_oom_error(err) { E_OUTOFMEMORY } else { E_INVALIDARG }
    }

    pub(super) unsafe fn note(pso: usize, this: *mut c_void, create: PsoFn, rs: usize,
        samplers: &[crate::gpu::recomp::pso::StaticSampler], flags: D3D12_PIPELINE_STATE_FLAGS,
        node: u32, record: u64, cs: &[u8], split_wave64: bool, windows: u32, remapped: bool) {
        if remapped && pso != 0 {
            let Some(device) = ID3D12Device::from_raw_borrowed(&this) else { return };
            let raw_rs = rs as *mut c_void;
            let Some(signature) = ID3D12RootSignature::from_raw_borrowed(&raw_rs) else { return };
            let old = FALLBACKS.lock().unwrap().insert(pso, Arc::new(Fallback {
                device: device.clone(), create, signature: signature.clone(), samplers: samplers.to_vec(), flags, node, record,
                cs: cs.to_vec(), split_wave64, windows,
                whole: dispatch_chain::WholeBuild::new(),
            }));
            drop(old);
        }
    }

    #[allow(dead_code)] // title-address counterpart to whole_exact
    pub(in crate::hle::d3d12x) unsafe fn whole(pso: usize) -> Result<ID3D12PipelineState, String> {
        let pso = super::super::pso_async::resolve(pso);
        whole_exact(pso)
    }

    pub(in crate::hle::d3d12x) unsafe fn whole_exact(pso: usize) -> Result<ID3D12PipelineState, String> {
        let fb = FALLBACKS.lock().unwrap().get(&pso).cloned().ok_or("no remap fallback")?;
        fb.whole.get_or_build(std::time::Instant::now(), || {
            let status = Arc::new(std::sync::atomic::AtomicI32::new(E_INVALIDARG));
            let (job_fb, job_status) = (fb.clone(), status.clone());
            let raw = super::super::pso_async::build_foreground(Box::new(move || unsafe {
                let dxil = match crate::gpu::recomp::pso::console_dxil_noremap(&job_fb.cs, job_fb.split_wave64, job_fb.windows) {
                    Ok(dxil) => dxil,
                    Err(e) => {
                        job_status.store(compile_error_hr(&e), Ordering::Release);
                        return 0
                    }
                };
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { NodeMask: job_fb.node, Flags: job_fb.flags, ..Default::default() };
                let mut raw = std::ptr::null_mut();
                let (hr, _, _) = compute_container(job_fb.device.as_raw(), job_fb.create, &desc,
                    job_fb.signature.as_raw(), &dxil, &job_fb.samplers, job_fb.record,
                    &pc_iid(Kind::PipelineState), &mut raw, ComputeBuild::Whole);
                job_status.store(hr, Ordering::Release);
                if hr >= 0 { raw as usize } else { 0 }
            }));
            if raw != 0 { Ok(ID3D12PipelineState::from_raw(raw as *mut c_void)) }
            else { Err(status.load(Ordering::Acquire)) }
        }).map_err(|hr| format!("un-remapped compute pipeline ({hr:#x})"))
    }

    pub(super) fn forget(pso: usize) {
        let gone = FALLBACKS.lock().unwrap().remove(&pso);
        drop(gone);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ComputeBuild {
    Whole,
    // Try the existing whole program before building a chain. A miss or rejected blob falls
    // through to building the pieces, whose cold build is cheaper.
    CachedWhole,
    // The chain root signature has a separate on-disk namespace.
    DirectChain,
    IndirectChain,
}

unsafe fn compute_container(this: *mut c_void, f: PsoFn, d: &D3D12_COMPUTE_PIPELINE_STATE_DESC, rs: *mut c_void,
    dxil: &[u8], samplers: &[crate::gpu::recomp::pso::StaticSampler], record: u64, riid: *const GUID,
    ppv: *mut *mut c_void, mode: ComputeBuild) -> (HRESULT, std::time::Duration, bool) {
    let mut copy: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::ptr::read(d);
    copy.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    copy.CS = bytecode(dxil);
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    let library_name = match mode {
        ComputeBuild::DirectChain => driver_cache::name_for_chain_root_node(
            dxil, copy.Flags.0, driver_cache::ChainRoot::DirectConstants, samplers, copy.NodeMask),
        ComputeBuild::IndirectChain => driver_cache::name_for_chain_root_node(
            dxil, copy.Flags.0, driver_cache::ChainRoot::IndirectCbv, samplers, copy.NodeMask),
        _ => driver_cache::name_for_root_node(dxil, copy.Flags.0, false, samplers, copy.NodeMask),
    };
    if !ppv.is_null() {
        if let Some(pso) = crate::shaders::pipeline_library::load_compute(&library_name, &copy) {
            *ppv = pso.into_raw();
            return (S_OK, std::time::Duration::ZERO, true);
        }
    }
    // A program built again gets the driver's own result of its first build: a retail title
    // made 14 big compute pipelines twice while loading, 13 - 30 s each the second time.
    let key = compute_blob_key_for(dxil, rs as usize, copy.Flags.0, copy.NodeMask);
    let mut cached = compute_blob(key);
    // … and one built in an earlier run gets the blob kept on disk then (`driver_cache`)
    let disk = match mode {
        ComputeBuild::DirectChain => driver_cache::path_for_chain_root_node(
            dxil, copy.Flags.0, driver_cache::ChainRoot::DirectConstants, samplers, copy.NodeMask),
        ComputeBuild::IndirectChain => driver_cache::path_for_chain_root_node(
            dxil, copy.Flags.0, driver_cache::ChainRoot::IndirectCbv, samplers, copy.NodeMask),
        _ => driver_cache::path_for_root_node(dxil, copy.Flags.0, false, samplers, copy.NodeMask),
    };
    let from_disk = cached.is_none();
    let mut loaded = None;
    if from_disk {
        loaded = disk.as_deref().and_then(driver_cache::load);
        cached = loaded.as_ref().map(driver_cache::LoadedBlob::shared);
    }
    let rejected_generation = cached.clone();
    if let Some(b) = &cached {
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr() as *const c_void, CachedBlobSizeInBytes: b.len() };
    } else if mode == ComputeBuild::CachedWhole {
        return (E_INVALIDARG, std::time::Duration::ZERO, false);
    }
    // in the background, a few at a time and within the commit headroom the program asks for
    let (mut hr, mut in_driver) = driver_compile_identified(
        "compute",
        dxil.len(),
        gate_compute_driver_compile(cached.is_some()),
        Some(record),
        cached.is_some(),
        || finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv),
    );
    let mut persist_took = in_driver;
    if hr < 0 && cached.is_some() {
        // a blob the driver does not take (another root signature at the same address…): compiled
        if let Some(expected) = rejected_generation.as_ref() {
            forget_compute_blob_if(key, expected);
        }
        if from_disk {
            driver_cache::note(false, loaded.take());
        }
        if mode == ComputeBuild::CachedWhole {
            return (hr, in_driver, false);
        }
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        let (retried, took) = driver_compile_identified(
            "compute retry",
            dxil.len(),
            gate_compute_driver_compile(false),
            Some(record),
            false,
            || finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv),
        );
        hr = retried;
        in_driver += took;
        persist_took = took;
    }
    let reused = hr >= 0 && cached.is_some() && copy.CachedPSO.CachedBlobSizeInBytes != 0;
    if hr >= 0 && !reused {
        keep_driver_blob(&COMPUTE_BLOBS, COMPUTE_BLOBS_MAX, key, dxil.len(), disk.as_deref(), persist_took,
            || read_pso_cached_blob(*ppv));
    } else if reused && from_disk {
        COMPUTE_BLOBS.admit(key, cached.as_ref().unwrap().clone(), COMPUTE_BLOBS_MAX);
        driver_cache::note(true, loaded.take());
    }
    (hr, in_driver, reused)
}

/// The blobs of `COMPUTE_BLOBS`, kept on disk between runs.
///
/// `COMPUTE_BLOBS` saves the second build of a program in one run; the first build of every
/// run still goes through the driver's compiler. On an NVIDIA GPU that is 5 – 27 s for each of
/// a retail title's larger compute programs when its world loads: the title dispatches without
/// them meanwhile and is ended before all are built, so the next run starts the same compiles
/// again. A blob whose build took `KEEP` or longer is written under the hash of its DXIL and
/// its flags (`<DXIL cache>\driver\`) and read when a later run has none in memory; the
/// driver refuses the blob of another driver, adapter or root signature (the file is dropped
/// and the program compiled as before). Off by default on AMD; `XCL_PSO_DISK_CACHE=1` forces
/// it on, and `XCL_PSO_DISK_CACHE=0` turns it off everywhere.
pub(crate) mod driver_cache {
    use crate::gpu::recomp::pso::StaticSampler;
    use std::io::Read;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// builds shorter than this are not worth a file
    const KEEP: std::time::Duration = std::time::Duration::from_millis(250);
    static HITS: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    static STORED: AtomicU64 = AtomicU64::new(0);
    static KEEP_FIXTURES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    pub(super) fn keep_fixtures() -> bool {
        KEEP_FIXTURES.load(Ordering::Relaxed)
    }

    pub(super) fn force_fixture_retention() {
        KEEP_FIXTURES.store(true, Ordering::Relaxed);
    }

    pub(super) fn counts() -> (u64, u64, u64) {
        (HITS.load(Ordering::Relaxed), STALE.load(Ordering::Relaxed), STORED.load(Ordering::Relaxed))
    }

    static REFERENCE_BLOBS: AtomicBool = AtomicBool::new(false);
    static ACTIVE_KEEPS: Mutex<Option<std::collections::HashSet<PathBuf>>> = Mutex::new(None);

    /// Split pipelines with different root ABIs cannot share a persisted driver blob.
    #[allow(dead_code)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ChainRoot {
        DirectConstants,
        IndirectCbv,
    }

    /// AMD's cached-PSO blobs name entries in the driver's own cache instead of containing
    /// self-sufficient code. They are not safe to carry across runs unless explicitly enabled.
    pub fn note_vendor(vendor: u32) {
        let references = reference_blob_vendor(vendor);
        let was = REFERENCE_BLOBS.fetch_or(references, Ordering::Relaxed);
        if references && !was && !enabled() {
            crate::xlog_info!("D3D12.X: driver-blob disk cache off on AMD (XCL_PSO_DISK_CACHE=1: force on)");
        }
    }

    fn reference_blob_vendor(vendor: u32) -> bool {
        vendor == 0x1002
    }

    /// The file of the pipeline made from `dxil` with `flags`.
    #[cfg(test)]
    pub fn name(dxil: &[u8], flags: i32) -> String {
        name_for(dxil, flags, false)
    }

    /// The legacy empty-sampler name, using `.chain.pso` for the dispatch-chain root signature.
    #[cfg(test)]
    pub fn name_for(dxil: &[u8], flags: i32, chained: bool) -> String {
        name_for_root(dxil, flags, chained, &[])
    }

    /// The cache file for a pipeline whose fixed ABI root-signature layout has these ordered
    /// static samplers. Empty samplers retain the existing name and cache compatibility.
    pub fn name_for_root(dxil: &[u8], flags: i32, chained: bool, samplers: &[StaticSampler]) -> String {
        name_for_root_node(dxil, flags, chained, samplers, 0)
    }

    /// Include the linked-adapter node in persisted driver state. Node zero keeps legacy names.
    pub fn name_for_root_node(dxil: &[u8], flags: i32, chained: bool, samplers: &[StaticSampler], node_mask: u32) -> String {
        let ext = if chained { "chain.pso" } else { "pso" };
        name_for_root_node_with_ext(dxil, flags, samplers, node_mask, ext)
    }

    /// The direct split ABI retains `.chain.pso`; the indirect-only CBV ABI has a separate
    /// namespace so a driver blob can never be offered with the wrong root signature.
    #[allow(dead_code)]
    pub fn name_for_chain_root_node(
        dxil: &[u8], flags: i32, root: ChainRoot, samplers: &[StaticSampler], node_mask: u32,
    ) -> String {
        let ext = match root {
            ChainRoot::DirectConstants => "chain.pso",
            ChainRoot::IndirectCbv => "indirect-chain.pso",
        };
        name_for_root_node_with_ext(dxil, flags, samplers, node_mask, ext)
    }

    fn name_for_root_node_with_ext(
        dxil: &[u8], flags: i32, samplers: &[StaticSampler], node_mask: u32, ext: &str,
    ) -> String {
        let mut base = format!("{:016x}-{:x}-{:x}", crate::gpu::shader::fnv64(dxil), dxil.len(), flags);
        if node_mask != 0 {
            base += &format!("-n{node_mask:x}");
        }
        if samplers.is_empty() {
            return format!("{base}.{ext}");
        }
        let bytes: Vec<u8> = samplers
            .iter()
            .flat_map(|(words, compare)| words.iter().copied().chain([u32::from(*compare)]))
            .flat_map(u32::to_le_bytes)
            .collect();
        format!("{base}-{:016x}-{:x}.{ext}", crate::gpu::shader::fnv64(&bytes), samplers.len())
    }

    /// The file of a translated graphics pipeline keyed by its full state + shader digest.
    pub fn graphics_name(hash: u64, bytes: usize) -> String {
        format!("gfx-{hash:016x}-{bytes:x}.pso")
    }

    fn env_setting() -> Option<bool> {
        static ON: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
        *ON.get_or_init(|| configured(std::env::var("XCL_PSO_DISK_CACHE").ok().as_deref()))
    }

    fn configured(value: Option<&str>) -> Option<bool> {
        value.map(|v| v.trim() != "0")
    }

    fn enabled_with(setting: Option<bool>, reference_blobs: bool) -> bool {
        setting.unwrap_or(!reference_blobs)
    }

    pub(crate) fn enabled() -> bool {
        enabled_with(env_setting(), REFERENCE_BLOBS.load(Ordering::Relaxed))
    }

    #[cfg(test)]
    pub fn path(dxil: &[u8], flags: i32) -> Option<PathBuf> {
        path_for(dxil, flags, false)
    }

    #[cfg(test)]
    pub fn path_for(dxil: &[u8], flags: i32, chained: bool) -> Option<PathBuf> {
        enabled().then(|| crate::gpu::recomp::dxc::cache_dir().join("driver").join(name_for(dxil, flags, chained)))
    }

    pub fn path_for_root(dxil: &[u8], flags: i32, chained: bool, samplers: &[StaticSampler]) -> Option<PathBuf> {
        enabled().then(|| crate::gpu::recomp::dxc::cache_dir().join("driver").join(name_for_root(dxil, flags, chained, samplers)))
    }

    pub fn path_for_root_node(dxil: &[u8], flags: i32, chained: bool, samplers: &[StaticSampler], node_mask: u32) -> Option<PathBuf> {
        enabled().then(|| crate::gpu::recomp::dxc::cache_dir().join("driver").join(name_for_root_node(dxil, flags, chained, samplers, node_mask)))
    }

    pub fn path_for_chain_root_node(
        dxil: &[u8], flags: i32, root: ChainRoot, samplers: &[StaticSampler], node_mask: u32,
    ) -> Option<PathBuf> {
        enabled().then(|| crate::gpu::recomp::dxc::cache_dir().join("driver")
            .join(name_for_chain_root_node(dxil, flags, root, samplers, node_mask)))
    }

    pub fn graphics_path(hash: u64, bytes: usize) -> Option<PathBuf> {
        enabled().then(|| crate::gpu::recomp::dxc::cache_dir().join("driver").join(graphics_name(hash, bytes)))
    }

    /// Bytes read from one exact cache-file generation. Keeping its handle open lets a later
    /// refusal dispose that file object without touching a replacement at the same pathname.
    pub struct LoadedBlob {
        bytes: Arc<Vec<u8>>,
        /// `None` when the blob was readable but the cache entry could not be opened for deletion.
        file: Option<std::fs::File>,
    }

    impl LoadedBlob {
        #[cfg(test)]
        fn bytes(&self) -> &[u8] { &self.bytes }
        pub fn shared(&self) -> Arc<Vec<u8>> { self.bytes.clone() }

        /// Remove the exact file object that supplied these bytes. If that operation is not
        /// available, leave the stale entry rather than remove a pathname that may name another
        /// process's newer replacement.
        pub fn invalidate(self) -> bool {
            use windows_sys::Win32::Storage::FileSystem::{
                FileStandardInfo, GetFileInformationByHandleEx, SetFileInformationByHandle,
                FILE_STANDARD_INFO,
            };
            let Some(file) = self.file else { return false };
            let handle = file.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
            let _stall = crate::stall::span(crate::stall::StallKind::CacheIo);
            if set_disposition_with(|class, info, size| unsafe {
                SetFileInformationByHandle(handle, class, info, size) != 0
            }) {
                return true;
            }
            let mut info = FILE_STANDARD_INFO::default();
            unsafe {
                GetFileInformationByHandleEx(
                    handle, FileStandardInfo, (&mut info as *mut FILE_STANDARD_INFO).cast(),
                    std::mem::size_of_val(&info) as u32,
                ) != 0 && info.DeletePending
            }
        }
    }

    /// Prefer immediate POSIX unlinking, but filesystems which do not implement the extended
    /// disposition class can still mark this exact open file object for deletion on close.
    fn set_disposition_with(mut set: impl FnMut(i32, *const std::ffi::c_void, u32) -> bool) -> bool {
        use windows_sys::Win32::Storage::FileSystem::{
            FileDispositionInfo, FileDispositionInfoEx,
            FILE_DISPOSITION_FLAG_DELETE, FILE_DISPOSITION_FLAG_POSIX_SEMANTICS,
            FILE_DISPOSITION_INFO, FILE_DISPOSITION_INFO_EX,
        };
        let ex = FILE_DISPOSITION_INFO_EX { Flags: FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS };
        if set(FileDispositionInfoEx, (&ex as *const FILE_DISPOSITION_INFO_EX).cast(), std::mem::size_of_val(&ex) as u32) {
            return true;
        }
        let basic = FILE_DISPOSITION_INFO { DeleteFile: true };
        set(FileDispositionInfo, (&basic as *const FILE_DISPOSITION_INFO).cast(), std::mem::size_of_val(&basic) as u32)
    }

    pub fn load(p: &Path) -> Option<LoadedBlob> {
        use windows_sys::Win32::Foundation::GENERIC_READ;
        use windows_sys::Win32::Storage::FileSystem::{
            DELETE, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
        };
        let open = |access| {
            let _stall = crate::stall::span(crate::stall::StallKind::CacheIo);
            std::fs::OpenOptions::new()
                .read(true)
                .access_mode(access)
                .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
                .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
                .open(p)
        };
        let (mut file, invalidatable) = open(GENERIC_READ | DELETE).map(|f| (f, true))
            .or_else(|_| open(GENERIC_READ).map(|f| (f, false))).ok()?;
        let mut bytes = Vec::new();
        {
            let _stall = crate::stall::span(crate::stall::StallKind::CacheIo);
            file.read_to_end(&mut bytes).ok()?;
        }
        (!bytes.is_empty()).then(|| LoadedBlob { bytes: Arc::new(bytes), file: invalidatable.then_some(file) })
    }

    /// A pipeline was made from a kept blob (`ok`), or the driver refused that exact blob: dropped.
    pub fn note(ok: bool, loaded: Option<LoadedBlob>) {
        if ok {
            let n = HITS.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: {n} pipelines made from the driver's blob of an earlier run (kept on disk)");
            }
        } else {
            if let Some(loaded) = loaded {
                let _ = loaded.invalidate();
            }
            if STALE.fetch_add(1, Ordering::Relaxed) == 0 {
                crate::xlog_info!("D3D12.X: a driver blob kept on disk was refused (another driver or adapter?) – such pipelines are compiled again");
            }
        }
    }

    fn create_staging_file(p: &Path) -> std::io::Result<(PathBuf, std::fs::File)> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        create_staging_file_with(p, &NEXT)
    }

    fn create_staging_file_with(p: &Path, next: &AtomicU64) -> std::io::Result<(PathBuf, std::fs::File)> {
        loop {
            let tmp = p.with_extension(format!("tmp{}-{}", std::process::id(), next.fetch_add(1, Ordering::Relaxed)));
            let opened = {
                let _stall = crate::stall::span(crate::stall::StallKind::CacheIo);
                std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp)
            };
            match opened {
                Ok(file) => return Ok((tmp, file)),
                // A prior process with this PID may have left a file behind. It is not ours.
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// `bytes` to `p`, under another name first: a run that is cut short leaves no half file.
    pub fn write(p: &Path, bytes: &[u8]) -> bool {
        use std::io::Write;
        if bytes.is_empty() {
            return false;
        }
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let Ok((tmp, mut file)) = create_staging_file(p) else { return false };
        let written = {
            let _stall = crate::stall::span(crate::stall::StallKind::CacheIo);
            file.write_all(bytes).is_ok()
        };
        drop(file);
        let ok = written && {
            let _stall = crate::stall::span(crate::stall::StallKind::CacheIo);
            std::fs::rename(&tmp, p).is_ok()
        };
        if !ok {
            let _stall = crate::stall::span(crate::stall::StallKind::CacheIo);
            let _ = std::fs::remove_file(&tmp);
        }
        ok
    }

    pub(super) fn worth_keeping(took: std::time::Duration) -> bool {
        took >= KEEP || keep_fixtures()
    }

    /// Exclusive publication turn for one pathname within this process. Different in-memory
    /// identities can share a disk key (notably compute root-signature pointers), so reserve the
    /// pathname before capturing bytes as well as reserving the in-memory key.
    pub(super) struct KeepTurn(PathBuf, std::time::Duration);

    impl KeepTurn {
        pub(super) fn new(p: &Path, took: std::time::Duration) -> Option<Self> {
            if !worth_keeping(took) {
                return None;
            }
            let mut active = ACTIVE_KEEPS.lock().unwrap();
            active.get_or_insert_with(std::collections::HashSet::new).insert(p.to_owned()).then(|| Self(p.to_owned(), took))
        }

        pub(super) fn store(&self, blob: &[u8]) -> bool {
            if write(&self.0, blob) {
                let n = STORED.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: {n} driver blobs kept on disk for the next run (this one: {} ms, {} KB)", self.1.as_millis(), blob.len() >> 10);
                }
                true
            } else {
                false
            }
        }
    }

    impl Drop for KeepTurn {
        fn drop(&mut self) {
            if let Some(active) = ACTIVE_KEEPS.lock().unwrap().as_mut() {
                active.remove(&self.0);
            }
        }
    }

    /// Keep `blob`, of a build that took `took`, for the next run.
    #[cfg(test)]
    pub fn keep(p: &Path, blob: &[u8], took: std::time::Duration) -> bool {
        if let Some(turn) = KeepTurn::new(p, took) {
            turn.store(blob)
        } else {
            false
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn load_bytes(p: &Path) -> Option<Vec<u8>> {
            load(p).map(|b| b.bytes().to_vec())
        }

        struct Directory(PathBuf);
        impl Drop for Directory {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
        }

        #[test]
        fn a_slow_blob_is_saved_after_the_memory_cache_fills() {
            use super::super::{keep_driver_blob, DriverBlobs, COMPUTE_BLOB_MIN_DXIL};
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-full-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let path = dir.join("late.pso");
            let cache = DriverBlobs::new();
            keep_driver_blob(&cache, 4, 1u64, COMPUTE_BLOB_MIN_DXIL, None, KEEP, || Some(b"full".to_vec()));
            keep_driver_blob(&cache, 4, 2u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP, || Some(b"late".to_vec()));
            let guard = cache.blobs.lock().unwrap();
            let (map, total) = guard.as_ref().unwrap();
            assert_eq!((map.len(), *total), (1, 4));
            assert_eq!(map[&1].as_slice(), b"full");
            drop(guard);
            assert_eq!(load_bytes(&path).as_deref(), Some(b"late".as_slice()));
        }

        #[test]
        fn concurrent_same_key_builders_publish_only_the_reserved_result() {
            use super::super::{keep_driver_blob, DriverBlobs, COMPUTE_BLOB_MIN_DXIL};
            use std::sync::atomic::{AtomicUsize, Ordering};
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-race-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let path = dir.join("same.pso");
            let cache = Arc::new(DriverBlobs::new());
            let entered = Arc::new(std::sync::Barrier::new(2));
            let release = Arc::new(std::sync::Barrier::new(2));
            let reads = Arc::new(AtomicUsize::new(0));
            let worker = {
                let (cache, entered, release, reads, path) = (cache.clone(), entered.clone(), release.clone(), reads.clone(), path.clone());
                std::thread::spawn(move || keep_driver_blob(&cache, 0, 7u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP, || {
                    reads.fetch_add(1, Ordering::Relaxed);
                    entered.wait();
                    release.wait();
                    Some(b"reserved".to_vec())
                }))
            };
            entered.wait();
            keep_driver_blob(&cache, 0, 7u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP * 2, || {
                reads.fetch_add(1, Ordering::Relaxed);
                Some(b"duplicate".to_vec())
            });
            release.wait();
            worker.join().unwrap();
            assert_eq!(reads.load(Ordering::Relaxed), 1);
            assert_eq!(load_bytes(&path).as_deref(), Some(b"reserved".as_slice()));
            // The cap refused RAM admission, but a later duplicate still cannot replace the
            // generation whose own duration and bytes were published.
            keep_driver_blob(&cache, 0, 7u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP * 3,
                || Some(b"later".to_vec()));
            assert_eq!(load_bytes(&path).as_deref(), Some(b"reserved".as_slice()));
        }

        #[test]
        fn different_memory_keys_cannot_race_the_same_disk_path() {
            use super::super::{keep_driver_blob, DriverBlobs, COMPUTE_BLOB_MIN_DXIL};
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-path-race-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let path = dir.join("shared.pso");
            let cache = Arc::new(DriverBlobs::new());
            let entered = Arc::new(std::sync::Barrier::new(2));
            let release = Arc::new(std::sync::Barrier::new(2));
            let worker = {
                let (cache, entered, release, path) = (cache.clone(), entered.clone(), release.clone(), path.clone());
                std::thread::spawn(move || keep_driver_blob(&cache, 0, 1u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP, || {
                    entered.wait();
                    release.wait();
                    Some(b"path owner".to_vec())
                }))
            };
            entered.wait();
            keep_driver_blob(&cache, 0, 2u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP,
                || Some(b"contender".to_vec()));
            release.wait();
            worker.join().unwrap();
            assert_eq!(load_bytes(&path).as_deref(), Some(b"path owner".as_slice()));
        }

        #[test]
        fn capture_failure_can_retry_and_rejection_can_replace_the_generation() {
            use super::super::{keep_driver_blob, DriverBlobs, COMPUTE_BLOB_MIN_DXIL};
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-retry-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let path = dir.join("retry.pso");
            let cache = DriverBlobs::new();
            keep_driver_blob(&cache, 0, 9u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP, || None);
            keep_driver_blob(&cache, 0, 9u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP, || Some(b"old".to_vec()));
            assert_eq!(load_bytes(&path).as_deref(), Some(b"old".as_slice()));
            cache.forget(9);
            keep_driver_blob(&cache, 0, 9u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP, || Some(b"fresh".to_vec()));
            assert_eq!(load_bytes(&path).as_deref(), Some(b"fresh".as_slice()));
        }

        #[test]
        fn a_duplicate_duration_is_not_charged_to_an_existing_blob() {
            use super::super::{keep_driver_blob, DriverBlobs, COMPUTE_BLOB_MIN_DXIL};
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-duration-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let path = dir.join("duration.pso");
            let cache = DriverBlobs::new();
            keep_driver_blob(&cache, 16, 3u64, COMPUTE_BLOB_MIN_DXIL, None, KEEP, || Some(b"first".to_vec()));
            keep_driver_blob(&cache, 16, 3u64, COMPUTE_BLOB_MIN_DXIL, Some(&path), KEEP * 2,
                || panic!("an existing blob must not be recaptured"));
            assert!(!path.exists(), "another builder's duration must not persist the first builder's bytes");
        }

        #[test]
        fn a_file_is_named_after_its_dxil_and_flags() {
            assert_eq!(name(b"abc", 0), name(b"abc", 0));
            assert_ne!(name(b"abc", 0), name(b"abd", 0));
            assert_ne!(name(b"abc", 0), name(b"abc", 1));
            // (the length is in the name: two programs of one hash and different lengths differ)
            assert!(name(&[0u8; 0x1234], 2).ends_with("-1234-2.pso"));
            assert_eq!(name_for(b"abc", 0, false), name(b"abc", 0));
            assert_ne!(name_for(b"abc", 0, true), name_for(b"abc", 0, false));
            assert!(name_for(&[0u8; 0x1234], 2, true).ends_with("-1234-2.chain.pso"));
        }

        #[test]
        fn static_sampler_root_signatures_have_distinct_compatible_names() {
            let wrap = ([0, 0, 0, 0], false);
            let clamp_u = ([2, 0, 0, 0], false);
            let comparison = (wrap.0, true);
            let file = |chained, samplers: &[StaticSampler]| name_for_root(b"same dxil", 0, chained, samplers);

            assert_eq!(file(false, &[]), name_for(b"same dxil", 0, false), "empty sampler cache compatibility");
            assert_eq!(file(true, &[]), name_for(b"same dxil", 0, true), "empty chain cache compatibility");
            assert_ne!(file(false, &[]), file(false, &[wrap]));
            assert_ne!(file(false, &[wrap]), file(false, &[clamp_u]));
            assert_ne!(file(false, &[wrap]), file(false, &[comparison]));
            assert_ne!(file(false, &[wrap, clamp_u]), file(false, &[clamp_u, wrap]));
            assert_ne!(file(false, &[wrap]), file(true, &[wrap]));
        }

        #[test]
        fn linked_adapter_nodes_have_distinct_compatible_names() {
            let legacy = name_for_root(b"same dxil", 3, false, &[]);
            assert_eq!(legacy, name_for_root_node(b"same dxil", 3, false, &[], 0));
            assert_ne!(legacy, name_for_root_node(b"same dxil", 3, false, &[], 2));
            assert_ne!(
                name_for_root_node(b"same dxil", 3, false, &[], 1),
                name_for_root_node(b"same dxil", 3, false, &[], 2)
            );
        }

        #[test]
        fn direct_and_indirect_chain_roots_have_distinct_compatible_names() {
            let direct = name_for_chain_root_node(b"same dxil", 3, ChainRoot::DirectConstants, &[], 0);
            assert_eq!(direct, name_for_root_node(b"same dxil", 3, true, &[], 0));
            assert_ne!(direct, name_for_chain_root_node(b"same dxil", 3, ChainRoot::IndirectCbv, &[], 0));
            assert!(name_for_chain_root_node(b"same dxil", 3, ChainRoot::IndirectCbv, &[], 0)
                .ends_with(".indirect-chain.pso"));
        }

        #[test]
        fn disk_blobs_default_off_only_for_amd_and_can_be_overridden() {
            assert!(reference_blob_vendor(0x1002));
            assert!(!reference_blob_vendor(0x10de)); // NVIDIA
            assert!(!reference_blob_vendor(0x8086)); // Intel
            assert!(!reference_blob_vendor(0x1414)); // WARP

            assert!(!enabled_with(None, true));
            assert!(enabled_with(None, false));
            assert!(enabled_with(configured(Some("1")), true));
            assert!(!enabled_with(configured(Some(" 0 ")), false));
            assert_eq!(configured(None), None);
        }

        #[test]
        fn a_kept_blob_is_read_back_and_a_refused_one_is_dropped() {
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-test-{}", std::process::id()));
            let p = dir.join("driver").join(name(b"program", 0));
            assert_eq!(load_bytes(&p), None, "nothing kept yet");
            assert!(!write(&p, &[]), "an empty blob is not kept");
            keep(&p, &[1, 2, 3], std::time::Duration::from_millis(10));
            assert_eq!(load_bytes(&p), None, "a build of milliseconds is not kept");
            keep(&p, &[1, 2, 3], std::time::Duration::from_secs(1));
            assert_eq!(load_bytes(&p).as_deref(), Some(&[1u8, 2, 3][..]));
            // no half file under the temporary name
            assert_eq!(std::fs::read_dir(p.parent().unwrap()).unwrap().count(), 1);
            // the driver took the blob: it stays; it refused it: gone, the program is compiled again
            note(true, load(&p));
            assert!(p.exists());
            note(false, load(&p));
            assert_eq!(load_bytes(&p), None);
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn an_older_process_cannot_remove_a_replacement_blob() {
            const CHILD: &str = "XCL_DRIVER_CACHE_REPLACEMENT_CHILD";
            let bytes = b"same bytes in both generations";
            if let Some(path) = std::env::var_os(CHILD) {
                assert!(write(&PathBuf::from(path), bytes), "replace the still-open old generation directly");
                return;
            }
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-replace-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let p = dir.join("program.pso");
            // Identical bytes make pathname/content comparisons unable to distinguish the two
            // generations: the open file object is the identity that must protect the second.
            assert!(write(&p, bytes));
            let delayed_reader = load(&p).unwrap();
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::real::driver_cache::tests::an_older_process_cannot_remove_a_replacement_blob", "--nocapture"])
                .env(CHILD, &p)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "replacement process failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(delayed_reader.invalidate(), "the old generation can be disposed through its handle");
            assert_eq!(load_bytes(&p).as_deref(), Some(bytes.as_slice()));
        }

        #[test]
        fn a_blob_without_delete_access_fails_closed() {
            let loaded = LoadedBlob { bytes: Arc::new(vec![1]), file: None };
            assert!(!loaded.invalidate());
        }

        #[test]
        fn unsupported_posix_disposition_falls_back_to_exact_handle_deletion() {
            use windows_sys::Win32::Storage::FileSystem::{FileDispositionInfo, FileDispositionInfoEx};
            let mut classes = Vec::new();
            assert!(set_disposition_with(|class, _, _| {
                classes.push(class);
                class == FileDispositionInfo
            }));
            assert_eq!(classes, [FileDispositionInfoEx, FileDispositionInfo]);
        }

        #[test]
        fn a_cache_symlink_never_disposes_its_target() {
            use std::os::windows::fs::symlink_file;
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-link-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let target = dir.join("outside.pso");
            let link = dir.join("cache.pso");
            std::fs::write(&target, b"not the cache entry").unwrap();
            if symlink_file(&target, &link).is_err() {
                return; // creating symlinks can require Developer Mode or an elevated token
            }
            if let Some(loaded) = load(&link) {
                let _ = loaded.invalidate();
            }
            assert_eq!(std::fs::read(&target).unwrap(), b"not the cache entry");
        }

        #[test]
        fn an_overlapping_writer_cannot_modify_the_published_blob() {
            use std::io::Write;
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-overlap-{}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let p = dir.join("program.pso");
            // Another writer, paused after the production staging-file creation step.
            let (_other_tmp, mut other) = create_staging_file(&p).unwrap();
            let complete = b"complete second blob";
            assert!(write(&p, complete));
            assert_eq!(load_bytes(&p).as_deref(), Some(complete.as_slice()));
            // Resuming the older writer must not change a file already published by this
            // invocation, even when the older payload is shorter than the newer one.
            other.write_all(b"old").unwrap();
            assert_eq!(load_bytes(&p).as_deref(), Some(complete.as_slice()), "published bytes must not alias another writer's staging file");
        }

        #[test]
        fn an_abandoned_staging_file_is_skipped_and_preserved() {
            use std::io::Write;
            let dir = std::env::temp_dir().join(format!("xcl-driver-cache-stale-{}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let _directory = Directory(dir.clone());
            let p = dir.join("program.pso");
            let occupied = p.with_extension(format!("tmp{}-0", std::process::id()));
            std::fs::write(&occupied, b"not ours").unwrap();
            let next = AtomicU64::new(0);
            let (tmp, mut file) = create_staging_file_with(&p, &next).unwrap();
            assert_eq!(next.load(Ordering::Relaxed), 2, "the occupied candidate was passed over");
            assert_ne!(tmp, occupied);
            file.write_all(b"ours").unwrap();
            drop(file);
            std::fs::rename(&tmp, &p).unwrap();
            assert_eq!(load_bytes(&p).as_deref(), Some(b"ours".as_slice()));
            assert_eq!(std::fs::read(&occupied).unwrap(), b"not ours");
        }
    }
}

/// A bounded set of driver blobs plus capture/publication ownership for duplicate builders.
struct DriverBlobs<K> {
    blobs: Mutex<Option<(HashMap<K, std::sync::Arc<Vec<u8>>>, usize)>>,
    keeping: Mutex<Option<(HashSet<K>, HashMap<K, std::sync::Weak<Vec<u8>>>)>>,
}

impl<K: Copy + Eq + std::hash::Hash> DriverBlobs<K> {
    const fn new() -> Self {
        Self { blobs: Mutex::new(None), keeping: Mutex::new(None) }
    }

    fn get(&self, key: K) -> Option<std::sync::Arc<Vec<u8>>> {
        self.blobs.lock().unwrap().as_ref().and_then(|(map, _)| map.get(&key).cloned())
    }

    fn reserve(&self, key: K) -> bool {
        let blobs = self.blobs.lock().unwrap();
        if blobs.as_ref().is_some_and(|(map, _)| map.contains_key(&key)) {
            return false;
        }
        let mut guard = self.keeping.lock().unwrap();
        let (active, persisted) = guard.get_or_insert_with(|| (HashSet::new(), HashMap::new()));
        !persisted.contains_key(&key) && active.insert(key)
    }

    /// Admit the exact disk generation accepted by the driver without copying it. Holding both
    /// locks excludes a builder reservation between the presence check and insertion.
    fn admit(&self, key: K, blob: std::sync::Arc<Vec<u8>>, limit: usize) -> bool {
        let mut blobs = self.blobs.lock().unwrap();
        let mut keeping = self.keeping.lock().unwrap();
        let (active, persisted) = keeping.get_or_insert_with(|| (HashSet::new(), HashMap::new()));
        if active.contains(&key) {
            return false;
        }
        let (map, total) = blobs.get_or_insert_with(|| (HashMap::new(), 0));
        if map.contains_key(&key) || blob.len() > limit.saturating_sub(*total) {
            return false;
        }
        *total += blob.len();
        persisted.insert(key, std::sync::Arc::downgrade(&blob));
        map.insert(key, blob);
        true
    }

    fn finish(&self, key: K, blob: Option<std::sync::Arc<Vec<u8>>>, limit: usize, persisted: bool) {
        let mut generation = None;
        if let Some(blob) = blob {
            let mut guard = self.blobs.lock().unwrap();
            let (map, total) = guard.get_or_insert_with(|| (HashMap::new(), 0));
            if !map.contains_key(&key) && blob.len() <= limit.saturating_sub(*total) {
                *total += blob.len();
                map.insert(key, blob);
                generation = map.get(&key).map(std::sync::Arc::downgrade);
            }
        }
        let mut guard = self.keeping.lock().unwrap();
        let (active, stored) = guard.as_mut().unwrap();
        active.remove(&key);
        if persisted {
            stored.insert(key, generation.unwrap_or_default());
        }
    }

    /// Remove only the generation that a rejected lookup actually used. A builder may have
    /// published a replacement under the same key while that driver call was in flight.
    fn forget_if(&self, key: K, expected: &std::sync::Arc<Vec<u8>>) -> bool {
        let removed = if let Some((map, total)) = self.blobs.lock().unwrap().as_mut() {
            if map.get(&key).is_some_and(|blob| std::sync::Arc::ptr_eq(blob, expected)) {
                let blob = map.remove(&key).unwrap();
                *total -= blob.len();
                true
            } else {
                false
            }
        } else {
            false
        };
        if removed {
            if let Some((_, persisted)) = self.keeping.lock().unwrap().as_mut() {
                if persisted.get(&key).and_then(std::sync::Weak::upgrade)
                    .is_some_and(|blob| std::sync::Arc::ptr_eq(&blob, expected))
                {
                    persisted.remove(&key);
                }
            }
        }
        removed
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn forget(&self, key: K) {
        if let Some((map, total)) = self.blobs.lock().unwrap().as_mut() {
            if let Some(blob) = map.remove(&key) {
                *total -= blob.len();
            }
        }
        if let Some((_, persisted)) = self.keeping.lock().unwrap().as_mut() {
            persisted.remove(&key);
        }
    }
}

/// The driver's compiled form of translated compute programs (`GetCachedBlob`),
/// by (DXIL hash, root signature, flags), up to `COMPUTE_BLOBS_MAX` bytes in all.
static COMPUTE_BLOBS: DriverBlobs<(u64, usize, i32)> = DriverBlobs::new();
const COMPUTE_BLOBS_MAX: usize = 512 << 20;
/// smaller programs compile in milliseconds: not kept
const COMPUTE_BLOB_MIN_DXIL: usize = 16 << 10;

#[cfg_attr(not(test), allow(dead_code))]
fn compute_blob_key(dxil: &[u8], rs: usize, flags: i32) -> (u64, usize, i32) {
    compute_blob_key_for(dxil, rs, flags, 0)
}

fn compute_blob_key_for(dxil: &[u8], rs: usize, flags: i32, node_mask: u32) -> (u64, usize, i32) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    dxil.hash(&mut h);
    node_mask.hash(&mut h);
    (h.finish(), rs, flags)
}

fn compute_blob(key: (u64, usize, i32)) -> Option<std::sync::Arc<Vec<u8>>> {
    COMPUTE_BLOBS.get(key)
}

#[cfg_attr(not(test), allow(dead_code))]
fn forget_compute_blob(key: (u64, usize, i32)) {
    COMPUTE_BLOBS.forget(key);
}

fn forget_compute_blob_if(key: (u64, usize, i32), expected: &std::sync::Arc<Vec<u8>>) -> bool {
    COMPUTE_BLOBS.forget_if(key, expected)
}

#[cfg(test)]
mod compute_blob_tests {
    #[test]
    fn recorded_split_selects_cached_whole_or_pieces() {
        use super::*;
        use crate::gpu::recomp::pso::{console_to_dxil_for, Stage};
        let Some(input) = std::env::var_os("XCL_SAMPLE_RUNTIME_COMPUTE") else { return };
        let cache = std::path::PathBuf::from(std::env::var_os("XCL_DXIL_CACHE").expect("private empty replay cache"));
        let mode = std::env::var("XCL_SAMPLE_CACHE_CASE").unwrap();
        assert_eq!(super::super::backend(), super::super::Backend::Real);
        let raw = device_raw();
        let code = std::fs::read(input).unwrap();
        let c = console_to_dxil_for(&code, Stage::Compute, 0, wave32_only(raw)).unwrap();
        assert!(c.dispatch_chain.is_some());
        let path = driver_cache::path(&c.dxil, 0).unwrap();
        assert!(path.starts_with(&cache));
        assert_eq!(path.exists(), mode == "offline", "expected driver cache entry at {}", path.display());
        unsafe {
            let root = abi_rs_for(raw, &c.samplers).unwrap();
            let create: PsoFn = std::mem::transmute(orig_fn(raw, slot_of(raw, "CreateComputePipelineState")));
            let iid = pc_iid(Kind::PipelineState);
            match mode.as_str() {
                "hit" => {
                    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root), CS: bytecode(&c.dxil), ..Default::default()
                    };
                    let mut ptr = std::ptr::null_mut();
                    assert!(create(raw, (&desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC).cast(), &iid, &mut ptr) >= 0);
                    let blob = ID3D12PipelineState::from_raw(ptr).GetCachedBlob().unwrap();
                    assert!(driver_cache::write(&path, std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize())));
                }
                "stale" => assert!(driver_cache::write(&path, b"invalid driver blob")),
                "miss" | "offline" => (),
                _ => panic!("unknown replay case"),
            }
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { CS: bytecode(&code), ..Default::default() };
            let mut out = std::ptr::null_mut();
            let start = std::time::Instant::now();
            let hr = translated_compute_pso_once(raw, create, (&desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC).cast(), &iid, &mut out)
                .expect("recognized compute shader").expect("translated pipeline");
            assert!(hr >= 0 && !out.is_null());
            let whole = LAST_COMPUTE_RS.with(|s| s.get()) == root as usize;
            println!("RUNTIME_CACHE case={mode} whole={whole} elapsed_ms={:.2}", start.elapsed().as_secs_f64()*1000.);
            let expected_whole = matches!(mode.as_str(), "hit" | "offline");
            assert_eq!(whole, expected_whole);
            assert_eq!(path.exists(), expected_whole);
            drop(ID3D12PipelineState::from_raw(out));
        }
    }

    #[test]
    fn a_blob_is_found_only_for_the_same_program_root_signature_and_flags_and_is_forgotten() {
        let dxil = vec![7u8; 1000];
        let key = super::compute_blob_key(&dxil, 0x1000, 0);
        assert_eq!(key, super::compute_blob_key(&dxil.clone(), 0x1000, 0));
        assert_eq!(key, super::compute_blob_key_for(&dxil, 0x1000, 0, 0));
        assert_ne!(key, super::compute_blob_key_for(&dxil, 0x1000, 0, 2));
        assert_ne!(key, super::compute_blob_key(&dxil, 0x2000, 0));
        assert_ne!(key, super::compute_blob_key(&dxil, 0x1000, 1));
        let mut other = dxil.clone();
        other[999] = 8;
        assert_ne!(key, super::compute_blob_key(&other, 0x1000, 0));
        {
            let mut g = super::COMPUTE_BLOBS.blobs.lock().unwrap();
            let (m, total) = g.get_or_insert_with(|| (super::HashMap::new(), 0));
            m.insert(key, std::sync::Arc::new(vec![1, 2, 3]));
            *total += 3;
        }
        assert_eq!(super::compute_blob(key).as_deref().map(|v| v.len()), Some(3));
        let before = super::COMPUTE_BLOBS.blobs.lock().unwrap().as_ref().unwrap().1;
        super::forget_compute_blob(key);
        assert!(super::compute_blob(key).is_none());
        assert_eq!(super::COMPUTE_BLOBS.blobs.lock().unwrap().as_ref().unwrap().1, before - 3);
    }

    #[test]
    fn delayed_rejection_cannot_remove_a_replacement_generation() {
        let cache = std::sync::Arc::new(super::DriverBlobs::new());
        assert!(cache.reserve(7u64));
        cache.finish(7, Some(std::sync::Arc::new(b"old".to_vec())), 32, false);
        let old = cache.get(7).unwrap();
        let release = std::sync::Arc::new(std::sync::Barrier::new(2));
        let worker = {
            let (cache, release, old) = (cache.clone(), release.clone(), old.clone());
            std::thread::spawn(move || {
                release.wait();
                cache.forget_if(7, &old)
            })
        };

        // Another rejection removes the old value and a builder publishes its replacement
        // before the delayed driver call reports its result.
        cache.forget(7);
        assert!(cache.reserve(7));
        let fresh = std::sync::Arc::new(b"fresh".to_vec());
        cache.finish(7, Some(fresh.clone()), 32, false);
        release.wait();
        assert!(!worker.join().unwrap());
        assert!(std::sync::Arc::ptr_eq(&cache.get(7).unwrap(), &fresh));
        assert!(cache.forget_if(7, &fresh), "the exact generation remains invalidatable");
    }

    #[test]
    fn concurrent_disk_admission_counts_one_exact_generation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let cache = std::sync::Arc::new(super::DriverBlobs::new());
        let blob = std::sync::Arc::new(b"accepted".to_vec());
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(9));
        let admitted = std::sync::Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8).map(|_| {
            let (cache, blob, barrier, admitted) =
                (cache.clone(), blob.clone(), barrier.clone(), admitted.clone());
            std::thread::spawn(move || {
                barrier.wait();
                if cache.admit(11u64, blob, 32) {
                    admitted.fetch_add(1, Ordering::Relaxed);
                }
            })
        }).collect();
        barrier.wait();
        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(admitted.load(Ordering::Relaxed), 1);
        let accepted = cache.get(11).unwrap();
        assert!(std::sync::Arc::ptr_eq(&accepted, &blob));
        assert_eq!(cache.blobs.lock().unwrap().as_ref().unwrap().1, blob.len());
        let other = std::sync::Arc::new(b"other".to_vec());
        assert!(!cache.forget_if(11, &other));
        assert_eq!(cache.blobs.lock().unwrap().as_ref().unwrap().1, blob.len());
        assert!(cache.forget_if(11, &accepted));
        assert_eq!(cache.blobs.lock().unwrap().as_ref().unwrap().1, 0);
        assert!(cache.reserve(11), "exact invalidation also releases persisted ownership");
        cache.finish(11, None, 32, false);
    }

    #[test]
    fn active_builder_wins_over_disk_admission() {
        let cache = super::DriverBlobs::new();
        assert!(cache.reserve(5u64));
        assert!(!cache.admit(5, std::sync::Arc::new(b"disk".to_vec()), 32));
        let built = std::sync::Arc::new(b"built".to_vec());
        cache.finish(5, Some(built.clone()), 32, false);
        let kept = cache.get(5).unwrap();
        assert!(std::sync::Arc::ptr_eq(&kept, &built));
        assert_eq!(cache.blobs.lock().unwrap().as_ref().unwrap().1, built.len());
    }
}

/// Read `GetCachedBlob` from either an adopted or raw PC `ID3D12PipelineState`.
unsafe fn read_pso_cached_blob(pso: *mut c_void) -> Option<Vec<u8>> {
    if pso.is_null() {
        return None;
    }
    let blob = if is_adopted(pso) {
        type GetBlob = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT;
        let get: GetBlob = std::mem::transmute(orig_fn(pso, slot_of(pso, "GetCachedBlob")));
        let mut out = std::ptr::null_mut();
        if get(pso, &mut out) < 0 || out.is_null() {
            return None;
        }
        ID3DBlob::from_raw(out)
    } else {
        ID3D12PipelineState::from_raw_borrowed(&pso)?.GetCachedBlob().ok()?
    };
    let size = blob.GetBufferSize();
    let data = blob.GetBufferPointer().cast::<u8>();
    if size == 0 || data.is_null() {
        return None;
    }
    Some(std::slice::from_raw_parts(data, size).to_vec())
}

/// Offer the exact result and duration of one successful build to the bounded RAM cache and,
/// when slow, the disk cache. One same-key owner captures and publishes at a time: another
/// completed builder cannot charge its duration to these bytes or overwrite them afterward.
fn keep_driver_blob<K: Copy + Eq + std::hash::Hash>(
    cache: &DriverBlobs<K>, limit: usize, key: K, dxil_len: usize,
    disk: Option<&std::path::Path>, took: std::time::Duration, read: impl FnOnce() -> Option<Vec<u8>>,
) {
    if (dxil_len < COMPUTE_BLOB_MIN_DXIL && !driver_cache::keep_fixtures()) || !cache.reserve(key) {
        return;
    }
    let keep_turn = disk.and_then(|path| driver_cache::KeepTurn::new(path, took));
    let Some(bytes) = read().filter(|bytes| !bytes.is_empty()) else {
        cache.finish(key, None, limit, false);
        return;
    };
    let blob = std::sync::Arc::new(bytes);
    // Publish to disk before exposing the blob through RAM: a concurrent consumer cannot reject
    // this generation while its staging file is being renamed into place.
    let persisted = keep_turn.as_ref().is_some_and(|turn| turn.store(&blob));
    cache.finish(key, Some(blob), limit, persisted);
}

/// The driver's compiled form of translated graphics pipelines (`GetCachedBlob`),
/// keyed by `(full_state_and_shader_hash, total_dxil_bytes, effective_root_signature)`, up to
/// `GRAPHICS_BLOBS_MAX`. The root address scopes RAM entries to the device; persisted paths are
/// already adapter-scoped and encode the root's static-sampler ABI in the state hash.
static GRAPHICS_BLOBS: DriverBlobs<(u64, usize, usize)> = DriverBlobs::new();
const GRAPHICS_BLOBS_MAX: usize = 256 << 20;

fn graphics_blob(key: (u64, usize, usize)) -> Option<std::sync::Arc<Vec<u8>>> {
    GRAPHICS_BLOBS.get(key)
}

fn forget_graphics_blob_if(key: (u64, usize, usize), expected: &std::sync::Arc<Vec<u8>>) -> bool {
    GRAPHICS_BLOBS.forget_if(key, expected)
}

/// Deterministic FNV-1a digest of every field in a translated PC graphics PSO (shaders, static
/// samplers, blend/rasterizer/depth-stencil state, input layout, render target / depth formats,
/// sample desc, topology, flags) for in-memory and on-disk driver blob caching.
pub(super) unsafe fn graphics_blob_key(
    copy: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    vs_dxil: &[u8],
    ps_dxil: Option<&[u8]>,
    gs_dxil: Option<&[u8]>,
    samplers: &[crate::gpu::recomp::pso::StaticSampler],
) -> (u64, usize) {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let feed = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    let feed_u64 = |h: &mut u64, v: u64| feed(h, &v.to_le_bytes());
    let feed_u32 = |h: &mut u64, v: u32| feed(h, &v.to_le_bytes());
    let feed_u8 = |h: &mut u64, v: u8| feed(h, &[v]);

    feed_u64(&mut h, crate::gpu::shader::fnv64(vs_dxil));
    feed_u64(&mut h, vs_dxil.len() as u64);
    if let Some(ps) = ps_dxil {
        feed_u64(&mut h, crate::gpu::shader::fnv64(ps));
        feed_u64(&mut h, ps.len() as u64);
    } else {
        feed_u64(&mut h, 0);
    }
    if let Some(gs) = gs_dxil {
        feed_u64(&mut h, crate::gpu::shader::fnv64(gs));
        feed_u64(&mut h, gs.len() as u64);
    } else {
        feed_u64(&mut h, 0);
    }
    feed_u64(&mut h, samplers.len() as u64);
    for (w, cmp) in samplers {
        for &x in w {
            feed_u32(&mut h, x);
        }
        feed_u8(&mut h, *cmp as u8);
    }
    // Hash BlendState, RasterizerState, and DepthStencilState field-by-field so struct
    // padding bytes (e.g. D3D12_RENDER_TARGET_BLEND_DESC tail padding and the 2 bytes after
    // DepthStencilState.StencilWriteMask) never cause nondeterministic cache keys.
    let bs = &copy.BlendState;
    feed_u32(&mut h, bs.AlphaToCoverageEnable.0 as u32);
    feed_u32(&mut h, bs.IndependentBlendEnable.0 as u32);
    for rt in &bs.RenderTarget {
        feed_u32(&mut h, rt.BlendEnable.0 as u32);
        feed_u32(&mut h, rt.LogicOpEnable.0 as u32);
        feed_u32(&mut h, rt.SrcBlend.0 as u32);
        feed_u32(&mut h, rt.DestBlend.0 as u32);
        feed_u32(&mut h, rt.BlendOp.0 as u32);
        feed_u32(&mut h, rt.SrcBlendAlpha.0 as u32);
        feed_u32(&mut h, rt.DestBlendAlpha.0 as u32);
        feed_u32(&mut h, rt.BlendOpAlpha.0 as u32);
        feed_u32(&mut h, rt.LogicOp.0 as u32);
        feed_u8(&mut h, rt.RenderTargetWriteMask);
    }
    feed_u32(&mut h, copy.SampleMask);
    let rs = &copy.RasterizerState;
    feed_u32(&mut h, rs.FillMode.0 as u32);
    feed_u32(&mut h, rs.CullMode.0 as u32);
    feed_u32(&mut h, rs.FrontCounterClockwise.0 as u32);
    feed_u32(&mut h, rs.DepthBias as u32);
    feed_u32(&mut h, rs.DepthBiasClamp.to_bits());
    feed_u32(&mut h, rs.SlopeScaledDepthBias.to_bits());
    feed_u32(&mut h, rs.DepthClipEnable.0 as u32);
    feed_u32(&mut h, rs.MultisampleEnable.0 as u32);
    feed_u32(&mut h, rs.AntialiasedLineEnable.0 as u32);
    feed_u32(&mut h, rs.ForcedSampleCount);
    feed_u32(&mut h, rs.ConservativeRaster.0 as u32);
    let ds = &copy.DepthStencilState;
    feed_u32(&mut h, ds.DepthEnable.0 as u32);
    feed_u32(&mut h, ds.DepthWriteMask.0 as u32);
    feed_u32(&mut h, ds.DepthFunc.0 as u32);
    feed_u32(&mut h, ds.StencilEnable.0 as u32);
    feed_u8(&mut h, ds.StencilReadMask);
    feed_u8(&mut h, ds.StencilWriteMask);
    for op in [&ds.FrontFace, &ds.BackFace] {
        feed_u32(&mut h, op.StencilFailOp.0 as u32);
        feed_u32(&mut h, op.StencilDepthFailOp.0 as u32);
        feed_u32(&mut h, op.StencilPassOp.0 as u32);
        feed_u32(&mut h, op.StencilFunc.0 as u32);
    }
    feed_u32(&mut h, copy.InputLayout.NumElements);
    if copy.InputLayout.NumElements > 0 && !copy.InputLayout.pInputElementDescs.is_null() {
        let elems = std::slice::from_raw_parts(copy.InputLayout.pInputElementDescs, copy.InputLayout.NumElements as usize);
        for e in elems {
            if !e.SemanticName.0.is_null() {
                feed(&mut h, std::ffi::CStr::from_ptr(e.SemanticName.0 as *const std::ffi::c_char).to_bytes_with_nul());
            } else {
                feed(&mut h, &[0]);
            }
            feed_u32(&mut h, e.SemanticIndex);
            feed_u32(&mut h, e.Format.0 as u32);
            feed_u32(&mut h, e.InputSlot);
            feed_u32(&mut h, e.AlignedByteOffset);
            feed_u32(&mut h, e.InputSlotClass.0 as u32);
            feed_u32(&mut h, e.InstanceDataStepRate);
        }
    }
    feed_u32(&mut h, copy.IBStripCutValue.0 as u32);
    feed_u32(&mut h, copy.PrimitiveTopologyType.0 as u32);
    feed_u32(&mut h, copy.NumRenderTargets);
    for f in &copy.RTVFormats {
        feed_u32(&mut h, f.0 as u32);
    }
    feed_u32(&mut h, copy.DSVFormat.0 as u32);
    feed_u32(&mut h, copy.SampleDesc.Count);
    feed_u32(&mut h, copy.SampleDesc.Quality);
    feed_u32(&mut h, copy.NodeMask);
    feed_u32(&mut h, copy.Flags.0 as u32);
    let total_bytes = vs_dxil.len() + ps_dxil.map_or(0, <[u8]>::len) + gs_dxil.map_or(0, <[u8]>::len);
    (h, total_bytes)
}

/// Stream-output declarations contain pointer-backed arrays that are not represented in the
/// graphics blob key. Keep those PSOs out of this cache rather than aliasing distinct layouts.
fn graphics_blob_cacheable(requested: bool, desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, gs_readable: bool) -> bool {
    requested
        && desc.StreamOutput.NumEntries == 0
        && desc.StreamOutput.NumStrides == 0
        && (desc.GS.BytecodeLength == 0 || gs_readable)
}

#[cfg(test)]
mod graphics_blob_tests {
    use super::*;

    #[test]
    fn classic_depth_bounds_record_preserves_native_state_and_cache_identity() {
        let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        d.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        d.DepthStencilState.StencilReadMask = 0x5a;
        let stream = depth_bounds_stream(&d);
        let fields = stream_subobjects(&stream.0).unwrap();
        let (_, bytes) = fields.iter().find(|(ty, _)| *ty == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1.0 as u32).unwrap();
        let state = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<D3D12_DEPTH_STENCIL_DESC1>()) };
        assert!(state.DepthBoundsTestEnable.as_bool());
        assert_eq!(state.DepthFunc, d.DepthStencilState.DepthFunc);
        assert_eq!(state.StencilReadMask, 0x5a);
        let key = (123, 456);
        assert_eq!(depth_bounds_key(key, false), key);
        assert_ne!(depth_bounds_key(key, true), key);
    }

    #[test]
    fn graphics_blob_key_tracks_shaders_samplers_and_state() {
        unsafe {
            let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
            d.NumRenderTargets = 1;
            d.RTVFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
            d.SampleDesc.Count = 1;
            let vs = b"vs-dxil-1";
            let ps = b"ps-dxil-1";
            let k0 = graphics_blob_key(&d, vs, Some(ps), None, &[]);
            assert_eq!(k0, graphics_blob_key(&d, vs, Some(ps), None, &[]));
            assert_ne!(k0, graphics_blob_key(&d, b"vs-dxil-2", Some(ps), None, &[]));
            assert_ne!(k0, graphics_blob_key(&d, vs, None, None, &[]));
            assert_ne!(k0, graphics_blob_key(&d, vs, Some(ps), Some(b"gs"), &[]));
            assert_ne!(k0, graphics_blob_key(&d, vs, Some(ps), None, &[([1, 2, 3, 4], false)]));
            let mut d2 = d;
            d2.RTVFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;
            assert_ne!(k0, graphics_blob_key(&d2, vs, Some(ps), None, &[]));
        }
    }

    #[test]
    fn graphics_blob_cache_skips_pointer_backed_stream_output() {
        let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        assert!(graphics_blob_cacheable(true, &d, false));
        assert!(!graphics_blob_cacheable(false, &d, false));
        d.StreamOutput.NumEntries = 1;
        assert!(!graphics_blob_cacheable(true, &d, false));
        d.StreamOutput.NumEntries = 0;
        d.StreamOutput.NumStrides = 1;
        assert!(!graphics_blob_cacheable(true, &d, false));
        d.StreamOutput.NumStrides = 0;
        d.GS.BytecodeLength = 4;
        assert!(!graphics_blob_cacheable(true, &d, false), "an unreadable native GS has no safe key");
        assert!(graphics_blob_cacheable(true, &d, true));
    }

    #[test]
    fn graphics_blob_key_ignores_struct_padding_bytes() {
        unsafe {
            let mut a: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::mem::zeroed();
            let mut b: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::mem::zeroed();
            // Fill every byte of DepthStencilState and BlendState with distinct poison patterns,
            // then populate every semantic field identically so only padding bytes differ.
            std::ptr::write_bytes(&mut a.DepthStencilState as *mut _ as *mut u8, 0xAA, std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC>());
            std::ptr::write_bytes(&mut b.DepthStencilState as *mut _ as *mut u8, 0x55, std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC>());
            std::ptr::write_bytes(&mut a.BlendState as *mut _ as *mut u8, 0xAA, std::mem::size_of::<D3D12_BLEND_DESC>());
            std::ptr::write_bytes(&mut b.BlendState as *mut _ as *mut u8, 0x55, std::mem::size_of::<D3D12_BLEND_DESC>());

            for d in [&mut a, &mut b] {
                d.DepthStencilState.DepthEnable = true.into();
                d.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
                d.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
                d.DepthStencilState.StencilEnable = false.into();
                d.DepthStencilState.StencilReadMask = 0xFF;
                d.DepthStencilState.StencilWriteMask = 0xFF;
                d.DepthStencilState.FrontFace = D3D12_DEPTH_STENCILOP_DESC::default();
                d.DepthStencilState.BackFace = D3D12_DEPTH_STENCILOP_DESC::default();
                d.BlendState.AlphaToCoverageEnable = false.into();
                d.BlendState.IndependentBlendEnable = false.into();
                for rt in &mut d.BlendState.RenderTarget {
                    rt.BlendEnable = false.into();
                    rt.LogicOpEnable = false.into();
                    rt.SrcBlend = D3D12_BLEND_ONE;
                    rt.DestBlend = D3D12_BLEND_ZERO;
                    rt.BlendOp = D3D12_BLEND_OP_ADD;
                    rt.SrcBlendAlpha = D3D12_BLEND_ONE;
                    rt.DestBlendAlpha = D3D12_BLEND_ZERO;
                    rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                    rt.LogicOp = D3D12_LOGIC_OP_NOOP;
                    rt.RenderTargetWriteMask = 0xF;
                }
            }
            let ka = graphics_blob_key(&a, b"vs", Some(b"ps"), None, &[]);
            let kb = graphics_blob_key(&b, b"vs", Some(b"ps"), None, &[]);
            assert_eq!(ka, kb, "padding bytes must not affect graphics_blob_key");

            b.DepthStencilState.StencilWriteMask = 0x7F;
            assert_ne!(ka, graphics_blob_key(&b, b"vs", Some(b"ps"), None, &[]));
        }
    }

    #[test]
    fn stale_cached_graphics_blob_preserves_rect_list_gs_on_retry() {
        const E_INVALIDARG: HRESULT = -2147024809;
        let gs = [1u8; 16];
        let blob = [0xABu8; 32];
        let mut copy = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        copy.GS = bytecode(&gs);
        let mut stale_evicted = 0usize;
        let mut attempts = Vec::new();
        let build = create_graphics_pso_cached(
            &mut copy,
            Some(&blob),
            true,
            || stale_evicted += 1,
            |label, has_cached, f| {
                let hr = f();
                attempts.push((label.to_string(), has_cached, hr));
                (hr, std::time::Duration::from_millis(if has_cached { 300 } else { 2 }))
            },
            |c| {
                // Driver rejects the stale CachedPSO, and accepts the fresh compile with rect GS intact.
                if !c.CachedPSO.pCachedBlob.is_null() {
                    E_INVALIDARG
                } else if !c.GS.pShaderBytecode.is_null() {
                    0
                } else {
                    E_INVALIDARG
                }
            },
        );
        assert_eq!(build.hr, 0);
        assert_eq!(stale_evicted, 1);
        assert!(!build.reused);
        assert!(build.cache_eligible, "fresh compile with rect GS intact should be cached");
        assert_eq!(build.built_in_driver, std::time::Duration::from_millis(2),
            "a rejected cache lookup must not make a fast fresh build eligible for persistence");
        assert!(!copy.GS.pShaderBytecode.is_null(), "stale cached blob must not strip rect GS");
        assert!(copy.CachedPSO.pCachedBlob.is_null());
        assert_eq!(attempts.len(), 2);
        assert_eq!((attempts[0].0.as_str(), attempts[0].1, attempts[0].2), ("graphics", true, E_INVALIDARG));
        assert_eq!((attempts[1].0.as_str(), attempts[1].1, attempts[1].2), ("graphics retry", false, 0));
    }

    #[test]
    fn rect_list_gs_fallback_does_not_cache_under_with_gs_key() {
        const E_INVALIDARG: HRESULT = -2147024809;
        let gs = [2u8; 16];
        let mut copy = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        copy.GS = bytecode(&gs);
        let mut calls = 0usize;
        let build = create_graphics_pso_cached(
            &mut copy,
            None,
            true,
            || panic!("no cached blob to evict"),
            |_, _, f| (f(), std::time::Duration::from_millis(1)),
            |c| {
                calls += 1;
                if c.GS.pShaderBytecode.is_null() { 0 } else { E_INVALIDARG }
            },
        );
        assert_eq!(build.hr, 0);
        assert_eq!(calls, 2);
        assert!(copy.GS.pShaderBytecode.is_null());
        assert!(!build.reused);
        assert!(!build.cache_eligible, "fallback no-GS PSO must not overwrite the with-GS cache key");
    }

    #[test]
    fn valid_cached_graphics_blob_reuses_without_retry() {
        let gs = [3u8; 16];
        let blob = [0xCDu8; 32];
        let mut copy = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        copy.GS = bytecode(&gs);
        let mut calls = 0usize;
        let build = create_graphics_pso_cached(
            &mut copy,
            Some(&blob),
            true,
            || panic!("valid cached blob must not be evicted"),
            |_, has_cached, f| {
                assert!(has_cached);
                (f(), std::time::Duration::from_millis(1))
            },
            |c| {
                calls += 1;
                assert!(!c.CachedPSO.pCachedBlob.is_null());
                assert!(!c.GS.pShaderBytecode.is_null());
                0
            },
        );
        assert_eq!(build.hr, 0);
        assert_eq!(calls, 1);
        assert!(build.reused);
        assert!(!build.cache_eligible);
        assert!(!copy.GS.pShaderBytecode.is_null());
    }

    #[test]
    fn native_stream_drops_stale_cache_before_rect_gs_fallback() {
        let mut with_gs = StreamOut::default();
        with_gs.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, b"vs");
        with_gs.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, b"gs");
        let mut without_gs = StreamOut::default();
        without_gs.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, b"vs");
        let mut stale = 0;
        let mut attempts = Vec::new();
        let build = create_graphics_stream_cached(
            &with_gs,
            Some(&without_gs),
            Some(b"stale blob"),
            || stale += 1,
            |_, cached, stream| {
                let subs = stream_subobjects(&stream.0).unwrap();
                let gs = subs.iter().any(|(ty, _)| *ty == 5);
                let has_blob = subs.iter().any(|(ty, _)| *ty == 19);
                attempts.push((cached, has_blob, gs));
                let hr = if cached || gs { E_INVALIDARG } else { S_OK };
                (hr, std::time::Duration::from_millis(1))
            },
        );
        assert_eq!(build.hr, S_OK);
        assert_eq!(stale, 1);
        assert_eq!(attempts, [(true, true, true), (false, false, true), (false, false, false)]);
        assert!(!build.cache_eligible, "the no-GS fallback has a different identity");
        assert!(!build.used_rect_gs);
    }

    #[test]
    fn native_stream_retries_only_cached_blob_refusals() {
        let mut stream = StreamOut::default();
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, b"vs");
        for failure in [E_OUTOFMEMORY, 0x887a_0005u32 as i32, 0x8000_4005u32 as i32] {
            let (mut calls, mut evictions) = (0, 0);
            let build = create_graphics_stream_cached(
                &stream,
                None,
                Some(b"blob"),
                || evictions += 1,
                |_, _, _| { calls += 1; (failure, std::time::Duration::ZERO) },
            );
            assert_eq!(build.hr, failure);
            assert_eq!((calls, evictions), (1, 0), "{failure:#x} is not evidence of a stale blob");
        }

        for stale in [E_INVALIDARG, 0x887e_0001u32 as i32, 0x887e_0002u32 as i32] {
            let (mut calls, mut evictions) = (0, 0);
            let build = create_graphics_stream_cached(
                &stream,
                None,
                Some(b"blob"),
                || evictions += 1,
                |_, cached, _| {
                    calls += 1;
                    (if cached { stale } else { E_OUTOFMEMORY }, std::time::Duration::ZERO)
                },
            );
            assert_eq!(build.hr, E_OUTOFMEMORY, "the final retry HRESULT is preserved");
            assert_eq!((calls, evictions), (2, 1));
        }
    }
}

struct GraphicsPsoBuild {
    hr: HRESULT,
    built_in_driver: std::time::Duration,
    reused: bool,
    cache_eligible: bool,
}

fn create_graphics_pso_cached(
    copy: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    cached: Option<&[u8]>,
    rect_gs: bool,
    mut on_stale_cache: impl FnMut(),
    mut run: impl FnMut(&str, bool, &mut dyn FnMut() -> HRESULT) -> (HRESULT, std::time::Duration),
    mut create: impl FnMut(&D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> HRESULT,
) -> GraphicsPsoBuild {
    if let Some(b) = cached {
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr() as *const c_void, CachedBlobSizeInBytes: b.len() };
        // Do not strip `rect_gs` while a cached blob is still attached: a stale blob must be
        // discarded first so the uncached retry compiles with `copy.GS` intact.
        let (hr, in_driver) = run("graphics", true, &mut || create(copy));
        if hr >= 0 {
            return GraphicsPsoBuild { hr, built_in_driver: in_driver, reused: true, cache_eligible: false };
        }
        on_stale_cache();
        copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        let (retried, took) = run("graphics retry", false, &mut || retry_without_rect_gs(copy, rect_gs, &mut create));
        let dropped_rect_gs = rect_gs && copy.GS.pShaderBytecode.is_null();
        return GraphicsPsoBuild {
            hr: retried,
            built_in_driver: took,
            reused: false,
            cache_eligible: retried >= 0 && !dropped_rect_gs,
        };
    }
    let (hr, in_driver) = run("graphics", false, &mut || retry_without_rect_gs(copy, rect_gs, &mut create));
    let dropped_rect_gs = rect_gs && copy.GS.pShaderBytecode.is_null();
    GraphicsPsoBuild {
        hr,
        built_in_driver: in_driver,
        reused: false,
        cache_eligible: hr >= 0 && !dropped_rect_gs,
    }
}

unsafe fn translated_graphics_pso(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void, cacheable: bool) -> Option<HRESULT> {
    with_switch_fallback(|| translated_graphics_pso_once(this, f, desc, riid, ppv, cacheable, false))
}

unsafe fn translated_graphics_pso_once(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void, cacheable: bool, depth_bounds: bool) -> Option<HRESULT> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    if !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return None;
    }
    let d = &*(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    let vs = shader_bytes(&d.VS)?;
    if !is_console(vs) {
        return None;
    }
    if shader_bytes(&d.HS).is_some() || shader_bytes(&d.DS).is_some() {
        note_translation_error("PSO with HS/DS – not supported yet".into());
        return None;
    }
    // made when first used (`pso_async::lazy`): not even translated now (covers both VS+PS and GS)
    if super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() {
        return None;
    }
    if let Some(gs) = shader_bytes(&d.GS).filter(|g| is_console(g)) {
        let _ = f;
        return Some(translated_gs_pso(this, d, vs, gs, riid, ppv));
    }
    // a GPU that can run 32-lane waves may run a wave64 VS / PS in waves of 32 (the driver
    // picks; RDNA2 does for pixel shaders): the lanes 32..63 it reads come from `upper_lanes`
    // (split_wave64)
    let split = graphics_wave32(this);
    let vs_c = match console_to_dxil_for(vs, Stage::Vertex, 0, split) {
        Ok(b) => b,
        Err(e) => {
            note_translation_error(format!("VS: {e}"));
            return None;
        }
    };
    let mut samplers = vs_c.samplers;
    let vs_dxil = vs_c.dxil;
    let ps_dxil = match shader_bytes(&d.PS) {
        Some(p) if is_console(p) => match console_to_dxil_for(p, Stage::Pixel, samplers.len() as u32, split) {
            Ok(b) => {
                samplers.extend(b.samplers);
                Some(b.dxil)
            }
            Err(e) => {
                note_translation_error(format!("PS: {e}"));
                return None;
            }
        },
        Some(p) => Some(p.to_vec()),
        None => None,
    };
    let rs = abi_rs_for(this, &samplers)?;
    let mut copy: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read(d);
    copy.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    copy.VS = bytecode(&vs_dxil);
    copy.PS = ps_dxil.as_deref().map(bytecode).unwrap_or_default();
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    // a rect list pipeline: the fourth corner of each rect comes from a generated GS; without
    // it each rect is drawn as the triangle of its three vertices
    let rect_gs = if d.PrimitiveTopologyType.0 == RECT_LIST_TYPE && copy.GS.pShaderBytecode.is_null() {
        rect_list_gs(vs, shader_bytes(&d.PS)).map_err(|e| note_translation_error(format!("rect list drawn as triangles: {e}"))).ok()
    } else {
        None
    };
    if let Some(gs) = &rect_gs {
        copy.GS = bytecode(gs);
    }
    pc_topology_type(&mut copy);
    let ps_hash = shader_bytes(&d.PS).map_or(0, crate::gpu::shader::fnv64);
    if debug_pso_switch("XCL_DEBUG_NO_STENCIL", ps_hash) {
        copy.DepthStencilState.StencilEnable = false.into();
    }
    if debug_pso_switch("XCL_DEBUG_NO_CULL", ps_hash) {
        copy.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    }
    let final_gs = shader_bytes(&copy.GS);
    let bytes = vs_dxil.len() + ps_dxil.as_ref().map_or(0, Vec::len) + final_gs.map_or(0, <[u8]>::len);
    let disk_key = depth_bounds_key(graphics_blob_key(&copy, &vs_dxil, ps_dxil.as_deref(), final_gs, &samplers), depth_bounds);
    let gkey = (disk_key.0, disk_key.1, rs as usize);
    let cacheable = graphics_blob_cacheable(cacheable, &copy, final_gs.is_some());
    let library_name = driver_cache::graphics_name(disk_key.0, disk_key.1);
    let library_pso = if !cacheable || ppv.is_null() {
        None
    } else if depth_bounds {
        let stream = depth_bounds_stream(&copy);
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: stream.0.len(), pPipelineStateSubobjectStream: stream.0.as_ptr() as _ };
        crate::shaders::pipeline_library::load_stream(&library_name, &desc)
    } else {
        crate::shaders::pipeline_library::load_graphics(&library_name, &copy)
    };
    let mut cached = (cacheable && library_pso.is_none()).then(|| graphics_blob(gkey)).flatten();
    let disk = cacheable.then(|| driver_cache::graphics_path(disk_key.0, disk_key.1)).flatten();
    let from_disk = cacheable && library_pso.is_none() && cached.is_none();
    let mut loaded = None;
    if from_disk {
        loaded = disk.as_deref().and_then(driver_cache::load);
        cached = loaded.as_ref().map(driver_cache::LoadedBlob::shared);
    }
    let rejected_generation = cached.clone();
    let build = if let Some(pso) = library_pso {
        *ppv = pso.into_raw();
        GraphicsPsoBuild { hr: S_OK, built_in_driver: std::time::Duration::ZERO, reused: false, cache_eligible: false }
    } else { create_graphics_pso_cached(
        &mut copy,
        cached.as_deref().map(Vec::as_slice),
        rect_gs.is_some(),
        || {
            if let Some(expected) = rejected_generation.as_ref() {
                forget_graphics_blob_if(gkey, expected);
            }
            if from_disk {
                driver_cache::note(false, loaded.take());
            }
        },
        |label, has_cached, f| driver_compile_identified(label, bytes, gate_compute_driver_compile(has_cached), Some(ps_hash), has_cached, f),
        |c| if depth_bounds {
            let stream = depth_bounds_stream(c);
            let desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: stream.0.len(), pPipelineStateSubobjectStream: stream.0.as_ptr() as _ };
            let create: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
            finish_pso(this, create, &desc as *const _ as _, riid, ppv)
        } else { finish_pso(this, f, c as *const _ as *const c_void, riid, ppv) },
    ) };
    let hr = build.hr;
    if build.reused {
        if let Some(blob) = cached.as_ref() {
            crate::shaders::warm_replay::accepted(disk.as_deref(), blob);
        }
    }
    if cacheable && build.cache_eligible {
        keep_driver_blob(&GRAPHICS_BLOBS, GRAPHICS_BLOBS_MAX, gkey, bytes, disk.as_deref(), build.built_in_driver,
            || read_pso_cached_blob(*ppv));
        if let (Some(p), Some(b)) = (disk.as_deref(), graphics_blob(gkey)) {
            crate::shaders::warm_replay::accepted(Some(p), &b);
        }
    } else if build.reused && from_disk {
        GRAPHICS_BLOBS.admit(gkey, cached.as_ref().unwrap().clone(), GRAPHICS_BLOBS_MAX);
        driver_cache::note(true, loaded.take());
    }
    if cacheable && hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        if let Some(pso) = ID3D12PipelineState::from_raw_borrowed(&*ppv) {
            if let Err(e) = crate::shaders::pipeline_library::store(driver_cache::graphics_name(gkey.0, gkey.1), pso) {
                note_translation_error(e);
                return Some(E_INVALIDARG);
            }
        }
    }
    if hr < 0 {
        note_translation_error(format!(
            "the driver rejected the PSO with the translated shaders ({hr:#x}; topology type {}, {} render targets, DSV format {}, HS {}, GS {})",
            copy.PrimitiveTopologyType.0,
            copy.NumRenderTargets,
            copy.DSVFormat.0,
            !copy.HS.pShaderBytecode.is_null(),
            !copy.GS.pShaderBytecode.is_null()
        ));
        return None;
    }
    let n = TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
    note_pso_rs(*ppv, rs);
    super::dred::label(*ppv as usize, &[("vs", Some(vs)), ("ps", shader_bytes(&d.PS))]);
    TRANSLATED_GRAPHICS.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    note_rect_list_gs(*ppv as usize, &copy, rect_gs.is_some());
    if super::present::trace_frame() {
        let ps = shader_bytes(&d.PS).map_or(0, crate::gpu::shader::fnv64);
        PSO_SHADERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, (crate::gpu::shader::fnv64(vs), ps));
        // the input layout (semantic index, format, slot, offset) – what the vertices are read as
        let il = &d.InputLayout;
        let elems: Vec<String> = if il.NumElements > 0 && !il.pInputElementDescs.is_null() {
            std::slice::from_raw_parts(il.pInputElementDescs, il.NumElements as usize)
                .iter()
                .map(|e| format!("{}{} fmt {} slot {} @{}", e.SemanticName.to_string().unwrap_or_default(), e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset as i32))
                .collect()
        } else {
            Vec::new()
        };
        PSO_LAYOUTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(*ppv as usize, elems.join(", "));
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 400 {
            let b = &d.BlendState.RenderTarget[0];
            let z = &d.DepthStencilState;
            crate::xlog_info!(
                "frame trace: PSO vs {:016x} ps {ps:016x}: RTs {} {:?} DSV {} samples {}, depth (enable {}, write {}, func {}), stencil {}, blend0 (enable {}, src {} dst {} op {}, alpha src {} dst {}, mask {:#x}), cull {} fill {} depth clip {}, sample mask {:#x}",
                crate::gpu::shader::fnv64(vs),
                d.NumRenderTargets,
                &d.RTVFormats[..d.NumRenderTargets.min(8) as usize].iter().map(|f| f.0).collect::<Vec<_>>(),
                d.DSVFormat.0,
                d.SampleDesc.Count,
                z.DepthEnable.as_bool(),
                z.DepthWriteMask.0,
                z.DepthFunc.0,
                z.StencilEnable.as_bool(),
                b.BlendEnable.as_bool(),
                b.SrcBlend.0,
                b.DestBlend.0,
                b.BlendOp.0,
                b.SrcBlendAlpha.0,
                b.DestBlendAlpha.0,
                b.RenderTargetWriteMask,
                d.RasterizerState.CullMode.0,
                d.RasterizerState.FillMode.0,
                d.RasterizerState.DepthClipEnable.as_bool(),
                d.SampleMask
            );
            if z.StencilEnable.as_bool() {
                let f = |o: &D3D12_DEPTH_STENCILOP_DESC| (o.StencilFunc.0, o.StencilPassOp.0, o.StencilFailOp.0, o.StencilDepthFailOp.0);
                crate::xlog_info!(
                    "frame trace: … its stencil: read mask {:#x} write mask {:#x}, front (func, pass, fail, depth fail) {:?}, back {:?}",
                    z.StencilReadMask,
                    z.StencilWriteMask,
                    f(&z.FrontFace),
                    f(&z.BackFace)
                );
            }
        }
    }
    if n < 5 || n % 1000 == 0 {
        crate::xlog_info!("D3D12.X: graphics PSO from translated console shaders (#{})", n + 1);
    }
    Some(hr)
}

// ---------------------------------------------------------------------------------------
// non-blocking pipeline creation (see `pso_async`)
// ---------------------------------------------------------------------------------------

pub(in crate::hle::d3d12x) mod compute_build_key;

/// A pipeline description that owns what it points to (shader blobs, input layout and its
/// semantic names), so a worker can build the pipeline after the title's memory is gone.
/// The title's borrowed root signature is cleared; translated paths install XCL's ABI signature.
#[allow(dead_code)] // the vectors only keep the copies alive
enum OwnedDesc {
    Compute(Box<D3D12_COMPUTE_PIPELINE_STATE_DESC>, Vec<Arc<[u8]>>),
    Graphics(Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC>, Vec<Arc<[u8]>>, Vec<D3D12_INPUT_ELEMENT_DESC>, Vec<std::ffi::CString>),
    Stream(Box<[usize; 2]>, Vec<u8>, Vec<Arc<[u8]>>),
}

unsafe impl Send for OwnedDesc {}
// Every pointer in an `OwnedDesc` refers to immutable backing storage owned by the same value.
// The translated builders only read it, so one retained recipe may safely be shared.
unsafe impl Sync for OwnedDesc {}

/// A hash of a pipeline's shaders in their order (never 0): what `pso_async` knows a used
/// pipeline by in the next run.
/// `shaders_key` of the one shader `blob`.
pub(crate) fn shader_key(blob: &[u8]) -> u64 {
    ((0xcbf2_9ce4_8422_2325u64 ^ crate::gpu::shader::fnv64(blob)).wrapping_mul(0x0000_0100_0000_01b3)) | 1
}

pub(crate) fn shaders_key<T: AsRef<[u8]>>(blobs: &[T]) -> u64 {
    shader_set_key(blobs.iter().map(AsRef::as_ref))
}

pub(crate) fn shader_set_key<'a>(blobs: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in blobs {
        h = (h ^ crate::gpu::shader::fnv64(b.as_ref())).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h | 1
}

#[cfg(test)]
mod shaders_key_tests {
    use std::sync::Arc;

    #[test]
    fn the_key_of_one_shader_is_the_key_of_a_set_of_it() {
        let a = vec![1u8, 2, 3];
        assert_eq!(super::shader_key(&a), super::shaders_key(&[a.clone()]));
        assert_ne!(super::shader_key(&a), super::shader_key(&[4u8, 5]));
    }

    #[test]
    fn a_key_follows_the_shaders_and_their_order_and_is_never_zero() {
        let (a, b): (Arc<[u8]>, Arc<[u8]>) = ([1u8, 2, 3].into(), [4u8, 5].into());
        let key = super::shaders_key(&[a.clone(), b.clone()]);
        assert_eq!(key, super::shaders_key(&[a.clone(), b.clone()]));
        assert_ne!(key, super::shaders_key(&[b.clone(), a.clone()]));
        assert_ne!(key, super::shaders_key(&[a.clone()]));
        assert_ne!(key, super::shaders_key(&[a, [4u8, 6].into()]));
        assert_ne!(super::shaders_key(&[] as &[Arc<[u8]>]), 0);
    }
}

impl OwnedDesc {
    /// The key of its shaders, and each shader's (`shader_key`, in its own order).
    fn shaders_key(&self) -> (u64, Vec<u64>) {
        let blobs = match self {
            OwnedDesc::Compute(_, b) => b,
            OwnedDesc::Graphics(_, b, ..) => b,
            OwnedDesc::Stream(_, _, b) => b,
        };
        let key = shaders_key(blobs);
        // each shader by itself too: what a title's shader preparation translates before the
        // next run (`pso_async::earlier_used_shaders`)
        let parts: Vec<u64> = blobs.iter().map(|b| shader_key(b)).collect();
        super::pso_async::shaders_of(key, parts.clone());
        (key, parts)
    }

    /// A graphics pipeline's entry for the next run's preparation (`gfx_records`), of the set
    /// `key` whose shaders' keys are `parts`, or an eligible mesh stream's entry.
    unsafe fn record(&self, key: u64, parts: &[u64]) -> Option<Vec<u8>> {
        if let OwnedDesc::Stream(_, stream, _) = self {
            return record_mesh_stream(stream);
        }
        let OwnedDesc::Graphics(d, ..) = self else { return None };
        // (`own_graphics` copied the stages it has, in this order)
        let mut parts = parts.iter();
        let mut shaders = [0u64; 5];
        for (s, b) in shaders.iter_mut().zip([&d.VS, &d.PS, &d.DS, &d.HS, &d.GS]) {
            if !b.pShaderBytecode.is_null() && b.BytecodeLength != 0 {
                *s = *parts.next()?;
            }
        }
        Some(super::gfx_records::encode(key, shaders, d))
    }

    fn ptr(&self) -> *const c_void {
        match self {
            OwnedDesc::Compute(d, _) => &**d as *const _ as *const c_void,
            OwnedDesc::Graphics(d, ..) => &**d as *const _ as *const c_void,
            OwnedDesc::Stream(d, ..) => &**d as *const _ as *const c_void,
        }
    }
}

/// Copy a shader blob into `blobs` and point `b` at the copy.
unsafe fn own_shader(b: &mut D3D12_SHADER_BYTECODE, blobs: &mut Vec<Arc<[u8]>>) {
    if let Some(bytes) = shader_bytes(b) {
        blobs.push(bytes.into());
        let v = blobs.last().unwrap();
        *b = bytecode(v);
    } else {
        *b = D3D12_SHADER_BYTECODE::default();
    }
}

unsafe fn own_compute(desc: *const c_void) -> Option<OwnedDesc> {
    let mut d: Box<D3D12_COMPUTE_PIPELINE_STATE_DESC> = Box::new(std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC));
    let mut blobs = Vec::new();
    own_shader(&mut d.CS, &mut blobs);
    // Console translation installs its ABI root signature. Do not retain the title's borrowed one.
    d.pRootSignature = std::mem::ManuallyDrop::new(None);
    d.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    Some(OwnedDesc::Compute(d, blobs))
}

/// None with stream output (not copied; such pipelines are built in the call).
unsafe fn own_graphics(desc: *const c_void) -> Option<OwnedDesc> {
    let mut d: Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC> = Box::new(std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    if d.StreamOutput.NumEntries > 0 || d.StreamOutput.NumStrides > 0 {
        return None;
    }
    // Counts are zero, so the pointers are semantically unused; clear them rather than retain
    // borrowed title addresses in an object declared safe to share and possibly reuse.
    d.StreamOutput = D3D12_STREAM_OUTPUT_DESC::default();
    let mut blobs = Vec::new();
    for b in [&mut d.VS, &mut d.PS, &mut d.DS, &mut d.HS, &mut d.GS] {
        own_shader(b, &mut blobs);
    }
    // Console translation installs its ABI root signature. Do not retain the title's borrowed one.
    d.pRootSignature = std::mem::ManuallyDrop::new(None);
    let n = d.InputLayout.NumElements as usize;
    let mut elems = Vec::new();
    let mut names = Vec::new();
    if n > 0 && !d.InputLayout.pInputElementDescs.is_null() {
        elems = std::slice::from_raw_parts(d.InputLayout.pInputElementDescs, n).to_vec();
        for e in &mut elems {
            names.push(std::ffi::CStr::from_ptr(e.SemanticName.0 as *const std::ffi::c_char).to_owned());
            e.SemanticName = PCSTR(names.last().unwrap().as_ptr() as *const u8);
        }
    }
    d.InputLayout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: if elems.is_empty() { std::ptr::null() } else { elems.as_ptr() }, NumElements: elems.len() as u32 };
    d.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    Some(OwnedDesc::Graphics(d, blobs, elems, names))
}

/// A pipeline state stream with its shaders copied. None when the stream is unreadable or
/// truncated, a shader pointer is unreadable, or another subobject points at title memory
/// (input layout, stream output, view instancing).
unsafe fn own_stream(desc: *const c_void) -> Option<OwnedDesc> {
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return None;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let mut buf = std::slice::from_raw_parts(base as *const u8, size).to_vec();
    let mut blobs: Vec<Arc<[u8]>> = Vec::new();
    let mut off = 0usize;
    while off + 8 <= size {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let (psize, palign) = super::null::subobject_payload(ty)?;
        let at = off.checked_add(if palign == 8 { 8 } else { 4 })?;
        let end = at.checked_add(psize)?;
        if end > size {
            return None;
        }
        let word = |buf: &[u8], k: usize| u64::from_le_bytes(buf[at + 8 * k..at + 8 * k + 8].try_into().unwrap());
        match ty {
            // Translated streams install XCL's ABI root signature.
            0 => buf[at..end].fill(0),
            1..=6 | 24 | 25 => {
                let (p, n) = (word(&buf, 0) as usize, word(&buf, 1) as usize);
                if p == 0 || n == 0 {
                    buf[at..at + 16].fill(0);
                } else if crate::gpu::shader::readable(p, n) {
                    blobs.push(std::slice::from_raw_parts(p as *const u8, n).into());
                    buf[at..at + 8].copy_from_slice(&(blobs.last().unwrap().as_ptr() as u64).to_le_bytes());
                } else {
                    return None;
                }
            }
            7 | 12 | 22 if buf[at..end].iter().any(|&b| b != 0) => return None,
            19 => buf[at..at + 16].fill(0),
            _ => {}
        }
        off = end.checked_add(7)? & !7;
    }
    let d = Box::new([buf.len(), buf.as_ptr() as usize]);
    Some(OwnedDesc::Stream(d, buf, blobs))
}

#[cfg(test)]
mod owned_desc_tests {
    use super::*;

    #[test]
    fn graphics_recipe_rejects_buffer_strides_without_declarations() {
        let stride = 16u32;
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        desc.StreamOutput.pBufferStrides = &stride;
        desc.StreamOutput.NumStrides = 1;
        desc.StreamOutput.NumEntries = 0;
        assert!(unsafe { own_graphics((&desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC).cast()) }.is_none());
    }

    #[test]
    fn graphics_recipe_clears_unused_stream_output_pointers() {
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        desc.StreamOutput.pSODeclaration = std::ptr::NonNull::dangling().as_ptr();
        desc.StreamOutput.pBufferStrides = std::ptr::NonNull::dangling().as_ptr();
        let Some(OwnedDesc::Graphics(owned, ..)) = (unsafe {
            own_graphics((&desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC).cast())
        }) else {
            panic!("zero-count stream output must be accepted")
        };
        assert!(owned.StreamOutput.pSODeclaration.is_null());
        assert!(owned.StreamOutput.pBufferStrides.is_null());
    }
}

/// Raw pointers a one-shot worker needs (the existing process-lifetime device invariant).
#[derive(Clone, Copy)]
struct Dev(usize, PsoFn);
unsafe impl Send for Dev {}

/// A gated rebuild recipe owns a device reference independently of the title's references.
#[derive(Clone)]
struct RecipeDev(ID3D12Device, PsoFn);

/// The stand-in for a pipeline with dummy shaders (draws / computes nothing).
unsafe fn standin_graphics(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let d = dummy();
    let mut copy: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    copy.VS = bytecode(&d.vs);
    copy.PS = bytecode(&d.ps);
    copy.DS = D3D12_SHADER_BYTECODE::default();
    copy.HS = D3D12_SHADER_BYTECODE::default();
    copy.GS = D3D12_SHADER_BYTECODE::default();
    copy.StreamOutput = std::mem::zeroed();
    copy.InputLayout = D3D12_INPUT_LAYOUT_DESC::default();
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    pc_topology_type(&mut copy);
    finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv)
}

unsafe fn standin_compute(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let mut copy: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    copy.CS = bytecode(&dummy().cs);
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv)
}

unsafe fn standin_stream(this: *mut c_void, f: PsoFn, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let Some((buf, _)) = fallback_stream(desc as *const u8) else { return E_INVALIDARG };
    let new_desc: [usize; 2] = [buf.len(), buf.as_ptr() as usize];
    finish_pso(this, f, new_desc.as_ptr() as *const c_void, riid, ppv)
}

/// What the stand-in of `desc` shares its driver pipeline by (`standin::Key`): None for a desc
/// that cannot be read. The blend, rasterizer and depth state are not in it – a stand-in draws
/// nothing.
unsafe fn graphics_key(this: *mut c_void, desc: *const c_void) -> Option<super::standin::Key> {
    let d = &*(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    let mut state = Vec::with_capacity(64);
    for w in [
        d.PrimitiveTopologyType.0 as u32,
        d.IBStripCutValue.0 as u32,
        d.NumRenderTargets,
        d.DSVFormat.0 as u32,
        d.SampleDesc.Count,
        d.SampleDesc.Quality,
        d.NodeMask,
        d.Flags.0 as u32,
    ]
    .into_iter()
    .chain(d.RTVFormats.iter().map(|f| f.0 as u32))
    {
        state.extend_from_slice(&w.to_le_bytes());
    }
    Some(super::standin::Key { device: this as usize, root_signature: std::mem::transmute_copy::<_, usize>(&d.pRootSignature), kind: 0, state })
}

unsafe fn compute_key(this: *mut c_void, desc: *const c_void) -> Option<super::standin::Key> {
    let d = &*(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let state = [d.NodeMask, d.Flags.0 as u32].iter().flat_map(|w| w.to_le_bytes()).collect();
    Some(super::standin::Key { device: this as usize, root_signature: std::mem::transmute_copy::<_, usize>(&d.pRootSignature), kind: 1, state })
}

/// `graphics_key` of a pipeline stream: its subobjects with the stand-in's shaders in place of
/// the title's, but the state a stand-in does not need.
unsafe fn stream_key(this: *mut c_void, desc: *const c_void) -> Option<super::standin::Key> {
    let (buf, _) = fallback_stream(desc as *const u8)?;
    let (mut state, mut root_signature) = (Vec::new(), 0usize);
    for (ty, p) in stream_subobjects(&buf)? {
        match ty {
            0 => root_signature = u64::from_le_bytes(p.get(..8)?.try_into().ok()?) as usize,
            // blend, sample mask, rasterizer, depth-stencil (and its later forms), cached blob
            8..=11 | 19 | 21 | 26..=28 => {}
            _ => {
                state.extend_from_slice(&ty.to_le_bytes());
                state.extend_from_slice(p);
            }
        }
    }
    Some(super::standin::Key { device: this as usize, root_signature, kind: 2, state })
}

/// Hand the title `standin`'s result now and build the real pipeline with `build` on a worker.
unsafe fn go_background(
    this: *mut c_void,
    f: PsoFn,
    desc: *const c_void,
    riid: *const GUID,
    ppv: *mut *mut c_void,
    compute: bool,
    owned: OwnedDesc,
    standin: unsafe fn(*mut c_void, PsoFn, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT,
    shared_by: unsafe fn(*mut c_void, *const c_void) -> Option<super::standin::Key>,
    build: unsafe fn(Dev, &OwnedDesc, *mut *mut c_void) -> bool,
) -> Option<HRESULT> {
    let key = if super::standin::enabled() && !ppv.is_null() { shared_by(this, desc) } else { None };
    let mut driver = 0;
    let hr = match key {
        Some(key) => {
            let made = super::standin::shell(key, || {
                let mut out = std::ptr::null_mut();
                let hr = standin(this, f, desc, &pc_iid(Kind::PipelineState), &mut out);
                if hr < 0 || out.is_null() { Err(hr) } else { Ok(out as usize) }
            });
            match made {
                Ok((shell, d)) => {
                    *ppv = shell;
                    driver = d;
                    S_OK
                }
                Err(hr) => hr,
            }
        }
        None => standin(this, f, desc, riid, ppv),
    };
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        static ONCE: AtomicU64 = AtomicU64::new(0);
        if ONCE.fetch_add(1, Ordering::Relaxed) < 3 {
            crate::xlog_warn!("D3D12.X: no stand-in pipeline ({hr:#x}) – building this one in the call");
        }
        return None;
    }
    let dev = Dev(this as usize, f);
    let (key, parts) = owned.shaders_key();
    // (a graphics pipeline used in this run is built before the next starts: `gfx_records`)
    let described = if super::pso_async::lazy() { owned.record(key, &parts) } else { None };
    let build_key = match &owned {
        OwnedDesc::Compute(desc, shaders) => shaders.first().map(|shader| {
            compute_build_key::ComputeBuildKey::capture(this as usize, desc, shader.clone(), wave32_only(this))
        }),
        _ => None,
    };
    // a big compute program that writes nothing but images: binds can go on without it
    let image_only = compute && super::pso_async::image_skip() && matches!(&owned, OwnedDesc::Compute(_, b) if b.first().is_some_and(|b| crate::gpu::effects::big_and_image_only(b)));
    let lazy = super::pso_async::lazy();
    if super::pso_async::rebuild_recipes_enabled() {
        let recipe_dev = RecipeDev(ID3D12Device::from_raw_borrowed(&this)?.clone(), f);
        let recipe = super::pso_async::Recipe::new(move || {
            let dev = Dev(recipe_dev.0.as_raw() as usize, recipe_dev.1);
            run_owned_build(dev, &owned, build)
        });
        super::pso_async::submit_recipe_keyed(*ppv as usize, compute, recipe, !lazy, if lazy { key } else { 0 }, build_key);
    } else {
        let job: super::pso_async::Job = Box::new(move || run_owned_build(dev, &owned, build));
        if lazy {
            super::pso_async::submit_lazy_keyed(*ppv as usize, compute, job, key, build_key);
        } else {
            super::pso_async::submit_eager(*ppv as usize, compute, job, key, build_key);
        }
    }
    if lazy {
        if let Some(entry) = described {
            super::pso_async::on_first_use(*ppv as usize, Box::new(move || super::gfx_records::note(entry)));
        }
    }
    if driver != 0 {
        super::pso_async::shared(*ppv as usize, driver);
    }
    if image_only {
        super::pso_async::image_only(*ppv as usize);
    }
    Some(hr)
}

fn run_owned_build(dev: Dev, owned: &OwnedDesc, build: unsafe fn(Dev, &OwnedDesc, *mut *mut c_void) -> bool) -> usize {
    let mut out: *mut c_void = std::ptr::null_mut();
    if unsafe { build(dev, owned, &mut out) } && !out.is_null() { out as usize } else { 0 }
}

unsafe fn build_graphics(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    matches!(translated_graphics_pso(dev.0 as *mut c_void, dev.1, d.ptr(), &iid, out, true), Some(hr) if hr >= 0)
}

unsafe fn build_graphics_uncacheable(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    matches!(translated_graphics_pso(dev.0 as *mut c_void, dev.1, d.ptr(), &iid, out, false), Some(hr) if hr >= 0)
}

/// This process is a shader preparation worker (`prepare_device`): it keeps the driver's blob of
/// every graphics pipeline it builds on disk, whatever its size and time, and none in memory.
static PREPARING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A shader preparation worker's device (`shaders::graphics_worker`), made on the title's adapter:
/// adopted as the title's is, so `prepare_graphics` takes the title's path. Raw, kept for the
/// process.
pub(crate) fn prepare_device(device: ID3D12Device) -> usize {
    PREPARING.store(true, Ordering::Relaxed);
    let raw = device.into_raw();
    unsafe { adopt(raw, Kind::Device) };
    raw as usize
}

/// A title's graphics pipeline `desc` (`gfx_records::Record::desc`) built on `device`
/// (`prepare_device`) as the title's background build makes it (`build_graphics`): the
/// translation, DXC and driver blob caches on disk get what the title will ask for. The pipeline
/// is released; false: not made (noted in the log as the title's would be).
pub(crate) unsafe fn prepare_graphics(device: usize, desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, depth_bounds: bool) -> bool {
    let this = device as *mut c_void;
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateGraphicsPipelineState")));
    let mut out: *mut c_void = std::ptr::null_mut();
    let hr = with_switch_fallback(|| translated_graphics_pso_once(this, f, desc as *const _ as _, &pc_iid(Kind::PipelineState), &mut out, true, depth_bounds));
    let made = matches!(hr, Some(hr) if hr >= 0) && !out.is_null();
    super::pso_async::release(out as usize);
    made
}

unsafe fn build_compute(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    matches!(translated_compute_pso(dev.0 as *mut c_void, dev.1, d.ptr(), &iid, out), Some(Ok(hr)) if hr >= 0)
}

unsafe fn build_stream(dev: Dev, d: &OwnedDesc, out: *mut *mut c_void) -> bool {
    let iid = pc_iid(Kind::PipelineState);
    let this = dev.0 as *mut c_void;
    let hr = translated_mesh_stream(this, d.ptr(), &iid, out).or_else(|| translated_graphics_stream(this, d.ptr(), &iid, out));
    matches!(hr, Some(hr) if hr >= 0)
}

pub(super) unsafe extern "system" fn dev_create_graphics_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    graphics_pso(this, desc, riid, ppv, false, true)
}

/// `dev_create_graphics_pso`; `stream`: the desc is a pipeline stream's (`stream_pipeline`).
unsafe fn graphics_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void, stream: bool, cacheable: bool) -> HRESULT {
    super::null::capture_graphics_desc(desc as *const u8);
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateGraphicsPipelineState")));
    let (translated, missed) = if super::pso_async::enabled() { attempt(|| translated_graphics_pso(this, f, desc, riid, ppv, cacheable)) } else { (translated_graphics_pso(this, f, desc, riid, ppv, cacheable), false) };
    if missed {
        let build = if cacheable { build_graphics } else { build_graphics_uncacheable };
        if let Some(hr) = own_graphics(desc).and_then(|o| go_background(this, f, desc, riid, ppv, false, o, standin_graphics, graphics_key, build)) {
            return hr;
        }
        if let Some(hr) = translated_graphics_pso(this, f, desc, riid, ppv, cacheable) {
            return hr;
        }
    } else if let Some(hr) = translated {
        return hr;
    }
    let pc = untranslated_stream_desc(desc, stream);
    let hr = finish_pso(this, f, pc.as_ref().map_or(desc, |g| g as *const _ as *const c_void), riid, ppv);
    if hr >= 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return hr;
    }
    let d = dummy();
    let mut copy: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    copy.VS = bytecode(&d.vs);
    copy.PS = bytecode(&d.ps);
    copy.DS = D3D12_SHADER_BYTECODE::default();
    copy.HS = D3D12_SHADER_BYTECODE::default();
    copy.GS = D3D12_SHADER_BYTECODE::default();
    copy.StreamOutput = std::mem::zeroed();
    copy.InputLayout = D3D12_INPUT_LAYOUT_DESC::default();
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    pc_topology_type(&mut copy);
    let hr2 = finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv);
    note_fallback("graphics PSO", hr, hr2 >= 0);
    if hr2 >= 0 {
        hr2
    } else {
        hr
    }
}

unsafe fn note_translated_compute(hr: HRESULT, ppv: *mut *mut c_void) {
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        TRANSLATED_COMPUTE.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    }
}

unsafe extern "system" fn dev_create_compute_pso(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    super::null::capture_compute_desc(desc as *const u8);
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateComputePipelineState")));
    let (mut console, missed) = if super::pso_async::enabled() { attempt(|| translated_compute_pso(this, f, desc, riid, ppv)) } else { (translated_compute_pso(this, f, desc, riid, ppv), false) };
    if missed {
        if let Some(hr) = own_compute(desc).and_then(|o| go_background(this, f, desc, riid, ppv, true, o, standin_compute, compute_key, build_compute)) {
            // a console program the worker is translating (its root signature comes with it);
            // the real PSO registers itself in TRANSLATED_COMPUTE once build_compute succeeds
            // (if build_compute fails, the stand-in stays unregistered so Dispatch /
            // ExecuteIndirect never bind XCL's ABI root signature onto it)
            LAST_COMPUTE_TRANSLATED.with(|c| c.set(true));
            LAST_COMPUTE_RS.with(|c| c.set(0));
            note_console_compute(hr, ppv);
            return hr;
        }
        console = translated_compute_pso(this, f, desc, riid, ppv);
    }
    LAST_COMPUTE_TRANSLATED.with(|c| c.set(matches!(console, Some(Ok(hr)) if hr >= 0)));
    if let Some(Ok(hr)) = console {
        if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() && super::pso_manifest::enabled() {
            let d = &*(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
            if let Some(cs) = shader_bytes(&d.CS) {
                super::pso_manifest::register_ready_compute(*ppv as usize, shaders_key(&[cs]));
            }
        }
        return hr;
    }
    // a console shader the recompiler could not take must not reach the driver: AMD's shader
    // compiler crashes on the stripped DXIL (NOBC bitcode) instead of rejecting it
    let hr = if console.is_some() { E_INVALIDARG } else { finish_pso(this, f, desc, riid, ppv) };
    if hr >= 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return hr;
    }
    let mut copy: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    copy.CS = bytecode(&dummy().cs);
    copy.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
    let hr2 = finish_pso(this, f, &copy as *const _ as *const c_void, riid, ppv);
    note_fallback("compute PSO", hr, hr2 >= 0);
    if console.is_some() {
        note_console_compute(hr2, ppv);
    }
    if hr2 >= 0 {
        hr2
    } else {
        hr
    }
}

// ---------------------------------------------------------------------------------------
// console geometry stages in two passes (MS / NGG, legacy GS): see `pso::TwoPass`
// ---------------------------------------------------------------------------------------

/// What XCL keeps for a mesh pipeline built from console MS / GS code: the title holds the
/// mesh pipeline state (pass-through MS + PS); the draw first runs `compute` (the console
/// program, one record per thread group into `res`), then the mesh pipeline over the records.
pub struct TwoPassPso {
    /// compute pipeline state, owned here, and its root signature (static samplers)
    pub compute: usize,
    pub compute_rs: usize,
    /// root signature of the mesh pipeline (the title's PSO; its PS's static samplers)
    pub mesh_rs: usize,
    pub record: crate::gpu::recomp::MsInterface,
    /// DispatchMesh groups per record (see `pso::TwoPass`)
    pub mesh_groups_per_record: u32,
    /// "MS" or "GS"
    pub kind: &'static str,
}

impl Drop for TwoPassPso {
    fn drop(&mut self) {
        // the compute pipeline state is XCL's (the root signatures are cached or the title's)
        super::pso_async::release(self.compute);
    }
}

static TWO_PASS: Mutex<Option<HashMap<usize, std::sync::Arc<TwoPassPso>>>> = Mutex::new(None);

/// The compute pass of a pipeline state XCL built from console MS / GS code.
pub fn two_pass_of(pso: usize) -> Option<std::sync::Arc<TwoPassPso>> {
    let pso = super::pso_async::resolve(pso);
    two_pass_of_exact(pso)
}

pub(super) fn two_pass_of_exact(pso: usize) -> Option<std::sync::Arc<TwoPassPso>> {
    TWO_PASS.lock().unwrap().as_ref()?.get(&pso).cloned()
}

pub fn two_pass_count() -> usize {
    TWO_PASS.lock().unwrap().as_ref().map_or(0, |m| m.len())
}

/// Lets an isolated integration-test process seed the disk cache with small fixtures.
#[doc(hidden)]
pub fn force_driver_cache_fixture_retention() {
    driver_cache::force_fixture_retention();
}

/// Disk hits, rejected blobs, and successful writes in this process.
#[doc(hidden)]
pub fn driver_cache_stats() -> (u64, u64, u64) {
    driver_cache::counts()
}

#[cfg(test)]
pub(super) fn register_two_pass_for_test(pso: usize, kind: &'static str) {
    if pso != 0 {
        TWO_PASS.lock().unwrap().get_or_insert_with(HashMap::new).insert(
            pso,
            std::sync::Arc::new(TwoPassPso {
                compute: 0,
                compute_rs: 0,
                mesh_rs: 0,
                record: crate::gpu::recomp::MsInterface { outputs: Vec::new(), threads: 64, max_vertices: 64, max_primitives: 64 },
                mesh_groups_per_record: 1,
                kind,
            }),
        );
    }
}

/// Subobject types whose payload is a `D3D12_SHADER_BYTECODE`.
const SHADER_SUBOBJECTS: [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE; 8] = [
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
];

/// A pipeline state stream being written: subobjects aligned as D3D12 expects
/// (type, payload at 4 or 8, size rounded to 8).
#[derive(Clone, Default)]
struct StreamOut(Vec<u8>, usize);

impl StreamOut {
    fn raw(&mut self, ty: u32, payload: &[u8], align8: bool) {
        let at = self.0.len();
        self.0.extend_from_slice(&ty.to_le_bytes());
        self.0.resize(at + if align8 { 8 } else { 4 }, 0);
        self.0.extend_from_slice(payload);
        if SHADER_SUBOBJECTS.iter().any(|t| t.0 as u32 == ty) {
            // a D3D12_SHADER_BYTECODE: pointer, then length
            self.1 += payload.get(8..16).and_then(|b| b.try_into().ok()).map_or(0, usize::from_le_bytes);
        }
        let end = (self.0.len() + 7) & !7;
        self.0.resize(end, 0);
    }
    fn put<T>(&mut self, ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, v: &T) {
        let b = unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) };
        self.raw(ty.0 as u32, b, std::mem::align_of::<T>() == 8);
    }
    fn shader(&mut self, ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, code: &[u8]) {
        self.put(ty, &D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as *const c_void, BytecodeLength: code.len() });
    }
    /// Bytes of shader code referenced by this stream. A shader subobject contains only a
    /// pointer and length; `self.0.len()` therefore does not include the code itself.
    fn shader_bytes(&self) -> usize {
        self.1
    }
}

fn depth_bounds_key(mut key: (u64, usize), enabled: bool) -> (u64, usize) {
    if enabled {
        for &byte in b"DB01" { key.0 = (key.0 ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3); }
    }
    key
}

/// The richer depth-stencil descriptor can only be passed through a pipeline stream.
fn depth_bounds_stream(d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> StreamOut {
    let mut out = StreamOut::default();
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &d.pRootSignature);
    for (stage, code) in [(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &d.VS),
        (D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &d.PS), (D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS, &d.DS),
        (D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS, &d.HS), (D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, &d.GS)] {
        if code.BytecodeLength != 0 { out.put(stage, code); }
    }
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, &super::gfx_records::depth_stencil1(&d.DepthStencilState, true));
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &d.BlendState);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &d.RasterizerState);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &d.SampleMask);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &d.SampleDesc);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
        &D3D12_RT_FORMAT_ARRAY { NumRenderTargets: d.NumRenderTargets, RTFormats: d.RTVFormats });
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &d.DSVFormat);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &d.PrimitiveTopologyType);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE, &d.IBStripCutValue);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, &d.InputLayout);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT, &d.StreamOutput);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK, &d.NodeMask);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS, &d.Flags);
    if d.CachedPSO.CachedBlobSizeInBytes != 0 { out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO, &d.CachedPSO); }
    out
}

/// The subobjects of a stream: (type, payload).
fn stream_subobjects(buf: &[u8]) -> Option<Vec<(u32, &[u8])>> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let (psize, palign) = super::null::subobject_payload(ty)?;
        let poff = if palign == 8 { 8 } else { 4 };
        out.push((ty, buf.get(off + poff..off + poff + psize)?));
        off += (poff + psize + 7) & !7;
    }
    Some(out)
}

/// The stream's subobjects up to the first of a type XCL does not know (logged): a console
/// shader in front of it must still be found.
// Checked parsing and extended rasterizer projection adapted from PR #2621 by Decodere.
fn known_subobjects_checked(buf: &[u8]) -> (Vec<(u32, &[u8])>, bool) {
    let mut out = Vec::new();
    let mut off = 0usize;
    loop {
        if off == buf.len() {
            return (out, true);
        }
        let Some(header) = off.checked_add(8) else { return (out, false) };
        if header > buf.len() {
            return (out, false);
        }
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let Some((psize, palign)) = super::null::subobject_payload(ty) else {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: pipeline stream: unknown subobject type {ty} at {off:#x} of {:#x} - the rest is not read", buf.len());
            }
            return (out, false);
        };
        let poff = if palign == 8 { 8 } else { 4 };
        let Some(start) = off.checked_add(poff) else { return (out, false) };
        let Some(end) = start.checked_add(psize) else { return (out, false) };
        let Some(p) = buf.get(start..end) else { return (out, false) };
        out.push((ty, p));
        let Some(next) = end.checked_add(7).map(|n| n & !7) else { return (out, false) };
        if next > buf.len() {
            return (out, false);
        }
        off = next;
    }
}

fn known_subobjects(buf: &[u8]) -> Vec<(u32, &[u8])> {
    known_subobjects_checked(buf).0
}

fn stream_node_mask(subobjects: &[(u32, &[u8])]) -> u32 {
    subobjects
        .iter()
        .find(|(ty, payload)| *ty == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK.0 as u32 && payload.len() >= 4)
        .map_or(0, |(_, payload)| u32::from_le_bytes(payload[..4].try_into().unwrap()))
}

unsafe fn payload_shader<'a>(p: &[u8]) -> Option<&'a [u8]> {
    let b = D3D12_SHADER_BYTECODE { pShaderBytecode: u64::from_le_bytes(p[0..8].try_into().ok()?) as usize as *const c_void, BytecodeLength: u64::from_le_bytes(p[8..16].try_into().ok()?) as usize };
    shader_bytes(&b)
}

/// Keep the source states the mesh translator will use again during preparation.
unsafe fn record_mesh_stream(stream: &[u8]) -> Option<Vec<u8>> {
    let subs = stream_subobjects(stream)?;
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
    let mut layout = Vec::with_capacity(subs.len());
    let (mut ms, mut amplification, mut ps) = (None, None, None);
    let mut depth_bounds = false;
    let (mut seen, mut size) = (0u32, 0usize);

    fn value<T: Copy>(payload: &[u8]) -> Option<T> {
        (payload.len() == std::mem::size_of::<T>()).then(|| unsafe { std::ptr::read_unaligned(payload.as_ptr().cast()) })
    }

    for (ty, payload) in subs {
        let (_, align) = super::null::subobject_payload(ty)?;
        size += (align + payload.len() + 7) & !7;
        let bit = 1u32 << ty;
        if seen & bit != 0 {
            return None;
        }
        seen |= bit;
        match ty {
            0 => {} // The translator supplies its own root signature.
            19 => continue, // The title's cache blob is never an input to translation.
            24 | 25 if payload.iter().all(|&b| b == 0) => continue,
            1..=5 if payload.iter().all(|&b| b == 0) => {}
            2 => ps = Some(payload_shader(payload)?),
            24 => amplification = Some(payload_shader(payload)?),
            25 => ms = Some(payload_shader(payload)?),
            8 => desc.BlendState = value(payload)?,
            9 => desc.SampleMask = value(payload)?,
            10 => desc.RasterizerState = value(payload)?,
            11 => desc.DepthStencilState = value(payload)?,
            12 if value::<D3D12_INPUT_LAYOUT_DESC>(payload).is_some_and(|d| d.NumElements == 0 && d.pInputElementDescs.is_null()) => {}
            21 => {
                let d = value::<D3D12_DEPTH_STENCIL_DESC1>(payload)?;
                depth_bounds = d.DepthBoundsTestEnable.as_bool();
                desc.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: d.DepthEnable, DepthWriteMask: d.DepthWriteMask, DepthFunc: d.DepthFunc,
                    StencilEnable: d.StencilEnable, StencilReadMask: d.StencilReadMask, StencilWriteMask: d.StencilWriteMask,
                    FrontFace: d.FrontFace, BackFace: d.BackFace,
                };
            }
            13 => desc.IBStripCutValue = value(payload)?,
            14 => desc.PrimitiveTopologyType = value(payload)?,
            15 => {
                let rt: D3D12_RT_FORMAT_ARRAY = value(payload)?;
                if rt.NumRenderTargets > 8 {
                    return None;
                }
                desc.NumRenderTargets = rt.NumRenderTargets;
                desc.RTVFormats = rt.RTFormats;
            }
            16 => desc.DSVFormat = value(payload)?,
            17 => desc.SampleDesc = value(payload)?,
            18 => desc.NodeMask = value(payload)?,
            20 => desc.Flags = value(payload)?,
            _ => return None,
        }
        layout.push(ty);
    }
    if size != stream.len() || (ms.is_some() && amplification.is_some()) {
        return None;
    }
    let ms = ms.or(amplification)?;
    if !crate::gpu::xbox::parse(ms).is_some_and(|shader| shader.stage == "MS") {
        return None;
    }
    let shaders = [shader_key(ms), ps.map_or(0, shader_key)];
    let key = shader_set_key(std::iter::once(ms).chain(ps));
    super::gfx_records::encode_mesh(key, shaders, &layout, &desc, depth_bounds)
}

/// Build both recorded mesh passes on the adopted device from `prepare_device`.
///
/// # Safety
/// `device` must be the live adopted device returned from `prepare_device`.
pub(crate) unsafe fn replay_mesh_record(
    device: usize,
    record: &super::gfx_records::MeshRecord,
    shaders: [Option<&[u8]>; 2],
) -> Result<(), String> {
    let [Some(ms), ps] = shaders else { return Err("mesh shader missing from the job".into()) };
    if record.shaders != [shader_key(ms), ps.map_or(0, shader_key)] ||
        record.key != shader_set_key(std::iter::once(ms).chain(ps))
    {
        return Err("mesh shaders do not match the recorded keys".into());
    }

    let d = record.state();
    let mut stream = StreamOut::default();
    for &ty in &record.layout {
        let subobject = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE(ty as i32);
        match ty {
            0 => stream.put(subobject, &std::ptr::null::<c_void>()),
            2 => match ps {
                Some(bytes) => stream.shader(subobject, bytes),
                None => stream.put(subobject, &D3D12_SHADER_BYTECODE::default()),
            },
            1 | 3..=5 => stream.put(subobject, &D3D12_SHADER_BYTECODE::default()),
            24 | 25 => stream.shader(subobject, ms),
            8 => stream.put(subobject, &d.BlendState),
            9 => stream.put(subobject, &d.SampleMask),
            10 => stream.put(subobject, &d.RasterizerState),
            11 => stream.put(subobject, &d.DepthStencilState),
            12 => stream.put(subobject, &D3D12_INPUT_LAYOUT_DESC::default()),
            21 => stream.put(subobject, &super::gfx_records::depth_stencil1(&d.DepthStencilState, record.depth_bounds)),
            13 => stream.put(subobject, &d.IBStripCutValue),
            14 => stream.put(subobject, &d.PrimitiveTopologyType),
            15 => stream.put(subobject, &D3D12_RT_FORMAT_ARRAY { RTFormats: d.RTVFormats, NumRenderTargets: d.NumRenderTargets }),
            16 => stream.put(subobject, &d.DSVFormat),
            17 => stream.put(subobject, &d.SampleDesc),
            18 => stream.put(subobject, &d.NodeMask),
            20 => stream.put(subobject, &d.Flags),
            _ => return Err(format!("unsupported mesh subobject {ty}")),
        }
    }
    let desc = [stream.0.len(), stream.0.as_ptr() as usize];
    let mut out = std::ptr::null_mut();
    let hr = unsafe {
        translated_mesh_stream(device as *mut c_void, desc.as_ptr().cast(), &pc_iid(Kind::PipelineState), &mut out)
    };
    super::pso_async::release(out as usize);
    match hr {
        Some(hr) if hr >= 0 && !out.is_null() => Ok(()),
        Some(hr) => Err(format!("mesh pipeline replay failed ({hr:#x})")),
        None => Err("mesh pipeline record could not be translated".into()),
    }
}

/// The passthrough mesh shader emits triangles for both translated MS and GS pipelines.
fn two_pass_mesh_stream(rs: *mut c_void, ms: &[u8], ps: Option<&[u8]>) -> StreamOut {
    let mut out = StreamOut::default();
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs);
    out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms);
    if let Some(ps) = ps {
        out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps);
    }
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    out
}

#[cfg(test)]
#[test]
fn passthrough_mesh_stream_accepts_a_triangle_driver_blob() {
    use crate::gpu::recomp::{dxc, runner};
    if dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }
    if super::backend() != super::Backend::Real {
        return runner::skip_test("no GPU");
    }
    let raw = device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap();
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    if unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS7).cast(), std::mem::size_of_val(&options) as u32) }.is_err()
        || options.MeshShaderTier.0 < D3D12_MESH_SHADER_TIER_1.0
    {
        return runner::skip_test("no mesh shaders");
    }
    let (ms, _) = dummy_mesh().expect("compile a synthetic triangle mesh shader");
    unsafe {
        let mut sig = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut sig, None).unwrap();
        let sig = sig.unwrap();
        let rs: ID3D12RootSignature = dev.CreateRootSignature(
            0, std::slice::from_raw_parts(sig.GetBufferPointer() as *const u8, sig.GetBufferSize()),
        ).unwrap();
        let create: PsoFn = std::mem::transmute(orig_fn(raw, slot_of(raw, "CreatePipelineState")));
        let build = |stream: &StreamOut| {
            let desc = [stream.0.len(), stream.0.as_ptr() as usize];
            let mut pso = std::ptr::null_mut();
            let hr = create(raw, desc.as_ptr().cast(), &pc_iid(Kind::PipelineState), &mut pso);
            assert!(hr >= 0 && !pso.is_null(), "mesh PSO creation failed ({hr:#x})");
            ID3D12PipelineState::from_raw(pso)
        };

        let mut reference = StreamOut::default();
        reference.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs.as_raw());
        reference.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms);
        reference.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
        let blob = build(&reference).GetCachedBlob().unwrap();
        let mut translated = two_pass_mesh_stream(rs.as_raw(), ms, None);
        let subobjects = stream_subobjects(&translated.0).unwrap();
        let topology = subobjects.iter().find(|(kind, _)| *kind == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY.0 as u32);
        assert_eq!(topology.map(|(_, payload)| *payload), Some(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0.to_le_bytes().as_slice()));
        translated.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO, &D3D12_CACHED_PIPELINE_STATE {
            pCachedBlob: blob.GetBufferPointer(),
            CachedBlobSizeInBytes: blob.GetBufferSize(),
        });
        build(&translated);
    }
}

#[allow(dead_code)] // Two-pass identities are consumed by the next stacked cache stage.
mod two_pass_identity;

/// The console PS of a two-pass pipeline (a PC PS passes as it is).
/// The PS of a two-pass pipeline (DXIL, static samplers – a PC PS passes as it is).
fn two_pass_ps(ps: Option<&[u8]>, split_wave64: bool) -> Result<(Option<Vec<u8>>, Vec<crate::gpu::recomp::pso::StaticSampler>), String> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    match ps {
        Some(p) if is_console(p) => console_to_dxil_for(p, Stage::Pixel, 0, split_wave64).map(|c| (Some(c.dxil), c.samplers)).map_err(|e| format!("PS: {e}")),
        Some(p) => Ok((Some(p.to_vec()), Vec::new())),
        None => Ok((None, Vec::new())),
    }
}

fn two_pass_driver_id(device: *mut c_void) -> Option<u64> {
    use windows::Win32::Graphics::Dxgi::*;
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&device) }?;
    let luid = unsafe { dev.GetAdapterLuid() };
    let key = (luid.HighPart, luid.LowPart);
    static IDS: OnceLock<Mutex<HashMap<(i32, u32), Option<u64>>>> = OnceLock::new();
    let mut ids = IDS.get_or_init(Default::default).lock().unwrap();
    *ids.entry(key).or_insert_with(|| unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter = factory.EnumAdapterByLuid(luid).ok()?;
        let desc = adapter.GetDesc().ok()?;
        let version = adapter.CheckInterfaceSupport(&IDXGIDevice::IID).ok()?;
        let mut h = two_pass_identity::Fnv64::new();
        h.u32(desc.VendorId);
        h.u32(desc.DeviceId);
        h.u32(desc.SubSysId);
        h.u32(desc.Revision);
        h.u64(version as u64);
        Some(h.finish())
    })
}

pub(crate) fn preparation_driver_identity(dev: &ID3D12Device) -> Option<u64> {
    two_pass_driver_id(dev.as_raw())
}

/// Create the mesh pipeline (RS and shaders already in the stream) and its compute pass.
#[allow(clippy::too_many_arguments)]
unsafe fn finish_two_pass(this: *mut c_void, stream: &StreamOut, tp: crate::gpu::recomp::pso::TwoPass, mesh_rs: *mut c_void, ps_samplers: &[crate::gpu::recomp::pso::StaticSampler], node_mask: u32, kind: &'static str, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() {
        return E_INVALIDARG;
    }
    *ppv = std::ptr::null_mut();
    let Some(rs) = abi_rs_for(this, &tp.samplers) else { return E_INVALIDARG };
    let mesh_bytes = stream.shader_bytes();
    let disk_key = (mesh_bytes != 0).then(|| two_pass_driver_id(this)).flatten().and_then(|driver| {
        two_pass_identity::two_pass_stream_key(&stream.0, ps_samplers, driver, crate::gpu::recomp::pso::mem_windows())
    });
    let gkey = disk_key.map(|(hash, bytes)| (hash, bytes, mesh_rs as usize));
    let gdisk = disk_key.and_then(|(hash, bytes)| driver_cache::graphics_path(hash, bytes));
    let stream_desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: stream.0.len(), pPipelineStateSubobjectStream: stream.0.as_ptr() as *mut c_void };
    let library_mesh = disk_key.and_then(|(hash, bytes)| {
        crate::shaders::pipeline_library::load_stream(&driver_cache::graphics_name(hash, bytes), &stream_desc)
    });
    let mesh_from_library = library_mesh.is_some();
    let mut gcached = if mesh_from_library { None } else { gkey.and_then(graphics_blob) };
    let gfrom_disk = !mesh_from_library && gcached.is_none();
    let mut gloaded = None;
    if gfrom_disk {
        gloaded = gdisk.as_deref().and_then(driver_cache::load);
        gcached = gloaded.as_ref().map(driver_cache::LoadedBlob::shared);
    }
    let rejected_generation = gcached.clone();

    let bytes = mesh_bytes + tp.cs.len();
    let huge_bytes = huge_pso_bytes();
    let medium = bytes < huge_bytes
        && medium_compute_compile("compute", tp.cs.len(), huge_bytes, ADAPTER_VENDOR.load(Ordering::Acquire));
    let gate = (bytes >= huge_bytes || medium) && std::thread::current().name().is_some_and(|n| n.starts_with("xcl-pso"));
    let mut slot = None;
    let mut take_cold_slot = || {
        if gate && slot.is_none() {
            let waited = std::time::Instant::now();
            let gate_stall = crate::stall::span(crate::stall::StallKind::DriverCompileGate);
            let slots = if medium { &*MEDIUM_COMPILES } else { compile_slots(bytes) };
            slot = Some(big_compile_turn(slots, || big_compile_commit_free(big_compile_headroom(bytes)), super::pso_async::build_is_wanted));
            drop(gate_stall);
            crate::xlog_info!("D3D12.X: two-pass {kind} PSO ({bytes} B DXIL) gated, waited {} ms", waited.elapsed().as_millis());
        }
    };
    let fs: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
    let create_mesh = |label, blob: Option<&[u8]>| {
        let mut cached_stream;
        let stream = if let Some(blob) = blob {
            cached_stream = StreamOut(stream.0.clone(), stream.1);
            cached_stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO, &D3D12_CACHED_PIPELINE_STATE {
                pCachedBlob: blob.as_ptr().cast(), CachedBlobSizeInBytes: blob.len(),
            });
            &cached_stream
        } else {
            stream
        };
        let desc = [stream.0.len(), stream.0.as_ptr() as usize];
        let mut raw = std::ptr::null_mut();
        let (hr, took) = driver_compile_identified(label, mesh_bytes, false, None, blob.is_some(), || {
            finish_pso(this, fs, desc.as_ptr().cast(), riid, &mut raw)
        });
        let pso = (!raw.is_null()).then(|| ID3D12PipelineState::from_raw(raw));
        (hr, pso, took)
    };
    if gcached.is_none() && !mesh_from_library {
        take_cold_slot();
    }
    let (mut hr, mut mesh_pso, mut gin_driver) = match library_mesh {
        Some(pso) => (S_OK, Some(pso), std::time::Duration::ZERO),
        None => create_mesh(kind, gcached.as_deref().map(Vec::as_slice)),
    };
    let mut greused = mesh_from_library || gcached.is_some();
    if !mesh_from_library && (hr < 0 || mesh_pso.is_none()) && gcached.is_some() {
        drop(mesh_pso.take());
        if let (Some(key), Some(expected)) = (gkey, rejected_generation.as_ref()) {
            forget_graphics_blob_if(key, expected);
        }
        if gfrom_disk {
            driver_cache::note(false, gloaded.take());
        }
        take_cold_slot();
        (hr, mesh_pso, gin_driver) = create_mesh("mesh pass retry", None);
        greused = false;
    }
    let Some(mesh_pso) = mesh_pso.filter(|_| hr >= 0) else {
        note_translation_error(format!("the driver rejected the mesh pipeline from the console {kind} ({hr:#x})"));
        return E_INVALIDARG;
    };
    if let Some(key) = gkey {
        if greused {
            if let Some(blob) = gcached.as_ref() {
                crate::shaders::warm_replay::accepted(gdisk.as_deref(), blob);
            }
            if let Some(blob) = gcached.as_ref().filter(|_| gfrom_disk) {
                GRAPHICS_BLOBS.admit(key, blob.clone(), GRAPHICS_BLOBS_MAX);
                driver_cache::note(true, gloaded.take());
            }
        } else {
            keep_driver_blob(&GRAPHICS_BLOBS, GRAPHICS_BLOBS_MAX, key, mesh_bytes, gdisk.as_deref(), gin_driver,
                || read_pso_cached_blob(mesh_pso.as_raw()));
            if let (Some(path), Some(blob)) = (gdisk.as_deref(), graphics_blob(key)) {
                crate::shaders::warm_replay::accepted(Some(path), &blob);
            }
        }
    }
    let fc: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreateComputePipelineState")));
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
    cdesc.NodeMask = node_mask;
    cdesc.pRootSignature = std::mem::transmute::<*mut c_void, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(rs);
    cdesc.CS = bytecode(&tp.cs);
    // Reuse the compiled driver blob of the compute pass across all `(MS/GS, PS)` pipelines that
    // share `tp.cs` (in memory and on disk), matching `translated_compute_pso_once`.
    let ckey = compute_blob_key_for(&tp.cs, rs as usize, 0, node_mask);
    let library_compute = crate::shaders::pipeline_library::load_compute(
        &driver_cache::name_for_root_node(&tp.cs, 0, false, &tp.samplers, node_mask),
        &cdesc,
    );
    let compute_from_library = library_compute.is_some();
    let mut ccached = if compute_from_library { None } else { compute_blob(ckey) };
    let cdisk = driver_cache::path_for_root_node(&tp.cs, 0, false, &tp.samplers, node_mask);
    let cfrom_disk = !compute_from_library && ccached.is_none();
    let mut cloaded = None;
    if cfrom_disk {
        cloaded = cdisk.as_deref().and_then(driver_cache::load);
        ccached = cloaded.as_ref().map(driver_cache::LoadedBlob::shared);
    }
    let rejected_cgeneration = ccached.clone();
    if let Some(b) = &ccached {
        cdesc.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr() as *const c_void, CachedBlobSizeInBytes: b.len() };
    } else if !compute_from_library {
        take_cold_slot();
    }
    let iid = pc_iid(Kind::PipelineState);
    let mut compute: *mut c_void = library_compute.map_or(std::ptr::null_mut(), |pso| pso.into_raw());
    let (mut hc, mut cin_driver) = if compute_from_library {
        (S_OK, std::time::Duration::ZERO)
    } else { driver_compile_identified("compute pass", tp.cs.len(), false, None, ccached.is_some(), || {
        driver_pso(fc, this, &cdesc as *const _ as *const c_void, &iid, &mut compute)
    }) };
    let mut cpersist_took = cin_driver;
    if hc < 0 && ccached.is_some() {
        if let Some(expected) = rejected_cgeneration.as_ref() {
            forget_compute_blob_if(ckey, expected);
        }
        if cfrom_disk {
            driver_cache::note(false, cloaded.take());
        }
        take_cold_slot();
        cdesc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        compute = std::ptr::null_mut();
        let (retried, took) = driver_compile_identified("compute pass retry", tp.cs.len(), false, None, false, || {
            driver_pso(fc, this, &cdesc as *const _ as *const c_void, &iid, &mut compute)
        });
        hc = retried;
        cin_driver += took;
        cpersist_took = took;
    }
    let creused = compute_from_library || (hc >= 0 && ccached.is_some() && cdesc.CachedPSO.CachedBlobSizeInBytes != 0);
    if creused {
        if let Some(blob) = ccached.as_ref() {
            crate::shaders::warm_replay::accepted(cdisk.as_deref(), blob);
        }
    }
    if hc >= 0 && !creused && !compute.is_null() {
        keep_driver_blob(&COMPUTE_BLOBS, COMPUTE_BLOBS_MAX, ckey, tp.cs.len(), cdisk.as_deref(), cpersist_took,
            || read_pso_cached_blob(compute));
        if let (Some(p), Some(b)) = (cdisk.as_deref(), compute_blob(ckey)) {
            crate::shaders::warm_replay::accepted(Some(p), &b);
        }
    } else if creused && cfrom_disk {
        COMPUTE_BLOBS.admit(ckey, ccached.as_ref().unwrap().clone(), COMPUTE_BLOBS_MAX);
        driver_cache::note(true, cloaded.take());
    }
    drop(slot);
    if hc < 0 || compute.is_null() {
        note_translation_error(format!("the driver rejected the compute pass of the console {kind} ({hc:#x})"));
        return E_INVALIDARG;
    }
    let library = crate::shaders::pipeline_library::store(
        match gkey {
            Some((hash, bytes, _)) => driver_cache::graphics_name(hash, bytes),
            None if crate::shaders::pipeline_library::building() => {
                super::pso_async::release(compute as usize);
                return E_INVALIDARG;
            }
            None => String::new(),
        },
        &mesh_pso,
    ).and_then(|_| {
        let compute_pso = ID3D12PipelineState::from_raw_borrowed(&compute)
            .ok_or_else(|| "The two-pass compute pipeline has no native object".to_string())?;
        crate::shaders::pipeline_library::store(
            driver_cache::name_for_root_node(&tp.cs, 0, false, &tp.samplers, node_mask),
            compute_pso,
        )
    });
    if let Err(e) = library {
        note_translation_error(e);
        super::pso_async::release(compute as usize);
        return E_INVALIDARG;
    }
    let mesh_pso = mesh_pso.into_raw();
    *ppv = mesh_pso;
    note_pso_rs(mesh_pso, mesh_rs);
    TWO_PASS.lock().unwrap().get_or_insert_with(HashMap::new).insert(mesh_pso as usize, std::sync::Arc::new(TwoPassPso { compute: compute as usize, compute_rs: rs as usize, mesh_rs: mesh_rs as usize, record: tp.record, mesh_groups_per_record: tp.mesh_groups_per_record, kind }));
    let n = TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
    if n < 5 || n % 1000 == 0 {
        crate::xlog_info!("D3D12.X: pipeline from the console {kind} in two passes (compute {} B + mesh {} B DXIL)", tp.cs.len(), tp.ms.len());
    }
    hr
}

/// CreatePipelineState with a console MS: Some(hr) when it was a console mesh pipeline
/// (translated, or E_INVALIDARG – console bytecode must not reach the driver).
unsafe fn translated_mesh_stream(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    with_switch_fallback(|| translated_mesh_stream_once(this, desc, riid, ppv))
}

unsafe fn translated_mesh_stream_once(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    use crate::gpu::recomp::pso::{console_mesh, is_console};
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return None;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let subs = stream_subobjects(std::slice::from_raw_parts(base as *const u8, size))?;
    let as_slot = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS.0 as u32).and_then(|s| payload_shader(s.1));
    let ms_slot = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32).and_then(|s| payload_shader(s.1));
    // A console MS container in the AS slot and no MS subobject (a retail game's minimap
    // pipelines): refused. Translated as the mesh shader they are, the title rendered nothing
    // from then on (a black screen every run); refused, it goes on to its menus. What it
    // expects of them is not known yet.
    if let (None, Some(a)) = (ms_slot, as_slot) {
        if is_console(a) && crate::gpu::xbox::parse(a).is_some_and(|sh| sh.stage == "MS") {
            note_translation_error(format!("mesh pipeline with its MS {:016x} in the AS slot – refused", crate::gpu::shader::fnv64(a)));
            return Some(E_INVALIDARG);
        }
    }
    let ms = ms_slot?;
    if !is_console(ms) {
        return None;
    }
    if let Some(a) = as_slot {
        let types: Vec<u32> = subs.iter().map(|s| s.0).collect();
        note_translation_error(format!(
            "mesh pipeline with AS {:016x} (MS {:016x}) – not supported yet; subobjects {types:?}",
            crate::gpu::shader::fnv64(a),
            crate::gpu::shader::fnv64(ms)
        ));
        return Some(E_INVALIDARG);
    }
    // made when first bound (`pso_async::lazy`): not even translated now
    if super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() {
        return None;
    }
    let ps = subs.iter().find(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS.0 as u32).and_then(|s| payload_shader(s.1));
    let console_ps = ps;
    let (tp, (ps, ps_samplers)) = match (console_mesh(ms, ps, wave32_only(this)), two_pass_ps(ps, graphics_wave32(this))) {
        (Ok(tp), Ok(ps)) => (tp, ps),
        (Err(e), _) => {
            note_translation_error(format!("MS: {e}"));
            return Some(E_INVALIDARG);
        }
        (_, Err(e)) => {
            note_translation_error(e);
            return Some(E_INVALIDARG);
        }
    };
    let Some(rs) = abi_rs_for(this, &ps_samplers) else { return Some(E_INVALIDARG) };
    // the title's state (blend, rasteriser, depth, formats …) with XCL's RS and shaders
    let mut out = two_pass_mesh_stream(rs, &tp.ms, ps.as_deref());
    // Extract PC-mapped render target formats upfront so BLEND (type 8) can be sanitized for
    // integer RTVs regardless of subobject order in the stream.
    let mut rt_count = 0u32;
    let mut rt_formats = [windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(0); 8];
    if let Some((_, p)) = subs.iter().find(|(ty, p)| *ty == 15 && p.len() >= 36) {
        for k in 0..8 {
            let f = super::device_x::pc_target_format(u32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap()));
            rt_formats[k] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(f as i32);
        }
        rt_count = u32::from_le_bytes(p[32..36].try_into().unwrap());
    }
    for (ty, payload) in &subs {
        let skip = [0, 2, 14, 19, 24, 25].contains(ty); // RS, PS, topology, cached PSO, AS, MS
        if !skip {
            let (_, palign) = super::null::subobject_payload(*ty).unwrap_or((0, 4));
            // BLEND: clear BlendEnable / LogicOpEnable on integer render targets (PC D3D12 rejects them)
            if *ty == 8 && payload.len() >= std::mem::size_of::<D3D12_BLEND_DESC>() {
                let mut blend: D3D12_BLEND_DESC = std::ptr::read_unaligned(payload.as_ptr() as *const D3D12_BLEND_DESC);
                sanitize_integer_rt_blend(&mut blend, rt_count, &rt_formats);
                out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend);
                continue;
            }
            // RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT: the console's formats as the PC's
            if (*ty == 15 && payload.len() >= 32) || (*ty == 16 && payload.len() >= 4) {
                let mut p = payload.to_vec();
                for k in 0..if *ty == 15 { 8 } else { 1 } {
                    let f = if *ty == 15 {
                        super::device_x::pc_target_format(u32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap()))
                    } else {
                        super::device_x::pc_view_format(u32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap()))
                    };
                    p[4 * k..4 * k + 4].copy_from_slice(&f.to_le_bytes());
                }
                out.raw(*ty, &p, palign == 8);
                continue;
            }
            out.raw(*ty, payload, palign == 8);
        }
    }
    let node_mask = stream_node_mask(&subs);
    let hr = finish_two_pass(this, &out, tp, rs, &ps_samplers, node_mask, "MS", riid, ppv);
    if hr >= 0 {
        super::dred::label(*ppv as usize, &[("ms", Some(ms)), ("ps", console_ps)]);
    }
    Some(hr)
}

/// A graphics PSO with a console VS + GS: the GS pipeline in two passes, as a mesh pipeline
/// with the description's state.
unsafe fn translated_gs_pso(this: *mut c_void, d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, vs: &[u8], gs: &[u8], riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let (tp, (ps, ps_samplers)) = match (
        crate::gpu::recomp::pso::console_geometry(vs, gs, shader_bytes(&d.PS), wave32_only(this)),
        two_pass_ps(shader_bytes(&d.PS), graphics_wave32(this)),
    ) {
        (Ok(tp), Ok(ps)) => (tp, ps),
        (Err(e), _) => {
            // what the containers hold: each FSSM's hardware stages and its programs
            let describe = |blob: &[u8]| -> String {
                crate::gpu::xbox::parse(blob).map_or("not a console container".into(), |sh| {
                    sh.fssm
                        .iter()
                        .map(|f| {
                            format!(
                                "[{} | {} programs {:?}, version {:#x}, {} B, code {:?} (from header {}), error {:?}]",
                                f.stages.iter().map(|x| x.label()).collect::<Vec<_>>().join(", "),
                                f.programs.len(),
                                f.programs.iter().map(|p| p.len).collect::<Vec<_>>(),
                                f.version,
                                f.data.len(),
                                f.code,
                                f.code_from_header,
                                f.error
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
            };
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: VS+GS pipeline not translated ({e}); VS {}; GS {}", describe(vs), describe(gs));
                if let Ok(dir) = std::env::var("XCL_DUMP_GS") {
                    let k = N.load(Ordering::Relaxed);
                    let _ = std::fs::write(format!("{dir}/vs{k}.bin"), vs);
                    let _ = std::fs::write(format!("{dir}/gs{k}.bin"), gs);
                }
            }
            note_translation_error(format!("VS+GS: {e}"));
            return E_INVALIDARG;
        }
        (_, Err(e)) => {
            note_translation_error(e);
            return E_INVALIDARG;
        }
    };
    let Some(rs) = abi_rs_for(this, &ps_samplers) else { return E_INVALIDARG };
    let mut out = two_pass_mesh_stream(rs, &tp.ms, ps.as_deref());
    let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: d.NumRenderTargets, RTFormats: d.RTVFormats };
    for f in rt.RTFormats.iter_mut() {
        *f = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_target_format(f.0 as u32) as i32);
    }
    let dsv = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_view_format(d.DSVFormat.0 as u32) as i32);
    let mut blend = d.BlendState;
    sanitize_integer_rt_blend(&mut blend, rt.NumRenderTargets, &rt.RTFormats);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &d.SampleMask);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &d.RasterizerState);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, &d.DepthStencilState);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &dsv);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &d.SampleDesc);
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK, &d.NodeMask);
    if d.Flags.0 != 0 {
        out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS, &d.Flags);
    }
    let hr = finish_two_pass(this, &out, tp, rs, &ps_samplers, d.NodeMask, "GS", riid, ppv);
    if hr >= 0 {
        super::dred::label(*ppv as usize, &[("vs", Some(vs)), ("gs", Some(gs)), ("ps", shader_bytes(&d.PS))]);
    }
    hr
}

/// Rewrite a pipeline state stream with stand-in shaders. None if not possible.
unsafe fn fallback_stream(desc: *const u8) -> Option<(Vec<u8>, [usize; 2])> {
    let size = *(desc as *const usize);
    let base = *(desc.add(8) as *const usize);
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let mut buf = std::slice::from_raw_parts(base as *const u8, size).to_vec();
    let d = dummy();
    let (mesh, rt_count, rt_formats) = {
        let mut mesh = false;
        let mut rt_count = 0u32;
        let mut rt_formats = [windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(0); 8];
        if let Some(subs) = stream_subobjects(&buf) {
            mesh = subs.iter().any(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32);
            if let Some((_, p)) = subs.iter().find(|(ty, p)| *ty == 15 && p.len() >= 36) {
                for k in 0..8 {
                    let f = super::device_x::pc_target_format(u32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap()));
                    rt_formats[k] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(f as i32);
                }
                rt_count = u32::from_le_bytes(p[32..36].try_into().unwrap());
            }
        }
        (mesh, rt_count, rt_formats)
    };
    let dm = if mesh { Some(dummy_mesh()?) } else { None };
    let mut off = 0usize;
    while off + 8 <= size {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let (psize, palign) = super::null::subobject_payload(ty)?;
        let poff = if palign == 8 { 8 } else { 4 };
        let at = off + poff;
        if at.checked_add(psize)? > size {
            return None; // SizeInBytes ends inside the payload
        }
        let put = |buf: &mut Vec<u8>, b: &[u8]| {
            // no shader: a null pointer – an empty slice's is dangling (0x1), which the driver
            // refused (a retail game's mesh stand-ins, E_INVALIDARG)
            let ptr = if b.is_empty() { 0 } else { b.as_ptr() as u64 };
            buf[at..at + 8].copy_from_slice(&ptr.to_le_bytes());
            buf[at + 8..at + 16].copy_from_slice(&(b.len() as u64).to_le_bytes());
        };
        match ty {
            1 => put(&mut buf, &d.vs),
            2 => put(&mut buf, dm.map_or(&d.ps[..], |m| &m.1[..])),
            6 => put(&mut buf, &d.cs),
            3 | 4 | 5 | 24 => put(&mut buf, &[]),
            25 => put(&mut buf, &dm?.0),
            8 if psize >= std::mem::size_of::<D3D12_BLEND_DESC>() => {
                let mut blend: D3D12_BLEND_DESC = std::ptr::read_unaligned(buf[at..].as_ptr() as *const D3D12_BLEND_DESC);
                sanitize_integer_rt_blend(&mut blend, rt_count, &rt_formats);
                let bytes = std::slice::from_raw_parts(&blend as *const D3D12_BLEND_DESC as *const u8, std::mem::size_of::<D3D12_BLEND_DESC>());
                buf[at..at + bytes.len()].copy_from_slice(bytes);
            }
            15 if psize >= 32 => {
                for k in 0..8 {
                    let f = super::device_x::pc_target_format(u32::from_le_bytes(buf[at + 4 * k..at + 4 * k + 4].try_into().unwrap()));
                    buf[at + 4 * k..at + 4 * k + 4].copy_from_slice(&f.to_le_bytes());
                }
            }
            16 if psize >= 4 => {
                let f = super::device_x::pc_view_format(u32::from_le_bytes(buf[at..at + 4].try_into().unwrap()));
                buf[at..at + 4].copy_from_slice(&f.to_le_bytes());
            }
            12 | 19 => buf[at..at + 16].fill(0), // input layout, cached PSO
            7 => buf[at..at + 32].fill(0),        // stream output
            _ => {}
        }
        off += (poff + psize + 7) & !7;
    }
    Some((buf, [size, 0]))
}

/// The console's pipeline topology type of rect lists (draws of topology 17, `list_x::RECT_LIST`).
const RECT_LIST_TYPE: i32 = 19;

/// `create(copy)`, and once more without the generated rect list GS (`rect_gs`) when the driver
/// refuses the pipeline with E_INVALIDARG: each rect is then drawn as the triangle of its three vertices,
/// as without the GS, rather than not at all (the untranslated fallback's dummy shaders).
fn rejected_rect_list_gs(hr: HRESULT, rect_gs: bool) -> bool {
    rect_gs && hr == E_INVALIDARG
}

fn retry_without_rect_gs(copy: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC, rect_gs: bool, mut create: impl FnMut(&D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> HRESULT) -> HRESULT {
    let hr = create(copy);
    if !rejected_rect_list_gs(hr, rect_gs) {
        return hr;
    }
    note_translation_error(format!("the driver rejected the PSO with the rect list GS ({hr:#x}): rect list drawn as triangles"));
    copy.GS = D3D12_SHADER_BYTECODE::default();
    create(copy)
}

/// What an untranslated pipeline from a stream is made from: a rect list's (type 19, which
/// `stream_pipeline` keeps for `translated_graphics_pso`) with the PC's topology type, as the
/// stream path made it before; None: `desc` as it is.
unsafe fn untranslated_stream_desc(desc: *const c_void, stream: bool) -> Option<D3D12_GRAPHICS_PIPELINE_STATE_DESC> {
    if !stream || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return None;
    }
    let mut g = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    if g.PrimitiveTopologyType.0 != RECT_LIST_TYPE {
        return None;
    }
    pc_topology_type(&mut g);
    Some(g)
}

/// The GS that completes a rect list's rects behind the console VS `vs` and the PS `ps`; refused
/// for a PS that is not a console one (its inputs' modes are not known).
fn rect_list_gs(vs: &[u8], ps: Option<&[u8]>) -> Result<Vec<u8>, String> {
    use crate::gpu::recomp::pso;
    if ps.is_some_and(|p| !pso::is_console(p)) {
        return Err("a PC pixel shader".into());
    }
    pso::rect_list_gs(vs)
}

/// Whether `f` is an integer (`*_UINT` / `*_SINT`) DXGI render target format on which PC D3D12
/// forbids `BlendEnable = TRUE` (console hardware silently ignores blending on integer RTs).
fn is_integer_dxgi_format(f: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT) -> bool {
    use windows::Win32::Graphics::Dxgi::Common::*;
    matches!(
        f,
        DXGI_FORMAT_R32G32B32A32_UINT
            | DXGI_FORMAT_R32G32B32A32_SINT
            | DXGI_FORMAT_R32G32B32_UINT
            | DXGI_FORMAT_R32G32B32_SINT
            | DXGI_FORMAT_R16G16B16A16_UINT
            | DXGI_FORMAT_R16G16B16A16_SINT
            | DXGI_FORMAT_R32G32_UINT
            | DXGI_FORMAT_R32G32_SINT
            | DXGI_FORMAT_R10G10B10A2_UINT
            | DXGI_FORMAT_R8G8B8A8_UINT
            | DXGI_FORMAT_R8G8B8A8_SINT
            | DXGI_FORMAT_R16G16_UINT
            | DXGI_FORMAT_R16G16_SINT
            | DXGI_FORMAT_R32_UINT
            | DXGI_FORMAT_R32_SINT
            | DXGI_FORMAT_R8G8_UINT
            | DXGI_FORMAT_R8G8_SINT
            | DXGI_FORMAT_R16_UINT
            | DXGI_FORMAT_R16_SINT
            | DXGI_FORMAT_R8_UINT
            | DXGI_FORMAT_R8_SINT
    )
}

/// Clear `BlendEnable` and `LogicOpEnable` for every active integer (`*_UINT`/`*_SINT`) render
/// target slot so PC D3D12 accepts console PSOs that leave blending enabled on integer RTs.
/// When `IndependentBlendEnable` is false and a mix of integer and non-integer RTs has blending
/// enabled on slot 0, promote to `IndependentBlendEnable = TRUE` so non-integer slots keep
/// blending while integer slots have it cleared (clearing `LogicOpEnable` on all slots as D3D12
/// requires when independent blend is enabled).
fn sanitize_integer_rt_blend(
    blend: &mut D3D12_BLEND_DESC,
    num_render_targets: u32,
    rt_formats: &[windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT; 8],
) {
    let n = (num_render_targets as usize).min(8);
    if n == 0 {
        return;
    }
    if blend.IndependentBlendEnable.as_bool() {
        for i in 0..n {
            if is_integer_dxgi_format(rt_formats[i]) {
                blend.RenderTarget[i].BlendEnable = false.into();
                blend.RenderTarget[i].LogicOpEnable = false.into();
            }
        }
        return;
    }
    let any_int = (0..n).any(|i| is_integer_dxgi_format(rt_formats[i]));
    if !any_int {
        return;
    }
    let rt0_blend = blend.RenderTarget[0].BlendEnable.as_bool();
    let rt0_logic = blend.RenderTarget[0].LogicOpEnable.as_bool();
    if !rt0_blend && !rt0_logic {
        return;
    }
    let any_non_int = (0..n).any(|i| !is_integer_dxgi_format(rt_formats[i]));
    if rt0_blend && any_non_int {
        let mut base = blend.RenderTarget[0];
        base.LogicOpEnable = false.into();
        blend.IndependentBlendEnable = true.into();
        for i in 0..8 {
            blend.RenderTarget[i] = base;
        }
        for i in 0..n {
            if is_integer_dxgi_format(rt_formats[i]) {
                blend.RenderTarget[i].BlendEnable = false.into();
                blend.RenderTarget[i].LogicOpEnable = false.into();
            }
        }
    } else {
        blend.RenderTarget[0].BlendEnable = false.into();
        blend.RenderTarget[0].LogicOpEnable = false.into();
    }
}

/// A topology type the PC has: the console's for rect lists (19, with draws of topology 17 –
/// `list_x::RECT_LIST`) is TRIANGLE, as those draws become triangle lists (the rects' fourth
/// corners come from `pso::rect_list_gs`, added by `translated_graphics_pso`); UNDEFINED (a stream
/// without one) is TRIANGLE, or PATCH with a hull shader – the PC refuses both with a VS.
fn pc_topology_type(g: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC) {
    // (and the console's D16S8 depth format the PC's: `device_x::pc_view_format`)
    g.DSVFormat = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_view_format(g.DSVFormat.0 as u32) as i32);
    for f in g.RTVFormats.iter_mut() {
        *f = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_target_format(f.0 as u32) as i32);
    }
    sanitize_integer_rt_blend(&mut g.BlendState, g.NumRenderTargets, &g.RTVFormats);
    if !(1..=4).contains(&g.PrimitiveTopologyType.0) {
        g.PrimitiveTopologyType = if g.HS.pShaderBytecode.is_null() { D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE } else { D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH };
    }
}

#[cfg(test)]
mod pso_blend_and_tsharp_tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    #[test]
    fn integer_rtv_blend_sanitization_clears_uint_sint_and_preserves_float() {
        assert!(is_integer_dxgi_format(DXGI_FORMAT_R8_UINT));
        assert!(is_integer_dxgi_format(DXGI_FORMAT_R16G16_SINT));
        assert!(is_integer_dxgi_format(DXGI_FORMAT_R10G10B10A2_UINT));
        assert!(is_integer_dxgi_format(DXGI_FORMAT_R32G32B32A32_SINT));
        assert!(!is_integer_dxgi_format(DXGI_FORMAT_R8G8B8A8_UNORM));
        assert!(!is_integer_dxgi_format(DXGI_FORMAT_R16G16B16A16_FLOAT));
        assert!(!is_integer_dxgi_format(DXGI_FORMAT_UNKNOWN));

        // Case 1: IndependentBlendEnable=TRUE -> clears only integer RT slots, preserves float slots.
        let mut g: D3D12_GRAPHICS_PIPELINE_STATE_DESC = unsafe { std::mem::zeroed() };
        g.NumRenderTargets = 3;
        g.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        g.RTVFormats[1] = DXGI_FORMAT_R8_UINT;
        g.RTVFormats[2] = DXGI_FORMAT_R16G16_SINT;
        g.BlendState.IndependentBlendEnable = true.into();
        for i in 0..3 {
            g.BlendState.RenderTarget[i].BlendEnable = true.into();
            g.BlendState.RenderTarget[i].LogicOpEnable = true.into();
            g.BlendState.RenderTarget[i].RenderTargetWriteMask = 0xF;
        }
        pc_topology_type(&mut g);
        assert!(g.BlendState.IndependentBlendEnable.as_bool());
        assert!(g.BlendState.RenderTarget[0].BlendEnable.as_bool());
        assert!(!g.BlendState.RenderTarget[1].BlendEnable.as_bool());
        assert!(!g.BlendState.RenderTarget[1].LogicOpEnable.as_bool());
        assert!(!g.BlendState.RenderTarget[2].BlendEnable.as_bool());
        assert!(!g.BlendState.RenderTarget[2].LogicOpEnable.as_bool());
        assert_eq!(g.BlendState.RenderTarget[1].RenderTargetWriteMask, 0xF);

        // Case 2: IndependentBlendEnable=FALSE with mixed float + integer RTs -> promotes to independent blend.
        let mut m: D3D12_GRAPHICS_PIPELINE_STATE_DESC = unsafe { std::mem::zeroed() };
        m.NumRenderTargets = 2;
        m.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        m.RTVFormats[1] = DXGI_FORMAT_R32_UINT;
        m.BlendState.IndependentBlendEnable = false.into();
        m.BlendState.RenderTarget[0].BlendEnable = true.into();
        m.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
        m.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        m.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
        pc_topology_type(&mut m);
        assert!(m.BlendState.IndependentBlendEnable.as_bool());
        assert!(m.BlendState.RenderTarget[0].BlendEnable.as_bool());
        assert!(!m.BlendState.RenderTarget[1].BlendEnable.as_bool());
        assert!(!m.BlendState.RenderTarget[1].LogicOpEnable.as_bool());
        assert_eq!(m.BlendState.RenderTarget[1].SrcBlend, D3D12_BLEND_SRC_ALPHA);

        // Case 3: IndependentBlendEnable=FALSE with all-integer RTs -> clears slot 0 without enabling independent blend.
        let mut a: D3D12_GRAPHICS_PIPELINE_STATE_DESC = unsafe { std::mem::zeroed() };
        a.NumRenderTargets = 1;
        a.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UINT;
        a.BlendState.IndependentBlendEnable = false.into();
        a.BlendState.RenderTarget[0].BlendEnable = true.into();
        a.BlendState.RenderTarget[0].LogicOpEnable = true.into();
        pc_topology_type(&mut a);
        assert!(!a.BlendState.IndependentBlendEnable.as_bool());
        assert!(!a.BlendState.RenderTarget[0].BlendEnable.as_bool());
        assert!(!a.BlendState.RenderTarget[0].LogicOpEnable.as_bool());
    }

    #[test]
    fn rt_format_array_layout_matches_stream_parsing() {
        assert_eq!(std::mem::offset_of!(D3D12_RT_FORMAT_ARRAY, RTFormats), 0);
        assert_eq!(std::mem::offset_of!(D3D12_RT_FORMAT_ARRAY, NumRenderTargets), 32);
        assert_eq!(std::mem::size_of::<D3D12_RT_FORMAT_ARRAY>(), 36);
    }

    #[test]
    fn srv_tsharp_populates_base_level_and_msaa_last_level() {
        // Non-MSAA 2D texture with MostDetailedMip > 0 (CS_MinTimestepReduce extracts BASE_LEVEL = (t[3] >> 12) & 0xF).
        let tex2d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 512,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 9,
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 3, MipLevels: 4, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
            },
        };
        let t = srv_tsharp_from_desc(42, &tex2d, Some(&srv)).unwrap();
        let img = crate::gpu::desc::image(&t).unwrap();
        assert_eq!(t[0], 42);
        assert_eq!((t[3] >> 12) & 0xF, 3, "BASE_LEVEL must encode MostDetailedMip");
        assert_eq!((t[3] >> 16) & 0xF, 6, "LAST_LEVEL = base + mips - 1");
        assert_eq!((t[5] >> 4) & 0xF, 8, "MAX_MIP = res.MipLevels - 1");
        assert_eq!((img.width, img.height, img.base_level, img.last_level, img.max_mip), (512, 256, 3, 6, 8));
        assert_eq!((img.width >> img.base_level, img.height >> img.base_level), (64, 32));
        let (re_res, re_srv) = crate::gpu::desc::image_d3d12(&img).unwrap();
        assert_eq!(re_res.MipLevels, 9);
        assert_eq!(unsafe { (re_srv.Anonymous.Texture2D.MostDetailedMip, re_srv.Anonymous.Texture2D.MipLevels) }, (3, 4));

        // MipLevels == u32::MAX (all remaining mips from MostDetailedMip through res.MipLevels - 1).
        let srv_all = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 2, MipLevels: u32::MAX, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
            },
            ..srv
        };
        let img_all = crate::gpu::desc::image(&srv_tsharp_from_desc(7, &tex2d, Some(&srv_all)).unwrap()).unwrap();
        assert_eq!((img_all.base_level, img_all.last_level, img_all.max_mip), (2, 8, 8));

        // MSAA 2D & 2DArray SRVs: LAST_LEVEL = log2(SampleDesc.Count) (PS_PuddleDetect / MSAA resinfo).
        for (samples, want_last) in [(2u32, 1u32), (4, 2), (8, 3)] {
            let msaa = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 1920,
                Height: 1080,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 },
                ..Default::default()
            };
            let msaa_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMS,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            let t_ms = srv_tsharp_from_desc(99, &msaa, Some(&msaa_srv)).unwrap();
            let img_ms = crate::gpu::desc::image(&t_ms).unwrap();
            assert_eq!(img_ms.ty, crate::gpu::desc::ImageType::D2Msaa);
            assert_eq!((img_ms.base_level, img_ms.last_level), (0, want_last), "{samples}x MSAA");
            let (re_ms, _) = crate::gpu::desc::image_d3d12(&img_ms).unwrap();
            assert_eq!(re_ms.SampleDesc.Count, samples);

            // Default NULL SRV desc on an MSAA resource must also populate LAST_LEVEL = log2(samples).
            let img_null = crate::gpu::desc::image(&srv_tsharp_from_desc(99, &msaa, None).unwrap()).unwrap();
            assert_eq!((img_null.ty, img_null.base_level, img_null.last_level), (crate::gpu::desc::ImageType::D2Msaa, 0, want_last));
        }

        let msaa_arr = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1280,
            Height: 720,
            DepthOrArraySize: 4,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 4, Quality: 0 },
            ..Default::default()
        };
        let msaa_arr_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DMSArray: D3D12_TEX2DMS_ARRAY_SRV { FirstArraySlice: 0, ArraySize: 4 },
            },
        };
        let img_arr = crate::gpu::desc::image(&srv_tsharp_from_desc(11, &msaa_arr, Some(&msaa_arr_srv)).unwrap()).unwrap();
        assert_eq!((img_arr.ty, img_arr.depth, img_arr.base_level, img_arr.last_level), (crate::gpu::desc::ImageType::D2MsaaArray, 4, 0, 2));
    }
}

enum StreamPipeline {
    Graphics(Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC>, bool),
    GraphicsStream,
    Compute(Box<D3D12_COMPUTE_PIPELINE_STATE_DESC>),
    /// neither with console shaders (or not readable)
    Unknown,
}

unsafe fn read_sub<T>(p: &[u8]) -> T {
    let mut d: T = std::mem::zeroed();
    put(&mut d, p);
    d
}

fn depth_bias_is_exact_i32(b: f32) -> bool {
    b.is_finite() && b >= i32::MIN as f32 && b < 2147483648.0 && b.fract() == 0.0
}

unsafe fn stream_needs_extended_rasterizer(subs: &[(u32, &[u8])]) -> bool {
    subs.iter().any(|&(ty, p)| match ty {
        27 => !depth_bias_is_exact_i32(read_sub::<D3D12_RASTERIZER_DESC1>(p).DepthBias),
        28 => {
            let d: D3D12_RASTERIZER_DESC2 = read_sub(p);
            !depth_bias_is_exact_i32(d.DepthBias) || d.LineRasterizationMode != D3D12_LINE_RASTERIZATION_MODE_ALIASED
        }
        _ => false,
    })
}

/// A pipeline stream with a console VS (+ PS / GS) or CS as the plain pipeline desc, which is
/// translated like any other (`dev_create_graphics_pso`). The stream itself must not reach the
/// driver: AMD's shader compiler fails on the console's stripped DXIL and takes the device with
/// it (0x887A0020 at the retail game's first graphics pipeline). Depth bounds, view instancing and the
/// float depth bias' fraction have no place in the plain desc.
unsafe fn stream_pipeline(desc: *const c_void) -> StreamPipeline {
    use crate::gpu::recomp::pso::is_console;
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return StreamPipeline::Unknown;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return StreamPipeline::Unknown;
    }
    let (subs, cacheable) = known_subobjects_checked(std::slice::from_raw_parts(base as *const u8, size));
    if subobjects_have_unreadable_shader(&subs) {
        return StreamPipeline::Unknown;
    }
    let console = |ty: u32| subs.iter().find(|s| s.0 == ty).and_then(|s| payload_shader(s.1)).is_some_and(is_console);
    if console(6) {
        let mut c: D3D12_COMPUTE_PIPELINE_STATE_DESC = std::mem::zeroed();
        for (ty, p) in &subs {
            match ty {
                0 => put(&mut c.pRootSignature, p),
                6 => put(&mut c.CS, p),
                18 => put(&mut c.NodeMask, p),
                20 => put(&mut c.Flags, p),
                _ => {}
            }
        }
        return StreamPipeline::Compute(Box::new(c));
    }
    if !console(1) {
        return StreamPipeline::Unknown;
    }
    // Unknown/truncated suffixes retain Stage 1's conservative projected, uncacheable path:
    // rewriting cannot preserve bytes whose layout is unknown. Console GS needs its two-pass path.
    if cacheable && !console(5) && stream_needs_extended_rasterizer(&subs) {
        return StreamPipeline::GraphicsStream;
    }
    let mut g: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::mem::zeroed();
    // what a stream leaves out has the PC's defaults
    g.SampleMask = u32::MAX;
    g.SampleDesc.Count = 1;
    g.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    g.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    g.RasterizerState.DepthClipEnable = true.into();
    for rt in &mut g.BlendState.RenderTarget {
        rt.SrcBlend = D3D12_BLEND_ONE;
        rt.DestBlend = D3D12_BLEND_ZERO;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = 15;
    }
    let keep = D3D12_DEPTH_STENCILOP_DESC { StencilFailOp: D3D12_STENCIL_OP_KEEP, StencilDepthFailOp: D3D12_STENCIL_OP_KEEP, StencilPassOp: D3D12_STENCIL_OP_KEEP, StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS };
    g.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: true.into(),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
        DepthFunc: D3D12_COMPARISON_FUNC_LESS,
        StencilEnable: false.into(),
        StencilReadMask: 0xFF,
        StencilWriteMask: 0xFF,
        FrontFace: keep,
        BackFace: keep,
    };
    let mut has_depth_stencil = false;
    for (ty, p) in &subs {
        match ty {
            0 => put(&mut g.pRootSignature, p),
            1 => put(&mut g.VS, p),
            2 => put(&mut g.PS, p),
            3 => put(&mut g.DS, p),
            4 => put(&mut g.HS, p),
            5 => put(&mut g.GS, p),
            7 => put(&mut g.StreamOutput, p),
            8 => put(&mut g.BlendState, p),
            9 => put(&mut g.SampleMask, p),
            10 => put(&mut g.RasterizerState, p),
            // DEPTH_STENCIL; DEPTH_STENCIL1 is that and the depth bounds test
            11 | 21 => {
                has_depth_stencil = true;
                put(&mut g.DepthStencilState, p);
            }
            12 => put(&mut g.InputLayout, p),
            13 => put(&mut g.IBStripCutValue, p),
            14 => put(&mut g.PrimitiveTopologyType, p),
            15 => {
                let mut rt: D3D12_RT_FORMAT_ARRAY = std::mem::zeroed();
                put(&mut rt, p);
                g.RTVFormats = rt.RTFormats;
                g.NumRenderTargets = rt.NumRenderTargets;
            }
            16 => put(&mut g.DSVFormat, p),
            17 => put(&mut g.SampleDesc, p),
            18 => put(&mut g.NodeMask, p),
            20 => put(&mut g.Flags, p),
            26 => {
                has_depth_stencil = true;
                let mut d: D3D12_DEPTH_STENCIL_DESC2 = std::mem::zeroed();
                put(&mut d, p);
                let face = |f: &D3D12_DEPTH_STENCILOP_DESC1| D3D12_DEPTH_STENCILOP_DESC { StencilFailOp: f.StencilFailOp, StencilDepthFailOp: f.StencilDepthFailOp, StencilPassOp: f.StencilPassOp, StencilFunc: f.StencilFunc };
                g.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: d.DepthEnable,
                    DepthWriteMask: d.DepthWriteMask,
                    DepthFunc: d.DepthFunc,
                    StencilEnable: d.StencilEnable,
                    StencilReadMask: d.FrontFace.StencilReadMask,
                    StencilWriteMask: d.FrontFace.StencilWriteMask,
                    FrontFace: face(&d.FrontFace),
                    BackFace: face(&d.BackFace),
                };
            }
            27 => {
                let mut d: D3D12_RASTERIZER_DESC1 = std::mem::zeroed();
                put(&mut d, p);
                g.RasterizerState = D3D12_RASTERIZER_DESC {
                    FillMode: d.FillMode,
                    CullMode: d.CullMode,
                    FrontCounterClockwise: d.FrontCounterClockwise,
                    DepthBias: d.DepthBias as i32,
                    DepthBiasClamp: d.DepthBiasClamp,
                    SlopeScaledDepthBias: d.SlopeScaledDepthBias,
                    DepthClipEnable: d.DepthClipEnable,
                    MultisampleEnable: d.MultisampleEnable,
                    AntialiasedLineEnable: d.AntialiasedLineEnable,
                    ForcedSampleCount: d.ForcedSampleCount,
                    ConservativeRaster: d.ConservativeRaster,
                };
            }
            28 => {
                let mut d: D3D12_RASTERIZER_DESC2 = std::mem::zeroed();
                put(&mut d, p);
                g.RasterizerState = D3D12_RASTERIZER_DESC {
                    FillMode: d.FillMode,
                    CullMode: d.CullMode,
                    FrontCounterClockwise: d.FrontCounterClockwise,
                    DepthBias: d.DepthBias as i32,
                    DepthBiasClamp: d.DepthBiasClamp,
                    SlopeScaledDepthBias: d.SlopeScaledDepthBias,
                    DepthClipEnable: d.DepthClipEnable,
                    ForcedSampleCount: d.ForcedSampleCount,
                    ConservativeRaster: d.ConservativeRaster,
                    ..Default::default()
                };
            }
            _ => {}
        }
    }
    // D3D12 pipeline state streams default DepthEnable to FALSE when no depth-stencil
    // subobject is supplied and DSVFormat is DXGI_FORMAT_UNKNOWN (0).
    if !has_depth_stencil && g.DSVFormat.0 == 0 {
        g.DepthStencilState.DepthEnable = false.into();
    }
    // a rect list stays one for translated_graphics_pso, which gives it its GS (untranslated, it
    // becomes the PC's in graphics_pso)
    if g.PrimitiveTopologyType.0 != RECT_LIST_TYPE {
        pc_topology_type(&mut g);
    }
    StreamPipeline::Graphics(Box::new(g), cacheable)
}

unsafe fn patch_sub<T>(out: &mut StreamOut, ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, payload: &[u8], f: impl FnOnce(&mut T)) {
    let mut v: T = std::ptr::read_unaligned(payload.as_ptr() as *const T);
    f(&mut v);
    out.put(ty, &v);
}

/// Native stream translation adapted from PR #2621 by Decodere.
unsafe fn rewrite_graphics_stream(
    subs: &[(u32, &[u8])], rs: *mut c_void, vs_dxil: &[u8], ps_dxil: Option<&[u8]>, rect_gs: Option<&[u8]>, no_stencil: bool, no_cull: bool,
) -> StreamOut {
    let mut out = StreamOut::default();
    out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &(rs as usize));
    out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs_dxil);
    if let Some(ps) = ps_dxil { out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps); }
    if let Some(gs) = rect_gs { out.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, gs); }
    let (mut rt_count, mut rt_formats) = (0u32, [windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(0); 8]);
    if let Some((_, p)) = subs.iter().find(|(ty, p)| *ty == 15 && p.len() >= 36) {
        for (k, ch) in p[..32].chunks_exact(4).enumerate() {
            rt_formats[k] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_target_format(u32::from_le_bytes(ch.try_into().unwrap())) as i32);
        }
        rt_count = u32::from_le_bytes(p[32..36].try_into().unwrap());
    }
    let default_topo = if subs.iter().any(|(ty, p)| *ty == 4 && payload_shader(p).is_some()) { D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH } else { D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE };
    let (mut saw_topology, mut saw_rasterizer) = (false, false);
    for &(ty, payload) in subs {
        if matches!(ty, 0 | 1 | 2 | 19) || (ty == 5 && rect_gs.is_some()) { continue; }
        if matches!(ty, 10 | 27 | 28) { saw_rasterizer = true; }
        let (_, palign) = super::null::subobject_payload(ty).unwrap_or((0, 4));
        match ty {
            8 if payload.len() >= std::mem::size_of::<D3D12_BLEND_DESC>() => patch_sub::<D3D12_BLEND_DESC>(&mut out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, payload, |b| sanitize_integer_rt_blend(b, rt_count, &rt_formats)),
            10 if no_cull && payload.len() >= std::mem::size_of::<D3D12_RASTERIZER_DESC>() => patch_sub::<D3D12_RASTERIZER_DESC>(&mut out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, payload, |r| r.CullMode = D3D12_CULL_MODE_NONE),
            27 if no_cull && payload.len() >= std::mem::size_of::<D3D12_RASTERIZER_DESC1>() => patch_sub::<D3D12_RASTERIZER_DESC1>(&mut out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1, payload, |r| r.CullMode = D3D12_CULL_MODE_NONE),
            28 if no_cull && payload.len() >= std::mem::size_of::<D3D12_RASTERIZER_DESC2>() => patch_sub::<D3D12_RASTERIZER_DESC2>(&mut out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2, payload, |r| r.CullMode = D3D12_CULL_MODE_NONE),
            11 if no_stencil && payload.len() >= std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC>() => patch_sub::<D3D12_DEPTH_STENCIL_DESC>(&mut out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, payload, |d| d.StencilEnable = false.into()),
            21 if no_stencil && payload.len() >= std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC1>() => patch_sub::<D3D12_DEPTH_STENCIL_DESC1>(&mut out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, payload, |d| d.StencilEnable = false.into()),
            26 if no_stencil && payload.len() >= std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC2>() => patch_sub::<D3D12_DEPTH_STENCIL_DESC2>(&mut out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, payload, |d| d.StencilEnable = false.into()),
            14 if payload.len() >= 4 => {
                saw_topology = true;
                let t = i32::from_le_bytes(payload[0..4].try_into().unwrap());
                out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &if (1..=4).contains(&t) { D3D12_PRIMITIVE_TOPOLOGY_TYPE(t) } else { default_topo });
            }
            15 if payload.len() >= 32 => {
                let mut p = payload.to_vec();
                for ch in p[..32].chunks_exact_mut(4) {
                    let f = super::device_x::pc_target_format(u32::from_le_bytes(ch.try_into().unwrap()));
                    ch.copy_from_slice(&f.to_le_bytes());
                }
                out.raw(15, &p, false);
            }
            16 if payload.len() >= 4 => out.raw(16, &super::device_x::pc_view_format(u32::from_le_bytes(payload[0..4].try_into().unwrap())).to_le_bytes(), false),
            _ => out.raw(ty, payload, palign == 8),
        }
    }
    if !saw_topology { out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &default_topo); }
    if no_cull && !saw_rasterizer {
        out.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default()
        });
    }
    out
}

struct GraphicsStreamBuild {
    hr: HRESULT,
    built_in_driver: std::time::Duration,
    reused: bool,
    cache_eligible: bool,
    used_rect_gs: bool,
}

/// HRESULTs documented for a cached pipeline made by another adapter or driver, plus the
/// invalid-argument result D3D12Core returns for an otherwise unreadable cached blob.
fn stale_cached_pso_refusal(hr: HRESULT) -> bool {
    matches!(hr as u32, 0x8007_0057 | 0x887e_0001 | 0x887e_0002)
}

fn create_graphics_stream_cached(
    stream: &StreamOut,
    without_rect_gs: Option<&StreamOut>,
    cached: Option<&[u8]>,
    mut on_stale_cache: impl FnMut(),
    mut run: impl FnMut(&str, bool, &StreamOut) -> (HRESULT, std::time::Duration),
) -> GraphicsStreamBuild {
    if let Some(blob) = cached {
        let mut with_cache = stream.clone();
        with_cache.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO, &D3D12_CACHED_PIPELINE_STATE {
            pCachedBlob: blob.as_ptr() as *const c_void,
            CachedBlobSizeInBytes: blob.len(),
        });
        let (hr, took) = run("graphics stream", true, &with_cache);
        if hr >= 0 {
            return GraphicsStreamBuild { hr, built_in_driver: took, reused: true, cache_eligible: false, used_rect_gs: without_rect_gs.is_some() };
        }
        if !stale_cached_pso_refusal(hr) {
            return GraphicsStreamBuild { hr, built_in_driver: took, reused: false, cache_eligible: false, used_rect_gs: without_rect_gs.is_some() };
        }
        // A cached blob describes the stream including its rect-list GS. Discard it before
        // deciding whether the driver objects to that GS rather than to the stale blob.
        on_stale_cache();
    }
    let label = if cached.is_some() { "graphics stream retry" } else { "graphics stream" };
    let (hr, took) = run(label, false, stream);
    if rejected_rect_list_gs(hr, without_rect_gs.is_some()) {
        note_translation_error(format!("the driver rejected the graphics stream with the rect list GS ({hr:#x}): rect list drawn as triangles"));
        let (fallback, fallback_took) = run("graphics stream without rect GS", false, without_rect_gs.unwrap());
        return GraphicsStreamBuild {
            hr: fallback,
            built_in_driver: fallback_took,
            reused: false,
            // The fallback is a different pipeline and must never replace the with-GS key.
            cache_eligible: false,
            used_rect_gs: false,
        };
    }
    GraphicsStreamBuild {
        hr,
        built_in_driver: took,
        reused: false,
        cache_eligible: hr >= 0,
        used_rect_gs: without_rect_gs.is_some(),
    }
}

unsafe fn translated_graphics_stream(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    with_switch_fallback(|| translated_graphics_stream_once(this, desc, riid, ppv))
}

unsafe fn translated_graphics_stream_once(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    use crate::gpu::recomp::pso::{console_to_dxil_for, is_console, Stage};
    if !crate::gpu::shader::readable(desc as usize, 16) { return None; }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) { return None; }
    let (subs, cacheable) = known_subobjects_checked(std::slice::from_raw_parts(base as *const u8, size));
    if !cacheable { return None; }
    if subobjects_have_unreadable_shader(&subs) { return Some(E_INVALIDARG); }
    let vs = subs.iter().find(|s| s.0 == 1).and_then(|s| payload_shader(s.1)).filter(|b| is_console(b))?;
    if subs.iter().any(|s| matches!(s.0, 3 | 4) && payload_shader(s.1).is_some()) {
        note_translation_error("graphics stream with HS/DS – not supported yet".into());
        return Some(E_INVALIDARG);
    }
    if super::pso_async::lazy() && crate::gpu::recomp::dxc::defer() { return None; }
    let split = graphics_wave32(this);
    let vs_c = match console_to_dxil_for(vs, Stage::Vertex, 0, split) { Ok(b) => b, Err(e) => { note_translation_error(format!("VS: {e}")); return Some(E_INVALIDARG); } };
    let mut samplers = vs_c.samplers;
    let console_ps = subs.iter().find(|s| s.0 == 2).and_then(|s| payload_shader(s.1));
    let ps_dxil = match console_ps {
        Some(p) if is_console(p) => match console_to_dxil_for(p, Stage::Pixel, samplers.len() as u32, split) {
            Ok(b) => { samplers.extend(b.samplers); Some(b.dxil) }
            Err(e) => { note_translation_error(format!("PS: {e}")); return Some(E_INVALIDARG); }
        },
        Some(p) => Some(p.to_vec()),
        None => None,
    };
    let is_rect = subs.iter().find(|s| s.0 == 14 && s.1.len() >= 4).is_some_and(|s| i32::from_le_bytes(s.1[0..4].try_into().unwrap()) == RECT_LIST_TYPE);
    let rect_gs = if is_rect && !subs.iter().any(|s| s.0 == 5 && payload_shader(s.1).is_some()) { rect_list_gs(vs, console_ps).map_err(|e| note_translation_error(format!("rect list drawn as triangles: {e}"))).ok() } else { None };
    let Some(rs) = abi_rs_for(this, &samplers) else { return Some(E_INVALIDARG) };
    let ps_hash = console_ps.map_or(0, crate::gpu::shader::fnv64);
    let (no_stencil, no_cull) = (debug_pso_switch("XCL_DEBUG_NO_STENCIL", ps_hash), debug_pso_switch("XCL_DEBUG_NO_CULL", ps_hash));
    let fs: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
    let stream = rewrite_graphics_stream(&subs, rs, &vs_c.dxil, ps_dxil.as_deref(), rect_gs.as_deref(), no_stencil, no_cull);
    let without_rect_gs = rect_gs.as_ref().map(|_| rewrite_graphics_stream(&subs, rs, &vs_c.dxil, ps_dxil.as_deref(), None, no_stencil, no_cull));
    let disk_key = two_pass_identity::graphics_stream_key(&stream.0, &samplers, crate::gpu::recomp::pso::mem_windows());
    let key = disk_key.map(|(hash, bytes)| (hash, bytes, rs as usize));
    let mut cached = key.and_then(graphics_blob);
    let disk = disk_key.and_then(|k| driver_cache::graphics_path(k.0, k.1));
    let from_disk = key.is_some() && cached.is_none();
    let mut loaded = None;
    if from_disk {
        loaded = disk.as_deref().and_then(driver_cache::load);
        cached = loaded.as_ref().map(driver_cache::LoadedBlob::shared);
    }
    let submit = |label: &str, has_cached: bool, out: &StreamOut| {
        let d = [out.0.len(), out.0.as_ptr() as usize];
        driver_compile_identified(label, out.shader_bytes(), gate_compute_driver_compile(has_cached), Some(ps_hash), has_cached,
            || finish_pso(this, fs, d.as_ptr() as *const c_void, riid, ppv))
    };
    let rejected_generation = cached.clone();
    let build = create_graphics_stream_cached(
        &stream,
        without_rect_gs.as_ref(),
        cached.as_deref().map(Vec::as_slice),
        || {
            if let (Some(k), Some(expected)) = (key, rejected_generation.as_ref()) {
                forget_graphics_blob_if(k, expected);
            }
            if from_disk { driver_cache::note(false, loaded.take()); }
        },
        submit,
    );
    let hr = build.hr;
    if let Some(k) = key.filter(|_| build.cache_eligible) {
        keep_driver_blob(&GRAPHICS_BLOBS, GRAPHICS_BLOBS_MAX, k, stream.shader_bytes(), disk.as_deref(), build.built_in_driver,
            || read_pso_cached_blob(*ppv));
    } else if build.reused && from_disk {
        GRAPHICS_BLOBS.admit(key.unwrap(), cached.as_ref().unwrap().clone(), GRAPHICS_BLOBS_MAX);
        driver_cache::note(true, loaded.take());
    }
    if hr < 0 {
        note_translation_error(format!("the driver rejected the graphics stream with the translated shaders ({hr:#x})"));
        return Some(hr);
    }
    let n = TRANSLATED_PSO.fetch_add(1, Ordering::Relaxed);
    if !ppv.is_null() && !(*ppv).is_null() {
        note_pso_rs(*ppv, rs);
        super::dred::label(*ppv as usize, &[("vs", Some(vs)), ("ps", console_ps)]);
        TRANSLATED_GRAPHICS.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
        if build.used_rect_gs { RECT_LIST_GS.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize); }
    }
    if n < 5 || n % 1000 == 0 { crate::xlog_info!("D3D12.X: graphics stream PSO from translated console shaders (#{})", n + 1); }
    Some(hr)
}

/// The bytes of a stream payload as a desc field (no references counted: the desc only borrows).
unsafe fn put<T>(to: &mut T, from: &[u8]) {
    let n = std::mem::size_of::<T>().min(from.len());
    std::ptr::copy_nonoverlapping(from.as_ptr(), to as *mut T as *mut u8, n);
}

/// True when a pipeline stream carries a shader subobject whose non-empty bytecode has a null or
/// unreadable pointer (handing it to D3D12Core would access-violate instead of failing).
unsafe fn stream_has_unreadable_shader(desc: *const c_void) -> bool {
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return false;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return false;
    }
    let buf = std::slice::from_raw_parts(base as *const u8, size);
    subobjects_have_unreadable_shader(&known_subobjects(buf))
}

fn subobjects_have_unreadable_shader(subs: &[(u32, &[u8])]) -> bool {
    subs.iter().any(|(ty, p)| {
        matches!(ty, 1..=6 | 24 | 25) && p.len() >= 16 && {
            let ptr = u64::from_le_bytes(p[0..8].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(p[8..16].try_into().unwrap()) as usize;
            (ptr == 0 && len != 0) || (ptr != 0 && len != 0 && !crate::gpu::shader::readable(ptr, len))
        }
    })
}

pub(super) unsafe extern "system" fn dev_create_pipeline_state(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    super::null::capture_stream_desc(desc as *const u8);
    let f: PsoFn = std::mem::transmute(orig_fn(this, slot_of(this, "CreatePipelineState")));
    let (translated, missed) = if super::pso_async::enabled() { attempt(|| translated_mesh_stream(this, desc, riid, ppv)) } else { (translated_mesh_stream(this, desc, riid, ppv), false) };
    if missed {
        if let Some(hr) = own_stream(desc).and_then(|o| go_background(this, f, desc, riid, ppv, false, o, standin_stream, stream_key, build_stream)) {
            return hr;
        }
        if let Some(hr) = translated_mesh_stream(this, desc, riid, ppv) {
            return hr;
        }
    } else if let Some(hr) = translated {
        return hr;
    }
    match stream_pipeline(desc) {
        StreamPipeline::Graphics(d, cacheable) => return graphics_pso(this, &*d as *const _ as *const c_void, riid, ppv, true, cacheable),
        StreamPipeline::GraphicsStream => {
            let (translated, missed) = if super::pso_async::enabled() { attempt(|| translated_graphics_stream(this, desc, riid, ppv)) } else { (translated_graphics_stream(this, desc, riid, ppv), false) };
            if missed {
                if let Some(hr) = own_stream(desc).and_then(|o| go_background(this, f, desc, riid, ppv, false, o, standin_stream, stream_key, build_stream)) {
                    return hr;
                }
                return translated_graphics_stream(this, desc, riid, ppv).unwrap_or(E_INVALIDARG);
            }
            return translated.unwrap_or(E_INVALIDARG);
        }
        StreamPipeline::Compute(d) => return dev_create_compute_pso(this, &*d as *const _ as *const c_void, riid, ppv),
        StreamPipeline::Unknown => {}
    }
    if stream_has_unreadable_shader(desc) {
        return E_INVALIDARG;
    }
    let hr = finish_pso(this, f, desc, riid, ppv);
    if hr >= 0 || !crate::gpu::shader::readable(desc as usize, 16) {
        return hr;
    }
    let Some((buf, _)) = fallback_stream(desc as *const u8) else {
        note_fallback("PSO stream", hr, false);
        return hr;
    };
    let new_desc: [usize; 2] = [buf.len(), buf.as_ptr() as usize];
    let hr2 = finish_pso(this, f, new_desc.as_ptr() as *const c_void, riid, ppv);
    note_fallback("PSO stream", hr, hr2 >= 0);
    if hr2 >= 0 {
        hr2
    } else {
        hr
    }
}

unsafe extern "system" fn dev_create_root_signature(this: *mut c_void, mask: u32, blob: *const c_void, len: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    crate::gpu::shader::capture("rootsig", blob as usize, len);
    type F = unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const GUID, *mut *mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "CreateRootSignature")));
    let mut hr = f(this, mask, blob, len, riid, ppv);
    if hr == E_NOINTERFACE {
        let iid = pc_iid(Kind::RootSignature);
        hr = f(this, mask, blob, len, &iid, ppv);
    }
    if hr < 0 {
        // console root signature rejected → empty one so the title keeps going
        let desc = D3D12_ROOT_SIGNATURE_DESC { Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, ..Default::default() };
        let mut ser = None;
        if D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).is_ok() {
            if let Some(b) = ser {
                let iid = pc_iid(Kind::RootSignature);
                let hr2 = f(this, mask, b.GetBufferPointer(), b.GetBufferSize(), &iid, ppv);
                if hr2 >= 0 {
                    let n = FALLBACK_RS.fetch_add(1, Ordering::Relaxed);
                    if n < 5 {
                        crate::xlog_warn!("D3D12.X: root signature rejected ({hr:#x}) – empty fallback (#{})", n + 1);
                    }
                    hr = hr2;
                }
            }
        }
    }
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() {
        adopt(*ppv, Kind::RootSignature);
        if !blob.is_null() {
            super::console_list::note_root_signature(*ppv as usize, std::slice::from_raw_parts(blob as *const u8, len));
        }
    }
    hr
}

// ---------------------------------------------------------------------------------------
// presentation tracking
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn dev_create_rtv(this: *mut c_void, res: *mut c_void, desc: *const c_void, handle: usize) {
    let _detail = crate::stall::detail_span_if(!super::title_textures::is_own_work(), crate::stall::StallKind::DetailViewDescriptor);
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void, usize);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "CreateRenderTargetView")));
    // a view in a console-only format, or of a target made R8G8B8A8 (`device_x::pc_resource_desc`)
    let mut copy = (!desc.is_null()).then(|| (desc as *const D3D12_RENDER_TARGET_VIEW_DESC).read_unaligned());
    if let Some(c) = copy.as_mut() {
        let rf = ID3D12Resource::from_raw_borrowed(&res).map_or(0, |r| r.GetDesc().Format.0 as u32);
        c.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(super::device_x::pc_shader_view_format(rf, c.Format.0 as u32) as i32);
    }
    let desc = copy.as_ref().map_or(desc, |c| c as *const _ as *const c_void);
    f(this, res, desc, handle);
    super::present::note_rtv(handle, res);
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 12 {
            let d = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc()).map(|d| (d.Width, d.Height, d.Format.0));
            crate::xlog_info!("frame trace: CreateRenderTargetView {handle:#x} → {res:p} {d:?}");
        }
    }
}

/// ClearRenderTargetView: as on the PC (`XCL_TRACE_FRAME`: the colours logged)
unsafe extern "system" fn cl_clear_rtv(this: *mut c_void, rtv: usize, color: *const [f32; 4], n: u32, rects: *const c_void) {
    let _detail = crate::stall::detail_span_if(!super::title_textures::is_own_work(), crate::stall::StallKind::DetailCopy);
    type F = unsafe extern "system" fn(*mut c_void, usize, *const [f32; 4], u32, *const c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "ClearRenderTargetView")));
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 16 {
            crate::xlog_info!("frame trace: ClearRenderTargetView {rtv:#x} → {:?} colour {:?}, {n} rects", super::present::rtv_resource(rtv), color.as_ref());
        }
        if let Some(r) = super::present::rtv_resource(rtv) {
            super::present::note_use(r, super::present::USE_CLEAR);
        }
    }
    f(this, rtv, color, n, rects)
}

unsafe extern "system" fn cl_om_set_render_targets(this: *mut c_void, n: u32, rtvs: *const usize, single: i32, dsv: *const usize) {
    let _detail = crate::stall::detail_span_if(!super::title_textures::is_own_work(), crate::stall::StallKind::DetailBind);
    type F = unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize);
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "OMSetRenderTargets")));
    f(this, n, rtvs, single, dsv);
    super::list_x::note_render_targets(this, n, rtvs, single, dsv);
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 12 {
            let first = if n > 0 && !rtvs.is_null() { *rtvs } else { 0 };
            crate::xlog_info!("frame trace: OMSetRenderTargets({n}, first {first:#x} → {:?}, single {single}, dsv {:?})", super::present::rtv_resource(first), dsv.as_ref().copied());
        }
    }
    if n > 0 && !rtvs.is_null() {
        super::present::note_rt0(this as usize, *rtvs);
        if super::present::trace_frame() {
            for k in 0..n as usize {
                let h = if single != 0 { *rtvs + k * 32 } else { *rtvs.add(k) };
                if let Some(r) = super::present::rtv_resource(h) {
                    super::present::note_use(r, super::present::USE_RT);
                }
            }
        }
    }
}

/// Bind an already-selected exact native generation without sending its raw pointer back through
/// the title-address resolver. Used by direct Dispatch after its retained bind is consumed.
pub(super) unsafe fn bind_pipeline_exact(this: *mut c_void, original: usize, raw: usize) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "SetPipelineState")));
    f(this, raw as *mut c_void);
    super::dred::mark_pipeline(this, original, raw);
}

/// Clear native pipeline state with an already-selected raw pointer. `raw == 0` is the
/// API-specific fail-closed action for an unavailable retained identity.
pub(super) unsafe fn clear_pipeline_exact(this: *mut c_void, original: usize, raw: usize) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "ClearState")));
    f(this, raw as *mut c_void);
    if original != 0 && raw != 0 {
        super::dred::mark_pipeline(this, original, raw);
    }
}

unsafe extern "system" fn cl_set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    super::bind::set_pipeline_state(this, pso)
}

unsafe extern "system" fn cl_execute_bundle(this: *mut c_void, bundle: *mut c_void) {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void);
    super::list_x::on_execute_bundle(this as usize, bundle as usize);
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "ExecuteBundle")));
    f(this, bundle);
}

/// The title's markers are counted for `XCL_DRED=1`'s pipeline labels (`dred`).
unsafe extern "system" fn cl_set_marker(this: *mut c_void, metadata: u32, data: *const c_void, size: u32) {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const c_void, u32);
    let f: F = std::mem::transmute(orig_fn(this, slot_of(this, "SetMarker")));
    f(this, metadata, data, size);
    super::dred::title_marker(this);
}

unsafe extern "system" fn cl_clear_state(this: *mut c_void, pso: *mut c_void) {
    super::bind::clear_pipeline_state(this, pso as usize)
}

unsafe extern "system" fn cl_draw_instanced(this: *mut c_void, vcount: u32, icount: u32, sv: u32, si: u32) {
    super::list_x::prepare_draw(this);
    type F = unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32);
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "DrawInstanced")));
    f(this, vcount, icount, sv, si);
}

unsafe extern "system" fn cl_draw_indexed_instanced(this: *mut c_void, icount: u32, inst: u32, si: u32, bv: i32, sinst: u32) {
    super::list_x::prepare_draw(this);
    type F = unsafe extern "system" fn(*mut c_void, u32, u32, u32, i32, u32);
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "DrawIndexedInstanced")));
    f(this, icount, inst, si, bv, sinst);
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn cl_execute_indirect(
    this: *mut c_void,
    sig: *mut c_void,
    max_count: u32,
    arg_buf: *mut c_void,
    arg_off: u64,
    cnt_buf: *mut c_void,
    cnt_off: u64,
) {
    super::list_x::prepare_draw(this);
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut c_void, u64, *mut c_void, u64);
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "ExecuteIndirect")));
    f(this, sig, max_count, arg_buf, arg_off, cnt_buf, cnt_off);
}

/// The last reference: what XCL keeps by the list's address goes before the runtime frees it,
/// as a list created on another thread can get the address at once. (Two last-but-one releases
/// at once miss each other here; the list's state then goes after it, as before.)
unsafe extern "system" fn cl_release(this: *mut c_void) -> u32 {
    let add: RefFn = std::mem::transmute(orig_fn(this, 1));
    let f: RefFn = std::mem::transmute(orig_fn(this, 2));
    let last = add(this) == 2;
    f(this);
    if last {
        forget_command_list(this);
    }
    let n = f(this);
    if n == 0 && !last {
        forget_command_list(this);
    }
    n
}

fn forget_command_list(list: *mut c_void) {
    super::list_x::on_release(list as usize);
    super::bind::forget(list);
    super::dred::forget_list(list);
}

unsafe extern "system" fn q_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    let n = super::queue_submit::release(this as usize, || f(this));
    if n == 0 {
        super::list_x::on_queue_release(this as usize);
    } else if n == 1 {
        super::present::retire_released_queue(this as usize);
    }
    n
}

thread_local! {
    static RETAINED_ALLOC_RELEASE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn release_retained_allocator(owner: ID3D12CommandAllocator) {
    let raw = owner.as_raw() as usize;
    RETAINED_ALLOC_RELEASE.with(|active| {
        let old = active.replace(raw);
        debug_assert_eq!(old, 0);
        drop(owner);
        active.set(old);
    });
}

unsafe extern "system" fn alloc_release(this: *mut c_void) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 2));
    if RETAINED_ALLOC_RELEASE.with(|active| active.get() == this as usize) {
        return f(this);
    }
    let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(orig_fn(this, 1));
    let (n, _) = super::alloc_pool::release_with_probe(
        || { add_ref(this); },
        || f(this),
        || super::alloc_pool::retain_release_probe(this),
    );
    if n == 0 {
        dispatch_chain::allocator_reset(this as usize);
        super::alloc_pool::forget(this as usize);
        super::list_x::on_allocator_release(this as usize);
    }
    n
}

/// Reset the (adopted) PC allocator `raw` itself.
pub(super) unsafe fn reset_pc_allocator(raw: *mut c_void) -> HRESULT {
    type F = unsafe extern "system" fn(*mut c_void) -> HRESULT;
    let f: F = std::mem::transmute(orig_fn(raw, slot!(raw, "Reset")));
    let hr = f(raw);
    if hr >= 0 {
        dispatch_chain::allocator_reset(raw as usize);
        // The GPU is done with allocations owned by this exact native backing.
        super::console_list::allocator_reset(raw);
    }
    hr
}

unsafe extern "system" fn alloc_reset(this: *mut c_void) -> HRESULT {
    let _detail = crate::stall::detail_span_if(!super::title_textures::is_own_work(), crate::stall::StallKind::DetailListLifecycle);
    // Lists may still hold recordings on it that the title submits again (`alloc_pool`). UMA
    // follows each native backing and is released only when reset_pc_allocator resets that one.
    match super::alloc_pool::title_reset(this) {
        Some(hr) => hr,
        None => reset_pc_allocator(this),
    }
}

// ---------------------------------------------------------------------------------------
// command queue layouts
// ---------------------------------------------------------------------------------------
//
// Each console title runs on the game OS it ships with, and the D3D12 runtime there sets the
// queue's vtable. Two layouts are known, with the same D3D12 SDK version (0x20004) and the
// same slots up to ExecuteCommandLists (#10):
// * the older (XDefiant, Minecraft, Fortnite): the PC's #11–#18 (SetMarker … Signal #14, Wait
//   #15, GetTimestampFrequency #16, GetClockCalibration #17, GetDesc #18), SignalX #23,
//   PresentX #30;
// * a newer one (Unicorn Overlord): two more methods before Signal – Signal #16, Wait #17,
//   GetTimestampFrequency #18, GetClockCalibration #19, GetDesc #20, SignalX #25, PresentX #32.
// Nothing the title passes at device creation tells them apart, but its calls do: Signal and
// Wait take a fence, the older layout's #16 / #17 (GetTimestampFrequency / GetClockCalibration)
// never do. A title calling #16 / #17 with one of its fences uses the newer layout, and the
// slots that differ follow that from then on.
//
// The same runtime moves the command list's console methods too: Unicorn Overlord records its
// texture uploads with CopyTextureRegionX at #85 (the older layout's #83; its #85 is
// SetPredicationBufferX, which dropped them – every texture stayed zero). From #85 on the
// list's console methods sit two slots later (`NEWER_LIST_SLOTS`), and from #72 on as well: a retail game
// calls Write64BitValueBottomOfPipeX at #72 and Wait64BitValueX at #74, Write32BitValueBottomOfPipeX
// at #71 (address, value, 0, 0) and ExecuteIndirect at #61 (signature, 1, argument buffer, offset,
// NULL, 0) – sent to FillMemoryWith32BitValueX, the title's indirect draws and dispatches never ran.
// So the shift starts at #59 or below; where is not known. The newer layout's #139 is a Reset with one argument more (the same
// title: (allocator, NULL, 0) on each list it takes from its pool before recording – as a stub
// the list stayed closed, the debug layer saw every call on it refused, and Close failed).

/// the title's runtime has the newer layouts (queue: two methods before Signal; command list:
/// console methods from CopyTextureRegionX on two slots later)
pub(super) static QUEUE_SHIFTED: AtomicBool = AtomicBool::new(false);

/// the title's runtime has the older queue layout: it signalled / waited on a fence at #14 / #15
/// (on the newer layout those slots are other methods)
static QUEUE_OLDER: AtomicBool = AtomicBool::new(false);

/// command list slots of the newer layout → the older layout's slot of the same method
/// (`DispatchMesh` sits at newer #129 after #126 RSSetShadingRate, #127 RSSetShadingRateImage,
/// and #128 Scarlett-only shading-rate/color struct; older #127, two slots before as the others).
const NEWER_LIST_SLOTS: [(usize, usize); 22] = [(61, 59), (62, 60), (63, 61), (64, 62), (65, 63), (70, 68), (71, 69), (72, 70), (73, 71), (74, 72), (80, 78), (83, 81), (85, 83), (86, 84), (87, 85), (90, 88), (91, 89), (92, 90), (95, 93), (96, 94), (129, 127), (139, 10)];

/// The title called device method `slot`, which only runtimes with the newer layouts have (a retail game
/// records its first lists before any queue call could tell).
pub(super) fn note_newer_device_method(slot: usize) {
    if !QUEUE_OLDER.load(Ordering::Acquire) && !QUEUE_SHIFTED.swap(true, Ordering::AcqRel) {
        crate::xlog_info!("D3D12.X: the title's runtime has the newer console layouts (device method #{slot} called)");
    }
}

/// Whether `p` is a fence the title created.
fn is_fence(p: usize) -> bool {
    p != 0 && FENCES.read().unwrap_or_else(|e| e.into_inner()).contains(&p)
}

/// first queue slot `queue_dispatch` handles
const QUEUE_FIRST: usize = 14;

/// A title's call of queue slot `slot` (arguments as integers; Signal / Wait / SignalX /
/// PresentX / GetTimestampFrequency / GetClockCalibration / GetDesc all fit).
unsafe fn queue_dispatch(slot: usize, this: *mut c_void, a1: usize, a2: usize, a3: usize, a4: usize) -> usize {
    const PC_SIGNAL: usize = 14;
    const PC_WAIT: usize = 15;
    const PC_TIMESTAMP_FREQUENCY: usize = 16;
    const PC_CLOCK_CALIBRATION: usize = 17;
    const PC_GET_DESC: usize = 18;
    if (slot == 16 || slot == 17) && is_fence(a1) && !QUEUE_SHIFTED.swap(true, Ordering::AcqRel) {
        crate::xlog_info!("D3D12.X: the title's runtime has the newer console layouts (queue: Signal #16, PresentX #32; command list: CopyTextureRegionX #85 – a fence passed to queue #{slot})");
    }
    let shifted = QUEUE_SHIFTED.load(Ordering::Acquire);
    if (slot == 14 || slot == 15) && !shifted && is_fence(a1) {
        QUEUE_OLDER.store(true, Ordering::Release);
    }
    // the method in the older layout's numbering
    let older = if shifted && slot >= 16 { slot - 2 } else if shifted { usize::MAX } else { slot };
    let pc = |s: usize| orig_fn(this, s);
    match older {
        // Signal takes a fence: prepare outside the turn, then validate/insert atomically with it.
        14 if is_fence(a1) => loop {
            let prepared = super::title_textures::prepare_downloads(this);
            let attempt = super::queue_submit::with(this as usize, || {
                let downloads = match super::title_textures::submit_prepared_downloads(this, prepared) {
                    Ok(downloads) => downloads,
                    Err(super::title_textures::SubmitDownloadsError::Stale) => return Err(None),
                    Err(super::title_textures::SubmitDownloadsError::Failed(error)) => return Err(Some(error)),
                };
                super::list_x::break_cycles_before_title_signal(this as usize, a1, a2 as u64);
                super::title_textures::tie_downloads(a1 as *mut c_void, a2 as u64, downloads);
                let f: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 = std::mem::transmute(pc(PC_SIGNAL));
                let hr = f(this, a1, a2);
                if hr >= 0 {
                    super::fence_x::note_queue_signal(a1, a2 as u64);
                    super::list_x::note_title_queue_signal(this as usize, a1, a2 as u64);
                }
                Ok(hr as u32 as usize)
            });
            match attempt {
                Ok(hr) => break hr,
                Err(None) => continue,
                Err(Some(error)) => {
                    crate::xlog_warn!("D3D12.X: {error}; title queue Signal not submitted");
                    break crate::util::E_FAIL as u32 as usize;
                }
            }
        },
        15 if is_fence(a1) => super::queue_submit::with(this as usize, || {
            let f: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 = std::mem::transmute(pc(PC_WAIT));
            let hr = f(this, a1, a2);
            if hr >= 0 {
                super::fence_x::note_queue_wait(a1, a2 as u64);
                super::list_x::note_title_queue_wait(this as usize, a1, a2 as u64);
            }
            hr as u32 as usize
        }),
        16 => {
            let f: unsafe extern "system" fn(*mut c_void, usize) -> i32 = std::mem::transmute(pc(PC_TIMESTAMP_FREQUENCY));
            f(this, a1) as u32 as usize
        }
        17 => {
            let f: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 = std::mem::transmute(pc(PC_CLOCK_CALIBRATION));
            f(this, a1, a2) as u32 as usize
        }
        // #18 before either layout is known: the older layout's GetDesc (a 16-byte structure) or
        // the newer one's GetTimestampFrequency (a UINT64). Only the first 8 bytes of the
        // description are written (type and priority), so an 8-byte out-param is not overrun.
        18 if !shifted && !QUEUE_OLDER.load(Ordering::Acquire) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_warn!("D3D12.X: queue #18 before the title's first Signal / Wait: the queue layout is not known yet, GetDesc writes only its first 8 bytes"));
            let f: unsafe extern "system" fn(*mut c_void, usize) -> usize = std::mem::transmute(pc(PC_GET_DESC));
            let mut desc = [0u8; 16];
            f(this, desc.as_mut_ptr() as usize);
            if a1 != 0 {
                std::ptr::copy_nonoverlapping(desc.as_ptr(), a1 as *mut u8, 8);
            }
            a1
        }
        // GetDesc returns its structure through the hidden pointer, and that pointer
        18 => {
            let f: unsafe extern "system" fn(*mut c_void, usize) -> usize = std::mem::transmute(pc(PC_GET_DESC));
            f(this, a1)
        }
        23 => q_signal_x(this, a1 as *mut c_void, a2 as u64, a3 as u32) as u32 as usize,
        30 => super::present::present_x(this, a1 as u32, a2 as *const u8, a3 as *const u8) as u32 as usize,
        // the newer layout's #35: ExecuteCommandListsX(count, lists, a value, flags) – a retail
        // game submits all its command lists this way (flags 3; 0 for the one it recorded at
        // start); as a stub nothing the title recorded ran on the GPU but its first two frames
        33 if shifted => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_info!("D3D12.X: queue #35 ExecuteCommandListsX ({} lists, value {a3:#x}, flags {a4:#x})", a1));
            q_execute_command_lists(this, a1 as u32, a2 as *const *mut c_void);
            0
        }
        _ => {
            // a console method not modelled: logged once, S_OK (as the stubs it replaces; frame
            // pacing on queue methods keeps working for titles without PresentX)
            static INFOS: Mutex<Vec<(usize, &'static stubs::StubInfo)>> = Mutex::new(Vec::new());
            let info = {
                let mut g = INFOS.lock().unwrap();
                match g.iter().find(|e| e.0 == slot) {
                    Some(e) => e.1,
                    None => {
                        let i = stubs::new_info(DLL, &format!("ID3D12CommandQueue::#{slot} ({} layout)", if shifted { "newer" } else { "older" }), 0);
                        i.hook.store(super::present::on_queue_console_method as usize, Ordering::Release);
                        g.push((slot, i));
                        i
                    }
                }
            };
            let _ = a4;
            stubs::record_call(info, 0);
            0
        }
    }
}

macro_rules! queue_slots {
    ($($slot:literal),*) => {
        /// the title's entry for each queue slot from `QUEUE_FIRST`
        static QUEUE_SLOTS: &[unsafe extern "system" fn(*mut c_void, usize, usize, usize, usize) -> usize] = &[$({
            unsafe extern "system" fn f(this: *mut c_void, a1: usize, a2: usize, a3: usize, a4: usize) -> usize {
                queue_dispatch($slot, this, a1, a2, a3, a4)
            }
            f
        }),*];
    };
}
queue_slots!(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35);

/// command queue #23 SignalX(fence, value, flush flags): Signal with the console's cache flush
/// options – the PC's Signal makes the queue's writes visible itself. (A stub that signalled
/// nothing left Fortnite's render fences pending: its render thread waited, textures were
/// never released, and the garbage collector ended the game after 10 s.)
unsafe extern "system" fn q_signal_x(this: *mut c_void, fence: *mut c_void, value: u64, _flags: u32) -> HRESULT {
    loop {
        let prepared = super::title_textures::prepare_downloads(this);
        match super::queue_submit::with(this as usize, || q_signal_x_serialized(this, fence, value, prepared)) {
            Ok(hr) => return hr,
            Err(super::title_textures::SubmitDownloadsError::Stale) => continue,
            Err(super::title_textures::SubmitDownloadsError::Failed(error)) => {
                crate::xlog_warn!("D3D12.X: {error}; title queue SignalX not submitted");
                return crate::util::E_FAIL;
            }
        }
    }
}

unsafe fn q_signal_x_serialized(this: *mut c_void, fence: *mut c_void, value: u64, prepared: super::title_textures::PreparedDownloads) -> Result<HRESULT, super::title_textures::SubmitDownloadsError> {
    const PC_SIGNAL: usize = 14;
    let downloads = super::title_textures::submit_prepared_downloads(this, prepared)?;
    super::list_x::break_cycles_before_title_signal(this as usize, fence as usize, value);
    super::title_textures::tie_downloads(fence, value, downloads);
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT = std::mem::transmute(orig_fn(this, PC_SIGNAL));
    let hr = f(this, fence, value);
    if hr >= 0 {
        super::fence_x::note_queue_signal(fence as usize, value);
        super::list_x::note_title_queue_signal(this as usize, fence as usize, value);
    }
    Ok(hr)
}

/// Fence::Signal (CPU)
unsafe extern "system" fn fence_completed(this: *mut c_void) -> u64 {
    let f: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(orig_fn(this, slot_of(this, "GetCompletedValue")));
    super::title_textures::safe_fence_value(this as usize, f(this))
}

unsafe extern "system" fn fence_signal(this: *mut c_void, value: u64) -> HRESULT {
    let f: unsafe extern "system" fn(*mut c_void, u64) -> HRESULT = std::mem::transmute(orig_fn(this, slot!(this, "Signal")));
    let hr = f(this, value);
    super::fence_x::note_cpu_signal(this as usize);
    hr
}

unsafe extern "system" fn q_execute_command_lists(this: *mut c_void, n: u32, lists: *const *mut c_void) {
    type F = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let f: F = std::mem::transmute(orig_fn(this, slot!(this, "ExecuteCommandLists")));
    // Internal XCL submits (`sparse` uploads/readbacks, `title_textures` uploads, tile zeroing)
    // run inside `own_work()` — sometimes while holding internal locks (`sparse::STATE`,
    // `vram::PAGES`). Forwarding them through the native slot, in the queue's ordered turn, skips
    // `present::note_execute` -> `late_release::drain_known` -> `vram::recycle_ready`, preventing
    // re-entrant lock self-deadlocks and keeping internal queues out of `present::QUEUES`.
    if n == 0 || lists.is_null() || super::title_textures::is_own_work() {
        super::queue_submit::with(this as usize, || f(this, n, lists));
        stop_if_device_removed();
        return;
    }
    crate::stall::note_submitter();
    // from here to the end: XCL's uploads and tile mappings before the lists, then the driver's
    // own work in the submission. Where that takes long, it is not the title's time
    // (`pso_async::title_held`)
    let _in_pc = super::pso_async::pc_call();
    let mut preparation_stall = crate::stall::span(crate::stall::StallKind::SubmitPreparation);
    let title_lists = std::slice::from_raw_parts(lists, n as usize);
    // console-shaped lists run as the PC lists they were translated into (split title lists expand
    // into all of their recorded PC segments in exact chronological order)
    let mut mapped: Vec<*mut c_void> = Vec::with_capacity(n as usize);
    for &l in title_lists {
        if let Some(pc) = super::console_list::pc_list(l) {
            mapped.push(pc);
        } else {
            // Deferred recording can wait for a pipeline. Keep that time solely in PipelineWait.
            drop(preparation_stall);
            super::list_x::record_deferred(l);
            preparation_stall = crate::stall::span(crate::stall::StallKind::SubmitPreparation);
            if !super::title_list::push_segments(l, &mut mapped) {
                mapped.push(l);
            }
        }
    }
    if super::list_x::any_recording_poisoned(&mapped) {
        crate::xlog_warn!("D3D12.X: ExecuteCommandLists rejected a poisoned physical recording");
        return;
    }
    // textures in title memory the CPU wrote since: uploaded again first
    prepare_title_submission(this);
    super::list_x::note_queue_for_lists(this as usize, &mapped);
    let plan = super::list_x::submission_plan(&mapped);
    drop(preparation_stall);
    let mut chunks = 0usize;
    let mut gate = super::alloc_pool::SubmissionGate::default();
    let mut submitted = Vec::with_capacity(mapped.len());
    let track_shared_view_epochs = super::vram::srv_tracking_enabled();
    let _ = super::list_x::execute_submission_plan_until_failure_on(this, &plan, |chunk| {
        super::queue_submit::with(this as usize, || {
            let lists = chunk.lists();
            if !super::list_x::submit_title_chunk(
                &mut gate,
                this,
                lists,
                track_shared_view_epochs && chunk.uses_shared_view_heap(),
                chunk.has_native_generations(),
                chunk.native_generations(),
                || {
                    let _ = super::alloc_pool::note_resubmits(this, lists);
                },
                || {
                    let started = gpu_timing_on().then(std::time::Instant::now);
                    f(this, lists.len() as u32, lists.as_ptr());
                    if let Some(started) = started {
                        note_submission_timing(started.elapsed(), lists.len(), this);
                    }
                },
            ) {
                return (false, None);
            }
            chunks += 1;
            submitted.extend_from_slice(lists);
            super::list_x::copy_scratch_submitted(this, lists);
            (true, chunk.submit_effects(this))
        })
    });
    super::alloc_pool::note_execute_call(chunks);
    super::present::note_execute(this, &submitted);
    stop_if_device_removed();
}

/// Complete correctness-critical uploads before any native title list executes. The D3D12
/// submission API is void, so returning after preparation failure would silently lose the
/// title's exact submission; an unorderable failure must end the title instead.
pub(super) unsafe fn prepare_title_submission(queue: *mut c_void) {
    use super::sparse::PreparationOutcome;
    let outcome = super::title_textures::refresh(queue);
    if outcome == PreparationOutcome::Ready {
        return;
    }
    stop_if_device_removed();
    let reason = match outcome {
        PreparationOutcome::RetryRequired => "sparse upload preparation still failed after bounded retries",
        PreparationOutcome::ExecutedUntracked => "a sparse upload executed without a retirement signal",
        PreparationOutcome::Ready => unreachable!(),
    };
    let message = format!("{reason} – the title is stopped before its GPU submission");
    crate::xlog_error!("{message}");
    crate::report::write_final(Some(message));
    end_title(0x8000_4005);
}

/// Wait (up to 10 s) until the GPU has finished the work submitted to the queues console code
/// used (before the process ends on purpose).
pub fn drain_gpu() {
    use crate::gpu::fence::{self, FenceError};

    let raw = device_raw();
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    unsafe {
        // a fence per queue (they finish in any order)
        let mut fences = Vec::new();
        for q in super::present::active_queues_held() {
            let submit = || {
                let f: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)
                    .map_err(|e| fence::api_error(dev, "CreateFence", e))?;
                native_queue_signal(&q, &f, 1).map_err(|e| fence::api_error(dev, "Signal", e))?;
                Ok(f)
            };
            match submit() {
                Ok(f) => fences.push(f),
                Err(e) => {
                    crate::xlog_error!("D3D12 queue drain: {e}");
                    if matches!(e, FenceError::Removed(_)) {
                        return;
                    }
                }
            }
        }
        let start = std::time::Instant::now();
        loop {
            let mut pending = false;
            for f in &fences {
                match fence::completed_value(f, dev) {
                    Ok(value) => pending |= value < 1,
                    Err(e) => {
                        crate::xlog_error!("D3D12 queue drain: {e}");
                        return;
                    }
                }
            }
            if !pending {
                return;
            }
            if start.elapsed().as_secs() >= 10 {
                crate::xlog_warn!("D3D12 queue drain: GPU fences still pending after 10 s");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}

/// DRED gives us a linked list in driver order. Keep the diagnostic bounded even on a
/// long chain, with room in the log for work that the GPU had already started.
unsafe fn report_dred_lists(head: *const D3D12_AUTO_BREADCRUMB_NODE, mut report: impl FnMut(&D3D12_AUTO_BREADCRUMB_NODE, usize)) {
    const SCAN_LIMIT: usize = 4096;
    const REPORT_LIMIT: usize = 64;
    let nodes = || std::iter::successors(head.as_ref(), |node| node.pNext.as_ref()).take(SCAN_LIMIT);
    let mut reported = 0;
    for started in [true, false] {
        for node in nodes() {
            let done = node.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
            if done < node.BreadcrumbCount && (done != 0) == started {
                report(node, done as usize);
                reported += 1;
                if reported == REPORT_LIMIT { return; }
            }
        }
    }
}

#[cfg(test)]
mod submission_timing_tests {
    use super::*;

    #[test]
    fn submission_timing_keeps_exact_total_histogram_and_slowest_sample() {
        let mut timing = SubmissionTiming::default();
        timing.record(0, 2);
        timing.record(1, 9);
        timing.record(2, 3);
        timing.record(3, 5);
        timing.record(8, 4);
        assert_eq!((timing.submissions, timing.total_ns, timing.slowest_ns, timing.slowest_lists), (5, 14, 8, 4));
        assert_eq!((timing.histogram[0], timing.histogram[1], timing.histogram[3]), (2, 2, 1));
        assert_eq!((timing.percentile_upper_ns(95), timing.percentile_upper_ns(99)), (15, 15));
    }

    #[test]
    fn submission_timing_saturates_totals_and_uses_bounded_last_bucket() {
        let mut timing = SubmissionTiming { submissions: u64::MAX, total_ns: u64::MAX, ..Default::default() };
        timing.record(u64::MAX, 7);
        assert_eq!((timing.submissions, timing.total_ns, timing.slowest_ns), (u64::MAX, u64::MAX, u64::MAX));
        assert_eq!(timing.histogram[crate::stall::HISTOGRAM_BUCKETS - 1], 1);
        assert_eq!(timing.percentile_upper_ns(99), u64::MAX);
    }

    #[test]
    fn gpu_completion_timing_retires_only_the_exact_completed_receipt() {
        let (first, first_completed) = super::super::alloc_pool::submission_receipt_for_test(0, 3);
        let (second, second_completed) = super::super::alloc_pool::submission_receipt_for_test(0, 7);
        let second_identity = second.fence_identity();
        let mut timing = GpuCompletionTiming::default();
        // A reused raw queue address cannot conflate the retained fence generations.
        timing.push(first, 2, 0x7000);
        timing.push(second, 5, 0x7000);

        first_completed.store(3, Ordering::Release);
        timing.retire_completed();
        assert_eq!(timing.tracked, 2);
        assert_eq!(timing.completed, 1);
        assert_eq!(timing.pending.len(), 1);
        assert_eq!(timing.pending[0].sequence, 2);
        assert_eq!(timing.pending[0].lists, 5);
        assert_eq!(timing.pending[0].queue, 0x7000);
        assert_eq!(timing.pending[0].receipt.fence_identity(), second_identity);
        assert_eq!(timing.pending[0].receipt.target_value(), 7);

        second_completed.store(7, Ordering::Release);
        timing.retire_completed();
        assert!(timing.pending.is_empty());
        assert_eq!(timing.completed, 2);
    }

    #[test]
    fn gpu_completion_timing_bounds_unobserved_receipts() {
        let (receipt, _) = super::super::alloc_pool::submission_receipt_for_test(0, 4);
        let mut timing = GpuCompletionTiming::default();
        timing.push_with_limit(receipt.clone(), 1, 0x8000, 2);
        timing.push_with_limit(receipt.clone(), 2, 0x8000, 2);
        timing.push_with_limit(receipt, 3, 0x8000, 2);

        assert_eq!(timing.tracked, 3);
        assert_eq!(timing.capacity_evictions, 1);
        assert_eq!(timing.pending.len(), 2);
        assert_eq!(timing.pending[0].sequence, 2);
        assert_eq!(timing.pending[1].sequence, 3);
    }

    #[test]
    fn gpu_completion_device_loss_is_not_counted_as_completion() {
        let (receipt, completed) = super::super::alloc_pool::submission_receipt_for_test(0, 4);
        let mut timing = GpuCompletionTiming::default();
        timing.push(receipt, 1, 0x8000);
        completed.store(u64::MAX, Ordering::Release);
        timing.retire_completed();
        assert!(timing.pending.is_empty());
        assert_eq!(timing.completed, 0);
        assert_eq!(timing.device_lost, 1);
    }

    #[test]
    fn gpu_completion_timing_counts_receipts_discarded_on_contention() {
        let timing = Mutex::new(GpuCompletionTiming::default());
        let contentions = AtomicU64::new(0);
        let (receipt, _) = super::super::alloc_pool::submission_receipt_for_test(0, 4);
        let held = timing.lock().unwrap();

        assert!(!try_note_gpu_completion_receipt(&timing, &contentions, receipt, 1, 0x9000));
        assert_eq!(contentions.load(Ordering::Relaxed), 1);
        assert_eq!(held.tracked, 0);
        assert!(held.pending.is_empty());
    }

    #[test]
    fn submission_warning_threshold_is_bounded_and_defaults() {
        assert!(gpu_timing_on_value(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(!gpu_timing_on_value(value));
        }
        assert_eq!(slow_submission_us_value(None), 50_000);
        assert_eq!(slow_submission_us_value(Some("bad")), 50_000);
        assert_eq!(slow_submission_us_value(Some("0")), 0);
        assert_eq!(slow_submission_us_value(Some("17")), 17_000);
        assert_eq!(slow_submission_us_value(Some("18446744073709551615")), u64::MAX);
    }
}

#[cfg(test)]
mod dred_tests {
    use super::*;

    #[test]
    fn a_late_partially_executed_list_is_reported_before_unstarted_lists() {
        let progress = [0u32, 1, 2];
        let mut nodes = vec![D3D12_AUTO_BREADCRUMB_NODE::default(); 606];
        for (i, node) in nodes.iter_mut().enumerate() {
            node.BreadcrumbCount = 2;
            node.pLastBreadcrumbValue = if i < 193 { &progress[0] } else { &progress[2] };
        }
        nodes[605].pLastBreadcrumbValue = &progress[1];
        for i in 1..nodes.len() {
            nodes[i - 1].pNext = &nodes[i];
        }
        let mut reported = Vec::new();
        unsafe { report_dred_lists(nodes.as_ptr(), |node, done| reported.push((std::ptr::from_ref(node), done))) };
        assert_eq!(reported.first(), Some(&(std::ptr::from_ref(&nodes[605]), 1)));
        assert_eq!(reported.len(), 64);
        for (i, &(node, done)) in reported[1..].iter().enumerate() {
            assert_eq!((node, done), (std::ptr::from_ref(&nodes[i]), 0));
        }
    }
}

/// If the device was removed: the reason, DRED's page fault (address, the allocations around
/// it) and the operations each command list got to – into the log, once.
pub fn log_device_removed() {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let raw = device_raw();
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    let Err(e) = (unsafe { dev.GetDeviceRemovedReason() }) else { return };
    if DONE.swap(true, Ordering::AcqRel) {
        return;
    }
    crate::xlog_error!("D3D12 device removed: {:#x} {}", e.code().0 as u32, e.message());
    // (to rule the system's memory in or out: a removal with no fault of the GPU's had 30 GB of
    // commit left)
    let (commit, limit) = crate::host::system_commit();
    if limit != 0 {
        crate::xlog_error!("system commit at the removal: {} of {} MB ({} MB left)", commit >> 20, limit >> 20, limit.saturating_sub(commit) >> 20);
    }
    unsafe {
        let Ok(dred) = dev.cast::<ID3D12DeviceRemovedExtendedData>() else { return };
        if let Ok(pf) = dred.GetPageFaultAllocationOutput() {
            let names = |mut n: *const D3D12_DRED_ALLOCATION_NODE| {
                let mut v = Vec::new();
                while !n.is_null() && v.len() < 8 {
                    let a = &*n;
                    let name = if a.ObjectNameA.is_null() { String::new() } else { std::ffi::CStr::from_ptr(a.ObjectNameA as *const std::ffi::c_char).to_string_lossy().into_owned() };
                    v.push(format!("type {} {}", a.AllocationType.0, name));
                    n = a.pNext;
                }
                v.join(", ")
            };
            let va = pf.PageFaultVA;
            let window = uma_window_vas().iter().position(|&w| va >= w && va < w + 0x1_0000_0000).map(|k| format!(" = window {k} + {:#x}", va - uma_window_vas()[k])).unwrap_or_default();
            crate::xlog_error!("DRED page fault at GPU VA {va:#x}{window}; allocations there: [{}]; recently freed: [{}]", names(pf.pHeadExistingAllocationNode), names(pf.pHeadRecentFreedAllocationNode));
        }
        if let Ok(bc) = dred.GetAutoBreadcrumbsOutput() {
            report_dred_lists(bc.pHeadAutoBreadcrumbNode, |node, done| {
                let count = node.BreadcrumbCount as usize;
                let ops: Vec<String> = (done.saturating_sub(3)..(done + 3).min(count)).map(|i| format!("{}{}", if i == done { ">" } else { "" }, (*node.pCommandHistory.add(i)).0)).collect();
                let list = node.pCommandList.as_ref().map_or(std::ptr::null_mut(), |l| l.as_raw());
                let history: Vec<i32> = if node.pCommandHistory.is_null() { Vec::new() } else { std::slice::from_raw_parts(node.pCommandHistory, count).iter().map(|o| o.0).collect() };
                let name = if node.pCommandListDebugNameW.is_null() { None } else { node.pCommandListDebugNameW.to_string().ok() };
                let key = super::dred::node_list(list as usize, name.as_deref());
                let mut bound = super::dred::bound_at(key, &history, done).map(|l| format!("; pipeline bound there: {l}")).unwrap_or_default();
                let after = super::dred::bound_after(key, &history, done);
                if !after.is_empty() {
                    bound += &format!("; bound after it: {}", after.iter().take(4).cloned().collect::<Vec<_>>().join(", "));
                }
                crate::xlog_error!("DRED: list {:p} on queue {:p}: {done} of {count} operations done, around: {}{bound}", list, node.pCommandQueue.as_ref().map_or(std::ptr::null_mut(), |q| q.as_raw()), ops.join(" "));
            });
        }
    }
}

/// The GPU hung or the device was removed: the title is stopped at once (a title that goes on
/// submitting work to a hung device can keep the driver from recovering), with the reason in
/// the log and the report.
pub fn stop_if_device_removed() {
    let raw = device_raw();
    let Some(dev) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    if let Err(e) = unsafe { dev.GetDeviceRemovedReason() } {
        log_device_removed();
        let msg = format!("the D3D12 device was removed ({:#x}: {}) – the title is stopped", e.code().0 as u32, e.message());
        crate::xlog_error!("{msg}");
        crate::report::write_final(Some(msg));
        crate::log::flush();
        unsafe {
            windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), e.code().0 as u32);
        }
    }
}

#[cfg(test)]
mod rect_list_pso_tests {
    use super::*;

    const REFUSED: HRESULT = 0x8007_0057u32 as i32;

    fn with_gs(gs: &[u8]) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC { GS: D3D12_SHADER_BYTECODE { pShaderBytecode: gs.as_ptr() as _, BytecodeLength: gs.len() }, ..Default::default() }
    }

    /// A pipeline the driver refuses with the rect list GS is made again without it, once; one
    /// it accepts keeps the GS, and a refused pipeline without a rect list GS is not retried.
    #[test]
    fn a_refused_rect_list_gs_is_retried_without_it() {
        let gs = [0u8; 4];
        let mut copy = with_gs(&gs);
        let mut calls = Vec::new();
        let hr = retry_without_rect_gs(&mut copy, true, |c| {
            calls.push(c.GS.pShaderBytecode.is_null());
            if c.GS.pShaderBytecode.is_null() {
                0
            } else {
                REFUSED
            }
        });
        assert_eq!((hr, calls), (0, vec![false, true]));
        let mut copy = with_gs(&gs);
        let mut n = 0;
        assert_eq!(
            retry_without_rect_gs(&mut copy, true, |_| {
                n += 1;
                0
            }),
            0
        );
        assert!(n == 1 && !copy.GS.pShaderBytecode.is_null());
        let mut n = 0;
        assert_eq!(
            retry_without_rect_gs(&mut copy, false, |_| {
                n += 1;
                REFUSED
            }),
            REFUSED
        );
        assert!(n == 1 && !copy.GS.pShaderBytecode.is_null());

        for rejected in [E_OUTOFMEMORY, 0x887A_0005u32 as i32, 0x8000_4005u32 as i32] {
            let mut copy = with_gs(&gs);
            let mut n = 0;
            assert_eq!(retry_without_rect_gs(&mut copy, true, |_| { n += 1; rejected }), rejected);
            assert_eq!(n, 1, "{rejected:#x} must not be treated as a rect-GS rejection");
        }
        let mut copy = with_gs(&gs);
        let mut n = 0;
        assert_eq!(retry_without_rect_gs(&mut copy, true, |_| { n += 1; if n == 1 { E_INVALIDARG } else { E_OUTOFMEMORY } }), E_OUTOFMEMORY);
        assert_eq!(n, 2, "the retry's HRESULT is returned unchanged");
    }

    /// `has_rect_list_gs`: a pipeline made with the rect list GS, not one made again without it
    /// (or never given it), and not once its address is forgotten.
    #[test]
    fn pipelines_with_the_rect_list_gs_are_known_until_forgotten() {
        let gs = [0u8; 4];
        let (kept, dropped, none) = (0x7E57_1000usize, 0x7E57_2000usize, 0x7E57_3000usize);
        let mut copy = with_gs(&gs);
        note_rect_list_gs(kept, &copy, true);
        assert!(retry_without_rect_gs(&mut copy, true, |c| if c.GS.pShaderBytecode.is_null() { 0 } else { REFUSED }) == 0);
        note_rect_list_gs(dropped, &copy, true);
        note_rect_list_gs(none, &with_gs(&gs), false);
        assert!(has_rect_list_gs(kept) && !has_rect_list_gs(dropped) && !has_rect_list_gs(none) && !has_rect_list_gs(0));
        forget_pso(kept);
        assert!(!has_rect_list_gs(kept));
    }

    /// An untranslated rect list pipeline from a stream gets the PC's topology type (TRIANGLE),
    /// as at 956a476; other types, and a direct CreateGraphicsPipelineState, are left as they are.
    #[test]
    fn untranslated_stream_rect_lists_get_the_pcs_topology_type() {
        let at = |t: i32| D3D12_GRAPHICS_PIPELINE_STATE_DESC { PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE(t), ..Default::default() };
        let desc = |g: &D3D12_GRAPHICS_PIPELINE_STATE_DESC| g as *const _ as *const c_void;
        let (rect, tri) = (at(RECT_LIST_TYPE), at(3));
        unsafe {
            let g = untranslated_stream_desc(desc(&rect), true).expect("a stream's rect list");
            assert_eq!(g.PrimitiveTopologyType, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
            assert!(untranslated_stream_desc(desc(&tri), true).is_none());
            assert!(untranslated_stream_desc(desc(&rect), false).is_none());
        }
    }
}

#[cfg(test)]
mod fallback_stream_tests {
    use super::*;

    #[test]
    fn a_stream_counts_referenced_shader_code_not_its_descriptors() {
        let mut stream = StreamOut::default();
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &[0; 11]);
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &[0; 7]);
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
        assert_eq!(stream.shader_bytes(), 18);
        assert_ne!(stream.0.len(), stream.shader_bytes(), "the stream stores bytecode descriptors, not code");
    }

    /// A stream whose SizeInBytes ends inside its last subobject's payload (a VS subobject with
    /// 8 of its 16 bytecode bytes) is not rewritten: None, not a range panic in the device hook
    /// (which aborts the title). The whole subobject gets the stand-in VS.
    #[test]
    fn a_stream_ending_inside_a_payload_is_not_rewritten() {
        let vs = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS.0 as u64;
        let stream = [vs, 0x1234, 0x10];
        unsafe {
            let cut: [usize; 2] = [16, stream.as_ptr() as usize];
            assert!(fallback_stream(cut.as_ptr() as *const u8).is_none());
            let whole: [usize; 2] = [24, stream.as_ptr() as usize];
            let (buf, size) = fallback_stream(whole.as_ptr() as *const u8).expect("a whole VS subobject");
            assert_eq!((buf.len(), size), (24, [24, 0]));
            assert_eq!(u64::from_le_bytes(buf[16..24].try_into().unwrap()), dummy().vs.len() as u64);
        }
    }
}

#[cfg(test)]
mod this_first_tests {
    use super::*;

    /// `this_first`: either argument order, and a return buffer whose first word is a stale copy
    /// of the object's vtable pointer (7a033c8: the retail game's C-way call wrote its result
    /// over the device). Synthetic objects; no GPU and no title.
    #[test]
    fn a_return_buffer_with_a_stale_vtable_copy_is_not_the_object() {
        let pc = [0usize; 32];
        let ext = [pc.as_ptr() as usize, 0];
        let vtbl = ext.as_ptr() as usize + 8;
        EXTS.lock().unwrap().get_or_insert_with(HashSet::new).insert(vtbl);
        let plain = 0x1234_5678usize;
        let p = &plain as *const usize as *mut c_void;
        // an object off the stack (as a real device is), and a stack buffer holding a stale copy
        // of its vtable pointer - the retail game's case
        let obj = Box::into_raw(Box::new([vtbl])) as *mut c_void;
        let buf = [vtbl];
        let b = buf.as_ptr() as *mut c_void;
        unsafe {
            assert!(is_adopted(obj) && is_adopted(b), "both look adopted");
            // one adopted: that one is the object, either position
            assert!(this_first(obj, p));
            assert!(!this_first(p, obj));
            // neither adopted: unchanged (the C way is assumed)
            assert!(!this_first(p, p) && !this_first(std::ptr::null_mut(), std::ptr::null_mut()));
            // one null argument: the other is taken as `this`, either position
            assert!(this_first(obj, std::ptr::null_mut()));
            assert!(!this_first(std::ptr::null_mut(), obj));
            // both adopted, one on this thread's stack: the stack one is the buffer, so the heap
            // one is `this` in either order - what 7a033c8 fixed
            assert!(this_first(obj, b));
            assert!(!this_first(b, obj));
            // both on the stack, neither the device: `a` (the C++ way)
            let stale = [vtbl];
            let s = stale.as_ptr() as *mut c_void;
            assert!(this_first(b, s) && this_first(s, b));
            drop(Box::from_raw(obj as *mut [usize; 1]));
        }
        if let Some(g) = EXTS.lock().unwrap().as_mut() {
            g.remove(&vtbl);
        }
    }
}

#[cfg(test)]
mod cache_report_tests {
    use super::*;

    #[test]
    fn reports_without_dxil_cache_statistics_still_deserialize() {
        let mut old = serde_json::to_value(D3d12Stats::default()).unwrap();
        old.as_object_mut().unwrap().remove("dxil_cache");
        let decoded: D3d12Stats = serde_json::from_value(old).unwrap();
        assert_eq!(
            decoded.dxil_cache,
            crate::gpu::recomp::dxc::CacheStats::default()
        );
    }

    #[test]
    fn pipeline_pressure_statistics_have_a_backward_compatible_shape() {
        let mut stats = D3d12Stats::default();
        stats.pipeline_pressure = super::super::pso_async::PipelinePressureStats {
            publications_detached: 1,
            policy_owner_retirements: 2,
            generation_owners_released: 3,
            restoration_attempts: 4,
            restoration_succeeded: 5,
            restoration_failed: 6,
        };
        let mut json = serde_json::to_value(stats).unwrap();
        assert_eq!(json["pipeline_pressure"]["publications_detached"], 1);
        assert_eq!(json["pipeline_pressure"]["policy_owner_retirements"], 2);
        assert_eq!(json["pipeline_pressure"]["generation_owners_released"], 3);
        assert_eq!(json["pipeline_pressure"]["restoration_attempts"], 4);
        assert_eq!(json["pipeline_pressure"]["restoration_succeeded"], 5);
        assert_eq!(json["pipeline_pressure"]["restoration_failed"], 6);

        json.as_object_mut().unwrap().remove("pipeline_pressure");
        let decoded: D3d12Stats = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.pipeline_pressure, Default::default());
    }

    #[test]
    fn reports_without_submission_distribution_still_deserialize() {
        let mut old = serde_json::to_value(D3d12Stats::default()).unwrap();
        let old = old.as_object_mut().unwrap();
        for field in [
            "submission_total_ns", "submission_mean_ms", "submission_p95_upper_ms",
            "submission_p99_upper_ms", "submission_histogram_log2_ns",
        ] {
            old.remove(field);
        }
        let decoded: D3d12Stats = serde_json::from_value(serde_json::Value::Object(old.clone())).unwrap();
        assert_eq!(decoded.submission_total_ns, 0);
        assert!(decoded.submission_histogram_log2_ns.is_empty());
    }
}

#[cfg(test)]
mod readback_routing_tests {
    use super::*;

    #[test]
    fn custom_write_back_only_stays_coherent_for_small_copy_dest_buffers() {
        let rb = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let custom_wb = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let copy_dest = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let gen_read = D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32;
        let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let none = D3D12_RESOURCE_FLAG_NONE;

        assert!(is_coherent_readback_buffer(&rb, copy_dest, none, 4 << 20));
        assert!(!is_coherent_readback_buffer(&rb, copy_dest, none, COHERENT_READBACK_MAX + 0x1_0000));
        assert!(is_cpu_readback_buffer(&rb, copy_dest, none));

        // General CUSTOM+WRITE_BACK buffers route to sparse VRAM, not coherent Window 3.
        assert!(!is_coherent_readback_buffer(&custom_wb, gen_read, none, 4 << 20));
        assert!(!is_cpu_readback_buffer(&custom_wb, gen_read, none));
        assert!(!is_coherent_readback_buffer(&custom_wb, copy_dest, uav, 4 << 20));
        assert!(!is_cpu_readback_buffer(&custom_wb, copy_dest, uav));

        // Small non-UAV COPY_DEST CUSTOM+WRITE_BACK buffers may use coherent Window 3; larger ones spill.
        assert!(is_coherent_readback_buffer(&custom_wb, copy_dest, none, 4 << 20));
        assert!(!is_coherent_readback_buffer(&custom_wb, copy_dest, none, COHERENT_CUSTOM_WB_MAX + 0x1_0000));
        assert!(is_cpu_readback_buffer(&custom_wb, copy_dest, none));
    }

    /// Bug #4: `place_console_buffer` coherent fallback must strip console-only resource flags
    /// and state bits before creating the cross-adapter placed resource.
    #[test]
    fn coherent_buffer_sanitizes_console_flags_and_state() {
        let raw_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0x10_0000,
            Width: 0x1_0000,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT(0x101), // console layout
            Flags: D3D12_RESOURCE_FLAGS(0x0130_0000 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0),
            ..Default::default()
        };
        let console_state = 0x0010_0000usize | (D3D12_RESOURCE_STATE_COPY_DEST.0 as usize);

        let (clean, st) = sanitize_coherent_buffer(&raw_desc, console_state, true);
        assert_eq!(clean.Alignment, 0);
        assert_eq!(clean.Layout, D3D12_TEXTURE_LAYOUT_ROW_MAJOR);
        assert_eq!(
            clean.Flags,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER
        );
        assert_eq!(st, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);

        // Verify against a real D3D12 device: raw console flags/state are rejected, sanitized succeeds.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            if let Ok(dev3) = dev.cast::<ID3D12Device3>() {
                use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
                let mem = VirtualAlloc(std::ptr::null(), 0x1_0000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                assert!(!mem.is_null());
                if let Ok(heap) = dev3.OpenExistingHeapFromAddress::<ID3D12Heap>(mem) {
                    let mut unsanitized = raw_desc;
                    unsanitized.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
                    unsanitized.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                    unsanitized.Alignment = 0;
                    let mut r_bad: Option<ID3D12Resource> = None;
                    assert!(
                        dev.CreatePlacedResource(&heap, 0, &unsanitized, D3D12_RESOURCE_STATES(console_state as i32), None, &mut r_bad).is_err(),
                        "PC D3D12 must reject unsanitized console flags/state"
                    );
                    let mut r_ok: Option<ID3D12Resource> = None;
                    dev.CreatePlacedResource(&heap, 0, &clean, D3D12_RESOURCE_STATES(st as i32), None, &mut r_ok)
                        .expect("sanitized coherent buffer must succeed on PC D3D12");
                    assert!(r_ok.is_some());
                    drop(r_ok);
                    drop(heap);
                }
                // D3D12CreateDevice gives the process's one device: once another test adopted it,
                // r_ok was adopted too, and its last Release is held (`late_release`) while a
                // queue of another test still has work. The heap pins `mem` until then (#1487).
                let start = std::time::Instant::now();
                let mut freed = VirtualFree(mem, 0, MEM_RELEASE);
                while freed == 0 && start.elapsed() < std::time::Duration::from_secs(10) {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    freed = VirtualFree(mem, 0, MEM_RELEASE);
                }
                assert_ne!(freed, 0, "the heap's pages were not let go within 10 s");
            }
        }
    }

    /// Bug #5: `CopyDescriptors` and `CopyDescriptorsSimple` must remap `XBOX_EXTRA_CBV_SRV_UAV`
    /// (`4`) to `D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV` (`0`) on both console and PC paths.
    #[test]
    fn copy_descriptors_remaps_xbox_extra_cbv_srv_uav_to_pc_enum() {
        use std::cell::RefCell;

        assert_eq!(pc_heap_type(XBOX_EXTRA_CBV_SRV_UAV), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
        assert_eq!(pc_heap_type(0), 0);
        assert_eq!(pc_heap_type(1), 1);
        assert_eq!(pc_heap_type(2), 2);
        assert_eq!(pc_heap_type(3), 3);

        thread_local! {
            static SIMPLE_TYPES: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
            static RANGE_TYPES: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
        }

        unsafe extern "system" fn fake_copy_simple(_this: *mut c_void, _n: u32, _dst: u64, _src: u64, ty: u32) {
            SIMPLE_TYPES.with(|v| v.borrow_mut().push(ty));
        }
        #[allow(clippy::too_many_arguments)]
        unsafe extern "system" fn fake_copy_ranges(
            _this: *mut c_void,
            _nd: u32,
            _dsts: *const u64,
            _dsizes: *const u32,
            _ns: u32,
            _srcs: *const u64,
            _ssizes: *const u32,
            ty: u32,
        ) {
            RANGE_TYPES.with(|v| v.borrow_mut().push(ty));
        }

        #[repr(C)]
        struct FakeDev {
            vt: *const usize,
        }

        let mut orig_vt = [0usize; 32];
        orig_vt[23] = fake_copy_ranges as usize;
        orig_vt[24] = fake_copy_simple as usize;
        let ext_vt = [orig_vt.as_ptr() as usize, 0usize];
        let mut dev = FakeDev {
            // SAFETY: `ext_vt` has 2 entries; `.add(1)` puts `orig_vt` at `ext[-1]`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
        };
        let dev_ptr = &mut dev as *mut FakeDev as *mut c_void;

        SIMPLE_TYPES.with(|v| v.borrow_mut().clear());
        RANGE_TYPES.with(|v| v.borrow_mut().clear());

        unsafe {
            // Non-console PC fallback paths
            dev_copy_descriptors_simple_x(dev_ptr, 1, 0x1000, 0x2000, XBOX_EXTRA_CBV_SRV_UAV);
            dev_copy_descriptors_simple_x(dev_ptr, 1, 0x1000, 0x2000, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32);
            let (d, s, sz) = ([0x1000u64], [0x2000u64], [1u32]);
            dev_copy_descriptors_x(dev_ptr, 1, d.as_ptr(), sz.as_ptr(), 1, s.as_ptr(), sz.as_ptr(), XBOX_EXTRA_CBV_SRV_UAV);
            dev_copy_descriptors_x(dev_ptr, 1, d.as_ptr(), sz.as_ptr(), 1, s.as_ptr(), sz.as_ptr(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV.0 as u32);

            // Console-slot paths (`copy_console_slots` invoked from both Simple and Ranges)
            let heap_obj = 0x5555_DEADusize;
            if crate::gpu::uma::get().is_some()
                && super::super::console_heap::register(heap_obj, 4, 0x10_0000, 0, 32, false, false, false)
            {
                let start = super::super::console_heap::start(heap_obj).unwrap();
                let dst_h = start + super::super::console_heap::SLOT as u64;
                dev_copy_descriptors_simple_x(dev_ptr, 1, dst_h, start, XBOX_EXTRA_CBV_SRV_UAV);
                let (cd, cs, csz) = ([dst_h], [start], [1u32]);
                dev_copy_descriptors_x(dev_ptr, 1, cd.as_ptr(), csz.as_ptr(), 1, cs.as_ptr(), csz.as_ptr(), XBOX_EXTRA_CBV_SRV_UAV);
                super::super::console_heap::unregister(heap_obj);

                SIMPLE_TYPES.with(|v| assert_eq!(*v.borrow(), vec![0, 1, 0, 0]));
            } else {
                SIMPLE_TYPES.with(|v| assert_eq!(*v.borrow(), vec![0, 1]));
            }
            RANGE_TYPES.with(|v| assert_eq!(*v.borrow(), vec![0, 2]));
        }
    }

    /// Console #53 / #54 ClearUnorderedAccessViewUint / Float translate their console handles
    /// (not the PC #49 / #50 the remap alone gives).
    #[test]
    fn console_clear_uav_slots_translate_handles() {
        use super::super::list_x as l;
        let slots = console_slots(Kind::CommandList);
        assert!(slots.contains(&(53, l::clear_uav_uint_x as usize)));
        assert!(slots.contains(&(54, l::clear_uav_float_x as usize)));
    }

    /// Bug #6: `Kind::DmaList` must install `cl_release` so releasing a DMA command list removes
    /// its entry from `list_x::LISTS`.
    #[test]
    fn dma_list_release_cleans_up_list_state() {
        assert!(
            overrides(Kind::DmaList).iter().any(|&(name, f)| name == "Release" && f == cl_release as usize),
            "Kind::DmaList must override Release with cl_release"
        );

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).expect("copy alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &alloc, None).expect("copy list");
            let raw = list.into_raw();
            let key = raw as usize;

            // Mirror `device_x::create_list`: adopt as CommandList, track via `list_x::on_create`,
            // then re-adopt as `Kind::DmaList`.
            adopt(raw, Kind::CommandList);
            super::super::list_x::on_create(raw, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COPY);
            readopt(raw, Kind::DmaList);
            assert!(super::super::list_x::has_list_state(key), "DMA list must be tracked in list_x::LISTS");

            // Invoke Release through the adopted vtable slot 2.
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt(raw).add(2));
            assert_eq!(release(raw), 0);
            assert!(!super::super::list_x::has_list_state(key), "releasing DMA list must remove its list_x::LISTS entry");
        }
    }
}

#[cfg(test)]
mod queue_tests {
    use super::*;
    use std::mem::{offset_of, size_of};
    use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue_Vtbl, ID3D12Fence_Vtbl};

    // a fake adopted queue: the extension vtable's word before slot 0 points at the PC table
    unsafe extern "system" fn get_desc(_: *mut c_void, out: usize) -> usize {
        let d = [1u32, 2, 3, 4];
        std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, out as *mut u8, 16);
        out
    }
    unsafe extern "system" fn signal(_: *mut c_void, _: usize, _: usize) -> i32 {
        0
    }
    unsafe extern "system" fn fake_fence_completed(_: *mut c_void) -> u64 {
        u64::MAX
    }

    #[repr(C)]
    struct FakeFence {
        vtable: *const usize,
    }

    /// Removes the synthetic fence and queue entries from every global table that
    /// `queue_dispatch(14, ...)` touches before the stack objects go out of scope.
    struct Cleanup {
        fence: usize,
        queue: usize,
    }

    impl Drop for Cleanup {
        fn drop(&mut self) {
            FENCES.write().unwrap_or_else(|e| e.into_inner()).retain(|&f| f != self.fence);
            super::super::fence_x::forget_fence_for_test(self.fence);
            super::super::list_x::forget_title_fence_for_test(self.fence, self.queue);
            QUEUE_OLDER.store(false, Ordering::Release);
        }
    }

    /// Queue #18 before the layout is known writes 8 bytes; after a Signal at #14 (the older
    /// layout) it is GetDesc with all 16.
    #[test]
    fn queue_slot_18_before_the_layout_is_known_writes_8_bytes() {
        const SLOT: usize = size_of::<usize>();
        let mut pc = [0usize; 32];
        pc[14] = signal as usize;
        pc[18] = get_desc as usize;
        let mut ext = [0usize; 1 + size_of::<ID3D12CommandQueue_Vtbl>() / SLOT];
        ext[0] = pc.as_ptr() as usize;
        ext[1 + offset_of!(ID3D12CommandQueue_Vtbl, GetDesc) / SLOT] = get_desc as usize;
        let vtbl = unsafe { ext.as_ptr().add(1) } as usize;
        let mut obj = [vtbl];
        let this = obj.as_mut_ptr() as *mut c_void;
        // A valid minimal ID3D12Fence COM object while registered: if a parallel test inspects
        // `SYNC_GRAPH.title_fences` before `Cleanup::drop` runs, `GetCompletedValue` succeeds
        // cleanly instead of dereferencing an unmapped sentinel address.
        let mut fence_vtbl = [0usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        fence_vtbl[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = fake_fence_completed as usize;
        let fake_fence = FakeFence { vtable: fence_vtbl.as_ptr() };
        let fence = (&fake_fence as *const FakeFence) as usize;
        let _cleanup = Cleanup { fence, queue: this as usize };
        QUEUE_SHIFTED.store(false, Ordering::Release);
        QUEUE_OLDER.store(false, Ordering::Release);
        unsafe {
            let mut out = [0xAAAA_AAAAu32; 4];
            let r = queue_dispatch(18, this, out.as_mut_ptr() as usize, 0, 0, 0);
            assert_eq!(r, out.as_ptr() as usize);
            assert_eq!(out, [1, 2, 0xAAAA_AAAA, 0xAAAA_AAAA], "only the first 8 bytes");
            FENCES.write().unwrap().push(fence);
            queue_dispatch(14, this, fence, 1, 0, 0);
            assert!(QUEUE_OLDER.load(Ordering::Acquire) && !QUEUE_SHIFTED.load(Ordering::Acquire));
            let mut out = [0xAAAA_AAAAu32; 4];
            queue_dispatch(18, this, out.as_mut_ptr() as usize, 0, 0, 0);
            assert_eq!(out, [1, 2, 3, 4], "older layout: the whole description");
        }
    }

    #[test]
    fn newer_layout_maps_dispatch_mesh_to_slot_129_not_128() {
        assert!(NEWER_LIST_SLOTS.contains(&(129, 127)), "DispatchMesh must map to newer slot #129");
        assert!(!NEWER_LIST_SLOTS.iter().any(|&(newer, _)| newer == 128), "newer slot #128 must not map to DispatchMesh");
    }

    #[test]
    fn own_work_execute_command_lists_bypasses_note_execute_and_drain() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLED: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "system" fn fake_qi(_this: *mut c_void, _iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
            if !out.is_null() {
                *out = std::ptr::null_mut();
            }
            -2147467262 // E_NOINTERFACE
        }
        unsafe extern "system" fn fake_ref(_this: *mut c_void) -> u32 { 1 }
        unsafe extern "system" fn fake_noop(_this: *mut c_void) {}
        unsafe extern "system" fn fake_execute(_this: *mut c_void, n: u32, _lists: *const *mut c_void) {
            CALLED.fetch_add(n as usize, Ordering::SeqCst);
        }

        let iface = layout::find("ID3D12CommandQueue").expect("ID3D12CommandQueue layout");
        let exec_slot = layout::slot(iface, "ExecuteCommandLists").expect("ExecuteCommandLists slot");
        let n = iface.methods.len();
        let mut fake_vtbl = vec![fake_noop as usize; n + 1];
        fake_vtbl[0] = fake_qi as usize;
        fake_vtbl[1] = fake_ref as usize;
        fake_vtbl[2] = fake_ref as usize;
        fake_vtbl[exec_slot] = fake_execute as usize;
        fake_vtbl[n] = 0;
        let orig = fake_vtbl.as_ptr() as usize;

        struct ClassCleanup(usize);
        impl Drop for ClassCleanup {
            fn drop(&mut self) {
                let mut exts = Vec::new();
                CLASSES.lock().unwrap().retain(|c| {
                    if c.orig == self.0 {
                        exts.push(c.ext);
                        false
                    } else {
                        true
                    }
                });
                if let Some(set) = EXTS.lock().unwrap().as_mut() {
                    for e in exts {
                        set.remove(&e);
                    }
                }
            }
        }
        let _cleanup = ClassCleanup(orig);

        let mut obj = orig;
        let obj_ptr = &mut obj as *mut usize as *mut c_void;
        struct LeaseCleanup(usize);
        impl Drop for LeaseCleanup {
            fn drop(&mut self) {
                crate::hle::d3d12x::queue_submit::release(self.0, || 0);
            }
        }
        let _lease_cleanup = LeaseCleanup(obj_ptr as usize);
        let ext = unsafe { class_for(obj_ptr, orig, Kind::CommandQueue) };
        unsafe { *(obj_ptr as *mut usize) = ext; }

        let mut dummy_list = std::ptr::null_mut::<c_void>();
        let activity_before = crate::hle::d3d12x::present::activity();
        let before_called = CALLED.load(Ordering::SeqCst);
        unsafe {
            crate::hle::d3d12x::title_textures::own_work(|| {
                q_execute_command_lists(obj_ptr, 1, &mut dummy_list as *mut *mut c_void);
            });
        }
        assert_eq!(CALLED.load(Ordering::SeqCst), before_called + 1, "original ExecuteCommandLists must still run");
        assert_eq!(crate::hle::d3d12x::present::activity(), activity_before, "own_work submit must bypass present::note_execute");
    }
}

#[cfg(test)]
mod srv_tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC1_UNORM_SRGB;

    #[test]
    fn a_cube_view_becomes_the_cube_array_of_one_cube() {
        let cube = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_BC1_UNORM_SRGB,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
            Shader4ComponentMapping: 0x1688,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV { MostDetailedMip: 2, MipLevels: 3, ResourceMinLODClamp: 1.5 },
            },
        };
        let v = cube_as_cube_array(&cube).expect("a cube view is rewritten");
        assert_eq!(v.ViewDimension, D3D12_SRV_DIMENSION_TEXTURECUBEARRAY);
        assert_eq!((v.Format, v.Shader4ComponentMapping), (DXGI_FORMAT_BC1_UNORM_SRGB, 0x1688), "format and mapping kept");
        let a = unsafe { v.Anonymous.TextureCubeArray };
        assert_eq!((a.MostDetailedMip, a.MipLevels, a.First2DArrayFace, a.NumCubes), (2, 3, 0, 1));
        assert_eq!(a.ResourceMinLODClamp, 1.5);

        // a cube array and every other view are forwarded as the title made them
        let mut array = cube;
        array.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        array.Anonymous.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV { NumCubes: 2, ..Default::default() };
        assert!(cube_as_cube_array(&array).is_none());
        let mut d2 = cube;
        d2.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        assert!(cube_as_cube_array(&d2).is_none());
    }

    #[test]
    fn a_default_view_of_a_depth_buffer_matches_its_sample_count_and_array_size() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_R24G8_TYPELESS, DXGI_SAMPLE_DESC};
        for (samples, slices, dimension, dim) in [
            (1, 1, D3D12_SRV_DIMENSION_TEXTURE2D, 4),
            (1, 3, D3D12_SRV_DIMENSION_TEXTURE2DARRAY, 5),
            (4, 1, D3D12_SRV_DIMENSION_TEXTURE2DMS, 6),
            (4, 3, D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY, 7),
        ] {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                DepthOrArraySize: slices,
                MipLevels: 1,
                Format: DXGI_FORMAT_R24G8_TYPELESS,
                SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 },
                ..Default::default()
            };
            let v = pc_srv(&desc, None).expect("a typeless depth buffer gets a view of its own");
            assert_eq!(v.Format, DXGI_FORMAT_R24_UNORM_X8_TYPELESS);
            assert_eq!(v.ViewDimension, dimension, "{samples} samples, {slices} slices");
            let a = &v.Anonymous;
            match dimension {
                D3D12_SRV_DIMENSION_TEXTURE2D => assert_eq!(unsafe { a.Texture2D.MipLevels }, u32::MAX),
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                    assert_eq!(unsafe { (a.Texture2DArray.FirstArraySlice, a.Texture2DArray.ArraySize, a.Texture2DArray.MipLevels) }, (0, u32::from(slices), u32::MAX))
                }
                D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => {
                    assert_eq!(unsafe { (a.Texture2DMSArray.FirstArraySlice, a.Texture2DMSArray.ArraySize) }, (0, u32::from(slices)))
                }
                _ => {}
            }
            // the console view (T#) of the resource without a description
            assert_eq!(resource_dim_to_view(&desc), dim, "{samples} samples, {slices} slices");
            assert_eq!(v.ViewDimension.0, dim);
        }
    }
}

#[cfg(test)]
mod console_format_query_tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    #[test]
    fn console_formats_are_asked_as_the_pcs() {
        for (console, pc) in [(116u32, 10u32), (117, 10), (118, 45), (119, 46), (120, 47)] {
            let mut d = [0u32; 3];
            d[0] = console;
            let back = unsafe { console_format_query(D3D12_FEATURE_FORMAT_SUPPORT.0, d.as_mut_ptr() as *mut u8, 12) };
            assert_eq!((back, d[0]), (Some(console), pc));
        }
        // a PC format, or a feature without one: untouched
        let mut d = [45u32, 0, 0];
        assert_eq!(unsafe { console_format_query(D3D12_FEATURE_FORMAT_SUPPORT.0, d.as_mut_ptr() as *mut u8, 12) }, None);
        let mut d = [118u32; 16];
        assert_eq!(unsafe { console_format_query(D3D12_FEATURE_D3D12_OPTIONS.0, d.as_mut_ptr() as *mut u8, 64) }, None);
        assert_eq!(d[0], 118);
        assert_eq!(unsafe { console_format_query(D3D12_FEATURE_FORMAT_INFO.0, std::ptr::null_mut(), 8) }, None);
    }

    /// A hardware device (None: no D3D12 GPU here – skipped).
    fn device() -> Option<ID3D12Device> {
        let mut dev: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev) }.ok()?;
        dev
    }

    /// CheckFeatureSupport of the PC device as `dev_check_feature_support` reaches it (vtable #13).
    fn pc_check(dev: &ID3D12Device) -> CheckFeatureSupportFn {
        unsafe { std::mem::transmute(*(*(dev.as_raw() as *const *const usize)).add(13)) }
    }

    /// GPU: the queries a title makes of D16_UNORM_S8_UINT (118) and its family succeed with the
    /// PC format's answers, and the caller's format field is the console's again. Asked as is, the
    /// PC has no such format.
    #[test]
    fn gpu_console_depth_formats_are_supported() {
        let Some(dev) = device() else { return };
        let (this, f) = (dev.as_raw(), pc_check(&dev));
        // the gap: the PC's own answer (unless another test adopted a device, and the vtable it
        // shares with this one already holds `dev_check_feature_support`)
        if f as usize != dev_check_feature_support as usize {
            let mut raw = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(118), ..Default::default() };
            let hr = unsafe { f(this, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut raw as *mut _ as *mut u8, size_of_val(&raw) as u32) };
            assert!(hr < 0 || raw.Support1.0 == 0, "the PC knows format 118 ({hr:#x}, {:#x})", raw.Support1.0);
        }

        for (format, need) in [
            (118, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL),
            (119, D3D12_FORMAT_SUPPORT1_SHADER_LOAD),
            (120, D3D12_FORMAT_SUPPORT1_SHADER_LOAD),
            (116, D3D12_FORMAT_SUPPORT1_RENDER_TARGET),
            (117, D3D12_FORMAT_SUPPORT1_RENDER_TARGET),
        ] {
            let mut s = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(format), ..Default::default() };
            let hr = unsafe { check_feature_support_x(f, this, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut s as *mut _ as *mut u8, size_of_val(&s) as u32) };
            assert!(hr >= 0, "FORMAT_SUPPORT {format}: {hr:#x}");
            assert_eq!(s.Format.0, format as i32, "the caller's format");
            assert!(s.Support1.0 & need.0 != 0, "FORMAT_SUPPORT {format}: {:#x}", s.Support1.0);
        }

        let mut ms = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: DXGI_FORMAT(118), SampleCount: 4, ..Default::default() };
        let hr = unsafe { check_feature_support_x(f, this, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS.0, &mut ms as *mut _ as *mut u8, size_of_val(&ms) as u32) };
        assert!(hr >= 0 && ms.NumQualityLevels >= 1, "MULTISAMPLE_QUALITY_LEVELS 118 ×4: {hr:#x} {}", ms.NumQualityLevels);
        assert_eq!((ms.Format.0, ms.SampleCount), (118, 4));

        let mut info = D3D12_FEATURE_DATA_FORMAT_INFO { Format: DXGI_FORMAT(118), PlaneCount: 0 };
        let hr = unsafe { check_feature_support_x(f, this, D3D12_FEATURE_FORMAT_INFO.0, &mut info as *mut _ as *mut u8, size_of_val(&info) as u32) };
        assert!(hr >= 0, "FORMAT_INFO 118: {hr:#x}");
        assert_eq!((info.Format.0, info.PlaneCount), (118, 2));

        // a PC format is asked as it is
        let mut s = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ..Default::default() };
        assert!(unsafe { check_feature_support_x(f, this, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut s as *mut _ as *mut u8, size_of_val(&s) as u32) } >= 0);
        assert_eq!(s.Format, DXGI_FORMAT_R8G8B8A8_UNORM);
    }

    /// GPU: allocation info of the console's 116 – 120 textures is a real size – the console's
    /// (XG), and the PC's for the resource made of the desc (`pc_resource_desc`), both sane.
    #[test]
    fn gpu_console_format_allocation_info_is_sane() {
        let Some(dev) = device() else { return };
        // (bytes a pixel at least, as XG lays them out: 119 alone is a plain 2-byte format, 120 alone
        // the stencil plane)
        for (format, flags, bytes) in [(118u32, 2u32, 3u64), (118, 0, 3), (119, 0, 2), (120, 0, 1), (116, 1, 4), (117, 1, 4)] {
            let mut x = [0u8; 0x38];
            x[0..4].copy_from_slice(&3u32.to_le_bytes());
            x[0x10..0x18].copy_from_slice(&1920u64.to_le_bytes());
            x[0x18..0x1C].copy_from_slice(&1080u32.to_le_bytes());
            x[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            x[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            x[0x20..0x24].copy_from_slice(&format.to_le_bytes());
            x[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            x[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
            let t = crate::hle::xg::Texture::new(&x).unwrap_or_else(|hr| panic!("XG {format}: {hr:#x}"));
            let (size, align) = (t.size(), t.alignment());
            assert!(size >= 1920 * 1080 * bytes && size < 1 << 30 && align.is_power_of_two(), "XG {format}: {size:#x} {align:#x}");
            let d: D3D12_RESOURCE_DESC = unsafe { std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&x).as_ptr() as *const _) };
            let pc = unsafe { dev.GetResourceAllocationInfo(0, &[d]) };
            assert!(pc.SizeInBytes != u64::MAX && pc.SizeInBytes >= 1920 * 1080 * 4 && pc.Alignment > 0, "PC {format} as {}: {:#x}", d.Format.0, pc.SizeInBytes);
        }
    }

    /// GPU: CreatePlacedResource (#29) of a D16_UNORM_S8_UINT (118) depth target with an
    /// optimized clear value of that format: the desc becomes the PC's (`pc_resource_desc`), and
    /// the clear value with it (`pc_clear`) – one still in 118 the PC refuses.
    #[test]
    fn gpu_placed_console_depth_target_with_clear_value() {
        let Some(dev) = device() else {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        unsafe {
            adopt(dev.as_raw(), Kind::Device);
            let heap_desc = D3D12_HEAP_DESC {
                SizeInBytes: 16 << 20,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            dev.CreateHeap(&heap_desc, &mut heap).expect("RT/DS heap");
            let heap = heap.unwrap();
            let mut x = [0u8; 0x38];
            x[0..4].copy_from_slice(&3u32.to_le_bytes());
            x[0x10..0x18].copy_from_slice(&256u64.to_le_bytes());
            x[0x18..0x1C].copy_from_slice(&256u32.to_le_bytes());
            x[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            x[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            x[0x20..0x24].copy_from_slice(&118u32.to_le_bytes());
            x[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            x[0x30..0x34].copy_from_slice(&(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 as u32).to_le_bytes());
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT(118), Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const GUID;
            let state = D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as usize;
            let mut p = std::ptr::null_mut();
            let hr = dev_create_placed_resource_x(dev.as_raw(), heap.as_raw() as usize, 0, x.as_ptr() as usize, state, &clear as *const _ as usize, riid, &mut p);
            assert!(hr >= 0 && !p.is_null(), "D16S8 target with a 118 clear value: {hr:#x}");
            let r = ID3D12Resource::from_raw(p);
            assert_eq!(r.GetDesc().Format.0, 44, "made R24G8_TYPELESS");
        }
    }
}

#[cfg(test)]
mod map_tests {
    use super::*;

    /// A title's Map reaches the PC's only for a resource the PC can map: a buffer in an upload,
    /// readback or CPU-visible custom heap. One in a DEFAULT heap, a custom heap without CPU pages
    /// or a reserved one gets its title address instead (the PC's Map on it is an invalid call).
    #[test]
    fn only_resources_with_cpu_pages_go_to_the_pc_map() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                return; // no D3D12 adapter
            }
            let dev = dev.unwrap();
            let buffer = |props: D3D12_HEAP_PROPERTIES, state: D3D12_RESOURCE_STATES| -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 0x1_0000,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).expect("buffer");
                r.unwrap()
            };
            let heap = |t| D3D12_HEAP_PROPERTIES { Type: t, ..Default::default() };
            let mappable = |r: &ID3D12Resource| pc_can_map(r.as_raw());

            assert!(mappable(&buffer(heap(D3D12_HEAP_TYPE_UPLOAD), D3D12_RESOURCE_STATE_GENERIC_READ)));
            assert!(mappable(&buffer(heap(D3D12_HEAP_TYPE_READBACK), D3D12_RESOURCE_STATE_COPY_DEST)));
            let default = buffer(heap(D3D12_HEAP_TYPE_DEFAULT), D3D12_RESOURCE_STATE_COMMON);
            assert!(!mappable(&default));
            // (what the PC answers there – the call this keeps from being made)
            let mut p = std::ptr::null_mut();
            assert!(default.Map(0, None, Some(&mut p)).is_err(), "a DEFAULT-heap buffer is not mappable on the PC");

            let custom = |page| D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: page, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            assert!(mappable(&buffer(custom(D3D12_CPU_PAGE_PROPERTY_WRITE_BACK), D3D12_RESOURCE_STATE_COMMON)));
            assert!(!mappable(&buffer(custom(D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE), D3D12_RESOURCE_STATE_COMMON)));

            // a reserved resource has no heap
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 256,
                Height: 256,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                ..Default::default()
            };
            let mut reserved: Option<ID3D12Resource> = None;
            if dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).is_ok() {
                let r = reserved.unwrap();
                let mut raw = r.as_raw();
                // what dev_create_reserved_resource does with what the PC made
                note_reserved(S_OK, &mut raw);
                assert!(is_reserved(raw) && !pc_can_map(raw), "a reserved resource is not asked for its heap");
                // a buffer at an address a reserved resource left without its last Release seen
                // is not taken for it (its description differs)
                let upload = buffer(heap(D3D12_HEAP_TYPE_UPLOAD), D3D12_RESOURCE_STATE_GENERIC_READ);
                let stale = desc_of(r.as_raw()).unwrap();
                RESERVED.lock().unwrap().as_mut().unwrap().insert(upload.as_raw() as usize, stale);
                assert!(!is_reserved(upload.as_raw()) && mappable(&upload));
                let mut g = RESERVED.lock().unwrap();
                let m = g.as_mut().unwrap();
                m.remove(&(raw as usize));
                m.remove(&(upload.as_raw() as usize));
            }
        }
    }

    #[test]
    fn known_stream_suffix_controls_cacheability() {
        let console_vs = crate::gpu::xbox::tests::sample(
            1,
            &[crate::gpu::xbox::tests::fssm(&crate::gpu::xbox::tests::CODE, &[])],
        );
        let mut stream = StreamOut::default();
        stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &console_vs);
        let (subs, cacheable) = known_subobjects_checked(&stream.0);
        assert!(cacheable);
        assert_eq!(subs.len(), 1);

        let mut unknown = stream.0.clone();
        unknown.extend_from_slice(&999u32.to_le_bytes());
        unknown.extend_from_slice(&0u32.to_le_bytes());
        let (subs, cacheable) = known_subobjects_checked(&unknown);
        assert!(!cacheable, "an unknown suffix must not share a projected-desc cache entry");
        assert_eq!(subs.len(), 1, "the known prefix remains usable");
        let desc = [unknown.len(), unknown.as_ptr() as usize];
        assert!(matches!(unsafe { stream_pipeline(desc.as_ptr() as *const c_void) }, StreamPipeline::Graphics(_, false)));

        let mut truncated_header = stream.0.clone();
        truncated_header.extend_from_slice(&[1, 2, 3, 4]);
        assert!(!known_subobjects_checked(&truncated_header).1);
        let mut truncated_payload = stream.0.clone();
        truncated_payload.pop();
        assert!(!known_subobjects_checked(&truncated_payload).1);
    }

    #[test]
    fn pipeline_stream_node_mask_defaults_and_round_trips() {
        assert_eq!(stream_node_mask(&[]), 0);
        let mut stream = StreamOut::default();
        stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK, &4u32);
        assert_eq!(stream_node_mask(&stream_subobjects(&stream.0).unwrap()), 4);
    }

    #[test]
    fn stream_pipeline_projects_legacy_rasterizer_fields() {
        let console_vs = crate::gpu::xbox::tests::sample(
            1,
            &[crate::gpu::xbox::tests::fssm(&crate::gpu::xbox::tests::CODE, &[])],
        );
        let parse = |ty: u32, raster: &[u8]| -> (D3D12_RASTERIZER_DESC, bool) {
            let mut stream = StreamOut::default();
            stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &console_vs);
            stream.raw(ty, raster, false);
            let desc = [stream.0.len(), stream.0.as_ptr() as usize];
            match unsafe { stream_pipeline(desc.as_ptr() as *const c_void) } {
                StreamPipeline::Graphics(g, cacheable) => (g.RasterizerState, cacheable),
                _ => panic!("expected graphics stream pipeline"),
            }
        };
        let bytes = |p: *const u8, n| unsafe { std::slice::from_raw_parts(p, n) };

        let r1 = D3D12_RASTERIZER_DESC1 {
            FillMode: D3D12_FILL_MODE_WIREFRAME,
            CullMode: D3D12_CULL_MODE_FRONT,
            FrontCounterClockwise: true.into(),
            DepthBias: 7.0,
            DepthBiasClamp: 2.5,
            SlopeScaledDepthBias: 1.5,
            DepthClipEnable: false.into(),
            MultisampleEnable: true.into(),
            AntialiasedLineEnable: true.into(),
            ForcedSampleCount: 4,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON,
        };
        let (got, cacheable) = parse(27, bytes(&r1 as *const _ as *const u8, std::mem::size_of_val(&r1)));
        assert!(cacheable);
        assert_eq!((got.FillMode, got.CullMode, got.DepthBias), (r1.FillMode, r1.CullMode, 7));
        assert!(got.MultisampleEnable.as_bool() && got.AntialiasedLineEnable.as_bool());
        assert_eq!((got.ForcedSampleCount, got.ConservativeRaster), (4, r1.ConservativeRaster));

        let r2 = D3D12_RASTERIZER_DESC2 {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: false.into(),
            DepthBias: -3.0,
            DepthBiasClamp: 3.5,
            SlopeScaledDepthBias: -0.5,
            DepthClipEnable: true.into(),
            LineRasterizationMode: D3D12_LINE_RASTERIZATION_MODE_ALIASED,
            ForcedSampleCount: 8,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON,
        };
        let (got, cacheable) = parse(28, bytes(&r2 as *const _ as *const u8, std::mem::size_of_val(&r2)));
        assert!(cacheable);
        assert_eq!((got.FillMode, got.CullMode, got.DepthBias), (r2.FillMode, r2.CullMode, -3));
        assert!(!got.MultisampleEnable.as_bool() && !got.AntialiasedLineEnable.as_bool());
        assert_eq!((got.ForcedSampleCount, got.ConservativeRaster), (8, r2.ConservativeRaster));
    }

    #[test]
    fn extended_rasterizer_state_stays_in_native_stream() {
        let vs = crate::gpu::xbox::tests::sample(
            1,
            &[crate::gpu::xbox::tests::fssm(&crate::gpu::xbox::tests::CODE, &[])],
        );
        let parse = |stream: &StreamOut| unsafe {
            stream_pipeline([stream.0.len(), stream.0.as_ptr() as usize].as_ptr() as *const c_void)
        };
        let make = |ty, payload: &[u8]| {
            let mut stream = StreamOut::default();
            stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &vs);
            stream.raw(ty, payload, false);
            stream
        };
        let bytes = |p: *const u8, n| unsafe { std::slice::from_raw_parts(p, n) };

        let r1 = D3D12_RASTERIZER_DESC1 { DepthBias: 1.5, FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_BACK, ..Default::default() };
        let s1 = make(27, bytes(&r1 as *const _ as *const u8, std::mem::size_of_val(&r1)));
        assert!(matches!(parse(&s1), StreamPipeline::GraphicsStream));

        let r2_fractional = D3D12_RASTERIZER_DESC2 { DepthBias: -2.25, LineRasterizationMode: D3D12_LINE_RASTERIZATION_MODE_ALIASED, ..Default::default() };
        let s2_fractional = make(28, bytes(&r2_fractional as *const _ as *const u8, std::mem::size_of_val(&r2_fractional)));
        assert!(matches!(parse(&s2_fractional), StreamPipeline::GraphicsStream));

        let r2 = D3D12_RASTERIZER_DESC2 {
            DepthBias: -2.0,
            FillMode: D3D12_FILL_MODE_WIREFRAME,
            CullMode: D3D12_CULL_MODE_FRONT,
            LineRasterizationMode: D3D12_LINE_RASTERIZATION_MODE_QUADRILATERAL_NARROW,
            ForcedSampleCount: 4,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON,
            ..Default::default()
        };
        let mut s2 = make(28, bytes(&r2 as *const _ as *const u8, std::mem::size_of_val(&r2)));
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO, &D3D12_CACHED_PIPELINE_STATE {
            pCachedBlob: vs.as_ptr() as *const c_void, CachedBlobSizeInBytes: vs.len(),
        });
        assert!(matches!(parse(&s2), StreamPipeline::GraphicsStream));

        let (subs, cacheable) = known_subobjects_checked(&s2.0);
        assert!(cacheable);
        let rewritten = unsafe { rewrite_graphics_stream(&subs, 0x1234usize as *mut c_void, b"PC VS", None, None, false, false) };
        let rewritten = stream_subobjects(&rewritten.0).expect("valid rewritten stream");
        assert!(rewritten.iter().all(|(ty, _)| *ty != 19), "cached PSO is never copied");
        assert_eq!(u64::from_le_bytes(rewritten.iter().find(|(ty, _)| *ty == 0).unwrap().1[..8].try_into().unwrap()), 0x1234);
        assert_eq!(unsafe { payload_shader(rewritten.iter().find(|(ty, _)| *ty == 1).unwrap().1) }, Some(&b"PC VS"[..]));
        let got: D3D12_RASTERIZER_DESC2 = unsafe { read_sub(rewritten.iter().find(|(ty, _)| *ty == 28).unwrap().1) };
        assert_eq!(got.DepthBias, -2.0);
        assert_eq!(got.LineRasterizationMode, D3D12_LINE_RASTERIZATION_MODE_QUADRILATERAL_NARROW);
        assert_eq!((got.ForcedSampleCount, got.ConservativeRaster), (4, D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON));

        let vi = D3D12_VIEW_INSTANCING_DESC { ViewInstanceCount: 1, ..Default::default() };
        let vi_only = make(22, bytes(&vi as *const _ as *const u8, std::mem::size_of_val(&vi)));
        assert!(matches!(parse(&vi_only), StreamPipeline::Graphics(_, true)), "VIEW_INSTANCING remains Stage 3");

        let mut unknown = s1.0.clone();
        unknown.extend_from_slice(&999u32.to_le_bytes());
        unknown.extend_from_slice(&0u32.to_le_bytes());
        let desc = [unknown.len(), unknown.as_ptr() as usize];
        assert!(matches!(unsafe { stream_pipeline(desc.as_ptr() as *const c_void) }, StreamPipeline::Graphics(_, false)),
            "an unknown suffix retains Stage 1's uncacheable projected path");

        for ty in [2, 5] {
            let mut unreadable = StreamOut::default();
            unreadable.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &vs);
            unreadable.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE(ty), &D3D12_SHADER_BYTECODE {
                pShaderBytecode: 0x1000usize as *const c_void, BytecodeLength: 64,
            });
            unreadable.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1, &r1);
            let desc = [unreadable.0.len(), unreadable.0.as_ptr() as usize];
            assert!(unsafe { stream_has_unreadable_shader(desc.as_ptr() as *const c_void) });
            assert!(matches!(parse(&unreadable), StreamPipeline::Unknown), "unreadable shader type {ty} must fail closed");
        }
        let mut null_ps = StreamOut::default();
        null_ps.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &vs);
        null_ps.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &D3D12_SHADER_BYTECODE {
            pShaderBytecode: std::ptr::null(), BytecodeLength: 64,
        });
        let desc = [null_ps.0.len(), null_ps.0.as_ptr() as usize];
        assert!(unsafe { stream_has_unreadable_shader(desc.as_ptr() as *const c_void) });
        assert!(matches!(parse(&null_ps), StreamPipeline::Unknown));
    }

    #[test]
    fn stream_pipeline_defaults_depth_enable_from_dsv_format() {
        let console_vs = crate::gpu::xbox::tests::sample(
            1,
            &[crate::gpu::xbox::tests::fssm(&crate::gpu::xbox::tests::CODE, &[])],
        );
        let push_sub = |buf: &mut Vec<u8>, ty: u32, payload: &[u8], align: usize| {
            buf.extend_from_slice(&ty.to_le_bytes());
            if align == 8 {
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
            buf.extend_from_slice(payload);
            while buf.len() % 8 != 0 {
                buf.push(0);
            }
        };
        let parse = |dsv_format: Option<u32>, explicit_depth: Option<bool>| -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            let mut stream = Vec::new();
            let mut vs_bc = [0u8; 16];
            vs_bc[0..8].copy_from_slice(&(console_vs.as_ptr() as usize).to_le_bytes());
            vs_bc[8..16].copy_from_slice(&console_vs.len().to_le_bytes());
            push_sub(&mut stream, 1, &vs_bc, 8);
            if let Some(fmt) = dsv_format {
                push_sub(&mut stream, 16, &fmt.to_le_bytes(), 4);
            }
            if let Some(en) = explicit_depth {
                let ds = D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: en.into(),
                    ..Default::default()
                };
                let bytes = unsafe {
                    std::slice::from_raw_parts(&ds as *const _ as *const u8, std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC>())
                };
                push_sub(&mut stream, 11, bytes, 4);
            }
            #[repr(C)]
            struct StreamDesc {
                size: usize,
                ptr: usize,
            }
            let desc = StreamDesc { size: stream.len(), ptr: stream.as_ptr() as usize };
            match unsafe { stream_pipeline(&desc as *const _ as *const c_void) } {
                StreamPipeline::Graphics(g, _) => *g,
                _ => panic!("expected graphics stream pipeline"),
            }
        };

        // Omitted DEPTH_STENCIL + omitted/UNKNOWN DSVFormat -> DepthEnable defaults to false.
        assert!(!parse(None, None).DepthStencilState.DepthEnable.as_bool());
        assert!(!parse(Some(0), None).DepthStencilState.DepthEnable.as_bool());
        // Omitted DEPTH_STENCIL + non-UNKNOWN DSVFormat (D32_FLOAT = 40) -> DepthEnable defaults to true.
        assert!(parse(Some(40), None).DepthStencilState.DepthEnable.as_bool());
        // Explicit DEPTH_STENCIL subobject is always preserved regardless of DSVFormat.
        assert!(parse(None, Some(true)).DepthStencilState.DepthEnable.as_bool());
        assert!(!parse(Some(40), Some(false)).DepthStencilState.DepthEnable.as_bool());
    }

    /// A console compute stream's FLAGS subobject reaches the projected descriptor:
    /// `dev_create_compute_pso` uses it for the driver's own cache key the same way a
    /// legacy desc's flags are used (#2607).
    #[test]
    fn a_compute_stream_keeps_its_flags_subobject() {
        let console_cs = crate::gpu::xbox::tests::sample(
            1,
            &[crate::gpu::xbox::tests::fssm(&crate::gpu::xbox::tests::CODE, &[])],
        );
        let parse = |flags: Option<u32>| -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
            let mut stream = Vec::new();
            let mut cs_bc = [0u8; 16];
            cs_bc[0..8].copy_from_slice(&(console_cs.as_ptr() as usize).to_le_bytes());
            cs_bc[8..16].copy_from_slice(&console_cs.len().to_le_bytes());
            stream.extend_from_slice(&6u32.to_le_bytes());
            stream.extend_from_slice(&0u32.to_le_bytes());
            stream.extend_from_slice(&cs_bc);
            while stream.len() % 8 != 0 {
                stream.push(0);
            }
            if let Some(f) = flags {
                stream.extend_from_slice(&20u32.to_le_bytes());
                stream.extend_from_slice(&f.to_le_bytes());
                while stream.len() % 8 != 0 {
                    stream.push(0);
                }
            }
            #[repr(C)]
            struct StreamDesc {
                size: usize,
                ptr: usize,
            }
            let desc = StreamDesc { size: stream.len(), ptr: stream.as_ptr() as usize };
            match unsafe { stream_pipeline(&desc as *const _ as *const c_void) } {
                StreamPipeline::Compute(c) => *c,
                _ => panic!("expected compute stream pipeline"),
            }
        };
        assert_eq!(parse(None).Flags.0, 0);
        assert_eq!(parse(Some(1)).Flags.0, 1, "the FLAGS subobject must reach the projected compute desc");
        assert_eq!(parse(Some(3)).Flags.0, 3);
    }
}

#[cfg(test)]
mod mesh_stream_tests {
    use super::*;

    #[test]
    fn driver_blob_export_failure_is_a_cache_miss() {
        unsafe extern "system" fn no_blob(_: *mut c_void, out: *mut *mut c_void) -> HRESULT {
            *out = std::ptr::null_mut();
            0x8000_FFFFu32 as HRESULT // E_UNEXPECTED
        }

        let iface = layout::find("ID3D12PipelineState").unwrap();
        let mut pc = vec![0usize; iface.methods.len()];
        pc[layout::slot(iface, "GetCachedBlob").unwrap()] = no_blob as *const () as usize;
        let mut fake = [pc.as_ptr() as usize];
        assert!(unsafe { read_pso_cached_blob(fake.as_mut_ptr().cast()) }.is_none());
    }

    #[test]
    fn empty_driver_blob_is_a_cache_miss() {
        unsafe extern "system" fn get_blob(this: *mut c_void, out: *mut *mut c_void) -> HRESULT {
            *out = *this.cast::<*mut c_void>().add(1);
            S_OK
        }
        unsafe extern "system" fn release(_: *mut c_void) -> u32 { 0 }
        unsafe extern "system" fn pointer(_: *mut c_void) -> *mut c_void {
            std::ptr::NonNull::<u8>::dangling().as_ptr().cast()
        }
        unsafe extern "system" fn size(_: *mut c_void) -> usize { 0 }

        let blob_vtable = [0, 0, release as *const () as usize, pointer as *const () as usize, size as *const () as usize];
        let mut blob = [blob_vtable.as_ptr() as usize];
        let iface = layout::find("ID3D12PipelineState").unwrap();
        let mut pc = vec![0usize; iface.methods.len()];
        pc[layout::slot(iface, "GetCachedBlob").unwrap()] = get_blob as *const () as usize;
        let mut fake = [pc.as_ptr() as usize, blob.as_mut_ptr() as usize];
        assert!(unsafe { read_pso_cached_blob(fake.as_mut_ptr().cast()) }.is_none());
    }

    #[test]
    fn own_stream_rejects_null_truncated_and_unreadable_streams() {
        unsafe {
            // null desc pointer and empty / null stream buffer
            assert!(own_stream(std::ptr::null()).is_none());
            let empty_desc: [usize; 2] = [0, 0];
            assert!(own_stream(empty_desc.as_ptr() as *const c_void).is_none());

            // truncated shader subobject (MS = 25, align 8, payload 16 B -> needs 24 B, only 16 B given)
            let mut trunc_ms = [0u8; 16];
            trunc_ms[0..4].copy_from_slice(&25u32.to_le_bytes());
            let d_trunc_ms: [usize; 2] = [trunc_ms.len(), trunc_ms.as_ptr() as usize];
            assert!(own_stream(d_trunc_ms.as_ptr() as *const c_void).is_none());

            // truncated INPUT_LAYOUT (12, align 8, payload 16 B -> only 12 B given)
            let mut trunc_il = [0u8; 12];
            trunc_il[0..4].copy_from_slice(&12u32.to_le_bytes());
            let d_trunc_il: [usize; 2] = [trunc_il.len(), trunc_il.as_ptr() as usize];
            assert!(own_stream(d_trunc_il.as_ptr() as *const c_void).is_none());

            // truncated CACHED_PSO (19, align 8, payload 16 B -> only 16 B given)
            let mut trunc_cache = [0u8; 16];
            trunc_cache[0..4].copy_from_slice(&19u32.to_le_bytes());
            let d_trunc_cache: [usize; 2] = [trunc_cache.len(), trunc_cache.as_ptr() as usize];
            assert!(own_stream(d_trunc_cache.as_ptr() as *const c_void).is_none());

            // unreadable shader pointer in MS subobject (must not leak raw pointer into background worker)
            let mut bad_ptr = [0u8; 24];
            bad_ptr[0..4].copy_from_slice(&25u32.to_le_bytes());
            bad_ptr[8..16].copy_from_slice(&0x1000u64.to_le_bytes());
            bad_ptr[16..24].copy_from_slice(&64u64.to_le_bytes());
            let d_bad: [usize; 2] = [bad_ptr.len(), bad_ptr.as_ptr() as usize];
            assert!(own_stream(d_bad.as_ptr() as *const c_void).is_none());

            // valid stream: readable MS blob copied, zero-len stale PS pointer zeroed, CACHED_PSO zeroed
            let shader = vec![0x44u8, 0x58, 0x42, 0x43, 1, 2, 3, 4];
            let mut valid = [0u8; 72];
            // [0..24]: MS (25) with readable shader
            valid[0..4].copy_from_slice(&25u32.to_le_bytes());
            valid[8..16].copy_from_slice(&(shader.as_ptr() as u64).to_le_bytes());
            valid[16..24].copy_from_slice(&(shader.len() as u64).to_le_bytes());
            // [24..48]: PS (2) with non-null pointer and length 0
            valid[24..28].copy_from_slice(&2u32.to_le_bytes());
            valid[32..40].copy_from_slice(&0xDEAD_BEEFu64.to_le_bytes());
            valid[40..48].copy_from_slice(&0u64.to_le_bytes());
            // [48..72]: CACHED_PSO (19) with non-zero bytes
            valid[48..52].copy_from_slice(&19u32.to_le_bytes());
            valid[56..72].fill(0xAA);

            let d_valid: [usize; 2] = [valid.len(), valid.as_ptr() as usize];
            let owned = own_stream(d_valid.as_ptr() as *const c_void).expect("valid stream");
            let OwnedDesc::Stream(_, buf, blobs) = owned else { panic!("expected Stream") };
            assert_eq!(blobs.len(), 1);
            assert_eq!(&*blobs[0], shader.as_slice());
            let rebased = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
            assert_eq!(rebased, blobs[0].as_ptr() as usize);
            assert_ne!(rebased, shader.as_ptr() as usize);
            assert_eq!(&buf[32..48], &[0u8; 16], "zero-len shader entry cleared");
            assert_eq!(&buf[56..72], &[0u8; 16], "cached PSO cleared");

            // A retained stream must not extend the title's root-signature lifetime. Translation
            // replaces it, while the shader payload remains rebased to recipe-owned bytes.
            let mut rooted = [0u8; 40];
            rooted[0..4].copy_from_slice(&0u32.to_le_bytes());
            rooted[8..16].copy_from_slice(&0xDEAD_BEEFu64.to_le_bytes());
            rooted[16..20].copy_from_slice(&25u32.to_le_bytes());
            rooted[24..32].copy_from_slice(&(shader.as_ptr() as u64).to_le_bytes());
            rooted[32..40].copy_from_slice(&(shader.len() as u64).to_le_bytes());
            let rooted_desc = [rooted.len(), rooted.as_ptr() as usize];
            let OwnedDesc::Stream(_, rooted_buf, rooted_blobs) = own_stream(rooted_desc.as_ptr().cast()).unwrap() else {
                panic!("expected Stream")
            };
            assert_eq!(&rooted_buf[8..16], &[0; 8]);
            assert_eq!(u64::from_le_bytes(rooted_buf[24..32].try_into().unwrap()) as usize, rooted_blobs[0].as_ptr() as usize);
        }
    }

    unsafe extern "system" fn fake_create_pso_null_ok(
        _: *mut c_void,
        _: *const c_void,
        _: *const GUID,
        ppv: *mut *mut c_void,
    ) -> HRESULT {
        if !ppv.is_null() {
            *ppv = std::ptr::null_mut();
        }
        S_OK
    }

    unsafe extern "system" fn fake_create_compute_unexpected(
        _: *mut c_void,
        _: *const c_void,
        _: *const GUID,
        _: *mut *mut c_void,
    ) -> HRESULT {
        panic!("CreateComputePipelineState must not be called when mesh PSO out-pointer is null");
    }

    #[test]
    fn finish_two_pass_rejects_null_mesh_out_pointer() {
        let iface = layout::find("ID3D12Device2").expect("ID3D12Device2 layout");
        let pso_slot = layout::slot(iface, "CreatePipelineState").expect("CreatePipelineState slot");
        let cs_slot = layout::slot(iface, "CreateComputePipelineState").expect("CreateComputePipelineState slot");
        let mut pc = vec![0usize; iface.methods.len()];
        pc[pso_slot] = fake_create_pso_null_ok as usize;
        pc[cs_slot] = fake_create_compute_unexpected as usize;

        let ext_buf = [pc.as_ptr() as usize, 0];
        let ext_vtbl = ext_buf.as_ptr() as usize + std::mem::size_of::<usize>();
        let mut fake_dev = [ext_vtbl];
        let this = fake_dev.as_mut_ptr() as *mut c_void;

        // Unique synthetic sampler key so we never touch global ABI_RS or real sampler caches.
        let sampler: crate::gpu::recomp::pso::StaticSampler = ([0x55AA_1968, 0xCAFE_1968, 0xBEEF_1968, 0x0001_1968], true);
        let key = (this as usize, sampler.0.iter().copied().chain([1u32]).collect::<Vec<_>>());
        let other_key = (0x0D15_71C7usize, key.1.clone());

        struct Cleanup {
            ext: usize,
            keys: Vec<(usize, Vec<u32>)>,
        }
        impl Drop for Cleanup {
            fn drop(&mut self) {
                CLASSES.lock().unwrap().retain(|c| c.ext != self.ext);
                if let Some(m) = ABI_RS_SAMPLERS.lock().unwrap().as_mut() {
                    for key in &self.keys { m.remove(key); }
                }
            }
        }
        CLASSES.lock().unwrap().push(Class {
            orig: pc.as_ptr() as usize,
            kind: Kind::Device,
            ext: ext_vtbl,
            iface,
            public: iface.methods.len(),
            scanned: iface.methods.len(),
        });
        {
            let mut roots = ABI_RS_SAMPLERS.lock().unwrap();
            let roots = roots.get_or_insert_with(HashMap::new);
            roots.insert(key.clone(), 0x1234_5678);
            roots.insert(other_key.clone(), 0x8765_4321);
        }
        assert_eq!(unsafe { abi_rs_for(this, &[sampler]) }, Some(0x1234_5678 as *mut c_void));
        assert_eq!(unsafe { abi_rs_for(other_key.0 as *mut c_void, &[sampler]) }, Some(0x8765_4321 as *mut c_void));
        let _cleanup = Cleanup { ext: ext_vtbl, keys: vec![key, other_key] };

        let make_tp = || crate::gpu::recomp::pso::TwoPass {
            cs: vec![0; 16],
            ms: vec![0; 16],
            record: crate::gpu::recomp::MsInterface { outputs: Vec::new(), threads: 32, max_vertices: 3, max_primitives: 1 },
            mesh_groups_per_record: 1,
            samplers: vec![sampler],
        };
        let stream = StreamOut::default();
        let iid = pc_iid(Kind::PipelineState);

        unsafe {
            // Case 1: driver returns S_OK but leaves *ppv == null
            let mut out: *mut c_void = std::ptr::null_mut();
            let hr = finish_two_pass(this, &stream, make_tp(), 0x1234_5678 as *mut c_void, &[], 0, "MS", &iid, &mut out);
            assert_eq!(hr, E_INVALIDARG);
            assert!(out.is_null());

            // Case 2: caller passes ppv == null
            let hr = finish_two_pass(this, &stream, make_tp(), 0x1234_5678 as *mut c_void, &[], 0, "MS", &iid, std::ptr::null_mut());
            assert_eq!(hr, E_INVALIDARG);
        }
    }
}

#[cfg(test)]
mod uav_companion_srv_tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    fn tex_desc(dim: D3D12_RESOURCE_DIMENSION, w: u64, h: u32, d: u16, mips: u16, fmt: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: dim,
            Width: w,
            Height: h,
            DepthOrArraySize: d,
            MipLevels: mips,
            Format: fmt,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
            ..Default::default()
        }
    }

    #[test]
    fn uav_companion_srv_covers_all_image_dimensions_and_wildcards() {
        let uav_flag = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        // Texture1D
        let rd1d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 16, 1, 1, 4, DXGI_FORMAT_R32_UINT, uav_flag);
        let u1d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture1D: D3D12_TEX1D_UAV { MipSlice: 2 } },
        };
        let s1d = uav_companion_srv(Some(&rd1d), Some(&u1d), false).expect("1D companion SRV");
        assert_eq!((s1d.ViewDimension, s1d.Format), (D3D12_SRV_DIMENSION_TEXTURE1D, DXGI_FORMAT_R32_UINT));
        assert_eq!(unsafe { (s1d.Anonymous.Texture1D.MostDetailedMip, s1d.Anonymous.Texture1D.MipLevels) }, (2, 1));

        // Texture1DArray with ArraySize = u32::MAX wildcard
        let rd1da = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 16, 1, 5, 2, DXGI_FORMAT_R32_UINT, uav_flag);
        let u1da = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture1DArray: D3D12_TEX1D_ARRAY_UAV { MipSlice: 1, FirstArraySlice: 2, ArraySize: u32::MAX },
            },
        };
        let s1da = uav_companion_srv(Some(&rd1da), Some(&u1da), false).expect("1DArray companion SRV");
        assert_eq!((s1da.ViewDimension, s1da.Format), (D3D12_SRV_DIMENSION_TEXTURE1DARRAY, DXGI_FORMAT_R32_UINT));
        assert_eq!(
            unsafe { (s1da.Anonymous.Texture1DArray.MostDetailedMip, s1da.Anonymous.Texture1DArray.MipLevels, s1da.Anonymous.Texture1DArray.FirstArraySlice, s1da.Anonymous.Texture1DArray.ArraySize) },
            (1, 1, 2, 3)
        );

        // Texture2D with PlaneSlice
        let rd2d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 32, 32, 1, 3, DXGI_FORMAT_R16G16B16A16_FLOAT, uav_flag);
        let u2d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV { MipSlice: 1, PlaneSlice: 1 } },
        };
        let s2d = uav_companion_srv(Some(&rd2d), Some(&u2d), false).expect("2D companion SRV");
        assert_eq!(s2d.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2D);
        assert_eq!(unsafe { (s2d.Anonymous.Texture2D.MostDetailedMip, s2d.Anonymous.Texture2D.MipLevels, s2d.Anonymous.Texture2D.PlaneSlice) }, (1, 1, 1));

        // Texture2DArray with ArraySize = u32::MAX wildcard
        let rd2da = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 32, 32, 6, 2, DXGI_FORMAT_R32_FLOAT, uav_flag);
        let u2da = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 2, ArraySize: u32::MAX, PlaneSlice: 0 },
            },
        };
        let s2da = uav_companion_srv(Some(&rd2da), Some(&u2da), false).expect("2DArray companion SRV");
        assert_eq!(s2da.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2DARRAY);
        assert_eq!(
            unsafe { (s2da.Anonymous.Texture2DArray.MostDetailedMip, s2da.Anonymous.Texture2DArray.MipLevels, s2da.Anonymous.Texture2DArray.FirstArraySlice, s2da.Anonymous.Texture2DArray.ArraySize) },
            (0, 1, 2, 4)
        );

        // Texture3D
        let rd3d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 8, 8, 8, 3, DXGI_FORMAT_R16G16B16A16_FLOAT, uav_flag);
        let u3d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_UAV { MipSlice: 1, FirstWSlice: 0, WSize: 4 } },
        };
        let s3d = uav_companion_srv(Some(&rd3d), Some(&u3d), false).expect("3D companion SRV");
        assert_eq!((s3d.ViewDimension, s3d.Format), (D3D12_SRV_DIMENSION_TEXTURE3D, DXGI_FORMAT_R16G16B16A16_FLOAT));
        assert_eq!(unsafe { (s3d.Anonymous.Texture3D.MostDetailedMip, s3d.Anonymous.Texture3D.MipLevels) }, (1, 1));

        // Default UAV (desc = None) on 3D and 2DArray
        let def3d = uav_companion_srv(Some(&rd3d), None, false).expect("default 3D companion SRV");
        assert_eq!((def3d.ViewDimension, def3d.Format), (D3D12_SRV_DIMENSION_TEXTURE3D, DXGI_FORMAT_R16G16B16A16_FLOAT));
        assert_eq!(unsafe { (def3d.Anonymous.Texture3D.MostDetailedMip, def3d.Anonymous.Texture3D.MipLevels) }, (0, 1));
        let def2da = uav_companion_srv(Some(&rd2da), None, false).expect("default 2DArray companion SRV");
        assert_eq!(def2da.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2DARRAY);
        assert_eq!(unsafe { def2da.Anonymous.Texture2DArray.ArraySize }, 6);

        // Null resource / DENY_SHADER_RESOURCE normalizes UNKNOWN and clamps ranges
        let null_u3d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_UAV { MipSlice: 3, FirstWSlice: 1, WSize: u32::MAX } },
        };
        let ns3d = uav_companion_srv(None, Some(&null_u3d), true).expect("null 3D companion SRV");
        assert_eq!((ns3d.ViewDimension, ns3d.Format), (D3D12_SRV_DIMENSION_TEXTURE3D, DXGI_FORMAT_R32_FLOAT));
        assert_eq!(unsafe { (ns3d.Anonymous.Texture3D.MostDetailedMip, ns3d.Anonymous.Texture3D.MipLevels) }, (0, 1));

        let deny_typeless = tex_desc(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            16,
            16,
            1,
            1,
            DXGI_FORMAT_R32_TYPELESS,
            uav_flag | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE,
        );
        let ns_deny = uav_companion_srv(Some(&deny_typeless), None, true).expect("DENY_SHADER_RESOURCE null SRV");
        assert_eq!((ns_deny.ViewDimension, ns_deny.Format), (D3D12_SRV_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_FLOAT));

        // Buffer UAV has no image companion SRV
        let ubuf = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        };
        assert!(uav_companion_srv(None, Some(&ubuf), false).is_none());

        // null_uav_desc matches dimension
        assert_eq!(null_uav_desc(Some(&u1d), None).ViewDimension, D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(null_uav_desc(Some(&u1da), None).ViewDimension, D3D12_UAV_DIMENSION_TEXTURE1DARRAY);
        assert_eq!(null_uav_desc(Some(&u2da), None).ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        assert_eq!(null_uav_desc(Some(&u3d), None).ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        assert_eq!(null_uav_desc(None, Some(&rd3d)).ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
    }

    /// GPU: verify on real SM 6.6 `ResourceDescriptorHeap` execution that:
    /// 1. `dev_create_uav_x` populates companion SRVs at `ResourceDescriptorHeap[t[0]]` for all
    ///    image UAV dimensions (`Texture1D`, `Texture1DArray`, `Texture2D`, `Texture2DArray`,
    ///    `Texture3D`, including `ArraySize = u32::MAX` and `Texture3D` mip > 0).
    /// 2. `dev_create_srv_x` and `dev_create_uav_x` with `pResource == NULL` write valid T#
    ///    descriptors referencing `slot.index` (never `[0; 8]` -> `ResourceDescriptorHeap[0]`),
    ///    so reading them on the GPU returns `0` instead of aliasing slot 0 or faulting.
    #[test]
    fn gpu_uav_companion_srv_and_null_image_views() {
        const CHILD: &str = "XCL_TEST_UAV_COMPANION_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "hle::d3d12x::real::uav_companion_srv_tests::gpu_uav_companion_srv_and_null_image_views", "--nocapture"])
                .env(CHILD, "1")
                .env("XCL_UMA_MB", "64")
                .env("XCL_COHERENT_MB", "64")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "child failed ({:?}):\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        unsafe {
            let mut dev_opt: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev_opt).is_err() {
                crate::gpu::recomp::runner::skip_test("no D3D12 hardware GPU");
                return;
            }
            let dev = dev_opt.unwrap();
            let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_6 };
            if dev.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as *mut c_void, std::mem::size_of_val(&sm) as u32).is_err()
                || sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0
            {
                crate::gpu::recomp::runner::skip_test("GPU lacks SM 6.6 ResourceDescriptorHeap");
                return;
            }

            assert!(crate::gpu::uma::get().is_some(), "UMA init failed in test subprocess");
            adopt(dev.as_raw(), Kind::Device);

            let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&heap_desc).expect("shader-visible heap");
            let pc_cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
            let pc_gpu = heap.GetGPUDescriptorHandleForHeapStart().ptr;
            let heap_key = heap.as_raw() as usize;
            assert!(super::super::console_heap::register(heap_key, 16, pc_cpu, pc_gpu, inc, false, false, true));
            let base_cpu = super::super::console_heap::start(heap_key).expect("console heap start");
            let slot_cpu = |k: u64| base_cpu + k * super::super::console_heap::SLOT as u64;
            let read_t = |k: u64| -> [u32; 8] {
                let mut w = [0u32; 8];
                std::ptr::copy_nonoverlapping(slot_cpu(k) as *const u32, w.as_mut_ptr(), 8);
                w
            };

            let default_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let make_tex = |desc: &D3D12_RESOURCE_DESC| -> ID3D12Resource {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut r)
                    .expect("CreateCommittedResource texture");
                r.unwrap()
            };

            let uav_flag = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            // Slot 0: poison texture (written with 0xBAD0_BAD0) so any null view that wrongly
            // indexes ResourceDescriptorHeap[0] reads 0xBAD0_BAD0 instead of 0.
            let t0 = make_tex(&tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 4, 4, 4, 1, DXGI_FORMAT_R32_UINT, uav_flag));
            let t1d = make_tex(&tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 4, 1, 1, 1, DXGI_FORMAT_R32_UINT, uav_flag));
            let t1da = make_tex(&tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 4, 1, 3, 1, DXGI_FORMAT_R32_UINT, uav_flag));
            let t2d = make_tex(&tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 4, 4, 1, 1, DXGI_FORMAT_R32_UINT, uav_flag));
            let t2da = make_tex(&tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 4, 4, 3, 1, DXGI_FORMAT_R32_UINT, uav_flag));
            let t3d = make_tex(&tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 4, 4, 4, 2, DXGI_FORMAT_R32_UINT, uav_flag));

            let u0 = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_UAV { MipSlice: 0, FirstWSlice: 0, WSize: 4 } },
            };
            let u1d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture1D: D3D12_TEX1D_UAV { MipSlice: 0 } },
            };
            let u1da = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1DARRAY,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture1DArray: D3D12_TEX1D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 1, ArraySize: u32::MAX },
                },
            };
            let u2d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 } },
            };
            let u2da = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 1, ArraySize: u32::MAX, PlaneSlice: 0 },
                },
            };
            let u3d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_UAV { MipSlice: 1, FirstWSlice: 0, WSize: 2 } },
            };
            let null_srv_3d = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_SRV { MostDetailedMip: 0, MipLevels: 1, ResourceMinLODClamp: 0.0 } },
            };
            let null_uav_3d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_UAV { MipSlice: 0, FirstWSlice: 0, WSize: 1 } },
            };
            let null_srv_2da = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MostDetailedMip: 0, MipLevels: 1, FirstArraySlice: 0, ArraySize: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
                },
            };

            dev_create_uav_x(dev.as_raw(), t0.as_raw(), std::ptr::null_mut(), &u0, slot_cpu(0));
            dev_create_uav_x(dev.as_raw(), t1d.as_raw(), std::ptr::null_mut(), &u1d, slot_cpu(1));
            dev_create_uav_x(dev.as_raw(), t1da.as_raw(), std::ptr::null_mut(), &u1da, slot_cpu(2));
            dev_create_uav_x(dev.as_raw(), t2d.as_raw(), std::ptr::null_mut(), &u2d, slot_cpu(3));
            dev_create_uav_x(dev.as_raw(), t2da.as_raw(), std::ptr::null_mut(), &u2da, slot_cpu(4));
            dev_create_uav_x(dev.as_raw(), t3d.as_raw(), std::ptr::null_mut(), &u3d, slot_cpu(5));
            dev_create_srv_x(dev.as_raw(), std::ptr::null_mut(), &null_srv_3d, slot_cpu(6));
            dev_create_uav_x(dev.as_raw(), std::ptr::null_mut(), std::ptr::null_mut(), &null_uav_3d, slot_cpu(7));
            dev_create_srv_x(dev.as_raw(), std::ptr::null_mut(), &null_srv_2da, slot_cpu(8));

            // Also exercise CPU-only staging heap -> shader-visible CopyDescriptorsSimple:
            // slot 9 receives a copy of the Texture3D UAV (slot 5) and slot 10 receives a copy
            // of the null Texture3D SRV (slot 6), verifying both PC descriptors and T# renumbering.
            let cpu_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            let cpu_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&cpu_heap_desc).expect("CPU-only heap");
            let cpu_heap_key = cpu_heap.as_raw() as usize;
            assert!(super::super::console_heap::register(
                cpu_heap_key,
                4,
                cpu_heap.GetCPUDescriptorHandleForHeapStart().ptr as u64,
                0,
                inc,
                false,
                false,
                false,
            ));
            let stage_base = super::super::console_heap::start(cpu_heap_key).expect("staging console heap start");
            let stage_cpu = |k: u64| stage_base + k * super::super::console_heap::SLOT as u64;
            dev_create_uav_x(dev.as_raw(), t3d.as_raw(), std::ptr::null_mut(), &u3d, stage_cpu(0));
            dev_create_srv_x(dev.as_raw(), std::ptr::null_mut(), &null_srv_3d, stage_cpu(1));
            dev_copy_descriptors_simple_x(dev.as_raw(), 2, slot_cpu(9), stage_cpu(0), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);

            let w0 = read_t(0);
            let w1d = read_t(1);
            let w1da = read_t(2);
            let w2d = read_t(3);
            let w2da = read_t(4);
            let w3d = read_t(5);
            let wn_srv3d = read_t(6);
            let wn_uav3d = read_t(7);
            let wn_srv2da = read_t(8);
            let w_copy_3d = read_t(9);
            let w_copy_null = read_t(10);

            // Verify Texture3D mip 1 depth is scaled (4 >> 1 = 2 -> depth - 1 = 1 in word 4 [12:0]).
            assert_eq!((w3d[0], w3d[3] >> 28, w3d[4] & 0x1FFF), (10, 10, 1), "3D mip 1 T# descriptor");
            // Verify null image SRVs/UAVs carry their real non-zero slot index and image type.
            assert_eq!((wn_srv3d[0], wn_srv3d[3] >> 28), (12, 10), "null 3D SRV must reference slot 6 (index 12)");
            assert_eq!((wn_uav3d[0], wn_uav3d[3] >> 28), (14, 10), "null 3D UAV must reference slot 7 (index 14)");
            assert_eq!((wn_srv2da[0], wn_srv2da[3] >> 28), (16, 13), "null 2DArray SRV must reference slot 8 (index 16)");
            // Verify copied descriptors were renumbered to their destination heap indices.
            assert_eq!((w_copy_3d[0], w_copy_3d[3] >> 28), (18, 10), "copied 3D UAV must renumber to slot 9 (index 18)");
            assert_eq!((w_copy_null[0], w_copy_null[3] >> 28), (20, 10), "copied null 3D SRV must renumber to slot 10 (index 20)");

            // Output buffer (10 uints) + readback buffer.
            let buf_desc = |flags: D3D12_RESOURCE_FLAGS| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut out_buf: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buf_desc(uav_flag), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut out_buf)
                .expect("output buffer");
            let out_buf = out_buf.unwrap();
            let rb_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let mut rb_buf: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &buf_desc(D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb_buf)
                .expect("readback buffer");
            let rb_buf = rb_buf.unwrap();

            // Root signature: 12 root constants at b0 + 1 root UAV at u0 + direct ResourceDescriptorHeap access.
            let params = [
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 12 } },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                ..Default::default()
            };
            let mut ser = None;
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .expect("create RS");

            let write_hlsl = r#"
cbuffer Idx : register(b0) {
    uint i0, i1d, i1da, i2d, i2da, i3d, in_srv3d, in_uav3d, in_srv2da, icopy_3d, icopy_null, pad;
};
[numthreads(1, 1, 1)]
void main() {
    RWTexture3D<uint> u0 = ResourceDescriptorHeap[i0 + 1];
    u0[uint3(0, 0, 0)] = 0xBAD0BAD0u;
    RWTexture1D<uint> u1d = ResourceDescriptorHeap[i1d + 1];
    u1d[2] = 0x11110001u;
    RWTexture1DArray<uint> u1da = ResourceDescriptorHeap[i1da + 1];
    u1da[uint2(2, 1)] = 0x22220002u;
    RWTexture2D<uint> u2d = ResourceDescriptorHeap[i2d + 1];
    u2d[uint2(1, 2)] = 0x33330003u;
    RWTexture2DArray<uint> u2da = ResourceDescriptorHeap[i2da + 1];
    u2da[uint3(1, 2, 1)] = 0x44440004u;
    RWTexture3D<uint> u3d = ResourceDescriptorHeap[i3d + 1];
    u3d[uint3(1, 0, 1)] = 0x55550005u;
    RWTexture3D<uint> unull = ResourceDescriptorHeap[in_uav3d + 1];
    unull[uint3(0, 0, 0)] = 0xDEADCAFEu;
}
"#;
            let read_hlsl = r#"
cbuffer Idx : register(b0) {
    uint i0, i1d, i1da, i2d, i2da, i3d, in_srv3d, in_uav3d, in_srv2da, icopy_3d, icopy_null, pad;
};
RWByteAddressBuffer out_buf : register(u0);
[numthreads(1, 1, 1)]
void main() {
    Texture1D<uint> s1d = ResourceDescriptorHeap[i1d];
    Texture1DArray<uint> s1da = ResourceDescriptorHeap[i1da];
    Texture2D<uint> s2d = ResourceDescriptorHeap[i2d];
    Texture2DArray<uint> s2da = ResourceDescriptorHeap[i2da];
    Texture3D<uint> s3d = ResourceDescriptorHeap[i3d];
    Texture3D<float> ns3d = ResourceDescriptorHeap[in_srv3d];
    Texture3D<uint> nu3d = ResourceDescriptorHeap[in_uav3d];
    Texture2DArray<uint> ns2da = ResourceDescriptorHeap[in_srv2da];
    Texture3D<uint> scopy3d = ResourceDescriptorHeap[icopy_3d];
    Texture3D<float> scopynull = ResourceDescriptorHeap[icopy_null];
    out_buf.Store(0,  s1d.Load(int2(2, 0)));
    out_buf.Store(4,  s1da.Load(int3(2, 1, 0)));
    out_buf.Store(8,  s2d.Load(int3(1, 2, 0)));
    out_buf.Store(12, s2da.Load(int4(1, 2, 1, 0)));
    out_buf.Store(16, s3d.Load(int4(1, 0, 1, 0)));
    out_buf.Store(20, asuint(ns3d.Load(int4(0, 0, 0, 0))));
    out_buf.Store(24, nu3d.Load(int4(0, 0, 0, 0)));
    out_buf.Store(28, ns2da.Load(int4(0, 0, 0, 0)));
    out_buf.Store(32, scopy3d.Load(int4(1, 0, 1, 0)));
    out_buf.Store(36, asuint(scopynull.Load(int4(0, 0, 0, 0))));
}
"#;
            let write_dxil = crate::gpu::recomp::dxc::compile(write_hlsl, "main", "cs_6_6", &[]).expect("compile write CS");
            let read_dxil = crate::gpu::recomp::dxc::compile(read_hlsl, "main", "cs_6_6", &[]).expect("compile read CS");

            let make_pso = |dxil: &[u8]| -> ID3D12PipelineState {
                let d = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: bytecode(dxil),
                    ..Default::default()
                };
                let pso = dev.CreateComputePipelineState(&d).expect("create compute PSO");
                let _ = std::mem::ManuallyDrop::into_inner(d.pRootSignature);
                pso
            };
            let write_pso = make_pso(&write_dxil);
            let read_pso = make_pso(&read_dxil);

            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("queue");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&write_pso)).expect("list");

            let idx_consts: [u32; 12] = [w0[0], w1d[0], w1da[0], w2d[0], w2da[0], w3d[0], wn_srv3d[0], wn_uav3d[0], wn_srv2da[0], w_copy_3d[0], w_copy_null[0], 0];
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootSignature(&rs);
            list.SetComputeRoot32BitConstants(0, 12, idx_consts.as_ptr() as *const c_void, 0);
            list.SetComputeRootUnorderedAccessView(1, out_buf.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);

            let transition = |r: &ID3D12Resource| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(Some(r.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    }),
                },
            };
            let barriers = [transition(&t0), transition(&t1d), transition(&t1da), transition(&t2d), transition(&t2da), transition(&t3d)];
            list.ResourceBarrier(&barriers);
            for b in barriers {
                let tr = std::mem::ManuallyDrop::into_inner(b.Anonymous.Transition);
                let _ = std::mem::ManuallyDrop::into_inner(tr.pResource);
            }

            list.SetPipelineState(&read_pso);
            list.Dispatch(1, 1, 1);

            let out_tr = [D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(Some(out_buf.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            }];
            list.ResourceBarrier(&out_tr);
            for b in out_tr {
                let tr = std::mem::ManuallyDrop::into_inner(b.Anonymous.Transition);
                let _ = std::mem::ManuallyDrop::into_inner(tr.pResource);
            }
            list.CopyBufferRegion(&rb_buf, 0, &out_buf, 0, 40);
            list.Close().expect("close list");

            let cmd: ID3D12CommandList = list.cast().unwrap();
            q.ExecuteCommandLists(&[Some(cmd)]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            q.Signal(&fence, 1).expect("signal");
            while fence.GetCompletedValue() < 1 {
                assert!(dev.GetDeviceRemovedReason().is_ok(), "device removed during GPU test: {:?}", dev.GetDeviceRemovedReason());
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(dev.GetDeviceRemovedReason().is_ok(), "device removed: {:?}", dev.GetDeviceRemovedReason());

            let mut mapped = std::ptr::null_mut();
            rb_buf.Map(0, None, Some(&mut mapped)).expect("map readback");
            let got = std::slice::from_raw_parts(mapped as *const u32, 10).to_vec();
            rb_buf.Unmap(0, None);

            super::super::console_heap::unregister(cpu_heap_key);
            super::super::console_heap::unregister(heap_key);

            assert_eq!(
                got,
                vec![0x1111_0001, 0x2222_0002, 0x3333_0003, 0x4444_0004, 0x5555_0005, 0, 0, 0, 0x5555_0005, 0],
                "companion SRVs (1D, 1DArray, 2D, 2DArray, 3D), null image views, and copied staging slots must all read back accurately"
            );
        }
    }
}

#[cfg(test)]
mod list_release_tests {
    use super::*;

    #[repr(C)]
    struct FakeList {
        vtable: usize,
        refs: u32,
    }

    const ALLOC: usize = 0x7E57_A130;
    const PSO: usize = 0x7E57_D002;

    unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
        let l = &mut *(this as *mut FakeList);
        l.refs += 1;
        l.refs
    }

    // the runtime frees the list, and a list created on another thread gets its address at once
    unsafe extern "system" fn release(this: *mut c_void) -> u32 {
        let l = &mut *(this as *mut FakeList);
        l.refs -= 1;
        if l.refs == 0 {
            super::super::bind::reset(this, ALLOC, PSO);
            super::super::alloc_pool::created_on(this as usize, ALLOC, 0);
        }
        l.refs
    }

    #[test]
    fn a_new_list_at_a_released_lists_address_keeps_its_state() {
        let pc = [0, add_ref as usize, release as usize];
        let ext = [pc.as_ptr() as usize, 0];
        let mut list = FakeList { vtable: ext.as_ptr() as usize + 8, refs: 2 };
        let this = &mut list as *mut FakeList as *mut c_void;
        super::super::bind::reset(this, ALLOC + 0x10, PSO + 0x10);
        let (kept, n) = unsafe {
            assert_eq!(cl_release(this), 1);
            (super::super::bind::tracked_pso(this), cl_release(this))
        };
        let pso = super::super::bind::tracked_pso(this);
        let backing = super::super::alloc_pool::recording_backing_for_test(this as usize);
        super::super::bind::forget(this);
        super::super::alloc_pool::forget_list(this as usize);
        assert_eq!(kept, PSO + 0x10, "another reference is left");
        assert_eq!(n, 0);
        assert_eq!(pso, PSO, "the new list's bind state");
        assert_eq!(backing, Some(ALLOC), "the new list's allocator");
    }
}

#[cfg(test)]
mod heap_release_order_tests {
    use super::*;

    #[repr(C)]
    struct FakeHeap {
        vtable: *const usize,
        refs: u32,
        scripted: bool,
        releases: u32,
        freed: fn(usize),
        ext_vtable: [usize; 4],
        orig_vtable: [usize; 3],
    }

    std::thread_local! {
        static EVENTS: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    fn event(name: &'static str) {
        EVENTS.with(|events| events.borrow_mut().push(name));
    }

    unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
        let heap = &mut *(this as *mut FakeHeap);
        if heap.scripted {
            3 // Make the helper take its compatibility path deterministically.
        } else {
            heap.refs += 1;
            heap.refs
        }
    }

    unsafe extern "system" fn release(this: *mut c_void) -> u32 {
        let heap = &mut *(this as *mut FakeHeap);
        if heap.scripted {
            heap.releases += 1;
            if heap.releases == 2 {
                (heap.freed)(this as usize);
                0
            } else {
                2
            }
        } else {
            heap.refs -= 1;
            if heap.refs == 0 {
                (heap.freed)(this as usize);
            }
            heap.refs
        }
    }

    fn fake(refs: u32, scripted: bool, freed: fn(usize)) -> *mut FakeHeap {
        let heap = Box::into_raw(Box::new(FakeHeap {
            vtable: std::ptr::null(),
            refs,
            scripted,
            releases: 0,
            freed,
            ext_vtable: [0; 4],
            orig_vtable: [0; 3],
        }));
        unsafe {
            (*heap).orig_vtable[1] = add_ref as usize;
            (*heap).orig_vtable[2] = release as usize;
            (*heap).ext_vtable[0] = (*heap).orig_vtable.as_ptr() as usize;
            (*heap).ext_vtable[2] = add_ref as usize;
            (*heap).ext_vtable[3] = release as usize;
            (*heap).vtable = (*heap).ext_vtable.as_ptr().add(1);
        }
        heap
    }

    fn note_freed(_: usize) {
        event("freed");
    }

    #[test]
    fn last_release_takes_before_free_and_frees_afterward() {
        EVENTS.with(|events| events.borrow_mut().clear());
        let heap = fake(2, false, note_freed);
        unsafe {
            assert_eq!(release_taking_first(heap.cast(), || event("take"), |()| event("after")), 1);
            assert!(EVENTS.with(|events| events.borrow().is_empty()), "a non-final release keeps the state");
            assert_eq!(release_taking_first(heap.cast(), || event("take"), |()| event("after")), 0);
            drop(Box::from_raw(heap));
        }
        assert_eq!(EVENTS.with(|events| events.borrow().clone()), ["take", "freed", "after"]);
    }

    #[test]
    fn missed_probe_takes_once_after_final_release() {
        EVENTS.with(|events| events.borrow_mut().clear());
        let heap = fake(0, true, note_freed);
        unsafe {
            assert_eq!(release_taking_first(heap.cast(), || event("take"), |()| event("after")), 0);
            drop(Box::from_raw(heap));
        }
        assert_eq!(
            EVENTS.with(|events| events.borrow().clone()),
            ["freed", "take", "after"],
            "the compatibility fallback cannot close the post-free address-reuse window"
        );
    }

    #[test]
    fn memory_heap_release_keeps_replacement_at_reused_address() {
        fn replace(addr: usize) {
            let replacement = HeapUma { size: 0x2_0000, align: 0x1_0000, prefer_coherent: false, cpu_readback: false, block: None };
            HEAP_UMA.lock().unwrap().get_or_insert_with(HashMap::new).insert(addr, replacement);
        }

        let heap = fake(1, false, replace);
        let addr = heap as usize;
        let old = HeapUma { size: 0x1_0000, align: 0x1_0000, prefer_coherent: false, cpu_readback: false, block: None };
        HEAP_UMA.lock().unwrap().get_or_insert_with(HashMap::new).insert(addr, old);
        unsafe {
            assert_eq!(mem_heap_release(heap.cast()), 0);
            drop(Box::from_raw(heap));
        }
        let kept = HEAP_UMA.lock().unwrap().as_mut().and_then(|heaps| heaps.remove(&addr));
        assert_eq!(kept.map(|heap| heap.size), Some(0x2_0000), "the final Release must not remove replacement bookkeeping");
    }
}

