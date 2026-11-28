//! The console command list's own methods.
//!
//! The console's ID3D12GraphicsCommandList keeps the PC methods (some moved:
//! `real::console_remap`) and holds console extensions from #60 on (as observed on Fortnite):
//! CopyMemoryToMemoryX #60, FillMemoryWith{32,64,128}BitValueX #61–#63, depth bounds #68,
//! Write{32,64}BitValueBottomOfPipeX #69 / #70, Wait{32,64}BitValueX #71 / #72,
//! DispatchX #78, DrawIndexedX #81, CopyResourceX #84, SetPredicationBufferX #85,
//! Write{32,64}BitValueTopOfPipeX #88 / #89, Write32BitValueBottomOfShaderX #90,
//! KickoffX #93, RSSetDepthBiasX #94. A title call of any other extension (GDS, hints…)
//! is logged and does nothing (`real::class_for`).
//!
//! Occlusion queries may nest on the console; the PC allows one active per list. When one
//! begins while another is active, the outer one ends there on the PC and resolves as visible;
//! the inner one is a real query (Fortnite nests its per-object queries in a dummy outer one).
//!
//! Console GPU addresses in unified memory are the title's own addresses; memory operations
//! go to that window's buffer (`real::uma_window_buffer_gpu`), which is in COMMON between them.
//! An index buffer at such an address is given to the PC as the window buffer's GPU address.
//! An address in a GPU view of the title's memory (a view window) has its tiles asked for first
//! (`view_want`); what the lists write there is read back to its CPU view, and the CPU reads its
//! values there (`cpu_at`): the CPU cannot touch the GPU view.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::mem::{transmute_copy, ManuallyDrop};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::device_x::{self, DepthBias};
use super::real;

const PC_GET_TYPE: usize = 8;
const PC_CLOSE: usize = 9;
const PC_RESET: usize = 10;
const PC_COPY_BUFFER_REGION: usize = 15;
const PC_COPY_TEXTURE_REGION: usize = 16;
const PC_COPY_RESOURCE: usize = 17;
const PC_SET_PIPELINE_STATE: usize = 25;
const PC_RESOURCE_BARRIER: usize = 26;
const PC_IA_SET_INDEX_BUFFER: usize = 43;
const PC_BEGIN_QUERY: usize = 52;
const PC_END_QUERY: usize = 53;
const PC_RESOLVE_QUERY_DATA: usize = 54;
const PC_SET_PREDICATION: usize = 55;
const PC_OM_SET_DEPTH_BOUNDS: usize = 62;
const PC_WRITE_BUFFER_IMMEDIATE: usize = 66;
const PC_QUEUE_EXECUTE_COMMAND_LISTS: usize = 10;

/// Write64BitValueBottomOfPipeX flags 1 / 2, Write64BitValueTopOfPipeX flags 4 / 8: the value
/// written is a GPU / system timestamp
const BOTTOM_TIMESTAMPS: u32 = 0x3;
const TOP_TIMESTAMPS: u32 = 0xC;
/// fills larger than this are not done (one pattern buffer per value, up to this size)
const MAX_FILL: u64 = 256 << 20;
/// Bounded wait timeout for Wait32BitValueX / Wait64BitValueX so an unsatisfied wait never hangs.
/// Overridable via `XCL_WAIT_TIMEOUT_MS` for diagnostics.
const WAIT_TIMEOUT: Duration = Duration::from_secs(2);
/// Watchdog timeout for GPU `queue.Wait` so a genuinely hung GPU cycle never triggers a 30 s
/// hang, while giving heavy initial 3D frames (>1 s of dispatches) plenty of room to finish
/// cleanly without prematurely breaking synchronization. Overridable via `XCL_GPU_WAIT_TIMEOUT_MS`
/// for diagnostics.
const GPU_WAIT_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(5000);
fn wait_timeout() -> Duration {
    static T: OnceLock<Duration> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("XCL_WAIT_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .map(Duration::from_millis)
            .unwrap_or(WAIT_TIMEOUT)
    })
}

fn gpu_wait_watchdog_timeout() -> Duration {
    static T: OnceLock<Duration> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("XCL_GPU_WAIT_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .map(Duration::from_millis)
            .unwrap_or(GPU_WAIT_WATCHDOG_TIMEOUT)
    })
}

fn once(counter: &AtomicU64, n: u64) -> bool {
    counter.fetch_add(1, Ordering::Relaxed) < n
}

/// a PC method of the list (its runtime's own)
unsafe fn pc<F: Copy>(list: *mut c_void, slot: usize) -> F {
    transmute_copy(&real::orig_fn(list, slot))
}

unsafe fn list_type(list: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
    pc::<unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE>(list, PC_GET_TYPE)(list)
}

/// The type of a list XCL keeps state for (`on_create`), without a call to the list; None for
/// any other
pub(super) fn tracked_list_type(list: *mut c_void) -> Option<D3D12_COMMAND_LIST_TYPE> {
    LISTS.lock().unwrap().as_ref()?.get(&(list as usize))?.list_type
}

/// A compute pipeline (a built one) on `list`, past the hook of a title's SetPipelineState.
pub(super) unsafe fn set_compute_pipeline(list: *mut c_void, pso: *mut c_void) {
    pc::<unsafe extern "system" fn(*mut c_void, *mut c_void)>(list, PC_SET_PIPELINE_STATE)(list, pso);
}

pub(super) unsafe fn transition(list: *mut c_void, res: usize, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: transmute_copy(&res),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(list, PC_RESOURCE_BARRIER)(list, 1, &b);
}

/// `op` with the window buffer `res` as a copy destination (back to COMMON after it)
unsafe fn as_copy_dest(list: *mut c_void, res: usize, op: impl FnOnce()) {
    transition(list, res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    op();
    transition(list, res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
}

unsafe fn copy_buffer_region(list: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    pc::<unsafe extern "system" fn(*mut c_void, usize, u64, usize, u64, u64)>(list, PC_COPY_BUFFER_REGION)(list, dst, dst_off, src, src_off, n);
}

fn warn(counter: &AtomicU64, what: &str, why: std::fmt::Arguments) {
    if once(counter, 4) {
        crate::xlog_warn!("D3D12.X: {what}: {why} – not done");
    }
}

// ---------------------------------------------------------------------------------------
// per-list state & live queue tracking (KickoffX, Wait32/64BitValueX, RSSetDepthBiasX,
// index buffer restoration, nested occlusion queries)
// ---------------------------------------------------------------------------------------

/// a query: (heap, index)
type Query = (usize, u32);

/// Number of command lists currently inside an active occlusion query (zero-cost fast-path guard
/// for `note_dropped_draw` on the hot draw path).
static ACTIVE_OCCLUSION_LISTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// One list's occlusion queries: the one active on the PC (with its type and whether any draw
/// inside it was skipped or rendered with a pending stand-in PSO) and the outer ones it ended early.
#[derive(Default)]
struct Occlusion {
    active: Option<(Query, D3D12_QUERY_TYPE, bool)>,
    ended_early: Vec<Query>,
}

impl Drop for Occlusion {
    fn drop(&mut self) {
        if self.active.is_some() {
            ACTIVE_OCCLUSION_LISTS.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Occlusion {
    /// Begin `q` on the PC; Some(outer): end that one on the PC first.
    fn begin(&mut self, q: Query, ty: D3D12_QUERY_TYPE) -> Option<(Query, D3D12_QUERY_TYPE)> {
        let outer = self.active.replace((q, ty, false));
        match outer {
            None => {
                ACTIVE_OCCLUSION_LISTS.fetch_add(1, Ordering::Relaxed);
                None
            }
            Some((o, oty, _)) => {
                self.ended_early.push(o);
                Some((o, oty))
            }
        }
    }

    /// End the currently active PC query early (e.g. before a `KickoffX` split), so `EndQuery`
    /// later resolves it as visible.
    fn end_active_early(&mut self) -> Option<(Query, D3D12_QUERY_TYPE)> {
        let (q, ty, _) = self.active.take()?;
        ACTIVE_OCCLUSION_LISTS.fetch_sub(1, Ordering::Relaxed);
        self.ended_early.push(q);
        Some((q, ty))
    }

    /// Record that a draw inside the currently active occlusion query was skipped or used a
    /// pending stand-in PSO, so the query must resolve as visible rather than falsely culling.
    fn mark_dropped_draw(&mut self) {
        if let Some(a) = &mut self.active {
            a.2 = true;
        }
    }

    /// Returns `(forward_pc_end, force_visible)`:
    /// - `forward_pc_end`: whether the PC list must call `EndQuery` for `q`.
    /// - `force_visible`: whether `ResolveQueryData` must overwrite `q` with a visible count
    ///   (either an outer nested/split query ended early, or an active query that saw a dropped draw).
    fn end(&mut self, q: Query) -> (bool, bool) {
        if self.active.is_some_and(|(a, _, _)| a == q) {
            let (_, _, dropped) = self.active.take().unwrap();
            ACTIVE_OCCLUSION_LISTS.fetch_sub(1, Ordering::Relaxed);
            return (true, dropped);
        }
        match self.ended_early.iter().position(|e| *e == q) {
            Some(i) => {
                self.ended_early.swap_remove(i);
                (false, true)
            }
            None => (true, false),
        }
    }
}

/// Called by `bind` when a graphics draw on `list` is skipped or uses a pending stand-in PSO:
/// if `list` currently has an open occlusion query, force it to resolve as visible so the title
/// does not falsely occlusion-cull world geometry while shaders compile or fail to bind.
pub(super) fn note_dropped_draw(list: *mut c_void) {
    if list.is_null() || ACTIVE_OCCLUSION_LISTS.load(Ordering::Relaxed) == 0 {
        return;
    }
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        s.occlusion.mark_dropped_draw();
    }
}

/// Recorded GPU-side memory wait from `Wait32BitValueX` (#71) or `Wait64BitValueX` (#72).
/// Evaluates `(mem_value & mask) <comparison> value` using `D3D12_COMPARISON_FUNC` (1..=8),
/// matching AMD PM4 `WAIT_REG_MEM` semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingWait {
    /// where the CPU has the value (`cpu_at`: the CPU view of an address in a GPU view)
    va: u64,
    width: u8, // 4 or 8
    comparison: u32,
    value: u64,
    mask: u64,
    /// True if this list already recorded a `Write64BitValue*` before this wait (potential
    /// intra-list fork-join where hoisting a GPU wait before list submission could deadlock).
    after_write64: bool,
}

fn compare_u64(op: u32, lhs: u64, rhs: u64) -> Option<bool> {
    match op {
        1 => Some(false),        // D3D12_COMPARISON_FUNC_NEVER
        2 => Some(lhs < rhs),    // D3D12_COMPARISON_FUNC_LESS
        3 => Some(lhs == rhs),   // D3D12_COMPARISON_FUNC_EQUAL
        4 => Some(lhs <= rhs),   // D3D12_COMPARISON_FUNC_LESS_EQUAL
        5 => Some(lhs > rhs),    // D3D12_COMPARISON_FUNC_GREATER
        6 => Some(lhs != rhs),   // D3D12_COMPARISON_FUNC_NOT_EQUAL
        7 => Some(lhs >= rhs),   // D3D12_COMPARISON_FUNC_GREATER_EQUAL
        8 => Some(true),         // D3D12_COMPARISON_FUNC_ALWAYS
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingWaitObservation {
    Satisfied,
    Pending,
    Unreadable,
    Invalid,
}

impl PendingWait {
    unsafe fn read_value(&self) -> Option<u64> {
        self.read_value_with_uma(crate::gpu::uma::get())
    }

    unsafe fn read_value_with_uma(&self, uma: Option<&crate::gpu::uma::Uma>) -> Option<u64> {
        let width = self.width as usize;
        if !matches!(width, 4 | 8) {
            return None;
        }
        let va = self.va as usize;
        if uma.is_some_and(|uma| uma.in_committed_window(va, width)) {
            // The committed coherent UMA window cannot fault. Keep its hot polling path to one
            // acquire load; ReadProcessMemory is reserved for addresses whose lifetime is not
            // controlled by the runtime.
            return Some(match self.width {
                4 => (*(va as *const AtomicU32)).load(Ordering::Acquire) as u64,
                8 => (*(va as *const AtomicU64)).load(Ordering::Acquire),
                _ => unreachable!(),
            });
        }
        let mut bytes = [0u8; 8];
        if !crate::hle::dstorage::read_own(va, bytes.as_mut_ptr(), width) {
            return None;
        }
        // `ReadProcessMemory` performs the possibly-faulting snapshot in the kernel, so a page
        // decommitted concurrently returns failure instead of faulting this submission thread.
        // Keep the title's writes before every host read that follows a successful snapshot.
        std::sync::atomic::fence(Ordering::Acquire);
        match self.width {
            4 => Some(u32::from_ne_bytes(bytes[..4].try_into().unwrap()) as u64),
            8 => Some(u64::from_ne_bytes(bytes)),
            _ => None,
        }
    }

    unsafe fn observe(&self) -> PendingWaitObservation {
        if !matches!(self.width, 4 | 8) {
            return PendingWaitObservation::Invalid;
        }
        let Some(cur) = self.read_value() else {
            return PendingWaitObservation::Unreadable;
        };
        match compare_u64(self.comparison, cur & self.mask, self.value) {
            Some(true) => PendingWaitObservation::Satisfied,
            Some(false) => PendingWaitObservation::Pending,
            None => PendingWaitObservation::Invalid,
        }
    }

    unsafe fn is_satisfied(&self) -> bool {
        self.observe() == PendingWaitObservation::Satisfied
    }

    unsafe fn wait_until_satisfied(&self, timeout: Duration) -> bool {
        if self.is_satisfied() {
            return true;
        }
        for _ in 0..512 {
            std::hint::spin_loop();
            if self.is_satisfied() {
                return true;
            }
        }
        let start = Instant::now();
        while start.elapsed() < timeout {
            let step = if start.elapsed() < Duration::from_millis(10) {
                Duration::from_micros(100)
            } else {
                Duration::from_millis(1)
            };
            std::thread::sleep(step);
            // a value a list wrote into a sparse window reaches this memory with its copy back
            super::sparse::settle_readbacks();
            if self.is_satisfied() {
                return true;
            }
        }
        self.is_satisfied()
    }
}

/// Recorded `OMSetRenderTargets` state for `KickoffX` restoration and `XCL_TRACE_FRAME`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RenderTargets {
    /// Requested `NumRenderTargetDescriptors` (clamped to D3D12 max 8; 0 if `handles` is empty).
    num_targets: u32,
    /// Saved descriptor handles: 1 base handle when `single != 0` and `num_targets > 0`,
    /// otherwise `num_targets` handles.
    handles: Vec<usize>,
    single: i32,
    dsv: Option<usize>,
}

/// What console code set on a list that a PC Reset clears: `KickoffX` closes, submits and
/// resets the PC list mid-recording, and on the console a kickoff leaves the list's state as it
/// was – so it is set again (descriptor heaps above all: translated shaders index the bound
/// heap, and a draw without one reads garbage descriptors).
#[derive(Default, Clone)]
struct Bindings {
    heaps: Vec<usize>,
    has_sampler_heap: bool,
    shared_view_heap: bool,
    topology: Option<u32>,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<windows::Win32::Foundation::RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    vertex_buffers: std::collections::BTreeMap<u32, D3D12_VERTEX_BUFFER_VIEW>,
    render_targets: Option<RenderTargets>,
    /// slot → the title's (address, bytes, stride) of the vertex buffer (`XCL_TRACE_FRAME`)
    vertex_titles: std::collections::BTreeMap<u32, (u64, u32, u32)>,
    /// the title's (address, bytes, format) of the index buffer (`XCL_TRACE_FRAME`)
    index_title: Option<(u64, u32, i32)>,
    /// the title set the console's rect list topology
    rect_list: bool,
}

/// Whether `list` draws rect lists now, and the title's vertex buffers (slot, (address, bytes, stride)).
pub(super) fn rect_list_state(list: *mut c_void) -> Option<Vec<(u32, (u64, u32, u32), Option<D3D12_VERTEX_BUFFER_VIEW>)>> {
    let g = LISTS.lock().unwrap();
    let b = &g.as_ref()?.get(&(list as usize))?.bindings;
    b.rect_list.then(|| b.vertex_titles.iter().map(|(&k, &v)| (k, v, b.vertex_buffers.get(&k).copied())).collect())
}

/// `pc_va` for other modules.
pub fn pc_va_of(va: u64) -> u64 {
    pc_va(va)
}

/// Where the CPU has the value at GPU address `va`: the CPU view of an address in a GPU view (the
/// title's CPU writes and reads it there; the CPU cannot touch the GPU view), else `va`. Waits
/// read their value there, and a write and a wait meet in one `SyncCell` whichever view each
/// names. One compare for an address outside the view windows.
pub fn cpu_at(va: u64) -> u64 {
    cpu_at_in(super::bind::view_span(), va)
}

/// `cpu_at` with the view windows `span` (`bind::view_span`).
fn cpu_at_in(span: (u64, u64), va: u64) -> u64 {
    if super::bind::in_span(span, va) {
        super::view_want::cpu(va)
    } else {
        va
    }
}

/// A GPU write at `va` reaches the CPU's view only when read back after the submit (`note_write`):
/// a window the GPU has a copy of (`Uma::is_gpu_copy`), or a GPU view, whose tiles are read
/// back to its CPU view (`sparse::readback_after_execute`).
fn read_back(va: u64) -> bool {
    crate::gpu::uma::get().is_some_and(|u| u.is_gpu_copy(va as usize) || u.is_view(va as usize))
}

/// `XCL_TRACE_FRAME`: the fixed-function state a list last set (topology, first viewport and
/// scissor, render targets) as text, and what the bound DSV names (`XCL_DUMP_TARGETS` keeps that).
pub(super) fn state_summary(list: *mut c_void) -> String {
    let mut out = String::new();
    let mut dsv = None;
    with_bindings(list, |b| {
        dsv = b.render_targets.as_ref().and_then(|rt| rt.dsv);
        let vp = b.viewports.first().map(|v| (v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth));
        let sc = b.scissors.first().map(|r| (r.left, r.top, r.right, r.bottom));
        out = format!(
            "topology {:?}, stencil ref {:?}, viewport {vp:?} ({} set), scissor {sc:?} ({} set), render targets {:?}{}",
            b.topology,
            b.stencil,
            b.viewports.len(),
            b.scissors.len(),
            b.render_targets.as_ref().map(|rt| (rt.num_targets as usize, rt.dsv.is_some())),
            b.render_targets.as_ref().and_then(|rt| rt.dsv).map_or(String::new(), |d| format!(", DSV {d:#x}"))
        );
    });
    // (after the bindings' lock: the map has its own; the resource is taken with a reference of
    // its own there, since another CreateDepthStencilView at the handle releases the map's)
    if let Some(x) = dsv.and_then(super::present::rtv_resource).and_then(|r| unsafe { super::present::view_resource(r) }) {
        let d = unsafe { x.GetDesc() };
        // flags: the resource's; the view's (1 read-only depth, 2 read-only stencil) when `depth_uav` keeps the DSV
        out += &format!(" names {:#x} {}x{} PC format {} flags {:#x}", x.as_raw() as usize, d.Width, d.Height, d.Format.0, d.Flags.0);
        if let Some(f) = dsv.and_then(super::depth_uav::dsv_flags) {
            out += &format!(", view flags {f:#x}");
        }
    }
    out
}

/// `XCL_TRACE_FRAME`: the vertex and index buffers a list binds, with their first vertices
/// and indices as the CPU sees the title's memory.
pub(super) unsafe fn geometry_summary(list: *mut c_void, draw: (bool, u32, u32, u32, i32)) -> String {
    let (vbs, ib) = {
        let mut out = (Vec::new(), None);
        with_bindings(list, |b| out = (b.vertex_titles.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>(), b.index_title));
        out
    };
    let words = |a: u64, n: usize| -> Option<Vec<u32>> {
        (a != 0 && crate::gpu::shader::readable(a as usize, 4 * n)).then(|| (0..n).map(|k| ((a as usize + 4 * k) as *const u32).read_unaligned()).collect())
    };
    let mut out = Vec::new();
    for (slot, (a, size, stride)) in vbs {
        let per = (stride as usize / 4).clamp(1, 12);
        let verts: Vec<String> = (0..4u64).filter_map(|i| words(a + i * stride as u64, per).map(|w| format!("{w:x?}"))).collect();
        out.push(format!("VB{slot} {a:#x} ({size} bytes, stride {stride}): {}", verts.join(" ")));
    }
    // the vertices the draw names: through its indices (indexed) or from its first vertex
    let (indexed, count, _, first, base) = draw;
    let n = count.min(6) as usize;
    let names: Option<Vec<u64>> = if indexed {
        ib.and_then(|(a, _, fmt)| {
            let w = if fmt == 57 { 2 } else { 4 };
            let at = a as usize + w * first as usize;
            (a != 0 && crate::gpu::shader::readable(at, w * n)).then(|| {
                (0..n).map(|k| if w == 2 { ((at + 2 * k) as *const u16).read_unaligned() as u64 } else { ((at + 4 * k) as *const u32).read_unaligned() as u64 }).map(|i| (i as i64 + base as i64) as u64).collect()
            })
        })
    } else {
        Some((0..n as u64).map(|k| first as u64 + k).collect())
    };
    out.push(format!("IB {:x?}; the draw's vertices (as indexed now) {names:?}", ib.map(|(a, size, fmt)| (a, size, fmt))));
    // the title fills ring buffers after recording a draw: the vertices are read again when
    // the list is executed (`dump_pending_geometry`)
    if let Some((_, vb)) = vbs_first(list) {
        PENDING_GEOMETRY.lock().unwrap().push((list as usize, vb, ib, draw));
    }
    out.join("; ")
}

/// draws whose vertices are dumped when their list runs: (list, (VB address, bytes, stride),
/// (IB address, bytes, format), draw arguments)
#[allow(clippy::type_complexity)]
static PENDING_GEOMETRY: Mutex<Vec<(usize, (u64, u32, u32), Option<(u64, u32, i32)>, (bool, u32, u32, u32, i32))>> = Mutex::new(Vec::new());

/// `XCL_TRACE_FRAME`: the vertices of traced draws as the title's memory holds them at the
/// next present (the CPU has written the frame's vertices by then).
pub(super) unsafe fn dump_pending_geometry() {
    let mine: Vec<_> = std::mem::take(&mut *PENDING_GEOMETRY.lock().unwrap());
    for (_, (a, _, stride), ib, (indexed, count, _, first, base)) in mine {
        let n = count.min(48) as usize;
        let names: Vec<u64> = match (indexed, ib) {
            (true, Some((ia, _, fmt))) => {
                let w = if fmt == 57 { 2 } else { 4 };
                let at = ia as usize + w * first as usize;
                if !crate::gpu::shader::readable(at, w * n) {
                    continue;
                }
                (0..n).map(|k| if w == 2 { ((at + 2 * k) as *const u16).read_unaligned() as u64 } else { ((at + 4 * k) as *const u32).read_unaligned() as u64 }).map(|i| (i as i64 + base as i64) as u64).collect()
            }
            _ => (0..n as u64).map(|k| first as u64 + k).collect(),
        };
        let per = (stride as usize / 4).clamp(1, 12);
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        let verts: Vec<String> = unique
            .iter()
            .take(24)
            .map(|&i| {
                let at = (a + i * stride as u64) as usize;
                let w: Option<Vec<u32>> = crate::gpu::shader::readable(at, 4 * per).then(|| (0..per).map(|k| ((at + 4 * k) as *const u32).read_unaligned()).collect());
                format!("v{i}: {w:x?}")
            })
            .collect();
        crate::xlog_info!("frame trace: … its vertices at the present (VB {a:#x}, stride {stride}, indices {names:?}): {}", verts.join("; "));
    }
}

fn vbs_first(list: *mut c_void) -> Option<(u32, (u64, u32, u32))> {
    let mut first = None;
    with_bindings(list, |b| first = b.vertex_titles.iter().next().map(|(&k, &v)| (k, v)));
    first
}

/// The depth-stencil view handle a list last bound (`XCL_TRACE_FRAME`).
pub(super) fn bound_dsv(list: *mut c_void) -> Option<usize> {
    let mut d = None;
    with_bindings(list, |b| d = b.render_targets.as_ref().and_then(|rt| rt.dsv));
    d
}

/// depth-stencil views bound by draws that read a watched target: their clears are traced
static WATCHED_DSVS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub(super) fn watch_dsv(dsv: usize) {
    let mut g = WATCHED_DSVS.lock().unwrap();
    if !g.contains(&dsv) && g.len() < 8 {
        g.push(dsv);
    }
}

fn with_bindings(list: *mut c_void, f: impl FnOnce(&mut Bindings)) {
    let mut g = LISTS.lock().unwrap();
    f(&mut g.get_or_insert_with(HashMap::new).entry(list as usize).or_default().bindings);
}

/// (tests) The view heaps XCL counts as bound on `list`; forgetting the list with `forget`.
#[cfg(test)]
pub(super) fn bound_heaps(list: *mut c_void, forget: bool) -> Vec<usize> {
    let mut heaps = Vec::new();
    with_bindings(list, |b| heaps = b.heaps.clone());
    if forget {
        LISTS.lock().unwrap().as_mut().map(|m| m.remove(&(list as usize)));
    }
    heaps
}

const PC_IA_SET_PRIMITIVE_TOPOLOGY: usize = 20;
const PC_RS_SET_VIEWPORTS: usize = 21;
const PC_RS_SET_SCISSOR_RECTS: usize = 22;
const PC_OM_SET_BLEND_FACTOR: usize = 23;
const PC_OM_SET_STENCIL_REF: usize = 24;
const PC_OM_SET_RENDER_TARGETS: usize = 46;

/// console #20 IASetPrimitiveTopology (PC #20)
/// The console's rect list (AMD's DI_PT_RECTLIST, 0x11): three vertices of a rectangle, the
/// fourth implied. The PC has no such topology: the draw is a triangle list, and a translated
/// pipeline of topology type 19 completes each rect in a geometry shader (`pso::rect_list_gs`);
/// where it has none (refused, see `real::rect_list_gs`) each rect is drawn as the triangle of
/// its three vertices (a retail game's full-screen passes).
pub(super) const RECT_LIST: u32 = 17;

pub(super) unsafe extern "system" fn ia_set_topology_x(this: *mut c_void, t: u32) {
    // a topology the PC does not have (console ones): logged once each
    if !matches!(t, 0..=5 | 10..=13 | 33..=64) {
        static SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
        let new = {
            let mut g = SEEN.lock().unwrap();
            let n = !g.contains(&t);
            if n {
                g.push(t);
            }
            n
        };
        if new {
            crate::xlog_warn!("D3D12.X: IASetPrimitiveTopology({t}) – a console topology the PC does not have{}", if t == RECT_LIST { ": rect lists are drawn as triangle lists, completed by a GS in rect list pipelines" } else { "" });
        }
    }
    if super::bind::queue_topology(this, t) {
        return;
    }
    apply_topology(this, t);
}

pub(super) unsafe fn apply_topology(this: *mut c_void, t: u32) {
    let rect = t == RECT_LIST;
    let t = if t == RECT_LIST { windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0 as u32 } else { t };
    with_bindings(this, |b| {
        b.topology = Some(t);
        b.rect_list = rect;
    });
    pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_IA_SET_PRIMITIVE_TOPOLOGY)(this, t)
}

/// console #21 RSSetViewports (PC #21)
pub(super) unsafe extern "system" fn rs_set_viewports_x(this: *mut c_void, n: u32, v: *const D3D12_VIEWPORT) {
    let _c = super::capture::op(this, || {
        let vs = if v.is_null() { &[][..] } else { std::slice::from_raw_parts(v, n as usize) };
        Some(super::capture::Op::Viewports(vs.iter().map(|v| [v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth]).collect()))
    });
    if !v.is_null() {
        let vs = std::slice::from_raw_parts(v, n as usize).to_vec();
        with_bindings(this, |b| b.viewports = vs);
    }
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT)>(this, PC_RS_SET_VIEWPORTS)(this, n, v)
}

/// console #22 RSSetScissorRects (PC #22)
pub(super) unsafe extern "system" fn rs_set_scissors_x(this: *mut c_void, n: u32, r: *const windows::Win32::Foundation::RECT) {
    let _c = super::capture::op(this, || {
        let rs = if r.is_null() { &[][..] } else { std::slice::from_raw_parts(r, n as usize) };
        Some(super::capture::Op::Scissors(rs.iter().map(|r| [r.left, r.top, r.right, r.bottom]).collect()))
    });
    if !r.is_null() && n > 0 {
        let rs = std::slice::from_raw_parts(r, n as usize);
        let rects: Vec<[i32; 4]> = rs.iter().map(|r| [r.left, r.top, r.right, r.bottom]).collect();
        if super::bind::queue_scissors(this, rects) {
            return;
        }
        with_bindings(this, |b| b.scissors = rs.to_vec());
    } else if r.is_null() || n == 0 {
        if super::bind::queue_scissors(this, Vec::new()) {
            return;
        }
    }
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const windows::Win32::Foundation::RECT)>(this, PC_RS_SET_SCISSOR_RECTS)(this, n, r)
}

pub(super) unsafe fn apply_scissors(this: *mut c_void, rects: &[[i32; 4]]) {
    let rs: Vec<windows::Win32::Foundation::RECT> = rects
        .iter()
        .map(|&[left, top, right, bottom]| windows::Win32::Foundation::RECT { left, top, right, bottom })
        .collect();
    if !rs.is_empty() {
        with_bindings(this, |b| b.scissors = rs.clone());
    }
    let ptr = if rs.is_empty() { std::ptr::null() } else { rs.as_ptr() };
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const windows::Win32::Foundation::RECT)>(this, PC_RS_SET_SCISSOR_RECTS)(this, rs.len() as u32, ptr)
}

/// console #23 OMSetBlendFactor (PC #23)
pub(super) unsafe extern "system" fn om_set_blend_factor_x(this: *mut c_void, f: *const [f32; 4]) {
    let v = f.as_ref().copied();
    with_bindings(this, |b| b.blend = v);
    pc::<unsafe extern "system" fn(*mut c_void, *const [f32; 4])>(this, PC_OM_SET_BLEND_FACTOR)(this, f)
}

/// console #24 OMSetStencilRef (PC #24)
pub(super) unsafe extern "system" fn om_set_stencil_ref_x(this: *mut c_void, r: u32) {
    if super::bind::queue_stencil_ref(this, r) {
        return;
    }
    apply_stencil_ref(this, r);
}

pub(super) unsafe fn apply_stencil_ref(this: *mut c_void, r: u32) {
    with_bindings(this, |b| b.stencil = Some(r));
    pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_OM_SET_STENCIL_REF)(this, r)
}

/// OMSetRenderTargets as the PC list got it (`real::cl_om_set_render_targets`)
pub(super) unsafe fn note_render_targets(this: *mut c_void, n: u32, rtvs: *const usize, single: i32, dsv: *const usize) {
    let num_targets = if rtvs.is_null() { 0 } else { n.min(D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT) };
    let read_count = if single != 0 { num_targets.min(1) } else { num_targets } as usize;
    // SAFETY: `rtvs` is non-null whenever `read_count > 0`, and D3D12 requires at least 1
    // handle when `RTsSingleHandleToDescriptorRange != 0` or `num_targets` handles otherwise.
    let handles = if read_count == 0 { Vec::new() } else { std::slice::from_raw_parts(rtvs, read_count).to_vec() };
    let d = dsv.as_ref().copied();
    with_bindings(this, |b| b.render_targets = Some(RenderTargets { num_targets, handles, single, dsv: d }));
}

/// The render target views bound on `list` (each one of a single descriptor range too).
pub(super) unsafe fn bound_render_targets(list: *mut c_void) -> Vec<usize> {
    static INC: AtomicU32 = AtomicU32::new(0);
    let Some(rt) = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).and_then(|s| s.bindings.render_targets.clone()) else {
        return Vec::new();
    };
    if rt.single == 0 || rt.num_targets <= 1 {
        return rt.handles;
    }
    let Some(&base) = rt.handles.first() else { return Vec::new() };
    let mut inc = INC.load(Ordering::Relaxed) as usize;
    if inc == 0 {
        inc = device_of(list).map_or(0, |d| d.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize);
        INC.store(inc as u32, Ordering::Relaxed);
    }
    (0..rt.num_targets as usize).map(|i| base + i * inc).collect()
}

/// Set again what `b` holds (after KickoffX's Reset or a deferred mid-list split).
unsafe fn restore_bindings(this: *mut c_void, b: &Bindings, ty: D3D12_COMMAND_LIST_TYPE) {
    if ty == D3D12_COMMAND_LIST_TYPE_COPY {
        return;
    }
    if !b.heaps.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(this, 28)(this, b.heaps.len() as u32, b.heaps.as_ptr());
    }
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT {
        return;
    }
    if let Some(t) = b.topology {
        pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_IA_SET_PRIMITIVE_TOPOLOGY)(this, t);
    }
    if !b.viewports.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT)>(this, PC_RS_SET_VIEWPORTS)(this, b.viewports.len() as u32, b.viewports.as_ptr());
    }
    if !b.scissors.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const windows::Win32::Foundation::RECT)>(this, PC_RS_SET_SCISSOR_RECTS)(this, b.scissors.len() as u32, b.scissors.as_ptr());
    }
    if let Some(f) = b.blend {
        pc::<unsafe extern "system" fn(*mut c_void, *const [f32; 4])>(this, PC_OM_SET_BLEND_FACTOR)(this, &f);
    }
    if let Some(r) = b.stencil {
        pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_OM_SET_STENCIL_REF)(this, r);
    }
    for (&slot, v) in &b.vertex_buffers {
        pc::<unsafe extern "system" fn(*mut c_void, u32, u32, *const D3D12_VERTEX_BUFFER_VIEW)>(this, PC_IA_SET_VERTEX_BUFFERS)(this, slot, 1, v);
    }
    if let Some(rt) = &b.render_targets {
        let d = rt.dsv.unwrap_or(0);
        let (n, rts_ptr) = if rt.handles.is_empty() { (0, std::ptr::null()) } else { (rt.num_targets, rt.handles.as_ptr()) };
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize)>(this, PC_OM_SET_RENDER_TARGETS)(
            this,
            n,
            rts_ptr,
            rt.single,
            if rt.dsv.is_some() { &d } else { std::ptr::null() },
        );
    }
}

/// Canonical half-open UMA ranges written by one command-list recording.
///
/// Keys are starts and values are exclusive ends. Overlapping and adjacent ranges are merged,
/// so lookup examines at most the predecessor and successor around the requested address.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RecordedWrites {
    ranges: BTreeMap<u64, u64>,
}

impl RecordedWrites {
    fn insert(&mut self, va: u64, size: u64) {
        if size == 0 {
            return;
        }
        let mut start = va;
        let mut end = va.saturating_add(size);
        if end <= start {
            return;
        }

        if let Some((&previous_start, &previous_end)) = self.ranges.range(..=start).next_back() {
            if previous_end >= start {
                start = previous_start;
                end = end.max(previous_end);
                self.ranges.remove(&previous_start);
            }
        }
        loop {
            let next = self.ranges.range(start..).next().map(|(&s, &e)| (s, e));
            match next {
                Some((next_start, next_end)) if next_start <= end => {
                    end = end.max(next_end);
                    self.ranges.remove(&next_start);
                }
                _ => break,
            }
        }
        self.ranges.insert(start, end);
    }

    fn overlaps(&self, va: u64, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let end = va.saturating_add(size);
        if end <= va {
            return false;
        }
        if self.ranges.range(..=va).next_back().is_some_and(|(_, &range_end)| range_end > va) {
            return true;
        }
        self.ranges.range(va..).next().is_some_and(|(&range_start, _)| range_start < end)
    }

    fn clear(&mut self) {
        self.ranges.clear();
    }

    fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn extend(&mut self, other: &Self) {
        for (&start, &end) in &other.ranges {
            self.insert(start, end - start);
        }
    }
}

/// When a recorded write becomes visible to later work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WriteProducer {
    InSegmentGpu,
    CpuAfterSegment,
}

/// Exact written ranges kept separately for each visibility class.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RecordedWriteProducers {
    in_segment_gpu: RecordedWrites,
    cpu_after_segment: RecordedWrites,
}

impl RecordedWriteProducers {
    fn ranges(&self, producer: WriteProducer) -> &RecordedWrites {
        match producer {
            WriteProducer::InSegmentGpu => &self.in_segment_gpu,
            WriteProducer::CpuAfterSegment => &self.cpu_after_segment,
        }
    }

    fn ranges_mut(&mut self, producer: WriteProducer) -> &mut RecordedWrites {
        match producer {
            WriteProducer::InSegmentGpu => &mut self.in_segment_gpu,
            WriteProducer::CpuAfterSegment => &mut self.cpu_after_segment,
        }
    }

    fn insert(&mut self, producer: WriteProducer, va: u64, size: u64) {
        self.ranges_mut(producer).insert(va, size);
    }

    fn overlaps(&self, producer: WriteProducer, va: u64, size: u64) -> bool {
        self.ranges(producer).overlaps(va, size)
    }

    fn clear(&mut self) {
        self.in_segment_gpu.clear();
        self.cpu_after_segment.clear();
    }

    fn is_empty(&self) -> bool {
        self.in_segment_gpu.is_empty() && self.cpu_after_segment.is_empty()
    }
}

#[derive(Default)]
struct ListState {
    allocator: usize,
    list_type: Option<D3D12_COMMAND_LIST_TYPE>,
    last_queue: Option<usize>,
    /// Title's currently bound PSO (before dynamic depth-bias override)
    bound_pso: usize,
    /// An unavailable non-evictable SetPipelineState made this recording unsafe to submit.
    poisoned: bool,
    /// Recipe-backed native generations referenced by this physical recording.
    native_generations: Vec<super::pso_async::NativeGeneration>,
    /// Allocation identities parallel to `native_generations`, for constant-time deduplication.
    native_generation_identities: HashSet<usize>,
    /// Active `RSSetDepthBiasX` override (None = zero / disabled)
    depth_bias: Option<DepthBias>,
    /// Last depth bias sent to `ID3D12GraphicsCommandList9::RSSetDepthBias` on this PC list
    /// since CreateCommandList / Reset / ClearState / KickoffX (resets to ZERO on each).
    native_bias: DepthBias,
    /// True when the PC list's bound PSO needs patching via the fallback cache before the next draw
    pso_dirty: bool,
    /// True when a fallback depth-bias derived PSO is currently bound on the PC list
    using_fallback_pso: bool,
    /// Bound index buffer (PC address), restored after `DrawIndexedX` and `KickoffX`
    index_buffer: Option<D3D12_INDEX_BUFFER_VIEW>,
    /// Last `OMSetDepthBounds` values, restored after `KickoffX`
    depth_bounds: Option<(f32, f32)>,
    occlusion: Occlusion,
    /// Address ranges written by this list since last Reset/KickoffX, by visibility class.
    written_ranges: RecordedWriteProducers,
    /// Writes produced by this physical segment only (not cumulative across a split).
    segment_write_producers: RecordedWriteProducers,
    /// Sparse-window `(va, bytes)` ranges written by memory ops (`Write32/64`, `Fill*`,
    /// `CopyMemoryToMemoryX`, timestamps) to copy back to the CPU view after execution.
    sparse_writes: Vec<(usize, usize)>,
    /// Writes into title textures produced by this physical segment.
    texture_writes: Vec<super::title_textures::GpuWrite>,
    /// Ordered state transitions of tracked textures in this physical segment.
    texture_transitions: Vec<super::title_textures::Transition>,
    /// A translated draw/dispatch in this physical segment successfully bound the shared view heap.
    uses_shared_view_heap: bool,
    /// Bottom-of-pipe immediates into a window the GPU has a copy of: (address, value, bytes).
    /// XCL knows what the GPU writes there, so the CPU's view gets the value from XCL once the
    /// submission is done on the GPU (`sparse::readback_after_execute`), without a copy back.
    sparse_imm: Vec<(usize, u64, u8)>,
    /// Non-timestamp 64-bit `(va, value)` writes recorded by `Write64BitValue{Bottom,Top}OfPipeX`
    /// so `ExecuteCommandLists` / `KickoffX` can signal per-address GPU fences for cross-queue
    /// `Wait64BitValueX` without blocking the submitter thread on the CPU.
    write64_signals: Vec<(u64, u64)>,
    /// Causal dump token for this exact recording; absent unless scheduled dumps enabled tracing.
    write64_trace: Option<Arc<Mutex<ListTrace>>>,
    /// External memory waits recorded on this list
    pending_waits: Vec<PendingWait>,
    /// bottom-of-pipe writes into memory the GPU does not see (`cpu_writes`)
    cpu_writes: Vec<super::cpu_writes::Write>,
    /// Dedicated 256-byte buffer holding the current predicate value (created lazily in COMMON state).
    pred_buf: Option<ID3D12Resource>,
    /// True while `pred_buf` is in `D3D12_RESOURCE_STATE_PREDICATION` on this open recording.
    pred_in_state: bool,
    /// Active `SetPredicationBufferX` op, restored across `KickoffX`.
    predication: Option<D3D12_PREDICATION_OP>,
    /// state a PC Reset clears (restored after KickoffX)
    bindings: Bindings,
    /// what this segment records at the submit (`defer`)
    deferred: Option<Deferred>,
    /// on a title list's first segment: the allocator its deferred segments record on (the title's may be recording another list by the submit)
    own_alloc: Option<ID3D12CommandAllocator>,
    /// depth buffers whose depth plane and twin hold the same on this recording (`depth_uav::plan_barriers`)
    depth_in_step: Vec<u64>,
}

/// A `Dispatch` or a dispatching `ExecuteIndirect` put off to the submit, with its arguments.
pub(super) enum Deferred {
    /// x, y, z and `DispatchX`'s flags (0 for `Dispatch`)
    Dispatch(u32, u32, u32, u32),
    /// Raw signature/resources rely on the existing D3D12 caller lifetime requirement through
    /// execution; this stage retains only the immutable classification snapshot.
    Indirect {
        sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64,
        signature: super::bind::IndirectSignatureInfo,
    },
}

/// Why a recording needs a physical command-list boundary.
pub(super) enum SegmentReason {
    WaitAfterWrite64,
    WaitAfterCpuWrite,
    DeferredDispatch(Deferred),
}

impl SegmentReason {
    fn extra_segments(&self) -> usize {
        match self {
            Self::WaitAfterWrite64 | Self::WaitAfterCpuWrite => 1,
            Self::DeferredDispatch(_) => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WaitPlacement {
    OrderedInSegment,
    RecordHere,
    SplitAfterCpuWrite,
    SplitAfterWrite64,
}

impl ListState {
    /// Keep one owner of an exact native generation in this recording. Returns a redundant
    /// owner so callers holding `LISTS` can destroy it after releasing the registry lock.
    fn pin_native_generation(
        &mut self,
        generation: super::pso_async::NativeGeneration,
    ) -> Option<super::pso_async::NativeGeneration> {
        if self.native_generations.last().is_some_and(|last| last.same(&generation)) {
            return Some(generation);
        }
        if !self.native_generation_identities.insert(generation.identity()) {
            return Some(generation);
        }
        self.native_generations.push(generation);
        None
    }

    fn take_native_generations(&mut self) -> Vec<super::pso_async::NativeGeneration> {
        self.native_generation_identities.clear();
        std::mem::take(&mut self.native_generations)
    }

    fn note_write(&mut self, producer: WriteProducer, va: u64, size: u64) {
        if size == 0 {
            return;
        }
        self.written_ranges.insert(producer, va, size);
        self.segment_write_producers.insert(producer, va, size);
        if producer == WriteProducer::InSegmentGpu && read_back(va) {
            if !self.has_sparse() {
                SPARSE_WRITE_LISTS.fetch_add(1, Ordering::AcqRel);
            }
            self.sparse_writes.push((va as usize, size as usize));
            super::sparse::will_write(va as usize, size as usize);
        }
    }

    fn has_sparse(&self) -> bool {
        !self.sparse_writes.is_empty() || !self.sparse_imm.is_empty()
    }

    /// `note_write` for an immediate of `bytes` (4 or 8) into a window the GPU has a copy of
    fn note_imm(&mut self, va: u64, value: u64, bytes: u8) {
        // (a GPU write of the segment: later waits in it are ordered after it, as `note_write`'s)
        let size = (bytes & 0x7f) as u64;
        self.written_ranges.insert(WriteProducer::InSegmentGpu, va, size);
        self.segment_write_producers.insert(WriteProducer::InSegmentGpu, va, size);
        if !self.has_sparse() {
            SPARSE_WRITE_LISTS.fetch_add(1, Ordering::AcqRel);
        }
        self.sparse_imm.push((va as usize, value, bytes));
        super::sparse::will_write(va as usize, (bytes & 0x7f) as usize);
    }

    fn clear_sparse_writes(&mut self) -> (Vec<(usize, usize)>, Vec<(usize, u64, u8)>) {
        if !self.has_sparse() {
            return (Vec::new(), Vec::new());
        }
        SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
        (std::mem::take(&mut self.sparse_writes), std::mem::take(&mut self.sparse_imm))
    }

    fn push_write64_signal(&mut self, va: u64, value: u64) {
        if self.write64_signals.is_empty() {
            WRITE64_SIGNAL_LISTS.fetch_add(1, Ordering::AcqRel);
        }
        self.write64_signals.push((va, value));
    }

    fn clear_write64_signals(&mut self) -> Vec<(u64, u64)> {
        if self.write64_signals.is_empty() {
            return Vec::new();
        }
        WRITE64_SIGNAL_LISTS.fetch_sub(1, Ordering::AcqRel);
        std::mem::take(&mut self.write64_signals)
    }

    fn has_prior_write(&self, va: u64, size: u64) -> bool {
        // GPU writes are already ordered before later commands in the same physical list.
        self.written_ranges.overlaps(WriteProducer::InSegmentGpu, va, size)
    }

    fn wait_placement(&self, va: u64, size: u64) -> WaitPlacement {
        if self.written_ranges.overlaps(WriteProducer::CpuAfterSegment, va, size) {
            WaitPlacement::SplitAfterCpuWrite
        } else if self.has_prior_write(va, size) {
            WaitPlacement::OrderedInSegment
        } else if !self.write64_signals.is_empty() {
            WaitPlacement::SplitAfterWrite64
        } else {
            WaitPlacement::RecordHere
        }
    }

    fn clear_waits(&mut self) {
        self.pending_waits.clear();
    }

    fn push_wait(&mut self, w: PendingWait) {
        self.pending_waits.push(w);
    }
}

/// Record that a successfully bound translated draw/dispatch can address the shared view heap.
/// Callers gate this with `srv_tracking_enabled` and the heap-binding result.
pub(super) fn note_shared_view_heap_use(list: *mut c_void) {
    if list.is_null() {
        return;
    }
    if let Some(state) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        state.uses_shared_view_heap = true;
    }
}

fn propagate_bundle_shared_view_use(states: &mut HashMap<usize, ListState>, parent: usize, bundle: usize) {
    if states.get(&bundle).is_some_and(|state| state.uses_shared_view_heap) {
        states.entry(parent).or_default().uses_shared_view_heap = true;
    }
}

pub(super) fn record_native_generation(list: usize, generation: Option<super::pso_async::NativeGeneration>) {
    if let Some(generation) = generation {
        let redundant = LISTS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(list)
            .or_default()
            .pin_native_generation(generation);
        drop(redundant);
    }
}

/// Opaque fixture for measuring the actual exact-generation recording path without a device.
#[cfg(feature = "pipeline-resolution-bench")]
#[doc(hidden)]
pub struct EmissionBookkeepingBench {
    generations: super::pso_async::NativeGenerationBenchSet,
    depth_lists: [usize; 4],
    worker_lists: Vec<usize>,
    _keys: Vec<Box<u8>>,
}

#[cfg(feature = "pipeline-resolution-bench")]
impl EmissionBookkeepingBench {
    pub fn new(workers: usize) -> Self {
        assert!(workers != 0);
        let generations = super::pso_async::NativeGenerationBenchSet::new(256);
        let keys = (0..4 + workers).map(|index| Box::new(index as u8)).collect::<Vec<_>>();
        let lists = keys.iter().map(|key| (&**key as *const u8) as usize).collect::<Vec<_>>();
        let depth_lists = [lists[0], lists[1], lists[2], lists[3]];
        let worker_lists = lists[4..].to_vec();
        assert!(LISTS
            .lock()
            .unwrap()
            .as_ref()
            .is_none_or(|states| lists.iter().all(|list| !states.contains_key(list))));
        for (slot, depth) in [1usize, 16, 64, 256].into_iter().enumerate() {
            for index in 0..depth {
                record_native_generation(depth_lists[slot], Some(generations.get(index)));
            }
        }
        for (worker, &list) in worker_lists.iter().enumerate() {
            record_native_generation(list, Some(generations.get(worker)));
        }
        Self { generations, depth_lists, worker_lists, _keys: keys }
    }

    /// Depth one hits the last-pin shortcut. Larger depths hit the identity index with an
    /// existing non-last generation.
    #[inline(never)]
    pub fn record_at_depth(&self, slot: usize) {
        record_native_generation(self.depth_lists[slot], Some(self.generations.get(0)));
    }

    #[inline(never)]
    pub fn record_worker(&self, worker: usize) {
        record_native_generation(self.worker_lists[worker], Some(self.generations.get(worker)));
    }

    #[inline(never)]
    pub fn record_workers_serial(&self, workers: usize, batch: usize) {
        for worker in 0..workers {
            for _ in 0..batch {
                self.record_worker(worker);
            }
        }
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
impl Drop for EmissionBookkeepingBench {
    fn drop(&mut self) {
        let removed = {
            let mut lists = LISTS.lock().unwrap();
            let Some(lists) = lists.as_mut() else { return };
            self.depth_lists
                .iter()
                .chain(&self.worker_lists)
                .filter_map(|list| lists.remove(list))
                .collect::<Vec<_>>()
        };
        drop(removed);
    }
}

#[cfg(test)]
pub(super) fn native_generation_count_for_test(list: usize) -> usize {
    LISTS.lock().unwrap().as_ref().and_then(|lists| lists.get(&list)).map_or(0, |state| state.native_generations.len())
}

#[cfg(test)]
pub(super) fn bound_pipeline_for_test(list: usize) -> usize {
    LISTS.lock().unwrap().as_ref().and_then(|lists| lists.get(&list)).map_or(0, |state| state.bound_pso)
}

pub(super) fn poison_recording(list: usize) {
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(list).or_default().poisoned = true;
    POISONED_EVER.store(true, Ordering::Release);
}

pub(super) fn recording_poisoned(list: usize) -> bool {
    if !POISONED_EVER.load(Ordering::Acquire) {
        return false;
    }
    LISTS.lock().unwrap().as_ref().and_then(|lists| lists.get(&list)).is_some_and(|s| s.poisoned)
}

pub(super) fn any_recording_poisoned(lists: &[*mut c_void]) -> bool {
    if !POISONED_EVER.load(Ordering::Acquire) {
        return false;
    }
    let states = LISTS.lock().unwrap();
    states.as_ref().is_some_and(|states| {
        lists.iter().any(|list| states.get(&(*list as usize)).is_some_and(|s| s.poisoned))
    })
}

pub(super) fn resolve_and_record_pipeline(list: usize, pso: usize) -> usize {
    let (resolved, generation) = super::pso_async::resolve_and_pin_if_enabled(pso);
    record_native_generation(list, generation);
    resolved
}

pub(super) fn resolve_and_record_pipeline_lazy(list: usize, pso: usize) -> usize {
    let (resolved, generation) = super::pso_async::resolve_and_pin_if_enabled_lazy(pso);
    record_native_generation(list, generation);
    resolved
}

pub(super) fn resolve_and_record_pipeline_wait(list: usize, pso: usize) -> usize {
    let (resolved, generation) = super::pso_async::resolve_and_pin_if_enabled_wait(pso);
    record_native_generation(list, generation);
    resolved
}

/// Snapshot bundle descriptor and PSO-generation use before the bundle may be reset.
pub(super) fn on_execute_bundle(parent: usize, bundle: usize) {
    let track_descriptors = super::vram::srv_tracking_enabled();
    let mut g = LISTS.lock().unwrap();
    let Some(m) = g.as_mut() else { return };
    let (pins, poisoned) = m.get(&bundle)
        .map(|s| (s.native_generations.clone(), s.poisoned)).unwrap_or_default();
    if poisoned {
        m.entry(parent).or_default().poisoned = true;
    }
    if !track_descriptors && pins.is_empty() { return }
    if track_descriptors {
        propagate_bundle_shared_view_use(m, parent, bundle);
    }
    let parent = m.entry(parent).or_default();
    let redundant = pins.into_iter().filter_map(|pin| parent.pin_native_generation(pin)).collect::<Vec<_>>();
    drop(g);
    drop(redundant);
}

pub(super) unsafe extern "system" fn execute_bundle(this: *mut c_void, bundle: *mut c_void) {
    on_execute_bundle(this as usize, bundle as usize);
    pc::<unsafe extern "system" fn(*mut c_void, *mut c_void)>(this, 27)(this, bundle);
}

/// Record immutable tracked-texture transitions on the physical segment that contains them.
pub(super) fn note_texture_transitions(list: usize, transitions: &[(usize, u32, D3D12_RESOURCE_STATES)]) {
    let captured = super::title_textures::capture_transitions(transitions);
    if captured.is_empty() {
        return;
    }
    if let Some(state) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&list)) {
        state.texture_transitions.extend(captured);
    }
}

/// Discard transitions when a translated physical list is about to be recorded from the start.
pub(super) fn clear_texture_transitions(list: usize) {
    if let Some(state) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&list)) {
        state.texture_transitions.clear();
    }
}

struct KickoffEffects {
    waits: Vec<PendingWait>,
    /// GPU writes and immediates into windows read back after the submit (`ListState::clear_sparse_writes`)
    sparse_writes: (Vec<(usize, usize)>, Vec<(usize, u64, u8)>),
    texture_writes: Vec<super::title_textures::GpuWrite>,
    texture_transitions: Vec<super::title_textures::Transition>,
    uses_shared_view_heap: bool,
    write64_signals: Vec<(u64, u64)>,
    cpu_writes: Vec<super::cpu_writes::Write>,
    native_generations: Vec<super::pso_async::NativeGeneration>,
}

fn take_kickoff_effects(state: &mut ListState) -> KickoffEffects {
    let effects = KickoffEffects {
        waits: std::mem::take(&mut state.pending_waits),
        sparse_writes: state.clear_sparse_writes(),
        texture_writes: std::mem::take(&mut state.texture_writes),
        texture_transitions: std::mem::take(&mut state.texture_transitions),
        uses_shared_view_heap: std::mem::take(&mut state.uses_shared_view_heap),
        write64_signals: state.clear_write64_signals(),
        cpu_writes: std::mem::take(&mut state.cpu_writes),
        native_generations: state.take_native_generations(),
    };
    state.written_ranges.clear();
    state.segment_write_producers.clear();
    effects
}

fn take_kickoff_effects_before_close(state: &mut ListState, has_fail_closed_standby: bool) -> Option<KickoffEffects> {
    (!has_fail_closed_standby).then(|| take_kickoff_effects(state))
}

fn restore_kickoff_native_generations(list: usize, pins: Vec<super::pso_async::NativeGeneration>) {
    if pins.is_empty() {
        return;
    }
    let mut lists = LISTS.lock().unwrap();
    let restored = match lists.as_mut().and_then(|lists| lists.get_mut(&list)) {
        Some(state) => Ok(pins.into_iter().filter_map(|pin| state.pin_native_generation(pin)).collect::<Vec<_>>()),
        None => Err(pins),
    };
    drop(lists);
    match restored {
        Ok(redundant) => drop(redundant),
        Err(stranded) => super::pso_async::poison_native_generations(stranded.iter()),
    }
}

fn settle_kickoff_standby<T, U>(
    standby: Option<(T, U)>,
    reset_succeeded: bool,
    return_spare: impl FnOnce(T, U),
    fail_closed: impl FnOnce(T, U, bool),
) {
    if let Some((wrapper, standby)) = standby {
        if reset_succeeded {
            return_spare(wrapper, standby);
        } else {
            fail_closed(wrapper, standby, false);
        }
    }
}

/// Recording state that a PC Reset loses and the continuation segment must inherit.
struct SplitCarry {
    allocator: usize,
    last_queue: Option<usize>,
    bound_pso: usize,
    depth_bias: Option<DepthBias>,
    index_buffer: Option<D3D12_INDEX_BUFFER_VIEW>,
    depth_bounds: Option<(f32, f32)>,
    predication: Option<D3D12_PREDICATION_OP>,
    pred_buf: Option<ID3D12Resource>,
    pred_in_state: bool,
    active_query: Option<(Query, D3D12_QUERY_TYPE)>,
    ended_early: Vec<Query>,
    written_ranges: RecordedWriteProducers,
    bindings: Bindings,
}

impl SplitCarry {
    fn capture(state: &ListState) -> Option<Self> {
        if state.allocator == 0 {
            return None;
        }
        let active_query = state.occlusion.active.map(|(q, ty, _)| (q, ty));
        let mut ended_early = state.occlusion.ended_early.clone();
        ended_early.extend(active_query.map(|(q, _)| q));
        Some(Self {
            allocator: state.allocator,
            last_queue: state.last_queue,
            bound_pso: state.bound_pso,
            depth_bias: state.depth_bias,
            index_buffer: state.index_buffer,
            depth_bounds: state.depth_bounds,
            predication: state.predication,
            pred_buf: state.pred_buf.clone(),
            pred_in_state: state.pred_in_state,
            active_query,
            ended_early,
            written_ranges: state.written_ranges.clone(),
            bindings: state.bindings.clone(),
        })
    }

    fn continuation_state(&self, ty: D3D12_COMMAND_LIST_TYPE) -> ListState {
        ListState {
            allocator: self.allocator,
            list_type: Some(ty),
            last_queue: self.last_queue,
            bound_pso: self.bound_pso,
            depth_bias: self.depth_bias,
            native_bias: DepthBias::ZERO,
            pso_dirty: false,
            using_fallback_pso: false,
            index_buffer: self.index_buffer,
            depth_bounds: self.depth_bounds,
            occlusion: Occlusion { active: None, ended_early: self.ended_early.clone() },
            written_ranges: self.written_ranges.clone(),
            pred_buf: self.pred_buf.clone(),
            pred_in_state: false,
            predication: self.predication,
            bindings: self.bindings.clone(),
            ..Default::default()
        }
    }

    fn commit_state(&self, state: &mut ListState) {
        state.occlusion.end_active_early();
        state.pred_in_state = false;
    }

}

#[derive(Clone, Copy)]
struct QueueInfo {
    queue: usize,
    id: super::title_textures::WriterQueueId,
    device: usize,
    ty: D3D12_COMMAND_LIST_TYPE,
}

static LISTS: Mutex<Option<HashMap<usize, ListState>>> = Mutex::new(None);
/// Closed standbys from `reserve_closed_segment` → the private allocator their empty recording is
/// on, kept while the list may be submitted with it. Not in `LISTS`: the allocator's last Release
/// takes `LISTS` (`on_allocator_release`), so it is always dropped with no lock held.
static STANDBY_ALLOCS: Mutex<Option<HashMap<usize, ID3D12CommandAllocator>>> = Mutex::new(None);

/// Forget `list`'s standby allocator (its recording is gone, or so is the list). `in_flight`: the
/// GPU may still run that recording, so the allocator is leaked rather than released.
fn drop_standby_alloc(list: usize) {
    let a = STANDBY_ALLOCS.lock().unwrap().as_mut().and_then(|m| m.remove(&list));
    if let Some(a) = a {
        if unsafe { super::alloc_pool::allocator_in_flight(a.as_raw() as usize) } {
            std::mem::forget(a);
        }
    }
}
static POISONED_EVER: AtomicBool = AtomicBool::new(false);

/// `f` with what is kept of every list.
fn every_list(mut f: impl FnMut(&mut ListState)) {
    if let Some(m) = LISTS.lock().unwrap().as_mut() {
        m.values_mut().for_each(&mut f);
    }
}
static LIVE_QUEUES: Mutex<Vec<QueueInfo>> = Mutex::new(Vec::new());
static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(0);
/// Number of lists with recorded sparse-window memory writes (fast-path guard for post-execute readback).
static SPARSE_WRITE_LISTS: AtomicUsize = AtomicUsize::new(0);
/// Number of lists with recorded non-timestamp `Write64BitValue*` signals (fast-path guard).
static WRITE64_SIGNAL_LISTS: AtomicUsize = AtomicUsize::new(0);
/// Set once any list uses `RSSetDepthBiasX` (fast-path guard for Draw* hooks).
static DEPTH_BIAS_USED: AtomicBool = AtomicBool::new(false);
/// Set once any list uses `SetPredicationBufferX` (fast-path guard for Close/Reset hooks).
static PREDICATION_USED: AtomicBool = AtomicBool::new(false);

/// Immutable effects recorded by one physical command-list segment at submission planning time.
#[derive(Clone, Default)]
pub(super) struct SegmentEffects {
    waits: Vec<PendingWait>,
    write_producers: RecordedWriteProducers,
    write64_signals: Vec<(u64, u64)>,
    sparse_writes: Vec<(usize, usize)>,
    /// Immediates into windows the GPU has a copy of: the CPU's view gets them from XCL.
    sparse_imm: Vec<(usize, u64, u8)>,
    texture_writes: Vec<super::title_textures::GpuWrite>,
    texture_transitions: Vec<super::title_textures::Transition>,
    uses_shared_view_heap: bool,
    cpu_writes: Vec<super::cpu_writes::Write>,
    native_generations: Vec<super::pso_async::NativeGeneration>,
}

impl SegmentEffects {
    pub(super) fn producer_overlaps(&self, producer: WriteProducer, va: u64, size: u64) -> bool {
        self.write_producers.overlaps(producer, va, size)
    }
}

/// One queue submission's physical lists and their effects, kept in matching chronological order.
pub(super) struct SubmissionPlan {
    lists: Vec<*mut c_void>,
    effects: Vec<SegmentEffects>,
    /// Exact recording tokens captured with `effects`; empty when causal dumps are disabled.
    traces: Vec<Option<Arc<Mutex<ListTrace>>>>,
    /// Per-segment dependency on an earlier CPU-after-segment producer; empty on the no-wait fast path.
    waits_after_cpu_write: Vec<bool>,
    has_waits: bool,
}

impl SubmissionPlan {
    fn capture(lists: &[*mut c_void]) -> Self {
        if lists.is_empty() {
            return Self { lists: Vec::new(), effects: Vec::new(), traces: Vec::new(), waits_after_cpu_write: Vec::new(), has_waits: false };
        }
        let g = LISTS.lock().unwrap();
        let effects: Vec<SegmentEffects> = lists
            .iter()
            .map(|&list| {
                g.as_ref()
                    .and_then(|m| m.get(&(list as usize)))
                    .map(|s| SegmentEffects {
                        waits: s.pending_waits.clone(),
                        write_producers: s.segment_write_producers.clone(),
                        write64_signals: s.write64_signals.clone(),
                        sparse_writes: s.sparse_writes.clone(),
                        sparse_imm: s.sparse_imm.clone(),
                        texture_writes: s.texture_writes.clone(),
                        texture_transitions: s.texture_transitions.clone(),
                        uses_shared_view_heap: s.uses_shared_view_heap,
                        cpu_writes: s.cpu_writes.clone(),
                        native_generations: s.native_generations.clone(),
                    })
                    .unwrap_or_default()
            })
            .collect();
        let traces = if sync_state_tracing() {
            lists
                .iter()
                .map(|&list| g.as_ref().and_then(|m| m.get(&(list as usize))).and_then(|s| s.write64_trace.clone()))
                .collect()
        } else {
            Vec::new()
        };
        drop(g);
        let has_waits = effects.iter().any(|e| !e.waits.is_empty());
        let waits_after_cpu_write = if has_waits {
            let mut prior_cpu_writes = RecordedWrites::default();
            effects
                .iter()
                .map(|effects| {
                    let depends = effects.waits.iter().any(|w| prior_cpu_writes.overlaps(w.va, w.width as u64));
                    prior_cpu_writes.extend(effects.write_producers.ranges(WriteProducer::CpuAfterSegment));
                    depends
                })
                .collect()
        } else {
            Vec::new()
        };
        Self { lists: lists.to_vec(), effects, traces, waits_after_cpu_write, has_waits }
    }

    fn cpu_producer_overlaps(&self, waits: &[PendingWait], before: usize) -> bool {
        self.effects[..before].iter().any(|e| {
            waits.iter().any(|w| e.producer_overlaps(WriteProducer::CpuAfterSegment, w.va, w.width as u64))
        })
    }

    fn chunk_needs_cpu_completion(&self, start: usize, end: usize, external_waits: &[PendingWait]) -> bool {
        self.effects[start..end].iter().any(|producer| {
            self.effects[end..]
                .iter()
                .flat_map(|e| &e.waits)
                .chain(external_waits)
                .any(|w| producer.producer_overlaps(WriteProducer::CpuAfterSegment, w.va, w.width as u64))
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct SubmissionChunk<'a> {
    lists: &'a [*mut c_void],
    effects: &'a [SegmentEffects],
    traces: &'a [Option<Arc<Mutex<ListTrace>>>],
    needs_cpu_completion: bool,
}

impl<'a> SubmissionChunk<'a> {
    pub(super) fn lists(self) -> &'a [*mut c_void] {
        self.lists
    }

    pub(super) fn needs_cpu_completion(self) -> bool {
        self.needs_cpu_completion
    }

    pub(super) fn uses_shared_view_heap(self) -> bool {
        self.effects.iter().any(|effects| effects.uses_shared_view_heap)
    }

    pub(super) fn native_generations(self) -> impl Clone + Iterator<Item = &'a super::pso_async::NativeGeneration> {
        self.effects.iter().flat_map(|effects| effects.native_generations.iter())
    }

    pub(super) fn has_native_generations(self) -> bool {
        self.effects.iter().any(|effects| !effects.native_generations.is_empty())
    }

    pub(super) unsafe fn submit_effects(self, queue: *mut c_void) -> Option<super::cpu_writes::Completion> {
        // The immutable submission plan is now the production path, so record execution here
        // after its caller has successfully submitted this exact chunk. The legacy
        // `signal_write64_fences` path below remains useful to tests but is not reached by the
        // title-facing submit path.
        for trace in self.traces.iter().flatten() {
            mark_trace(trace, ListEvent::Executed);
        }
        submit_segment_effects(queue, self.effects, self.needs_cpu_completion)
    }
}

/// Submit one exact title chunk, adding one shared retirement receipt when a recording
/// can read the shared view heap or owns native pipeline generation pins. The ordinary path stays
/// on `SubmissionGate::submit`; opt-in timing may append a best-effort diagnostic marker.
pub(super) unsafe fn submit_title_chunk<'a>(
    gate: &mut super::alloc_pool::SubmissionGate,
    queue: *mut c_void,
    lists: &[*mut c_void],
    protect_shared_view: bool,
    protect_native_generations: bool,
    native_generations: impl Iterator<Item = &'a super::pso_async::NativeGeneration>,
    prepare: impl FnOnce(),
    execute: impl FnOnce(),
) -> bool {
    if any_recording_poisoned(lists) {
        return false;
    }
    let track_gpu_completion = super::real::gpu_timing_on();
    if !protect_shared_view && !protect_native_generations {
        let submitted = gate.submit(queue, lists, prepare, execute);
        if submitted && track_gpu_completion {
            match super::alloc_pool::try_timing_receipt(queue) {
                Some(receipt) => super::real::note_gpu_completion_receipt(receipt, lists.len(), queue as usize),
                None => super::real::note_gpu_completion_receipt_failure(),
            }
        }
        return submitted;
    }
    let mut epoch = None;
    let submitted = gate.submit_receipted(
        queue,
        lists,
        || {
            if protect_shared_view {
                epoch = Some(super::vram::reserve_shared_view_epoch());
            }
            prepare();
        },
        execute,
    );
    if track_gpu_completion {
        match &submitted {
            super::alloc_pool::ReceiptedSubmission::Executed(receipt) => {
                super::real::note_gpu_completion_receipt(receipt.clone(), lists.len(), queue as usize);
            }
            super::alloc_pool::ReceiptedSubmission::ExecutedUntracked => {
                super::real::note_gpu_completion_receipt_failure();
            }
            super::alloc_pool::ReceiptedSubmission::NotExecuted => {}
        }
    }
    finish_title_submission(
        submitted,
        epoch,
        protect_shared_view,
        protect_native_generations,
        native_generations,
    )
}

fn finish_title_submission<'a>(
    submitted: super::alloc_pool::ReceiptedSubmission,
    epoch: Option<super::vram::SharedViewEpoch>,
    protect_shared_view: bool,
    protect_native_generations: bool,
    native_generations: impl Iterator<Item = &'a super::pso_async::NativeGeneration>,
) -> bool {
    match submitted {
        super::alloc_pool::ReceiptedSubmission::NotExecuted => {
            if let Some(epoch) = epoch {
                super::vram::cancel_shared_view_epoch(epoch);
            }
            false
        }
        super::alloc_pool::ReceiptedSubmission::Executed(receipt) => {
            if protect_native_generations {
                if protect_shared_view {
                    super::vram::attach_shared_view_epoch(
                        epoch.expect("executed shared-view submission without an epoch"),
                        receipt.clone(),
                    );
                }
                super::pso_async::retire_native_generations(receipt, native_generations);
            } else if protect_shared_view {
                super::vram::attach_shared_view_epoch(
                    epoch.expect("executed shared-view submission without an epoch"),
                    receipt,
                );
            }
            true
        }
        super::alloc_pool::ReceiptedSubmission::ExecutedUntracked => {
            if protect_shared_view {
                super::vram::poison_shared_view_epoch(
                    epoch.expect("executed shared-view submission without an epoch"),
                );
            }
            if protect_native_generations {
                super::pso_async::poison_native_generations(native_generations);
            }
            true
        }
    }
}

/// Record a newly created title command queue so `KickoffX` can find a compatible queue.
pub fn on_queue_create(queue: usize, device: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    if queue == 0 {
        return;
    }
    let mut g = LIVE_QUEUES.lock().unwrap();
    g.retain(|q| q.queue != queue);
    g.push(QueueInfo { queue, id: new_queue_id(queue), device, ty });
}

fn new_queue_id(raw: usize) -> super::title_textures::WriterQueueId {
    let generation = NEXT_QUEUE_ID.fetch_add(1, Ordering::Relaxed).wrapping_add(1).max(1);
    super::title_textures::WriterQueueId { raw, generation }
}

pub(super) fn queue_identity(raw: usize) -> Option<super::title_textures::WriterQueueId> {
    LIVE_QUEUES.lock().unwrap().iter().find(|q| q.queue == raw).map(|q| q.id)
}

/// Remove a released command queue so `KickoffX` never submits to a dead queue.
pub(super) fn on_queue_release(queue: usize) {
    let mut scratch = COPY_SCRATCH.lock().unwrap();
    scratch.fences.remove(&queue);
    scratch.collect_released();
    drop(scratch);
    stop_queue_submissions(queue);
}

/// Stop non-owning internal users (notably `KickoffX`) from selecting a retiring queue.
pub(super) fn stop_queue_submissions(queue: usize) {
    LIVE_QUEUES.lock().unwrap().retain(|q| q.queue != queue);
    every_list(|s| {
        if s.last_queue == Some(queue) {
            s.last_queue = None;
        }
    });
}

/// Remove every cached use of a retired raw queue address before it can be reused. This is later
/// than `on_queue_release`: the retirement marker has completed, so old sync edges are finished.
/// Cached COM fences are extracted and dropped after their locks are gone.
pub(super) fn forget_queue(id: super::present::QueueId) {
    let queue = id.raw();
    let scratch_fences = {
        let mut scratch = COPY_SCRATCH.lock().unwrap();
        let mut gone = Vec::new();
        if let Some((fence, _)) = scratch.fences.remove(&queue) {
            gone.push(fence);
        }
        for allocation in scratch.active.values_mut() {
            if let Some((fence, _)) = allocation.marks.remove(&queue) {
                gone.push(fence);
            }
        }
        for allocation in &mut scratch.released {
            if let Some((fence, _)) = allocation.marks.remove(&queue) {
                gone.push(fence);
            }
        }
        gone
    };
    drop(scratch_fences);
    LIVE_QUEUES.lock().unwrap().retain(|info| info.queue != queue);
    if let Some(lists) = LISTS.lock().unwrap().as_mut() {
        for state in lists.values_mut() {
            if state.last_queue == Some(queue) {
                state.last_queue = None;
            }
        }
    }
    let removed_waits = SYNC_GRAPH.lock().unwrap().as_mut().map_or(0, |graph| graph.forget_queue(queue));
    if removed_waits != 0 {
        ARMED_GPU_WAITS.fetch_sub(removed_waits, Ordering::AcqRel);
    }
    let internal_waits = INTERNAL_WAITS.lock().unwrap().as_mut().and_then(|waits| waits.remove(&queue));
    drop(internal_waits);
}

#[cfg(test)]
thread_local! {
    /// Optional test interceptor for `try_native_depth_bias` so unit tests deterministically
    /// verify the native `ID3D12GraphicsCommandList9::RSSetDepthBias` path and its fallback
    /// even on Windows builds whose inbox D3D12Core.dll stops at `ID3D12GraphicsCommandList7`.
    static TEST_NATIVE_BIAS: std::cell::RefCell<Option<Box<dyn FnMut(usize, DepthBias) -> bool>>> = const { std::cell::RefCell::new(None) };
}

/// Apply depth bias natively via `ID3D12GraphicsCommandList9::RSSetDepthBias` (exact f32,
/// without rounding to i32 or deriving a patched PSO). Returns `false` if unavailable.
unsafe fn try_native_depth_bias(list: *mut c_void, bias: DepthBias) -> bool {
    if list.is_null() {
        return false;
    }
    #[cfg(test)]
    if let Some(res) = TEST_NATIVE_BIAS.with(|h| h.borrow_mut().as_mut().map(|f| f(list as usize, bias))) {
        return res;
    }
    let Some(cl) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else {
        return false;
    };
    let Ok(cl9) = cl.cast::<ID3D12GraphicsCommandList9>() else {
        return false;
    };
    cl9.RSSetDepthBias(bias.depth_bias(), bias.depth_bias_clamp(), bias.slope_scaled_depth_bias());
    true
}

/// Record a newly created command list's allocator, initial pipeline, and type.
///
/// # Safety
/// `list` must be null or a valid `ID3D12GraphicsCommandList` COM pointer.
pub unsafe fn on_create(list: *mut c_void, alloc: usize, pso: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    on_create_with_backing(list, alloc, alloc, pso, ty)
}

/// Register the exact PC list owned by a console-shaped list. Its adopted Reset and final Release
/// use the normal hooks below, so this state has the same recording and object lifetime as `list`.
pub(super) unsafe fn on_console_list_create(list: *mut c_void, alloc: usize, pso: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    on_create(list, alloc, pso, ty)
}

/// Record a newly created command list's logical title allocator and native recording backing.
///
/// # Safety
/// `list` must be null or a valid `ID3D12GraphicsCommandList` COM pointer.
pub(super) unsafe fn on_create_with_backing(list: *mut c_void, alloc: usize, backing: usize, pso: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    if list.is_null() {
        return;
    }
    let key = list as usize;
    // (of a list that was here before and whose release was not seen: the GPU may still use it)
    release_copy_scratch(key);
    super::real::dispatch_chain::list_reset(key);
    let old = {
        let mut g = LISTS.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let old = m.remove(&key);
        if let Some(ref old) = old {
            if old.has_sparse() {
                SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
            if !old.write64_signals.is_empty() {
                WRITE64_SIGNAL_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
        }
        super::alloc_pool::created_on(key, backing, ty.0);
        m.insert(key, ListState { allocator: alloc, list_type: Some(ty), bound_pso: pso, ..Default::default() });
        old
    };
    drop(old);
    super::lut3d::list_reset(key);
    // Dynamic-depth-bias PSOs ignore baked RasterizerState depth bias; seed the list with the
    // initial PSO's static bias if non-zero (CreateCommandList resets RSSetDepthBias to 0).
    if ty != D3D12_COMMAND_LIST_TYPE_BUNDLE && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(list, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
}

/// Clean up all tracked state when a command list is released for good.
pub(super) fn on_release(list: usize) {
    trace_list(list, ListEvent::Released);
    super::alloc_pool::forget_list(list);
    release_copy_scratch(list);
    super::real::dispatch_chain::list_reset(list);
    super::lut3d::list_reset(list);
    drop_standby_alloc(list);
    let old = LISTS.lock().unwrap().as_mut().and_then(|m| m.remove(&list));
    if let Some(old) = old {
        if old.has_sparse() {
            SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
        if !old.write64_signals.is_empty() {
            WRITE64_SIGNAL_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
        // (the title may release a list the GPU still runs: its allocator must outlive that)
        if let Some(a) = old.own_alloc {
            // (the segments that recorded on it go with the list: they hold its PC allocators no
            // more, which are released with it once the GPU is done with them)
            super::alloc_pool::unpin_users(a.as_raw() as usize);
            if unsafe { super::alloc_pool::allocator_in_flight(a.as_raw() as usize) } {
                std::mem::forget(a);
            }
        }
    }
}

/// Clear per-recording sync/write tracking on a split segment moved into `title_list`'s spare pool.
pub(super) fn clear_segment_state(list: usize) {
    trace_list(list, ListEvent::Reset);
    super::lut3d::list_reset(list);
    let old_generations = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&list)) else { return };
        s.occlusion = Occlusion::default();
        s.written_ranges.clear();
        s.segment_write_producers.clear();
        s.clear_sparse_writes();
        s.clear_write64_signals();
        s.clear_waits();
        s.texture_writes.clear();
        s.texture_transitions.clear();
        s.uses_shared_view_heap = false;
        s.cpu_writes.clear();
        s.predication = None;
        s.pred_in_state = false;
        s.bindings = Bindings::default();
        s.deferred = None;
        s.take_native_generations()
    };
    drop(old_generations);
}

#[cfg(test)]
pub(super) fn has_list_state(list: usize) -> bool {
    LISTS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&list))
}

/// Forget a released command allocator so `KickoffX` will not reset a list with a dead allocator.
pub(super) fn on_allocator_release(alloc: usize) {
    every_list(|s| {
        if s.allocator == alloc {
            s.allocator = 0;
        }
    });
}

/// Track `SetPipelineState` on a command list (for `RSSetDepthBiasX` and `KickoffX`),
/// syncing `ID3D12GraphicsCommandList9::RSSetDepthBias` when the bound PSO uses native dynamic
/// depth bias.
pub(super) unsafe fn on_set_pipeline_state(list: *mut c_void, pso: usize) {
    let key = list as usize;
    let sync_target = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        // XCL-owned exact binds bypass this hook, so every call here is title-visible.
        s.bound_pso = pso;
        let effective = s.bound_pso;
        s.using_fallback_pso = false;
        let is_bundle = s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE);
        let info = if effective != 0 { device_x::base_depth_bias_info(effective) } else { None };
        if !is_bundle {
            if let Some((true, static_bias)) = info {
                let target = s.depth_bias.unwrap_or(static_bias);
                s.pso_dirty = false;
                if target != s.native_bias {
                    Some((target, s.depth_bias.is_some()))
                } else {
                    None
                }
            } else {
                s.pso_dirty = s.depth_bias.is_some();
                if s.pso_dirty {
                    DEPTH_BIAS_USED.store(true, Ordering::Release);
                }
                None
            }
        } else {
            s.pso_dirty = s.depth_bias.is_some();
            if s.pso_dirty {
                DEPTH_BIAS_USED.store(true, Ordering::Release);
            }
            None
        }
    };
    if let Some((target, has_override)) = sync_target {
        if try_native_depth_bias(list, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.native_bias = target;
            }
        } else if has_override {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
    }
}

/// Track `ClearState` on a command list: resets dynamic state and binds `pso`.
pub(super) unsafe fn on_clear_state(list: *mut c_void, pso: usize) {
    let key = list as usize;
    let is_bundle = {
        let mut g = LISTS.lock().unwrap();
        if let Some(s) = g.get_or_insert_with(HashMap::new).get_mut(&key) {
            s.bound_pso = pso;
            s.depth_bias = None;
            s.native_bias = DepthBias::ZERO;
            s.pso_dirty = false;
            s.using_fallback_pso = false;
            s.index_buffer = None;
            s.depth_bounds = None;
            s.predication = None;
            s.bindings = Bindings::default();
            s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)
        } else {
            false
        }
    };
    if !is_bundle && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(list, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
}

/// Associate executed lists with `queue` and keep `queue` at the head of `LIVE_QUEUES`.
pub(super) unsafe fn note_queue_for_lists(queue: usize, lists: &[*mut c_void]) {
    if queue == 0 {
        return;
    }
    {
        let mut g = LIVE_QUEUES.lock().unwrap();
        if let Some(pos) = g.iter().position(|q| q.queue == queue) {
            let info = g.remove(pos);
            g.push(info);
        } else {
            let raw = queue as *mut c_void;
            if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
                let ty = q.GetDesc().Type;
                let mut dev: Option<ID3D12Device> = None;
                let dev_raw = if q.GetDevice(&mut dev).is_ok() {
                    dev.as_ref().map_or(0, |d| d.as_raw() as usize)
                } else {
                    0
                };
                g.push(QueueInfo { queue, id: new_queue_id(queue), device: dev_raw, ty });
            }
        }
    }
    for &l in lists {
        if !l.is_null() {
            LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(l as usize).or_default().last_queue = Some(queue);
        }
    }
}

fn pick_queue(last_queue: Option<usize>, device: usize, ty: D3D12_COMMAND_LIST_TYPE) -> Option<usize> {
    let g = LIVE_QUEUES.lock().unwrap();
    if let Some(q) = last_queue {
        if g.iter().any(|info| info.queue == q) {
            return Some(q);
        }
    }
    g.iter().rev().find(|info| info.ty == ty && (device == 0 || info.device == 0 || info.device == device)).map(|info| info.queue)
}

fn note_write(list: usize, producer: WriteProducer, va: u64, size: u64) {
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(list).or_default().note_write(producer, va, size);
}

fn note_texture_write(list: usize, res: usize, subresource: u32) {
    let Some(write) = super::title_textures::capture_gpu_write(res, subresource) else { return };
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(list).or_default().texture_writes.push(write);
}

/// A PC copy into `[offset, offset + n)` of console buffer `dst` whose memory the GPU only has
/// a copy of in a committed window (`Uma::shadowed`: readback buffers live there, and titles
/// read them through a pointer they keep mapped, without another Map), or placed in a GPU view
/// (what the GPU writes there is its tiles', and the title reads it at the CPU view): the bytes
/// come back to the CPU's view after the submit, like the lists' own memory writes.
fn note_copy_into(list: usize, dst: usize, offset: u64, n: u64) {
    let Some(u) = crate::gpu::uma::get().filter(|u| u.shadowed() || super::bind::view_span().1 != 0) else { return };
    let Some((p, bytes)) = real::console_buffer(dst).filter(|&(p, _)| (u.shadowed() && u.is_committed(p)) || u.is_view(p)) else { return };
    let offset = offset.min(bytes as u64);
    let (at, n) = (p as u64 + offset, n.min(bytes as u64 - offset));
    // (a buffer placed in a GPU view is over its tiles already: this counts the path)
    super::bind::want_view(super::view_want::Path::CopyInto, at, n);
    note_write(list, WriteProducer::InSegmentGpu, at, n);
}

// ---------------------------------------------------------------------------------------
// per-address GPU fence bridge for monotonic 64-bit Write64BitValue* -> Wait64BitValueX
// ---------------------------------------------------------------------------------------

/// Recent `Write32/64BitValue*` call for timeout diagnostics.
#[derive(Clone, Copy)]
struct WriteTraceEntry {
    va: u64,
    value: u64,
    flags: u32,
    list_type: i32,
    what: &'static str,
}

struct WriteTraceRing {
    entries: [Option<WriteTraceEntry>; 32],
    next: usize,
    total: u64,
}

impl WriteTraceRing {
    const fn new() -> Self {
        Self { entries: [None; 32], next: 0, total: 0 }
    }

    fn push(&mut self, e: WriteTraceEntry) {
        self.entries[self.next] = Some(e);
        self.next = (self.next + 1) % self.entries.len();
        self.total += 1;
    }

    fn snapshot(&self) -> (u64, Vec<WriteTraceEntry>) {
        let mut out = Vec::with_capacity(self.entries.len());
        for i in 0..self.entries.len() {
            let idx = (self.next + i) % self.entries.len();
            if let Some(e) = self.entries[idx] {
                out.push(e);
            }
        }
        (self.total, out)
    }
}

static WRITE_TRACE: Mutex<WriteTraceRing> = Mutex::new(WriteTraceRing::new());

fn record_write_trace(list: *mut c_void, va: u64, value: u64, flags: u32, what: &'static str) {
    let ty = LISTS.lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .and_then(|s| s.list_type)
        .map_or(-1, |t| t.0);
    WRITE_TRACE.lock().unwrap().push(WriteTraceEntry { va, value, flags, list_type: ty, what });
}

/// Active GPU `queue.Wait(&fence, val)` tracked for exact cycle detection and deadlock watchdog.
#[derive(Clone, Copy)]
struct ArmedGpuWait {
    val: u64,
    waiting_queue: usize,
    waiting_queue_ty: i32,
    /// Per-queue submission sequence number at which this `Wait` was issued on `waiting_queue`.
    wait_seq: u64,
    armed_at: Instant,
    after_write64: bool,
}

/// Exact GPU `queue.Signal(&fence, val)` submission point on `queue`.
#[derive(Clone, Copy)]
struct SubmittedSignal {
    val: u64,
    queue: usize,
    /// Per-queue submission sequence number at which this `Signal` was issued on `queue`.
    seq: u64,
}

/// Per-address GPU fence state bridging `Write64BitValue{Bottom,Top}OfPipeX` to cross-queue
/// `Wait64BitValueX` (`GREATER_EQUAL`, full 64-bit mask).
#[derive(Default)]
struct SyncCell {
    fence: Option<ID3D12Fence>,
    /// Raw `ID3D12Device` pointer owning `fence`.
    device: usize,
    /// Highest non-zero value recorded in a `Write64BitValue*` targeting this VA.
    recorded_max: u64,
    /// Command list type of the most recent recording advancing `recorded_max`.
    recorded_list_type: Option<D3D12_COMMAND_LIST_TYPE>,
    /// Last queue the recording list was executed on (if known).
    recorded_last_queue: Option<usize>,
    /// The PC list of the recording that set `recorded_max`, when, and what came to that
    /// recording (the sync-state dump).
    recorded_by: Option<(usize, Instant, Arc<Mutex<ListTrace>>)>,
    /// Highest value submitted via `queue.Signal(&fence, v)` or released on CPU.
    submitted_max: u64,
    /// Queue that issued the latest GPU `Signal` (`None` if released on CPU).
    submitted_queue: Option<usize>,
    /// Highest value explicitly released on CPU (cycle-break or watchdog). Protects against
    /// vkd3d-proton rewinding `fence->virtual_value` when an older in-flight GPU signal retires.
    cpu_signaled_max: u64,
    /// Set once a strictly decreasing write (`value < submitted_max`) is observed at this VA,
    /// disabling GPU-fence bridging and falling back to CPU polling.
    non_monotonic: bool,
    /// Active GPU `queue.Wait(&fence, v)` calls on this cell not yet confirmed completed on GPU.
    armed_waits: Vec<ArmedGpuWait>,
    /// Active GPU `queue.Signal(&fence, v)` calls on this cell not yet confirmed completed on GPU.
    submitted_signals: Vec<SubmittedSignal>,
}

// SAFETY: `ID3D12Fence` is a free-threaded COM object in D3D12.
unsafe impl Send for SyncCell {}

/// Tracked title `ID3D12CommandQueue::Wait` / `Signal` edges on a title `ID3D12Fence`, used so
/// exact cycle detection sees mixed cycles involving both `Wait64BitValueX` and title fences.
#[derive(Default)]
struct TitleFenceSync {
    waits: Vec<(u64, usize, u64)>,   // (val, waiting_queue, wait_seq)
    signals: Vec<SubmittedSignal>,   // (val, signaling_queue, sig_seq)
}

#[derive(Default)]
struct SyncGraph {
    cells: HashMap<u64, SyncCell>,
    title_fences: HashMap<usize, TitleFenceSync>,
    queue_seq: HashMap<usize, u64>,
}

impl SyncGraph {
    fn next_seq(&mut self, queue: usize) -> u64 {
        let s = self.queue_seq.entry(queue).or_insert(0);
        *s += 1;
        *s
    }

    fn forget_queue(&mut self, queue: usize) -> usize {
        self.queue_seq.remove(&queue);
        let mut armed = 0;
        for cell in self.cells.values_mut() {
            if cell.recorded_last_queue == Some(queue) {
                cell.recorded_last_queue = None;
            }
            if cell.submitted_queue == Some(queue) {
                cell.submitted_queue = None;
            }
            let before = cell.armed_waits.len();
            cell.armed_waits.retain(|wait| wait.waiting_queue != queue);
            armed += before - cell.armed_waits.len();
            cell.submitted_signals.retain(|signal| signal.queue != queue);
        }
        for fence in self.title_fences.values_mut() {
            fence.waits.retain(|&(_, waiting_queue, _)| waiting_queue != queue);
            fence.signals.retain(|signal| signal.queue != queue);
        }
        armed
    }
}

static SYNC_GRAPH: Mutex<Option<SyncGraph>> = Mutex::new(None);
static ARMED_GPU_WAITS: AtomicUsize = AtomicUsize::new(0);
static SYNC_STATE_TRACING: AtomicBool = AtomicBool::new(false);

/// Enable recording-level causal detail only when at least one scheduled sync dump will use it.
pub fn enable_sync_state_tracing() {
    SYNC_STATE_TRACING.store(true, Ordering::Release);
}

#[inline]
fn sync_state_tracing() -> bool {
    SYNC_STATE_TRACING.load(Ordering::Acquire)
}

/// What became of a recording of a PC list that recorded a `Write64BitValue*`: when it was last
/// executed, and the Reset or Release that ended it (the sync-state dump; a cell recorded and
/// never submitted names its list).
#[derive(Default, Clone, Copy)]
struct ListTrace {
    executed: Option<Instant>,
    reset: Option<Instant>,
    released: Option<Instant>,
}

#[derive(Clone, Copy)]
enum ListEvent {
    Executed,
    Reset,
    Released,
}

/// Notes `event` on the recording `list` has open, when it recorded a `Write64BitValue*`. A Reset
/// or a Release ends the recording: what comes to the list (or its address) after is another's.
fn trace_list(list: usize, event: ListEvent) {
    if !sync_state_tracing() {
        return;
    }
    let t = {
        let mut g = LISTS.lock().unwrap();
        let Some(state) = g.as_mut().and_then(|m| m.get_mut(&list)) else { return };
        match event {
            ListEvent::Executed => state.write64_trace.clone(),
            ListEvent::Reset | ListEvent::Released => state.write64_trace.take(),
        }
    };
    if let Some(t) = t {
        mark_trace(&t, event);
    }
}

fn mark_trace(trace: &Mutex<ListTrace>, event: ListEvent) {
    let mut trace = trace.lock().unwrap();
    let at = Some(Instant::now());
    match event {
        ListEvent::Executed => trace.executed = at,
        ListEvent::Reset => trace.reset = at,
        ListEvent::Released => trace.released = at,
    }
}

/// `by <list> <age> ms ago (executed …, reset …, released …)`: each as its time after the
/// recording `t`, or `no` when it has not come to it.
fn recording_trace(list: usize, at: Instant, t: &Mutex<ListTrace>) -> String {
    let Ok(t) = t.try_lock().map(|t| *t) else { return format!("by {list:#x} {} ms ago", at.elapsed().as_millis()) };
    let after = |e: Option<Instant>| e.map_or("no".to_string(), |e| format!("+{} ms", e.saturating_duration_since(at).as_millis()));
    format!("by {list:#x} {} ms ago (executed {}, reset {}, released {})", at.elapsed().as_millis(), after(t.executed), after(t.reset), after(t.released))
}
static SYNC_WATCHDOG_WAKE: OnceLock<(Mutex<()>, Condvar)> = OnceLock::new();
static QUEUE_IDS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// A `queue.Wait` XCL itself put on a queue (sparse uploads and readbacks, texture page
/// mappings). These are not in `SYNC_GRAPH`; the GPU wait watchdog lists the unmet ones, so a
/// producer queue that stands behind one of them shows in the log.
struct InternalWait {
    what: &'static str,
    /// the cell's address for a bridged `Wait64BitValueX` (0: not one)
    va: u64,
    fence: ID3D12Fence,
    value: u64,
    at: Instant,
}

/// queue → its private progress fence, the submits signalled on it, and when each of the
/// last ones was signalled (diagnostic: how far behind a queue is, and since when)
struct Progress {
    fence: ID3D12Fence,
    submitted: u64,
    recent: std::collections::VecDeque<(u64, Instant)>,
}

unsafe impl Send for Progress {}

static PROGRESS: Mutex<Option<HashMap<usize, Progress>>> = Mutex::new(None);

/// After a submit to `queue`: a Signal of its private progress fence.
pub(super) unsafe fn note_submitted(queue: *mut c_void) {
    if queue.is_null() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let mut g = PROGRESS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if !m.contains_key(&(queue as usize)) {
        let mut dev: Option<ID3D12Device> = None;
        if q.GetDevice(&mut dev).is_err() {
            return;
        }
        let Some(dev) = dev else { return };
        let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
        m.insert(queue as usize, Progress { fence, submitted: 0, recent: Default::default() });
    }
    let p = m.get_mut(&(queue as usize)).unwrap();
    p.submitted += 1;
    if super::real::native_queue_signal(q, &p.fence, p.submitted).is_ok() {
        p.recent.push_back((p.submitted, Instant::now()));
        while p.recent.len() > 4096 {
            p.recent.pop_front();
        }
    }
}

/// `q<id> <done>/<submitted> submits (oldest unfinished for <ms> ms)` for every queue behind.
fn progress_summary() -> String {
    let rows: Vec<(usize, String)> = {
        let mut g = PROGRESS.lock().unwrap();
        g.iter_mut()
            .flat_map(|m| m.iter_mut())
            .filter_map(|(&q, p)| {
                let done = unsafe { super::real::fence_completed(&p.fence) };
                while p.recent.front().is_some_and(|&(n, _)| n <= done) {
                    p.recent.pop_front();
                }
                (done < p.submitted).then(|| (q, format!("{done}/{} submits (oldest unfinished for {} ms)", p.submitted, p.recent.front().map_or(0, |&(_, t)| t.elapsed().as_millis()))))
            })
            .collect()
    };
    if rows.is_empty() {
        return "no queue is behind".into();
    }
    let mut rows: Vec<String> = rows.into_iter().map(|(q, s)| format!("q{} {s}", queue_id(q))).collect();
    rows.sort();
    rows.join(", ")
}

/// Diagnostic: the wait-for edges of the sync graph, one per GPU wait not yet met: who waits
/// for which value of which cell (or title fence), and which submitted signal (queue, its
/// place among that queue's calls) will meet it. A wait for a value no signal has been submitted
/// for says what was recorded instead.
fn edges_summary() -> String {
    let g = SYNC_GRAPH.lock().unwrap();
    g.as_ref().map_or("none".into(), edges_of)
}

fn edges_of(graph: &SyncGraph) -> String {
    let mut rows: Vec<String> = Vec::new();
    for (&va, cell) in &graph.cells {
        let done = cell.fence.as_ref().map_or(0, |f| unsafe { super::real::fence_completed(f) });
        for aw in &cell.armed_waits {
            if aw.val <= done {
                continue;
            }
            let met = cell.submitted_signals.iter().find(|s| s.val >= aw.val);
            let by = match met {
                Some(s) => format!("q{}@{} (signal {:#x})", queue_id(s.queue), s.seq, s.val),
                None => format!("none submitted (recorded up to {:#x}, submitted up to {:#x} by {:?})", cell.recorded_max, cell.submitted_max, cell.submitted_queue.map(queue_id)),
            };
            rows.push(format!("q{}@{} waits {va:#x} >= {:#x} (done {done:#x}) <- {by}", queue_id(aw.waiting_queue), aw.wait_seq, aw.val));
        }
    }
    for (&f, tf) in &graph.title_fences {
        for &(val, q, seq) in &tf.waits {
            let met = tf.signals.iter().find(|s| s.val >= val);
            let by = match met {
                Some(s) => format!("q{}@{} (signal {:#x})", queue_id(s.queue), s.seq, s.val),
                None => "none submitted".to_string(),
            };
            rows.push(format!("q{}@{seq} waits title fence {f:#x} >= {val:#x} <- {by}", queue_id(q)));
        }
    }
    rows.sort();
    let n = rows.len();
    rows.truncate(140);
    format!("{n} waits: {}", rows.join("; "))
}

/// `queue` was made to wait for the fence of the cell at `va` to reach `value` (a bridged
/// `Wait64BitValueX`): listed with XCL's own waits while it is unmet, whatever the sync graph
/// has done with it since.
fn note_cell_wait(queue: usize, va: u64, fence: &ID3D12Fence, value: u64) {
    let mut g = INTERNAL_WAITS.lock().unwrap();
    let waits = g.get_or_insert_with(HashMap::new).entry(queue).or_default();
    waits.retain(|w| unsafe { super::real::fence_completed(&w.fence) } < w.value);
    waits.push(InternalWait { what: "Wait64 cell", va, fence: fence.clone(), value, at: Instant::now() });
}

static INTERNAL_WAITS: Mutex<Option<HashMap<usize, Vec<InternalWait>>>> = Mutex::new(None);

/// `queue` was made to wait for `fence` >= `value` by XCL (`what` says for what).
pub(super) fn note_internal_wait(queue: usize, what: &'static str, fence: &ID3D12Fence, value: u64) {
    let mut g = INTERNAL_WAITS.lock().unwrap();
    let waits = g.get_or_insert_with(HashMap::new).entry(queue).or_default();
    waits.retain(|w| unsafe { super::real::fence_completed(&w.fence) } < w.value);
    waits.push(InternalWait { what, va: 0, fence: fence.clone(), value, at: Instant::now() });
}

/// The internal waits not yet met: `q<id>: <what> >= <value> (done <completed>, <age> ms)`.
fn internal_waits_summary() -> String {
    let unmet: Vec<(usize, String)> = {
        let g = INTERNAL_WAITS.lock().unwrap();
        g.iter()
            .flat_map(|m| m.iter())
            .flat_map(|(&q, waits)| waits.iter().map(move |w| (q, w)))
            .filter_map(|(q, w)| {
                let done = unsafe { super::real::fence_completed(&w.fence) };
                let what = if w.va != 0 { format!("{} {:#x}", w.what, w.va) } else { w.what.to_string() };
                (done < w.value).then(|| (q, format!("{what} >= {:#x} (done {done:#x}, {} ms)", w.value, w.at.elapsed().as_millis())))
            })
            .collect()
    };
    if unmet.is_empty() {
        return "none".into();
    }
    unmet.into_iter().map(|(q, s)| format!("q{}: {s}", queue_id(q))).collect::<Vec<_>>().join(", ")
}

/// The `Wait64BitValueX` cells not yet at their highest recorded value, or with armed GPU waits:
/// `rec` above `sub` with nothing armed is a list that writes the value, recorded and never
/// submitted (its submitting thread is held). `try_lock`: a dump can come while a submitting
/// thread holds the graph. (@GreemSC's diagnostic, from #2244's review.)
fn cells_behind_summary() -> String {
    let mut parts = Vec::new();
    if let Ok(g) = SYNC_GRAPH.try_lock() {
        for (&va, c) in g.iter().flat_map(|gr| gr.cells.iter()) {
            let done = c.fence.as_ref().map_or(0, |f| unsafe { f.GetCompletedValue() });
            if c.armed_waits.is_empty() && done >= c.recorded_max {
                continue;
            }
            // (the title can decommit the cell's page at any moment: a read that cannot fault)
            let mut v = 0u64;
            let mem = if unsafe { crate::hle::dstorage::read_own(va as usize, &mut v as *mut u64 as *mut u8, 8) } { format!("{v:#x}") } else { "unreadable".into() };
            let armed: Vec<String> = c.armed_waits.iter().map(|a| format!("{:#x}@q{} {} ms", a.val, queue_id(a.waiting_queue), a.armed_at.elapsed().as_millis())).collect();
            parts.push(format!(
                "{va:#x}: mem {mem} done {done:#x} sub {:#x}@q{:?} rec {:#x}{} cpu {:#x}{} armed [{}]",
                c.submitted_max,
                c.submitted_queue.map(queue_id),
                c.recorded_max,
                c.recorded_by.as_ref().map_or(String::new(), |(l, at, t)| format!(" {}", recording_trace(*l, *at, t))),
                c.cpu_signaled_max,
                if c.non_monotonic { " non-monotonic" } else { "" },
                armed.join(", ")
            ));
        }
    } else {
        parts.push("(sync graph locked)".into());
    }
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join("; ")
    }
}

/// What the GPU wait watchdog logs at a time-out, logged on request (`XCL_THREAD_DUMP_AT`), so a
/// title that stops with no time-out shows whether a fence or a queue is behind, and which
/// `Wait64BitValueX` cells are behind what was recorded for them. On a thread of its own: a lock
/// the stall holds does not hold up the next thread dump.
pub fn log_sync_state(at: &str) {
    let at = at.to_string();
    let _ = std::thread::Builder::new().name("xcl-sync-state".into()).spawn(move || {
        crate::xlog_info!("D3D12.X: sync state at {at}: pending title ID3D12Fences: {}", super::fence_x::pending_summary());
        crate::xlog_info!("D3D12.X: sync state at {at}: XCL's own queue waits unmet: {}", internal_waits_summary());
        crate::xlog_info!("D3D12.X: sync state at {at}: {} GPU waits armed", ARMED_GPU_WAITS.load(Ordering::Acquire));
        crate::xlog_info!("D3D12.X: sync state at {at}: Wait64 cells behind: {}", cells_behind_summary());
    });
}

/// Diagnostic (`XCL_SYNC_TRACE=1`): the last sync events in the order XCL made them.
static SYNC_TRACE: Mutex<std::collections::VecDeque<(u64, String)>> = Mutex::new(std::collections::VecDeque::new());

fn sync_trace_on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("XCL_SYNC_TRACE").is_ok_and(|v| v.trim() == "1"));
    *ON
}

fn sync_trace(what: impl FnOnce() -> String) {
    if !sync_trace_on() {
        return;
    }
    static T0: LazyLock<Instant> = LazyLock::new(Instant::now);
    let t = T0.elapsed().as_micros() as u64;
    let w = what();
    let mut g = SYNC_TRACE.lock().unwrap();
    if g.len() >= 1500 {
        g.pop_front();
    }
    g.push_back((t, w));
}

/// Log the ring (`why`), `times` times in all for each reason's counter.
fn dump_sync_trace(why: &str, n: &AtomicU64, times: u64) {
    if !sync_trace_on() || n.fetch_add(1, Ordering::Relaxed) >= times {
        return;
    }
    let g = SYNC_TRACE.lock().unwrap();
    let last = g.back().map_or(0, |e| e.0);
    crate::xlog_warn!("SYNC TRACE ({why}): {} events, the last at 0:", g.len());
    for (t, w) in g.iter() {
        crate::xlog_warn!("SYNC {:>9.3} {w}", (*t as f64 - last as f64) / 1000.0);
    }
}

fn queue_id(q: usize) -> usize {
    let mut g = QUEUE_IDS.lock().unwrap();
    if let Some(pos) = g.iter().position(|&x| x == q) {
        pos
    } else {
        let id = g.len();
        g.push(q);
        id
    }
}

/// Remove any `armed_waits` and `submitted_signals` already satisfied on the GPU (or if the device
/// was removed). Returns the fence's current completed value (or `0` if no fence exists yet).
unsafe fn prune_completed_waits(cell: &mut SyncCell) -> u64 {
    let Some(ref fence) = cell.fence else { return 0 };
    let completed = super::real::fence_completed(&fence);
    if completed == u64::MAX {
        let removed = cell.armed_waits.len();
        cell.armed_waits.clear();
        cell.submitted_signals.clear();
        if removed > 0 {
            ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
        }
        return completed;
    }
    if !cell.armed_waits.is_empty() {
        let before = cell.armed_waits.len();
        cell.armed_waits.retain(|aw| aw.val > completed);
        let removed = before - cell.armed_waits.len();
        if removed > 0 {
            ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
        }
    }
    if !cell.submitted_signals.is_empty() {
        cell.submitted_signals.retain(|s| s.val > completed);
    }
    completed
}

/// Prune completed waits/signals from a tracked title `ID3D12Fence`.
unsafe fn prune_title_fence(fence_raw: usize, tf: &mut TitleFenceSync) {
    if tf.waits.is_empty() && tf.signals.is_empty() {
        return;
    }
    let p = fence_raw as *mut c_void;
    let Some(f) = ID3D12Fence::from_raw_borrowed(&p) else { return };
    let completed = super::real::fence_completed(&f);
    if completed == u64::MAX {
        tf.waits.clear();
        tf.signals.clear();
    } else if completed > 0 {
        tf.waits.retain(|&(val, _, _)| val > completed);
        tf.signals.retain(|s| s.val > completed);
    }
}

/// Exact `(queue, seq)` cycle detection BEFORE emitting `sig_q.Signal(&cell_va_fence, sig_val)`.
///
/// Theorem: in any GPU deadlock cycle of length `k >= 1` (`q_0 -> q_1 -> ... -> q_0`), every queue
/// `q_i` submits its `Wait` strictly before its `Signal` (`wait_seq < sig_seq`). Therefore the
/// chronologically last call closing any cycle is ALWAYS a `Signal`. Checking the exact submitted
/// `(queue, seq)` graph right before emitting each `Signal` catches 100% of cycles (including
/// same-queue self-deadlocks, 2..N-queue cycles across rotating compute queues, and mixed cycles
/// involving title `ID3D12Fence` waits/signals) before any cyclic wait-signal pair reaches Vulkan.
///
/// Returns `true` iff emitting `sig_q.Signal` for `(target_va, sig_val)` would close a cycle.
unsafe fn would_signal_close_cycle(graph: &mut SyncGraph, sig_q: usize, target_va: u64, sig_val: u64) -> bool {
    // Prune completed waits/signals on target cell first.
    let target_waiters: Vec<(usize, u64)> = {
        let Some(target) = graph.cells.get_mut(&target_va) else { return false };
        let completed = prune_completed_waits(target);
        if completed == u64::MAX || target.armed_waits.is_empty() {
            return false;
        }
        // A wait an earlier signal of this cell already satisfies (GPU or CPU) does not wait on
        // this one: counting it made every pipelined frame (wait N armed, signal N+1 next) a cycle.
        let covered = target.submitted_max.max(target.cpu_signaled_max);
        target
            .armed_waits
            .iter()
            .filter(|aw| aw.val <= sig_val && aw.val > covered)
            .map(|aw| (aw.waiting_queue, aw.wait_seq))
            .collect()
    };
    if target_waiters.is_empty() {
        return false;
    }
    // k = 1: direct same-queue self-deadlock (`sig_q` is already waiting on `target_va <= sig_val`).
    if target_waiters.iter().any(|&(wq, _)| wq == sig_q) {
        return true;
    }

    // Prune all cells & title fences and build exact edge list: (waiter_q, wait_seq, prod_q, prod_sig_seq).
    // Include earlier submitted signals on `target_va` too (e.g. frame N-1 wait -> signal).
    let mut edges: Vec<(usize, u64, usize, u64)> = Vec::new();
    for (&_va, cell) in graph.cells.iter_mut() {
        if cell.armed_waits.is_empty() {
            continue;
        }
        let completed = prune_completed_waits(cell);
        if completed == u64::MAX || cell.armed_waits.is_empty() {
            continue;
        }
        for aw in &cell.armed_waits {
            // Earliest submitted GPU signal satisfying `aw.val` (matches vkd3d-proton update_count order).
            if let Some(s) = cell.submitted_signals.iter().find(|s| s.val >= aw.val) {
                edges.push((aw.waiting_queue, aw.wait_seq, s.queue, s.seq));
            }
        }
    }
    for (&tf_raw, tf) in graph.title_fences.iter_mut() {
        prune_title_fence(tf_raw, tf);
        for &(val, wq, wseq) in &tf.waits {
            if let Some(s) = tf.signals.iter().find(|s| s.val >= val) {
                edges.push((wq, wseq, s.queue, s.seq));
            }
        }
    }
    if edges.is_empty() {
        return false;
    }

    // BFS over `(queue, limit_seq)`: all waits on `sig_q` sit before our new `sig_q.Signal` (`u64::MAX`).
    let mut best_limit: HashMap<usize, u64> = HashMap::new();
    let mut stack: Vec<(usize, u64)> = vec![(sig_q, u64::MAX)];
    best_limit.insert(sig_q, u64::MAX);
    while let Some((cur_q, limit_seq)) = stack.pop() {
        for &(wq, wseq, pq, pseq) in &edges {
            if wq != cur_q || wseq >= limit_seq {
                continue;
            }
            // If `pq` has a wait on `target_va <= sig_val` strictly BEFORE `pq`'s signal `pseq`,
            // then `pq`'s signal cannot execute until `target_va` completes -> exact cycle!
            if target_waiters.iter().any(|&(twq, twseq)| twq == pq && twseq < pseq) {
                return true;
            }
            let prev = best_limit.get(&pq).copied().unwrap_or(0);
            if pseq > prev {
                best_limit.insert(pq, pseq);
                stack.push((pq, pseq));
            }
        }
    }
    false
}

/// Record a title `ID3D12CommandQueue::Wait(fence, value)` in the exact sync graph.
pub(super) unsafe fn note_title_queue_wait(queue: usize, fence: usize, value: u64) {
    if queue == 0 || fence == 0 || value == 0 {
        return;
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if once(&LOGGED, 8) {
        crate::xlog_info!("D3D12.X: title ID3D12CommandQueue::Wait on q{} fence {fence:#x} >= {value:#x}", queue_id(queue));
    }
    let mut g = SYNC_GRAPH.lock().unwrap();
    let graph = g.get_or_insert_with(SyncGraph::default);
    let seq = graph.next_seq(queue);
    let tf = graph.title_fences.entry(fence).or_default();
    prune_title_fence(fence, tf);
    tf.waits.push((value, queue, seq));
}

/// Record a title `ID3D12CommandQueue::Signal(fence, value)` in the exact sync graph.
pub(super) unsafe fn note_title_queue_signal(queue: usize, fence: usize, value: u64) {
    if queue == 0 || fence == 0 || value == 0 {
        return;
    }
    let mut g = SYNC_GRAPH.lock().unwrap();
    let graph = g.get_or_insert_with(SyncGraph::default);
    let seq = graph.next_seq(queue);
    let tf = graph.title_fences.entry(fence).or_default();
    prune_title_fence(fence, tf);
    tf.signals.push(SubmittedSignal { val: value, queue, seq });
}

/// Publish a cycle-broken `Write64` value into title memory (`fetch_max`) so CPU polling and
/// subsequent `Wait64BitValueX` checks observe the counter advancing. Sparse UMA windows are
/// skipped: CPU writes there dirty write-watched pages and would upload stale sibling bytes into
/// tiled VRAM. Committed coherent UMA takes the O(1) fast path; non-UMA title memory reuses
/// `cpu_writes::max_u64_now` (off the hot path: cycle breaks only).
unsafe fn publish_cpu_cycle_break(va: u64, val: u64) {
    publish_cpu_cycle_break_with(va, val, crate::gpu::uma::get());
}

unsafe fn publish_cpu_cycle_break_with(va: u64, val: u64, uma: Option<&crate::gpu::uma::Uma>) {
    if val == 0 || va == 0 || !va.is_multiple_of(8) {
        return;
    }
    let addr = va as usize;
    if uma.is_some_and(|u| u.is_sparse(addr)) {
        return;
    }
    if uma.is_some_and(|u| u.in_committed_window(addr, 8)) {
        (*(addr as *const AtomicU64)).fetch_max(val, Ordering::AcqRel);
    } else {
        super::cpu_writes::max_u64_now(addr, val);
    }
}

/// Remove a synthetic test fence and queue entry from `SYNC_GRAPH` before they are dropped.
#[cfg(test)]
pub(super) fn forget_title_fence_for_test(fence: usize, queue: usize) {
    if let Some(graph) = SYNC_GRAPH.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        graph.title_fences.remove(&fence);
        graph.queue_seq.remove(&queue);
    }
}

/// Before a title `ID3D12CommandQueue::Signal(title_fence, value)` is submitted, check if it would
/// close a mixed cycle passing through any `Wait64BitValueX` `SyncCell`. If so, release that
/// `SyncCell` on CPU BEFORE the title signal executes so Vulkan never enters a semaphore cycle.
pub(super) unsafe fn break_cycles_before_title_signal(sig_q: usize, title_fence: usize, sig_val: u64) {
    if sig_q == 0 || title_fence == 0 || sig_val == 0 {
        return;
    }
    let to_release: Vec<(u64, u64, ID3D12Fence)> = {
        let mut g = SYNC_GRAPH.lock().unwrap();
        let Some(graph) = g.as_mut() else { return };
        let Some(tf) = graph.title_fences.get_mut(&title_fence) else { return };
        prune_title_fence(title_fence, tf);
        // (a wait an earlier signal of the fence satisfies does not wait on this one, as for the
        // cells in `would_signal_close_cycle`)
        let covered = tf.signals.iter().map(|s| s.val).max().unwrap_or(0);
        let target_waiters: Vec<(usize, u64)> = tf.waits.iter().filter(|&&(v, _, _)| v <= sig_val && v > covered).map(|&(_, wq, wseq)| (wq, wseq)).collect();
        if target_waiters.is_empty() {
            return;
        }
        // Build edges tagging whether each edge is a Wait64 SyncCell `(Some((va, aw_val)))` or title fence (`None`).
        let mut edges: Vec<(usize, u64, usize, u64, Option<(u64, u64)>)> = Vec::new();
        for (&va, cell) in graph.cells.iter_mut() {
            if cell.armed_waits.is_empty() {
                continue;
            }
            let completed = prune_completed_waits(cell);
            if completed == u64::MAX || cell.armed_waits.is_empty() {
                continue;
            }
            for aw in &cell.armed_waits {
                if let Some(s) = cell.submitted_signals.iter().find(|s| s.val >= aw.val) {
                    edges.push((aw.waiting_queue, aw.wait_seq, s.queue, s.seq, Some((va, aw.val))));
                }
            }
        }
        for (&tf_raw, other_tf) in graph.title_fences.iter_mut() {
            if tf_raw == title_fence {
                continue;
            }
            prune_title_fence(tf_raw, other_tf);
            for &(val, wq, wseq) in &other_tf.waits {
                if let Some(s) = other_tf.signals.iter().find(|s| s.val >= val) {
                    edges.push((wq, wseq, s.queue, s.seq, None));
                }
            }
        }
        // BFS carrying the first Wait64 SyncCell edge encountered on the path from `sig_q`.
        let mut best_limit: HashMap<usize, u64> = HashMap::new();
        let mut stack: Vec<(usize, u64, Option<(u64, u64)>)> = vec![(sig_q, u64::MAX, None)];
        best_limit.insert(sig_q, u64::MAX);
        let mut break_cells: Vec<(u64, u64)> = Vec::new();
        while let Some((cur_q, limit_seq, first_cell)) = stack.pop() {
            for &(wq, wseq, pq, pseq, cell_edge) in &edges {
                if wq != cur_q || wseq >= limit_seq {
                    continue;
                }
                let tracked = first_cell.or(cell_edge);
                if target_waiters.iter().any(|&(twq, twseq)| twq == pq && twseq < pseq) {
                    if let Some((va, val)) = tracked {
                        if !break_cells.iter().any(|&(bva, bval)| bva == va && bval >= val) {
                            break_cells.push((va, val));
                        }
                    }
                    continue;
                }
                let prev = best_limit.get(&pq).copied().unwrap_or(0);
                if pseq > prev {
                    best_limit.insert(pq, pseq);
                    stack.push((pq, pseq, tracked));
                }
            }
        }
        let mut out = Vec::new();
        for (va, val) in break_cells {
            if let Some(cell) = graph.cells.get_mut(&va) {
                if cell.non_monotonic {
                    continue;
                }
                cell.cpu_signaled_max = cell.cpu_signaled_max.max(val);
                let before = cell.armed_waits.len();
                cell.armed_waits.retain(|aw| aw.val > val);
                let removed = before - cell.armed_waits.len();
                if removed > 0 {
                    ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
                }
                cell.submitted_signals.retain(|s| s.val > val);
                if let Some(ref f) = cell.fence {
                    out.push((va, val, f.clone()));
                }
            }
        }
        out
    };
    for (va, val, fence) in to_release {
        publish_cpu_cycle_break(va, val);
        let _ = fence.Signal(val);
        static MIXED_WARN: AtomicU64 = AtomicU64::new(0);
        if once(&MIXED_WARN, 16) {
            crate::xlog_warn!(
                "D3D12.X: Wait64 fence {va:#x} >= {val:#x} released on CPU before title queue q{} Signal({title_fence:#x}, {sig_val:#x}) to break mixed GPU cycle",
                queue_id(sig_q)
            );
        }
    }
}

/// Whether an armed GPU wait has outlasted the watchdog; its clock stands while a submit or bind is held for a pipeline build.
fn wait_expired(aw: &mut ArmedGpuWait, held: bool) -> bool {
    if held {
        aw.armed_at = Instant::now();
    }
    aw.armed_at.elapsed() >= gpu_wait_watchdog_timeout()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExpiredGpuWait {
    val: u64,
    waiting_queue: usize,
    waiting_queue_ty: i32,
}

/// Remove elapsed waits for diagnostics without advancing any fence state. An elapsed wall-clock
/// timeout does not prove that the producer's work completed and must never synthesize completion.
fn take_expired_gpu_wait(cell: &mut SyncCell, held: bool) -> Option<ExpiredGpuWait> {
    let mut expired = None;
    cell.armed_waits.retain_mut(|aw| {
        if !wait_expired(aw, held) {
            return true;
        }
        if expired.is_none_or(|old: ExpiredGpuWait| aw.val >= old.val) {
            expired = Some(ExpiredGpuWait {
                val: aw.val,
                waiting_queue: aw.waiting_queue,
                waiting_queue_ty: aw.waiting_queue_ty,
            });
        }
        false
    });
    expired
}

/// `XCL_WAIT_TIMEOUT_CONTINUE=1` (off unless set): a wait that times out with no proven producer
/// does not stop the title. The GPU queue wait is released on the CPU (its fence is signalled to
/// the value waited for, nvidia-dev's watchdog before the merge) and a CPU fallback wait goes on;
/// what the title then reads may be stale. Default: the title stops (`stop_for_unresolved_wait`).
fn wait_timeout_continues() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| wait_timeout_continues_from(std::env::var("XCL_WAIT_TIMEOUT_CONTINUE").ok().as_deref()))
}

fn wait_timeout_continues_from(v: Option<&str>) -> bool {
    v.is_some_and(|v| v.trim() == "1")
}

/// `XCL_WAIT_TIMEOUT_CONTINUE`: the expired wait at `val` on `cell` counts as signalled on the
/// CPU (a later GPU signal at or below it no longer stands, and a rewind is guarded against);
/// the fence to signal to `val`, if the cell has one.
fn release_expired_on_cpu(cell: &mut SyncCell, val: u64) -> Option<ID3D12Fence> {
    cell.cpu_signaled_max = cell.cpu_signaled_max.max(val);
    cell.submitted_signals.retain(|s| s.val > val);
    cell.fence.clone()
}

fn stop_for_unresolved_wait(reason: &str) -> ! {
    let message = format!("{reason} – the title is stopped before its remaining GPU submission");
    crate::xlog_error!("{message}");
    crate::report::write_final(Some(message));
    super::real::end_title(0x8000_4005)
}

fn wake_sync_watchdog() {
    let pair = sync_watchdog_pair(&SYNC_WATCHDOG_WAKE, || {
        let _ = std::thread::Builder::new().name("xcl-sync-watchdog".into()).spawn(|| loop {
            let (lock, cv) = SYNC_WATCHDOG_WAKE.get().unwrap();
            let guard = lock.lock().unwrap();
            if ARMED_GPU_WAITS.load(Ordering::Acquire) == 0 {
                let _g = cv.wait(guard).unwrap();
                continue;
            }
            let _ = cv.wait_timeout(guard, Duration::from_millis(50)).unwrap();
            let (expired, rewinds) = {
                let mut g = SYNC_GRAPH.lock().unwrap();
                // the graph as it is while the waits stand (what is released below changes it at once)
                static BEFORE: AtomicU64 = AtomicU64::new(0);
                if let Some(graph) = g.as_ref() {
                    let timeout = gpu_wait_watchdog_timeout();
                    let late = graph.cells.values().any(|c| {
                        let done = c.fence.as_ref().map_or(0, |f| unsafe { super::real::fence_completed(f) });
                        c.armed_waits.iter().any(|w| w.val > done && w.armed_at.elapsed() >= timeout)
                    });
                    if late && BEFORE.fetch_add(1, Ordering::Relaxed) < 3 {
                        crate::xlog_warn!("D3D12.X: wait edges before the release: {}", edges_of(graph));
                        static DUMPS: AtomicU64 = AtomicU64::new(0);
                        dump_sync_trace("before the watchdog's release", &DUMPS, 2);
                    }
                }
                let mut out: Vec<(u64, u64, usize, i32, Option<usize>, Option<i32>, u64, u64, u64)> = Vec::new();
                let mut rew: Vec<(ID3D12Fence, u64)> = Vec::new();
                let continues = wait_timeout_continues();
                if let Some(graph) = g.as_mut() {
                    for (&va, cell) in graph.cells.iter_mut() {
                        if cell.armed_waits.is_empty() && cell.cpu_signaled_max == 0 {
                            continue;
                        }
                        let completed = unsafe { prune_completed_waits(cell) };
                        if completed == u64::MAX {
                            continue;
                        }
                        // Guard against vkd3d-proton rewinding `fence->virtual_value` when an older
                        // in-flight GPU signal retires after a CPU cycle-break/watchdog signal.
                        if cell.cpu_signaled_max > completed {
                            if let Some(ref f) = cell.fence {
                                rew.push((f.clone(), cell.cpu_signaled_max));
                            }
                        }
                        if cell.armed_waits.is_empty() {
                            continue;
                        }
                        let before = cell.armed_waits.len();
                        let held = super::pso_async::title_held();
                        let expired = take_expired_gpu_wait(cell, held);
                        let removed = before - cell.armed_waits.len();
                        if removed > 0 {
                            ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
                        }
                        if let Some(expired) = expired {
                            if continues {
                                if let Some(f) = release_expired_on_cpu(cell, expired.val) {
                                    rew.push((f, expired.val));
                                }
                            }
                            out.push((
                                va,
                                expired.val,
                                expired.waiting_queue,
                                expired.waiting_queue_ty,
                                cell.submitted_queue.or(cell.recorded_last_queue),
                                cell.recorded_list_type.map(|t| t.0),
                                cell.recorded_max,
                                cell.submitted_max,
                                completed,
                            ));
                        }
                    }
                }
                (out, rew)
            };
            for (fence, val) in rewinds {
                unsafe {
                    let _ = fence.Signal(val);
                }
            }
            let had_expired = !expired.is_empty();
            if had_expired {
                super::indirect_guard::report_now();
            }
            for &(va, val, waiter_q, waiter_ty, prod_q, prod_ty, rec, sub, done) in &expired {
                let wq = queue_id(waiter_q);
                let pq = prod_q.map(queue_id);
                crate::xlog_error!(
                    "D3D12.X: GPU queue.Wait at {va:#x} >= {val:#x} did not complete within {} ms (waiter=q{wq}/ty{waiter_ty}, prod=q{pq:?}/ty{prod_ty:?}, recorded={rec:#x}, submitted={sub:#x}, completed={done:#x})",
                    gpu_wait_watchdog_timeout().as_millis()
                );
            }
            if had_expired {
                crate::xlog_warn!("D3D12.X: pending title ID3D12Fences during GPU wait timeout: {}", super::fence_x::pending_summary());
                crate::xlog_warn!("D3D12.X: XCL's own queue waits unmet during GPU wait timeout: {}", internal_waits_summary());
                crate::xlog_warn!("D3D12.X: queues behind during GPU wait timeout: {}", progress_summary());
                static PRINTED: AtomicU64 = AtomicU64::new(0);
                if PRINTED.fetch_add(1, Ordering::Relaxed) < 3 {
                    crate::xlog_warn!("D3D12.X: wait edges at that time-out: {}", edges_summary());
                }
                if wait_timeout_continues() {
                    crate::xlog_warn!("D3D12.X: XCL_WAIT_TIMEOUT_CONTINUE=1 – the timed-out GPU queue waits were released on the CPU and the title goes on");
                } else {
                    stop_for_unresolved_wait("a GPU queue wait exceeded its watchdog without a graph-proven cycle");
                }
            }
        });
    });
    pair.1.notify_one();
}

/// The watchdog's wake pair. The first call makes it and then starts the thread with `spawn`:
/// the thread may run before `spawn` returns, and it needs the pair first.
fn sync_watchdog_pair(wake: &OnceLock<(Mutex<()>, Condvar)>, spawn: impl FnOnce()) -> &(Mutex<()>, Condvar) {
    let mut first = false;
    let pair = wake.get_or_init(|| {
        first = true;
        (Mutex::new(()), Condvar::new())
    });
    if first {
        spawn();
    }
    pair
}

unsafe fn queue_device_and_type(queue: *mut c_void) -> Option<(ID3D12Device, usize, D3D12_COMMAND_LIST_TYPE)> {
    if queue.is_null() {
        return None;
    }
    let q = ID3D12CommandQueue::from_raw_borrowed(&queue)?;
    let ty = q.GetDesc().Type;
    let mut dev: Option<ID3D12Device> = None;
    q.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let raw = dev.as_raw() as usize;
    Some((dev, raw, ty))
}

unsafe fn ensure_cell_fence(va: u64, dev: &ID3D12Device, dev_raw: usize) -> Option<ID3D12Fence> {
    {
        let g = SYNC_GRAPH.lock().unwrap();
        if let Some(cell) = g.as_ref().and_then(|gr| gr.cells.get(&va)) {
            if cell.non_monotonic {
                return None;
            }
            if let Some(ref f) = cell.fence {
                return (cell.device == dev_raw).then(|| f.clone());
            }
        }
    }
    let created: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
    let mut g = SYNC_GRAPH.lock().unwrap();
    let cell = g.get_or_insert_with(SyncGraph::default).cells.entry(va).or_default();
    if cell.non_monotonic {
        return None;
    }
    if let Some(ref existing) = cell.fence {
        return (cell.device == dev_raw).then(|| existing.clone());
    }
    cell.fence = Some(created.clone());
    cell.device = dev_raw;
    Some(created)
}

fn note_write64_recorded(list: *mut c_void, va: u64, value: u64) {
    if value == 0 || !va.is_multiple_of(8) {
        return;
    }
    // (the cell is the CPU's: a wait on the other view of the same memory meets it)
    let va = cpu_at(va);
    let (lty, last_q, recording) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        s.push_write64_signal(va, value);
        let recording = if sync_state_tracing() {
            Some(s.write64_trace.get_or_insert_with(Default::default).clone())
        } else {
            None
        };
        (s.list_type, s.last_queue, recording)
    };
    let rescue: Option<(ID3D12Fence, u64, u64)> = {
        let mut g = SYNC_GRAPH.lock().unwrap();
        let cell = g.get_or_insert_with(SyncGraph::default).cells.entry(va).or_default();
        if cell.non_monotonic {
            return;
        }
        // Strictly decreasing write (`value < submitted_max`) marks this address non-monotonic.
        // Duplicate same-value writes from parallel command-list recording threads remain monotonic.
        if value < cell.submitted_max {
            let max_armed = cell.armed_waits.iter().map(|aw| aw.val).max().unwrap_or(0);
            let removed = cell.armed_waits.len();
            if removed > 0 {
                cell.armed_waits.clear();
                ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
            }
            cell.submitted_signals.clear();
            cell.non_monotonic = true;
            let prev_sub = cell.submitted_max;
            let old_fence = cell.fence.take();
            old_fence.filter(|_| max_armed > 0).map(|f| (f, max_armed, prev_sub))
        } else {
            if value > cell.recorded_max {
                cell.recorded_max = value;
                cell.recorded_by = recording.map(|recording| (list as usize, Instant::now(), recording));
                if lty.is_some() {
                    cell.recorded_list_type = lty;
                }
                if last_q.is_some() {
                    cell.recorded_last_queue = last_q;
                }
            }
            None
        }
    };
    // Rescue any armed GPU waits BEFORE dropping the fence so no vkd3d_queue thread can be orphaned.
    if let Some((old_fence, max_armed, prev_sub)) = rescue {
        unsafe {
            let _ = old_fence.Signal(max_armed);
        }
        static NON_MONO_WARN: AtomicU64 = AtomicU64::new(0);
        if once(&NON_MONO_WARN, 16) {
            crate::xlog_warn!(
                "D3D12.X: Write64 at {va:#x} became non-monotonic ({value:#x} < submitted {prev_sub:#x}) – released armed GPU waits up to {max_armed:#x} on CPU"
            );
        }
    }
}

unsafe fn signal_write64_pairs(queue: *mut c_void, pairs: &[(u64, u64)]) {
    if queue.is_null() || pairs.is_empty() {
        return;
    }
    let Some((dev, dev_raw, q_ty)) = queue_device_and_type(queue) else { return };
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let sig_q = queue as usize;
    // Coalesce per-VA max value within this batch.
    let mut max_by_va: Vec<(u64, u64)> = Vec::with_capacity(pairs.len());
    for &(va, val) in pairs {
        if val == 0 {
            continue;
        }
        if let Some(entry) = max_by_va.iter_mut().find(|e| e.0 == va) {
            entry.1 = entry.1.max(val);
        } else {
            max_by_va.push((va, val));
        }
    }
    for (va, val) in max_by_va {
        let skip = {
            let g = SYNC_GRAPH.lock().unwrap();
            g.as_ref()
                .and_then(|gr| gr.cells.get(&va))
                .is_some_and(|c| c.non_monotonic || c.submitted_max >= val)
        };
        if skip {
            continue;
        }
        let Some(fence) = ensure_cell_fence(va, &dev, dev_raw) else { continue };
        enum SignalAction {
            Skip,
            GpuSignal,
            CpuCycleBreak(Vec<(usize, bool)>),
        }
        let action = {
            let mut g = SYNC_GRAPH.lock().unwrap();
            let graph = g.get_or_insert_with(SyncGraph::default);
            let eligible = match graph.cells.get_mut(&va) {
                Some(cell) => {
                    let completed = prune_completed_waits(cell);
                    if cell.non_monotonic || cell.device != dev_raw || val <= cell.submitted_max {
                        None
                    } else {
                        Some(completed)
                    }
                }
                None => None,
            };
            match eligible {
                None => SignalAction::Skip,
                Some(completed) => {
                    if would_signal_close_cycle(graph, sig_q, va, val) {
                        let cell = graph.cells.get_mut(&va).unwrap();
                        let waiters: Vec<(usize, bool)> =
                            cell.armed_waits.iter().filter(|aw| aw.val <= val).map(|aw| (aw.waiting_queue, aw.after_write64)).collect();
                        cell.submitted_max = val;
                        cell.submitted_queue = None;
                        cell.cpu_signaled_max = cell.cpu_signaled_max.max(val);
                        if val > cell.recorded_max {
                            cell.recorded_max = val;
                        }
                        let before = cell.armed_waits.len();
                        cell.armed_waits.retain(|aw| aw.val > val);
                        let removed = before - cell.armed_waits.len();
                        if removed > 0 {
                            ARMED_GPU_WAITS.fetch_sub(removed, Ordering::AcqRel);
                        }
                        cell.submitted_signals.retain(|s| s.val > val);
                        SignalAction::CpuCycleBreak(waiters)
                    } else {
                        let sig_seq = graph.next_seq(sig_q);
                        let cell = graph.cells.get_mut(&va).unwrap();
                        cell.submitted_max = val;
                        cell.submitted_queue = Some(sig_q);
                        if val > cell.recorded_max {
                            cell.recorded_max = val;
                        }
                        if completed < val && cell.cpu_signaled_max < val {
                            cell.submitted_signals.push(SubmittedSignal { val, queue: sig_q, seq: sig_seq });
                            SignalAction::GpuSignal
                        } else {
                            SignalAction::Skip
                        }
                    }
                }
            }
        };
        sync_trace(|| {
            let how = match &action {
                SignalAction::Skip => "nobody needs it".to_string(),
                SignalAction::GpuSignal => "SIGNAL".to_string(),
                SignalAction::CpuCycleBreak(w) => format!("CYCLE – signalled on the CPU, waiters {:?}", w.iter().map(|x| queue_id(x.0)).collect::<Vec<_>>()),
            };
            let seq = SYNC_GRAPH.lock().unwrap().as_ref().and_then(|g| g.cells.get(&va)).and_then(|c| c.submitted_signals.iter().rev().find(|x| x.queue == sig_q && x.val == val).map(|x| x.seq));
            format!("q{}@{} signal {va:#x} = {val:#x}: {how}", queue_id(sig_q), seq.map_or("-".to_string(), |x| x.to_string()))
        });
        if matches!(action, SignalAction::CpuCycleBreak(_)) {
            static DUMPS: AtomicU64 = AtomicU64::new(0);
            dump_sync_trace("cycle break", &DUMPS, 3);
        }
        match action {
            SignalAction::Skip => {}
            SignalAction::GpuSignal => {
                if let Err(e) = super::real::native_queue_signal(q, &fence, val) {
                    crate::xlog_warn!("D3D12.X: Write64 fence Signal({va:#x}, {val:#x}) failed ({e}) – signalling on CPU");
                    let _ = fence.Signal(val);
                }
            }
            SignalAction::CpuCycleBreak(waiters) => {
                publish_cpu_cycle_break(va, val);
                let _ = fence.Signal(val);
                static BREAK_WARN: AtomicU64 = AtomicU64::new(0);
                if once(&BREAK_WARN, 32) {
                    let sq = queue_id(sig_q);
                    let mut wids: Vec<String> = waiters
                        .into_iter()
                        .map(|(w, aw)| if aw { format!("q{}(after_w64)", queue_id(w)) } else { format!("q{}", queue_id(w)) })
                        .collect();
                    wids.sort();
                    wids.dedup();
                    crate::xlog_warn!(
                        "D3D12.X: Write64 Signal at {va:#x} = {val:#x} on q{sq}/ty{} would close a GPU cycle (waiters=[{}]) – signalled on CPU",
                        q_ty.0,
                        wids.join(",")
                    );
                }
            }
        }
    }
}

/// After `queue.ExecuteCommandLists(lists)`: signal per-address GPU fences for any
/// `Write64BitValue{Bottom,Top}OfPipeX` commands recorded on `lists`.
pub(super) unsafe fn signal_write64_fences(queue: *mut c_void, lists: &[*mut c_void]) {
    if WRITE64_SIGNAL_LISTS.load(Ordering::Acquire) == 0 || queue.is_null() || lists.is_empty() {
        return;
    }
    for &l in lists {
        trace_list(l as usize, ListEvent::Executed);
    }
    let pairs: Vec<(u64, u64)> = {
        let mut out = Vec::new();
        for &l in lists {
            if let Some(s) = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(l as usize))) {
                out.extend_from_slice(&s.write64_signals);
            }
        }
        out
    };
    signal_write64_pairs(queue, &pairs);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GpuWaitOutcome {
    /// Satisfied on GPU (via `queue.Wait` or same-queue FIFO order) — no CPU wait needed.
    Handled,
    /// Skipping this wait avoids a same-batch self-deadlock.
    CycleSkip,
    /// Not eligible for GPU fence bridging — fall back to bounded CPU poll.
    FallbackCpu,
}

fn wait_is_resolved(outcome: GpuWaitOutcome, cpu_satisfied: bool) -> bool {
    match outcome {
        GpuWaitOutcome::Handled | GpuWaitOutcome::CycleSkip => true,
        GpuWaitOutcome::FallbackCpu => cpu_satisfied,
    }
}

/// Attempt to satisfy a `Wait64BitValueX(GREATER_EQUAL, u64::MAX)` on the GPU via
/// `queue.Wait(&fence, w.value)` (or same-queue FIFO ordering) instead of blocking the
/// submission thread on the CPU.
///
/// Cycle breaking is performed with 100% precision at `Signal` time (`signal_write64_pairs` /
/// `break_cycles_before_title_signal`) rather than guessing producer queues from previous frames,
/// so rotating multi-queue schedules neither skip valid barriers nor form Vulkan timeline cycles.
unsafe fn try_gpu_wait(queue: *mut c_void, w: &PendingWait, list_writes: &[(u64, u64)]) -> GpuWaitOutcome {
    if queue.is_null()
        || w.width != 8
        || w.comparison != D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32
        || w.mask != u64::MAX
        || w.value == 0
    {
        return GpuWaitOutcome::FallbackCpu;
    }
    let Some((dev, dev_raw, q_ty)) = queue_device_and_type(queue) else { return GpuWaitOutcome::FallbackCpu };
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return GpuWaitOutcome::FallbackCpu };
    let q_raw = queue as usize;

    // Fast path: check eligibility, CPU-released bypass, and same-queue FIFO before allocating a fence.
    let pre = {
        let g = SYNC_GRAPH.lock().unwrap();
        g.as_ref().and_then(|gr| gr.cells.get(&w.va)).map(|c| {
            (c.non_monotonic, c.recorded_max, c.submitted_max, c.submitted_queue, c.cpu_signaled_max)
        })
    };
    let Some((non_mono, rec_max, sub_max, sub_q, cpu_sig_max)) = pre else {
        return GpuWaitOutcome::FallbackCpu;
    };
    if non_mono || rec_max < w.value {
        return GpuWaitOutcome::FallbackCpu;
    }
    // Already released on CPU (cycle break or watchdog): never re-arm a GPU wait that could
    // observe a vkd3d-proton fence rewind.
    if cpu_sig_max >= w.value {
        sync_trace(|| format!("q{} wait {:#x} >= {:#x}: released on the CPU before", queue_id(q_raw), w.va, w.value));
        return GpuWaitOutcome::Handled;
    }
    // Producer already submitted earlier on THIS EXACT queue: D3D12 FIFO executes it strictly
    // before this list without needing a CPU wait or redundant queue.Wait.
    if sub_max >= w.value && sub_q == Some(q_raw) {
        sync_trace(|| format!("q{} wait {:#x} >= {:#x}: this queue signals {sub_max:#x} before", queue_id(q_raw), w.va, w.value));
        return GpuWaitOutcome::Handled;
    }
    // Same-batch self-wait: this very submission writes `(w.va, >= w.value)`, so hoisting a GPU
    // Wait ahead of it on the same queue would immediately self-deadlock.
    if sub_max < w.value && list_writes.iter().any(|&(wva, wv)| wva == w.va && wv >= w.value) {
        static SELF_WARN: AtomicU64 = AtomicU64::new(0);
        if once(&SELF_WARN, 8) {
            crate::xlog_warn!(
                "D3D12.X: Wait64BitValueX at {:#x} >= {:#x} skipped to avoid same-batch self-deadlock (q{}/ty{})",
                w.va,
                w.value,
                queue_id(q_raw),
                q_ty.0
            );
        }
        return GpuWaitOutcome::CycleSkip;
    }

    let Some(fence) = ensure_cell_fence(w.va, &dev, dev_raw) else { return GpuWaitOutcome::FallbackCpu };

    let (unsubmitted, final_rec_ty, final_prod_q, wait_seq) = {
        let mut g = SYNC_GRAPH.lock().unwrap();
        let Some(graph) = g.as_mut() else { return GpuWaitOutcome::FallbackCpu };
        let pre_ok = match graph.cells.get_mut(&w.va) {
            Some(cell) => {
                if cell.non_monotonic || cell.device != dev_raw || cell.recorded_max < w.value {
                    return GpuWaitOutcome::FallbackCpu;
                }
                if cell.cpu_signaled_max >= w.value {
                    return GpuWaitOutcome::Handled;
                }
                let completed = prune_completed_waits(cell);
                if completed >= w.value {
                    // Fence already reached `w.value`: trust it if CPU memory agrees (not reset).
                    drop(g);
                    super::sparse::settle_readbacks();
                    return if w.is_satisfied() { GpuWaitOutcome::Handled } else { GpuWaitOutcome::FallbackCpu };
                }
                if cell.submitted_max >= w.value && cell.submitted_queue == Some(q_raw) {
                    return GpuWaitOutcome::Handled;
                }
                (cell.submitted_max < w.value, cell.recorded_list_type, cell.submitted_queue.or(cell.recorded_last_queue))
            }
            None => return GpuWaitOutcome::FallbackCpu,
        };
        let wait_seq = graph.next_seq(q_raw);
        let cell = graph.cells.get_mut(&w.va).unwrap();
        cell.armed_waits.push(ArmedGpuWait {
            val: w.value,
            waiting_queue: q_raw,
            waiting_queue_ty: q_ty.0,
            wait_seq,
            armed_at: Instant::now(),
            after_write64: w.after_write64,
        });
        ARMED_GPU_WAITS.fetch_add(1, Ordering::AcqRel);
        (pre_ok.0, pre_ok.1, pre_ok.2, wait_seq)
    };
    sync_trace(|| {
        format!(
            "q{}@{wait_seq} WAIT {:#x} >= {:#x} (submitted {sub_max:#x} by {:?}{}{})",
            queue_id(q_raw),
            w.va,
            w.value,
            sub_q.map(queue_id),
            if unsubmitted { ", not signalled yet" } else { "" },
            if w.after_write64 { ", after a Write64 of its list" } else { "" }
        )
    });

    if let Err(e) = super::real::native_queue_wait(q, &fence, w.value) {
        let mut g = SYNC_GRAPH.lock().unwrap();
        if let Some(cell) = g.as_mut().and_then(|gr| gr.cells.get_mut(&w.va)) {
            if let Some(pos) = cell.armed_waits.iter().rposition(|aw| aw.val == w.value && aw.waiting_queue == q_raw && aw.wait_seq == wait_seq) {
                cell.armed_waits.remove(pos);
                ARMED_GPU_WAITS.fetch_sub(1, Ordering::AcqRel);
            }
        }
        crate::xlog_warn!("D3D12.X: Wait64 GPU queue.Wait({:#x}, {:#x}) failed ({e}) – falling back to CPU wait", w.va, w.value);
        return GpuWaitOutcome::FallbackCpu;
    }

    note_cell_wait(q_raw, w.va, &fence, w.value);
    wake_sync_watchdog();
    if unsubmitted {
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        if once(&LOGGED, 8) {
            crate::xlog_info!(
                "D3D12.X: Wait64BitValueX at {:#x} >= {:#x} bridged to GPU queue.Wait (waiter=q{}/ty{}, prod=q{:?}/ty{:?}; wait-before-signal, after_write64={})",
                w.va,
                w.value,
                queue_id(q_raw),
                q_ty.0,
                final_prod_q.map(queue_id),
                final_rec_ty.map(|t| t.0),
                w.after_write64
            );
        }
    }
    GpuWaitOutcome::Handled
}

#[cfg(test)]
unsafe fn wait_all_on_with_timeout(
    queue: *mut c_void,
    waits: &[PendingWait],
    list_writes: &[(u64, u64)],
    timeout: Duration,
) -> bool {
    wait_all_on_noting_cpu(queue, waits, list_writes, timeout).0
}

/// `wait_all_on_with_timeout`, and whether a wait was waited for on the CPU.
unsafe fn wait_all_on_noting_cpu(
    queue: *mut c_void,
    waits: &[PendingWait],
    list_writes: &[(u64, u64)],
    timeout: Duration,
) -> (bool, bool) {
    static TIMEOUTS: AtomicU64 = AtomicU64::new(0);
    static RING_DUMPED: AtomicU64 = AtomicU64::new(0);
    let q_ty = queue_device_and_type(queue).map_or(-1, |(_, _, t)| t.0);
    let mut on_cpu = false;
    for w in waits {
        if w.is_satisfied() {
            sync_trace(|| format!("q{} wait {:#x} >= {:#x}: memory holds it", queue_id(queue as usize), w.va, w.value));
            continue;
        }
        let outcome = try_gpu_wait(queue, w, list_writes);
        if wait_is_resolved(outcome, false) {
            continue;
        }
        on_cpu = true;
        sync_trace(|| format!("q{} wait {:#x} (op {}) {:#x}: on the CPU", queue_id(queue as usize), w.va, w.comparison, w.value));
        let memory_stall = crate::stall::span(crate::stall::StallKind::MemorySynchronization);
        let satisfied = w.wait_until_satisfied(timeout);
        drop(memory_stall);
        if wait_is_resolved(outcome, satisfied) {
            continue;
        }
        if once(&TIMEOUTS, 40) {
            let cell_diag = {
                let g = SYNC_GRAPH.lock().unwrap();
                g.as_ref().and_then(|gr| gr.cells.get(&w.va)).map(|c| (c.recorded_max, c.submitted_max, c.non_monotonic, c.recorded_list_type.map(|t| t.0), c.submitted_queue.map(queue_id)))
            };
            let behind = progress_summary();
            crate::xlog_warn!(
                "D3D12.X: Wait{}BitValueX timed out after {} ms at {:#x} (op {}, want {:#x}, mask {:#x}; memory holds {:x?}; queue_type={}, after_write64={}, sync={cell_diag:x?})",
                w.width * 8,
                timeout.as_millis(),
                w.va,
                w.comparison,
                w.value,
                w.mask,
                w.read_value(),
                q_ty,
                w.after_write64
            );
            if !behind.is_empty() {
                crate::xlog_warn!("D3D12.X: queues behind at that time-out: {behind}");
            }
            if once(&RING_DUMPED, 2) {
                let (total, entries) = WRITE_TRACE.lock().unwrap().snapshot();
                let summary: Vec<String> = entries
                    .iter()
                    .rev()
                    .take(16)
                    .map(|e| format!("{}({:#x}={:#x},f={:#x},ty={})", e.what, e.va, e.value, e.flags, e.list_type))
                    .collect();
                crate::xlog_warn!("D3D12.X: recent Write32/64 calls ({total} total): [{}]", summary.join(", "));
            }
        }
        return (false, on_cpu);
    }
    (true, on_cpu)
}

/// The waits of a list before it goes: on the GPU where they can be, else on the CPU. Whether one
/// was waited for on the CPU.
unsafe fn wait_all_on(queue: *mut c_void, waits: &[PendingWait], list_writes: &[(u64, u64)]) -> bool {
    let (resolved, on_cpu) = wait_all_on_noting_cpu(queue, waits, list_writes, wait_timeout());
    // (`XCL_WAIT_TIMEOUT_CONTINUE=1`: the list goes out after the time-out, as before the merge on nvidia-dev)
    if !resolved && !wait_timeout_continues() {
        stop_for_unresolved_wait("a CPU fallback memory wait timed out without a proven completion");
    }
    on_cpu
}

/// Copy any GPU memory writes into sparse VRAM windows recorded by `lists` back into their
/// CPU view after `queue.ExecuteCommandLists` so CPU reads and `Wait32/64BitValueX` see them.
pub(super) unsafe fn readback_sparse_writes(queue: *mut c_void, lists: &[*mut c_void]) {
    if SPARSE_WRITE_LISTS.load(Ordering::Acquire) == 0 || queue.is_null() || lists.is_empty() {
        return;
    }
    let (ranges, imms): (Vec<(usize, usize)>, Vec<(usize, u64, u8)>) = {
        let (mut out, mut imm) = (Vec::new(), Vec::new());
        for &l in lists {
            if let Some(s) = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(l as usize))) {
                out.extend_from_slice(&s.sparse_writes);
                imm.extend_from_slice(&s.sparse_imm);
            }
        }
        (out, imm)
    };
    if ranges.is_empty() && imms.is_empty() {
        return;
    }
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        // (lock order: the queue's submit lease, then the copy back's submit lock; the lease is
        // re-entrant, so a caller already inside it keeps its turn)
        super::queue_submit::with(queue as usize, || super::sparse::readback_after_execute(q, &ranges, &imms));
    }
}

/// After `lists` went to `queue`: their writes into memory the GPU does not see (`cpu_writes`).
pub(super) unsafe fn cpu_writes_after(queue: *mut c_void, lists: &[*mut c_void]) {
    let writes: Vec<super::cpu_writes::Write> = {
        let of = |&l: &*mut c_void| LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(l as usize))).map(|s| s.cpu_writes.clone());
        lists.iter().filter_map(of).flatten().collect()
    };
    super::cpu_writes::after(queue, writes);
}

/// After a wait on the CPU inside a submit: what the title committed meanwhile (the thread that
/// writes the value may have) is mapped, and cleared, before the lists that go out now – the
/// submit's own `before_execute` came before the wait.
unsafe fn mapped_after_wait(queue: *mut c_void) {
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        // (fails closed as the submit's own preparation does: `real::prepare_title_submission`)
        match super::sparse::before_execute(q, &[], 0x1000, None) {
            super::sparse::PreparationOutcome::Ready => {}
            super::sparse::PreparationOutcome::RetryRequired => stop_for_unresolved_wait("sparse upload preparation after a wait inside a submit still failed after bounded retries"),
            super::sparse::PreparationOutcome::ExecutedUntracked => stop_for_unresolved_wait("a sparse upload after a wait inside a submit executed without a retirement signal"),
        }
    }
}

/// `XCL_SPLIT_SPARSE=cpu` (off unless set): after a split at waits that all went to the GPU (or
/// held already), the sparse windows are not brought up to date again (`mapped_after_wait`): the
/// lists before and after the split went out within the submit, as an unsplit submit's do. A
/// retail title's world split many of its ~19 submits a frame at Wait64BitValueX waits bridged to
/// the GPU, and each split took the sparse state's lock again.
fn split_sparse_after_cpu_waits_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPLIT_SPARSE").is_ok_and(|v| v.trim().eq_ignore_ascii_case("cpu")))
}

/// Whether a split, whose waits were (`on_cpu`) or were not waited for on the CPU, brings the
/// sparse windows up to date again (`only_after_cpu`: `XCL_SPLIT_SPARSE=cpu`).
fn sparse_again_after_split(on_cpu: bool, only_after_cpu: bool) -> bool {
    on_cpu || !only_after_cpu
}

/// Capture every mutable per-list effect once, before any part of this submission reaches the
/// driver. Resetting or re-recording a list after this point cannot change the returned plan.
pub(super) fn submission_plan(lists: &[*mut c_void]) -> SubmissionPlan {
    SubmissionPlan::capture(lists)
}

/// Publish the effects of an already-submitted chunk in the same order as the legacy consumers.
unsafe fn submit_segment_effects(queue: *mut c_void, effects: &[SegmentEffects], needs_cpu_completion: bool) -> Option<super::cpu_writes::Completion> {
    let texture_writes: Vec<_> = effects.iter().flat_map(|e| e.texture_writes.iter().copied()).collect();
    let texture_transitions: Vec<_> = effects.iter().flat_map(|e| e.texture_transitions.iter().copied()).collect();
    super::title_textures::publish_submission_effects(queue_identity(queue as usize), &texture_writes, &texture_transitions);
    let write64_signals: Vec<(u64, u64)> = effects.iter().flat_map(|e| e.write64_signals.iter().copied()).collect();
    signal_write64_pairs(queue, &write64_signals);

    let sparse_writes: Vec<(usize, usize)> = effects.iter().flat_map(|e| e.sparse_writes.iter().copied()).collect();
    let sparse_imm: Vec<(usize, u64, u8)> = effects.iter().flat_map(|e| e.sparse_imm.iter().copied()).collect();
    if !sparse_writes.is_empty() || !sparse_imm.is_empty() {
        if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
            // (lease before the copy back's submit lock, as in `readback_sparse_writes`)
            super::queue_submit::with(queue as usize, || super::sparse::readback_after_execute(q, &sparse_writes, &sparse_imm));
        }
    }

    let cpu_writes: Vec<super::cpu_writes::Write> = effects.iter().flat_map(|e| e.cpu_writes.iter().cloned()).collect();
    if needs_cpu_completion {
        super::cpu_writes::after_with_completion(queue, cpu_writes)
    } else {
        super::cpu_writes::after(queue, cpu_writes);
        None
    }
}

/// Execute a plan and retain completion receipts for its deferred CPU writes.
unsafe fn execute_submission_plan_with_completions(
    queue: *mut c_void,
    plan: &SubmissionPlan,
    external_waits: &[PendingWait],
    mut submit: impl FnMut(SubmissionChunk<'_>) -> (bool, Option<super::cpu_writes::Completion>),
) -> (Vec<super::cpu_writes::Completion>, bool) {
    let chunk = |start, end| SubmissionChunk {
        lists: &plan.lists[start..end],
        effects: &plan.effects[start..end],
        traces: if plan.traces.is_empty() { &[] } else { &plan.traces[start..end] },
        needs_cpu_completion: plan.chunk_needs_cpu_completion(start, end, external_waits),
    };
    if !plan.has_waits {
        let (submitted, completion) = submit(chunk(0, plan.lists.len()));
        return (completion.into_iter().collect(), submitted);
    }
    sync_trace(|| {
        let rows: Vec<String> = plan
            .lists
            .iter()
            .zip(&plan.effects)
            .map(|(l, e)| {
                let wrapper = super::title_list::wrapper_of(*l).map_or(0, |p| p as usize) & 0xFFFFF;
                let waits: Vec<String> = e.waits.iter().map(|p| format!("W{:x}>={:x}{}", p.va & 0xFFF, p.value, if p.is_satisfied() { "ok" } else { "" })).collect();
                let writes: Vec<String> = e.write64_signals.iter().map(|p| format!("S{:x}={:x}", p.0 & 0xFFF, p.1)).collect();
                format!("{:x}/{wrapper:x}[{} | {}]", *l as usize & 0xFFFFF, waits.join(" "), writes.join(" "))
            })
            .collect();
        format!("q{} EXECUTE {} lists: {}", queue_id(queue as usize), plan.lists.len(), rows.join(" "))
    });
    let mut completions = Vec::new();
    let mut start = 0usize;
    for (i, effects) in plan.effects.iter().enumerate() {
        if effects.waits.is_empty() {
            continue;
        }
        let after_cpu_write = plan.waits_after_cpu_write[i];
        if !after_cpu_write && effects.waits.iter().all(|w| w.is_satisfied()) {
            continue;
        }
        if i > start {
            sync_trace(|| format!("q{} submit lists {start}..{i}", queue_id(queue as usize)));
            let (submitted, completion) = submit(chunk(start, i));
            if !submitted {
                return (completions, false);
            }
            completions.extend(completion);
        }
        if after_cpu_write {
            for completion in completions.drain(..) {
                let _ = completion.wait();
            }
        }
        let t = Instant::now();
        let on_cpu = wait_all_on(queue, &effects.waits, &effects.write64_signals);
        let t_waited = Instant::now();
        if sparse_again_after_split(on_cpu, split_sparse_after_cpu_waits_only()) {
            mapped_after_wait(queue);
        }
        super::submit_cost::split(t_waited - t, t_waited.elapsed());
        start = i;
    }
    if start < plan.lists.len() {
        sync_trace(|| format!("q{} submit lists {start}..{}", queue_id(queue as usize), plan.lists.len()));
        let (submitted, completion) = submit(chunk(start, plan.lists.len()));
        if !submitted {
            return (completions, false);
        }
        completions.extend(completion);
    }
    (completions, true)
}

pub(super) unsafe fn execute_submission_plan_until_failure_on(
    queue: *mut c_void,
    plan: &SubmissionPlan,
    submit: impl FnMut(SubmissionChunk<'_>) -> (bool, Option<super::cpu_writes::Completion>),
) -> (Vec<super::cpu_writes::Completion>, bool) {
    execute_submission_plan_with_completions(queue, plan, &[], submit)
}

/// Execute a plan for an ordinary queue submission, retaining deferred CPU-write receipts.
pub(super) unsafe fn execute_submission_plan_with_completions_on(
    queue: *mut c_void,
    plan: &SubmissionPlan,
    mut submit: impl FnMut(SubmissionChunk<'_>) -> Option<super::cpu_writes::Completion>,
) -> Vec<super::cpu_writes::Completion> {
    execute_submission_plan_with_completions(queue, plan, &[], |chunk| (true, submit(chunk))).0
}

/// Execute an immutable submission plan, splitting before an unsatisfied memory wait.
pub(super) unsafe fn execute_submission_plan_on(queue: *mut c_void, plan: &SubmissionPlan, mut submit: impl FnMut(SubmissionChunk<'_>)) {
    let _ = execute_submission_plan_with_completions(queue, plan, &[], |chunk| {
        submit(chunk);
        (true, None)
    });
}

/// Submit `lists` through `submit`, splitting the batch before any list whose `Wait32/64BitValueX`
/// condition is not yet satisfied so earlier lists in the same batch can run on the GPU first.
/// When `queue` is non-null, monotonic 64-bit waits backed by a recorded `Write64BitValue*` are
/// bridged to GPU-side `queue.Wait` without blocking the submitter thread on the CPU.
pub(super) unsafe fn execute_with_waits_on(queue: *mut c_void, lists: &[*mut c_void], mut submit: impl FnMut(&[*mut c_void])) {
    let plan = submission_plan(lists);
    execute_submission_plan_on(queue, &plan, |chunk| submit(chunk.lists()));
}

#[cfg(test)]
pub(super) unsafe fn execute_with_waits(lists: &[*mut c_void], submit: impl FnMut(&[*mut c_void])) {
    execute_with_waits_on(std::ptr::null_mut(), lists, submit);
}

// ---------------------------------------------------------------------------------------
// memory copy / fill / immediate write / timestamp
// ---------------------------------------------------------------------------------------

/// command list #60 CopyMemoryToMemoryX(destination, source, size, flags)
pub(super) unsafe extern "system" fn copy_memory_x(this: *mut c_void, dst: u64, src: u64, size: u64, _flags: u32) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    static N: AtomicU64 = AtomicU64::new(0);
    const W: &str = "CopyMemoryToMemoryX";
    let refused = size == 0 || !size.is_multiple_of(4);
    let _c = super::capture::op(this, || {
        use super::capture as c;
        Some(c::Op::Copy { kind: "memory", dst: c::Loc::Address(dst), src: c::Loc::Address(src), at: [0; 3], bx: None, bytes: Some(size), refused })
    });
    if refused {
        return warn(&N, W, format_args!("size {size:#x} is not a non-zero multiple of 4"));
    }
    // In a GPU view: its tiles, asked for here, are filled from the CPU view before the list runs,
    // and what the copy writes is read back to the CPU view (`note_write`)
    super::bind::want_view(super::view_want::Path::MemoryOp, src, size);
    super::bind::want_view(super::view_want::Path::MemoryOp, dst, size);
    let (Some((d, doff)), Some((s, soff))) = (real::uma_window_buffer_gpu(dst as usize, size as usize), real::uma_window_buffer_gpu(src as usize, size as usize)) else {
        note_gpu_write("CopyMemoryToMemoryX", dst as usize, size);
        return warn(&N, W, format_args!("{src:#x} → {dst:#x} ({size:#x} bytes) is not in unified memory"));
    };
    note_gpu_write("CopyMemoryToMemoryX", dst as usize, size);
    if d == s && dst < src + size && src < dst + size {
        return warn(&N, W, format_args!("{src:#x} → {dst:#x} ({size:#x} bytes) overlap"));
    }
    note_write(this as usize, WriteProducer::InSegmentGpu, dst, size);
    if d == s {
        // one window: one buffer cannot be in COPY_SOURCE and COPY_DEST at once (#958), so the
        // bytes go through a scratch buffer of the list
        let Some((t, capacity)) = copy_scratch(this, size) else {
            // none to be had: within the buffer, as D3D12 does not define it (the ranges do not overlap)
            as_copy_dest(this, d, || copy_buffer_region(this, d, doff, s, soff, size));
            return;
        };
        // Later copies in this recording may need more than its first allocation. The ranges
        // are disjoint, so each chunk can reuse the same scratch after writing its destination.
        for offset in (0..size).step_by(capacity as usize) {
            let bytes = (size - offset).min(capacity);
            transition(this, s, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            as_copy_dest(this, t, || copy_buffer_region(this, t, 0, s, soff + offset, bytes));
            transition(this, s, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            transition(this, t, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            as_copy_dest(this, d, || copy_buffer_region(this, d, doff + offset, t, 0, bytes));
            transition(this, t, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        }
        static SCRATCH_COPIES: AtomicU64 = AtomicU64::new(0);
        let count = SCRATCH_COPIES.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_power_of_two() {
            crate::xlog_info!("D3D12.X: CopyMemoryToMemoryX same-window scratch copies recorded: {count}");
        }
        return;
    }
    transition(this, s, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    as_copy_dest(this, d, || copy_buffer_region(this, d, doff, s, soff, size));
    transition(this, s, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

/// Process-wide committed scratch budget, including buffers awaiting GPU completion.
const MAX_COPY_SCRATCH: u64 = 64 << 20;

struct CopyScratch {
    buffer: ID3D12Resource,
    bytes: u64,
    recorded: bool,
    untracked_submission: bool,
    marks: HashMap<usize, (ID3D12Fence, u64)>,
}

impl CopyScratch {
    fn idle(&mut self) -> bool {
        self.marks.retain(|_, (fence, value)| unsafe { super::real::fence_completed(&fence) < *value });
        !self.untracked_submission && self.marks.is_empty()
    }
}

#[derive(Default)]
struct CopyScratchCache {
    active: HashMap<usize, CopyScratch>,
    released: Vec<CopyScratch>,
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    bytes: u64,
    reaping: bool,
}

impl CopyScratchCache {
    fn collect_released(&mut self) {
        self.released.retain_mut(|scratch| {
            if !scratch.idle() {
                return true;
            }
            self.bytes -= scratch.bytes;
            false
        });
    }

    fn release(&mut self, list: usize) {
        if let Some(scratch) = self.active.remove(&list) {
            self.released.push(scratch);
        }
        self.collect_released();
    }

    unsafe fn acquire(&mut self, list: *mut c_void, bytes: u64) -> Option<(usize, u64)> {
        self.collect_released();
        let key = list as usize;
        if let Some(scratch) = self.active.get_mut(&key) {
            // Reset can happen while the previous recording is executing on another queue.
            if !scratch.recorded && !scratch.idle() {
                return None;
            }
            if scratch.recorded || scratch.bytes >= bytes || bytes > MAX_COPY_SCRATCH - self.bytes + scratch.bytes {
                scratch.recorded = true;
                return Some((scratch.buffer.as_raw() as usize, scratch.bytes));
            }
            // Drop the old allocation before growing so the peak stays within the budget.
            self.bytes -= self.active.remove(&key).unwrap().bytes;
        }
        if bytes > MAX_COPY_SCRATCH - self.bytes {
            return None;
        }
        let buffer = make_copy_scratch(list, bytes)?;
        let raw = buffer.as_raw() as usize;
        self.active.insert(key, CopyScratch {
            buffer, bytes, recorded: true, untracked_submission: false, marks: HashMap::new(),
        });
        self.bytes += bytes;
        Some((raw, bytes))
    }
}

static COPY_SCRATCH: LazyLock<Mutex<CopyScratchCache>> = LazyLock::new(|| Mutex::new(CopyScratchCache::default()));

fn release_copy_scratch(list: usize) {
    let mut cache = COPY_SCRATCH.lock().unwrap();
    cache.release(list);
    if cache.reaping || !cache.released.iter().any(|s| !s.untracked_submission) {
        return;
    }
    // A title can stop submitting work after releasing the list. Reclaim its scratch when
    // the GPU finishes even if no call touches the cache again.
    cache.reaping = std::thread::Builder::new().name("xcl-copy-scratch".into()).spawn(|| loop {
        {
            let mut cache = COPY_SCRATCH.lock().unwrap();
            cache.collect_released();
            if !cache.released.iter().any(|s| !s.untracked_submission) {
                cache.reaping = false;
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(16));
    }).is_ok();
}

unsafe fn copy_scratch(list: *mut c_void, n: u64) -> Option<(usize, u64)> {
    if n > MAX_COPY_SCRATCH || list_type(list) == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return None;
    }
    let bytes = n.next_power_of_two().max(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64);
    COPY_SCRATCH.lock().unwrap().acquire(list, bytes)
}

fn reset_copy_scratch(list: *mut c_void) {
    real::dispatch_chain::list_reset(list as usize);
    if let Some(scratch) = COPY_SCRATCH.lock().unwrap().active.get_mut(&(list as usize)) {
        scratch.recorded = false;
    }
}

/// Mark GPU ownership on the queue that actually submitted each PC list or split segment.
pub(super) unsafe fn copy_scratch_submitted(queue: *mut c_void, lists: &[*mut c_void]) {
    let mut cache = COPY_SCRATCH.lock().unwrap();
    cache.collect_released();
    if !lists.iter().any(|l| cache.active.get(&(*l as usize)).is_some_and(|s| s.recorded)) {
        return;
    }
    let mark = (|| {
        let q = ID3D12CommandQueue::from_raw_borrowed(&queue)?;
        let key = queue as usize;
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.fences.entry(key) {
            let mut dev: Option<ID3D12Device> = None;
            q.GetDevice(&mut dev).ok()?;
            entry.insert((dev?.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok()?, 0));
        }
        let (fence, value) = cache.fences.get_mut(&key)?;
        let next = *value + 1;
        super::real::native_queue_signal(q, &*fence, next).ok()?;
        *value = next;
        Some((fence.clone(), next))
    })();
    for list in lists {
        if let Some(scratch) = cache.active.get_mut(&(*list as usize)).filter(|s| s.recorded) {
            if let Some((fence, value)) = &mark {
                scratch.marks.insert(fence.as_raw() as usize, (fence.clone(), *value));
            } else {
                // Without a completion mark the GPU could still access this buffer.
                scratch.untracked_submission = true;
            }
        }
    }
}

unsafe fn make_copy_scratch(list: *mut c_void, bytes: u64) -> Option<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    device_of(list)?
        .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut b)
        .ok()?;
    b
}

unsafe fn device_of(list: *mut c_void) -> Option<ID3D12Device> {
    let mut dev: Option<ID3D12Device> = None;
    ID3D12GraphicsCommandList::from_raw_borrowed(&list)?.GetDevice(&mut dev).ok()?;
    dev
}

/// upload buffers holding a repeated 16-byte pattern: (pattern, buffer, bytes)
static PATTERNS: Mutex<Vec<([u8; 16], ID3D12Resource, u64)>> = Mutex::new(Vec::new());

/// A buffer of at least `n` bytes repeating `pat` (the device's upload heap).
unsafe fn pattern_buffer(list: *mut c_void, pat: [u8; 16], n: u64) -> Option<usize> {
    let mut cache = PATTERNS.lock().unwrap();
    if let Some((_, b, _)) = cache.iter().find(|(p, _, len)| *p == pat && *len >= n) {
        return Some(b.as_raw() as usize);
    }
    // a larger one replaces an older pattern buffer only when nothing may still read it: kept
    if cache.len() >= 32 {
        return None;
    }
    let dev = device_of(list)?;
    let len = n.next_power_of_two().max(0x1_0000);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        None,
        &mut b,
    )
    .ok()?;
    let b = b?;
    let mut p: *mut c_void = std::ptr::null_mut();
    b.Map(0, None, Some(&mut p)).ok()?;
    for chunk in std::slice::from_raw_parts_mut(p as *mut u8, len as usize).chunks_exact_mut(16) {
        chunk.copy_from_slice(&pat);
    }
    b.Unmap(0, None);
    let raw = b.as_raw() as usize;
    cache.push((pat, b, len));
    Some(raw)
}

unsafe fn fill(list: *mut c_void, va: u64, size: u64, pat: [u8; 16], unit: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    if size == 0 || !size.is_multiple_of(unit) || size > MAX_FILL {
        return warn(&N, what, format_args!("size {size:#x} (a non-zero multiple of {unit} up to {MAX_FILL:#x} is filled)"));
    }
    // (in a GPU view: on its tiles, read back to the CPU view after the submit)
    super::bind::want_view(super::view_want::Path::MemoryOp, va, size);
    let Some((d, off)) = real::uma_window_buffer_gpu(va as usize, size as usize) else {
        return warn(&N, what, format_args!("{va:#x} ({size:#x} bytes) is not in unified memory"));
    };
    let Some(src) = pattern_buffer(list, pat, size) else {
        return warn(&N, what, format_args!("no pattern buffer for {pat:x?}"));
    };
    note_write(list as usize, WriteProducer::InSegmentGpu, va, size);
    as_copy_dest(list, d, || copy_buffer_region(list, d, off, src, 0, size));
}

fn pattern(bytes: &[u8]) -> [u8; 16] {
    let mut p = [0u8; 16];
    for (i, b) in p.iter_mut().enumerate() {
        *b = bytes[i % bytes.len()];
    }
    p
}

/// command list #61 FillMemoryWith32BitValueX(address, size, value)
pub(super) unsafe extern "system" fn fill32_x(this: *mut c_void, va: u64, size: u64, value: u32) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    fill(this, va, size, pattern(&value.to_le_bytes()), 4, "FillMemoryWith32BitValueX");
}

/// command list #62 FillMemoryWith64BitValueX(address, size, value)
pub(super) unsafe extern "system" fn fill64_x(this: *mut c_void, va: u64, size: u64, value: u64) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    fill(this, va, size, pattern(&value.to_le_bytes()), 8, "FillMemoryWith64BitValueX");
}

/// command list #63 FillMemoryWith128BitValueX(address, size, value*)
pub(super) unsafe extern "system" fn fill128_x(this: *mut c_void, va: u64, size: u64, value: *const [u8; 16]) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    if !value.is_null() {
        fill(this, va, size, value.read_unaligned(), 16, "FillMemoryWith128BitValueX");
    }
}

/// `XCL_IMM_BY_CPU=0`: a bottom-of-pipe immediate into a window the GPU has a copy of comes
/// back to the CPU's view with a copy, like the lists' other memory writes. Otherwise XCL stores
/// the value it was given there once the submission is done on the GPU. A retail title's CPU
/// waits on such cells; copied back, one of them held a value up to 46 frames old in the CPU's
/// view (and older than it had held before) while its GPU fence was reached.
fn imm_by_cpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_IMM_BY_CPU").map_or(true, |v| v.trim() != "0"))
}

/// Values written by the GPU at `va` (WriteBufferImmediate: MARKER_IN = when the command is
/// reached, MARKER_OUT = when the work before it is done).
unsafe fn write(list: *mut c_void, va: u64, words: &[u32], mode: D3D12_WRITEBUFFERIMMEDIATE_MODE, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = 4 * words.len() as u64;
    if !va.is_multiple_of(n) {
        return warn(&N, what, format_args!("{va:#x} is not {n}-byte aligned"));
    }
    // (in a GPU view: on its tiles, and the CPU view gets the value after the submit)
    super::bind::want_view(super::view_want::Path::MemoryOp, va, n);
    let (Some((d, _)), Some(at)) = (real::uma_window_buffer_gpu(va as usize, n as usize), real::uma_va_of(va as usize)) else {
        // the title's own memory: written by the CPU once the GPU is there (never a GPU view)
        if super::cpu_writes::writable(va as usize, n as usize) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
                s.note_write(WriteProducer::CpuAfterSegment, va, n);
                s.cpu_writes.push(super::cpu_writes::Write::Immediate(va as usize, words.to_vec()));
                return;
            }
        }
        return warn(&N, what, format_args!("{va:#x} is not in unified memory"));
    };
    // into a window the GPU only has a copy of: the CPU's view gets the value from XCL
    if words.len() <= 2 && imm_by_cpu() && read_back(va) {
        let value = words.iter().enumerate().fold(0u64, |v, (i, &w)| v | (w as u64) << (32 * i));
        // a counter the title's queues write in turn (a `SyncCell` whose values only rose): a list
        // that runs late does not take the CPU's view of it back (`sparse::IMM_KEEP_MAX`)
        let counter = words.len() == 2 && SYNC_GRAPH.lock().unwrap().as_ref().and_then(|g| g.cells.get(&cpu_at(va))).is_some_and(|c| !c.non_monotonic && c.recorded_max != 0);
        let bytes = n as u8 | if counter { super::sparse::IMM_KEEP_MAX } else { 0 };
        LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(list as usize).or_default().note_imm(va, value, bytes);
    } else {
        note_write(list as usize, WriteProducer::InSegmentGpu, va, n);
    }
    let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = words.iter().enumerate().map(|(i, &w)| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: at + 4 * i as u64, Value: w }).collect();
    let modes = vec![mode; words.len()];
    type F = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
    as_copy_dest(list, d, || pc::<F>(list, PC_WRITE_BUFFER_IMMEDIATE)(list, params.len() as u32, params.as_ptr(), modes.as_ptr()));
}

/// timestamp queries (one heap per device, used round-robin) and a lazily created READBACK
/// buffer for timestamps targeting memory outside unified memory
struct TimestampPool {
    heap: ID3D12QueryHeap,
    /// `(buffer, mapped_base)` of `TIMESTAMP_SLOTS * 8` bytes in a READBACK heap
    readback: Option<(ID3D12Resource, usize)>,
}

// SAFETY: D3D12 query heaps and resources are free-threaded COM objects; the persistent
// readback mapping lives as long as `readback` is held.
unsafe impl Send for TimestampPool {}

static TIMESTAMPS: Mutex<Option<HashMap<usize, TimestampPool>>> = Mutex::new(None);
const TIMESTAMP_SLOTS: u32 = 4096;
static NEXT_TIMESTAMP: AtomicU32 = AtomicU32::new(0);

unsafe fn ensure_timestamp_readback(dev: &ID3D12Device, pool: &mut TimestampPool) -> Option<(ID3D12Resource, usize)> {
    if let Some((ref buf, base)) = pool.readback {
        return Some((buf.clone(), base));
    }
    let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let rd = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: TIMESTAMP_SLOTS as u64 * 8,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut rb: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).ok()?;
    let rb = rb?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut ptr)).ok()?;
    if ptr.is_null() {
        return None;
    }
    let base = ptr as usize;
    pool.readback = Some((rb.clone(), base));
    Some((rb, base))
}

/// The GPU timestamp (the PC queue's ticks) written at `va` when the work before it is done.
unsafe fn timestamp(list: *mut c_void, va: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let ty = list_type(list);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return warn(&N, what, format_args!("a timestamp on a type-{} list", ty.0));
    }
    if !va.is_multiple_of(8) {
        return warn(&N, what, format_args!("{va:#x} is not 8-byte aligned"));
    }
    // (in a GPU view: resolved into its tile, read back to the CPU view after the submit)
    super::bind::want_view(super::view_want::Path::MemoryOp, va, 8);
    let uma_dst = real::uma_window_buffer_gpu(va as usize, 8);
    if uma_dst.is_none() && !super::cpu_writes::writable(va as usize, 8) {
        return warn(&N, what, format_args!("{va:#x} is not in unified memory"));
    }
    let Some(dev) = device_of(list) else { return };
    let (heap, readback) = {
        let mut m = TIMESTAMPS.lock().unwrap();
        let m = m.get_or_insert_with(HashMap::new);
        let pool = match m.entry(dev.as_raw() as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                let mut h: Option<ID3D12QueryHeap> = None;
                let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: TIMESTAMP_SLOTS, NodeMask: 0 };
                match dev.CreateQueryHeap(&desc, &mut h) {
                    Ok(()) if h.is_some() => slot.insert(TimestampPool { heap: h.unwrap(), readback: None }),
                    _ => return warn(&N, what, format_args!("no timestamp query heap")),
                }
            }
        };
        let rb = if uma_dst.is_none() {
            match ensure_timestamp_readback(&dev, pool) {
                Some(rb) => Some(rb),
                None => return warn(&N, what, format_args!("no timestamp readback buffer")),
            }
        } else {
            None
        };
        (pool.heap.as_raw() as usize, rb)
    };
    let i = NEXT_TIMESTAMP.fetch_add(1, Ordering::Relaxed) % TIMESTAMP_SLOTS;
    type Resolve = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);
    if let Some((d, off)) = uma_dst {
        note_write(list as usize, WriteProducer::InSegmentGpu, va, 8);
        pc::<unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32)>(list, PC_END_QUERY)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
        as_copy_dest(list, d, || pc::<Resolve>(list, PC_RESOLVE_QUERY_DATA)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i, 1, d, off));
    } else if let Some((rb, base)) = readback {
        // Queue the CPU copy only if the list is tracked; otherwise avoid emitting an unused resolve.
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else {
            return warn(&N, what, format_args!("{va:#x} is not in unified memory"));
        };
        let rb_raw = rb.as_raw() as usize;
        let off = i as u64 * 8;
        s.note_write(WriteProducer::CpuAfterSegment, va, 8);
        s.cpu_writes.push(super::cpu_writes::Write::ReadbackU64 { va: va as usize, src: base + off as usize, _buf: rb });
        drop(g);
        pc::<unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32)>(list, PC_END_QUERY)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
        pc::<Resolve>(list, PC_RESOLVE_QUERY_DATA)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i, 1, rb_raw, off);
    }
}

/// command list #69 Write32BitValueBottomOfPipeX(address, value, flush, flags)
///
/// # Safety
/// `this` must be a valid adopted `ID3D12GraphicsCommandList` pointer.
pub unsafe extern "system" fn write32_bottom_x(this: *mut c_void, va: u64, value: u32, _flush: u32, flags: u32) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    record_write_trace(this, va, value as u64, flags, "W32B");
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, "Write32BitValueBottomOfPipeX");
}

/// command list #70 Write64BitValueBottomOfPipeX(address, value, flush, flags)
pub(super) unsafe extern "system" fn write64_bottom_x(this: *mut c_void, va: u64, value: u64, _flush: u32, flags: u32) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    const W: &str = "Write64BitValueBottomOfPipeX";
    record_write_trace(this, va, value, flags, "W64B");
    if flags & BOTTOM_TIMESTAMPS != 0 {
        return timestamp(this, va, W);
    }
    note_write64_recorded(this, va, value);
    write(this, va, &[value as u32, (value >> 32) as u32], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, W);
}

/// The title Resets its list (`title_list` `wrapper`) while the GPU has not finished the last
/// submit of the PC list `primary` behind it: a console lets the title submit the new recording
/// at once, the PC does not let `primary` be submitted again yet (#1452, debug layer id 553;
/// NVIDIA removes the device). The new recording goes onto a spare PC list of the wrapper whose
/// submits are done, or a new one, reset like `primary` would be. None: it could not be.
pub(super) unsafe fn reset_stand_in(wrapper: *mut c_void, primary: *mut c_void, alloc: usize, pso: usize) -> Option<usize> {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListLifecycle);
    let ty = list_type(primary);
    if ty == D3D12_COMMAND_LIST_TYPE_BUNDLE || alloc == 0 {
        return None;
    }
    let spare = match super::title_list::pop_spare(wrapper) {
        Some(p) => p,
        None => {
            let pc_alloc = super::alloc_pool::current(alloc);
            let made = real::create_raw_pc_command_list(device_of(primary)?.as_raw(), ty, pc_alloc, 0)?;
            on_create(made, pc_alloc, 0, ty);
            // (closed: it is reset below like a spare)
            let ext = *(made as *const usize);
            let hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(made, PC_CLOSE)(made);
            real::preserve_vtable(made, ext);
            if hr < 0 {
                super::title_list::put_spare(wrapper, made);
                return None;
            }
            made
        }
    };
    if reset_pc_inner(spare, alloc, pso) < 0 {
        super::title_list::put_spare(wrapper, spare);
        return None;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: a command list reset while its last submit is on the GPU records on another PC list (#{n})");
    }
    Some(spare as usize)
}

/// Deferred mid-list split for `Wait32/64BitValueX` that depends on a producer in the current
/// recording. Closes `list` and opens/reuses a sibling PC list on the same allocator, carrying all
/// bound D3D12 + XCL root-argument state across the split.
///
/// At `ExecuteCommandLists`, `title_list::push_segments` expands the wrapper into `[seg_0, seg_1, ...]`
/// Submission runs segment N's Write64 signal or deferred CPU store before segment N+1's wait.
unsafe fn try_split_for_wait(list: *mut c_void, placement: WaitPlacement) -> Option<*mut c_void> {
    let (reason, why) = match placement {
        WaitPlacement::SplitAfterCpuWrite => (SegmentReason::WaitAfterCpuWrite, "CPU-after-segment write"),
        WaitPlacement::SplitAfterWrite64 => {
            static ENABLED: OnceLock<bool> = OnceLock::new();
            if !*ENABLED.get_or_init(|| std::env::var("XCL_SPLIT_MIDLIST_WAIT").map_or(true, |v| v.trim() != "0")) {
                return None;
            }
            (SegmentReason::WaitAfterWrite64, "Write64")
        }
        _ => return None,
    };
    let next_pc = split_recording(list, reason)?;
    static SPLITS: AtomicU64 = AtomicU64::new(0);
    let n = SPLITS.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || (n.is_power_of_two() && n <= 256) {
        crate::xlog_info!("D3D12.X: split command list (type {}) around mid-list Wait after {why} (#{n})", list_type(next_pc).0);
    }
    Some(next_pc)
}

fn wait_target_after_split(list: *mut c_void, placement: WaitPlacement, continuation: Option<*mut c_void>) -> Option<*mut c_void> {
    match (placement, continuation) {
        (_, Some(next)) => Some(next),
        (WaitPlacement::SplitAfterCpuWrite, None) => None,
        (WaitPlacement::SplitAfterWrite64, None) => Some(list),
        _ => Some(list),
    }
}

/// Defer a direct dispatch against the bind's retained pipeline identity. Address reuse after
/// SetPipelineState cannot redirect either the demand or the later exact emission.
pub(super) unsafe fn defer_retained(
    list: *mut c_void,
    resolution: &super::pso_async::PipelineResolution,
    what: Deferred,
) -> bool {
    if recording_poisoned(list as usize) {
        return false;
    }
    if !resolution.submit_demand_pending() {
        return false;
    }
    // A configured initial-build timeout may record immediately, but an evicted exact generation
    // has no legal stand-in. Always install its retained ticket before splitting for submit.
    if resolution.is_deferred_restoration() {
        return resolution.need_by_submit()
            && defer_with(list, what, exact_restoration_waits_bounded(super::pso_async::waits_bounded()), || {});
    }
    defer_with(list, what, super::pso_async::waits_bounded(), || { let _ = resolution.need_by_submit(); })
}

fn exact_restoration_waits_bounded(_configured: bool) -> bool { false }

unsafe fn defer_with(list: *mut c_void, what: Deferred, waits_bounded: bool, demand: impl FnOnce()) -> bool {
    // (one the submit records now waits there: `record_deferred`)
    if super::pso_async::is_at_submit() {
        return false;
    }
    // (without the allocator swap the submit would reset the list's own allocator while the GPU
    // may still run what its last submit recorded on it: the dispatch waits in the call)
    if !super::alloc_pool::enabled() {
        super::pso_async::note_put_off(super::pso_async::PutOff::Refused);
        return false;
    }
    // A bounded bind records now and waits for the same bound. Deferring it only adds a segment;
    // thousands of these exhausted command-list memory in a retail title with a 1 ms budget.
    let mut refused = false;
    if !defer_pipeline_wait(waits_bounded, demand, || {
        refused = split_recording(list, SegmentReason::DeferredDispatch(what)).is_none();
        !refused
    }) {
        if refused {
            super::pso_async::note_put_off(super::pso_async::PutOff::Refused);
        }
        return false;
    }
    super::pso_async::note_put_off(super::pso_async::PutOff::Deferred);
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: dispatch with a pipeline still being built put off to the submit (#{n})");
    }
    true
}

fn defer_pipeline_wait(waits_bounded: bool, demand: impl FnOnce(), split: impl FnOnce() -> bool) -> bool {
    if waits_bounded {
        return false;
    }
    // The split can allocate and reset native objects. Let the worker and its compiler/driver
    // admission run concurrently with that preparation; a failed split waits for the same build
    // immediately in the caller.
    demand();
    split()
}

/// Records the deferred dispatches of `wrapper`'s recording, waiting for their pipelines as a bind did.
pub(super) unsafe fn record_deferred(wrapper: *mut c_void) {
    let mut segs = Vec::new();
    super::title_list::push_segments(wrapper, &mut segs);
    let Some(&first) = segs.first() else { return };
    segs.retain(|&seg| seg == first || LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(seg as usize))).is_some_and(|s| s.deferred.is_some()));
    if segs.len() < 2 {
        return;
    }
    // (on the primary list, which stays the wrapper's: `deferred_segment`)
    let own = super::title_list::backing(wrapper).and_then(|p| LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(p as usize))?.own_alloc.clone()));
    // (its waits and own builds are the submit's: the put-off line)
    let _at_submit = super::pso_async::at_submit();
    let mut last: Option<*mut c_void> = None;
    for &seg in segs.iter().skip(1) {
        let Some((what, ty, heaps)) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(seg as usize))).and_then(|s| Some((s.deferred.take()?, s.list_type?, s.bindings.heaps.clone()))) else {
            continue;
        };
        let Some(own) = own.as_ref() else {
            super::bind::dispatch_dropped();
            super::pso_async::note_put_off(super::pso_async::PutOff::Dropped);
            continue;
        };
        let own = own.as_raw();
        let ext = *(seg as *const usize);
        let mut hr = match last {
            Some(_) => 0,
            None => super::alloc_pool::title_reset(own).unwrap_or_else(|| real::reset_pc_allocator(own)),
        };
        if hr >= 0 {
            hr = super::alloc_pool::reset_list(seg as usize, own as usize, ty.0, |pc_alloc| pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(seg, PC_RESET)(seg, pc_alloc, 0));
        }
        real::preserve_vtable(seg, ext);
        if hr < 0 {
            static RESET_ERR: AtomicU64 = AtomicU64::new(0);
            warn(&RESET_ERR, "deferred dispatch", format_args!("Reset failed ({hr:#x})"));
            super::bind::dispatch_dropped();
            super::pso_async::note_put_off(super::pso_async::PutOff::Dropped);
            continue;
        }
        reset_copy_scratch(seg);
        if !heaps.is_empty() {
            pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(seg, 28)(seg, heaps.len() as u32, heaps.as_ptr());
        }
        if let Some(last) = last {
            super::bind::pass_arena(last, seg);
            real::dispatch_chain::list_follows(last as usize, seg as usize);
        }
        // (a recording of its own, made here and not through `cl_reset`: its pipeline is named
        // before the dispatch, for the device-removal report)
        super::dred::reset(seg);
        let p = super::bind::bound_pso(seg);
        if p != 0 {
            super::dred::mark_pipeline(seg, p, super::pso_async::peek(p));
        }
        match what {
            Deferred::Dispatch(x, y, z, flags) => super::bind::dispatch_flags(seg, x, y, z, flags),
            Deferred::Indirect { sig, max, args, args_off, count_buf, count_off, signature } => {
                super::bind::execute_indirect_retained(
                    seg, sig, max, args, args_off, count_buf, count_off, signature,
                )
            }
        }
        let built = super::pso_async::has_real(super::bind::bound_pso(seg));
        super::pso_async::note_put_off(if built { super::pso_async::PutOff::Recorded } else { super::pso_async::PutOff::Skipped });
        super::bind::flush_two_pass(seg, super::bind::RunEnd::Segment);
        super::gpu_times::before_close(seg);
        let ext = *(seg as *const usize);
        let hr = segment_close(wrapper, seg);
        real::preserve_vtable(seg, ext);
        if hr < 0 {
            // a recording the runtime refused is not submitted: the segment leaves the recording
            // (a list whose Close failed is still open and cannot be reset), its call is dropped,
            // and the next one does not follow it
            static CLOSE_ERR: AtomicU64 = AtomicU64::new(0);
            warn(&CLOSE_ERR, "deferred call", format_args!("Close failed ({hr:#x}), the call is dropped"));
            super::bind::dispatch_dropped();
            if let Some(last) = last {
                super::bind::pass_arena(seg, last);
            }
            super::title_list::drop_segment(wrapper, seg);
            continue;
        }
        last = Some(seg);
    }
}

/// (tests) What the next native Close of a put-off segment of this wrapper does instead
/// (`segment_close`; tests run in parallel, so it is the given wrapper's only).
#[cfg(test)]
type CloseHook = Box<dyn FnOnce(*mut c_void) -> i32 + Send>;
#[cfg(test)]
static CLOSE_HOOK: Mutex<Vec<(usize, CloseHook)>> = Mutex::new(Vec::new());

/// The native Close of `wrapper`'s put-off segment `p`: its HRESULT.
unsafe fn segment_close(wrapper: *mut c_void, p: *mut c_void) -> i32 {
    #[cfg(test)]
    {
        let hook = {
            let mut g = CLOSE_HOOK.lock().unwrap();
            g.iter().position(|h| h.0 == wrapper as usize).map(|i| g.remove(i).1)
        };
        if let Some(hook) = hook {
            return hook(p);
        }
    }
    let _ = wrapper;
    pc::<unsafe extern "system" fn(*mut c_void) -> i32>(p, PC_CLOSE)(p)
}

/// The allocator `list`'s put-off calls recorded on is reset once the GPU is done with them, so
/// their split dispatches' scratch does not outlive them.
unsafe fn reset_own_alloc(list: *mut c_void) {
    let own = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))?.own_alloc.clone());
    let Some(own) = own else { return };
    let own = own.as_raw();
    // (without `alloc_pool` nothing says whether the GPU is done: the next submit resets it)
    if super::alloc_pool::enabled() && !super::alloc_pool::allocator_in_flight(own as usize) {
        super::alloc_pool::title_reset(own).unwrap_or_else(|| real::reset_pc_allocator(own));
    }
}

/// Ends `list`'s segment and opens the next, carrying its recording state across the boundary.
pub(super) unsafe fn split_recording(list: *mut c_void, reason: SegmentReason) -> Option<*mut c_void> {
    if recording_poisoned(list as usize) {
        return None;
    }
    let wrapper = super::title_list::wrapper_of(list)?;
    let ty = list_type(list);
    if ty == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return None;
    }
    let extra_segments = reason.extra_segments();
    let deferred_what = match reason {
        SegmentReason::DeferredDispatch(what) => {
            // (a predicated call stays in its list)
            let plain = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).is_some_and(|s| s.allocator != 0 && s.predication.is_none());
            if !plain {
                return None;
            }
            Some(what)
        }
        SegmentReason::WaitAfterWrite64 | SegmentReason::WaitAfterCpuWrite => None,
    };
    let carry = {
        let g = LISTS.lock().unwrap();
        SplitCarry::capture(g.as_ref()?.get(&(list as usize))?)?
    };
    let alloc = carry.allocator;
    let bound_pso = carry.bound_pso;
    let depth_bias = carry.depth_bias;
    let index_buffer = carry.index_buffer;
    let depth_bounds = carry.depth_bounds;
    let predication = carry.predication;
    let pred_buf = &carry.pred_buf;
    let bindings = &carry.bindings;

    let dev = device_of(list)?;
    let dev_raw = dev.as_raw() as usize;
    if !super::title_list::prepare_split(wrapper, extra_segments) {
        return None;
    }
    {
        let mut g = LISTS.lock().unwrap();
        if g.get_or_insert_with(HashMap::new).try_reserve(extra_segments).is_err() {
            return None;
        }
    }
    let next_pc = reserve_closed_segment(wrapper, &dev, ty)?;
    let deferred = if let Some(what) = deferred_what {
        match prepare_deferred_segment(wrapper, list, ty) {
            Some(seg) => Some((seg, ListState {
                list_type: Some(ty),
                bindings: Bindings {
                    heaps: bindings.heaps.clone(),
                    has_sampler_heap: bindings.has_sampler_heap,
                    shared_view_heap: bindings.shared_view_heap,
                    ..Default::default()
                },
                deferred: Some(what),
                ..Default::default()
            })),
            None => {
                super::title_list::put_spare(wrapper, next_pc);
                return None;
            }
        }
    } else {
        None
    };
    let mut continuation_state = carry.continuation_state(ty);
    // (what the segments put in unified memory is held for the PC allocator they record on, as
    // `next_pc`'s: the GPU is done with it when with that list's other segments)
    let on = super::alloc_pool::current(alloc);
    let next_bindings = super::bind::prepare_split(list, next_pc, on);
    let (init_pso, init_generation) = if ty == D3D12_COMMAND_LIST_TYPE_COPY {
        (0, None)
    } else {
        next_bindings.initial_pipeline()
    };
    if let Some(generation) = init_generation {
        drop(continuation_state.pin_native_generation(generation));
    }
    let deferred_bindings = deferred.as_ref().map(|(seg, _)| super::bind::prepare_split(list, *seg, on));

    // Commit starts here: all allocation is complete and the continuation exists closed.
    {
        let mut g = LISTS.lock().unwrap();
        let s = g.as_mut()?.get_mut(&(list as usize))?;
        carry.commit_state(s);
    }
    super::bind::flush_two_pass(list, super::bind::RunEnd::Segment);
    if let Some(((h, i), qty)) = carry.active_query {
        pc::<QueryFn>(list, PC_END_QUERY)(list, h, qty, i);
    }
    if carry.pred_in_state {
        if let Some(pb) = pred_buf {
            transition(list, pb.as_raw() as usize, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
        }
    }

    super::gpu_times::before_close(list);
    let ext = *(list as *const usize);
    let close_hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(list, PC_CLOSE)(list);
    real::preserve_vtable(list, ext);
    if close_hr < 0 {
        super::title_list::fail_closed(wrapper, list, next_pc, true);
        if let Some((seg, _)) = deferred {
            super::title_list::put_spare(wrapper, seg);
        }
        static CLOSE_ERR: AtomicU64 = AtomicU64::new(0);
        warn(&CLOSE_ERR, "mid-list split", format_args!("Close failed ({close_hr:#x})"));
        return None;
    }
    if !reset_reserved_segment(next_pc, alloc, ty, init_pso) {
        super::title_list::fail_closed(wrapper, list, next_pc, false);
        if let Some((seg, _)) = deferred {
            super::title_list::put_spare(wrapper, seg);
        }
        static RESET_ERR: AtomicU64 = AtomicU64::new(0);
        warn(&RESET_ERR, "mid-list split", format_args!("continuation Reset failed"));
        return None;
    }

    // the next segment is a recording of its own (reset or made here, not through `cl_reset`)
    super::dred::reset(next_pc);
    if init_pso != 0 {
        super::dred::mark_pipeline(next_pc, bound_pso, init_pso);
    }

    let old_next = {
        let mut g = LISTS.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let old = m.remove(&(next_pc as usize));
        if let Some(ref o) = old {
            if o.has_sparse() {
                SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
            if !o.write64_signals.is_empty() {
                WRITE64_SIGNAL_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
        }
        m.insert(
            next_pc as usize,
            continuation_state,
        );
        old
    };
    drop(old_next);
    super::bind::install_prepared_split(next_bindings);
    super::lut3d::list_reset(next_pc as usize);
    restore_bindings(next_pc, bindings, ty);

    if ty == D3D12_COMMAND_LIST_TYPE_DIRECT {
        let info = if bound_pso != 0 { device_x::base_depth_bias_info(bound_pso) } else { None };
        let mut applied_native = false;
        if let Some((true, static_bias)) = info {
            let target = depth_bias.unwrap_or(static_bias);
            if target.is_zero() {
                applied_native = true;
            } else if try_native_depth_bias(next_pc, target) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(next_pc as usize))) {
                    s.native_bias = target;
                }
                applied_native = true;
            }
        } else if bound_pso == 0 {
            if let Some(b) = depth_bias {
                if device_x::dynamic_depth_bias_supported(dev_raw as *mut c_void) && try_native_depth_bias(next_pc, b) {
                    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(next_pc as usize))) {
                        s.native_bias = b;
                    }
                    applied_native = true;
                }
            }
        }
        if depth_bias.is_some() && !applied_native {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(next_pc as usize))) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
        if let Some(ref view) = index_buffer {
            set_index_buffer(next_pc, Some(view));
        }
        if let Some((min, max)) = depth_bounds {
            pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(next_pc, PC_OM_SET_DEPTH_BOUNDS)(next_pc, min, max);
        }
    }
    if ty != D3D12_COMMAND_LIST_TYPE_COPY {
        if let (Some(op), Some(pb)) = (predication, pred_buf) {
            let pb_raw = pb.as_raw() as usize;
            transition(next_pc, pb_raw, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PREDICATION);
            pc::<unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP)>(next_pc, PC_SET_PREDICATION)(next_pc, pb_raw, 0, op);
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(next_pc as usize))) {
                s.pred_in_state = true;
            }
        }
    }

    if let Some((seg, state)) = deferred {
        {
            let mut g = LISTS.lock().unwrap();
            g.get_or_insert_with(HashMap::new).insert(seg as usize, state);
        }
        super::bind::install_prepared_split(deferred_bindings.unwrap());
        super::bind::pass_arena(list, next_pc);
        super::title_list::push_segment(wrapper, seg);
    }
    super::title_list::activate_segment(wrapper, list, next_pc);
    Some(next_pc)
}

/// Reserve a known-closed continuation without associating the producer's live allocator with a
/// second recording list. A new object is opened on a private allocator and immediately closed.
unsafe fn reserve_closed_segment(
    wrapper: *mut c_void,
    dev: &ID3D12Device,
    ty: D3D12_COMMAND_LIST_TYPE,
) -> Option<*mut c_void> {
    if let Some(spare) = super::title_list::pop_spare(wrapper) {
        note_reserved_closed(spare as usize, ty);
        return Some(spare);
    }
    // CreateCommandList1 produces exactly the allocator-free closed object the transaction
    // needs. Keep the older path as a compatibility fallback for devices without ID3D12Device4.
    #[cfg(test)]
    let close_is_hooked = CLOSE_HOOK.lock().unwrap().iter().any(|h| h.0 == wrapper as usize);
    #[cfg(not(test))]
    let close_is_hooked = false;
    if !close_is_hooked {
        if let Some(made) = real::create_closed_pc_command_list(dev.as_raw(), ty) {
            // A list from CreateCommandList1 that was never recorded is not executable: submitting
            // it removes the device (DXGI_ERROR_INVALID_CALL). The failed-split route submits this
            // standby, so record it empty once on a private allocator first.
            if let Some(alloc) = record_empty_closed(wrapper, dev, made, ty) {
                note_reserved_closed(made as usize, ty);
                let old = STANDBY_ALLOCS.lock().unwrap().get_or_insert_with(HashMap::new).insert(made as usize, alloc);
                drop(old);
                return Some(made);
            }
            pc::<unsafe extern "system" fn(*mut c_void) -> u32>(made, 2)(made);
        }
    }
    let temp: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).ok()?;
    if !real::is_adopted(temp.as_raw()) {
        real::adopt(temp.as_raw(), real::Kind::CommandAllocator);
    }
    let made = real::create_raw_pc_command_list(dev.as_raw(), ty, temp.as_raw() as usize, 0)?;
    let ext = *(made as *const usize);
    let hr = segment_close(wrapper, made);
    real::preserve_vtable(made, ext);
    if hr >= 0 {
        note_reserved_closed(made as usize, ty);
        Some(made)
    } else {
        static CLOSE_ERR: AtomicU64 = AtomicU64::new(0);
        warn(&CLOSE_ERR, "put-off segment", format_args!("Close failed ({hr:#x})"));
        pc::<unsafe extern "system" fn(*mut c_void) -> u32>(made, 2)(made);
        None
    }
}

/// Reset `made` (closed, never recorded) on a private allocator and Close it with nothing recorded,
/// so that it can be submitted. Returns that allocator, which must outlive the recording.
unsafe fn record_empty_closed(wrapper: *mut c_void, dev: &ID3D12Device, made: *mut c_void, ty: D3D12_COMMAND_LIST_TYPE) -> Option<ID3D12CommandAllocator> {
    let Ok(temp) = dev.CreateCommandAllocator::<ID3D12CommandAllocator>(ty) else {
        return None;
    };
    if !real::is_adopted(temp.as_raw()) {
        real::adopt(temp.as_raw(), real::Kind::CommandAllocator);
    }
    let ext = *(made as *const usize);
    let reset = pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(made, PC_RESET)(made, temp.as_raw() as usize, 0);
    // (the PC Reset may put the runtime's vtable back: restore ours before the Close)
    real::preserve_vtable(made, ext);
    let ok = reset >= 0 && segment_close(wrapper, made) >= 0;
    real::preserve_vtable(made, ext);
    ok.then_some(temp)
}

/// A closed standby must not look recordable if its Reset as the continuation later fails.
fn note_reserved_closed(list: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    clear_segment_state(list);
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(list).or_default();
    s.allocator = 0;
    s.list_type = Some(ty);
}

unsafe fn reset_reserved_segment(list: *mut c_void, alloc: usize, ty: D3D12_COMMAND_LIST_TYPE, init_pso: usize) -> bool {
    let ext = *(list as *const usize);
    let hr = super::alloc_pool::reset_list(list as usize, alloc, ty.0, |pc_alloc| {
        pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(list, PC_RESET)(list, pc_alloc, init_pso)
    });
    real::preserve_vtable(list, ext);
    if hr >= 0 {
        // (it records on the title's allocator now, not on its standby one)
        drop_standby_alloc(list as usize);
        reset_copy_scratch(list);
        true
    } else {
        false
    }
}

/// Prepare a closed, empty dispatch segment without publishing it into the recording.
unsafe fn prepare_deferred_segment(wrapper: *mut c_void, list: *mut c_void, ty: D3D12_COMMAND_LIST_TYPE) -> Option<*mut c_void> {
    let dev = device_of(list)?;
    // (kept on the primary PC list: the first segment of a recording can be a spare standing in
    // for it, `title_list::reset`, which goes back to the spares, and a split that takes it
    // again replaces what is kept of it; `record_deferred` looks it up there)
    let first = super::title_list::backing(wrapper)? as usize;
    let own = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&first)?.own_alloc.clone());
    let _own = match own {
        Some(a) => a,
        None => {
            let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).ok()?;
            if !real::is_adopted(a.as_raw()) {
                real::adopt(a.as_raw(), real::Kind::CommandAllocator);
            }
            LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(first).or_default().own_alloc = Some(a.clone());
            a
        }
    };
    reserve_closed_segment(wrapper, &dev, ty)
}

/// Record a `Wait32BitValueX` or `Wait64BitValueX` memory wait on `list`.
///
/// If an earlier write in the same open list already targeted an overlapping UMA range, the
/// `COMMON -> COPY_DEST -> COMMON` barrier around that write already orders subsequent work
/// on the GPU, so we do not stall before submission (which would deadlock waiting on our own
/// unsubmitted list). An overlapping CPU-after-segment write must instead close the producer
/// segment and complete its deferred store before this wait is evaluated. A prior `Write64` to
/// another cell keeps its existing split so its signal runs before the wait.
unsafe fn wait_memory(list: *mut c_void, va: u64, width: u8, comparison: u32, value: u64, mask: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = width as u64;
    if !va.is_multiple_of(n) {
        return warn(&N, what, format_args!("{va:#x} is not {n}-byte aligned"));
    }
    // (the wait reads the memory on the CPU: the title's own memory too, which `cpu_writes` fills;
    // in a GPU view at its CPU view, where what the lists write into its tiles is read back)
    let at = cpu_at(va);
    super::bind::want_view(super::view_want::Path::Wait, va, n);
    if real::uma_window_buffer_gpu(va as usize, width as usize).is_none() && !crate::gpu::shader::readable(at as usize, width as usize) {
        return warn(&N, what, format_args!("{va:#x} is not readable memory"));
    }
    if compare_u64(comparison, 0, 0).is_none() {
        return warn(&N, what, format_args!("unknown comparison op {comparison}"));
    }
    if comparison == D3D12_COMPARISON_FUNC_ALWAYS.0 as u32 {
        return;
    }
    if comparison == D3D12_COMPARISON_FUNC_NEVER.0 as u32 {
        return warn(&N, what, format_args!("NEVER comparison at {va:#x} (skipped to avoid hang)"));
    }
    let placement = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        let placement = s.wait_placement(va, n);
        match placement {
            WaitPlacement::OrderedInSegment => return,
            WaitPlacement::RecordHere => {
                s.push_wait(PendingWait { va: at, width, comparison, value, mask, after_write64: false });
                return;
            }
            WaitPlacement::SplitAfterCpuWrite | WaitPlacement::SplitAfterWrite64 => placement,
        }
    };
    let continuation = try_split_for_wait(list, placement);
    let Some(target) = wait_target_after_split(list, placement, continuation) else {
        static SPLIT_FAILED: AtomicU64 = AtomicU64::new(0);
        return warn(
            &SPLIT_FAILED,
            what,
            format_args!("could not create a segment boundary after the deferred CPU write; the dependent wait was not recorded"),
        );
    };
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(target as usize).or_default();
    let after_write64 = !s.write64_signals.is_empty();
    s.push_wait(PendingWait { va: at, width, comparison, value, mask, after_write64 });
}

/// command list #71 Wait32BitValueX(address, comparison, value, mask, flags)
pub(super) unsafe extern "system" fn wait32_x(this: *mut c_void, va: u64, comparison: u32, value: u32, mask: u32, _flags: u32) {
    wait_memory(this, va, 4, comparison, value as u64, mask as u64, "Wait32BitValueX");
}

/// command list #72 Wait64BitValueX(address, comparison, value, flags): the whole 64-bit value
/// is compared. What comes after the value is not a mask: a retail game passes a 32-bit 1 there (the
/// upper half of the stack slot is whatever was left in it – 0x231_0000_0001), and read as a
/// mask its waits for a counter at 2 ≥ 1, 4 ≥ 2 … never held: each timed out (2 s a frame, the
/// game stood still after its first cinematic).
pub(super) unsafe extern "system" fn wait64_x(this: *mut c_void, va: u64, comparison: u32, value: u64, _flags: u64, _more: u32) {
    wait_memory(this, va, 8, comparison, value, u64::MAX, "Wait64BitValueX");
}

/// command list #88 Write32BitValueTopOfPipeX(address, value, flags)
pub(super) unsafe extern "system" fn write32_top_x(this: *mut c_void, va: u64, value: u32, flags: u32) {
    record_write_trace(this, va, value as u64, flags, "W32T");
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, "Write32BitValueTopOfPipeX");
}

/// command list #89 Write64BitValueTopOfPipeX(address, value, flags)
pub(super) unsafe extern "system" fn write64_top_x(this: *mut c_void, va: u64, value: u64, flags: u32) {
    const W: &str = "Write64BitValueTopOfPipeX";
    record_write_trace(this, va, value, flags, "W64T");
    if flags & TOP_TIMESTAMPS != 0 {
        return timestamp(this, va, W);
    }
    note_write64_recorded(this, va, value);
    write(this, va, &[value as u32, (value >> 32) as u32], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, W);
}

/// command list #90 Write32BitValueBottomOfShaderX(address, value, stage)
pub(super) unsafe extern "system" fn write32_bottom_of_shader_x(this: *mut c_void, va: u64, value: u32, stage: u32) {
    record_write_trace(this, va, value as u64, stage, "W32S");
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, "Write32BitValueBottomOfShaderX");
}

/// command list #68 OMSetDepthBounds(min, max) (PC #62)
pub(super) unsafe extern "system" fn set_depth_bounds_x(this: *mut c_void, min: f32, max: f32) {
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(this as usize).or_default().depth_bounds = Some((min, max));
    pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(this, PC_OM_SET_DEPTH_BOUNDS)(this, min, max);
}

/// command list #78 DispatchX(x, y, z, flags)
pub(super) unsafe extern "system" fn dispatch_x(this: *mut c_void, x: u32, y: u32, z: u32, flags: u32) {
    super::bind::dispatch_flags(this, x, y, z, flags);
}

unsafe fn emit_uav_barrier_pc(this: *mut c_void) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }),
        },
    };
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(this, PC_RESOURCE_BARRIER)(this, 1, &b);
}

/// command list #146 console compute/UAV barrier (used by `XboxBvhBuilder.dll` between BVH
/// build passes): emits a global UAV barrier so UMA writes become visible to the next pass
pub(super) unsafe extern "system" fn uav_barrier_x(this: *mut c_void, _n: u32, _barriers: *const c_void) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListBarrier);
    emit_uav_barrier_pc(this);
}

/// A UAV barrier for all resources on `list`, where a list takes one (a copy list and a bundle
/// do not).
unsafe fn flush_barrier(list: *mut c_void) {
    let ty = list_type(list);
    if ty == D3D12_COMMAND_LIST_TYPE_DIRECT || ty == D3D12_COMMAND_LIST_TYPE_COMPUTE {
        emit_uav_barrier_pc(list);
    }
}

/// command list #131 FlushGpuCacheRangeX / #132 FlushShaderCacheRangeX (cache flags, wait flags,
/// address, bytes), the newer layout's: the console writes its caches back (or drops what they
/// hold) for a range of memory and waits for the work before it. It is how a title orders a pass
/// behind the one that wrote what it reads – a retail title calls them thousands of times a minute,
/// and answered with nothing its passes had no order there. The PC has no ranges of caches: a UAV
/// barrier for everything.
pub(super) unsafe extern "system" fn flush_cache_range_x(this: *mut c_void, flags: u32, wait: u32, _va: u64, bytes: u64) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListBarrier);
    if (flags != 0 || wait != 0) && bytes != 0 {
        flush_barrier(this);
    }
}

/// command list #67 (newer layout: #69) FlushPipelineX(flags, address, bytes)
pub(super) unsafe extern "system" fn flush_pipeline_x(this: *mut c_void, _flags: u32, _va: u64, _bytes: u64) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListBarrier);
    flush_barrier(this);
}

/// command list #84 CopyResourceX(destination, source, flags); in place (the console's
/// compression operations) there is nothing to do on PC
pub(super) unsafe extern "system" fn copy_resource_flags_x(this: *mut c_void, dst: usize, src: usize, _flags: u32) {
    if dst != src {
        copy_resource_x(this, dst, src);
    }
}

/// Return `list`'s dedicated 256-byte predication buffer, creating it lazily in `COMMON` state.
unsafe fn predication_buffer(list: *mut c_void) -> Option<usize> {
    if let Some(p) = LISTS.lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .and_then(|s| s.pred_buf.as_ref())
        .map(|r| r.as_raw() as usize)
    {
        return Some(p);
    }
    let dev = device_of(list)?;
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let mut buf: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).ok()?;
    let buf = buf?;
    let (raw, extra) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        if let Some(existing) = s.pred_buf.as_ref() {
            (existing.as_raw() as usize, Some(buf))
        } else {
            let raw = buf.as_raw() as usize;
            s.pred_buf = Some(buf);
            (raw, None)
        }
    };
    drop(extra);
    Some(raw)
}

/// Before `Close()` on a command list: return `pred_buf` to `COMMON` if it is currently in
/// `D3D12_RESOURCE_STATE_PREDICATION`.
pub(super) unsafe fn before_close(list: *mut c_void) {
    if !PREDICATION_USED.load(Ordering::Acquire) || list.is_null() {
        return;
    }
    let pb = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else {
            return;
        };
        if s.pred_in_state {
            s.pred_in_state = false;
            s.pred_buf.as_ref().map(|r| r.as_raw() as usize)
        } else {
            None
        }
    };
    if let Some(pb) = pb {
        transition(list, pb, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
    }
}

/// Clear active predication state when `ID3D12GraphicsCommandList::Reset` resets a recording.
pub(super) fn on_pc_reset(list: *mut c_void) {
    reset_copy_scratch(list);
    if list.is_null() {
        return;
    }
    let clear_predication = PREDICATION_USED.load(Ordering::Acquire);
    let old_generations = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else { return };
        s.poisoned = false;
        if clear_predication {
            s.predication = None;
            s.pred_in_state = false;
        }
        s.take_native_generations()
    };
    drop(old_generations);
}

/// Copy 8 predicate bytes from `(uma_buf, off)` into `pb` with exact barrier transitions and
/// bind `SetPredication(pb, 0, op)` on `this`.
unsafe fn apply_predication_copy(this: *mut c_void, uma_buf: usize, off: u64, pb: usize, op: D3D12_PREDICATION_OP) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP);
    let f = pc::<F>(this, PC_SET_PREDICATION);
    let (was_active, was_in_state) = LISTS.lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(this as usize)))
        .map_or((false, false), |s| (s.predication.is_some(), s.pred_in_state));
    // Disarm any active predicate before copying the new predicate value so a failing predicate
    // does not suppress `CopyBufferRegion` when switching directly between buffers.
    if was_active {
        f(this, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    }
    transition(this, uma_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    let pb_before = if was_in_state { D3D12_RESOURCE_STATE_PREDICATION } else { D3D12_RESOURCE_STATE_COMMON };
    transition(this, pb, pb_before, D3D12_RESOURCE_STATE_COPY_DEST);
    copy_buffer_region(this, pb, 0, uma_buf, off, 8);
    transition(this, uma_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    transition(this, pb, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION);
    f(this, pb, 0, op);
    PREDICATION_USED.store(true, Ordering::Release);
    if let Some(s) = LISTS.lock().unwrap().get_or_insert_with(HashMap::new).get_mut(&(this as usize)) {
        s.predication = Some(op);
        s.pred_in_state = true;
    }
}

/// command list #85 SetPredicationBufferX({ u64 address; u32 operation }*)
///
/// # Safety
/// `this` must be a valid adopted `ID3D12GraphicsCommandList` pointer, and non-null `desc`
/// must point to readable memory if present.
pub unsafe extern "system" fn set_predication_buffer_x(this: *mut c_void, desc: *const u64) {
    static N: AtomicU64 = AtomicU64::new(0);
    let ty = list_type(this);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return warn(&N, "SetPredicationBufferX", format_args!("unsupported command list type {}", ty.0));
    }
    type F = unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP);
    if desc.is_null() {
        let f = pc::<F>(this, PC_SET_PREDICATION);
        f(this, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
        let pb = {
            let mut g = LISTS.lock().unwrap();
            g.as_mut().and_then(|m| m.get_mut(&(this as usize))).and_then(|s| {
                s.predication = None;
                if s.pred_in_state {
                    s.pred_in_state = false;
                    s.pred_buf.as_ref().map(|r| r.as_raw() as usize)
                } else {
                    None
                }
            })
        };
        if let Some(pb) = pb {
            transition(this, pb, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
        }
        return;
    }
    if !crate::gpu::shader::readable(desc as usize, 12) {
        return warn(&N, "SetPredicationBufferX", format_args!("descriptor {desc:p} is not readable"));
    }
    let (va, op_raw) = (desc.read_unaligned(), (desc.add(1) as *const u32).read_unaligned());
    if op_raw > 1 {
        return warn(&N, "SetPredicationBufferX", format_args!("unknown predication op {op_raw}"));
    }
    // (the GPU alone reads the predicate: in a GPU view, from its tile)
    super::bind::want_view(super::view_want::Path::MemoryOp, va, 8);
    let Some((uma_buf, off)) = real::uma_window_buffer_gpu(va as usize, 8).filter(|&(b, _)| b != 0 && va.is_multiple_of(8)) else {
        return warn(&N, "SetPredicationBufferX", format_args!("{va:#x} is not 8-byte aligned in unified memory"));
    };
    let Some(pb) = predication_buffer(this) else {
        return warn(&N, "SetPredicationBufferX", format_args!("could not allocate predication buffer"));
    };
    apply_predication_copy(this, uma_buf, off, pb, D3D12_PREDICATION_OP(op_raw as i32));
}

/// command list #93 KickoffX(): flush commands recorded so far by closing the list, submitting
/// it to a compatible live command queue, and resetting it on the same command allocator so
/// recording can continue seamlessly until the title's normal `Close` + `ExecuteCommandLists`.
///
/// If no compatible queue or allocator is known yet (e.g. a bundle, or a list created before
/// any queue of its type exists), `KickoffX` is a safe logged no-op and the recorded work runs
/// at `Close` + `ExecuteCommandLists`.
///
/// # Safety
/// `this` must be a valid adopted `ID3D12GraphicsCommandList` pointer.
pub unsafe extern "system" fn kickoff_x(this: *mut c_void) {
    static NO_QUEUE: AtomicU64 = AtomicU64::new(0);
    static ERR: AtomicU64 = AtomicU64::new(0);
    let mut physical = Vec::new();
    if !super::title_list::wrapper_of(this).is_some_and(|wrapper| super::title_list::push_segments(wrapper, &mut physical)) {
        physical.push(this);
    }
    if any_recording_poisoned(&physical) {
        return warn(&ERR, "KickoffX", format_args!("recording is poisoned by an unavailable pipeline bind"));
    }
    let ty = list_type(this);
    if ty == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return;
    }
    let dev = device_of(this);
    let dev_raw = dev.as_ref().map_or(0, |d| d.as_raw() as usize);
    let (alloc, queue) = {
        let g = LISTS.lock().unwrap();
        let Some(s) = g.as_ref().and_then(|m| m.get(&(this as usize))) else {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – list allocator unknown, deferring to Close + ExecuteCommandLists");
            }
            return;
        };
        if s.allocator == 0 {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – list allocator unknown, deferring to Close + ExecuteCommandLists");
            }
            return;
        }
        let Some(q) = pick_queue(s.last_queue, dev_raw, s.list_type.unwrap_or(ty)) else {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – no live queue of type {} yet, deferring to Close + ExecuteCommandLists", ty.0);
            }
            return;
        };
        (s.allocator, q)
    };
    let Some(queue_pin) = super::present::pin_queue_for_internal_work(queue) else {
        return warn(&ERR, "KickoffX", format_args!("selected queue is retiring; deferring to Close + ExecuteCommandLists"));
    };
    let wrapper = super::title_list::wrapper_of(this);
    let standby = if let Some(wrapper) = wrapper {
        let Some(dev) = dev.as_ref() else {
            return warn(&ERR, "KickoffX", format_args!("could not get the command-list device for fail-closed preflight"));
        };
        if !super::title_list::prepare_fail_closed(wrapper) {
            return warn(&ERR, "KickoffX", format_args!("could not reserve fail-closed bookkeeping"));
        }
        let Some(standby) = reserve_closed_segment(wrapper, dev, ty) else {
            return warn(&ERR, "KickoffX", format_args!("could not reserve a closed standby"));
        };
        Some((wrapper, standby))
    } else {
        None
    };
    let (bound_pso, depth_bias, index_buffer, depth_bounds, predication, pred_buf, pred_in_state, active_query, preclose_effects, current_trace) = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(this as usize))) else {
            if let Some((wrapper, standby)) = standby {
                super::title_list::put_spare(wrapper, standby);
            }
            return;
        };
        let active_query = s.occlusion.end_active_early();
        let preclose_effects = take_kickoff_effects_before_close(s, standby.is_some());
        let pred_in_state = std::mem::replace(&mut s.pred_in_state, false);
        let pred_buf = s.pred_buf.as_ref().map(|r| r.as_raw() as usize);
        let current_trace = sync_state_tracing().then(|| s.write64_trace.clone()).flatten();
        (s.bound_pso, s.depth_bias, s.index_buffer, s.depth_bounds, s.predication, pred_buf, pred_in_state, active_query, preclose_effects, current_trace)
    };

    super::bind::flush_two_pass(this, super::bind::RunEnd::Segment);
    // End any active occlusion query before Close (PC forbids closing with an open query).
    if let Some(((h, i), qty)) = active_query {
        pc::<QueryFn>(this, PC_END_QUERY)(this, h, qty, i);
    }
    if pred_in_state {
        if let Some(pb) = pred_buf {
            transition(this, pb, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
        }
    }

    super::gpu_times::before_close(this);
    let ext = *(this as *const usize);
    let close_hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, PC_CLOSE)(this);
    real::preserve_vtable(this, ext);
    if close_hr < 0 {
        if let Some(effects) = preclose_effects {
            restore_kickoff_native_generations(this as usize, effects.native_generations);
        }
        if let Some((wrapper, standby)) = standby {
            super::title_list::fail_closed(wrapper, this, standby, true);
        }
        return warn(&ERR, "KickoffX", format_args!("Close failed ({close_hr:#x})"));
    }
    let effects = match preclose_effects {
        Some(effects) => effects,
        None => LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))).map(take_kickoff_effects).unwrap_or(KickoffEffects {
            waits: Vec::new(), sparse_writes: (Vec::new(), Vec::new()), texture_writes: Vec::new(), texture_transitions: Vec::new(), uses_shared_view_heap: false, write64_signals: Vec::new(), cpu_writes: Vec::new(), native_generations: Vec::new(),
        }),
    };
    let KickoffEffects {
        waits, sparse_writes, texture_writes, texture_transitions, uses_shared_view_heap,
        write64_signals, cpu_writes, native_generations,
    } = effects;
    let kickoff_wrapper = super::title_list::wrapper_of(this);
    if let Some(wrapper) = kickoff_wrapper {
        record_deferred(wrapper);
    }
    super::capture::kickoff(this, queue);

    let q_raw = queue_pin.as_raw();
    real::prepare_title_submission(q_raw, &[this]);
    super::gpu_profile::note_queue(q_raw);
    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let exec: ExecFn = transmute_copy(&real::orig_fn(q_raw, PC_QUEUE_EXECUTE_COMMAND_LISTS));

    // If this list was previously split mid-recording around a memory wait, flush all earlier closed
    // segments in exact order before submitting `this`, then resume recording on `primary`.
    let kickoff_split = super::title_list::reclaim_for_kickoff(this);
    let mut prior_cpu_completions = Vec::new();
    let mut wait_after_cpu_write = false;
    let mut gate = super::alloc_pool::SubmissionGate::default();
    let track_shared_view_epochs = super::vram::srv_tracking_enabled();
    if let Some((_, ref prior, _)) = kickoff_split {
        if !prior.is_empty() {
            let plan = submission_plan(prior);
            wait_after_cpu_write = plan.cpu_producer_overlaps(&waits, plan.effects.len());
            let mut submitted = Vec::with_capacity(prior.len());
            let (completions, _) = execute_submission_plan_with_completions(q_raw, &plan, &waits, |chunk| {
                super::queue_submit::with(q_raw as usize, || {
                    let lists = chunk.lists();
                    if !submit_title_chunk(
                        &mut gate,
                        q_raw,
                        lists,
                        track_shared_view_epochs && chunk.uses_shared_view_heap(),
                        chunk.has_native_generations(),
                        chunk.native_generations(),
                        || {},
                        || exec(q_raw, lists.len() as u32, lists.as_ptr()),
                    ) {
                        return (false, None);
                    }
                    super::gpu_times::executed(q_raw, lists);
                    submitted.extend_from_slice(lists);
                    copy_scratch_submitted(q_raw, lists);
                    let completion = chunk.submit_effects(q_raw);
                    note_submitted(q_raw);
                    (true, completion)
                })
            });
            prior_cpu_completions = completions;
            // (`XCL_DUMP_TARGETS`' 3D textures: what ran is in their states, the segments record anew)
            super::lut3d::submitted(&submitted);
            for &p in &submitted {
                super::lut3d::list_reset(p as usize);
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(p as usize))) {
                    s.texture_transitions.clear();
                }
            }
        }
    }
    if !gate.failed() && !waits.is_empty() {
        if wait_after_cpu_write {
            for completion in prior_cpu_completions.drain(..) {
                let _ = completion.wait();
            }
        }
        if sparse_again_after_split(wait_all_on(q_raw, &waits, &write64_signals), split_sparse_after_cpu_waits_only()) {
            mapped_after_wait(q_raw);
        }
    }

    let one = [this];
    let current_submitted = super::queue_submit::with(q_raw as usize, || {
        let submitted = submit_title_chunk(
            &mut gate,
            q_raw,
            &one,
            track_shared_view_epochs && uses_shared_view_heap,
            !native_generations.is_empty(),
            native_generations.iter(),
            || {},
            || exec(q_raw, 1, one.as_ptr()),
        );
        if submitted {
            if let Some(trace) = &current_trace {
                mark_trace(trace, ListEvent::Executed);
            }
            super::gpu_times::executed(q_raw, &one);
            super::lut3d::submitted(&one);
            super::lut3d::list_reset(this as usize);
            copy_scratch_submitted(q_raw, &one);
            super::title_textures::publish_submission_effects(queue_identity(q_raw as usize), &texture_writes, &texture_transitions);
            if let Some(w) = kickoff_wrapper {
                super::title_textures::forget_list_closed(w as usize);
            }
            signal_write64_pairs(q_raw, &write64_signals);
            if !sparse_writes.0.is_empty() || !sparse_writes.1.is_empty() {
                if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&q_raw) {
                    super::sparse::readback_after_execute(q, &sparse_writes.0, &sparse_writes.1);
                }
            }
            super::cpu_writes::after(q_raw, cpu_writes);
            note_submitted(q_raw);
        }
        submitted
    });
    if !current_submitted {
        if once(&ERR, 8) {
            crate::xlog_warn!("D3D12.X: KickoffX – could not reserve submission retirement state; resetting the unsubmitted recording");
        }
    }
    // Submitted or discarded, `this` starts a new recording in the common Reset tail.

    let target = if let Some((primary, _, reclaimed)) = kickoff_split {
        let (ended_early, pred_buf_obj, bindings_copy) = {
            let mut g = LISTS.lock().unwrap();
            if let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                (std::mem::take(&mut s.occlusion.ended_early), s.pred_buf.clone(), s.bindings.clone())
            } else {
                (Vec::new(), None, Bindings::default())
            }
        };
        // (`primary` records on the PC allocator the title's allocator stands for: its Reset below)
        super::bind::inherit_for_split(this, primary, super::alloc_pool::current(alloc));
        for seg in reclaimed {
            super::bind::reset(seg as *mut c_void, 0, 0);
            clear_segment_state(seg);
            super::alloc_pool::retire_recording(seg);
        }
        let old_generations = {
            let mut g = LISTS.lock().unwrap();
            if let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(primary as usize))) {
                s.allocator = alloc;
                s.bound_pso = bound_pso;
                s.depth_bias = depth_bias;
                s.index_buffer = index_buffer;
                s.depth_bounds = depth_bounds;
                s.predication = predication;
                s.pred_buf = pred_buf_obj;
                s.pred_in_state = false;
                s.occlusion = Occlusion { active: None, ended_early };
                s.written_ranges.clear();
                s.segment_write_producers.clear();
                s.clear_sparse_writes();
                s.clear_write64_signals();
                s.clear_waits();
                s.texture_writes.clear();
                s.cpu_writes.clear();
                s.bindings = bindings_copy;
                s.take_native_generations()
            } else {
                Vec::new()
            }
        };
        drop(old_generations);
        primary
    } else {
        this
    };

    let ext_closed = *(target as *const usize);
    let (init_pso, init_generation) = super::bind::initial_pipeline(target);
    let reset_hr = super::alloc_pool::reset_list(target as usize, alloc, list_type(target).0, |pc_alloc| {
        pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(target, PC_RESET)(target, pc_alloc, init_pso)
    });
    real::preserve_vtable(target, ext_closed);
    settle_kickoff_standby(
        standby,
        reset_hr >= 0,
        |wrapper, standby| super::title_list::put_spare(wrapper, standby),
        |wrapper, standby, producer_broken| super::title_list::fail_closed(wrapper, target, standby, producer_broken),
    );
    if reset_hr < 0 {
        return warn(&ERR, "KickoffX", format_args!("Reset after kickoff failed ({reset_hr:#x})"));
    }
    record_native_generation(target as usize, init_generation);
    trace_list(target as usize, ListEvent::Reset);
    reset_copy_scratch(target);
    let bindings = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(target as usize))).map(|s| s.bindings.clone());
    if let Some(b) = bindings {
        restore_bindings(target, &b, ty);
    }

    // Reset() cleared the PC list's native RSSetDepthBias state to (0, 0, 0) and bound init_pso.
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
        s.native_bias = DepthBias::ZERO;
        s.using_fallback_pso = false;
        s.pso_dirty = false;
        // (other queues may run between what went and what follows)
        s.depth_in_step.clear();
    }
    if ty == D3D12_COMMAND_LIST_TYPE_DIRECT {
        let info = if bound_pso != 0 { device_x::base_depth_bias_info(bound_pso) } else { None };
        let mut applied_native = false;
        if let Some((true, static_bias)) = info {
            let target_bias = depth_bias.unwrap_or(static_bias);
            if target_bias.is_zero() {
                applied_native = true;
            } else if try_native_depth_bias(target, target_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
                    s.native_bias = target_bias;
                }
                applied_native = true;
            }
        } else if bound_pso == 0 {
            if let Some(b) = depth_bias {
                if device_x::dynamic_depth_bias_supported(dev_raw as *mut c_void) && try_native_depth_bias(target, b) {
                    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
                        s.native_bias = b;
                    }
                    applied_native = true;
                }
            }
        }
        if depth_bias.is_some() && !applied_native {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
        if let Some(ref view) = index_buffer {
            set_index_buffer(target, Some(view));
        }
        if let Some((min, max)) = depth_bounds {
            pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(target, PC_OM_SET_DEPTH_BOUNDS)(target, min, max);
        }
    }
    if let (Some(op), Some(pb)) = (predication, pred_buf) {
        transition(target, pb, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PREDICATION);
        pc::<unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP)>(target, PC_SET_PREDICATION)(target, pb, 0, op);
        if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(target as usize))) {
            s.pred_in_state = true;
        }
    }
}

/// command list #94 RSSetDepthBiasX(depthBias, depthBiasClamp, slopeScaledDepthBias):
/// dynamic depth bias override. Uses `ID3D12GraphicsCommandList9::RSSetDepthBias` natively
/// when `D3D12_FEATURE_D3D12_OPTIONS16::DynamicDepthBiasSupported` and the bound PSO support
/// it (preserving exact `f32` bias with zero PSO derivation), falling back to the patched-PSO
/// cache (`device_x::depth_bias_pso`) otherwise.
pub(super) unsafe extern "system" fn set_depth_bias_x(this: *mut c_void, depth_bias: f32, depth_bias_clamp: f32, slope_scaled_depth_bias: f32) {
    static NON_FINITE: AtomicU64 = AtomicU64::new(0);
    let clean = |v: f32| {
        if v.is_finite() {
            v
        } else {
            if once(&NON_FINITE, 4) {
                crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – non-finite value {v} clamped to 0.0");
            }
            0.0
        }
    };
    let bias = DepthBias::new(clean(depth_bias), clean(depth_bias_clamp), clean(slope_scaled_depth_bias));
    let key = this as usize;

    let (bound_pso, is_bundle) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        (s.bound_pso, s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE))
    };

    let can_native = if is_bundle {
        false
    } else if bound_pso != 0 {
        matches!(device_x::base_depth_bias_info(bound_pso), Some((true, _)))
    } else {
        device_of(this).is_some_and(|dev| device_x::dynamic_depth_bias_supported(dev.as_raw()))
    };

    if can_native {
        let target = if bias.is_zero() { DepthBias::ZERO } else { bias };
        let (need_restore_base, need_native_call) = {
            let mut g = LISTS.lock().unwrap();
            let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
            s.depth_bias = if bias.is_zero() { None } else { Some(bias) };
            s.pso_dirty = false;
            let restore = s.using_fallback_pso && s.bound_pso != 0;
            s.using_fallback_pso = false;
            (if restore { Some(s.bound_pso) } else { None }, target != s.native_bias)
        };
        if let Some(base) = need_restore_base {
            let resolved = resolve_and_record_pipeline(this as usize, base);
            pc::<unsafe extern "system" fn(*mut c_void, usize)>(this, PC_SET_PIPELINE_STATE)(this, resolved);
        }
        if !need_native_call {
            return;
        }
        if try_native_depth_bias(this, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.native_bias = target;
            }
            return;
        }
        // ID3D12GraphicsCommandList9 unavailable: fall through to patched-PSO fallback.
        if !bias.is_zero() {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
        return;
    }

    let restore = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        if bias.is_zero() {
            let was_active = s.depth_bias.take().is_some();
            let was_fallback = std::mem::take(&mut s.using_fallback_pso);
            s.pso_dirty = false;
            if (was_active || was_fallback) && s.bound_pso != 0 {
                Some(s.bound_pso)
            } else {
                None
            }
        } else {
            if s.depth_bias != Some(bias) || !s.using_fallback_pso {
                s.depth_bias = Some(bias);
                s.pso_dirty = true;
                DEPTH_BIAS_USED.store(true, Ordering::Release);
            }
            None
        }
    };
    if let Some(base) = restore {
        let resolved = resolve_and_record_pipeline(this as usize, base);
        pc::<unsafe extern "system" fn(*mut c_void, usize)>(this, PC_SET_PIPELINE_STATE)(this, resolved);
    }
}

/// Apply a pending fallback `RSSetDepthBiasX` override before a draw call on `list`.
pub(super) unsafe fn prepare_draw(list: *mut c_void) {
    if !DEPTH_BIAS_USED.load(Ordering::Acquire) {
        return;
    }
    let (bound_pso, bias) = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else {
            return;
        };
        if !s.pso_dirty {
            return;
        }
        s.pso_dirty = false;
        (s.bound_pso, s.depth_bias)
    };
    if bound_pso == 0 {
        return;
    }
    let (target, is_fallback) = match bias {
        Some(b) => {
            static MISS: AtomicU64 = AtomicU64::new(0);
            let Some(dev) = device_of(list) else { return };
            match device_x::depth_bias_pso(dev.as_raw(), bound_pso, b) {
                Some(derived) => (derived, true),
                None => {
                    if once(&MISS, 4) {
                        crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – could not derive pipeline for {bound_pso:#x}, using base pipeline");
                    }
                    return;
                }
            }
        }
        None => (bound_pso, false),
    };
    let resolved = resolve_and_record_pipeline(list as usize, target);
    pc::<unsafe extern "system" fn(*mut c_void, usize)>(list, PC_SET_PIPELINE_STATE)(list, resolved);
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        s.using_fallback_pso = is_fallback;
    }
}

// ---------------------------------------------------------------------------------------
// index buffers & DrawIndexedX
// ---------------------------------------------------------------------------------------

/// a console GPU address as the PC's (unified memory: the window buffer's)
/// A GPU address the title gave: its memory's PC address when it is the title's memory, else
/// as it is (a PC resource's GPU address – it may lie in a window's range by number).
fn pc_va(va: u64) -> u64 {
    match crate::gpu::uma::get() {
        Some(u) => pc_va_in(u, va, real::uma_window_va),
        None => va,
    }
}

/// `pc_va` in `u`, whose windows have the PC addresses `window_va` (`real::uma_window_va`). A
/// GPU view of the title's memory without a PC address of its window gets none (0, read as
/// nothing): as it is, the console address is no PC address and hangs the device.
fn pc_va_in(u: &crate::gpu::uma::Uma, va: u64, window_va: impl Fn(usize) -> Option<u64>) -> u64 {
    let p = va as usize;
    let Some(k) = u.window_of(p).filter(|_| u.is_title_address(p)) else { return va };
    match window_va(k) {
        Some(w) => w + (p - u.window(k).0) as u64,
        None if u.is_view_window(k) => 0,
        None => va,
    }
}

/// The bytes of a view of `bytes` at the title's address `va` that the PC can bind as one range:
/// up to the end of the memory window `va` is in. The title's GPU view is one range, but its
/// windows are separate buffers on the PC (a GPU view spans six of 4 GiB): a view across a
/// window's end, bound as it is, runs past that window's buffer into addresses nothing maps.
/// The GPU reads zeros past the clamped size. (Logged once.)
fn view_bytes(va: u64, bytes: u32) -> u32 {
    match crate::gpu::uma::get().and_then(|u| bytes_in_window(u, va, bytes)) {
        Some(n) => {
            static N: AtomicBool = AtomicBool::new(false);
            if !N.swap(true, Ordering::Relaxed) {
                crate::xlog_warn!("D3D12.X: a vertex or index buffer at {va:#x} ({bytes} bytes) runs past the end of its memory window: bound up to it ({n} bytes)");
            }
            n
        }
        None => bytes,
    }
}

/// `view_bytes` in `u`: the clamped size when `[va, va + bytes)` starts in the title's memory
/// of a window and runs past that window's end (None: it does not).
fn bytes_in_window(u: &crate::gpu::uma::Uma, va: u64, bytes: u32) -> Option<u32> {
    let p = va as usize;
    let k = u.window_of(p)?;
    let (base, size) = u.window(k);
    let room = (base + size).checked_sub(p)?;
    // (the title's memory is looked up only for a view across the end: a lock)
    (bytes as usize > room && u.is_title_address(p)).then_some(room as u32)
}

/// `pc_va` for an index or vertex buffer of `bytes` bytes bound on `list`. A retail title also
/// binds buffers in its ordinary memory (a private allocation outside the unified-memory
/// windows – on the console the GPU reads all of the title's memory). The PC's GPU has no
/// address for that (IASetIndexBuffer of it: debug layer message 725, then the device hangs,
/// 0x887a0006), so the bytes are copied into the list's upload memory as they are at the bind;
/// what the title writes there afterwards is not seen.
///
/// A buffer that stays bound (`mirror`) gets one block of unified memory for good – its
/// *mirror*, the same for every bind of that address and size – and each draw copies the part
/// it reads there first (`sync_title_buffers`), as the title wrote it by then. (A copy of the
/// whole buffer at every bind filled unified memory within two minutes of a retail title's
/// menus, and missed what the title wrote after the bind.)
unsafe fn gpu_va(list: *mut c_void, va: u64, bytes: usize, what: &str, mirror: bool) -> u64 {
    // in a GPU view: its window's PC address, over tiles asked for here
    super::bind::want_view(super::view_want::Path::VertexIndex, va, bytes as u64);
    let t = pc_va(va);
    if t != va || va == 0 || bytes == 0 {
        return t;
    }
    // in a window by number: a PC resource's GPU address (never mirrored: a view window is no
    // memory the CPU reads)
    if crate::gpu::uma::get().is_none_or(|u| u.contains(va as usize) || u.is_view(va as usize)) {
        return t;
    }
    // a buffer bound before has its mirror: no `readable` (a `VirtualQuery` walk of the title's
    // memory when it is not resident – most of a retail title's render thread at a bind per
    // draw); `sync_title_buffers` checks what it reads
    if let Some(p) = mirror.then(|| mirrored(va, bytes)).flatten() {
        return pc_va(p as u64);
    }
    if !crate::gpu::shader::readable(va as usize, bytes) {
        return t;
    }
    if mirror {
        if let Some(p) = mirror_of(va, bytes, what) {
            return pc_va(p as u64);
        }
    }
    let p = super::bind::allocate_upload_memory(list, bytes, 16);
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if p.is_null() {
        if n < 4 {
            crate::xlog_warn!("D3D12.X: {what} at {va:#x} ({bytes} bytes) is outside unified memory and there is no room for a copy");
        }
        return t;
    }
    std::ptr::copy_nonoverlapping(va as *const u8, p, bytes);
    if n < 4 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: {what} at {va:#x} ({bytes} bytes) is outside unified memory – bound as a copy made now (#{n})");
    }
    pc_va(p as u64)
}

/// Ranges of the title's memory with a mirror in unified memory: (start, bytes, mirror). A
/// title binds one buffer at many offsets (a ring it fills through the frame: a retail title's
/// views all end at the buffer's end), so a mirror covers a 64 KiB-aligned range and a view
/// inside a mirrored range uses that mirror at its offset.
static MIRRORS: Mutex<Vec<(u64, usize, usize)>> = Mutex::new(Vec::new());
/// bytes of mirrors made (never freed)
static MIRROR_BYTES: AtomicU64 = AtomicU64::new(0);
/// Append generation. Incremented while `MIRRORS` is locked, before publishing a new range, so a
/// lookup that observes it refreshes under that lock and cannot retain a stale negative result.
static MIRROR_GENERATION: AtomicU64 = AtomicU64::new(0);
std::thread_local! {
    /// Mirrors are append-only. A full ordered snapshot answers both hits and authoritative misses
    /// without the global lock until the append generation changes.
    static MIRROR_LOOKUP: std::cell::RefCell<(u64, Vec<(u64, usize, usize)>)> = const { std::cell::RefCell::new((0, Vec::new())) };
}

fn mirror_in(ranges: &[(u64, usize, usize)], va: u64, bytes: usize) -> Option<usize> {
    ranges.iter()
        .find(|&&(start, len, _)| va >= start && va + bytes as u64 <= start + len as u64)
        .map(|&(start, _, mirror)| mirror + (va - start) as usize)
}

fn cached_mirror(
    cache: &mut (u64, Vec<(u64, usize, usize)>), generation: u64, va: u64, bytes: usize,
    refresh: impl FnOnce() -> Vec<(u64, usize, usize)>,
) -> Option<usize> {
    if cache.0 != generation {
        cache.1 = refresh();
        cache.0 = generation;
    }
    mirror_in(&cache.1, va, bytes)
}

/// Where `[va, va + bytes)` is in a mirror, if it is in one.
fn mirrored(va: u64, bytes: usize) -> Option<usize> {
    let generation = MIRROR_GENERATION.load(Ordering::Acquire);
    if generation == 0 {
        return None;
    }
    MIRROR_LOOKUP.with(|cache| {
        cached_mirror(&mut cache.borrow_mut(), generation, va, bytes, || MIRRORS.lock().unwrap().clone())
    })
}

/// The mirror address of the title's buffer at `va`; its range's mirror is made (and filled
/// once) the first time.
unsafe fn mirror_of(va: u64, bytes: usize, what: &str) -> Option<usize> {
    if let Some(p) = mirrored(va, bytes) {
        return Some(p);
    }
    let mut g = MIRRORS.lock().unwrap();
    // again under the lock: two lists binding one new range at once must share a mirror (a
    // second one would never be synced – every later lookup finds the first)
    if let Some(&(s, _, m)) = g.iter().find(|&&(s, l, _)| va >= s && va + bytes as u64 <= s + l as u64) {
        return Some(m + (va - s) as usize);
    }
    let start = va & !0xffff;
    let len = ((va + bytes as u64 + 0xffff) & !0xffff) - start;
    // only as much as is the title's readable memory (the view itself is)
    let (start, len) = if crate::gpu::shader::readable(start as usize, len as usize) { (start, len as usize) } else { (va, bytes) };
    if MIRROR_BYTES.load(Ordering::Relaxed) + len as u64 > 2 << 30 {
        static FULL: AtomicBool = AtomicBool::new(false);
        if !FULL.swap(true, Ordering::Relaxed) {
            crate::xlog_warn!("D3D12.X: 2 GiB of mirrors for buffers outside unified memory – further ones ({what} at {va:#x}) are bound as a copy made at the bind, which later writes do not reach");
        }
        return None;
    }
    let u = crate::gpu::uma::get()?;
    let p = match u.has_sparse().then(|| super::sparse::alloc(len, 0x1_0000)).flatten() {
        Some((p, _)) => p,
        None => u.alloc(len, 0x1_0000)? as usize,
    };
    std::ptr::copy_nonoverlapping(start as *const u8, p as *mut u8, len);
    // Invalidate cached misses before publishing while still holding the lock. A lookup that sees
    // the new generation blocks on `g` until this range is present in the refreshed snapshot.
    MIRROR_GENERATION.fetch_add(1, Ordering::Release);
    g.push((start, len, p));
    let total = MIRROR_BYTES.fetch_add(len as u64, Ordering::Relaxed) + len as u64;
    if g.len() <= 8 || g.len().is_power_of_two() {
        crate::xlog_warn!("D3D12.X: {what} at {va:#x} ({bytes} bytes) is outside unified memory – {start:#x}..{:#x} mirrored at {p:#x}, each draw copies what it reads ({} mirrors, {} MB)", start + len as u64, g.len(), total >> 20);
    }
    Some(p + (va - start) as usize)
}

/// Before a draw on `list`: the parts of mirrored buffers it reads are copied from the title's
/// memory. `indexed`: (first index, index count, base vertex); else `first_vertex`, `vertices`.
pub(super) unsafe fn sync_title_buffers(list: *mut c_void, indexed: Option<(u32, u32, i32)>, first_vertex: u32, vertices: u32, instances: u32) {
    if MIRROR_BYTES.load(Ordering::Relaxed) == 0 {
        return;
    }
    let (vbs, ib) = {
        let g = LISTS.lock().unwrap();
        let Some(b) = g.as_ref().and_then(|m| m.get(&(list as usize))).map(|s| &s.bindings) else { return };
        (b.vertex_titles.values().copied().collect::<Vec<_>>(), b.index_title)
    };
    let mirrors: Vec<Option<usize>> = {
        let mut v: Vec<Option<usize>> = vbs.iter().map(|&(va, bytes, _)| mirrored(va, bytes as usize)).collect();
        v.push(ib.and_then(|(va, bytes, _)| mirrored(va, bytes as usize)));
        if v.iter().all(Option::is_none) {
            return;
        }
        v
    };
    let copy = |va: u64, mirror: usize, bytes: u32, from: u64, len: u64| {
        let from = from.min(bytes as u64);
        let len = len.min(bytes as u64 - from) as usize;
        if len > 0 && crate::gpu::shader::readable(va as usize + from as usize, len) {
            std::ptr::copy_nonoverlapping((va + from) as *const u8, (mirror + from as usize) as *mut u8, len);
        }
    };
    // the vertices read: [lo, hi)
    let (lo, hi) = match (indexed, ib) {
        (Some((first, count, base)), Some((va, bytes, format))) => {
            let size: u64 = if format == DXGI_FORMAT_R32_UINT.0 { 4 } else { 2 };
            let (from, len) = (first as u64 * size, count as u64 * size);
            if let Some(m) = mirrors[vbs.len()] {
                copy(va, m, bytes, from, len);
            }
            let n = (len.min((bytes as u64).saturating_sub(from)) / size) as usize;
            if n == 0 || !crate::gpu::shader::readable((va + from) as usize, n * size as usize) {
                (0, u64::MAX)
            } else {
                let at = (va + from) as *const u8;
                let (mut lo, mut hi) = (u32::MAX, 0u32);
                for k in 0..n {
                    let i = if size == 4 { at.add(4 * k).cast::<u32>().read_unaligned() } else { at.add(2 * k).cast::<u16>().read_unaligned() as u32 };
                    // (a strip's cut value counts as an index: whether 0xffff is one is the
                    // topology's to say, and a range too wide only copies more)
                    lo = lo.min(i);
                    hi = hi.max(i);
                }
                if lo > hi {
                    return;
                }
                ((lo as i64 + base as i64).max(0) as u64, (hi as i64 + base as i64 + 1).max(0) as u64)
            }
        }
        (Some(_), None) => (0, u64::MAX),
        (None, _) => (first_vertex as u64, first_vertex as u64 + vertices as u64),
    };
    for (k, &(va, bytes, stride)) in vbs.iter().enumerate() {
        let Some(m) = mirrors[k] else { continue };
        // a buffer stepped by instance is not told apart: with instances (callers pass more
        // than 1 for a first instance other than 0 too), all of it
        if instances > 1 || hi == u64::MAX || stride == 0 || hi.saturating_mul(stride as u64) > bytes as u64 {
            copy(va, m, bytes, 0, bytes as u64);
        } else {
            copy(va, m, bytes, lo * stride as u64, (hi - lo) * stride as u64);
        }
    }
}

unsafe fn set_index_buffer(list: *mut c_void, view: Option<&D3D12_INDEX_BUFFER_VIEW>) {
    let f = pc::<unsafe extern "system" fn(*mut c_void, *const D3D12_INDEX_BUFFER_VIEW)>(list, PC_IA_SET_INDEX_BUFFER);
    f(list, view.map_or(std::ptr::null(), |v| v as *const _));
}

/// Ensure the shared shader-visible PC sampler heap exists when a console device is live so
/// directly-indexed root signatures (`SAMPLER_HEAP_DIRECTLY_INDEXED`) never run with 0 sampler
/// heaps bound (D3D12 #1419 / `0x887A0001`).
unsafe fn ensure_global_sampler_heap() -> Option<ID3D12DescriptorHeap> {
    if let Some(s) = super::console_heap::global_sampler_heap() {
        return Some(s);
    }
    let dev_raw = real::device_raw();
    if let Some(dev) = ID3D12Device::from_raw_borrowed(&dev_raw) {
        let _ = super::console_heap::shared_sampler_heap(dev);
    }
    super::console_heap::global_sampler_heap()
}

/// Build the default PC descriptor heap list to bind when a command list is missing a required
/// shader-visible view heap (`heap_windows`) or sampler heap (`SAMPLER_HEAP_DIRECTLY_INDEXED`).
/// Returns `Some((heaps, has_sampler_heap))` when `SetDescriptorHeaps` must be updated.
fn default_bound_heaps(view: Option<usize>, sampler: Option<usize>, current: &[usize], has_sampler: bool) -> Option<(Vec<usize>, bool)> {
    if current.is_empty() {
        let mut out = Vec::with_capacity(2);
        if let Some(v) = view {
            out.push(v);
        }
        if let Some(s) = sampler {
            out.push(s);
        }
        return (!out.is_empty()).then_some((out, sampler.is_some()));
    }
    if current.len() == 1 {
        if !has_sampler {
            if let Some(s) = sampler {
                return Some((vec![current[0], s], true));
            }
        } else if let Some(v) = view {
            return Some((vec![v, current[0]], true));
        }
    }
    None
}

/// The heaps a list binds for console code: view and sampler heaps hosted in the shared
/// shader-visible heaps (`console_heap`) bind those shared heaps whatever the title names
/// (a console descriptor table may be copied via raw CPU memcpy from any heap); heaps that
/// kept their own shader-visible PC heap bind themselves for their local indices. Heaps whose
/// own PC backing was downgraded to CPU-only cannot be bound directly.
pub(super) unsafe fn pc_heaps_with_shared(n: u32, heaps: *const usize) -> (Vec<Option<ID3D12DescriptorHeap>>, bool) {
    let mut view_heap = super::console_heap::global_heap();
    let shared_view = view_heap.as_ref().map(|heap| heap.as_raw() as usize);
    let mut sampler_heap = ensure_global_sampler_heap();
    let mut out = Vec::new();
    for k in 0..n as usize {
        let p = *heaps.add(k) as *mut c_void;
        if p.is_null() || super::console_heap::in_global(p as usize) || super::console_heap::in_global_sampler(p as usize) {
            continue;
        }
        let Some(h) = ID3D12DescriptorHeap::from_raw_borrowed(&p) else { continue };
        let desc = h.GetDesc();
        if desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV {
            if desc.Flags.contains(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
                view_heap = Some(h.clone());
            }
        } else if desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER {
            if desc.Flags.contains(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
                sampler_heap = Some(h.clone());
            }
        } else {
            out.push(Some(h.clone()));
        }
    }
    let uses_shared_view = shared_view.is_some_and(|shared| view_heap.as_ref().is_some_and(|heap| heap.as_raw() as usize == shared));
    out.extend(view_heap.map(Some));
    out.extend(sampler_heap.map(Some));
    (out, uses_shared_view)
}

/// Translated shaders with directly-indexed root signatures (`CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`
/// / `SAMPLER_HEAP_DIRECTLY_INDEXED`) and heap windows (`pso::heap_windows`) require valid
/// shader-visible view and sampler heaps to be bound even if the title bound 0 heaps (or only a
/// view heap with 0 sampler heaps).
pub(super) unsafe fn ensure_view_heap(this: *mut c_void) -> bool {
    let view = if crate::gpu::recomp::pso::heap_windows() {
        super::console_heap::global_heap()
    } else {
        None
    };
    let sampler = ensure_global_sampler_heap();
    let view_raw = view.as_ref().map(|h| h.as_raw() as usize);
    let sampler_raw = sampler.as_ref().map(|h| h.as_raw() as usize);
    if view_raw.is_none() && sampler_raw.is_none() {
        return false;
    }
    let mut to_bind: Option<Vec<usize>> = None;
    let mut shared_view_bound = false;
    with_bindings(this, |b| {
        if let Some((next, has_sampler)) = default_bound_heaps(view_raw, sampler_raw, &b.heaps, b.has_sampler_heap) {
            b.heaps = next.clone();
            b.has_sampler_heap = has_sampler;
            b.shared_view_heap |= view_raw.is_some_and(|view| next.contains(&view));
            to_bind = Some(next);
        }
        shared_view_bound = b.shared_view_heap;
    });
    if let Some(raw) = to_bind {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(this, 28)(this, raw.len() as u32, raw.as_ptr());
    }
    shared_view_bound
}

/// console #32 SetDescriptorHeaps (PC #28)
pub(super) unsafe extern "system" fn set_descriptor_heaps_x(this: *mut c_void, n: u32, heaps: *const usize) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBind);
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::BindState);
    const PC_SET_DESCRIPTOR_HEAPS: usize = 28;
    let f = pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(this, PC_SET_DESCRIPTOR_HEAPS);
    let _c = super::capture::op(this, || {
        let given = if heaps.is_null() { &[][..] } else { std::slice::from_raw_parts(heaps, n as usize) };
        Some(super::capture::Op::Heaps(given.iter().map(|&h| super::console_heap::base(h).map_or((h, 0, false), |(s, b)| (h, b, s))).collect()))
    });
    super::bind::heaps_changed(this);
    super::bind::set_heaps(this, if heaps.is_null() { &[] } else { std::slice::from_raw_parts(heaps, n as usize) });
    if heaps.is_null() || n == 0 {
        // no heaps: nothing bound any more (`ensure_view_heap` binds the shared heaps again
        // before a translated draw or dispatch reads memory windows or samplers from them)
        with_bindings(this, |b| {
            b.heaps.clear();
            b.has_sampler_heap = false;
            b.shared_view_heap = false;
        });
        return f(this, n, heaps);
    }
    if super::console_heap::global_heap().is_none() && super::console_heap::global_sampler_heap().is_none() {
        let given = std::slice::from_raw_parts(heaps, n as usize).to_vec();
        let has_sampler = given.iter().any(|&p| {
            let p = p as *mut c_void;
            ID3D12DescriptorHeap::from_raw_borrowed(&p).is_some_and(|h| h.GetDesc().Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
        });
        with_bindings(this, |b| {
            b.heaps = given;
            b.has_sampler_heap = has_sampler;
            b.shared_view_heap = false;
        });
        return f(this, n, heaps);
    }
    let (h, shared_view_heap) = pc_heaps_with_shared(n, heaps);
    let has_sampler = h.iter().any(|x| x.as_ref().is_some_and(|h| h.GetDesc().Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER));
    let raw: Vec<usize> = h.iter().map(|x| x.as_ref().map_or(0, |x| x.as_raw() as usize)).collect();
    let kept = raw.clone();
    with_bindings(this, |b| {
        b.heaps = kept;
        b.has_sampler_heap = has_sampler;
        b.shared_view_heap = shared_view_heap;
    });
    f(this, raw.len() as u32, raw.as_ptr())
}

const PC_IA_SET_VERTEX_BUFFERS: usize = 44;

/// command list #48 IASetVertexBuffers(start slot, count, views) (PC #44): the addresses are
/// the title's (its memory: the GPU address of that memory for the PC)
pub(super) unsafe extern "system" fn ia_set_vertex_buffers_x(this: *mut c_void, start: u32, n: u32, views: *const D3D12_VERTEX_BUFFER_VIEW) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::BindState);
    let f = pc::<unsafe extern "system" fn(*mut c_void, u32, u32, *const D3D12_VERTEX_BUFFER_VIEW)>(this, PC_IA_SET_VERTEX_BUFFERS);
    if views.is_null() || n == 0 {
        return f(this, start, n, views);
    }
    let v: Vec<D3D12_VERTEX_BUFFER_VIEW> = (0..n as usize)
        .map(|k| {
            let mut v = views.add(k).read_unaligned();
            let title = v.BufferLocation;
            v.SizeInBytes = view_bytes(title, v.SizeInBytes);
            v.BufferLocation = gpu_va(this, v.BufferLocation, v.SizeInBytes as usize, "vertex buffer", true);
            if super::present::trace_frame() {
                static N: AtomicU64 = AtomicU64::new(0);
                let i = N.fetch_add(1, Ordering::Relaxed);
                if i < 12 || (i.is_power_of_two() && i < 1 << 16) {
                    let n = (v.SizeInBytes as usize).min(256);
                    let data = (title != 0 && crate::gpu::shader::readable(title as usize, n)).then(|| std::slice::from_raw_parts(title as *const u8, n).iter().filter(|&&b| b != 0).count());
                    let win = crate::gpu::uma::get().and_then(|u| u.window_of(title as usize).map(|w| (w, u.window(w), u.is_sparse(title as usize))));
                    let mut mbi: windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                    let q = crate::hle::kernelx::VirtualQuery(title as *const c_void, &mut mbi, std::mem::size_of_val(&mbi));
                    crate::xlog_info!(
                        "frame trace: vertex buffer #{i} slot {}: title {title:#x} → PC {:#x}, {} bytes, stride {}; non-zero in the first {n} bytes (CPU view): {data:?}; window (index, (base, size), sparse) {win:x?}; memory state {:#x} protect {:#x} (query {q})",
                        start + k as u32,
                        v.BufferLocation,
                        v.SizeInBytes,
                        v.StrideInBytes,
                        mbi.State,
                        mbi.Protect
                    );
                }
            }
            v
        })
        .collect();
    let kept = v.clone();
    let titles: Vec<(u64, u32, u32)> = (0..n as usize).map(|k| { let t = views.add(k).read_unaligned(); (t.BufferLocation, t.SizeInBytes, t.StrideInBytes) }).collect();
    with_bindings(this, |b| {
        for (k, view) in kept.into_iter().enumerate() {
            b.vertex_buffers.insert(start + k as u32, view);
            b.vertex_titles.insert(start + k as u32, titles[k]);
        }
    });
    f(this, start, n, v.as_ptr())
}

/// command list #47 IASetIndexBuffer(view) (PC #43): the address is the title's
pub(super) unsafe extern "system" fn ia_set_index_buffer_x(this: *mut c_void, view: *const D3D12_INDEX_BUFFER_VIEW) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::BindState);
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
    if view.is_null() {
        s.index_buffer = None;
        drop(g);
        return set_index_buffer(this, None);
    }
    let mut v = view.read_unaligned();
    s.bindings.index_title = Some((v.BufferLocation, v.SizeInBytes, v.Format.0));
    drop(g);
    v.SizeInBytes = view_bytes(v.BufferLocation, v.SizeInBytes);
    v.BufferLocation = gpu_va(this, v.BufferLocation, v.SizeInBytes as usize, "index buffer", true);
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
        s.index_buffer = Some(v);
    }
    set_index_buffer(this, Some(&v));
}

/// command list #81 DrawIndexedX(index address, index format, index count, flags): one
/// instance of `count` indices read from that address; the bound index buffer stays
pub(super) unsafe extern "system" fn draw_indexed_x(this: *mut c_void, va: u64, format: DXGI_FORMAT, count: u32, _flags: u32) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::DrawRecord);
    let _c = super::bind::capture_work(this, || super::capture::WorkKind::DrawIndexedX { va, format: format.0, indices: count }, true);
    static N: AtomicU64 = AtomicU64::new(0);
    let stride = match format {
        DXGI_FORMAT_R16_UINT => 2,
        DXGI_FORMAT_R32_UINT => 4,
        f => return warn(&N, "DrawIndexedX", format_args!("index format {}", f.0)),
    };
    if count == 0 {
        return;
    }
    let Some(list) = super::bind::prepare_indexed_draw(this, count, 1, 0, 0) else {
        return;
    };
    let bound = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).and_then(|s| s.index_buffer));
    sync_title_buffers(this, None, 0, 0, 2);
    // (indices past the clamped size read as 0)
    let bytes = view_bytes(va, count.saturating_mul(stride));
    let view = D3D12_INDEX_BUFFER_VIEW { BufferLocation: gpu_va(this, va, bytes as usize, "DrawIndexedX indices", false), SizeInBytes: bytes, Format: format };
    set_index_buffer(this, Some(&view));
    super::bind::emit_prepared_indexed_draw(&list, count, 1, 0, 0, 0);
    set_index_buffer(this, bound.as_ref());
}

// ---------------------------------------------------------------------------------------
// nested occlusion queries & Reset
// ---------------------------------------------------------------------------------------

/// outer queries ended early, and the result they resolve to
static VISIBLE: Mutex<Option<HashMap<Query, u64>>> = Mutex::new(None);
/// what such an OCCLUSION query reports (BINARY_OCCLUSION: 1)
const VISIBLE_SAMPLES: u64 = 0x1_0000;

fn is_occlusion(ty: D3D12_QUERY_TYPE) -> bool {
    ty == D3D12_QUERY_TYPE_OCCLUSION || ty == D3D12_QUERY_TYPE_BINARY_OCCLUSION
}

type QueryFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32);

/// command list #10 Reset(allocator, initial pipeline) as console code calls it on a PC list.
/// When `this` is the PC list a title's list (`title_list`) records on, the call came through
/// that list's thunk to its PC list – the newer console layout's Reset (#139, allocator, NULL,
/// 0) does, as it is not the title list's own #10 – and it is that list's Reset (`title_list::
/// reset`: the primary PC list again, or a stand-in while the primary's last submit runs). Else
/// `reset_pc`.
///
/// # Safety
/// `this` must be a valid adopted `ID3D12GraphicsCommandList` pointer.
pub unsafe extern "system" fn reset_x(this: *mut c_void, alloc: usize, pso: usize) -> i32 {
    if let Some(w) = super::title_list::wrapper_of(this) {
        return super::title_list::reset(w, alloc, pso);
    }
    let hr = reset_pc(this, alloc, pso);
    if hr >= 0 {
        super::capture::list_begin(this as usize, pso);
    }
    hr
}

/// Reset of the PC list `this` (allocator, initial pipeline): updates allocator/pipeline and
/// clears per-recording state (queries, index buffer, depth bias, pending waits, written ranges)
///
/// # Safety
/// `this` must be a valid adopted `ID3D12GraphicsCommandList` pointer.
pub(super) unsafe fn reset_pc(this: *mut c_void, alloc: usize, pso: usize) -> i32 {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListLifecycle);
    reset_pc_inner(this, alloc, pso)
}

unsafe fn reset_pc_inner(this: *mut c_void, alloc: usize, pso: usize) -> i32 {
    // A failed native Reset leaves the old recording alive. Do not discard its tracked state or
    // transfer its allocations until the driver accepts the new recording.
    let ext = *(this as *const usize);
    let prepared = super::bind::PreparedNativePipeline::new(pso);
    let resolved = prepared.raw();
    // (the PC allocator the title's allocator stands for now: `alloc_pool`)
    let ty = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize))).and_then(|s| s.list_type).map_or(-1, |t| t.0);
    let mut on = alloc;
    let hr = super::alloc_pool::reset_list(this as usize, alloc, ty, |pc_alloc| {
        on = pc_alloc;
        pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(this, PC_RESET)(this, pc_alloc, resolved)
    });
    real::preserve_vtable(this, ext);
    if hr < 0 {
        return hr;
    }
    trace_list(this as usize, ListEvent::Reset);

    reset_copy_scratch(this);
    reset_own_alloc(this);
    let (is_bundle, old_generations) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
        s.allocator = alloc;
        s.bound_pso = pso;
        s.poisoned = false;
        s.depth_bias = None;
        s.native_bias = DepthBias::ZERO;
        s.pso_dirty = false;
        s.using_fallback_pso = false;
        s.index_buffer = None;
        s.depth_bounds = None;
        s.occlusion = Occlusion::default();
        s.written_ranges.clear();
        s.segment_write_producers.clear();
        s.clear_sparse_writes();
        s.clear_write64_signals();
        s.clear_waits();
        s.texture_writes.clear();
        s.texture_transitions.clear();
        s.uses_shared_view_heap = false;
        s.cpu_writes.clear();
        let old_generations = s.take_native_generations();
        s.predication = None;
        s.pred_in_state = false;
        s.bindings = Bindings::default();
        s.depth_in_step.clear();
        (s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE), old_generations)
    };
    drop(old_generations);
    // (what the new recording puts in unified memory belongs to the PC allocator it is on)
    prepared.commit(this, on);
    super::lut3d::list_reset(this as usize);
    super::dred::reset(this);
    if pso != 0 && resolved != 0 {
        super::dred::mark_pipeline(this, pso, resolved);
    }
    if !is_bundle && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(this, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
    hr
}

/// the D3D12_RESOURCE_STATES bits the PC knows
const PC_STATES: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0xF_0000 | 0x20_0000 | 0x40_0000 | 0x80_0000 | 0x100_0000;

/// #26 ResourceBarrier(count, barriers): console barriers (D3D12_RESOURCE_BARRIER, 32 bytes) as
/// PC ones – state bits the PC does not know are dropped (logged: the console's own), a
/// transition left with nothing to do goes, aliasing barriers name placed textures
pub(super) unsafe extern "system" fn resource_barrier_x(this: *mut c_void, n: u32, barriers: *const [u32; 8]) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailBarrier);
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListBarrier);
    static N: AtomicU64 = AtomicU64::new(0);
    if barriers.is_null() || n == 0 {
        return;
    }
    let capture = super::capture::op(this, || Some(super::capture::barrier(&(0..n as usize).map(|k| barriers.add(k).read_unaligned()).collect::<Vec<_>>())));
    let mut out: Vec<[u32; 8]> = Vec::with_capacity(n as usize + 1);
    // a UAV barrier for all resources is in `out` / a transition from or to UNORDERED_ACCESS
    let (mut any_uav, mut uav_transition) = (false, false);
    for k in 0..n as usize {
        let mut b = barriers.add(k).read_unaligned();
        let (ty, flags) = (b[0], b[1]);
        if super::present::trace_frame() && ty == 0 {
            let r = (b[2] as u64 | (b[3] as u64) << 32) as usize;
            if super::present::is_watched(r) {
                static W: AtomicU64 = AtomicU64::new(0);
                if W.fetch_add(1, Ordering::Relaxed) < 24 {
                    crate::xlog_info!("frame trace: ResourceBarrier on a watched target {r:#x}: subresource {:#x}, {:#x} → {:#x}, flags {flags:#x}", b[4], b[5], b[6]);
                }
            }
        }
        if ty > 2 {
            if N.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!("D3D12.X: ResourceBarrier: console barrier type {ty} flags {flags:#x} dropped ({:x?})", b);
            }
            continue;
        }
        // Console cache hints must not discard the transition or memory ordering itself.
        // Keep the PC split-barrier bits and let the PC perform its normal synchronization.
        const PC_FLAGS: u32 = (D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY.0 | D3D12_RESOURCE_BARRIER_FLAG_END_ONLY.0) as u32;
        if flags & !PC_FLAGS != 0 {
            if N.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!("D3D12.X: ResourceBarrier: console flag bits {:#x} dropped", flags & !PC_FLAGS);
            }
            b[1] &= PC_FLAGS;
        }
        if ty == 1 {
            // Keep the identity of textures that really share placed GPU memory.
            // Committed resources still need the wildcard form on the PC.
            for words in b[2..6].chunks_exact_mut(2) {
                let resource = (words[0] as u64 | (words[1] as u64) << 32) as usize;
                if resource != 0 && !super::vram::is_placed_texture(resource) {
                    words.fill(0);
                }
            }
        }
        if ty == 2 {
            // UAV (resource +8): translated shaders reach title memory through XCL's memory
            // window buffers – other resources over the same memory – which a UAV barrier for
            // the title's resource alone does not order (AMD drivers wait and flush for any UAV
            // barrier, NVIDIA's need not): a UAV barrier for all resources, once per batch
            b[2..4].fill(0);
            if any_uav {
                continue;
            }
            any_uav = true;
        }
        if ty == 0 {
            // transition: resource (+8), subresource (+16), before (+20), after (+24)
            let (before, after) = (b[5], b[6]);
            if (before | after) & !PC_STATES != 0 && N.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!("D3D12.X: ResourceBarrier: console state bits {:#x} → {:#x} dropped", before & !PC_STATES, after & !PC_STATES);
            }
            b[5] &= PC_STATES;
            b[6] &= PC_STATES;
            // VERTEX_AND_CONSTANT_BUFFER / INDEX_BUFFER are buffer states on the PC; the console
            // takes them for textures too (Minecraft moves a texture to a generic-read
            // combination) – the PC list then fails to close
            if (b[5] | b[6]) & BUFFER_ONLY_STATES != 0 && !is_buffer((b[2] as u64 | (b[3] as u64) << 32) as usize) {
                b[5] &= !BUFFER_ONLY_STATES;
                b[6] &= !BUFFER_ONLY_STATES;
            }
            if b[5] == b[6] {
                continue;
            }
            // UAV writes before (or after) the transition: the same holds for the window buffers
            uav_transition |= (b[5] | b[6]) & UNORDERED_ACCESS != 0;
        }
        out.push(b);
    }
    if uav_transition && !any_uav {
        out.push([2, 0, 0, 0, 0, 0, 0, 0]);
    }
    // depth buffers whose depth plane can have a twin: in PARK where the title has them in
    // UNORDERED_ACCESS, the twin copied around the transitions (`depth_uav`)
    let plane_copies = super::depth_uav::any_kept().then(|| {
        let direct = list_type(this) == D3D12_COMMAND_LIST_TYPE_DIRECT;
        let mut in_step = std::mem::take(&mut LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(this as usize).or_default().depth_in_step);
        let plan = super::depth_uav::plan_barriers(&mut out, direct, &mut in_step);
        if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
            s.depth_in_step = in_step;
        }
        plan
    });
    // the states uploads into textures kept in step start from once this list has run
    if super::title_textures::any() || super::lut3d::on() {
        let t: Vec<(usize, u32, D3D12_RESOURCE_STATES)> =
            out.iter().filter(|b| b[0] == 0).map(|b| ((b[2] as u64 | (b[3] as u64) << 32) as usize, b[4], D3D12_RESOURCE_STATES(b[6] as i32))).collect();
        note_texture_transitions(this as usize, &t);
        super::lut3d::note_transitions(this as usize, &t);
    }
    // (a superseded target the title moves is used again: `placements`)
    for b in out.iter().filter(|b| b[0] == 0) {
        super::placements::note_use((b[2] as u64 | (b[3] as u64) << 32) as usize);
    }
    capture.amend(|op| {
        if let super::capture::Op::Barrier { sent, plane_copies: copies, .. } = op {
            sent.clone_from(&out);
            *copies = plane_copies.as_ref().map_or(0, |p| p.to_plane.len() + p.to_twin.len());
        }
    });
    if let Some(p) = &plane_copies {
        plane_copies_on(this, &p.to_plane, false);
    }
    if !out.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const [u32; 8])>(this, PC_RESOURCE_BARRIER)(this, out.len() as u32, out.as_ptr());
    }
    if let Some(p) = &plane_copies {
        plane_copies_on(this, &p.to_twin, true);
    }
}

/// The copies between depth planes and their twins around a title's barriers
/// (`depth_uav::plan_barriers`) on `list`: into the twins (`to_twin`, after the barriers: the
/// buffer rests in `depth_uav::PARK`, COPY_SOURCE) or into the planes (before them: the buffer
/// goes PARK → COPY_DEST, where the title's transition out of UNORDERED_ACCESS takes it on). The
/// twin rests in UNORDERED_ACCESS. Whole subresources, of one format family on both sides (a
/// D3D12 copy of a depth plane takes the whole of it).
unsafe fn plane_copies_on(list: *mut c_void, copies: &[super::depth_uav::PlaneCopy], to_twin: bool) {
    let barriers = |b: &[[u32; 8]]| pc::<unsafe extern "system" fn(*mut c_void, u32, *const [u32; 8])>(list, PC_RESOURCE_BARRIER)(list, b.len() as u32, b.as_ptr());
    let tr = |res: usize, sub: u32, before: u32, after: u32| [0, 0, res as u32, (res as u64 >> 32) as u32, sub, before, after, 0];
    let all = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    let at = |res: usize, k: u32| D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k } };
    let copy = pc::<unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX)>(list, PC_COPY_TEXTURE_REGION);
    let (copy_src, copy_dst) = (D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
    for c in copies {
        let twin = c.twin.as_raw() as usize;
        let twin_in = if to_twin { copy_dst } else { copy_src };
        if to_twin {
            barriers(&[tr(twin, all, UNORDERED_ACCESS, twin_in)]);
        } else {
            barriers(&[tr(twin, all, UNORDERED_ACCESS, twin_in), tr(c.depth, c.sub, super::depth_uav::PARK, copy_dst)]);
        }
        for &k in &c.subs {
            let (dst, src) = if to_twin { (at(twin, k), at(c.depth, k)) } else { (at(c.depth, k), at(twin, k)) };
            copy(list, &dst, 0, 0, 0, &src, std::ptr::null());
        }
        barriers(&[tr(twin, all, twin_in, UNORDERED_ACCESS)]);
    }
}

/// D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | INDEX_BUFFER: valid for buffers only
const BUFFER_ONLY_STATES: u32 = 0x1 | 0x2;

/// D3D12_RESOURCE_STATE_UNORDERED_ACCESS
const UNORDERED_ACCESS: u32 = 0x8;

/// Whether `res` is a buffer (null / unknown: taken as one – its states are left alone).
unsafe fn is_buffer(res: usize) -> bool {
    let r = res as *mut c_void;
    ID3D12Resource::from_raw_borrowed(&r).is_none_or(|x| x.GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
}

/// DMA list #9 Close: as on the PC; a failure is logged (the title stops on it)
pub(super) unsafe extern "system" fn dma_close_x(this: *mut c_void) -> i32 {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailListLifecycle);
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListLifecycle);
    let ext = *(this as *const usize);
    let hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, 9)(this);
    real::preserve_vtable(this, ext);
    if hr >= 0
        && !super::title_textures::is_own_work()
        && !super::present::presenting()
        && super::title_list::wrapper_of(this).is_none()
        && !super::console_list::is_backing(this)
    {
        super::title_textures::note_list_closed(this as usize);
    }
    if hr < 0 {
        let ty = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, PC_GET_TYPE)(this);
        crate::xlog_warn!("D3D12.X: DMA list Close → {:#x} (PC list type {ty})", hr as u32);
    }
    hr
}

/// a copy location with the offset of a buffer placed below its console address added
unsafe fn pc_location(l: *const D3D12_TEXTURE_COPY_LOCATION) -> Option<D3D12_TEXTURE_COPY_LOCATION> {
    let mut l = std::ptr::read(l.as_ref()?);
    if l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
        l.Anonymous.PlacedFootprint.Offset += real::buffer_delta(transmute_copy::<_, usize>(&l.pResource));
    }
    Some(l)
}

/// CopyBufferRegion(dst, dst offset, src, src offset, bytes) (PC #15; DMA list #11)
/// `XCL_TRACE_FRAME`: a GPU copy into title memory that a texture kept in step lies in (the
/// texture does not see GPU writes)
fn note_gpu_write(what: &str, va: usize, n: u64) {
    if !super::present::trace_frame() {
        return;
    }
    let hit = super::title_textures::overlapping(va, n as usize);
    if hit.is_empty() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if k < 16 || k.is_power_of_two() {
        crate::xlog_info!("frame trace: {what} #{k} writes {n:#x} bytes at {va:#x} – into title textures {hit:x?}");
    }
}

pub(super) unsafe extern "system" fn copy_buffer_region_x(this: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailCopy);
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    let _c = super::capture::op(this, || {
        use super::capture as c;
        Some(c::Op::Copy { kind: "buffer_region", dst: c::buffer(dst, dst_off), src: c::buffer(src, src_off), at: [0; 3], bx: None, bytes: Some(n), refused: false })
    });
    copy_buffer_region_inner(this, dst, dst_off, src, src_off, n)
}

unsafe fn copy_buffer_region_inner(this: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    if super::present::trace_frame() {
        if let Some(a) = real::title_address(dst) {
            note_gpu_write("CopyBufferRegion", a + dst_off as usize, n);
        }
    }
    note_copy_into(this as usize, dst, dst_off, n);
    pc::<unsafe extern "system" fn(*mut c_void, usize, u64, usize, u64, u64)>(this, PC_COPY_BUFFER_REGION)(this, dst, dst_off + real::buffer_delta(dst), src, src_off + real::buffer_delta(src), n)
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): copies out of it – that
/// object, or another at its title address (the first 16)
unsafe fn trace_watched_copy(what: &str, dst: usize, src: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let addr = |r: usize| real::title_address(r).or_else(|| super::title_textures::address(r));
    if !super::present::is_watched(src) && (addr(src).is_none() || addr(src) != addr(w)) {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 16 {
        return;
    }
    if !super::present::is_presented(dst) {
        super::present::follow(dst);
    }
    let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Flags.0) });
    crate::xlog_info!(
        "frame trace: {what} out of a watched target {src:#x} {:?} into {dst:#x} {:?}{}",
        d(src),
        d(dst),
        if super::present::is_presented(dst) { " (presented)" } else { "" }
    );
}

/// CopyTextureRegion(dst, x, y, z, src, box) (PC #16; DMA list #12)
pub(super) unsafe extern "system" fn copy_texture_region_x(this: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: *const D3D12_TEXTURE_COPY_LOCATION, bx: *const D3D12_BOX) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailCopy);
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    let _c = super::capture::op(this, || {
        use super::capture as c;
        let b = bx.as_ref().map(|b| [b.left, b.top, b.front, b.right, b.bottom, b.back]);
        Some(c::Op::Copy { kind: "texture_region", dst: c::location(dst), src: c::location(src), at: [x, y, z], bx: b, bytes: None, refused: dst.is_null() || src.is_null() })
    });
    if let Some(d) = dst.as_ref() {
        let r: usize = transmute_copy(&d.pResource);
        super::present::note_use(r, super::present::USE_COPY);
        let sub = if d.Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX {
            d.Anonymous.SubresourceIndex
        } else {
            D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES
        };
        note_texture_write(this as usize, r, sub);
        super::placements::note_use(r);
        if let Some(s) = src.as_ref() {
            super::placements::note_use(transmute_copy(&s.pResource));
        }
        super::lut3d::wrote(r, true, || {
            let from = src.as_ref().map_or("nothing".to_string(), |sl| {
                let sr: usize = transmute_copy(&sl.pResource);
                if sl.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
                    let f = sl.Anonymous.PlacedFootprint;
                    format!("buffer {sr:#x} (title address {:?}) at {:#x}, footprint format {} {}x{}x{} row pitch {}", real::title_address(sr).map(|a| format!("{a:#x}")), f.Offset, f.Footprint.Format.0, f.Footprint.Width, f.Footprint.Height, f.Footprint.Depth, f.Footprint.RowPitch)
                } else {
                    format!("subresource {} of {sr:#x}", sl.Anonymous.SubresourceIndex)
                }
            });
            let b = bx.as_ref().map(|b| (b.left, b.top, b.front, b.right, b.bottom, b.back));
            format!("CopyTextureRegion into subresource {} at ({x}, {y}, {z}) from {from}, box {b:?}", d.Anonymous.SubresourceIndex)
        });
        if super::present::trace_frame() {
            if let Some(va) = super::title_textures::address(r) {
                static N: AtomicU64 = AtomicU64::new(0);
                let k = N.fetch_add(1, Ordering::Relaxed);
                if k < 8 || k.is_power_of_two() {
                    // the source: a buffer footprint – where its bytes are for the CPU, and whether
                    // there is anything there
                    let src_info = src.as_ref().map(|sl| {
                        let sr: usize = transmute_copy(&sl.pResource);
                        let rr = sr as *mut c_void;
                        let heap = ID3D12Resource::from_raw_borrowed(&rr).and_then(|x| {
                            let mut hp = D3D12_HEAP_PROPERTIES::default();
                            let mut hf = D3D12_HEAP_FLAGS::default();
                            x.GetHeapProperties(Some(&mut hp), Some(&mut hf)).ok().map(|_| hp.Type.0)
                        });
                        let (ty, fp) = (sl.Type.0, if sl.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT { Some(sl.Anonymous.PlacedFootprint) } else { None });
                        let cpu = real::title_address(sr);
                        let nonzero = match (cpu, fp) {
                            (Some(a), Some(f)) => {
                                let at = a + f.Offset as usize;
                                let n = (f.Footprint.RowPitch as usize * 4).min(4096);
                                if crate::gpu::shader::readable(at, n) { Some(std::slice::from_raw_parts(at as *const u8, n).iter().filter(|&&b| b != 0).count()) } else { None }
                            }
                            _ => None,
                        };
                        format!("src {sr:#x} type {ty} heap {heap:?} title address {:?} delta {:#x} footprint {:?} nonzero in the first rows {nonzero:?}", cpu.map(|a| format!("{a:#x}")), real::buffer_delta(sr), fp.map(|f| (f.Offset, f.Footprint.Format.0, f.Footprint.Width, f.Footprint.Height, f.Footprint.RowPitch)))
                    });
                    crate::xlog_info!("frame trace: CopyTextureRegion #{k} into title texture {r:#x} (at {va:#x}), subresource {}; {src_info:?}", d.Anonymous.SubresourceIndex);
                }
            }
        }
    }
    if super::present::trace_frame() {
        if let (Some(d), Some(sl)) = (dst.as_ref(), src.as_ref()) {
            trace_watched_copy("CopyTextureRegion", transmute_copy(&d.pResource), transmute_copy(&sl.pResource));
            // footprints the PC's copy rules do not allow (row pitch: 256 bytes, offset: 512)
            if sl.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
                let f = sl.Anonymous.PlacedFootprint;
                if f.Footprint.RowPitch % 256 != 0 || f.Offset % 512 != 0 {
                    // each (format, pitch alignment, offset alignment) once
                    static SEEN: Mutex<Vec<(i32, u32, u64)>> = Mutex::new(Vec::new());
                    let key = (f.Footprint.Format.0, f.Footprint.RowPitch % 256, f.Offset % 512);
                    let new = {
                        let mut g = SEEN.lock().unwrap();
                        let n = !g.contains(&key) && g.len() < 24;
                        if n {
                            g.push(key);
                        }
                        n
                    };
                    if new {
                        crate::xlog_info!(
                            "frame trace: CopyTextureRegion from a footprint the PC does not take: offset {:#x}, format {}, {}x{}x{}, row pitch {} (box {:?}, to ({x}, {y}, {z}))",
                            f.Offset,
                            f.Footprint.Format.0,
                            f.Footprint.Width,
                            f.Footprint.Height,
                            f.Footprint.Depth,
                            f.Footprint.RowPitch,
                            bx.as_ref().map(|b| (b.left, b.top, b.right, b.bottom))
                        );
                    }
                }
            }
        }
    }
    if let Some(l) = dst.as_ref().filter(|l| l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT) {
        // (the rows of the footprint: at most what the copy writes)
        let f = &l.Anonymous.PlacedFootprint;
        let n = f.Footprint.RowPitch as u64 * f.Footprint.Height.max(1) as u64 * f.Footprint.Depth.max(1) as u64;
        note_copy_into(this as usize, transmute_copy::<_, usize>(&l.pResource), f.Offset, n);
    }
    let (Some(d), Some(s)) = (pc_location(dst), pc_location(src)) else { return };
    pc::<unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX)>(this, PC_COPY_TEXTURE_REGION)(this, &d, x, y, z, &s, bx);
    std::mem::forget((d, s)); // borrowed resource pointers
}

/// command list #83 CopyTextureRegionX(dst, x, y, z, src, box, flags): CopyTextureRegion with
/// console copy flags (the title's texture streaming calls it where the DMA list gets #12)
pub(super) unsafe extern "system" fn copy_texture_region_flags_x(this: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: *const D3D12_TEXTURE_COPY_LOCATION, bx: *const D3D12_BOX, _flags: u32) {
    copy_texture_region_x(this, dst, x, y, z, src, bx)
}

const PC_CLEAR_DEPTH_STENCIL_VIEW: usize = 47;

/// command list #51 ClearDepthStencilView(dsv, flags, depth, stencil, rects, rect list) (PC #47)
pub(super) unsafe extern "system" fn clear_depth_stencil_x(this: *mut c_void, dsv: usize, flags: u32, depth: f32, stencil: u8, n: u32, rects: *const windows::Win32::Foundation::RECT) {
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    let _c = super::capture::op(this, || Some(super::capture::clear("dsv", dsv as u64, None, [depth.to_bits(), stencil as u32, 0, 0], flags, n)));
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        static W: AtomicU64 = AtomicU64::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        let watched = WATCHED_DSVS.lock().unwrap().contains(&dsv);
        if k < 8 || k.is_power_of_two() || (watched && W.fetch_add(1, Ordering::Relaxed) < 12) {
            crate::xlog_info!(
                "frame trace: ClearDepthStencilView #{k} {dsv:#x}{}: flags {flags:#x} (1 depth, 2 stencil), depth {depth}, stencil {stencil}, {n} rects",
                if watched { " (bound by a draw reading a watched target)" } else { "" }
            );
        }
    }
    pc::<unsafe extern "system" fn(*mut c_void, usize, u32, f32, u8, u32, *const windows::Win32::Foundation::RECT)>(this, PC_CLEAR_DEPTH_STENCIL_VIEW)(this, dsv, flags, depth, stencil, n, rects)
}

const PC_CLEAR_UAV_UINT: usize = 49;
const PC_CLEAR_UAV_FLOAT: usize = 50;

/// The handles a ClearUnorderedAccessView* of console code goes to the PC with.
#[derive(Debug, PartialEq)]
enum UavClearHandles {
    /// both are console slots: their PC UAVs (GPU: in the heap lists bind; CPU: in the
    /// title's CPU-only heap)
    Console { gpu: u64, cpu: u64 },
    /// neither is a console slot: PC handles already
    Pc,
    /// one console slot and one not, or a slot with no shader-visible UAV
    Unusable,
}

fn uav_clear_handles(gpu: Option<super::console_heap::Slot>, cpu: Option<super::console_heap::Slot>) -> UavClearHandles {
    match (gpu, cpu) {
        (None, None) => UavClearHandles::Pc,
        (Some(g), Some(c)) if !g.sampler && !c.sampler && g.pc_gpu_uav != 0 => UavClearHandles::Console { gpu: g.pc_gpu_uav, cpu: c.pc_uav },
        _ => UavClearHandles::Unusable,
    }
}

/// The resource a ClearUnorderedAccessView* of `res` through the GPU slot `gpu` goes to the PC
/// with: `res`, or the twin of a depth buffer the slot's view names (`depth_uav`). None: not
/// done (what the view names is not known). Also for the clears of console-recorded lists.
pub(super) fn uav_clear_resource(res: usize, gpu: Option<super::console_heap::Slot>) -> Option<usize> {
    if !super::depth_uav::any() {
        return Some(res);
    }
    let named = gpu.filter(|s| s.shared_srv != 0).map(|s| super::console_heap::names_of(&s).1);
    match super::depth_uav::clear_target(res, named) {
        super::depth_uav::Clear::AsIs => Some(res),
        super::depth_uav::Clear::Twin(t) => Some(t),
        super::depth_uav::Clear::NotDone => None,
    }
}

/// command list #53 / #54 ClearUnorderedAccessViewUint / Float(GPU handle, CPU handle,
/// resource, values[4], rects, rect list) (PC #49 / #50): console handles are addresses of
/// console heap slots (heap #9 / #10), not PC descriptors. A depth buffer whose UAV views are
/// views of its twin (`depth_uav`) is cleared through the twin the GPU slot's view names.
unsafe fn clear_uav(this: *mut c_void, slot: usize, gpu: u64, cpu: u64, res: usize, values: *const c_void, n: u32, rects: *const windows::Win32::Foundation::RECT) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailCopy);
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    let capture = super::capture::op(this, || {
        let v = values.cast::<[u32; 4]>().as_ref().copied().unwrap_or_default();
        Some(super::capture::clear(if slot == PC_CLEAR_UAV_UINT { "uav_uint" } else { "uav_float" }, gpu, Some(res), v, 0, n))
    });
    let refused = || capture.amend(|op| if let super::capture::Op::Clear { refused, .. } = op { *refused = true });
    let gpu_slot = super::console_heap::slot(gpu);
    let (gpu, cpu) = match uav_clear_handles(gpu_slot, super::console_heap::slot(cpu)) {
        UavClearHandles::Console { gpu, cpu } => (gpu, cpu),
        UavClearHandles::Pc => (gpu, cpu),
        UavClearHandles::Unusable => {
            static N: AtomicU64 = AtomicU64::new(0);
            if once(&N, 1) {
                crate::xlog_warn!("D3D12.X: ClearUnorderedAccessView with GPU handle {gpu:#x} / CPU handle {cpu:#x}: not both a console slot with a shader-visible UAV – not done");
            }
            return refused();
        }
    };
    let Some(res) = uav_clear_resource(res, gpu_slot) else { return refused() };
    pc::<unsafe extern "system" fn(*mut c_void, u64, u64, usize, *const c_void, u32, *const windows::Win32::Foundation::RECT)>(this, slot)(this, gpu, cpu, res, values, n, rects)
}

pub(super) unsafe extern "system" fn clear_uav_uint_x(this: *mut c_void, gpu: u64, cpu: u64, res: usize, values: *const u32, n: u32, rects: *const windows::Win32::Foundation::RECT) {
    clear_uav(this, PC_CLEAR_UAV_UINT, gpu, cpu, res, values as *const c_void, n, rects)
}

pub(super) unsafe extern "system" fn clear_uav_float_x(this: *mut c_void, gpu: u64, cpu: u64, res: usize, values: *const f32, n: u32, rects: *const windows::Win32::Foundation::RECT) {
    clear_uav(this, PC_CLEAR_UAV_FLOAT, gpu, cpu, res, values as *const c_void, n, rects)
}

const PC_RESOLVE_SUBRESOURCE: usize = 19;
const PC_RESOLVE_SUBRESOURCE_REGION: usize = 64;

/// `XCL_TRACE_FRAME`: a resolve (first 12 and powers of two) – what goes where
unsafe fn trace_resolve(what: &str, dst: usize, src: usize, format: i32, extra: &str) {
    if !super::present::trace_frame() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if k >= 12 && !k.is_power_of_two() {
        return;
    }
    let desc = |r: usize| {
        let p = r as *mut c_void;
        ID3D12Resource::from_raw_borrowed(&p).map(|x| {
            let d = x.GetDesc();
            (d.Width, d.Height, d.Format.0, d.SampleDesc.Count)
        })
    };
    crate::xlog_info!(
        "frame trace: {what} #{k} {src:#x} {:?} → {dst:#x} {:?}{} format {format}{extra}",
        desc(src),
        desc(dst),
        if super::present::is_presented(dst) { " (presented)" } else { "" }
    );
}

/// command list #19 ResolveSubresource(dst, dst sub, src, src sub, format) (PC #19)
pub(super) unsafe extern "system" fn resolve_subresource_x(this: *mut c_void, dst: usize, dst_sub: u32, src: usize, src_sub: u32, format: i32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailCopy);
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    let _c = super::capture::op(this, || {
        use super::capture as c;
        Some(c::Op::Resolve { dst: c::describe(dst), dst_sub, src: c::describe(src), src_sub, format, region: None })
    });
    super::present::note_use(dst, super::present::USE_COPY);
    note_texture_write(this as usize, dst, dst_sub);
    trace_resolve("ResolveSubresource", dst, src, format, "");
    pc::<unsafe extern "system" fn(*mut c_void, usize, u32, usize, u32, i32)>(this, PC_RESOLVE_SUBRESOURCE)(this, dst, dst_sub, src, src_sub, format)
}

/// command list #108 ResolveSubresourceRegion(dst, dst sub, x, y, src, src sub, rect, format,
/// mode) (PC #64)
#[allow(clippy::too_many_arguments)]
pub(super) unsafe extern "system" fn resolve_subresource_region_x(this: *mut c_void, dst: usize, dst_sub: u32, x: u32, y: u32, src: usize, src_sub: u32, rect: *const windows::Win32::Foundation::RECT, format: i32, mode: i32) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailCopy);
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    let _c = super::capture::op(this, || {
        use super::capture as c;
        let r = rect.as_ref().map(|r| [r.left, r.top, r.right, r.bottom]);
        Some(c::Op::Resolve { dst: c::describe(dst), dst_sub, src: c::describe(src), src_sub, format, region: Some((x, y, r, mode)) })
    });
    super::present::note_use(dst, super::present::USE_COPY);
    note_texture_write(this as usize, dst, dst_sub);
    if super::present::trace_frame() {
        trace_resolve("ResolveSubresourceRegion", dst, src, format, &format!(" mode {mode} at ({x}, {y}) rect {:?}", rect.as_ref().map(|r| (r.left, r.top, r.right, r.bottom))));
    }
    pc::<unsafe extern "system" fn(*mut c_void, usize, u32, u32, u32, usize, u32, *const windows::Win32::Foundation::RECT, i32, i32)>(this, PC_RESOLVE_SUBRESOURCE_REGION)(this, dst, dst_sub, x, y, src, src_sub, rect, format, mode)
}

/// CopyResource(dst, src) (PC #17; DMA list #13): buffers placed below their console address
/// are copied as their own bytes (the PC resources also hold what lies before them)
pub(super) unsafe extern "system" fn copy_resource_x(this: *mut c_void, dst: usize, src: usize) {
    let _detail = crate::stall::detail_span(crate::stall::StallKind::DetailCopy);
    let _t = crate::hle::api_cost::time_if(super::look_stats::deep(), crate::hle::api_cost::Call::ListCopy);
    let _c = super::capture::op(this, || {
        use super::capture as c;
        Some(c::Op::Copy { kind: "resource", dst: c::buffer(dst, 0), src: c::buffer(src, 0), at: [0; 3], bx: None, bytes: None, refused: false })
    });
    super::present::note_use(dst, super::present::USE_COPY);
    note_texture_write(this as usize, dst, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
    super::placements::note_use(dst);
    super::placements::note_use(src);
    super::lut3d::wrote(dst, true, || format!("CopyResource from {src:#x}"));
    if super::present::trace_frame() {
        trace_watched_copy("CopyResource", dst, src);
    }
    let (dd, sd) = (real::buffer_delta(dst), real::buffer_delta(src));
    if dd == 0 && sd == 0 {
        note_copy_into(this as usize, dst, 0, u64::MAX);
        return pc::<unsafe extern "system" fn(*mut c_void, usize, usize)>(this, PC_COPY_RESOURCE)(this, dst, src);
    }
    let n = real::console_buffer(dst).or(real::console_buffer(src)).map_or(0, |b| b.1 as u64);
    copy_buffer_region_inner(this, dst, 0, src, 0, n);
}

/// command list #28 BeginQuery(heap, type, index) (PC #52)
pub(super) unsafe extern "system" fn begin_query_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    static N: AtomicU64 = AtomicU64::new(0);
    if is_occlusion(ty) {
        let outer = LISTS.lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(this as usize)
            .or_default()
            .occlusion
            .begin((heap, index), ty);
        if let Some(((h, i), oty)) = outer {
            if once(&N, 1) {
                crate::xlog_info!("D3D12.X: nested occlusion query – the outer one ends there and reports visible, the inner one counts");
            }
            pc::<QueryFn>(this, PC_END_QUERY)(this, h, oty, i);
        }
        if let Some(m) = VISIBLE.lock().unwrap().as_mut() {
            m.remove(&(heap, index));
        }
    }
    pc::<QueryFn>(this, PC_BEGIN_QUERY)(this, heap, ty, index);
}

/// command list #29 EndQuery(heap, type, index) (PC #53)
pub(super) unsafe extern "system" fn end_query_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    if is_occlusion(ty) {
        let (forward, force_visible) = LISTS.lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(this as usize)
            .or_default()
            .occlusion
            .end((heap, index));
        if force_visible {
            let v = if ty == D3D12_QUERY_TYPE_BINARY_OCCLUSION { 1 } else { VISIBLE_SAMPLES };
            VISIBLE.lock().unwrap().get_or_insert_with(HashMap::new).insert((heap, index), v);
        }
        if !forward {
            return;
        }
    }
    pc::<QueryFn>(this, PC_END_QUERY)(this, heap, ty, index);
}

/// command list #30 ResolveQueryData(heap, type, first, count, destination, offset) (PC #54):
/// outer occlusion queries ended early (or queries whose draws were dropped/pending) are written
/// over the resolved data (after it, in COPY_DEST as ResolveQueryData wants the destination).
/// Buffers placed below their console address add `buffer_delta(dest)` to the destination offset.
pub(super) unsafe extern "system" fn resolve_query_data_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, first: u32, count: u32, dest: usize, offset: u64) {
    // (8 bytes a query; pipeline statistics, the largest, are 11 values)
    note_copy_into(this as usize, dest, offset, count as u64 * if ty == D3D12_QUERY_TYPE_PIPELINE_STATISTICS { 88 } else { 8 });
    let pc_offset = offset.saturating_add(real::buffer_delta(dest));
    type F = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);
    pc::<F>(this, PC_RESOLVE_QUERY_DATA)(this, heap, ty, first, count, dest, pc_offset);
    if !is_occlusion(ty) {
        return;
    }
    let visible: Vec<(u32, u64)> = match VISIBLE.lock().unwrap().as_ref() {
        Some(m) if !m.is_empty() => (first..first.saturating_add(count)).filter_map(|i| m.get(&(heap, i)).map(|&v| (i, v))).collect(),
        _ => return,
    };
    let dest = dest as *mut c_void;
    if dest.is_null() || visible.is_empty() {
        return;
    }
    // `dest` may be an adopted console ID3D12Resource whose slot 11 (`GetGPUVirtualAddress`)
    // returns the console CPU UMA address; `WriteBufferImmediate` requires the real PC GPU VA.
    // Guard `real::orig_fn` with `real::is_adopted` so unadopted PC resources (and test fakes)
    // do not read `vt[-1]` out of bounds.
    let pc_va = if real::is_adopted(dest) {
        type GpuVaFn = unsafe extern "system" fn(*mut c_void) -> u64;
        std::mem::transmute::<usize, GpuVaFn>(real::orig_fn(dest, 11))(dest)
    } else if let Some(res) = ID3D12Resource::from_raw_borrowed(&dest) {
        res.GetGPUVirtualAddress()
    } else {
        return;
    };
    if pc_va == 0 {
        return;
    }
    let at = pc_va.saturating_add(pc_offset);
    let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = visible
        .iter()
        .flat_map(|&(i, v)| {
            let a = at + 8 * (i - first) as u64;
            [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a, Value: v as u32 }, D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a + 4, Value: (v >> 32) as u32 }]
        })
        .collect();
    let modes = vec![D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; params.len()];
    type W = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
    pc::<W>(this, PC_WRITE_BUFFER_IMMEDIATE)(this, params.len() as u32, params.as_ptr(), modes.as_ptr());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    struct ReentrantGenerationPso {
        vtable: *const usize,
        refs: AtomicUsize,
        release_outside_lists: *const AtomicBool,
    }

    unsafe extern "system" fn generation_add_ref(this: *mut c_void) -> u32 {
        (*(this as *mut ReentrantGenerationPso)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn generation_release(this: *mut c_void) -> u32 {
        let n = (*(this as *mut ReentrantGenerationPso)).refs.fetch_sub(1, Ordering::Relaxed) - 1;
        let probe = (*(this as *mut ReentrantGenerationPso)).release_outside_lists;
        if n == 0 && !probe.is_null() {
            (*probe).store(LISTS.try_lock().is_ok(), Ordering::Release);
        }
        n as u32
    }

    fn test_generation(
        standin: usize,
    ) -> (super::super::pso_async::NativeGeneration, &'static ReentrantGenerationPso) {
        let vtable = Box::leak(Box::new([0usize, generation_add_ref as usize, generation_release as usize]));
        let native = Box::leak(Box::new(ReentrantGenerationPso {
            vtable: vtable.as_ptr(), refs: AtomicUsize::new(1), release_outside_lists: std::ptr::null(),
        }));
        let raw = native as *mut ReentrantGenerationPso as usize;
        super::super::pso_async::submit_lazy(standin, false, Box::new(move || raw), 0);
        assert_eq!(super::super::pso_async::resolve_wait(standin), raw);
        let resolution = super::super::pso_async::begin_pipeline_resolution(standin).unwrap();
        let super::super::pso_async::PipelineChoice::Available { generation, .. } = resolution.resolve_for_emission()
            else { panic!("completed one-shot generation unavailable") };
        (generation, native)
    }

    #[test]
    fn bundle_generations_propagate_and_final_release_runs_outside_lists() {
        let release_outside_lists = Box::leak(Box::new(AtomicBool::new(false)));
        let standin = 0x7E57_B017usize;
        let bundle = 0x7E57_B018usize;
        let parent = 0x7E57_B019usize;
        let vtable = Box::leak(Box::new([0usize, generation_add_ref as usize, generation_release as usize]));
        let native = Box::leak(Box::new(ReentrantGenerationPso {
            vtable: vtable.as_ptr(), refs: AtomicUsize::new(1), release_outside_lists,
        }));
        let raw = native as *mut ReentrantGenerationPso as usize;
        super::super::pso_async::submit_lazy(standin, false, Box::new(move || raw), 0);
        assert_eq!(super::super::pso_async::resolve_wait(standin), raw);
        let resolution = super::super::pso_async::begin_pipeline_resolution(standin).unwrap();
        let super::super::pso_async::PipelineChoice::Available { generation, .. } = resolution.resolve_for_emission()
            else { panic!("completed one-shot generation unavailable") };
        record_native_generation(bundle, Some(generation.clone()));
        record_native_generation(bundle, Some(generation));
        {
            let g = LISTS.lock().unwrap();
            let state = &g.as_ref().unwrap()[&bundle];
            assert_eq!((state.native_generations.len(), state.native_generation_identities.len()), (1, 1));
        }
        drop(resolution);

        on_execute_bundle(parent, bundle);
        on_execute_bundle(parent, bundle);
        assert_eq!(native_generation_count_for_test(parent), 1, "bundle generation did not reach its parent");
        let raw_owner = super::super::pso_async::released(standin);
        assert_eq!(raw_owner, raw);
        super::super::pso_async::release(raw_owner);
        clear_segment_state(bundle);
        assert!(!release_outside_lists.load(Ordering::Acquire), "bundle clear released the parent's generation");
        clear_segment_state(parent);
        assert!(release_outside_lists.load(Ordering::Acquire), "final COM Release ran under LISTS");
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn title_submission_without_protected_state_uses_the_ordinary_fast_path() {
        let prepared = AtomicBool::new(false);
        let executed = AtomicBool::new(false);
        let mut gate = super::super::alloc_pool::SubmissionGate::default();
        assert!(unsafe {
            submit_title_chunk(
                &mut gate,
                std::ptr::null_mut(),
                &[],
                false,
                false,
                std::iter::empty::<&super::super::pso_async::NativeGeneration>(),
                || prepared.store(true, Ordering::Release),
                || executed.store(true, Ordering::Release),
            )
        });
        assert!(prepared.load(Ordering::Acquire));
        assert!(executed.load(Ordering::Acquire));
        assert!(!gate.failed());
    }

    #[test]
    fn receipted_title_submission_retires_the_exact_generation_on_completion() {
        let standin = 0x7E57_B027usize;
        let vtable = Box::leak(Box::new([0usize, generation_add_ref as usize, generation_release as usize]));
        let native = Box::leak(Box::new(ReentrantGenerationPso {
            vtable: vtable.as_ptr(), refs: AtomicUsize::new(1), release_outside_lists: std::ptr::null(),
        }));
        let raw = native as *mut ReentrantGenerationPso as usize;
        super::super::pso_async::submit_lazy(standin, false, Box::new(move || raw), 0);
        assert_eq!(super::super::pso_async::resolve_wait(standin), raw);
        let resolution = super::super::pso_async::begin_pipeline_resolution(standin).unwrap();
        let super::super::pso_async::PipelineChoice::Available { generation, .. } = resolution.resolve_for_emission()
            else { panic!("completed one-shot generation unavailable") };
        let (receipt, completed) = super::super::alloc_pool::submission_receipt_for_test(0, 1);
        assert!(finish_title_submission(
            super::super::alloc_pool::ReceiptedSubmission::Executed(receipt),
            None,
            false,
            true,
            std::iter::once(&generation),
        ));
        drop(generation);
        drop(resolution);
        let raw_owner = super::super::pso_async::released(standin);
        super::super::pso_async::release(raw_owner);
        assert_eq!(native.refs.load(Ordering::Relaxed), 1, "in-flight receipt released its generation");

        completed.store(1, Ordering::Release);
        super::super::pso_async::sweep_native_generations_for_test();
        assert_eq!(native.refs.load(Ordering::Relaxed), 0, "completed receipt retained its generation");
    }

    #[test]
    fn generation_capture_deduplicates_and_survives_recording_reset() {
        let (first, first_native) = test_generation(0x7E57_B037);
        let (second, second_native) = test_generation(0x7E57_B038);
        let mut state = ListState::default();
        assert!(state.pin_native_generation(first.clone()).is_none());
        assert!(state.pin_native_generation(first.clone()).is_some(), "last-generation fast path missed");
        assert!(state.pin_native_generation(second.clone()).is_none());
        assert!(state.pin_native_generation(first.clone()).is_some(), "identity index missed A/B/A duplicate");
        assert_eq!((state.native_generations.len(), state.native_generation_identities.len()), (2, 2));

        let submitted = state.take_native_generations();
        assert!(state.native_generations.is_empty());
        assert!(state.native_generation_identities.is_empty());
        assert!(state.pin_native_generation(first.clone()).is_none(), "new recording did not accept prior identity");
        assert_eq!((submitted.len(), state.native_generations.len()), (2, 1));

        let retry = 0x7E57_B039;
        LISTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(retry, ListState::default());
        restore_kickoff_native_generations(retry, vec![first.clone(), second.clone(), first.clone()]);
        {
            let g = LISTS.lock().unwrap();
            let restored = &g.as_ref().unwrap()[&retry];
            assert_eq!((restored.native_generations.len(), restored.native_generation_identities.len()), (2, 2));
        }
        clear_segment_state(retry);
        LISTS.lock().unwrap().as_mut().unwrap().remove(&retry);

        drop(state);
        drop(submitted);
        drop(first);
        drop(second);
        super::super::pso_async::release(super::super::pso_async::released(0x7E57_B037));
        super::super::pso_async::release(super::super::pso_async::released(0x7E57_B038));
        assert_eq!((first_native.refs.load(Ordering::Relaxed), second_native.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn creation_keeps_logical_allocator_separate_from_recording_backing() {
        let list = 0x7e57_a101usize as *mut c_void;
        let (logical, backing) = (0x7e57_a110usize, 0x7e57_a120usize);
        unsafe { on_create_with_backing(list, logical, backing, 0, D3D12_COMMAND_LIST_TYPE_DIRECT) };

        let tracked = LISTS.lock().unwrap().as_ref().unwrap()[&(list as usize)].allocator;
        assert_eq!(tracked, logical);
        assert_eq!(super::super::alloc_pool::recording_backing_for_test(list as usize), Some(backing));

        on_allocator_release(logical);
        assert_eq!(LISTS.lock().unwrap().as_ref().unwrap()[&(list as usize)].allocator, 0);
        assert_eq!(super::super::alloc_pool::recording_backing_for_test(list as usize), Some(backing));

        LISTS.lock().unwrap().as_mut().unwrap().remove(&(list as usize));
        super::super::alloc_pool::forget_list(list as usize);
    }

    #[test]
    fn mirror_snapshot_keeps_order_and_refreshes_only_on_miss() {
        let first = (100, 100, 1_000);
        let second = (150, 100, 2_000);
        let mut cache = (0, Vec::new());
        assert_eq!(cached_mirror(&mut cache, 1, 220, 10, || vec![first, second]), Some(2_070));
        // The complete ordered snapshot was retained, not just the range found above: where the
        // ranges overlap, lookup still chooses the first global entry without refreshing.
        assert_eq!(cached_mirror(&mut cache, 1, 160, 10, || panic!("positive hit locked")), Some(1_060));
        assert_eq!(cached_mirror(&mut cache, 1, 400, 10, || panic!("unchanged miss locked")), None);

        let third = (500, 50, 3_000);
        assert_eq!(cached_mirror(&mut cache, 2, 510, 10, || vec![first, second, third]), Some(3_010));
        assert_eq!(mirror_in(&cache.1, 550, 1), None, "ranges are half-open");
    }

    /// A wait that times out with no proven producer stops the title unless
    /// `XCL_WAIT_TIMEOUT_CONTINUE` is exactly 1.
    #[test]
    fn an_unresolved_wait_stops_the_title_unless_told_to_continue() {
        assert!(!wait_timeout_continues_from(None), "default: the title stops");
        assert!(!wait_timeout_continues_from(Some("0")));
        assert!(!wait_timeout_continues_from(Some("yes")));
        assert!(wait_timeout_continues_from(Some("1")));
        assert!(wait_timeout_continues_from(Some(" 1 ")));
    }

    /// `XCL_WAIT_TIMEOUT_CONTINUE=1`: a GPU wait the watchdog releases counts as signalled on the
    /// CPU at its value; GPU signals at or below it no longer stand, later ones do.
    #[test]
    fn a_wait_released_on_the_cpu_marks_its_cell_signalled_there() {
        let mut cell = SyncCell {
            cpu_signaled_max: 3,
            submitted_signals: vec![
                SubmittedSignal { val: 5, queue: 0xA, seq: 1 },
                SubmittedSignal { val: 9, queue: 0xA, seq: 2 },
            ],
            ..Default::default()
        };
        assert!(release_expired_on_cpu(&mut cell, 7).is_none(), "no fence: nothing to signal");
        assert_eq!(cell.cpu_signaled_max, 7);
        assert_eq!(cell.submitted_signals.iter().map(|s| s.val).collect::<Vec<_>>(), vec![9]);
        release_expired_on_cpu(&mut cell, 2);
        assert_eq!(cell.cpu_signaled_max, 7, "a lower release does not take it back");
    }

    /// A recorded `Wait64` cell whose page the title decommitted, or made inaccessible, is listed
    /// as unreadable by the sync-state dump, which does not fault on it.
    #[test]
    fn the_cells_behind_summary_reads_a_decommitted_or_protected_cell_without_faulting() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE};
        unsafe {
            let mem = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert_ne!(mem, 0);
            let (decommitted, protected) = (mem as u64, (mem + 0x1000) as u64);
            *(decommitted as *mut u64) = 7;
            *(protected as *mut u64) = 7;
            {
                let mut g = SYNC_GRAPH.lock().unwrap();
                let gr = g.get_or_insert_with(Default::default);
                for va in [decommitted, protected] {
                    gr.cells.insert(va, SyncCell { recorded_max: 9, ..Default::default() });
                }
            }
            assert_ne!(VirtualFree(mem as *mut c_void, 0x1000, MEM_DECOMMIT), 0);
            let mut old = 0;
            assert_ne!(VirtualProtect((mem + 0x1000) as *const c_void, 0x1000, PAGE_NOACCESS, &mut old), 0);
            // (other tests take the graph too: the summary only tries the lock)
            let mut s = cells_behind_summary();
            for _ in 0..200 {
                if !s.contains("(sync graph locked)") {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
                s = cells_behind_summary();
            }
            if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
                gr.cells.remove(&decommitted);
                gr.cells.remove(&protected);
            }
            VirtualFree(mem as *mut c_void, 0, MEM_RELEASE);
            assert!(s.contains(&format!("{decommitted:#x}: mem unreadable done 0x0")), "{s}");
            assert!(s.contains(&format!("{protected:#x}: mem unreadable done 0x0")), "{s}");
        }
    }

    /// A cell recorded and never submitted names the list that recorded it, and what came to that
    /// recording: here a Reset (which ends it: the Release after comes to none), and no execute.
    /// A recording on the list after a Reset, or on a list made at its address after a Release,
    /// is another one: its execute does not come to the cell.
    #[test]
    fn the_cells_behind_summary_names_the_recording_list_and_what_came_to_it() {
        enable_sync_state_tracing();
        let cell = Box::new([AtomicU64::new(0), AtomicU64::new(0)]);
        let va = &cell[0] as *const AtomicU64 as u64;
        let other = &cell[1] as *const AtomicU64 as u64;
        // (an address no real list has: the box's own)
        let token = Box::new(0u64);
        let list = &*token as *const u64 as usize;
        let summary = || {
            // (other tests take the graph too: the summary only tries the lock)
            for _ in 0..200 {
                let s = cells_behind_summary();
                if !s.contains("(sync graph locked)") {
                    return s;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            panic!("the sync graph stayed locked");
        };
        let entry_of = |va: u64, s: &str| s.split("; ").find(|p| p.starts_with(&format!("{va:#x}:"))).map(str::to_string).unwrap_or_else(|| panic!("{s}"));
        let entry = |s: &str| entry_of(va, s);

        note_write64_recorded(list as *mut c_void, va, 0x619);
        let s = entry(&summary());
        assert!(s.contains(&format!("rec 0x619 by {list:#x} ")), "{s}");
        assert!(s.contains("(executed no, reset no, released no)"), "{s}");
        trace_list(list, ListEvent::Reset);
        on_release(list);
        let s = entry(&summary());
        assert!(s.contains("(executed no, reset +"), "{s}");
        assert!(s.contains(", released no)"), "{s}");

        // a Reset, an unrelated recording on the list, and its execute
        note_write64_recorded(list as *mut c_void, va, 0x61a);
        trace_list(list, ListEvent::Reset);
        note_write64_recorded(list as *mut c_void, other, 1);
        let lists = [list as *mut c_void];
        let plan = submission_plan(&lists);
        unsafe { execute_submission_plan_on(std::ptr::null_mut(), &plan, |chunk| drop(chunk.submit_effects(std::ptr::null_mut()))) };
        let s = summary();
        assert!(entry(&s).contains("rec 0x61a by "), "{s}");
        assert!(entry(&s).contains("(executed no, reset +"), "{s}");
        assert!(entry_of(other, &s).contains("(executed +"), "{s}");

        // a new recording (after the Reset that ends the executed one), a Release, and a list at
        // the same address that records and is executed
        trace_list(list, ListEvent::Reset);
        note_write64_recorded(list as *mut c_void, va, 0x61b);
        on_release(list);
        note_write64_recorded(list as *mut c_void, other, 2);
        let plan = submission_plan(&lists);
        unsafe { execute_submission_plan_on(std::ptr::null_mut(), &plan, |chunk| drop(chunk.submit_effects(std::ptr::null_mut()))) };
        let s = summary();
        assert!(entry(&s).contains("rec 0x61b by "), "{s}");
        assert!(entry(&s).contains("(executed no, reset no, released +"), "{s}");
        assert!(entry_of(other, &s).contains("(executed +"), "{s}");

        // Planning snapshots the exact generation. A Reset and new recording before publication
        // must not redirect the old submission's Executed event to the new recording.
        trace_list(list, ListEvent::Reset);
        note_write64_recorded(list as *mut c_void, va, 0x61c);
        let captured = submission_plan(&lists);
        trace_list(list, ListEvent::Reset);
        note_write64_recorded(list as *mut c_void, other, 3);
        unsafe { execute_submission_plan_on(std::ptr::null_mut(), &captured, |chunk| drop(chunk.submit_effects(std::ptr::null_mut()))) };
        let s = summary();
        assert!(entry(&s).contains("rec 0x61c by "), "{s}");
        assert!(entry(&s).contains("(executed +") && entry(&s).contains(", reset +"), "{s}");
        assert!(entry_of(other, &s).contains("rec 0x3 by "), "{s}");
        assert!(entry_of(other, &s).contains("(executed no, reset no, released no)"), "{s}");

        if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
            gr.cells.remove(&va);
            gr.cells.remove(&other);
        }
        on_release(list);
    }

    // Exercise the console entry point through a fake PC list, without a device or title data.
    fn captured_resource_barriers(input: &[[u32; 8]]) -> Vec<[u32; 8]> {
        #[repr(C)]
        struct FakeList {
            vt: *const usize,
            sent: Mutex<Vec<[u32; 8]>>,
        }
        unsafe extern "system" fn get_type(_: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
            D3D12_COMMAND_LIST_TYPE_DIRECT
        }
        unsafe extern "system" fn barrier(this: *mut c_void, n: u32, b: *const [u32; 8]) {
            let list = &*(this as *const FakeList);
            list.sent.lock().unwrap().extend_from_slice(std::slice::from_raw_parts(b, n as usize));
        }
        let mut orig_vt = [0usize; 64];
        orig_vt[PC_GET_TYPE] = get_type as usize;
        orig_vt[PC_RESOURCE_BARRIER] = barrier as usize;
        let mut ext_vt = [0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        // The CPU fixtures use a stable word value as their resource identity. Give it a
        // real object-shaped backing while the hook runs, including an active capture's GetDesc.
        unsafe extern "system" fn get_desc(_: *mut c_void, out: *mut D3D12_RESOURCE_DESC) -> *mut D3D12_RESOURCE_DESC {
            out.write(D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 64, Height: 1, DepthOrArraySize: 1,
                MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            });
            out
        }
        let mut resource_vt = [0usize; 16];
        resource_vt[10] = get_desc as usize;
        let resource = resource_vt.as_ptr();
        let resource_words = [(&resource as *const _ as usize) as u32, ((&resource as *const _ as usize) as u64 >> 32) as u32];
        let replace_resource = |barriers: &mut [[u32; 8]], from: [u32; 2], to: [u32; 2]| {
            for b in barriers {
                let end = match b[0] { 0 | 2 => 4, 1 => 6, _ => 2 };
                for words in b[2..end].chunks_exact_mut(2) {
                    if words == from { words.copy_from_slice(&to); }
                }
            }
        };
        let mut input = input.to_vec();
        replace_resource(&mut input, [0x1234, 0], resource_words);
        let mut list = FakeList { vt: unsafe { ext_vt.as_ptr().add(1) }, sent: Mutex::new(Vec::new()) };
        unsafe { resource_barrier_x(&mut list as *mut FakeList as *mut c_void, input.len() as u32, input.as_ptr()) };
        let mut sent = list.sent.into_inner().unwrap();
        replace_resource(&mut sent, resource_words, [0x1234, 0]);
        sent
    }

    #[test]
    fn resource_barrier_console_flags_keep_transition_and_split_bits() {
        for extra in [0, 4, 0x0001_0000, 0x8000_0000, 0x8000_0004] {
            for split in [0, 1, 2] {
                // COPY_DEST -> COPY_SOURCE does not need a resource description lookup.
                let input = [0, extra | split, 0x1234, 0, 7, 0x400, 0x800, 0];
                let mut expected = input;
                expected[1] = split;
                assert_eq!(captured_resource_barriers(&[input]), vec![expected], "extra={extra:#x}, split={split}");
            }
        }
    }

    #[test]
    fn resource_barrier_console_flags_keep_uav_ordering_and_aliasing() {
        let transition = [0, 4, 0x1234, 0, u32::MAX, 8, 0x400, 0];
        let mut expected = transition;
        expected[1] = 0;
        let uav = [2, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(captured_resource_barriers(&[transition]), vec![expected, uav]);
        // Explicit UAV barriers still cover all aliases, and duplicate UAV barriers collapse.
        assert_eq!(captured_resource_barriers(&[[2, 4, 0x1234, 0, 0, 0, 0, 0], uav]), vec![uav]);
        assert_eq!(captured_resource_barriers(&[[1, 4, 0, 0, 0, 0, 0, 0]]), vec![[1, 0, 0, 0, 0, 0, 0, 0]]);
    }

    #[test]
    fn resource_barrier_console_flags_leave_existing_filters_in_place() {
        assert!(captured_resource_barriers(&[]).is_empty());
        assert!(captured_resource_barriers(&[[3, 4, 0, 0, 0, 0, 0, 0]]).is_empty());
        // Removing console-only state bits can still leave a transition with no work to do.
        assert!(captured_resource_barriers(&[[0, 4, 0x1234, 0, 0, 0x8000_0400, 0x400, 0]]).is_empty());
        let input = [0, 4, 0x1234, 0, 0, 0x8000_0400, 0x800, 0];
        assert_eq!(captured_resource_barriers(&[input]), vec![[0, 0, 0x1234, 0, 0, 0x400, 0x800, 0]]);
    }

    #[test]
    fn gpu_resource_barrier_console_hint_preserves_texture_copy() {
        use crate::gpu::{fence, recomp::runner};

        let dev = match runner::device() {
            Ok(dev) => dev,
            Err(e) => {
                eprintln!("barrier test device unavailable: {e}");
                runner::skip_test("no GPU");
                return;
            }
        };
        println!("barrier texture-copy adapter: {}", runner::adapter().unwrap_or_default());
        unsafe {
            // One ordinary transition, the console hint, and the hint on a split pair.
            // No command uses the texture between the two halves of the split transition.
            for flags in [&[0u32][..], &[0x0001_0000][..], &[0x0001_0001, 0x0001_0002][..]] {
                let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 4, Height: 4, DepthOrArraySize: 1, MipLevels: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..Default::default()
                };
                let mut texture: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut texture).unwrap();
                let texture = texture.unwrap();
                let rtv: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).unwrap();
                let handle = rtv.GetCPUDescriptorHandleForHeapStart();
                dev.CreateRenderTargetView(&texture, None, handle);
                list.ClearRenderTargetView(handle, &[0.0, 1.0, 0.0, 1.0], None);

                let raw = texture.as_raw() as usize as u64;
                let input: Vec<[u32; 8]> = flags.iter().map(|&f| [0, f, raw as u32, (raw >> 32) as u32, u32::MAX, 4, 0x800, 0]).collect();
                let mut expected = input.clone();
                for b in &mut expected { b[1] &= 3; }
                let translated = captured_resource_barriers(&input);
                // Fail the original-runtime control before submitting a missing transition.
                assert_eq!(translated, expected, "console hint must retain the complete native transition: {flags:x?}");
                let native: Vec<D3D12_RESOURCE_BARRIER> = translated.iter().map(|b| std::mem::transmute_copy(b)).collect();
                list.ResourceBarrier(&native);

                let readback_desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 4 * 256, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
                };
                let mut readback: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
                let readback = readback.unwrap();
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: desc.Format, Width: 4, Height: 4, Depth: 1, RowPitch: 256 }
                    } },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: transmute_copy(&texture), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let complete = queue.Signal(&done, 1).map_err(|e| format!("Signal: {e}"))
                    .and_then(|_| fence::wait(&done, 1, &dev, None, 2000).map_err(|e| e.to_string()));
                if let Err(e) = complete {
                    // A timed-out copy may still own these objects. Retain this tiny batch until
                    // the test process exits rather than releasing memory the GPU could use.
                    std::mem::forget((queue, alloc, list, done, texture, rtv, readback));
                    panic!("bounded barrier texture copy did not complete: {e}");
                }
                let mut ptr = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut ptr)).unwrap();
                let pixels: Vec<[u8; 4]> = (0..4).flat_map(|y| (0..4).map(move |x| (ptr as *const u8).add(y * 256 + x * 4).cast::<[u8; 4]>().read_unaligned())).collect();
                readback.Unmap(0, None);
                assert_eq!(pixels, vec![[0, 255, 0, 255]; 16], "texture copy after flags {flags:x?}");
            }
        }
    }
    use std::sync::Arc;

    /// An internal wait shows in the watchdog's summary while its fence is short of the value,
    /// and is gone from it once the fence gets there.
    #[test]
    fn an_internal_wait_is_listed_until_its_fence_reaches_the_value() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let queue = fence.as_raw() as usize; // any address no real queue has
            note_internal_wait(queue, "test wait 7d1c", &fence, 3);
            let s = internal_waits_summary();
            assert!(s.contains("test wait 7d1c >= 0x3 (done 0x0"), "{s}");
            fence.Signal(3).unwrap();
            let s = internal_waits_summary();
            assert!(!s.contains("test wait 7d1c"), "{s}");
        }
    }

    /// The closed standby the failed-split route submits must be executable: a list straight from
    /// CreateCommandList1 was never recorded and removes the device when submitted.
    #[test]
    fn a_reserved_closed_standby_can_be_submitted() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let Some(standby) = reserve_closed_segment(0x5ba5_d000 as *mut c_void, &dev, ty) else {
                crate::gpu::recomp::runner::skip_test("no ID3D12Device4");
                return;
            };
            let list = ID3D12CommandList::from_raw_borrowed(&standby).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            queue.ExecuteCommandLists(&[Some(list.clone())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let t = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 && t.elapsed().as_secs() < 5 && dev.GetDeviceRemovedReason().is_ok() {
                std::thread::yield_now();
            }
            assert!(dev.GetDeviceRemovedReason().is_ok(), "submitting the standby removed the device: {:?}", dev.GetDeviceRemovedReason());
            assert!(fence.GetCompletedValue() >= 1);
        }
    }

    #[test]
    fn copy_scratch_budget_counts_resources_until_gpu_completion() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let first: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            first.Close().unwrap();
            let second: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut cache = CopyScratchCache::default();
            let small = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
            let first_raw = first.as_raw();
            let second_raw = second.as_raw();

            let initial = cache.acquire(first_raw, small).unwrap();
            assert_eq!(cache.acquire(first_raw, MAX_COPY_SCRATCH), Some(initial));
            let scratch = cache.active.get_mut(&(first_raw as usize)).unwrap();
            scratch.recorded = false;
            scratch.marks.insert(fence.as_raw() as usize, (fence.clone(), 1));
            assert!(cache.acquire(first_raw, MAX_COPY_SCRATCH).is_none());
            fence.Signal(1).unwrap();
            assert_eq!(cache.acquire(first_raw, MAX_COPY_SCRATCH).unwrap().1, MAX_COPY_SCRATCH);
            assert_eq!(cache.bytes, MAX_COPY_SCRATCH);
            assert!(cache.acquire(second_raw, small).is_none());

            let scratch = cache.active.get_mut(&(first_raw as usize)).unwrap();
            scratch.marks.insert(fence.as_raw() as usize, (fence.clone(), 2));
            cache.release(first_raw as usize);
            assert!(!cache.active.contains_key(&(first_raw as usize)));
            assert!(cache.acquire(second_raw, small).is_none());
            fence.Signal(2).unwrap();
            assert!(cache.acquire(second_raw, small).is_some());
            assert_eq!(cache.bytes, small);
            cache.release(second_raw as usize);
            assert_eq!(cache.bytes, 0);
        }
    }

    /// A new list at the address of one whose release was not seen does not go on with that one's
    /// scratch buffer, which the GPU may still use.
    #[test]
    fn a_new_list_takes_no_copy_scratch_of_an_earlier_one() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let (raw, key) = (list.as_raw(), list.as_raw() as usize);
            let small = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
            {
                let mut cache = COPY_SCRATCH.lock().unwrap();
                cache.acquire(raw, small).unwrap();
                let scratch = cache.active.get_mut(&key).unwrap();
                scratch.marks.insert(fence.as_raw() as usize, (fence.clone(), 1));
            }
            on_create(raw, alloc.as_raw() as usize, 0, ty);
            // Read under the lock, assert after it: a failed assert while holding the lock poisons
            // COPY_SCRATCH, the list's Release hook (`real::cl_release`) then panics on it while the
            // test unwinds, and that aborts the whole test binary.
            let (active, released) = {
                let cache = COPY_SCRATCH.lock().unwrap();
                (cache.active.contains_key(&key), !cache.released.is_empty())
            };
            assert!(!active, "the new list starts without one");
            assert!(released, "the GPU may still use it: it stays until the fence");
            fence.Signal(1).unwrap();
            on_release(key);
            super::super::alloc_pool::forget(alloc.as_raw() as usize);
        }
    }

    static RESET_TEST_HR: AtomicU32 = AtomicU32::new(0);
    static RESET_TEST_REPLACEMENT_VT: AtomicU64 = AtomicU64::new(0);

    unsafe extern "system" fn reset_test_native(this: *mut c_void, _alloc: usize, _pso: usize) -> i32 {
        *(this as *mut usize) = RESET_TEST_REPLACEMENT_VT.load(Ordering::SeqCst) as usize;
        RESET_TEST_HR.load(Ordering::SeqCst) as i32
    }

    #[test]
    fn nested_occlusion_queries_end_the_outer_one_early() {
        let ty = D3D12_QUERY_TYPE_OCCLUSION;
        let mut o = Occlusion::default();
        assert_eq!(o.begin((1, 0), ty), None);
        assert_eq!(o.begin((1, 1), ty), Some(((1, 0), ty)), "the outer one ends on the PC");
        assert_eq!(o.end((1, 1)), (true, false), "inner: a real query");
        // more per-object queries inside the same outer one
        assert_eq!(o.begin((1, 2), ty), None);
        assert_eq!(o.end((1, 2)), (true, false));
        assert_eq!(o.end((1, 0)), (false, true), "outer: already ended");
        // overlapping, not nested: A, B, end A, end B
        assert_eq!(o.begin((1, 3), ty), None);
        assert_eq!(o.begin((1, 4), ty), Some(((1, 3), ty)));
        assert_eq!(o.end((1, 3)), (false, true));
        assert_eq!(o.end((1, 4)), (true, false));
        assert!(o.active.is_none() && o.ended_early.is_empty());
        assert_eq!(o.end((1, 9)), (true, false), "a query not begun here: the PC's call");
        // kickoff ends active query early so EndQuery later resolves it as visible
        assert_eq!(o.begin((1, 5), D3D12_QUERY_TYPE_BINARY_OCCLUSION), None);
        assert_eq!(o.end_active_early(), Some(((1, 5), D3D12_QUERY_TYPE_BINARY_OCCLUSION)));
        assert_eq!(o.end((1, 5)), (false, true));
    }

    #[test]
    fn reset_commits_state_only_after_native_success_and_always_restores_vtable() {
        let mut native = vec![0usize; PC_RESET + 1];
        native[PC_RESET] = reset_test_native as usize;
        let extended = [native.as_ptr() as usize, 0usize];
        let ext = unsafe { extended.as_ptr().add(1) as usize };
        let replacement = 0x1234_5678usize;
        let mut object = [ext];
        let list = object.as_mut_ptr() as *mut c_void;
        let key = list as usize;

        unsafe { on_create(list, 0x111, 0, D3D12_COMMAND_LIST_TYPE_DIRECT) };
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&key).unwrap();
            s.allocator = 0x111;
            s.bound_pso = 0x222;
            s.depth_bias = Some(DepthBias::new(-3.5, 0.25, 2.0));
            s.native_bias = DepthBias::new(-4.5, 0.5, 3.0);
            s.pso_dirty = true;
            s.using_fallback_pso = true;
            s.index_buffer = Some(D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: 0x5000,
                SizeInBytes: 12,
                Format: DXGI_FORMAT_R32_UINT,
            });
            s.depth_bounds = Some((0.25, 0.75));
            s.written_ranges.insert(WriteProducer::InSegmentGpu, 0x6000, 0x10);
        }
        super::super::bind::seed_reset_state_for_test(list, 0x111, 0x222, vec![0x4444]);
        RESET_TEST_REPLACEMENT_VT.store(replacement as u64, Ordering::SeqCst);

        RESET_TEST_HR.store(0x8000_4005, Ordering::SeqCst);
        assert!(unsafe { reset_x(list, 0x333, 0x555) } < 0);
        let failed_backing = super::super::alloc_pool::recording_backing_for_test(key);
        assert_eq!(object[0], ext);
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!((s.allocator, s.bound_pso), (0x111, 0x222));
            assert_eq!(s.depth_bias, Some(DepthBias::new(-3.5, 0.25, 2.0)));
            assert_eq!(s.native_bias, DepthBias::new(-4.5, 0.5, 3.0));
            assert!(s.pso_dirty && s.using_fallback_pso);
            assert_eq!(s.index_buffer.as_ref().unwrap().BufferLocation, 0x5000);
            assert_eq!(s.depth_bounds, Some((0.25, 0.75)));
            assert_eq!(s.written_ranges.in_segment_gpu.ranges, BTreeMap::from([(0x6000, 0x6010)]));
        }
        assert_eq!(
            super::super::bind::reset_state_for_test(list),
            (0x111, 0x222, vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![0x4444])
        );
        assert!(super::super::console_list::take_pending_for_test(0x111).is_empty());

        RESET_TEST_HR.store(0, Ordering::SeqCst);
        assert_eq!(unsafe { reset_x(list, 0x333, 0x555) }, 0);
        let successful_backing = super::super::alloc_pool::recording_backing_for_test(key);
        assert_eq!(object[0], ext);
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!((s.allocator, s.bound_pso), (0x333, 0x555));
            assert!(s.depth_bias.is_none() && s.index_buffer.is_none() && s.depth_bounds.is_none());
            assert!(!s.pso_dirty && !s.using_fallback_pso && s.written_ranges.is_empty());
        }
        assert_eq!(
            super::super::bind::reset_state_for_test(list),
            (0x333, 0x555, Vec::new(), Vec::new(), Vec::new())
        );
        assert_eq!(super::super::console_list::take_pending_for_test(0x111), vec![0x4444]);

        LISTS.lock().unwrap().as_mut().unwrap().remove(&key);
        super::super::bind::remove_state_for_test(list);
        unsafe {
            super::super::alloc_pool::forget(0x111);
            super::super::alloc_pool::forget(0x333);
        }
        assert_eq!(failed_backing, Some(0x111), "failed Reset must keep holding the old recording's allocator");
        assert_eq!(successful_backing, Some(0x333), "only successful Reset moves the recording to its new allocator");
    }

    unsafe extern "system" fn reset_ok_native(_this: *mut c_void, _alloc: usize, _pso: usize) -> i32 {
        0
    }

    /// A list reset on a title allocator that stands for another PC allocator (`alloc_pool`)
    /// records on that one, and what the recording puts in unified memory is held for it after
    /// the next Reset – not for the title's allocator, whose own submits the GPU may have passed
    /// while this one's still run (#1358).
    #[test]
    fn a_recordings_memory_is_held_for_the_pc_allocator_it_is_on() {
        let mut native = vec![0usize; PC_RESET + 1];
        native[PC_RESET] = reset_ok_native as usize;
        let extended = [native.as_ptr() as usize, 0usize];
        let ext = unsafe { extended.as_ptr().add(1) as usize };
        let mut object = [ext];
        let list = object.as_mut_ptr() as *mut c_void;
        let key = list as usize;
        let ids = Box::new([0u8; 4]);
        let title = ids.as_ptr() as usize;
        let (pc, other, block) = (title + 1, title + 2, title + 3);
        super::super::alloc_pool::stand_in_for_test(title, pc);
        unsafe { on_create(list, title, 0, D3D12_COMMAND_LIST_TYPE_DIRECT) };

        assert_eq!(unsafe { reset_x(list, title, 0) }, 0);
        let recording_on = super::super::bind::reset_state_for_test(list).0;
        super::super::bind::seed_reset_state_for_test(list, recording_on, 0, vec![block]);
        assert_eq!(unsafe { reset_x(list, other, 0) }, 0);
        let held_for_pc = super::super::console_list::take_pending_for_test(pc);
        let held_for_title = super::super::console_list::take_pending_for_test(title);

        LISTS.lock().unwrap().as_mut().unwrap().remove(&key);
        super::super::bind::remove_state_for_test(list);
        super::super::alloc_pool::stand_in_for_test(title, title);
        unsafe {
            super::super::alloc_pool::forget(title);
            super::super::alloc_pool::forget(pc);
            super::super::alloc_pool::forget(other);
        }
        assert_eq!(recording_on, pc, "the recording is on the PC allocator the title's allocator stands for");
        assert_eq!(held_for_pc, vec![block]);
        assert!(held_for_title.is_empty());
    }

    #[test]
    fn fill_patterns_repeat_the_value() {
        assert_eq!(pattern(&0x1122_3344u32.to_le_bytes())[..8], [0x44, 0x33, 0x22, 0x11, 0x44, 0x33, 0x22, 0x11]);
        assert_eq!(pattern(&[7u8; 16]), [7u8; 16]);
    }

    #[test]
    fn wait_comparison_ops_cover_all_d3d12_funcs() {
        assert_eq!(compare_u64(0, 1, 1), None);
        assert_eq!(compare_u64(9, 1, 1), None);
        assert_eq!(compare_u64(1, 5, 5), Some(false)); // NEVER
        assert_eq!(compare_u64(2, 4, 5), Some(true));  // LESS
        assert_eq!(compare_u64(2, 5, 5), Some(false));
        assert_eq!(compare_u64(3, 5, 5), Some(true));  // EQUAL
        assert_eq!(compare_u64(3, 4, 5), Some(false));
        assert_eq!(compare_u64(4, 5, 5), Some(true));  // LESS_EQUAL
        assert_eq!(compare_u64(4, 6, 5), Some(false));
        assert_eq!(compare_u64(5, 6, 5), Some(true));  // GREATER
        assert_eq!(compare_u64(5, 5, 5), Some(false));
        assert_eq!(compare_u64(6, 4, 5), Some(true));  // NOT_EQUAL
        assert_eq!(compare_u64(6, 5, 5), Some(false));
        assert_eq!(compare_u64(7, 5, 5), Some(true));  // GREATER_EQUAL
        assert_eq!(compare_u64(7, 4, 5), Some(false));
        assert_eq!(compare_u64(8, 0, 1), Some(true));  // ALWAYS
    }

    #[test]
    fn pending_wait_32_and_64_bit_with_mask_and_cross_thread_signal() {
        let cell32 = Arc::new(AtomicU32::new(0xDEAD_0000));
        let w32 = PendingWait {
            va: Arc::as_ptr(&cell32) as u64,
            width: 4,
            comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
            value: 0x0000_005A,
            mask: 0x0000_00FF,
            after_write64: false,
        };
        unsafe {
            assert!(!w32.is_satisfied());
            let writer = cell32.clone();
            let t = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(5));
                writer.store(0xBEEF_005A, Ordering::Release);
            });
            assert!(w32.wait_until_satisfied(Duration::from_secs(1)));
            t.join().unwrap();
        }

        let cell64 = Arc::new(AtomicU64::new(0x0000_0001_0000_0000));
        let w64 = PendingWait {
            va: Arc::as_ptr(&cell64) as u64,
            width: 8,
            comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
            value: 0x0000_0002_0000_0000,
            mask: u64::MAX,
            after_write64: false,
        };
        unsafe {
            assert!(!w64.is_satisfied());
            // bounded timeout returns false without hanging
            assert!(!w64.wait_until_satisfied(Duration::from_millis(5)));
            cell64.store(0x0000_0002_0000_0001, Ordering::Release);
            assert!(w64.is_satisfied());
        }
    }

    #[test]
    fn committed_uma_wait_uses_coherent_atomic_observation() {
        let uma = crate::gpu::uma::Uma::for_runtime(16 << 20).expect("runtime UMA");
        let va = uma.alloc(0x1000, 8).expect("coherent wait cell") as usize;
        unsafe {
            (*(va as *const AtomicU64)).store(0x1234_5678_9abc_def0, Ordering::Release);
            let wait = PendingWait {
                va: va as u64,
                width: 8,
                comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
                value: 0,
                mask: u64::MAX,
                after_write64: false,
            };
            assert_eq!(wait.read_value_with_uma(Some(&uma)), Some(0x1234_5678_9abc_def0));
        }
        assert!(uma.free(va));
    }

    #[test]
    fn an_unreadable_wait_cell_is_not_completion() {
        let wait = PendingWait {
            va: 0x10,
            width: 8,
            comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
            value: 1,
            mask: u64::MAX,
            after_write64: false,
        };
        unsafe {
            assert_eq!(wait.observe(), PendingWaitObservation::Unreadable);
            assert!(!wait.is_satisfied(), "an unmapped address is not proof of completion");
            assert!(!wait_all_on_with_timeout(
                std::ptr::null_mut(),
                &[wait],
                &[],
                Duration::ZERO,
            ), "an unreadable CPU fallback must fail closed");
        }
    }

    #[test]
    fn a_decommitted_wait_cell_is_observed_without_faulting() {
        use windows_sys::Win32::System::Memory::{
            VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE,
            PAGE_READWRITE,
        };

        unsafe {
            let page = VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            );
            assert!(!page.is_null());
            let value = page.cast::<AtomicU64>();
            (*value).store(7, Ordering::Release);
            let wait = PendingWait {
                va: value as u64,
                width: 8,
                comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
                value: 7,
                mask: u64::MAX,
                after_write64: false,
            };
            assert_eq!(wait.observe(), PendingWaitObservation::Satisfied);

            assert_ne!(VirtualFree(page, 0x1000, MEM_DECOMMIT), 0);
            assert_eq!(wait.observe(), PendingWaitObservation::Unreadable);
            assert_ne!(VirtualFree(page, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn only_proven_gpu_resolution_or_a_satisfied_cpu_wait_allows_submission() {
        assert!(wait_is_resolved(GpuWaitOutcome::Handled, false));
        assert!(wait_is_resolved(GpuWaitOutcome::CycleSkip, false),
            "an exact graph-proven cycle break remains accepted");
        assert!(wait_is_resolved(GpuWaitOutcome::FallbackCpu, true),
            "a delayed producer that satisfies the CPU poll remains accepted");
        assert!(!wait_is_resolved(GpuWaitOutcome::FallbackCpu, false),
            "elapsed time alone must not make an unresolved dependency executable");

        let cell = AtomicU64::new(0);
        let missing = PendingWait {
            va: &cell as *const AtomicU64 as u64,
            width: 8,
            comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
            value: 1,
            mask: u64::MAX,
            after_write64: false,
        };
        assert!(!unsafe {
            wait_all_on_with_timeout(std::ptr::null_mut(), &[missing], &[], Duration::from_millis(1))
        });
    }

    /// `wait_all_on` says whether it waited on the CPU: not for waits that held already, yes for
    /// one it had to wait for there (no queue: no GPU wait); the split's sparse update follows it
    /// only with `XCL_SPLIT_SPARSE=cpu`.
    #[test]
    fn a_split_brings_the_sparse_windows_up_to_date_after_a_cpu_wait() {
        let cell = Arc::new(AtomicU64::new(7));
        let w = |value: u64| PendingWait {
            va: Arc::as_ptr(&cell) as u64,
            width: 8,
            comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
            value,
            mask: u64::MAX,
            after_write64: false,
        };
        unsafe {
            assert!(!wait_all_on(std::ptr::null_mut(), &[w(5), w(7)], &[]), "held already: no CPU wait");
            let writer = cell.clone();
            let t = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(5));
                writer.store(9, Ordering::Release);
            });
            assert!(wait_all_on(std::ptr::null_mut(), &[w(5), w(9)], &[]), "waited on the CPU");
            t.join().unwrap();
        }
        assert!(sparse_again_after_split(true, true));
        assert!(!sparse_again_after_split(false, true));
        assert!(sparse_again_after_split(false, false) && sparse_again_after_split(true, false), "as before without the switch");
    }

    #[test]
    fn wait64_flags_do_not_mask_the_recorded_counter() {
        let mut token = 0usize;
        let list = (&mut token as *mut usize).cast::<c_void>();
        let cell = AtomicU64::new(0);
        let va = &cell as *const AtomicU64 as u64;
        for (current, wanted, flags) in [
            (2, 1, 0x231_0000_0001),
            (4, 2, 0x270_0000_0001),
            (5, 3, 1),
            (0x1_0000_0004, 0x1_0000_0002, 0xFFFF_FFFF_0000_0001),
        ] {
            cell.store(current, Ordering::Release);
            unsafe { wait64_x(list, va, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, wanted, flags, 0) };
            let waits = LISTS.lock().unwrap().as_ref().unwrap()[&(list as usize)].pending_waits.clone();
            on_release(list as usize);
            assert_eq!(waits.len(), 1);
            let w = waits[0];
            assert_eq!((w.width, w.value, w.mask), (8, wanted, u64::MAX));
            unsafe {
                assert!(w.is_satisfied(), "{current:#x} >= {wanted:#x}; flags {flags:#x}");
                cell.store(wanted - 1, Ordering::Release);
                assert!(!w.is_satisfied(), "a genuinely earlier counter must still wait");
                cell.store(wanted, Ordering::Release);
                assert!(w.is_satisfied(), "equality satisfies GREATER_EQUAL");
            }
        }
    }

    #[test]
    fn wait64_satisfied_counter_keeps_submission_together_and_wait32_keeps_its_mask() {
        let mut tokens = [0usize; 2];
        let lists = [tokens.as_mut_ptr().cast::<c_void>(), unsafe { tokens.as_mut_ptr().add(1).cast::<c_void>() }];
        let cell64 = AtomicU64::new(4);
        let cell32 = AtomicU32::new(0xDEAD_005A);
        unsafe {
            wait64_x(lists[1], &cell64 as *const AtomicU64 as u64, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 2, 0x270_0000_0001, 0);
        }
        let mut batches = Vec::new();
        unsafe { execute_with_waits(&lists, |chunk| batches.push(chunk.to_vec())) };
        on_release(lists[1] as usize);
        assert_eq!(batches, vec![lists.to_vec()], "an already satisfied counter must not split the batch");

        unsafe {
            wait32_x(lists[0], &cell32 as *const AtomicU32 as u64, D3D12_COMPARISON_FUNC_EQUAL.0 as u32, 0x5A, 0xFF, 0);
        }
        let waits = LISTS.lock().unwrap().as_ref().unwrap()[&(lists[0] as usize)].pending_waits.clone();
        on_release(lists[0] as usize);
        assert_eq!(waits.len(), 1);
        assert_eq!((waits[0].width, waits[0].mask), (4, 0xFF));
        assert!(unsafe { waits[0].is_satisfied() });
    }

    #[test]
    fn written_ranges_detect_same_list_overlap() {
        let mut s = ListState::default();
        s.note_write(WriteProducer::InSegmentGpu, 0x1000, 8);
        assert!(s.has_prior_write(0x1000, 4));
        assert!(s.has_prior_write(0x1004, 4));
        assert!(s.has_prior_write(0x0FFC, 8));
        assert!(!s.has_prior_write(0x1008, 4));
        assert!(!s.has_prior_write(0x0FF8, 8));
    }

    #[test]
    fn write_producer_ranges_keep_overlapping_kinds_separate() {
        let mut s = ListState::default();
        s.note_write(WriteProducer::InSegmentGpu, 0x1000, 0x10);
        s.note_write(WriteProducer::CpuAfterSegment, 0x1008, 0x10);

        assert!(s.written_ranges.overlaps(WriteProducer::InSegmentGpu, 0x1008, 8));
        assert!(!s.written_ranges.overlaps(WriteProducer::InSegmentGpu, 0x1010, 8));
        assert!(s.written_ranges.overlaps(WriteProducer::CpuAfterSegment, 0x1010, 8));
        assert!(!s.written_ranges.overlaps(WriteProducer::CpuAfterSegment, 0x1000, 8));
        assert!(s.has_prior_write(0x1008, 8));
        assert!(!s.has_prior_write(0x1010, 8), "deferred CPU writes do not change wait policy in this wave");
    }

    #[test]
    fn wait_placement_distinguishes_cpu_gpu_and_write64_producers() {
        let mut s = ListState::default();
        assert_eq!(s.wait_placement(0x1000, 8), WaitPlacement::RecordHere);

        s.note_write(WriteProducer::CpuAfterSegment, 0x2000, 8);
        assert_eq!(s.wait_placement(0x1000, 8), WaitPlacement::RecordHere, "an unrelated CPU write does not split");
        assert_eq!(s.wait_placement(0x2000, 8), WaitPlacement::SplitAfterCpuWrite);

        s.note_write(WriteProducer::InSegmentGpu, 0x3000, 8);
        assert_eq!(s.wait_placement(0x3000, 8), WaitPlacement::OrderedInSegment);
        s.note_write(WriteProducer::CpuAfterSegment, 0x3004, 4);
        assert_eq!(s.wait_placement(0x3000, 8), WaitPlacement::SplitAfterCpuWrite, "CPU completion wins for mixed overlapping producers");

        s.write64_signals.push((0x4000, 1));
        assert_eq!(s.wait_placement(0x5000, 8), WaitPlacement::SplitAfterWrite64);
        assert_eq!(s.wait_placement(0x3000, 4), WaitPlacement::OrderedInSegment, "GPU overlap keeps the existing Write64 behavior");
    }

    #[test]
    fn failed_cpu_wait_split_never_reuses_the_producer_segment() {
        let producer = 0x7ffe_1801usize as *mut c_void;
        let continuation = 0x7ffe_1802usize as *mut c_void;
        assert_eq!(wait_target_after_split(producer, WaitPlacement::SplitAfterCpuWrite, Some(continuation)), Some(continuation));
        assert_eq!(wait_target_after_split(producer, WaitPlacement::SplitAfterCpuWrite, None), None);
        assert_eq!(
            wait_target_after_split(producer, WaitPlacement::SplitAfterWrite64, None),
            Some(producer),
            "keep the existing Write64 fallback"
        );
    }

    #[test]
    fn recorded_writes_merge_ascending_ranges() {
        let mut writes = RecordedWrites::default();
        writes.insert(0x1000, 8);
        writes.insert(0x1008, 8);
        writes.insert(0x1010, 8);
        assert_eq!(writes.ranges, BTreeMap::from([(0x1000, 0x1018)]));
    }

    #[test]
    fn recorded_writes_merge_descending_ranges() {
        let mut writes = RecordedWrites::default();
        writes.insert(0x1010, 8);
        writes.insert(0x1008, 8);
        writes.insert(0x1000, 8);
        assert_eq!(writes.ranges, BTreeMap::from([(0x1000, 0x1018)]));
    }

    #[test]
    fn recorded_writes_merge_adjacent_ranges() {
        let mut writes = RecordedWrites::default();
        writes.insert(0x1000, 4);
        writes.insert(0x1008, 4);
        writes.insert(0x1004, 4);
        assert_eq!(writes.ranges, BTreeMap::from([(0x1000, 0x100c)]));
    }

    #[test]
    fn recorded_writes_merge_overlapping_ranges() {
        let mut writes = RecordedWrites::default();
        writes.insert(0x1008, 0x10);
        writes.insert(0x1000, 0x10);
        writes.insert(0x1004, 0x20);
        assert_eq!(writes.ranges, BTreeMap::from([(0x1000, 0x1024)]));
    }

    #[test]
    fn recorded_writes_keep_disjoint_ranges() {
        let mut writes = RecordedWrites::default();
        writes.insert(0x1000, 8);
        writes.insert(0x1020, 8);
        writes.insert(0x1040, 8);
        assert_eq!(writes.ranges, BTreeMap::from([(0x1000, 0x1008), (0x1020, 0x1028), (0x1040, 0x1048)]));
    }

    #[test]
    fn recorded_writes_handle_thousands_in_reverse_order() {
        let mut writes = RecordedWrites::default();
        for i in (0..4096u64).rev() {
            writes.insert(0x1000 + i * 16, 8);
        }
        assert_eq!(writes.ranges.len(), 4096);
        for i in 0..4096u64 {
            let va = 0x1000 + i * 16;
            assert!(writes.overlaps(va, 8));
            assert!(!writes.overlaps(va + 8, 8));
        }
    }

    #[test]
    fn recorded_writes_preserve_gaps_when_bridging_neighbors() {
        let mut writes = RecordedWrites::default();
        writes.insert(0x1000, 8);
        writes.insert(0x1010, 8);
        writes.insert(0x1020, 8);
        assert!(!writes.overlaps(0x1008, 8));
        assert!(!writes.overlaps(0x1018, 8));

        writes.insert(0x1008, 8);
        assert!(writes.overlaps(0x1008, 8));
        assert!(!writes.overlaps(0x1018, 8));
        assert_eq!(writes.ranges, BTreeMap::from([(0x1000, 0x1018), (0x1020, 0x1028)]));
    }

    #[test]
    fn submission_plan_preserves_segment_and_effect_order() {
        let lists = [0x7ffe_1001usize as *mut c_void, 0x7ffe_1002usize as *mut c_void, 0x7ffe_1003usize as *mut c_void];
        {
            let mut g = LISTS.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            for (i, &list) in lists.iter().enumerate() {
                m.insert(
                    list as usize,
                    ListState {
                        write64_signals: vec![(0x1000 + i as u64, 10 + i as u64)],
                        sparse_writes: vec![(0x2000 + i, 4 + i)],
                        texture_writes: vec![super::super::title_textures::GpuWrite { res: 0x2500 + i, generation: 20 + i as u64, subresource: i as u32 }],
                        texture_transitions: vec![super::super::title_textures::Transition {
                            res: 0x2600 + i,
                            generation: 30 + i as u64,
                            subresource: i as u32,
                            after: D3D12_RESOURCE_STATE_COPY_DEST,
                        }],
                        cpu_writes: vec![super::super::cpu_writes::Write::Immediate(0x3000 + i, vec![i as u32])],
                        ..Default::default()
                    },
                );
            }
        }

        let plan = SubmissionPlan::capture(&lists);
        {
            let mut g = LISTS.lock().unwrap();
            let m = g.as_mut().unwrap();
            for &list in &lists {
                m.remove(&(list as usize));
            }
        }

        assert_eq!(plan.lists, lists);
        for (i, effects) in plan.effects.iter().enumerate() {
            assert_eq!(effects.write64_signals, vec![(0x1000 + i as u64, 10 + i as u64)]);
            assert_eq!(effects.sparse_writes, vec![(0x2000 + i, 4 + i)]);
            assert_eq!(effects.texture_writes, vec![super::super::title_textures::GpuWrite { res: 0x2500 + i, generation: 20 + i as u64, subresource: i as u32 }]);
            assert_eq!(effects.texture_transitions.iter().map(|t| t.res).collect::<Vec<_>>(), vec![0x2600 + i]);
            match effects.cpu_writes.as_slice() {
                [super::super::cpu_writes::Write::Immediate(va, words)] => assert_eq!((*va, words.as_slice()), (0x3000 + i, &[i as u32][..])),
                _ => panic!("unexpected CPU write snapshot"),
            }
        }
    }

    #[test]
    fn submission_plan_keeps_shared_view_use_after_rerecord() {
        let list = 0x7ffe_1901usize as *mut c_void;
        LISTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(
            list as usize,
            ListState { uses_shared_view_heap: true, ..Default::default() },
        );
        let plan = SubmissionPlan::capture(&[list]);
        LISTS.lock().unwrap().as_mut().unwrap().get_mut(&(list as usize)).unwrap().uses_shared_view_heap = false;

        let mut captured = false;
        unsafe {
            execute_submission_plan_on(std::ptr::null_mut(), &plan, |chunk| captured = chunk.uses_shared_view_heap());
        }
        assert!(captured, "the immutable plan owns the submitted recording's flag");
        LISTS.lock().unwrap().as_mut().unwrap().remove(&(list as usize));
    }

    #[test]
    fn console_replay_registers_the_physical_list_until_release() {
        let pc = 0x7ffe_1981usize as *mut c_void;
        assert!(!has_list_state(pc as usize));
        unsafe { on_console_list_create(pc, 0xa110, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE) };
        note_shared_view_heap_use(pc);

        let plan = SubmissionPlan::capture(&[pc]);
        let mut captured = false;
        unsafe {
            execute_submission_plan_on(std::ptr::null_mut(), &plan, |chunk| captured = chunk.uses_shared_view_heap());
        }
        assert!(captured, "the plan uses the physical PC list mapped at queue submission");

        on_release(pc as usize);
        assert!(!has_list_state(pc as usize), "the physical list's final Release removes its state");
    }

    #[test]
    fn clearing_a_reused_segment_clears_shared_view_use() {
        let list = 0x7ffe_1a01usize;
        LISTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(
            list,
            ListState { uses_shared_view_heap: true, ..Default::default() },
        );
        clear_segment_state(list);

        let mut g = LISTS.lock().unwrap();
        assert!(!g.as_ref().unwrap()[&list].uses_shared_view_heap);
        g.as_mut().unwrap().remove(&list);
    }

    #[test]
    fn kickoff_moves_shared_view_use_to_the_finished_segment() {
        let mut state = ListState { uses_shared_view_heap: true, ..Default::default() };
        let effects = take_kickoff_effects(&mut state);
        assert!(effects.uses_shared_view_heap);
        assert!(!state.uses_shared_view_heap, "the continuing recording starts a new segment");
        assert!(!take_kickoff_effects(&mut state).uses_shared_view_heap);
    }

    #[test]
    fn execute_bundle_use_survives_bundle_reset() {
        let (parent, bundle) = (0x7ffe_1b01usize, 0x7ffe_1b02usize);
        let mut states = HashMap::from([
            (parent, ListState::default()),
            (bundle, ListState { uses_shared_view_heap: true, ..Default::default() }),
        ]);
        propagate_bundle_shared_view_use(&mut states, parent, bundle);
        states.get_mut(&bundle).unwrap().uses_shared_view_heap = false;
        assert!(states[&parent].uses_shared_view_heap);

        propagate_bundle_shared_view_use(&mut states, parent, bundle);
        assert!(states[&parent].uses_shared_view_heap, "executing an unused bundle cannot clear prior use");
    }

    #[test]
    fn queue_identity_changes_when_a_raw_address_is_reused() {
        let raw = 0x7ffe_1801usize;
        on_queue_create(raw, 0x2000, D3D12_COMMAND_LIST_TYPE_DIRECT);
        let first = queue_identity(raw).unwrap();
        on_queue_release(raw);
        on_queue_create(raw, 0x2000, D3D12_COMMAND_LIST_TYPE_DIRECT);
        let second = queue_identity(raw).unwrap();
        assert_eq!((first.raw, second.raw), (raw, raw));
        assert_ne!(first.generation, second.generation);
        on_queue_release(raw);
    }

    #[test]
    fn submission_plan_is_isolated_from_reset_and_rerecord() {
        let list = 0x7ffe_2001usize as *mut c_void;
        let old_wait = PendingWait { va: 0x4000, width: 8, comparison: 7, value: 4, mask: u64::MAX, after_write64: true };
        {
            let mut g = LISTS.lock().unwrap();
            let mut state = ListState {
                pending_waits: vec![old_wait],
                write64_signals: vec![(0x4000, 4)],
                sparse_writes: vec![(0x5000, 8)],
                texture_writes: vec![super::super::title_textures::GpuWrite { res: 0x5500, generation: 4, subresource: 2 }],
                texture_transitions: vec![super::super::title_textures::Transition {
                    res: 0x5600,
                    generation: 4,
                    subresource: 2,
                    after: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }],
                cpu_writes: vec![super::super::cpu_writes::Write::Immediate(0x6000, vec![4])],
                ..Default::default()
            };
            state.note_write(WriteProducer::InSegmentGpu, 0x4100, 8);
            state.note_write(WriteProducer::CpuAfterSegment, 0x4200, 8);
            g.get_or_insert_with(HashMap::new).insert(list as usize, state);
        }
        let plan = SubmissionPlan::capture(&[list]);

        {
            let mut g = LISTS.lock().unwrap();
            let state = g.as_mut().unwrap().get_mut(&(list as usize)).unwrap();
            state.pending_waits = vec![PendingWait { value: 9, ..old_wait }];
            state.write64_signals = vec![(0x4000, 9)];
            state.sparse_writes = vec![(0x7000, 16)];
            state.texture_writes = vec![super::super::title_textures::GpuWrite { res: 0x7500, generation: 9, subresource: 3 }];
            state.texture_transitions = vec![super::super::title_textures::Transition {
                res: 0x7600,
                generation: 9,
                subresource: 3,
                after: D3D12_RESOURCE_STATE_RENDER_TARGET,
            }];
            state.cpu_writes = vec![super::super::cpu_writes::Write::Immediate(0x8000, vec![9])];
            state.written_ranges.clear();
            state.segment_write_producers.clear();
            state.note_write(WriteProducer::CpuAfterSegment, 0x4300, 8);
        }

        let effects = &plan.effects[0];
        assert_eq!(effects.waits, vec![old_wait]);
        assert_eq!(effects.write64_signals, vec![(0x4000, 4)]);
        assert_eq!(effects.sparse_writes, vec![(0x5000, 8)]);
        assert_eq!(effects.texture_writes, vec![super::super::title_textures::GpuWrite { res: 0x5500, generation: 4, subresource: 2 }]);
        assert_eq!(effects.texture_transitions, vec![super::super::title_textures::Transition {
            res: 0x5600,
            generation: 4,
            subresource: 2,
            after: D3D12_RESOURCE_STATE_COPY_SOURCE,
        }]);
        assert!(effects.producer_overlaps(WriteProducer::InSegmentGpu, 0x4100, 8));
        assert!(effects.producer_overlaps(WriteProducer::CpuAfterSegment, 0x4200, 8));
        assert!(!effects.producer_overlaps(WriteProducer::CpuAfterSegment, 0x4300, 8));
        match effects.cpu_writes.as_slice() {
            [super::super::cpu_writes::Write::Immediate(va, words)] => assert_eq!((*va, words.as_slice()), (0x6000, &[4][..])),
            _ => panic!("unexpected CPU write snapshot"),
        }
        LISTS.lock().unwrap().as_mut().unwrap().remove(&(list as usize));
    }

    #[test]
    fn submission_plan_prepares_prior_cpu_write_dependencies() {
        let lists = [0x7ffe_2401usize as *mut c_void, 0x7ffe_2402usize as *mut c_void, 0x7ffe_2403usize as *mut c_void, 0x7ffe_2404usize as *mut c_void];
        let wait = |va| PendingWait { va, width: 8, comparison: 7, value: 1, mask: u64::MAX, after_write64: false };
        let mut states: Vec<ListState> = (0..4).map(|_| ListState::default()).collect();
        states[0].note_write(WriteProducer::CpuAfterSegment, 0x1000, 8);
        states[1].pending_waits.push(wait(0x1000));
        states[2].note_write(WriteProducer::CpuAfterSegment, 0x2000, 8);
        states[2].pending_waits.push(wait(0x2000));
        states[3].pending_waits.push(wait(0x2000));
        {
            let mut g = LISTS.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            for (&list, state) in lists.iter().zip(states) {
                m.insert(list as usize, state);
            }
        }
        let plan = SubmissionPlan::capture(&lists);
        assert_eq!(plan.waits_after_cpu_write, vec![false, true, false, true], "same-segment producers must not satisfy their own wait");
        let mut g = LISTS.lock().unwrap();
        for list in lists {
            g.as_mut().unwrap().remove(&(list as usize));
        }
    }

    #[test]
    fn empty_submission_plan_uses_the_lock_free_fast_path() {
        let lists_guard = LISTS.lock().unwrap();
        let plan = SubmissionPlan::capture(&[]);
        drop(lists_guard);
        assert!(plan.lists.is_empty() && plan.effects.is_empty() && plan.waits_after_cpu_write.is_empty() && !plan.has_waits);

        let mut submitted = 0;
        unsafe {
            execute_submission_plan_on(std::ptr::null_mut(), &plan, |chunk| {
                assert!(chunk.lists().is_empty());
                submitted += 1;
            });
        }
        assert_eq!(submitted, 1);
    }

    #[test]
    fn nonempty_no_wait_plan_loop_does_not_index_empty_dependencies() {
        let list = 0x7ffe_2601usize as *mut c_void;
        let mut plan = SubmissionPlan::capture(&[list]);
        assert!(plan.waits_after_cpu_write.is_empty());
        // Exercise the loop independently of capture's no-wait early-return flag.
        plan.has_waits = true;
        let mut submitted = Vec::new();
        unsafe {
            execute_submission_plan_on(std::ptr::null_mut(), &plan, |chunk| submitted.push(chunk.lists().to_vec()));
        }
        assert_eq!(submitted, vec![vec![list]]);
    }

    #[test]
    fn cpu_producer_completion_precedes_the_continuation_wait() {
        let producer = 0x7ffe_2801usize as *mut c_void;
        let continuation = 0x7ffe_2802usize as *mut c_void;
        let cell = Box::new(AtomicU64::new(1));
        let va = &*cell as *const AtomicU64 as u64;
        {
            let mut g = LISTS.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            let mut first = ListState::default();
            first.note_write(WriteProducer::CpuAfterSegment, va, 8);
            first.texture_transitions.push(super::super::title_textures::Transition {
                res: 0x8100,
                generation: 1,
                subresource: 0,
                after: D3D12_RESOURCE_STATE_COPY_SOURCE,
            });
            m.insert(producer as usize, first);
            m.insert(
                continuation as usize,
                ListState {
                    pending_waits: vec![PendingWait {
                        va,
                        width: 8,
                        comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
                        value: 1,
                        mask: u64::MAX,
                        after_write64: false,
                    }],
                    texture_transitions: vec![super::super::title_textures::Transition {
                        res: 0x8200,
                        generation: 1,
                        subresource: 0,
                        after: D3D12_RESOURCE_STATE_COPY_DEST,
                    }],
                    ..Default::default()
                },
            );
        }
        let plan = SubmissionPlan::capture(&[producer, continuation]);
        {
            let mut g = LISTS.lock().unwrap();
            let m = g.as_mut().unwrap();
            m.remove(&(producer as usize));
            m.remove(&(continuation as usize));
        }
        assert!(plan.effects[0].producer_overlaps(WriteProducer::CpuAfterSegment, va, 8));
        assert!(!plan.effects[1].producer_overlaps(WriteProducer::CpuAfterSegment, va, 8));
        assert_eq!(plan.waits_after_cpu_write, vec![false, true]);
        assert!(plan.chunk_needs_cpu_completion(0, 1, &[]));
        assert!(!plan.chunk_needs_cpu_completion(1, 2, &[]));
        let unrelated = PendingWait { va: va + 8, width: 8, comparison: 7, value: 1, mask: u64::MAX, after_write64: false };
        assert!(!SubmissionPlan {
            lists: vec![producer],
            effects: vec![plan.effects[0].clone()],
            traces: Vec::new(),
            waits_after_cpu_write: Vec::new(),
            has_waits: false,
        }
        .chunk_needs_cpu_completion(0, 1, &[unrelated]));

        let completed = Arc::new(AtomicBool::new(false));
        let mut chunks = Vec::new();
        let mut transition_chunks = Vec::new();
        unsafe {
            let leftovers = execute_submission_plan_with_completions(std::ptr::null_mut(), &plan, &[], |chunk| {
                let lists = chunk.lists().to_vec();
                chunks.push(lists.clone());
                transition_chunks.push(chunk.effects.iter().flat_map(|e| &e.texture_transitions).map(|t| t.res).collect::<Vec<_>>());
                if lists == vec![producer] {
                    assert!(chunk.needs_cpu_completion());
                    let (receipt, done) = super::super::cpu_writes::completion_pair_for_test();
                    let completed = completed.clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(20));
                        completed.store(true, Ordering::Release);
                        let _ = done.send(());
                    });
                    (true, Some(receipt))
                } else {
                    assert!(!chunk.needs_cpu_completion());
                    assert!(completed.load(Ordering::Acquire), "the CPU producer must complete before the wait segment is submitted");
                    (true, None)
                }
            }).0;
            assert!(leftovers.is_empty());
        }
        assert_eq!(chunks, vec![vec![producer], vec![continuation]]);
        assert_eq!(transition_chunks, vec![vec![0x8100], vec![0x8200]], "wait splitting keeps transition effects with their physical segment");

        let mut callbacks = 0;
        let completed_all = unsafe {
            execute_submission_plan_with_completions(std::ptr::null_mut(), &plan, &[], |_| {
                callbacks += 1;
                (false, None)
            })
            .1
        };
        assert!(!completed_all);
        assert_eq!(callbacks, 1, "a failed first chunk suppresses later waits and submissions");
    }

    #[test]
    fn segment_reasons_keep_the_existing_segment_costs() {
        assert_eq!(SegmentReason::WaitAfterWrite64.extra_segments(), 1);
        assert_eq!(SegmentReason::WaitAfterCpuWrite.extra_segments(), 1);
        let deferred = SegmentReason::DeferredDispatch(Deferred::Dispatch(1, 2, 3, 0x8020));
        let SegmentReason::DeferredDispatch(Deferred::Dispatch(x, y, z, flags)) = &deferred else { unreachable!() };
        assert_eq!((*x, *y, *z, *flags), (1, 2, 3, 0x8020), "deferred replay retains DispatchX flags");
        assert_eq!(deferred.extra_segments(), 2);
    }

    #[test]
    fn bounded_wait_skips_defer_and_unbounded_demand_precedes_split() {
        let demanded = std::cell::Cell::new(0);
        let mut splits = 0;
        assert!(!defer_pipeline_wait(true, || demanded.set(demanded.get() + 1), || { splits += 1; true }));
        assert_eq!(demanded.get(), 0);
        assert_eq!(splits, 0);
        assert!(defer_pipeline_wait(false, || demanded.set(demanded.get() + 1), || {
            assert_eq!(demanded.get(), 1, "pipeline demand must be visible before split preparation");
            splits += 1;
            true
        }));
        assert_eq!(demanded.get(), 1);
        assert_eq!(splits, 1, "the default still defers to the submit");
        assert!(!exact_restoration_waits_bounded(true),
            "XCL_PSO_WAIT_MS must not turn mandatory exact restoration into a skipped dispatch");
    }

    #[test]
    fn failed_split_preflight_does_not_consume_recording_state() {
        let query = (0x7000, 2);
        let mut state = ListState { pred_in_state: true, ..Default::default() };
        assert!(state.occlusion.begin(query, D3D12_QUERY_TYPE_OCCLUSION).is_none());
        assert!(SplitCarry::capture(&state).is_none());
        assert!(state.pred_in_state);
        assert!(state.occlusion.active.is_some());
        assert!(state.occlusion.ended_early.is_empty());
    }

    #[test]
    fn reserved_closed_standby_cannot_kickoff_stale_recording_state() {
        let list = 0x7ffe_2a01usize;
        let wait = PendingWait { va: 0x4000, width: 8, comparison: 7, value: 1, mask: u64::MAX, after_write64: false };
        let mut state = ListState {
            allocator: 0xa110,
            list_type: Some(D3D12_COMMAND_LIST_TYPE_COMPUTE),
            pending_waits: vec![wait],
            texture_writes: vec![super::super::title_textures::GpuWrite { res: 0x4800, generation: 2, subresource: 0 }],
            cpu_writes: vec![super::super::cpu_writes::Write::Immediate(0x5000, vec![1])],
            ..Default::default()
        };
        state.segment_write_producers.insert(WriteProducer::CpuAfterSegment, 0x5000, 4);
        LISTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(list, state);

        note_reserved_closed(list, D3D12_COMMAND_LIST_TYPE_DIRECT);

        let mut g = LISTS.lock().unwrap();
        let state = &g.as_ref().unwrap()[&list];
        assert_eq!(state.allocator, 0, "KickoffX must take its allocator-unknown no-op path");
        assert_eq!(state.list_type, Some(D3D12_COMMAND_LIST_TYPE_DIRECT));
        assert!(state.pending_waits.is_empty() && state.texture_writes.is_empty() && state.cpu_writes.is_empty() && state.segment_write_producers.is_empty());
        g.as_mut().unwrap().remove(&list);
    }

    #[test]
    fn close_failure_keeps_effects_and_query_bookkeeping_on_the_producer() {
        let query = (0x7100, 3);
        let wait = PendingWait { va: 0xd440, width: 8, comparison: 7, value: 5, mask: u64::MAX, after_write64: false };
        let mut state = ListState { allocator: 0xa110, pending_waits: vec![wait], ..Default::default() };
        state.written_ranges.insert(WriteProducer::CpuAfterSegment, 0x3000, 8);
        assert!(state.occlusion.begin(query, D3D12_QUERY_TYPE_OCCLUSION).is_none());
        let carry = SplitCarry::capture(&state).unwrap();
        carry.commit_state(&mut state);
        assert!(state.occlusion.active.is_none());
        assert_eq!(state.occlusion.ended_early, vec![query]);
        assert_eq!(state.pending_waits.len(), 1);
        assert!(state.written_ranges.overlaps(WriteProducer::CpuAfterSegment, 0x3000, 8));
    }

    #[test]
    fn wrapper_kickoff_defers_effect_consumption_until_close_succeeds() {
        let wait = PendingWait { va: 0x4000, width: 8, comparison: 7, value: 1, mask: u64::MAX, after_write64: false };
        let mut state = ListState {
            pending_waits: vec![wait],
            texture_writes: vec![super::super::title_textures::GpuWrite { res: 0x4800, generation: 3, subresource: 1 }],
            cpu_writes: vec![super::super::cpu_writes::Write::Immediate(0x5000, vec![1])],
            ..Default::default()
        };
        state.note_write(WriteProducer::CpuAfterSegment, 0x5000, 4);

        assert!(take_kickoff_effects_before_close(&mut state, true).is_none());
        assert_eq!(state.pending_waits, vec![wait]);
        assert_eq!(state.texture_writes.len(), 1);
        assert_eq!(state.cpu_writes.len(), 1);
        assert!(state.written_ranges.overlaps(WriteProducer::CpuAfterSegment, 0x5000, 4));
        assert!(state.segment_write_producers.overlaps(WriteProducer::CpuAfterSegment, 0x5000, 4));

        let effects = take_kickoff_effects_before_close(&mut state, false).unwrap();
        assert_eq!(effects.waits, vec![wait]);
        assert_eq!(effects.texture_writes.len(), 1);
        assert_eq!(effects.cpu_writes.len(), 1);
        assert!(state.pending_waits.is_empty() && state.texture_writes.is_empty() && state.cpu_writes.is_empty());
        assert!(state.written_ranges.is_empty() && state.segment_write_producers.is_empty());
    }

    #[test]
    fn kickoff_standby_ownership_settles_only_after_target_reset() {
        use std::cell::Cell;
        let returned = Cell::new(None);
        let failed = Cell::new(None);
        settle_kickoff_standby(
            Some((0x7100usize, 0x7200usize)),
            true,
            |w, s| returned.set(Some((w, s))),
            |w, s, producer_broken| failed.set(Some((w, s, producer_broken))),
        );
        assert_eq!(returned.get(), Some((0x7100, 0x7200)));
        assert_eq!(failed.get(), None);

        returned.set(None);
        settle_kickoff_standby(
            Some((0x7300usize, 0x7400usize)),
            false,
            |w, s| returned.set(Some((w, s))),
            |w, s, producer_broken| failed.set(Some((w, s, producer_broken))),
        );
        assert_eq!(returned.get(), None);
        assert_eq!(failed.get(), Some((0x7300, 0x7400, false)));

        failed.set(None);
        settle_kickoff_standby(
            None::<(usize, usize)>,
            false,
            |w, s| returned.set(Some((w, s))),
            |w, s, producer_broken| failed.set(Some((w, s, producer_broken))),
        );
        assert_eq!(returned.get(), None);
        assert_eq!(failed.get(), None);
    }

    #[test]
    fn split_carry_recreates_the_continuation_state() {
        let query = (0x7100, 3);
        let earlier_query = (0x7000, 2);
        let mut state = ListState {
            allocator: 0xa110,
            list_type: Some(D3D12_COMMAND_LIST_TYPE_DIRECT),
            last_queue: Some(0x9000),
            bound_pso: 0xb220,
            depth_bias: Some(DepthBias::new(-2.5, 0.25, 4.0)),
            native_bias: DepthBias::new(-1.0, 1.0, 1.0),
            pso_dirty: true,
            using_fallback_pso: true,
            index_buffer: Some(D3D12_INDEX_BUFFER_VIEW { BufferLocation: 0xc330, SizeInBytes: 24, Format: DXGI_FORMAT_R32_UINT }),
            depth_bounds: Some((0.2, 0.8)),
            pred_in_state: true,
            predication: Some(D3D12_PREDICATION_OP_NOT_EQUAL_ZERO),
            pending_waits: vec![PendingWait { va: 0xd440, width: 8, comparison: 7, value: 5, mask: u64::MAX, after_write64: false }],
            write64_signals: vec![(0xe550, 6)],
            sparse_writes: vec![(0xf660, 8)],
            texture_writes: vec![super::super::title_textures::GpuWrite { res: 0xf770, generation: 5, subresource: 0 }],
            bindings: Bindings { heaps: vec![0x1010, 0x2020], topology: Some(4), stencil: Some(9), ..Default::default() },
            ..Default::default()
        };
        state.note_write(WriteProducer::InSegmentGpu, 0x3000, 0x20);
        state.occlusion.ended_early.push(earlier_query);
        assert!(state.occlusion.begin(query, D3D12_QUERY_TYPE_OCCLUSION).is_none());

        let carry = SplitCarry::capture(&state).unwrap();
        assert!(state.pred_in_state && state.occlusion.active.is_some(), "preflight must not mutate the producer");
        assert_eq!(state.occlusion.ended_early, vec![earlier_query]);
        carry.commit_state(&mut state);
        assert!(!state.pred_in_state && state.occlusion.active.is_none());
        assert_eq!(state.occlusion.ended_early, vec![earlier_query, query]);
        state.bindings.heaps.clear();
        state.written_ranges.clear();
        let continued = carry.continuation_state(D3D12_COMMAND_LIST_TYPE_DIRECT);

        assert_eq!(carry.active_query, Some((query, D3D12_QUERY_TYPE_OCCLUSION)));
        assert_eq!(continued.allocator, 0xa110);
        assert_eq!(continued.last_queue, Some(0x9000));
        assert_eq!(continued.bound_pso, 0xb220);
        assert_eq!(continued.depth_bias, Some(DepthBias::new(-2.5, 0.25, 4.0)));
        assert_eq!(continued.native_bias, DepthBias::ZERO);
        assert!(!continued.pso_dirty && !continued.using_fallback_pso);
        assert_eq!(continued.index_buffer.as_ref().unwrap().BufferLocation, 0xc330);
        assert_eq!(continued.depth_bounds, Some((0.2, 0.8)));
        assert_eq!(continued.predication, Some(D3D12_PREDICATION_OP_NOT_EQUAL_ZERO));
        assert!(!continued.pred_in_state);
        assert_eq!(continued.occlusion.ended_early, vec![earlier_query, query]);
        assert_eq!(continued.bindings.heaps, vec![0x1010, 0x2020]);
        assert_eq!(continued.bindings.topology, Some(4));
        assert_eq!(continued.bindings.stencil, Some(9));
        assert!(continued.written_ranges.overlaps(WriteProducer::InSegmentGpu, 0x3010, 4));
        assert!(continued.segment_write_producers.is_empty(), "a continuation has not produced the carried ranges");
        assert!(continued.pending_waits.is_empty() && continued.write64_signals.is_empty() && continued.sparse_writes.is_empty() && continued.texture_writes.is_empty());
        assert!(continued.cpu_writes.is_empty() && continued.deferred.is_none() && continued.own_alloc.is_none());
        assert_eq!(state.pending_waits.len(), 1, "effects remain on the closed segment");
        assert_eq!(state.write64_signals, vec![(0xe550, 6)]);
        assert_eq!(state.sparse_writes, vec![(0xf660, 8)]);
        assert_eq!(state.texture_writes.len(), 1);
        assert!(state.segment_write_producers.overlaps(WriteProducer::InSegmentGpu, 0x3010, 4));
    }

    #[test]
    fn execute_with_waits_splits_batches_before_unsatisfied_list() {
        // Use synthetic non-zero list pointers distinct from real COM objects.
        let l0 = 0x7000_0001usize as *mut c_void;
        let l1 = 0x7000_0002usize as *mut c_void;
        let l2 = 0x7000_0003usize as *mut c_void;
        let cell = Box::new(AtomicU32::new(0));
        let va = &*cell as *const AtomicU32 as u64;

        unsafe {
            on_create(l0, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(l1, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(l2, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
        }
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&(l1 as usize)).unwrap();
            s.push_wait(PendingWait {
                va,
                width: 4,
                comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
                value: 42,
                mask: u32::MAX as u64,
                after_write64: false,
            });
        }

        let mut batches: Vec<Vec<usize>> = Vec::new();
        unsafe {
            execute_with_waits(&[l0, l1, l2], |chunk| {
                batches.push(chunk.iter().map(|&p| p as usize).collect());
                // Simulate l0's GPU execution satisfying l1's wait!
                if chunk.contains(&l0) {
                    cell.store(42, Ordering::Release);
                }
            });
        }
        assert_eq!(batches, vec![vec![l0 as usize], vec![l1 as usize, l2 as usize]]);

        // Now that cell == 42 is already satisfied, the whole batch submits in one call.
        batches.clear();
        unsafe {
            execute_with_waits(&[l0, l1, l2], |chunk| {
                batches.push(chunk.iter().map(|&p| p as usize).collect());
            });
        }
        assert_eq!(batches, vec![vec![l0 as usize, l1 as usize, l2 as usize]]);

        on_release(l0 as usize);
        on_release(l1 as usize);
        on_release(l2 as usize);
    }

    #[test]
    fn queue_picking_prefers_last_queue_and_respects_release() {
        let dev = 0x8000_0001usize;
        let q1 = 0x8000_0010usize;
        let q2 = 0x8000_0020usize;
        let q_copy = 0x8000_0030usize;
        on_queue_create(q1, dev, D3D12_COMMAND_LIST_TYPE_DIRECT);
        on_queue_create(q2, dev, D3D12_COMMAND_LIST_TYPE_DIRECT);
        on_queue_create(q_copy, dev, D3D12_COMMAND_LIST_TYPE_COPY);

        // Most recently created matching queue wins when no last_queue
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q2));
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_COPY), Some(q_copy));
        // Explicit last_queue wins when still live
        assert_eq!(pick_queue(Some(q1), dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q1));
        // Releasing q1 falls back to q2
        on_queue_release(q1);
        assert_eq!(pick_queue(Some(q1), dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q2));
        on_queue_release(q2);
        on_queue_release(q_copy);
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_DIRECT), None);
    }

    #[test]
    fn sync_graph_forgets_every_reference_to_a_retired_queue() {
        let (gone, kept) = (0x9100usize, 0x9200usize);
        let mut graph = SyncGraph::default();
        graph.queue_seq.insert(gone, 3);
        graph.queue_seq.insert(kept, 4);
        let mut cell = SyncCell { recorded_last_queue: Some(gone), submitted_queue: Some(gone), ..Default::default() };
        cell.armed_waits.push(ArmedGpuWait {
            val: 1,
            waiting_queue: gone,
            waiting_queue_ty: 0,
            wait_seq: 1,
            armed_at: Instant::now(),
            after_write64: false,
        });
        cell.armed_waits.push(ArmedGpuWait { waiting_queue: kept, ..cell.armed_waits[0] });
        cell.submitted_signals.push(SubmittedSignal { val: 1, queue: gone, seq: 1 });
        cell.submitted_signals.push(SubmittedSignal { val: 2, queue: kept, seq: 2 });
        graph.cells.insert(0x1000, cell);
        graph.title_fences.insert(
            0x2000,
            TitleFenceSync {
                waits: vec![(1, gone, 1), (2, kept, 2)],
                signals: vec![SubmittedSignal { val: 1, queue: gone, seq: 1 }, SubmittedSignal { val: 2, queue: kept, seq: 2 }],
            },
        );

        assert_eq!(graph.forget_queue(gone), 1);
        assert!(!graph.queue_seq.contains_key(&gone) && graph.queue_seq.contains_key(&kept));
        let cell = &graph.cells[&0x1000];
        assert_eq!((cell.recorded_last_queue, cell.submitted_queue), (None, None));
        assert_eq!(cell.armed_waits.iter().map(|wait| wait.waiting_queue).collect::<Vec<_>>(), [kept]);
        assert_eq!(cell.submitted_signals.iter().map(|signal| signal.queue).collect::<Vec<_>>(), [kept]);
        let title = &graph.title_fences[&0x2000];
        assert_eq!(title.waits.iter().map(|&(_, queue, _)| queue).collect::<Vec<_>>(), [kept]);
        assert_eq!(title.signals.iter().map(|signal| signal.queue).collect::<Vec<_>>(), [kept]);
    }

    #[test]
    fn depth_bias_state_tracks_fallback_and_clear() {
        let l = 0x9000_0001usize as *mut c_void;
        let key = l as usize;
        unsafe {
            on_create(l, 0x100, 0xAAA, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_set_pipeline_state(l, 0xBBB);
        }
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&key).unwrap();
            assert_eq!(s.bound_pso, 0xBBB);
            assert!(!s.pso_dirty);
            s.depth_bias = Some(DepthBias::new(-10.0, 0.0, 1.0));
            s.pso_dirty = true;
        }
        // Switching to a non-dynamic PSO while depth bias is active keeps pso_dirty for fallback
        unsafe { on_set_pipeline_state(l, 0xCCC) };
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!(s.bound_pso, 0xCCC);
            assert!(s.pso_dirty);
        }
        // ClearState resets depth bias, native_bias, and dirty flag
        unsafe { on_clear_state(l, 0xDDD) };
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!(s.bound_pso, 0xDDD);
            assert!(s.depth_bias.is_none());
            assert_eq!(s.native_bias, DepthBias::ZERO);
            assert!(!s.pso_dirty);
        }
        on_release(key);
    }

    #[test]
    fn depth_bias_native_list9_and_bundle_fallback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let supported = device_x::probe_dynamic_depth_bias(&dev);
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            let has_list9 = list.cast::<ID3D12GraphicsCommandList9>().is_ok();
            let raw = list.as_raw();
            let key = raw as usize;

            let pso_dyn_zero = 0x9100_0001usize;
            let pso_dyn_static = 0x9100_0002usize;
            let pso_static_only = 0x9100_0003usize;
            let static_bias = DepthBias::new(-500.0, 0.0, 0.0);
            device_x::register_test_base(pso_dyn_zero, true, DepthBias::ZERO);
            device_x::register_test_base(pso_dyn_static, true, static_bias);
            device_x::register_test_base(pso_static_only, false, DepthBias::ZERO);

            let cache_before = device_x::depth_bias_cache_len();
            on_create(raw, alloc.as_raw() as usize, pso_dyn_zero, D3D12_COMMAND_LIST_TYPE_DIRECT);

            if supported && has_list9 {
                // Fractional f32 depth bias takes the native ID3D12GraphicsCommandList9 path
                // without rounding to i32 or dirtying the patched-PSO fallback cache.
                set_depth_bias_x(raw, -125.75, 0.05, 1.25);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.depth_bias, Some(DepthBias::new(-125.75, 0.05, 1.25)));
                    assert_eq!(s.native_bias, DepthBias::new(-125.75, 0.05, 1.25));
                    assert_eq!(s.native_bias.depth_bias(), -125.75);
                    assert!(!s.pso_dirty);
                }
                assert_eq!(device_x::depth_bias_cache_len(), cache_before);

                // Resetting depth bias to 0 updates native_bias to ZERO without touching fallback cache.
                set_depth_bias_x(raw, 0.0, 0.0, 0.0);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert!(s.depth_bias.is_none());
                    assert_eq!(s.native_bias, DepthBias::ZERO);
                    assert!(!s.pso_dirty);
                }

                // Binding a dynamic PSO with non-zero static bias syncs native_bias to its static bias.
                on_set_pipeline_state(raw, pso_dyn_static);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.native_bias, static_bias);
                    assert!(!s.pso_dirty);
                }

                // Switching back to a zero-static-bias dynamic PSO restores native_bias to ZERO.
                on_set_pipeline_state(raw, pso_dyn_zero);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.native_bias, DepthBias::ZERO);
                    assert!(!s.pso_dirty);
                }
            }

            // Non-dynamic PSO must fall back to the patched-PSO cache (pso_dirty = true).
            on_set_pipeline_state(raw, pso_static_only);
            set_depth_bias_x(raw, -50.0, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert_eq!(s.depth_bias, Some(DepthBias::new(-50.0, 0.0, 0.0)));
                assert!(s.pso_dirty);
            }
            on_release(key);

            // Bundle lists cannot call ID3D12GraphicsCommandList9::RSSetDepthBias and must fall back.
            let b_raw = 0x9100_00F0usize as *mut c_void;
            on_create(b_raw, 0x100, pso_dyn_zero, D3D12_COMMAND_LIST_TYPE_BUNDLE);
            set_depth_bias_x(b_raw, -75.0, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&(b_raw as usize)).unwrap();
                assert!(s.pso_dirty);
                assert_eq!(s.native_bias, DepthBias::ZERO);
            }
            on_release(b_raw as usize);

            device_x::forget_base(pso_dyn_zero);
            device_x::forget_base(pso_dyn_static);
            device_x::forget_base(pso_static_only);
        }
    }

    #[test]
    fn depth_bias_native_path_exact_f32_and_list9_fallback() {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Verify PC vtable slot of ID3D12GraphicsCommandList9::RSSetDepthBias is in layout table.
        let cl9 = super::super::layout::find("ID3D12GraphicsCommandList9").expect("ID3D12GraphicsCommandList9 in layout");
        assert_eq!(super::super::layout::slot(cl9, "RSSetDepthBias"), Some(82));

        let calls: Rc<RefCell<Vec<(usize, f32, f32, f32)>>> = Rc::new(RefCell::new(Vec::new()));
        let allow = Rc::new(std::cell::Cell::new(true));
        let calls_cb = calls.clone();
        let allow_cb = allow.clone();
        TEST_NATIVE_BIAS.with(|h| {
            *h.borrow_mut() = Some(Box::new(move |list, bias| {
                if !allow_cb.get() {
                    return false;
                }
                calls_cb.borrow_mut().push((list, bias.depth_bias(), bias.depth_bias_clamp(), bias.slope_scaled_depth_bias()));
                true
            }));
        });

        let l = 0x9200_0001usize as *mut c_void;
        let key = l as usize;
        let pso_dyn_zero = 0x9200_0010usize;
        let pso_dyn_static = 0x9200_0011usize;
        let pso_static_only = 0x9200_0012usize;
        device_x::register_test_base(pso_dyn_zero, true, DepthBias::ZERO);
        device_x::register_test_base(pso_dyn_static, true, DepthBias::new(-500.0, 0.25, 2.0));
        device_x::register_test_base(pso_static_only, false, DepthBias::ZERO);

        let cache_before = device_x::depth_bias_cache_len();
        unsafe {
            // 1. Initial PSO with baked static bias seeds native RSSetDepthBias on CreateCommandList.
            on_create(l, 0x100, pso_dyn_static, D3D12_COMMAND_LIST_TYPE_DIRECT);
            assert_eq!(*calls.borrow(), vec![(key, -500.0, 0.25, 2.0)]);

            // 2. Switching to zero-static-bias dynamic PSO resets native RSSetDepthBias to (0, 0, 0).
            on_set_pipeline_state(l, pso_dyn_zero);
            assert_eq!(calls.borrow().last(), Some(&(key, 0.0, 0.0, 0.0)));

            // 3. Fractional f32 override is forwarded verbatim (no i32 rounding, no fallback PSO cache).
            set_depth_bias_x(l, -1000.5, 0.05, 1.5);
            assert_eq!(calls.borrow().last(), Some(&(key, -1000.5, 0.05, 1.5)));
            assert_eq!(device_x::depth_bias_cache_len(), cache_before);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(!s.pso_dirty);
                assert_eq!(s.native_bias, DepthBias::new(-1000.5, 0.05, 1.5));
            }

            // 4. Switching between dynamic PSOs while override is active keeps the override without redundant calls.
            let n_before = calls.borrow().len();
            on_set_pipeline_state(l, pso_dyn_static);
            assert_eq!(calls.borrow().len(), n_before);

            // 5. Resetting to (0, 0, 0) sets native bias to (0, 0, 0); ClearState with static-bias PSO re-seeds it.
            set_depth_bias_x(l, 0.0, 0.0, 0.0);
            assert_eq!(calls.borrow().last(), Some(&(key, 0.0, 0.0, 0.0)));
            on_clear_state(l, pso_dyn_static);
            assert_eq!(calls.borrow().last(), Some(&(key, -500.0, 0.25, 2.0)));

            // 6. If ID3D12GraphicsCommandList9 is unavailable, set_depth_bias_x falls back cleanly to pso_dirty.
            on_set_pipeline_state(l, pso_dyn_zero);
            allow.set(false);
            set_depth_bias_x(l, -42.5, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(s.pso_dirty);
                assert_eq!(s.depth_bias, Some(DepthBias::new(-42.5, 0.0, 0.0)));
            }

            // 7. Non-dynamic PSO always uses the patched-PSO fallback and never invokes RSSetDepthBias.
            allow.set(true);
            let n_before = calls.borrow().len();
            on_set_pipeline_state(l, pso_static_only);
            set_depth_bias_x(l, -10.0, 0.0, 0.0);
            assert_eq!(calls.borrow().len(), n_before);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(s.pso_dirty);
            }
        }

        TEST_NATIVE_BIAS.with(|h| *h.borrow_mut() = None);
        on_release(key);
        device_x::forget_base(pso_dyn_zero);
        device_x::forget_base(pso_dyn_static);
        device_x::forget_base(pso_static_only);
    }

    #[test]
    fn resolve_query_data_applies_buffer_delta_to_pc_resolve_and_occlusion_writeback() {
        use std::cell::RefCell;

        thread_local! {
            static RESOLVED: RefCell<Vec<(usize, u32, u32, usize, u64)>> = const { RefCell::new(Vec::new()) };
            static WRITES: RefCell<Vec<(u64, u32)>> = const { RefCell::new(Vec::new()) };
        }

        unsafe extern "system" fn fake_resolve(
            _this: *mut c_void,
            heap: usize,
            _ty: D3D12_QUERY_TYPE,
            first: u32,
            count: u32,
            dest: usize,
            offset: u64,
        ) {
            RESOLVED.with(|r| r.borrow_mut().push((heap, first, count, dest, offset)));
        }

        unsafe extern "system" fn fake_write_immediate(
            _this: *mut c_void,
            count: u32,
            params: *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER,
            _modes: *const D3D12_WRITEBUFFERIMMEDIATE_MODE,
        ) {
            let slice = std::slice::from_raw_parts(params, count as usize);
            WRITES.with(|w| {
                let mut w = w.borrow_mut();
                for p in slice {
                    w.push((p.Dest, p.Value));
                }
            });
        }

        const BASE_GPU_VA: u64 = 0x1_0000_0000;
        unsafe extern "system" fn fake_gpu_va(_this: *mut c_void) -> u64 {
            BASE_GPU_VA
        }

        #[repr(C)]
        struct FakeCom {
            vt: *const usize,
        }

        let mut orig_list_vt = [0usize; 70];
        orig_list_vt[PC_RESOLVE_QUERY_DATA] = fake_resolve as usize;
        orig_list_vt[PC_WRITE_BUFFER_IMMEDIATE] = fake_write_immediate as usize;
        // Adopted objects store `orig` at `ext[-1]`.
        let ext_list_vt = [orig_list_vt.as_ptr() as usize, 0usize];
        let mut fake_list = FakeCom {
            // SAFETY: `ext_list_vt` has 2 elements; `.add(1)` points at `ext[0]` with `ext[-1] = orig`.
            vt: unsafe { ext_list_vt.as_ptr().add(1) },
        };

        let mut res_vt = [0usize; 16];
        res_vt[11] = fake_gpu_va as usize;
        let mut fake_res = FakeCom { vt: res_vt.as_ptr() };

        let heap = 0x5555_0001usize;
        let dest = &mut fake_res as *mut FakeCom as usize;
        let delta = 0x3400u64;
        let title_offset = 0x100u64;

        real::set_test_buffer_delta(dest, Some(delta));
        VISIBLE.lock().unwrap().get_or_insert_with(HashMap::new).insert((heap, 2), VISIBLE_SAMPLES);
        RESOLVED.with(|r| r.borrow_mut().clear());
        WRITES.with(|w| w.borrow_mut().clear());

        unsafe {
            resolve_query_data_x(
                &mut fake_list as *mut FakeCom as *mut c_void,
                heap,
                D3D12_QUERY_TYPE_OCCLUSION,
                1,
                3,
                dest,
                title_offset,
            );
        }

        real::set_test_buffer_delta(dest, None);
        if let Some(m) = VISIBLE.lock().unwrap().as_mut() {
            m.remove(&(heap, 2));
        }

        let expected_off = title_offset + delta;
        RESOLVED.with(|r| {
            assert_eq!(*r.borrow(), vec![(heap, 1, 3, dest, expected_off)]);
        });
        WRITES.with(|w| {
            let expected_va = BASE_GPU_VA + expected_off + 8; // query index 2 = slot (2 - 1) = +8 bytes
            assert_eq!(
                *w.borrow(),
                vec![
                    (expected_va, VISIBLE_SAMPLES as u32),
                    (expected_va + 4, (VISIBLE_SAMPLES >> 32) as u32),
                ]
            );
        });
    }

    /// SetDescriptorHeaps(0, NULL) unbinds every heap: the list must not still count the
    /// shared heap as bound, or a translated draw after it reads the memory windows
    /// (`ResourceDescriptorHeap[0..3]`) from no heap at all.
    #[test]
    fn unbinding_all_heaps_lets_the_shared_heap_be_bound_again() {
        #[repr(C)]
        struct FakeList {
            vt: *const usize,
            calls: Mutex<Vec<Vec<usize>>>,
        }
        unsafe extern "system" fn fake_set_heaps(this: *mut c_void, n: u32, heaps: *const usize) {
            let v = if heaps.is_null() { Vec::new() } else { std::slice::from_raw_parts(heaps, n as usize).to_vec() };
            (*(this as *const FakeList)).calls.lock().unwrap().push(v);
        }
        let mut orig_vt = [0usize; 64];
        orig_vt[28] = fake_set_heaps as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let fake = Box::new(FakeList { vt: unsafe { ext_vt.as_ptr().add(1) }, calls: Mutex::new(Vec::new()) });
        let list = &*fake as *const FakeList as *mut c_void;
        with_bindings(list, |b| b.heaps = vec![0x1234_0000]);
        unsafe {
            set_descriptor_heaps_x(list, 0, std::ptr::null());
            let mut bound = Vec::new();
            with_bindings(list, |b| bound = b.heaps.clone());
            assert!(bound.is_empty(), "no heap is bound after SetDescriptorHeaps(0, NULL)");
            assert_eq!(fake.calls.lock().unwrap().last(), Some(&Vec::new()));
            ensure_view_heap(list);
            with_bindings(list, |b| bound = b.heaps.clone());
            let mut expected = Vec::new();
            if crate::gpu::recomp::pso::heap_windows() {
                if let Some(g) = super::super::console_heap::global_heap() {
                    expected.push(g.as_raw() as usize);
                }
            }
            if let Some(s) = super::super::console_heap::global_sampler_heap() {
                expected.push(s.as_raw() as usize);
            }
            assert_eq!(bound, expected, "the shared heaps are bound again");
            if !expected.is_empty() {
                assert_eq!(fake.calls.lock().unwrap().last(), Some(&expected));
            }
        }
        LISTS.lock().unwrap().as_mut().map(|m| m.remove(&(list as usize)));
    }

    #[test]
    fn draw_indexed_x_routes_through_bind_and_restores_index_buffer() {
        #[repr(C)]
        struct FakeDrawList {
            vt: *const usize,
            refs: AtomicUsize,
            draws: Mutex<Vec<(u32, u32, u32, i32, u32)>>,
            index_buffers: Mutex<Vec<Option<D3D12_INDEX_BUFFER_VIEW>>>,
        }

        unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
            (*(this as *const FakeDrawList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }
        unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
            (*(this as *const FakeDrawList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }
        unsafe extern "system" fn fake_draw_indexed(this: *mut c_void, ic: u32, inst: u32, si: u32, bv: i32, sinst: u32) {
            (*(this as *const FakeDrawList)).draws.lock().unwrap().push((ic, inst, si, bv, sinst));
        }
        unsafe extern "system" fn fake_set_ib(this: *mut c_void, view: *const D3D12_INDEX_BUFFER_VIEW) {
            let v = view.as_ref().copied();
            (*(this as *const FakeDrawList)).index_buffers.lock().unwrap().push(v);
        }
        unsafe extern "system" fn fake_noop_ptr(_this: *mut c_void, _p: *mut c_void) {}
        unsafe extern "system" fn fake_noop_uav(_this: *mut c_void, _slot: u32, _va: u64) {}
        unsafe extern "system" fn fake_set_heaps(_this: *mut c_void, _n: u32, _heaps: *const usize) {}

        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[13] = fake_draw_indexed as usize;
        orig_vt[25] = fake_noop_ptr as usize; // SetPipelineState
        orig_vt[28] = fake_set_heaps as usize; // SetDescriptorHeaps (memory windows from the heap)
        orig_vt[30] = fake_noop_ptr as usize; // SetGraphicsRootSignature
        orig_vt[42] = fake_noop_uav as usize; // SetGraphicsRootUnorderedAccessView
        orig_vt[43] = fake_set_ib as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeDrawList {
            // SAFETY: `ext_vt` has 65 entries so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            draws: Mutex::new(Vec::new()),
            index_buffers: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakeDrawList as *mut c_void;

        // The index buffers' addresses: reserved, never committed, so not readable and never
        // mirrored into unified memory. A fixed address (0x1_0000_1000) was committed memory once
        // earlier tests had allocated there, and the draw then bound its mirror.
        let reserved = unsafe {
            windows_sys::Win32::System::Memory::VirtualAlloc(std::ptr::null(), 0x1_0000, windows_sys::Win32::System::Memory::MEM_RESERVE, windows_sys::Win32::System::Memory::PAGE_NOACCESS)
        } as u64;
        assert_ne!(reserved, 0, "reserve the index buffers' addresses");
        let ib = |k: u64| reserved + k * 0x1000;
        let orig_ib = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: ib(1),
            SizeInBytes: 64,
            Format: DXGI_FORMAT_R16_UINT,
        };

        unsafe {
            ia_set_index_buffer_x(list_ptr, &orig_ib);
            fake.index_buffers.lock().unwrap().clear();

            // 1. Untranslated (PC) PSO: routes through bind::draw_indexed_instanced -> DrawIndexedInstanced,
            // records DRAW_ARGS, binds temp IB, then restores orig_ib.
            super::super::bind::reset(list_ptr, 0x7400_0100, 0);
            // (a draw with a PC pipeline needs the title's root signature on the list)
            super::super::bind::set_graphics_root_signature(list_ptr, 0x7400_0200 as *mut c_void);
            let before = super::super::bind::counts_snapshot();
            draw_indexed_x(list_ptr, ib(2), DXGI_FORMAT_R16_UINT, 6, 0);
            let after = super::super::bind::counts_snapshot();
            assert_eq!(super::super::bind::last_draw_args(), (true, 6, 1, 0, 0));
            assert!(after[2] > before[2], "DRAWS_PC must increment via bind::draw_indexed_instanced");
            assert_eq!(*fake.draws.lock().unwrap(), vec![(6, 1, 0, 0, 0)]);
            let ibs = std::mem::take(&mut *fake.index_buffers.lock().unwrap());
            assert_eq!(ibs.len(), 2);
            assert_eq!(ibs[0].unwrap().BufferLocation, ib(2));
            assert_eq!(ibs[0].unwrap().SizeInBytes, 12);
            assert_eq!(ibs[0].unwrap().Format, DXGI_FORMAT_R16_UINT);
            assert_eq!(ibs[1].unwrap().BufferLocation, orig_ib.BufferLocation);
            assert_eq!(ibs[1].unwrap().SizeInBytes, orig_ib.SizeInBytes);

            // 2. Translated graphics PSO: enters bind::graphics_bound translated path (DRAWS or
            // DRAWS_SKIPPED, NOT DRAWS_PC) instead of bypassing bind.rs directly to PC slot 13.
            let translated_pso = 0x7400_0055usize;
            real::register_translated_graphics_for_test(translated_pso);
            super::super::bind::reset(list_ptr, 0x7400_0100, translated_pso);
            let before = super::super::bind::counts_snapshot();
            draw_indexed_x(list_ptr, ib(3), DXGI_FORMAT_R32_UINT, 12, 0);
            let after = super::super::bind::counts_snapshot();
            assert_eq!(super::super::bind::last_draw_args(), (true, 12, 1, 0, 0));
            assert_eq!(
                (after[0] + after[1]) - (before[0] + before[1]),
                1,
                "translated graphics PSO must go through bind::graphics_bound (DRAWS or DRAWS_SKIPPED)"
            );
            assert_eq!(after[2], before[2], "translated graphics PSO must not take the PC draw path");
            let ibs = std::mem::take(&mut *fake.index_buffers.lock().unwrap());
            if after[0] > before[0] {
                assert_eq!(ibs.len(), 2);
                assert_eq!(ibs[0].unwrap().SizeInBytes, 48);
                assert_eq!(ibs[0].unwrap().Format, DXGI_FORMAT_R32_UINT);
                assert_eq!(ibs[1].unwrap().BufferLocation, orig_ib.BufferLocation);
            } else {
                assert!(ibs.is_empty(), "a rejected preparation must not mutate the native index buffer");
            }

            // 3. Zero count or invalid format returns early without touching IB or draws.
            let draws_before = fake.draws.lock().unwrap().len();
            draw_indexed_x(list_ptr, ib(4), DXGI_FORMAT_R16_UINT, 0, 0);
            draw_indexed_x(list_ptr, ib(4), DXGI_FORMAT_R8_UINT, 3, 0);
            assert_eq!(fake.draws.lock().unwrap().len(), draws_before);
            assert!(fake.index_buffers.lock().unwrap().is_empty());

            real::forget_pso_for_test(translated_pso);
        }

        super::super::bind::forget(list_ptr);
        on_release(list_ptr as usize);
        unsafe { windows_sys::Win32::System::Memory::VirtualFree(reserved as *mut c_void, 0, windows_sys::Win32::System::Memory::MEM_RELEASE) };
    }

    #[test]
    fn kickoff_x_render_targets_preserve_mrt_count_for_single_descriptor_range() {
        #[derive(Debug, PartialEq, Eq)]
        struct RtCall {
            n: u32,
            handles: Vec<usize>,
            rtvs_was_null: bool,
            single: i32,
            dsv: Option<usize>,
        }

        #[repr(C)]
        struct FakeRtList {
            vt: *const usize,
            calls: Mutex<Vec<RtCall>>,
        }

        unsafe extern "system" fn fake_om_set_render_targets(
            this: *mut c_void,
            n: u32,
            rtvs: *const usize,
            single: i32,
            dsv: *const usize,
        ) {
            let read_n = if rtvs.is_null() {
                0
            } else if single != 0 {
                n.min(1) as usize
            } else {
                n as usize
            };
            let handles = if read_n == 0 { Vec::new() } else { std::slice::from_raw_parts(rtvs, read_n).to_vec() };
            (*(this as *const FakeRtList)).calls.lock().unwrap().push(RtCall {
                n,
                handles,
                rtvs_was_null: rtvs.is_null(),
                single,
                dsv: dsv.as_ref().copied(),
            });
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[PC_OM_SET_RENDER_TARGETS] = fake_om_set_render_targets as usize;
        let mut ext_vt = [0usize; 2];
        ext_vt[0] = orig_vt.as_ptr() as usize;

        let fake = Box::new(FakeRtList {
            // SAFETY: `ext_vt` has 2 entries so `.add(1)` has `orig_vt` at `-1` as `real::orig_fn` expects.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            calls: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakeRtList as *mut c_void;

        let restore = || {
            let b = LISTS.lock().unwrap().as_ref().unwrap().get(&(list_ptr as usize)).unwrap().bindings.clone();
            unsafe { restore_bindings(list_ptr, &b, D3D12_COMMAND_LIST_TYPE_DIRECT) };
        };

        unsafe {
            // 1. Single contiguous descriptor range for 4 MRTs + DSV: only 1 handle exists at `rtvs`,
            //    yet restore_bindings must pass `n = 4` (not 1!) to OMSetRenderTargets.
            let single_base = [0x1000usize];
            let dsv_val = 0x9000usize;
            note_render_targets(list_ptr, 4, single_base.as_ptr(), 1, &dsv_val);
            assert!(
                state_summary(list_ptr).contains("render targets Some((4, true))"),
                "state_summary should report full MRT count: {}",
                state_summary(list_ptr)
            );
            restore();

            // 2. Oversized count with single range is clamped to D3D12 max (8).
            note_render_targets(list_ptr, 12, single_base.as_ptr(), 1, std::ptr::null());
            restore();

            // 3. Separate handles (single == 0) for 3 MRTs.
            let multi = [0x2000usize, 0x2020, 0x2040];
            note_render_targets(list_ptr, 3, multi.as_ptr(), 0, std::ptr::null());
            restore();

            // 4. Depth-only (n == 0, rtvs == null, dsv != null).
            note_render_targets(list_ptr, 0, std::ptr::null(), 0, &dsv_val);
            restore();

            // 5. Null rtvs pointer with non-zero n must safely restore n == 0, null pointer.
            note_render_targets(list_ptr, 4, std::ptr::null(), 1, std::ptr::null());
            restore();
        }

        let calls = fake.calls.lock().unwrap();
        assert_eq!(
            *calls,
            vec![
                RtCall { n: 4, handles: vec![0x1000], rtvs_was_null: false, single: 1, dsv: Some(0x9000) },
                RtCall { n: 8, handles: vec![0x1000], rtvs_was_null: false, single: 1, dsv: None },
                RtCall { n: 3, handles: vec![0x2000, 0x2020, 0x2040], rtvs_was_null: false, single: 0, dsv: None },
                RtCall { n: 0, handles: vec![], rtvs_was_null: true, single: 0, dsv: Some(0x9000) },
                RtCall { n: 0, handles: vec![], rtvs_was_null: true, single: 1, dsv: None },
            ]
        );

        on_release(list_ptr as usize);
    }

    #[test]
    fn set_predication_buffer_state_transitions_and_lifecycle() {
        #[derive(Debug, PartialEq, Eq)]
        enum PredEvent {
            Barrier(usize, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES),
            Copy(usize, u64, usize, u64, u64),
            SetPred(usize, u64, D3D12_PREDICATION_OP),
        }

        #[repr(C)]
        struct PredDesc {
            va: u64,
            op: u32,
        }

        #[repr(C)]
        struct FakePredList {
            vt: *const usize,
            ty: Mutex<D3D12_COMMAND_LIST_TYPE>,
            events: Mutex<Vec<PredEvent>>,
        }

        #[repr(C)]
        struct FakeComRes {
            vt: *const usize,
            refs: AtomicUsize,
        }

        unsafe extern "system" fn fake_get_type(this: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
            *(*(this as *const FakePredList)).ty.lock().unwrap()
        }

        unsafe extern "system" fn fake_resource_barrier(this: *mut c_void, count: u32, barriers: *const D3D12_RESOURCE_BARRIER) {
            let mut events = (*(this as *const FakePredList)).events.lock().unwrap();
            for i in 0..count as usize {
                let b = &*barriers.add(i);
                assert_eq!(b.Type, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION);
                let tr = &*b.Anonymous.Transition;
                let res = (std::ptr::addr_of!(tr.pResource) as *const usize).read_unaligned();
                events.push(PredEvent::Barrier(res, tr.StateBefore, tr.StateAfter));
            }
        }

        unsafe extern "system" fn fake_copy_buffer_region(
            this: *mut c_void,
            dst: usize,
            dst_off: u64,
            src: usize,
            src_off: u64,
            n: u64,
        ) {
            (*(this as *const FakePredList))
                .events
                .lock()
                .unwrap()
                .push(PredEvent::Copy(dst, dst_off, src, src_off, n));
        }

        unsafe extern "system" fn fake_set_predication(
            this: *mut c_void,
            res: usize,
            off: u64,
            op: D3D12_PREDICATION_OP,
        ) {
            (*(this as *const FakePredList))
                .events
                .lock()
                .unwrap()
                .push(PredEvent::SetPred(res, off, op));
        }

        unsafe extern "system" fn res_add_ref(this: *mut c_void) -> u32 {
            (*(this as *const FakeComRes)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }

        unsafe extern "system" fn res_release(this: *mut c_void) -> u32 {
            (*(this as *const FakeComRes)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[PC_GET_TYPE] = fake_get_type as usize;
        orig_vt[PC_COPY_BUFFER_REGION] = fake_copy_buffer_region as usize;
        orig_vt[PC_RESOURCE_BARRIER] = fake_resource_barrier as usize;
        orig_vt[PC_SET_PREDICATION] = fake_set_predication as usize;

        let mut ext_vt = [0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakePredList {
            // SAFETY: `ext_vt` has 65 entries so `.add(1)` points at slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            ty: Mutex::new(D3D12_COMMAND_LIST_TYPE_DIRECT),
            events: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakePredList as *mut c_void;
        let list_key = list_ptr as usize;

        let mut res_vt = [0usize; 16];
        res_vt[1] = res_add_ref as usize;
        res_vt[2] = res_release as usize;
        let fake_res = Box::new(FakeComRes {
            vt: res_vt.as_ptr(),
            refs: AtomicUsize::new(1),
        });
        let pb_raw = &*fake_res as *const FakeComRes as usize;
        let uma_buf = 0xDEAD_BEEFusize;

        let take_events = || std::mem::take(&mut *fake.events.lock().unwrap());

        unsafe {
            // Seed ListState with fake dedicated predication resource and verify initial rejections.
            on_create(list_ptr, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            {
                let mut g = LISTS.lock().unwrap();
                let s = g.as_mut().unwrap().get_mut(&list_key).unwrap();
                s.pred_buf = Some(ID3D12Resource::from_raw(pb_raw as *mut c_void));
            }

            // (a) BUNDLE and COPY lists are rejected without emitting any commands.
            *fake.ty.lock().unwrap() = D3D12_COMMAND_LIST_TYPE_BUNDLE;
            set_predication_buffer_x(list_ptr, std::ptr::null());
            assert!(take_events().is_empty(), "BUNDLE list must reject SetPredicationBufferX");

            *fake.ty.lock().unwrap() = D3D12_COMMAND_LIST_TYPE_COPY;
            set_predication_buffer_x(list_ptr, std::ptr::null());
            assert!(take_events().is_empty(), "COPY list must reject SetPredicationBufferX");

            // (b) DIRECT and COMPUTE lists reject invalid descriptors / ops / alignments / non-UMA VAs.
            let non_uma_slot = 0u64;
            let non_uma_va = &non_uma_slot as *const u64 as u64;
            let bad_op = PredDesc { va: 0x1000, op: 2 };
            let unaligned = PredDesc { va: 0x1004, op: 0 };
            let non_uma = PredDesc { va: non_uma_va, op: 0 };
            let bad_desc = std::ptr::without_provenance::<u64>(1);
            for ty in [D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE] {
                *fake.ty.lock().unwrap() = ty;
                set_predication_buffer_x(list_ptr, bad_desc);
                set_predication_buffer_x(list_ptr, &bad_op as *const PredDesc as *const u64);
                set_predication_buffer_x(list_ptr, &unaligned as *const PredDesc as *const u64);
                set_predication_buffer_x(list_ptr, &non_uma as *const PredDesc as *const u64);
                assert!(take_events().is_empty(), "invalid inputs on {ty:?} must emit no events");
            }

            // (c) Initial apply_predication_copy emits exact barriers + copy + SetPredication without disarm.
            *fake.ty.lock().unwrap() = D3D12_COMMAND_LIST_TYPE_DIRECT;
            apply_predication_copy(list_ptr, uma_buf, 0x100, pb_raw, D3D12_PREDICATION_OP_EQUAL_ZERO);
            assert_eq!(
                take_events(),
                vec![
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                    PredEvent::Copy(pb_raw, 0, uma_buf, 0x100, 8),
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION),
                    PredEvent::SetPred(pb_raw, 0, D3D12_PREDICATION_OP_EQUAL_ZERO),
                ]
            );

            // Rejections while a predicate is active must neither emit events nor disturb tracked state.
            for ty in [D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE] {
                *fake.ty.lock().unwrap() = ty;
                set_predication_buffer_x(list_ptr, bad_desc);
                set_predication_buffer_x(list_ptr, &bad_op as *const PredDesc as *const u64);
                set_predication_buffer_x(list_ptr, &unaligned as *const PredDesc as *const u64);
                set_predication_buffer_x(list_ptr, &non_uma as *const PredDesc as *const u64);
                assert!(take_events().is_empty());
            }
            *fake.ty.lock().unwrap() = D3D12_COMMAND_LIST_TYPE_DIRECT;
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&list_key).unwrap();
                assert_eq!(s.predication, Some(D3D12_PREDICATION_OP_EQUAL_ZERO));
                assert!(s.pred_in_state);
            }

            // (d) Direct switch without NULL disarms active predicate BEFORE copy and transitions PREDICATION -> COPY_DEST.
            apply_predication_copy(list_ptr, uma_buf, 0x200, pb_raw, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);
            assert_eq!(
                take_events(),
                vec![
                    PredEvent::SetPred(0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO),
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COPY_DEST),
                    PredEvent::Copy(pb_raw, 0, uma_buf, 0x200, 8),
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION),
                    PredEvent::SetPred(pb_raw, 0, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO),
                ]
            );

            // (e) ClearState clears predication while keeping pred_in_state == true:
            // re-arming omits redundant SetPred(0, ...) yet transitions pb_raw from PREDICATION -> COPY_DEST,
            // and ClearState + before_close still transitions pb_raw from PREDICATION -> COMMON (once).
            on_clear_state(list_ptr, 0);
            apply_predication_copy(list_ptr, uma_buf, 0x300, pb_raw, D3D12_PREDICATION_OP_EQUAL_ZERO);
            assert_eq!(
                take_events(),
                vec![
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COPY_DEST),
                    PredEvent::Copy(pb_raw, 0, uma_buf, 0x300, 8),
                    PredEvent::Barrier(uma_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PREDICATION),
                    PredEvent::SetPred(pb_raw, 0, D3D12_PREDICATION_OP_EQUAL_ZERO),
                ]
            );

            on_clear_state(list_ptr, 0);
            before_close(list_ptr);
            assert_eq!(
                take_events(),
                vec![PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON)]
            );
            before_close(list_ptr);
            assert!(take_events().is_empty(), "second before_close must be idempotent");

            // (f) NULL transitions PREDICATION -> COMMON when active; second NULL emits 0 barriers.
            apply_predication_copy(list_ptr, uma_buf, 0x100, pb_raw, D3D12_PREDICATION_OP_EQUAL_ZERO);
            take_events();
            set_predication_buffer_x(list_ptr, std::ptr::null());
            assert_eq!(
                take_events(),
                vec![
                    PredEvent::SetPred(0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO),
                    PredEvent::Barrier(pb_raw, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON),
                ]
            );
            set_predication_buffer_x(list_ptr, std::ptr::null());
            assert_eq!(take_events(), vec![PredEvent::SetPred(0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO)]);

            // (g) on_pc_reset clears predication & pred_in_state so subsequent before_close emits 0 barriers.
            apply_predication_copy(list_ptr, uma_buf, 0x100, pb_raw, D3D12_PREDICATION_OP_EQUAL_ZERO);
            take_events();
            on_pc_reset(list_ptr);
            before_close(list_ptr);
            assert!(take_events().is_empty(), "on_pc_reset must clear pred_in_state before before_close");
        }

        // (h) on_release removes ListState and releases the dedicated COM resource.
        on_release(list_key);
        assert!(!has_list_state(list_key));
        assert_eq!(fake_res.refs.load(Ordering::Relaxed), 0);
    }

    /// Write64BitValueBottomOfPipeX / Write64BitValueTopOfPipeX with timestamp flags targeting
    /// writable memory outside unified memory resolve through the timestamp readback buffer and
    /// land once the queue completes. Unaligned and unwritable addresses are ignored.
    #[test]
    fn timestamps_outside_unified_memory_land_after_the_queue() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();

            let list_ptr = list.as_raw();
            let orig_vt = *(list_ptr as *const *const usize);
            let mut ext_vt = [0usize; 65];
            ext_vt[0] = orig_vt as usize;
            std::ptr::copy_nonoverlapping(orig_vt, ext_vt.as_mut_ptr().add(1), 64);

            on_create(list_ptr, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            *(list_ptr as *mut *const usize) = ext_vt.as_ptr().add(1);

            let bottom_ts = Box::new(0u64);
            let top_ts = Box::new(0u64);
            let bottom_va = bottom_ts.as_ref() as *const u64 as u64;
            let top_va = top_ts.as_ref() as *const u64 as u64;

            // Unaligned and null targets are ignored without queueing CPU writes.
            write64_bottom_x(list_ptr, bottom_va + 4, 0, 0, BOTTOM_TIMESTAMPS);
            write64_bottom_x(list_ptr, 0, 0, 0, BOTTOM_TIMESTAMPS);
            {
                let g = LISTS.lock().unwrap();
                assert!(g.as_ref().unwrap()[&(list_ptr as usize)].cpu_writes.is_empty());
            }

            // Valid 8-byte aligned non-UMA targets record GPU timestamp queries + readback copies.
            write64_top_x(list_ptr, top_va, 0, TOP_TIMESTAMPS);
            write64_bottom_x(list_ptr, bottom_va, 0, 0, BOTTOM_TIMESTAMPS);
            {
                let g = LISTS.lock().unwrap();
                let state = &g.as_ref().unwrap()[&(list_ptr as usize)];
                assert_eq!(state.cpu_writes.len(), 2);
                assert!(state.written_ranges.overlaps(WriteProducer::CpuAfterSegment, top_va, 8));
                assert!(state.written_ranges.overlaps(WriteProducer::CpuAfterSegment, bottom_va, 8));
                assert!(!state.written_ranges.overlaps(WriteProducer::InSegmentGpu, top_va, 8));
            }

            // Restore the real vtable before closing and executing on the PC queue.
            *(list_ptr as *mut *const usize) = orig_vt;
            list.Close().unwrap();
            let cl: ID3D12CommandList = list.cast().unwrap();
            q.ExecuteCommandLists(&[Some(cl)]);
            cpu_writes_after(q.as_raw(), &[list_ptr]);

            let t = Instant::now();
            while (*(bottom_va as *const AtomicU64)).load(Ordering::Acquire) == 0 || (*(top_va as *const AtomicU64)).load(Ordering::Acquire) == 0 {
                assert!(t.elapsed().as_secs() < 5, "non-UMA GPU timestamp never landed (top={}, bottom={})", *top_ts, *bottom_ts);
                std::thread::yield_now();
            }
            assert_ne!(*top_ts, 0);
            assert_ne!(*bottom_ts, 0);

            on_release(list_ptr as usize);
        }
    }

    #[test]
    fn write_buffer_immediate_on_open_existing_heap_visible_to_cpu() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let dev3: ID3D12Device3 = dev.cast().unwrap();

            let mem = VirtualAlloc(std::ptr::null(), 0x1_0000, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
            assert!(!mem.is_null());
            let base = mem as usize;

            let heap: ID3D12Heap = dev3.OpenExistingHeapFromAddress(mem).expect("OpenExistingHeapFromAddress");
            let buf_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 0x1_0000,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
                ..Default::default()
            };
            let mut buf: Option<ID3D12Resource> = None;
            dev.CreatePlacedResource(&heap, 0, &buf_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).unwrap();
            let buf = buf.unwrap();
            let gpu_va = buf.GetGPUVirtualAddress();

            // Upload buffer holding known u64 for CopyBufferRegion comparison
            let up_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut up: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &up_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut up,
            )
            .unwrap();
            let up = up.unwrap();
            let mut up_ptr = std::ptr::null_mut();
            up.Map(0, None, Some(&mut up_ptr)).unwrap();
            *(up_ptr as *mut u64) = 0xCAFE_BABE_DEAD_BEEFu64;
            up.Unmap(0, None);

            // Readback buffer to inspect what the GPU itself sees at +0x100, +0x180, +0x200
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &up_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut rb,
            )
            .unwrap();
            let rb = rb.unwrap();

            // 1. DIRECT queue: WriteBufferImmediate at +0x100 and CopyBufferRegion at +0x200
            let dq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let da: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let dl: ID3D12GraphicsCommandList2 = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &da, None).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let make_barrier = |before, after| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(buf.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: before,
                        StateAfter: after,
                    }),
                },
            };

            dl.ResourceBarrier(&[make_barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
            let p_direct = [
                D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: gpu_va + 0x100, Value: 0x0000_0677 },
                D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: gpu_va + 0x104, Value: 0x0000_0000 },
            ];
            let m_direct = [D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; 2];
            dl.WriteBufferImmediate(2, p_direct.as_ptr(), Some(m_direct.as_ptr()));
            dl.CopyBufferRegion(&buf, 0x200, &up, 0, 8);
            dl.ResourceBarrier(&[make_barrier(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
            dl.Close().unwrap();
            let dcl: ID3D12CommandList = dl.cast().unwrap();
            dq.ExecuteCommandLists(&[Some(dcl)]);
            dq.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, &dev, None, 2000).unwrap();

            // 2. COMPUTE queue: WriteBufferImmediate at +0x180
            let cq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let ca: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let cl2: ID3D12GraphicsCommandList2 = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &ca, None).unwrap();
            cl2.ResourceBarrier(&[make_barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
            let p_comp = [
                D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: gpu_va + 0x180, Value: 0x0000_0677 },
                D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: gpu_va + 0x184, Value: 0x1122_3344 },
            ];
            let m_comp = [D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; 2];
            cl2.WriteBufferImmediate(2, p_comp.as_ptr(), Some(m_comp.as_ptr()));
            cl2.ResourceBarrier(&[make_barrier(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
            cl2.Close().unwrap();
            let ccl: ID3D12CommandList = cl2.cast().unwrap();
            cq.ExecuteCommandLists(&[Some(ccl)]);
            cq.Signal(&fence, 2).unwrap();
            crate::gpu::fence::wait(&fence, 2, &dev, None, 2000).unwrap();

            // 3. GPU readback of +0x100, +0x180, +0x200
            let da2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let dl2: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &da2, None).unwrap();
            dl2.CopyBufferRegion(&rb, 0, &buf, 0x100, 8);
            dl2.CopyBufferRegion(&rb, 8, &buf, 0x180, 8);
            dl2.CopyBufferRegion(&rb, 16, &buf, 0x200, 8);
            dl2.Close().unwrap();
            let dcl2: ID3D12CommandList = dl2.cast().unwrap();
            dq.ExecuteCommandLists(&[Some(dcl2)]);
            dq.Signal(&fence, 3).unwrap();
            crate::gpu::fence::wait(&fence, 3, &dev, None, 2000).unwrap();

            let cpu_direct = *((base + 0x100) as *const u64);
            let cpu_comp = *((base + 0x180) as *const u64);
            let cpu_copy = *((base + 0x200) as *const u64);
            let mut rb_ptr = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut rb_ptr)).unwrap();
            let gpu_direct = *(rb_ptr as *const u64);
            let gpu_comp = *((rb_ptr as *const u64).add(1));
            let gpu_copy = *((rb_ptr as *const u64).add(2));
            rb.Unmap(0, None);

            drop(dl);
            drop(dl2);
            drop(cl2);
            drop(buf);
            drop(heap);
            VirtualFree(mem, 0, MEM_RELEASE);

            assert_eq!(cpu_direct, 0x677, "DIRECT WriteBufferImmediate must be visible to CPU (gpu={gpu_direct:#x})");
            assert_eq!(cpu_comp, 0x1122_3344_0000_0677, "COMPUTE WriteBufferImmediate must be visible to CPU (gpu={gpu_comp:#x})");
            assert_eq!(cpu_copy, 0xCAFE_BABE_DEAD_BEEF, "DIRECT CopyBufferRegion must be visible to CPU (gpu={gpu_copy:#x})");
        }
    }

    #[test]
    fn sync_watchdog_pair_is_set_before_the_thread_starts() {
        static WAKE: OnceLock<(Mutex<()>, Condvar)> = OnceLock::new();
        let mut seen = Vec::new();
        for _ in 0..2 {
            // the thread runs its first line before the spawn returns
            sync_watchdog_pair(&WAKE, || seen.push(std::thread::spawn(|| WAKE.get().is_some()).join().unwrap()));
        }
        assert_eq!(seen, [true]);
    }

    /// Submitting a DIRECT list with `Wait64BitValueX(va, >=, 0x677)` BEFORE submitting the
    /// COMPUTE list that recorded `Write64BitValueBottomOfPipeX(va, 0x677)` must bridge via
    /// GPU `queue.Wait`/`queue.Signal` without blocking the submitter thread for `WAIT_TIMEOUT`.
    #[test]
    fn cross_queue_wait64_before_signal_does_not_deadlock_submit_thread() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let dq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let da: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let ca: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let dl: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &da, None).unwrap();
            let cl: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &ca, None).unwrap();
            let done_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let cell = Box::new(AtomicU64::new(0));
            let va = &*cell as *const AtomicU64 as u64;

            let dl_ptr = dl.as_raw();
            let cl_ptr = cl.as_raw();
            on_create(dl_ptr, da.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(cl_ptr, ca.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            // Worker threads record both lists: COMPUTE writes `va = 0x677`, DIRECT waits `va >= 0x677`.
            note_write64_recorded(cl_ptr, va, 0x677);
            wait64_x(dl_ptr, va, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x677, 1, 0);
            dl.Close().unwrap();
            cl.Close().unwrap();

            // Single submitter thread submits DIRECT (consumer) FIRST — before COMPUTE (producer)!
            let t0 = Instant::now();
            let dcl: ID3D12CommandList = dl.cast().unwrap();
            execute_with_waits_on(dq.as_raw(), &[dl_ptr], |chunk| {
                assert_eq!(chunk, &[dl_ptr]);
                dq.ExecuteCommandLists(&[Some(dcl.clone())]);
                signal_write64_fences(dq.as_raw(), chunk);
            });
            let direct_submit_ms = t0.elapsed().as_millis();
            assert!(
                direct_submit_ms < 500,
                "wait-before-signal must bridge to GPU queue.Wait without blocking CPU (took {direct_submit_ms} ms)"
            );
            dq.Signal(&done_fence, 1).unwrap();

            // Now submit COMPUTE (producer) which signals the per-VA GPU fence.
            let ccl: ID3D12CommandList = cl.cast().unwrap();
            execute_with_waits_on(cq.as_raw(), &[cl_ptr], |chunk| {
                assert_eq!(chunk, &[cl_ptr]);
                cq.ExecuteCommandLists(&[Some(ccl.clone())]);
                signal_write64_fences(cq.as_raw(), chunk);
            });

            // DIRECT queue must now unblock and finish promptly on the GPU.
            crate::gpu::fence::wait(&done_fence, 1, &dev, None, 2000).expect("DIRECT queue must complete after COMPUTE signals per-VA fence");

            on_release(dl_ptr as usize);
            on_release(cl_ptr as usize);
            SYNC_GRAPH.lock().unwrap().as_mut().map(|gr| gr.cells.remove(&va));
        }
    }

    /// Cross-queue cycle detection (`DIRECT` waits on `COMPUTE` while `COMPUTE` is already
    /// blocked on an unsubmitted write from `DIRECT`) and same-queue FIFO bypass must both
    /// complete immediately without stalling CPU or hanging GPU.
    #[test]
    fn cross_queue_cycle_and_same_queue_fifo_do_not_deadlock() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let dq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let da: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let ca1: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let ca2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let dl: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &da, None).unwrap();
            let cl1: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &ca1, None).unwrap();
            let cl2: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &ca2, None).unwrap();
            let done_d: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done_c: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let cell_a = Box::new(AtomicU64::new(0));
            let cell_b = Box::new(AtomicU64::new(0));
            let va_a = &*cell_a as *const AtomicU64 as u64;
            let va_b = &*cell_b as *const AtomicU64 as u64;

            let (dl_ptr, cl1_ptr, cl2_ptr) = (dl.as_raw(), cl1.as_raw(), cl2.as_raw());
            on_create(dl_ptr, da.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(cl1_ptr, ca1.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(cl2_ptr, ca2.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            // Fork-join cycle: DIRECT writes A then waits on B; COMPUTE1 waits on A then writes B;
            // COMPUTE2 waits on B (same-queue FIFO after COMPUTE1) after writing something else.
            note_write64_recorded(dl_ptr, va_a, 0x702);
            wait64_x(dl_ptr, va_b, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x702, 1, 0);
            wait64_x(cl1_ptr, va_a, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x702, 1, 0);
            note_write64_recorded(cl1_ptr, va_b, 0x702);
            wait64_x(cl2_ptr, va_b, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x702, 1, 0);
            dl.Close().unwrap();
            cl1.Close().unwrap();
            cl2.Close().unwrap();

            let t0 = Instant::now();
            let ccl1: ID3D12CommandList = cl1.cast().unwrap();
            execute_with_waits_on(cq.as_raw(), &[cl1_ptr], |chunk| {
                cq.ExecuteCommandLists(&[Some(ccl1.clone())]);
                signal_write64_fences(cq.as_raw(), chunk);
            });
            // Same-queue FIFO bypass: cl2 waits on va_b already submitted on cq.
            let ccl2: ID3D12CommandList = cl2.cast().unwrap();
            execute_with_waits_on(cq.as_raw(), &[cl2_ptr], |chunk| {
                cq.ExecuteCommandLists(&[Some(ccl2.clone())]);
                signal_write64_fences(cq.as_raw(), chunk);
            });
            cq.Signal(&done_c, 1).unwrap();

            // Cycle breaking: dl writes va_a (which cq is blocked on) and waits on va_b (from cq).
            // Must detect the cycle, skip hoisting va_b ahead of dl, submit dl immediately, and unblock BOTH queues!
            let dcl: ID3D12CommandList = dl.cast().unwrap();
            execute_with_waits_on(dq.as_raw(), &[dl_ptr], |chunk| {
                dq.ExecuteCommandLists(&[Some(dcl.clone())]);
                signal_write64_fences(dq.as_raw(), chunk);
            });
            dq.Signal(&done_d, 1).unwrap();

            assert!(t0.elapsed().as_millis() < 500, "submits must not stall CPU (took {} ms)", t0.elapsed().as_millis());
            crate::gpu::fence::wait(&done_d, 1, &dev, None, 2000).expect("DIRECT must finish without GPU deadlock");
            crate::gpu::fence::wait(&done_c, 1, &dev, None, 2000).expect("COMPUTE must finish once DIRECT signals A");

            on_release(dl_ptr as usize);
            on_release(cl1_ptr as usize);
            on_release(cl2_ptr as usize);
            if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
                gr.cells.remove(&va_a);
                gr.cells.remove(&va_b);
            }
        }
    }

    /// `publish_cpu_cycle_break_with` advances writable 8-byte-aligned memory monotonically via
    /// `fetch_max` (both plain title memory and committed coherent UMA), while safely skipping
    /// sparse UMA windows, zero/unaligned targets, and read-only / inaccessible / guard pages.
    #[test]
    fn publish_cpu_cycle_break_uses_fetch_max_and_skips_sparse_or_unwritable() {
        use windows_sys::Win32::System::Memory::{
            VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_GUARD,
            PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        };

        unsafe {
            // Plain title memory: monotonic fetch_max, never rewinds.
            let cell = Box::new(AtomicU64::new(0));
            let va = &*cell as *const AtomicU64 as u64;
            publish_cpu_cycle_break_with(va, 0, None);
            assert_eq!(cell.load(Ordering::Acquire), 0);
            publish_cpu_cycle_break_with(va, 0x702, None);
            assert_eq!(cell.load(Ordering::Acquire), 0x702);
            publish_cpu_cycle_break_with(va, 0x500, None);
            assert_eq!(cell.load(Ordering::Acquire), 0x702, "must not rewind below current value");
            publish_cpu_cycle_break_with(va, 0x900, None);
            assert_eq!(cell.load(Ordering::Acquire), 0x900);

            // Misaligned and null addresses are ignored safely.
            let buf = Box::new([AtomicU64::new(0), AtomicU64::new(0)]);
            let base = buf.as_ptr() as u64;
            publish_cpu_cycle_break_with(base + 4, 0x1234, None);
            assert_eq!(buf[0].load(Ordering::Acquire), 0);
            assert_eq!(buf[1].load(Ordering::Acquire), 0);
            publish_cpu_cycle_break_with(0, 0x1234, None);

            // Non-writable and guard pages are skipped without faulting.
            let page = VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            assert!(!page.is_null());
            let page_va = page as u64;
            (*(page as *const AtomicU64)).store(0x11, Ordering::Release);
            let mut old = 0u32;
            assert_ne!(VirtualProtect(page, 0x1000, PAGE_READONLY, &mut old), 0);
            publish_cpu_cycle_break_with(page_va, 0x22, None);
            assert_eq!((*(page as *const AtomicU64)).load(Ordering::Acquire), 0x11, "PAGE_READONLY must be skipped");

            // Surviving PAGE_NOACCESS / PAGE_GUARD without faulting and still reading 0x11 once
            // PAGE_READWRITE is restored verifies both protections were skipped.
            assert_ne!(VirtualProtect(page, 0x1000, PAGE_NOACCESS, &mut old), 0);
            publish_cpu_cycle_break_with(page_va, 0x33, None);

            assert_ne!(VirtualProtect(page, 0x1000, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
            publish_cpu_cycle_break_with(page_va, 0x44, None);

            assert_ne!(VirtualProtect(page, 0x1000, PAGE_READWRITE, &mut old), 0);
            assert_eq!((*(page as *const AtomicU64)).load(Ordering::Acquire), 0x11, "NOACCESS/GUARD must not write");
            VirtualFree(page, 0, MEM_RELEASE);

            // Real runtime UMA layout: committed coherent window gets updated; sparse window is
            // skipped even when its CPU page is committed and writable.
            let uma = crate::gpu::uma::Uma::for_runtime(16 << 20).expect("runtime UMA");
            let coherent = uma.alloc(0x1000, 8).expect("coherent UMA slot") as usize;
            assert!(uma.in_committed_window(coherent, 8) && !uma.is_sparse(coherent));
            (*(coherent as *const AtomicU64)).store(0, Ordering::Release);
            publish_cpu_cycle_break_with(coherent as u64, 0xabc, Some(&uma));
            assert_eq!((*(coherent as *const AtomicU64)).load(Ordering::Acquire), 0xabc);
            assert!(uma.free(coherent));

            let (sparse_va, _) = uma.sparse_alloc(0x1_0000, 0x1_0000).expect("sparse UMA slot");
            assert!(uma.is_sparse(sparse_va));
            let committed = VirtualAlloc(sparse_va as *const c_void, 0x1000, MEM_COMMIT, PAGE_READWRITE);
            assert_eq!(committed as usize, sparse_va);
            (*(sparse_va as *const AtomicU64)).store(0, Ordering::Release);
            publish_cpu_cycle_break_with(sparse_va as u64, 0xdef, Some(&uma));
            assert_eq!(
                (*(sparse_va as *const AtomicU64)).load(Ordering::Acquire),
                0,
                "sparse UMA window must not be written on CPU cycle break"
            );
            VirtualFree(sparse_va as *mut c_void, 0x1000, MEM_DECOMMIT);
            assert!(uma.sparse_release(sparse_va).is_some());
        }
    }

    /// A CPU cycle-break (`SignalAction::CpuCycleBreak` and `break_cycles_before_title_signal`)
    /// must publish the released value into title memory with `fetch_max` so CPU polls and
    /// subsequent `Wait64BitValueX` checks observe it without re-arming or rewinding.
    #[test]
    fn cpu_cycle_break_publishes_value_to_title_memory_and_keeps_it_monotonic() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let dq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cq: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let da: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let ca: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let dl: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &da, None).unwrap();
            let cl: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &ca, None).unwrap();
            let done_d: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done_c: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            // 1. 2-queue cycle broken by `SignalAction::CpuCycleBreak` publishes 0x702 to `cell_a`.
            let cell_a = Box::new(AtomicU64::new(0));
            let cell_b = Box::new(AtomicU64::new(0));
            let va_a = &*cell_a as *const AtomicU64 as u64;
            let va_b = &*cell_b as *const AtomicU64 as u64;
            let (dl_ptr, cl_ptr) = (dl.as_raw(), cl.as_raw());
            on_create(dl_ptr, da.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(cl_ptr, ca.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            note_write64_recorded(dl_ptr, va_a, 0x702);
            wait64_x(dl_ptr, va_b, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x702, 1, 0);
            wait64_x(cl_ptr, va_a, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x702, 1, 0);
            note_write64_recorded(cl_ptr, va_b, 0x702);
            dl.Close().unwrap();
            cl.Close().unwrap();

            let ccl: ID3D12CommandList = cl.cast().unwrap();
            execute_with_waits_on(cq.as_raw(), &[cl_ptr], |chunk| {
                cq.ExecuteCommandLists(&[Some(ccl.clone())]);
                signal_write64_fences(cq.as_raw(), chunk);
            });
            cq.Signal(&done_c, 1).unwrap();
            assert_eq!(cell_a.load(Ordering::Acquire), 0);

            let dcl: ID3D12CommandList = dl.cast().unwrap();
            execute_with_waits_on(dq.as_raw(), &[dl_ptr], |chunk| {
                dq.ExecuteCommandLists(&[Some(dcl.clone())]);
                signal_write64_fences(dq.as_raw(), chunk);
            });
            dq.Signal(&done_d, 1).unwrap();

            assert_eq!(cell_a.load(Ordering::Acquire), 0x702, "CpuCycleBreak must publish released value to title memory");
            let follow_up = PendingWait {
                va: va_a,
                width: 8,
                comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
                value: 0x702,
                mask: u64::MAX,
                after_write64: false,
            };
            assert!(follow_up.is_satisfied(), "subsequent CPU/memory wait must see the published value");
            assert!(try_gpu_wait(cq.as_raw(), &follow_up, &[]) == GpuWaitOutcome::Handled);
            {
                let g = SYNC_GRAPH.lock().unwrap();
                let c = g.as_ref().unwrap().cells.get(&va_a).unwrap();
                assert_eq!(c.cpu_signaled_max, 0x702);
                assert!(c.armed_waits.is_empty(), "released wait must not be re-armed");
            }

            crate::gpu::fence::wait(&done_d, 1, &dev, None, 2000).expect("DIRECT must finish after cycle break");
            crate::gpu::fence::wait(&done_c, 1, &dev, None, 2000).expect("COMPUTE must finish after cycle break");

            // 2. Same-queue self-deadlock cycle-break must NOT rewind memory already holding a higher value.
            let cell_hi = Box::new(AtomicU64::new(0));
            let va_hi = &*cell_hi as *const AtomicU64 as u64;
            let a_w: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a_s: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let l_w: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a_w, None).unwrap();
            let l_s: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a_s, None).unwrap();
            let done_hi: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let (lw_ptr, ls_ptr) = (l_w.as_raw(), l_s.as_raw());
            on_create(lw_ptr, a_w.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(ls_ptr, a_s.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            note_write64_recorded(ls_ptr, va_hi, 0x800);
            wait64_x(lw_ptr, va_hi, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x800, 1, 0);
            l_w.Close().unwrap();
            l_s.Close().unwrap();

            let cl_w: ID3D12CommandList = l_w.cast().unwrap();
            execute_with_waits_on(cq.as_raw(), &[lw_ptr], |chunk| {
                cq.ExecuteCommandLists(&[Some(cl_w.clone())]);
                signal_write64_fences(cq.as_raw(), chunk);
            });
            // Simulate a newer write already advancing memory past 0x800 before the cycle-break fires.
            cell_hi.store(0x900, Ordering::Release);
            let cl_s: ID3D12CommandList = l_s.cast().unwrap();
            execute_with_waits_on(cq.as_raw(), &[ls_ptr], |chunk| {
                cq.ExecuteCommandLists(&[Some(cl_s.clone())]);
                signal_write64_fences(cq.as_raw(), chunk);
            });
            cq.Signal(&done_hi, 1).unwrap();
            {
                let g = SYNC_GRAPH.lock().unwrap();
                assert_eq!(g.as_ref().unwrap().cells[&va_hi].cpu_signaled_max, 0x800, "self-deadlock must trigger CpuCycleBreak");
            }
            assert_eq!(cell_hi.load(Ordering::Acquire), 0x900, "CpuCycleBreak must not rewind memory");
            crate::gpu::fence::wait(&done_hi, 1, &dev, None, 2000).expect("self-deadlock must complete");

            // 3. Mixed title-fence + Wait64 cycle broken by `break_cycles_before_title_signal`
            //    also publishes the released value into title memory.
            let cell_m = Box::new(AtomicU64::new(0));
            let va_m = &*cell_m as *const AtomicU64 as u64;
            let cell_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let title_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let qa = dq.as_raw() as usize;
            let qb = cq.as_raw() as usize;
            let tf_raw = title_fence.as_raw() as usize;
            {
                let mut g = SYNC_GRAPH.lock().unwrap();
                let gr = g.get_or_insert_with(SyncGraph::default);
                gr.cells.insert(
                    va_m,
                    SyncCell {
                        fence: Some(cell_fence.clone()),
                        device: dev.as_raw() as usize,
                        recorded_max: 0x50,
                        submitted_max: 0x50,
                        submitted_queue: Some(qb),
                        armed_waits: vec![ArmedGpuWait {
                            val: 0x50,
                            waiting_queue: qa,
                            waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_DIRECT.0,
                            wait_seq: 1,
                            armed_at: Instant::now(),
                            after_write64: false,
                        }],
                        submitted_signals: vec![SubmittedSignal { val: 0x50, queue: qb, seq: 2 }],
                        ..Default::default()
                    },
                );
                // Matches the `fetch_sub(removed)` performed by `break_cycles_before_title_signal`.
                ARMED_GPU_WAITS.fetch_add(1, Ordering::AcqRel);
                gr.title_fences.insert(tf_raw, TitleFenceSync { waits: vec![(1, qb, 1)], signals: Vec::new() });
            }
            break_cycles_before_title_signal(qa, tf_raw, 1);
            assert_eq!(cell_m.load(Ordering::Acquire), 0x50, "mixed cycle break must publish released value to title memory");
            crate::gpu::fence::wait(&cell_fence, 0x50, &dev, None, 2000).expect("mixed cycle break must signal cell fence");

            for p in [dl_ptr, cl_ptr, lw_ptr, ls_ptr] {
                on_release(p as usize);
            }
            if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
                gr.cells.remove(&va_a);
                gr.cells.remove(&va_b);
                gr.cells.remove(&va_hi);
                gr.cells.remove(&va_m);
                gr.title_fences.remove(&tf_raw);
            }
        }
    }

    /// Exact `(queue, seq)` cycle detection must catch k=1 self-deadlocks, k=2 and k=3 transitive
    /// cycles (including mixed title-fence + Wait64 cycles) while NEVER false-positiving on
    /// pipelined frame N/N+1 or back-to-back cross-queue handshakes (`wait_seq > prod_sig_seq`).
    #[test]
    fn pipelined_next_frame_write_is_not_false_cycle() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();

            // 1. Back-to-back handshake (NOT a cycle):
            //    q0 signals A=10 at seq=2, then waits on B=10 at seq=3.
            //    q1 waits on A=10 at seq=1, then wants to signal B=10 (seq=2).
            //    Since q0's wait on B (seq=3) is AFTER q0's signal of A (seq=2), NOT a cycle!
            let f_a: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let f_b: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut g = SyncGraph::default();
            g.cells.insert(
                0x1000,
                SyncCell {
                    fence: Some(f_a),
                    device: dev.as_raw() as usize,
                    recorded_max: 10,
                    submitted_max: 10,
                    submitted_queue: Some(0xC001),
                    armed_waits: vec![ArmedGpuWait {
                        val: 10,
                        waiting_queue: 0xD001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_DIRECT.0,
                        wait_seq: 1,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    submitted_signals: vec![SubmittedSignal { val: 10, queue: 0xC001, seq: 2 }],
                    ..Default::default()
                },
            );
            g.cells.insert(
                0x2000,
                SyncCell {
                    fence: Some(f_b),
                    device: dev.as_raw() as usize,
                    recorded_max: 10,
                    armed_waits: vec![ArmedGpuWait {
                        val: 10,
                        waiting_queue: 0xC001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0,
                        wait_seq: 3, // AFTER q0's signal of A (seq=2)!
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    ..Default::default()
                },
            );
            assert!(
                !would_signal_close_cycle(&mut g, 0xD001, 0x2000, 10),
                "back-to-back handshake (wait_seq=3 > sig_seq=2) must NOT be treated as a cycle"
            );
            // Now invert q0's order: q0 waited on B at seq=1 BEFORE signaling A at seq=2 -> TRUE cycle!
            g.cells.get_mut(&0x2000).unwrap().armed_waits[0].wait_seq = 1;
            assert!(
                would_signal_close_cycle(&mut g, 0xD001, 0x2000, 10),
                "2-queue cycle (wait_seq=1 < sig_seq=2) MUST be detected at Signal time"
            );
            // k = 1 self-deadlock: q0 signaling B when q0 itself is waiting on B.
            assert!(
                would_signal_close_cycle(&mut g, 0xC001, 0x2000, 10),
                "same-queue self-deadlock MUST be detected at Signal time"
            );

            // 2. 3-queue transitive cycle: Direct -> C1 -> C2 -> Direct.
            //    C2 waits on D (0x3000, seq=1), then signals C2_out (0x2000, seq=2).
            //    C1 waits on C2_out (0x2000, seq=1), then signals C1_out (0x1000, seq=2).
            //    Direct waits on C1_out (0x1000, seq=1), then wants to signal D (0x3000).
            let f1: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let f2: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let f3: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut g3 = SyncGraph::default();
            g3.cells.insert(
                0x1000,
                SyncCell {
                    fence: Some(f1),
                    device: dev.as_raw() as usize,
                    recorded_max: 0x78d,
                    submitted_max: 0x78d,
                    submitted_queue: Some(0xC001),
                    armed_waits: vec![ArmedGpuWait {
                        val: 0x78d,
                        waiting_queue: 0xD001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_DIRECT.0,
                        wait_seq: 1,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    submitted_signals: vec![SubmittedSignal { val: 0x78d, queue: 0xC001, seq: 2 }],
                    ..Default::default()
                },
            );
            g3.cells.insert(
                0x2000,
                SyncCell {
                    fence: Some(f2),
                    device: dev.as_raw() as usize,
                    recorded_max: 0x78d,
                    submitted_max: 0x78d,
                    submitted_queue: Some(0xC002),
                    armed_waits: vec![ArmedGpuWait {
                        val: 0x78d,
                        waiting_queue: 0xC001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0,
                        wait_seq: 1,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    submitted_signals: vec![SubmittedSignal { val: 0x78d, queue: 0xC002, seq: 2 }],
                    ..Default::default()
                },
            );
            g3.cells.insert(
                0x3000,
                SyncCell {
                    fence: Some(f3),
                    device: dev.as_raw() as usize,
                    recorded_max: 0x78d,
                    armed_waits: vec![ArmedGpuWait {
                        val: 0x78d,
                        waiting_queue: 0xC002,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0,
                        wait_seq: 1,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    ..Default::default()
                },
            );
            assert!(
                would_signal_close_cycle(&mut g3, 0xD001, 0x3000, 0x78d),
                "3-queue transitive cycle Direct -> C1 -> C2 -> Direct MUST be detected at Signal time"
            );
            // If C2 waited on D AFTER signaling C2_out (wait_seq=3 > sig_seq=2), NOT a cycle.
            g3.cells.get_mut(&0x3000).unwrap().armed_waits[0].wait_seq = 3;
            assert!(
                !would_signal_close_cycle(&mut g3, 0xD001, 0x3000, 0x78d),
                "pipelined 3-queue chain with wait_seq > sig_seq must NOT be treated as a cycle"
            );

            // 3. Next frame's signal on a cell whose armed wait an earlier signal already covers
            //    (a retail title, frame 0x6dc): C (0xC001) signalled A=0x6db, Direct (0xD001)
            //    waits A >= 0x6db (seq 3544) and later signals B=0x6dc (seq 3572); C waits on
            //    B >= 0x6dc (seq 34) and now signals A=0x6dc. Direct's wait is satisfied by C's
            //    0x6db signal, not by this one: no cycle.
            let fa: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let fb: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut gp = SyncGraph::default();
            gp.cells.insert(
                0x4000,
                SyncCell {
                    fence: Some(fa),
                    device: dev.as_raw() as usize,
                    recorded_max: 0x6dc,
                    submitted_max: 0x6db,
                    submitted_queue: Some(0xC001),
                    armed_waits: vec![ArmedGpuWait {
                        val: 0x6db,
                        waiting_queue: 0xD001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_DIRECT.0,
                        wait_seq: 3544,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    submitted_signals: vec![SubmittedSignal { val: 0x6db, queue: 0xC001, seq: 16 }],
                    ..Default::default()
                },
            );
            gp.cells.insert(
                0x5000,
                SyncCell {
                    fence: Some(fb),
                    device: dev.as_raw() as usize,
                    recorded_max: 0x6dc,
                    submitted_max: 0x6dc,
                    submitted_queue: Some(0xD001),
                    armed_waits: vec![ArmedGpuWait {
                        val: 0x6dc,
                        waiting_queue: 0xC001,
                        waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0,
                        wait_seq: 34,
                        armed_at: Instant::now(),
                        after_write64: false,
                    }],
                    submitted_signals: vec![SubmittedSignal { val: 0x6dc, queue: 0xD001, seq: 3572 }],
                    ..Default::default()
                },
            );
            assert!(
                !would_signal_close_cycle(&mut gp, 0xC001, 0x4000, 0x6dc),
                "a wait an earlier signal already satisfies must NOT make the next frame's signal a cycle"
            );
            // Direct waiting on A >= 0x6dc instead (only this signal satisfies it): a true cycle.
            gp.cells.get_mut(&0x4000).unwrap().armed_waits[0].val = 0x6dc;
            assert!(
                would_signal_close_cycle(&mut gp, 0xC001, 0x4000, 0x6dc),
                "a wait only this signal satisfies, armed before the waiter's own signal, MUST be a cycle"
            );
        }
    }

    /// The console's cache flushes are UAV barriers for all resources on lists that take them; a
    /// flush of nothing, a copy list's and a bundle's are not.
    #[test]
    fn cache_flushes_are_uav_barriers() {
        #[repr(C)]
        struct FakeFlushList {
            vt: *const usize,
            ty: i32,
            barriers: Mutex<Vec<(i32, usize)>>,
        }
        unsafe extern "system" fn get_type(this: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
            D3D12_COMMAND_LIST_TYPE((*(this as *const FakeFlushList)).ty)
        }
        unsafe extern "system" fn barrier(this: *mut c_void, n: u32, b: *const D3D12_RESOURCE_BARRIER) {
            for k in 0..n as usize {
                let b = &*b.add(k);
                let res = (*b.Anonymous.UAV).pResource.as_ref().map_or(0, |r| r.as_raw() as usize);
                (*(this as *const FakeFlushList)).barriers.lock().unwrap().push((b.Type.0, res));
            }
        }
        let mut orig_vt = [0usize; 64];
        orig_vt[PC_GET_TYPE] = get_type as usize;
        orig_vt[PC_RESOURCE_BARRIER] = barrier as usize;
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);
        let mut fake = Box::new(FakeFlushList { vt: unsafe { ext_vt.as_ptr().add(1) }, ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0, barriers: Mutex::new(Vec::new()) });
        let uav = (D3D12_RESOURCE_BARRIER_TYPE_UAV.0, 0);
        unsafe {
            let list = &mut *fake as *mut FakeFlushList as *mut c_void;
            flush_cache_range_x(list, 0xc3c0, 0x8002_0000, 0, 0xFFFF_FFFF_FF00);
            flush_cache_range_x(list, 0, 0x8000_0000, 0x1000, 0x100);
            flush_pipeline_x(list, 1, 0, 0);
            assert_eq!(*fake.barriers.lock().unwrap(), vec![uav; 3]);
            // nothing asked for, or of no memory
            flush_cache_range_x(list, 0, 0, 0, 0x1000);
            flush_cache_range_x(list, 0xc3c0, 0, 0, 0);
            assert_eq!(fake.barriers.lock().unwrap().len(), 3);
            fake.ty = D3D12_COMMAND_LIST_TYPE_DIRECT.0;
            flush_cache_range_x(list, 0xc3c0, 0, 0, 1);
            assert_eq!(fake.barriers.lock().unwrap().len(), 4);
            for ty in [D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_LIST_TYPE_BUNDLE] {
                fake.ty = ty.0;
                flush_cache_range_x(list, 0xc3c0, 0, 0, 1);
                flush_pipeline_x(list, 1, 0, 0);
            }
            assert_eq!(fake.barriers.lock().unwrap().len(), 4);
        }
    }

    /// The next frame's Signal of a title fence does not break a "cycle" through a wait an earlier
    /// signal of that fence satisfies; one that only the new signal satisfies is a cycle, broken
    /// by releasing the cell's wait.
    #[test]
    fn a_title_fence_wait_an_earlier_signal_covers_is_no_cycle() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let title: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let cell: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let (q0, q1, va, f) = (0xC0F0usize, 0xC0F1usize, 0x7E57_C0F0u64, title.as_raw() as usize);
            // q1 signalled the fence at 1; q0 waits for that and signals the cell later, which q1
            // waits for; q1 is about to signal the fence at 2
            let set = |wait: u64| {
                let mut g = SYNC_GRAPH.lock().unwrap();
                let gr = g.get_or_insert_with(SyncGraph::default);
                let mut tf = TitleFenceSync::default();
                tf.waits.push((wait, q0, 5));
                tf.signals.push(SubmittedSignal { val: 1, queue: q1, seq: 3 });
                gr.title_fences.insert(f, tf);
                if !gr.cells.contains_key(&va) {
                    ARMED_GPU_WAITS.fetch_add(1, Ordering::AcqRel);
                    gr.cells.insert(
                        va,
                        SyncCell {
                            fence: Some(cell.clone()),
                            device: dev.as_raw() as usize,
                            recorded_max: 9,
                            submitted_max: 9,
                            submitted_queue: Some(q0),
                            armed_waits: vec![ArmedGpuWait { val: 9, waiting_queue: q1, waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0, wait_seq: 4, armed_at: Instant::now(), after_write64: false }],
                            submitted_signals: vec![SubmittedSignal { val: 9, queue: q0, seq: 7 }],
                            ..Default::default()
                        },
                    );
                }
            };
            set(1);
            break_cycles_before_title_signal(q1, f, 2);
            assert_eq!(cell.GetCompletedValue(), 0, "the wait is the earlier signal's: nothing is released");
            set(2);
            break_cycles_before_title_signal(q1, f, 2);
            assert_eq!(cell.GetCompletedValue(), 9, "a wait only the new signal satisfies: released");
            let mut g = SYNC_GRAPH.lock().unwrap();
            let gr = g.as_mut().unwrap();
            gr.title_fences.remove(&f);
            let left = gr.cells.remove(&va).map_or(0, |c| c.armed_waits.len());
            if left > 0 {
                ARMED_GPU_WAITS.fetch_sub(left, Ordering::AcqRel);
            }
        }
    }

    /// Reproduces two real multi-queue hazards seen in retail titles:
    /// 1. Rotating compute queues where cell `va` was previously written by `cq0` in frame N,
    ///    then in frame N+1 `cq1` waits on `va` before `cq1` itself submits the write to `va`
    ///    (same-queue self-deadlock despite stale `submitted_queue == cq0`).
    /// 2. A Wait64 cell becoming non-monotonic (`value < submitted_max`) while another queue
    ///    already has an armed `queue.Wait` on it — must rescue the waiter instead of orphaning it.
    #[test]
    fn rotating_queue_self_deadlock_and_non_monotonic_rescue_do_not_hang() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let cq0: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let cq1: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let a0: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a1: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let l0: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a0, None).unwrap();
            let l1: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a1, None).unwrap();
            let l2: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a2, None).unwrap();
            let done1: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let cell_rot = Box::new(AtomicU64::new(0));
            let va_rot = &*cell_rot as *const AtomicU64 as u64;
            let (l0_ptr, l1_ptr, l2_ptr) = (l0.as_raw(), l1.as_raw(), l2.as_raw());
            on_create(l0_ptr, a0.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(l1_ptr, a1.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(l2_ptr, a2.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            // Frame 0x69f: cq0 writes va_rot = 0x69f (sets stale submitted_queue = cq0).
            note_write64_recorded(l0_ptr, va_rot, 0x69f);
            l0.Close().unwrap();
            let cl0: ID3D12CommandList = l0.cast().unwrap();
            execute_with_waits_on(cq0.as_raw(), &[l0_ptr], |chunk| {
                cq0.ExecuteCommandLists(&[Some(cl0.clone())]);
                signal_write64_fences(cq0.as_raw(), chunk);
            });

            // Frame 0x6a0: l2 records write va_rot = 0x6a0, l1 waits on va_rot >= 0x6a0.
            // Both l1 and l2 are submitted on cq1 (rotated from cq0 -> cq1)!
            note_write64_recorded(l2_ptr, va_rot, 0x6a0);
            wait64_x(l1_ptr, va_rot, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 0x6a0, 1, 0);
            l1.Close().unwrap();
            l2.Close().unwrap();

            let t0 = Instant::now();
            let cl1: ID3D12CommandList = l1.cast().unwrap();
            let cl2: ID3D12CommandList = l2.cast().unwrap();
            execute_with_waits_on(cq1.as_raw(), &[l1_ptr], |chunk| {
                cq1.ExecuteCommandLists(&[Some(cl1.clone())]);
                signal_write64_fences(cq1.as_raw(), chunk);
            });
            execute_with_waits_on(cq1.as_raw(), &[l2_ptr], |chunk| {
                cq1.ExecuteCommandLists(&[Some(cl2.clone())]);
                signal_write64_fences(cq1.as_raw(), chunk);
            });
            cq1.Signal(&done1, 1).unwrap();

            // Must break self-deadlock immediately on CPU at Signal time (not wait 5s for watchdog!).
            crate::gpu::fence::wait(&done1, 1, &dev, None, 1500).expect("cq1 must complete immediately without self-deadlock");
            assert!(t0.elapsed().as_millis() < 1500, "took {} ms", t0.elapsed().as_millis());

            // Part 2: Non-monotonic rescue while a GPU wait is armed.
            let cell_nm = Box::new(AtomicU64::new(0));
            let va_nm = &*cell_nm as *const AtomicU64 as u64;
            let a3: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a4: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a5: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let l3: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a3, None).unwrap();
            let l4: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a4, None).unwrap();
            let l5: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a5, None).unwrap();
            let done2: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let (l3_ptr, l4_ptr, l5_ptr) = (l3.as_raw(), l4.as_raw(), l5.as_raw());
            on_create(l3_ptr, a3.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(l4_ptr, a4.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            on_create(l5_ptr, a5.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);

            // Submit 10 first -> submitted_max = 10.
            note_write64_recorded(l3_ptr, va_nm, 10);
            l3.Close().unwrap();
            let cl3: ID3D12CommandList = l3.cast().unwrap();
            execute_with_waits_on(cq0.as_raw(), &[l3_ptr], |chunk| {
                cq0.ExecuteCommandLists(&[Some(cl3.clone())]);
                signal_write64_fences(cq0.as_raw(), chunk);
            });
            // Record 20 (unsubmitted), then arm cq1 wait for >= 20.
            note_write64_recorded(l4_ptr, va_nm, 20);
            wait64_x(l5_ptr, va_nm, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 20, 1, 0);
            l5.Close().unwrap();
            let cl5: ID3D12CommandList = l5.cast().unwrap();
            execute_with_waits_on(cq1.as_raw(), &[l5_ptr], |chunk| {
                cq1.ExecuteCommandLists(&[Some(cl5.clone())]);
                signal_write64_fences(cq1.as_raw(), chunk);
            });
            cq1.Signal(&done2, 1).unwrap();
            // Now record a strictly decreasing write (5 < submitted_max 10) -> non_monotonic!
            // Must rescue cq1's armed wait on CPU before dropping the fence!
            note_write64_recorded(l4_ptr, va_nm, 5);
            crate::gpu::fence::wait(&done2, 1, &dev, None, 1500).expect("cq1 must be rescued when cell goes non-monotonic");

            for p in [l0_ptr, l1_ptr, l2_ptr, l3_ptr, l4_ptr, l5_ptr] {
                on_release(p as usize);
            }
            if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
                gr.cells.remove(&va_rot);
                gr.cells.remove(&va_nm);
            }
        }
    }

    /// Verifies that CPU-signalling a fence while a GPU `queue.Signal` to the same value is
    /// already in flight (watchdog release path) does not hang subsequent queue completions.
    #[test]
    fn cpu_watchdog_signal_with_inflight_gpu_signal_completes_cleanly() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q1: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let q2: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let f_a: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let f_b: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done1: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done2: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            // Deliberate 2-queue cycle with in-flight GPU signals to value 10 on both fences.
            q1.Wait(&f_a, 10).unwrap();
            q1.Signal(&f_b, 10).unwrap();
            q1.Signal(&done1, 1).unwrap();
            q2.Wait(&f_b, 10).unwrap();
            q2.Signal(&f_a, 10).unwrap();
            q2.Signal(&done2, 1).unwrap();

            // Simulate watchdog breaking the deadlock from CPU on both fences while GPU signals are in flight.
            f_a.Signal(10).unwrap();
            f_b.Signal(10).unwrap();

            crate::gpu::fence::wait(&done1, 1, &dev, None, 2000).expect("q1 must complete after watchdog CPU signal");
            crate::gpu::fence::wait(&done2, 1, &dev, None, 2000).expect("q2 must complete after watchdog CPU signal");
        }
    }

    /// #1452: the title Resets its list and submits the new recording while the GPU still runs
    /// the previous one (held here by a queue Wait on a fence not yet signalled). The new
    /// recording must be on another PC list (the PC forbids submitting one again before its last
    /// submit is done, debug layer id 553), also the times after (not on a stand-in moved to the
    /// spares while its submit runs); once the GPU is done, the list
    /// records on its own PC list again. Both ways a title Resets: the list's #10, and the newer
    /// console layout's #139, which goes through the list's thunk to the PC list it records on
    /// (`reset_x` on that one) – the retail title's way, which the stand-in did not see at first.
    #[test]
    fn a_reset_while_the_last_submit_runs_records_on_another_pc_list() {
        let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
            return crate::gpu::recomp::runner::skip_test("no GPU");
        };
        for newer_layout in [false, true] {
            reset_while_the_last_submit_runs(&dev, newer_layout);
        }
    }

    fn reset_while_the_last_submit_runs(dev: &ID3D12Device, newer_layout: bool) {
        unsafe {
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let a = alloc.as_raw() as usize;
            let primary = dev.CreateCommandList::<_, _, ID3D12GraphicsCommandList>(0, ty, &alloc, None).unwrap().into_raw();
            real::adopt(primary, real::Kind::CommandList);
            on_create(primary, a, 0, ty);
            super::super::bind::reset(primary, a, 0);
            let w = super::super::title_list::wrap(primary);
            let slot = |i: usize| *(*(w as *const *const usize)).add(i);
            let close: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(slot(9));
            let reset_10: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 = std::mem::transmute(slot(10));
            // #139 (allocator, NULL, 0): the thunk passes the PC list last recorded on (`F_BACKING`)
            let reset = |w: *mut c_void, a: usize, pso: usize, last: *mut c_void| if newer_layout { reset_x(last, a, pso) } else { reset_10(w, a, pso) };
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(slot(2));
            let exec: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) =
                std::mem::transmute(*(*(q.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            // Close, then submit as ExecuteCommandLists does: the PC lists the title's list stands for
            let submit = || {
                assert_eq!(close(w), 0);
                let mut segs = Vec::new();
                assert!(super::super::title_list::push_segments(w, &mut segs));
                let again = super::super::alloc_pool::note_resubmits(q.as_raw(), &segs);
                exec(q.as_raw(), segs.len() as u32, segs.as_ptr());
                super::super::alloc_pool::executed(q.as_raw(), &segs);
                (segs, again)
            };

            q.Wait(&gate, 1).unwrap();
            let (first, _) = submit();
            assert_eq!(first, vec![primary]);
            assert_eq!(reset(w, a, 0, first[0]), 0);
            let (second, again) = submit();
            assert_eq!(reset(w, a, 0, second[0]), 0);
            let (third, again3) = submit();
            // the first stand-in is a spare now, its submit still on the GPU
            assert_eq!(reset(w, a, 0, third[0]), 0);
            let (more, again4) = submit();
            gate.Signal(1).unwrap();
            q.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, &dev, None, 5000).expect("the queue finishes");
            assert_eq!(reset(w, a, 0, more[0]), 0);
            let (fourth, _) = submit();
            q.Signal(&done, 2).unwrap();
            crate::gpu::fence::wait(&done, 2, &dev, None, 5000).expect("the queue finishes");
            let removed = dev.GetDeviceRemovedReason();
            assert_eq!(release(w), 0);

            let how = if newer_layout { "Reset #139" } else { "Reset #10" };
            assert_eq!(second.len(), 1);
            assert_ne!(second[0], primary, "{how}: the new recording must not be on the PC list still on the GPU");
            assert_eq!(again, 0, "{how}: no PC list is submitted while its last submit runs");
            assert_eq!(third.len(), 1);
            assert!(third[0] != primary && third[0] != second[0], "{how}: nor on the one standing in, also still on the GPU");
            assert_eq!(again3, 0, "{how}");
            assert_eq!(more.len(), 1);
            assert!(![primary, second[0], third[0]].contains(&more[0]), "{how}: nor on a spare whose submit is still on the GPU");
            assert_eq!(again4, 0, "{how}");
            assert_eq!(fourth, vec![primary], "{how}: once the GPU is done the list records on its own PC list");
            assert!(removed.is_ok());
        }
    }

    #[test]
    fn a_gpu_wait_does_not_expire_while_the_title_is_held_for_a_build() {
        let old = Instant::now() - gpu_wait_watchdog_timeout() * 2;
        let mut aw = ArmedGpuWait { val: 1, waiting_queue: 0, waiting_queue_ty: 0, wait_seq: 1, armed_at: old, after_write64: false };
        assert!(!wait_expired(&mut aw, true));
        assert!(!wait_expired(&mut aw, false), "its clock starts again when the hold ends");
        aw.armed_at = old;
        assert!(wait_expired(&mut aw, false));
    }

    #[test]
    fn an_elapsed_gpu_wait_does_not_synthesize_fence_completion() {
        let old = Instant::now() - gpu_wait_watchdog_timeout() * 2;
        let mut cell = SyncCell {
            cpu_signaled_max: 3,
            submitted_max: 9,
            armed_waits: vec![ArmedGpuWait {
                val: 9,
                waiting_queue: 0x7000,
                waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0,
                wait_seq: 2,
                armed_at: old,
                after_write64: false,
            }],
            submitted_signals: vec![SubmittedSignal { val: 9, queue: 0x7100, seq: 1 }],
            ..Default::default()
        };

        let expired = take_expired_gpu_wait(&mut cell, false).expect("the elapsed wait");
        assert_eq!(expired, ExpiredGpuWait {
            val: 9,
            waiting_queue: 0x7000,
            waiting_queue_ty: D3D12_COMMAND_LIST_TYPE_COMPUTE.0,
        });
        assert!(cell.armed_waits.is_empty());
        assert_eq!(cell.cpu_signaled_max, 3, "watchdog expiry is not a CPU signal");
        assert_eq!(cell.submitted_signals.len(), 1, "the real producer remains authoritative");
        assert_eq!(cell.submitted_signals[0].val, 9);
    }

    #[test]
    fn a_dispatch_with_a_pipeline_still_building_waits_at_the_submit() {
        use super::super::{bind, pso_async, title_list};
        use std::time::{Duration, Instant};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp = CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>());
                if warp.and_then(|w| D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev)).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let raw: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &alloc, None).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let standin = 0x7E57_D001usize;
            static BUILD: AtomicBool = AtomicBool::new(false);
            pso_async::submit(
                standin,
                true,
                Box::new(|| {
                    let t = Instant::now();
                    while !BUILD.load(Ordering::Acquire) && t.elapsed() < Duration::from_secs(10) {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    0
                }),
            );
            let pc0 = raw.into_raw();
            real::adopt(pc0, real::Kind::CommandList);
            on_create(pc0, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            bind::reset(pc0, alloc.as_raw() as usize, standin);
            let wrap = title_list::wrap(pc0);

            const DISPATCHES: usize = 210;
            let mut segs = vec![pc0];
            for n in 1..=DISPATCHES {
                let t = Instant::now();
                bind::dispatch(*segs.last().unwrap(), 1, 1, 1);
                assert!(t.elapsed() < Duration::from_millis(700), "dispatch {n} waited for the build ({:?})", t.elapsed());
                segs.clear();
                assert!(title_list::push_segments(wrap, &mut segs));
            }
            assert_eq!(segs.len(), 1 + 2 * DISPATCHES, "the list before each dispatch, the dispatch, the list after it");
            assert!(pso_async::is_pending(standin));

            let close_fn: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(*(*(wrap as *const *const usize)).add(9));
            assert_eq!(close_fn(wrap), 0);
            assert!(pso_async::is_pending(standin), "Close waited for the build");

            let skipped = bind::counts_snapshot()[4];
            let build = std::thread::spawn(|| {
                std::thread::sleep(Duration::from_millis(300));
                BUILD.store(true, Ordering::Release);
            });
            record_deferred(wrap);
            assert!(!pso_async::is_pending(standin), "the submit waits for the build");
            assert!(bind::counts_snapshot()[4] > skipped, "the dispatch of a build that failed is skipped, at the submit");
            build.join().unwrap();

            type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
            let exec: ExecFn = std::mem::transmute(*(*(queue.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
            exec(queue.as_raw(), segs.len() as u32, segs.as_ptr());
            queue.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, &dev, None, 2000).expect("the split list runs");

            let rel_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(wrap as *const *const usize)).add(2));
            assert_eq!(rel_fn(wrap), 0);
            let _ = pso_async::released(standin);
        }
    }

    /// A direct queue, its allocator and a title list wrapped as XCL wraps one, bound to `standin`
    /// (None: no adapter).
    unsafe fn put_off_rig(standin: usize) -> Option<(ID3D12Device, ID3D12CommandQueue, ID3D12CommandAllocator, *mut c_void, *mut c_void)> {
        use super::super::{bind, title_list};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            let warp = CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>());
            if warp.and_then(|w| D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev)).is_err() {
                return None;
            }
        }
        let dev = dev.unwrap();
        let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
        let raw: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
        let pc0 = raw.into_raw();
        real::adopt(pc0, real::Kind::CommandList);
        on_create(pc0, alloc.as_raw() as usize, 0, ty);
        bind::reset(pc0, alloc.as_raw() as usize, standin);
        let wrap = title_list::wrap(pc0);
        Some((dev, queue, alloc, pc0, wrap))
    }

    /// A put-off call whose recording the driver refuses at the submit (its segment's Close
    /// fails) is not submitted as it is: the segment goes out empty, the call is dropped, and the
    /// rest of the submission runs (its fence is signalled, the device stays).
    #[test]
    fn a_put_off_call_whose_close_fails_is_not_submitted() {
        use super::super::{bind, pso_async, title_list};
        use std::time::{Duration, Instant};
        unsafe {
            let standin = 0x7E57_C105usize;
            static BUILD: AtomicBool = AtomicBool::new(false);
            BUILD.store(false, Ordering::Release);
            pso_async::submit(standin, true, Box::new(|| {
                let t = Instant::now();
                while !BUILD.load(Ordering::Acquire) && t.elapsed() < Duration::from_secs(10) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                0
            }));
            let Some((dev, queue, _alloc, pc0, wrap)) = put_off_rig(standin) else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            bind::dispatch(pc0, 1, 1, 1);
            let mut segs = Vec::new();
            assert!(title_list::push_segments(wrap, &mut segs));
            assert_eq!(segs.len(), 3, "the dispatch was put off");

            // the replay's recording fails in the runtime: a query begun and never ended
            let mut heap: Option<ID3D12QueryHeap> = None;
            dev.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_OCCLUSION, Count: 1, NodeMask: 0 }, &mut heap).unwrap();
            let heap = heap.unwrap();
            let h = heap.as_raw() as usize;
            static REFUSED: AtomicU32 = AtomicU32::new(0);
            CLOSE_HOOK.lock().unwrap().push((wrap as usize, Box::new(move |p| {
                pc::<QueryFn>(p, PC_BEGIN_QUERY)(p, h, D3D12_QUERY_TYPE_OCCLUSION, 0);
                let hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(p, PC_CLOSE)(p);
                REFUSED.store(hr as u32, Ordering::Release);
                hr
            })));
            let close_fn: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(*(*(wrap as *const *const usize)).add(9));
            assert_eq!(close_fn(wrap), 0);
            BUILD.store(true, Ordering::Release);
            record_deferred(wrap);
            assert!(!CLOSE_HOOK.lock().unwrap().iter().any(|h| h.0 == wrap as usize), "the replay closed its segment");
            assert!((REFUSED.load(Ordering::Acquire) as i32) < 0, "the runtime refused the recording ({:#x})", REFUSED.load(Ordering::Acquire));
            // what the submit collects after the replay (`q_execute_command_lists`)
            segs.clear();
            assert!(title_list::push_segments(wrap, &mut segs));
            assert_eq!(segs.len(), 2, "the refused segment is not submitted");

            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
            let exec: ExecFn = std::mem::transmute(*(*(queue.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
            exec(queue.as_raw(), segs.len() as u32, segs.as_ptr());
            queue.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, &dev, None, 2000).expect("the submission runs");
            assert!(dev.GetDeviceRemovedReason().is_ok(), "the device stays");

            let rel_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(wrap as *const *const usize)).add(2));
            rel_fn(wrap);
            let _ = pso_async::released(standin);
        }
    }

    /// A put-off segment that does not close when it is made is not used: the call waits in its
    /// list instead of being put off.
    #[test]
    fn a_put_off_segment_that_does_not_close_is_not_used() {
        use super::super::{bind, pso_async, title_list};
        unsafe {
            let standin = 0x7E57_C106usize;
            static BUILD: AtomicBool = AtomicBool::new(false);
            BUILD.store(false, Ordering::Release);
            pso_async::submit(standin, true, Box::new(|| {
                let t = std::time::Instant::now();
                while !BUILD.load(Ordering::Acquire) && t.elapsed() < std::time::Duration::from_secs(10) {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                0
            }));
            let Some((_dev, _queue, _alloc, pc0, wrap)) = put_off_rig(standin) else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            CLOSE_HOOK.lock().unwrap().push((wrap as usize, Box::new(|p| {
                pc::<unsafe extern "system" fn(*mut c_void) -> i32>(p, PC_CLOSE)(p);
                windows::Win32::Foundation::E_FAIL.0
            })));
            // (the call that is not put off waits for the build in the call; started here, after
            // the device is made, so the build is still going when the call comes)
            let ends = std::thread::spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(300));
                BUILD.store(true, Ordering::Release);
            });
            bind::dispatch(pc0, 1, 1, 1);
            ends.join().unwrap();
            assert!(!CLOSE_HOOK.lock().unwrap().iter().any(|h| h.0 == wrap as usize), "the put-off segment was closed");
            let mut segs = Vec::new();
            assert!(title_list::push_segments(wrap, &mut segs));
            assert_eq!(segs.len(), 1, "nothing was put off");
            let rel_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(wrap as *const *const usize)).add(2));
            rel_fn(wrap);
            let _ = pso_async::released(standin);
        }
    }

    /// On the software adapter (the test above opens the default one): two dispatches put off on
    /// a title list are recorded at its submit, on the allocator kept on the list's primary PC
    /// list, and wait there for the build without being put off again; the segments run.
    #[test]
    fn put_off_dispatches_wait_at_the_submit_and_run_on_the_software_adapter() {
        use super::super::{bind, pso_async, title_list};
        use std::time::Duration;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            let warp = CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>());
            if warp.and_then(|w| D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev)).is_err() {
                crate::gpu::recomp::runner::skip_test("no WARP adapter");
                return;
            }
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let raw: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &alloc, None).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            let standin = 0x7E57_D101usize;
            pso_async::submit(
                standin,
                true,
                Box::new(|| {
                    std::thread::sleep(Duration::from_millis(500));
                    0
                }),
            );
            let pc0 = raw.into_raw();
            real::adopt(pc0, real::Kind::CommandList);
            on_create(pc0, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            bind::reset(pc0, alloc.as_raw() as usize, standin);
            let wrap = title_list::wrap(pc0);

            let mut segs = vec![pc0];
            for _ in 0..2 {
                bind::dispatch(*segs.last().unwrap(), 1, 1, 1);
                segs.clear();
                assert!(title_list::push_segments(wrap, &mut segs));
            }
            assert_eq!(segs.len(), 5, "the list before each dispatch, the dispatch, the list after it");
            let close_fn: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(*(*(wrap as *const *const usize)).add(9));
            assert_eq!(close_fn(wrap), 0);

            assert!(pso_async::is_pending(standin));
            let refused = pso_async::put_off_count(pso_async::PutOff::Refused);
            record_deferred(wrap);
            assert!(!pso_async::is_pending(standin), "the submit waits for the build");
            assert_eq!(pso_async::put_off_count(pso_async::PutOff::Refused), refused, "a dispatch the submit records is not put off again: it waits there");
            let own = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(pc0 as usize))?.own_alloc.clone());
            assert!(own.is_some(), "the list's own allocator is kept on its primary PC list");

            type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
            let exec: ExecFn = std::mem::transmute(*(*(queue.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
            exec(queue.as_raw(), segs.len() as u32, segs.as_ptr());
            queue.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, &dev, None, 2000).expect("the split list runs");

            let rel_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(wrap as *const *const usize)).add(2));
            assert_eq!(rel_fn(wrap), 0);
            let _ = pso_async::released(standin);
        }
    }

    /// Reproduces the exact intra-list fork-join pattern seen on retail compute queues:
    ///   q_compute list O_c: Write64(cell_A, frame) -> Wait64(cell_B >= frame) -> Write64(cell_C, frame)
    ///   q_direct  list O_d: Wait64(cell_A >= frame) -> Write64(cell_B, frame)
    /// Without deferred mid-list splitting, hoisting Wait64(cell_B) ahead of Write64(cell_A) creates
    /// a 2-queue deadlock cycle every frame. With deferred splitting, O_c expands into `[pc_c0, pc_c1]`,
    /// completes on GPU with ZERO CPU cycle breaks, and reuses `pc_c1` from the spare pool on Reset.
    #[test]
    fn mid_list_write64_wait64_split_avoids_cycle_and_reuses_spares() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q_comp: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let q_dir: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let a_comp: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let a_dir: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let raw_c: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &a_comp, None).unwrap();
            let raw_d: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a_dir, None).unwrap();
            let done_c: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done_d: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            // Transfer raw_c into `title_list::wrap` (the wrapper owns the COM ref).
            let pc_c0 = raw_c.into_raw();
            real::adopt(pc_c0, real::Kind::CommandList);
            on_create(pc_c0, a_comp.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE);
            super::super::bind::reset(pc_c0, a_comp.as_raw() as usize, 0);
            let wrap_c = super::super::title_list::wrap(pc_c0);

            let pc_d0 = raw_d.as_raw();
            real::adopt(pc_d0, real::Kind::CommandList);
            on_create(pc_d0, a_dir.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);

            let cell_a = Box::new(AtomicU64::new(0));
            let cell_b = Box::new(AtomicU64::new(0));
            let cell_c = Box::new(AtomicU64::new(0));
            let va_a = &*cell_a as *const AtomicU64 as u64;
            let va_b = &*cell_b as *const AtomicU64 as u64;
            let va_c = &*cell_c as *const AtomicU64 as u64;

            let mut first_seg1: *mut c_void = std::ptr::null_mut();
            for frame in 1u64..=2 {
                if frame > 1 {
                    a_comp.Reset().unwrap();
                    a_dir.Reset().unwrap();
                    // Reset through `title_list` slot #10: must switch back to `pc_c0` and keep `seg1` as spare.
                    let reset_fn: unsafe extern "system" fn(*mut c_void, usize, usize) -> i32 =
                        std::mem::transmute(*(*(wrap_c as *const *const usize)).add(10));
                    assert_eq!(reset_fn(wrap_c, a_comp.as_raw() as usize, 0), 0);
                    assert_eq!(reset_x(pc_d0, a_dir.as_raw() as usize, 0), 0);
                }

                // Record O_d: Wait64(A >= frame) -> Write64(B, frame) -> Close (seed cell_B recorded
                // first so O_c's Wait64(B) bridges to GPU even though O_c submits before O_d).
                note_write64_recorded(pc_c0, va_a, frame);
                let mut g = SYNC_GRAPH.lock().unwrap();
                let cell_b_entry = g.get_or_insert_with(SyncGraph::default).cells.entry(va_b).or_default();
                cell_b_entry.recorded_max = cell_b_entry.recorded_max.max(frame);
                cell_b_entry.recorded_list_type = Some(D3D12_COMMAND_LIST_TYPE_DIRECT);
                drop(g);
                // Call wait64_x on the active PC segment (`pc_c0`); it looks up `wrap_c`, splits into
                // `[pc_c0, seg1]`, and switches `wrap_c`'s `F_BACKING` to `seg1`.
                wait64_x(pc_c0, va_b, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, frame, 0, 0);

                // Verify the split created/reused a second segment.
                let mut segs_c = Vec::new();
                assert!(super::super::title_list::push_segments(wrap_c, &mut segs_c));
                assert_eq!(segs_c.len(), 2, "frame {frame}: mid-list Wait64 after Write64 must split into 2 segments");
                assert_eq!(segs_c[0], pc_c0);
                assert_ne!(segs_c[1], pc_c0);
                if frame == 1 {
                    first_seg1 = segs_c[1];
                } else {
                    assert_eq!(segs_c[1], first_seg1, "frame 2 split must reuse spare PC command list");
                }

                // Write64(C, frame) lands on the second segment.
                note_write64_recorded(segs_c[1], va_c, frame);

                // Close O_c via title_list slot #9 (closes segs_c[1]; segs_c[0] was already closed on split).
                let close_fn: unsafe extern "system" fn(*mut c_void) -> i32 =
                    std::mem::transmute(*(*(wrap_c as *const *const usize)).add(9));
                assert_eq!(close_fn(wrap_c), 0, "frame {frame}: Close on split title_list failed");

                // Record O_d: Wait64(A >= frame) -> Write64(B, frame) -> Close.
                wait64_x(pc_d0, va_a, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, frame, 0, 0);
                note_write64_recorded(pc_d0, va_b, frame);
                let cl_d_raw = ID3D12GraphicsCommandList::from_raw_borrowed(&pc_d0).unwrap();
                cl_d_raw.Close().unwrap();

                // Execute O_c segments on q_comp FIRST (just like the real title!).
                execute_with_waits_on(q_comp.as_raw(), &segs_c, |chunk| {
                    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
                    let exec: ExecFn = std::mem::transmute(*(*(q_comp.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
                    exec(q_comp.as_raw(), chunk.len() as u32, chunk.as_ptr());
                    signal_write64_fences(q_comp.as_raw(), chunk);
                    if chunk.contains(&pc_c0) {
                        cell_a.store(frame, Ordering::Release);
                    }
                });
                q_comp.Signal(&done_c, frame).unwrap();

                // Execute O_d on q_dir SECOND.
                let one_d = [pc_d0];
                execute_with_waits_on(q_dir.as_raw(), &one_d, |chunk| {
                    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
                    let exec: ExecFn = std::mem::transmute(*(*(q_dir.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
                    exec(q_dir.as_raw(), chunk.len() as u32, chunk.as_ptr());
                    signal_write64_fences(q_dir.as_raw(), chunk);
                    cell_b.store(frame, Ordering::Release);
                });
                q_dir.Signal(&done_d, frame).unwrap();

                crate::gpu::fence::wait(&done_c, frame, &dev, None, 2000).expect("compute queue must complete without cycle");
                crate::gpu::fence::wait(&done_d, frame, &dev, None, 2000).expect("direct queue must complete without cycle");

                // CRITICAL: verify ZERO CPU cycle breaks occurred on ANY of the 3 cells!
                let g = SYNC_GRAPH.lock().unwrap();
                let gr = g.as_ref().unwrap();
                for (name, va) in [("cell_A", va_a), ("cell_B", va_b), ("cell_C", va_c)] {
                    let c = gr.cells.get(&va).unwrap();
                    assert_eq!(c.cpu_signaled_max, 0, "frame {frame}: {name} must NOT trigger a CPU cycle break");
                    assert_eq!(c.submitted_max, frame, "frame {frame}: {name} must reach frame {frame} on GPU");
                }
            }

            // Release wrapped list (must release primary + spare segment cleanly).
            let rel_fn: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(*(*(wrap_c as *const *const usize)).add(2));
            assert_eq!(rel_fn(wrap_c), 0);
            on_release(pc_d0 as usize);
            if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
                gr.cells.remove(&va_a);
                gr.cells.remove(&va_b);
                gr.cells.remove(&va_c);
            }
        }
    }

    /// #1452: `KickoffX` on a split list submits its segments and moves them to the spares; the
    /// GPU still runs them (held here by a queue Wait on a fence not yet signalled). The next
    /// split of the recording that goes on must not record on the segment just submitted (the PC
    /// forbids submitting it again before that submit is done, debug layer id 553).
    #[test]
    fn a_split_after_kickoff_does_not_take_the_segment_just_submitted() {
        let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
            return crate::gpu::recomp::runner::skip_test("no GPU");
        };
        unsafe {
            let ty = D3D12_COMMAND_LIST_TYPE_DIRECT;
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            // (`KickoffX` submits through the queue's original ExecuteCommandLists)
            real::adopt(q.as_raw(), real::Kind::CommandQueue);
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let a = alloc.as_raw() as usize;
            let primary = dev.CreateCommandList::<_, _, ID3D12GraphicsCommandList>(0, ty, &alloc, None).unwrap().into_raw();
            real::adopt(primary, real::Kind::CommandList);
            on_create(primary, a, 0, ty);
            super::super::bind::reset(primary, a, 0);
            let w = super::super::title_list::wrap(primary);
            note_queue_for_lists(q.as_raw() as usize, &[primary]);
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            // a write, then a wait on another cell, splits the list; the waited-for memory holds
            // the value already, so the waits pass on the CPU
            let cell_a = Box::new(AtomicU64::new(0));
            let cell_b = Box::new(AtomicU64::new(5));
            let (va_a, va_b) = (&*cell_a as *const AtomicU64 as u64, &*cell_b as *const AtomicU64 as u64);
            let split = |list: *mut c_void, frame: u64| {
                note_write64_recorded(list, va_a, frame);
                wait64_x(list, va_b, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, frame, 0, 0);
                let mut segs = Vec::new();
                assert!(super::super::title_list::push_segments(w, &mut segs));
                segs
            };

            q.Wait(&gate, 1).unwrap();
            let first = split(primary, 1);
            kickoff_x(first[first.len() - 1]);
            let second = split(primary, 2);
            gate.Signal(1).unwrap();
            q.Signal(&done, 1).unwrap();
            let ran = crate::gpu::fence::wait(&done, 1, &dev, None, 5000).is_ok();
            let removed = dev.GetDeviceRemovedReason();
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(w as *const *const usize)).add(2));
            assert_eq!(release(w), 0);
            if let Some(gr) = SYNC_GRAPH.lock().unwrap().as_mut() {
                gr.cells.remove(&cpu_at(va_a));
                gr.cells.remove(&cpu_at(va_b));
            }

            assert_eq!(first.len(), 2, "the wait after the write splits the list");
            assert_eq!(first[0], primary);
            assert_eq!(second.len(), 2, "the recording after KickoffX splits again");
            assert_eq!(second[0], primary, "after KickoffX the recording goes on on the primary PC list");
            assert_ne!(second[1], first[1], "the segment KickoffX just submitted is still on the GPU");
            assert!(ran, "the queue finishes");
            assert!(removed.is_ok());
        }
    }

    /// A buffer outside unified memory that has a mirror is bound at its mirror again without
    /// asking whether the title's memory is readable: the source made unreadable after the first
    /// bind still binds the same mirror (before, the bind asked `readable` first and gave the
    /// title's address back, which the PC's GPU has no address for).
    #[test]
    fn a_mirrored_buffer_binds_its_mirror_without_asking_readable() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualProtect, MEM_COMMIT, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE};
        if crate::gpu::uma::get().is_none() {
            crate::gpu::recomp::runner::skip_test("no unified memory");
            return;
        }
        unsafe {
            let len = 0x1_0000usize;
            let src = VirtualAlloc(std::ptr::null(), len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as *mut u8;
            assert!(!src.is_null());
            for k in 0..len {
                *src.add(k) = k as u8;
            }
            let (va, bytes) = (src as u64 + 0x100, 0x400usize);
            let first = gpu_va(std::ptr::null_mut(), va, bytes, "test vertex buffer", true);
            assert_ne!(first, va, "the first bind makes a mirror");
            assert!(mirrored(va, bytes).is_some());
            let mut old = 0u32;
            assert_ne!(VirtualProtect(src as *const _, len, PAGE_NOACCESS, &mut old), 0);
            assert!(!crate::gpu::shader::readable(va as usize, bytes));
            assert_eq!(gpu_va(std::ptr::null_mut(), va, bytes, "test vertex buffer", true), first);
            // the range stays reserved (and unreadable): the mirror table is never emptied, so its
            // address must not come back for another test's memory
        }
    }

    /// Unified memory with the 16-window layout of its own (the process's is chosen once, from
    /// the environment): its view windows hold GPU views.
    fn sixteen_windows() -> crate::gpu::uma::Uma {
        crate::gpu::uma::Uma::for_runtime_windows(16 << 20, 16 << 20, crate::gpu::uma::MAX_HEAP_WINDOWS).expect("16-window aperture")
    }

    /// An index or vertex buffer in a GPU view goes to the PC as its view window's address; without
    /// one it goes as none, never as the console address (no PC address: the device hangs).
    #[test]
    fn a_gpu_view_buffer_reaches_the_pc_at_its_windows_address() {
        let u = sixteen_windows();
        let g = u.view_reserve(1 << 20).expect("a GPU view");
        let k = u.window_of(g).unwrap();
        assert!(u.is_view_window(k));
        let window_va = |w: usize| Some(0x7A00_0000_0000 + w as u64 * crate::gpu::uma::WINDOW as u64);
        let at = g as u64 + 0x40;
        assert_eq!(pc_va_in(&u, at, window_va), window_va(k).unwrap() + (g - u.window(k).0) as u64 + 0x40);
        assert_eq!(pc_va_in(&u, at, |_| None), 0, "a view window that is not open on the GPU");
        // in a view window but no GPU view of the title's: an address by number, as it is
        let other = g as u64 + (2 << 20);
        assert_eq!(pc_va_in(&u, other, window_va), other);
        assert_eq!(pc_va_in(&u, 0x1234_0000, window_va), 0x1234_0000);
        assert_eq!(u.view_release(g), Some(1 << 20));
    }

    /// A vertex or index buffer of a GPU view across a view window's end is bound only up to that
    /// end: the windows are separate buffers on the PC. One inside a window stays as it is.
    #[test]
    fn a_view_across_a_window_end_is_bound_up_to_it() {
        use crate::gpu::uma::{GRAIN, WINDOW};
        let u = sixteen_windows();
        let g = u.view_reserve(WINDOW + GRAIN).expect("a GPU view over two windows");
        let end = (g + WINDOW) as u64;
        assert_eq!(bytes_in_window(&u, end - 0x100, 0x200), Some(0x100), "across the end");
        assert_eq!(bytes_in_window(&u, end - 0x200, 0x200), None, "up to the end");
        assert_eq!(bytes_in_window(&u, g as u64 + 0x40, 0x1000), None, "inside");
        assert_eq!(bytes_in_window(&u, end + 0x10, 0x100), None, "in the next window");
        assert_eq!(bytes_in_window(&u, 0x1234_0000, 0x1000_0000), None, "not in a window");
        assert_eq!(u.view_release(g), Some(WINDOW + GRAIN));
    }

    /// A wait on an address in a GPU view reads the value at its CPU view, where the title writes
    /// it and what the lists write into the view's tiles is read back; a write and a wait naming
    /// either view are kept in one sync cell. Without view windows nothing changes.
    #[test]
    fn a_wait_on_a_gpu_view_reads_its_cpu_view() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        // a size no other test pairs views of (the newest CPU view of a size takes its GPU view)
        const SIZE: usize = 0x2D_0000;
        let u = sixteen_windows();
        let span = super::super::bind::view_span_of(&u);
        let g = u.view_reserve(SIZE).expect("a GPU view");
        unsafe {
            let c = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert_ne!(c, 0);
            crate::hle::kernelx::mirrors::reserved(c, SIZE, 0x2040_2000, PAGE_READWRITE);
            crate::hle::kernelx::mirrors::reserved(g, SIZE, 0x2040_2000, 0x801);
            let x = 0x1_0008u64;
            let at = cpu_at_in(span, g as u64 + x);
            assert_eq!(at, c as u64 + x, "the CPU view");
            assert_eq!(cpu_at_in(span, c as u64 + x), at, "a write at the CPU view and a wait at the GPU view share their cell");
            ((c as u64 + x) as *mut u64).write(0x5843_4C00_0000_0042);
            let w = PendingWait { va: at, width: 8, comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, value: 0x5843_4C00_0000_0042, mask: u64::MAX, after_write64: false };
            assert_eq!(w.read_value(), Some(0x5843_4C00_0000_0042));
            assert!(w.is_satisfied());
            assert!(!PendingWait { value: 0x5843_4C00_0000_0043, ..w }.is_satisfied());
            // the GPU view is no memory the CPU reads: a wait kept there would read nothing (and pass)
            assert_eq!(PendingWait { va: g as u64 + x, ..w }.read_value(), None);
            // layouts without view windows: every address is its own
            assert_eq!(cpu_at_in((0, 0), g as u64 + x), g as u64 + x);
            crate::hle::kernelx::mirrors::released(g);
            crate::hle::kernelx::mirrors::released(c);
            assert_eq!(cpu_at_in(span, g as u64 + x), g as u64 + x, "unpaired");
            VirtualFree(c as *mut c_void, 0, MEM_RELEASE);
        }
        assert_eq!(u.view_release(g), Some(SIZE));
    }

    #[test]
    fn occlusion_tracks_dropped_draws_and_active_counter() {
        let mut occ = Occlusion::default();
        // Clean query with no dropped draws: forwards PC EndQuery and keeps true GPU count.
        assert_eq!(occ.begin((1, 0), D3D12_QUERY_TYPE_OCCLUSION), None);
        assert!(ACTIVE_OCCLUSION_LISTS.load(Ordering::Relaxed) >= 1);
        assert_eq!(occ.end((1, 0)), (true, false));

        // Query that sees a dropped/pending draw: forwards PC EndQuery AND forces visible.
        assert_eq!(occ.begin((1, 1), D3D12_QUERY_TYPE_OCCLUSION), None);
        occ.mark_dropped_draw();
        assert_eq!(occ.end((1, 1)), (true, true));

        // Nested query: outer ends early (force visible, no second PC EndQuery), inner counts cleanly.
        assert_eq!(occ.begin((1, 2), D3D12_QUERY_TYPE_OCCLUSION), None);
        assert_eq!(occ.begin((1, 3), D3D12_QUERY_TYPE_BINARY_OCCLUSION), Some(((1, 2), D3D12_QUERY_TYPE_OCCLUSION)));
        assert!(ACTIVE_OCCLUSION_LISTS.load(Ordering::Relaxed) >= 1);
        assert_eq!(occ.end((1, 3)), (true, false));
        assert_eq!(occ.end((1, 2)), (false, true));

        // End-to-end `note_dropped_draw` fast-path + LISTS integration.
        let fake_list = 0xDEAD_BEE0usize as *mut c_void;
        LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(fake_list as usize).or_default().occlusion.begin((7, 42), D3D12_QUERY_TYPE_OCCLUSION);
        note_dropped_draw(fake_list);
        let res = LISTS.lock().unwrap().as_mut().unwrap().remove(&(fake_list as usize)).unwrap().occlusion.end((7, 42));
        assert_eq!(res, (true, true));
    }

    #[test]
    fn default_bound_heaps_always_includes_sampler_heap_when_available() {
        assert_eq!(default_bound_heaps(Some(0x10), Some(0x20), &[], false), Some((vec![0x10, 0x20], true)));
        assert_eq!(default_bound_heaps(None, Some(0x20), &[], false), Some((vec![0x20], true)));
        assert_eq!(default_bound_heaps(Some(0x10), None, &[], false), Some((vec![0x10], false)));
        // Shared view heap bound alone -> append shared sampler heap
        assert_eq!(default_bound_heaps(Some(0x10), Some(0x20), &[0x10], false), Some((vec![0x10, 0x20], true)));
        // Non-global view heap bound alone, or heap_windows() == false (view == None) -> still append shared sampler heap
        assert_eq!(default_bound_heaps(Some(0x10), Some(0x20), &[0x77], false), Some((vec![0x77, 0x20], true)));
        assert_eq!(default_bound_heaps(None, Some(0x20), &[0x77], false), Some((vec![0x77, 0x20], true)));
        // Sampler heap bound alone when heap_windows() needs the shared view heap -> prepend view heap
        assert_eq!(default_bound_heaps(Some(0x10), Some(0x20), &[0x99], true), Some((vec![0x10, 0x99], true)));
        assert_eq!(default_bound_heaps(None, Some(0x20), &[0x99], true), None);
        // Both heaps already bound -> no-op
        assert_eq!(default_bound_heaps(Some(0x10), Some(0x20), &[0x10, 0x20], true), None);
        assert_eq!(default_bound_heaps(Some(0x10), Some(0x20), &[0x10, 0x99], true), None);
    }

    /// ClearUnorderedAccessView* handles: two console slots → their PC UAVs, two PC handles as
    /// they are, anything else not usable.
    #[test]
    fn uav_clear_handles_translate_console_slots() {
        use super::super::console_heap::Slot;
        let s = |pc_uav: u64, pc_gpu_uav: u64, sampler: bool| Some(Slot { generation: 0, at: 0x1000, index: 1, pc_srv: 0, pc_uav, pc_gpu_uav, sampler, shared_srv: 0, k: 0, remaining: 1 });
        assert_eq!(uav_clear_handles(s(0x10, 0x5010, false), s(0x20, 0, false)), UavClearHandles::Console { gpu: 0x5010, cpu: 0x20 });
        assert_eq!(uav_clear_handles(None, None), UavClearHandles::Pc);
        // a console slot beside a PC handle, a GPU slot of a heap that is not shader visible,
        // sampler slots
        assert_eq!(uav_clear_handles(s(0x10, 0x5010, false), None), UavClearHandles::Unusable);
        assert_eq!(uav_clear_handles(None, s(0x20, 0, false)), UavClearHandles::Unusable);
        assert_eq!(uav_clear_handles(s(0x10, 0, false), s(0x20, 0, false)), UavClearHandles::Unusable);
        assert_eq!(uav_clear_handles(s(0x10, 0x5010, false), s(0, 0, true)), UavClearHandles::Unusable);
        assert_eq!(uav_clear_handles(s(0, 0, true), s(0x20, 0, false)), UavClearHandles::Unusable);
    }

    /// Console code clears a UAV with its console heap slots (heap #9 / #10 give it their
    /// addresses): the clear reaches the PC with the slots' UAVs. Before, the slot addresses
    /// went to the PC method as handles. On WARP (a software adapter).
    #[test]
    fn clear_uav_with_console_handles_clears_the_slots_view() {
        use super::super::console_heap;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        if crate::gpu::uma::get().is_none() {
            crate::gpu::recomp::runner::skip_test("no unified memory");
            return;
        }
        unsafe {
            let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                crate::gpu::recomp::runner::skip_test("no WARP");
                return;
            };
            let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                crate::gpu::recomp::runner::skip_test("no WARP");
                return;
            };
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no WARP");
                return;
            }
            let dev = dev.unwrap();
            const N: u32 = 64;
            let desc = |flags| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4 * N as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let committed = |ty, flags, state| {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: ty, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc(flags), state, None, &mut r).unwrap();
                r.unwrap()
            };
            let buf = committed(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            let rb = committed(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            // the title's heaps of one view slot each (SRV, UAV): a CPU-only one and a shader-visible one
            let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let heap = |flags| -> ID3D12DescriptorHeap {
                dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 2, Flags: flags, NodeMask: 0 }).unwrap()
            };
            let (cpu_heap, gpu_heap) = (heap(D3D12_DESCRIPTOR_HEAP_FLAG_NONE), heap(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE));
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { NumElements: N, ..Default::default() } },
            };
            for h in [&cpu_heap, &gpu_heap] {
                let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.GetCPUDescriptorHandleForHeapStart().ptr + inc as usize };
                dev.CreateUnorderedAccessView(&buf, None, Some(&uav), at);
            }
            let (co, go) = (cpu_heap.as_raw() as usize, gpu_heap.as_raw() as usize);
            assert!(console_heap::register(co, 1, cpu_heap.GetCPUDescriptorHandleForHeapStart().ptr as u64, 0, inc, false, false, false));
            assert!(console_heap::register(go, 1, gpu_heap.GetCPUDescriptorHandleForHeapStart().ptr as u64, gpu_heap.GetGPUDescriptorHandleForHeapStart().ptr, inc, false, false, true));
            let (console_cpu, console_gpu) = (console_heap::start(co).unwrap(), console_heap::start(go).unwrap());

            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).unwrap();
            l.SetDescriptorHeaps(&[Some(gpu_heap.clone())]);
            // the title's call (#53) on the adopted list, then the PC vtable back
            let raw = l.as_raw();
            real::adopt(raw, real::Kind::CommandList);
            clear_uav_uint_x(raw, console_gpu, console_cpu, buf.as_raw() as usize, [0xDEAD_BEEFu32; 4].as_ptr(), 0, std::ptr::null());
            let ext = *(raw as *const *const usize);
            *(raw as *mut usize) = *ext.sub(1);
            let b = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(buf.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            };
            l.ResourceBarrier(&[b]);
            l.CopyResource(&rb, &buf);
            l.Close().unwrap();
            q.ExecuteCommandLists(&[Some(l.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            q.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, &dev, None, 5000).expect("the clear completes");
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).unwrap();
            let got = std::slice::from_raw_parts(p as *const u32, N as usize).to_vec();
            rb.Unmap(0, None);
            console_heap::unregister(co);
            console_heap::unregister(go);
            assert!(got.iter().all(|&v| v == 0xDEAD_BEEF), "the UAV was not cleared: {:x?}", &got[..4]);
        }
    }
}
