//! Presentation for the real backend.
//!
//! Console titles do not use DXGI swap chains: they render into their own back buffers
//! and call `PresentX` – command queue slot 30, `PresentX(planeCount, planes, parameters)`
//! with plane 0 `{ u64 frame token; u32 resource count @8; ID3D12Resource** @0x10 }` (as
//! observed on Fortnite). XCL presents that resource in its own window at once and tells the
//! frame clock (`frame`) the token flipped.
//!
//! Until a title's first `PresentX`, a fallback guesses the frame: XCL remembers which
//! resource every RTV descriptor points to and the first render target of each command list,
//! and after ExecuteCommandLists presents the latest back-buffer-like render target – on a
//! call of another console-only queue method, or rate-limited after submissions.
//!
//! The back buffer is copied on the title's own queue (implicit promotion from
//! COMMON/PRESENT state, which consoles require before PresentX).

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering};
use std::sync::{mpsc, Condvar, Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

use windows::core::{Interface, HSTRING, PCSTR, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MonitorFromPoint, MONITORINFO, MONITOR_DEFAULTTONEAREST, MONITOR_DEFAULTTOPRIMARY, MONITOR_FROM_FLAGS};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::CreateEventW;
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::gpu::fence::{self, FenceError};
use crate::stubs::StubInfo;

static RTVS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static LISTS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static CANDIDATE: Mutex<Option<(usize, usize)>> = Mutex::new(None); // (queue, resource) – resource AddRef'd
static PRESENTS: AtomicU64 = AtomicU64::new(0);
static LAST_FRAME_DUMP: Mutex<Option<FrameDumpSample>> = Mutex::new(None);
static BOUNDARIES: AtomicU64 = AtomicU64::new(0);
static LAST_BOUNDARY_MS: AtomicU64 = AtomicU64::new(0);
static HWND_RAW: AtomicIsize = AtomicIsize::new(0);
/// the title called PresentX: the fallback guessing is off
static PRESENTX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PRESENTX_LOCK: Mutex<()> = Mutex::new(());

/// command queue #30 PresentX(planeCount, planes, parameters): plane 0's first resource is
/// the frame; S_OK always (a console present does not fail for the title).
pub unsafe extern "system" fn present_x(queue: *mut c_void, plane_count: u32, planes: *const u8, _params: *const u8) -> i32 {
    // (`XCL_LOOK_STATS=2`: a frame's time on the thread that submits, added up)
    let _cost = super::look_stats::deep().then(|| crate::hle::api_cost::time(crate::hle::api_cost::Call::Present));
    crate::prof_zone!("PresentX");
    if plane_count == 0 || planes.is_null() || !crate::gpu::shader::readable(planes as usize, 0x18) {
        return 0;
    }
    let token = (planes as *const u64).read_unaligned();
    let _present = PRESENTX_LOCK.lock().unwrap();
    present_x_frame(queue, plane_count, planes);
    // The title can use flip statistics to decide when to reuse its buffer.
    // Queue our copy before reporting the flip.
    super::frame::mark_flipped(token);
    crate::prof_frame!();
    0
}

const PLANE_STRIDE: usize = 0x48;
const MAX_PLANES: usize = 8;

/// Read `(resource, pSourceRect)` pairs from `PresentX`'s `D3D12XBOX_PRESENT_PLANE_PARAMETERS` array.
unsafe fn extract_planes(plane_count: u32, planes: *const u8) -> Vec<(usize, Option<[i32; 4]>)> {
    let mut out = Vec::new();
    for i in 0..(plane_count as usize).min(MAX_PLANES) {
        let plane = planes.add(i * PLANE_STRIDE);
        if !crate::gpu::shader::readable(plane as usize, 0x18) {
            break;
        }
        let count = (plane.add(8) as *const u32).read_unaligned();
        let list = (plane.add(0x10) as *const usize).read_unaligned();
        if count == 0 || !crate::gpu::shader::readable(list, 8) {
            continue;
        }
        let res = (list as *const usize).read_unaligned();
        if res == 0 {
            continue;
        }
        let source_rect = if crate::gpu::shader::readable(plane as usize, 0x20) {
            let rect_ptr = (plane.add(0x18) as *const usize).read_unaligned();
            if rect_ptr != 0 && crate::gpu::shader::readable(rect_ptr, 16) {
                Some([
                    (rect_ptr as *const i32).read_unaligned(),
                    ((rect_ptr + 4) as *const i32).read_unaligned(),
                    ((rect_ptr + 8) as *const i32).read_unaligned(),
                    ((rect_ptr + 12) as *const i32).read_unaligned(),
                ])
            } else {
                None
            }
        } else {
            None
        };
        out.push((res, source_rect));
    }
    out
}

fn source_uv_rect(rect: Option<[i32; 4]>, w: u32, h: u32) -> [f32; 4] {
    if let Some([l, t, r, b]) = rect {
        if w > 0 && h > 0 && l >= 0 && t >= 0 && r > l && b > t && r <= w as i32 && b <= h as i32 {
            return [l as f32 / w as f32, t as f32 / h as f32, r as f32 / w as f32, b as f32 / h as f32];
        }
    }
    [0.0, 0.0, 1.0, 1.0]
}

unsafe fn present_x_frame(queue: *mut c_void, plane_count: u32, planes: *const u8) {
    let frames = BOUNDARIES.fetch_add(1, Ordering::Relaxed) + 1;
    super::targets::frame_end();
    super::gpu_times::present(frames);
    super::bvh_want::frame();
    if frames % 300 == 0 {
        // the title's frame rate, for comparing runs (every 300 frames)
        static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let now = std::time::Instant::now();
        if let Some(t) = LAST.lock().unwrap().replace(now) {
            crate::xlog_info!("D3D12.X: {frames} frames presented, {:.1} fps over the last 300", 300.0 / (now - t).as_secs_f64());
        }
    }
    super::real::stop_if_device_removed();
    if !PRESENTX.swap(true, Ordering::Relaxed) {
        crate::xlog_info!("D3D12.X: PresentX – the title presents its own frames ({plane_count} plane(s))");
    }
    {
        // a change in the number of planes (the console's display composes them – a UI
        // plane over the game's): logged with the planes' raw words
        static LAST: AtomicU64 = AtomicU64::new(1);
        if LAST.swap(plane_count as u64, Ordering::Relaxed) != plane_count as u64 {
            let n = (PLANE_STRIDE / 8) * plane_count as usize;
            let words: Vec<u64> = if crate::gpu::shader::readable(planes as usize, 8 * n) { (0..n).map(|k| (planes as *const u64).add(k).read_unaligned()).collect() } else { Vec::new() };
            crate::xlog_info!("D3D12.X: PresentX now with {plane_count} plane(s): {words:x?}");
        }
    }
    let extracted = extract_planes(plane_count, planes);
    if super::capture::enabled() {
        super::capture::present(queue as usize, &extracted.iter().map(|e| e.0).collect::<Vec<_>>(), frames);
    }
    if extracted.is_empty() || !enabled() {
        return;
    }
    {
        let mut p = PRESENTED.lock().unwrap();
        for &(res, _) in &extracted {
            if !p.contains(&res) {
                p.push(res);
            }
        }
    }
    let res = extracted[0].0;
    if trace_frame() {
        super::list_x::dump_pending_geometry();
    }
    if trace_frame() {
        static K: AtomicU64 = AtomicU64::new(0);
        let k = K.fetch_add(1, Ordering::Relaxed);
        if k == 2 || k == 300 {
            super::console_heap::census();
        }
    }
    if trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 3 || n % 50 == 0 {
            let u = USES.lock().unwrap().as_ref().and_then(|m| m.get(&res).copied()).unwrap_or_default();
            let d = ID3D12Resource::from_raw_borrowed(&(res as *mut c_void)).map(|r| r.GetDesc()).map(|d| (d.Width, d.Height, d.Format.0, d.Flags.0));
            let busiest: Vec<String> = {
                let g = USES.lock().unwrap();
                let mut v: Vec<(usize, [u64; 4])> = g.as_ref().map(|m| m.iter().map(|(&k, &u)| (k, u)).collect()).unwrap_or_default();
                v.sort_by_key(|(_, u)| std::cmp::Reverse(u[0] + u[1] + u[2]));
                v.iter().take(6).map(|(k, u)| format!("{k:#x}: rt {} uav {} copy {} clear {}", u[0], u[1], u[2], u[3])).collect()
            };
            crate::xlog_info!("frame trace: PresentX #{n} of {res:#x} {:?} – rt binds {}, UAV views {}, copy destination {}, clears {}; most used: {}", d, u[0], u[1], u[2], u[3], busiest.join("; "));
        }
    }
    present_now(Duration::ZERO, Some((queue as usize, extracted)));
}

thread_local! {
    static IN_PRESENT: Cell<bool> = const { Cell::new(false) };
}

fn now_ms() -> u64 {
    static T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_millis() as u64
}

pub fn counters() -> (u64, u64) {
    (PRESENTS.load(Ordering::Relaxed), BOUNDARIES.load(Ordering::Relaxed))
}

fn enabled() -> bool {
    !crate::state::config().present.eq_ignore_ascii_case("off")
}

unsafe fn add_ref(p: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(1));
    f(p);
}

unsafe fn release(p: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(2));
    f(p);
}

pub unsafe fn note_rtv(handle: usize, res: *mut c_void) {
    if !enabled() {
        return;
    }
    let mut g = RTVS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if !res.is_null() {
        add_ref(res as usize);
    }
    let old = if res.is_null() { m.remove(&handle) } else { m.insert(handle, res as usize) };
    if let Some(o) = old {
        release(o);
    }
}

/// `XCL_TRACE_FRAME=1`: what console code does with each resource it later presents –
/// render-target binds, UAV views, copy destinations, clears (logged at the first presents)
static USES: Mutex<Option<HashMap<usize, [u64; 4]>>> = Mutex::new(None);
pub const USE_RT: usize = 0;
pub const USE_UAV: usize = 1;
pub const USE_COPY: usize = 2;
pub const USE_CLEAR: usize = 3;

pub fn trace_frame() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_TRACE_FRAME").is_some())
}

/// `res` was used as `what` (USE_*)
pub fn note_use(res: usize, what: usize) {
    if trace_frame() && res != 0 {
        USES.lock().unwrap().get_or_insert_with(HashMap::new).entry(res).or_default()[what] += 1;
    }
}

/// resources the title presented
static PRESENTED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub fn is_presented(res: usize) -> bool {
    PRESENTED.lock().unwrap().contains(&res)
}

/// the resource an RTV handle names (render-target binds)
pub fn rtv_resource(handle: usize) -> Option<usize> {
    RTVS.lock().unwrap().as_ref()?.get(&handle).copied()
}

/// The resource `r` while a view still names it (AddRef'd), for `XCL_DUMP_TARGETS`.
pub(super) unsafe fn view_resource(r: usize) -> Option<ID3D12Resource> {
    let g = RTVS.lock().unwrap();
    if !g.as_ref()?.values().any(|&v| v == r) {
        return None;
    }
    ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).cloned()
}

/// The resource bound as render target 0 on `list` (`XCL_TRACE_FRAME`).
pub fn rt0_resource(list: usize) -> Option<usize> {
    let h = *LISTS.lock().unwrap().as_ref()?.get(&list)?;
    rtv_resource(h)
}

pub fn note_rt0(list: usize, handle: usize) {
    if !enabled() {
        return;
    }
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(list, handle);
}

/// Format the swap chain uses for a back buffer of `f` (None = not presentable).
pub fn swap_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match f {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB | DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB | DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

unsafe fn backbuffer_like(res: usize) -> Option<D3D12_RESOURCE_DESC> {
    let raw = res as *mut c_void;
    let r = ID3D12Resource::from_raw_borrowed(&raw)?;
    let d = r.GetDesc();
    let ok = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.Width >= 640
        && d.Height >= 360
        && d.DepthOrArraySize == 1
        && d.SampleDesc.Count == 1
        && swap_format(d.Format).is_some();
    ok.then_some(d)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct QueueId {
    raw: usize,
    generation: u64,
}

impl QueueId {
    #[allow(dead_code)] // Used by queue-cache invalidation in a later unit.
    pub(super) fn raw(self) -> usize {
        self.raw
    }
}

#[derive(Clone)]
#[allow(dead_code)] // Stage B marker: read by the retirement worker in a later unit.
pub(super) struct QueueRetirementMark {
    pub(super) fence: ID3D12Fence,
    pub(super) value: u64,
}

enum Retiring<M> {
    Preparing,
    Mark(M),
    Failed,
}

enum QueueState<M> {
    Active,
    Retiring(Retiring<M>),
}

struct QueueEntry<T, M> {
    id: QueueId,
    owner: T,
    state: QueueState<M>,
}

struct OwnedQueues<T, M> {
    entries: Vec<QueueEntry<T, M>>,
    next_generation: u64,
}

impl<T: Clone, M> OwnedQueues<T, M> {
    const fn new() -> Self {
        Self { entries: Vec::new(), next_generation: 1 }
    }

    /// Add the first owner for `key`. `make` is not called for a duplicate, so no temporary
    /// owner can be dropped while the registry lock is held.
    fn ensure_with(&mut self, raw: usize, make: impl FnOnce() -> Option<T>) -> Option<QueueId> {
        if self.entries.iter().any(|entry| entry.id.raw == raw) {
            return None;
        }
        let owner = make()?;
        let id = QueueId { raw, generation: self.next_generation };
        self.next_generation = self.next_generation.wrapping_add(1).max(1);
        self.entries.push(QueueEntry { id, owner, state: QueueState::Active });
        Some(id)
    }

    fn snapshot(&self) -> Vec<T> {
        self.entries.iter().map(|entry| entry.owner.clone()).collect()
    }

    fn active_snapshot(&self) -> Vec<T> {
        self.entries
            .iter()
            .filter(|entry| matches!(&entry.state, QueueState::Active))
            .map(|entry| entry.owner.clone())
            .collect()
    }

    fn begin_retiring(&mut self, id: QueueId) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) else { return false };
        if !matches!(&entry.state, QueueState::Active) {
            return false;
        }
        entry.state = QueueState::Retiring(Retiring::Preparing);
        true
    }

    fn begin_retiring_if(&mut self, raw: usize, ready: impl FnOnce(&T) -> bool) -> Option<(QueueId, T)> {
        let entry = self.entries.iter_mut().find(|entry| entry.id.raw == raw && matches!(&entry.state, QueueState::Active))?;
        if !ready(&entry.owner) {
            return None;
        }
        entry.state = QueueState::Retiring(Retiring::Preparing);
        Some((entry.id, entry.owner.clone()))
    }

    fn mark_retiring(&mut self, id: QueueId, mark: M) -> Result<(), M> {
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) else { return Err(mark) };
        if !matches!(&entry.state, QueueState::Retiring(Retiring::Preparing)) {
            return Err(mark);
        }
        entry.state = QueueState::Retiring(Retiring::Mark(mark));
        Ok(())
    }

    fn fail_retiring(&mut self, id: QueueId) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) else { return false };
        if !matches!(&entry.state, QueueState::Retiring(Retiring::Preparing)) {
            return false;
        }
        entry.state = QueueState::Retiring(Retiring::Failed);
        true
    }

    /// Copy marks for completion checks after unlocking.
    fn retirement_marks(&self) -> Vec<(QueueId, M)>
    where
        M: Clone,
    {
        self.entries
            .iter()
            .filter_map(|entry| match &entry.state {
                QueueState::Retiring(Retiring::Mark(mark)) => Some((entry.id, mark.clone())),
                _ => None,
            })
            .collect()
    }

    /// Remove entries whose marks were found complete outside the registry lock, and return
    /// their owners for release after unlocking.
    fn sweep(&mut self, completed: &[QueueId]) -> Vec<(QueueId, T, M)> {
        let (mut kept, mut gone) = (Vec::with_capacity(self.entries.len()), Vec::new());
        for entry in std::mem::take(&mut self.entries) {
            match entry {
                QueueEntry { id, owner, state: QueueState::Retiring(Retiring::Mark(mark)) } if completed.contains(&id) => {
                    gone.push((id, owner, mark));
                }
                entry => kept.push(entry),
            }
        }
        self.entries = kept;
        gone
    }
}

/// Queues the title submitted work to. One process-lifetime owner per raw queue, as before.
static QUEUES: Mutex<OwnedQueues<ID3D12CommandQueue, QueueRetirementMark>> = Mutex::new(OwnedQueues::new());
static QUEUE_STATE_CHANGED: Condvar = Condvar::new();

/// Keep the registry owner from queue creation onward, so non-owning users such as `KickoffX`
/// can pin it safely even before the title's first ordinary submission.
pub(super) unsafe fn track_queue(raw: usize) {
    let mut queues = QUEUES.lock().unwrap();
    let _ = queues.ensure_with(raw, || ID3D12CommandQueue::from_raw_borrowed(&(raw as *mut c_void)).cloned());
}

/// Strong snapshot of the queues the title submitted work to. Every AddRef happens while the
/// registry is locked; callers own the returned references and drop them after the lock is gone.
pub fn known_queues_held() -> Vec<ID3D12CommandQueue> {
    QUEUES.lock().unwrap().snapshot()
}

/// Strong snapshot of queues which can still accept internal signals/submissions.
pub(super) fn active_queues_held() -> Vec<ID3D12CommandQueue> {
    QUEUES.lock().unwrap().active_snapshot()
}

pub(super) struct QueueCompletionSnapshot {
    pub(super) active: Vec<ID3D12CommandQueue>,
    pub(super) retiring: Vec<(ID3D12Fence, u64)>,
    pub(super) blocked: bool,
}

/// Completion sources for work submitted before this instant. Active queue owners prevent their
/// retirement transition until callers signal them; installed retirement marks cover queues which
/// already transitioned. Preparing is allowed to install its mark; failed retirement has no
/// trustworthy completion source and makes the snapshot fail-closed.
pub(super) fn queue_completion_snapshot() -> QueueCompletionSnapshot {
    let mut queues = QUEUES.lock().unwrap();
    while queues.entries.iter().any(|entry| matches!(&entry.state, QueueState::Retiring(Retiring::Preparing))) {
        queues = QUEUE_STATE_CHANGED.wait(queues).unwrap();
    }
    let mut out = QueueCompletionSnapshot { active: Vec::new(), retiring: Vec::new(), blocked: false };
    for entry in &queues.entries {
        match &entry.state {
            QueueState::Active => out.active.push(entry.owner.clone()),
            QueueState::Retiring(Retiring::Mark(mark)) => out.retiring.push((mark.fence.clone(), mark.value)),
            QueueState::Retiring(Retiring::Preparing | Retiring::Failed) => out.blocked = true,
        }
    }
    out
}

pub(super) fn queue_accepts_internal_work(raw: usize) -> bool {
    QUEUES
        .lock()
        .unwrap()
        .entries
        .iter()
        .find(|entry| entry.id.raw == raw)
        .is_none_or(|entry| matches!(&entry.state, QueueState::Active))
}

pub(super) unsafe fn pin_queue_for_internal_work(raw: usize) -> Option<ID3D12CommandQueue> {
    let queues = QUEUES.lock().unwrap();
    if let Some(entry) = queues.entries.iter().find(|entry| entry.id.raw == raw) {
        return matches!(&entry.state, QueueState::Active).then(|| entry.owner.clone());
    }
    ID3D12CommandQueue::from_raw_borrowed(&(raw as *mut c_void)).cloned()
}

#[allow(dead_code)] // Stage B API: wired to queue Release in a later unit.
pub(super) unsafe fn begin_queue_retirement(raw: usize) -> Option<(QueueId, ID3D12CommandQueue)> {
    let this = raw as *mut c_void;
    let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(super::real::orig_fn(this, 1));
    let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(super::real::orig_fn(this, 2));
    loop {
        let mut queues = QUEUES.lock().unwrap();
        if !queues.entries.iter().any(|entry| entry.id.raw == raw && matches!(&entry.state, QueueState::Active)) {
            return None;
        }
        let refs_with_probe = add_ref(this);
        let retirement = queues.begin_retiring_if(raw, |_| refs_with_probe == 2);
        drop(queues);
        let refs = release(this);
        if retirement.is_some() || refs != 1 {
            return retirement;
        }
        // The last pre-transition snapshot dropped while the probe was held; retry because its
        // hooked Release could not observe the registry-only count hidden by our probe.
    }
}

#[allow(dead_code)] // Stage B API: wired to queue Release in a later unit.
pub(super) fn mark_queue_retirement(id: QueueId, mark: QueueRetirementMark) -> Result<(), QueueRetirementMark> {
    let result = {
        let mut queues = QUEUES.lock().unwrap();
        queues.mark_retiring(id, mark)
    };
    QUEUE_STATE_CHANGED.notify_all();
    result
}

#[allow(dead_code)] // Stage B API: wired to queue Release in a later unit.
pub(super) fn fail_queue_retirement(id: QueueId) -> bool {
    let failed = QUEUES.lock().unwrap().fail_retiring(id);
    QUEUE_STATE_CHANGED.notify_all();
    failed
}

#[allow(dead_code)] // Stage B API: completion is checked and owners are released in a later unit.
pub(super) fn queue_retirement_marks() -> Vec<(QueueId, QueueRetirementMark)> {
    QUEUES.lock().unwrap().retirement_marks()
}

#[allow(dead_code)] // Stage B API: returned owners must be dropped after this function unlocks.
pub(super) fn sweep_queue_retirements(completed: &[QueueId]) -> Vec<(QueueId, ID3D12CommandQueue)> {
    let retired = {
        let mut queues = QUEUES.lock().unwrap();
        queues.sweep(completed)
    };
    retired.into_iter().map(|(id, owner, mark)| {
        drop(mark);
        (id, owner)
    }).collect()
}

fn quiesce_queue(raw: usize) {
    super::list_x::stop_queue_submissions(raw);
    let candidate = {
        let mut candidate = CANDIDATE.lock().unwrap();
        candidate.is_some_and(|(queue, _)| queue == raw).then(|| candidate.take().unwrap().1)
    };
    let presenter = {
        let mut presenter = PRESENTER.lock().unwrap();
        presenter.as_ref().is_some_and(|p| p.queue == raw).then(|| presenter.take().unwrap())
    };
    if let Some(resource) = candidate {
        unsafe { release(resource) };
    }
    drop(presenter);
}

/// Remove every cache keyed by this queue's raw address after its retirement mark completes and
/// before the final owner is released. No callee holds one subsystem lock while taking another.
#[allow(dead_code)] // Stage C seam: called by the retirement worker in a later unit.
pub(super) fn invalidate_queue_caches(id: QueueId) {
    let raw = id.raw();
    quiesce_queue(raw);
    super::list_x::forget_queue(id);
    super::alloc_pool::forget_queue(id);
    super::cpu_writes::forget_queue(id);
    super::late_release::forget_queue(id);
    super::sparse::forget_queue(id);
    super::title_textures::forget_queue(id);
    super::vram::forget_queue(id);
}

fn retirement_worker() -> Option<&'static mpsc::Sender<()>> {
    static WORKER: OnceLock<Option<mpsc::Sender<()>>> = OnceLock::new();
    WORKER
        .get_or_init(|| {
            let (tx, rx) = mpsc::channel();
            std::thread::Builder::new()
                .name("xcl-queue-retirement".to_string())
                .spawn(move || {
                    while rx.recv().is_ok() {
                        loop {
                            let marks = queue_retirement_marks();
                            if marks.is_empty() {
                                break;
                            }
                            let completed: Vec<_> = marks
                                .iter()
                                .filter_map(|(id, mark)| (unsafe { mark.fence.GetCompletedValue() } >= mark.value).then_some(*id))
                                .collect();
                            drop(marks);
                            for (id, owner) in sweep_queue_retirements(&completed) {
                                invalidate_queue_caches(id);
                                drop(owner);
                            }
                            if completed.is_empty() {
                                std::thread::sleep(Duration::from_millis(10));
                            }
                        }
                    }
                })
                .ok()
                .map(|_| tx)
        })
        .as_ref()
}

/// The hooked Release found that only the registry owner remains. Atomically verify that fact,
/// stop internal submissions, and place the final completion marker. Any marker failure pins the
/// registry owner permanently, so its raw address cannot be reused with stale cache state.
pub(super) unsafe fn retire_released_queue(raw: usize) {
    let Some((id, queue)) = begin_queue_retirement(raw) else { return };
    quiesce_queue(raw);
    let mark = (|| {
        let mut dev: Option<ID3D12Device> = None;
        queue.GetDevice(&mut dev).ok()?;
        let fence = dev?.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok()?;
        super::real::native_queue_signal(&queue, &fence, 1).ok()?;
        Some(QueueRetirementMark { fence, value: 1 })
    })();
    let Some(mark) = mark else {
        let _ = fail_queue_retirement(id);
        return;
    };
    if let Err(mark) = mark_queue_retirement(id, mark) {
        let _ = fail_queue_retirement(id);
        drop(mark);
        return;
    }
    if let Some(worker) = retirement_worker() {
        let _ = worker.send(());
    }
}

static EXECUTES: AtomicU64 = AtomicU64::new(0);

/// XCL is presenting on this thread (its own copy of the frame is being submitted).
pub(super) fn presenting() -> bool {
    IN_PRESENT.with(|c| c.get())
}

/// (PresentX calls, ExecuteCommandLists calls) so far
pub fn activity() -> (u64, u64) {
    (PRESENTS.load(Ordering::Relaxed), EXECUTES.load(Ordering::Relaxed))
}

pub unsafe fn note_execute(queue: *mut c_void, lists: &[*mut c_void]) {
    EXECUTES.fetch_add(1, Ordering::Relaxed);
    track_queue(queue as usize);
    super::late_release::drain_known(false);
    if !enabled() || IN_PRESENT.with(|c| c.get()) || PRESENTX.load(Ordering::Relaxed) {
        return;
    }
    let handle = {
        let g = LISTS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        lists.iter().rev().find_map(|l| m.get(&(*l as usize)).copied())
    };
    let Some(h) = handle else { return };
    let res = RTVS.lock().unwrap().as_ref().and_then(|m| m.get(&h).copied());
    let Some(res) = res else { return };
    if backbuffer_like(res).is_none() {
        return;
    }
    {
        let mut c = CANDIDATE.lock().unwrap();
        if c.map(|(_, r)| r) != Some(res) {
            add_ref(res);
            if let Some((_, old)) = c.replace((queue as usize, res)) {
                release(old);
            }
        }
    }
    // no console present call seen recently → present on submissions (rate limited)
    if now_ms().saturating_sub(LAST_BOUNDARY_MS.load(Ordering::Relaxed)) > 500 {
        present_now(Duration::from_millis(15), None);
    }
}

/// StubHook for console-only queue methods (PresentX, frame events, …).
pub fn on_queue_console_method(_info: &StubInfo, _caller: usize) {
    if !enabled() || PRESENTX.load(Ordering::Relaxed) {
        return;
    }
    BOUNDARIES.fetch_add(1, Ordering::Relaxed);
    LAST_BOUNDARY_MS.store(now_ms(), Ordering::Relaxed);
    let _cost = super::look_stats::deep().then(|| crate::hle::api_cost::time(crate::hle::api_cost::Call::Present));
    unsafe { present_now(Duration::from_millis(4), None) };
}

// ---------------------------------------------------------------------------------------
// window
// ---------------------------------------------------------------------------------------

pub(crate) mod preparation;

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    if let Some(result) = preparation::on_message(hwnd, msg, wp) { return result; }
    // keyboard and mouse as the gamepad: raw mouse input, the cursor held in the window
    if crate::hle::kbm::on_message(hwnd.0, msg, wp.0, lp.0) {
        return LRESULT(0);
    }
    match msg {
        WM_CLOSE => {
            crate::xlog_info!("XCL window closed – ending the game process");
            crate::report::write_final(None);
            crate::log::flush();
            windows::Win32::System::Threading::ExitProcess(0);
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            LRESULT(0)
        }
        WM_TIMER if wp.0 == COMPILE_TIMER => {
            show_compiling(hwnd);
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wp, lp),
    }
}

/// The game's name in the XCL window's title: its display name from MicrosoftGame.config,
/// else its package identity.
fn window_name() -> String {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        let Some(b) = crate::state::boot() else { return "XCL".into() };
        crate::launcher::display_name(&b.game_root).unwrap_or_else(|| b.title.clone())
    })
    .clone()
}

/// the XCL window's timer for `show_compiling`
const COMPILE_TIMER: usize = 0x5C0;

struct FrameRate {
    sampled_at: Instant,
    presented: u64,
    fps: f64,
}

impl FrameRate {
    fn new(now: Instant, presented: u64) -> Self {
        Self {
            sampled_at: now,
            presented,
            fps: 0.0,
        }
    }

    fn sample(&mut self, now: Instant, presented: u64) -> f64 {
        let elapsed = now.duration_since(self.sampled_at);
        if elapsed >= Duration::from_secs(1) {
            self.fps = presented.saturating_sub(self.presented) as f64 / elapsed.as_secs_f64();
            self.sampled_at = now;
            self.presented = presented;
        }
        self.fps
    }
}

/// While XCL compiles shaders (DXC, programs the disk cache did not have – the title draws
/// nothing with those pipelines meanwhile: after an XCL update the screen stays black while
/// thousands recompile) or the driver builds pipelines, the window's title says so
/// (`compiling_text`). It stays 2 s after the last compile (compiles in the title's own calls
/// come one by one), then only the game name and presented frame rate remain.
/// "<title> | compiling shaders: DXC N running, M done | driver N building[ (W waiting for a
/// slot)], D done[, Q pipelines queued][, F failed]": `dxc` (running, done), `driver` builds
/// (running, waiting for a huge-compile slot, done; `real::driver_build_counts`).
fn compiling_text(title: &str, (running, done): (usize, usize), (building, waiting, built): (usize, usize, usize), queued: usize, failed: usize) -> String {
    let waiting = if waiting > 0 { format!(" ({waiting} waiting for a slot)") } else { String::new() };
    let queued = if queued > 0 { format!(", {queued} pipelines queued") } else { String::new() };
    let failed = if failed > 0 { format!(", {failed} failed") } else { String::new() };
    format!("{title} | compiling shaders: DXC {running} running, {done} done | driver {building} building{waiting}, {built} done{queued}{failed}")
}

unsafe fn show_compiling(hwnd: HWND) {
    static SHOWN: Mutex<Option<String>> = Mutex::new(None);
    static LAST_ACTIVE: Mutex<Option<Instant>> = Mutex::new(None);
    static FRAME_RATE: Mutex<Option<FrameRate>> = Mutex::new(None);
    let now = Instant::now();
    let presented = PRESENTS.load(Ordering::Relaxed);
    let fps = FRAME_RATE
        .lock()
        .unwrap()
        .get_or_insert_with(|| FrameRate::new(now, presented))
        .sample(now, presented);
    let (_, failed, queued) = super::pso_async::counts();
    let dxc = crate::gpu::recomp::dxc::compile_counts();
    let driver = super::real::driver_build_counts();
    let active = dxc.0 > 0 || driver.0 > 0 || driver.1 > 0 || queued > 0;
    let recent = {
        let mut last = LAST_ACTIVE.lock().unwrap();
        if active {
            *last = Some(Instant::now());
        }
        last.is_some_and(|t| t.elapsed() < Duration::from_secs(2))
    };
    let title = format!("XCL – {} | {fps:.1} FPS", window_name());
    let text = if recent { compiling_text(&title, dxc, driver, queued, failed) } else { title };
    let mut shown = SHOWN.lock().unwrap();
    if shown.as_deref() != Some(text.as_str()) {
        let _ = SetWindowTextW(hwnd, &HSTRING::from(text.as_str()));
        *shown = Some(text);
    }
}

fn window_thread() {
    unsafe {
        let Ok(inst) = GetModuleHandleW(PCWSTR::null()) else { return };
        let class = windows::core::w!("XCLGameWindow");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: inst.into(),
            lpszClassName: class,
            hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
            ..Default::default()
        };
        RegisterClassW(&wc);
        let cfg = crate::state::config();
        let title = window_name();
        let mut rc = RECT { left: 0, top: 0, right: cfg.window_width as i32, bottom: cfg.window_height as i32 };
        let _ = AdjustWindowRect(&mut rc, WS_OVERLAPPEDWINDOW, false);
        let (w, h) = (rc.right - rc.left, rc.bottom - rc.top);
        let (x, y) = window_position(&cfg.window_pos, w, h);
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            &HSTRING::from(format!("XCL – {title}")),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            x,
            y,
            w,
            h,
            None,
            None,
            Some(inst.into()),
            None,
        );
        let Ok(hwnd) = hwnd else {
            crate::xlog_error!("could not create the XCL window");
            return;
        };
        crate::xlog_info!("XCL window created ({}x{})", cfg.window_width, cfg.window_height);
        if !preparation::active() { crate::hle::kbm::attach(hwnd.0); }
        SetTimer(Some(hwnd), COMPILE_TIMER, 500, None);
        HWND_RAW.store(hwnd.0 as isize, Ordering::Release);
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

/// The XCL window's top-left from `window_pos` (`XCL_WINDOW_POS` overrides): "auto" keeps
/// CW_USEDEFAULT (Windows may pick any monitor), "center" centers on the primary monitor's
/// work area, "x,y" is clamped into the nearest monitor's work area — a multi-monitor desktop
/// otherwise gets the window on whichever monitor Windows' cascade picks.
unsafe fn window_position(setting: &str, w: i32, h: i32) -> (i32, i32) {
    let env = std::env::var("XCL_WINDOW_POS").ok();
    let mode = env.as_deref().unwrap_or(setting).trim();
    if mode.is_empty() || mode.eq_ignore_ascii_case("auto") {
        return (CW_USEDEFAULT, CW_USEDEFAULT);
    }
    let work = |pt: POINT, flags: u32| -> Option<RECT> {
        let mon = MonitorFromPoint(pt, MONITOR_FROM_FLAGS(flags));
        if mon.is_invalid() {
            return None;
        }
        let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
        GetMonitorInfoW(mon, &mut mi).as_bool().then_some(mi.rcWork)
    };
    if mode.eq_ignore_ascii_case("center") {
        let Some(r) = work(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY.0) else {
            return (CW_USEDEFAULT, CW_USEDEFAULT);
        };
        return (r.left + ((r.right - r.left) - w).max(0) / 2, r.top + ((r.bottom - r.top) - h).max(0) / 2);
    }
    let Some((xs, ys)) = mode.split_once(',') else {
        crate::xlog_warn!("window_pos: expected \"auto\", \"center\" or \"x,y\", got {mode:?} – using auto");
        return (CW_USEDEFAULT, CW_USEDEFAULT);
    };
    let (Ok(x), Ok(y)) = (xs.trim().parse::<i32>(), ys.trim().parse::<i32>()) else {
        crate::xlog_warn!("window_pos: expected \"auto\", \"center\" or \"x,y\", got {mode:?} – using auto");
        return (CW_USEDEFAULT, CW_USEDEFAULT);
    };
    // clamp into the work area of the monitor nearest the requested top-left
    let Some(r) = work(POINT { x, y }, MONITOR_DEFAULTTONEAREST.0) else {
        return (x, y);
    };
    (x.clamp(r.left, (r.right - w).max(r.left)), y.clamp(r.top, (r.bottom - h).max(r.top)))
}

fn ensure_window() -> Option<HWND> {
    static START: Once = Once::new();
    START.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-window".into()).spawn(window_thread);
    });
    let t0 = Instant::now();
    while HWND_RAW.load(Ordering::Acquire) == 0 {
        if t0.elapsed() > Duration::from_secs(3) {
            return None;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Some(HWND(HWND_RAW.load(Ordering::Acquire) as *mut c_void))
}

// ---------------------------------------------------------------------------------------
// presenter
// ---------------------------------------------------------------------------------------

/// Number of in-flight present command allocators (`XCL_PRESENT_SYNC=1` drops to 1).
const PRESENT_SLOTS: usize = 2;
/// Swap chain buffer count (triple buffered so frame N copies into a free buffer while frame N-1
/// finishes on the GPU and an earlier buffer scans out).
const SWAP_BUFFERS: u32 = 3;

/// Pipelined fence ring tracking which submitted present fence owns each command-allocator slot.
/// With `slots == 2`, presenting frame `N` waits only for frame `N - 2` (overlapping CPU/GPU work
/// across frames `N - 1` and `N`), while resize, queue switch, and `XCL_FRAME_DUMP` readbacks
/// drain all slots cleanly.
struct PresentRing {
    slot_fences: Vec<u64>,
    cur: usize,
    recorded_value: u64,
}

impl PresentRing {
    fn new(slots: usize) -> Self {
        Self { slot_fences: vec![0; slots.max(1)], cur: 0, recorded_value: 0 }
    }

    /// Observe `presenter_value` (`Presenter.value`) before starting the next present.
    /// If a new submission occurred since our last call (`presenter_value > recorded_value`),
    /// attaches `presenter_value` to the previous slot, rotates to the next slot, and returns
    /// `(slot_to_use, fence_to_wait, rotated_from_prev)`.
    fn begin_present(&mut self, presenter_value: u64) -> (usize, u64, Option<usize>) {
        if presenter_value > self.recorded_value {
            let prev = self.cur;
            self.slot_fences[prev] = presenter_value;
            self.recorded_value = presenter_value;
            let next = (prev + 1) % self.slot_fences.len();
            self.cur = next;
            let rotated = (next != prev).then_some(prev);
            return (next, self.slot_fences[next], rotated);
        }
        (self.cur, self.slot_fences[self.cur], None)
    }

    fn mark_slot_reusable(&mut self, slot: usize) {
        self.slot_fences[slot] = 0;
    }

    /// Mark all slots idle after draining up to `drained_value` (resize / queue switch / dump).
    fn mark_all_drained(&mut self, drained_value: u64) {
        self.slot_fences.fill(0);
        if drained_value > self.recorded_value {
            self.recorded_value = drained_value;
        }
    }
}

fn present_slot_count() -> usize {
    if std::env::var("XCL_PRESENT_SYNC").is_ok_and(|v| v.trim() == "1") {
        1
    } else {
        PRESENT_SLOTS
    }
}

unsafe fn prepare_pipelined_slot(p: &mut Presenter, is_present_x: bool) -> Result<(), FenceError> {
    let (slot, wait_val, rotated) = p.ring.begin_present(p.value);
    if let Some(prev) = rotated {
        p.slot_sources[prev] = std::mem::take(&mut p.copy_sources);
        // If frame N-1 had an XCL_FRAME_DUMP wait timeout, keep its readback buffers alive on
        // slot `prev` until that slot's fence completes (preventing GPU UAF).
        p.slot_dumps[prev] = std::mem::take(&mut p.dumps);
        if let Some(spare) = p.spare_alloc.as_mut() {
            std::mem::swap(&mut p.alloc, spare);
        }
    }
    if wait_val > 0 {
        let timeout_ms = if is_present_x { 2000 } else { 100 };
        fence::wait(&p.fence, wait_val, &p.dev, Some(p.event), timeout_ms)?;
        p.ring.mark_slot_reusable(slot);
        p.slot_sources[slot].clear();
        p.slot_dumps[slot].clear();
    }
    Ok(())
}

fn mark_pipe_drained(p: &mut Presenter) {
    p.ring.mark_all_drained(p.value);
    for s in &mut p.slot_sources {
        s.clear();
    }
    for d in &mut p.slot_dumps {
        d.clear();
    }
}

fn compile_fx(src: &str, target: &str) -> Option<Vec<u8>> {
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
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
            (Ok(()), Some(b)) => Some(std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec()),
            _ => None,
        }
    }
}

fn compositor_shaders() -> Option<&'static (Vec<u8>, Vec<u8>)> {
    static S: OnceLock<Option<(Vec<u8>, Vec<u8>)>> = OnceLock::new();
    S.get_or_init(|| {
        let vs = compile_fx(
            "cbuffer Rect : register(b0) { float4 uv_rect; };\n\
             struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n\
             VSOut main(uint id : SV_VertexID) {\n\
                 float2 p = float2((id & 1) ? 3.0 : -1.0, (id & 2) ? -3.0 : 1.0);\n\
                 float2 t = float2((id & 1) ? 2.0 : 0.0, (id & 2) ? 2.0 : 0.0);\n\
                 VSOut o;\n\
                 o.pos = float4(p, 0.0, 1.0);\n\
                 o.uv = lerp(uv_rect.xy, uv_rect.zw, t);\n\
                 return o;\n\
             }",
            "vs_5_0",
        )?;
        let ps = compile_fx(
            "Texture2D tex : register(t0);\n\
             SamplerState smp : register(s0);\n\
             float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {\n\
                 return tex.Sample(smp, uv);\n\
             }",
            "ps_5_0",
        )?;
        Some((vs, ps))
    })
    .as_ref()
}

struct Compositor {
    root_sig: ID3D12RootSignature,
    psos: HashMap<i32, (ID3D12PipelineState, ID3D12PipelineState)>,
    rtv_heap: ID3D12DescriptorHeap,
    srv_heap: ID3D12DescriptorHeap,
    srv_inc: u32,
}

impl Compositor {
    unsafe fn new(dev: &ID3D12Device) -> Option<Self> {
        let range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &range },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
        ];
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxLOD: f32::MAX,
            ShaderRegister: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            ..Default::default()
        };
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 1,
            pStaticSamplers: &sampler,
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut ser: Option<ID3DBlob> = None;
        D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).ok()?;
        let ser = ser?;
        let root_sig: ID3D12RootSignature = dev
            .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
            .ok()?;
        let rtv_heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })
            .ok()?;
        let srv_heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                // a set per present in flight: the GPU still reads the last one's
                NumDescriptors: (MAX_PLANES * PRESENT_SLOTS) as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .ok()?;
        let srv_inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        Some(Self { root_sig, psos: HashMap::new(), rtv_heap, srv_heap, srv_inc })
    }

    unsafe fn psos_for(&mut self, dev: &ID3D12Device, fmt: DXGI_FORMAT) -> Option<(ID3D12PipelineState, ID3D12PipelineState)> {
        if let Some(pair) = self.psos.get(&fmt.0) {
            return Some(pair.clone());
        }
        let (vs, ps) = compositor_shaders()?;
        let make_pso = |blend: bool| -> Option<ID3D12PipelineState> {
            let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(self.root_sig.clone())),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pd.RasterizerState.DepthClipEnable = false.into();
            pd.DepthStencilState.DepthEnable = false.into();
            pd.DepthStencilState.StencilEnable = false.into();
            pd.RTVFormats[0] = fmt;
            let rt = &mut pd.BlendState.RenderTarget[0];
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            if blend {
                rt.BlendEnable = true.into();
                rt.SrcBlend = D3D12_BLEND_ONE;
                rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                rt.BlendOp = D3D12_BLEND_OP_ADD;
                rt.SrcBlendAlpha = D3D12_BLEND_ONE;
                rt.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            } else {
                rt.BlendEnable = false.into();
            }
            dev.CreateGraphicsPipelineState(&pd).ok()
        };
        let pair = (make_pso(false)?, make_pso(true)?);
        self.psos.insert(fmt.0, pair.clone());
        Some(pair)
    }

    unsafe fn record(
        &mut self,
        dev: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        dst: &ID3D12Resource,
        dst_w: u32,
        dst_h: u32,
        dst_fmt: DXGI_FORMAT,
        planes: &[(ID3D12Resource, D3D12_RESOURCE_DESC, [f32; 4])],
        slot: usize,
    ) -> bool {
        let first = slot % PRESENT_SLOTS * MAX_PLANES;
        let Some((opaque_pso, blend_pso)) = self.psos_for(dev, dst_fmt) else { return false };
        let rtv_cpu = self.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: dst_fmt,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        dev.CreateRenderTargetView(dst, Some(&rtv_desc), rtv_cpu);
        let srv_cpu_base = self.srv_heap.GetCPUDescriptorHandleForHeapStart();
        for (i, (res, desc, _)) in planes.iter().take(MAX_PLANES).enumerate() {
            let mut srv_cpu = srv_cpu_base;
            srv_cpu.ptr += (first + i) * self.srv_inc as usize;
            let mut srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: swap_format(desc.Format).unwrap_or(desc.Format),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            srv_desc.Anonymous.Texture2D.MipLevels = 1;
            dev.CreateShaderResourceView(res, Some(&srv_desc), srv_cpu);
        }
        list.OMSetRenderTargets(1, Some(&rtv_cpu), false, None);
        list.RSSetViewports(&[D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: dst_w as f32,
            Height: dst_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        }]);
        list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: dst_w as i32, bottom: dst_h as i32 }]);
        list.SetGraphicsRootSignature(&self.root_sig);
        list.SetDescriptorHeaps(&[Some(self.srv_heap.clone())]);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        let srv_gpu_base = self.srv_heap.GetGPUDescriptorHandleForHeapStart();
        for (i, (_, _, uv_rect)) in planes.iter().take(MAX_PLANES).enumerate() {
            list.SetPipelineState(if i == 0 { &opaque_pso } else { &blend_pso });
            list.SetGraphicsRoot32BitConstants(0, 4, uv_rect.as_ptr() as *const c_void, 0);
            let mut gpu = srv_gpu_base;
            gpu.ptr += ((first + i) as u64) * (self.srv_inc as u64);
            list.SetGraphicsRootDescriptorTable(1, gpu);
            list.DrawInstanced(3, 1, 0, 0);
        }
        true
    }
}

struct Presenter {
    queue: usize,
    dev: ID3D12Device,
    sc: Option<IDXGISwapChain3>,
    w: u32,
    h: u32,
    fmt: DXGI_FORMAT,
    alloc: ID3D12CommandAllocator,
    spare_alloc: Option<ID3D12CommandAllocator>,
    ring: PresentRing,
    slot_sources: Vec<Vec<ID3D12Resource>>,
    slot_dumps: Vec<Vec<FrameDump>>,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    copy_sources: Vec<ID3D12Resource>,
    compositor: Option<Compositor>,
    dumps: Vec<FrameDump>,
    disabled: bool,
    value: u64,
    event: HANDLE,
    last: Instant,
}

struct FrameDump {
    _source: ID3D12Resource,
    buffer: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    format: DXGI_FORMAT,
    tag: String,
}

unsafe impl Send for Presenter {}

static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

unsafe fn transition(res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    transition_sub(res, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after)
}

unsafe fn transition_sub(res: &ID3D12Resource, sub: u32, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::transmute_copy(res),
                Subresource: sub,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

unsafe fn new_presenter(queue: usize) -> Option<Presenter> {
    let dev_raw = super::real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&dev_raw)?;
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
    let spare_alloc = if present_slot_count() > 1 {
        dev.CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()
    } else {
        None
    };
    let actual_slots = if spare_alloc.is_some() { 2 } else { 1 };
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).ok()?;
    list.Close().ok()?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
    let event = CreateEventW(None, false, false, None).ok()?;
    Some(Presenter {
        queue, dev: dev.clone(), sc: None, w: 0, h: 0, fmt: DXGI_FORMAT_UNKNOWN,
        alloc, spare_alloc, ring: PresentRing::new(actual_slots),
        slot_sources: vec![Vec::new(); actual_slots], slot_dumps: (0..actual_slots).map(|_| Vec::new()).collect(),
        list, fence,
        copy_sources: Vec::new(), compositor: None, dumps: Vec::new(), disabled: false, value: 0, event,
        last: Instant::now() - Duration::from_secs(1),
    })
}

unsafe fn ensure_swapchain(p: &mut Presenter, hwnd: HWND, w: u32, h: u32, fmt: DXGI_FORMAT) -> bool {
    if let Some(sc) = p.sc.clone() {
        if p.w == w && p.h == h && p.fmt == fmt {
            return true;
        }
        // Drain every in-flight present before ResizeBuffers so no GPU copy references old buffers.
        if p.value > 0 {
            if let Err(e) = fence::wait(&p.fence, p.value, &p.dev, Some(p.event), 2000) {
                p.disabled = matches!(&e, FenceError::Removed(_));
                crate::xlog_error!("present: drain before resize: {e}");
                return false;
            }
            mark_pipe_drained(p);
        }
        if sc.ResizeBuffers(SWAP_BUFFERS, w, h, fmt, DXGI_SWAP_CHAIN_FLAG(0)).is_ok() {
            p.w = w;
            p.h = h;
            p.fmt = fmt;
            crate::xlog_info!("present: swap chain {w}x{h} {:?}", fmt);
            return true;
        }
        return false;
    }
    let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory2>(DXGI_CREATE_FACTORY_FLAGS(0)) else { return false };
    let qraw = p.queue as *mut c_void;
    let Some(queue) = ID3D12CommandQueue::from_raw_borrowed(&qraw) else { return false };
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: w,
        Height: h,
        Format: fmt,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: SWAP_BUFFERS,
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        ..Default::default()
    };
    match factory.CreateSwapChainForHwnd(queue, hwnd, &desc, None, None) {
        Ok(sc1) => {
            let _ = factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
            p.sc = sc1.cast::<IDXGISwapChain3>().ok();
            p.w = w;
            p.h = h;
            p.fmt = fmt;
            crate::xlog_info!("present: swap chain {w}x{h} {:?} – the game image will appear in the XCL window", fmt);
            crate::xlog_info!("present: swap chain window {:#x}", hwnd.0 as usize);
            p.sc.is_some()
        }
        Err(e) => {
            crate::xlog_error!("present: CreateSwapChainForHwnd: {e}");
            false
        }
    }
}

unsafe fn present_now(min_interval: Duration, frame: Option<(usize, Vec<(usize, Option<[i32; 4]>)>)>) {
    if IN_PRESENT.with(|c| c.replace(true)) {
        return;
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            IN_PRESENT.with(|c| c.set(false));
        }
    }
    let _reset = Reset;
    let is_frame = frame.is_some();
    let (queue, raw_planes, mut guard) = if let Some((queue, planes)) = frame {
        (queue, planes, PRESENTER.lock().unwrap())
    } else {
        let Ok(guard) = PRESENTER.try_lock() else { return };
        if PRESENTX.load(Ordering::Relaxed) {
            return;
        }
        let candidate = CANDIDATE.lock().unwrap();
        let Some((queue, res)) = *candidate else { return };
        (queue, vec![(res, None)], guard)
    };
    let Some(_queue_pin) = pin_queue_for_internal_work(queue) else { return };
    let resolved: Vec<(ID3D12Resource, D3D12_RESOURCE_DESC, [f32; 4])> = raw_planes
        .into_iter()
        .filter_map(|(res, rect)| {
            let desc = backbuffer_like(res)?;
            let raw = res as *mut c_void;
            let r = ID3D12Resource::from_raw_borrowed(&raw)?.clone();
            let uv = source_uv_rect(rect, desc.Width as u32, desc.Height);
            Some((r, desc, uv))
        })
        .collect();
    if resolved.is_empty() {
        return;
    }
    // PresentX on another queue than the presenter was made for (the fallback presented once from
    // a queue before the title's first PresentX, issue #103): the swap chain belongs to a queue –
    // a new presenter for the title's present queue, after the old one's copy is done
    if is_frame && guard.as_ref().is_some_and(|p| p.queue != queue) {
        if let Some(old) = guard.as_ref() {
            let _ = fence::wait(&old.fence, old.value, &old.dev, Some(old.event), 1000);
        }
        crate::xlog_info!("D3D12.X: PresentX from another queue ({queue:#x}) – the XCL window's swap chain moves to it");
        *guard = None;
    }
    if guard.is_none() {
        *guard = new_presenter(queue);
    }
    let Some(p) = guard.as_mut() else { return };
    if p.disabled || p.last.elapsed() < min_interval {
        return;
    }
    let (w, h) = if resolved.len() == 1 {
        (resolved[0].1.Width as u32, resolved[0].1.Height)
    } else {
        let mw = resolved.iter().map(|(_, d, _)| d.Width as u32).max().unwrap_or(0);
        let mh = resolved.iter().map(|(_, d, _)| d.Height).max().unwrap_or(0);
        (mw, mh)
    };
    let Some(fmt) = swap_format(resolved[0].1.Format) else { return };
    let Some(hwnd) = ensure_window() else { return };
    if p.queue != queue {
        return; // swap chain is bound to the first queue we presented from
    }
    // rotate to the oldest in-flight allocator slot and wait only for its fence (frame N-2)
    let waited = {
        let _cost = super::look_stats::deep().then(|| crate::hle::api_cost::time(crate::hle::api_cost::Call::PresentCopyWait));
        prepare_pipelined_slot(p, is_frame)
    };
    if let Err(e) = waited {
        p.disabled = matches!(&e, FenceError::Removed(_));
        if !matches!(&e, FenceError::Timeout(_)) {
            crate::xlog_error!("present: previous copy: {e}");
        }
        return;
    }
    p.copy_sources.clear();
    p.dumps.clear();
    if !ensure_swapchain(p, hwnd, w, h, fmt) {
        return;
    }
    let Some(sc) = p.sc.clone() else { return };
    let Ok(bb) = sc.GetBuffer::<ID3D12Resource>(sc.GetCurrentBackBufferIndex()) else { return };
    if p.alloc.Reset().is_err() || p.list.Reset(&p.alloc, None).is_err() {
        return;
    }
    // `XCL_DUMP_TARGETS`: a busy frame's render targets go with its picture (copied in the
    // state the title left them, as `XCL_TRACE_FRAME`'s; its frame dump is also the
    // `last_frame_dump` of the report)
    let targets = super::targets::take_pending();
    let dumping_targets = targets.is_some();
    let due = frame_dump_due() || dumping_targets;
    let mut dumps = Vec::new();
    for (tag, res, d) in targets.map_or(Vec::new(), |f| super::targets::to_dump(PRESENTS.load(Ordering::Relaxed), &f)) {
        if let Some((buffer, footprint)) = readback_copy(&p.list, &res, &d) {
            dumps.push(FrameDump { _source: res, buffer, footprint, format: d.Format, tag });
        }
    }
    // and what each depth UAV twin holds (`depth_uav`; it rests in UNORDERED_ACCESS)
    if dumping_targets {
        for (tag, twin) in super::depth_uav::twins_to_dump() {
            let d = twin.GetDesc();
            p.list.ResourceBarrier(&[transition(&twin, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let copied = readback_copy(&p.list, &twin, &d);
            p.list.ResourceBarrier(&[transition(&twin, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            if let Some((buffer, footprint)) = copied {
                dumps.push(FrameDump { _source: twin, buffer, footprint, format: d.Format, tag });
            }
        }
        // and the small 3D textures (`lut3d`), from the state the title's work left them in
        for (tag, tex, d, state) in super::lut3d::to_dump() {
            let moved = super::lut3d::needs_transition(state);
            if moved {
                p.list.ResourceBarrier(&[transition_sub(&tex, 0, state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            }
            let copied = readback_copy(&p.list, &tex, &d);
            if moved {
                p.list.ResourceBarrier(&[transition_sub(&tex, 0, D3D12_RESOURCE_STATE_COPY_SOURCE, state)]);
            }
            if let Some((buffer, footprint)) = copied {
                dumps.push(FrameDump { _source: tex, buffer, footprint, format: d.Format, tag });
            }
        }
    }
    let mut composited = false;
    if resolved.len() > 1 {
        if p.compositor.is_none() {
            p.compositor = Compositor::new(&p.dev);
        }
        if let Some(comp) = p.compositor.as_mut() {
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                let info: Vec<String> = resolved
                    .iter()
                    .enumerate()
                    .map(|(i, (_, d, uv))| format!("plane {i}: {}x{} {:?} uv {uv:?}", d.Width, d.Height, d.Format))
                    .collect();
                crate::xlog_info!("present: compositing {} planes into {w}x{h} {:?} ({})", resolved.len(), fmt, info.join(", "));
            }
            p.list.ResourceBarrier(&[transition(&bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
            if comp.record(&p.dev, &p.list, &bb, w, h, fmt, &resolved, p.ring.cur) {
                if due {
                    p.list.ResourceBarrier(&[transition(&bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    let bb_desc = bb.GetDesc();
                    if let Some((buffer, footprint)) = readback_copy(&p.list, &bb, &bb_desc) {
                        dumps.push(FrameDump { _source: bb.clone(), buffer, footprint, format: bb_desc.Format, tag: "frame".into() });
                    }
                    p.list.ResourceBarrier(&[transition(&bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT)]);
                } else {
                    p.list.ResourceBarrier(&[transition(&bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)]);
                }
                composited = true;
            } else {
                p.list.ResourceBarrier(&[transition(&bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)]);
            }
        } else {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                crate::xlog_warn!("present: compositor unavailable – falling back to plane 0");
            }
        }
    }
    if !composited {
        let (src, desc, _) = &resolved[0];
        let b1 = [transition(&bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST)];
        p.list.ResourceBarrier(&b1);
        if desc.Width as u32 == p.w && desc.Height == p.h {
            p.list.CopyResource(&bb, src);
        } else if desc.Width > 0 && desc.Height > 0 && p.w > 0 && p.h > 0 {
            let cw = (desc.Width as u32).min(p.w);
            let ch = desc.Height.min(p.h);
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(bb.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(src.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let box_ = D3D12_BOX { left: 0, top: 0, front: 0, right: cw, bottom: ch, back: 1 };
            p.list.CopyTextureRegion(&dst, 0, 0, 0, &src_loc, Some(&box_));
            let _ = ManuallyDrop::into_inner(dst.pResource);
            let _ = ManuallyDrop::into_inner(src_loc.pResource);
        }
        if due {
            if let Some((buffer, footprint)) = readback_copy(&p.list, src, desc) {
                dumps.push(FrameDump { _source: src.clone(), buffer, footprint, format: desc.Format, tag: "frame".into() });
            }
        }
        let b2 = [transition(&bb, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT)];
        p.list.ResourceBarrier(&b2);
    }
    // with `XCL_TRACE_FRAME`: also the render target the title binds most (its scene / UI)
    // (and the targets the watched one's contents went on to: rt_<n>_<k>)
    if due && trace_frame() {
        let busiest = busiest_rt();
        let followed = FOLLOWED.lock().unwrap();
        // (a followed target is an address too: taken only while an RTV still holds it)
        let targets = busiest
            .into_iter()
            .chain(followed.iter().filter_map(|&r| view_resource(r)))
            .filter_map(|res| {
                let d = res.GetDesc();
                if d.SampleDesc.Count > 1 {
                    return None;
                }
                let (buffer, footprint) = readback_copy(&p.list, &res, &d)?;
                Some((res, buffer, footprint, d.Format))
            });
        for (k, (res, buffer, footprint, format)) in targets.enumerate() {
            let tag = if k == 0 { "rt".into() } else { format!("rt{k}") };
            dumps.push(FrameDump { _source: res, buffer, footprint, format, tag });
        }
    }
    if p.list.Close().is_err() {
        return;
    }
    let qraw = queue as *mut c_void;
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&qraw) else { return };
    let Ok(cl) = p.list.cast::<ID3D12CommandList>() else { return };
    super::real::native_queue_execute(q, &cl);
    p.copy_sources = resolved.into_iter().map(|(r, _, _)| r).collect();
    p.dumps = dumps;
    p.value += 1;
    if let Err(e) = super::real::native_queue_signal(q, &p.fence, p.value) {
        // The copy is already submitted. Without its signal, keep its resources and allocator.
        p.disabled = true;
        let e = fence::api_error(&p.dev, "Signal", e);
        crate::xlog_error!("present: submit copy: {e}");
        return;
    }
    if dumping_targets {
        super::targets::submitted();
    }
    let presented = {
        let _stall = crate::stall::span(crate::stall::StallKind::Present);
        let _in_driver = super::watch::in_driver(super::watch::Call::Present);
        sc.Present(0, DXGI_PRESENT(0)).ok()
    };
    if let Err(e) = presented {
        let e = fence::api_error(&p.dev, "Present", e);
        p.disabled = matches!(&e, FenceError::Removed(_));
        crate::xlog_error!("present: {e}");
        return;
    }
    p.last = Instant::now();
    if !p.dumps.is_empty() {
        match fence::wait(&p.fence, p.value, &p.dev, Some(p.event), 2000) {
            Ok(()) => {
                for dump in p.dumps.drain(..) {
                    if dump.tag.starts_with("twin") {
                        super::depth_uav::log_values(&dump.buffer, &dump.footprint, &dump.tag);
                        continue;
                    }
                    if dump.tag.starts_with("lut") {
                        super::lut3d::log_values(&dump.buffer, &dump.footprint, &dump.tag);
                        continue;
                    }
                    if dump.tag.starts_with("target") {
                        super::targets::log_census(&dump.buffer, &dump.footprint, &dump.tag);
                    }
                    write_frame(&dump.buffer, &dump.footprint, dump.format, &dump.tag);
                }
                mark_pipe_drained(p);
                p.copy_sources.clear();
            }
            Err(e) => {
                p.disabled = matches!(&e, FenceError::Removed(_));
                // (a copy queued behind the title's work is no stopped GPU: not the fence's
                // text, which a watchdog reads as one; its buffers stay on the ring slot)
                if let FenceError::Timeout(ms) = e {
                    crate::xlog_warn!("present: frame dump copy: not finished after {ms} ms – this dump is skipped");
                } else {
                    crate::xlog_error!("present: frame dump copy: {e}");
                }
                if p.disabled {
                    return;
                }
            }
        }
    }
    let n = PRESENTS.fetch_add(1, Ordering::Relaxed);
    if super::pso_manifest::present_milestone(n + 1) {
        super::pso_manifest::note(super::pso_manifest::Event::Present { n: n + 1 });
    }
    if n == 0 {
        crate::xlog_info!("present: first game frame shown ({w}x{h})");
    }
}

/// `XCL_FRAME_DUMP=1`: every 120th shown frame is written to the logs folder (`XCL_FRAME_DUMP=n`,
/// n > 1: every n-th; a title loading at a few frames a second shows few)
/// (`frame_<n>.bmp`, a quarter of each side; `XCL_FRAME_DUMP_FULL=1`: full size) – what the title presented, before the window.
fn frame_dump_due() -> bool {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| match std::env::var("XCL_FRAME_DUMP") {
        Ok(v) => v.parse().ok().filter(|&n| n > 1).unwrap_or(120),
        Err(_) => 0,
    });
    every > 0 && PRESENTS.load(Ordering::Relaxed) % every == 20.min(every - 1)
}

/// Frames shown so far.
pub fn presents() -> u64 {
    PRESENTS.load(Ordering::Relaxed)
}

/// Content sampled by an existing `XCL_FRAME_DUMP` readback, not proof of gameplay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FrameDumpSample {
    /// Zero-based presentation index, as in `frame_<n>.bmp`.
    pub frame: u64,
    pub width: u32,
    pub height: u32,
    pub format: i32,
    /// One texel every `sample_step` pixels in each direction (normally 4).
    pub sample_step: u32,
    pub sampled_pixels: u64,
    /// Samples whose decoded 8-bit R+G+B exceeds 30; None for an undecoded format.
    pub non_dark_samples: Option<u64>,
}

pub fn last_frame_dump() -> Option<FrameDumpSample> {
    LAST_FRAME_DUMP.lock().ok().and_then(|s| *s)
}

/// a copy of subresource 0 of `src` into a new readback buffer, recorded on `list`
unsafe fn readback_copy(list: &ID3D12GraphicsCommandList, src: &ID3D12Resource, desc: &D3D12_RESOURCE_DESC) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let dev: ID3D12Device = {
        let mut d: Option<ID3D12Device> = None;
        src.GetDevice(&mut d).ok()?;
        d?
    };
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0u64;
    dev.GetCopyableFootprints(desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
    let mut rb: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        &mut rb,
    )
    .ok()?;
    let rb = rb?;
    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
    let s = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(src), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
    list.CopyTextureRegion(&dst, 0, 0, 0, &s, None);
    Some((rb, fp))
}

/// texel → 8-bit RGB for the formats titles present in (None: not decoded)
fn rgb(format: DXGI_FORMAT, t: u32) -> Option<[u8; 3]> {
    Some(match format {
        DXGI_FORMAT_R10G10B10A2_UNORM => [(t >> 2 & 0xFF) as u8, (t >> 12 & 0xFF) as u8, (t >> 22 & 0xFF) as u8],
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => [t as u8, (t >> 8) as u8, (t >> 16) as u8],
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => [(t >> 16) as u8, (t >> 8) as u8, t as u8],
        // HDR: tone-mapped for viewing
        DXGI_FORMAT_R11G11B10_FLOAT => {
            let tm = |v: f32| ((v / (1.0 + v)).sqrt() * 255.0) as u8;
            [tm(small_float(t & 0x7FF, 6)), tm(small_float(t >> 11 & 0x7FF, 6)), tm(small_float(t >> 22, 5))]
        }
        _ => return None,
    })
}

/// A texel of `format` (its bytes: `targets::texel_bytes` of them) as a colour to look at, for
/// the formats `rgb` does not take: a frame's HDR and data targets (`XCL_DUMP_TARGETS`). A
/// float channel is tone-mapped as the HDR formats are, its sign dropped (a normal, a motion
/// vector); an integer target shows its low bytes. A flat picture over a scene that was drawn
/// could not be followed through these: they were written black, "format not decoded".
fn texel_rgb(format: DXGI_FORMAT, t: &[u8]) -> Option<[u8; 3]> {
    let u16_at = |i: usize| u16::from_le_bytes([t[i], t[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([t[i], t[i + 1], t[i + 2], t[i + 3]]);
    // (NaN and infinities: white, so that they show)
    let tm = |v: f32| if v.is_finite() { ((v.abs() / (1.0 + v.abs())).sqrt() * 255.0) as u8 } else { 255 };
    let half = |i: usize| half_float(u16_at(i));
    let float = |i: usize| f32::from_bits(u32_at(i));
    if t.len() < super::targets::texel_bytes(format) {
        return None;
    }
    Some(match format {
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS => [tm(half(0)), tm(half(2)), tm(half(4))],
        DXGI_FORMAT_R16G16B16A16_UNORM => [t[1], t[3], t[5]],
        DXGI_FORMAT_R32G32B32A32_FLOAT | DXGI_FORMAT_R32G32B32A32_TYPELESS => [tm(float(0)), tm(float(4)), tm(float(8))],
        DXGI_FORMAT_R16G16_FLOAT | DXGI_FORMAT_R16G16_TYPELESS => [tm(half(0)), tm(half(2)), 0],
        DXGI_FORMAT_R16G16_UNORM => [t[1], t[3], 0],
        DXGI_FORMAT_R32G32_FLOAT => [tm(float(0)), tm(float(4)), 0],
        DXGI_FORMAT_R32G32_UINT => [t[0], t[4], t[1]],
        DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_D32_FLOAT | DXGI_FORMAT_R32_TYPELESS => [tm(float(0)); 3],
        DXGI_FORMAT_R32_UINT => [t[0], t[1], t[2]],
        DXGI_FORMAT_R16_FLOAT | DXGI_FORMAT_R16_TYPELESS => [tm(half(0)); 3],
        DXGI_FORMAT_R16_UNORM | DXGI_FORMAT_D16_UNORM => [t[1]; 3],
        DXGI_FORMAT_R16_UINT => [t[0], t[1], 0],
        DXGI_FORMAT_R8G8_UNORM | DXGI_FORMAT_R8G8_TYPELESS => [t[0], t[1], 0],
        DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_A8_UNORM | DXGI_FORMAT_R8_TYPELESS => [t[0]; 3],
        DXGI_FORMAT_R8_UINT => [t[0].wrapping_mul(37), t[0].wrapping_mul(91), t[0].wrapping_mul(151)],
        DXGI_FORMAT_R8G8B8A8_TYPELESS => [t[0], t[1], t[2]],
        DXGI_FORMAT_B8G8R8A8_TYPELESS => [t[2], t[1], t[0]],
        DXGI_FORMAT_R10G10B10A2_TYPELESS => return rgb(DXGI_FORMAT_R10G10B10A2_UNORM, u32_at(0)),
        _ if t.len() >= 4 => return rgb(format, u32_at(0)),
        _ => return None,
    })
}

/// An IEEE half float.
pub(super) fn half_float(h: u16) -> f32 {
    let (sign, e, m) = (if h >> 15 != 0 { -1.0f32 } else { 1.0 }, (h >> 10 & 31) as i32, (h & 0x3FF) as f32 / 1024.0);
    sign * match e {
        0 => m * 2f32.powi(-14),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + m) * 2f32.powi(e - 15),
    }
}

/// the largest render target the draws `XCL_TRACE_SHADERS` picked drew into (AddRef'd): the
/// one the `XCL_TRACE_FRAME` dumps show instead of the busiest
static WATCHED_RT: Mutex<Option<usize>> = Mutex::new(None);
/// targets that what the watched one holds went on to (drawn or copied from it; AddRef'd)
static FOLLOWED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub fn watched_rt() -> Option<usize> {
    *WATCHED_RT.lock().unwrap()
}

/// The watched target or one its contents went on to.
pub fn is_watched(r: usize) -> bool {
    watched_rt() == Some(r) || FOLLOWED.lock().unwrap().contains(&r)
}

/// `r` now holds (some of) what the watched target held: its readers are traced too.
pub unsafe fn follow(r: usize) {
    let mut g = FOLLOWED.lock().unwrap();
    if r == 0 || g.contains(&r) || g.len() >= 6 || watched_rt() == Some(r) {
        return;
    }
    add_ref(r);
    g.push(r);
}

pub unsafe fn watch_rt(rt: usize) {
    let area = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map_or(0, |x| {
        let d = x.GetDesc();
        if d.SampleDesc.Count > 1 { 0 } else { d.Width * d.Height as u64 }
    });
    let mut g = WATCHED_RT.lock().unwrap();
    if *g == Some(rt) || area(rt) <= g.map_or(0, area) {
        return;
    }
    add_ref(rt);
    if let Some(old) = g.replace(rt) {
        release(old);
    }
}

/// the single-sampled resource bound as a render target most often (`XCL_TRACE_FRAME`; a
/// multisampled one cannot be copied to a buffer – the present list would not close).
///
/// `USES` and the watched target are addresses, not references. A target is taken only while an
/// RTV still names it, which holds it (`view_resource`): a retail title's world releases
/// transient targets all the time, and GetDesc on one of those addresses ended the process (an
/// access violation under `present_now`, in every trace run that also dumped frames or targets).
unsafe fn busiest_rt() -> Option<ID3D12Resource> {
    if let Some(r) = *WATCHED_RT.lock().unwrap() {
        return view_resource(r);
    }
    let mut bound: Vec<(usize, u64)> = USES.lock().unwrap().as_ref()?.iter().filter(|(_, u)| u[USE_RT] > 0).map(|(&r, u)| (r, u[USE_RT])).collect();
    bound.sort_by_key(|&(_, binds)| std::cmp::Reverse(binds));
    bound.into_iter().find_map(|(r, _)| view_resource(r).filter(|x| x.GetDesc().SampleDesc.Count <= 1))
}

/// an 11- / 10-bit unsigned float (5-bit exponent) -> f32
pub(super) fn small_float(v: u32, mantissa_bits: u32) -> f32 {
    let (e, m) = (v >> mantissa_bits, v & ((1 << mantissa_bits) - 1));
    let m = m as f32 / (1u32 << mantissa_bits) as f32;
    match e {
        0 => m * 2f32.powi(-14),
        31 => 65504.0,
        _ => (1.0 + m) * 2f32.powi(e as i32 - 15),
    }
}

unsafe fn write_frame(rb: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, format: DXGI_FORMAT, tag: &str) {
    let mut p: *mut c_void = std::ptr::null_mut();
    if rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
        return;
    }
    let (w, h, pitch) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize);
    // every 4th texel of every 4th row; `XCL_FRAME_DUMP_FULL=1`: every texel (1-pixel detail)
    let step = if std::env::var_os("XCL_FRAME_DUMP_FULL").is_some() { 1 } else { 4 };
    let (ow, oh) = ((w / step).max(1), (h / step).max(1));
    let row = (ow * 3).div_ceil(4) * 4;
    let mut bmp = Vec::with_capacity(54 + row * oh);
    let size = (54 + row * oh) as u32;
    bmp.extend_from_slice(b"BM");
    for v in [size, 0, 54, 40, ow as u32, oh as u32] {
        bmp.extend_from_slice(&v.to_le_bytes());
    }
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    for _ in 0..6 {
        bmp.extend_from_slice(&0u32.to_le_bytes());
    }
    let (mut lit, mut decoded) = (0usize, true);
    let mut raw: HashMap<u32, usize> = HashMap::new();
    // (a texel of 1, 2, 8 or 16 bytes is read at its own size: read as 4 bytes at a 4-byte
    // stride, an HDR target showed every other half texel and a one-byte one ran past its row)
    let bytes = super::targets::texel_bytes(format);
    for y in (0..oh).rev() {
        let start = bmp.len();
        for x in 0..ow {
            let texel = std::slice::from_raw_parts((p as *const u8).add(fp.Offset as usize + step * y * pitch + bytes * step * x), bytes);
            let mut first = [0u8; 4];
            first[..bytes.min(4)].copy_from_slice(&texel[..bytes.min(4)]);
            let t = u32::from_le_bytes(first);
            *raw.entry(t).or_default() += 1;
            let c = texel_rgb(format, texel).unwrap_or_else(|| {
                decoded = false;
                [0; 3]
            });
            if c.iter().map(|&v| v as u32).sum::<u32>() > 30 {
                lit += 1;
            }
            bmp.extend_from_slice(&[c[2], c[1], c[0]]);
        }
        bmp.resize(start + row, 0);
    }
    rb.Unmap(0, None);
    let n = PRESENTS.load(Ordering::Relaxed);
    // `XCL_TRACE_FRAME` also dumps intermediate targets; only the presented image belongs here.
    if tag == "frame" {
        if let Ok(mut last) = LAST_FRAME_DUMP.lock() {
            *last = Some(FrameDumpSample {
                frame: n,
                width: w as u32,
                height: h as u32,
                format: format.0,
                sample_step: step as u32,
                sampled_pixels: (ow * oh) as u64,
                non_dark_samples: decoded.then_some(lit as u64),
            });
        }
    }
    let Some(boot) = crate::state::boot() else { return };
    let path = boot.logs_dir().join(format!("{tag}_{n}.bmp"));
    let _ = std::fs::write(&path, &bmp);
    crate::xlog_info!(
        "present: {tag} {n} ({w}x{h}, format {}) written to {} – {lit} of {} samples not black{}",
        format.0,
        path.display(),
        ow * oh,
        if decoded { "" } else { " (format not decoded)" }
    );
    let mut top: Vec<(u32, usize)> = raw.into_iter().collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));
    crate::xlog_info!("present: {tag} {n} raw texels (most common): {}", top.iter().take(6).map(|(v, c)| format!("{v:#010x}×{c}")).collect::<Vec<_>>().join(" "));
}

impl Drop for Presenter {
    fn drop(&mut self) {
        unsafe {
            if !self.disabled && self.value > 0 {
                let _ = fence::wait(&self.fence, self.value, &self.dev, Some(self.event), 1000);
            }
            let _ = CloseHandle(self.event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_queue_registry_deduplicates_without_making_a_temporary_owner() {
        let mut registry: OwnedQueues<std::sync::Arc<i32>, ()> = OwnedQueues::new();
        assert!(registry.ensure_with(0xA, || Some(std::sync::Arc::new(1))).is_some());
        assert!(registry.ensure_with(0xA, || -> Option<std::sync::Arc<i32>> { panic!("duplicate owner was made") }).is_none());
        assert!(registry.ensure_with(0xB, || Some(std::sync::Arc::new(2))).is_some());
        let values: Vec<_> = registry.snapshot().iter().map(|v| **v).collect();
        assert_eq!(values, [1, 2]);
    }

    #[test]
    fn owned_queue_snapshot_has_its_own_lifetime() {
        let owner = std::sync::Arc::new(());
        let weak = std::sync::Arc::downgrade(&owner);
        let mut registry: OwnedQueues<std::sync::Arc<()>, ()> = OwnedQueues::new();
        assert!(registry.ensure_with(0xA, || Some(owner)).is_some());
        let snapshot = registry.snapshot();
        drop(registry);
        assert!(weak.upgrade().is_some(), "the snapshot did not keep its owner alive");
        drop(snapshot);
        assert!(weak.upgrade().is_none(), "the caller did not release its snapshot owner");
    }

    #[test]
    fn retiring_queue_stays_in_snapshots_until_its_mark_completes() {
        let mut registry: OwnedQueues<std::sync::Arc<i32>, u64> = OwnedQueues::new();
        let owner = std::sync::Arc::new(1);
        let weak = std::sync::Arc::downgrade(&owner);
        let id = registry.ensure_with(0xA, || Some(owner)).unwrap();
        assert!(registry.begin_retiring(id));
        assert_eq!(registry.snapshot().iter().map(|v| **v).collect::<Vec<_>>(), [1]);
        assert!(registry.retirement_marks().is_empty());
        assert!(registry.sweep(&[id]).is_empty(), "Preparing must remain pinned");
        assert!(registry.mark_retiring(id, 7).is_ok());
        assert_eq!(registry.retirement_marks(), [(id, 7)]);
        assert!(registry.sweep(&[]).is_empty());
        assert_eq!(registry.snapshot().iter().map(|v| **v).collect::<Vec<_>>(), [1]);
        let gone = registry.sweep(&[id]);
        assert_eq!(gone.iter().map(|(gone_id, owner, mark)| (*gone_id, **owner, *mark)).collect::<Vec<_>>(), [(id, 1, 7)]);
        assert!(registry.snapshot().is_empty());
        assert!(weak.upgrade().is_some(), "sweep dropped an owner instead of returning it");
        drop(gone);
        assert!(weak.upgrade().is_none(), "the returned owner was not released by its caller");
    }

    #[test]
    fn failed_retirement_is_permanently_pinned() {
        let mut registry: OwnedQueues<std::sync::Arc<i32>, u64> = OwnedQueues::new();
        let id = registry.ensure_with(0xA, || Some(std::sync::Arc::new(1))).unwrap();
        assert!(registry.begin_retiring(id));
        assert!(registry.fail_retiring(id));
        assert_eq!(registry.mark_retiring(id, 1), Err(1), "an uninstalled mark must be returned");
        assert!(!registry.begin_retiring(id));
        assert!(registry.retirement_marks().is_empty());
        assert!(registry.sweep(&[id]).is_empty());
        assert_eq!(registry.snapshot().iter().map(|v| **v).collect::<Vec<_>>(), [1]);
    }

    #[test]
    fn reused_raw_queue_gets_a_new_generation() {
        let mut registry: OwnedQueues<std::sync::Arc<i32>, u64> = OwnedQueues::new();
        let old = registry.ensure_with(0xA, || Some(std::sync::Arc::new(1))).unwrap();
        assert!(registry.begin_retiring(old));
        assert!(registry.mark_retiring(old, 1).is_ok());
        drop(registry.sweep(&[old]));
        let new = registry.ensure_with(0xA, || Some(std::sync::Arc::new(2))).unwrap();
        assert_ne!(old, new);
        assert!(!registry.begin_retiring(old), "a stale generation matched the reused address");
        assert!(registry.begin_retiring(new));
    }

    #[test]
    fn held_submission_snapshot_delays_atomic_retirement() {
        let mut registry: OwnedQueues<std::sync::Arc<i32>, u64> = OwnedQueues::new();
        let id = registry.ensure_with(0xA, || Some(std::sync::Arc::new(1))).unwrap();
        let held = registry.active_snapshot();
        assert!(registry.begin_retiring_if(0xA, |owner| std::sync::Arc::strong_count(owner) == 1).is_none());
        assert_eq!(registry.active_snapshot().len(), 1, "failed preflight changed Active state");
        drop(held);
        let (retiring, token) = registry
            .begin_retiring_if(0xA, |owner| std::sync::Arc::strong_count(owner) == 1)
            .expect("registry became sole owner");
        assert_eq!(retiring, id);
        assert_eq!(*token, 1);
    }

    #[test]
    fn preparing_queue_is_visible_but_cannot_repopulate_submission_cache() {
        let mut registry: OwnedQueues<std::sync::Arc<i32>, u64> = OwnedQueues::new();
        registry.ensure_with(0xA, || Some(std::sync::Arc::new(1))).unwrap();
        let (_, token) = registry.begin_retiring_if(0xA, |_| true).unwrap();
        assert_eq!(registry.snapshot().len(), 1, "retirement lost its diagnostic owner");
        assert!(registry.active_snapshot().is_empty(), "a post-transition producer acquired the queue");
        let mut cache = vec![0xA];
        cache.clear();
        cache.extend(registry.active_snapshot().into_iter().map(|_| 0xA));
        assert!(cache.is_empty(), "an inactive producer repopulated invalidated state");
        drop(token);
    }

    #[test]
    fn the_title_counts_dxc_compiles_and_driver_builds() {
        assert_eq!(
            compiling_text("XCL – t | 30.0 FPS", (0, 12), (2, 1, 40), 5, 0),
            "XCL – t | 30.0 FPS | compiling shaders: DXC 0 running, 12 done | driver 2 building (1 waiting for a slot), 40 done, 5 pipelines queued"
        );
        assert_eq!(compiling_text("t", (3, 4), (0, 0, 0), 0, 1), "t | compiling shaders: DXC 3 running, 4 done | driver 0 building, 0 done, 1 failed");
    }

    #[test]
    fn dumped_texels_of_hdr_and_data_formats_have_a_colour() {
        assert_eq!(half_float(0x3C00), 1.0);
        assert_eq!(half_float(0xC000), -2.0);
        assert_eq!(half_float(0x0000), 0.0);
        assert!(half_float(0x0001) > 0.0 && half_float(0x0001) < 1e-6, "a denormal");
        assert!(half_float(0x7C00).is_infinite() && half_float(0x7E00).is_nan());
        // RGBA16F (1, 0, 4, 1): tone-mapped x / (1 + x), square root
        let hdr: Vec<u8> = [0x3C00u16, 0, 0x4400, 0x3C00].iter().flat_map(|h| h.to_le_bytes()).collect();
        assert_eq!(texel_rgb(DXGI_FORMAT_R16G16B16A16_FLOAT, &hdr), Some([180, 0, 228]));
        // a NaN shows (white), a negative value by its size
        let odd: Vec<u8> = [0x7E00u16, 0xBC00, 0, 0].iter().flat_map(|h| h.to_le_bytes()).collect();
        assert_eq!(texel_rgb(DXGI_FORMAT_R16G16B16A16_FLOAT, &odd), Some([255, 180, 0]));
        assert_eq!(texel_rgb(DXGI_FORMAT_R8_UNORM, &[200]), Some([200; 3]));
        assert_eq!(texel_rgb(DXGI_FORMAT_R32_UINT, &[1, 2, 3, 4]), Some([1, 2, 3]));
        assert_eq!(texel_rgb(DXGI_FORMAT_R16G16_UNORM, &[0, 10, 0, 20]), Some([10, 20, 0]));
        assert_eq!(texel_rgb(DXGI_FORMAT_R32_FLOAT, &1.0f32.to_le_bytes()), Some([180; 3]));
        // the formats `rgb` knows go through it
        assert_eq!(texel_rgb(DXGI_FORMAT_R8G8B8A8_UNORM, &[1, 2, 3, 4]), Some([1, 2, 3]));
        assert_eq!(texel_rgb(DXGI_FORMAT_B8G8R8A8_UNORM, &[1, 2, 3, 4]), Some([3, 2, 1]));
        // fewer bytes than the format's texel: no colour (and no read past them)
        assert_eq!(texel_rgb(DXGI_FORMAT_R16G16B16A16_FLOAT, &hdr[..4]), None);
        assert_eq!(texel_rgb(DXGI_FORMAT_BC1_UNORM, &[0; 4]), None, "a format with no texels of its own");
    }

    #[test]
    fn present_ring_overlaps_previous_frame_and_drains_cleanly() {
        let mut ring = PresentRing::new(PRESENT_SLOTS);
        // Frame 1 (p.value == 0) -> slot 0, no wait, no rotation.
        assert_eq!(ring.begin_present(0), (0, 0, None));

        // Frame 2 (p.value == 1 after frame 1 submitted) -> rotates 0 -> 1, no wait (overlaps frame 1).
        assert_eq!(ring.begin_present(1), (1, 0, Some(0)));

        // Frame 3 (p.value == 2 after frame 2 submitted) -> rotates 1 -> 0, waits only for frame 1 (N-2).
        assert_eq!(ring.begin_present(2), (0, 1, Some(1)));
        // Retry on timeout (p.value still 2) stays on slot 0 without rotating again.
        assert_eq!(ring.begin_present(2), (0, 1, None));
        ring.mark_slot_reusable(0);
        assert_eq!(ring.begin_present(2), (0, 0, None));

        // Frame 4 (p.value == 3 after frame 3 submitted) -> rotates 0 -> 1, waits only for frame 2 (N-2).
        assert_eq!(ring.begin_present(3), (1, 2, Some(0)));
        ring.mark_slot_reusable(1);

        // Drain after frame 4 submitted (e.g. XCL_FRAME_DUMP / ResizeBuffers) clears all slots.
        ring.mark_all_drained(4);
        assert_eq!(ring.begin_present(4), (1, 0, None));
        assert_eq!(ring.begin_present(5), (0, 0, Some(1)));

        // 1-slot sync mode waits on every previous frame N-1 without rotating.
        let mut sync = PresentRing::new(1);
        assert_eq!(sync.begin_present(0), (0, 0, None));
        assert_eq!(sync.begin_present(1), (0, 1, None));
        sync.mark_slot_reusable(0);
        assert_eq!(sync.begin_present(2), (0, 2, None));
    }

    #[test]
    fn frame_rate_uses_frame_and_time_deltas() {
        let start = Instant::now();
        let mut rate = FrameRate::new(start, 1_000);
        assert_eq!(
            rate.sample(start + Duration::from_millis(1_500), 1_030),
            20.0
        );
        assert_eq!(
            rate.sample(start + Duration::from_millis(2_750), 1_080),
            40.0
        );
    }

    #[test]
    fn frame_rate_waits_for_a_full_sample() {
        let start = Instant::now();
        let mut rate = FrameRate::new(start, 0);
        assert_eq!(rate.sample(start, 0), 0.0);
        assert_eq!(rate.sample(start + Duration::from_millis(500), 15), 0.0);
        assert_eq!(rate.sample(start + Duration::from_secs(1), 30), 30.0);
        assert_eq!(rate.sample(start + Duration::from_millis(1_500), 35), 30.0);
        assert_eq!(rate.sample(start + Duration::from_secs(2), 40), 10.0);
    }

    #[test]
    fn frame_rate_reports_stalls_and_recovery() {
        let start = Instant::now();
        let mut rate = FrameRate::new(start, 0);
        assert_eq!(rate.sample(start + Duration::from_secs(1), 60), 60.0);
        assert_eq!(rate.sample(start + Duration::from_secs(2), 60), 0.0);
        assert_eq!(rate.sample(start + Duration::from_secs(4), 70), 5.0);
    }

    // XCL_WINDOW_POS is process-global; one test covers all modes in sequence.
    #[test]
    fn window_position_modes() {
        unsafe {
            std::env::remove_var("XCL_WINDOW_POS");
            assert_eq!(window_position("auto", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));
            assert_eq!(window_position("", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));
            assert_eq!(window_position("nonsense", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));
            assert_eq!(window_position("12,abc", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));

            let (cx, cy) = window_position("center", 1280, 720);
            if cx != CW_USEDEFAULT {
                // centered on the primary work area: inside it, and not off its left/top
                let mi = {
                    let mon = MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY);
                    let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
                    assert!(GetMonitorInfoW(mon, &mut mi).as_bool());
                    mi
                };
                assert!(cx >= mi.rcWork.left && cy >= mi.rcWork.top);
                assert!(cx + 1280 <= mi.rcWork.right || mi.rcWork.right - mi.rcWork.left < 1280);
            }

            // a position far outside every monitor clamps into the nearest work area
            let (x, y) = window_position("100000,100000", 1280, 720);
            let mon = MonitorFromPoint(POINT { x, y }, MONITOR_DEFAULTTONEAREST);
            assert!(!mon.is_invalid());

            // the env override wins over the setting
            std::env::set_var("XCL_WINDOW_POS", "center");
            assert_eq!(window_position("auto", 1280, 720), window_position("center", 1280, 720));
            std::env::remove_var("XCL_WINDOW_POS");
        }
    }

    #[test]
    fn parse_present_planes_and_uv_rects() {
        let mut rect0 = [160i32, 90, 800, 450];
        let mut res0: usize = 0x1111_0000;
        let mut res1: usize = 0x2222_0000;
        let mut buf = [0u8; 2 * PLANE_STRIDE];
        let write_u64 = |b: &mut [u8], off: usize, v: u64| b[off..off + 8].copy_from_slice(&v.to_le_bytes());
        write_u64(&mut buf, 0x08, 1);
        write_u64(&mut buf, 0x10, (&mut res0 as *mut usize) as u64);
        write_u64(&mut buf, 0x18, rect0.as_mut_ptr() as u64);
        write_u64(&mut buf, PLANE_STRIDE + 0x08, 1);
        write_u64(&mut buf, PLANE_STRIDE + 0x10, (&mut res1 as *mut usize) as u64);
        write_u64(&mut buf, PLANE_STRIDE + 0x18, 0);

        let got = unsafe { extract_planes(2, buf.as_ptr()) };
        assert_eq!(got, vec![(res0, Some(rect0)), (res1, None)]);
        assert_eq!(source_uv_rect(got[0].1, 1600, 900), [0.1, 0.1, 0.5, 0.5]);
        assert_eq!(source_uv_rect(got[1].1, 3840, 2160), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(source_uv_rect(Some([0, 0, 4000, 900]), 1920, 1080), [0.0, 0.0, 1.0, 1.0]);
    }
}

#[cfg(test)]
mod frame_dump_tests {
    use super::*;
    use super::super::real::D3d12Stats;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

    #[test]
    fn reports_without_frame_dumps_remain_readable() {
        let stats = D3d12Stats { presents: 100, ..Default::default() };
        let mut json = serde_json::to_value(&stats).unwrap();
        assert!(json["last_frame_dump"].is_null(), "presents alone do not establish content");
        json.as_object_mut().unwrap().remove("last_frame_dump");
        let old: D3d12Stats = serde_json::from_value(json).unwrap();
        assert_eq!(old.presents, 100);
        assert_eq!(old.last_frame_dump, None);
    }

    #[test]
    fn mapped_frame_dump_distinguishes_dark_bright_and_undecoded_content() {
        unsafe {
            // WARP needs no physical GPU. Fill a CPU-visible buffer with the same padded
            // footprint used by a readback, so this exercises write_frame's actual decoder.
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter = match factory.EnumWarpAdapter() {
                Ok(a) => a,
                Err(_) => {
                    crate::gpu::recomp::runner::skip_test("no WARP");
                    return;
                }
            };
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
            let dev = dev.unwrap();
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 512 + 256 * 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut buffer,
            ).unwrap();
            let buffer = buffer.unwrap();
            let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 512,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: 8, Height: 8, Depth: 1, RowPitch: 256,
                },
            };
            let fill = |red: u8| {
                let mut p: *mut c_void = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut p)).unwrap();
                let bytes = std::slice::from_raw_parts_mut(p as *mut u8, desc.Width as usize);
                bytes.fill(255); // padding must not be counted as bright pixels
                for y in 0..8 {
                    for x in 0..8 {
                        let start = 512 + y * 256 + x * 4;
                        bytes[start..start + 4].copy_from_slice(&[red, 0, 0, 255]);
                    }
                }
                buffer.Unmap(0, None);
            };
            fill(30);
            write_frame(&buffer, &fp, fp.Footprint.Format, "frame");
            let dark = last_frame_dump().unwrap();
            assert_eq!((dark.width, dark.height, dark.format), (8, 8, 28));
            assert_eq!(dark.sampled_pixels, (8 / dark.sample_step).pow(2) as u64);
            assert_eq!(dark.non_dark_samples, Some(0));

            fill(31);
            write_frame(&buffer, &fp, fp.Footprint.Format, "rt");
            assert_eq!(last_frame_dump(), Some(dark), "intermediate targets must not replace frame content");
            write_frame(&buffer, &fp, fp.Footprint.Format, "frame");
            let bright = last_frame_dump().unwrap();
            assert_eq!(bright.non_dark_samples, Some(bright.sampled_pixels));
            // (a format no dump decodes, at the footprint's 4 bytes a texel: shared-exponent
            // RGB is no render target's)
            assert_eq!(texel_rgb(DXGI_FORMAT_R9G9B9E5_SHAREDEXP, &[31, 0, 0, 255]), None);
            write_frame(&buffer, &fp, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, "frame");
            let json = serde_json::to_value(last_frame_dump().unwrap()).unwrap();
            assert!(json["non_dark_samples"].is_null(), "undecoded is different from dark");
        }
    }

    #[test]
    fn compositor_blends_ui_plane_over_scene_plane() {
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
                .unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut comp = Compositor::new(&dev).unwrap();

            let make_tex = |w: u32, h: u32, flags: D3D12_RESOURCE_FLAGS| -> ID3D12Resource {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: w as u64,
                        Height: h,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: flags,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut r,
                )
                .unwrap();
                r.unwrap()
            };
            let upload_tex = |tex: &ID3D12Resource, w: u32, h: u32, fill: &dyn Fn(u32, u32) -> [u8; 4]| -> ID3D12Resource {
                let pitch = (w * 4).div_ceil(256) * 256;
                let mut up: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: (pitch * h) as u64,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut up,
                )
                .unwrap();
                let up = up.unwrap();
                let mut p: *mut c_void = std::ptr::null_mut();
                up.Map(0, None, Some(&mut p)).unwrap();
                let bytes = std::slice::from_raw_parts_mut(p as *mut u8, (pitch * h) as usize);
                for y in 0..h {
                    for x in 0..w {
                        let off = (y * pitch + x * 4) as usize;
                        bytes[off..off + 4].copy_from_slice(&fill(x, y));
                    }
                }
                up.Unmap(0, None);
                let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(tex),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&up),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: w,
                                Height: h,
                                Depth: 1,
                                RowPitch: pitch,
                            },
                        },
                    },
                };
                list.ResourceBarrier(&[transition(tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
                list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                list.ResourceBarrier(&[transition(tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
                up
            };

            let scene = make_tex(4, 4, D3D12_RESOURCE_FLAG_NONE);
            let ui = make_tex(8, 8, D3D12_RESOURCE_FLAG_NONE);
            let out = make_tex(8, 8, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            let _up0 = upload_tex(&scene, 4, 4, &|_, _| [0, 0, 255, 255]);
            let _up1 = upload_tex(&ui, 8, 8, &|x, y| {
                if y == 2 && x < 4 {
                    [0, 128, 0, 128]
                } else if x < 4 {
                    [0, 0, 0, 0]
                } else {
                    [255, 0, 0, 255]
                }
            });

            list.ResourceBarrier(&[transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
            let planes = [
                (scene.clone(), scene.GetDesc(), [0.0, 0.0, 1.0, 1.0]),
                (ui.clone(), ui.GetDesc(), [0.0, 0.0, 1.0, 1.0]),
            ];
            assert!(comp.record(&dev, &list, &out, 8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, &planes, 0));
            list.ResourceBarrier(&[transition(&out, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let out_desc = out.GetDesc();
            let (rb, fp) = readback_copy(&list, &out, &out_desc).unwrap();
            list.Close().unwrap();

            let cl: ID3D12CommandList = list.cast().unwrap();
            queue.ExecuteCommandLists(&[Some(cl)]);
            queue.Signal(&fence, 1).unwrap();
            fence::wait(&fence, 1, &dev, None, 5000).unwrap();

            let mut p: *mut c_void = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).unwrap();
            let px = |x: usize, y: usize| -> [u8; 4] {
                let off = fp.Offset as usize + y * fp.Footprint.RowPitch as usize + x * 4;
                let s = std::slice::from_raw_parts((p as *const u8).add(off), 4);
                [s[0], s[1], s[2], s[3]]
            };
            assert_eq!(px(1, 1), [0, 0, 255, 255], "transparent UI exposes scene plane");
            assert_eq!(px(6, 1), [255, 0, 0, 255], "opaque UI overwrites scene plane");
            let blend = px(1, 2);
            assert!(blend[0] <= 2 && (126..=130).contains(&blend[1]) && (125..=129).contains(&blend[2]) && blend[3] == 255, "unexpected blend {blend:?}");
            rb.Unmap(0, None);
        }
    }
}
