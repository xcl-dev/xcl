//! Write32/64BitValueBottomOfPipeX into memory the GPU does not see: done by the CPU once the GPU
//! has finished the work before them.
//!
//! A console title marks its GPU progress with bottom-of-pipe writes into its own memory and
//! waits on them (a retail game: plain `VirtualAlloc` memory, outside XCL's unified memory).
//! The PC GPU cannot write there; dropped, the title waited for ever after its second frame.
//! Here each queue gets a fence of XCL's: after a submission whose lists carry such writes the
//! queue signals it, and a worker thread does the writes, in submission order, when the GPU is
//! there – what bottom of pipe means.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Threading as th;

/// A bottom-of-pipe write into memory outside unified memory, applied by the CPU once the
/// queue's fence reaches the submission that recorded it.
#[derive(Clone)]
pub(super) enum Write {
    /// Immediate dwords to store starting at `va`.
    Immediate(usize, Vec<u32>),
    /// A 64-bit value resolved by the GPU into a persistently mapped READBACK buffer slot
    /// (`src`), copied atomically to `va` once the queue's fence completes. Holding `_buf` keeps
    /// the mapping alive until the worker finishes.
    ReadbackU64 { va: usize, src: usize, _buf: ID3D12Resource },
}

struct Job {
    fence: ID3D12Fence,
    value: u64,
    writes: Vec<Write>,
    /// how long the worker waits for `value` before writing anyway (`WAIT_MS`)
    wait_ms: u32,
    /// told whether the queue got to `value` before the writes were done
    done: Option<mpsc::Sender<bool>>,
}

// SAFETY: the fence is a free-threaded COM object; the writes are plain data
unsafe impl Send for Job {}

/// queue → its fence and the last value signalled
struct QueueFence {
    fence: ID3D12Fence,
    value: u64,
}

unsafe impl Send for QueueFence {}

static FENCES: Mutex<Option<HashMap<usize, QueueFence>>> = Mutex::new(None);

/// Forget a retired queue's cached fence before its address can be reused. Submitted jobs own
/// fence clones, so extracting this entry cannot let their writes run early.
pub(super) fn forget_queue(id: super::present::QueueId) {
    let gone = {
        let mut fences = FENCES.lock().unwrap();
        fences.as_mut().and_then(|fences| fences.remove(&id.raw()))
    };
    drop(gone);
}

/// how long the worker waits for one submission before writing anyway (a removed device, a
/// queue that never gets there)
const WAIT_MS: u32 = 10_000;

fn worker() -> &'static Mutex<mpsc::Sender<Job>> {
    static W: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
    W.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        let spawned = std::thread::Builder::new().name("xcl-cpu-writes".into()).spawn(move || unsafe {
            let event = th::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
            for job in rx {
                let mut reached = super::real::fence_completed(&job.fence) >= job.value;
                if !reached && job.fence.SetEventOnCompletion(job.value, HANDLE(event as *mut c_void)).is_ok() {
                    // the event can also be set by an earlier job's value, registered before
                    // that job gave up: the fence is checked again until the time is up
                    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(job.wait_ms.into());
                    loop {
                        let left = deadline.saturating_duration_since(std::time::Instant::now());
                        if left.is_zero() {
                            break;
                        }
                        th::WaitForSingleObject(event, left.as_millis().max(1) as u32);
                        reached = super::real::fence_completed(&job.fence) >= job.value;
                        if reached {
                            break;
                        }
                    }
                    if !reached {
                        static LATE: AtomicU64 = AtomicU64::new(0);
                        let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                        if k.is_power_of_two() {
                            crate::xlog_warn!("D3D12.X: {k} writes after the GPU so far done although their queue had not got there after {} ms (fence at {}, wanted {}; {} writes)", job.wait_ms, super::real::fence_completed(&job.fence), job.value, job.writes.len());
                        }
                    }
                }
                for w in &job.writes {
                    match w {
                        Write::Immediate(va, words) => write_now(*va, words),
                        Write::ReadbackU64 { va, src, .. } => {
                            let value = (*(*src as *const AtomicU64)).load(Ordering::Acquire);
                            write_u64_now(*va, value);
                        }
                    }
                }
                if let Some(done) = job.done {
                    let _ = done.send(reached);
                }
            }
        });
        if let Err(e) = spawned {
            crate::xlog_error!("D3D12.X: no thread for the writes after the GPU: {e}");
        }
        Mutex::new(tx)
    })
}

/// Where the CPU writes for the GPU at `va`: at the CPU view behind a GPU view of the title's
/// memory in a view window (the CPU cannot touch the GPU view; its tiles show the CPU view, and
/// the write reaches them as any CPU write does), else at `va`.
fn cpu_at(va: usize) -> usize {
    cpu_at_in(va, |a| super::view_want::is_view(a as u64))
}

/// `cpu_at`, with `in_view` saying which addresses lie in a view window.
fn cpu_at_in(va: usize, in_view: impl Fn(usize) -> bool) -> usize {
    if in_view(va) {
        super::view_want::cpu(va as u64) as usize
    } else {
        va
    }
}

/// The dwords at `va`, if the memory is still there and writable.
unsafe fn write_now(va: usize, words: &[u32]) {
    write_now_in(va, words, &|a, n| if super::view_want::is_view(a as u64) { super::view_want::cpu_run(a as u64, n as u64).map(|c| c as usize) } else { Some(a) });
}

/// `write_now`, with `run` giving where the CPU writes `[a, a + n)` (`cpu_at`) when that is one
/// piece, None when the range covers pages of a GPU view kept apart (the queue's page mappings):
/// then the dwords are written in two calls, split at the first 64 KiB page boundary, each part
/// where its own pages are.
unsafe fn write_now_in(va: usize, words: &[u32], run: &dyn Fn(usize, usize) -> Option<usize>) {
    let bytes = 4 * words.len();
    let asked = va;
    let Some(va) = run(va, bytes) else {
        let first = ((va | 0xFFFF) - va + 1).div_ceil(4);
        if first < words.len() {
            write_now_in(va, &words[..first], run);
            write_now_in(va + 4 * first, &words[first..], run);
        }
        return;
    };
    if !crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
            return;
        }
        if m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 || (m.BaseAddress as usize + m.RegionSize) < va + bytes {
            return;
        }
    }
    for (i, &w) in words.iter().enumerate() {
        (*((va + 4 * i) as *const AtomicU32)).store(w, Ordering::Release);
    }
    note_placed(asked, va, bytes);
}

/// A write for the GPU at `va` was made at `at`, where XCL keeps the page the queue's page
/// mappings put at `va` (`kernelx::mirrors`): it is told as XCL's own write there
/// (`title_textures::note_known_write`) – that place may be a home, which is not write-watched,
/// and the tiles and textures that show the page would not see it. One atomic load while nothing
/// is mapped.
fn note_placed(va: usize, at: usize, bytes: usize) {
    use crate::hle::kernelx::mirrors::{any_mapped, shows, Shows};
    if at != va && any_mapped() && matches!(shows(va), Shows::Page { .. }) {
        super::title_textures::note_known_write(at, bytes);
    }
}

/// Whether `[va, va + 8)` is 8-byte aligned, committed, and writable without `PAGE_GUARD`.
unsafe fn u64_writable_now(va: usize) -> bool {
    if va == 0 || !va.is_multiple_of(8) {
        return false;
    }
    let Some(end) = va.checked_add(8) else { return false };
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, 8)) {
        return true;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE};
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
        return false;
    }
    m.Protect & PAGE_GUARD == 0
        && m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
        && (m.BaseAddress as usize).saturating_add(m.RegionSize) >= end
}

/// An 8-byte aligned 64-bit value at `va`, if the memory is still there and writable.
unsafe fn write_u64_now(va: usize, value: u64) {
    let (asked, va) = (va, cpu_at(va));
    if u64_writable_now(va) {
        (*(va as *const AtomicU64)).store(value, Ordering::Release);
        note_placed(asked, va, 8);
    }
}

/// Advance an 8-byte aligned 64-bit value at `va` to at least `value` (`fetch_max`), if the memory
/// is still committed and writable (a GPU view's at its CPU view, `cpu_at`). Best-effort check
/// before the atomic write (same TOCTOU trade-off as `write_u64_now`: a concurrent title decommit
/// between check and store is unguarded).
pub(super) unsafe fn max_u64_now(va: usize, value: u64) {
    let (asked, va) = (va, cpu_at(va));
    if u64_writable_now(va) {
        (*(va as *const AtomicU64)).fetch_max(value, Ordering::AcqRel);
        note_placed(asked, va, 8);
    }
}

/// Whether `[va, va + bytes)` is memory the CPU can write for the GPU later (a GPU view's at its
/// CPU view, `cpu_at`).
pub(super) fn writable(va: usize, bytes: usize) -> bool {
    let va = cpu_at(va);
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        return true;
    }
    crate::hle::kernelx::lazy::ensure(va, bytes);
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let found = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0;
    found
        && m.State == MEM_COMMIT
        && m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
        && m.BaseAddress as usize + m.RegionSize >= va + bytes
}

/// Receipt published only after the worker has applied every deferred store in a submission.
pub(super) struct Completion(mpsc::Receiver<bool>);

impl Completion {
    /// Whether the stores were done after the GPU had finished their submission. False when the
    /// worker wrote anyway after `WAIT_MS` (a removed device, a queue that never got there) and
    /// when the job never reached the worker (no fence, a failed `Signal`, no worker thread).
    pub(super) fn wait(self) -> bool {
        let _stall = crate::stall::span(crate::stall::StallKind::MemorySynchronization);
        self.0.recv().unwrap_or(false)
    }
}

#[cfg(test)]
pub(super) fn completion_pair_for_test() -> (Completion, mpsc::Sender<bool>) {
    let (tx, rx) = mpsc::channel();
    (Completion(rx), tx)
}

/// Schedule `writes` from lists just submitted to `queue` after the GPU has finished them. With
/// `completion`, every nonempty `writes` gets a receipt; one whose job is dropped before the worker
/// has it reads as failed, its sender going with the job.
unsafe fn schedule(queue: *mut c_void, writes: Vec<Write>, completion: bool, wait_ms: u32) -> Option<Completion> {
    if writes.is_empty() {
        return None;
    }
    let (done, receipt) = if completion {
        let (tx, rx) = mpsc::channel();
        (Some(tx), Some(Completion(rx)))
    } else {
        (None, None)
    };
    enqueue(queue, writes, wait_ms, done);
    receipt
}

unsafe fn enqueue(queue: *mut c_void, writes: Vec<Write>, wait_ms: u32, done: Option<mpsc::Sender<bool>>) {
    if queue.is_null() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let job = {
        let mut g = FENCES.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let entry = match m.entry(queue as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let mut dev: Option<ID3D12Device> = None;
                if q.GetDevice(&mut dev).is_err() {
                    return;
                }
                let Some(dev) = dev else { return };
                let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
                v.insert(QueueFence { fence, value: 0 })
            }
        };
        entry.value += 1;
        if super::real::native_queue_signal(q, &entry.fence, entry.value).is_err() {
            return;
        }
        Job { fence: entry.fence.clone(), value: entry.value, writes, wait_ms, done }
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| crate::xlog_info!("D3D12.X: bottom-of-pipe writes outside unified memory are done by the CPU after the GPU ({} in the first submission)", job.writes.len()));
    let _ = worker().lock().unwrap().send(job);
}

pub(super) unsafe fn after(queue: *mut c_void, writes: Vec<Write>) {
    let _ = schedule(queue, writes, false, WAIT_MS);
}

pub(super) unsafe fn after_with_completion(queue: *mut c_void, writes: Vec<Write>) -> Option<Completion> {
    schedule(queue, writes, true, WAIT_MS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::*;

    /// The value lands after the queue has done what came before it.
    #[test]
    fn writes_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cell = Box::new([0u32; 2]);
            let va = cell.as_ptr() as usize;
            assert!(writable(va, 8));
            let completion = after_with_completion(q.as_raw(), vec![Write::Immediate(va, vec![7, 9])]).expect("scheduled CPU write");
            assert!(completion.wait());
            assert_eq!(*cell, [7, 9]);
        }
    }

    /// A write for the GPU into a GPU view of the title's memory is made at the CPU view behind
    /// it (the CPU cannot touch the GPU view: such a write was not done); any other address, and
    /// every one without view windows, is written where it is.
    #[test]
    fn a_write_into_a_gpu_view_is_made_at_its_cpu_view() {
        use crate::hle::kernelx::mirrors;
        const SIZE: usize = 71 << 16;
        // (addresses far above anything Windows hands out: only the pairing is used)
        let (c, g) = (0x7260_0000_0000usize, 0x7268_0000_0000usize);
        mirrors::reserved(c, SIZE, 0x2040_2000, 0x04);
        assert!(mirrors::reserved(g, SIZE, 0x2040_2000, 0x801));
        let view = |a: usize| (g..g + SIZE).contains(&a);
        assert_eq!(cpu_at_in(g + 0x4_5678, view), c + 0x4_5678);
        assert_eq!(cpu_at_in(c + 0x10, view), c + 0x10);
        assert_eq!(cpu_at_in(g + 0x10, |_| false), g + 0x10, "no view windows: as it was");
        mirrors::released(c);
    }

    /// Dwords for the GPU over two pages of a GPU view whose pages are kept apart land, each,
    /// where its own page is; a write that is one piece is made in one.
    #[test]
    fn a_write_over_pages_kept_apart_is_made_page_by_page() {
        unsafe {
            const PAGE: usize = 0x1_0000;
            // (two pages of a view, kept in two buffers of their own)
            let (mut one, mut two) = (vec![0u32; 8], vec![0u32; 8]);
            let (p1, p2) = (one.as_mut_ptr() as usize, two.as_mut_ptr() as usize);
            let g = 0x7268_0000_0000usize;
            let place = |a: usize| if a < g + PAGE { p1 + 16 + (a - (g + PAGE - 16)) } else { p2 + (a - (g + PAGE)) };
            let calls = std::cell::Cell::new(0);
            let run = |a: usize, n: usize| {
                calls.set(calls.get() + 1);
                ((a ^ (a + n.max(1) - 1)) < PAGE).then(|| place(a))
            };
            write_now_in(g + PAGE - 8, &[1, 2, 3, 4, 5], &run);
            assert_eq!(one[4..], [0, 0, 1, 2]);
            assert_eq!(two[..4], [3, 4, 5, 0]);
            assert_eq!(calls.get(), 3, "split once, at the page boundary");
            write_now_in(g + PAGE + 4, &[6, 7], &run);
            assert_eq!(two[..4], [3, 6, 7, 0]);
            assert_eq!(calls.get(), 4, "inside one page: one piece");
        }
    }

    /// A write for the GPU into a GPU-view page the queue's page mappings named is made where its
    /// page is kept – here its home, which is not write-watched – and told as XCL's own write
    /// there; one into a page never named is made at the CPU view, as before, and not told.
    #[test]
    fn a_write_where_a_mapped_page_is_kept_is_a_known_write() {
        use crate::hle::kernelx::{self, mirrors::{self, PAGE}};
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        unsafe {
            if crate::gpu::uma::get().is_none() {
                return crate::gpu::recomp::runner::skip_test("no unified memory");
            }
            // a pair of views of a size of the test's own, and a physical page at its home: the pool
            let size = 211 * PAGE;
            let cpu = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_READWRITE) as usize;
            let gpu = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(cpu != 0 && gpu != 0);
            mirrors::reserved(cpu, size, 0x2040_2000, 0x04);
            assert!(mirrors::reserved(gpu, size, 0x2040_2000, 0x801));
            let (mut n, mut pages) = (1usize, [0usize; 1]);
            assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let home = kernelx::MapTitlePhysicalPages(std::ptr::null_mut(), 1, 0, 4, pages.as_mut_ptr()) as usize;
            assert_ne!(home, 0);
            // pool page 0 at view page 1
            let ranges: [[u32; 3]; 1] = [[0, 0, 1]];
            let batch = [(gpu + PAGE) as u64, 1, ranges.as_ptr() as u64];
            assert!(mirrors::apply_batches(1, batch.as_ptr() as usize, home, 1).is_some());
            let run = |a: usize, n: usize| super::super::view_want::cpu_run(a as u64, n as u64).map(|c| c as usize);
            let known = |at: usize| super::super::title_textures::known_in(at, PAGE);
            write_now_in(gpu + PAGE + 0x40, &[1, 2], &run);
            assert_eq!(*((home + 0x40) as *const [u32; 2]), [1, 2]);
            assert_eq!(known(home), vec![(home + 0x40, 8)]);
            // a page never named: the CPU view, write-watched as the title's
            assert_eq!(VirtualAlloc(cpu as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, cpu);
            write_now_in(gpu + 0x40, &[3], &run);
            assert_eq!(*((cpu + 0x40) as *const u32), 3);
            assert!(known(cpu).is_empty());
            super::super::title_textures::forget_known_in(home, PAGE);
            kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr());
            mirrors::released(cpu);
            VirtualFree(cpu as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(gpu as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A receipt is true only when the queue got to the writes' submission: false when the worker
    /// wrote after giving up, and a later job's wait outlasts the event of the value given up on.
    #[test]
    fn a_receipt_says_whether_the_queue_got_there() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let cell = Box::new([0u32; 2]);
            let va = cell.as_ptr() as usize;

            // the queue stands at the gate: the worker gives up after 50 ms and writes anyway
            q.Wait(&gate, 1).unwrap();
            let late = schedule(q.as_raw(), vec![Write::Immediate(va, vec![7])], true, 50).expect("a receipt");
            assert!(!late.wait(), "a write done before its queue got there is not confirmed");

            // the next job's value comes after a second gate; opening the first sets the worker's
            // event for the value given up on, which must not end the next job's wait
            q.Wait(&gate, 2).unwrap();
            let next = schedule(q.as_raw(), vec![Write::Immediate(va + 4, vec![9])], true, 5_000).expect("a receipt");
            gate.Signal(1).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert_eq!((*((va + 4) as *const AtomicU32)).load(Ordering::Acquire), 0, "written before its queue got there");
            gate.Signal(2).unwrap();
            assert!(next.wait(), "the queue got to the second job");
            assert_eq!(*cell, [7, 9]);

            // no queue: the job never reaches the worker, and the receipt says so
            let lost = schedule(std::ptr::null_mut(), vec![Write::Immediate(va, vec![1])], true, 50).expect("a receipt");
            assert!(!lost.wait());
            assert!(schedule(q.as_raw(), Vec::new(), true, 50).is_none(), "no writes, no receipt");
        }
    }

    /// A ReadbackU64 write copies the 64-bit value from the mapped readback buffer after the queue finishes.
    #[test]
    fn readback_u64_lands_after_the_queue() {
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
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let expected = 0x1234_5678_9ABC_DEF0u64;
            (*(ptr as *const AtomicU64)).store(expected, Ordering::Release);

            let dst = Box::new(0u64);
            let va = dst.as_ref() as *const u64 as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::ReadbackU64 { va, src: ptr as usize, _buf: rb }]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU64)).load(Ordering::Acquire) != expected {
                assert!(t.elapsed().as_secs() < 5, "readback value never written");
                std::thread::yield_now();
            }
            assert_eq!(*dst, expected);
        }
    }
}
