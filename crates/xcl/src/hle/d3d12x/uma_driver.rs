//! A coherent UMA window in the driver's memory instead of the title's own pages opened with
//! `OpenExistingHeapFromAddress`. On NVIDIA, GPU loads and atomics on the imported pages ran 8-22x
//! slower than on a driver CUSTOM L0 heap, and a whole-frame replay ran window 3's programs 4-270x
//! faster from such a heap. The title keeps its addresses: the window's pages are copied out, the
//! window is released, every other large free stretch of address space is reserved for a moment,
//! so the driver's CPU mapping of the new heap can only land in the hole the window left, and the
//! pages go back. When the mapping lands anywhere else, the window is made again and imported as
//! before. The heap is made with write watch (`D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH`), so the window
//! stays in `Uma::watch_regions` and the title's CPU writes reach its textures as before; a driver
//! that refuses it, or whose mapping `GetWriteWatch` cannot read, keeps the import.
//!
//!
//! `mixed:<start_mb>:<size_mb>[,...]` makes the window several driver heaps instead of one: each
//! range a `GPU_UPLOAD` heap (video memory the CPU writes through the BAR; on NVIDIA it stays in
//! video memory up to 2 GiB) and the rest CUSTOM write-back L0 heaps, each segment's CPU mapping at
//! its place in the window and one reserved buffer over all of them on the GPU
//! (`UpdateTileMappings`). The `GPU_UPLOAD` segments are not write-watched (write-combined pages
//! refuse the watch trap's read-only protection); the title texture code reads them through a copy
//! on the GPU (`read_vram_bytes`).
//!
//! `XCL_UMA_DRIVER`: `custom` or `1` on, `mixed:...` on with the ranges above, `0` off; unset: on
//! (one CUSTOM heap) for an NVIDIA adapter outside Wine.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS,
    PAGE_READWRITE,
};

const MEM_WRITE_WATCH: u32 = 0x20_0000;
const TILE: usize = 0x1_0000;

#[link(name = "kernel32")]
extern "system" {
    fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
}

/// Whether the coherent windows of `device` become driver heaps (see the module's comment).
pub(super) fn wanted(device: &ID3D12Device) -> bool {
    decide(std::env::var("XCL_UMA_DRIVER").ok().as_deref(), crate::util::under_wine, || vendor(device))
}

/// `wanted` from the switch's value, whether the process runs under Wine and the adapter's vendor.
fn decide(switch: Option<&str>, wine: impl FnOnce() -> bool, vendor: impl FnOnce() -> Option<u32>) -> bool {
    match switch.map(str::trim) {
        Some("custom" | "1") => true,
        Some(s) if s.starts_with("mixed:") => true,
        Some("0") => false,
        _ => !wine() && vendor() == Some(0x10DE),
    }
}

/// `XCL_UMA_DRIVER=mixed:<start_mb>:<size_mb>[,...]`: the window's `GPU_UPLOAD` ranges
/// `(start, len)` in bytes, in order; None for any other value.
fn mixed_ranges() -> Option<Vec<(usize, usize)>> {
    parse_mixed(&std::env::var("XCL_UMA_DRIVER").ok()?)
}

fn parse_mixed(raw: &str) -> Option<Vec<(usize, usize)>> {
    let list = raw.trim().strip_prefix("mixed:")?;
    let mut ranges = Vec::new();
    for part in list.split(',') {
        let (a, b) = part.trim().split_once(':')?;
        let (start_mb, size_mb): (usize, usize) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
        if size_mb != 0 {
            ranges.push((start_mb << 20, size_mb << 20));
        }
    }
    ranges.sort_by_key(|r| r.0);
    (!ranges.is_empty()).then_some(ranges)
}

/// PCI vendor id of `device`'s adapter.
fn vendor(device: &ID3D12Device) -> Option<u32> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter = factory.EnumAdapterByLuid(device.GetAdapterLuid()).ok()?;
        Some(adapter.GetDesc().ok()?.VendorId)
    }
}

/// The written pages of `[base, base + size)` that hold anything but zeros, with their bytes.
unsafe fn written_pages(base: usize, size: usize) -> Result<Vec<(usize, Vec<u8>)>, String> {
    let mut out = Vec::new();
    let mut buf: Vec<*mut c_void> = vec![std::ptr::null_mut(); 1 << 16];
    let end = base + size;
    let mut at = base;
    while at < end {
        let (mut n, mut page) = (buf.len(), 0u32);
        let status = GetWriteWatch(0, at as *const c_void, end - at, buf.as_mut_ptr(), &mut n, &mut page);
        if status != 0 {
            return Err(format!("GetWriteWatch at {at:#x} failed ({status})"));
        }
        for &p in &buf[..n] {
            let s = std::slice::from_raw_parts(p as *const u8, page as usize);
            if s.iter().any(|&b| b != 0) {
                out.push((p as usize, s.to_vec()));
            }
        }
        if n < buf.len() {
            break;
        }
        at = buf[n - 1] as usize + page as usize;
    }
    Ok(out)
}

/// Reserves every free stretch of 1 GiB or more (but its first 256 MiB, room for the driver's own
/// small mappings) outside `[hole, hole + len)`.
unsafe fn groom(hole: usize, len: usize) -> Vec<usize> {
    groom_with(hole, len, 1 << 30, 256 << 20)
}

/// `groom` for stretches of `min_region` or more, leaving `headroom` at the start of each.
unsafe fn groom_with(hole: usize, len: usize, min_region: usize, headroom: usize) -> Vec<usize> {
    let (top, mut a, mut got) = (0x7FFF_FFFE_0000usize, 0x1_0000usize, Vec::new());
    while a < top {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(a as *const c_void, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 {
            break;
        }
        let (base, end) = (m.BaseAddress as usize, (m.BaseAddress as usize + m.RegionSize).min(top));
        a = end;
        if m.State != MEM_FREE || m.RegionSize < min_region {
            continue;
        }
        let s = ((base + 0xFFFF) & !0xFFFF).saturating_add(headroom);
        for (s, e) in [(s, end.min(hole)), (s.max(hole + len), end)] {
            let mut s = s;
            while s < e {
                let n = (e - s).min(1 << 40);
                let p = VirtualAlloc(s as *const c_void, n, MEM_RESERVE, PAGE_NOACCESS);
                if !p.is_null() {
                    got.push(p as usize);
                }
                s += n;
            }
        }
    }
    got
}

unsafe fn ungroom(got: Vec<usize>) {
    for p in got {
        VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
    }
}

/// Window `[base, base + size)` (a whole `VirtualAlloc` allocation of its own) remade as a driver
/// heap mapped at `base`, its pages as they were and write-watched: the heap and a buffer over all
/// of it, mapped for good. On failure the window is the title's memory again, its pages as they
/// were.
pub(super) unsafe fn remake(device: &ID3D12Device, base: usize, size: usize) -> Result<(ID3D12Heap, ID3D12Resource), String> {
    remake_with_after_release(device, base, size, || Ok(()))
}

unsafe fn remake_with_after_release(
    device: &ID3D12Device,
    base: usize,
    size: usize,
    after_release: impl FnOnce() -> Result<(), String>,
) -> Result<(ID3D12Heap, ID3D12Resource), String> {
    let t0 = std::time::Instant::now();
    let pages = written_pages(base, size)?;
    let kept: usize = pages.iter().map(|p| p.1.len()).sum();
    if VirtualFree(base as *mut c_void, 0, MEM_RELEASE) == 0 {
        return Err(format!("VirtualFree of the window at {base:#x} failed"));
    }
    let restore = |why: String| -> String {
        let p = VirtualAlloc(base as *const c_void, size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
        if p as usize != base {
            let os_error = windows_sys::Win32::Foundation::GetLastError();
            crate::xlog_error!(
                "D3D12.X: coherent window could not be restored at {base:#x} ({size} bytes, got {p:?}, Windows error {os_error}) after {why}; stopping before guest memory is used"
            );
            crate::log::flush();
            windows_sys::Win32::System::Threading::TerminateProcess(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                0x7C1_0003,
            );
            std::process::abort();
        }
        for (a, bytes) in &pages {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
        }
        why
    };
    let groomed = groom(base, size);
    let reservations = groomed.len();
    let made = (|| -> Result<(ID3D12Heap, ID3D12Resource, usize), String> {
        after_release()?;
        let properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        // write-watched as the title's own window was: its CPU writes to textures there are found so
        let flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH;
        let desc = D3D12_HEAP_DESC { SizeInBytes: size as u64, Properties: properties, Alignment: 0x1_0000, Flags: flags };
        let mut heap: Option<ID3D12Heap> = None;
        device.CreateHeap(&desc, &mut heap).map_err(|e| format!("CreateHeap (write-watched): {e}"))?;
        let heap = heap.ok_or("CreateHeap")?;
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        device.CreatePlacedResource(&heap, 0, &buf_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).map_err(|e| format!("CreatePlacedResource: {e}"))?;
        let buf = buf.ok_or("CreatePlacedResource")?;
        let mut p: *mut c_void = std::ptr::null_mut();
        buf.Map(0, Some(&D3D12_RANGE::default()), Some(&mut p)).map_err(|e| format!("Map: {e}"))?;
        Ok((heap, buf, p as usize))
    })();
    ungroom(groomed);
    // (`GetWriteWatch` reads the mapping: the window can stay in `Uma::watch_regions`)
    let watched = || {
        let (mut one, mut n, mut page) = ([std::ptr::null_mut::<c_void>(); 1], 1usize, 0u32);
        GetWriteWatch(0, base as *const c_void, size, one.as_mut_ptr(), &mut n, &mut page) == 0
    };
    match made {
        Ok((heap, buf, p)) if p == base && watched() => {
            for (a, bytes) in &pages {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
            }
            crate::xlog_info!(
                "D3D12.X: the coherent window at {base:#x} ({} MB) is a driver CUSTOM write-back L0 heap mapped at its address, write-watched ({reservations} reservations, {} pages kept, {} KB, {} ms)",
                size >> 20,
                pages.len(),
                kept >> 10,
                t0.elapsed().as_millis()
            );
            Ok((heap, buf))
        }
        Ok((heap, buf, p)) => {
            buf.Unmap(0, None);
            drop((buf, heap));
            Err(restore(if p == base {
                "GetWriteWatch cannot read the heap's mapping".to_string()
            } else {
                format!("Map gave {p:#x}, not the window's {base:#x} ({reservations} reservations)")
            }))
        }
        Err(e) => Err(restore(e)),
    }
}

/// A coherent window remade as driver heap(s): the buffer shaders bind, the heaps `(offset, len,
/// heap)` that resources placed in the window go to, and the write-watched parts `(va, len)` when
/// they are not the whole window (a mixed window's CUSTOM segments).
pub(super) struct Remade {
    pub buf_raw: usize,
    pub gpu_va: u64,
    pub heaps: Vec<(usize, usize, usize)>,
    pub watched_regions: Option<Vec<(usize, usize)>>,
}

/// `remake`, or `remake_mixed` with `XCL_UMA_DRIVER=mixed:...` when `first` (the ranges are those
/// of the first coherent window; a 16-window layout's further ones are one CUSTOM heap each).
pub(super) unsafe fn remake_window(device: &ID3D12Device, base: usize, size: usize, first: bool) -> Result<Remade, String> {
    if let Some(ranges) = mixed_ranges().filter(|_| first) {
        if gpu_upload_heaps(device) {
            return remake_mixed(device, base, size, &ranges);
        }
        crate::xlog_warn!("D3D12.X: XCL_UMA_DRIVER=mixed: the adapter has no GPU_UPLOAD heaps (Resizable BAR off?), so the coherent window is one CUSTOM heap");
    }
    let (heap, buf) = remake(device, base, size)?;
    let gpu_va = buf.GetGPUVirtualAddress();
    Ok(Remade { buf_raw: buf.into_raw() as usize, gpu_va, heaps: vec![(0, size, heap.into_raw() as usize)], watched_regions: None })
}

/// Whether the adapter makes `GPU_UPLOAD` heaps (`D3D12_OPTIONS16`; needs Resizable BAR). Asked
/// before the window is released: a segment's mapping stays with the driver after its heap is
/// dropped, so a window cannot be made whole again once a segment has been mapped.
unsafe fn gpu_upload_heaps(device: &ID3D12Device) -> bool {
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    device
        .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &mut opts as *mut _ as *mut c_void, std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32)
        .is_ok()
        && opts.GPUUploadHeapSupported.as_bool()
}

#[cfg(test)]
thread_local! {
    /// tests: `mapped_segment` refuses `GPU_UPLOAD` heaps on this thread
    static REFUSE_GPU_UPLOAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn heap_props(upload: bool) -> D3D12_HEAP_PROPERTIES {
    if upload {
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() }
    } else {
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        }
    }
}

fn buffer_desc(size: usize) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

/// `XCL_UMA_UPLOAD_PRIORITY=1` (read once): the `GPU_UPLOAD` segments of a mixed window get
/// `D3D12_RESIDENCY_PRIORITY_MAXIMUM`, so Windows demotes other heaps first when the process is
/// over its video memory budget.
fn upload_priority_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_UMA_UPLOAD_PRIORITY").is_ok_and(|v| v.trim() == "1"))
}

/// One segment of a mixed window: a heap of `size` bytes (write-watched when `watch`), a buffer
/// over all of it and where the driver mapped it for the CPU.
unsafe fn mapped_segment(device: &ID3D12Device, size: usize, upload: bool, watch: bool) -> Result<(ID3D12Heap, ID3D12Resource, usize), String> {
    #[cfg(test)]
    if upload && REFUSE_GPU_UPLOAD.with(|r| r.get()) {
        return Err("CreateHeap: refused by the test".into());
    }
    let flags = if watch { D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH } else { D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
    let desc = D3D12_HEAP_DESC { SizeInBytes: size as u64, Properties: heap_props(upload), Alignment: 0x1_0000, Flags: flags };
    let mut heap: Option<ID3D12Heap> = None;
    device.CreateHeap(&desc, &mut heap).map_err(|e| format!("CreateHeap{}: {e}", if watch { " (write-watched)" } else { "" }))?;
    let heap = heap.ok_or("CreateHeap")?;
    if upload && upload_priority_wanted() {
        // (a process over its video memory budget has heaps demoted to system memory: not this one,
        // which every console program reads its buffers from)
        let set = device.cast::<ID3D12Device1>().and_then(|d| {
            let obj: ID3D12Pageable = heap.cast()?;
            d.SetResidencyPriority(1, &Some(obj), &D3D12_RESIDENCY_PRIORITY_MAXIMUM)
        });
        match set {
            Ok(()) => crate::xlog_info!("D3D12.X: XCL_UMA_UPLOAD_PRIORITY: the {} MB GPU_UPLOAD segment has the highest residency priority", size >> 20),
            Err(e) => crate::xlog_warn!("D3D12.X: XCL_UMA_UPLOAD_PRIORITY: SetResidencyPriority: {e}"),
        }
    }
    let mut buf: Option<ID3D12Resource> = None;
    device.CreatePlacedResource(&heap, 0, &buffer_desc(size), D3D12_RESOURCE_STATE_COMMON, None, &mut buf).map_err(|e| format!("CreatePlacedResource: {e}"))?;
    let buf = buf.ok_or("CreatePlacedResource")?;
    let mut p: *mut c_void = std::ptr::null_mut();
    buf.Map(0, Some(&D3D12_RANGE::default()), Some(&mut p)).map_err(|e| format!("Map: {e}"))?;
    Ok((heap, buf, p as usize))
}

/// `[0, size)` as contiguous `(offset, len, upload)` segments: `vram_ranges` (sorted) on 64 KiB
/// tiles as `GPU_UPLOAD` segments, the rest CUSTOM.
fn build_segments(size: usize, vram_ranges: &[(usize, usize)]) -> Result<Vec<(usize, usize, bool)>, String> {
    let mut segs = Vec::new();
    let mut cur = 0usize;
    for &(start, len) in vram_ranges {
        let s = start.min(size) & !(TILE - 1);
        let e = start.saturating_add(len).min(size) & !(TILE - 1);
        if e <= s {
            continue;
        }
        if s < cur {
            return Err(format!("overlapping mixed ranges at {s:#x} < {cur:#x}"));
        }
        if s > cur {
            segs.push((cur, s - cur, false));
        }
        segs.push((s, e - s, true));
        cur = e;
    }
    if cur < size {
        segs.push((cur, size - cur, false));
    }
    if !segs.iter().any(|s| s.2) {
        return Err("no GPU_UPLOAD range inside the window".into());
    }
    Ok(segs)
}

/// The window could not be made again at its address after it was released: the title's memory
/// there is gone, so the process stops before guest code uses it (as `remake` does).
unsafe fn stop_unbacked(base: usize, size: usize, got: *mut c_void, why: &str) -> ! {
    let os_error = windows_sys::Win32::Foundation::GetLastError();
    crate::xlog_error!(
        "D3D12.X: coherent window could not be restored at {base:#x} ({size} bytes, got {got:?}, Windows error {os_error}) after {why}; stopping before guest memory is used"
    );
    crate::log::flush();
    windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), 0x7C1_0003);
    std::process::abort();
}

/// `remake` with `XCL_UMA_DRIVER=mixed:...`: the window as one driver heap per segment
/// (`build_segments`), each mapped for the CPU at its place in the window, and one reserved buffer
/// over the window whose tiles map to the segments' heaps. The CUSTOM segments are write-watched
/// (a segment that refuses it fails the remake); the `GPU_UPLOAD` ones are not, and are read
/// through `read_vram_bytes`. On failure the window is the title's memory again, its pages as they
/// were.
unsafe fn remake_mixed(device: &ID3D12Device, base: usize, size: usize, vram_ranges: &[(usize, usize)]) -> Result<Remade, String> {
    let t0 = std::time::Instant::now();
    let spec = build_segments(size, vram_ranges)?;
    let min_seg = spec.iter().map(|s| s.1).min().unwrap_or(64 << 20);
    let pages = written_pages(base, size)?;
    let kept: usize = pages.iter().map(|p| p.1.len()).sum();
    if VirtualFree(base as *mut c_void, 0, MEM_RELEASE) == 0 {
        return Err(format!("VirtualFree of the window at {base:#x} failed"));
    }
    let restore = |why: String, placed: Vec<(usize, usize, bool, ID3D12Heap, ID3D12Resource)>, placeholders: Vec<usize>| -> String {
        for (_, _, _, h, b) in placed {
            b.Unmap(0, None);
            drop((b, h));
        }
        for p in placeholders {
            if p != 0 {
                VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
            }
        }
        let p = VirtualAlloc(base as *const c_void, size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
        if p as usize != base {
            stop_unbacked(base, size, p, &why);
        }
        for (a, bytes) in &pages {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
        }
        why
    };

    // every segment's place reserved on its own, so one segment's mapping cannot land in another's
    let mut placeholders: Vec<usize> = Vec::with_capacity(spec.len());
    for &(off, len, _) in &spec {
        let want = base + off;
        let p = VirtualAlloc(want as *const c_void, len, MEM_RESERVE, PAGE_NOACCESS);
        if p as usize != want {
            if !p.is_null() {
                VirtualFree(p, 0, MEM_RELEASE);
            }
            return Err(restore(format!("placeholder reserve at {want:#x} ({} MB) got {p:?}", len >> 20), Vec::new(), placeholders));
        }
        placeholders.push(want);
    }

    // every free stretch of `min_seg` or more outside the window reserved, but 16 MiB at its start
    // for the driver's own small mappings
    let groomed = groom_with(base, size, min_seg, (16 << 20).min(min_seg / 4));
    let reservations = groomed.len();
    let readable = |at: usize, len: usize| {
        let (mut one, mut n, mut page) = ([std::ptr::null_mut::<c_void>(); 1], 1usize, 0u32);
        GetWriteWatch(0, at as *const c_void, len, one.as_mut_ptr(), &mut n, &mut page) == 0
    };
    let mut placed: Vec<(usize, usize, bool, ID3D12Heap, ID3D12Resource)> = Vec::with_capacity(spec.len());
    let mut fail: Option<String> = None;
    for (idx, &(off, len, upload)) in spec.iter().enumerate() {
        let want = base + off;
        let ph = std::mem::replace(&mut placeholders[idx], 0);
        if VirtualFree(ph as *mut c_void, 0, MEM_RELEASE) == 0 {
            fail = Some(format!("VirtualFree of the placeholder at {want:#x} failed"));
            break;
        }
        // (no write watch on GPU_UPLOAD: its write-combined pages refuse the watch trap's
        // read-only protection, which then retried tens of thousands of times)
        let mut made = mapped_segment(device, len, upload, !upload);
        let mut upload = upload;
        if upload && !matches!(&made, Ok((_, _, p)) if *p == want) {
            // a refused GPU_UPLOAD segment becomes a CUSTOM one: the segments mapped so far cannot
            // be given back (their mappings stay with the driver), so the window is kept whole
            let why = match &made {
                Ok((_, _, p)) => format!("Map gave {p:#x}"),
                Err(e) => e.clone(),
            };
            if let Ok((heap, buf, _)) = std::mem::replace(&mut made, Err(String::new())) {
                buf.Unmap(0, None);
                drop((buf, heap));
            }
            crate::xlog_warn!("D3D12.X: XCL_UMA_DRIVER=mixed: segment {idx} ({} MB GPU_UPLOAD): {why}; it is a CUSTOM write-watched heap instead", len >> 20);
            upload = false;
            made = mapped_segment(device, len, false, true);
        }
        let what = if upload { "GPU_UPLOAD" } else { "CUSTOM" };
        match made {
            Ok((heap, buf, p)) if p == want && (upload || readable(want, len)) => placed.push((off, len, upload, heap, buf)),
            Ok((heap, buf, p)) => {
                buf.Unmap(0, None);
                drop((buf, heap));
                fail = Some(if p == want {
                    format!("segment {idx} ({} MB {what}): GetWriteWatch cannot read its mapping", len >> 20)
                } else {
                    format!("segment {idx} ({} MB {what}): Map gave {p:#x}, wanted {want:#x}", len >> 20)
                });
                break;
            }
            Err(e) => {
                fail = Some(format!("segment {idx} ({} MB {what}): {e}", len >> 20));
                break;
            }
        }
    }
    ungroom(groomed);
    if let Some(why) = fail {
        return Err(restore(why, placed, placeholders));
    }

    // one reserved buffer over the window, its 64 KiB tiles mapped onto the segments' heaps
    let reserved = (|| -> Result<ID3D12Resource, String> {
        let mut res: Option<ID3D12Resource> = None;
        device.CreateReservedResource(&buffer_desc(size), D3D12_RESOURCE_STATE_COMMON, None, &mut res).map_err(|e| format!("CreateReservedResource: {e}"))?;
        let res = res.ok_or("CreateReservedResource")?;
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
            .map_err(|e| format!("CreateCommandQueue: {e}"))?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| format!("CreateFence: {e}"))?;
        for (off, len, _, heap, _) in &placed {
            let count = (*len / TILE) as u32;
            let coord = D3D12_TILED_RESOURCE_COORDINATE { X: (*off / TILE) as u32, ..Default::default() };
            let region = D3D12_TILE_REGION_SIZE { NumTiles: count, UseBox: false.into(), ..Default::default() };
            let flag = D3D12_TILE_RANGE_FLAG_NONE;
            queue.UpdateTileMappings(&res, 1, Some(&coord), Some(&region), heap, 1, Some(&flag), Some(&0u32), Some(&count), D3D12_TILE_MAPPING_FLAG_NONE);
        }
        queue.Signal(&fence, 1).map_err(|e| format!("Signal: {e}"))?;
        crate::gpu::fence::wait(&fence, 1, device, None, 2000).map_err(|e| format!("tile mapping wait: {e}"))?;
        Ok(res)
    })();
    let res = match reserved {
        Ok(r) => r,
        Err(why) => return Err(restore(why, placed, placeholders)),
    };

    for (a, bytes) in &pages {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), *a as *mut u8, bytes.len());
    }
    let mut parts = Vec::with_capacity(placed.len());
    let mut heaps = Vec::with_capacity(placed.len());
    let mut watched = Vec::new();
    let heaps_upload: Vec<(usize, usize)> = placed.iter().filter(|p| p.2).map(|p| (p.0, p.1)).collect();
    for (off, len, upload, heap, placed_buf) in placed {
        parts.push(format!("{off:#x}+{}MB {}", len >> 20, if upload { "GPU_UPLOAD" } else { "CUSTOM, write-watched" }));
        if !upload {
            watched.push((base + off, len));
        }
        heaps.push((off, len, heap.into_raw() as usize));
        // the segment's CPU mapping lives as long as its buffer: kept for the process
        std::mem::forget(placed_buf);
    }
    crate::xlog_info!(
        "D3D12.X: the coherent window at {base:#x} ({} MB) is mixed driver heaps mapped at its address [{}] ({reservations} reservations, {} pages kept, {} KB, {} ms)",
        size >> 20,
        parts.join(", "),
        pages.len(),
        kept >> 10,
        t0.elapsed().as_millis()
    );
    let vram: Vec<(usize, usize)> = heaps_upload.iter().map(|&(off, len)| (base + off, base + off + len)).collect();
    if !vram.is_empty() {
        init_vram_reader(device, base, size, &res, vram);
    }
    let gpu_va = res.GetGPUVirtualAddress();
    Ok(Remade { buf_raw: res.into_raw() as usize, gpu_va, heaps, watched_regions: Some(watched) })
}

static VRAM_READER_ACTIVE: AtomicBool = AtomicBool::new(false);
static VRAM_READER: Mutex<Option<VramReader>> = Mutex::new(None);

/// Reads the `GPU_UPLOAD` segments of a mixed window by a copy on the GPU into a readback buffer.
struct VramReader {
    window_base: usize,
    window_end: usize,
    vram_ranges: Vec<(usize, usize)>,
    device: ID3D12Device,
    src_buf: ID3D12Resource,
    queue: ID3D12CommandQueue,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_val: u64,
    readback: ID3D12Resource,
    readback_ptr: *mut u8,
    readback_cap: usize,
}

unsafe impl Send for VramReader {}

unsafe fn make_readback(device: &ID3D12Device, cap: usize) -> Result<(ID3D12Resource, *mut u8), String> {
    let mut rb: Option<ID3D12Resource> = None;
    let desc = D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_NONE, ..buffer_desc(cap) };
    device
        .CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(|e| format!("readback CreateCommittedResource({cap}): {e}"))?;
    let rb = rb.ok_or("readback")?;
    let mut p: *mut c_void = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut p)).map_err(|e| format!("readback Map: {e}"))?;
    Ok((rb, p as *mut u8))
}

unsafe fn init_vram_reader(device: &ID3D12Device, window_base: usize, window_size: usize, src_buf: &ID3D12Resource, vram_ranges: Vec<(usize, usize)>) {
    let made = (|| -> Result<VramReader, String> {
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() })
            .map_err(|e| format!("copy queue: {e}"))?;
        let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).map_err(|e| format!("copy allocator: {e}"))?;
        let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &alloc, None).map_err(|e| format!("copy list: {e}"))?;
        list.Close().map_err(|e| format!("copy list Close: {e}"))?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| format!("copy fence: {e}"))?;
        let cap = 32 << 20;
        let (readback, readback_ptr) = make_readback(device, cap)?;
        Ok(VramReader {
            window_base,
            window_end: window_base + window_size,
            vram_ranges,
            device: device.clone(),
            src_buf: src_buf.clone(),
            queue,
            alloc,
            list,
            fence,
            fence_val: 0,
            readback,
            readback_ptr,
            readback_cap: cap,
        })
    })();
    match made {
        Ok(r) => {
            *VRAM_READER.lock().unwrap() = Some(r);
            VRAM_READER_ACTIVE.store(true, Ordering::Release);
        }
        Err(e) => crate::xlog_warn!("D3D12.X: no GPU reader for the mixed window's GPU_UPLOAD segments ({e}): they are read through the BAR"),
    }
}

/// `[at, at + len)` read by a copy on the GPU (`CopyBufferRegion` to a readback buffer) where it
/// overlaps a `GPU_UPLOAD` segment of a mixed window: CPU reads of that video memory through the
/// BAR ran at 12-84 MiB/s. None where it does not.
pub(super) unsafe fn read_vram_bytes(at: usize, len: usize) -> Option<Vec<u8>> {
    if len == 0 || !VRAM_READER_ACTIVE.load(Ordering::Acquire) {
        return None;
    }
    let mut g = VRAM_READER.lock().unwrap();
    let r = g.as_mut()?;
    let end = at.checked_add(len)?;
    if at < r.window_base || end > r.window_end || !r.vram_ranges.iter().any(|&(s, e)| at < e && end > s) {
        return None;
    }
    if len > r.readback_cap {
        let cap = len.next_power_of_two().max(32 << 20);
        let (rb, ptr) = make_readback(&r.device, cap).ok()?;
        r.readback.Unmap(0, None);
        r.readback = rb;
        r.readback_ptr = ptr;
        r.readback_cap = cap;
    }
    r.alloc.Reset().ok()?;
    r.list.Reset(&r.alloc, None).ok()?;
    r.list.CopyBufferRegion(&r.readback, 0, &r.src_buf, (at - r.window_base) as u64, len as u64);
    r.list.Close().ok()?;
    let cl: ID3D12CommandList = r.list.cast().ok()?;
    r.queue.ExecuteCommandLists(&[Some(cl)]);
    r.fence_val += 1;
    r.queue.Signal(&r.fence, r.fence_val).ok()?;
    crate::gpu::fence::wait(&r.fence, r.fence_val, &r.device, None, 5000).ok()?;
    let mut out = vec![0u8; len];
    std::ptr::copy_nonoverlapping(r.readback_ptr, out.as_mut_ptr(), len);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{build_segments, decide, parse_mixed};

    /// The tests that remake a mixed window run one at a time: a remake reserves the process's other
    /// free stretches so that the driver maps each segment at its place, and another remake releasing
    /// its reservations meanwhile lets a segment land elsewhere, which stops the process.
    static MIXED_REMAKE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Runs the test `name` of this module again on its own in a child process and fails when the
    /// child does; false in the child, which runs the test's body. After the lib's earlier tests the
    /// remake's `CreateCommandQueue` can fail with E_OUTOFMEMORY (with one test thread too), and a
    /// remake that fails after a segment was mapped stops the process: in the test binary that ended
    /// every test after it.
    fn ran_in_child(name: &str) -> bool {
        use std::io::Write;
        const CHILD: &str = "XCL_TEST_UMA_REMAKE_CHILD";
        if std::env::var_os(CHILD).is_some() {
            return false;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &format!("hle::d3d12x::uma_driver::tests::{name}"), "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let transcript = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        for line in transcript.lines().filter(|l| l.starts_with("SKIPPED ")) {
            let _ = writeln!(std::io::stderr(), "{line}");
        }
        assert!(output.status.success(), "{transcript}");
        assert!(transcript.contains("1 passed"), "{transcript}");
        true
    }

    /// On by default for NVIDIA outside Wine only; the switch overrides either way.
    #[test]
    fn driver_windows_by_default_on_nvidia_outside_wine() {
        assert!(decide(None, || false, || Some(0x10DE)));
        assert!(!decide(None, || true, || Some(0x10DE)));
        assert!(!decide(None, || false, || Some(0x1002)));
        assert!(!decide(None, || false, || None));
        assert!(!decide(Some("0"), || false, || Some(0x10DE)));
        assert!(decide(Some("custom"), || true, || Some(0x1002)));
        assert!(decide(Some(" 1 "), || false, || None));
        assert!(decide(Some("mixed:1536:1792"), || true, || Some(0x1002)));
    }

    #[test]
    fn mixed_ranges_parse_in_order() {
        assert_eq!(parse_mixed("mixed:1536:1792"), Some(vec![(1536 << 20, 1792 << 20)]));
        assert_eq!(parse_mixed(" mixed:2048:512,512:256 "), Some(vec![(512 << 20, 256 << 20), (2048 << 20, 512 << 20)]));
        assert_eq!(parse_mixed("mixed:0:0"), None);
        assert_eq!(parse_mixed("mixed:12"), None);
        assert_eq!(parse_mixed("custom"), None);
    }

    #[test]
    fn mixed_segments_cover_the_window() {
        let segs = build_segments(4095 << 20, &[(1536 << 20, 1792 << 20)]).unwrap();
        assert_eq!(segs, vec![(0, 1536 << 20, false), (1536 << 20, 1792 << 20, true), (3328 << 20, 767 << 20, false)]);
        assert!(build_segments(256 << 20, &[(512 << 20, 64 << 20)]).is_err());
        assert!(build_segments(256 << 20, &[(0, 128 << 20), (64 << 20, 64 << 20)]).is_err());
    }

    /// A mixed window keeps the title's bytes in every segment, and the GPU reads the CPU's later
    /// writes to each segment through the one reserved buffer; `read_vram_bytes` reads only the
    /// `GPU_UPLOAD` segment.
    #[test]
    fn mixed_window_keeps_its_bytes_and_the_gpu_sees_every_segment() {
        use super::*;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        let _alone = MIXED_REMAKE.lock().unwrap_or_else(|e| e.into_inner());
        if ran_in_child("mixed_window_keeps_its_bytes_and_the_gpu_sees_every_segment") {
            return;
        }
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev) }.is_err() {
            return crate::gpu::recomp::runner::skip_test("no D3D12 device");
        }
        let dev = dev.unwrap();
        if !unsafe { gpu_upload_heaps(&dev) } {
            return crate::gpu::recomp::runner::skip_test("no GPU_UPLOAD heaps");
        }
        unsafe {
            let size = 256 << 20;
            let base = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            let at = [0x1000, (64 << 20) + 0x1000, (192 << 20) + 0x1000];
            for (i, &o) in at.iter().enumerate() {
                std::ptr::write_volatile((base + o) as *mut u32, 0x1111_0000 + i as u32);
            }
            let remade = match remake_mixed(&dev, base, size, &[(64 << 20, 128 << 20)]) {
                Ok(r) => r,
                Err(e) if e.contains("CreateHeap") => return crate::gpu::recomp::runner::skip_test("no GPU_UPLOAD heaps"),
                Err(e) => panic!("remake_mixed: {e}"),
            };
            assert_eq!(remade.heaps.len(), 3);
            assert_eq!(remade.watched_regions, Some(vec![(base, 64 << 20), (base + (192 << 20), 64 << 20)]));
            for (i, &o) in at.iter().enumerate() {
                assert_eq!(std::ptr::read_volatile((base + o) as *const u32), 0x1111_0000 + i as u32);
                std::ptr::write_volatile((base + o + 0x1000) as *mut u32, 0xCAFE_0000 + i as u32);
            }
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let (rb, ptr) = make_readback(&dev, 256).unwrap();
            let raw = remade.buf_raw as *mut c_void;
            let src = ID3D12Resource::from_raw_borrowed(&raw).unwrap();
            for (i, &o) in at.iter().enumerate() {
                list.CopyBufferRegion(&rb, 4 * i as u64, src, (o + 0x1000) as u64, 4);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            queue.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, &dev, None, 2000).unwrap();
            assert_eq!(std::slice::from_raw_parts(ptr as *const u32, 3), &[0xCAFE_0000, 0xCAFE_0001, 0xCAFE_0002]);
            assert!(read_vram_bytes(base + at[0] + 0x1000, 4).is_none());
            let got = read_vram_bytes(base + at[1] + 0x1000, 4).expect("read of the GPU_UPLOAD segment");
            assert_eq!(u32::from_le_bytes(got.try_into().unwrap()), 0xCAFE_0001);
        }
    }

    /// A `GPU_UPLOAD` segment refused after a CUSTOM one was mapped (an adapter without Resizable
    /// BAR, or one out of BAR space): the window stays whole, that segment a write-watched CUSTOM
    /// heap. (It stopped the process: the mapped segment's pages stay with the driver after its
    /// heap is dropped, so the window could not be reserved again.)
    #[test]
    fn a_refused_gpu_upload_segment_becomes_custom() {
        use super::*;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        let _alone = MIXED_REMAKE.lock().unwrap_or_else(|e| e.into_inner());
        if ran_in_child("a_refused_gpu_upload_segment_becomes_custom") {
            return;
        }
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev) }.is_err() {
            return crate::gpu::recomp::runner::skip_test("no D3D12 device");
        }
        let dev = dev.unwrap();
        unsafe {
            let size = 256 << 20;
            let base = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            let at = [0x1000, (64 << 20) + 0x1000, (192 << 20) + 0x1000];
            for (i, &o) in at.iter().enumerate() {
                std::ptr::write_volatile((base + o) as *mut u32, 0x2222_0000 + i as u32);
            }
            REFUSE_GPU_UPLOAD.with(|r| r.set(true));
            let remade = remake_mixed(&dev, base, size, &[(64 << 20, 128 << 20)]);
            REFUSE_GPU_UPLOAD.with(|r| r.set(false));
            let remade = remade.expect("remake_mixed with the GPU_UPLOAD segment refused");
            assert_eq!(remade.heaps.len(), 3);
            assert_eq!(remade.watched_regions, Some(vec![(base, 64 << 20), (base + (64 << 20), 128 << 20), (base + (192 << 20), 64 << 20)]));
            for (i, &o) in at.iter().enumerate() {
                assert_eq!(std::ptr::read_volatile((base + o) as *const u32), 0x2222_0000 + i as u32);
                std::ptr::write_volatile((base + o + 0x1000) as *mut u32, 0xBEEF_0000 + i as u32);
            }
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let (rb, ptr) = make_readback(&dev, 256).unwrap();
            let raw = remade.buf_raw as *mut c_void;
            let src = ID3D12Resource::from_raw_borrowed(&raw).unwrap();
            for (i, &o) in at.iter().enumerate() {
                list.CopyBufferRegion(&rb, 4 * i as u64, src, (o + 0x1000) as u64, 4);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            queue.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, &dev, None, 2000).unwrap();
            assert_eq!(std::slice::from_raw_parts(ptr as *const u32, 3), &[0xBEEF_0000, 0xBEEF_0001, 0xBEEF_0002]);
        }
    }

    #[test]
    fn failed_write_watch_does_not_release_the_original_window() {
        use super::*;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        const SIZE: usize = 1 << 16;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            std::ptr::write_bytes(base as *mut u8, 0x5a, SIZE);
            let mut dev = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
                return crate::gpu::recomp::runner::skip_test("no D3D12 device");
            }
            // A regular VirtualAlloc block lacks MEM_WRITE_WATCH, forcing GetWriteWatch to fail.
            assert!(remake(&dev.unwrap(), base, SIZE).is_err());
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(VirtualQuery(base as *const c_void, &mut info, std::mem::size_of_val(&info)), 0);
            assert_eq!(info.AllocationBase as usize, base);
            assert_eq!(info.State, MEM_COMMIT);
            assert!(std::slice::from_raw_parts(base as *const u8, SIZE).iter().all(|&b| b == 0x5a));
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn failed_restore_stops_before_importing_an_unbacked_window() {
        use super::*;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        const SIZE: usize = 1 << 16;
        const CODE: i32 = 0x7C1_0003;
        if std::env::var_os("XCL_TEST_RESTORE_COLLISION").is_some() {
            unsafe {
                let mut dev = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
                let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
                assert_ne!(base, 0);
                (base as *mut u8).write(0x5a);
                let result = remake_with_after_release(&dev.unwrap(), base, SIZE, || {
                    let occupied = VirtualAlloc(base as *const c_void, SIZE, MEM_RESERVE, PAGE_NOACCESS);
                    assert_eq!(occupied as usize, base);
                    Err("synthetic post-release failure".into())
                });
                eprintln!("continued after failed restore: {result:?}");
            }
            return;
        }
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            return crate::gpu::recomp::runner::skip_test("no D3D12 device");
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "hle::d3d12x::uma_driver::tests::failed_restore_stops_before_importing_an_unbacked_window", "--nocapture"])
            .env("XCL_TEST_RESTORE_COLLISION", "1")
            .output()
            .unwrap();
        let transcript = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert_eq!(output.status.code(), Some(CODE), "{transcript}");
        assert!(transcript.contains("coherent window could not be restored"), "{transcript}");
        assert!(!transcript.contains("continued after failed restore"), "{transcript}");
    }
}
