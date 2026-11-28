//! kernelx.dll – the console flavour of kernel32/kernelbase.
//!
//! Only functions that differ from Windows are implemented here; everything else falls
//! back to the real KERNELBASE/KERNEL32 export of the same name (see resolver).
//! Signatures and constants: public GDK docs + WinDurango (LGPL-3.0) as reference only (no code taken).

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use windows_sys::Win32::Foundation as fnd;
use windows_sys::Win32::System::LibraryLoader as ll;
use windows_sys::Win32::System::Memory as mem;
use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
use windows_sys::Win32::System::Threading as thr;

use crate::modules;
use crate::pe::ImportName;
use crate::resolver;
use crate::util::{dll_key, from_ansi_ptr, from_wide_ptr, write_ansi_out, write_wide_out};

type HANDLE = *mut c_void;
type BOOL = i32;

/// The title's own unhandled-exception filter goes behind XCL's (`eh`): XCL's crash report
/// first, then the title's handler (its crash reporter) as on the console.
pub unsafe extern "system" fn SetUnhandledExceptionFilter(f: usize) -> usize {
    crate::eh::set_title_filter(f)
}

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![
        // registry: the console mode the GDK thunks read (XSystemGetDeviceType)
        RegOpenKeyW,
        RegOpenKeyExW,
        RegQueryValueExW,
        RegCloseKey,
        // memory
        VirtualAlloc,
        VirtualAllocEx,
        VirtualFree,
        VirtualFreeEx,
        VirtualProtect,
        VirtualProtectEx,
        VirtualQuery,
        VirtualQueryEx,
        XMemAlloc,
        XMemFree,
        XMemAllocDefault,
        XMemFreeDefault,
        XMemSetAllocationHooks,
        XMemVirtualQuery,
        XMemVirtualAlloc,
        XMemAllocatePhysicalPages,
        XMemMapPhysicalPages,
        XMemMapPhysicalPagesScatter,
        XMemFreePhysicalPages,
        XMemTransferMemory,
        XMemGetWorkingSetStatistics,
        XMemIsToolsMemAvailable,
        TitleMemoryStatus,
        JobTitleMemoryStatus,
        ToolingMemoryStatus,
        AllocateTitlePhysicalPages,
        FreeTitlePhysicalPages,
        MapTitlePhysicalPages,
        // app state (minappmodelx): a title is never suspended or constrained here
        RegisterAppStateChangeNotification,
        UnregisterAppStateChangeNotification,
        RegisterAppConstrainedChangeNotification,
        UnregisterAppConstrainedChangeNotification,
        // threads / system
        SetUnhandledExceptionFilter,
        Sleep,
        SetThreadName,
        GetThreadName,
        SetThreadpoolAffinityMask,
        SetThreadAffinityMask,
        SetThreadGroupAffinity,
        QueryProcessorSchedulingStatistics,
        GetConsoleType,
        OutputDebugStringA,
        OutputDebugStringW,
        GetSystemOSVersion,
        // modules
        LoadLibraryW,
        LoadLibraryA,
        LoadLibraryExW,
        LoadLibraryExA,
        LoadPackagedLibrary,
        GetProcAddress,
        GetModuleHandleW,
        GetModuleHandleA,
        GetModuleHandleExW,
        GetModuleHandleExA,
        GetModuleFileNameW,
        GetModuleFileNameA,
        FreeLibrary,
        FreeLibraryWhenCallbackReturns,
        DisableThreadLibraryCalls,
    ]
}

// ---------------------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------------------

pub const MEM_GRAPHICS: u32 = 0x1000_0000;
mod placeholders;
pub mod lazy;
pub const MEM_TITLE: u32 = 0x4000_0000;
const STD_ALLOC_MASK: u32 = 0x1000 | 0x2000 | 0x8_0000 | 0x10_0000 | 0x20_0000 | 0x100_0000;
const STD_PROTECT_MASK: u32 = 0xFF | 0x100 | 0x200 | 0x400;

static WARNED_FLAGS: AtomicU32 = AtomicU32::new(0);

pub fn sanitize_alloc_type(t: u32) -> u32 {
    let s = t & STD_ALLOC_MASK;
    let dropped = t & !STD_ALLOC_MASK;
    if dropped != 0 && WARNED_FLAGS.fetch_or(dropped, Ordering::Relaxed) & dropped != dropped {
        crate::xlog_debug!("VirtualAlloc: dropping console allocation flags {dropped:#x}");
    }
    if s == 0 {
        0x1000 | 0x2000
    } else {
        s
    }
}

pub fn sanitize_protect(p: u32) -> u32 {
    let s = p & STD_PROTECT_MASK;
    if s & 0xFF == 0 {
        (s & !0xFF) | mem::PAGE_READWRITE
    } else {
        s
    }
}

/// Graphics address space the title reserved (no commit) that unified memory had no room for
/// – where it maps physical pages (a texture streaming pool): base → size. Write-watched, like
/// unified memory, so textures in it follow CPU writes (`d3d12x::title_textures`).
static GRAPHICS_RESERVATIONS: std::sync::Mutex<std::collections::BTreeMap<usize, usize>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
static GRAPHICS_RESERVATIONS_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Generation counter incremented whenever `GRAPHICS_RESERVATIONS` changes.
pub fn graphics_reservations_gen() -> u64 {
    GRAPHICS_RESERVATIONS_GEN.load(Ordering::Acquire)
}

/// The title's graphics reservations (base, size).
pub fn graphics_reservations() -> Vec<(usize, usize)> {
    GRAPHICS_RESERVATIONS.lock().unwrap().iter().map(|(&b, &s)| (b, s)).collect()
}

/// Whether `p` lies in one of the title's graphics reservations.
pub fn in_graphics_reservation(p: usize) -> bool {
    GRAPHICS_RESERVATIONS.lock().unwrap().range(..=p).next_back().is_some_and(|(&b, &s)| p < b + s)
}

const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// GPU views of the title's CPU memory. A retail game reserves 24 GB of 64 KiB pages twice
/// in a row: once readable by the CPU (PAGE_READWRITE), once with PAGE_NOACCESS |
/// PAGE_GRAPHICS_NOACCESS. It maps physical pages only into the first, yet places GPU
/// resources at the same offset in the second (its Bink video planes: the decoder writes the
/// first view, the video shader samples textures in the second). On the console the second
/// reservation is a GPU-only view of the first; here it is separate, empty memory – the
/// video was its colour of zero luma and chroma, a green screen. `source` names the CPU view
/// behind an address of such a GPU view, which `d3d12x::title_textures` reads and watches.
pub mod mirrors {
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{LazyLock, Mutex, RwLock};

    /// MEM_LARGE_PAGES | MEM_PHYSICAL: the console's 64 KiB pages
    const PAGES_64K: u32 = 0x2040_0000;
    /// PAGE_GRAPHICS_NOACCESS
    const GRAPHICS_NOACCESS: u32 = 0x800;

    struct State {
        /// CPU-readable reservations of 64 KiB pages, oldest first: (base, size)
        cpu: Vec<(usize, usize)>,
        /// GPU view base → (size, CPU view base)
        gpu: BTreeMap<usize, (usize, usize)>,
    }
    static STATE: Mutex<State> = Mutex::new(State { cpu: Vec::new(), gpu: BTreeMap::new() });
    /// The pairs in `STATE.gpu`: while there are none no GPU view shows CPU memory, and a
    /// decommit asks nothing more (`cpu_view_part`).
    static PAIRS: AtomicUsize = AtomicUsize::new(0);

    /// Whether `(t, p)` is a CPU-inaccessible GPU-view alias reservation (`MEM_64K_PAGES`
    /// reserve with `PAGE_GRAPHICS_NOACCESS | PAGE_NOACCESS`).
    pub fn is_gpu_view_reservation(t: u32, p: u32) -> bool {
        t & PAGES_64K == PAGES_64K && t & 0x1000 == 0 && p & GRAPHICS_NOACCESS != 0 && p & 0xFF == super::mem::PAGE_NOACCESS
    }

    /// The most recent CPU view of `size` bytes without a GPU view yet.
    fn unpaired_cpu(s: &State, size: usize) -> Option<usize> {
        let taken: Vec<usize> = s.gpu.values().map(|&(_, c)| c).collect();
        s.cpu.iter().rev().find(|&&(b, n)| n == size && !taken.contains(&b)).map(|&(b, _)| b)
    }

    /// Whether a GPU-view reservation of `size` bytes made now would be paired (`reserved`): a
    /// CPU view of that size waits for its GPU view.
    pub fn would_pair(size: usize) -> bool {
        unpaired_cpu(&STATE.lock().unwrap(), size).is_some()
    }

    /// A reservation of `size` bytes at `base` with allocation type `t` and protection `p`.
    /// True: it is a GPU view and was paired with its CPU view.
    pub fn reserved(base: usize, size: usize, t: u32, p: u32) -> bool {
        if t & PAGES_64K != PAGES_64K || t & 0x1000 != 0 {
            return false;
        }
        let mut s = STATE.lock().unwrap();
        if p & GRAPHICS_NOACCESS != 0 && p & 0xFF == super::mem::PAGE_NOACCESS {
            if let Some(cpu) = unpaired_cpu(&s, size) {
                if s.gpu.insert(base, (size, cpu)).is_none() {
                    PAIRS.fetch_add(1, Ordering::Release);
                }
                drop(s);
                crate::xlog_info!("kernelx: reservation {base:#x} ({size:#x} bytes, no CPU access) is the GPU view of {cpu:#x}");
                // what `source` says for these addresses changed (readers of GPU views look again)
                super::note_gpu_view_change(base, size);
                return true;
            }
        } else if p & 0xFF != super::mem::PAGE_NOACCESS {
            s.cpu.push((base, size));
        }
        false
    }

    /// The reservation at `base` was released (before its address can be handed out again: a
    /// new reservation there would otherwise still be paired). A pair that goes takes the
    /// tiles that showed its CPU view with it (`sparse::forget_mirrors`, told with this lock
    /// dropped: sparse asks `source` with its own held), and both addresses may be given out anew.
    pub fn released(base: usize) {
        let mut s = STATE.lock().unwrap();
        s.cpu.retain(|&(b, _)| b != base);
        // (GPU view, size, CPU view)
        let mut gone: Vec<(usize, usize, usize)> = s.gpu.remove(&base).map(|(n, c)| (base, n, c)).into_iter().collect();
        s.gpu.retain(|&g, &mut (n, c)| {
            if c == base {
                gone.push((g, n, c));
            }
            c != base
        });
        PAIRS.fetch_sub(gone.len(), Ordering::Release);
        // the pages the page-mapping method named in the views that go (with the views' lock held)
        if !gone.is_empty() && MAPPED.load(Ordering::Acquire) != 0 {
            let mut p = write_pages();
            for &(g, n, _) in &gone {
                let named: Vec<usize> = p.shown.range(g..g + n).map(|(&k, _)| k).collect();
                for k in named {
                    p.unset(k);
                }
            }
        }
        drop(s);
        for (g, n, c) in gone {
            super::note_gpu_view_change(g, n);
            for (a, len) in [(g, n), (c, n)] {
                #[cfg(test)]
                super::tests::FORGOTTEN.with(|f| f.borrow_mut().push((a, len)));
                crate::hle::d3d12x::sparse::forget_mirrors(a, len);
            }
        }
    }

    /// The part of `[a, a + n)` (`n == 0`: from `a` to the end of its reservation) that lies in
    /// the CPU view of a pair – memory whose pages tiles of its GPU view may show – as (start,
    /// bytes). One atomic load while nothing is paired.
    pub fn cpu_view_part(a: usize, n: usize) -> Option<(usize, usize)> {
        if PAIRS.load(Ordering::Acquire) == 0 {
            return None;
        }
        let s = STATE.lock().unwrap();
        s.gpu.values().find_map(|&(size, c)| {
            let end = c + size;
            (a >= c && a < end).then(|| (a, if n == 0 { end - a } else { n.min(end - a) }))
        })
    }

    /// The CPU-view address behind `va` when `va` lies in a GPU view.
    pub fn source(va: usize) -> Option<usize> {
        let s = STATE.lock().unwrap();
        let (&gpu, &(size, cpu)) = s.gpu.range(..=va).next_back()?;
        (va < gpu + size).then(|| cpu + (va - gpu))
    }

    /// All GPU-view addresses paired with `cpu_va` (reverse of `source`).
    pub fn gpu_views_of(cpu_va: usize) -> Vec<usize> {
        let s = STATE.lock().unwrap();
        s.gpu
            .iter()
            .filter_map(|(&gpu, &(size, cpu))| (cpu_va >= cpu && cpu_va < cpu + size).then(|| gpu + (cpu_va - cpu)))
            .collect()
    }

    /// Bytes of a page the queue's page-mapping method maps (the console's 64 KiB pages).
    pub const PAGE: usize = 0x1_0000;

    /// The mappings of one call of the queue's page-mapping method, `CopyPageMappingsBatchX(count,
    /// batches, pool, pool pages, …)` (queue #29; #31 in the newer layout), checked whole: (GPU-view
    /// page, the pool page it is given or `None`: unmapped), in the call's order. `count` batches
    /// of 24 bytes – the first destination address (u64), the number of ranges (u32), a pointer
    /// to them at +16 – each range three u32: the kind (0: `count` pool pages from `start` on,
    /// 1: pool page `start` at each of `count` pages, 2: unmapped, 3: left as they are), `start`
    /// and `count`. The pool is `pool_pages` pages of `PAGE` at `pool`. `None` when a batch or a
    /// range is not readable, a range leaves the pool or has an unknown kind, or the call names
    /// over `MOST` pages. No lock.
    pub unsafe fn decode_batches(count: u32, batches: usize, pool: usize, pool_pages: u32) -> Option<Vec<(usize, Option<usize>)>> {
        const BATCH: usize = 24;
        const RANGE: usize = 12;
        const MOST: usize = 1 << 20;
        let readable = crate::gpu::shader::readable;
        let n = count as usize;
        if n == 0 {
            return Some(Vec::new());
        }
        if n > MOST || !readable(batches, n * BATCH) {
            return None;
        }
        let mut changes: Vec<(usize, Option<usize>)> = Vec::new();
        for b in 0..n {
            let p = batches + b * BATCH;
            let dest = (p as *const u64).read_unaligned() as usize & !(PAGE - 1);
            let ranges_n = ((p + 8) as *const u32).read_unaligned() as usize;
            let ranges = ((p + 16) as *const usize).read_unaligned();
            if ranges_n == 0 {
                continue;
            }
            if ranges_n > MOST || !readable(ranges, ranges_n * RANGE) {
                return None;
            }
            let mut at = dest;
            for r in 0..ranges_n {
                let q = (ranges + r * RANGE) as *const u32;
                let (kind, start, len) = (q.read_unaligned(), q.add(1).read_unaligned() as usize, q.add(2).read_unaligned() as usize);
                let inside = |k: usize| start.checked_add(k).is_some_and(|e| e <= pool_pages as usize);
                if changes.len() + len > MOST || at.checked_add(len * PAGE).is_none() {
                    return None;
                }
                match kind {
                    0 if inside(len) => changes.extend((0..len).map(|k| (at + k * PAGE, Some(pool + (start + k) * PAGE)))),
                    1 if start < pool_pages as usize => changes.extend((0..len).map(|k| (at + k * PAGE, Some(pool + start * PAGE)))),
                    2 => changes.extend((0..len).map(|k| (at + k * PAGE, None))),
                    3 => {}
                    _ => return None,
                }
                at += len * PAGE;
            }
        }
        Some(changes)
    }

    /// What a GPU-view page the page-mapping method named shows (`apply_batches`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Shown {
        /// the physical page (0: nothing)
        pub page: usize,
        /// where XCL keeps that page's bytes now (0: nothing)
        pub at: usize,
        /// the place is memory whose decommit and release kernelx sees: a tile may show it
        pub tile: bool,
        /// the call (`calls`) that last changed what it shows
        pub call: u64,
        /// its key in `Pages::log`
        pub seq: u64,
    }

    /// The record of the page-mapping method: the GPU-view pages it named, and what leads to them.
    #[derive(Default)]
    struct Pages {
        /// GPU-view page → what it shows
        shown: BTreeMap<usize, Shown>,
        /// physical page → the GPU-view pages that show it
        by_page: HashMap<usize, Vec<usize>>,
        /// (place, GPU-view page); pages with no place are not in it
        by_at: BTreeSet<(usize, usize)>,
        /// 16 MiB block of a place (`place >> 24`) → the pages kept in it
        blocks: HashMap<usize, u32>,
        /// seq → GPU-view page: one entry a page, its last change
        log: BTreeMap<u64, usize>,
        seq: u64,
    }

    impl Pages {
        /// `g` shows `page`, kept at `at`, from `call` on. False, and nothing logged, when it
        /// shows that page at that place already (the title named the mapping it has).
        fn set(&mut self, g: usize, page: usize, at: usize, tile: bool, call: u64) -> bool {
            if self.shown.get(&g).is_some_and(|e| e.page == page && e.at == at) {
                return false;
            }
            self.unlink(g);
            self.seq += 1;
            let seq = self.seq;
            if page != 0 {
                self.by_page.entry(page).or_default().push(g);
            }
            self.link_at(at, g);
            self.log.insert(seq, g);
            self.shown.insert(g, Shown { page, at, tile, call, seq });
            MAPPED.store(self.shown.len(), Ordering::Release);
            CHANGE_SEQ.store(seq, Ordering::Release);
            true
        }

        /// `g` leaves the record (its view went).
        fn unset(&mut self, g: usize) {
            self.unlink(g);
            self.shown.remove(&g);
            MAPPED.store(self.shown.len(), Ordering::Release);
        }

        /// `g`'s entry out of the indexes and the log (it stays in `shown`).
        fn unlink(&mut self, g: usize) {
            let Some(e) = self.shown.get(&g).copied() else { return };
            if let Some(v) = self.by_page.get_mut(&e.page) {
                v.retain(|&x| x != g);
                if v.is_empty() {
                    self.by_page.remove(&e.page);
                }
            }
            self.unlink_at(e.at, g);
            self.log.remove(&e.seq);
        }

        /// `g` is kept at `at` (0: nothing) in `by_at` and `blocks`.
        fn link_at(&mut self, at: usize, g: usize) {
            if at != 0 {
                self.by_at.insert((at, g));
                let n = self.blocks.entry(at >> 24).or_insert(0);
                *n += 1;
                if *n == 1 {
                    SHOWN_GEN.fetch_add(1, Ordering::Release);
                }
            }
        }

        /// `g` is no longer kept at `at` (0: nothing) in `by_at` and `blocks`.
        fn unlink_at(&mut self, at: usize, g: usize) {
            if at != 0 {
                self.by_at.remove(&(at, g));
                if let Some(n) = self.blocks.get_mut(&(at >> 24)) {
                    *n -= 1;
                    if *n == 0 {
                        self.blocks.remove(&(at >> 24));
                        SHOWN_GEN.fetch_add(1, Ordering::Release);
                    }
                }
            }
        }

        /// The page `g` shows is kept at `at` now (0: nowhere), with `tile` for it. A move
        /// keeps its call and its place in the log; `changed` (a call) logs it as a change of
        /// what `g` shows: its bytes appear or go.
        fn place(&mut self, g: usize, at: usize, tile: bool, changed: Option<u64>) {
            let Some(mut e) = self.shown.get(&g).copied() else { return };
            self.unlink_at(e.at, g);
            self.link_at(at, g);
            (e.at, e.tile) = (at, tile);
            if let Some(call) = changed {
                self.log.remove(&e.seq);
                self.seq += 1;
                (e.seq, e.call) = (self.seq, call);
                self.log.insert(e.seq, g);
                CHANGE_SEQ.store(e.seq, Ordering::Release);
            }
            self.shown.insert(g, e);
        }
    }

    /// The record, under a lock of its own: `STATE` → `PAGES` where both are held (`apply_batches`,
    /// `released`); every reader takes it alone. Nothing is called with it held.
    static PAGES: LazyLock<RwLock<Pages>> = LazyLock::new(Default::default);
    /// The pages in the record: while there are none, a reader takes today's path after one
    /// atomic load.
    static MAPPED: AtomicUsize = AtomicUsize::new(0);
    /// The calls of the page-mapping method that were recorded.
    static CALLS: AtomicU64 = AtomicU64::new(0);
    /// `Pages::seq`: the last change of the record.
    static CHANGE_SEQ: AtomicU64 = AtomicU64::new(0);
    /// Counts the times a 16 MiB block came to hold the place of a shown page, or ceased to.
    static SHOWN_GEN: AtomicU64 = AtomicU64::new(0);
    /// GPU-view pages whose page moved, and whose place followed it (`page_at`).
    static MOVED: AtomicU64 = AtomicU64::new(0);

    fn read_pages() -> std::sync::RwLockReadGuard<'static, Pages> {
        PAGES.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write_pages() -> std::sync::RwLockWriteGuard<'static, Pages> {
        PAGES.write().unwrap_or_else(|e| e.into_inner())
    }

    /// What one `apply_batches` call did, in 64 KiB pages.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Mapped {
        /// given a pool page with a physical page the title mapped there
        pub mapped: u64,
        /// unmapped
        pub unmapped: u64,
        /// given a pool page with no physical page mapped there (or one with no place): they
        /// show nothing
        pub empty: u64,
        /// outside every GPU view: not recorded
        pub elsewhere: u64,
    }

    /// One call of the queue's page-mapping method (`decode_batches`): each page of a GPU view it
    /// names records the physical page the title had mapped at its pool page then (the console
    /// copies the mapping at the call; `Away::title`) and where XCL keeps that page's bytes now.
    /// A pool page with no page mapped shows nothing. `None` – nothing recorded – for a call
    /// `decode_batches` refuses. AWAY is held from the lookup to the record, so a page that moves
    /// in between cannot be recorded at the place it left.
    pub unsafe fn apply_batches(count: u32, batches: usize, pool: usize, pool_pages: u32) -> Option<Mapped> {
        let changes = decode_batches(count, batches, pool, pool_pages)?;
        let call = CALLS.fetch_add(1, Ordering::AcqRel) + 1;
        let away = super::AWAY.lock().unwrap();
        // (GPU-view page, None: unmapped, or (physical page, place, tile) with 0 for none)
        let changes: Vec<(usize, Option<(usize, usize, bool)>)> = changes
            .into_iter()
            .map(|(g, to)| {
                let to = to.map(|pool| {
                    let page = away.as_ref().and_then(|a| a.title.get(&pool).copied()).unwrap_or(0);
                    let (at, home) = if page != 0 { super::place_of(away.as_ref(), page) } else { (0, false) };
                    (page, at, at != 0 && (home || super::freeing_is_seen(at)))
                });
                (g, to)
            })
            .collect();
        // (the views stay while the record is written)
        let s = STATE.lock().unwrap();
        let in_view = |va: usize| s.gpu.range(..=va).next_back().is_some_and(|(&g, &(size, _))| va < g + size);
        let mut pages = write_pages();
        let mut out = Mapped::default();
        // (the GPU-view pages that show something else now)
        let mut changed: Vec<usize> = Vec::new();
        for (g, to) in changes {
            if !in_view(g) {
                out.elsewhere += 1;
                continue;
            }
            let set = match to {
                None => {
                    out.unmapped += 1;
                    pages.set(g, 0, 0, false, call)
                }
                Some((page, at, _)) if page == 0 || at == 0 => {
                    out.empty += 1;
                    pages.set(g, 0, 0, false, call)
                }
                Some((page, at, tile)) => {
                    out.mapped += 1;
                    pages.set(g, page, at, tile, call)
                }
            };
            if set {
                changed.push(g);
            }
        }
        drop(pages);
        drop(s);
        // (the tiles over them are linked again at the next submit)
        crate::hle::d3d12x::sparse::remapped(&changed);
        drop(away);
        Some(out)
    }

    /// What an address shows to the GPU (`shows`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Shows {
        /// not in a GPU view
        NoView,
        /// a page the page-mapping method never named: the CPU view at the same offset (this
        /// address of it)
        Same(usize),
        /// the mapped page: this byte's address where it is kept; `tile`: a tile may show it
        Page { at: usize, tile: bool },
        /// nothing (zeros): unmapped, or no page behind it
        Nothing,
    }

    /// What `va` shows. One atomic load more than `source` while nothing is mapped.
    pub fn shows(va: usize) -> Shows {
        if MAPPED.load(Ordering::Acquire) != 0 {
            let p = read_pages();
            if let Some(e) = p.shown.get(&(va & !(PAGE - 1))) {
                return if e.at == 0 { Shows::Nothing } else { Shows::Page { at: e.at + (va & (PAGE - 1)), tile: e.tile } };
            }
        }
        source(va).map_or(Shows::NoView, Shows::Same)
    }

    /// The first and the last 64 KiB page of `[va, va + len)` (`len` 0: one byte).
    fn span(va: usize, len: usize) -> (usize, usize) {
        (va & !(PAGE - 1), va.saturating_add(len.max(1) - 1) & !(PAGE - 1))
    }

    /// Where the pages of `[va, va + len)` are read: `None` when the page-mapping method named
    /// none of them. Else `at` gets one address for each 64 KiB page from `va`'s on: where the
    /// mapped page is kept, 0 for nothing, and for a page never named the CPU view's same
    /// offset (outside a view the page itself) with bit 0 set (read as any title memory). The
    /// counts: [kept elsewhere, kept at the CPU view's same offset, never named, nothing].
    pub fn resolve(va: usize, len: usize, at: &mut Vec<usize>) -> Option<[u32; 4]> {
        at.clear();
        if MAPPED.load(Ordering::Acquire) == 0 {
            return None;
        }
        let (first, last) = span(va, len);
        let view = STATE.lock().unwrap().gpu.range(..=first).next_back().map(|(&g, &(size, c))| (g, size, c));
        let same = |g: usize| match view {
            Some((b, size, c)) if g >= b && g < b + size => c + (g - b),
            _ => g,
        };
        let p = read_pages();
        if p.shown.range(first..=last).next().is_none() {
            return None;
        }
        let mut counts = [0u32; 4];
        let mut g = first;
        loop {
            at.push(match p.shown.get(&g) {
                None => {
                    counts[2] += 1;
                    same(g) | 1
                }
                Some(e) if e.at == 0 => {
                    counts[3] += 1;
                    0
                }
                Some(e) => {
                    counts[(e.at == same(g)) as usize] += 1;
                    e.at
                }
            });
            if g == last {
                break;
            }
            g += PAGE;
        }
        Some(counts)
    }

    /// Whether the page-mapping method named a page of `[va, va + len)`.
    pub fn touched(va: usize, len: usize) -> bool {
        if MAPPED.load(Ordering::Acquire) == 0 {
            return false;
        }
        let (first, last) = span(va, len);
        read_pages().shown.range(first..=last).next().is_some()
    }

    /// Whether any page of a GPU view is in the record.
    pub fn any_mapped() -> bool {
        MAPPED.load(Ordering::Acquire) != 0
    }

    /// The calls of the page-mapping method recorded so far.
    pub fn calls() -> u64 {
        CALLS.load(Ordering::Acquire)
    }

    /// Whether the queue's page-mapping method is applied (`apply_batches`). Off
    /// (XCL_PAGE_MAPPINGS=0) its queue slot is the stub it was and the record stays empty.
    pub fn page_mappings() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            let on = page_mappings_on(std::env::var("XCL_PAGE_MAPPINGS").ok().as_deref());
            if !on {
                crate::xlog_info!("kernelx: XCL_PAGE_MAPPINGS=0 – the queue's page mappings (CopyPageMappingsBatchX) are left out: a GPU-view page shows the CPU view at the same offset");
            }
            on
        })
    }

    /// XCL_PAGE_MAPPINGS `v`: on unless it is 0.
    pub fn page_mappings_on(v: Option<&str>) -> bool {
        v.map_or(true, |v| v.trim() != "0")
    }

    /// XCL keeps physical page `page`'s bytes at `at` now (0: nowhere; `tile`: a tile may show
    /// it there): the GPU-view pages that show it follow. A move is no change of what they show
    /// (their call and their place in the log stay), but a page that gets its first place, or
    /// loses it, is: its bytes appear or go. `None` when no page shows it elsewhere (one atomic
    /// load while nothing is recorded); else the pages that followed, each with where it was kept
    /// and whether a tile could show it there (`Shown::at`, `Shown::tile`).
    /// Called on the moving thread with AWAY held, before the memory it left can go.
    pub(super) fn page_at(page: usize, at: usize, tile: bool) -> Option<Vec<(usize, usize, bool)>> {
        if MAPPED.load(Ordering::Acquire) == 0 || !read_pages().by_page.contains_key(&page) {
            return None;
        }
        let mut p = write_pages();
        let moving: Vec<(usize, usize, bool)> =
            p.by_page.get(&page)?.iter().filter_map(|&g| p.shown.get(&g).filter(|e| e.at != at || e.tile != tile).map(|e| (g, e.at, e.tile))).collect();
        if moving.is_empty() {
            return None;
        }
        let call = calls();
        for &(g, was, _) in &moving {
            p.place(g, at, tile, ((was == 0) != (at == 0)).then_some(call));
        }
        drop(p);
        MOVED.fetch_add(moving.len() as u64, Ordering::Relaxed);
        Some(moving)
    }

    /// Physical page `page` was freed: the GPU-view pages that showed it show nothing from here
    /// on (a change of what they show). Returns them. Caller holds AWAY.
    pub(super) fn page_freed(page: usize) -> Vec<usize> {
        if MAPPED.load(Ordering::Acquire) == 0 || !read_pages().by_page.contains_key(&page) {
            return Vec::new();
        }
        let mut p = write_pages();
        let gone = p.by_page.get(&page).cloned().unwrap_or_default();
        let call = calls();
        for &g in &gone {
            p.set(g, 0, 0, false, call);
        }
        gone
    }

    /// The memory `[base, base + len)` was released: the shown pages kept there have no place
    /// from now on (they show nothing until their page is placed again; a change of what they
    /// show). Returns those GPU-view pages.
    pub(super) fn place_gone(base: usize, len: usize) -> Vec<usize> {
        if MAPPED.load(Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let end = base.saturating_add(len);
        if read_pages().by_at.range((base, 0)..(end, 0)).next().is_none() {
            return Vec::new();
        }
        let mut p = write_pages();
        let gone: Vec<usize> = p.by_at.range((base, 0)..(end, 0)).map(|&(_, g)| g).collect();
        let call = calls();
        for &g in &gone {
            p.place(g, 0, false, Some(call));
        }
        gone
    }

    /// The GPU-view pages in `[lo, hi)` whose content changed after cursor `seen` (0 at first),
    /// each once, with the call that last changed it, in the order of their last change – at
    /// most `most` – and the cursor to pass next. Nothing is drained: each reader keeps a cursor
    /// of its own. One atomic load when nothing changed.
    pub fn changes_since(seen: u64, lo: usize, hi: usize, most: usize) -> (u64, Vec<(usize, u64)>) {
        let mut out = Vec::new();
        if CHANGE_SEQ.load(Ordering::Acquire) == seen {
            return (seen, out);
        }
        let p = read_pages();
        let mut cursor = seen;
        for (&seq, &g) in p.log.range(seen.saturating_add(1)..) {
            if g >= lo && g < hi {
                if out.len() >= most {
                    return (cursor, out);
                }
                out.push((g, p.shown.get(&g).map_or(0, |e| e.call)));
            }
            cursor = seq;
        }
        (p.seq.max(cursor), out)
    }

    /// The GPU-view pages a CPU write reaches, added to `out` for `dirty` (sorted 4 KiB pages
    /// written): each written page of a place where a mapped page is kept, at every GPU-view
    /// page that shows it; and a written page of a CPU view at its GPU-view page when the
    /// page-mapping method never named that one (it shows the CPU view at the same offset). A
    /// GPU-view page that is mapped or unmapped gets nothing from the CPU view under it.
    pub fn views_of_written(dirty: &[usize], out: &mut Vec<usize>) {
        if MAPPED.load(Ordering::Acquire) == 0 || dirty.is_empty() {
            return;
        }
        // (GPU view, size, CPU view)
        let pairs: Vec<(usize, usize, usize)> = STATE.lock().unwrap().gpu.iter().map(|(&g, &(n, c))| (g, n, c)).collect();
        let p = read_pages();
        let mut i = 0;
        while i < dirty.len() {
            let page = dirty[i] & !(PAGE - 1);
            let n = dirty[i..].iter().take_while(|&&d| d & !(PAGE - 1) == page).count();
            let written = &dirty[i..i + n];
            for &(_, g) in p.by_at.range((page, 0)..=(page, usize::MAX)) {
                out.extend(written.iter().map(|&d| g + (d - page)));
            }
            for &(gv, size, c) in &pairs {
                if page >= c && page < c + size && !p.shown.contains_key(&(gv + (page - c))) {
                    out.extend(written.iter().map(|&d| gv + (d - c)));
                }
            }
            i += n;
        }
    }

    /// The 16 MiB blocks that hold the place of a shown page, as (start, end), in order.
    pub fn shown_blocks() -> Vec<(usize, usize)> {
        if MAPPED.load(Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let mut v: Vec<(usize, usize)> = read_pages().blocks.keys().map(|&b| (b << 24, (b + 1) << 24)).collect();
        v.sort_unstable();
        v
    }

    /// Changes each time a 16 MiB block comes to hold the place of a shown page, or ceases to.
    pub fn shown_gen() -> u64 {
        SHOWN_GEN.load(Ordering::Acquire)
    }

    /// The GPU-view pages that followed their physical page to another place so far.
    pub fn moved_pages() -> u64 {
        MOVED.load(Ordering::Relaxed)
    }

    /// What the record says of GPU-view page `g` (a test's own pages).
    #[cfg(test)]
    pub(super) fn shown_at(g: usize) -> Option<Shown> {
        read_pages().shown.get(&g).copied()
    }
}

/// Where XCL keeps page `page`'s bytes now: the address the title mapped it at last, else its
/// home (`true`); `(0, false)` when no allocated page has that number. Caller holds AWAY.
fn place_of(away: Option<&Away>, page: usize) -> (usize, bool) {
    if let Some(&at) = away.and_then(|a| a.page.get(&page)) {
        return (at, false);
    }
    page_home(page).map_or((0, false), |h| (h, true))
}

/// Whether kernelx sees memory at `at` decommitted or released, and can tell sparse first:
/// unified memory or a graphics reservation (a tile of a GPU view may show such a place).
fn freeing_is_seen(at: usize) -> bool {
    crate::gpu::uma::get().is_some_and(|u| u.contains(at)) || in_graphics_reservation(at)
}

/// XCL keeps physical page `page`'s bytes at `at` now (0: nowhere; `home`: its home): the
/// GPU-view pages that show it follow (`mirrors::page_at`). `whole`: its 64 KiB were copied
/// there, a known write at the new address – a CPU write made just before the move went with the
/// page and has no other report. Call it with AWAY held, once the bytes are in place and before
/// the memory the page left can go.
unsafe fn page_placed(page: usize, at: usize, home: bool, whole: bool) {
    if !mirrors::any_mapped() {
        return;
    }
    let tile = at != 0 && (home || freeing_is_seen(at));
    let Some(moving) = mirrors::page_at(page, at, tile) else { return };
    // the tiles linked over the place it left follow it, before that memory can go: those of
    // the view pages kept where a tile could show them, and now kept where one may. The others
    // are linked again (to nothing, or to the new place): their bytes appear or go, or no tile
    // was linked where they were – a page that leaves plain memory for memory a tile may show
    let from = if tile { moving.iter().find(|m| m.1 != 0 && m.2).map_or(0, |m| m.1) } else { 0 };
    let (mut follow, mut relink): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    for &(g, was, could) in &moving {
        let to = if from != 0 && was == from && could { &mut follow } else { &mut relink };
        to.push(g);
    }
    #[cfg(test)]
    tests::PLACED.with(|s| s.borrow_mut().push((page, from, at, follow.clone(), relink.clone())));
    if !relink.is_empty() {
        crate::hle::d3d12x::sparse::remapped(&relink);
    }
    if !follow.is_empty() {
        crate::hle::d3d12x::sparse::moved(&follow, from, at, whole);
    }
    if whole {
        crate::hle::d3d12x::title_textures::note_known_write(at, PHYS_PAGE);
    }
}

/// Graphics memory (MEM_GRAPHICS, committed, no fixed address) comes from the unified
/// memory region the GPU uses directly (`gpu::uma`); a graphics reservation (no commit) is
/// write-watched address space of Windows; everything else from Windows.
/// The console's MEM_64K_PAGES (MEM_LARGE_PAGES | MEM_PHYSICAL): title memory of 64 KiB
/// pages – GPU-visible on the console like all title memory.
const MEM_64K_PAGES: u32 = 0x2040_0000;

/// Commit `[a, a + n)` of a sparse window (the title's reservations there) and give it video
/// memory (`d3d12x::sparse`).
unsafe fn commit_sparse(a: usize, n: usize, p: u32) -> *mut c_void {
    let protection = sanitize_protect(p);
    let q = mem::VirtualAlloc(a as *const c_void, n, mem::MEM_COMMIT, protection);
    if !q.is_null() {
        crate::hle::d3d12x::sparse::committed_with_protection(q as usize, n, read_write(protection));
    }
    q
}

/// A writable protection without PAGE_GUARD.
fn read_write(protection: u32) -> bool {
    matches!(protection & 0xFF, 0x04 | 0x08 | 0x40 | 0x80) && protection & 0x100 == 0
}

fn own_process(process: HANDLE) -> bool {
    if process == unsafe { thr::GetCurrentProcess() } {
        return true;
    }
    let id = unsafe { thr::GetProcessId(process) };
    id != 0 && id == unsafe { thr::GetCurrentProcessId() }
}

unsafe fn note_protection(addr: *const c_void, size: usize, protection: u32) {
    crate::hle::d3d12x::sparse::protection_changed(addr as usize, size, read_write(protection));
}

pub unsafe extern "system" fn VirtualAlloc(addr: *const c_void, size: usize, t: u32, p: u32) -> *mut c_void {
    let _cost = super::api_cost::time(super::api_cost::Call::Memory);
    if let Some(q) = placeholders::alloc(addr, size, t, sanitize_protect(p)) {
        trace_mem("VirtualAlloc (placeholder)", addr as usize, size, t as u64, p, q as usize);
        return q;
    }
    let uma = crate::gpu::uma::get();
    // committing inside one of the title's reservations in a sparse window
    if !addr.is_null() && t & 0x1000 != 0 && uma.is_some_and(|u| u.is_sparse(addr as usize)) {
        let q = commit_sparse(addr as usize, size, p);
        trace_mem("VirtualAlloc (sparse commit)", addr as usize, size, t as u64, p, q as usize);
        return q;
    }
    // committing in a GPU view (a view window): plain pages, as before the views had windows –
    // never the sparse windows' commit, whose tiles would take these pages for the view's bytes
    if !addr.is_null() && t & 0x1000 != 0 && uma.is_some_and(|u| u.is_view(addr as usize)) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::xlog_warn!("kernelx: the title commits memory in a GPU view ({addr:?}, {size:#x} bytes): plain pages there, which the GPU does not see"));
    }
    // a GPU view of the title's CPU memory (`mirrors`) with a CPU view waiting for it: in the view
    // windows (16 windows), where shaders reach it; the CPU view stays where it is. Taken before
    // the 64 KiB-page branch below: that one is for the CPU view, which is reserved first.
    if addr.is_null() && t & 0x2000 != 0 && mirrors::is_gpu_view_reservation(t, p) && mirrors::would_pair(size) {
        if let Some((u, q)) = uma.and_then(|u| Some((u, u.view_reserve(size)?))) {
            if mirrors::reserved(q, size, t, p) {
                trace_mem("VirtualAlloc (GPU view, view windows)", 0, size, t as u64, p, q);
                return q as *mut c_void;
            }
            // (another thread paired that CPU view in the meantime: as before the windows)
            u.view_release(q);
        }
    }
    // the title's own reservations of 64 KiB pages: in a sparse window, so the GPU sees them.
    // Skip CPU-inaccessible GPU-view mirror reservations (`PAGE_NOACCESS | PAGE_GRAPHICS_NOACCESS`):
    // their pixels live in the paired CPU view (`mirrors::source`), so putting them in sparse VA
    // wastes sparse address space and write-watch overhead while still needing mirror pairing.
    if addr.is_null() && t & 0x2000 != 0 && t & MEM_64K_PAGES == MEM_64K_PAGES && !mirrors::is_gpu_view_reservation(t, p) {
        if let Some(q) = uma.and_then(|u| u.sparse_reserve(size)) {
            if t & 0x1000 != 0 && commit_sparse(q, size, p).is_null() {
                uma.map(|u| u.sparse_release(q));
                return std::ptr::null_mut();
            }
            mirrors::reserved(q, size, t, p);
            trace_mem("VirtualAlloc (64K pages, sparse)", 0, size, t as u64, p, q);
            return q as *mut c_void;
        }
    }
    if addr.is_null() && t & MEM_GRAPHICS != 0 && t & 0x1000 != 0 {
        let align = if t & 0x8000_0000 != 0 {
            4 << 20
        } else if t & 0x2000_0000 != 0 && t & 0x0040_0000 == 0 {
            2 << 20
        } else {
            crate::gpu::uma::GRAIN
        };
        let rounded = size.max(1).div_ceil(align) * align;
        if let Some((q, _)) = crate::hle::d3d12x::sparse::alloc(rounded, align) {
            trace_mem("VirtualAlloc (graphics, sparse VRAM)", 0, rounded, t as u64, p, q);
            return q as *mut c_void;
        }
        if let Some(q) = uma.and_then(|u| u.alloc(rounded, align)) {
            trace_mem("VirtualAlloc (graphics, UMA)", 0, rounded, t as u64, p, q as usize);
            return q as *mut c_void;
        }
    }
    if addr.is_null() && t & MEM_GRAPHICS != 0 && t & 0x2000 != 0 && !mirrors::is_gpu_view_reservation(t, p) {
        // on the console all graphics memory is GPU-visible, reserved address space too (titles
        // reserve a range, then commit or map pages and place resources in it): prefer the
        // sparse VRAM windows first so reservations consume zero system RAM until committed
        let reserved = if t & 0x8000_0000 != 0 {
            let rounded = size.max(1).div_ceil(4 << 20) * (4 << 20);
            uma.and_then(|u| u.sparse_alloc(rounded, 4 << 20)).map(|(q, _)| q)
        } else {
            uma.and_then(|u| u.sparse_reserve(size))
        };
        if let Some(q) = reserved {
            mirrors::reserved(q, size, t, p);
            trace_mem("VirtualAlloc (graphics reserve, sparse)", 0, size, t as u64, p, q);
            return q as *mut c_void;
        }
        if let Some(q) = uma.and_then(|u| u.reserve(size, 0x1_0000)) {
            mirrors::reserved(q as usize, size, t, p);
            trace_mem("VirtualAlloc (graphics reserve, UMA)", 0, size, t as u64, p, q as usize);
            return q as *mut c_void;
        }
        let q = mem::VirtualAlloc(addr, size, mem::MEM_RESERVE | MEM_WRITE_WATCH, sanitize_protect(p));
        if !q.is_null() {
            GRAPHICS_RESERVATIONS.lock().unwrap().insert(q as usize, size);
            GRAPHICS_RESERVATIONS_GEN.fetch_add(1, Ordering::Release);
            return q;
        }
    }
    // the title's GPU heaps (committed with a PAGE_GRAPHICS_* protection, write-combined: memory
    // the CPU writes and the GPU reads, as a PC game's upload heaps): in the coherent window, which
    // the GPU reads where it is – one copy (a retail game's 4.4 GB; as plain memory its shaders
    // read zeros, in the sparse windows video memory would have doubled them). The window is as
    // large as what the title put in it last time with this setting (`uma::coherent_hint`).
    if addr.is_null() && t & 0x1000 != 0 && p & PAGE_GRAPHICS_ANY != 0 && size <= crate::gpu::uma::MAX_SIZE && std::env::var("XCL_COHERENT_HEAPS").map_or(true, |v| v != "0") {
        let align = if t & 0x2000_0000 != 0 && t & 0x0040_0000 == 0 { 2 << 20 } else { crate::gpu::uma::GRAIN };
        static ASKED: AtomicUsize = AtomicUsize::new(0);
        // `XCL_COHERENT_HEAPS=sparse[:N]`: heaps of at least N MB (all with no N) go to the
        // sparse windows instead – video memory, the CPU's writes uploaded before each
        // submission (`d3d12x::sparse`). In the coherent window the GPU reads them over the bus,
        // uncached, and on NVIDIA its writes that do not fill whole lines run at ~0.1 GB/s
        // (#505). Costs a second copy in video memory and the write-watch scan of it.
        if sparse_heaps_min().is_some_and(|min| size >= min) {
            let rounded = size.max(1).div_ceil(align) * align;
            if let Some((q, _)) = crate::hle::d3d12x::sparse::alloc(rounded, align) {
                crate::xlog_info!("kernelx: a GPU heap of {} MB in the sparse windows (XCL_COHERENT_HEAPS)", size >> 20);
                trace_mem("VirtualAlloc (graphics heap, sparse VRAM)", 0, rounded, t as u64, p, q);
                // nothing of it in the coherent window, noted all the same: a run whose heaps all
                // come here stores what it put there too (`uma::note_coherent_need`)
                crate::gpu::uma::note_coherent_need(ASKED.load(Ordering::Relaxed));
                return q as *mut c_void;
            }
        }
        let asked = ASKED.fetch_add(size, Ordering::Relaxed) + size;
        crate::gpu::uma::note_coherent_need(asked);
        if let Some(u) = uma {
            // what XCL keeps of the window for itself (readback heaps; its per-draw arguments have
            // a pool of their own)
            const KEEP: usize = 96 << 20;
            let fits = u.size().saturating_sub(u.used()) >= size + KEEP;
            if let Some(q) = fits.then(|| u.alloc(size, align)).flatten() {
                trace_mem("VirtualAlloc (graphics, coherent window)", 0, size, t as u64, p, q as usize);
                return q as *mut c_void;
            }
            // diagnostic: a GPU heap the coherent window has no room for (plain memory from here)
            static UNFIT: AtomicUsize = AtomicUsize::new(0);
            static UNFIT_BYTES: AtomicUsize = AtomicUsize::new(0);
            let k = UNFIT.fetch_add(1, Ordering::Relaxed) + 1;
            let total = UNFIT_BYTES.fetch_add(size, Ordering::Relaxed) + size;
            if k <= 8 || k.is_power_of_two() {
                let (free, _, used, _) = u.committed_state();
                crate::xlog_warn!("kernelx: a GPU heap of {} MB does not fit in the coherent window ({} MB used, {} MB free): #{k}, {} MB in all", size >> 20, used >> 20, free >> 20, total >> 20);
            }
        }
    }
    // GPU-visible title memory the windows above had no room for (a retail game: a 24 GB
    // reservation of 64 KiB pages, 2.2 GB of large pages with a graphics protection) is plain
    // memory here. Textures the title places in it are kept in step with it
    // (`d3d12x::title_textures`), which needs its CPU writes to be seen: it is write-watched
    // like a graphics reservation. Without that such a texture was uploaded once, empty, and
    // never again (the game's glyph atlas: its menus lost their x-height letters).
    if gpu_visible_fallback(addr, t, p) {
        let q = reserve_watched(size, t, p);
        if !q.is_null() {
            mirrors::reserved(q as usize, size, t, p);
            trace_mem("VirtualAlloc (GPU-visible, write-watched)", addr as usize, size, t as u64, p, q as usize);
            return q;
        }
    }
    let q = mem::VirtualAlloc(addr, size, sanitize_alloc_type(t), sanitize_protect(p));
    trace_mem("VirtualAlloc", addr as usize, size, t as u64, p, q as usize);
    q
}

/// `XCL_COHERENT_HEAPS=sparse` (every GPU heap) or `sparse:N` (heaps of at least N MB): the
/// smallest heap the sparse windows take instead of the coherent window. Read once.
fn sparse_heaps_min() -> Option<usize> {
    static MIN: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| std::env::var("XCL_COHERENT_HEAPS").ok().and_then(|v| parse_sparse_heaps(&v)))
}

/// Which of the title's GPU heaps go to the coherent window, as the window's size is kept for
/// each (`uma::coherent_hint_for`).
pub(crate) fn coherent_heaps_route() -> String {
    heaps_route(std::env::var("XCL_COHERENT_HEAPS").ok().as_deref())
}

/// `coherent_heaps_route` of a value of `XCL_COHERENT_HEAPS`, read as `VirtualAlloc` reads it:
/// `off` (`0`: none), `sparse` (only those the sparse windows have no room for), `sparse:N`
/// (those under N MB too), else every heap (`uma::ALL_HEAPS`).
fn heaps_route(v: Option<&str>) -> String {
    match v {
        Some("0") => "off".into(),
        Some(v) => match parse_sparse_heaps(v) {
            Some(0) => "sparse".into(),
            Some(min) => format!("sparse:{}", min >> 20),
            None => crate::gpu::uma::ALL_HEAPS.into(),
        },
        None => crate::gpu::uma::ALL_HEAPS.into(),
    }
}

/// `sparse` -> 0, `sparse:N` -> N MB in bytes; anything else -> None
fn parse_sparse_heaps(v: &str) -> Option<usize> {
    match v.trim().strip_prefix("sparse")? {
        "" => Some(0),
        rest => rest.strip_prefix(':')?.trim().parse::<usize>().ok().and_then(|mb| mb.checked_mul(1 << 20)),
    }
}

/// Whether a VirtualAlloc that reached the plain-memory fallback is GPU-visible title memory
/// (a reservation with MEM_GRAPHICS, MEM_LARGE_PAGES / MEM_64K_PAGES or a PAGE_GRAPHICS_*
/// protection, at no fixed address), which is reserved write-watched (`reserve_watched`).
fn gpu_visible_fallback(addr: *const c_void, t: u32, p: u32) -> bool {
    addr.is_null() && t & 0x2000 != 0 && (t & (MEM_GRAPHICS | 0x2000_0000) != 0 || p & PAGE_GRAPHICS_ANY != 0)
}

/// Plain memory for GPU-visible title memory, write-watched and recorded with the graphics
/// reservations (whose ranges `title_textures` watches; `VirtualFree` forgets it). Null if
/// Windows refuses it.
unsafe fn reserve_watched(size: usize, t: u32, p: u32) -> *mut c_void {
    let q = mem::VirtualAlloc(std::ptr::null(), size, sanitize_alloc_type(t) | MEM_WRITE_WATCH, sanitize_protect(p));
    if !q.is_null() {
        GRAPHICS_RESERVATIONS.lock().unwrap().insert(q as usize, size);
        GRAPHICS_RESERVATIONS_GEN.fetch_add(1, Ordering::Release);
    }
    q
}

/// PAGE_GRAPHICS_NOACCESS … PAGE_GRAPHICS_NOCACHE (0x800 – 0x40000): the console's protection of
/// GPU-visible memory
const PAGE_GRAPHICS_ANY: u32 = 0x7_F800;

/// `XCL_TRACE_MEM=1`: the title's allocations of 64 KiB or more (the first 60; `XCL_TRACE_MEM=n`:
/// the first n when n > 1)
fn trace_mem(what: &str, addr: usize, size: usize, flags: u64, protect: u32, got: usize) {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    static N: AtomicUsize = AtomicUsize::new(0);
    // the call being traced may have just set the title's last error: reading XCL_TRACE_MEM
    // (the first trace; 203 ERROR_ENVVAR_NOT_FOUND when unset) or logging must not change it
    keep_last_error(|| {
        let limit = *LIMIT.get_or_init(|| match std::env::var("XCL_TRACE_MEM") {
            Ok(v) => v.parse().ok().filter(|&n| n > 1).unwrap_or(60),
            Err(_) => 0,
        });
        if size >= 0x1_0000 && N.fetch_add(1, Ordering::Relaxed) < limit {
            crate::xlog_info!("mem: {what}({addr:#x}, {size:#x}, flags {flags:#x}, protect {protect:#x}) → {got:#x}");
        }
    })
}

/// `f` with the thread's last error as it was before.
fn keep_last_error<R>(f: impl FnOnce() -> R) -> R {
    let e = unsafe { fnd::GetLastError() };
    let r = f();
    unsafe { fnd::SetLastError(e) };
    r
}

/// The part of `[a, a + n)` (`n == 0`: to the end of the reservation at `a`) that lies in a
/// paired CPU view (`mirrors`): the pages `source_going` tells sparse about. One atomic load
/// while nothing is paired; else a look at the pairs. While the page-mapping method shows pages
/// in GPU views (`mirrors::any_mapped`) any memory may hold the place of one that a tile shows:
/// all of `[a, a + n)` then (sparse decides by its own index whether a tile shows it); one atomic
/// load more while nothing is mapped.
unsafe fn source_part(a: usize, n: usize) -> Option<(usize, usize)> {
    if let Some((lo, len)) = mirrors::cpu_view_part(a, n) {
        #[cfg(test)]
        tests::SOURCES_GOING.with(|s| s.borrow_mut().push((lo, len, committed_page(lo))));
        return Some((lo, len));
    }
    if n != 0 && mirrors::any_mapped() {
        #[cfg(test)]
        tests::KEPT_GOING.with(|s| s.borrow_mut().push((a, n, committed_page(a))));
        return Some((a, n));
    }
    None
}

/// `os` decommits (or, with `release`, releases) pages of `[a, a + n)` (`n == 0`: to the end of
/// the reservation at `a`) in Windows. Those of a paired CPU view (`mirrors`) may be shown by
/// tiles of its GPU view, which a submit fills from them and scans with the sparse state's lock
/// held – that read faults on a page that goes meanwhile. So sparse is told around the call
/// (`sparse::source_decommit` / `source_release`): the pages are holes for every submit from
/// before Windows takes them until after, and a submit that asks Windows about them in between
/// does not take them back. Telling it only before left that gap. Call it with no lock held
/// that sparse takes (AWAY is fine: AWAY → sparse is the order); `os` runs with no lock of
/// sparse's held.
unsafe fn source_going<R>(a: usize, n: usize, release: bool, os: impl FnOnce() -> R) -> R {
    source_part_going(source_part(a, n), release, os)
}

/// `source_going` for a part looked up before (`source_part`: a release unpairs the views
/// before Windows is called, and the part is not found after that).
unsafe fn source_part_going<R>(part: Option<(usize, usize)>, release: bool, os: impl FnOnce() -> R) -> R {
    match part {
        Some((lo, len)) if release => crate::hle::d3d12x::sparse::source_release(lo, len, os),
        Some((lo, len)) => crate::hle::d3d12x::sparse::source_decommit(lo, len, os),
        None => os(),
    }
}

/// Blocks of the unified memory region go back to it (MEM_RELEASE) or stay (MEM_DECOMMIT
/// zeroes them); others to Windows.
pub unsafe extern "system" fn VirtualFree(addr: *mut c_void, size: usize, t: u32) -> BOOL {
    let _cost = super::api_cost::time(super::api_cost::Call::Memory);
    if let Some(ok) = placeholders::free(addr, size, t) {
        return ok;
    }
    // the title's reservations in a sparse window: decommit (and on release give the address
    // space back); their video memory goes with it
    if let Some(u) = crate::gpu::uma::get().filter(|u| u.is_sparse(addr as usize)) {
        let a = addr as usize;
        let Some((n, release)) = reservation_part(a, size, t, u.sparse_reservation(a)) else { return 0 };
        if n > 0 {
            // sparse first: under its lock, so no copy of CPU writes to video memory reads the
            // pages while they go (the submit thread's copy faulted on them)
            crate::hle::d3d12x::sparse::decommitted(a, n);
            // (the CPU view of a pair may lie here too: tiles of its GPU view show these pages)
            if source_going(a, n, false, || mem::VirtualFree(addr, n, mem::MEM_DECOMMIT)) == 0 {
                return 0;
            }
        }
        if release {
            // unpaired before the address can be handed out again; and the pages kept here show
            // nothing from now on, before then too (a submit would link a tile over it)
            mirrors::released(a);
            forget_title_mappings(a, n);
            let gone = mirrors::place_gone(a, n);
            u.sparse_release(a);
            // (tiles may show pages kept here)
            if mirrors::any_mapped() {
                crate::hle::d3d12x::sparse::forget_mirrors(a, n);
            }
            crate::hle::d3d12x::sparse::remapped(&gone);
        }
        return 1;
    }
    // a GPU view in the view windows: what the GPU has of it goes first (`sparse::decommitted`
    // drops its tiles), then pages the title may have committed there; on release the pair, then
    // the address space
    if let Some(u) = crate::gpu::uma::get().filter(|u| u.is_view(addr as usize)) {
        let a = addr as usize;
        let Some((n, release)) = reservation_part(a, size, t, u.view_reservation(a)) else { return 0 };
        if n > 0 {
            crate::hle::d3d12x::sparse::decommitted(a, n);
            if mem::VirtualFree(addr, n, mem::MEM_DECOMMIT) == 0 {
                return 0;
            }
        }
        if release {
            mirrors::released(a);
            u.view_release(a);
            forget_title_mappings(a, n);
        }
        trace_mem(if release { "VirtualFree (GPU view, released)" } else { "VirtualFree (GPU view, decommitted)" }, a, n, t as u64, 0, 1);
        return 1;
    }
    if let Some(u) = crate::gpu::uma::get().filter(|u| u.contains(addr as usize)) {
        if t & 0x8000 != 0 {
            // (the block stays committed in its window: nothing goes in Windows, the tiles do; the
            // pages the title mapped here, and the shown pages kept here, go before the block can
            // be handed out again)
            let a = addr as usize;
            let n = u.block_len(a).unwrap_or(0);
            forget_title_mappings(a, n);
            let gone = mirrors::place_gone(a, n);
            source_going(a, n, true, || ());
            mirrors::released(a);
            crate::hle::d3d12x::sparse::remapped(&gone);
            return u.free(a) as BOOL;
        }
        if t & 0x4000 != 0 && size > 0 {
            std::ptr::write_bytes(addr as *mut u8, 0, size);
        }
        return 1;
    }
    // a decommit, or a release (of the whole reservation: size 0), of a paired CPU view – the
    // title's CPU view is plain write-watched memory – is told to sparse before Windows does it
    let release = t & mem::MEM_RELEASE != 0;
    // (looked up before the pair goes: a graphics reservation is unpaired below, before Windows;
    // size 0 is the whole of a graphics reservation, which may hold places of shown pages)
    let whole = if size == 0 { GRAPHICS_RESERVATIONS.lock().unwrap().get(&(addr as usize)).copied().unwrap_or(0) } else { size };
    let part = if release != (t & mem::MEM_DECOMMIT != 0) && !(release && size != 0) { source_part(addr as usize, whole) } else { None };
    // a graphics reservation is unpaired before Windows can hand its address out again
    let graphics = t & 0x8000 != 0 && size == 0 && GRAPHICS_RESERVATIONS.lock().unwrap().contains_key(&(addr as usize));
    if graphics {
        mirrors::released(addr as usize);
    }
    let ok = source_part_going(part, release, || mem::VirtualFree(addr, size, t));
    if ok != 0 && t & 0x8000 != 0 {
        let released = GRAPHICS_RESERVATIONS.lock().unwrap().remove(&(addr as usize));
        if let Some(len) = released {
            GRAPHICS_RESERVATIONS_GEN.fetch_add(1, Ordering::Release);
            if !graphics {
                mirrors::released(addr as usize);
            }
            forget_title_mappings(addr as usize, len);
            crate::hle::d3d12x::sparse::remapped(&mirrors::place_gone(addr as usize, len));
        }
    }
    ok
}

/// The part of the reservation `res` (start, bytes) that `VirtualFree(a, size, t)` gives back –
/// all of it on MEM_RELEASE or a decommit of size 0 – and whether it is a release. None: the
/// arguments do not name one (ERROR_INVALID_PARAMETER is set).
unsafe fn reservation_part(a: usize, size: usize, t: u32, res: Option<(usize, usize)>) -> Option<(usize, bool)> {
    let release = t & mem::MEM_RELEASE != 0;
    let decommit = t & mem::MEM_DECOMMIT != 0;
    let n = match res {
        _ if release == decommit || (release && size != 0) => None,
        Some((start, len)) if release || size == 0 => (start == a).then_some(len),
        Some((start, len)) => a.checked_add(size).is_some_and(|end| end <= start + len).then_some(size),
        None => None,
    };
    if n.is_none() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
    }
    n.map(|n| (n, release))
}

pub unsafe extern "system" fn VirtualFreeEx(proc_: HANDLE, addr: *mut c_void, size: usize, t: u32) -> BOOL {
    if own_process(proc_) {
        let a = addr as usize;
        // (a graphics reservation's decommits too: one may be a paired CPU view, `source_going`)
        if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(a) || u.contains(a) || u.is_view(a)) || in_graphics_reservation(a) {
            return VirtualFree(addr, size, t);
        }
    }
    mem::VirtualFreeEx(proc_, addr, size, t)
}

pub unsafe extern "system" fn VirtualAllocEx(proc_: HANDLE, addr: *const c_void, size: usize, t: u32, p: u32) -> *mut c_void {
    if own_process(proc_) {
        return VirtualAlloc(addr, size, t, p);
    }
    mem::VirtualAllocEx(proc_, addr, size, sanitize_alloc_type(t), sanitize_protect(p))
}

pub unsafe extern "system" fn VirtualProtect(addr: *const c_void, size: usize, p: u32, old: *mut u32) -> BOOL {
    let protection = sanitize_protect(p);
    let writable = read_write(protection);
    if !writable {
        note_protection(addr, size, protection);
    }
    let ok = mem::VirtualProtect(addr, size, protection, old);
    if ok != 0 && writable {
        note_protection(addr, size, protection);
    }
    ok
}

pub unsafe extern "system" fn VirtualProtectEx(proc_: HANDLE, addr: *const c_void, size: usize, p: u32, old: *mut u32) -> BOOL {
    let protection = sanitize_protect(p);
    let own = own_process(proc_);
    let writable = read_write(protection);
    if own && !writable {
        note_protection(addr, size, protection);
    }
    let ok = mem::VirtualProtectEx(proc_, addr, size, protection, old);
    if ok != 0 && own && writable {
        note_protection(addr, size, protection);
    }
    ok
}

/// XALLOC_ATTRIBUTES bit fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XAllocAttrs {
    pub object_type: u32,
    pub page_size: u32,
    pub allocator_id: u32,
    pub alignment: u32,
    pub memory_type: u32,
}

impl XAllocAttrs {
    pub fn decode(a: u64) -> Self {
        XAllocAttrs {
            object_type: (a & 0x3FFF) as u32,
            page_size: ((a >> 14) & 0x3) as u32,
            allocator_id: ((a >> 16) & 0xFF) as u32,
            alignment: ((a >> 24) & 0x1F) as u32,
            memory_type: ((a >> 29) & 0xF) as u32,
        }
    }
}

unsafe fn aligned_heap_alloc(size: usize, align: usize) -> *mut c_void {
    let Some(total) = size.checked_add(align).and_then(|n| n.checked_add(16)) else {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return std::ptr::null_mut();
    };
    let raw = mem::HeapAlloc(mem::GetProcessHeap(), mem::HEAP_ZERO_MEMORY, total) as usize;
    if raw == 0 {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return std::ptr::null_mut();
    }
    let aligned = (raw + 16 + align - 1) & !(align - 1);
    *((aligned - 8) as *mut usize) = raw;
    aligned as *mut c_void
}

unsafe fn aligned_heap_free(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    let raw = *((p as usize - 8) as *const usize);
    mem::HeapFree(mem::GetProcessHeap(), 0, raw as *const c_void);
}

/// Rounded byte size and alignment for an XMem graphics allocation honoring `page_size`
/// (`2` = 2 MiB pages, otherwise 64 KiB sparse tile grain) and `alignment`.
fn xmem_graphics_layout(size: usize, a: XAllocAttrs) -> Option<(usize, usize)> {
    let page_grain = match a.page_size {
        2 => 2 << 20,
        _ => crate::gpu::uma::GRAIN,
    };
    let align = (1usize << a.alignment.min(usize::BITS - 2)).max(page_grain);
    let rounded = size.max(1).checked_next_multiple_of(page_grain)?;
    Some((rounded, align))
}

pub unsafe extern "system" fn XMemAllocDefault(size: usize, attrs: u64) -> *mut c_void {
    let a = XAllocAttrs::decode(attrs);
    if a.memory_type == 0 {
        let q = aligned_heap_alloc(size, 1usize << a.alignment.clamp(4, usize::BITS - 2));
        trace_mem("XMemAlloc", 0, size, attrs, 0, q as usize);
        return q;
    }
    let Some((rounded, align)) = xmem_graphics_layout(size, a) else {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return std::ptr::null_mut();
    };
    // Graphics memory types live in the sparse VRAM pool (fallback to coherent UMA, then Windows).
    if let Some((q, _)) = crate::hle::d3d12x::sparse::alloc(rounded, align) {
        trace_mem("XMemAlloc (graphics, sparse VRAM)", 0, rounded, attrs, 0, q);
        return q as *mut c_void;
    }
    if let Some(q) = crate::gpu::uma::get().and_then(|u| u.alloc(rounded, align)) {
        trace_mem("XMemAlloc (graphics, UMA)", 0, rounded, attrs, 0, q as usize);
        return q as *mut c_void;
    }
    mem::VirtualAlloc(std::ptr::null(), rounded, mem::MEM_COMMIT | mem::MEM_RESERVE, mem::PAGE_READWRITE)
}

pub unsafe extern "system" fn XMemFreeDefault(p: *mut c_void, attrs: u64) {
    if p.is_null() {
        return;
    }
    if XAllocAttrs::decode(attrs).memory_type == 0 {
        aligned_heap_free(p);
    } else if !crate::hle::d3d12x::sparse::free(p as usize) && !crate::gpu::uma::get().is_some_and(|u| u.free(p as usize)) {
        mem::VirtualFree(p, 0, mem::MEM_RELEASE);
    }
}

type XMemAllocFn = unsafe extern "system" fn(usize, u64) -> *mut c_void;
type XMemFreeFn = unsafe extern "system" fn(*mut c_void, u64);
static XMEM_ALLOC_HOOK: AtomicUsize = AtomicUsize::new(0);
static XMEM_FREE_HOOK: AtomicUsize = AtomicUsize::new(0);

pub unsafe extern "system" fn XMemAlloc(size: usize, attrs: u64) -> *mut c_void {
    match XMEM_ALLOC_HOOK.load(Ordering::Acquire) {
        0 => XMemAllocDefault(size, attrs),
        f => std::mem::transmute::<usize, XMemAllocFn>(f)(size, attrs),
    }
}

pub unsafe extern "system" fn XMemFree(p: *mut c_void, attrs: u64) {
    match XMEM_FREE_HOOK.load(Ordering::Acquire) {
        0 => XMemFreeDefault(p, attrs),
        f => std::mem::transmute::<usize, XMemFreeFn>(f)(p, attrs),
    }
}

/// The exe's exported data `__XMemAllocationHooks__` {alloc, free, …}: the allocator functions
/// of a title that replaces XMem's, which the console's runtime libraries call for the memory
/// they need (`runtime_alloc`). Titles see there what the runtime allocates – and may supply the
/// memory themselves.
static TITLE_HOOKS: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];

pub unsafe fn adopt_title_hooks(main: &crate::modules::MappedModule) {
    let Some(crate::pe::ExportTarget::Rva(rva)) = main.exports.by_name.get("__XMemAllocationHooks__") else { return };
    let p = (main.base + *rva as usize) as *const usize;
    let (alloc, free) = (p.read_unaligned(), p.add(1).read_unaligned());
    let inside = |f: usize| f > main.base && f < main.base + main.size;
    if inside(alloc) && inside(free) {
        TITLE_HOOKS[1].store(free, Ordering::Release);
        TITLE_HOOKS[0].store(alloc, Ordering::Release);
        crate::xlog_info!("XMem: the title has allocation hooks ({}, {})", crate::modules::describe_address(alloc), crate::modules::describe_address(free));
    }
}

/// XALLOC_ATTRIBUTES of a descriptor heap's memory: object type 4, 64 KiB pages and alignment,
/// allocator 0x80 (the D3D runtime), graphics memory. The retail game's heap class (its alloc hook for
/// allocator 0x80 compares object type and size) gives the memory of its heaps this way and
/// asserts that they start there.
pub const D3D_DESCRIPTOR_HEAP: u64 = 4 | 1 << 14 | 0x80 << 16 | 16 << 24 | 1 << 29;

/// XALLOC_ATTRIBUTES of a display target's memory: 2 MiB pages (the retail game's hook for its back buffers
/// takes every request with them), allocator 0x80, graphics memory.
pub const D3D_DISPLAY_TARGET: u64 = 2 << 14 | 0x80 << 16 | 16 << 24 | 1 << 29;

/// What the title's allocation hook gives a console runtime library (None: no hook, or null).
pub unsafe fn runtime_alloc(size: usize, attrs: u64) -> Option<usize> {
    let f = TITLE_HOOKS[0].load(Ordering::Acquire);
    if f == 0 {
        return None;
    }
    let p = std::mem::transmute::<usize, XMemAllocFn>(f)(size, attrs) as usize;
    (p != 0).then_some(p)
}

pub unsafe fn runtime_free(p: usize, attrs: u64) {
    let f = TITLE_HOOKS[1].load(Ordering::Acquire);
    if f != 0 {
        std::mem::transmute::<usize, XMemFreeFn>(f)(p as *mut c_void, attrs);
    }
}

/// `void* XMemVirtualAlloc(void* address, SIZE_T size, DWORD type, XALLOC_ATTRIBUTES attributes,
/// DWORD protect)`: VirtualAlloc with an allocator's attributes – a graphics memory type is
/// GPU-visible memory (as MEM_GRAPHICS: sparse VRAM pool).
pub unsafe extern "system" fn XMemVirtualAlloc(addr: *const c_void, size: usize, t: u32, attrs: u64, p: u32) -> *mut c_void {
    let a = XAllocAttrs::decode(attrs);
    let mut t = if a.memory_type != 0 { t | MEM_GRAPHICS } else { t };
    if a.memory_type != 0 && a.page_size == 2 {
        t |= 0x8000_0000;
    }
    VirtualAlloc(addr, size, t, p)
}

/// Fast-path `VirtualQuery` / `VirtualQueryEx` / `XMemVirtualQuery` for a committed coherent
/// UMA window: `OpenExistingHeapFromAddress` makes a kernel `VirtualQuery` on a 4 GiB coherent
/// window walk the whole region (~75–80 ms, holding `MmLockAddressSpace`), stalling
/// `GetWriteWatch` on other threads. A committed coherent window is `PAGE_READWRITE` as a whole.
unsafe fn query_committed_window(address: *const c_void, mbi: *mut mem::MEMORY_BASIC_INFORMATION, len: usize) -> Option<usize> {
    let need = std::mem::size_of::<mem::MEMORY_BASIC_INFORMATION>();
    if mbi.is_null() || len < need {
        return None;
    }
    let addr = address as usize;
    let u = crate::gpu::uma::get()?;
    if !u.in_committed_window(addr, 1) {
        return None;
    }
    let k = u.window_of(addr)?;
    let (win_base, win_len) = u.window(k);
    let page_base = addr & !0xFFF;
    mbi.write_unaligned(mem::MEMORY_BASIC_INFORMATION {
        BaseAddress: page_base as *mut c_void,
        AllocationBase: win_base as *mut c_void,
        AllocationProtect: mem::PAGE_READWRITE,
        PartitionId: 0,
        RegionSize: (win_base + win_len).saturating_sub(page_base),
        State: mem::MEM_COMMIT,
        Protect: mem::PAGE_READWRITE,
        Type: mem::MEM_PRIVATE,
    });
    Some(need)
}

fn is_current_process(process: HANDLE) -> bool {
    process == (-1isize as HANDLE) || process == unsafe { thr::GetCurrentProcess() }
}

pub unsafe extern "system" fn VirtualQuery(address: *const c_void, mbi: *mut mem::MEMORY_BASIC_INFORMATION, len: usize) -> usize {
    if let Some(n) = query_committed_window(address, mbi, len) {
        return n;
    }
    mem::VirtualQuery(address, mbi, len)
}

pub unsafe extern "system" fn VirtualQueryEx(process: HANDLE, address: *const c_void, mbi: *mut mem::MEMORY_BASIC_INFORMATION, len: usize) -> usize {
    if is_current_process(process) {
        if let Some(n) = query_committed_window(address, mbi, len) {
            return n;
        }
    }
    mem::VirtualQueryEx(process, address, mbi, len)
}

/// The field at +48 of `XMEM_BASIC_INFORMATION` is 16 bytes; callers inspect more than its low
/// u32. Zero all of it, then write the flags we know: bit 0 is CPU-cached/readable memory, bit 1
/// is GPU-accessible memory. Never leave stack garbage in the rest of the field.
unsafe fn write_xmem_flags(info: *mut c_void, flags: u32) {
    let p = (info as *mut u8).add(48);
    std::ptr::write_bytes(p, 0, 16);
    p.cast::<u32>().write_unaligned(flags);
}

/// Whether the address is memory the title can use from the GPU: a coherent UMA window, or a
/// real sparse-VRAM reservation (not merely a PC address that falls in a sparse window by number).
fn xmem_gpu_visible(address: *const c_void) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    let p = address as usize;
    u.is_committed(p) || u.is_title_address(p)
}

/// `BOOL XMemVirtualQuery(HANDLE process, const void* address, XMEM_BASIC_INFORMATION* info)`
/// (signature from XboxBvhBuilder.dll's call sites): MEMORY_BASIC_INFORMATION plus a 16-byte
/// flags field at +48. Bit 0 marks committed readable memory (CPU cached), bit 1 marks
/// GPU-accessible UMA / sparse-VRAM memory. The rest is written as zero, not left to the caller.
pub unsafe extern "system" fn XMemVirtualQuery(process: HANDLE, address: *const c_void, info: *mut c_void) -> BOOL {
    if info.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let mbi = info as *mut mem::MEMORY_BASIC_INFORMATION;
    if is_current_process(process) && query_committed_window(address, mbi, std::mem::size_of::<mem::MEMORY_BASIC_INFORMATION>()).is_some() {
        write_xmem_flags(info, 1 | 2);
        return 1;
    }
    if mem::VirtualQueryEx(process, address, mbi, std::mem::size_of::<mem::MEMORY_BASIC_INFORMATION>()) == 0 {
        return 0;
    }
    const READABLE: u32 = mem::PAGE_READONLY
        | mem::PAGE_READWRITE
        | mem::PAGE_WRITECOPY
        | mem::PAGE_EXECUTE_READ
        | mem::PAGE_EXECUTE_READWRITE
        | mem::PAGE_EXECUTE_WRITECOPY;
    let protect = (*mbi).Protect;
    let cpu = (*mbi).State == mem::MEM_COMMIT && (protect & mem::PAGE_GUARD) == 0 && (protect & READABLE) != 0;
    let gpu = xmem_gpu_visible(address);
    write_xmem_flags(info, (cpu as u32) | ((gpu as u32) << 1));
    1
}

/// `BOOL XMemGetWorkingSetStatistics(UINT32 allocator, void* stats)` (as FSR's upscaler calls
/// it: allocator 0x20000001, 128-byte buffer, reads +8 only on success). No statistics:
/// FALSE, which the caller reports as 0 bytes.
/// `BOOL XMemGetWorkingSetStatistics(DWORD flags, XMEM_WORKING_SET_STATISTICS* stats)`: +0 the
/// title's memory limit, +8 what it uses – engines size their pools and texture streaming from
/// these (Fortnite: flags 0x10000001). Only those two are written: the structure's size
/// differs between GDK versions (Fortnite's exe passes 0x78 bytes, its EOSSDK – older – has
/// at most 0x50 bytes before its stack cookie, which zeroing 0x78 bytes destroyed: the DLL's
/// static initializer failed its cookie check, 0xC0000409).
pub unsafe extern "system" fn XMemGetWorkingSetStatistics(_flags: u32, stats: *mut u8) -> BOOL {
    if stats.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let limit = crate::state::config().title_memory_budget_mb() * 1024 * 1024;
    (stats as *mut u64).write_unaligned(limit);
    (stats.add(8) as *mut u64).write_unaligned(process_private_bytes().min(limit));
    1
}

/// `BOOL XMemTransferMemory(direction, size)`: moving budget between the title and the tools
/// pools – there is one budget here
pub unsafe extern "system" fn XMemTransferMemory(_direction: u32, _size: usize) -> BOOL {
    1
}

/// The title's debug output (console builds print a lot there) goes to XCL's log as well.
pub unsafe extern "system" fn OutputDebugStringA(s: *const u8) {
    if let Some(m) = from_ansi_ptr(s) {
        crate::xlog_info!("[debug] {}", m.trim_end());
    }
}

pub unsafe extern "system" fn OutputDebugStringW(s: *const u16) {
    if let Some(m) = from_wide_ptr(s) {
        crate::xlog_info!("[debug] {}", m.trim_end());
    }
}

pub unsafe extern "system" fn XMemSetAllocationHooks(alloc: usize, free: usize) {
    if alloc == 0 {
        XMEM_ALLOC_HOOK.store(0, Ordering::Release);
        XMEM_FREE_HOOK.store(0, Ordering::Release);
    } else {
        XMEM_ALLOC_HOOK.store(alloc, Ordering::Release);
        XMEM_FREE_HOOK.store(free, Ordering::Release);
    }
}

#[repr(C)]
pub struct TitleMemoryStatusT {
    pub dw_length: u32,
    pub dw_reserved: u32,
    pub total: u64,
    pub avail: u64,
    pub legacy_used: u64,
    pub legacy_peak: u64,
    pub legacy_avail: u64,
    pub title_used: u64,
    pub title_avail: u64,
    pub legacy_page_table_used: u64,
    pub title_page_table_used: u64,
}

fn process_private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    unsafe {
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        if K32GetProcessMemoryInfo(thr::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb) != 0 {
            // Wine fills the fields before PrivateUsage only; PagefileUsage is the same number
            // on Windows (the commit charge)
            if c.PrivateUsage != 0 { c.PrivateUsage as u64 } else { c.PagefileUsage as u64 }
        } else {
            0
        }
    }
}

pub unsafe extern "system" fn TitleMemoryStatus(buf: *mut TitleMemoryStatusT) -> BOOL {
    if buf.is_null() || (*buf).dw_length < std::mem::size_of::<TitleMemoryStatusT>() as u32 {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let total = crate::state::config().title_memory_budget_mb() * 1024 * 1024;
    let used = process_private_bytes().min(total);
    let b = &mut *buf;
    b.total = total;
    b.avail = total - used;
    b.legacy_used = 0;
    b.legacy_peak = 0;
    b.legacy_avail = 0;
    b.title_used = used;
    b.title_avail = total - used;
    b.legacy_page_table_used = 0;
    b.title_page_table_used = 0;
    1
}

pub unsafe extern "system" fn JobTitleMemoryStatus(buf: *mut TitleMemoryStatusT) -> BOOL {
    TitleMemoryStatus(buf)
}

pub unsafe extern "system" fn ToolingMemoryStatus(buf: *mut u32) -> BOOL {
    if buf.is_null() || *buf < 40 {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    std::ptr::write_bytes((buf as *mut u8).add(8), 0, 32);
    1
}

/// `BOOL XMemIsToolsMemAvailable()`: whether tooling (debug) memory is available on the console.
/// Returns FALSE (0) in retail title execution.
pub unsafe extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

/// Title physical pages (64 KiB) emulated in the unified memory region: an allocation is one
/// contiguous UMA block – its pages' home; a page's number counts 4 KiB frames (`page_number`;
/// 0 means "none": unmap). Mapping consecutive pages at their home (or at no chosen address)
/// returns that UMA address (GPU-visible, no copy). Mapping pages at another address – address
/// space the title reserved, in any order – commits it and moves their data there, and a home
/// in a sparse window gives its memory back meanwhile (`move_home_away`); unmapping them (or
/// mapping them elsewhere) moves it back home: a page is in one place at a time, keeps its
/// contents and is committed once. A home in a sparse window is only reserved until the page is
/// used there (mapped at home, or read / written through `read_physical_pages` /
/// `write_physical_pages`): until then the page reads as zeros. (A retail game allocates ~5.7 GB
/// of pages at once and maps them into its own reservation: committed at home first, with
/// video-memory tiles, they took the system's commit.)
const PHYS_PAGE: usize = 0x1_0000;

/// Diagnostic: the last events on the title's 64 KiB pages, for the report of an unhandled
/// exception (`describe`): what happened lately at the pages the registers point into.
pub mod pagelog {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    const KEEP: usize = 400_000;
    const PAGE: usize = 0x1_0000;

    #[derive(Clone, Copy)]
    struct Event {
        at_ms: u64,
        what: &'static str,
        /// the address the event is at, a second address (a move's source) and a number (page / bytes)
        a: usize,
        b: usize,
        n: usize,
    }

    static LOG: Mutex<VecDeque<Event>> = Mutex::new(VecDeque::new());
    static COUNTS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

    fn now_ms() -> u64 {
        static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        T0.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
    }

    /// `what` happened at address `a` (`b`: another address it involves, `n`: a page number or bytes).
    pub fn note(what: &'static str, a: usize, b: usize, n: usize) {
        let mut g = LOG.lock().unwrap_or_else(|e| e.into_inner());
        if g.len() >= KEEP {
            g.pop_front();
        }
        g.push_back(Event { at_ms: now_ms(), what, a, b, n });
        drop(g);
        let mut c = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
        match c.iter_mut().find(|e| e.0 == what) {
            Some(e) => e.1 += 1,
            None => c.push((what, 1)),
        }
    }

    /// The events so far by kind.
    pub fn counts() -> String {
        // (called from the exception filter, perhaps with the lock held by the thread that faulted)
        let Ok(c) = COUNTS.try_lock() else { return "(busy)".into() };
        c.iter().map(|(w, n)| format!("{n} {w}")).collect::<Vec<_>>().join(", ")
    }

    /// The last events (up to 6) at the 64 KiB page of `addr`, newest first; empty: none kept.
    pub fn describe(addr: usize) -> String {
        let page = addr & !(PAGE - 1);
        let Ok(g) = LOG.try_lock() else { return String::new() };
        let now = now_ms();
        let hit = |e: &&Event| (e.a..e.a + if e.what.starts_with("DirectStorage") { e.n.max(1) } else { PAGE }).contains(&addr) || e.a & !(PAGE - 1) == page || (e.b != 0 && e.b & !(PAGE - 1) == page);
        g.iter().rev().filter(hit).take(6).map(|e| format!("{} ms ago: {} at {:#x}{}{}", now.saturating_sub(e.at_ms), e.what, e.a, if e.b != 0 { format!(" (other address {:#x})", e.b) } else { String::new() }, if e.n != 0 { format!(" [{:#x}]", e.n) } else { String::new() })).collect::<Vec<_>>().join("; ")
    }
}

/// Size of one console physical page (64 KiB).
pub const PHYS_PAGE_SIZE: usize = PHYS_PAGE;

/// first page index → (pages in the block, pages still allocated, the block's home)
static PHYS_BLOCKS: std::sync::Mutex<std::collections::BTreeMap<usize, (usize, usize, usize)>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// the next page index handed out (page numbers are not addresses: a block's home is in a
/// sparse window, the coherent one, or – without room there – plain reserved memory)
static NEXT_PAGE_INDEX: AtomicUsize = AtomicUsize::new(0);

/// Pages mapped away from home: address → page, and page → address.
#[derive(Default)]
struct Away {
    at: std::collections::BTreeMap<usize, usize>,
    page: std::collections::HashMap<usize, usize>,
    /// homes in a sparse window whose pages went away in this call, given back at its end in
    /// runs (`give_back_homes`): one page at a time, a retail game's start took 20 s
    gone: Vec<usize>,
    /// homes committed ahead of the pages the call in progress unmaps (`commit_homes_ahead`)
    ready: std::collections::HashSet<usize>,
    /// address → the page the title mapped there last. `at` cannot serve: it holds an address
    /// only while the page's bytes are kept there, and when the title maps the same page at a
    /// second address XCL moves the bytes and `move_away` drops the first one – on the console
    /// the page stays mapped there too, and that first address is the pool address the queue's
    /// page-mapping method names.
    title: std::collections::HashMap<usize, usize>,
}
static AWAY: std::sync::Mutex<Option<Away>> = std::sync::Mutex::new(None);

/// The title released `[base, base + len)`: the pages it mapped there are mapped there no more.
fn forget_title_mappings(base: usize, len: usize) {
    if let Some(a) = AWAY.lock().unwrap().as_mut() {
        let end = base.saturating_add(len);
        // (a range of fewer pages than there are entries: each of its pages, not every entry)
        let (first, pages) = (base & !(PHYS_PAGE - 1), len.div_ceil(PHYS_PAGE) + 1);
        if pages < a.title.len() {
            for at in (first..end).step_by(PHYS_PAGE).filter(|&at| at >= base) {
                a.title.remove(&at);
            }
        } else {
            a.title.retain(|&at, _| at < base || at >= end);
        }
    }
}

/// Whether a physical page is mapped at the page holding `va` away from its home – committed
/// or not yet (`lazy`: a page never used is committed at its first access).
pub fn physical_page_at(va: usize) -> bool {
    AWAY.lock().unwrap().as_ref().is_some_and(|a| a.at.contains_key(&(va & !(PHYS_PAGE - 1))))
}

/// Counts the calls that map or unmap physical pages in a GPU view (`mirrors`): what
/// `physical_page_at` says there may have changed.
static GPU_VIEW_MAP_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Generation of the physical-page mappings inside GPU views.
pub fn gpu_view_map_gen() -> u64 {
    GPU_VIEW_MAP_GEN.load(Ordering::Acquire)
}

/// Pages are mapped or unmapped at `va`: a GPU view's mappings changed if it lies in one.
fn note_mapping_at(va: usize) {
    note_mapping_range(va, PHYS_PAGE);
}

/// The address ranges of the last mapping changes inside GPU views with the generation each made
/// (`gpu_view_map_gen`), oldest first, and the newest generation dropped from the front (`0`:
/// none): a reader that has seen generation n looks at the ranges with a later one.
static GPU_VIEW_MAP_LOG: std::sync::Mutex<(u64, std::collections::VecDeque<(u64, usize, usize)>)> = std::sync::Mutex::new((0, std::collections::VecDeque::new()));
const MAP_LOG: usize = 4096;

/// Pages are mapped or unmapped in `[lo, lo + len)`.
fn note_mapping_range(lo: usize, len: usize) {
    if len == 0 || (mirrors::source(lo).is_none() && mirrors::source(lo + len - 1).is_none()) {
        return;
    }
    note_gpu_view_change(lo, len);
}

/// What `[lo, lo + len)` of a GPU view shows changed (its mappings, or the view was paired with
/// its CPU view or went): a new generation of `gpu_view_map_gen` with the range in the log.
fn note_gpu_view_change(lo: usize, len: usize) {
    let mut g = GPU_VIEW_MAP_LOG.lock().unwrap();
    let gen = GPU_VIEW_MAP_GEN.load(Ordering::Relaxed) + 1;
    g.1.push_back((gen, lo, lo.saturating_add(len)));
    if g.1.len() > MAP_LOG {
        if let Some((dropped, _, _)) = g.1.pop_front() {
            g.0 = dropped;
        }
    }
    GPU_VIEW_MAP_GEN.store(gen, Ordering::Release);
}

/// The ranges whose mappings in GPU views changed after generation `since`, or None when the log
/// no longer has them all.
pub fn gpu_view_map_changes(since: u64) -> Option<Vec<(usize, usize)>> {
    let g = GPU_VIEW_MAP_LOG.lock().unwrap();
    if since < g.0 {
        return None;
    }
    Some(g.1.iter().filter(|e| e.0 > since).map(|e| (e.1, e.2)).collect())
}

/// Page numbers count 4 KiB frames, as on the console: a 64 KiB page's is a multiple of 16 (a retail game
/// packs them shifted right by 4). Index 0 of unified memory is page 16 - 0 means "no page".
const FRAMES: usize = PHYS_PAGE >> 12;

fn page_number(index: usize) -> usize {
    (index + 1) * FRAMES
}

fn page_index(page: usize) -> Option<usize> {
    page.is_multiple_of(FRAMES).then(|| (page / FRAMES).checked_sub(1)).flatten()
}

/// The home address of page `page` if it belongs to a live allocation.
fn page_home(page: usize) -> Option<usize> {
    let index = page_index(page)?;
    let blocks = PHYS_BLOCKS.lock().ok()?;
    let (&first, &(count, live, home)) = blocks.range(..=index).next_back()?;
    (live > 0 && index < first.checked_add(count)?).then(|| home.checked_add((index - first).checked_mul(PHYS_PAGE)?)).flatten()
}

/// Whether every page of `pages` (at least one) is a physical page allocated now.
pub fn physical_pages_live(pages: &[usize]) -> bool {
    !pages.is_empty() && pages.iter().all(|&p| page_home(p).is_some())
}

/// Whether a home is only reserved until it is used (a sparse window's, or plain memory's –
/// the coherent window is committed as a whole).
fn lazy_home(home: usize) -> bool {
    crate::gpu::uma::get().is_none_or(|u| u.is_sparse(home) || !u.contains(home))
}

/// Resolve the live virtual addresses of the physical pages covering `offset..offset+len`.
/// Caller must hold `AWAY`.
unsafe fn resolve_physical_pages(away: Option<&Away>, pages: &[usize], offset: usize, len: usize) -> Option<Vec<usize>> {
    if offset >= PHYS_PAGE {
        return None;
    }
    let span = offset.checked_add(len)?;
    let need = span.div_ceil(PHYS_PAGE);
    if pages.len() < need {
        return None;
    }
    let mut addrs = Vec::with_capacity(need);
    for &p in &pages[..need] {
        let home = page_home(p)?;
        let addr = match away.and_then(|a| a.page.get(&p).copied()) {
            Some(at) => at,
            None => home_again(home)?,
        };
        addrs.push(addr);
    }
    Some(addrs)
}

/// Copy `dst.len()` bytes starting at `offset` within `pages` (each a 64 KiB physical page
/// number from `AllocateTitlePhysicalPages`). Returns `false` if `offset >= PHYS_PAGE_SIZE`,
/// `pages` is too short, or any page is not a live allocation.
pub unsafe fn read_physical_pages(pages: &[usize], offset: usize, dst: &mut [u8]) -> bool {
    if dst.is_empty() {
        return offset < PHYS_PAGE;
    }
    let g = AWAY.lock().unwrap();
    let Some(addrs) = resolve_physical_pages(g.as_ref(), pages, offset, dst.len()) else { return false };
    let mut done = 0;
    let mut page_off = offset;
    for addr in addrs {
        let chunk = (PHYS_PAGE - page_off).min(dst.len() - done);
        std::ptr::copy_nonoverlapping((addr + page_off) as *const u8, dst.as_mut_ptr().add(done), chunk);
        done += chunk;
        page_off = 0;
    }
    true
}

/// Copy `src` into `pages` starting at `offset` within the first page. Validates every page
/// before writing so the destination gets all bytes or nothing.
pub unsafe fn write_physical_pages(pages: &[usize], offset: usize, src: &[u8]) -> bool {
    if src.is_empty() {
        return offset < PHYS_PAGE;
    }
    let g = AWAY.lock().unwrap();
    let Some(addrs) = resolve_physical_pages(g.as_ref(), pages, offset, src.len()) else { return false };
    let mut done = 0;
    let mut page_off = offset;
    for addr in addrs {
        let chunk = (PHYS_PAGE - page_off).min(src.len() - done);
        pagelog::note("DirectStorage write by page number", addr + page_off, 0, chunk);
        std::ptr::copy_nonoverlapping(src.as_ptr().add(done), (addr + page_off) as *mut u8, chunk);
        crate::hle::d3d12x::title_textures::note_known_write(addr + page_off, chunk);
        done += chunk;
        page_off = 0;
    }
    true
}

/// Homes `move_home_away` collects before it gives them back (`give_back_homes`).
const GIVE_BACK_PAGES: usize = 4096;

/// Pages `map_pages` mapped in a sparse window, told to `sparse::committed` once per run of
/// neighbouring pages instead of once a page. Each call is an `UpdateTileMappings` and a fence
/// signal: about 0.5 ms a page on NVIDIA, and a retail game in Series S mode maps 3.4 GB in one
/// `MapTitlePhysicalPages` at boot – 24 s on its main thread, and its stall check ended it.
struct SparseRun {
    start: usize,
    end: usize,
    commit: unsafe fn(usize, usize),
}

impl SparseRun {
    fn new(commit: unsafe fn(usize, usize)) -> Self {
        SparseRun { start: 0, end: 0, commit }
    }

    /// The page at `at` was mapped in a sparse window.
    unsafe fn add(&mut self, at: usize) {
        if self.start == self.end || at != self.end {
            self.flush();
            self.start = at;
        }
        self.end = at + PHYS_PAGE;
    }

    /// Whether the page at `a` waits in the run.
    fn holds(&self, a: usize) -> bool {
        (self.start..self.end).contains(&a)
    }

    /// The run's pages are committed now (before anything decommits one of them).
    unsafe fn flush(&mut self) {
        if self.end > self.start {
            (self.commit)(self.start, self.end - self.start);
        }
        (self.start, self.end) = (0, 0);
    }
}

/// A page's data went from `home` to another address: a home in a sparse window is
/// decommitted at the end of the call or every 256 MB (`give_back_homes`; its GPU tiles go too: the GPU does
/// not see the page while it is away). True if homes were given back now.
unsafe fn move_home_away(away: &mut Away, home: usize) -> bool {
    if lazy_home(home) {
        away.gone.push(home);
        // (256 MB at a time: a call that maps gigabytes would otherwise hold them twice until
        // it ends)
        if away.gone.len() >= GIVE_BACK_PAGES {
            give_back_homes(away);
            return true;
        }
    }
    false
}

/// The homes `move_home_away` collected, decommitted in runs of neighbouring pages.
unsafe fn give_back_homes(away: &mut Away) {
    let mut homes = std::mem::take(&mut away.gone);
    homes.sort_unstable();
    homes.dedup();
    let mut i = 0;
    while i < homes.len() {
        let start = homes[i];
        let mut end = start + PHYS_PAGE;
        i += 1;
        while i < homes.len() && homes[i] == end {
            end += PHYS_PAGE;
            i += 1;
        }
        if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(start)) {
            crate::hle::d3d12x::sparse::decommitted(start, end - start);
        }
        // (a tile may show a page kept at its home: `source_part`)
        source_going(start, end - start, false, || mem::VirtualFree(start as *mut c_void, end - start, mem::MEM_DECOMMIT));
    }
}

/// The page's home, committed again if `move_home_away` gave it back (None: it could not be).
unsafe fn home_again(home: usize) -> Option<usize> {
    if lazy_home(home) && !committed_page(home) {
        if mem::VirtualAlloc(home as *const c_void, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null() {
            crate::xlog_warn!("physical page: its home {home:#x} could not be committed again");
            return None;
        }
        if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(home)) {
            crate::hle::d3d12x::sparse::committed(home, PHYS_PAGE);
        }
    }
    Some(home)
}

/// The pages of a scatter whose homes are committed ahead together: they and the memory the pages
/// leave are both committed while these are copied (64 MiB).
const HOMES_AHEAD: usize = 1024;

/// The homes of the pages mapped at `ats`, which the caller unmaps next, committed in runs of
/// neighbouring pages (`Away::ready`) – `home_again` does it page by page, and in a sparse window
/// each is a tile mapping of its own: a retail game unmapping 4136 and then 6720 pages while it
/// loads, the homes of each call one run, spent 2.7 s in those two calls. A home committed
/// already (another page may be mapped at that address, given up in this call), or that cannot
/// be, is left to `home_again`.
unsafe fn commit_homes_ahead(away: &mut Away, ats: impl Iterator<Item = usize>) {
    let mut homes: Vec<usize> = ats
        .filter_map(|at| away.at.get(&at).filter(|_| !lazy::is_marked(at)).and_then(|&p| page_home(p)))
        .filter(|&h| lazy_home(h) && !away.gone.contains(&h))
        .collect();
    homes.sort_unstable();
    homes.dedup();
    let mut i = 0;
    while i < homes.len() {
        let start = homes[i];
        let mut end = start + PHYS_PAGE;
        i += 1;
        while i < homes.len() && homes[i] == end {
            end += PHYS_PAGE;
            i += 1;
        }
        let mut a = start;
        while a < end {
            let mut m: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            if mem::VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
                break;
            }
            let to = (m.BaseAddress as usize + m.RegionSize).min(end);
            // (whole pages: a state that changes inside one is `home_again`'s)
            let (first, last) = (a.next_multiple_of(PHYS_PAGE), to & !(PHYS_PAGE - 1));
            if m.State == mem::MEM_RESERVE
                && first < last
                && !mem::VirtualAlloc(first as *const c_void, last - first, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null()
            {
                if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(first)) {
                    crate::hle::d3d12x::sparse::committed(first, last - first);
                }
                #[cfg(test)]
                tests::HOME_RUNS.with(|c| c.set(c.get() + 1));
                away.ready.extend((first..last).step_by(PHYS_PAGE));
            }
            a = to;
        }
    }
}

/// `n` pages from `home` committed (a sparse home is only reserved until it is used).
unsafe fn commit_homes(home: usize, n: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !lazy_home(home) {
        return true;
    }
    let len = n * PHYS_PAGE;
    if committed_range(home, len) {
        return true;
    }
    if mem::VirtualAlloc(home as *const c_void, len, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null() {
        crate::xlog_warn!("physical pages: {n} at their home {home:#x} could not be committed");
        return false;
    }
    if u.is_sparse(home) {
        crate::hle::d3d12x::sparse::committed(home, len);
    }
    true
}

/// Whether all of `[a, a + len)` is committed: one VirtualQuery per run of pages in one state
/// (page by page each asked walked the rest of the run).
fn committed_range(a: usize, len: usize) -> bool {
    if len > 0 && crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(a, len)) {
        return true;
    }
    let mut p = a;
    while p < a + len {
        let mut m: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { mem::VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 || m.State != mem::MEM_COMMIT {
            return false;
        }
        p = m.BaseAddress as usize + m.RegionSize;
    }
    true
}

fn committed_page(a: usize) -> bool {
    committed_page_in(a, &mut StateRun::default())
}

/// The run of pages around an address that one VirtualQuery found in one state. `map_pages`
/// asks about the pages of a range one after the other and nothing it does in between (a page
/// left lazy) changes that state: a run answers them without asking again (VirtualQuery walks
/// the run to its end, and under Wine 98,304 fresh pages took 13 s).
#[derive(Default)]
struct StateRun {
    start: usize,
    end: usize,
    committed: bool,
}

/// `committed_page`, remembering in `run` the state of the pages around `a`.
fn committed_page_in(a: usize, run: &mut StateRun) -> bool {
    if (run.start..run.end).contains(&a) {
        return run.committed;
    }
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(a, 1)) {
        return true;
    }
    // A page in the working set is committed: asking for that is one entry, where VirtualQuery
    // walks the page table to the end of the committed range the page is in (gigabytes of
    // homes: 0.25 ms a page, and a scatter of pages from their homes grew with its square).
    let mut ws = windows_sys::Win32::System::ProcessStatus::PSAPI_WORKING_SET_EX_INFORMATION { VirtualAddress: a as *mut c_void, ..Default::default() };
    if unsafe { windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx(thr::GetCurrentProcess(), &mut ws as *mut _ as *mut c_void, std::mem::size_of_val(&ws) as u32) } != 0
        && unsafe { ws.VirtualAttributes.Flags } & 1 != 0
    {
        return true;
    }
    #[cfg(test)]
    VIRTUAL_QUERIES.with(|n| n.set(n.get() + 1));
    let mut m: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { mem::VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 {
        return false;
    }
    let start = m.BaseAddress as usize;
    *run = StateRun { start, end: start + m.RegionSize, committed: m.State == mem::MEM_COMMIT };
    run.committed
}

#[cfg(test)]
thread_local! {
    /// The VirtualQuery calls `committed_page_in` made on this thread.
    static VIRTUAL_QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Copy what is mapped at `at` back home and unmap it (Windows memory is decommitted); the homes
/// in `committed` the caller has committed. True if memory was committed or decommitted (a page
/// never used, or none mapped at `at`, leaves it as it was).
unsafe fn unmap_away(away: &mut Away, at: usize, committed: &std::ops::Range<usize>) -> bool {
    if let Some(page) = away.at.remove(&at) {
        away.page.remove(&page);
        // never used where it was mapped: its home still holds the page (zeros)
        if lazy::unmark(at) {
            pagelog::note("page unmapped (never used there)", at, 0, page);
            page_placed(page, page_home(page).unwrap_or(0), true, false);
            return false;
        }
        pagelog::note("page unmapped (copied home)", at, 0, page);
        // (a home this call has not given back yet is still committed)
        let own = page_home(page);
        let pending = away.gone.len();
        away.gone.retain(|&h| Some(h) != own);
        let home = if away.gone.len() < pending || own.is_some_and(|h| committed.contains(&h) || away.ready.remove(&h)) {
            own
        } else {
            own.and_then(|h| home_again(h))
        };
        if let Some(home) = home {
            std::ptr::copy_nonoverlapping(at as *const u8, home as *mut u8, PHYS_PAGE);
            #[cfg(test)]
            tests::HOME_COPIES.with(|c| c.set(c.get() + 1));
        }
        page_placed(page, home.unwrap_or(0), true, home.is_some());
        // (an address among the homes the caller maps back is one of them: it stays)
        if !committed.contains(&at) {
            release_at(at);
        }
        true
    } else {
        false
    }
}

/// The memory at `at`, which a page mapped away from home left, decommitted (UMA memory that is
/// committed as a whole stays).
unsafe fn release_at(at: usize) {
    if crate::gpu::uma::get().is_some_and(|u| u.is_committed(at)) {
        return;
    }
    if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(at)) {
        crate::hle::d3d12x::sparse::decommitted(at, PHYS_PAGE);
    }
    // (most pages are mapped in the CPU view of a pair: the tiles of its GPU view show them)
    source_going(at, PHYS_PAGE, false, || mem::VirtualFree(at as *mut c_void, PHYS_PAGE, mem::MEM_DECOMMIT));
}

/// A page mapped away from home (at `old`) that the title maps at `at`: what `old` holds copied
/// to `at`, and `old` given up – its home is not used. By way of its home a move was four
/// commits and decommits, each a tile mapping in a sparse window: a retail game moving pages one
/// at a time after its first movie spent 20 s in them and its watchdog ended it. False: the page
/// was never used at `old` (its home holds it, zeros) and is not mapped anywhere now.
unsafe fn move_away(away: &mut Away, old: usize, at: usize, page: usize, in_uma: bool) -> Result<bool, ()> {
    if lazy::unmark(old) {
        away.at.remove(&old);
        away.page.remove(&page);
        pagelog::note("page mapped elsewhere, was never used here", old, at, page);
        return Ok(false);
    }
    pagelog::note("page moved here from its other mapping (that one is given up)", at, old, page);
    // (a destination that cannot be committed leaves the page where it was, mapped as before)
    if !in_uma && mem::VirtualAlloc(at as *const c_void, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null() {
        return Err(());
    }
    away.at.remove(&old);
    away.page.remove(&page);
    // (before its bytes come, as in `map_pages`)
    if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(at)) {
        crate::hle::d3d12x::sparse::committed(at, PHYS_PAGE);
    }
    std::ptr::copy_nonoverlapping(old as *const u8, at as *mut u8, PHYS_PAGE);
    page_placed(page, at, false, true);
    release_at(old);
    away.at.insert(at, page);
    away.page.insert(page, at);
    Ok(true)
}

/// Reserve home storage for `bytes` of title physical pages: sparse UMA first, coherent UMA
/// only when this `Uma` has no sparse windows (so overflow on runtime layouts never eats the
/// coherent window's `KEEP` headroom or creates non-lazy homes), and finally plain
/// `MEM_RESERVE` memory committed on first use (`home_again`).
/// In the sparse windows from window 1 on while they have room (`sparse_alloc_above`): homes are
/// never 32-bit addressed, window 0 is the console's low 4 GiB. A retail game on the Series S
/// profile had 664+ MB of pages (its page numbers) before it made its view heap; with their homes
/// first-fit in window 0, beside the GPU heaps of `XCL_COHERENT_HEAPS=sparse`, the memory it gave
/// the heap was in window 1, which descriptor tables cannot reach (`console_heap::title_memory`),
/// and it stopped on its own breakpoint.
unsafe fn alloc_physical_home(u: &crate::gpu::uma::Uma, bytes: usize) -> Option<usize> {
    if let Some((p, _)) = u.sparse_alloc_above(bytes, PHYS_PAGE) {
        return Some(p);
    }
    if !u.has_sparse() {
        if let Some(p) = u.alloc(bytes, PHYS_PAGE) {
            return Some(p as usize);
        }
    }
    let p = mem::VirtualAlloc(std::ptr::null(), bytes, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!(
            "AllocateTitlePhysicalPages({}): no room in unified memory – their home is plain memory",
            bytes / PHYS_PAGE
        )
    });
    (p != 0).then_some(p)
}

pub unsafe extern "system" fn AllocateTitlePhysicalPages(_p: HANDLE, _t: u32, num: *mut usize, arr: *mut usize) -> BOOL {
    if num.is_null() || arr.is_null() || *num == 0 {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let n = *num;
    let Some(bytes) = n.checked_mul(PHYS_PAGE) else {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    };
    let Some(u) = crate::gpu::uma::get() else {
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    };
    let block = alloc_physical_home(u, bytes);
    let Some(block) = block else {
        crate::xlog_warn!("AllocateTitlePhysicalPages({n}): no room");
        fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    };
    let first = NEXT_PAGE_INDEX.fetch_add(n, Ordering::Relaxed);
    for i in 0..n {
        *arr.add(i) = page_number(first + i);
    }
    PHYS_BLOCKS.lock().unwrap().insert(first, (n, n, block));
    1
}

pub unsafe extern "system" fn FreeTitlePhysicalPages(_p: HANDLE, n: usize, arr: *mut usize) -> BOOL {
    let Some(u) = crate::gpu::uma::get() else { return 0 };
    if arr.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    // freed pages mapped elsewhere: that address keeps whatever it holds, the mapping goes
    if let Some(away) = AWAY.lock().unwrap().as_mut() {
        for i in 0..n {
            if let Some(at) = away.page.remove(&*arr.add(i)) {
                away.at.remove(&at);
                note_mapping_at(at);
                pagelog::note("page freed while mapped (the address keeps its bytes)", at, 0, *arr.add(i));
            }
            // (those at home too: their home goes with the block)
            crate::hle::d3d12x::sparse::remapped(&mirrors::page_freed(*arr.add(i)));
        }
        // (page numbers are never handed out again: their entries go)
        if !away.title.is_empty() {
            let freed: std::collections::HashSet<usize> = (0..n).map(|i| *arr.add(i)).collect();
            away.title.retain(|_, p| !freed.contains(p));
        }
    }
    let mut g = PHYS_BLOCKS.lock().unwrap();
    for i in 0..n {
        let Some(page) = page_index(*arr.add(i)) else { continue };
        // the block holding this page
        let Some((&first, &(count, _, home))) = g.range(..=page).next_back() else { continue };
        if page >= first + count {
            continue;
        }
        let e = g.get_mut(&first).unwrap();
        e.1 = e.1.saturating_sub(1);
        if e.1 == 0 {
            g.remove(&first);
            // (a tile may show a page kept in the block: `source_part`)
            source_going(home, count * PHYS_PAGE, true, || {
                if u.is_sparse(home) {
                    crate::hle::d3d12x::sparse::free(home);
                } else if u.contains(home) {
                    u.free(home);
                } else {
                    mem::VirtualFree(home as *mut c_void, 0, mem::MEM_RELEASE);
                }
            });
        }
    }
    1
}

pub unsafe extern "system" fn MapTitlePhysicalPages(va: *mut c_void, n: usize, _t: u32, _p: u32, arr: *mut usize) -> *mut c_void {
    let _cost = super::api_cost::time(super::api_cost::Call::PhysicalPages);
    let mut g = AWAY.lock().unwrap();
    let away = g.get_or_insert_with(Away::default);
    let r = map_pages(away, va, n, arr);
    give_back_homes(away);
    r
}

/// `MapTitlePhysicalPages` with `AWAY` held; the homes it moves away are given back by the
/// caller (`give_back_homes`).
unsafe fn map_pages(away: &mut Away, va: *mut c_void, n: usize, arr: *mut usize) -> *mut c_void {
    note_mapping_range(va as usize, n.saturating_mul(PHYS_PAGE));
    if crate::gpu::uma::get().is_none() || n == 0 || (va.is_null() && arr.is_null()) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    // pages in a GPU view (a view window) are plain pages there, as before the views had windows
    // (`contains` and `is_sparse` are false there: no sparse commit), which the GPU does not see
    if !va.is_null() && crate::gpu::uma::get().is_some_and(|u| u.is_view(va as usize)) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::xlog_warn!("kernelx: the title maps physical pages in a GPU view ({va:?}, {n} pages): plain pages there, which the GPU does not see"));
    }
    let pages: Vec<usize> = if arr.is_null() { vec![0; n] } else { (0..n).map(|i| *arr.add(i)).collect() };
    if pages.iter().any(|&p| p != 0 && page_home(p).is_none()) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    // consecutive pages at their home (or no address chosen): that UMA memory itself
    let consecutive = pages[0] != 0 && (1..n).all(|i| pages[0].checked_add(i * FRAMES) == Some(pages[i]));
    let home = if consecutive { page_home(pages[0]).filter(|_| page_home(pages[n - 1]).is_some()) } else { None };
    if va.is_null() || home == Some(va as usize) {
        let Some(home) = home else {
            fnd::SetLastError(fnd::ERROR_NOT_SUPPORTED); // scattered pages need an address
            return std::ptr::null_mut();
        };
        if !commit_homes(home, n) {
            fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
            return std::ptr::null_mut();
        }
        // (pages this call moved away and back are home again, not given back)
        away.gone.retain(|&h| h < home || h >= home + n * PHYS_PAGE);
        for (i, &p) in pages.iter().enumerate() {
            away.title.insert(home + i * PHYS_PAGE, p);
        }
        for &p in &pages {
            if let Some(&at) = away.page.get(&p) {
                unmap_away(away, at, &(home..home + n * PHYS_PAGE));
            }
        }
        return home as *mut c_void;
    }
    let base = va as usize;
    if !base.is_multiple_of(PHYS_PAGE) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    let in_uma = |a: usize| crate::gpu::uma::get().is_some_and(|u| u.is_committed(a));
    // the state of the addresses and homes asked about: dropped when `unmap_away` commits or
    // decommits and when `move_home_away` gives homes back. The rest of a step changes only an
    // address, or a home not asked about again (the addresses rise; a page mapped twice is
    // unmapped first) – bar an address that is another page's home (a home in plain memory):
    // that answer was wrong before the runs too.
    let (mut at_run, mut home_run) = (StateRun::default(), StateRun::default());
    let mut sparse_run = SparseRun::new(crate::hle::d3d12x::sparse::committed);
    for (i, &p) in pages.iter().enumerate() {
        let at = base + i * PHYS_PAGE;
        // what was mapped at `at` goes; the new page is written down once it is in place
        away.title.remove(&at);
        if unmap_away(away, at, &(0..0)) {
            (at_run, home_run) = Default::default();
        }
        if p == 0 {
            continue;
        }
        if let Some(&old) = away.page.get(&p) {
            // (the move decommits `old`: committed first if it waits in the run)
            if sparse_run.holds(old) {
                sparse_run.flush();
            }
            // (a move commits `at` and decommits `old`, which may lie in a run already answered:
            // both runs go – #556)
            match move_away(away, old, at, p, in_uma(at)) {
                Ok(true) => {
                    (at_run, home_run) = Default::default();
                    away.title.insert(at, p);
                    continue;
                }
                // a lazy page is only unmarked: memory is as it was, the runs stay
                Ok(false) => {}
                Err(()) => {
                    sparse_run.flush();
                    fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
                    return std::ptr::null_mut();
                }
            }
        }
        let fresh = !in_uma(at) && !committed_page_in(at, &mut at_run);
        let home = page_home(p).unwrap();
        // a page never used (its home not committed) at the title's own reserved address space:
        // committed on first use (`lazy`) – GPU-owned targets placed there are never touched
        let plain = crate::gpu::uma::get().is_none_or(|u| !u.contains(at));
        if fresh && plain && !committed_page_in(home, &mut home_run) {
            lazy::mark(at);
            away.at.insert(at, p);
            away.page.insert(p, at);
            away.title.insert(at, p);
            page_placed(p, at, false, false);
            pagelog::note("page mapped (never used: zeros on first touch)", at, 0, p);
            continue;
        }
        pagelog::note("page mapped (its bytes copied from home, or zeros)", at, 0, p);
        if !in_uma(at) && mem::VirtualAlloc(at as *const c_void, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null() {
            sparse_run.flush();
            fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
            return std::ptr::null_mut();
        }
        let in_sparse = crate::gpu::uma::get().is_some_and(|u| u.is_sparse(at));
        if committed_page(home) {
            // (told before its bytes come: a look for CPU writes between the two would take the
            // written pages of a tile that has no video memory yet, and nothing reported them again)
            if in_sparse {
                crate::hle::d3d12x::sparse::committed(at, PHYS_PAGE);
            }
            std::ptr::copy_nonoverlapping(home as *const u8, at as *mut u8, PHYS_PAGE);
            // (before the home can be given back)
            page_placed(p, at, false, true);
            // (giving homes back decommits them, and one may be an address in the run)
            if sparse_run.holds(home) || away.gone.len() + 1 >= GIVE_BACK_PAGES {
                sparse_run.flush();
            }
            if move_home_away(away, home) {
                (at_run, home_run) = Default::default();
            }
        } else {
            if !fresh {
                std::ptr::write_bytes(at as *mut u8, 0, PHYS_PAGE); // a page never used: zeros
            }
            if in_sparse {
                sparse_run.add(at);
            }
            page_placed(p, at, false, true);
        }
        away.at.insert(at, p);
        away.page.insert(p, at);
        away.title.insert(at, p);
    }
    sparse_run.flush();
    va
}

/// `BOOL XMemAllocatePhysicalPages(XALLOC_ATTRIBUTES, SIZE_T* count, ULONG64* pages)`: the
/// title physical pages above (64 KiB)
pub unsafe extern "system" fn XMemAllocatePhysicalPages(_attrs: u64, num: *mut usize, arr: *mut u64) -> BOOL {
    AllocateTitlePhysicalPages(std::mem::zeroed(), 0, num, arr as *mut usize)
}

/// `void* XMemMapPhysicalPages(void* address, SIZE_T count, ULONG64* pages)`
pub unsafe extern "system" fn XMemMapPhysicalPages(va: *mut c_void, n: usize, arr: *mut u64) -> *mut c_void {
    MapTitlePhysicalPages(va, n, 0, 0, arr as *mut usize)
}

/// `BOOL XMemMapPhysicalPagesScatter(void** addresses, SIZE_T count, ULONG64* pages)`: one 64 KiB
/// page at each address (page 0: the address is unmapped). No page array unmaps every address:
/// a retail game unmaps that way while it streams its world and stops on its own `int3` when the
/// call fails.
pub unsafe extern "system" fn XMemMapPhysicalPagesScatter(addresses: *const *mut c_void, n: usize, pages: *mut u64) -> BOOL {
    let _cost = super::api_cost::time(super::api_cost::Call::PhysicalPages);
    if addresses.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    if crate::gpu::uma::get().is_none() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let mut g = AWAY.lock().unwrap();
    let away = g.get_or_insert_with(Away::default);
    // no page array: page 0 (unmap) for every address
    let ok = (0..n).all(|i| {
        // a part of the call that only unmaps
        if i % HOMES_AHEAD == 0 {
            let part = i..n.min(i + HOMES_AHEAD);
            if part.len() > 1 && part.clone().all(|k| pages.is_null() || *pages.add(k) == 0) {
                commit_homes_ahead(away, part.map(|k| *addresses.add(k) as usize));
            }
        }
        let mut none = 0usize;
        let page = if pages.is_null() { &mut none as *mut usize } else { pages.add(i) as *mut usize };
        !map_pages(away, *addresses.add(i), 1, page).is_null()
    });
    // (a call that failed part way: homes no page went back to)
    away.gone.extend(away.ready.drain());
    give_back_homes(away);
    ok as BOOL
}

/// `BOOL XMemFreePhysicalPages(SIZE_T count, ULONG64* pages)`
pub unsafe extern "system" fn XMemFreePhysicalPages(n: usize, arr: *mut u64) -> BOOL {
    FreeTitlePhysicalPages(std::mem::zeroed(), n, arr as *mut usize)
}

// ---------------------------------------------------------------------------------------
// App state (minappmodelx.dll, appnotify.h)
// ---------------------------------------------------------------------------------------

/// The registration handed out: the callbacks are never called (XCL does not suspend titles
/// or constrain them for the system UI).
static APP_NOTIFICATION: u8 = 0;

/// `ULONG RegisterAppStateChangeNotification(routine, context, PAPPSTATE_REGISTRATION* out)`
pub unsafe extern "system" fn RegisterAppStateChangeNotification(_routine: usize, _ctx: *mut c_void, out: *mut *const u8) -> u32 {
    if out.is_null() {
        return fnd::ERROR_INVALID_PARAMETER;
    }
    *out = &APP_NOTIFICATION;
    0
}

pub unsafe extern "system" fn UnregisterAppStateChangeNotification(_registration: *const u8) {}

/// `ULONG RegisterAppConstrainedChangeNotification(routine, context, PAPPCONSTRAIN_REGISTRATION* out)`
pub unsafe extern "system" fn RegisterAppConstrainedChangeNotification(routine: usize, ctx: *mut c_void, out: *mut *const u8) -> u32 {
    RegisterAppStateChangeNotification(routine, ctx, out)
}

pub unsafe extern "system" fn UnregisterAppConstrainedChangeNotification(_registration: *const u8) {}

// ---------------------------------------------------------------------------------------
// Threads / system info
// ---------------------------------------------------------------------------------------

fn hresult_to_win32(hr: i32) -> u32 {
    if (hr as u32 & 0xFFFF_0000) == 0x8007_0000 {
        hr as u32 & 0xFFFF
    } else {
        fnd::ERROR_INVALID_PARAMETER
    }
}

/// a sleep is taken in steps of at most this, so that time held from then on is not counted
const SLEEP_STEP_MS: u32 = 20;
/// how often a sleep looks whether the title is still held
const SLEEP_HELD_POLL_MS: u32 = 5;
/// the shortest sleep that is a wait for time: a shorter one is a poll and is passed on as it is
const SLEEP_HOLD_MIN_MS: u32 = 100;

/// Sleep(ms): `ms` of the title's time. The time XCL holds one of the title's threads in a bind
/// for a pipeline still being built (`pso_async::title_held`) is not the title's: a console
/// title never waits there. A retail title's own stall check – a thread that ticks once a second
/// with Sleep and ends the process after 30 ticks without a sign of its main thread – ended it
/// while the driver built the pipelines of its world (#505, #232). Only a sleep of
/// `SLEEP_HOLD_MIN_MS` or more is taken so: a thread that polls with short sleeps (streaming,
/// sound) goes on while a bind waits. `XCL_SLEEP_HOLD=<ms>` sets that length, `0`: the PC's
/// Sleep as it is.
pub unsafe extern "system" fn Sleep(ms: u32) {
    static MIN: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let Some(min) = *MIN.get_or_init(|| sleep_hold_min(std::env::var("XCL_SLEEP_HOLD").ok().as_deref())) else {
        return thr::Sleep(ms);
    };
    title_sleep(ms, min, |ms| thr::Sleep(ms), crate::hle::d3d12x::pso_async::title_held)
}

/// `XCL_SLEEP_HOLD`: the shortest sleep whose held time is not counted (`None`: no sleep).
fn sleep_hold_min(v: Option<&str>) -> Option<u32> {
    match v.map(str::trim) {
        None => Some(SLEEP_HOLD_MIN_MS),
        Some("0") => None,
        Some(v) => Some(v.parse().unwrap_or(SLEEP_HOLD_MIN_MS)),
    }
}

/// `ms` of time in which the title was not `held`, slept through `sleep` (a yield, an endless
/// sleep and one shorter than `min` as they are).
fn title_sleep(ms: u32, min: u32, sleep: impl Fn(u32), held: impl Fn() -> bool) {
    if ms == 0 || ms == thr::INFINITE || ms < min {
        return sleep(ms);
    }
    let mut left = std::time::Duration::from_millis(ms as u64);
    while !left.is_zero() {
        if held() {
            sleep(SLEEP_HELD_POLL_MS);
            continue;
        }
        let t = std::time::Instant::now();
        sleep((left.as_millis() as u32).clamp(1, SLEEP_STEP_MS));
        left = left.saturating_sub(t.elapsed());
    }
}

pub unsafe extern "system" fn SetThreadName(h: HANDLE, name: *const u16) -> BOOL {
    if let Some(n) = from_wide_ptr(name) {
        crate::xlog_debug!("SetThreadName({n})");
    }
    let hr = thr::SetThreadDescription(h, name);
    if hr < 0 {
        fnd::SetLastError(hresult_to_win32(hr));
        return 0;
    }
    1
}

pub unsafe extern "system" fn GetThreadName(h: HANDLE, buf: *mut u16, len: usize, ret_len: *mut usize) -> BOOL {
    if buf.is_null() && len > 0 {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let mut p: *mut u16 = std::ptr::null_mut();
    let hr = thr::GetThreadDescription(h, &mut p);
    if hr < 0 || p.is_null() {
        fnd::SetLastError(hresult_to_win32(hr));
        return 0;
    }
    let s = from_wide_ptr(p).unwrap_or_default();
    fnd::LocalFree(p as *mut c_void);
    let n = s.encode_utf16().count();
    if !ret_len.is_null() {
        *ret_len = n;
    }
    if buf.is_null() || len <= n {
        fnd::SetLastError(fnd::ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    write_wide_out(&s, buf, len.min(u32::MAX as usize) as u32);
    1
}

pub unsafe extern "system" fn SetThreadpoolAffinityMask(_pool: *mut c_void, _mask: usize) -> BOOL {
    1
}

/// An affinity mask the PC takes. The title sees the PC's processors, so a mask made from them
/// passes unchanged. A console mask (its 7 cores / 14 threads) that names processors outside the
/// process's affinity – a PC with fewer processors, or a process limited to some – makes Windows
/// fail the call (ERROR_INVALID_PARAMETER) when none of its processors is there or any is outside
/// the process's mask: it is cut to the processors there are, and when none is left, each console
/// processor goes to the PC's processor at its index modulo their count.
pub fn fit_affinity_mask(mask: usize, allowed: usize) -> usize {
    if mask == 0 || allowed == 0 || mask & !allowed == 0 {
        return mask;
    }
    if mask & allowed != 0 {
        return mask & allowed;
    }
    let pcs: Vec<u32> = (0..usize::BITS).filter(|b| allowed >> b & 1 != 0).collect();
    (0..usize::BITS).filter(|b| mask >> b & 1 != 0).fold(0, |m, b| m | 1 << pcs[b as usize % pcs.len()])
}

/// The process's affinity mask; 0 (masks pass unchanged) with more than one processor group.
fn process_affinity() -> usize {
    unsafe {
        if thr::GetActiveProcessorGroupCount() != 1 {
            return 0;
        }
        let (mut process, mut system) = (0usize, 0usize);
        if thr::GetProcessAffinityMask(thr::GetCurrentProcess(), &mut process, &mut system) == 0 {
            return 0;
        }
        process
    }
}

fn fitted_affinity(mask: usize, what: &str) -> usize {
    static LOGGED: std::sync::Mutex<std::collections::BTreeSet<usize>> = std::sync::Mutex::new(std::collections::BTreeSet::new());
    let allowed = process_affinity();
    let fit = fit_affinity_mask(mask, allowed);
    if fit != mask && LOGGED.lock().unwrap_or_else(|e| e.into_inner()).insert(mask) {
        crate::xlog_info!("{what}: mask {mask:#x} names processors outside the process's {allowed:#x} – {fit:#x}");
    }
    fit
}

pub unsafe extern "system" fn SetThreadAffinityMask(h: HANDLE, mask: usize) -> usize {
    thr::SetThreadAffinityMask(h, fitted_affinity(mask, "SetThreadAffinityMask"))
}

pub unsafe extern "system" fn SetThreadGroupAffinity(h: HANDLE, a: *const GROUP_AFFINITY, prev: *mut GROUP_AFFINITY) -> BOOL {
    if a.is_null() || (*a).Group != 0 {
        return thr::SetThreadGroupAffinity(h, a, prev);
    }
    let mut fit = *a;
    fit.Mask = fitted_affinity(fit.Mask, "SetThreadGroupAffinity");
    thr::SetThreadGroupAffinity(h, &fit, prev)
}

pub unsafe extern "system" fn QueryProcessorSchedulingStatistics(out: *mut [u64; 3]) {
    if out.is_null() {
        return;
    }
    let mut idle = std::mem::zeroed::<fnd::FILETIME>();
    let mut kern = std::mem::zeroed::<fnd::FILETIME>();
    let mut user = std::mem::zeroed::<fnd::FILETIME>();
    thr::GetSystemTimes(&mut idle, &mut kern, &mut user);
    let ft = |f: fnd::FILETIME| ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
    let (i, k, u) = (ft(idle), ft(kern), ft(user));
    *out = [(k + u).saturating_sub(i), i, k + u];
}

pub unsafe extern "system" fn GetConsoleType() -> u32 {
    crate::state::config().console_type_raw
}

#[repr(C)]
pub struct SystemOsVersionInfo {
    pub major: u8,
    pub minor: u8,
    pub build: u16,
    pub revision: u16,
}

#[repr(C)]
struct OsVersionInfoW {
    size: u32,
    major: u32,
    minor: u32,
    build: u32,
    platform: u32,
    csd: [u16; 128],
}

#[link(name = "ntdll", kind = "raw-dylib")]
extern "system" {
    fn RtlGetVersion(info: *mut OsVersionInfoW) -> i32;
    fn NtQueryTimerResolution(coarsest: *mut u32, finest: *mut u32, current: *mut u32) -> i32;
    fn NtSetTimerResolution(desired: u32, set: u8, current: *mut u32) -> i32;
}

pub fn host_os_version() -> (u32, u32, u32) {
    unsafe {
        let mut v: OsVersionInfoW = std::mem::zeroed();
        v.size = std::mem::size_of::<OsVersionInfoW>() as u32;
        RtlGetVersion(&mut v);
        (v.major, v.minor, v.build)
    }
}

/// The finest timer Windows offers (0.5 ms) for the title: with the default 15.6 ms tick every
/// sleep rounds up (a retail game's audio mixer ran at a third of real time). Windows 11 keeps
/// it while the window is hidden only when told to.
pub fn fine_timer() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
    };
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        StateMask: 0,
    };
    let size = std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32;
    let kept = unsafe { SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state as *const _ as *const c_void, size) } != 0;
    // in 100 ns units; "finest" is the smallest period the system supports
    let (mut coarsest, mut finest, mut was) = (0u32, 0u32, 0u32);
    if unsafe { NtQueryTimerResolution(&mut coarsest, &mut finest, &mut was) } < 0 || finest == 0 {
        crate::xlog_warn!("timer: NtQueryTimerResolution failed – Windows' default resolution");
        return;
    }
    let mut now = 0u32;
    let status = unsafe { NtSetTimerResolution(finest, 1, &mut now) };
    crate::xlog_info!(
        "timer: {} (was {:.2} ms, now {:.2} ms; kept while the window is hidden: {})",
        if status >= 0 { "finest resolution for the title" } else { "raising the resolution failed" },
        was as f64 / 10_000.0,
        now as f64 / 10_000.0,
        if kept { "yes" } else { "no" }
    );
}

pub unsafe extern "system" fn GetSystemOSVersion(out: *mut SystemOsVersionInfo) {
    if out.is_null() {
        return;
    }
    let (major, minor, build) = host_os_version();
    *out = SystemOsVersionInfo { major: major as u8, minor: minor as u8, build: build as u16, revision: 0 };
}

// ---------------------------------------------------------------------------------------
// Registry: the console mode
// ---------------------------------------------------------------------------------------
//
// The GDK thunks a title links compute XSystemGetDeviceType from
// HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM, value ConsoleMode (REG_DWORD):
// 0x64 Series X (XboxScarlettAnaconda), 0x60 Series S (Lockhart), 0x68 devkit, 0x40 / 0x50 /
// 0x58 Xbox One S / X / devkit, no key = PC – read from the thunk in the game's XCurl.dll.
// Only that key is answered here; everything else goes to the real registry.

const OEM_KEY: &str = "software\\microsoft\\windows nt\\currentversion\\oem";
/// handle of the answered key (never reaches the real registry)
const OEM_HANDLE: usize = 0x5843_4C4F; // "XCLO"

fn console_mode() -> u32 {
    if crate::state::config().is_series_s() {
        0x60
    } else {
        0x64
    }
}

unsafe fn is_oem_key(root: usize, path: *const u16) -> bool {
    let hklm = windows_sys::Win32::System::Registry::HKEY_LOCAL_MACHINE as usize;
    root == hklm && from_wide_ptr(path).is_some_and(|p| p.trim_start_matches('\\').eq_ignore_ascii_case(OEM_KEY))
}

pub unsafe extern "system" fn RegOpenKeyExW(root: usize, path: *const u16, options: u32, sam: u32, out: *mut usize) -> u32 {
    if is_oem_key(root, path) && !out.is_null() {
        crate::xlog_debug!("registry: OEM key → console mode {:#x}", console_mode());
        *out = OEM_HANDLE;
        return 0;
    }
    windows_sys::Win32::System::Registry::RegOpenKeyExW(root as _, path, options, sam, out as _)
}

pub unsafe extern "system" fn RegOpenKeyW(root: usize, path: *const u16, out: *mut usize) -> u32 {
    if is_oem_key(root, path) && !out.is_null() {
        *out = OEM_HANDLE;
        return 0;
    }
    windows_sys::Win32::System::Registry::RegOpenKeyW(root as _, path, out as _)
}

pub unsafe extern "system" fn RegQueryValueExW(key: usize, name: *const u16, reserved: *mut u32, ty: *mut u32, data: *mut u8, size: *mut u32) -> u32 {
    if key != OEM_HANDLE {
        return windows_sys::Win32::System::Registry::RegQueryValueExW(key as _, name, reserved, ty, data, size);
    }
    const ERROR_FILE_NOT_FOUND: u32 = 2;
    const ERROR_MORE_DATA: u32 = 234;
    if !from_wide_ptr(name).is_some_and(|n| n.eq_ignore_ascii_case("ConsoleMode")) {
        return ERROR_FILE_NOT_FOUND;
    }
    if !ty.is_null() {
        *ty = 4; // REG_DWORD
    }
    if !data.is_null() {
        if size.is_null() || *size < 4 {
            if !size.is_null() {
                *size = 4;
            }
            return ERROR_MORE_DATA;
        }
        std::ptr::copy_nonoverlapping(console_mode().to_le_bytes().as_ptr(), data, 4);
    }
    if !size.is_null() {
        *size = 4;
    }
    0
}

pub unsafe extern "system" fn RegCloseKey(key: usize) -> u32 {
    if key == OEM_HANDLE {
        0
    } else {
        windows_sys::Win32::System::Registry::RegCloseKey(key as _)
    }
}

// ---------------------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------------------

const LOAD_AS_DATA: u32 = 0x2 | 0x20 | 0x40; // DATAFILE | IMAGE_RESOURCE | DATAFILE_EXCLUSIVE

fn load_library_impl(name: Option<String>) -> usize {
    let Some(name) = name else {
        unsafe { fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER) };
        return 0;
    };
    match resolver::load_library(&name) {
        Some(h) => {
            crate::xlog_debug!("LoadLibrary({name}) → {h:#x}");
            h
        }
        None => {
            crate::xlog_warn!("LoadLibrary({name}) → NULL");
            unsafe { fnd::SetLastError(fnd::ERROR_MOD_NOT_FOUND) };
            0
        }
    }
}

pub unsafe extern "system" fn LoadLibraryW(name: *const u16) -> usize {
    load_library_impl(from_wide_ptr(name))
}

pub unsafe extern "system" fn LoadLibraryA(name: *const u8) -> usize {
    load_library_impl(from_ansi_ptr(name))
}

unsafe fn load_as_data(name: &str, file: HANDLE, flags: u32) -> usize {
    let real = crate::vfs::get().and_then(|v| v.real_path_for_read(name)).unwrap_or_else(|| name.to_string());
    ll::LoadLibraryExW(crate::util::wide(&real).as_ptr(), file, flags) as usize
}

pub unsafe extern "system" fn LoadLibraryExW(name: *const u16, file: HANDLE, flags: u32) -> usize {
    if flags & LOAD_AS_DATA != 0 {
        let Some(name) = from_wide_ptr(name) else {
            fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
            return 0;
        };
        return load_as_data(&name, file, flags);
    }
    load_library_impl(from_wide_ptr(name))
}

pub unsafe extern "system" fn LoadLibraryExA(name: *const u8, file: HANDLE, flags: u32) -> usize {
    if flags & LOAD_AS_DATA != 0 {
        let Some(name) = from_ansi_ptr(name) else {
            fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
            return 0;
        };
        return load_as_data(&name, file, flags);
    }
    load_library_impl(from_ansi_ptr(name))
}

pub unsafe extern "system" fn LoadPackagedLibrary(name: *const u16, _reserved: u32) -> usize {
    load_library_impl(from_wide_ptr(name))
}

pub unsafe extern "system" fn GetProcAddress(h: usize, name: *const u8) -> usize {
    if name.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let what = if (name as usize) >> 16 == 0 {
        ImportName::Ordinal(name as usize as u16)
    } else {
        ImportName::Name { hint: 0, name: crate::util::from_cstr_ptr(name).unwrap_or_default() }
    };
    let h = if h == 0 { modules::main_module().map(|m| m.base).unwrap_or(0) } else { h };
    match resolver::get_proc_address(h, &what) {
        Some(a) => a,
        None => {
            fnd::SetLastError(fnd::ERROR_PROC_NOT_FOUND);
            0
        }
    }
}

fn module_handle_impl(name: &str) -> usize {
    module_handle_in(crate::hle::registry(), name)
}

fn module_handle_in(reg: &crate::hle::Registry, name: &str) -> usize {
    let key = dll_key(name);
    match reg.route(&key) {
        Some(crate::hle::Route::Hle(i)) => {
            let m = &reg.modules[*i];
            if m.aliases_are_system() && key != m.name {
                // Real Windows DLLs (kernel32 / kernelbase / advapi32 / Windows API sets) keep
                // their genuine HMODULE; console-only aliases (advapi32legacy, kernel32legacy,
                // minappmodelx) fall back to the HLE pseudo handle, matching LoadLibrary.
                // SAFETY: `wide(name)` produces a valid NUL-terminated UTF-16 string.
                let real = unsafe { ll::GetModuleHandleW(crate::util::wide(name).as_ptr()) as usize };
                if real != 0 {
                    return real;
                }
            }
            return modules::hle_handle(m.name);
        }
        Some(crate::hle::Route::Stub) => {
            // Return the pseudo handle only if the @stub DLL has already been loaded or imported,
            // preserving GetModuleHandle's non-loading probe semantics for optional DLLs.
            return modules::find_hle_handle(&key).unwrap_or(0);
        }
        Some(crate::hle::Route::System) => {
            // SAFETY: `wide(name)` produces a valid NUL-terminated UTF-16 string.
            return unsafe { ll::GetModuleHandleW(crate::util::wide(name).as_ptr()) as usize };
        }
        None => {}
    }
    if let Some(m) = modules::find_mapped_by_name(&key) {
        return m.base;
    }
    // SAFETY: `wide(name)` produces a valid NUL-terminated UTF-16 string.
    unsafe { ll::GetModuleHandleW(crate::util::wide(name).as_ptr()) as usize }
}

fn main_or_real_base() -> usize {
    modules::main_module()
        .map(|m| m.base)
        .unwrap_or_else(|| unsafe { ll::GetModuleHandleW(std::ptr::null()) as usize })
}

pub unsafe extern "system" fn GetModuleHandleW(name: *const u16) -> usize {
    match from_wide_ptr(name) {
        None => main_or_real_base(),
        Some(n) => {
            let h = module_handle_impl(&n);
            if h == 0 {
                fnd::SetLastError(fnd::ERROR_MOD_NOT_FOUND);
            }
            h
        }
    }
}

pub unsafe extern "system" fn GetModuleHandleA(name: *const u8) -> usize {
    match from_ansi_ptr(name) {
        None => main_or_real_base(),
        Some(n) => {
            let h = module_handle_impl(&n);
            if h == 0 {
                fnd::SetLastError(fnd::ERROR_MOD_NOT_FOUND);
            }
            h
        }
    }
}

const FROM_ADDRESS: u32 = 0x4;

unsafe fn module_handle_ex(flags: u32, name: Option<String>, addr: usize, out: *mut usize) -> BOOL {
    if out.is_null() {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let h = if flags & FROM_ADDRESS != 0 {
        match modules::image_base_for_addr(addr) {
            Some(b) => b,
            None => {
                let mut r = std::ptr::null_mut();
                if ll::GetModuleHandleExW(flags, addr as *const u16, &mut r) != 0 {
                    r as usize
                } else {
                    0
                }
            }
        }
    } else {
        match name {
            None => main_or_real_base(),
            Some(n) => module_handle_impl(&n),
        }
    };
    *out = h;
    if h == 0 {
        fnd::SetLastError(fnd::ERROR_MOD_NOT_FOUND);
        0
    } else {
        1
    }
}

pub unsafe extern "system" fn GetModuleHandleExW(flags: u32, name: *const u16, out: *mut usize) -> BOOL {
    let n = if flags & FROM_ADDRESS != 0 { None } else { from_wide_ptr(name) };
    module_handle_ex(flags, n, name as usize, out)
}

pub unsafe extern "system" fn GetModuleHandleExA(flags: u32, name: *const u8, out: *mut usize) -> BOOL {
    let n = if flags & FROM_ADDRESS != 0 { None } else { from_ansi_ptr(name) };
    module_handle_ex(flags, n, name as usize, out)
}

fn module_path_for(h: usize) -> Option<String> {
    let main = modules::main_module();
    if h == 0 || main.as_ref().map(|m| m.base) == Some(h) {
        return main.map(|m| m.path.to_string_lossy().into_owned());
    }
    if let Some(m) = modules::find_mapped_by_base(h) {
        return Some(m.path.to_string_lossy().into_owned());
    }
    modules::hle_name_by_handle(h).map(|n| format!("C:\\Windows\\System32\\{n}"))
}

pub unsafe extern "system" fn GetModuleFileNameW(h: usize, buf: *mut u16, size: u32) -> u32 {
    match module_path_for(h) {
        Some(p) => write_wide_out(&p, buf, size),
        None => ll::GetModuleFileNameW(h as _, buf, size),
    }
}

pub unsafe extern "system" fn GetModuleFileNameA(h: usize, buf: *mut u8, size: u32) -> u32 {
    match module_path_for(h) {
        Some(p) => write_ansi_out(&p, buf, size),
        None => ll::GetModuleFileNameA(h as _, buf, size),
    }
}

pub unsafe extern "system" fn FreeLibrary(h: usize) -> BOOL {
    if modules::find_mapped_by_base(h).is_some() || modules::hle_name_by_handle(h).is_some() {
        return 1;
    }
    fnd::FreeLibrary(h as _)
}

/// A thread-pool callback asks for its module to be freed when it returns (the title's
/// msvcp140 does, for the reference it took on itself). A module XCL mapped is not in Windows'
/// loader – the pool's FreeLibrary fails and raises 0xC000070E in the worker (Minecraft: 20
/// workers at once, then a fail-fast). XCL's modules are never unloaded: nothing to do for them.
pub unsafe extern "system" fn FreeLibraryWhenCallbackReturns(instance: *mut c_void, h: usize) {
    let main = modules::main_module().is_some_and(|m| m.base == h);
    if main || modules::find_mapped_by_base(h).is_some() || modules::hle_name_by_handle(h).is_some() {
        return;
    }
    thr::FreeLibraryWhenCallbackReturns(instance as _, h as _)
}

pub unsafe extern "system" fn DisableThreadLibraryCalls(h: usize) -> BOOL {
    if let Some(m) = modules::find_mapped_by_base(h) {
        m.no_thread_calls.store(true, Ordering::Relaxed);
        return 1;
    }
    ll::DisableThreadLibraryCalls(h as _)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `XCL_COHERENT_HEAPS=sparse[:N]`: which heaps go to the sparse windows
    #[test]
    fn sparse_heaps_setting() {
        assert_eq!(parse_sparse_heaps("sparse"), Some(0), "every heap");
        assert_eq!(parse_sparse_heaps("sparse:1024"), Some(1024 << 20));
        assert_eq!(parse_sparse_heaps(" sparse: 512 "), Some(512 << 20));
        assert_eq!(parse_sparse_heaps("0"), None, "heaps off the coherent window altogether: not this");
        assert_eq!(parse_sparse_heaps("1"), None);
        assert_eq!(parse_sparse_heaps("sparse:"), None);
        assert_eq!(parse_sparse_heaps("sparse:big"), None);
        assert_eq!(parse_sparse_heaps("sparsex"), None);
        assert_eq!(parse_sparse_heaps(&format!("sparse:{}", usize::MAX)), None, "no overflow");
    }

    /// The setting the coherent window's size is kept for: what goes there
    /// differs between them, and the same heaps' setting is one name however it is written.
    #[test]
    fn coherent_heaps_routes() {
        assert_eq!(heaps_route(None), crate::gpu::uma::ALL_HEAPS);
        assert_eq!(heaps_route(Some("1")), crate::gpu::uma::ALL_HEAPS, "not a setting: every heap, as VirtualAlloc reads it");
        assert_eq!(heaps_route(Some("0")), "off");
        assert_eq!(heaps_route(Some("sparse")), "sparse");
        assert_eq!(heaps_route(Some("sparse:0")), "sparse", "the same heaps as `sparse`");
        assert_eq!(heaps_route(Some(" sparse: 16 ")), "sparse:16");
    }

    /// Neighbouring sparse pages a mapping call maps are committed as one run; a gap starts
    /// another, and `holds` names the pages still waiting (they are flushed before a decommit).
    #[test]
    fn sparse_pages_are_committed_once_per_run() {
        static CALLS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
        unsafe fn note(a: usize, n: usize) {
            CALLS.lock().unwrap().push((a, n));
        }
        let a = 0x10_0000_0000usize;
        let mut run = SparseRun::new(note);
        unsafe {
            for i in 0..3 {
                run.add(a + i * PHYS_PAGE);
            }
            assert!(CALLS.lock().unwrap().is_empty(), "nothing committed while the run grows");
            assert!(run.holds(a + PHYS_PAGE) && !run.holds(a + 3 * PHYS_PAGE));
            // a page past a gap: the run before it goes first
            run.add(a + 5 * PHYS_PAGE);
            run.flush();
            run.flush();
        }
        assert_eq!(*CALLS.lock().unwrap(), vec![(a, 3 * PHYS_PAGE), (a + 5 * PHYS_PAGE, PHYS_PAGE)], "one call per run, none for an empty run");
        assert!(!run.holds(a));
    }

    fn real_sleep(ms: u32) {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    }

    /// A title's Sleep does not count the time the title is held (a bind waiting for a pipeline
    /// still being built): it returns once that is over and its own time has gone by after it.
    #[test]
    fn a_sleep_does_not_count_the_time_the_title_is_held() {
        use std::sync::atomic::AtomicBool;
        let held = AtomicBool::new(true);
        let start = std::time::Instant::now();
        std::thread::scope(|s| {
            s.spawn(|| {
                real_sleep(300);
                held.store(false, Ordering::Release);
            });
            title_sleep(60, 60, real_sleep, || held.load(Ordering::Acquire));
        });
        let took = start.elapsed();
        assert!(took >= std::time::Duration::from_millis(355), "back after {took:?}: the 300 ms held were counted");
    }

    /// Time held from the middle of a sleep on is not counted either (but for the step under way).
    #[test]
    fn a_sleep_stops_counting_when_the_title_gets_held() {
        use std::sync::atomic::AtomicBool;
        let held = AtomicBool::new(false);
        let start = std::time::Instant::now();
        std::thread::scope(|s| {
            s.spawn(|| {
                real_sleep(50);
                held.store(true, Ordering::Release);
                real_sleep(300);
                held.store(false, Ordering::Release);
            });
            title_sleep(200, SLEEP_HOLD_MIN_MS, real_sleep, || held.load(Ordering::Acquire));
        });
        let took = start.elapsed();
        // 200 ms of its own, less at most one step counted while held, and the 300 ms held
        assert!(took >= std::time::Duration::from_millis(200 + 300 - SLEEP_STEP_MS as u64 - 15), "back after {took:?}");
    }

    /// With the title not held a sleep takes its time, in steps; a yield and an endless sleep
    /// are passed on as they are.
    #[test]
    fn a_sleep_of_a_title_not_held_takes_its_time() {
        let calls = std::cell::RefCell::new(Vec::new());
        let sleep = |ms: u32| {
            calls.borrow_mut().push(ms);
            if ms != 0 && ms != thr::INFINITE {
                real_sleep(ms);
            }
        };
        title_sleep(0, 1, sleep, || false);
        title_sleep(thr::INFINITE, 1, sleep, || false);
        assert_eq!(*calls.borrow(), [0, thr::INFINITE]);
        calls.borrow_mut().clear();
        let t = std::time::Instant::now();
        title_sleep(50, 1, sleep, || false);
        let took = t.elapsed();
        assert!(took >= std::time::Duration::from_millis(50) && took < std::time::Duration::from_millis(400), "took {took:?}");
        assert!(calls.borrow().iter().all(|&ms| (1..=SLEEP_STEP_MS).contains(&ms)), "{:?}", calls.borrow());
        // a sleep shorter than a step is one sleep, as before
        calls.borrow_mut().clear();
        title_sleep(1, 1, sleep, || false);
        assert_eq!(*calls.borrow(), [1]);
    }

    /// A sleep shorter than the shortest one held is a poll: it is passed on as it is, also
    /// while the title is held. `XCL_SLEEP_HOLD` sets that length, `0` takes no sleep.
    #[test]
    fn a_short_sleep_is_passed_on_while_the_title_is_held() {
        let calls = std::cell::RefCell::new(Vec::new());
        let sleep = |ms: u32| calls.borrow_mut().push(ms);
        title_sleep(SLEEP_HOLD_MIN_MS - 1, SLEEP_HOLD_MIN_MS, sleep, || true);
        assert_eq!(*calls.borrow(), [SLEEP_HOLD_MIN_MS - 1]);
        assert_eq!(sleep_hold_min(None), Some(SLEEP_HOLD_MIN_MS));
        assert_eq!(sleep_hold_min(Some("0")), None);
        assert_eq!(sleep_hold_min(Some(" 250 ")), Some(250));
        assert_eq!(sleep_hold_min(Some("on")), Some(SLEEP_HOLD_MIN_MS));
    }

    #[test]
    fn fine_timer_makes_short_sleeps_short() {
        // Windows' Sleep (std::thread::sleep has its own high-resolution timer)
        fine_timer();
        let t = std::time::Instant::now();
        for _ in 0..20 {
            unsafe { windows_sys::Win32::System::Threading::Sleep(1) };
        }
        let per = t.elapsed() / 20;
        assert!(per < std::time::Duration::from_millis(5), "Sleep(1) took {per:?}: the 15.6 ms tick");
    }

    #[test]
    fn affinity_masks_outside_the_process_are_fitted() {
        // a mask from the PC's processors is the title's choice
        assert_eq!(fit_affinity_mask(0x3F, 0xFF), 0x3F);
        assert_eq!(fit_affinity_mask(0, 0xF), 0);
        // the console's 7 cores on a 4-processor PC: the ones there are
        assert_eq!(fit_affinity_mask(0x7F, 0xF), 0xF);
        // only processors the PC lacks: dealt onto the PC's by index
        assert_eq!(fit_affinity_mask(1 << 6, 0xF), 1 << 2);
        assert_eq!(fit_affinity_mask(0x3000, 0xF), 0x3);
        assert_eq!(fit_affinity_mask(1 << 20, 0b1010_1010), 0b10);
    }

    /// Windows refuses a mask none of whose processors the process has (ERROR_INVALID_PARAMETER);
    /// the title's call gets the processors there are. (Only XCL's answer is checked: Wine's own
    /// may differ.)
    #[test]
    fn set_thread_affinity_mask_takes_console_masks() {
        let allowed = process_affinity();
        if allowed == 0 || allowed == usize::MAX {
            println!("SKIPPED set_thread_affinity_mask_takes_console_masks: every processor bit is taken");
            return;
        }
        let missing = !allowed & allowed.wrapping_add(1);
        let first = allowed & allowed.wrapping_neg();
        std::thread::spawn(move || unsafe {
            let me = thr::GetCurrentThread();
            assert_ne!(SetThreadAffinityMask(me, missing), 0, "{missing:#x} in a {allowed:#x} process");
            assert_eq!(thr::SetThreadAffinityMask(me, allowed), fit_affinity_mask(missing, allowed));
            let mut a: GROUP_AFFINITY = std::mem::zeroed();
            a.Mask = missing | first;
            assert_eq!(SetThreadGroupAffinity(me, &a, std::ptr::null_mut()), 1);
        })
        .join()
        .unwrap();
    }

    /// Tracing keeps the last error of the call it traces: a lookup of an unset environment
    /// variable (what trace_mem's first call does) sets 203, which reached the title instead of
    /// the call's own error (xmem_and_physical_page_overflows_fail_cleanly, when it ran first).
    #[test]
    fn tracing_keeps_the_last_error() {
        unsafe {
            fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
            keep_last_error(|| assert!(std::env::var_os("XCL_TEST_UNSET_8C1F").is_none()));
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);
            fnd::SetLastError(fnd::ERROR_NOT_ENOUGH_MEMORY);
            trace_mem("test", 0, 0x1_0000, 0, 0, 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);
        }
    }

    /// The GDK thunks' device type lookup: RegOpenKeyW(HKLM\…\OEM) + RegQueryValueExW(ConsoleMode).
    #[test]
    fn console_mode_registry_value() {
        unsafe {
            let hklm = windows_sys::Win32::System::Registry::HKEY_LOCAL_MACHINE as usize;
            let path = crate::util::wide("Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM");
            let mut k = 0usize;
            assert_eq!(RegOpenKeyW(hklm, path.as_ptr(), &mut k), 0);
            let (mut ty, mut v, mut n) = (0u32, 0u32, 4u32);
            let name = crate::util::wide("ConsoleMode");
            assert_eq!(RegQueryValueExW(k, name.as_ptr(), std::ptr::null_mut(), &mut ty, &mut v as *mut u32 as *mut u8, &mut n), 0);
            assert_eq!((ty, n), (4, 4));
            assert_eq!(v, console_mode());
            let other = crate::util::wide("Other");
            assert_eq!(RegQueryValueExW(k, other.as_ptr(), std::ptr::null_mut(), &mut ty, &mut v as *mut u32 as *mut u8, &mut n), 2);
            assert_eq!(RegCloseKey(k), 0);
            // any other key: the real registry
            let real = crate::util::wide("Software\\Microsoft\\Windows NT\\CurrentVersion");
            let mut r = 0usize;
            assert_eq!(RegOpenKeyExW(hklm, real.as_ptr(), 0, 0x20019, &mut r), 0);
            assert_ne!(r, OEM_HANDLE);
            assert_eq!(RegCloseKey(r), 0);
        }
    }

    #[test]
    fn console_alloc_flags_are_stripped() {
        assert_eq!(sanitize_alloc_type(MEM_GRAPHICS | 0x1000 | 0x2000), 0x3000);
        assert_eq!(sanitize_alloc_type(MEM_TITLE), 0x3000);
        assert_eq!(sanitize_protect(0x1000_0004), 0x04);
        assert_eq!(sanitize_protect(0x2000_0000), mem::PAGE_READWRITE);
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 0x10000, MEM_GRAPHICS | MEM_TITLE | 0x3000, 0x0400_0004);
            assert!(!p.is_null());
            *(p as *mut u64) = 0xABCD;
            // graphics memory: from the GPU-visible region (when it could be created)
            if let Some(u) = crate::gpu::uma::get() {
                assert!(u.contains(p as usize));
            }
            assert_eq!(VirtualFree(p, 0, mem::MEM_RELEASE), 1);
            let q = VirtualAlloc(std::ptr::null(), 0x10000, 0x3000, 0x04); // ordinary memory: Windows
            assert!(crate::gpu::uma::get().is_none_or(|u| !u.contains(q as usize)));
            assert_eq!(VirtualFree(q, 0, mem::MEM_RELEASE), 1);
        }
    }

    #[test]
    fn xmem_placeholder_split_replace_and_coalesce() {
        unsafe {
            // The title splits a placeholder, replaces each piece with a physical-page
            // reservation, then maps pages at those exact GPU-visible addresses.
            let grain = PHYS_PAGE;
            let base = XMemVirtualAlloc(std::ptr::null(), 3 * grain, 0x42000, 7, mem::PAGE_NOACCESS);
            assert!(!base.is_null());
            let split = VirtualFree(base, grain, 0x8002);
            if split == 0 {
                VirtualFree(base, 0, mem::MEM_RELEASE);
            }
            assert_eq!(split, 1, "split a placeholder without releasing its address space");
            let rest = base.byte_add(grain);
            assert_eq!(VirtualFree(rest, grain, 0x8002), 1);
            assert_eq!(XMemVirtualAlloc(base, grain, 0x20406000, 0x27, 0x1004), base);
            assert!(XMemVirtualAlloc(rest, 2 * grain, 0x20406000, 0x27, 0x1004).is_null(), "replacement must match one placeholder");
            if let Some(u) = crate::gpu::uma::get() {
                assert!(u.is_title_address(base as usize), "replacement stays GPU-visible");
            }
            assert_eq!(VirtualAlloc(base, grain, mem::MEM_COMMIT, mem::PAGE_READWRITE), base);
            *(base as *mut u32) = 0x12345678;
            assert_eq!(VirtualFree(base, grain, 0x8002), 1, "return a replacement to a placeholder");
            assert_eq!(VirtualFree(base, 3 * grain, 0x8001), 1, "coalesce adjacent placeholders");
            assert_eq!(VirtualFree(base, 0, mem::MEM_RELEASE), 1);
        }
    }

    #[test]
    fn xmem_graphics_page_size_two_uses_two_mib_grain() {
        const PAGE: usize = 2 << 20;
        let attrs = XAllocAttrs::decode((2 << 14) | (1 << 29));
        for (size, rounded) in [(0, PAGE), (1, PAGE), (PAGE - 1, PAGE), (PAGE, PAGE), (PAGE + 1, 2 * PAGE), (2 * PAGE, 2 * PAGE)] {
            assert_eq!(xmem_graphics_layout(size, attrs), Some((rounded, PAGE)), "size {size}");
        }
        let larger_alignment = XAllocAttrs::decode((2 << 14) | (23 << 24) | (1 << 29));
        assert_eq!(xmem_graphics_layout(1, larger_alignment), Some((PAGE, 8 << 20)), "alignment does not enlarge the allocation's page grain");
    }

    #[test]
    fn xmem_graphics_page_grain_preserves_other_codes_and_checks_overflow() {
        const PAGE: usize = 2 << 20;
        // Code 3 is reserved; keep its existing fallback rather than defining a new contract here.
        for code in [0, 1, 3] {
            let attrs = XAllocAttrs::decode((code << 14) | (1 << 29));
            assert_eq!(xmem_graphics_layout(0x1_0001, attrs), Some((0x2_0000, 0x1_0000)));
            assert!(xmem_graphics_layout(usize::MAX, attrs).is_none());
        }
        let attrs = XAllocAttrs::decode((2 << 14) | (1 << 29));
        let largest = usize::MAX & !(PAGE - 1);
        assert_eq!(xmem_graphics_layout(largest, attrs), Some((largest, PAGE)));
        assert!(xmem_graphics_layout(largest + 1, attrs).is_none());
        assert!(xmem_graphics_layout(usize::MAX, attrs).is_none());
    }

    #[test]
    fn xmem_two_mib_layout_accounts_and_releases_owned_memory() {
        const PAGE: usize = 2 << 20;
        // An owned CPU-only window keeps global UMA state and GPU initialization out of the fixture.
        let u = crate::gpu::uma::Uma::with_windows(1, 16 << 20).expect("owned memory window");
        let attrs = XAllocAttrs::decode((2 << 14) | (1 << 29));
        for (requested, expected) in [(1, PAGE), (PAGE, PAGE), (PAGE + 1, 2 * PAGE)] {
            let (size, align) = xmem_graphics_layout(requested, attrs).unwrap();
            let p = u.alloc(size, align).expect("owned allocation");
            let (recorded, used) = (u.block_len(p as usize), u.used());
            let address = p as usize;
            unsafe {
                p.write(0x37);
                p.add(requested - 1).write(0x69);
            }
            // Reclaim before assertions so the negative control also leaves no live allocation.
            let released = u.free(address);
            assert!(released);
            assert_eq!(u.used(), 0);
            assert_eq!(u.block_len(address), None);
            assert!(!u.free(address), "one allocation is released only once");
            assert_eq!(address % PAGE, 0);
            assert_eq!(recorded, Some(expected), "request {requested}: recorded allocation size");
            assert_eq!(used, expected, "request {requested}: memory charged");
        }
    }

    #[test]
    fn xmem_alignment_and_zeroing() {
        let attrs = |align_log2: u64, mtype: u64| (align_log2 << 24) | (mtype << 29);
        assert_eq!(XAllocAttrs::decode(attrs(8, 0)).alignment, 8);
        unsafe {
            let p = XMemAlloc(1000, attrs(8, 0));
            assert!(!p.is_null());
            assert_eq!(p as usize % 256, 0);
            assert!(std::slice::from_raw_parts(p as *const u8, 1000).iter().all(|&b| b == 0));
            XMemFree(p, attrs(8, 0));
            let g = XMemAlloc(0x20000, attrs(0, 1)); // graphics memory → VirtualAlloc
            assert!(!g.is_null());
            assert_eq!(g as usize % 0x10000, 0);
            XMemFree(g, attrs(0, 1));
        }
    }

    #[test]
    fn xmem_virtual_query_marks_cpu_readable_memory() {
        unsafe {
            let mut info = [0u8; 64];
            let x = 5u32;
            let proc_ = thr::GetCurrentProcess();
            assert_eq!(XMemVirtualQuery(proc_, &x as *const u32 as _, info.as_mut_ptr() as _), 1);
            let mbi = &*(info.as_ptr() as *const mem::MEMORY_BASIC_INFORMATION);
            assert_eq!(mbi.State, mem::MEM_COMMIT);
            assert_eq!(u32::from_le_bytes(info[48..52].try_into().unwrap()) & 3, 1);
            // reserved, not committed: not CPU-readable
            let r = mem::VirtualAlloc(std::ptr::null(), 0x10000, mem::MEM_RESERVE, mem::PAGE_NOACCESS);
            let mut info = [0u8; 64];
            assert_eq!(XMemVirtualQuery(proc_, r, info.as_mut_ptr() as _), 1);
            assert_eq!(info[48] & 3, 0);
            mem::VirtualFree(r, 0, mem::MEM_RELEASE);
        }
    }

    /// The field at +48 is a 16-byte flags field, not just the low u32: callers that inspect
    /// GPU visibility or the whole field must not see the buffer's previous contents.
    #[test]
    fn xmem_virtual_query_writes_the_whole_flag_field_and_gpu_visibility() {
        #[repr(C, align(16))]
        struct Info([u8; 64]);

        unsafe {
            let proc_ = thr::GetCurrentProcess();
            let x = 5u32;
            let mut info = Info([0xCC; 64]);
            assert_eq!(XMemVirtualQuery(proc_, &x as *const u32 as _, info.0.as_mut_ptr() as _), 1);
            let flags0 = u32::from_ne_bytes(info.0[48..52].try_into().unwrap());
            assert_eq!(flags0 & 3, 1, "first flags were {flags0:#x}");
            assert_eq!(flags0 & 2, 0, "plain stack memory is not GPU-visible");
            assert!(info.0[52..64].iter().all(|&b| b == 0), "flag tail was not written: {:?}", &info.0[48..64]);

            // A reserved page has no CPU-readable flag either, but the tail is still written.
            let r = mem::VirtualAlloc(std::ptr::null(), 0x10000, mem::MEM_RESERVE, mem::PAGE_NOACCESS);
            assert!(!r.is_null());
            let mut info = Info([0xCC; 64]);
            assert_eq!(XMemVirtualQuery(proc_, r, info.0.as_mut_ptr() as _), 1);
            assert_eq!(u32::from_ne_bytes(info.0[48..52].try_into().unwrap()) & 3, 0);
            assert!(info.0[52..64].iter().all(|&b| b == 0));
            mem::VirtualFree(r, 0, mem::MEM_RELEASE);

            // Coherent UMA and sparse VRAM are visible to both the CPU and the GPU. Skip only
            // if UMA is unavailable or has no room: that depends on the machine's address space.
            let Some(u) = crate::gpu::uma::get() else { return };
            if let Some(p) = u.alloc(0x1000, 0x1000) {
                let mut info = Info([0xCC; 64]);
                assert_eq!(XMemVirtualQuery(proc_, p as *const c_void, info.0.as_mut_ptr() as _), 1);
                let flags = u32::from_ne_bytes(info.0[48..52].try_into().unwrap());
                assert_eq!(flags & 3, 3, "UMA memory must be CPU- and GPU-visible: {flags:#x}");
                assert!(info.0[52..64].iter().all(|&b| b == 0));
                assert!(u.free(p as usize));
            }
            if let Some((s, len)) = u.sparse_alloc(0x1000, 0x1000) {
                let mut info = Info([0xCC; 64]);
                assert_eq!(XMemVirtualQuery(proc_, s as *const c_void, info.0.as_mut_ptr() as _), 1);
                let flags = u32::from_ne_bytes(info.0[48..52].try_into().unwrap());
                assert_eq!(flags & 2, 2, "sparse VRAM must be reported as GPU-visible: {flags:#x}");
                assert!(info.0[52..64].iter().all(|&b| b == 0));
                assert_eq!(u.sparse_release(s), Some(len));
            }
        }
    }

    #[test]
    fn physical_pages_in_uma() {
        unsafe {
            let Some(u) = crate::gpu::uma::get() else { return };
            let mut n = 3usize;
            let mut pages = [0usize; 3];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            // 4 KiB frame numbers: multiples of 16, consecutive pages 16 apart
            assert!(pages[0].is_multiple_of(16) && pages[1] == pages[0] + 16);
            let p = MapTitlePhysicalPages(std::ptr::null_mut(), 3, 0, 4, pages.as_mut_ptr());
            assert!(!p.is_null() && u.contains(p as usize) && (p as usize - u.base()) % PHYS_PAGE == 0);
            *(p as *mut u32).add(PHYS_PAGE / 4) = 0x55AA; // page 1
            // the same pages mapped again: the same memory
            let q = MapTitlePhysicalPages(p, 3, 0, 4, pages.as_mut_ptr());
            assert_eq!(q, p);
            // a single page of the block maps to its own place
            let r = MapTitlePhysicalPages(std::ptr::null_mut(), 1, 0, 4, pages.as_mut_ptr().add(1));
            assert_eq!(*(r as *const u32), 0x55AA);
            // scattered pages: refused
            let mut scattered = [pages[2], pages[0]];
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, scattered.as_mut_ptr()).is_null());
            // … but one by one they go to addresses of their own
            let at = [page_home(pages[0]).unwrap(), page_home(pages[2]).unwrap()].map(|a| a as *mut c_void);
            let mut ids = [pages[0] as u64, pages[2] as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 2, ids.as_mut_ptr()), 1);
            ids[1] += 1;
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 2, ids.as_mut_ptr()), 0, "not a page number");
            // Page numbers must refer to one registered allocation; arithmetic is checked before
            // constructing an address inside the reserved UMA region.
            let mut outside = [page_number(u.size() / PHYS_PAGE)];
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 1, 0, 4, outside.as_mut_ptr()).is_null());
            let mut wrapping = [usize::MAX];
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, wrapping.as_mut_ptr()).is_null());
            // freed: the block is gone (other tests use UMA at the same time, so not `u.used()`)
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 3, pages.as_mut_ptr()), 1);
            assert!(!PHYS_BLOCKS.lock().unwrap().contains_key(&page_index(pages[0]).unwrap()));
            assert!(MapTitlePhysicalPages(std::ptr::null_mut(), 3, 0, 4, pages.as_mut_ptr()).is_null());
        }
    }

    /// A page mapped away from a home in a sparse window is committed once: the home gives its
    /// memory back while the page is away and has the page's contents again when it returns.
    #[test]
    fn a_page_away_from_home_is_committed_once() {
        unsafe {
            let Some(u) = crate::gpu::uma::get() else { return };
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let home = page_home(pages[1]).unwrap();
            if !u.is_sparse(home) {
                FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr());
                return;
            }
            assert!(!committed_page(home), "a sparse home is only reserved until it is used");
            let mut never = [pages[0]];
            let r0 = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert_eq!(MapTitlePhysicalPages(r0 as *mut c_void, 1, 0, 4, never.as_mut_ptr()) as usize, r0);
            assert_eq!(*(r0 as *const u32), 0, "a page never used reads as zeros");
            assert!(!committed_page(home - PHYS_PAGE), "and its home stays reserved");
            assert_eq!(MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, pages.as_mut_ptr()) as usize + PHYS_PAGE, home);
            *(home as *mut u32) = 0x1234_5678;
            let r = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let mut one = [pages[1]];
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r);
            assert_eq!(*(r as *const u32), 0x1234_5678);
            assert!(!committed_page(home), "the home is given back while the page is away");
            *(r as *mut u32) = 0x9ABC_DEF0;
            // back home (the whole allocation at its home): committed, with what was written away
            let p = MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, pages.as_mut_ptr()) as usize;
            assert_eq!(p + PHYS_PAGE, home);
            assert!(committed_page(home));
            assert_eq!(*(home as *const u32), 0x9ABC_DEF0);
            assert!(!committed_page(r), "the other address is decommitted");
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
            mem::VirtualFree(r0 as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// Fresh pages mapped into a reservation in one call are left lazy with a few VirtualQuery
    /// calls, not two per page (13 s under Wine for a title's 98,304 pages); a committed page
    /// among them is still told from the reserved ones.
    #[test]
    fn mapping_fresh_pages_does_not_query_each_page() {
        const N: usize = 64;
        unsafe {
            let Some(u) = crate::gpu::uma::get() else { return };
            let mut n = N;
            let mut pages = [0usize; N];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let home = page_home(pages[0]).unwrap();
            if !u.is_sparse(home) {
                FreeTitlePhysicalPages(std::ptr::null_mut(), N, pages.as_mut_ptr());
                return;
            }
            let r = mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let used = r + 3 * PHYS_PAGE;
            assert!(!mem::VirtualAlloc(used as *const c_void, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
            *(used as *mut u32) = 0xABAB_ABAB;
            let before = VIRTUAL_QUERIES.with(|q| q.get());
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, r);
            let queries = VIRTUAL_QUERIES.with(|q| q.get()) - before;
            assert!(queries <= 8, "{queries} VirtualQuery calls for {N} pages");
            assert_eq!(*(used as *const u32), 0, "a committed page is told from the reserved ones: it is not left lazy");
            assert!(!committed_page(r + 2 * PHYS_PAGE) && !committed_page(r + 4 * PHYS_PAGE), "the others are left lazy");
            assert!(!committed_page(page_home(pages[N - 1]).unwrap()), "and their homes stay reserved");
            assert_eq!(*((r + 4 * PHYS_PAGE) as *const u32), 0);
            // back home: the pages left lazy are unmarked (a freed one stays marked, and the address
            // is another test's later)
            assert!(!MapTitlePhysicalPages(std::ptr::null_mut(), N, 0, 4, pages.as_mut_ptr()).is_null());
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), N, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// `XMemMapPhysicalPagesScatter` with no page array unmaps each address, as page 0 does (a
    /// retail game unmaps so while it streams; refused, it stopped on its own `int3`).
    #[test]
    fn scatter_without_pages_unmaps() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let mut n = 2;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let r = mem::VirtualAlloc(std::ptr::null(), 2 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert!(r != 0);
            let addrs = [r as *mut c_void, (r + PHYS_PAGE) as *mut c_void];
            let mut ids = [pages[0] as u64, pages[1] as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, ids.as_mut_ptr()), 1);
            (r as *mut u32).write_volatile(7);
            ((r + PHYS_PAGE) as *mut u32).write_volatile(9);
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, std::ptr::null_mut()), 1, "no page array: unmapped, not refused");
            for a in addrs {
                assert!(!crate::gpu::shader::readable(a as usize, 4), "{a:p} still mapped");
            }
            // the pages kept what was written: mapped again, they read it
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, ids.as_mut_ptr()), 1);
            assert_eq!(((r as *const u32).read_volatile(), ((r + PHYS_PAGE) as *const u32).read_volatile()), (7, 9));
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, std::ptr::null_mut()), 1);
            // no addresses: the same answer with and without a page array
            let mut none = [0u64; 1];
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 0, std::ptr::null_mut()), XMemMapPhysicalPagesScatter(addrs.as_ptr(), 0, none.as_mut_ptr()));
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// `n` pages with a sparse home (None, with nothing allocated: not on this adapter).
    unsafe fn sparse_pages(n: usize) -> Option<Vec<usize>> {
        let u = crate::gpu::uma::get()?;
        let (mut num, mut pages) = (n, vec![0usize; n]);
        assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut num, pages.as_mut_ptr()), 1);
        if u.is_sparse(page_home(pages[0]).unwrap()) {
            return Some(pages);
        }
        FreeTitlePhysicalPages(std::ptr::null_mut(), n, pages.as_mut_ptr());
        None
    }

    /// Pages mapped, half of them written, then all mapped again in one call – at another
    /// address or the same one: the written ones keep their contents. The first pages of the
    /// call are never-used (nothing to move), the ones after them are moved: what a run of
    /// the pages asked about before says is no longer true (the home they moved to is committed
    /// by now, the address they left decommitted).
    unsafe fn remap_keeps_what_was_written(same_address: bool) {
        const N: usize = 8;
        let Some(mut pages) = sparse_pages(N) else { return };
        let r1 = mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
        let r2 = if same_address { r1 } else { mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize };
        assert_eq!(MapTitlePhysicalPages(r1 as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, r1);
        for i in N / 2..N {
            *((r1 + i * PHYS_PAGE) as *mut u32) = 0x100 + i as u32;
        }
        assert_eq!(MapTitlePhysicalPages(r2 as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, r2);
        for i in 0..N {
            let want = if i >= N / 2 { 0x100 + i as u32 } else { 0 };
            assert_eq!(*((r2 + i * PHYS_PAGE) as *const u32), want, "page {i} (same address: {same_address})");
        }
        // back home: the pages left lazy are unmarked (a freed one stays marked, and the address
        // is another test's later)
        assert!(!MapTitlePhysicalPages(std::ptr::null_mut(), N, 0, 4, pages.as_mut_ptr()).is_null());
        assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), N, pages.as_mut_ptr()), 1);
        mem::VirtualFree(r1 as *mut c_void, 0, mem::MEM_RELEASE);
        if !same_address {
            mem::VirtualFree(r2 as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    #[test]
    fn pages_mapped_again_at_another_address_keep_what_was_written() {
        unsafe { remap_keeps_what_was_written(false) }
    }

    #[test]
    fn pages_mapped_again_at_the_same_address_keep_what_was_written() {
        unsafe { remap_keeps_what_was_written(true) }
    }

    /// Pages never used, mapped again (they are only unmarked: memory is as it was) cost no
    /// more queries than the first time.
    #[test]
    fn mapping_lazy_pages_again_does_not_query_each_page() {
        const N: usize = 64;
        unsafe {
            let Some(mut pages) = sparse_pages(N) else { return };
            let r1 = mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let r2 = mem::VirtualAlloc(std::ptr::null(), N * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert_eq!(MapTitlePhysicalPages(r1 as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, r1);
            for to in [r1, r2] {
                let before = VIRTUAL_QUERIES.with(|q| q.get());
                assert_eq!(MapTitlePhysicalPages(to as *mut c_void, N, 0, 4, pages.as_mut_ptr()) as usize, to);
                let queries = VIRTUAL_QUERIES.with(|q| q.get()) - before;
                assert!(queries <= 8, "{queries} VirtualQuery calls for {N} pages mapped again");
            }
            // back home: the pages left lazy are unmarked (a freed one stays marked, and the address
            // is another test's later)
            assert!(!MapTitlePhysicalPages(std::ptr::null_mut(), N, 0, 4, pages.as_mut_ptr()).is_null());
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), N, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r1 as *mut c_void, 0, mem::MEM_RELEASE);
            mem::VirtualFree(r2 as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// `XMemMapPhysicalPagesScatter` against a model of physical pages: random scatters of
    /// pages to addresses (page 0: unmap), writes through the addresses and maps of the whole
    /// allocation at its home. After each step every page reads what was last written to it
    /// wherever it is mapped, a page never written reads zeros, and an address with no page is
    /// not committed.
    #[test]
    fn scatter_matches_a_model_of_physical_pages() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            const PAGES: usize = 24;
            const SLOTS: usize = 32;
            for seed in 1..=4u64 {
                let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let mut next = move || {
                    x ^= x >> 12;
                    x ^= x << 25;
                    x ^= x >> 27;
                    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
                };
                let mut n = PAGES;
                let mut pages = [0usize; PAGES];
                assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
                let r = mem::VirtualAlloc(std::ptr::null(), SLOTS * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
                assert!(r != 0);
                let slot = |s: usize| r + s * PHYS_PAGE;
                // model: what each page holds (first and last dword), where it is (None: home)
                let mut value = [(0u32, 0u32); PAGES];
                let mut at: [Option<usize>; PAGES] = [None; PAGES];
                let mut held: [Option<usize>; SLOTS] = [None; SLOTS];
                let mut home: Option<usize> = None;
                let read = |a: usize| ((a as *const u32).read_volatile(), ((a + PHYS_PAGE - 4) as *const u32).read_volatile());
                for step in 0..600 {
                    match next() % 8 {
                        // a scatter of 1–4 (slot, page or none)
                        0..=3 => {
                            let k = 1 + (next() % 4) as usize;
                            let mut addrs = Vec::new();
                            let mut ids = Vec::new();
                            for _ in 0..k {
                                let s = (next() % SLOTS as u64) as usize;
                                let p = if next() % 5 == 0 { None } else { Some((next() % PAGES as u64) as usize) };
                                addrs.push(slot(s) as *mut c_void);
                                ids.push(p.map_or(0, |p| pages[p] as u64));
                                // model, in call order: what the slot held goes home; the page leaves
                                // where it was
                                if let Some(q) = held[s].take() {
                                    at[q] = None;
                                }
                                if let Some(p) = p {
                                    if let Some(old) = at[p].take() {
                                        held[old] = None;
                                    }
                                    at[p] = Some(s);
                                    held[s] = Some(p);
                                }
                            }
                            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), k, ids.as_mut_ptr()), 1, "seed {seed} step {step}");
                        }
                        // a write through a mapped slot
                        4..=6 => {
                            let s = (next() % SLOTS as u64) as usize;
                            if let Some(p) = held[s] {
                                let v = ((next() as u32) | 1, (next() as u32) | 1);
                                (slot(s) as *mut u32).write_volatile(v.0);
                                ((slot(s) + PHYS_PAGE - 4) as *mut u32).write_volatile(v.1);
                                value[p] = v;
                            }
                        }
                        // the whole allocation at its home: every page leaves its slot
                        _ => {
                            let h = MapTitlePhysicalPages(std::ptr::null_mut(), PAGES, 0, 4, pages.as_mut_ptr()) as usize;
                            assert!(h != 0);
                            home = Some(h);
                            at = [None; PAGES];
                            held = [None; SLOTS];
                            for (p, &v) in value.iter().enumerate() {
                                assert_eq!(read(h + p * PHYS_PAGE), v, "seed {seed} step {step}: page {p} at home");
                            }
                        }
                    }
                    for s in 0..SLOTS {
                        match held[s] {
                            Some(p) => assert_eq!(read(slot(s)), value[p], "seed {seed} step {step}: page {p} at slot {s}"),
                            None => assert!(!committed_page(slot(s)), "seed {seed} step {step}: slot {s} has no page but is committed"),
                        }
                    }
                }
                let _ = home;
                assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), PAGES, pages.as_mut_ptr()), 1);
                mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
            }
        }
    }

    /// A home a page went back to in a call, then decommitted in the same call because another
    /// page was mapped at that address and unmapped again, is not taken as committed: mapping
    /// its page away does not read decommitted memory.
    #[test]
    fn a_home_decommitted_in_the_same_call_is_asked_again() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let (p, q) = (pages[0], pages[1]);
            let h = page_home(p).unwrap();
            let r = mem::VirtualAlloc(std::ptr::null(), 2 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let (s, t) = (r as *mut c_void, (r + PHYS_PAGE) as *mut c_void);
            let mut one = [p as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&s, 1, one.as_mut_ptr()), 1);
            *(s as *mut u32) = 7;
            // one call: p home (h holds it), q at h, h unmapped (decommitted), p to t
            let at = [s, h as *mut c_void, h as *mut c_void, t];
            let mut ids = [0, q as u64, 0, p as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 4, ids.as_mut_ptr()), 1);
            // (t reads: committed, or committed on first use)
            let _ = (t as *const u32).read_volatile();
            assert!(!committed_page(s as usize));
            let mut none = [0u64, 0];
            let both = [s, t];
            assert_eq!(XMemMapPhysicalPagesScatter(both.as_ptr(), 2, none.as_mut_ptr()), 1);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    thread_local! {
        /// pages `unmap_away` copied back home on this thread
        pub(super) static HOME_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// runs of homes `commit_homes_ahead` committed on this thread
        pub(super) static HOME_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// what `source_going` told sparse on this thread: (start, bytes, whether the first page
        /// was still committed then)
        pub(super) static SOURCES_GOING: std::cell::RefCell<Vec<(usize, usize, bool)>> = const { std::cell::RefCell::new(Vec::new()) };
        /// what `source_going` told sparse on this thread of memory of no pair while pages were
        /// shown (`mirrors::any_mapped`): (start, bytes, whether the first page was still
        /// committed then)
        pub(super) static KEPT_GOING: std::cell::RefCell<Vec<(usize, usize, bool)>> = const { std::cell::RefCell::new(Vec::new()) };
        /// what `page_placed` told sparse on this thread: (page, the place the tiles follow it
        /// from – 0: none, its new place, the GPU-view pages whose tiles follow it, those linked
        /// again)
        pub(super) static PLACED: std::cell::RefCell<Vec<(usize, usize, usize, Vec<usize>, Vec<usize>)>> = const { std::cell::RefCell::new(Vec::new()) };
        /// the ranges `mirrors::released` had sparse forget on this thread
        pub(super) static FORGOTTEN: std::cell::RefCell<Vec<(usize, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// The CPU view of a pair (`mirrors`) as the title's is – plain write-watched memory – with
    /// a GPU view at `g` (only paired: never touched). `size` is the test's own, so no other
    /// test's views pair with these.
    unsafe fn cpu_view(size: usize, g: usize) -> usize {
        let c = reserve_watched(size, 0x2040_2000, mem::PAGE_READWRITE) as usize;
        assert!(c != 0 && in_graphics_reservation(c), "the CPU view is a graphics reservation: the looks take it");
        mirrors::reserved(c, size, 0x2040_2000, mem::PAGE_READWRITE);
        assert!(mirrors::reserved(g, size, 0x2040_2000, 0x801));
        assert_eq!(mirrors::source(g + 0x1_2340), Some(c + 0x1_2340));
        c
    }

    /// The title's decommit and release of its CPU view are told to sparse while the pages are
    /// still committed (a submit's scan reads the pages tiles of the GPU view show, and faulted
    /// on one decommitted under it), the release as the whole view; memory of no pair is not
    /// told. The release ends the pair, and sparse forgets the tiles over either view (their
    /// addresses may be handed out again).
    #[test]
    fn a_decommit_or_release_of_a_cpu_view_is_told_to_sparse_first() {
        const SIZE: usize = 61 * PHYS_PAGE;
        let g = 0x7230_0000_0000usize;
        unsafe {
            let c = cpu_view(SIZE, g);
            let other = reserve_watched(SIZE, 0x2000, mem::PAGE_READWRITE) as usize;
            for at in [c, c + 3 * PHYS_PAGE, other + PHYS_PAGE] {
                assert_eq!(VirtualAlloc(at as *const c_void, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize, at);
            }
            SOURCES_GOING.with(|s| s.borrow_mut().clear());
            FORGOTTEN.with(|f| f.borrow_mut().clear());
            assert_eq!(VirtualFree((c + 3 * PHYS_PAGE + 0x4000) as *mut c_void, 0x2000, mem::MEM_DECOMMIT), 1);
            assert_eq!(SOURCES_GOING.with(|s| s.take()), vec![(c + 3 * PHYS_PAGE + 0x4000, 0x2000, true)], "told while it was still committed");
            assert!(!committed_page(c + 3 * PHYS_PAGE + 0x4000) && committed_page(c + 3 * PHYS_PAGE));
            // through VirtualFreeEx too
            assert_eq!(VirtualFreeEx(thr::GetCurrentProcess(), (c + 3 * PHYS_PAGE) as *mut c_void, 0x1000, mem::MEM_DECOMMIT), 1);
            assert_eq!(SOURCES_GOING.with(|s| s.take()), vec![(c + 3 * PHYS_PAGE, 0x1000, true)]);
            assert_eq!(VirtualFree((other + PHYS_PAGE) as *mut c_void, PHYS_PAGE, mem::MEM_DECOMMIT), 1);
            assert!(SOURCES_GOING.with(|s| s.take()).is_empty(), "memory of no pair");
            // a release that names no reservation's start does nothing
            assert_eq!(VirtualFree(c as *mut c_void, PHYS_PAGE, mem::MEM_RELEASE), 0);
            assert!(SOURCES_GOING.with(|s| s.take()).is_empty());
            assert_eq!(VirtualFree(c as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert_eq!(SOURCES_GOING.with(|s| s.take()), vec![(c, SIZE, true)], "the whole view, before Windows releases it");
            assert_eq!(FORGOTTEN.with(|f| f.take()), vec![(g, SIZE), (c, SIZE)]);
            assert_eq!(mirrors::source(g), None);
            assert_eq!(VirtualFree(other as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert!(SOURCES_GOING.with(|s| s.take()).is_empty() && FORGOTTEN.with(|f| f.take()).is_empty());
        }
    }

    /// A physical page mapped never used into the CPU view of a pair is left lazy; the title's
    /// first write commits it – the tiles of the GPU view may have been asked for before, with a
    /// hole there – and the look finds the page written, which is how sparse clears the hole and
    /// sends the page. Moved to another address, or unmapped, the page is told to sparse while it
    /// is still committed (`release_at`).
    #[test]
    fn a_lazy_page_of_a_cpu_view_is_found_written_and_told_to_sparse_when_it_goes() {
        const SIZE: usize = 59 * PHYS_PAGE;
        let g = 0x7240_0000_0000usize;
        unsafe {
            let Some(mut pages) = sparse_pages(1) else { return };
            let c = cpu_view(SIZE, g);
            let at = c + 2 * PHYS_PAGE;
            assert_eq!(MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, at);
            assert!(lazy::is_marked(at) && !committed_page(at), "never used: left lazy");
            let mut buf = [std::ptr::null_mut::<c_void>(); 16];
            let (mut n, mut gran) = (buf.len(), 0u32);
            assert_eq!(mem::GetWriteWatch(1, c as *const c_void, SIZE, buf.as_mut_ptr(), &mut n, &mut gran), 0);
            ((at + 0x1238) as *mut u32).write_volatile(0xC0FF_EE00);
            assert!(committed_page(at) && !lazy::is_marked(at), "committed by the write");
            n = buf.len();
            assert_eq!(mem::GetWriteWatch(1, c as *const c_void, SIZE, buf.as_mut_ptr(), &mut n, &mut gran), 0);
            assert_eq!(&buf[..n], &[(at + 0x1000) as *mut c_void], "the look finds the page written");

            SOURCES_GOING.with(|s| s.borrow_mut().clear());
            let to = c + 5 * PHYS_PAGE;
            assert_eq!(MapTitlePhysicalPages(to as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, to);
            assert_eq!(*((to + 0x1238) as *const u32), 0xC0FF_EE00);
            assert_eq!(SOURCES_GOING.with(|s| s.take()), vec![(at, PHYS_PAGE, true)], "the address it moved from, before it was decommitted");
            assert!(!committed_page(at));
            let mut none = [0usize];
            assert_eq!(MapTitlePhysicalPages(to as *mut c_void, 1, 0, 4, none.as_mut_ptr()) as usize, to);
            assert_eq!(SOURCES_GOING.with(|s| s.take()), vec![(to, PHYS_PAGE, true)], "unmapped");
            assert!(!committed_page(to));
            // a page of no pair goes untold
            let r = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, r);
            *(r as *mut u32) = 1;
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, none.as_mut_ptr()) as usize, r);
            assert!(SOURCES_GOING.with(|s| s.take()).is_empty());
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr()), 1);
            assert_eq!(VirtualFree(c as *mut c_void, 0, mem::MEM_RELEASE), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// Pages allocated and mapped one by one (page i at address `n - 1 - i` of a reservation),
    /// each but those in `unused` written: (pages, their home, the reservation, its addresses in
    /// page order).
    unsafe fn pages_away(n: usize, unused: &[usize]) -> (Vec<usize>, usize, usize, Vec<*mut c_void>) {
        let mut count = n;
        let mut pages = vec![0usize; n];
        assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut count, pages.as_mut_ptr()), 1);
        let h = page_home(pages[0]).unwrap();
        let r = mem::VirtualAlloc(std::ptr::null(), n * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
        assert!(r != 0);
        let addrs: Vec<*mut c_void> = (0..n).map(|i| (r + (n - 1 - i) * PHYS_PAGE) as *mut c_void).collect();
        let mut ids: Vec<u64> = pages.iter().map(|&p| p as u64).collect();
        assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), n, ids.as_mut_ptr()), 1);
        for (i, &a) in addrs.iter().enumerate().filter(|(i, _)| !unused.contains(i)) {
            *(a as *mut u32) = i as u32 + 1;
            *((a as usize + PHYS_PAGE - 4) as *mut u32) = !(i as u32);
        }
        (pages, h, r, addrs)
    }

    unsafe fn check_home(pages: &mut [usize], h: usize, unused: &[usize]) {
        assert_eq!(MapTitlePhysicalPages(std::ptr::null_mut(), pages.len(), 0, 4, pages.as_mut_ptr()) as usize, h);
        for i in 0..pages.len() {
            let a = h + i * PHYS_PAGE;
            let want = if unused.contains(&i) { (0, 0) } else { (i as u32 + 1, !(i as u32)) };
            assert_eq!((*(a as *const u32), *((a + PHYS_PAGE - 4) as *const u32)), want, "page {i}");
        }
    }

    /// A scatter that unmaps many pages commits their neighbouring homes together, whatever the
    /// order of the addresses: a run per `HOMES_AHEAD` pages of the call and per gap (a page
    /// never used has nothing to bring home). An address given twice, or with no page, changes
    /// nothing; every page reads at home what was written to it.
    #[test]
    fn a_scatter_that_unmaps_commits_homes_in_runs() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let n = HOMES_AHEAD + 40;
            let unused = [100, 500];
            let (mut pages, h, r, mut addrs) = pages_away(n, &unused);
            assert!(lazy_home(h) && !committed_page(h));
            let free = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE);
            addrs.push(addrs[n - 1]);
            addrs.push(free);
            let mut none = vec![0u64; addrs.len()];
            let (runs, copies) = (HOME_RUNS.with(|c| c.get()), HOME_COPIES.with(|c| c.get()));
            assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), addrs.len(), none.as_mut_ptr()), 1);
            assert_eq!(HOME_COPIES.with(|c| c.get()) - copies, n - unused.len());
            // pages 0..1024 in three runs (two gaps), the other 40 in one
            assert_eq!(HOME_RUNS.with(|c| c.get()) - runs, 4);
            assert!(addrs.iter().all(|&a| !committed_page(a as usize)));
            check_home(&mut pages, h, &unused);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), n, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
            mem::VirtualFree(free, 0, mem::MEM_RELEASE);
        }
    }

    /// A home at which another page is mapped is not committed ahead: that page's unmapping in
    /// the same call decommits the address, and the home is asked again.
    #[test]
    fn a_home_another_page_is_mapped_at_is_not_committed_ahead() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let (mut pages, h, r, addrs) = pages_away(4, &[]);
            // page 0 goes to page 2's home
            let at = [(h + 2 * PHYS_PAGE) as *mut c_void];
            let mut one = [pages[0] as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 1, one.as_mut_ptr()), 1);
            let all = [at[0], addrs[1], addrs[2], addrs[3]];
            let mut none = [0u64; 4];
            let runs = HOME_RUNS.with(|c| c.get());
            assert_eq!(XMemMapPhysicalPagesScatter(all.as_ptr(), 4, none.as_mut_ptr()), 1);
            assert_eq!(HOME_RUNS.with(|c| c.get()) - runs, 2, "homes 0..2 and 3");
            check_home(&mut pages, h, &[]);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 4, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// A scatter refused part way gives back the homes committed ahead of the pages it did not
    /// reach; those stay mapped with what they hold.
    #[test]
    fn a_scatter_refused_part_way_gives_its_homes_back() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let (mut pages, h, r, addrs) = pages_away(4, &[]);
            let bad = (addrs[2] as usize + 0x1000) as *mut c_void;
            let all = [addrs[0], bad, addrs[1], addrs[2], addrs[3]];
            let mut none = [0u64; 5];
            assert_eq!(XMemMapPhysicalPagesScatter(all.as_ptr(), 5, none.as_mut_ptr()), 0);
            assert!(committed_page(h) && (1..4).all(|i| !committed_page(h + i * PHYS_PAGE)));
            assert!(!committed_page(addrs[0] as usize));
            assert!((1..4).all(|i| *(addrs[i] as *const u32) == i as u32 + 1));
            check_home(&mut pages, h, &[]);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 4, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// A page mapped away that the title maps somewhere else goes there straight, with what it
    /// holds, and not by way of its home; the address it left is given up.
    #[test]
    fn a_page_away_moves_on_without_going_home() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let mut n = 1usize;
            let mut pages = [0usize; 1];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let r = mem::VirtualAlloc(std::ptr::null(), 2 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let (a, b) = (r as *mut c_void, (r + PHYS_PAGE) as *mut c_void);
            let mut one = [pages[0] as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&a, 1, one.as_mut_ptr()), 1);
            *(a as *mut u32) = 7;
            *((r + PHYS_PAGE - 4) as *mut u32) = 8;
            let before = HOME_COPIES.with(|c| c.get());
            assert_eq!(XMemMapPhysicalPagesScatter(&b, 1, one.as_mut_ptr()), 1);
            assert_eq!(HOME_COPIES.with(|c| c.get()), before, "the page went by way of its home");
            assert_eq!(*(b as *const u32), 7);
            assert_eq!(*((r + 2 * PHYS_PAGE - 4) as *const u32), 8);
            assert!(!committed_page(r), "the address the page left stays committed");
            let mut zero = [0u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&b, 1, zero.as_mut_ptr()), 1);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// A move whose destination cannot be committed fails with the page still mapped away where
    /// it was: its bytes and both lookups are kept.
    #[test]
    fn a_page_whose_move_fails_stays_mapped_where_it_was() {
        unsafe {
            let old = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize;
            *(old as *mut u32) = 7;
            // an address with nothing reserved: committing there fails
            let at = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            mem::VirtualFree(at as *mut c_void, 0, mem::MEM_RELEASE);
            let page = 0x7E57_0000usize;
            let mut away = Away { at: Default::default(), page: Default::default(), gone: Vec::new(), ready: Default::default(), ..Default::default() };
            away.at.insert(old, page);
            away.page.insert(page, old);
            assert!(move_away(&mut away, old, at, page, false).is_err());
            assert_eq!(away.page.get(&page), Some(&old), "the page lost where it is");
            assert_eq!(away.at.get(&old), Some(&page));
            assert_eq!(*(old as *const u32), 7);
            mem::VirtualFree(old as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// The page the title mapped at `at` last (`Away::title`).
    fn title_at(at: usize) -> Option<usize> {
        AWAY.lock().unwrap().as_ref().and_then(|a| a.title.get(&at).copied())
    }

    /// An address keeps the page the title mapped there last, also once XCL moved the page's
    /// bytes to another address the title mapped it at; an unmap, another page, a free, a failed
    /// map and the release of the reservation each leave it right.
    #[test]
    fn an_address_keeps_the_page_the_title_mapped_there_last() {
        let null = std::ptr::null_mut();
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(null, 0, &mut n, pages.as_mut_ptr()), 1);
            let (p, q) = (pages[0], pages[1]);
            let h = page_home(p).unwrap();
            // a graphics reservation of the test's own
            let r = reserve_watched(4 * PHYS_PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            assert!(r != 0);
            let (r1, r2) = (r, r + PHYS_PAGE);
            let (mut one_p, mut one_q, mut none) = ([p], [q], [0usize]);
            assert_eq!(MapTitlePhysicalPages(r1 as *mut c_void, 1, 0, 4, one_p.as_mut_ptr()) as usize, r1);
            *(r1 as *mut u32) = 5;
            assert_eq!(title_at(r1), Some(p));
            // mapped at a second address: its bytes move there, the first address still names it
            assert_eq!(MapTitlePhysicalPages(r2 as *mut c_void, 1, 0, 4, one_p.as_mut_ptr()) as usize, r2);
            assert_eq!(*(r2 as *const u32), 5);
            assert!(AWAY.lock().unwrap().as_ref().is_some_and(|a| a.page.get(&p) == Some(&r2) && !a.at.contains_key(&r1)), "the bytes moved");
            assert_eq!((title_at(r1), title_at(r2)), (Some(p), Some(p)));
            // unmapped at the first
            assert_eq!(MapTitlePhysicalPages(r1 as *mut c_void, 1, 0, 4, none.as_mut_ptr()) as usize, r1);
            assert_eq!((title_at(r1), title_at(r2)), (None, Some(p)));
            // another page at the second: replaced (p goes home with its bytes)
            assert_eq!(MapTitlePhysicalPages(r2 as *mut c_void, 1, 0, 4, one_q.as_mut_ptr()) as usize, r2);
            assert_eq!(title_at(r2), Some(q));
            // a map that fails (nothing reserved there, and p's home holds bytes) leaves no entry
            let bad = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            mem::VirtualFree(bad as *mut c_void, 0, mem::MEM_RELEASE);
            assert!(MapTitlePhysicalPages(bad as *mut c_void, 1, 0, 4, one_p.as_mut_ptr()).is_null());
            assert_eq!(title_at(bad), None);
            // both at their home: their homes name them
            assert_eq!(MapTitlePhysicalPages(null, 2, 0, 4, pages.as_mut_ptr()) as usize, h);
            assert_eq!(*(h as *const u32), 5);
            assert_eq!((title_at(h), title_at(h + PHYS_PAGE)), (Some(p), Some(q)));
            // the release of the reservation forgets what was mapped in it
            assert_eq!(VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert_eq!((title_at(r1), title_at(r2)), (None, None));
            assert_eq!(title_at(h), Some(p), "outside the reservation");
            // freed: no address names them
            assert_eq!(FreeTitlePhysicalPages(null, 2, pages.as_mut_ptr()), 1);
            assert!(AWAY.lock().unwrap().as_ref().is_some_and(|a| a.title.values().all(|&v| v != p && v != q)));
        }
    }

    /// A batch of the queue's page-mapping method: the first destination, the number of ranges
    /// (`kind, start, count` each) and a pointer to them.
    #[repr(C)]
    struct Batch {
        dest: u64,
        ranges_n: u32,
        _pad: u32,
        ranges: usize,
    }
    fn batch(dest: usize, r: &[[u32; 3]]) -> Batch {
        Batch { dest: dest as u64, ranges_n: r.len() as u32, _pad: 0, ranges: r.as_ptr() as usize }
    }

    /// The page-mapping method's batches decode to (GPU-view page, pool page or unmapped) in the
    /// call's order, from the destination's page on; a call with anything wrong in it gives
    /// nothing at all.
    #[test]
    fn the_page_mapping_batches_are_decoded_whole() {
        use mirrors::{decode_batches, PAGE};
        let (d, d2, pool, pool_pages) = (0x7100_0004_0000usize, 0x7300_0000_0000usize, 0x7200_0000_0000usize, 64u32);
        unsafe {
            // two pool pages from 10 on, one page kept, pool page 5 three times, one unmapped;
            // then a second batch of two pages
            let r1 = [[0, 10, 2], [3, 0, 1], [1, 5, 3], [2, 0, 1]];
            let r2 = [[0, 0, 2]];
            let b = [batch(d + 0x1234, &r1), batch(d2, &r2)];
            let want = vec![
                (d, Some(pool + 10 * PAGE)),
                (d + PAGE, Some(pool + 11 * PAGE)),
                (d + 3 * PAGE, Some(pool + 5 * PAGE)),
                (d + 4 * PAGE, Some(pool + 5 * PAGE)),
                (d + 5 * PAGE, Some(pool + 5 * PAGE)),
                (d + 6 * PAGE, None),
                (d2, Some(pool)),
                (d2 + PAGE, Some(pool + PAGE)),
            ];
            assert_eq!(decode_batches(2, b.as_ptr() as usize, pool, pool_pages), Some(want));
            // a batch with no ranges gives nothing and refuses nothing
            let none = Batch { dest: d as u64, ranges_n: 0, _pad: 0, ranges: 0 };
            assert_eq!(decode_batches(1, [none].as_ptr() as usize, pool, pool_pages), Some(Vec::new()));
            assert_eq!(decode_batches(0, 0, pool, pool_pages), Some(Vec::new()), "no batches");

            // bad calls: a range leaving the pool (after a good one), an unknown kind, ranges and
            // batches that are not readable, more than MOST pages
            let leaves = [[0, 0, 1], [0, 63, 2]];
            assert_eq!(decode_batches(1, [batch(d, &leaves)].as_ptr() as usize, pool, pool_pages), None);
            let one_out = [[1, 64, 1]];
            assert_eq!(decode_batches(1, [batch(d, &one_out)].as_ptr() as usize, pool, pool_pages), None);
            let unknown = [[7, 0, 1]];
            assert_eq!(decode_batches(1, [batch(d, &unknown)].as_ptr() as usize, pool, pool_pages), None);
            let unreadable = Batch { dest: d as u64, ranges_n: 1, _pad: 0, ranges: 0 };
            assert_eq!(decode_batches(1, [unreadable].as_ptr() as usize, pool, pool_pages), None);
            assert_eq!(decode_batches(1, 0, pool, pool_pages), None, "batches not readable");
            let many = [[2, 0, (1 << 20) + 1]];
            assert_eq!(decode_batches(1, [batch(d, &many)].as_ptr() as usize, pool, pool_pages), None);
            assert_eq!(decode_batches((1 << 20) + 1, b.as_ptr() as usize, pool, pool_pages), None);
        }
    }

    /// GPU-view pages the page-mapping method names record the physical page the title had
    /// mapped at their pool page, where XCL keeps it now and whether a tile may show it there –
    /// also when the title mapped it elsewhere before the call; a pool page with no page, and an
    /// unmapped page, show nothing; a page not named shows the CPU view. The same mapping named
    /// again changes nothing, a bad call records nothing, the views' release drops their pages.
    #[test]
    fn gpu_view_pages_record_the_physical_page_mapped_into_them() {
        use mirrors::{apply_batches, resolve, shown_at, shows, touched, Mapped, Shows, PAGE};
        const SIZE: usize = 199 * PAGE;
        let g = 0x7270_0000_0000usize;
        let null = std::ptr::null_mut();
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let mut n = 3usize;
            let mut pages = [0usize; 3];
            assert_eq!(AllocateTitlePhysicalPages(null, 0, &mut n, pages.as_mut_ptr()), 1);
            let [a, b, c] = pages;
            let cv = cpu_view(SIZE, g);
            // the pool, a graphics reservation: page a used at pool page 1, b never used at 2,
            // nothing at 0; c mapped at pool page 3, then kept in plain memory
            let pool = reserve_watched(4 * PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            let plain = mem::VirtualAlloc(std::ptr::null(), PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert!(pool != 0 && plain != 0);
            for (at, page) in [(pool + PAGE, a), (pool + 2 * PAGE, b), (pool + 3 * PAGE, c), (plain, c)] {
                let mut one = [page];
                assert_eq!(MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, at);
            }
            *((pool + PAGE) as *mut u32) = 0xA;

            // pool pages 0 to 2 at view pages 2 to 4, page 5 left as it is, 6 unmapped; and two
            // pages outside any view
            let r1 = [[0, 0, 3], [3, 0, 1], [2, 0, 1]];
            let r2 = [[1, 1, 2]];
            let bs = [batch(g + 2 * PAGE + 0x10, &r1), batch(pool, &r2)];
            assert_eq!(apply_batches(2, bs.as_ptr() as usize, pool, 4), Some(Mapped { mapped: 2, unmapped: 1, empty: 1, elsewhere: 2 }));
            assert_eq!(shows(g + 8), Shows::Same(cv + 8), "never named");
            assert_eq!(shows(g + 2 * PAGE), Shows::Nothing, "no page at pool page 0");
            assert_eq!(shows(g + 3 * PAGE + 8), Shows::Page { at: pool + PAGE + 8, tile: true });
            assert_eq!(shows(g + 4 * PAGE), Shows::Page { at: pool + 2 * PAGE, tile: true }, "a page never used is kept where it was mapped");
            assert_eq!(shows(g + 5 * PAGE + 8), Shows::Same(cv + 5 * PAGE + 8), "kind 3 leaves it");
            assert_eq!(shows(g + 6 * PAGE), Shows::Nothing, "unmapped");
            assert_eq!(shows(pool), Shows::NoView);
            assert!(shown_at(pool).is_none() && shown_at(pool + PAGE).is_none(), "outside any view: not recorded");
            assert_eq!(shown_at(g + 3 * PAGE).map(|e| e.page), Some(a));
            // one address a page, view pages 1 to 6
            let mut at = Vec::new();
            assert_eq!(resolve(g + PAGE + 0x10, 5 * PAGE, &mut at), Some([2, 0, 2, 2]));
            assert_eq!(at, vec![(cv + PAGE) | 1, 0, pool + PAGE, pool + 2 * PAGE, (cv + 5 * PAGE) | 1, 0]);
            assert_eq!(resolve(g + 20 * PAGE, PAGE, &mut at), None, "no page named");
            assert!(at.is_empty());
            assert!(touched(g + 3 * PAGE + 0x100, 4) && touched(g, 3 * PAGE));
            assert!(!touched(g, 2 * PAGE) && !touched(g + 7 * PAGE, 30 * PAGE));

            // the title maps a into the CPU view (its bytes move there); a call made after that
            // still finds a by its pool address, kept at the CPU view's same offset
            let mut one = [a];
            assert_eq!(MapTitlePhysicalPages((cv + 7 * PAGE) as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, cv + 7 * PAGE);
            assert_eq!(*((cv + 7 * PAGE) as *const u32), 0xA);
            let r = [[0, 1, 1]];
            assert_eq!(apply_batches(1, [batch(g + 7 * PAGE, &r)].as_ptr() as usize, pool, 4), Some(Mapped { mapped: 1, ..Mapped::default() }));
            assert_eq!(shows(g + 7 * PAGE + 4), Shows::Page { at: cv + 7 * PAGE + 4, tile: true });
            assert_eq!(resolve(g + 7 * PAGE, PAGE, &mut at), Some([0, 1, 0, 0]));
            assert_eq!(at, vec![cv + 7 * PAGE]);
            // the same mapping named again changes nothing
            let before = shown_at(g + 7 * PAGE);
            assert_eq!(apply_batches(1, [batch(g + 7 * PAGE, &r)].as_ptr() as usize, pool, 4), Some(Mapped { mapped: 1, ..Mapped::default() }));
            assert_eq!(shown_at(g + 7 * PAGE), before, "neither its call nor its place in the log");
            // a bad call records nothing
            let bad = [[0, 1, 1], [0, 3, 2]];
            assert_eq!(apply_batches(1, [batch(g + 8 * PAGE, &bad)].as_ptr() as usize, pool, 4), None);
            assert_eq!(shows(g + 8 * PAGE), Shows::Same(cv + 8 * PAGE));
            // a page kept outside graphics memory: no tile may show it there
            let r = [[1, 3, 1]];
            assert_eq!(apply_batches(1, [batch(g + 9 * PAGE, &r)].as_ptr() as usize, pool, 4), Some(Mapped { mapped: 1, ..Mapped::default() }));
            assert_eq!(shows(g + 9 * PAGE), Shows::Page { at: plain, tile: false });
            // remapped to pool page 1: a, with a later call
            let call = shown_at(g + 9 * PAGE).unwrap().call;
            let r = [[0, 1, 1]];
            assert_eq!(apply_batches(1, [batch(g + 9 * PAGE, &r)].as_ptr() as usize, pool, 4), Some(Mapped { mapped: 1, ..Mapped::default() }));
            assert!(shown_at(g + 9 * PAGE).is_some_and(|e| e.page == a && e.call > call));
            assert_eq!(shows(g + 9 * PAGE), Shows::Page { at: cv + 7 * PAGE, tile: true });

            // the pages unmapped and freed, then the views released: their pages go with them
            let mut none = [0usize];
            for at in [pool + 2 * PAGE, plain, cv + 7 * PAGE] {
                assert_eq!(MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, none.as_mut_ptr()) as usize, at);
            }
            assert_eq!(FreeTitlePhysicalPages(null, 3, pages.as_mut_ptr()), 1);
            assert_eq!(VirtualFree(cv as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert_eq!(shows(g + 3 * PAGE), Shows::NoView);
            assert!((0..10).all(|k| shown_at(g + k * PAGE).is_none()), "released with its view");
            assert!(!touched(g, SIZE));
            assert_eq!(VirtualFree(pool as *mut c_void, 0, mem::MEM_RELEASE), 1);
            mem::VirtualFree(plain as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// A GPU-view page follows its physical page where XCL keeps it: mapped elsewhere (its bytes
    /// move there), unmapped (home, with its bytes), mapped at the CPU view's same offset, and a
    /// page never used mapped elsewhere (a place not committed yet). A move is no change of what
    /// the page shows; a freed page shows nothing.
    #[test]
    fn a_gpu_view_page_follows_its_physical_page() {
        use mirrors::{apply_batches, resolve, shown_at, shows, Shown, Shows, PAGE};
        const SIZE: usize = 137 * PAGE;
        let g = 0x7280_0000_0000usize;
        let null = std::ptr::null_mut();
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(null, 0, &mut n, pages.as_mut_ptr()), 1);
            let [a, b] = pages;
            let (home_a, home_b) = (page_home(a).unwrap(), page_home(b).unwrap());
            let cv = cpu_view(SIZE, g);
            // the pool and another address, graphics reservations both
            let pool = reserve_watched(2 * PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            let r = reserve_watched(2 * PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            assert!(pool != 0 && r != 0);
            for (at, page) in [(pool, a), (pool + PAGE, b)] {
                let mut one = [page];
                assert_eq!(MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, at);
            }
            *((pool + 0x10) as *mut u32) = 0xA;
            // a at view pages 2 and 5, b (never used) at 3
            let (r1, r2) = ([[0, 0, 2]], [[0, 0, 1]]);
            let bs = [batch(g + 2 * PAGE, &r1), batch(g + 5 * PAGE, &r2)];
            assert!(apply_batches(2, bs.as_ptr() as usize, pool, 2).is_some());
            assert_eq!(shows(g + 2 * PAGE), Shows::Page { at: pool, tile: true });
            let first = shown_at(g + 2 * PAGE).unwrap();

            // mapped elsewhere: its bytes move, both view pages follow, what they show is unchanged
            let mut one = [a];
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r);
            assert_eq!(*((r + 0x10) as *const u32), 0xA);
            for k in [2, 5] {
                assert_eq!(shows(g + k * PAGE + 0x10), Shows::Page { at: r + 0x10, tile: true });
            }
            assert!(shown_at(g + 2 * PAGE).is_some_and(|e| e.page == a && e.call == first.call && e.seq == first.seq), "a move is no change");
            // unmapped there: home, with its bytes
            let mut none = [0usize];
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, none.as_mut_ptr()) as usize, r);
            assert_eq!(shows(g + 5 * PAGE), Shows::Page { at: home_a, tile: true });
            assert_eq!(*((home_a + 0x10) as *const u32), 0xA);
            // mapped at the CPU view's offset of view page 2: kept at the same offset there, and
            // elsewhere for view page 5
            assert_eq!(MapTitlePhysicalPages((cv + 2 * PAGE) as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, cv + 2 * PAGE);
            assert_eq!(*((cv + 2 * PAGE + 0x10) as *const u32), 0xA);
            let mut at = Vec::new();
            assert_eq!(resolve(g + 2 * PAGE, PAGE, &mut at), Some([0, 1, 0, 0]));
            assert_eq!(resolve(g + 5 * PAGE, PAGE, &mut at), Some([1, 0, 0, 0]));
            assert_eq!(at, vec![cv + 2 * PAGE]);
            // b, never used, mapped elsewhere: its place follows, not committed (unless its home
            // is committed memory: then its bytes were copied)
            let mut one = [b];
            assert_eq!(MapTitlePhysicalPages((r + PAGE) as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r + PAGE);
            assert_eq!(shows(g + 3 * PAGE), Shows::Page { at: r + PAGE, tile: true });
            assert_eq!(committed_page(r + PAGE), !lazy_home(home_b));
            assert_eq!(first, shown_at(g + 2 * PAGE).map(|e| Shown { at: first.at, tile: first.tile, ..e }).unwrap(), "only the place changed");

            // freed (unmapped first): what the view pages showed is gone, a change of what they show
            for at in [cv + 2 * PAGE, r + PAGE] {
                assert_eq!(MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, none.as_mut_ptr()) as usize, at);
            }
            assert_eq!(FreeTitlePhysicalPages(null, 2, pages.as_mut_ptr()), 1);
            for k in [2, 3, 5] {
                assert_eq!(shows(g + k * PAGE), Shows::Nothing);
            }
            assert!(shown_at(g + 2 * PAGE).is_some_and(|e| e.page == 0 && e.at == 0 && e.seq > first.seq));
            assert_eq!(VirtualFree(cv as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert!((0..8).all(|k| shown_at(g + k * PAGE).is_none()));
            for q in [pool, r] {
                assert_eq!(VirtualFree(q as *mut c_void, 0, mem::MEM_RELEASE), 1);
            }
        }
    }

    /// When a shown page moves, the tiles of its GPU-view pages follow it from a place a tile
    /// could show to another such place (`sparse::moved`, from the place it left). They are linked
    /// again when it goes to plain memory (no tile may show it there), and when it comes back
    /// from there: no tile was linked over plain memory, so none can follow it.
    #[test]
    fn a_page_leaving_plain_memory_has_the_tiles_over_it_linked_again() {
        use mirrors::{apply_batches, shows, Shows, PAGE};
        const SIZE: usize = 223 * PAGE;
        let g = 0x72D0_0000_0000usize;
        let null = std::ptr::null_mut();
        let placed = || PLACED.with(|s| s.take());
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let mut n = 1usize;
            let mut pages = [0usize; 1];
            assert_eq!(AllocateTitlePhysicalPages(null, 0, &mut n, pages.as_mut_ptr()), 1);
            let a = pages[0];
            let home = page_home(a).unwrap();
            let cv = cpu_view(SIZE, g);
            // the pool and another address, graphics reservations both; and plain memory
            let pool = reserve_watched(PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            let r = reserve_watched(2 * PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            let plain = mem::VirtualAlloc(std::ptr::null(), PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            assert!(pool != 0 && r != 0 && plain != 0);
            let (mut one, mut none) = ([a], [0usize]);
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, pool);
            *(pool as *mut u32) = 0xA;
            // a at view pages 1 and 2
            let ranges = [[1, 0, 2]];
            assert!(apply_batches(1, [batch(g + PAGE, &ranges)].as_ptr() as usize, pool, 1).is_some());
            assert_eq!(shows(g + PAGE), Shows::Page { at: pool, tile: true });
            let views = vec![g + PAGE, g + 2 * PAGE];
            placed();

            // mapped in plain memory: linked again (to nothing there)
            assert_eq!(MapTitlePhysicalPages(plain as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, plain);
            assert_eq!(shows(g + 2 * PAGE), Shows::Page { at: plain, tile: false });
            assert_eq!(placed(), vec![(a, 0, plain, vec![], views.clone())]);
            // from there to a graphics reservation: linked again, to the new place
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r);
            assert_eq!(*(r as *const u32), 0xA);
            assert_eq!(shows(g + PAGE), Shows::Page { at: r, tile: true });
            assert_eq!(placed(), vec![(a, 0, r, vec![], views.clone())]);
            // from there to another: they follow it from the place it left
            assert_eq!(MapTitlePhysicalPages((r + PAGE) as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r + PAGE);
            assert_eq!(placed(), vec![(a, r, r + PAGE, views.clone(), vec![])]);
            // unmapped: home, with its bytes; they follow it there
            assert_eq!(MapTitlePhysicalPages((r + PAGE) as *mut c_void, 1, 0, 4, none.as_mut_ptr()) as usize, r + PAGE);
            assert_eq!(shows(g + PAGE), Shows::Page { at: home, tile: true });
            assert_eq!(placed(), vec![(a, r + PAGE, home, views.clone(), vec![])]);

            assert_eq!(FreeTitlePhysicalPages(null, 1, pages.as_mut_ptr()), 1);
            for q in [cv, pool, r] {
                assert_eq!(VirtualFree(q as *mut c_void, 0, mem::MEM_RELEASE), 1);
            }
            mem::VirtualFree(plain as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// The title's release of a block of unified memory (MEM_GRAPHICS) that holds the place of a
    /// shown page is told to sparse as the whole block; the page shows nothing from then on and
    /// the title's mapping there is forgotten – before the block can be handed out again.
    #[test]
    fn the_release_of_unified_memory_under_a_shown_page_leaves_it_showing_nothing() {
        use mirrors::{apply_batches, shows, Shows, PAGE};
        const SIZE: usize = 227 * PAGE;
        let g = 0x72E0_0000_0000usize;
        let null = std::ptr::null_mut();
        let kept = || KEPT_GOING.with(|s| s.take());
        unsafe {
            let Some(u) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let Some(block) = u.alloc(2 * PAGE, PAGE).map(|p| p as usize) else { return crate::gpu::recomp::runner::skip_test("no room in unified memory") };
            let mut n = 1usize;
            let mut pages = [0usize; 1];
            assert_eq!(AllocateTitlePhysicalPages(null, 0, &mut n, pages.as_mut_ptr()), 1);
            let cv = cpu_view(SIZE, g);
            // the page mapped in the block, at view page 1
            let mut one = [pages[0]];
            assert_eq!(MapTitlePhysicalPages((block + PAGE) as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, block + PAGE);
            let ranges = [[0, 1, 1]];
            assert!(apply_batches(1, [batch(g + PAGE, &ranges)].as_ptr() as usize, block, 2).is_some());
            assert_eq!(shows(g + PAGE), Shows::Page { at: block + PAGE, tile: true });
            kept();

            assert_eq!(VirtualFree(block as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert_eq!(kept(), vec![(block, 2 * PAGE, true)], "the whole block, still committed");
            assert_eq!(shows(g + PAGE), Shows::Nothing);
            assert!(AWAY.lock().unwrap().as_ref().is_some_and(|a| !a.title.contains_key(&(block + PAGE))), "the title's mapping there is forgotten");

            // (freed while mapped there: the address keeps its bytes)
            assert_eq!(FreeTitlePhysicalPages(null, 1, pages.as_mut_ptr()), 1);
            assert_eq!(VirtualFree(cv as *mut c_void, 0, mem::MEM_RELEASE), 1);
        }
    }

    /// The changed pages of a GPU view are told in the order of their last change, each once,
    /// with the call that changed it last; a move is no change; `most` hands them out one by one
    /// and the range keeps other pages out; the pages of a released view are not told.
    #[test]
    fn changed_gpu_view_pages_are_told_once_in_order() {
        use mirrors::{apply_batches, changes_since, shown_at, shows, Shows, PAGE};
        const SIZE: usize = 139 * PAGE;
        let g = 0x7290_0000_0000usize;
        let (lo, hi, all) = (g, g + SIZE, usize::MAX);
        let null = std::ptr::null_mut();
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let mut n = 3usize;
            let mut pages = [0usize; 3];
            assert_eq!(AllocateTitlePhysicalPages(null, 0, &mut n, pages.as_mut_ptr()), 1);
            let cv = cpu_view(SIZE, g);
            let pool = reserve_watched(3 * PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            let r = reserve_watched(PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            assert!(pool != 0 && r != 0);
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 3, 0, 4, pages.as_mut_ptr()) as usize, pool);
            let apply = |ranges: &[[u32; 3]], at: usize| assert!(apply_batches(1, [batch(at, ranges)].as_ptr() as usize, pool, 3).is_some());
            let call = |k: usize| shown_at(g + k * PAGE).unwrap().call;
            let (start, got) = changes_since(0, lo, hi, all);
            assert!(got.is_empty(), "a view of the test's own");

            // one call of three pages
            apply(&[[0, 0, 3]], g + PAGE);
            let c1 = call(1);
            let (s1, got) = changes_since(start, lo, hi, all);
            assert_eq!(got, vec![(g + PAGE, c1), (g + 2 * PAGE, c1), (g + 3 * PAGE, c1)]);
            assert!(changes_since(s1, lo, hi, all).1.is_empty(), "told once");
            // remapped: that page comes again, last, with its new call
            apply(&[[0, 2, 1]], g + PAGE);
            assert!(call(1) > c1);
            let (s2, got) = changes_since(s1, lo, hi, all);
            assert_eq!(got, vec![(g + PAGE, call(1))]);
            // a move is no change: the page of view page 2 mapped elsewhere
            let mut one = [pages[1]];
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r);
            assert_eq!(shows(g + 2 * PAGE), Shows::Page { at: r, tile: true });
            let (s3, got) = changes_since(s2, lo, hi, all);
            assert!(got.is_empty(), "a move");
            // view page 2 unmapped, then 3, then 2 mapped again: 3 and 2, once each
            apply(&[[2, 0, 1]], g + 2 * PAGE);
            apply(&[[2, 0, 1]], g + 3 * PAGE);
            apply(&[[0, 1, 1]], g + 2 * PAGE);
            assert_eq!(shows(g + 2 * PAGE), Shows::Page { at: r, tile: true });
            let want = vec![(g + 3 * PAGE, call(3)), (g + 2 * PAGE, call(2))];
            assert!(call(2) > call(3));
            assert_eq!(changes_since(s3, lo, hi, all).1, want);
            // one at a time
            let (t1, got) = changes_since(s3, lo, hi, 1);
            assert_eq!(got, want[..1]);
            let (t2, got) = changes_since(t1, lo, hi, 1);
            assert_eq!(got, want[1..]);
            assert!(changes_since(t2, lo, hi, 1).1.is_empty());
            // the range: view page 3 alone
            assert_eq!(changes_since(s3, g + 3 * PAGE, g + 4 * PAGE, all).1, want[..1]);

            // released with the view: none of them is told
            assert_eq!(VirtualFree(cv as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert!(changes_since(start, lo, hi, all).1.is_empty());
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 3, 0, 4, std::ptr::null_mut()) as usize, pool);
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, std::ptr::null_mut()) as usize, r);
            assert_eq!(FreeTitlePhysicalPages(null, 3, pages.as_mut_ptr()), 1);
            for q in [pool, r] {
                assert_eq!(VirtualFree(q as *mut c_void, 0, mem::MEM_RELEASE), 1);
            }
        }
    }

    /// A write where a mapped page is kept reaches every GPU-view page that shows it, and none
    /// once the page left that address; a write to the CPU view reaches the GPU-view page at the
    /// same offset only while the page-mapping method never named that page. The 16 MiB blocks
    /// of the places follow.
    #[test]
    fn a_write_where_a_page_is_kept_is_a_write_to_the_views_showing_it() {
        use mirrors::{apply_batches, shown_blocks, shown_gen, views_of_written, PAGE};
        const SIZE: usize = 149 * PAGE;
        const BLOCK: usize = 1 << 24;
        let g = 0x72A0_0000_0000usize;
        let null = std::ptr::null_mut();
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let mut n = 1usize;
            let mut pages = [0usize; 1];
            assert_eq!(AllocateTitlePhysicalPages(null, 0, &mut n, pages.as_mut_ptr()), 1);
            let cv = cpu_view(SIZE, g);
            // the pool in a 16 MiB block of the test's own
            let space = reserve_watched(2 * BLOCK, 0x2000, mem::PAGE_READWRITE) as usize;
            let r = reserve_watched(PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            assert!(space != 0 && r != 0);
            let pool = (space + BLOCK - 1) & !(BLOCK - 1);
            assert_eq!(MapTitlePhysicalPages(pool as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, pool);
            *(pool as *mut u32) = 1;
            let gen = shown_gen();
            // the page at view pages 1 and 4
            let ranges = [[0, 0, 1]];
            let bs = [batch(g + PAGE, &ranges), batch(g + 4 * PAGE, &ranges)];
            assert!(apply_batches(2, bs.as_ptr() as usize, pool, 1).is_some());
            assert!(shown_blocks().contains(&(pool, pool + BLOCK)) && shown_gen() > gen);
            let twins = |dirty: &[usize]| {
                let mut v = Vec::new();
                views_of_written(dirty, &mut v);
                v.sort_unstable();
                v
            };
            assert_eq!(twins(&[pool + 0x1000, pool + 0x3000]), vec![g + PAGE + 0x1000, g + PAGE + 0x3000, g + 4 * PAGE + 0x1000, g + 4 * PAGE + 0x3000]);
            // moved: the address it left reaches none of them, its new place both
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, r);
            assert!(twins(&[pool + 0x1000]).is_empty());
            assert_eq!(twins(&[r + 0x2000]), vec![g + PAGE + 0x2000, g + 4 * PAGE + 0x2000]);
            let rb = r & !(BLOCK - 1);
            assert!(!shown_blocks().contains(&(pool, pool + BLOCK)) && shown_blocks().contains(&(rb, rb + BLOCK)));
            // the CPU view under a page never named reaches it; under a mapped or unmapped one, not
            assert_eq!(twins(&[cv + 6 * PAGE + 0x2000, cv + 6 * PAGE + 0x5000]), vec![g + 6 * PAGE + 0x2000, g + 6 * PAGE + 0x5000]);
            assert!(twins(&[cv + PAGE + 0x2000]).is_empty(), "mapped");
            assert!(apply_batches(1, [batch(g + 6 * PAGE, &[[2, 0, 1]])].as_ptr() as usize, pool, 1).is_some());
            assert!(twins(&[cv + 6 * PAGE + 0x2000]).is_empty(), "unmapped");

            assert_eq!(VirtualFree(cv as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert!(twins(&[r + 0x2000]).is_empty(), "released with the view");
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, std::ptr::null_mut()) as usize, r);
            assert_eq!(FreeTitlePhysicalPages(null, 1, pages.as_mut_ptr()), 1);
            for q in [space, r] {
                assert_eq!(VirtualFree(q as *mut c_void, 0, mem::MEM_RELEASE), 1);
            }
        }
    }

    /// While GPU views show mapped pages, memory of no pair is told to sparse before it goes (a
    /// tile may show a page kept there): the address a page leaves, while still committed; the
    /// home given back once the page was mapped away from it again; the release of the block's
    /// home. The release of the pool leaves the page kept there showing nothing.
    #[test]
    fn memory_under_a_shown_page_is_told_to_sparse_before_it_goes() {
        use mirrors::{apply_batches, shows, Shows, PAGE};
        const SIZE: usize = 151 * PAGE;
        let g = 0x72B0_0000_0000usize;
        let null = std::ptr::null_mut();
        let kept = || KEPT_GOING.with(|s| s.take());
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(null, 0, &mut n, pages.as_mut_ptr()), 1);
            let [a, b] = pages;
            let home = page_home(a).unwrap();
            let cv = cpu_view(SIZE, g);
            let pool = reserve_watched(2 * PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            let r = reserve_watched(2 * PAGE, 0x2000, mem::PAGE_READWRITE) as usize;
            assert!(pool != 0 && r != 0);
            for (at, page) in [(pool, a), (pool + PAGE, b)] {
                let mut one = [page];
                assert_eq!(MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, at);
                *(at as *mut u32) = page as u32;
            }
            // a at view page 1, b at 2
            let ranges = [[0, 0, 2]];
            assert!(apply_batches(1, [batch(g + PAGE, &ranges)].as_ptr() as usize, pool, 2).is_some());
            assert_eq!(shows(g + PAGE), Shows::Page { at: pool, tile: true });
            SOURCES_GOING.with(|s| s.borrow_mut().clear());
            kept();

            // a mapped elsewhere: the address it leaves is told while still committed
            let (mut one, mut none) = ([a], [0usize]);
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r);
            assert_eq!(kept(), vec![(pool, PAGE, true)]);
            assert!(!committed_page(pool));
            // unmapped there (home), then mapped elsewhere again: the home is given back, told while
            // still committed (a home in committed memory stays)
            assert_eq!(MapTitlePhysicalPages(r as *mut c_void, 1, 0, 4, none.as_mut_ptr()) as usize, r);
            assert_eq!(kept(), vec![(r, PAGE, true)]);
            assert_eq!(shows(g + PAGE), Shows::Page { at: home, tile: true });
            assert_eq!(MapTitlePhysicalPages((r + PAGE) as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, r + PAGE);
            assert_eq!(*((r + PAGE) as *const u32), a as u32);
            assert_eq!(kept(), if lazy_home(home) { vec![(home, PAGE, true)] } else { Vec::new() });
            assert_eq!(shows(g + PAGE), Shows::Page { at: r + PAGE, tile: true });
            assert!(SOURCES_GOING.with(|s| s.take()).is_empty(), "no pair");

            // the pool released: b, kept there, shows nothing; a is kept elsewhere
            assert_eq!(VirtualFree(pool as *mut c_void, 0, mem::MEM_RELEASE), 1);
            assert_eq!(kept(), vec![(pool, 2 * PAGE, false)]);
            assert_eq!(shows(g + 2 * PAGE), Shows::Nothing);
            assert_eq!(shows(g + PAGE), Shows::Page { at: r + PAGE, tile: true });
            // a home again, then both freed: the release of their home block is told
            assert_eq!(MapTitlePhysicalPages((r + PAGE) as *mut c_void, 1, 0, 4, none.as_mut_ptr()) as usize, r + PAGE);
            kept();
            assert_eq!(FreeTitlePhysicalPages(null, 2, pages.as_mut_ptr()), 1);
            assert_eq!(kept(), vec![(home, 2 * PAGE, true)]);
            assert_eq!(shows(g + PAGE), Shows::Nothing);
            for q in [cv, r] {
                assert_eq!(VirtualFree(q as *mut c_void, 0, mem::MEM_RELEASE), 1);
            }
        }
    }

    /// A page mapped at another page's home, then the whole allocation mapped back home: the
    /// first page goes home and leaves that address committed, as the second one's home.
    #[test]
    fn a_page_away_at_another_home_goes_home_with_it() {
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let (q, p) = (pages[0], pages[1]);
            let h = page_home(p).unwrap();
            let r = mem::VirtualAlloc(std::ptr::null(), PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
            let s = r as *mut c_void;
            let mut one = [p as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&s, 1, one.as_mut_ptr()), 1);
            *(s as *mut u32) = 7;
            let at = h as *mut c_void;
            let mut one = [q as u64];
            assert_eq!(XMemMapPhysicalPagesScatter(&at, 1, one.as_mut_ptr()), 1);
            *(h as *mut u32) = 9;
            let home = MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, pages.as_mut_ptr()) as usize;
            assert_eq!(home, page_home(q).unwrap());
            assert_eq!(*(home as *const u32), 9);
            assert_eq!(*(h as *const u32), 7);
            assert!(!committed_page(r));
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }

    /// Timing (`--ignored --nocapture`; `XCL_SCATTER_PAGES=4096,8192`): pages written at their
    /// home, scattered from there into a reservation in one call, scattered again in reverse,
    /// and mapped back home.
    #[test]
    #[ignore]
    fn scatter_timing() {
        let sizes: Vec<usize> = std::env::var("XCL_SCATTER_PAGES").ok().map_or(vec![4096, 8192, 16384], |v| v.split(',').map(|s| s.parse().unwrap()).collect());
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        for n in sizes {
            unsafe {
                let mut cnt = n;
                let mut pages = vec![0usize; n];
                assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut cnt, pages.as_mut_ptr()), 1);
                let h = MapTitlePhysicalPages(std::ptr::null_mut(), n, 0, 4, pages.as_mut_ptr()) as usize;
                assert!(h != 0);
                for i in 0..n {
                    *((h + i * PHYS_PAGE) as *mut u32) = i as u32 + 1;
                }
                let r = mem::VirtualAlloc(std::ptr::null(), n * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) as usize;
                let addrs: Vec<*mut c_void> = (0..n).map(|i| (r + i * PHYS_PAGE) as *mut c_void).collect();
                let mut ids: Vec<u64> = pages.iter().map(|&p| p as u64).collect();
                let t = std::time::Instant::now();
                assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), n, ids.as_mut_ptr()), 1);
                let from_home = ms(t);
                ids.reverse();
                let t = std::time::Instant::now();
                assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), n, ids.as_mut_ptr()), 1);
                let reverse = ms(t);
                for i in (0..n).step_by(97) {
                    assert_eq!(*((r + i * PHYS_PAGE) as *const u32), (n - i) as u32);
                }
                let t = std::time::Instant::now();
                assert_eq!(MapTitlePhysicalPages(std::ptr::null_mut(), n, 0, 4, pages.as_mut_ptr()) as usize, h);
                let back = ms(t);
                for i in (0..n).step_by(97) {
                    assert_eq!(*((h + i * PHYS_PAGE) as *const u32), i as u32 + 1);
                }
                eprintln!("scatter_timing pages={n} from_home={from_home:.1}ms reverse={reverse:.1}ms back_home={back:.1}ms");
                FreeTitlePhysicalPages(std::ptr::null_mut(), n, pages.as_mut_ptr());
                mem::VirtualFree(r as *mut c_void, 0, mem::MEM_RELEASE);
            }
        }
    }

    #[test]
    fn title_memory_status_reports_budget() {
        let mut s: TitleMemoryStatusT = unsafe { std::mem::zeroed() };
        s.dw_length = std::mem::size_of::<TitleMemoryStatusT>() as u32;
        assert_eq!(std::mem::size_of::<TitleMemoryStatusT>(), 80);
        assert_eq!(unsafe { TitleMemoryStatus(&mut s) }, 1);
        assert_eq!(s.total, 13_824 * 1024 * 1024);
        assert!(s.avail <= s.total);
        // the process's own memory counts as used (Wine leaves PrivateUsage 0)
        assert!(process_private_bytes() > 0 && s.avail < s.total);
        s.dw_length = 8;
        assert_eq!(unsafe { TitleMemoryStatus(&mut s) }, 0);
    }

    #[test]
    fn xmem_is_tools_mem_available_returns_false() {
        assert_eq!(unsafe { XMemIsToolsMemAvailable() }, 0);
    }

    #[test]
    fn get_proc_address_and_load_library_ex_reject_null_name() {
        unsafe {
            let k32 = ll::GetModuleHandleW(crate::util::wide("kernel32.dll").as_ptr()) as usize;
            assert_ne!(k32, 0);
            fnd::SetLastError(0);
            assert!(ll::GetProcAddress(k32 as _, std::ptr::null()).is_none());
            let win32_err = fnd::GetLastError();

            fnd::SetLastError(0);
            assert_eq!(GetProcAddress(k32, std::ptr::null()), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);
            // Windows' own answer; Wine's GetProcAddress says ERROR_PROC_NOT_FOUND instead
            if !crate::util::under_wine() {
                assert_eq!(win32_err, fnd::ERROR_INVALID_PARAMETER);
            }

            // Non-zero ordinal lookup still resolves when valid (ordinal #1 on kernel32).
            assert_ne!(GetProcAddress(k32, 1usize as *const u8), 0);

            fnd::SetLastError(0);
            assert_eq!(LoadLibraryExW(std::ptr::null(), std::ptr::null_mut(), LOAD_AS_DATA), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

            fnd::SetLastError(0);
            assert_eq!(LoadLibraryExA(std::ptr::null(), std::ptr::null_mut(), LOAD_AS_DATA), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);
        }
    }

    #[test]
    fn get_thread_name_sizing_truncation_and_errors() {
        std::thread::spawn(|| unsafe {
            let cur = thr::GetCurrentThread();

            // Empty thread name: capacity 0 fails with ERROR_INSUFFICIENT_BUFFER, capacity 1 writes NUL.
            let empty = crate::util::wide("");
            assert_eq!(SetThreadName(cur, empty.as_ptr()), 1);
            let mut ret = 99usize;
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(cur, std::ptr::null_mut(), 0, &mut ret), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(ret, 0);
            let mut nul_only = [0xBEEFu16; 1];
            assert_eq!(GetThreadName(cur, nul_only.as_mut_ptr(), 1, &mut ret), 1);
            assert_eq!((ret, nul_only[0]), (0, 0));

            let want = crate::util::wide("xcl-worker-1");
            assert_eq!(SetThreadName(cur, want.as_ptr()), 1);

            // Null buffer with non-zero length -> ERROR_INVALID_PARAMETER.
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(cur, std::ptr::null_mut(), 4, &mut ret), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

            // Size query (null buffer, len == 0) -> FALSE + ERROR_INSUFFICIENT_BUFFER with ret_len = 12.
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(cur, std::ptr::null_mut(), 0, &mut ret), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(ret, 12);

            // Buffer equal to char count (no room for NUL terminator) -> FALSE + ERROR_INSUFFICIENT_BUFFER.
            let mut exact = [0xBEEFu16; 12];
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(cur, exact.as_mut_ptr(), exact.len(), &mut ret), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(ret, 12);

            // Buffer with room for chars + NUL -> TRUE and exact match (both with and without ret_len).
            let mut full = [0u16; 13];
            assert_eq!(GetThreadName(cur, full.as_mut_ptr(), full.len(), &mut ret), 1);
            assert_eq!(ret, 12);
            assert_eq!(&full[..], &want[..]);
            full.fill(0);
            assert_eq!(GetThreadName(cur, full.as_mut_ptr(), full.len(), std::ptr::null_mut()), 1);
            assert_eq!(&full[..], &want[..]);

            // Invalid thread handle -> FALSE + Win32 error set.
            fnd::SetLastError(0);
            assert_eq!(GetThreadName(std::ptr::null_mut(), full.as_mut_ptr(), full.len(), &mut ret), 0);
            assert_ne!(fnd::GetLastError(), 0);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn xmem_virtual_query_excludes_guard_pages() {
        unsafe {
            let proc_ = thr::GetCurrentProcess();
            let page = mem::VirtualAlloc(std::ptr::null(), 0x1000, mem::MEM_COMMIT | mem::MEM_RESERVE, mem::PAGE_READWRITE | mem::PAGE_GUARD);
            assert!(!page.is_null());
            let mut info = [0u64; 8];
            assert_eq!(XMemVirtualQuery(proc_, page, info.as_mut_ptr() as _), 1);
            assert_eq!((info[6] as u32) & 3, 0, "PAGE_GUARD page must not be marked CPU-readable");
            mem::VirtualFree(page, 0, mem::MEM_RELEASE);
        }
    }

    #[test]
    fn xmem_and_physical_page_overflows_fail_cleanly() {
        unsafe {
            // CPU heap allocation overflow -> null + ERROR_NOT_ENOUGH_MEMORY.
            fnd::SetLastError(0);
            assert!(XMemAllocDefault(usize::MAX, 0).is_null());
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);

            // Graphics layout rounding overflow (usize::MAX) and largest 64 KiB multiple
            // (which previously overflowed `size + GRAIN - 1` inside Uma::sparse_alloc / alloc_where).
            let gfx_attrs = 1u64 << 29;
            assert!(xmem_graphics_layout(usize::MAX, XAllocAttrs::decode(gfx_attrs)).is_none());
            fnd::SetLastError(0);
            assert!(XMemAllocDefault(usize::MAX, gfx_attrs).is_null());
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);

            let max_aligned = usize::MAX & !(crate::gpu::uma::GRAIN - 1);
            assert!(XMemAllocDefault(max_aligned, gfx_attrs).is_null());

            // Physical page byte-count overflow -> FALSE + ERROR_NOT_ENOUGH_MEMORY.
            let mut huge = usize::MAX;
            let mut page = 0usize;
            fnd::SetLastError(0);
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut huge, &mut page), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_NOT_ENOUGH_MEMORY);
        }
    }

    /// A no-access graphics reservation of 64 KiB pages is the GPU view of the CPU-readable one
    /// of its size before it: textures there read the CPU view (`title_textures`).
    #[test]
    fn gpu_view_reservations_name_their_cpu_view() {
        // (addresses far above anything Windows hands out: only the bookkeeping is tested)
        let (a, b, size) = (0x7000_0000_0000usize, 0x7006_0000_0000usize, 0x6_0000_0000usize);
        let other = 0x7010_0000_0000usize;
        mirrors::reserved(a, size, 0x2040_2000, mem::PAGE_READWRITE);
        mirrors::reserved(other, 0x1000_0000, 0x2040_2000, mem::PAGE_READWRITE); // another size
        assert_eq!(mirrors::source(b + 0x1234_0000), None, "no GPU view yet");
        mirrors::reserved(b, size, 0x2040_2000, 0x801);
        assert_eq!(mirrors::source(b), Some(a));
        assert_eq!(mirrors::source(b + 0x4524_0000), Some(a + 0x4524_0000));
        assert_eq!(mirrors::source(b + size), None, "past its end");
        assert_eq!(mirrors::source(a + 0x10), None, "the CPU view is no GPU view");
        // committed memory or other protections are no GPU views
        let c = 0x7020_0000_0000usize;
        mirrors::reserved(c, size, 0x2040_3000, 0x801);
        assert_eq!(mirrors::source(c), None);
        mirrors::released(a);
        assert_eq!(mirrors::source(b), None, "released with its CPU view");
        mirrors::released(other);
    }

    /// A pair that forms or goes is a change of what the GPU view shows: readers of GPU views
    /// (`gpu_view_map_changes`) look at its range again – once paired, a texture placed there
    /// before reads the CPU view, and once unpaired it no longer does.
    #[test]
    fn a_gpu_view_pair_that_forms_or_goes_is_a_change_of_the_view() {
        // (addresses far above anything Windows hands out: only the bookkeeping is tested)
        let (a, b, size) = (0x7100_0000_0000usize, 0x7106_0000_0000usize, 0x6_1000_0000usize);
        let seen = |since: u64| gpu_view_map_changes(since).unwrap_or_default().contains(&(b, b + size));
        assert!(!mirrors::would_pair(size));
        mirrors::reserved(a, size, 0x2040_2000, mem::PAGE_READWRITE);
        assert!(mirrors::would_pair(size), "a CPU view waits for its GPU view");
        let before = gpu_view_map_gen();
        assert!(mirrors::reserved(b, size, 0x2040_2000, 0x801));
        assert!(!mirrors::would_pair(size));
        assert!(gpu_view_map_gen() > before && seen(before), "the pair is a change of the view");
        let paired = gpu_view_map_gen();
        mirrors::released(b);
        assert_eq!(mirrors::source(b), None);
        assert!(gpu_view_map_gen() > paired && seen(paired), "the end of the pair is a change too");
        // the CPU view waits again; released with it, a GPU view goes too
        assert!(mirrors::would_pair(size));
        assert!(mirrors::reserved(b, size, 0x2040_2000, 0x801));
        let paired = gpu_view_map_gen();
        mirrors::released(a);
        assert!(seen(paired), "a GPU view whose CPU view went");
        assert!(!mirrors::would_pair(size) && !mirrors::reserved(b, size, 0x2040_2000, 0x801));
    }

    /// GPU-visible title memory that reaches the plain-memory fallback (64 KiB pages, large
    /// pages, a graphics protection) is write-watched: a texture placed there sees the CPU's
    /// writes. The fallback is called directly: through `VirtualAlloc` these sizes land in a
    /// unified-memory window whenever one has room (once any test has made the windows).
    #[test]
    fn gpu_visible_fallback_memory_is_write_watched() {
        let null = std::ptr::null();
        // (MEM_64K_PAGES | MEM_RESERVE; MEM_LARGE_PAGES | MEM_RESERVE | MEM_COMMIT with a
        // PAGE_GRAPHICS_* protection; MEM_GRAPHICS | MEM_RESERVE | MEM_COMMIT)
        let gpu = [(0x2040_2000u32, mem::PAGE_READWRITE), (0x2000_3000, 0x2404), (MEM_GRAPHICS | 0x3000, mem::PAGE_READWRITE)];
        for &(t, p) in &gpu {
            assert!(gpu_visible_fallback(null, t, p), "flags {t:#x} protect {p:#x}");
        }
        // plain memory stays plain: CPU-only reservations, commits at a fixed address
        assert!(!gpu_visible_fallback(null, 0x3000, mem::PAGE_READWRITE));
        assert!(!gpu_visible_fallback(null, 0x1000, 0x2404), "a commit without a reservation");
        assert!(!gpu_visible_fallback(0x1_0000 as *const c_void, 0x2040_2000, mem::PAGE_READWRITE), "a fixed address");
        unsafe {
            for (t, p) in gpu {
                let size = 0x40_0000;
                let q = reserve_watched(size, t, p);
                assert!(!q.is_null(), "flags {t:#x}");
                let a = q as usize;
                assert!(in_graphics_reservation(a), "flags {t:#x}: a fallback allocation is watched");
                if t & 0x1000 == 0 {
                    assert!(!VirtualAlloc((a + 0x2_0000) as *const c_void, 0x1_0000, 0x1000, mem::PAGE_READWRITE).is_null());
                }
                let mut buf = [std::ptr::null_mut::<c_void>(); 16];
                let (mut n, mut gran) = (buf.len(), 0u32);
                assert_eq!(mem::GetWriteWatch(1, q, size, buf.as_mut_ptr(), &mut n, &mut gran), 0);
                ((a + 0x2_1234) as *mut u8).write_volatile(7);
                n = buf.len();
                assert_eq!(mem::GetWriteWatch(1, q, size, buf.as_mut_ptr(), &mut n, &mut gran), 0);
                assert_eq!(&buf[..n], &[(a + 0x2_1000) as *mut c_void], "flags {t:#x}: the written page is reported");
                assert_eq!(VirtualFree(q, 0, mem::MEM_RELEASE), 1);
                assert!(!in_graphics_reservation(a));
            }
        }
    }

    #[test]
    fn virtual_free_ex_and_sparse_validation() {
        unsafe {
            let cur = thr::GetCurrentProcess();

            // Failed MEM_RELEASE (non-zero size) must keep a tracked graphics reservation registered,
            // and VirtualFreeEx(GetCurrentProcess(), ...) must release & untrack it.
            let host = mem::VirtualAlloc(std::ptr::null(), 0x10000, mem::MEM_RESERVE, mem::PAGE_NOACCESS);
            assert!(!host.is_null());
            GRAPHICS_RESERVATIONS.lock().unwrap().insert(host as usize, 0x10000);
            assert!(in_graphics_reservation(host as usize));
            assert_eq!(VirtualFree(host, 0x1000, mem::MEM_RELEASE), 0);
            assert!(in_graphics_reservation(host as usize));
            assert_eq!(VirtualFreeEx(cur, host, 0, mem::MEM_RELEASE), 1);
            assert!(!in_graphics_reservation(host as usize));

            // Sparse UMA reservation: invalid flag combos / non-zero MEM_RELEASE fail without freeing;
            // VirtualFreeEx(GetCurrentProcess(), ...) releases cleanly.
            let sp = VirtualAlloc(std::ptr::null(), 0x10000, mem::MEM_RESERVE | MEM_GRAPHICS, mem::PAGE_READWRITE);
            if !sp.is_null() && crate::gpu::uma::get().is_some_and(|u| u.is_sparse(sp as usize)) {
                fnd::SetLastError(0);
                assert_eq!(VirtualFree(sp, 0x1000, mem::MEM_RELEASE), 0);
                assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

                fnd::SetLastError(0);
                assert_eq!(VirtualFree(sp, 0, mem::MEM_RELEASE | mem::MEM_DECOMMIT), 0);
                assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

                fnd::SetLastError(0);
                assert_eq!(VirtualFree(sp, 0, 0), 0);
                assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);

                assert_eq!(VirtualFreeEx(cur, sp, 0, mem::MEM_RELEASE), 1);
                assert!(crate::gpu::uma::get().unwrap().sparse_reservation(sp as usize).is_none());
            } else if !sp.is_null() {
                assert_eq!(VirtualFreeEx(cur, sp, 0, mem::MEM_RELEASE), 1);
            }
        }
    }

    #[test]
    fn get_module_handle_matches_load_library_for_console_aliases_and_stubs() {
        unsafe {
            // Real Windows system DLL keeps its genuine Windows HMODULE.
            let real_k32 = ll::GetModuleHandleW(crate::util::wide("kernel32.dll").as_ptr()) as usize;
            assert_ne!(real_k32, 0);
            assert_eq!(GetModuleHandleW(crate::util::wide("kernel32.dll").as_ptr()), real_k32);
            assert_eq!(GetModuleHandleA(c"kernel32.dll".as_ptr() as *const u8), real_k32);

            // Console-only kernelx aliases return the kernelx HLE pseudo handle (matching LoadLibrary).
            let kx = modules::hle_handle(crate::hle::KERNELX);
            for dll in ["advapi32legacy.dll", "kernel32legacy.dll", "minappmodelx.dll"] {
                let w = crate::util::wide(dll);
                let h_load = LoadLibraryW(w.as_ptr());
                assert_eq!(h_load, kx, "LoadLibraryW({dll})");
                assert_eq!(GetModuleHandleW(w.as_ptr()), kx, "GetModuleHandleW({dll})");

                let c = std::ffi::CString::new(dll).unwrap();
                assert_eq!(GetModuleHandleA(c.as_ptr() as *const u8), kx, "GetModuleHandleA({dll})");

                let mut out = 0usize;
                assert_eq!(GetModuleHandleExW(0, w.as_ptr(), &mut out), 1, "GetModuleHandleExW({dll})");
                assert_eq!(out, kx);
            }

            // Unknown DLL still returns NULL + ERROR_MOD_NOT_FOUND.
            fnd::SetLastError(0);
            assert_eq!(GetModuleHandleW(crate::util::wide("no_such_optional_dll_xyz.dll").as_ptr()), 0);
            assert_eq!(fnd::GetLastError(), fnd::ERROR_MOD_NOT_FOUND);
        }

        // @stub DLLs: unloaded probe returns 0; once loaded (hle_handle created), returns that handle.
        let mut cfg = crate::config::Config::default();
        cfg.aliases.insert("unit_stub_probe.dll".into(), "@stub".into());
        let reg = crate::hle::build(&cfg);
        assert_eq!(module_handle_in(&reg, "unit_stub_probe.dll"), 0);
        let stub_h = modules::hle_handle("unit_stub_probe.dll");
        assert_ne!(stub_h, 0);
        assert_eq!(module_handle_in(&reg, "unit_stub_probe.dll"), stub_h);
    }

    #[test]
    fn virtual_query_fastpath_covers_committed_coherent_window() {
        let Some(u) = crate::gpu::uma::get() else { return };
        let Some(k) = u.committed_window() else { return };
        let (base, len) = u.window(k);
        assert!(len >= 0x2000);
        let addr = (base + 0x1234) as *const c_void;
        unsafe {
            let mut mbi: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            let n = VirtualQuery(addr, &mut mbi, std::mem::size_of_val(&mbi));
            assert_eq!(n, std::mem::size_of::<mem::MEMORY_BASIC_INFORMATION>());
            assert_eq!(mbi.BaseAddress as usize, base + 0x1000);
            assert_eq!(mbi.AllocationBase as usize, base);
            assert_eq!(mbi.RegionSize, len - 0x1000);
            assert_eq!(mbi.State, mem::MEM_COMMIT);
            assert_eq!(mbi.Protect, mem::PAGE_READWRITE);

            let mut mbi2: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            let n2 = VirtualQueryEx(thr::GetCurrentProcess(), addr, &mut mbi2, std::mem::size_of_val(&mbi2));
            assert_eq!(n2, n);
            assert_eq!(mbi2.RegionSize, mbi.RegionSize);

            let mut xinfo = [0u8; 64];
            assert_eq!(XMemVirtualQuery(thr::GetCurrentProcess(), addr, xinfo.as_mut_ptr() as *mut c_void), 1);
            // A coherent UMA window is CPU-readable and GPU-accessible.
            assert_eq!(u32::from_ne_bytes(xinfo[48..52].try_into().unwrap()), 3);
            assert!(committed_range(base + 0x1000, 0x1000));
            assert!(committed_page(base + 0x1000));
        }
    }

    #[test]
    fn sparse_64k_reservation_registers_gpu_view_mirror_and_releases_cleanly() {
        unsafe {
            // Reserve a 64 KiB-page CPU view (lands in sparse UMA when space is available) and
            // pair it with a PAGE_NOACCESS | PAGE_GRAPHICS_NOACCESS GPU view of the same size.
            let size = 4 << 20;
            let cpu = VirtualAlloc(std::ptr::null(), size, mem::MEM_RESERVE | MEM_64K_PAGES, mem::PAGE_READWRITE);
            assert!(!cpu.is_null());
            let gpu = VirtualAlloc(std::ptr::null(), size, mem::MEM_RESERVE | MEM_64K_PAGES, mem::PAGE_NOACCESS | 0x800);
            assert!(!gpu.is_null());
            assert_eq!(
                mirrors::source(gpu as usize + 0x2_0000),
                Some(cpu as usize + 0x2_0000),
                "GPU view must resolve to CPU view even when CPU view lives in sparse UMA"
            );
            // GPU-view alias must not consume sparse UMA VA: it goes to a view window (16 windows)
            // or stays outside unified memory.
            if let Some(u) = crate::gpu::uma::get() {
                assert!(!u.is_sparse(gpu as usize), "CPU-inaccessible GPU view should stay outside sparse UMA");
                let views = (0..u.shader_windows()).any(|k| u.is_view_window(k));
                assert_eq!(u.is_view(gpu as usize), views, "the GPU view is in a view window when there are any");
                assert!(!u.contains(gpu as usize), "the CPU cannot touch a GPU view");
            }
            assert_eq!(VirtualFree(gpu, 0, mem::MEM_RELEASE), 1);
            assert_eq!(mirrors::source(gpu as usize + 0x2_0000), None);
            assert_eq!(VirtualFree(cpu, 0, mem::MEM_RELEASE), 1);
        }
    }

    #[test]
    fn physical_home_overflow_with_sparse_windows_skips_coherent_window() {
        let u = crate::gpu::uma::Uma::for_runtime(16 << 20).unwrap();
        let coherent_before = u.used();
        let mut held = Vec::new();
        for chunk in [4usize << 30, 1usize << 30, 64usize << 20, 1usize << 20, PHYS_PAGE] {
            while let Some(p) = u.sparse_reserve(chunk) {
                held.push(p);
            }
        }
        assert!(u.sparse_alloc(2 * PHYS_PAGE, PHYS_PAGE).is_none());

        unsafe {
            let home = alloc_physical_home(&u, 2 * PHYS_PAGE).unwrap();
            assert!(!u.contains(home), "overflow on sparse UMA must not consume the coherent window");
            assert_eq!(u.used(), coherent_before, "coherent window headroom must stay untouched");
            assert!(!committed_page(home), "plain-memory home must stay reserved until first use");
            assert!(lazy_home(home));
            assert_eq!(mem::VirtualFree(home as *mut c_void, 0, mem::MEM_RELEASE), 1);
        }
        for p in held {
            assert!(u.sparse_release(p).is_some());
        }
    }

    /// Homes of physical pages leave window 0 – the console's low 4 GiB – to the title: after
    /// the Series S run's 2.9 GB of sparse GPU heaps and 664 MB of pages, its 38 MB view heap still
    /// fits there (the homes in window 0 left it none, and the heap went to window 1).
    #[test]
    fn physical_homes_leave_window_0_to_the_title() {
        let u = crate::gpu::uma::Uma::for_runtime(16 << 20).unwrap();
        unsafe {
            let heaps = u.sparse_alloc(2892 << 20, 2 << 20).unwrap().0;
            let home = alloc_physical_home(&u, 664 << 20).unwrap();
            assert_eq!(u.window_of(home), Some(1), "the home keeps out of window 0");
            assert!(lazy_home(home));
            let view_heap = u.sparse_alloc(0x260_0000, crate::gpu::uma::GRAIN).unwrap().0;
            assert_eq!(u.window_of(view_heap), Some(0));
            for p in [view_heap, home, heaps] {
                assert!(u.sparse_release(p).is_some());
            }
        }
    }

    #[test]
    fn physical_home_without_sparse_windows_uses_coherent_window() {
        let u = crate::gpu::uma::Uma::with_windows(1, 16 << 20).unwrap();
        assert!(!u.has_sparse());
        unsafe {
            let home = alloc_physical_home(&u, 2 * PHYS_PAGE).unwrap();
            assert!(u.is_committed(home), "coherent-only UMA should still back physical homes");
            assert!(u.free(home));
        }
    }
}
