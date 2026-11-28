//! Textures a title places in its own memory (CreatePlacedResourceX): on the console the GPU
//! samples that memory, laid out as the console's layout library says (`hle::xg`, checked
//! against the console's own). Here a PC texture holds a detiled copy: uploaded when the
//! texture is created and again before the title's next ExecuteCommandLists once the CPU wrote
//! any of its pages (unified memory and the title's graphics reservations are write-watched,
//! `gpu::uma`, `kernelx::VirtualAlloc`) – textures streamed in after creation (mips loaded
//! later, physical pages mapped in, DirectStorage) included.
//!
//! The first upload is not done when the texture is made (a title makes thousands while it
//! streams – Fortnite's render thread then spent its time in CreatePlacedResourceX and its
//! render command fences ran late): it is pending until the title's next ExecuteCommandLists,
//! where all pending and rewritten textures go in batches on one list each.
//!
//! The copy waits for the GPU on an XCL queue of the device. Memory that is not readable
//! (reserved, not mapped yet) reads as zero. GPU writes into the title's memory (copies to the
//! address) are not seen; the title's own GPU work on the texture goes to the PC texture.
//!
//! A copy starts and ends in the state the title's submitted work left the texture in: its
//! transitions of these textures are captured in each physical command-list segment's immutable
//! submission effects and applied in submission order – per subresource once one is named.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::texture_upload;
use crate::gpu::texture::{self, Subresource};
use crate::hle::xg;

/// GetWriteWatch with atomic reset of the reported writes.
const WRITE_WATCH_FLAG_RESET: u32 = 1;
#[link(name = "kernel32")]
extern "system" {
    fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
    fn ResetWriteWatch(base: *const c_void, size: usize) -> u32;
}

/// A texture whose upload is due, or whose memory was written: a reference while that is seen to.
struct Due {
    res: ID3D12Resource,
    device: usize,
    /// where its bytes are read
    src: usize,
    tex: Arc<xg::Texture>,
    states: Vec<D3D12_RESOURCE_STATES>,
    known: texture_upload::Known,
    /// all of it goes up, whatever was written
    whole: bool,
    /// what of it goes up again (`texture_upload::sent`)
    again: texture_upload::Change,
    /// (`Entry::target`)
    target: bool,
    /// where its pages are read (`Entry::paged`)
    pages: Option<Arc<texture_upload::PageMap>>,
    /// registration generation (Entry::generation) when it was picked
    generation: u64,
}

/// One incarnation of a title command queue. The raw address alone can be reused after release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WriterQueueId {
    pub(super) raw: usize,
    pub(super) generation: u64,
}

/// A GPU write captured while a physical command-list segment is recorded. Its registration
/// generation prevents a late/replayed segment from marking a new object at a reused address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GpuWrite {
    pub(super) res: usize,
    pub(super) generation: u64,
    pub(super) subresource: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuWriter {
    epoch: u64,
    queue: WriterQueueId,
}

struct Entry {
    device: usize,
    /// PC allocation size of a committed resource (zero for placed resources and test entries)
    committed_bytes: u64,
    /// the title's address of the texture
    va: usize,
    /// where its bytes are read and its CPU writes watched: `va`, or – for a texture in a GPU
    /// view of the title's CPU memory with nothing mapped there – the same offset in the CPU
    /// view (`kernelx::mirrors`)
    src: usize,
    /// `va` lies in a GPU view: `src` is checked again when that view's mappings change
    in_gpu_view: bool,
    /// the title mapped pages of a GPU view under it (the queue's page mappings): `src` is `va`,
    /// and each upload reads its pages where they are kept (`texture_upload::PageMap`)
    paged: bool,
    tex: Arc<xg::Texture>,
    /// the state the title's submitted work leaves the texture in (uploads leave it there):
    /// the creation state, then what its transitions give – one state for every subresource,
    /// or one per subresource (D3D12 order) while they differ
    states: Vec<D3D12_RESOURCE_STATES>,
    /// not uploaded yet (made since the last submit)
    pending: bool,
    /// the title's GPU work wrote it (CopyTextureRegion into it): its CPU view is stale from
    /// then on and no longer uploaded
    gpu_written: bool,
    /// the calls of the queue's page mappings made when a list that writes it was last submitted
    /// (`kernelx::mirrors::calls`): a mapping changed under it by a later call is newer than the
    /// GPU's write
    written_call: u64,
    /// Last submitted GPU writer: one for the whole resource while uniform, otherwise one per
    /// subresource in D3D12 order. Empty until a tracked write is submitted.
    gpu_writers: Vec<Option<GpuWriter>>,
    /// CPU memory is the source of this texture and must be watched/uploaded. False for a
    /// GPU-owned UAV/render target retained only for future coherency readback.
    cpu_backed: bool,
    /// backed by shared `vram::Pages` tiles (overlapping placements share video memory)
    placed: bool,
    /// tiles were unmapped after a different-range placement superseded this texture
    unmapped: bool,
    /// monotonic registration generation (newest wins when uncovering shadowed textures)
    generation: u64,
    /// what its pages held at its last upload
    known: texture_upload::Known,
    /// the GPU can write it other than by copies (a render target, a UAV): XCL does not see all of
    /// that, so when the CPU's bytes of it change it goes up as a whole
    target: bool,
}

/// textures waiting for their first upload
static PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// raw COM pointers of the title's objects; uploads hold a reference while they run
unsafe impl Send for Entry {}

struct Table {
    /// Active PC textures (newest at each range) → its place in the title's memory
    by_res: BTreeMap<usize, Entry>,
    /// Active `(va, PC texture)` → end address (`va + size`); entries with different `(va, end)` are disjoint
    by_va: BTreeMap<(usize, usize), usize>,
    /// Older overlapped textures kept dormant until released or uncovered when a newer one is released
    shadowed: BTreeMap<usize, Entry>,
    /// Shadowed `(va, PC texture)` → end address (may overlap)
    shadowed_by_va: BTreeMap<(usize, usize), usize>,
    /// Largest byte span currently in `shadowed` (bounds range queries)
    max_shadowed_size: usize,
    next_generation: u64,
    /// the active textures by the 1 MiB pieces of their source (`Entry::src`) they cover: which of
    /// them lie under a written page. A title has tens of thousands of textures; a submit asks
    /// about the few its dirty pages touch
    src_pieces: BTreeMap<usize, Vec<usize>>,
    /// textures whose first upload is still to come (`Entry::pending`; may name ones that went)
    pending_res: std::collections::BTreeSet<usize>,
    /// 16 MiB blocks of the textures' sources -> how many active ones cover it: the part of the
    /// watched ranges that textures make (cheap to rebuild from, unlike all textures)
    coarse: BTreeMap<usize, u32>,
}

/// the size of `Table::coarse`'s blocks
const SRC_COARSE_SHIFT: usize = 24;

/// goes up whenever a block of `Table::coarse` becomes used or unused
static COARSE_GEN: AtomicU64 = AtomicU64::new(0);

/// the piece (1 MiB) of an address, for `Table::src_pieces`
const SRC_PIECE_SHIFT: usize = 20;

/// Whether `src` lies in a view window of unified memory: never watched (`Uma::watch_regions`),
/// so a texture read there (`Entry::paged`) adds no block to `Table::coarse`.
fn in_view_window(src: usize) -> bool {
    crate::gpu::uma::get().is_some_and(|u| u.is_view(src))
}

impl Table {
    const fn new() -> Self {
        Table {
            by_res: BTreeMap::new(),
            by_va: BTreeMap::new(),
            shadowed: BTreeMap::new(),
            shadowed_by_va: BTreeMap::new(),
            max_shadowed_size: 0,
            next_generation: 0,
            src_pieces: BTreeMap::new(),
            pending_res: std::collections::BTreeSet::new(),
            coarse: BTreeMap::new(),
        }
    }

    fn link_src(&mut self, res: usize, src: usize, size: usize) {
        for p in (src >> SRC_PIECE_SHIFT)..=((src + size.max(1) - 1) >> SRC_PIECE_SHIFT) {
            self.src_pieces.entry(p).or_default().push(res);
        }
        if in_view_window(src) {
            return;
        }
        for b in (src >> SRC_COARSE_SHIFT)..=((src + size.max(1) - 1) >> SRC_COARSE_SHIFT) {
            let c = self.coarse.entry(b).or_insert(0);
            *c += 1;
            if *c == 1 {
                COARSE_GEN.fetch_add(1, Ordering::Release);
            }
        }
    }

    fn unlink_src(&mut self, res: usize, src: usize, size: usize) {
        for p in (src >> SRC_PIECE_SHIFT)..=((src + size.max(1) - 1) >> SRC_PIECE_SHIFT) {
            if let Some(v) = self.src_pieces.get_mut(&p) {
                v.retain(|&r| r != res);
                if v.is_empty() {
                    self.src_pieces.remove(&p);
                }
            }
        }
        if in_view_window(src) {
            return;
        }
        for b in (src >> SRC_COARSE_SHIFT)..=((src + size.max(1) - 1) >> SRC_COARSE_SHIFT) {
            let gone = match self.coarse.get_mut(&b) {
                Some(c) if *c > 1 => {
                    *c -= 1;
                    false
                }
                Some(_) => true,
                None => false,
            };
            if gone {
                self.coarse.remove(&b);
                COARSE_GEN.fetch_add(1, Ordering::Release);
            }
        }
    }

    /// The textures (active) whose source the dirty pages `dirty` (sorted) may touch.
    fn touched_by(&self, dirty: &[usize]) -> std::collections::BTreeSet<usize> {
        let mut out = std::collections::BTreeSet::new();
        let mut last = usize::MAX;
        for &p in dirty {
            let piece = p >> SRC_PIECE_SHIFT;
            if piece != last {
                last = piece;
                if let Some(v) = self.src_pieces.get(&piece) {
                    out.extend(v.iter().copied());
                }
            }
        }
        out
    }

    fn has_active_overlap(&self, va: usize, end: usize) -> bool {
        if let Some((_, &oend)) = self.by_va.range(..(va, 0)).next_back() {
            if oend > va {
                return true;
            }
        }
        end > va && self.by_va.range((va, 0)..(end, 0)).next().is_some()
    }

    fn active_overlaps(&self, va: usize, end: usize) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for (&(ova, ores), &oend) in self.by_va.range(..(va, 0)).rev() {
            if oend <= va {
                break;
            }
            out.push((ova, ores, oend));
        }
        if end > va {
            for (&(ova, ores), &oend) in self.by_va.range((va, 0)..(end, 0)) {
                out.push((ova, ores, oend));
            }
        }
        out
    }

    fn shadowed_overlaps(&self, va: usize, end: usize) -> Vec<(usize, usize, usize)> {
        if self.shadowed.is_empty() || end <= va {
            return Vec::new();
        }
        let lo = va.saturating_sub(self.max_shadowed_size);
        self.shadowed_by_va
            .range((lo, 0)..(end, 0))
            .filter(|&(_, &send)| send > va)
            .map(|(&(sva, sres), &send)| (sva, sres, send))
            .collect()
    }

    /// Permanently retire active and shadowed entries overlapping `[va, end)` (used when a
    /// GPU-owned render target / depth / UAV overwrites the address range). Returns
    /// `(pending_delta, retired_count, placed_res_to_unmap)`.
    fn retire_range(&mut self, va: usize, end: usize) -> (isize, usize, Vec<usize>) {
        let (mut pending_delta, mut retired, mut to_unmap) = (0isize, 0usize, Vec::new());
        for (ova, ores, _) in self.active_overlaps(va, end) {
            self.by_va.remove(&(ova, ores));
            if let Some(old) = self.by_res.remove(&ores) {
                self.unlink_src(ores, old.src, old.tex.size() as usize);
                if old.pending {
                    pending_delta -= 1;
                }
                if old.placed && !old.unmapped {
                    to_unmap.push(ores);
                }
                retired += 1;
            }
        }
        for (sva, sres, _) in self.shadowed_overlaps(va, end) {
            self.shadowed_by_va.remove(&(sva, sres));
            if let Some(old) = self.shadowed.remove(&sres) {
                if old.placed && !old.unmapped {
                    to_unmap.push(sres);
                }
                retired += 1;
            }
        }
        if self.shadowed.is_empty() {
            self.max_shadowed_size = 0;
        }
        (pending_delta, retired, to_unmap)
    }

    /// Insert `res`, shadowing older active entries whose memory `[va, va + size)` it overlaps so
    /// dead placements never re-upload over shared `vram::Pages` tiles (two committed resources
    /// with the exact same range may coexist as format aliases). Placed entries superseded by a
    /// *different* range are returned in `to_unmap` so their unshared tiles return to the pool;
    /// exact same-range entries stay mapped because they already share every tile.
    fn insert(&mut self, res: usize, mut entry: Entry) -> (isize, Vec<usize>) {
        let va = entry.va;
        let end = va.saturating_add(entry.tex.size() as usize);
        let mut pending_delta = 0isize;
        let mut to_unmap = Vec::new();
        if let Some(old) = self.by_res.remove(&res) {
            self.unlink_src(res, old.src, old.tex.size() as usize);
            self.by_va.remove(&(old.va, res));
            if old.pending {
                pending_delta -= 1;
            }
        }
        if let Some(old) = self.shadowed.remove(&res) {
            self.shadowed_by_va.remove(&(old.va, res));
        }
        for (ova, ores, oend) in self.active_overlaps(va, end) {
            if !entry.placed && ova == va && oend == end && self.by_res.get(&ores).is_some_and(|e| !e.placed) {
                continue;
            }
            self.by_va.remove(&(ova, ores));
            if let Some(mut old) = self.by_res.remove(&ores) {
                self.unlink_src(ores, old.src, old.tex.size() as usize);
                if old.pending {
                    pending_delta -= 1;
                }
                if old.placed && !old.unmapped && (ova != va || oend != end) {
                    old.unmapped = true;
                    to_unmap.push(ores);
                }
                if old.placed && entry.placed {
                    texture_upload::inherit(&entry.known, &old.known);
                }
                self.max_shadowed_size = self.max_shadowed_size.max(oend.saturating_sub(ova));
                self.shadowed_by_va.insert((ova, ores), oend);
                self.shadowed.insert(ores, old);
            }
        }
        // Also release tiles of any already-shadowed exact-range sibling now overwritten by a
        // different range (e.g. A and B shared an unaligned range, then C overwrites both).
        for (sva, sres, send) in self.shadowed_overlaps(va, end) {
            if sva != va || send != end {
                if let Some(old) = self.shadowed.get_mut(&sres) {
                    if old.placed && !old.unmapped {
                        old.unmapped = true;
                        to_unmap.push(sres);
                    }
                }
            }
        }
        self.next_generation = self.next_generation.wrapping_add(1).max(1);
        entry.generation = self.next_generation;
        if entry.pending {
            pending_delta += 1;
            self.pending_res.insert(res);
        }
        self.by_va.insert((va, res), end);
        let (src, size) = (entry.src, entry.tex.size() as usize);
        self.link_src(res, src, size);
        self.by_res.insert(res, entry);
        (pending_delta, to_unmap)
    }

    /// Remove `res` on COM release, reactivating any shadowed older textures it uncovered (newest
    /// first). Returns `(was_present, pending_delta, unmapped_res_to_remap)`.
    fn remove(&mut self, res: usize) -> (bool, isize, Vec<usize>) {
        if let Some(old) = self.shadowed.remove(&res) {
            self.shadowed_by_va.remove(&(old.va, res));
            if self.shadowed.is_empty() {
                self.max_shadowed_size = 0;
            }
            return (true, 0, Vec::new());
        }
        let Some(old) = self.by_res.remove(&res) else { return (false, 0, Vec::new()) };
        self.unlink_src(res, old.src, old.tex.size() as usize);
        self.by_va.remove(&(old.va, res));
        let mut pending_delta = if old.pending { -1 } else { 0 };
        let va = old.va;
        let end = va.saturating_add(old.tex.size() as usize);
        let mut cands: Vec<(u64, usize, usize, usize)> = self
            .shadowed_overlaps(va, end)
            .into_iter()
            .filter_map(|(sva, sres, send)| self.shadowed.get(&sres).map(|e| (e.generation, sres, sva, send)))
            .collect();
        cands.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let mut to_remap = Vec::new();
        for (_, sres, sva, send) in cands {
            if self.has_active_overlap(sva, send) {
                continue;
            }
            self.shadowed_by_va.remove(&(sva, sres));
            if let Some(mut e) = self.shadowed.remove(&sres) {
                // (the one that lay over it may have been uploaded over the same video memory)
                texture_upload::forget(&e.known);
                if e.unmapped {
                    e.unmapped = false;
                    to_remap.push(sres);
                    if !e.gpu_written {
                        e.pending = true;
                    }
                }
                if e.pending {
                    pending_delta += 1;
                }
                self.by_va.insert((sva, sres), send);
                let (src, size) = (e.src, e.tex.size() as usize);
                self.link_src(sres, src, size);
                if e.pending {
                    self.pending_res.insert(sres);
                }
                self.by_res.insert(sres, e);
            }
        }
        if self.shadowed.is_empty() {
            self.max_shadowed_size = 0;
        }
        (true, pending_delta, to_remap)
    }

    fn get_mut(&mut self, res: usize) -> Option<&mut Entry> {
        let Self { by_res, shadowed, .. } = self;
        by_res.get_mut(&res).or_else(|| shadowed.get_mut(&res))
    }

    fn committed_bytes_for(&self, device: usize) -> (usize, u64) {
        self.by_res.values().chain(self.shadowed.values()).filter(|e| e.device == device && !e.placed)
            .fold((0usize, 0u64), |(n, bytes), e| (n.saturating_add(1), bytes.saturating_add(e.committed_bytes)))
    }

    fn overlapping(&self, va: usize, len: usize) -> Vec<(usize, usize)> {
        self.active_overlaps(va, va.saturating_add(len)).into_iter().take(4).map(|(ova, ores, _)| (ores, ova)).collect()
    }
}

fn adjust_pending(delta: isize) {
    if delta > 0 {
        PENDING.fetch_add(delta as usize, Ordering::Relaxed);
    } else if delta < 0 {
        PENDING.fetch_sub((-delta) as usize, Ordering::Relaxed);
    }
}

unsafe fn clone_resources(ptrs: &[usize]) -> Vec<ID3D12Resource> {
    ptrs.iter()
        .filter_map(|&raw| {
            let p = raw as *mut c_void;
            ID3D12Resource::from_raw_borrowed(&p).map(|r| r.clone())
        })
        .collect()
}

/// PC texture → its place in the title's memory
static TEXTURES: Mutex<Table> = Mutex::new(Table::new());
// The dirty-page scan consumes write-watch state for every submission thread.
// Keep the scan, pending-list update and GPU uploads together.
static REFRESH: Mutex<()> = Mutex::new(());
static TEXTURES_GEN: AtomicU64 = AtomicU64::new(0);
static TEXTURES_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// cached `watch_ranges` keyed by `watch_key`
static WATCH_CACHE: Mutex<Option<((u64, u64, u64, u64), Arc<Vec<(usize, usize)>>)>> = Mutex::new(None);

static CPU_TEXTURES_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// the queue uploads go to, per device
static QUEUES: Mutex<BTreeMap<usize, ID3D12CommandQueue>> = Mutex::new(BTreeMap::new());
static UPLOADS: AtomicU64 = AtomicU64::new(0);

/// Number of textures kept in title memory and uploads done so far.
pub fn counts() -> (usize, u64) {
    (TEXTURES_LEN.load(Ordering::Relaxed), UPLOADS.load(Ordering::Relaxed))
}

/// Sampled textures tracked for CPU coherency on `device` that use committed resources: count and
/// their PC allocation bytes. This deliberately excludes untracked render/depth/UAV fallbacks.
/// Shadowed resources remain live and are included once until the title releases them.
pub(super) fn committed_bytes_for(device: usize) -> (usize, u64) {
    TEXTURES.lock().unwrap().committed_bytes_for(device)
}

unsafe fn committed_allocation_bytes(device: *mut c_void, resource: *mut c_void, placed: bool) -> u64 {
    if placed || !super::memory_report::enabled() { return 0 }
    let Some(dev) = ID3D12Device::from_raw_borrowed(&device) else { return 0 };
    let Some(res) = ID3D12Resource::from_raw_borrowed(&resource) else { return 0 };
    let bytes = dev.GetResourceAllocationInfo(0, &[res.GetDesc()]).SizeInBytes;
    if bytes == u64::MAX { 0 } else { bytes }
}

fn update_lengths(t: &Table) {
    TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
    CPU_TEXTURES_LEN.store(t.by_res.values().filter(|e| e.cpu_backed).count(), Ordering::Relaxed);
}

/// Whether a non-depth render target / UAV texture is small enough to be a CPU-filled glyph or UI
/// atlas (`<= 1M` pixels, up to `2048` on either axis, e.g. a retail title's `2048x128` R8 glyph cache).
fn is_small_cpu_filled_target(w: u64, h: u32, samples: u32, flags: u32) -> bool {
    let pixels = w.saturating_mul(h as u64);
    w <= 2048 && h <= 2048 && (64 * 64..=1024 * 1024).contains(&pixels) && samples <= 1 && flags & 0x2 == 0
}

/// A texture the title placed at `va` (its console desc `desc`): kept in step with the memory.
pub(super) unsafe fn register(device: *mut c_void, res: *mut c_void, va: usize, desc: &[u8; 0x38], state: D3D12_RESOURCE_STATES) -> Result<(), String> {
    // (a new texture object, or the address of an old one again: no stale fingerprint schedule)
    STREAM_RECHECK.lock().unwrap().remove(&(res as usize));
    let tex = xg::Texture::new(desc).map_err(|hr| format!("layout {hr:#x}"))?;
    // the console's 32-bit 116 / 117 floats are half floats on the PC (`pc_resource_desc`): the
    // title's bytes are not the resource's texels
    if matches!(u32::from_le_bytes(desc[0x20..0x24].try_into().unwrap()), 116 | 117) {
        return Err("a console float format the PC holds as R16G16B16A16_FLOAT".into());
    }
    // Render targets and depth buffers are the GPU's (their metadata planes, empty planes) –
    // except a small single-sample colour target with texels: a retail title makes its glyph
    // caches that way (256x256 R8 with ALLOW_RENDER_TARGET, two planes) and fills them from the
    // CPU. Left to the GPU they stayed as they were made and its menus' body text lost most of
    // its letters. Such a target is uploaded like any texture when the CPU wrote its pages; one
    // the title only renders to is not CPU-written and so not uploaded over.
    // `XCL_SYNC_SMALL_TARGETS=0` turns this off.
    let small_target = {
        let (w, h) = (u64::from_le_bytes(desc[0x10..0x18].try_into().unwrap()), u32::from_le_bytes(desc[0x18..0x1c].try_into().unwrap()));
        let (samples, flags) = (u32::from_le_bytes(desc[0x24..0x28].try_into().unwrap()), u32::from_le_bytes(desc[0x30..0x34].try_into().unwrap()));
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF.get_or_init(|| std::env::var("XCL_SYNC_SMALL_TARGETS").is_ok_and(|v| v.trim() == "0"));
        !off && is_small_cpu_filled_target(w, h, samples, flags)
    };
    if tex.texels().is_none() || (tex.planes.len() != 1 && !small_target) {
        return Err("a render target / depth buffer".into());
    }
    if tex.planes.len() != 1 {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 2 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: a small render target at {va:#x} ({} planes) is kept in step with the title's memory – a glyph cache the CPU fills ({n} so far)", tex.planes.len());
        }
    }
    if ID3D12Resource::from_raw_borrowed(&res).is_none() {
        return Err("no resource".into());
    }
    let (src, paged) = source_of(va, tex.size() as usize);
    if src != va {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 2 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: a texture at {va:#x} lies in a GPU view of the title's CPU memory: kept in step with {src:#x} ({n} so far)");
        }
    }
    if paged {
        note_paged(va);
    }
    let in_gpu_view = crate::hle::kernelx::mirrors::source(va).is_some();
    let placed = super::vram::is_placed_texture(res as usize);
    let e = Entry {
        device: device as usize,
        committed_bytes: committed_allocation_bytes(device, res, placed),
        va,
        src,
        in_gpu_view,
        paged,
        tex: Arc::new(tex),
        states: vec![state],
        pending: true,
        gpu_written: false,
        written_call: 0,
        gpu_writers: Vec::new(),
        cpu_backed: true,
        placed,
        unmapped: false,
        generation: 0,
        known: Default::default(),
        target: u32::from_le_bytes(desc[0x30..0x34].try_into().unwrap()) & 0x7 != 0,
    };
    let (hot_src, hot_bytes) = (e.src, e.tex.size() as usize);
    let to_unmap = {
        let mut t = TEXTURES.lock().unwrap();
        let (delta, unmap_ptrs) = t.insert(res as usize, e);
        adjust_pending(delta);
        update_lengths(&t);
        clone_resources(&unmap_ptrs)
    };
    for r in &to_unmap {
        super::vram::unmap_superseded(r);
    }
    TEXTURES_GEN.fetch_add(1, Ordering::Release);
    // the data the title is about to write there is for the next submit
    mark_hot(hot_src, hot_bytes);
    Ok(())
}

/// Validate and retain the console layout of a GPU-owned UAV/render target. Depth/stencil,
/// MSAA, format-promoted resources, and multiple texel planes are deliberately unsupported:
/// their PC bytes cannot be tiled back into the console layout without conversion/metadata.
fn gpu_owned_layout(desc: &[u8; 0x38], pc: &D3D12_RESOURCE_DESC) -> Result<xg::Texture, String> {
    let u = |o: usize| u32::from_le_bytes(desc[o..o + 4].try_into().unwrap());
    let flags = u(0x30);
    if flags & 2 != 0 || flags & 5 == 0 {
        return Err("not a supported colour render target / UAV".into());
    }
    if (pc.Flags.0 as u32) & 5 == 0 {
        return Err("not GPU-writable on the PC".into());
    }
    if u(0x24) != 1 || pc.SampleDesc.Count != 1 {
        return Err("a multisampled texture".into());
    }
    if u(0x20) != pc.Format.0 as u32 {
        return Err("a console format promoted on the PC".into());
    }
    if u(0) != pc.Dimension.0 as u32 || u64::from_le_bytes(desc[0x10..0x18].try_into().unwrap()) != pc.Width || u(0x18) != pc.Height {
        return Err("a PC resource shape different from its console layout".into());
    }
    let tex = xg::Texture::new(desc).map_err(|hr| format!("layout {hr:#x}"))?;
    let surfaces = tex.planes.iter().filter(|p| matches!(&p.kind, xg::PlaneKind::Surface(_))).count();
    let s = tex.texels().filter(|_| surfaces == 1).ok_or("not exactly one supported texel plane")?;
    let layers = if s.depth > 0 { s.depth } else { s.array };
    if s.chain.levels.is_empty()
        || pc.MipLevels as usize != s.chain.levels.len()
        || pc.DepthOrArraySize as u32 != layers
        || tex.size() == 0
        || usize::try_from(tex.size()).is_err()
    {
        return Err("console and PC subresources differ".into());
    }
    Ok(tex)
}

/// Register metadata for a successfully created GPU-owned placement. No COM reference is kept;
/// the resource's final Release removes this generation before its raw address can be reused.
pub(super) unsafe fn register_gpu_owned(
    device: *mut c_void,
    res: *mut c_void,
    va: usize,
    desc: &[u8; 0x38],
    state: D3D12_RESOURCE_STATES,
) -> Result<(), String> {
    let resource = ID3D12Resource::from_raw_borrowed(&res).ok_or("no resource")?;
    let tex = gpu_owned_layout(desc, &resource.GetDesc())?;
    let size = usize::try_from(tex.size()).map_err(|_| "texture size does not fit the title address space")?;
    va.checked_add(size).ok_or("texture address range overflows")?;
    let src = mirror_source(va);
    let e = Entry {
        device: device as usize,
        committed_bytes: 0,
        va,
        src,
        in_gpu_view: crate::hle::kernelx::mirrors::source(va).is_some(),
        tex: Arc::new(tex),
        states: vec![state],
        pending: false,
        gpu_written: true,
        gpu_writers: Vec::new(),
        cpu_backed: false,
        placed: true,
        unmapped: false,
        generation: 0,
        paged: false,
        written_call: 0,
        known: Default::default(),
        target: false,
    };
    let to_unmap = {
        let mut t = TEXTURES.lock().unwrap();
        let (delta, unmap_ptrs) = t.insert(res as usize, e);
        adjust_pending(delta);
        update_lengths(&t);
        clone_resources(&unmap_ptrs)
    };
    for r in &to_unmap {
        super::vram::unmap_superseded(r);
    }
    TEXTURES_GEN.fetch_add(1, Ordering::Release);
    Ok(())
}

/// Where a texture at `va` is read from: the CPU view behind a GPU view (`kernelx::mirrors`)
/// when nothing is mapped at `va` itself, else `va`. A physical page mapped at `va` counts
/// before its first use too, while it is still reserved memory (`kernelx::physical_page_at`).
fn mirror_source(va: usize) -> usize {
    let Some(src) = crate::hle::kernelx::mirrors::source(va) else { return va };
    if crate::hle::kernelx::physical_page_at(va) {
        return va;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    #[cfg(test)]
    tests::QUERIES.with(|n| n.set(n.get() + 1));
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let mapped = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0 && m.State == MEM_COMMIT;
    if mapped { va } else { src }
}

/// Where a texture at `va` of `size` bytes is read, and whether through a table of its pages: a
/// texture over pages of a GPU view the queue's page mappings named is read at `va`, each page
/// where it is kept (no lock of AWAY, no VirtualQuery); any other as `mirror_source` says.
fn source_of(va: usize, size: usize) -> (usize, bool) {
    if crate::hle::kernelx::mirrors::touched(va, size) {
        return (va, true);
    }
    (mirror_source(va), false)
}

/// A texture at `va` is read through the pages the title mapped under it.
fn note_paged(va: usize) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 2 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: a texture at {va:#x} lies in a GPU view whose pages the title mapped (CopyPageMappingsBatchX): read where each page is kept ({n} so far)");
    }
}

/// The pages of the uploads read through a table: [kept elsewhere, kept at the CPU view's same
/// offset, at that offset never mapped, unmapped] (`kernelx::mirrors::resolve`).
static PAGE_COUNTS: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// What `PAGE_COUNTS` counted so far.
pub fn page_counts() -> [u64; 4] {
    PAGE_COUNTS.each_ref().map(|c| c.load(Ordering::Relaxed))
}

/// The table an upload of the texture at `va` of `size` bytes reads its pages through (None: the
/// title mapped none of them); `count`: its pages go into `PAGE_COUNTS`.
fn page_map(va: usize, size: usize, count: bool) -> Option<Arc<texture_upload::PageMap>> {
    let mut at = Vec::new();
    let counts = crate::hle::kernelx::mirrors::resolve(va, size, &mut at)?;
    if count {
        for (c, n) in PAGE_COUNTS.iter().zip(counts) {
            c.fetch_add(n as u64, Ordering::Relaxed);
        }
    }
    Some(Arc::new(texture_upload::PageMap::new(va, at)))
}

/// the changed GPU-view pages of the queue's page mappings a submit takes at most (one upload
/// batch of them); the rest at the next
const CHANGED_MAX: usize = BATCH_BYTES / crate::hle::kernelx::mirrors::PAGE;

/// textures read again for a mapping changed under them
static REREAD: AtomicU64 = AtomicU64::new(0);

/// The textures read again so far because a mapping changed under them.
pub fn rereads() -> u64 {
    REREAD.load(Ordering::Relaxed)
}

/// Physical pages were mapped or unmapped in a GPU view since the last look: textures there
/// whose source changed (a page now mapped under one, or gone from it) read from the new one,
/// uploaded again at this submit. The pages the queue's page mappings changed go up again in
/// the textures read through them (`mapping_changed`).
fn recheck_gpu_view_sources() {
    use crate::hle::kernelx::mirrors;
    static SEEN: AtomicU64 = AtomicU64::new(0);
    // (the cursor of the changed pages of the queue's page mappings: one reader, under REFRESH)
    static PAGES_SEEN: AtomicU64 = AtomicU64::new(0);
    let gen = crate::hle::kernelx::gpu_view_map_gen();
    let seen = SEEN.swap(gen, Ordering::AcqRel);
    let (cursor, pages) = mirrors::changes_since(PAGES_SEEN.load(Ordering::Acquire), 0, usize::MAX, CHANGED_MAX);
    PAGES_SEEN.store(cursor, Ordering::Release);
    if seen == gen && pages.is_empty() {
        return;
    }
    let mut changed = false;
    // (the shadowed textures too: one comes back when the texture over it goes)
    let mut g = TEXTURES.lock().unwrap();
    let t = &mut *g;
    // the textures over a changed page: the newest call among their changed pages, and those
    // pages' bytes of them
    let mut remapped: BTreeMap<usize, (u64, Vec<(usize, usize)>)> = BTreeMap::new();
    for &(p, call) in &pages {
        let end = p + mirrors::PAGE;
        for (va, r, to) in t.active_overlaps(p, end).into_iter().chain(t.shadowed_overlaps(p, end)) {
            let e = remapped.entry(r).or_insert((0, Vec::new()));
            e.0 = e.0.max(call);
            e.1.push((p.max(va) - va, end.min(to) - va));
        }
    }
    // the textures whose source may have changed: those overlapping the ranges whose mappings
    // changed since the last look (all of them when the log of changes has lost some)
    let names: Option<std::collections::BTreeSet<usize>> = if seen == gen {
        Some(std::collections::BTreeSet::new())
    } else {
        crate::hle::kernelx::gpu_view_map_changes(seen).map(|ranges| {
            let mut s = std::collections::BTreeSet::new();
            for (lo, hi) in ranges {
                s.extend(t.active_overlaps(lo, hi).into_iter().map(|(_, r, _)| r));
                s.extend(t.shadowed_overlaps(lo, hi).into_iter().map(|(_, r, _)| r));
            }
            s
        })
    }
    .map(|mut s| {
        s.extend(remapped.keys().copied());
        s
    });
    let mut moved: Vec<(usize, usize, usize, usize, bool)> = Vec::new();
    let mut again: Vec<usize> = Vec::new();
    let Table { by_res, shadowed, .. } = t;
    let mut look = |res: usize, e: &mut Entry, active: bool| {
        let (old, size) = (e.src, e.tex.size() as usize);
        if recheck_source(e) {
            changed = true;
            if active {
                moved.push((res, old, e.src, size, e.pending));
            }
        }
        if let Some((call, parts)) = remapped.remove(&res) {
            if mapping_changed(e, call, parts, active) {
                again.push(res);
            }
        }
    };
    match &names {
        Some(s) => {
            for &r in s {
                if let Some(e) = by_res.get_mut(&r) {
                    look(r, e, true);
                } else if let Some(e) = shadowed.get_mut(&r) {
                    look(r, e, false);
                }
            }
        }
        None => {
            for (&r, e) in by_res.iter_mut() {
                look(r, e, true);
            }
            for (&r, e) in shadowed.iter_mut() {
                look(r, e, false);
            }
        }
    }
    for (res, old, new, size, pending) in moved {
        t.unlink_src(res, old, size);
        t.link_src(res, new, size);
        if pending {
            t.pending_res.insert(res);
        }
    }
    drop(g);
    if !again.is_empty() {
        AGAIN.lock().unwrap().extend(again);
    }
    if changed {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The queue's page mappings changed `parts` (from, to) of `e` by call `call`. A texture read
/// through its pages sends them again with its next upload, as known changes with no comparison
/// (a clean pool page newly mapped holds bytes the PC texture never had): true, and it goes into
/// AGAIN. A shadowed one goes up whole once it is uncovered, as on a change of its source. The
/// GPU's write of a list submitted at or after that call stays; an older one gives way.
fn mapping_changed(e: &mut Entry, call: u64, parts: Vec<(usize, usize)>, active: bool) -> bool {
    if !e.paged || e.pending || (e.gpu_written && e.written_call >= call) {
        return false;
    }
    e.gpu_written = false;
    REREAD.fetch_add(1, Ordering::Relaxed);
    if !active {
        e.pending = true;
        PENDING.fetch_add(1, Ordering::Relaxed);
        return false;
    }
    texture_upload::stale(&e.known, parts);
    true
}

/// `e`'s source again (a texture in a GPU view): when it changed, `e` reads the new one and is
/// uploaded at the next submit. Whether it changed.
fn recheck_source(e: &mut Entry) -> bool {
    if !e.cpu_backed || !e.in_gpu_view {
        return false;
    }
    let (src, paged) = source_of(e.va, e.tex.size() as usize);
    if src == e.src && paged == e.paged {
        return false;
    }
    if paged && !e.paged {
        note_paged(e.va);
    }
    (e.src, e.paged) = (src, paged);
    if !e.pending {
        e.pending = true;
        PENDING.fetch_add(1, Ordering::Relaxed);
    }
    true
}

/// Prepare a recorded GPU write and capture the current registration generation. Lazy level
/// mappings must exist before submission; provenance itself is published only after submission.
pub(super) fn capture_gpu_write(res: usize, subresource: u32) -> Option<GpuWrite> {
    if super::vram::level_pages() {
        // (a texture whose levels get their pages one by one: all of them before the GPU's write)
        let p = res as *mut c_void;
        unsafe {
            if let Some(r) = ID3D12Resource::from_raw_borrowed(&p) {
                super::vram::map_levels(r, None);
            }
        }
    }
    let t = TEXTURES.lock().unwrap();
    let e = t.by_res.get(&res).or_else(|| t.shadowed.get(&res))?;
    Some(GpuWrite { res, generation: e.generation, subresource })
}

/// Publish writes from command-list segments that the native submission gate accepted. On the
/// console their memory now belongs to the GPU result, so its stale CPU view is not uploaded.
#[cfg(test)]
pub(super) fn publish_gpu_writes(queue: WriterQueueId, writes: &[GpuWrite]) {
    publish_submission_effects(Some(queue), writes, &[]);
}

/// Atomically publish a submitted chunk's GPU writers and ordered state transitions.
pub(super) fn publish_submission_effects(queue: Option<WriterQueueId>, writes: &[GpuWrite], transitions: &[Transition]) {
    if writes.is_empty() && transitions.is_empty() {
        return;
    }
    let mut newly_written = Vec::new();
    {
        let mut t = TEXTURES.lock().unwrap();
        // (the stamp is for the queue's page mappings: with them off, a publish takes no call count)
        let call = crate::hle::kernelx::mirrors::page_mappings().then(crate::hle::kernelx::mirrors::calls);
        if let Some(queue) = queue {
            for &write in writes {
                let Some(e) = t.get_mut(write.res).filter(|e| e.generation == write.generation) else { continue };
                let n = subresource_count(&e.tex);
                if write.subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES && !(n <= 1 && write.subresource == 0) && write.subresource as usize >= n {
                    continue;
                }
                let epoch = NEXT_GPU_EPOCH.fetch_add(1, Ordering::Relaxed) + 1;
                let writer = GpuWriter { epoch, queue };
                apply_gpu_writer(&mut e.gpu_writers, n, write.subresource, writer);
                if let Some(call) = call {
                    // (`Entry::written_call`: a mapping changed by a later call is newer than this write)
                    e.written_call = call;
                }
                if !e.gpu_written {
                    e.gpu_written = true;
                    GPU_WRITTEN.fetch_add(1, Ordering::Relaxed);
                    newly_written.push(write.res);
                }
            }
        }
        for &transition in transitions {
            let Some(e) = t.get_mut(transition.res).filter(|e| e.generation == transition.generation) else { continue };
            let n = subresource_count(&e.tex);
            if transition.subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES
                && !(n <= 1 && transition.subresource == 0)
                && transition.subresource as usize >= n
            {
                continue;
            }
            apply_transition(&mut e.states, n, transition.subresource, transition.after);
        }
    }
    if !newly_written.is_empty() {
        let mut recheck = STREAM_RECHECK.lock().unwrap();
        for res in newly_written {
            recheck.remove(&res);
        }
    }
}

static GPU_WRITTEN: AtomicU64 = AtomicU64::new(0);
static NEXT_GPU_EPOCH: AtomicU64 = AtomicU64::new(0);

/// The textures kept in step whose memory `[va, va + len)` overlaps (PC texture, its address).
pub fn overlapping(va: usize, len: usize) -> Vec<(usize, usize)> {
    TEXTURES.lock().unwrap().overlapping(va, len)
}

/// A GPU-owned placement (render target / depth / UAV) now occupies `[va, va + len)`: retire
/// superseded sampled textures there so CPU dirty scans do not upload over the shared tiles and
/// any unshared extra tiles return to the pool.
pub(super) unsafe fn retire_overlapping(va: usize, len: usize) {
    if !any() || len == 0 {
        return;
    }
    let src = mirror_source(va);
    let gpu_aliases: Vec<usize> = crate::hle::kernelx::mirrors::gpu_views_of(va)
        .into_iter()
        .filter(|&g| g != va && mirror_source(g) == va)
        .collect();
    let (retired, to_unmap) = {
        let mut t = TEXTURES.lock().unwrap();
        let (mut delta, mut retired, mut unmap_ptrs) = t.retire_range(va, va.saturating_add(len));
        if src != va {
            let (d2, r2, u2) = t.retire_range(src, src.saturating_add(len));
            delta += d2;
            retired += r2;
            unmap_ptrs.extend(u2);
        }
        for g in gpu_aliases {
            let (d3, r3, u3) = t.retire_range(g, g.saturating_add(len));
            delta += d3;
            retired += r3;
            unmap_ptrs.extend(u3);
        }
        adjust_pending(delta);
        update_lengths(&t);
        (retired, clone_resources(&unmap_ptrs))
    };
    for r in &to_unmap {
        super::vram::unmap_superseded(r);
    }
    if retired > 0 {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The PC texture was released for good.
pub fn forget(res: usize) {
    STREAM_RECHECK.lock().unwrap().remove(&res);
    let (removed, to_remap) = {
        let mut t = TEXTURES.lock().unwrap();
        let (removed, delta, remap_ptrs) = t.remove(res);
        adjust_pending(delta);
        update_lengths(&t);
        (removed, unsafe { clone_resources(&remap_ptrs) })
    };
    for r in &to_remap {
        unsafe {
            super::vram::remap_if_unmapped(r);
            // (lazy levels: one the GPU wrote is not uploaded again, so all of it has pages)
            if super::vram::level_pages() && TEXTURES.lock().unwrap().by_res.get(&(r.as_raw() as usize)).is_some_and(|e| e.gpu_written) {
                super::vram::map_levels(r, None);
            }
        }
    }
    if removed {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The title-memory address of a texture kept in step with it.
pub fn address(res: usize) -> Option<usize> {
    let t = TEXTURES.lock().unwrap();
    t.by_res.get(&res).or_else(|| t.shadowed.get(&res)).map(|e| e.va)
}

/// A recorded transition of a tracked texture. The registration generation keeps a captured
/// command-list effect from applying to a new resource at a reused COM address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Transition {
    pub(super) res: usize,
    pub(super) generation: u64,
    pub(super) subresource: u32,
    pub(super) after: D3D12_RESOURCE_STATES,
}

/// Whether any texture is kept in step (callers skip collecting transitions otherwise).
pub fn any() -> bool {
    TEXTURES_LEN.load(Ordering::Relaxed) != 0
}

/// Capture the transitions of textures currently kept in step while a physical segment records.
pub(super) fn capture_transitions(transitions: &[(usize, u32, D3D12_RESOURCE_STATES)]) -> Vec<Transition> {
    if !any() || transitions.is_empty() {
        return Vec::new();
    }
    let t = TEXTURES.lock().unwrap();
    transitions
        .iter()
        .filter_map(|&(res, subresource, after)| {
            let e = t.by_res.get(&res).or_else(|| t.shadowed.get(&res))?;
            Some(Transition { res, generation: e.generation, subresource, after })
        })
        .collect()
}

/// Subresources of a texture kept in step (as `subresources` makes them).
fn subresource_count(tex: &xg::Texture) -> usize {
    tex.texels().map_or(1, |s| s.array as usize * s.chain.levels.len())
}

/// Apply one writer to all subresources or one known subresource. The representation mirrors
/// `states`: one value while uniform, one per subresource while they differ.
fn apply_gpu_writer(writers: &mut Vec<Option<GpuWriter>>, n: usize, sub: u32, writer: GpuWriter) {
    if sub == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES || (n <= 1 && sub == 0) {
        *writers = vec![Some(writer)];
        return;
    }
    let k = sub as usize;
    if writers.is_empty() {
        *writers = vec![None; n];
    } else if writers.len() != n {
        *writers = vec![writers[0]; n];
    }
    writers[k] = Some(writer);
    if writers.iter().all(|&w| w == Some(writer)) {
        writers.truncate(1);
    }
}

/// `states` (one for all `n` subresources, or one each) after a transition of subresource `sub`
/// (or all of them) to `after`; a subresource the texture does not have is ignored.
pub(super) fn apply_transition(states: &mut Vec<D3D12_RESOURCE_STATES>, n: usize, sub: u32, after: D3D12_RESOURCE_STATES) {
    if sub == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES || (n <= 1 && sub == 0) {
        *states = vec![after];
        return;
    }
    let k = sub as usize;
    if k >= n {
        return;
    }
    if states.len() != n {
        *states = vec![states[0]; n];
    }
    states[k] = after;
    if states.iter().all(|&s| s == after) {
        states.truncate(1);
    }
}

thread_local! {
    /// XCL's own submits on this thread (texture uploads): they read no title memory on the GPU,
    /// so they do not look for CPU writes first – that look is the title's submits' job
    static BUSY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Reusable `GetWriteWatch` output buffer (avoids 128 KiB alloc+zero+free per submit).
    static WATCH_BUF: std::cell::RefCell<Vec<*mut c_void>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Reusable dirty page list for `refresh_once`.
    static DIRTY_BUF: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Reusable list of the GPU-view pages the dirty pages are written to (`views_of_written`).
    static TWINS_BUF: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// What the watched ranges are made of, as generations.
fn watch_key() -> (u64, u64, u64, u64) {
    (super::sparse::mapped_gen(), COARSE_GEN.load(Ordering::Acquire), crate::hle::kernelx::graphics_reservations_gen(), crate::hle::kernelx::mirrors::shown_gen())
}

/// The ranges the last look used, and whether they are still all there is.
fn last_watch_ranges() -> Option<(Arc<Vec<(usize, usize)>>, bool)> {
    let g = WATCH_CACHE.lock().unwrap();
    g.as_ref().map(|(k, ranges)| (Arc::clone(ranges), *k == watch_key()))
}

#[cfg_attr(not(test), allow(dead_code))]
fn watched_texture_span(src: usize, bytes: u64) -> Option<(usize, usize)> {
    let bytes = usize::try_from(bytes).ok()?;
    if bytes == 0 {
        return None;
    }
    let end = src.checked_add(bytes)?.checked_add(0xFFF)? & !0xFFF;
    Some((src & !0xFFF, end))
}

#[inline(always)]
fn texture_needs_upload(
    cpu_backed: bool,
    pending: bool,
    gpu_written: bool,
    dirty_overlap: impl FnOnce() -> bool,
) -> bool {
    if !cpu_backed {
        return false;
    }
    if pending {
        return true;
    }
    if gpu_written {
        return false;
    }
    dirty_overlap()
}

/// What the watch ranges last made cost, for the write-watch trace (`note_watch_measurement`):
/// the bytes of what is wanted itself (`plan_watch_ranges` with no call budget; 0 when the ranges
/// were made without `measure`) and the bytes watched beyond it (gaps, whole blocks).
static WATCH_PLAN: Mutex<(usize, usize)> = Mutex::new((0, 0));

fn cached_watch_ranges(u: &crate::gpu::uma::Uma, measure: bool) -> Arc<Vec<(usize, usize)>> {
    let key = watch_key();
    {
        let g = WATCH_CACHE.lock().unwrap();
        if let Some((k, ref ranges)) = *g {
            if k == key {
                return Arc::clone(ranges);
            }
        }
    }
    // Recompute the watched ranges (textures kept in step + mapped sparse tiles clipped to
    // write-watched regions) only when the set of textures, tiles, or reservations changed.
    let mut regions = u.watch_regions();
    regions.extend(crate::hle::kernelx::graphics_reservations());
    // (by 16 MiB blocks, not by texture: a title has tens of thousands, and their sources change
    // every submit while it loads)
    let mut wanted: Vec<(usize, usize)> = TEXTURES.lock().unwrap().coarse.keys().map(|&b| (b << SRC_COARSE_SHIFT, (b + 1) << SRC_COARSE_SHIFT)).collect();
    wanted.extend(super::sparse::mapped_ranges());
    // (the blocks where the pages the queue's page mappings show are kept: the pool, the CPU view)
    wanted.extend(crate::hle::kernelx::mirrors::shown_blocks());
    let exact = measure.then(|| plan_watch_ranges(&regions, wanted.clone(), usize::MAX));
    let computed = watch_ranges(&regions, wanted);
    let wanted_bytes = exact.as_ref().map_or(0, |exact| exact.iter().fold(0usize, |bytes, &(_, n)| bytes.saturating_add(n)));
    let scanned_bytes = computed.iter().fold(0usize, |bytes, &(_, n)| bytes.saturating_add(n));
    let mut g = WATCH_CACHE.lock().unwrap();
    *WATCH_PLAN.lock().unwrap() = (wanted_bytes, scanned_bytes.saturating_sub(wanted_bytes));
    // (ranges that came out the same stay the same object: `hot_runs` keeps its blocks by it)
    let computed = match g.take() {
        Some((_, old)) if *old == computed => old,
        _ => Arc::new(computed),
    };
    *g = Some((key, Arc::clone(&computed)));
    computed
}

/// Run `f` as XCL's own GPU work: ExecuteCommandLists inside it does not refresh.
pub(super) fn own_work<R>(f: impl FnOnce() -> R) -> R {
    let was = BUSY.with(|b| b.replace(true));
    let r = f();
    BUSY.with(|b| b.set(was));
    r
}

/// Whether the current thread is inside XCL's own internal GPU setup/upload/readback work.
pub(super) fn is_own_work() -> bool {
    BUSY.with(|b| b.get())
}

/// Before the title's GPU work: textures whose memory the CPU wrote since the last look are
/// uploaded again.
pub(super) unsafe fn refresh(queue: *mut c_void) -> super::sparse::PreparationOutcome {
    // the uploads' own ExecuteCommandLists come back here
    if BUSY.with(|b| b.get()) {
        return super::sparse::PreparationOutcome::Ready;
    }
    let measure = super::memory_report::enabled();
    let wait_t0 = measure.then(std::time::Instant::now);
    let _refresh = REFRESH.lock().unwrap();
    let active_t0 = measure.then(std::time::Instant::now);
    let mut look = measure.then(super::sparse::LookSample::default);
    let outcome = own_work(|| refresh_once(queue, true, look.as_mut()));
    if let (Some(wait_t0), Some(active_t0), Some(look)) = (wait_t0, active_t0, look) {
        super::sparse::note_look_sample(look, active_t0.elapsed(), active_t0.duration_since(wait_t0));
    }
    outcome
}

/// microseconds since the first call, never 0
fn now_us() -> u64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64 + 1
}

/// `(began_us, frame)` of the last look for CPU writes (`began_us == 0`: none yet; `frame` is
/// `present::activity().0` when the look began so every new frame takes at least one fresh look).
static LAST_LOOK: Mutex<(u64, u64)> = Mutex::new((0, u64::MAX));
/// title list -> when the title closed it (`now_us`)
static CLOSED: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);

fn last_look() -> (u64, u64) {
    *LAST_LOOK.lock().unwrap()
}

fn advance_look(state: &mut (u64, u64), began: u64, frame: u64) {
    if state.0 == 0 || frame > state.1 {
        state.0 = began;
        state.1 = frame;
    } else if frame == state.1 {
        state.0 = state.0.max(began);
    }
}

/// Record that a write-watch look began at `began` (`now_us`) during `frame`.
/// Monotonic in `(frame, began)` so an older `XCL_LOOK_AHEAD` round never rewinds a newer frame.
fn note_look_began(began: u64, frame: u64) {
    let mut g = LAST_LOOK.lock().unwrap();
    advance_look(&mut g, began, frame);
}

/// The title closed `list` (its own pointer): what the list's GPU work reads was written by now.
pub(super) fn note_list_closed(list: usize) {
    CLOSED.lock().unwrap().get_or_insert_with(Default::default).insert(list, now_us());
}

/// Clear any recorded close time for `list` (reset, reopened via `KickoffX`, or released) so a
/// reopened or pointer-reused list never retains a stale `t < LOOK_BEGAN` timestamp.
pub(super) fn forget_list_closed(list: usize) {
    let mut g = CLOSED.lock().unwrap();
    if let Some(m) = g.as_mut() {
        if !m.is_empty() {
            m.remove(&list);
        }
    }
}

#[cfg(test)]
pub(super) fn is_list_closed_for_test(list: usize) -> bool {
    CLOSED.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&list))
}

/// `XCL_LOOK_SKIP=1`: a submit in the same frame as the last look whose lists were all closed
/// before that look began goes without a look of its own (that look took what the lists read).
/// Every new frame still guarantees at least one fresh look. Off unless set: a title may write
/// what a closed list reads until it submits the list.
fn look_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_LOOK_SKIP").is_ok_and(|v| v.trim() == "1"))
}

/// when the walk last made a block hot (`now_us`): no look before that took the block
static WALK_FOUND: AtomicU64 = AtomicU64::new(0);

fn closed_before_last_look_in(lists: &[*mut c_void], began: u64, look_frame: u64, current_frame: u64, walk_found: u64) -> bool {
    if began == 0 || lists.is_empty() {
        return false;
    }
    // Every new frame guarantees at least one fresh write-watch look.
    if look_frame != current_frame {
        return false;
    }
    // (a block the walk found written since: the look that began then did not take it)
    if walk_found >= began {
        return false;
    }
    let g = CLOSED.lock().unwrap();
    let Some(m) = g.as_ref() else { return false };
    lists.iter().all(|&l| m.get(&(l as usize)).is_some_and(|&t| t < began))
}

fn closed_before_last_look(lists: &[*mut c_void]) -> bool {
    let (began, look_frame) = last_look();
    let current_frame = super::present::activity().0;
    let walk_found = WALK_FOUND.load(Ordering::Acquire);
    closed_before_last_look_in(lists, began, look_frame, current_frame, walk_found)
}

#[inline]
fn should_look(all_closed_before_last_look: bool, skip_enabled: bool, presenting: bool) -> bool {
    !(all_closed_before_last_look && skip_enabled) && !presenting
}

/// When the youngest of `lists` was closed (None: one was not seen closing).
fn youngest_closed(lists: &[*mut c_void]) -> Option<u64> {
    let g = CLOSED.lock().unwrap();
    let m = g.as_ref()?;
    lists.iter().map(|&l| m.get(&(l as usize)).copied()).try_fold(0, |a, t| Some(t?.max(a)))
}

thread_local! {
    /// diagnostic: how long ago the youngest list of this thread's submit was closed (µs)
    static CLOSED_AGO: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// `refresh` before the title's ExecuteCommandLists of `lists` (its own pointers).
pub(super) unsafe fn refresh_for(queue: *mut c_void, lists: &[*mut c_void]) -> super::sparse::PreparationOutcome {
    if BUSY.with(|b| b.get()) {
        return super::sparse::PreparationOutcome::Ready;
    }
    let measure = super::memory_report::enabled();
    let t0 = std::time::Instant::now();
    let _refresh = REFRESH.lock().unwrap();
    let locked = std::time::Instant::now();
    super::submit_cost::lock_wait(locked.duration_since(t0));
    // with the lock: the look before this one is done, and its uploads are submitted
    let old = closed_before_last_look(lists);
    static N: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    let n = N[0].fetch_add(1, Ordering::Relaxed) + 1;
    let k = N[1].fetch_add(old as u64, Ordering::Relaxed) + old as u64;
    if n % 1024 == 0 {
        crate::xlog_info!(
            "D3D12.X: {k} of {n} ExecuteCommandLists so far had all their lists closed before the last look for CPU writes began ({})",
            if look_skip() { "their looks were left out" } else { "XCL_LOOK_SKIP=1 leaves their looks out" }
        );
    }
    let t1 = std::time::Instant::now();
    // (XCL's copy of the frame to its window reads no memory of the title)
    let look = should_look(old, look_skip(), super::present::presenting());
    if look && super::look_stats::on() {
        CLOSED_AGO.with(|c| c.set(youngest_closed(lists).map(|t| now_us().saturating_sub(t))));
    }
    let mut sample = measure.then(super::sparse::LookSample::default);
    let outcome = own_work(|| refresh_once(queue, look, sample.as_mut()));
    super::submit_cost::refresh(t1.elapsed());
    if let Some(sample) = sample {
        super::sparse::note_look_sample(sample, locked.elapsed(), locked.duration_since(t0));
    }
    outcome
}

/// The pages of `[base, base + size)` written since the last look, added to `dirty` and reset
/// (`page`: their size). A full `buf` goes on after the last page it reported: starting again at
/// `base` walked the pages already reset each time (4 GiB with every page written: 105 ms a
/// look, 47 ms going on).
#[derive(Default)]
struct WatchMeasurement {
    calls: u64,
    elapsed_ns: u64,
}

unsafe fn take_written(
    base: usize,
    size: usize,
    buf: &mut [*mut c_void],
    dirty: &mut Vec<usize>,
    page: &mut u32,
    mut measurement: Option<&mut WatchMeasurement>,
) -> bool {
    let Some(end) = base.checked_add(size) else { return false };
    let mut at = base;
    // (`XCL_LOOK_STATS=2`: each call timed, for the look split)
    let split = super::look_stats::deep();
    while at < end {
        let mut n = buf.len();
        let t = (split || measurement.is_some()).then(std::time::Instant::now);
        let failed = GetWriteWatch(WRITE_WATCH_FLAG_RESET, at as *const c_void, end - at, buf.as_mut_ptr(), &mut n, page) != 0;
        if let Some(t) = t {
            let took = t.elapsed();
            if split {
                super::look_stats::watch_call(took, end - at, if failed { 0 } else { n });
            }
            if let Some(m) = measurement.as_mut() {
                m.calls += 1;
                m.elapsed_ns = m.elapsed_ns.saturating_add(took.as_nanos().min(u64::MAX as u128) as u64);
            }
        }
        if failed {
            return false;
        }
        dirty.extend(buf[..n].iter().map(|&p| p as usize));
        if n == 0 || n < buf.len() {
            break;
        }
        let Some(next) = (buf[n - 1] as usize).checked_add(*page as usize) else { return false };
        at = next;
    }
    true
}

fn note_watch_measurement(ranges: &[(usize, usize)], measurement: &WatchMeasurement) {
    static TOTALS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
    let (wanted_bytes, excess_bytes) = *WATCH_PLAN.lock().unwrap();
    let values = [
        1,
        wanted_bytes as u64,
        excess_bytes as u64,
        ranges.len() as u64,
        measurement.calls,
        measurement.elapsed_ns,
    ];
    let mut totals = [0u64; 6];
    for (i, value) in values.into_iter().enumerate() {
        totals[i] = TOTALS[i].fetch_add(value, Ordering::Relaxed).saturating_add(value);
    }
    if totals[0].is_power_of_two() {
        crate::xlog_info!(
            "write-watch trace: {} looks, {} MB wanted + {} MB gaps, {} planned / {} actual calls, {} us in GetWriteWatch",
            totals[0], totals[1] >> 20, totals[2] >> 20, totals[3], totals[4], totals[5] / 1000
        );
    }
}

/// `XCL_WATCH_HOT=0`: every look before a submit walks all the watched memory, as before.
/// Otherwise a look takes the 2 MiB blocks written in the last few seconds only, and a thread of
/// its own walks the rest in small steps and makes a block hot when it finds a written page there
/// (within ~10 ms; its pages stay reported until a look takes them). A look over 8 GiB of
/// committed memory took 4.3-4.9 ms however few pages were written, on one thread or split over
/// sixteen (measured, Windows 11 26300); 64 looks of 2 MiB took 0.2 ms. A retail title's world
/// made ~66 submits a frame on one thread, each behind a 7 ms look over ~11 GiB.
fn hot_looks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_WATCH_HOT").map_or(true, |v| v.trim() != "0"))
}

const HOT_BLOCK: usize = 2 << 20;
/// the gap up to which hot blocks go into one write-watch call (`XCL_WATCH_GAP_MB`, 64 unless
/// given): a call waits for the process's memory lock, which is busy while the title loads, and
/// costs more than walking some megabytes of pages
fn hot_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| std::env::var("XCL_WATCH_GAP_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(64) << 20)
}
/// a block nothing was written to for this long is left to the walk again (`XCL_WATCH_HOT_MS`,
/// 1500 unless given)
fn hot_for() -> std::time::Duration {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    std::time::Duration::from_millis(*MS.get_or_init(|| std::env::var("XCL_WATCH_HOT_MS").ok().and_then(|v| v.trim().parse().ok()).filter(|&v| v >= 20).unwrap_or(1500)))
}

#[derive(Default)]
struct HotBlocks {
    /// block -> when a look or the walk last found a written page in it
    hot: BTreeMap<usize, std::time::Instant>,
    /// the ranges `known` was made for
    ranges: Option<Arc<Vec<(usize, usize)>>>,
    /// the blocks of those ranges (a block new to the ranges starts hot)
    known: std::collections::HashSet<usize>,
}

static HOT: Mutex<Option<HotBlocks>> = Mutex::new(None);
/// the ranges of the last look, for the walk
static WALK_RANGES: Mutex<Option<Arc<Vec<(usize, usize)>>>> = Mutex::new(None);
/// a look is at its write-watch calls: the walk stays out of the way (the kernel does one such
/// call at a time; with the walk going all the time, a look of 45 calls waited behind it for
/// ~1.3 ms in all)
static LOOKING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// looks, runs they took, bytes they covered, blocks the walk made hot
static HOT_STATS: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// The hot blocks of `ranges` as runs, each inside one range (a write-watch call covers one
/// allocation), and how many blocks are hot.
fn hot_runs(ranges: &Arc<Vec<(usize, usize)>>, now: std::time::Instant) -> (Vec<(usize, usize)>, usize) {
    let mut g = HOT.lock().unwrap();
    let h = g.get_or_insert_with(HotBlocks::default);
    // a block new to the ranges starts hot: memory just mapped is often written and used at once
    if !h.ranges.as_ref().is_some_and(|r| Arc::ptr_eq(r, ranges)) {
        let mut known = std::collections::HashSet::with_capacity(h.known.len());
        for &(base, size) in ranges.iter() {
            let mut b = base & !(HOT_BLOCK - 1);
            while b < base + size {
                if !h.known.contains(&b) {
                    h.hot.insert(b, now);
                }
                known.insert(b);
                b += HOT_BLOCK;
            }
        }
        h.known = known;
        h.ranges = Some(Arc::clone(ranges));
    }
    let keep = hot_for();
    h.hot.retain(|_, t| now.saturating_duration_since(*t) < keep);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (&b, _) in h.hot.iter() {
        let mut j = ranges.partition_point(|&(base, size)| base + size <= b);
        while let Some(&(base, size)) = ranges.get(j) {
            if base >= b + HOT_BLOCK {
                break;
            }
            let (lo, hi) = (b.max(base), (b + HOT_BLOCK).min(base + size));
            if lo < hi {
                match runs.last_mut() {
                    // one call for hot blocks up to HOT_GAP apart in the same range: a call costs
                    // about as much as walking 6 MiB of committed pages
                    Some(last) if last.0 >= base && lo - (last.0 + last.1) <= hot_gap() => last.1 = hi - last.0,
                    _ => runs.push((lo, hi - lo)),
                }
            }
            j += 1;
        }
    }
    (runs, h.hot.len())
}

/// The blocks of `pages`, which a look found written, are hot from `now` on.
fn keep_hot(pages: &[usize], now: std::time::Instant) {
    if pages.is_empty() {
        return;
    }
    if let Some(h) = HOT.lock().unwrap().as_mut() {
        let mut last = usize::MAX;
        for &p in pages {
            let b = p & !(HOT_BLOCK - 1);
            if b != last {
                h.hot.insert(b, now);
                last = b;
            }
        }
    }
}

/// `XCL_LOOK_AHEAD=ms` (off unless given): a thread takes the written pages of the hot blocks every
/// so many milliseconds while the title submits. They are dirty by XCL's account from then on
/// (`AHEAD`): the next submit takes them with what its own look finds, and one whose lists were
/// closed before such a round began needs no look (`XCL_LOOK_SKIP`). A look's time goes up with the
/// pages it takes – each is a page-table entry changed and flushed on every core, ~0.5 µs with the
/// title's threads running – and a retail title's world had ~8,000 a look: 5 ms on the thread that
/// submits.
fn look_ahead() -> Option<std::time::Duration> {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *MS.get_or_init(|| std::env::var("XCL_LOOK_AHEAD").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0));
    (ms > 0 && hot_looks()).then(|| std::time::Duration::from_millis(ms))
}

/// pages found ahead of the next submit
static AHEAD: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// one look, or a piece of the thread's, at a time: what the one before was told is in `AHEAD` or in
/// its submit's hands when the next one asks
static ONE_LOOK: Mutex<()> = Mutex::new(());
/// when the last submit came (`now_us`; the thread rests while there are none)
static LAST_SUBMIT: AtomicU64 = AtomicU64::new(0);

/// The written pages of `[at, at + len)` taken into `ahead`: how many.
unsafe fn take_piece(at: usize, len: usize, buf: &mut [*mut c_void], ahead: &Mutex<Vec<usize>>) -> usize {
    let (mut found, mut page) = (Vec::new(), 0x1000u32);
    {
        let _one = ONE_LOOK.lock().unwrap_or_else(|e| e.into_inner());
        LOOKING.store(true, Ordering::Release);
        take_written(at, len, buf, &mut found, &mut page, None);
        LOOKING.store(false, Ordering::Release);
        ahead.lock().unwrap().extend_from_slice(&found);
    }
    keep_hot(&found, std::time::Instant::now());
    found.len()
}

/// The written pages of all hot blocks, in pieces (a submit's look waits for one at most).
unsafe fn take_round(buf: &mut [*mut c_void]) -> usize {
    const PIECE: usize = 64 << 20;
    // (frame before `began`: guarantees `began` is never earlier than `frame`'s start,
    // and both precede asking for ranges so memory new after that is not in the round)
    let frame = super::present::activity().0;
    let began = now_us();
    let Some((ranges, all)) = last_watch_ranges() else { return 0 };
    let mut n = 0;
    for (base, size) in hot_runs(&ranges, std::time::Instant::now()).0 {
        for at in (base..base + size).step_by(PIECE) {
            n += take_piece(at, PIECE.min(base + size - at), buf, &AHEAD);
        }
    }
    // (memory new since the last look: that one's ranges are made again by the next submit's, and
    // it does not stand for this round)
    if all {
        note_look_began(began, frame);
    }
    n
}

fn start_look_ahead() {
    let Some(pause) = look_ahead() else { return };
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        crate::xlog_info!("D3D12.X: XCL_LOOK_AHEAD: a thread takes the written pages every {} ms between the submits", pause.as_millis());
        let _ = std::thread::Builder::new().name("xcl-look-ahead".into()).spawn(move || {
            let mut buf: Vec<*mut c_void> = vec![std::ptr::null_mut(); 16384];
            let (mut rounds, mut pages, mut us, mut longest) = (0u64, 0u64, 0u64, 0u64);
            loop {
                std::thread::sleep(pause);
                if now_us().saturating_sub(LAST_SUBMIT.load(Ordering::Relaxed)) > 200_000 {
                    continue;
                }
                let t = std::time::Instant::now();
                let n = unsafe { take_round(&mut buf) };
                let took = t.elapsed().as_micros() as u64;
                (rounds, pages, us, longest) = (rounds + 1, pages + n as u64, us + took, longest.max(took));
                if rounds % 4096 == 0 {
                    crate::xlog_info!(
                        "D3D12.X: {rounds} rounds ahead of the submits so far; the last 4096: {} pages and {:.2} ms each on average, the longest {:.1} ms",
                        pages / 4096,
                        us as f64 / 4096e3,
                        longest as f64 / 1e3
                    );
                    (pages, us, longest) = (0, 0, 0);
                }
            }
        });
    });
}

/// The written pages of the hot blocks inside `ranges` (see `hot_looks`).
unsafe fn take_hot(
    ranges: &Arc<Vec<(usize, usize)>>,
    buf: &mut [*mut c_void],
    dirty: &mut Vec<usize>,
    page: &mut u32,
    mut measurement: Option<&mut WatchMeasurement>,
) {
    start_walk(ranges);
    let now = std::time::Instant::now();
    let (runs, hot_now) = hot_runs(ranges, now);
    let first = dirty.len();
    LOOKING.store(true, Ordering::Release);
    for &(base, size) in &runs {
        take_written(base, size, buf, dirty, page, measurement.as_deref_mut());
    }
    LOOKING.store(false, Ordering::Release);
    keep_hot(&dirty[first..], now);
    let n = HOT_STATS[0].fetch_add(1, Ordering::Relaxed) + 1;
    let r = HOT_STATS[1].fetch_add(runs.len() as u64, Ordering::Relaxed) + runs.len() as u64;
    let covered: usize = runs.iter().map(|r| r.1).sum();
    let bytes = HOT_STATS[2].fetch_add(covered as u64, Ordering::Relaxed) + covered as u64;
    if n.is_power_of_two() && n >= 256 {
        let watched: usize = ranges.iter().map(|r| r.1).sum();
        crate::xlog_info!(
            "D3D12.X: looks for CPU writes: {n} so far, {} runs and {} MB a look on average ({} MB watched in {} ranges); {hot_now} blocks hot now, {} made hot by the walk",
            r / n,
            (bytes / n) >> 20,
            watched >> 20,
            ranges.len(),
            HOT_STATS[3].load(Ordering::Relaxed)
        );
    }
}

fn start_walk(ranges: &Arc<Vec<(usize, usize)>>) {
    {
        let mut g = WALK_RANGES.lock().unwrap();
        if !g.as_ref().is_some_and(|r| Arc::ptr_eq(r, ranges)) {
            *g = Some(Arc::clone(ranges));
        }
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-watch-walk".into()).spawn(|| unsafe { walk() });
        crate::xlog_info!(
            "D3D12.X: a look for CPU writes takes the blocks written in the last {} ms; a thread walks the rest (XCL_WATCH_HOT=0: every look walks everything)",
            hot_for().as_millis()
        );
    });
}

/// The walk: all watched memory in steps of 64 MiB, without taking the written pages (they stay
/// reported until a look takes them); a block with a written page is hot from then on.
unsafe fn walk() {
    const STEP: usize = 32 << 20;
    let mut buf: Vec<*mut c_void> = vec![std::ptr::null_mut(); 8192];
    let mut found: Vec<usize> = Vec::new();
    loop {
        let ranges = WALK_RANGES.lock().unwrap().clone();
        let Some(ranges) = ranges else {
            std::thread::sleep(std::time::Duration::from_millis(5));
            continue;
        };
        // (`XCL_LOOK_STATS=2`: how long a pass takes – how long a write to a block no look takes
        // may wait to be found)
        let pass = super::look_stats::deep().then(std::time::Instant::now);
        for &(base, size) in ranges.iter() {
            let end = base + size;
            let mut at = base;
            while at < end {
                let stop = (at + STEP).min(end);
                let mut from = at;
                while LOOKING.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                while from < stop {
                    let (mut n, mut page) = (buf.len(), 0u32);
                    if GetWriteWatch(0, from as *const c_void, stop - from, buf.as_mut_ptr(), &mut n, &mut page) != 0 || n == 0 {
                        break;
                    }
                    let mut last = usize::MAX;
                    for &p in &buf[..n] {
                        let b = p as usize & !(HOT_BLOCK - 1);
                        if b != last {
                            found.push(b);
                            last = b;
                        }
                    }
                    if n < buf.len() {
                        break;
                    }
                    // go on after the block of the last page reported
                    from = (buf[n - 1] as usize & !(HOT_BLOCK - 1)) + HOT_BLOCK;
                }
                if !found.is_empty() {
                    let now = std::time::Instant::now();
                    let mut g = HOT.lock().unwrap();
                    let h = g.get_or_insert_with(HotBlocks::default);
                    for b in found.drain(..) {
                        if let std::collections::btree_map::Entry::Vacant(v) = h.hot.entry(b) {
                            v.insert(now);
                            HOT_STATS[3].fetch_add(1, Ordering::Relaxed);
                            WALK_FOUND.fetch_max(now_us(), Ordering::Release);
                        }
                    }
                }
                at = stop;
            }
        }
        if let Some(pass) = pass {
            super::look_stats::walk_pass(pass.elapsed());
        }
        // a pass takes the process's memory lock some hundred times (a call each); the pause keeps
        // the thread from holding up the title's own calls
        std::thread::sleep(std::time::Duration::from_millis(15));
    }
}

/// Ranges of the title's memory XCL wrote itself (DirectStorage data) since the last look: a
/// title's loads write gigabytes that way, and the write watch would find the pages one look after
/// the other (each block of them stayed hot for `hot_for`, and every look walked it).
static KNOWN: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// `XCL_DS_KNOWN_WRITES=0`: what DirectStorage writes is found by the write watch, as before.
fn known_writes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DS_KNOWN_WRITES").map_or(true, |v| v.trim() != "0"))
}

/// XCL wrote `[at, at + len)` of the title's memory itself (DirectStorage data). Called once the
/// bytes are there and before the title can learn of it. The write watch is cleared for the pages
/// inside the range (written whole by this, and by nobody else while the request ran); the pages
/// at its ends stay in it, for a neighbour's writes.
pub fn note_known_write(at: usize, len: usize) {
    if len == 0 || !known_writes() || crate::gpu::uma::get().is_none() {
        return;
    }
    let (lo, hi) = ((at + 0xFFF) & !0xFFF, (at + len) & !0xFFF);
    if hi > lo {
        // (not write-watched memory: nothing to clear)
        unsafe { ResetWriteWatch(lo as *const c_void, hi - lo) };
    }
    KNOWN.lock().unwrap().push((at, len));
}

#[cfg(test)]
pub(super) fn known_in(at: usize, len: usize) -> Vec<(usize, usize)> {
    KNOWN.lock().unwrap().iter().copied().filter(|r| r.0 >= at && r.0 < at + len).collect()
}

#[cfg(test)]
pub(super) fn forget_known_in(at: usize, len: usize) {
    KNOWN.lock().unwrap().retain(|r| r.0 < at || r.0 >= at + len);
}

/// The pages of the known writes, added to `dirty`.
fn take_known(dirty: &mut Vec<usize>) {
    let ranges = std::mem::take(&mut *KNOWN.lock().unwrap());
    for (at, len) in ranges {
        let mut p = at & !0xFFF;
        while p < at + len {
            dirty.push(p);
            p += 0x1000;
        }
    }
}

/// Memory the title just committed or placed something in: the looks take its blocks for a
/// while, as they do a block new to the ranges (a write made right after the commit is there
/// for the next submit).
pub(super) fn mark_hot(at: usize, len: usize) {
    if len == 0 || !hot_looks() {
        return;
    }
    let mut g = HOT.lock().unwrap_or_else(|e| e.into_inner());
    // (nothing looked yet: blocks new to the ranges start hot)
    let Some(h) = g.as_mut() else { return };
    let now = std::time::Instant::now();
    let mut b = at & !(HOT_BLOCK - 1);
    while b < at + len {
        h.hot.insert(b, now);
        b += HOT_BLOCK;
    }
}

/// Diagnostic: where the written pages of the looks are, in 64 MiB pieces. The 12 with the most
/// pages are logged every 4096 looks, each with its UMA window and that window's kind.
fn note_dirty_regions(u: &crate::gpu::uma::Uma) {
    static COUNTS: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
    static LOOKS: AtomicU64 = AtomicU64::new(0);
    const PIECE: usize = 64 << 20;
    DIRTY_BUF.with(|d| {
        let d = d.borrow();
        if d.is_empty() {
            return;
        }
        let mut g = COUNTS.lock().unwrap();
        let m = g.get_or_insert_with(Default::default);
        let (mut last, mut n) = (usize::MAX, 0u64);
        for &p in d.iter() {
            let b = p & !(PIECE - 1);
            if b != last {
                if n != 0 {
                    *m.entry(last).or_insert(0) += n;
                }
                last = b;
                n = 0;
            }
            n += 1;
        }
        if n != 0 {
            *m.entry(last).or_insert(0) += n;
        }
    });
    let looks = LOOKS.fetch_add(1, Ordering::Relaxed) + 1;
    if looks % 2048 == 0 {
        let mut g = COUNTS.lock().unwrap();
        if let Some(m) = g.as_mut() {
            let total: u64 = m.values().sum();
            let mut v: Vec<(usize, u64)> = m.iter().map(|(&a, &c)| (a, c)).collect();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = v
                .iter()
                .take(12)
                .map(|&(a, c)| {
                    let w = u.window_of(a);
                    let kind = match w {
                        Some(k) if u.is_committed_window(k) => "committed",
                        Some(k) if u.is_view_window(k) => "GPU view",
                        Some(_) => "sparse",
                        None => "outside",
                    };
                    format!("{a:#x} ({kind}{}): {} %", w.map_or(String::new(), |k| format!(" window {k}")), c * 100 / total.max(1))
                })
                .collect();
            crate::xlog_info!("D3D12.X: written pages of the last 2048 looks by 64 MiB piece ({total} pages in {} pieces): {}", v.len(), top.join(", "));
            m.clear();
        }
    }
}

/// `refresh_parts`, then the title's queue waits for XCL's own uploads it has not waited for.
/// `look`: take the pages written since the last look (not with `XCL_LOOK_SKIP`'s skipped ones);
/// `sample`: the look's counters (`memory_report`).
unsafe fn refresh_once(queue: *mut c_void, look: bool, sample: Option<&mut super::sparse::LookSample>) -> super::sparse::PreparationOutcome {
    let outcome = refresh_parts(queue, look, sample);
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        wait_for_own_uploads(q);
    }
    outcome
}

unsafe fn refresh_parts(queue: *mut c_void, look: bool, mut sample: Option<&mut super::sparse::LookSample>) -> super::sparse::PreparationOutcome {
    let Some(u) = crate::gpu::uma::get() else { return super::sparse::PreparationOutcome::Ready };
    let sparse = u.has_sparse();
    if !sparse && CPU_TEXTURES_LEN.load(Ordering::Relaxed) == 0 {
        return super::sparse::PreparationOutcome::Ready;
    }
    let t_recheck = std::time::Instant::now();
    recheck_gpu_view_sources();
    retry_failed();
    super::file_reads::take_finished();
    super::submit_cost::recheck(t_recheck.elapsed());
    let mut page = 0x1000u32;
    let t_ranges = std::time::Instant::now();
    LAST_SUBMIT.store(now_us(), Ordering::Relaxed);
    // (frame before `began`, and both before ranges: memory new after that is not in this look)
    let frame = super::present::activity().0;
    let began = now_us();
    let ranges = cached_watch_ranges(u, sample.is_some());
    // (also of a submit without a look: memory that is new is walked from now on)
    if hot_looks() {
        start_walk(&ranges);
    }
    super::submit_cost::ranges(t_ranges.elapsed());
    let watch_t0 = std::time::Instant::now();
    let mut ahead = 0usize;
    let mut measurement = sample.is_some().then(WatchMeasurement::default);
    let dirty_len = DIRTY_BUF.with(|dcell| {
        let mut dirty = dcell.borrow_mut();
        dirty.clear();
        // (with a look of its own: what the thread's piece in progress finds is there first)
        let _one = look.then(|| ONE_LOOK.lock().unwrap_or_else(|e| e.into_inner()));
        // (a submit without a look of its own takes these too: they are dirty by XCL's account)
        take_known(&mut dirty);
        {
            let mut found = AHEAD.lock().unwrap();
            ahead = found.len();
            dirty.append(&mut found);
        }
        if !look {
            dirty.sort_unstable();
            dirty.dedup();
            return dirty.len();
        }
        if super::look_stats::deep() {
            // (the walk made a block hot since the look before began: written up to a walk pass
            // ago, and only this look takes it)
            let before = last_look().0;
            super::look_stats::walk_stale(before != 0 && WALK_FOUND.load(Ordering::Acquire) >= before);
            super::sparse::note_look();
        }
        note_look_began(began, frame);
        WATCH_BUF.with(|wcell| {
            let mut buf = wcell.borrow_mut();
            if buf.len() < 16384 {
                buf.resize(16384, std::ptr::null_mut());
            }
            if hot_looks() {
                start_look_ahead();
                take_hot(&ranges, &mut buf, &mut dirty, &mut page, measurement.as_mut());
            } else {
                for &(base, size) in ranges.iter() {
                    take_written(base, size, &mut buf, &mut dirty, &mut page, measurement.as_mut());
                }
            }
        });
        dirty.sort_unstable();
        dirty.dedup();
        dirty.len()
    });
    if look {
        super::sparse::note_watch(watch_t0.elapsed(), dirty_len);
        note_dirty_regions(u);
        DIRTY_BUF.with(|d| super::look_stats::look(&d.borrow(), ahead, watch_t0.elapsed(), CLOSED_AGO.with(|c| c.take())));
        if let Some(measurement) = &measurement {
            note_watch_measurement(&ranges, measurement);
        }
    }
    if let Some(sample) = sample.as_deref_mut() {
        super::sparse::note_watch_sample(
            sample,
            watch_t0.elapsed(),
            dirty_len,
            ranges.iter().fold(0usize, |bytes, range| bytes.saturating_add(range.1)),
            ranges.len(),
        );
    }
    if super::present::trace_frame() {
        // every 600th look: what the CPU's view of a few textures holds now (does the title's
        // data reach the address at all, and is the texture still kept in step?)
        static LOOKS: AtomicU64 = AtomicU64::new(0);
        if LOOKS.fetch_add(1, Ordering::Relaxed) % 600 == 599 {
            let t = TEXTURES.lock().unwrap();
            // 16 samples of 4 KiB over the whole texture (GFX10 keeps the small levels first,
            // level 0 last)
            let content = |e: &Entry| -> usize {
                let size = e.tex.size() as usize;
                // (a texture read through its pages: where they are kept)
                let pages = if e.paged { page_map(e.va, size, false) } else { None };
                let mut sample = [0u8; 4096];
                (0..16usize)
                    .map(|k| e.src + (size.saturating_sub(4096)) * k / 15)
                    .map(|a| match &pages {
                        Some(m) => {
                            m.copy(a, &mut sample);
                            sample.iter().filter(|&&b| b != 0).count()
                        }
                        None => with_title_bytes(a, 4096, |s| s.iter().filter(|&&b| b != 0).count()),
                    })
                    .sum()
            };
            let (mut with, mut without) = (0, 0);
            let mut shown = 0;
            for e in t.by_res.values().filter(|e| e.cpu_backed) {
                let c = content(e);
                if c > 0 { with += 1 } else { without += 1 }
                if c > 0 && shown < 4 {
                    shown += 1;
                    crate::xlog_info!("frame trace: title texture at {:#x} ({} bytes) has data in memory ({c} of 65536 sampled bytes); GPU-written {}", e.va, e.tex.size(), e.gpu_written);
                }
            }
            crate::xlog_info!("frame trace: title textures: {with} with data in memory, {without} all zero; {dirty_len} dirty pages this look");
        }
    }
    // the title's own memory: what the CPU wrote goes to its video memory first
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        let outcome = DIRTY_BUF.with(|dcell| {
            let dirty = dcell.borrow();
            super::bvh_want::written(&dirty, page as usize);
            let mut outcome = super::sparse::PreparationOutcome::RetryRequired;
            for _ in 0..3 {
                outcome = super::sparse::before_execute(q, &dirty, page as usize, sample.as_deref_mut());
                if outcome != super::sparse::PreparationOutcome::RetryRequired {
                    break;
                }
                super::real::stop_if_device_removed();
                std::thread::yield_now();
            }
            outcome
        });
        if outcome != super::sparse::PreparationOutcome::Ready {
            return outcome;
        }
        super::vram::before_execute(q);
    }
    // a write where a page the queue's page mappings show is kept is a write to the GPU-view pages
    // that show it: the textures read through their pages find it by their own range (sparse had
    // the real addresses above)
    let dirty_len = if crate::hle::kernelx::mirrors::any_mapped() {
        DIRTY_BUF.with(|dcell| {
            let mut dirty = dcell.borrow_mut();
            TWINS_BUF.with(|tcell| {
                let mut twins = tcell.borrow_mut();
                twins.clear();
                crate::hle::kernelx::mirrors::views_of_written(&dirty, &mut twins);
                dirty.extend_from_slice(&twins);
            });
            dirty.sort_unstable();
            dirty.dedup();
            dirty.len()
        })
    } else {
        dirty_len
    };
    let recheck_frame = {
        let (p, b) = super::present::counters();
        p.max(b)
    };
    recheck_streamed_unwatched(&ranges, recheck_frame);
    if dirty_len == 0 && PENDING.load(Ordering::Relaxed) == 0 && AGAIN.lock().unwrap().is_empty() {
        return super::sparse::PreparationOutcome::Ready;
    }
    let target = upload_target(ID3D12CommandQueue::from_raw_borrowed(&queue));
    // textures made since the last submit and those with a written page; a reference each
    // while they are uploaded
    let measure_selection = sample.is_some();
    let (todo, selection_measurement): (Vec<Due>, _) = DIRTY_BUF.with(|dcell| {
        let dirty = dcell.borrow();
        let wait_started = measure_selection.then(std::time::Instant::now);
        let mut t = TEXTURES.lock().unwrap();
        let locked_at = wait_started.map(|_| std::time::Instant::now());
        let (mut visited, mut dirty_tests, mut matches, mut retained) = (0usize, 0usize, 0usize, 0usize);
        let mut names = t.touched_by(&dirty);
        names.extend(std::mem::take(&mut t.pending_res));
        let waiting = std::mem::take(&mut *AGAIN.lock().unwrap());
        names.extend(waiting.iter().copied());
        let mut later = std::collections::BTreeSet::new();
        let mut todo = Vec::new();
        for res in names {
            let Some(e) = t.by_res.get_mut(&res) else { continue };
            if measure_selection {
                visited += 1;
            }
            let mut again = texture_upload::Change::None;
            if waiting.contains(&res) && e.cpu_backed && !e.gpu_written {
                let (queue, waited) = target.on(e.device);
                match texture_upload::take_again(&e.known, queue, waited) {
                    texture_upload::Again::Now(c) => again = c,
                    texture_upload::Again::NotYet => drop(later.insert(res)),
                    texture_upload::Again::Nothing => {}
                }
            }
            if measure_selection && e.cpu_backed && !e.pending && !e.gpu_written {
                dirty_tests += 1;
            }
            // (a GPU-owned texture is never uploaded: `texture_needs_upload`)
            let wanted = texture_needs_upload(
                e.cpu_backed,
                e.pending || again != texture_upload::Change::None,
                e.gpu_written,
                || {
                    let i = dirty.partition_point(|&p| p + page as usize <= e.src);
                    dirty.get(i).is_some_and(|&p| p < e.src + e.tex.size() as usize)
                },
            );
            if !wanted {
                continue;
            }
            if measure_selection {
                matches += 1;
            }
            let whole = e.pending;
            if e.pending {
                e.pending = false;
                PENDING.fetch_sub(1, Ordering::Relaxed);
            }
            let p = res as *mut c_void;
            // (its pages resolved once, for all reads of this upload)
            let pages = if e.paged { page_map(e.va, e.tex.size() as usize, true) } else { None };
            if let Some(r) = ID3D12Resource::from_raw_borrowed(&p) {
                todo.push(Due {
                    res: r.clone(),
                    device: e.device,
                    src: e.src,
                    tex: Arc::clone(&e.tex),
                    states: e.states.clone(),
                    known: Arc::clone(&e.known),
                    whole,
                    again,
                    target: e.target,
                    pages,
                    generation: e.generation,
                });
                if measure_selection {
                    retained += 1;
                }
            }
        }
        AGAIN.lock().unwrap().extend(later);
        drop(t);
        let finished_at = locked_at.map(|_| std::time::Instant::now());
        let measurement = wait_started.zip(locked_at).zip(finished_at).map(|((started, locked), finished)| {
            (locked.duration_since(started), finished.duration_since(locked), visited, dirty_tests, matches, retained)
        });
        (todo, measurement)
    });
    if let (Some(sample), Some((wait, hold, visited, dirty_tests, matches, retained))) = (sample.as_deref_mut(), selection_measurement) {
        super::sparse::note_texture_selection(sample, wait, hold, visited, dirty_tests, matches, retained);
    }
    seed_stream_recheck(&todo, &ranges, recheck_frame);
    let t_tex = std::time::Instant::now();
    DIRTY_BUF.with(|dcell| upload_all(todo, &dcell.borrow(), page as usize, ID3D12CommandQueue::from_raw_borrowed(&queue)));
    super::submit_cost::textures(t_tex.elapsed());
    super::sparse::PreparationOutcome::Ready
}

#[cfg(feature = "texture-selection-bench")]
#[derive(Clone, Copy)]
pub enum TextureSelectionScenario {
    Empty,
    SparsePending,
    AllPending,
    AllGpuWritten,
    OneDirty,
    SparseDirty,
    DenseDirty,
    MixedAliases,
}

#[cfg(feature = "texture-selection-bench")]
struct TextureSelectionBenchEntry {
    src: usize,
    size: usize,
    cpu_backed: bool,
    pending: bool,
    gpu_written: bool,
}

/// CPU-only model of the exact selection predicate and table lock. It deliberately excludes COM
/// retention and uploads: the benchmark separates scan/convoy cost from those Windows operations.
#[cfg(feature = "texture-selection-bench")]
pub struct TextureSelectionBench {
    entries: Mutex<BTreeMap<usize, TextureSelectionBenchEntry>>,
    dirty: Vec<usize>,
    pending: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "texture-selection-bench")]
impl TextureSelectionBench {
    const PAGE: usize = 0x1000;

    pub fn new(count: usize, scenario: TextureSelectionScenario) -> Self {
        let base = 0x1000_0000usize;
        let mut entries = BTreeMap::new();
        let mut pending = 0;
        for i in 0..count {
            let aliased = matches!(scenario, TextureSelectionScenario::MixedAliases) && i % 16 == 1;
            let alias_group = i - aliased as usize;
            let src = base + alias_group * 2 * Self::PAGE;
            let cpu_backed = !matches!(scenario, TextureSelectionScenario::MixedAliases) || i % 8 != 4;
            let is_pending = matches!(scenario, TextureSelectionScenario::AllPending)
                || matches!(scenario, TextureSelectionScenario::SparsePending) && i % 64 == 0
                || matches!(scenario, TextureSelectionScenario::MixedAliases) && i % 127 == 0;
            let gpu_written = matches!(scenario, TextureSelectionScenario::AllGpuWritten)
                || matches!(scenario, TextureSelectionScenario::MixedAliases) && i % 11 == 0;
            pending += is_pending as usize;
            entries.insert(i + 1, TextureSelectionBenchEntry {
                src,
                size: Self::PAGE + (alias_group % 3) * Self::PAGE,
                cpu_backed,
                pending: is_pending,
                gpu_written,
            });
        }
        let pages = match scenario {
            TextureSelectionScenario::Empty | TextureSelectionScenario::SparsePending | TextureSelectionScenario::AllPending => 0,
            TextureSelectionScenario::OneDirty => 1,
            TextureSelectionScenario::AllGpuWritten | TextureSelectionScenario::SparseDirty | TextureSelectionScenario::MixedAliases => 64,
            TextureSelectionScenario::DenseDirty => 4096,
        };
        let stride = if matches!(scenario, TextureSelectionScenario::DenseDirty) { 1 } else { 31 };
        let mut dirty: Vec<_> = (0..pages).map(|i| base + i * stride * Self::PAGE).collect();
        dirty.sort_unstable();
        Self { entries: Mutex::new(entries), dirty, pending: std::sync::atomic::AtomicUsize::new(pending) }
    }

    #[inline(never)]
    pub fn scan(&self) -> (usize, usize) {
        if self.dirty.is_empty() && self.pending.load(Ordering::Relaxed) == 0 {
            return (0, 0);
        }
        let entries = self.entries.lock().unwrap();
        let mut matches = 0usize;
        let mut checksum = 0usize;
        for (&resource, entry) in entries.iter() {
            if texture_needs_upload(
                entry.cpu_backed,
                entry.pending,
                entry.gpu_written,
                || {
                    let i = self.dirty.partition_point(|&p| p + Self::PAGE <= entry.src);
                    self.dirty.get(i).is_some_and(|&p| p < entry.src + entry.size)
                },
            ) {
                matches += 1;
                checksum ^= resource;
            }
        }
        (matches, checksum)
    }
}

static UNCHANGED: AtomicU64 = AtomicU64::new(0);

/// `XCL_TEXTURE_PARTS=0`: a texture that changed goes up as a whole.
fn upload_parts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TEXTURE_PARTS").map_or(true, |v| v.trim() != "0"))
}

/// `XCL_TEXTURE_SKIP_UNCHANGED=0` uploads what was written without comparing it.
fn unchanged_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TEXTURE_SKIP_UNCHANGED").map_or(true, |v| v.trim() != "0"))
}

/// Frame milestones (relative to last upload or detected content change) at which unwatched
/// textures (e.g. physical-page / `MEM_MAPPED` views that `GetWriteWatch` cannot track) are
/// re-fingerprinted to catch late streaming writes that land after an initial zeroed upload.
const STREAM_RECHECK_MILESTONES: [u64; 5] = [4, 12, 40, 120, 400];
/// Per-slice sample size (start, mid, end -> at most 48 KiB hashed per due texture).
const STREAM_SAMPLE_SLICE: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StreamRecheck {
    base_frame: u64,
    stage: u8,
    /// `None`: no whole upload since the entry was made (a partial upload made it), so the PC
    /// texture has no known content to compare with and the first milestone queues a whole upload.
    last_fp: Option<u64>,
}

static STREAM_RECHECK: Mutex<BTreeMap<usize, StreamRecheck>> = Mutex::new(BTreeMap::new());
static LAST_RECHECK_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
static STREAM_RECHECK_HITS: AtomicU64 = AtomicU64::new(0);

/// `XCL_TEXTURE_STREAM_RECHECK=0` disables spaced-milestone fingerprinting for unwatched textures.
fn stream_recheck_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TEXTURE_STREAM_RECHECK").map_or(true, |v| v.trim() != "0"))
}

/// Whether every 4 KiB page of `[src, src + len)` is covered by the sorted, disjoint
/// write-watched spans `watched` (`(base, size)` pairs from `cached_watch_ranges`).
fn range_is_fully_watched(src: usize, len: usize, watched: &[(usize, usize)]) -> bool {
    if len == 0 {
        return true;
    }
    let mut lo = src & !0xFFF;
    let hi = src.saturating_add(len).saturating_add(0xFFF) & !0xFFF;
    if lo >= hi {
        return true;
    }
    let mut i = watched.partition_point(|&(b, s)| b.saturating_add(s) <= lo);
    while lo < hi {
        let Some(&(b, s)) = watched.get(i) else {
            return false;
        };
        if b > lo || s == 0 {
            return false;
        }
        lo = b.saturating_add(s);
        i += 1;
    }
    true
}

/// Bounded (<= 48 KiB) start/mid/end fingerprint of a texture's backing memory, read via
/// `read_committed` so uncommitted or concurrently decommitted pages read as zero without faulting.
unsafe fn sample_fingerprint(src: usize, size: usize) -> u64 {
    if size == 0 {
        return 0;
    }
    use texture_upload::content_hash;
    if size <= 3 * STREAM_SAMPLE_SLICE {
        let buf = read_committed(src, size);
        return content_hash(&buf);
    }
    let mid = src.saturating_add((size - STREAM_SAMPLE_SLICE) / 2);
    let end = src.saturating_add(size - STREAM_SAMPLE_SLICE);
    let h0 = content_hash(&read_committed(src, STREAM_SAMPLE_SLICE));
    let h1 = content_hash(&read_committed(mid, STREAM_SAMPLE_SLICE));
    let h2 = content_hash(&read_committed(end, STREAM_SAMPLE_SLICE));
    let mut mix = [0u8; 32];
    mix[0..8].copy_from_slice(&h0.to_le_bytes());
    mix[8..16].copy_from_slice(&h1.to_le_bytes());
    mix[16..24].copy_from_slice(&h2.to_le_bytes());
    mix[24..32].copy_from_slice(&(size as u64).to_le_bytes());
    content_hash(&mix)
}

/// Record baseline fingerprints for unwatched textures that go up whole in `todo` so late streaming
/// writes into mapped physical pages are caught at subsequent frame milestones.
unsafe fn seed_stream_recheck(todo: &[Due], watched: &[(usize, usize)], frame: u64) {
    if !stream_recheck_enabled() || todo.is_empty() {
        return;
    }
    let mut seeded = Vec::new();
    let mut unsent = Vec::new();
    for d in todo {
        let size = d.tex.size() as usize;
        // (a texture read through its pages counts as watched: they are found written where they
        // are kept, and a fingerprint of the view's own address would read zeros)
        if size == 0 || d.pages.is_some() || range_is_fully_watched(d.src, size, watched) {
            continue;
        }
        // A partial upload sends only the written pages it knows of, or nothing when they are
        // unchanged, so the PC texture does not hold what its memory holds now: the baseline of
        // the last whole upload stays, and so does its schedule (moving the schedule on every
        // partial upload would keep a texture written every few frames from reaching a milestone).
        // A texture with no entry (retired, pruned, never seeded) gets one without a baseline, so
        // its first milestone queues the whole upload that sets one.
        if !d.whole {
            unsent.push(d.res.as_raw() as usize);
            continue;
        }
        let fp = sample_fingerprint(d.src, size);
        seeded.push((d.res.as_raw() as usize, StreamRecheck { base_frame: frame, stage: 0, last_fp: Some(fp) }));
    }
    if !seeded.is_empty() || !unsent.is_empty() {
        let mut g = STREAM_RECHECK.lock().unwrap();
        for (res, state) in seeded {
            g.insert(res, state);
        }
        for res in unsent {
            g.entry(res).or_insert(StreamRecheck { base_frame: frame, stage: 0, last_fp: None });
        }
    }
}

/// At most once per presented frame, re-fingerprint due unwatched textures at spaced milestones.
unsafe fn recheck_streamed_unwatched(watched: &[(usize, usize)], frame: u64) {
    if !stream_recheck_enabled() {
        return;
    }
    if LAST_RECHECK_FRAME.swap(frame, Ordering::AcqRel) == frame {
        return;
    }
    recheck_streamed_unwatched_at(watched, frame);
}

/// Re-fingerprint due unwatched textures at `frame`, re-queuing changed entries and resetting
/// their milestone schedule so subsequent higher-resolution mip streams are also caught.
unsafe fn recheck_streamed_unwatched_at(watched: &[(usize, usize)], frame: u64) {
    let due: Vec<(usize, Option<u64>)> = {
        let g = STREAM_RECHECK.lock().unwrap();
        if g.is_empty() {
            return;
        }
        g.iter()
            .filter_map(|(&res, st)| {
                let target = *STREAM_RECHECK_MILESTONES.get(st.stage as usize)?;
                (frame.saturating_sub(st.base_frame) >= target).then_some((res, st.last_fp))
            })
            .collect()
    };
    if due.is_empty() {
        return;
    }

    let mut check = Vec::with_capacity(due.len());
    let mut prune = Vec::new();
    {
        let t = TEXTURES.lock().unwrap();
        for (res, prev_fp) in due {
            match t.by_res.get(&res) {
                Some(e) if e.gpu_written => prune.push(res),
                Some(e) if e.pending => {}
                Some(e) => {
                    let size = e.tex.size() as usize;
                    if e.paged || range_is_fully_watched(e.src, size, watched) {
                        prune.push(res);
                    } else {
                        check.push((res, e.src, size, prev_fp));
                    }
                }
                None => {
                    if !t.shadowed.contains_key(&res) {
                        prune.push(res);
                    }
                }
            }
        }
    }
    if !prune.is_empty() {
        let mut g = STREAM_RECHECK.lock().unwrap();
        for res in prune {
            g.remove(&res);
        }
    }
    if check.is_empty() {
        return;
    }

    let results: Vec<(usize, Option<u64>, u64)> = check
        .into_iter()
        .map(|(res, src, size, prev_fp)| (res, prev_fp, sample_fingerprint(src, size)))
        .collect();

    let mut changed = Vec::new();
    {
        let mut g = STREAM_RECHECK.lock().unwrap();
        for (res, prev_fp, cur_fp) in results {
            let Some(st) = g.get_mut(&res) else {
                continue;
            };
            // (an entry without a baseline is always queued)
            if prev_fp != Some(cur_fp) {
                *st = StreamRecheck { base_frame: frame, stage: 0, last_fp: Some(cur_fp) };
                changed.push(res);
            } else {
                st.stage = st.stage.saturating_add(1);
                if st.stage as usize >= STREAM_RECHECK_MILESTONES.len() {
                    g.remove(&res);
                }
            }
        }
    }
    if !changed.is_empty() {
        let mut queued = 0u64;
        {
            let mut t = TEXTURES.lock().unwrap();
            for res in changed {
                if let Some(e) = t.by_res.get_mut(&res) {
                    if !e.gpu_written && !e.pending {
                        e.pending = true;
                        PENDING.fetch_add(1, Ordering::Relaxed);
                        // (a submit takes the pending ones from this set, not from every texture)
                        t.pending_res.insert(res);
                        queued += 1;
                    }
                }
            }
        }
        if queued > 0 {
            let total = STREAM_RECHECK_HITS.fetch_add(queued, Ordering::Relaxed) + queued;
            if total == queued || (total.is_power_of_two() && total >= 16) {
                crate::xlog_info!("D3D12.X: title textures: streamed unwatched re-uploads triggered: {total}");
            }
        }
    }
}

/// bytes and textures of one batch at most: a batch's upload buffer is kept at its size, and each GPU
/// copy list stays short enough for the 2 s a driver gives one, when thousands of small textures
/// stream at once
const BATCH_BYTES: usize = 32 << 20;
const BATCH_ITEMS: usize = 256;

/// Whether another upload `(next_dev, next_bytes)` still fits in the current batch of
/// `picked_count` textures totalling `picked_bytes` on `batch_dev`. An empty batch always takes
/// its first upload (even if larger than `BATCH_BYTES`) so progress is guaranteed.
fn batch_fits(picked_count: usize, picked_bytes: usize, batch_dev: usize, next_dev: usize, next_bytes: usize) -> bool {
    picked_count == 0
        || (picked_count < BATCH_ITEMS
            && next_dev == batch_dev
            && picked_bytes.saturating_add(next_bytes) <= BATCH_BYTES)
}

struct QueueUploads {
    fence: ID3D12Fence,
    value: u64,
    in_flight: std::collections::VecDeque<(u64, texture_upload::Job)>,
    bytes: u64,
}

unsafe impl Send for QueueUploads {}

type QueueUploadState = Arc<Mutex<Option<QueueUploads>>>;

/// Queue-local serialization for upload retirement. The registry is held only long enough to
/// acquire a strong state lease; driver calls and waits on one title queue do not block another.
static ON_QUEUE: Mutex<BTreeMap<usize, QueueUploadState>> = Mutex::new(BTreeMap::new());

fn queue_upload_state(key: usize) -> QueueUploadState {
    Arc::clone(
        ON_QUEUE
            .lock()
            .unwrap()
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(None))),
    )
}

/// Drop completed uploads cached for a retired title queue after releasing the cache lock, and
/// what it waited for of XCL's own uploads (a queue made later at the same address has waited for
/// none of them).
pub(super) fn forget_queue(id: super::present::QueueId) {
    let raw = id.raw();
    let gone = ON_QUEUE.lock().unwrap().remove(&raw);
    drop(gone);
    OWN_WAITED.lock().unwrap().remove(&raw);
    let known: Vec<texture_upload::Known> = {
        let t = TEXTURES.lock().unwrap();
        t.by_res.values().chain(t.shadowed.values()).map(|e| Arc::clone(&e.known)).collect()
    };
    for k in &known {
        texture_upload::forget_queue(k, raw);
    }
}

/// upload buffers the GPU has not read yet, before a submit waits for it: a moment past the first,
/// in earnest past the second (a title's queue can stand at a wait that another submit ends)
const IN_FLIGHT_BYTES: u64 = 512 << 20;
const IN_FLIGHT_MAX: u64 = 2 << 30;

unsafe fn title_queue_for(dev: &ID3D12Device, title: Option<&ID3D12CommandQueue>) -> Option<ID3D12CommandQueue> {
    let q = title?;
    if !super::present::queue_accepts_internal_work(q.as_raw() as usize) {
        return None;
    }
    if q.GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT {
        return None;
    }
    let mut qd: Option<ID3D12Device> = None;
    q.GetDevice(&mut qd).ok()?;
    (qd?.as_raw() == dev.as_raw()).then(|| q.clone())
}

/// The uploads of `q` the GPU is done with, oldest first, and, while more than `cap` bytes are in
/// flight, the older ones as it gets done with them, for up to `wait_ms`. The caller gives them
/// back (`texture_upload::give_back`) once it no longer holds the queue's state.
unsafe fn reclaim(q: &mut QueueUploads, cap: u64, wait_ms: u64) -> Vec<texture_upload::Job> {
    let mut retired = Vec::new();
    let start = std::time::Instant::now();
    while let Some(&(v, _)) = q.in_flight.front() {
        if super::real::fence_completed(&q.fence) < v {
            if q.bytes <= cap || start.elapsed().as_millis() as u64 >= wait_ms {
                break;
            }
            // (no event: a wait that gives up would leave its registration behind)
            std::thread::sleep(std::time::Duration::from_micros(500));
            continue;
        }
        let Some((_, job)) = q.in_flight.pop_front() else { break };
        q.bytes -= job.bytes;
        retired.push(job);
    }
    retired
}

/// The queue a submit's uploads go to, by device, and the one it waits for first.
struct Target {
    /// the title's queue, when they go to it
    title: Option<(usize, usize)>,
}

impl Target {
    /// (the queue, the one it has waited for) of uploads for `device`; 0: none.
    fn on(&self, device: usize) -> (usize, usize) {
        let own = QUEUES.lock().unwrap().get(&device).map_or(0, |q| q.as_raw() as usize);
        match self.title {
            Some((q, d)) if d == device => (q, own),
            _ => (own, 0),
        }
    }
}

unsafe fn upload_target(title: Option<&ID3D12CommandQueue>) -> Target {
    let of = |q: &ID3D12CommandQueue| {
        if q.GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT {
            return None;
        }
        let mut dev: Option<ID3D12Device> = None;
        q.GetDevice(&mut dev).ok()?;
        Some((q.as_raw() as usize, dev?.as_raw() as usize))
    };
    Target { title: title.and_then(of) }
}

/// textures with something to send again (`texture_upload::sent`)
static AGAIN: Mutex<std::collections::BTreeSet<usize>> = Mutex::new(std::collections::BTreeSet::new());

/// Upload `items` on `queue`, a DIRECT one: the ones that could not be, the bytes of the others and
/// where they went (None: there were none).
unsafe fn submit_on(dev: &ID3D12Device, queue: &ID3D12CommandQueue, items: &[texture_upload::Item]) -> Result<(Vec<(usize, String)>, u64, Option<texture_upload::Sent>), String> {
    // The textures' tile mappings made since the queue last waited for them (a texture made on
    // another thread after `refresh_once`'s wait, or its levels mapped just now by
    // `map_levels_with_texels`): a copy into a tile still unmapped is a copy-engine page fault
    // on NVIDIA (`DmaPageFaultWrite`).
    super::vram::before_execute(queue);
    let key = queue.as_raw() as usize;
    let state = queue_upload_state(key);
    let mut slot = state.lock().unwrap();
    if slot.is_none() {
        // (before anything runs that it would have to follow)
        let fence = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).map_err(|e| e.to_string())?;
        *slot = Some(QueueUploads { fence, value: 0, in_flight: std::collections::VecDeque::new(), bytes: 0 });
    }
    let Some(q) = slot.as_mut() else { return Err("no fence for the queue".into()) };
    // (a piece that panicked on the way into the upload buffer: the batch fails before anything
    // is executed and is tried again, `RETRY`, rather than going up with what the buffer held)
    let submitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| texture_upload::submit(dev, queue, items)));
    let (failed, f, bytes) = submitted.map_err(|_| "texture upload: filling the upload buffer panicked".to_string())??;
    let Some(f) = f else { return Ok((failed, 0, None)) };
    // (through the PC queue slot: a title's queue may have the console's layout)
    if let Err(e) = super::real::native_queue_signal(queue, &q.fence, q.value + 1) {
        drop(slot);
        // the GPU may still run the list and read the buffer: they stay when it does not get there
        match texture::wait(dev, queue) {
            Ok(()) => drop(f),
            Err(_) => std::mem::forget(f),
        }
        return Err(e.to_string());
    }
    q.value += 1;
    q.bytes += f.bytes;
    let sent = texture_upload::Sent { queue: key, fence: q.fence.clone(), value: q.value };
    q.in_flight.push_back((sent.value, f));
    let mut retired = reclaim(q, IN_FLIGHT_BYTES, 20);
    retired.extend(reclaim(q, IN_FLIGHT_MAX, 10_000));
    drop(slot);
    retired.into_iter().for_each(texture_upload::give_back);
    Ok((failed, bytes, Some(sent)))
}

/// The uploads on XCL's own queue (its fence, the last value): each of the title's queues waits for
/// them before its next lists (on the GPU; that queue waits for nothing of the title's).
static OWN_UPLOADS: Mutex<Option<(ID3D12Fence, u64)>> = Mutex::new(None);
/// title queue → the value it waited for last
static OWN_WAITED: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());

unsafe fn wait_for_own_uploads(queue: &ID3D12CommandQueue) {
    let Some((fence, v)) = OWN_UPLOADS.lock().unwrap().clone() else { return };
    let key = queue.as_raw() as usize;
    let mut g = OWN_WAITED.lock().unwrap();
    if g.get(&key).is_some_and(|&w| w >= v) {
        return;
    }
    // (through the PC queue slot: a title's queue may have the console's layout)
    if super::real::fence_completed(&fence) < v && super::real::native_queue_wait(queue, &fence, v).is_ok() {
        super::list_x::note_internal_wait(key, "texture uploads", &fence, v);
    }
    g.insert(key, v);
}

/// The pages `[src, src + size)` lies on that were written since the look that took them last; they
/// stay reported. None for memory that is not watched.
unsafe fn written_pages(src: usize, size: usize) -> Vec<usize> {
    const PAGE: usize = 0x1000;
    let (mut at, end) = (src / PAGE * PAGE, (src + size).next_multiple_of(PAGE));
    let mut out = Vec::new();
    let mut buf = [std::ptr::null_mut::<c_void>(); 512];
    // (the walk stays out of the way, as in a look)
    let was = LOOKING.swap(true, Ordering::AcqRel);
    while at < end {
        let (mut n, mut page) = (buf.len(), PAGE as u32);
        if GetWriteWatch(0, at as *const c_void, end - at, buf.as_mut_ptr(), &mut n, &mut page) != 0 {
            break;
        }
        out.extend(buf[..n].iter().map(|&p| p as usize));
        if n < buf.len() {
            break;
        }
        at = buf[n - 1] as usize + PAGE;
    }
    LOOKING.store(was, Ordering::Release);
    out
}

/// Whether the looks take every block `[src, src + size)` lies in.
fn all_hot(src: usize, size: usize) -> bool {
    if !hot_looks() {
        return true;
    }
    let g = HOT.lock().unwrap();
    let Some(h) = g.as_ref() else { return false };
    (src & !(HOT_BLOCK - 1)..src + size).step_by(HOT_BLOCK).all(|b| h.hot.contains_key(&b))
}

/// The textures of failed batches whose time came (`retry_failed`), as (texture, registration
/// generation): one still at that generation, CPU-backed, not written by the GPU and not pending is
/// made pending again (a whole upload); a replacement, removal, GPU writer or non-CPU-backed entry
/// owns its own state and is left alone. Returns the textures made pending.
fn requeue_generations(table: &mut Table, keys: &[(usize, u64)]) -> Vec<usize> {
    let mut added = Vec::new();
    for &(resource, generation) in keys {
        let Some(entry) = table.by_res.get_mut(&resource) else { continue };
        if entry.generation == generation && entry.cpu_backed && !entry.gpu_written && !entry.pending {
            entry.pending = true;
            added.push(resource);
        }
    }
    for &resource in &added {
        table.pending_res.insert(resource);
    }
    added
}

/// What changed of the textures `todo` (written pages: `dirty`, sorted) goes up, in batches of up to
/// `BATCH_BYTES` and `BATCH_ITEMS` (one list each).
unsafe fn upload_all(todo: Vec<Due>, dirty: &[usize], page: usize, title: Option<&ID3D12CommandQueue>) {
    static N: AtomicU64 = AtomicU64::new(0);
    let warn = |e: &str| {
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("title texture: upload failed: {e}");
        }
    };
    // The written pages that hold what they held at their texture's last upload are nothing to do:
    // the write watch is per page and a title writes next to its textures (and rewrites what it
    // wrote).
    let t = std::time::Instant::now();
    let mut changes: std::collections::VecDeque<texture_upload::Change> = {
        let pages: Vec<std::borrow::Cow<[usize]>> = todo
            .iter()
            .filter(|d| !d.whole)
            .map(|d| {
                let size = d.tex.size() as usize;
                let (from, to) = (dirty.partition_point(|&p| p + page <= d.src), dirty.partition_point(|&p| p < d.src + size));
                let mine = &dirty[from..to.max(from)];
                // (a look takes the blocks written lately: the others' pages of this texture too, or
                // it would go up in part as it was and in part as it is; not for one read through
                // its pages, whose own address is not watched)
                let also = if mine.is_empty() || d.pages.is_some() || all_hot(d.src, size) { Vec::new() } else { written_pages(d.src, size) };
                if also.iter().all(|p| mine.binary_search(p).is_ok()) {
                    return mine.into();
                }
                let mut all = [mine, &also].concat();
                all.sort_unstable();
                all.dedup();
                all.into()
            })
            .collect();
        let written: Vec<texture_upload::Written> = todo
            .iter()
            .filter(|d| !d.whole)
            .zip(&pages)
            .map(|(d, pages)| texture_upload::Written { src: d.src, size: d.tex.size() as usize, dirty: pages, known: &d.known, pages: d.pages.as_deref() })
            .collect();
        let (changes, compared) = texture_upload::changed(&written, unchanged_skip());
        let same = changes.iter().filter(|c| **c == texture_upload::Change::None).count() as u64;
        let before = UNCHANGED.fetch_add(same, Ordering::Relaxed);
        if same > 0 && (before == 0 || (before >= 512 && before.ilog2() != (before + same).ilog2())) {
            crate::xlog_info!("title textures: {} uploads left out so far – the texture's memory held what it held at its last upload", before + same);
        }
        super::submit_cost::textures_compared(written.len(), same as usize, compared, t.elapsed());
        changes.into()
    };
    let mut todo = todo.into_iter().filter_map(|d| {
        let change = if d.whole { texture_upload::Change::All } else { changes.pop_front()? };
        let changed = match texture_upload::join(change, d.again.clone()) {
            texture_upload::Change::None => return None,
            texture_upload::Change::All => None,
            texture_upload::Change::Parts(_) if d.target || !upload_parts() => None,
            texture_upload::Change::Parts(p) => Some(p),
        };
        let size = d.tex.size() as usize;
        // (whole blocks go up, and whole volumes)
        const BLOCK: usize = 0x1_0000;
        let volume = d.tex.texels().is_some_and(|s| s.depth > 0);
        let parts = changed.as_ref().filter(|_| !volume);
        let bytes = parts.map_or(size, |p| p.iter().map(|r| (r.1.div_ceil(BLOCK) - r.0 / BLOCK) * BLOCK).sum::<usize>().min(size));
        Some((d, changed, bytes))
    }).peekable();
    while todo.peek().is_some() {
        let (mut items, mut known): (Vec<texture_upload::Item>, Vec<texture_upload::Known>) = (Vec::new(), Vec::new());
    let mut generations: Vec<u64> = Vec::new();
        let (mut bytes, mut device) = (0usize, 0usize);
        while let Some((d, mut changed, n)) = todo.next_if(|(d, _, n)| batch_fits(items.len(), bytes, device, d.device, *n)) {
            device = d.device;
            bytes += n;
            let mut unmapped = false;
            if super::vram::level_pages() && super::vram::has_lazy_levels(d.res.as_raw() as usize) {
                // (the queue waits for these mappings before the copies: `vram::before_execute`
                // below; what a level's fresh pages held is another texture's, so all of it goes up)
                if !map_levels_with_texels(&d.res, &subresources_in(d.src, &d.tex, d.pages.as_deref())).is_empty() {
                    changed = None;
                }
                unmapped = super::vram::has_unmapped_levels(d.res.as_raw() as usize);
            }
            if unmapped {
                // A level left without pages (zeros when looked at above) may get its texels
                // before the copy reads them: the copy to its tiles is dropped, and what it read
                // must not be kept as what the PC texture holds – the next write then looks the
                // same and the level never goes up. The next write sends all of it, and the level
                // gets its pages then.
                texture_upload::unknown(&d.known);
            } else if changed.is_none() {
                // (before its bytes are read: what is written from here on is compared)
                texture_upload::all_sent(&d.known, d.src, d.tex.size() as usize);
            }
            if super::present::trace_frame() {
                note_content(d.src, &d.res, &subresources_in(d.src, &d.tex, d.pages.as_deref()));
            }
            generations.push(d.generation);
            items.push(texture_upload::Item { res: d.res, src: d.src, tex: d.tex, states: d.states, changed, known: Some(Arc::clone(&d.known)), pages: d.pages });
            known.push(d.known);
        }
        let p = device as *mut c_void;
        let Some(dev) = ID3D12Device::from_raw_borrowed(&p) else { continue };
        let t_submit = std::time::Instant::now();
        // (`submit_on` waits for the tile mappings made before it: the level pages mapped above too)
        let done = own_work(|| match title_queue_for(dev, title) {
            Some(q) => {
                // what went up on XCL's own queue first: these may be newer bytes of the same texture
                wait_for_own_uploads(&q);
                submit_on(dev, &q, &items)
            }
            // not from a compute or copy queue: on XCL's own, which the title's queues wait for
            None => {
                let queue = upload_queue(dev, device)?;
                let r = submit_on(dev, &queue, &items)?;
                if let Some(s) = &r.2 {
                    *OWN_UPLOADS.lock().unwrap() = Some((s.fence.clone(), s.value));
                }
                Ok(r)
            }
        });
        // (what did not go up is not what the PC texture holds: as a whole at its next write)
        match done {
            Ok((failed, bytes, to)) => {
                for (k, e) in &failed {
                    warn(e);
                    texture_upload::forget(&known[*k]);
                }
                if let Some(to) = to {
                    let waited = QUEUES.lock().unwrap().get(&device).map_or(0, |q| q.as_raw() as usize);
                    for (k, it) in items.iter().enumerate().filter(|(k, _)| !failed.iter().any(|f| f.0 == *k)) {
                        // (XCL's own queue waits for none of the title's)
                        if texture_upload::sent(&known[k], it.changed.as_deref(), &to, if to.queue == waited { 0 } else { waited }) {
                            AGAIN.lock().unwrap().insert(it.res.as_raw() as usize);
                        }
                    }
                }
                UPLOADS.fetch_add((items.len() - failed.len()) as u64, Ordering::Relaxed);
                super::submit_cost::textures_uploaded(items.len() - failed.len(), bytes as usize, t_submit.elapsed());
            }
            Err(e) => {
                warn(&e);
                known.iter().for_each(texture_upload::forget);
                // (no memory for a buffer, most likely: again in a moment)
                let at = std::time::Instant::now() + RETRY_AFTER;
                RETRY.lock().unwrap().extend(items.iter().zip(&generations).map(|(it, &generation)| (at, it.res.as_raw() as usize, generation)));
            }
        }
    }
}

/// textures of batches that could not be uploaded, with their registration generation, and when
/// they are tried again
static RETRY: Mutex<Vec<(std::time::Instant, usize, u64)>> = Mutex::new(Vec::new());
const RETRY_AFTER: std::time::Duration = std::time::Duration::from_millis(100);

/// Those whose time has come are to be uploaded as a whole, like new ones: each still at the
/// generation it failed at (`requeue_generations`).
fn retry_failed() {
    let mut g = RETRY.lock().unwrap();
    if g.is_empty() {
        return;
    }
    let now = std::time::Instant::now();
    let mut due = Vec::new();
    g.retain(|&(at, res, generation)| {
        if at > now {
            return true;
        }
        due.push((res, generation));
        false
    });
    let added = requeue_generations(&mut TEXTURES.lock().unwrap(), &due);
    PENDING.fetch_add(added.len(), Ordering::Relaxed);
}

/// An upload of `changed` (None: all) of `res`, a texture of the console description `desc` with its
/// bytes in `mem`, on `queue`, waited for: for the test of its barriers under the debug layer.
#[doc(hidden)]
pub unsafe fn upload_and_wait(dev: &ID3D12Device, queue: &ID3D12CommandQueue, res: &ID3D12Resource, desc: &[u8; 0x38], mem: &[u8], states: Vec<D3D12_RESOURCE_STATES>, changed: Option<Vec<(usize, usize)>>) -> Result<(), String> {
    let tex = Arc::new(xg::Texture::new(desc).map_err(|hr| format!("layout {hr:#x}"))?);
    if mem.len() < tex.size() as usize {
        return Err(format!("{} bytes of {}", mem.len(), tex.size()));
    }
    let item = texture_upload::Item { res: res.clone(), src: mem.as_ptr() as usize, tex, states, changed, known: None, pages: None };
    let (failed, job, _) = own_work(|| texture_upload::submit(dev, queue, &[item]))?;
    texture::wait(dev, queue)?;
    job.into_iter().for_each(texture_upload::give_back);
    failed.into_iter().next().map_or(Ok(()), |(_, e)| Err(e))
}

/// (`vram::level_pages`) Before an upload of title texture `r`: its levels with texels in `subs`
/// (its subresources as they go up) get their video memory pages if they have none, and those
/// with pages and no texels now give them back (`vram::unmap_levels`) - a texture placed with
/// lazy levels; any other: nothing to do. A level of zeros has no texels: a streamed
/// texture's larger levels before the title has memory behind them (read as zeros,
/// `read_committed`). It keeps no pages, reads as zeros as it would with them, and its copy goes
/// to tiles that are not mapped, which drops it (tiled resources tier 2). Returns the levels that
/// got pages now (`vram::map_levels`).
unsafe fn map_levels_with_texels(r: &ID3D12Resource, subs: &[Subresource]) -> Vec<u32> {
    if !super::vram::has_lazy_levels(r.as_raw() as usize) {
        return Vec::new();
    }
    // (one slice: its subresources are its levels)
    let present: Vec<u32> = subs.iter().enumerate().filter(|(_, s)| !all_zero(&s.data)).map(|(i, _)| i as u32).collect();
    // a level that had texels and has none now: the title took its memory, and it gives back its
    // pages (this upload would only have written zeros to them)
    let zero: Vec<u32> = (0..subs.len() as u32).filter(|l| !present.contains(l)).collect();
    super::vram::unmap_levels(r, &zero);
    super::vram::map_levels(r, Some(&present))
}

/// Whether every byte of `b` is zero (eight at a time: gigabytes a second where it is).
fn all_zero(b: &[u8]) -> bool {
    let mut words = b.chunks_exact(8);
    words.all(|w| u64::from_ne_bytes(w.try_into().unwrap()) == 0) && words.remainder().iter().all(|&x| x == 0)
}

/// `XCL_TRACE_FRAME`: how much of a title texture's top level is not zero, each time it is
/// uploaded (the first 48 uploads of textures of 64x64 and more).
fn note_content(va: usize, r: &ID3D12Resource, subs: &[texture::Subresource]) {
    static N: AtomicU64 = AtomicU64::new(0);
    let Some(top) = subs.first() else { return };
    let d = unsafe { r.GetDesc() };
    if d.Width < 64 || d.Height < 64 || N.fetch_add(1, Ordering::Relaxed) >= 48 {
        return;
    }
    let nonzero = top.data.iter().filter(|&&b| b != 0).count();
    crate::xlog_info!(
        "frame trace: title texture at {va:#x} ({}x{} format {}) uploaded: {nonzero} of {} bytes of level 0 not zero",
        d.Width,
        d.Height,
        d.Format.0,
        top.data.len()
    );
}

/// `XCL_TRACE_FRAME`: what the GPU holds in each level of the first slice of title texture `res`
/// (read back on XCL's queue; the texture is taken to be in a shader-resource state) against the
/// CPU's view of the memory now. Each texture once, four at most (`XCL_TRACE_PROBES=n`: n).
pub unsafe fn probe_gpu_content(res: usize) {
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let most = *MOST.get_or_init(|| std::env::var("XCL_TRACE_PROBES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4));
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&res) || g.len() >= most {
            return;
        }
        g.push(res);
    }
    let Some((device, va, tex, gpu_written, paged)) = TEXTURES.lock().unwrap().by_res.get(&res).map(|e| (e.device, e.src, Arc::clone(&e.tex), e.gpu_written, e.paged)) else { return };
    let d = device as *mut c_void;
    let r = res as *mut c_void;
    let (Some(dev), Some(r)) = (ID3D12Device::from_raw_borrowed(&d), ID3D12Resource::from_raw_borrowed(&r)) else { return };
    let rd = r.GetDesc();
    let levels = (rd.MipLevels as usize).clamp(1, 16);
    let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); levels];
    let (mut rows, mut row_bytes, mut total) = (vec![0u32; levels], vec![0u64; levels], 0u64);
    dev.GetCopyableFootprints(&rd, 0, levels as u32, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(row_bytes.as_mut_ptr()), Some(&mut total));
    let result = own_work(|| -> Result<Vec<Vec<u8>>, String> {
        let queue = upload_queue(dev, device)?;
        super::vram::before_execute(&queue);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| e.to_string())?;
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
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(|e| e.to_string())?;
        let rb = rb.ok_or("no readback buffer")?;
        let barrier = |sub: u32, a: D3D12_RESOURCE_STATES, b: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: sub, StateBefore: a, StateAfter: b }),
            },
        };
        let srv = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        for (i, fp) in fps.iter().enumerate() {
            list.ResourceBarrier(&[barrier(i as u32, srv, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[barrier(i as u32, D3D12_RESOURCE_STATE_COPY_SOURCE, srv)]);
        }
        list.Close().map_err(|e| e.to_string())?;
        let list = list.cast().map_err(|e| e.to_string())?;
        super::real::native_queue_execute(&queue, &list);
        texture::wait(dev, &queue)?;
        let mut p: *mut c_void = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(levels);
        for (i, fp) in fps.iter().enumerate() {
            let mut level = Vec::with_capacity(rows[i] as usize * row_bytes[i] as usize);
            for y in 0..rows[i] as usize {
                level.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize), row_bytes[i] as usize));
            }
            out.push(level);
        }
        rb.Unmap(0, None);
        Ok(out)
    });
    // (a texture read through its pages: where they are kept, with a table of its own)
    let pages = if paged { page_map(va, tex.size() as usize, false) } else { None };
    let cpu = subresources_in(va, &tex, pages.as_deref());
    match result {
        Ok(gpu) => {
            let nonzero = |b: &[u8]| b.iter().filter(|&&x| x != 0).count();
            let per: Vec<String> = gpu
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let c = cpu.get(i).map_or(&[][..], |s| s.data.as_slice());
                    format!("L{i} {} B: GPU {} not zero, CPU {}, {} equal", g.len(), nonzero(g), nonzero(c), g.iter().zip(c.iter()).filter(|(a, b)| a == b).count())
                })
                .collect();
            crate::xlog_info!(
                "frame trace: GPU content of title texture {res:#x} at {va:#x} ({}x{} format {}, {} mips, GPU-written {gpu_written}): {}; first bytes of level 0 {:02x?}",
                rd.Width,
                rd.Height,
                rd.Format.0,
                rd.MipLevels,
                per.join("; "),
                gpu.first().map_or(&[][..], |g| &g[..g.len().min(16)])
            );
        }
        Err(e) => crate::xlog_warn!("frame trace: GPU content of title texture {res:#x}: {e}"),
    }
}

/// XCL's upload queues made so far.
pub(super) fn upload_queues() -> Vec<ID3D12CommandQueue> {
    QUEUES.lock().unwrap().values().cloned().collect()
}

/// XCL's queue for uploads on `dev`.
unsafe fn upload_queue(dev: &ID3D12Device, device: usize) -> Result<ID3D12CommandQueue, String> {
    let mut q = QUEUES.lock().unwrap();
    if let Some(q) = q.get(&device) {
        return Ok(q.clone());
    }
    let made: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| e.to_string())?;
    q.insert(device, made.clone());
    Ok(made)
}

/// Ranges closer than this are scanned as one. A write-watch call costs ~3.5 µs however little it
/// covers, while 2 GiB of committed pages took 0.8–1.3 ms on Windows 11: an extra call breaks even
/// with scanning roughly 5–9 MiB of gap. Eight MiB still joins the 1 MiB-spaced sparse ranges that
/// otherwise made thousands of calls per submit, without scanning up to 64 MiB of unrelated pages
/// between ranges. Gap pages may be reported too; uploads still keep to the ranges they track.
const WATCH_GAP: usize = 8 << 20;

#[derive(Clone, Copy)]
struct WatchSpan {
    region: usize,
    start: usize,
    end: usize,
}

/// Clip `wanted` to exact overlap/adjacent spans in the disjoint allocations `regions`, then bridge
/// the smallest gaps toward `target_calls`. This is best effort: allocations are never bridged, so
/// the result has at least one call per touched allocation. Invalid or overflowing allocations are
/// ignored. Production keeps using `watch_ranges` until a Windows measurement establishes a target.
fn plan_watch_ranges(regions: &[(usize, usize)], mut wanted: Vec<(usize, usize)>, target_calls: usize) -> Vec<(usize, usize)> {
    wanted.sort_unstable();
    let mut normalized = Vec::<(usize, usize)>::new();
    for (start, end) in wanted {
        if start >= end {
            continue;
        }
        match normalized.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => normalized.push((start, end)),
        }
    }
    let mut exact = Vec::<WatchSpan>::new();
    for (region, &(base, size)) in regions.iter().enumerate() {
        let Some(end) = base.checked_add(size).filter(|&end| end > base) else { continue };
        let first = normalized.partition_point(|&(_, b)| b <= base);
        for &(a, b) in normalized[first..].iter().take_while(|&&(a, _)| a < end) {
            let (start, stop) = (a.max(base), b.min(end));
            if start >= stop {
                continue;
            }
            match exact.last_mut() {
                Some(last) if last.region == region && start <= last.end => last.end = last.end.max(stop),
                _ => exact.push(WatchSpan { region, start, end: stop }),
            }
        }
    }
    if exact.len() > target_calls {
        let mut gaps: Vec<(usize, usize, usize)> = exact
            .windows(2)
            .enumerate()
            .filter(|(_, pair)| pair[0].region == pair[1].region)
            .map(|(i, pair)| (pair[1].start - pair[0].end, pair[0].start, i))
            .collect();
        gaps.sort_unstable();
        let bridge_count = gaps.len().min(exact.len() - target_calls);
        let mut bridge = vec![false; exact.len().saturating_sub(1)];
        for &(_, _, i) in &gaps[..bridge_count] {
            bridge[i] = true;
        }
        let mut coalesced = Vec::<WatchSpan>::new();
        for (i, span) in exact.into_iter().enumerate() {
            match coalesced.last_mut() {
                Some(last) if bridge[i - 1] => last.end = span.end,
                _ => coalesced.push(span),
            }
        }
        exact = coalesced;
    }
    exact.sort_unstable_by_key(|s| (s.start, s.end));
    exact.into_iter().map(|s| (s.start, s.end - s.start)).collect()
}

/// The parts of `wanted` (start, end) inside `regions` (base, size), merged (gaps under
/// `WATCH_GAP` too) and sorted, as (base, size) – each within one region (a write-watch call
/// covers one allocation).
fn watch_ranges(regions: &[(usize, usize)], mut wanted: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    wanted.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in wanted {
        match merged.last_mut() {
            Some(last) if a <= last.1.saturating_add(WATCH_GAP) => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    // each region against the merged ranges it meets (sorted and disjoint: a binary search, not
    // a pass over all of them – both lists run to thousands while a game streams)
    let mut out = Vec::new();
    for &(base, size) in regions {
        let end = base + size;
        let first = merged.partition_point(|&(_, b)| b <= base);
        for &(a, b) in merged[first..].iter().take_while(|&&(a, _)| a < end) {
            let (lo, hi) = (a.max(base), b.min(end));
            if lo < hi {
                out.push((lo, hi - lo));
            }
        }
    }
    out.sort_unstable();
    out
}

/// Copy `[src, src + len)` into `dst` without faulting if another thread unmaps or
/// decommits a page concurrently: `ReadProcessMemory` catches the fault in kernel mode and returns
/// `0`. Because Windows may return `read == 0` when a fault occurs later in the span, binary-split
/// failed multi-page ranges at 4 KiB boundaries so all still-committed pages are copied in O(log N)
/// calls and only unreadable pages are zeroed.
pub(super) unsafe fn copy_committed_region(src: usize, dst: *mut u8, len: usize) {
    copy_committed(src, dst, len, true)
}

/// `copy_committed_region`, but a page that cannot be read keeps what `dst` holds (the per-draw
/// mirror sync compares it with the mirror's bytes: a page the title freed keeps the mirror's).
pub(super) unsafe fn copy_committed_pages(src: usize, dst: *mut u8, len: usize) {
    copy_committed(src, dst, len, false)
}

unsafe fn copy_committed(src: usize, dst: *mut u8, len: usize, zero: bool) {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const PAGE: usize = 0x1000;

    unsafe fn copy_rec(proc: *mut c_void, src: usize, dst: *mut u8, len: usize, zero: bool) {
        if len == 0 {
            return;
        }
        let mut read = 0usize;
        if ReadProcessMemory(proc, src as *const c_void, dst as *mut c_void, len, &mut read) != 0 {
            return;
        }
        let read = read.min(len);
        let src = src + read;
        let dst = dst.add(read);
        let len = len - read;
        let page_end = (src & !(PAGE - 1)).saturating_add(PAGE).min(src + len);
        if page_end == src + len {
            if zero {
                std::ptr::write_bytes(dst, 0, len);
            }
            return;
        }
        let mid = ((src + len / 2) & !(PAGE - 1)).max(page_end);
        let left = mid - src;
        copy_rec(proc, src, dst, left, zero);
        copy_rec(proc, mid, dst.add(left), len - left, zero);
    }

    copy_rec(GetCurrentProcess(), src, dst, len, zero);
}

/// The bytes at `at` into `to` in one read, all zeros when it fails: for memory committed whole
/// or not at all (a physical page where it is kept), with no question to Windows about it.
pub(super) unsafe fn copy_or_zeros(at: usize, to: &mut [u8]) {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut read = 0usize;
    if ReadProcessMemory(GetCurrentProcess(), at as *const c_void, to.as_mut_ptr() as *mut c_void, to.len(), &mut read) == 0 || read < to.len() {
        to.fill(0);
    }
}

/// `len` bytes at `at`: the committed, readable pages copied, the rest zero (a streaming pool
/// has only some of its pages mapped).
pub(super) unsafe fn read_committed(at: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    read_committed_into(at, &mut out);
    out
}

/// `read_committed` of `out.len()` bytes at `at` into `out` (every byte of it written).
pub(super) unsafe fn read_committed_into(at: usize, out: &mut [u8]) {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let len = out.len();
    let Some(limit) = at.checked_add(len) else { return out.fill(0) };
    // (the coherent window is committed as a whole: no `VirtualQuery`, which sizes a region by
    // walking it)
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(at, len)) {
        std::ptr::copy_nonoverlapping(at as *const u8, out.as_mut_ptr(), len);
        return;
    }
    // All of it readable (the usual case): one copy. `VirtualQuery` scans from the page asked
    // about to the end of its run of like pages, and the title's memory is a few runs of hundreds
    // of MiB – a 16 KiB texture cost ~120 µs at the start of a 64 MiB run, ~1.3 ms of 512 MiB.
    // `ReadProcessMemory` refuses a range with a page that is not committed, no-access or a guard
    // page (and leaves the guard in place: `title_bytes_leave_guard_pages_alone`). Under Wine
    // that test fails with it (#2844), so there it is region by region.
    let mut read = 0usize;
    if len == 0 {
        return;
    }
    if !crate::util::under_wine() && ReadProcessMemory(GetCurrentProcess(), at as *const c_void, out.as_mut_ptr() as *mut c_void, len, &mut read) != 0 {
        return;
    }
    // with holes: region by region (a refused `ReadProcessMemory` costs more than the walk; a
    // retail title makes its textures before it maps their memory: 13.6 GB of them went up at
    // 0.5 GB/s while its world loaded when every page was tried on its own)
    let mut p = at;
    while p < limit {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.RegionSize == 0 {
            break;
        }
        let end = (m.BaseAddress as usize).saturating_add(m.RegionSize).min(limit);
        if end <= p {
            break;
        }
        let readable = m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0; // R / RW / WC / X-R…, no guard
        if readable {
            copy_committed_region(p, out.as_mut_ptr().add(p - at), end - p);
        } else {
            out[p - at..end - at].fill(0);
        }
        p = end;
    }
    out[p - at..].fill(0);
}

thread_local! {
    /// Reusable copy of title bytes for `with_title_bytes` (a fresh one per texture cost its
    /// allocation and page faults – as much as the detile of a video frame).
    static TITLE_BYTES: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A `TITLE_BYTES` buffer grown past this is let go after its read (a thread does not keep the
/// size of the largest texture it ever read)
const TITLE_BYTES_KEPT: usize = 64 << 20;

/// `f` of the title's `len` bytes at `at`: where they are in the coherent window (committed as a
/// whole, never decommitted), by the GPU's copy in a GPU_UPLOAD segment of a mixed driver window
/// (CPU reads of that video memory through the BAR are slow), else as `read_committed` reads
/// them, into a buffer reused on this thread – the title can decommit its other memory from
/// another thread at any moment, and a read in place then faults.
pub(super) unsafe fn with_title_bytes<R>(at: usize, len: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    if let Some(bytes) = super::uma_driver::read_vram_bytes(at, len) {
        return f(&bytes);
    }
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(at, len)) {
        return f(std::slice::from_raw_parts(at as *const u8, len));
    }
    TITLE_BYTES.with(|b| match b.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < len {
                buf.resize(len, 0);
            }
            // (`read_committed_into` writes every byte: what an earlier read left does not show
            // through a hole)
            read_committed_into(at, &mut buf[..len]);
            let r = f(&buf[..len]);
            if buf.capacity() > TITLE_BYTES_KEPT {
                *buf = Vec::new();
            }
            r
        }
        Err(_) => f(&read_committed(at, len)),
    })
}

/// Every subresource (D3D12 order) of a texture from the title's memory at `va`.
#[cfg(test)]
pub(super) unsafe fn subresources(va: usize, tex: &xg::Texture) -> Vec<Subresource> {
    subresources_in(va, tex, None)
}

/// Every subresource (D3D12 order) of a texture from the title's memory at `va`, read through
/// `pages` when it has a table of them.
unsafe fn subresources_in(va: usize, tex: &xg::Texture, pages: Option<&texture_upload::PageMap>) -> Vec<Subresource> {
    let Some(tex) = tex.texels() else { return Vec::new() };
    let c = &tex.chain;
    let bpe = tex.bpe as usize;
    if tex.depth > 0 {
        return texture_upload::read(pages, va, tex.size() as usize, |mem| c.levels.iter().enumerate().map(|(i, l)| Subresource { depth: l.depth, rows: l.height, row_bytes: l.width * bpe, data: c.detile_volume(mem, i) }).collect());
    }
    let mut out = Vec::with_capacity(tex.array as usize * c.levels.len());
    for s in 0..tex.array as usize {
        texture_upload::read(pages, va + s * c.slice_bytes, c.slice_bytes, |slice| {
            for (i, l) in c.levels.iter().enumerate() {
                out.push(Subresource { depth: 1, rows: l.height, row_bytes: l.width * bpe, data: c.detile_level(slice, i, s) });
            }
        });
    }
    out
}

/// The bounded part of a mapped D3D12 readback buffer which holds one subresource.
/// `width`, `height`, and `depth` are texels (not format blocks); `size` bounds all reads.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Stage 3 seam; queue/readback activation is a later change.
pub(super) struct ReadbackFootprint {
    pub offset: usize,
    pub size: usize,
    pub row_pitch: usize,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: u32,
}

#[derive(Debug, PartialEq, Eq)]
#[allow(dead_code)] // Stage 3 seam; queue/readback activation is a later change.
pub(super) enum ReverseLayoutError {
    UnsupportedLayout,
    Format,
    Subresource,
    Dimensions,
    RowPitch,
    Footprint,
    Destination,
}

/// Put one exact PC readback footprint back into its XG-tiled subresource. This is deliberately
/// pure: callers own both byte slices, and no resource, queue, title address, or COM object is
/// touched. Metadata planes and format conversions are not reconstructible and fail closed.
#[allow(dead_code)] // Stage 3 seam; queue/readback activation is a later change.
pub(super) fn reverse_layout_subresource(
    tex: &xg::Texture,
    format: u32,
    subresource: u32,
    footprint: ReadbackFootprint,
    readback: &[u8],
    console: &mut [u8],
) -> Result<(), ReverseLayoutError> {
    if tex.main != 0 || tex.planes.len() != 1 {
        return Err(ReverseLayoutError::UnsupportedLayout);
    }
    let (surface, plane_offset) = tex.plane_surface(0).ok_or(ReverseLayoutError::UnsupportedLayout)?;
    let (elem_log2, side) = xg::element(format).ok_or(ReverseLayoutError::Format)?;
    if footprint.format != format || surface.bpe != 1u32 << elem_log2 || surface.side != side {
        return Err(ReverseLayoutError::Format);
    }
    let mips = surface.chain.levels.len();
    let (mip, slice) = if surface.depth > 0 {
        let mip = usize::try_from(subresource).map_err(|_| ReverseLayoutError::Subresource)?;
        if mip >= mips {
            return Err(ReverseLayoutError::Subresource);
        }
        (mip, None)
    } else {
        let total = mips.checked_mul(surface.array as usize).ok_or(ReverseLayoutError::Subresource)?;
        let sub = usize::try_from(subresource).map_err(|_| ReverseLayoutError::Subresource)?;
        if sub >= total {
            return Err(ReverseLayoutError::Subresource);
        }
        (sub % mips, Some(sub / mips))
    };
    let level = &surface.chain.levels[mip];
    let width = (surface.width >> mip).max(1);
    let height = (surface.height >> mip).max(1);
    let depth = if surface.depth > 0 { (surface.depth >> mip).max(1) } else { 1 };
    if footprint.width != width || footprint.height != height || footprint.depth != depth {
        return Err(ReverseLayoutError::Dimensions);
    }
    let row_bytes = level.width.checked_mul(surface.bpe as usize).ok_or(ReverseLayoutError::Footprint)?;
    if level.width != (width as usize).div_ceil(side as usize)
        || level.height != (height as usize).div_ceil(side as usize)
        || level.depth != depth as usize
    {
        return Err(ReverseLayoutError::UnsupportedLayout);
    }
    if footprint.row_pitch < row_bytes || footprint.row_pitch % 256 != 0 || footprint.offset % 512 != 0 {
        return Err(ReverseLayoutError::RowPitch);
    }
    let slice_pitch = footprint.row_pitch.checked_mul(level.height).ok_or(ReverseLayoutError::Footprint)?;
    let last_slice = slice_pitch.checked_mul(level.depth.saturating_sub(1)).ok_or(ReverseLayoutError::Footprint)?;
    let last_row = footprint.row_pitch.checked_mul(level.height.saturating_sub(1)).ok_or(ReverseLayoutError::Footprint)?;
    let used = last_slice.checked_add(last_row).and_then(|n| n.checked_add(row_bytes)).ok_or(ReverseLayoutError::Footprint)?;
    let footprint_end = footprint.offset.checked_add(footprint.size).ok_or(ReverseLayoutError::Footprint)?;
    if used > footprint.size || footprint_end > readback.len() {
        return Err(ReverseLayoutError::Footprint);
    }
    let console_size = usize::try_from(tex.size()).map_err(|_| ReverseLayoutError::Destination)?;
    if console.len() < console_size {
        return Err(ReverseLayoutError::Destination);
    }
    let linear_size = row_bytes
        .checked_mul(level.height)
        .and_then(|n| n.checked_mul(level.depth))
        .ok_or(ReverseLayoutError::Footprint)?;
    let mut linear = Vec::new();
    linear.try_reserve_exact(linear_size).map_err(|_| ReverseLayoutError::Footprint)?;
    linear.resize(linear_size, 0);
    for z in 0..level.depth {
        for y in 0..level.height {
            let from = footprint.offset + z * slice_pitch + y * footprint.row_pitch;
            let to = (z * level.height + y) * row_bytes;
            linear[to..to + row_bytes].copy_from_slice(&readback[from..from + row_bytes]);
        }
    }
    let plane_at = usize::try_from(plane_offset).map_err(|_| ReverseLayoutError::Destination)?;
    if let Some(slice) = slice {
        let at = plane_at
            .checked_add(slice.checked_mul(surface.chain.slice_bytes).ok_or(ReverseLayoutError::Destination)?)
            .ok_or(ReverseLayoutError::Destination)?;
        let end = at.checked_add(surface.chain.slice_bytes).ok_or(ReverseLayoutError::Destination)?;
        let dst = console.get_mut(at..end).ok_or(ReverseLayoutError::Destination)?;
        surface.chain.tile_level(dst, mip, slice, &linear);
    } else {
        let bytes = usize::try_from(surface.size()).map_err(|_| ReverseLayoutError::Destination)?;
        let end = plane_at.checked_add(bytes).ok_or(ReverseLayoutError::Destination)?;
        let dst = console.get_mut(plane_at..end).ok_or(ReverseLayoutError::Destination)?;
        surface.chain.tile_volume(dst, mip, &linear);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(super) struct DownloadSubresource {
    pub subresource: u32,
    pub state: D3D12_RESOURCE_STATES,
    pub footprint: ReadbackFootprint,
}

/// Validate driver footprints against the immutable XG layout and explicit tracked states.
/// The driver may add only the required D3D row/placement alignment, never enlarge the work.
fn validate_download_layout(
    tex: &xg::Texture,
    desc: &D3D12_RESOURCE_DESC,
    format: u32,
    selected: &[(u32, D3D12_RESOURCE_STATES)],
    driver: &[(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u32, u64)],
) -> Result<(Vec<DownloadSubresource>, usize), String> {
    let surface = tex.texels().ok_or("no texel surface")?;
    let (elem_log2, side) = xg::element(format).ok_or("unsupported texture format")?;
    if tex.main != 0
        || tex.planes.len() != 1
        || surface.bpe != 1u32 << elem_log2
        || surface.side != side
        || desc.Dimension.0 as u32 != tex.dimension()
        || desc.SampleDesc.Count != 1
        || desc.SampleDesc.Quality != 0
        || desc.Format.0 as u32 != format
    {
        return Err("unsupported metadata, samples, or format conversion".into());
    }
    let mips = surface.chain.levels.len();
    let layers = if surface.depth > 0 { surface.depth } else { surface.array };
    if desc.Width != surface.width as u64
        || desc.Height != surface.height
        || desc.DepthOrArraySize as u32 != layers
        || desc.MipLevels as usize != mips
        || selected.is_empty()
        || selected.len() != driver.len()
    {
        return Err("resource shape or selection differs from the retained layout".into());
    }
    let subresources = if surface.depth > 0 { mips } else { mips.checked_mul(surface.array as usize).ok_or("subresource count overflow")? };
    let mut seen = Vec::new();
    seen.try_reserve_exact(subresources).map_err(|_| "cannot allocate download selection")?;
    seen.resize(subresources, false);
    let mut out = Vec::new();
    out.try_reserve_exact(selected.len()).map_err(|_| "cannot allocate download layout")?;
    let mut total = 0usize;
    for (&(subresource, state), &(placed, rows, driver_row_bytes)) in selected.iter().zip(driver) {
        if state.0 < 0 {
            return Err("invalid tracked resource state".into());
        }
        let sub = subresource as usize;
        let (mip, depth) = if surface.depth > 0 {
            if sub >= mips { return Err("subresource out of range".into()) }
            (sub, (surface.depth >> sub).max(1))
        } else {
            let count = mips.checked_mul(surface.array as usize).ok_or("subresource count overflow")?;
            if sub >= count { return Err("subresource out of range".into()) }
            (sub % mips, 1)
        };
        if std::mem::replace(&mut seen[sub], true) { return Err("duplicate subresource".into()) }
        let level = &surface.chain.levels[mip];
        let width = (surface.width >> mip).max(1);
        let height = (surface.height >> mip).max(1);
        let row_bytes = level.width.checked_mul(surface.bpe as usize).ok_or("row size overflow")?;
        let expected_pitch = row_bytes.checked_next_multiple_of(256).ok_or("row pitch overflow")?;
        let expected_offset = total.checked_next_multiple_of(512).ok_or("footprint offset overflow")?;
        if placed.Offset != expected_offset as u64
            || placed.Footprint.Format.0 as u32 != format
            || placed.Footprint.Width != width
            || placed.Footprint.Height != height
            || placed.Footprint.Depth != depth
            || placed.Footprint.RowPitch as usize != expected_pitch
            || rows as usize != level.height
            || driver_row_bytes != row_bytes as u64
            || level.depth != depth as usize
        {
            return Err("driver footprint differs from the exact subresource".into());
        }
        let slice_pitch = expected_pitch.checked_mul(level.height).ok_or("slice pitch overflow")?;
        let used = slice_pitch
            .checked_mul(level.depth.saturating_sub(1))
            .and_then(|n| n.checked_add(expected_pitch.checked_mul(level.height.saturating_sub(1))?))
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or("footprint size overflow")?;
        total = expected_offset.checked_add(used).ok_or("download size overflow")?;
        out.push(DownloadSubresource {
            subresource,
            state,
            footprint: ReadbackFootprint { offset: expected_offset, size: used, row_pitch: expected_pitch, width, height, depth, format },
        });
    }
    Ok((out, total))
}

struct DownloadObjects {
    _source: ID3D12Resource,
    readback: ID3D12Resource,
    _allocator: ID3D12CommandAllocator,
    list: ID3D12CommandList,
    _queue: ID3D12CommandQueue,
    _fence_hold: ID3D12Fence,
}

/// A closed command list and every object it may reference. The caller must hold its state/order
/// serialization from the supplied state snapshot until `submit` inserts this list immediately
/// after the writers represented by that snapshot; otherwise the recorded barriers are stale.
#[allow(dead_code)]
pub(super) struct TextureDownloadPlan {
    device: ID3D12Device,
    queue_type: D3D12_COMMAND_LIST_TYPE,
    tex: Arc<xg::Texture>,
    source: ID3D12Resource,
    readback: ID3D12Resource,
    allocator: ID3D12CommandAllocator,
    list: ID3D12CommandList,
    fence: ID3D12Fence,
    footprints: Vec<DownloadSubresource>,
    bytes: usize,
}

#[allow(dead_code)]
pub(super) struct SubmittedDownload {
    pub ticket: PendingDownload,
    /// `Some` means Execute had no reportable result and the following Signal failed. The ticket
    /// can never be mapped and deliberately retains/leaks its objects rather than releasing them.
    pub signal_error: Option<String>,
}

#[allow(dead_code)]
pub(super) struct PendingDownload {
    tex: Arc<xg::Texture>,
    objects: Option<DownloadObjects>,
    fence: ID3D12Fence,
    fence_value: Option<u64>,
    footprints: Vec<DownloadSubresource>,
    bytes: usize,
}

impl Drop for PendingDownload {
    fn drop(&mut self) {
        let done = self.fence_value.is_some_and(|v| {
            let got = unsafe { super::real::native_fence_completed(&self.fence) };
            got != u64::MAX && got >= v
        });
        if !done {
            if let Some(objects) = self.objects.take() {
                std::mem::forget(objects);
            }
        }
    }
}

#[allow(dead_code)]
impl PendingDownload {
    pub fn is_complete(&self) -> bool {
        self.fence_value.is_some_and(|v| {
            let got = unsafe { super::real::native_fence_completed(&self.fence) };
            got != u64::MAX && got >= v
        })
    }

    /// Copy the mapped readback only after the submission fence is observed. No wait is done.
    pub unsafe fn readback_bytes(&self) -> Result<Option<Vec<u8>>, String> {
        if !self.is_complete() {
            return Ok(None);
        }
        let objects = self.objects.as_ref().ok_or("download objects unavailable")?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(self.bytes).map_err(|_| "cannot allocate completed texture download")?;
        let mut p = std::ptr::null_mut();
        objects
            .readback
            .Map(0, Some(&D3D12_RANGE { Begin: 0, End: self.bytes }), Some(&mut p))
            .map_err(|e| e.to_string())?;
        if p.is_null() {
            objects.readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            return Err("readback Map returned no address".into());
        }
        bytes.extend_from_slice(std::slice::from_raw_parts(p as *const u8, self.bytes));
        objects.readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        Ok(Some(bytes))
    }

    pub fn footprints(&self) -> &[DownloadSubresource] { &self.footprints }
    pub fn layout(&self) -> &xg::Texture { &self.tex }

    fn arm(&self, event: windows::Win32::Foundation::HANDLE) {
        if let Some(value) = self.fence_value { unsafe { let _ = self.fence.SetEventOnCompletion(value, event); } }
    }
}

#[allow(dead_code)]
impl TextureDownloadPlan {
    /// Execute the already-closed list at the caller's serialized queue position. Ownership is
    /// transferred to the pending ticket before Execute because ExecuteCommandLists has no status.
    pub unsafe fn submit(self, queue: &ID3D12CommandQueue) -> Result<SubmittedDownload, String> {
        let mut queue_device: Option<ID3D12Device> = None;
        queue.GetDevice(&mut queue_device).map_err(|e| e.to_string())?;
        if queue.GetDesc().Type != self.queue_type
            || queue_device.as_ref().is_none_or(|d| d.as_raw() != self.device.as_raw())
        {
            return Err("texture download queue type or source device differs from its plan".into());
        }
        let mut pending = PendingDownload {
            tex: self.tex,
            objects: Some(DownloadObjects {
                _source: self.source,
                readback: self.readback,
                _allocator: self.allocator,
                list: self.list,
                _queue: queue.clone(),
                _fence_hold: self.fence.clone(),
            }),
            fence: self.fence,
            fence_value: None,
            footprints: self.footprints,
            bytes: self.bytes,
        };
        let list = pending.objects.as_ref().unwrap().list.clone();
        super::real::native_queue_execute(queue, &list);
        let signal_error = match super::real::native_queue_signal(queue, &pending.fence, 1) {
            Ok(()) => { pending.fence_value = Some(1); None }
            Err(e) => Some(e.to_string()),
        };
        Ok(SubmittedDownload { ticket: pending, signal_error })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DownloadWrite { subresource: u32, state: D3D12_RESOURCE_STATES, writer: GpuWriter }

pub(super) struct ReadyDownload { res: usize, generation: u64, target: usize, format: u32, writes: Vec<DownloadWrite>, ticket: PendingDownload }

struct FenceDownloads { fence: ID3D12Fence, value: u64, downloads: Vec<Option<ReadyDownload>> }

#[derive(Default)]
struct FenceDownloadState {
    batches: Vec<FenceDownloads>,
    /// A detached fence is being published outside this mutex. Its mirrored completion must
    /// remain hidden until the owner restores any work which could not be published.
    processing: BTreeSet<usize>,
    /// Ties which have claimed a fence but have not installed their batch yet. Native polling,
    /// event arming, and reservation bookkeeping stay outside this mutex during that interval.
    tying: BTreeMap<usize, usize>,
}

impl FenceDownloadState {
    /// Register an installer. True means this is the first pending work for this fence and must
    /// ensure that a safe baseline exists. Production fences are AddRef-retained for the process
    /// lifetime, so their raw addresses cannot be reused while this state matters.
    fn begin_tie(&mut self, fence: usize) -> bool {
        let initialize_safe = !self.processing.contains(&fence)
            && !self.tying.contains_key(&fence)
            && !self.batches.iter().any(|batch| batch.fence.as_raw() as usize == fence);
        *self.tying.entry(fence).or_insert(0) += 1;
        initialize_safe
    }

    fn finish_tie(&mut self, fence: usize) {
        let count = self.tying.get_mut(&fence).expect("finishing an unregistered texture download tie");
        *count -= 1;
        if *count == 0 { self.tying.remove(&fence); }
    }

    fn tie_in_progress(&self, fence: usize) -> bool {
        self.tying.contains_key(&fence)
    }

    fn begin_processing(&mut self, fence: usize) -> bool {
        !self.tie_in_progress(fence) && self.processing.insert(fence)
    }

    fn any_pending(&self) -> bool {
        !self.batches.is_empty() || !self.processing.is_empty() || !self.tying.is_empty()
    }
}

static FENCE_DOWNLOADS: Mutex<FenceDownloadState> = Mutex::new(FenceDownloadState {
    batches: Vec::new(),
    processing: BTreeSet::new(),
    tying: BTreeMap::new(),
});
static DOWNLOADS_PENDING: AtomicBool = AtomicBool::new(false);
static SCHEDULED_DOWNLOADS: Mutex<BTreeSet<(usize, u64, u64, u32, u64)>> = Mutex::new(BTreeSet::new());
static SAFE_FENCE_VALUES: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());

fn ensure_safe_baseline(values: &mut BTreeMap<usize, u64>, fence: usize) {
    values.entry(fence).or_insert(0);
}

fn note_safe_value(values: &mut BTreeMap<usize, u64>, fence: usize, value: u64) {
    values.entry(fence).and_modify(|safe| *safe = (*safe).max(value)).or_insert(value);
}

fn writer_at(entry: &Entry, subresource: usize) -> Option<GpuWriter> {
    match entry.gpu_writers.as_slice() {
        [] => None, [writer] => *writer, writers => writers.get(subresource).copied().flatten(),
    }
}

fn download_writes(res: usize, entry: &Entry, queue: WriterQueueId) -> Vec<DownloadWrite> {
    let n = subresource_count(&entry.tex);
    let scheduled = SCHEDULED_DOWNLOADS.lock().unwrap();
    (0..n).filter_map(|subresource| {
        let writer = writer_at(entry, subresource)?;
        (writer.queue == queue && !scheduled.contains(&(res, entry.generation, writer.queue.generation, subresource as u32, writer.epoch)))
            .then(|| DownloadWrite { subresource: subresource as u32, state: if entry.states.len() == n { entry.states[subresource] } else { entry.states[0] }, writer })
    }).collect()
}

struct DownloadSnapshot { res: usize, generation: u64, target: usize, format: u32, tex: Arc<xg::Texture>, device: ID3D12Device, source: ID3D12Resource, writes: Vec<DownloadWrite>, plan: Option<TextureDownloadPlan> }

pub(super) struct PreparedDownloads { id: WriterQueueId, snapshots: Vec<DownloadSnapshot> }

pub(super) enum SubmitDownloadsError { Stale, Failed(String) }

fn download_current(entry: &Entry, generation: u64, target: usize, writes: &[DownloadWrite]) -> bool {
    entry.generation == generation && entry.src == target && writes.iter().all(|w| writer_at(entry, w.subresource as usize) == Some(w.writer))
}

/// Build readback lists outside the queue turn. The state is validated again at insertion.
/// Writes owned by another queue are left for a signal on that exact queue.
pub(super) unsafe fn prepare_downloads(queue: *mut c_void) -> PreparedDownloads {
    let id = super::list_x::queue_identity(queue as usize).unwrap_or(WriterQueueId { raw: queue as usize, generation: 0 });
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return PreparedDownloads { id, snapshots: Vec::new() } };
    let queue_type = q.GetDesc().Type;
    if !download_queue_type_supported(queue_type) { return PreparedDownloads { id, snapshots: Vec::new() } }
    let candidates: Vec<DownloadSnapshot> = {
        let table = TEXTURES.lock().unwrap();
        table.by_res.iter().filter_map(|(&res, entry)| {
            let writes = download_writes(res, entry, id);
            if writes.is_empty() { return None }
            let source = ID3D12Resource::from_raw_borrowed(&(res as *mut c_void))?.clone();
            let device = ID3D12Device::from_raw_borrowed(&(entry.device as *mut c_void))?.clone();
            Some(DownloadSnapshot { res, generation: entry.generation, target: entry.src, format: source.GetDesc().Format.0 as u32,
                tex: Arc::clone(&entry.tex), device, source, writes, plan: None })
        }).collect()
    };
    let mut snapshots = Vec::new();
    for mut snapshot in candidates {
        let selected: Vec<_> = snapshot.writes.iter().map(|w| (w.subresource, w.state)).collect();
        let Ok(plan) = prepare_texture_download(&snapshot.device, &snapshot.source, queue_type, Arc::clone(&snapshot.tex), snapshot.format, &selected) else { continue };
        snapshot.plan = Some(plan);
        snapshots.push(snapshot);
    }
    PreparedDownloads { id, snapshots }
}

/// Validate and insert prepared lists while the caller owns `queue`'s turn. A stale snapshot
/// submits nothing, so the caller can rebuild outside the turn and retry.
pub(super) unsafe fn submit_prepared_downloads(queue: *mut c_void, prepared: PreparedDownloads) -> Result<Vec<ReadyDownload>, SubmitDownloadsError> {
    if prepared.snapshots.is_empty() { return Ok(Vec::new()) }
    if super::list_x::queue_identity(queue as usize) != Some(prepared.id) { return Err(SubmitDownloadsError::Stale) }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return Ok(Vec::new()) };
    let table = TEXTURES.lock().unwrap();
    let mut scheduled = SCHEDULED_DOWNLOADS.lock().unwrap();
    let valid = prepared.snapshots.iter().all(|snapshot| {
        table.by_res.get(&snapshot.res).is_some_and(|entry| {
            entry.generation == snapshot.generation
                && entry.src == snapshot.target
                && snapshot.writes.iter().all(|w| {
                    !scheduled.contains(&(snapshot.res, snapshot.generation, w.writer.queue.generation, w.subresource, w.writer.epoch))
                        &&
                    writer_at(entry, w.subresource as usize) == Some(w.writer)
                        && (if entry.states.len() == subresource_count(&entry.tex) {
                            entry.states[w.subresource as usize]
                        } else {
                            entry.states[0]
                        }) == w.state
                })
        })
    });
    if !valid { return Err(SubmitDownloadsError::Stale) }
    let keys: Vec<_> = prepared.snapshots.iter().flat_map(|snapshot| snapshot.writes.iter().map(|write| {
        (snapshot.res, snapshot.generation, write.writer.queue.generation, write.subresource, write.writer.epoch)
    })).collect();
    for &key in &keys { scheduled.insert(key); }
    drop(scheduled);
    drop(table);
    let mut ready = Vec::new();
    for snapshot in prepared.snapshots {
        let submitted = match snapshot.plan.unwrap().submit(q) {
            Ok(submitted) => submitted,
            Err(error) => { rollback_reservations(&keys); return Err(SubmitDownloadsError::Failed(error)) }
        };
        if let Some(error) = submitted.signal_error { rollback_reservations(&keys); return Err(SubmitDownloadsError::Failed(format!("texture download completion signal failed: {error}"))) }
        ready.push(ReadyDownload { res: snapshot.res, generation: snapshot.generation, target: snapshot.target, format: snapshot.format, writes: snapshot.writes, ticket: submitted.ticket });
    }
    Ok(ready)
}

fn rollback_reservations(keys: &[(usize, u64, u64, u32, u64)]) {
    let mut scheduled = SCHEDULED_DOWNLOADS.lock().unwrap();
    for key in keys { scheduled.remove(key); }
}

/// Tie already-submitted readbacks to the title fence queued immediately after them.
pub(super) unsafe fn tie_downloads(fence: *mut c_void, value: u64, downloads: Vec<ReadyDownload>) {
    if downloads.is_empty() { return }
    let Some(fence) = ID3D12Fence::from_raw_borrowed(&fence).cloned() else { return };
    let key = fence.as_raw() as usize;
    {
        let mut state = FENCE_DOWNLOADS.lock().unwrap();
        if state.begin_tie(key) {
            ensure_safe_baseline(&mut SAFE_FENCE_VALUES.lock().unwrap(), key);
        }
        DOWNLOADS_PENDING.store(true, Ordering::Release);
    }
    let event = super::fence_x::wake_event();
    for download in &downloads { download.ticket.arm(event); }
    {
        let mut scheduled = SCHEDULED_DOWNLOADS.lock().unwrap();
        for download in &downloads { for write in &download.writes {
            scheduled.insert((download.res, download.generation, write.writer.queue.generation, write.subresource, write.writer.epoch));
        } }
    }
    {
        let mut state = FENCE_DOWNLOADS.lock().unwrap();
        state.batches.push(FenceDownloads { fence: fence.clone(), value, downloads: downloads.into_iter().map(Some).collect() });
        state.finish_tie(key);
        DOWNLOADS_PENDING.store(true, Ordering::Release);
    }
    let _ = fence.SetEventOnCompletion(value, event);
}

fn publish_download(download: &ReadyDownload) -> Result<bool, String> {
    let Some(readback) = (unsafe { download.ticket.readback_bytes()? }) else { return Ok(false) };
    let size = usize::try_from(download.ticket.layout().size()).map_err(|_| "texture download destination is too large")?;
    let current = unsafe { with_title_bytes(download.target, size, <[u8]>::to_vec) };
    let mut console = Vec::new();
    console.try_reserve_exact(size).map_err(|_| "cannot allocate texture download destination")?;
    console.extend_from_slice(&current);
    for (write, footprint) in download.writes.iter().zip(download.ticket.footprints()) {
        reverse_layout_subresource(download.ticket.layout(), download.format, write.subresource, footprint.footprint, &readback, &mut console)
            .map_err(|e| format!("cannot reverse texture download layout: {e:?}"))?;
    }
    let mut table = TEXTURES.lock().unwrap();
    let Some(entry) = table.by_res.get_mut(&download.res) else { return Ok(true) };
    if !download_current(entry, download.generation, download.target, &download.writes) { return Ok(true) }
    if !unsafe { write_changed_pages(download.target, &current, &console) } { return Ok(false) }
    let n = subresource_count(&entry.tex);
    if entry.gpu_writers.len() != n {
        entry.gpu_writers = vec![entry.gpu_writers.first().copied().flatten(); n];
    }
    for write in &download.writes {
        let slot = &mut entry.gpu_writers[write.subresource as usize];
        if *slot == Some(write.writer) { *slot = None; }
    }
    if entry.gpu_writers.iter().all(Option::is_none) {
        entry.gpu_writers.clear();
        entry.gpu_written = false;
    }
    Ok(true)
}

/// Write changed 4 KiB pages without faulting if the title concurrently unmaps one. Every
/// changed page must be committed and writable; otherwise its fence remains hidden and retries.
unsafe fn write_changed_pages(at: usize, old: &[u8], new: &[u8]) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if old.len() != new.len() { return false }
    let mut offset = 0;
    while offset < new.len() {
        let end = (((at + offset) & !4095) + 4096).saturating_sub(at).min(new.len());
        if old[offset..end] == new[offset..end] { offset = end; continue }
        let address = at + offset;
        let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(address as *const c_void, &mut info, std::mem::size_of_val(&info)) == 0
            || info.State != MEM_COMMIT
            || info.Protect & 0xCC == 0
            || info.Protect & PAGE_GUARD != 0
            || (info.BaseAddress as usize).saturating_add(info.RegionSize) < at + end
        {
            return false;
        }
        let mut written = 0;
        if WriteProcessMemory(GetCurrentProcess(), address as *const c_void, new[offset..end].as_ptr() as *const c_void, end - offset, &mut written) == 0
            || written != end - offset {
            return false;
        }
        offset = end;
    }
    true
}

fn unschedule(download: &ReadyDownload) {
    unschedule_writes(download.res, download.generation, &download.writes);
}

fn unschedule_writes(res: usize, generation: u64, writes: &[DownloadWrite]) {
    let mut scheduled = SCHEDULED_DOWNLOADS.lock().unwrap();
    for write in writes { scheduled.remove(&(res, generation, write.writer.queue.generation, write.subresource, write.writer.epoch)); }
}

fn batch_ready(blocked: &mut bool, target: u64, completed: u64) -> bool {
    if *blocked || target > completed { *blocked = true; false } else { true }
}

fn take_matching<T>(values: &mut Vec<T>, mut matches: impl FnMut(&T) -> bool) -> Vec<T> {
    let mut taken = Vec::new();
    let mut kept = Vec::with_capacity(values.len());
    for value in values.drain(..) {
        if matches(&value) { taken.push(value) } else { kept.push(value) }
    }
    *values = kept;
    taken
}

/// Put detached older work before work tied to the same fence while it was being processed.
fn restore_before_matching<T>(values: &mut Vec<T>, mut older: Vec<T>, matches: impl FnMut(&T) -> bool) {
    older.extend(take_matching(values, matches));
    values.extend(older);
}

fn matching_through<T>(values: &[T], completed: u64, mut target: impl FnMut(&T) -> Option<u64>) -> bool {
    values.iter().any(|value| target(value).is_some_and(|target| target <= completed))
}

/// Apply every completed download through `value`. False withholds the title's mirrored fence
/// value until a transiently unavailable readback or destination can be retried.
pub(super) fn complete_fence_downloads(fence: usize, value: u64) -> bool {
    if !DOWNLOADS_PENDING.load(Ordering::Acquire) { return true }
    let mut batches = {
        let mut state = FENCE_DOWNLOADS.lock().unwrap();
        if !state.begin_processing(fence) {
            return false;
        }
        let batches = take_matching(&mut state.batches, |batch| batch.fence.as_raw() as usize == fence);
        if batches.is_empty() {
            state.processing.remove(&fence);
            note_safe_value(&mut SAFE_FENCE_VALUES.lock().unwrap(), fence, value);
            return true;
        }
        batches
    };
    let mut ok = true;
    let mut blocked = false;
    for batch in &mut batches {
        if !batch_ready(&mut blocked, batch.value, value) { ok &= batch.value > value; continue }
        for download in &mut batch.downloads {
            let Some(ready) = download.as_ref() else { continue };
            match publish_download(ready) {
                Ok(true) => {
                    unschedule(ready);
                    *download = None;
                }
                Ok(false) => {
                    ok = false;
                    blocked = true;
                }
                Err(error) => {
                    crate::xlog_warn!("D3D12.X: {error}; holding title fence {fence:#x} at {value:#x}");
                    ok = false;
                    blocked = true;
                }
            }
        }
        blocked |= batch.downloads.iter().any(Option::is_some);
    }
    batches.retain(|batch| batch.downloads.iter().any(Option::is_some));
    {
        let mut state = FENCE_DOWNLOADS.lock().unwrap();
        // A tie which arrived while this fence was detached was not in `batches`. If the caller's
        // completed value already covers it, this pass cannot expose that value until another pass
        // publishes the arrival. Check while holding the same lock which accepts new ties.
        ok &= !matching_through(&state.batches, value, |batch| {
            (batch.fence.as_raw() as usize == fence).then_some(batch.value)
        });
        ok &= !state.tie_in_progress(fence);
        restore_before_matching(&mut state.batches, batches, |batch| batch.fence.as_raw() as usize == fence);
        state.processing.remove(&fence);
        if ok {
            note_safe_value(&mut SAFE_FENCE_VALUES.lock().unwrap(), fence, value);
        }
        DOWNLOADS_PENDING.store(state.any_pending(), Ordering::Release);
    }
    ok
}

pub(super) fn safe_fence_value(fence: usize, native: u64) -> u64 {
    if !DOWNLOADS_PENDING.load(Ordering::Acquire) {
        note_safe_value(&mut SAFE_FENCE_VALUES.lock().unwrap(), fence, native);
        return native;
    }
    if native != u64::MAX && complete_fence_downloads(fence, native) {
        native
    } else {
        SAFE_FENCE_VALUES.lock().unwrap().get(&fence).copied().unwrap_or(0)
    }
}

/// Poll pending title fences even when a title never requested a completed-value mirror.
pub(super) fn poll_fence_downloads() {
    if !DOWNLOADS_PENDING.load(Ordering::Acquire) { return }
    let pending: Vec<_> = {
        let state = FENCE_DOWNLOADS.lock().unwrap();
        state.batches.iter().map(|batch| batch.fence.clone()).collect()
    };
    for fence in pending {
        let value = unsafe { super::real::native_fence_completed(&fence) };
        if value != u64::MAX {
            let _ = complete_fence_downloads(fence.as_raw() as usize, value);
        }
    }
}

fn download_queue_type_supported(ty: D3D12_COMMAND_LIST_TYPE) -> bool {
    matches!(ty, D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE)
}

/// Build but do not submit a download. Preparation performs no global lookup or queue operation.
#[allow(dead_code)]
pub(super) unsafe fn prepare_texture_download(
    dev: &ID3D12Device,
    source: &ID3D12Resource,
    queue_type: D3D12_COMMAND_LIST_TYPE,
    tex: Arc<xg::Texture>,
    format: u32,
    selected: &[(u32, D3D12_RESOURCE_STATES)],
) -> Result<TextureDownloadPlan, String> {
    if !download_queue_type_supported(queue_type) {
        return Err("texture download requires a direct or compute queue".into());
    }
    let mut source_device: Option<ID3D12Device> = None;
    source.GetDevice(&mut source_device).map_err(|e| e.to_string())?;
    if source_device.as_ref().is_none_or(|d| d.as_raw() != dev.as_raw()) {
        return Err("texture download source belongs to another device".into());
    }
    let desc = source.GetDesc();
    let mut queried = Vec::new();
    queried.try_reserve_exact(selected.len()).map_err(|_| "cannot allocate driver footprints")?;
    let mut base = 0u64;
    for &(subresource, _) in selected {
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut rows, mut row_bytes, mut ignored_total) = (0u32, 0u64, 0u64);
        dev.GetCopyableFootprints(&desc, subresource, 1, base, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut ignored_total));
        queried.push((fp, rows, row_bytes));
        let pitch = fp.Footprint.RowPitch as u64;
        let slice_pitch = pitch.checked_mul(rows as u64).ok_or("driver footprint overflow")?;
        base = fp.Offset
            .checked_add(slice_pitch.checked_mul(fp.Footprint.Depth.saturating_sub(1) as u64).ok_or("driver footprint overflow")?)
            .and_then(|n| n.checked_add(pitch.checked_mul(rows.saturating_sub(1) as u64)?))
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or("driver footprint overflow")?;
    }
    let (footprints, bytes) = validate_download_layout(&tex, &desc, format, selected, &queried)?;
    let allocator = dev.CreateCommandAllocator(queue_type).map_err(|e| e.to_string())?;
    let graphics: ID3D12GraphicsCommandList = dev.CreateCommandList(0, queue_type, &allocator, None).map_err(|e| e.to_string())?;
    let mut readback = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        &mut readback,
    ).map_err(|e| e.to_string())?;
    let readback = readback.ok_or("no readback resource")?;
    let barrier = |subresource, before, after| D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
            pResource: std::mem::transmute_copy(source), Subresource: subresource, StateBefore: before, StateAfter: after,
        }) },
    };
    for sub in &footprints {
        if sub.state != D3D12_RESOURCE_STATE_COPY_SOURCE {
            graphics.ResourceBarrier(&[barrier(sub.subresource, sub.state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        }
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: sub.footprint.offset as u64,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(format as i32), Width: sub.footprint.width, Height: sub.footprint.height, Depth: sub.footprint.depth, RowPitch: sub.footprint.row_pitch as u32 },
        } } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(source), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub.subresource } };
        graphics.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        if sub.state != D3D12_RESOURCE_STATE_COPY_SOURCE {
            graphics.ResourceBarrier(&[barrier(sub.subresource, D3D12_RESOURCE_STATE_COPY_SOURCE, sub.state)]);
        }
    }
    graphics.Close().map_err(|e| e.to_string())?;
    let list: ID3D12CommandList = graphics.cast().map_err(|e| e.to_string())?;
    let fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| e.to_string())?;
    Ok(TextureDownloadPlan { device: dev.clone(), queue_type, tex, source: source.clone(), readback, allocator, list, fence, footprints, bytes })
}


#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A 64x64 R8 texture's upload from memory full of `value`, with that memory.
    fn r8_item(texture: &ID3D12Resource, value: u8) -> (texture_upload::Item, Vec<u8>) {
        let tex = Arc::new(xg::Texture::new(&console_desc(3, 64, 64, 1, 1, 61, 1)).unwrap());
        let mem = vec![value; tex.size() as usize];
        (texture_upload::Item { res: texture.clone(), src: mem.as_ptr() as usize, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], changed: None, known: None, pages: None }, mem)
    }

    unsafe fn read_r8(dev: &ID3D12Device, queue: &ID3D12CommandQueue, texture: &ID3D12Resource) -> Vec<u8> {
        own_work(|| {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0;
            dev.GetCopyableFootprints(&texture.GetDesc(), 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
            let mut readback = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() },
                D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback,
            ).unwrap();
            let readback: ID3D12Resource = readback.unwrap();
            let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
            let barrier = |before, after| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(texture), Subresource: 0, StateBefore: before, StateAfter: after }) },
                ..Default::default()
            };
            list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(texture), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            crate::gpu::fence::wait(&fence, 1, dev, None, 5_000).unwrap();
            let mut data = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut data)).unwrap();
            let mut out = Vec::with_capacity(64 * 64);
            for y in 0..64 {
                out.extend_from_slice(std::slice::from_raw_parts((data as *const u8).add(footprint.Offset as usize + y * footprint.Footprint.RowPitch as usize), 64));
            }
            readback.Unmap(0, None);
            out
        })
    }

    /// The title queue's upload state for a test: none left by an earlier queue at the same address,
    /// and none left behind.
    struct DirectQueueUploads(usize);

    impl DirectQueueUploads {
        fn new(queue: &ID3D12CommandQueue) -> Self {
            let key = queue.as_raw() as usize;
            let stale = ON_QUEUE.lock().unwrap().remove(&key);
            drop(stale);
            Self(key)
        }
    }

    impl Drop for DirectQueueUploads {
        fn drop(&mut self) {
            let state = ON_QUEUE.lock().unwrap().remove(&self.0);
            drop(state);
        }
    }

    #[test]
    fn gpu_uploads_on_the_title_queue_run_before_its_later_work_and_are_released() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC};

        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap();
        let _guard = TEST_LOCK.lock().unwrap();
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        unsafe {
            let raw_device = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let compute: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            assert!(title_queue_for(dev, Some(&queue)).is_some());
            assert!(title_queue_for(dev, Some(&compute)).is_none());
            assert!(title_queue_for(dev, None).is_none());
            let mut texture = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
                D3D12_RESOURCE_STATE_COMMON, None, &mut texture,
            ).unwrap();
            let texture: ID3D12Resource = texture.unwrap();
            let key = queue.as_raw() as usize;
            let _queue_uploads = DirectQueueUploads::new(&queue);
            let upload = |value: u8| {
                let (item, _mem) = r8_item(&texture, value);
                own_work(|| submit_on(dev, &queue, &[item])).unwrap();
            };
            for value in [0x21, 0xC4] {
                upload(value);
                assert_eq!(read_r8(dev, &queue, &texture), vec![value; 64 * 64], "the title queue's next work sees the upload");
            }
            upload(0x5E);
            {
                let state = ON_QUEUE.lock().unwrap().get(&key).unwrap().clone();
                let mut slot = state.lock().unwrap();
                let q = slot.as_mut().unwrap();
                assert_eq!(q.value, 3);
                let retired = reclaim(q, 0, 10_000);
                assert!(q.in_flight.is_empty() && q.bytes == 0, "over the cap the oldest uploads are waited for and released");
                assert!(q.fence.GetCompletedValue() >= 3);
                drop(slot);
                retired.into_iter().for_each(texture_upload::give_back);
            }
            assert_eq!(read_r8(dev, &queue, &texture), vec![0x5E; 64 * 64]);
        }
    }

    /// An upload on XCL's own queue (for a compute queue's submit) is waited for by the next submit
    /// on each queue, on the GPU and once; the thread that submits does not wait.
    #[test]
    fn gpu_every_queue_waits_once_for_the_uploads_on_the_own_queue() {
        let _guard = TEST_LOCK.lock().unwrap();
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        unsafe {
            let raw_device = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let was = OWN_UPLOADS.lock().unwrap().replace((gate.clone(), 1));
            let types = [D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY];
            let queues: Vec<ID3D12CommandQueue> = types.iter().map(|&t| dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: t, ..Default::default() }).unwrap()).collect();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let t0 = std::time::Instant::now();
            for (k, q) in queues.iter().enumerate() {
                OWN_WAITED.lock().unwrap().remove(&(q.as_raw() as usize));
                wait_for_own_uploads(q);
                // (a second time: nothing more to wait for)
                wait_for_own_uploads(q);
                let own: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                q.Signal(&own, 1).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(20));
                assert_eq!(own.GetCompletedValue(), 0, "queue {k} went on without the upload");
                q.Signal(&done, k as u64 + 1).unwrap();
            }
            assert!(t0.elapsed().as_millis() < 2_000, "the submitting thread waited");
            gate.Signal(1).unwrap();
            for (k, q) in queues.iter().enumerate() {
                let own: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                q.Signal(&own, 1).unwrap();
                crate::gpu::fence::wait(&own, 1, dev, None, 5_000).unwrap_or_else(|e| panic!("queue {k}: {e:?}"));
                OWN_WAITED.lock().unwrap().remove(&(q.as_raw() as usize));
            }
            *OWN_UPLOADS.lock().unwrap() = was;
        }
    }

    /// Uploads of one texture that go to different queues end in the order they were made, whichever
    /// queue stands still: an older one on XCL's own queue behind a newer one of a direct queue's
    /// submit, and an older one on a direct queue behind a newer one of a compute queue's.
    #[test]
    fn gpu_uploads_of_a_texture_on_two_queues_keep_their_order() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC};
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap();
        let _guard = TEST_LOCK.lock().unwrap();
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        unsafe {
            let raw_device = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap();
            let Some(mem) = crate::gpu::uma::get().and_then(|u| u.alloc(0x1_0000, 0x1_0000)) else { return };
            let direct: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let compute: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let own = upload_queue(dev, raw_device as usize).unwrap();
            let mut texture = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
                D3D12_RESOURCE_STATE_COMMON, None, &mut texture,
            ).unwrap();
            let texture: ID3D12Resource = texture.unwrap();
            let key = texture.as_raw() as usize;
            std::ptr::write_bytes(mem, 0, 0x1_0000);
            register(raw_device, texture.as_raw(), mem as usize, &console_desc(3, 64, 64, 1, 1, DXGI_FORMAT_R8_UNORM.0 as u32, 1), D3D12_RESOURCE_STATE_COMMON).unwrap();
            let write = |value: u8, q: &ID3D12CommandQueue| {
                std::ptr::write_bytes(mem, value, 0x1_0000);
                note_known_write(mem as usize, 0x1_0000);
                refresh(q.as_raw());
            };
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            write(0x11, &direct);
            assert_eq!(read_r8(dev, &direct, &texture), vec![0x11; 64 * 64]);

            // XCL's own queue stands; an upload for the compute queue is behind that, then one for
            // the direct queue, which has to wait for it
            own.Wait(&gate, 1).unwrap();
            write(0x22, &compute);
            write(0x33, &direct);
            let after: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            direct.Signal(&after, 1).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(30));
            assert_eq!(super::super::real::fence_completed(&after), 0, "the direct queue did not wait for the older upload");
            gate.Signal(1).unwrap();
            assert_eq!(read_r8(dev, &direct, &texture), vec![0x33; 64 * 64], "the newer upload is the one that stays");

            // the direct queue stands with an upload on it; one for the compute queue runs at once on
            // XCL's own queue, and goes up again behind the older one with the direct queue's next
            direct.Wait(&gate, 2).unwrap();
            write(0x44, &direct);
            write(0x55, &compute);
            assert!(AGAIN.lock().unwrap().contains(&key), "it can be overtaken");
            refresh(compute.as_raw());
            assert!(AGAIN.lock().unwrap().contains(&key), "not behind the older one yet");
            refresh(direct.as_raw());
            assert!(!AGAIN.lock().unwrap().contains(&key), "went up again on the direct queue");
            gate.Signal(2).unwrap();
            assert_eq!(read_r8(dev, &direct, &texture), vec![0x55; 64 * 64], "the newer upload is the one that stays");
            // with nothing pending nothing goes up again
            write(0x66, &compute);
            texture::wait(dev, &own).unwrap();
            assert!(!AGAIN.lock().unwrap().contains(&key));
            refresh(direct.as_raw());
            assert_eq!(read_r8(dev, &direct, &texture), vec![0x66; 64 * 64]);
            forget(key);
            crate::gpu::uma::get().unwrap().free(mem as usize);
            dev.GetDeviceRemovedReason().unwrap();
        }
    }

    /// An upload on the title's queue waits there for the texture page mappings made before it,
    /// as one on XCL's upload queue does: without the wait its copy can run into tiles whose
    /// mapping has not run yet (a copy-engine write page fault on NVIDIA).
    #[test]
    fn gpu_a_title_queue_upload_waits_for_the_page_mappings_before_it() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC};

        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap();
        let _guard = TEST_LOCK.lock().unwrap();
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        unsafe {
            let raw_device = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let Some(mapped) = super::super::vram::note_mapping_for_test(dev) else {
                crate::gpu::recomp::runner::skip_test("no texture pages on this device");
                return;
            };
            let key = queue.as_raw() as usize;
            assert!(super::super::vram::waited_for_test(raw_device as usize, key) < mapped);
            let mut texture = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
                D3D12_RESOURCE_STATE_COMMON, None, &mut texture,
            ).unwrap();
            let texture: ID3D12Resource = texture.unwrap();
            let _queue_uploads = DirectQueueUploads::new(&queue);
            let (item, _mem) = r8_item(&texture, 0x3C);
            own_work(|| submit_on(dev, &queue, &[item])).unwrap();
            assert_eq!(
                super::super::vram::waited_for_test(raw_device as usize, key),
                mapped,
                "the title queue's upload must wait for the page mappings made before it"
            );
            assert_eq!(read_r8(dev, &queue, &texture), vec![0x3C; 64 * 64]);
        }
    }

    #[test]
    fn gpu_an_upload_still_running_is_kept_when_its_wait_times_out() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC};

        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap();
        let _guard = TEST_LOCK.lock().unwrap();
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        unsafe {
            let raw_device = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            let mut texture = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
                D3D12_RESOURCE_STATE_COMMON, None, &mut texture,
            ).unwrap();
            let texture: ID3D12Resource = texture.unwrap();
            let key = queue.as_raw() as usize;
            let _queue_uploads = DirectQueueUploads::new(&queue);
            let (item, _mem) = r8_item(&texture, 0x77);
            own_work(|| submit_on(dev, &queue, &[item])).unwrap();
            {
                let state = ON_QUEUE.lock().unwrap().get(&key).unwrap().clone();
                let mut slot = state.lock().unwrap();
                let q = slot.as_mut().unwrap();
                let retired = reclaim(q, 0, 50);
                assert_eq!(q.in_flight.len(), 1, "a batch the GPU has not run stays in flight when its wait times out");
                assert!(q.bytes > 0 && retired.is_empty());
            }
            gate.Signal(1).unwrap();
            {
                let state = ON_QUEUE.lock().unwrap().get(&key).unwrap().clone();
                let mut slot = state.lock().unwrap();
                let q = slot.as_mut().unwrap();
                let retired = reclaim(q, 0, 10_000);
                assert!(q.in_flight.is_empty() && q.bytes == 0);
                drop(slot);
                retired.into_iter().for_each(texture_upload::give_back);
            }
            assert_eq!(read_r8(dev, &queue, &texture), vec![0x77; 64 * 64]);
        }
    }

    /// Real uploads, dirty-page detection and a source changing after registration:
    /// bookkeeping alone cannot show that the bytes the GPU sees follow the mirror.
    #[test]
    fn gpu_mirror_upload_follows_cpu_writes_and_later_mapping_changes() {
        use crate::hle::kernelx;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC};
        use windows_sys::Win32::System::Memory as mem;

        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        // Larger than all four UMA windows together. These public allocations
        // must use the watched Windows fallback, reserving space rather than RAM.
        const SIZE: usize = 24usize << 30;
        const OFFSET: usize = 0x2_0000;
        const PAGE: usize = 0x1_0000;
        struct Reservation(usize);
        impl Drop for Reservation {
            fn drop(&mut self) {
                unsafe { kernelx::VirtualFree(self.0 as *mut c_void, 0, mem::MEM_RELEASE); }
            }
        }
        struct RegisteredTexture(usize);
        impl Drop for RegisteredTexture {
            fn drop(&mut self) { forget(self.0); }
        }
        struct PhysicalPage { id: usize, at: usize }
        impl Drop for PhysicalPage {
            fn drop(&mut self) {
                unsafe {
                    if kernelx::physical_page_at(self.at) {
                        kernelx::MapTitlePhysicalPages(self.at as *mut c_void, 1, 0, 4, std::ptr::null_mut());
                    }
                    kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, &mut self.id);
                }
            }
        }
        unsafe fn reserve(protect: u32) -> Reservation {
            let at = kernelx::VirtualAlloc(std::ptr::null(), SIZE, 0x2040_2000, protect) as usize;
            assert_ne!(at, 0, "reserve-only fallback must succeed");
            let reservation = Reservation(at);
            assert!(!crate::gpu::uma::get().unwrap().contains(at), "exercise the Windows fallback");
            assert!(kernelx::in_graphics_reservation(at), "fallback must be write-watched");
            reservation
        }
        unsafe {
            let raw_device = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let _queue_uploads = DirectQueueUploads::new(&queue);
            let cpu = reserve(mem::PAGE_READWRITE);
            let gpu = reserve(0x801);
            let (src, va) = (cpu.0 + OFFSET, gpu.0 + OFFSET);
            assert_eq!(kernelx::mirrors::source(va), Some(src));
            assert_eq!(kernelx::VirtualAlloc(src as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize, src);
            let console = console_desc(3, 64, 64, 1, 1, DXGI_FORMAT_R8_UNORM.0 as u32, 1);
            assert!(xg::Texture::new(&console).unwrap().size() <= PAGE as u64);
            let mut texture = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
                D3D12_RESOURCE_STATE_COMMON, None, &mut texture,
            ).unwrap();
            let texture: ID3D12Resource = texture.unwrap();
            let key = texture.as_raw() as usize;
            register(raw_device, texture.as_raw(), va, &console, D3D12_RESOURCE_STATE_COMMON).unwrap();
            let _registered = RegisteredTexture(key);
            for value in [0x3D, 0x9A] {
                std::ptr::write_bytes(src as *mut u8, value, PAGE);
                refresh(queue.as_raw());
                assert_eq!(read_r8(dev, &queue, &texture), vec![value; 64 * 64], "GPU contents follow CPU writes");
                assert_eq!(address(key), Some(va), "the title address stays in the GPU view");
            }
            let (mut count, mut id) = (1usize, 0usize);
            assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut count, &mut id), 1);
            let mut mapped = PhysicalPage { id, at: va };
            assert_eq!(kernelx::MapTitlePhysicalPages(va as *mut c_void, 1, 0, 4, &mut mapped.id) as usize, va);
            assert!(kernelx::physical_page_at(va));
            let mut info: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(mem::VirtualQuery(va as *const c_void, &mut info, std::mem::size_of_val(&info)), 0);
            assert_eq!(info.State, mem::MEM_RESERVE, "exercise the lazy mapping before its first touch");
            assert_eq!(mirror_source(va), va, "a mapped lazy page overrides the CPU mirror");
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![0; 64 * 64], "the untouched physical page supplies zeros");
            std::ptr::write_bytes(va as *mut u8, 0xC7, PAGE);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![0xC7; 64 * 64]);
            assert_eq!(kernelx::MapTitlePhysicalPages(va as *mut c_void, 1, 0, 4, std::ptr::null_mut()) as usize, va);
            assert!(!kernelx::physical_page_at(va));
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![0x9A; 64 * 64], "unmapping restores the earlier CPU view");
            assert_eq!(address(key), Some(va));

            // the queue's page mappings: a texture registered over a page mapped into the GPU
            // view reads the pool page, where it is kept
            struct Pool(Vec<usize>);
            impl Drop for Pool {
                fn drop(&mut self) {
                    unsafe { kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), self.0.len(), self.0.as_mut_ptr()) };
                }
            }
            let (mut n, mut ids) = (2usize, vec![0usize; 2]);
            assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, ids.as_mut_ptr()), 1);
            let mut pool = Pool(ids);
            let home = kernelx::MapTitlePhysicalPages(std::ptr::null_mut(), 2, 0, 4, pool.0.as_mut_ptr()) as usize;
            assert_ne!(home, 0);
            std::ptr::write_bytes(home as *mut u8, 0x51, PAGE);
            std::ptr::write_bytes((home + PAGE) as *mut u8, 0x62, PAGE);
            let g = va + 4 * PAGE;
            map_pages(g, &[[0, 0, 1]], home, 2);
            let mut paged = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
                D3D12_RESOURCE_STATE_COMMON, None, &mut paged,
            ).unwrap();
            let paged: ID3D12Resource = paged.unwrap();
            register(raw_device, paged.as_raw(), g, &console, D3D12_RESOURCE_STATE_COMMON).unwrap();
            let _paged = RegisteredTexture(paged.as_raw() as usize);
            assert!(TEXTURES.lock().unwrap().by_res.get(&(paged.as_raw() as usize)).is_some_and(|e| e.paged && e.src == g));
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0x51; 64 * 64], "the pool page mapped there");
            // a mapping that changes under it: a pool page written and looked at before it was
            // mapped, with no write since; the same bytes in another page; an unmap
            map_pages(g, &[[0, 1, 1]], home, 2);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0x62; 64 * 64], "a clean pool page newly mapped");
            std::ptr::write_bytes(home as *mut u8, 0x62, PAGE);
            let reread = rereads();
            map_pages(g, &[[0, 0, 1]], home, 2);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0x62; 64 * 64], "equal bytes in the old and the new page");
            assert!(rereads() > reread, "read again for the mapping, equal bytes or not");
            map_pages(g, &[[2, 0, 1]], home, 2);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0; 64 * 64], "unmapped: nothing");
            map_pages(g, &[[0, 1, 1]], home, 2);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0x62; 64 * 64]);
            // writes where the page is kept: in the pool; just before the page moves into the CPU
            // view, with no look in between; where it is kept after that; by its page number
            // (`mark_hot`: the looks take those blocks now, not once the walk finds them)
            std::ptr::write_bytes((home + PAGE) as *mut u8, 0x73, PAGE);
            mark_hot(home, 2 * PAGE);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0x73; 64 * 64], "a write to the pool page");
            std::ptr::write_bytes((home + PAGE) as *mut u8, 0x84, PAGE);
            let kept = src + 6 * PAGE;
            let mut one = [pool.0[1]];
            assert_eq!(kernelx::MapTitlePhysicalPages(kept as *mut c_void, 1, 0, 4, one.as_mut_ptr()) as usize, kept);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0x84; 64 * 64], "a write made just before the page moved");
            std::ptr::write_bytes(kept as *mut u8, 0x95, PAGE);
            mark_hot(kept, PAGE);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0x95; 64 * 64], "a write where the page is kept now");
            assert!(kernelx::write_physical_pages(&[pool.0[1]], 0, &[0xA6; PAGE]));
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0xA6; 64 * 64], "a write by page number");
            // the GPU's write of a submitted list stays over a CPU write, and gives way to a later
            // mapping; a CPU write after that goes up
            let write = capture_gpu_write(paged.as_raw() as usize, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES).unwrap();
            publish_gpu_writes(WriterQueueId { raw: queue.as_raw() as usize, generation: 1 }, &[write]);
            std::ptr::write_bytes(kept as *mut u8, 0xB7, PAGE);
            mark_hot(kept, PAGE);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0xA6; 64 * 64], "the GPU's");
            map_pages(g, &[[0, 0, 1]], home, 2);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0x62; 64 * 64], "a later mapping takes it back");
            std::ptr::write_bytes(home as *mut u8, 0xC8, PAGE);
            mark_hot(home, PAGE);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &paged), vec![0xC8; 64 * 64], "and a CPU write after it goes up");
            assert_eq!(kernelx::MapTitlePhysicalPages(kept as *mut c_void, 1, 0, 4, std::ptr::null_mut()) as usize, kept);
            dev.GetDeviceRemovedReason().unwrap();
        }
    }

    /// A physical page mapped into a GPU view but not used yet (reserved until its first
    /// access, `kernelx::lazy`) is that address's own source, not the CPU view behind it.
    /// (Test by Kalmarv, review of PR #283.)
    #[test]
    fn a_lazily_mapped_page_in_a_gpu_view_is_its_own_source() {
        use crate::hle::kernelx;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const VIEW: usize = 16 << 20;
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let cpu = VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE, PAGE_READWRITE) as usize;
            let gpu = VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE, PAGE_READWRITE) as usize;
            kernelx::mirrors::reserved(cpu, VIEW, 0x2040_2000, 0x04);
            kernelx::mirrors::reserved(gpu, VIEW, 0x2040_2000, 0x801);
            assert_eq!(mirror_source(gpu + 0x2_0000), cpu + 0x2_0000); // nothing mapped: the CPU view
            let (mut n, mut pages) = (1usize, [0usize; 1]);
            assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let at = gpu + 0x2_0000;
            // a texture placed there before the page is mapped reads the CPU view …
            let mut d = [0u8; 0x38];
            d[0..4].copy_from_slice(&3u32.to_le_bytes());
            d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
            d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
            d[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            d[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
            d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
            let tex = Arc::new(xg::Texture::new(&d).unwrap());
            let mut e = Entry { device: 0, committed_bytes: 0, va: at, src: mirror_source(at), in_gpu_view: true, paged: false, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: false, written_call: 0, gpu_writers: Vec::new(), cpu_backed: true, placed: false, unmapped: false, generation: 0, known: Default::default(), target: false };
            assert_eq!(e.src, cpu + 0x2_0000);
            assert!(!recheck_source(&mut e), "nothing changed yet");
            assert_eq!(kernelx::MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, at);
            // … and the page mapped under it later, uploaded again
            assert!(recheck_source(&mut e));
            assert_eq!(e.src, at);
            assert!(e.pending);
            PENDING.fetch_sub(1, Ordering::Relaxed);
            let before = mirror_source(at);
            (at as *mut u32).write_volatile(7); // its first use commits it
            let after = mirror_source(at);
            let mut none = [0usize; 1];
            kernelx::MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, none.as_mut_ptr());
            kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr());
            kernelx::mirrors::released(cpu);
            VirtualFree(cpu as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(gpu as *mut c_void, 0, MEM_RELEASE);
            assert_eq!(after, at);
            assert_eq!(before, at, "a page mapped at the address but not used yet is its source too");
        }
    }

    thread_local! {
        /// the VirtualQuery calls `mirror_source` made on this thread
        pub(super) static QUERIES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// One call of the queue's page-mapping method: `ranges` (kind, start, count) from GPU-view
    /// address `dest` on, in the pool of `pages` pages at `pool`.
    unsafe fn map_pages(dest: usize, ranges: &[[u32; 3]], pool: usize, pages: u32) {
        // (a batch: the first destination, the number of ranges, a pointer to them)
        let batch = [dest as u64, ranges.len() as u64, ranges.as_ptr() as u64];
        assert!(crate::hle::kernelx::mirrors::apply_batches(1, batch.as_ptr() as usize, pool, pages).is_some());
    }

    /// A pair of views of `size` bytes (a size of the test's own: no other test's views pair with
    /// these) and `n` physical pages of the title at their home, the pool: (CPU view, GPU view,
    /// the pages, their home).
    unsafe fn paged_views(size: usize, n: usize) -> (usize, usize, Vec<usize>, usize) {
        use crate::hle::kernelx::{self, mirrors};
        use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESERVE, PAGE_READWRITE};
        let cpu = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_READWRITE) as usize;
        let gpu = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_READWRITE) as usize;
        assert!(cpu != 0 && gpu != 0);
        mirrors::reserved(cpu, size, 0x2040_2000, 0x04);
        assert!(mirrors::reserved(gpu, size, 0x2040_2000, 0x801));
        let (mut k, mut pages) = (n, vec![0usize; n]);
        assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut k, pages.as_mut_ptr()), 1);
        let home = kernelx::MapTitlePhysicalPages(std::ptr::null_mut(), n, 0, 4, pages.as_mut_ptr()) as usize;
        assert_ne!(home, 0);
        (cpu, gpu, pages, home)
    }

    /// The pages and the views of `paged_views` go.
    unsafe fn drop_paged_views(cpu: usize, gpu: usize, mut pages: Vec<usize>) {
        use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
        crate::hle::kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), pages.len(), pages.as_mut_ptr());
        crate::hle::kernelx::mirrors::released(cpu);
        VirtualFree(cpu as *mut c_void, 0, MEM_RELEASE);
        VirtualFree(gpu as *mut c_void, 0, MEM_RELEASE);
    }

    /// A texture over GPU-view pages the queue's page mappings named is read at its own address
    /// through a table of where each page is kept, found with no VirtualQuery; one over pages
    /// never named takes the CPU view, as before. The table counts its pages by kind.
    #[test]
    fn a_texture_over_mapped_pages_is_read_where_they_are_kept() {
        use crate::hle::kernelx::mirrors::PAGE;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, PAGE_READWRITE};
        let _guard = TEST_LOCK.lock().unwrap();
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let (cpu, gpu, pages, home) = paged_views(179 * PAGE, 2);
            std::ptr::write_bytes(home as *mut u8, 0x5A, PAGE);
            assert_eq!(VirtualAlloc(cpu as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, cpu);
            std::ptr::write_bytes(cpu as *mut u8, 0x77, PAGE);
            // pool page 0 at view page 1, view page 2 unmapped
            map_pages(gpu + PAGE, &[[0, 0, 1], [2, 0, 1]], home, 2);
            let queries = || QUERIES.with(|n| n.get());
            let q = queries();
            assert_eq!(source_of(gpu + PAGE + 0x100, 0x4000), (gpu + PAGE + 0x100, true), "over a mapped page: its own address");
            assert_eq!(source_of(gpu + 2 * PAGE, 0x4000), (gpu + 2 * PAGE, true), "over an unmapped one too");
            assert_eq!(queries(), q, "no VirtualQuery");
            assert_eq!(source_of(gpu + 5 * PAGE, 0x4000), (cpu + 5 * PAGE, false), "never named: the CPU view, as before");
            assert_eq!(queries(), q + 1);
            // a texture over view pages 0 to 3: never named, kept at the pool, nothing, never named
            let before = page_counts();
            let m = page_map(gpu + PAGE - 0x2000, 2 * PAGE + 0x4000, true).expect("a table");
            let after = page_counts();
            assert_eq!(std::array::from_fn::<u64, 4, _>(|k| after[k] - before[k]), [1, 0, 2, 1]);
            assert!(page_map(gpu + PAGE, 0x4000, false).is_some() && page_counts() == after, "a table of its own is not counted");
            assert!(page_map(gpu + 10 * PAGE, PAGE, true).is_none(), "no page named");
            let mut b = vec![0xEEu8; 3 * PAGE];
            m.copy(gpu, &mut b);
            assert!(b[..PAGE].iter().all(|&x| x == 0x77), "the CPU view under a page never named");
            assert!(b[PAGE..2 * PAGE].iter().all(|&x| x == 0x5A), "the pool page where it is kept");
            assert!(b[2 * PAGE..].iter().all(|&x| x == 0), "nothing");
            drop_paged_views(cpu, gpu, pages);
        }
    }

    /// The GPU's write of a texture read through its pages counts from the submit of the list
    /// that records it: a mapping changed by a later call takes the texture back (its pages go up
    /// again as they are), one changed before that submit leaves it the GPU's – a list recorded
    /// before the call and submitted after it, a list submitted again – and a list reset writes
    /// nothing more.
    #[test]
    fn a_gpu_write_counts_from_the_submit_of_its_list() {
        use crate::hle::kernelx::mirrors::{self, PAGE};
        use texture_upload::{Again, Change};
        let _guard = TEST_LOCK.lock().unwrap();
        // (no submit looks while a texture of no resource is in the table)
        let _refresh = REFRESH.lock().unwrap();
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let (cpu, gpu, pages, home) = paged_views(181 * PAGE, 2);
            map_pages(gpu, &[[0, 0, 1]], home, 2);
            recheck_gpu_view_sources();
            let tex = Arc::new(xg::Texture::new(&console_desc(3, 128, 128, 1, 1, 28, 0x105)).unwrap());
            assert_eq!(tex.size() as usize, PAGE);
            // a key of no resource, never dereferenced
            let res = 0x7E57_0000_5000usize;
            let known: texture_upload::Known = Default::default();
            texture_upload::all_sent(&known, gpu, PAGE);
            let e = Entry { device: 0, va: gpu, src: gpu, in_gpu_view: true, paged: true, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: false, written_call: 0, placed: false, unmapped: false, generation: 0, known: Arc::clone(&known), target: false, committed_bytes: 0, gpu_writers: Vec::new(), cpu_backed: true };
            TEXTURES.lock().unwrap().insert(res, e);
            TEXTURES_LEN.fetch_add(1, Ordering::Relaxed);
            let state = || TEXTURES.lock().unwrap().by_res.get(&res).map(|e| (e.gpu_written, e.written_call)).unwrap();
            let again = || (AGAIN.lock().unwrap().remove(&res), texture_upload::take_again(&known, 0, 0));
            let page_again = || (true, Again::Now(Change::Parts(vec![(0, PAGE)])));
            let queue = WriterQueueId { raw: 0x7E57_0000_6000, generation: 1 };
            let all = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            // recorded before a call, submitted after it
            let w1 = capture_gpu_write(res, all).unwrap();
            assert_eq!(state(), (false, 0), "nothing is marked as it is recorded");
            map_pages(gpu, &[[0, 1, 1]], home, 2);
            let c1 = mirrors::calls();
            publish_gpu_writes(queue, &[w1]);
            assert!(state().0 && state().1 >= c1, "marked and stamped at its submit");
            recheck_gpu_view_sources();
            assert!(state().0, "the GPU's write came after the mapping");
            assert_eq!(again(), (false, Again::Nothing));
            // a mapping changed by a later call takes it back
            map_pages(gpu, &[[2, 0, 1]], home, 2);
            recheck_gpu_view_sources();
            assert!(!state().0, "an older GPU write gives way");
            assert_eq!(again(), page_again(), "the page goes up again as it is, with no comparison");
            // a segment submitted before a call and again after it
            let w2 = capture_gpu_write(res, all).unwrap();
            publish_gpu_writes(queue, &[w2]);
            map_pages(gpu, &[[0, 0, 1]], home, 2);
            publish_gpu_writes(queue, &[w1, w2]);
            recheck_gpu_view_sources();
            assert!(state().0, "the second submit came after the mapping");
            assert_eq!(again(), (false, Again::Nothing));
            // a submit of segments that write nothing marks nothing
            map_pages(gpu, &[[0, 1, 1]], home, 2);
            publish_gpu_writes(queue, &[]);
            recheck_gpu_view_sources();
            assert!(!state().0);
            assert_eq!(again(), page_again());
            forget(res);
            drop_paged_views(cpu, gpu, pages);
        }
    }

    /// A texture read through its pages finds its written pages through the GPU-view pages a
    /// write reaches: a write where a mapped page is kept, and one in the CPU view under a page
    /// the queue's page mappings never named; those that differ from what went up are sent.
    #[test]
    fn an_untouched_page_of_a_paged_texture_is_found_written() {
        use crate::hle::kernelx::mirrors::{views_of_written, PAGE};
        use texture_upload::{Change, Written};
        use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, PAGE_READWRITE};
        let _guard = TEST_LOCK.lock().unwrap();
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let (cpu, gpu, pages, home) = paged_views(191 * PAGE, 1);
            assert_eq!(VirtualAlloc((cpu + PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, cpu + PAGE);
            std::ptr::write_bytes(home as *mut u8, 0x21, PAGE);
            std::ptr::write_bytes((cpu + PAGE) as *mut u8, 0x43, PAGE);
            // a texture over view pages 0 (the pool page mapped there) and 1 (never named)
            map_pages(gpu, &[[0, 0, 1]], home, 1);
            let m = page_map(gpu, 2 * PAGE, false).expect("a table");
            let known: texture_upload::Known = Default::default();
            texture_upload::all_sent(&known, gpu, 2 * PAGE);
            let compare = |dirty: &[usize]| texture_upload::changed(&[Written { src: gpu, size: 2 * PAGE, dirty, known: &known, pages: Some(&m) }], true).0.remove(0);
            let all: Vec<usize> = (0..2 * PAGE / 0x1000).map(|k| gpu + k * 0x1000).collect();
            assert_eq!(compare(&all), Change::Parts(vec![(0, 2 * PAGE)]), "nothing was known of it");
            // a write where the mapped page is kept, and one in the CPU view under the other page
            let (a, b) = (home + 0x2000, cpu + PAGE + 0x3000);
            let mut written = vec![a, b];
            written.sort_unstable();
            let look = |written: &[usize]| {
                let mut twins = Vec::new();
                views_of_written(written, &mut twins);
                twins.sort_unstable();
                twins.dedup();
                assert!(twins.contains(&(gpu + 0x2000)) && twins.contains(&(gpu + PAGE + 0x3000)), "{twins:x?}");
                compare(&twins)
            };
            assert_eq!(look(&written), Change::None, "they hold what went up");
            *(a as *mut u8) ^= 1;
            *(b as *mut u8) ^= 1;
            assert_eq!(look(&written), Change::Parts(vec![(0x2000, 0x3000), (PAGE + 0x3000, PAGE + 0x4000)]));
            assert_eq!(look(&written), Change::None);
            drop_paged_views(cpu, gpu, pages);
        }
    }

    /// `written_pages` reports what was written and not taken, at any number, and leaves it reported.
    #[test]
    fn written_pages_are_reported_without_being_taken() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const N: usize = 1500;
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), N << 12, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            assert!(written_pages(p, N << 12).is_empty());
            for k in 0..N {
                *((p + (k << 12) + 5) as *mut u8) = 1;
            }
            let all: Vec<usize> = (0..N).map(|k| p + (k << 12)).collect();
            assert_eq!(written_pages(p, N << 12), all);
            assert_eq!(written_pages(p + 0x1800, 0x1000), vec![p + 0x1000, p + 0x2000], "the pages a range lies on");
            // (they are still there for a look)
            let (mut buf, mut dirty, mut page) = (vec![std::ptr::null_mut(); 4096], Vec::new(), 0x1000);
            take_written(p, N << 12, &mut buf, &mut dirty, &mut page, None);
            assert_eq!(dirty, all);
            assert!(written_pages(p, N << 12).is_empty());
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
            let plain = VirtualAlloc(std::ptr::null(), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            *(plain as *mut u8) = 1;
            assert!(written_pages(plain, 0x1000).is_empty(), "memory that is not watched");
            VirtualFree(plain as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A submit whose lists were closed before the last look began goes without a look of its own
    /// (`XCL_LOOK_SKIP`) – not once the walk has found a written block since: no look took that one.
    #[test]
    fn a_block_the_walk_found_since_ends_the_skip() {
        let list = 0x7E57_1157usize as *mut c_void;
        note_list_closed(list as usize);
        std::thread::sleep(std::time::Duration::from_millis(2));
        // (other tests' submits only raise the time of the last look)
        let frame = super::super::present::activity().0;
        note_look_began(now_us(), frame);
        let (began, _) = last_look();
        if WALK_FOUND.load(Ordering::Acquire) < began {
            assert!(closed_before_last_look(&[list]));
        }
        WALK_FOUND.store(u64::MAX, Ordering::Release);
        assert!(!closed_before_last_look(&[list]), "the look before did not take the block the walk found");
        // (the next submit of the process looks for itself: nothing is lost by this)
        WALK_FOUND.store(now_us(), Ordering::Release);
        forget_list_closed(list as usize);
    }

    #[test]
    fn closed_timestamp_cleared_on_list_reset() {
        let list = 0x7E57_2001usize as *mut c_void;
        note_list_closed(list as usize);
        let began = now_us() + 1000;
        assert!(closed_before_last_look_in(&[list], began, 5, 5, 0));
        // (a Reset of the list: list_x forgets its close)
        forget_list_closed(list as usize);
        assert!(!is_list_closed_for_test(list as usize), "a reset must clear CLOSED");
        assert!(!closed_before_last_look_in(&[list], began, 5, 5, 0), "reset list must not keep a stale CLOSED timestamp");
    }

    #[test]
    fn closed_timestamp_cleared_on_release_and_pointer_reuse() {
        let list = 0x7E57_2002usize as *mut c_void;
        note_list_closed(list as usize);
        let began = now_us() + 1000;
        assert!(closed_before_last_look_in(&[list], began, 7, 7, 0));
        // Release removes CLOSED before deallocation so a newly allocated list at the same
        // address does not inherit the old object's pre-look close timestamp.
        forget_list_closed(list as usize);
        assert!(!is_list_closed_for_test(list as usize));
        assert!(!closed_before_last_look_in(&[list], began, 7, 7, 0), "reused list pointer must not skip before closing");
    }

    #[test]
    fn new_frame_forces_look_even_for_pre_closed_list() {
        let list = 0x7E57_2003usize as *mut c_void;
        note_list_closed(list as usize);
        let began = now_us() + 1000;
        assert!(closed_before_last_look_in(&[list], began, 10, 10, 0), "same frame skips pre-closed list");
        assert!(!closed_before_last_look_in(&[list], began, 10, 11, 0), "new frame must force at least one fresh look");
        assert!(should_look(closed_before_last_look_in(&[list], began, 10, 11, 0), true, false));
        // `advance_look` advances monotonically across frames and never rewinds on a late round.
        let mut st = (0u64, u64::MAX);
        advance_look(&mut st, 100, 0);
        assert_eq!(st, (100, 0));
        advance_look(&mut st, 200, 0);
        assert_eq!(st, (200, 0));
        advance_look(&mut st, 150, 0);
        assert_eq!(st, (200, 0));
        advance_look(&mut st, 250, 1);
        assert_eq!(st, (250, 1));
        advance_look(&mut st, 300, 0);
        assert_eq!(st, (250, 1));
        forget_list_closed(list as usize);
    }

    #[test]
    fn same_frame_pre_closed_lists_skip_look_when_enabled() {
        let l1 = 0x7E57_2004usize as *mut c_void;
        let l2 = 0x7E57_2005usize as *mut c_void;
        note_list_closed(l1 as usize);
        note_list_closed(l2 as usize);
        let began = now_us() + 1000;
        let old = closed_before_last_look_in(&[l1, l2], began, 12, 12, began - 1);
        assert!(old);
        assert!(!should_look(old, true, false), "XCL_LOOK_SKIP=1 skips look for same-frame pre-closed lists");
        assert!(should_look(old, false, false), "default off still runs look");
        forget_list_closed(l1 as usize);
        forget_list_closed(l2 as usize);
    }

    #[test]
    fn list_closed_after_look_began_runs_look() {
        let l1 = 0x7E57_2006usize as *mut c_void;
        let l2 = 0x7E57_2007usize as *mut c_void;
        note_list_closed(l1 as usize);
        std::thread::sleep(std::time::Duration::from_millis(2));
        let began = now_us();
        std::thread::sleep(std::time::Duration::from_millis(2));
        note_list_closed(l2 as usize);
        let old = closed_before_last_look_in(&[l1, l2], began, 15, 15, 0);
        assert!(!old, "list closed after LOOK_BEGAN must force a look even in the same frame");
        assert!(should_look(old, true, false));
        forget_list_closed(l1 as usize);
        forget_list_closed(l2 as usize);
    }

    #[test]
    fn walk_found_at_or_after_look_began_runs_look() {
        let list = 0x7E57_2008usize as *mut c_void;
        note_list_closed(list as usize);
        let began = now_us() + 1000;
        assert!(closed_before_last_look_in(&[list], began, 20, 20, began - 1));
        assert!(!closed_before_last_look_in(&[list], began, 20, 20, began), "WALK_FOUND == LOOK_BEGAN must run look");
        assert!(!closed_before_last_look_in(&[list], began, 20, 20, began + 1), "WALK_FOUND > LOOK_BEGAN must run look");
        assert!(should_look(closed_before_last_look_in(&[list], began, 20, 20, began), true, false));
        forget_list_closed(list as usize);
    }

    /// Pages found ahead of a submit are kept until one takes them, are not reported again, and a
    /// page written after that is found again.
    #[test]
    fn pages_found_ahead_are_kept_for_the_next_submit() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const N: usize = 256;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N << 12, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            let mut buf = vec![std::ptr::null_mut::<c_void>(); 64];
            let ahead = Mutex::new(Vec::new());
            let write = |k: usize| ((base + (k << 12)) as *mut u8).write_volatile(1);
            for k in (0..N).step_by(2) {
                write(k);
            }
            assert_eq!(take_piece(base, N << 12, &mut buf, &ahead), N / 2, "past a full buffer too");
            assert_eq!(*ahead.lock().unwrap(), (0..N).step_by(2).map(|k| base + (k << 12)).collect::<Vec<_>>());
            let (mut dirty, mut page) = (Vec::new(), 0u32);
            take_written(base, N << 12, &mut buf, &mut dirty, &mut page, None);
            assert!(dirty.is_empty(), "a look does not find them again");
            write(8);
            write(9);
            assert_eq!(take_piece(base, N << 12, &mut buf, &ahead), 2);
            assert_eq!(ahead.lock().unwrap()[N / 2..], [base + 0x8000, base + 0x9000], "written again: found again, behind the others");
            // (only of its own piece)
            write(0);
            write(200);
            assert_eq!(take_piece(base + (100 << 12), 100 << 12, &mut buf, &ahead), 0);
            assert_eq!(take_piece(base + (100 << 12), 101 << 12, &mut buf, &ahead), 1);
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// More written pages than one call reports: each is found once, in order, and the look
    /// resets them – a buffer that fills in the middle of the range and one that fills exactly.
    #[test]
    fn written_pages_past_a_full_buffer_are_all_found_once() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 1000;
        const PAGE: usize = 0x1000;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert!(base != 0);
            let mut buf = vec![std::ptr::null_mut::<c_void>(); 64];
            let (mut dirty, mut page) = (Vec::new(), 0u32);
            take_written(base, PAGES * PAGE, &mut buf, &mut dirty, &mut page, None);
            for written in [(0..PAGES).filter(|i| i % 7 != 3).collect::<Vec<_>>(), (0..128).map(|i| i * 5).collect()] {
                let want: Vec<usize> = written.iter().map(|i| base + i * PAGE).collect();
                for &a in &want {
                    (a as *mut u8).write_volatile(1);
                }
                dirty.clear();
                take_written(base, PAGES * PAGE, &mut buf, &mut dirty, &mut page, None);
                assert_eq!(page as usize, PAGE);
                assert_eq!(dirty, want, "{} pages written", want.len());
                dirty.clear();
                take_written(base, PAGES * PAGE, &mut buf, &mut dirty, &mut page, None);
                assert!(dirty.is_empty(), "the look reset what it reported");
            }
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    #[ignore = "manual Windows GetWriteWatch timing"]
    fn benchmark_write_watch_calls_against_gap_bytes() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGE: usize = 0x1000;
        const SIZE: usize = 49 << 20;
        fn distribution(mut samples: Vec<u64>) -> (u64, u64, u64) {
            samples.sort_unstable();
            (samples[0], samples[samples.len() / 2], samples[samples.len() - 1])
        }
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            let mut buf = vec![std::ptr::null_mut::<c_void>(); 64];
            let (mut dirty, mut page) = (Vec::with_capacity(64), 0u32);
            assert!(take_written(base, SIZE, &mut buf, &mut dirty, &mut page, None));
            for gap_mb in [4usize, 6, 8, 10, 16] {
                let gap = gap_mb << 20;
                let spans: Vec<usize> = (0..4).map(|i| base + i * (PAGE + gap)).collect();
                let mut samples: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::new());
                for trial in 0..9 {
                    let order = if trial % 2 == 0 { [false, true] } else { [true, false] };
                    for merged in order {
                        for written in [false, true] {
                            dirty.clear();
                            assert!(take_written(base, SIZE, &mut buf, &mut dirty, &mut page, None));
                            dirty.clear();
                            if written {
                                for &at in &spans { (at as *mut u8).write_volatile(1) }
                            }
                            let mut measurement = WatchMeasurement::default();
                            if merged {
                                assert!(take_written(spans[0], spans[3] + PAGE - spans[0], &mut buf, &mut dirty, &mut page, Some(&mut measurement)));
                                assert_eq!(measurement.calls, 1);
                            } else {
                                for &at in &spans {
                                    assert!(take_written(at, PAGE, &mut buf, &mut dirty, &mut page, Some(&mut measurement)));
                                }
                                assert_eq!(measurement.calls, spans.len() as u64);
                            }
                            dirty.sort_unstable();
                            assert_eq!(dirty, if written { spans.clone() } else { Vec::new() });
                            samples[(merged as usize) * 2 + written as usize].push(measurement.elapsed_ns);
                        }
                    }
                }
                println!("gap {gap_mb} MiB ns min/median/max: separate clean {:?}, dirty {:?}; merged clean {:?}, dirty {:?}", distribution(samples[0].clone()), distribution(samples[1].clone()), distribution(samples[2].clone()), distribution(samples[3].clone()));
            }
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    const SRV: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0);
    const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn gpu_owned_metadata_accepts_exact_colour_layouts_only() {
        let desc = console_desc(3, 64, 64, 2, 3, 28, 0x105);
        let with_flags = |mut d: [u8; 0x38], flags: u32| {
            d[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
            d
        };
        let pc = |d: &[u8; 0x38]| unsafe {
            std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(d).as_ptr() as *const D3D12_RESOURCE_DESC)
        };

        let uav = with_flags(desc, 4);
        let tex = gpu_owned_layout(&uav, &pc(&uav)).expect("single-plane UAV layout");
        assert_eq!(subresource_count(&tex), 6);

        let rt = with_flags(console_desc(3, 128, 128, 1, 1, 28, 0x105), 1);
        let tex = gpu_owned_layout(&rt, &pc(&rt)).expect("render target with non-texel CMASK metadata");
        assert_eq!(tex.planes.len(), 2);
        assert_eq!(subresource_count(&tex), 1);

        let depth = with_flags(desc, 2);
        assert!(gpu_owned_layout(&depth, &pc(&depth)).is_err());
        let msaa = {
            let mut d = with_flags(desc, 1);
            d[0x24..0x28].copy_from_slice(&4u32.to_le_bytes());
            d
        };
        assert!(gpu_owned_layout(&msaa, &pc(&msaa)).is_err());
        let promoted = with_flags(console_desc(3, 64, 64, 1, 1, 67, 0x105), 1);
        assert_ne!(u32::from_le_bytes(promoted[0x20..0x24].try_into().unwrap()), pc(&promoted).Format.0 as u32);
        assert!(gpu_owned_layout(&promoted, &pc(&promoted)).is_err());
        let planar = with_flags(console_desc(3, 64, 64, 1, 1, 45, 0x105), 1);
        assert!(gpu_owned_layout(&planar, &pc(&planar)).is_err());
        let mut wrong_shape = pc(&uav);
        wrong_shape.MipLevels -= 1;
        assert!(gpu_owned_layout(&uav, &wrong_shape).is_err());
    }

    /// One state while every subresource has it, one each while they differ.
    #[test]
    fn transitions_per_subresource_and_back_to_one_state() {
        let mut s = vec![D3D12_RESOURCE_STATE_COMMON];
        apply_transition(&mut s, 4, ALL, SRV);
        assert_eq!(s, vec![SRV]);
        apply_transition(&mut s, 4, 2, D3D12_RESOURCE_STATE_COPY_SOURCE);
        assert_eq!(s, vec![SRV, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE, SRV]);
        apply_transition(&mut s, 4, 9, D3D12_RESOURCE_STATE_COPY_DEST); // no subresource 9
        assert_eq!(s, vec![SRV, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE, SRV]);
        apply_transition(&mut s, 4, 2, SRV);
        assert_eq!(s, vec![SRV]);
        apply_transition(&mut s, 4, 1, D3D12_RESOURCE_STATE_COPY_DEST);
        apply_transition(&mut s, 4, ALL, D3D12_RESOURCE_STATE_COMMON);
        assert_eq!(s, vec![D3D12_RESOURCE_STATE_COMMON]);
        // a texture of one subresource: subresource 0 is all of it
        let mut one = vec![SRV];
        apply_transition(&mut one, 1, 0, D3D12_RESOURCE_STATE_COPY_SOURCE);
        assert_eq!(one, vec![D3D12_RESOURCE_STATE_COPY_SOURCE]);
    }

    #[test]
    fn submitted_gpu_writes_keep_generation_queue_epoch_and_subresource() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tex = Arc::new(xg::Texture::new(&console_desc(3, 64, 64, 2, 3, 28, 0x105)).unwrap());
        let res = 0x7E57_1000_1000usize;
        let entry = |tex: &Arc<xg::Texture>| Entry {
            device: 0,
            committed_bytes: 0,
            va: 0x7E57_1000_0000,
            src: 0x7E57_1000_0000,
            in_gpu_view: false,
            tex: Arc::clone(tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending: false,
            gpu_written: false,
            gpu_writers: Vec::new(),
            cpu_backed: true,
            placed: false,
            unmapped: false,
            generation: 0,
            paged: false,
            written_call: 0,
            known: Default::default(),
            target: false,
        };
        let generation = {
            let mut t = TEXTURES.lock().unwrap();
            t.insert(res, entry(&tex));
            t.by_res[&res].generation
        };
        let first = GpuWrite { res, generation, subresource: 2 };
        let q1 = WriterQueueId { raw: 0x1000, generation: 7 };
        publish_gpu_writes(q1, &[first]);
        let first_epoch = {
            let t = TEXTURES.lock().unwrap();
            let e = &t.by_res[&res];
            assert!(e.gpu_written);
            assert_eq!(e.gpu_writers.len(), 6);
            assert!(e.gpu_writers.iter().enumerate().all(|(i, w)| (i == 2) == w.is_some()));
            let w = e.gpu_writers[2].unwrap();
            assert_eq!(w.queue, q1);
            w.epoch
        };

        let q2 = WriterQueueId { raw: 0x2000, generation: 9 };
        publish_gpu_writes(q2, &[GpuWrite { res, generation, subresource: 4 }]);
        {
            let t = TEXTURES.lock().unwrap();
            let e = &t.by_res[&res];
            assert_eq!(e.gpu_writers[2].unwrap().queue, q1);
            let last = e.gpu_writers[4].unwrap();
            assert_eq!(last.queue, q2);
            assert!(last.epoch > first_epoch);
            let q1_writes = download_writes(res, e, q1);
            let q2_writes = download_writes(res, e, q2);
            assert_eq!(q1_writes.iter().map(|w| w.subresource).collect::<Vec<_>>(), [2]);
            assert_eq!(q2_writes.iter().map(|w| w.subresource).collect::<Vec<_>>(), [4]);
            let key = (res, generation, q1.generation, 2, first_epoch);
            SCHEDULED_DOWNLOADS.lock().unwrap().insert(key);
            assert!(download_writes(res, e, q1).is_empty(), "an in-flight epoch is not downloaded twice");
            SCHEDULED_DOWNLOADS.lock().unwrap().remove(&key);
        }

        let stale = download_writes(res, &TEXTURES.lock().unwrap().by_res[&res], q1);
        let key = (res, generation, q1.generation, 2, first_epoch);
        SCHEDULED_DOWNLOADS.lock().unwrap().insert(key);
        publish_gpu_writes(q2, &[GpuWrite { res, generation, subresource: 2 }]);
        assert!(!download_current(&TEXTURES.lock().unwrap().by_res[&res], generation, 0x7E57_1000_0000, &stale));
        unschedule_writes(res, generation, &stale);
        assert!(!SCHEDULED_DOWNLOADS.lock().unwrap().contains(&key));

        // Reusing the resource address creates another registration generation: a captured
        // effect from the old object must not mark it.
        {
            let mut t = TEXTURES.lock().unwrap();
            t.remove(res);
            t.insert(res, entry(&tex));
            assert_ne!(t.by_res[&res].generation, generation);
        }
        publish_gpu_writes(q1, &[first]);
        assert!(!TEXTURES.lock().unwrap().by_res[&res].gpu_written);
        TEXTURES.lock().unwrap().remove(res);
    }

    #[test]
    fn fence_download_batches_preserve_insertion_order() {
        let mut blocked = false;
        assert!(!batch_ready(&mut blocked, 20, 10));
        assert!(!batch_ready(&mut blocked, 5, 10), "later eligible batch passed an earlier one");
        blocked = true;
        assert!(!batch_ready(&mut blocked, 10, 10), "later batch passed an incomplete one");
    }

    #[test]
    fn detached_fence_batches_return_before_concurrent_arrivals() {
        let mut batches = vec![(1, "old-a"), (2, "other-a"), (1, "old-b")];
        let mut detached = take_matching(&mut batches, |batch| batch.0 == 1);
        assert_eq!(detached, [(1, "old-a"), (1, "old-b")]);
        assert_eq!(batches, [(2, "other-a")]);

        // The first old batch completed while detached; two newer batches were tied meanwhile.
        detached.remove(0);
        batches.extend([(1, "new-a"), (2, "other-b"), (1, "new-b")]);
        restore_before_matching(&mut batches, detached, |batch| batch.0 == 1);
        assert_eq!(
            batches.iter().filter(|batch| batch.0 == 1).map(|batch| batch.1).collect::<Vec<_>>(),
            ["old-b", "new-a", "new-b"],
            "an unresolved older download must keep later fence work hidden",
        );
        assert_eq!(
            batches.iter().filter(|batch| batch.0 == 2).map(|batch| batch.1).collect::<Vec<_>>(),
            ["other-a", "other-b"],
            "processing one fence must preserve another fence's order",
        );
    }

    #[test]
    fn concurrent_arrival_through_completed_value_withholds_visibility() {
        let arrivals = [(2, 4), (1, 11), (2, 7)];
        let for_fence_one = |batch: &(usize, u64)| (batch.0 == 1).then_some(batch.1);
        assert!(!matching_through(&arrivals, 10, for_fence_one), "a future arrival does not hide value 10");

        let arrivals = [(2, 4), (1, 11), (1, 10)];
        assert!(
            matching_through(&arrivals, 10, for_fence_one),
            "a concurrent arrival covered by value 10 must be published before value 10 is visible",
        );
    }

    #[test]
    fn in_progress_tie_withholds_fence_processing() {
        let mut state = FenceDownloadState::default();
        let mut safe = BTreeMap::from([(7, 99)]);
        if state.begin_tie(7) { ensure_safe_baseline(&mut safe, 7); }
        assert_eq!(safe[&7], 99, "first pending work preserves the last exposed completion");
        if state.begin_tie(7) { note_safe_value(&mut safe, 7, 123); }
        assert_eq!(safe[&7], 99, "another installer must not advance the safe baseline");
        assert!(state.any_pending());
        assert!(!state.begin_processing(7), "completion must not pass a partially installed tie");
        state.finish_tie(7);
        assert!(!state.begin_processing(7), "every installer must finish before completion starts");
        state.finish_tie(7);
        assert!(state.begin_processing(7));

        assert!(FenceDownloadState::default().begin_tie(8));
        ensure_safe_baseline(&mut safe, 8);
        assert_eq!(safe[&8], 0, "a fence never exposed before starts from the conservative baseline");
    }

    #[test]
    fn download_reservation_rollback_clears_every_key() {
        let keys = [(0xBAD0, 1, 2, 3, 4), (0xBAD1, 5, 6, 7, 8)];
        SCHEDULED_DOWNLOADS.lock().unwrap().extend(keys);
        rollback_reservations(&keys);
        assert!(keys.iter().all(|key| !SCHEDULED_DOWNLOADS.lock().unwrap().contains(key)));
    }

    /// A texture kept in step: the title's transitions count once their list is submitted, in
    /// submission order, each time it is; a Reset forgets them; other resources are not noted.
    #[test]
    fn transitions_apply_when_their_list_is_submitted() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // RGBA8 64×64, 3 mips, 2 slices (6 subresources), 4KB_S
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&3u32.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&2u16.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&3u16.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
        let tex = Arc::new(xg::Texture::new(&d).unwrap());
        assert_eq!(subresource_count(&tex), 6);
        // fake resource / list keys, never dereferenced: GPU-written, so no refresh uploads it
        let (res, other) = (0x7E57_0000_1000usize, 0x7E57_0000_2000usize);
        TEXTURES.lock().unwrap().insert(res, Entry { device: 0, committed_bytes: 0, va: 0x7E57_0000_0000, src: 0x7E57_0000_0000, in_gpu_view: false, paged: false, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: true, written_call: 0, gpu_writers: Vec::new(), cpu_backed: true, placed: false, unmapped: false, generation: 0, known: Default::default(), target: false });
        TEXTURES_LEN.fetch_add(1, Ordering::Relaxed);
        let states = || TEXTURES.lock().unwrap().by_res.get(&res).unwrap().states.clone();
        let e1 = capture_transitions(&[(res, ALL, SRV), (other, ALL, D3D12_RESOURCE_STATE_RENDER_TARGET), (res, 4, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        assert_eq!(e1.len(), 2, "only the texture kept in step is captured");
        assert_eq!(states(), vec![D3D12_RESOURCE_STATE_COMMON], "recorded, not run yet");
        let e2 = capture_transitions(&[(res, 4, SRV), (res, 0, D3D12_RESOURCE_STATE_COPY_DEST)]);
        publish_submission_effects(None, &[], &e1);
        let mut want = vec![SRV; 6];
        want[4] = D3D12_RESOURCE_STATE_COPY_SOURCE;
        assert_eq!(states(), want);
        publish_submission_effects(None, &[], &e2);
        want[4] = SRV;
        want[0] = D3D12_RESOURCE_STATE_COPY_DEST;
        assert_eq!(states(), want);
        // l1 runs again (a closed list may): its transitions again, after l2's
        publish_submission_effects(None, &[], &e2);
        publish_submission_effects(None, &[], &e1);
        assert_eq!(states(), {
            let mut w = vec![SRV; 6];
            w[4] = D3D12_RESOURCE_STATE_COPY_SOURCE;
            w
        });
        forget(res);
    }

    /// The title's bytes outside the coherent window are read as `read_committed` reads them (the
    /// pages that are not committed as zeros, never what an earlier read on the thread left);
    /// inside it they are used where they are.
    #[test]
    fn title_bytes_are_copied_outside_the_coherent_window() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 << 12, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert!(!VirtualAlloc(base as *const c_void, 3 << 12, MEM_COMMIT, PAGE_READWRITE).is_null());
            for page in 0..3usize {
                std::ptr::write_bytes((base + (page << 12)) as *mut u8, 0xA0 + page as u8, 1 << 12);
            }
            with_title_bytes(base, 3 << 12, |b| {
                assert_ne!(b.as_ptr() as usize, base, "memory outside the coherent window is not read in place");
                assert_eq!(b, &read_committed(base, 3 << 12)[..]);
                assert!((0..3usize).all(|p| b[p << 12..(p + 1) << 12].iter().all(|&x| x == 0xA0 + p as u8)));
            });
            VirtualFree((base + (1 << 12)) as *mut c_void, 1 << 12, MEM_DECOMMIT);
            // the hole reads as zeros, not as the page the last (as large) read copied
            with_title_bytes(base, 3 << 12, |b| {
                assert_eq!(b, &read_committed(base, 3 << 12)[..]);
                assert!(b[..1 << 12].iter().all(|&x| x == 0xA0) && b[1 << 12..2 << 12].iter().all(|&x| x == 0) && b[2 << 12..].iter().all(|&x| x == 0xA2));
            });
            // a shorter read after a longer one: its hole too, and only its length
            with_title_bytes(base + (1 << 12), 1 << 12, |b| assert!(b.len() == 1 << 12 && b.iter().all(|&x| x == 0)));
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
            // a read of more than `TITLE_BYTES_KEPT` does not stay with the thread
            let big = (65 << 20) + 1;
            let at = VirtualAlloc(std::ptr::null(), big, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            if at != 0 {
                std::ptr::write_bytes(at as *mut u8, 0x6D, big);
                with_title_bytes(at, big, |b| assert!(b.len() == big && b.iter().all(|&x| x == 0x6D)));
                assert!(TITLE_BYTES.with(|b| b.borrow().capacity()) <= TITLE_BYTES_KEPT, "the buffer of a large read is let go");
                VirtualFree(at as *mut c_void, 0, MEM_RELEASE);
            }
        }
        let Some(u) = crate::gpu::uma::get() else { return println!("title_bytes_are_copied_outside_the_coherent_window: no UMA, in-place read not checked") };
        let Some(p) = u.committed_window().and_then(|k| u.alloc_in(k, 1 << 16, 1 << 16)) else {
            return println!("title_bytes_are_copied_outside_the_coherent_window: no coherent window, in-place read not checked");
        };
        unsafe {
            std::ptr::write_bytes(p, 0x5C, 1 << 16);
            with_title_bytes(p as usize, 1 << 16, |b| {
                assert_eq!(b.as_ptr(), p as *const u8, "the coherent window is read in place");
                assert!(b.len() == 1 << 16 && b.iter().all(|&x| x == 0x5C));
            });
        }
        u.free(p as usize);
    }

    /// Only the wanted parts of the watched regions are scanned, merged, one region each.
    #[test]
    fn watch_ranges_clip_and_merge() {
        let regions = [(0x10000, 0x10000), (0x40000, 0x10000), (0x1000_0000, 0x1000_0000)];
        let wanted = vec![(0x12000, 0x13000), (0x11000, 0x12800), (0x1F000, 0x41000), (0x60000, 0x61000)];
        assert_eq!(watch_ranges(&regions, wanted), vec![(0x11000, 0xF000), (0x40000, 0x10000)]);
        // ranges a gap of under WATCH_GAP apart: one scan; further apart: two
        let wanted = vec![(0x1000_0000, 0x1000_1000), (0x1008_0000, 0x1008_1000), (0x1100_0000, 0x1100_1000), (0x1800_0000, 0x1800_1000)];
        assert_eq!(watch_ranges(&regions, wanted), vec![(0x1000_0000, 0x8_1000), (0x1100_0000, 0x1000), (0x1800_0000, 0x1000)]);
        // a merged range over two regions: a piece in each, whatever order the regions come in
        let regions = [(0x40000, 0x10000), (0x10000, 0x10000)];
        assert_eq!(watch_ranges(&regions, vec![(0x18000, 0x48000)]), vec![(0x18000, 0x8000), (0x40000, 0x8000)]);
        // a region between two merged ranges, and one past them all
        let regions = [(0x2000_0000, 0x10_0000), (0x9000_0000, 0x1000)];
        assert_eq!(watch_ranges(&regions, vec![(0x1000_0000, 0x1000_1000), (0x2008_0000, 0x2009_0000)]), vec![(0x2008_0000, 0x1_0000)]);
    }

    #[test]
    fn watch_ranges_merge_at_the_gap_limit_and_split_past_it() {
        let base = 0x1000_0000usize;
        let regions = [(base, 4 * WATCH_GAP)];
        let first = (base, base + 0x1000);
        let at_limit = (first.1 + WATCH_GAP, first.1 + WATCH_GAP + 0x1000);
        assert_eq!(watch_ranges(&regions, vec![at_limit, first]), vec![(base, 2 * 0x1000 + WATCH_GAP)]);

        let past_limit = (first.1 + WATCH_GAP + 1, first.1 + WATCH_GAP + 0x1001);
        assert_eq!(watch_ranges(&regions, vec![past_limit, first]), vec![(base, 0x1000), (past_limit.0, 0x1000)]);
    }

    #[test]
    fn watch_ranges_are_sorted_disjoint_in_region_and_cover_every_wanted_part() {
        let regions = [(0x4000_0000usize, 0x100_0000usize), (0x1000_0000, 0x200_0000), (0x8000_0000, 0x20_0000)];
        let wanted = vec![
            (0x0FFF_F000, 0x1000_2000),
            (0x1008_0000, 0x1008_1000),
            (0x11FF_F000, 0x1200_1000),
            (0x4000_8000, 0x4000_A000),
            (0x40FF_F000, 0x4100_1000),
            (0x8008_0000, 0x8008_1000),
            (0xA000_0000, 0xA000_1000),
        ];
        let ranges = watch_ranges(&regions, wanted.clone());

        assert!(ranges.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0), "ranges must be sorted and disjoint: {ranges:x?}");
        for &(base, size) in &ranges {
            let end = base + size;
            assert!(size > 0 && regions.iter().any(|&(rbase, rsize)| rbase <= base && end <= rbase + rsize), "range {base:#x}..{end:#x} is outside one allocation");
        }
        for &(start, end) in &wanted {
            for &(rbase, rsize) in &regions {
                let (lo, hi) = (start.max(rbase), end.min(rbase + rsize));
                if lo < hi {
                    assert!(ranges.iter().any(|&(base, size)| base <= lo && hi <= base + size), "wanted part {lo:#x}..{hi:#x} is not covered");
                }
            }
        }
    }

    #[test]
    fn budgeted_watch_ranges_merge_exact_spans_by_smallest_gap() {
        let regions = [(100, 100)];
        let wanted = vec![(165, 175), (150, 160), (120, 130), (100, 110), (135, 145), (108, 120)];
        assert_eq!(plan_watch_ranges(&regions, wanted.clone(), usize::MAX), vec![(100, 30), (135, 10), (150, 10), (165, 10)]);
        assert_eq!(plan_watch_ranges(&regions, wanted.clone(), 3), vec![(100, 45), (150, 10), (165, 10)], "equal gaps choose the lower address");
        assert_eq!(plan_watch_ranges(&regions, wanted.into_iter().rev().collect(), 2), vec![(100, 60), (165, 10)], "input order does not change the selected gaps");
    }

    #[test]
    fn budgeted_watch_ranges_do_not_bridge_allocations_or_invalid_regions() {
        let regions = [(100, 20), (120, 20), (usize::MAX - 4, 8), (200, 0)];
        let wanted = vec![(105, 135), (usize::MAX - 3, usize::MAX), (200, 210)];
        assert_eq!(plan_watch_ranges(&regions, wanted, 0), vec![(105, 15), (120, 15)], "zero target cannot merge two allocations");
    }

    #[test]
    fn watched_texture_span_rejects_size_and_rounding_overflow() {
        assert_eq!(watched_texture_span(0x1234, 1), Some((0x1000, 0x2000)));
        assert_eq!(watched_texture_span(0x1234, 0), None);
        assert_eq!(watched_texture_span(usize::MAX - 0xFFF, 0x1000), None);
        assert_eq!(watched_texture_span(usize::MAX - 0x800, 1), None);
    }

    #[test]
    fn texture_upload_selection_preserves_flag_and_page_boundaries() {
        const PAGE: usize = 0x1000;
        let dirty = [0x1000usize, 0x4000];
        for cpu_backed in [false, true] {
            for pending in [false, true] {
                for gpu_written in [false, true] {
                    let selected = texture_needs_upload(cpu_backed, pending, gpu_written, || {
                        let i = dirty.partition_point(|&p| p + PAGE <= 0x1800);
                        dirty.get(i).is_some_and(|&p| p < 0x1800 + PAGE)
                    });
                    assert_eq!(selected, cpu_backed && (pending || !gpu_written), "cpu {cpu_backed}, pending {pending}, GPU-written {gpu_written}");
                }
            }
        }
        let overlap = |src: usize, dirty: &[usize]| {
            let i = dirty.partition_point(|&p| p + PAGE <= src);
            dirty.get(i).is_some_and(|&p| p < src + PAGE)
        };
        assert!(!texture_needs_upload(true, false, false, || overlap(0x2000, &[0x1000])), "page end equals texture start");
        assert!(!texture_needs_upload(true, false, false, || overlap(0x1000, &[0x2000])), "page start equals texture end");
        assert!(texture_needs_upload(true, false, false, || overlap(0x1800, &[0x1000])), "half-open ranges overlap");
    }

    #[test]
    fn texture_upload_selection_reads_dirty_overlap_only_when_needed() {
        let calls = std::cell::Cell::new(0);
        let selected = |cpu_backed, pending, gpu_written| {
            texture_needs_upload(cpu_backed, pending, gpu_written, || {
                calls.set(calls.get() + 1);
                true
            })
        };
        assert!(!selected(false, true, false));
        assert!(selected(true, true, true));
        assert!(!selected(true, false, true));
        assert_eq!(calls.get(), 0, "an excluded or pending entry evaluated its dirty range");
        assert!(selected(true, false, false));
        assert_eq!(calls.get(), 1);
    }

    #[cfg(feature = "texture-selection-bench")]
    #[test]
    fn texture_selection_fixture_keeps_empty_gate_and_exact_aliases() {
        assert_eq!(TextureSelectionBench::new(128, TextureSelectionScenario::Empty).scan(), (0, 0));
        assert_eq!(TextureSelectionBench::new(128, TextureSelectionScenario::AllPending).scan().0, 128);
        assert_eq!(TextureSelectionBench::new(128, TextureSelectionScenario::AllGpuWritten).scan().0, 0);
        let mixed = TextureSelectionBench::new(32, TextureSelectionScenario::MixedAliases);
        {
            let entries = mixed.entries.lock().unwrap();
            assert_eq!((entries[&1].src, entries[&1].size), (entries[&2].src, entries[&2].size), "alias fixture ranges differ");
        }
        let entry = |src| TextureSelectionBenchEntry {
            src,
            size: TextureSelectionBench::PAGE,
            cpu_backed: true,
            pending: false,
            gpu_written: false,
        };
        let aliases = TextureSelectionBench {
            entries: Mutex::new(BTreeMap::from([(1, entry(0x1000)), (2, entry(0x1000))])),
            dirty: vec![0x1000],
            pending: std::sync::atomic::AtomicUsize::new(0),
        };
        assert_eq!(aliases.scan(), (2, 3), "exact-range aliases remain distinct candidates");
    }

    #[test]
    fn budgeted_watch_ranges_cover_every_wanted_byte() {
        let regions = [(20, 20), (80, 30), (45, 10)];
        let wanted = vec![(10, 25), (24, 29), (35, 50), (52, 54), (70, 87), (90, 120)];
        for budget in [usize::MAX, 4, 1] {
            let planned = plan_watch_ranges(&regions, wanted.clone(), budget);
            for &(base, size) in &regions {
                for at in base..base + size {
                    let wanted_here = wanted.iter().any(|&(a, b)| a <= at && at < b);
                    let covered = planned.iter().any(|&(a, n)| a <= at && at < a + n);
                    assert!(!wanted_here || covered, "{at} is missing with budget {budget}");
                }
            }
        }
    }

    #[test]
    fn failed_upload_requeues_only_the_exact_cpu_generation() {
        let tex = Arc::new(xg::Texture::new(&console_desc(3, 4, 4, 1, 1, 28, 0x105)).unwrap());
        let entry = |generation, gpu_written| Entry {
            device: 0,
            committed_bytes: 0,
            va: 0x1000,
            src: 0x1000,
            in_gpu_view: false,
            tex: Arc::clone(&tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending: false,
            gpu_written,
            gpu_writers: Vec::new(),
            cpu_backed: true,
            placed: false,
            unmapped: false,
            generation,
            paged: false,
            written_call: 0,
            known: Default::default(),
            target: false,
        };
        let mut table = Table::new();
        table.by_res.insert(1, entry(7, false));
        table.by_res.insert(2, entry(8, false));
        table.by_res.insert(3, entry(9, true));

        assert_eq!(requeue_generations(&mut table, &[(1, 7), (2, 7), (3, 9), (4, 1)]), [1]);
        assert!(table.by_res[&1].pending);
        assert!(table.pending_res.contains(&1) && table.pending_res.len() == 1, "nv's pending set follows");
        assert!(!table.by_res[&2].pending, "a replacement generation owns its own pending state");
        assert!(!table.by_res[&3].pending, "GPU-written content must not be overwritten by a retry");
    }


    #[test]
    fn small_cpu_filled_target_accepts_wide_glyph_atlas_and_rejects_large_or_depth() {
        assert!(is_small_cpu_filled_target(2048, 128, 1, 0x300005), "2048x128 R8 glyph atlas must sync");
        assert!(is_small_cpu_filled_target(1024, 1024, 1, 0x1));
        assert!(is_small_cpu_filled_target(64, 64, 1, 0x5));
        assert!(!is_small_cpu_filled_target(1, 1, 1, 0x1), "1x1 placeholders skipped");
        assert!(!is_small_cpu_filled_target(2560, 1440, 1, 0x1), "full-screen RT skipped");
        assert!(!is_small_cpu_filled_target(2048, 1024, 1, 0x1), ">1M pixel RT skipped");
        assert!(!is_small_cpu_filled_target(256, 256, 1, 0x2), "depth buffers skipped");
        assert!(!is_small_cpu_filled_target(256, 256, 4, 0x1), "MSAA RTs skipped");
    }

    pub(in crate::hle::d3d12x) fn console_desc(dim: u32, w: u64, h: u32, array_or_depth: u16, mips: u16, format: u32, layout: u32) -> [u8; 0x38] {
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&dim.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&w.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&h.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&array_or_depth.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
        d
    }

    #[test]
    fn committed_sampled_bytes_follow_device_shadowing_and_release() {
        const M: u64 = 1 << 20;
        let tex = Arc::new(xg::Texture::new(&console_desc(3, 64, 64, 1, 1, 28, 0x105)).unwrap());
        let entry = |device, va, committed_bytes, placed| Entry {
            device,
            committed_bytes,
            va,
            src: va,
            in_gpu_view: false,
            tex: Arc::clone(&tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending: false,
            gpu_written: false,
            gpu_writers: Vec::new(),
            cpu_backed: true,
            placed,
            unmapped: false,
            generation: 0,
            paged: false,
            written_call: 0,
            known: Default::default(),
            target: false,
        };
        let mut table = Table::new();
        table.insert(1, entry(7, 0x10_0000, 3 * M, false));
        table.insert(2, entry(7, 0x10_2000, 2 * M, false)); // shadows 1, which remains live
        table.insert(3, entry(7, 0x20_0000, 8 * M, true)); // shared pages, not committed
        table.insert(4, entry(8, 0x30_0000, 11 * M, false));
        assert_eq!(table.committed_bytes_for(7), (2, 5 * M));
        assert_eq!(table.committed_bytes_for(8), (1, 11 * M));
        assert!(table.remove(2).0);
        assert_eq!(table.committed_bytes_for(7), (1, 3 * M), "the uncovered sampled texture is counted once");
        assert!(table.remove(1).0);
        assert_eq!(table.committed_bytes_for(7), (0, 0), "logical release removes the partial diagnostic owner");
    }

    fn readback_for(tex: &xg::Texture, format: u32, subresource: u32) -> (ReadbackFootprint, Vec<u8>, Vec<u8>) {
        let s = tex.texels().unwrap();
        let mips = s.chain.levels.len();
        let mip = subresource as usize % mips;
        let l = &s.chain.levels[mip];
        let (w, h) = ((s.width >> mip).max(1), (s.height >> mip).max(1));
        let depth = if s.depth > 0 { (s.depth >> mip).max(1) } else { 1 };
        let row_bytes = l.width * s.bpe as usize;
        let row_pitch = row_bytes.next_multiple_of(256);
        let slice_pitch = row_pitch * l.height;
        let size = slice_pitch * (l.depth - 1) + row_pitch * (l.height - 1) + row_bytes;
        let mut readback = vec![0xE7; 512 + size + 31];
        let mut linear = vec![0; row_bytes * l.height * l.depth];
        for z in 0..l.depth {
            for y in 0..l.height {
                for x in 0..row_bytes {
                    let v = (z * 71 + y * 17 + x * 3 + 5) as u8;
                    readback[512 + z * slice_pitch + y * row_pitch + x] = v;
                    linear[(z * l.height + y) * row_bytes + x] = v;
                }
            }
        }
        (ReadbackFootprint { offset: 512, size, row_pitch, width: w, height: h, depth, format }, readback, linear)
    }

    fn driver_download_layout(
        tex: &xg::Texture,
        format: u32,
        selected: &[(u32, D3D12_RESOURCE_STATES)],
    ) -> Vec<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u32, u64)> {
        let s = tex.texels().unwrap();
        let mips = s.chain.levels.len();
        let mut end = 0usize;
        selected
            .iter()
            .map(|&(sub, _)| {
                let mip = if s.depth > 0 { sub as usize } else { sub as usize % mips };
                let l = &s.chain.levels[mip];
                let row_bytes = l.width * s.bpe as usize;
                let pitch = row_bytes.next_multiple_of(256);
                let offset = end.next_multiple_of(512);
                end = offset + pitch * l.height * (l.depth - 1) + pitch * (l.height - 1) + row_bytes;
                (
                    D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: offset as u64,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(format as i32),
                            Width: (s.width >> mip).max(1),
                            Height: (s.height >> mip).max(1),
                            Depth: if s.depth > 0 { (s.depth >> mip).max(1) } else { 1 },
                            RowPitch: pitch as u32,
                        },
                    },
                    l.height as u32,
                    row_bytes as u64,
                )
            })
            .collect()
    }

    #[test]
    fn download_plans_use_writer_capable_queue_types() {
        assert!(download_queue_type_supported(D3D12_COMMAND_LIST_TYPE_DIRECT));
        assert!(download_queue_type_supported(D3D12_COMMAND_LIST_TYPE_COMPUTE));
        assert!(!download_queue_type_supported(D3D12_COMMAND_LIST_TYPE_COPY));
    }

    #[test]
    fn download_layout_preserves_explicit_states_and_bounds_array_and_volume_footprints() {
        let array_desc = console_desc(3, 17, 9, 2, 3, 28, 0x105);
        let array = xg::Texture::new(&array_desc).unwrap();
        let desc: D3D12_RESOURCE_DESC = unsafe { std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&array_desc).as_ptr() as *const _) };
        let selected = [(0, D3D12_RESOURCE_STATE_COMMON), (5, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)];
        let driver = driver_download_layout(&array, 28, &selected);
        let (layout, bytes) = validate_download_layout(&array, &desc, 28, &selected, &driver).unwrap();
        assert_eq!(layout.iter().map(|s| (s.subresource, s.state)).collect::<Vec<_>>(), selected);
        assert_eq!(bytes, layout.last().unwrap().footprint.offset + layout.last().unwrap().footprint.size);

        let volume_desc = console_desc(4, 32, 16, 8, 4, 28, 0x105);
        let volume = xg::Texture::new(&volume_desc).unwrap();
        let desc: D3D12_RESOURCE_DESC = unsafe { std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&volume_desc).as_ptr() as *const _) };
        let selected = [(0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS), (2, D3D12_RESOURCE_STATE_COPY_SOURCE)];
        let driver = driver_download_layout(&volume, 28, &selected);
        let (layout, _) = validate_download_layout(&volume, &desc, 28, &selected, &driver).unwrap();
        assert!(layout[1].footprint.offset >= layout[0].footprint.offset + layout[0].footprint.size);
        assert_eq!((layout[1].footprint.width, layout[1].footprint.height, layout[1].footprint.depth), (8, 4, 2));
    }

    #[test]
    fn download_layout_rejects_stale_or_driver_enlarged_selections() {
        let bytes = console_desc(3, 16, 8, 2, 2, 28, 0x105);
        let tex = xg::Texture::new(&bytes).unwrap();
        let desc: D3D12_RESOURCE_DESC = unsafe { std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&bytes).as_ptr() as *const _) };
        let selected = [(0, D3D12_RESOURCE_STATE_COMMON), (3, D3D12_RESOURCE_STATE_RENDER_TARGET)];
        let exact = driver_download_layout(&tex, 28, &selected);
        let mut bad = exact.clone();
        bad[1].0.Footprint.RowPitch += 256;
        assert!(validate_download_layout(&tex, &desc, 28, &selected, &bad).is_err());
        let mut bad = exact.clone();
        bad[0].2 += 4;
        assert!(validate_download_layout(&tex, &desc, 28, &selected, &bad).is_err());
        let duplicate = [(0, D3D12_RESOURCE_STATE_COMMON), (0, D3D12_RESOURCE_STATE_RENDER_TARGET)];
        assert!(validate_download_layout(&tex, &desc, 28, &duplicate, &driver_download_layout(&tex, 28, &duplicate)).is_err());
        let invalid_state = [(0, D3D12_RESOURCE_STATES(-1))];
        assert!(validate_download_layout(&tex, &desc, 28, &invalid_state, &driver_download_layout(&tex, 28, &invalid_state)).is_err());
        assert!(validate_download_layout(&tex, &desc, 28, &selected[..1], &exact).is_err());
        let mut wrong_dimension = desc;
        wrong_dimension.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        assert!(validate_download_layout(&tex, &wrong_dimension, 28, &selected, &exact).is_err());

        let rgba = xg::Texture::new(&console_desc(3, 4, 4, 1, 1, 28, 0x105)).unwrap();
        let bc = console_desc(3, 4, 4, 1, 1, 77, 0x105); // both rows are 16 bytes; block side differs
        let bc_desc: D3D12_RESOURCE_DESC = unsafe { std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&bc).as_ptr() as *const _) };
        let one = [(0, D3D12_RESOURCE_STATE_RENDER_TARGET)];
        assert!(validate_download_layout(&rgba, &bc_desc, 77, &one, &driver_download_layout(&rgba, 77, &one)).is_err());
        let mut rt = console_desc(3, 16, 8, 1, 1, 28, 0x105);
        rt[0x30..0x34].copy_from_slice(&1u32.to_le_bytes());
        let rt = xg::Texture::new(&rt).unwrap();
        assert!(validate_download_layout(&rt, &desc, 28, &one, &driver_download_layout(&rt, 28, &one)).is_err());
    }

    #[test]
    fn reverse_layout_tiles_an_edge_array_mip_without_touching_guards() {
        let tex = xg::Texture::new(&console_desc(3, 17, 9, 2, 3, 28, 0x105)).unwrap();
        let sub = 5; // array slice 1, mip 2
        let (fp, readback, linear) = readback_for(&tex, 28, sub);
        let size = tex.size() as usize;
        let mut console = vec![0xCD; size + 37];
        reverse_layout_subresource(&tex, 28, sub, fp, &readback, &mut console).unwrap();

        let s = tex.texels().unwrap();
        let slice = &console[s.chain.slice_bytes..2 * s.chain.slice_bytes];
        assert_eq!(s.chain.detile_level(slice, 2, 1), linear);
        assert!(console[size..].iter().all(|&b| b == 0xCD), "destination guard changed");
        assert!(readback[..512].iter().all(|&b| b == 0xE7));
        assert!(readback[512 + fp.size..].iter().all(|&b| b == 0xE7));
    }

    #[test]
    fn reverse_layout_handles_volume_depth_and_block_geometry() {
        let volume = xg::Texture::new(&console_desc(4, 32, 16, 8, 4, 28, 0x105)).unwrap();
        let (fp, readback, linear) = readback_for(&volume, 28, 2);
        assert_eq!((fp.width, fp.height, fp.depth), (8, 4, 2));
        let mut console = vec![0u8; volume.size() as usize];
        reverse_layout_subresource(&volume, 28, 2, fp, &readback, &mut console).unwrap();
        assert_eq!(volume.texels().unwrap().chain.detile_volume(&console, 2), linear);

        let blocks = xg::Texture::new(&console_desc(3, 10, 9, 1, 1, 71, 0x105)).unwrap();
        let (fp, readback, linear) = readback_for(&blocks, 71, 0);
        assert_eq!((fp.width, fp.height, fp.depth), (10, 9, 1));
        assert_eq!(blocks.texels().unwrap().chain.levels[0].width, 3);
        let mut console = vec![0u8; blocks.size() as usize];
        reverse_layout_subresource(&blocks, 71, 0, fp, &readback, &mut console).unwrap();
        assert_eq!(blocks.texels().unwrap().chain.detile_level(&console, 0, 0), linear);
    }

    #[test]
    fn reverse_layout_rejects_metadata_malformed_footprints_and_truncation() {
        let tex = xg::Texture::new(&console_desc(3, 16, 8, 1, 1, 28, 0x105)).unwrap();
        let (fp, readback, _) = readback_for(&tex, 28, 0);
        let mut console = vec![0u8; tex.size() as usize];
        let call = |f: ReadbackFootprint, src: &[u8], dst: &mut [u8]| reverse_layout_subresource(&tex, 28, 0, f, src, dst);
        assert_eq!(call(ReadbackFootprint { format: 87, ..fp }, &readback, &mut console), Err(ReverseLayoutError::Format));
        assert_eq!(reverse_layout_subresource(&tex, 28, 1, fp, &readback, &mut console), Err(ReverseLayoutError::Subresource));
        assert_eq!(call(ReadbackFootprint { width: 15, ..fp }, &readback, &mut console), Err(ReverseLayoutError::Dimensions));
        assert_eq!(call(ReadbackFootprint { row_pitch: 255, ..fp }, &readback, &mut console), Err(ReverseLayoutError::RowPitch));
        assert_eq!(call(ReadbackFootprint { offset: 1, ..fp }, &readback, &mut console), Err(ReverseLayoutError::RowPitch));
        assert_eq!(call(ReadbackFootprint { size: fp.size - 1, ..fp }, &readback, &mut console), Err(ReverseLayoutError::Footprint));
        assert_eq!(call(fp, &readback[..512 + fp.size - 1], &mut console), Err(ReverseLayoutError::Footprint));
        assert_eq!(call(fp, &readback, &mut console[..tex.size() as usize - 1]), Err(ReverseLayoutError::Destination));
        let overflow = ReadbackFootprint { offset: usize::MAX & !511, ..fp };
        assert_eq!(call(overflow, &readback, &mut console), Err(ReverseLayoutError::Footprint));

        let mut rt = console_desc(3, 16, 8, 1, 1, 28, 0x105);
        rt[0x30..0x34].copy_from_slice(&1u32.to_le_bytes());
        let rt = xg::Texture::new(&rt).unwrap();
        let mut rt_console = vec![0u8; rt.size() as usize];
        assert_eq!(reverse_layout_subresource(&rt, 28, 0, fp, &readback, &mut rt_console), Err(ReverseLayoutError::UnsupportedLayout));
    }

    /// Overlapping textures shadow earlier ones so dead placements never re-upload over shared
    /// `vram::Pages` tiles; different-range overlaps request unmapping while exact same-range
    /// overlaps stay mapped; releasing a newer placement reactivates uncovered older ones.
    #[test]
    fn overlapping_placements_retire_superseded_entries() {
        let tex_16k = Arc::new(xg::Texture::new(&console_desc(3, 64, 64, 1, 1, 28, 0x105)).unwrap());
        let tex_64k = Arc::new(xg::Texture::new(&console_desc(3, 128, 128, 1, 1, 28, 0x105)).unwrap());
        assert_eq!((tex_16k.size(), tex_64k.size()), (0x4000, 0x1_0000));
        let entry = |va: usize, tex: &Arc<xg::Texture>, pending: bool, placed: bool| Entry {
            device: 0,
            committed_bytes: 0,
            va,
            src: va,
            in_gpu_view: false,
            paged: false,
            tex: Arc::clone(tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending,
            gpu_written: false,
            written_call: 0,
            gpu_writers: Vec::new(),
            cpu_backed: true,
            placed,
            unmapped: false,
            generation: 0,
            known: Default::default(),
            target: false,
        };
        let mut tbl = Table::new();
        // Two adjacent placed 16 KiB textures + one placed 64 KiB texture after them.
        assert_eq!(tbl.insert(1, entry(0x10_0000, &tex_16k, true, true)), (1, vec![]));
        assert_eq!(tbl.insert(2, entry(0x10_4000, &tex_16k, true, true)), (1, vec![]));
        assert_eq!(tbl.insert(3, entry(0x11_0000, &tex_64k, false, true)), (0, vec![]));
        assert_eq!(tbl.by_res.len(), 3);
        // Exact same-range sibling over 2 shadows 2 without unmapping (tiles are shared).
        assert_eq!(tbl.insert(20, entry(0x10_4000, &tex_16k, false, true)), (-1, vec![]));
        assert!(!tbl.by_res.contains_key(&2) && tbl.shadowed.contains_key(&2));
        // A 64 KiB placed texture at 0x10_0000 supersedes 1, 20, and shadowed sibling 2 (all different range).
        let (delta, mut unmap) = tbl.insert(4, entry(0x10_0000, &tex_64k, true, true));
        unmap.sort_unstable();
        assert_eq!((delta, unmap), (0, vec![1, 2, 20]));
        assert!(tbl.by_res.contains_key(&3) && tbl.by_res.contains_key(&4));
        // Releasing 4 uncovers newest non-overlapping shadowed entries (1 and 20) and requests remap.
        let (removed, delta, mut remap) = tbl.remove(4);
        remap.sort_unstable();
        assert_eq!((removed, delta, remap), (true, 1, vec![1, 20]));
        assert!(tbl.by_res.contains_key(&1) && tbl.by_res.contains_key(&20) && tbl.shadowed.contains_key(&2));
        // Releasing 20 uncovers 2; releasing middle of a 3-chain still uncovers oldest when top goes.
        let (removed, _, remap) = tbl.remove(20);
        assert_eq!((removed, remap), (true, vec![2]));
        assert!(tbl.by_res.contains_key(&2));
        assert_eq!(tbl.insert(5, entry(0x10_8000, &tex_64k, true, true)).0, 1);
        assert_eq!(tbl.overlapping(0x10_0000, 0x1_0000), vec![(1, 0x10_0000), (2, 0x10_4000), (5, 0x10_8000)]);
        // Two committed resources with the exact same range coexist; a shifted one shadows them.
        assert_eq!(tbl.insert(10, entry(0x20_0000, &tex_16k, true, false)), (1, vec![]));
        assert_eq!(tbl.insert(11, entry(0x20_0000, &tex_16k, true, false)), (1, vec![]));
        assert!(tbl.by_res.contains_key(&10) && tbl.by_res.contains_key(&11));
        assert_eq!(tbl.insert(12, entry(0x20_0000, &tex_64k, true, false)), (-1, vec![]));
        assert!(!tbl.by_res.contains_key(&10) && !tbl.by_res.contains_key(&11) && tbl.by_res.contains_key(&12));
    }

    /// Real GPU (#248): placed title textures share `vram::Pages` so streaming unreleased
    /// overlapping placements stays bounded in video memory, superseded textures never corrupt
    /// newer placements on shared tiles, 4 KiB-aligned neighbours never false-alias, and mip
    /// chains (including packed mips), cubemap arrays (`extra_pages`) and 3D volumes upload intact.
    #[test]
    fn placed_title_textures_share_vram_and_preserve_pixels() {
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: no adapter");
                return;
            };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: software adapter");
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: no D3D12 device");
                return;
            };
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32).is_err()
                || opts.ResourceHeapTier.0 < 2
                || opts.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0
            {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: tiled tier < 2");
                return;
            }
            // An adapter that reports tier 2 but refuses a reserved cube with packed mips
            // (GitHub's runner) does not run the shared-page path this test is about.
            {
                let mut cube: D3D12_RESOURCE_DESC = std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&console_desc(3, 256, 256, 6, 9, 28, 0x105)).as_ptr() as *const _);
                cube.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
                let mut r: Option<ID3D12Resource> = None;
                if dev.CreateReservedResource(&cube, D3D12_RESOURCE_STATE_COMMON, None, &mut r).is_err() || r.is_none() {
                    println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: this adapter makes no reserved cube with mips");
                    return;
                }
            }
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let Some(uma) = crate::gpu::uma::get() else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: no UMA");
                return;
            };
            let pool_size = 16usize << 20;
            let Some(base) = uma.alloc(pool_size, 0x1_0000).map(|p| p as usize) else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: UMA alloc failed");
                return;
            };
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local_vram = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_tex = |va: usize, desc: &[u8; 0x38]| -> ID3D12Resource {
                let mut p = std::ptr::null_mut();
                let hr = super::super::device_x::create_placed_resource_x(dev.as_raw(), va as u64, desc.as_ptr(), D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p);
                assert!(hr >= 0 && !p.is_null(), "CreatePlacedResourceX failed: {hr:#x}");
                assert!(super::super::vram::is_placed_texture(p as usize), "expected shared vram::Pages texture at {va:#x}");
                ID3D12Resource::from_raw(p)
            };
            let write_tiled = |va: usize, desc: &[u8; 0x38], seed: u8| {
                let tex = xg::Texture::new(desc).unwrap();
                let s = tex.texels().unwrap();
                let bytes = std::slice::from_raw_parts_mut(va as *mut u8, tex.size() as usize);
                bytes.fill(0);
                if s.depth > 0 {
                    for (m, l) in s.chain.levels.iter().enumerate() {
                        let pat: Vec<u8> = (0..l.width * l.height * l.depth * s.bpe as usize).map(|k| seed.wrapping_add(m as u8).wrapping_add((k * 13) as u8)).collect();
                        s.chain.tile_volume(bytes, m, &pat);
                    }
                } else {
                    for slice in 0..s.array as usize {
                        let dst = &mut bytes[slice * s.chain.slice_bytes..(slice + 1) * s.chain.slice_bytes];
                        for (m, l) in s.chain.levels.iter().enumerate() {
                            let pat: Vec<u8> = (0..l.width * l.height * s.bpe as usize).map(|k| seed.wrapping_add((slice * 31 + m * 7) as u8).wrapping_add((k * 13) as u8)).collect();
                            s.chain.tile_level(dst, m, slice, &pat);
                        }
                    }
                }
            };
            let read_sub = |r: &ID3D12Resource, sub: u32| -> Vec<u8> {
                let rd = r.GetDesc();
                let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
                dev.GetCopyableFootprints(&rd, sub, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
                let mut rb: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut rb,
                )
                .unwrap();
                let rb = rb.unwrap();
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.Close().unwrap();
                own_work(|| {
                    super::super::vram::before_execute(&queue);
                    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                    texture::wait(&dev, &queue).unwrap();
                });
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).unwrap();
                let slice_pitch = fp.Footprint.RowPitch as usize * rows as usize;
                let mut out = Vec::with_capacity(fp.Footprint.Depth as usize * rows as usize * row_bytes as usize);
                for z in 0..fp.Footprint.Depth as usize {
                    for y in 0..rows as usize {
                        out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(z * slice_pitch + y * fp.Footprint.RowPitch as usize), row_bytes as usize));
                    }
                }
                rb.Unmap(0, None);
                out
            };

            // 1. Streaming 512 unreleased 1 MiB textures (512 MiB placed) across 16 MiB stays bounded
            // with zero additional page growth after the 16 MiB working set is covered.
            let dev_key = dev.as_raw() as usize;
            let dev_pages = || super::super::vram::page_bytes_for(dev_key).0;
            let stream_desc = console_desc(3, 512, 512, 1, 1, 28, 0x109); // 64KB_S, 1 MiB
            let (pages0, vram0, kept0) = (dev_pages(), local_vram(), counts().0);
            let mut leaked_refs = Vec::with_capacity(512);
            for k in 0..16usize {
                leaked_refs.push(place_tex(base + k * (1 << 20), &stream_desc));
            }
            let (pages16, vram16) = (dev_pages(), local_vram());
            for k in 16..512usize {
                leaked_refs.push(place_tex(base + (k % 16) * (1 << 20), &stream_desc));
            }
            let (pages_delta, vram_delta, vram_after_fill, kept_delta) = (
                dev_pages() - pages0,
                local_vram().saturating_sub(vram0) as usize,
                local_vram().saturating_sub(vram16) as usize,
                counts().0 - kept0,
            );
            println!("512 unreleased 1 MiB textures over 16 MiB: +{pages_delta} B pages (after fill: +{} B), +{vram_delta} B local VRAM (after fill: +{vram_after_fill} B), +{kept_delta} active kept", dev_pages() - pages16);
            assert_eq!(pages_delta, 16 << 20, "pages must equal covered 16 MiB, not 512 MiB");
            assert_eq!(dev_pages(), pages16, "no pages added after working set is covered");
            // (the process's local video memory is not asserted: other GPU tests in the process
            // allocate at the same time - on an RX 9070 it read +672 MB under --test-threads=4 and
            // stayed within the bound alone; the page count above is what this change controls)
            let _ = vram_after_fill;
            assert_eq!(kept_delta, 16, "superseded placements must be retired from active tracking");

            // 2. Overlap integrity (rows 4, 5, 6) + unmap/remap on different-range overlap:
            // (a) Exact same range: old object sees new uploads (row 4); releasing new object
            // reactivates old object for subsequent CPU rewrites (row 6).
            let same_desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB
            let same_a = place_tex(base + 0x1000, &same_desc); // 4 KiB offset too!
            write_tiled(base + 0x1000, &same_desc, 0x1A);
            refresh(queue.as_raw());
            let same_b = place_tex(base + 0x1000, &same_desc);
            write_tiled(base + 0x1000, &same_desc, 0x2B);
            refresh(queue.as_raw());
            let want_2b = subresources(base + 0x1000, &xg::Texture::new(&same_desc).unwrap()).swap_remove(0).data;
            assert_eq!(read_sub(&same_b, 0), want_2b, "row 3: new placement readback");
            assert_eq!(read_sub(&same_a, 0), want_2b, "row 4: older exact-range alias shares tiles");
            drop(same_b);
            write_tiled(base + 0x1000, &same_desc, 0x3C);
            refresh(queue.as_raw());
            assert_eq!(read_sub(&same_a, 0), subresources(base + 0x1000, &xg::Texture::new(&same_desc).unwrap())[0].data, "row 6: releasing newer placement reactivates older one");
            drop(same_a);

            // (b) Partial / different-range overlap (row 5 + unmap dead tail + remap on release):
            let old_desc = console_desc(3, 512, 512, 1, 1, 28, 0x109); // 1 MiB
            let new_desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB
            drop(leaked_refs);
            let old_res = place_tex(base, &old_desc);
            write_tiled(base, &old_desc, 0xAA);
            refresh(queue.as_raw());
            assert_eq!(read_sub(&old_res, 0), subresources(base, &xg::Texture::new(&old_desc).unwrap())[0].data);
            // Place B at base + 256 KiB (overlaps old_res tail); writing into old_res's first
            // 256 KiB must NOT re-upload old_res over B (row 5).
            let new_res = place_tex(base + 0x4_0000, &new_desc);
            write_tiled(base + 0x4_0000, &new_desc, 0x55);
            refresh(queue.as_raw());
            let want_b = subresources(base + 0x4_0000, &xg::Texture::new(&new_desc).unwrap()).swap_remove(0).data;
            *(base as *mut u8) ^= 0xFF; // dirty page outside B, inside old_res
            refresh(queue.as_raw());
            assert_eq!(read_sub(&new_res, 0), want_b, "row 5: stale shadowed texture corrupted B");
            // Releasing B remaps previously-unmapped old_res cleanly and restores its pixels.
            drop(new_res);
            write_tiled(base, &old_desc, 0xD4);
            refresh(queue.as_raw());
            assert_eq!(read_sub(&old_res, 0), subresources(base, &xg::Texture::new(&old_desc).unwrap())[0].data, "reactivated unmapped texture remapped cleanly");
            drop(old_res);

            // 3. 4 KiB-aligned adjacent textures inside one 64 KiB page + straddling page boundary,
            // plus reclaiming dead extra_pages when overwritten by a larger placement without Release.
            let d16 = console_desc(3, 64, 64, 1, 1, 28, 0x105); // 4KB_S, 16 KiB
            let d64 = console_desc(3, 128, 128, 1, 1, 28, 0x105); // 4KB_S, 64 KiB
            let adj_specs = [(0x0000usize, d16, 0x11u8), (0x4000, d16, 0x22), (0x8000, d16, 0x33), (0xC000, d64, 0x44), (0x1_C000, d16, 0x55)];
            let adj: Vec<_> = adj_specs.iter().map(|&(off, d, _)| place_tex(base + off, &d)).collect();
            for &(off, d, seed) in &adj_specs {
                write_tiled(base + off, &d, seed);
            }
            refresh(queue.as_raw());
            for (r, &(off, d, _)) in adj.iter().zip(&adj_specs) {
                assert_eq!(read_sub(r, 0), subresources(base + off, &xg::Texture::new(&d).unwrap())[0].data, "4 KiB neighbour at +{off:#x} crosstalk");
            }
            let pages_before_reuse = dev_pages();
            // Binding sweeps pages idle for two seconds, so the device-wide count can fall
            // here even when this placement shares the extra tile it already has.
            let first = adj[1].as_raw() as usize;
            let before = super::super::vram::mapped_tiles_for(dev_key, first);
            let reused = place_tex(base + 0x4000, &d16);
            if super::super::vram::is_placed_texture(first) {
                let before = before.expect("the first unaligned placement is mapped");
                assert!(!before.1.is_empty(), "a 16 KiB texture in a 64 KiB page uses an extra tile");
                assert_eq!(
                    super::super::vram::mapped_tiles_for(dev_key, first).as_ref(),
                    Some(&before),
                    "the first placement keeps its tiles when the va is placed again",
                );
                assert_eq!(
                    super::super::vram::mapped_tiles_for(dev_key, reused.as_raw() as usize).as_ref().map(|t| &t.1),
                    Some(&before.1),
                    "reusing unaligned va must share extra_pages tile",
                );
            } else {
                assert_eq!(dev_pages(), pages_before_reuse, "reusing unaligned va must share extra_pages tile");
            }
            // Overwrite the first 64 KiB (covering +0x0000, +0x4000 (both adj[1] & reused), +0x8000, +0xC000)
            // WITHOUT releasing adj/reused: dead extra_pages (+0x4000, +0x8000) must drop immediately!
            let cover = place_tex(base, &d64);
            let pages_after_cover = dev_pages();
            assert!(pages_after_cover < pages_before_reuse, "superseded unaligned extra_pages must be reclaimed immediately ({pages_before_reuse} -> {pages_after_cover})");
            drop((cover, reused, adj));

            // 4. Full mip chain (standard + packed mips, RGBA8 & BC7), tiny all-packed mips,
            // 6-slice cubemap (exercises extra_pages reuse), and 3D volume texture (Tier 3).
            let shapes = [
                ("2D 512x512 RGBA8 10 mips", console_desc(3, 512, 512, 1, 10, 28, 0x109)),
                ("2D 1024x1024 BC7 11 mips", console_desc(3, 1024, 1024, 1, 11, 98, 0x109)),
                ("2D 64x64 RGBA8 all-packed 7 mips", console_desc(3, 64, 64, 1, 7, 28, 0x105)),
                ("2D 256x256x6 cubemap 9 mips", console_desc(3, 256, 256, 6, 9, 28, 0x105)),
                ("3D 64x64x64 4 mips", console_desc(4, 64, 64, 64, 4, 28, 0x105)),
            ];
            for (label, d) in shapes {
                let mut p = std::ptr::null_mut();
                let hr = super::super::device_x::create_placed_resource_x(dev.as_raw(), base as u64, d.as_ptr(), D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p);
                assert!(hr >= 0 && !p.is_null(), "{label}: hr {hr:#x}");
                let res = ID3D12Resource::from_raw(p);
                let is_3d = u32::from_le_bytes(d[0..4].try_into().unwrap()) == 4;
                let array = u16::from_le_bytes(d[0x1C..0x1E].try_into().unwrap());
                let placed = super::super::vram::is_placed_texture(res.as_raw() as usize);
                if array > 1 && !is_3d {
                    // arrays and cubes with packed mips: a driver may refuse them reserved (GitHub's
                    // runner did with this cube); the texture is then committed – what the pixels
                    // below check either way
                    if !placed {
                        println!("{label}: not reserved on this adapter (committed instead)");
                    }
                } else if (!is_3d || opts.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_3.0)
                    && !(is_3d
                        && xg::Texture::new(&d)
                            .unwrap()
                            .texels()
                            .is_some_and(|s| s.first_tail < s.chain.levels.len()))
                {
                    assert!(placed, "{label} should use shared vram::Pages");
                } else if is_3d
                    && xg::Texture::new(&d)
                        .unwrap()
                        .texels()
                        .is_some_and(|s| s.first_tail < s.chain.levels.len())
                {
                    assert!(
                        !placed,
                        "{label} has an XG packed mip tail; use the committed detile/upload path"
                    );
                }
                write_tiled(base, &d, 0x71);
                refresh(queue.as_raw());
                let want = subresources(base, &xg::Texture::new(&d).unwrap());
                for (sub, exp) in want.iter().enumerate() {
                    assert_eq!(read_sub(&res, sub as u32), exp.data, "{label}: subresource {sub} mismatch");
                }
                // Placing a second cubemap at the same va must reuse every tile including extra_pages.
                if label.contains("cubemap") {
                    // Binding sweeps unrelated targets idle for two seconds, so a device-wide
                    // byte count can fall here even when every tile of this cube is shared.
                    let mappings = super::super::vram::tile_mappings_for(dev.as_raw() as usize, res.as_raw() as usize).expect("cube is mapped");
                    assert!(!mappings.0.is_empty() || !mappings.1.is_empty(), "cube must have mapped tiles");
                    let res2 = place_tex(base, &d);
                    assert_eq!(super::super::vram::tile_mappings_for(dev.as_raw() as usize, res2.as_raw() as usize).as_ref(), Some(&mappings),
                        "the second cube must reuse every title-page and extra-page tile");
                    assert_eq!(super::super::vram::tile_mappings_for(dev.as_raw() as usize, res.as_raw() as usize).as_ref(), Some(&mappings),
                        "the original cube must keep the same tile mappings");
                    println!("{label}: reused {} title-page tiles and {} extra-page tiles", mappings.0.len(), mappings.1.len());
                    drop(res2);
                }
                drop(res);
            }
            uma.free(base);
        }
    }

    #[test]
    fn range_is_fully_watched_handles_gaps_and_adjacent_spans() {
        let watched = [(0x10_0000usize, 0x1_0000usize), (0x11_0000, 0x2_0000), (0x20_0000, 0x1_0000)];
        assert!(range_is_fully_watched(0x10_0000, 0x8000, &watched));
        // Spanning two adjacent watched ranges is fully watched.
        assert!(range_is_fully_watched(0x10_F000, 0x4000, &watched));
        assert!(range_is_fully_watched(0x10_0000, 0x3_0000, &watched));
        // Crossing an unwatched gap or starting outside watched ranges is not fully watched.
        assert!(!range_is_fully_watched(0x12_F000, 0x2000, &watched));
        assert!(!range_is_fully_watched(0x13_0000, 0x1000, &watched));
        assert!(!range_is_fully_watched(0x0F_F000, 0x2000, &watched));
        assert!(!range_is_fully_watched(0x30_0000, 0x4000, &watched));
        assert!(!range_is_fully_watched(0x10_0000, 0x1000, &[]));
    }

    #[test]
    fn sample_fingerprint_detects_start_mid_and_end_changes_in_large_textures() {
        let mut small = vec![0u8; 32 * 1024];
        let h0 = unsafe { sample_fingerprint(small.as_ptr() as usize, small.len()) };
        small[12345] = 0x5A;
        let h1 = unsafe { sample_fingerprint(small.as_ptr() as usize, small.len()) };
        assert_ne!(h0, h1, "small texture mutation must change fingerprint");

        let mut large = vec![0u8; 256 * 1024];
        let base = unsafe { sample_fingerprint(large.as_ptr() as usize, large.len()) };
        // Start 16 KiB slice
        large[1024] = 0x11;
        let hs = unsafe { sample_fingerprint(large.as_ptr() as usize, large.len()) };
        assert_ne!(base, hs, "start-slice write must change fingerprint");
        // Middle 16 KiB slice
        large[128 * 1024] = 0x22;
        let hm = unsafe { sample_fingerprint(large.as_ptr() as usize, large.len()) };
        assert_ne!(hs, hm, "mid-slice write must change fingerprint");
        // End 16 KiB slice
        let last = large.len() - 1;
        large[last] = 0x33;
        let he = unsafe { sample_fingerprint(large.as_ptr() as usize, large.len()) };
        assert_ne!(hm, he, "end-slice write must change fingerprint");
    }

    #[test]
    fn gpu_owned_alias_generations_restore_the_previous_address_owner() {
        let tex = Arc::new(xg::Texture::new(&console_desc(3, 64, 64, 1, 1, 28, 0x105)).unwrap());
        let entry = |cpu_backed: bool| Entry {
            device: 0,
            committed_bytes: 0,
            va: 0x20_0000,
            src: 0x20_0000,
            in_gpu_view: false,
            tex: Arc::clone(&tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending: false,
            gpu_written: !cpu_backed,
            gpu_writers: Vec::new(),
            cpu_backed,
            placed: !cpu_backed,
            unmapped: false,
            generation: 0,
            paged: false,
            written_call: 0,
            known: Default::default(),
            target: false,
        };
        let mut t = Table::new();
        t.insert(1, entry(true));
        let cpu_generation = t.by_res[&1].generation;
        t.insert(2, entry(false));
        let first_gpu_generation = t.by_res[&2].generation;
        assert!(t.shadowed.contains_key(&1));
        t.insert(3, entry(false));
        assert!(t.shadowed.contains_key(&2));
        assert!(t.by_res[&3].generation > first_gpu_generation && first_gpu_generation > cpu_generation);

        t.remove(3);
        assert!(t.by_res.contains_key(&2), "the previous exact-range GPU alias returns");
        t.remove(2);
        assert!(t.by_res.get(&1).is_some_and(|e| e.cpu_backed && e.generation == cpu_generation));
    }

    #[test]
    fn stream_recheck_schedule_catches_late_writes_and_resets_milestones() {
        // (STREAM_RECHECK is shared with the other stream recheck test)
        let _guard = TEST_LOCK.lock().unwrap();
        let desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB RGBA8
        let tex = Arc::new(xg::Texture::new(&desc).expect("layout"));
        let size = tex.size() as usize;
        let mut backing = vec![0u8; size];
        let va = backing.as_ptr() as usize;
        let res_id = 0x7410_0001usize;
        let gpu_res_id = 0x7410_0002usize;

        let e_stream = Entry {
            device: 0,
            committed_bytes: 0,
            va,
            src: va,
            in_gpu_view: false,
            paged: false,
            tex: Arc::clone(&tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending: false,
            gpu_written: false,
            written_call: 0,
            gpu_writers: Vec::new(),
            cpu_backed: true,
            placed: false,
            unmapped: false,
            generation: 0,
            known: Default::default(),
            target: false,
        };
        let e_gpu = Entry {
            device: 0,
            committed_bytes: 0,
            va: va + size,
            src: va,
            in_gpu_view: false,
            paged: false,
            tex: Arc::clone(&tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending: false,
            gpu_written: true,
            written_call: 0,
            gpu_writers: Vec::new(),
            cpu_backed: true,
            placed: false,
            unmapped: false,
            generation: 0,
            known: Default::default(),
            target: false,
        };
        {
            let mut t = TEXTURES.lock().unwrap();
            let (d1, _) = t.insert(res_id, e_stream);
            let (d2, _) = t.insert(gpu_res_id, e_gpu);
            adjust_pending(d1 + d2);
        }
        let fp0 = unsafe { sample_fingerprint(va, size) };
        {
            let mut g = STREAM_RECHECK.lock().unwrap();
            g.insert(res_id, StreamRecheck { base_frame: 100, stage: 0, last_fp: Some(fp0) });
            g.insert(gpu_res_id, StreamRecheck { base_frame: 100, stage: 0, last_fp: Some(fp0) });
        }

        // Before first milestone (+4 frames -> frame 104), nothing is checked or re-queued.
        backing[size - 16] = 0xAB;
        unsafe { recheck_streamed_unwatched_at(&[], 103) };
        assert!(!TEXTURES.lock().unwrap().by_res.get(&res_id).unwrap().pending);

        // At milestone 0 (frame 104), late write is detected: pending becomes true, schedule resets
        // to base_frame=104, stage=0, and the gpu_written entry is pruned without re-queuing.
        unsafe { recheck_streamed_unwatched_at(&[], 104) };
        {
            let mut t = TEXTURES.lock().unwrap();
            let e = t.by_res.get_mut(&res_id).unwrap();
            assert!(e.pending, "late write to unwatched texture must re-queue upload at milestone 0");
            e.pending = false;
            PENDING.fetch_sub(1, Ordering::Relaxed);
            assert!(!t.by_res.get(&gpu_res_id).unwrap().pending, "gpu_written texture must not be re-queued");
        }
        {
            let g = STREAM_RECHECK.lock().unwrap();
            assert!(!g.contains_key(&gpu_res_id), "gpu_written texture must be pruned from STREAM_RECHECK");
            let st = g.get(&res_id).copied().expect("schedule reset after change");
            assert_eq!((st.base_frame, st.stage), (104, 0));
            assert_ne!(st.last_fp, Some(fp0));
        }

        // Advance through all 5 milestones with no further writes: entry must not re-queue and must
        // clean itself up after milestone 4 (+400 frames -> frame 504).
        for f in [108, 116, 144, 224, 504] {
            unsafe { recheck_streamed_unwatched_at(&[], f) };
            assert!(!TEXTURES.lock().unwrap().by_res.get(&res_id).unwrap().pending, "unchanged texture must not re-queue at frame {f}");
        }
        assert!(!STREAM_RECHECK.lock().unwrap().contains_key(&res_id), "entry must retire after final 400-frame milestone");

        forget(res_id);
        forget(gpu_res_id);
    }

    /// A partial upload does not replace the baseline of the last whole upload: an unwatched write
    /// it did not send is still caught at the milestone.
    #[test]
    fn stream_recheck_keeps_the_whole_upload_baseline_over_partial_uploads() {
        let _guard = TEST_LOCK.lock().unwrap();
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            // (software adapter only: the resource only gives the Due its key)
            let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 0x1_0000,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, None, &mut res).unwrap();
            let res = res.unwrap();
            let res_id = res.as_raw() as usize;

            let desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB RGBA8
            let tex = Arc::new(xg::Texture::new(&desc).expect("layout"));
            let size = tex.size() as usize;
            let mut backing = vec![0u8; size];
            let va = backing.as_ptr() as usize;
            let entry = Entry {
                device: 0,
                va,
                src: va,
                in_gpu_view: false,
                paged: false,
                tex: Arc::clone(&tex),
                states: vec![D3D12_RESOURCE_STATE_COMMON],
                pending: false,
                gpu_written: false,
                written_call: 0,
                placed: false,
                unmapped: false,
                generation: 0,
                known: Default::default(),
                target: false,
                committed_bytes: 0,
                gpu_writers: Vec::new(),
                cpu_backed: true,
            };
            let generation = {
                let mut t = TEXTURES.lock().unwrap();
                let (d, _) = t.insert(res_id, entry);
                adjust_pending(d);
                t.by_res[&res_id].generation
            };
            let due = |whole: bool| Due {
                res: res.clone(),
                device: 0,
                src: va,
                tex: Arc::clone(&tex),
                states: vec![D3D12_RESOURCE_STATE_COMMON],
                known: Default::default(),
                whole,
                again: texture_upload::Change::None,
                target: false,
                pages: None,
                generation,
            };

            // The whole upload at frame 0 sets the baseline.
            seed_stream_recheck(&[due(true)], &[], 0);
            let fp0 = sample_fingerprint(va, size);
            assert_eq!(STREAM_RECHECK.lock().unwrap().get(&res_id).copied(), Some(StreamRecheck { base_frame: 0, stage: 0, last_fp: Some(fp0) }));

            // An unwatched page is written, then a partial upload (of other pages) runs at frame 2.
            backing[size - 16] = 0xCD;
            seed_stream_recheck(&[due(false)], &[], 2);
            assert_eq!(
                STREAM_RECHECK.lock().unwrap().get(&res_id).copied(),
                Some(StreamRecheck { base_frame: 0, stage: 0, last_fp: Some(fp0) }),
                "a partial upload must not replace the baseline or move the schedule"
            );

            // At frame 6 milestone 0 is due from either base frame, and it finds what the GPU does
            // not hold yet.
            recheck_streamed_unwatched_at(&[], 6);
            {
                let mut t = TEXTURES.lock().unwrap();
                let e = t.by_res.get_mut(&res_id).unwrap();
                assert!(e.pending, "the unwatched write the partial upload did not send must queue the texture");
                e.pending = false;
                PENDING.fetch_sub(1, Ordering::Relaxed);
            }

            forget(res_id);
        }
    }

    /// A partial upload of a texture whose recheck entry has retired makes an entry without a
    /// baseline: its first milestone queues a whole upload, which sets the baseline.
    #[test]
    fn stream_recheck_partial_upload_without_an_entry_queues_a_whole_upload() {
        let _guard = TEST_LOCK.lock().unwrap();
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            // (software adapter only: the resource only gives the Due its key)
            let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 0x1_0000,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, None, &mut res).unwrap();
            let res = res.unwrap();
            let res_id = res.as_raw() as usize;

            let desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB RGBA8
            let tex = Arc::new(xg::Texture::new(&desc).expect("layout"));
            let size = tex.size() as usize;
            let mut backing = vec![0u8; size];
            let va = backing.as_ptr() as usize;
            let entry = Entry {
                device: 0,
                va,
                src: va,
                in_gpu_view: false,
                paged: false,
                tex: Arc::clone(&tex),
                states: vec![D3D12_RESOURCE_STATE_COMMON],
                pending: false,
                gpu_written: false,
                written_call: 0,
                placed: false,
                unmapped: false,
                generation: 0,
                known: Default::default(),
                target: false,
                committed_bytes: 0,
                gpu_writers: Vec::new(),
                cpu_backed: true,
            };
            let generation = {
                let mut t = TEXTURES.lock().unwrap();
                let (d, _) = t.insert(res_id, entry);
                adjust_pending(d);
                t.by_res[&res_id].generation
            };
            let due = |whole: bool| Due {
                res: res.clone(),
                device: 0,
                src: va,
                tex: Arc::clone(&tex),
                states: vec![D3D12_RESOURCE_STATE_COMMON],
                known: Default::default(),
                whole,
                again: texture_upload::Change::None,
                target: false,
                pages: None,
                generation,
            };
            let pending = || TEXTURES.lock().unwrap().by_res.get(&res_id).unwrap().pending;
            let entry = || STREAM_RECHECK.lock().unwrap().get(&res_id).copied();

            // A whole upload at frame 0, then every milestone finds nothing new and the entry retires.
            seed_stream_recheck(&[due(true)], &[], 0);
            for f in [4, 16, 56, 176, 576] {
                recheck_streamed_unwatched_at(&[], f);
                assert!(!pending(), "unchanged texture must not re-queue at frame {f}");
            }
            assert_eq!(entry(), None, "entry must retire after final 400-frame milestone");

            // An unwatched page is written, a partial upload (of other pages) runs at frame 600, and
            // another unwatched page is written after it.
            backing[16] = 0x11;
            seed_stream_recheck(&[due(false)], &[], 600);
            assert_eq!(entry(), Some(StreamRecheck { base_frame: 600, stage: 0, last_fp: None }), "a partial upload without an entry makes one without a baseline");
            backing[size - 16] = 0x22;
            // A later partial upload leaves that entry as it is.
            seed_stream_recheck(&[due(false)], &[], 602);
            assert_eq!(entry(), Some(StreamRecheck { base_frame: 600, stage: 0, last_fp: None }), "a partial upload must not touch an existing entry");

            recheck_streamed_unwatched_at(&[], 603);
            assert!(!pending(), "nothing is checked before the first milestone");
            recheck_streamed_unwatched_at(&[], 604);
            {
                let mut t = TEXTURES.lock().unwrap();
                let e = t.by_res.get_mut(&res_id).unwrap();
                assert!(e.pending, "the first milestone of an entry without a baseline must queue the texture");
                e.pending = false;
                PENDING.fetch_sub(1, Ordering::Relaxed);
            }

            // The whole upload that follows sets the baseline: with no further write, the next
            // milestone queues nothing.
            seed_stream_recheck(&[due(true)], &[], 605);
            let fp = sample_fingerprint(va, size);
            assert_eq!(entry(), Some(StreamRecheck { base_frame: 605, stage: 0, last_fp: Some(fp) }));
            recheck_streamed_unwatched_at(&[], 609);
            assert!(!pending(), "after the whole upload an unchanged texture must not re-queue");
            assert_eq!(entry(), Some(StreamRecheck { base_frame: 605, stage: 1, last_fp: Some(fp) }));

            forget(res_id);
        }
    }

    /// Real GPU, `vram::level_pages` (`XCL_TEXTURE_LEVEL_PAGES=1`): a sampled texture placed on the
    /// shared pages gets the pages of a level when an upload has texels for it. Its memory has
    /// every level but the largest at first, as a title streams a texture: no page before the
    /// first upload, then pages for those levels only and their texels right, zeros where there
    /// are none. With memory under the largest level too, a texture placed there again (as a
    /// title does) takes that level's pages - the others are shared - and holds all of it. A
    /// texture the GPU writes gets the pages of all its levels. When the title takes the memory
    /// from under the largest level again, that level gives its pages back at the next upload
    /// (the other texture on them keeps them), and gets them again with its texels. A small level
    /// that shares its pages with levels that keep texels clears with the next upload.
    #[test]
    fn a_placed_texture_gets_the_pages_of_a_level_with_its_texels() {
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap();
        let _guard = TEST_LOCK.lock().unwrap();
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const NAME: &str = "a_placed_texture_gets_the_pages_of_a_level_with_its_texels";
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return println!("SKIPPED {NAME}: no adapter") };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return println!("SKIPPED {NAME}: software adapter");
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev else { return println!("SKIPPED {NAME}: no D3D12 device") };
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32).is_err()
                || opts.ResourceHeapTier.0 < 2
                || opts.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0
            {
                return println!("SKIPPED {NAME}: tiled tier < 2");
            }
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            // 1024x1024 RGBA8, 11 levels, 64 KiB blocks: 5.4 MiB on the console
            let d = console_desc(3, 1024, 1024, 1, 11, 28, 0x109);
            let place_tex = |va: usize| -> ID3D12Resource {
                let mut p = std::ptr::null_mut();
                let hr = super::super::device_x::create_placed_resource_x(dev.as_raw(), va as u64, d.as_ptr(), D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p);
                assert!(hr >= 0 && !p.is_null(), "CreatePlacedResourceX failed: {hr:#x}");
                ID3D12Resource::from_raw(p)
            };
            let read_sub = |r: &ID3D12Resource, sub: u32| -> Vec<u8> {
                let rd = r.GetDesc();
                let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
                dev.GetCopyableFootprints(&rd, sub, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
                let mut rb: Option<ID3D12Resource> = None;
                let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
                let rb = rb.unwrap();
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.Close().unwrap();
                own_work(|| {
                    super::super::vram::before_execute(&queue);
                    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                    texture::wait(&dev, &queue).unwrap();
                });
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).unwrap();
                let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
                for y in 0..rows as usize {
                    out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(y * fp.Footprint.RowPitch as usize), row_bytes as usize));
                }
                rb.Unmap(0, None);
                out
            };
            let tex = xg::Texture::new(&d).unwrap();
            let surface = tex.texels().unwrap();
            let (size, chain) = (tex.size() as usize, &surface.chain);
            // the whole texture as the title will have it (no byte of a level zero), and what its
            // levels then hold
            let mut whole = vec![0u8; size];
            for (m, l) in chain.levels.iter().enumerate() {
                let pat: Vec<u8> = (0..l.width * l.height * surface.bpe as usize).map(|k| 1 + ((m * 7 + k * 13) % 251) as u8).collect();
                chain.tile_level(&mut whole[..chain.slice_bytes], m, 0, &pat);
            }
            let want = subresources(whole.as_ptr() as usize, &tex);
            assert_eq!(want.len(), 11);
            // its place in the title's memory: everything below mip 0 is there (the levels follow
            // each other from the smallest to mip 0, `tiling::Chain`)
            let at = VirtualAlloc(std::ptr::null(), size.next_multiple_of(0x1_0000), MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(at, 0);
            let top = chain.levels[0].offset;
            assert!(top > 0 && top % 0x1000 == 0 && top < size, "mip 0 is the last part of the slice: from {top:#x} of {size:#x}");
            assert!(!VirtualAlloc(at as *const c_void, top, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::copy_nonoverlapping(whole.as_ptr(), at as *mut u8, top);

            super::super::vram::LEVEL_PAGES_FOR_TEST.store(true, Ordering::Relaxed);
            let dev_key = dev.as_raw() as usize;
            let dev_pages = || super::super::vram::page_bytes_for(dev_key).0;
            let p0 = dev_pages();
            let small = place_tex(at);
            super::super::vram::LEVEL_PAGES_FOR_TEST.store(false, Ordering::Relaxed);
            if !super::super::vram::has_lazy_levels(small.as_raw() as usize) {
                // (not reserved, or a tiling that does not add up: it has its pages as before)
                drop(small);
                VirtualFree(at as *mut c_void, 0, MEM_RELEASE);
                return println!("SKIPPED {NAME}: this adapter's texture was not placed with lazy levels");
            }
            super::super::vram::LEVEL_PAGES_FOR_TEST.store(true, Ordering::Relaxed);
            assert_eq!(dev_pages(), p0, "no pages before the first upload");
            // the tiles of mip 0 and of all of it, as this driver lays them out
            let (mut all, mut n, mut mip0) = (0u32, 1u32, D3D12_SUBRESOURCE_TILING::default());
            dev.GetResourceTiling(&small, Some(&mut all), None, None, Some(&mut n), 0, &mut mip0);
            let top_tiles = mip0.WidthInTiles * mip0.HeightInTiles as u32 * mip0.DepthInTiles as u32;
            assert!(n == 1 && top_tiles > 0 && top_tiles < all, "mip 0 has tiles of its own: {top_tiles} of {all}");
            refresh(queue.as_raw());
            assert_eq!(dev_pages() - p0, ((all - top_tiles) as usize) << 16, "pages for the levels with texels, none for mip 0");
            for sub in 1..11u32 {
                assert!(read_sub(&small, sub) == want[sub as usize].data, "level {sub} after the first upload");
            }
            assert!(read_sub(&small, 0).iter().all(|&b| b == 0), "a level without pages reads as zeros");
            // (what its upload read is not kept: mip 0 may have been written while it was read)
            assert!(super::super::vram::has_unmapped_levels(small.as_raw() as usize));
            // the title maps memory under mip 0 and fills it
            assert!(!VirtualAlloc((at + top) as *const c_void, size - top, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::copy_nonoverlapping(whole.as_ptr().add(top), (at + top) as *mut u8, size - top);
            let full = place_tex(at);
            refresh(queue.as_raw());
            assert_eq!(dev_pages() - p0, (all as usize) << 16, "mip 0's pages come with its texels; the other levels' are shared");
            assert!(!super::super::vram::has_unmapped_levels(full.as_raw() as usize));
            for sub in 0..11u32 {
                assert!(read_sub(&full, sub) == want[sub as usize].data, "level {sub} once all of it is there");
            }
            // what the GPU writes needs pages under all of the texture: the first one's mip 0 gets
            // the pages of its address, which hold what the second one's upload put there
            let write = capture_gpu_write(small.as_raw() as usize, ALL).unwrap();
            publish_gpu_writes(WriterQueueId { raw: queue.as_raw() as usize, generation: 1 }, &[write]);
            assert_eq!(dev_pages() - p0, (all as usize) << 16, "the same pages");
            assert!(read_sub(&small, 0) == want[0].data, "a GPU-written texture has pages under all its levels");
            // the title takes the memory from under mip 0 again (a streamed level it let go): the
            // second texture's next upload finds mip 0 all zeros, and mip 0 gives back its pages;
            // the first one, on the same pages, keeps them
            let full_key = full.as_raw() as usize;
            assert_eq!(super::super::vram::mapped_level_tiles(full_key), Some(all as u64));
            assert_ne!(VirtualFree((at + top) as *mut c_void, size - top, MEM_DECOMMIT), 0);
            assert!(map_levels_with_texels(&full, &subresources(at, &tex)).is_empty(), "no level got pages");
            assert_eq!(super::super::vram::mapped_level_tiles(full_key), Some((all - top_tiles) as u64), "mip 0 gave its pages back");
            assert!(super::super::vram::has_unmapped_levels(full_key));
            assert!(read_sub(&full, 0).iter().all(|&b| b == 0), "a level that gave its pages back reads as zeros");
            for sub in 1..11u32 {
                assert!(read_sub(&full, sub) == want[sub as usize].data, "level {sub} keeps its pages and texels");
            }
            assert!(read_sub(&small, 0) == want[0].data, "the other texture on those pages keeps them");
            // mip 0's texels come back: its pages too, the ones the other texture holds
            assert!(!VirtualAlloc((at + top) as *const c_void, size - top, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::copy_nonoverlapping(whole.as_ptr().add(top), (at + top) as *mut u8, size - top);
            assert_eq!(map_levels_with_texels(&full, &subresources(at, &tex)), vec![0], "mip 0 got its pages again");
            assert_eq!(super::super::vram::mapped_level_tiles(full_key), Some(all as u64));
            assert!(read_sub(&full, 0) == want[0].data, "mip 0 on the pages the other texture kept");
            // a small level whose pages hold levels that keep texels: zeros there clear it at the
            // next upload, and its neighbours keep their texels
            let cpu = std::slice::from_raw_parts_mut(at as *mut u8, chain.slice_bytes);
            chain.tile_level(cpu, 10, 0, &vec![0u8; want[10].data.len()]);
            // The synthetic allocation here is not write-watched. Queue the normal batch update.
            {
                let mut t = TEXTURES.lock().unwrap();
                let key = full.as_raw() as usize;
                let e = t.by_res.get_mut(&key).unwrap();
                assert!(!e.pending, "the preceding refresh uploaded this texture");
                e.pending = true;
                t.pending_res.insert(key);
                adjust_pending(1);
            }
            refresh(queue.as_raw());
            assert!(read_sub(&full, 10).iter().all(|&b| b == 0), "a small level sharing the zero span was cleared");
            assert!(read_sub(&full, 0) == want[0].data, "mip 0 kept its texels");
            assert!(read_sub(&full, 1) == want[1].data, "a nonzero mip kept its texels");
            assert!(read_sub(&full, 9) == want[9].data, "the small level's neighbour kept its texels");
            super::super::vram::LEVEL_PAGES_FOR_TEST.store(false, Ordering::Relaxed);
            drop((small, full));
            VirtualFree(at as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `read_committed` copies committed pages and leaves uncommitted gaps zeroed, including
    /// unaligned start/end offsets.
    #[test]
    fn read_committed_copies_committed_pages_and_zeroes_gaps() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGE: usize = 0x10000;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert_ne!(VirtualAlloc(base as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            assert_ne!(VirtualAlloc((base + 2 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            std::ptr::write_bytes(base as *mut u8, 0x3A, PAGE);
            std::ptr::write_bytes((base + 2 * PAGE) as *mut u8, 0x7C, PAGE);

            let start = base + 0x0B00;
            let len = 2 * PAGE + 0x0500;
            let got = read_committed(start, len);
            assert_eq!(got.len(), len);
            assert!(got[..PAGE - 0x0B00].iter().all(|&b| b == 0x3A));
            assert!(got[PAGE - 0x0B00..2 * PAGE - 0x0B00].iter().all(|&b| b == 0));
            assert!(got[2 * PAGE - 0x0B00..].iter().all(|&b| b == 0x7C));

            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `read_committed` must not fault with 0xC0000005 when another thread concurrently
    /// decommits/recommits pages between `VirtualQuery` and the copy, and surrounding committed
    /// pages must remain intact.
    #[test]
    fn read_committed_tolerates_concurrent_decommit() {
        use std::sync::atomic::AtomicBool;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGE: usize = 0x10000;
        const PAGES: usize = 16;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert_ne!(VirtualAlloc(base as *const c_void, PAGES * PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            // Keep first and last 64 KiB pages permanently committed with known sentinels; race the middle.
            std::ptr::write_bytes(base as *mut u8, 0x5A, PAGE);
            std::ptr::write_bytes((base + (PAGES - 1) * PAGE) as *mut u8, 0xA5, PAGE);

            let stop = Arc::new(AtomicBool::new(false));
            let stop_w = Arc::clone(&stop);
            let mutator = std::thread::spawn(move || {
                let mut i = 0usize;
                while !stop_w.load(Ordering::Relaxed) {
                    let slot = 1 + (i % (PAGES - 2));
                    let p = (base + slot * PAGE) as *const c_void;
                    let _ = VirtualFree(p as *mut c_void, PAGE, MEM_DECOMMIT);
                    let q = VirtualAlloc(p, PAGE, MEM_COMMIT, PAGE_READWRITE);
                    if !q.is_null() {
                        std::ptr::write_bytes(q as *mut u8, 0x3C, 4096);
                    }
                    i = i.wrapping_add(1);
                }
            });

            // Unaligned start/end across the racing region (mirrors the streaming pool layout).
            let off = 0x0B00;
            let len = (PAGES - 1) * PAGE + 0x0500;
            for _ in 0..400 {
                let out = read_committed(base + off, len);
                assert_eq!(out.len(), len);
                assert!(out[..PAGE - off].iter().all(|&b| b == 0x5A));
                assert!(out[len - (off + 0x0500)..].iter().all(|&b| b == 0xA5));
            }

            stop.store(true, Ordering::Relaxed);
            mutator.join().unwrap();
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn batch_fits_caps_items_bytes_and_device() {
        // Empty batch always accepts its first upload, even when larger than BATCH_BYTES.
        assert!(batch_fits(0, 0, 0, 1, BATCH_BYTES + 4096));
        // Once an oversized texture is in the batch, the next upload starts a new batch.
        assert!(!batch_fits(1, BATCH_BYTES + 4096, 1, 1, 4096));

        // Many tiny textures split at BATCH_ITEMS even when well below BATCH_BYTES.
        assert!(batch_fits(BATCH_ITEMS - 1, (BATCH_ITEMS - 1) * 4096, 1, 1, 4096));
        assert!(!batch_fits(BATCH_ITEMS, BATCH_ITEMS * 4096, 1, 1, 4096));

        // Byte cap accepts exact BATCH_BYTES, splits past it, and does not overflow.
        assert!(batch_fits(1, BATCH_BYTES - 4096, 1, 1, 4096));
        assert!(!batch_fits(1, BATCH_BYTES - 4096, 1, 1, 4097));
        assert!(!batch_fits(1, usize::MAX - 10, 1, 1, 100));

        // Different device always starts a new batch.
        assert!(!batch_fits(1, 4096, 1, 2, 4096));
    }

    /// Hashing the title's bytes must not fault with 0xC0000005 when another thread decommits
    /// and commits pages of them meanwhile (memory outside the coherent window is not read in
    /// place).
    #[test]
    fn title_bytes_survive_concurrent_decommit() {
        use std::sync::atomic::AtomicBool;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGE: usize = 0x10000;
        const PAGES: usize = 16;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert_ne!(VirtualAlloc(base as *const c_void, PAGES * PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            let stop = Arc::new(AtomicBool::new(false));
            let stop_w = Arc::clone(&stop);
            // decommits made while the hashing ran (none: the test would prove nothing)
            let decommits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let decommits_w = Arc::clone(&decommits);
            // each 4 KiB of the title's bytes holds its own number: a read sees it, or zeros where
            // a page was decommitted (a recommitted page reads as zeros; a page decommitted while
            // it was copied, its number then zeros)
            let tag = |k: usize| 1 + (k % 255) as u8;
            for k in 0..PAGES * PAGE / 4096 {
                std::ptr::write_bytes((base + k * 4096) as *mut u8, tag(k), 4096);
            }
            let mutator = std::thread::spawn(move || {
                let mut i = 0usize;
                while !stop_w.load(Ordering::Relaxed) {
                    let p = (base + (1 + i % (PAGES - 2)) * PAGE) as *mut c_void;
                    let freed = VirtualFree(p, PAGE, MEM_DECOMMIT) != 0;
                    decommits_w.fetch_add(1, Ordering::Relaxed);
                    assert!(freed, "decommit");
                    assert!(!VirtualAlloc(p, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null(), "recommit");
                    i = i.wrapping_add(1);
                }
            });
            // (the mutator is running before the first hash)
            while decommits.load(Ordering::Relaxed) == 0 && !mutator.is_finished() {
                std::thread::yield_now();
            }
            let before = decommits.load(Ordering::Relaxed);
            for _ in 0..2000 {
                let ok = with_title_bytes(base, PAGES * PAGE, |b| {
                    std::hint::black_box(texture_upload::content_hash(b));
                    b.chunks(4096).enumerate().all(|(k, c)| {
                        let n = c.iter().position(|&x| x != tag(k)).unwrap_or(c.len());
                        c[n..].iter().all(|&x| x == 0)
                    })
                });
                if !ok {
                    stop.store(true, Ordering::Relaxed);
                    let _ = mutator.join();
                    panic!("a read returned bytes that were neither the title's nor zeros");
                }
            }
            let during = decommits.load(Ordering::Relaxed) - before;
            stop.store(true, Ordering::Relaxed);
            mutator.join().unwrap();
            assert!(during > 0, "no page was decommitted while the bytes were hashed");
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A guard or no-access page in the title's bytes reads as zeros, and the guard stays (a
    /// title's guard page is its own: reading must not use it up).
    #[test]
    fn title_bytes_leave_guard_pages_alone() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE};
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 << 12, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            std::ptr::write_bytes(base as *mut u8, 0x77, 3 << 12);
            let mut old = 0;
            assert_ne!(VirtualProtect((base + (1 << 12)) as *const c_void, 1 << 12, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
            assert_ne!(VirtualProtect((base + (2 << 12)) as *const c_void, 1 << 12, PAGE_NOACCESS, &mut old), 0);
            with_title_bytes(base, 3 << 12, |b| {
                assert!(b[..1 << 12].iter().all(|&x| x == 0x77));
                assert!(b[1 << 12..].iter().all(|&x| x == 0), "guard and no-access pages read as zeros");
            });
            let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery((base + (1 << 12)) as *const c_void, &mut m, std::mem::size_of_val(&m));
            assert_ne!(m.Protect & PAGE_GUARD, 0, "the guard page is still a guard page");
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn title_queue_upload_states_are_leased_and_serialized_per_queue() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (first_key, second_key) = (0x7ffe_7100usize, 0x7ffe_7200usize);
        ON_QUEUE.lock().unwrap().remove(&first_key);
        ON_QUEUE.lock().unwrap().remove(&second_key);

        let first = queue_upload_state(first_key);
        let second = queue_upload_state(second_key);
        let first_guard = first.lock().unwrap();
        assert!(ON_QUEUE.try_lock().is_ok(), "one queue must not hold the upload registry");
        assert!(second.try_lock().is_ok(), "one queue must not serialize another queue's uploads");

        let removed = ON_QUEUE.lock().unwrap().remove(&first_key).unwrap();
        assert!(Arc::ptr_eq(&first, &removed), "registry removal returns the leased generation");
        let replacement = queue_upload_state(first_key);
        assert!(!Arc::ptr_eq(&first, &replacement), "raw-address reuse gets a fresh queue-local state");

        drop(first_guard);
        drop(removed);
        ON_QUEUE.lock().unwrap().remove(&first_key);
        ON_QUEUE.lock().unwrap().remove(&second_key);
    }
}
