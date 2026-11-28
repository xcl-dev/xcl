//! Textures a title places at one address in two formats: written through one, read through
//! the other.
//!
//! A retail title packs light as R9G9B9E5 into R32_UINT textures (its compute programs cannot
//! store R9G9B9E5) and samples the same memory through an R9G9B9E5 texture it places at the same
//! address: its sky, its light probes, the bounced light its lighting pass adds. On the console
//! the two are one memory. Here the written one is GPU-owned (`vram`) and the sampled one is a
//! texture of its own, kept in step with the title's memory (`title_textures`), which nothing
//! writes: what is read through it is stale.
//!
//! This module keeps the title's placed textures by address. Where a sampled one lies in the
//! memory of a written one of the same size, levels, layers and texel width and of another
//! format, the written one's texels are copied into it on the GPU, through a buffer (D3D12
//! copies between two formats of one texel width in no other way). The copy is recorded where
//! the title says the written one is written no more for now:
//!
//! * its enhanced barrier of the written texture (console list #146) whose access before is
//!   render target or unordered access. Only that is read of the barrier: XCL does not move its
//!   PC textures by enhanced barriers, so the two are in the states the legacy barriers left;
//! * its legacy transition of the written texture out of RENDER_TARGET / UNORDERED_ACCESS.
//!
//! Behind `XCL_ALIAS_COPY=1`; with the switch off nothing is kept and every hook stops at one
//! atomic load. (After the format-aliased copies of xcl-dev/xcl#1989 by raylitebull: the pairs
//! by address and shape, and the copy through a buffer.)

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// RENDER_TARGET | UNORDERED_ACCESS
const WRITTEN: u32 = 0x4 | 0x8;
const COPY_DEST: u32 = 0x400;
const COPY_SOURCE: u32 = 0x800;
/// `Tex::now`: a transition named a part of the texture only.
const STATE_UNKNOWN: u32 = u32::MAX;
const BEGIN_ONLY: u32 = 1;
const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

/// `XCL_ALIAS_COPY=1` (read once).
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("XCL_ALIAS_COPY").is_some_and(|v| v == "1");
        if on {
            crate::xlog_info!("D3D12.X: alias copies are on (XCL_ALIAS_COPY=1): a sampled texture placed in the memory of a written one of another format is copied from it where the title stops writing that one");
        }
        on
    })
}

/// What the two textures of a pair have in common.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Shape {
    dimension: i32,
    width: u64,
    height: u32,
    depth: u16,
    mips: u16,
    /// bytes a texel
    texel: u32,
}

struct Tex {
    dev: usize,
    va: usize,
    shape: Shape,
    /// its PC format
    format: u32,
    /// a render target or a UAV texture: what it holds is what the GPU wrote
    written: bool,
    desc: D3D12_RESOURCE_DESC,
    /// the state the title's legacy transitions left it in
    now: u32,
}

/// The buffer a pair's texels go through, and each level and layer as each of the two formats
/// lays it out for a copy (the written one's first).
struct Through {
    buffer: ID3D12Resource,
    parts: Vec<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>,
}

#[derive(Default)]
struct Table {
    by_res: HashMap<usize, Tex>,
    /// the textures at an address
    at: HashMap<usize, Vec<usize>>,
    /// by (written, sampled); None: the pair cannot be copied
    through: HashMap<(usize, usize), Option<Arc<Through>>>,
    /// buffers of pairs that went: kept, as a list in flight may still name them
    gone: Vec<Arc<Through>>,
}

// D3D12 objects are free-threaded
unsafe impl Send for Table {}

static TABLE: Mutex<Option<Table>> = Mutex::new(None);
/// Textures kept: the gate of `plan` and `plan_enhanced`.
static KEPT: AtomicUsize = AtomicUsize::new(0);
static PAIRS: AtomicU64 = AtomicU64::new(0);
/// Copies recorded: asked for by an enhanced barrier, by a legacy transition.
static COPIES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Copies left out: a state not known, a list that is neither DIRECT nor COMPUTE, a pair that
/// cannot be copied.
static LEFT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Bytes a texel of the PC format `f` (None: block-compressed, depth, video or unknown).
fn texel_bytes(f: u32) -> Option<u32> {
    Some(match f {
        1..=4 => 16,
        5..=8 => 12,
        9..=18 => 8,
        23..=43 | 67 | 87..=93 => 4,
        48..=59 | 85 | 86 | 115 => 2,
        60..=65 => 1,
        _ => return None,
    })
}

fn subresources(d: &D3D12_RESOURCE_DESC) -> u32 {
    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { d.DepthOrArraySize.max(1) as u32 };
    d.MipLevels.max(1) as u32 * layers
}

/// The title placed the texture `res` (PC description at `pc`, 0x38 bytes) at `va` in `state`
/// (`device_x::create_placed_resource_x`).
pub(super) unsafe fn note_created(dev: *mut c_void, res: *mut c_void, va: usize, pc: *const u8, state: u32) {
    if on() {
        keep(dev as usize, res as usize, va, &std::ptr::read_unaligned(pc as *const D3D12_RESOURCE_DESC), state);
    }
}

/// `note_created` without the switch.
fn keep(dev: usize, res: usize, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32) {
    let flags = desc.Flags.0 as u32;
    if res == 0 || va == 0 || desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || desc.SampleDesc.Count > 1 || flags & 2 != 0 {
        return;
    }
    let format = desc.Format.0 as u32;
    let Some(texel) = texel_bytes(format) else { return };
    let shape = Shape { dimension: desc.Dimension.0, width: desc.Width, height: desc.Height, depth: desc.DepthOrArraySize.max(1), mips: desc.MipLevels.max(1), texel };
    let written = flags & 5 != 0;
    let mut g = TABLE.lock().unwrap();
    let t = g.get_or_insert_with(Table::default);
    if t.by_res.insert(res, Tex { dev, va, shape, format, written, desc: *desc, now: state }).is_none() {
        KEPT.fetch_add(1, Ordering::Relaxed);
    }
    let here = t.at.entry(va).or_default();
    if !here.contains(&res) {
        here.push(res);
    }
    let other = here.iter().filter(|&&o| o != res).filter_map(|o| t.by_res.get(o)).find(|o| o.shape == shape && o.written != written && o.format != format).map(|o| o.format);
    if let Some(other) = other {
        let n = PAIRS.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 8 || n.is_power_of_two() {
            let (w, r) = if written { (format, other) } else { (other, format) };
            crate::xlog_info!(
                "D3D12.X: alias copy: at {va:#x} a {}x{}x{} texture of {} levels is written as format {w} and read as format {r} ({n} such pairs so far)",
                shape.width,
                shape.height,
                shape.depth,
                shape.mips
            );
        }
    }
}

/// The title's last reference to `res` went.
pub(super) fn forget(res: usize) {
    if KEPT.load(Ordering::Relaxed) == 0 {
        return;
    }
    let mut g = TABLE.lock().unwrap();
    let Some(t) = g.as_mut() else { return };
    let Some(x) = t.by_res.remove(&res) else { return };
    KEPT.fetch_sub(1, Ordering::Relaxed);
    if let Some(v) = t.at.get_mut(&x.va) {
        v.retain(|&r| r != res);
        if v.is_empty() {
            t.at.remove(&x.va);
        }
    }
    let keys: Vec<(usize, usize)> = t.through.keys().filter(|k| k.0 == res || k.1 == res).copied().collect();
    for k in keys {
        if let Some(Some(th)) = t.through.remove(&k) {
            t.gone.push(th);
        }
    }
}

/// The sampled textures in the memory of the written texture `res`.
fn sampled_of(t: &Table, res: usize) -> Vec<usize> {
    let Some(x) = t.by_res.get(&res).filter(|x| x.written) else { return Vec::new() };
    t.at.get(&x.va).map_or(Vec::new(), |v| v.iter().copied().filter(|o| t.by_res.get(o).is_some_and(|y| !y.written && y.shape == x.shape && y.format != x.format)).collect())
}

/// A copy to record: (the written texture, the sampled one, what asked for it: `COPIES`).
pub(super) type Job = (usize, usize, usize);

/// One copy of a pair for a call.
fn once_each(mut jobs: Vec<Job>) -> Vec<Job> {
    let mut seen: Vec<(usize, usize)> = Vec::new();
    jobs.retain(|j| {
        let new = !seen.contains(&(j.0, j.1));
        seen.push((j.0, j.1));
        new
    });
    jobs
}

/// The title's legacy barriers `barriers` (console words; `pc_states`: the state bits the PC
/// knows): the states they leave the kept textures in are kept, and a written texture's
/// transition out of RENDER_TARGET / UNORDERED_ACCESS asks for the copies into its sampled
/// ones, recorded after the barriers (`record`).
pub(super) fn plan(barriers: &[[u32; 8]], pc_states: u32) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    if KEPT.load(Ordering::Relaxed) == 0 {
        return jobs;
    }
    let mut g = TABLE.lock().unwrap();
    let Some(t) = g.as_mut() else { return jobs };
    // (transitions; the first half of a split one counts at its end)
    for b in barriers.iter().filter(|b| b[0] == 0 && b[1] & BEGIN_ONLY == 0) {
        let (res, sub, before, after) = ((b[2] as u64 | (b[3] as u64) << 32) as usize, b[4], b[5] & pc_states, b[6] & pc_states);
        let Some(x) = t.by_res.get_mut(&res) else { continue };
        x.now = if sub == ALL || subresources(&x.desc) == 1 { after } else { STATE_UNKNOWN };
        if before & WRITTEN != 0 && after & WRITTEN == 0 {
            jobs.extend(sampled_of(t, res).into_iter().map(|r| (res, r, 1)));
        }
    }
    once_each(jobs)
}

/// The title's enhanced barriers (console list #146, `Barrier(n, groups)`: D3D12_BARRIER_GROUPs
/// as the PC has them, 16 bytes a group, 0x40 a texture barrier: sync before and after, access
/// before and after, layout before and after, the resource at +0x18). A barrier of a written
/// texture whose access before is render target (0x8) or unordered access (0x10) asks for the
/// copies into its sampled ones, recorded after the call's UAV barrier (`record`).
pub(super) unsafe fn plan_enhanced(n: u32, groups: usize) -> Vec<Job> {
    const NO_ACCESS: u32 = 0x8000_0000;
    let mut jobs: Vec<Job> = Vec::new();
    let readable = |p: usize, len: usize| crate::gpu::shader::readable(p, len);
    if PAIRS.load(Ordering::Relaxed) == 0 || n == 0 || n > 1024 || groups == 0 || groups % 8 != 0 || !readable(groups, n as usize * 16) {
        return jobs;
    }
    let g = TABLE.lock().unwrap();
    let Some(t) = g.as_ref() else { return jobs };
    for k in 0..n as usize {
        let p = groups + k * 16;
        let (ty, count, at) = ((p as *const u32).read_unaligned(), ((p + 4) as *const u32).read_unaligned(), ((p + 8) as *const u64).read_unaligned() as usize);
        // (texture barriers)
        if ty != 1 || count == 0 || count > 1 << 16 || at == 0 || !readable(at, count as usize * 0x40) {
            continue;
        }
        for i in 0..count as usize {
            let mut w = [0u32; 16];
            std::ptr::copy_nonoverlapping((at + i * 0x40) as *const u8, w.as_mut_ptr() as *mut u8, 0x40);
            let (access_before, res) = (w[2], (w[6] as u64 | (w[7] as u64) << 32) as usize);
            if access_before & NO_ACCESS == 0 && access_before & (0x8 | 0x10) != 0 {
                jobs.extend(sampled_of(t, res).into_iter().map(|r| (res, r, 0)));
            }
        }
    }
    once_each(jobs)
}

/// The PC's own method `slot` of the device `dev` (past XCL's hooks when it is adopted: the
/// buffer is XCL's, not a title's).
unsafe fn dev_fn(dev: *mut c_void, slot: usize) -> usize {
    if super::real::is_adopted(dev) {
        super::real::orig_fn(dev, slot)
    } else {
        *(*(dev as *const *const usize)).add(slot)
    }
}

/// `Through` for a written texture of `from` and a sampled one of `to` on `dev`: None when the
/// two do not lay their levels and layers out alike for a copy.
unsafe fn make_through(dev: *mut c_void, from: &D3D12_RESOURCE_DESC, to: &D3D12_RESOURCE_DESC) -> Option<Through> {
    use windows_sys::core::GUID;
    // (every level of every layer is a part: an array of 64 layers with two levels has 128)
    let parts = subresources(from);
    if parts == 0 || parts != subresources(to) {
        return None;
    }
    type Footprints = unsafe extern "system" fn(*mut c_void, *const D3D12_RESOURCE_DESC, u32, u32, u64, *mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT, *mut u32, *mut u64, *mut u64);
    let footprints: Footprints = std::mem::transmute(dev_fn(dev, 38));
    let layout = |d: &D3D12_RESOURCE_DESC| {
        let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize];
        let mut bytes = 0u64;
        let d = D3D12_RESOURCE_DESC { Alignment: 0, ..*d };
        footprints(dev, &d, 0, parts, 0, rows.as_mut_ptr(), std::ptr::null_mut(), std::ptr::null_mut(), &mut bytes);
        (rows, bytes)
    };
    let ((of_from, bytes), (of_to, bytes_to)) = (layout(from), layout(to));
    let alike = of_from.iter().zip(&of_to).all(|(a, b)| {
        let (fa, fb) = (a.Footprint, b.Footprint);
        a.Offset == b.Offset && (fa.Width, fa.Height, fa.Depth, fa.RowPitch) == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
    });
    if bytes == 0 || bytes == u64::MAX || bytes != bytes_to || !alike {
        return None;
    }
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
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
    type Create = unsafe extern "system" fn(*mut c_void, *const D3D12_HEAP_PROPERTIES, i32, *const D3D12_RESOURCE_DESC, i32, *const c_void, *const GUID, *mut *mut c_void) -> i32;
    let create: Create = std::mem::transmute(dev_fn(dev, 27));
    let mut raw = std::ptr::null_mut();
    let hr = create(dev, &heap, 0, &desc, 0, std::ptr::null(), &ID3D12Resource::IID as *const _ as *const GUID, &mut raw);
    if hr < 0 || raw.is_null() {
        return None;
    }
    Some(Through { buffer: ID3D12Resource::from_raw(raw), parts: of_from.into_iter().zip(of_to).collect() })
}

/// The copies `jobs` (`plan`, `plan_enhanced`) on `list`, after the title's barriers: each
/// written texture's texels into its sampled one through the pair's buffer, with the moves to
/// the copy states and back to the states the title's legacy barriers have the two in. On a
/// DIRECT or COMPUTE list XCL keeps state for.
pub(super) unsafe fn record(list: *mut c_void, jobs: &[Job]) {
    if jobs.is_empty() {
        return;
    }
    if !matches!(super::list_x::tracked_list_type(list), Some(D3D12_COMMAND_LIST_TYPE_DIRECT) | Some(D3D12_COMMAND_LIST_TYPE_COMPUTE)) {
        LEFT[1].fetch_add(jobs.len() as u64, Ordering::Relaxed);
        return;
    }
    let tr = |res: usize, before: u32, after: u32| [0, 0, res as u32, (res as u64 >> 32) as u32, ALL, before, after, 0];
    for &(w, r, why) in jobs {
        let (through, w_now, r_now) = {
            let mut g = TABLE.lock().unwrap();
            let Some(t) = g.as_mut() else { return };
            let (Some(x), Some(y)) = (t.by_res.get(&w), t.by_res.get(&r)) else { continue };
            if x.now == STATE_UNKNOWN || y.now == STATE_UNKNOWN {
                LEFT[0].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let (dev, from, to, now) = (x.dev as *mut c_void, x.desc, y.desc, (x.now, y.now));
            let Some(through) = t.through.entry((w, r)).or_insert_with(|| make_through(dev, &from, &to).map(Arc::new)).clone() else {
                LEFT[2].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            (through, now.0, now.1)
        };
        let buffer = through.buffer.as_raw() as usize;
        let mut before = vec![tr(buffer, 0, COPY_DEST)];
        let mut after = vec![tr(buffer, COPY_SOURCE, 0)];
        for (res, now, copy) in [(w, w_now, COPY_SOURCE), (r, r_now, COPY_DEST)] {
            if now != copy {
                before.push(tr(res, now, copy));
                after.push(tr(res, copy, now));
            }
        }
        super::list_x::texture_through_buffer(list, w, r, buffer, &through.parts, &before, &tr(buffer, COPY_DEST, COPY_SOURCE), &after);
        // (what the sampled one holds is the GPU's now: no upload from the title's memory over it)
        super::title_textures::note_gpu_write(list as usize, r);
        COPIES[why].fetch_add(1, Ordering::Relaxed);
        let n = COPIES[0].load(Ordering::Relaxed) + COPIES[1].load(Ordering::Relaxed);
        if n <= 4 || n.is_power_of_two() {
            crate::xlog_info!(
                "D3D12.X: alias copy: {w:#x} (in state {w_now:#x}) into {r:#x} (in state {r_now:#x}); {} copies at enhanced barriers and {} at legacy transitions so far, left out: {} for a state not known, {} on a list of another kind, {} of pairs that cannot be copied",
                COPIES[0].load(Ordering::Relaxed),
                COPIES[1].load(Ordering::Relaxed),
                LEFT[0].load(Ordering::Relaxed),
                LEFT[1].load(Ordering::Relaxed),
                LEFT[2].load(Ordering::Relaxed)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(format: u32, width: u64, height: u32, mips: u16, flags: i32) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: width,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: mips,
            Format: DXGI_FORMAT(format as i32),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAGS(flags),
            ..Default::default()
        }
    }

    const NONE: [Job; 0] = [];

    fn pairs_of(res: usize) -> Vec<usize> {
        sampled_of(TABLE.lock().unwrap().as_ref().unwrap(), res)
    }

    /// A sampled texture is the pair of a written one at its address when size, levels and
    /// texel width agree and the format does not; a buffer, a depth buffer and a
    /// block-compressed texture are not kept at all.
    #[test]
    fn a_sampled_texture_in_the_memory_of_a_written_one_of_another_format_is_its_pair() {
        let (va, w, r) = (0x7AC0_0000_1000usize, 0x7AC1_0000_0010usize, 0x7AC1_0000_0020usize);
        // R32_UINT with unordered access, R9G9B9E5 sampled
        keep(1, w, va, &texture(42, 640, 360, 1, 4), 0x8);
        keep(1, r, va, &texture(67, 640, 360, 1, 0), 0xC0);
        assert_eq!(pairs_of(w), [r]);
        assert_eq!(pairs_of(r), [] as [usize; 0], "asked of the sampled one: none");
        // another size, another count of levels, a wider texel, the same format, another address
        for (k, (d, at)) in [(texture(67, 320, 180, 1, 0), va), (texture(67, 640, 360, 2, 0), va), (texture(10, 640, 360, 1, 0), va), (texture(42, 640, 360, 1, 0), va), (texture(67, 640, 360, 1, 0), va + 0x1_0000)]
            .into_iter()
            .enumerate()
        {
            keep(1, r + 0x100 + k, at, &d, 0xC0);
        }
        assert_eq!(pairs_of(w), [r], "none of those is its pair");
        // a second written one there (a render target) of the same shape is no pair of the first
        keep(1, w + 0x1000, va, &texture(42, 640, 360, 1, 1), 0x4);
        assert_eq!(pairs_of(w), [r]);
        assert_eq!(pairs_of(w + 0x1000), [r], "the sampled one is the pair of both");
        keep(1, w + 0x2000, va, &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, ..texture(0, 4096, 1, 1, 0) }, 0);
        keep(1, w + 0x2001, va, &texture(45, 640, 360, 1, 2), 0x10);
        keep(1, w + 0x2002, va, &texture(71, 640, 360, 1, 0), 0xC0);
        assert!(!TABLE.lock().unwrap().as_ref().unwrap().by_res.contains_key(&(w + 0x2000)), "a buffer is not kept");
        assert!((0x2001..=0x2002).all(|k| !TABLE.lock().unwrap().as_ref().unwrap().by_res.contains_key(&(w + k))), "nor a depth buffer, nor a BC1 texture");
        // the sampled one gone: no pair
        forget(r);
        assert_eq!(pairs_of(w), [] as [usize; 0]);
        for res in (0..5).map(|k| r + 0x100 + k).chain([w, w + 0x1000]) {
            forget(res);
        }
    }

    /// A written texture's legacy transition out of UNORDERED_ACCESS or RENDER_TARGET asks for
    /// the copy; one that stays written, the first half of a split one and a transition of the
    /// sampled one do not. The states are followed.
    #[test]
    fn a_written_textures_transition_out_of_a_written_state_asks_for_the_copy() {
        let (va, w, r) = (0x7AC0_0000_2000usize, 0x7AC2_0000_0010usize, 0x7AC2_0000_0020usize);
        keep(1, w, va, &texture(42, 128, 128, 1, 5), 0x8);
        keep(1, r, va, &texture(67, 128, 128, 1, 0), 0xC0);
        let tr = |res: usize, flags: u32, sub: u32, before: u32, after: u32| [0, flags, res as u32, (res as u64 >> 32) as u32, sub, before, after, 0];
        let now = |res: usize| TABLE.lock().unwrap().as_ref().unwrap().by_res[&res].now;
        let all = u32::MAX;
        assert_eq!(plan(&[tr(w, 0, ALL, 0x8, 0xC0)], all), [(w, r, 1)], "unordered access to shader resource");
        assert_eq!(now(w), 0xC0);
        assert_eq!(plan(&[tr(w, 0, 0, 0xC0, 0x4)], all), NONE, "into a written state: nothing to copy yet");
        assert_eq!(now(w), 0x4, "its one subresource");
        assert_eq!(plan(&[tr(w, BEGIN_ONLY, ALL, 0x4, 0x800)], all), NONE, "the first half of a split transition");
        assert_eq!(now(w), 0x4);
        assert_eq!(plan(&[tr(w, 2, ALL, 0x4, 0x800), tr(w, 0, ALL, 0x800, 0x4), tr(w, 0, ALL, 0x4, 0x8)], all), [(w, r, 1)], "its end: once for the call; render target to unordered access asks for none");
        assert_eq!(now(w), 0x8);
        assert_eq!(plan(&[tr(r, 0, ALL, 0xC0, 0x400)], all), NONE, "the sampled one's transition asks for none");
        assert_eq!(now(r), 0x400);
        // console-only state bits are not the PC's
        assert_eq!(plan(&[tr(r, 0, ALL, 0x400, 0x10_0000 | 0xC0)], 0xFFFF), NONE);
        assert_eq!(now(r), 0xC0);
        // a UAV barrier and an aliasing barrier ask for none
        assert_eq!(plan(&[[2, 0, w as u32, (w as u64 >> 32) as u32, 0, 0, 0, 0], [1, 0, w as u32, (w as u64 >> 32) as u32, r as u32, (r as u64 >> 32) as u32, 0, 0]], all), NONE);
        forget(w);
        forget(r);
    }

    /// An enhanced barrier of a written texture whose access before is unordered access or
    /// render target asks for the copy; other accesses, other textures, buffer and global groups
    /// and a call that is no list of groups do not.
    #[test]
    fn an_enhanced_barrier_of_a_written_texture_asks_for_the_copy() {
        let (va, w, r) = (0x7AC0_0000_3000usize, 0x7AC3_0000_0010usize, 0x7AC3_0000_0020usize);
        keep(1, w, va, &texture(42, 192, 108, 1, 4), 0x8);
        keep(1, r, va, &texture(67, 192, 108, 1, 0), 0xC0);
        // sync, sync, access before, access after, layout before, layout after, resource, range, flags
        let barrier = |res: usize, access: (u32, u32)| -> [u32; 16] { [0x1000, 0x1000, access.0, access.1, 3, 6, res as u32, (res as u64 >> 32) as u32, u32::MAX, 0, 0, 0, 0, 0, 0, 0] };
        #[repr(C, align(8))]
        struct Group {
            ty: u32,
            count: u32,
            at: u64,
        }
        let run = |ty: u32, b: &[[u32; 16]]| unsafe { plan_enhanced(1, &Group { ty, count: b.len() as u32, at: b.as_ptr() as u64 } as *const Group as usize) };
        assert_eq!(run(1, &[barrier(w, (0x10, 0x80))]), [(w, r, 0)], "unordered access to shader resource, as a retail title has it");
        assert_eq!(run(1, &[barrier(w, (0x8, 0x80)), barrier(w, (0x10, 0x10))]), [(w, r, 0)], "render target; once for the call");
        assert_eq!(run(1, &[barrier(w, (0x80, 0x10))]), NONE, "read before: nothing written since");
        assert_eq!(run(1, &[barrier(w, (0x8000_0000, 0x80))]), NONE, "no access before");
        assert_eq!(run(1, &[barrier(r, (0x10, 0x80)), barrier(0x7AC3_0000_0F00, (0x10, 0x80))]), NONE, "the sampled one, a texture not kept");
        assert_eq!(run(2, &[barrier(w, (0x10, 0x80))]), NONE, "a buffer group");
        assert_eq!(run(0, &[barrier(w, (0x10, 0x80))]), NONE, "a global group");
        unsafe {
            assert_eq!(plan_enhanced(0, 0), NONE);
            // (XboxBvhBuilder.dll's call: one description that is no group of texture barriers)
            let odd = [0u64; 2];
            assert_eq!(plan_enhanced(1, odd.as_ptr() as usize), NONE);
            assert_eq!(plan_enhanced(1, 0x10), NONE, "not readable");
            assert_eq!(plan_enhanced(5000, odd.as_ptr() as usize), NONE, "too many groups to be one");
        }
        forget(w);
        forget(r);
    }

    /// The copy on a GPU (WARP): an R32_UINT array texture with unordered access, 70 layers and
    /// two levels (140 parts) holds a pattern; after the enhanced barrier that ends its writing,
    /// its R9G9B9E5 pair, read back, holds the same bytes in every level of every layer.
    #[test]
    fn gpu_a_written_textures_texels_reach_its_sampled_pair_at_the_enhanced_barrier() {
        use super::super::real::{adopt, Kind};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};
        unsafe {
            let skip = || crate::gpu::recomp::runner::skip_test("no WARP device");
            let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return skip() };
            let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else { return skip() };
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                return skip();
            }
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let committed = |d: &D3D12_RESOURCE_DESC, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| -> ID3D12Resource {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, d, state, None, &mut r).expect("a resource");
                r.unwrap()
            };
            let (read_states, default) = (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_HEAP_TYPE_DEFAULT);
            const PARTS: usize = 140;
            let (of_written, of_sampled) = (D3D12_RESOURCE_DESC { DepthOrArraySize: 70, ..texture(42, 16, 8, 2, 4) }, D3D12_RESOURCE_DESC { DepthOrArraySize: 70, ..texture(67, 16, 8, 2, 0) });
            let written = committed(&of_written, default, D3D12_RESOURCE_STATE_COPY_DEST);
            let sampled = committed(&of_sampled, default, read_states);
            let layout = |d: &D3D12_RESOURCE_DESC| {
                let (mut rows, mut bytes) = (vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); PARTS], 0u64);
                dev.GetCopyableFootprints(d, 0, PARTS as u32, 0, Some(rows.as_mut_ptr()), None, None, Some(&mut bytes));
                (rows, bytes)
            };
            let ((rows_w, bytes), (rows_r, _)) = (layout(&of_written), layout(&of_sampled));
            let buffer = |heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
                committed(
                    &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() },
                    heap,
                    state,
                )
            };
            // every texel another value
            let texel = |part: usize, x: usize, y: usize| ((part as u32 + 1) << 16 | (y as u32) << 8 | x as u32).wrapping_mul(0x9E37_79B1);
            let up = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            let mut p = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).expect("map upload");
            for (level, f) in rows_w.iter().enumerate() {
                for y in 0..f.Footprint.Height as usize {
                    for x in 0..f.Footprint.Width as usize {
                        ((p as *mut u8).add(f.Offset as usize + y * f.Footprint.RowPitch as usize + x * 4) as *mut u32).write_unaligned(texel(level, x, y));
                    }
                }
            }
            up.Unmap(0, None);
            let lraw = list.as_raw();
            // (PC calls of the list by slot: once the list is adopted its own table is XCL's)
            let pc_vt = *(lraw as *const *const usize);
            let barrier = |res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES| {
                let r = res.as_raw() as usize;
                let b = [0u32, 0, r as u32, (r as u64 >> 32) as u32, ALL, before.0 as u32, after.0 as u32, 0];
                let call: unsafe extern "system" fn(*mut c_void, u32, *const [u32; 8]) = std::mem::transmute(*pc_vt.add(26));
                call(lraw, 1, &b);
            };
            let copy = |dst: &D3D12_TEXTURE_COPY_LOCATION, src: &D3D12_TEXTURE_COPY_LOCATION| {
                let call: unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const c_void) = std::mem::transmute(*pc_vt.add(16));
                call(lraw, dst, 0, 0, 0, src, std::ptr::null());
            };
            let placed = |b: &ID3D12Resource, f: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(b), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *f } };
            let level = |t: &ID3D12Resource, k: usize| D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(t), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
            for (k, f) in rows_w.iter().enumerate() {
                copy(&level(&written, k), &placed(&up, f));
            }
            barrier(&written, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            // the two as a title placed them, the list as a title's
            let (va, w, r) = (0x7AC0_0000_4000usize, written.as_raw() as usize, sampled.as_raw() as usize);
            keep(dev.as_raw() as usize, w, va, &of_written, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32);
            keep(dev.as_raw() as usize, r, va, &of_sampled, read_states.0 as u32);
            adopt(lraw, Kind::CommandList);
            super::super::list_x::on_create(lraw, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            // its enhanced barrier of the written one: unordered access to shader resource
            let words: [u32; 16] = [0x1000, 0x1000, 0x10, 0x80, 3, 6, w as u32, (w as u64 >> 32) as u32, u32::MAX, 0, 0, 0, 0, 0, 0, 0];
            let group: [u64; 2] = [1 | 1 << 32, words.as_ptr() as u64];
            let jobs = plan_enhanced(1, group.as_ptr() as usize);
            assert_eq!(jobs, [(w, r, 0)]);
            let copies = COPIES[0].load(Ordering::Relaxed);
            record(lraw, &jobs);
            assert!(COPIES[0].load(Ordering::Relaxed) > copies, "the copy is recorded");

            // the sampled one, from the state the title has it in, to a readback buffer
            let back = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
            barrier(&sampled, read_states, D3D12_RESOURCE_STATE_COPY_SOURCE);
            for (k, f) in rows_r.iter().enumerate() {
                copy(&placed(&back, f), &level(&sampled, k));
            }
            let close: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(*pc_vt.add(9));
            assert!(close(lraw) >= 0, "close list");
            // (the PC queue's own call with the list as it is: no reference of the list changes hands)
            let q = queue.as_raw();
            let execute: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) = std::mem::transmute(*(*(q as *const *const usize)).add(10));
            execute(q, 1, &lraw);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            queue.Signal(&fence, 1).expect("signal");
            crate::gpu::fence::wait(&fence, 1, &dev, None, 5000).expect("the copies complete");
            assert!(dev.GetDeviceRemovedReason().is_ok(), "the device is there");

            let mut p = std::ptr::null_mut();
            back.Map(0, None, Some(&mut p)).expect("map readback");
            let mut wrong = Vec::new();
            for (k, f) in rows_r.iter().enumerate() {
                for y in 0..f.Footprint.Height as usize {
                    for x in 0..f.Footprint.Width as usize {
                        let got = ((p as *const u8).add(f.Offset as usize + y * f.Footprint.RowPitch as usize + x * 4) as *const u32).read_unaligned();
                        if got != texel(k, x, y) && wrong.len() < 8 {
                            wrong.push((k, x, y, got));
                        }
                    }
                }
            }
            back.Unmap(0, None);
            assert!(wrong.is_empty(), "(part, x, y, bits) of the sampled texture that are not the written one's: {wrong:x?}");
            forget(w);
            forget(r);
        }
    }
}
