//! The live placements of the title's textures on video memory shared by address (`vram`): each
//! PC texture, the title pages it has (`vram::owned_pages`: tile i on the page at `start + i *
//! 64 KiB`), its kind and when it came. Two textures on the same pages share video memory as on
//! the console – what one is written with, the other shows.
//!
//! `XCL_PLACE_CHECK=1` logs placements over the pages of a live texture (the first 32 in full)
//! and counts them every 2 s, by how their pages lie and by kind.
//!
//! A placement over other pages of a live target supersedes it (`supersedes`; `XCL_PLACE_RETIRE=0`
//! turns it off): what is uploaded into or drawn to the newer one then no longer shows in the
//! older one's tiles. Its tiles leave the shared pages once the queues are past the work submitted
//! before (`late_release::hold_unmap`), and come back when the title uses it again (`note_use`: a
//! barrier of it, also a texture barrier in list #146's groups; a copy of it; its view bound as a
//! render target or depth buffer, or cleared) or releases the newer ones over it while it still
//! holds it.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::{ID3D12Resource, D3D12_RESOURCE_DESC};

/// What the GPU does with a placed texture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kind {
    /// a render target, depth buffer or UAV texture (`vram::gpu_owned`)
    Target,
    /// sampled: kept in step with the title's memory (`title_textures`)
    Sampled,
}

impl Kind {
    fn word(self) -> &'static str {
        match self {
            Kind::Target => "target",
            Kind::Sampled => "sampled",
        }
    }
}

/// Where a placement's video memory is: the title's pages of a device, or an arena (`vram`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Backing {
    Pages(usize),
    Arena(u64),
}

/// The PC description of a placed texture, for the log.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct Desc {
    pub width: u64,
    pub height: u32,
    pub depth: u16,
    pub mips: u16,
    pub format: i32,
    pub flags: i32,
}

impl Desc {
    pub(super) fn of(d: &D3D12_RESOURCE_DESC) -> Desc {
        Desc { width: d.Width, height: d.Height, depth: d.DepthOrArraySize, mips: d.MipLevels, format: d.Format.0, flags: d.Flags.0 }
    }
}

impl std::fmt::Display for Desc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{} format {} flags {:#x} {} mips", self.width, self.height, self.depth, self.format, self.flags, self.mips)
    }
}

#[derive(Clone, Debug)]
struct Placement {
    /// the title address it was placed at (`vram::canonical_va`)
    va: usize,
    /// its title pages `[start, end)` (none of its own: `start == end`)
    start: usize,
    end: usize,
    backing: Backing,
    kind: Kind,
    seq: u64,
    at: Instant,
    desc: Desc,
    /// the title released it: it goes on the PC once the GPU is past it (`late_release`)
    released: bool,
    /// false: superseded, its tiles off the shared pages (or about to be)
    mapped: bool,
    /// the newer placements it is superseded by
    under: Vec<usize>,
    /// the older ones it supersedes
    over: Vec<usize>,
}

/// How a newer placement's pages lie to an older one's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Overlap {
    /// the same pages: one memory under two descriptions
    Same,
    /// all its pages are the older one's
    Inside,
    /// all the older one's pages are its
    Around,
    /// some of each one's pages are the other's
    Partial,
}

impl Overlap {
    fn words(self) -> &'static str {
        match self {
            Overlap::Same => "on the same pages as",
            Overlap::Inside => "inside",
            Overlap::Around => "around",
            Overlap::Partial => "partly over",
        }
    }
}

/// How placement `new`'s pages lie to `old`'s (None: they share no video memory). Placements at
/// one address share their tiles past their own pages too (`vram::Pages::extra_pages`).
fn overlap(new: &Placement, old: &Placement) -> Option<Overlap> {
    if new.backing != old.backing {
        return None;
    }
    let shared = new.start < old.end && old.start < new.end;
    if !shared && new.va != old.va {
        return None;
    }
    Some(if (new.start, new.end) == (old.start, old.end) {
        Overlap::Same
    } else if old.start <= new.start && new.end <= old.end {
        Overlap::Inside
    } else if new.start <= old.start && old.end <= new.end {
        Overlap::Around
    } else {
        Overlap::Partial
    })
}

/// Whether a newer placement lying `overlap` to an older one supersedes it: a target the title
/// still holds, mapped, on the title's pages (an arena has no tiles to unmap), and on other pages
/// than the newer one (the same pages are one memory under two descriptions, shared as on the
/// console). An older sampled texture is retired by `title_textures` (under a sampled one:
/// `Table::insert`; under a target: `retire_overlapping`).
pub(super) fn supersedes(overlap: Overlap, old_kind: Kind, old_backing: Backing, old_released: bool, old_mapped: bool) -> bool {
    overlap != Overlap::Same && old_kind == Kind::Target && matches!(old_backing, Backing::Pages(_)) && !old_released && old_mapped
}

/// A placement over the pages of an older live one.
#[derive(Clone, Debug)]
pub(super) struct Case {
    pub old: usize,
    pub overlap: Overlap,
    /// (the newer one's, the older one's)
    pub kinds: (Kind, Kind),
    /// placements between them
    pub ago: u64,
    pub ms: u64,
    /// the older one was released by the title and is not gone on the PC yet
    pub released: bool,
    pub old_desc: Desc,
    /// the older one's (address, first page, end of its pages)
    pub old_range: (usize, usize, usize),
    pub old_seq: u64,
    /// the newer one supersedes it
    pub retired: bool,
}

/// the placements are found by the 2 MiB blocks of the title's addresses they cover
const BLOCK_SHIFT: usize = 21;

#[derive(Default)]
struct Registry {
    live: HashMap<usize, Placement>,
    /// block → the placements over it
    blocks: BTreeMap<usize, Vec<usize>>,
    seq: u64,
    /// the superseded ones the title holds (`SUPERSEDED`)
    unmapped: usize,
}

/// What a placement found (`Registry::place`).
#[derive(Default)]
struct Placed {
    seq: u64,
    /// with `look`: the older ones on its video memory, oldest first
    cases: Vec<Case>,
    /// the older ones it supersedes
    victims: Vec<usize>,
    /// ones to map again (`Registry::release`)
    remap: Vec<usize>,
}

impl Registry {
    /// The blocks of `p`'s pages and of its address.
    fn blocks_of(p: &Placement) -> std::ops::RangeInclusive<usize> {
        let last = if p.end > p.start { p.end - 1 } else { p.start };
        (p.va.min(p.start) >> BLOCK_SHIFT)..=(last.max(p.va) >> BLOCK_SHIFT)
    }

    /// The live placements but `p` itself that share video memory with `p`.
    fn over(&self, res: usize, p: &Placement) -> Vec<(usize, Overlap)> {
        let mut out: Vec<(usize, Overlap)> = Vec::new();
        for b in Self::blocks_of(p) {
            for &o in self.blocks.get(&b).into_iter().flatten() {
                if o == res || out.iter().any(|&(k, _)| k == o) {
                    continue;
                }
                if let Some(ov) = self.live.get(&o).and_then(|old| overlap(p, old)) {
                    out.push((o, ov));
                }
            }
        }
        out
    }

    /// Texture `res` placed; with `look`, the older live ones it shares video memory with, and
    /// with `retire` the ones it supersedes (`supersedes`).
    #[allow(clippy::too_many_arguments)]
    fn place(&mut self, res: usize, va: usize, (start, end): (usize, usize), backing: Backing, kind: Kind, desc: Desc, now: Instant, look: bool, retire: bool) -> Placed {
        // (an object's address again: the one before went without a word)
        let remap = self.remove(res).unwrap_or_default();
        self.seq += 1;
        let mut p = Placement { va, start, end: end.max(start), backing, kind, seq: self.seq, at: now, desc, released: false, mapped: true, under: Vec::new(), over: Vec::new() };
        let found = if look || retire { self.over(res, &p) } else { Vec::new() };
        let mut out = Placed { seq: self.seq, remap, ..Default::default() };
        for &(o, overlap) in &found {
            let Some(old) = self.live.get_mut(&o) else { continue };
            let retired = retire && supersedes(overlap, old.kind, old.backing, old.released, old.mapped);
            if look {
                out.cases.push(Case {
                    old: o,
                    overlap,
                    kinds: (kind, old.kind),
                    ago: p.seq - old.seq,
                    ms: now.saturating_duration_since(old.at).as_millis() as u64,
                    released: old.released,
                    old_desc: old.desc,
                    old_range: (old.va, old.start, old.end),
                    old_seq: old.seq,
                    retired,
                });
            }
            if retired {
                old.mapped = false;
                old.under.push(res);
                p.over.push(o);
                self.unmapped += 1;
                out.victims.push(o);
            }
        }
        out.cases.sort_by_key(|c| c.old_seq);
        out.victims.sort_by_key(|o| self.live.get(o).map_or(0, |q| q.seq));
        for b in Self::blocks_of(&p) {
            self.blocks.entry(b).or_default().push(res);
        }
        self.live.insert(res, p);
        out
    }

    /// The title released `res`; it stays until it is gone on the PC (`remove`). Returns the older
    /// ones it superseded that are to be mapped again: the title holds them, and no other newer one
    /// it holds lies over other pages of them (those supersede them from now on).
    fn release(&mut self, res: usize) -> Vec<usize> {
        let Some(p) = self.live.get_mut(&res).filter(|p| !p.released) else { return Vec::new() };
        p.released = true;
        if !p.mapped {
            // (its tiles go with it: `vram::forget`)
            self.unmapped -= 1;
        }
        let (over, under) = (std::mem::take(&mut p.over), std::mem::take(&mut p.under));
        for n in under {
            if let Some(q) = self.live.get_mut(&n) {
                q.over.retain(|&r| r != res);
            }
        }
        let mut remap = Vec::new();
        for o in over {
            let Some(old) = self.live.get_mut(&o) else { continue };
            old.under.retain(|&r| r != res);
            if !old.under.is_empty() || old.released || old.mapped {
                continue;
            }
            let old = old.clone();
            let newer: Vec<usize> = self
                .over(o, &old)
                .into_iter()
                .filter(|&(n, ov)| ov != Overlap::Same && self.live.get(&n).is_some_and(|q| q.seq > old.seq && !q.released))
                .map(|(n, _)| n)
                .collect();
            for n in &newer {
                if let Some(q) = self.live.get_mut(n) {
                    q.over.push(o);
                }
            }
            if let Some(old) = self.live.get_mut(&o) {
                if newer.is_empty() {
                    old.mapped = true;
                    self.unmapped -= 1;
                    remap.push(o);
                } else {
                    old.under = newer;
                }
            }
        }
        remap
    }

    /// `res` is gone on the PC (None: it was not here); what to map again, as `release`.
    fn remove(&mut self, res: usize) -> Option<Vec<usize>> {
        let remap = self.release(res);
        let p = self.live.remove(&res)?;
        for b in Self::blocks_of(&p) {
            if let Some(v) = self.blocks.get_mut(&b) {
                v.retain(|&r| r != res);
                if v.is_empty() {
                    self.blocks.remove(&b);
                }
            }
        }
        Some(remap)
    }

    /// The title uses superseded `res` again: it is mapped again (the title shares the memory, as
    /// the console lets it). Returns the newer ones it was superseded by (none: it was not).
    fn use_again(&mut self, res: usize) -> Vec<usize> {
        let Some(p) = self.live.get_mut(&res).filter(|p| !p.mapped && !p.released) else { return Vec::new() };
        p.mapped = true;
        self.unmapped -= 1;
        let under = std::mem::take(&mut p.under);
        for n in &under {
            if let Some(q) = self.live.get_mut(n) {
                q.over.retain(|&r| r != res);
            }
        }
        under
    }

    /// Whether `res` is superseded (its tiles are to leave the shared pages).
    fn superseded(&self, res: usize) -> bool {
        self.live.get(&res).is_some_and(|p| !p.mapped && !p.released)
    }
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

/// `XCL_PLACE_CHECK=1`: placements over the pages of live ones logged and counted.
pub(super) fn check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_PLACE_CHECK").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("D3D12.X: XCL_PLACE_CHECK=1 - placements over the pages of live textures are logged (the first 32) and counted every 2 s");
        }
        on
    })
}

/// `XCL_PLACE_RETIRE` (on; `=0` off): a placement over other pages of a live target supersedes it.
pub(super) fn retire() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = !std::env::var("XCL_PLACE_RETIRE").is_ok_and(|v| v.trim() == "0");
        if !on {
            crate::xlog_info!("D3D12.X: XCL_PLACE_RETIRE=0 - a target stays on the pages of a newer placement over it");
        }
        on
    })
}

/// superseded placements the title holds (`Registry::unmapped`): with none, `note_use` takes no lock
static SUPERSEDED: AtomicUsize = AtomicUsize::new(0);

/// The counts of the 2 s line: [made, aliases at the same range, over released ones not gone
/// yet, partial (tt, st, ts, ss), one inside the other (tt, st, ts, ss), aliases of a sampled
/// texture over a target, older ones superseded, their tiles taken off the shared pages,
/// superseded ones used again] – tt: a target over a target, st: a sampled texture over a
/// target, ...
static COUNTS: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];
static LAST_LINE: Mutex<Option<Instant>> = Mutex::new(None);

/// The kinds' place in `COUNTS`' groups of four.
fn kinds_index((new, old): (Kind, Kind)) -> usize {
    (new == Kind::Sampled) as usize + 2 * (old == Kind::Sampled) as usize
}

/// The 2 s line of `counts` (`COUNTS`).
fn counts_line(c: &[u64; 15]) -> String {
    let p = c[3] + c[4] + c[5] + c[6];
    let i = c[7] + c[8] + c[9] + c[10];
    format!(
        "D3D12.X: placements (2 s): {} made, {} aliases at the same range ({} sampled over a target), {p} partial overlaps (tt/st/ts/ss {}/{}/{}/{}), {i} one inside the other (tt/st/ts/ss {}/{}/{}/{}), {} over ones the title released (not gone yet); {} older ones superseded, {} retired (off the shared pages), {} used again",
        c[0], c[1], c[11], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[2], c[12], c[13], c[14]
    )
}

/// The full line of a case (the first 32).
fn case_line(res: usize, seq: u64, kind: Kind, desc: &Desc, (va, start, end): (usize, usize, usize), c: &Case) -> String {
    format!(
        "D3D12.X: placement #{seq} ({} {desc}, object {res:#x}) at {va:#x}, pages {start:#x}..{end:#x}, is {} #{} ({} {}, object {:#x}) at {:#x}, pages {:#x}..{:#x}, placed {} placements / {} ms before{}{}",
        kind.word(),
        c.overlap.words(),
        c.old_seq,
        c.kinds.1.word(),
        c.old_desc,
        c.old,
        c.old_range.0,
        c.old_range.1,
        c.old_range.2,
        c.ago,
        c.ms,
        if c.released { ", released by the title (not gone yet)" } else { "" },
        if c.retired { "; it supersedes it" } else { "" }
    )
}

/// The cases of a placement into `COUNTS`, the first 32 logged; the 2 s line when it is due.
fn note(res: usize, seq: u64, kind: Kind, desc: &Desc, range: (usize, usize, usize), cases: &[Case]) {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    COUNTS[0].fetch_add(1, Ordering::Relaxed);
    for c in cases {
        let k = match c.overlap {
            Overlap::Same => 1,
            Overlap::Partial => 3 + kinds_index(c.kinds),
            Overlap::Inside | Overlap::Around => 7 + kinds_index(c.kinds),
        };
        COUNTS[k].fetch_add(1, Ordering::Relaxed);
        if c.overlap == Overlap::Same && c.kinds == (Kind::Sampled, Kind::Target) {
            COUNTS[11].fetch_add(1, Ordering::Relaxed);
        }
        if c.released {
            COUNTS[2].fetch_add(1, Ordering::Relaxed);
        }
        if c.retired {
            COUNTS[12].fetch_add(1, Ordering::Relaxed);
        }
        // (an alias in full only of another kind: a sampled one is uploaded over what the GPU wrote)
        if (c.overlap != Overlap::Same || c.kinds.0 != c.kinds.1) && LOGGED.fetch_add(1, Ordering::Relaxed) < 32 {
            crate::xlog_info!("{}", case_line(res, seq, kind, desc, range, c));
        }
    }
    let now = Instant::now();
    let due = {
        let mut last = LAST_LINE.lock().unwrap();
        let due = last.is_some_and(|t| now.duration_since(t).as_secs() >= 2);
        if due || last.is_none() {
            *last = Some(now);
        }
        due
    };
    if due {
        let c: [u64; 15] = std::array::from_fn(|k| COUNTS[k].swap(0, Ordering::Relaxed));
        crate::xlog_info!("{}", counts_line(&c));
    }
}

/// A reference of XCL's own to placement `raw`, one the title holds (its release takes the
/// registry's lock first: `released`). Dropped outside that lock: it may be the last one.
unsafe fn resource(raw: usize) -> Option<ID3D12Resource> {
    let p = raw as *mut c_void;
    ID3D12Resource::from_raw_borrowed(&p).cloned()
}

/// The superseded ones of `remap` go back on the shared pages (`vram::remap_if_unmapped`), under
/// the registry's lock: a deferred unmap (`unmap_if_superseded`) is before or after it. Returns
/// the references taken, to drop outside the lock.
unsafe fn map_again(remap: &[usize]) -> Vec<ID3D12Resource> {
    let held: Vec<ID3D12Resource> = remap.iter().filter_map(|&o| resource(o)).collect();
    for r in &held {
        super::vram::remap_if_unmapped(r);
    }
    held
}

/// Texture `res` placed at title address `va` on `pages` (`[start, end)`: its own title pages),
/// its tiles mapped. Returns the older targets it supersedes: their tiles are to leave the shared
/// pages once the queues are past the work submitted before (`late_release::hold_unmap`).
pub(super) fn placed(res: usize, va: usize, pages: (usize, usize), backing: Backing, kind: Kind, desc: &D3D12_RESOURCE_DESC) -> Vec<ID3D12Resource> {
    let (look, retire) = (check(), retire());
    let desc = Desc::of(desc);
    let (out, victims, held) = {
        let mut g = REGISTRY.lock().unwrap();
        let r = g.get_or_insert_with(Registry::default);
        let out = r.place(res, va, pages, backing, kind, desc, Instant::now(), look, retire);
        SUPERSEDED.store(r.unmapped, Ordering::Relaxed);
        let victims: Vec<ID3D12Resource> = out.victims.iter().filter_map(|&o| unsafe { resource(o) }).collect();
        let held = unsafe { map_again(&out.remap) };
        (out, victims, held)
    };
    drop(held);
    if look {
        note(res, out.seq, kind, &desc, (va, pages.0, pages.1.max(pages.0)), &out.cases);
    }
    victims
}

/// What `f` does to the registry, and the ones it says to map again mapped again.
fn settle(f: impl FnOnce(&mut Registry) -> Vec<usize>) {
    let held = {
        let mut g = REGISTRY.lock().unwrap();
        let Some(r) = g.as_mut() else { return };
        let remap = f(r);
        SUPERSEDED.store(r.unmapped, Ordering::Relaxed);
        unsafe { map_again(&remap) }
    };
    drop(held);
}

/// The title released placed texture `res` (`vram::forget`, after its pages went idle): the
/// older targets only it superseded go back on the shared pages.
pub(super) fn released(res: usize) {
    settle(|r| r.release(res));
}

/// Resource `res` is gone on the PC (or goes now).
pub(super) fn gone(res: usize) {
    settle(|r| r.remove(res).unwrap_or_default());
}

/// A superseded target's tiles leave the shared pages (`late_release::hold_unmap`, the queues
/// past the work submitted before) – unless the title used it again in the meantime, released
/// it, or released the newer ones over it.
pub(super) unsafe fn unmap_if_superseded(res: &ID3D12Resource) {
    let g = REGISTRY.lock().unwrap();
    if g.as_ref().is_some_and(|r| r.superseded(res.as_raw() as usize)) {
        super::vram::unmap_superseded_target(res);
        COUNTS[13].fetch_add(1, Ordering::Relaxed);
    }
}

/// The title uses `res` (a barrier, a copy): a superseded target goes back on the shared pages –
/// on the console the title may use both over one memory when it knows they share it. Logged
/// once for each pair (the first 64). The work recorded now waits for the mapping
/// (`vram::before_execute` at its submit).
pub(super) fn note_use(res: usize) {
    if res == 0 || SUPERSEDED.load(Ordering::Relaxed) == 0 {
        return;
    }
    static LOGGED: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);
    let held = {
        let mut g = REGISTRY.lock().unwrap();
        let Some(r) = g.as_mut() else { return };
        let under = r.use_again(res);
        SUPERSEDED.store(r.unmapped, Ordering::Relaxed);
        if under.is_empty() {
            return;
        }
        COUNTS[14].fetch_add(1, Ordering::Relaxed);
        if let Some(old) = r.live.get(&res) {
            for n in &under {
                let Some(new) = r.live.get(n) else { continue };
                let mut seen = LOGGED.lock().unwrap();
                let seen = seen.get_or_insert_with(Default::default);
                if seen.len() < 64 && seen.insert((old.seq, new.seq)) {
                    crate::xlog_info!(
                        "D3D12.X: a superseded texture is used again: #{} ({} {}, object {res:#x}) at {:#x}, pages {:#x}..{:#x}, under #{} ({} {}, object {n:#x}) at {:#x}, pages {:#x}..{:#x}, placed {} placements later; it goes back on the shared pages",
                        old.seq, old.kind.word(), old.desc, old.va, old.start, old.end, new.seq, new.kind.word(), new.desc, new.va, new.start, new.end, new.seq - old.seq
                    );
                }
            }
        }
        unsafe { map_again(&[res]) }
    };
    drop(held);
}

/// Whether the title holds superseded targets: with none a use has nothing to map again, and the
/// callers that have to look for the texture first (`note_dsv_use`, list #146's groups) do not.
pub(super) fn any_superseded() -> bool {
    SUPERSEDED.load(Ordering::Relaxed) != 0
}

/// DSV handle → the placed texture it names (`note_dsv`). The view is made long before a newer
/// placement supersedes its texture, so the handles of all placed ones are kept; an entry whose
/// texture is gone names no live placement and does nothing (`Registry::use_again`).
static DSVS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// CreateDepthStencilView of `res` at `handle`: kept when `res` is a placed texture.
pub(super) fn note_dsv(handle: usize, res: usize) {
    let placed = res != 0 && REGISTRY.lock().unwrap().as_ref().is_some_and(|r| r.live.contains_key(&res));
    let mut g = DSVS.lock().unwrap();
    if placed {
        g.get_or_insert_with(HashMap::new).insert(handle, res);
    } else if let Some(m) = g.as_mut() {
        m.remove(&handle);
    }
}

/// `n` DSVs copied from `src` to `dst` (handles `inc` apart): each destination names the placed
/// texture its source names, or none.
pub(super) fn dsvs_copied(n: u32, dst: u64, src: u64, inc: u64) {
    let mut g = DSVS.lock().unwrap();
    let Some(m) = g.as_mut() else { return };
    for k in 0..n as u64 {
        let to = (dst + k * inc) as usize;
        match m.get(&((src + k * inc) as usize)).copied() {
            Some(res) => m.insert(to, res),
            None => m.remove(&to),
        };
    }
}

/// The placed texture the DSV at `handle` names.
fn dsv_texture(handle: usize) -> Option<usize> {
    DSVS.lock().unwrap().as_ref()?.get(&handle).copied()
}

/// The title binds or clears the depth buffer of the DSV at `handle`: a use of it (`note_use`).
pub(super) fn note_dsv_use(handle: usize) {
    if !any_superseded() {
        return;
    }
    if let Some(res) = dsv_texture(handle) {
        note_use(res);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K64: usize = 0x1_0000;

    /// A DSV is kept for a placed texture only, and a handle written again (made there, or
    /// copied there) names the new one.
    #[test]
    fn a_dsv_handle_names_the_placed_texture_it_was_last_made_of() {
        // (addresses and handles no other test uses: the registry and the table are the module's)
        let (a, b, other) = (0x7e57_0000_1000usize, 0x7e57_0000_2000usize, 0x7e57_0000_3000usize);
        let (h1, h2) = (0x7e57_d000usize, 0x7e57_d020usize);
        {
            let mut g = REGISTRY.lock().unwrap();
            let r = g.get_or_insert_with(Registry::default);
            r.place(a, 0x7e57_4000_0000, (0x7e57_4000_0000, 0x7e57_4000_0000 + 6 * K64), Backing::Pages(0x7e57), Kind::Target, Desc::default(), Instant::now(), false, false);
            r.place(b, 0x7e57_5000_0000, (0x7e57_5000_0000, 0x7e57_5000_0000 + 8 * K64), Backing::Pages(0x7e57), Kind::Target, Desc::default(), Instant::now(), false, false);
        }
        note_dsv(h1, a);
        note_dsv(h2, other);
        assert_eq!((dsv_texture(h1), dsv_texture(h2)), (Some(a), None), "a view of a texture that is not placed is not kept");
        note_dsv(h1, b);
        assert_eq!(dsv_texture(h1), Some(b));
        // (copies: two handles 0x20 apart from h1, then from a handle that names nothing)
        let (h3, nothing) = (0x7e57_d100usize, 0x7e57_d200usize);
        note_dsv(h2, a);
        dsvs_copied(2, h3 as u64, h1 as u64, 0x20);
        assert_eq!((dsv_texture(h3), dsv_texture(h3 + 0x20)), (Some(b), Some(a)));
        dsvs_copied(1, h3 as u64, nothing as u64, 0x20);
        assert_eq!((dsv_texture(h3), dsv_texture(h3 + 0x20)), (None, Some(a)), "a copy of a view that names no placed texture");
        note_dsv(h2, other);
        note_dsv(h3 + 0x20, other);
        note_dsv(h1, other);
        assert_eq!(dsv_texture(h1), None, "the handle names a texture that is not placed now");
        let mut g = REGISTRY.lock().unwrap();
        let r = g.as_mut().unwrap();
        r.remove(a);
        r.remove(b);
    }
    const MIB: usize = 1 << 20;

    /// A placement looked at, not retiring: (its sequence number, its cases).
    fn place(r: &mut Registry, res: usize, va: usize, pages: (usize, usize), kind: Kind) -> (u64, Vec<Case>) {
        on(r, res, va, pages, Backing::Pages(1), kind)
    }

    fn on(r: &mut Registry, res: usize, va: usize, pages: (usize, usize), backing: Backing, kind: Kind) -> (u64, Vec<Case>) {
        let p = r.place(res, va, pages, backing, kind, Desc::default(), Instant::now(), true, false);
        (p.seq, p.cases)
    }

    /// A placement retiring, not looked at: (the ones it supersedes, the ones to map again).
    fn retiring(r: &mut Registry, res: usize, at: usize, len: usize, kind: Kind) -> (Vec<usize>, Vec<usize>) {
        let p = r.place(res, at, (at, at + len), Backing::Pages(1), kind, Desc::default(), Instant::now(), false, true);
        assert!(p.cases.is_empty(), "not looked at");
        (p.victims, p.remap)
    }

    /// The superseded ones the title holds, as the registry counts them and as they are.
    fn superseded(r: &Registry) -> Vec<usize> {
        let mut v: Vec<usize> = r.live.keys().copied().filter(|&k| r.superseded(k)).collect();
        v.sort_unstable();
        assert_eq!(v.len(), r.unmapped);
        v
    }

    fn overlaps(cases: &[Case]) -> Vec<(usize, Overlap)> {
        cases.iter().map(|c| (c.old, c.overlap)).collect()
    }

    /// Placements are classified by how their pages lie to the live ones'.
    #[test]
    fn a_placement_is_classified_by_its_pages() {
        let mut r = Registry::default();
        let x = 0x40_0000_0000usize;
        // a 14.4 MiB target, then: the same pages, an R8 target inside it, one partly over its end,
        // one around all of it, one past it
        assert!(place(&mut r, 1, x, (x, x + 0xe7_0000), Kind::Target).1.is_empty());
        assert_eq!(overlaps(&place(&mut r, 2, x, (x, x + 0xe7_0000), Kind::Target).1), [(1, Overlap::Same)]);
        let (_, c) = place(&mut r, 3, x + 0xba_0000, (x + 0xba_0000, x + 0xe4_0000), Kind::Target);
        assert_eq!(overlaps(&c), [(1, Overlap::Inside), (2, Overlap::Inside)]);
        let (_, c) = place(&mut r, 4, x + 0xe0_0000, (x + 0xe0_0000, x + 0x120_0000), Kind::Sampled);
        assert_eq!(overlaps(&c), [(1, Overlap::Partial), (2, Overlap::Partial), (3, Overlap::Partial)]);
        assert_eq!(c[0].kinds, (Kind::Sampled, Kind::Target));
        let (_, c) = place(&mut r, 5, x - K64, (x - K64, x + 0x200_0000), Kind::Sampled);
        assert_eq!(overlaps(&c), [(1, Overlap::Around), (2, Overlap::Around), (3, Overlap::Around), (4, Overlap::Around)]);
        assert!(place(&mut r, 6, x + 0x200_0000, (x + 0x200_0000, x + 0x210_0000), Kind::Sampled).1.is_empty());
    }

    /// Ranges that only touch share nothing; nor do placements on other video memory.
    #[test]
    fn neighbours_and_other_memory_share_nothing() {
        let mut r = Registry::default();
        let x = 0x10_0000_0000usize;
        place(&mut r, 1, x, (x, x + 4 * MIB), Kind::Target);
        assert!(place(&mut r, 2, x + 4 * MIB, (x + 4 * MIB, x + 8 * MIB), Kind::Target).1.is_empty());
        assert!(place(&mut r, 3, x - MIB, (x - MIB, x), Kind::Sampled).1.is_empty());
        // (another device's pages, an arena)
        assert!(on(&mut r, 4, x, (x, x + 4 * MIB), Backing::Pages(2), Kind::Target).1.is_empty());
        assert!(on(&mut r, 5, x, (x, x + 4 * MIB), Backing::Arena(7), Kind::Target).1.is_empty());
        let (_, c) = on(&mut r, 6, x + MIB, (x + MIB, x + 2 * MIB), Backing::Arena(7), Kind::Target);
        assert_eq!(overlaps(&c), [(5, Overlap::Inside)]);
    }

    /// A texture at a 4 KiB-aligned address with no page of its own (`vram::owned_pages`) shares
    /// tiles only with one at the same address.
    #[test]
    fn textures_without_pages_of_their_own_share_by_address() {
        let mut r = Registry::default();
        let x = 0x20_0000_0000usize + 0x1000;
        let first = x.next_multiple_of(K64);
        place(&mut r, 1, x, (first, first), Kind::Sampled);
        assert!(place(&mut r, 2, x + 0x2000, (first, first), Kind::Sampled).1.is_empty(), "another address: other tiles");
        assert_eq!(overlaps(&place(&mut r, 3, x, (first, first), Kind::Sampled).1), [(1, Overlap::Same)]);
        // (one with pages from the next 64 KiB on)
        assert_eq!(overlaps(&place(&mut r, 4, x, (first, first + K64), Kind::Target).1), [(1, Overlap::Around), (3, Overlap::Around)]);
    }

    /// A placement over 2 MiB blocks is found from each of them, and once.
    #[test]
    fn a_large_placement_is_found_once_from_every_block() {
        let mut r = Registry::default();
        let x = 0x30_0000_0000usize;
        place(&mut r, 1, x, (x, x + 64 * MIB), Kind::Target);
        for at in [x, x + 31 * MIB, x + 64 * MIB - K64] {
            let (_, c) = place(&mut r, 100 + at, at, (at, at + K64), Kind::Sampled);
            assert_eq!(overlaps(&c), [(1, Overlap::Inside)], "at {at:#x}");
            r.remove(100 + at);
        }
    }

    /// Sequence numbers and ages count placements; a released one is found until it is gone, and
    /// an object's address placed again is a new placement.
    #[test]
    fn released_ones_stay_until_gone_and_sequence_numbers_count_placements() {
        let mut r = Registry::default();
        let x = 0x50_0000_0000usize;
        let (s1, _) = place(&mut r, 1, x, (x, x + 2 * MIB), Kind::Target);
        place(&mut r, 2, x + 8 * MIB, (x + 8 * MIB, x + 9 * MIB), Kind::Target);
        assert!(r.release(1).is_empty());
        let (s3, c) = place(&mut r, 3, x + MIB, (x + MIB, x + 3 * MIB), Kind::Sampled);
        assert_eq!((s3 - s1, c.len(), c[0].ago, c[0].old_seq, c[0].released), (2, 1, 2, s1, true));
        assert!(r.remove(1).is_some() && r.remove(1).is_none());
        assert!(place(&mut r, 4, x, (x, x + MIB), Kind::Target).1.is_empty(), "1 is gone, 3 starts where 4 ends");
        // (the address of object 4 again, elsewhere: the old placement is not found)
        let (_, c) = place(&mut r, 4, x + 8 * MIB, (x + 8 * MIB, x + 9 * MIB), Kind::Target);
        assert_eq!(overlaps(&c), [(2, Overlap::Same)]);
        assert!(place(&mut r, 5, x, (x, x + K64), Kind::Target).1.is_empty(), "object 4 was moved, object 3 starts past it");
        assert!(r.blocks.values().flatten().all(|res| r.live.contains_key(res)));
    }

    /// The 2 s line's counts by kind: the newer one's first.
    #[test]
    fn the_counts_line_names_the_kinds_newer_first() {
        assert_eq!(kinds_index((Kind::Target, Kind::Target)), 0);
        assert_eq!(kinds_index((Kind::Sampled, Kind::Target)), 1);
        assert_eq!(kinds_index((Kind::Target, Kind::Sampled)), 2);
        assert_eq!(kinds_index((Kind::Sampled, Kind::Sampled)), 3);
        let line = counts_line(&[9, 2, 1, 1, 2, 0, 0, 0, 0, 3, 0, 1, 4, 3, 1]);
        assert!(line.starts_with("D3D12.X: placements (2 s): 9 made, 2 aliases at the same range (1 sampled over a target), 3 partial overlaps (tt/st/ts/ss 1/2/0/0), 3 one inside the other (tt/st/ts/ss 0/0/3/0)"), "{line}");
        assert!(line.ends_with("; 4 older ones superseded, 3 retired (off the shared pages), 1 used again"), "{line}");
    }

    /// Only a target the title holds, still mapped, on the title's pages, and on other pages than
    /// the newer one is superseded.
    #[test]
    fn only_a_held_mapped_target_on_other_pages_is_superseded() {
        let pages = Backing::Pages(1);
        for o in [Overlap::Inside, Overlap::Around, Overlap::Partial] {
            assert!(supersedes(o, Kind::Target, pages, false, true), "{o:?}");
        }
        assert!(!supersedes(Overlap::Same, Kind::Target, pages, false, true), "an alias stays shared");
        assert!(!supersedes(Overlap::Partial, Kind::Sampled, pages, false, true), "title_textures retires sampled ones");
        assert!(!supersedes(Overlap::Partial, Kind::Target, Backing::Arena(3), false, true), "no tiles in an arena");
        assert!(!supersedes(Overlap::Partial, Kind::Target, pages, true, true), "released");
        assert!(!supersedes(Overlap::Partial, Kind::Target, pages, false, false), "superseded already");
    }

    /// A sampled texture over part of a target supersedes it; the target is mapped again when the
    /// newer one is released while the title holds it. An alias and an older sampled one are not.
    #[test]
    fn a_placement_over_other_pages_of_a_target_supersedes_it_until_released() {
        let mut r = Registry::default();
        let x = 0x60_0000_0000usize;
        assert_eq!(retiring(&mut r, 1, x, 0xe7_0000, Kind::Target), (vec![], vec![]));
        assert_eq!(retiring(&mut r, 2, x, 0xe7_0000, Kind::Target), (vec![], vec![]), "the same pages: an alias");
        assert_eq!(retiring(&mut r, 3, x + 16 * MIB, MIB, Kind::Sampled), (vec![], vec![]));
        assert_eq!(retiring(&mut r, 4, x + 0xe0_0000, 2 * MIB, Kind::Sampled), (vec![1, 2], vec![]), "over both aliases' ends");
        assert_eq!(superseded(&r), [1, 2]);
        assert!(r.use_again(3).is_empty(), "never superseded");
        assert_eq!(retiring(&mut r, 5, x + 16 * MIB + K64, K64, Kind::Target), (vec![], vec![]), "over a sampled one: title_textures");
        let mut back = r.release(4);
        back.sort_unstable();
        assert_eq!(back, [1, 2]);
        assert!(superseded(&r).is_empty());
        assert!(r.release(4).is_empty(), "once");
    }

    /// A placement over several targets supersedes each; with one of them released by the title
    /// first, only the others are mapped again.
    #[test]
    fn a_placement_over_several_targets_supersedes_each() {
        let mut r = Registry::default();
        let x = 0x70_0000_0000usize;
        for k in 0..3 {
            retiring(&mut r, 1 + k, x + k * 4 * MIB, 4 * MIB, Kind::Target);
        }
        assert_eq!(retiring(&mut r, 9, x + MIB, 10 * MIB, Kind::Target).0, [1, 2, 3], "partly, around, partly");
        assert!(r.release(2).is_empty(), "the older one released: nothing to map again");
        assert_eq!(superseded(&r), [1, 3]);
        assert_eq!(r.remove(9), Some(vec![1, 3]));
        assert!(superseded(&r).is_empty());
        assert!(r.live[&1].over.is_empty() && r.live[&1].under.is_empty());
    }

    /// When the newer one goes while a still newer one the title holds lies over other pages of
    /// the old one, that one supersedes it from then on.
    #[test]
    fn the_next_newer_one_keeps_a_target_superseded() {
        let mut r = Registry::default();
        let x = 0x80_0000_0000usize;
        retiring(&mut r, 1, x, 8 * MIB, Kind::Target);
        assert_eq!(retiring(&mut r, 2, x, 2 * MIB, Kind::Sampled).0, [1]);
        assert_eq!(retiring(&mut r, 3, x + 6 * MIB, 4 * MIB, Kind::Sampled).0, Vec::<usize>::new(), "superseded already");
        assert!(r.release(2).is_empty(), "3 still lies over it");
        assert_eq!(superseded(&r), [1]);
        assert_eq!(r.live[&1].under, [3]);
        assert_eq!(r.release(3), [1]);
        assert!(superseded(&r).is_empty());
    }

    /// A superseded target the title uses again goes back on the shared pages, and the newer
    /// one's release then has nothing to map again.
    #[test]
    fn a_superseded_target_used_again_is_mapped_again() {
        let mut r = Registry::default();
        let x = 0x90_0000_0000usize;
        retiring(&mut r, 1, x, 4 * MIB, Kind::Target);
        assert_eq!(retiring(&mut r, 2, x + 3 * MIB, 4 * MIB, Kind::Target).0, [1]);
        assert_eq!(r.use_again(1), [2]);
        assert!(superseded(&r).is_empty());
        assert!(r.use_again(1).is_empty(), "once");
        assert!(r.release(2).is_empty());
        // (superseded again by a later placement; then the title releases the old one)
        assert_eq!(retiring(&mut r, 3, x + MIB, MIB, Kind::Sampled).0, [1]);
        assert!(r.release(1).is_empty() && superseded(&r).is_empty());
        assert!(r.use_again(1).is_empty(), "released");
        assert!(r.remove(3).unwrap().is_empty());
    }
}
