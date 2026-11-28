//! Shader capture and classification.
//!
//! Every shader blob a title hands to D3D12.X (pipeline state creation, root
//! signatures) is classified and – once per unique hash – written to
//! `<data>\shaders\`. Unknown formats are scanned for RDNA2 programs and a
//! disassembly is written next to the blob. This is how we learn, per title, whether
//! shaders still carry DXIL/DXBC or only console GPU microcode.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::{rdna2, xbox};

/// DXBC container part FourCCs that are well understood on PC.
const KNOWN_PARTS: &[&str] = &[
    "DXIL", "SHEX", "SHDR", "ISGN", "OSGN", "ISG1", "OSG1", "PSG1", "PCSG", "PSV0", "RTS0", "STAT", "ILDB", "ILDN",
    "HASH", "SFI0", "RDEF", "SPDB", "PRIV", "VERS", "SRCI", "PDBI", "SPDB", "OSG5", "IFCE", "RDAT", "MESH",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Part {
    pub fourcc: String,
    pub offset: usize,
    pub size: usize,
}

/// Parse a DXBC-style container ("DXBC", hash, version, size, part table).
pub fn container_parts(blob: &[u8]) -> Option<Vec<Part>> {
    if blob.len() < 32 || &blob[0..4] != b"DXBC" {
        return None;
    }
    let rd = |o: usize| blob.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
    let count = rd(28)?;
    if count > 64 {
        return None;
    }
    let mut parts = Vec::with_capacity(count);
    for i in 0..count {
        let off = rd(32 + i * 4)?;
        let fourcc = blob.get(off..off + 4)?;
        let size = rd(off + 4)?;
        if off + 8 + size > blob.len() {
            return None;
        }
        parts.push(Part {
            fourcc: String::from_utf8_lossy(fourcc).chars().map(|c| if c.is_ascii_graphic() { c } else { '?' }).collect(),
            offset: off + 8,
            size,
        });
    }
    Some(parts)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// DXBC container with a DXIL part – runnable on PC D3D12 (after re-signing if needed)
    Dxil,
    /// DXBC container with SM5 byte code – runnable on PC D3D12
    Dxbc,
    /// console container: DXIL part with XDAT → XBCH → FSSM (native RDNA2 code), no bitcode
    XboxNative,
    /// DXBC container with only parts we do not know (console microcode?)
    ContainerUnknown,
    /// ELF object (AMD code objects are ELF)
    Elf,
    /// no container, but RDNA2 programs were found
    Rdna2Raw,
    Unknown,
}

impl Kind {
    pub fn label(&self) -> &'static str {
        match self {
            Kind::Dxil => "dxil",
            Kind::Dxbc => "dxbc",
            Kind::XboxNative => "xbox-native",
            Kind::ContainerUnknown => "container-unknown",
            Kind::Elf => "elf",
            Kind::Rdna2Raw => "rdna2-raw",
            Kind::Unknown => "unknown",
        }
    }
}

pub fn classify(blob: &[u8]) -> (Kind, Vec<Part>) {
    if let Some(parts) = container_parts(blob) {
        let has = |n: &str| parts.iter().any(|p| p.fourcc == n);
        let kind = if has("DXIL") && xbox::is_xbox(blob) {
            Kind::XboxNative
        } else if has("DXIL") {
            Kind::Dxil
        } else if has("SHEX") || has("SHDR") {
            Kind::Dxbc
        } else if has("RTS0") && parts.iter().all(|p| KNOWN_PARTS.contains(&p.fourcc.as_str())) {
            Kind::Dxbc // root signature container
        } else {
            Kind::ContainerUnknown
        };
        return (kind, parts);
    }
    if blob.len() >= 4 && &blob[0..4] == b"\x7fELF" {
        return (Kind::Elf, Vec::new());
    }
    if !rdna2::find_programs(blob).is_empty() {
        return (Kind::Rdna2Raw, Vec::new());
    }
    (Kind::Unknown, Vec::new())
}

/// Disassembly of every plausible RDNA2 program in the blob's unknown regions.
pub fn rdna2_listing(blob: &[u8]) -> String {
    if let Some(sh) = xbox::parse(blob) {
        return xbox::listing(&sh);
    }
    let mut regions: Vec<(String, usize, usize)> = Vec::new();
    match container_parts(blob) {
        Some(parts) => {
            for p in parts.iter().filter(|p| !KNOWN_PARTS.contains(&p.fourcc.as_str())) {
                regions.push((format!("part {}", p.fourcc), p.offset, p.size));
            }
        }
        None => regions.push(("blob".into(), 0, blob.len())),
    }
    let mut out = String::new();
    for (label, off, size) in regions {
        let data = &blob[off..off + size];
        for (start, len) in rdna2::find_programs(data) {
            out.push_str(&format!("; {label}: program at +{:#x} ({} bytes)\n", off + start, len));
            for l in rdna2::disassemble(&data[start..start + len], (off + start) as u64, false) {
                let hex: Vec<String> = l.bytes.iter().map(|b| format!("{b:02x}")).collect();
                out.push_str(&format!("  {:08x}: {:<48} {}\n", l.pc, l.text, hex.join(" ")));
            }
        }
    }
    out
}

pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ShaderStats {
    pub unique: usize,
    pub submitted: usize,
    pub by_kind: BTreeMap<String, usize>,
    pub unknown_parts: BTreeMap<String, usize>,
    pub by_stage: BTreeMap<String, usize>,
    pub rejected_pointers: usize,
    pub dump_dir: String,
    /// console containers: hardware stages of their FSSM variants (VS, PS, NGG, ES, GS+VS, CS …)
    #[serde(default)]
    pub xbox_hw_stages: BTreeMap<String, usize>,
    /// console containers whose FSSM could not be fully parsed
    #[serde(default)]
    pub xbox_errors: usize,
    /// RDNA2 instructions decoded from unique console shaders (recompiler priorities)
    #[serde(default)]
    pub rdna2_instructions: u64,
    #[serde(default)]
    pub rdna2_undecoded_dwords: u64,
    /// (mnemonic, count), most frequent first
    #[serde(default)]
    pub rdna2_mnemonics: Vec<(String, u64)>,
    /// recompiler at capture: programs translated (HLSL next to the dumps), complete ones,
    /// by mode, mnemonics still missing
    #[serde(default)]
    pub recomp_programs: usize,
    #[serde(default)]
    pub recomp_complete: usize,
    #[serde(default)]
    pub recomp_modes: BTreeMap<String, usize>,
    #[serde(default)]
    pub recomp_unsupported: BTreeMap<String, usize>,
    /// DXC on the translations (`XCL_RECOMP_DXC=1`; cached in `recomp::dxc::cache_dir()`)
    #[serde(default)]
    pub recomp_dxc_ok: usize,
    #[serde(default)]
    pub recomp_dxc_cached: usize,
    #[serde(default)]
    pub recomp_dxc_failed: usize,
}

struct State {
    seen: HashSet<u64>,
    stats: ShaderStats,
    insts: xbox::InstStats,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn dump_dir() -> PathBuf {
    crate::state::data_dir().join("shaders")
}

/// Is `[p, p+len)` committed, readable memory? (Game-provided pointers may be garbage
/// when our guess of a console struct layout is wrong.)
pub fn readable(p: usize, len: usize) -> bool {
    readable_with(p, len, true)
}

/// `readable` without the pages this thread found readable a moment ago (`SEEN`): for memory
/// that may have been freed since, as the next page of an instruction (`eh::code_bytes_at`).
pub fn readable_fresh(p: usize, len: usize) -> bool {
    readable_with(p, len, false)
}

fn readable_with(p: usize, len: usize, recent: bool) -> bool {
    if p == 0 || len == 0 || len > 256 << 20 {
        return false;
    }
    let end = match p.checked_add(len) {
        Some(e) => e,
        None => return false,
    };
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(p, len)) {
        return true;
    }
    let (first, last) = (p >> 12, (end - 1) >> 12);
    let short = last - first < SEEN_PAGES;
    if recent && short && seen_readable(first, last) {
        return true;
    }
    // Under Wine the questions below are system calls under its virtual-memory lock:
    // 20-130 µs each, more while another thread commits or frees memory, and a retail game's
    // world asked ~10,000 times a second. A range of up to 16 pages is read instead, a byte a
    // page (`IsBadReadPtr`: it catches the page fault of a page that cannot be read).
    // `XCL_READABLE_PROBE=0`: asked of the system, as before.
    if last - first < 16 && probe_on() {
        PROBING.with(|f| f.set(true));
        let ok = unsafe { windows_sys::Win32::System::Memory::IsBadReadPtr(p as *const _, len) } == 0;
        PROBING.with(|f| f.set(false));
        if ok && short {
            note_readable(first, last);
        }
        return ok;
    }
    let ok = readable_asked(p, end);
    if ok && short {
        note_readable(first, last);
    }
    ok
}

/// Whether `readable` reads ranges of up to 16 pages instead of asking the system: under
/// Wine, unless `XCL_READABLE_PROBE=0`.
fn probe_on() -> bool {
    static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PROBE.get_or_init(|| {
        let on = crate::util::under_wine() && std::env::var("XCL_READABLE_PROBE").map_or(true, |v| v.trim() != "0");
        if on {
            unsafe { windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(1, Some(probe_guard_page)) };
        }
        on
    })
}

thread_local! {
    /// This thread is in `readable`'s `IsBadReadPtr`.
    static PROBING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Wine's `IsBadReadPtr` catches access violations only: a guard page it read ended the
/// process (`STATUS_GUARD_PAGE_VIOLATION` unhandled). During the probe the page is made a guard
/// page again (the fault took that away) and the fault is passed on as an access violation, so
/// the probe answers "not readable" as on Windows and the guard stays.
unsafe extern "system" fn probe_guard_page(info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS) -> i32 {
    use windows_sys::Win32::System::Memory::{VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, PAGE_GUARD};
    const GUARD_PAGE_VIOLATION: i32 = 0x8000_0001u32 as i32;
    const ACCESS_VIOLATION: i32 = 0xC000_0005u32 as i32;
    let rec = &mut *(*info).ExceptionRecord;
    if rec.ExceptionCode == GUARD_PAGE_VIOLATION && rec.NumberParameters >= 2 && PROBING.with(|f| f.get()) {
        let page = (rec.ExceptionInformation[1] & !0xFFF) as *const std::ffi::c_void;
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        let mut old = 0u32;
        if VirtualQuery(page, &mut mbi, std::mem::size_of_val(&mbi)) != 0 {
            VirtualProtect(page, 1 << 12, mbi.Protect | PAGE_GUARD, &mut old);
        }
        rec.ExceptionCode = ACCESS_VIOLATION;
    }
    0 // EXCEPTION_CONTINUE_SEARCH
}

/// Ranges of up to this many pages are remembered when readable (`SEEN`).
const SEEN_PAGES: usize = 4;
/// How long a page found readable is taken as readable without asking again.
const SEEN_FOR: std::time::Duration = std::time::Duration::from_millis(250);

thread_local! {
    /// Pages this thread found readable a moment ago (page number, when), a ring. Translating a
    /// retail game's world frames asked ~100,000 times per 2 s (objects' vtables in `is_adopted`,
    /// the title's view descs), ~20 µs each under Wine (a system call under its virtual-memory lock):
    /// 1.5–2.8 s of the title's threads per 2 s.
    /// shortcut: a page freed within `SEEN_FOR` of a check still reads as readable on this thread;
    /// the checks guard against garbage pointers, which point at pages never seen readable.
    static SEEN: std::cell::RefCell<([(usize, std::time::Instant); 64], usize)> =
        std::cell::RefCell::new(([(usize::MAX, std::time::Instant::now()); 64], 0));
}

fn seen_readable(first: usize, last: usize) -> bool {
    SEEN.with(|s| {
        let s = s.borrow();
        (first..=last).all(|pg| s.0.iter().any(|&(q, at)| q == pg && at.elapsed() < SEEN_FOR))
    })
}

fn note_readable(first: usize, last: usize) {
    let now = std::time::Instant::now();
    SEEN.with(|s| {
        let mut s = s.borrow_mut();
        for pg in first..=last {
            let at = s.1;
            s.0[at] = (pg, now);
            s.1 = (at + 1) % s.0.len();
        }
    })
}

/// `readable` asked of the system: the working set first, `VirtualQuery` when it cannot say.
fn readable_asked(p: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS};
    if let Some(answer) = pages_readable(p, end) {
        return answer;
    }
    let mut cur = p;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = unsafe { VirtualQuery(cur as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
        if n == 0 || mbi.State != MEM_COMMIT || mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 || mbi.Protect == 0 {
            return false;
        }
        cur = mbi.BaseAddress as usize + mbi.RegionSize;
    }
    true
}

/// Whether each page of `[p, end)` is readable, asked of the working set – the kernel answers for
/// up to 512 pages in one call that touches none of them. `readable` asks this first because
/// `VirtualQuery` sizes the whole region around an address by walking it: in a title's 2 GB block
/// of committed memory that took 20–30 ms a call. A retail game's streaming thread, which has
/// the page arrays of its DirectStorage requests checked, spent most of the world load there;
/// so did the submits, which check each texture of more than 64 KiB that was written (30–50 ms
/// of a frame).
///
/// None: ask `VirtualQuery` – the working set cannot be asked, or a range of up to 16 pages (a
/// title's pointer that may be garbage) has a page that is not resident (never touched, paged
/// out, guard, reserved, no access). In a longer range such a page is read by the kernel (a
/// byte of it): a committed one is then resident, as the caller's reads would make it, any other
/// is not readable.
fn pages_readable(p: usize, end: usize) -> Option<bool> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Memory::{PAGE_GUARD, PAGE_NOACCESS};
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const SMALL: usize = 16;
    const BATCH: usize = 512;
    let (first, last) = (p >> 12, (end - 1) >> 12);
    let small = last - first < SMALL;
    let mut ws: [PSAPI_WORKING_SET_EX_INFORMATION; BATCH] = unsafe { std::mem::zeroed() };
    let mut at = first;
    while at <= last {
        let n = (last - at + 1).min(BATCH);
        for (k, e) in ws[..n].iter_mut().enumerate() {
            e.VirtualAddress = ((at + k) << 12) as *mut std::ffi::c_void;
            e.VirtualAttributes.Flags = 0;
        }
        let size = (n * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32;
        if unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), ws.as_mut_ptr() as *mut std::ffi::c_void, size) } == 0 {
            return None;
        }
        for e in &ws[..n] {
            // PSAPI_WORKING_SET_EX_BLOCK: bit 0 Valid, bits 4-14 Win32Protection
            let flags = unsafe { e.VirtualAttributes.Flags };
            let protect = ((flags >> 4) & 0x7FF) as u32;
            if flags & 1 != 0 && protect != 0 && protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 {
                continue;
            }
            if small {
                return None;
            }
            let (mut byte, mut got) = (0u8, 0usize);
            let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), e.VirtualAddress as *const std::ffi::c_void, &mut byte as *mut u8 as *mut std::ffi::c_void, 1, &mut got) };
            if ok == 0 || got != 1 {
                return Some(false);
            }
        }
        at += n;
    }
    Some(true)
}

/// Record a shader blob submitted by the title. Safe on garbage pointers.
pub fn capture(stage: &str, ptr: usize, len: usize) {
    if ptr == 0 && len == 0 {
        return; // unused stage
    }
    let mut g = STATE.lock().unwrap();
    let st = g.get_or_insert_with(|| State { seen: HashSet::new(), stats: ShaderStats::default(), insts: xbox::InstStats::default() });
    st.stats.submitted += 1;
    if !readable(ptr, len) {
        st.stats.rejected_pointers += 1;
        if st.stats.rejected_pointers <= 5 {
            crate::xlog_warn!("[SHADER] {stage}: invalid pointer/size ({ptr:#x}, {len}) – different struct layout on the console?");
        }
        return;
    }
    let blob = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec();
    let h = fnv64(&blob);
    *st.stats.by_stage.entry(stage.to_string()).or_default() += 1;
    if !st.seen.insert(h) {
        return;
    }
    st.stats.unique += 1;
    let (kind, parts) = classify(&blob);
    *st.stats.by_kind.entry(kind.label().to_string()).or_default() += 1;
    for p in parts.iter().filter(|p| !KNOWN_PARTS.contains(&p.fourcc.as_str())) {
        *st.stats.unknown_parts.entry(p.fourcc.clone()).or_default() += 1;
    }
    drop(g);
    if !dumps_enabled(stage, h) {
        let names: Vec<&str> = parts.iter().map(|p| p.fourcc.as_str()).collect();
        crate::xlog_info!("[SHADER] {stage} {h:016x}: {} B, format {}, parts [{}]", blob.len(), kind.label(), names.join(","));
        if kind == Kind::XboxNative {
            account_xbox(&blob, h);
        }
        return;
    }
    let dir = dump_dir();
    let _ = std::fs::create_dir_all(&dir);
    let base = dir.join(format!("{h:016x}_{stage}"));
    let _ = std::fs::write(base.with_extension("bin"), &blob);
    let names: Vec<&str> = parts.iter().map(|p| p.fourcc.as_str()).collect();
    crate::xlog_info!("[SHADER] {stage} {h:016x}: {} B, format {}, parts [{}]", blob.len(), kind.label(), names.join(","));
    if kind == Kind::XboxNative {
        capture_xbox(&blob, &base, h);
    } else if matches!(kind, Kind::ContainerUnknown | Kind::Rdna2Raw | Kind::Unknown) {
        let listing = rdna2_listing(&blob);
        if !listing.is_empty() {
            let _ = std::fs::write(base.with_extension("rdna2.txt"), listing);
            crate::xlog_info!("[SHADER] {h:016x}: RDNA2 code found – disassembly in {}", base.with_extension("rdna2.txt").display());
        }
    }
}

/// `XCL_DUMP_SHADERS=1`: every shader the title submits is written to the title's `shaders`
/// folder – the blob, a console program's FSSMs, its disassembly and its HLSL (translated once
/// more for that: the report's `recomp_*` counts come from it). Off by default: a retail game's
/// ~17 000 dumps took 4.8 GB, and translating each again took its loading threads ~0.1 s a
/// program at every start.
/// `XCL_DUMP_SHADERS=ps,0123456789abcdef`: only the shaders of those stages (as the log names
/// them: `vs`, `ps`, `cs` …) and the ones with those dump hashes (16 hex digits).
fn dumps_enabled(stage: &str, h: u64) -> bool {
    static WANTED: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    dump_wanted(WANTED.get_or_init(|| dump_list(&std::env::var("XCL_DUMP_SHADERS").unwrap_or_default())), stage, h)
}

/// The items of an `XCL_DUMP_SHADERS` value.
fn dump_list(v: &str) -> Vec<String> {
    v.split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect()
}

/// Whether a list of `XCL_DUMP_SHADERS` items asks for the shader of `stage` with dump hash `h`.
fn dump_wanted(list: &[String], stage: &str, h: u64) -> bool {
    list.iter().any(|s| s == "1" || s.eq_ignore_ascii_case(stage) || (s.len() == 16 && u64::from_str_radix(s, 16) == Ok(h)))
}

/// Console container without dumps: its FSSM variants and instructions accounted.
fn account_xbox(blob: &[u8], h: u64) {
    let Some(sh) = xbox::parse(blob) else { return };
    let mut insts = xbox::InstStats::default();
    insts.add_shader(&sh);
    let failed = !sh.notes.is_empty() || sh.fssm.iter().any(|f| f.error.is_some() || f.programs.is_empty());
    if let Some(st) = STATE.lock().unwrap().as_mut() {
        for f in &sh.fssm {
            *st.stats.xbox_hw_stages.entry(f.stage_label()).or_default() += 1;
        }
        st.stats.xbox_errors += failed as usize;
        st.insts.merge(&insts);
    }
    if failed {
        crate::xlog_info!("[SHADER] {h:016x}: console container {} – WARNING: incomplete parse (xcl disasm)", sh.stage);
    }
}

/// Console container: account its FSSM variants and instructions, write the
/// decompressed FSSMs and the disassembly next to the blob.
fn capture_xbox(blob: &[u8], base: &std::path::Path, h: u64) {
    let Some(sh) = xbox::parse(blob) else { return };
    let mut insts = xbox::InstStats::default();
    insts.add_shader(&sh);
    let failed = !sh.notes.is_empty() || sh.fssm.iter().any(|f| f.error.is_some() || f.programs.is_empty());
    if let Some(st) = STATE.lock().unwrap().as_mut() {
        for f in &sh.fssm {
            *st.stats.xbox_hw_stages.entry(f.stage_label()).or_default() += 1;
        }
        st.stats.xbox_errors += failed as usize;
        st.insts.merge(&insts);
    }
    for (i, f) in sh.fssm.iter().enumerate() {
        let _ = std::fs::write(base.with_extension(format!("fssm{i}.bin")), &f.data);
    }
    let listing = xbox::listing(&sh);
    let _ = std::fs::write(base.with_extension("rdna2.txt"), &listing);
    recompile(blob, &sh, base);
    let stages: Vec<String> = sh.fssm.iter().map(|f| f.stage_label()).collect();
    crate::xlog_info!(
        "[SHADER] {h:016x}: console container {} SM {}.{}, FSSM [{}], {} RDNA2 instructions{}",
        sh.stage,
        sh.shader_model.0,
        sh.shader_model.1,
        stages.join(", "),
        insts.instructions,
        if failed { " – WARNING: incomplete parse (xcl disasm)" } else { "" }
    );
}

/// Translate every program of a captured console shader (mode from its interface) and
/// write the HLSL next to the dumps; with `XCL_RECOMP_DXC=1` also compile it (cached).
fn recompile(blob: &[u8], sh: &xbox::XboxShader, base: &std::path::Path) {
    use crate::gpu::recomp;
    let dxc = std::env::var("XCL_RECOMP_DXC").is_ok_and(|v| v == "1");
    for (i, f) in sh.fssm.iter().enumerate() {
        let mode = recomp::interface(blob, sh, f).unwrap_or(recomp::Mode::Kernel);
        let (label, profile) = match &mode {
            recomp::Mode::Vertex(_) => ("VS", "vs_6_6"),
            recomp::Mode::Pixel(_) => ("PS", "ps_6_6"),
            recomp::Mode::Mesh(_) => ("MS", "cs_6_6"),
            recomp::Mode::Geometry(_) => ("GS", "cs_6_6"),
            recomp::Mode::Compute(_) => ("CS", "cs_6_6"),
            recomp::Mode::Kernel => ("kernel", "cs_6_6"),
        };
        for (j, p) in f.programs.iter().enumerate() {
            let code = &f.data[p.offset..p.offset + p.len];
            let opt = recomp::Options { mode: mode.clone(), wave64: recomp::fssm_wave64(f, code), ..Default::default() };
            let Ok(t) = recomp::translate(code, &opt) else { continue };
            let _ = std::fs::write(base.with_extension(format!("fssm{i}.p{j}.hlsl")), &t.hlsl);
            let built = if dxc && t.complete() { Some(recomp::dxc::compile_cached(&t.hlsl, "main", profile)) } else { None };
            if let Some(Err(e)) = &built {
                crate::xlog_warn!("[SHADER] DXC rejected the translation {}: {}", base.display(), e.lines().next().unwrap_or(""));
            }
            if let Some(st) = STATE.lock().unwrap().as_mut() {
                let s = &mut st.stats;
                s.recomp_programs += 1;
                s.recomp_complete += t.complete() as usize;
                *s.recomp_modes.entry(label.into()).or_default() += 1;
                for (m, n) in &t.unsupported {
                    *s.recomp_unsupported.entry(m.clone()).or_default() += n;
                }
                match &built {
                    Some(Ok((_, true))) => s.recomp_dxc_cached += 1,
                    Some(Ok(_)) => s.recomp_dxc_ok += 1,
                    Some(Err(_)) => s.recomp_dxc_failed += 1,
                    None => {}
                }
            }
        }
    }
}

pub fn stats() -> ShaderStats {
    let g = STATE.lock().ok();
    let mut s = g.as_ref().and_then(|g| g.as_ref().map(|st| st.stats.clone())).unwrap_or_default();
    if let Some(st) = g.as_ref().and_then(|g| g.as_ref()) {
        s.rdna2_instructions = st.insts.instructions;
        s.rdna2_undecoded_dwords = st.insts.undecoded;
        s.rdna2_mnemonics = st.insts.sorted();
    }
    s.dump_dir = dump_dir().to_string_lossy().into_owned();
    s
}

#[cfg(test)]
mod tests {
    /// Ranges of any size are answered page by page: committed pages (touched or not) are
    /// readable, a reserved or no-access page anywhere in the range is not.
    #[test]
    fn readable_answers_for_long_ranges_page_by_page() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE};
        const PAGES: usize = 1200;
        unsafe {
            // (reserved pages of its own behind the committed ones)
            let base = VirtualAlloc(std::ptr::null(), (PAGES + 16) << 12, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert!(!VirtualAlloc(base as *const _, PAGES << 12, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert!(super::readable(base, PAGES << 12), "committed, none touched");
            for k in (0..PAGES).step_by(3) {
                *((base + (k << 12)) as *mut u8) = 1;
            }
            assert!(super::readable(base, PAGES << 12), "committed, some touched");
            assert!(super::readable(base + 100, (PAGES << 12) - 100), "not from a page's start");
            assert!(!super::readable(base, (PAGES << 12) + 1), "past the end");
            let hole = base + (700 << 12);
            VirtualFree(hole as *mut _, 1 << 12, MEM_DECOMMIT);
            assert!(!super::readable(base, PAGES << 12), "a reserved page inside");
            assert!(super::readable(base, 700 << 12), "up to the hole");
            assert!(super::readable(hole + (1 << 12), 499 << 12), "behind the hole");
            assert!(!super::readable(hole, 16), "the hole itself");
            let mut old = 0;
            assert_ne!(VirtualProtect((base + (40 << 12)) as *const _, 1 << 12, PAGE_NOACCESS, &mut old), 0);
            assert!(!super::readable(base, 100 << 12), "a no-access page inside");
            assert!(super::readable(base, 40 << 12));
            VirtualFree(base as *mut _, 0, MEM_RELEASE);
        }
    }

    use super::*;

    fn container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let header = 32 + parts.len() * 4;
        let mut v = vec![0u8; header];
        v[0..4].copy_from_slice(b"DXBC");
        v[24..28].copy_from_slice(&1u32.to_le_bytes());
        v[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
        for (i, (cc, data)) in parts.iter().enumerate() {
            let off = v.len() as u32;
            v[32 + i * 4..36 + i * 4].copy_from_slice(&off.to_le_bytes());
            v.extend_from_slice(*cc);
            v.extend_from_slice(&(data.len() as u32).to_le_bytes());
            v.extend_from_slice(data);
        }
        let total = v.len() as u32;
        v[24..28].copy_from_slice(&total.to_le_bytes());
        v
    }

    #[test]
    fn classifies_containers() {
        assert_eq!(classify(&container(&[(b"ISG1", b"x"), (b"DXIL", b"bitcode")])).0, Kind::Dxil);
        assert_eq!(classify(&container(&[(b"SHEX", b"code")])).0, Kind::Dxbc);
        // a made-up console part with RDNA2 code inside: s_mov_b32 s0, s1 ; s_endpgm
        let code = [0x01u8, 0x03, 0x80, 0xbe, 0x00, 0x00, 0x81, 0xbf];
        let blob = container(&[(b"XBOX", &code)]);
        let (k, parts) = classify(&blob);
        assert_eq!(k, Kind::ContainerUnknown);
        assert_eq!(parts[0].fourcc, "XBOX");
        let listing = rdna2_listing(&blob);
        assert!(listing.contains("s_mov_b32 s0, s1"), "{listing}");
        assert!(listing.contains("s_endpgm"));
        assert_eq!(classify(&code).0, Kind::Rdna2Raw);
        assert_eq!(classify(b"\x7fELF....").0, Kind::Elf);
        assert_eq!(classify(b"random junk!").0, Kind::Unknown);
    }

    #[test]
    fn classifies_and_lists_console_containers() {
        use crate::gpu::xbox::tests::{fssm, sample, CODE};
        let blob = sample(0, &[fssm(&CODE, &[(5, 1)])]);
        let (k, parts) = classify(&blob);
        assert_eq!(k, Kind::XboxNative);
        assert!(parts.iter().any(|p| p.fourcc == "DXIL"));
        let l = rdna2_listing(&blob);
        assert!(l.contains("FSSM #0 [PS]") && l.contains("v_add_f32"), "{l}");
    }

    /// `XCL_DUMP_SHADERS`: `1` is every shader, a stage name that stage, 16 hex digits one
    /// shader – and a hash is not taken for a stage, nor "1" for a hash.
    #[test]
    fn a_dump_list_names_stages_and_hashes() {
        let h = 0x0942_b913_6a89_143du64;
        assert!(dump_wanted(&dump_list("1"), "cs", h));
        assert!(!dump_wanted(&dump_list(""), "ps", h));
        assert!(!dump_wanted(&dump_list("0"), "ps", 0));
        let ps = dump_list(" PS , 0942b9136a89143d ");
        assert!(dump_wanted(&ps, "ps", 7), "a stage");
        assert!(dump_wanted(&ps, "vs", h), "a hash, whatever its stage");
        assert!(!dump_wanted(&ps, "vs", h + 1));
        assert!(!dump_wanted(&dump_list("0000000000000001"), "vs", 7));
        assert!(dump_wanted(&dump_list("0000000000000001"), "vs", 1));
        assert!(!dump_wanted(&dump_list("1x"), "vs", 1), "not every shader");
    }

    #[test]
    fn readable_rejects_garbage() {
        assert!(!readable(0, 10));
        assert!(!readable(0x10, 10));
        let v = vec![1u8; 100];
        assert!(readable(v.as_ptr() as usize, 100));
    }

    #[test]
    fn readable_answers_from_the_working_set_without_touching_pages() {
        use windows_sys::Win32::System::Memory::*;
        unsafe {
            // 0, 1 written; 2 no access; 3 guard; 4 reserved; 5, 6 committed and never touched
            let base = VirtualAlloc(std::ptr::null(), 7 << 12, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert_ne!(base, 0);
            let page = |k: usize| base + (k << 12);
            for (k, protect) in [(0, PAGE_READWRITE), (1, PAGE_READWRITE), (2, PAGE_NOACCESS), (3, PAGE_READWRITE | PAGE_GUARD), (5, PAGE_READWRITE), (6, PAGE_READONLY)] {
                assert!(!VirtualAlloc(page(k) as *const _, 1 << 12, MEM_COMMIT, protect).is_null());
            }
            std::ptr::write_volatile(page(0) as *mut u8, 1);
            std::ptr::write_volatile(page(1) as *mut u8, 1);
            // written pages: answered from the working set
            assert_eq!(pages_readable(page(0), page(2)), Some(true));
            assert!(readable(page(0), 2 << 12));
            assert!(readable(page(0) + 100, 8));
            // untouched pages: the region walk answers, and they stay untouched
            assert_eq!(pages_readable(page(5), page(5) + 8), None);
            assert!(readable(page(5), 8));
            assert!(readable(page(6), 1 << 12));
            // under Wine `readable` reads a byte of each page instead (`probe_on`): resident then
            assert_eq!(pages_readable(page(5), page(7)), if probe_on() { Some(true) } else { None });
            assert_eq!(pages_readable(page(3), page(3) + 8), None, "a guard page is not touched to answer");
            assert!(!readable(page(2), 8));
            assert!(!readable(page(3), 8));
            assert!(!readable(page(4), 8));
            // asking about a guard page leaves it a guard page
            let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(VirtualQuery(page(3) as *const _, &mut mbi, std::mem::size_of_val(&mbi)), 0);
            assert_ne!(mbi.Protect & PAGE_GUARD, 0);
            // a range is readable only when each of its pages is
            assert!(!readable(page(1) + 0xFF0, 0x20));
            assert!(!readable(page(4) + 0xFF0, 0x20));
            assert!(!readable(page(0), 20 << 12));
            VirtualFree(base as *mut _, 0, MEM_RELEASE);
        }
        // a range of many pages: the working set too
        let v = vec![1u8; 32 << 12];
        assert_eq!(pages_readable(v.as_ptr() as usize, v.as_ptr() as usize + v.len()), Some(true));
        assert!(readable(v.as_ptr() as usize, v.len()));
    }
}
