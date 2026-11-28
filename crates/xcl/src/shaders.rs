//! `xcl shaders <file|folder> [--top N] [--json] [--recomp] [--compile N]` – statistics over
//! a shader corpus: formats, stages, FSSM variants and the RDNA2 instruction histogram that
//! sets the recompiler's priorities. Inputs: a single blob, a folder (e.g. `<data>\shaders\`)
//! or an RPF9 shader pack (records found by scanning, see `gpu::pack`). `--recomp`
//! translates every program to HLSL (kernel mode) and reports what is still missing;
//! `--compile N` also compiles N of the complete translations with DXC. `--export-hlsl`
//! writes deterministic compute HLSL, and `--measure-pso` times one exact driver compile.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::gpu::{dxbc, pack, rdna2, recomp, shader, xbox};
use crate::hle::d3d12x::gfx_records;
pub(crate) mod pso_corpus;
pub(crate) mod pipeline_library;
mod milestone;
mod warm_blobs;
mod warm_pack;
mod warm_library;
pub(crate) mod warm_replay;

pub(crate) fn warm_settings_key() -> u64 { warm_settings(std::env::vars_os()) }

// Driver and cache directory are keyed separately. Worker PIDs, title paths, scheduling
// and observation settings do not affect the recipe the driver validates.
fn warm_settings(vars: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>) -> u64 {
    let mut settings = Vec::new();
    for (name, value) in vars {
        let name = name.to_string_lossy().to_ascii_uppercase();
        if !name.starts_with("XCL_") { continue; }
        if matches!(name.as_str(),
            "XCL_HOST" | "XCL_DATA_ROOT" | "XCL_ADAPTER" | "XCL_DXIL_CACHE" | "XCL_UMA_MB" |
            "XCL_PRECOMPILE_PARENT" | "XCL_PREPARATION_PARENT" | "XCL_PREPARATION_MENU" |
            "XCL_PRECOMPILE_SLICE" | "XCL_PRECOMPILE_COVERAGE_MAP" | "XCL_PRECOMPILE_COVERAGE_DIR" |
            "XCL_DXC_THREADS" | "XCL_PIPELINE_THREADS" | "XCL_PREPARE_WORKERS" | "XCL_PIPELINE_LIBRARY_OUT" | "XCL_PREPARE" |
            "XCL_PREPARE_HISTORY_PRESENT" | "XCL_PREPARE_HISTORY_SECONDS" |
            "XCL_PREPARE_GRAPHICS" | "XCL_PREPARE_ALL_LIVE" | "XCL_RUN_SECONDS" | "XCL_PRESS_AT" |
            "XCL_PRESS_AFTER_POLL" | "XCL_ACTIVITY" | "XCL_SKIP_BINK" | "XCL_FRAME_DUMP" |
            "XCL_FRAME_DUMP_FULL" | "XCL_GPU_CULPRIT" | "XCL_DSTORAGE_LATE_PAGE_MS" | "XCL_DISPATCH_TILE_AXIS") {
            continue;
        }
        settings.push((name, value));
    }
    settings.sort();
    warm_blobs::digest(&settings)
}

#[cfg(test)]
#[test]
fn warm_settings_ignore_worker_identity_but_keep_backend_changes() {
    let key = |entries: &[(&str, &str)]| warm_settings(entries.iter().map(|(k, v)| ((*k).into(), (*v).into())));
    let offline = key(&[("XCL_DIRECT_DXIL", "1")]);
    let worker = key(&[("XCL_PRECOMPILE_PARENT", "812"), ("XCL_DXC_THREADS", "2"), ("xcl_direct_dxil", "1")]);
    assert_eq!(offline, worker);
    assert_ne!(offline, key(&[("xcl_direct_dxil", "all")]));
}

/// `--iface`: how console programs connect to their DXBC signatures (hypothesis checks for
/// the VS/PS interface of the recompiler).
#[derive(Default)]
struct IfaceAgg {
    ps: usize,
    ps_in_regs: usize,
    ps_eq_regs: usize,
    ps_examples: Vec<String>,
    vs: usize,
    vs_param_in_regs: usize,
    vs_param_eq: usize,
    vs_param_examples: Vec<String>,
    vs_inputs: usize,
    vs_in_vec4: usize,
    vs_in_packed: usize,
    vs_input_examples: Vec<String>,
    /// attr / param index = rank among the non-system elements (register order)
    ps_rank: usize,
    vs_rank: usize,
    /// (hw stage, smallest base B with inputs ⊆ {B + 4·register + c}) → programs
    vs_base: BTreeMap<String, usize>,
    /// PS: SPI_PS_INPUT_ENA at FSSM +0x248 explains the I/J VGPRs of v_interp / the initial VGPRs read
    ps_ena_ij: usize,
    ps_ena_reads: usize,
    ps_ena_examples: Vec<String>,
    /// hardware stage → (programs, VGPR read before written → programs, SGPR → programs, export target → programs)
    inputs: BTreeMap<String, StageInputs>,
}

#[derive(Default)]
struct StageInputs {
    programs: usize,
    vgpr: BTreeMap<u64, usize>,
    sgpr: BTreeMap<u64, usize>,
    exports: BTreeMap<u64, usize>,
    sendmsg: BTreeMap<u64, usize>,
    /// PSV0: "kind / max verts / max prims / topology / numthreads / groupshared" → programs
    psv: BTreeMap<String, usize>,
    /// first instruction reading an input register (v0–v7, s0–s3) → programs
    first_use: BTreeMap<String, usize>,
}

impl IfaceAgg {
    fn merge(&mut self, o: IfaceAgg) {
        self.ps += o.ps;
        self.ps_in_regs += o.ps_in_regs;
        self.ps_eq_regs += o.ps_eq_regs;
        self.vs += o.vs;
        self.vs_param_in_regs += o.vs_param_in_regs;
        self.vs_param_eq += o.vs_param_eq;
        self.vs_inputs += o.vs_inputs;
        self.vs_in_vec4 += o.vs_in_vec4;
        self.vs_in_packed += o.vs_in_packed;
        self.ps_rank += o.ps_rank;
        self.ps_ena_ij += o.ps_ena_ij;
        self.ps_ena_reads += o.ps_ena_reads;
        self.ps_ena_examples.extend(o.ps_ena_examples);
        self.ps_ena_examples.truncate(8);
        self.vs_rank += o.vs_rank;
        for (k, v) in o.vs_base {
            *self.vs_base.entry(k).or_default() += v;
        }
        for (a, b) in [(&mut self.ps_examples, o.ps_examples), (&mut self.vs_param_examples, o.vs_param_examples), (&mut self.vs_input_examples, o.vs_input_examples)] {
            a.extend(b);
            a.truncate(8);
        }
        for (k, x) in o.inputs {
            let e = self.inputs.entry(k).or_default();
            e.programs += x.programs;
            for (a, b) in [(&mut e.vgpr, x.vgpr), (&mut e.sgpr, x.sgpr), (&mut e.exports, x.exports), (&mut e.sendmsg, x.sendmsg)] {
                for (r, n) in b {
                    *a.entry(r).or_default() += n;
                }
            }
            for (r, n) in x.psv {
                *e.psv.entry(r).or_default() += n;
            }
            for (r, n) in x.first_use {
                *e.first_use.entry(r).or_default() += n;
            }
        }
    }

    fn add(&mut self, name: &str, blob: &[u8], sh: &xbox::XboxShader) {
        let (isg, osg) = dxbc::signatures(blob);
        let user = |s: &[dxbc::SigElement]| -> Vec<(u32, u8, String)> {
            s.iter().filter(|e| e.system_value == 0 && !e.name.starts_with("SV_")).map(|e| (e.register, e.mask, e.semantic())).collect()
        };
        let sig_txt = |s: &[dxbc::SigElement]| s.iter().map(|e| format!("{}@r{}:{:x}", e.semantic(), e.register, e.mask)).collect::<Vec<_>>().join(" ");
        for f in &sh.fssm {
            let Some(p) = f.programs.first() else { continue };
            let code = &f.data[p.offset..p.offset + p.len];
            let mut attrs = std::collections::BTreeSet::new();
            let mut ij = std::collections::BTreeSet::new();
            let mut params = std::collections::BTreeSet::new();
            let (mut read, mut written) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
            let (mut sread, mut swritten) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
            let (mut exps, mut msgs) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
            let mut first: Vec<String> = Vec::new();
            let mut off = 0;
            while off + 4 <= code.len() {
                let Ok(d) = rdna2::decode(&code[off..], off as u64) else { break };
                off += d.len;
                let n = d.inst.name;
                if n.starts_with("V_INTERP") {
                    let a = rdna2::field_value(d.enc, &d.words, "ATTR")
                        .or_else(|| rdna2::field_value(d.enc, &d.words, "SRC0").map(|v| v & 0x3f));
                    if let Some(a) = a {
                        attrs.insert(a as u32);
                    }
                    if n == "V_INTERP_P1_F32" || n == "V_INTERP_P2_F32" {
                        if let Some(v) = rdna2::field_value(d.enc, &d.words, "VSRC") {
                            ij.insert(v);
                        }
                    }
                }
                if n == "EXP" {
                    let t = rdna2::field_value(d.enc, &d.words, "TGT").unwrap_or(0);
                    if (32..64).contains(&t) {
                        params.insert((t - 32) as u32);
                    }
                }
                let (r, w) = rdna2::vgpr_uses(&d);
                for v in r {
                    if !written.contains(&v) {
                        if v < 8 && !read.contains(&v) {
                            first.push(format!("v{v}: {}", rdna2::format(&d)));
                        }
                        read.insert(v);
                    }
                }
                written.extend(w);
                let (r, w) = rdna2::sgpr_uses(&d);
                for v in r {
                    if !swritten.contains(&v) {
                        if v < 4 && !sread.contains(&v) {
                            first.push(format!("s{v}: {}", rdna2::format(&d)));
                        }
                        sread.insert(v);
                    }
                }
                swritten.extend(w);
                if n == "EXP" {
                    exps.insert(rdna2::field_value(d.enc, &d.words, "TGT").unwrap_or(0));
                }
                if n == "S_SENDMSG" || n == "S_SENDMSGHALT" {
                    msgs.insert(rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) & 0xF);
                }
            }
            {
                let hw0 = f.stages.first().map(|s| s.hw).unwrap_or(99);
                let e = self.inputs.entry(format!("{} {}", sh.stage, xbox::hw_stage_name(hw0))).or_default();
                e.programs += 1;
                for &v in &read {
                    *e.vgpr.entry(v).or_default() += 1;
                }
                for &v in &sread {
                    *e.sgpr.entry(v).or_default() += 1;
                }
                for &t in &exps {
                    *e.exports.entry(t).or_default() += 1;
                }
                for &m in &msgs {
                    *e.sendmsg.entry(m).or_default() += 1;
                }
                let key = match dxbc::psv(blob) {
                    Some(p) if p.kind == Some(13) => format!(
                        "MS v{} p{} topo{} threads{:?} lds{} wave{}-{}",
                        p.ms_max_vertices, p.ms_max_primitives, p.ms_topology, p.numthreads.unwrap_or_default(), p.ms_groupshared, p.min_wave, p.max_wave
                    ),
                    Some(p) => format!("kind {:?} threads{:?} wave{}-{}", p.kind, p.numthreads, p.min_wave, p.max_wave),
                    None => "no PSV0".into(),
                };
                *e.psv.entry(key).or_default() += 1;
                for u in first {
                    *e.first_use.entry(u).or_default() += 1;
                }
            }
            let hw = f.stages.first().map(|s| s.hw).unwrap_or(99);
            if sh.stage == "PS" && hw == 5 && !attrs.is_empty() {
                self.ps += 1;
                let regs: std::collections::BTreeSet<u32> = user(&isg).iter().map(|x| x.0).collect();
                let inside = attrs.is_subset(&regs);
                self.ps_in_regs += inside as usize;
                self.ps_eq_regs += (attrs == regs) as usize;
                let n = user(&isg).len() as u32;
                self.ps_rank += attrs.iter().all(|&a| a < n) as usize;
                // VGPR layout of the enabled PS inputs (hardware order, sizes 2/2/2/3/2/2/2/1…)
                let ena = f.data.get(0x248..0x24C).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
                let sizes = [2u64, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];
                let (mut at, mut pairs) = (0u64, std::collections::BTreeSet::new());
                for (bit, sz) in sizes.iter().enumerate() {
                    if ena >> bit & 1 == 1 {
                        if bit <= 6 && bit != 3 {
                            pairs.insert(at);
                            pairs.insert(at + 1);
                        }
                        at += sz;
                    }
                }
                let ij_ok = ij.is_subset(&pairs);
                let rd_ok = read.iter().all(|&v| v < at);
                self.ps_ena_ij += ij_ok as usize;
                self.ps_ena_reads += rd_ok as usize;
                if (!ij_ok || !rd_ok) && self.ps_ena_examples.len() < 8 {
                    self.ps_ena_examples.push(format!("{name}: ena {ena:#x} ({at} VGPR) I/J {ij:?} read before write {:?}", read.iter().take(12).collect::<Vec<_>>()));
                }
                if !inside && self.ps_examples.len() < 8 {
                    self.ps_examples.push(format!("{name}: attr {attrs:?} vs ISG1 {}", sig_txt(&isg)));
                }
            }
            if sh.stage == "VS" && (hw == 4 || hw == 2) {
                self.vs += 1;
                let regs: std::collections::BTreeSet<u32> = user(&osg).iter().map(|x| x.0).collect();
                let inside = params.is_subset(&regs);
                self.vs_param_in_regs += inside as usize;
                self.vs_param_eq += (params == regs) as usize;
                let n = user(&osg).len() as u32;
                let rank_ok = params.iter().all(|&a| a < n);
                self.vs_rank += rank_ok as usize;
                if !rank_ok && self.vs_param_examples.len() < 8 {
                    self.vs_param_examples.push(format!("RANK {name} [{}]: param {params:?} vs OSG1 {}", xbox::hw_stage_name(hw), sig_txt(&osg)));
                }

                // attribute VGPRs the fetch shader fills: read before written, above the 4 system VGPRs
                let ins: std::collections::BTreeSet<u64> = read.iter().copied().filter(|&v| v >= 4).collect();
                let u = user(&isg);
                if !ins.is_empty() && !u.is_empty() {
                    self.vs_inputs += 1;
                    let vec4: std::collections::BTreeSet<u64> =
                        u.iter().flat_map(|(r, m, _)| (0..4).filter(move |c| m >> c & 1 == 1).map(move |c| 4 + 4 * *r as u64 + c)).collect();
                    let mut packed = std::collections::BTreeSet::new();
                    let mut sorted = u.clone();
                    sorted.sort();
                    let mut next = 4u64;
                    for (_, m, _) in &sorted {
                        let n = 8 - m.leading_zeros() as u64; // highest component + 1
                        packed.extend(next..next + n);
                        next += n;
                    }
                    // system values with a register (SV_VertexID / SV_InstanceID) arrive like attributes
                    let sv_regs: Vec<u32> = isg.iter().filter(|e| matches!(e.system_value, 6 | 8)).map(|e| e.register).collect();
                    for b in [4u64, 9] {
                        let all: std::collections::BTreeSet<u64> =
                            u.iter().map(|x| x.0).chain(sv_regs.iter().copied()).flat_map(|r| (0..4).map(move |c| b + 4 * r as u64 + c)).collect();
                        if ins.is_subset(&all) {
                            *self.vs_base.entry(format!("{} matches v{b}+4r+c", xbox::hw_stage_name(hw))).or_default() += 1;
                        } else if b == 4 && hw == 4 && self.vs_input_examples.len() < 8 {
                            let extra: Vec<u64> = ins.difference(&all).copied().collect();
                            self.vs_input_examples.push(format!("v4 NO: {name}: reads {ins:?} outside: {extra:?} | ISG1 {}", sig_txt(&isg)));
                        }
                    }
                    *self.vs_base.entry(format!("{} total", xbox::hw_stage_name(hw))).or_default() += 1;
                    let a = ins.is_subset(&vec4);
                    let b = ins.is_subset(&packed);
                    self.vs_in_vec4 += a as usize;
                    self.vs_in_packed += b as usize;
let _ = b;
                }
            }
        }
    }
}

static HEADER_SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `xcl shaders --example MNEMONIC`
static EXAMPLE_FOR: std::sync::OnceLock<String> = std::sync::OnceLock::new();

#[derive(Default)]
struct RecompAgg {
    programs: usize,
    complete: usize,
    wave32: usize,
    errors: usize,
    error_examples: Vec<String>,
    /// mnemonic → (instructions, programs that use it)
    missing: HashMap<String, (usize, usize)>,
    /// programs whose only missing mnemonic is this one
    sole: HashMap<String, usize>,
    /// "mnemonic: reason" → programs
    why: HashMap<String, usize>,
    /// (name, HLSL, profile) of complete translations kept for the DXC check
    samples: Vec<(String, String, &'static str)>,
    /// programs translated in a graphics mode (VS / PS interface from the container)
    by_mode: BTreeMap<String, (usize, usize)>,
    hlsl_bytes: usize,
    /// VS / PS (the driver picks the wave size): wave32 programs (run split in 64-lane waves)
    /// and wave64 programs whose cross-lane reads break in 32-lane waves (kind → programs)
    graphics_wave32: usize,
    graphics_wave64: usize,
    wave_hazards: BTreeMap<String, usize>,
    /// … and the reads of lanes ≥ 32 the 32-lane form resolves (kind → programs)
    wave_resolved: BTreeMap<String, usize>,
    /// `--example MNEMONIC`: listing of the first program missing it, its HLSL and profile
    example: Option<(String, String, &'static str)>,
    /// `lane_masks` (direct DXIL): programs analysed, those with no guard left, vector
    /// instructions, those still guarded, lane branches, uniform branches; failures by reason
    lanes: [usize; 6],
    lanes_failed: BTreeMap<String, usize>,
    /// compute programs, per-lane DXIL (`lower_dxil`) vs #1426's per-instruction form: tried,
    /// per-lane ok, per-instruction ok, both ok, per-lane bytes and per-instruction bytes where
    /// both are ok; why the per-lane form refused (first words of the reason)
    cs_forms: [usize; 6],
    cs_lane_why: BTreeMap<String, usize>,
    /// survey of every refusal, per stage (CS, MS): tried, wave64, per-lane ok, per-instruction
    /// ok, both ok, per-lane refused only for instructions (no structural reason), the same with
    /// the per-instruction form ok, refused by the lane-mask analysis
    forms: BTreeMap<String, [usize; 8]>,
    /// "stage | reason" → programs (each reason once per program): every per-lane reason, the
    /// structural ones that are a program's only structural reason, the per-instruction reasons
    lane_every: BTreeMap<String, usize>,
    lane_sole: BTreeMap<String, usize>,
    inst_why: BTreeMap<String, usize>,
}

impl RecompAgg {
    fn merge(&mut self, o: RecompAgg, keep: usize) {
        self.programs += o.programs;
        self.complete += o.complete;
        self.wave32 += o.wave32;
        self.errors += o.errors;
        self.error_examples.extend(o.error_examples);
        self.error_examples.truncate(10);
        for (k, (a, b)) in o.missing {
            let e = self.missing.entry(k).or_default();
            e.0 += a;
            e.1 += b;
        }
        for (k, v) in o.sole {
            *self.sole.entry(k).or_default() += v;
        }
        for (k, v) in o.why {
            *self.why.entry(k).or_default() += v;
        }
        self.samples.extend(o.samples);
        self.samples.truncate(keep);
        if self.example.is_none() {
            self.example = o.example;
        }
        self.graphics_wave32 += o.graphics_wave32;
        self.graphics_wave64 += o.graphics_wave64;
        for (k, v) in o.wave_hazards {
            *self.wave_hazards.entry(k).or_default() += v;
        }
        for (k, v) in o.wave_resolved {
            *self.wave_resolved.entry(k).or_default() += v;
        }
        for (k, (a, b)) in o.by_mode {
            let e = self.by_mode.entry(k).or_default();
            e.0 += a;
            e.1 += b;
        }
        self.hlsl_bytes += o.hlsl_bytes;
        for (a, b) in self.lanes.iter_mut().zip(o.lanes) {
            *a += b;
        }
        for (k, v) in o.lanes_failed {
            *self.lanes_failed.entry(k).or_default() += v;
        }
        for (a, b) in self.cs_forms.iter_mut().zip(o.cs_forms) {
            *a += b;
        }
        for (k, v) in o.cs_lane_why {
            *self.cs_lane_why.entry(k).or_default() += v;
        }
        for (k, v) in o.forms {
            for (a, b) in self.forms.entry(k).or_default().iter_mut().zip(v) {
                *a += b;
            }
        }
        for (m, o) in [(&mut self.lane_every, o.lane_every), (&mut self.lane_sole, o.lane_sole), (&mut self.inst_why, o.inst_why)] {
            for (k, v) in o {
                *m.entry(k).or_default() += v;
            }
        }
    }

    fn add(&mut self, name: &str, code: &[u8], wave64: bool, keep: usize, mode: recomp::Mode) {
        // `XCL_RECOMP_ONLY=<hash>` (the hex prefix of an `XCL_DUMP_HLSL` file): that program alone
        if let Ok(only) = std::env::var("XCL_RECOMP_ONLY") {
            if format!("{:016x}", name_hash(name)) != only {
                return;
            }
        }
        // `XCL_RECOMP_STAGES=CS,MS`: only programs of these stages
        if let Ok(st) = std::env::var("XCL_RECOMP_STAGES") {
            let l = match &mode {
                recomp::Mode::Vertex(_) => "VS",
                recomp::Mode::Pixel(_) => "PS",
                recomp::Mode::Kernel => "kernel",
                recomp::Mode::Mesh(_) => "MS",
                recomp::Mode::Geometry(_) => "GS",
                recomp::Mode::Compute(_) => "CS",
            };
            if !st.split(',').any(|x| x == l) {
                return;
            }
        }
        self.programs += 1;
        self.wave32 += !wave64 as usize;
        let (label, profile) = match &mode {
            recomp::Mode::Vertex(_) => ("VS", "vs_6_6"),
            recomp::Mode::Pixel(_) => ("PS", "ps_6_6"),
            recomp::Mode::Kernel => ("kernel", "cs_6_6"),
            recomp::Mode::Mesh(_) => ("MS", "cs_6_6"),
            recomp::Mode::Geometry(_) => ("GS", "cs_6_6"),
            recomp::Mode::Compute(_) => ("CS", "cs_6_6"),
        };
        if matches!(label, "VS" | "PS") {
            if wave64 {
                self.graphics_wave64 += 1;
                let (hz, resolved) = recomp::wave_lane_reads(code);
                for k in &hz {
                    *self.wave_hazards.entry(k.to_string()).or_default() += 1;
                }
                for k in &resolved {
                    *self.wave_resolved.entry(k.to_string()).or_default() += 1;
                }
                if !resolved.is_empty() {
                    *self.wave_resolved.entry("(programs)".into()).or_default() += 1;
                }
                if !hz.is_empty() {
                    *self.wave_hazards.entry("(programs at risk)".into()).or_default() += 1;
                }
            } else {
                self.graphics_wave32 += 1;
            }
        }
        if let recomp::Mode::Compute(cs) = &mode {
            let lane = recomp::lower_dxil::compute_code(code, cs, wave64);
            let o = crate::gpu::dxil::kernel::KernelOptions { wave64, split: false, mem_windows: 1, sampler_base: 0 };
            let inst = crate::gpu::dxil::kernel::translate_compute(code, o, cs).map(|k| k.dxil);
            self.cs_forms[0] += 1;
            self.cs_forms[1] += lane.is_ok() as usize;
            self.cs_forms[2] += inst.is_ok() as usize;
            if let (Ok(a), Ok(b)) = (&lane, &inst) {
                self.cs_forms[3] += 1;
                self.cs_forms[4] += a.len();
                self.cs_forms[5] += b.len();
            }
            if let Err(e) = &lane {
                let key: String = e.split([':', '(']).next().unwrap_or("").split_whitespace().take(5).collect::<Vec<_>>().join(" ");
                *self.cs_lane_why.entry(key).or_default() += 1;
            }
        }
        let survey_cs = match &mode {
            recomp::Mode::Compute(cs) => Some(("CS", cs.clone())),
            // the mesh pass is a compute pass of the NGG program: its threads in a 1-D group
            recomp::Mode::Mesh(ms) => Some(("MS", recomp::CsInterface { threads: [ms.threads, 1, 1], tid_components: 1, ..Default::default() })),
            _ => None,
        };
        if let Some((st, cs)) = survey_cs {
            let lane = recomp::lower_dxil::survey_compute_code(code, &cs, wave64);
            let o = crate::gpu::dxil::kernel::KernelOptions { wave64, split: false, mem_windows: 1, sampler_base: 0 };
            let inst = match &mode {
                recomp::Mode::Mesh(ms) => crate::gpu::dxil::kernel::translate_mesh(code, o, ms).map(|_| ()),
                _ => crate::gpu::dxil::kernel::translate_compute(code, o, &cs).map(|_| ()),
            };
            let mut keys: Vec<(String, bool)> = lane.iter().map(|r| (survey_key(r), survey_structural(r))).collect();
            keys.sort();
            keys.dedup();
            let structural: Vec<&String> = keys.iter().filter(|k| k.1).map(|k| &k.0).collect();
            let lm = lane.len() == 1 && ["lane conditions", "indirect jump", "branch to the entry", "irreducible", "no fixpoint", "cannot decode"].iter().any(|w| lane[0].starts_with(w));
            let f = self.forms.entry(st.to_string()).or_default();
            let bridge = !lane.is_empty() && structural.is_empty();
            for (i, c) in [true, wave64, lane.is_empty(), inst.is_ok(), lane.is_empty() && inst.is_ok(), bridge, bridge && inst.is_ok(), lm].into_iter().enumerate() {
                f[i] += c as usize;
            }
            for (k, _) in &keys {
                *self.lane_every.entry(format!("{st} | {k}")).or_default() += 1;
            }
            if let [k] = structural[..] {
                *self.lane_sole.entry(format!("{st} | {k}")).or_default() += 1;
            }
            if let Err(e) = &inst {
                *self.inst_why.entry(format!("{st} | {}", survey_key(e.lines().next().unwrap_or("")))).or_default() += 1;
            }
        }
        match recomp::lane_masks::analyze_code(code, wave64) {
            Ok(s) => {
                let counts = [1, s.guarded.is_empty() as usize, s.vector, s.guarded.len(), s.lane_branches, s.uniform_branches];
                for (a, b) in self.lanes.iter_mut().zip(counts) {
                    *a += b;
                }
            }
            Err(why) => *self.lanes_failed.entry(why.to_string()).or_default() += 1,
        }
        // `XCL_RECOMP_NO_HLSL=1`: the direct DXIL survey alone
        if std::env::var("XCL_RECOMP_NO_HLSL").is_ok() {
            return;
        }
        // `XCL_RECOMP_WINDOWS=4`: as a title with unified memory translates (`pso::mem_windows`)
        let mem_windows = std::env::var("XCL_RECOMP_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        match recomp::translate(code, &recomp::Options { mode, wave64, mem_windows, ..Default::default() }) {
            Ok(t) => {
                let e = self.by_mode.entry(label.to_string()).or_default();
                e.0 += 1;
                e.1 += t.complete() as usize;
                self.hlsl_bytes += t.hlsl.len();
                for (m, n) in &t.unsupported {
                    let e = self.missing.entry(m.clone()).or_default();
                    e.0 += n;
                    e.1 += 1;
                }
                for k in t.unsupported_why.keys() {
                    *self.why.entry(k.clone()).or_default() += 1;
                }
                if t.unsupported.len() == 1 {
                    *self.sole.entry(t.unsupported.keys().next().unwrap().clone()).or_default() += 1;
                }
                if let (Some(want), None) = (EXAMPLE_FOR.get(), &self.example) {
                    // `--example MNEMONIC` (first program missing it) or `--example name=SUBSTR`
                    let hit = match want.strip_prefix("name=") {
                        Some(sub) => name.contains(sub),
                        None => t.unsupported.contains_key(want),
                    };
                    if hit {
                        let why: Vec<&String> = t.unsupported_why.keys().collect();
                        let mut txt = format!("{name} (mode {label}, wave{}): {why:?}
", if wave64 { 64 } else { 32 });
                        for l in crate::gpu::rdna2::disassemble(code, 0, false) {
                            txt.push_str(&format!("  {:04x}: {}
", l.pc, l.text));
                        }
                        self.example = Some((txt, t.hlsl.clone(), profile));
                    }
                }
                if t.complete() {
                    dump_hlsl(name, label, profile, &t.hlsl);
                    self.complete += 1;
                    if self.samples.len() < keep {
                        self.samples.push((name.to_string(), t.hlsl, profile));
                    }
                }
            }
            Err(e) => {
                self.errors += 1;
                if self.error_examples.len() < 10 {
                    self.error_examples.push(format!("{name}: {e}"));
                }
            }
        }
    }
}

#[derive(Default)]
struct Agg {
    /// translate programs (`--recomp`), keep this many complete HLSL samples
    recomp_on: bool,
    keep: usize,
    rc: RecompAgg,
    iface_on: bool,
    iface: IfaceAgg,
    total: usize,
    by_kind: BTreeMap<String, usize>,
    by_stage: BTreeMap<String, usize>,
    by_hw: BTreeMap<String, usize>,
    fssm: usize,
    programs: usize,
    data_blocks: usize,
    no_code: usize,
    failed: Vec<String>,
    failed_count: usize,
    /// examples: FSSM with several programs, programs with undecodable dwords
    odd: Vec<String>,
    odd_count: usize,
    inflate_errors: usize,
    insts: xbox::InstStats,
    /// in how many shaders a mnemonic appears
    shader_count: HashMap<String, usize>,
    /// distinct mnemonics of every console shader (for coverage)
    sets: Vec<Vec<String>>,
}

impl Agg {
    fn merge(&mut self, o: Agg) {
        self.total += o.total;
        for (k, v) in o.by_kind {
            *self.by_kind.entry(k).or_default() += v;
        }
        for (k, v) in o.by_stage {
            *self.by_stage.entry(k).or_default() += v;
        }
        for (k, v) in o.by_hw {
            *self.by_hw.entry(k).or_default() += v;
        }
        self.fssm += o.fssm;
        self.programs += o.programs;
        self.data_blocks += o.data_blocks;
        self.no_code += o.no_code;
        self.failed_count += o.failed_count;
        self.failed.extend(o.failed);
        self.failed.truncate(20);
        self.odd_count += o.odd_count;
        self.odd.extend(o.odd);
        self.odd.truncate(20);
        self.inflate_errors += o.inflate_errors;
        self.insts.merge(&o.insts);
        for (k, v) in o.shader_count {
            *self.shader_count.entry(k).or_default() += v;
        }
        self.sets.extend(o.sets);
        let keep = self.keep;
        self.rc.merge(o.rc, keep);
        self.iface.merge(o.iface);
    }

    fn add(&mut self, name: &str, blob: &[u8]) {
        self.total += 1;
        let (kind, _) = shader::classify(blob);
        *self.by_kind.entry(kind.label().to_string()).or_default() += 1;
        if kind != shader::Kind::XboxNative {
            return;
        }
        let Some(sh) = xbox::parse(blob) else { return };
        *self.by_stage.entry(sh.stage.clone()).or_default() += 1;
        let mut failed = !sh.notes.is_empty() || sh.fssm.is_empty();
        let mut st = xbox::InstStats::default();
        for f in &sh.fssm {
            self.fssm += 1;
            *self.by_hw.entry(f.stage_label()).or_default() += 1;
            self.programs += f.programs.len();
            self.data_blocks += f.data_blocks.len();
            if f.programs.is_empty() {
                self.no_code += 1;
            }
            failed |= f.error.is_some() || f.programs.is_empty() || !f.code_from_header;
        }
        st.add_shader(&sh);
        let multi = sh.fssm.iter().filter(|f| f.programs.len() > 1).count();
        if multi > 0 || st.undecoded > 0 {
            self.odd_count += 1;
            if self.odd.len() < 20 {
                self.odd.push(format!("{name}: FSSMs with several programs: {multi}, undecoded dwords: {}", st.undecoded));
            }
        }
        if failed {
            self.failed_count += 1;
            if self.failed.len() < 20 {
                let why: Vec<String> =
                    sh.notes.iter().cloned().chain(sh.fssm.iter().filter_map(|f| f.error.clone())).collect();
                self.failed.push(format!("{name}: {}", if why.is_empty() { "no code / code found by scanning".into() } else { why.join("; ") }));
            }
        }
        if self.iface_on {
            self.iface.add(name, blob, &sh);
        }
        // `--example name=SUBSTR` only needs the matching programs
        let wanted = match EXAMPLE_FOR.get().and_then(|w| w.strip_prefix("name=")) {
            Some(sub) => name.contains(sub),
            None => true,
        };
        if self.recomp_on && wanted {
            for (fi, f) in sh.fssm.iter().enumerate() {
                let mode = recomp::interface(blob, &sh, f).unwrap_or(recomp::Mode::Kernel);
                for p in &f.programs {
                    let keep = self.keep;
                    let code = &f.data[p.offset..p.offset + p.len];
                    self.rc.add(&format!("{name} FSSM#{fi}"), code, recomp::fssm_wave64(f, code), keep, mode.clone());
                    // `--example name=…`: FSSM header (up to the code) as dwords, once
                    if EXAMPLE_FOR.get().is_some_and(|w| w.starts_with("name=")) && !HEADER_SHOWN.swap(true, Ordering::Relaxed) {
                        let end = f.code.map(|c| c.0).unwrap_or(0x400).min(f.data.len()).min(0x800);
                        let mut txt = format!("FSSM header {name} (up to the code, {end:#x} B):
");
                        for (i, c) in f.data[..end].chunks(4).enumerate() {
                            if i % 8 == 0 {
                                txt.push_str(&format!("  {:04x}:", i * 4));
                            }
                            let v = u32::from_le_bytes([c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0), *c.get(3).unwrap_or(&0)]);
                            txt.push_str(&format!(" {v:08x}"));
                            if i % 8 == 7 {
                                txt.push(10 as char);
                            }
                        }
                        println!("{txt}");
                    }
                }
            }
        }
        let set: Vec<String> = st.by_mnemonic.keys().cloned().collect();
        for m in &set {
            *self.shader_count.entry(m.clone()).or_default() += 1;
        }
        self.sets.push(set);
        self.insts.merge(&st);
    }
}

enum Input {
    Pack { data: Vec<u8>, records: Vec<pack::Record> },
    Files(Vec<PathBuf>),
}

impl Input {
    fn len(&self) -> usize {
        match self {
            Input::Pack { records, .. } => records.len(),
            Input::Files(f) => f.len(),
        }
    }

    /// (name, blob); blob is None when a record does not inflate
    fn get(&self, i: usize) -> (String, Option<Vec<u8>>) {
        match self {
            Input::Pack { data, records } => (format!("{} @{:#x}", records[i].name, records[i].offset), pack::inflate(data, &records[i])),
            Input::Files(f) => (f[i].display().to_string(), std::fs::read(&f[i]).ok()),
        }
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect_files(&p, out);
            continue;
        }
        let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        // our own side files in <data>\shaders\ (FSSM dumps, listings)
        if name.ends_with(".txt") || name.contains(".fssm") {
            continue;
        }
        out.push(p);
    }
}

fn analyze(input: &Input, recomp_on: bool, keep: usize, iface_on: bool) -> Agg {
    let n = input.len();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(32);
    let parts: Vec<Agg> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut a = Agg { recomp_on, keep, iface_on, ..Default::default() };
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        match input.get(i) {
                            (name, Some(blob)) => a.add(&name, &blob),
                            (_, None) => a.inflate_errors += 1,
                        }
                    }
                    a
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut agg = Agg { recomp_on, keep, iface_on, ..Default::default() };
    for p in parts {
        agg.merge(p);
    }
    agg
}

fn fmt_map(m: &BTreeMap<String, usize>) -> String {
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    v.iter().map(|(k, c)| format!("{k} {c}")).collect::<Vec<_>>().join(", ")
}

/// Mnemonics in implementation order: used by the most shaders first (ties: more
/// frequent first) – the order that unlocks whole shaders soonest.
fn by_shader_count(agg: &Agg, ranked: &[(String, u64)]) -> Vec<String> {
    let mut v: Vec<(&String, usize, u64)> = ranked.iter().map(|(m, c)| (m, agg.shader_count.get(m).copied().unwrap_or(0), *c)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then_with(|| a.0.cmp(b.0)));
    v.into_iter().map(|x| x.0.clone()).collect()
}

/// Shaders whose every mnemonic is among the first `k` of `order`.
fn coverage(agg: &Agg, order: &[String], ks: &[usize]) -> Vec<(usize, usize)> {
    let rank: HashMap<&str, usize> = order.iter().enumerate().map(|(i, m)| (m.as_str(), i)).collect();
    let max_rank: Vec<usize> = agg.sets.iter().map(|s| s.iter().map(|m| rank[m.as_str()] + 1).max().unwrap_or(0)).collect();
    ks.iter().map(|&k| (k, max_rank.iter().filter(|&&r| r <= k).count())).collect()
}

struct ShadersOptions {
    top: usize,
    json: bool,
    compile: usize,
    recomp: bool,
    iface: bool,
    example: Option<String>,
    /// `--windows`; None: the runtime's count for `--precompile` / `--export-hlsl` /
    /// `--measure-pso` (`configured_runtime_windows`), 4 for `--direct`
    windows: Option<u32>,
    split: Option<bool>,
    export_hlsl: Option<PathBuf>,
    min_kb: usize,
    count: usize,
    record_hash: Option<u64>,
    measure_pso: Option<u64>,
    repeat: usize,
    precompile: bool,
    prepare_corpus: bool,
    limit: usize,
    only: Option<String>,
    pipelines: bool,
    used_only: bool,
    records: Option<PathBuf>,
    pso_manifest: Option<PathBuf>,
    through_present: Option<u64>,
    /// `--graphics-records`: the path is a job file of `graphics_in_workers` (`graphics_worker`)
    graphics_records: bool,
    direct: bool,
    sample: usize,
    seed: u64,
    validate: bool,
    compare: bool,
    threads: Option<usize>,
    trace: bool,
    max_bytes: usize,
    timeout: u64,
    lanes: bool,
    hash: Option<u64>,
    dump: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<ShadersOptions, String> {
    let mut opts = ShadersOptions {
        top: 60,
        json: false,
        compile: 0,
        recomp: false,
        iface: false,
        example: None,
        windows: None,
        split: None,
        export_hlsl: None,
        min_kb: 1024,
        count: 2,
        record_hash: None,
        measure_pso: None,
        repeat: 1,
        precompile: false,
        prepare_corpus: false,
        limit: usize::MAX,
        only: None,
        pipelines: false,
        used_only: false,
        records: None,
        pso_manifest: None,
        through_present: None,
        graphics_records: false,
        direct: false,
        sample: 200,
        seed: 1,
        validate: false,
        compare: false,
        threads: None,
        trace: false,
        max_bytes: 16384,
        timeout: 10,
        lanes: false,
        hash: None,
        dump: None,
    };
    let mut p = crate::cli::Parser::new(args);
    while let Some(arg) = p.next_arg() {
        match arg {
            "--top" => opts.top = p.parse_usize("--top")?,
            "--json" => opts.json = true,
            "--compile" => opts.compile = p.parse_usize("--compile")?,
            "--recomp" => opts.recomp = true,
            "--iface" => opts.iface = true,
            "--example" => opts.example = Some(p.value("--example")?.to_string()),
            "--windows" => {
                let w = p.parse_usize("--windows")?;
                opts.windows = Some(u32::try_from(w).map_err(|_| format!("invalid value for --windows: {w}"))?);
            }
            "--split" => opts.split = Some(p.parse_bool_01("--split")?),
            "--export-hlsl" => opts.export_hlsl = Some(PathBuf::from(p.value("--export-hlsl")?)),
            "--min-kb" => opts.min_kb = p.parse_usize("--min-kb")?,
            "--count" => opts.count = p.parse_usize("--count")?,
            "--record-hash" => opts.record_hash = Some(p.parse_hex_u64("--record-hash")?),
            "--measure-pso" => opts.measure_pso = Some(p.parse_hex_u64("--measure-pso")?),
            "--repeat" => opts.repeat = p.parse_usize("--repeat")?,
            "--precompile" => opts.precompile = true,
            "--prepare-corpus" => opts.prepare_corpus = true,
            "--limit" => opts.limit = p.parse_usize("--limit")?,
            "--only" => opts.only = Some(p.value("--only")?.to_ascii_lowercase()),
            "--pipelines" => opts.pipelines = true,
            "--used-only" | "--used" => opts.used_only = true,
            "--records" => opts.records = Some(PathBuf::from(p.value("--records")?)),
            "--pso-manifest" => opts.pso_manifest = Some(PathBuf::from(p.value("--pso-manifest")?)),
            "--through-present" => {
                let n = p.parse_u64("--through-present")?;
                if n == 0 {
                    return Err("--through-present needs a positive present count".into());
                }
                opts.through_present = Some(n);
            }
            "--graphics-records" => opts.graphics_records = true,
            "--direct" => opts.direct = true,
            "--sample" => opts.sample = p.parse_usize("--sample")?,
            "--seed" => opts.seed = p.parse_u64("--seed")?,
            "--validate" => opts.validate = true,
            "--compare" => opts.compare = true,
            "--threads" => opts.threads = Some(p.parse_usize("--threads")?),
            "--trace" => opts.trace = true,
            "--max-bytes" => opts.max_bytes = p.parse_usize("--max-bytes")?,
            "--timeout" => opts.timeout = p.parse_u64("--timeout")?,
            "--lanes" => opts.lanes = true,
            "--hash" => opts.hash = Some(p.parse_hex_u64("--hash")?),
            "--dump" => opts.dump = Some(PathBuf::from(p.value("--dump")?)),
            other => return Err(crate::cli::Parser::unexpected(other)),
        }
    }
    Ok(opts)
}

const PSO_PRECOMPILE_HELP: &str = "\
Usage: xcl pso-precompile <game folder | shaders.rpf> [--threads N] [--records FILE] [--used]\n\
Translates compute programs and prepares their driver pipeline cache before a run.\n\
  --threads N     maximum driver compiles at once\n\
  --records FILE  restrict preparation to hexadecimal record hashes or pipeline keys\n\
  --used          restrict preparation to keys from used-pipelines.txt\n";

/// Prepare the compute pipelines in a title's shader package before its run.
pub fn pso_precompile(game: &Path, args: &[String]) -> i32 {
    if matches!(game.to_str(), Some("--help" | "-h")) || args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{PSO_PRECOMPILE_HELP}");
        return 0;
    }
    let Some(pack) = shader_package(game) else {
        eprintln!("xcl pso-precompile: no shaders.rpf in {}", game.display());
        return 2;
    };
    let mut all = vec!["--precompile".to_string(), "--pipelines".to_string()];
    all.extend(args.iter().cloned());
    run(&pack, &all)
}

/// A title's shader package: the path itself when it is a file, else `shaders.rpf` in its folder.
fn shader_package(game: &Path) -> Option<PathBuf> {
    if game.is_file() {
        return Some(game.to_path_buf());
    }
    let pack = game.join("shaders.rpf");
    pack.is_file().then_some(pack)
}

pub fn run(path: &Path, args: &[String]) -> i32 {
    let opts = match parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("xcl shaders: {e}");
            return 2;
        }
    };
    let milestone = match (&opts.pso_manifest, opts.through_present) {
        (Some(manifest), Some(present)) => {
            let compatible = opts.precompile
                && opts.pipelines
                && opts.only.as_deref().is_none_or(|stage| stage.eq_ignore_ascii_case("cs"))
                && !opts.used_only
                && opts.records.is_none()
                && opts.limit == usize::MAX
                && opts.export_hlsl.is_none()
                && opts.measure_pso.is_none()
                && !opts.direct
                && !opts.prepare_corpus
                && !opts.graphics_records;
            if !compatible {
                eprintln!("use --precompile --pipelines --pso-manifest FILE --through-present N for a compute milestone");
                return 2;
            }
            Some((manifest.as_path(), present))
        }
        (None, None) => None,
        _ => {
            eprintln!("--pso-manifest and --through-present must be used together");
            return 2;
        }
    };
    if opts.prepare_corpus {
        if opts.split.is_some() || opts.limit != usize::MAX || opts.only.is_some() || opts.used_only {
            eprintln!("xcl shaders: --prepare-corpus requires all staged recipes and their recorded memory configuration");
            return 2;
        }
        let progress = LaunchPreparation { adapter: "auto".into(), ..Default::default() };
        let result = pso_corpus::discover(path)
            .and_then(|c| c.ok_or_else(|| "No pipeline corpus found in this title folder".to_string()))
            .and_then(|c| c.prepare_with_threads(&progress, opts.threads))
            .and_then(|_| progress.check_corpus_complete());
        return match result {
            Ok(()) => { println!("Prepared all {} discovered pipeline recipes", progress.done.load(Ordering::Acquire)); 0 }
            Err(e) => { eprintln!("xcl shaders: {e}"); 2 }
        };
    }
    // the runtime's memory windows: as many as a run has (`XCL_UMA_WINDOWS`) unless `--windows`
    // says, as its shaders take them (`runtime_windows`), as `--precompile`
    let runtime_windows = {
        let windows = opts.windows.map_or_else(crate::gpu::uma::configured_runtime_windows, |w| w as usize);
        let heap = windows > 1 && recomp::pso::heap_windows();
        if !recomp::valid_window_count(windows, heap) {
            eprintln!("xcl shaders: --windows {windows}: 1 to 4, or 8 or 16 with XCL_HEAP_WINDOWS=1");
            return 2;
        }
        let windows = windows as u32;
        move || runtime_windows(windows)
    };
    if opts.graphics_records {
        return graphics_worker(path, runtime_windows(), opts.threads);
    }
    let top = opts.top;
    let json = opts.json;
    let compile = opts.compile;
    let recomp_on = compile > 0 || opts.recomp;
    let iface_on = opts.iface;
    if let Some(m) = &opts.example {
        let _ = EXAMPLE_FOR.set(m.clone());
    }
    let t0 = std::time::Instant::now();

    let (input, what) = if path.is_dir() {
        let mut files = Vec::new();
        collect_files(path, &mut files);
        let n = files.len();
        (Input::Files(files), format!("folder, {n} files"))
    } else {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("xcl shaders: {}: {e}", path.display());
                return 2;
            }
        };
        if data.starts_with(pack::RPF9_MAGIC) {
            let records = pack::scan(&data, 0x40);
            let what = format!("RPF9 package {:.1} MB, {} shader records", data.len() as f64 / 1048576.0, records.len());
            (Input::Pack { data, records }, what)
        } else if opts.export_hlsl.is_some() || opts.measure_pso.is_some() || milestone.is_some() {
            // one shader file: the record to export or measure
            (Input::Files(vec![path.to_path_buf()]), "single file".to_string())
        } else {
            let mut a = Agg { recomp_on, keep: compile, iface_on, ..Default::default() };
            a.add(&path.display().to_string(), &data);
            return print(path, "single file", a, top, json, t0, compile);
        }
    };
    if let Some((manifest, cutoff)) = milestone {
        return match milestone::run(&input, manifest, cutoff, runtime_windows(), opts.split) {
            Ok(()) => 0,
            Err(e) => { eprintln!("xcl shaders: {e}"); 2 }
        };
    }
    if let Some(dir) = &opts.export_hlsl {
        return export_hlsl(
            &input,
            dir,
            opts.min_kb,
            opts.count,
            runtime_windows(),
            opts.split,
            opts.record_hash,
        );
    }
    if let Some(hash) = opts.measure_pso {
        return measure_pso(
            &input,
            hash,
            opts.repeat.max(1),
            runtime_windows(),
            opts.split,
        );
    }
    if opts.precompile {
        let mut selected = match &opts.records {
            Some(path) => match std::fs::read_to_string(path) {
                Ok(text) => Some(parse_pipeline_keys(&text)),
                Err(e) => {
                    eprintln!("xcl shaders --records {}: {e}", path.display());
                    return 2;
                }
            },
            None => None,
        };
        if opts.used_only {
            selected.get_or_insert_with(Default::default).extend(crate::hle::d3d12x::pso_async::earlier_used_keys());
        }
        // the runtime's key: heap windows with `XCL_HEAP_WINDOWS=1` (`pso::mem_windows`)
        let windows = runtime_windows();
        let coverage = match ComputeCoverage::worker() {
            Ok(coverage) => coverage,
            Err(e) => { eprintln!("xcl shaders: {e}"); return 2; }
        };
        let indirect = (opts.used_only && opts.pipelines)
            .then(crate::hle::d3d12x::real::earlier_indirect_split_records);
        return match precompile(&input, opts.limit, windows, opts.split, opts.only.as_deref(), opts.pipelines,
            selected.as_ref(), indirect.as_ref(), opts.threads, t0, None, coverage.as_ref()) {
            Ok(()) => 0,
            Err(e) => { eprintln!("xcl shaders: {e}"); 2 }
        };
    }
    if opts.direct {
        return direct_sweep(&input, &opts, t0);
    }
    let agg = analyze(&input, recomp_on, compile, iface_on);
    drop(input);
    print(path, &what, agg, top, json, t0, compile)
}

/// The runtime's memory windows, as the offline commands key and translate programs.
fn runtime_windows(windows: u32) -> u32 {
    recomp::pso::windows_for(windows)
}

#[cfg(test)]
fn parse_record_hash(s: &str) -> Result<u64, String> {
    crate::cli::parse_hex_u64_str(s)
}

fn is_compute_container(blob: &[u8]) -> bool {
    if recomp::pso::fspx(blob).is_some() {
        return true;
    }
    xbox::parse(blob).is_some_and(|sh| sh.fssm.iter().any(|f| matches!(recomp::interface(blob, &sh, f), Some(recomp::Mode::Compute(_)))))
}

fn record_by_hash(input: &Input, target: u64) -> Result<(String, Vec<u8>), String> {
    let mut unreadable = 0usize;
    for i in 0..input.len() {
        let (name, blob) = input.get(i);
        let Some(blob) = blob else {
            unreadable += 1;
            continue;
        };
        if shader::fnv64(&blob) == target {
            return Ok((name, blob));
        }
    }
    Err(format!("record {target:016x} not found ({unreadable} record(s) could not be read or inflated)"))
}

/// Write compute HLSL in input order. `record_hash` selects exactly one console blob and
/// bypasses the size threshold, so a timing experiment can name its source reproducibly.
fn export_hlsl(input: &Input, dir: &Path, min_kb: usize, count: usize, windows: u32, split: Option<bool>, record_hash: Option<u64>) -> i32 {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("{}: {e}", dir.display());
        return 2;
    }
    let split = split.unwrap_or_else(|| recomp::pso::split_wave64_forced() || recomp::runner::wave64_supported().is_ok_and(|w| !w));
    let wanted = if record_hash.is_some() { 1 } else { count };
    let min_bytes = min_kb.saturating_mul(1024);
    let mut written = 0usize;
    let mut failures = 0usize;
    let mut matched = false;
    let mut exported = HashSet::new();
    for i in 0..input.len() {
        if written == wanted {
            break;
        }
        let (name, blob) = input.get(i);
        let Some(blob) = blob else {
            failures += 1;
            eprintln!("  {name}: could not read or inflate record");
            continue;
        };
        let hash = shader::fnv64(&blob);
        if record_hash.is_some_and(|want| want != hash) {
            continue;
        }
        if exported.contains(&hash) {
            continue;
        }
        if !is_compute_container(&blob) {
            if record_hash.is_some() {
                matched = true;
                failures += 1;
                eprintln!("  {name} ({hash:016x}): not a console compute shader");
                break;
            }
            continue;
        }
        matched = true;
        let hlsl = match recomp::pso::console_hlsl(&blob, recomp::pso::Stage::Compute, windows, split) {
            Ok((hlsl, _)) => hlsl,
            Err(e) => {
                failures += 1;
                eprintln!("  {name} ({hash:016x}): {e}");
                if record_hash.is_some() {
                    break;
                }
                continue;
            }
        };
        if record_hash.is_none() && hlsl.len() < min_bytes {
            continue;
        }
        let file = format!("{hash:016x}.hlsl");
        if let Err(e) = std::fs::write(dir.join(&file), &hlsl) {
            failures += 1;
            eprintln!("  {name} ({hash:016x}): cannot write {file}: {e}");
            if record_hash.is_some() {
                break;
            }
            continue;
        }
        println!("{:>8} KB  {file} {name}", hlsl.len() / 1024);
        exported.insert(hash);
        written += 1;
    }
    if record_hash.is_some() && !matched {
        eprintln!("  record {:016x} not found", record_hash.unwrap());
    }
    println!(
        "{written} program(s) written to {} (split wave64: {split}, {} windows{}, {failures} failure(s))",
        dir.display(),
        windows & !(recomp::HEAP_WINDOWS | recomp::HEAP_FLAT),
        if windows & recomp::HEAP_WINDOWS != 0 { " from the descriptor heap" } else { "" }
    );
    if written == wanted { 0 } else {
        eprintln!("xcl shaders --export-hlsl: requested {wanted} program(s), wrote {written}");
        2
    }
}

/// Why `measure_pso` did not measure: this machine cannot (no device, no cache control,
/// no DXC), or the record or the driver failed.
#[derive(Debug)]
enum MeasureError {
    Unavailable(String),
    Failed(String),
}

/// Compile one exact console compute shader through DXC, then time only the driver's
/// `CreateComputePipelineState`. The implicit D3D and driver caches must both be disabled;
/// refusing to do that is an error rather than a misleading warm-cache measurement.
/// This is the HLSL path the runtime uses by default (not `XCL_DIRECT_DXIL`).
fn measure_pso(input: &Input, hash: u64, repeat: usize, windows: u32, split: Option<bool>) -> i32 {
    match measure_pso_result(input, hash, repeat, windows, split) {
        Ok(()) => 0,
        Err(MeasureError::Unavailable(e) | MeasureError::Failed(e)) => {
            eprintln!("xcl shaders --measure-pso: {e}");
            2
        }
    }
}

fn measure_pso_result(input: &Input, hash: u64, repeat: usize, windows: u32, split: Option<bool>) -> Result<(), MeasureError> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::*;
    use MeasureError::{Failed, Unavailable};

    let (name, blob) = record_by_hash(input, hash).map_err(Failed)?;
    if !is_compute_container(&blob) {
        return Err(Failed(format!("{name} is not a console compute shader")));
    }
    recomp::dxc::location().map_err(|e| Unavailable(format!("DXC: {e}")))?;
    let pipeline = Pipelines::new().map_err(|e| Unavailable(format!("no GPU device: {e}")))?;
    // before translating and compiling (tens of seconds for a big program): a machine that
    // cannot turn the caches off fails at once
    let dev9 = pipeline.dev.cast::<ID3D12Device9>().map_err(|e| Unavailable(format!("ID3D12Device9 is required to disable the driver cache: {e}")))?;
    let cache_kinds = D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED | D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER;
    unsafe { dev9.ShaderCacheControl(cache_kinds, D3D12_SHADER_CACHE_CONTROL_FLAG_DISABLE) }
        .map_err(|e| Unavailable(format!("cannot disable the driver cache (Windows Developer Mode is required): {e}")))?;
    let split = split.unwrap_or_else(|| recomp::pso::split_wave64_forced() || recomp::pso::wave32_only_for_device(&pipeline.dev));
    let dxc_start = std::time::Instant::now();
    // `XCL_DIRECT_DXIL=1`: the direct translator's container instead (no HLSL fallback, so the
    // number is never silently the other path's)
    let (dxil, samplers) = if crate::gpu::dxil::kernel::enabled() {
        let c = recomp::pso::direct_compute(&blob, 0, windows, split).map_err(|e| Failed(format!("{name}: direct: {e}")))?;
        (c.dxil, c.samplers)
    } else {
        let (translation, profile) = recomp::pso::console_translation(&blob, recomp::pso::Stage::Compute, 0, windows, split, false).map_err(|e| Failed(format!("{name}: {e}")))?;
        let dxil = recomp::dxc::compile(&translation.hlsl, "main", profile, &[]).map_err(|e| Failed(format!("DXC: {e}")))?;
        (dxil, translation.static_samplers)
    };
    let dxc_ms = dxc_start.elapsed().as_millis();
    // `XCL_MEASURE_DUMP=<dir>`: the measured container as <hash>.<path>.dxil there
    if let Ok(dir) = std::env::var("XCL_MEASURE_DUMP") {
        let _ = std::fs::write(Path::new(&dir).join(format!("{hash:016x}.{}.dxil", if crate::gpu::dxil::kernel::enabled() { "direct" } else { "hlsl" })), &dxil);
    }
    let rs = recomp::pso::abi_root_signature_with(&pipeline.dev, true, &samplers).map_err(|e| Failed(format!("root signature: {e}")))?;
    let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs)),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let mut times = Vec::with_capacity(repeat);
    let mut failure = None;
    for run in 1..=repeat {
        let start = std::time::Instant::now();
        let result = unsafe { pipeline.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) };
        let elapsed = start.elapsed();
        match result {
            Ok(pso) => drop(pso),
            Err(e) => {
                failure = Some(Failed(format!("CreateComputePipelineState run {run}: {e}")));
                break;
            }
        }
        println!("PSO run {run}/{repeat}: {:.3} s", elapsed.as_secs_f64());
        times.push(elapsed.as_secs_f64());
    }
    // the root signature's reference, on every path
    drop(unsafe { std::mem::ManuallyDrop::take(&mut desc.pRootSignature) });
    if let Some(e) = failure {
        return Err(e);
    }
    times.sort_by(f64::total_cmp);
    let median = if times.len() & 1 == 0 { (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0 } else { times[times.len() / 2] };
    println!(
        "PSO {hash:016x} [{}]: {} bytes DXIL, {dxc_ms} ms translate, {repeat} cache-off run(s), median {:.3} s, min {:.3} s, max {:.3} s, {name}",
        if crate::gpu::dxil::kernel::enabled() { "direct" } else { "hlsl" },
        dxil.len(),
        median,
        times[0],
        times[times.len() - 1]
    );
    Ok(())
}

/// `--precompile`: translate every VS / PS / MS / CS program and compile it into the DXIL cache
/// (`recomp::dxc::cache_dir()`), matching the runtime path so a title does not translate or run
/// DXC on first use. CS and standalone VS / PS variants also warm the runtime's `translated/v2`
/// cache (including the direct graphics backend). Resumable: cached programs are skipped. GS
/// halves need their pipeline partner and are left to the runtime.
/// The options are the runtime's: `windows` memory windows as a title's unified memory has
/// (`pso::mem_windows`, 4; `--windows 1` for a title without it), and the split wave64 form when
/// this machine's GPU (`XCL_ADAPTER`) runs 32-lane waves. Still different at run time: a paired
/// PS follows its VS's static samplers (only sampler base 0 can be known here), and the mesh
/// pass-through is built with its pixel shader.
/// `--pipelines`: each compute program's pipeline made on the GPU as a title's would be (XCL's
/// ABI root signature with its static samplers). Save the driver's compiled blob in the same
/// cache the runtime reads at startup. Nothing runs on the GPU.
/// Programs whose DXIL is over `XCL_PIPELINE_MAX_KB` (8 MiB by default; 0 turns the cap off)
/// and that have no blob yet are counted as capped and left to the runtime: their driver
/// compile takes tens of minutes and gigabytes, and stopped the pass on the biggest programs
/// (the > 8 MB HLSL programs DXC compiles at `-O0`, #1269).
struct Pipelines {
    dev: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    signatures: Mutex<std::collections::HashMap<(PipelineRoot, Vec<u32>), windows::Win32::Graphics::Direct3D12::ID3D12RootSignature>>,
    warm_blobs: Option<warm_blobs::Verified>,
    verified: AtomicUsize,
    made: AtomicUsize,
    refused: AtomicUsize,
    reused: AtomicUsize,
    /// Successful uncached native creations (as distinct from accepted saved blobs).
    created: AtomicUsize,
    /// What durable artifact, if any, each successful uncached creation produced.
    persistence: PersistenceCounters,
    persistence_warned: AtomicBool,
    /// pipelines an OS-backed claim proves another preparation process is already building
    elsewhere: AtomicUsize,
    split_programs: AtomicUsize,
    split_pieces: AtomicUsize,
    indirect_programs: AtomicUsize,
    /// programs over `max_dxil` with no blob yet, and their DXIL bytes
    capped: AtomicUsize,
    capped_bytes: AtomicUsize,
    /// intra-process builds keyed by full pipeline cache identity, so duplicate threads wait
    /// rather than entering the driver together when the disk claim is unavailable
    inflight: (Mutex<InflightBuilds>, std::sync::Condvar),
    /// driver compiles running and the most at once (`XCL_PIPELINE_THREADS`, 4): each of the
    /// biggest takes ~1.5 GB in the driver – fifteen at once took 24 GB
    running: (Mutex<usize>, std::sync::Condvar),
    most: usize,
    /// DXIL bytes a pipeline is still handed to the driver (`XCL_PIPELINE_MAX_KB`, 8 MiB; 0: off)
    max_dxil: usize,
    #[cfg(test)]
    cache_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PersistenceOutcome {
    Saved,
    BlobUnavailable,
    Disabled,
    WriteFailed,
}

#[derive(Default)]
struct PersistenceCounters {
    saved: AtomicUsize,
    blob_unavailable: AtomicUsize,
    disabled: AtomicUsize,
    write_failed: AtomicUsize,
}

#[derive(Debug, Default, Eq, PartialEq)]
struct PersistenceSnapshot {
    saved: usize,
    blob_unavailable: usize,
    disabled: usize,
    write_failed: usize,
}

impl PersistenceCounters {
    fn note(&self, outcome: PersistenceOutcome) {
        let counter = match outcome {
            PersistenceOutcome::Saved => &self.saved,
            PersistenceOutcome::BlobUnavailable => &self.blob_unavailable,
            PersistenceOutcome::Disabled => &self.disabled,
            PersistenceOutcome::WriteFailed => &self.write_failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> PersistenceSnapshot {
        PersistenceSnapshot {
            saved: self.saved.load(Ordering::Relaxed),
            blob_unavailable: self.blob_unavailable.load(Ordering::Relaxed),
            disabled: self.disabled.load(Ordering::Relaxed),
            write_failed: self.write_failed.load(Ordering::Relaxed),
        }
    }
}

impl PersistenceSnapshot {
    fn total(&self) -> usize {
        self.saved + self.blob_unavailable + self.disabled + self.write_failed
    }
}

fn record_fresh_creation(
    outcome: Option<PersistenceOutcome>, created: &AtomicUsize, persistence: &PersistenceCounters,
) {
    let Some(outcome) = outcome else { return };
    created.fetch_add(1, Ordering::Relaxed);
    persistence.note(outcome);
}

#[cfg(test)]
mod pipeline_persistence_tests {
    use super::{
        record_fresh_creation, PersistenceCounters, PersistenceOutcome, PersistenceSnapshot,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_fresh_creation_has_one_persistence_outcome() {
        let created = AtomicUsize::new(0);
        let counters = PersistenceCounters::default();
        record_fresh_creation(None, &created, &counters);
        assert_eq!(
            created.load(Ordering::Relaxed),
            0,
            "a failed native creation has no persistence outcome",
        );
        for outcome in [
            PersistenceOutcome::Saved,
            PersistenceOutcome::BlobUnavailable,
            PersistenceOutcome::Disabled,
            PersistenceOutcome::WriteFailed,
            PersistenceOutcome::WriteFailed,
        ] {
            record_fresh_creation(Some(outcome), &created, &counters);
        }
        let snapshot = counters.snapshot();
        assert_eq!(snapshot, PersistenceSnapshot {
            saved: 1,
            blob_unavailable: 1,
            disabled: 1,
            write_failed: 2,
        });
        assert_eq!(snapshot.total(), 5);
        assert_eq!(snapshot.total(), created.load(Ordering::Relaxed));
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PipelineRoot {
    Whole,
    DirectChain,
    IndirectChain,
}

fn split_piece_roots(prewarm_indirect: bool) -> &'static [PipelineRoot] {
    if prewarm_indirect {
        &[PipelineRoot::DirectChain, PipelineRoot::IndirectChain]
    } else {
        &[PipelineRoot::DirectChain]
    }
}

fn split_whole_roots(prewarm_indirect: bool) -> &'static [PipelineRoot] {
    if prewarm_indirect {
        &[PipelineRoot::DirectChain, PipelineRoot::IndirectChain]
    } else {
        &[]
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PipelineCacheIdentity {
    dxil_len: usize,
    dxil_hash_a: u64,
    dxil_hash_b: u64,
    flags: i32,
    node_mask: u32,
    root: PipelineRoot,
    sampler_words: Vec<u32>,
    cache_path: Option<PathBuf>,
}

impl PipelineCacheIdentity {
    fn new(
        dxil: &[u8],
        flags: i32,
        node_mask: u32,
        root: PipelineRoot,
        sampler_words: Vec<u32>,
        cache_path: Option<PathBuf>,
    ) -> Self {
        let mut alt = 0x9e37_79b9_7f4a_7c15u64;
        for &byte in dxil {
            alt ^= u64::from(byte);
            alt = alt.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        }
        Self {
            dxil_len: dxil.len(),
            dxil_hash_a: shader::fnv64(dxil),
            dxil_hash_b: alt,
            flags,
            node_mask,
            root,
            sampler_words,
            cache_path,
        }
    }
}

#[derive(Default)]
struct InflightBuilds {
    active: HashSet<PipelineCacheIdentity>,
}

struct InflightGuard<'a> {
    state: &'a (Mutex<InflightBuilds>, std::sync::Condvar),
    id: PipelineCacheIdentity,
}

impl<'a> InflightGuard<'a> {
    fn enter(state: &'a (Mutex<InflightBuilds>, std::sync::Condvar), id: PipelineCacheIdentity) -> Self {
        let mut builds = state.0.lock().unwrap_or_else(|e| e.into_inner());
        while builds.active.contains(&id) {
            builds = state.1.wait(builds).unwrap_or_else(|e| e.into_inner());
        }
        builds.active.insert(id.clone());
        Self { state, id }
    }
}

impl Drop for InflightGuard<'_> {
    fn drop(&mut self) {
        let mut builds = self.state.0.lock().unwrap_or_else(|e| e.into_inner());
        builds.active.remove(&self.id);
        self.state.1.notify_all();
    }
}

#[cfg(test)]
mod inflight_build_tests {
    use super::{InflightBuilds, InflightGuard, PipelineCacheIdentity, PipelineRoot};
    use std::path::PathBuf;
    use std::sync::{mpsc, Condvar, Mutex};
    use std::time::Duration;

    fn identity(dxil: &[u8], flags: i32, node: u32, root: PipelineRoot, samplers: &[u32], path: &str) -> PipelineCacheIdentity {
        PipelineCacheIdentity::new(dxil, flags, node, root, samplers.to_vec(), Some(PathBuf::from(path)))
    }

    #[test]
    fn identity_covers_every_pipeline_input() {
        let base = identity(b"abc", 0, 0, PipelineRoot::Whole, &[1, 2], "a.pso");
        assert_eq!(base, identity(b"abc", 0, 0, PipelineRoot::Whole, &[1, 2], "a.pso"));
        assert_ne!(base, identity(b"abd", 0, 0, PipelineRoot::Whole, &[1, 2], "a.pso"));
        assert_ne!(base, identity(b"abc", 1, 0, PipelineRoot::Whole, &[1, 2], "a.pso"));
        assert_ne!(base, identity(b"abc", 0, 2, PipelineRoot::Whole, &[1, 2], "a.pso"));
        assert_ne!(base, identity(b"abc", 0, 0, PipelineRoot::DirectChain, &[1, 2], "a.pso"));
        assert_ne!(base, identity(b"abc", 0, 0, PipelineRoot::IndirectChain, &[1, 2], "a.pso"));
        assert_ne!(base, identity(b"abc", 0, 0, PipelineRoot::Whole, &[1, 3], "a.pso"));
        assert_ne!(base, identity(b"abc", 0, 0, PipelineRoot::Whole, &[1, 2], "b.pso"));
        assert_ne!(base, PipelineCacheIdentity::new(b"abc", 0, 0, PipelineRoot::Whole, vec![1, 2], None));
    }

    #[test]
    fn indirect_selection_adds_indirect_pieces_without_changing_direct_pieces() {
        assert_eq!(super::split_piece_roots(false), &[PipelineRoot::DirectChain]);
        assert_eq!(super::split_piece_roots(true), &[PipelineRoot::DirectChain, PipelineRoot::IndirectChain]);
        assert!(super::split_whole_roots(false).is_empty());
        assert_eq!(
            super::split_whole_roots(true),
            &[PipelineRoot::DirectChain, PipelineRoot::IndirectChain],
        );
    }

    #[test]
    fn same_build_waits_while_distinct_builds_and_failures_can_proceed() {
        let state = (Mutex::new(InflightBuilds::default()), Condvar::new());
        let base = identity(b"program", 0, 0, PipelineRoot::Whole, &[], "same.pso");
        let other = identity(b"other", 0, 0, PipelineRoot::Whole, &[], "other.pso");
        let first = InflightGuard::enter(&state, base.clone());
        drop(InflightGuard::enter(&state, other));

        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                started_tx.send(()).unwrap();
                let _same = InflightGuard::enter(&state, base.clone());
                done_tx.send(()).unwrap();
            });
            started_rx.recv().unwrap();
            assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err(), "same pipeline did not wait");
            drop(first);
            done_rx.recv_timeout(Duration::from_secs(1)).expect("waiter stayed blocked");
        });

        let retry = identity(b"retry", 0, 0, PipelineRoot::Whole, &[], "retry.pso");
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _failed = InflightGuard::enter(&state, retry.clone());
            panic!("simulated build failure");
        }));
        assert!(panicked.is_err());
        drop(InflightGuard::enter(&state, retry));
        assert!(state.0.lock().unwrap_or_else(|e| e.into_inner()).active.is_empty());
    }
}

enum ClaimAcquire {
    Acquired(Claim),
    HeldElsewhere,
    Unavailable(std::io::Error),
}

/// Exclusive cross-process ownership of one driver-cache build (`<blob>.claim`). The kernel
/// deletes the file when the handle closes, including after forced process termination.
struct Claim {
    _file: std::fs::File,
}

fn claim_path_for(blob: &Path) -> PathBuf {
    let mut name = blob.as_os_str().to_owned();
    name.push(".claim");
    PathBuf::from(name)
}

impl Claim {
    fn acquire(blob: &Path) -> ClaimAcquire {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Foundation::{ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, GENERIC_READ, GENERIC_WRITE};
        use windows_sys::Win32::Storage::FileSystem::{DELETE, FILE_ATTRIBUTE_TEMPORARY, FILE_FLAG_DELETE_ON_CLOSE};

        let path = claim_path_for(blob);
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            if let Err(e) = std::fs::create_dir_all(dir) {
                return ClaimAcquire::Unavailable(e);
            }
        }
        // OPEN_ALWAYS adopts an unlocked orphan from the old create/delete protocol. DELETE is
        // required to apply FILE_FLAG_DELETE_ON_CLOSE to that existing file.
        let open = || {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .access_mode(GENERIC_READ | GENERIC_WRITE | DELETE)
                .share_mode(0)
                .attributes(FILE_ATTRIBUTE_TEMPORARY)
                .custom_flags(FILE_FLAG_DELETE_ON_CLOSE)
                .open(&path)
        };
        for attempt in 0..4 {
            match open() {
                Ok(file) => return ClaimAcquire::Acquired(Claim { _file: file }),
                Err(e) if e.raw_os_error() == Some(ERROR_SHARING_VIOLATION as i32) => return ClaimAcquire::HeldElsewhere,
                // A just-closed delete-on-close handle can leave a brief delete-pending window.
                Err(e) if e.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32) && attempt < 3 => {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                Err(e) => return ClaimAcquire::Unavailable(e),
            }
        }
        unreachable!("claim acquisition loop always returns")
    }
}

struct RunningPermit<'a> {
    state: &'a (Mutex<usize>, std::sync::Condvar),
}

impl<'a> RunningPermit<'a> {
    fn enter(state: &'a (Mutex<usize>, std::sync::Condvar), most: usize) -> Self {
        let mut running = state.0.lock().unwrap_or_else(|e| e.into_inner());
        while *running >= most {
            running = state.1.wait(running).unwrap_or_else(|e| e.into_inner());
        }
        *running += 1;
        Self { state }
    }
}

impl Drop for RunningPermit<'_> {
    fn drop(&mut self) {
        let mut running = self.state.0.lock().unwrap_or_else(|e| e.into_inner());
        *running = running.saturating_sub(1);
        self.state.1.notify_one();
    }
}

#[cfg(test)]
mod running_permit_tests {
    use super::RunningPermit;
    use std::sync::{mpsc, Condvar, Mutex};
    use std::time::Duration;

    #[test]
    fn unwind_releases_the_slot_and_wakes_a_waiter() {
        let state = (Mutex::new(0usize), Condvar::new());
        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _permit = RunningPermit::enter(&state, 1);
                scope.spawn(|| {
                    started_tx.send(()).unwrap();
                    let _waiting = RunningPermit::enter(&state, 1);
                    acquired_tx.send(()).unwrap();
                });
                started_rx.recv().unwrap();
                panic!("simulated driver panic");
            }));
            assert!(panicked.is_err());
            acquired_rx.recv_timeout(Duration::from_secs(1)).expect("waiter stayed blocked after unwind");
        });
        assert_eq!(*state.0.lock().unwrap_or_else(|e| e.into_inner()), 0);
    }
}

#[cfg(test)]
mod pipeline_claim_tests {
    use super::{claim_path_for, Claim, ClaimAcquire};
    use std::io::{BufRead, BufReader, Write};
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(prefix: &str) -> Self {
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
            let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct ChildGuard(std::process::Child);
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[test]
    fn claim_child_role_entrypoint() {
        if std::env::var("XCL_CLAIM_TEST_ROLE").as_deref() != Ok("hold") {
            return;
        }
        let blob = PathBuf::from(std::env::var("XCL_CLAIM_TEST_BLOB").expect("blob path"));
        let _claim = match Claim::acquire(&blob) {
            ClaimAcquire::Acquired(claim) => claim,
            ClaimAcquire::HeldElsewhere => panic!("unexpected contention in child"),
            ClaimAcquire::Unavailable(e) => panic!("claim unavailable in child: {e}"),
        };
        println!("LOCKED");
        std::io::stdout().flush().unwrap();
        std::thread::sleep(Duration::from_secs(60));
    }

    #[test]
    fn claims_are_exclusive_adopt_orphans_and_report_filesystem_errors() {
        let dir = TempDir::new("xcl-claim");
        let blob = dir.0.join("nested").join("one.pso");
        let first = match Claim::acquire(&blob) {
            ClaimAcquire::Acquired(claim) => claim,
            _ => panic!("first claim must succeed"),
        };
        assert!(claim_path_for(&blob).is_file());
        assert!(matches!(Claim::acquire(&blob), ClaimAcquire::HeldElsewhere));
        assert!(matches!(Claim::acquire(&dir.0.join("two.pso")), ClaimAcquire::Acquired(_)));
        drop(first);
        assert!(!claim_path_for(&blob).exists());

        std::fs::write(claim_path_for(&blob), b"orphan from old protocol").unwrap();
        let adopted = match Claim::acquire(&blob) {
            ClaimAcquire::Acquired(claim) => claim,
            _ => panic!("unlocked orphan must be adopted"),
        };
        drop(adopted);
        assert!(!claim_path_for(&blob).exists());

        let file = dir.0.join("not-a-directory");
        std::fs::write(&file, b"x").unwrap();
        assert!(matches!(Claim::acquire(&file.join("bad.pso")), ClaimAcquire::Unavailable(_)));
    }

    #[test]
    fn forced_process_termination_releases_a_claim_immediately() {
        let dir = TempDir::new("xcl-claim-kill");
        let blob = dir.0.join("driver").join("killed.pso");
        let mut child = ChildGuard(std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "shaders::pipeline_claim_tests::claim_child_role_entrypoint", "--nocapture"])
            .env("XCL_CLAIM_TEST_ROLE", "hold")
            .env("XCL_CLAIM_TEST_BLOB", &blob)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap());
        let mut lines = BufReader::new(child.0.stdout.take().unwrap()).lines();
        let ready = lines.by_ref().any(|line| line.is_ok_and(|line| line.trim() == "LOCKED"));
        assert!(ready, "child did not acquire claim");
        assert!(matches!(Claim::acquire(&blob), ClaimAcquire::HeldElsewhere));

        child.0.kill().unwrap();
        let _ = child.0.wait();
        assert!(!claim_path_for(&blob).exists());
        let recovered = match Claim::acquire(&blob) {
            ClaimAcquire::Acquired(claim) => claim,
            _ => panic!("terminated owner's claim must be free"),
        };
        drop(recovered);
        assert!(!claim_path_for(&blob).exists());
    }
}

/// `XCL_PIPELINE_MAX_KB`: the KiB of DXIL `--pipelines` still compiles (default 8 MiB; 0: no
/// cap). Above it sit the programs whose driver compile takes tens of minutes and gigabytes —
/// the > 8 MB HLSL programs DXC compiles at `-O0` (#1269) — and they hang the pass.
fn pipeline_cap_kb(set: Option<&str>) -> usize {
    set.and_then(|v| v.parse().ok()).unwrap_or(8192)
}

fn default_pipeline_threads(cores: usize, memory_gb: Option<u64>) -> usize {
    let by_memory = memory_gb.map_or(4, |gb| (gb / 2) as usize);
    cores.min(by_memory).clamp(1, 32)
}

fn pipeline_threads() -> usize {
    if let Some(threads) = std::env::var("XCL_PIPELINE_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        return threads.max(1);
    }
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut memory: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    memory.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    let memory_gb = (unsafe { GlobalMemoryStatusEx(&mut memory) } != 0).then_some(memory.ullTotalPhys >> 30);
    default_pipeline_threads(cores, memory_gb)
}

#[cfg(test)]
mod pipeline_cap_tests {
    use super::{default_pipeline_threads, pipeline_cap_kb};

    #[test]
    fn the_cap_is_8_mib_unless_set_and_off_at_zero() {
        assert_eq!(pipeline_cap_kb(None), 8192);
        assert_eq!(pipeline_cap_kb(Some("512")), 512);
        assert_eq!(pipeline_cap_kb(Some("0")), 0);
        // not a number: the default, not a crash
        assert_eq!(pipeline_cap_kb(Some("off")), 8192);
    }

    #[test]
    fn driver_threads_follow_the_core_and_memory_limits() {
        assert_eq!(default_pipeline_threads(24, Some(128)), 24);
        assert_eq!(default_pipeline_threads(24, Some(16)), 8);
        assert_eq!(default_pipeline_threads(8, None), 4);
        assert_eq!(default_pipeline_threads(64, Some(512)), 32);
        assert_eq!(default_pipeline_threads(4, Some(1)), 1);
    }

    #[test]
    fn stale_blob_cannot_bypass_the_offline_pipeline_cap() {
        use super::{Pipelines, recomp, Ordering};
        use std::time::{Instant, SystemTime, UNIX_EPOCH};

        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
        }
        if let Err(reason) = recomp::dxc::location() {
            recomp::runner::skip_test(&format!("no DXC: {reason}"));
            return;
        }
        let mut pipelines = match Pipelines::new() {
            Ok(pipelines) => pipelines,
            Err(reason) => {
                recomp::runner::skip_test(&format!("no D3D12 device: {reason}"));
                return;
            }
        };
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let scratch = Scratch(std::env::temp_dir().join(format!("xcl-stale-pso-{}-{nonce}", std::process::id())));
        std::fs::create_dir(&scratch.0).expect("scratch cache directory");
        let hlsl = "RWByteAddressBuffer mem : register(u2); [numthreads(1,1,1)] void main() { mem.Store(0, 42); }";
        let dxil = recomp::dxc::compile(hlsl, "main", "cs_6_6", &[]).expect("DXC compile");
        let path = scratch.0.join("stale.pso");
        std::fs::write(&path, b"deliberately invalid cached PSO").expect("stale blob");

        pipelines.max_dxil = dxil.len() - 1;
        let start = Instant::now();
        pipelines.make_with_cache_path(&recomp::pso::Compiled { dispatch_chain: None, dxil: dxil.clone(), samplers: vec![], dispatch_remap: None }, Some(path.clone()));
        let elapsed = start.elapsed();
        let capped = pipelines.capped.load(Ordering::Relaxed);
        let made = pipelines.made.load(Ordering::Relaxed);
        let refused = pipelines.refused.load(Ordering::Relaxed);
        let output = recomp::runner::dispatch(&dxil, [1, 1, 1], &[], &[0; 64]).expect("runtime GPU dispatch");
        println!("adapter={} dxil_bytes={} offline_ms={} capped={} made={} refused={} gpu_word0={}", recomp::runner::adapter().unwrap_or_default(), dxil.len(), elapsed.as_millis(), capped, made, refused, output[0]);
        assert_eq!(output[0], 42);
        assert_eq!((capped, made, refused), (1, 0, 0));
        assert!(!path.exists(), "rejected blob should not be retried next run");
    }

    #[test]
    fn split_compute_program_prebuilds_and_caches_chain_pieces() {
        use super::{Pipelines, recomp, Ordering};
        use crate::hle::d3d12x::real::driver_cache;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};
        use windows::Win32::Graphics::Direct3D12::*;

        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
        }
        if let Err(reason) = recomp::dxc::location() {
            recomp::runner::skip_test(&format!("no DXC: {reason}"));
            return;
        }
        let mut pipelines = match Pipelines::new() {
            Ok(pipelines) => pipelines,
            Err(reason) => {
                recomp::runner::skip_test(&format!("no D3D12 device: {reason}"));
                return;
            }
        };
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let scratch = Scratch(std::env::temp_dir().join(format!("xcl-split-pso-{}-{nonce}", std::process::id())));
        std::fs::create_dir(&scratch.0).expect("scratch cache directory");
        pipelines.cache_dir = Some(scratch.0.clone());

        // Each piece binds the dispatch-chain UAV/CBV (space2), which only `dispatch_chain_root_signature`
        // declares: building them with the regular ABI root signature would be refused by D3D12.
        let piece0 = recomp::dxc::compile(
            "RWByteAddressBuffer mem : register(u2); RWByteAddressBuffer sc : register(u0, space2); \
             cbuffer ch : register(b0, space2) { uint4 meta; }; \
             [numthreads(1,1,1)] void main() { sc.Store(0, meta.w + 1); mem.Store(0, meta.x); }",
            "main",
            "cs_6_6",
            &[],
        )
        .expect("piece0 DXC compile");
        let piece1 = recomp::dxc::compile(
            "RWByteAddressBuffer mem : register(u2); RWByteAddressBuffer sc : register(u0, space2); \
             cbuffer ch : register(b0, space2) { uint4 meta; }; \
             [numthreads(1,1,1)] void main() { uint v = sc.Load(0); mem.Store(4, v + meta.w + 2); }",
            "main",
            "cs_6_6",
            &[],
        )
        .expect("piece1 DXC compile");
        // Larger than both pieces so a cap between piece size and whole size proves each piece
        // is capped individually rather than against the unsplit whole container.
        let whole = recomp::dxc::compile(
            "RWByteAddressBuffer mem : register(u2); \
             [numthreads(1,1,1)] void main() { \
             uint x = mem.Load(0); uint y = mem.Load(4); uint z = mem.Load(8); uint w = mem.Load(12); \
             mem.Store(0, x ^ y ^ z ^ w ^ 0x12345678u); mem.Store(4, (x + y) * (z + w) + 7u); \
             mem.Store(8, (x * 3u) ^ (y * 5u) ^ (z * 7u)); mem.Store(12, x + y + z + w); }",
            "main",
            "cs_6_6",
            &[],
        )
        .expect("whole DXC compile");
        let max_piece = piece0.len().max(piece1.len());
        let compiled = recomp::pso::Compiled {
            dispatch_chain: Some(Arc::new(crate::gpu::dxil::kernel::DispatchChain {
                pieces: vec![piece0.clone(), piece1.clone()],
                dwords_per_group: 4,
            })),
            dxil: whole.clone(),
            samplers: vec![],
            dispatch_remap: None,
        };

        pipelines.max_dxil = max_piece;
        pipelines.make(&compiled);
        let p0_path = scratch.0.join(driver_cache::name_for(&piece0, 0, true));
        let p1_path = scratch.0.join(driver_cache::name_for(&piece1, 0, true));
        let whole_path = scratch.0.join(driver_cache::name_for(&whole, 0, false));
        assert_eq!(pipelines.split_programs.load(Ordering::Relaxed), 1);
        assert_eq!(pipelines.split_pieces.load(Ordering::Relaxed), 2);
        assert_eq!(pipelines.made.load(Ordering::Relaxed), 2);
        assert_eq!(pipelines.refused.load(Ordering::Relaxed), 0);
        assert_eq!(pipelines.reused.load(Ordering::Relaxed), 0);
        assert_eq!(pipelines.capped.load(Ordering::Relaxed), 0);
        assert!(p0_path.is_file() && p0_path.metadata().unwrap().len() > 0, "piece 0 .chain.pso missing");
        assert!(p1_path.is_file() && p1_path.metadata().unwrap().len() > 0, "piece 1 .chain.pso missing");
        assert!(!whole_path.exists(), "unsplit whole .pso should not be written for a split compute program");

        // Second pass reuses both cached piece blobs.
        pipelines.make(&compiled);
        assert_eq!(pipelines.reused.load(Ordering::Relaxed), 2);
        assert_eq!(pipelines.made.load(Ordering::Relaxed), 4);
        assert_eq!(pipelines.refused.load(Ordering::Relaxed), 0);

        // Runtime-side check: the dispatch-chain root signature accepts each cached .chain.pso blob.
        let chain_rs = recomp::pso::dispatch_chain_root_signature(&pipelines.dev, &[]).expect("dispatch chain RS");
        for (dxil, path) in [(&piece0, &p0_path), (&piece1, &p1_path)] {
            let blob = std::fs::read(path).expect("read cached .chain.pso");
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(chain_rs.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
                CachedPSO: D3D12_CACHED_PIPELINE_STATE { pCachedBlob: blob.as_ptr().cast(), CachedBlobSizeInBytes: blob.len() },
                ..Default::default()
            };
            let pso = unsafe { pipelines.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }
                .expect("runtime dispatch_chain RS accepts offline .chain.pso");
            drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
            let got = unsafe { pso.GetCachedBlob() }.expect("cached blob");
            assert_eq!(unsafe { got.GetBufferSize() }, blob.len());
        }

        // A corrupt .chain.pso over the per-piece cap is removed and capped without rebuilding.
        std::fs::write(&p1_path, b"corrupt chain blob").expect("corrupt piece1 blob");
        pipelines.max_dxil = piece1.len() - 1;
        let reused_before = pipelines.reused.load(Ordering::Relaxed);
        pipelines.make(&compiled);
        assert!(!p1_path.exists(), "rejected over-cap .chain.pso should be removed");
        assert_eq!(pipelines.capped.load(Ordering::Relaxed), 1);
        if piece0.len() <= pipelines.max_dxil {
            assert_eq!(pipelines.reused.load(Ordering::Relaxed), reused_before + 1);
        }
    }
}

// D3D12 devices and root signatures are free-threaded
unsafe impl Sync for Pipelines {}

impl Pipelines {
    fn new() -> Result<Pipelines, String> { Self::for_adapter("auto") }

    fn for_adapter(adapter: &str) -> Result<Pipelines, String> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| e.to_string())?;
            let sel = crate::gpu::adapter::Selector::from_env_or(adapter);
            let (dev, vendor) = if sel == crate::gpu::adapter::Selector::Warp {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().map_err(|e| e.to_string())?;
                (crate::gpu::agility::create_device(&warp, D3D_FEATURE_LEVEL_12_0).map_err(|e| e.to_string())?, 0)
            } else {
                let (adapter, desc) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next().ok_or_else(|| crate::gpu::adapter::no_match(&factory, &sel))?;
                (crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).map_err(|e| e.to_string())?, desc.VendorId)
            };
            crate::hle::d3d12x::real::driver_cache::note_vendor(vendor);
            let most = pipeline_threads();
            let max_dxil = pipeline_cap_kb(std::env::var("XCL_PIPELINE_MAX_KB").ok().as_deref()).saturating_mul(1024);
            Ok(Pipelines {
                dev,
                signatures: Mutex::new(Default::default()),
                warm_blobs: None,
                verified: AtomicUsize::new(0),
                made: AtomicUsize::new(0),
                refused: AtomicUsize::new(0),
                reused: AtomicUsize::new(0),
                created: AtomicUsize::new(0),
                persistence: PersistenceCounters::default(),
                persistence_warned: AtomicBool::new(false),
                elsewhere: AtomicUsize::new(0),
                split_programs: AtomicUsize::new(0),
                split_pieces: AtomicUsize::new(0),
                indirect_programs: AtomicUsize::new(0),
                capped: AtomicUsize::new(0),
                capped_bytes: AtomicUsize::new(0),
                inflight: (Mutex::new(InflightBuilds::default()), std::sync::Condvar::new()),
                running: (Mutex::new(0), std::sync::Condvar::new()),
                most,
                max_dxil,
                #[cfg(test)]
                cache_dir: None,
            })
        }
    }

    /// The compute pipeline of `c` (made and released). When direct translation split the program
    /// into a dispatch chain of at least two pieces, each piece is built and cached under the
    /// dispatch-chain root signature (`.chain.pso`) as the runtime does (`dispatch_chain::build`),
    /// instead of the unsplit whole container.
    fn make(&self, c: &recomp::pso::Compiled) {
        self.make_selected(c, false, None);
    }

    #[cfg(test)]
    fn make_with_cache_path(&self, c: &recomp::pso::Compiled, path_override: Option<PathBuf>) {
        self.make_selected(c, false, path_override);
    }

    fn make_selected(&self, c: &recomp::pso::Compiled, prewarm_indirect: bool, path_override: Option<PathBuf>) {
        if path_override.is_none() {
            if let Some(chain) = c.dispatch_chain.as_deref().filter(|chain| chain.pieces.len() >= 2) {
                self.split_programs.fetch_add(1, Ordering::Relaxed);
                self.split_pieces.fetch_add(chain.pieces.len(), Ordering::Relaxed);
                for &root in split_piece_roots(prewarm_indirect) {
                    self.make_chain_pieces(&chain.pieces, &c.samplers, root);
                }
                if prewarm_indirect {
                    self.indirect_programs.fetch_add(1, Ordering::Relaxed);
                    for &root in split_whole_roots(true) {
                        self.make_one(&c.dxil, &c.samplers, root, None);
                    }
                }
                return;
            }
        }
        self.make_one(&c.dxil, &c.samplers, PipelineRoot::Whole, path_override);
    }

    fn prepare_compute(&self, blob: &[u8], windows: u32) -> Result<(), String> {
        let split = recomp::pso::wave32_only_for_device(&self.dev);
        let program = recomp::pso::console_to_dxil_in(blob, recomp::pso::Stage::Compute, 0, split, windows)?;
        self.prepare_compiled_compute(&program)
    }

    fn prepare_compiled_compute(&self, program: &recomp::pso::Compiled) -> Result<(), String> {
        let make = |dxil: &[u8], root| {
            self.make_required(dxil, &program.samplers, root, None, true)
        };
        make(&program.dxil, PipelineRoot::Whole)?;
        if let Some(chain) = program.dispatch_chain.as_deref().filter(|c| c.pieces.len() >= 2) {
            for piece in &chain.pieces { make(piece, PipelineRoot::DirectChain)?; }
            make(&program.dxil, PipelineRoot::DirectChain)?;
        }
        Ok(())
    }

    fn make_chain_pieces(&self, pieces: &[Vec<u8>], samplers: &[recomp::pso::StaticSampler], root: PipelineRoot) {
        let workers = pieces.len().min(self.most);
        if workers <= 1 {
            for piece in pieces {
                self.make_one(piece, samplers, root, None);
            }
        } else {
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..workers {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(piece) = pieces.get(i) else { break };
                        self.make_one(piece, samplers, root, None);
                    });
                }
            });
        }
    }

    fn make_one(&self, dxil: &[u8], samplers: &[recomp::pso::StaticSampler], root: PipelineRoot, path_override: Option<PathBuf>) {
        let _ = self.make_required(dxil, samplers, root, path_override, false);
    }

    fn make_required(&self, dxil: &[u8], samplers: &[recomp::pso::StaticSampler], root: PipelineRoot,
                     path_override: Option<PathBuf>, required: bool) -> Result<(), String> {
        use windows::Win32::Graphics::Direct3D12::*;
        let sampler_words: Vec<u32> = samplers
            .iter()
            .flat_map(|(w, cmp)| w.iter().copied().chain([*cmp as u32]))
            .collect();
        let key = (root, sampler_words.clone());
        let layout = self.warm_blobs.as_ref().map(|_| warm_blobs::digest(&key));
        let rs = {
            let mut g = self.signatures.lock().unwrap();
            match g.get(&key) {
                Some(rs) => rs.clone(),
                None => {
                    let created = match root {
                        PipelineRoot::Whole => recomp::pso::abi_root_signature_with(&self.dev, true, samplers),
                        PipelineRoot::DirectChain => recomp::pso::dispatch_chain_root_signature(&self.dev, samplers),
                        PipelineRoot::IndirectChain => recomp::pso::dispatch_chain_indirect_root_signature(&self.dev, samplers),
                    };
                    match created {
                        Ok(rs) => {
                            g.insert(key, rs.clone());
                            rs
                        }
                        Err(e) => {
                            self.refused.fetch_add(1, Ordering::Relaxed);
                            return Err(format!("Creating compute root signature: {e}"));
                        }
                    }
                }
            }
        };
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(rs)),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const _, BytecodeLength: dxil.len() },
            ..Default::default()
        };
        use crate::hle::d3d12x::real::driver_cache;
        let library_name = match root {
            PipelineRoot::Whole => driver_cache::name_for_root(dxil, desc.Flags.0, false, samplers),
            PipelineRoot::DirectChain => driver_cache::name_for_chain_root_node(
                dxil, desc.Flags.0, driver_cache::ChainRoot::DirectConstants, samplers, desc.NodeMask,
            ),
            PipelineRoot::IndirectChain => driver_cache::name_for_chain_root_node(
                dxil, desc.Flags.0, driver_cache::ChainRoot::IndirectCbv, samplers, desc.NodeMask,
            ),
        };
        let path = path_override.or_else(|| {
            #[cfg(test)]
            if let Some(dir) = self.cache_dir.as_deref() {
                return Some(dir.join(match root {
                    PipelineRoot::Whole => driver_cache::name_for_root(dxil, desc.Flags.0, false, samplers),
                    PipelineRoot::DirectChain => driver_cache::name_for_chain_root_node(
                        dxil, desc.Flags.0, driver_cache::ChainRoot::DirectConstants, samplers, desc.NodeMask,
                    ),
                    PipelineRoot::IndirectChain => driver_cache::name_for_chain_root_node(
                        dxil, desc.Flags.0, driver_cache::ChainRoot::IndirectCbv, samplers, desc.NodeMask,
                    ),
                }));
            }
            match root {
                PipelineRoot::Whole => driver_cache::path_for_root(dxil, desc.Flags.0, false, samplers),
                PipelineRoot::DirectChain => driver_cache::path_for_chain_root_node(
                    dxil, desc.Flags.0, driver_cache::ChainRoot::DirectConstants, samplers, desc.NodeMask,
                ),
                PipelineRoot::IndirectChain => driver_cache::path_for_chain_root_node(
                    dxil, desc.Flags.0, driver_cache::ChainRoot::IndirectCbv, samplers, desc.NodeMask,
                ),
            }
        });
        let mut loaded = path.as_deref().and_then(driver_cache::load);
        let mut cached = loaded.as_ref().map(driver_cache::LoadedBlob::shared);
        let receipt = self.warm_blobs.as_ref().zip(layout).zip(path.as_deref());
        if let (Some(((book, layout), path)), Some(blob)) = (receipt, cached.as_ref()) {
            if !pipeline_library::building() && book.contains(&warm_blobs::stamp(path, layout, blob)) {
                self.verified.fetch_add(1, Ordering::Relaxed);
                drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
                return Ok(());
            }
        }
        if let Some(pso) = pipeline_library::load_compute(&library_name, &desc) {
            self.reused.fetch_add(1, Ordering::Relaxed);
            let library = pipeline_library::store(library_name, &pso);
            drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
            let done = self.made.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 25 == 0 {
                eprintln!("  pipelines: {} made, {} refused", self.made.load(Ordering::Relaxed), self.refused.load(Ordering::Relaxed));
            }
            return library;
        }
        let over_cap = self.max_dxil != 0 && dxil.len() > self.max_dxil;
        // over the cap and no blob from an earlier run: the driver's compile of this one takes
        // tens of minutes and gigabytes; the pass finishes without it, and the runtime builds it
        // when a title needs the pipeline. `XCL_PIPELINE_MAX_KB` 0 turns the cap off.
        if cached.is_none() && over_cap {
            self.capped.fetch_add(1, Ordering::Relaxed);
            self.capped_bytes.fetch_add(dxil.len(), Ordering::Relaxed);
            drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
            return Err(format!("Compute DXIL exceeds the native pipeline size limit ({} bytes)", dxil.len()));
        }
        let identity = PipelineCacheIdentity::new(dxil, desc.Flags.0, desc.NodeMask, root, sampler_words, path.clone());
        let _inflight = InflightGuard::enter(&self.inflight, identity);
        // A valid blob may be used by many processes, but validation itself can fall through to
        // a full driver compile when the blob is stale. Claim before either path so only one
        // process can validate/rebuild/remove this exact cache entry at a time.
        let _claim = match path.as_deref() {
            Some(path) => match Claim::acquire(path) {
                ClaimAcquire::Acquired(claim) => Some(claim),
                ClaimAcquire::HeldElsewhere if required => None,
                ClaimAcquire::HeldElsewhere => {
                    self.elsewhere.fetch_add(1, Ordering::Relaxed);
                    drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
                    return Err("Another process is preparing this compute pipeline".into());
                }
                ClaimAcquire::Unavailable(e) => {
                    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !WARNED.swap(true, Ordering::Relaxed) {
                        crate::xlog_warn!("Shader preparation: cannot coordinate claim for {}: {e}; building locally", path.display());
                    }
                    None
                }
            },
            None => None,
        };
        if _claim.is_some() {
            // The previous owner can publish and close between our first lookup and OPEN_ALWAYS.
            loaded = path.as_deref().and_then(driver_cache::load);
            cached = loaded.as_ref().map(driver_cache::LoadedBlob::shared);
            if cached.is_none() && over_cap {
                self.capped.fetch_add(1, Ordering::Relaxed);
                self.capped_bytes.fetch_add(dxil.len(), Ordering::Relaxed);
                drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
                return Err(format!("Compute DXIL exceeds the native pipeline size limit ({} bytes)", dxil.len()));
            }
        }
        let _permit = RunningPermit::enter(&self.running, self.most);
        // (another process may have written the blob while this one waited for its turn)
        if cached.is_none() {
            loaded = path.as_deref().and_then(driver_cache::load);
            cached = loaded.as_ref().map(driver_cache::LoadedBlob::shared);
        }
        let from_cache = cached.as_ref().and_then(|bytes| {
            desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: bytes.as_ptr().cast(), CachedBlobSizeInBytes: bytes.len() };
            // a driver that cannot use the blob may compile from the DXIL itself and still
            // return Ok: only the pipeline carrying a blob of the size we handed over is a hit
            unsafe { self.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }
                .ok()
                .filter(|pso| unsafe { pso.GetCachedBlob() }.is_ok_and(|b| unsafe { b.GetBufferSize() } == bytes.len()))
        });
        // A stale blob is not a usable cache hit. Do not let its presence bypass the offline
        // size cap and start the expensive uncached driver compile.
        if cached.is_some() && from_cache.is_none() {
            if let Some(loaded) = loaded.take() {
                let _ = loaded.invalidate();
            }
        }
        if cached.is_some() && from_cache.is_none() && over_cap {
            self.capped.fetch_add(1, Ordering::Relaxed);
            self.capped_bytes.fetch_add(dxil.len(), Ordering::Relaxed);
            drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
            return Err(format!("The cached compute pipeline was rejected and {} bytes of DXIL exceed the limit", dxil.len()));
        }
        let r = if let Some(pso) = from_cache {
            if let (Some(((book, layout), path)), Some(blob)) = (receipt, cached.as_ref()) {
                book.record(warm_blobs::stamp(path, layout, blob));
            }
            self.reused.fetch_add(1, Ordering::Relaxed);
            Ok(pso)
        } else {
            desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
            let pso = unsafe { self.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) };
            let persist_path = path.as_deref().filter(|_| _claim.is_some() || cached.is_some());
            let persistence = pso.as_ref().ok().map(|pso| match persist_path {
                None => PersistenceOutcome::Disabled,
                Some(path) => match unsafe { pso.GetCachedBlob() } {
                    Ok(blob) => {
                        let bytes = unsafe { blob.GetBufferPointer() }.cast::<u8>();
                        let len = unsafe { blob.GetBufferSize() };
                        if bytes.is_null() || len == 0 {
                            PersistenceOutcome::BlobUnavailable
                        } else {
                            let bytes = unsafe { std::slice::from_raw_parts(bytes, len) };
                            if driver_cache::write(path, bytes) {
                                if let Some(((book, layout), _)) = receipt {
                                    book.record(warm_blobs::stamp(path, layout, bytes));
                                }
                                PersistenceOutcome::Saved
                            } else {
                                PersistenceOutcome::WriteFailed
                            }
                        }
                    }
                    Err(_) => PersistenceOutcome::BlobUnavailable,
                },
            });
            record_fresh_creation(persistence, &self.created, &self.persistence);
            if let Some(outcome) = persistence {
                if matches!(outcome, PersistenceOutcome::BlobUnavailable | PersistenceOutcome::WriteFailed)
                    && !self.persistence_warned.swap(true, Ordering::Relaxed)
                {
                    if let Some(path) = persist_path {
                        crate::xlog_warn!("Shader preparation: failed to persist driver blob {}; this pipeline will be rebuilt on the next run", path.display());
                    }
                }
            }
            pso
        };
        let library = match &r {
            Ok(pso) => pipeline_library::store(library_name, pso),
            Err(_) => Ok(()),
        };
        drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
        let done = match &r {
            Ok(_) => self.made.fetch_add(1, Ordering::Relaxed) + 1,
            Err(_) => self.refused.fetch_add(1, Ordering::Relaxed) + 1,
        };
        if done % 25 == 0 {
            eprintln!("  pipelines: {} made, {} refused", self.made.load(Ordering::Relaxed), self.refused.load(Ordering::Relaxed));
        }
        r.map(drop).map_err(|e| format!("Creating native compute pipeline ({} bytes of DXIL): {e}", dxil.len()))?;
        library
    }
}

fn pipeline_key(blob: &[u8]) -> u64 {
    crate::hle::d3d12x::real::shaders_key(&[blob.to_vec()])
}

fn parse_pipeline_keys(text: &str) -> std::collections::HashSet<u64> {
    text.lines().filter_map(|line| u64::from_str_radix(line.trim(), 16).ok()).filter(|&key| key != 0).collect()
}

/// The selector accepts the record hash printed by the compile log or the pipeline key written
/// to `used-pipelines.txt`. Return the keys this record satisfied so unmatched inputs can be named.
fn selected_keys(blob: &[u8], selected: &std::collections::HashSet<u64>) -> Vec<u64> {
    [shader::fnv64(blob), pipeline_key(blob)].into_iter().filter(|key| selected.contains(key)).collect()
}

/// A compute program a title used in an earlier run (`--used-only`): its pipeline's key, as the
/// runtime notes it (`real::shaders_key` of the one shader), is among `used`.
#[cfg(test)]
fn used_compute(blob: &[u8], used: &std::collections::HashSet<u64>) -> bool {
    used.contains(&pipeline_key(blob))
}

struct PipelineJob {
    used: bool,
    prewarm_indirect: bool,
    compiled: recomp::pso::Compiled,
}

fn order_pipeline_jobs(jobs: &mut [PipelineJob]) {
    jobs.sort_by(|a, b| b.used.cmp(&a.used).then(b.compiled.dxil.len().cmp(&a.compiled.dxil.len())));
}

fn make_pipelines(pipelines: &Pipelines, mut jobs: Vec<PipelineJob>) {
    order_pipeline_jobs(&mut jobs);
    let used = jobs.iter().filter(|job| job.used).count();
    println!("pipelines: {} compute programs ({used} used earlier: first), {} driver compiles at once", jobs.len(), pipelines.most);
    run_pipeline_jobs(jobs, pipelines.most, |job| pipelines.make_selected(&job.compiled, job.prewarm_indirect, None));
}

fn run_pipeline_jobs<T: Send>(jobs: Vec<T>, workers: usize, build: impl Fn(&T) + Sync) {
    let workers = workers.min(jobs.len());
    let jobs = Mutex::new(jobs.into_iter());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let job = jobs.lock().unwrap().next();
                let Some(job) = job else { break };
                // Release the job's buffers as soon as its driver build returns.
                build(&job);
            });
        }
    });
}

#[cfg(test)]
mod pipeline_selection_tests {
    use super::*;

    fn job(used: bool, bytes: usize) -> PipelineJob {
        PipelineJob {
            used,
            prewarm_indirect: false,
            compiled: recomp::pso::Compiled { dispatch_chain: None, dispatch_remap: None, dxil: vec![0; bytes], samplers: Vec::new() },
        }
    }

    #[test]
    fn earlier_pipelines_go_first_then_largest_first() {
        let mut jobs = vec![job(false, 10), job(true, 5), job(false, 900), job(true, 50)];
        order_pipeline_jobs(&mut jobs);
        let order: Vec<_> = jobs.iter().map(|job| (job.used, job.compiled.dxil.len())).collect();
        assert_eq!(order, [(true, 50), (true, 5), (false, 900), (false, 10)]);
    }

    #[test]
    fn offline_pipeline_jobs_drop_their_inputs_after_each_build() {
        use std::sync::Arc;

        let jobs: Vec<_> = (0..3).map(Arc::new).collect();
        let inputs: Vec<_> = jobs.iter().map(Arc::downgrade).collect();
        let next = AtomicUsize::new(0);
        run_pipeline_jobs(jobs, 1, |job| {
            let index = next.fetch_add(1, Ordering::Relaxed);
            assert_eq!(**job, index, "jobs keep their priority order");
            assert!(inputs[..index].iter().all(|input| input.upgrade().is_none()),
                "completed jobs still hold their inputs");
            assert!(inputs[index..].iter().all(|input| input.upgrade().is_some()),
                "active and queued jobs need their inputs");
        });
        assert_eq!(next.into_inner(), inputs.len());
        assert!(inputs.iter().all(|input| input.upgrade().is_none()));
    }

    #[test]
    fn selectors_name_record_and_pipeline_matches() {
        let blob = vec![1, 2, 3, 4];
        let record = shader::fnv64(&blob);
        let pipeline = pipeline_key(&blob);
        let selected = [record, pipeline, 0x1234].into_iter().collect();
        let mut matched = selected_keys(&blob, &selected);
        matched.sort_unstable();
        let mut expected = vec![record, pipeline];
        expected.sort_unstable();
        assert_eq!(matched, expected);
        assert_eq!(parse_pipeline_keys("not-a-key\n0000000000000000\n0000000000001234\n"), [0x1234].into_iter().collect());
    }
}

#[derive(Default)]
pub(crate) struct LaunchPreparation {
    pub adapter: String,
    pub done: AtomicUsize,
    pub total: AtomicUsize,
    pub cancel: std::sync::atomic::AtomicBool,
    /// The worker processes building compute pipelines (0: none; `done` and `total` then count
    /// the records translated here, with them the records the workers have passed).
    pub workers: AtomicUsize,
    /// The workers build graphics pipelines (`graphics_in_workers`): `done` and `total` count those.
    pub graphics: std::sync::atomic::AtomicBool,
    /// While replaying the discovered corpus, only verified native successes advance `done`.
    pub corpus: std::sync::atomic::AtomicBool,
    /// The shaders earlier runs used (`pso_async::earlier_used_shaders`), when there are any:
    /// only their records are translated before the title starts, and it translates another
    /// when it first uses it. `XCL_PREPARE_ALL=1`: every record, as without the list.
    pub only: Option<HashSet<u64>>,
}

impl LaunchPreparation {
    pub(crate) fn begin_corpus(&self, total: usize) {
        self.done.store(0, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
        self.corpus.store(true, Ordering::Release);
    }

    pub(crate) fn complete_corpus(&self, count: usize) -> Result<(), String> {
        if self.cancel.load(Ordering::Relaxed) { return Err("Pipeline preparation cancelled".into()); }
        if !self.corpus.load(Ordering::Acquire) || count > self.total.load(Ordering::Relaxed) {
            return Err("Native completion count exceeds the discovered corpus".into());
        }
        self.done.fetch_max(count, Ordering::Release);
        Ok(())
    }

    pub(crate) fn check_corpus_complete(&self) -> Result<(), String> {
        if self.cancel.load(Ordering::Relaxed) { return Err("Pipeline preparation cancelled".into()); }
        let done = self.done.load(Ordering::Acquire);
        let total = self.total.load(Ordering::Relaxed);
        if !self.corpus.load(Ordering::Acquire) || total == 0 || done != total {
            return Err(format!("Native pipeline preparation incomplete: {done} of {total} recipes confirmed"));
        }
        Ok(())
    }
}

/// Unpacked shader containers in the title's shaders directory. Do not follow directory
/// junctions into another tree or back into the shader directory.
fn loose_inputs(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    use std::os::windows::fs::MetadataExt;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.file_attributes() & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT != 0 { continue; }
        let path = entry.path();
        if metadata.is_dir() {
            loose_inputs(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("dxbc")) {
            files.push(path);
        }
    }
    Ok(())
}

pub(crate) fn prepare_for_launch(root: &Path, progress: &LaunchPreparation) -> Result<(), String> {
    let summary = pso_corpus::summary(root)?;
    if let Some(corpus) = &summary {
        if warm_library::reuse(corpus, &progress.adapter) {
            let (graphics, compute) = corpus.counts();
            let total = graphics.checked_add(compute).ok_or("Pipeline corpus count overflow")?;
            progress.begin_corpus(total);
            progress.complete_corpus(total)?;
            crate::xlog_info!("Pipeline preparation: reused the complete native pipeline libraries ({total} recipes)");
            return Ok(());
        }
    }
    let corpus = pso_corpus::discover(root)?;
    let cutoff = std::env::var("XCL_PREPARE_HISTORY_PRESENT").ok();
    let seconds = std::env::var("XCL_PREPARE_HISTORY_SECONDS").ok();
    if let Some(options) = milestone::launch_options(cutoff.as_deref(), seconds.as_deref())? {
        let inputs = title_inputs(root)?;
        if inputs.is_empty() { return Err("No shader records are available for history preparation".into()) }
        let mut windows = crate::gpu::uma::configured_runtime_windows() as u32;
        if windows > 1 && recomp::pso::heap_windows() { windows |= recomp::HEAP_WINDOWS; }
        let manifest = crate::gpu::recomp::dxc::cache_dir().join("pso-manifest.txt");
        let selected = milestone::history_selection(&manifest, options.cutoff)?;
        progress.done.store(0, Ordering::Relaxed);
        progress.total.store(selected.compute.len() + selected.graphics.len(), Ordering::Relaxed);
        progress.workers.store(1, Ordering::Relaxed);
        let began = std::time::Instant::now();
        let mut timing = milestone::HistoryTiming::new(began, options.budget);
        milestone::prepare_launch(&inputs, selected.compute, &options, windows, &progress.adapter, progress, began, &mut timing)?;
        prepare_history_graphics(&inputs, selected.graphics, windows, &progress.adapter, progress, began, options.budget, &mut timing)?;
        timing.finish();
        return Ok(());
    }
    prepare_title_inputs(root, progress, corpus.as_ref())?;
    if let Some(corpus) = corpus {
        corpus.prepare(progress)?;
        progress.check_corpus_complete()?;
    }
    Ok(())
}

fn title_inputs(root: &Path) -> Result<Vec<Input>, String> {
    let mut inputs = Vec::new();
    let pack = root.join("shaders.rpf");
    if pack.try_exists().map_err(|e| e.to_string())? {
        let data = std::fs::read(&pack).map_err(|e| format!("Reading {}: {e}", pack.display()))?;
        if !data.starts_with(pack::RPF9_MAGIC) { return Err(format!("Unrecognized shader pack: {}", pack.display())); }
        let records = pack::scan(&data, 0x40);
        if records.is_empty() { return Err(format!("No readable shader records in {}", pack.display())); }
        inputs.push(Input::Pack { data, records });
    }
    let loose = root.join("shaders");
    if loose.is_dir() {
        let mut files = Vec::new();
        loose_inputs(&loose, &mut files).map_err(|e| format!("Reading {}: {e}", loose.display()))?;
        if !files.is_empty() { inputs.push(Input::Files(files)); }
    }
    Ok(inputs)
}

fn prepare_title_inputs(root: &Path, progress: &LaunchPreparation, corpus: Option<&pso_corpus::Corpus>) -> Result<(), String> {
    let inputs = title_inputs(root)?;
    let pack = root.join("shaders.rpf");
    progress.total.store(inputs.iter().map(Input::len).sum(), Ordering::Relaxed);
    let mut windows = crate::gpu::uma::configured_runtime_windows() as u32;
    if windows > 1 && recomp::pso::heap_windows() { windows |= recomp::HEAP_WINDOWS; }
    let workers = prepare_workers();
    for input in &inputs {
        let coverage = corpus.filter(|c| c.compute_directory().is_some())
            .map(|corpus| ComputeCoverage::Published(corpus, Mutex::default(), AtomicUsize::new(0)));
        if progress.cancel.load(Ordering::Relaxed) { return Err("Shader preparation cancelled".into()); }
        // a pack: its stages here, its compute pipelines in worker processes (the driver builds
        // about two pipelines at a time for one process, whatever the processors)
        if workers > 1 && matches!(input, Input::Pack { .. }) {
            let n = input.len();
            precompile(input, usize::MAX, windows, None, None, false, None, None, None,
                std::time::Instant::now(), Some(progress), coverage.as_ref())?;
            if progress.cancel.load(Ordering::Relaxed) { return Err("Shader preparation cancelled".into()); }
            if coverage.as_ref().is_some_and(ComputeCoverage::all_staged) { continue; }
            pipelines_in_workers(&pack, windows, workers, n, progress, coverage.as_ref())?;
            continue;
        }
        precompile(input, usize::MAX, windows, None, None, true, None, None, None,
            std::time::Instant::now(), Some(progress), coverage.as_ref())?;
    }
    if progress.cancel.load(Ordering::Relaxed) { return Err("Shader preparation cancelled".into()); }
    if std::env::var("XCL_PREPARE_GRAPHICS").is_ok_and(|v| v.trim() == "0") { return Ok(()); }
    graphics_in_workers(&inputs, windows, workers, progress, corpus.and_then(|c| c.graphics_path()))
}

/// Only records verified byte-for-byte against the discovered corpus may leave their native
/// preparation to that pass. A worker also checks the index, key and bytes of its own pack.
enum ComputeCoverage<'a> {
    Published(&'a pso_corpus::Corpus, Mutex<BTreeMap<usize, u64>>, AtomicUsize),
    Slice(PathBuf, BTreeMap<usize, u64>),
}

const COVERAGE_MAP: &str = "XCL_PRECOMPILE_COVERAGE_MAP";
const COVERAGE_DIR: &str = "XCL_PRECOMPILE_COVERAGE_DIR";

struct CoverageFile(PathBuf);
impl Drop for CoverageFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

impl ComputeCoverage<'_> {
    fn all_staged(&self) -> bool {
        matches!(self, Self::Published(_, _, remaining) if remaining.load(Ordering::Relaxed) == 0)
    }

    fn covers(&self, index: usize, blob: &[u8]) -> Result<bool, String> {
        let key = shader::fnv64(blob);
        let path = match self {
            Self::Published(corpus, _, remaining) => match corpus.compute_path(key) {
                Some(path) => path.to_path_buf(),
                None => { remaining.fetch_add(1, Ordering::Relaxed); return Ok(false); }
            },
            Self::Slice(dir, keys) if keys.get(&index) == Some(&key) => dir.join(format!("{key:016x}.dxbc")),
            Self::Slice(..) => return Ok(false),
        };
        let staged = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if blob != staged {
            if shader::fnv64(&staged) != key { return Err(format!("Staged compute shader {key:016x} changed")); }
            if let Self::Published(_, _, remaining) = self { remaining.fetch_add(1, Ordering::Relaxed); }
            return Ok(false);
        }
        if let Self::Published(_, selected, _) = self { selected.lock().unwrap().insert(index, key); }
        Ok(true)
    }

    fn slice_file(&self) -> Result<Option<(CoverageFile, &Path)>, String> {
        use std::io::Write;
        let Self::Published(corpus, selected, _) = self else { return Ok(None) };
        let Some(dir) = corpus.compute_directory() else { return Ok(None) };
        let selected = selected.lock().unwrap();
        if selected.is_empty() { return Ok(None); }
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos();
        let path = std::env::temp_dir().join(format!("xcl-compute-coverage-{}-{nonce}.bin", std::process::id()));
        let mut file = std::fs::File::create_new(&path).map_err(|e| e.to_string())?;
        let cleanup = CoverageFile(path);
        let mut bytes = Vec::with_capacity(selected.len()*16);
        for (&i, &key) in selected.iter() { bytes.extend_from_slice(&(i as u64).to_le_bytes()); bytes.extend_from_slice(&key.to_le_bytes()); }
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        Ok(Some((cleanup, dir)))
    }

    fn worker() -> Result<Option<Self>, String> {
        if std::env::var_os("XCL_PRECOMPILE_PARENT").is_none() { return Ok(None); }
        let Some(path) = std::env::var_os(COVERAGE_MAP) else { return Ok(None) };
        let dir = std::env::var_os(COVERAGE_DIR).ok_or("Missing staged compute directory for worker")?;
        Self::read_slice(Path::new(&path), PathBuf::from(dir)).map(Some)
    }

    fn read_slice(path: &Path, dir: PathBuf) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() % 16 != 0 { return Err("Incomplete compute worker coverage map".into()); }
        let keys = bytes.chunks_exact(16).map(|b| (u64::from_le_bytes(b[..8].try_into().unwrap()) as usize,
            u64::from_le_bytes(b[8..].try_into().unwrap()))).collect();
        Ok(Self::Slice(dir, keys))
    }
}

/// `XCL_PREPARE_WORKERS`: the processes that build a pack's compute pipelines before a title
/// starts (1: in this process, as before). Unset: the logical processors less four, and no
/// more than 3 GB of the commit the system has left for each.
fn prepare_workers() -> usize {
    if let Some(n) = std::env::var("XCL_PREPARE_WORKERS").ok().and_then(|v| v.parse::<usize>().ok()) {
        return n.clamp(1, 32);
    }
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (used, limit) = crate::host::system_commit();
    workers_for_commit(cores, used, limit)
}

/// The workers for `cores` logical processors and the system's commit charge `used` and its
/// `limit`, both in bytes as `host::system_commit` gives them.
fn workers_for_commit(cores: usize, used: u64, limit: u64) -> usize {
    workers_for(cores, limit.saturating_sub(used) >> 20)
}

/// The workers for `cores` logical processors (threads, not cores) and `free_mb` of commit left.
fn workers_for(cores: usize, free_mb: u64) -> usize {
    cores.saturating_sub(4).min(32).min((free_mb / 3072) as usize).max(1)
}

/// The record count of a builder's progress line (`  400/26503 (12 s)`).
fn progress_count(line: &str) -> Option<usize> {
    let (d, rest) = line.trim().split_once('/')?;
    let (n, rest) = rest.split_once(' ')?;
    n.parse::<usize>().ok()?;
    if !rest.starts_with('(') { return None; }
    d.parse().ok()
}

/// The compute pipelines of `pack` built by `workers` processes of this program, each on its own
/// slice of the records (`XCL_PRECOMPILE_SLICE`). A failed worker prevents guest entry.
fn pipelines_in_workers(pack: &Path, windows: u32, workers: usize, records: usize, progress: &LaunchPreparation,
                        coverage: Option<&ComputeCoverage>) -> Result<(), String> {
    let me = std::env::current_exe().map_err(|e| e.to_string())?;
    let series_s = crate::hle::d3d12x::pso_async::profile_is_series_s();
    let map = coverage.map(|c| c.slice_file()).transpose()?.flatten();
    // only the pipelines an earlier run of the title used, when that list exists and the title's
    // shaders are limited to those an earlier run used (`LaunchPreparation::only`; the title
    // builds another when it first uses it)
    let used_only = progress.only.is_some() && !crate::hle::d3d12x::pso_async::earlier_used_keys().is_empty();
    let t0 = std::time::Instant::now();
    crate::xlog_info!("Shader preparation: compute pipelines in {workers} worker processes{}", if used_only { ", those an earlier run used" } else { "" });
    let (cancelled, failed, _) = in_workers(workers, records, progress, |i| {
        let mut c = pipeline_worker_command(&me, pack, windows, used_only, &format!("{i}/{workers}"), series_s);
        if let Some((file, dir)) = &map { c.env(COVERAGE_MAP, &file.0).env(COVERAGE_DIR, dir); }
        if std::env::var_os("XCL_ADAPTER").is_none() && !progress.adapter.is_empty() { c.env("XCL_ADAPTER", &progress.adapter); }
        c
    });
    if cancelled { return Err("Shader preparation cancelled".into()); }
    if failed != 0 { return Err(format!("{failed} of {workers} compute pipeline workers failed")); }
    crate::xlog_info!("Shader preparation: compute pipelines done in {:.0} s ({failed} of {workers} workers failed)", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Compute pipeline worker `slice` (`i/n`) of `pack`, with the console profile: the worker reads
/// that profile's used-pipeline list (`used_pipelines_name`).
fn pipeline_worker_command(me: &Path, pack: &Path, windows: u32, used_only: bool, slice: &str, series_s: bool) -> std::process::Command {
    let mut c = worker_command(me);
    c.arg("shaders").arg(pack).args(["--precompile", "--only", "cs", "--pipelines", "--windows"]).arg((windows & !recomp::HEAP_WINDOWS).to_string());
    if used_only { c.arg("--used-only"); }
    c.env("XCL_PRECOMPILE_SLICE", slice);
    c.env("XCL_DEVICE_TYPE", if series_s { "lockhart" } else { "anaconda" });
    c
}

/// A preparation worker: this program, not a host and not a launcher's child, a builder that
/// ends when this process does.
fn worker_command(me: &Path) -> std::process::Command {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let mut c = std::process::Command::new(me);
    for gone in ["XCL_HOST", "XCL_PREPARATION_PARENT", "XCL_PREPARATION_MENU", "XCL_RUN_SECONDS", COVERAGE_MAP, COVERAGE_DIR] { c.env_remove(gone); }
    c.env("XCL_PRECOMPILE_PARENT", std::process::id().to_string());
    if std::env::var_os("XCL_DXC_THREADS").is_none() { c.env("XCL_DXC_THREADS", "2"); }
    if std::env::var_os("XCL_PIPELINE_THREADS").is_none() { c.env("XCL_PIPELINE_THREADS", "1"); }
    c.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::piped()).creation_flags(CREATE_NO_WINDOW);
    c
}

/// `workers` worker processes (`command(i)`), their progress lines (`progress_count`, each of its
/// own part) the count of `records` in all on the screen while they run: whether preparation was
/// cancelled (they are ended then), how many failed, and the summary line each wrote last
/// (`graphics_summary`).
fn in_workers(workers: usize, records: usize, progress: &LaunchPreparation, mut command: impl FnMut(usize) -> std::process::Command) -> (bool, usize, Vec<String>) {
    use std::io::BufRead;
    // the workers' pass has its own count on the screen: the records they have passed
    let (done_before, total_before) = (progress.done.load(Ordering::Relaxed), progress.total.load(Ordering::Relaxed));
    progress.done.store(0, Ordering::Relaxed);
    progress.total.store(records, Ordering::Relaxed);
    progress.workers.store(workers, Ordering::Relaxed);
    let counts: Vec<AtomicUsize> = (0..workers).map(|_| AtomicUsize::new(0)).collect();
    let summaries: Vec<Mutex<String>> = (0..workers).map(|_| Mutex::new(String::new())).collect();
    let mut children = Vec::new();
    for i in 0..workers {
        match command(i).spawn() {
            Ok(child) => children.push((i, child)),
            Err(e) => crate::xlog_warn!("Shader preparation: worker {i} did not start: {e}"),
        }
    }
    let cancelled = std::thread::scope(|s| {
        for (i, child) in children.iter_mut() {
            let Some(err) = child.stderr.take() else { continue };
            let (count, summary) = (&counts[*i], &summaries[*i]);
            s.spawn(move || {
                // (a builder ends its progress lines with a line feed; others pass by)
                let mut r = std::io::BufReader::new(err);
                let mut buf = Vec::new();
                while r.read_until(b'\n', &mut buf).is_ok_and(|n| n > 0) {
                    for part in String::from_utf8_lossy(&buf).split(['\r', '\n']) {
                        if let Some(d) = progress_count(part) { count.store(d, Ordering::Relaxed); }
                        if graphics_summary(part).is_some() { *summary.lock().unwrap() = part.to_string(); }
                    }
                    buf.clear();
                }
            });
        }
        loop {
            // (a worker counts the records of its own slice)
            let d: usize = counts.iter().map(|c| c.load(Ordering::Relaxed)).sum();
            progress.done.store(d.min(records), Ordering::Relaxed);
            if progress.cancel.load(Ordering::Relaxed) {
                for (_, child) in children.iter_mut() { let _ = child.kill(); }
                return true;
            }
            if children.iter_mut().all(|(_, child)| !matches!(child.try_wait(), Ok(None))) { return false; }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });
    let mut failed = workers - children.len();
    for (i, child) in children.iter_mut() {
        match child.wait() {
            Ok(st) if st.success() => {}
            Ok(st) => { failed += 1; crate::xlog_warn!("Shader preparation: worker {i} ended with {st}"); }
            Err(e) => { failed += 1; crate::xlog_warn!("Shader preparation: worker {i}: {e}"); }
        }
    }
    progress.workers.store(0, Ordering::Relaxed);
    progress.total.store(total_before, Ordering::Relaxed);
    progress.done.store(done_before, Ordering::Relaxed);
    (cancelled, failed, summaries.into_iter().map(|s| s.into_inner().unwrap()).collect())
}

/// The graphics pipelines earlier runs used (`gfx_records`), built before the title starts by
/// `workers` processes of this program (`graphics_worker`) the way the title builds them, so the
/// title finds their translations and driver blobs on disk and builds each in milliseconds. In
/// worker processes as the compute pipelines: the driver builds about two pipelines at a time for
/// one process, and a driver that ends a worker ends no title. Their shaders come from the title's
/// shader files (`inputs`) by key, as those of `used-shaders.txt`; each worker gets a file of its
/// part (`gfx_records`' format, with the shaders it needs). A worker must confirm every recipe.
fn exact_graphics_records(
    recipe_ids: &[u64], payloads: impl IntoIterator<Item = Vec<u8>>,
) -> Result<Vec<GraphicsHistoryRecord>, String> {
    let wanted: HashSet<_> = recipe_ids.iter().copied().collect();
    let mut found = HashMap::new();
    for payload in payloads {
        let id = gfx_records::payload_key(&payload);
        if !wanted.contains(&id) || found.contains_key(&id) { continue; }
        let entry = gfx_records::decode(&payload).ok_or_else(|| format!("Unrecognized graphics history recipe {id:016x}"))?;
        if let Some(record) = GraphicsHistoryRecord::from_entry(entry) { found.insert(id, record); }
    }
    let missing: Vec<_> = recipe_ids.iter().filter(|id| !found.contains_key(id)).map(|id| format!("{id:016x}")).collect();
    if !missing.is_empty() {
        return Err(format!("Missing exact graphics history recipes ({}): {}", missing.len(), missing.join(", ")));
    }
    Ok(recipe_ids.iter().map(|id| found.remove(id).unwrap()).collect())
}

fn prepare_history_graphics(
    inputs: &[Input], recipe_ids: Vec<u64>, windows: u32, adapter: &str,
    progress: &LaunchPreparation, began: std::time::Instant, budget: std::time::Duration,
    timing: &mut milestone::HistoryTiming,
) -> Result<(), String> {
    if recipe_ids.is_empty() || began.elapsed() >= budget { return Ok(()) }
    let mut payloads = Vec::new();
    for path in [gfx_records::path(), gfx_records::shared_history_path()] {
        if !path.try_exists().map_err(|e| e.to_string())? { continue; }
        payloads.extend(gfx_records::read_history_payloads(&path)?);
    }
    let records = exact_graphics_records(&recipe_ids, payloads)?;
    let shader_keys: HashSet<_> = records.iter().flat_map(|record| record.identity().1.iter().copied())
        .filter(|key| *key != 0).collect();
    let shaders = shaders_by_key(inputs, shader_keys, progress);
    let (ready, missing) = buildable(records.iter(), &shaders);
    if missing != 0 || ready.len() != records.len() {
        return Err(format!("{missing} exact graphics history recipes have missing or mismatched shaders"));
    }
    recomp::pso::set_title_windows(windows);
    recomp::dxc::set_memory_reserve(2);
    let cache_before = crate::hle::d3d12x::real::driver_cache_stats();
    let pipelines = Pipelines::for_adapter(adapter)?;
    let device = crate::hle::d3d12x::real::prepare_device(pipelines.dev.clone());
    let mut confirmed = 0usize;
    for (recipe_id, record) in recipe_ids.iter().copied().zip(&records) {
        if progress.cancel.load(Ordering::Relaxed) { return Err("Pipeline preparation cancelled".into()) }
        if began.elapsed() >= budget { break }
        let code = |key| shaders.get(&key).map(Vec::as_slice);
        let kind = if matches!(record, GraphicsHistoryRecord::Mesh(_)) { "mesh" } else { "graphics" };
        let started = std::time::Instant::now();
        let ok = unsafe { match record {
            GraphicsHistoryRecord::Legacy(record) => {
                let desc = record.desc(record.shaders.map(code));
                crate::hle::d3d12x::real::prepare_graphics(device, &desc, record.depth_bounds)
            }
            GraphicsHistoryRecord::Mesh(record) =>
                crate::hle::d3d12x::real::replay_mesh_record(device, record, record.shaders.map(code)).is_ok(),
        }};
        timing.record(kind, recipe_id, started.elapsed());
        if !ok { return Err(format!("Graphics history recipe {:016x} failed", record.identity().0)) }
        confirmed += 1;
        progress.done.fetch_add(1, Ordering::Relaxed);
    }
    crate::xlog_info!(
        "History preparation: {confirmed} of {} exact graphics pipelines confirmed through the launch milestone in {:.1} s",
        records.len(), began.elapsed().as_secs_f64()
    );
    let cache_after = crate::hle::d3d12x::real::driver_cache_stats();
    crate::xlog_info!(
        "History preparation graphics cache outcomes: {} driver-blob hits, {} rejected blobs, {} new blobs",
        cache_after.0.saturating_sub(cache_before.0), cache_after.1.saturating_sub(cache_before.1),
        cache_after.2.saturating_sub(cache_before.2)
    );
    Ok(())
}

fn graphics_in_workers(inputs: &[Input], windows: u32, workers: usize, progress: &LaunchPreparation, staged_graphics: Option<&Path>) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let path = gfx_records::path();
    let local = if path.try_exists().map_err(|e| e.to_string())? { gfx_records::read_history_payloads(&path)? } else { Vec::new() };
    let mut records = Vec::new();
    let mut embedded = std::collections::HashMap::new();
    for payload in local {
        match gfx_records::decode(&payload).ok_or("Unrecognized graphics history entry")? {
            gfx_records::Entry::Shader(key, code) => {
                if crate::hle::d3d12x::real::shader_key(&code) != key {
                    return Err(format!("Graphics history shader {key:016x} failed its identity check"));
                }
                embedded.insert(key, code);
            }
            entry => if let Some(record) = GraphicsHistoryRecord::from_entry(entry) { records.push((payload, record)); },
        }
    }
    let mut local_count = records.len();
    let legacy = gfx_records::shared_history_path();
    if !inputs.is_empty() && legacy != path {
        let scoped: HashSet<_> = records.iter().map(|(bytes, _)| bytes.as_slice()).collect();
        let candidates: Vec<_> = gfx_records::read(&legacy).into_iter().filter(|bytes| !scoped.contains(bytes.as_slice()))
            .filter_map(|payload| Some((GraphicsHistoryRecord::from_entry(gfx_records::decode(&payload)?)?, payload))).collect();
        records.extend(candidates.into_iter().map(|(record, payload)| (payload, record)));
    }
    if records.is_empty() {
        crate::xlog_info!("Shader preparation: no graphics pipelines of an earlier run to build");
        return Ok(());
    }
    if let Some(staged) = staged_graphics {
        let before = records.len();
        local_count = merge_staged_history(staged, &mut records, local_count, &mut embedded)?;
        if before != records.len() {
            crate::xlog_info!("Shader preparation: the mandatory staged job covers {} exact graphics history recipes", before - records.len());
        }
        if records.is_empty() { return Ok(()); }
    }
    let wanted: HashSet<_> = records.iter().flat_map(|(_, record)| record.identity().1.iter().copied())
        .filter(|&key| key != 0 && !embedded.contains_key(&key)).collect();
    let mut shaders = if wanted.is_empty() { HashMap::new() } else { shaders_by_key(inputs, wanted, progress) };
    shaders.extend(embedded);
    if progress.cancel.load(Ordering::Relaxed) { return Err("Shader preparation cancelled".into()); }
    let (buildable, missing) = buildable(records.iter().map(|(_, r)| r), &shaders);
    let local_missing = local_count - buildable.iter().take_while(|&&i| i < local_count).count();
    if local_missing != 0 {
        crate::xlog_info!("Shader preparation: deferring {local_missing} of {local_count} recorded title graphics recipes until their shader sources are available");
    }
    let shared_missing = missing - local_missing;
    if shared_missing != 0 { crate::xlog_info!("Shader preparation: {shared_missing} shared graphics recipes have no matching shaders in this title"); }
    if buildable.is_empty() { return Ok(()); }
    let workers = workers.clamp(1, buildable.len());
    // each worker's part, with the shaders it needs (removed when they are done)
    let mut jobs = Vec::new();
    for i in 0..workers {
        let path = std::env::temp_dir().join(format!("xcl-prepare-{}-{i}.bin", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut w = gfx_records::Writer::open(&path, u64::MAX).map_err(|e| format!("{}: {e}", path.display()))?;
        let part: Vec<usize> = buildable.iter().copied().skip(i).step_by(workers).collect();
        let mut keys = HashSet::new();
        for &k in &part {
            w.add(&records[k].0).map_err(|e| format!("{}: {e}", path.display()))?;
            keys.extend(records[k].1.identity().1.iter().copied().filter(|&key| key != 0));
        }
        for key in keys {
            w.add(&gfx_records::shader_entry(key, &shaders[&key])).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        jobs.push((path, part.len()));
    }
    let me = std::env::current_exe().map_err(|e| e.to_string())?;
    crate::xlog_info!("Shader preparation: the {} graphics pipelines earlier runs used, in {workers} worker processes", buildable.len());
    progress.graphics.store(true, Ordering::Relaxed);
    let (cancelled, failed, summaries) = in_workers(workers, buildable.len(), progress, |i| {
        let mut c = worker_command(&me);
        c.arg("shaders").arg(&jobs[i].0).args(["--graphics-records", "--windows"]).arg((windows & !recomp::HEAP_WINDOWS).to_string());
        // (the title's adapter: its blobs are the ones it takes)
        if std::env::var_os("XCL_ADAPTER").is_none() && !progress.adapter.is_empty() { c.env("XCL_ADAPTER", &progress.adapter); }
        c
    });
    progress.graphics.store(false, Ordering::Relaxed);
    for (path, _) in &jobs { let _ = std::fs::remove_file(path); }
    if cancelled { return Err("Shader preparation cancelled".into()); }
    let sum = summaries.iter().filter_map(|s| graphics_summary(s)).fold([0usize; 4], |a, b| std::array::from_fn(|k| a[k] + b[k]));
    if failed != 0 || sum[1] != 0 || sum[0] != buildable.len() {
        return Err(format!("Graphics preparation incomplete: {} of {} confirmed, {} refused, {failed} workers failed",
            sum[0], buildable.len(), sum[1]));
    }
    crate::xlog_info!(
        "Shader preparation: graphics pipelines done in {:.0} s: {} made, {} failed; driver blobs: {} from disk, {} kept ({failed} of {workers} workers failed)",
        t0.elapsed().as_secs_f64(), sum[0], sum[1], sum[2], sum[3]
    );
    Ok(())
}

/// Only identical serialized descriptors can be left for the subsequent mandatory staged job.
fn merge_staged_history(
    staged: &Path, records: &mut Vec<(Vec<u8>, GraphicsHistoryRecord)>, local_count: usize,
    embedded: &mut HashMap<u64, Vec<u8>>,
) -> Result<usize, String> {
    let mut covered: HashMap<&[u8], bool> = records.iter().map(|(p, _)| (p.as_slice(), false)).collect();
    let wanted: HashSet<u64> = records.iter().flat_map(|(_, r)| r.identity().1.iter().copied()).filter(|k| *k != 0).collect();
    gfx_records::visit_payloads(staged, |payload| {
        if let Some((key, code)) = gfx_records::shader_payload(payload) {
            if wanted.contains(&key) {
                if crate::hle::d3d12x::real::shader_key(code) != key || embedded.get(&key).is_some_and(|prior| prior.as_slice() != code) {
                    return Err(format!("Staged graphics shader {key:016x} failed its identity check"));
                }
                embedded.entry(key).or_insert_with(|| code.to_vec());
            }
        } else if let Some(present) = covered.get_mut(payload) {
            *present = true;
        }
        Ok(())
    })?;
    let keep: Vec<bool> = records.iter().map(|(p, _)| !covered[p.as_slice()]).collect();
    let remaining_local = keep[..local_count].iter().filter(|&&k| k).count();
    let mut keep = keep.into_iter();
    records.retain(|_| keep.next().unwrap());
    Ok(remaining_local)
}

/// The bytes of the shaders `wanted` (`real::shader_key`) in the title's shader files, looked
/// for on every processor.
fn shaders_by_key(inputs: &[Input], wanted: HashSet<u64>, progress: &LaunchPreparation) -> std::collections::HashMap<u64, Vec<u8>> {
    let found = Mutex::new(std::collections::HashMap::new());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    for input in inputs {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= input.len() || progress.cancel.load(Ordering::Relaxed) { break; }
                    let Some(blob) = input.get(i).1 else { continue };
                    let key = crate::hle::d3d12x::real::shader_key(&blob);
                    if wanted.contains(&key) { found.lock().unwrap().insert(key, blob); }
                });
            }
        });
    }
    found.into_inner().unwrap()
}

enum GraphicsHistoryRecord {
    Legacy(gfx_records::Record),
    Mesh(gfx_records::MeshRecord),
}

fn legacy_two_pass(keys: &[u64; 5], shaders: &HashMap<u64, Vec<u8>>) -> bool {
    keys[2] == 0 && keys[3] == 0
        && shaders.get(&keys[0]).is_some_and(|vs| recomp::pso::is_console(vs))
        && shaders.get(&keys[4]).is_some_and(|gs| recomp::pso::is_console(gs))
}

impl GraphicsHistoryRecord {
    fn from_entry(entry: gfx_records::Entry) -> Option<Self> {
        match entry {
            gfx_records::Entry::Pipeline(record) => Some(Self::Legacy(record)),
            gfx_records::Entry::MeshPipeline(record) => Some(Self::Mesh(record)),
            gfx_records::Entry::Shader(..) => None,
        }
    }

    fn identity(&self) -> (u64, &[u64]) {
        match self {
            Self::Legacy(record) => (record.key, &record.shaders),
            Self::Mesh(record) => (record.key, &record.shaders),
        }
    }
}

/// The records (by their index) whose shaders are all in `shaders` and make their set's key, and
/// how many are not.
fn buildable<'a>(records: impl Iterator<Item = &'a GraphicsHistoryRecord>, shaders: &HashMap<u64, Vec<u8>>) -> (Vec<usize>, usize) {
    let (mut ok, mut missing) = (Vec::new(), 0);
    for (i, record) in records.enumerate() {
        let (key, keys) = record.identity();
        let blobs: Option<Vec<&[u8]>> = keys.iter().filter(|&&k| k != 0).map(|k| shaders.get(k).map(Vec::as_slice)).collect();
        match blobs {
            Some(blobs) if crate::hle::d3d12x::real::shader_set_key(blobs.iter().copied()) == key => ok.push(i),
            _ => missing += 1,
        }
    }
    (ok, missing)
}

/// The counts of a graphics worker's last line (`graphics pipelines: 5 made, 1 failed; driver
/// blobs: 2 from disk, 3 kept`): made, failed, blobs from disk, blobs kept.
fn graphics_summary(line: &str) -> Option<[usize; 4]> {
    let rest = line.trim().strip_prefix("graphics pipelines: ")?;
    let mut n = rest.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse::<usize>());
    let out = [n.next()?.ok()?, n.next()?.ok()?, n.next()?.ok()?, n.next()?.ok()?];
    n.next().is_none().then_some(out)
}

fn corpus_threads(requested: Option<usize>, default: usize) -> usize {
    requested.or_else(|| std::env::var("XCL_PIPELINE_THREADS").ok()?.parse().ok()).unwrap_or(default).clamp(1, 32)
}

/// `xcl shaders <job> --graphics-records` (a worker of `graphics_in_workers`): the graphics
/// pipelines of a job file built as the title builds them, with the
/// title's memory windows (`windows`) and on its adapter (`XCL_ADAPTER`), `XCL_PIPELINE_THREADS`
/// at a time. A progress line a pipeline, then the summary `graphics_summary` reads.
fn graphics_worker(job: &Path, windows: u32, requested: Option<usize>) -> i32 {
    use crate::hle::d3d12x::real;
    end_with_parent();
    let mut shaders = std::collections::HashMap::new();
    let mut records = Vec::new();
    let entries = match gfx_records::read_job(job) {
        Ok(entries) => entries,
        Err(e) => { eprintln!("xcl shaders: {}: {e}", job.display()); return 2; }
    };
    let mut recipe_keys = Vec::new();
    for (key, entry) in entries {
        match entry {
            gfx_records::Entry::Shader(key, code) => {
                if real::shader_key(&code) != key || shaders.get(&key).is_some_and(|prior| prior != &code) {
                    eprintln!("xcl shaders: invalid or conflicting shader {key:016x} in {}", job.display());
                    return 2;
                }
                shaders.insert(key, code);
            }
            entry => if let Some(record) = GraphicsHistoryRecord::from_entry(entry) { records.push(record); recipe_keys.push(key); },
        }
    }
    let n = records.len();
    let (ready, missing) = buildable(records.iter(), &shaders);
    if missing != 0 { eprintln!("xcl shaders: {missing} graphics records have missing or mismatched shaders"); }
    if ready.is_empty() {
        eprintln!("graphics pipelines: 0 made, {missing} failed; driver blobs: 0 from disk, 0 kept");
        return if missing == 0 { 0 } else { 2 };
    }
    // (no title memory here: its windows are given)
    crate::gpu::uma::leave_out();
    recomp::pso::set_title_windows(windows);
    recomp::dxc::set_memory_reserve(2);
    let library_path = std::env::var_os("XCL_PIPELINE_LIBRARY_OUT").map(PathBuf::from);
    let (device, book) = match Pipelines::for_adapter("auto") {
        Ok(p) => {
            if library_path.is_some() {
                match pipeline_library::begin(
                    &p.dev,
                    library_path.as_ref().map(|path| path.with_extension("")).as_deref(),
                ) {
                    Ok(_) => {}
                    Err(e) => { eprintln!("xcl shaders: {e}"); return 2; }
                }
            }
            let cache = recomp::dxc::cache_dir();
            let book = warm_replay::native_key(&p.dev).and_then(|key| {
                let known = warm_replay::driver_files(&cache.join("driver")).unwrap_or_default();
                warm_replay::Index::open(&cache, &known, warm_blobs::digest(&(key, &recipe_keys)), windows)
            });
            (real::prepare_device(p.dev), book)
        },
        Err(e) => { eprintln!("xcl shaders: no pipeline device: {e}"); return 2; }
    };
    let threads = corpus_threads(requested, 1).min(ready.len());
    let verified = AtomicUsize::new(0);
    let (next, made, failed) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(missing));
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&index) = ready.get(i) else { break };
                let r = &records[index];
                let code = |key| shaders.get(&key).map(Vec::as_slice);
                let two_pass = match r {
                    GraphicsHistoryRecord::Mesh(_) => true,
                    GraphicsHistoryRecord::Legacy(record) => legacy_two_pass(&record.shaders, &shaders),
                };
                // (a pipeline that panics fails; the others are built)
                let key = recipe_keys[index];
                let run = || std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match r {
                    GraphicsHistoryRecord::Legacy(record) => {
                        let desc = record.desc(record.shaders.map(code));
                        unsafe { real::prepare_graphics(device, &desc, record.depth_bounds) }
                    }
                    GraphicsHistoryRecord::Mesh(record) => {
                        unsafe { real::replay_mesh_record(device, record, record.shaders.map(code)).is_ok() }
                    }
                })).unwrap_or(false);
                let ok = if let Some(book) = &book {
                    if !pipeline_library::building() && book.can_defer(key) {
                        verified.fetch_add(1, Ordering::Relaxed);
                        true
                    } else {
                        let (ok, files) = warm_replay::capture(run);
                        if ok { book.add(key, files, two_pass); }
                        ok
                    }
                } else { run() };
                if ok {
                    let d = made.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!("  {d}/{n} ({:.0} s)", t0.elapsed().as_secs_f64());
                } else {
                    failed.fetch_add(1, Ordering::Relaxed);
                    eprintln!("xcl shaders: failed graphics record {} (shader set {:016x})", index + 1, r.identity().0);
                }
            });
        }
    });
    if let Some(book) = book { book.save(); }
    eprintln!("previously verified native graphics recipes: {}", verified.into_inner());
    let (hits, _, stored) = real::driver_cache_stats();
    let mut failed = failed.into_inner();
    if failed == 0 {
        if let Some(path) = library_path.as_deref().filter(|_| pipeline_library::building()) {
            match pipeline_library::finish(path) {
                Ok(n) => eprintln!("pipeline library: {n} graphics entries"),
                Err(e) => { eprintln!("xcl shaders: {e}"); failed += 1; }
            }
        }
    } else {
        pipeline_library::abort();
    }
    eprintln!("graphics pipelines: {} made, {failed} failed; driver blobs: {hits} from disk, {stored} kept", made.into_inner());
    if failed == 0 { 0 } else { 2 }
}

/// `XCL_PRECOMPILE_PARENT=pid` (set for a title's preparation workers): this process ends when
/// that one does, so no builder is left on the GPU behind a title that was ended.
fn end_with_parent() {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, WaitForSingleObject, INFINITE, PROCESS_SYNCHRONIZE};
    let Some(pid) = std::env::var("XCL_PRECOMPILE_PARENT").ok().and_then(|v| v.parse::<u32>().ok()) else { return };
    let h = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, pid) };
    if h.is_null() {
        // (gone already)
        std::process::exit(3);
    }
    let h = h as usize;
    std::thread::spawn(move || unsafe {
        WaitForSingleObject(h as _, INFINITE);
        CloseHandle(h as _);
        std::process::exit(3);
    });
}

fn precompile_slice(startup_owner: bool, value: Option<&str>) -> Option<(usize, usize)> {
    if startup_owner { return None; }
    let (i, n) = value?.split_once('/')?;
    Some((i.trim().parse().ok()?, n.trim().parse::<usize>().ok().filter(|n| *n > 0)?))
}

fn used_indirect_split(blob: &[u8], used: &std::collections::HashSet<u64>) -> bool {
    used.contains(&crate::gpu::shader::fnv64(blob))
}

/// Standalone graphics variants whose runtime cache keys can be known without a pipeline pairing.
/// Runtime selection also uses the first hardware stage of the first matching FSSM (4: VS, 5: PS).
fn standalone_graphics_variants(blob: &[u8], sh: &xbox::XboxShader, split: bool, windows: u32) -> Vec<(recomp::pso::Stage, u32, bool, u32)> {
    [(recomp::pso::Stage::Vertex, 4), (recomp::pso::Stage::Pixel, 5)]
        .into_iter()
        .filter(|&(stage, hw)| {
            let Some(f) = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == hw)) else { return false };
            matches!((recomp::interface(blob, sh, f), stage), (Some(recomp::Mode::Vertex(_)), recomp::pso::Stage::Vertex) | (Some(recomp::Mode::Pixel(_)), recomp::pso::Stage::Pixel))
        })
        .map(|(stage, _)| (stage, 0, split, windows))
        .collect()
}

fn precompile(input: &Input, limit: usize, windows: u32, split: Option<bool>, only: Option<&str>, pipelines: bool,
    selected: Option<&std::collections::HashSet<u64>>, indirect: Option<&std::collections::HashSet<u64>>,
    pipeline_threads: Option<usize>, t0: std::time::Instant, startup: Option<&LaunchPreparation>,
    coverage: Option<&ComputeCoverage>) -> Result<(), String> {
    if startup.is_none() { end_with_parent(); }
    let mandatory = startup.is_some() || std::env::var_os("XCL_PRECOMPILE_PARENT").is_some();
    let pipelines = if pipelines {
        let mut p = Pipelines::for_adapter(startup.map_or("auto", |p| p.adapter.as_str()))
            .map_err(|e| format!("No pipeline device: {e}"))?;
        if let Some(threads) = pipeline_threads {
            p.most = threads.max(1);
        }
        if mandatory { p.warm_blobs = warm_blobs::Verified::for_device(&p.dev); }
        if mandatory && std::env::var_os("XCL_PIPELINE_MAX_KB").is_none() {
            p.max_dxil = 0;
        }
        Some(p)
    } else {
        None
    };
    let pipelines = pipelines.as_ref();
    let n = input.len().min(limit);
    // `XCL_PRECOMPILE_SLICE=i/n` (experiment): this process takes only every n-th shader, from the
    // i-th, so n processes share the work and each has its own driver compiler
    let slice = precompile_slice(startup.is_some(), std::env::var("XCL_PRECOMPILE_SLICE").ok().as_deref());
    // a progress line every 200 records; every record for a title waiting on this builder
    // (`pipelines_in_workers` shows the count)
    let every = if std::env::var_os("XCL_PRECOMPILE_PARENT").is_some() { 1 } else { 200 };
    let next = AtomicUsize::new(0);
    let (done, compiled, cached, failed, skipped) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let matched: Mutex<std::collections::HashSet<u64>> = Mutex::new(Default::default());
    let pipeline_jobs: Mutex<Vec<PipelineJob>> = Mutex::new(Vec::new());
    let used_earlier = if pipelines.is_some() && startup.is_none() {
        crate::hle::d3d12x::pso_async::earlier_used_keys()
    } else {
        Default::default()
    };
    if let Some(selected) = selected {
        println!("selection: {} record hashes or pipeline keys requested", selected.len());
    }
    if let Some(i) = indirect {
        println!("--used-only: {} split programs earlier runs dispatched indirectly get direct and indirect pieces/whole", i.len());
    }
    // the compiles run on the DXC pool; these threads translate and keep it fed
    let threads = recomp::dxc::compile_threads() + 2;
    // no title runs: only a little commit is kept free besides the compiles
    if startup.is_none() { recomp::dxc::set_memory_reserve(2); }
    // VS / PS: the 32-lane form wherever the GPU can run 32-lane waves (`pso::graphics_wave32`);
    // `--split` sets both
    let startup_device = pipelines.filter(|_| startup.is_some());
    let graphics_split = split.unwrap_or_else(|| startup_device.map_or_else(recomp::runner::graphics_wave32, |p| recomp::pso::graphics_wave32_for_device(&p.dev)));
    let split = split.unwrap_or_else(|| startup_device.map_or_else(|| recomp::pso::split_wave64_forced() || recomp::runner::wave64_supported().is_ok_and(|w| !w), |p| recomp::pso::wave32_only_for_device(&p.dev)));
    // A graphics-stage proof skips translation only. Compute records still pass through
    // coverage and native preparation, including a changed or newly staged compute job.
    let pack_proof = startup.and_then(|owner| {
        let Input::Pack { data, .. } = input else { return None };
        if n != input.len() || only.is_some() || selected.is_some() || slice.is_some() { return None; }
        let extra_device;
        let device = match pipelines {
            Some(p) => &p.dev,
            None => {
                extra_device = Pipelines::for_adapter(&owner.adapter).ok()?;
                &extra_device.dev
            }
        };
        warm_pack::Stages::open(warm_replay::native_key(device)?, data, n, windows, split, graphics_split)
    });
    println!(
        "precompiling {n} shaders into {} ({} compile threads; {} memory windows{}{})",
        recomp::dxc::cache_dir().display(),
        recomp::dxc::compile_threads(),
        windows & !(recomp::HEAP_WINDOWS | recomp::HEAP_FLAT),
        if windows & recomp::HEAP_WINDOWS != 0 { " from the descriptor heap" } else { "" },
        if split { ", split wave64 for a GPU with 32-lane waves" } else { "" }
    );
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                if startup.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) { break; }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                if slice.is_some_and(|(part, count)| i % count != part) {
                    continue;
                }
                let record_done = || {
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if let Some(p) = startup { p.done.fetch_add(1, Ordering::Relaxed); }
                    if d % every == 0 { eprintln!("  {d}/{n} ({:.0} s)", t0.elapsed().as_secs_f64()); }
                };
                if pack_proof.as_ref().is_some_and(|p| p.covered(i)) {
                    record_done();
                    continue;
                }
                let (name, blob) = input.get(i);
                let Some(blob) = blob else {
                    if startup.is_some() { failed.fetch_add(1, Ordering::Relaxed); }
                    record_done();
                    continue;
                };
                if let Some(selected) = selected {
                    let keys = selected_keys(&blob, selected);
                    if keys.is_empty() {
                        record_done();
                        continue;
                    }
                    matched.lock().unwrap().extend(keys);
                }
                let Some(sh) = xbox::parse(&blob) else {
                    if startup.is_some() && !matches!(shader::classify(&blob).0, shader::Kind::Dxil | shader::Kind::Dxbc) {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                    record_done();
                    continue;
                };
                if only.is_some_and(|o| !sh.stage.eq_ignore_ascii_case(o)) {
                    record_done();
                    continue;
                }
                if startup.and_then(|p| p.only.as_ref()).is_some_and(|used| !used.contains(&crate::hle::d3d12x::real::shader_key(&blob))) {
                    record_done();
                    continue;
                }
                // a compute program as a title's pipeline translates it (`pso::console_to_dxil_in`):
                // into the translation cache too, so the title does not translate it again
                if sh.fssm.iter().any(|f| matches!(recomp::interface(&blob, &sh, f), Some(recomp::Mode::Compute(_)))) {
                    if let Some(coverage) = coverage {
                        match coverage.covers(i, &blob) {
                            Ok(true) => { record_done(); continue; }
                            Err(e) => {
                                failed.fetch_add(1, Ordering::Relaxed);
                                let mut logs = errors.lock().unwrap();
                                if logs.len() < 10 { logs.push(format!("{name}: {e}")); }
                                record_done();
                                continue;
                            }
                            Ok(false) => {}
                        }
                    }
                    match recomp::pso::console_to_dxil_in(&blob, recomp::pso::Stage::Compute, 0, split, windows) {
                        Ok(c) => {
                            if let Some(p) = pipelines {
                                if mandatory {
                                    if let Err(e) = p.prepare_compiled_compute(&c) {
                                        failed.fetch_add(1, Ordering::Relaxed);
                                        let mut logs = errors.lock().unwrap();
                                        if logs.len() < 10 { logs.push(format!("{name}: {e}")); }
                                        record_done();
                                        continue;
                                    }
                                } else {
                                    let used = used_earlier.contains(&pipeline_key(&blob));
                                    let prewarm_indirect = indirect.is_some_and(|i| used_indirect_split(&blob, i));
                                    pipeline_jobs.lock().unwrap().push(PipelineJob { used, prewarm_indirect, compiled: c });
                                }
                            }
                            compiled.fetch_add(1, Ordering::Relaxed)
                        }
                        Err(e) => {
                            let mut g = errors.lock().unwrap();
                            if g.len() < 10 {
                                g.push(format!("{name}: {}", e.lines().find(|l| l.contains("error")).unwrap_or(e.lines().next().unwrap_or(""))));
                            }
                            failed.fetch_add(1, Ordering::Relaxed)
                        }
                    };
                    record_done();
                    continue;
                }
                for (stage, sampler_base, split, windows) in standalone_graphics_variants(&blob, &sh, graphics_split, windows) {
                    match recomp::pso::console_to_dxil_in(&blob, stage, sampler_base, split, windows) {
                        Ok(_) => compiled.fetch_add(1, Ordering::Relaxed),
                        Err(e) => {
                            let mut g = errors.lock().unwrap();
                            if g.len() < 10 {
                                g.push(format!("{name}: {}", e.lines().find(|l| l.contains("error")).unwrap_or(e.lines().next().unwrap_or(""))));
                            }
                            failed.fetch_add(1, Ordering::Relaxed)
                        }
                    };
                }
                let mut stage_outputs = Vec::new();
                for f in &sh.fssm {
                    let Some(mode) = recomp::interface(&blob, &sh, f) else { continue };
                    let jobs: Vec<(&str, Option<String>)> = match &mode {
                        recomp::Mode::Vertex(_) | recomp::Mode::Pixel(_) => continue,
                        recomp::Mode::Compute(_) => vec![("cs_6_6", None)],
                        recomp::Mode::Mesh(ms) => vec![("cs_6_6", None), ("ms_6_6", Some(recomp::mesh_passthrough(ms)))],
                        _ => {
                            skipped.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    };
                    for p in &f.programs {
                        let code = &f.data[p.offset..p.offset + p.len];
                        // as `pso` translates: with the title's memory windows (`console_translation`,
                        // `translate_mesh`)
                        let split = if matches!(mode, recomp::Mode::Vertex(_) | recomp::Mode::Pixel(_)) { graphics_split } else { split };
                        let opt = recomp::Options { mode: mode.clone(), wave64: recomp::fssm_wave64(f, code), split_wave64: split, mem_windows: windows, rodata: f.rodata(p), ..Default::default() };
                        let t = match recomp::translate(code, &opt) {
                            Ok(t) if t.complete() => t,
                            Ok(_) => {
                                if startup.is_some() { skipped.fetch_add(1, Ordering::Relaxed); }
                                continue;
                            }
                            Err(e) => {
                                if startup.is_some() {
                                    failed.fetch_add(1, Ordering::Relaxed);
                                    let mut logs = errors.lock().unwrap();
                                    if logs.len() < 10 { logs.push(format!("{name}: {e}")); }
                                }
                                continue;
                            }
                        };
                        for (profile, src) in &jobs {
                            let src = src.as_deref().unwrap_or(&t.hlsl);
                            match recomp::dxc::compile_cached(src, "main", profile) {
                                Ok((bytes, hit)) => {
                                    if pack_proof.is_some() {
                                        stage_outputs.push((recomp::dxc::cache_key(src, "main", profile), bytes.len() as u64));
                                    }
                                    if hit { &cached } else { &compiled }.fetch_add(1, Ordering::Relaxed)
                                }
                                Err(e) => {
                                    let mut g = errors.lock().unwrap();
                                    if g.len() < 10 {
                                        g.push(format!("{name}: {}", e.lines().find(|l| l.contains("error")).unwrap_or("")));
                                    }
                                    failed.fetch_add(1, Ordering::Relaxed)
                                }
                            };
                        }
                    }
                }
                if let Some(proof) = &pack_proof { proof.record(i, stage_outputs); }
                record_done();
            });
        }
    });
    if let Some(selected) = selected {
        let matched = matched.into_inner().unwrap();
        let mut unmatched: Vec<_> = selected.difference(&matched).copied().collect();
        unmatched.sort_unstable();
        println!("selection: {} of {} keys matched shader-package records", matched.len(), selected.len());
        for key in unmatched {
            println!("  unmatched key {key:016x}");
        }
    }
    let failed = failed.into_inner();
    let skipped_reason = if startup.is_some() { "runtime pairing or unsupported translation" } else { "GS – paired at pipeline creation" };
    if let Some(ComputeCoverage::Published(_, selected, _)) = coverage {
        let count = selected.lock().unwrap().len();
        if count != 0 { crate::xlog_info!("Shader preparation: {count} byte-matched compute records will be built by the staged corpus"); }
    }
    println!(
        "compiled {}, already cached {}, errors {}, skipped ({skipped_reason}) {}, time {:.1} s",
        compiled.into_inner(),
        cached.into_inner(),
        failed,
        skipped.into_inner(),
        t0.elapsed().as_secs_f64()
    );
    if let Some(p) = pipelines {
        if startup.is_none() {
            make_pipelines(p, pipeline_jobs.into_inner().unwrap());
        }
        println!("pipelines made on the GPU {}, refused {}", p.made.load(Ordering::Relaxed), p.refused.load(Ordering::Relaxed));
        if let Some(book) = &p.warm_blobs {
            book.save();
            println!("previously verified native compute pipelines {}", p.verified.load(Ordering::Relaxed));
        }
        println!("pipelines loaded from the driver cache {}", p.reused.load(Ordering::Relaxed));
        let created = p.created.load(Ordering::Relaxed);
        let persistence = p.persistence.snapshot();
        println!(
            "fresh GPU pipeline creations {created}: driver blobs saved {}, unavailable {}, persistence disabled {}, cache writes failed {}",
            persistence.saved, persistence.blob_unavailable, persistence.disabled, persistence.write_failed,
        );
        debug_assert_eq!(persistence.total(), created);
        debug_assert_eq!(
            created + p.reused.load(Ordering::Relaxed),
            p.made.load(Ordering::Relaxed),
        );
        if persistence.write_failed != 0 {
            println!("pipeline driver-cache writes failed {}", persistence.write_failed);
        }
        let elsewhere = p.elsewhere.load(Ordering::Relaxed);
        if elsewhere != 0 {
            println!("pipelines left to another preparation process {elsewhere}");
        }
        let split_programs = p.split_programs.load(Ordering::Relaxed);
        if split_programs != 0 {
            println!("split compute programs: {} ({} pieces)", split_programs, p.split_pieces.load(Ordering::Relaxed));
        }
        let indirect_programs = p.indirect_programs.load(Ordering::Relaxed);
        if indirect_programs != 0 {
            println!("indirect split programs (direct and indirect pieces/whole): {indirect_programs}");
        }
        let capped = p.capped.load(Ordering::Relaxed);
        if capped != 0 {
            println!("pipelines over the {} MiB DXIL cap, left to the runtime: {} ({} MiB of DXIL; XCL_PIPELINE_MAX_KB)", p.max_dxil >> 20, capped, p.capped_bytes.load(Ordering::Relaxed) >> 20);
        }
    }
    let errors = errors.into_inner().unwrap();
    for e in &errors { println!("  {e}"); }
    if mandatory {
        if startup.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) { return Err("Shader preparation cancelled".into()); }
        // (no pipelines here when worker processes build them: `pipelines_in_workers`)
        let refused = pipelines.map_or(0, |p| p.refused.load(Ordering::Relaxed));
        let capped = pipelines.map_or(0, |p| p.capped.load(Ordering::Relaxed));
        if failed + refused + capped > 0 {
            let cause = errors.first().map(String::as_str).unwrap_or(
                if capped > 0 { "Raise XCL_PIPELINE_MAX_KB or set it to 0 to prepare larger pipelines." }
                else { "Check the shader pack and driver diagnostics." });
            return Err(format!("Shader preparation stopped: {failed} record failures, {refused} refused pipelines, {capped} over the size cap. {cause}"));
        }
    }
    if let Some(proof) = pack_proof { proof.save(); }
    Ok(())
}

/// Compile the kept HLSL samples with DXC on all cores: (ok, failures with the first error line).
fn compile_samples(samples: &[(String, String, &'static str)]) -> (usize, Vec<String>) {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let times: Mutex<Vec<(f64, usize, String)>> = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let parts: Vec<(usize, Vec<String>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let (mut ok, mut bad) = (0usize, Vec::new());
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some((name, hlsl, profile)) = samples.get(i) else { break };
                        let t0 = std::time::Instant::now();
                        match recomp::dxc::compile(hlsl, "main", profile, &[]) {
                            Ok(_) => ok += 1,
                            Err(e) => bad.push(format!("{name}: {}", e.lines().find(|l| l.contains("error")).unwrap_or(e.as_str()))),
                        }
                        times.lock().unwrap().push((t0.elapsed().as_secs_f64(), hlsl.len(), name.clone()));
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n % 100 == 0 {
                            eprintln!("DXC: {n}/{}", samples.len());
                        }
                    }
                    (ok, bad)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let ok = parts.iter().map(|p| p.0).sum();
    let mut t = times.into_inner().unwrap();
    t.sort_by(|a, b| a.0.total_cmp(&b.0));
    if !t.is_empty() {
        let q = |f: f64| t[((t.len() - 1) as f64 * f) as usize].0;
        let sum: f64 = t.iter().map(|x| x.0).sum();
        eprintln!(
            "DXC time per shader: median {:.2} s, p90 {:.2} s, max {:.2} s, sum {:.0} s (threads: in parallel)",
            q(0.5),
            q(0.9),
            q(1.0),
            sum
        );
        for (secs, len, name) in t.iter().rev().take(5) {
            eprintln!("  {secs:.1} s, HLSL {} KB: {name}", len / 1024);
        }
    }
    (ok, parts.into_iter().flat_map(|p| p.1).collect())
}

#[allow(clippy::too_many_arguments)]
fn print(path: &Path, what: &str, agg: Agg, top: usize, json: bool, t0: std::time::Instant, compile: usize) -> i32 {
    let rc = &agg.rc;
    let compiled = if compile > 0 && !rc.samples.is_empty() { Some(compile_samples(&rc.samples)) } else { None };
    let mut missing: Vec<(&String, &(usize, usize))> = rc.missing.iter().collect();
    missing.sort_by(|a, b| b.1 .1.cmp(&a.1 .1).then_with(|| a.0.cmp(b.0)));
    let ranked = agg.insts.sorted();
    let ks: Vec<usize> = [10, 25, 50, 75, 100, 150, 200, 300].into_iter().filter(|&k| k < ranked.len()).chain([ranked.len()]).collect();
    let order = by_shader_count(&agg, &ranked);
    let cov = coverage(&agg, &order, &ks);
    let n_xbox = agg.sets.len();
    let total_inst = agg.insts.instructions.max(1);

    if json {
        let mnemonics: Vec<serde_json::Value> = ranked
            .iter()
            .map(|(m, c)| serde_json::json!({ "mnemonic": m, "count": c, "shaders": agg.shader_count.get(m).copied().unwrap_or(0) }))
            .collect();
        let v = serde_json::json!({
            "path": path.display().to_string(),
            "source": what,
            "shaders": agg.total,
            "inflate_errors": agg.inflate_errors,
            "by_kind": agg.by_kind,
            "by_dxil_stage": agg.by_stage,
            "by_hw_stage": agg.by_hw,
            "fssm": agg.fssm,
            "programs": agg.programs,
            "embedded_data_blocks": agg.data_blocks,
            "fssm_without_code": agg.no_code,
            "parse_problems": agg.failed_count,
            "parse_problem_examples": agg.failed,
            "unusual": agg.odd_count,
            "unusual_examples": agg.odd,
            "instructions": agg.insts.instructions,
            "undecoded_dwords": agg.insts.undecoded,
            "distinct_mnemonics": ranked.len(),
            "implementation_order": order,
            "coverage_top_k": cov.iter().map(|(k, c)| serde_json::json!({ "k": k, "shaders": c })).collect::<Vec<_>>(),
            "mnemonics": mnemonics,
            "recomp": agg.recomp_on.then(|| serde_json::json!({
                "programs": rc.programs,
                "complete": rc.complete,
                "wave32_programs": rc.wave32,
                "translate_errors": rc.errors,
                "translate_error_examples": rc.error_examples,
                "hlsl_bytes": rc.hlsl_bytes,
                "missing": missing.iter().map(|(m, (n, p))| serde_json::json!({
                    "mnemonic": m, "instructions": n, "programs": p, "sole_blocker": rc.sole.get(*m).copied().unwrap_or(0)
                })).collect::<Vec<_>>(),
                "dxc_compiled": compiled.as_ref().map(|c| c.0),
                "dxc_failed": compiled.as_ref().map(|c| c.1.clone()),
            })),
            "seconds": t0.elapsed().as_secs_f64(),
        });
        println!("{}", serde_json::to_string_pretty(&v).unwrap());
        return 0;
    }

    println!("{} – {what}", path.display());
    println!("shaders: {}{}", agg.total, if agg.inflate_errors > 0 { format!(" (+{} records could not be unpacked)", agg.inflate_errors) } else { String::new() });
    println!("formats: {}", fmt_map(&agg.by_kind));
    if n_xbox == 0 {
        println!("no console containers (xbox-native)");
        return 0;
    }
    println!("stages (DXIL): {}", fmt_map(&agg.by_stage));
    println!("FSSM variants by hardware stage: {}", fmt_map(&agg.by_hw));
    println!("FSSM: {}, programs: {}, data blocks in code: {}, FSSM without code: {}", agg.fssm, agg.programs, agg.data_blocks, agg.no_code);
    println!(
        "RDNA2 instructions: {}, undecoded dwords: {} ({:.4}%), distinct mnemonics: {}",
        agg.insts.instructions,
        agg.insts.undecoded,
        agg.insts.undecoded as f64 * 100.0 / (agg.insts.instructions + agg.insts.undecoded).max(1) as f64,
        ranked.len()
    );
    if agg.failed_count > 0 {
        println!("parse problems: {} containers, e.g.:", agg.failed_count);
        for f in &agg.failed {
            println!("  {f}");
        }
    }
    if agg.odd_count > 0 {
        println!("unusual: {} containers (several programs in an FSSM or undecoded dwords), e.g.:", agg.odd_count);
        for o in agg.odd.iter().take(10) {
            println!("  {o}");
        }
    }
    println!("coverage: shaders made up only of the first K mnemonics (order: number of shaders it appears in):");
    for (k, c) in &cov {
        let last = order.get(k.saturating_sub(1)).map(String::as_str).unwrap_or("");
        println!("  K = {k:>4}: {c:>6} / {n_xbox} ({:>5.1}%)   K-th: {last}", *c as f64 * 100.0 / n_xbox as f64);
    }
    println!("top {} mnemonics (count, % of instructions, cumulative %, number of shaders):", top.min(ranked.len()));
    let mut cum = 0u64;
    for (i, (m, c)) in ranked.iter().take(top).enumerate() {
        cum += c;
        let sc = agg.shader_count.get(m).copied().unwrap_or(0);
        println!(
            "  {:>4}. {m:<34} {c:>10} {:>6.2}% {:>6.2}%  {sc:>6} ({:.0}%)",
            i + 1,
            *c as f64 * 100.0 / total_inst as f64,
            cum as f64 * 100.0 / total_inst as f64,
            sc as f64 * 100.0 / n_xbox as f64
        );
    }
    if agg.iface_on {
        let i = &agg.iface;
        println!("interface (hypotheses):");
        println!(
            "  PS: {} programs with v_interp; attributes ⊆ ISG1 registers: {}, = ISG1 registers: {}, attribute < number of non-SV elements (rank): {}",
            i.ps, i.ps_in_regs, i.ps_eq_regs, i.ps_rank
        );
        for e in &i.ps_examples {
            println!("    {e}");
        }
        println!(
            "  PS: SPI_PS_INPUT_ENA from FSSM+0x248 explains the v_interp I/J: {} / {}, all VGPRs read before write fit the layout: {} / {}",
            i.ps_ena_ij, i.ps, i.ps_ena_reads, i.ps
        );
        for e in &i.ps_ena_examples {
            println!("    {e}");
        }
        println!(
            "  VS: {} programs; exp param ⊆ OSG1 registers: {}, = OSG1 registers: {}, param < number of non-SV elements (rank): {}",
            i.vs, i.vs_param_in_regs, i.vs_param_eq, i.vs_rank
        );
        for e in &i.vs_param_examples {
            println!("    {e}");
        }
        println!(
            "  VS inputs (VGPRs ≥ v4 read before write): {} programs; matching v4+4·register: {}, with consecutive attributes packed: {}",
            i.vs_inputs, i.vs_in_vec4, i.vs_in_packed
        );
        for e in &i.vs_input_examples {
            println!("    {e}");
        }
        println!("  inputs by stage (API hardware stage: register read before write → number of programs; exports; s_sendmsg):");
        let sreg = |r: u64| match r {
            106 => "vcc_lo".to_string(),
            107 => "vcc_hi".to_string(),
            124 => "m0".to_string(),
            126 => "exec_lo".to_string(),
            127 => "exec_hi".to_string(),
            r => format!("s{r}"),
        };
        let tgt = |t: u64| match t {
            0..=7 => format!("mrt{t}"),
            8 => "mrtz".into(),
            9 => "null".into(),
            12..=15 => format!("pos{}", t - 12),
            20 => "prim".into(),
            32..=63 => format!("param{}", t - 32),
            t => format!("tgt{t}"),
        };
        for (k, e) in &i.inputs {
            let pct = |n: usize| n * 100 / e.programs.max(1);
            let list = |m: &BTreeMap<u64, usize>, f: &dyn Fn(u64) -> String| {
                m.iter().filter(|(_, &n)| pct(n) >= 1).map(|(r, &n)| format!("{} {}%", f(*r), pct(n))).collect::<Vec<_>>().join(", ")
            };
            println!("    {k}: {} programs", e.programs);
            println!("      VGPR: {}", list(&e.vgpr, &|r| format!("v{r}")));
            println!("      SGPR: {}", list(&e.sgpr, &sreg));
            println!("      exp: {}", list(&e.exports, &tgt));
            println!("      sendmsg: {}", list(&e.sendmsg, &|m| m.to_string()));
            let mut ps: Vec<_> = e.psv.iter().collect();
            ps.sort_by(|a, b| b.1.cmp(a.1));
            println!("      PSV0: {}", ps.iter().take(6).map(|(k, n)| format!("{k}: {n}")).collect::<Vec<_>>().join("; "));
            let mut fu: Vec<_> = e.first_use.iter().collect();
            fu.sort_by(|a, b| a.0.split(':').next().cmp(&b.0.split(':').next()).then(b.1.cmp(a.1)));
            let mut shown: BTreeMap<String, usize> = BTreeMap::new();
            for (k, n) in fu {
                let reg = k.split(':').next().unwrap_or("").to_string();
                let c = shown.entry(reg).or_default();
                if *c < 4 && *n * 100 >= e.programs {
                    println!("      first use {k}  ({n})");
                }
                *c += 1;
            }
        }
        println!("  VS input layout (smallest base B: inputs ⊆ B + 4·register + component): {}", fmt_map(&i.vs_base));
    }
    if agg.recomp_on {
        println!(
            "recompiler (kernel mode): programs {}, fully translated {} ({:.1}%), translation errors {}, wave32: {}, HLSL: {:.1} MB",
            rc.programs,
            rc.complete,
            rc.complete as f64 * 100.0 / rc.programs.max(1) as f64,
            rc.errors,
            rc.wave32,
            rc.hlsl_bytes as f64 / 1048576.0
        );
        for e in &rc.error_examples {
            println!("  error: {e}");
        }
        for (m, (n, c)) in &rc.by_mode {
            println!("  mode {m}: {n} programs, complete {c} ({:.1}%)", *c as f64 * 100.0 / (*n).max(1) as f64);
        }
        println!("missing instructions (number of programs; programs it blocks as the only missing one; occurrences):");
        for (m, (n, p)) in missing.iter().take(top) {
            println!("  {m:<34} {p:>6} {:>6} {n:>9}", rc.sole.get(*m).copied().unwrap_or(0));
        }
        let mut why: Vec<(&String, &usize)> = rc.why.iter().collect();
        why.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        println!("reasons (instruction: reason → number of programs):");
        for (k, v) in why.iter().take(top) {
            println!("  {k:<60} {v:>6}");
        }
        println!(
            "VS/PS and the wave size (the driver picks it): wave32 {} (handled – the halves of a 64-wide wave run separately), wave64 {}; hazards in 32-thread waves: {:?}; reads of lanes ≥ 32 resolved (32-lane GPUs): {:?}",
            rc.graphics_wave32, rc.graphics_wave64, rc.wave_hazards, rc.wave_resolved
        );
        let c = &rc.cs_forms;
        let mut why: Vec<_> = rc.cs_lane_why.iter().collect();
        why.sort_by(|a, b| b.1.cmp(a.1));
        println!(
            "compute programs, direct DXIL: {} tried; per-lane form {} ok, per-instruction form (#1426) {} ok; where both are ok ({}): per-lane {} KB, per-instruction {} KB of DXIL; per-lane refused for: {:?}",
            c[0], c[1], c[2], c[3], c[4] / 1024, c[5] / 1024, &why[..why.len().min(15)]
        );
        for (st, f) in &rc.forms {
            println!(
                "{st} programs, every refusal: {} tried ({} wave64); per-lane ok {}, per-instruction ok {}, both {}; per-lane refused for instructions only {} (per-instruction ok on {} of them); lane-mask analysis refused {}",
                f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]
            );
        }
        for (title, m) in [
            ("per-lane reasons (stage | reason → programs; [S] structural)", &rc.lane_every),
            ("the only structural reason of a program (stage | reason → programs)", &rc.lane_sole),
            ("per-instruction form (#1426) refused (stage | first line → programs)", &rc.inst_why),
        ] {
            let mut v: Vec<_> = m.iter().collect();
            v.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            println!("{title}:");
            for (k, n) in v.iter().take(top.max(40)) {
                let tag = if survey_structural(k.split(" | ").nth(1).unwrap_or("")) { "[S] " } else { "" };
                println!("  {n:>6}  {tag}{k}");
            }
        }
        let l = &rc.lanes;
        println!(
            "lane masks (direct DXIL, no EXEC emulation): analysed {} of {} programs, {} with no guard left; vector instructions {}, still guarded {} ({:.1}%); branches per lane {}, uniform {}; not analysed: {:?}",
            l[0],
            rc.programs,
            l[1],
            l[2],
            l[3],
            l[3] as f64 * 100.0 / l[2].max(1) as f64,
            l[4],
            l[5],
            rc.lanes_failed
        );
        if let Some((x, hlsl, profile)) = &rc.example {
            println!("example:
{x}");
            match recomp::dxc::compile(hlsl, "main", profile, &[]) {
                Ok(b) => println!("DXC ({profile}): OK, {} B DXIL", b.len()),
                Err(e) => println!("DXC ({profile}): {}", e.split("--- HLSL ---").next().unwrap_or("").trim()),
            }
            if let Ok(path) = std::env::var("XCL_EXAMPLE_HLSL") {
                let _ = std::fs::write(&path, hlsl);
                println!("HLSL saved: {path}");
            }
        }
        if let Some((ok, bad)) = &compiled {
            println!("DXC (vs/ps/cs_6_6) on a sample of complete translations: {ok} OK, {} errors", bad.len());
            for b in bad.iter().take(10) {
                println!("  {b}");
            }
        }
    }
    println!("time: {:.1} s", t0.elapsed().as_secs_f64());
    0
}

#[cfg(test)]
mod tests {
    /// The workers of a title's shader preparation: four logical processors are left free,
    /// 3 GB of commit for each, and one where there is no room.
    #[test]
    fn preparation_workers_leave_two_processors_and_follow_the_commit_left() {
        assert_eq!(workers_for(32, 100_000), 28);
        assert_eq!(workers_for(8, 100_000), 4);
        assert_eq!(workers_for(32, 10_000), 3);
        assert_eq!(workers_for(2, 100_000), 1);
        assert_eq!(workers_for(64, 1_000_000), 32);
        assert_eq!(workers_for(32, 0), 1);
    }

    /// The commit figures come in bytes: 6 GiB left on 16 logical processors is room for two
    /// workers, not twelve; a commit that is not known (0, 0) gives one.
    #[test]
    fn preparation_workers_take_the_commit_left_in_bytes() {
        const GIB: u64 = 1 << 30;
        assert_eq!(workers_for_commit(16, 10 * GIB, 16 * GIB), 2);
        assert_eq!(workers_for_commit(16, 0, 64 * GIB), 12);
        assert_eq!(workers_for_commit(16, 0, 0), 1);
        assert_eq!(workers_for_commit(16, 20 * GIB, 16 * GIB), 1);
    }

    /// A builder's progress line gives its record count; its other lines give none.
    #[test]
    fn a_builders_progress_line_gives_its_count() {
        assert_eq!(progress_count("  400/26503 (12 s)"), Some(400));
        assert_eq!(progress_count("26400/26503 (1339 s)"), Some(26400));
        assert_eq!(progress_count("xcl: [SHADER] direct DXIL fallback stage=CS reason=a/b (c)"), None);
        assert_eq!(progress_count("  pipelines: 50 made, 0 refused"), None);
        assert_eq!(progress_count("3/4"), None);
    }

    /// A graphics worker's summary gives its counts; its progress lines and others give none.
    #[test]
    fn a_graphics_workers_summary_gives_its_counts() {
        assert_eq!(graphics_summary("graphics pipelines: 512 made, 3 failed; driver blobs: 40 from disk, 470 kept"), Some([512, 3, 40, 470]));
        assert_eq!(graphics_summary("  graphics pipelines: 0 made, 0 failed; driver blobs: 0 from disk, 0 kept\r"), Some([0; 4]));
        assert_eq!(graphics_summary("  12/512 (3 s)"), None);
        assert_eq!(graphics_summary("graphics pipelines: 5 made"), None);
        assert_eq!(graphics_summary("graphics pipelines: 1 made, 2 failed; driver blobs: 3 from disk, 4 kept, 5 more"), None);
        assert_eq!(progress_count("graphics pipelines: 512 made, 3 failed; driver blobs: 40 from disk, 470 kept"), None);
    }

    /// A graphics record is built when every shader it names is found and they make its set's
    /// key (the key `used-pipelines.txt` has); one with a shader not in the title's files, or
    /// other shaders under its keys, is left out.
    #[test]
    fn a_graphics_record_is_built_only_with_its_own_shaders() {
        use crate::hle::d3d12x::gfx_records::{decode, encode, encode_mesh};
        use crate::hle::d3d12x::real::{shader_key, shaders_key};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
        };
        let (vs, ps, ms) = (b"vertex".to_vec(), b"pixel".to_vec(), b"mesh".to_vec());
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        let record = |key, shaders| {
            GraphicsHistoryRecord::from_entry(decode(&unsafe { encode(key, shaders, &desc) }).unwrap()).unwrap()
        };
        let both = [shader_key(&vs), shader_key(&ps), 0, 0, 0];
        let mesh = |key, shaders| {
            let layout = [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS.0 as u32, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0 as u32];
            let payload = encode_mesh(key, shaders, &layout, &desc, false).unwrap();
            GraphicsHistoryRecord::from_entry(decode(&payload).unwrap()).unwrap()
        };
        let records = [
            record(shaders_key(&[vs.clone(), ps.clone()]), both),
            record(shaders_key(&[vs.clone()]), [shader_key(&vs), 0, 0, 0, 0]),
            record(shaders_key(&[vs.clone(), b"other".to_vec()]), [shader_key(&vs), shader_key(b"other"), 0, 0, 0]),
            record(shaders_key(&[ps.clone(), vs.clone()]), both),
            mesh(shaders_key(&[ms.clone(), ps.clone()]), [shader_key(&ms), shader_key(&ps)]),
            mesh(shaders_key(&[ps.clone(), ms.clone()]), [shader_key(&ms), shader_key(&ps)]),
        ];
        let shaders = HashMap::from([(shader_key(&vs), vs), (shader_key(&ps), ps), (shader_key(&ms), ms)]);
        assert_eq!(buildable(records.iter(), &shaders), (vec![0, 1, 4], 3));
    }

    #[test]
    fn exact_graphics_history_selection_preserves_manifest_order() {
        use windows::Win32::Graphics::Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC;
        let payload = |mask| unsafe {
            gfx_records::encode(
                7, [7, 0, 0, 0, 0],
                &D3D12_GRAPHICS_PIPELINE_STATE_DESC { SampleMask: mask, ..Default::default() },
            )
        };
        let first = payload(1);
        let second = payload(2);
        let ids = [gfx_records::payload_key(&second), gfx_records::payload_key(&first)];
        let selected = exact_graphics_records(&ids, vec![first, second, vec![0xff]]).unwrap();
        assert_eq!(selected.iter().map(|record| match record {
            GraphicsHistoryRecord::Legacy(record) => record.desc([None; 5]).SampleMask,
            GraphicsHistoryRecord::Mesh(_) => 0,
        }).collect::<Vec<_>>(), [2, 1]);
        let Err(error) = exact_graphics_records(&[3], Vec::new()) else { panic!("missing recipe accepted") };
        assert!(error.contains("0000000000000003"));
    }

    #[test]
    fn staged_history_defers_exact_recipes_and_supplies_sources_for_other_variants() {
        use crate::hle::d3d12x::real::shader_key;
        use windows::Win32::Graphics::Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC;
        let path = std::env::temp_dir().join(format!("xcl-staged-history-{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let code = b"vertex source";
        let key = shader_key(code);
        let pipeline = |mask| {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC { SampleMask: mask, ..Default::default() };
            let payload = unsafe { gfx_records::encode(key, [key, 0, 0, 0, 0], &desc) };
            let record = GraphicsHistoryRecord::from_entry(gfx_records::decode(&payload).unwrap()).unwrap();
            (payload, record)
        };
        let mut records = vec![pipeline(1), pipeline(2), pipeline(1), pipeline(3)];
        {
            let mut staged = gfx_records::Writer::open(&path, u64::MAX).unwrap();
            staged.add(&records[0].0).unwrap();
            staged.add(&gfx_records::shader_entry(key, code)).unwrap();
        }
        let mut sources = HashMap::new();
        assert_eq!(merge_staged_history(&path, &mut records, 2, &mut sources).unwrap(), 1);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, pipeline(2).0);
        assert_eq!(records[1].0, pipeline(3).0);
        assert_eq!(buildable(records.iter().map(|(_, r)| r), &sources), (vec![0, 1], 0));
        {
            let mut staged = gfx_records::Writer::open(&path, u64::MAX).unwrap();
            staged.add(&gfx_records::shader_entry(key, b"wrong")).unwrap();
        }
        assert!(merge_staged_history(&path, &mut records, 1, &mut sources).unwrap_err().contains("identity check"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_graphics_job_with_a_missing_or_incorrectly_named_shader_fails() {
        use crate::hle::d3d12x::gfx_records::{encode_mesh, shader_entry, Writer};
        use crate::hle::d3d12x::real::shader_key;
        let path = std::env::temp_dir().join(format!("xcl-graphics-missing-{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let bytes = b"mesh";
        let key = shader_key(bytes);
        {
            let mut job = Writer::open(&path, u64::MAX).unwrap();
            job.add(&encode_mesh(key, [key, 0], &[25], &Default::default(), false).unwrap()).unwrap();
        }
        assert_eq!(graphics_worker(&path, 1, Some(1)), 2);
        {
            let mut job = Writer::open(&path, u64::MAX).unwrap();
            job.add(&shader_entry(key, b"different mesh")).unwrap();
        }
        assert_eq!(graphics_worker(&path, 1, Some(1)), 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn explicit_pipeline_worker_counts_can_use_all_thirty_two_slots() {
        for count in [1, 2, 8, 16, 32] {
            assert_eq!(corpus_threads(Some(count), 1), count);
        }
        assert_eq!(corpus_threads(Some(0), 1), 1);
        assert_eq!(corpus_threads(Some(33), 1), 32);
    }

    /// Offline commands and the runtime must use the same cache identity. Environment switches
    /// are process-global (`OnceLock`), so exercise each setting in a fresh test process.
    #[test]
    fn precompile_windows_are_the_runtimes() {
        use crate::gpu::recomp::{pso, HEAP_FLAT, HEAP_WINDOWS};
        const RUN: &str = "XCL_TEST_PRECOMPILE_WINDOWS";
        if let Ok(want) = std::env::var(RUN) {
            assert_eq!(super::runtime_windows(4), pso::windows_for(4));
            assert_eq!(super::runtime_windows(4), want.parse::<u32>().unwrap());
            return;
        }
        for (flat, want) in [("1", 4 | HEAP_WINDOWS | HEAP_FLAT), ("0", 4 | HEAP_WINDOWS)] {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "shaders::tests::precompile_windows_are_the_runtimes", "--nocapture"])
                .env(RUN, want.to_string())
                .env("XCL_HEAP_WINDOWS", "1")
                .env("XCL_HEAP_FLAT", flat)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(out.status.success() && stdout.contains("1 passed"), "XCL_HEAP_FLAT={flat}: {stdout}\n{}", String::from_utf8_lossy(&out.stderr));
        }
    }

    /// `--used-only` keeps a compute program exactly when the runtime noted its pipeline: the
    /// key of a pipeline of that one shader, as `real::shaders_key` makes it.
    #[test]
    fn used_only_keeps_the_compute_programs_an_earlier_run_used() {
        let (a, b) = (vec![1u8, 2, 3, 4], vec![5u8, 6, 7, 8]);
        let used: std::collections::HashSet<u64> = [crate::hle::d3d12x::real::shaders_key(&[a.clone()])].into_iter().collect();
        assert!(super::used_compute(&a, &used));
        assert!(!super::used_compute(&b, &used));
        // the keys as the runtime writes them, one hexadecimal key a line
        let text = format!("{:016x}
", crate::hle::d3d12x::real::shaders_key(&[b.clone()]));
        let keys: std::collections::HashSet<u64> = text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).collect();
        assert!(super::used_compute(&b, &keys));
        let indirect: std::collections::HashSet<u64> = [crate::gpu::shader::fnv64(&a)].into();
        assert!(super::used_indirect_split(&a, &indirect));
        assert!(!super::used_indirect_split(&b, &indirect));
    }

    #[test]
    fn pipeline_worker_command_forwards_console_profile() {
        let dev_env = |series_s| {
            let cmd = super::pipeline_worker_command(Path::new("xcl.exe"), Path::new("shaders.rpf"), 4, false, "0/2", series_s);
            cmd.get_envs()
                .find_map(|(k, v)| (k == std::ffi::OsStr::new("XCL_DEVICE_TYPE")).then(|| v.map(|s| s.to_string_lossy().into_owned())))
                .flatten()
        };
        assert_eq!(dev_env(true).as_deref(), Some("lockhart"));
        assert_eq!(dev_env(false).as_deref(), Some("anaconda"));
    }

    use super::*;
    use crate::gpu::xbox::tests::{fssm, sample, CODE};

    pub(super) fn compute_sample() -> Vec<u8> {
        let mut compute = fssm(&CODE, &[(6, 3)]);
        for offset in [0x13C, 0x140, 0x144] {
            compute[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
        }
        sample(5, &[compute])
    }

    #[test]
    fn legacy_console_gs_needs_both_native_passes() {
        let shaders = HashMap::from([
            (1, sample(1, &[fssm(&CODE, &[(2, 0)])])),
            (2, sample(2, &[fssm(&CODE, &[(3, 2)])])),
            (3, b"PC shader".to_vec()),
        ]);
        assert!(legacy_two_pass(&[1, 0, 0, 0, 2], &shaders));
        assert!(!legacy_two_pass(&[1, 0, 3, 0, 2], &shaders));
        assert!(!legacy_two_pass(&[1, 0, 0, 0, 3], &shaders));
        assert!(!legacy_two_pass(&[1, 0, 0, 0, 0], &shaders));
    }

    #[test]
    fn staged_compute_covers_only_its_exact_sources_in_loose_and_pack_passes() {
        let name = format!("xcl-compute-membership-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
        let root = std::env::temp_dir().join(name);
        let base = root.join(".xcl/pso-corpus");
        let dir = base.join("compute");
        std::fs::create_dir_all(&dir).unwrap();
        let blob = compute_sample();
        let file = dir.join(format!("{:016x}.dxbc", shader::fnv64(&blob)));
        std::fs::write(&file, &blob).unwrap();
        std::fs::write(base.join("windows"), "4").unwrap();
        std::fs::write(base.join("complete"), "0 1").unwrap();
        let corpus = pso_corpus::discover(&root).unwrap().unwrap();
        let parent = ComputeCoverage::Published(&corpus, Mutex::default(), AtomicUsize::new(0));
        assert!(parent.covers(3, &blob).unwrap());
        assert!(parent.all_staged(), "no compute slice workers needed");
        assert!(!parent.covers(6, b"different source").unwrap());
        assert!(!parent.all_staged(), "uncovered compute records still need the slice");
        let (map, directory) = parent.slice_file().unwrap().unwrap();
        let child = ComputeCoverage::read_slice(&map.0, directory.to_owned()).unwrap();
        assert!(child.covers(3, &blob).unwrap());
        assert!(!child.covers(7, &blob).unwrap(), "index absent from the checked pack");
        assert!(!child.covers(3, b"changed input").unwrap(), "parent and child pack differ");
        std::fs::write(&file, b"changed after discovery").unwrap();
        assert!(child.covers(3, &blob).is_err(), "corrupt staged data cannot satisfy a recipe");
        std::fs::remove_file(&file).unwrap();
        assert!(parent.covers(3, &blob).is_err(), "missing staged data cannot satisfy a recipe");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn launch_owner_ignores_inherited_slices_and_workers_drop_foreign_coverage() {
        assert_eq!(precompile_slice(true, Some("1/2")), None);
        assert_eq!(precompile_slice(false, Some("1/2")), Some((1, 2)));
        let command = worker_command(Path::new("xcl.exe"));
        for name in [COVERAGE_MAP, COVERAGE_DIR] {
            assert!(command.get_envs().any(|(key, value)| key == name && value.is_none()));
        }
    }

    #[test]
    fn graphics_precompile_variants_match_runtime_stage_and_key_inputs() {
        let vs_blob = sample(1, &[fssm(&CODE, &[(4, 0)]), fssm(&CODE, &[(4, 0)])]);
        let ps_blob = sample(0, &[fssm(&CODE, &[(5, 1)])]);
        let vs = xbox::parse(&vs_blob).unwrap();
        let ps = xbox::parse(&ps_blob).unwrap();
        assert_eq!(
            standalone_graphics_variants(&vs_blob, &vs, true, 4 | recomp::HEAP_WINDOWS),
            vec![(recomp::pso::Stage::Vertex, 0, true, 4 | recomp::HEAP_WINDOWS)],
            "duplicate FSSMs still warm the one runtime-selected VS key",
        );
        assert_eq!(
            standalone_graphics_variants(&ps_blob, &ps, false, 1),
            vec![(recomp::pso::Stage::Pixel, 0, false, 1)],
            "the PS key keeps the exact split/windows and the only knowable sampler base",
        );
        let es_blob = sample(1, &[fssm(&CODE, &[(3, 0), (4, 0)])]);
        let es = xbox::parse(&es_blob).unwrap();
        assert!(standalone_graphics_variants(&es_blob, &es, false, 1).is_empty(), "runtime only classifies the first hardware stage");
    }

    #[test]
    fn record_hash_is_strict_hex() {
        assert_eq!(parse_record_hash("0x0123aB").unwrap(), 0x123AB);
        for bad in ["", "0x", "xyz", "1234567890abcdef0"] {
            assert!(parse_record_hash(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn hlsl_export_is_ordered_and_exact_hash_selects_one_record() {
        struct Temp(std::path::PathBuf);
        impl Drop for Temp {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let root = Temp(std::env::temp_dir().join(format!("xcl-export-hlsl-{}", std::process::id())));
        let _ = std::fs::remove_dir_all(&root.0);
        std::fs::create_dir_all(&root.0).unwrap();
        let first = compute_sample();
        let mut second = first.clone();
        *second.last_mut().unwrap() ^= 1; // outside the shader data; a distinct console-blob hash
        let first_path = root.0.join("first.bin");
        let second_path = root.0.join("second.bin");
        std::fs::write(&first_path, &first).unwrap();
        std::fs::write(&second_path, &second).unwrap();
        let input = Input::Files(vec![second_path, first_path]);

        let ordered = root.0.join("ordered");
        assert_eq!(export_hlsl(&input, &ordered, 0, 1, 4, Some(false), None), 0);
        let second_hash = shader::fnv64(&second);
        assert!(ordered.join(format!("{second_hash:016x}.hlsl")).is_file());
        assert_eq!(std::fs::read_dir(&ordered).unwrap().count(), 1);

        let exact = root.0.join("exact");
        let first_hash = shader::fnv64(&first);
        assert_eq!(export_hlsl(&input, &exact, usize::MAX, 9, 4, Some(false), Some(first_hash)), 0);
        assert!(exact.join(format!("{first_hash:016x}.hlsl")).is_file());
        assert_eq!(std::fs::read_dir(&exact).unwrap().count(), 1);

        let short = root.0.join("short");
        assert_eq!(export_hlsl(&input, &short, 0, 3, 4, Some(false), None), 2);
        let missing = root.0.join("missing");
        assert_eq!(export_hlsl(&input, &missing, 0, 1, 4, Some(false), Some(u64::MAX)), 2);

        // a non-compute record is skipped silently, unless it is the one asked for
        let other_path = root.0.join("other.bin");
        let other = b"not a console shader".to_vec();
        std::fs::write(&other_path, &other).unwrap();
        let mixed = Input::Files(vec![other_path.clone(), root.0.join("first.bin")]);
        assert_eq!(export_hlsl(&mixed, &root.0.join("mixed"), 0, 1, 4, Some(false), None), 0);
        assert_eq!(export_hlsl(&mixed, &root.0.join("other"), 0, 1, 4, Some(false), Some(shader::fnv64(&other))), 2);

        // through the command line: one shader file is an input of its own
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let single = root.0.join("single");
        let hex = format!("{first_hash:016x}");
        assert_eq!(run(&root.0.join("first.bin"), &args(&["--export-hlsl", single.to_str().unwrap(), "--record-hash", &hex, "--split", "0"])), 0);
        assert!(single.join(format!("{hex}.hlsl")).is_file());
    }

    #[test]
    fn a_flag_without_its_value_is_an_error() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let dir = std::env::temp_dir().join(format!("xcl-flag-value-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        assert_eq!(run(&dir, &args(&["--export-hlsl", "--record-hash", "DEADBEEF"])), 2);
        assert!(!std::path::Path::new("--record-hash").exists(), "the next flag is not taken as the value");
        assert_eq!(run(&dir, &args(&["--measure-pso"])), 2);
        assert_eq!(run(&dir, &args(&["--export-hlsl", "out", "--record-hash"])), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pso_measurement_compiles_an_exact_synthetic_record() {
        if let Err(e) = recomp::runner::wave64_supported() {
            recomp::runner::skip_test(&e);
            return;
        }
        struct Temp(std::path::PathBuf);
        impl Drop for Temp {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let path = Temp(std::env::temp_dir().join(format!("xcl-measure-pso-{}.bin", std::process::id())));
        let blob = compute_sample();
        std::fs::write(&path.0, &blob).unwrap();
        let hash = shader::fnv64(&blob);
        match measure_pso_result(&Input::Files(vec![path.0.clone()]), hash, 1, 4, None) {
            Ok(()) => {}
            // no DXC, a WARP-only or cacheless device, cache control without Developer Mode
            Err(MeasureError::Unavailable(e)) => recomp::runner::skip_test(&e),
            Err(MeasureError::Failed(e)) => panic!("{e}"),
        }
    }

    #[test]
    fn aggregates_console_shaders() {
        let mut a = Agg::default();
        a.add("PS_a", &sample(0, &[fssm(&CODE, &[(5, 1)])]));
        a.add("VS_b", &sample(1, &[fssm(&CODE, &[(2, 0)]), fssm(&CODE, &[(4, 0)])]));
        a.add("junk", b"not a shader");
        assert_eq!(a.total, 3);
        assert_eq!(a.by_kind.get("xbox-native"), Some(&2));
        assert_eq!(a.by_hw.get("ES"), Some(&1));
        assert_eq!(a.fssm, 3);
        assert_eq!(a.programs, 3);
        assert_eq!(a.insts.instructions, 9);
        assert_eq!(a.failed_count, 0, "{:?}", a.failed);
        assert_eq!(a.shader_count.get("s_endpgm"), Some(&2));
        let ranked = a.insts.sorted();
        let order = by_shader_count(&a, &ranked);
        assert_eq!(order.len(), 3);
        let cov = coverage(&a, &order, &[1, 3]);
        assert_eq!(cov, vec![(1, 0), (3, 2)]);
    }

    /// The recompiler statistics take a program's wave size as the pipelines do: from the FSSM
    /// header's wave32 flag where the code has no EXEC write to tell.
    #[test]
    fn recompiler_statistics_count_wave32_by_the_header_flag() {
        let wave32 = |flags: u32| {
            let mut f = fssm(&CODE, &[(6, 5)]);
            f[0xD4..0xD8].copy_from_slice(&flags.to_le_bytes());
            let mut a = Agg { recomp_on: true, ..Default::default() };
            a.add("CS_a", &sample(5, &[f]));
            (a.rc.programs, a.rc.wave32)
        };
        assert_eq!(wave32(0x20), (1, 1));
        assert_eq!(wave32(0), (1, 0));
    }

    /// Real data check (skipped unless XCL_SAMPLE_RPF points at an RPF9 shader pack).
    #[test]
    fn retail_pack_sample() {
        let Ok(p) = std::env::var("XCL_SAMPLE_RPF") else {
            println!("XCL_SAMPLE_RPF not set – skipping");
            return;
        };
        let data = std::fs::read(&p).unwrap();
        let mut records = pack::scan(&data, 0x40);
        records.truncate(300);
        let agg = analyze(&Input::Pack { data, records }, false, 0, false);
        assert_eq!(agg.inflate_errors, 0);
        assert_eq!(agg.by_kind.get("xbox-native"), Some(&agg.total), "{:?}", agg.by_kind);
        assert_eq!(agg.failed_count, 0, "{:?}", agg.failed);
        assert_eq!(agg.insts.undecoded, 0);
    }
}

/// `XCL_DUMP_HLSL=<folder>` (`--recomp`): every `XCL_DUMP_EVERY`-th complete translation (by a
/// hash of its name, default 100) is written there – samples for timing DXC on its own.
fn dump_hlsl(name: &str, label: &str, profile: &str, hlsl: &str) {
    let Some(dir) = std::env::var_os("XCL_DUMP_HLSL") else { return };
    let every: u64 = std::env::var("XCL_DUMP_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(100).max(1);
    let h = name_hash(name);
    if h % every != 0 {
        return;
    }
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{h:016x}_{label}_{profile}.hlsl")), hlsl);
}

/// FNV-1a of a program's name (`XCL_DUMP_HLSL` file names, `XCL_RECOMP_ONLY`).
fn name_hash(name: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// `--direct [--sample N] [--seed S] [--validate] [--compare] [--split 0|1] [--windows W]`:
/// every program of a seeded sample of N records (default 200 programs' worth; `--sample 0` =
/// all) through the direct DXIL translator; per-stage outcomes, a ranked histogram of what it
/// cannot translate, translation times and (`--compare`) DXIL size vs HLSL + DXC.
fn direct_sweep(input: &Input, opts: &ShadersOptions, t0: std::time::Instant) -> i32 {
    use crate::gpu::dxil::coverage::{self, Report, Sample};
    let top = opts.top;
    let json = opts.json;
    let want = if opts.sample == 0 { usize::MAX } else { opts.sample };
    let seed = opts.seed;
    let validate = opts.validate;
    let compare = opts.compare;
    let windows = opts.windows.unwrap_or(4);
    let split = opts.split.unwrap_or(true);
    // records hold one or more programs: sample records, stop at `want` programs
    let order = coverage::sample_indices(input.len(), want, seed);
    let next = AtomicUsize::new(0);
    let taken = AtomicUsize::new(0);
    let threads = opts
        .threads
        .map(|v| v.max(1))
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8));
    let trace = opts.trace;
    // big programs translate slowly (SSA over every guarded access): skip above the cap,
    // give up on one after the timeout (its thread is left to finish; the process ends soon)
    let max_bytes = opts.max_bytes;
    let timeout = std::time::Duration::from_secs(opts.timeout);
    let skipped = AtomicUsize::new(0);
    // `--lanes`: compute programs through the per-lane form too (`recomp::lower_dxil`); where it
    // translates, its DXIL size against the per-instruction form's, and the refusals ranked
    let lanes = opts.lanes;
    let lanes_tried = AtomicUsize::new(0);
    let lanes_ok = AtomicUsize::new(0);
    let lanes_bytes = AtomicUsize::new(0);
    let lanes_vs = AtomicUsize::new(0);
    let lanes_why: Mutex<HashMap<String, usize>> = Mutex::new(HashMap::new());
    // `--hash H`: only the container whose `[SHADER]` hash (FNV-64 of the container) is H, every
    // record searched; `--dump DIR`: write each program's direct DXIL (and with `--compare` the
    // HLSL path's source and DXIL) to DIR, for diffing the two (`dxc -dumpbin`)
    let only_hash = opts.hash;
    let dump_dir = opts.dump.as_ref();
    if let Some(d) = dump_dir {
        let _ = std::fs::create_dir_all(d);
    }
    let (order, want, max_bytes) = if only_hash.is_some() { ((0..input.len()).collect(), usize::MAX, 0) } else { (order, want, max_bytes) };
    // translator panics are counted, not printed (`--trace` keeps the messages)
    if !trace {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let results: Mutex<Vec<(String, Sample)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&ri) = order.get(i) else { break };
                if taken.load(Ordering::Relaxed) >= want {
                    break;
                }
                let (name, Some(blob)) = input.get(ri) else { continue };
                let blob_hash = crate::gpu::shader::fnv64(&blob);
                if only_hash.is_some_and(|h| h != blob_hash) {
                    continue;
                }
                let Some(sh) = xbox::parse(&blob) else { continue };
                for (fi, f) in sh.fssm.iter().enumerate() {
                    let Some(mode) = recomp::interface(&blob, &sh, f) else { continue };
                    for p in &f.programs {
                        if taken.fetch_add(1, Ordering::Relaxed) >= want {
                            return;
                        }
                        let code = &f.data[p.offset..p.offset + p.len];
                        if max_bytes > 0 && code.len() > max_bytes {
                            skipped.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        let wave64 = recomp::fssm_wave64(f, code);
                        let o = crate::gpu::dxil::kernel::KernelOptions { wave64, split: wave64 && split, mem_windows: windows, sampler_base: 0 };
                        if trace {
                            eprintln!("direct: {name} FSSM#{fi} ({} B, {})", code.len(), coverage::stage_name(&mode));
                        }
                        let t = std::time::Instant::now();
                        let r = {
                            let (tx, rx) = std::sync::mpsc::channel();
                            let (c, m, rodata) = (code.to_vec(), mode.clone(), f.rodata(p));
                            std::thread::spawn(move || {
                                let _ = tx.send(coverage::translate_mode(&c, &m, o, validate, &rodata));
                            });
                            rx.recv_timeout(timeout).unwrap_or_else(|_| Err(format!("timeout: over {} s", timeout.as_secs())))
                        };
                        let time = t.elapsed();
                        if let (true, recomp::Mode::Compute(cs)) = (lanes, &mode) {
                            lanes_tried.fetch_add(1, Ordering::Relaxed);
                            // under the sweep's timeout too (the thread is left to finish)
                            let (tx, rx) = std::sync::mpsc::channel();
                            let (c, csc) = (code.to_vec(), cs.clone());
                            std::thread::spawn(move || {
                                let _ = tx.send(recomp::lower_dxil::compute_with(&c, &csc, o));
                            });
                            match rx.recv_timeout(timeout).unwrap_or_else(|_| Err(format!("timeout: over {} s", timeout.as_secs()))) {
                                Ok(k) => {
                                    lanes_ok.fetch_add(1, Ordering::Relaxed);
                                    lanes_bytes.fetch_add(k.dxil.len(), Ordering::Relaxed);
                                    lanes_vs.fetch_add(r.as_ref().map_or(0, |d| d.len()), Ordering::Relaxed);
                                    if trace {
                                        eprintln!("per-lane: {name} FSSM#{fi}: {} B, per-instruction {:?} B", k.dxil.len(), r.as_ref().map(|d| d.len()));
                                    }
                                }
                                Err(e) => *lanes_why.lock().unwrap().entry(survey_key(&e)).or_default() += 1,
                            }
                        }
                        let dump = |suffix: &str, bytes: &[u8]| {
                            if let Some(d) = &dump_dir {
                                let _ = std::fs::write(d.join(format!("{blob_hash:016x}_f{fi}_{:x}.{suffix}", p.offset)), bytes);
                            }
                        };
                        dump("guest.bin", code);
                        match &r {
                            Ok(dxil) => dump("direct.dxil", dxil),
                            Err(e) => dump("direct.err.txt", e.as_bytes()),
                        }
                        let hlsl_size = (compare && r.is_ok())
                            .then(|| {
                                let profile = match mode {
                                    recomp::Mode::Vertex(_) => "vs_6_6",
                                    recomp::Mode::Pixel(_) => "ps_6_6",
                                    _ => "cs_6_6",
                                };
                                let opt = recomp::Options { mode: mode.clone(), wave64, split_wave64: wave64 && split, mem_windows: windows, rodata: f.rodata(p), ..Default::default() };
                                let t = recomp::translate(code, &opt).ok().filter(|t| t.complete())?;
                                dump("hlsl", t.hlsl.as_bytes());
                                let d = recomp::dxc::compile(&t.hlsl, "main", profile, &[]).ok()?;
                                dump("hlsl.dxil", &d);
                                Some(d.len())
                            })
                            .flatten();
                        let sample = Sample { stage: coverage::stage_name(&mode), result: r.map(|d| d.len()), time, hlsl_size };
                        results.lock().unwrap().push((format!("{name} FSSM#{fi}"), sample));
                    }
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by(|a, b| a.0.cmp(&b.0));
    let mut rep = Report::default();
    for (name, s) in &results {
        rep.add(name, s);
    }
    if json {
        println!("{}", rep.json(top));
    } else {
        println!("{} programs from {} sampled records (seed {seed}; {} over {max_bytes} B skipped){}{}", results.len(), order.len().min(input.len()), skipped.load(Ordering::Relaxed), if validate { ", validated" } else { "" }, if compare { ", compared with HLSL+DXC" } else { "" });
        if lanes {
            let mut why: Vec<(String, usize)> = lanes_why.into_inner().unwrap().into_iter().collect();
            why.sort_by(|a, b| b.1.cmp(&a.1));
            why.truncate(top);
            println!("per-lane form: {} of {} compute programs; where it translates, {} B against {} B per instruction; refused for: {why:?}", lanes_ok.load(Ordering::Relaxed), lanes_tried.load(Ordering::Relaxed), lanes_bytes.load(Ordering::Relaxed), lanes_vs.load(Ordering::Relaxed));
        }
        print!("{}", rep.text(top));
        println!("time {:.1} s", t0.elapsed().as_secs_f64());
    }
    0
}

/// A survey reason without its program's numbers: registers as `sN`, other numbers as `N`
/// (an `s_setreg`'s hardware register is kept), at most eight words.
fn survey_key(r: &str) -> String {
    let r = r.split(" (").next().unwrap_or(r);
    let words: Vec<&str> = r.split_whitespace().collect();
    let mut out: Vec<String> = Vec::new();
    for (i, raw) in words.iter().enumerate() {
        let w = raw.trim_end_matches([':', ',']);
        let digits = |x: &str| !x.is_empty() && x.bytes().all(|b| b.is_ascii_digit());
        let w = if w.strip_prefix('s').is_some_and(digits) {
            "sN".to_string()
        } else if digits(w) && !(i > 0 && words[i - 1] == "hwreg") {
            "N".to_string()
        } else {
            w.to_string()
        };
        out.push(if raw.ends_with(':') { format!("{w}:") } else { w });
        if out.len() == 8 {
            break;
        }
    }
    out.join(" ")
}

/// A per-lane refusal no instruction lowering removes (the CFG, masks across joins, lanes
/// elsewhere, the group shape), as opposed to an instruction the per-lane form lacks.
fn survey_structural(r: &str) -> bool {
    [
        "2-D / 3-D",
        "elsewhere",
        "read as a number",
        "read as a mask",
        "across two masks",
        "successors",
        "unsupported branch",
        "neither a number nor a mask",
        "empty program",
        "lane conditions",
        "indirect jump",
        "branch to the entry",
        "irreducible",
        "no fixpoint",
        "cannot decode",
    ]
    .iter()
    .any(|w| r.contains(w))
}
