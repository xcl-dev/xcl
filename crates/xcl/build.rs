use std::path::{Path, PathBuf};
use std::process::Command;

// `pso.rs` also contains cache storage and pipeline setup. Keep it out of the source hash so
// those changes do not discard translations; bump this only when its container parsing,
// translation options, profiles, or post-translation HLSL changes.
const TRANSLATION_GLUE_REVISION: &str = "pso-glue-v5";

const TRANSLATOR_SOURCES: &[&str] = &[
    "src/gpu/dxbc.rs",
    "src/gpu/rdna2.rs",
    "src/gpu/rdna2_spec.rs",
    "src/gpu/rdna2_xbox.rs",
    "src/gpu/recomp/dxc.rs",
    "src/gpu/recomp/emit.rs",
    "src/gpu/recomp/mesh_wave.rs",
    "src/gpu/recomp/mod.rs",
    "src/gpu/recomp/upper_lanes.rs",
    "src/gpu/recomp/x64_masks.rs",
    "src/gpu/shader.rs",
    "src/gpu/xbox.rs",
];

/// The per-lane form's sources (`XCL_PER_LANE`): part of only its own cache entries' identity,
/// so editing them does not invalidate the HLSL path's.
const PER_LANE_SOURCES: &[&str] = &[
    "src/gpu/recomp/lane_masks.rs",
    "src/gpu/recomp/lower.rs",
    "src/gpu/recomp/lower/wave.rs",
    "src/gpu/recomp/lower_dxil.rs",
    "src/gpu/recomp/ssa.rs",
];

/// The direct DXIL translator's sources (`XCL_DIRECT_DXIL`), part of only its own cache entries'
/// identity: every `.rs` file under `dir` but the test files (`*tests.rs`; tests inside the other
/// files are hashed with them), so a new one cannot be forgotten.
fn direct_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("list direct DXIL sources") {
        let path = entry.expect("list direct DXIL sources").path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if path.is_dir() {
            direct_sources(&path, out);
        } else if name.ends_with(".rs") && !name.ends_with("tests.rs") {
            out.push(path);
        }
    }
}

// FNV-1a over the sources, each CRLF taken as LF (`source_hash.rs`, tested with the crate)
#[path = "src/source_hash.rs"]
mod source_hash;
use source_hash::{add, add_source};

// Bump when the native builder (including the D3D12.X glue in real.rs) changes the
// mapping from a journal recipe to driver descriptors or blob acceptance. Shader
// translator/toolchain, driver, adapter, settings and per-file stamps are keyed separately.
const NATIVE_PREPARATION_ABI: &[u8] = b"xcl-native-preparation-v1";
const PREPARATION_LAYOUT_SOURCES: &[&str] = &[
    "src/gpu/recomp/pso.rs",          // root signatures, translation glue, split pipeline layout
    "src/hle/d3d12x/real/two_pass_identity.rs", // driver-blob identities for both mesh passes
    "src/hle/d3d12x/gfx_records.rs",  // normalized recipe fields and reconstruction
];

fn preparation_layout_id(root: &Path) -> u64 {
    let mut id = 0xcbf2_9ce4_8422_2325u64;
    add(&mut id, NATIVE_PREPARATION_ABI);
    for &relative in PREPARATION_LAYOUT_SOURCES {
        println!("cargo:rerun-if-changed={relative}");
        add(&mut id, relative.as_bytes());
        add(&mut id, &[0]);
        add_source(&mut id, &std::fs::read(root.join(relative)).expect("read pipeline layout input"));
        add(&mut id, &[0]);
    }
    id
}

fn preparation_source_id(mut id: u64) -> u64 {
    for name in ["TARGET", "PROFILE", "CARGO_ENCODED_RUSTFLAGS", "OPT_LEVEL", "DEBUG"] {
        add(&mut id, name.as_bytes());
        add(&mut id, std::env::var(name).unwrap_or_default().as_bytes());
        add(&mut id, &[0]);
    }
    let rustc = std::process::Command::new(std::env::var_os("RUSTC").expect("rustc"))
        .arg("-vV").output().expect("read compiler identity");
    assert!(rustc.status.success());
    add_source(&mut id, &rustc.stdout);
    let mut features: Vec<_> = std::env::vars().filter(|(name,_)| name.starts_with("CARGO_FEATURE_")).collect();
    features.sort();
    for (name, value) in features { add(&mut id, name.as_bytes()); add(&mut id, value.as_bytes()); add(&mut id, &[0]); }
    id
}

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTC");
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let rustc_version = Command::new(rustc).arg("--version").output().ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=XCL_BUILD_RUSTC={rustc_version}");
    println!("cargo:rustc-env=XCL_BUILD_PROFILE={}", std::env::var("PROFILE").unwrap_or_else(|_| "unknown".into()));
    println!("cargo:rustc-env=XCL_BUILD_OPT_LEVEL={}", std::env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".into()));
    println!("cargo:rustc-env=XCL_BUILD_DEBUG_ASSERTIONS={}", std::env::var("DEBUG").unwrap_or_else(|_| "unknown".into()));

    // xcl.exe without high-entropy ASLR: the title's memory then lies under 2^40, which a
    // retail title's BVH builder requires of its GPU addresses (#585, PR #631)
    match std::env::var("CARGO_CFG_TARGET_ENV").as_deref() {
        Ok("msvc") => println!("cargo:rustc-link-arg-bin=xcl=/HIGHENTROPYVA:NO"),
        Ok("gnu") => println!("cargo:rustc-link-arg-bin=xcl=-Wl,--disable-high-entropy-va"),
        _ => {}
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let layout = preparation_layout_id(root);
    println!("cargo:rustc-env=XCL_PIPELINE_LAYOUT_ID={layout:016x}");
    println!("cargo:rustc-env=XCL_NATIVE_PREPARATION_ID={:016x}", preparation_source_id(layout));
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    add(&mut h, b"xcl-translator-source-v1\0");
    add(&mut h, TRANSLATION_GLUE_REVISION.as_bytes());
    add(&mut h, &[0]);
    for relative in TRANSLATOR_SOURCES {
        println!("cargo:rerun-if-changed={relative}");
        add(&mut h, relative.as_bytes());
        add(&mut h, &[0]);
        add_source(&mut h, &std::fs::read(root.join(relative)).expect("read shader translator source"));
        add(&mut h, &[0]);
    }
    println!("cargo:rustc-env=XCL_TRANSLATOR_SOURCE_ID={h:016x}");
    let mut p = 0xcbf2_9ce4_8422_2325u64;
    for relative in PER_LANE_SOURCES {
        println!("cargo:rerun-if-changed={relative}");
        add(&mut p, relative.as_bytes());
        add(&mut p, &[0]);
        add_source(&mut p, &std::fs::read(root.join(relative)).expect("read per-lane source"));
        add(&mut p, &[0]);
    }
    println!("cargo:rustc-env=XCL_PER_LANE_SOURCE_ID={p:016x}");
    // a directory: cargo looks at every file under it
    println!("cargo:rerun-if-changed=src/gpu/dxil");
    let mut sources = Vec::new();
    direct_sources(&root.join("src/gpu/dxil"), &mut sources);
    sources.sort();
    let mut d = 0xcbf2_9ce4_8422_2325u64;
    for path in &sources {
        add(&mut d, path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/").as_bytes());
        add(&mut d, &[0]);
        add_source(&mut d, &std::fs::read(path).expect("read direct DXIL source"));
        add(&mut d, &[0]);
    }
    println!("cargo:rustc-env=XCL_DIRECT_DXIL_SOURCE_ID={d:016x}");
}
