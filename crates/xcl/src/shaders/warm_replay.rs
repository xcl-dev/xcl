//! A recipe can be left on disk only after a successful replay identified every
//! native blob it used. The render device still creates the live PSO on first use.
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::os::windows::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Interrupted writes or a damaged index fall back to ordinary preparation.
pub(crate) fn read_receipt<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    let (checksum, payload) = bytes.split_at_checked(8)?;
    if checksum != crate::gpu::shader::fnv64(payload).to_le_bytes() { return None; }
    serde_json::from_slice(payload).ok()
}

pub(crate) fn write_receipt<T: Serialize>(path: &Path, receipt: &T) -> bool {
    let Some(bytes) = receipt_bytes(receipt) else { return false };
    crate::hle::d3d12x::real::driver_cache::write(path, &bytes)
}

pub(crate) fn receipt_bytes<T: Serialize>(receipt: &T) -> Option<Vec<u8>> {
    let mut bytes = vec![0; 8];
    if serde_json::to_writer(&mut bytes, receipt).is_err() { return None; }
    let checksum = crate::gpu::shader::fnv64(&bytes[8..]).to_le_bytes();
    bytes[..8].copy_from_slice(&checksum);
    Some(bytes)
}

pub(crate) type Stamp = [u64; 3];
pub(crate) type Inventory = HashMap<String, Stamp>;

fn stamp(m: &std::fs::Metadata) -> Option<Stamp> {
    (m.is_file() && m.len() != 0 && m.file_attributes() & 0x400 == 0).then(|| [m.len(), m.creation_time(), m.last_write_time()])
}

pub(crate) fn driver_files(dir: &Path) -> Option<Inventory> {
    let mut files = Inventory::new();
    for e in std::fs::read_dir(dir).ok()? {
        let e = e.ok()?;
        let m = e.metadata().ok()?;
        if m.file_attributes() & 0x400 != 0 { continue; }
        let Some(stamp) = stamp(&m) else { continue };
        if let Ok(name) = e.file_name().into_string() {
            if name.ends_with(".pso") { files.insert(name, stamp); }
        }
    }
    Some(files)
}
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct File { name: String, stamp: Stamp }
type Recipes = HashMap<u64, Vec<File>>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pass { Graphics, Mesh, Compute }

pub(super) struct Captured { file: File, pass: Pass }

thread_local! { static CAPTURE: RefCell<Option<Vec<Captured>>> = const { RefCell::new(None) }; }

pub(super) fn capture(f: impl FnOnce() -> bool) -> (bool, Vec<Captured>) {
    CAPTURE.with(|slot| { *slot.borrow_mut() = Some(Vec::new()); });
    let success = f();
    let files = CAPTURE.with(|slot| slot.take().unwrap_or_default());
    (success, files)
}

#[cfg(test)]
pub(crate) fn capture_for_test(f: impl FnOnce() -> bool) -> bool { capture(f).0 }

/// Keep only receipts recorded before a build attempt that must be retried.
pub(crate) fn checkpoint() -> usize { CAPTURE.with(|slot| slot.borrow().as_ref().map_or(0, Vec::len)) }

pub(crate) fn discard_since(checkpoint: usize) {
    CAPTURE.with(|slot| {
        if let Some(files) = slot.borrow_mut().as_mut() { files.truncate(checkpoint); }
    });
}

/// Record only bytes accepted by the native driver that still occupy this cache file.
pub(crate) fn accepted(path: Option<&Path>, bytes: &[u8], pass: Pass) {
    CAPTURE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(files) = slot.as_mut() else { return };
        let verified = || {
            let path = path?;
            let before = stamp(&path.metadata().ok()?)?;
            if before[0] != bytes.len() as u64 || std::fs::read(path).ok()?.as_slice() != bytes { return None; }
            if stamp(&path.metadata().ok()?)? != before { return None; }
            Some(File { name: path.file_name()?.to_str()?.to_owned(), stamp: before })
        };
        if let Some(file) = verified() { files.push(Captured { file, pass }); }
    });
}

pub(super) struct Index {
    path: PathBuf,
    entries: Recipes,
    new: Mutex<Recipes>,
}

impl Index {
    pub(super) fn open(dir: &Path, known: &Inventory, key: u64, windows: u32) -> Option<Self> {
        let path = dir.join(format!("warm-recipes-v3-{key:016x}-{windows:02x}.bin"));
        let mut entries: Recipes = read_receipt(&path).unwrap_or_default();
        entries.retain(|_, files| !files.is_empty() && files.iter().all(|f| known.get(&f.name) == Some(&f.stamp)));
        Some(Self { path, entries, new: Default::default() })
    }
    pub(super) fn can_defer(&self, hash: u64) -> bool { self.entries.contains_key(&hash) }
    pub(super) fn add(&self, hash: u64, files: Vec<Captured>, mesh: bool) {
        let complete = if mesh {
            files.len() == 2 && files.iter().any(|f| f.pass == Pass::Mesh)
                && files.iter().any(|f| f.pass == Pass::Compute)
                && files[0].file.name != files[1].file.name
        } else {
            files.len() == 1 && files[0].pass == Pass::Graphics
        };
        if complete { self.new.lock().unwrap().insert(hash, files.into_iter().map(|f| f.file).collect()); }
    }
    pub(super) fn save(mut self) {
        let new = self.new.into_inner().unwrap();
        if new.is_empty() { return; }
        self.entries.extend(new);
        if !write_receipt(&self.path, &self.entries) {
            crate::xlog_warn!("PSO preparation: could not save the verified recipe index");
        }
    }
}

pub(super) fn native_key(dev: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> Option<u64> {
    let gpu = crate::hle::d3d12x::real::preparation_driver_identity(dev)?;
    let translation = crate::gpu::recomp::pso::translation_cache_identity();
    if translation == 0 || !crate::hle::d3d12x::real::driver_cache::enabled() { return None; }
    let cache = std::path::absolute(crate::gpu::recomp::dxc::cache_dir()).ok()?;
    Some(super::warm_blobs::digest(&(env!("XCL_NATIVE_PREPARATION_ID"), gpu, translation, cache, super::warm_settings_key())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_mesh_needs_both_unchanged_driver_files() {
        use crate::hle::d3d12x::real::driver_cache;
        let dir = std::env::temp_dir().join(format!("xcl-warm-mesh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let graphics = dir.join(driver_cache::graphics_name(3, 64));
        let compute = dir.join(driver_cache::name_for_root_node(b"synthetic compute", 0, false, &[], 0));
        std::fs::write(&graphics, b"graphics").unwrap();
        std::fs::write(&compute, b"compute").unwrap();
        let open = || Index::open(&dir, &driver_files(&dir).unwrap(), 1, 4).unwrap();
        let capture_passes = |compute_bytes: &[u8]| capture(|| {
            accepted(Some(&graphics), b"graphics", Pass::Mesh);
            accepted(Some(&compute), compute_bytes, Pass::Compute);
            true
        }).1;
        let book = open();
        book.add(1, capture_passes(b"wrong"), true);
        book.add(2, capture_passes(b"compute"), true);
        book.save();
        assert!(!open().can_defer(1));
        assert!(open().can_defer(2));
        std::fs::remove_file(&compute).unwrap();
        assert!(!open().can_defer(2));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn native_mesh_cache_names_make_complete_receipts() {
        use crate::hle::d3d12x::real::driver_cache;
        let dir = std::env::temp_dir().join(format!("xcl-warm-native-mesh-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let mesh_a = dir.join(driver_cache::graphics_name(1, 64));
        let mesh_b = dir.join(driver_cache::graphics_name(2, 64));
        let compute = dir.join(driver_cache::name_for_root_node(b"synthetic compute", 0, false, &[], 0));
        std::fs::write(&mesh_a, b"mesh a").unwrap();
        std::fs::write(&mesh_b, b"mesh b").unwrap();
        std::fs::write(&compute, b"compute").unwrap();
        let open = || Index::open(&dir, &driver_files(&dir).unwrap(), 2, 4).unwrap();
        let book = open();
        for (key, mesh, bytes) in [(1, &mesh_a, b"mesh a".as_slice()), (2, &mesh_b, b"mesh b".as_slice())] {
            let (_, files) = capture(|| {
                accepted(Some(mesh), bytes, Pass::Mesh);
                accepted(Some(&compute), b"compute", Pass::Compute);
                true
            });
            book.add(key, files, true);
        }
        let (_, wrong_roles) = capture(|| {
            accepted(Some(&mesh_a), b"mesh a", Pass::Mesh);
            accepted(Some(&mesh_b), b"mesh b", Pass::Mesh);
            true
        });
        book.add(3, wrong_roles, true);
        let (_, stale_mesh) = capture(|| {
            let start = checkpoint();
            accepted(Some(&mesh_a), b"mesh a", Pass::Mesh);
            discard_since(start);
            accepted(Some(&compute), b"compute", Pass::Compute);
            true
        });
        book.add(4, stale_mesh, true);
        let (_, retried_pair) = capture(|| {
            let start = checkpoint();
            accepted(Some(&mesh_a), b"mesh a", Pass::Mesh);
            discard_since(start);
            accepted(Some(&mesh_b), b"mesh b", Pass::Mesh);
            accepted(Some(&compute), b"compute", Pass::Compute);
            true
        });
        book.add(5, retried_pair, true);
        book.save();
        assert!(open().can_defer(1));
        assert!(open().can_defer(2));
        assert!(!open().can_defer(3));
        assert!(!open().can_defer(4));
        assert!(open().can_defer(5));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn old_graphics_receipt_is_revalidated() {
        let dir = std::env::temp_dir().join(format!("xcl-warm-v2-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let graphics = dir.join(crate::hle::d3d12x::real::driver_cache::graphics_name(4, 64));
        std::fs::write(&graphics, b"graphics").unwrap();
        let old = dir.join("warm-recipes-v2-0000000000000004-04.bin");
        let files = vec![File { name: graphics.file_name().unwrap().to_str().unwrap().to_owned(),
            stamp: stamp(&graphics.metadata().unwrap()).unwrap() }];
        assert!(write_receipt(&old, &HashMap::from([(7u64, files)])));
        let book = Index::open(&dir, &driver_files(&dir).unwrap(), 4, 4).unwrap();
        assert!(!book.can_defer(7), "old one-file receipts cannot prove a two-pass build");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
