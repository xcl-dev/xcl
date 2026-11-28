//! Diagnostic: render-target / depth-stencil descriptors that console code copied itself.
//!
//! The console's descriptor helpers are compiled into the title. Its CopyDescriptorsSimple of
//! render-target or depth-stencil descriptors is a plain memory copy of N × 112 / 128 bytes (the
//! console's descriptor sizes), not a device call. XCL's RTV / DSV heaps are PC heaps with PC
//! increments, so such a copy leaves handles that no CreateRenderTargetView /
//! CreateDepthStencilView / device copy wrote. A handle in one of the title's RTV / DSV heaps that
//! is bound or cleared without one of those is logged, with its heap and offset (a few lines, then
//! the checks stop). Handles outside the title's heaps (XCL's own) are not checked.
//!
//! The device's +0x10 (the console's null constant-buffer address, which inline null CBVs use) and
//! +0x18 (the byte a validation build reads to choose the inline path) are logged once per device:
//! on XCL they are PC runtime data. No behaviour change.

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

const RTV: u32 = 2;
const DSV: u32 = 3;
/// unknown-handle lines logged before the checks stop
const MAX_LINES: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Heap {
    start: usize,
    n: usize,
    inc: usize,
    ty: u32,
}

/// the title's RTV / DSV heaps (CPU start, descriptor count, increment, type)
static HEAPS: RwLock<Vec<Heap>> = RwLock::new(Vec::new());
/// handles a view creation or a device copy wrote
static KNOWN: RwLock<Option<HashSet<usize>>> = RwLock::new(None);
static LINES: AtomicU64 = AtomicU64::new(0);
static DONE: AtomicBool = AtomicBool::new(false);

fn is_target(ty: u32) -> bool {
    ty == RTV || ty == DSV
}

/// A title heap of PC type `ty` was created at CPU handle `start`.
pub(super) fn note_heap(ty: u32, start: usize, n: usize, inc: usize) {
    if !is_target(ty) || start == 0 || inc == 0 {
        return;
    }
    let mut g = HEAPS.write().unwrap();
    // a new heap at a released one's address replaces it
    g.retain(|h| h.start + h.n * h.inc <= start || start + n * inc <= h.start);
    g.push(Heap { start, n, inc, ty });
}

fn heap_of(handle: usize) -> Option<Heap> {
    HEAPS.read().unwrap().iter().find(|h| handle >= h.start && handle < h.start + h.n * h.inc).copied()
}

/// The increment of the title heap `handle` is in (0: none).
pub(super) fn increment_at(handle: usize) -> usize {
    heap_of(handle).map_or(0, |h| h.inc)
}

/// A view was written at `handle` (CreateRenderTargetView / CreateDepthStencilView).
pub(super) fn note_view(handle: usize) {
    if handle != 0 && !DONE.load(Ordering::Relaxed) {
        KNOWN.write().unwrap().get_or_insert_with(HashSet::new).insert(handle);
    }
}

/// A device copy of PC type `ty` wrote `n` descriptors from `dst` on.
pub(super) fn note_copy(ty: u32, dst: usize, n: u32) {
    if !is_target(ty) || dst == 0 || DONE.load(Ordering::Relaxed) {
        return;
    }
    let Some(h) = heap_of(dst) else { return };
    let mut g = KNOWN.write().unwrap();
    let set = g.get_or_insert_with(HashSet::new);
    for k in 0..n as usize {
        set.insert(dst + k * h.inc);
    }
}

/// What the unknown-handle line says about `handle` in `heap` (None: nothing to report).
fn unknown(handle: usize, heap: Option<Heap>, known: bool) -> Option<String> {
    let h = heap?;
    if known {
        return None;
    }
    let off = handle - h.start;
    let kind = if h.ty == RTV { "RTV" } else { "DSV" };
    Some(format!(
        "{kind} heap at {:#x} ({} descriptors of {} B), offset {off:#x} = descriptor {} + {} B",
        h.start,
        h.n,
        h.inc,
        off / h.inc,
        off % h.inc
    ))
}

/// `what` (a list method) got `handle`: log it when it is in a title RTV / DSV heap and nothing
/// wrote it through the device.
pub(super) fn check(handle: usize, what: &str) {
    if handle == 0 || DONE.load(Ordering::Relaxed) {
        return;
    }
    let heap = heap_of(handle);
    if heap.is_none() {
        return;
    }
    let known = KNOWN.read().unwrap().as_ref().is_some_and(|s| s.contains(&handle));
    let Some(place) = unknown(handle, heap, known) else { return };
    let n = LINES.fetch_add(1, Ordering::Relaxed);
    if n >= MAX_LINES {
        DONE.store(true, Ordering::Relaxed);
        return;
    }
    crate::xlog_warn!(
        "D3D12.X: {what} with handle {handle:#x} that no view creation or device copy wrote ({place}): console code may have copied render-target / depth descriptors itself at the console size (112 / 128 B)"
    );
    if n + 1 == MAX_LINES {
        DONE.store(true, Ordering::Relaxed);
    }
}

/// The title's device was adopted: log the fields console code reads inline (once per device).
pub(super) unsafe fn note_device(dev: *mut c_void) {
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    if dev.is_null() || !crate::gpu::shader::readable(dev as usize, 0x20) {
        return;
    }
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&(dev as usize)) || g.len() >= 4 {
            return;
        }
        g.push(dev as usize);
    }
    let null_cb = (dev as *const u8).add(0x10).cast::<u64>().read_unaligned();
    let inline = *(dev as *const u8).add(0x18);
    let in_uma = crate::gpu::uma::get().is_some_and(|u| u.contains(null_cb as usize));
    crate::xlog_info!(
        "D3D12.X: device {dev:p}: +0x10 (console null constant-buffer address) {null_cb:#x}{}, +0x18 (validation build: 0 = inline descriptor writes) {inline:#x}",
        if in_uma { " (in unified memory)" } else { " (not in unified memory)" }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unwritten_handle_in_a_title_heap_is_reported_with_its_place() {
        let h = Heap { start: 0x1000, n: 8, inc: 32, ty: RTV };
        // a console copy of descriptor 1 lands 112 B in: descriptor 3 + 16 B on a 32 B heap
        let s = unknown(0x1000 + 112, Some(h), false).unwrap();
        assert!(s.contains("RTV heap at 0x1000") && s.contains("offset 0x70 = descriptor 3 + 16 B"), "{s}");
        assert_eq!(unknown(0x1000 + 32, Some(h), true), None, "a written handle");
        assert_eq!(unknown(0x9000, None, false), None, "outside the title's heaps");
    }

    #[test]
    fn heaps_and_copies_mark_handles() {
        note_heap(DSV, 0x7_0000, 4, 64);
        note_heap(0, 0x8_0000, 4, 64);
        assert_eq!(heap_of(0x7_0000 + 3 * 64).map(|h| h.ty), Some(DSV));
        assert_eq!(heap_of(0x7_0000 + 4 * 64), None);
        assert_eq!(heap_of(0x8_0000), None, "a view heap is not tracked");
        note_copy(DSV, 0x7_0000 + 64, 2);
        let known = |x: usize| KNOWN.read().unwrap().as_ref().is_some_and(|s| s.contains(&x));
        assert!(known(0x7_0000 + 64) && known(0x7_0000 + 128) && !known(0x7_0000 + 192));
        // a heap at the same address replaces the old one
        note_heap(RTV, 0x7_0000, 2, 32);
        assert_eq!(heap_of(0x7_0000).map(|h| (h.ty, h.inc)), Some((RTV, 32)));
    }
}
