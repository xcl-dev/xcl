//! The console's quad sample mask: command list #142 of the newer layout, `SetQuadSampleMaskX(u64)`
//! (the name as in xcl-dev/xcl PR #1274, which leaves it a no-op). The u64 is the two
//! PA_SC_AA_MASK registers of AMD's rasteriser: 16 sample bits for each pixel of a 2×2 quad,
//! X0Y0 in bits 0..15, X1Y0 in 16..31, X0Y1 in 32..47, X1Y1 in 48..63 (register names
//! PA_SC_AA_MASK_X0Y0_X1Y0 / _X0Y1_X1Y1). A retail title sets it per draw from fade tables
//! (0, 1, 2, 3 or 4 pixels of the quad, each 0 or 0xFFFF, and the complement for the other
//! side of a cross-fade): dithered LOD fades. As a stub every pixel is drawn: what should
//! fade in a dither pattern is drawn whole.
//!
//! The PC has no such mask: the translated pixel shaders test it (`recomp::IO_QUAD_KILL`, the
//! inverted words in the draw's io block, written by `bind`) and discard the pixels it takes
//! away. Off unless `XCL_SAMPLE_MASK=1`: off, the mask is taken as all ones (the io words stay
//! 0 and the shaders' test never discards).

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// `XCL_SAMPLE_MASK=1` applies the mask (no translation changes: the test is in every pixel
/// shader). Off by default.
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SAMPLE_MASK").is_ok_and(|v| v.trim() == "1"))
}

/// The io words (`recomp::IO_QUAD_KILL`) of a mask: the samples it takes away, quad row by quad
/// row ([0] = pixels (0, 0) and (1, 0), [1] = (0, 1) and (1, 1)). All ones: [0, 0].
pub(super) fn kill_words(mask: u64) -> [u32; 2] {
    let k = !mask;
    [k as u32, (k >> 32) as u32]
}

/// Whether the translated pixel shaders keep the pixel at (x, y) of the screen under the io
/// `words` (`emit::quad_mask_prologue`): not when every sample bit of its quad position is set.
#[cfg(test)]
pub(super) fn pixel_kept(words: [u32; 2], x: u32, y: u32) -> bool {
    let w = words[(y & 1) as usize];
    (w >> (16 * (x & 1))) & 0xFFFF != 0xFFFF
}

/// calls of #142 by the title, of them with a mask that takes something away
static CALLS: AtomicU64 = AtomicU64::new(0);
static PARTIAL: AtomicU64 = AtomicU64::new(0);
/// draws bound with a partial mask
static MASKED_DRAWS: AtomicU64 = AtomicU64::new(0);

/// Command list #142 `SetQuadSampleMaskX(mask)` (newer layout only: on the older one #142 is
/// another method, answered as the stub did). The mask holds for the list's next draws, until
/// the next call or the list's reset (the title assumes all ones after a reset: it resends only
/// a changed mask, and its cache starts at all ones).
pub(super) unsafe extern "system" fn set_quad_sample_mask_x(this: *mut c_void, mask: u64) -> u64 {
    if !super::real::QUEUE_SHIFTED.load(Ordering::Acquire) {
        return 0;
    }
    let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    let partial = if mask != u64::MAX { PARTIAL.fetch_add(1, Ordering::Relaxed) + 1 } else { PARTIAL.load(Ordering::Relaxed) };
    if n == 1 {
        crate::xlog_info!(
            "D3D12.X: list #142 SetQuadSampleMaskX({mask:#x}) – the quad sample mask is {}",
            if enabled() { "on (XCL_SAMPLE_MASK=1)" } else { "off: every pixel is drawn (XCL_SAMPLE_MASK=1 applies it)" }
        );
    }
    if n.is_power_of_two() && n >= 1024 {
        crate::xlog_info!(
            "D3D12.X: quad sample mask: {n} calls, {partial} with a partial mask, {} draws bound with one",
            MASKED_DRAWS.load(Ordering::Relaxed)
        );
    }
    super::bind::set_quad_kill(this, if enabled() { kill_words(mask) } else { [0; 2] });
    0
}

/// A draw's io block got a partial mask.
pub(super) fn note_masked_draw() {
    MASKED_DRAWS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::{kill_words, pixel_kept};

    /// the pixels of the quad a mask keeps, as bits (x + 2y)
    fn kept(mask: u64) -> u32 {
        let w = kill_words(mask);
        (0..4).filter(|&i| pixel_kept(w, i & 1, i >> 1)).map(|i| 1 << i).sum()
    }

    #[test]
    fn all_ones_takes_nothing_away_and_writes_zero_io_words() {
        assert_eq!(kill_words(u64::MAX), [0, 0]);
        assert_eq!(kept(u64::MAX), 0xF);
        assert_eq!(kill_words(0), [u32::MAX, u32::MAX]);
        assert_eq!(kept(0), 0);
    }

    /// The masks a retail title sets (counted over a run: all ones, the two diagonals, "3 of 4"
    /// and "1 of 4") and its fade tables' order of the 16-bit words: word i = pixel
    /// (i & 1, i >> 1).
    #[test]
    fn the_titles_masks_keep_their_quad_pixels() {
        let cases: [(u64, u32); 10] = [
            (0xFFFF_FFFF_FFFF_FFFF, 0xF),
            (0x0000_FFFF_FFFF_0000, 0b0110),
            (0xFFFF_0000_0000_FFFF, 0b1001),
            (0x0000_FFFF_FFFF_FFFF, 0b0111),
            (0xFFFF_0000_FFFF_FFFF, 0b1011),
            (0xFFFF_FFFF_0000_FFFF, 0b1101),
            (0xFFFF_FFFF_FFFF_0000, 0b1110),
            (0x0000_0000_0000_FFFF, 0b0001),
            (0x0000_FFFF_0000_0000, 0b0100),
            (0xFFFF_0000_0000_0000, 0b1000),
        ];
        for (mask, want) in cases {
            assert_eq!(kept(mask), want, "{mask:#018x}");
            // the complement (the other side of a cross-fade) keeps the other pixels
            assert_eq!(kept(!mask), !want & 0xF, "!{mask:#018x}");
        }
        // the test reads the parity of the screen position only
        let w = kill_words(0x0000_0000_0000_FFFF);
        assert!(pixel_kept(w, 10, 4) && !pixel_kept(w, 11, 4) && !pixel_kept(w, 10, 5) && !pixel_kept(w, 11, 5));
    }

    /// A pixel with some of its samples kept is drawn whole (the title only sets 0 / 0xFFFF).
    #[test]
    fn a_pixel_keeps_its_samples_if_any_is_kept() {
        let w = kill_words(0xFFFF_FFFF_FFFF_0001);
        assert!(pixel_kept(w, 0, 0));
        let w = kill_words(0xFFFF_FFFF_FFFF_0000);
        assert!(!pixel_kept(w, 0, 0) && pixel_kept(w, 1, 0) && pixel_kept(w, 0, 1) && pixel_kept(w, 1, 1));
    }
}
