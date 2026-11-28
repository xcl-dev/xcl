//! `xg_xs.dll` – the console's texture layout library: where the mips and texels of a texture
//! go in the title's own memory (Fortnite imports `XGCreateTextureComputer`). XCL answers from
//! its own GFX10 model (`gpu::tiling::Chain`, checked on retail textures) in the console's
//! structures; `tests/xg_oracle.rs` compares the answers with the console's own library
//! (`XCL_SAMPLE_XG`).
//!
//! * `XGCreateTextureComputer(D3D12_RESOURCE_DESC*, computer**)`: 2D textures, arrays and
//!   cubes with one sample. `Layout` 1 = linear, 0x100–0x11F = swizzle mode `Layout & 0x1F`,
//!   0 / 2 = the library's choice (`optimal_mode`).
//! * The computer (no QueryInterface): AddRef, Release, GetResourceLayout(XG_RESOURCE_LAYOUT*),
//!   GetResourceSizeBytes, GetResourceBaseAlignmentBytes, GetMipLevelOffsetBytes(plane, mip),
//!   GetTexelElementOffsetBytes(plane, mip, x, y, slice, sample, bit*), GetTexelCoordinate,
//!   CopyIntoSubresource(dst, plane, subresource, src, row pitch, slice pitch),
//!   CopyFromSubresource(dst, plane, subresource, src, row pitch, slice pitch) – a newer library
//!   has a 32-bit argument before `src` (`copy_args`) –, then 2 more.
//! * XG_RESOURCE_LAYOUT (0x1B40): size @0, alignment @8, mips @0x10, planes @0x14, planes
//!   from 0x18, 0x6C8 bytes each (usage @0, size @8, slice size @0x10, alignment @0x20, bytes
//!   per element @0x28, first mip in the tail @0x2C, mips in the tail @0x30, then 15 mips of
//!   0x70 from 0x38: size @0, offset @8, slice size @0x10, pitch in pixels @0x18 / bytes
//!   @0x1C, alignment @0x20, padded width / height / depth @0x24 / 0x28 / 0x2C, width /
//!   height in elements @0x30 / 0x34 and pixels @0x38 / 0x3C, depth @0x40, samples @0x44,
//!   swizzle mode @0x48, 1 @0x60, in the tail @0x64), dimension @0x1B38. Levels in the mip
//!   tail are reported as 256-byte tiles at their offset inside the tail block.

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

use crate::gpu::tiling::{Chain, Layout};
use crate::util::{HRESULT, S_OK};

const E_NOTIMPL: HRESULT = 0x8000_4001u32 as i32;
const E_POINTER: HRESULT = 0x8000_4003u32 as i32;
const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;
const E_FAIL: HRESULT = 0x8000_4005u32 as i32;

pub const LAYOUT_BYTES: usize = 0x1B40;
const PLANE: usize = 0x18;
const MIP: usize = 0x38;
const MIP_BYTES: usize = 0x70;
const PLANE_BYTES: usize = MIP + 15 * MIP_BYTES;
const MAX_MIPS: usize = 15;

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![
        XGCreateTextureComputer, XGSuggestSwizzleMode,
        XGCreateTexture1DComputer, XGCreateTexture2DComputer, XGCreateTexture3DComputer,
        XGComputeTexture1DLayout, XGComputeTexture2DLayout, XGComputeTexture3DLayout,
        XGComputeBufferLayout, XGCreateTextureComputerFromDescriptor, XGComputeTileShape,
        XGComputeOptimalSwizzleMode, XGComputeOptimalDepthStencilSwizzleMode, XGValidateDepthStencilSwizzleMode,
        XGInitializeLibrary, XGTerminateLibrary, XGSetHardwareVersion,
        XGEnableExclusiveLocking, XGEnterExclusiveLock, XGLeaveExclusiveLock,
        XGEncodeDCC, XGDecodeDCC, XGValidateDescriptor, XGShuffleTextureBufferForDirectStorage,
        XGCreateBVHComputer, XGCreateBVHComputer2,
    ]
}

/// The swizzle mode the console's library picks for a 2D texture with `Layout` 0 / 2: 64KB_D
/// for `Layout` 2, depth 64KB_Z_X, render targets 64KB_R_X, otherwise 4KB_S – or for UAVs
/// 4KB_S_X up to 64 KB of texels, else 64KB_R_X.
pub fn optimal_mode(layout: u32, flags: u32, bits_per_pixel: u64, width: u64, height: u32) -> u32 {
    let (rt, depth, uav) = (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0);
    if depth {
        24
    } else if layout == 2 {
        10
    } else if rt {
        27
    } else if !uav {
        5
    } else if bits_per_pixel * width * height as u64 <= 0x8_0000 {
        21
    } else {
        27
    }
}

/// One tiled surface of a texture (color, depth or stencil) as the console lays it out.
pub struct Surface {
    pub chain: Chain,
    pub mode: u32,
    pub bpe: u32,
    /// pixels per element side (4 for BCn)
    pub side: u32,
    pub width: u32,
    pub height: u32,
    pub array: u32,
    /// depth slices of a 3D texture, 0: a 2D one
    pub depth: u32,
    pub first_tail: usize,
}

/// What a plane of a texture holds.
pub enum PlaneKind {
    Surface(Surface),
    /// compression metadata (CMASK of a render target, HTILE of a depth buffer): only its size
    Meta { bytes: u64, align: u64 },
    /// a plane the library leaves empty (the texels of a BCn render target, the depth of a
    /// depth buffer in a color format)
    Blank,
}

/// A plane of a texture: its usage code in XG_RESOURCE_LAYOUT (1 color, 2 CMASK, 4 HTILE,
/// 5 / 6 depth / stencil, 7 / 8 depth / stencil of a depth buffer) and offset in the resource.
pub struct Plane {
    pub usage: u32,
    pub offset: u64,
    pub kind: PlaneKind,
}

/// A 2D texture as the console lays it out: its planes, and the one the texels are in.
pub struct Texture {
    pub planes: Vec<Plane>,
    pub main: usize,
    pub mips: u32,
    /// a depth buffer in a color format: the library reports a size of 0
    pub empty: bool,
    /// D3D12_RESOURCE_DIMENSION
    dimension: u32,
}

/// (log2 bytes per element, pixels per element side) of a DXGI format; None for the 96-bit
/// and packed 4:2:2 ones (not tiled) and unknown formats. The console's own 116 / 117
/// (R10G10B10_7E3_A2_FLOAT / _6E4_A2_FLOAT) are 32 bits a pixel, like R10G10B10A2.
pub fn element(f: u32) -> Option<(u32, u32)> {
    Some(match f {
        1..=4 => (4, 1),
        9..=22 => (3, 1),
        23..=47 | 67 | 87..=93 | 116 | 117 => (2, 1),
        48..=59 | 85 | 86 | 115 => (1, 1),
        60..=65 => (0, 1),
        70..=72 | 79..=81 => (3, 4),
        73..=78 | 82..=84 | 94..=99 => (4, 4),
        _ => return None,
    })
}

/// 256-byte tile size in elements (log2 bytes per element → (w, h))
fn micro(e: u32) -> (usize, usize) {
    (1 << [4, 4, 3, 3, 2][e as usize], 1 << [4, 3, 3, 2, 2][e as usize])
}

impl Texture {
    pub(crate) fn dimension(&self) -> u32 { self.dimension }

    /// From a D3D12_RESOURCE_DESC (0x38 bytes) as the console's.
    pub fn new(d: &[u8; 0x38]) -> Result<Texture, HRESULT> {
        Self::with_pipes(d, crate::gpu::texture::console_pipes())
    }

    /// `new` for a given pipe / packer configuration (only the XOR modes depend on it).
    pub fn with_pipes(d: &[u8; 0x38], pipes: crate::gpu::tiling::PipeConfig) -> Result<Texture, HRESULT> {
        let u = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
        let w16 = |o: usize| u16::from_le_bytes(d[o..o + 2].try_into().unwrap()) as u32;
        let (dim, width, height, array, mips, format, samples, layout, flags) =
            (u(0), u64::from_le_bytes(d[0x10..0x18].try_into().unwrap()), u(0x18), w16(0x1C), w16(0x1E), u(0x20), u(0x24), u(0x2C), u(0x30));
        // buffers and MSAA are not modelled
        if dim == 4 && samples <= 1 {
            return Self::volume(width, height, array, mips, format, layout, flags, pipes);
        }
        if !matches!(dim, 2 | 3) || samples > 1 {
            return Err(E_NOTIMPL);
        }
        let line = dim == 2;
        if line && (height != 1 || flags & 3 != 0 || !matches!(layout, 0 | 1 | 0x100 | 0x11B)) {
            return Err(E_INVALIDARG);
        }
        if width == 0 || width > 16384 || height == 0 || height > 16384 || array == 0 {
            return Err(E_INVALIDARG);
        }
        // depth + stencil formats: a 4-byte depth plane and a 1-byte stencil plane
        // depth + stencil formats: a 4-byte depth plane (the console's D16_UNORM_S8_UINT family,
        // 118 – 120: a 2-byte one) and a 1-byte stencil plane
        // (its R16_UNORM_X8_TYPELESS view format, 119, the library lays out as a plain 2-byte
        // colour format – and as the depth + stencil pair for a depth buffer)
        let d16s8 = matches!(format, 118 | 120) || (format == 119 && flags & 3 == 2 && mips == 1);
        let planar = matches!(format, 19..=22 | 44..=47) || d16s8;
        let (e, side) = if d16s8 || format == 119 { (1, 1) } else if planar { (2, 1) } else { element(format).ok_or(E_INVALIDARG)? };
        let mode = match layout {
            1 => 0,
            0 if line => 0, // a 1D texture is linear
            0x100..=0x11F => layout & 0x1F,
            // the choice goes by the whole format's size (D32_S8X24: 64 bits a pixel)
            0 | 2 => optimal_mode(layout, flags, if matches!(format, 19..=22) { 64 } else { (8u64 << e) / (side * side) as u64 }, width, height),
            _ => return Err(E_INVALIDARG),
        };
        let full = 32 - (width as u32).max(height).leading_zeros();
        let mips = if mips == 0 { full } else { mips };
        if mips as usize > MAX_MIPS {
            return Err(E_INVALIDARG);
        }
        let surface = |e: u32, side: u32| Surface::new(mode, e, side, width as u32, height, array, mips, pipes);
        let (rt, ds) = (flags & 1 != 0, flags & 2 != 0);
        if planar && layout == 2 && format != 119 {
            return Err(E_INVALIDARG); // a stencil format with the 64KB_D choice
        }
        // a depth buffer (one level: it gets an HTILE) is in the Z modes; in a color format it
        // is left empty. Render target + depth flags: a render target in a Z mode, unless the
        // format has a stencil
        let depth_buffer = ds && mips == 1 && (planar || !rt);
        if depth_buffer && mode != 24 {
            return Err(E_INVALIDARG);
        }
        let depth_format = planar || matches!(format, 39 | 40 | 53 | 55);
        // the texels of a BCn render target / depth buffer of one level are left out too
        let blank = (depth_buffer && !depth_format) || ((rt || ds) && side > 1 && mips == 1);
        // and then the resource is empty – except a render target's CMASK
        let empty = blank && ds;
        // the stencil view format of D16S8 (120): its depth plane takes no room, but in a
        // one-level depth buffer
        let stencil_only = format == 120 && !depth_buffer;
        let main = if blank || stencil_only { PlaneKind::Blank } else { PlaneKind::Surface(surface(e, side)?) };
        let stencil = if planar { Some(surface(0, 1)?) } else { None };
        // metadata planes, per slice: CMASK 1 byte per 128 pixels, HTILE per 16, of the size
        // padded to 1024 × 1024
        let meta = |div: u64| (width.div_ceil(1024) * 1024) * (height.div_ceil(1024) * 1024) as u64 / div * array as u64;
        let mut planes: Vec<(u32, PlaneKind)> = Vec::new();
        let (main_usage, stencil_usage) = if ds { (7, 8) } else if planar { (5, 6) } else { (1, 0) };
        if depth_buffer {
            planes.push((4, PlaneKind::Meta { bytes: meta(16), align: 0x1_0000 }));
        }
        let main_at = planes.len();
        planes.push((if ds && !planar && !depth_buffer { 1 } else { main_usage }, main));
        if let Some(s) = stencil {
            planes.push((stencil_usage, PlaneKind::Surface(s)));
        }
        // a render target of one level gets a CMASK (not with the 64KB_D choice or the _T modes)
        if rt && !depth_buffer && !planar && !empty && mips == 1 && layout != 2 && !(0x110..=0x113).contains(&layout) {
            planes.push((2, PlaneKind::Meta { bytes: meta(128), align: 0x2000 }));
        }
        let mut offset = 0u64;
        let planes = planes
            .into_iter()
            .map(|(usage, kind)| {
                let (bytes, align) = match &kind {
                    PlaneKind::Surface(s) => (s.size(), s.alignment()),
                    PlaneKind::Meta { bytes, align } => (*bytes, *align),
                    PlaneKind::Blank => (0, 0),
                };
                // an empty plane is at 0 and takes no room
                let at = if matches!(kind, PlaneKind::Blank) { 0 } else { offset.div_ceil(align) * align };
                offset = offset.max(at + bytes);
                Plane { usage, offset: at, kind }
            })
            .collect();
        Ok(Texture { planes, main: main_at, mips, empty, dimension: dim })
    }

    /// A 3D texture: one plane of texels, 4KB_S unless the layout names a mode or it is a UAV
    /// (Fortnite's newer library: 4KB_S for UAVs too, in 2D as well).
    #[allow(clippy::too_many_arguments)]
    fn volume(width: u64, height: u32, depth: u32, mips: u32, format: u32, layout: u32, flags: u32, pipes: crate::gpu::tiling::PipeConfig) -> Result<Texture, HRESULT> {
        if width == 0 || width > 2048 || height == 0 || height > 2048 || depth == 0 || depth > 2048 {
            return Err(E_INVALIDARG);
        }
        // render targets / depth: their planes are not modelled
        if flags & 3 != 0 {
            return Err(E_NOTIMPL);
        }
        let (e, side) = element(format).ok_or(E_INVALIDARG)?;
        let mode = match layout {
            1 => 0,
            0x100..=0x11F => layout & 0x1F,
            // UAVs: the 2D choice, by the texels of all the depth slices
            0 if flags & 4 != 0 => optimal_mode(0, flags, (8u64 << e) / (side * side) as u64 * depth as u64, width, height),
            0 => 5,
            2 => return Err(E_NOTIMPL),
            _ => return Err(E_INVALIDARG),
        };
        let full = 32 - (width as u32).max(height).max(depth).leading_zeros();
        let mips = if mips == 0 { full } else { mips };
        if mips as usize > MAX_MIPS {
            return Err(E_INVALIDARG);
        }
        let l = Layout::new_3d(mode, e, pipes).ok_or(E_INVALIDARG)?;
        let chain = Chain::new_3d(l, width as u32, height, depth, side, mips);
        let first_tail = chain.levels.iter().position(|m| m.tail.is_some()).unwrap_or(chain.levels.len());
        let s = Surface { chain, mode, bpe: 1 << e, side, width: width as u32, height, array: 1, depth, first_tail };
        Ok(Texture { planes: vec![Plane { usage: 1, offset: 0, kind: PlaneKind::Surface(s) }], main: 0, mips, empty: false, dimension: 4 })
    }

    pub fn size(&self) -> u64 {
        if self.empty {
            return 0;
        }
        self.planes.iter().map(|p| p.offset + p.bytes()).max().unwrap_or(0)
    }

    pub fn alignment(&self) -> u64 {
        self.planes.iter().map(|p| p.align()).max().unwrap_or(0x100)
    }

    /// the surface of plane `plane` and where it starts (None: metadata or no such plane)
    pub fn plane_surface(&self, plane: u32) -> Option<(&Surface, u64)> {
        let p = self.planes.get(plane as usize)?;
        match &p.kind {
            PlaneKind::Surface(s) => Some((s, p.offset)),
            PlaneKind::Meta { .. } | PlaneKind::Blank => None,
        }
    }

    /// the surface the texels are in (None: the library leaves them out)
    pub fn texels(&self) -> Option<&Surface> {
        self.plane_surface(self.main as u32).map(|(s, _)| s)
    }

    /// byte offset of level `i` of the texels from the start of the resource's first slice
    pub fn mip_offset(&self, i: usize) -> u64 {
        self.plane_surface(self.main as u32).map_or(0, |(s, at)| at + s.mip_offset(i))
    }

    /// byte offset of element (x, y) of level `i` in slice `slice` of the texels
    pub fn element_offset(&self, i: usize, x: usize, y: usize, slice: usize) -> u64 {
        self.plane_surface(self.main as u32).map_or(0, |(s, at)| at + s.element_offset(i, x, y, slice))
    }

    /// XG_RESOURCE_LAYOUT
    pub fn write(&self, b: &mut [u8; LAYOUT_BYTES]) {
        b.fill(0);
        let q = |b: &mut [u8; LAYOUT_BYTES], o: usize, v: u64| b[o..o + 8].copy_from_slice(&v.to_le_bytes());
        let w = |b: &mut [u8; LAYOUT_BYTES], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        q(b, 0, self.size());
        q(b, 8, self.alignment());
        w(b, 0x10, self.mips);
        w(b, 0x14, self.planes.len() as u32);
        w(b, 0x1B38, self.dimension);
        for (k, p) in self.planes.iter().enumerate() {
            let pb = PLANE + k * PLANE_BYTES;
            w(b, pb, p.usage);
            q(b, pb + 0x18, p.offset);
            match &p.kind {
                PlaneKind::Surface(s) => s.write_plane(b, pb),
                PlaneKind::Blank => {}
                PlaneKind::Meta { bytes, align } => {
                    q(b, pb + 8, *bytes);
                    q(b, pb + 0x20, *align);
                    q(b, pb + MIP, *bytes);
                    w(b, pb + MIP + 0x20, *align as u32);
                }
            }
        }
    }
}

impl Plane {
    fn bytes(&self) -> u64 {
        match &self.kind {
            PlaneKind::Surface(s) => s.size(),
            PlaneKind::Meta { bytes, .. } => *bytes,
            PlaneKind::Blank => 0,
        }
    }

    fn align(&self) -> u64 {
        match &self.kind {
            PlaneKind::Surface(s) => s.alignment(),
            PlaneKind::Meta { align, .. } => *align,
            PlaneKind::Blank => 0,
        }
    }
}

impl Surface {
    /// `width` × `height` pixels, `array` slices, `mips` levels of `1 << e`-byte elements in
    /// swizzle mode `mode`.
    #[allow(clippy::too_many_arguments)]
    fn new(mode: u32, e: u32, side: u32, width: u32, height: u32, array: u32, mips: u32, pipes: crate::gpu::tiling::PipeConfig) -> Result<Surface, HRESULT> {
        // the modes the console's library takes for a 2D texture (VAR 28 / 31: not modelled)
        let bc = side > 1;
        match mode {
            0 | 1 | 2 | 5 | 6 | 9 | 10 | 17 | 18 | 21 | 22 | 25 | 26 | 27 => {}
            24 if e <= 3 && !bc => {}
            28 | 31 => return Err(E_NOTIMPL),
            _ => return Err(E_INVALIDARG),
        }
        let l = Layout::new(mode, e, pipes).ok_or(E_INVALIDARG)?;
        let chain = Chain::new(l, width, height, side, mips);
        let first_tail = chain.levels.iter().position(|m| m.tail.is_some()).unwrap_or(chain.levels.len());
        Ok(Surface { chain, mode, bpe: 1 << e, side, width, height, array, depth: 0, first_tail })
    }

    pub fn size(&self) -> u64 {
        let slices = if self.depth > 0 { self.chain.layers() } else { self.array as usize };
        (self.chain.slice_bytes * slices) as u64
    }

    /// depth slices of level `i` (an array's slices) and the same padded to whole blocks
    fn slices(&self, i: usize) -> (u32, u32) {
        if self.depth == 0 {
            return (self.array, self.array);
        }
        let d = (self.depth >> i).max(1);
        if self.mode == 0 {
            return (d, self.depth); // (what the library writes for every linear level)
        }
        if self.chain.layout.block_d == 1 && self.chain.levels.len() > 1 {
            return (d, 1); // thin with mips: the library's sizes are per slice
        }
        // what a level is padded from halves rounding up, as its width and height do
        (d, self.depth.div_ceil(1 << i).next_multiple_of(self.chain.layout.block_d as u32))
    }

    pub fn alignment(&self) -> u64 {
        self.chain.layout.block_bytes as u64
    }

    /// (padded width, padded height) in elements of level `i`
    fn padded(&self, i: usize) -> (usize, usize) {
        let l = &self.chain.layout;
        if i < self.first_tail {
            let h = (self.height.div_ceil(self.side) as usize).div_ceil(1 << i);
            (self.chain.levels[i].pitch, h.div_ceil(l.block_h) * l.block_h)
        } else {
            let (mw, mh) = micro(l.elem_log2);
            let k = i - self.first_tail;
            (((l.block_w / 2) >> k).max(mw), (l.block_h >> k).max(mh))
        }
    }

    /// byte offset of level `i` from the start of a slice
    pub fn mip_offset(&self, i: usize) -> u64 {
        let m = &self.chain.levels[i];
        m.tail.unwrap_or(m.offset) as u64
    }

    /// byte offset of element (x, y) of level `i` in slice `slice`; the slice is z of the 2D
    /// patterns (the XOR modes take pipe bits from it)
    pub fn element_offset(&self, i: usize, x: usize, y: usize, slice: usize) -> u64 {
        let m = &self.chain.levels[i];
        if self.chain.layout.block_d > 1 {
            return self.chain.element_at_z(m, x, y, slice) as u64;
        }
        (slice * self.chain.slice_bytes + m.offset + self.chain.layout.offset_z(x + m.origin.0, y + m.origin.1, slice, m.pitch)) as u64
    }

    /// this surface's plane of XG_RESOURCE_LAYOUT (its header at `pb`)
    fn write_plane(&self, b: &mut [u8; LAYOUT_BYTES], pb: usize) {
        let q = |b: &mut [u8; LAYOUT_BYTES], o: usize, v: u64| b[o..o + 8].copy_from_slice(&v.to_le_bytes());
        let w = |b: &mut [u8; LAYOUT_BYTES], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        let mips = self.chain.levels.len();
        q(b, pb + 8, self.size());
        q(b, pb + 0x10, self.chain.slice_bytes as u64);
        q(b, pb + 0x20, self.alignment());
        w(b, pb + 0x28, self.bpe);
        w(b, pb + 0x2C, self.first_tail as u32);
        w(b, pb + 0x30, (mips - self.first_tail) as u32);
        for i in 0..mips {
            let m = &self.chain.levels[i];
            let r = pb + MIP + i * MIP_BYTES;
            let tail = i >= self.first_tail;
            let (pw, ph) = self.padded(i);
            let bytes = (pw * ph) as u64 * self.bpe as u64;
            let (slices, padded_slices) = self.slices(i);
            q(b, r, if self.depth > 0 { bytes * padded_slices as u64 } else { bytes });
            q(b, r + 8, self.mip_offset(i));
            q(b, r + 0x10, bytes);
            w(b, r + 0x18, pw as u32 * self.side);
            w(b, r + 0x1C, pw as u32 * self.bpe);
            w(b, r + 0x20, if tail { 0x100 } else { self.alignment() as u32 });
            w(b, r + 0x24, pw as u32);
            w(b, r + 0x28, ph as u32);
            w(b, r + 0x2C, padded_slices);
            w(b, r + 0x30, m.width as u32);
            w(b, r + 0x34, m.height as u32);
            w(b, r + 0x38, (self.width >> i).max(1));
            w(b, r + 0x3C, (self.height >> i).max(1));
            w(b, r + 0x40, slices);
            w(b, r + 0x44, 1);
            // the tail is in 256-byte tiles of the same kind (S 1, D 2, R 3; Z keeps its mode)
            w(b, r + 0x48, if tail && !self.mode.is_multiple_of(4) { self.mode % 4 } else { self.mode });
            if !tail && (24..=27).contains(&self.mode) {
                w(b, r + 0x50, 0x8000);
                w(b, r + 0x58, 0x8000);
            }
            w(b, r + 0x60, self.chain.layout.block_d as u32);
            // a volume's tail levels: what the library writes at +8, +0x28, +0x2C and +0x68 is not
            // modelled (their texels are: `element_offset`)
            w(b, r + 0x64, tail as u32);
        }
    }
}

// ---------------------------------------------------------------------------------------
// the computer
// ---------------------------------------------------------------------------------------

#[repr(C)]
struct Computer {
    vtbl: *const usize,
    refs: AtomicU32,
    tex: Texture,
}

fn vtbl() -> &'static [usize] {
    static V: OnceLock<&'static [usize]> = OnceLock::new();
    V.get_or_init(|| {
        let mut v = vec![
            add_ref as usize,
            release as usize,
            get_resource_layout as usize,
            get_size as usize,
            get_alignment as usize,
            get_mip_offset as usize,
            get_texel_offset as usize,
            not_implemented as usize,
            copy_into as usize,
            copy_from as usize,
            get_resource_tiling as usize,
        ];
        // #11 texture-view descriptor getter, #12 IsTiledResource (a bool: the generic stub's
        // E_NOTIMPL would read as true), #13 swizzle-pattern info getter
        for i in v.len()..14 {
            if i == 12 {
                v.push(is_tiled_resource as usize);
                continue;
            }
            let (stub, _) = crate::stubs::make_stub_info("xg_xs.dll", &format!("XGTextureAddressComputer::#{i}"), E_NOTIMPL as u32 as u64);
            v.push(stub);
        }
        Box::leak(v.into_boxed_slice())
    })
}

unsafe fn tex<'a>(p: *mut c_void) -> &'a Texture {
    &(*(p as *const Computer)).tex
}

/// `HRESULT XGCreateTextureComputer(const D3D12_RESOURCE_DESC*, XGTextureAddressComputer**)`
/// `HRESULT XGSuggestSwizzleMode(const D3D12_RESOURCE_DESC*, UINT, D3D12_TEXTURE_LAYOUT* out)`
/// (a newer library's; a retail game passes 0 and its desc's own layout field as `out`): the explicit
/// layout (0x100 | mode) the library takes for `Layout` 0. The choice itself is checked against
/// the console's library (`optimal_mode`), this function is not – none at hand exports it.
pub unsafe extern "system" fn XGSuggestSwizzleMode(desc: *const u8, _what: u32, out: *mut u32) -> HRESULT {
    if desc.is_null() || out.is_null() {
        return E_POINTER;
    }
    let mut d: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    d[0x2C..0x30].fill(0);
    if d[0] == 2 {
        *out = 0x100; // a 1D texture: linear
        return S_OK;
    }
    let r = Texture::new(&d).and_then(|t| t.texels().map(|s| s.mode).ok_or(E_NOTIMPL));
    let mode = match r {
        Ok(m) => m,
        // not modelled (MSAA, 3D render targets): the usual choice by use
        Err(_) => {
            let u = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
            let bits = element(u(0x20)).map_or(32, |(e, _)| 8u64 << e);
            optimal_mode(0, u(0x30), bits, u64::from_le_bytes(d[0x10..0x18].try_into().unwrap()), u(0x18))
        }
    };
    *out = 0x100 | mode;
    S_OK
}

pub unsafe extern "system" fn XGCreateTextureComputer(desc: *const u8, out: *mut *mut c_void) -> HRESULT {
    if desc.is_null() || out.is_null() {
        return E_POINTER;
    }
    *out = std::ptr::null_mut();
    match Texture::new(&*(desc as *const [u8; 0x38])) {
        Ok(tex) => {
            *out = Box::into_raw(Box::new(Computer { vtbl: vtbl().as_ptr(), refs: AtomicU32::new(1), tex })) as *mut c_void;
            S_OK
        }
        Err(hr) => {
            static N: AtomicU32 = AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                let d = &*(desc as *const [u8; 0x38]);
                crate::xlog_warn!("XGCreateTextureComputer {hr:#x}: {:x?}", &d[..0x34]);
            }
            hr
        }
    }
}

/// The older, D3D11-shaped texture descs (u32 fields) of `XGCreateTexture1D/2D/3DComputer` and
/// `XGComputeTexture1D/2D/3DLayout`, as the D3D12-shaped desc `Texture::new` takes.
/// 2D: width, height, mips, array, format, samples, quality, usage, bind, CPU access, misc,
/// swizzle mode, pitch; 1D: without height and the two sample fields; 3D: depth for the array,
/// no sample fields. Bind: render target 0x20, depth 0x40, UAV 0x80; misc 0x4: a cube. The mode
/// is explicit (0 linear, 1–31); a requested pitch (≠ 0) is not modelled.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OldDesc {
    D1,
    D2,
    D3,
}

const MISC_CUBE: u32 = 0x4;

/// D3D11 bind flags → D3D12_RESOURCE_FLAGS (render target, depth-stencil, UAV)
fn resource_flags(bind: u32) -> u32 {
    (bind & 0x20 != 0) as u32 | ((bind & 0x40 != 0) as u32) << 1 | ((bind & 0x80 != 0) as u32) << 2
}

/// A D3D12_RESOURCE_DESC (0x38 bytes) as `Texture::new` reads it.
fn d3d12_desc(dim: u32, width: u64, height: u32, depth_or_array: u32, mips: u32, format: u32, samples: u32, layout: u32, flags: u32) -> Result<[u8; 0x38], HRESULT> {
    let (Ok(depth_or_array), Ok(mips)) = (u16::try_from(depth_or_array), u16::try_from(mips)) else { return Err(E_INVALIDARG) };
    let mut d = [0u8; 0x38];
    d[0..4].copy_from_slice(&dim.to_le_bytes());
    d[0x10..0x18].copy_from_slice(&width.to_le_bytes());
    d[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
    d[0x1C..0x1E].copy_from_slice(&depth_or_array.to_le_bytes());
    d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
    d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
    d[0x24..0x28].copy_from_slice(&samples.max(1).to_le_bytes());
    d[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
    d[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
    Ok(d)
}

unsafe fn old_desc(kind: OldDesc, p: *const u32) -> Result<[u8; 0x38], HRESULT> {
    if p.is_null() {
        return Err(E_POINTER);
    }
    let f = |i: usize| p.add(i).read_unaligned();
    // (dimension, width, height, depth or array, mips, format, samples, bind, misc, mode, pitch)
    let (dim, w, h, n, mips, format, samples, bind, misc, mode, pitch) = match kind {
        OldDesc::D1 => (2, f(0), 1, f(2), f(1), f(3), 1, f(5), f(7), f(8), f(9)),
        OldDesc::D2 => (3, f(0), f(1), f(3), f(2), f(4), f(5), f(8), f(10), f(11), f(12)),
        OldDesc::D3 => (4, f(0), f(1), f(2), f(3), f(4), 1, f(6), f(8), f(9), f(10)),
    };
    if mode > 0x1F {
        return Err(E_INVALIDARG);
    }
    if pitch != 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::xlog_warn!("xg: a texture desc with a requested pitch ({pitch}) is not modelled"));
        return Err(E_NOTIMPL);
    }
    let cube = kind == OldDesc::D2 && misc & MISC_CUBE != 0;
    let n = if cube { n.checked_mul(6).ok_or(E_INVALIDARG)? } else { n };
    d3d12_desc(dim, w as u64, h, n, mips, format, samples, 0x100 | mode, resource_flags(bind))
}

unsafe fn old_desc_computer(kind: OldDesc, desc: *const u32, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    *out = std::ptr::null_mut();
    match old_desc(kind, desc) {
        Ok(d) => XGCreateTextureComputer(d.as_ptr(), out),
        Err(hr) => hr,
    }
}

unsafe fn old_desc_layout(kind: OldDesc, desc: *const u32, out: *mut u8) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    match old_desc(kind, desc).and_then(|d| Texture::new(&d)) {
        Ok(t) => {
            t.write(&mut *(out as *mut [u8; LAYOUT_BYTES]));
            S_OK
        }
        Err(hr) => hr,
    }
}

/// `HRESULT XGCreateTexture1DComputer(const XG_TEXTURE1D_DESC*, XGTextureAddressComputer**)`
pub unsafe extern "system" fn XGCreateTexture1DComputer(desc: *const u32, out: *mut *mut c_void) -> HRESULT {
    old_desc_computer(OldDesc::D1, desc, out)
}

/// `HRESULT XGCreateTexture2DComputer(const XG_TEXTURE2D_DESC*, XGTextureAddressComputer**)`
pub unsafe extern "system" fn XGCreateTexture2DComputer(desc: *const u32, out: *mut *mut c_void) -> HRESULT {
    old_desc_computer(OldDesc::D2, desc, out)
}

/// `HRESULT XGCreateTexture3DComputer(const XG_TEXTURE3D_DESC*, XGTextureAddressComputer**)`
pub unsafe extern "system" fn XGCreateTexture3DComputer(desc: *const u32, out: *mut *mut c_void) -> HRESULT {
    old_desc_computer(OldDesc::D3, desc, out)
}

/// `HRESULT XGComputeTexture1DLayout(const XG_TEXTURE1D_DESC*, XG_RESOURCE_LAYOUT*)`
pub unsafe extern "system" fn XGComputeTexture1DLayout(desc: *const u32, out: *mut u8) -> HRESULT {
    old_desc_layout(OldDesc::D1, desc, out)
}

/// `HRESULT XGComputeTexture2DLayout(const XG_TEXTURE2D_DESC*, XG_RESOURCE_LAYOUT*)`
pub unsafe extern "system" fn XGComputeTexture2DLayout(desc: *const u32, out: *mut u8) -> HRESULT {
    old_desc_layout(OldDesc::D2, desc, out)
}

/// `HRESULT XGComputeTexture3DLayout(const XG_TEXTURE3D_DESC*, XG_RESOURCE_LAYOUT*)`
pub unsafe extern "system" fn XGComputeTexture3DLayout(desc: *const u32, out: *mut u8) -> HRESULT {
    old_desc_layout(OldDesc::D3, desc, out)
}

/// `HRESULT XGComputeBufferLayout(const XG_BUFFER_DESC*, XG_RESOURCE_LAYOUT*)` (desc: byte
/// width, usage, bind, CPU access, misc, structure stride): one linear plane and level of the
/// byte width, 256-byte aligned (the alignment is not checked against the console's library).
pub unsafe extern "system" fn XGComputeBufferLayout(desc: *const u32, out: *mut u8) -> HRESULT {
    if desc.is_null() || out.is_null() {
        return E_POINTER;
    }
    let bytes = desc.read_unaligned();
    let b = &mut *(out as *mut [u8; LAYOUT_BYTES]);
    b.fill(0);
    let q = |b: &mut [u8; LAYOUT_BYTES], o: usize, v: u64| b[o..o + 8].copy_from_slice(&v.to_le_bytes());
    let w = |b: &mut [u8; LAYOUT_BYTES], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let (size, align) = (bytes as u64, 256u64);
    q(b, 0, size);
    q(b, 8, align);
    w(b, 0x10, 1);
    w(b, 0x14, 1);
    w(b, 0x1B38, 1); // D3D12_RESOURCE_DIMENSION_BUFFER
    let (pb, m) = (PLANE, PLANE + MIP);
    w(b, pb, 1);
    q(b, pb + 8, size);
    q(b, pb + 0x10, size);
    q(b, pb + 0x20, align);
    w(b, pb + 0x28, 1);
    q(b, m, size);
    q(b, m + 0x10, size);
    w(b, m + 0x18, bytes);
    w(b, m + 0x1C, bytes);
    w(b, m + 0x20, align as u32);
    for o in [0x24, 0x30, 0x38] {
        w(b, m + o, bytes);
    }
    for o in [0x28, 0x2C, 0x34, 0x3C, 0x40, 0x44, 0x60] {
        w(b, m + o, 1);
    }
    S_OK
}

/// `HRESULT XGCreateTextureComputerFromDescriptor(const UINT32 descriptor[8], computer**)`: a
/// hardware texture descriptor (T#) read with `gpu::desc`; E_INVALIDARG for anything else.
pub unsafe extern "system" fn XGCreateTextureComputerFromDescriptor(words: *const [u32; 8], out: *mut *mut c_void) -> HRESULT {
    if words.is_null() || out.is_null() {
        return E_POINTER;
    }
    *out = std::ptr::null_mut();
    let Some(t) = crate::gpu::desc::image(&words.read_unaligned()) else { return E_INVALIDARG };
    let Some((r, _)) = crate::gpu::desc::image_d3d12(&t) else { return E_INVALIDARG };
    match d3d12_desc(r.Dimension.0 as u32, r.Width, r.Height, r.DepthOrArraySize as u32, r.MipLevels as u32, r.Format.0 as u32, r.SampleDesc.Count, 0x100 | t.swizzle_mode, 0) {
        Ok(d) => XGCreateTextureComputer(d.as_ptr(), out),
        Err(hr) => hr,
    }
}

/// `HRESULT XGComputeTileShape(dimension, format, bytes per element, samples, XG_TILE_SHAPE*
/// texels, XG_TILE_SHAPE* elements)`: D3D12's standard 64 KiB tile shape (outputs optional; the
/// texel shape is zeros for an unknown format).
pub unsafe extern "system" fn XGComputeTileShape(dim: u32, format: u32, bpe: u32, samples: u32, texels: *mut [u32; 3], elements: *mut [u32; 3]) -> HRESULT {
    let Some(shape) = tile_shape(dim, bpe, samples) else { return E_INVALIDARG };
    if !elements.is_null() {
        *elements = shape;
    }
    if !texels.is_null() {
        *texels = match element(format) {
            Some((_, side)) => [shape[0] * side, shape[1] * side, shape[2]],
            None => [0; 3],
        };
    }
    S_OK
}

/// D3D12's standard 64 KiB tile shape in elements (D3D12_RESOURCE_DIMENSION: 2 1D, 3 2D, 4 3D);
/// each MSAA step halves the height, then the width.
fn tile_shape(dim: u32, bpe: u32, samples: u32) -> Option<[u32; 3]> {
    if !bpe.is_power_of_two() || bpe > 16 || !samples.max(1).is_power_of_two() || samples > 16 {
        return None;
    }
    let e = bpe.trailing_zeros() as usize;
    Some(match dim {
        2 => [0x1_0000 / bpe, 1, 1],
        3 => {
            let (mut w, mut h) = ([256, 256, 128, 128, 64][e], [256, 128, 128, 64, 64][e]);
            for k in 0..samples.max(1).trailing_zeros() {
                if k % 2 == 0 { h /= 2 } else { w /= 2 }
            }
            [w, h, 1]
        }
        4 if samples <= 1 => [[64, 32, 32, 32, 16][e], [32, 32, 32, 16, 16][e], [32, 32, 16, 16, 16][e]],
        _ => return None,
    })
}

/// `XG_SWIZZLE_MODE XGComputeOptimalSwizzleMode(dimension, format, width, height, depth or
/// array, samples, bind, misc)`: the mode `XGSuggestSwizzleMode` takes for the same texture.
pub unsafe extern "system" fn XGComputeOptimalSwizzleMode(dim: u32, format: u32, width: u32, height: u32, depth_or_array: u32, samples: u32, bind: u32, misc: u32) -> u32 {
    let n = if dim == 3 && misc & MISC_CUBE != 0 { depth_or_array.saturating_mul(6) } else { depth_or_array };
    let Ok(d) = d3d12_desc(dim, width as u64, height.max(1), n.clamp(1, 0xFFFF), 1, format, samples, 0, resource_flags(bind)) else { return 0 };
    let mut layout = 0u32;
    if XGSuggestSwizzleMode(d.as_ptr(), 0, &mut layout) < 0 {
        return 0;
    }
    layout & 0x1F
}

/// `HRESULT XGComputeOptimalDepthStencilSwizzleMode(format, width, height, depth, samples,
/// compressed, tiled, texture-compatible, XG_SWIZZLE_MODE*)`: 64KB_Z_X, as depth buffers have.
pub unsafe extern "system" fn XGComputeOptimalDepthStencilSwizzleMode(_format: u32, _w: u32, _h: u32, _depth: u32, _samples: u32, _compressed: u32, _tiled: u32, _texture: u32, out: *mut u32) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    *out = 0x18;
    S_OK
}

/// `HRESULT XGValidateDepthStencilSwizzleMode(format, samples, mode)`: the Z modes.
pub unsafe extern "system" fn XGValidateDepthStencilSwizzleMode(_format: u32, _samples: u32, mode: u32) -> HRESULT {
    if matches!(mode, 4 | 8 | 0x10 | 0x14 | 0x18) { S_OK } else { E_FAIL }
}

static INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `HRESULT XGInitializeLibrary()`: S_OK, S_FALSE when it already was.
pub unsafe extern "system" fn XGInitializeLibrary() -> HRESULT {
    if INITIALIZED.swap(true, Ordering::AcqRel) { crate::util::S_FALSE } else { S_OK }
}

/// `HRESULT XGTerminateLibrary()`: S_OK, S_FALSE when it was not initialized.
pub unsafe extern "system" fn XGTerminateLibrary() -> HRESULT {
    if INITIALIZED.swap(false, Ordering::AcqRel) { S_OK } else { crate::util::S_FALSE }
}

static HARDWARE_VERSION: AtomicU32 = AtomicU32::new(u32::MAX);

/// `HRESULT XGSetHardwareVersion(version)` (4 Series S, 5 Series X, 6 devkit): recorded and
/// logged once. The layouts keep the pipe configuration of the console XCL reports.
pub unsafe extern "system" fn XGSetHardwareVersion(version: u32) -> HRESULT {
    if HARDWARE_VERSION.swap(version, Ordering::Relaxed) == u32::MAX {
        let series_s = crate::state::config().is_series_s();
        let differs = if series_s { version != 4 } else { version == 4 };
        crate::xlog_info!(
            "xg: hardware version {version}{}",
            if differs { format!(" – the layouts stay those of the {} XCL reports", if series_s { "Series S" } else { "Series X" }) } else { String::new() }
        );
    }
    S_OK
}

/// `void XGEnableExclusiveLocking(BOOL)` / `void XGLeaveExclusiveLock()`: no locking.
pub unsafe extern "system" fn XGEnableExclusiveLocking(_on: i32) {}

pub unsafe extern "system" fn XGLeaveExclusiveLock() {}

/// `BOOL XGEnterExclusiveLock()`: FALSE, locking is off.
pub unsafe extern "system" fn XGEnterExclusiveLock() -> i32 {
    0
}

/// Entry points no title is known to call: E_NOTIMPL and one log line each. Their arguments are
/// not known well enough to null an output.
macro_rules! xg_not_implemented {
    ($($name:ident),*) => {$(
        pub unsafe extern "system" fn $name() -> HRESULT {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_warn!(concat!("xg: ", stringify!($name), " is not implemented")));
            E_NOTIMPL
        }
    )*};
}

xg_not_implemented!(XGEncodeDCC, XGDecodeDCC, XGValidateDescriptor, XGShuffleTextureBufferForDirectStorage, XGCreateBVHComputer, XGCreateBVHComputer2);

unsafe extern "system" fn add_ref(p: *mut c_void) -> u32 {
    (*(p as *const Computer)).refs.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn release(p: *mut c_void) -> u32 {
    let n = (*(p as *const Computer)).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if n == 0 {
        drop(Box::from_raw(p as *mut Computer));
    }
    n
}

unsafe extern "system" fn get_resource_layout(p: *mut c_void, out: *mut u8) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    tex(p).write(&mut *(out as *mut [u8; LAYOUT_BYTES]));
    S_OK
}

unsafe extern "system" fn get_size(p: *mut c_void) -> u64 {
    tex(p).size()
}

unsafe extern "system" fn get_alignment(p: *mut c_void) -> u64 {
    tex(p).alignment()
}

unsafe extern "system" fn get_mip_offset(p: *mut c_void, plane: u32, mip: u32) -> u64 {
    match tex(p).plane_surface(plane) {
        Some((s, at)) if (mip as usize) < s.chain.levels.len() => at + s.mip_offset(mip as usize),
        _ => 0,
    }
}

unsafe extern "system" fn get_texel_offset(p: *mut c_void, plane: u32, mip: u32, x: u64, y: u32, slice: u32, _sample: u32, bit: *mut u32) -> u64 {
    if !bit.is_null() {
        *bit = 0;
    }
    match tex(p).plane_surface(plane) {
        Some((s, at)) if (mip as usize) < s.chain.levels.len() && slice < s.slices(mip as usize).0 => at + s.element_offset(mip as usize, x as usize, y as usize, slice as usize),
        _ => 0,
    }
}

unsafe extern "system" fn not_implemented(_p: *mut c_void) -> HRESULT {
    E_NOTIMPL
}

/// #12 `bool IsTiledResource()`: never – the computer does not model reserved textures.
unsafe extern "system" fn is_tiled_resource(_p: *mut c_void) -> u8 {
    0
}

/// #10 GetResourceTiling(&tiles, &packed mip info, &tile shape, &count, first subresource,
/// tilings) – D3D12's, for the texture's 64 KiB tiles (as the console's library answers: other
/// swizzle modes E_FAIL; the mip tail is tile 0 of a slice; `count` comes back as the standard
/// – not packed – subresources among the asked ones, written at their index from `first`, the
/// packed ones left as they were). Volumes: not modelled (E_NOTIMPL).
unsafe extern "system" fn get_resource_tiling(p: *mut c_void, tiles: *mut u32, packed: *mut [u32; 3], shape: *mut [u32; 3], count: *mut u32, first: u32, out: *mut [u32; 3]) -> HRESULT {
    let Some(s) = tex(p).texels() else { return E_FAIL };
    if s.depth > 0 {
        return E_NOTIMPL;
    }
    let l = &s.chain.layout;
    if l.block_bytes != TILE_BYTES {
        return E_FAIL;
    }
    let tile = TILE_BYTES as u64;
    let mips = s.chain.levels.len();
    let std_mips = s.first_tail.min(mips);
    if !tiles.is_null() {
        *tiles = s.size().div_ceil(tile) as u32;
    }
    if !packed.is_null() {
        let n_packed = (mips - std_mips) as u32;
        let tail_tiles = if n_packed > 0 { 1 } else { 0 };
        *packed = [std_mips as u32 | n_packed << 8, tail_tiles, 0];
    }
    if !shape.is_null() {
        *shape = [(l.block_w as u32) * s.side, (l.block_h as u32) * s.side, l.block_d as u32];
    }
    if !count.is_null() {
        let asked = *count as usize;
        let subs = mips * s.array as usize;
        let mut written = 0u32;
        for k in 0..asked {
            let sub = first as usize + k;
            if sub >= subs {
                break;
            }
            let (i, slice) = (sub % mips, sub / mips);
            if i >= std_mips {
                continue;
            }
            written += 1;
            if !out.is_null() {
                let (pw, ph) = s.padded(i);
                let start = (slice as u64 * s.chain.slice_bytes as u64 + s.mip_offset(i)) / tile;
                *out.add(k) = [pw.div_ceil(l.block_w) as u32, ph.div_ceil(l.block_h) as u32 | 1 << 16, start as u32];
            }
        }
        *count = written;
    }
    S_OK
}

/// bytes of a tiled resource's tile
const TILE_BYTES: usize = 0x1_0000;

/// (surface, its offset, level, slice) of a subresource (mip + slice × mips) of a plane
fn subresource(t: &Texture, plane: u32, sub: u32) -> Option<(&Surface, u64, usize, usize)> {
    let (s, at) = t.plane_surface(plane)?;
    let mips = s.chain.levels.len();
    let (i, k) = (sub as usize % mips, sub as usize / mips);
    (k < s.array as usize).then_some((s, at, i, k))
}

/// The arguments after `subresource` of CopyIntoSubresource / CopyFromSubresource: (pointer, row
/// pitch, slice pitch). The library XDefiant and Fortnite ship takes (pointer, row pitch, slice
/// pitch); a newer one (a retail game's) has a 32-bit argument before the pointer – passed as 0,
/// stored as a dword, so the slot's upper half is whatever the stack held. Read the older way,
/// that call gave the low half of a pointer as the row pitch and the copy wrote rows gigabytes
/// past the destination. Told apart per call: a small value there with a readable pointer after
/// it is the newer form.
unsafe fn copy_args(a4: usize, a5: usize, a6: usize, a7: usize) -> (*const u8, u32, u32) {
    // (address space, not readable memory: the title's pages are committed on first use)
    let in_use = |a: usize| {
        use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_FREE};
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        a >= 0x1_0000 && VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0 && m.State != MEM_FREE
    };
    let newer = a4 as u32 <= 0xFFFF && in_use(a5);
    // (a retail game calls both forms)
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let bit = 1 << newer as u32;
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        crate::xlog_info!("xg: CopyInto/FromSubresource called in the {} form", if newer { "newer 7-argument" } else { "6-argument" });
    }
    if newer {
        if a4 as u32 != 0 {
            static EXTRA: std::sync::Once = std::sync::Once::new();
            EXTRA.call_once(|| crate::xlog_warn!("xg: CopyInto/FromSubresource with {:#x} in its added argument (meaning unknown, ignored)", a4 as u32));
        }
        (a5 as *const u8, a6 as u32, a7 as u32)
    } else {
        (a4 as *const u8, a5 as u32, a6 as u32)
    }
}

/// rows of `src` (`row_pitch` apart) → the level's place in `dst` (the whole resource)
unsafe extern "system" fn copy_into(p: *mut c_void, dst: *mut u8, plane: u32, sub: u32, a4: usize, a5: usize, a6: usize, a7: usize) -> HRESULT {
    let (src, row_pitch, slice_pitch) = copy_args(a4, a5, a6, a7);
    let Some((t, at, i, k)) = subresource(tex(p), plane, sub).filter(|_| !dst.is_null() && !src.is_null()) else { return E_INVALIDARG };
    let m = t.chain.levels[i];
    let row = m.width * t.bpe as usize;
    if t.depth > 0 {
        let mut linear = vec![0u8; row * m.height * m.depth];
        for (n, to) in linear.chunks_exact_mut(row).enumerate() {
            let from = n / m.height * slice_pitch as usize + n % m.height * row_pitch as usize;
            to.copy_from_slice(std::slice::from_raw_parts(src.add(from), row));
        }
        t.chain.tile_volume(std::slice::from_raw_parts_mut(dst.add(at as usize), t.size() as usize), i, &linear);
        return S_OK;
    }
    let mut linear = vec![0u8; row * m.height];
    for y in 0..m.height {
        linear[y * row..(y + 1) * row].copy_from_slice(std::slice::from_raw_parts(src.add(y * row_pitch as usize), row));
    }
    let slice = std::slice::from_raw_parts_mut(dst.add(at as usize + k * t.chain.slice_bytes), t.chain.slice_bytes);
    t.chain.tile_level(slice, i, k, &linear);
    S_OK
}

/// the level's place in `src` (the whole resource) → rows of `dst` (`row_pitch` apart)
unsafe extern "system" fn copy_from(p: *mut c_void, dst: *mut u8, plane: u32, sub: u32, a4: usize, a5: usize, a6: usize, a7: usize) -> HRESULT {
    let (src, row_pitch, slice_pitch) = copy_args(a4, a5, a6, a7);
    let Some((t, at, i, k)) = subresource(tex(p), plane, sub).filter(|_| !dst.is_null() && !src.is_null()) else { return E_INVALIDARG };
    let m = t.chain.levels[i];
    let row = m.width * t.bpe as usize;
    if t.depth > 0 {
        let linear = t.chain.detile_volume(std::slice::from_raw_parts(src.add(at as usize), t.size() as usize), i);
        for (n, from) in linear.chunks_exact(row).enumerate() {
            let to = n / m.height * slice_pitch as usize + n % m.height * row_pitch as usize;
            std::ptr::copy_nonoverlapping(from.as_ptr(), dst.add(to), row);
        }
        return S_OK;
    }
    let slice = std::slice::from_raw_parts(src.add(at as usize + k * t.chain.slice_bytes), t.chain.slice_bytes);
    let linear = t.chain.detile_level(slice, i, k);
    for y in 0..m.height {
        std::ptr::copy_nonoverlapping(linear[y * row..].as_ptr(), dst.add(y * row_pitch as usize), row);
    }
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(width: u64, height: u32, array: u16, mips: u16, format: u32, layout: u32) -> [u8; 0x38] {
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&3u32.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&width.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&array.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
        d
    }

    /// CopyInto / CopyFromSubresource in both forms: the 6-argument one, and the newer one with a
    /// 32-bit argument before the pointer whose stack slot's upper half holds garbage – the same
    /// bytes tiled and read back, the pitch not taken from the pointer.
    #[test]
    fn subresource_copies_take_both_argument_forms() {
        type Six = unsafe extern "system" fn(*mut c_void, *mut u8, u32, u32, *const u8, u32, u32) -> HRESULT;
        type Seven = unsafe extern "system" fn(*mut c_void, *mut u8, u32, u32, usize, *const u8, u32, u32) -> HRESULT;
        let d = desc(64, 32, 1, 1, 28, 0x109);
        let mut c: *mut c_void = std::ptr::null_mut();
        unsafe {
            assert!(XGCreateTextureComputer(d.as_ptr(), &mut c) >= 0);
            let t = Texture::new(&d).unwrap();
            let size = t.size() as usize;
            let pitch = 64 * 4 + 32;
            let lin: Vec<u8> = (0..pitch * 32).map(|i| (i * 7 + i / 251) as u8).collect();
            let slot = |k: usize| *(*(c as *const *const usize)).add(k);
            let (into6, from6): (Six, Six) = (std::mem::transmute(slot(8)), std::mem::transmute(slot(9)));
            let (into7, from7): (Seven, Seven) = (std::mem::transmute(slot(8)), std::mem::transmute(slot(9)));
            let (mut a, mut b) = (vec![0u8; size], vec![0u8; size]);
            assert_eq!(into6(c, a.as_mut_ptr(), 0, 0, lin.as_ptr(), pitch as u32, 0), S_OK);
            let garbage = 0x0000_0016_0000_0000usize; // a dword 0 under an old stack value
            assert_eq!(into7(c, b.as_mut_ptr(), 0, 0, garbage, lin.as_ptr(), pitch as u32, 0), S_OK);
            assert_eq!(a, b, "both forms tile the same bytes");
            let (mut back6, mut back7) = (vec![0u8; pitch * 32], vec![0u8; pitch * 32]);
            assert_eq!(from6(c, back6.as_mut_ptr(), 0, 0, a.as_ptr(), pitch as u32, 0), S_OK);
            assert_eq!(from7(c, back7.as_mut_ptr(), 0, 0, garbage, a.as_ptr(), pitch as u32, 0), S_OK);
            for y in 0..32 {
                let row = y * pitch..y * pitch + 64 * 4;
                assert_eq!(back6[row.clone()], lin[row.clone()], "6-argument form, row {y}");
                assert_eq!(back7[row.clone()], lin[row], "7-argument form, row {y}");
            }
            release(c);
        }
    }

    fn layout_of_computer(c: *mut c_void) -> Vec<u8> {
        let mut b = vec![0xCDu8; LAYOUT_BYTES];
        unsafe {
            assert_eq!(get_resource_layout(c, b.as_mut_ptr()), S_OK);
            release(c);
        }
        b
    }

    /// XG_TEXTURE2D_DESC: width, height, mips, array, format, samples, quality, usage, bind,
    /// CPU access, misc, swizzle mode, pitch
    fn desc_2d(width: u32, height: u32, mips: u32, array: u32, format: u32, bind: u32, misc: u32, mode: u32) -> [u32; 13] {
        [width, height, mips, array, format, 1, 0, 0, bind, 0, misc, mode, 0]
    }

    #[test]
    fn older_2d_desc_computers_and_layouts_match_the_d3d12_desc_ones() {
        unsafe {
            for (bind, mode, flags) in [(0, 9, 0u32), (0x20, 0x1B, 1), (0x80, 0x15, 4), (0, 0, 0)] {
                let old = desc_2d(300, 200, 0, 3, 28, bind, 0, mode);
                let mut a = std::ptr::null_mut();
                assert_eq!(XGCreateTexture2DComputer(old.as_ptr(), &mut a), S_OK);
                let mut d = desc(300, 200, 3, 0, 28, 0x100 | mode);
                d[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
                let mut b = std::ptr::null_mut();
                assert_eq!(XGCreateTextureComputer(d.as_ptr(), &mut b), S_OK);
                let la = layout_of_computer(a);
                assert_eq!(la, layout_of_computer(b), "bind {bind:#x} mode {mode:#x}");
                let mut lc = vec![0u8; LAYOUT_BYTES];
                assert_eq!(XGComputeTexture2DLayout(old.as_ptr(), lc.as_mut_ptr()), S_OK);
                assert_eq!(lc, la);
            }
        }
    }

    #[test]
    fn older_desc_cubes_1d_and_3d_and_bad_modes() {
        let size = |l: &[u8]| u64::from_le_bytes(l[0..8].try_into().unwrap());
        unsafe {
            let mut one = vec![0u8; LAYOUT_BYTES];
            let mut cube = vec![0u8; LAYOUT_BYTES];
            assert_eq!(XGComputeTexture2DLayout(desc_2d(64, 64, 1, 1, 28, 0, 0, 9).as_ptr(), one.as_mut_ptr()), S_OK);
            assert_eq!(XGComputeTexture2DLayout(desc_2d(64, 64, 1, 1, 28, 0, MISC_CUBE, 9).as_ptr(), cube.as_mut_ptr()), S_OK);
            let mut six = vec![0u8; LAYOUT_BYTES];
            let mut d = desc(64, 64, 6, 1, 28, 0x109);
            d[0x30..0x34].fill(0);
            Texture::new(&d).unwrap().write(&mut *(six.as_mut_ptr() as *mut [u8; LAYOUT_BYTES]));
            assert_eq!(cube, six, "a cube is six slices");
            assert_eq!(size(&cube), 6 * size(&one));

            let mut c = std::ptr::null_mut::<c_void>();
            assert_eq!(XGCreateTexture2DComputer(desc_2d(64, 64, 1, 1, 28, 0, 0, 0xFFFF_FFFF).as_ptr(), &mut c), E_INVALIDARG);
            assert!(c.is_null());
            let mut pitched = desc_2d(64, 64, 1, 1, 28, 0, 0, 0);
            pitched[12] = 64;
            assert_eq!(XGCreateTexture2DComputer(pitched.as_ptr(), &mut c), E_NOTIMPL);

            // 1D: width, mips, array, format, usage, bind, CPU access, misc, mode, pitch
            let line = [256u32, 1, 1, 28, 0, 0, 0, 0, 0, 0];
            assert_eq!(XGCreateTexture1DComputer(line.as_ptr(), &mut c), S_OK);
            assert_eq!(get_size(c), 1024);
            release(c);
            // 3D: width, height, depth, mips, format, usage, bind, CPU access, misc, mode, pitch
            let vol = [32u32, 32, 8, 1, 28, 0, 0, 0, 0, 9, 0];
            let mut l3 = vec![0u8; LAYOUT_BYTES];
            assert_eq!(XGComputeTexture3DLayout(vol.as_ptr(), l3.as_mut_ptr()), S_OK);
            let mut d3 = desc(32, 32, 8, 1, 28, 0x109);
            d3[0..4].copy_from_slice(&4u32.to_le_bytes());
            let mut want = vec![0u8; LAYOUT_BYTES];
            Texture::new(&d3).unwrap().write(&mut *(want.as_mut_ptr() as *mut [u8; LAYOUT_BYTES]));
            assert_eq!(l3, want);
        }
    }

    #[test]
    fn optimal_swizzle_mode_is_the_library_choice() {
        unsafe {
            // (bind, D3D12 flags): plain, render target, depth, small and large UAV
            for (fmt, w, h, bind, flags) in [(28, 256, 256, 0, 0), (28, 256, 256, 0x20, 1), (40, 256, 256, 0x40, 2), (28, 64, 64, 0x80, 4), (28, 1024, 1024, 0x80, 4)] {
                let bits = 8u64 << element(fmt).unwrap().0;
                assert_eq!(XGComputeOptimalSwizzleMode(3, fmt, w, h, 1, 1, bind, 0), optimal_mode(0, flags, bits, w as u64, h), "bind {bind:#x} {w}x{h}");
            }
            let mut m = 0;
            assert_eq!(XGComputeOptimalDepthStencilSwizzleMode(40, 64, 64, 1, 1, 1, 1, 0, &mut m), S_OK);
            assert_eq!(m, 0x18);
            assert_eq!(XGValidateDepthStencilSwizzleMode(40, 1, 0x18), S_OK);
            assert_eq!(XGValidateDepthStencilSwizzleMode(40, 1, 0x1B), E_FAIL);
        }
    }

    #[test]
    fn tile_shapes_are_d3d12s_standard_ones() {
        let mut t = [9u32; 3];
        let mut e = [9u32; 3];
        unsafe {
            assert_eq!(XGComputeTileShape(3, 28, 4, 1, &mut t, &mut e), S_OK);
            assert_eq!((t, e), ([128, 128, 1], [128, 128, 1]));
            assert_eq!(XGComputeTileShape(3, 71, 8, 1, &mut t, &mut e), S_OK); // BC1: 4x4 texels an element
            assert_eq!((t, e), ([512, 256, 1], [128, 64, 1]));
            assert_eq!(XGComputeTileShape(3, 28, 4, 4, &mut t, std::ptr::null_mut()), S_OK);
            assert_eq!(t, [64, 64, 1]);
            assert_eq!(XGComputeTileShape(4, 0, 1, 1, &mut t, &mut e), S_OK);
            assert_eq!((t, e), ([0; 3], [64, 32, 32]));
            assert_eq!(XGComputeTileShape(3, 28, 3, 1, &mut t, &mut e), E_INVALIDARG);
        }
    }

    #[test]
    fn a_texture_descriptor_gives_the_computer_of_its_texture() {
        use crate::gpu::desc::{gfx10_format, xcl_image_for_view, ImageType};
        let g = gfx10_format(windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28)).unwrap();
        let words = xcl_image_for_view(0, g, ImageType::D2, 100, 50, 1, 3);
        unsafe {
            let mut a = std::ptr::null_mut();
            assert_eq!(XGCreateTextureComputerFromDescriptor(&words, &mut a), S_OK);
            let mut b = std::ptr::null_mut();
            assert_eq!(XGCreateTextureComputer(desc(100, 50, 1, 3, 28, 0x100 | ((words[3] >> 20) & 0x1F)).as_ptr(), &mut b), S_OK);
            assert_eq!(layout_of_computer(a), layout_of_computer(b));
            assert_eq!(XGCreateTextureComputerFromDescriptor(&[0u32; 8], &mut a), E_INVALIDARG);
            assert!(a.is_null());
        }
    }

    #[test]
    fn library_state_and_buffer_layout() {
        unsafe {
            XGTerminateLibrary();
            assert_eq!(XGInitializeLibrary(), S_OK);
            assert_eq!(XGInitializeLibrary(), crate::util::S_FALSE);
            assert_eq!(XGTerminateLibrary(), S_OK);
            assert_eq!(XGEnterExclusiveLock(), 0);
            assert_eq!(XGEncodeDCC(), E_NOTIMPL);
            let mut l = vec![0xCDu8; LAYOUT_BYTES];
            assert_eq!(XGComputeBufferLayout([1000u32, 0, 0, 0, 0, 0].as_ptr(), l.as_mut_ptr()), S_OK);
            assert_eq!(u64::from_le_bytes(l[0..8].try_into().unwrap()), 1000);
            assert_eq!(u64::from_le_bytes(l[8..16].try_into().unwrap()), 256);
            assert_eq!(u32::from_le_bytes(l[0x10..0x14].try_into().unwrap()), 1);
            assert_eq!(u32::from_le_bytes(l[0x1B38..0x1B3C].try_into().unwrap()), 1);
        }
    }

    #[test]
    fn computer_has_all_14_methods_and_is_not_a_tiled_resource() {
        let v = vtbl();
        assert_eq!(v.len(), 14);
        let mut out = std::ptr::null_mut();
        unsafe {
            assert_eq!(XGCreateTextureComputer(desc(64, 64, 1, 1, 28, 0x109).as_ptr(), &mut out), S_OK);
            let f: unsafe extern "system" fn(*mut c_void) -> u8 = std::mem::transmute(v[12]);
            assert_eq!(f(out), 0);
            release(out);
        }
    }

    /// As the console's library answers (seen on it): RGBA8 256×256, full chain, 64KB_S.
    #[test]
    fn rgba8_64kb_s_layout() {
        let t = Texture::new(&desc(256, 256, 1, 0, 28, 0x109)).unwrap();
        let mut b = [0u8; LAYOUT_BYTES];
        t.write(&mut b);
        let q = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let w = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        assert_eq!((q(0), q(8), w(0x10), w(PLANE + 0x2C), w(PLANE + 0x30)), (0x60000, 0x10000, 9, 2, 7));
        let mip = |i: usize, k: usize| w(PLANE + MIP + i * MIP_BYTES + 4 * k);
        let offsets: Vec<u32> = (0..9).map(|i| mip(i, 2)).collect();
        assert_eq!(offsets, [0x20000, 0x10000, 0x8000, 0x4000, 0x2000, 0x1000, 0x800, 0x600, 0x500]);
        assert_eq!((mip(2, 0), mip(2, 9), mip(2, 10), mip(2, 18), mip(6, 9), mip(6, 10)), (0x8000, 0x40, 0x80, 1, 8, 8));
        assert_eq!(optimal_mode(0, 0, 32, 256, 256), 5);
    }

    fn usages_and_bytes(t: &Texture) -> Vec<(u32, u32)> {
        t.planes.iter().map(|p| (p.usage, if let PlaneKind::Surface(s) = &p.kind { s.bpe } else { 0 })).collect()
    }

    /// The console's D16_UNORM_S8_UINT (118): a 2-byte depth plane and a 1-byte stencil plane,
    /// after the HTILE in a depth buffer – a real size, which the PC's allocation info had not
    /// (u64::MAX, and the title stopped on its assertion).
    #[test]
    fn d16s8_is_planar_depth_and_stencil() {
        let mut x = desc(1920, 1080, 1, 1, 118, 0);
        x[0x30..0x34].copy_from_slice(&2u32.to_le_bytes()); // ALLOW_DEPTH_STENCIL
        let t = Texture::new(&x).unwrap();
        assert_eq!(usages_and_bytes(&t), [(4, 0), (7, 2), (8, 1)]);
        let end = t.planes.iter().map(|p| p.offset + p.bytes()).max().unwrap();
        assert!(t.planes.windows(2).all(|w| w[0].offset + w[0].bytes() <= w[1].offset));
        assert_eq!(t.size(), end);
        assert!(t.size() >= 1920 * 1080 * 3 && t.size() < 1920 * 1080 * 8, "{:#x}", t.size());
        assert_eq!(t.alignment(), 0x1_0000);
        // the same format, not a depth buffer: depth and stencil planes only
        let t = Texture::new(&desc(512, 512, 1, 1, 118, 0)).unwrap();
        assert_eq!(usages_and_bytes(&t), [(5, 2), (6, 1)]);
        assert!(t.size() >= 512 * 512 * 3 && t.size() != u64::MAX);
    }

    /// The console's R10G10B10_7E3_A2_FLOAT / _6E4_A2_FLOAT (116 / 117): 32 bits a pixel, laid
    /// out as R10G10B10A2 (24).
    #[test]
    fn console_floats_are_32_bits() {
        let rgb10a2 = Texture::new(&desc(1920, 1080, 1, 1, 24, 0)).unwrap();
        for f in [116, 117] {
            let t = Texture::new(&desc(1920, 1080, 1, 1, f, 0)).unwrap();
            assert_eq!(usages_and_bytes(&t), [(1, 4)]);
            assert_eq!((t.size(), t.alignment()), (rgb10a2.size(), rgb10a2.alignment()));
        }
    }
}
