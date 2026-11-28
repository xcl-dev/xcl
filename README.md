<p align="center">
  <img src="assets/readme/banner.svg" width="100%" alt="XCL – console games you own, running natively on Windows: no CPU emulation, RDNA2 shaders recompiled to DXIL, Direct3D 12">
</p>

<p align="center">
  <a href="https://discord.gg/xcl"><img alt="Discord: join the server" src="https://img.shields.io/badge/Discord-join%20the%20server-7c5cff?style=for-the-badge"></a>
  <a href="https://github.com/xcl-dev/xcl/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/xcl-dev/xcl/ci.yml?branch=main&style=for-the-badge&label=CI"></a>
  <a href="LICENSE"><img alt="License: GPL-3.0-or-later" src="https://img.shields.io/badge/license-GPL--3.0--or--later-7c5cff?style=for-the-badge"></a>
  <img alt="Windows 10 / 11 x64" src="https://img.shields.io/badge/Windows-10%20%2F%2011%20x64-2f80ed?style=for-the-badge">
  <img alt="Direct3D 12, Shader Model 6.6" src="https://img.shields.io/badge/D3D12-SM%206.6-1fb9d8?style=for-the-badge">
  <img alt="Rust" src="https://img.shields.io/badge/Rust-stable--msvc-1fb9d8?style=for-the-badge">
</p>

<p align="center">
  <b><a href="#games">Games</a></b> ·
  <b><a href="#highlights">Highlights</a></b> ·
  <b><a href="#how-it-works">How it works</a></b> ·
  <b><a href="#quick-start">Quick start</a></b> ·
  <b><a href="#what-works">What works</a></b> ·
  <b><a href="#controls">Controls</a></b> ·
  <b><a href="#building-and-usage">Building and usage</a></b> ·
  <b><a href="#tests">Tests</a></b> ·
  <b><a href="#contributing">Contributing</a></b>
</p>

# XCL – Xbox Series X|S → Windows compatibility layer

XCL is a compatibility layer that runs Xbox Series X|S (GDK) games you own on a Windows PC - in the spirit of Wine/Proton. There is no CPU emulation: the game's x64 code runs natively, and XCL provides its own, independently written implementations of the console system libraries the PC does not have (`kernelx.dll`, XGameRuntime, `d3d12_xs.dll`, …). The console's GPU code (RDNA2 machine code) is recompiled into PC shaders (HLSL → DXIL).

> [!IMPORTANT]
> XCL works only on unpacked game files you already own. It does not download, decrypt or unpack game packages (XVC / MSIXVC) and contains nothing for circumventing DRM. The repository holds no code, headers or binaries from the console SDK, no keys and no game data; the Windows components XCL relies on (XGameRuntime, XAudio2, D3D12, DXC) come from your own system or the public Windows SDK and are not distributed with XCL.

<p align="center"><img src="assets/readme/divider.svg" width="100%" alt=""></p>

## Games

<p align="center">
  <img src="assets/readme/status.svg" width="100%" alt="A newer retail game, about 60 % of the way to playable (estimate): ~4 s to its first frame, 30 fps in its menus, ~5 fps in its 3D world, 100 % of its shaders translate">
</p>

| Game | State | What works | Limits |
|---|---|---|---|
| **Minecraft Preview**<br><sub>`Microsoft.MinecraftBeta` 1.18.3064.0</sub> | ✅&nbsp;playable | fully playable | – |
| **Unicorn Overlord**<br><sub>`SEGAofAmericaInc.Pa253b0c1Y` 1.0.0.6</sub> | ✅&nbsp;playable | fully playable | – |
| **Fortnite**<br><sub>`436609B6.FortniteClient` 2.0.1461.7919</sub> | ✅&nbsp;works | works fully, as far as the game allows without its online services | the game itself is online-only (multiplayer servers) |
| **XDefiant**<br><sub>`Ubisoft.XDefiant-Game` 1.0.0.186</sub> | ✅&nbsp;works | works fully, as far as the game allows without its servers | the game's own servers have been shut down |
| **A newer retail game** | 🚧&nbsp;**≈&nbsp;60&nbsp;%**<br><sub>estimate</sub> | starts in ~4 s, runs at a steady 30 fps without a crash, shows its menus and goes on past the first one to its settings; fits a 32 GB machine (~15 GB committed in the menus). With the NVIDIA integration branch ([#2057](https://github.com/xcl-dev/xcl/pull/2057)) it goes on into its 3D world: **~5 fps there** on an RTX 5090 (14 fps on average over a whole 5-minute run, its menus and loading included) | not playable yet: the picture of the world is incomplete (its minimap and other mesh pipelines are refused), and it does not keep running there for long |

## Highlights

<table>
  <tr>
    <td width="33%" valign="top" align="center">
      <img src="assets/readme/icon-native.svg" width="56" alt=""><br>
      <p align="left"><b>No CPU emulation</b> – the game's x64 code runs natively, in the spirit of Wine / Proton.</p>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/readme/icon-shader.svg" width="56" alt=""><br>
      <p align="left"><b>100 % of a retail game's shaders translate</b> – all 26 698 RDNA2 programs (26 503 records) of its shader pack become HLSL; random samples compile to DXIL with 0 errors.</p>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/readme/icon-dll.svg" width="56" alt=""><br>
      <p align="left"><b>The game's own console DLLs run on the PC</b> – its Bink 2 video codec decodes the game's movies, its ray-tracing BVH builder builds acceleration structures on a PC GPU, the AMD FSR 4 console upscaler runs a whole frame on a PC GPU, the GDK HTTP library initialises.</p>
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top" align="center">
      <img src="assets/readme/icon-texture.svg" width="56" alt=""><br>
      <p align="left"><b>Console texture layouts</b> – the archive test reads and detiles all 912 entries in the retail sample, including cubemaps and volume textures.</p>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/readme/icon-storage.svg" width="56" alt=""><br>
      <p align="left"><b>The console's storage queues</b> – DirectStorage with file and memory sources, zlib and BCPack (BC1, BC3 – BC7).</p>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/readme/icon-test.svg" width="56" alt=""><br>
      <p align="left"><b>370+ automated tests</b>, including GPU tests that compare translated shaders against expected results instruction by instruction.</p>
    </td>
  </tr>
</table>

## How it works

<p align="center">
  <img src="assets/readme/how.svg" width="100%" alt="How XCL works: the game's x64 code goes through the PE loader to the CPU and calls the console system libraries, which XCL implements; its RDNA2 shaders go through the shader recompiler to the Direct3D 12 driver and the GPU; its textures, sound and video go through XCL's memory, textures and DirectStorage">
</p>

The game's code runs on the CPU as it is; each call into a console library lands in XCL's own
implementation of it, and each shader is translated once and cached. The details:
[`docs/specs/2026-11-20-xcl-design.md`](docs/specs/2026-11-20-xcl-design.md), status and
measurements: [`HANDOFF.md`](HANDOFF.md).

## Quick start

1. **Build** – Windows 10 / 11 x64, Rust and the Windows SDK ([all requirements](#building-and-usage)):

   ```powershell
   cargo build --release --workspace
   ```

2. **Look at your game** – its unpacked files, the folder that holds `MicrosoftGame.config`:

   ```powershell
   target\release\xcl.exe inspect D:\Game\game.exe
   ```

3. **Run it** – or open `xcl.exe` from Explorer and choose the game:

   ```powershell
   target\release\xcl.exe run D:\Game
   ```

A gamepad is recommended ([Controls](#controls)). The game folder is never modified.

<p align="center"><img src="assets/readme/divider.svg" width="100%" alt=""></p>

## What works

✅ done and verified

<details>
<summary><b>Running the game's code</b> – PE loader, kernelx, Memory, File system, DirectStorage</summary>

| Feature | State | What it does | Verified by |
|---|---|---|---|
| **PE loader** | ✅ | relocations, imports by name / ordinal / delay-load, TLS callbacks, exception tables, game DLLs, data-driven import routing (HLE / Windows / stub) | the game's 16 console DLLs load with 0 unresolved imports |
| **kernelx** | ✅ | title memory (budget from MicrosoftGame.Config, physical pages), threads on the PC's processors, modules, OS / console info | unit + end-to-end tests |
| **Memory** | ✅ | title graphics memory shared by CPU and GPU without copies, up to 16 GB (4 GiB windows) | GPU tests: loads, stores and atomics across windows |
| **File system** | ✅ | console paths (`GAME:`, `G:`, `D:` …) → the game folder; writes go to an overlay, the game folder is never modified; the console's drives: `S:\` (persistent local storage) and `T:\` (temp) in the title's data folder; free disk space answered for console paths | end-to-end: the test title's folder stays byte-identical; PLS and free space through the real call path |
| **DirectStorage** | ✅ | the console's DirectStorage queues: file and memory sources, queued event notifications, zlib, and BCPack – the console's texture compression – for BC1, BC3, BC4, BC5, BC6H and BC7 (BC2 and the hardware swizzle are refused) | unit tests on synthetic streams; retail BCPack chunks of all six modes byte-identical with a reference decoder (`XCL_SAMPLE_BCPACK`) |

</details>

<details>
<summary><b>System services</b> – XGameRuntime, Input, Audio</summary>

| Feature | State | What it does | Verified by |
|---|---|---|---|
| **XGameRuntime** | ✅ | the GDK thunk protocol real titles use, laid out from a table of all 396 GDK runtime functions; offline XUser, XGameSave, persistent local storage, XPackage, XGame (title id), XSystem – the title sees an Xbox Series X / S; task queues, async, networking and dialogs through the PC's XGameRuntime | the game's libHttpClient initialises; users, saves and storage through the real call path; slot layout cross-checked against the game's own thunks |
| **Input** | ✅ | GameInput: one gamepad – the PC's XInput controller or a PlayStation controller, and keyboard and mouse as a gamepad (see [Controls](#controls)) | unit tests; an Xbox controller in Minecraft; a DualShock 4 (USB) |
| **Audio** | ✅ | XAudio2 → system XAudio2.9 (PCM / float voices; Opus 0x704F decoded via `MSOpusDecoder.dll`; XMA1 / XMA2 decoded in pure Rust) | unit + end-to-end tests |

</details>

<details>
<summary><b>Graphics</b> – Shader recompiler, Pipelines, D3D12.X compute, D3D12.X graphics, Textures, Ray tracing, Upscaling</summary>

| Feature | State | What it does | Verified by |
|---|---|---|---|
| **Shader recompiler** | ✅ | RDNA2 → HLSL → DXIL: VS, PS, CS, mesh (NGG), GS; atomics, LDS, scratch, images, samplers incl. static ones, wave ops, ray-tracing instructions; structured control flow; parallel DXC, disk cache | 26 698 / 26 698 programs of the retail pack; 0 DXC errors on random samples; per-instruction GPU tests |
| **Pipelines** | ✅ | PSOs from console shaders: VS+PS, CS, MS+PS, VS+GS+PS; root signature with the shaders' static samplers; non-blocking creation – a pipeline whose shaders are not compiled yet gets a stand-in at once and is built in the background | GPU tests drawing the game's own VS / PS pairs, also built in the background |
| **D3D12.X compute** | ✅ | console device slots, console descriptor heaps (T# / V# / S#), buffers at their console addresses, compute command lists recorded in the console format and replayed (tables, barriers, clears, copies) | the game's BVH builder and FSR 4 record and run through XCL |
| **D3D12.X graphics** | ✅ | device, resources and heaps (also placed in title memory), pipelines; the title's graphics command lists – draws, indirect draws, graphics root arguments, barriers, copies, clears – and command lists recorded by console code, replayed; `PresentX` and frame pacing into an XCL window | Minecraft Preview plays its worlds (millions of draws, none skipped); Fortnite and XDefiant draw their screens |
| **Textures** | ✅ | GFX10 tiling (all 14 2D swizzle modes, the 6 3D thick modes, linear), mip chains with mip tail, arrays, cubes and 3D textures → PC textures; supported GPU-written title subresources copy back before their writer queue's title fence becomes visible | 912 / 912 entries detiled in the tested retail archive; GPU upload bit exact in test fixtures |
| **Ray tracing** | ✅ | BLAS / TLAS built on the GPU by the game's own XboxBvhBuilder; RDNA2 BVH node format; BVH intersection instructions emulated in shaders | GPU traversal = brute force reference; the game's builder output checked on the GPU |
| **Upscaling** | ✅ | the game's AMD FSR 4 console DLL (`amd_fidelityfx_upscaler_xbox.dll`): context, the 29 pipelines in the console driver's own container (FSPX) translated, a frame's 20 passes recorded and run on the GPU | end-to-end with the real DLL: a 1280×720 → 1920×1080 frame on an RX 9070, the input's gradient in the output |

Far-jump address calculations are resolved to fixed targets only when branches enter at the start of the calculation.
An entry partway through it keeps the jump indirect and subject to the recompiler's existing indirect-branch limits.

</details>

<details>
<summary><b>Media and tools</b> – Video, Tools</summary>

| Feature | State | What it does | Verified by |
|---|---|---|---|
| **Video** | ✅ | the game's Bink 2 console DLL decodes its movies (720p / 1080p) on the CPU; the game's own YUV → RGB display shaders (`PS_Bink…` in its pack) translate like any other. Bink's GPU-assisted mode needs no Bink-specific support: its compute shaders ship with the game and go through the recompiler too (no game seen using it yet) | a game movie decodes frame by frame, picture checked |
| **Tools** | ✅ | `xcl inspect`, `xcl run`, `xcl disasm` (RDNA2, console shader containers), `xcl shaders` (pack statistics, recompile / compile runs), `xcl rpf` | – |

</details>

## Controls

A **gamepad is recommended**: console games are made for one, and their menus and controls expect it.

- **Gamepad:** any controller Windows sees through XInput (Xbox controllers, wired or wireless) is the
  game's gamepad – buttons, sticks, triggers and rumble. It can be plugged in or out while the game runs.
  GameInput keeps half a second of recent state changes so titles can read inputs between frames.
- **PlayStation controllers** (DualShock 4 wired or over Bluetooth, DualSense wired) are the gamepad
  when no XInput controller is connected – buttons, sticks and triggers, no rumble or light bar.
  `XCL_PLAYSTATION_PAD=0` turns this off.
- **Keyboard and mouse** work too, as a gamepad: WASD the left stick, the mouse the right stick, mouse
  buttons the triggers, the wheel the bumpers, Enter / Space A, Escape / Backspace B, arrows the D-pad,
  E / Q Y / X, Shift / Ctrl the stick clicks, Tab View, P Menu. `keyboard_mouse` and `mouse_sensitivity` in
  the game's `xcl.toml` turn it off or change the mouse speed. A gamepad and the keyboard / mouse can be
  used together.

To override keyboard and mouse controls for one title, add an optional table to its `xcl.toml`.
For example, use arrow keys for the left stick, clear the arrow-key D-pad bindings, and swap the
mouse buttons:

```toml
[input_profile]
move_up = ["up"]
move_left = ["left"]
move_down = ["down"]
move_right = ["right"]
dpad_up = []
dpad_left = []
dpad_down = []
dpad_right = []
left_trigger = ["mouse_left"]
right_trigger = ["mouse_right"]
```

Omitted entries use the built-in bindings. An empty list clears a binding. The output names are
`move_up`, `move_left`, `move_down`, `move_right`, `a`, `b`, `x`, `y`, the four `dpad_*`
directions, `view`, `menu`, `left_bumper`, `right_bumper`, `left_stick`, `right_stick`,
`left_trigger`, and `right_trigger`. Each accepts a list of alternative inputs. Input names
include `a` through `z`, `digit0` through `digit9`, `num0` through `num9`, `f1` through `f12`,
`left_shift`, `right_ctrl`, `right_alt`, `mouse_middle`, `mouse_x1`, and `mouse_x2`.
The `physical_q`, `physical_w`, `physical_e`, `physical_a`, `physical_s`, and `physical_d`
inputs follow the positions used by the default layout. Set `wheel_up` and `wheel_down` to a
single gamepad button name such as `right_bumper` or `y`, or to `unbound`.
Restart the title to apply changes.

## Building and usage

Windows 10/11 x64, Rust (`stable-x86_64-pc-windows-msvc`), the Windows SDK (DXC: `dxcompiler.dll`
and `dxil.dll`, looked up in `Windows Kits\10\bin\<version>\x64`, next to `xcl.exe` or in
`%XCL_DXC_DIR%`), a D3D12 GPU with Shader Model 6.6 and mesh shaders, and the system XGameRuntime
(`System32\xgameruntime.dll`, part of Windows / Gaming Services). On **Windows 10** (22H2
KB5029331+), inbox `D3D12Core.dll` stops at SM 6.5; place the Microsoft D3D12 Agility SDK
(`D3D12Core.dll`, plus `d3d12SDKLayers.dll` if using `XCL_D3D12_DEBUG=1`) in a `D3D12\` folder
next to `xcl.exe`, directly next to `xcl.exe`, in `%LOCALAPPDATA%\XCL\D3D12\`, or point `%XCL_D3D12_DIR%` at it
(for `cargo test`, use `%LOCALAPPDATA%\XCL\D3D12\` or `%XCL_D3D12_DIR%`).

- **DXC:** 1.8 or newer is recommended, from the
  [DirectXShaderCompiler releases](https://github.com/microsoft/DirectXShaderCompiler/releases): unpack one
  and point `XCL_DXC_DIR` at its `bin\x64`. A Windows SDK may hold an older one (the same `10.0.26100.0`
  folder has DXC 1.7.2308 or 1.8.2502, depending on the SDK's revision). DXC 1.7's DXIL was 2–2.7 times the
  size of 1.9's for translated compute programs and took an NVIDIA driver 1.4–2.2 times as long to build
  pipelines from ([#156](https://github.com/ivall/xcl/issues/156)). `xcl run` prints the DXC it loaded, and
  `report.json` has it under `dxc`. A compute program with many memory-window accesses is compiled
  as two libraries, the program and its accessors, and linked, which is much faster for DXC
  (`XCL_LINK_MEM=0` turns it off); the linker needs DXC 1.8 or newer. With an older one the first
  attempt fails, `xcl.log` says so, and every program is compiled whole after that.
- **GPU:** developed on AMD RDNA4. The console's geometry and NGG programs run as a compute pass plus a
  mesh pass, so a GPU without mesh shaders draws none of them. On GPUs with 32-lane waves only
  (NVIDIA) console wave64 compute, geometry and mesh programs run as two 32-lane halves that exchange
  masks and lane values ([`docs/wave64-on-wave32.md`](docs/wave64-on-wave32.md)) – slower than
  64-lane waves, most for groups of several waves; wave64 vertex and pixel shaders that read lanes 32–63
  can still differ there. Recognized compiler-pattern wave64 vertex and pixel min/max/and/or
  reductions and constant-lane scalar spill slots preserve missing partial-wave lanes; other
  missing-lane reads can still differ. With `XCL_DIRECT_DXIL=1` and `XCL_PER_LANE=1`, compute groups
  made up of complete native waves can use the per-lane form for branches on whole-wave masks;
  split-wave64 programs that access LDS stay on the HLSL path to preserve instruction order
  between their two hardware wave32 halves.
  Test results on other GPUs: [#42](https://github.com/ivall/xcl/issues/42).
- **`XCL_BVH_WATERTIGHT=1`** (off by default): the ray/triangle test of `image_bvh_intersect_ray`
  becomes watertight, so a ray exactly through an edge two triangles share hits one of them instead
  of possibly missing both. Changes the translated shaders (they get their own cache entries).
  Ported from victory-fork, after Mesa RADV (see `THIRD_PARTY.md`).

```powershell
cargo build --release --workspace          # target\release\xcl.exe

xcl inspect D:\Game\game.exe               # the game's imports and what XCL has for them
xcl run D:\Game [-- game arguments]        # run
xcl disasm <shader.bin> [--hex]            # RDNA2 disassembler (console containers too)
xcl shaders D:\data\shaders.rpf --top 50   # pack statistics; --recomp / --compile N
xcl shaders D:\data\shaders.rpf --precompile --only cs --pipelines   # build compute pipelines and save driver blobs for the next run (except by default on AMD)
xcl shaders D:\data\shaders.rpf --precompile --only cs --pipelines --used-only   # only the compute pipelines an earlier run of the title used
xcl shaders D:\data\shaders.rpf --export-hlsl out --record-hash HEX  # exact compute HLSL for investigation
xcl shaders D:\data\shaders.rpf --measure-pso HEX --repeat 3         # cold driver PSO timing (cache disabled)
xcl rpf D:\data\shaders.rpf --extract NAME --out folder
```

<details>
<summary><b>Switches, caches, logs and the files XCL writes</b></summary>

- XCL uses the game window to show preparation progress before executing title code. It prepares
  supported shader packs, supported loose shaders under `shaders/`, recorded graphics history and
  staged pipeline recipes. Closing the window cancels launch. A failed staged recipe stops launch.
  With presentation disabled, preparation runs without a window. A staged corpus still gates
  launch when the null renderer is selected. For an unstaged title set `XCL_PREPARE=0`
  to skip startup preparation, or `XCL_PREPARE_GRAPHICS=0` to skip optional graphics history.
  Native recipes validated on this adapter and driver reuse their verified blobs across
  rebuilds. Compute verification also survives changes to the preparation worker count.
  Changes to translation, pipeline layout, settings or cache files trigger validation. During
  validation, exact recipes that the driver accepts are imported from the previous pipeline
  library; incompatible or missing recipes are rebuilt normally.
- To prepare known graphics, two-pass mesh and compute recipes offline, use a local
  [archive layout profile](scripts/pso_corpus/archive-profile.md) with the following commands.
  The staging command accepts several journals, deduplicates their recipes and atomically publishes
  a content identity for later preparation receipts.

  ```powershell
  python scripts/pso_corpus/pso_archive.py profile.json archive.bin pipelines.bin
  python scripts/pso_corpus/stage.py --title <title-folder> pipelines.bin
  xcl shaders <title-folder> --prepare-corpus --threads 8
  ```

  The offline command accepts up to 32 workers. Normal launch also prepares the staged corpus.
  When a shader pack or local history overlaps the corpus, XCL prepares each exact match once.
  Use the same adapter and memory-window
  configuration for offline preparation and play so the driver can reuse the results.
  Graphics state combinations absent from the supplied recipes and local history can still
  require a first-use compile during play.
- For compute shaders that XCL can split, the runtime first tries the saved driver blob for
  the whole shader. A usable blob lets the GPU run it in one dispatch. If the blob is missing
  or the driver rejects it, XCL builds the split pieces.
- A split program first seen through `ExecuteIndirect` is also kept in
  `used-indirect-split.txt` beside the DXIL cache. A later `--pipelines --used-only` pass still
  builds its direct-dispatch pieces and whole fallback, and additionally saves both its pieces and
  whole fallback under the indirect root; other split programs do not pay for those potentially
  long driver compiles.
- With a v1 PSO manifest recorded by `XCL_PSO_MANIFEST=1`, `--through-present N` builds the
  compute pipelines first used up to that recorded present, in first-use order. A missing cutoff
  or unmatched compute key stops the pass before any build. Graphics PSOs require additional state.
- **`--pipelines` size cap**: a compute program whose DXIL is over 8 MiB and that has no saved
  blob is left to the runtime instead of being compiled in the pass — those compiles take tens
  of minutes and gigabytes and stopped the pass on the biggest programs. `XCL_PIPELINE_MAX_KB`
  sets the cap in KiB (0 turns it off).
- **Without a command line** (xcl.exe opened from Explorer) XCL shows a small menu: *Choose a
  game…* (pick its exe or `MicrosoftGame.config`) or *Run again* (the last one). On an error the
  window stays open until Enter.
- XCL works on **unpacked** game files you own; it contains nothing for decrypting or
  circumventing DRM.
- **The game folder is never modified** – writes go to the overlay.
- **Switches for how a title runs** (set them in the environment of `xcl run`):
  - `XCL_SKIP_BINK`: skips the Bink 2 movies of a title that ships a `bink2*.dll` (its logos and
    intros), by taking them straight to their last frame without decoding. Off when unset or `0`,
    `off`, `false` or `no`; `all` also skips looping movies; any other value (`1`, `intro` …)
    skips the ones that do not loop.
  - `XCL_MUTE_TTS`: the title gets no Windows speech synthesis: its XAccessibility requests are
    answered by XCL's own object, which has closed captions and high contrast only. Off by
    default; any value except empty, `0`, `off` or `false` (in any case) turns it on.
  - `XCL_FRAME_INTERVAL_US`: changes the console frame clock. Unset, the title's own frame
    interval (`SetFrameIntervalX`) is kept. `0`, `off` or `uncapped` removes the cap, so
    `WaitFrameEventX` returns at once with a fresh token. A number of microseconds is an upper
    bound on the interval, never below 1000: `8333` for 120 Hz, `4167` for 240 Hz.
  - `XCL_WAIT_TIMEOUT_MS` and `XCL_GPU_WAIT_TIMEOUT_MS`, for diagnostics: the longest XCL waits
    for a `Wait32BitValueX` / `Wait64BitValueX` that nothing satisfies (2000 ms) and the
    watchdog for a GPU queue wait (5000 ms). A value that is not a number above 0 keeps the
    default. Expiry stops the title before later GPU work is submitted; it never treats elapsed
    time as proof that the waited-for work completed.
- Translated shaders are reused across XCL rebuilds when the translator sources and DXC
  toolchain are unchanged; `XCL_TRANSLATED_CACHE=0` disables that disk cache.
  Loading a cached shader without split pieces reuses the file buffer, avoiding a second
  payload-sized allocation.
- XCL saves an eligible driver blob when its pipeline takes at least 250 ms to build, even
  after the bounded in-memory blob cache fills. This is off by default on AMD;
  `XCL_PSO_DISK_CACHE=1` forces it on, and `XCL_PSO_DISK_CACHE=0` disables it elsewhere.
  Compute blobs with different ordered static-sampler root signatures use separate entries.
- Direct compute dispatches can split large programs into pipelines built in parallel.
  `XCL_DIRECT_SPLIT_INSTRUCTIONS` sets the target instructions per piece (default 12500,
  zero disables splitting). Loops stay in one piece. A strongly connected component over the
  target gets its own piece, up to four times the target. Larger components and indirect
  dispatches use the whole pipeline.
- Huge pipelines (translated DXIL of `XCL_HUGE_PSO_BYTES` or more, default 262144) built on the
  background `xcl-pso*` workers take turns in the driver's compiler: `XCL_HUGE_PSO_CONCURRENCY`
  at once (default 1, maximum 64). Pipelines of 4 MiB or more use a separate one-wide lane so
  they cannot occupy the ordinary huge-pipeline slot; one of them can therefore run next to the
  ordinary ones, and each gated compile starts only with 3 GB of commit free. Smaller pipelines
  and those a game thread waits for never wait. `xcl.log` shows each huge one as a numbered
  `driver compile start #K` / `driver compile finish #K: … PSO, N B DXIL, M ms` pair (compute
  ones with `record HASH`, the console record's hash that `--measure-pso` and
  `--export-hlsl --record-hash` take), with whether it was gated and how long it waited;
  `XCL_PSO_TRACE=1` logs every driver compile. A huge compute program the driver refuses is
  logged with the driver's verdict (`the driver refused the compute PSO of record HASH … with
  0x…`): its dispatches are skipped.
  On AMD, cold 128–256 KiB compute pipelines use a separate lane (three background / four
  wanted at once on hosts with at least 32 GiB RAM, two / three below that), configurable with
  `XCL_MEDIUM_PSO_CONCURRENCY` and `XCL_MEDIUM_PSO_WANTED_CONCURRENCY`.
- With `XCL_DRED=1`, a D3D12 device removal logs GPU breadcrumbs in `xcl.log`.
  XCL examines up to 4,096 command lists and reports up to 64 unfinished lists, starting with
  those on which the GPU completed at least one operation. For each, it names the translated
  pipeline bound at the last operation DRED saw complete and those bound after it, by their
  console shaders' hashes (`cs`, `vs` / `ps`, `ms` / `ps`, `vs` / `gs` / `ps`).
- `XCL_TEXTURE_LEVEL_PAGES=1` gives sampled title textures backing as their mip levels stream in.
  Zero-filled mip uploads in a batch share one staging span and still clear any mapped tiles.
- `XCL_DSTORAGE_PARALLEL_DECOMPRESS=1` enables parallel CPU decompression for file-backed zlib
  requests with virtual destinations. Defaults: four workers and a shared 32 MiB staging budget
  (`XCL_DSTORAGE_WORKERS`, `XCL_DSTORAGE_MEMORY_BUDGET_MIB`). Requests retire in queue order.
- Every run writes `report.json` (called stubs, unknown D3D12.X slots and interfaces, XGameRuntime
  classes asked for, shader statistics, audio formats, crash).
  `d3d12.presents` counts presentations, not visible content. With `XCL_FRAME_DUMP=N` (`N > 1`),
  the existing optional frame dumps also set `d3d12.last_frame_dump`: the latest sampled frame's
  zero-based index, source dimensions, DXGI format, sample step, sample count and `non_dark_samples`
  (decoded 8-bit R+G+B > 30). The field is `null` until a frame is sampled; `non_dark_samples` is
  `null` if its format cannot be decoded. Intermediate render-target dumps do not update it.
  Sampling uses every fourth pixel in each direction (`XCL_FRAME_DUMP_FULL=1`: every pixel).
  A coarse sample can miss detail; a non-dark sample does not establish correct rendering or gameplay.
  With `XCL_STALL_TRACE=1`, `report.json` also contains versioned aggregate counts, total/max time and
  log2-nanosecond histograms for coarse runtime stalls, plus a bounded list of the newest spans of at
  least 50 ms. The tracing is off by default.
  `XCL_STALL_TRACE_DETAIL=1` includes high-frequency draw, dispatch, binding, descriptor/view,
  barrier, copy and command-list lifecycle recording costs on the title's first observed submitting
  thread; it also enables the coarse trace. Use it only for a diagnostic run because every recorded
  call reads the clock and updates counters.
  `XCL_GPU_TIMING=1` times each native title `ExecuteCommandLists` chunk on the CPU. The count and
  slowest duration/list count are written to `report.json`; chunks taking at least 50 ms are logged
  (`XCL_GPU_TIMING_MS` changes that threshold). It also retains a bounded set of exact
  queue-generation fence receipts and polls them without waiting while collecting the report.
  Tracked, completed and pending receipt counts are reported separately, along with exact-marker
  failures, receipts discarded on tracker contention and pending receipts evicted at the capacity
  bound. They are the correlation foundation for GPU timestamp durations, not GPU timing samples.
- **Diagnostics for a bug report** (all off by default; set them in the environment of `xcl run`):
  - `XCL_RUN_SECONDS=n`: the title ends itself after `n` seconds, with the GPU drained and
    `xcl.log` and `report.json` written, so a run of a fixed length can be attached. A value that
    is not a whole number is ignored.
  - `XCL_ACTIVITY=1` (any value turns it on): every 2 seconds `xcl.log` gets an `activity (2 s)`
    line with what the title did in that time (presents, submits, draws, dispatches, frame-event
    waits, DirectStorage requests), a `memory:` line with the process's commit and video memory in
    use, a `memory by owner` line splitting the known XCL allocations from `other`, and sampled
    texture resident / SRV-addressable / virtual texel totals. It also reports completed background
    pipelines retained behind stand-ins and how many frames ago they were last used, and the
    active foreground time plus
    serialization wait spent looking for CPU writes before submits (write-watch bytes/ranges/pages,
    sparse queue/fence-state/copy phases and zero/dirty volume, upload-gate wait/hold time split
    between submit preparation and synchronous readback, and other refresh work). A negative
    `other` exposes an overlapping or stale sample; unavailable
    adapter segments say `unknown`. Where the lines stop changing is where the title stopped.
  - `XCL_PSO_PRESSURE_OBSERVE=1`: every 2 seconds, shadow the compute-pipeline eviction policy
    without changing residency. It retains rebuild inputs and logs bounded policy outcomes when
    adapter-local usage reaches 90% of budget. `XCL_ACTIVITY` is not required.
  - `XCL_PSO_PRESSURE_EVICT=1`: enable that policy. At most one rebuildable non-console compute
    pipeline unused for 600 frames and resident for 60 frames is evicted per sample (at most 32
    candidates scanned and 8 exact transactions attempted); demand restores its exact identity.
    Eviction stops below 75% of budget. This experimental opt-in takes precedence over observe mode.
    `report.json` records cumulative detach, owner-retirement and restoration totals under
    `d3d12.pipeline_pressure`. A generation-owner release means XCL issued COM `Release`; it does
    not prove that the driver immediately reclaimed the allocation or video memory.
  - `XCL_THREAD_DUMP_AT=s1,s2,…`: at each of those seconds after the start, the state of every
    thread of the title goes into `xcl.log`, for a title that stalls.
  - `XCL_TRACE_FAULTS=1` (any value): the first 32 access violations are logged as they are raised
    (read, write or execute, the address, some registers and the callers on the stack), before the
    title's own handlers see them – for a title whose crash filter ends the process without a trace.
  - `XCL_TRACE_MEM=1` (any value turns it on): the title's memory allocations of 64 KiB or more
    are logged, the first 60 (`XCL_TRACE_MEM=n` with `n > 1`: the first `n`). The same variable
    also logs XCL's own allocations of 1 MB or more in the title's committed memory window, and
    the ones that fail.
  - `XCL_TRACE_IMPORTS`: logs the arguments and the result of every call of the imports it names.
    A comma-separated list: `1` or `hle` for every function XCL implements (also what a list with
    no word to add means), `all` for the system's functions too (short runs only), any other word
    is the start of a `dll!Function` name to add, and `-word` one to leave out; the name is the
    DLL's file name with its extension and case does not matter, so for example
    `hle,-kernelx.dll!GetTickCount,gameinput`.
  - `XCL_D3D12_TRACE=1` (any value): every call console code makes to a public D3D12.X method is
    logged with the interface and method name. Very verbose: use it for a short run.
- Title data: `%LOCALAPPDATA%\XCL\<title>\` (`%XCL_DATA_ROOT%`): `xcl.toml`, `overlay\`, `saves\`,
  `pls\` (the title's `S:\`), `temp\` (`T:\`), `cache\`, `logs\xcl.log`, `logs\report.json` (the
  previous run's: `xcl.prev.log`, `report.prev.json`).

</details>

## Reproducible cold and warm runs

On a clean commit, build and run the same title once with fresh XCL caches and once with those
exact caches warm:

```powershell
.\scripts\perf-run.ps1 -Game 'D:\Games\SomeTitle' -OutDir 'D:\xcl-perf\run-1' -Seconds 120
```

The script builds `xcl` in release mode and refuses a dirty worktree or a non-empty output
folder. Both runs use a new temporary `XCL_DATA_ROOT` and `XCL_DXIL_CACHE`; the warm run reuses
the cold run's directories. They are deleted afterwards, and existing user caches are never read,
changed, or deleted. The output folder contains only each run's `xcl.log` and `report.json`, plus
a versioned `manifest.json` with the commit, run order, hardware/toolchain, selected XCL
environment, cache identity, durations, and exit status. It contains no game files or cache data.

The manifest identifies the workload with a SHA-256 metadata fingerprint over its sorted relative
file paths, lengths, and UTC modification-time ticks. Only the fingerprint and its algorithm/version
are recorded; the inventory's file names and absolute paths are not persisted. This is a metadata
fingerprint, not a content hash: a content change that preserves the path, length, and modification
time is not detected.

The manifest calls these states **XCL-cold** and **XCL-warm**. GPU driver caches are
driver-controlled and are explicitly recorded as `uncontrolled`; this is not a fully cold-driver
benchmark.

Compare two completed artifact folders without modifying them:

```powershell
.\scripts\perf-compare.ps1 -Baseline 'D:\xcl-perf\main' -Candidate 'D:\xcl-perf\change' > comparison.json
```

The console shows raw cold/warm duration, exit/timeout state, stall totals/counts/maxima, the five
compute-build coalescing counters, and ranked-preload submissions, starts and promotions. The
allocator section also reports execute/chunk topology, retirement-signal outcomes, mark writes and
allocator-state lock timing. Sparse upload-gate submit/readback wait and hold times are summed from
the activity log; artifacts from older builds label those counters missing rather than zero. The
redirected JSON keeps those values, raw deltas, and stall log2
histograms. Missing reports or preparation summaries and unsupported preparation versions remain
explicit states; they are not treated as zero. The tool
refuses different workload fingerprints or identity algorithm/versions, run order, requested
duration, runtime environment, system, GPU, or GPU driver. Missing reports and stall summaries are
labelled as missing; it does not infer significance or percentages.

## Tests

```powershell
.\scripts\test.ps1                         # build + benchmark smoke + all workspace tests
.\scripts\perf-run.tests.ps1               # cold/warm runner fixture (no game is run)
.\scripts\perf-compare.tests.ps1           # comparison fixture (synthetic artifacts)
cargo bench -p xcl                          # synthetic performance benchmarks
cargo bench -p xcl --bench core             # run one benchmark target
cargo bench -p xcl --bench core -- --json   # one versioned JSON object per result
cargo run -p xcl --bin xcl-bench-compare -- baseline.ndjson candidate.ndjson --max-regression-percent 5
```

Benchmarks use generated inputs and public XCL APIs; they do not require game files or a title
executable. The dependency-free harness prints median, mean and minimum time plus throughput; a
positional substring selects benchmarks, and `cargo bench -p xcl -- --test` runs every closure once
as a fast smoke check. `--json` emits newline-delimited, versioned records containing the sample
count and iterations, nanoseconds per iteration, optional throughput, and the build profile,
build profile, effective optimization level, debug-assertion setting, Rust compiler, OS, architecture
and CPU identity used for the run. The build fields describe the compiler invocation that produced
the benchmark, not whichever `rustc` is on `PATH` when it runs. `XCL_BENCH_PROFILE`,
`XCL_BENCH_OPT_LEVEL`, `XCL_BENCH_DEBUG_ASSERTIONS`, `XCL_BENCH_TOOLCHAIN`, `XCL_BENCH_OS`,
`XCL_BENCH_ARCH` and `XCL_BENCH_CPU` override those identity
fields for reproducible CI; set the CPU override on systems where automatic detection reports
`null`. The records are suitable for saved baselines and automated comparisons.
`xcl-bench-compare` requires an explicit allowed regression
percentage, compares the observed medians, and exits 1 when any benchmark exceeds it. It refuses
different benchmark sets or optional environment/hardware metadata. This is a deterministic
threshold check, not a claim of statistical significance.

Unit tests, an end-to-end test with our own "console title" (`tests/testtitle`) and GPU tests
(D3D12, SM 6.6; skipped without a GPU / DXC; `XCL_ADAPTER=1` or `XCL_ADAPTER=radeon` picks
another adapter than the first high-performance one, for titles too). A GPU test that cannot run on
the adapter (no GPU / DXC / wave64 / mesh shaders) passes too, but prints `SKIPPED <test>: <reason>`, and
`scripts/test.ps1` counts them at the end; `XCL_REQUIRE_GPU=wave64` (or `mesh`, or `wave64,mesh`) makes
those skips fail. Tests on real game data are opt-in through
environment variables pointing at your own files – `XCL_SAMPLE_RPF` (shader pack),
`XCL_SAMPLE_DLLS` (the game's console DLLs), `XCL_SAMPLE_TTPF` (an RDS5 texture archive, checked entry by entry),
`XCL_SAMPLE_BCPACK` (a folder of BCPack streams `<name>.bc<mode>` with a reference decoder's blocks of each, `<name>.vic<mode>`), `XCL_SAMPLE_BK2` (a movie), `XCL_SAMPLE_BVH4`, `XCL_SAMPLE_AWC` (a sound) – described in `HANDOFF.md`; without them those
tests pass without running anything (`scripts/test.ps1` lists the ones not set). No game data is
part of the repository.

## Contributing

Help is welcome – [`CONTRIBUTING.md`](CONTRIBUTING.md) has what a pull request needs, in short: small,
one problem, tested. Test results on your GPU go to [#42](https://github.com/xcl-dev/xcl/issues/42),
questions and talk to the [Discord server](https://discord.gg/xcl).

<p align="center"><img src="assets/readme/divider.svg" width="100%" alt=""></p>

## Licenses

**GPL-3.0-or-later** (`LICENSE`); third-party sources: `THIRD_PARTY.md` (including AMD's
machine-readable ISA, AMD addrlib swizzle patterns, LLVM disassembler tests, VIctory, FFmpeg,
libHttpClient headers). WinDurango serves only as a reference – no code is ported from it. The
repository contains no headers or binaries from the console SDK.

The artwork in `assets/readme` is drawn for this repository and under its license. XCL is an independent
project: it is not affiliated with, endorsed or sponsored by the owners of the trademarks named here,
which are named only to say what XCL runs and what it runs on.
