//! The host process: started by the launcher with the game's command line. Sets up
//! logging, the VFS, hooks and HLE, maps the title and jumps to its entry point.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::Boot;

pub static HOST_MODE: AtomicBool = AtomicBool::new(false);

/// The system's commit charge and its limit in bytes (RAM plus page files): what all programs'
/// memory, and all video memory, is counted against. (0, 0): not known.
pub fn system_commit() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
    unsafe {
        let mut p: PERFORMANCE_INFORMATION = std::mem::zeroed();
        p.cb = std::mem::size_of_val(&p) as u32;
        if K32GetPerformanceInfo(&mut p, p.cb) == 0 {
            return (0, 0);
        }
        (p.CommitTotal as u64 * p.PageSize as u64, p.CommitLimit as u64 * p.PageSize as u64)
    }
}

/// Whether commit left over (`limit - commit`) is logged as low: under `LOW_COMMIT` the first
/// time, then each time it is half of what was logged last.
fn low_commit_logged(left: u64, last: u64) -> bool {
    left < LOW_COMMIT && (last == 0 || left <= last / 2)
}

/// Commit left under which the log warns: a retail title's world took the last 6 GB in seconds.
const LOW_COMMIT: u64 = 8 << 30;

/// The system is running out of commit: said in the log (once, then at each halving). When it
/// runs out, allocations fail for every program. (Measured on a retail title's world, whose
/// process held 45 GB beside 71 GB of other programs under a 147 GB limit: the process's own
/// memory counts against the commit, its 24 GB of video memory do not.)
fn warn_low_commit(commit: u64, limit: u64, private: u64, video: u64) {
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let left = limit.saturating_sub(commit);
    let last = LAST.load(Ordering::Relaxed);
    if !low_commit_logged(left, last) {
        return;
    }
    LAST.store(left.max(1), Ordering::Relaxed);
    crate::xlog_warn!(
        "memory: the system is running out of commit: {} MB left of {} MB (RAM + page file). This process holds {} MB (and {} MB of video memory, which do not count), other programs the rest. When it runs out allocations fail for every program: close other programs or give Windows a larger page file",
        left >> 20,
        limit >> 20,
        private >> 20,
        video >> 20
    );
}

pub fn host_main(boot_path: &Path) -> i32 {
    let boot: Boot = match std::fs::read_to_string(boot_path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("xcl host: cannot read {}: {e}", boot_path.display());
            return 0x7C1_0001;
        }
    };
    crate::log::init(
        Some(&boot.logs_dir().join("xcl.log")),
        crate::log::Level::parse(&boot.config.log_level),
        boot.config.log_to_stderr,
    );
    crate::log::log_panics();
    crate::xlog_info!("XCL {} – host for '{}'", env!("CARGO_PKG_VERSION"), boot.title);
    crate::xlog_info!("game: {} (read-only folder: {})", boot.exe_path.display(), boot.game_root.display());
    crate::xlog_info!("XCL data: {}", boot.data_dir.display());
    crate::xlog_info!(
        "console: {} (title memory {} MiB, mode {})",
        if boot.config.is_series_s() { "Xbox Series S (lockhart)" } else { "Xbox Series X (anaconda)" },
        boot.config.title_memory_mb,
        boot.config.title_memory
    );
    if std::env::var_os("XCL_STALL_TRACE_DETAIL").is_some() {
        crate::stall::enable_detail();
        crate::xlog_info!("detailed stall attribution enabled (XCL_STALL_TRACE_DETAIL)");
    } else if std::env::var_os("XCL_STALL_TRACE").is_some() {
        crate::stall::enable();
        crate::xlog_info!("stall attribution enabled (XCL_STALL_TRACE)");
    }

    if std::env::var_os("XCL_UMA_MB").is_none() && boot.config.uma_mb != 0 {
        std::env::set_var("XCL_UMA_MB", boot.config.uma_mb.to_string());
    }
    crate::state::set_boot(boot.clone());
    if let Err(e) = crate::hle::d3d12x::present::preparation::before_guest(&boot) {
        crate::xlog_error!("Shader preparation: {e}");
        eprintln!("xcl: {e}");
        return 2;
    }
    let _ = std::thread::Builder::new().name("xcl-split-counts".into()).spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(10));
        crate::hle::d3d12x::real::log_dispatch_chain_counts();
    });
    // Activity diagnostics and pressure policy share one bounded two-second sampler. Pressure-only
    // mode queries only adapter-local usage; it does not enable the activity hot-path probes.
    let activity = crate::hle::d3d12x::memory_report::enabled();
    let pressure = crate::hle::d3d12x::pso_async::pressure_sampling_enabled();
    if activity || pressure {
        let name = if activity { "xcl-activity" } else { "xcl-pso-pressure" };
        let sampler = std::thread::Builder::new().name(name.into()).spawn(move || {
            let snap = || {
                let (p, e) = crate::hle::d3d12x::present::activity();
                let (d, x) = crate::hle::d3d12x::bind::activity();
                let f = crate::hle::d3d12x::frame::activity();
                let (r, b, fail) = crate::hle::dstorage::counters();
                [p, e, d, x, f[0], f[1], f[2], f[3], r, b >> 20, fail]
            };
            let mut last = activity.then(snap);
            let mut last_looks = activity.then(crate::hle::d3d12x::sparse::look_counters);
            let mut pressure_turn = 0u64;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let owner = activity.then(crate::hle::d3d12x::memory_report::snapshot);
                if pressure {
                    pressure_turn = pressure_turn.wrapping_add(1);
                    let local = owner.as_ref().map_or_else(
                        crate::hle::d3d12x::memory_report::local_segment,
                        |owner| owner.local,
                    );
                    if let Some(line) = crate::hle::d3d12x::pso_async::observe_pressure_turn(pressure_turn, local) {
                        crate::xlog_info!("{line}");
                    }
                }
                if !activity { continue }
                let now = snap();
                let d: Vec<u64> = now.iter().zip(last.take().unwrap()).map(|(a, b)| a - b).collect();
                crate::xlog_info!(
                    "activity (2 s): {} presents, {} submits, {} draws, {} dispatches, {} frame-event waits ({} timed out), {} frame statistics ({} not flipped); DirectStorage {} requests, {} MB, {} failed",
                    d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]
                );
                let looks = crate::hle::d3d12x::sparse::look_counters();
                crate::xlog_info!("{}", crate::hle::d3d12x::sparse::look_line(
                    crate::hle::d3d12x::sparse::look_delta(looks, last_looks.unwrap()), 2,
                ));
                last_looks = Some(looks);
                // where the memory is: the process's commit, video memory, and XCL's parts of it
                let owner = owner.unwrap();
                let private = owner.commit.unwrap_or(0);
                let local = owner.local.map_or(0, |s| s.usage);
                let non_local = owner.non_local.map_or(0, |s| s.usage);
                let (arenas, arenas_made) = crate::hle::d3d12x::vram::arena_bytes();
                let (pools, pools_mapped) = crate::hle::d3d12x::sparse::pool_bytes();
                let (pages, page_pools) = crate::hle::d3d12x::vram::page_bytes();
                let mb = |b: u64| b >> 20;
                // the system's commit: this process's memory counts against it with every other
                // program's (video memory does not), and where it ends allocations fail
                let (commit, commit_limit) = system_commit();
                let mut commit_text = String::new();
                if commit_limit != 0 {
                    commit_text = format!("; system commit {} of {} MB", mb(commit), mb(commit_limit));
                    warn_low_commit(commit, commit_limit, private as u64, local);
                }
                crate::xlog_info!(
                    "memory: {} MB committed by the process; video memory {} MB local, {} MB non-local; target pages {} MB (pools {} MB); arenas {} MB in use, {} MB made; sparse pools {} MB ({} MB mapped){commit_text}",
                    mb(private),
                    mb(local),
                    mb(non_local),
                    mb(pages as u64),
                    mb(page_pools as u64),
                    mb(arenas as u64),
                    mb(arenas_made as u64),
                    mb(pools as u64),
                    mb(pools_mapped as u64)
                );
                crate::xlog_info!("{}", owner.line());
                let pso = crate::hle::d3d12x::pso_async::retention_stats();
                crate::xlog_info!(
                    "pipeline retention: {} completed background PSOs behind stand-ins; {} unused for 60+ frames, {} for 600+ frames; oldest {} frames",
                    pso.retained_reals,
                    pso.unused_60_frames,
                    pso.unused_600_frames,
                    pso.oldest_unused_frames
                );
                let ts = crate::hle::d3d12x::vram::texel_stats();
                if ts.sampled_textures > 0 {
                    let pct = if ts.total_texels > 0 { ts.resident_texels.saturating_mul(100) / ts.total_texels } else { 0 };
                    let mtex = |n: u64| n as f64 / 1_000_000.0;
                    let evict = if ts.evicted_tiles > 0 {
                        format!("; evicted {} mips ({} MB)", ts.evicted_mips, (ts.evicted_tiles * 65536) >> 20)
                    } else {
                        String::new()
                    };
                    crate::xlog_info!(
                        "texture residency: {} sampled ({} with SRV); texels {:.1}M resident / {:.1}M srv-sampled / {:.1}M virtual ({}% resident){evict}",
                        ts.sampled_textures,
                        ts.srv_textures,
                        mtex(ts.resident_texels),
                        mtex(ts.srv_sampled_texels),
                        mtex(ts.total_texels),
                        pct
                    );
                }
                last = Some(now);
            }
        });
        if let Err(e) = sampler {
            crate::xlog_warn!("D3D12.X: could not start the activity/pressure sampler: {e}");
        }
    }
    // `XCL_THREAD_DUMP_AT=s1,s2…`: where every thread is at those seconds (a stalled title), and
    // which title fences and XCL queue waits are behind then
    if let Ok(v) = std::env::var("XCL_THREAD_DUMP_AT") {
        let at: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if !at.is_empty() {
            crate::hle::d3d12x::list_x::enable_sync_state_tracing();
        }
        let _ = std::thread::Builder::new().name("xcl-thread-dump".into()).spawn(move || {
            let t0 = std::time::Instant::now();
            for s in at {
                let left = s - t0.elapsed().as_secs_f64();
                if left > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(left));
                }
                crate::eh::dump_threads();
                crate::hle::d3d12x::list_x::log_sync_state(&format!("{s} s"));
            }
        });
    }
    crate::stubs::init_db(&boot.config.stub_returns);
    crate::hle::init(&boot.config);
    crate::hle::kernelx::fine_timer();
    crate::loader::set_cache_dir(boot.cache_dir());
    let mut dirs = vec![boot.game_root.clone()];
    if let Some(p) = boot.exe_path.parent() {
        if p != boot.game_root {
            dirs.insert(0, p.to_path_buf());
        }
    }
    crate::resolver::set_search_dirs(dirs);

    let mounts: Vec<(String, String)> = boot.config.mounts.iter().map(|(k, v)| (k.clone(), boot.expand(v))).collect();
    // (the shader cache is XCL's too: on a drive a console mount names it must stay where it is)
    crate::vfs::init(crate::vfs::Vfs::new(&boot.game_root, &boot.data_dir, mounts).own_folder(&crate::gpu::recomp::dxc::cache_dir()));
    let _ = std::fs::create_dir_all(boot.data_dir.join("temp"));

    if let Err(e) = crate::vfs_hooks::install() {
        crate::xlog_error!("VFS: not all hooks installed: {e}");
    }
    if let Err(e) = crate::eh::install_pc_hook() {
        crate::xlog_error!("hook RtlPcToFileHeader: {e} – C++ exceptions in the game may not work");
    }
    crate::eh::install_crash_handler();
    crate::eh::start_probes();
    crate::hle::d3d12x::backend_before_guest();
    HOST_MODE.store(true, Ordering::Release);

    // `XCL_RUN_SECONDS=n`: the title ends itself after n seconds – the GPU drained, the log and
    // report written (a process killed from outside with GPU work in flight can hang in the driver)
    if let Some(secs) = std::env::var("XCL_RUN_SECONDS").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            let (shown, presented) = crate::hle::d3d12x::present::counters();
            crate::xlog_info!("XCL_RUN_SECONDS: {secs} s – ending the title ({shown} frames shown, {presented} PresentX calls)");
            crate::report::write_final(None);
            crate::log::flush();
            crate::hle::d3d12x::pso_async::shutdown_used();
            crate::hle::d3d12x::real::shutdown_indirect_split_records(std::time::Duration::from_millis(500));
            crate::hle::d3d12x::real::end_title(0);
        });
    }

    let main = match crate::loader::load_main(&boot.exe_path) {
        Ok(m) => m,
        Err(e) => {
            crate::xlog_error!("loading the game failed: {e}");
            crate::report::write_final(Some(format!("loader: {e}")));
            return 0x7C1_0002;
        }
    };
    unsafe { crate::hle::kernelx::adopt_title_hooks(&main) };
    crate::report::write_final(None);
    let code = crate::loader::run_main(&main);
    crate::xlog_info!("game entry point returned code {code}");
    crate::report::write_final(None);
    crate::log::flush();
    crate::hle::d3d12x::pso_async::shutdown_used();
    crate::hle::d3d12x::real::shutdown_indirect_split_records(std::time::Duration::from_millis(500));
    code as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_commit_is_logged_under_the_mark_and_at_each_halving() {
        assert!(!low_commit_logged(LOW_COMMIT, 0), "at the mark: not yet");
        assert!(low_commit_logged(LOW_COMMIT - 1, 0));
        assert!(!low_commit_logged(5 << 30, 7 << 30), "less, but not half of what was logged last");
        assert!(low_commit_logged(3 << 30, 7 << 30));
        assert!(low_commit_logged(0, 1), "none left");
        assert!(!low_commit_logged(9 << 30, 1 << 30), "back above the mark");
    }

    #[test]
    fn the_system_commit_is_known_and_under_its_limit() {
        let (commit, limit) = system_commit();
        assert!(commit > 0 && commit <= limit, "{commit} of {limit}");
    }
}
