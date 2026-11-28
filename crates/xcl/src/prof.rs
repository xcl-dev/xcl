//! Tracy instrumentation (`--features profile-tracy`); without the feature every macro here
//! expands to nothing. Zones are only recorded once `start()` ran (the host process), so the
//! launcher, `xcl shaders` and tests are unaffected even with the feature on.

/// Starts the Tracy client in this process (host mode only).
pub fn start() {
    #[cfg(feature = "profile-tracy")]
    {
        let c = tracy_client::Client::start();
        c.set_thread_name("xcl-host-main");
        c.message("xcl host started", 0);
    }
}

/// What the title's `_register_thread_local_exe_atexit_callback` import gets with the feature
/// on. Tracy's C++ `thread_local`s make xcl.exe's own CRT startup register a callback, and the
/// UCRT takes one per process: the title's CRT startup registering its own would `terminate()`
/// (0xC0000409). The title's main-thread `thread_local` destructors are then not run at exit;
/// the other threads' run from its TLS callbacks as before.
#[cfg(feature = "profile-tracy")]
pub unsafe extern "C" fn register_thread_local_exe_atexit_callback(_callback: usize) {}

/// A zone for the rest of the enclosing block: `prof_zone!("name")`, or
/// `prof_zone!("name", text_expr)` to attach a text (evaluated only while profiling).
#[macro_export]
macro_rules! prof_zone {
    ($name:literal) => {
        #[cfg(feature = "profile-tracy")]
        let _prof_zone = tracy_client::Client::running().map(|c| c.span(tracy_client::span_location!($name), 0));
    };
    ($name:literal, $text:expr) => {
        #[cfg(feature = "profile-tracy")]
        let _prof_zone = tracy_client::Client::running().map(|c| {
            let s = c.span(tracy_client::span_location!($name), 0);
            s.emit_text(&$text);
            s
        });
    };
}

/// The end of a presented frame.
#[macro_export]
macro_rules! prof_frame {
    () => {
        #[cfg(feature = "profile-tracy")]
        if let Some(c) = tracy_client::Client::running() {
            c.frame_mark();
        }
    };
}

/// A value on a named plot.
#[macro_export]
macro_rules! prof_plot {
    ($name:literal, $value:expr) => {
        #[cfg(feature = "profile-tracy")]
        if let Some(c) = tracy_client::Client::running() {
            c.plot(tracy_client::plot_name!($name), $value as f64);
        }
    };
}

#[cfg(test)]
mod tests {
    // Tests never call `start()`: the macros must be no-ops without a running client (with the
    // feature on, `Client::running()` is `None`; off, they expand to nothing), and a zone's text
    // must not be evaluated then.
    #[test]
    fn macros_are_no_ops_without_a_running_client() {
        let evaluated = std::cell::Cell::new(false);
        let _text = || {
            evaluated.set(true);
            String::from("text")
        };
        {
            crate::prof_zone!("plain zone");
            crate::prof_zone!("zone with text", _text());
            crate::prof_plot!("plot", 1u64);
            crate::prof_frame!();
        }
        assert!(!evaluated.get(), "a zone's text was built without a running client");
    }
}
