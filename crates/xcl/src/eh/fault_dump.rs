//! `XCL_FAULT_DUMP=1`: the memory around the pointers of an unhandled exception.
//!
//! The crash line names the registers; this shows what the title's data there looked like: for
//! each register, the faulting address and, when the faulting address is below 4 GiB, that address
//! under the upper half of each register that points at readable memory (a title that lost the
//! upper half of a pointer faults at the truncated address), 256 bytes on either side as 32-bit
//! words. Off by default; at most the first 2 faults of the process, one at a time.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Bytes shown on either side of a pointer.
const AROUND: u64 = 0x100;
/// Bytes per printed line (8 words).
const LINE: u64 = 32;
/// Lines per fault, headers included.
pub const MAX_LINES: usize = 300;
/// Truncated-pointer candidates per fault.
const MAX_REBASED: usize = 4;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_FAULT_DUMP").is_ok_and(|v| v == "1"))
}

/// The 16-byte line holding `p` can be read.
fn line_readable(p: u64, read: &dyn Fn(u64, &mut [u8]) -> bool) -> bool {
    let mut b = [0u8; 16];
    read(p & !0xF, &mut b)
}

/// The pointers to dump, named, without repeats: the faulting address, the faulting address
/// (from 64 KiB to 4 GiB) under the upper half of each register that is a readable pointer, then
/// the registers. The ones this report exists for come first, so the line cap never cuts them.
pub fn candidates(regs: &[(&str, u64)], fault: Option<u64>, read: &dyn Fn(u64, &mut [u8]) -> bool) -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = Vec::new();
    let push = |out: &mut Vec<(String, u64)>, name: String, p: u64| -> bool {
        let new = !out.iter().any(|(_, q)| *q == p) && line_readable(p, read);
        if new {
            out.push((name, p));
        }
        new
    };
    if let Some(f) = fault {
        push(&mut out, "fault".into(), f);
        // below 64 KiB a fault is a null pointer plus an offset, not a truncated pointer
        if f >> 32 == 0 && f >= 0x10000 {
            let mut highs: Vec<u64> = Vec::new();
            for (n, v) in regs {
                let hi = v >> 32;
                if highs.len() >= MAX_REBASED {
                    break;
                }
                if hi == 0 || highs.contains(&hi) || !line_readable(*v, read) {
                    continue;
                }
                // only rebased addresses that were dumped count against the budget
                if push(&mut out, format!("fault under {n}'s upper half"), (hi << 32) | f) {
                    highs.push(hi);
                }
            }
        }
    }
    for (n, v) in regs {
        push(&mut out, (*n).to_string(), *v);
    }
    out
}

/// The dump lines of `cands`: a header per pointer, then 32-byte lines from 256 bytes before to
/// 256 bytes after it, the line holding it marked; a line printed for an earlier pointer is not
/// printed again. At most `cap` lines.
pub fn format(cands: &[(String, u64)], read: &dyn Fn(u64, &mut [u8]) -> bool, cap: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut done: Vec<u64> = Vec::new();
    for (name, p) in cands {
        let at = p & !(LINE - 1);
        let (lo, hi) = (at.saturating_sub(AROUND), at.saturating_add(AROUND));
        if lines.len() + 1 >= cap {
            lines.push("fault dump: line limit reached".to_string());
            return lines;
        }
        lines.push(format!("fault dump: {name}={p:#x}"));
        let mut a = lo;
        while a <= hi {
            if !done.contains(&a) {
                if lines.len() + 1 >= cap {
                    lines.push("fault dump: line limit reached".to_string());
                    return lines;
                }
                done.push(a);
                let mark = if a == at { " <==" } else { "" };
                let mut b = [0u8; LINE as usize];
                if read(a, &mut b) {
                    let w: String = b.chunks(4).map(|c| format!(" {:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
                    lines.push(format!("fault dump: {a:#014x}:{w}{mark}"));
                } else {
                    lines.push(format!("fault dump: {a:#014x}: unreadable{mark}"));
                }
            }
            match a.checked_add(LINE) {
                Some(n) => a = n,
                None => break,
            }
        }
    }
    lines
}

/// The whole report for one fault (empty when the switch is off or after the first 2 faults).
pub fn report(on: bool, count: &AtomicUsize, regs: &[(&str, u64)], fault: Option<u64>, read: &dyn Fn(u64, &mut [u8]) -> bool) -> Vec<String> {
    if !on || count.fetch_add(1, Ordering::Relaxed) >= 2 {
        return Vec::new();
    }
    format(&candidates(regs, fault, read), read, MAX_LINES)
}

/// Reads the current process through `ReadProcessMemory`, which fails instead of faulting where
/// memory is not readable – also when another thread of the title frees it meanwhile, which a
/// `readable` check before a plain copy would miss. One call per read and no `VirtualQuery`
/// walk, which in a title's large blocks takes milliseconds.
fn read_here(a: u64, buf: &mut [u8]) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let Ok(p) = usize::try_from(a) else { return false };
    if p < 0x10000 {
        return false;
    }
    let mut got = 0usize;
    let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), p as *const std::ffi::c_void, buf.as_mut_ptr().cast(), buf.len(), &mut got) };
    ok != 0 && got == buf.len()
}

/// Logs the dump for a title's unhandled exception, when `XCL_FAULT_DUMP=1`.
pub fn log(regs: &[(&str, u64)], fault: Option<u64>) {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    static LOCK: Mutex<()> = Mutex::new(());
    if !enabled() {
        return;
    }
    // a fault while a report is being written (on this thread or another) gets no second one
    let _g = match LOCK.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return,
    };
    for l in report(true, &COUNT, regs, fault, &read_here) {
        crate::xlog_error!("{l}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // memory [base, base + len) is readable and holds its own address's low bytes
    fn mem(ranges: &'static [(u64, u64)]) -> impl Fn(u64, &mut [u8]) -> bool {
        move |a, buf| {
            let Some(end) = a.checked_add(buf.len() as u64) else { return false };
            if !ranges.iter().any(|&(b, l)| a >= b && end <= b + l) {
                return false;
            }
            for (i, x) in buf.iter_mut().enumerate() {
                *x = (a + i as u64) as u8;
            }
            true
        }
    }

    #[test]
    fn candidates_take_readable_registers_and_the_fault_once() {
        let r = mem(&[(0x1000, 0x1000)]);
        let regs = [("rax", 0x1010), ("rbx", 0x9000), ("rcx", 0x1010), ("rdx", 0x1800)];
        let c = candidates(&regs, Some(0x1800), &r);
        assert_eq!(c, vec![("fault".to_string(), 0x1800), ("rax".to_string(), 0x1010)]);
        let c = candidates(&regs, None, &r);
        assert_eq!(c, vec![("rax".to_string(), 0x1010), ("rdx".to_string(), 0x1800)]);
    }

    #[test]
    fn candidates_rebase_a_truncated_fault_under_pointer_upper_halves() {
        let r = mem(&[(0x7_0001_0000, 0x1000), (0x8_0000_0000, 0x100), (0x9_0001_0000, 0x100)]);
        let regs = [("rax", 0x7_0001_0010), ("rbx", 0x7_0001_0020), ("rcx", 0x8_0000_0000), ("rdx", 0x9_0000_0000)];
        // rax and rbx share an upper half; rdx does not point at readable memory, so its upper
        // half is not tried although 9<<32 | fault is readable
        let c = candidates(&regs, Some(0x1_0080), &r);
        let names: Vec<&str> = c.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["fault under rax's upper half", "rax", "rbx", "rcx"]);
        assert_eq!(c[0].1, 0x7_0001_0080);
        let c = candidates(&regs, Some(0x2_0000), &r);
        assert_eq!(c.len(), 3);
        // no rebasing above 4 GiB
        let c = candidates(&regs, Some(0x1_0001_0080), &r);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn unreadable_rebases_do_not_use_up_the_budget() {
        static R: [(u64, u64); 7] = [(1 << 32, 0x100), (2 << 32, 0x100), (3 << 32, 0x100), (4 << 32, 0x100), (5 << 32, 0x100), (6 << 32, 0x100), ((5 << 32) | 0x2_0000, 0x100)];
        let r = mem(&R);
        let regs: Vec<(&str, u64)> = ["a", "b", "c", "d", "e", "f"].iter().zip(1u64..).map(|(n, i)| (*n, i << 32)).collect();
        let c = candidates(&regs, Some(0x2_0000), &r);
        assert_eq!(c[0], ("fault under e's upper half".to_string(), (5 << 32) | 0x2_0000));
    }

    #[test]
    fn a_fault_below_64k_is_not_rebased() {
        let r = mem(&[(0x7_0000_0000, 0x10000)]);
        let c = candidates(&[("rax", 0x7_0000_1000)], Some(0x80), &r);
        assert_eq!(c, vec![("rax".to_string(), 0x7_0000_1000)]);
    }

    #[test]
    fn the_rebased_fault_survives_the_cap() {
        let r = mem(&[(1 << 32, 1 << 32)]);
        let names: Vec<String> = (0..16).map(|i| format!("r{i}")).collect();
        let regs: Vec<(&str, u64)> = names.iter().zip(0u64..).map(|(n, i)| (n.as_str(), 0x1_0000_5000 + i * 0x10_0000)).collect();
        let c = candidates(&regs, Some(0x2_0000), &r);
        assert_eq!(c.len(), 17);
        let l = format(&c, &r, MAX_LINES);
        assert_eq!(l[0], "fault dump: fault under r0's upper half=0x100020000");
        assert_eq!(l.last().unwrap(), "fault dump: line limit reached");
    }

    #[test]
    fn format_stops_at_the_top_of_the_address_space() {
        let all = |_: u64, b: &mut [u8]| {
            b.fill(0);
            true
        };
        let l = format(&[("rax".into(), u64::MAX - 8)], &all, MAX_LINES);
        assert_eq!(l.len(), 1 + 9);
        assert!(l.last().unwrap().starts_with("fault dump: 0xffffffffffffffe0:") && l.last().unwrap().ends_with(" <=="));
    }

    #[test]
    fn candidates_rebase_at_most_a_few() {
        static R: [(u64, u64); 6] = [(0x1_0000_0000, 0x2_0000), (0x2_0000_0000, 0x2_0000), (0x3_0000_0000, 0x2_0000), (0x4_0000_0000, 0x2_0000), (0x5_0000_0000, 0x2_0000), (0x6_0000_0000, 0x2_0000)];
        let r = mem(&R);
        let regs: Vec<(&str, u64)> = ["a", "b", "c", "d", "e", "f"].iter().zip(1u64..).map(|(n, i)| (*n, i << 32)).collect();
        let c = candidates(&regs, Some(0x1_0000), &r);
        assert_eq!(c.iter().filter(|(n, _)| n.starts_with("fault under")).count(), MAX_REBASED);
    }

    #[test]
    fn format_marks_the_line_and_prints_words() {
        let r = mem(&[(0x1000, 0x1000)]);
        let l = format(&[("rax".into(), 0x1234)], &r, MAX_LINES);
        assert_eq!(l[0], "fault dump: rax=0x1234");
        assert_eq!(l.len(), 1 + 17);
        assert_eq!(l[1], "fault dump: 0x000000001120: 23222120 27262524 2b2a2928 2f2e2d2c 33323130 37363534 3b3a3938 3f3e3d3c");
        let marked: Vec<&String> = l.iter().filter(|s| s.ends_with(" <==")).collect();
        assert_eq!(marked.len(), 1);
        assert!(marked[0].starts_with("fault dump: 0x000000001220: 23222120"));
    }

    #[test]
    fn format_says_unreadable_and_skips_lines_already_shown() {
        let r = mem(&[(0x1000, 0x100)]);
        let l = format(&[("rax".into(), 0x1000), ("rbx".into(), 0x1040)], &r, MAX_LINES);
        assert_eq!(l[1], "fault dump: 0x000000000f00: unreadable");
        assert!(l[9].starts_with("fault dump: 0x000000001000: 03020100") && l[9].ends_with(" <=="));
        // rbx's window adds only the two lines past rax's
        assert_eq!(l.len(), 1 + 17 + 1 + 2);
        assert_eq!(l[18], "fault dump: rbx=0x1040");
    }

    #[test]
    fn format_stops_at_the_cap() {
        let r = mem(&[(0, 0x10_0000)]);
        let c: Vec<(String, u64)> = (0..40u64).map(|i| (format!("r{i}"), 0x1000 + i * 0x1000)).collect();
        let l = format(&c, &r, MAX_LINES);
        assert_eq!(l.len(), MAX_LINES);
        assert_eq!(l.last().unwrap(), "fault dump: line limit reached");
    }

    #[test]
    fn report_is_empty_when_off_and_after_two_faults() {
        let r = mem(&[(0x1000, 0x1000)]);
        let regs = [("rax", 0x1100)];
        let n = AtomicUsize::new(0);
        assert!(report(false, &n, &regs, None, &r).is_empty());
        assert!(!report(true, &n, &regs, None, &r).is_empty());
        assert!(!report(true, &n, &regs, None, &r).is_empty());
        assert!(report(true, &n, &regs, None, &r).is_empty());
    }
}
