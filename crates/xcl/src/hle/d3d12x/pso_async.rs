//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A lazy pipeline (`lazy`) is built at its first use or ahead of it: those
//! of earlier runs in their order (`used::rank`), every other one at the lowest priority
//! (`Backlog`). A draw or dispatch with a pipeline still being built waits for it
//! (`resolve_wait`) where it is used, a `Dispatch` at its submit (`waits_for`,
//! `need_by_submit`): a dispatch until it is built (a skipped one corrupts what it computes)
//! unless `XCL_CS_PSO_WAIT_MS` limits that, a draw only while its thread has waited less than
//! `XCL_GFX_PSO_WAIT_MS` in the last second when that is set (`gfx_wait_left`) – then it and
//! the later draws of that pipeline draw nothing until it is built. An ExecuteIndirect whose
//! pipeline is still being built after that is skipped.
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock, Weak};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12PipelineState;

use super::build_queue::{BuildQueue, JobHandle, JobStatus, Priority};

const QUEUE_IDLE: u8 = 0;
const QUEUE_ACTIVE: u8 = 1;
const QUEUE_DONE: u8 = 2;

/// Process-lifetime ranked-preload outcomes; `started` means dequeued in the Preload band.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct RankedPreloadStats {
    pub submissions: u64,
    pub started: u64,
    pub promoted: u64,
}

/// Rare-path pressure lifecycle totals. Each field is a relaxed cumulative counter, so a snapshot
/// across fields is approximate. A released generation owner means XCL issued that COM owner's
/// `Release`; it does not prove the driver immediately reclaimed the allocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct PipelinePressureStats {
    pub publications_detached: u64,
    pub policy_owner_retirements: u64,
    pub generation_owners_released: u64,
    /// Unique exact restoration jobs accepted by the process coordinator; coalescing is excluded.
    pub restoration_attempts: u64,
    pub restoration_succeeded: u64,
    pub restoration_failed: u64,
}

#[derive(Default)]
struct PipelinePressureCounters {
    publications_detached: AtomicU64,
    policy_owner_retirements: AtomicU64,
    generation_owners_released: AtomicU64,
    restoration_attempts: AtomicU64,
    restoration_succeeded: AtomicU64,
    restoration_failed: AtomicU64,
}

impl PipelinePressureCounters {
    fn snapshot(&self) -> PipelinePressureStats {
        PipelinePressureStats {
            publications_detached: self.publications_detached.load(Ordering::Relaxed),
            policy_owner_retirements: self.policy_owner_retirements.load(Ordering::Relaxed),
            generation_owners_released: self.generation_owners_released.load(Ordering::Relaxed),
            restoration_attempts: self.restoration_attempts.load(Ordering::Relaxed),
            restoration_succeeded: self.restoration_succeeded.load(Ordering::Relaxed),
            restoration_failed: self.restoration_failed.load(Ordering::Relaxed),
        }
    }
}

fn pressure_counters() -> Arc<PipelinePressureCounters> {
    static COUNTERS: OnceLock<Arc<PipelinePressureCounters>> = OnceLock::new();
    COUNTERS.get_or_init(|| Arc::new(PipelinePressureCounters::default())).clone()
}

pub fn pipeline_pressure_stats() -> PipelinePressureStats { pressure_counters().snapshot() }

fn pressure_observation_line(
    observation: PressureObservation, stats: PipelinePressureStats,
) -> String {
    format!(
        "{}; cumulative: {} detached, {} policy owners retired, {} generation owners released, {} restoration attempts ({} succeeded, {} failed)",
        observation.line(), stats.publications_detached, stats.policy_owner_retirements,
        stats.generation_owners_released, stats.restoration_attempts,
        stats.restoration_succeeded, stats.restoration_failed,
    )
}

struct Pending {
    build: Arc<Build>,
    /// the driver pipeline binds get till then (`shared`); 0: the stand-in is one itself
    driver: AtomicUsize,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// Exact serialized graphics/mesh recipe identity for the first-use manifest; 0 = none.
    manifest_recipe: AtomicU64,
    track_used: bool,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
    /// its dispatches may be left out while it is built (`image_only`): binds do not wait for it
    spared: std::sync::atomic::AtomicBool,
    /// what its first use tells besides `used::note` (`on_first_use`)
    on_use: Mutex<Option<OnUse>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PipelineKind {
    Unknown,
    Graphics,
    /// Compute submitted while console-shaped command-list playback owns pipeline binding.
    ConsoleListCompute,
    Compute,
}

impl PipelineKind {
    fn submitted(compute: bool, console_lists: bool) -> Self {
        match (compute, console_lists) {
            (false, _) => Self::Graphics,
            (true, true) => Self::ConsoleListCompute,
            (true, false) => Self::Compute,
        }
    }
}

/// A pipeline build independent of the title-visible stand-ins that requested it.
struct Build {
    /// Authoritative pipeline kind supplied by its creation path. Never infer this from the
    /// optional compute coalescing key: not every compute build has one.
    kind: PipelineKind,
    compute_key: Option<Arc<super::real::compute_build_key::ComputeBuildKey>>,
    /// Immutable inputs for rebuilding this pipeline after its first job has been consumed.
    #[allow(dead_code)] // retained for the later opt-in eviction/rebuild stage
    recipe: Option<Recipe>,
    /// the build, until a worker takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// Whether a title-visible stand-in owns this build, and its exact queued-count state.
    tracked: AtomicBool,
    queue_state: AtomicU8,
    /// the real pipeline state (one owned raw pointer shared by all `Pending`s), 0 until built
    /// or when building failed
    real: AtomicUsize,
    /// Lazily retained exact owner for an ordinary one-shot result. Creation and removal are
    /// serialized by `done`, so a title release cannot cross the first exact pin.
    one_shot_generation: Mutex<Option<Arc<Generation<ID3D12PipelineState>>>>,
    /// Last frame in which any owner resolved this build, for `XCL_ACTIVITY` retention ages.
    last_use_frame: AtomicU64,
    /// Frame in which `real` became retained; age never begins before the build completed.
    completed_frame: AtomicU64,
    /// Low bits count timestamp writers; the high bit gives final eviction validation exclusive
    /// access without putting a mutex on the bind hot path.
    eviction_timestamp_access: AtomicUsize,
    /// This build already has one use record in the bounded eviction-candidate journal.
    /// Touched only while recipe retention is enabled.
    candidate_queued: AtomicBool,
    /// Exact armed eviction visible before its native owners may be detached. Kept separate
    /// from recipe state so coordinator locks never overlap the build/rebuild/generation chain.
    eviction: Mutex<Option<EvictedBuildToken>>,
    /// Lock-free observation that terminal state has been fully published. Exact generation,
    /// release, and condvar decisions remain serialized by `done`.
    completed: AtomicBool,
    done: Mutex<bool>,
    cv: Condvar,
    /// every stand-in interested in this build was released before it finished
    orphan: std::sync::atomic::AtomicBool,
    /// Title-visible `Pending`s sharing this build. Changed while holding `done`.
    owners: AtomicUsize,
    /// One of the title's threads waits, or a recorded submit will wait, for this build. Polled
    /// by DXC and driver admission.
    wanted: AtomicBool,
    /// Record-time submit demand has completed worker startup, promotion, and scheduling.
    by_submit: AtomicBool,
    demand: Mutex<Demand>,
    queued: Mutex<Option<JobHandle>>,
    /// a compute pipeline's console program (`fnv64` of its CS, as in the `[SHADER]` and driver
    /// compile lines); 0 = not known
    record: u64,
    /// made by a lazy submit (the pipeline line counts these)
    lazy: bool,
    /// one of its stand-ins was used (`Pending::note_use_at` sets it; `build_is_ahead`, the line)
    used: AtomicBool,
    /// a bind waited for it past its budget, or binds go on without it (`build_is_late`)
    late: AtomicBool,
    /// its build gave the driver a huge program (`note_huge`)
    huge: AtomicBool,
    /// a wait for it passed 1 s and was logged (once a build)
    warned: AtomicBool,
    /// its translation and DXC compile on their own, until the backlog runs them (`Warm`)
    warm: Mutex<Option<Warm>>,
}

const EVICTION_CANDIDATE_CAPACITY: usize = 1024;

/// Bounded lock-free MPMC ring. Bind threads append weak identities, so an unconsumed record does
/// not retain a released pipeline; a future policy can drain a fixed number on a background thread
/// without scanning the title-visible registry.
struct CandidateSlot<T> {
    sequence: AtomicUsize,
    value: std::cell::UnsafeCell<std::mem::MaybeUninit<Weak<T>>>,
}

unsafe impl<T: Send + Sync> Sync for CandidateSlot<T> {}

struct CandidateQueue<T, const N: usize> {
    enqueue: AtomicUsize,
    dequeue: AtomicUsize,
    slots: [CandidateSlot<T>; N],
}

unsafe impl<T: Send + Sync, const N: usize> Sync for CandidateQueue<T, N> {}

impl<T, const N: usize> CandidateQueue<T, N> {
    fn new() -> Self {
        // The sequence protocol needs distinct empty/full generations for each slot. At capacity
        // one the producer can observe its just-published sequence as the next empty generation
        // and overwrite an unread value. Signed sequence differences also require a half-range
        // bound across counter wrap.
        assert!(N > 1 && N.is_power_of_two() && N <= isize::MAX as usize);
        Self {
            enqueue: AtomicUsize::new(0),
            dequeue: AtomicUsize::new(0),
            slots: std::array::from_fn(|sequence| CandidateSlot {
                sequence: AtomicUsize::new(sequence),
                value: std::cell::UnsafeCell::new(std::mem::MaybeUninit::uninit()),
            }),
        }
    }

    fn push(&self, value: Weak<T>) -> Result<(), Weak<T>> {
        let mut position = self.enqueue.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[position & (N - 1)];
            let sequence = slot.sequence.load(Ordering::Acquire);
            let difference = sequence.wrapping_sub(position) as isize;
            if difference == 0 {
                match self.enqueue.compare_exchange_weak(
                    position, position.wrapping_add(1), Ordering::Relaxed, Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        unsafe { (*slot.value.get()).write(value) };
                        slot.sequence.store(position.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => position = actual,
                }
            } else if difference < 0 {
                return Err(value);
            } else {
                position = self.enqueue.load(Ordering::Relaxed);
            }
        }
    }

    fn pop(&self) -> Option<Weak<T>> {
        let mut position = self.dequeue.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[position & (N - 1)];
            let sequence = slot.sequence.load(Ordering::Acquire);
            let difference = sequence.wrapping_sub(position.wrapping_add(1)) as isize;
            if difference == 0 {
                match self.dequeue.compare_exchange_weak(
                    position, position.wrapping_add(1), Ordering::Relaxed, Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let value = unsafe { (*slot.value.get()).assume_init_read() };
                        slot.sequence.store(position.wrapping_add(N), Ordering::Release);
                        return Some(value);
                    }
                    Err(actual) => position = actual,
                }
            } else if difference < 0 {
                return None;
            } else {
                position = self.dequeue.load(Ordering::Relaxed);
            }
        }
    }
}

impl<T, const N: usize> Drop for CandidateQueue<T, N> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

type EvictionCandidateQueue = CandidateQueue<Build, EVICTION_CANDIDATE_CAPACITY>;

fn eviction_candidates() -> &'static EvictionCandidateQueue {
    static CANDIDATES: OnceLock<EvictionCandidateQueue> = OnceLock::new();
    CANDIDATES.get_or_init(EvictionCandidateQueue::new)
}

/// Who waits for the pipeline, and the preload worker building it.
#[derive(Default)]
struct Demand {
    /// binds waiting for it: a wait that ran out of its budget leaves (`Build::unwant`), one
    /// that saw the build end stays counted. `Build::wanted` is `waiters != 0`.
    waiters: u32,
    /// a submit counts among `waiters` before it gets to it (`want`, `need_by_submit`; left with
    /// the wait for it)
    ahead: bool,
    /// thread id of the preload worker (lowest priority) building it, 0 = none
    preload_thread: u32,
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Build>>> = const { std::cell::RefCell::new(None) };
    /// this is one of the workers reserved for low-priority preload builds
    static PRELOAD_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// this thread serves the normal (`BoundSoon` through `Foreground`) worker band
    static NORMAL_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// when the build this thread runs (`run_build`) started
    static JOB_START: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
    /// the build this thread runs gave the driver a huge program (`note_huge`)
    static JOB_HUGE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// a backlog worker's build: whether it came from `Backlog::compute`; None once another
    /// worker took this one's place (`build_parks`)
    static PRELOAD_STEP: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// explicit late-build override for off-pool workers (e.g. `dispatch_chain`'s `xcl-pso-whole`)
    static JOB_LATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// This thread is a worker and one of the title's threads waits for what it is building: the
/// build must not stand back for the title's sake (`real::big_compile_turn`).
pub(super) fn build_is_wanted() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|build| build.wanted.load(Ordering::Acquire)))
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build as `build_is_wanted`; a thread that is not XCL's is the
/// title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(build) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || build.wanted.load(Ordering::Acquire));
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

/// The time since the build this thread runs started (None: it runs none): the driver compile
/// log says where a slow pipeline's time went before the driver got it.
pub(super) fn job_elapsed() -> Option<std::time::Duration> {
    JOB_START.with(|s| s.get()).map(|t| t.elapsed())
}

/// The build this thread runs gives the driver a huge program (`real::huge_pso_bytes`).
pub(super) fn note_huge() {
    JOB_HUGE.with(|h| h.set(true));
}

/// This thread is a worker building a pipeline a bind waited for and went on without
/// (`resolve_within`), or a background `xcl-pso-whole` worker (`with_late_build`): nobody waits
/// for it now, but the title's draws or dispatches with it are left out until it is built
/// (`real::huge_compile_turn`).
pub(super) fn build_is_late() -> bool {
    JOB_LATE.with(|l| l.get())
        || BUILDING.with(|b| b.borrow().as_ref().is_some_and(|build| build.late.load(Ordering::Acquire)))
}

/// Run `f` with `build_is_late() == true` on the current thread, restoring the previous state on
/// return or unwind.
pub(super) fn with_late_build<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            JOB_LATE.with(|l| l.set(self.0));
        }
    }
    let prev = JOB_LATE.with(|l| l.replace(true));
    let _reset = Reset(prev);
    f()
}

/// This thread is a preload or backlog worker building a pipeline nobody has used yet: its huge
/// driver compile gives its turn to any other (`real::big_compile_turn`).
pub(super) fn build_is_ahead() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|build| {
        !build.used.load(Ordering::Relaxed) && build.demand.lock().unwrap().preload_thread != 0
    }))
}

/// The build ahead on this thread waits for its pipeline's first use before its driver compile
/// (`real::huge_compile_turn`: no giant one is made ahead), which may never come. On a backlog
/// worker another worker starts in its place and the lane of new compute pipelines goes to the
/// next one (`Backlog::built`), so the backlog does not stand still behind it; this thread ends
/// after the build. (A normal worker's wait has main's `parked_for_big_compile` instead; on a
/// `xcl-pso-pre-*` worker this does nothing.)
pub(super) fn build_parks() {
    let Some(new_compute) = PRELOAD_STEP.with(|s| s.take()) else { return };
    let Some(q) = preload_backlog() else { return };
    q.list.lock().unwrap().built(new_compute);
    q.cv.notify_one();
    spawn_backlog_worker(q);
}

struct RunningBuild<'a> {
    build: &'a Build,
    preload: bool,
}

impl Drop for RunningBuild<'_> {
    fn drop(&mut self) {
        if self.preload {
            // (under the lock: `want` does not raise this thread after it)
            let mut demand = self.build.demand.lock().unwrap();
            demand.preload_thread = 0;
            if self.build.wanted.load(Ordering::Acquire) {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
            }
        }
        BUILDING.with(|b| *b.borrow_mut() = None);
    }
}

struct RestoreBuilding(Option<Arc<Build>>);

impl Drop for RestoreBuilding {
    fn drop(&mut self) {
        BUILDING.with(|b| *b.borrow_mut() = self.0.take());
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "unknown panic"
    }
}

/// Run a pipeline job without letting one bad shader strand every waiter on its shared build.
fn run_job(job: Job) -> usize {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
        Ok(real) => real,
        Err(payload) => {
            crate::xlog_error!("D3D12.X: panic while building pipeline: {}", panic_message(payload.as_ref()));
            0
        }
    }
}

/// A worker's pipeline build (`preload`: by one of the lowest-priority workers).
fn run_build(build: &Arc<Build>, job: Job) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_NORMAL};
    let preload = PRELOAD_WORKER.get();
    BUILDING.with(|b| *b.borrow_mut() = Some(build.clone()));
    crate::prof_zone!("async PSO build", if preload { "preload" } else { "on demand" });
    if preload {
        let mut d = build.demand.lock().unwrap();
        d.preload_thread = unsafe { GetCurrentThreadId() };
        if build.wanted.load(Ordering::Acquire) {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    let real = {
        let _running = RunningBuild { build, preload };
        run_timed_job(build, job)
    };
    complete(build, real);
}

/// `run_job` for `build`: when it started and whether its program was huge are noted for the logs
/// (`job_elapsed`, `note_huge`; the pipeline line counts huge lazy builds).
fn run_timed_job(build: &Build, job: Job) -> usize {
    JOB_START.with(|s| s.set(Some(std::time::Instant::now())));
    JOB_HUGE.with(|h| h.set(false));
    let real = run_job(job);
    JOB_START.with(|s| s.set(None));
    if JOB_HUGE.with(|h| h.replace(false)) {
        build.huge.store(true, Ordering::Relaxed);
    }
    real
}

impl Build {
    fn is_evictable(&self) -> bool { self.kind == PipelineKind::Compute }

    fn new(job: Job, sent: bool) -> Self {
        Self::new_keyed_kind(job, sent, None, None, PipelineKind::Unknown)
    }

    #[cfg(test)]
    fn new_keyed(
        job: Job,
        sent: bool,
        compute_key: Option<Arc<super::real::compute_build_key::ComputeBuildKey>>,
        recipe: Option<Recipe>,
    ) -> Self {
        Self::new_keyed_kind(job, sent, compute_key, recipe, PipelineKind::Unknown)
    }

    fn new_keyed_kind(
        job: Job,
        sent: bool,
        compute_key: Option<Arc<super::real::compute_build_key::ComputeBuildKey>>,
        recipe: Option<Recipe>,
        kind: PipelineKind,
    ) -> Self {
        Self {
            kind,
            compute_key,
            recipe,
            job: Mutex::new(Some(job)),
            sent: std::sync::atomic::AtomicBool::new(sent),
            tracked: AtomicBool::new(false),
            queue_state: AtomicU8::new(QUEUE_IDLE),
            real: AtomicUsize::new(0),
            one_shot_generation: Mutex::new(None),
            last_use_frame: AtomicU64::new(super::present::presents()),
            completed_frame: AtomicU64::new(0),
            eviction_timestamp_access: AtomicUsize::new(0),
            candidate_queued: AtomicBool::new(false),
            eviction: Mutex::new(None),
            completed: AtomicBool::new(false),
            done: Mutex::new(false),
            cv: Condvar::new(),
            orphan: Default::default(),
            owners: AtomicUsize::new(1),
            wanted: AtomicBool::new(false),
            by_submit: AtomicBool::new(false),
            demand: Default::default(),
            queued: Mutex::new(None),
            record: 0,
            lazy: false,
            used: AtomicBool::new(false),
            late: AtomicBool::new(false),
            huge: AtomicBool::new(false),
            warned: AtomicBool::new(false),
            warm: Mutex::new(None),
        }
    }

    /// What the submit that made the build knows of it: made by a lazy submit, the compute
    /// pipeline's console program (`record`, 0: not known) and its translation ahead (`warm`).
    fn with_submit(mut self, lazy: bool, record: u64, warm: Option<Warm>) -> Self {
        self.lazy = lazy;
        self.record = record;
        self.warm = Mutex::new(warm);
        self
    }

    fn note_queued(&self) {
        if self.tracked.load(Ordering::Acquire) {
            // Publish the count before ACTIVE: completion may run as soon as the queue accepts
            // the job and must never subtract an entry which has not been added yet.
            QUEUED_BUILDS.fetch_add(1, Ordering::Relaxed);
            if !mark_queued(&self.queue_state) {
                QUEUED_BUILDS.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    fn finish_queued(&self) {
        if mark_queue_done(&self.queue_state) {
            QUEUED_BUILDS.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Publish terminal state after the caller has installed the result while holding `done`.
    /// The release flag is observational only; waiters and exact owner transitions keep using
    /// the mutex so the condvar predicate and generation lifetime cannot be separated.
    fn publish_completed(&self, mut done: std::sync::MutexGuard<'_, bool>) {
        debug_assert!(!*done);
        *done = true;
        self.finish_queued();
        self.completed.store(true, Ordering::Release);
        self.cv.notify_all();
    }

    fn promote(&self, priority: Priority) {
        if let Some(job) = self.queued.lock().unwrap().as_ref() {
            let _ = job.promote(priority);
        }
    }

    fn cancel_queued(&self) {
        if let Some(job) = self.queued.lock().unwrap().take() {
            let _ = job.cancel();
        }
    }

    /// One of the title's threads waits or a recorded submit will wait for it: the worker
    /// building it goes on now, at a normal priority.
    fn want(&self) {
        self.add_waiter(&mut self.demand.lock().unwrap());
        self.wanted_now();
    }

    /// What a new waiter starts besides its count (not under the `demand` lock: `promote` takes
    /// the queue's).
    fn wanted_now(&self) {
        let _ = start_normal_workers();
        self.promote(Priority::Foreground);
    }

    /// (under the `demand` lock)
    fn add_waiter(&self, d: &mut Demand) {
        d.waiters += 1;
        if d.waiters == 1 {
            self.wanted.store(true, Ordering::Release);
            Self::set_priority(d, windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL);
        }
    }

    /// A bind that waited for it ran out of its budget and goes on without it: when nobody else
    /// waits, the build is no longer wanted – back at the lowest priority on a preload worker and
    /// behind the huge-pipeline gate as an unwanted one (`real::big_compile_turn`): it waits for
    /// free commit and for the unwanted slots, where a wanted build skips the commit wait and has
    /// more slots, and many of those at once ran the process out of commit. (A job already
    /// promoted to `Foreground` keeps its place in the queue.)
    fn unwant(&self) {
        let mut d = self.demand.lock().unwrap();
        d.waiters = d.waiters.saturating_sub(1);
        if d.waiters == 0 {
            self.wanted.store(false, Ordering::Release);
            Self::set_priority(&d, windows_sys::Win32::System::Threading::THREAD_PRIORITY_LOWEST);
        }
    }

    /// The wait that ran out of its budget was the one a submit had counted ahead (`want`): that
    /// one leaves too.
    fn unwant_ahead(&self) {
        let ahead = std::mem::take(&mut self.demand.lock().unwrap().ahead);
        if ahead {
            self.unwant();
        }
    }

    /// The preload worker building it, if one does, at `priority` (under the `demand` lock: the
    /// worker does not end its build meanwhile).
    fn set_priority(d: &Demand, priority: windows_sys::Win32::System::Threading::THREAD_PRIORITY) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_SET_LIMITED_INFORMATION};
        if d.preload_thread == 0 {
            return;
        }
        unsafe {
            let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
            if !h.is_null() {
                SetThreadPriority(h, priority);
                CloseHandle(h);
            }
        }
    }
}

impl Build {
    fn is_compute(&self) -> bool {
        matches!(self.kind, PipelineKind::Compute | PipelineKind::ConsoleListCompute)
    }

    /// Nobody has taken it yet: not sent to the workers (a bind wants it: they build it), not
    /// released, not being built.
    fn unclaimed(&self) -> bool {
        !self.sent.load(Ordering::Acquire) && self.untaken()
    }

    /// Not released and not being built (sent or not).
    fn untaken(&self) -> bool {
        !self.orphan.load(Ordering::Acquire) && self.job.lock().unwrap().is_some()
    }
}

fn mark_queued(state: &AtomicU8) -> bool {
    state.compare_exchange(QUEUE_IDLE, QUEUE_ACTIVE, Ordering::AcqRel, Ordering::Acquire).is_ok()
}

fn mark_queue_done(state: &AtomicU8) -> bool {
    state.swap(QUEUE_DONE, Ordering::AcqRel) == QUEUE_ACTIVE
}

impl Pending {
    fn note_use(&self) {
        self.note_use_at(super::present::presents());
    }

    fn note_use_at(&self, frame: u64) {
        self.build.used.store(true, Ordering::Relaxed);
        if super::memory_report::enabled() || self.build.recipe.is_some() {
            let _writer = self.build.eviction_timestamp_writer();
            self.build.last_use_frame.fetch_max(frame, Ordering::Relaxed);
        }
        let event = super::pso_manifest::Event::Use {
            key: self.key,
            compute: self.compute,
            ready: self.build.real.load(Ordering::Acquire) != 0,
            recipe: match self.manifest_recipe.load(Ordering::Acquire) { 0 => None, key => Some(key) },
        };
        if self.key != 0 && super::pso_manifest::note_first(&self.used, event) {
            if self.track_used {
                used::note(self.key);
                let on_use = self.on_use.lock().unwrap().take();
                if let Some(f) = on_use {
                    f();
                }
            }
        }
    }

    /// Record use of an exactly owned resident generation. The timestamp is published before
    /// its weak eviction identity, so a concurrent policy turn either observes the new age or
    /// is forced to abort by final timestamp validation.
    fn note_resident_use(&self) {
        self.note_resident_use_on(super::present::presents(), eviction_candidates());
    }

    fn note_resident_use_on<const N: usize>(
        &self, frame: u64, candidates: &CandidateQueue<Build, N>,
    ) {
        self.note_use_at(frame);
        if self.build.recipe.is_some() {
            note_eviction_candidate_identity_on(&self.build, candidates);
        }
    }
}

const EVICTION_TIMESTAMP_CLAIM: usize = 1usize << (usize::BITS - 1);

struct EvictionTimestampWriter<'a>(&'a AtomicUsize);
struct EvictionTimestampClaim<'a>(&'a AtomicUsize);

impl Drop for EvictionTimestampWriter<'_> {
    fn drop(&mut self) { self.0.fetch_sub(1, Ordering::Release); }
}

impl Drop for EvictionTimestampClaim<'_> {
    fn drop(&mut self) { self.0.store(0, Ordering::Release); }
}

impl Build {
    fn eviction_timestamp_writer(&self) -> EvictionTimestampWriter<'_> {
        loop {
            let readers = self.eviction_timestamp_access.load(Ordering::Acquire);
            if readers & EVICTION_TIMESTAMP_CLAIM == 0 && readers != EVICTION_TIMESTAMP_CLAIM - 1
                && self.eviction_timestamp_access.compare_exchange_weak(
                    readers, readers + 1, Ordering::Acquire, Ordering::Relaxed,
                ).is_ok() {
                return EvictionTimestampWriter(&self.eviction_timestamp_access);
            }
            std::thread::yield_now();
        }
    }

    fn claim_eviction_timestamps(&self) -> EvictionTimestampClaim<'_> {
        loop {
            let readers = self.eviction_timestamp_access.load(Ordering::Acquire);
            if readers & EVICTION_TIMESTAMP_CLAIM == 0
                && self.eviction_timestamp_access.compare_exchange_weak(
                    readers, readers | EVICTION_TIMESTAMP_CLAIM, Ordering::AcqRel, Ordering::Acquire,
                ).is_ok() {
                while self.eviction_timestamp_access.load(Ordering::Acquire) != EVICTION_TIMESTAMP_CLAIM {
                    std::thread::yield_now();
                }
                return EvictionTimestampClaim(&self.eviction_timestamp_access);
            }
            std::thread::yield_now();
        }
    }
}

const PENDING_SHARDS: usize = 64;

/// Title-visible stand-in address → its build. Bind resolution is called by every recording
/// thread, so unrelated pipelines must not queue on one process-wide map lock.
struct PendingRegistry<V> {
    shards: [Mutex<Option<HashMap<usize, V>>>; PENDING_SHARDS],
}

impl<V> PendingRegistry<V> {
    const fn new() -> Self {
        Self { shards: [const { Mutex::new(None) }; PENDING_SHARDS] }
    }

    fn insert(&self, key: usize, value: V) -> Option<V> {
        self.shards[pending_shard_index(key)].lock().unwrap().get_or_insert_with(HashMap::new).insert(key, value)
    }

    fn remove(&self, key: usize) -> Option<V> {
        self.shards[pending_shard_index(key)].lock().unwrap().as_mut().and_then(|map| map.remove(&key))
    }
}

impl<V: Clone> PendingRegistry<V> {
    fn get(&self, key: usize) -> Option<V> {
        self.shards[pending_shard_index(key)].lock().unwrap().as_ref()?.get(&key).cloned()
    }
}

#[inline]
fn pending_shard_index(key: usize) -> usize {
    let mut mixed = (key >> 4).wrapping_mul(0x9E37_79B1);
    mixed ^= mixed >> 16;
    mixed & (PENDING_SHARDS - 1)
}

static PENDING: PendingRegistry<Arc<Pending>> = PendingRegistry::new();

const IN_FLIGHT_SHARDS: usize = 64;
type ComputeBuildKey = super::real::compute_build_key::ComputeBuildKey;

/// Equal compute requests share only while their build is in flight. Weak values do not retain
/// completed builds; terminal removal normally releases their keys, while a dead weak left by an
/// interrupted lifecycle is cleaned when its exact key is next encountered. Independent keys do
/// not queue on one process-wide lock.
struct InFlightRegistry {
    shards: [Mutex<Option<HashMap<Arc<ComputeBuildKey>, Weak<Build>>>>; IN_FLIGHT_SHARDS],
}

impl InFlightRegistry {
    const fn new() -> Self {
        Self { shards: [const { Mutex::new(None) }; IN_FLIGHT_SHARDS] }
    }

    fn shard(&self, key: &ComputeBuildKey) -> &Mutex<Option<HashMap<Arc<ComputeBuildKey>, Weak<Build>>>> {
        &self.shards[in_flight_shard_index(key)]
    }

    /// Remove only this exact build. A late completion must not remove a replacement for the key.
    fn remove_exact(&self, key: &ComputeBuildKey, build: &Weak<Build>, on_removed: impl FnOnce()) -> bool {
        let removed = {
            let mut shard = self.shard(key).lock().unwrap();
            let Some(builds) = shard.as_mut() else { return false };
            if builds.get(key).is_some_and(|registered| Weak::ptr_eq(registered, build)) {
                on_removed();
                builds.remove_entry(key)
            } else {
                None
            }
        };
        let found = removed.is_some();
        drop(removed);
        found
    }
}

#[inline]
fn in_flight_shard_index(key: &ComputeBuildKey) -> usize {
    // DefaultHasher::new has fixed keys, unlike HashMap's randomized builder. Hash the canonical
    // key once so every equal request selects the same shard before the map performs full equality.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish() as usize & (IN_FLIGHT_SHARDS - 1)
}

static IN_FLIGHT: InFlightRegistry = InFlightRegistry::new();

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ComputeCoalescingStats {
    pub keyed_submissions: u64,
    pub coalescing_hits: u64,
    pub unique_keyed_builds: u64,
    pub current_in_flight: usize,
    pub peak_in_flight: usize,
}

struct CoalescingCounters {
    coalescing_hits: AtomicU64,
    unique_keyed_builds: AtomicU64,
    current_in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
}

impl CoalescingCounters {
    const fn new() -> Self {
        Self {
            coalescing_hits: AtomicU64::new(0),
            unique_keyed_builds: AtomicU64::new(0),
            current_in_flight: AtomicUsize::new(0),
            peak_in_flight: AtomicUsize::new(0),
        }
    }

    fn hit(&self) {
        self.coalescing_hits.fetch_add(1, Ordering::Relaxed);
    }

    fn unique(&self) {
        self.unique_keyed_builds.fetch_add(1, Ordering::Relaxed);
        let current = self.current_in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_in_flight.fetch_max(current, Ordering::Relaxed);
    }

    fn finished(&self, count: usize) {
        if count != 0 {
            self.current_in_flight.fetch_sub(count, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> ComputeCoalescingStats {
        let coalescing_hits = self.coalescing_hits.load(Ordering::Relaxed);
        let unique_keyed_builds = self.unique_keyed_builds.load(Ordering::Relaxed);
        ComputeCoalescingStats {
            keyed_submissions: coalescing_hits.saturating_add(unique_keyed_builds),
            coalescing_hits,
            unique_keyed_builds,
            current_in_flight: self.current_in_flight.load(Ordering::Relaxed),
            peak_in_flight: self.peak_in_flight.load(Ordering::Relaxed),
        }
    }
}

static COALESCING: CoalescingCounters = CoalescingCounters::new();

pub fn compute_coalescing_stats() -> ComputeCoalescingStats {
    COALESCING.snapshot()
}

/// Unique completed native pipelines retained behind title-visible stand-ins, by frames since
/// their last use. Measurement only: command lists do not retain bound D3D12 objects, so these
/// ages are not evidence that a pipeline is safe to release.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetentionStats {
    pub retained_reals: usize,
    pub unused_60_frames: usize,
    pub unused_600_frames: usize,
    pub oldest_unused_frames: u64,
}

fn retention_stats_from(
    now: u64,
    builds: impl IntoIterator<Item = (usize, usize, u64, u64)>,
) -> RetentionStats {
    let mut seen = std::collections::HashSet::new();
    let mut stats = RetentionStats::default();
    for (identity, real, last_use, completed) in builds {
        if real == 0 || !seen.insert(identity) {
            continue;
        }
        let age = now.saturating_sub(last_use.max(completed));
        stats.retained_reals += 1;
        stats.unused_60_frames += (age >= 60) as usize;
        stats.unused_600_frames += (age >= 600) as usize;
        stats.oldest_unused_frames = stats.oldest_unused_frames.max(age);
    }
    stats
}

/// Snapshot for the `XCL_ACTIVITY` thread. The registry locks are taken only by that observer;
/// bind-time tracking is a present counter load plus one relaxed atomic maximum.
pub fn retention_stats() -> RetentionStats {
    if !super::memory_report::enabled() {
        return RetentionStats::default();
    }
    let mut builds = Vec::new();
    for shard in &PENDING.shards {
        let entries = shard.lock().unwrap();
        builds.extend(entries.iter().flat_map(|map| map.values()).map(|pending| {
            let build = &pending.build;
            (
                Arc::as_ptr(build) as usize,
                build.real.load(Ordering::Acquire),
                build.last_use_frame.load(Ordering::Relaxed),
                build.completed_frame.load(Ordering::Relaxed),
            )
        }));
    }
    retention_stats_from(super::present::presents(), builds)
}

/// One point-in-time, pressure-ready resident selected off the bind thread. Exact detachment must
/// revalidate this snapshot inside the admission and timestamp transaction before taking owners.
#[allow(dead_code)]
struct EvictionCandidate {
    build: Arc<Build>,
    last_use_frame: u64,
    completed_frame: u64,
    clear_queued_on_drop: bool,
}

impl Drop for EvictionCandidate {
    fn drop(&mut self) {
        if self.clear_queued_on_drop {
            self.build.candidate_queued.store(false, Ordering::Release);
        }
    }
}

impl EvictionCandidate {
    fn try_requeue_on<const N: usize>(mut self, candidates: &CandidateQueue<Build, N>) -> Result<(), Self> {
        if candidates.push(Arc::downgrade(&self.build)).is_err() {
            return Err(self);
        }
        self.clear_queued_on_drop = false;
        Ok(())
    }

    /// Return a deferred candidate to the bounded weak journal without opening a duplicate-record
    /// window. If a crossing producer filled the freed slot, dropping this owner clears the flag
    /// so a later bind can retry instead of leaving the build permanently marked as queued.
    fn requeue_on<const N: usize>(self, candidates: &CandidateQueue<Build, N>) -> bool {
        self.try_requeue_on(candidates).is_ok()
    }

    fn into_held(mut self) -> HeldEvictionCandidate {
        let build = Arc::downgrade(&self.build);
        self.clear_queued_on_drop = false;
        HeldEvictionCandidate { build }
    }
}

/// A saturated observer must conserve the deduplicated identity without retaining the pipeline,
/// its rebuild recipe, or its native generation after the title releases them.
struct HeldEvictionCandidate { build: Weak<Build> }
enum HeldRequeue { Requeued, Held(HeldEvictionCandidate), Expired }

impl HeldEvictionCandidate {
    fn try_requeue_on<const N: usize>(self, candidates: &CandidateQueue<Build, N>) -> HeldRequeue {
        let Some(build) = self.build.upgrade() else { return HeldRequeue::Expired };
        if !build.is_evictable() {
            build.candidate_queued.store(false, Ordering::Release);
            return HeldRequeue::Expired;
        }
        drop(build);
        match candidates.push(self.build.clone()) {
            Ok(()) => HeldRequeue::Requeued,
            Err(_) => HeldRequeue::Held(self),
        }
    }
}

/// Recipe-enabled binds append at most one weak identity per build. Recipe retention is explicit,
/// or implied by a pressure-policy gate; ordinary binds pay no candidate check, queue operation,
/// registry scan, or allocation.
#[allow(dead_code)] // exercised by policy tests; production use resumes with pressure activation
fn note_eviction_candidate_on<const N: usize>(build: &Arc<Build>, frame: u64, candidates: &CandidateQueue<Build, N>) {
    if !build.is_evictable() {
        build.last_use_frame.fetch_max(frame, Ordering::Relaxed);
        return;
    }
    let _writer = build.eviction_timestamp_writer();
    build.last_use_frame.fetch_max(frame, Ordering::Relaxed);
    note_eviction_candidate_identity_on(build, candidates);
}

fn note_eviction_candidate_identity_on<const N: usize>(
    build: &Arc<Build>, candidates: &CandidateQueue<Build, N>,
) {
    if !build.is_evictable() {
        return;
    }
    if build.candidate_queued.load(Ordering::Acquire) {
        return;
    }
    if build.candidate_queued.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return;
    }
    if candidates.push(Arc::downgrade(build)).is_err() {
        build.candidate_queued.store(false, Ordering::Release);
    }
}

fn note_eviction_candidate_identity(build: &Arc<Build>) {
    note_eviction_candidate_identity_on(build, eviction_candidates());
}

fn resident_candidate(build: &Arc<Build>) -> Option<EvictionCandidate> {
    if !build.is_evictable() {
        return None;
    }
    let done = build.done.lock().unwrap();
    if !*done || build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
        return None;
    }
    let recipe = build.recipe.as_ref()?;
    let rebuild = recipe.0.rebuild.lock().unwrap();
    if rebuild.active || rebuild.waiters != 0 {
        return None;
    }
    let real = build.real.load(Ordering::Acquire);
    if real == 0 {
        return None;
    }
    let generation = recipe.0.generation.lock().unwrap();
    if generation.residency != Residency::Resident
        || generation.resident.as_ref().is_none_or(|resident| resident.raw != real)
    {
        return None;
    }
    let last_use_frame = build.last_use_frame.load(Ordering::Relaxed);
    let completed_frame = build.completed_frame.load(Ordering::Relaxed);
    drop(generation);
    drop(rebuild);
    drop(done);
    Some(EvictionCandidate {
        build: build.clone(), last_use_frame, completed_frame, clear_queued_on_drop: true,
    })
}

/// Drain at most `limit` journal records on a background policy turn. The weak allocation identity
/// prevents a released stand-in address from turning an old record into a new pipeline.
#[allow(dead_code)]
fn drain_eviction_candidates_on<const N: usize>(
    candidates: &CandidateQueue<Build, N>, limit: usize,
) -> Vec<EvictionCandidate> {
    let mut selected = Vec::new();
    for _ in 0..limit {
        let Some(build) = candidates.pop() else { break };
        let Some(build) = build.upgrade() else { continue };
        match resident_candidate(&build) {
            Some(candidate) => selected.push(candidate),
            None => {
                // A released/orphaned/active/nonresident build is not actionable. Allow a later
                // use after its state changes to journal it again.
                build.candidate_queued.store(false, Ordering::Release);
            }
        }
    }
    selected
}

#[allow(dead_code)]
fn drain_eviction_candidates(limit: usize) -> Vec<EvictionCandidate> {
    drain_eviction_candidates_on(eviction_candidates(), limit)
}

#[derive(Clone, Copy)]
struct EvictionSelectionLimits {
    max_scan: usize,
    max_select: usize,
    min_unused_frames: u64,
    min_resident_frames: u64,
}

/// One bounded, point-in-time policy candidate. Actual eviction must extend the exact
/// install-and-revalidate transaction through native detachment without reopening its race window.
#[allow(dead_code)]
struct SelectedEvictionCandidate {
    candidate: EvictionCandidate,
    unused_frames: u64,
    resident_frames: u64,
}

/// Frame counters practically never wrap. If they do, treating pre-wrap timestamps as recent is
/// conservative: a policy pauses eviction instead of assigning an enormous, false age.
fn conservative_frame_age(now: u64, since: u64) -> u64 {
    now.checked_sub(since).unwrap_or(0)
}

fn partition_eviction_candidates(
    candidates: Vec<EvictionCandidate>, now: u64, limits: EvictionSelectionLimits,
) -> (Vec<SelectedEvictionCandidate>, Vec<EvictionCandidate>) {
    if limits.max_select == 0 {
        return (Vec::new(), candidates);
    }
    let mut selected = Vec::new();
    let mut deferred = Vec::new();
    for candidate in candidates {
        // A bind after the journal record was drained still updates this atomic while the
        // candidate's dedup flag prevents another record. Use that latest point-in-time age,
        // not the older snapshot captured by `resident_candidate`.
        let last_use = candidate.build.last_use_frame.load(Ordering::Relaxed);
        let completed = candidate.build.completed_frame.load(Ordering::Relaxed);
        let unused_frames = conservative_frame_age(now, last_use);
        let resident_frames = conservative_frame_age(now, completed);
        if unused_frames >= limits.min_unused_frames && resident_frames >= limits.min_resident_frames {
            selected.push(SelectedEvictionCandidate { candidate, unused_frames, resident_frames });
        } else {
            deferred.push(candidate);
        }
    }
    selected.sort_unstable_by(|a, b| {
        b.unused_frames
            .cmp(&a.unused_frames)
            .then_with(|| b.resident_frames.cmp(&a.resident_frames))
            .then_with(|| {
                (Arc::as_ptr(&a.candidate.build) as usize).cmp(&(Arc::as_ptr(&b.candidate.build) as usize))
            })
    });
    if selected.len() > limits.max_select {
        deferred.extend(selected.drain(limits.max_select..).map(|selected| selected.candidate));
    }
    (selected, deferred)
}

/// Inspect and rank a bounded amount of journalled work on a future background policy turn. This
/// remains deliberately uncalled: pressure admission and the eviction/rehydration trigger are
/// separate stages.
#[allow(dead_code)]
fn select_eviction_candidates_on<const N: usize>(
    candidates: &CandidateQueue<Build, N>, now: u64, limits: EvictionSelectionLimits,
) -> Vec<SelectedEvictionCandidate> {
    if limits.max_scan == 0 || limits.max_select == 0 {
        return Vec::new();
    }
    let (selected, deferred) =
        partition_eviction_candidates(drain_eviction_candidates_on(candidates, limits.max_scan), now, limits);
    for candidate in deferred {
        let _ = candidate.requeue_on(candidates);
    }
    selected
}

#[allow(dead_code)]
fn select_eviction_candidates(limits: EvictionSelectionLimits) -> Vec<SelectedEvictionCandidate> {
    select_eviction_candidates_on(eviction_candidates(), super::present::presents(), limits)
}

/// Pure admission for the background eviction/rehydration policy. It deliberately consumes
/// an explicit memory-report sample: sampling DXGI here would put adapter work on whichever thread
/// asked for a decision. Its default remains disabled.
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct PressureAdmissionConfig {
    enabled: bool,
    low_basis_points: u16,
    high_basis_points: u16,
    max_sample_age_turns: u64,
    max_evictions_per_turn: usize,
    rehydrate_backoff_turns: u64,
    max_rehydrate_backoff_turns: u64,
}

impl Default for PressureAdmissionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            low_basis_points: 7_500,
            high_basis_points: 9_000,
            max_sample_age_turns: 1,
            max_evictions_per_turn: 1,
            rehydrate_backoff_turns: 2,
            max_rehydrate_backoff_turns: 64,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct PressureSample {
    observed_turn: u64,
    segment: super::memory_report::Segment,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum PressureMode {
    #[default]
    Idle,
    Evicting,
    Rehydrating,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum PressurePolicyMode {
    #[default]
    Shadow,
    Active,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PressureDecision {
    max_evictions: usize,
    allow_rehydrate: bool,
}

#[allow(dead_code)]
struct PressureAdmission {
    config: PressureAdmissionConfig,
    mode: PressureMode,
    last_observed_turn: Option<u64>,
    accepted_policy_turn: Option<u64>,
    eviction_budget_turn: Option<u64>,
    evictions_succeeded: usize,
    retry_rehydrate_at: Option<u64>,
    next_rehydrate_backoff: u64,
}

const PRESSURE_MAX_TURN_HORIZON: u64 = u64::MAX / 2;

/// Serial turns practically never differ by half their range. Treat a small wrapped difference as
/// forward progress so one wrap does not make every subsequent pressure sample permanently stale.
fn pressure_turn_after(turn: u64, previous: u64) -> bool {
    let difference = turn.wrapping_sub(previous);
    difference != 0 && difference <= PRESSURE_MAX_TURN_HORIZON
}

#[allow(dead_code)]
impl PressureAdmission {
    fn new(config: PressureAdmissionConfig) -> Option<Self> {
        (config.low_basis_points < config.high_basis_points
            && config.high_basis_points <= 10_000
            && config.max_sample_age_turns <= PRESSURE_MAX_TURN_HORIZON
            && config.rehydrate_backoff_turns != 0
            && config.rehydrate_backoff_turns <= config.max_rehydrate_backoff_turns
            && config.max_rehydrate_backoff_turns <= PRESSURE_MAX_TURN_HORIZON)
            .then_some(Self {
                next_rehydrate_backoff: config.rehydrate_backoff_turns,
                config,
                mode: PressureMode::Idle,
                last_observed_turn: None,
                accepted_policy_turn: None,
                eviction_budget_turn: None,
                evictions_succeeded: 0,
                retry_rehydrate_at: None,
            })
    }

    fn decide(&mut self, now: u64, sample: Option<PressureSample>) -> PressureDecision {
        if !self.config.enabled {
            return PressureDecision::default();
        }
        let Some(sample) = sample else { return PressureDecision::default() };
        if sample.segment.budget == 0
            || pressure_turn_after(sample.observed_turn, now)
            || now.wrapping_sub(sample.observed_turn) > self.config.max_sample_age_turns
            || self.last_observed_turn.is_some_and(|last| !pressure_turn_after(sample.observed_turn, last))
            || self.accepted_policy_turn.is_some_and(|last| {
                now != last && !pressure_turn_after(now, last)
            })
        {
            return PressureDecision::default();
        }
        self.last_observed_turn = Some(sample.observed_turn);
        self.accepted_policy_turn = Some(now);
        let usage = u128::from(sample.segment.usage) * 10_000;
        let budget = u128::from(sample.segment.budget);
        if usage >= budget * u128::from(self.config.high_basis_points) {
            self.mode = PressureMode::Evicting;
        } else if usage <= budget * u128::from(self.config.low_basis_points)
            && self.mode == PressureMode::Evicting
        {
            self.mode = PressureMode::Rehydrating;
        }
        match self.mode {
            PressureMode::Idle => PressureDecision::default(),
            PressureMode::Evicting => {
                if self.eviction_budget_turn != Some(now) {
                    self.eviction_budget_turn = Some(now);
                    self.evictions_succeeded = 0;
                }
                let max_evictions = self.config.max_evictions_per_turn.saturating_sub(self.evictions_succeeded);
                PressureDecision { max_evictions, allow_rehydrate: false }
            }
            PressureMode::Rehydrating => PressureDecision {
                max_evictions: 0,
                allow_rehydrate: self.retry_rehydrate_at.is_none_or(|retry| {
                    now == retry || pressure_turn_after(now, retry)
                }),
            },
        }
    }

    /// Charge only native pipelines which the exact transaction actually detached. Scans and
    /// failed attempts have their own observer budgets and must not consume this allowance.
    fn note_evictions(&mut self, now: u64, succeeded: usize) {
        if self.mode != PressureMode::Evicting || self.eviction_budget_turn != Some(now) {
            return;
        }
        self.evictions_succeeded = self.evictions_succeeded
            .saturating_add(succeeded)
            .min(self.config.max_evictions_per_turn);
    }

    fn note_rehydrate_result(&mut self, now: u64, succeeded: bool) {
        if self.mode != PressureMode::Rehydrating {
            return;
        }
        if succeeded {
            self.retry_rehydrate_at = None;
            self.next_rehydrate_backoff = self.config.rehydrate_backoff_turns;
            return;
        }
        let delay = self.next_rehydrate_backoff;
        self.retry_rehydrate_at = Some(now.wrapping_add(delay));
        self.next_rehydrate_backoff = delay.saturating_mul(2).min(self.config.max_rehydrate_backoff_turns);
    }

    fn note_rehydration_complete(&mut self) {
        if self.mode == PressureMode::Rehydrating {
            self.mode = PressureMode::Idle;
            self.retry_rehydrate_at = None;
            self.next_rehydrate_backoff = self.config.rehydrate_backoff_turns;
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LedgerNativeGeneration {
    id: u64,
    identity: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvictedBuildState {
    Reserved,
    Evicted,
    RehydrateQueued,
    Restored,
    Released,
    Poisoned,
}

impl EvictedBuildState {
    fn terminal(self) -> bool {
        matches!(self, Self::Restored | Self::Released | Self::Poisoned)
    }
}

#[allow(dead_code)]
#[derive(Clone)]
struct EvictedBuildToken {
    ledger_generation: u64,
    ledger_instance: u64,
    slot: usize,
    slot_generation: u64,
    serial: u64,
    build: Weak<Build>,
    native: LedgerNativeGeneration,
}

impl EvictedBuildToken {
    fn exact(&self, other: &Self) -> bool {
        self.ledger_generation == other.ledger_generation
            && self.ledger_instance == other.ledger_instance
            && self.slot == other.slot
            && self.slot_generation == other.slot_generation
            && self.serial == other.serial
            && Weak::ptr_eq(&self.build, &other.build)
            && self.native == other.native
    }

    fn belongs_to(&self, build: &Build) -> bool {
        std::ptr::eq(self.build.as_ptr(), build)
    }
}

#[allow(dead_code)]
impl Build {
    fn eviction_token(&self) -> Option<EvictedBuildToken> {
        self.eviction.lock().unwrap().clone()
    }

    fn install_eviction_token(&self, token: &EvictedBuildToken) -> bool {
        if !token.belongs_to(self) {
            return false;
        }
        let mut installed = self.eviction.lock().unwrap();
        if installed.is_some() {
            return false;
        }
        *installed = Some(token.clone());
        true
    }

    fn clear_eviction_token(&self, token: &EvictedBuildToken) -> bool {
        let mut installed = self.eviction.lock().unwrap();
        if installed.as_ref().is_none_or(|current| !current.exact(token)) {
            return false;
        }
        *installed = None;
        true
    }
}

struct EvictedBuildEntry {
    token: EvictedBuildToken,
    state: EvictedBuildState,
}

struct EvictedBuildSlot {
    generation: u64,
    entry: Option<EvictedBuildEntry>,
}

struct EvictedBuildLedgerInner {
    serial: u64,
    exhausted: bool,
    slots: Box<[EvictedBuildSlot]>,
}

/// A future eviction path must reserve and arm one exact record before detaching native owners.
/// This remains private and unconstructed; snapshots contain only weak identities and hold no
/// ledger lock while a later pressure worker inspects Build or D3D state.
#[allow(dead_code)]
struct EvictedBuildLedger<const N: usize> {
    generation: u64,
    instance: u64,
    inner: Mutex<EvictedBuildLedgerInner>,
}

#[allow(dead_code)]
#[derive(Clone)]
struct EvictedBuildRecord {
    token: EvictedBuildToken,
    state: EvictedBuildState,
}

static NEXT_EVICTED_LEDGER_INSTANCE: AtomicU64 = AtomicU64::new(1);

fn allocate_evicted_ledger_instance(counter: &AtomicU64) -> Option<u64> {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.checked_add(1)?;
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Some(current),
            Err(actual) => current = actual,
        }
    }
}

#[allow(dead_code)]
impl<const N: usize> EvictedBuildLedger<N> {
    fn new(generation: u64) -> Option<Self> {
        if N == 0 || generation == 0 {
            return None;
        }
        let instance = allocate_evicted_ledger_instance(&NEXT_EVICTED_LEDGER_INSTANCE)?;
        Some(Self {
            generation,
            instance,
            inner: Mutex::new(EvictedBuildLedgerInner {
                serial: 0,
                exhausted: false,
                slots: (0..N)
                    .map(|_| EvictedBuildSlot { generation: 0, entry: None })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            }),
        })
    }

    fn reserve<'a>(
        &'a self, build: &Arc<Build>, native: LedgerNativeGeneration,
    ) -> Option<EvictedBuildReservation<'a, N>> {
        let mut inner = self.inner.lock().unwrap();
        if inner.exhausted {
            return None;
        }
        let slot = inner.slots.iter().position(|slot| slot.entry.is_none())?;
        let Some(serial) = inner.serial.checked_add(1) else {
            inner.exhausted = true;
            return None;
        };
        let Some(slot_generation) = inner.slots[slot].generation.checked_add(1) else {
            inner.exhausted = true;
            return None;
        };
        let token = EvictedBuildToken {
            ledger_generation: self.generation,
            ledger_instance: self.instance,
            slot,
            slot_generation,
            serial,
            build: Arc::downgrade(build),
            native,
        };
        inner.serial = serial;
        inner.slots[slot].generation = slot_generation;
        inner.slots[slot].entry = Some(EvictedBuildEntry { token: token.clone(), state: EvictedBuildState::Reserved });
        Some(EvictedBuildReservation { ledger: self, token, armed: false })
    }

    fn set_state(&self, token: &EvictedBuildToken, from: EvictedBuildState, to: EvictedBuildState) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.slots.get_mut(token.slot).and_then(|slot| slot.entry.as_mut()) else { return false };
        if entry.state != from || !entry.token.exact(token) {
            return false;
        }
        entry.state = to;
        true
    }

    fn rehydrate_queued(&self, token: &EvictedBuildToken) -> bool {
        self.set_state(token, EvictedBuildState::Evicted, EvictedBuildState::RehydrateQueued)
    }

    fn rehydrate_retry(&self, token: &EvictedBuildToken) -> bool {
        self.set_state(token, EvictedBuildState::RehydrateQueued, EvictedBuildState::Evicted)
    }

    fn state(&self, token: &EvictedBuildToken) -> Option<EvictedBuildState> {
        let inner = self.inner.lock().unwrap();
        let entry = inner.slots.get(token.slot)?.entry.as_ref()?;
        entry.token.exact(token).then_some(entry.state)
    }

    fn finish(&self, token: &EvictedBuildToken, terminal: EvictedBuildState) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.slots.get_mut(token.slot).and_then(|slot| slot.entry.as_mut()) else { return false };
        if !entry.token.exact(token) {
            return false;
        }
        let allowed = match terminal {
            EvictedBuildState::Restored => entry.state == EvictedBuildState::RehydrateQueued,
            EvictedBuildState::Released | EvictedBuildState::Poisoned => {
                matches!(entry.state, EvictedBuildState::Evicted | EvictedBuildState::RehydrateQueued)
            }
            _ => false,
        };
        if !allowed { return false }
        entry.state = terminal;
        true
    }

    fn retire(&self, token: &EvictedBuildToken) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(slot) = inner.slots.get_mut(token.slot) else { return false };
        if slot.entry.as_ref().is_none_or(|entry| !entry.state.terminal() || !entry.token.exact(token)) {
            return false;
        }
        slot.entry = None;
        true
    }

    fn records(&self, limit: usize) -> Vec<EvictedBuildRecord> {
        let mut inner = self.inner.lock().unwrap();
        let mut records = Vec::new();
        for slot in &mut inner.slots {
            let Some(entry) = slot.entry.as_mut() else { continue };
            if !entry.state.terminal() && entry.token.build.strong_count() == 0 {
                entry.state = EvictedBuildState::Released;
            }
            if records.len() == limit {
                break;
            }
            records.push(EvictedBuildRecord { token: entry.token.clone(), state: entry.state });
        }
        records
    }
}

#[allow(dead_code)]
struct EvictedBuildReservation<'a, const N: usize> {
    ledger: &'a EvictedBuildLedger<N>,
    token: EvictedBuildToken,
    armed: bool,
}

#[allow(dead_code)]
impl<const N: usize> EvictedBuildReservation<'_, N> {
    /// Arm before native detachment. From this point, dropping either this guard or its returned
    /// token cannot erase the fail-closed record; a failed detach must finish it as Poisoned.
    fn arm_detach(mut self) -> Option<EvictedBuildToken> {
        if !self.ledger.set_state(&self.token, EvictedBuildState::Reserved, EvictedBuildState::Evicted) {
            return None;
        }
        self.armed = true;
        Some(self.token.clone())
    }
}

impl<const N: usize> Drop for EvictedBuildReservation<'_, N> {
    fn drop(&mut self) {
        if !self.armed {
            let mut inner = self.ledger.inner.lock().unwrap();
            if let Some(slot) = inner.slots.get_mut(self.token.slot) {
                if slot.entry.as_ref().is_some_and(|entry| {
                    entry.state == EvictedBuildState::Reserved && entry.token.exact(&self.token)
                }) {
                    slot.entry = None;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PressureObservation {
    policy: PressurePolicyMode,
    mode: PressureMode,
    scanned: usize,
    eligible: usize,
    deferred: usize,
    discarded: usize,
    requeue_saturated: usize,
    attempted: usize,
    evicted: usize,
    prepared: usize,
    recent: usize,
    changed: usize,
    busy: usize,
    released: usize,
    ledger_full: usize,
}

impl PressureObservation {
    fn note_preflight(&mut self, result: EvictionPreflight) {
        self.attempted += 1;
        match result {
            EvictionPreflight::Prepared => self.prepared += 1,
            EvictionPreflight::Recent => self.recent += 1,
            EvictionPreflight::Changed => self.changed += 1,
            EvictionPreflight::Busy => self.busy += 1,
            EvictionPreflight::Released => self.released += 1,
            EvictionPreflight::LedgerFull => self.ledger_full += 1,
        }
    }

    fn attempt_outcomes(self) -> usize {
        self.evicted + self.prepared + self.recent + self.changed + self.busy
            + self.released + self.ledger_full
    }

    fn attempts_accounted(self) -> bool { self.attempted == self.attempt_outcomes() }

    fn line(self) -> String {
        debug_assert!(self.attempts_accounted());
        format!(
            "pipeline pressure observation: {:?} {:?}; {} scanned, {} eligible, {} deferred, {} discarded, {} requeue saturated; {} attempted, {} evicted; preflight {} prepared, {} recent, {} changed, {} busy, {} released, {} ledger full",
            self.policy, self.mode, self.scanned, self.eligible, self.deferred, self.discarded, self.requeue_saturated,
            self.attempted, self.evicted, self.prepared, self.recent, self.changed, self.busy,
            self.released, self.ledger_full,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvictionPreflight { Prepared, Recent, Changed, Busy, Released, LedgerFull }

struct PressureObserver {
    admission: PressureAdmission,
    limits: EvictionSelectionLimits,
    held: Vec<HeldEvictionCandidate>,
    work_budget_turn: Option<u64>,
    scans_issued: usize,
    attempts_issued: usize,
}

enum PressureAttempt {
    Skipped(SelectedEvictionCandidate),
    Aborted(EvictionPreflight, EvictionCandidate),
    Detached(DetachedExactEviction),
}

impl PressureObserver {
    fn new(config: PressureAdmissionConfig, limits: EvictionSelectionLimits) -> Option<Self> {
        PressureAdmission::new(config).map(|admission| Self {
            admission, held: Vec::with_capacity(limits.max_scan), limits,
            work_budget_turn: None, scans_issued: 0, attempts_issued: 0,
        })
    }

    /// Observe and immediately requeue every actionable candidate. This deliberately cannot evict:
    /// the diagnostic validates pressure admission and candidate flow without changing residency.
    fn observe_on<const N: usize>(
        &mut self,
        candidates: &CandidateQueue<Build, N>,
        turn: u64,
        frame: u64,
        sample: Option<PressureSample>,
    ) -> PressureObservation {
        self.observe_on_after_drain(candidates, turn, frame, sample, || {})
    }

    fn observe_exact_on<const N: usize, const E: usize, const R: usize>(
        &mut self,
        candidates: &CandidateQueue<Build, N>,
        turn: u64,
        frame: u64,
        sample: Option<PressureSample>,
        coordinator: &RestorationCoordinator<E, R>,
    ) -> PressureObservation {
        let limits = self.limits;
        self.observe_on_with_attempt(candidates, turn, frame, sample, || {}, PressurePolicyMode::Shadow,
            |selected| {
                let result = preflight_exact_eviction_on(&selected, frame, limits, coordinator, || {});
                PressureAttempt::Aborted(result, selected.candidate)
            }).0
    }

    fn evict_exact_on<const N: usize, const E: usize, const R: usize>(
        &mut self,
        candidates: &CandidateQueue<Build, N>,
        turn: u64,
        frame: u64,
        sample: Option<PressureSample>,
        coordinator: &RestorationCoordinator<E, R>,
    ) -> (PressureObservation, Option<DetachedExactEviction>) {
        let limits = self.limits;
        self.observe_on_with_attempt(candidates, turn, frame, sample, || {}, PressurePolicyMode::Active,
            |selected| match detach_exact_eviction_on(
                selected, frame, limits, coordinator, || {}, || {},
            ) {
                ExactDetachAttempt::Aborted(result, candidate) => {
                    PressureAttempt::Aborted(result, candidate)
                }
                ExactDetachAttempt::Detached(detached) => PressureAttempt::Detached(detached),
            })
    }

    fn observe_on_after_drain<const N: usize>(
        &mut self,
        candidates: &CandidateQueue<Build, N>,
        turn: u64,
        frame: u64,
        sample: Option<PressureSample>,
        after_drain: impl FnOnce(),
    ) -> PressureObservation {
        self.observe_on_with_preflight(candidates, turn, frame, sample, after_drain, |_| None)
    }

    fn observe_on_with_preflight<const N: usize>(
        &mut self,
        candidates: &CandidateQueue<Build, N>,
        turn: u64,
        frame: u64,
        sample: Option<PressureSample>,
        after_drain: impl FnOnce(),
        mut preflight: impl FnMut(&SelectedEvictionCandidate) -> Option<EvictionPreflight>,
    ) -> PressureObservation {
        self.observe_on_with_attempt(candidates, turn, frame, sample, after_drain,
            PressurePolicyMode::Shadow, |selected| match preflight(&selected) {
                Some(result) => PressureAttempt::Aborted(result, selected.candidate),
                None => PressureAttempt::Skipped(selected),
            }).0
    }

    fn observe_on_with_attempt<const N: usize>(
        &mut self,
        candidates: &CandidateQueue<Build, N>,
        turn: u64,
        frame: u64,
        sample: Option<PressureSample>,
        after_drain: impl FnOnce(),
        policy: PressurePolicyMode,
        mut attempt: impl FnMut(SelectedEvictionCandidate) -> PressureAttempt,
    ) -> (PressureObservation, Option<DetachedExactEviction>) {
        let decision = self.admission.decide(turn, sample);
        let mut observation = PressureObservation {
            policy, mode: self.admission.mode, ..Default::default()
        };
        let mut detached = None;
        if self.admission.accepted_policy_turn == Some(turn) && self.work_budget_turn != Some(turn) {
            self.work_budget_turn = Some(turn);
            self.scans_issued = 0;
            self.attempts_issued = 0;
        }

        // Conserve exact weak identities until this one-scan backlog returns to the journal.
        let pending = std::mem::replace(&mut self.held, Vec::with_capacity(self.limits.max_scan));
        for candidate in pending {
            match candidate.try_requeue_on(candidates) {
                HeldRequeue::Requeued => {}
                HeldRequeue::Held(candidate) => self.held.push(candidate),
                HeldRequeue::Expired => observation.discarded += 1,
            }
        }
        observation.requeue_saturated = self.held.len();
        if !self.held.is_empty() || decision.max_evictions == 0 {
            return (observation, detached);
        }
        let limits = EvictionSelectionLimits {
            max_scan: self.limits.max_scan.saturating_sub(self.scans_issued),
            max_select: self.limits.max_select.saturating_sub(self.attempts_issued),
            ..self.limits
        };
        let mut drained = Vec::new();
        let mut nonactionable = Vec::new();
        for _ in 0..limits.max_scan {
            let Some(weak) = candidates.pop() else { break };
            observation.scanned += 1;
            self.scans_issued += 1;
            let Some(build) = weak.upgrade() else {
                observation.discarded += 1;
                continue;
            };
            if !build.is_evictable() {
                build.candidate_queued.store(false, Ordering::Release);
                observation.discarded += 1;
                continue;
            }
            match resident_candidate(&build) {
                Some(candidate) => drained.push(candidate),
                None => nonactionable.push(HeldEvictionCandidate { build: weak }),
            }
        }
        after_drain();
        let (selected, deferred) = partition_eviction_candidates(drained, frame, limits);
        observation.eligible = selected.len();
        observation.deferred = deferred.len() + nonactionable.len();
        let mut actionable = Vec::with_capacity(selected.len() + deferred.len());
        for selected in selected {
            if policy == PressurePolicyMode::Active && detached.is_some() {
                actionable.push(selected.candidate);
                continue;
            }
            match attempt(selected) {
                PressureAttempt::Skipped(selected) => actionable.push(selected.candidate),
                PressureAttempt::Aborted(result, candidate) => {
                    observation.note_preflight(result);
                    self.attempts_issued += 1;
                    actionable.push(candidate);
                }
                PressureAttempt::Detached(owner) => {
                    observation.attempted += 1;
                    observation.evicted += 1;
                    self.attempts_issued += 1;
                    self.admission.note_evictions(turn, 1);
                    detached = Some(owner);
                }
            }
        }
        actionable.extend(deferred);
        for candidate in actionable {
            if let Err(candidate) = candidate.try_requeue_on(candidates) {
                self.held.push(candidate.into_held());
            }
        }
        for candidate in nonactionable {
            match candidate.try_requeue_on(candidates) {
                HeldRequeue::Requeued => {}
                HeldRequeue::Held(candidate) => self.held.push(candidate),
                HeldRequeue::Expired => observation.discarded += 1,
            }
        }
        observation.requeue_saturated = self.held.len();
        debug_assert!(observation.attempts_accounted());
        (observation, detached)
    }
}

fn pressure_gate_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| value.trim() == "1")
}

fn pressure_policy_mode_values(
    evict: Option<&str>, observe: Option<&str>,
) -> Option<PressurePolicyMode> {
    if pressure_gate_value(evict) {
        Some(PressurePolicyMode::Active)
    } else if pressure_gate_value(observe) {
        Some(PressurePolicyMode::Shadow)
    } else {
        None
    }
}

fn pressure_policy_mode() -> Option<PressurePolicyMode> {
    static MODE: OnceLock<Option<PressurePolicyMode>> = OnceLock::new();
    *MODE.get_or_init(|| pressure_policy_mode_values(
        std::env::var("XCL_PSO_PRESSURE_EVICT").ok().as_deref(),
        std::env::var("XCL_PSO_PRESSURE_OBSERVE").ok().as_deref(),
    ))
}

/// Whether the host should take the bounded two-second DXGI memory sample. This does not enable
/// the unrelated high-frequency diagnostics behind `XCL_ACTIVITY`.
pub(crate) fn pressure_sampling_enabled() -> bool { pressure_policy_mode().is_some() }

fn pressure_observer() -> Option<&'static Mutex<PressureObserver>> {
    static OBSERVER: OnceLock<Option<Mutex<PressureObserver>>> = OnceLock::new();
    OBSERVER.get_or_init(|| {
        let policy = pressure_policy_mode()?;
        debug_assert!(rebuild_recipes_enabled());
        let config = PressureAdmissionConfig {
            enabled: true,
            max_sample_age_turns: 0,
            max_evictions_per_turn: 1,
            ..Default::default()
        };
        let limits = EvictionSelectionLimits {
            max_scan: 32,
            max_select: 8,
            min_unused_frames: 600,
            min_resident_frames: 60,
        };
        match policy {
            PressurePolicyMode::Shadow => crate::xlog_info!(
                "D3D12.X: XCL_PSO_PRESSURE_OBSERVE=1 - shadowing compute pipeline pressure eviction"
            ),
            PressurePolicyMode::Active => crate::xlog_info!(
                "D3D12.X: XCL_PSO_PRESSURE_EVICT=1 - evicting old rebuildable compute pipelines under video-memory pressure"
            ),
        }
        PressureObserver::new(config, limits).map(Mutex::new)
    }).as_ref()
}

/// Called only by the existing activity observer after it has obtained and released the DXGI
/// memory snapshot. No title or submission thread samples the adapter or scans candidates.
pub(crate) fn observe_pressure_turn(
    turn: u64, local: Option<super::memory_report::Segment>,
) -> Option<String> {
    let observer = pressure_observer()?;
    let sample = local.map(|segment| PressureSample { observed_turn: turn, segment });
    let (observation, detached) = {
        let mut observer = observer.lock().unwrap();
        match pressure_policy_mode().expect("pressure observer without a policy") {
            PressurePolicyMode::Shadow => (observer.observe_exact_on(
                eviction_candidates(), turn, super::present::presents(), sample,
                pipeline_restoration_coordinator(),
            ), None),
            PressurePolicyMode::Active => observer.evict_exact_on(
                eviction_candidates(), turn, super::present::presents(), sample,
                pipeline_restoration_coordinator(),
            ),
        }
    };
    if let Some(detached) = detached { detached.retire_pressure(pressure_counters()) }
    Some(pressure_observation_line(observation, pipeline_pressure_stats()))
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RehydratePriority { Background, Foreground }

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RehydrateRequestReceipt { Refused, Queued { priority: RehydratePriority } }

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RehydrateCompletion { Reserved, Refused, Queued, Cancelled, Panicked, BuildFailed, Released,
    Succeeded { replacement: LedgerNativeGeneration } }

impl RehydrateCompletion {
    fn terminal(self) -> bool { !matches!(self, Self::Reserved | Self::Queued) }
}

#[allow(dead_code)]
#[derive(Clone)]
struct RehydrateAttemptToken { registry_instance: u64, slot: usize, slot_generation: u64,
    serial: u64, attempt_generation: u64, evicted: EvictedBuildToken }

mod rehydrate_success {
    use super::*;
    enum Authority { Current, Published }
    pub(super) struct Proof { attempt: RehydrateAttemptToken, build: Weak<Build>, recipe: Recipe,
        replacement: NativeGeneration, authority: Authority }
    impl Proof {
        pub(super) fn new(attempt: &RehydrateAttemptToken, build: &Arc<Build>,
            replacement: &NativeGeneration) -> Option<Self> {
            if !Weak::ptr_eq(&attempt.evicted.build, &Arc::downgrade(build)) { return None }
            let recipe = build.recipe.as_ref()?;
            let (current, residency) = recipe.pin_with_residency(build.real.load(Ordering::Acquire));
            (residency == Residency::Resident && current.as_ref().is_some_and(|current| current.same(replacement))).then(||
                Self { attempt: attempt.clone(), build: Arc::downgrade(build), recipe: recipe.clone(),
                    replacement: replacement.clone(), authority: Authority::Current })
        }
        pub(super) fn published(attempt: &RehydrateAttemptToken, build: &Build, recipe: &Recipe,
            replacement: &NativeGeneration) -> Option<Self> {
            if attempt.evicted.build.as_ptr() != build as *const Build
                || !Weak::ptr_eq(&replacement.recipe, &Arc::downgrade(&recipe.0))
                || build.recipe.as_ref().is_none_or(|owned| !Arc::ptr_eq(&owned.0, &recipe.0)) { return None }
            let (current, residency) = recipe.pin_with_residency(replacement.raw());
            if residency != Residency::Resident || current.as_ref().is_none_or(|current| !current.same(replacement)) { return None }
            Some(Self { attempt: attempt.clone(), build: attempt.evicted.build.clone(), recipe: recipe.clone(),
                replacement: replacement.clone(), authority: Authority::Published })
        }
        pub(super) fn valid_locked(&self, attempt: &RehydrateAttemptToken, build: &Arc<Build>) -> bool {
            if !self.attempt.exact(attempt) { return false }
            if !Weak::ptr_eq(&self.build, &Arc::downgrade(build)) { return false }
            let Some(recipe) = build.recipe.as_ref() else { return false };
            Arc::ptr_eq(&recipe.0, &self.recipe.0) && match self.authority {
                Authority::Current => recipe.0.resident_generation.load(Ordering::Acquire) == self.replacement.id(),
                Authority::Published => Weak::ptr_eq(&self.replacement.recipe, &Arc::downgrade(&recipe.0)),
            }
        }

        pub(super) fn replacement(&self) -> LedgerNativeGeneration { self.replacement.ledger_identity() }
        pub(super) fn pin(&self) -> NativeGeneration { self.replacement.clone() }
    }
}

impl RehydrateAttemptToken {
    fn exact(&self, other: &Self) -> bool {
        self.registry_instance == other.registry_instance
            && self.slot == other.slot
            && self.slot_generation == other.slot_generation
            && self.serial == other.serial
            && self.attempt_generation == other.attempt_generation
            && self.evicted.exact(&other.evicted)
    }
}

struct RehydrateReceiptEntry { token: RehydrateAttemptToken, state: RehydrateCompletion, priority: RehydratePriority }
struct RehydrateReceiptSlot { generation: u64, entry: Option<RehydrateReceiptEntry> }
struct RehydrateReceiptInner { serial: u64, exhausted: bool, slots: Box<[RehydrateReceiptSlot]> }

/// Bounded truth about asynchronous rehydration attempts. Queue submission and pipeline building
/// happen outside this lock; accepting a queue request records only Queued, never success.
#[allow(dead_code)]
struct RehydrateReceiptRegistry<const N: usize> { instance: u64, inner: Mutex<RehydrateReceiptInner> }

#[allow(dead_code)]
#[derive(Clone)]
struct RehydrateReceiptRecord { token: RehydrateAttemptToken, state: RehydrateCompletion, priority: RehydratePriority }

static NEXT_REHYDRATE_RECEIPT_INSTANCE: AtomicU64 = AtomicU64::new(1);

#[allow(dead_code)]
impl<const N: usize> RehydrateReceiptRegistry<N> {
    fn new() -> Option<Self> {
        if N == 0 { return None }
        let instance = allocate_evicted_ledger_instance(&NEXT_REHYDRATE_RECEIPT_INSTANCE)?;
        Some(Self { instance, inner: Mutex::new(RehydrateReceiptInner { serial: 0, exhausted: false,
            slots: (0..N).map(|_| RehydrateReceiptSlot { generation: 0, entry: None })
                .collect::<Vec<_>>().into_boxed_slice() }) })
    }

    fn reserve<'a>(&'a self, evicted: &EvictedBuildToken) -> Option<RehydrateReceiptReservation<'a, N>> {
        let mut inner = self.inner.lock().unwrap();
        self.reserve_locked(&mut inner, evicted)
    }

    fn reserve_unique<'a>(
        &'a self, evicted: &EvictedBuildToken, allow_new: bool,
    ) -> Result<Option<RehydrateReceiptReservation<'a, N>>, RehydrateReceiptRecord> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(entry) = inner.slots.iter().filter_map(|slot| slot.entry.as_ref())
            .find(|entry| entry.token.evicted.exact(evicted)) {
            return Err(RehydrateReceiptRecord {
                token: entry.token.clone(), state: entry.state, priority: entry.priority,
            });
        }
        if !allow_new { return Ok(None) }
        Ok(self.reserve_locked(&mut inner, evicted))
    }

    fn reserve_locked<'a>(
        &'a self, inner: &mut RehydrateReceiptInner, evicted: &EvictedBuildToken,
    ) -> Option<RehydrateReceiptReservation<'a, N>> {
        if inner.exhausted { return None }
        let slot = inner.slots.iter().position(|slot| slot.entry.is_none())?;
        let Some(serial) = inner.serial.checked_add(1) else {
            inner.exhausted = true;
            return None;
        };
        let Some(slot_generation) = inner.slots[slot].generation.checked_add(1) else {
            inner.exhausted = true;
            return None;
        };
        let token = RehydrateAttemptToken { registry_instance: self.instance, slot, slot_generation,
            serial, attempt_generation: serial, evicted: evicted.clone() };
        inner.serial = serial;
        inner.slots[slot].generation = slot_generation;
        inner.slots[slot].entry = Some(RehydrateReceiptEntry {
            token: token.clone(), state: RehydrateCompletion::Reserved, priority: RehydratePriority::Background,
        });
        Some(RehydrateReceiptReservation { registry: self, token, submitted: false })
    }

    fn publish(&self, token: &RehydrateAttemptToken, state: RehydrateCompletion) -> bool {
        self.publish_checked(token, state)
    }

    fn publish_checked(&self, token: &RehydrateAttemptToken, mut state: RehydrateCompletion) -> bool {
        if !state.terminal() || matches!(state, RehydrateCompletion::Succeeded { .. }) { return false }
        let live_build = token.evicted.build.upgrade();
        if live_build.is_none() { state = RehydrateCompletion::Released }
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.slots.get_mut(token.slot).and_then(|slot| slot.entry.as_mut()) else { return false };
        if !entry.token.exact(token) { return false }
        let allowed = match state {
            RehydrateCompletion::Refused => matches!(entry.state, RehydrateCompletion::Reserved | RehydrateCompletion::Queued),
            RehydrateCompletion::Released => matches!(entry.state, RehydrateCompletion::Reserved | RehydrateCompletion::Queued),
            RehydrateCompletion::Cancelled | RehydrateCompletion::Panicked
            | RehydrateCompletion::BuildFailed => entry.state == RehydrateCompletion::Queued,
            RehydrateCompletion::Succeeded { replacement } => entry.state == RehydrateCompletion::Queued
                && replacement != entry.token.evicted.native,
            _ => false,
        };
        if !allowed { return false }
        entry.state = state;
        true
    }

    fn publish_success(&self, token: &RehydrateAttemptToken, proof: &rehydrate_success::Proof) -> bool {
        let Some(build) = token.evicted.build.upgrade() else {
            return self.publish(token, RehydrateCompletion::Released);
        };
        let done = build.done.lock().unwrap();
        if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            drop(done);
            return self.publish(token, RehydrateCompletion::Released);
        }
        if !proof.valid_locked(token, &build) {
            return false;
        }
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.slots.get_mut(token.slot).and_then(|slot| slot.entry.as_mut()) else { return false };
        let replacement = proof.replacement();
        if entry.state != RehydrateCompletion::Queued || !entry.token.exact(token)
            || replacement == entry.token.evicted.native { return false }
        entry.state = RehydrateCompletion::Succeeded { replacement }; true
    }

    fn accept(&self, token: &RehydrateAttemptToken, priority: RehydratePriority) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.slots.get_mut(token.slot).and_then(|slot| slot.entry.as_mut()) else { return false };
        if entry.state != RehydrateCompletion::Reserved || !entry.token.exact(token) { return false }
        entry.state = RehydrateCompletion::Queued; entry.priority = priority; true
    }

    fn promote(&self, token: &RehydrateAttemptToken) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.slots.get_mut(token.slot).and_then(|slot| slot.entry.as_mut()) else { return false };
        if entry.state != RehydrateCompletion::Queued || !entry.token.exact(token) { return false }
        entry.priority = RehydratePriority::Foreground; true
    }

    fn records(&self, limit: usize) -> Vec<RehydrateReceiptRecord> {
        let mut inner = self.inner.lock().unwrap();
        let mut records = Vec::new();
        for slot in &mut inner.slots {
            let Some(entry) = slot.entry.as_mut() else { continue };
            if !entry.state.terminal() && entry.token.evicted.build.strong_count() == 0 {
                entry.state = RehydrateCompletion::Released;
            }
            if records.len() == limit { break }
            records.push(RehydrateReceiptRecord {
                token: entry.token.clone(), state: entry.state, priority: entry.priority,
            });
        }
        records
    }

    fn retire(&self, token: &RehydrateAttemptToken) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(slot) = inner.slots.get_mut(token.slot) else { return false };
        if slot.entry.as_ref().is_none_or(|entry| !entry.state.terminal() || !entry.token.exact(token)) {
            return false;
        }
        slot.entry = None;
        true
    }
}

#[allow(dead_code)]
struct RehydrateReceiptReservation<'a, const N: usize> { registry: &'a RehydrateReceiptRegistry<N>,
    token: RehydrateAttemptToken, submitted: bool }

#[allow(dead_code)]
impl<'a, const N: usize> RehydrateReceiptReservation<'a, N> {
    fn refused(mut self) -> Option<(RehydrateRequestReceipt, RehydrateAttemptToken)> {
        self.submitted = self.registry.publish(&self.token, RehydrateCompletion::Refused);
        self.submitted.then(|| (RehydrateRequestReceipt::Refused, self.token.clone()))
    }

    fn accepted(mut self, priority: RehydratePriority) -> Option<(RehydrateRequestReceipt, RehydrateQueuedReceipt<'a, N>)> {
        if !self.registry.accept(&self.token, priority) {
            return None;
        }
        self.submitted = true;
        let token = self.token.clone();
        Some((RehydrateRequestReceipt::Queued { priority }, RehydrateQueuedReceipt {
            registry: self.registry, token, completed: false,
        }))
    }

    fn accepted_token(mut self, priority: RehydratePriority) -> Option<RehydrateAttemptToken> {
        if !self.registry.accept(&self.token, priority) { return None }
        self.submitted = true;
        Some(self.token.clone())
    }
}

impl<const N: usize> Drop for RehydrateReceiptReservation<'_, N> {
    fn drop(&mut self) {
        if !self.submitted {
            let mut inner = self.registry.inner.lock().unwrap();
            if let Some(slot) = inner.slots.get_mut(self.token.slot) {
                if slot.entry.as_ref().is_some_and(|entry| {
                    entry.state == RehydrateCompletion::Reserved && entry.token.exact(&self.token)
                }) {
                    slot.entry = None;
                }
            }
        }
    }
}

#[allow(dead_code)]
struct RehydrateQueuedReceipt<'a, const N: usize> { registry: &'a RehydrateReceiptRegistry<N>,
    token: RehydrateAttemptToken, completed: bool }

#[allow(dead_code)]
impl<const N: usize> RehydrateQueuedReceipt<'_, N> {
    fn replacement(&self, build: &Arc<Build>, replacement: &NativeGeneration) -> Option<rehydrate_success::Proof> {
        rehydrate_success::Proof::new(&self.token, build, replacement)
    }

    fn complete(mut self, completion: RehydrateCompletion) -> bool {
        self.completed = self.registry.publish(&self.token, completion);
        self.completed
    }

    fn complete_success(mut self, proof: rehydrate_success::Proof) -> bool {
        self.completed = self.registry.publish_success(&self.token, &proof);
        self.completed
    }
}

impl<const N: usize> Drop for RehydrateQueuedReceipt<'_, N> {
    fn drop(&mut self) {
        if !self.completed {
            let completion = if std::thread::panicking() {
                RehydrateCompletion::Panicked
            } else {
                RehydrateCompletion::Cancelled
            };
            let _ = self.registry.publish(&self.token, completion);
        }
    }
}

#[allow(dead_code)]
#[derive(Clone)]
struct RestorationRetry { token: EvictedBuildToken }

#[allow(dead_code)]
/// Private and unwired: composes exact eviction identity with one queue job and its strong result.
struct RestorationCoordinator<const E: usize, const R: usize> {
    evicted: EvictedBuildLedger<E>,
    receipts: Arc<RehydrateReceiptRegistry<R>>,
    admission: Mutex<()>,
    attempts: Mutex<Vec<Arc<RestorationAttempt>>>,
    pressure_stats: Option<Arc<PipelinePressureCounters>>,
    #[cfg(test)]
    captured: Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>,
    #[cfg(test)]
    before_rebuild: Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>,
    #[cfg(test)]
    before_classified_request: Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>,
}

/// An exact ledger record installed on its build before native detachment can begin. Dropping an
/// uncommitted preparation clears only that record and poisons its ledger slot; committing leaves
/// both visible for a later restoration request.
#[allow(dead_code)]
struct InstalledEviction<'a, const E: usize, const R: usize> {
    coordinator: &'a RestorationCoordinator<E, R>,
    build: Arc<Build>,
    token: Option<EvictedBuildToken>,
}

#[allow(dead_code)]
impl<const E: usize, const R: usize> InstalledEviction<'_, E, R> {
    fn token(&self) -> &EvictedBuildToken {
        self.token.as_ref().expect("committed eviction preparation")
    }

    /// Consume only after exact generation revalidation and native detachment have succeeded.
    fn commit(mut self) -> EvictedBuildToken {
        self.token.take().expect("committed eviction preparation")
    }

    fn cleanup(&mut self) {
        let Some(token) = self.token.take() else { return };
        let _ = self.build.clear_eviction_token(&token);
        let _ = self.coordinator.evicted.finish(&token, EvictedBuildState::Poisoned);
        let _ = self.coordinator.evicted.retire(&token);
    }
}

impl<const E: usize, const R: usize> Drop for InstalledEviction<'_, E, R> {
    fn drop(&mut self) {
        self.cleanup();
    }
}

struct ExactEvictionSnapshot {
    generation: NativeGeneration,
    last_use_frame: u64,
    completed_frame: u64,
}

fn exact_eviction_snapshot(build: &Build) -> Result<ExactEvictionSnapshot, EvictionPreflight> {
    if !build.is_evictable() {
        return Err(EvictionPreflight::Changed);
    }
    let done = build.done.lock().unwrap();
    if !*done {
        return Err(EvictionPreflight::Busy);
    }
    if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
        return Err(EvictionPreflight::Released);
    }
    let Some(recipe) = build.recipe.as_ref() else { return Err(EvictionPreflight::Changed) };
    let rebuild = recipe.0.rebuild.lock().unwrap();
    if rebuild.active || rebuild.waiters != 0 {
        return Err(EvictionPreflight::Busy);
    }
    let raw = build.real.load(Ordering::Acquire);
    if raw == 0 {
        return Err(EvictionPreflight::Changed);
    }
    let (generation, residency) = recipe.pin_with_residency(raw);
    let Some(generation) = generation.filter(|_| residency == Residency::Resident) else {
        return Err(EvictionPreflight::Changed);
    };
    let last_use_frame = build.last_use_frame.load(Ordering::Relaxed);
    let completed_frame = build.completed_frame.load(Ordering::Relaxed);
    drop(rebuild);
    drop(done);
    Ok(ExactEvictionSnapshot { generation, last_use_frame, completed_frame })
}

fn exact_eviction_snapshot_is_old_enough(
    snapshot: &ExactEvictionSnapshot, frame: u64, limits: EvictionSelectionLimits,
) -> bool {
    conservative_frame_age(frame, snapshot.last_use_frame) >= limits.min_unused_frames
        && conservative_frame_age(frame, snapshot.completed_frame) >= limits.min_resident_frames
}

struct PreparedExactEviction<'a, const E: usize, const R: usize> {
    // Fields drop in declaration order: clean the transient token while admission is still held.
    installed: InstalledEviction<'a, E, R>,
    _admission: std::sync::MutexGuard<'a, ()>,
    first: ExactEvictionSnapshot,
    frame: u64,
    limits: EvictionSelectionLimits,
}

#[allow(dead_code)]
struct DetachedExactEviction {
    raw: Option<usize>,
    pins: Option<[NativeGeneration; 2]>,
    token: EvictedBuildToken,
}

#[allow(dead_code)]
impl DetachedExactEviction {
    fn token(&self) -> &EvictedBuildToken { &self.token }

    /// Native destruction may re-enter pipeline bookkeeping. The transaction returns every owner
    /// so its caller can retire them only after releasing policy, admission, and state locks.
    fn retire(mut self) { self.release_owners(); }

    fn retire_pressure(mut self, counters: Arc<PipelinePressureCounters>) {
        counters.publications_detached.fetch_add(1, Ordering::Relaxed);
        if let Some(pins) = self.pins.as_ref() {
            pins[0].generation.pressure_release.arm(counters.clone());
        }
        self.release_owners();
        counters.policy_owner_retirements.fetch_add(1, Ordering::Relaxed);
    }

    fn release_owners(&mut self) {
        drop(self.pins.take());
        if let Some(raw) = self.raw.take() { release(raw) }
    }
}

impl Drop for DetachedExactEviction {
    fn drop(&mut self) { self.release_owners(); }
}

#[allow(dead_code)]
enum ExactDetachAttempt {
    Aborted(EvictionPreflight, EvictionCandidate),
    Detached(DetachedExactEviction),
}

fn prepare_exact_eviction_on<'a, const E: usize, const R: usize>(
    build: &Arc<Build>, frame: u64, limits: EvictionSelectionLimits,
    coordinator: &'a RestorationCoordinator<E, R>,
) -> Result<PreparedExactEviction<'a, E, R>, EvictionPreflight> {
    if !build.is_evictable() {
        return Err(EvictionPreflight::Changed);
    }
    let admission = coordinator.admission.lock().unwrap();
    if build.eviction_token().is_some() {
        return Err(EvictionPreflight::Busy);
    }
    let first = exact_eviction_snapshot(build)?;
    if !exact_eviction_snapshot_is_old_enough(&first, frame, limits) {
        return Err(EvictionPreflight::Recent);
    }
    let Some(reservation) = coordinator.reserve_eviction(build, first.generation.ledger_identity()) else {
        return Err(EvictionPreflight::LedgerFull);
    };
    let Some(token) = reservation.arm_detach() else { return Err(EvictionPreflight::Changed) };
    if !build.install_eviction_token(&token) {
        let _ = coordinator.evicted.finish(&token, EvictedBuildState::Poisoned);
        let _ = coordinator.evicted.retire(&token);
        return Err(EvictionPreflight::Busy);
    }
    let installed = InstalledEviction { coordinator, build: build.clone(), token: Some(token) };
    Ok(PreparedExactEviction { installed, _admission: admission, first, frame, limits })
}

impl<const E: usize, const R: usize> PreparedExactEviction<'_, E, R> {
    fn abort_after_revalidation(self) -> EvictionPreflight {
        let result = match exact_eviction_snapshot(&self.installed.build) {
            Ok(second) if !self.first.generation.same(&second.generation) => EvictionPreflight::Changed,
            Ok(second) if self.first.last_use_frame != second.last_use_frame
                || self.first.completed_frame != second.completed_frame
                || !exact_eviction_snapshot_is_old_enough(&second, self.frame, self.limits) => {
                    EvictionPreflight::Recent
                }
            Ok(_) => EvictionPreflight::Prepared,
            Err(result) => result,
        };
        drop(self);
        result
    }

    #[allow(dead_code)]
    fn commit_exact(self, after_final_timestamp: impl FnOnce()) -> Result<DetachedExactEviction, EvictionPreflight> {
        let build = self.installed.build.clone();
        let timestamp_claim = build.claim_eviction_timestamps();
        let done = build.done.lock().unwrap();
        if !build.is_evictable() { return Err(EvictionPreflight::Changed) }
        if !*done { return Err(EvictionPreflight::Busy) }
        if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            return Err(EvictionPreflight::Released);
        }
        let Some(recipe) = build.recipe.as_ref() else { return Err(EvictionPreflight::Changed) };
        let rebuild = recipe.0.rebuild.lock().unwrap();
        if rebuild.active || rebuild.waiters != 0 { return Err(EvictionPreflight::Busy) }
        let raw = build.real.load(Ordering::Acquire);
        let mut slot = recipe.0.generation.lock().unwrap();
        if slot.residency != Residency::Resident || raw != self.first.generation.raw()
            || slot.resident.as_ref().is_none_or(|generation| {
                !Arc::ptr_eq(generation, &self.first.generation.generation)
            }) {
            return Err(EvictionPreflight::Changed);
        }
        let last_use_frame = build.last_use_frame.load(Ordering::Relaxed);
        let completed_frame = build.completed_frame.load(Ordering::Relaxed);
        let second = ExactEvictionSnapshot {
            generation: self.first.generation.clone(), last_use_frame, completed_frame,
        };
        if self.first.last_use_frame != last_use_frame
            || self.first.completed_frame != completed_frame
            || !exact_eviction_snapshot_is_old_enough(&second, self.frame, self.limits) {
            return Err(EvictionPreflight::Recent);
        }
        after_final_timestamp();
        let Some(detached) = slot.detach(raw) else { return Err(EvictionPreflight::Changed) };
        recipe.0.resident_generation.store(0, Ordering::Release);
        let old_raw = build.real.swap(0, Ordering::AcqRel);
        debug_assert_eq!(old_raw, raw);
        let token = self.installed.commit();
        let first = self.first.generation;
        let admission = self._admission;
        drop(slot); drop(rebuild); drop(done); drop(timestamp_claim); drop(admission);
        Ok(DetachedExactEviction {
            raw: Some(old_raw),
            pins: Some([first, NativeGeneration { generation: detached, recipe: Arc::downgrade(&recipe.0) }]),
            token,
        })
    }
}

#[allow(dead_code)]
fn detach_exact_eviction_on<const E: usize, const R: usize>(
    selected: SelectedEvictionCandidate, frame: u64, limits: EvictionSelectionLimits,
    coordinator: &RestorationCoordinator<E, R>, after_install: impl FnOnce(),
    after_final_timestamp: impl FnOnce(),
) -> ExactDetachAttempt {
    let prepared = match prepare_exact_eviction_on(&selected.candidate.build, frame, limits, coordinator) {
        Ok(prepared) => prepared,
        Err(result) => return ExactDetachAttempt::Aborted(result, selected.candidate),
    };
    after_install();
    match prepared.commit_exact(after_final_timestamp) {
        Ok(detached) => {
            drop(selected.candidate);
            ExactDetachAttempt::Detached(detached)
        }
        Err(result) => ExactDetachAttempt::Aborted(result, selected.candidate),
    }
}

/// Exercise the exact production transaction through publication and revalidation, then always
/// abort it. The admission lock prevents a restoration request from observing the transient
/// `Evicted` record while the native pipeline remains resident.
fn preflight_exact_eviction_on<const E: usize, const R: usize>(
    selected: &SelectedEvictionCandidate,
    frame: u64,
    limits: EvictionSelectionLimits,
    coordinator: &RestorationCoordinator<E, R>,
    after_install: impl FnOnce(),
) -> EvictionPreflight {
    let build = &selected.candidate.build;
    let prepared = match prepare_exact_eviction_on(build, frame, limits, coordinator) {
        Ok(prepared) => prepared,
        Err(result) => return result,
    };
    after_install();
    prepared.abort_after_revalidation()
}

struct RestorationAttempt {
    token: RehydrateAttemptToken,
    job: Mutex<Option<JobHandle>>,
    submitted: Condvar,
    foreground: AtomicBool,
    outcome: Mutex<Option<RestorationOutcome>>,
    pressure_stats: Option<Arc<PipelinePressureCounters>>,
    pressure_reported: AtomicBool,
}

#[allow(dead_code)]
enum RestorationOutcome {
    Failed(RehydrateCompletion),
    Succeeded(NativeGeneration),
}

impl RestorationAttempt {
    fn report_terminal(&self, succeeded: bool) {
        if self.pressure_reported.compare_exchange(
            false, true, Ordering::AcqRel, Ordering::Acquire,
        ).is_err() {
            return;
        }
        if let Some(stats) = &self.pressure_stats {
            let counter = if succeeded {
                &stats.restoration_succeeded
            } else {
                &stats.restoration_failed
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn finish(&self, outcome: RestorationOutcome) {
        let succeeded = matches!(outcome, RestorationOutcome::Succeeded(_));
        let mut current = self.outcome.lock().unwrap();
        if current.is_some() {
            return;
        }
        *current = Some(outcome);
        drop(current);
        self.report_terminal(succeeded);
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct RestorationTicket<const R: usize> {
    receipts: Arc<RehydrateReceiptRegistry<R>>,
    attempt: Arc<RestorationAttempt>,
}

#[allow(dead_code)]
enum RestorationQueue<'a, const E: usize, const R: usize> {
    Queued(RestorationQueued<'a, E, R>),
    Existing,
    Promoted,
}

#[allow(dead_code)]
struct RestorationQueued<'a, const E: usize, const R: usize> {
    coordinator: &'a RestorationCoordinator<E, R>,
    receipt: RehydrateQueuedReceipt<'a, R>,
}

#[allow(dead_code)]
impl<const E: usize, const R: usize> RestorationCoordinator<E, R> {
    fn new(generation: u64) -> Option<Self> {
        Self::new_with_pressure_stats(generation, None)
    }

    fn new_with_pressure_stats(
        generation: u64, pressure_stats: Option<Arc<PipelinePressureCounters>>,
    ) -> Option<Self> {
        Some(Self {
            evicted: EvictedBuildLedger::new(generation)?,
            receipts: Arc::new(RehydrateReceiptRegistry::new()?),
            admission: Mutex::new(()),
            attempts: Mutex::new(Vec::new()),
            pressure_stats,
            #[cfg(test)]
            captured: Arc::new(Mutex::new(None)),
            #[cfg(test)]
            before_rebuild: Arc::new(Mutex::new(None)),
            #[cfg(test)]
            before_classified_request: Arc::new(Mutex::new(None)),
        })
    }

    fn reserve_eviction<'a>(
        &'a self, build: &Arc<Build>, native: LedgerNativeGeneration,
    ) -> Option<EvictedBuildReservation<'a, E>> {
        self.evicted.reserve(build, native)
    }

    /// Arm the exact ledger record, then publish it on the build. A future eviction path must
    /// perform its state-chain revalidation and detach only while holding the returned guard.
    fn install_eviction<'a>(
        &'a self, build: &Arc<Build>, native: LedgerNativeGeneration,
    ) -> Option<InstalledEviction<'a, E, R>> {
        let token = self.reserve_eviction(build, native)?.arm_detach()?;
        if !build.install_eviction_token(&token) {
            let _ = self.evicted.finish(&token, EvictedBuildState::Poisoned);
            let _ = self.evicted.retire(&token);
            return None;
        }
        Some(InstalledEviction { coordinator: self, build: build.clone(), token: Some(token) })
    }

    fn refuse_rehydrate(&self, evicted: &EvictedBuildToken) -> bool {
        let _admission = self.admission.lock().unwrap();
        match self.receipts.reserve_unique(evicted, true) {
            Ok(Some(reservation)) => reservation.refused().is_some(),
            Ok(None) | Err(_) => false,
        }
    }

    fn queue_rehydrate(
        &self, evicted: &EvictedBuildToken, priority: RehydratePriority, background_ready: bool,
    ) -> Option<RestorationQueue<'_, E, R>> {
        let _admission = self.admission.lock().unwrap();
        let allow_new = priority == RehydratePriority::Foreground || background_ready;
        let reservation = match self.receipts.reserve_unique(evicted, allow_new) {
            Err(record) => {
                if !record.state.terminal() && priority == RehydratePriority::Foreground
                    && record.priority != RehydratePriority::Foreground {
                    return self.receipts.promote(&record.token).then_some(RestorationQueue::Promoted);
                }
                return Some(RestorationQueue::Existing);
            }
            Ok(Some(reservation)) => reservation,
            Ok(None) => return None,
        };
        let (_, receipt) = reservation.accepted(priority)?;
        if !self.evicted.rehydrate_queued(evicted) {
            drop(receipt);
            return None;
        }
        Some(RestorationQueue::Queued(RestorationQueued { coordinator: self, receipt }))
    }

    /// Snapshot receipt state first, then mutate the exact eviction record, then retire the
    /// receipt. No two policy locks overlap, and a stale token can never touch a reused slot.
    fn reconcile(&self, limit: usize) -> usize {
        let attempts = self.attempts.lock().unwrap().clone();
        for attempt in &attempts {
            if attempt.outcome.lock().unwrap().is_some() { continue }
            let status = attempt.job.lock().unwrap().as_ref().map(JobHandle::status);
            let completion = match status {
                Some(JobStatus::Cancelled) => Some(RehydrateCompletion::Cancelled),
                Some(JobStatus::Panicked) => Some(RehydrateCompletion::Panicked),
                _ => None,
            };
            if let Some(completion) = completion {
                if self.receipts.publish(&attempt.token, completion) {
                    attempt.report_terminal(false);
                }
            }
        }
        let records = self.receipts.records(limit);
        let mut reconciled = 0;
        for record in records.into_iter().filter(|record| record.state.terminal()) {
            if let Some(attempt) = attempts.iter().find(|attempt| attempt.token.exact(&record.token)) {
                attempt.report_terminal(matches!(record.state, RehydrateCompletion::Succeeded { .. }));
            }
            let token = &record.token.evicted;
            let ok = match record.state {
                RehydrateCompletion::Refused => self.evicted.rehydrate_retry(token)
                    || self.evicted.state(token) == Some(EvictedBuildState::Evicted),
                RehydrateCompletion::Cancelled | RehydrateCompletion::Panicked | RehydrateCompletion::BuildFailed => {
                    self.evicted.rehydrate_retry(token) || self.evicted.state(token) == Some(EvictedBuildState::Evicted)
                }
                RehydrateCompletion::Released => self.evicted.finish(token, EvictedBuildState::Released)
                    || self.evicted.state(token) == Some(EvictedBuildState::Released),
                RehydrateCompletion::Succeeded { .. } => self.evicted.finish(token, EvictedBuildState::Restored)
                    || self.evicted.state(token) == Some(EvictedBuildState::Restored),
                _ => false,
            };
            if !ok && self.evicted.state(token).is_some_and(|state| !state.terminal()) {
                let _ = self.evicted.finish(token, EvictedBuildState::Poisoned);
            }
            if self.receipts.retire(&record.token) { reconciled += 1 }
            // Receipt, build, and ledger locks never overlap. Exact matching makes this safe
            // after address or bounded-slot reuse, and retryable failures remain nonterminal.
            if self.evicted.state(token).is_some_and(EvictedBuildState::terminal) {
                if let Some(build) = token.build.upgrade() {
                    let _ = build.clear_eviction_token(token);
                }
                let _ = self.evicted.retire(token);
            }
        }
        let removed = {
            let mut attempts = self.attempts.lock().unwrap();
            let (active, removed) = std::mem::take(&mut *attempts).into_iter().partition(|attempt| {
                attempt.outcome.lock().unwrap().is_none()
                    && attempt.job.lock().unwrap().as_ref().is_none_or(|job| {
                        matches!(job.status(), JobStatus::Queued | JobStatus::Running)
                    })
            });
            *attempts = active;
            removed
        };
        drop(removed); // NativeGeneration/COM destruction can re-enter pipeline bookkeeping.
        reconciled
    }

    fn retries(&self, limit: usize) -> Vec<RestorationRetry> {
        let _admission = self.admission.lock().unwrap();
        self.evicted.records(limit).into_iter().filter_map(|record| {
            (record.state == EvictedBuildState::Evicted).then_some(RestorationRetry { token: record.token })
        }).collect()
    }

    /// Private queue seam for a later bind integration. Submission never invokes the work inline;
    /// foreground demand uses the normal worker band, while an exact queued background attempt is
    /// promoted through its retained `JobHandle`.
    fn request_exact_on(
        &self, evicted: &EvictedBuildToken, priority: RehydratePriority, q: &BuildQueue,
    ) -> Option<RestorationTicket<R>> {
        let build = evicted.build.upgrade()?;
        let admission = self.admission.lock().unwrap();
        if let Some(attempt) = self.attempts.lock().unwrap().iter()
            .find(|attempt| attempt.token.evicted.exact(evicted)).cloned() {
            drop(admission);
            if priority == RehydratePriority::Foreground {
                attempt.foreground.store(true, Ordering::Release);
                if let Some(job) = attempt.job.lock().unwrap().as_ref() {
                    let _ = job.promote(Priority::Foreground);
                }
                let _ = self.receipts.promote(&attempt.token);
            }
            return Some(RestorationTicket { receipts: self.receipts.clone(), attempt });
        }
        let receipt = match self.receipts.reserve_unique(evicted, true) {
            Ok(Some(receipt)) => receipt,
            Err(_) => return None,
            Ok(None) => return None,
        };
        let token = receipt.accepted_token(priority)?;
        if let Some(stats) = &self.pressure_stats {
            stats.restoration_attempts.fetch_add(1, Ordering::Relaxed);
        }
        let attempt = Arc::new(RestorationAttempt {
            token: token.clone(), job: Mutex::new(None), submitted: Condvar::new(),
            foreground: AtomicBool::new(priority == RehydratePriority::Foreground), outcome: Mutex::new(None),
            pressure_stats: self.pressure_stats.clone(), pressure_reported: AtomicBool::new(false),
        });
        self.attempts.lock().unwrap().push(attempt.clone());
        drop(admission);
        if !self.evicted.rehydrate_queued(evicted) {
            let _ = self.receipts.publish(&token, RehydrateCompletion::Cancelled);
            attempt.finish(RestorationOutcome::Failed(RehydrateCompletion::Cancelled));
            attempt.submitted.notify_all();
            return Some(RestorationTicket { receipts: self.receipts.clone(), attempt });
        }
        let reservation = match reserve_rehydrate_on(&build) {
            Ok(reservation) => reservation,
            Err(result) => {
                let completion = if result == RehydrateRequest::Abandoned {
                    RehydrateCompletion::Released
                } else {
                    RehydrateCompletion::Cancelled
                };
                let _ = self.receipts.publish(&token, completion);
                attempt.finish(RestorationOutcome::Failed(completion));
                attempt.submitted.notify_all();
                return Some(RestorationTicket { receipts: self.receipts.clone(), attempt });
            }
        };
        let worker_attempt = attempt.clone();
        let receipts = self.receipts.clone();
        let queued_build = build.clone();
        #[cfg(test)]
        let captured = self.captured.clone();
        #[cfg(test)]
        let before_rebuild = self.before_rebuild.clone();
        let queue_priority = if attempt.foreground.load(Ordering::Acquire) {
            Priority::Foreground
        } else {
            Priority::Speculative
        };
        let job = q.submit(queue_priority, Box::new(move || {
            #[cfg(test)]
            let before_rebuild = before_rebuild.lock().unwrap().take();
            #[cfg(test)]
            if let Some(before_rebuild) = before_rebuild { before_rebuild() }
            let recipe = reservation.recipe.clone();
            let (result, proof) = rebuild_reserved_pinned(&queued_build, &recipe, Some(&token));
            drop(reservation);
            #[cfg(test)]
            let captured = captured.lock().unwrap().take();
            #[cfg(test)]
            if let Some(captured) = captured { captured() }
            let outcome = match (result, proof) {
                (RebuildResult::Published { .. }, Some(proof)) => {
                    let pin = proof.pin();
                    if receipts.publish_success(&token, &proof) { RestorationOutcome::Succeeded(pin) }
                    else { RestorationOutcome::Failed(RehydrateCompletion::Released) }
                }
                (RebuildResult::Published { .. }, None) => RestorationOutcome::Failed(RehydrateCompletion::BuildFailed),
                (RebuildResult::Failed, _) => RestorationOutcome::Failed(RehydrateCompletion::BuildFailed),
                (RebuildResult::Abandoned, _) => RestorationOutcome::Failed(RehydrateCompletion::Released),
                (RebuildResult::NotReady, _) => RestorationOutcome::Failed(RehydrateCompletion::Cancelled),
            };
            if let RestorationOutcome::Failed(completion) = &outcome {
                let _ = receipts.publish(&token, *completion);
            }
            worker_attempt.finish(outcome);
        }));
        match job {
            Some(job) => {
                *attempt.job.lock().unwrap() = Some(job.clone());
                if attempt.foreground.load(Ordering::Acquire) { let _ = job.promote(Priority::Foreground); }
                attempt.submitted.notify_all();
            }
            None => {
                let _ = self.receipts.publish(&attempt.token, RehydrateCompletion::Refused);
                attempt.finish(RestorationOutcome::Failed(RehydrateCompletion::Refused));
                attempt.submitted.notify_all();
            }
        }
        Some(RestorationTicket { receipts: self.receipts.clone(), attempt })
    }

    fn request_foreground(&self, evicted: &EvictedBuildToken) -> Option<RestorationTicket<R>> {
        if !start_normal_workers() { return None }
        self.request_exact_on(evicted, RehydratePriority::Foreground, queue())
    }
}

#[allow(dead_code)]
impl<const R: usize> RestorationTicket<R> {
    fn token(&self) -> &RehydrateAttemptToken { &self.attempt.token }

    /// Waits only on this queue job, with no Build, recipe, ledger, list, or coordinator lock.
    fn wait_and_take_pin(&self) -> Result<NativeGeneration, RehydrateCompletion> {
        let waiting = self.attempt.outcome.lock().unwrap().is_none();
        let _pipeline_wait = waiting.then(|| crate::stall::span(crate::stall::StallKind::PipelineWait)).flatten();
        let _restoration_wait = waiting.then(|| crate::stall::span(crate::stall::StallKind::PipelineRestoreWait)).flatten();
        let mut job = self.attempt.job.lock().unwrap();
        while job.is_none() && self.attempt.outcome.lock().unwrap().is_none() {
            job = self.attempt.submitted.wait(job).unwrap();
        }
        let queued = job.clone();
        drop(job);
        let status = queued.map(|job| job.wait());
        let outcome = self.attempt.outcome.lock().unwrap();
        match outcome.as_ref() {
            Some(RestorationOutcome::Succeeded(pin)) => Ok(pin.clone()),
            Some(RestorationOutcome::Failed(completion)) => Err(*completion),
            None => {
                let completion = match status {
                    Some(JobStatus::Panicked) => RehydrateCompletion::Panicked,
                    Some(JobStatus::Cancelled) => RehydrateCompletion::Cancelled,
                    _ => RehydrateCompletion::Cancelled,
                };
                drop(outcome);
                if self.receipts.publish(&self.attempt.token, completion) {
                    self.attempt.report_terminal(false);
                }
                Err(completion)
            }
        }
    }
}

#[allow(dead_code)]
impl<const E: usize, const R: usize> RestorationQueued<'_, E, R> {
    fn token(&self) -> &RehydrateAttemptToken { &self.receipt.token }
    fn replacement(&self, build: &Arc<Build>, replacement: &NativeGeneration) -> Option<rehydrate_success::Proof> {
        self.receipt.replacement(build, replacement)
    }
    fn promote(&self) -> bool { self.coordinator.receipts.promote(&self.receipt.token) }
    fn complete(self, completion: RehydrateCompletion) -> bool { self.receipt.complete(completion) }
    fn complete_success(self, proof: rehydrate_success::Proof) -> bool { self.receipt.complete_success(proof) }
}

fn forget_in_flight(build: &Arc<Build>) {
    let Some(key) = &build.compute_key else { return };
    let build = Arc::downgrade(build);
    IN_FLIGHT.remove_exact(key, &build, || COALESCING.finished(1));
}

fn complete(build: &Arc<Build>, real: usize) {
    // (its translation ahead is of no use now: what it holds goes)
    drop(build.warm.lock().unwrap().take());
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
        if build.lazy {
            LAZY_BUILT.fetch_add(1, Ordering::Relaxed);
            if !build.used.load(Ordering::Relaxed) {
                LAZY_AHEAD.fetch_add(1, Ordering::Relaxed);
            }
            if build.huge.load(Ordering::Relaxed) {
                LAZY_HUGE.fetch_add(1, Ordering::Relaxed);
            }
        }
        let _writer = build.eviction_timestamp_writer();
        build.completed_frame.store(super::present::presents(), Ordering::Relaxed);
    }
    let (orphan, replaced_generation) = {
        let done = build.done.lock().unwrap();
        let orphan = build.orphan.load(Ordering::Acquire);
        let mut replaced_generation = None;
        if !orphan {
            if real != 0 {
                if let Some(recipe) = &build.recipe {
                    replaced_generation = recipe.publish(real).1;
                }
            } else if let Some(recipe) = &build.recipe {
                recipe.mark_failed();
            }
            build.real.store(real, Ordering::Release);
        }
        build.publish_completed(done);
        (orphan, replaced_generation)
    };
    // outside `done`: a pipeline Release override can re-enter pipeline bookkeeping
    drop(replaced_generation);
    if orphan {
        release(real);
    }
    forget_in_flight(build);
}

/// Result shared by callers that requested the same overlapping recipe rebuild.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RebuildResult {
    NotReady,
    Failed,
    Abandoned,
    Published { generation: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RebuildRequest {
    Queued,
    Coalesced,
    NotReady,
    Abandoned,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RehydrateRequest {
    Queued,
    Coalesced,
    NotReady,
    Abandoned,
    Resident,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvictResult {
    Evicted { generation: u64 },
    NotReady,
    Active,
    Abandoned,
    Unavailable,
    NotResident,
}

/// Detach both owners of the current native generation. This remains deliberately uncalled until
/// a policy and a nonblocking rehydration path choose the correctness/performance tradeoff after
/// eviction. Exact recording and receipt pins independently keep the detached generation alive.
#[allow(dead_code)]
fn try_evict_current(build: &Build) -> EvictResult {
    let done = build.done.lock().unwrap();
    if !*done {
        return EvictResult::NotReady;
    }
    if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
        return EvictResult::Abandoned;
    }
    let Some(recipe) = build.recipe.as_ref() else { return EvictResult::Unavailable };
    // Keep the rebuild turn reserved through the raw-owner swap. Every rebuild reservation takes
    // this lock, and publication takes `done`, so neither transition can cross this one.
    let rebuild = recipe.0.rebuild.lock().unwrap();
    if rebuild.active || rebuild.waiters != 0 {
        return EvictResult::Active;
    }
    let raw = build.real.load(Ordering::Acquire);
    if raw == 0 {
        return EvictResult::NotResident;
    }
    let Some(generation) = recipe.detach_current(raw) else { return EvictResult::NotResident };
    let generation_id = generation.id();
    let old_raw = build.real.swap(0, Ordering::AcqRel);
    debug_assert_eq!(old_raw, raw);
    drop(rebuild);
    drop(done);

    // COM Release can re-enter pipeline bookkeeping. Drop both owners outside every state lock;
    // an exact recording/receipt pin is another Arc to `generation` and survives independently.
    drop(generation);
    release(old_raw);
    EvictResult::Evicted { generation: generation_id }
}

/// Build and atomically publish one replacement generation. This remains deliberately uncalled
/// until submitted recording pins are retained to exact GPU retirement receipts: publication is
/// safe, but a policy must not make an old generation reclaimable while the GPU can still use it.
#[allow(dead_code)]
fn rebuild_once(build: &Build) -> Option<RebuildResult> {
    let recipe = build.recipe.as_ref()?;
    {
        let done = build.done.lock().unwrap();
        if !*done {
            return Some(RebuildResult::NotReady);
        }
        if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            return Some(RebuildResult::Abandoned);
        }
    }
    match recipe.begin_rebuild() {
        RebuildTurn::Follower(result) => return Some(result),
        RebuildTurn::Leader => {}
    }

    Some(rebuild_reserved(build, recipe))
}

/// Run an attempt which already owns the recipe's rebuild turn.
fn rebuild_reserved(build: &Build, recipe: &Recipe) -> RebuildResult {
    rebuild_reserved_pinned(build, recipe, None).0
}

/// The exact published generation is retained before `Build.done` is released, so a later
/// eviction cannot invalidate a successful foreground restoration before its waiter pins it.
fn rebuild_reserved_pinned(
    build: &Build, recipe: &Recipe, attempt: Option<&RehydrateAttemptToken>,
) -> (RebuildResult, Option<rehydrate_success::Proof>) {
    // A queued request can lose its last title-visible owner before a worker starts it. The queued
    // Arc keeps this state alive only so the attempt can retire; it must not revive the pipeline.
    {
        let done = build.done.lock().unwrap();
        if !*done || build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            drop(done);
            recipe.finish_rebuild(RebuildResult::Abandoned);
            return (RebuildResult::Abandoned, None);
        }
    }

    // Do expensive driver work without Build.done or the recipe state held. Binds keep using the
    // current immutable generation throughout the rebuild.
    let raw = run_job(recipe.initial_job());
    let mut replaced_generation = None;
    let mut replaced_raw = 0;
    let mut published = None;
    let _timestamp_writer = (raw != 0).then(|| build.eviction_timestamp_writer());
    let result = if raw == 0 {
        RebuildResult::Failed
    } else {
        let _done = build.done.lock().unwrap();
        if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            RebuildResult::Abandoned
        } else {
            let (generation, replaced) = recipe.publish(raw);
            published = attempt.and_then(|attempt| rehydrate_success::Proof::published(attempt, build, recipe, &generation));
            replaced_generation = replaced;
            replaced_raw = build.real.swap(raw, Ordering::AcqRel);
            build.completed_frame.store(super::present::presents(), Ordering::Relaxed);
            RebuildResult::Published { generation: generation.id() }
        }
    };

    // Publish the result before COM Release, which can re-enter pipeline bookkeeping. Never drop
    // either old owner under Build.done or while this attempt is still blocking followers.
    drop(_timestamp_writer);
    recipe.finish_rebuild(result);
    drop(replaced_generation);
    if matches!(result, RebuildResult::Published { .. }) {
        release(replaced_raw);
    } else if raw != 0 {
        release(raw);
    }
    (result, published)
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
static LAZY_MADE: AtomicUsize = AtomicUsize::new(0);
static LAZY_GRAPHICS: AtomicUsize = AtomicUsize::new(0);
static LAZY_BUILT: AtomicUsize = AtomicUsize::new(0);
static LAZY_AHEAD: AtomicUsize = AtomicUsize::new(0);
static LAZY_HUGE: AtomicUsize = AtomicUsize::new(0);

/// What the pipeline line (`stats_line`) shows.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
struct Stats {
    made: usize,
    graphics: usize,
    built: usize,
    ahead: usize,
    huge: usize,
    /// translated ahead (`Backlog`), still in the backlog
    warmed: usize,
    backlog: usize,
    /// binds' waits for pipelines being built: (waits, µs) so far (`api_cost::Call::PipelineWait`)
    waits: (u64, u64),
}

fn stats() -> Stats {
    let l = |a: &AtomicUsize| a.load(Ordering::Relaxed);
    Stats {
        made: l(&LAZY_MADE),
        graphics: l(&LAZY_GRAPHICS),
        built: l(&LAZY_BUILT),
        ahead: l(&LAZY_AHEAD),
        huge: l(&LAZY_HUGE),
        warmed: l(&WARMED),
        backlog: l(&BACKLOG),
        waits: crate::hle::api_cost::totals()[crate::hle::api_cost::Call::PipelineWait as usize],
    }
}

/// The pipeline line for `now` when something changed since `last` (written 2 s before): how
/// many lazy pipelines the title made, how many are built and how many of those before their
/// first use (a bind did not wait for them), and how long binds stood still waiting for
/// pipelines in those 2 s (counted when a wait ends).
fn stats_line(now: &Stats, last: &Stats, huge_kb: usize) -> Option<String> {
    if now == last {
        return None;
    }
    Some(format!(
        "D3D12.X: lazy pipelines: {} made ({} graphics), {} built ({} before their first use, {} with {huge_kb} KB of DXIL or more), {} translated ahead, {} in the build-ahead backlog; binds waited {} ms for pipelines being built ({} waits in 2 s)",
        now.made,
        now.graphics,
        now.built,
        now.ahead,
        now.huge,
        now.warmed,
        now.backlog,
        (now.waits.1 - last.waits.1) / 1000,
        now.waits.0 - last.waits.0,
    ))
}

/// The pipeline line every 2 s while it changes (`stats_line`), and the line of calls put off to
/// the submit (`put_off_line`).
fn start_stats() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    // (a test's pipelines are not the title's)
    if cfg!(test) {
        return;
    }
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-pso-stats".into()).spawn(|| {
            let mut last = Stats::default();
            let mut last_put_off = PutOffStats::default();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = stats();
                if let Some(line) = stats_line(&now, &last, super::real::huge_pso_bytes() >> 10) {
                    crate::xlog_info!("{line}");
                }
                last = now;
                let now = put_off_stats();
                if let Some(line) = put_off_line(&now, &last_put_off) {
                    crate::xlog_info!("{line}");
                }
                last_put_off = now;
            }
        });
    });
}
static QUEUED_BUILDS: AtomicUsize = AtomicUsize::new(0);
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// What became of a call put off to the submit (`list_x::defer`), for the line of them
/// (`put_off_line`).
#[derive(Clone, Copy)]
pub(super) enum PutOff {
    /// put off: its list split around it
    Deferred,
    /// its pipeline would have been waited for, but the list could not be split (no room for
    /// more segments, a predicated or bundle list, one that is no title list, `XCL_ALLOC_SWAP=0`):
    /// waited in the call
    Refused,
    /// recorded at a submit with its pipeline built
    Recorded,
    /// recorded at a submit, but its pipeline's build failed or ran out of its budget: skipped
    Skipped,
    /// not recorded at the submit (no allocator of the list's own, its Reset failed)
    Dropped,
}

static PUT_OFF: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
/// pipelines a put-off call needed (`need_by_submit`)
static NEEDED_MADE: AtomicUsize = AtomicUsize::new(0);
/// waits at a submit for the pipelines of its put-off calls: count, µs
static SUBMIT_WAITS: AtomicU64 = AtomicU64::new(0);
static SUBMIT_WAIT_US: AtomicU64 = AtomicU64::new(0);

pub(super) fn note_put_off(what: PutOff) {
    PUT_OFF[what as usize].fetch_add(1, Ordering::Relaxed);
    start_stats();
}

thread_local! {
    /// this thread records the put-off calls of a submit (`at_submit`)
    static AT_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While it lives this thread records a submit's put-off calls: its waits for their pipelines
/// are the submit's (`put_off_line`).
pub(super) struct AtSubmit(bool);

pub(super) fn at_submit() -> AtSubmit {
    AtSubmit(AT_SUBMIT.with(|a| a.replace(true)))
}

impl Drop for AtSubmit {
    fn drop(&mut self) {
        AT_SUBMIT.with(|a| a.set(self.0));
    }
}

pub(super) fn is_at_submit() -> bool {
    AT_SUBMIT.with(|a| a.get())
}

/// How many put-off calls came to `what` so far.
#[cfg(test)]
pub(super) fn put_off_count(what: PutOff) -> usize {
    PUT_OFF[what as usize].load(Ordering::Relaxed)
}

/// What the line of put-off calls (`put_off_line`) shows.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
struct PutOffStats {
    /// by `PutOff`
    calls: [usize; 5],
    needed: usize,
    /// waits at submits: (count, µs)
    waits: (u64, u64),
    /// the most segments a title list's recording had (`title_list::segments_peak`)
    segments: usize,
}

fn put_off_stats() -> PutOffStats {
    PutOffStats {
        calls: std::array::from_fn(|k| PUT_OFF[k].load(Ordering::Relaxed)),
        needed: NEEDED_MADE.load(Ordering::Relaxed),
        waits: (SUBMIT_WAITS.load(Ordering::Relaxed), SUBMIT_WAIT_US.load(Ordering::Relaxed)),
        segments: super::title_list::segments_peak(),
    }
}

/// The line of calls put off to the submit for `now` when it changed since `last` (written 2 s
/// before) and some call was put off or would have been: how many, what became of them at the
/// submits, the pipelines they needed, and how long submits stood in their waits for them.
/// (A thread that waits never builds the pipeline itself here: a foreground worker does.)
fn put_off_line(now: &PutOffStats, last: &PutOffStats) -> Option<String> {
    let c = |k: PutOff| now.calls[k as usize];
    if now == last || c(PutOff::Deferred) + c(PutOff::Refused) == 0 {
        return None;
    }
    Some(format!(
        "D3D12.X: put-off dispatches: {} put off to the submit ({} in 2 s), {} waited in the call (no split); at submits {} recorded, {} skipped (build failed or late), {} dropped; {} pipelines needed by a submit; submits waited {} ms for them in 2 s ({} waits; {} ms so far); at most {} segments in a recording",
        c(PutOff::Deferred),
        c(PutOff::Deferred) - last.calls[PutOff::Deferred as usize],
        c(PutOff::Refused),
        c(PutOff::Recorded),
        c(PutOff::Skipped),
        c(PutOff::Dropped),
        now.needed,
        (now.waits.1 - last.waits.1) / 1000,
        now.waits.0 - last.waits.0,
        now.waits.1 / 1000,
        now.segments,
    ))
}

/// The PC calls of the title's threads (`pc_call`): the ones going on, and how long the ones
/// that ended took since the last frame was shown.
struct PcCalls {
    /// (id, when it began)
    now: Vec<(u64, std::time::Instant)>,
    /// the frames shown when `spent` began to count (`None` before the first observation)
    frame: Option<u64>,
    /// when the latest frame advance was first observed; calls that started earlier are clamped here
    frame_since: Option<std::time::Instant>,
    /// the time of the PC calls that ended since then
    spent: std::time::Duration,
}

impl PcCalls {
    const fn new() -> Self {
        Self { now: Vec::new(), frame: None, frame_since: None, spent: std::time::Duration::ZERO }
    }

    /// `frame` frames have been shown: the calls before the last of them are not this wait's.
    fn at(&mut self, frame: u64, now: std::time::Instant) {
        match self.frame {
            None => self.frame = Some(frame),
            Some(prev) if frame > prev => {
                self.frame = Some(frame);
                self.frame_since = Some(now);
                self.spent = std::time::Duration::ZERO;
            }
            Some(_) => {}
        }
    }

    fn clamp(&self, since: std::time::Instant) -> std::time::Instant {
        self.frame_since.map_or(since, |f| since.max(f))
    }

    fn begin(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        self.now.push((id, now));
    }

    fn end(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        if let Some(i) = self.now.iter().position(|c| c.0 == id) {
            let (_, since) = self.now.swap_remove(i);
            self.spent += now.saturating_duration_since(self.clamp(since));
        }
    }

    /// Whether a call going on holds the title: the PC calls since the last frame, it included,
    /// have taken `PC_CALL_HOLDS_AFTER`.
    fn holds(&mut self, frame: u64, now: std::time::Instant) -> bool {
        self.at(frame, now);
        let spent = self.spent;
        self.now.iter().any(|(_, since)| spent + now.saturating_duration_since(self.clamp(*since)) >= PC_CALL_HOLDS_AFTER)
    }
}

static PC_CALLS: Mutex<PcCalls> = Mutex::new(PcCalls::new());
/// From this much time in PC calls since the last frame on, a PC call holds the title: the
/// console's calls return at once.
const PC_CALL_HOLDS_AFTER: std::time::Duration = std::time::Duration::from_millis(250);

/// Whether one of the title's threads is held where a console title never waits: a bind is
/// waiting for a pipeline still being built, or a PC call is going on after such calls have
/// kept the title's threads for long (`pc_call`). `kernelx::Sleep` does not count that time.
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0 || held_by_pc_call()
}

fn held_by_pc_call() -> bool {
    PC_CALLS.lock().unwrap().holds(super::present::presents(), std::time::Instant::now())
}

/// A call of the title into D3D12 that XCL or the PC's driver can keep for long where the
/// console's takes no time: a submission (XCL's uploads and tile mappings before the lists, the
/// driver's compiling in them), a texture's placement, a pipeline built in the call. While it
/// lasts the title is held (`title_held`), once the PC calls since the last frame shown, this
/// one included, have taken `PC_CALL_HOLDS_AFTER`: one long call, or many short ones in a row
/// (a retail title placed 16 384 textures in 29 s without a frame, 1 - 50 ms each). A title
/// that shows its frames is not held by its submissions.
pub struct PcCall(u64);

pub fn pc_call() -> PcCall {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    PC_CALLS.lock().unwrap().begin(id, super::present::presents(), std::time::Instant::now());
    PcCall(id)
}

impl Drop for PcCall {
    fn drop(&mut self) {
        PC_CALLS.lock().unwrap().end(self.0, super::present::presents(), std::time::Instant::now());
    }
}

/// Whether this thread is building a pipeline for `build` (a worker, not one of the title's).
pub(super) fn building_here() -> bool {
    BUILDING.with(|b| b.borrow().is_some())
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held;

impl Held {
    fn new() -> Held {
        HELD.fetch_add(1, Ordering::AcqRel);
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

fn recipe_gate_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| value.trim() == "1")
}

/// Whether background builds retain their immutable inputs after the initial one-shot job.
/// Explicit recipe retention is observation-only; either pressure gate also needs these recipes.
pub(super) fn rebuild_recipes_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("XCL_PSO_REBUILD_RECIPES").ok();
        let explicit = recipe_gate_value(value.as_deref());
        let on = explicit || pressure_policy_mode().is_some();
        if explicit && pressure_policy_mode().is_none() {
            crate::xlog_info!("D3D12.X: XCL_PSO_REBUILD_RECIPES=1 - retaining pipeline rebuild inputs only; pipelines are not evicted or rebuilt");
        }
        on
    })
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

struct Generation<T> {
    id: u64,
    raw: usize,
    _owner: T,
    // Declared after `_owner`: its drop records only after that generation owner's COM Release.
    pressure_release: GenerationPressureRelease,
}

#[derive(Default)]
struct GenerationPressureRelease(OnceLock<Arc<PipelinePressureCounters>>);

impl GenerationPressureRelease {
    fn arm(&self, counters: Arc<PipelinePressureCounters>) { let _ = self.0.set(counters); }
}

impl Drop for GenerationPressureRelease {
    fn drop(&mut self) {
        if let Some(counters) = self.0.get() {
            counters.generation_owners_released.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct GenerationSlot<T> {
    next: u64,
    resident: Option<Arc<Generation<T>>>,
    residency: Residency,
}

impl<T> Default for GenerationSlot<T> {
    fn default() -> Self { Self { next: 0, resident: None, residency: Residency::Rebuilding } }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Residency {
    Resident,
    Evicted,
    Rebuilding,
    Failed,
}

impl<T> GenerationSlot<T> {
    fn publish(&mut self, raw: usize, owner: T) -> (Arc<Generation<T>>, Option<Arc<Generation<T>>>) {
        self.next = self.next.checked_add(1).expect("pipeline generation id exhausted");
        let generation = Arc::new(Generation {
            id: self.next, raw, _owner: owner, pressure_release: Default::default(),
        });
        let replaced = self.resident.replace(generation.clone());
        self.residency = Residency::Resident;
        (generation, replaced)
    }

    fn pin(&self, raw: usize) -> Option<Arc<Generation<T>>> {
        self.resident.as_ref().filter(|generation| generation.raw == raw).cloned()
    }

    fn mark_rebuilding(&mut self) -> Residency {
        let previous = self.residency;
        self.residency = Residency::Rebuilding;
        previous
    }

    fn finish_rebuild(&mut self, result: RebuildResult, previous: Option<Residency>) {
        self.residency = if self.resident.is_some() {
            Residency::Resident
        } else {
            match result {
                RebuildResult::Failed => Residency::Failed,
                RebuildResult::Abandoned | RebuildResult::NotReady => match previous {
                    Some(Residency::Evicted) => Residency::Evicted,
                    Some(Residency::Failed) => Residency::Failed,
                    _ => Residency::Failed,
                },
                RebuildResult::Published { .. } => Residency::Failed,
            }
        };
    }

    fn detach(&mut self, raw: usize) -> Option<Arc<Generation<T>>> {
        if self.residency != Residency::Resident || self.resident.as_ref()?.raw != raw {
            return None;
        }
        self.residency = Residency::Evicted;
        self.resident.take()
    }
}

struct RecipeInner {
    build: Box<dyn Fn() -> usize + Send + Sync>,
    generation: Mutex<GenerationSlot<ID3D12PipelineState>>,
    resident_generation: AtomicU64,
    rebuild: Mutex<RebuildState>,
    rebuild_done: Condvar,
}

#[derive(Default)]
struct RebuildState {
    active: bool,
    completed: u64,
    waiters: usize,
    last: Option<RebuildResult>,
    from: Option<Residency>,
}

#[derive(Debug)]
enum RebuildTurn {
    Leader,
    Follower(RebuildResult),
}

/// One immutable native pipeline generation. Its strong owner keeps the exact object alive even
/// after the title releases its stand-in or a later rebuild publishes the same raw address.
#[derive(Clone)]
pub(super) struct NativeGeneration {
    generation: Arc<Generation<ID3D12PipelineState>>,
    recipe: Weak<RecipeInner>,
}

impl NativeGeneration {
    #[allow(dead_code)] // consumed by the later retirement diagnostics stage
    pub(super) fn id(&self) -> u64 { self.generation.id }
    #[allow(dead_code)] // exposed to focused generation-capture tests
    pub(super) fn raw(&self) -> usize { self.generation.raw }
    pub(super) fn same(&self, other: &Self) -> bool { Arc::ptr_eq(&self.generation, &other.generation) }
    // The owner allocation, rather than `raw`, distinguishes ABA reuse by a later generation.
    pub(super) fn identity(&self) -> usize { Arc::as_ptr(&self.generation) as usize }
    #[allow(dead_code)]
    fn ledger_identity(&self) -> LedgerNativeGeneration {
        LedgerNativeGeneration { id: self.id(), identity: self.identity() }
    }
}

impl Build {
    /// Pin a completed one-shot result while the caller holds `done`. The extra COM owner and
    /// allocation are paid only when a retained bind snapshot first selects the pipeline.
    fn pin_one_shot(&self, raw: usize) -> Option<NativeGeneration> {
        if raw == 0 || self.recipe.is_some() {
            return None;
        }
        let mut cached = self.one_shot_generation.lock().unwrap();
        let generation = match cached.as_ref() {
            Some(generation) if generation.raw == raw => generation.clone(),
            Some(_) => return None,
            None => {
                let p = raw as *mut std::ffi::c_void;
                let owner = unsafe { ID3D12PipelineState::from_raw_borrowed(&p) }?.clone();
                let generation = Arc::new(Generation {
                    id: 1, raw, _owner: owner, pressure_release: Default::default(),
                });
                *cached = Some(generation.clone());
                generation
            }
        };
        Some(NativeGeneration { generation, recipe: Weak::new() })
    }
}

struct GenerationRetirements<G, R> {
    pending: Vec<(R, Vec<G>)>,
    poisoned: Vec<G>,
    poisoned_identities: Option<HashSet<usize>>,
}

impl<G, R> GenerationRetirements<G, R> {
    const fn new() -> Self {
        Self { pending: Vec::new(), poisoned: Vec::new(), poisoned_identities: None }
    }

    fn sweep(&mut self, complete: impl Fn(&R) -> bool) -> Vec<(R, Vec<G>)> {
        let mut retired = Vec::new();
        let mut i = 0;
        while i < self.pending.len() {
            if complete(&self.pending[i].0) {
                retired.push(self.pending.swap_remove(i));
            } else {
                i += 1;
            }
        }
        retired
    }

    fn push(&mut self, receipt: R, pins: Vec<G>) {
        if !pins.is_empty() {
            self.pending.push((receipt, pins));
        }
    }

    /// Retain new identities permanently and return duplicates for destruction by the caller.
    fn poison(&mut self, pins: Vec<G>, identity: impl Fn(&G) -> usize) -> Vec<G> {
        let identities = self.poisoned_identities.get_or_insert_with(HashSet::new);
        let mut duplicates = Vec::new();
        for pin in pins {
            if identities.insert(identity(&pin)) {
                self.poisoned.push(pin);
            } else {
                duplicates.push(pin);
            }
        }
        duplicates
    }
}

/// Snapshot one receipt per exact fence owner, observe those owners without the state mutex, then
/// retire every batch proven complete by that monotonic observation. Receipts appended during the
/// probes are eligible only when they use an owner present in the snapshot.
fn sweep_retirements<G, R, K, V>(
    state: &Mutex<GenerationRetirements<G, R>>,
    key: impl Fn(&R) -> K,
    observe: impl Fn(&R) -> V,
    complete: impl Fn(&R, &HashMap<K, V>) -> bool,
) where
    R: Clone,
    K: Eq + std::hash::Hash,
{
    let probes = {
        let retiring = state.lock().unwrap();
        let mut probes = HashMap::new();
        for (receipt, _) in &retiring.pending {
            probes.entry(key(receipt)).or_insert_with(|| receipt.clone());
        }
        probes
    };
    let observed = probes.into_iter().map(|(key, receipt)| (key, observe(&receipt))).collect();
    let retired = {
        let mut retiring = state.lock().unwrap();
        retiring.sweep(|receipt| complete(receipt, &observed))
    };
    drop(retired);
}

fn dedup_generations<G>(pins: impl Iterator<Item = G>, identity: impl Fn(&G) -> usize) -> Vec<G> {
    let mut unique = Vec::new();
    let mut identities = HashSet::new();
    for pin in pins {
        if identities.insert(identity(&pin)) {
            unique.push(pin);
        }
    }
    unique
}

static RETIRING_GENERATIONS: Mutex<GenerationRetirements<NativeGeneration, super::alloc_pool::SubmissionReceipt>> =
    Mutex::new(GenerationRetirements::new());

fn sweep_native_generations() {
    sweep_retirements(
        &RETIRING_GENERATIONS,
        super::alloc_pool::SubmissionReceipt::fence_identity,
        super::alloc_pool::SubmissionReceipt::completed_value,
        |receipt, observed| {
            observed
                .get(&receipt.fence_identity())
                .is_some_and(|&completed| completed >= receipt.target_value())
        },
    );
}

/// Keep the exact native generations used by one executed title submission until its strong
/// queue receipt completes. Completed batches are swept whenever another pinned batch arrives.
pub(super) fn retire_native_generations<'a>(
    receipt: super::alloc_pool::SubmissionReceipt,
    pins: impl Iterator<Item = &'a NativeGeneration>,
) {
    if receipt.is_complete() {
        return;
    }
    let pins = dedup_generations(pins.cloned(), NativeGeneration::identity);
    if pins.is_empty() {
        return;
    }
    RETIRING_GENERATIONS.lock().unwrap().push(receipt, pins);
    sweep_native_generations();
}

/// Native execution began but no exact retirement marker could be recorded. Retain every exact
/// generation permanently: dropping any of them could free a pipeline still referenced by GPU work.
pub(super) fn poison_native_generations<'a>(pins: impl Iterator<Item = &'a NativeGeneration>) {
    let pins = dedup_generations(pins.cloned(), NativeGeneration::identity);
    if pins.is_empty() {
        return;
    }
    sweep_native_generations();
    let mut retiring = RETIRING_GENERATIONS.lock().unwrap();
    let duplicates = retiring.poison(pins, NativeGeneration::identity);
    drop(retiring);
    drop(duplicates);
}

#[cfg(test)]
pub(super) fn sweep_native_generations_for_test() {
    sweep_native_generations();
}

/// Rebuildable immutable inputs for one pipeline. The first build and any later rebuild use the
/// same operation; retaining it does not by itself schedule or run more work. Its existing opt-in
/// allocation also owns the native generation; one-shot builds instead allocate an owner lazily
/// when a retained bind snapshot first selects them.
#[derive(Clone)]
pub(super) struct Recipe(Arc<RecipeInner>);

impl Recipe {
    pub(super) fn new(build: impl Fn() -> usize + Send + Sync + 'static) -> Self {
        Self(Arc::new(RecipeInner {
            build: Box::new(build),
            generation: Mutex::new(GenerationSlot::default()),
            resident_generation: AtomicU64::new(0),
            rebuild: Mutex::new(RebuildState::default()),
            rebuild_done: Condvar::new(),
        }))
    }

    fn run(&self) -> usize {
        (self.0.build)()
    }

    /// Publish before `Build.real` becomes visible. The returned generation owns one independent
    /// COM reference; the raw result remains the existing `Build.real` owner.
    fn publish(&self, raw: usize) -> (NativeGeneration, Option<NativeGeneration>) {
        let p = raw as *mut std::ffi::c_void;
        let owner = unsafe { ID3D12PipelineState::from_raw_borrowed(&p) }.expect("non-null pipeline result").clone();
        let mut slot = self.0.generation.lock().unwrap();
        let (generation, replaced) = slot.publish(raw, owner);
        self.0.resident_generation.store(generation.id, Ordering::Release);
        drop(slot);
        let wrap = |generation| NativeGeneration { generation, recipe: Arc::downgrade(&self.0) };
        (wrap(generation), replaced.map(wrap))
    }

    fn pin_with_residency(&self, raw: usize) -> (Option<NativeGeneration>, Residency) {
        let generation = self.0.generation.lock().unwrap();
        (generation.pin(raw).map(|generation| NativeGeneration {
            generation, recipe: Arc::downgrade(&self.0),
        }), generation.residency)
    }

    fn mark_failed(&self) {
        let mut generation = self.0.generation.lock().unwrap();
        if generation.resident.is_none() {
            generation.residency = Residency::Failed;
            self.0.resident_generation.store(0, Ordering::Release);
        }
    }

    fn detach_current(&self, raw: usize) -> Option<NativeGeneration> {
        let mut slot = self.0.generation.lock().unwrap();
        let detached = slot.detach(raw).map(|generation| NativeGeneration {
            generation, recipe: Arc::downgrade(&self.0),
        });
        if detached.is_some() { self.0.resident_generation.store(0, Ordering::Release) }
        drop(slot);
        detached
    }

    fn initial_job(&self) -> Job {
        let recipe = self.clone();
        Box::new(move || recipe.run())
    }

    fn begin_rebuild(&self) -> RebuildTurn {
        let mut state = self.0.rebuild.lock().unwrap();
        loop {
            if !state.active && state.waiters == 0 {
                state.active = true;
                state.from = Some(self.0.generation.lock().unwrap().mark_rebuilding());
                return RebuildTurn::Leader;
            }
            if state.active {
                let completed = state.completed;
                state.waiters += 1;
                while state.completed == completed {
                    state = self.0.rebuild_done.wait(state).unwrap();
                }
                let result = state.last.expect("completed rebuild without a result");
                state.waiters -= 1;
                if state.waiters == 0 {
                    self.0.rebuild_done.notify_all();
                }
                return RebuildTurn::Follower(result);
            }
            // Followers of the completed attempt must consume `last` before another leader can
            // overwrite it. A request arriving during this drain starts (or joins) the next one.
            state = self.0.rebuild_done.wait(state).unwrap();
        }
    }

    /// Reserve one asynchronous attempt without waiting for an attempt already in flight.
    fn reserve_rebuild(&self) -> bool {
        let mut state = self.0.rebuild.lock().unwrap();
        if state.active || state.waiters != 0 {
            return false;
        }
        state.active = true;
        state.from = Some(self.0.generation.lock().unwrap().mark_rebuilding());
        true
    }

    /// Reserve a build only when no native generation is resident. Failed attempts are retryable;
    /// a live current generation is never replaced by this path.
    fn reserve_rehydrate(&self) -> Result<u64, RehydrateRequest> {
        let mut state = self.0.rebuild.lock().unwrap();
        if state.active || state.waiters != 0 {
            return Err(RehydrateRequest::Coalesced);
        }
        let mut generation = self.0.generation.lock().unwrap();
        if !matches!(generation.residency, Residency::Evicted | Residency::Failed) || generation.resident.is_some() {
            return Err(RehydrateRequest::Resident);
        }
        let turn = state.completed;
        state.active = true;
        state.from = Some(generation.mark_rebuilding());
        Ok(turn)
    }

    fn finish_rebuild(&self, result: RebuildResult) {
        let mut state = self.0.rebuild.lock().unwrap();
        self.0.generation.lock().unwrap().finish_rebuild(result, state.from.take());
        state.last = Some(result);
        state.active = false;
        state.completed = state.completed.checked_add(1).expect("pipeline rebuild serial exhausted");
        self.0.rebuild_done.notify_all();
    }

    /// A queued reservation was refused or discarded before it ran. Restore its retryable state
    /// and wake any synchronous follower which joined it.
    fn cancel_reserved_rebuild(&self, turn: u64) {
        let mut state = self.0.rebuild.lock().unwrap();
        if !state.active || state.completed != turn {
            return;
        }
        self.0.generation.lock().unwrap().finish_rebuild(RebuildResult::NotReady, state.from.take());
        state.last = Some(RebuildResult::NotReady);
        state.active = false;
        state.completed = state.completed.checked_add(1).expect("pipeline rebuild serial exhausted");
        self.0.rebuild_done.notify_all();
    }
}

/// Own one exact accepted rehydration turn until its queued closure completes. Queue refusal,
/// cancellation, shutdown, or an unwind drops the closure and restores a retryable residency;
/// its completed serial prevents an old guard from cancelling a newer turn.
struct RehydrateReservation {
    recipe: Recipe,
    turn: u64,
}

impl Drop for RehydrateReservation {
    fn drop(&mut self) {
        self.recipe.cancel_reserved_rebuild(self.turn);
    }
}

/// Drop an unstarted orphaned job outside all locks and wake any concurrent waiter.
fn drop_unstarted_job(build: &Arc<Build>) {
    let job = build.job.lock().unwrap().take();
    if let Some(job) = job {
        build.cancel_queued();
        {
            let done = build.done.lock().unwrap();
            build.publish_completed(done);
        }
        drop(job);
        // (its translation ahead is of no use either, as in `complete`: the backlog may hold
        // the build a while yet)
        drop(build.warm.lock().unwrap().take());
        forget_in_flight(build);
    }
}

fn queue() -> &'static BuildQueue {
    static Q: OnceLock<BuildQueue> = OnceLock::new();
    Q.get_or_init(BuildQueue::new)
}

pub fn ranked_preload_stats() -> RankedPreloadStats {
    let snapshot = queue().ranked_preload_stats();
    RankedPreloadStats {
        submissions: snapshot.submissions,
        started: snapshot.started,
        promoted: snapshot.promoted,
    }
}

fn start_normal_workers() -> bool {
    static STARTED: OnceLock<bool> = OnceLock::new();
    *STARTED.get_or_init(|| {
        let q = queue();
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        let mut started = 0;
        for k in 0..n {
            let worker = q.worker_between(Priority::BoundSoon, Priority::Foreground);
            if std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || {
                NORMAL_WORKER.set(true);
                worker.run();
            }).is_ok() {
                started += 1;
            }
        }
        started != 0
    })
}

/// Normal workers waiting for a huge driver-compile slot, and bounded replacements serving their
/// priority band. Preload workers have their own band and do not need foreground replacements.
static PARKED_NORMAL: AtomicUsize = AtomicUsize::new(0);
static NORMAL_HELPERS: AtomicUsize = AtomicUsize::new(0);
static NORMAL_HELPERS_STARTED: AtomicUsize = AtomicUsize::new(0);

/// A normal pipeline worker is about to wait for a huge driver-compile slot. A bounded helper
/// serves its queue until workers return; calls from title and preload threads do nothing.
pub(super) fn parked_for_big_compile() -> ParkedWorker {
    if !NORMAL_WORKER.get() {
        return ParkedWorker(false);
    }
    let parked = PARKED_NORMAL.fetch_add(1, Ordering::AcqRel) + 1;
    let most = parked.min(crate::gpu::recomp::dxc::compile_threads());
    let mut helpers = NORMAL_HELPERS.load(Ordering::Acquire);
    while helpers < most {
        match NORMAL_HELPERS.compare_exchange(helpers, helpers + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                spawn_normal_helper(helpers);
                break;
            }
            Err(now) => helpers = now,
        }
    }
    ParkedWorker(true)
}

pub(super) struct ParkedWorker(bool);

impl Drop for ParkedWorker {
    fn drop(&mut self) {
        if self.0 {
            PARKED_NORMAL.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

fn spawn_normal_helper(k: usize) {
    let worker = queue().worker_between(Priority::BoundSoon, Priority::Foreground);
    let n = NORMAL_HELPERS_STARTED.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: a pipeline worker waits for a huge compile slot – a helper serves its queue meanwhile ({n} helpers so far)");
    }
    let spawned = std::thread::Builder::new().name(format!("xcl-pso-h{k}")).stack_size(64 << 20).spawn(move || {
        NORMAL_WORKER.set(true);
        let mut retired = false;
        worker.run_while(|| {
            retired = retire_normal_helper();
            !retired
        });
        if !retired {
            NORMAL_HELPERS.fetch_sub(1, Ordering::AcqRel);
        }
    });
    if spawned.is_err() {
        NORMAL_HELPERS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Reserve this helper's retirement once more helpers exist than parked normal workers.
fn retire_normal_helper() -> bool {
    let mut helpers = NORMAL_HELPERS.load(Ordering::Acquire);
    while helpers > PARKED_NORMAL.load(Ordering::Acquire) {
        match NORMAL_HELPERS.compare_exchange(helpers, helpers - 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(now) => helpers = now,
        }
    }
    false
}

fn start_preload_workers() -> bool {
    static STARTED: OnceLock<bool> = OnceLock::new();
    *STARTED.get_or_init(|| {
        let q = queue();
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        let mut started = 0;
        for k in 0..n {
            let worker = q.worker_between(Priority::Speculative, Priority::Preload);
            if std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                PRELOAD_WORKER.set(true);
                worker.run();
            }).is_ok() {
                started += 1;
            }
        }
        started != 0
    })
}

fn run_reserved_rebuild(build: Arc<Build>, recipe: Recipe) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_NORMAL};
    let preload = PRELOAD_WORKER.get();
    BUILDING.with(|b| *b.borrow_mut() = Some(build.clone()));
    if preload {
        let mut demand = build.demand.lock().unwrap();
        demand.preload_thread = unsafe { GetCurrentThreadId() };
        if build.wanted.load(Ordering::Acquire) {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    let _running = RunningBuild { build: &build, preload };
    let _ = rebuild_reserved(&build, &recipe);
}

fn run_reserved_rehydrate(build: Arc<Build>, reservation: RehydrateReservation) {
    let recipe = reservation.recipe.clone();
    run_reserved_rebuild(build, recipe);
    // `rebuild_reserved` completed the turn. Drop is intentionally still armed so an unwind or a
    // future early return cannot strand Rebuilding; it observes inactive after normal completion.
    drop(reservation);
}

fn reserve_rehydrate_on(build: &Arc<Build>) -> Result<RehydrateReservation, RehydrateRequest> {
    let done = build.done.lock().unwrap();
    if !*done { return Err(RehydrateRequest::NotReady) }
    if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
        return Err(RehydrateRequest::Abandoned);
    }
    if build.real.load(Ordering::Acquire) != 0 { return Err(RehydrateRequest::Resident) }
    let Some(recipe) = build.recipe.clone() else { return Err(RehydrateRequest::Unavailable) };
    let turn = recipe.reserve_rehydrate()?;
    Ok(RehydrateReservation { recipe, turn })
}

fn request_rebuild_on(build: Arc<Build>, q: &BuildQueue) -> RebuildRequest {
    let recipe = {
        let done = build.done.lock().unwrap();
        if !*done {
            return RebuildRequest::NotReady;
        }
        if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            return RebuildRequest::Abandoned;
        }
        let Some(recipe) = build.recipe.clone() else { return RebuildRequest::Unavailable };
        if !recipe.reserve_rebuild() {
            return RebuildRequest::Coalesced;
        }
        recipe
    };
    let failed_recipe = recipe.clone();
    if q.submit(Priority::Speculative, Box::new(move || run_reserved_rebuild(build, recipe))).is_none() {
        failed_recipe.finish_rebuild(RebuildResult::Failed);
        return RebuildRequest::Unavailable;
    }
    RebuildRequest::Queued
}

fn request_rehydrate_on(build: Arc<Build>, q: &BuildQueue) -> RehydrateRequest {
    let reservation = match reserve_rehydrate_on(&build) {
        Ok(reservation) => reservation,
        Err(result) => return result,
    };
    if q.submit(Priority::Speculative, Box::new(move || run_reserved_rehydrate(build, reservation))).is_none() {
        // The refused work is dropped by `submit`, restoring Evicted/Failed before it returns.
        return RehydrateRequest::Unavailable;
    }
    RehydrateRequest::Queued
}

/// Queue one low-priority replacement build. This is intentionally uncalled until an eviction
/// policy has selected a pipeline; requesting never waits for driver work on the title thread.
#[allow(dead_code)]
fn request_rebuild(pso: usize) -> RebuildRequest {
    let Some(pending) = entry(pso) else { return RebuildRequest::Unavailable };
    if !start_preload_workers() {
        return RebuildRequest::Unavailable;
    }
    request_rebuild_on(pending.build.clone(), queue())
}

/// Queue low-priority reconstruction only for an absent recipe-backed generation. This is an
/// uncalled foundation: an eviction policy must decide when absence is acceptable before any bind
/// path may trigger it.
#[allow(dead_code)]
fn request_rehydrate(pso: usize) -> RehydrateRequest {
    let Some(pending) = entry(pso) else { return RehydrateRequest::Unavailable };
    if !start_preload_workers() {
        return RehydrateRequest::Unavailable;
    }
    request_rehydrate_on(pending.build.clone(), queue())
}

fn submit_job(q: &BuildQueue, build: &Arc<Build>, priority: Priority, preload: bool, preload_rank: Option<usize>) -> Option<JobHandle> {
    let queued_build = build.clone();
    q.submit_ranked(priority, preload_rank, Box::new(move || {
        let on_preload_worker = PRELOAD_WORKER.get();
        let built = build_queued(&queued_build);
        if built && preload && on_preload_worker {
            note_built_ahead();
        }
        queued_build.queued.lock().unwrap().take();
    }))
}

#[cfg(test)]
fn enqueue_on(q: &BuildQueue, build: &Arc<Build>, priority: Priority, preload: bool) {
    *build.queued.lock().unwrap() = submit_job(q, build, priority, preload, None);
}

fn schedule(build: &Arc<Build>, priority: Priority, preload: bool, preload_rank: Option<usize>) {
    if preload {
        if !start_preload_workers() {
            return;
        }
    } else if !start_normal_workers() {
        fail_unrunnable_build(build);
        return;
    }
    let mut queued = build.queued.lock().unwrap();
    let mut refused = false;
    if !build.sent.swap(true, Ordering::AcqRel) {
        let priority = if build.wanted.load(Ordering::Acquire) || build.by_submit.load(Ordering::Acquire) {
            Priority::Foreground
        } else {
            priority
        };
        *queued = submit_job(queue(), build, priority, preload, preload_rank);
        if queued.is_some() {
            build.note_queued();
        } else {
            refused = true;
        }
    }
    drop(queued);
    if refused {
        fail_unrunnable_build(build);
    }
}

/// No normal worker can run this job, or its queue refused it. Fail instead of leaving title
/// waiters asleep on work which cannot make progress.
fn fail_unrunnable_build(build: &Arc<Build>) {
    build.cancel_queued();
    let job = build.job.lock().unwrap().take();
    if let Some(job) = job {
        drop(job);
        complete(build, 0);
    }
}

fn build_queued(build: &Arc<Build>) -> bool {
    // (released by the title meanwhile)
    if build.orphan.load(Ordering::Acquire) {
        drop_unstarted_job(build);
        return false;
    }
    // (None: an orphaned build was dropped before this queued work ran)
    let job = build.job.lock().unwrap().take();
    if let Some(job) = job {
        run_build(build, job);
        true
    } else {
        false
    }
}

/// This thread is a preload or backlog worker whose build nobody waits for (yet).
fn build_is_backlog_only() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|build| {
        build.demand.lock().unwrap().preload_thread != 0 && !build.wanted.load(Ordering::Acquire)
    }))
}

/// The caller also drains its batch. Multiple workers can therefore build
/// different chains without ever waiting for a free worker to start a piece.
/// A preload or backlog worker's build nobody waits for keeps its pieces on its own thread: helpers
/// would go to the workers whose builds binds wait for, ahead of them.
pub(super) fn build_pieces(jobs: Vec<Job>) -> Vec<usize> {
    let count = jobs.len();
    let jobs = Arc::new(Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>()));
    let (send, recv) = mpsc::channel();
    let drain = move || loop {
        let job = jobs.lock().unwrap().pop();
        let Some((i, job)) = job else { return };
        let _ = send.send((i, run_job(job)));
    };
    // the helpers build for this worker's pipeline: the title waiting for it waits for them too
    let parent = BUILDING.with(|b| b.borrow().clone());
    let width = if build_is_backlog_only() { 1 } else { count.min(crate::gpu::recomp::dxc::compile_threads()) };
    if width > 1 {
        let _ = start_normal_workers();
    }
    for _ in 1..width {
        let (drain, parent) = (drain.clone(), parent.clone());
        let _ = queue().submit(Priority::BoundSoon, Box::new(move || {
            let before = BUILDING.with(|b| b.replace(parent));
            let _restore = RestoreBuilding(before);
            drain();
        }));
    }
    drain();
    let mut results = vec![0; count];
    for (i, result) in recv.iter().take(count) { results[i] = result; }
    results
}

/// Run one demanded auxiliary pipeline build on the foreground worker queue and wait for it.
/// Unlike `build_pieces`, the caller never drains or executes the job itself.
pub(super) fn build_foreground(job: Job) -> usize {
    let build = Arc::new(Build::new(job, false));
    build.want();
    schedule(&build, Priority::Foreground, false, None);
    let _held = Held::new();
    let _stall = crate::stall::span(crate::stall::StallKind::PipelineWait);
    let mut done = build.done.lock().unwrap();
    while !*done {
        done = build.cv.wait(done).unwrap();
    }
    build.real.load(Ordering::Acquire)
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_eager_with(standin, compute, job, 0, None, 0);
}

pub(super) fn submit_eager(
    standin: usize,
    compute: bool,
    job: Job,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
) {
    submit_eager_with(standin, compute, job, key, compute_key, 0);
}

/// `submit_eager`, with a compute pipeline's console program for the log (`record`; 0: not known).
pub(super) fn submit_eager_with(
    standin: usize,
    compute: bool,
    job: Job,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
    record: u64,
) {
    let key = if super::pso_manifest::enabled() { key } else { 0 };
    submit_as(standin, compute, job, None, true, key, compute_key, None, record);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`) –
/// and built ahead of it by the backlog when no run used them yet (`send_ahead`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_lazy_keyed_with(standin, compute, job, key, None, None, 0);
}

pub(super) fn submit_lazy_keyed(
    standin: usize,
    compute: bool,
    job: Job,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
) {
    submit_lazy_keyed_with(standin, compute, job, key, compute_key, None, 0);
}

/// `submit_lazy_keyed`, with the pipeline's translation and DXC compile on their own (`warm`): the
/// backlog does that for every pipeline it holds before the driver builds any of them. `record`:
/// a compute pipeline's console program, for the log (0: not known).
pub(super) fn submit_lazy_keyed_with(
    standin: usize,
    compute: bool,
    job: Job,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
    warm: Option<Warm>,
    record: u64,
) {
    submit_as(standin, compute, job, None, false, key, compute_key, warm, record);
}

pub(super) fn submit_recipe_keyed(
    standin: usize,
    compute: bool,
    recipe: Recipe,
    send: bool,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
) {
    submit_recipe_keyed_with(standin, compute, recipe, send, key, compute_key, None, 0);
}

/// `submit_recipe_keyed` with a lazy pipeline's translation ahead (`warm`) and a compute
/// pipeline's console program (`record`), as `submit_lazy_keyed_with` has them.
#[allow(clippy::too_many_arguments)]
pub(super) fn submit_recipe_keyed_with(
    standin: usize,
    compute: bool,
    recipe: Recipe,
    send: bool,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
    warm: Option<Warm>,
    record: u64,
) {
    let job = recipe.initial_job();
    submit_as(standin, compute, job, Some(recipe), send, key, compute_key, warm, record);
}

/// A lazy pipeline's translation and DXC compile without the driver's: what makes the DXIL
/// caches (in memory and on disk) hold its shaders, so its build is the driver's compile only.
pub type Warm = Box<dyn FnOnce() + Send>;

#[allow(clippy::too_many_arguments)]
fn submit_as(
    standin: usize,
    compute: bool,
    job: Job,
    recipe: Option<Recipe>,
    send: bool,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
    warm: Option<Warm>,
    record: u64,
) {
    debug_assert!(compute || compute_key.is_none());
    let kind = PipelineKind::submitted(compute, super::console_list::enabled());
    // (false: an earlier submit of the same compute build made it and decided where it goes)
    let mut created = true;
    let build = if let Some(compute_key) = compute_key {
        let mut removed = None;
        let mut retired_build = None;
        let build = {
            let mut shard = IN_FLIGHT.shard(&compute_key).lock().unwrap();
            let builds = shard.get_or_insert_with(HashMap::new);
            let active = builds.get(&compute_key).and_then(Weak::upgrade).and_then(|build| {
                if build.kind != kind {
                    retired_build = Some(build);
                    return None;
                }
                let done = build.done.lock().unwrap();
                let active = !*done && !build.orphan.load(Ordering::Acquire);
                if active {
                    build.owners.fetch_add(1, Ordering::Relaxed);
                }
                drop(done);
                if active {
                    Some(build)
                } else {
                    retired_build = Some(build);
                    None
                }
            });
            if let Some(build) = active {
                COALESCING.hit();
                created = false;
                build
            } else {
                // A dead, completed, or orphaned value is removed only when this exact key is
                // submitted again. Keep the removed Arc/Weak alive until after unlocking.
                removed = builds.remove_entry(&compute_key);
                if removed.is_some() {
                    COALESCING.finished(1);
                }
                let compute_key = Arc::new(compute_key);
                let build = Arc::new(
                    Build::new_keyed_kind(job, false, Some(compute_key.clone()), recipe, kind).with_submit(!send, record, warm),
                );
                builds.insert(compute_key, Arc::downgrade(&build));
                COALESCING.unique();
                build
            }
        };
        drop(removed);
        drop(retired_build);
        build
    } else {
        Arc::new(Build::new_keyed_kind(job, false, None, recipe, kind).with_submit(!send, record, warm))
    };
    let p = Arc::new(Pending {
        build: build.clone(),
        driver: AtomicUsize::new(0),
        compute,
        key,
        manifest_recipe: AtomicU64::new(0),
        track_used: !send,
        used: Default::default(),
        late: Default::default(),
        spared: Default::default(),
        on_use: Default::default(),
    });
    build.tracked.store(true, Ordering::Release);
    if !send {
        LAZY_MADE.fetch_add(1, Ordering::Relaxed);
        if !compute {
            LAZY_GRAPHICS.fetch_add(1, Ordering::Relaxed);
        }
        start_stats();
    }
    PENDING.insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        schedule(&build, Priority::BoundSoon, false, None);
    } else if let Some(rank) = used::rank(key).filter(|_| preload_enabled()) {
        schedule(&build, Priority::Preload, true, Some(rank));
    } else if created && key != 0 {
        // (shaders no earlier run used: the backlog builds it ahead of its first use, its
        // translation first when it has one, as commit and `XCL_PSO_AHEAD` allow)
        send_ahead(&build);
    }
}

/// Run on the thread of a lazy pipeline's first use, once (it is to hand work to another).
pub type OnUse = Box<dyn FnOnce() + Send>;

/// What the first use of lazy pipeline `standin` tells besides `used::note`: a graphics one's
/// description for the next run's preparation (`gfx_records::note`). Told before the title has
/// the stand-in.
#[cfg(test)]
pub(super) fn on_first_use(standin: usize, f: OnUse) {
    if let Some(p) = entry(standin) {
        *p.on_use.lock().unwrap() = Some(f);
    }
}

/// Attach the exact graphics recipe before the title can first use the returned stand-in.
pub(super) fn on_first_use_graphics(standin: usize, payload: Vec<u8>) {
    if let Some(p) = entry(standin) {
        p.manifest_recipe.store(super::gfx_records::payload_key(&payload), Ordering::Release);
        *p.on_use.lock().unwrap() = Some(Box::new(move || super::gfx_records::note(payload)));
    }
}

/// The shaders of a pipeline's set, told when the set's key is made (`real::OwnedDesc`).
pub(crate) fn shaders_of(key: u64, shaders: Vec<u64>) {
    used::parts(key, shaders);
}

/// Whether the active console profile is Xbox Series S (`lockhart`): checks `state::boot()`
/// first (already folded with `xcl.toml` and `XCL_DEVICE_TYPE` in the host process), then falls
/// back to `XCL_DEVICE_TYPE` for standalone `xcl shaders --used-only` and preparation workers
/// where no boot record is loaded.
pub(crate) fn profile_is_series_s() -> bool {
    profile_is_series_s_from(
        crate::state::boot().map(|b| b.config.is_series_s()),
        std::env::var("XCL_DEVICE_TYPE").ok().as_deref(),
    )
}

fn profile_is_series_s_from(boot_series_s: Option<bool>, env_device_type: Option<&str>) -> bool {
    if let Some(series_s) = boot_series_s {
        return series_s;
    }
    env_device_type.and_then(crate::config::parse_device_type) == Some("lockhart")
}

/// File name for a per-profile usage or replay manifest beside the DXIL cache: Series X
/// (`anaconda`, default) keeps `{stem}.{ext}` unchanged; Series S (`lockhart`) uses
/// `{stem}-lockhart.{ext}` so switching profiles does not cross-pollinate startup preparation
/// or runtime preload lists.
pub(crate) fn used_file_name(stem: &str, ext: &str, series_s: bool) -> String {
    if series_s { format!("{stem}-lockhart.{ext}") } else { format!("{stem}.{ext}") }
}

pub(crate) fn used_pipelines_name() -> String {
    used_file_name("used-pipelines", "txt", profile_is_series_s())
}

pub(crate) fn used_shaders_name() -> String {
    used_file_name("used-shaders", "txt", profile_is_series_s())
}

/// The shaders of the pipelines earlier runs used, each by its own key (`used-shaders.txt`
/// or `used-shaders-lockhart.txt` beside the DXIL cache): what a title's shader preparation
/// translates. Empty before a run with this list has used a pipeline.
pub fn earlier_used_shaders() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join(used_shaders_name())).unwrap_or_default())
}

/// The keys of the pipelines earlier runs used (`used-pipelines.txt` or
/// `used-pipelines-lockhart.txt` beside the DXIL cache): what
/// `xcl shaders --precompile --used-only` builds.
pub fn earlier_used_keys() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join(used_pipelines_name())).unwrap_or_default())
}

/// Ask the used-pipeline writer to finish without ever joining it unboundedly.
pub fn shutdown_used() {
    used::shutdown(std::time::Duration::from_millis(500));
}

/// Best effort at process termination: only the one writer touches the append handle, and a
/// loader-lock-blocked writer cannot hold termination longer than `timeout`.
pub fn flush_used_bounded(timeout: std::time::Duration) {
    used::shutdown(timeout);
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::{HashMap, HashSet};
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};
    use std::time::Duration;

    use super::super::used_journal::Journal;

    struct Used {
        earlier: RwLock<HashMap<u64, usize>>,
        now: Journal,
        /// the shaders of each set made in this run, by the set's key; the shaders of the sets
        /// used so far and `used-shaders.txt`, where they are kept for the next run
        parts: Mutex<std::collections::HashMap<u64, Vec<u64>>>,
        shaders: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    static U: OnceLock<Used> = OnceLock::new();

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        parse_ranked(text).into_keys().collect()
    }

    /// First occurrence order of the valid, nonzero keys in a `used-pipelines.txt`.
    pub fn parse_ranked(text: &str) -> HashMap<u64, usize> {
        let mut ranked = HashMap::new();
        for key in text.lines().filter_map(|line| u64::from_str_radix(line.trim(), 16).ok()).filter(|&key| key != 0) {
            let rank = ranked.len();
            ranked.entry(key).or_insert(rank);
        }
        ranked
    }

    fn get() -> &'static Used {
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used {
                    earlier: RwLock::new(HashMap::new()),
                    now: Journal::memory(HashSet::new()),
                    parts: Default::default(),
                    shaders: Mutex::new((HashSet::new(), None)),
                };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join(super::used_pipelines_name());
            let earlier = parse_ranked(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            let now = Journal::file(path.clone(), earlier.keys().copied().collect());
            let shaders_path = path.with_file_name(super::used_shaders_name());
            let shaders = parse(&std::fs::read_to_string(&shaders_path).unwrap_or_default());
            let shaders_file = std::fs::OpenOptions::new().create(true).append(true).open(&shaders_path).ok();
            Used {
                now,
                earlier: RwLock::new(earlier),
                parts: Default::default(),
                shaders: Mutex::new((shaders, shaders_file)),
            }
        })
    }

    pub fn shutdown(timeout: Duration) {
        if let Some(used) = U.get() {
            let _ = used.now.shutdown_bounded(timeout);
        }
    }

    /// First-use rank when an earlier run used a pipeline of these shaders.
    pub fn rank(key: u64) -> Option<usize> {
        (key != 0).then(|| get().earlier.read().unwrap().get(&key).copied()).flatten()
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        get().now.note(key);
        // its shaders, each once (also those of a set an earlier run used: the list of
        // shaders is younger than the list of sets)
        let parts = get().parts.lock().unwrap().get(&key).cloned().unwrap_or_default();
        let mut g = get().shaders.lock().unwrap();
        for part in parts {
            if g.0.insert(part) {
                if let Some(f) = g.1.as_mut() {
                    let _ = writeln!(f, "{part:016x}");
                }
            }
        }
    }

    /// The shaders of the set `key`, each by its own key (`real::shader_key`).
    pub fn parts(key: u64, shaders: Vec<u64>) {
        if key != 0 {
            get().parts.lock().unwrap().entry(key).or_insert(shaders);
        }
    }

    /// (tests) This run used the shader `key`.
    #[cfg(test)]
    pub fn shader_noted(key: u64) -> bool {
        get().shaders.lock().unwrap().0.contains(&key)
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.contains(key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        let mut earlier = get().earlier.write().unwrap();
        let rank = earlier.len();
        earlier.entry(key).or_insert(rank);
    }
}

/// Build pipelines from an earlier run in the preload worker band (`XCL_PSO_PRELOAD=0`: off).
fn preload_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0"))
}

/// The workers that build lazy pipelines no earlier run used ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off, with main's ranked preload), in the order of `Backlog`, on threads
/// of their own (`xcl-pso-ahead-*`) beside main's preload band (`xcl-pso-pre-*`: the pipelines of
/// earlier runs, `used::rank`), at the lowest priority: a pipeline that is bound is not held up
/// by the ones made ahead (whoever takes `job` first builds it; a bind's wait raises the worker
/// building it), and the title's own threads are not either.
struct Preload {
    list: Mutex<Backlog>,
    cv: Condvar,
}

/// The lazy pipelines no earlier run used, to build ahead of their first use, in this order: the
/// compute ones binds go on without (`spared`); the title's other compute pipelines; then its
/// graphics pipelines, first translated and compiled by DXC all of them (`Warm`: milliseconds to
/// seconds each, no driver memory kept, and the disk caches hold them for the next run), then
/// built by the driver one by one. (Those of earlier runs are main's ranked preload.) A retail
/// title made its ~2 700 pipelines in its first 6 s and bound some of them for the first time up
/// to 150 s later: each such bind waited 7 – 15 s for its pipeline, one after another, and no
/// frame came for 156 s. A compute pipeline is waited for at its first dispatch (or its
/// dispatches are skipped: `XCL_CS_PSO_WAIT_MS`), and one no earlier run used was built only
/// then: a retail title's thread spent 57 s in the driver on one. Those are built by one worker
/// at a time (`compute_out`): a title has ~1 600 of them, some a minute each in the driver, and
/// with every worker on them the graphics pipelines would wait for all of them and be built at
/// their first draws instead. A pipeline a bind wants is not taken from here (`unclaimed`): the
/// workers build it at once, and a huge driver compile ahead of its use gives its turn to any
/// other and takes no giant slot (`real::big_compile_turn`).
#[derive(Default)]
struct Backlog {
    /// compute pipelines no earlier run used
    compute: std::collections::VecDeque<Arc<Build>>,
    /// a worker builds one of `compute` or `spared` (`next` hands out no other until `built`)
    compute_out: bool,
    /// compute pipelines binds go on without while they are built (`image_only`), sent here at
    /// their first bind (`send_spared`): used, so built before `compute` and with no room or cap
    spared: std::collections::VecDeque<Arc<Build>>,
    warm: std::collections::VecDeque<Arc<Build>>,
    cold: std::collections::VecDeque<Arc<Build>>,
}

/// What a backlog worker does with a pipeline.
enum Step {
    /// its build (`ahead`: of a pipeline nobody used yet – counts against `ahead_cap`;
    /// `new_compute`: in the lane of `Backlog::compute`, `Backlog::built` after it)
    Build { ahead: bool, new_compute: bool },
    /// its translation and DXC compile; then back to the backlog for its build
    Warm(Warm),
}

impl Backlog {
    fn push(&mut self, build: Arc<Build>) {
        if build.is_compute() {
            self.compute.push_back(build);
        } else if build.warm.lock().unwrap().is_some() {
            self.warm.push_back(build);
        } else {
            self.cold.push_back(build);
        }
    }

    fn len(&self) -> usize {
        self.compute.len() + self.spared.len() + self.warm.len() + self.cold.len()
    }

    /// The next piece of work. `ahead`: pipelines nobody used yet may be worked on now
    /// (`XCL_PSO_AHEAD`, free commit: `ahead_room`); `builds`: one more of them may be built by
    /// the driver (`ahead_cap`). Without them those wait here.
    fn next(&mut self, ahead: bool, builds: bool) -> Option<(Arc<Build>, Step)> {
        // (`sent`: its copy in another lane is passed over there)
        if !self.compute_out {
            while let Some(build) = self.spared.pop_front() {
                if build.untaken() {
                    self.compute_out = true;
                    return Some((build, Step::Build { ahead: false, new_compute: true }));
                }
            }
        }
        if ahead && builds && !self.compute_out {
            while let Some(build) = self.compute.pop_front() {
                if build.unclaimed() {
                    self.compute_out = true;
                    return Some((build, Step::Build { ahead: true, new_compute: true }));
                }
            }
        }
        if !ahead {
            return None;
        }
        while let Some(build) = self.warm.pop_front() {
            if !build.unclaimed() {
                continue;
            }
            let warm = build.warm.lock().unwrap().take();
            match warm {
                Some(w) => return Some((build, Step::Warm(w))),
                None => self.cold.push_back(build),
            }
        }
        if !builds {
            return None;
        }
        while let Some(build) = self.cold.pop_front() {
            if build.unclaimed() {
                return Some((build, Step::Build { ahead: true, new_compute: false }));
            }
        }
        None
    }

    /// The build `next` handed out as `Step::Build` is over (built, or taken by a bind).
    fn built(&mut self, new_compute: bool) {
        if new_compute {
            self.compute_out = false;
        }
    }
}

/// Lazy pipelines no earlier run used that the driver builds ahead of their first use
/// (`XCL_PSO_AHEAD`, default 4096; 0: none, and none translated ahead either – only those of
/// earlier runs are built ahead then). Each pipeline built keeps ~400 KB in the driver whether
/// the title uses it or not: a retail game made ~51 000 and used few.
fn ahead_cap() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| ahead_cap_of(std::env::var("XCL_PSO_AHEAD").ok().as_deref()))
}

fn ahead_cap_of(v: Option<&str>) -> usize {
    v.and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(4096)
}

/// Commit that must be free for building ahead pipelines no earlier run used: the 6 GB DXC's
/// compiles keep free for a running title (`dxc::wait_for_memory`) and 2 GB more. Below it they
/// wait in the backlog (pipelines of an earlier run are built as before).
const AHEAD_FREE: u64 = 8 << 30;

fn ahead_room() -> bool {
    ahead_cap() > 0 && super::real::big_compile_commit_free(AHEAD_FREE)
}

/// pipelines no earlier run used: translated ahead, built ahead by the backlog (`ahead_cap`);
/// pipelines in the backlog
static WARMED: AtomicUsize = AtomicUsize::new(0);
static AHEAD_BUILDS: AtomicUsize = AtomicUsize::new(0);
static BACKLOG: AtomicUsize = AtomicUsize::new(0);

/// One more pipeline built ahead of its first use (by the backlog, or main's ranked preload):
/// the log line at powers of two, one count for both.
fn note_built_ahead() {
    static DONE: AtomicUsize = AtomicUsize::new(0);
    let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
    }
}

/// The backlog and its workers, started at the first use (None: `XCL_PSO_PRELOAD=0`).
fn preload_backlog() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| preload_enabled().then(|| Preload { list: Mutex::new(Backlog::default()), cv: Condvar::new() }))
        .as_ref()?;
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for _ in 0..n {
            spawn_backlog_worker(q);
        }
    });
    Some(q)
}

/// One more worker of the backlog (`preload_backlog`; `build_parks` starts one in a parked one's
/// place). It builds through `run_build` as a preload worker: a bind's wait raises it.
fn spawn_backlog_worker(q: &'static Preload) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let k = NEXT.fetch_add(1, Ordering::Relaxed);
    let _ = std::thread::Builder::new().name(format!("xcl-pso-ahead-{k}")).stack_size(64 << 20).spawn(move || {
        unsafe {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
        }
        PRELOAD_WORKER.set(true);
        loop {
            let (build, step) = {
                let mut g = q.list.lock().unwrap();
                loop {
                    let next = g.next(ahead_room(), AHEAD_BUILDS.load(Ordering::Relaxed) < ahead_cap());
                    BACKLOG.store(g.len(), Ordering::Relaxed);
                    if let Some(next) = next {
                        break next;
                    }
                    // (pipelines waiting for free commit are looked at again in a second)
                    g = q.cv.wait_timeout(g, std::time::Duration::from_secs(1)).unwrap().0;
                }
            };
            let (ahead, new_compute) = match step {
                Step::Warm(warm) => {
                    // (a translation that panics loses its warm-up, not this worker)
                    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(warm)) {
                        crate::xlog_error!("D3D12.X: panic while translating a pipeline ahead: {}", panic_message(payload.as_ref()));
                    }
                    WARMED.fetch_add(1, Ordering::Relaxed);
                    if build.unclaimed() {
                        q.list.lock().unwrap().cold.push_back(build);
                    }
                    continue;
                }
                Step::Build { ahead, new_compute } => (ahead, new_compute),
            };
            // (released meanwhile: its job goes; None: a bind's worker took it and builds it)
            let job = if build.orphan.load(Ordering::Acquire) {
                drop_unstarted_job(&build);
                None
            } else {
                build.job.lock().unwrap().take()
            };
            if let Some(job) = job {
                if ahead {
                    AHEAD_BUILDS.fetch_add(1, Ordering::Relaxed);
                }
                PRELOAD_STEP.with(|s| s.set(Some(new_compute)));
                run_build(&build, job);
                let parked = PRELOAD_STEP.with(|s| s.take()).is_none();
                note_built_ahead();
                // (its lane went back and another worker took its place: `build_parks`)
                if parked {
                    return;
                }
            }
            q.list.lock().unwrap().built(new_compute);
        }
    });
}

/// A lazy pipeline no earlier run used (called by `submit_as` for a Build it created, `key` != 0,
/// not known to `used::rank`): to the backlog, built ahead of its first use. Nothing with
/// `XCL_PSO_AHEAD=0` or without the backlog.
fn send_ahead(build: &Arc<Build>) {
    if ahead_cap() == 0 {
        return;
    }
    let Some(q) = preload_backlog() else { return };
    let mut b = q.list.lock().unwrap();
    b.push(build.clone());
    BACKLOG.store(b.len(), Ordering::Relaxed);
    q.cv.notify_one();
}

/// The build of a pipeline binds do not wait for (`image_only`) goes to the backlog `q`, in the
/// lane of new compute pipelines (`Backlog::spared`: one at a time), not to the workers builds a
/// bind waits for go to: there each of them, unwanted, waited for the one unwanted slot of huge
/// driver compiles, and with more of them than workers a build a bind waited for stood in the
/// queue behind them. The caller has set `build.sent` (`resolve_spared`). Without the backlog
/// (`XCL_PSO_PRELOAD=0`): to those workers.
fn send_spared(build: &Arc<Build>, q: Option<&Preload>) {
    let Some(q) = q else {
        // (`schedule` sends only a build not sent yet)
        build.sent.store(false, Ordering::Release);
        schedule(build, Priority::BoundSoon, false, None);
        return;
    };
    let mut b = q.list.lock().unwrap();
    b.spared.push_back(build.clone());
    BACKLOG.store(b.len(), Ordering::Relaxed);
    q.cv.notify_one();
}
#[cfg(test)]
mod used_tests {
    use super::*;

    /// The shaders of a set are kept when a pipeline of the set is used, each once, and not
    /// before: a set that is only made keeps nothing.
    #[test]
    fn the_shaders_of_a_used_set_are_kept_for_the_next_run() {
        let (set, other) = (0x5e70_0000_0000_0001u64, 0x5e70_0000_0000_0003u64);
        let (vs, ps, cs) = (0x5e70_0000_0000_0011u64, 0x5e70_0000_0000_0013u64, 0x5e70_0000_0000_0015u64);
        shaders_of(set, vec![vs, ps]);
        shaders_of(other, vec![cs]);
        assert!(!used::shader_noted(vs) && !used::shader_noted(ps), "made, not used yet");
        used::note(set);
        assert!(used::shader_noted(vs) && used::shader_noted(ps), "the shaders of the used set");
        assert!(!used::shader_noted(cs), "the other set was not used");
    }

    #[repr(C)]
    struct RebuildFakePso {
        vtable: *const usize,
        refs: AtomicUsize,
    }

    unsafe extern "system" fn rebuild_fake_add_ref(this: *mut std::ffi::c_void) -> u32 {
        (*(this as *mut RebuildFakePso)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn rebuild_fake_release(this: *mut std::ffi::c_void) -> u32 {
        (*(this as *mut RebuildFakePso)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    thread_local! {
        static RESTORE_REENTER_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static EVICTION_RELEASE_PROBE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    struct EvictionReleaseProbe {
        done: *const Mutex<bool>,
        rebuild: *const Mutex<RebuildState>,
        generation: *const Mutex<GenerationSlot<ID3D12PipelineState>>,
        admission: *const Mutex<()>,
        timestamp_access: *const AtomicUsize,
        observer: *const Mutex<PressureObserver>,
        pressure_stats: *const PipelinePressureCounters,
    }

    static PIPELINE_RESTORATION_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn pipeline_restoration_test_guard() -> std::sync::MutexGuard<'static, ()> {
        PIPELINE_RESTORATION_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    unsafe extern "system" fn rebuild_reenter_release(this: *mut std::ffi::c_void) -> u32 {
        let remaining = rebuild_fake_release(this);
        if remaining == 0 {
            RESTORE_REENTER_ATTEMPTS.with(|address| if address.get() != 0 { assert!(
                (&*(address.get() as *const Mutex<Vec<Arc<RestorationAttempt>>>)).try_lock().is_ok(),
                "final native drop ran under the attempts lock",
            ) });
            EVICTION_RELEASE_PROBE.with(|address| if address.get() != 0 {
                let probe = &*(address.get() as *const EvictionReleaseProbe);
                assert!((&*probe.done).try_lock().is_ok());
                assert!((&*probe.rebuild).try_lock().is_ok());
                assert!((&*probe.generation).try_lock().is_ok());
                assert!((&*probe.admission).try_lock().is_ok());
                assert_eq!((&*probe.timestamp_access).load(Ordering::Acquire), 0);
                if !probe.observer.is_null() {
                    assert!((&*probe.observer).try_lock().is_ok(),
                        "final native drop ran under the pressure observer lock");
                }
                if !probe.pressure_stats.is_null() {
                    assert_eq!((&*probe.pressure_stats).generation_owners_released.load(Ordering::Relaxed), 0,
                        "generation release was counted before its COM owner dropped");
                }
            });
        }
        remaining
    }

    fn pressure_config() -> PressureAdmissionConfig {
        PressureAdmissionConfig {
            enabled: true,
            low_basis_points: 7_000,
            high_basis_points: 9_000,
            max_sample_age_turns: 1,
            max_evictions_per_turn: 3,
            rehydrate_backoff_turns: 2,
            max_rehydrate_backoff_turns: 8,
        }
    }

    fn pressure_sample(turn: u64, usage: u64) -> PressureSample {
        PressureSample {
            observed_turn: turn,
            segment: super::super::memory_report::Segment { usage, budget: 100 },
        }
    }

    fn ledger_build() -> Arc<Build> {
        Arc::new(Build::new(Box::new(|| 0), false))
    }

    fn ledger_native(id: u64) -> LedgerNativeGeneration {
        LedgerNativeGeneration { id, identity: id as usize + 0x1000 }
    }

    fn completed_recipe_build_as(recipe: &Recipe, kind: PipelineKind) -> Arc<Build> {
        let build = Arc::new(Build::new_keyed_kind(
            recipe.initial_job(), false, None, Some(recipe.clone()), kind,
        ));
        let job = build.job.lock().unwrap().take().unwrap();
        complete(&build, job());
        build
    }

    fn completed_recipe_build(recipe: &Recipe) -> Arc<Build> {
        completed_recipe_build_as(recipe, PipelineKind::Compute)
    }

    fn selected_candidate(build: &Arc<Build>) -> SelectedEvictionCandidate {
        SelectedEvictionCandidate {
            candidate: resident_candidate(build).unwrap(), unused_frames: 100, resident_frames: 100,
        }
    }

    fn assert_shadow_blocks_restoration(
        selected: &SelectedEvictionCandidate,
        coordinator: &Arc<RestorationCoordinator<2, 8>>,
        action: impl FnOnce(Arc<RestorationCoordinator<2, 8>>, EvictedBuildToken) + Send + 'static,
    ) {
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let mut worker = None;
        let limits = EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        };
        assert_eq!(preflight_exact_eviction_on(selected, 1, limits, coordinator, || {
            let token = selected.candidate.build.eviction_token().unwrap();
            let coordinator = coordinator.clone();
            worker = Some(std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                action(coordinator, token);
                done_tx.send(()).unwrap();
            }));
            started_rx.recv().unwrap();
            assert!(done_rx.recv_timeout(std::time::Duration::from_millis(20)).is_err());
        }), EvictionPreflight::Prepared);
        done_rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        worker.unwrap().join().unwrap();
    }

    #[test]
    fn pressure_admission_rejects_ambiguous_serial_horizons() {
        let mut config = pressure_config();
        config.max_sample_age_turns = PRESSURE_MAX_TURN_HORIZON;
        config.rehydrate_backoff_turns = PRESSURE_MAX_TURN_HORIZON;
        config.max_rehydrate_backoff_turns = PRESSURE_MAX_TURN_HORIZON;
        assert!(PressureAdmission::new(config).is_some(), "the exact serial horizon was rejected");

        let ambiguous = PRESSURE_MAX_TURN_HORIZON + 1;
        let mut sample_age = pressure_config();
        sample_age.max_sample_age_turns = ambiguous;
        assert!(PressureAdmission::new(sample_age).is_none());
        let mut initial_backoff = pressure_config();
        initial_backoff.rehydrate_backoff_turns = ambiguous;
        initial_backoff.max_rehydrate_backoff_turns = ambiguous;
        assert!(PressureAdmission::new(initial_backoff).is_none());
        let mut maximum_backoff = pressure_config();
        maximum_backoff.max_rehydrate_backoff_turns = ambiguous;
        assert!(PressureAdmission::new(maximum_backoff).is_none());
    }

    #[test]
    fn pressure_admission_maximum_cooldown_does_not_appear_expired() {
        let mut config = pressure_config();
        config.max_sample_age_turns = PRESSURE_MAX_TURN_HORIZON;
        config.rehydrate_backoff_turns = PRESSURE_MAX_TURN_HORIZON;
        config.max_rehydrate_backoff_turns = PRESSURE_MAX_TURN_HORIZON;
        let mut state = PressureAdmission::new(config).unwrap();
        assert_eq!(state.decide(0, Some(pressure_sample(0, 95))).max_evictions, 3);
        assert!(state.decide(1, Some(pressure_sample(1, 60))).allow_rehydrate);
        state.note_rehydrate_result(1, false);
        assert!(!state.decide(2, Some(pressure_sample(2, 60))).allow_rehydrate,
            "a maximum unambiguous cooldown appeared immediately expired");
        assert!(!state.decide(PRESSURE_MAX_TURN_HORIZON,
            Some(pressure_sample(PRESSURE_MAX_TURN_HORIZON, 60))).allow_rehydrate);
        let retry = PRESSURE_MAX_TURN_HORIZON + 1;
        assert!(state.decide(retry, Some(pressure_sample(retry, 60))).allow_rehydrate);
    }

    #[test]
    fn pressure_admission_hysteresis_bounds_each_eviction_turn() {
        let mut state = PressureAdmission::new(pressure_config()).unwrap();
        assert_eq!(state.decide(1, Some(pressure_sample(1, 85))), PressureDecision::default());
        assert_eq!(state.decide(2, Some(pressure_sample(2, 90))).max_evictions, 3);
        assert_eq!(state.decide(3, Some(pressure_sample(3, 85))).max_evictions, 3,
            "falling below the high watermark churned the policy");
        assert!(state.decide(4, Some(pressure_sample(4, 70))).allow_rehydrate);
        assert!(state.decide(5, Some(pressure_sample(5, 85))).allow_rehydrate,
            "rising above the low watermark churned the policy");
        assert_eq!(state.decide(6, Some(pressure_sample(6, 90))).max_evictions, 3);
    }

    #[test]
    fn pressure_admission_charges_only_successful_evictions_within_a_turn() {
        let mut state = PressureAdmission::new(pressure_config()).unwrap();
        assert_eq!(state.decide(10, Some(pressure_sample(9, 95))).max_evictions, 3);
        state.note_evictions(10, 0);
        assert_eq!(state.evictions_succeeded, 0, "a failed attempt spent the success allowance");
        state.note_evictions(10, 1);
        assert_eq!(state.decide(10, Some(pressure_sample(10, 95))).max_evictions, 2);
        state.note_evictions(10, 2);
        assert_eq!(state.evictions_succeeded, 3,
            "successful evictions did not exhaust the same-turn allowance");
        state.note_evictions(9, 3);
        assert_eq!(state.evictions_succeeded, 3,
            "an out-of-turn result changed the current allowance");
        assert_eq!(state.decide(11, Some(pressure_sample(11, 95))).max_evictions, 3,
            "a new policy turn did not replenish its bounded allowance");
    }

    #[test]
    fn pressure_attempt_telemetry_accounts_shadow_and_eviction_outcomes_separately() {
        let mut shadow = PressureObservation::default();
        shadow.note_preflight(EvictionPreflight::Prepared);
        shadow.note_preflight(EvictionPreflight::Busy);
        assert_eq!((shadow.attempted, shadow.prepared, shadow.evicted), (2, 1, 0));
        assert!(shadow.attempts_accounted());

        let detached = PressureObservation { attempted: 2, evicted: 1, recent: 1, ..Default::default() };
        assert!(detached.attempts_accounted());
        assert!(!PressureObservation { attempted: 2, evicted: 1, ..Default::default() }.attempts_accounted());
    }

    #[test]
    fn pressure_admission_rejects_missing_invalid_and_stale_samples() {
        let mut state = PressureAdmission::new(pressure_config()).unwrap();
        assert_eq!(state.decide(1, Some(pressure_sample(1, 95))).max_evictions, 3);
        assert_eq!(state.decide(2, None), PressureDecision::default());
        assert_eq!(state.decide(2, Some(pressure_sample(1, 95))), PressureDecision::default());
        assert_eq!(state.decide(4, Some(pressure_sample(2, 50))), PressureDecision::default());
        let invalid = PressureSample {
            observed_turn: 4,
            segment: super::super::memory_report::Segment { usage: 95, budget: 0 },
        };
        assert_eq!(state.decide(4, Some(invalid)), PressureDecision::default());
        assert_eq!(state.decide(5, Some(pressure_sample(5, 85))).max_evictions, 3,
            "an unusable sample changed pressure state");

        let mut disabled = PressureAdmission::new(PressureAdmissionConfig::default()).unwrap();
        assert_eq!(disabled.decide(1, Some(pressure_sample(1, 100))), PressureDecision::default());
    }

    #[test]
    fn pressure_admission_freshness_is_conservative_across_turn_wrap() {
        let mut state = PressureAdmission::new(pressure_config()).unwrap();
        assert_eq!(state.decide(u64::MAX, Some(pressure_sample(0, 95))), PressureDecision::default(),
            "a post-wrap sample was accepted before its turn");
        assert_eq!(state.decide(u64::MAX, Some(pressure_sample(u64::MAX, 95))).max_evictions, 3);
        assert_eq!(state.decide(0, Some(pressure_sample(0, 95))).max_evictions, 3,
            "turn wrap did not advance the sample and eviction budget");
        assert_eq!(state.decide(0, Some(pressure_sample(u64::MAX, 95))), PressureDecision::default(),
            "a pre-wrap sample was accepted twice");
        assert_eq!(state.decide(2, Some(pressure_sample(0, 95))), PressureDecision::default(),
            "a wrapped sample older than the configured age was accepted");

        let mut recovery = PressureAdmission::new(pressure_config()).unwrap();
        assert_eq!(recovery.decide(u64::MAX - 2, Some(pressure_sample(u64::MAX - 2, 95))).max_evictions, 3);
        assert!(recovery.decide(u64::MAX - 1, Some(pressure_sample(u64::MAX - 1, 60))).allow_rehydrate);
        recovery.note_rehydrate_result(u64::MAX - 1, false);
        assert!(!recovery.decide(u64::MAX, Some(pressure_sample(u64::MAX, 60))).allow_rehydrate);
        assert!(recovery.decide(0, Some(pressure_sample(0, 60))).allow_rehydrate,
            "rehydration backoff did not expire across turn wrap");
    }

    #[test]
    fn pressure_admission_backs_off_failures_and_resets_after_recovery() {
        let mut state = PressureAdmission::new(pressure_config()).unwrap();
        assert_eq!(state.decide(1, Some(pressure_sample(1, 95))).max_evictions, 3);
        assert!(state.decide(2, Some(pressure_sample(2, 60))).allow_rehydrate);
        state.note_rehydrate_result(2, false);
        assert!(!state.decide(3, Some(pressure_sample(3, 60))).allow_rehydrate);
        assert!(state.decide(4, Some(pressure_sample(4, 60))).allow_rehydrate);
        state.note_rehydrate_result(4, false);
        assert!(!state.decide(7, Some(pressure_sample(7, 60))).allow_rehydrate);
        assert!(state.decide(8, Some(pressure_sample(8, 60))).allow_rehydrate);
        state.note_rehydrate_result(8, true);
        assert!(state.decide(9, Some(pressure_sample(9, 60))).allow_rehydrate);
        state.note_rehydrate_result(9, false);
        assert!(!state.decide(10, Some(pressure_sample(10, 60))).allow_rehydrate,
            "a successful recovery did not reset the failure backoff");
        assert!(state.decide(11, Some(pressure_sample(11, 60))).allow_rehydrate);
        state.note_rehydration_complete();
        assert_eq!(state.decide(12, Some(pressure_sample(12, 60))), PressureDecision::default());
    }

    #[test]
    fn evicted_ledger_reserves_before_detach_and_stays_bounded() {
        let ledger = EvictedBuildLedger::<2>::new(7).unwrap();
        let builds = [ledger_build(), ledger_build(), ledger_build()];
        drop(ledger.reserve(&builds[0], ledger_native(1)).unwrap());
        assert!(ledger.records(2).is_empty(), "a pre-detach reservation drop stayed live");

        let first = ledger.reserve(&builds[0], ledger_native(1)).unwrap().arm_detach().unwrap();
        let second = ledger.reserve(&builds[1], ledger_native(2)).unwrap().arm_detach().unwrap();
        assert!(ledger.reserve(&builds[2], ledger_native(3)).is_none(), "a full ledger admitted a detach");
        assert_eq!(ledger.records(1).len(), 1, "enumeration ignored its bound");
        let failed = first.clone();
        drop(first);
        assert_eq!(ledger.records(2).len(), 2, "dropping an armed token erased detached state");
        assert!(ledger.finish(&failed, EvictedBuildState::Poisoned));
        assert_eq!(ledger.records(2)[0].state, EvictedBuildState::Poisoned);
        assert!(ledger.finish(&second, EvictedBuildState::Released));
    }

    #[test]
    fn evicted_ledger_large_capacity_stays_off_the_caller_stack() {
        assert!(std::mem::size_of::<EvictedBuildLedger<1024>>() < 4096);
        let ledger = EvictedBuildLedger::<1024>::new(8).unwrap();
        assert_eq!(ledger.inner.lock().unwrap().slots.len(), 1024);
    }

    #[test]
    fn evicted_ledger_instance_identity_prevents_cross_ledger_mutation() {
        let first = EvictedBuildLedger::<1>::new(7).unwrap();
        let second = EvictedBuildLedger::<1>::new(7).unwrap();
        let build = ledger_build();
        let native = ledger_native(3);
        let first_token = first.reserve(&build, native).unwrap().arm_detach().unwrap();
        let second_token = second.reserve(&build, native).unwrap().arm_detach().unwrap();
        assert_ne!(first_token.ledger_instance, second_token.ledger_instance);
        assert!(!second.rehydrate_queued(&first_token), "a token crossed identical caller generations");
        assert_eq!(second.records(1)[0].state, EvictedBuildState::Evicted);
        assert!(second.rehydrate_queued(&second_token));
    }

    #[test]
    fn evicted_ledger_rejects_stale_identity_and_out_of_order_terminal() {
        let ledger = EvictedBuildLedger::<1>::new(9).unwrap();
        let first_build = ledger_build();
        let first = ledger.reserve(&first_build, ledger_native(4)).unwrap().arm_detach().unwrap();
        let mut wrong_native = first.clone();
        wrong_native.native = ledger_native(5);
        assert!(!ledger.rehydrate_queued(&wrong_native));
        let mut wrong_ledger = first.clone();
        wrong_ledger.ledger_generation += 1;
        assert!(!ledger.rehydrate_queued(&wrong_ledger));
        assert!(!ledger.finish(&first, EvictedBuildState::Restored),
            "an evicted build was restored before rehydration queued");
        assert!(ledger.rehydrate_queued(&first));
        assert!(ledger.finish(&first, EvictedBuildState::Restored));
        assert!(!ledger.finish(&first, EvictedBuildState::Released), "a terminal state changed out of order");
        assert!(ledger.retire(&first));

        let second_build = ledger_build();
        let second = ledger.reserve(&second_build, ledger_native(5)).unwrap().arm_detach().unwrap();
        assert_eq!(first.slot, second.slot);
        assert_ne!(first.slot_generation, second.slot_generation);
        assert!(!ledger.finish(&first, EvictedBuildState::Poisoned), "a stale token mutated a reused slot");
        assert_eq!(ledger.records(1)[0].state, EvictedBuildState::Evicted);
        assert!(ledger.rehydrate_queued(&second));
        assert!(ledger.finish(&second, EvictedBuildState::Poisoned));
        assert!(ledger.retire(&second));

        let third = ledger.reserve(&second_build, ledger_native(6)).unwrap().arm_detach().unwrap();
        assert!(ledger.rehydrate_queued(&third));
        assert!(ledger.finish(&third, EvictedBuildState::Released));
    }

    #[test]
    fn evicted_ledger_retires_expired_weak_build_identity() {
        let ledger = EvictedBuildLedger::<1>::new(11).unwrap();
        let build = ledger_build();
        let token = ledger.reserve(&build, ledger_native(6)).unwrap().arm_detach().unwrap();
        drop(build);
        assert_eq!(ledger.records(1)[0].state, EvictedBuildState::Released);

        let replacement = ledger_build();
        let mut reused_address = token.clone();
        reused_address.build = Arc::downgrade(&replacement);
        assert!(!ledger.retire(&reused_address), "a different allocation identity retired the entry");
        assert!(ledger.retire(&token));
    }

    #[test]
    fn evicted_ledger_counter_overflow_fails_closed() {
        let build = ledger_build();
        let serial = EvictedBuildLedger::<1>::new(13).unwrap();
        serial.inner.lock().unwrap().serial = u64::MAX;
        assert!(serial.reserve(&build, ledger_native(7)).is_none());
        assert!(serial.inner.lock().unwrap().exhausted);

        let generation = EvictedBuildLedger::<1>::new(14).unwrap();
        generation.inner.lock().unwrap().slots[0].generation = u64::MAX;
        assert!(generation.reserve(&build, ledger_native(8)).is_none());
        assert!(generation.inner.lock().unwrap().exhausted);

        let instance = AtomicU64::new(u64::MAX - 1);
        assert_eq!(allocate_evicted_ledger_instance(&instance), Some(u64::MAX - 1));
        assert_eq!(allocate_evicted_ledger_instance(&instance), None);
    }

    fn receipt_evicted(build: &Arc<Build>, id: u64) -> EvictedBuildToken {
        EvictedBuildLedger::<1>::new(id + 100).unwrap()
            .reserve(build, ledger_native(id)).unwrap().arm_detach().unwrap()
    }

    fn coordinated_evicted<const E: usize, const R: usize>(
        coordinator: &RestorationCoordinator<E, R>, build: &Arc<Build>, id: u64,
    ) -> EvictedBuildToken {
        coordinator.reserve_eviction(build, ledger_native(id)).unwrap().arm_detach().unwrap()
    }

    #[test]
    fn rehydrate_receipts_separate_request_from_completion() {
        let build = ledger_build(); let evicted = receipt_evicted(&build, 20);
        let registry = RehydrateReceiptRegistry::<3>::new().unwrap();
        let (request, refused) = registry.reserve(&evicted).unwrap().refused().unwrap();
        assert_eq!(request, RehydrateRequestReceipt::Refused);
        assert_eq!(registry.records(3)[0].state, RehydrateCompletion::Refused); assert!(registry.retire(&refused));
        let (request, queued) = registry.reserve(&evicted).unwrap()
            .accepted(RehydratePriority::Background).unwrap();
        assert_eq!(request, RehydrateRequestReceipt::Queued { priority: RehydratePriority::Background });
        let queued_token = queued.token.clone();
        assert_eq!(registry.records(3)[0].state, RehydrateCompletion::Queued,
            "queue acceptance was treated as successful completion");
        assert!(registry.promote(&queued_token));
        assert_eq!(registry.records(3)[0].priority, RehydratePriority::Foreground);
        drop(queued);
        assert_eq!(registry.records(3)[0].state, RehydrateCompletion::Cancelled);
    }

    #[test]
    fn rehydrate_receipts_large_capacity_stays_off_the_caller_stack() {
        assert!(std::mem::size_of::<RehydrateReceiptRegistry<1024>>() < 4096);
        let registry = RehydrateReceiptRegistry::<1024>::new().unwrap();
        assert_eq!(registry.inner.lock().unwrap().slots.len(), 1024);
    }

    #[test]
    fn rehydrate_receipts_report_panic_failure_and_release() {
        let build = ledger_build(); let evicted = receipt_evicted(&build, 21);
        let registry = RehydrateReceiptRegistry::<4>::new().unwrap();
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (_, _queued) = registry.reserve(&evicted).unwrap()
                .accepted(RehydratePriority::Foreground).unwrap();
            panic!("queued build panic");
        }));
        assert!(panic_result.is_err()); assert_eq!(registry.records(4)[0].state, RehydrateCompletion::Panicked);
        let (_, failed) = registry.reserve(&evicted).unwrap().accepted(RehydratePriority::Background).unwrap();
        assert!(failed.complete(RehydrateCompletion::BuildFailed));
        let (_, released) = registry.reserve(&evicted).unwrap().accepted(RehydratePriority::Background).unwrap();
        assert!(released.complete(RehydrateCompletion::Released));
    }

    #[test]
    fn rehydrate_receipts_reject_duplicate_stale_and_release_races() {
        let build = ledger_build(); let evicted = receipt_evicted(&build, 23);
        let registry = RehydrateReceiptRegistry::<1>::new().unwrap();
        let reservation = registry.reserve(&evicted).unwrap();
        assert!(!registry.publish(&reservation.token, RehydrateCompletion::BuildFailed),
            "a completion published before queue acceptance");
        let (_, success) = reservation.accepted(RehydratePriority::Background).unwrap();
        let old = success.token.clone();
        assert!(!registry.publish(&old, RehydrateCompletion::Succeeded { replacement: evicted.native }),
            "the evicted native generation was reported as its own replacement");
        assert!(success.complete(RehydrateCompletion::BuildFailed));
        assert!(!registry.publish(&old, RehydrateCompletion::BuildFailed), "a duplicate completion mutated state");
        assert!(registry.retire(&old));
        let (_, racing) = registry.reserve(&evicted).unwrap().accepted(RehydratePriority::Background).unwrap();
        let current = racing.token.clone();
        assert_ne!(old.attempt_generation, current.attempt_generation);
        assert!(!registry.publish(&old, RehydrateCompletion::Panicked), "a stale attempt mutated its replacement");
        let replacement_build = ledger_build();
        let mut reused = current.clone();
        reused.evicted.build = Arc::downgrade(&replacement_build);
        assert!(!registry.publish(&reused, RehydrateCompletion::BuildFailed),
            "a different build allocation completed the current attempt");
        assert!(registry.publish(&current, RehydrateCompletion::Released));
        assert!(!racing.complete(RehydrateCompletion::Succeeded { replacement: ledger_native(25) }),
            "success won after release completed the attempt");
        assert_eq!(registry.records(1)[0].state, RehydrateCompletion::Released);
    }

    #[test]
    fn rehydrate_success_requires_exact_current_build_generation_and_liveness() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let job = build.job.lock().unwrap().take().unwrap(); complete(&build, job());
        let replacement = recipe.pin_with_residency(raw).0.unwrap(); let evicted = receipt_evicted(&build, 27);
        let registry = RehydrateReceiptRegistry::<1>::new().unwrap();
        let (_, success) = registry.reserve(&evicted).unwrap().accepted(RehydratePriority::Background).unwrap();
        assert!(success.replacement(&ledger_build(), &replacement).is_none(), "another build proved replacement");
        assert!(!registry.publish(&success.token, RehydrateCompletion::Succeeded { replacement: ledger_native(99) }),
            "a forged native identity bypassed replacement proof");
        let mut foreign = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) }; let foreign_raw = &mut foreign as *mut RebuildFakePso as usize;
        let foreign_recipe = Recipe::new(move || foreign_raw);
        let foreign_build = Arc::new(Build::new_keyed(foreign_recipe.initial_job(), false, None, Some(foreign_recipe.clone())));
        let foreign_job = foreign_build.job.lock().unwrap().take().unwrap(); complete(&foreign_build, foreign_job());
        let foreign_pin = foreign_recipe.pin_with_residency(foreign_raw).0.unwrap();
        assert!(rehydrate_success::Proof::published(&success.token, &build, &recipe, &foreign_pin).is_none(),
            "same-number generation from another recipe proved success");
        let proof = success.replacement(&build, &replacement).unwrap(); let token = success.token.clone();
        let mut newer_native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let newer_raw = &mut newer_native as *mut RebuildFakePso as usize;
        let (newer, replaced) = recipe.publish(newer_raw); let old_raw = build.real.swap(newer_raw, Ordering::AcqRel);
        release(old_raw); drop(replaced);
        assert!(rehydrate_success::Proof::published(&success.token, &build, &recipe, &replacement).is_none(),
            "an older generation was relabeled as the published replacement");
        assert!(!success.complete_success(proof), "generation A completed after B became current");
        assert_eq!(registry.records(1)[0].state, RehydrateCompletion::Cancelled); assert!(registry.retire(&token));
        let (_, success) = registry.reserve(&evicted).unwrap().accepted(RehydratePriority::Background).unwrap();
        let proof = success.replacement(&build, &newer).unwrap(); let token = success.token.clone();
        assert!(success.complete_success(proof));
        assert_eq!(registry.records(1)[0].state,
            RehydrateCompletion::Succeeded { replacement: newer.ledger_identity() });
        assert!(registry.retire(&token));
        let (_, expired) = registry.reserve(&evicted).unwrap().accepted(RehydratePriority::Background).unwrap();
        let proof = expired.replacement(&build, &newer).unwrap();
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(build); release(raw_owner);
        assert!(expired.complete_success(proof));
        assert_eq!(registry.records(1)[0].state, RehydrateCompletion::Released);
        drop(replacement); drop(newer); drop(recipe); assert_eq!(native.refs.load(Ordering::Relaxed), 0);
        assert_eq!(newer_native.refs.load(Ordering::Relaxed), 0);
        let foreign_owner = foreign_build.real.swap(0, Ordering::AcqRel); drop(foreign_pin); drop(foreign_build);
        release(foreign_owner); drop(foreign_recipe);
        assert_eq!(foreign.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn rehydrate_guard_drop_resolves_expired_build_as_released() {
        let build = ledger_build(); let evicted = receipt_evicted(&build, 28);
        let registry = RehydrateReceiptRegistry::<1>::new().unwrap();
        let (_, queued) = registry.reserve(&evicted).unwrap().accepted(RehydratePriority::Background).unwrap();
        drop(build);
        drop(queued);
        assert_eq!(registry.records(1)[0].state, RehydrateCompletion::Released);
    }

    #[test]
    fn rehydrate_receipt_capacity_and_counters_fail_before_queue() {
        let build = ledger_build(); let evicted = receipt_evicted(&build, 26);
        let registry = RehydrateReceiptRegistry::<1>::new().unwrap();
        let reservation = registry.reserve(&evicted).unwrap();
        assert!(registry.reserve(&evicted).is_none());
        drop(reservation); assert!(registry.records(1).is_empty());
        registry.inner.lock().unwrap().serial = u64::MAX;
        assert!(registry.reserve(&evicted).is_none());
        assert!(registry.inner.lock().unwrap().exhausted);
        let generation = RehydrateReceiptRegistry::<1>::new().unwrap();
        generation.inner.lock().unwrap().slots[0].generation = u64::MAX;
        assert!(generation.reserve(&evicted).is_none());
        assert!(generation.inner.lock().unwrap().exhausted);
    }

    #[test]
    fn restoration_coordinator_refusal_and_full_ledger_retry() {
        let coordinator = RestorationCoordinator::<1, 2>::new(301).unwrap();
        let build = ledger_build();
        let reserved = coordinator.reserve_eviction(&build, ledger_native(40)).unwrap();
        assert!(coordinator.reserve_eviction(&ledger_build(), ledger_native(41)).is_none());
        drop(reserved);
        let evicted = coordinated_evicted(&coordinator, &build, 40);
        assert!(coordinator.refuse_rehydrate(&evicted));
        assert_eq!(coordinator.reconcile(2), 1);
        assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::Evicted));
        assert!(coordinator.retries(1)[0].token.exact(&evicted));
    }

    #[test]
    fn installed_eviction_is_visible_before_detach_commit() {
        let coordinator = RestorationCoordinator::<1, 1>::new(308).unwrap();
        let build = ledger_build();
        let installed = coordinator.install_eviction(&build, ledger_native(50)).unwrap();
        let visible = build.eviction_token().expect("armed eviction was not installed");
        assert!(visible.exact(installed.token()));
        assert_eq!(coordinator.evicted.state(&visible), Some(EvictedBuildState::Evicted));

        let committed = installed.commit();
        assert!(build.eviction_token().is_some_and(|token| token.exact(&committed)));
        assert!(build.clear_eviction_token(&committed));
        assert!(coordinator.evicted.finish(&committed, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&committed));
    }

    #[test]
    fn failed_eviction_revalidation_clears_and_retires_installation() {
        let coordinator = RestorationCoordinator::<1, 1>::new(309).unwrap();
        let build = ledger_build();
        {
            let installed = coordinator.install_eviction(&build, ledger_native(51)).unwrap();
            assert!(build.eviction_token().is_some_and(|token| token.exact(installed.token())));
            // A future eviction path drops the guard when exact generation revalidation fails.
        }
        assert!(build.eviction_token().is_none());
        assert!(coordinator.evicted.records(1).is_empty());
    }

    #[test]
    fn stale_eviction_token_cannot_clear_reused_installation() {
        let coordinator = RestorationCoordinator::<1, 1>::new(310).unwrap();
        let build = ledger_build();
        let first = coordinator.install_eviction(&build, ledger_native(52)).unwrap().commit();
        assert!(build.clear_eviction_token(&first));
        assert!(coordinator.evicted.finish(&first, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&first));

        let second = coordinator.install_eviction(&build, ledger_native(53)).unwrap().commit();
        assert_eq!(first.slot, second.slot);
        assert_ne!(first.slot_generation, second.slot_generation);
        assert!(!build.clear_eviction_token(&first));
        assert!(build.eviction_token().is_some_and(|token| token.exact(&second)));
        assert!(build.clear_eviction_token(&second));
        assert!(coordinator.evicted.finish(&second, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&second));
    }

    #[test]
    fn rehydrate_registry_atomically_returns_existing_before_readiness() {
        let build = ledger_build(); let evicted = receipt_evicted(&build, 49);
        let registry = RehydrateReceiptRegistry::<2>::new().unwrap();
        let reservation = match registry.reserve_unique(&evicted, true) {
            Ok(Some(reservation)) => reservation, _ => panic!("initial unique reservation failed"),
        };
        let existing = match registry.reserve_unique(&evicted, true) {
            Err(existing) => existing, _ => panic!("duplicate reservation was admitted"),
        };
        assert!(existing.token.evicted.exact(&evicted));
        assert!(matches!(registry.reserve_unique(&evicted, false), Err(existing)
            if existing.token.evicted.exact(&evicted)));
        drop(reservation);
        assert!(matches!(registry.reserve_unique(&evicted, false), Ok(None)));
    }

    #[test]
    fn restoration_coordinator_maps_failures_and_foreground_deduplicates() {
        let coordinator = RestorationCoordinator::<1, 2>::new(302).unwrap();
        let build = ledger_build(); let evicted = coordinated_evicted(&coordinator, &build, 42);
        let queued = match coordinator.queue_rehydrate(&evicted, RehydratePriority::Background, true).unwrap() {
            RestorationQueue::Queued(queued) => queued, _ => panic!("first request was not queued"),
        };
        assert!(matches!(coordinator.queue_rehydrate(&evicted, RehydratePriority::Background, false),
            Some(RestorationQueue::Existing)), "backoff hid an already-live attempt");
        assert!(matches!(coordinator.queue_rehydrate(&evicted, RehydratePriority::Foreground, false),
            Some(RestorationQueue::Promoted)));
        assert!(matches!(coordinator.queue_rehydrate(&evicted, RehydratePriority::Foreground, false),
            Some(RestorationQueue::Existing)));
        assert_eq!(coordinator.receipts.records(2).len(), 1);
        assert_eq!(coordinator.receipts.records(2)[0].priority, RehydratePriority::Foreground);
        drop(queued); assert_eq!(coordinator.reconcile(2), 1);
        assert!(coordinator.queue_rehydrate(&evicted, RehydratePriority::Background, false).is_none());
        for completion in [RehydrateCompletion::Panicked, RehydrateCompletion::BuildFailed] {
            let queued = match coordinator.queue_rehydrate(&evicted, RehydratePriority::Background, true).unwrap() {
                RestorationQueue::Queued(queued) => queued, _ => unreachable!(),
            };
            assert!(queued.complete(completion)); assert_eq!(coordinator.reconcile(2), 1);
            assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::Evicted));
        }
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _queued = match coordinator.queue_rehydrate(&evicted, RehydratePriority::Foreground, false).unwrap() {
                RestorationQueue::Queued(queued) => queued, _ => unreachable!(),
            };
            panic!("restoration worker panic");
        }));
        assert!(panic_result.is_err()); assert_eq!(coordinator.reconcile(2), 1);
        assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::Evicted));
    }

    #[test]
    fn restoration_coordinator_atomically_deduplicates_concurrent_queues() {
        let coordinator = RestorationCoordinator::<1, 2>::new(306).unwrap();
        let build = ledger_build(); let evicted = coordinated_evicted(&coordinator, &build, 47);
        let barrier = std::sync::Barrier::new(3);
        let queued = AtomicUsize::new(0);
        let existing = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    barrier.wait();
                    match coordinator.queue_rehydrate(&evicted, RehydratePriority::Background, true).unwrap() {
                        RestorationQueue::Queued(receipt) => {
                            queued.fetch_add(1, Ordering::Relaxed);
                            assert!(receipt.complete(RehydrateCompletion::BuildFailed));
                        }
                        RestorationQueue::Existing => { existing.fetch_add(1, Ordering::Relaxed); }
                        RestorationQueue::Promoted => panic!("background request promoted"),
                    }
                });
            }
            barrier.wait();
        });
        assert_eq!((queued.load(Ordering::Relaxed), existing.load(Ordering::Relaxed)), (1, 1));
        assert_eq!(coordinator.receipts.records(2).len(), 1);
        assert_eq!(coordinator.reconcile(2), 1);
        assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::Evicted));
    }

    #[test]
    fn restoration_coordinator_atomically_orders_queue_against_refusal() {
        let coordinator = RestorationCoordinator::<1, 2>::new(307).unwrap();
        let build = ledger_build(); let evicted = coordinated_evicted(&coordinator, &build, 48);
        let barrier = std::sync::Barrier::new(3);
        let admitted = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                match coordinator.queue_rehydrate(&evicted, RehydratePriority::Background, true).unwrap() {
                    RestorationQueue::Queued(receipt) => {
                        admitted.fetch_add(1, Ordering::Relaxed);
                        assert!(receipt.complete(RehydrateCompletion::BuildFailed));
                    }
                    RestorationQueue::Existing => {}
                    RestorationQueue::Promoted => unreachable!(),
                }
            });
            scope.spawn(|| {
                barrier.wait();
                if coordinator.refuse_rehydrate(&evicted) {
                    admitted.fetch_add(1, Ordering::Relaxed);
                }
            });
            barrier.wait();
        });
        assert_eq!(admitted.load(Ordering::Relaxed), 1);
        assert_eq!(coordinator.receipts.records(2).len(), 1);
        assert_eq!(coordinator.reconcile(2), 1);
        assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::Evicted));
    }

    #[test]
    fn restoration_coordinator_release_race_is_terminal_and_idempotent() {
        let coordinator = RestorationCoordinator::<1, 1>::new(303).unwrap();
        let build = ledger_build(); let evicted = coordinated_evicted(&coordinator, &build, 43);
        let queued = match coordinator.queue_rehydrate(&evicted, RehydratePriority::Background, true).unwrap() {
            RestorationQueue::Queued(queued) => queued, _ => unreachable!(),
        };
        drop(build); drop(queued);
        assert_eq!(coordinator.reconcile(1), 1);
        assert_eq!(coordinator.evicted.state(&evicted), None);
        assert_eq!(coordinator.reconcile(1), 0);
        assert!(coordinator.retries(1).is_empty());
    }

    #[test]
    fn restoration_coordinator_release_clears_installation_and_reuses_capacity() {
        let coordinator = RestorationCoordinator::<1, 1>::new(303).unwrap();
        let build = ledger_build();
        let evicted = coordinator.install_eviction(&build, ledger_native(43)).unwrap().commit();
        let queued = match coordinator.queue_rehydrate(&evicted, RehydratePriority::Background, true).unwrap() {
            RestorationQueue::Queued(queued) => queued, _ => unreachable!(),
        };
        assert!(queued.complete(RehydrateCompletion::Released));
        assert_eq!(coordinator.reconcile(1), 1);
        assert!(build.eviction_token().is_none());
        assert_eq!(coordinator.evicted.state(&evicted), None);
        drop(coordinator.install_eviction(&build, ledger_native(44)).expect("terminal slot was not reusable"));
        assert_eq!(coordinator.reconcile(1), 0);
        assert!(coordinator.retries(1).is_empty());
    }

    #[test]
    fn restoration_coordinator_stale_terminal_cannot_clear_reused_installation() {
        let coordinator = RestorationCoordinator::<1, 1>::new(305).unwrap();
        let build = ledger_build();
        let first = coordinator.install_eviction(&build, ledger_native(45)).unwrap().commit();
        let queued = match coordinator.queue_rehydrate(&first, RehydratePriority::Background, true).unwrap() {
            RestorationQueue::Queued(queued) => queued, _ => unreachable!(),
        };
        assert!(queued.complete(RehydrateCompletion::Released));
        assert_eq!(coordinator.reconcile(1), 1);
        let second = coordinator.install_eviction(&build, ledger_native(46)).unwrap().commit();
        assert_eq!(first.slot, second.slot);
        assert_ne!(first.slot_generation, second.slot_generation);
        assert!(coordinator.receipts.reserve(&first).unwrap().refused().is_some());
        assert_eq!(coordinator.reconcile(1), 1);
        assert_eq!(coordinator.evicted.state(&second), Some(EvictedBuildState::Evicted),
            "a stale receipt poisoned a reused eviction slot");
        assert!(build.eviction_token().is_some_and(|token| token.exact(&second)),
            "a stale terminal receipt cleared the replacement installation");
        assert!(build.clear_eviction_token(&second));
        assert!(coordinator.evicted.finish(&second, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&second));
        assert_eq!(coordinator.reconcile(1), 0);
    }

    #[test]
    fn restoration_coordinator_exact_success_restores_after_receipt() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let job = build.job.lock().unwrap().take().unwrap(); complete(&build, job());
        let replacement = recipe.pin_with_residency(raw).0.unwrap();
        let coordinator = RestorationCoordinator::<1, 1>::new(304).unwrap();
        let evicted = coordinator.install_eviction(&build, ledger_native(44)).unwrap().commit();
        let queued = match coordinator.queue_rehydrate(&evicted, RehydratePriority::Foreground, false).unwrap() {
            RestorationQueue::Queued(queued) => queued, _ => unreachable!(),
        };
        let proof = queued.replacement(&build, &replacement).unwrap();
        assert!(queued.complete_success(proof));
        assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::RehydrateQueued));
        assert_eq!(coordinator.reconcile(1), 1);
        assert!(build.eviction_token().is_none());
        assert_eq!(coordinator.evicted.state(&evicted), None);
        drop(coordinator.install_eviction(&build, ledger_native(45)).expect("restored slot was not reusable"));
        assert_eq!(coordinator.reconcile(1), 0);
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(build); release(raw_owner); drop(replacement); drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn exact_restoration_promotes_without_inline_work_and_transfers_its_pin() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let replacement_vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_reenter_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: replacement_vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let calls = Arc::new(AtomicUsize::new(0));
        let build_next = next.clone(); let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            build_next.load(Ordering::Acquire)
        });
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        let first_pin = recipe.pin_with_residency(first_raw).0.unwrap();
        let pressure_stats = Arc::new(PipelinePressureCounters::default());
        let coordinator = RestorationCoordinator::<2, 2>::new_with_pressure_stats(
            401, Some(pressure_stats.clone()),
        ).unwrap();
        let evicted = coordinator.reserve_eviction(&build, first_pin.ledger_identity()).unwrap().arm_detach().unwrap();
        drop(first_pin);
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        next.store(replacement_raw, Ordering::Release);
        let (captured_tx, captured_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        *coordinator.captured.lock().unwrap() = Some(Box::new(move || {
            captured_tx.send(()).unwrap(); continue_rx.recv().unwrap();
        }));

        let q = BuildQueue::new();
        let background = coordinator.request_exact_on(&evicted, RehydratePriority::Background, &q).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1, "request ran driver work inline");
        let foreground = coordinator.request_exact_on(&evicted, RehydratePriority::Foreground, &q).unwrap();
        assert!(background.token().exact(foreground.token()), "foreground demand did not coalesce exactly");
        assert!(!q.try_run_one_between(Priority::Speculative, Priority::Preload));
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| q.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
            captured_rx.recv().unwrap();
            assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 2 });
            continue_tx.send(()).unwrap();
            assert!(worker.join().unwrap());
        });
        let pin = foreground.wait_and_take_pin().unwrap();
        assert_eq!((pin.raw(), pin.id(), calls.load(Ordering::Relaxed)), (replacement_raw, 2, 2));
        assert_eq!(pin.raw(), replacement_raw, "a second detach invalidated the handed-off generation");
        drop(background.wait_and_take_pin().unwrap());

        drop(background); drop(foreground); drop(pin);
        RESTORE_REENTER_ATTEMPTS.with(|address| address.set(&coordinator.attempts as *const _ as usize));
        assert_eq!(coordinator.reconcile(2), 1);
        assert_eq!(pressure_stats.snapshot(), PipelinePressureStats {
            restoration_attempts: 1,
            restoration_succeeded: 1,
            ..Default::default()
        }, "foreground promotion counted a duplicate restoration attempt or outcome");
        RESTORE_REENTER_ATTEMPTS.with(|address| address.set(0));
        drop(coordinator); drop(build); drop(recipe);
        assert_eq!((first.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn exact_restoration_tickets_report_every_terminal_and_do_not_reuse_attempts() {
        let recipe = Recipe::new(|| 0);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe)));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        let pressure_stats = Arc::new(PipelinePressureCounters::default());
        let coordinator = RestorationCoordinator::<1, 1>::new_with_pressure_stats(
            402, Some(pressure_stats.clone()),
        ).unwrap();
        let evicted = coordinated_evicted(&coordinator, &build, 60);

        let refused = BuildQueue::new(); refused.shutdown();
        let refused = coordinator.request_exact_on(&evicted, RehydratePriority::Foreground, &refused).unwrap();
        assert!(matches!(refused.wait_and_take_pin(), Err(RehydrateCompletion::Refused))); assert_eq!(coordinator.reconcile(1), 1);
        let cancelled = BuildQueue::new();
        let first = coordinator.request_exact_on(&evicted, RehydratePriority::Background, &cancelled).unwrap();
        let first_token = first.token().clone(); cancelled.shutdown();
        assert!(matches!(first.wait_and_take_pin(), Err(RehydrateCompletion::Cancelled))); assert_eq!(coordinator.reconcile(1), 1);
        let failed = BuildQueue::new();
        let second = coordinator.request_exact_on(&evicted, RehydratePriority::Foreground, &failed).unwrap();
        assert!(!first_token.exact(second.token())); assert!(failed.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        assert!(matches!(second.wait_and_take_pin(), Err(RehydrateCompletion::BuildFailed))); assert_eq!(coordinator.reconcile(1), 1);
        *coordinator.before_rebuild.lock().unwrap() = Some(Box::new(|| panic!("restoration panic")));
        let panicked = BuildQueue::new();
        let panic_ticket = coordinator.request_exact_on(&evicted, RehydratePriority::Foreground, &panicked).unwrap();
        assert!(panicked.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        assert_eq!(coordinator.reconcile(1), 1, "an unwaited panic was not reconciled");
        assert!(matches!(panic_ticket.wait_and_take_pin(), Err(RehydrateCompletion::Panicked)));
        let retry_q = BuildQueue::new();
        let retry = coordinator.request_exact_on(&evicted, RehydratePriority::Foreground, &retry_q).unwrap();
        assert!(retry_q.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        assert!(matches!(retry.wait_and_take_pin(), Err(RehydrateCompletion::BuildFailed))); assert_eq!(coordinator.reconcile(1), 1);
        let released_q = BuildQueue::new();
        let released = coordinator.request_exact_on(&evicted, RehydratePriority::Foreground, &released_q).unwrap();
        build.owners.store(0, Ordering::Release);
        assert!(released_q.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        assert!(matches!(released.wait_and_take_pin(), Err(RehydrateCompletion::Released))); assert_eq!(coordinator.reconcile(1), 1);
        assert_eq!(pressure_stats.snapshot(), PipelinePressureStats {
            restoration_attempts: 6,
            restoration_failed: 6,
            ..Default::default()
        }, "terminal restoration attempts were missed or counted more than once");
    }

    fn structured_pending(build: Arc<Build>, compute: bool) -> Arc<Pending> {
        Arc::new(Pending {
            build,
            driver: AtomicUsize::new(0),
            compute,
            key: 0,
            manifest_recipe: AtomicU64::new(0),
            track_used: true,
            used: AtomicBool::new(false),
            late: AtomicBool::new(false),
            spared: AtomicBool::new(false),
            on_use: Mutex::new(None),
        })
    }

    #[test]
    fn dormant_restoration_coordinator_has_one_stable_identity() {
        let first = pipeline_restoration_coordinator() as *const _;
        let second = pipeline_restoration_coordinator() as *const _;
        assert_eq!(first, second);
        assert!(PIPELINE_RESTORATION_EVICTIONS > 0 && PIPELINE_RESOLUTION_RECEIPTS > 0);

        let original = 0x7E57_D010usize;
        let pending = structured_pending(Arc::new(Build::new(Box::new(|| {
            panic!("dormant classification ran initial driver work inline")
        }), false)), false);
        PENDING.insert(original, pending.clone());
        let resolution = begin_pipeline_resolution(original).expect("registered pipeline");
        assert_eq!(resolution.identity_for_test(), (original, Arc::as_ptr(&pending) as usize));
        assert!(Arc::ptr_eq(&PENDING.remove(original).unwrap(), &pending));
    }

    #[test]
    fn emission_resolves_the_retained_recipe_after_title_address_reuse() {
        let original = 0x7E57_D013usize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let old = structured_pending(build.clone(), false);
        let resolution = PipelineResolution(classify_retained(original, &old), old.compute, unavailable_set(&old));
        let replacement = structured_pending(
            Arc::new(Build::new(Box::new(|| panic!("replacement build ran")), false)), false,
        );
        PENDING.insert(original, replacement.clone());

        let PipelineChoice::Available { raw: selected, generation } =
            resolution.resolve_for_emission() else { panic!("retained recipe was unavailable") };
        assert_eq!((selected, generation.id()), (raw, 1));
        assert!(!*replacement.build.done.lock().unwrap(), "replacement build was consumed");
        assert!(replacement.build.job.lock().unwrap().is_some(), "replacement job was taken");

        assert!(Arc::ptr_eq(&PENDING.remove(original).unwrap(), &replacement));
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(generation); drop(resolution); drop(old); drop(build); drop(recipe); release(raw_owner);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn completed_one_shot_resolution_owns_one_exact_generation_after_address_reuse() {
        let original = 0x7E57_D016usize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        submit_lazy(original, false, Box::new(move || raw), 0);
        assert_eq!(resolve_wait(original), raw);
        assert_eq!(native.refs.load(Ordering::Relaxed), 1, "completion or ordinary resolution eagerly created an exact owner");
        let pending = entry(original).unwrap();

        let resolution = begin_pipeline_resolution(original).expect("completed one-shot pipeline");
        assert_eq!(native.refs.load(Ordering::Relaxed), 2, "first exact selection owns one COM reference");
        let PipelineChoice::Available { raw: first_raw, generation: first } = resolution.resolve_for_emission()
            else { panic!("completed one-shot was unavailable") };
        let PipelineChoice::Available { raw: second_raw, generation: second } = resolution.resolve_for_emission()
            else { panic!("repeated one-shot selection was unavailable") };
        assert_eq!((first_raw, second_raw), (raw, raw));
        assert!(first.same(&second), "repeated selection allocated another generation identity");
        assert_eq!(native.refs.load(Ordering::Relaxed), 2, "repeated selection changed the steady COM refcount");

        let raw_owner = released(original);
        assert_eq!(raw_owner, raw);
        assert!(pending.build.one_shot_generation.lock().unwrap().is_none(), "last title owner kept the build cache");
        submit_lazy(original, false, Box::new(|| panic!("replacement address was resolved")), 0);
        let PipelineChoice::Available { raw: retained_raw, generation: retained } = resolution.resolve_for_emission()
            else { panic!("retained one-shot generation was lost") };
        assert_eq!(retained_raw, raw);
        assert!(first.same(&retained));

        assert_eq!(released(original), 0);
        release(raw_owner);
        drop(retained); drop(second); drop(first); drop(resolution); drop(pending);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn initial_resolution_promotes_one_exact_generation_across_release_and_reuse() {
        let original = 0x7E57_D019usize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let build = Arc::new(Build::new(Box::new(move || raw), false));
        let pending = structured_pending(build.clone(), false);
        assert!(PENDING.insert(original, pending.clone()).is_none());
        let resolution = begin_pipeline_resolution(original).expect("unfinished pipeline");
        let indirect_clone = resolution.clone();
        assert!(resolution.submit_demand_pending());

        let job = build.job.lock().unwrap().take().unwrap();
        complete(&build, job());
        let IndirectPipelineChoice::Exact { generation: indirect_first, .. } = resolution.resolve_for_indirect()
            else { panic!("completed initial indirect pipeline was unavailable") };
        let IndirectPipelineChoice::Exact { generation: indirect_second, .. } = indirect_clone.resolve_for_indirect()
            else { panic!("cloned indirect resolution lost its promoted winner") };
        assert!(indirect_first.same(&indirect_second));
        let PipelineChoice::Available { generation: first, .. } = resolution.resolve_for_emission()
            else { panic!("completed initial pipeline was unavailable") };
        let PipelineChoice::Available { generation: second, .. } = resolution.resolve_for_emission()
            else { panic!("promoted initial pipeline was unavailable") };
        assert!(first.same(&second));
        assert!(!resolution.submit_demand_pending());
        let (reset_raw, reset) = resolution.ready_for_reset().expect("promoted reset generation");
        assert_eq!(reset_raw, raw);
        assert!(first.same(&reset));
        let nonblocking = resolution.0.nonblocking();
        assert_eq!((nonblocking.raw, nonblocking.bound_unbuilt), (raw, false));
        assert!(first.same(nonblocking.pin.as_ref().unwrap()));
        let exact = resolution.0.wait_exact().unwrap().expect("promoted exact choice");
        assert_eq!((exact.raw, exact.bound_unbuilt), (raw, false));
        assert!(first.same(exact.pin.as_ref().unwrap()));

        let raw_owner = released(original);
        assert_eq!(raw_owner, raw);
        let replacement = structured_pending(Arc::new(Build::new(Box::new(|| 0), false)), false);
        assert!(PENDING.insert(original, replacement.clone()).is_none());
        let PipelineChoice::Available { raw: retained_raw, generation: retained } = resolution.resolve_for_emission()
            else { panic!("release lost the promoted exact generation") };
        assert_eq!(retained_raw, raw);
        assert!(first.same(&retained));

        assert!(Arc::ptr_eq(&PENDING.remove(original).unwrap(), &replacement));
        release(raw_owner);
        drop(exact); drop(nonblocking); drop(retained); drop(reset); drop(second); drop(first);
        drop(indirect_second); drop(indirect_first); drop(indirect_clone); drop(resolution); drop(pending); drop(build);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn retained_native_lazy_choice_ignores_address_reuse_and_shares_promotion() {
        let original = 0x7E57_D030usize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let build = Arc::new(Build::new(Box::new(move || raw), false));
        let pending = structured_pending(build.clone(), false);
        assert!(PENDING.insert(original, pending.clone()).is_none());
        let resolution = begin_pipeline_resolution(original).unwrap();
        let clone = resolution.clone();
        let replacement = structured_pending(Arc::new(Build::new(Box::new(|| {
            panic!("replacement address was consulted")
        }), false)), false);
        assert!(Arc::ptr_eq(&PENDING.insert(original, replacement.clone()).unwrap(), &pending));

        let job = build.job.lock().unwrap().take().unwrap();
        build.sent.store(true, Ordering::Release); // the test owns `job`; suppress an empty queue entry
        assert!(matches!(
            resolution.select_for_native_bind(NativePipelineBindMode::Lazy),
            NativePipelineChoice::InitialFallback { raw, bound_unbuilt: true } if raw == original
        ));
        assert!(replacement.build.job.lock().unwrap().is_some());
        complete(&build, job());
        let ready = PipelineResolution(classify_retained(original, &pending), false, UnavailableSet::Poison);
        let NativePipelineChoice::Exact { raw: ready_raw, generation: ready_generation } =
            ready.select_for_native_bind(NativePipelineBindMode::ForBind)
            else { panic!("ready retained native choice was not exact") };
        let NativePipelineChoice::Exact { raw: first_raw, generation: first } =
            clone.select_for_native_bind(NativePipelineBindMode::Lazy)
            else { panic!("completed retained native choice was not exact") };
        let NativePipelineChoice::Exact { raw: second_raw, generation: second } =
            resolution.select_for_native_bind(NativePipelineBindMode::Lazy)
            else { panic!("promoted native winner was lost") };
        assert_eq!((ready_raw, first_raw, second_raw), (raw, raw, raw));
        assert!(ready_generation.same(&first));
        assert!(first.same(&second));
        assert_eq!(native.refs.load(Ordering::Relaxed), 2, "promotion allocated extra native owners");

        assert!(Arc::ptr_eq(&PENDING.remove(original).unwrap(), &replacement));
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        release(raw_owner);
        drop(second); drop(first); drop(ready_generation); drop(ready);
        drop(clone); drop(resolution); drop(pending); drop(build);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn retained_native_for_bind_waits_only_for_compute_initial() {
        fn pending_build(compute: bool) -> (Arc<Build>, Arc<Pending>, Arc<AtomicBool>) {
            let finish = Arc::new(AtomicBool::new(false));
            let build_finish = finish.clone();
            let build = Arc::new(Build::new(Box::new(move || {
                while !build_finish.load(Ordering::Acquire) { std::thread::yield_now(); }
                0
            }), false));
            (build.clone(), structured_pending(build, compute), finish)
        }
        let (compute_build, compute_pending, compute_finish) = pending_build(true);
        let compute = PipelineResolution(
            StructuredResolve::initial(0x7E57_D031, compute_pending, 0x7E57_D031, true), true, UnavailableSet::Omit,
        );
        assert!(matches!(
            compute.select_for_native_bind_with_budget(
                NativePipelineBindMode::ForBind, Some(std::time::Duration::ZERO),
            ),
            NativePipelineChoice::InitialFallback { raw: 0x7E57_D031, bound_unbuilt: true }
        ));
        compute_finish.store(true, Ordering::Release);
        let mut done = compute_build.done.lock().unwrap();
        while !*done { done = compute_build.cv.wait(done).unwrap(); }
        drop(done);

        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let (release_build, wait_for_release) = mpsc::channel();
        let exact_build = Arc::new(Build::new(Box::new(move || {
            wait_for_release.recv().unwrap();
            raw
        }), false));
        let exact = PipelineResolution(StructuredResolve::initial(
            0x7E57_D035, structured_pending(exact_build.clone(), true), 0x7E57_D035, true,
        ), true, UnavailableSet::Omit);
        let (result_tx, result_rx) = mpsc::channel();
        let waiter = exact.clone();
        let thread = std::thread::spawn(move || {
            result_tx.send(waiter.select_for_native_bind_with_budget(
                NativePipelineBindMode::ForBind, None,
            )).unwrap();
        });
        assert!(matches!(
            result_rx.recv_timeout(std::time::Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ), "compute ForBind returned before its retained build completed");
        release_build.send(()).unwrap();
        let NativePipelineChoice::Exact { raw: selected, generation } =
            result_rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap()
            else { panic!("compute ForBind did not return its completed exact generation") };
        thread.join().unwrap();
        assert_eq!(selected, raw);
        let raw_owner = exact_build.real.swap(0, Ordering::AcqRel);
        release(raw_owner);
        drop(generation); drop(exact); drop(exact_build);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);

        let (graphics_build, graphics_pending, graphics_finish) = pending_build(false);
        let graphics = PipelineResolution(
            StructuredResolve::initial(0x7E57_D032, graphics_pending, 0x7E57_D032, true), false, UnavailableSet::Poison,
        );
        assert!(matches!(
            graphics.select_for_native_bind_with_budget(NativePipelineBindMode::ForBind, None),
            NativePipelineChoice::InitialFallback { raw: 0x7E57_D032, bound_unbuilt: true }
        ), "graphics ForBind waited instead of using lazy policy");
        graphics_finish.store(true, Ordering::Release);
        let mut done = graphics_build.done.lock().unwrap();
        while !*done { done = graphics_build.cv.wait(done).unwrap(); }
    }

    #[test]
    fn retained_native_deferred_is_fail_closed_then_observes_exact_publication() {
        let original = 0x7E57_D033usize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut restored = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let restored_raw = &mut restored as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || first_raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        let pending = structured_pending(build.clone(), true);
        let coordinator = RestorationCoordinator::<4, PIPELINE_RESOLUTION_RECEIPTS>::new(0xD033).unwrap();
        let pin = recipe.pin_with_residency(first_raw).0.unwrap();
        let evicted = coordinator.install_eviction(&build, pin.ledger_identity()).unwrap().commit();
        drop(pin);
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        let restoration = Arc::new(RestorationDemand::new(evicted));
        let resolution = PipelineResolution(StructuredResolve::Deferred {
            original, pending: pending.clone(), fallback: original,
            restoration: restoration.clone(),
        }, true, UnavailableSet::Omit);
        assert!(matches!(
            resolution.select_for_native_bind(NativePipelineBindMode::Lazy),
            NativePipelineChoice::Unavailable
        ));
        assert!(matches!(
            resolution.select_for_native_bind(NativePipelineBindMode::ForBind),
            NativePipelineChoice::Unavailable
        ));
        assert!(!restoration.demanded.load(Ordering::Acquire));
        assert!(restoration.requested.get().is_none(), "native selection requested restoration");

        let (published, replaced) = recipe.publish(restored_raw);
        assert!(replaced.is_none());
        build.real.store(restored_raw, Ordering::Release);
        build.owners.store(0, Ordering::Release);
        assert!(matches!(
            resolution.select_for_native_bind(NativePipelineBindMode::Lazy),
            NativePipelineChoice::Unavailable
        ), "released deferred identity resurrected a resident generation");
        build.owners.store(1, Ordering::Release);
        let NativePipelineChoice::Exact { raw, generation } =
            resolution.select_for_native_bind(NativePipelineBindMode::Lazy)
            else { panic!("published retained restoration was unavailable") };
        assert_eq!(raw, restored_raw);
        assert_eq!(generation.id(), published.id());
        let PipelineChoice::Available { generation: emission, .. } = resolution.resolve_for_emission()
            else { panic!("emission did not converge on resident deferred generation") };
        assert!(generation.same(&emission));
        let terminal = PipelineResolution(StructuredResolve::Terminal {
            original, pending: pending.clone(), fallback: original,
            failure: StructuredResolveFailure::RestorationUnavailable,
        }, true, UnavailableSet::Omit);
        assert!(matches!(
            terminal.select_for_native_bind(NativePipelineBindMode::Lazy),
            NativePipelineChoice::Unavailable
        ));
        assert!(matches!(terminal.resolve_for_emission(), PipelineChoice::Unavailable));

        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(terminal); drop(emission); drop(generation); drop(published); drop(resolution);
        drop(restoration); drop(pending); drop(build); drop(recipe);
        release(raw_owner);
        assert_eq!((first.refs.load(Ordering::Relaxed), restored.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn retained_native_unpromoted_release_is_unavailable() {
        let build = Arc::new(Build::new(Box::new(|| panic!("released build ran")), false));
        let pending = structured_pending(build.clone(), true);
        let resolution = PipelineResolution(
            StructuredResolve::initial(0x7E57_D034, pending, 0x7E57_D034, true), true, UnavailableSet::Omit,
        );
        build.owners.store(0, Ordering::Release);
        for mode in [NativePipelineBindMode::Lazy, NativePipelineBindMode::ForBind] {
            assert!(matches!(resolution.select_for_native_bind(mode), NativePipelineChoice::Unavailable));
        }
    }

    #[test]
    fn initial_pending_probe_does_not_take_the_completion_lock() {
        let build = Arc::new(Build::new(Box::new(|| 0), false));
        let pending = structured_pending(build.clone(), true);
        let resolution = PipelineResolution(StructuredResolve::initial(0x7E57_D01B, pending, 0x7E57_D01B, true), true, UnavailableSet::Omit);
        let done = build.done.lock().unwrap();
        let (tx, rx) = mpsc::channel();
        let thread = std::thread::spawn(move || tx.send(resolution.submit_demand_pending()).unwrap());
        assert!(rx.recv_timeout(std::time::Duration::from_secs(10)).expect("completion probe took the done lock"));
        drop(done);
        thread.join().unwrap();
    }

    #[test]
    fn indirect_resolution_preserves_compute_kind_and_bounded_initial_fallback() {
        assert!(pipeline_resolution_for_compute_test(0x7E57_D01A, || {}).is_compute());
        let original = 0x7E57_D01Cusize;
        let finish = Arc::new(AtomicBool::new(false));
        let build_finish = finish.clone();
        let build = Arc::new(Build::new(Box::new(move || {
            while !build_finish.load(Ordering::Acquire) { std::thread::yield_now(); }
            0
        }), false));
        let pending = structured_pending(build.clone(), true);
        PENDING.insert(original, pending.clone());
        let resolution = begin_pipeline_resolution(original).unwrap();
        let replacement = structured_pending(
            Arc::new(Build::new(Box::new(|| panic!("replacement build ran")), false)), false,
        );
        PENDING.insert(original, replacement.clone());
        assert!(resolution.is_compute(), "address reuse changed the retained pipeline kind");
        assert!(resolution.clone().is_compute(), "a clone lost the retained pipeline kind");

        assert!(matches!(
            resolution.resolve_for_indirect_with_budget(Some(std::time::Duration::ZERO)),
            IndirectPipelineChoice::InitialFallback { raw, bound_unbuilt: true } if raw == original
        ));
        assert!(!*replacement.build.done.lock().unwrap());
        assert!(replacement.build.job.lock().unwrap().is_some(), "fallback consulted the reused address");

        finish.store(true, Ordering::Release);
        let mut done = build.done.lock().unwrap();
        while !*done { done = build.cv.wait(done).unwrap(); }
        drop(done);
        assert!(matches!(
            resolution.resolve_for_indirect(),
            IndirectPipelineChoice::InitialFallback { raw, bound_unbuilt: false } if raw == original
        ), "a completed initial build failure did not preserve its ordinary stand-in result");
        assert!(Arc::ptr_eq(&PENDING.remove(original).unwrap(), &replacement));
    }

    #[test]
    fn indirect_resolution_rejects_released_and_stale_initial_fallbacks() {
        let released_build = Arc::new(Build::new(Box::new(|| panic!("released build ran")), false));
        let released_pending = structured_pending(released_build.clone(), true);
        let released = PipelineResolution(
            classify_retained(0x7E57_D01D, &released_pending), released_pending.compute, UnavailableSet::Poison,
        );
        released_build.owners.store(0, Ordering::Release);
        assert!(matches!(released.resolve_for_indirect(), IndirectPipelineChoice::Unavailable));
        released_build.owners.store(1, Ordering::Release);
        released_build.orphan.store(true, Ordering::Release);
        assert!(matches!(released.resolve_for_indirect(), IndirectPipelineChoice::Unavailable));

        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let pending = structured_pending(build.clone(), true);
        let stale = PipelineResolution(classify_retained(0x7E57_D01E, &pending), pending.compute, UnavailableSet::Poison);
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        let ready = PipelineResolution(classify_retained(0x7E57_D01F, &pending), pending.compute, UnavailableSet::Poison);
        assert!(ready.is_compute() && ready.clone().is_compute(), "a ready snapshot lost compute kind");
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        assert!(matches!(stale.resolve_for_indirect(), IndirectPipelineChoice::Unavailable),
            "an evicted initial generation escaped as its stand-in fallback");
        drop(ready); drop(stale); drop(pending); drop(build); drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn indirect_resolution_revalidates_release_after_wait() {
        let original = 0x7E57_D022usize;
        let started = Arc::new(AtomicBool::new(false));
        let finish = Arc::new(AtomicBool::new(false));
        let (build_started, build_finish) = (started.clone(), finish.clone());
        let build = Arc::new(Build::new(Box::new(move || {
            build_started.store(true, Ordering::Release);
            while !build_finish.load(Ordering::Acquire) { std::thread::yield_now(); }
            0
        }), false));
        let pending = structured_pending(build.clone(), true);
        assert!(PENDING.insert(original, pending.clone()).is_none());
        let resolution = begin_pipeline_resolution(original).unwrap();
        let waiter = std::thread::spawn(move || resolution.resolve_for_indirect_with_budget(None));
        while !started.load(Ordering::Acquire) { std::thread::yield_now(); }
        assert_eq!(released(original), 0);
        assert_eq!(build.owners.load(Ordering::Acquire), 0);
        assert!(build.orphan.load(Ordering::Acquire));
        finish.store(true, Ordering::Release);
        assert!(matches!(waiter.join().unwrap(), IndirectPipelineChoice::Unavailable),
            "release during the wait escaped as an unpinned initial fallback");
    }

    #[test]
    fn terminal_paths_release_publish_completion_after_their_result() {
        fn observe_after(build: Arc<Build>, publish: impl FnOnce()) -> usize {
            let gate = Arc::new(std::sync::Barrier::new(2));
            let observer_gate = gate.clone();
            let observer = build.clone();
            let (tx, rx) = mpsc::channel();
            let thread = std::thread::spawn(move || {
                observer_gate.wait();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while !observer.completed.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
                    std::thread::yield_now();
                }
                tx.send(observer.completed.load(Ordering::Acquire).then(|| observer.real.load(Ordering::Acquire))).unwrap();
            });
            gate.wait();
            publish();
            let observed = rx.recv_timeout(std::time::Duration::from_secs(10))
                .expect("completion observer did not finish")
                .expect("terminal state was not release-published");
            thread.join().unwrap();
            observed
        }

        let success = Arc::new(Build::new(Box::new(|| 0xC011_E7ED), false));
        let success_job = success.job.lock().unwrap().take().unwrap();
        let observed = observe_after(success.clone(), || complete(&success, success_job()));
        assert_eq!(observed, 0xC011_E7ED, "completion became visible before its result");
        success.real.store(0, Ordering::Release);

        let failed = Arc::new(Build::new(Box::new(|| 0), false));
        drop(failed.job.lock().unwrap().take());
        let observed = observe_after(failed.clone(), || complete(&failed, 0));
        assert_eq!(observed, 0, "failed completion published a native result");
        assert!(*failed.done.lock().unwrap());

        let unstarted = Arc::new(Build::new(Box::new(|| 0), false));
        unstarted.owners.store(0, Ordering::Release);
        unstarted.orphan.store(true, Ordering::Release);
        drop_unstarted_job(&unstarted);
        assert!(unstarted.completed.load(Ordering::Acquire));
        assert!(*unstarted.done.lock().unwrap());
        assert!(unstarted.job.lock().unwrap().is_none());

        let running = Arc::new(Build::new(Box::new(|| 0), false));
        let running_job = running.job.lock().unwrap().take().unwrap();
        running.owners.store(0, Ordering::Release);
        running.orphan.store(true, Ordering::Release);
        drop_unstarted_job(&running);
        assert!(!running.completed.load(Ordering::Acquire), "an orphaned running job completed before its worker");
        complete(&running, running_job());
        assert!(running.completed.load(Ordering::Acquire));
    }

    #[test]
    fn unavailable_initial_resolution_remains_retryable_until_exact_publication() {
        let original = 0x7E57_D01Ausize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), true, None, Some(recipe.clone())));
        drop(build.job.lock().unwrap().take());
        complete(&build, 0);
        let pending = structured_pending(build.clone(), false);
        let resolution = PipelineResolution(classify_retained(original, &pending), pending.compute, UnavailableSet::Poison);
        assert!(matches!(resolution.resolve_for_emission(), PipelineChoice::Unavailable));

        let (published, replaced) = recipe.publish(raw);
        assert!(replaced.is_none());
        build.real.store(raw, Ordering::Release);
        let PipelineChoice::Available { generation, .. } = resolution.resolve_for_emission()
            else { panic!("later exact publication was hidden by an unavailable attempt") };
        assert_eq!(generation.id(), published.id());
        let PipelineChoice::Available { generation: cached, .. } = resolution.resolve_for_emission()
            else { panic!("published generation was not promoted") };
        assert!(generation.same(&cached));

        drop(cached); drop(generation); drop(published); drop(resolution); drop(pending); drop(build); drop(recipe);
        release(raw);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn concurrent_initial_promotions_return_the_single_cell_winner() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first_native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut second_native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first_native as *mut RebuildFakePso as usize;
        let second_raw = &mut second_native as *mut RebuildFakePso as usize;
        let owner = |raw| unsafe {
            ID3D12PipelineState::from_raw((raw as *mut RebuildFakePso).cast::<std::ffi::c_void>())
        };
        let first = NativeGeneration {
            generation: Arc::new(Generation {
                id: 1, raw: first_raw, _owner: owner(first_raw), pressure_release: Default::default(),
            }),
            recipe: Weak::new(),
        };
        let second = NativeGeneration {
            generation: Arc::new(Generation {
                id: 2, raw: second_raw, _owner: owner(second_raw), pressure_release: Default::default(),
            }),
            recipe: Weak::new(),
        };
        let promoted = Arc::new(OnceLock::new());
        let indirect_promoted = Arc::new(OnceLock::new());
        let start = Arc::new(std::sync::Barrier::new(3));
        let choices = std::thread::scope(|scope| {
            let first_thread = {
                let (promoted, indirect_promoted, start) =
                    (promoted.clone(), indirect_promoted.clone(), start.clone());
                scope.spawn(move || {
                    start.wait();
                    let ordinary = promote_initial_choice(&promoted, PipelineChoice::Available {
                        raw: first_raw, generation: first.clone(),
                    });
                    (ordinary, promote_indirect_initial_choice(&indirect_promoted, first))
                })
            };
            let second_thread = {
                let (promoted, indirect_promoted, start) =
                    (promoted.clone(), indirect_promoted.clone(), start.clone());
                scope.spawn(move || {
                    start.wait();
                    let ordinary = promote_initial_choice(&promoted, PipelineChoice::Available {
                        raw: second_raw, generation: second.clone(),
                    });
                    (ordinary, promote_indirect_initial_choice(&indirect_promoted, second))
                })
            };
            start.wait();
            [first_thread.join().unwrap(), second_thread.join().unwrap()]
        });
        let [(PipelineChoice::Available { generation: first, .. }, IndirectPipelineChoice::Exact { generation: indirect_first, .. }),
            (PipelineChoice::Available { generation: second, .. }, IndirectPipelineChoice::Exact { generation: indirect_second, .. })] = choices
            else { panic!("a concurrent available choice became unavailable") };
        assert!(first.same(&second));
        assert!(first.same(promoted.get().unwrap()));
        assert!(indirect_first.same(&indirect_second));
        assert!(indirect_first.same(indirect_promoted.get().unwrap()));
        let pending = structured_pending(Arc::new(Build::new(Box::new(|| 0), false)), true);
        let IndirectPipelineChoice::Exact { generation: after_local_fallback, .. } =
            finish_indirect_initial_choice(
                &indirect_promoted, &pending,
                IndirectPipelineChoice::InitialFallback { raw: 0xBAD, bound_unbuilt: true },
            ) else { panic!("a local fallback hid the concurrently promoted winner") };
        assert!(after_local_fallback.same(indirect_promoted.get().unwrap()));
        let PipelineChoice::Available { generation: after_local_failure, .. } =
            promote_initial_choice(&promoted, PipelineChoice::Unavailable)
            else { panic!("a local failure hid the concurrently promoted winner") };
        assert!(first.same(&after_local_failure));
        drop(after_local_failure); drop(after_local_fallback); drop(pending);
        drop(indirect_second); drop(indirect_first);
        drop(second); drop(first); drop(indirect_promoted); drop(promoted);
        assert_eq!((first_native.refs.load(Ordering::Relaxed), second_native.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn coalesced_one_shot_owners_share_one_generation_until_final_release() {
        let first_address = 0x7E57_D017usize;
        let second_address = 0x7E57_D018usize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let build = Arc::new(Build::new(Box::new(move || raw), false));
        build.owners.store(2, Ordering::Release);
        PENDING.insert(first_address, structured_pending(build.clone(), false));
        PENDING.insert(second_address, structured_pending(build.clone(), false));
        assert_eq!(resolve_wait(first_address), raw);

        let first_resolution = begin_pipeline_resolution(first_address).unwrap();
        let second_resolution = begin_pipeline_resolution(second_address).unwrap();
        let PipelineChoice::Available { generation: first, .. } = first_resolution.resolve_for_emission()
            else { panic!("first coalesced owner was unavailable") };
        let PipelineChoice::Available { generation: second, .. } = second_resolution.resolve_for_emission()
            else { panic!("second coalesced owner was unavailable") };
        assert!(first.same(&second));
        assert_eq!(native.refs.load(Ordering::Relaxed), 2, "coalesced bind allocated another COM owner");

        assert_eq!(released(first_address), 0);
        assert!(build.one_shot_generation.lock().unwrap().is_some(), "non-final owner cleared the cache");
        let raw_owner = released(second_address);
        assert_eq!(raw_owner, raw);
        assert!(build.one_shot_generation.lock().unwrap().is_none(), "final owner retained the cache");
        release(raw_owner);
        drop(second); drop(first); drop(second_resolution); drop(first_resolution); drop(build);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn emission_ready_choice_keeps_the_bind_generation_after_rebuild() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut second = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let second_raw = &mut second as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        let pending = structured_pending(build.clone(), false);
        let resolution = PipelineResolution(classify_retained(0x7E57_D014, &pending), pending.compute, UnavailableSet::Poison);

        next.store(second_raw, Ordering::Release);
        assert_eq!(rebuild_once(&build), Some(RebuildResult::Published { generation: 2 }));
        let PipelineChoice::Available { raw, generation } = resolution.resolve_for_emission()
            else { panic!("captured generation was unavailable") };
        assert_eq!((raw, generation.id()), (first_raw, 1));
        assert_eq!(first.refs.load(Ordering::Relaxed), 1, "bind pin did not retain generation one");

        drop(generation); drop(resolution);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(pending); drop(build); drop(recipe); release(raw_owner);
        assert_eq!(second.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn released_one_shot_initial_never_exposes_an_unowned_raw() {
        let original = 0x7E57_D015usize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        submit_lazy(original, false, Box::new(move || raw), 0);
        let pending = entry(original).unwrap();
        let resolution = begin_pipeline_resolution(original).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        let together = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let releaser = scope.spawn(|| {
                together.wait();
                let released_raw = released(original);
                assert_eq!(released_raw, raw);
                release(released_raw);
                released_raw
            });
            together.wait();
            match resolution.resolve_for_emission() {
                PipelineChoice::Available { raw: selected, generation } => {
                    assert_eq!(selected, raw);
                    assert!(native.refs.load(Ordering::Relaxed) > 0, "available choice lost its COM owner");
                    drop(generation);
                }
                PipelineChoice::Unavailable => {}
            }
            assert_eq!(releaser.join().unwrap(), raw);
        });
        drop(resolution);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0, "an unowned one-shot raw escaped");
    }

    #[test]
    fn bind_snapshot_defers_evicted_pipeline_without_requesting_restoration() {
        let _global = pipeline_restoration_test_guard();
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            raw
        });
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        let pin = recipe.pin_with_residency(raw).0.unwrap();
        let coordinator = pipeline_restoration_coordinator();
        let evicted = coordinator.install_eviction(&build, pin.ledger_identity()).unwrap().commit();
        drop(pin);
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        let original = 0x7E57_D012usize;
        let pending = structured_pending(build.clone(), false);
        PENDING.insert(original, pending.clone());

        let resolution = begin_pipeline_resolution(original).expect("registered evicted pipeline");
        assert!(matches!(&resolution.0, StructuredResolve::Deferred { pending: retained, .. }
            if Arc::ptr_eq(retained, &pending)));
        assert!(resolution.submit_demand_pending());
        assert_eq!(calls.load(Ordering::Relaxed), 1, "bind snapshot ran the rebuild recipe");
        assert!(build.queued.lock().unwrap().is_none(), "bind snapshot queued build work");
        assert!(coordinator.receipts.records(1).is_empty(), "bind snapshot reserved a receipt");
        assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::Evicted));

        assert!(Arc::ptr_eq(&PENDING.remove(original).unwrap(), &pending));
        assert!(build.clear_eviction_token(&evicted));
        assert!(coordinator.evicted.finish(&evicted, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&evicted));
        drop(resolution); drop(pending); drop(build); drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn retained_submit_demand_restores_one_exact_generation_after_address_reuse() {
        let _global = pipeline_restoration_test_guard();
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let calls = Arc::new(AtomicUsize::new(0));
        let build_next = next.clone(); let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            build_next.load(Ordering::Acquire)
        });
        let build = Arc::new(Build::new_keyed_kind(
            recipe.initial_job(), false, None, Some(recipe.clone()), PipelineKind::Compute,
        ));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        let coordinator = pipeline_restoration_coordinator();
        build.candidate_queued.store(true, Ordering::Release);
        let limits = EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        };
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let mut crossing = None;
        let detached = match detach_exact_eviction_on(
            selected_candidate(&build), 1, limits, coordinator, || {}, || {
                let crossing_build = build.clone();
                crossing = Some(std::thread::spawn(move || {
                    started_tx.send(()).unwrap();
                    let _writer = crossing_build.eviction_timestamp_writer();
                    crossing_build.last_use_frame.store(2, Ordering::Relaxed);
                    done_tx.send(()).unwrap();
                }));
                started_rx.recv().unwrap();
                assert!(done_rx.recv_timeout(std::time::Duration::from_millis(20)).is_err(),
                    "crossing timestamp writer entered an exclusive eviction claim");
            },
        ) {
            ExactDetachAttempt::Detached(detached) => detached,
            ExactDetachAttempt::Aborted(result, _) => panic!("exact detach aborted: {result:?}"),
        };
        done_rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        crossing.unwrap().join().unwrap();
        let evicted = detached.token().clone();
        assert_eq!(build.real.load(Ordering::Acquire), 0);
        assert_eq!(recipe.pin_with_residency(first_raw).1, Residency::Evicted);
        assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::Evicted));
        assert!(build.eviction_token().is_some_and(|token| token.exact(&evicted)));
        assert!(!build.candidate_queued.load(Ordering::Acquire), "successful candidate was requeued");
        assert!(coordinator.attempts.lock().unwrap().is_empty(), "detachment started restoration work");
        assert!(coordinator.admission.try_lock().is_ok());
        assert!(build.done.try_lock().is_ok());
        assert!(recipe.0.rebuild.try_lock().is_ok());
        assert!(recipe.0.generation.try_lock().is_ok());
        assert_eq!(first.refs.load(Ordering::Relaxed), 2, "native owners were destroyed inside commit");
        detached.retire();
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
        next.store(replacement_raw, Ordering::Release);
        let original = 0x7E57_D019usize;
        let pending = structured_pending(build.clone(), true);
        PENDING.insert(original, pending.clone());
        let resolution = begin_pipeline_resolution(original).unwrap();
        assert!(resolution.submit_demand_pending());
        assert_eq!(calls.load(Ordering::Relaxed), 1, "classification ran the rebuild recipe");

        let replacement_pending = structured_pending(Arc::new(Build::new(Box::new(|| 0), false)), false);
        assert!(Arc::ptr_eq(&PENDING.insert(original, replacement_pending.clone()).unwrap(), &pending));
        let clones = [resolution.clone(), resolution.clone()];
        let demanded = std::thread::scope(|scope| {
            let first = scope.spawn(|| clones[0].need_by_submit());
            let second = scope.spawn(|| clones[1].need_by_submit());
            [first.join().unwrap(), second.join().unwrap()]
        });
        assert_eq!(demanded.into_iter().filter(|demanded| *demanded).count(), 1,
            "concurrent clones did not install exactly one submit demand");
        assert!(!resolution.submit_demand_pending(), "concurrent demand did not publish its ticket");
        let PipelineChoice::Available { raw, generation } = resolution.resolve_for_emission()
            else { panic!("exact restoration was unavailable") };
        assert_eq!((raw, generation.id(), calls.load(Ordering::Relaxed)), (replacement_raw, 2, 2));
        assert!(build.eviction_token().is_none(), "terminal restoration was not reconciled");
        assert!(coordinator.evicted.state(&evicted).is_none(), "exact eviction record was not retired");

        assert!(Arc::ptr_eq(&PENDING.remove(original).unwrap(), &replacement_pending));
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(generation); drop(clones); drop(resolution); drop(replacement_pending);
        drop(pending); drop(build); drop(recipe); release(raw_owner);
        assert_eq!((first.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn active_pressure_observer_detach_feeds_retained_demand_restoration() {
        let _global = pipeline_restoration_test_guard();
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let calls = Arc::new(AtomicUsize::new(0));
        let build_next = next.clone(); let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            build_next.load(Ordering::Acquire)
        });
        let build = Arc::new(Build::new_keyed_kind(
            recipe.initial_job(), false, None, Some(recipe.clone()), PipelineKind::Compute,
        ));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        build.completed_frame.store(0, Ordering::Relaxed);
        let candidates = CandidateQueue::<Build, 2>::new();
        note_eviction_candidate_on(&build, 0, &candidates);
        let mut observer = PressureObserver::new(PressureAdmissionConfig {
            enabled: true, max_sample_age_turns: 0, max_evictions_per_turn: 1,
            ..Default::default()
        }, EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        }).unwrap();
        let coordinator = pipeline_restoration_coordinator();
        let (observed, detached) = observer.evict_exact_on(
            &candidates, 1, 100, Some(pressure_sample(1, 100)), coordinator,
        );
        assert_eq!(observed, PressureObservation {
            policy: PressurePolicyMode::Active, mode: PressureMode::Evicting,
            scanned: 1, eligible: 1, attempted: 1, evicted: 1,
            ..Default::default()
        });
        let detached = detached.expect("active observer did not detach its exact candidate");
        let evicted = detached.token().clone();
        assert_eq!(build.real.load(Ordering::Acquire), 0);
        assert!(build.eviction_token().is_some_and(|token| token.exact(&evicted)));
        assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::Evicted));
        assert!(candidates.pop().is_none(), "successful candidate remained in the pressure journal");
        detached.retire();
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);

        next.store(replacement_raw, Ordering::Release);
        let original = 0x7E57_D02Busize;
        let pending = structured_pending(build.clone(), true);
        PENDING.insert(original, pending.clone());
        let resolution = begin_pipeline_resolution(original).expect("registered evicted pipeline");
        assert!(matches!(&resolution.0, StructuredResolve::Deferred { pending: retained, .. }
            if Arc::ptr_eq(retained, &pending)));
        let reused = structured_pending(Arc::new(Build::new(Box::new(|| 0), false)), false);
        assert!(Arc::ptr_eq(&PENDING.insert(original, reused.clone()).unwrap(), &pending));
        assert!(resolution.need_by_submit(), "retained demand did not queue exact restoration");
        let PipelineChoice::Available { raw, generation } = resolution.resolve_for_emission()
            else { panic!("observer-detached pipeline was not restored") };
        assert_eq!((raw, generation.id(), calls.load(Ordering::Relaxed)), (replacement_raw, 2, 2));
        assert!(build.eviction_token().is_none(), "terminal restoration was not reconciled");
        assert!(coordinator.evicted.state(&evicted).is_none(), "eviction ledger record was not retired");

        assert!(Arc::ptr_eq(&PENDING.remove(original).unwrap(), &reused));
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(generation); drop(resolution); drop(reused); drop(pending);
        drop(build); drop(recipe); release(raw_owner);
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn retained_failed_demand_observes_another_binds_successful_retry() {
        let _global = pipeline_restoration_test_guard();
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(raw)); let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        let pin = recipe.pin_with_residency(raw).0.unwrap();
        let coordinator = pipeline_restoration_coordinator();
        let evicted = coordinator.install_eviction(&build, pin.ledger_identity()).unwrap().commit();
        drop(pin); assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        next.store(0, Ordering::Release);
        let pending = structured_pending(build.clone(), true);
        let resolution = PipelineResolution(classify_retained(0x7E57_D020, &pending), pending.compute, UnavailableSet::Poison);
        assert!(matches!(&resolution.0, StructuredResolve::Deferred { .. }));

        assert!(resolution.need_by_submit());
        assert!(matches!(resolution.resolve_for_indirect(), IndirectPipelineChoice::Unavailable),
            "failed exact restoration escaped through the initial fallback policy");
        assert_eq!(coordinator.evicted.state(&evicted), Some(EvictedBuildState::Evicted),
            "failed restoration was not returned to retryable eviction state");
        assert!(coordinator.receipts.records(1).is_empty(), "terminal failure receipt was not reconciled");
        assert!(build.eviction_token().is_some_and(|token| token.exact(&evicted)),
            "retryable failure lost its exact eviction identity");

        next.store(replacement_raw, Ordering::Release);
        let retry = PipelineResolution(classify_retained(0x7E57_D021, &pending), pending.compute, UnavailableSet::Poison);
        assert!(retry.need_by_submit());
        let IndirectPipelineChoice::Exact { raw: retry_raw, generation: retry_generation } = retry.resolve_for_indirect()
            else { panic!("second bind did not restore the exact recipe") };
        assert_eq!(retry_raw, replacement_raw);
        let PipelineChoice::Available { raw: first_raw, generation: first_generation } = resolution.resolve_for_emission()
            else { panic!("failed demand did not observe the resident retry") };
        assert_eq!(first_raw, replacement_raw);
        assert!(first_generation.same(&retry_generation));
        assert!(build.eviction_token().is_none(), "successful retry was not reconciled");
        assert!(coordinator.evicted.state(&evicted).is_none(), "successful retry left the exact eviction record");

        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(first_generation); drop(retry_generation); drop(retry); drop(resolution);
        drop(pending); drop(build); drop(recipe); release(raw_owner);
        assert_eq!((native.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn abandoned_terminal_demand_is_swept_before_the_next_exact_request() {
        let _global = pipeline_restoration_test_guard();
        let coordinator = pipeline_restoration_coordinator();
        reconcile_pipeline_restorations();

        let abandoned_build = ledger_build();
        let abandoned_evicted = coordinator.install_eviction(
            &abandoned_build, ledger_native(73),
        ).unwrap().commit();
        assert!(coordinator.refuse_rehydrate(&abandoned_evicted));
        let abandoned_attempt = coordinator.receipts.records(PIPELINE_RESOLUTION_RECEIPTS)
            .into_iter().find(|record| record.token.evicted.exact(&abandoned_evicted))
            .expect("refused exact attempt was not recorded").token;
        let abandoned: RestorationDemand<PIPELINE_RESOLUTION_RECEIPTS> =
            RestorationDemand::new(abandoned_evicted.clone());
        assert!(abandoned.requested.set(DeferredDemand::Failed(
            StructuredResolveFailure::Refused,
        )).is_ok());
        drop(abandoned);
        assert!(coordinator.receipts.records(PIPELINE_RESOLUTION_RECEIPTS)
            .iter().any(|record| record.token.exact(&abandoned_attempt)),
            "demand destruction swept global restoration state");

        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(
            recipe.initial_job(), false, None, Some(recipe.clone()),
        ));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        let pin = recipe.pin_with_residency(first_raw).0.unwrap();
        let evicted = coordinator.install_eviction(&build, pin.ledger_identity()).unwrap().commit();
        drop(pin);
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        next.store(replacement_raw, Ordering::Release);
        let pending = structured_pending(build.clone(), true);
        let demand: RestorationDemand<PIPELINE_RESOLUTION_RECEIPTS> =
            RestorationDemand::new(evicted);
        let requested = demand.request(&pending);

        assert!(!coordinator.receipts.records(PIPELINE_RESOLUTION_RECEIPTS)
            .iter().any(|record| record.token.exact(&abandoned_attempt)),
            "the next exact request did not sweep the abandoned terminal attempt");
        let generation = match requested {
            DeferredDemand::Ready(generation) => Some(generation),
            DeferredDemand::Restoring(ticket) => ticket.wait_and_take_pin().ok(),
            DeferredDemand::Failed(_) => None,
        };
        reconcile_pipeline_restorations();

        assert!(abandoned_build.clear_eviction_token(&abandoned_evicted));
        assert!(coordinator.evicted.finish(&abandoned_evicted, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&abandoned_evicted));
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(generation); drop(demand); drop(pending); drop(build); drop(recipe);
        release(raw_owner);
        assert_eq!((first.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn restoration_coordinator_reuses_its_bounded_eviction_capacity() {
        let coordinator = RestorationCoordinator::<2, PIPELINE_RESOLUTION_RECEIPTS>::new(510).unwrap();
        let first_build = ledger_build();
        let second_build = ledger_build();
        let third_build = ledger_build();
        let first = coordinator.install_eviction(&first_build, ledger_native(70)).unwrap().commit();
        let second = coordinator.install_eviction(&second_build, ledger_native(71)).unwrap().commit();
        assert!(coordinator.install_eviction(&third_build, ledger_native(72)).is_none());

        assert!(first_build.clear_eviction_token(&first));
        assert!(coordinator.evicted.finish(&first, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&first));
        drop(coordinator.install_eviction(&third_build, ledger_native(72))
            .expect("retired bounded slot was not reusable"));

        assert!(second_build.clear_eviction_token(&second));
        assert!(coordinator.evicted.finish(&second, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&second));
        assert!(coordinator.evicted.records(2).is_empty());
    }

    #[test]
    fn dormant_resolution_maps_queue_shutdown_without_inline_driver_work() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            raw
        });
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        let pin = recipe.pin_with_residency(raw).0.unwrap();
        let coordinator = RestorationCoordinator::<1, PIPELINE_RESOLUTION_RECEIPTS>::new(511).unwrap();
        let evicted = coordinator.install_eviction(&build, pin.ledger_identity()).unwrap().commit();
        drop(pin);
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        let pending = structured_pending(build.clone(), false);
        let q = BuildQueue::new();
        q.shutdown();

        let resolution = begin_pipeline_resolution_on(0x7E57_D011, &pending, &coordinator, &q);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "classification ran driver work inline");
        assert!(matches!(resolution.0.wait_exact(), Err(StructuredResolveFailure::Refused)));
        assert_eq!(coordinator.reconcile(1), 1);

        assert!(build.clear_eviction_token(&evicted));
        assert!(coordinator.evicted.finish(&evicted, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&evicted));
        drop(resolution); drop(pending); drop(build); drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn structured_resolve_retains_pending_across_title_address_reuse() {
        let original = 0x7E57_D001usize;
        let old = structured_pending(Arc::new(Build::new(Box::new(|| 0), false)), false);
        let coordinator = RestorationCoordinator::<1, 1>::new(501).unwrap();
        let q = BuildQueue::new();
        let outcome = classify_structured_on(original, &old, &coordinator, &q);
        let replacement = structured_pending(Arc::new(Build::new(Box::new(|| 0), false)), true);
        PENDING.insert(original, replacement.clone());

        match outcome {
            StructuredResolve::Initial { original: retained_original, pending, fallback, bound_unbuilt, .. } => {
                assert_eq!((retained_original, fallback, bound_unbuilt), (original, original, true));
                assert!(Arc::ptr_eq(&pending, &old), "address reuse redirected a retained decision");
                assert!(!Arc::ptr_eq(&pending, &replacement));
            }
            _ => panic!("unfinished build was not classified as initial"),
        }
        assert!(Arc::ptr_eq(&PENDING.remove(original).unwrap(), &replacement));
    }

    #[test]
    fn structured_resolve_fails_closed_without_exact_eviction_record() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        let pending = structured_pending(build.clone(), false);
        let coordinator = RestorationCoordinator::<1, 1>::new(502).unwrap();
        let outcome = classify_structured_on(0x7E57_D002, &pending, &coordinator, &BuildQueue::new());
        assert!(matches!(outcome, StructuredResolve::Terminal {
            failure: StructuredResolveFailure::MissingEvictionRecord, ..
        }));
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
        drop(pending); drop(build); drop(recipe);
    }

    #[test]
    fn structured_resolve_keeps_exact_pin_during_speculative_rebuild() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        assert!(matches!(recipe.begin_rebuild(), RebuildTurn::Leader));
        let pending = structured_pending(build.clone(), false);
        let coordinator = RestorationCoordinator::<1, 1>::new(504).unwrap();
        let outcome = classify_structured_on(0x7E57_D004, &pending, &coordinator, &BuildQueue::new());
        let (chosen, pin) = match &outcome {
            StructuredResolve::Ready { raw, pin: Some(pin), .. } => (*raw, pin.clone()),
            _ => panic!("a rebuilding replacement hid the current exact generation"),
        };
        drop(outcome);
        assert_eq!((chosen, pin.raw()), (raw, raw));
        recipe.finish_rebuild(RebuildResult::Failed);
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(pin); drop(pending); drop(build); drop(recipe); release(raw_owner);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn structured_resolve_promotes_exact_background_without_inline_driver_work() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let calls = Arc::new(AtomicUsize::new(0));
        let build_next = next.clone(); let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            build_next.load(Ordering::Acquire)
        });
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        let pin = recipe.pin_with_residency(first_raw).0.unwrap();
        let coordinator = RestorationCoordinator::<1, 1>::new(503).unwrap();
        let evicted = coordinator.install_eviction(&build, pin.ledger_identity()).unwrap().commit();
        drop(pin);
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        next.store(replacement_raw, Ordering::Release);
        let pending = structured_pending(build.clone(), false);
        let q = BuildQueue::new();
        let background = coordinator.request_exact_on(&evicted, RehydratePriority::Background, &q).unwrap();
        let outcome = classify_structured_on(0x7E57_D003, &pending, &coordinator, &q);
        let StructuredResolve::Restoring { ticket, .. } = &outcome else { panic!("eviction did not classify as restoring") };
        assert!(background.token().exact(ticket.token()), "foreground classification did not coalesce exactly");
        assert_eq!(calls.load(Ordering::Relaxed), 1, "classification ran driver work inline");
        assert!(!q.try_run_one_between(Priority::Speculative, Priority::Preload), "attempt was not promoted");
        assert!(q.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        let resolved = outcome.wait_exact().unwrap().unwrap();
        assert_eq!((resolved.raw, resolved.bound_unbuilt, calls.load(Ordering::Relaxed)),
            (replacement_raw, false, 2));
        assert!(resolved.pin.is_some());

        drop(resolved); drop(outcome); drop(background);
        assert_eq!(coordinator.reconcile(1), 1);
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(pending); drop(build); drop(recipe); release(raw_owner);
        assert_eq!((first.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn structured_resolve_prefers_publication_over_stale_cancelled_request() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw)); let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap(); complete(&build, initial());
        let pin = recipe.pin_with_residency(first_raw).0.unwrap();
        let coordinator = RestorationCoordinator::<1, 2>::new(505).unwrap();
        let evicted = coordinator.install_eviction(&build, pin.ledger_identity()).unwrap().commit();
        drop(pin); assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        next.store(replacement_raw, Ordering::Release);
        let pending = structured_pending(build.clone(), false); let q = BuildQueue::new();
        let background = coordinator.request_exact_on(&evicted, RehydratePriority::Background, &q).unwrap();
        let (seen_tx, seen_rx) = mpsc::channel(); let (continue_tx, continue_rx) = mpsc::channel();
        *coordinator.before_classified_request.lock().unwrap() = Some(Box::new(move || {
            seen_tx.send(()).unwrap(); continue_rx.recv().unwrap();
        }));
        std::thread::scope(|scope| {
            let classifier = scope.spawn(|| classify_structured_on(0x7E57_D005, &pending, &coordinator, &q));
            seen_rx.recv().unwrap();
            assert!(q.try_run_one_between(Priority::Speculative, Priority::Preload));
            drop(background.wait_and_take_pin().unwrap()); drop(background);
            assert_eq!(coordinator.reconcile(1), 1);
            continue_tx.send(()).unwrap();
            let outcome = classifier.join().unwrap();
            assert!(matches!(outcome, StructuredResolve::Ready { raw, pin: Some(_), .. } if raw == replacement_raw),
                "stale cancelled request hid a concurrently published generation");
            drop(outcome);
        });
        assert_eq!(coordinator.reconcile(2), 1, "stale request was not retired");
        let raw_owner = build.real.swap(0, Ordering::AcqRel);
        drop(pending); drop(build); drop(recipe); release(raw_owner);
        assert_eq!((first.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn pressure_observer_bounds_inspection_and_requeues_every_resident() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = [
            RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) },
            RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) },
            RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) },
        ];
        let mut builds = Vec::new();
        let mut recipes = Vec::new();
        for value in &mut native {
            let raw = value as *mut RebuildFakePso as usize;
            let recipe = Recipe::new(move || raw);
            let build = Arc::new(Build::new_keyed_kind(
                recipe.initial_job(), false, None, Some(recipe.clone()), PipelineKind::Compute,
            ));
            let job = build.job.lock().unwrap().take().unwrap();
            complete(&build, job());
            builds.push(build);
            recipes.push(recipe);
        }
        let queue = CandidateQueue::<Build, 4>::new();
        for (build, last_use) in builds.iter().zip([99, 10, 20]) {
            build.completed_frame.store(0, Ordering::Relaxed);
            note_eviction_candidate_on(build, last_use, &queue);
        }

        let config = PressureAdmissionConfig {
            enabled: true,
            max_sample_age_turns: 1,
            max_evictions_per_turn: 1,
            ..PressureAdmissionConfig::default()
        };
        let limits = EvictionSelectionLimits {
            max_scan: 3, max_select: 2, min_unused_frames: 10, min_resident_frames: 0,
        };
        let mut observer = PressureObserver::new(config, limits).unwrap();
        let coordinator = RestorationCoordinator::<4, 4>::new(1).unwrap();
        let positions = || (
            queue.enqueue.load(Ordering::Relaxed), queue.dequeue.load(Ordering::Relaxed),
        );

        let before = positions();
        let mut disabled = PressureObserver::new(PressureAdmissionConfig::default(), limits).unwrap();
        assert_eq!(disabled.observe_on(&queue, 7, 100, Some(pressure_sample(7, 100))), PressureObservation::default());
        assert_eq!(positions(), before, "a disabled observer touched the candidate journal");
        assert_eq!(observer.observe_on(&queue, 8, 100, None), PressureObservation::default());
        assert_eq!(positions(), before, "a missing sample touched the candidate journal");
        let stale = PressureSample {
            observed_turn: 7,
            segment: super::super::memory_report::Segment { usage: 95, budget: 100 },
        };
        assert_eq!(observer.observe_on(&queue, 9, 100, Some(stale)), PressureObservation::default());
        assert_eq!(positions(), before, "a stale sample touched the candidate journal");

        let high = PressureSample {
            observed_turn: 9,
            segment: super::super::memory_report::Segment { usage: 95, budget: 100 },
        };
        let observed = observer.observe_exact_on(&queue, 10, 100, Some(high), &coordinator);
        assert_eq!(positions(), (before.0 + 3, before.1 + 3), "an observed candidate was not requeued exactly once");
        assert_eq!(observed, PressureObservation {
            mode: PressureMode::Evicting,
            scanned: 3,
            eligible: 2,
            deferred: 1,
            discarded: 0,
            requeue_saturated: 0,
            attempted: 2,
            prepared: 2,
            ..Default::default()
        });
        assert!(observed.attempts_accounted());
        assert_eq!((observer.scans_issued, observer.attempts_issued, observer.admission.evictions_succeeded),
            (3, 2, 0), "scan, attempt, and success accounting were not independent");
        assert!(builds.iter().all(|build| build.real.load(Ordering::Acquire) != 0));
        assert!(builds.iter().all(|build| build.candidate_queued.load(Ordering::Acquire)));
        assert!(builds.iter().all(|build| build.eviction_token().is_none()));
        assert!(coordinator.evicted.records(4).is_empty());

        let same_turn = PressureSample {
            observed_turn: 10,
            segment: super::super::memory_report::Segment { usage: 95, budget: 100 },
        };
        assert_eq!(observer.observe_on(&queue, 10, 100, Some(same_turn)), PressureObservation {
            mode: PressureMode::Evicting, ..Default::default()
        }, "a second sample replenished the same-turn scan or attempt allowance");

        // Deterministically model producers consuming every freed slot between drain and requeue.
        // The observer must retain exact candidates instead of clearing their queued identity.
        let fillers: Vec<_> = (0..3).map(|_| Arc::new(Build::new(Box::new(|| 0), false))).collect();
        let next = PressureSample {
            observed_turn: 11,
            segment: super::super::memory_report::Segment { usage: 95, budget: 100 },
        };
        let saturated = observer.observe_on_after_drain(&queue, 11, 100, Some(next), || {
            for filler in &fillers {
                assert!(queue.push(Arc::downgrade(filler)).is_ok());
            }
        });
        assert_eq!(saturated, PressureObservation {
            mode: PressureMode::Evicting,
            scanned: 3,
            eligible: 2,
            deferred: 1,
            discarded: 0,
            requeue_saturated: 2,
            ..Default::default()
        });
        assert_eq!(observer.held.len(), 2);
        assert!(observer.held.iter().all(|candidate| candidate.build.upgrade()
            .is_some_and(|build| build.candidate_queued.load(Ordering::Acquire))));
        let blocked = observer.observe_on(&queue, 12, 100, None);
        assert_eq!(blocked.requeue_saturated, 2);
        assert_eq!(blocked.scanned, 0, "a saturated conservation backlog allowed another drain");
        let crossing = observer.held[0].build.upgrade().unwrap();
        let crossing_recipe = crossing.recipe.as_ref().unwrap().clone();
        assert!(matches!(crossing_recipe.begin_rebuild(), RebuildTurn::Leader));
        let before_crossing_bind = positions();
        note_eviction_candidate_on(&crossing, 100, &queue);
        assert_eq!(positions(), before_crossing_bind, "a crossing bind duplicated the held identity");

        let mut seen = HashSet::new();
        for _ in 0..2 {
            let build = queue.pop().unwrap().upgrade().unwrap();
            if builds.iter().any(|expected| Arc::ptr_eq(expected, &build)) {
                assert!(seen.insert(Arc::as_ptr(&build) as usize));
            }
        }
        let recovered = observer.observe_on(&queue, 13, 100, None);
        assert_eq!(recovered.requeue_saturated, 0);
        assert!(observer.held.is_empty());
        crossing_recipe.finish_rebuild(RebuildResult::Failed);
        drop(crossing);
        drop(crossing_recipe);
        for _ in 0..4 {
            let build = queue.pop().unwrap().upgrade().unwrap();
            if builds.iter().any(|expected| Arc::ptr_eq(expected, &build)) {
                assert!(seen.insert(Arc::as_ptr(&build) as usize), "observation duplicated a journal record");
            }
        }
        assert_eq!(seen.len(), 3);
        assert!(queue.pop().is_none());
        for build in &builds {
            build.candidate_queued.store(false, Ordering::Release);
        }

        let finals: Vec<_> = builds.iter().map(|build| build.real.swap(0, Ordering::AcqRel)).collect();
        drop(builds);
        drop(recipes);
        for raw in finals {
            release(raw);
        }
        assert!(native.iter().all(|value| value.refs.load(Ordering::Relaxed) == 0));
    }

    #[test]
    fn exact_eviction_preflight_aborts_without_mutating_the_resident_pipeline() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = completed_recipe_build(&recipe);
        build.completed_frame.store(1, Ordering::Relaxed);
        build.last_use_frame.store(2, Ordering::Relaxed);
        let selected = selected_candidate(&build);
        let coordinator = RestorationCoordinator::<1, 1>::new(1).unwrap();
        let limits = EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 10, min_resident_frames: 10,
        };

        assert_eq!(preflight_exact_eviction_on(&selected, 100, limits, &coordinator, || {}),
            EvictionPreflight::Prepared);
        assert_eq!(build.real.load(Ordering::Acquire), raw);
        assert!(build.eviction_token().is_none());
        assert!(coordinator.evicted.records(1).is_empty());
        assert_eq!(recipe.pin_with_residency(raw).1, Residency::Resident);
        assert_eq!(native.refs.load(Ordering::Relaxed), 2);

        let panic_coordinator = RestorationCoordinator::<1, 1>::new(2).unwrap();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = preflight_exact_eviction_on(&selected, 100, limits, &panic_coordinator, || panic!("hook"));
        })).is_err());
        assert!(build.eviction_token().is_none());
        assert!(panic_coordinator.evicted.records(1).is_empty());
        assert_eq!(recipe.pin_with_residency(raw).1, Residency::Resident);

        let pin = recipe.pin_with_residency(raw).0.unwrap();
        let existing = coordinator.install_eviction(&build, pin.ledger_identity()).unwrap().commit();
        assert_eq!(preflight_exact_eviction_on(&selected, 100, limits, &coordinator, || {}),
            EvictionPreflight::Busy);
        assert!(build.eviction_token().is_some_and(|token| token.exact(&existing)));
        assert!(build.clear_eviction_token(&existing));
        assert!(coordinator.evicted.finish(&existing, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&existing));
        drop(pin);

        build.last_use_frame.store(100, Ordering::Relaxed);
        assert_eq!(preflight_exact_eviction_on(&selected, 100, limits, &coordinator, || {}),
            EvictionPreflight::Recent);
        build.last_use_frame.store(2, Ordering::Relaxed);
        assert_eq!(preflight_exact_eviction_on(&selected, 100, limits, &coordinator, || {
            assert!(matches!(recipe.begin_rebuild(), RebuildTurn::Leader));
        }),
            EvictionPreflight::Busy);
        recipe.finish_rebuild(RebuildResult::Failed);
        assert_eq!(preflight_exact_eviction_on(&selected, 100, limits, &coordinator, || {
            build.owners.store(0, Ordering::Release);
        }),
            EvictionPreflight::Released);

        release(build.real.swap(0, Ordering::AcqRel));
        drop(selected); drop(build); drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn every_restoration_admission_path_waits_for_shadow_cleanup() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = completed_recipe_build(&recipe);
        let selected = selected_candidate(&build);
        let coordinator = Arc::new(RestorationCoordinator::<2, 8>::new(1).unwrap());

        assert_shadow_blocks_restoration(&selected, &coordinator,
            |coordinator, token| { let _ = coordinator.refuse_rehydrate(&token); });
        let _ = coordinator.reconcile(8);
        assert_shadow_blocks_restoration(&selected, &coordinator,
            |coordinator, token| { let _ = coordinator.queue_rehydrate(
                &token, RehydratePriority::Background, true); });
        let _ = coordinator.reconcile(8);
        assert_shadow_blocks_restoration(&selected, &coordinator,
            |coordinator, token| { let q = BuildQueue::new(); q.shutdown();
                let _ = coordinator.request_exact_on(&token, RehydratePriority::Foreground, &q); });
        let _ = coordinator.reconcile(8);
        assert_shadow_blocks_restoration(&selected, &coordinator,
            |coordinator, _| { let _ = coordinator.retries(1); });

        release(build.real.swap(0, Ordering::AcqRel));
        drop(selected); drop(build); drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn exact_eviction_preflight_rejects_generation_change_after_install() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut second = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let second_raw = &mut second as *mut RebuildFakePso as usize;
        let source = Arc::new(AtomicUsize::new(first_raw));
        let recipe = Recipe::new({ let source = source.clone(); move || source.load(Ordering::Acquire) });
        let build = completed_recipe_build(&recipe);
        let selected = selected_candidate(&build);
        build.candidate_queued.store(true, Ordering::Release);
        let coordinator = RestorationCoordinator::<1, 1>::new(1).unwrap();
        let limits = EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        };

        let candidate = match detach_exact_eviction_on(selected, 1, limits, &coordinator, || {
            source.store(second_raw, Ordering::Release);
            build.completed_frame.store(1, Ordering::Relaxed);
            assert!(matches!(rebuild_once(&build), Some(RebuildResult::Published { generation: 2 })));
        }, || {}) {
            ExactDetachAttempt::Aborted(EvictionPreflight::Changed, candidate) => candidate,
            _ => panic!("generation crossing did not abort exact detachment"),
        };
        assert_eq!(build.real.load(Ordering::Acquire), second_raw);
        assert!(build.eviction_token().is_none());
        assert!(coordinator.evicted.records(1).is_empty());
        assert_eq!((first.refs.load(Ordering::Relaxed), second.refs.load(Ordering::Relaxed)), (0, 2));
        let queue = CandidateQueue::<Build, 2>::new();
        assert!(candidate.try_requeue_on(&queue).is_ok());
        assert!(queue.pop().unwrap().upgrade().is_some());
        assert!(queue.pop().is_none(), "aborted candidate was requeued more than once");
        build.candidate_queued.store(false, Ordering::Release);

        release(build.real.swap(0, Ordering::AcqRel));
        drop(build); drop(recipe);
        assert_eq!((first.refs.load(Ordering::Relaxed), second.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn detached_exact_eviction_unwind_retires_owners_outside_every_lock() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_reenter_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = completed_recipe_build(&recipe);
        let coordinator = RestorationCoordinator::<1, 1>::new(1).unwrap();
        let probe = EvictionReleaseProbe {
            done: &build.done,
            rebuild: &recipe.0.rebuild,
            generation: &recipe.0.generation,
            admission: &coordinator.admission,
            timestamp_access: &build.eviction_timestamp_access,
            observer: std::ptr::null(),
            pressure_stats: std::ptr::null(),
        };
        EVICTION_RELEASE_PROBE.with(|address| address.set(&probe as *const _ as usize));
        let limits = EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        };
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let ExactDetachAttempt::Detached(_detached) = detach_exact_eviction_on(
                selected_candidate(&build), 1, limits, &coordinator, || {}, || {},
            ) else { panic!("exact detachment aborted") };
            panic!("drop detached owners during unwind");
        }));
        EVICTION_RELEASE_PROBE.with(|address| address.set(0));
        assert!(panicked.is_err());
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
        let token = build.eviction_token().unwrap();
        assert!(build.clear_eviction_token(&token));
        assert!(coordinator.evicted.finish(&token, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&token));
    }

    #[test]
    fn pressure_lifecycle_distinguishes_detach_retirement_and_final_generation_release() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_reenter_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = completed_recipe_build(&recipe);
        let retained = recipe.pin_with_residency(raw).0.unwrap();
        let coordinator = RestorationCoordinator::<1, 1>::new(1).unwrap();
        let limits = EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        };
        let ExactDetachAttempt::Detached(detached) = detach_exact_eviction_on(
            selected_candidate(&build), 1, limits, &coordinator, || {}, || {},
        ) else { panic!("exact detachment aborted") };
        let stats = Arc::new(PipelinePressureCounters::default());
        let probe = EvictionReleaseProbe {
            done: &build.done,
            rebuild: &recipe.0.rebuild,
            generation: &recipe.0.generation,
            admission: &coordinator.admission,
            timestamp_access: &build.eviction_timestamp_access,
            observer: std::ptr::null(),
            pressure_stats: Arc::as_ptr(&stats),
        };
        EVICTION_RELEASE_PROBE.with(|address| address.set(&probe as *const _ as usize));
        detached.retire_pressure(stats.clone());
        assert_eq!(stats.snapshot(), PipelinePressureStats {
            publications_detached: 1,
            policy_owner_retirements: 1,
            ..Default::default()
        });
        let line = pressure_observation_line(PressureObservation {
            policy: PressurePolicyMode::Active,
            mode: PressureMode::Evicting,
            evicted: 1,
            attempted: 1,
            ..Default::default()
        }, stats.snapshot());
        assert!(line.ends_with(
            "cumulative: 1 detached, 1 policy owners retired, 0 generation owners released, 0 restoration attempts (0 succeeded, 0 failed)"
        ), "pressure line did not snapshot lifecycle counters after owner retirement: {line}");
        assert_eq!(native.refs.load(Ordering::Relaxed), 1,
            "policy retirement released a generation still retained by recorded work");
        drop(retained);
        assert_eq!(stats.generation_owners_released.load(Ordering::Relaxed), 1);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
        EVICTION_RELEASE_PROBE.with(|address| address.set(0));

        let token = build.eviction_token().unwrap();
        assert!(build.clear_eviction_token(&token));
        assert!(coordinator.evicted.finish(&token, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&token));
    }

    #[test]
    fn exact_eviction_preflight_reports_full_ledger_and_timestamp_crossing() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = completed_recipe_build(&recipe);
        let selected = selected_candidate(&build);
        let coordinator = RestorationCoordinator::<1, 1>::new(1).unwrap();
        let occupied_build = ledger_build();
        let occupied = coordinator.reserve_eviction(&occupied_build, ledger_native(9)).unwrap().arm_detach().unwrap();
        let limits = EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        };
        assert_eq!(preflight_exact_eviction_on(&selected, 10, limits, &coordinator, || {}),
            EvictionPreflight::LedgerFull);
        assert!(coordinator.evicted.finish(&occupied, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&occupied));

        let candidate = match detach_exact_eviction_on(selected, 10, limits, &coordinator, || {
            build.last_use_frame.store(1, Ordering::Relaxed);
        }, || {}) {
            ExactDetachAttempt::Aborted(EvictionPreflight::Recent, candidate) => candidate,
            _ => panic!("timestamp crossing did not abort exact detachment"),
        };
        assert!(build.eviction_token().is_none());
        assert!(coordinator.evicted.records(1).is_empty());
        drop(candidate);

        release(build.real.swap(0, Ordering::AcqRel));
        drop(build); drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn saturated_pressure_backlog_does_not_retain_a_released_native_pipeline() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed_kind(
            recipe.initial_job(), false, None, Some(recipe.clone()), PipelineKind::Compute,
        ));
        let job = build.job.lock().unwrap().take().unwrap();
        complete(&build, job());
        let queue = CandidateQueue::<Build, 2>::new();
        note_eviction_candidate_on(&build, 0, &queue);
        let fillers = [Arc::new(Build::new(Box::new(|| 0), false)),
            Arc::new(Build::new(Box::new(|| 0), false))];
        let config = PressureAdmissionConfig {
            enabled: true, max_evictions_per_turn: 1, ..PressureAdmissionConfig::default()
        };
        let mut observer = PressureObserver::new(config, EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        }).unwrap();
        let saturated = observer.observe_on_after_drain(&queue, 1, 1, Some(pressure_sample(1, 100)), || {
            for filler in &fillers {
                assert!(queue.push(Arc::downgrade(filler)).is_ok());
            }
        });
        assert_eq!(saturated.requeue_saturated, 1);
        assert_eq!(observer.held.len(), 1);

        build.owners.store(0, Ordering::Release);
        release(build.real.swap(0, Ordering::AcqRel));
        drop(build);
        drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0, "weak backlog retained the native generation");
        assert!(observer.held[0].build.upgrade().is_none());
        let expired = observer.observe_on(&queue, 2, 1, None);
        assert_eq!((expired.requeue_saturated, expired.discarded), (0, 1));
        assert!(observer.held.is_empty(), "an expired weak identity remained backlogged");
    }

    #[test]
    fn pressure_observer_reports_every_popped_invalid_record() {
        let queue = CandidateQueue::<Build, 4>::new();
        let dead = Arc::new(Build::new(Box::new(|| 0), false));
        assert!(queue.push(Arc::downgrade(&dead)).is_ok());
        drop(dead);
        let nonresident = Arc::new(Build::new_keyed_kind(
            Box::new(|| 0), false, None, None, PipelineKind::Compute,
        ));
        nonresident.candidate_queued.store(true, Ordering::Release);
        assert!(queue.push(Arc::downgrade(&nonresident)).is_ok());
        let config = PressureAdmissionConfig {
            enabled: true, max_evictions_per_turn: 2, ..PressureAdmissionConfig::default()
        };
        let mut observer = PressureObserver::new(config, EvictionSelectionLimits {
            max_scan: 2, max_select: 2, min_unused_frames: 0, min_resident_frames: 0,
        }).unwrap();
        assert_eq!(observer.observe_on(&queue, 1, 1, Some(pressure_sample(1, 100))), PressureObservation {
            mode: PressureMode::Evicting,
            scanned: 2,
            eligible: 0,
            deferred: 1,
            discarded: 1,
            requeue_saturated: 0,
            ..Default::default()
        });
        assert!(queue.pop().unwrap().upgrade().is_some());
    }

    #[test]
    fn pressure_observer_discards_an_injected_ineligible_record_once() {
        let queue = CandidateQueue::<Build, 2>::new();
        let injected = Arc::new(Build::new_keyed_kind(
            Box::new(|| 0), false, None, None, PipelineKind::ConsoleListCompute,
        ));
        injected.candidate_queued.store(true, Ordering::Release);
        assert!(queue.push(Arc::downgrade(&injected)).is_ok());
        let config = PressureAdmissionConfig {
            enabled: true, max_evictions_per_turn: 1, ..PressureAdmissionConfig::default()
        };
        let mut observer = PressureObserver::new(config, EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        }).unwrap();

        assert_eq!(observer.observe_on(&queue, 1, 1, Some(pressure_sample(1, 100))), PressureObservation {
            mode: PressureMode::Evicting,
            scanned: 1,
            discarded: 1,
            ..Default::default()
        });
        assert!(!injected.candidate_queued.load(Ordering::Acquire));
        assert!(queue.pop().is_none());
        assert!(observer.held.is_empty());
        assert_eq!(observer.observe_on(&queue, 2, 1, Some(pressure_sample(2, 100))).scanned, 0,
            "the observer requeued an ineligible identity");
    }

    #[test]
    fn rejected_pressure_turns_do_not_replenish_observer_work() {
        let queue = CandidateQueue::<Build, 2>::new();
        let nonresident = Arc::new(Build::new_keyed_kind(
            Box::new(|| 0), false, None, None, PipelineKind::Compute,
        ));
        nonresident.candidate_queued.store(true, Ordering::Release);
        assert!(queue.push(Arc::downgrade(&nonresident)).is_ok());
        let config = PressureAdmissionConfig {
            enabled: true, max_sample_age_turns: 2, max_evictions_per_turn: 1,
            ..PressureAdmissionConfig::default()
        };
        let mut observer = PressureObserver::new(config, EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        }).unwrap();

        assert_eq!(observer.observe_on(&queue, 10, 1, Some(pressure_sample(8, 100))).scanned, 1);
        let positions = (queue.enqueue.load(Ordering::Relaxed), queue.dequeue.load(Ordering::Relaxed));
        assert_eq!(observer.observe_on(&queue, 9, 1, Some(pressure_sample(9, 100))).scanned, 0,
            "a backward policy turn with a forward sample replenished scan work");
        assert_eq!(observer.observe_on(&queue, 11, 1, None).scanned, 0,
            "a missing sample replenished scan work");
        assert_eq!(observer.observe_on(&queue, 10, 1, Some(pressure_sample(10, 100))).scanned, 0,
            "a later valid sample replenished an already-consumed policy turn");
        assert_eq!((queue.enqueue.load(Ordering::Relaxed), queue.dequeue.load(Ordering::Relaxed)), positions);
        assert_eq!((observer.work_budget_turn, observer.scans_issued), (Some(10), 1));

        assert!(queue.pop().unwrap().upgrade().is_some());
        nonresident.candidate_queued.store(false, Ordering::Release);
    }

    #[test]
    fn pressure_observer_work_replenishes_once_across_turn_wrap() {
        let queue = CandidateQueue::<Build, 2>::new();
        let nonresident = Arc::new(Build::new_keyed_kind(
            Box::new(|| 0), false, None, None, PipelineKind::Compute,
        ));
        nonresident.candidate_queued.store(true, Ordering::Release);
        assert!(queue.push(Arc::downgrade(&nonresident)).is_ok());
        let config = PressureAdmissionConfig {
            enabled: true, max_evictions_per_turn: 1, ..PressureAdmissionConfig::default()
        };
        let mut observer = PressureObserver::new(config, EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        }).unwrap();

        assert_eq!(observer.observe_on(&queue, u64::MAX, 1,
            Some(pressure_sample(u64::MAX, 100))).scanned, 1);
        assert_eq!(observer.observe_on(&queue, 0, 1, Some(pressure_sample(0, 100))).scanned, 1,
            "the serial turn wrap did not replenish work");
        let positions = (queue.enqueue.load(Ordering::Relaxed), queue.dequeue.load(Ordering::Relaxed));
        assert_eq!(observer.observe_on(&queue, 0, 1, Some(pressure_sample(0, 100))).scanned, 0,
            "the wrapped turn replenished work more than once");
        assert_eq!((queue.enqueue.load(Ordering::Relaxed), queue.dequeue.load(Ordering::Relaxed)), positions);
        assert_eq!((observer.work_budget_turn, observer.scans_issued), (Some(0), 1));

        assert!(queue.pop().unwrap().upgrade().is_some());
        nonresident.candidate_queued.store(false, Ordering::Release);
    }

    #[test]
    fn rebuild_recipe_gate_requires_exact_one() {
        assert!(recipe_gate_value(Some("1")));
        assert!(recipe_gate_value(Some(" 1\t")));
        assert!(pressure_gate_value(Some(" 1\t")));
        assert_eq!(pressure_policy_mode_values(Some("1"), Some("1")),
            Some(PressurePolicyMode::Active));
        assert_eq!(pressure_policy_mode_values(None, Some(" 1\t")),
            Some(PressurePolicyMode::Shadow));
        for value in [None, Some(""), Some("0"), Some("true"), Some("2")] {
            assert!(!recipe_gate_value(value));
            assert!(!pressure_gate_value(value));
            assert_eq!(pressure_policy_mode_values(value, value), None);
        }
    }

    #[test]
    fn active_pressure_observer_preserves_attempt_and_success_budgets() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_reenter_release as usize];
        let mut native = [
            RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) },
            RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) },
            RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) },
        ];
        let mut builds = Vec::new();
        let mut recipes = Vec::new();
        let queue = CandidateQueue::<Build, 4>::new();
        for (last_use, value) in native.iter_mut().enumerate() {
            let raw = value as *mut RebuildFakePso as usize;
            let recipe = Recipe::new(move || raw);
            let build = completed_recipe_build(&recipe);
            build.completed_frame.store(0, Ordering::Relaxed);
            note_eviction_candidate_on(&build, last_use as u64, &queue);
            builds.push(build);
            recipes.push(recipe);
        }
        let config = PressureAdmissionConfig {
            enabled: true, max_sample_age_turns: 0, max_evictions_per_turn: 1,
            ..Default::default()
        };
        let limits = EvictionSelectionLimits {
            max_scan: 3, max_select: 3, min_unused_frames: 0, min_resident_frames: 0,
        };
        let observer = Mutex::new(PressureObserver::new(config, limits).unwrap());
        let coordinator = RestorationCoordinator::<4, 4>::new(1).unwrap();
        let busy_pin = recipes[0].pin_with_residency(
            builds[0].real.load(Ordering::Acquire),
        ).0.unwrap();
        let busy = coordinator.install_eviction(
            &builds[0], busy_pin.ledger_identity(),
        ).unwrap().commit();
        drop(busy_pin);
        let (observed, detached) = observer.lock().unwrap().evict_exact_on(
            &queue, 1, 100, Some(pressure_sample(1, 100)), &coordinator,
        );
        assert_eq!(observed, PressureObservation {
            policy: PressurePolicyMode::Active,
            mode: PressureMode::Evicting,
            scanned: 3,
            eligible: 3,
            attempted: 2,
            evicted: 1,
            busy: 1,
            ..Default::default()
        });
        assert!(observed.attempts_accounted());
        {
            let state = observer.lock().unwrap();
            assert_eq!((state.attempts_issued, state.admission.evictions_succeeded), (2, 1),
                "the failed exact attempt consumed the success allowance");
        }
        let evicted = builds.iter().position(|build| build.real.load(Ordering::Acquire) == 0).unwrap();
        assert_ne!(evicted, 0, "the already-armed candidate unexpectedly detached");
        assert!(builds[evicted].eviction_token().is_some());
        assert!(!builds[evicted].candidate_queued.load(Ordering::Acquire));
        assert!(builds.iter().enumerate().filter(|(i, _)| *i != evicted)
            .all(|(_, build)| build.candidate_queued.load(Ordering::Acquire)));
        assert!(queue.pop().unwrap().upgrade().is_some());
        assert!(queue.pop().unwrap().upgrade().is_some());
        assert!(queue.pop().is_none(), "the successful candidate was requeued");

        let release_probe = EvictionReleaseProbe {
            done: &builds[evicted].done,
            rebuild: &recipes[evicted].0.rebuild,
            generation: &recipes[evicted].0.generation,
            admission: &coordinator.admission,
            timestamp_access: &builds[evicted].eviction_timestamp_access,
            observer: &observer,
            pressure_stats: std::ptr::null(),
        };
        EVICTION_RELEASE_PROBE.with(|address| address.set(&release_probe as *const _ as usize));
        detached.unwrap().retire();
        EVICTION_RELEASE_PROBE.with(|address| address.set(0));
        let token = builds[evicted].eviction_token().unwrap();
        assert!(builds[evicted].clear_eviction_token(&token));
        assert!(coordinator.evicted.finish(&token, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&token));
        assert!(builds[0].clear_eviction_token(&busy));
        assert!(coordinator.evicted.finish(&busy, EvictedBuildState::Poisoned));
        assert!(coordinator.evicted.retire(&busy));
        for (_, build) in builds.iter().enumerate().filter(|(i, _)| *i != evicted) {
            build.candidate_queued.store(false, Ordering::Release);
            release(build.real.swap(0, Ordering::AcqRel));
        }
        drop(builds); drop(recipes);
        assert!(native.iter().all(|value| value.refs.load(Ordering::Relaxed) == 0));
    }

    #[test]
    fn only_non_console_compute_builds_enter_the_eviction_journal() {
        assert_eq!(PipelineKind::submitted(false, false), PipelineKind::Graphics);
        assert_eq!(PipelineKind::submitted(false, true), PipelineKind::Graphics);
        assert_eq!(PipelineKind::submitted(true, true), PipelineKind::ConsoleListCompute);
        assert_eq!(PipelineKind::submitted(true, false), PipelineKind::Compute);

        let queue = CandidateQueue::<Build, 4>::new();
        for kind in [PipelineKind::Unknown, PipelineKind::Graphics, PipelineKind::ConsoleListCompute] {
            let build = Arc::new(Build::new_keyed_kind(Box::new(|| 0), false, None, None, kind));
            build.last_use_frame.store(0, Ordering::Relaxed);
            note_eviction_candidate_on(&build, 17, &queue);
            assert_eq!(build.last_use_frame.load(Ordering::Relaxed), 17,
                "{kind:?} lost retention-age tracking");
            assert!(!build.candidate_queued.load(Ordering::Acquire), "{kind:?} entered the journal");
        }
        assert!(queue.pop().is_none());

        let compute = Arc::new(Build::new_keyed_kind(
            Box::new(|| 0), false, None, None, PipelineKind::Compute,
        ));
        note_eviction_candidate_on(&compute, 17, &queue);
        assert!(compute.candidate_queued.load(Ordering::Acquire));
        assert!(queue.pop().unwrap().upgrade().is_some());
        compute.candidate_queued.store(false, Ordering::Release);

        let injected = Arc::new(Build::new_keyed_kind(
            Box::new(|| 0), false, None, None, PipelineKind::Graphics,
        ));
        injected.candidate_queued.store(true, Ordering::Release);
        assert!(queue.push(Arc::downgrade(&injected)).is_ok());
        assert!(drain_eviction_candidates_on(&queue, 1).is_empty());
        assert!(!injected.candidate_queued.load(Ordering::Acquire),
            "draining an injected ineligible record left its dedup bit stuck");
    }

    #[test]
    fn retained_ready_use_refreshes_and_journals_its_exact_old_build_once() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = completed_recipe_build(&recipe);
        let pending = structured_pending(build.clone(), true);
        let resolution = PipelineResolution(
            classify_retained(0x7E57_D02A, &pending), true, UnavailableSet::Omit,
        );
        let replacement = structured_pending(
            Arc::new(Build::new_keyed_kind(
                Box::new(|| 0), false, None, None, PipelineKind::Compute,
            )),
            true,
        );
        let StructuredResolve::Ready { pending: retained, .. } = &resolution.0 else {
            panic!("completed recipe did not retain a Ready identity")
        };
        assert!(Arc::ptr_eq(retained, &pending));
        assert!(!Arc::ptr_eq(retained, &replacement));

        let queue = CandidateQueue::<Build, 2>::new();
        build.last_use_frame.store(1, Ordering::Relaxed);
        replacement.build.last_use_frame.store(3, Ordering::Relaxed);
        retained.note_resident_use_on(17, &queue);
        assert_eq!(build.last_use_frame.load(Ordering::Relaxed), 17);
        assert_eq!(replacement.build.last_use_frame.load(Ordering::Relaxed), 3,
            "retained Ready use was redirected to a replacement identity");
        retained.note_resident_use_on(29, &queue);
        assert_eq!(build.last_use_frame.load(Ordering::Relaxed), 29);
        let queued = queue.pop().unwrap().upgrade().unwrap();
        assert!(Arc::ptr_eq(&queued, &build));
        assert!(queue.pop().is_none(), "repeated exact use duplicated the journal identity");
        build.candidate_queued.store(false, Ordering::Release);

        let PipelineChoice::Available { raw: selected, generation } =
            resolution.resolve_for_emission() else { panic!("retained Ready use was unavailable") };
        assert_eq!(selected, raw);
        assert!(build.candidate_queued.load(Ordering::Acquire),
            "the production Ready emission path did not journal its retained build");

        drop(generation); drop(queued); drop(resolution); drop(replacement); drop(pending);
        release(build.real.swap(0, Ordering::AcqRel));
        drop(build); drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn injected_console_compute_cannot_prepare_or_commit_exact_eviction() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = completed_recipe_build_as(&recipe, PipelineKind::ConsoleListCompute);
        let candidate = || EvictionCandidate {
            build: build.clone(),
            last_use_frame: build.last_use_frame.load(Ordering::Relaxed),
            completed_frame: build.completed_frame.load(Ordering::Relaxed),
            clear_queued_on_drop: true,
        };
        let selected = SelectedEvictionCandidate {
            candidate: candidate(), unused_frames: 100, resident_frames: 100,
        };
        let coordinator = RestorationCoordinator::<2, 2>::new(1).unwrap();
        let limits = EvictionSelectionLimits {
            max_scan: 1, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        };
        let hook_ran = AtomicBool::new(false);
        assert_eq!(preflight_exact_eviction_on(&selected, 1, limits, &coordinator,
            || hook_ran.store(true, Ordering::Release)), EvictionPreflight::Changed);
        assert!(!hook_ran.load(Ordering::Acquire));
        assert!(build.eviction_token().is_none());
        assert!(coordinator.evicted.records(2).is_empty());
        assert_eq!((build.real.load(Ordering::Acquire), native.refs.load(Ordering::Relaxed)), (raw, 2));

        // Forge the state after preparation to exercise the final independent kind gate. The
        // installed token must unwind without detaching either native owner.
        let generation = recipe.pin_with_residency(raw).0.unwrap();
        let admission = coordinator.admission.lock().unwrap();
        let installed = coordinator.install_eviction(&build, generation.ledger_identity()).unwrap();
        let prepared = PreparedExactEviction {
            installed,
            _admission: admission,
            first: ExactEvictionSnapshot {
                generation,
                last_use_frame: build.last_use_frame.load(Ordering::Relaxed),
                completed_frame: build.completed_frame.load(Ordering::Relaxed),
            },
            frame: 1,
            limits,
        };
        assert!(matches!(prepared.commit_exact(|| panic!("ineligible final hook ran")),
            Err(EvictionPreflight::Changed)));
        assert!(build.eviction_token().is_none());
        assert!(coordinator.evicted.records(2).is_empty());
        assert_eq!((build.real.load(Ordering::Acquire), native.refs.load(Ordering::Relaxed)), (raw, 2));

        drop(selected);
        release(build.real.swap(0, Ordering::AcqRel));
        drop(build); drop(recipe);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn eviction_candidate_ring_is_bounded_reusable_and_does_not_retain_builds() {
        assert!(std::panic::catch_unwind(CandidateQueue::<usize, 0>::new).is_err());
        assert!(std::panic::catch_unwind(CandidateQueue::<usize, 1>::new).is_err());
        let queue = CandidateQueue::<usize, 2>::new();
        let first = Arc::new(1usize);
        let second = Arc::new(2usize);
        let third = Arc::new(3usize);
        assert!(queue.push(Arc::downgrade(&first)).is_ok());
        assert!(queue.push(Arc::downgrade(&second)).is_ok());
        assert!(queue.push(Arc::downgrade(&third)).is_err(), "a full candidate ring accepted an unbounded record");
        assert_eq!(*queue.pop().unwrap().upgrade().unwrap(), 1);
        assert!(queue.push(Arc::downgrade(&third)).is_ok(), "a drained ring slot was not reusable");
        assert_eq!(*queue.pop().unwrap().upgrade().unwrap(), 2);
        assert_eq!(*queue.pop().unwrap().upgrade().unwrap(), 3);
        assert!(queue.pop().is_none());

        let builds = CandidateQueue::<Build, 2>::new();
        let build = Arc::new(Build::new_keyed_kind(
            Box::new(|| 0), false, None, None, PipelineKind::Compute,
        ));
        let weak = Arc::downgrade(&build);
        note_eviction_candidate_on(&build, 7, &builds);
        drop(build);
        assert!(weak.upgrade().is_none(), "an unconsumed candidate retained a released build");
        assert!(builds.pop().unwrap().upgrade().is_none(), "a released build record changed identity");
    }

    #[test]
    fn eviction_candidate_ring_mpmc_delivers_every_weak_once() {
        const COUNT: usize = 512;
        let queue = CandidateQueue::<usize, 64>::new();
        let values: Vec<_> = (0..COUNT).map(Arc::new).collect();
        let seen: Vec<_> = (0..COUNT).map(|_| AtomicUsize::new(0)).collect();
        let consumed = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for producer in 0..4 {
                let queue = &queue;
                let values = &values;
                scope.spawn(move || {
                    for index in (producer..COUNT).step_by(4) {
                        let mut value = Arc::downgrade(&values[index]);
                        loop {
                            match queue.push(value) {
                                Ok(()) => break,
                                Err(returned) => {
                                    value = returned;
                                    std::thread::yield_now();
                                }
                            }
                        }
                    }
                });
            }
            for _ in 0..4 {
                let queue = &queue;
                let seen = &seen;
                let consumed = &consumed;
                scope.spawn(move || loop {
                    if consumed.load(Ordering::Acquire) == COUNT {
                        break;
                    }
                    let Some(value) = queue.pop() else {
                        std::thread::yield_now();
                        continue;
                    };
                    let index = *value.upgrade().expect("payload died while retained by the test");
                    seen[index].fetch_add(1, Ordering::Relaxed);
                    consumed.fetch_add(1, Ordering::Release);
                });
            }
        });
        assert_eq!(consumed.load(Ordering::Relaxed), COUNT);
        assert!(seen.iter().all(|count| count.load(Ordering::Relaxed) == 1), "a payload was lost or delivered twice");
        assert!(values.iter().all(|value| Arc::weak_count(value) == 0), "a queued Weak payload was not dropped exactly once");
        assert!(queue.pop().is_none());
    }

    #[test]
    fn eviction_candidates_deduplicate_bound_drain_and_filter_ineligible_builds() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut second = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let second_raw = &mut second as *mut RebuildFakePso as usize;
        let first_recipe = Recipe::new(move || first_raw);
        let second_recipe = Recipe::new(move || second_raw);
        let first_build = Arc::new(Build::new_keyed_kind(
            first_recipe.initial_job(), false, None, Some(first_recipe.clone()), PipelineKind::Compute,
        ));
        let second_build = Arc::new(Build::new_keyed_kind(
            second_recipe.initial_job(), false, None, Some(second_recipe.clone()), PipelineKind::Compute,
        ));
        let first_job = first_build.job.lock().unwrap().take().unwrap();
        let second_job = second_build.job.lock().unwrap().take().unwrap();
        complete(&first_build, first_job());
        complete(&second_build, second_job());
        let queue = CandidateQueue::<Build, 4>::new();

        // Ordinary/default resolution bookkeeping does not touch the opt-in candidate journal.
        let pending = Pending {
            build: first_build.clone(), driver: AtomicUsize::new(0), compute: false, key: 0,
            manifest_recipe: AtomicU64::new(0),
            track_used: true,
            used: AtomicBool::new(false), late: AtomicBool::new(false), spared: AtomicBool::new(false),
            on_use: Mutex::new(None),
        };
        pending.note_use();
        assert!(!first_build.candidate_queued.load(Ordering::Acquire));

        note_eviction_candidate_on(&first_build, 10, &queue);
        note_eviction_candidate_on(&first_build, 30, &queue);
        note_eviction_candidate_on(&second_build, 20, &queue);
        let limits = EvictionSelectionLimits {
            max_scan: 2, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        };
        let selected = select_eviction_candidates_on(&queue, 100, limits);
        assert_eq!(selected.len(), 1);
        assert!(Arc::ptr_eq(&selected[0].candidate.build, &second_build));
        assert_eq!(selected[0].unused_frames, 80);
        assert_eq!(drain_eviction_candidates_on(&queue, 0).len(), 0, "a zero budget inspected work");
        drop(selected);
        let selected = select_eviction_candidates_on(&queue, 100, limits);
        assert_eq!(selected.len(), 1);
        assert!(Arc::ptr_eq(&selected[0].candidate.build, &first_build));
        drop(selected);

        let recent_limits = EvictionSelectionLimits {
            max_scan: 2, max_select: 1, min_unused_frames: 10, min_resident_frames: 0,
        };
        note_eviction_candidate_on(&first_build, 95, &queue);
        for _ in 0..3 {
            assert!(select_eviction_candidates_on(&queue, 100, recent_limits).is_empty());
            assert!(first_build.candidate_queued.load(Ordering::Acquire));
        }
        let selected = select_eviction_candidates_on(&queue, 105, recent_limits);
        assert_eq!(selected.len(), 1, "a deferred candidate required another bind to reappear");
        assert!(Arc::ptr_eq(&selected[0].candidate.build, &first_build));
        assert!(queue.pop().is_none(), "repeated deferral duplicated a candidate record");
        drop(selected);

        assert!(matches!(first_recipe.begin_rebuild(), RebuildTurn::Leader));
        note_eviction_candidate_on(&first_build, 40, &queue);
        assert!(select_eviction_candidates_on(&queue, 100, limits).is_empty(), "an active rebuild became evictable");
        first_recipe.finish_rebuild(RebuildResult::Failed);
        first_build.owners.store(0, Ordering::Release);
        note_eviction_candidate_on(&first_build, 50, &queue);
        assert!(select_eviction_candidates_on(&queue, 100, limits).is_empty(), "a released build became evictable");

        drop(pending);
        let first_final = first_build.real.swap(0, Ordering::AcqRel);
        let second_final = second_build.real.swap(0, Ordering::AcqRel);
        drop(first_build);
        drop(second_build);
        drop(first_recipe);
        drop(second_recipe);
        release(first_final);
        release(second_final);
        assert_eq!((first.refs.load(Ordering::Relaxed), second.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn eviction_candidate_policy_uses_latest_age_and_orders_oldest_first() {
        let make = |last_use, completed| {
            let build = Arc::new(Build::new(Box::new(|| 0), false));
            build.last_use_frame.store(last_use, Ordering::Relaxed);
            build.completed_frame.store(completed, Ordering::Relaxed);
            build.candidate_queued.store(true, Ordering::Release);
            EvictionCandidate {
                build, last_use_frame: last_use, completed_frame: completed,
                clear_queued_on_drop: true,
            }
        };
        let oldest = make(10, 5);
        let next = make(20, 1);
        let recent = make(95, 0);
        let young_resident = make(1, 95);
        let wrapped = make(u64::MAX, 0);
        let oldest_build = oldest.build.clone();
        let next_build = next.build.clone();
        let limits = EvictionSelectionLimits {
            max_scan: 4, max_select: 2, min_unused_frames: 10, min_resident_frames: 20,
        };

        let (selected, _) =
            partition_eviction_candidates(vec![recent, next, wrapped, young_resident, oldest], 100, limits);
        assert_eq!(selected.len(), 2);
        assert!(Arc::ptr_eq(&selected[0].candidate.build, &oldest_build));
        assert_eq!((selected[0].unused_frames, selected[0].resident_frames), (90, 95));
        assert!(Arc::ptr_eq(&selected[1].candidate.build, &next_build));

        // A use after background drain is still visible through the Build atomic, even though the
        // candidate's dedup flag prevents a second journal record while this owner exists.
        let used_after_drain = make(30, 1);
        used_after_drain.build.last_use_frame.store(99, Ordering::Relaxed);
        assert!(partition_eviction_candidates(vec![used_after_drain], 100, limits).0.is_empty());
        assert_eq!(conservative_frame_age(3, u64::MAX), 0, "frame wrap assigned a false old age");
    }

    #[test]
    fn eviction_candidate_requeue_full_clears_the_dedup_flag() {
        let queue = CandidateQueue::<Build, 2>::new();
        let retained = [
            Arc::new(Build::new(Box::new(|| 0), false)),
            Arc::new(Build::new(Box::new(|| 0), false)),
        ];
        assert!(queue.push(Arc::downgrade(&retained[0])).is_ok());
        assert!(queue.push(Arc::downgrade(&retained[1])).is_ok());

        let build = Arc::new(Build::new(Box::new(|| 0), false));
        build.candidate_queued.store(true, Ordering::Release);
        let candidate = EvictionCandidate {
            build: build.clone(), last_use_frame: 0, completed_frame: 0,
            clear_queued_on_drop: true,
        };
        assert!(!candidate.requeue_on(&queue));
        assert!(!build.candidate_queued.load(Ordering::Acquire));
    }

    #[test]
    fn build_retains_a_reusable_recipe_after_its_initial_job_is_consumed() {
        let calls = Arc::new(AtomicUsize::new(0));
        let recipe_calls = calls.clone();
        let recipe = Recipe::new(move || {
            recipe_calls.fetch_add(1, Ordering::Relaxed) + 0x5100
        });
        let build = Build::new_keyed(recipe.initial_job(), false, None, Some(recipe));

        let initial = build.job.lock().unwrap().take().unwrap();
        assert_eq!(initial(), 0x5100);
        assert!(build.job.lock().unwrap().is_none());
        assert_eq!(build.recipe.as_ref().unwrap().run(), 0x5101);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn native_generation_slot_pins_exact_aba_owner_and_ids_publications() {
        struct Owner(Arc<AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) { self.0.fetch_add(1, Ordering::Relaxed); }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let mut slot = GenerationSlot::default();
        let (first, replaced) = slot.publish(0x7E57_ABA0, Owner(drops.clone()));
        assert!(replaced.is_none());
        let pinned = slot.pin(0x7E57_ABA0).unwrap();
        assert_eq!(first.id, 1);
        let (second, replaced) = slot.publish(0x7E57_ABA0, Owner(drops.clone()));
        assert_eq!(second.id, 2);
        assert!(!Arc::ptr_eq(&pinned, &second), "a reused raw address aliased its older generation");
        let unique = dedup_generations(
            [first.clone(), first.clone(), second.clone(), second.clone()].into_iter(),
            |generation| Arc::as_ptr(generation) as usize,
        );
        assert_eq!(unique.len(), 2, "a reused raw address collapsed distinct generations");
        drop(unique);
        drop(first);
        drop(replaced);
        assert_eq!(drops.load(Ordering::Relaxed), 0, "the old generation died while a pin held it");
        drop(pinned);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        drop(second);
        drop(slot);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[derive(Clone)]
    struct FakeReceipt {
        fence: Arc<AtomicU64>,
        value: u64,
    }

    fn fake_receipt_complete(receipt: &FakeReceipt) -> bool {
        receipt.fence.load(Ordering::Relaxed) >= receipt.value
    }

    fn fake_fence_identity(receipt: &FakeReceipt) -> usize {
        Arc::as_ptr(&receipt.fence) as usize
    }

    fn fake_completed_value(receipt: &FakeReceipt) -> u64 {
        receipt.fence.load(Ordering::Relaxed)
    }

    fn fake_complete_observed(receipt: &FakeReceipt, observed: &HashMap<usize, u64>) -> bool {
        observed.get(&fake_fence_identity(receipt)).is_some_and(|&value| value >= receipt.value)
    }

    fn sweep_fake<G>(state: &Mutex<GenerationRetirements<G, FakeReceipt>>) {
        sweep_retirements(state, fake_fence_identity, fake_completed_value, fake_complete_observed);
    }

    #[test]
    fn generation_retirement_deduplicates_and_uses_the_exact_receipt_owner() {
        let generation = Arc::new(());
        let pins = dedup_generations(
            [generation.clone(), generation.clone()].into_iter(),
            |generation| Arc::as_ptr(generation) as usize,
        );
        assert_eq!(pins.len(), 1);

        let old_fence = Arc::new(AtomicU64::new(0));
        let replacement_fence = Arc::new(AtomicU64::new(99));
        let retiring = Mutex::new(GenerationRetirements::new());
        retiring.lock().unwrap().push(FakeReceipt { fence: old_fence.clone(), value: 7 }, pins);
        sweep_fake(&retiring);
        assert_eq!(retiring.lock().unwrap().pending.len(), 1, "a replacement queue generation retired an old receipt");
        assert_eq!(Arc::strong_count(&generation), 2, "the in-flight batch did not retain its generation");

        old_fence.store(7, Ordering::Relaxed);
        sweep_fake(&retiring);
        assert!(retiring.lock().unwrap().pending.is_empty());
        assert_eq!(Arc::strong_count(&generation), 1);
        assert_eq!(replacement_fence.load(Ordering::Relaxed), 99);

        retiring.lock().unwrap().push(FakeReceipt { fence: old_fence, value: 7 }, vec![generation.clone()]);
        sweep_fake(&retiring);
        assert!(retiring.lock().unwrap().pending.is_empty(), "an already-complete receipt retained a final batch");
    }

    #[test]
    fn generation_retirement_probes_each_exact_fence_once_and_honors_cutoffs() {
        let (a, b) = (Arc::new(AtomicU64::new(5)), Arc::new(AtomicU64::new(0)));
        let state = Mutex::new(GenerationRetirements::new());
        {
            let mut retiring = state.lock().unwrap();
            for value in [2, 5, 8] {
                retiring.push(FakeReceipt { fence: a.clone(), value }, vec![value]);
            }
            retiring.push(FakeReceipt { fence: b.clone(), value: 1 }, vec![1]);
        }
        let (a_probes, b_probes) = (AtomicUsize::new(0), AtomicUsize::new(0));
        sweep_retirements(
            &state,
            fake_fence_identity,
            |receipt| {
                assert!(state.try_lock().is_ok(), "fence observation ran under the retirement mutex");
                if Arc::ptr_eq(&receipt.fence, &a) {
                    a_probes.fetch_add(1, Ordering::Relaxed);
                } else if Arc::ptr_eq(&receipt.fence, &b) {
                    b_probes.fetch_add(1, Ordering::Relaxed);
                } else {
                    panic!("unexpected fence owner");
                }
                fake_completed_value(receipt)
            },
            fake_complete_observed,
        );

        assert_eq!(
            (a_probes.load(Ordering::Relaxed), b_probes.load(Ordering::Relaxed)),
            (1, 1),
            "each exact fence owner must be probed once",
        );
        let retiring = state.lock().unwrap();
        assert_eq!(retiring.pending.len(), 2);
        assert!(retiring.pending.iter().any(|(r, _)| Arc::ptr_eq(&r.fence, &a) && r.value == 8));
        assert!(retiring.pending.iter().any(|(r, _)| Arc::ptr_eq(&r.fence, &b) && r.value == 1));
    }

    #[test]
    fn generation_retirement_applies_a_snapshot_to_only_observed_fences() {
        let (a, b) = (Arc::new(AtomicU64::new(5)), Arc::new(AtomicU64::new(99)));
        let state = Mutex::new(GenerationRetirements::new());
        state.lock().unwrap().push(FakeReceipt { fence: a.clone(), value: 3 }, vec![3]);
        let appended = AtomicBool::new(false);
        sweep_retirements(
            &state,
            fake_fence_identity,
            |receipt| {
                assert!(state.try_lock().is_ok(), "fence observation ran under the retirement mutex");
                if !appended.swap(true, Ordering::Relaxed) {
                    let mut retiring = state.lock().unwrap();
                    retiring.push(FakeReceipt { fence: a.clone(), value: 4 }, vec![4]);
                    retiring.push(FakeReceipt { fence: a.clone(), value: 7 }, vec![7]);
                    retiring.push(FakeReceipt { fence: b.clone(), value: 1 }, vec![1]);
                }
                fake_completed_value(receipt)
            },
            fake_complete_observed,
        );

        let retiring = state.lock().unwrap();
        assert_eq!(retiring.pending.len(), 2);
        assert!(retiring.pending.iter().any(|(r, _)| Arc::ptr_eq(&r.fence, &a) && r.value == 7));
        assert!(retiring.pending.iter().any(|(r, _)| Arc::ptr_eq(&r.fence, &b) && r.value == 1));
    }

    #[test]
    fn generation_retirement_drops_completed_pins_without_the_mutex() {
        struct DropPin {
            state: Weak<Mutex<GenerationRetirements<DropPin, FakeReceipt>>>,
            dropped: Arc<AtomicBool>,
            unlocked: Arc<AtomicBool>,
        }
        impl Drop for DropPin {
            fn drop(&mut self) {
                let unlocked = self.state.upgrade().is_some_and(|state| state.try_lock().is_ok());
                self.unlocked.store(unlocked, Ordering::Relaxed);
                self.dropped.store(true, Ordering::Relaxed);
            }
        }

        let state = Arc::new(Mutex::new(GenerationRetirements::new()));
        let (dropped, unlocked) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let pin = DropPin { state: Arc::downgrade(&state), dropped: dropped.clone(), unlocked: unlocked.clone() };
        state.lock().unwrap().push(
            FakeReceipt { fence: Arc::new(AtomicU64::new(1)), value: 1 },
            vec![pin],
        );
        sweep_fake(&state);

        assert!(dropped.load(Ordering::Relaxed));
        assert!(unlocked.load(Ordering::Relaxed), "completed generation dropped under the retirement mutex");
    }

    #[test]
    fn generation_poison_returns_duplicates_for_drop_after_unlock() {
        struct PoisonPin {
            id: usize,
            state: Weak<Mutex<GenerationRetirements<PoisonPin, FakeReceipt>>>,
            dropped: Arc<AtomicBool>,
            unlocked: Arc<AtomicBool>,
        }
        impl Drop for PoisonPin {
            fn drop(&mut self) {
                let unlocked = self.state.upgrade().is_some_and(|state| state.try_lock().is_ok());
                self.unlocked.store(unlocked, Ordering::Relaxed);
                self.dropped.store(true, Ordering::Relaxed);
            }
        }

        let state = Arc::new(Mutex::new(GenerationRetirements::new()));
        let (kept_dropped, kept_unlocked) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let kept = PoisonPin {
            id: 7, state: Arc::downgrade(&state), dropped: kept_dropped.clone(), unlocked: kept_unlocked,
        };
        drop(state.lock().unwrap().poison(vec![kept], |pin| pin.id));
        let (duplicate_dropped, duplicate_unlocked) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let duplicate = PoisonPin {
            id: 7, state: Arc::downgrade(&state), dropped: duplicate_dropped.clone(), unlocked: duplicate_unlocked.clone(),
        };
        let duplicates = state.lock().unwrap().poison(vec![duplicate], |pin| pin.id);
        assert!(!duplicate_dropped.load(Ordering::Relaxed), "duplicate poison pin dropped inside the mutex");
        assert!(!kept_dropped.load(Ordering::Relaxed), "the permanently retained generation was released");
        drop(duplicates);
        assert!(duplicate_dropped.load(Ordering::Relaxed));
        assert!(duplicate_unlocked.load(Ordering::Relaxed), "duplicate poison pin dropped under the mutex");
    }

    #[test]
    fn untracked_generation_pins_are_never_swept() {
        let generation = Arc::new(());
        let weak = Arc::downgrade(&generation);
        let pins = dedup_generations(
            [generation.clone(), generation.clone()].into_iter(),
            |generation| Arc::as_ptr(generation) as usize,
        );
        let mut retiring = GenerationRetirements::<Arc<()>, FakeReceipt>::new();
        drop(retiring.poison(pins, |generation| Arc::as_ptr(generation) as usize));
        drop(retiring.poison(vec![generation.clone()], |generation| Arc::as_ptr(generation) as usize));
        assert_eq!(retiring.poisoned.len(), 1, "a later poison duplicated an existing generation");
        drop(generation);

        drop(retiring.sweep(fake_receipt_complete));
        assert!(weak.upgrade().is_some(), "sweeping tracked receipts released an untracked generation");
        assert_eq!(retiring.poisoned.len(), 1);
    }

    #[test]
    fn newly_completed_batch_is_returned_for_drop_after_unlock() {
        struct DropPin(Arc<AtomicUsize>);
        impl Drop for DropPin {
            fn drop(&mut self) { self.0.fetch_add(1, Ordering::Relaxed); }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let fence = Arc::new(AtomicU64::new(1));
        let retiring = Mutex::new(GenerationRetirements::new());
        retiring.lock().unwrap().push(FakeReceipt { fence, value: 1 }, vec![DropPin(drops.clone())]);
        sweep_fake(&retiring);

        assert!(retiring.lock().unwrap().pending.is_empty());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn concurrent_generation_publications_are_monotonic() {
        let slot = Arc::new(Mutex::new(GenerationSlot::default()));
        std::thread::scope(|scope| {
            for thread in 0..4usize {
                let slot = slot.clone();
                scope.spawn(move || {
                    for k in 0..100usize {
                        let (_, replaced) = slot.lock().unwrap().publish(thread * 100 + k + 1, ());
                        drop(replaced);
                    }
                });
            }
        });
        let slot = slot.lock().unwrap();
        assert_eq!(slot.next, 400);
        assert_eq!(slot.resident.as_ref().unwrap().id, 400);
    }

    #[test]
    fn overlapping_recipe_rebuild_requests_share_one_result() {
        let recipe = Recipe::new(|| 0);
        assert!(matches!(recipe.begin_rebuild(), RebuildTurn::Leader));
        let follower_recipe = recipe.clone();
        let follower = std::thread::spawn(move || follower_recipe.begin_rebuild());
        let started = std::time::Instant::now();
        loop {
            if recipe.0.rebuild.lock().unwrap().waiters == 1 {
                break;
            }
            assert!(started.elapsed() < std::time::Duration::from_secs(2), "follower did not join the active rebuild");
            std::thread::yield_now();
        }
        let result = RebuildResult::Published { generation: 7 };
        recipe.finish_rebuild(result);
        assert!(matches!(follower.join().unwrap(), RebuildTurn::Follower(r) if r == result));
        assert_eq!(recipe.0.rebuild.lock().unwrap().completed, 1);
    }

    #[test]
    fn next_leader_waits_for_prior_followers_to_consume_their_result() {
        let recipe = Recipe::new(|| 0);
        let first = RebuildResult::Published { generation: 11 };
        {
            // Deterministically model a completed attempt whose notified follower has not yet
            // reacquired the mutex. `last` must remain immutable until that waiter drains.
            let mut state = recipe.0.rebuild.lock().unwrap();
            state.completed = 1;
            state.last = Some(first);
            state.waiters = 1;
        }
        let next_recipe = recipe.clone();
        let (leader_tx, leader_rx) = mpsc::channel();
        let next = std::thread::spawn(move || leader_tx.send(next_recipe.begin_rebuild()).unwrap());
        assert!(leader_rx.recv_timeout(std::time::Duration::from_millis(50)).is_err(), "a new leader crossed an undrained prior follower");

        let consumed = {
            let mut state = recipe.0.rebuild.lock().unwrap();
            let result = state.last.unwrap();
            state.waiters = 0;
            recipe.0.rebuild_done.notify_all();
            result
        };
        assert_eq!(consumed, first);
        assert!(matches!(leader_rx.recv().unwrap(), RebuildTurn::Leader));
        recipe.finish_rebuild(RebuildResult::Published { generation: 12 });
        next.join().unwrap();
    }

    #[test]
    fn failed_rebuild_keeps_current_generation_and_success_replaces_it() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut second = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let second_raw = &mut second as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        assert_eq!((build.real.load(Ordering::Acquire), first.refs.load(Ordering::Relaxed)), (first_raw, 2));

        next.store(0, Ordering::Release);
        assert_eq!(rebuild_once(&build), Some(RebuildResult::Failed));
        assert_eq!(build.real.load(Ordering::Acquire), first_raw);
        assert_eq!(recipe.pin_with_residency(first_raw).0.unwrap().id(), 1);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Resident);

        next.store(second_raw, Ordering::Release);
        assert_eq!(rebuild_once(&build), Some(RebuildResult::Published { generation: 2 }));
        assert_eq!(build.real.load(Ordering::Acquire), second_raw);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0, "both superseded owners release outside publication locks");
        assert_eq!(second.refs.load(Ordering::Relaxed), 2);

        let final_raw = build.real.swap(0, Ordering::AcqRel);
        drop(build);
        drop(recipe);
        assert_eq!(second.refs.load(Ordering::Relaxed), 1);
        release(final_raw);
        assert_eq!(second.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn eviction_detaches_both_native_owners_but_an_exact_pin_survives() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        let pin = recipe.pin_with_residency(raw).0.expect("resident generation");

        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        assert_eq!(build.real.load(Ordering::Acquire), 0);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Evicted);
        assert!(recipe.pin_with_residency(raw).0.is_none(), "evicted generation remained current");
        assert_eq!(native.refs.load(Ordering::Relaxed), 1, "the exact pin did not retain the native object");
        assert_eq!(try_evict_current(&build), EvictResult::NotResident, "repeated eviction detached another owner");

        drop(pin);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn eviction_between_resolution_and_pinning_falls_back_from_the_stale_raw() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let standin = 0x7E57_C154usize;
        let recipe = Recipe::new(move || raw);
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());

        // Deterministically interleave eviction in the real resolver's gap: it observed `raw`,
        // but has not yet taken Build.done and converted that observation into an exact pin.
        let resolved = resolve_pending(&pending, standin);
        let bound_unbuilt = pending.bound_unbuilt(standin, resolved);
        assert_eq!((resolved, bound_unbuilt), (raw, false));
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });
        assert_eq!(native.refs.load(Ordering::Relaxed), 0, "eviction did not release both unpinned owners");

        let (chosen, pin, stale_fallback) = pin_resolved(standin, &pending, resolved, bound_unbuilt);
        assert_eq!(chosen, standin, "pin validation returned the freed pre-eviction raw pointer");
        assert!(pin.is_none());
        assert!(stale_fallback, "ExecuteIndirect must treat an eviction fallback as unbuilt");
        assert_eq!(released(standin), 0);
    }

    #[test]
    fn steady_evicted_compute_resolution_remains_fail_closed_as_unbuilt() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let standin = 0x7E57_C155usize;
        let recipe = Recipe::new(move || raw);
        submit_recipe_keyed(standin, true, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });

        let ((chosen, bound_unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        assert_eq!((chosen, bound_unbuilt), (standin, true));
        assert!(pin.is_none());
        assert_eq!(released(standin), 0);
        drop(pending);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn eviction_refuses_an_active_rebuild_and_restores_resident_state_on_failure() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());

        assert!(matches!(recipe.begin_rebuild(), RebuildTurn::Leader));
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Rebuilding);
        assert_eq!(try_evict_current(&build), EvictResult::Active);
        assert_eq!(build.real.load(Ordering::Acquire), raw);
        recipe.finish_rebuild(RebuildResult::Failed);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Resident);

        let final_raw = build.real.swap(0, Ordering::AcqRel);
        drop(build);
        drop(recipe);
        release(final_raw);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn eviction_refuses_a_build_after_its_last_title_owner_releases() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let standin = 0x7E57_C153usize;
        let recipe = Recipe::new(move || raw);
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());

        let final_raw = released(standin);
        assert_eq!(try_evict_current(&pending.build), EvictResult::Abandoned);
        release(final_raw);
        drop(pending);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn failed_initial_recipe_has_no_evictable_residency() {
        let standin = 0x7E57_C156usize;
        let recipe = Recipe::new(|| 0);
        submit_recipe_keyed(standin, true, recipe.clone(), false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());

        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Failed);
        assert_eq!(try_evict_current(&pending.build), EvictResult::NotResident);
        let ((chosen, bound_unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        assert_eq!((chosen, bound_unbuilt), (standin, false), "a real build failure changed to transient-unbuilt");
        assert!(pin.is_none());
        assert_eq!(released(standin), 0);
    }

    #[test]
    fn release_during_rebuild_abandons_and_releases_the_new_raw() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let block = Arc::new((Mutex::new((false, false)), Condvar::new()));
        let build_block = block.clone();
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || {
            let mut state = build_block.0.lock().unwrap();
            if state.0 {
                state.1 = true;
                build_block.1.notify_all();
                while state.0 {
                    state = build_block.1.wait(state).unwrap();
                }
            }
            build_next.load(Ordering::Acquire)
        });
        let standin = 0x7E57_C150usize;
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());

        next.store(replacement_raw, Ordering::Release);
        block.0.lock().unwrap().0 = true;
        let rebuilding = pending.build.clone();
        let thread = std::thread::spawn(move || rebuild_once(&rebuilding));
        {
            let mut state = block.0.lock().unwrap();
            while !state.1 {
                state = block.1.wait(state).unwrap();
            }
        }
        let old_raw = released(standin);
        release(old_raw);
        {
            let mut state = block.0.lock().unwrap();
            state.0 = false;
            block.1.notify_all();
        }
        assert_eq!(thread.join().unwrap(), Some(RebuildResult::Abandoned));
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0);
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 0, "an abandoned new raw leaked");
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn async_rebuild_requests_coalesce_without_running_on_the_requester() {
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            0
        });
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe)));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        let q = BuildQueue::new();

        assert_eq!(request_rebuild_on(build.clone(), &q), RebuildRequest::Queued);
        assert_eq!(request_rebuild_on(build.clone(), &q), RebuildRequest::Coalesced);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "requesting performed driver work");
        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(build.recipe.as_ref().unwrap().0.rebuild.lock().unwrap().completed, 1);
    }

    #[test]
    fn async_rebuild_released_before_start_does_not_run_or_resurrect() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            first_raw
        });
        let standin = 0x7E57_C151usize;
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        let q = BuildQueue::new();
        assert_eq!(request_rebuild_on(pending.build.clone(), &q), RebuildRequest::Queued);

        let current = released(standin);
        release(current);
        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(calls.load(Ordering::Relaxed), 1, "orphaned queued request ran its recipe");
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0, "queued work resurrected a released build");
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn async_rebuild_failure_keeps_current_generation() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        next.store(0, Ordering::Release);
        let q = BuildQueue::new();

        assert_eq!(request_rebuild_on(build.clone(), &q), RebuildRequest::Queued);
        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(build.real.load(Ordering::Acquire), first_raw);
        assert_eq!(recipe.pin_with_residency(first_raw).0.unwrap().id(), 1);

        let final_raw = build.real.swap(0, Ordering::AcqRel);
        drop(build);
        drop(recipe);
        release(final_raw);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn async_rebuild_released_while_worker_runs_abandons_replacement() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let block = Arc::new((Mutex::new((false, false)), Condvar::new()));
        let build_block = block.clone();
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || {
            if build_next.load(Ordering::Acquire) == replacement_raw {
                let mut state = build_block.0.lock().unwrap();
                state.0 = true;
                build_block.1.notify_all();
                while !state.1 { state = build_block.1.wait(state).unwrap(); }
            }
            build_next.load(Ordering::Acquire)
        });
        let standin = 0x7E57_C152usize;
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        next.store(replacement_raw, Ordering::Release);
        let q = Arc::new(BuildQueue::new());
        assert_eq!(request_rebuild_on(pending.build.clone(), &q), RebuildRequest::Queued);
        let worker_q = q.clone();
        let worker = std::thread::spawn(move || worker_q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        {
            let mut state = block.0.lock().unwrap();
            while !state.0 { state = block.1.wait(state).unwrap(); }
        }
        let old = released(standin);
        release(old);
        block.0.lock().unwrap().1 = true;
        block.1.notify_all();
        assert!(worker.join().unwrap());
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0);
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 0);
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn async_rehydrate_coalesces_stays_unbuilt_until_worker_publication() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let standin = 0x7E57_C157usize;
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            build_next.load(Ordering::Acquire)
        });
        submit_recipe_keyed(standin, true, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        assert!(!pending.build.sent.swap(true, Ordering::AcqRel));
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        let q = BuildQueue::new();
        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Resident);
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });
        next.store(replacement_raw, Ordering::Release);

        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Queued);
        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Coalesced);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "requester performed driver work");
        let ((chosen, unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        assert_eq!((chosen, unbuilt), (standin, true));
        assert!(pin.is_none());

        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(pending.build.real.load(Ordering::Acquire), replacement_raw);
        assert_eq!(pending.build.recipe.as_ref().unwrap().0.generation.lock().unwrap().residency, Residency::Resident);
        let ((chosen, unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        assert_eq!((chosen, unbuilt), (replacement_raw, false));
        drop(pin);

        let final_raw = released(standin);
        release(final_raw);
        drop(pending);
        assert_eq!((first.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn refused_cancelled_and_failed_rehydrate_remain_retryable() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });

        let refused = BuildQueue::new();
        refused.shutdown();
        assert_eq!(request_rehydrate_on(build.clone(), &refused), RehydrateRequest::Unavailable);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Evicted);
        let cancelled = BuildQueue::new();
        assert_eq!(request_rehydrate_on(build.clone(), &cancelled), RehydrateRequest::Queued);
        cancelled.shutdown();
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Evicted);

        next.store(0, Ordering::Release);
        let failed = BuildQueue::new();
        assert_eq!(request_rehydrate_on(build.clone(), &failed), RehydrateRequest::Queued);
        assert!(failed.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Failed);
        next.store(replacement_raw, Ordering::Release);
        let retry = BuildQueue::new();
        assert_eq!(request_rehydrate_on(build.clone(), &retry), RehydrateRequest::Queued);
        assert!(retry.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(build.real.load(Ordering::Acquire), replacement_raw);

        let final_raw = build.real.swap(0, Ordering::AcqRel);
        drop(build);
        drop(recipe);
        release(final_raw);
        assert_eq!((first.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn completed_rehydrate_guard_cannot_cancel_a_newer_turn() {
        let recipe = Recipe::new(|| 0);
        recipe.mark_failed();
        let first_turn = recipe.reserve_rehydrate().unwrap();
        let old = RehydrateReservation { recipe: recipe.clone(), turn: first_turn };
        recipe.finish_rebuild(RebuildResult::Failed);

        let second_turn = recipe.reserve_rehydrate().unwrap();
        let new = RehydrateReservation { recipe: recipe.clone(), turn: second_turn };
        assert_eq!((first_turn, second_turn), (0, 1));
        drop(old);
        {
            let state = recipe.0.rebuild.lock().unwrap();
            assert!(state.active, "old guard cancelled the newer reservation");
            assert_eq!(state.completed, second_turn);
        }
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Rebuilding);

        drop(new);
        let state = recipe.0.rebuild.lock().unwrap();
        assert!(!state.active);
        assert_eq!(state.completed, second_turn + 1);
        drop(state);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Failed);
    }

    #[test]
    fn rehydrate_released_before_worker_does_not_run_or_resurrect() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let standin = 0x7E57_C158usize;
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            first_raw
        });
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });
        let q = BuildQueue::new();
        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Queued);

        assert_eq!(released(standin), 0);
        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(calls.load(Ordering::Relaxed), 1, "released rehydration ran its recipe");
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0);
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn rehydrate_released_during_worker_abandons_new_raw() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let block = Arc::new((Mutex::new((false, false)), Condvar::new()));
        let build_block = block.clone();
        let recipe = Recipe::new(move || {
            if build_calls.fetch_add(1, Ordering::Relaxed) != 0 {
                let mut state = build_block.0.lock().unwrap();
                state.0 = true;
                build_block.1.notify_all();
                while !state.1 { state = build_block.1.wait(state).unwrap(); }
                replacement_raw
            } else {
                first_raw
            }
        });
        let standin = 0x7E57_C159usize;
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });
        let q = Arc::new(BuildQueue::new());
        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Queued);
        let worker_q = q.clone();
        let worker = std::thread::spawn(move || worker_q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        {
            let mut state = block.0.lock().unwrap();
            while !state.0 { state = block.1.wait(state).unwrap(); }
        }
        assert_eq!(released(standin), 0);
        block.0.lock().unwrap().1 = true;
        block.1.notify_all();
        assert!(worker.join().unwrap());
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0);
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 0);
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn capture_preserves_lazy_for_bind_wait_and_identity_choices() {
        let standin = 0x7E57_C130usize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let real = &mut native as *mut RebuildFakePso as usize;
        let (go_tx, go_rx) = mpsc::channel();
        let (started_tx, started_rx) = mpsc::channel();
        submit_lazy(standin, true, Box::new(move || {
            started_tx.send(()).unwrap();
            go_rx.recv().unwrap();
            real
        }), 0);

        let (lazy, lazy_pin) = resolve_and_pin_mode(standin, ResolveMode::Lazy);
        assert_eq!(lazy, standin);
        assert!(lazy_pin.is_none());
        let (identity, identity_pin) = resolve_and_pin_if_enabled_identity(standin);
        assert_eq!(identity, standin);
        assert!(identity_pin.is_none());
        started_rx.recv().unwrap();
        let (result_tx, result_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| result_tx.send(resolve_and_pin_mode(standin, ResolveMode::ForBind).0).unwrap());
            assert!(result_rx.recv_timeout(std::time::Duration::from_millis(10)).is_err(), "for-bind compute resolution must still wait");
            go_tx.send(()).unwrap();
        });
        assert_eq!(result_rx.recv().unwrap(), real);
        let (waited, pin) = resolve_and_pin_mode(standin, ResolveMode::Wait);
        assert_eq!(waited, real);
        let pin = pin.expect("completed one-shot wait must own its exact generation");
        assert_eq!(pin.raw(), real);
        assert_eq!(native.refs.load(Ordering::Relaxed), 2);
        let raw_owner = released(standin);
        assert_eq!(raw_owner, real);
        release(raw_owner);
        drop(pin);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn capture_never_resolves_against_a_reused_pending_address() {
        let standin = 0x7E57_C138usize;
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let old_real = &mut native as *mut RebuildFakePso as usize;
        submit_lazy(standin, false, Box::new(move || old_real), 0);
        let old = entry(standin).unwrap();
        let job = old.build.job.lock().unwrap().take().unwrap();
        complete(&old.build, job());
        let removed = PENDING.remove(standin).unwrap();

        submit_lazy(standin, false, Box::new(|| 0x7E57_C13Ausize), 0);
        let (chosen, pin) = resolve_and_pin_pending(standin, &old, ResolveMode::Lazy);
        assert_eq!(chosen, old_real, "selection must use the retained Pending, not its reused map slot");
        let pin = pin.expect("completed one-shot selection must own its exact generation");
        assert_eq!(pin.raw(), old_real);
        assert_eq!(native.refs.load(Ordering::Relaxed), 2);

        assert_eq!(released(standin), 0);
        PENDING.insert(standin, removed.clone());
        let raw_owner = released(standin);
        assert_eq!(raw_owner, old_real);
        release(raw_owner);
        drop(pin);
        drop(removed);
        drop(old);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn bound_wait_capture_pins_the_exact_published_generation() {
        #[repr(C)]
        struct FakePso { vtable: *const usize, refs: AtomicUsize }
        unsafe extern "system" fn add_ref(this: *mut std::ffi::c_void) -> u32 {
            (*(this as *mut FakePso)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }
        unsafe extern "system" fn release(this: *mut std::ffi::c_void) -> u32 {
            (*(this as *mut FakePso)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }

        let vtable = [0usize, add_ref as usize, release as usize];
        let mut fake = FakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut fake as *mut FakePso as usize;
        let standin = 0x7E57_C140usize;
        let recipe = Recipe::new(move || raw);
        submit_recipe_keyed(standin, true, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let job = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, job());

        let ((chosen, bound_unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        let pin = pin.expect("published recipe generation must be captured");
        assert_eq!((chosen, bound_unbuilt, pin.raw(), pin.id()), (raw, false, raw, 1));
        assert_eq!(fake.refs.load(Ordering::Relaxed), 2, "generation pin shares the recipe's exact COM owner");

        drop(pin);
        assert_eq!(released(standin), raw);
        drop(pending);
        assert_eq!(unsafe { release(raw as *mut std::ffi::c_void) }, 0);
    }

    #[test]
    fn a_panicking_pipeline_job_completes_failed_and_wakes_its_waiter() {
        let build = Arc::new(Build::new(Box::new(|| panic!("simulated pipeline panic")), false));
        let waiting = build.clone();
        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let (done, _) = waiting
                .cv
                .wait_timeout_while(waiting.done.lock().unwrap(), std::time::Duration::from_secs(5), |done| !*done)
                .unwrap();
            tx.send((*done, waiting.real.load(Ordering::Acquire))).unwrap();
        });

        let job = build.job.lock().unwrap().take().unwrap();
        run_build(&build, job);
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(5)), Ok((true, 0)));
        waiter.join().unwrap();
        assert!(BUILDING.with(|building| building.borrow().is_none()), "panicking build left worker-local demand installed");

        let next = Arc::new(Build::new(Box::new(|| 0x5151), false));
        let job = next.job.lock().unwrap().take().unwrap();
        run_build(&next, job);
        assert_eq!(next.real.load(Ordering::Acquire), 0x5151, "the worker path did not survive a prior panic");

        let refused = Arc::new(Build::new(Box::new(|| 0x6161), false));
        fail_unrunnable_build(&refused);
        assert!(*refused.done.lock().unwrap(), "worker-start/submission failure did not complete the build");
        assert!(refused.job.lock().unwrap().is_none());
        assert_eq!(refused.real.load(Ordering::Acquire), 0);

        assert_eq!(
            build_pieces(vec![Box::new(|| 0x11), Box::new(|| panic!("simulated piece panic")), Box::new(|| 0x33)]),
            [0x11, 0, 0x33],
            "one failed split piece stranded or discarded its sibling results"
        );
    }

    #[test]
    fn pending_registry_routes_exact_lifetimes_across_independent_shards() {
        let registry = Arc::new(PendingRegistry::new());
        let keys: Vec<_> = (0..256).map(|i| 0x7E00_0000usize + i * 0x10).collect();
        let used: std::collections::HashSet<_> = keys.iter().map(|&key| pending_shard_index(key)).collect();
        assert!(used.len() >= PENDING_SHARDS / 2, "sequential aligned handles reached only {} shards", used.len());

        std::thread::scope(|scope| {
            for chunk in keys.chunks(32) {
                let registry = registry.clone();
                scope.spawn(move || {
                    for &key in chunk {
                        assert_eq!(registry.insert(key, key ^ 0x55AA), None);
                        assert_eq!(registry.get(key), Some(key ^ 0x55AA));
                        assert_eq!(registry.remove(key), Some(key ^ 0x55AA));
                        assert_eq!(registry.get(key), None);
                    }
                });
            }
        });

        let reused = 0x7E57_F000usize;
        assert_eq!(registry.insert(reused, 1), None);
        assert_eq!(registry.insert(reused, 2), Some(1), "replacement did not return the exact old address entry");
        assert_eq!(registry.remove(reused), Some(2));
        assert_eq!(registry.remove(reused), None);
    }

    #[test]
    fn a_busy_pending_shard_does_not_hold_an_unrelated_lookup() {
        let registry = Arc::new(PendingRegistry::new());
        let first = 0x7E10_0000usize;
        let second = (first + 0x10..).step_by(0x10).find(|&key| pending_shard_index(key) != pending_shard_index(first)).unwrap();
        registry.insert(second, 7usize);

        let held = registry.shards[pending_shard_index(first)].lock().unwrap();
        let (tx, rx) = mpsc::channel();
        let other = registry.clone();
        let thread = std::thread::spawn(move || tx.send(other.get(second)).unwrap());
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(2)), Ok(Some(7)));
        drop(held);
        thread.join().unwrap();
    }

    #[test]
    fn queued_count_state_is_once_only_and_completion_wins_races() {
        let state = AtomicU8::new(QUEUE_IDLE);
        assert!(mark_queued(&state));
        assert!(!mark_queued(&state));
        assert!(mark_queue_done(&state));
        assert!(!mark_queue_done(&state));
        assert!(!mark_queued(&state));

        let completed_first = AtomicU8::new(QUEUE_IDLE);
        assert!(!mark_queue_done(&completed_first));
        assert!(!mark_queued(&completed_first));

        let build = Build::new(Box::new(|| 0), false);
        build.note_queued();
        assert_eq!(build.queue_state.load(Ordering::Relaxed), QUEUE_IDLE, "auxiliary builds are not reported as title pipelines");
        build.tracked.store(true, Ordering::Release);
        build.note_queued();
        assert_eq!(build.queue_state.load(Ordering::Relaxed), QUEUE_ACTIVE);
        build.finish_queued();
        assert_eq!(build.queue_state.load(Ordering::Relaxed), QUEUE_DONE);
    }

    #[test]
    fn demanded_auxiliary_build_runs_on_a_worker() {
        let caller = std::thread::current().id();
        assert_eq!(build_foreground(Box::new(move || {
            assert_ne!(std::thread::current().id(), caller);
            0x7E57_C0DE
        })), 0x7E57_C0DE);
    }

    #[test]
    fn coalescing_counter_math_tracks_lifetime_current_and_peak() {
        let counters = CoalescingCounters::new();
        counters.unique();
        counters.hit();
        counters.unique();
        counters.finished(1);
        assert_eq!(
            counters.snapshot(),
            ComputeCoalescingStats {
                keyed_submissions: 3,
                coalescing_hits: 1,
                unique_keyed_builds: 2,
                current_in_flight: 1,
                peak_in_flight: 2,
            }
        );
        counters.finished(1);
        assert_eq!(counters.snapshot().current_in_flight, 0);
        assert_eq!(counters.snapshot().peak_in_flight, 2);
    }

    #[test]
    fn retention_age_counts_unique_reals_from_completion_or_last_use() {
        let stats = retention_stats_from(1_000, [
            (1, 0x101, 200, 300),
            (1, 0x101, 200, 300), // another stand-in sharing build 1
            (2, 0x202, 100, 995),
            (3, 0, 0, 0),        // failed or incomplete
            (4, 0x404, 0, 0),
        ]);
        assert_eq!(stats, RetentionStats {
            retained_reals: 3,
            unused_60_frames: 2,
            unused_600_frames: 2,
            oldest_unused_frames: 1_000,
        });
    }

    fn compute_key(tag: u8) -> super::super::real::compute_build_key::ComputeBuildKey {
        super::super::real::compute_build_key::ComputeBuildKey::capture(
            1,
            &windows::Win32::Graphics::Direct3D12::D3D12_COMPUTE_PIPELINE_STATE_DESC::default(),
            [tag].into(),
            false,
        )
    }

    fn compute_key_in_another_shard(key: &ComputeBuildKey) -> ComputeBuildKey {
        (0..=u8::MAX)
            .map(compute_key)
            .find(|candidate| in_flight_shard_index(candidate) != in_flight_shard_index(key))
            .expect("test keys did not cover another in-flight shard")
    }

    #[test]
    fn unrelated_in_flight_shard_does_not_block() {
        let registry = Arc::new(InFlightRegistry::new());
        let first = compute_key(0);
        let second = compute_key_in_another_shard(&first);
        let held = registry.shard(&first).lock().unwrap();
        let (tx, rx) = mpsc::channel();
        let other_registry = registry.clone();
        let thread = std::thread::spawn(move || {
            let _other = other_registry.shard(&second).lock().unwrap();
            tx.send(()).unwrap();
        });

        rx.recv_timeout(std::time::Duration::from_secs(1)).expect("unrelated key waited for the held shard");
        drop(held);
        thread.join().unwrap();
    }

    #[test]
    fn old_in_flight_completion_cannot_remove_replacement() {
        let registry = InFlightRegistry::new();
        let key = Arc::new(compute_key(0x8A));
        let old = Arc::new(Build::new_keyed(Box::new(|| 0), false, Some(key.clone()), None));
        let replacement = Arc::new(Build::new_keyed(Box::new(|| 0), false, Some(key.clone()), None));
        let old_weak = Arc::downgrade(&old);
        let replacement_weak = Arc::downgrade(&replacement);
        let displaced = {
            let mut shard = registry.shard(&key).lock().unwrap();
            let builds = shard.get_or_insert_with(HashMap::new);
            builds.insert(key.clone(), old_weak.clone());
            builds.insert(key.clone(), replacement_weak.clone())
        };
        drop(displaced);

        assert!(!registry.remove_exact(&key, &old_weak, || {}));
        assert!(registry.shard(&key).lock().unwrap().as_ref().unwrap().get(&key).and_then(Weak::upgrade)
            .is_some_and(|registered| Arc::ptr_eq(&registered, &replacement)));
        assert!(registry.remove_exact(&key, &replacement_weak, || {}));
        assert!(!registry.remove_exact(&key, &replacement_weak, || {}), "one registry entry is finished only once");
    }

    #[test]
    fn terminal_retirement_precedes_other_shard_unique_count() {
        let registry = Arc::new(InFlightRegistry::new());
        let counters = Arc::new(CoalescingCounters::new());
        let key = Arc::new(compute_key(0x8B));
        let other_key = Arc::new(compute_key_in_another_shard(&key));
        let old_weak = {
            let old = Arc::new(Build::new_keyed(Box::new(|| 0), false, Some(key.clone()), None));
            Arc::downgrade(&old)
        };
        registry.shard(&key).lock().unwrap().get_or_insert_with(HashMap::new)
            .insert(key.clone(), old_weak.clone());
        counters.unique();

        let replacement = Arc::new(Build::new_keyed(Box::new(|| 0), false, Some(other_key.clone()), None));
        let replacement_weak = Arc::downgrade(&replacement);
        assert!(registry.remove_exact(&key, &old_weak, || {
            counters.finished(1);
            assert_eq!(counters.snapshot().current_in_flight, 0);
            let (replacement_registry, replacement_counters) = (registry.clone(), counters.clone());
            let replacement_thread = std::thread::spawn(move || {
                let displaced = {
                    let mut shard = replacement_registry.shard(&other_key).lock().unwrap();
                    let displaced = shard.get_or_insert_with(HashMap::new).insert(other_key, replacement_weak);
                    replacement_counters.unique();
                    displaced
                };
                drop(displaced);
            });
            replacement_thread.join().expect("other-shard replacement was blocked by terminal retirement");
            assert_eq!(counters.snapshot().current_in_flight, 1);
        }));
        assert_eq!(counters.snapshot(), ComputeCoalescingStats {
            keyed_submissions: 2,
            coalescing_hits: 0,
            unique_keyed_builds: 2,
            current_in_flight: 1,
            peak_in_flight: 1,
        });
        drop(replacement);
    }

    fn submit_compute(standin: usize, tag: u8, job: Job) {
        submit_lazy_keyed(standin, true, job, 0, Some(compute_key(tag)));
    }

    #[test]
    fn coalesced_compute_owners_retain_the_winning_recipe() {
        let standins = [0x7E57_C091usize, 0x7E57_C092];
        let winner = Recipe::new(|| 0xC091);
        let duplicate = Recipe::new(|| panic!("duplicate recipe ran"));
        submit_recipe_keyed(standins[0], true, winner, false, 0, Some(compute_key(0x91)));
        submit_recipe_keyed(standins[1], true, duplicate, false, 0, Some(compute_key(0x91)));

        let (a, b) = (entry(standins[0]).unwrap(), entry(standins[1]).unwrap());
        assert!(Arc::ptr_eq(&a.build, &b.build));
        assert_eq!(a.build.recipe.as_ref().unwrap().run(), 0xC091);
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0);
    }

    #[test]
    fn concurrent_equal_compute_submissions_run_one_job() {
        let standins = [0x7E57_C101usize, 0x7E57_C102];
        let gate = Arc::new(std::sync::Barrier::new(3));
        let runs = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = standins.iter().map(|&standin| {
            let (gate, runs) = (gate.clone(), runs.clone());
            std::thread::spawn(move || {
                gate.wait();
                submit_compute(standin, 0xC1, Box::new(move || {
                    runs.fetch_add(1, Ordering::Relaxed);
                    0xC101
                }));
            })
        }).collect();
        gate.wait();
        for thread in threads { thread.join().unwrap() }

        let (a, b) = (entry(standins[0]).unwrap(), entry(standins[1]).unwrap());
        assert!(Arc::ptr_eq(&a.build, &b.build));
        assert_eq!(resolve_wait_for(standins[0], None), 0xC101);
        assert_eq!(resolve_wait_for(standins[1], None), 0xC101);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(released(standins[0]), 0, "the shared COM owner stays with the other stand-in");
        assert_eq!(released(standins[1]), 0xC101, "the last stand-in releases the one owned reference");
    }

    #[test]
    fn bound_state_follows_the_choice_from_a_shared_build() {
        let standins = [0x7E57_C111usize, 0x7E57_C112];
        let drivers = [0x7E57_D111usize, 0x7E57_D112];
        let (started_tx, started_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel::<()>();
        submit_compute(standins[0], 0xE6, Box::new(move || {
            started_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
            0xC111
        }));
        submit_compute(standins[1], 0xE6, Box::new(|| panic!("duplicate shared build ran")));
        for (&standin, &driver) in standins.iter().zip(&drivers) {
            shared(standin, driver);
            image_only(standin);
        }

        let before = standins.map(|standin| resolve_wait_bound_for(standin, None));
        assert_eq!(before, [(drivers[0], true), (drivers[1], true)]);
        started_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker did not start the build");
        finish_tx.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(standins[0]) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(standins[0]), "shared build did not finish");
        for ((&standin, &driver), &(bound, _)) in standins.iter().zip(&drivers).zip(&before) {
            let pending = entry(standin).unwrap();
            assert!(pending.bound_unbuilt(standin, bound), "the earlier driver stand-in stays unbuilt");
            assert_eq!(resolve_wait_bound_for(standin, None), (0xC111, false));
            assert_ne!(driver, 0xC111);
        }
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0xC111);
    }

    #[test]
    fn a_failed_builds_permanent_stand_in_is_not_unbuilt() {
        let standin = 0x7E57_C121usize;
        let driver = 0x7E57_D121usize;
        submit_lazy(standin, true, Box::new(|| 0), 0);
        shared(standin, driver);
        assert_eq!(resolve_wait_bound_for(standin, None), (driver, false));
        assert_eq!(resolve_wait_bound_for(0x7E57_C122, None), (0x7E57_C122, false));
        assert_eq!(released(standin), 0);
    }

    #[test]
    fn no_budget_wait_runs_a_shared_build_on_a_wanted_worker() {
        let standins = [0x7E57_C701usize, 0x7E57_C702];
        let (started_tx, started_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel::<()>();
        submit_compute(standins[0], 0xC7, Box::new(move || {
            started_tx.send((std::thread::current().name().map(str::to_owned), build_is_wanted(), wanted_here()())).unwrap();
            let _ = finish_rx.recv();
            0xC701
        }));
        submit_compute(standins[1], 0xC7, Box::new(|| panic!("duplicate shared build ran")));
        let (a, b) = (entry(standins[0]).unwrap(), entry(standins[1]).unwrap());
        assert!(Arc::ptr_eq(&a.build, &b.build), "equal compute requests share one Build");

        let (result_tx, result_rx) = mpsc::channel();
        let waiters: Vec<_> = standins.iter().map(|&standin| {
            let result_tx = result_tx.clone();
            std::thread::spawn(move || result_tx.send((standin, resolve_wait_for(standin, None))).unwrap())
        }).collect();
        drop(result_tx);
        let (thread, wanted, wanted_probe) = started_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker did not start the build");
        assert!(thread.as_deref().is_some_and(|name| name.starts_with("xcl-pso-")), "build ran on caller {thread:?}");
        assert!(wanted && wanted_probe, "the worker did not observe foreground demand");
        assert!(matches!(result_rx.try_recv(), Err(mpsc::TryRecvError::Empty)), "a caller returned before the build finished");

        finish_tx.send(()).unwrap();
        let mut results: Vec<_> = result_rx.iter().collect();
        results.sort_unstable();
        for waiter in waiters { waiter.join().unwrap(); }
        assert_eq!(results, [(standins[0], 0xC701), (standins[1], 0xC701)]);
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0xC701);
    }

    #[test]
    fn unequal_compute_keys_keep_separate_builds() {
        let standins = [0x7E57_C201usize, 0x7E57_C202];
        submit_compute(standins[0], 0xC2, Box::new(|| 0));
        submit_compute(standins[1], 0xC3, Box::new(|| 0));
        assert!(!Arc::ptr_eq(&entry(standins[0]).unwrap().build, &entry(standins[1]).unwrap().build));
        assert_eq!((released(standins[0]), released(standins[1])), (0, 0));
    }

    #[test]
    fn failed_coalesced_build_wakes_every_waiter() {
        let standins = [0x7E57_C301usize, 0x7E57_C302];
        let runs = Arc::new(AtomicUsize::new(0));
        for &standin in &standins {
            let runs = runs.clone();
            submit_compute(standin, 0xC4, Box::new(move || {
                runs.fetch_add(1, Ordering::Relaxed);
                0
            }));
        }
        let threads: Vec<_> = standins.iter().map(|&standin| {
            std::thread::spawn(move || resolve_wait_for(standin, None))
        }).collect();
        for (thread, standin) in threads.into_iter().zip(standins) {
            assert_eq!(thread.join().unwrap(), standin);
        }
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!((released(standins[0]), released(standins[1])), (0, 0));
    }

    #[test]
    fn coalesced_preload_is_promoted_once_for_both_pending_owners() {
        let standins = [0x7E57_C401usize, 0x7E57_C402];
        submit_compute(standins[0], 0xC5, Box::new(|| 0xC401));
        submit_compute(standins[1], 0xC5, Box::new(|| panic!("duplicate job ran")));
        let build = entry(standins[0]).unwrap().build.clone();
        assert!(Arc::ptr_eq(&build, &entry(standins[1]).unwrap().build));
        let q = BuildQueue::new();
        enqueue_on(&q, &build, Priority::Preload, true);
        entry(standins[1]).unwrap().build.promote(Priority::Foreground);
        assert!(!q.try_run_one_between(Priority::Speculative, Priority::Preload));
        assert!(q.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        assert_eq!(build.real.load(Ordering::Acquire), 0xC401);
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0xC401);
    }

    #[test]
    fn submit_demand_is_shared_by_coalesced_stand_ins() {
        let standins = [0x7E57_C411usize, 0x7E57_C412];
        let (wanted_tx, wanted_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel();
        submit_compute(standins[0], 0xD7, Box::new(move || {
            wanted_tx.send(build_is_wanted()).unwrap();
            go_rx.recv().unwrap();
            0xC411
        }));
        submit_compute(standins[1], 0xD7, Box::new(|| panic!("duplicate job ran")));
        let build = entry(standins[0]).unwrap().build.clone();
        assert!(Arc::ptr_eq(&build, &entry(standins[1]).unwrap().build));

        need_by_submit(standins[1]);
        assert!(build.wanted.load(Ordering::Acquire));
        assert!(build.by_submit.load(Ordering::Acquire), "coalesced owners did not share submit demand");
        assert!(wanted_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it"));
        go_tx.send(()).unwrap();
        assert_eq!(resolve_wait_for(standins[0], None), 0xC411);
        assert_eq!(resolve_wait_for(standins[1], None), 0xC411);
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0xC411);
    }

    #[test]
    fn releasing_one_waiter_does_not_orphan_shared_build() {
        let standins = [0x7E57_C501usize, 0x7E57_C502];
        let runs = Arc::new(AtomicUsize::new(0));
        let count = runs.clone();
        submit_compute(standins[0], 0xC6, Box::new(move || {
            count.fetch_add(1, Ordering::Relaxed);
            0xC501
        }));
        submit_compute(standins[1], 0xC6, Box::new(|| panic!("duplicate job ran")));
        let build = entry(standins[1]).unwrap().build.clone();
        assert_eq!(released(standins[0]), 0);
        assert!(!build.orphan.load(Ordering::Acquire));
        assert!(build.job.lock().unwrap().is_some());
        assert_eq!(resolve_wait_for(standins[1], None), 0xC501);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(released(standins[1]), 0xC501);
    }

    #[test]
    fn build_completion_reaches_all_owners() {
        let build = Arc::new(Build::new(Box::new(|| 0), false));
        let waiter_build = build.clone();
        assert!(Arc::ptr_eq(&build, &waiter_build));
        let waiter = std::thread::spawn(move || {
            let mut done = waiter_build.done.lock().unwrap();
            while !*done {
                done = waiter_build.cv.wait(done).unwrap();
            }
            waiter_build.real.load(Ordering::Acquire)
        });

        complete(&build, 0xB017_D001);
        assert_eq!(waiter.join().unwrap(), 0xB017_D001);
        assert_eq!(build.real.swap(0, Ordering::AcqRel), 0xB017_D001);
    }

    #[test]
    fn wanted_preload_moves_to_foreground_and_builds_once() {
        let q = BuildQueue::new();
        let runs = Arc::new(AtomicUsize::new(0));
        let count = runs.clone();
        let build = Arc::new(Build::new(Box::new(move || {
            count.fetch_add(1, Ordering::Relaxed);
            0xB017_D002
        }), true));
        enqueue_on(&q, &build, Priority::Preload, true);

        build.want();
        assert!(!q.try_run_one_between(Priority::Speculative, Priority::Preload));
        assert!(q.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        assert!(!q.try_run_one());
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(build.real.swap(0, Ordering::AcqRel), 0xB017_D002);
    }

    #[test]
    fn caller_can_steal_a_queued_build_without_running_it_twice() {
        let q = BuildQueue::new();
        let runs = Arc::new(AtomicUsize::new(0));
        let count = runs.clone();
        let build = Arc::new(Build::new(Box::new(move || {
            count.fetch_add(1, Ordering::Relaxed);
            0xB017_D003
        }), true));
        enqueue_on(&q, &build, Priority::BoundSoon, false);

        let job = build.job.lock().unwrap().take().unwrap();
        build.cancel_queued();
        complete(&build, job());
        assert!(!q.try_run_one());
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(build.real.swap(0, Ordering::AcqRel), 0xB017_D003);
    }

    /// A PC call that has kept one of the title's threads for long holds the title, a short one
    /// does not, and the hold ends with the call.
    #[test]
    fn a_long_pc_call_holds_the_title() {
        let mut calls = PcCalls::new();
        let began = std::time::Instant::now();
        calls.begin(1, 0, began);
        calls.begin(2, 0, began + 2 * PC_CALL_HOLDS_AFTER);
        assert!(!calls.holds(0, began), "a call that has just begun holds the title");
        let later = began + 2 * PC_CALL_HOLDS_AFTER;
        assert!(calls.holds(0, later), "a PC call has lasted {:?}, but the title is not held", 2 * PC_CALL_HOLDS_AFTER);
        calls.end(1, 0, later);
        calls.end(2, 0, later);
        assert!(!calls.holds(0, later), "the calls are over, the title is still held");
    }

    /// Short PC calls add up between two frames: none of them is long, but after many in a row
    /// the one going on holds the title. The title's own time between them is not held, and a
    /// frame shown ends it.
    #[test]
    fn short_pc_calls_between_two_frames_hold_the_title() {
        let mut calls = PcCalls::new();
        let step = PC_CALL_HOLDS_AFTER / 10;
        let mut now = std::time::Instant::now();
        for id in 0..20 {
            calls.begin(id, 7, now);
            assert!(id >= 9 || !calls.holds(7, now + step), "{id} short calls hold the title already");
            now += step;
            calls.end(id, 7, now);
        }
        assert!(!calls.holds(7, now), "no PC call is going on, the title is held");
        calls.begin(20, 7, now);
        assert!(calls.holds(7, now), "twenty short PC calls since the last frame and one going on, but the title is not held");
        assert!(!calls.holds(8, now), "a frame was shown, the calls before it still hold the title");
        calls.end(20, 8, now);
    }

    /// While a bind waits for a pipeline still being built, the title is held (`title_held`):
    /// `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = started.send(());
            let _ = finish.recv();
            0x5252
        }), 0);
        let bind = std::thread::spawn(move || resolve_wait_for(standin, None));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        // (built in the bind, or by a worker with the bind about to wait for it)
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    /// A stand-in that is no driver object (`shared`) never reaches the driver: binds get the
    /// driver pipeline shared with other stand-ins until the real one is built, and after a build
    /// that failed.
    #[test]
    fn a_bind_of_a_shared_stand_in_gets_its_driver_pipeline() {
        let (fails, builds, driver) = (0x7E57_B011usize, 0x7E57_B012usize, 0x7E57_B013usize);
        submit_lazy(fails, false, Box::new(|| 0), 0);
        submit_lazy(builds, false, Box::new(|| 0x1234), 0);
        for s in [fails, builds] {
            shared(s, driver);
            assert_eq!(entry(s).unwrap().stand(s), driver);
        }
        assert_eq!(resolve_wait_for(fails, None), driver, "its build failed");
        assert_eq!(resolve(fails), driver);
        assert_eq!(resolve_wait_for(builds, None), 0x1234);
        assert_eq!(resolve_for_bind(builds), 0x1234);
        // (0x1234 is no pipeline: taken, not released)
        assert_eq!((released(fails), released(builds)), (0, 0x1234));
    }

    /// A bind does not wait for a compute pipeline that writes only images: it gets the stand-in
    /// at once, with no budget, the build goes to the workers
    /// unwanted, and once built the bind gets the real one. Without `image_only` the same bind
    /// waits until the pipeline is built.
    #[test]
    fn a_bind_does_not_wait_for_a_pipeline_that_writes_only_images() {
        let (spared, waited) = (0x7E57_B0A1usize, 0x7E57_B0A2usize);
        let (go, finish) = mpsc::channel::<()>();
        let (tx, rx) = mpsc::channel();
        submit_lazy(spared, true, Box::new(move || {
            let _ = tx.send(wanted_here()());
            let _ = finish.recv();
            0x5A5A
        }), 0);
        image_only(spared);
        // (on a thread: a bind that waits would wait for `go`, below)
        let (bound_tx, bound) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = bound_tx.send((resolve_wait_for(spared, None), resolve_wait_for(spared, None)));
        });
        let binds = bound.recv_timeout(std::time::Duration::from_secs(5));
        if binds.is_err() {
            let _ = go.send(());
        }
        assert_eq!(binds.expect("the binds waited for the build"), (spared, spared), "still being built: the stand-in");
        assert!(is_pending(spared));
        assert!(!rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it"), "nobody waits: the build is not wanted");
        go.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(spared) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(spared, None), 0x5A5A, "built by a worker");
        // the same build without the mark: the bind waits for it
        submit_lazy(waited, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            0x5B5B
        }), 0);
        assert_eq!(resolve_wait_for(waited, None), 0x5B5B);
        let _ = released(spared);
        let _ = released(waited);
    }

    #[test]
    fn a_pipeline_a_submit_needs_is_built_before_the_queue() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let backlog = 8 * crate::gpu::recomp::dxc::compile_threads();
        let standins: Vec<usize> = (0..=backlog).map(|i| 0x7E57_E000usize + i).collect();
        for &standin in &standins {
            let order = order.clone();
            submit(standin, true, Box::new(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                order.lock().unwrap().push(standin);
                0x6161
            }));
        }
        let last = *standins.last().unwrap();
        need_by_submit(last);
        let start = std::time::Instant::now();
        while standins.iter().any(|&s| is_pending(s)) && start.elapsed() < std::time::Duration::from_secs(30) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let order = order.lock().unwrap();
        assert_eq!(order.len(), standins.len());
        let at = order.iter().position(|&s| s == last).unwrap();
        assert!(at < order.len() / 2, "made last and needed by a submit, it was built {at} of {}", order.len());
        for &standin in &standins {
            let _ = released(standin);
        }
    }

    #[test]
    fn a_build_a_submit_needs_is_wanted_from_record_time() {
        let standin = 0x7E57_B006usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if build_is_wanted() { 0x5757 } else { 0 }
        }));
        started.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker did not start the build");
        need_by_submit(standin);
        assert_eq!(resolve_wait_for(standin, None), 0x5757, "record-time demand did not reach the running build");
        assert_eq!(released(standin), 0x5757);
    }

    #[test]
    fn binding_a_completed_compute_pipeline_neither_relooks_it_up_nor_requests_work() {
        let standin = 0x7E57_B00Ausize;
        submit(standin, true, Box::new(|| 0x5455));
        let p = entry(standin).unwrap();
        drop(p.build.cv.wait_while(p.build.done.lock().unwrap(), |done| !*done).unwrap());

        assert_eq!(resolve_for_bind(standin), 0x5455);
        assert_eq!(resolve_wait_for(standin, None), 0x5455);
        assert!(!p.build.wanted.load(Ordering::Acquire));
        assert_eq!(released(standin), 0x5455);
    }

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    /// DXC admission probes demand while holding its own queue lock: submit demand is published
    /// through the atomic wanted signal, so the probe cannot take this build's demand lock.
    #[test]
    fn submit_demand_wanted_probe_is_lock_free() {
        let build = Arc::new(Build::new(Box::new(|| 0), true));
        let pending = structured_pending(build.clone(), true);
        let before = BUILDING.with(|building| building.replace(Some(build.clone())));
        let _restore = RestoreBuilding(before);
        let wanted = wanted_here();
        need_pending_by_submit(&pending);
        assert!(wanted(), "submit demand reaches DXC and driver admission");

        let demand = build.demand.lock().unwrap();
        let (tx, rx) = mpsc::channel();
        let thread = std::thread::spawn(move || tx.send(wanted()).unwrap());
        assert!(rx.recv_timeout(std::time::Duration::from_secs(2)).expect("wanted probe took the demand lock"));
        drop(demand);
        thread.join().unwrap();
    }

    #[test]
    fn submit_demand_publishes_after_work_and_repeats_without_build_locks() {
        let build = Arc::new(Build::new(Box::new(|| 0), true));
        let pending = structured_pending(build.clone(), true);
        let queued = build.queued.lock().unwrap();
        let (first_tx, first_rx) = mpsc::channel();
        let first_pending = pending.clone();
        let first = std::thread::spawn(move || {
            need_pending_by_submit(&first_pending);
            first_tx.send(()).unwrap();
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !build.wanted.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(build.wanted.load(Ordering::Acquire), "first demand did not become wanted");
        assert!(!build.by_submit.load(Ordering::Acquire), "demand published before promotion completed");
        assert!(matches!(first_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        drop(queued);
        first_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("first demand did not complete");
        first.join().unwrap();
        assert!(build.by_submit.load(Ordering::Acquire));

        let queued = build.queued.lock().unwrap();
        let demand = build.demand.lock().unwrap();
        let (repeat_tx, repeat_rx) = mpsc::channel();
        let repeat = std::thread::spawn(move || {
            need_pending_by_submit(&pending);
            repeat_tx.send(()).unwrap();
        });
        repeat_rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("published demand took the queued or preload lock");
        drop(demand);
        drop(queued);
        repeat.join().unwrap();
    }

    #[test]
    fn concurrent_first_submit_demand_calls_publish_one_shared_state() {
        let runs = Arc::new(AtomicUsize::new(0));
        let job_runs = runs.clone();
        let build = Arc::new(Build::new(Box::new(move || {
            job_runs.fetch_add(1, Ordering::Relaxed);
            0
        }), false));
        let pending = structured_pending(build.clone(), true);
        let gate = Arc::new(std::sync::Barrier::new(9));
        let callers: Vec<_> = (0..8).map(|_| {
            let (pending, gate) = (pending.clone(), gate.clone());
            std::thread::spawn(move || {
                gate.wait();
                need_pending_by_submit(&pending);
            })
        }).collect();
        gate.wait();
        for caller in callers {
            caller.join().unwrap();
        }
        assert!(build.wanted.load(Ordering::Acquire));
        assert!(build.by_submit.load(Ordering::Acquire));
        assert!(build.sent.load(Ordering::Acquire));
        let (done, timeout) = build.cv.wait_timeout_while(
            build.done.lock().unwrap(),
            std::time::Duration::from_secs(10),
            |done| !*done,
        ).unwrap();
        assert!(!timeout.timed_out() && *done, "concurrent demand did not complete its build");
        assert_eq!(runs.load(Ordering::Relaxed), 1, "concurrent first callers executed more than one job");
    }

    /// A compile a worker's build starts is wanted once a bind waits for the build – asked from
    /// another thread, as the DXC workers do; a thread that builds nothing (no title runs here)
    /// wants nothing.
    #[test]
    fn a_waiting_bind_makes_the_builds_compiles_wanted() {
        let standin = 0x7E57_B005usize;
        let (tx, rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            tx.send(wanted_here()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x5454
        }));
        let wanted = rx.recv().unwrap();
        assert!(!wanted(), "nobody waits yet");
        assert_eq!(resolve_wait_for(standin, None), 0x5454);
        assert!(wanted());
        assert!(!wanted_here()());
        let _ = released(standin);
    }

    /// A worker's build hears that a bind waits for it (and only its own build does).
    #[test]
    fn a_waiting_bind_makes_the_build_wanted() {
        let standin = 0x7E57_B004usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            let unwanted_first = !build_is_wanted();
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if unwanted_first && build_is_wanted() { 0x5353 } else { 0 }
        }));
        started.recv().unwrap();
        assert!(!build_is_wanted(), "this thread builds nothing");
        assert_eq!(resolve_wait_for(standin, None), 0x5353, "the bind's wait reached the build");
        let _ = released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    #[test]
    fn used_file_ranks_follow_unique_valid_first_occurrences() {
        let ranks = used::parse_ranked("10\nnot a key\n10\n0\n 20 \ninvalid\n30\n20\n");
        assert_eq!(ranks.len(), 3);
        assert_eq!(ranks.get(&0x10), Some(&0));
        assert_eq!(ranks.get(&0x20), Some(&1));
        assert_eq!(ranks.get(&0x30), Some(&2));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// not kept as used until its first use (the backlog may build it ahead of that: `send_ahead`),
    /// and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(!used::noted(unknown_key), "nobody used it and no earlier run did");
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    /// What a pipeline's first use tells (a graphics one's description: `gfx_records`) is told
    /// once, at that use, not when it is made or built.
    #[test]
    fn a_first_use_is_told_once() {
        let (standin, key) = (0x7E57_7011usize, 0x5EED_0011u64);
        let told = Arc::new(AtomicUsize::new(0));
        submit_lazy(standin, true, Box::new(|| 0x4331), key);
        let t = told.clone();
        on_first_use(standin, Box::new(move || { t.fetch_add(1, Ordering::SeqCst); }));
        assert_eq!(told.load(Ordering::SeqCst), 0, "not at its making");
        assert_eq!(resolve_wait(standin), 0x4331);
        assert_eq!(told.load(Ordering::SeqCst), 1);
        assert_eq!(resolve(standin), 0x4331);
        assert_eq!(resolve_wait(standin), 0x4331);
        assert_eq!(told.load(Ordering::SeqCst), 1, "once");
        on_first_use(0x7E57_7012, Box::new(|| unreachable!("no such pipeline")));
        released(standin);
    }

    #[test]
    fn graphics_first_use_carries_the_exact_serialized_recipe_identity() {
        let standin = 0x7E57_7013usize;
        let payload = b"canonical graphics recipe".to_vec();
        submit_lazy(standin, false, Box::new(|| 0x4332), 0x5EED_0013);
        on_first_use_graphics(standin, payload.clone());
        let pending = entry(standin).unwrap();
        assert_eq!(pending.manifest_recipe.load(Ordering::Acquire), super::super::gfx_records::payload_key(&payload));
        assert!(pending.on_use.lock().unwrap().is_some());
        released(standin);
    }

    static WORKER_SATURATION_LOCK: Mutex<()> = Mutex::new(());

    /// Normal workers parked at the huge-driver-compile gate leave their priority band live, and
    /// the bounded replacement workers retire after the gate opens.
    #[test]
    fn parked_normal_workers_leave_the_queue_to_bounded_helpers() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::time::{Duration, Instant};

        struct OpenGate(Arc<(Mutex<bool>, Condvar)>);
        impl Drop for OpenGate {
            fn drop(&mut self) {
                *self.0 .0.lock().unwrap() = true;
                self.0 .1.notify_all();
            }
        }

        let outside = parked_for_big_compile();
        assert!(!outside.0, "a title/test thread must not start a replacement");
        drop(outside);

        start_normal_workers();
        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open = OpenGate(gate.clone());
        for _ in 0..n {
            let (tx, gate) = (started_tx.clone(), gate.clone());
            queue().submit(Priority::Foreground, Box::new(move || {
                let _parked = parked_for_big_compile();
                tx.send(()).unwrap();
                let mut open = gate.0.lock().unwrap();
                while !*open {
                    open = gate.1.wait(open).unwrap();
                }
            })).unwrap();
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("every blocker parked");
        }
        assert!(NORMAL_HELPERS.load(Ordering::Acquire) <= n, "replacement count is bounded by the normal pool");
        let (ran_tx, ran_rx) = mpsc::channel();
        queue().submit(Priority::Foreground, Box::new(move || ran_tx.send(()).unwrap())).unwrap();
        ran_rx.recv_timeout(Duration::from_secs(10)).expect("a replacement served the queue");
        drop(open);

        let deadline = Instant::now() + Duration::from_secs(10);
        while (PARKED_NORMAL.load(Ordering::Acquire) != 0 || NORMAL_HELPERS.load(Ordering::Acquire) != 0) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(PARKED_NORMAL.load(Ordering::Acquire), 0);
        assert_eq!(NORMAL_HELPERS.load(Ordering::Acquire), 0, "replacement workers retired");
    }

    /// A queued normal worker job whose stand-in was released before a worker starts it is
    /// skipped as orphaned instead of entering the compiler.
    #[test]
    fn a_released_queued_pipeline_is_not_built_by_normal_workers() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_9720usize + k;
            let started_tx = started_tx.clone();
            let gate = gate.clone();
            submit(standin, false, Box::new(move || {
                let _ = started_tx.send(());
                let mut open = gate.0.lock().unwrap();
                while !*open {
                    open = gate.1.wait(open).unwrap();
                }
                0x9720 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }

        let target = 0x7E57_97FFusize;
        let ran = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let ran_job = ran.clone();
        let drop_flag = DropFlag(dropped.clone());
        submit(target, false, Box::new(move || {
            let _keep = &drop_flag;
            ran_job.store(true, Ordering::Release);
            0x9721
        }));
        assert_eq!(released(target), 0);

        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !dropped.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(dropped.load(Ordering::Acquire), "queued orphan job was never drained");
        assert!(!ran.load(Ordering::Acquire), "released job reached the compiler");

        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }

    #[test]
    fn pc_call_crossing_frame_boundary_does_not_charge_prior_frame_time() {
        let mut calls = PcCalls::new();
        let t0 = std::time::Instant::now();
        let ms = std::time::Duration::from_millis;
        calls.begin(1, 7, t0);
        // Frame 8 shown after 290 ms; call 1 ends 10 ms later -> only 10 ms counts on frame 8.
        let t_frame8 = t0 + PC_CALL_HOLDS_AFTER + ms(40);
        assert!(!calls.holds(8, t_frame8), "call from frame 7 must not immediately hold frame 8");
        calls.end(1, 8, t_frame8 + ms(10));
        let t_next = t_frame8 + ms(11);
        calls.begin(2, 8, t_next);
        assert!(!calls.holds(8, t_next), "pre-frame duration leaked into frame 8's budget");
        // A call genuinely stuck >= PC_CALL_HOLDS_AFTER after frame 8 still holds the title.
        assert!(!calls.holds(8, t_next + PC_CALL_HOLDS_AFTER - ms(20)));
        assert!(calls.holds(8, t_next + PC_CALL_HOLDS_AFTER));
        calls.end(2, 8, t_next + PC_CALL_HOLDS_AFTER);

        // Also when frame 9 is first noticed inside end() itself, pre-frame time is clamped to 0.
        let t_f9 = t_next + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(3, 8, t_f9);
        let t_f9_end = t_f9 + PC_CALL_HOLDS_AFTER + ms(50);
        calls.end(3, 9, t_f9_end);
        calls.begin(4, 9, t_f9_end + ms(1));
        assert!(!calls.holds(9, t_f9_end + ms(1)), "end() on new frame must not charge prior-frame duration");
        calls.end(4, 9, t_f9_end + ms(1) + PC_CALL_HOLDS_AFTER);

        // A stale lower frame number must never rewind self.frame or wipe frame 9's spent budget.
        let t_stale = t_f9_end + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(5, 8, t_stale);
        assert!(calls.holds(9, t_stale), "stale lower frame rewound frame 9 or cleared its spent budget");
        calls.end(5, 9, t_stale);
    }

    /// The pipeline line's counts: lazy pipelines made (graphics among them), built at their
    /// first use or before it, with a huge program; a bind's wait is counted with its time, and
    /// one under a second is not logged.
    #[test]
    fn lazy_pipelines_and_the_waits_for_them_are_counted() {
        let before = stats();
        let (first_use, ahead) = (0x7E57_C001usize, 0x7E57_C002usize);
        submit_lazy(first_use, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(5));
            note_huge();
            0x6161
        }), 0);
        let key = 0x5EED_C002u64;
        used::earlier_run_used(key);
        submit_lazy(ahead, false, Box::new(|| 0x6262), key);
        let start = std::time::Instant::now();
        while is_pending(ahead) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(ahead), "built ahead with no bind");
        assert_eq!(resolve_wait_for(first_use, None), 0x6161);
        assert!(!entry(first_use).unwrap().build.warned.load(Ordering::Relaxed), "a short wait is not logged");
        let after = stats();
        assert!(after.made >= before.made + 2 && after.graphics > before.graphics, "{before:?} → {after:?}");
        assert!(after.built >= before.built + 2 && after.ahead > before.ahead && after.huge > before.huge, "{before:?} → {after:?}");
        assert!(after.waits.0 > before.waits.0 && after.waits.1 >= before.waits.1 + 5000, "{before:?} → {after:?}");
        released(first_use);
        released(ahead);
    }

    /// The line is written when a count changed, with the waits of its 2 s only.
    #[test]
    fn the_pipeline_line_comes_when_a_count_changed() {
        let last = Stats { made: 10, graphics: 8, built: 2, ahead: 1, huge: 0, warmed: 4, backlog: 6, waits: (3, 4_000_000) };
        assert_eq!(stats_line(&last, &last, 256), None);
        let now = Stats { built: 3, waits: (5, 6_500_000), ..last };
        let line = stats_line(&now, &last, 256).unwrap();
        assert!(line.contains("10 made (8 graphics), 3 built (1 before their first use, 0 with 256 KB of DXIL or more), 4 translated ahead, 6 in the build-ahead backlog"), "{line}");
        assert!(line.contains("waited 2500 ms for pipelines being built (2 waits in 2 s)"), "{line}");
    }

    /// A compute pipeline sent to the workers when made (`XCL_LAZY_PSO=0`) carries its console
    /// program as a lazy one does: a late one is logged by it (`resolve_within`).
    #[test]
    fn a_pipeline_sent_to_the_workers_when_made_carries_its_record() {
        let standin = 0x7E57_C026usize;
        let (go, finish) = mpsc::channel::<()>();
        submit_eager_with(standin, true, Box::new(move || {
            let _ = finish.recv();
            0x6A6A
        }), 0, None, 0x5EED_C026);
        assert_eq!(entry(standin).map(|p| p.build.record), Some(0x5EED_C026));
        go.send(()).unwrap();
        assert_eq!(resolve_wait_for(standin, None), 0x6A6A);
        released(standin);
    }

    /// A lazy graphics pipeline of shaders no run used yet is translated, then built, with no
    /// bind – when commit is free.
    #[test]
    fn a_lazy_graphics_pipeline_is_translated_then_built_ahead_of_its_first_use() {
        let standin = 0x7E57_C010usize;
        let warmed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let w = warmed.clone();
        let room = ahead_room();
        submit_lazy_keyed_with(
            standin,
            false,
            Box::new(move || if w.load(Ordering::Acquire) { 0x6464 } else { 1 }),
            0x5EED_C010,
            None,
            Some(Box::new(move || warmed.store(true, Ordering::Release))),
            0,
        );
        let start = std::time::Instant::now();
        let limit = std::time::Duration::from_secs(if room { 10 } else { 1 });
        while is_pending(standin) && start.elapsed() < limit {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        if room {
            assert!(!is_pending(standin), "built with no bind");
            assert_eq!(resolve(standin), 0x6464, "translated before the driver built it");
        } else {
            assert!(is_pending(standin), "short of commit: not built ahead");
        }
        released(standin);
    }

    /// A build's warm-up goes when the build is over or will not run (`complete`,
    /// `drop_unstarted_job`): what the translation ahead holds is of no use then.
    #[test]
    fn a_finished_or_dropped_build_lets_its_warm_up_go() {
        let built = Arc::new(Build::new(Box::new(|| 0), false).with_submit(true, 0, Some(Box::new(|| {}) as Warm)));
        assert!(built.lazy && built.warm.lock().unwrap().is_some());
        complete(&built, 0);
        assert!(built.warm.lock().unwrap().is_none(), "complete drops it");
        let dropped = Arc::new(Build::new(Box::new(|| 0), false).with_submit(true, 0x5EED_C027, Some(Box::new(|| {}) as Warm)));
        assert_eq!(dropped.record, 0x5EED_C027);
        drop_unstarted_job(&dropped);
        assert!(dropped.warm.lock().unwrap().is_none(), "an unstarted job's goes with it");
    }

    /// A dispatch is put off to the submit only where it would wait now: not with a pipeline that
    /// writes only images (its dispatches are left out while it is built) nor with one a bind
    /// already waited for past its budget. A submit that says it will wait for a pipeline counts
    /// once among its waiters however often it says so, and leaves when that wait runs out; a
    /// late or spared pipeline it does not count at all.
    #[test]
    fn a_dispatch_is_put_off_only_where_it_would_wait() {
        let (plain, spared) = (0x7E57_F001usize, 0x7E57_F002usize);
        let (go, finish) = mpsc::channel::<()>();
        let finish = Arc::new(Mutex::new(finish));
        for standin in [plain, spared] {
            let finish = finish.clone();
            submit_lazy(standin, true, Box::new(move || {
                let _ = finish.lock().unwrap().recv_timeout(std::time::Duration::from_secs(10));
                0x7171
            }), 0);
        }
        image_only(spared);
        assert!(waits_for(plain));
        assert!(!waits_for(spared), "its dispatches are left out while it is built");
        let p = entry(plain).unwrap();
        want(plain);
        want(plain);
        assert_eq!(p.build.demand.lock().unwrap().waiters, 1, "a submit counts once");
        assert!(p.build.wanted.load(Ordering::Acquire));
        assert_eq!(resolve_within(&p, plain, std::time::Duration::from_millis(50)), plain);
        assert_eq!(p.build.demand.lock().unwrap().waiters, 0, "the submit's count leaves with the wait that ran out");
        assert!(!p.build.wanted.load(Ordering::Acquire), "nobody waits: not wanted");
        assert!(!waits_for(plain), "late: binds go on without it");
        // a later submit that puts it off does not wait for it: it does not count
        want(plain);
        assert_eq!(p.build.demand.lock().unwrap().waiters, 0, "a late build stays unwanted");
        assert_eq!(resolve_within(&p, plain, std::time::Duration::from_millis(50)), plain);
        assert_eq!(p.build.demand.lock().unwrap().waiters, 0);
        // a submit counted before another thread's wait made it late leaves at its own
        {
            let mut d = p.build.demand.lock().unwrap();
            d.ahead = true;
            p.build.add_waiter(&mut d);
        }
        assert_eq!(resolve_within(&p, plain, std::time::Duration::from_millis(50)), plain);
        assert_eq!(p.build.demand.lock().unwrap().waiters, 0, "the submit's count leaves with its wait for a late build");
        want(spared);
        assert_eq!(entry(spared).unwrap().build.demand.lock().unwrap().waiters, 0, "nobody waits for a spared build");
        drop(go);
        let start = std::time::Instant::now();
        while (is_pending(plain) || is_pending(spared)) && start.elapsed() < std::time::Duration::from_secs(20) {
            let _ = resolve(spared);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!waits_for(plain) && !waits_for(spared), "built");
        let _ = released(plain);
        let _ = released(spared);
    }

    #[test]
    fn the_put_off_line_says_what_became_of_the_put_off_dispatches() {
        let last = PutOffStats::default();
        assert_eq!(put_off_line(&last, &last), None);
        let idle = PutOffStats { needed: 5, ..last };
        assert_eq!(put_off_line(&idle, &last), None, "nothing put off: no line");
        let now = PutOffStats { calls: [12, 1, 9, 2, 1], needed: 7, waits: (6, 2_500_000), segments: 24 };
        let before = PutOffStats { calls: [4, 0, 4, 0, 0], waits: (2, 500_000), ..Default::default() };
        let line = put_off_line(&now, &before).unwrap();
        assert!(line.contains("12 put off to the submit (8 in 2 s), 1 waited in the call"), "{line}");
        assert!(line.contains("at submits 9 recorded, 2 skipped (build failed or late), 1 dropped"), "{line}");
        assert!(line.contains("7 pipelines needed by a submit;"), "{line}");
        assert!(line.contains("submits waited 2000 ms for them in 2 s (4 waits; 2500 ms so far)"), "{line}");
        assert!(line.contains("at most 24 segments in a recording"), "{line}");
        assert_eq!(put_off_line(&now, &now), None, "nothing changed: no line");
    }

    /// A wait that passes a second is noted on its pipeline (and logged once for it).
    #[test]
    fn a_wait_past_a_second_is_noted_on_its_pipeline() {
        let standin = 0x7E57_C003usize;
        submit(standin, false, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(1100));
            0x6363
        }));
        let p = entry(standin).unwrap();
        assert_eq!(resolve_wait_for(standin, None), 0x6363);
        assert!(p.build.warned.load(Ordering::Relaxed));
        released(standin);
    }

    /// A build that takes 600 ms and gives `real`.
    fn slow(real: usize) -> Job {
        Box::new(move || {
            std::thread::sleep(std::time::Duration::from_millis(600));
            real
        })
    }

    /// The budget for graphics pipelines is a thread's for a second, not one for each pipeline:
    /// once it is spent the next pipeline is not waited for at all, it comes back with time
    /// (half of it in half a second) and never grows past itself. With a budget for each
    /// pipeline a title's thread stood still that long for every one of hundreds of pipelines
    /// it used for the first time.
    #[test]
    fn the_graphics_budget_is_a_threads_for_a_second() {
        use std::time::{Duration, Instant};
        std::thread::spawn(|| {
            let budget = Duration::from_millis(100);
            let t0 = Instant::now();
            assert_eq!(gfx_wait_left(budget, t0), budget, "all of it at first");
            gfx_wait_spent(Duration::from_millis(100));
            assert_eq!(gfx_wait_left(budget, t0), Duration::ZERO, "spent: the next pipeline is not waited for");
            let half = gfx_wait_left(budget, t0 + Duration::from_millis(500));
            assert!((half.as_secs_f64() - 0.050).abs() < 0.001, "half a second later half of it is back: {half:?}");
            assert_eq!(gfx_wait_left(budget, t0 + Duration::from_secs(5)), budget, "never more than the budget");
            gfx_wait_spent(Duration::from_millis(30));
            let left = gfx_wait_left(budget, t0 + Duration::from_secs(5));
            assert!((left.as_secs_f64() - 0.070).abs() < 0.001, "what was waited is taken off: {left:?}");
            gfx_wait_spent(Duration::from_secs(9));
            assert_eq!(gfx_wait_left(budget, t0 + Duration::from_secs(5)), Duration::ZERO, "a long wait leaves nothing, not a debt");
        })
        .join()
        .unwrap();
    }

    /// With a budget for graphics pipelines a draw waits no longer than it and later draws of
    /// that pipeline do not wait; a dispatch waits for its compute pipeline however long it takes.
    #[test]
    fn only_a_graphics_pipeline_is_left_after_its_budget() {
        use std::time::{Duration, Instant};
        let (gfx, cs) = (0x7E57_C020usize, 0x7E57_C021usize);
        submit(gfx, false, slow(0x6565));
        submit(cs, true, slow(0x6666));
        let budget = Some(Duration::from_millis(30));
        let t = Instant::now();
        assert_eq!(resolve_wait_budgets(gfx, None, budget, None), gfx, "not built within the budget: the stand-in");
        assert!(t.elapsed() < Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = Instant::now();
        assert_eq!(resolve_wait_budgets(gfx, None, budget, None), gfx);
        assert!(t.elapsed() < Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        assert_eq!(resolve_wait_budgets(cs, None, budget, None), 0x6666, "a compute pipeline is waited for");
        let start = Instant::now();
        while is_pending(gfx) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_budgets(gfx, None, budget, None), 0x6565, "built by a worker");
        let all = Some(Duration::from_secs(5));
        assert_eq!(budget_for(true, all, budget, None), all, "XCL_PSO_WAIT_MS still holds for compute");
        assert_eq!(budget_for(false, all, None, None), all);
        assert_eq!(budget_for(false, all, budget, None), budget);
        assert_eq!(budget_of(None), None);
        assert_eq!(budget_of(Some("0")), None, "0: wait until built");
        assert_eq!(budget_of(Some(" 250 ")), Some(Duration::from_millis(250)));
        assert_eq!(budget_of(Some("soon")), None);
        released(gfx);
        released(cs);
    }

    /// A build a bind gave up waiting for is no longer wanted (it goes back behind the gate of
    /// huge driver compiles); it was wanted while the bind waited.
    #[test]
    fn a_build_is_no_longer_wanted_once_its_wait_ran_out() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_C022usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, false, Box::new(move || {
            started_tx.send(()).unwrap();
            let until = |want: bool| {
                let t = Instant::now();
                while build_is_wanted() != want && t.elapsed() < Duration::from_secs(10) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                build_is_wanted() == want
            };
            if until(true) && until(false) { 0x6767 } else { 1 }
        }));
        started.recv().unwrap();
        assert_eq!(resolve_wait_budgets(standin, None, Some(Duration::from_millis(300)), None), standin);
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(resolve(standin), 0x6767, "wanted while the bind waited, not after it gave up");
        released(standin);
    }

    /// With a budget of compute pipelines (`XCL_CS_PSO_WAIT_MS`) a dispatch waits no longer than
    /// it and later dispatches of that pipeline do not wait; the bind's thread does not build the
    /// pipeline, a worker does – also a lazy one nobody sent to the workers yet. Without it a
    /// compute pipeline is waited for as before.
    #[test]
    fn a_compute_pipeline_is_waited_for_no_longer_than_its_own_budget() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_C023usize;
        let (built_on, on) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = built_on.send(std::thread::current().id());
            let _ = finish.recv();
            0x6868
        }), 0);
        let cs = Some(Duration::from_millis(30));
        let t = Instant::now();
        assert_eq!(resolve_wait_budgets(standin, None, None, cs), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < Duration::from_millis(400), "waited {:?}", t.elapsed());
        let builder = on.recv_timeout(Duration::from_secs(10)).expect("a worker builds it");
        assert_ne!(builder, std::thread::current().id(), "the bind's thread does not build it");
        let t = Instant::now();
        assert_eq!(resolve_wait_budgets(standin, None, None, cs), standin);
        assert!(t.elapsed() < Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        go.send(()).unwrap();
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_budgets(standin, None, None, cs), 0x6868, "built by a worker");
        released(standin);
        let (all, gfx) = (Some(Duration::from_secs(5)), Some(Duration::from_millis(100)));
        assert_eq!(budget_for(true, None, None, None), None, "unset: a compute pipeline is waited for until it is built");
        assert_eq!(budget_for(true, None, gfx, None), None, "the graphics budget is not one of compute pipelines");
        assert_eq!(budget_for(true, all, gfx, None), all, "unset: XCL_PSO_WAIT_MS as before");
        assert_eq!(budget_for(true, all, gfx, cs), cs, "its own before XCL_PSO_WAIT_MS");
        assert_eq!(budget_for(true, None, None, cs), cs);
        assert_eq!(budget_for(false, all, None, cs), all, "not one of graphics pipelines");
        assert_eq!(budget_for(false, None, gfx, cs), gfx);
        assert_eq!(budget_for(false, None, None, cs), None);
    }

    /// A late compute program is logged once, not once for each of its pipelines.
    #[test]
    fn a_late_compute_program_is_logged_once() {
        assert!(first_late(0x5EED_C024));
        assert!(!first_late(0x5EED_C024), "another pipeline of the same program");
        assert!(first_late(0x5EED_C025));
    }

    /// `peek` names the real pipeline once built and, unlike `resolve`, neither notes a use nor
    /// sends a lazy one to the workers.
    #[test]
    fn a_peek_changes_nothing() {
        let standin = 0x7E57_C0DEusize;
        let build = Arc::new(Build::new(Box::new(|| 0), false));
        let p = Arc::new(Pending {
            build: build.clone(),
            driver: AtomicUsize::new(0),
            compute: false,
            key: 0x5EED_C0DE,
            manifest_recipe: AtomicU64::new(0),
            track_used: true,
            used: Default::default(),
            late: Default::default(),
            spared: Default::default(),
            on_use: Default::default(),
        });
        PENDING.insert(standin, p.clone());
        assert_eq!(peek(standin), standin, "not built");
        assert!(!build.sent.load(Ordering::Acquire) && !build.used.load(Ordering::Relaxed));
        build.real.store(0x6A6B, Ordering::Release);
        assert_eq!(peek(standin), 0x6A6B);
        assert_eq!(peek(0x7E57_C0DF), 0x7E57_C0DF, "not async");
        assert!(Arc::ptr_eq(&PENDING.remove(standin).unwrap(), &p));
    }

    #[test]
    fn used_manifests_separate_series_s_from_series_x() {
        assert_eq!(used_file_name("used-pipelines", "txt", false), "used-pipelines.txt");
        assert_eq!(used_file_name("used-shaders", "txt", false), "used-shaders.txt");
        assert_eq!(used_file_name("used-graphics-pipelines", "bin", false), "used-graphics-pipelines.bin");
        assert_eq!(used_file_name("used-pipelines", "txt", true), "used-pipelines-lockhart.txt");
        assert_eq!(used_file_name("used-shaders", "txt", true), "used-shaders-lockhart.txt");
        assert_eq!(used_file_name("used-graphics-pipelines", "bin", true), "used-graphics-pipelines-lockhart.bin");

        // Host boot config takes precedence over environment variables.
        assert!(profile_is_series_s_from(Some(true), Some("anaconda")));
        assert!(!profile_is_series_s_from(Some(false), Some("lockhart")));
        // Standalone CLI / preparation workers without a boot record fall back to XCL_DEVICE_TYPE.
        for raw in ["lockhart", "Series S", "series_s", "xss", "S"] {
            assert!(profile_is_series_s_from(None, Some(raw)), "{raw}");
        }
        for raw in ["anaconda", "Series X", "xsx", "X", "", "unknown"] {
            assert!(!profile_is_series_s_from(None, Some(raw)), "{raw}");
        }
        assert!(!profile_is_series_s_from(None, None));
    }

    #[test]
    fn released_queued_pipeline_drops_job_immediately_before_workers_drain() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        struct OpenGate(Arc<(Mutex<bool>, Condvar)>);
        impl Drop for OpenGate {
            fn drop(&mut self) {
                *self.0 .0.lock().unwrap() = true;
                self.0 .1.notify_all();
            }
        }

        // Saturate normal workers and verify released() drops the queued job immediately.
        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_normal = OpenGate(gate.clone());
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_A100usize + k;
            let (tx, g) = (started_tx.clone(), gate.clone());
            submit(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA100 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }
        let target_n = 0x7E57_A1FFusize;
        let (ran_n, dropped_n) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_n) = (ran_n.clone(), DropFlag(dropped_n.clone()));
        submit(target_n, false, Box::new(move || {
            let _keep = &flag_n;
            ran_job.store(true, Ordering::Release);
            0xA1FF
        }));
        assert_eq!(released(target_n), 0);
        assert!(dropped_n.load(Ordering::Acquire), "normal queued job was not dropped immediately on released()");
        drop(open_normal);
        assert!(!ran_n.load(Ordering::Acquire));
        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }

        // Saturate preload workers and verify released() drops the preloaded job immediately.
        let m = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        let (p_tx, p_rx) = mpsc::channel();
        let p_gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_pre = OpenGate(p_gate.clone());
        let mut p_blockers = Vec::with_capacity(m);
        for k in 0..m {
            let standin = 0x7E57_A200usize + k;
            let key = 0x5EED_A200u64 + k as u64;
            used::earlier_run_used(key);
            let (tx, g) = (p_tx.clone(), p_gate.clone());
            submit_lazy(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA200 + k
            }), key);
            p_blockers.push(standin);
        }
        for _ in 0..m {
            p_rx.recv_timeout(Duration::from_secs(10)).expect("preload workers saturated");
        }
        let target_p = 0x7E57_A2FFusize;
        let key_p = 0x5EED_A2FFu64;
        used::earlier_run_used(key_p);
        let (ran_p, dropped_p) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_p) = (ran_p.clone(), DropFlag(dropped_p.clone()));
        submit_lazy(target_p, false, Box::new(move || {
            let _keep = &flag_p;
            ran_job.store(true, Ordering::Release);
            0xA2FF
        }), key_p);
        assert_eq!(released(target_p), 0);
        assert!(dropped_p.load(Ordering::Acquire), "preload queued job was not dropped immediately on released()");
        drop(open_pre);
        assert!(!ran_p.load(Ordering::Acquire));
        for standin in p_blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.get(pso)
}

#[cfg(test)]
pub(super) fn pending_identity_for_test(pso: usize) -> Option<(usize, usize)> {
    entry(pso).map(|pending| (pso, Arc::as_ptr(&pending) as usize))
}

impl Pending {
    /// What a bind of the stand-in `pso` gives the driver while the pipeline is not built.
    fn stand(&self, pso: usize) -> usize {
        match self.driver.load(Ordering::Acquire) {
            0 => pso,
            d => d,
        }
    }

    /// `bound` was this pending pipeline's stand-in before its build finished successfully.
    /// A failed build keeps its stand-in permanently, so that is a finished pipeline choice.
    fn bound_unbuilt(&self, pso: usize, bound: usize) -> bool {
        let done = self.build.done.lock().unwrap();
        bound == self.stand(pso) && !(*done && self.build.real.load(Ordering::Acquire) == 0)
    }
}

/// Stand-in `standin` is no driver object (`standin::shell`): binds get `driver` in its place.
/// Told before the title has the stand-in.
pub fn shared(standin: usize, driver: usize) {
    if let Some(p) = entry(standin) {
        p.driver.store(driver, Ordering::Release);
    }
}

/// The pipeline to use for `pso` now: the real one once built, else the stand-in's driver
/// pipeline (`pso` itself when it is one).
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    resolve_pending(&p, pso)
}

fn resolve_pending(p: &Pending, pso: usize) -> usize {
    p.note_use();
    match p.build.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers (one binds do not wait for to the
            // backlog's lane, as `resolve_spared` sends it: a bind no longer waits for a compute
            // pipeline, its dispatch does)
            let stand = p.stand(pso);
            if p.spared.load(Ordering::Acquire) {
                if !p.build.sent.swap(true, Ordering::AcqRel) {
                    send_spared(&p.build, preload_backlog());
                }
            } else {
                schedule(&p.build, Priority::BoundSoon, false, None);
                p.build.promote(Priority::Foreground);
            }
            stand
        }
        r => r,
    }
}

/// `resolve` with no side effects (for a report): the real pipeline behind `pso` once built,
/// else `pso` itself – no use noted, nothing sent to the workers.
pub(super) fn peek(pso: usize) -> usize {
    entry(pso).map(|p| p.build.real.load(Ordering::Acquire)).filter(|&r| r != 0).unwrap_or(pso)
}

/// `resolve`, waiting for a pipeline still being built on a foreground worker.
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_budgets(pso, wait_budget(), gfx_wait_budget(), cs_wait_budget())
}

/// The pipeline to bind and whether that exact choice was a stand-in for an unfinished build.
/// The answer stays true if a successful build ends after this bind chose its stand-in.
pub fn resolve_wait_bound(pso: usize) -> (usize, bool) {
    let Some(p) = entry(pso) else { return (pso, false) };
    let bound = resolve_wait_budgets(pso, wait_budget(), gfx_wait_budget(), cs_wait_budget());
    (bound, p.bound_unbuilt(pso, bound))
}

/// `resolve_wait_bound` with one `budget` for the pipeline (None: until it is built).
#[cfg(test)]
fn resolve_wait_bound_for(pso: usize, budget: Option<std::time::Duration>) -> (usize, bool) {
    let Some(p) = entry(pso) else { return (pso, false) };
    let bound = resolve_wait_for(pso, budget);
    (bound, p.bound_unbuilt(pso, bound))
}

/// `resolve_wait` with the budget of every pipeline (`XCL_PSO_WAIT_MS`), the one of graphics
/// pipelines (`XCL_GFX_PSO_WAIT_MS`) and the one of compute pipelines (`XCL_CS_PSO_WAIT_MS`).
fn resolve_wait_budgets(pso: usize, all: Option<std::time::Duration>, gfx: Option<std::time::Duration>, cs: Option<std::time::Duration>) -> usize {
    let Some(compute) = entry(pso).map(|p| p.compute) else { return resolve_wait_for(pso, None) };
    match gfx.filter(|_| !compute) {
        // the graphics budget is a thread's for a second, not one for each pipeline
        Some(budget) => {
            let start = std::time::Instant::now();
            let r = resolve_wait_for(pso, Some(gfx_wait_left(budget, start)));
            gfx_wait_spent(start.elapsed());
            r
        }
        None => resolve_wait_for(pso, budget_for(compute, all, gfx, cs)),
    }
}

thread_local! {
    /// What this thread may still wait for graphics pipelines (`gfx_wait_left`): the
    /// milliseconds left, and when they were last counted.
    static GFX_WAIT: std::cell::Cell<(f64, Option<std::time::Instant>)> = const { std::cell::Cell::new((0.0, None)) };
}

/// How long this thread may wait for a graphics pipeline at `now`, with `budget` to spend in a
/// second: what is left of it (it fills again at `budget` a second, to `budget` at most).
///
/// The budget is not one for each pipeline: the first frames of a title's world bind hundreds
/// of pipelines never used before, one draw after another on one thread. With 500 ms for each,
/// a retail title's thread stood still 500 ms more than 500 times in a row on a cold cache and
/// showed no frame for minutes, although every one of those pipelines was being built
/// meanwhile. With a share of the thread's time instead, the wait is spent on the first ones
/// (and on any that is built within milliseconds: a pipeline made from a kept blob), the
/// thread goes on, and the rest draw once they are built.
fn gfx_wait_left(budget: std::time::Duration, now: std::time::Instant) -> std::time::Duration {
    GFX_WAIT.with(|w| {
        let (left, at) = w.get();
        let full = budget.as_secs_f64() * 1000.0;
        let left = match at {
            None => full,
            Some(at) => (left + now.saturating_duration_since(at).as_secs_f64() * full).min(full),
        };
        w.set((left, Some(now)));
        std::time::Duration::from_secs_f64(left / 1000.0)
    })
}

/// This thread waited `waited` for a graphics pipeline (`gfx_wait_left`).
fn gfx_wait_spent(waited: std::time::Duration) {
    GFX_WAIT.with(|w| {
        let (left, at) = w.get();
        w.set(((left - waited.as_secs_f64() * 1000.0).max(0.0), at));
    })
}

/// How long a bind waits for a pipeline of this kind: a compute one by `cs` if there is one, else
/// by `all` (a skipped dispatch leaves out data for later work: `cs_wait_budget`), a graphics one
/// by `gfx` if there is one.
fn budget_for(compute: bool, all: Option<std::time::Duration>, gfx: Option<std::time::Duration>, cs: Option<std::time::Duration>) -> Option<std::time::Duration> {
    if compute { cs.or(all) } else { gfx.or(all) }
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    let Some(p) = entry(pso) else {
        super::pso_manifest::note_ready_compute(pso);
        return pso;
    };
    p.note_use();
    resolve_wait_pending(&p, pso, budget)
}

fn resolve_wait_pending(p: &Arc<Pending>, pso: usize, budget: Option<std::time::Duration>) -> usize {
    if let real @ 1.. = p.build.real.load(Ordering::Acquire) {
        return real;
    }
    if p.spared.load(Ordering::Acquire) {
        return resolve_spared(p, pso);
    }
    if let Some(budget) = budget {
        return resolve_within(p, pso, budget);
    }
    if *p.build.done.lock().unwrap() {
        return built_or(p, pso);
    }
    // (the title's thread is held from here until the pipeline is built)
    let _stall = crate::stall::span(crate::stall::StallKind::PipelineWait);
    let mut wait = Wait::new(p);
    // Mark it wanted before an unsent build is queued: the worker and its DXC/driver
    // admission see demand from their first instruction. An already queued build is
    // promoted by `want`.
    p.build.want();
    schedule(&p.build, Priority::Foreground, false, None);
    wait.until_done(None);
    built_or(p, pso)
}

/// `p`'s real pipeline once built, else `pso` (its stand-in).
fn built_or(p: &Pending, pso: usize) -> usize {
    match p.build.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => r,
    }
}

/// A wait of one of the title's threads for a pipeline still being built: the title is held
/// (`title_held`), the time is counted (`api_cost::Call::PipelineWait`, the pipeline line), and
/// the first wait for a pipeline that passes 1 s is logged with the thread and the pipeline's
/// kind – a draw or dispatch of the title stands still meanwhile.
struct Wait<'a> {
    p: &'a Pending,
    start: std::time::Instant,
    /// this wait logged that it passed 1 s
    logged: bool,
    _held: Held,
    _cost: crate::hle::api_cost::Timer,
}

/// A wait this long is logged (once a build).
const LONG_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

impl<'a> Wait<'a> {
    fn new(p: &'a Pending) -> Wait<'a> {
        Wait { p, start: std::time::Instant::now(), logged: false, _held: Held::new(), _cost: crate::hle::api_cost::time(crate::hle::api_cost::Call::PipelineWait) }
    }

    fn kind(&self) -> &'static str {
        if self.p.compute { "compute" } else { "graphics" }
    }

    /// Until the pipeline is built (true) or `budget` from the start of the wait is over
    /// (false; None: no budget).
    fn until_done(&mut self, budget: Option<std::time::Duration>) -> bool {
        let build = &self.p.build;
        loop {
            let waited = self.start.elapsed();
            if waited >= LONG_WAIT && !build.warned.swap(true, Ordering::Relaxed) {
                self.logged = true;
                crate::xlog_warn!(
                    "D3D12.X: thread {} has waited {} ms for a {} pipeline still being built (its draw or dispatch stands still)",
                    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() },
                    waited.as_millis(),
                    self.kind()
                );
            }
            let mut step = budget.map(|b| b.saturating_sub(waited));
            if !build.warned.load(Ordering::Relaxed) {
                let to_long = LONG_WAIT.saturating_sub(waited);
                step = Some(step.map_or(to_long, |s| s.min(to_long)));
            }
            let done = build.done.lock().unwrap();
            let done = match step {
                None => build.cv.wait_while(done, |d| !*d).unwrap(),
                Some(s) => build.cv.wait_timeout_while(done, s, |d| !*d).unwrap().0,
            };
            if *done {
                return true;
            }
            drop(done);
            if budget.is_some_and(|b| self.start.elapsed() >= b) {
                return false;
            }
        }
    }
}

impl Drop for Wait<'_> {
    fn drop(&mut self) {
        let waited = self.start.elapsed();
        if is_at_submit() {
            SUBMIT_WAITS.fetch_add(1, Ordering::Relaxed);
            SUBMIT_WAIT_US.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
        }
        if self.logged || (waited >= LONG_WAIT && !self.p.build.warned.swap(true, Ordering::Relaxed)) {
            crate::xlog_warn!(
                "D3D12.X: thread {} waited {} ms for a {} pipeline being built",
                unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() },
                waited.as_millis(),
                self.kind()
            );
        }
    }
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game's foreground worker can spend over 30 s in the driver on the
/// biggest translated programs; the waiting title thread remains held for that time.
fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| budget_of(std::env::var("XCL_PSO_WAIT_MS").ok().as_deref()))
}

/// Whether binds have an explicit time bound. A dispatch is then recorded immediately instead of
/// deferred to the submit, where it would only wait for the same bound.
pub(super) fn waits_bounded() -> bool {
    wait_budget().is_some()
}

/// How long a thread's draws wait in a second for graphics pipelines still being built
/// (`XCL_GFX_PSO_WAIT_MS`; unset or 0: as `wait_budget`, for each pipeline; `gfx_wait_left` has
/// why it is a share of the thread's time). A draw past it goes on with the stand-in, which draws
/// nothing, and so do the later draws of that pipeline until it is built. On a retail title the
/// draws of its world waited 7 – 15 s each for pipelines built one after another: no frame for
/// 156 s. Off by default: on a cold cache a logo and UI text drawn once and kept were missing
/// when draws did not wait.
fn gfx_wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| budget_of(std::env::var("XCL_GFX_PSO_WAIT_MS").ok().as_deref()))
}

/// How long a bind waits for a compute pipeline still being built (`XCL_CS_PSO_WAIT_MS`; unset
/// or 0: as `wait_budget`). The bind's thread never builds the pipeline itself: the build goes to
/// the workers (`resolve_within`; a huge driver compile also waits for free commit at the gate
/// once nobody waits), and past the budget the pipeline's dispatches are skipped until it is
/// built – what they compute is missing or old meanwhile, the counts of later ExecuteIndirects
/// among it (`indirect_guard` zeroes only counts out of range). A retail title's thread spent
/// 57 s in the driver's compiler on one program no earlier run had used, and another of its
/// threads waited for it all that time: no frame for 57 s.
fn cs_wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| budget_of(std::env::var("XCL_CS_PSO_WAIT_MS").ok().as_deref()))
}

/// A compute pipeline's dispatches go on without it while it is built (`XCL_CS_PSO_WAIT_MS`):
/// what builds one on a title's thread hands it to a thread of its own instead
/// (`real::dispatch_chain::whole_ready`).
pub(super) fn cs_goes_on() -> bool {
    cs_wait_budget().is_some()
}

/// A budget in milliseconds from the environment (unset, 0 or not a number: none).
fn budget_of(v: Option<&str>) -> Option<std::time::Duration> {
    v.and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis)
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// draws and dispatches are skipped) until a worker has built it. Its build is no longer wanted
/// (`Build::unwant`), but it was used: it goes before the builds ahead of first use.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration) -> usize {
    if *p.build.done.lock().unwrap() {
        return built_or(p, pso);
    }
    schedule(&p.build, Priority::BoundSoon, false, None);
    if p.late.load(Ordering::Acquire) {
        // (a submit that counted itself ahead, `want`, before another wait made it late: it
        // waits no more either)
        p.build.unwant_ahead();
        return p.stand(pso);
    }
    // (the title's thread is held from here until the pipeline is built or the budget is over)
    let _stall = crate::stall::span(crate::stall::StallKind::PipelineWait);
    let mut wait = Wait::new(p);
    p.build.want();
    if wait.until_done(Some(budget)) {
        return built_or(p, pso);
    }
    drop(wait);
    p.late.store(true, Ordering::Release);
    p.build.late.store(true, Ordering::Release);
    p.build.unwant();
    p.build.unwant_ahead();
    if p.compute && p.build.record != 0 && cs_wait_budget().is_some() {
        if first_late(p.build.record) {
            crate::xlog_warn!("D3D12.X: compute pipeline of record {:016x} not built after {} ms: its dispatches are skipped until it is", p.build.record, budget.as_millis());
        }
        return p.stand(pso);
    }
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    p.stand(pso)
}

/// The first late compute pipeline of console program `record` (`resolve_within`): its line is
/// written once a program, which a title makes with several root signatures.
fn first_late(record: u64) -> bool {
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    SEEN.lock().unwrap().get_or_insert_with(Default::default).insert(record)
}

/// `XCL_PSO_IMAGE_SKIP=1`: a bind does not wait for a big compute program that writes nothing
/// but images (`gpu::effects::big_and_image_only`); its dispatches are left out until a worker
/// has built it. Off by default: what such a program draws is missing for that long.
///
/// A retail title's world start binds one ray-tracing resolve program per material (73 – 118
/// KiB of console code, about 1 MB of DXIL each): an RDNA2 driver takes 22 – 172 s for each,
/// one bind at a time, and the title stands still for all of them. `XCL_PSO_WAIT_MS` goes on without every late
/// program, also the culling passes whose counts an `ExecuteIndirect` reads next – a GPU hang.
pub fn image_skip() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("XCL_PSO_IMAGE_SKIP").is_ok_and(|v| v.trim() == "1"))
}

/// The compute pipeline behind stand-in `standin` writes nothing but images: binds go on
/// without it while it is built (`resolve_spared`). Told before the title has the stand-in.
pub fn image_only(standin: usize) {
    if let Some(p) = entry(standin) {
        p.spared.store(true, Ordering::Release);
    }
}

/// `resolve_wait` for a pipeline binds do not wait for (`image_only`): the real one once built,
/// else the stand-in, with the build sent to the workers. Nobody waits, so the build is not
/// `wanted`: a big one takes its turn among the driver's compiles and waits for free commit.
fn resolve_spared(p: &Arc<Pending>, pso: usize) -> usize {
    match p.build.real.load(Ordering::Acquire) {
        0 => {
            if !p.build.sent.swap(true, Ordering::AcqRel) {
                send_spared(&p.build, preload_backlog());
            }
            p.build.late.store(true, Ordering::Release);
            if !*p.build.done.lock().unwrap() && !p.late.swap(true, Ordering::AcqRel) {
                static N: AtomicUsize = AtomicUsize::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 16 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: a compute program that writes only images is still being built: its dispatches are left out until it is ({} so far)", n + 1);
                }
            }
            p.stand(pso)
        }
        r => r,
    }
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => {
            p.note_use();
            resolve_wait_pending(&p, pso, budget_for(true, wait_budget(), None, cs_wait_budget()))
        }
        Some(p) => resolve_pending(&p, pso),
        None => {
            super::pso_manifest::note_ready_compute(pso);
            pso
        }
    }
}

/// Resolve and strongly pin the exact native generation as one operation. The
/// generation is chosen while holding `Build.done`, the same lock that publishes it and that
/// `released` uses before giving up the original raw owner. Recipe-backed and one-shot builds
/// therefore return the same generation-owned result.
pub(super) fn resolve_and_pin_for_bind(pso: usize) -> (usize, Option<NativeGeneration>) {
    resolve_and_pin_mode(pso, ResolveMode::ForBind)
}

#[derive(Clone, Copy)]
enum ResolveMode { Lazy, Wait, ForBind }

/// One retained resolver decision. Bind state snapshots it without waiting; a later emission
/// stage can consume it without looking up a possibly reused title address.
#[allow(dead_code)]
#[derive(Clone)]
enum StructuredResolve<const R: usize> {
    Ready {
        original: usize,
        pending: Arc<Pending>,
        raw: usize,
        pin: Option<NativeGeneration>,
    },
    Initial {
        original: usize,
        pending: Arc<Pending>,
        fallback: usize,
        bound_unbuilt: bool,
        /// First exact generation successfully consumed from this retained bind. Every clone
        /// shares the winner; unavailable attempts remain retryable.
        promoted: Arc<OnceLock<NativeGeneration>>,
    },
    /// Exact identity retained, but no restoration was requested by this snapshot.
    Deferred {
        original: usize,
        pending: Arc<Pending>,
        fallback: usize,
        restoration: Arc<RestorationDemand<R>>,
    },
    Restoring {
        original: usize,
        pending: Arc<Pending>,
        fallback: usize,
        ticket: RestorationTicket<R>,
    },
    Terminal {
        original: usize,
        pending: Arc<Pending>,
        fallback: usize,
        failure: StructuredResolveFailure,
    },
}

const PIPELINE_RESTORATION_EVICTIONS: usize = 1024;
const PIPELINE_RESOLUTION_RECEIPTS: usize = 1024;
type PipelineRestorationCoordinator =
    RestorationCoordinator<PIPELINE_RESTORATION_EVICTIONS, PIPELINE_RESOLUTION_RECEIPTS>;

/// Exact restoration identity shared by every clone of one retained bind. Classification only
/// captures the already-installed eviction token; submit/emission demand owns worker startup and
/// the single request publication.
struct RestorationDemand<const R: usize> {
    evicted: EvictedBuildToken,
    demanded: AtomicBool,
    requested: OnceLock<DeferredDemand<R>>,
}

#[derive(Clone)]
enum DeferredDemand<const R: usize> {
    Ready(NativeGeneration),
    Restoring(RestorationTicket<R>),
    Failed(StructuredResolveFailure),
}

impl<const R: usize> RestorationDemand<R> {
    fn new(evicted: EvictedBuildToken) -> Self {
        Self { evicted, demanded: AtomicBool::new(false), requested: OnceLock::new() }
    }
}

impl RestorationDemand<PIPELINE_RESOLUTION_RECEIPTS> {
    fn resident(pending: &Pending) -> Option<NativeGeneration> {
        let recipe = pending.build.recipe.as_ref()?;
        let raw = pending.build.real.load(Ordering::Acquire);
        let (pin, residency) = recipe.pin_with_residency(raw);
        pin.filter(|_| matches!(residency, Residency::Resident | Residency::Rebuilding))
    }

    fn request(&self, pending: &Pending) -> DeferredDemand<PIPELINE_RESOLUTION_RECEIPTS> {
        self.demanded.store(true, Ordering::Release);
        if let Some(pin) = Self::resident(pending) {
            return DeferredDemand::Ready(pin);
        }
        let requested = self.requested.get_or_init(|| {
            let _demand = crate::stall::span(crate::stall::StallKind::PipelineRestoreDemand);
            if let Some(pin) = Self::resident(pending) {
                return DeferredDemand::Ready(pin);
            }
            // Retire terminal attempts left by abandoned lists before consuming another bounded
            // slot. A failed ticket is terminal for this retained bind; a later bind gets a new
            // demand cell and may retry the still-exact eviction token.
            reconcile_pipeline_restorations();
            let requested = pipeline_restoration_coordinator().request_foreground(&self.evicted);
            // Publication/reconciliation can cross the exact-token snapshot and request. Prefer
            // the retained build's newly owned resident generation over a stale refusal.
            if let Some(pin) = Self::resident(pending) {
                return DeferredDemand::Ready(pin);
            }
            requested.map_or(
                DeferredDemand::Failed(StructuredResolveFailure::RestorationUnavailable),
                DeferredDemand::Restoring,
            )
        }).clone();
        // A failed attempt is terminal only for this demand cell. Another retained bind may
        // retry the same exact eviction and publish its replacement before this bind emits.
        Self::resident(pending).map_or(requested, DeferredDemand::Ready)
    }
}

/// Merely obtaining the coordinator neither starts workers nor enables eviction.
fn pipeline_restoration_coordinator() -> &'static PipelineRestorationCoordinator {
    static COORDINATOR: OnceLock<PipelineRestorationCoordinator> = OnceLock::new();
    COORDINATOR.get_or_init(|| {
        PipelineRestorationCoordinator::new_with_pressure_stats(1, Some(pressure_counters()))
            .expect("nonzero pipeline restoration capacities")
    })
}

fn reconcile_pipeline_restorations() {
    let _reconcile = crate::stall::span(crate::stall::StallKind::PipelineRestoreReconcile);
    let _ = pipeline_restoration_coordinator().reconcile(PIPELINE_RESOLUTION_RECEIPTS);
}

/// Opaque retained resolver decision carried by command-list state.
/// Immutable compute kind and unavailable-Set policy are retained across address reuse/release.
#[allow(dead_code)] // consumed by a later emission stage; retaining it pins exact identity now
#[derive(Clone)]
pub(super) struct PipelineResolution(
    StructuredResolve<PIPELINE_RESOLUTION_RECEIPTS>, bool, UnavailableSet,
);

#[derive(Clone, Copy)]
enum UnavailableSet { Omit, Poison }

fn unavailable_set(pending: &Pending) -> UnavailableSet {
    let evictable = pending.build.is_evictable();
    debug_assert!(!evictable || pending.compute);
    if pending.compute && evictable { UnavailableSet::Omit } else { UnavailableSet::Poison }
}

/// Generation-aware emission result. `Unavailable` is fail-closed: emission must never recover
/// by looking up the title address again after this retained identity was selected.
#[allow(dead_code)] // consumed by the later command-list integration stage
pub(super) enum PipelineChoice {
    Available { raw: usize, generation: NativeGeneration },
    Unavailable,
}

#[allow(dead_code)] // consumed by the ExecuteIndirect integration stage
pub(super) enum IndirectPipelineChoice {
    Exact { raw: usize, generation: NativeGeneration },
    InitialFallback { raw: usize, bound_unbuilt: bool },
    Unavailable,
}

/// Retained native command-list bind decision. Exact owns the selected generation;
/// InitialFallback is legal only for this snapshot's still-live initial build. Activation must
/// apply each API's fail-closed Unavailable behavior without registry relookup or stale fallback.
#[allow(dead_code)] // consumed by the native command-list bind integration stage
pub(super) enum NativePipelineChoice {
    Exact { raw: usize, generation: NativeGeneration },
    InitialFallback { raw: usize, bound_unbuilt: bool },
    Unavailable,
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // consumed by the native command-list bind integration stage
pub(super) enum NativePipelineBindMode { Lazy, ForBind }

/// Wrap one nonblocking decision without exposing coordinator or receipt types to command-list
/// state. This does not start workers and never runs a queued driver build on the caller.
#[allow(dead_code)] // activated with the later restoration-request stage
fn begin_pipeline_resolution_on<const E: usize>(
    original: usize,
    pending: &Arc<Pending>,
    coordinator: &RestorationCoordinator<E, PIPELINE_RESOLUTION_RECEIPTS>,
    q: &BuildQueue,
) -> PipelineResolution {
    PipelineResolution(
        classify_structured_on(original, pending, coordinator, q),
        pending.compute, unavailable_set(pending),
    )
}

/// Classify a bind without waiting or running driver work. Worker startup, consumption of the
/// decision, and pressure policy remain separate stages.
pub(super) fn begin_pipeline_resolution(original: usize) -> Option<PipelineResolution> {
    let pending = entry(original)?;
    Some(PipelineResolution(
        classify_retained(original, &pending), pending.compute, unavailable_set(&pending),
    ))
}

impl PipelineResolution {
    #[allow(dead_code)] // consumed by the ExecuteIndirect integration stage
    pub(super) fn is_compute(&self) -> bool { self.1 }

    /// Whether every later consumer is retained-resolution emission and therefore fails closed
    /// when this identity is unavailable. Other pipeline kinds must poison a successful recording.
    pub(super) fn unavailable_is_emission_guarded(&self) -> bool {
        matches!(self.2, UnavailableSet::Omit)
    }

    /// Select a native bind from this retained identity only. Lazy never waits; ForBind keeps
    /// the existing bounded compute wait while graphics remains lazy.
    #[allow(dead_code)] // consumed by the native command-list bind integration stage
    pub(super) fn select_for_native_bind(
        &self, mode: NativePipelineBindMode,
    ) -> NativePipelineChoice {
        self.select_for_native_bind_with_budget(mode, wait_budget())
    }

    #[allow(dead_code)] // testable policy core for the next integration stage
    fn select_for_native_bind_with_budget(
        &self, mode: NativePipelineBindMode, budget: Option<std::time::Duration>,
    ) -> NativePipelineChoice {
        if let StructuredResolve::Initial { original, pending, promoted, .. } = &self.0 {
            if let Some(generation) = promoted.get() {
                pending.note_resident_use();
                return native_exact(generation.clone());
            }
            if pending.build.owners.load(Ordering::Acquire) == 0
                || pending.build.orphan.load(Ordering::Acquire)
            {
                return NativePipelineChoice::Unavailable;
            }
            let raw = match mode {
                NativePipelineBindMode::ForBind if self.1 => {
                    pending.note_use();
                    resolve_wait_pending(pending, *original, budget)
                }
                NativePipelineBindMode::Lazy | NativePipelineBindMode::ForBind => {
                    resolve_pending(pending, *original)
                }
            };
            let bound_unbuilt = pending.bound_unbuilt(*original, raw);
            let (raw, generation, stale_fallback) =
                pin_resolved(*original, pending, raw, bound_unbuilt);
            return match (generation, stale_fallback) {
                (Some(generation), _) => promote_native_choice(promoted, generation),
                (None, false) => finish_native_initial_choice(
                    promoted, pending,
                    NativePipelineChoice::InitialFallback { raw, bound_unbuilt },
                ),
                (None, true) => finish_native_initial_choice(
                    promoted, pending, NativePipelineChoice::Unavailable,
                ),
            };
        }
        match (&self.0, mode) {
            (StructuredResolve::Ready { pending, pin: Some(generation), .. }, _) => {
                pending.note_resident_use();
                native_exact(generation.clone())
            }
            (StructuredResolve::Deferred { pending, .. }, _) => {
                if pending.build.owners.load(Ordering::Acquire) == 0
                    || pending.build.orphan.load(Ordering::Acquire)
                {
                    NativePipelineChoice::Unavailable
                } else {
                    RestorationDemand::resident(pending).map_or(
                        NativePipelineChoice::Unavailable,
                        |generation| {
                            pending.note_resident_use();
                            native_exact(generation)
                        },
                    )
                }
            }
            _ => NativePipelineChoice::Unavailable,
        }
    }

    #[allow(dead_code)] // consumed by the ExecuteIndirect integration stage
    pub(super) fn resolve_for_indirect(&self) -> IndirectPipelineChoice {
        self.resolve_for_indirect_with_budget(wait_budget())
    }

    fn resolve_for_indirect_with_budget(
        &self, budget: Option<std::time::Duration>,
    ) -> IndirectPipelineChoice {
        if let StructuredResolve::Initial { original, pending, promoted, .. } = &self.0 {
            if let Some(generation) = promoted.get() {
                pending.note_resident_use();
                return IndirectPipelineChoice::Exact {
                    raw: generation.raw(), generation: generation.clone(),
                };
            }
            if pending.build.owners.load(Ordering::Acquire) == 0
                || pending.build.orphan.load(Ordering::Acquire)
            {
                return IndirectPipelineChoice::Unavailable;
            }
            pending.note_use();
            let bound = resolve_wait_pending(pending, *original, budget);
            let bound_unbuilt = pending.bound_unbuilt(*original, bound);
            let (raw, generation, stale_fallback) = pin_resolved(
                *original, pending, bound, bound_unbuilt,
            );
            return match (generation, stale_fallback) {
                (Some(generation), _) => {
                    promote_indirect_initial_choice(promoted, generation)
                }
                (None, false) => finish_indirect_initial_choice(
                    promoted, pending,
                    IndirectPipelineChoice::InitialFallback { raw, bound_unbuilt },
                ),
                (None, true) => finish_indirect_initial_choice(
                    promoted, pending, IndirectPipelineChoice::Unavailable,
                ),
            };
        }
        match self.resolve_for_emission() {
            PipelineChoice::Available { raw, generation } => {
                IndirectPipelineChoice::Exact { raw, generation }
            }
            PipelineChoice::Unavailable => IndirectPipelineChoice::Unavailable,
        }
    }

    /// Whether submit can move exact work off recording without looking up the title address.
    pub(super) fn submit_demand_pending(&self) -> bool {
        matches!(&self.0, StructuredResolve::Initial { pending, bound_unbuilt: true, promoted, .. }
            if promoted.get().is_none()
                && pending.build.owners.load(Ordering::Acquire) != 0
                && !pending.build.orphan.load(Ordering::Acquire)
                && !pending.build.completed.load(Ordering::Acquire))
            || matches!(&self.0, StructuredResolve::Deferred { restoration, .. }
                if !restoration.demanded.load(Ordering::Acquire))
    }

    pub(super) fn is_deferred_restoration(&self) -> bool {
        matches!(&self.0, StructuredResolve::Deferred { .. })
    }

    /// Publish exact demand before a deferred submit. Initial work is promoted as before; an
    /// evicted generation creates or promotes one coordinator attempt on normal workers.
    pub(super) fn need_by_submit(&self) -> bool {
        match &self.0 {
            StructuredResolve::Initial { pending, bound_unbuilt: true, promoted, .. } => {
                if promoted.get().is_none()
                    && pending.build.owners.load(Ordering::Acquire) != 0
                    && !pending.build.orphan.load(Ordering::Acquire)
                {
                    need_pending_by_submit(pending);
                }
                true
            }
            StructuredResolve::Deferred { pending, restoration, .. } => {
                if restoration.demanded.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
                    return false;
                }
                let _ = restoration.request(pending);
                true
            }
            _ => false,
        }
    }

    /// Exact completed generation safe to pass as a native Reset initial pipeline. An unfinished
    /// or unavailable retained identity returns None instead of consulting its reusable address.
    pub(super) fn ready_for_reset(&self) -> Option<(usize, NativeGeneration)> {
        match &self.0 {
            StructuredResolve::Ready { pending, raw, pin: Some(generation), .. } => {
                pending.note_resident_use();
                Some((*raw, generation.clone()))
            }
            StructuredResolve::Initial { pending, promoted, .. } => {
                promoted.get().map(|generation| {
                    pending.note_resident_use();
                    (generation.raw(), generation.clone())
                })
            }
            _ => None,
        }
    }

    /// Resolve this bind's retained identity for correctness-critical emission. No branch looks
    /// up `original` in the registry: an address reused since the bind cannot redirect the wait.
    /// An evicted generation waits only on the exact ticket retained by this bind.
    #[allow(dead_code)] // consumed by the later command-list integration stage
    pub(super) fn resolve_for_emission(&self) -> PipelineChoice {
        match &self.0 {
            StructuredResolve::Ready { pending, raw, pin: Some(generation), .. } => {
                pending.note_resident_use();
                PipelineChoice::Available { raw: *raw, generation: generation.clone() }
            }
            StructuredResolve::Initial { original, pending, promoted, .. } => {
                if let Some(generation) = promoted.get() {
                    pending.note_resident_use();
                    return PipelineChoice::Available { raw: generation.raw(), generation: generation.clone() };
                }
                promote_initial_choice(promoted, resolve_retained_for_emission(*original, pending))
            }
            StructuredResolve::Deferred { pending, restoration, .. } => {
                let generation = match restoration.request(pending) {
                    DeferredDemand::Ready(generation) => Some(generation),
                    DeferredDemand::Restoring(ticket) => ticket.wait_and_take_pin().ok(),
                    DeferredDemand::Failed(failure) => { let _terminal = failure; None }
                }.or_else(|| RestorationDemand::resident(pending));
                // Clone/take the exact success pin above before terminal reconciliation can drop
                // the coordinator's attempt owner.
                reconcile_pipeline_restorations();
                generation.map_or(PipelineChoice::Unavailable, |generation| {
                    pending.note_resident_use();
                    PipelineChoice::Available { raw: generation.raw(), generation }
                })
            }
            StructuredResolve::Ready { pin: None, .. }
            | StructuredResolve::Restoring { .. }
            | StructuredResolve::Terminal { .. } => PipelineChoice::Unavailable,
        }
    }
}

#[allow(dead_code)] // consumed by the native command-list bind integration stage
fn native_exact(generation: NativeGeneration) -> NativePipelineChoice {
    NativePipelineChoice::Exact { raw: generation.raw(), generation }
}

#[allow(dead_code)] // consumed by the native command-list bind integration stage
fn promote_native_choice(
    promoted: &OnceLock<NativeGeneration>, generation: NativeGeneration,
) -> NativePipelineChoice {
    let _ = promoted.set(generation);
    native_exact(promoted.get().expect("initial native exact promotion").clone())
}

#[allow(dead_code)] // consumed by the native command-list bind integration stage
fn finish_native_initial_choice(
    promoted: &OnceLock<NativeGeneration>, pending: &Pending, local: NativePipelineChoice,
) -> NativePipelineChoice {
    if let Some(generation) = promoted.get() {
        return native_exact(generation.clone());
    }
    if pending.build.owners.load(Ordering::Acquire) == 0
        || pending.build.orphan.load(Ordering::Acquire)
    {
        return NativePipelineChoice::Unavailable;
    }
    local
}

fn promote_indirect_initial_choice(
    promoted: &OnceLock<NativeGeneration>, generation: NativeGeneration,
) -> IndirectPipelineChoice {
    let _ = promoted.set(generation);
    let generation = promoted.get().expect("initial exact promotion").clone();
    IndirectPipelineChoice::Exact { raw: generation.raw(), generation }
}

fn finish_indirect_initial_choice(
    promoted: &OnceLock<NativeGeneration>, pending: &Pending, local: IndirectPipelineChoice,
) -> IndirectPipelineChoice {
    if let Some(generation) = promoted.get() {
        return IndirectPipelineChoice::Exact {
            raw: generation.raw(), generation: generation.clone(),
        };
    }
    if pending.build.owners.load(Ordering::Acquire) == 0
        || pending.build.orphan.load(Ordering::Acquire)
    {
        return IndirectPipelineChoice::Unavailable;
    }
    local
}

fn promote_initial_choice(
    promoted: &OnceLock<NativeGeneration>,
    choice: PipelineChoice,
) -> PipelineChoice {
    let PipelineChoice::Available { generation, .. } = choice else {
        return promoted.get().map_or(PipelineChoice::Unavailable, |generation| {
            PipelineChoice::Available { raw: generation.raw(), generation: generation.clone() }
        });
    };
    let _ = promoted.set(generation);
    let generation = promoted.get().expect("an available initial generation was not promoted").clone();
    PipelineChoice::Available { raw: generation.raw(), generation }
}

/// Opaque fixture for the feature-gated pipeline-resolution microbenchmark. It uses the real
/// retained resolver with prewarmed fake COM owners, but starts no workers and calls no driver.
#[cfg(feature = "pipeline-resolution-bench")]
#[doc(hidden)]
pub struct PipelineResolutionBench {
    retained_ready: PipelineResolution,
    retained_ready_recipe_tracked: PipelineResolution,
    initial_completed: PipelineResolution,
    initial_unpromoted_shared: PipelineResolution,
    initial_unpromoted_distinct: Vec<PipelineResolution>,
    registry_original: usize,
    registry_pending: Arc<Pending>,
    tracked_build: Arc<Build>,
    _raw_owners: Vec<ID3D12PipelineState>,
    _natives: Vec<Box<PipelineResolutionBenchPso>>,
    _vtable: Box<[usize; 3]>,
}

#[cfg(feature = "pipeline-resolution-bench")]
#[repr(C)]
struct PipelineResolutionBenchPso {
    vtable: *const usize,
    refs: AtomicUsize,
}

// The benchmark vtable is immutable and its only mutable state is atomic reference counting.
#[cfg(feature = "pipeline-resolution-bench")]
unsafe impl Send for PipelineResolutionBenchPso {}
#[cfg(feature = "pipeline-resolution-bench")]
unsafe impl Sync for PipelineResolutionBenchPso {}

#[cfg(feature = "pipeline-resolution-bench")]
unsafe extern "system" fn pipeline_resolution_bench_add_ref(this: *mut std::ffi::c_void) -> u32 {
    (*(this as *mut PipelineResolutionBenchPso)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
}

#[cfg(feature = "pipeline-resolution-bench")]
unsafe extern "system" fn pipeline_resolution_bench_release(this: *mut std::ffi::c_void) -> u32 {
    (*(this as *mut PipelineResolutionBenchPso)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
}

/// Distinct exact-generation identities backed by one fake COM object. Field order keeps every
/// generation owner alive only while the fake object and its vtable still exist.
#[cfg(feature = "pipeline-resolution-bench")]
pub(super) struct NativeGenerationBenchSet {
    generations: Vec<NativeGeneration>,
    _raw_owner: ID3D12PipelineState,
    _native: Box<PipelineResolutionBenchPso>,
    _vtable: Box<[usize; 3]>,
}

#[cfg(feature = "pipeline-resolution-bench")]
impl NativeGenerationBenchSet {
    pub(super) fn new(count: usize) -> Self {
        assert!(count != 0);
        let vtable = Box::new([
            0,
            pipeline_resolution_bench_add_ref as usize,
            pipeline_resolution_bench_release as usize,
        ]);
        let mut native = Box::new(PipelineResolutionBenchPso {
            vtable: vtable.as_ptr(),
            refs: AtomicUsize::new(1),
        });
        let raw = (&mut *native as *mut PipelineResolutionBenchPso).cast::<std::ffi::c_void>();
        let raw_owner = unsafe { ID3D12PipelineState::from_raw(raw) };
        let generations = (0..count)
            .map(|index| NativeGeneration {
                generation: Arc::new(Generation {
                    id: index as u64 + 1,
                    raw: raw as usize,
                    _owner: raw_owner.clone(),
                    pressure_release: Default::default(),
                }),
                recipe: Weak::new(),
            })
            .collect();
        Self { generations, _raw_owner: raw_owner, _native: native, _vtable: vtable }
    }

    pub(super) fn get(&self, index: usize) -> NativeGeneration {
        self.generations[index].clone()
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
impl PipelineResolutionBench {
    pub fn new() -> Self {
        let vtable = Box::new([
            0,
            pipeline_resolution_bench_add_ref as usize,
            pipeline_resolution_bench_release as usize,
        ]);
        let mut natives = Vec::new();
        let mut raw_owners = Vec::new();
        let mut raws = Vec::new();
        for index in 0..4 {
            let mut native = Box::new(PipelineResolutionBenchPso {
                vtable: vtable.as_ptr(),
                refs: AtomicUsize::new(1),
            });
            let raw = (&mut *native as *mut PipelineResolutionBenchPso).cast::<std::ffi::c_void>();
            if index < 3 {
                raw_owners.push(unsafe { ID3D12PipelineState::from_raw(raw) });
            }
            raws.push(raw as usize);
            natives.push(native);
        }

        let make_pending = || Arc::new(Pending {
            build: Arc::new(Build::new(Box::new(|| 0), false)),
            driver: AtomicUsize::new(0),
            compute: false,
            key: 0,
            used: AtomicBool::new(false),
            late: AtomicBool::new(false),
            spared: AtomicBool::new(false),
        });
        let finish = |pending: &Pending, raw| {
            drop(pending.build.job.lock().unwrap().take());
            let done = pending.build.done.lock().unwrap();
            pending.build.real.store(raw, Ordering::Release);
            pending.build.publish_completed(done);
        };

        let ready_pending = make_pending();
        finish(&ready_pending, raws[0]);
        let retained_ready = PipelineResolution(classify_retained(raws[0] ^ 0x101, &ready_pending), ready_pending.compute, UnavailableSet::Poison);
        assert!(matches!(&retained_ready.0, StructuredResolve::Ready { .. }));

        let tracked_raw = raws[3];
        let tracked_recipe = Recipe::new(move || tracked_raw);
        let tracked_build = Arc::new(Build::new_keyed_kind(
            tracked_recipe.initial_job(), false, None, Some(tracked_recipe), PipelineKind::Compute,
        ));
        let tracked_job = tracked_build.job.lock().unwrap().take().unwrap();
        complete(&tracked_build, tracked_job());
        let tracked_pending = Arc::new(Pending {
            build: tracked_build.clone(),
            driver: AtomicUsize::new(0),
            compute: true,
            key: 0,
            used: AtomicBool::new(false),
            late: AtomicBool::new(false),
            spared: AtomicBool::new(false),
        });
        let retained_ready_recipe_tracked = PipelineResolution(
            classify_retained(tracked_raw ^ 0x505, &tracked_pending),
            tracked_pending.compute,
            UnavailableSet::Omit,
        );
        assert!(matches!(&retained_ready_recipe_tracked.0, StructuredResolve::Ready { .. }));
        let _ = Self::consume(retained_ready_recipe_tracked.resolve_for_emission());
        assert!(tracked_build.candidate_queued.load(Ordering::Acquire));

        let initial_pending = make_pending();
        let initial_completed = PipelineResolution(classify_retained(raws[1] ^ 0x202, &initial_pending), initial_pending.compute, UnavailableSet::Poison);
        assert!(matches!(&initial_completed.0, StructuredResolve::Initial { .. }));
        finish(&initial_pending, raws[1]);
        let _ = Self::consume(initial_completed.resolve_for_emission());

        let make_unpromoted = || {
            let pending = make_pending();
            let resolution = PipelineResolution(classify_retained(raws[1] ^ 0x303, &pending), pending.compute, UnavailableSet::Poison);
            finish(&pending, raws[1]);
            resolution
        };
        let initial_unpromoted_shared = make_unpromoted();
        let initial_unpromoted_distinct = (0..8).map(|_| make_unpromoted()).collect();

        let registry_pending = make_pending();
        finish(&registry_pending, raws[2]);
        let registry_original = raws[2] ^ 0x404;
        assert!(PENDING.insert(registry_original, registry_pending.clone()).is_none());
        let registry_warm = begin_pipeline_resolution(registry_original).expect("benchmark registry fixture");
        let _ = Self::consume(registry_warm.resolve_for_emission());

        Self {
            retained_ready,
            retained_ready_recipe_tracked,
            initial_completed,
            initial_unpromoted_shared,
            initial_unpromoted_distinct,
            registry_original,
            registry_pending,
            tracked_build,
            _raw_owners: raw_owners,
            _natives: natives,
            _vtable: vtable,
        }
    }

    #[inline(never)]
    pub fn retained_ready(&self) -> usize {
        Self::consume(self.retained_ready.resolve_for_emission())
    }

    #[inline(never)]
    pub fn retained_ready_recipe_tracked(&self) -> usize {
        Self::consume(self.retained_ready_recipe_tracked.resolve_for_emission())
    }

    pub(super) fn retained_resolution(&self) -> PipelineResolution {
        self.retained_ready.clone()
    }

    #[inline(never)]
    pub fn registry_ready(&self) -> usize {
        let resolution = begin_pipeline_resolution(self.registry_original).expect("benchmark registry fixture disappeared");
        Self::consume(resolution.resolve_for_emission())
    }

    #[inline(never)]
    pub fn initial_completed(&self) -> usize {
        Self::consume(self.initial_completed.resolve_for_emission())
    }

    #[inline(never)]
    pub fn initial_completed_dispatch(&self) -> usize {
        if self.initial_completed.submit_demand_pending() {
            return 0;
        }
        Self::consume(self.initial_completed.resolve_for_emission())
    }

    #[inline(never)]
    pub fn initial_unpromoted_shared_probe(&self) -> bool {
        self.initial_unpromoted_shared.submit_demand_pending()
    }

    #[inline(never)]
    pub fn initial_unpromoted_distinct_probe(&self, worker: usize) -> bool {
        self.initial_unpromoted_distinct[worker % self.initial_unpromoted_distinct.len()].submit_demand_pending()
    }

    #[inline(never)]
    pub fn initial_promoted_probe(&self) -> bool {
        self.initial_completed.submit_demand_pending()
    }

    fn consume(choice: PipelineChoice) -> usize {
        match choice {
            PipelineChoice::Available { raw, generation } => raw.rotate_left(13) ^ generation.id() as usize,
            PipelineChoice::Unavailable => panic!("benchmark pipeline became unavailable"),
        }
    }
}

#[cfg(feature = "pipeline-resolution-bench")]
impl Drop for PipelineResolutionBench {
    fn drop(&mut self) {
        let removed = PENDING.remove(self.registry_original).expect("benchmark registry fixture disappeared");
        assert!(Arc::ptr_eq(&removed, &self.registry_pending));
        release(self.tracked_build.real.swap(0, Ordering::AcqRel));
    }
}

#[cfg(test)]
struct PipelineResolutionDropProbe(Option<Box<dyn FnOnce() + Send>>);

#[cfg(test)]
impl Drop for PipelineResolutionDropProbe {
    fn drop(&mut self) {
        if let Some(on_drop) = self.0.take() {
            on_drop();
        }
    }
}

#[cfg(test)]
pub(super) fn pipeline_resolution_for_test(
    original: usize,
    on_drop: impl FnOnce() + Send + 'static,
) -> PipelineResolution {
    pipeline_resolution_for_kind_test(original, false, on_drop)
}

#[cfg(test)]
pub(super) fn pipeline_resolution_for_compute_test(
    original: usize,
    on_drop: impl FnOnce() + Send + 'static,
) -> PipelineResolution {
    pipeline_resolution_for_kind_test(original, true, on_drop)
}

#[cfg(test)]
fn pipeline_resolution_for_kind_test(
    original: usize,
    compute: bool,
    on_drop: impl FnOnce() + Send + 'static,
) -> PipelineResolution {
    let probe = PipelineResolutionDropProbe(Some(Box::new(on_drop)));
    let build = Arc::new(Build::new(Box::new(move || {
        let _keep = &probe;
        0
    }), false));
    let pending = Arc::new(Pending {
        build,
        driver: AtomicUsize::new(0),
        compute,
        key: 0,
        manifest_recipe: AtomicU64::new(0),
        track_used: true,
        used: AtomicBool::new(false),
        late: AtomicBool::new(false),
        spared: AtomicBool::new(false),
        on_use: Mutex::new(None),
    });
    PipelineResolution(
        StructuredResolve::initial(original, pending, original, true),
        compute, if compute { UnavailableSet::Omit } else { UnavailableSet::Poison },
    )
}

#[cfg(test)]
impl PipelineResolution {
    pub(super) fn identity_for_test(&self) -> (usize, usize) {
        match &self.0 {
            StructuredResolve::Initial { original, pending, .. }
            | StructuredResolve::Deferred { original, pending, .. }
            | StructuredResolve::Restoring { original, pending, .. }
            | StructuredResolve::Terminal { original, pending, .. } => {
                (*original, Arc::as_ptr(pending) as usize)
            }
            StructuredResolve::Ready { original, .. } => (*original, 0),
        }
    }

    pub(super) fn unavailable_for_test(&self) -> Self {
        let (original, pending, fallback) = match &self.0 {
            StructuredResolve::Initial { original, pending, fallback, .. }
            | StructuredResolve::Deferred { original, pending, fallback, .. }
            | StructuredResolve::Restoring { original, pending, fallback, .. }
            | StructuredResolve::Terminal { original, pending, fallback, .. } => {
                (*original, pending.clone(), *fallback)
            }
            StructuredResolve::Ready { .. } => panic!("ready resolution cannot be made unavailable"),
        };
        Self(StructuredResolve::Terminal {
            original, pending, fallback,
            failure: StructuredResolveFailure::RestorationUnavailable,
        }, self.1, self.2)
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StructuredResolveFailure {
    MissingEvictionRecord,
    RestorationUnavailable,
    Refused,
    Cancelled,
    Panicked,
    BuildFailed,
    Released,
}

#[allow(dead_code)]
struct StructuredResolvedPipeline {
    raw: usize,
    pin: Option<NativeGeneration>,
    bound_unbuilt: bool,
}

#[allow(dead_code)]
impl<const R: usize> StructuredResolve<R> {
    fn initial(original: usize, pending: Arc<Pending>, fallback: usize, bound_unbuilt: bool) -> Self {
        Self::Initial {
            original,
            pending,
            fallback,
            bound_unbuilt,
            promoted: Arc::new(OnceLock::new()),
        }
    }

    /// Never waits. An unpromoted initial build or restoration returns only its retained fallback.
    fn nonblocking(&self) -> StructuredResolvedPipeline {
        match self {
            Self::Ready { raw, pin, .. } => StructuredResolvedPipeline {
                raw: *raw, pin: pin.clone(), bound_unbuilt: false,
            },
            Self::Initial { fallback, bound_unbuilt, promoted, .. } => match promoted.get() {
                Some(generation) => StructuredResolvedPipeline {
                    raw: generation.raw(), pin: Some(generation.clone()), bound_unbuilt: false,
                },
                None => StructuredResolvedPipeline {
                    raw: *fallback, pin: None, bound_unbuilt: *bound_unbuilt,
                },
            },
            Self::Deferred { fallback, .. } | Self::Restoring { fallback, .. }
            | Self::Terminal { fallback, .. } => {
                StructuredResolvedPipeline { raw: *fallback, pin: None, bound_unbuilt: true }
            }
        }
    }

    /// Wait only for an already-classified exact restoration. A successfully promoted initial
    /// choice is already exact; other initial cold-build policy remains with the existing resolver.
    fn wait_exact(&self) -> Result<Option<StructuredResolvedPipeline>, StructuredResolveFailure> {
        match self {
            Self::Ready { raw, pin, .. } => Ok(Some(StructuredResolvedPipeline {
                raw: *raw, pin: pin.clone(), bound_unbuilt: false,
            })),
            Self::Initial { promoted, .. } => Ok(promoted.get().map(|generation| StructuredResolvedPipeline {
                raw: generation.raw(), pin: Some(generation.clone()), bound_unbuilt: false,
            })),
            Self::Deferred { .. } => Ok(None),
            Self::Restoring { ticket, .. } => ticket.wait_and_take_pin()
                .map(|pin| Some(StructuredResolvedPipeline { raw: pin.raw(), pin: Some(pin), bound_unbuilt: false }))
                .map_err(StructuredResolveFailure::from),
            Self::Terminal { failure, .. } => Err(*failure),
        }
    }
}

impl From<RehydrateCompletion> for StructuredResolveFailure {
    fn from(completion: RehydrateCompletion) -> Self {
        match completion {
            RehydrateCompletion::Refused => Self::Refused,
            RehydrateCompletion::Cancelled => Self::Cancelled,
            RehydrateCompletion::Panicked => Self::Panicked,
            RehydrateCompletion::BuildFailed => Self::BuildFailed,
            RehydrateCompletion::Released => Self::Released,
            RehydrateCompletion::Reserved | RehydrateCompletion::Queued
            | RehydrateCompletion::Succeeded { .. } => Self::RestorationUnavailable,
        }
    }
}

/// Retain one point-in-time bind decision. This takes no coordinator receipt, starts no worker,
/// and submits no work. An absent generation keeps only its exact installed eviction identity.
fn classify_retained<const R: usize>(original: usize, pending: &Arc<Pending>) -> StructuredResolve<R> {
    let done = pending.build.done.lock().unwrap();
    let raw = pending.build.real.load(Ordering::Acquire);
    let fallback = pending.stand(original);
    if !*done {
        return StructuredResolve::initial(original, pending.clone(), fallback, true);
    }
    let Some(recipe) = pending.build.recipe.as_ref() else {
        return match pending.build.pin_one_shot(raw) {
            Some(pin) => StructuredResolve::Ready {
                original, pending: pending.clone(), raw, pin: Some(pin),
            },
            None => StructuredResolve::initial(original, pending.clone(), fallback, false),
        };
    };
    let (pin, residency) = recipe.pin_with_residency(raw);
    match (pin, residency) {
        (Some(pin), Residency::Resident | Residency::Rebuilding) => {
            StructuredResolve::Ready {
                original, pending: pending.clone(), raw, pin: Some(pin),
            }
        }
        (_, Residency::Failed) => match pending.build.eviction_token() {
            Some(evicted) => StructuredResolve::Deferred {
                original, pending: pending.clone(), fallback,
                restoration: Arc::new(RestorationDemand::new(evicted)),
            },
            None => StructuredResolve::initial(original, pending.clone(), fallback, false),
        },
        _ => match pending.build.eviction_token() {
            Some(evicted) => StructuredResolve::Deferred {
                original, pending: pending.clone(), fallback,
                restoration: Arc::new(RestorationDemand::new(evicted)),
            },
            None => StructuredResolve::Terminal {
                original, pending: pending.clone(), fallback,
                failure: StructuredResolveFailure::MissingEvictionRecord,
            },
        },
    }
}

/// Classify one retained Pending without running driver work or waiting. The caller supplies the
/// private coordinator and queue so this foundation neither creates a live policy singleton nor
/// starts workers. No Build, recipe, registry, coordinator, or queue lock overlaps submission.
#[allow(dead_code)]
fn classify_structured_on<const E: usize, const R: usize>(
    original: usize,
    pending: &Arc<Pending>,
    coordinator: &RestorationCoordinator<E, R>,
    q: &BuildQueue,
) -> StructuredResolve<R> {
    enum Snapshot {
        Ready(usize, Option<NativeGeneration>),
        Initial(usize, bool),
        Failed(usize),
        Restore(usize),
    }
    let snapshot = || {
        let done = pending.build.done.lock().unwrap();
        let raw = pending.build.real.load(Ordering::Acquire);
        let fallback = pending.stand(original);
        if !*done {
            return Snapshot::Initial(fallback, true);
        }
        let Some(recipe) = pending.build.recipe.as_ref() else {
            return match pending.build.pin_one_shot(raw) {
                Some(pin) => Snapshot::Ready(raw, Some(pin)),
                None => Snapshot::Initial(fallback, false),
            };
        };
        let (pin, residency) = recipe.pin_with_residency(raw);
        match (pin, residency) {
            // A speculative replacement may mark the recipe Rebuilding while the old exact
            // resident generation remains published and fully usable.
            (Some(pin), Residency::Resident | Residency::Rebuilding) => Snapshot::Ready(raw, Some(pin)),
            (None, Residency::Evicted | Residency::Rebuilding) => Snapshot::Restore(fallback),
            (None, Residency::Failed) => Snapshot::Failed(fallback),
            _ => Snapshot::Restore(fallback),
        }
    };
    let fallback = match snapshot() {
        Snapshot::Ready(raw, pin) => return StructuredResolve::Ready {
            original, pending: pending.clone(), raw, pin,
        },
        Snapshot::Initial(fallback, bound_unbuilt) => {
            return StructuredResolve::initial(original, pending.clone(), fallback, bound_unbuilt);
        }
        Snapshot::Failed(fallback) if pending.build.eviction_token().is_none() => {
            return StructuredResolve::initial(original, pending.clone(), fallback, false);
        }
        Snapshot::Failed(fallback) => fallback,
        Snapshot::Restore(fallback) => fallback,
    };
    let Some(evicted) = pending.build.eviction_token() else {
        // Terminal reconciliation clears the token after publication. Reclassify once so that
        // race returns the new exact resident pin instead of a false missing-record failure.
        return match snapshot() {
            Snapshot::Ready(raw, pin) => StructuredResolve::Ready {
                original, pending: pending.clone(), raw, pin,
            },
            Snapshot::Initial(fallback, bound_unbuilt) => {
                StructuredResolve::initial(original, pending.clone(), fallback, bound_unbuilt)
            }
            Snapshot::Failed(fallback) => {
                StructuredResolve::initial(original, pending.clone(), fallback, false)
            }
            Snapshot::Restore(fallback) => StructuredResolve::Terminal {
                original, pending: pending.clone(), fallback,
                failure: StructuredResolveFailure::MissingEvictionRecord,
            },
        };
    };
    #[cfg(test)]
    let before_request = coordinator.before_classified_request.lock().unwrap().take();
    #[cfg(test)]
    if let Some(before_request) = before_request {
        before_request();
    }
    let requested = coordinator.request_exact_on(&evicted, RehydratePriority::Foreground, q);
    // Success may publish and reconcile after the token snapshot but before this request. Its
    // exact current pin wins over the stale request's Cancelled terminal state.
    if let Snapshot::Ready(raw, pin) = snapshot() {
        return StructuredResolve::Ready {
            original, pending: pending.clone(), raw, pin,
        };
    }
    match requested {
        Some(ticket) => StructuredResolve::Restoring {
            original, pending: pending.clone(), fallback, ticket,
        },
        None => StructuredResolve::Terminal {
            original, pending: pending.clone(), fallback,
            failure: StructuredResolveFailure::RestorationUnavailable,
        },
    }
}

fn resolve_and_pin_mode(pso: usize, mode: ResolveMode) -> (usize, Option<NativeGeneration>) {
    let Some(p) = entry(pso) else { return (pso, None) };
    resolve_and_pin_pending(pso, &p, mode)
}

fn resolve_and_pin_pending(pso: usize, p: &Arc<Pending>, mode: ResolveMode) -> (usize, Option<NativeGeneration>) {
    // Resolve directly against this retained Pending: a concurrent release/address reuse cannot
    // make selection consult a new registry entry while validation still uses the old recipe.
    let resolved = match mode {
        ResolveMode::Lazy => resolve_pending(p, pso),
        ResolveMode::Wait => {
            p.note_use();
            resolve_wait_pending(p, pso, wait_budget())
        }
        ResolveMode::ForBind if p.compute => {
            p.note_use();
            resolve_wait_pending(p, pso, wait_budget())
        }
        ResolveMode::ForBind => resolve_pending(p, pso),
    };
    let bound_unbuilt = p.bound_unbuilt(pso, resolved);
    let (resolved, generation, _) = pin_resolved(pso, p, resolved, bound_unbuilt);
    (resolved, generation)
}

fn pin_resolved(pso: usize, p: &Pending, resolved: usize, bound_unbuilt: bool) -> (usize, Option<NativeGeneration>, bool) {
    let recipe = p.build.recipe.as_ref();
    if recipe.is_some() {
        // The resolver already published this use's timestamp. Queue only the weak identity here
        // so exact pinning does not perform a second timestamp writer round trip.
        note_eviction_candidate_identity(&p.build);
    }
    let _done = p.build.done.lock().unwrap();
    let current = p.build.real.load(Ordering::Acquire);
    if current == resolved {
        return match recipe {
            Some(recipe) => {
                let (generation, residency) = recipe.pin_with_residency(resolved);
                let unavailable = generation.is_none() && matches!(residency, Residency::Evicted | Residency::Rebuilding);
                (resolved, generation, unavailable)
            }
            None => (resolved, p.build.pin_one_shot(resolved), false),
        };
    }
    // Preserve a resolver's deliberate stand-in choice. If an eventual rebuild replaced a
    // native generation between resolution and this lock, select and pin the current publication.
    // Eviction can instead leave no publication: fail closed to the retained stand-in, never the
    // ownerless raw pointer which resolution observed before it took this lock.
    if bound_unbuilt {
        return (resolved, None, false);
    }
    let Some(recipe) = recipe else {
        return match p.build.pin_one_shot(current) {
            Some(generation) => (current, Some(generation), false),
            None => {
                let stand = p.stand(pso);
                (stand, None, resolved != stand)
            }
        };
    };
    let (generation, residency) = recipe.pin_with_residency(current);
    match generation {
        Some(generation) => (current, Some(generation), false),
        None => {
            let stand = p.stand(pso);
            let unavailable = resolved != stand || matches!(residency, Residency::Evicted | Residency::Rebuilding);
            (stand, None, unavailable)
        }
    }
}

/// Preserve the ordinary resolver exactly unless recipe rebuilding was explicitly enabled.
pub(super) fn resolve_and_pin_if_enabled(pso: usize) -> (usize, Option<NativeGeneration>) {
    if rebuild_recipes_enabled() { resolve_and_pin_for_bind(pso) } else { (resolve_for_bind(pso), None) }
}

pub(super) fn resolve_and_pin_if_enabled_lazy(pso: usize) -> (usize, Option<NativeGeneration>) {
    if rebuild_recipes_enabled() { resolve_and_pin_mode(pso, ResolveMode::Lazy) } else { (resolve(pso), None) }
}

pub(super) fn resolve_and_pin_if_enabled_wait(pso: usize) -> (usize, Option<NativeGeneration>) {
    if rebuild_recipes_enabled() { resolve_and_pin_mode(pso, ResolveMode::Wait) } else { (resolve_wait(pso), None) }
}

pub(super) fn resolve_and_pin_if_enabled_identity(pso: usize) -> (usize, Option<NativeGeneration>) {
    (pso, None)
}

#[cfg(test)]
fn resolve_wait_bound_and_pin(pso: usize) -> ((usize, bool), Option<NativeGeneration>) {
    let Some(p) = entry(pso) else { return ((pso, false), None) };
    p.note_use();
    let bound = resolve_wait_pending(&p, pso, wait_budget());
    let choice = (bound, p.bound_unbuilt(pso, bound));
    let (resolved, generation, stale_fallback) = pin_resolved(pso, &p, choice.0, choice.1);
    ((resolved, choice.1 || stale_fallback), generation)
}

/// Resolve one retained Pending to an exactly owned generation without consulting its title
/// address. One-shot and recipe-backed builds share the same wait policy at this boundary.
fn resolve_retained_for_emission(pso: usize, p: &Arc<Pending>) -> PipelineChoice {
    if p.build.owners.load(Ordering::Acquire) == 0
        || p.build.orphan.load(Ordering::Acquire)
    {
        return PipelineChoice::Unavailable;
    }
    p.note_use();
    let bound = resolve_wait_pending(p, pso, wait_budget());
    let bound_unbuilt = p.bound_unbuilt(pso, bound);
    let (raw, generation, _) = pin_resolved(pso, p, bound, bound_unbuilt);
    match generation {
        Some(generation) => PipelineChoice::Available { raw, generation },
        None => PipelineChoice::Unavailable,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release when this is its last owner
/// (0: another `Pending` owns the build, none, or still being built – the worker's result is
/// then released when its last owner is gone).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.remove(pso) else { return 0 };
    let mut orphaned = false;
    let mut dropped_generation = None;
    let released = {
        let done = p.build.done.lock().unwrap();
        let owners = p.build.owners.fetch_sub(1, Ordering::Relaxed) - 1;
        if *done {
            if owners == 0 {
                dropped_generation = p.build.one_shot_generation.lock().unwrap().take();
                p.build.real.swap(0, Ordering::AcqRel)
            } else {
                0
            }
        } else {
            if owners == 0 {
                // the worker stores its result into a Build nobody will read: mark it orphaned
                p.build.orphan.store(true, Ordering::Release);
                p.build.finish_queued();
                orphaned = true;
            }
            0
        }
    };
    // COM destruction can re-enter pipeline bookkeeping; never drop this owner under `done`.
    drop(dropped_generation);
    if orphaned {
        forget_in_flight(&p.build);
        drop_unstarted_job(&p.build);
    }
    released
}

/// Whether a draw or dispatch with `pso` now has the pipeline it stands for: not a stand-in, or
/// one whose build succeeded (`put_off_line`'s skipped ones are those that had not).
pub(super) fn has_real(pso: usize) -> bool {
    entry(pso).is_none_or(|p| p.build.real.load(Ordering::Acquire) != 0)
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.build.done.lock().unwrap())
}

/// `pso` is a stand-in whose real pipeline is being built or was built, and `resolved`, what a
/// `resolve_wait` gave for it, is the stand-in (`pso` itself, or the driver pipeline a shared
/// stand-in binds: `shared`): the bind has the stand-in, also when the build finished since (a
/// failed build's stand-in is what draws for good). (`Pending::bound_unbuilt`, as
/// `resolve_wait_bound` answers it.)
pub fn gave_standin(pso: usize, resolved: usize) -> bool {
    entry(pso).is_some_and(|p| p.bound_unbuilt(pso, resolved))
}

/// Whether a dispatch with `pso` would wait for its build now (`resolve_wait`): a pipeline still
/// being built that binds do not go on without – not one whose dispatches are left out while it
/// is built (`image_only`) or that a bind already waited for past its budget (`late`). Those go
/// on as they did, in the call, and their builds keep their place (not before the workers'
/// queue: `need_by_submit`).
pub fn waits_for(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !p.spared.load(Ordering::Acquire) && !p.late.load(Ordering::Acquire) && !*p.build.done.lock().unwrap())
}

/// A submit is about to wait for `pso` (one of its next put-off calls): it counts among the
/// build's waiters (`Build::want`) once, however often the submit says so, and leaves with
/// the wait for it when that runs out of its budget (`resolve_within`). Not for a pipeline the
/// submit will not wait for (late, or spared): nothing would take that count back, and its build
/// would stay wanted – ahead of the commit gate – until it is built.
pub fn want(pso: usize) {
    if let Some(p) = entry(pso) {
        want_ahead(&p);
    }
}

/// (`want`) Whether this call counted the submit among the waiters.
fn want_ahead(p: &Pending) -> bool {
    {
        // (under the lock: a wait that makes it late takes `ahead` back after it, `resolve_within`)
        let mut d = p.build.demand.lock().unwrap();
        if p.late.load(Ordering::Acquire) || p.spared.load(Ordering::Acquire) || std::mem::replace(&mut d.ahead, true) {
            return false;
        }
        p.build.add_waiter(&mut d);
    }
    p.build.wanted_now();
    true
}

/// A call with `pso` was put off to the submit, which will wait for it: its shared build is
/// wanted from record time (counted once, as `want`) and goes before the workers' queue.
pub fn need_by_submit(pso: usize) {
    let Some(p) = entry(pso) else { return };
    need_pending_by_submit(&p);
}

fn need_pending_by_submit(p: &Arc<Pending>) {
    if p.build.by_submit.load(Ordering::Acquire) {
        return;
    }
    want_ahead(p);
    schedule(&p.build, Priority::BoundSoon, false, None);
    // True means all record-time demand work above is complete. Concurrent first callers may
    // repeat the idempotent promotion/schedule, but no caller observes publication too early.
    if !p.build.by_submit.swap(true, Ordering::AcqRel) {
        NEEDED_MADE.fetch_add(1, Ordering::Relaxed);
    }
}

/// What stand-in `pso` is (being) built from, for a diagnostic line (`dred::mark_pipeline`: a
/// pipeline bound with no label of its own). None: not a stand-in.
pub fn describe(pso: usize) -> Option<String> {
    let p = entry(pso)?;
    let state = if p.build.real.load(Ordering::Acquire) != 0 { "" } else { ", not built yet" };
    Some(match (p.compute, p.build.record) {
        (true, 0) => format!("a compute pipeline {pso:#x}{state}"),
        (true, record) => format!("cs {record:016x}{state}"),
        (false, _) => format!("a graphics pipeline {pso:#x} (shaders {:016x}){state}", p.key),
    })
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), QUEUED_BUILDS.load(Ordering::Relaxed))
}

#[cfg(test)]
mod backlog_tests {
    use super::*;

    /// A graphics pipeline for the backlog (`warm`: with its translation on its own), never built;
    /// `key` is its `record` (what `drain` returns).
    fn pending(key: u64, warm: bool) -> Arc<Build> {
        Arc::new(Build::new_keyed_kind(Box::new(|| 0), false, None, None, PipelineKind::Graphics)
            .with_submit(true, key, warm.then(|| Box::new(|| {}) as Warm)))
    }

    /// A compute pipeline for the backlog, never built.
    fn pending_compute(key: u64) -> Arc<Build> {
        Arc::new(Build::new_keyed_kind(Box::new(|| 0), false, None, None, PipelineKind::Compute)
            .with_submit(true, key, None))
    }

    /// A backlog worker's build nobody waits for builds the pieces of its dispatch chain itself:
    /// none of them goes to the workers whose builds binds wait for.
    #[test]
    fn a_build_ahead_keeps_its_pieces_on_its_own_thread() {
        let p = pending_compute(0x5EED_C1A1);
        p.demand.lock().unwrap().preload_thread = 1;
        let before = BUILDING.with(|b| b.replace(Some(p.clone())));
        let here = std::thread::current().id();
        let jobs: Vec<Job> = (0..6)
            .map(|_| {
                Box::new(move || {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    usize::from(std::thread::current().id() == here)
                }) as Job
            })
            .collect();
        let on_here = build_pieces(jobs);
        BUILDING.with(|b| *b.borrow_mut() = before);
        assert_eq!(on_here, vec![1; 6], "every piece on the backlog worker's thread");
    }

    /// A piece whose build panics fails (0), on the caller's thread or a helper's, and the
    /// caller still gets every other piece.
    #[test]
    fn a_piece_that_panics_fails_and_the_others_are_built() {
        let jobs: Vec<Job> = (0..8usize)
            .map(|i| {
                Box::new(move || {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    if i % 2 == 1 {
                        panic!("piece {i}");
                    }
                    i + 1
                }) as Job
            })
            .collect();
        assert_eq!(build_pieces(jobs), [1, 0, 3, 0, 5, 0, 7, 0]);
    }

    /// Work from the backlog as its workers do (a translated one goes back for its build): the
    /// record and step of each piece, in order.
    fn drain(b: &mut Backlog, ahead: bool, builds: bool) -> Vec<(u64, &'static str)> {
        let mut done = Vec::new();
        while let Some((p, step)) = b.next(ahead, builds) {
            match step {
                Step::Warm(w) => {
                    w();
                    done.push((p.record, "warm"));
                    b.cold.push_back(p);
                }
                Step::Build { ahead, new_compute } => {
                    p.job.lock().unwrap().take();
                    done.push((p.record, if ahead { "build ahead" } else { "build" }));
                    b.built(new_compute);
                }
            }
        }
        done
    }

    /// Every pipeline is translated before the driver builds any of them; one a bind wants is
    /// left to the workers.
    #[test]
    fn the_backlog_translates_every_new_pipeline_before_building_it() {
        let mut b = Backlog::default();
        let wanted = pending(4, true);
        wanted.sent.store(true, Ordering::Release);
        b.push(pending(2, true));
        b.push(wanted);
        b.push(pending(5, false));
        b.push(pending(3, true));
        assert_eq!(drain(&mut b, true, true), [(2, "warm"), (3, "warm"), (5, "build ahead"), (2, "build ahead"), (3, "build ahead")]);
        assert_eq!(b.len(), 0);
    }

    /// Switched off or short of commit (`ahead` false), nothing is worked on; past the cap
    /// (`builds` false) the pipelines are translated but not built. They wait.
    #[test]
    fn nothing_is_built_ahead_when_switched_off_or_short_of_commit() {
        let mut b = Backlog::default();
        b.push(pending(1, true));
        b.push(pending(2, false));
        assert!(drain(&mut b, false, true).is_empty());
        assert_eq!(b.len(), 2, "they wait");
        assert_eq!(drain(&mut b, true, false), [(1, "warm")]);
        assert_eq!(b.len(), 2);
        assert_eq!(drain(&mut b, true, true), [(2, "build ahead"), (1, "build ahead")]);
        assert_eq!(ahead_cap_of(None), 4096);
        assert_eq!(ahead_cap_of(Some(" 0 ")), 0, "XCL_PSO_AHEAD=0: off");
        assert_eq!(ahead_cap_of(Some("100")), 100);
        assert_eq!(ahead_cap_of(Some("lots")), 4096);
    }

    /// Compute pipelines have a lane of their own, before the graphics ones; they are built only
    /// with room (`ahead`) and under the cap (`builds`), and one a bind wants is left to the
    /// workers.
    #[test]
    fn new_compute_pipelines_are_built_before_graphics_ones() {
        let mut b = Backlog::default();
        let wanted = pending_compute(6);
        wanted.sent.store(true, Ordering::Release);
        b.push(pending(1, true));
        b.push(pending_compute(3));
        b.push(wanted);
        b.push(pending_compute(5));
        assert_eq!(drain(&mut b, true, true), [(3, "build ahead"), (5, "build ahead"), (1, "warm"), (1, "build ahead")]);
        assert_eq!(b.len(), 0);
        b.push(pending_compute(3));
        b.push(pending(2, false));
        assert!(drain(&mut b, false, true).is_empty());
        assert_eq!(b.len(), 2, "short of commit: they wait");
        assert!(drain(&mut b, true, false).is_empty());
        assert_eq!(b.len(), 2, "past the cap: they wait");
        assert_eq!(drain(&mut b, true, true), [(3, "build ahead"), (2, "build ahead")]);
    }

    /// While a worker builds a new compute pipeline, the other workers get the graphics ones, not
    /// the next new compute one.
    #[test]
    fn one_worker_at_a_time_builds_new_compute_pipelines() {
        let mut b = Backlog::default();
        b.push(pending_compute(3));
        b.push(pending_compute(4));
        b.push(pending(1, false));
        let take = |b: &mut Backlog| b.next(true, true).map(|(p, step)| (p.record, matches!(step, Step::Build { new_compute: true, .. })));
        assert_eq!(take(&mut b), Some((3, true)));
        assert_eq!(take(&mut b), Some((1, false)), "a graphics one while the new compute one is built");
        assert_eq!(take(&mut b), None, "the next new compute one waits for the one being built");
        b.built(false);
        assert_eq!(take(&mut b), None);
        b.built(true);
        assert_eq!(take(&mut b), Some((4, true)));
    }

    /// A pipeline binds go on without (`image_only`) goes to the backlog at its first bind, not to
    /// the workers builds a bind waits for go to, and is built in the lane of new compute ones:
    /// before them and one at a time with them, also short of commit and past the cap. Its place
    /// in a lane from when it was made is passed over.
    #[test]
    fn a_pipeline_binds_go_on_without_is_built_by_the_backlog_one_at_a_time() {
        let q = Preload { list: Mutex::new(Backlog::default()), cv: Condvar::new() };
        let spared = pending_compute(7);
        {
            let mut b = q.list.lock().unwrap();
            b.push(pending_compute(3));
            b.push(spared.clone());
            b.push(pending(2, false));
        }
        // (as `resolve_spared` does)
        spared.sent.store(true, Ordering::Release);
        send_spared(&spared, Some(&q));
        let mut b = q.list.lock().unwrap();
        let mut take = |ahead: bool| {
            b.next(ahead, ahead).map(|(p, step)| {
                p.job.lock().unwrap().take();
                let Step::Build { ahead, new_compute } = step else { panic!("{} was not built", p.record) };
                (p.record, ahead, new_compute)
            })
        };
        assert_eq!(take(false), Some((7, false, true)), "short of commit and past the cap it is built, in the new compute lane");
        assert_eq!(take(true), Some((2, true, false)), "a graphics one while it is built");
        assert_eq!(take(true), None, "the next new compute one waits for it");
        b.built(true);
        assert_eq!(b.next(true, true).map(|(p, _)| p.record), Some(3));
        b.built(true);
        assert!(b.next(true, true).is_none(), "its place from when it was made is passed over");
        assert_eq!(b.len(), 0);
    }

    /// Bound under a wait budget, a pipeline binds go on without is built by a backlog worker,
    /// not by the workers builds a bind waits for go to.
    #[test]
    fn a_pipeline_binds_go_on_without_is_not_built_by_the_workers_binds_wait_for() {
        if preload_backlog().is_none() {
            crate::gpu::recomp::runner::skip_test("no backlog (XCL_PSO_PRELOAD=0)");
            return;
        }
        let standin = 0x7E57_B0A3usize;
        let (tx, rx) = mpsc::channel();
        submit_lazy(standin, true, Box::new(move || {
            let _ = tx.send(std::thread::current().name().map(str::to_owned));
            0x5C5C
        }), 0);
        image_only(standin);
        assert_eq!(resolve_wait_for(standin, Some(std::time::Duration::from_millis(30))), standin, "still being built: the stand-in");
        let on = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it");
        assert!(on.as_deref().is_some_and(|n| n.starts_with("xcl-pso-ahead-")), "built on {on:?}");
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, None), 0x5C5C);
        let _ = released(standin);
    }

    /// A lazy compute pipeline of shaders no run used yet is built with no bind – when commit is
    /// free.
    #[test]
    fn a_lazy_compute_pipeline_of_new_shaders_is_built_ahead_of_its_first_use() {
        let standin = 0x7E57_C011usize;
        let room = ahead_room() && preload_backlog().is_some();
        submit_lazy(standin, true, Box::new(|| 0x6969), 0x5EED_C011);
        let start = std::time::Instant::now();
        let limit = std::time::Duration::from_secs(if room { 10 } else { 1 });
        while is_pending(standin) && start.elapsed() < limit {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        if room {
            assert!(!is_pending(standin), "built with no bind");
            assert_eq!(resolve(standin), 0x6969);
        } else {
            assert!(is_pending(standin), "short of commit: not built ahead");
        }
        released(standin);
    }

    /// The names of the backlog's workers alive now (`spawn_backlog_worker`).
    fn preload_workers() -> Vec<String> {
        use windows_sys::Win32::Foundation::{CloseHandle, LocalFree, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
        use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetThreadDescription, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
        let mut names = Vec::new();
        unsafe {
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            assert_ne!(snap, INVALID_HANDLE_VALUE, "no thread snapshot");
            let mut te: THREADENTRY32 = std::mem::zeroed();
            te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
            let mut ok = Thread32First(snap, &mut te) != 0;
            while ok {
                let h = if te.th32OwnerProcessID == GetCurrentProcessId() { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, te.th32ThreadID) } else { std::ptr::null_mut() };
                if !h.is_null() {
                    let mut p: *mut u16 = std::ptr::null_mut();
                    if GetThreadDescription(h, &mut p) >= 0 && !p.is_null() {
                        let n = (0..).take_while(|&i| *p.add(i) != 0).count();
                        let name = String::from_utf16_lossy(std::slice::from_raw_parts(p, n));
                        LocalFree(p.cast());
                        if name.starts_with("xcl-pso-ahead-") {
                            names.push(name);
                        }
                    }
                    CloseHandle(h);
                }
                ok = Thread32Next(snap, &mut te) != 0;
            }
            CloseHandle(snap);
        }
        names
    }

    /// A giant build ahead on a backlog worker waits for its pipeline's first use
    /// (`real::huge_compile_turn`) and does not stop the lane of new compute pipelines meanwhile:
    /// the next one is built while it waits, by the worker that took the waiting one's place
    /// (`build_parks`). Once a bind waits for it, it gets the giant slot and is built; its thread
    /// then ends without giving the lane back a second time (it holds another worker's build), and
    /// as many workers are left as before. Without that a pipeline the title never uses held the
    /// lane for good.
    #[test]
    fn a_parked_giant_build_ahead_leaves_the_lane_of_new_compute_pipelines_to_the_next() {
        use std::time::{Duration, Instant};
        let until = |done: &dyn Fn() -> bool, limit: Duration| {
            let start = Instant::now();
            while !done() && start.elapsed() < limit {
                std::thread::sleep(Duration::from_millis(5));
            }
            done()
        };
        if preload_backlog().is_none() || !ahead_room() {
            crate::gpu::recomp::runner::skip_test("no build ahead (XCL_PSO_PRELOAD=0, XCL_PSO_AHEAD=0 or short of commit)");
            return;
        }
        // (as `preload_backlog` starts them; a thread names itself once it runs)
        let workers = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        assert!(until(&|| preload_workers().len() == workers, Duration::from_secs(10)), "the backlog's workers: {:?}", preload_workers());
        let (giant, next, holder, last) = (0x7E57_C030usize, 0x7E57_C031usize, 0x7E57_C032usize, 0x7E57_C033usize);
        // the giant build: its worker's name, then its turn at the giant slot
        let (on_tx, on) = mpsc::channel();
        let (through_tx, through) = mpsc::channel();
        submit_lazy(giant, true, Box::new(move || {
            let _ = on_tx.send(std::thread::current().name().map(str::to_owned));
            let slot = super::super::real::huge_compile_turn(super::super::real::GIANT_DRIVER_COMPILE);
            let _ = through_tx.send(());
            drop(slot);
            0x6B6B
        }), 0x5EED_C030);
        let parked = on.recv_timeout(Duration::from_secs(10)).expect("a backlog worker builds it ahead").unwrap_or_default();
        assert!(parked.starts_with("xcl-pso-ahead-"), "built on {parked:?}");
        submit_lazy(next, true, Box::new(|| 0x6C6C), 0x5EED_C031);
        assert!(until(&|| !is_pending(next), Duration::from_secs(10)), "the next new compute pipeline is not built while the giant one waits");
        assert!(is_pending(giant), "nobody used the giant one: it is not built");
        assert!(through.try_recv().is_err(), "a giant build ahead took its turn");
        assert!(until(&|| preload_workers().len() == workers + 1, Duration::from_secs(10)), "another worker took the waiting one's place: {:?}", preload_workers());
        // a new compute build the lane holds when the giant build ends
        let (held_tx, held) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(holder, true, Box::new(move || {
            let _ = held_tx.send(());
            let _ = finish.recv_timeout(Duration::from_secs(20));
            0x6D6D
        }), 0x5EED_C032);
        held.recv_timeout(Duration::from_secs(10)).expect("the lane's next build starts");
        assert_eq!(resolve_wait_for(giant, None), 0x6B6B, "wanted, the giant build is made");
        assert!(through.try_recv().is_ok(), "it went through the giant slot");
        assert!(until(&|| !preload_workers().contains(&parked), Duration::from_secs(10)), "the waiting worker's thread did not end");
        assert_eq!(preload_workers().len(), workers, "as many workers as before: {:?}", preload_workers());
        // the lane is still the holder's: the giant build's end did not give it back again
        submit_lazy(last, true, Box::new(|| 0x6E6E), 0x5EED_C033);
        std::thread::sleep(Duration::from_millis(300));
        assert!(is_pending(last), "a second new compute build went while one was out");
        go.send(()).unwrap();
        assert!(until(&|| !is_pending(last), Duration::from_secs(10)), "the lane went back after the holder's build");
        assert_eq!(resolve(next), 0x6C6C);
        assert_eq!(resolve(holder), 0x6D6D);
        assert_eq!(resolve(last), 0x6E6E);
        for standin in [giant, next, holder, last] {
            released(standin);
        }
    }
}
