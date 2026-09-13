//! Run-scoped worker pools, progress, and cooperative interruption.
use crate::{Result, RunReport, SupportedModuleReport};
pub use refeff_core::execution::{CancellationToken, Control, Interrupted};
use std::{cell::RefCell, sync::Arc};

/// Create a calculation workspace, honoring WASI's explicitly mounted scratch root.
/// Rust's `std::env::temp_dir` panics on WASI even when `TMPDIR` is set.
#[doc(hidden)]
pub fn temporary_workspace(prefix: &str) -> std::io::Result<tempfile::TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.prefix(prefix);
    #[cfg(target_os = "wasi")]
    {
        let root = std::env::var_os("TMPDIR")
            .filter(|path| !path.is_empty())
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
        builder.tempdir_in(root)
    }
    #[cfg(not(target_os = "wasi"))]
    builder.tempdir()
}

/// An event emitted at an actual scheduler boundary.
#[derive(Clone, Debug)]
pub enum Event {
    /// Files owned by a reused stage, relative to its calculation directory.
    Artifacts {
        root: std::path::PathBuf,
        paths: Vec<std::path::PathBuf>,
    },
    Diagnostic {
        code: &'static str,
        module: &'static str,
        message: String,
    },
    Spectrum(Spectrum),
    Paths(Arc<refeff_io::PathsDatData>),
    /// Preparation or execution of this stage is beginning.
    StageStarted(&'static str),
    /// Completed iterations or energy points within a running stage.
    Advanced {
        name: &'static str,
        completed: usize,
        total: usize,
    },
    /// This stage has finished successfully.
    StageCompleted(SupportedModuleReport),
}
/// Callback invoked synchronously on the calculation's coordinator.
pub type Observer = Arc<dyn Fn(Event) + Send + Sync>;
/// Per-calculation controls. No global Rayon initialization is performed.
#[derive(Clone, Default)]
pub struct ExecutionOptions {
    pub threads: Option<usize>,
    pub control: Control,
    pub observer: Option<Observer>,
    /// Capture final chi/xmu values from this output directory.
    pub spectrum_root: Option<std::path::PathBuf>,
    /// Omit final chi/xmu text only when no later stage consumes it.
    pub omit_final_spectra: bool,
}
thread_local! { static OBSERVER: RefCell<Option<Observer>> = RefCell::new(None); }
pub(crate) fn start(name: &'static str) -> Result<()> {
    refeff_core::execution::checkpoint()?;
    let name = crate::ModuleName::parse(name).map_or(name, crate::ModuleName::as_str);
    emit(Event::StageStarted(name));
    // A callback may itself request cancellation.
    refeff_core::execution::checkpoint()?;
    Ok(())
}
pub(crate) fn emit(event: Event) {
    let observer = OBSERVER.with(|observer| observer.borrow().clone());
    if let Some(observer) = observer {
        observer(event);
    }
}

/// Collect cache-owned files while forwarding every event to the caller.
pub(crate) fn collect_declared_artifacts<T>(
    root: &std::path::Path,
    run: impl FnOnce() -> Result<T>,
) -> Result<(T, Vec<std::path::PathBuf>)> {
    let files = Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new()));
    let received = files.clone();
    let root = root.to_path_buf();
    let previous = OBSERVER.with(|observer| observer.borrow().clone());
    let forward = previous.clone();
    let observer: Observer = Arc::new(move |event| {
        if let Event::Artifacts {
            root: directory,
            paths,
        } = &event
        {
            if *directory == root {
                received
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .extend(paths.iter().cloned());
            }
        }
        if let Some(forward) = &forward {
            forward(event);
        }
    });
    struct Restore(Option<Observer>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OBSERVER.with(|observer| *observer.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(previous);
    OBSERVER.with(|current| *current.borrow_mut() = Some(observer));
    let result = run()?;
    let files = files
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .cloned()
        .collect();
    Ok((result, files))
}

pub(crate) fn captures_spectra() -> bool {
    SPECTRA.with(|settings| settings.borrow().0.is_some())
}
/// Execute with scoped settings, using an owned worker pool on native targets.
/// WebAssembly runs on the current thread; requested thread counts are clamped to one.
pub fn with_execution<T: Send>(
    options: &ExecutionOptions,
    run: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    options.control.check()?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let control = options.control.clone();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.threads.unwrap_or(0))
            .start_handler(move |_| refeff_core::execution::install_worker(control.clone()))
            .build()?;
        refeff_linalg::with_parallelism(pool.current_num_threads(), || {
            pool.install(|| with_settings(options, run))
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        refeff_linalg::with_parallelism(1, || {
            refeff_core::execution::with_control(options.control.clone(), || {
                with_settings(options, run)
            })
        })
    }
}

fn with_settings<T>(options: &ExecutionOptions, run: impl FnOnce() -> Result<T>) -> Result<T> {
    struct Restore(Option<Observer>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OBSERVER.with(|current| *current.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(OBSERVER.with(|current| current.replace(options.observer.clone())));
    struct RestoreSpectra((Option<std::path::PathBuf>, bool));
    impl Drop for RestoreSpectra {
        fn drop(&mut self) {
            SPECTRA.with(|settings| *settings.borrow_mut() = self.0.clone());
        }
    }
    let _spectra = RestoreSpectra(SPECTRA.with(|settings| {
        settings.replace((options.spectrum_root.clone(), options.omit_final_spectra))
    }));
    options.control.check()?;
    run()
}
/// Execute the scheduler using isolated settings and live events.
pub fn execute_with_options(
    input: &std::path::Path,
    output: &std::path::Path,
    options: &ExecutionOptions,
) -> Result<RunReport> {
    with_execution(options, || crate::execute_feff(input, output))
}

thread_local! { static ACTIONS: RefCell<std::collections::BTreeMap<&'static str, crate::StageStatus>> = const { RefCell::new(std::collections::BTreeMap::new()) }; }
pub(crate) fn record_action(stage: &'static str, action: crate::StageStatus) {
    ACTIONS.with(|actions| actions.borrow_mut().insert(stage, action));
}
pub(crate) fn take_action(stage: &str) -> Option<crate::StageStatus> {
    ACTIONS.with(|actions| actions.borrow_mut().remove(stage))
}

pub(crate) fn advance(name: &'static str, completed: usize, total: usize) -> Result<()> {
    refeff_core::execution::checkpoint()?;
    emit(Event::Advanced {
        name,
        completed,
        total,
    });
    refeff_core::execution::checkpoint()?;
    Ok(())
}

/// Typed final spectrum transported independently of FEFF text serialization.
#[derive(Clone, Debug)]
pub enum Spectrum {
    Chi(Arc<refeff_io::ChiDatData>),
    Xmu(Arc<refeff_io::XmuDatData>),
}
thread_local! { static SPECTRA: RefCell<(Option<std::path::PathBuf>, bool)> = const { RefCell::new((None, false)) }; }
pub(crate) fn retain_chi(path: &std::path::Path, data: &refeff_io::ChiDatData) -> bool {
    retain_spectrum(path, "chi.dat", || Spectrum::Chi(Arc::new(data.clone())))
}
pub(crate) fn retain_xmu(path: &std::path::Path, data: &refeff_io::XmuDatData) -> bool {
    retain_spectrum(path, "xmu.dat", || Spectrum::Xmu(Arc::new(data.clone())))
}
pub(crate) fn retain_paths(path: &std::path::Path, data: refeff_io::PathsDatData) {
    if SPECTRA.with(|settings| {
        settings
            .borrow()
            .0
            .as_ref()
            .is_some_and(|root| root.join("paths.dat") == path)
    }) {
        emit(Event::Paths(Arc::new(data)));
    }
}
fn retain_spectrum(
    path: &std::path::Path,
    name: &str,
    spectrum: impl FnOnce() -> Spectrum,
) -> bool {
    let (root, omit) = SPECTRA.with(|settings| settings.borrow().clone());
    if root.is_some_and(|root| root.join(name) == path) {
        emit(Event::Spectrum(spectrum()));
        return omit;
    }
    false
}
