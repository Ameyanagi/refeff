#![forbid(unsafe_code)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

//! Typed programmatic facade for the FEFF10-compatible Rust pipeline.
//!
//! [`Runner::run_files`] is the stable file-backed entry point. Numerical
//! kernels and format codecs remain available from `refeff-core` and
//! `refeff-io`; CLI parsing is deliberately absent from this API.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub use refeff_core as core;
pub use refeff_core::execution::{CancellationToken, Interrupted};
pub use refeff_io as io;
pub use refeff_io::codec::{
    FeffCodec, FileFormat, FormatDescriptor, NumericTolerance, Representation, identify_format,
};

/// Result alias for facade operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by the typed runner boundary.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A request contains an invalid path or conflicting policy.
    #[error("invalid run request: {0}")]
    InvalidRequest(String),
    /// Existing output is forbidden by the selected policy.
    #[error("output directory {path} is not empty")]
    OutputConflict {
        /// Conflicting output directory.
        path: PathBuf,
    },
    /// Filesystem operation failed.
    #[error("I/O operation failed for {path}: {source}")]
    Io {
        /// File or directory being accessed.
        path: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: std::io::Error,
    },
    /// The current file-backed engine failed.
    #[error("FEFF pipeline failed: {message}")]
    Engine {
        /// Context-rich engine error rendered without exposing `anyhow` in
        /// the public facade.
        message: String,
    },
    /// Structured pipeline failure, including completed work and its original cause.
    #[error("{code}: {source}")]
    Pipeline {
        code: &'static str,
        module: Option<String>,
        stages: Vec<StageReport>,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The requested calculation is unavailable in this Cargo feature set.
    #[error("FEFF module `{module}` requires Cargo feature `{feature}`")]
    FeatureDisabled {
        /// Canonical FEFF module name.
        module: &'static str,
        /// Cargo feature that enables it.
        feature: &'static str,
    },
    /// An artifact name is empty, absolute, or could escape its workspace.
    #[error("invalid artifact path {path}")]
    InvalidArtifactPath {
        /// Rejected path.
        path: PathBuf,
    },
}

/// FEFF pipeline stages exposed to programmatic callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Module {
    /// Input parsing and handoff generation.
    Rdinp,
    /// Atomic potentials and wavefunctions.
    Atomic,
    /// Self-consistent muffin-tin potentials.
    Pot,
    /// Local density of states.
    Ldos,
    /// Core-hole screening.
    Screen,
    /// Constrained-RPA response.
    Crpa,
    /// Optical constants database stage.
    Opcons,
    /// Phase shifts and cross sections.
    Xsph,
    /// Full multiple-scattering solve.
    Fms,
    /// Green's-function trace projection.
    Mkgtr,
    /// Scattering path search.
    Path,
    /// Path amplitude generation.
    Genfmt,
    /// Final spectrum assembly.
    Ff2x,
    /// Spectral-function convolution.
    Sfconv,
    /// Compton profiles.
    Compton,
    /// Electron energy-loss spectra.
    Eels,
    /// EELS mixed dynamic form factor.
    EelsMdff,
    /// Charge-density output.
    Rhorrp,
    /// Dynamical-matrix Debye-Waller calculation.
    Dmdw,
    /// Band-structure calculation.
    Band,
    /// Full-spectrum optical constants.
    FullSpectrum,
    /// Resonant inelastic X-ray scattering.
    Rixs,
    /// On-shell self-energy calculation.
    SelfEnergy,
    /// Potential text rendering.
    Wpot,
}

impl Module {
    /// Canonical FEFF-compatible stage name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rdinp => refeff_engine::ModuleName::Rdinp,
            Self::Atomic => refeff_engine::ModuleName::Atomic,
            Self::Pot => refeff_engine::ModuleName::Pot,
            Self::Ldos => refeff_engine::ModuleName::Ldos,
            Self::Screen => refeff_engine::ModuleName::Screen,
            Self::Crpa => refeff_engine::ModuleName::Crpa,
            Self::Opcons => refeff_engine::ModuleName::Opcons,
            Self::Xsph => refeff_engine::ModuleName::Xsph,
            Self::Fms => refeff_engine::ModuleName::Fms,
            Self::Mkgtr => refeff_engine::ModuleName::Mkgtr,
            Self::Path => refeff_engine::ModuleName::Path,
            Self::Genfmt => refeff_engine::ModuleName::Genfmt,
            Self::Ff2x => refeff_engine::ModuleName::Ff2x,
            Self::Sfconv => refeff_engine::ModuleName::Sfconv,
            Self::Compton => refeff_engine::ModuleName::Compton,
            Self::Eels => refeff_engine::ModuleName::Eels,
            Self::EelsMdff => refeff_engine::ModuleName::Mdff,
            Self::Rhorrp => refeff_engine::ModuleName::Rhorrp,
            Self::Dmdw => refeff_engine::ModuleName::Dmdw,
            Self::Band => refeff_engine::ModuleName::Band,
            Self::FullSpectrum => refeff_engine::ModuleName::Fullspectrum,
            Self::Rixs => refeff_engine::ModuleName::Rixs,
            Self::SelfEnergy => refeff_engine::ModuleName::SelfEnergy,
            Self::Wpot => refeff_engine::ModuleName::Wpot,
        }
        .as_str()
    }
}

impl fmt::Display for Module {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Policy for files already present in a run's output directory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExistingOutputPolicy {
    /// Validate compatible artifacts and regenerate stale artifacts.
    #[default]
    ReuseValidated,
    /// Compute in a clean staging directory, then replace the entire output directory, including unrelated files.
    Recompute,
    /// Reject a non-empty output directory.
    ErrorOnConflict,
}

/// File-backed pipeline request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRunRequest {
    /// Root `feff.inp` file.
    pub input: PathBuf,
    /// Directory that receives FEFF-compatible outputs.
    pub output: PathBuf,
    /// Existing-output policy.
    pub existing_output_policy: ExistingOutputPolicy,
}

/// An owned collection of relative FEFF workspace files.
///
/// The collection is suitable both for auxiliary inputs (`spring.inp`, CIF,
/// DYM, or included card files) and for generated output. Paths are always
/// relative and cannot contain `.` or `..` components.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArtifactSet {
    files: BTreeMap<PathBuf, Vec<u8>>,
}

impl ArtifactSet {
    /// Create an empty workspace.
    pub const fn new() -> Self {
        Self {
            files: BTreeMap::new(),
        }
    }

    /// Insert or replace a file and return the previous payload, if any.
    pub fn insert(
        &mut self,
        path: impl Into<PathBuf>,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<Option<Vec<u8>>> {
        let path = path.into();
        validate_artifact_path(&path)?;
        Ok(self.files.insert(path, bytes.into()))
    }

    /// Read one file by its relative path.
    #[must_use]
    pub fn get(&self, path: impl AsRef<Path>) -> Option<&[u8]> {
        self.files.get(path.as_ref()).map(Vec::as_slice)
    }

    /// Remove one file by its relative path.
    pub fn remove(&mut self, path: impl AsRef<Path>) -> Option<Vec<u8>> {
        self.files.remove(path.as_ref())
    }

    /// Return whether the workspace contains a path.
    #[must_use]
    pub fn contains(&self, path: impl AsRef<Path>) -> bool {
        self.files.contains_key(path.as_ref())
    }

    /// Number of files in the workspace.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Return whether the workspace contains no files.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Iterate in stable path order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = ArtifactRef<'_>> {
        self.files.iter().map(|(path, bytes)| ArtifactRef {
            path,
            bytes,
            format: identify_format(path),
        })
    }
}

/// Borrowed view of one in-memory FEFF file.
#[derive(Debug, Clone, Copy)]
pub struct ArtifactRef<'a> {
    /// Workspace-relative filename.
    pub path: &'a Path,
    /// Complete file payload.
    pub bytes: &'a [u8],
    /// Registered FEFF format metadata, when the filename is known.
    pub format: Option<FormatDescriptor>,
}

/// Request for a file-compatible run backed by memory rather than a caller
/// managed directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRunRequest {
    /// Relative path of the root FEFF input inside `artifacts`.
    pub input: PathBuf,
    /// Initial workspace, including the root input and auxiliary files.
    pub artifacts: ArtifactSet,
}

impl MemoryRunRequest {
    /// Create a workspace containing `feff.inp`.
    pub fn new(input: impl Into<Vec<u8>>) -> Self {
        let mut artifacts = ArtifactSet::new();
        // This constant path is valid by construction.
        artifacts
            .files
            .insert(PathBuf::from("feff.inp"), input.into());
        Self {
            input: PathBuf::from("feff.inp"),
            artifacts,
        }
    }

    /// Use a different relative root-input name.
    pub fn with_input_name(mut self, input: impl Into<PathBuf>) -> Result<Self> {
        let input = input.into();
        validate_artifact_path(&input)?;
        let bytes = self
            .artifacts
            .remove("feff.inp")
            .ok_or_else(|| Error::InvalidRequest("memory request has no feff.inp".to_string()))?;
        self.artifacts.insert(&input, bytes)?;
        self.input = input;
        Ok(self)
    }

    /// Add or replace an auxiliary workspace file.
    pub fn insert_artifact(
        &mut self,
        path: impl Into<PathBuf>,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<Option<Vec<u8>>> {
        self.artifacts.insert(path, bytes)
    }
}

/// Result of an in-memory run.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRunResult {
    /// Typed final spectra, retained directly where possible.
    pub spectra: Spectra,
    /// Typed scattering paths, retained from PATH when that stage runs.
    pub paths: Option<io::PathsDatData>,
    /// Typed execution report. Its paths are workspace-relative.
    pub report: RunReport,
    /// Selected final workspace files, including inputs when requested.
    pub artifacts: ArtifactSet,
}

impl FileRunRequest {
    /// Create a request using validated cache reuse.
    pub fn new(input: impl Into<PathBuf>, output: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            existing_output_policy: ExistingOutputPolicy::default(),
        }
    }

    /// Select how existing output is handled.
    #[must_use]
    pub const fn with_existing_output_policy(mut self, policy: ExistingOutputPolicy) -> Self {
        self.existing_output_policy = policy;
        self
    }
}

/// Whether a completed stage reused or generated artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StageAction {
    /// Existing validated artifacts were reused.
    Reused,
    /// Artifacts were generated or repaired.
    Generated,
}

/// Report for one completed stage.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageReport {
    /// Stage name as emitted by the FEFF-compatible scheduler.
    pub name: String,
    /// Reuse or generation action.
    pub action: StageAction,
    /// Number of rows or artifacts handled.
    pub count: usize,
    /// Unit associated with `count`.
    pub unit: String,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Non-fatal diagnostic produced during a run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Diagnostic {
    /// Stable diagnostic code.
    pub code: String,
    /// Optional originating module.
    pub module: Option<Module>,
    /// Human-readable detail.
    pub message: String,
}

/// Typed summary of a completed file-backed run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RunReport {
    /// Input file used for the run.
    pub input: PathBuf,
    /// Output directory used for the run.
    pub output: PathBuf,
    /// Number of parsed cards.
    pub cards: usize,
    /// Number of expanded atoms.
    pub atoms: usize,
    /// Number of unique potentials.
    pub potentials: usize,
    /// Completed stage reports.
    pub stages: Vec<StageReport>,
    /// Relative paths of generated or retained artifacts.
    pub artifacts: Vec<PathBuf>,
    /// Non-fatal diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

/// Progress event sent to a library callback.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ProgressEvent<'a> {
    /// A run is starting.
    RunStarted(&'a FileRunRequest),
    /// An in-memory run is starting.
    MemoryRunStarted(&'a MemoryRunRequest),
    /// Preparation or execution of a stage has begun.
    StageStarted(&'a str),
    /// Live progress within an iterative stage.
    StageProgress {
        name: &'a str,
        completed: usize,
        total: usize,
    },
    /// A run failed or was interrupted.
    RunFailed(&'a Error),
    /// Cancellation or deadline stopped the calculation cooperatively.
    RunCancelled(&'a Error),
    /// A stage completed.
    StageCompleted(&'a StageReport),
    /// The run completed.
    RunCompleted(&'a RunReport),
}

/// Callback for observing long-running work without coupling the library to
/// a logging framework.
pub trait ProgressSink: Send + Sync {
    /// Observe one progress event.
    fn event(&self, event: ProgressEvent<'_>);
}

/// Configurable FEFF pipeline runner.
#[derive(Default)]
pub struct Runner {
    threads: Option<NonZeroUsize>,
    progress: Option<Arc<dyn ProgressSink>>,
    control: refeff_core::execution::Control,
    artifact_selection: ArtifactSelection,
}

impl Runner {
    /// Construct a runner using process defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bound the calculation's owned worker pool. ReFEFF faer calculations are serialized.
    /// WebAssembly runs on one thread regardless of this bound.
    #[must_use]
    pub fn with_threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Use a caller-owned cooperative cancellation handle.
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.control.cancellation = cancellation;
        self
    }
    /// Stop cooperatively once a monotonic deadline expires.
    pub fn with_deadline(mut self, deadline: std::time::Instant) -> Self {
        self.control.deadline = Some(deadline);
        self
    }
    /// Select returned memory artifacts before reading their payloads.
    pub fn with_artifacts(mut self, selection: ArtifactSelection) -> Self {
        self.artifact_selection = selection;
        self
    }
    /// Install a progress callback.
    #[must_use]
    pub fn with_progress_sink(mut self, sink: Arc<dyn ProgressSink>) -> Self {
        self.progress = Some(sink);
        self
    }

    /// Execute the FEFF-compatible file pipeline.
    pub fn run_files(&self, request: FileRunRequest) -> Result<RunReport> {
        self.finish(self.run_files_inner(request))
    }
    fn finish<T>(&self, result: Result<T>) -> Result<T> {
        if let (Err(error), Some(sink)) = (&result, &self.progress) {
            if matches!(
                error,
                Error::Pipeline {
                    code: "interrupted",
                    ..
                }
            ) {
                sink.event(ProgressEvent::RunCancelled(error));
            } else {
                sink.event(ProgressEvent::RunFailed(error));
            }
        }
        result
    }
    fn run_files_inner(&self, request: FileRunRequest) -> Result<RunReport> {
        validate_request(&request)?;
        if let Some(sink) = &self.progress {
            sink.event(ProgressEvent::RunStarted(&request));
        }

        let report = match request.existing_output_policy {
            ExistingOutputPolicy::ReuseValidated => {
                fs::create_dir_all(&request.output)
                    .map_err(|source| io_error(&request.output, source))?;
                self.run_engine(&request.input, &request.output)?
            }
            ExistingOutputPolicy::ErrorOnConflict => {
                ensure_empty_output(&request.output)?;
                fs::create_dir_all(&request.output)
                    .map_err(|source| io_error(&request.output, source))?;
                self.run_engine(&request.input, &request.output)?
            }
            ExistingOutputPolicy::Recompute => self.run_recomputed(&request)?,
        };

        if let Some(sink) = &self.progress {
            sink.event(ProgressEvent::RunCompleted(&report));
        }
        Ok(report)
    }

    /// Execute the same FEFF-compatible scheduler against an owned in-memory
    /// workspace.
    ///
    /// A private temporary directory is used only as the compatibility
    /// transport for legacy FEFF file formats. Callers neither manage that
    /// directory nor receive ephemeral paths in the result.
    pub fn run_in_memory(&self, request: MemoryRunRequest) -> Result<MemoryRunResult> {
        self.finish(self.run_in_memory_inner(request))
    }
    fn run_in_memory_inner(&self, request: MemoryRunRequest) -> Result<MemoryRunResult> {
        validate_memory_request(&request)?;
        if let Some(sink) = &self.progress {
            sink.event(ProgressEvent::MemoryRunStarted(&request));
        }

        let workspace = refeff_engine::execution::temporary_workspace("refeff-")
            .map_err(|source| io_error(Path::new("."), source))?;
        materialize_artifacts(&request.artifacts, workspace.path())?;
        let input = workspace.path().join(&request.input);
        let retained = Arc::new(std::sync::Mutex::new(RetainedOutputs::default()));
        let mut report =
            self.run_engine_capture(&input, workspace.path(), Some(retained.clone()))?;
        report.input = request.input.clone();
        report.output = PathBuf::from(".");
        let mut artifacts = ArtifactSet::new();
        // Reuse the report inventory; select before allocating payload buffers.
        for path in &report.artifacts {
            if self.artifact_selection.includes(path) {
                let full = workspace.path().join(path);
                artifacts.insert(
                    path,
                    fs::read(&full).map_err(|source| io_error(&full, source))?,
                )?;
            }
        }
        for artifact in request.artifacts.iter() {
            if self.artifact_selection.includes(artifact.path) && !artifacts.contains(artifact.path)
            {
                artifacts.insert(artifact.path, artifact.bytes)?;
            }
        }
        report.artifacts = artifacts.files.keys().cloned().collect();

        let retained = std::mem::take(
            &mut *retained
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        let mut spectra = retained.spectra;
        let mut paths = retained.paths;
        if paths.is_none()
            && report.stages.iter().any(|stage| stage.name == "path")
            && workspace.path().join("paths.dat").is_file()
        {
            paths = Some(
                io::read_paths_dat(workspace.path().join("paths.dat")).map_err(|source| {
                    Error::Pipeline {
                        code: "output",
                        module: Some("path".into()),
                        stages: report.stages.clone(),
                        source: Box::new(source),
                    }
                })?,
            );
        }
        // Convolution modifies the FF2X result; decode that later stage's final files.
        if report.stages.iter().any(|stage| stage.name == "sfconv") {
            spectra = Spectra::default();
        }
        if spectra.chi.is_none() && workspace.path().join("chi.dat").is_file() {
            spectra.chi = Some(
                io::chi_dat::read_chi_dat(workspace.path().join("chi.dat")).map_err(|source| {
                    Error::Pipeline {
                        code: "output",
                        module: Some("ff2x".into()),
                        stages: report.stages.clone(),
                        source: Box::new(source),
                    }
                })?,
            );
        }
        if spectra.xmu.is_none() && workspace.path().join("xmu.dat").is_file() {
            spectra.xmu = Some(
                io::xmu_dat::read_xmu_dat(workspace.path().join("xmu.dat")).map_err(|source| {
                    Error::Pipeline {
                        code: "output",
                        module: Some("ff2x".into()),
                        stages: report.stages.clone(),
                        source: Box::new(source),
                    }
                })?,
            );
        }
        if let Some(sink) = &self.progress {
            sink.event(ProgressEvent::RunCompleted(&report));
        }
        Ok(MemoryRunResult {
            report,
            artifacts,
            spectra,
            paths,
        })
    }

    fn run_recomputed(&self, request: &FileRunRequest) -> Result<RunReport> {
        let parent = request
            .output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|source| io_error(parent, source))?;
        let staging = tempfile::Builder::new()
            .prefix(".refeff-recompute-")
            .tempdir_in(parent)
            .map_err(|source| io_error(parent, source))?;
        let staged = self.run_engine(&request.input, staging.path())?;
        publish_recomputed(staging, &request.output)?;
        Ok(RunReport {
            input: request.input.clone(),
            output: request.output.clone(),
            ..staged
        })
    }
}

impl Runner {
    fn run_engine(&self, input: &Path, output: &Path) -> Result<RunReport> {
        self.run_engine_capture(input, output, None)
    }
    fn run_engine_capture(
        &self,
        input: &Path,
        output: &Path,
        spectra: Option<Arc<std::sync::Mutex<RetainedOutputs>>>,
    ) -> Result<RunReport> {
        let before = inventory(output)?;
        let declared_artifacts = Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new()));
        let received_artifacts = declared_artifacts.clone();
        let artifact_root = output.to_path_buf();
        let diagnostics = Arc::new(std::sync::Mutex::new(Vec::new()));
        let received_diagnostics = diagnostics.clone();
        let stages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let current = Arc::new(std::sync::Mutex::new(None));
        let (recorded, stage_name, sink) = (stages.clone(), current.clone(), self.progress.clone());
        let capture = spectra.is_some();
        // Ordinary EXAFS can terminate at typed FF2X outputs. Other workflows keep
        // their file transport until every downstream consumer has a typed input.
        let omit_final_spectra = capture
            && matches!(self.artifact_selection, ArtifactSelection::None)
            && io::FeffInput::parse_file(input)
                .and_then(|input| io::FeffDocument::from_input(&input))
                .is_ok_and(|doc| {
                    doc.ispec == 0
                        && !doc.sfconv
                        && !doc.eels.enabled
                        && !doc.rixs.run
                        && !doc.compton.do_compton
                        && doc.full_spectrum_input.m_full_spectrum == 0
                });
        let options = refeff_engine::execution::ExecutionOptions {
            spectrum_root: capture.then(|| output.to_path_buf()),
            omit_final_spectra,
            threads: self.threads.map(NonZeroUsize::get),
            control: self.control.clone(),
            observer: Some(Arc::new(move |event| match event {
                refeff_engine::execution::Event::Artifacts { root, paths } => {
                    if root == artifact_root {
                        received_artifacts
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .extend(paths);
                    }
                }
                refeff_engine::execution::Event::Diagnostic {
                    code,
                    module,
                    message,
                } => {
                    received_diagnostics
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(Diagnostic {
                            code: code.into(),
                            module: match module {
                                "path" => Some(Module::Path),
                                "genfmt" => Some(Module::Genfmt),
                                "ff2x" => Some(Module::Ff2x),
                                _ => None,
                            },
                            message,
                        });
                }
                refeff_engine::execution::Event::Spectrum(value) => {
                    if let Some(spectra) = &spectra {
                        let mut result = spectra
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        match value {
                            refeff_engine::execution::Spectrum::Chi(data) => {
                                result.spectra.chi = Some(Arc::unwrap_or_clone(data))
                            }
                            refeff_engine::execution::Spectrum::Xmu(data) => {
                                result.spectra.xmu = Some(Arc::unwrap_or_clone(data))
                            }
                        }
                    }
                }
                refeff_engine::execution::Event::Paths(data) => {
                    if let Some(spectra) = &spectra {
                        spectra
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .paths = Some(Arc::unwrap_or_clone(data));
                    }
                }
                refeff_engine::execution::Event::Advanced {
                    name,
                    completed,
                    total,
                } => {
                    if let Some(sink) = &sink {
                        sink.event(ProgressEvent::StageProgress {
                            name,
                            completed,
                            total,
                        });
                    }
                }
                refeff_engine::execution::Event::StageStarted(name) => {
                    *stage_name
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(name.to_owned());
                    if let Some(sink) = &sink {
                        sink.event(ProgressEvent::StageStarted(name));
                    }
                }
                refeff_engine::execution::Event::StageCompleted(stage) => {
                    let stage = stage_report(stage);
                    recorded
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(stage.clone());
                    if let Some(sink) = &sink {
                        sink.event(ProgressEvent::StageCompleted(&stage));
                    }
                }
            })),
        };
        let engine = refeff_engine::execution::execute_with_options(input, output, &options)
            .map_err(|error| {
                let code = if error
                    .chain()
                    .any(|source| source.downcast_ref::<Interrupted>().is_some())
                {
                    "interrupted"
                } else if let Some(input_error) = error.downcast_ref::<io::IoError>() {
                    if matches!(input_error, io::IoError::Io { .. }) {
                        "io"
                    } else {
                        "input"
                    }
                } else if error.downcast_ref::<refeff_engine::EngineError>().is_some() {
                    "feature_disabled"
                } else {
                    "pipeline"
                };
                let error = Error::Pipeline {
                    code,
                    module: current
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone(),
                    stages: stages
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone(),
                    source: error.into_boxed_dyn_error(),
                };
                error
            })?;
        let completed: Vec<_> = engine.stages.into_iter().map(stage_report).collect();
        let declared = declared_artifacts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let artifacts =
            inventory(output)?
                .into_iter()
                .filter(|(path, stamp)| {
                    let retained =
                        path.components().count() == 1
                            && identify_format(path).is_some_and(|format| {
                                format.producer == "rdinp"
                                    || completed.iter().any(|stage| {
                                        stage.name.strip_prefix(format.producer).is_some_and(
                                            |suffix| suffix.is_empty() || suffix.starts_with('-'),
                                        )
                                    })
                            });
                    retained || declared.contains(path) || before.get(path) != Some(stamp)
                })
                .map(|(path, _)| path)
                .collect();
        Ok(RunReport {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            cards: engine.rdinp.cards,
            atoms: engine.rdinp.atoms,
            potentials: engine.rdinp.potentials,
            stages: completed,
            artifacts,
            diagnostics: diagnostics
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
        })
    }
}
fn stage_report(stage: refeff_engine::SupportedModuleReport) -> StageReport {
    StageReport {
        name: stage.name.to_owned(),
        action: match stage.status {
            refeff_engine::StageStatus::Cached => StageAction::Reused,
            refeff_engine::StageStatus::Generated => StageAction::Generated,
        },
        count: stage.count,
        unit: stage.unit.to_owned(),
        duration_ms: stage.duration_ms,
    }
}

fn validate_request(request: &FileRunRequest) -> Result<()> {
    if !request.input.is_file() {
        return Err(Error::InvalidRequest(format!(
            "input {} is not a file",
            request.input.display()
        )));
    }
    if request.output.as_os_str().is_empty() {
        return Err(Error::InvalidRequest(
            "output directory must not be empty".to_string(),
        ));
    }
    if request.existing_output_policy == ExistingOutputPolicy::Recompute {
        validate_recompute_destination(request)?;
    }
    Ok(())
}

fn validate_recompute_destination(request: &FileRunRequest) -> Result<()> {
    if !request.output.exists() {
        return Ok(());
    }
    if !request.output.is_dir() {
        return Err(Error::InvalidRequest(format!(
            "recompute output {} is not a directory",
            request.output.display()
        )));
    }
    let output =
        fs::canonicalize(&request.output).map_err(|source| io_error(&request.output, source))?;
    let current = fs::canonicalize(".").map_err(|source| io_error(Path::new("."), source))?;
    if current.starts_with(&output) {
        return Err(Error::InvalidRequest(format!(
            "recompute output {} contains the current working directory",
            request.output.display()
        )));
    }
    let input =
        fs::canonicalize(&request.input).map_err(|source| io_error(&request.input, source))?;
    if input.starts_with(&output) {
        return Err(Error::InvalidRequest(format!(
            "recompute output {} contains its input {}",
            request.output.display(),
            request.input.display()
        )));
    }
    Ok(())
}

fn validate_memory_request(request: &MemoryRunRequest) -> Result<()> {
    validate_artifact_path(&request.input)?;
    if !request.artifacts.contains(&request.input) {
        return Err(Error::InvalidRequest(format!(
            "memory workspace does not contain root input {}",
            request.input.display()
        )));
    }
    Ok(())
}

fn validate_artifact_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::InvalidArtifactPath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn materialize_artifacts(artifacts: &ArtifactSet, root: &Path) -> Result<()> {
    for artifact in artifacts.iter() {
        let path = root.join(artifact.path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| io_error(parent, source))?;
        }
        fs::write(&path, artifact.bytes).map_err(|source| io_error(&path, source))?;
    }
    Ok(())
}

#[cfg(test)]
fn read_artifacts(root: &Path) -> Result<ArtifactSet> {
    let mut artifacts = ArtifactSet::new();
    for path in collect_artifacts(root)? {
        let bytes = fs::read(root.join(&path)).map_err(|source| io_error(&path, source))?;
        artifacts.insert(path, bytes)?;
    }
    Ok(artifacts)
}

fn ensure_empty_output(output: &Path) -> Result<()> {
    if !output.exists() {
        return Ok(());
    }
    let mut entries = fs::read_dir(output).map_err(|source| io_error(output, source))?;
    if entries
        .next()
        .transpose()
        .map_err(|source| io_error(output, source))?
        .is_some()
    {
        return Err(Error::OutputConflict {
            path: output.to_path_buf(),
        });
    }
    Ok(())
}

fn inventory(root: &Path) -> Result<BTreeMap<PathBuf, (u64, Option<std::time::SystemTime>)>> {
    if !root.exists() {
        return Ok(BTreeMap::new());
    }
    collect_artifacts(root)?
        .into_iter()
        .map(|path| {
            let full = root.join(&path);
            let metadata = fs::symlink_metadata(&full).map_err(|source| io_error(&full, source))?;
            Ok((path, (metadata.len(), metadata.modified().ok())))
        })
        .collect()
}

fn collect_artifacts(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    collect_artifacts_into(root, root, &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn collect_artifacts_into(root: &Path, current: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(current).map_err(|source| io_error(current, source))? {
        let entry = entry.map_err(|source| io_error(current, source))?;
        let kind = entry
            .file_type()
            .map_err(|source| io_error(current, source))?;
        let path = entry.path();
        if kind.is_symlink() || entry.file_name() == ".refeff-cache" {
            continue;
        }
        if kind.is_dir() {
            collect_artifacts_into(root, &path, paths)?;
        } else if kind.is_file() {
            paths.push(
                path.strip_prefix(root)
                    .map_err(|error| Error::InvalidRequest(error.to_string()))?
                    .to_path_buf(),
            );
        }
    }
    Ok(())
}

fn publish_recomputed(staging: tempfile::TempDir, output: &Path) -> Result<()> {
    if !output.exists() {
        let staging_path = staging.keep();
        return fs::rename(&staging_path, output).map_err(|source| {
            let _ = fs::remove_dir_all(&staging_path);
            io_error(output, source)
        });
    }

    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let backup_slot = tempfile::Builder::new()
        .prefix(".refeff-backup-")
        .tempdir_in(parent)
        .map_err(|source| io_error(parent, source))?;
    let backup = backup_slot.path().to_path_buf();
    backup_slot
        .close()
        .map_err(|source| io_error(&backup, source))?;
    fs::rename(output, &backup).map_err(|source| io_error(output, source))?;

    let staging_path = staging.keep();
    if let Err(publish_error) = fs::rename(&staging_path, output) {
        let rollback = fs::rename(&backup, output);
        let _ = fs::remove_dir_all(&staging_path);
        return match rollback {
            Ok(()) => Err(io_error(output, publish_error)),
            Err(rollback_error) => Err(Error::Engine {
                message: format!(
                    "failed to publish recomputed output {}: {publish_error}; rollback from {} also failed: {rollback_error}",
                    output.display(),
                    backup.display()
                ),
            }),
        };
    }
    fs::remove_dir_all(&backup).map_err(|source| io_error(&backup, source))
}

fn io_error(path: &Path, source: std::io::Error) -> Error {
    Error::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Common facade imports for applications embedding FEFF.
pub mod prelude {
    pub use crate::{
        ArtifactRef, ArtifactSelection, ArtifactSet, CancellationToken, ExistingOutputPolicy,
        FileRunRequest, MemoryRunRequest, MemoryRunResult, Module, ProgressEvent, ProgressSink,
        RunReport, Runner, Spectra, StageAction, StageReport,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_names_match_feff_entry_points() {
        assert_eq!(Module::Mkgtr.as_str(), "mkgtr");
        assert_eq!(Module::Opcons.as_str(), "opconsat");
        assert_eq!(Module::Ff2x.as_str(), "ff2x");
    }

    #[test]
    fn conflict_policy_rejects_nonempty_directory() -> Result<()> {
        let directory = refeff_engine::execution::temporary_workspace("refeff-")
            .map_err(|source| io_error(Path::new("."), source))?;
        fs::write(directory.path().join("existing"), b"data")
            .map_err(|source| io_error(directory.path(), source))?;
        let error = ensure_empty_output(directory.path()).expect_err("conflict must fail");
        assert!(matches!(error, Error::OutputConflict { .. }));
        Ok(())
    }

    #[test]
    fn artifact_set_rejects_paths_outside_workspace() {
        let mut artifacts = ArtifactSet::new();
        for path in ["", ".", "../secret", "nested/../secret", "/absolute"] {
            let error = artifacts
                .insert(path, b"data".to_vec())
                .expect_err("unsafe artifact path must fail");
            assert!(matches!(error, Error::InvalidArtifactPath { .. }));
        }
    }

    #[test]
    fn artifact_set_round_trips_nested_files_with_format_metadata() -> Result<()> {
        let mut artifacts = ArtifactSet::new();
        artifacts.insert("feff.inp", b"TITLE memory\nEND\n".to_vec())?;
        artifacts.insert("nested/pot.bin", b"payload".to_vec())?;
        let directory = refeff_engine::execution::temporary_workspace("refeff-")
            .map_err(|source| io_error(Path::new("."), source))?;
        materialize_artifacts(&artifacts, directory.path())?;

        let restored = read_artifacts(directory.path())?;
        assert_eq!(restored, artifacts);
        assert_eq!(
            restored
                .iter()
                .find(|artifact| artifact.path.ends_with("pot.bin"))
                .and_then(|artifact| artifact.format)
                .map(|descriptor| descriptor.format),
            Some(FileFormat::PotBin)
        );
        Ok(())
    }

    #[test]
    fn recompute_publication_replaces_stale_output_tree() -> Result<()> {
        let root = refeff_engine::execution::temporary_workspace("refeff-")
            .map_err(|source| io_error(Path::new("."), source))?;
        let output = root.path().join("output");
        fs::create_dir_all(&output).map_err(|source| io_error(&output, source))?;
        fs::write(output.join("stale.dat"), b"stale")
            .map_err(|source| io_error(&output, source))?;
        let staging = tempfile::Builder::new()
            .prefix("stage-")
            .tempdir_in(root.path())
            .map_err(|source| io_error(root.path(), source))?;
        fs::write(staging.path().join("fresh.dat"), b"fresh")
            .map_err(|source| io_error(staging.path(), source))?;

        publish_recomputed(staging, &output)?;

        assert!(!output.join("stale.dat").exists());
        assert_eq!(
            fs::read(output.join("fresh.dat")).map_err(|source| io_error(&output, source))?,
            b"fresh"
        );
        Ok(())
    }

    #[test]
    fn recompute_rejects_output_containing_current_directory() -> Result<()> {
        let input_dir = refeff_engine::execution::temporary_workspace("refeff-")
            .map_err(|source| io_error(Path::new("."), source))?;
        let input = input_dir.path().join("feff.inp");
        fs::write(&input, b"TITLE validation only\nEND\n")
            .map_err(|source| io_error(&input, source))?;
        let request = FileRunRequest::new(&input, ".")
            .with_existing_output_policy(ExistingOutputPolicy::Recompute);

        let error = validate_request(&request).expect_err("current directory must be protected");

        assert!(matches!(error, Error::InvalidRequest(_)));
        Ok(())
    }

    #[test]
    fn in_memory_run_returns_owned_relative_artifacts() -> Result<()> {
        let input = br#"TITLE disabled in-memory pipeline
CONTROL 0 0 0 0 0 0
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#;

        let result = Runner::new().run_in_memory(MemoryRunRequest::new(input.to_vec()))?;

        assert_eq!(result.report.input, Path::new("feff.inp"));
        assert_eq!(result.report.output, Path::new("."));
        assert!(result.artifacts.contains("feff.inp"));
        assert!(result.artifacts.contains("pot.inp"));
        assert!(
            result
                .report
                .artifacts
                .iter()
                .all(|path| path.is_relative())
        );
        Ok(())
    }
}

/// Choose which workspace payloads an in-memory run returns. Intermediate stage
/// serialization is still required by the compatibility scheduler.
#[derive(Debug, Clone, Default)]
pub enum ArtifactSelection {
    #[default]
    All,
    None,
    Spectra,
    Paths(Vec<PathBuf>),
}
impl ArtifactSelection {
    fn includes(&self, path: &Path) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::Spectra => matches!(path.to_str(), Some("chi.dat" | "xmu.dat")),
            Self::Paths(paths) => paths.iter().any(|selected| selected == path),
        }
    }
}
/// Decoded final EXAFS spectra. Files and raw path amplitudes remain available
/// separately; these tables represent the assembled spectrum.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Spectra {
    pub chi: Option<io::chi_dat::ChiDatData>,
    pub xmu: Option<io::xmu_dat::XmuDatData>,
}
#[derive(Default, Clone)]
struct RetainedOutputs {
    spectra: Spectra,
    paths: Option<io::PathsDatData>,
}
impl MemoryRunResult {
    /// Return the retained final spectra. The result wrapper preserves the
    /// original decoding API; new callers can borrow the `spectra` field.
    pub fn spectra(&self) -> std::result::Result<Spectra, io::IoError> {
        Ok(self.spectra.clone())
    }
}
