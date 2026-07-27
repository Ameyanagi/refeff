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
            Self::Rdinp => "rdinp",
            Self::Atomic => "atomic",
            Self::Pot => "pot",
            Self::Ldos => "ldos",
            Self::Screen => "screen",
            Self::Crpa => "crpa",
            Self::Opcons => "opconsat",
            Self::Xsph => "xsph",
            Self::Fms => "fms",
            Self::Mkgtr => "mkgtr",
            Self::Path => "path",
            Self::Genfmt => "genfmt",
            Self::Ff2x => "ff2x",
            Self::Sfconv => "sfconv",
            Self::Compton => "compton",
            Self::Eels => "eels",
            Self::EelsMdff => "eelsmdff",
            Self::Rhorrp => "rhorrp",
            Self::Dmdw => "dmdw",
            Self::Band => "band",
            Self::FullSpectrum => "fullspectrum",
            Self::Rixs => "rixs",
            Self::SelfEnergy => "self",
            Self::Wpot => "wpot",
        }
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
    /// Compute in a clean staging directory, then replace generated files.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRunResult {
    /// Typed execution report. Its paths are workspace-relative.
    pub report: RunReport,
    /// Complete final workspace, including inputs and generated artifacts.
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
}

impl Runner {
    /// Construct a runner using process defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bound process-wide Rayon/faer worker threads.
    #[must_use]
    pub fn with_threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = Some(threads);
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
        validate_request(&request)?;
        if let Some(sink) = &self.progress {
            sink.event(ProgressEvent::RunStarted(&request));
        }
        refeff_engine::configure_parallelism(self.threads.map(NonZeroUsize::get));

        let report = match request.existing_output_policy {
            ExistingOutputPolicy::ReuseValidated => {
                fs::create_dir_all(&request.output)
                    .map_err(|source| io_error(&request.output, source))?;
                run_engine(&request.input, &request.output)?
            }
            ExistingOutputPolicy::ErrorOnConflict => {
                ensure_empty_output(&request.output)?;
                fs::create_dir_all(&request.output)
                    .map_err(|source| io_error(&request.output, source))?;
                run_engine(&request.input, &request.output)?
            }
            ExistingOutputPolicy::Recompute => self.run_recomputed(&request)?,
        };

        if let Some(sink) = &self.progress {
            for stage in &report.stages {
                sink.event(ProgressEvent::StageCompleted(stage));
            }
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
        validate_memory_request(&request)?;
        if let Some(sink) = &self.progress {
            sink.event(ProgressEvent::MemoryRunStarted(&request));
        }
        refeff_engine::configure_parallelism(self.threads.map(NonZeroUsize::get));

        let workspace = tempfile::tempdir().map_err(|source| io_error(Path::new("."), source))?;
        materialize_artifacts(&request.artifacts, workspace.path())?;
        let input = workspace.path().join(&request.input);
        let mut report = run_engine(&input, workspace.path())?;
        report.input = request.input.clone();
        report.output = PathBuf::from(".");
        report.artifacts = collect_artifacts(workspace.path())?;
        let artifacts = read_artifacts(workspace.path())?;

        if let Some(sink) = &self.progress {
            for stage in &report.stages {
                sink.event(ProgressEvent::StageCompleted(stage));
            }
            sink.event(ProgressEvent::RunCompleted(&report));
        }
        Ok(MemoryRunResult { report, artifacts })
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
        let staged = run_engine(&request.input, staging.path())?;
        publish_recomputed(staging, &request.output)?;
        Ok(RunReport {
            input: request.input.clone(),
            output: request.output.clone(),
            artifacts: collect_artifacts(&request.output)?,
            ..staged
        })
    }
}

fn run_engine(input: &Path, output: &Path) -> Result<RunReport> {
    let engine = refeff_engine::execute_feff(input, output).map_err(|error| Error::Engine {
        message: format!("{error:#}"),
    })?;
    let stages = engine
        .stages
        .into_iter()
        .map(|stage| StageReport {
            name: stage.name.to_string(),
            action: match stage.status {
                refeff_engine::StageStatus::Cached => StageAction::Reused,
                refeff_engine::StageStatus::Generated => StageAction::Generated,
            },
            count: stage.count,
            unit: stage.unit.to_string(),
            duration_ms: stage.duration_ms,
        })
        .collect();
    Ok(RunReport {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        cards: engine.rdinp.cards,
        atoms: engine.rdinp.atoms,
        potentials: engine.rdinp.potentials,
        stages,
        artifacts: collect_artifacts(output)?,
        diagnostics: Vec::new(),
    })
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

fn read_artifacts(root: &Path) -> Result<ArtifactSet> {
    let mut artifacts = ArtifactSet::new();
    read_artifacts_into(root, root, &mut artifacts)?;
    Ok(artifacts)
}

fn read_artifacts_into(root: &Path, current: &Path, artifacts: &mut ArtifactSet) -> Result<()> {
    for entry in fs::read_dir(current).map_err(|source| io_error(current, source))? {
        let path = entry.map_err(|source| io_error(current, source))?.path();
        if path.is_dir() {
            read_artifacts_into(root, &path, artifacts)?;
        } else {
            let relative = path
                .strip_prefix(root)
                .map_err(|error| Error::InvalidRequest(error.to_string()))?;
            let bytes = fs::read(&path).map_err(|source| io_error(&path, source))?;
            artifacts.insert(relative, bytes)?;
        }
    }
    Ok(())
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

fn collect_artifacts(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    collect_artifacts_into(root, root, &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn collect_artifacts_into(root: &Path, current: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(current).map_err(|source| io_error(current, source))? {
        let path = entry.map_err(|source| io_error(current, source))?.path();
        if path.is_dir() {
            collect_artifacts_into(root, &path, paths)?;
        } else {
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
        ArtifactRef, ArtifactSet, ExistingOutputPolicy, FileRunRequest, MemoryRunRequest,
        MemoryRunResult, Module, ProgressEvent, ProgressSink, RunReport, Runner, StageAction,
        StageReport,
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
        let directory = tempfile::tempdir().map_err(|source| io_error(Path::new("."), source))?;
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
        let directory = tempfile::tempdir().map_err(|source| io_error(Path::new("."), source))?;
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
        let root = tempfile::tempdir().map_err(|source| io_error(Path::new("."), source))?;
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
        let input_dir = tempfile::tempdir().map_err(|source| io_error(Path::new("."), source))?;
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
