//! Conservative opt-in provenance cache. Legacy validation remains the audit oracle.
use crate::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
    sync::OnceLock,
};
#[derive(Serialize, Deserialize)]
struct Manifest {
    schema: u32,
    fingerprint: String,
    count: usize,
    artifacts: Vec<PathBuf>,
}

#[derive(Serialize, Deserialize)]
struct PipelineManifest {
    schema: u32,
    fingerprint: String,
    stages: Vec<CachedStage>,
    artifacts: Vec<PathBuf>,
}
#[derive(Serialize, Deserialize)]
struct CachedStage {
    name: String,
    unit: String,
    count: usize,
}
impl CachedStage {
    fn restore(&self) -> Option<crate::SupportedModuleReport> {
        // A partial scientific boundary is not a completed reusable module.
        let module = crate::ModuleName::parse(&self.name).ok()?;
        if module.disabled_feature().is_some() {
            return None;
        }
        let unit = [
            "file(s)",
            "path(s)",
            "row(s)",
            "section(s)",
            "pole(s)",
            "target(s)",
        ]
        .into_iter()
        .find(|unit| *unit == self.unit)?;
        Some(crate::SupportedModuleReport {
            name: module.as_str(),
            count: self.count,
            unit,
            status: crate::StageStatus::Cached,
            duration_ms: 0,
        })
    }
}

/// Validate a completed file pipeline before entering expensive discovery.
/// RDINP has already regenerated the current input's handoffs. Typed captures
/// execute normally so observers receive native values, including omitted text.
pub(crate) fn pipeline(
    root: &Path,
    reports: &mut Vec<crate::SupportedModuleReport>,
    compute: impl FnOnce(&mut Vec<crate::SupportedModuleReport>) -> Result<()>,
) -> Result<()> {
    if std::env::var("REFEFF_CACHE").as_deref() != Ok("provenance")
        || crate::execution::captures_spectra()
    {
        return compute(reports);
    }
    pipeline_provenance(root, reports, compute)
}

fn pipeline_provenance(
    root: &Path,
    reports: &mut Vec<crate::SupportedModuleReport>,
    compute: impl FnOnce(&mut Vec<crate::SupportedModuleReport>) -> Result<()>,
) -> Result<()> {
    let path = root.join(".refeff-cache/pipeline.json");
    let before = fingerprint(root).ok().flatten();
    if let (Some(key), Ok(bytes)) = (&before, std::fs::read(&path)) {
        if let Ok(manifest) = serde_json::from_slice::<PipelineManifest>(&bytes) {
            if manifest.schema == 1 && manifest.fingerprint == *key {
                if let Some(stages) = manifest
                    .stages
                    .iter()
                    .map(CachedStage::restore)
                    .collect::<Option<Vec<_>>>()
                {
                    for mut stage in stages {
                        crate::execution::start(stage.name)?;
                        crate::execution::record_action(stage.name, crate::StageStatus::Cached);
                        crate::print_stage_line(reports.len() + 1, &mut stage);
                        reports.push(stage);
                    }
                    crate::execution::emit(crate::execution::Event::Artifacts {
                        root: root.into(),
                        paths: manifest.artifacts,
                    });
                    return Ok(());
                }
            }
        }
    }
    let files_before = before.and_then(|_| file_stamps(root).ok());
    let (_, declared) = crate::execution::collect_declared_artifacts(root, || compute(reports))?;
    // Failure to save an optional acceleration must not fail the calculation.
    let publish = (|| -> Result<()> {
        if let (Some(fingerprint), Some(files_before)) = (fingerprint(root)?, files_before) {
            let stages: Vec<_> = reports
                .iter()
                .map(|stage| CachedStage {
                    name: stage.name.into(),
                    unit: stage.unit.into(),
                    count: stage.count,
                })
                .collect();
            if stages.is_empty() || stages.iter().any(|stage| stage.restore().is_none()) {
                return Ok(());
            }
            let artifacts = file_stamps(root)?
                .into_iter()
                .filter(|(name, stamp)| {
                    files_before.get(name) != Some(stamp)
                        || declared.contains(name)
                        || (name.components().count() == 1
                            && refeff_io::codec::identify_format(name).is_some_and(|format| {
                                reports.iter().any(|stage| stage.name == format.producer)
                            }))
                })
                .map(|(name, _)| name)
                .collect();
            std::fs::create_dir_all(root.join(".refeff-cache"))?;
            let pending = path.with_extension("pending");
            std::fs::write(
                &pending,
                serde_json::to_vec(&PipelineManifest {
                    schema: 1,
                    fingerprint,
                    stages,
                    artifacts,
                })?,
            )?;
            std::fs::rename(pending, path)?;
        }
        Ok(())
    })();
    if let Err(error) = publish {
        crate::execution::emit(crate::execution::Event::Diagnostic {
            code: "cache_write_failed",
            module: "run",
            message: format!(
                "Calculation succeeded, but pipeline provenance was not saved: {error:#}"
            ),
        });
    }
    Ok(())
}
/// Cache only complete, bounded, symlink-free workspaces. Hashes cover payloads,
/// filenames, executable identity, features, environment policy and thread count.
fn fingerprint(root: &Path) -> Result<Option<String>> {
    static BUILD: OnceLock<Option<String>> = OnceLock::new();
    let Some(build) = BUILD.get_or_init(|| {
        std::env::current_exe()
            .ok()
            .and_then(|path| digest_file(&path).ok())
    }) else {
        return Ok(None);
    };
    let mut entries = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    let mut bytes = 0_u64;
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_symlink() {
                return Ok(None);
            }
            if entry.file_name() == ".refeff-cache" {
                continue;
            }
            if kind.is_dir() {
                pending.push(entry.path());
                continue;
            }
            if !kind.is_file() {
                return Ok(None);
            }
            let path = entry.path();
            bytes = bytes.saturating_add(entry.metadata()?.len());
            if entries.len() >= 8192 || bytes > 512 * 1024 * 1024 {
                return Ok(None);
            }
            // External auxiliary references cannot be certified by a workspace hash.
            if path.extension().is_some_and(|ext| ext == "inp") {
                let text = std::fs::read_to_string(&path)?;
                if text
                    .split_whitespace()
                    .any(|token| token.contains('/') || token.contains('\\'))
                {
                    return Ok(None);
                }
            }
            entries.insert(path.strip_prefix(root)?.to_path_buf(), digest_file(&path)?);
        }
    }
    let mut hash = Sha256::new();
    hash.update(build.as_bytes());
    hash.update(format!(
        "schema=1;full={};sfconv={};threads={}",
        cfg!(feature = "full"),
        cfg!(feature = "sfconv"),
        rayon::current_num_threads()
    ));
    let mut environment: Vec<_> = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("FEFF") || key.starts_with("REFEFF") || key.starts_with("RAYON")
        })
        .collect();
    environment.sort();
    hash.update(serde_json::to_vec(&(entries, environment))?);
    Ok(Some(format!("{:x}", hash.finalize())))
}
fn digest_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub(crate) fn run(
    root: &Path,
    stage: &'static str,
    compute: impl FnOnce() -> Result<usize>,
) -> Result<usize> {
    if std::env::var("REFEFF_CACHE").as_deref() != Ok("provenance") {
        return compute();
    }
    run_provenance(root, stage, compute)
}
fn run_provenance(
    root: &Path,
    stage: &'static str,
    compute: impl FnOnce() -> Result<usize>,
) -> Result<usize> {
    // A file fingerprint cannot restore native values that were intentionally
    // omitted from text output. Typed callers must execute their capture path.
    if crate::execution::captures_spectra() {
        return compute();
    }
    let path: PathBuf = root.join(".refeff-cache").join(format!("{stage}.json"));
    let before = fingerprint(root).ok().flatten();
    if let (Some(key), Ok(bytes)) = (&before, std::fs::read(&path)) {
        if let Ok(manifest) = serde_json::from_slice::<Manifest>(&bytes) {
            if manifest.schema == 2 && manifest.fingerprint == *key {
                crate::execution::record_action(stage, crate::StageStatus::Cached);
                crate::execution::emit(crate::execution::Event::Artifacts {
                    root: root.into(),
                    paths: manifest.artifacts,
                });
                return Ok(manifest.count);
            }
        }
    }
    crate::execution::emit(crate::execution::Event::Diagnostic {
        code: "cache_fallback",
        module: stage,
        message: if before.is_none() {
            "Workspace cannot be certified; using legacy semantic validation."
        } else {
            "No matching provenance and payload integrity; using legacy semantic validation."
        }
        .into(),
    });
    let files_before = before.as_ref().and_then(|_| file_stamps(root).ok());
    let count = compute()?;
    let publish = (|| -> Result<()> {
        if let (Some(fingerprint), Some(files_before)) = (fingerprint(root)?, files_before) {
            let artifacts = file_stamps(root)?
                .into_iter()
                .filter(|(name, stamp)| files_before.get(name) != Some(stamp))
                .map(|(name, _)| name)
                .collect();
            std::fs::create_dir_all(root.join(".refeff-cache"))?;
            let pending = path.with_extension("pending");
            std::fs::write(
                &pending,
                serde_json::to_vec(&Manifest {
                    schema: 2,
                    fingerprint,
                    count,
                    artifacts,
                })?,
            )?;
            std::fs::rename(pending, path)?;
        }
        Ok(())
    })();
    if let Err(error) = publish {
        crate::execution::emit(crate::execution::Event::Diagnostic {
            code: "cache_write_failed",
            module: stage,
            message: format!("Calculation succeeded, but provenance was not saved: {error:#}"),
        });
    }
    Ok(count)
}

fn file_stamps(root: &Path) -> Result<BTreeMap<PathBuf, (u64, std::time::SystemTime)>> {
    let mut result = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            if entry.file_name() == ".refeff-cache" {
                continue;
            }
            let kind = entry.file_type()?;
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file() {
                let metadata = entry.metadata()?;
                result.insert(
                    entry.path().strip_prefix(root)?.to_path_buf(),
                    (metadata.len(), metadata.modified()?),
                );
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_capture_executes_even_when_stage_file_provenance_matches() -> Result<()> {
        let root = tempfile::tempdir()?;
        run_provenance(root.path(), "path", || Ok(1))?;
        let emitted = std::sync::Arc::new(std::sync::Mutex::new(false));
        let received = emitted.clone();
        let options = crate::execution::ExecutionOptions {
            spectrum_root: Some(root.path().to_path_buf()),
            threads: Some(rayon::current_num_threads()),
            observer: Some(std::sync::Arc::new(move |event| {
                if matches!(event, crate::execution::Event::Paths(_)) {
                    *received.lock().unwrap() = true;
                }
            })),
            ..Default::default()
        };
        crate::execution::with_execution(&options, || {
            run_provenance(root.path(), "path", || {
                crate::execution::retain_paths(
                    &root.path().join("paths.dat"),
                    refeff_io::PathsDatData {
                        titles: Vec::new(),
                        paths: Vec::new(),
                    },
                );
                Ok(1)
            })
        })?;
        assert!(
            *emitted.lock().unwrap(),
            "native output must reach the observer"
        );
        Ok(())
    }
    #[cfg(feature = "exafs")]
    #[test]
    fn pipeline_hit_obeys_cancellation_requested_by_a_stage_observer() -> Result<()> {
        let root = tempfile::tempdir()?;
        let token = crate::execution::CancellationToken::default();
        let requested = token.clone();
        let options = crate::execution::ExecutionOptions {
            threads: Some(1),
            control: crate::execution::Control {
                cancellation: token,
                ..Default::default()
            },
            observer: Some(std::sync::Arc::new(move |event| {
                if matches!(event, crate::execution::Event::StageStarted("genfmt")) {
                    requested.cancel();
                }
            })),
            ..Default::default()
        };
        let error = crate::execution::with_execution(&options, || {
            pipeline_provenance(root.path(), &mut Vec::new(), |reports| {
                std::fs::write(root.path().join("feff0001.dat"), "valid")?;
                reports.push(crate::SupportedModuleReport {
                    name: "genfmt",
                    unit: "file(s)",
                    count: 1,
                    status: crate::StageStatus::Generated,
                    duration_ms: 1,
                });
                Ok(())
            })?;
            let mut restored = Vec::new();
            let result = pipeline_provenance(root.path(), &mut restored, |_| {
                anyhow::bail!("cache should hit")
            });
            assert!(restored.is_empty());
            result
        })
        .expect_err("the stage observer requested cancellation");
        assert!(
            error
                .downcast_ref::<crate::execution::Interrupted>()
                .is_some()
        );
        Ok(())
    }
    #[cfg(feature = "exafs")]
    #[test]
    fn pipeline_reuses_events_and_artifacts_but_revalidates_changed_inputs_and_outputs()
    -> Result<()> {
        let root = tempfile::tempdir()?;
        std::fs::write(root.path().join("pot.inp"), "source one")?;
        std::fs::write(root.path().join("notes.txt"), "unrelated")?;
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let received = events.clone();
        let options = crate::execution::ExecutionOptions {
            threads: Some(1),
            observer: Some(std::sync::Arc::new(move |event| {
                received
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(event);
            })),
            ..Default::default()
        };
        crate::execution::with_execution(&options, || {
            let computations = std::cell::Cell::new(0);
            let compute = |reports: &mut Vec<crate::SupportedModuleReport>| {
                computations.set(computations.get() + 1);
                std::fs::write(root.path().join("feff0001.dat"), "valid output")?;
                reports.push(crate::SupportedModuleReport {
                    name: "genfmt",
                    unit: "file(s)",
                    count: 1,
                    status: crate::StageStatus::Generated,
                    duration_ms: 2,
                });
                Ok(())
            };
            pipeline_provenance(root.path(), &mut Vec::new(), compute)?;
            let mut restored = Vec::new();
            pipeline_provenance(root.path(), &mut restored, compute)?;
            assert_eq!(computations.get(), 1);
            assert_eq!(restored[0].status, crate::StageStatus::Cached);
            assert!(
                events
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .iter()
                    .any(
                        |event| matches!(event, crate::execution::Event::Artifacts { paths, .. }
                    if paths == &vec![PathBuf::from("feff0001.dat")])
                    )
            );
            std::fs::write(root.path().join("feff0001.dat"), "wrong output")?;
            pipeline_provenance(root.path(), &mut Vec::new(), compute)?;
            assert_eq!(computations.get(), 2);
            std::fs::write(root.path().join("pot.inp"), "source two")?;
            pipeline_provenance(root.path(), &mut Vec::new(), compute)?;
            assert_eq!(computations.get(), 3);
            Ok(())
        })
    }

    #[test]
    fn fast_hit_retains_the_generated_artifact_manifest() -> Result<()> {
        let root = tempfile::tempdir()?;
        std::fs::write(root.path().join("notes.txt"), "unrelated")?;
        let artifacts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let received = artifacts.clone();
        let options = crate::execution::ExecutionOptions {
            threads: Some(1),
            observer: Some(std::sync::Arc::new(move |event| {
                if let crate::execution::Event::Artifacts { paths, .. } = event {
                    *received
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = paths;
                }
            })),
            ..Default::default()
        };
        crate::execution::with_execution(&options, || {
            run_provenance(root.path(), "genfmt", || {
                std::fs::write(root.path().join("feff0001.dat"), "generated")?;
                Ok(1)
            })?;
            assert_eq!(
                run_provenance(root.path(), "genfmt", || anyhow::bail!(
                    "valid cache should not execute"
                ))?,
                1
            );
            Ok(())
        })?;
        assert_eq!(
            *artifacts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec![PathBuf::from("feff0001.dat")]
        );
        Ok(())
    }
    #[test]
    fn provenance_covers_inputs_outputs_and_nested_payload_integrity() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("paths.inp"), "input")?;
        std::fs::create_dir(dir.path().join("nested"))?;
        std::fs::write(dir.path().join("nested/source.dat"), "source")?;
        let first = fingerprint(dir.path())?.expect("bounded workspace");
        std::fs::write(dir.path().join("paths.dat"), "output")?;
        let second = fingerprint(dir.path())?.expect("bounded workspace");
        assert_ne!(first, second);
        std::fs::write(dir.path().join("nested/source.dat"), "tampered")?;
        assert_ne!(Some(second), fingerprint(dir.path())?);
        std::fs::write(dir.path().join("paths.inp"), "../external")?;
        assert!(fingerprint(dir.path())?.is_none());
        Ok(())
    }
    #[cfg(unix)]
    #[test]
    fn provenance_never_follows_symlinks() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::os::unix::fs::symlink(dir.path(), dir.path().join("cycle"))?;
        assert!(fingerprint(dir.path())?.is_none());
        Ok(())
    }
}
