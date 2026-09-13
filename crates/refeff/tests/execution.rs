#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use refeff::{CancellationToken, Error, FileRunRequest, ProgressEvent, ProgressSink, Runner};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
const INPUT: &str =
    "TITLE contracts\nCONTROL 0 0 0 0 0 0\nPOTENTIALS\n0 29 Cu\nATOMS\n0 0 0 0 Cu\nEND\n";

#[test]
fn execution_controls_reach_checkpoints_and_restore_after_failure() {
    use refeff_engine::execution::{ExecutionOptions, with_execution};
    let token = CancellationToken::default();
    let options = ExecutionOptions {
        control: refeff::core::execution::Control {
            cancellation: token.clone(),
            ..Default::default()
        },
        threads: Some(1),
        ..Default::default()
    };
    let error = with_execution(&options, || {
        token.cancel();
        refeff::core::execution::checkpoint()?;
        Ok(())
    })
    .unwrap_err();
    assert_eq!(
        error.downcast_ref::<refeff::Interrupted>(),
        Some(&refeff::Interrupted::Cancelled)
    );
    // WASI installs controls on the coordinator rather than disposable workers.
    // Failed runs must not leave cancellation attached to the next calculation.
    assert!(refeff::core::execution::checkpoint().is_ok());
    with_execution(&ExecutionOptions::default(), || {
        refeff::core::execution::checkpoint()?;
        Ok(())
    })
    .unwrap();
}

struct Sink {
    token: CancellationToken,
    events: Mutex<Vec<String>>,
    cancel_stage: bool,
}
impl ProgressSink for Sink {
    fn event(&self, event: ProgressEvent<'_>) {
        let name = match event {
            ProgressEvent::RunStarted(_) => "run".to_owned(),
            ProgressEvent::StageStarted(name) => {
                if self.cancel_stage {
                    self.token.cancel();
                }
                format!("start:{name}")
            }
            ProgressEvent::StageCompleted(stage) => format!("done:{}", stage.name),
            ProgressEvent::RunCompleted(_) => "complete".to_owned(),
            ProgressEvent::RunFailed(_) => "failed".to_owned(),
            ProgressEvent::RunCancelled(_) => "cancelled".to_owned(),
            _ => return,
        };
        self.events.lock().unwrap().push(name);
    }
}
#[test]
fn live_callback_can_cancel_before_stage_writes() {
    let root = refeff_engine::execution::temporary_workspace("refeff-test-").unwrap();
    let input = root.path().join("feff.inp");
    std::fs::write(&input, INPUT).unwrap();
    let output = root.path().join("out");
    let token = CancellationToken::default();
    let sink = Arc::new(Sink {
        token: token.clone(),
        events: Mutex::new(Vec::new()),
        cancel_stage: true,
    });
    let error = Runner::new()
        .with_threads(NonZeroUsize::new(1).unwrap())
        .with_cancellation(token)
        .with_progress_sink(sink.clone())
        .run_files(FileRunRequest::new(input, &output))
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Pipeline {
            code: "interrupted",
            ..
        }
    ));
    assert!(!output.join("pot.inp").exists());
    assert_eq!(
        *sink.events.lock().unwrap(),
        ["run", "start:rdinp", "cancelled"]
    );
}
#[test]
fn deadline_is_checked_before_writes() {
    let root = refeff_engine::execution::temporary_workspace("refeff-test-").unwrap();
    let input = root.path().join("feff.inp");
    std::fs::write(&input, INPUT).unwrap();
    let output = root.path().join("out");
    assert!(
        Runner::new()
            .with_deadline(std::time::Instant::now())
            .run_files(FileRunRequest::new(input, &output))
            .is_err()
    );
    assert!(!output.join("pot.inp").exists());
}
#[test]
fn sequential_thread_settings_work_without_global_pool_initialization() {
    for threads in [1, 2, 1] {
        let root = refeff_engine::execution::temporary_workspace("refeff-test-").unwrap();
        let input = root.path().join("feff.inp");
        std::fs::write(&input, INPUT).unwrap();
        let output = root.path().join("out");
        std::fs::create_dir(&output).unwrap();
        std::fs::write(output.join("notes.txt"), "user notes").unwrap();
        let report = Runner::new()
            .with_threads(NonZeroUsize::new(threads).unwrap())
            .run_files(FileRunRequest::new(input, output))
            .unwrap();
        assert!(
            !report
                .artifacts
                .iter()
                .any(|path| path.ends_with("notes.txt"))
        );
    }
}
#[cfg(unix)]
#[test]
fn artifact_traversal_does_not_follow_symlink_cycles_or_escape() {
    let root = refeff_engine::execution::temporary_workspace("refeff-test-").unwrap();
    let input = root.path().join("feff.inp");
    std::fs::write(&input, INPUT).unwrap();
    let output = root.path().join("out");
    std::fs::create_dir(&output).unwrap();
    std::os::unix::fs::symlink(&output, output.join("cycle")).unwrap();
    std::os::unix::fs::symlink(root.path(), output.join("outside")).unwrap();
    let report = Runner::new()
        .with_threads(NonZeroUsize::new(1).unwrap())
        .run_files(FileRunRequest::new(input, output))
        .unwrap();
    assert!(
        report
            .artifacts
            .iter()
            .all(|path| !path.starts_with("cycle") && !path.starts_with("outside"))
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "full EXAFS parity runs in the release gate"
)]
fn typed_only_exafs_matches_serialized_spectra() {
    use refeff::{ArtifactSelection, MemoryRunRequest};
    let input = include_bytes!("data/znse.inp");
    let runner = || Runner::new().with_threads(NonZeroUsize::new(1).unwrap());
    let file = runner()
        .run_in_memory(MemoryRunRequest::new(input.to_vec()))
        .unwrap();
    let typed = runner()
        .with_artifacts(ArtifactSelection::None)
        .run_in_memory(MemoryRunRequest::new(input.to_vec()))
        .unwrap();
    assert!(typed.artifacts.is_empty());
    assert_eq!(typed.paths, file.paths);
    assert!(!typed.paths.as_ref().unwrap().paths.is_empty());
    let expected = refeff::io::chi_dat::parse_chi_dat(
        std::str::from_utf8(file.artifacts.get("chi.dat").unwrap()).unwrap(),
    )
    .unwrap();
    let actual = typed.spectra.chi.unwrap();
    assert_eq!(actual.point_count(), expected.point_count());
    for (a, b) in actual.chi.iter().zip(expected.chi.iter()) {
        assert!((a - b).abs() <= 5e-8 + b.abs() * 5e-5);
    }
    assert_eq!(
        typed.spectra.xmu.unwrap().photon_energy_ev.len(),
        file.spectra.xmu.unwrap().photon_energy_ev.len()
    );
}
