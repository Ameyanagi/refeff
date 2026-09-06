//! Measurement driver used by scripts/benchmark-matrix.py.
use refeff::{ArtifactSelection, ExistingOutputPolicy, FileRunRequest, MemoryRunRequest, Runner};
use std::{num::NonZeroUsize, path::PathBuf, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("expected INPUT OUTPUT MODE THREADS".into());
    }
    let input = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    let threads: usize = args[3].parse()?;
    let mut runner = Runner::new();
    if let Some(n) = NonZeroUsize::new(threads) {
        runner = runner.with_threads(n);
    }
    let start = Instant::now();
    let (stages, bytes) = if args[2] == "memory" || args[2] == "typed" {
        if args[2] == "typed" {
            runner = runner.with_artifacts(ArtifactSelection::None);
        }
        let result = runner.run_in_memory(MemoryRunRequest::new(std::fs::read(input)?))?;
        (
            result.report.stages.len(),
            result
                .artifacts
                .iter()
                .map(|file| file.bytes.len() as u64)
                .sum::<u64>(),
        )
    } else {
        let policy = if args[2] == "recompute" {
            ExistingOutputPolicy::Recompute
        } else {
            ExistingOutputPolicy::ReuseValidated
        };
        let report = runner
            .run_files(FileRunRequest::new(input, &output).with_existing_output_policy(policy))?;
        let bytes = report
            .artifacts
            .iter()
            .map(|path| std::fs::metadata(output.join(path)).map(|m| m.len()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<u64>();
        (report.stages.len(), bytes)
    };
    println!(
        "{{\"seconds\":{},\"stages\":{},\"artifact_bytes\":{}}}",
        start.elapsed().as_secs_f64(),
        stages,
        bytes
    );
    Ok(())
}
