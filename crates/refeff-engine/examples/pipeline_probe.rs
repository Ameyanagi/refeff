//! Minimal embedding probe for feature-specific release builds.

use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(
        args.next()
            .ok_or("usage: pipeline_probe <feff.inp> <output-dir>")?,
    );
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: pipeline_probe <feff.inp> <output-dir>")?,
    );
    if args.next().is_some() {
        return Err("usage: pipeline_probe <feff.inp> <output-dir>".into());
    }

    let started = Instant::now();
    let report = refeff_engine::execute_feff(&input, &output)?;
    let path_count = std::fs::read_dir(&output)?
        .filter_map(std::result::Result::ok)
        .filter(|entry| {
            entry.file_name().to_str().is_some_and(|name| {
                name.len() == 12
                    && name.starts_with("feff")
                    && name.ends_with(".dat")
                    && name[4..name.len() - 4]
                        .bytes()
                        .all(|byte| byte.is_ascii_digit())
            })
        })
        .count();
    println!(
        "elapsed_ms={:.3} stages={} path_files={path_count}",
        started.elapsed().as_secs_f64() * 1_000.0,
        report.stages.len(),
    );
    for stage in report.stages {
        println!(
            "stage={} duration_ms={} count={} {}",
            stage.name, stage.duration_ms, stage.count, stage.unit
        );
    }
    Ok(())
}
