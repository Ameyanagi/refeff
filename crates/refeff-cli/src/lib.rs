#![forbid(unsafe_code)]

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use refeff_io::{FeffDocument, FeffInput, rdinp};

#[derive(Debug, Parser)]
#[command(
    name = "refeff",
    version,
    about = "Pure-Rust FEFF10 compatibility port"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Inspect {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    Rdinp {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    Run {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    Module {
        name: String,
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
}

/// Summary of the parsed input handled by the `rdinp` compatibility stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdinpReport {
    /// Number of active FEFF cards parsed from the input.
    pub cards: usize,
    /// Number of atoms extracted from the `ATOMS` table.
    pub atoms: usize,
    /// Number of unique potential rows extracted from `POTENTIALS`.
    pub potentials: usize,
    /// FEFF-style RDINP stdout/log summary, when currently renderable.
    pub stdout: Option<String>,
}

/// Dispatch a parsed `refeff` command.
pub fn run_cli(cli: Cli) -> Result<()> {
    match cli.command.unwrap_or(Command::Run {
        input: PathBuf::from("feff.inp"),
    }) {
        Command::Inspect { input } => inspect(input),
        Command::Rdinp { input } => run_rdinp(input),
        Command::Run { input } => run_feff(input),
        Command::Module { name, input } => run_module(&name, input),
    }
}

fn inspect(input: PathBuf) -> Result<()> {
    let parsed = FeffInput::parse_file(&input)?;
    let cards = parsed.cards().count();
    let atom_rows = parsed.section_rows("ATOMS").count();
    let potential_rows = parsed.section_rows("POTENTIALS").count();
    println!(
        "input={} cards={} atoms={} potentials={}",
        input.display(),
        cards,
        atom_rows,
        potential_rows
    );
    Ok(())
}

/// Run the supported FEFF `rdinp` compatibility stage in the current directory.
pub fn run_rdinp(input: PathBuf) -> Result<()> {
    let report = execute_rdinp(&input, Path::new("."))?;
    let mut stdout = std::io::stdout().lock();
    if let Some(summary) = &report.stdout {
        stdout.write_all(summary.as_bytes())?;
    } else {
        writeln!(
            stdout,
            "rdinp: parsed {} cards, {} atoms, {} potentials",
            report.cards, report.atoms, report.potentials
        )?;
    }
    Ok(())
}

fn run_feff(input: PathBuf) -> Result<()> {
    run_feff_to_dir(&input, Path::new("."))
}

fn run_feff_to_dir(input: &Path, output_dir: &Path) -> Result<()> {
    let report = execute_rdinp(input, output_dir)?;
    bail!(
        "full FEFF numerical execution is not implemented yet; completed rdinp for {} cards, {} atoms, {} potentials from {}",
        report.cards,
        report.atoms,
        report.potentials,
        input.display()
    )
}

fn run_module(name: &str, input: PathBuf) -> Result<()> {
    if name.eq_ignore_ascii_case("rdinp") {
        return run_rdinp(input);
    }

    let parsed = FeffInput::parse_file(&input)?;
    bail!(
        "module {name} is not implemented yet; parsed {} active lines from {}",
        parsed.lines.len(),
        input.display()
    )
}

fn execute_rdinp(input: &Path, output_dir: &Path) -> Result<RdinpReport> {
    let parsed = FeffInput::parse_file(input)?;
    let document = FeffDocument::from_input(&parsed)?;
    let outputs = rdinp::text_outputs(&document)?;
    let stdout = rdinp::rdinp_log_dat_string(&document).ok();
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    for (name, content) in outputs {
        let output_path = output_dir.join(&name);
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    if let Some(content) = &stdout {
        let output_path = output_dir.join("log.dat");
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }

    Ok(RdinpReport {
        cards: parsed.cards().count(),
        atoms: document.atoms.len(),
        potentials: document.potentials.len(),
        stdout,
    })
}

#[cfg(test)]
mod tests {
    use super::{execute_rdinp, run_feff_to_dir};
    use anyhow::{Context, Result};

    fn write_minimal_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu smoke test
EDGE K
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn write_dmdw_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
DEBYE 450 315 5 dym/force.dym 6 0 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        Ok(())
    }

    fn minimal_dym_text() -> &'static str {
        concat!(
            "    1\n",
            "    1\n",
            "   29\n",
            "   63.546000\n",
            "    0.00000000    0.00000000    0.00000000\n",
            "    1    1\n",
            "  1.000000E+00  0.000000E+00  0.000000E+00\n",
            "  0.000000E+00  1.000000E+00  0.000000E+00\n",
            "  0.000000E+00  0.000000E+00  1.000000E+00\n",
        )
    }

    #[test]
    fn rdinp_stage_writes_supported_outputs_to_requested_dir() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_minimal_input(&input)?;

        let report = execute_rdinp(&input, &output)?;

        assert_eq!(report.cards, 6);
        assert_eq!(report.atoms, 2);
        assert_eq!(report.potentials, 2);
        assert!(
            report
                .stdout
                .as_deref()
                .is_some_and(|stdout| stdout.starts_with("Launching FEFF version"))
        );
        assert!(output.join("atoms.dat").is_file());
        assert!(output.join("geom.dat").is_file());
        assert!(output.join(".dimensions.dat").is_file());
        assert!(output.join("log.dat").is_file());
        assert!(output.join("rixs.inp").is_file());
        Ok(())
    }

    #[test]
    fn rdinp_stage_copies_relative_dmdw_auxiliary_to_requested_dir() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        let dym_dir = temp.path().join("dym");
        std::fs::create_dir_all(&dym_dir)?;
        write_dmdw_input(&input)?;
        std::fs::write(dym_dir.join("force.dym"), minimal_dym_text())?;

        execute_rdinp(&input, &output)?;

        assert_eq!(
            std::fs::read_to_string(output.join("dym").join("force.dym"))?,
            minimal_dym_text()
        );
        Ok(())
    }

    #[test]
    fn full_run_writes_rdinp_outputs_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_minimal_input(&input)?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(error.to_string().contains("completed rdinp"));
        assert!(output.join("pot.inp").is_file());
        assert!(output.join("xsph.inp").is_file());
        Ok(())
    }
}
