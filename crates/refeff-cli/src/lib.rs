#![forbid(unsafe_code)]

mod compton;
mod fullspectrum;
mod opcons;
mod wpot;

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
    /// FEFF-style RDINP stdout summary, when currently renderable.
    pub stdout: Option<String>,
}

/// Dispatch a parsed `refeff` command.
pub fn run_cli(cli: Cli) -> Result<()> {
    let command = match cli.command {
        Some(command) => command,
        None => Command::Run {
            input: PathBuf::from("feff.inp"),
        },
    };
    match command {
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
    let module_reports = run_supported_cached_modules(output_dir)?;
    bail!(
        "full FEFF numerical execution is not implemented yet; completed rdinp for {} cards, {} atoms, {} potentials from {}; {}",
        report.cards,
        report.atoms,
        report.potentials,
        input.display(),
        supported_module_summary(&module_reports)
    )
}

fn run_module(name: &str, input: PathBuf) -> Result<()> {
    if name.eq_ignore_ascii_case("rdinp") {
        return run_rdinp(input);
    }
    if name.eq_ignore_ascii_case("wpot") {
        let count = wpot::run_for_input(&input)?;
        println!(
            "wpot: wrote {count} potential output file(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("opcons") || name.eq_ignore_ascii_case("opconsat") {
        let count = opcons::run_for_input(&input)?;
        println!(
            "opcons: wrote loss.dat with {count} row(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("compton") {
        let count = compton::run_for_input(&input)?;
        println!(
            "compton: wrote compton.dat with {count} row(s) beside {}",
            input.display()
        );
        return Ok(());
    }
    if name.eq_ignore_ascii_case("fullspectrum") {
        let count = fullspectrum::run_for_input(&input)?;
        println!(
            "fullspectrum: wrote optical constants with {count} row(s) beside {}",
            input.display()
        );
        return Ok(());
    }

    let parsed = FeffInput::parse_file(&input)?;
    bail!(
        "module {name} is not implemented yet; parsed {} active lines from {}",
        parsed.lines.len(),
        input.display()
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SupportedModuleReport {
    name: &'static str,
    count: usize,
    unit: &'static str,
}

fn run_supported_cached_modules(work_dir: &Path) -> Result<Vec<SupportedModuleReport>> {
    let mut reports = Vec::new();
    if work_dir.join("pot.bin").is_file() && work_dir.join("apot.bin").is_file() {
        reports.push(SupportedModuleReport {
            name: "wpot",
            count: wpot::run_in_dir(work_dir).context("failed to run supported wpot stage")?,
            unit: "file(s)",
        });
    }

    if opcons::has_complete_table_inputs(work_dir)? {
        let count = opcons::run_in_dir(work_dir).context("failed to run supported opcons stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "opcons",
                count,
                unit: "row(s)",
            });
        }
    }

    if compton::has_cached_profile_inputs(work_dir)? {
        let count =
            compton::run_in_dir(work_dir).context("failed to run supported compton stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "compton",
                count,
                unit: "row(s)",
            });
        }
    }

    if fullspectrum::has_cached_optical_inputs(work_dir)? {
        let count = fullspectrum::run_in_dir(work_dir)
            .context("failed to run supported fullspectrum stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "fullspectrum",
                count,
                unit: "row(s)",
            });
        }
    }

    Ok(reports)
}

fn supported_module_summary(reports: &[SupportedModuleReport]) -> String {
    if reports.is_empty() {
        return "no supported cached stages were run".to_string();
    }

    let details = reports
        .iter()
        .map(|report| format!("{}={} {}", report.name, report.count, report.unit))
        .collect::<Vec<_>>()
        .join(", ");
    format!("supported cached stages run: {details}")
}

pub(crate) fn work_dir_for_input(input: &Path) -> &Path {
    match input
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        Some(parent) => parent,
        None => Path::new("."),
    }
}

fn execute_rdinp(input: &Path, output_dir: &Path) -> Result<RdinpReport> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let error_sentinel = output_dir.join(".feff.error");
    std::fs::write(&error_sentinel, rdinp::rdinp_error_sentinel_string())
        .with_context(|| format!("failed to write {}", error_sentinel.display()))?;

    let parsed = FeffInput::parse_file(input)?;
    let document = match FeffDocument::from_input(&parsed) {
        Ok(document) => document,
        Err(error) => {
            if let Ok(content) = rdinp::rdinp_error_log_string(&parsed, &error) {
                let output_path = output_dir.join("log.dat");
                std::fs::write(&output_path, content)
                    .with_context(|| format!("failed to write {}", output_path.display()))?;
            }
            return Err(error.into());
        }
    };
    let outputs = rdinp::text_outputs(&document)?;
    let log_dat = rdinp::rdinp_log_dat_string(&document).ok();
    let stdout = rdinp::rdinp_stdout_string(&document).ok();
    for (name, content) in outputs {
        let output_path = output_dir.join(name.as_ref());
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    if let Some(content) = &log_dat {
        let output_path = output_dir.join("log.dat");
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    match std::fs::remove_file(&error_sentinel) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to remove {}", error_sentinel.display()));
        }
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
    use super::{execute_rdinp, opcons, run_feff_to_dir, wpot};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3};
    use num_complex::Complex64;
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::rdinp;
    use refeff_io::{
        ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection,
        ApotBinType, EpsDatData, JzzpDatData, PotBinData, PotBinScalars, parse_loss_dat,
        read_compton_dat, read_opcons_dat, read_sumrules_dat, write_apot_bin, write_eps_dat,
        write_jzzp_dat, write_pot_bin,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

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

    fn write_highz_template_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE test_element
HIGHZ
POTENTIALS
       0    XXX   Te
END
"#,
        )?;
        Ok(())
    }

    fn write_opcons_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu opcons run
OPCONS
NUMDENS 0 1.0
POTENTIALS
0 29 Cu
END
"#,
        )?;
        Ok(())
    }

    fn write_compton_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu compton cache run
COMPTON 1.0 3 0
CGRID 1.0 2 2 3 3
END
"#,
        )?;
        Ok(())
    }

    fn write_fullspectrum_cached_input(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            r#"
TITLE Cu fullspectrum cache run
FULLSPECTRUM
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

    fn sample_fullspectrum_eps_dat() -> EpsDatData {
        EpsDatData {
            header_lines: vec!["# sample eps.dat".to_string()],
            omega: Array1::from_vec(vec![1.0, 2.0, 4.0, 7.0]),
            epsilon: Array1::from_vec(vec![
                Complex64::new(0.2, 0.05),
                Complex64::new(0.4, 0.12),
                Complex64::new(0.1, 0.07),
                Complex64::new(0.3, 0.03),
            ]),
            background_epsilon: Array1::from_vec(vec![
                Complex64::new(0.1, 0.02),
                Complex64::new(0.2, 0.04),
                Complex64::new(0.05, 0.025),
                Complex64::new(0.15, 0.01),
            ]),
            sigma: Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04]),
        }
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
    fn wpot_module_writes_potential_dat_outputs_from_bin_state() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin_data())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin_data())?;

        let count = wpot::run_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert_eq!(
            std::fs::read_to_string(temp.path().join("pot00.dat"))?
                .lines()
                .nth(4)
                .context("missing first potential data row")?,
            "    1  1.5073E-04 -7.6250E-01  1.1937E-03 -1.2200E+00 -4.4700E-01  2.7852E-03"
        );
        Ok(())
    }

    #[test]
    fn opcons_module_writes_loss_and_epsilon_from_tables() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin_data())?;
        std::fs::write(
            temp.path().join("opcons.inp"),
            concat!(
                "run_opcons\n",
                " T\n",
                "print_eps\n",
                " T\n",
                "NumDens(0:nphx)\n",
                "  1.0000000000000000\n",
            ),
        )?;
        std::fs::write(
            temp.path().join("opconsCu.dat"),
            concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n",),
        )?;

        let count = opcons::run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let loss = parse_loss_dat(&std::fs::read_to_string(temp.path().join("loss.dat"))?)?;
        assert_eq!(loss.point_count(), 3);
        assert_close(loss.energy_ev[0], 1.0, 1.0e-12);
        assert_close(
            loss.loss[0],
            0.5 / (2.0_f64.powi(2) + 0.5_f64.powi(2)),
            1.0e-6,
        );
        assert!(temp.path().join("epsilon.dat").is_file());
        Ok(())
    }

    #[test]
    fn opcons_module_matches_feff_reference_loss_when_present() -> Result<()> {
        let Some(zip_path) = reference_opcons_zip()? else {
            eprintln!("skipping OPCONS reference test; Cu_OPCONS REFERENCE.zip not found");
            return Ok(());
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            eprintln!("skipping OPCONS reference test; unzip command not found");
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        for name in ["feff.inp", "opconsCu.dat"] {
            std::fs::write(
                temp.path().join(name),
                unzip_reference_entry(&zip_path, &format!("REFERENCE/{name}"))?,
            )?;
        }
        std::fs::write(
            temp.path().join("opcons.inp"),
            concat!(
                "run_opcons\n",
                " T\n",
                "print_eps\n",
                " F\n",
                "NumDens(0:nphx)\n",
                "  8.640712681512044E-004  8.640712681512043E-002\n",
            ),
        )?;
        let expected_loss = parse_loss_dat(&String::from_utf8(unzip_reference_entry(
            &zip_path,
            "REFERENCE/loss.dat",
        )?)?)?;

        let count = opcons::run_in_dir(temp.path())?;

        let actual_loss = parse_loss_dat(&std::fs::read_to_string(temp.path().join("loss.dat"))?)?;
        assert_eq!(count, expected_loss.point_count());
        assert_eq!(actual_loss.point_count(), expected_loss.point_count());
        for ((actual_energy, expected_energy), (actual_loss, expected_loss)) in actual_loss
            .energy_ev
            .iter()
            .zip(expected_loss.energy_ev.iter())
            .zip(actual_loss.loss.iter().zip(expected_loss.loss.iter()))
        {
            assert_close(*actual_energy, *expected_energy, 2.0e-6);
            assert_close(*actual_loss, *expected_loss, 2.0e-5);
        }
        Ok(())
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
        assert!(!output.join(".feff.error").exists());
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
        assert!(
            error
                .to_string()
                .contains("no supported cached stages were run")
        );
        assert!(output.join("pot.inp").is_file());
        assert!(output.join("xsph.inp").is_file());
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_wpot_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_minimal_input(&input)?;
        write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;
        write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: wpot=1 file(s)")
        );
        assert!(output.join("pot00.dat").is_file());
        Ok(())
    }

    #[test]
    fn full_run_skips_incomplete_wpot_cache_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_minimal_input(&input)?;
        write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("no supported cached stages were run")
        );
        assert!(!output.join("pot00.dat").exists());
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_opcons_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_opcons_input(&input)?;
        std::fs::write(
            output.join("opconsCu.dat"),
            concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n"),
        )?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: opcons=3 row(s)")
        );
        assert!(output.join("loss.dat").is_file());
        Ok(())
    }

    #[test]
    fn full_run_skips_opcons_stage_when_tables_are_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_opcons_input(&input)?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("no supported cached stages were run")
        );
        assert!(!output.join("loss.dat").exists());
        Ok(())
    }

    #[test]
    fn full_run_skips_compton_stage_when_jzzp_cache_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_compton_cached_input(&input)?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("no supported cached stages were run")
        );
        assert!(!output.join("compton.dat").exists());
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_compton_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_compton_cached_input(&input)?;
        write_jzzp_dat(output.join("jzzp.dat"), &sample_jzzp_data())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: compton=3 row(s)")
        );
        assert_eq!(
            read_compton_dat(output.join("compton.dat"))?.point_count(),
            3
        );
        Ok(())
    }

    #[test]
    fn full_run_executes_cached_fullspectrum_stage_before_unported_module_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        std::fs::create_dir_all(&output)?;
        write_fullspectrum_cached_input(&input)?;
        write_eps_dat(output.join("eps.dat"), &sample_fullspectrum_eps_dat())?;
        write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;

        let error = run_feff_to_dir(&input, &output)
            .err()
            .context("downstream modules should still be unported")?;

        assert!(
            error
                .to_string()
                .contains("supported cached stages run: fullspectrum=4 row(s)")
        );
        assert_eq!(
            read_opcons_dat(output.join("opconsKK.dat"))?.point_count(),
            4
        );
        assert_eq!(
            read_sumrules_dat(output.join("sumrules.dat"))?.point_count(),
            4
        );
        assert!(output.join("opcons.dat").is_file());
        assert!(output.join("opcons0.dat").is_file());
        Ok(())
    }

    #[test]
    fn failed_rdinp_writes_feff_style_error_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("out");
        write_highz_template_input(&input)?;

        let error = execute_rdinp(&input, &output)
            .err()
            .context("HIGHZ template should fail during rdinp extraction")?;

        assert!(error.to_string().contains("XXX"));
        assert_eq!(
            std::fs::read_to_string(output.join("log.dat"))?,
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "Using finite nucleus.\n",
                " Error reading input, bad line follows:\n",
                " 0    XXX   Te\n",
                "RDINP fatal error.\n",
            )
        );
        assert_eq!(
            std::fs::read_to_string(output.join(".feff.error"))?,
            rdinp::rdinp_error_sentinel_string()
        );
        Ok(())
    }

    fn reference_opcons_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/MPSE/Cu_OPCONS/REFERENCE.zip");
        Ok(path.is_file().then_some(path))
    }

    fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
        let output = Command::new("unzip")
            .arg("-p")
            .arg(zip_path)
            .arg(entry)
            .output()
            .with_context(|| format!("failed to read {entry} from {}", zip_path.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "failed to extract {entry} from {}: {stderr}",
                zip_path.display()
            );
        }
        Ok(output.stdout)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    fn sample_jzzp_data() -> JzzpDatData {
        JzzpDatData {
            ns: 2,
            nphi: 2,
            nz: 3,
            nzp: 3,
            smax: 1.0,
            phimax: std::f64::consts::PI,
            zmax: 1.0,
            zpmax: 1.0,
            values: Array2::from_shape_fn((3, 3), |(z, zp)| {
                0.2 + z as f64 * 0.1 + zp as f64 * 0.05
            }),
        }
    }

    fn sample_pot_bin_data() -> PotBinData {
        let potentials = 1;
        PotBinData {
            titles: vec!["CLI wpot smoke test".to_string()],
            pad_width: 8,
            nohole: 0,
            ihole: 1,
            interstitial_selector: 0,
            automatic_folp: 0,
            jump_mode: 0,
            unfreeze_f: 0,
            scalars: PotBinScalars {
                average_norman_radius: 1.0,
                fermi_level: 0.0,
                interstitial_potential: 0.0,
                interstitial_density: 0.0,
                edge_position: 0.0,
                amplitude_reduction: 1.0,
                relaxation_energy: 0.0,
                plasmon_frequency: 0.0,
                core_valence_energy: 0.0,
                density_radius: 1.0,
                fermi_momentum: 0.0,
                total_charge: 0.0,
                total_volume: 1.0,
            },
            muffin_tin_indices: Array1::from_vec(vec![12]),
            muffin_tin_radii: Array1::from_vec(vec![1.1]),
            norman_indices: Array1::from_vec(vec![40]),
            atomic_numbers: Array1::from_vec(vec![29]),
            kappa: Array1::zeros(POT_BIN_ORBITALS),
            norman_radii: Array1::from_vec(vec![2.1]),
            overlap_factors: Array1::ones(potentials),
            max_overlap_factors: Array1::ones(potentials),
            potential_multiplicities: Array1::ones(potentials),
            ionization: Array1::zeros(potentials),
            initial_large_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
            initial_small_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
            large_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
            small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
            large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
            small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
            electron_density: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, _)| 0.035 * (row + 1) as f64,
            ),
            coulomb_potential: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, _)| -1.2 - 0.02 * (row + 1) as f64,
            ),
            total_potential: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, _)| -0.45 + 0.003 * (row + 1) as f64,
            ),
            valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potentials)),
            orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
            occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
            norman_charges: Array1::zeros(potentials),
            valence_occupancy: Array2::zeros((4, potentials)),
            raw_text: None,
        }
    }

    fn sample_apot_bin_data() -> ApotBinData {
        ApotBinData {
            sections: vec![
                apot_matrix_section(
                    8,
                    "rho(r,0:nphx+1) - atomic density for each unique potential",
                    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, 2), |(row, potential)| {
                        0.015 * (row + 1) as f64 + 0.25 * potential as f64
                    }),
                ),
                apot_matrix_section(
                    11,
                    "vcoul(r,nph) - coulomb potential for each unique potential.",
                    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, 2), |(row, potential)| {
                        -0.75 * (potential + 1) as f64 - 0.0125 * (row + 1) as f64
                    }),
                ),
            ],
        }
    }

    fn apot_matrix_section(
        section_number: usize,
        header: &str,
        values: Array2<f64>,
    ) -> ApotBinSection {
        ApotBinSection {
            section_number,
            headers: vec![header.to_string()],
            header_texts: vec![format!(" {header}")],
            column_labels: vec![],
            column_label_text: None,
            payload: ApotBinPayload::Matrix(ApotBinMatrix {
                value_type: ApotBinType::Double,
                values: ApotBinMatrixValues::Real(values),
            }),
            trailing_headers: vec![],
            trailing_header_texts: vec![],
        }
    }
}
