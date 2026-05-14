#![forbid(unsafe_code)]

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use refeff_io::{
    FeffDocument, FeffInput, potential_dat_outputs_from_bins, rdinp, read_apot_bin, read_pot_bin,
};

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
    if name.eq_ignore_ascii_case("wpot") {
        let count = run_wpot_for_input(&input)?;
        println!(
            "wpot: wrote {count} potential output file(s) beside {}",
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

fn run_wpot_for_input(input: &Path) -> Result<usize> {
    let work_dir = input
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    run_wpot_in_dir(work_dir)
}

fn run_wpot_in_dir(work_dir: &Path) -> Result<usize> {
    let pot_path = work_dir.join("pot.bin");
    let apot_path = work_dir.join("apot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let apot = read_apot_bin(&apot_path)
        .with_context(|| format!("failed to read {}", apot_path.display()))?;
    let outputs = potential_dat_outputs_from_bins(&pot, &apot)
        .context("failed to render FEFF wpot potential outputs")?;
    let count = outputs.len();
    for (name, content) in outputs {
        let output_path = work_dir.join(&name);
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    Ok(count)
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
    use super::{execute_rdinp, run_feff_to_dir, run_wpot_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3};
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::rdinp;
    use refeff_io::{
        ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection,
        ApotBinType, PotBinData, PotBinScalars, write_apot_bin, write_pot_bin,
    };

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

        let count = run_wpot_in_dir(temp.path())?;

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
        assert!(output.join("pot.inp").is_file());
        assert!(output.join("xsph.inp").is_file());
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
