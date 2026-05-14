#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use ndarray::Array1;
use refeff_core::{CombinedEpsilon, EpsilonTable, atomic_symbol, combine_epsilon_tables};
use refeff_io::{
    FEFF_BOHR_ANGSTROM, FeffDocument, FeffInput, LossDatData, OpconsInput, PotBinData, PotInput,
    potential_dat_outputs_from_bins, rdinp, read_apot_bin, read_pot_bin, write_loss_dat,
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
    if name.eq_ignore_ascii_case("opcons") || name.eq_ignore_ascii_case("opconsat") {
        let count = run_opcons_for_input(&input)?;
        println!(
            "opcons: wrote loss.dat with {count} row(s) beside {}",
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

fn run_opcons_for_input(input: &Path) -> Result<usize> {
    let work_dir = input
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    run_opcons_in_dir(work_dir)
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

fn run_opcons_in_dir(work_dir: &Path) -> Result<usize> {
    let input_path = work_dir.join("opcons.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    let input = OpconsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))?;
    if !input.run_opcons {
        return Ok(0);
    }

    let pot_state = read_optional_pot_bin(work_dir)?;
    let atomic_numbers = opcons_atomic_numbers(work_dir, pot_state.as_ref())?;
    let weights = opcons_number_densities(&input, pot_state.as_ref(), atomic_numbers.len())?;
    let tables = atomic_numbers
        .iter()
        .map(|&atomic_number| read_opcons_epsilon_table(work_dir, atomic_number))
        .collect::<Result<Vec<_>>>()?;
    let combined = combine_epsilon_tables(&tables, &weights)
        .context("failed to combine OPCONS epsilon tables")?;

    let loss = LossDatData {
        header_lines: vec!["# E(eV)    Loss".to_string()],
        energy_ev: combined.energy_ev.clone(),
        loss: combined.loss.clone(),
    };
    write_loss_dat(work_dir.join("loss.dat"), &loss)
        .with_context(|| format!("failed to write {}", work_dir.join("loss.dat").display()))?;

    if input.print_eps {
        write_epsilon_dat(&work_dir.join("epsilon.dat"), &combined)?;
    }

    Ok(combined.point_count())
}

fn read_optional_pot_bin(work_dir: &Path) -> Result<Option<PotBinData>> {
    let path = work_dir.join("pot.bin");
    if path.is_file() {
        Ok(Some(read_pot_bin(&path).with_context(|| {
            format!("failed to read {}", path.display())
        })?))
    } else {
        Ok(None)
    }
}

fn opcons_atomic_numbers(work_dir: &Path, pot_state: Option<&PotBinData>) -> Result<Vec<usize>> {
    if let Some(pot) = pot_state {
        return Ok(pot.atomic_numbers.to_vec());
    }

    let path = work_dir.join("pot.inp");
    if !path.is_file() {
        return opcons_atomic_numbers_from_feff_input(work_dir);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {} for OPCONS components", path.display()))?;
    let input = match PotInput::parse_str(&path, &text) {
        Ok(input) => input,
        Err(error) if work_dir.join("feff.inp").is_file() => {
            return opcons_atomic_numbers_from_feff_input(work_dir).with_context(|| {
                format!(
                    "failed to parse {} for OPCONS components: {error}",
                    path.display()
                )
            });
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!("failed to parse {} for OPCONS components", path.display())
            });
        }
    };
    input
        .potentials
        .iter()
        .map(|potential| {
            usize::try_from(potential.z)
                .with_context(|| format!("invalid OPCONS atomic number {}", potential.z))
        })
        .collect()
}

fn opcons_atomic_numbers_from_feff_input(work_dir: &Path) -> Result<Vec<usize>> {
    let path = work_dir.join("feff.inp");
    let parsed = FeffInput::parse_file(&path)
        .with_context(|| format!("failed to parse {} for OPCONS components", path.display()))?;
    let document = FeffDocument::from_input(&parsed).with_context(|| {
        format!(
            "failed to interpret {} for OPCONS components",
            path.display()
        )
    })?;
    document
        .potentials
        .iter()
        .map(|potential| {
            let z = potential.z.with_context(|| {
                format!("OPCONS potential {} has no atomic number", potential.ipot)
            })?;
            usize::try_from(z)
                .with_context(|| format!("invalid OPCONS atomic number {z} in {}", path.display()))
        })
        .collect()
}

fn opcons_number_densities(
    input: &OpconsInput,
    pot_state: Option<&PotBinData>,
    component_count: usize,
) -> Result<Vec<f64>> {
    if input.number_densities.len() < component_count {
        bail!(
            "opcons.inp provides {} number densities but {component_count} components are required",
            input.number_densities.len()
        );
    }

    let defaults = if input
        .number_densities
        .iter()
        .take(component_count)
        .any(|density| *density < 0.0)
    {
        let pot = pot_state.context(
            "negative OPCONS number densities require pot.bin Norman radii from the POT stage",
        )?;
        Some(default_opcons_number_densities(pot)?)
    } else {
        None
    };

    (0..component_count)
        .map(|index| {
            let density = input.number_densities[index];
            if !density.is_finite() {
                bail!("OPCONS number density {index} must be finite, got {density}");
            }
            if density < 0.0 {
                defaults
                    .as_ref()
                    .and_then(|values| values.get(index).copied())
                    .with_context(|| format!("missing default OPCONS number density {index}"))
            } else {
                Ok(density)
            }
        })
        .collect()
}

fn default_opcons_number_densities(pot: &PotBinData) -> Result<Vec<f64>> {
    let component_count = pot.potential_count();
    if pot.norman_radii.len() != component_count
        || pot.potential_multiplicities.len() != component_count
    {
        bail!("pot.bin OPCONS arrays have inconsistent component lengths");
    }

    let total_volume = pot
        .potential_multiplicities
        .iter()
        .zip(pot.norman_radii.iter())
        .try_fold(0.0, |sum, (&multiplicity, &norman_radius)| {
            if !multiplicity.is_finite() || !norman_radius.is_finite() {
                bail!("pot.bin OPCONS multiplicity and Norman radius values must be finite");
            }
            let radius_angstrom = norman_radius * FEFF_BOHR_ANGSTROM;
            Ok(sum + multiplicity * 4.0 * std::f64::consts::PI * radius_angstrom.powi(3) / 3.0)
        })?;
    if !total_volume.is_finite() || total_volume <= 0.0 {
        bail!("pot.bin OPCONS Norman volume must be positive, got {total_volume}");
    }

    pot.potential_multiplicities
        .iter()
        .enumerate()
        .map(|(index, &multiplicity)| {
            if !multiplicity.is_finite() {
                bail!("pot.bin OPCONS multiplicity {index} must be finite");
            }
            Ok(multiplicity / total_volume)
        })
        .collect()
}

fn read_opcons_epsilon_table(work_dir: &Path, atomic_number: usize) -> Result<EpsilonTable> {
    let symbol = atomic_symbol(atomic_number)
        .with_context(|| format!("invalid OPCONS atomic number {atomic_number}"))?;
    let path = work_dir.join(format!("opcons{symbol}.dat"));
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_opcons_epsilon_table(&path, &text)
}

fn parse_opcons_epsilon_table(path: &Path, text: &str) -> Result<EpsilonTable> {
    let mut energy_ev = Vec::new();
    let mut epsilon1_minus_one = Vec::new();
    let mut epsilon2 = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with(['#', '!', 'c', 'C']) {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 3 {
            bail!(
                "{}:{} OPCONS epsilon row requires 3 fields, got {}",
                path.display(),
                line_number,
                fields.len()
            );
        }
        energy_ev.push(parse_feff_f64(path, line_number, "energy", fields[0])?);
        epsilon1_minus_one.push(parse_feff_f64(path, line_number, "epsilon1", fields[1])?);
        epsilon2.push(parse_feff_f64(path, line_number, "epsilon2", fields[2])?);
    }

    Ok(EpsilonTable {
        energy_ev: Array1::from_vec(energy_ev),
        epsilon1_minus_one: Array1::from_vec(epsilon1_minus_one),
        epsilon2: Array1::from_vec(epsilon2),
    })
}

fn parse_feff_f64(path: &Path, line: usize, field: &str, token: &str) -> Result<f64> {
    token.replace(['D', 'd'], "E").parse().with_context(|| {
        format!(
            "failed to parse {field} token {token:?} at {}:{line}",
            path.display()
        )
    })
}

fn write_epsilon_dat(path: &Path, combined: &CombinedEpsilon) -> Result<()> {
    let mut out = String::new();
    writeln!(out, "# E(eV)    eps1    eps2")?;
    for ((energy, epsilon1), epsilon2) in combined
        .energy_ev
        .iter()
        .zip(combined.epsilon1.iter())
        .zip(combined.epsilon2.iter())
    {
        writeln!(out, "{energy:22.15E} {epsilon1:22.15E} {epsilon2:22.15E}")?;
    }
    std::fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))
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
    use super::{execute_rdinp, run_feff_to_dir, run_opcons_in_dir, run_wpot_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3};
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::rdinp;
    use refeff_io::{
        ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection,
        ApotBinType, PotBinData, PotBinScalars, parse_loss_dat, write_apot_bin, write_pot_bin,
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

        let count = run_opcons_in_dir(temp.path())?;

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

        let count = run_opcons_in_dir(temp.path())?;

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
