use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::Array1;
use refeff_core::{
    CombinedEpsilon, EpsilonTable, atomic_symbol, combine_epsilon_tables,
    opcons::epsilon_table_from_database,
};
use refeff_io::{
    FEFF_BOHR_ANGSTROM, FeffDocument, FeffInput, LossDatData, OpconsInput, PotBinData, PotInput,
    read_pot_bin, write_loss_dat,
};

use crate::work_dir_for_input;

/// Run FEFF `OPCONS`/`OPCONSAT` optical-loss generation beside an input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether FEFF optical-loss generation has valid or generatable table inputs.
///
/// Existing elemental tables remain valid user overrides. Missing tables are
/// available from FEFF10's bundled `epsdb`; malformed existing overrides are
/// rejected rather than silently replaced.
pub(crate) fn has_complete_table_inputs(work_dir: &Path) -> Result<bool> {
    let input_path = work_dir.join("opcons.inp");
    if !input_path.is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !input.run_opcons {
        return Ok(false);
    }

    let Ok(pot_state) = read_optional_pot_bin(work_dir) else {
        return Ok(false);
    };
    let Ok(atomic_numbers) = opcons_atomic_numbers(work_dir, pot_state.as_ref()) else {
        return Ok(false);
    };
    let component_count = atomic_numbers.len();
    if input.number_densities.len() < component_count {
        return Ok(false);
    }
    if input
        .number_densities
        .iter()
        .take(component_count)
        .any(|density| *density < 0.0)
        && pot_state.is_none()
    {
        return Ok(false);
    }
    for atomic_number in atomic_numbers {
        let symbol = atomic_symbol(atomic_number)
            .with_context(|| format!("invalid OPCONS atomic number {atomic_number}"))?;
        let path = work_dir.join(format!("opcons{symbol}.dat"));
        if path.is_file() {
            if read_opcons_epsilon_table(work_dir, atomic_number).is_err() {
                return Ok(false);
            }
        } else if epsilon_table_from_database(atomic_number).is_err() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Write `loss.dat`, and optionally `epsilon.dat`, from FEFF optical constants.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !input.run_opcons {
        return Ok(0);
    }

    let pot_state = read_optional_pot_bin(work_dir)?;
    let atomic_numbers = opcons_atomic_numbers(work_dir, pot_state.as_ref())?;
    let weights = opcons_number_densities(&input, pot_state.as_ref(), atomic_numbers.len())?;
    let tables = atomic_numbers
        .iter()
        .map(|&atomic_number| load_or_generate_opcons_epsilon_table(work_dir, atomic_number))
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

fn read_input(work_dir: &Path) -> Result<OpconsInput> {
    let input_path = work_dir.join("opcons.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    OpconsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
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

fn load_or_generate_opcons_epsilon_table(
    work_dir: &Path,
    atomic_number: usize,
) -> Result<EpsilonTable> {
    let symbol = atomic_symbol(atomic_number)
        .with_context(|| format!("invalid OPCONS atomic number {atomic_number}"))?;
    let path = work_dir.join(format!("opcons{symbol}.dat"));
    if path.is_file() {
        return read_opcons_epsilon_table(work_dir, atomic_number);
    }

    let table = epsilon_table_from_database(atomic_number)
        .with_context(|| format!("no FEFF OPCONS epsdb source is available for {symbol}"))?;
    write_opcons_epsilon_table(&path, &table)?;
    // FEFF's epsdb writes with IOMod's default E20.10 format and AddEps then
    // reads that text back. Re-read here to preserve the same decimal
    // round-trip before interpolation and combination.
    read_opcons_epsilon_table(work_dir, atomic_number)
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

fn write_opcons_epsilon_table(path: &Path, table: &EpsilonTable) -> Result<()> {
    if table.epsilon1_minus_one.len() != table.point_count()
        || table.epsilon2.len() != table.point_count()
    {
        bail!(
            "cannot write {}: OPCONS epsilon table columns have inconsistent lengths",
            path.display()
        );
    }

    let mut out = String::new();
    for ((energy, epsilon1), epsilon2) in table
        .energy_ev
        .iter()
        .zip(table.epsilon1_minus_one.iter())
        .zip(table.epsilon2.iter())
    {
        // Rust normalizes E notation to one digit before the decimal, while
        // Fortran E20.10 uses `0.xxxxxxxxxx`; nine fractional digits therefore
        // retain the same ten significant digits as FEFF's source-table write.
        writeln!(out, "{energy:20.9E} {epsilon1:20.9E} {epsilon2:20.9E}")?;
    }
    std::fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))
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

#[cfg(test)]
mod tests {
    use super::{has_complete_table_inputs, read_opcons_epsilon_table, run_in_dir};
    use anyhow::Result;
    use refeff_io::read_loss_dat;

    #[test]
    fn opcons_module_generates_missing_copper_table_from_feff_epsdb() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_single_component_input(temp.path())?;
        write_single_component_feff_input(temp.path())?;

        assert!(!temp.path().join("opconsCu.dat").exists());
        assert!(has_complete_table_inputs(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 181);
        let table = read_opcons_epsilon_table(temp.path(), 29)?;
        assert_eq!(table.point_count(), 182);
        assert_eq!(table.energy_ev[0], 0.250_657_997_1e-2);
        assert_eq!(table.epsilon1_minus_one[0], -0.711_151_962_3e2);
        assert_eq!(table.epsilon2[0], 0.435_246_002_2e3);
        assert_eq!(table.energy_ev[181], 0.100_000_000_0e6);
        let loss = read_loss_dat(temp.path().join("loss.dat"))?;
        assert_eq!(loss.point_count(), 181);
        assert!((loss.loss[0] - 0.002_239_44).abs() < 1.0e-8);
        Ok(())
    }

    #[test]
    fn opcons_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("opcons.inp"),
            b"not an opcons.inp handoff\n",
        )?;

        assert!(!has_complete_table_inputs(temp.path())?);
        let error = run_in_dir(temp.path())
            .expect_err("malformed OPCONS input should fail through explicit run");
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("opcons.inp"), "{chain}");
        assert!(!temp.path().join("loss.dat").exists());
        assert!(!temp.path().join("epsilon.dat").exists());
        Ok(())
    }

    #[test]
    fn opcons_module_does_not_claim_malformed_pot_source_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_single_component_input(temp.path())?;
        write_single_component_table(temp.path())?;
        std::fs::write(temp.path().join("pot.bin"), b"not a pot.bin source\n")?;

        assert!(!has_complete_table_inputs(temp.path())?);
        let error = run_in_dir(temp.path())
            .expect_err("malformed OPCONS pot source should fail through explicit run");
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("pot.bin"), "{chain}");
        assert!(!temp.path().join("loss.dat").exists());
        assert!(!temp.path().join("epsilon.dat").exists());
        Ok(())
    }

    #[test]
    fn opcons_module_does_not_claim_orphan_table_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("opconsCu.dat"),
            concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n"),
        )?;

        assert!(!has_complete_table_inputs(temp.path())?);
        assert!(!temp.path().join("loss.dat").exists());
        assert!(!temp.path().join("epsilon.dat").exists());
        Ok(())
    }

    fn write_single_component_input(work_dir: &std::path::Path) -> Result<()> {
        std::fs::write(
            work_dir.join("opcons.inp"),
            "run_opcons\n T\nprint_eps\n F\nNumDens(0:nphx)\n  1.0000000000000000\n",
        )?;
        Ok(())
    }

    fn write_single_component_feff_input(work_dir: &std::path::Path) -> Result<()> {
        std::fs::write(
            work_dir.join("feff.inp"),
            "TITLE Cu OPCONS epsdb source test\nPOTENTIALS\n0 29 Cu\nEND\n",
        )?;
        Ok(())
    }

    fn write_single_component_table(work_dir: &std::path::Path) -> Result<()> {
        std::fs::write(
            work_dir.join("opconsCu.dat"),
            concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n"),
        )?;
        Ok(())
    }
}
