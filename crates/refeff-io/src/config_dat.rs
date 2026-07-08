//! FEFF `config.dat` electron-configuration output support.
//!
//! FEFF writes `config.dat` from the atomic configuration module after applying
//! core-hole, screening, and ionicity adjustments. Each potential record stores
//! forty occupation values and forty valence-occupation values, with an
//! optional forty-value spin row in older dump paths.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2};
use refeff_core::{
    FEFF_ORBITAL_KAPPAS, FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS, FEFF_ORBITAL_SLOT_COUNT,
    OrbitalConfiguration,
};

use crate::error::{IoError, Result};

/// Number of orbital slots written in each FEFF `config.dat` occupation row.
pub const CONFIG_DAT_ORBITAL_COUNT: usize = 40;

const CONFIG_DAT_LEGACY_ORBITAL_COUNT: usize = 29;
const CONFIG_DAT_PATH: &str = "config.dat";

/// One potential record from FEFF `config.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigDatPotential {
    /// FEFF potential index, `iph`.
    pub potential_index: i32,
    /// Atomic number for this potential.
    pub atomic_number: i32,
    /// Element label written by FEFF.
    pub element: String,
    /// Atomic occupation numbers, `iocc(1:40)`.
    pub occupations: Array1<f64>,
    /// Valence occupation numbers, `ival(1:40)`.
    pub valence_occupations: Array1<f64>,
    /// Optional spin occupation numbers, `ispn(1:40)`.
    pub spin_occupations: Option<Array1<f64>>,
}

/// Parsed contents of FEFF `config.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigDatData {
    /// Comment/header lines before the potential records.
    pub header_lines: Vec<String>,
    /// Potential records in FEFF file order.
    pub potentials: Vec<ConfigDatPotential>,
}

/// Compacted orbital metadata from FEFF `config.dat` for RHORRP wavefunctions.
///
/// FEFF writes `config.dat` after applying core-hole, screening, and ionicity
/// adjustments. RHORRP needs the same adjusted `xnel`, `kap`, and per-potential
/// `norb` metadata, compacted from the 40 FEFF orbital slots in file order.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpConfigOrbitalTables {
    /// Adjusted electron occupations, shaped `(max_orbitals, potentials)`.
    pub electron_counts_by_potential: Array2<f64>,
    /// Adjusted valence occupations compacted in the same order as `electron_counts_by_potential`.
    pub valence_counts_by_potential: Array2<f64>,
    /// Relativistic kappa for each compacted orbital, shaped like `electron_counts_by_potential`.
    pub kappa_by_potential: Array2<i32>,
    /// Zero-based FEFF orbital slot for each compacted orbital, shaped like `electron_counts_by_potential`.
    pub orbital_slots_by_potential: Array2<usize>,
    /// Number of occupied compacted orbitals for each potential.
    pub bound_orbital_counts: Vec<usize>,
}

impl ConfigDatData {
    /// Number of potential records.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.potentials.len()
    }

    /// Whether any potential record includes a spin row.
    #[must_use]
    pub fn has_spin_occupations(&self) -> bool {
        self.potentials
            .iter()
            .any(|potential| potential.spin_occupations.is_some())
    }
}

/// Parse FEFF `config.dat` text.
pub fn parse_config_dat(text: &str) -> Result<ConfigDatData> {
    let mut lines = text.lines().enumerate().peekable();
    let mut header_lines = Vec::new();
    let mut has_spin_header = false;
    while let Some((_, raw)) = lines.peek() {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            lines.next();
            continue;
        }
        if !line.trim_start().starts_with('#') {
            break;
        }
        has_spin_header |= line.contains("ispn");
        header_lines.push(line.to_string());
        lines.next();
    }

    let mut potentials = Vec::new();
    while let Some((index, raw)) = lines.next() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            header_lines.push(raw.trim_end().to_string());
            has_spin_header |= line.contains("ispn");
            continue;
        }
        let line_number = index + 1;
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 3 {
            return parse_error(
                line_number,
                format!(
                    "potential row has {} token(s), expected at least 3",
                    tokens.len()
                ),
            );
        }
        validate_occupation_token_count(line_number, "potential row", tokens.len() - 3)?;
        let potential_index = parse_i32(line_number, "potential_index", tokens[0])?;
        let atomic_number = parse_i32(line_number, "atomic_number", tokens[1])?;
        let element = parse_element(line_number, tokens[2])?;
        let occupations = parse_occupation_values(line_number, &tokens[3..])?;

        let (valence_line_number, valence_line) =
            next_nonempty_line(&mut lines, "valence occupation row")?;
        let valence_tokens = valence_line.split_whitespace().collect::<Vec<_>>();
        validate_occupation_token_count(
            valence_line_number,
            "valence occupation row",
            valence_tokens.len(),
        )?;
        let valence_occupations = parse_occupation_values(valence_line_number, &valence_tokens)?;

        let spin_occupations = if has_spin_header {
            let (spin_line_number, spin_line) =
                next_nonempty_line(&mut lines, "spin occupation row")?;
            let spin_tokens = spin_line.split_whitespace().collect::<Vec<_>>();
            validate_occupation_token_count(
                spin_line_number,
                "spin occupation row",
                spin_tokens.len(),
            )?;
            Some(parse_occupation_values(spin_line_number, &spin_tokens)?)
        } else {
            None
        };

        potentials.push(ConfigDatPotential {
            potential_index,
            atomic_number,
            element,
            occupations,
            valence_occupations,
            spin_occupations,
        });
    }

    let data = ConfigDatData {
        header_lines,
        potentials,
    };
    validate_config_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `config.dat` text.
pub fn config_dat_string(data: &ConfigDatData) -> Result<String> {
    validate_config_dat(data)?;
    let mut out = String::new();
    if data.header_lines.is_empty() {
        write_default_header(&mut out, data.has_spin_occupations())?;
    } else {
        for line in &data.header_lines {
            writeln!(out, "{line}")?;
        }
    }
    for potential in &data.potentials {
        write!(
            out,
            "{:3}  {:3}  {:<2}  ",
            potential.potential_index, potential.atomic_number, potential.element
        )?;
        write_occupation_values(&mut out, &potential.occupations)?;
        writeln!(out)?;
        write!(out, "{:14}", "")?;
        write_occupation_values(&mut out, &potential.valence_occupations)?;
        writeln!(out)?;
        if let Some(spin_occupations) = &potential.spin_occupations {
            write!(out, "{:14}", "")?;
            write_occupation_values(&mut out, spin_occupations)?;
            writeln!(out)?;
        }
    }
    Ok(out)
}

/// Build FEFF `config.dat` records from compacted `getorb` configurations.
///
/// FEFF `COMMON/m_config.f90::DumpConfig2` writes `config.dat` after core-hole,
/// screening, and ionicity adjustments by expanding compacted `(n, kappa)`
/// orbital arrays back into the 40-slot configuration table. This helper
/// performs the same expansion for one compacted configuration per potential
/// index. Spin rows are omitted, matching FEFF10's `DumpConfig2` default.
pub fn config_dat_from_orbital_configurations<S: AsRef<str>>(
    atomic_numbers: &[usize],
    elements: &[S],
    configurations: &[OrbitalConfiguration],
) -> Result<ConfigDatData> {
    if atomic_numbers.len() != elements.len() || atomic_numbers.len() != configurations.len() {
        return parse_error(
            0,
            format!(
                "config.dat builder got {} atomic number(s), {} element label(s), and {} configuration(s)",
                atomic_numbers.len(),
                elements.len(),
                configurations.len()
            ),
        );
    }

    let potentials = atomic_numbers
        .iter()
        .zip(elements.iter())
        .zip(configurations.iter())
        .enumerate()
        .map(|(index, ((atomic_number, element), configuration))| {
            let (occupations, valence_occupations) =
                expand_compacted_configuration(index + 1, configuration)?;
            Ok(ConfigDatPotential {
                potential_index: i32::try_from(index).map_err(|_| {
                    parse_error_value(index + 1, "potential index cannot be represented as i32")
                })?,
                atomic_number: i32::try_from(*atomic_number).map_err(|_| {
                    parse_error_value(index + 1, "atomic number cannot be represented as i32")
                })?,
                element: parse_element(index + 1, element.as_ref())?,
                occupations,
                valence_occupations,
                spin_occupations: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let data = ConfigDatData {
        header_lines: Vec::new(),
        potentials,
    };
    validate_config_dat(&data)?;
    Ok(data)
}

/// Compact already-adjusted FEFF `config.dat` rows into RHORRP orbital tables.
///
/// This is the inverse of the FEFF `config.dat` expansion for the pieces RHORRP
/// consumes: nonzero 40-slot occupation or valence-occupation entries become
/// compacted `xnel` rows, while `kap` is copied from FEFF's fixed slot map.
/// FEFF `config.dat` can store valence-only orbitals after core-valence
/// separation, so those rows are included with their valence occupation.
/// RHORRP receives SCF valence counts from `pot.bin` in the FEFF handoff path.
pub fn rhorrp_orbital_tables_from_config_dat(
    data: &ConfigDatData,
) -> Result<RhorrpConfigOrbitalTables> {
    validate_config_dat(data)?;

    let compacted = data
        .potentials
        .iter()
        .enumerate()
        .map(|(index, potential)| compact_rhorrp_config_potential(index + 1, index, potential))
        .collect::<Result<Vec<_>>>()?;
    let potential_count = compacted.len();
    let bound_orbital_counts = compacted
        .iter()
        .map(|potential| potential.electron_counts.len())
        .collect::<Vec<_>>();
    let max_orbitals = bound_orbital_counts.iter().copied().max().unwrap_or(0);

    let mut electron_counts_by_potential = Array2::zeros((max_orbitals, potential_count));
    let mut valence_counts_by_potential = Array2::zeros((max_orbitals, potential_count));
    let mut kappa_by_potential = Array2::zeros((max_orbitals, potential_count));
    let mut orbital_slots_by_potential = Array2::zeros((max_orbitals, potential_count));
    for (potential_index, potential) in compacted.iter().enumerate() {
        for orbital_index in 0..potential.electron_counts.len() {
            let electron_count = potential.electron_counts[orbital_index];
            let valence_count = potential.valence_counts[orbital_index];
            let kappa = potential.kappa[orbital_index];
            let slot = potential.orbital_slots[orbital_index];
            electron_counts_by_potential[(orbital_index, potential_index)] = electron_count;
            valence_counts_by_potential[(orbital_index, potential_index)] = valence_count;
            kappa_by_potential[(orbital_index, potential_index)] = kappa;
            orbital_slots_by_potential[(orbital_index, potential_index)] = slot;
        }
    }

    Ok(RhorrpConfigOrbitalTables {
        electron_counts_by_potential,
        valence_counts_by_potential,
        kappa_by_potential,
        orbital_slots_by_potential,
        bound_orbital_counts,
    })
}

/// Read FEFF `config.dat` text from a file.
pub fn read_config_dat(path: impl AsRef<Path>) -> Result<ConfigDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_config_dat(&text)
}

/// Write FEFF `config.dat` text to a file.
pub fn write_config_dat(path: impl AsRef<Path>, data: &ConfigDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, config_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

#[derive(Debug, Clone, PartialEq)]
struct CompactedRhorrpConfigPotential {
    electron_counts: Vec<f64>,
    valence_counts: Vec<f64>,
    kappa: Vec<i32>,
    orbital_slots: Vec<usize>,
}

fn compact_rhorrp_config_potential(
    row: usize,
    expected_potential_index: usize,
    potential: &ConfigDatPotential,
) -> Result<CompactedRhorrpConfigPotential> {
    let potential_index = usize::try_from(potential.potential_index)
        .map_err(|_| parse_error_value(row, "potential index must be non-negative"))?;
    if potential_index != expected_potential_index {
        return parse_error(
            row,
            format!(
                "RHORRP orbital tables require contiguous potential index {expected_potential_index}, got {}",
                potential.potential_index
            ),
        );
    }

    let mut electron_counts = Vec::new();
    let mut valence_counts = Vec::new();
    let mut kappa = Vec::new();
    let mut orbital_slots = Vec::new();
    for (slot, ((&occupation, &valence_occupation), &slot_kappa)) in potential
        .occupations
        .iter()
        .zip(potential.valence_occupations.iter())
        .zip(FEFF_ORBITAL_KAPPAS.iter())
        .enumerate()
    {
        if occupation < 0.0 {
            return parse_error(row, format!("occupation slot {} is negative", slot + 1));
        }
        if valence_occupation < 0.0 {
            return parse_error(
                row,
                format!("valence occupation slot {} is negative", slot + 1),
            );
        }
        let electron_count = if occupation > 0.0 {
            occupation
        } else {
            valence_occupation
        };
        if electron_count > 0.0 {
            electron_counts.push(electron_count);
            valence_counts.push(valence_occupation);
            kappa.push(slot_kappa);
            orbital_slots.push(slot);
        }
    }

    if electron_counts.is_empty() {
        return parse_error(row, "at least one occupied orbital is required");
    }

    Ok(CompactedRhorrpConfigPotential {
        electron_counts,
        valence_counts,
        kappa,
        orbital_slots,
    })
}

fn write_default_header(out: &mut String, with_spin: bool) -> Result<()> {
    writeln!(out, "# Configuration of all atom types in feff.inp.")?;
    writeln!(
        out,
        "# Atomic occupation numbers including core hole, screening, and ionicity (but no SCF)."
    )?;
    if with_spin {
        writeln!(out, "# iph, z,name,  iocc/ival/ispn (i=1,40)")?;
    } else {
        writeln!(out, "# iph, z,name,  iocc/ival (i=1,40)")?;
    }
    Ok(())
}

fn write_occupation_values(out: &mut String, values: &Array1<f64>) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(out, "  ")?;
        }
        write!(out, "{value:5.2}")?;
    }
    Ok(())
}

fn parse_occupation_values(line_number: usize, tokens: &[&str]) -> Result<Array1<f64>> {
    validate_occupation_token_count(line_number, "occupation row", tokens.len())?;
    let values = tokens
        .iter()
        .map(|token| parse_f64(line_number, "occupation", token))
        .collect::<Result<Vec<_>>>()
        .map(Array1::from_vec)?;
    Ok(pad_occupation_values(values))
}

fn validate_occupation_token_count(line_number: usize, label: &str, count: usize) -> Result<usize> {
    match count {
        CONFIG_DAT_ORBITAL_COUNT | CONFIG_DAT_LEGACY_ORBITAL_COUNT => Ok(count),
        _ => parse_error(
            line_number,
            format!(
                "{label} has {count} occupation token(s), expected {CONFIG_DAT_LEGACY_ORBITAL_COUNT} or {CONFIG_DAT_ORBITAL_COUNT}"
            ),
        ),
    }
}

fn pad_occupation_values(values: Array1<f64>) -> Array1<f64> {
    if values.len() == CONFIG_DAT_ORBITAL_COUNT {
        return values;
    }
    let mut padded = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    for (index, value) in values.into_iter().enumerate() {
        padded[index] = value;
    }
    padded
}

fn parse_element(line_number: usize, token: &str) -> Result<String> {
    if token
        .chars()
        .all(|character| character.is_ascii_alphabetic())
    {
        Ok(token.to_string())
    } else {
        parse_error(line_number, format!("invalid element label {token:?}"))
    }
}

fn validate_config_dat(data: &ConfigDatData) -> Result<()> {
    if data.potentials.is_empty() {
        return parse_error(0, "at least one potential record is required");
    }
    for (index, potential) in data.potentials.iter().enumerate() {
        let row = index + 1;
        if potential.potential_index < 0 {
            return parse_error(row, "potential index must be non-negative");
        }
        if potential.atomic_number <= 0 {
            return parse_error(row, "atomic number must be positive");
        }
        if potential.element.is_empty() {
            return parse_error(row, "element label is required");
        }
        validate_occupation_array("occupations", &potential.occupations, row)?;
        validate_occupation_array("valence_occupations", &potential.valence_occupations, row)?;
        if let Some(spin_occupations) = &potential.spin_occupations {
            validate_occupation_array("spin_occupations", spin_occupations, row)?;
        }
    }
    Ok(())
}

fn validate_occupation_array(field: &'static str, values: &Array1<f64>, row: usize) -> Result<()> {
    if values.len() != CONFIG_DAT_ORBITAL_COUNT {
        return parse_error(
            row,
            format!(
                "{field} has {} value(s), expected {CONFIG_DAT_ORBITAL_COUNT}",
                values.len()
            ),
        );
    }
    for value in values {
        if !value.is_finite() {
            return parse_error(row, format!("{field} contains a non-finite value"));
        }
    }
    Ok(())
}

fn expand_compacted_configuration(
    row: usize,
    configuration: &OrbitalConfiguration,
) -> Result<(Array1<f64>, Array1<f64>)> {
    validate_compacted_len(
        row,
        "principal_quantum_numbers",
        configuration.principal_quantum_numbers.len(),
        configuration.orbital_count,
    )?;
    validate_compacted_len(
        row,
        "kappa",
        configuration.kappa.len(),
        configuration.orbital_count,
    )?;
    validate_compacted_len(
        row,
        "electron_counts",
        configuration.electron_counts.len(),
        configuration.orbital_count,
    )?;
    validate_compacted_len(
        row,
        "valence_counts",
        configuration.valence_counts.len(),
        configuration.orbital_count,
    )?;

    let mut occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    let mut valence_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    let mut assigned = [false; CONFIG_DAT_ORBITAL_COUNT];
    for compacted in 0..configuration.orbital_count {
        let n = configuration.principal_quantum_numbers[compacted];
        let kappa = configuration.kappa[compacted];
        let slot = orbital_slot_for_quantum_numbers(n, kappa).ok_or_else(|| {
            parse_error_value(
                row,
                format!(
                    "compacted orbital {} has unknown n={n}, kappa={kappa}",
                    compacted + 1
                ),
            )
        })?;
        if assigned[slot] {
            return parse_error(
                row,
                format!("multiple compacted orbitals map to FEFF slot {}", slot + 1),
            );
        }
        assigned[slot] = true;
        occupations[slot] = configuration.electron_counts[compacted];
        valence_occupations[slot] = configuration.valence_counts[compacted];
    }
    Ok((occupations, valence_occupations))
}

fn validate_compacted_len(
    row: usize,
    field: &'static str,
    actual: usize,
    required: usize,
) -> Result<()> {
    if actual < required {
        parse_error(
            row,
            format!("{field} has {actual} value(s), expected at least {required}"),
        )
    } else {
        Ok(())
    }
}

fn orbital_slot_for_quantum_numbers(n: i32, kappa: i32) -> Option<usize> {
    if CONFIG_DAT_ORBITAL_COUNT != FEFF_ORBITAL_SLOT_COUNT {
        return None;
    }
    FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS
        .iter()
        .zip(FEFF_ORBITAL_KAPPAS.iter())
        .position(|(&slot_n, &slot_kappa)| slot_n == n && slot_kappa == kappa)
}

fn next_nonempty_line<'a>(
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
    field: &'static str,
) -> Result<(usize, &'a str)> {
    for (index, raw) in lines {
        let line = raw.trim_end();
        if !line.trim().is_empty() {
            return Ok((index + 1, line));
        }
    }
    parse_error(0, format!("missing {field}"))
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn parse_i32(line: usize, field: &'static str, token: &str) -> Result<i32> {
    token
        .parse::<i32>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: CONFIG_DAT_PATH.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use refeff_core::{OrbitalConfigurationInput, orbital_configuration};

    use super::*;

    #[test]
    fn parses_config_dat_records() -> Result<()> {
        let parsed = parse_config_dat(CONFIG_DAT)?;
        assert_eq!(parsed.potential_count(), 2);
        assert_eq!(parsed.header_lines.len(), 3);
        assert!(!parsed.has_spin_occupations());
        assert_eq!(parsed.potentials[0].potential_index, 0);
        assert_eq!(parsed.potentials[0].atomic_number, 29);
        assert_eq!(parsed.potentials[0].element, "Cu");
        assert_eq!(
            parsed.potentials[0].occupations.len(),
            CONFIG_DAT_ORBITAL_COUNT
        );
        assert_eq!(parsed.potentials[0].occupations[0], 1.0);
        assert_eq!(parsed.potentials[0].valence_occupations[7], 4.0);
        assert_eq!(parsed.potentials[1].occupations[0], 2.0);

        let rendered = config_dat_string(&parsed)?;
        assert_eq!(rendered, CONFIG_DAT);
        assert_eq!(parse_config_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_spin_config_dat_records() -> Result<()> {
        let parsed = parse_config_dat(SPIN_CONFIG_DAT)?;
        assert_eq!(parsed.potential_count(), 1);
        assert!(parsed.has_spin_occupations());
        let spin_occupations = parsed.potentials[0]
            .spin_occupations
            .as_ref()
            .ok_or_else(|| parse_error_value(0, "missing test spin row"))?;
        assert_eq!(spin_occupations[0], 1.0);
        assert_eq!(parse_config_dat(&config_dat_string(&parsed)?)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_legacy_twenty_nine_orbital_config_dat_by_padding_to_current_shape() -> Result<()> {
        let parsed = parse_config_dat(&legacy_config_dat_string())?;
        assert_eq!(parsed.potential_count(), 1);
        assert!(parsed.has_spin_occupations());
        let potential = &parsed.potentials[0];
        assert_eq!(potential.occupations.len(), CONFIG_DAT_ORBITAL_COUNT);
        assert_eq!(
            potential.valence_occupations.len(),
            CONFIG_DAT_ORBITAL_COUNT
        );
        assert_eq!(potential.occupations[0], 2.0);
        assert_eq!(potential.occupations[28], 3.0);
        assert_eq!(potential.occupations[29], 0.0);
        assert_eq!(potential.valence_occupations[8], 6.0);
        assert_eq!(potential.valence_occupations[29], 0.0);
        let spin = potential
            .spin_occupations
            .as_ref()
            .ok_or_else(|| parse_error_value(0, "missing legacy spin row"))?;
        assert_eq!(spin[8], 1.0);
        assert_eq!(spin[29], 0.0);
        Ok(())
    }

    #[test]
    fn rejects_bad_config_dat_inputs() {
        assert!(parse_config_dat("").is_err());
        assert!(parse_config_dat("# header only\n").is_err());
        assert!(parse_config_dat(&CONFIG_DAT.replace("Cu", "C1")).is_err());
        assert!(parse_config_dat(&CONFIG_DAT.replace("29", "0")).is_err());
        assert!(parse_config_dat(&CONFIG_DAT.replace("1.00", "NaN")).is_err());
        assert!(parse_config_dat(&CONFIG_DAT.replacen("   0.00   0.00", "   0.00", 1)).is_err());
    }

    #[test]
    fn compacts_rhorrp_orbital_tables_from_config_dat() -> Result<()> {
        let parsed = parse_config_dat(CONFIG_DAT)?;

        let tables = rhorrp_orbital_tables_from_config_dat(&parsed)?;

        assert_eq!(tables.bound_orbital_counts, vec![11, 10]);
        assert_eq!(tables.electron_counts_by_potential.shape(), &[11, 2]);
        assert_eq!(tables.valence_counts_by_potential.shape(), &[11, 2]);
        assert_eq!(tables.kappa_by_potential.shape(), &[11, 2]);
        assert_eq!(tables.orbital_slots_by_potential.shape(), &[11, 2]);
        assert_eq!(tables.electron_counts_by_potential[(0, 0)], 1.0);
        assert_eq!(tables.electron_counts_by_potential[(10, 0)], 1.0);
        assert_eq!(tables.electron_counts_by_potential[(10, 1)], 0.0);
        assert_eq!(tables.valence_counts_by_potential[(10, 0)], 1.0);
        assert_eq!(tables.valence_counts_by_potential[(10, 1)], 0.0);
        assert_eq!(tables.kappa_by_potential[(0, 0)], -1);
        assert_eq!(tables.kappa_by_potential[(10, 0)], 1);
        assert_eq!(tables.kappa_by_potential[(10, 1)], 0);
        assert_eq!(tables.orbital_slots_by_potential[(0, 0)], 0);
        assert_eq!(tables.orbital_slots_by_potential[(10, 0)], 10);
        assert_eq!(tables.orbital_slots_by_potential[(10, 1)], 0);
        Ok(())
    }

    #[test]
    fn rejects_invalid_rhorrp_orbital_table_inputs() -> Result<()> {
        let parsed = parse_config_dat(CONFIG_DAT)?;

        let mut valence_without_occupation = parsed.clone();
        valence_without_occupation.potentials[0].occupations[7] = 0.0;
        valence_without_occupation.potentials[0].valence_occupations[7] = 4.0;
        let valence_only = rhorrp_orbital_tables_from_config_dat(&valence_without_occupation)?;
        assert_eq!(valence_only.bound_orbital_counts[0], 11);
        assert_eq!(valence_only.electron_counts_by_potential[(7, 0)], 4.0);
        assert_eq!(valence_only.valence_counts_by_potential[(7, 0)], 4.0);
        assert_eq!(valence_only.orbital_slots_by_potential[(7, 0)], 7);

        let mut no_occupied_orbitals = parsed.clone();
        no_occupied_orbitals.potentials[0].occupations.fill(0.0);
        no_occupied_orbitals.potentials[0]
            .valence_occupations
            .fill(0.0);
        assert!(rhorrp_orbital_tables_from_config_dat(&no_occupied_orbitals).is_err());

        let mut negative_occupation = parsed.clone();
        negative_occupation.potentials[0].occupations[0] = -1.0;
        assert!(rhorrp_orbital_tables_from_config_dat(&negative_occupation).is_err());

        let mut swapped_potentials = parsed;
        swapped_potentials.potentials.swap(0, 1);
        assert!(rhorrp_orbital_tables_from_config_dat(&swapped_potentials).is_err());
        Ok(())
    }

    #[test]
    fn builds_config_dat_from_compacted_orbital_configuration() -> anyhow::Result<()> {
        let (occupations, valence_occupations, spin_occupations) = copper_slot_rows();
        let configuration = orbital_configuration(OrbitalConfigurationInput {
            atomic_number: 29,
            hole_index: 0,
            ionicity: 0.0,
            unfreeze_f_or_higher: false,
            occupations: occupations.view(),
            valence_occupations: valence_occupations.view(),
            spin_occupations: spin_occupations.view(),
            next_occupations: occupations.view(),
        })?;

        let data = config_dat_from_orbital_configurations(&[29], &["Cu"], &[configuration])?;

        assert_eq!(data.potential_count(), 1);
        assert!(!data.has_spin_occupations());
        assert_eq!(data.potentials[0].potential_index, 0);
        assert_eq!(data.potentials[0].atomic_number, 29);
        assert_eq!(data.potentials[0].element, "Cu");
        assert_eq!(data.potentials[0].occupations, occupations);
        assert_eq!(data.potentials[0].valence_occupations, valence_occupations);
        assert!(config_dat_string(&data)?.contains("including core hole, screening, and ionicity"));
        Ok(())
    }

    #[test]
    fn rejects_invalid_compacted_config_dat_build_inputs() -> anyhow::Result<()> {
        let (occupations, valence_occupations, spin_occupations) = copper_slot_rows();
        let configuration = orbital_configuration(OrbitalConfigurationInput {
            atomic_number: 29,
            hole_index: 0,
            ionicity: 0.0,
            unfreeze_f_or_higher: false,
            occupations: occupations.view(),
            valence_occupations: valence_occupations.view(),
            spin_occupations: spin_occupations.view(),
            next_occupations: occupations.view(),
        })?;

        assert!(
            config_dat_from_orbital_configurations(
                &[29],
                &["Cu", "O"],
                std::slice::from_ref(&configuration),
            )
            .is_err()
        );

        let mut invalid = configuration;
        invalid.principal_quantum_numbers[0] = 99;
        assert!(config_dat_from_orbital_configurations(&[29], &["Cu"], &[invalid]).is_err());
        Ok(())
    }

    fn copper_slot_rows() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let mut occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
        let mut valence_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
        let spin_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
        for (slot, occupation) in [
            (1, 2.0),
            (2, 2.0),
            (3, 2.0),
            (4, 4.0),
            (5, 2.0),
            (6, 2.0),
            (7, 4.0),
            (8, 4.0),
            (9, 6.0),
            (10, 1.0),
        ] {
            occupations[slot - 1] = occupation;
        }
        for (slot, occupation) in [(8, 4.0), (9, 6.0), (10, 1.0)] {
            valence_occupations[slot - 1] = occupation;
        }
        (occupations, valence_occupations, spin_occupations)
    }

    fn legacy_config_dat_string() -> String {
        let mut occupations = vec![0.0; CONFIG_DAT_LEGACY_ORBITAL_COUNT];
        let mut valence = vec![0.0; CONFIG_DAT_LEGACY_ORBITAL_COUNT];
        let mut spin = vec![0.0; CONFIG_DAT_LEGACY_ORBITAL_COUNT];
        occupations[0] = 2.0;
        occupations[1] = 2.0;
        occupations[8] = 6.0;
        occupations[28] = 3.0;
        valence[8] = 6.0;
        spin[8] = 1.0;
        format!(
            "# Configuration of all atom types in feff.inp.\n# iph, z,name,  iocc/ival/ispn (i=1,29)\n  0   29  Cu  {}\n              {}\n              {}\n",
            occupation_row(&occupations),
            occupation_row(&valence),
            occupation_row(&spin),
        )
    }

    fn occupation_row(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:5.2}"))
            .collect::<Vec<_>>()
            .join("   ")
    }

    const CONFIG_DAT: &str = r#"# Configuration of all atom types in feff.inp.
  # Atomic occupation numbers including core hole, screening, and ionicity (but no SCF).
# iph, z,name,  iocc/ival (i=1,40)
  0   29  Cu   1.00   2.00   2.00   4.00   2.00   2.00   4.00   4.00   6.00   1.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
               0.00   0.00   0.00   0.00   0.00   0.00   0.00   4.00   6.00   1.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  1   29  Cu   2.00   2.00   2.00   4.00   2.00   2.00   4.00   4.00   6.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
               0.00   0.00   0.00   0.00   0.00   0.00   0.00   4.00   6.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
"#;

    const SPIN_CONFIG_DAT: &str = r#"# Configuration of all atom types in feff.inp.
# iph, z,name,  iocc/ival/ispn (i=1,40)
  0   29  Cu   1.00   2.00   2.00   4.00   2.00   2.00   4.00   4.00   6.00   1.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
               0.00   0.00   0.00   0.00   0.00   0.00   0.00   4.00   6.00   1.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
               1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
"#;
}
