//! FEFF `config.dat` electron-configuration output support.
//!
//! FEFF writes `config.dat` from the atomic configuration module after applying
//! core-hole, screening, and ionicity adjustments. Each potential record stores
//! forty occupation values and forty valence-occupation values, with an
//! optional forty-value spin row in older dump paths.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;
use refeff_core::{
    FEFF_ORBITAL_KAPPAS, FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS, FEFF_ORBITAL_SLOT_COUNT,
    OrbitalConfiguration,
};

use crate::error::{IoError, Result};

/// Number of orbital slots written in each FEFF `config.dat` occupation row.
pub const CONFIG_DAT_ORBITAL_COUNT: usize = 40;

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
        if tokens.len() != CONFIG_DAT_ORBITAL_COUNT + 3 {
            return parse_error(
                line_number,
                format!(
                    "potential row has {} token(s), expected {}",
                    tokens.len(),
                    CONFIG_DAT_ORBITAL_COUNT + 3
                ),
            );
        }
        let potential_index = parse_i32(line_number, "potential_index", tokens[0])?;
        let atomic_number = parse_i32(line_number, "atomic_number", tokens[1])?;
        let element = parse_element(line_number, tokens[2])?;
        let occupations = parse_occupation_values(line_number, &tokens[3..])?;

        let (valence_line_number, valence_line) =
            next_nonempty_line(&mut lines, "valence occupation row")?;
        let valence_tokens = valence_line.split_whitespace().collect::<Vec<_>>();
        if valence_tokens.len() != CONFIG_DAT_ORBITAL_COUNT {
            return parse_error(
                valence_line_number,
                format!(
                    "valence occupation row has {} token(s), expected {CONFIG_DAT_ORBITAL_COUNT}",
                    valence_tokens.len()
                ),
            );
        }
        let valence_occupations = parse_occupation_values(valence_line_number, &valence_tokens)?;

        let spin_occupations = if has_spin_header {
            let (spin_line_number, spin_line) =
                next_nonempty_line(&mut lines, "spin occupation row")?;
            let spin_tokens = spin_line.split_whitespace().collect::<Vec<_>>();
            if spin_tokens.len() != CONFIG_DAT_ORBITAL_COUNT {
                return parse_error(
                    spin_line_number,
                    format!(
                        "spin occupation row has {} token(s), expected {CONFIG_DAT_ORBITAL_COUNT}",
                        spin_tokens.len()
                    ),
                );
            }
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
    if tokens.len() != CONFIG_DAT_ORBITAL_COUNT {
        return parse_error(
            line_number,
            format!(
                "occupation row has {} token(s), expected {CONFIG_DAT_ORBITAL_COUNT}",
                tokens.len()
            ),
        );
    }
    tokens
        .iter()
        .map(|token| parse_f64(line_number, "occupation", token))
        .collect::<Result<Vec<_>>>()
        .map(Array1::from_vec)
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
    fn rejects_bad_config_dat_inputs() {
        assert!(parse_config_dat("").is_err());
        assert!(parse_config_dat("# header only\n").is_err());
        assert!(parse_config_dat(&CONFIG_DAT.replace("Cu", "C1")).is_err());
        assert!(parse_config_dat(&CONFIG_DAT.replace("29", "0")).is_err());
        assert!(parse_config_dat(&CONFIG_DAT.replace("1.00", "NaN")).is_err());
        assert!(parse_config_dat(&CONFIG_DAT.replacen("   0.00   0.00", "   0.00", 1)).is_err());
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
