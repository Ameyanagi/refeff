//! FEFF `fpf0.dat` atomic form-factor output support.
//!
//! FEFF's ATOM stage writes `fpf0.dat` for downstream anomalous-scattering
//! calculations. The file contains the absorber atomic number, scalar f-prime
//! corrections, dipole oscillator records, and a tabulated nonresonant
//! form-factor grid `f0(Q)`.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

const FPF0_PATH: &str = "fpf0.dat";

/// One oscillator-strength row from FEFF `fpf0.dat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fpf0Oscillator {
    /// FEFF oscillator strength for this transition.
    pub oscillator_strength: f64,
    /// Bound-orbital energy in FEFF atomic units.
    pub excitation_energy: f64,
    /// FEFF orbital index associated with this oscillator row.
    pub orbital_index: usize,
}

/// Parsed contents of FEFF `fpf0.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct Fpf0DatData {
    /// Absorber atomic number from the `atom Z =` header line.
    pub atomic_number: i32,
    /// Total-energy contribution to f-prime, written as `5/3*E_tot/mc**2`.
    pub total_energy_fprime: f64,
    /// FEFF relativistic correction term, `fpcorr`.
    pub relativistic_correction: f64,
    /// Dipole oscillator table in FEFF file order.
    pub oscillators: Vec<Fpf0Oscillator>,
    /// Momentum-transfer grid `Q` in inverse Angstrom.
    pub form_factor_momentum: Array1<f64>,
    /// Nonresonant atomic form-factor table `f0(Q)`.
    pub form_factor: Array1<f64>,
}

impl Fpf0DatData {
    /// Number of oscillator rows.
    #[must_use]
    pub fn oscillator_count(&self) -> usize {
        self.oscillators.len()
    }

    /// Number of `f0(Q)` table rows.
    #[must_use]
    pub fn form_factor_count(&self) -> usize {
        self.form_factor.len()
    }
}

/// Parse FEFF `fpf0.dat` text.
pub fn parse_fpf0_dat(text: &str) -> Result<Fpf0DatData> {
    let mut lines = text.lines().enumerate();
    let (z_line_number, z_line) = next_nonempty_line(&mut lines, "atom Z header")?;
    let atomic_number = parse_atomic_number(z_line_number, z_line)?;

    let (correction_line_number, correction_line) =
        next_nonempty_line(&mut lines, "f-prime correction row")?;
    let correction_tokens = correction_line.split_whitespace().collect::<Vec<_>>();
    if correction_tokens.len() < 2 {
        return parse_error(
            correction_line_number,
            format!(
                "correction row has {} token(s), expected at least 2",
                correction_tokens.len()
            ),
        );
    }
    let total_energy_fprime = parse_f64(
        correction_line_number,
        "total_energy_fprime",
        correction_tokens[0],
    )?;
    let relativistic_correction = parse_f64(
        correction_line_number,
        "relativistic_correction",
        correction_tokens[1],
    )?;

    let (count_line_number, count_line) = next_nonempty_line(&mut lines, "oscillator count")?;
    let count_tokens = count_line.split_whitespace().collect::<Vec<_>>();
    if count_tokens.len() != 1 {
        return parse_error(
            count_line_number,
            format!(
                "oscillator-count row has {} token(s), expected 1",
                count_tokens.len()
            ),
        );
    }
    let oscillator_count = parse_usize(count_line_number, "oscillator_count", count_tokens[0])?;

    let oscillators = (0..oscillator_count)
        .map(|_| parse_oscillator_row(&mut lines))
        .collect::<Result<Vec<_>>>()?;

    let mut form_factor_momentum = Vec::new();
    let mut form_factor = Vec::new();
    for (index, raw) in lines {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let line_number = index + 1;
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 2 {
            return parse_error(
                line_number,
                format!("form-factor row has {} token(s), expected 2", tokens.len()),
            );
        }
        form_factor_momentum.push(parse_f64(line_number, "form_factor_momentum", tokens[0])?);
        form_factor.push(parse_f64(line_number, "form_factor", tokens[1])?);
    }

    let data = Fpf0DatData {
        atomic_number,
        total_energy_fprime,
        relativistic_correction,
        oscillators,
        form_factor_momentum: Array1::from_vec(form_factor_momentum),
        form_factor: Array1::from_vec(form_factor),
    };
    validate_fpf0_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `fpf0.dat` text.
pub fn fpf0_dat_string(data: &Fpf0DatData) -> Result<String> {
    validate_fpf0_dat(data)?;
    let mut out = String::new();
    writeln!(out, "  atom Z = {:12}", data.atomic_number)?;
    write_fortran_exp(&mut out, data.total_energy_fprime, 19, 5)?;
    write_fortran_exp(&mut out, data.relativistic_correction, 19, 5)?;
    out.push_str(" total energy part of fprime - 5/3*E_tot/mc**2\n");
    writeln!(out, "{:12}", data.oscillator_count())?;
    for oscillator in &data.oscillators {
        writeln!(
            out,
            "{:9.5} {:11.3} {:3}",
            oscillator.oscillator_strength, oscillator.excitation_energy, oscillator.orbital_index
        )?;
    }
    for (momentum, form_factor) in data
        .form_factor_momentum
        .iter()
        .zip(data.form_factor.iter())
    {
        writeln!(out, "{momentum:5.1} {form_factor:9.4}")?;
    }
    Ok(out)
}

/// Read FEFF `fpf0.dat` text from a file.
pub fn read_fpf0_dat(path: impl AsRef<Path>) -> Result<Fpf0DatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_fpf0_dat(&text)
}

/// Write FEFF `fpf0.dat` text to a file.
pub fn write_fpf0_dat(path: impl AsRef<Path>, data: &Fpf0DatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, fpf0_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

fn parse_oscillator_row<'a>(
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
) -> Result<Fpf0Oscillator> {
    let (line_number, line) = next_nonempty_line(lines, "oscillator row")?;
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 3 {
        return parse_error(
            line_number,
            format!("oscillator row has {} token(s), expected 3", tokens.len()),
        );
    }
    Ok(Fpf0Oscillator {
        oscillator_strength: parse_f64(line_number, "oscillator_strength", tokens[0])?,
        excitation_energy: parse_f64(line_number, "excitation_energy", tokens[1])?,
        orbital_index: parse_usize(line_number, "orbital_index", tokens[2])?,
    })
}

fn parse_atomic_number(line_number: usize, line: &str) -> Result<i32> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if !tokens
        .windows(2)
        .any(|window| window[0].eq_ignore_ascii_case("atom") && window[1].eq_ignore_ascii_case("Z"))
    {
        return parse_error(line_number, "expected atom Z header");
    }
    let token = tokens
        .last()
        .ok_or_else(|| parse_error_value(line_number, "missing atomic number"))?;
    parse_i32(line_number, "atomic_number", token)
}

fn validate_fpf0_dat(data: &Fpf0DatData) -> Result<()> {
    if data.atomic_number <= 0 {
        return parse_error(1, "atomic number must be positive");
    }
    validate_finite("total_energy_fprime", data.total_energy_fprime, 2)?;
    validate_finite("relativistic_correction", data.relativistic_correction, 2)?;
    if data.oscillators.is_empty() {
        return parse_error(0, "at least one oscillator row is required");
    }
    if data.form_factor_count() == 0 {
        return parse_error(0, "at least one form-factor row is required");
    }
    if data.form_factor_momentum.len() != data.form_factor_count() {
        return parse_error(
            0,
            format!(
                "form-factor momentum count {} does not match form-factor count {}",
                data.form_factor_momentum.len(),
                data.form_factor_count()
            ),
        );
    }
    for (index, oscillator) in data.oscillators.iter().enumerate() {
        let row = index + 1;
        validate_finite("oscillator_strength", oscillator.oscillator_strength, row)?;
        validate_finite("excitation_energy", oscillator.excitation_energy, row)?;
        if oscillator.orbital_index == 0 {
            return parse_error(row, "orbital index must be positive");
        }
    }
    for (index, (momentum, form_factor)) in data
        .form_factor_momentum
        .iter()
        .zip(data.form_factor.iter())
        .enumerate()
    {
        let row = index + 1;
        validate_finite("form_factor_momentum", *momentum, row)?;
        validate_finite("form_factor", *form_factor, row)?;
    }
    Ok(())
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

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token
        .parse::<usize>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(row, format!("{field} must be finite"))
    }
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: FPF0_PATH.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fpf0_dat() -> Result<()> {
        let parsed = parse_fpf0_dat(FPF0_DAT)?;
        assert_eq!(parsed.atomic_number, 29);
        assert_eq!(parsed.total_energy_fprime, -0.146_689);
        assert_eq!(parsed.relativistic_correction, -0.083_924_2);
        assert_eq!(parsed.oscillator_count(), 3);
        assert_eq!(parsed.oscillators[0].oscillator_strength, 2.0);
        assert_eq!(parsed.oscillators[0].excitation_energy, -332.657);
        assert_eq!(parsed.oscillators[0].orbital_index, 1);
        assert_eq!(parsed.form_factor_count(), 4);
        assert_eq!(
            parsed.form_factor_momentum.as_slice(),
            Some(&[0.0, 0.5, 1.0, 1.5][..])
        );
        assert_eq!(parsed.form_factor[1], 28.643);

        let rendered = fpf0_dat_string(&parsed)?;
        assert_eq!(rendered, FPF0_DAT);
        assert_eq!(parse_fpf0_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn rejects_bad_fpf0_dat_inputs() {
        assert!(parse_fpf0_dat("").is_err());
        assert!(parse_fpf0_dat("atom A = 29\n").is_err());
        assert!(parse_fpf0_dat(&FPF0_DAT.replace("29", "0")).is_err());
        assert!(parse_fpf0_dat(&FPF0_DAT.replacen("           3\n", "           0\n", 1)).is_err());
        assert!(parse_fpf0_dat(&FPF0_DAT.replace("0.0   29.0000", "0.0 NaN")).is_err());
        assert!(parse_fpf0_dat(&FPF0_DAT.replace("2.00000", "2.00000 3.0")).is_err());
    }

    const FPF0_DAT: &str = r#"  atom Z =           29
       -1.46689E-01       -8.39242E-02 total energy part of fprime - 5/3*E_tot/mc**2
           3
  2.00000    -332.657   1
  0.00162     -36.320   3
  0.00317     -35.556   4
  0.0   29.0000
  0.5   28.6430
  1.0   27.6260
  1.5   26.0430
"#;
}
