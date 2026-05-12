//! FEFF `xsecl.bin` text/PAD atomic cross-section decomposition codec.
//!
//! `XSPH/xsectjas.f90` writes this handoff when `ldecmx >= 0`. The file starts
//! with the FEFF final-state count, transition-index count, and doubled initial
//! angular momentum `jinit`, followed by one `kiind/lgind/ljind/lind` row per
//! transition index. The remaining payload is one complex PAD block per energy,
//! each containing `atomxsec(1:kfinmax, ie)`.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array2, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::pad::{decode_complex, encode_complex};

/// FEFF `xsecl.bin` transition-index metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XseclBinTransition {
    /// Relativistic final-state kappa index, `kiind`.
    pub final_state_kappa: i32,
    /// Decomposition angular-momentum channel, `lgind`.
    pub decomposition_channel: i32,
    /// Total-angular-momentum channel, `ljind`.
    pub total_angular_momentum_channel: i32,
    /// Orbital angular-momentum channel, `lind`.
    pub orbital_angular_momentum: i32,
}

/// FEFF `xsecl.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct XseclBinData {
    /// PAD field width, `npadx`, supplied by the neighboring FEFF handoff files.
    pub pad_width: usize,
    /// Doubled initial total angular momentum, `jinit`.
    pub initial_state_j: i32,
    /// Transition-index table with `indmax` entries.
    pub transitions: Vec<XseclBinTransition>,
    /// Atomic cross sections as `(energy, final_state)`, matching FEFF blocks
    /// of `atomxsec(1:kfinmax, ie)`.
    pub atom_cross_sections: Array2<Complex64>,
}

impl XseclBinData {
    /// Number of energy blocks represented in the PAD payload, `nex`.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.atom_cross_sections.len_of(Axis(0))
    }

    /// Number of final-state cross-section slots, `kfinmax`.
    #[must_use]
    pub fn final_state_count(&self) -> usize {
        self.atom_cross_sections.len_of(Axis(1))
    }

    /// Number of transition-index rows, `indmax`.
    #[must_use]
    pub fn transition_index_count(&self) -> usize {
        self.transitions.len()
    }
}

/// Render FEFF `xsecl.bin` text.
pub fn xsecl_bin_string(data: &XseclBinData) -> Result<String> {
    validate_xsecl_bin(data)?;

    let mut out = String::new();
    write_i5_line(
        &mut out,
        &[
            i64_from_usize(data.final_state_count(), "kfinmax")?,
            i64_from_usize(data.transition_index_count(), "indmax")?,
            i64::from(data.initial_state_j),
        ],
    )?;
    for transition in &data.transitions {
        write_i5_line(
            &mut out,
            &[
                i64::from(transition.final_state_kappa),
                i64::from(transition.decomposition_channel),
                i64::from(transition.total_angular_momentum_channel),
                i64::from(transition.orbital_angular_momentum),
            ],
        )?;
    }
    for energy in 0..data.energy_count() {
        let values = data
            .atom_cross_sections
            .row(energy)
            .iter()
            .copied()
            .collect::<Vec<_>>();
        out.push_str(&encode_complex(&values, data.pad_width)?);
    }
    Ok(out)
}

/// Parse FEFF `xsecl.bin` text.
pub fn parse_xsecl_bin(text: &str, pad_width: usize, energy_count: usize) -> Result<XseclBinData> {
    let mut lines = text.lines().enumerate();
    let (header_line, header) = next_nonempty(&mut lines, "header")?;
    let header = parse_i64_row(header_line, header, 3)?;
    let final_state_count = usize_from_i64(header[0], "kfinmax")?;
    let transition_index_count = usize_from_i64(header[1], "indmax")?;
    let initial_state_j = i32_from_i64(header[2], "jinit")?;

    let mut transitions = Vec::with_capacity(transition_index_count);
    for _ in 0..transition_index_count {
        let (line, row) = next_nonempty(&mut lines, "transition")?;
        let values = parse_i64_row(line, row, 4)?;
        transitions.push(XseclBinTransition {
            final_state_kappa: i32_from_i64(values[0], "kiind")?,
            decomposition_channel: i32_from_i64(values[1], "lgind")?,
            total_angular_momentum_channel: i32_from_i64(values[2], "ljind")?,
            orbital_angular_momentum: i32_from_i64(values[3], "lind")?,
        });
    }

    let payload = lines.map(|(_, line)| line).collect::<Vec<_>>().join("\n");
    let payload = if payload.is_empty() {
        String::new()
    } else {
        format!("{payload}\n")
    };
    let expected = checked_product("atomxsec", energy_count, final_state_count)?;
    let values = decode_complex(&payload, pad_width, expected)?;
    if values.len() != expected {
        return Err(IoError::XseclBinShape {
            field: "atomxsec",
            actual: vec![values.len()],
            expected: vec![expected],
        });
    }
    let atom_cross_sections = Array2::from_shape_vec((energy_count, final_state_count), values)
        .map_err(|_| IoError::XseclBinShape {
            field: "atomxsec",
            actual: vec![energy_count, final_state_count],
            expected: vec![energy_count, final_state_count],
        })?;

    let data = XseclBinData {
        pad_width,
        initial_state_j,
        transitions,
        atom_cross_sections,
    };
    validate_xsecl_bin(&data)?;
    Ok(data)
}

/// Write FEFF `xsecl.bin` text to a file.
pub fn write_xsecl_bin(path: impl AsRef<Path>, data: &XseclBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, xsecl_bin_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `xsecl.bin` text from a file.
pub fn read_xsecl_bin(
    path: impl AsRef<Path>,
    pad_width: usize,
    energy_count: usize,
) -> Result<XseclBinData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_xsecl_bin(&text, pad_width, energy_count)
}

fn validate_xsecl_bin(data: &XseclBinData) -> Result<()> {
    let final_state_count = data.final_state_count();
    if data.energy_count() == 0 {
        return Err(invalid_xsecl_bin("nex", "at least one energy is required"));
    }
    if final_state_count == 0 {
        return Err(invalid_xsecl_bin(
            "kfinmax",
            "at least one final state is required",
        ));
    }
    if data.transition_index_count() > final_state_count {
        return Err(invalid_xsecl_bin(
            "indmax",
            format!(
                "transition index count {} exceeds final-state count {}",
                data.transition_index_count(),
                final_state_count
            ),
        ));
    }

    check_i5(i64_from_usize(final_state_count, "kfinmax")?, "kfinmax")?;
    check_i5(
        i64_from_usize(data.transition_index_count(), "indmax")?,
        "indmax",
    )?;
    check_i5(i64::from(data.initial_state_j), "jinit")?;
    for transition in &data.transitions {
        check_i5(i64::from(transition.final_state_kappa), "kiind")?;
        check_i5(i64::from(transition.decomposition_channel), "lgind")?;
        check_i5(
            i64::from(transition.total_angular_momentum_channel),
            "ljind",
        )?;
        check_i5(i64::from(transition.orbital_angular_momentum), "lind")?;
    }

    for value in data.atom_cross_sections.iter() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(invalid_xsecl_bin("atomxsec", "all values must be finite"));
        }
    }
    Ok(())
}

fn next_nonempty<'a>(
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
    field: &'static str,
) -> Result<(usize, &'a str)> {
    for (line, text) in lines {
        if !text.trim().is_empty() {
            return Ok((line + 1, text));
        }
    }
    Err(IoError::XseclBinMissing { field })
}

fn parse_i64_row(line: usize, text: &str, expected: usize) -> Result<Vec<i64>> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != expected {
        return Err(IoError::XseclBinRowWidth {
            line,
            actual: tokens.len(),
            expected,
        });
    }
    tokens
        .iter()
        .map(|token| parse_i64_token(line, token))
        .collect()
}

fn parse_i64_token(line: usize, token: &str) -> Result<i64> {
    token.parse::<i64>().map_err(|_| IoError::XseclBinParse {
        field: "integer",
        line,
        token: token.to_string(),
    })
}

fn write_i5_line(out: &mut String, values: &[i64]) -> Result<()> {
    for value in values {
        write!(out, "{value:>5}")?;
    }
    out.push('\n');
    Ok(())
}

fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_xsecl_bin(field, "array element count overflowed"))
}

fn i64_from_usize(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| invalid_xsecl_bin(field, format!("value {value} does not fit in i64")))
}

fn usize_from_i64(value: i64, field: &'static str) -> Result<usize> {
    if value < 0 {
        return Err(invalid_xsecl_bin(
            field,
            format!("value {value} must be non-negative"),
        ));
    }
    usize::try_from(value)
        .map_err(|_| invalid_xsecl_bin(field, format!("value {value} does not fit in usize")))
}

fn i32_from_i64(value: i64, field: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| invalid_xsecl_bin(field, format!("value {value} does not fit in i32")))
}

fn check_i5(value: i64, field: &'static str) -> Result<()> {
    if value.to_string().len() > 5 {
        Err(invalid_xsecl_bin(
            field,
            format!("value {value} does not fit FEFF i5 output"),
        ))
    } else {
        Ok(())
    }
}

fn invalid_xsecl_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidXseclBin {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_header_and_transition_rows_like_feff() -> Result<()> {
        let text = xsecl_bin_string(&sample_xsecl_bin())?;
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("    4    3    1"));
        assert_eq!(lines.next(), Some("   -1    0    0    0"));
        assert_eq!(lines.next(), Some("    2    1    1    1"));
        assert_eq!(lines.next(), Some("   -2    2    2    2"));
        assert!(matches!(lines.next(), Some(line) if line.starts_with('$')));
        Ok(())
    }

    #[test]
    fn roundtrips_xsecl_bin_text_with_pad_tolerance() -> Result<()> {
        let data = sample_xsecl_bin();
        let parsed = parse_xsecl_bin(
            &xsecl_bin_string(&data)?,
            data.pad_width,
            data.energy_count(),
        )?;
        assert_eq!(parsed.pad_width, data.pad_width);
        assert_eq!(parsed.initial_state_j, data.initial_state_j);
        assert_eq!(parsed.transitions, data.transitions);
        assert_eq!(
            parsed.atom_cross_sections.dim(),
            data.atom_cross_sections.dim()
        );
        for (actual, expected) in parsed
            .atom_cross_sections
            .iter()
            .zip(data.atom_cross_sections.iter())
        {
            assert!(
                (actual.re - expected.re).abs() <= expected.re.abs().max(1.0) * 1.0e-6,
                "{actual} != {expected}"
            );
            assert!(
                (actual.im - expected.im).abs() <= expected.im.abs().max(1.0) * 1.0e-6,
                "{actual} != {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn preserves_per_energy_final_state_order() -> Result<()> {
        let data = sample_xsecl_bin();
        let parsed = parse_xsecl_bin(
            &xsecl_bin_string(&data)?,
            data.pad_width,
            data.energy_count(),
        )?;
        assert_close(
            parsed.atom_cross_sections[(0, 0)],
            data.atom_cross_sections[(0, 0)],
        );
        assert_close(
            parsed.atom_cross_sections[(0, 1)],
            data.atom_cross_sections[(0, 1)],
        );
        assert_close(
            parsed.atom_cross_sections[(1, 0)],
            data.atom_cross_sections[(1, 0)],
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_shapes_and_tokens() {
        let mut bad = sample_xsecl_bin();
        bad.transitions.push(XseclBinTransition {
            final_state_kappa: 3,
            decomposition_channel: 3,
            total_angular_momentum_channel: 3,
            orbital_angular_momentum: 3,
        });
        bad.transitions.push(XseclBinTransition {
            final_state_kappa: -4,
            decomposition_channel: 4,
            total_angular_momentum_channel: 4,
            orbital_angular_momentum: 4,
        });
        assert!(matches!(
            xsecl_bin_string(&bad),
            Err(IoError::InvalidXseclBin {
                field: "indmax",
                ..
            })
        ));

        assert!(matches!(
            parse_xsecl_bin("    4 nope    1\n", 8, 2),
            Err(IoError::XseclBinParse { line: 1, .. })
        ));
    }

    fn sample_xsecl_bin() -> XseclBinData {
        XseclBinData {
            pad_width: 8,
            initial_state_j: 1,
            transitions: vec![
                XseclBinTransition {
                    final_state_kappa: -1,
                    decomposition_channel: 0,
                    total_angular_momentum_channel: 0,
                    orbital_angular_momentum: 0,
                },
                XseclBinTransition {
                    final_state_kappa: 2,
                    decomposition_channel: 1,
                    total_angular_momentum_channel: 1,
                    orbital_angular_momentum: 1,
                },
                XseclBinTransition {
                    final_state_kappa: -2,
                    decomposition_channel: 2,
                    total_angular_momentum_channel: 2,
                    orbital_angular_momentum: 2,
                },
            ],
            atom_cross_sections: Array2::from_shape_fn((2, 4), |(energy, final_state)| {
                Complex64::new(
                    0.1 * (energy + 1) as f64 + 0.01 * final_state as f64,
                    -0.05 * (energy + 1) as f64 - 0.005 * final_state as f64,
                )
            }),
        }
    }

    fn assert_close(actual: Complex64, expected: Complex64) {
        assert!((actual.re - expected.re).abs() <= 1.0e-6);
        assert!((actual.im - expected.im).abs() <= 1.0e-6);
    }
}
