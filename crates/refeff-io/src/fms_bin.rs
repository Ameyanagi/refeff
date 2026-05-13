//! FEFF `fms.bin` text/PAD FMS result codec.
//!
//! `MKGTR/getgtr.f90` writes this printable handoff file for FF2X. The current
//! FEFF10 path writes a six-integer header with an explicit spectrum count;
//! older JAS/NRIXS paths write five integers and one spectrum. Some FEFF10
//! builds also leave the header count as zero while still writing one PAD
//! spectrum. All forms are parsed here while the writer emits the modern
//! six-integer shape.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array2;
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::repeated_ints;
use crate::pad::{decode_complex, encode_complex};

/// FEFF10 default PAD width for `fms.bin`.
pub const FMS_BIN_DEFAULT_PAD_WIDTH: usize = 8;

/// FEFF `fms.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsBinData {
    /// FMS cluster radius in Angstrom from the `FMS rfms=` title line.
    pub cluster_radius_angstrom: f64,
    /// Number of energy points, `ne`.
    pub energy_count: usize,
    /// Number of main energy points, `ne1`.
    pub main_energy_count: usize,
    /// Number of auxiliary energy points, `ne3`.
    pub auxiliary_energy_count: usize,
    /// Highest unique potential index, `nph`.
    pub highest_potential_index: usize,
    /// PAD field width, `npadx`.
    pub pad_width: usize,
    /// FMS trace spectra as `(spectrum, energy)`, matching FEFF's `ip, ie`
    /// write order.
    pub spectra: Array2<Complex64>,
}

impl FmsBinData {
    /// Number of spectra in the `fms.bin` PAD payload, `nip`.
    #[must_use]
    pub fn spectrum_count(&self) -> usize {
        self.spectra.nrows()
    }
}

/// Render FEFF `fms.bin` text.
pub fn fms_bin_string(data: &FmsBinData) -> Result<String> {
    validate_fms_bin(data)?;

    let mut out = String::new();
    writeln!(out, "FMS rfms={:>7.4}", data.cluster_radius_angstrom)?;
    writeln!(
        out,
        "{}",
        repeated_ints(
            [
                i64_from_usize(data.energy_count, "ne")?,
                i64_from_usize(data.main_energy_count, "ne1")?,
                i64_from_usize(data.auxiliary_energy_count, "ne3")?,
                i64_from_usize(data.highest_potential_index, "nph")?,
                i64_from_usize(data.pad_width, "npadx")?,
                i64_from_usize(data.spectrum_count(), "nip")?,
            ],
            7,
        )
    )?;
    let payload = data.spectra.iter().copied().collect::<Vec<_>>();
    out.push_str(&encode_complex(&payload, data.pad_width)?);
    Ok(out)
}

/// Parse FEFF `fms.bin` text.
pub fn parse_fms_bin(text: &str) -> Result<FmsBinData> {
    let mut lines = text.lines();
    let title = next_nonempty(&mut lines, "title")?;
    let cluster_radius_angstrom = parse_cluster_radius(title)?;
    let counts = parse_counts(next_nonempty(&mut lines, "counts")?)?;
    let energy_count = counts[0];
    let payload = lines.collect::<Vec<_>>().join("\n");
    let payload = if payload.is_empty() {
        String::new()
    } else {
        format!("{payload}\n")
    };
    let payload_count = count_complex_pad_values(&payload, counts[4])?;
    let spectrum_count = match counts.as_slice() {
        [_, _, _, _, _] => 1,
        [_, _, _, _, _, 0] if payload_count > 0 => {
            if energy_count == 0 {
                return Err(invalid_fms_bin("ne", "at least one energy is required"));
            }
            if payload_count % energy_count != 0 {
                return Err(IoError::FmsBinShape {
                    field: "gtr",
                    actual: vec![payload_count],
                    expected: vec![energy_count],
                });
            }
            payload_count / energy_count
        }
        [_, _, _, _, _, declared] => *declared,
        _ => {
            return Err(invalid_fms_bin(
                "counts",
                format!("expected 5 or 6 integer fields, got {}", counts.len()),
            ));
        }
    };
    let expected = checked_product("gtr", energy_count, spectrum_count)?;
    let values = decode_complex(&payload, counts[4], expected)?;
    if values.len() != expected {
        return Err(IoError::FmsBinShape {
            field: "gtr",
            actual: vec![values.len()],
            expected: vec![expected],
        });
    }
    let spectra = Array2::from_shape_vec((spectrum_count, energy_count), values).map_err(|_| {
        IoError::FmsBinShape {
            field: "gtr",
            actual: vec![spectrum_count, energy_count],
            expected: vec![spectrum_count, energy_count],
        }
    })?;

    let data = FmsBinData {
        cluster_radius_angstrom,
        energy_count,
        main_energy_count: counts[1],
        auxiliary_energy_count: counts[2],
        highest_potential_index: counts[3],
        pad_width: counts[4],
        spectra,
    };
    validate_fms_bin(&data)?;
    Ok(data)
}

/// Write FEFF `fms.bin` text to a file.
pub fn write_fms_bin(path: impl AsRef<Path>, data: &FmsBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, fms_bin_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `fms.bin` text from a file.
pub fn read_fms_bin(path: impl AsRef<Path>) -> Result<FmsBinData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_fms_bin(&text)
}

fn validate_fms_bin(data: &FmsBinData) -> Result<()> {
    if !data.cluster_radius_angstrom.is_finite() {
        return Err(invalid_fms_bin("rfms", "value must be finite"));
    }
    let radius = format!("{:>7.4}", data.cluster_radius_angstrom);
    if radius.len() > 7 {
        return Err(invalid_fms_bin(
            "rfms",
            format!("formatted value {radius:?} exceeds width 7"),
        ));
    }
    if data.energy_count == 0 {
        return Err(invalid_fms_bin("ne", "at least one energy is required"));
    }
    if data.main_energy_count == 0 || data.main_energy_count > data.energy_count {
        return Err(invalid_fms_bin(
            "ne1",
            format!(
                "main energy count {} must be in 1..={}",
                data.main_energy_count, data.energy_count
            ),
        ));
    }
    if data.auxiliary_energy_count > data.energy_count {
        return Err(invalid_fms_bin(
            "ne3",
            format!(
                "auxiliary energy count {} exceeds energy count {}",
                data.auxiliary_energy_count, data.energy_count
            ),
        ));
    }
    ensure_i_width("ne", data.energy_count, 7)?;
    ensure_i_width("ne1", data.main_energy_count, 7)?;
    ensure_i_width("ne3", data.auxiliary_energy_count, 7)?;
    ensure_i_width("nph", data.highest_potential_index, 7)?;
    ensure_i_width("npadx", data.pad_width, 7)?;
    ensure_i_width("nip", data.spectrum_count(), 7)?;

    if data.spectra.ncols() != data.energy_count {
        return Err(IoError::FmsBinShape {
            field: "gtr",
            actual: vec![data.spectra.nrows(), data.spectra.ncols()],
            expected: vec![data.spectrum_count(), data.energy_count],
        });
    }
    for value in data.spectra.iter() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(invalid_fms_bin("gtr", "all values must be finite"));
        }
    }
    Ok(())
}

fn next_nonempty<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
    field: &'static str,
) -> Result<&'a str> {
    for line in lines {
        if !line.trim().is_empty() {
            return Ok(line);
        }
    }
    Err(IoError::FmsBinMissing { field })
}

fn parse_cluster_radius(line: &str) -> Result<f64> {
    let Some((_, value)) = line.split_once("rfms=") else {
        return Err(invalid_fms_bin(
            "title",
            format!("expected FMS rfms title, got {line:?}"),
        ));
    };
    let token = value
        .split_whitespace()
        .next()
        .ok_or(IoError::FmsBinMissing { field: "rfms" })?;
    token.parse::<f64>().map_err(|_| IoError::FmsBinParse {
        field: "rfms",
        token: token.to_string(),
    })
}

fn parse_counts(line: &str) -> Result<Vec<usize>> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 5 && tokens.len() != 6 {
        return Err(invalid_fms_bin(
            "counts",
            format!("expected 5 or 6 integer fields, got {}", tokens.len()),
        ));
    }
    tokens
        .iter()
        .map(|token| {
            token.parse::<usize>().map_err(|_| IoError::FmsBinParse {
                field: "counts",
                token: (*token).to_string(),
            })
        })
        .collect()
}

fn count_complex_pad_values(text: &str, npack: usize) -> Result<usize> {
    if npack <= 2 {
        return Err(IoError::InvalidPadWidth(npack));
    }

    let unit_len = 2 * npack;
    let mut count = 0_usize;
    for line in text.lines() {
        let Some(found) = line.chars().next() else {
            continue;
        };
        if found != '$' {
            return Err(IoError::PadMarker {
                expected: '$',
                found,
            });
        }
        let payload = &line[found.len_utf8()..];
        if payload.len() % unit_len != 0 {
            return Err(IoError::PadPayload {
                payload_len: payload.len(),
                unit_len,
            });
        }
        count += payload.len() / unit_len;
    }
    Ok(count)
}

fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_fms_bin(field, "array element count overflowed"))
}

fn i64_from_usize(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| invalid_fms_bin(field, format!("value {value} does not fit in i64")))
}

fn ensure_i_width(field: &'static str, value: usize, width: usize) -> Result<()> {
    if value.to_string().len() > width {
        Err(invalid_fms_bin(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

fn invalid_fms_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidFmsBin {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_fms_header_like_feff() -> Result<()> {
        let text = fms_bin_string(&sample_fms_bin())?;
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("FMS rfms= 5.5000"));
        assert_eq!(
            lines.next(),
            Some("       3       2       1       1       8       2")
        );
        assert!(matches!(lines.next(), Some(line) if line.starts_with('$')));
        Ok(())
    }

    #[test]
    fn roundtrips_fms_bin_text_with_pad_tolerance() -> Result<()> {
        let data = sample_fms_bin();
        let parsed = parse_fms_bin(&fms_bin_string(&data)?)?;
        assert_eq!(parsed.cluster_radius_angstrom, data.cluster_radius_angstrom);
        assert_eq!(parsed.energy_count, data.energy_count);
        assert_eq!(parsed.main_energy_count, data.main_energy_count);
        assert_eq!(parsed.auxiliary_energy_count, data.auxiliary_energy_count);
        assert_eq!(parsed.highest_potential_index, data.highest_potential_index);
        assert_eq!(parsed.pad_width, data.pad_width);
        assert_eq!(parsed.spectra.dim(), data.spectra.dim());
        for (actual, expected) in parsed.spectra.iter().zip(data.spectra.iter()) {
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
    fn parses_legacy_five_integer_header_as_single_spectrum() -> Result<()> {
        let mut data = sample_fms_bin();
        data.spectra = Array2::from_shape_fn((1, 3), |(_, energy)| {
            Complex64::new(0.1 * energy as f64, -0.01 * energy as f64)
        });
        let payload = encode_complex(
            &data.spectra.iter().copied().collect::<Vec<_>>(),
            data.pad_width,
        )?;
        let text = format!("FMS rfms= 5.5000\n   3   2   1   1   8\n{payload}");
        let parsed = parse_fms_bin(&text)?;
        assert_eq!(parsed.spectrum_count(), 1);
        assert_eq!(parsed.energy_count, 3);
        assert_eq!(parsed.spectra.dim(), (1, 3));
        Ok(())
    }

    #[test]
    fn infers_zero_declared_spectrum_count_from_feff_payload() -> Result<()> {
        let spectra = Array2::from_shape_fn((1, 3), |(_, energy)| {
            Complex64::new(0.1 * energy as f64, -0.01 * energy as f64)
        });
        let payload = encode_complex(
            &spectra.iter().copied().collect::<Vec<_>>(),
            FMS_BIN_DEFAULT_PAD_WIDTH,
        )?;
        let text = format!("FMS rfms=-1.0000\n   3   2   0   1   8   0\n{payload}");
        let parsed = parse_fms_bin(&text)?;
        assert_eq!(parsed.spectrum_count(), 1);
        assert_eq!(parsed.spectra.dim(), (1, 3));
        Ok(())
    }

    #[test]
    fn parses_empty_zero_spectrum_fms_bin() -> Result<()> {
        let parsed = parse_fms_bin("FMS rfms=-1.0000\n   3   2   0   1   8   0\n")?;
        assert_eq!(parsed.spectrum_count(), 0);
        assert_eq!(parsed.spectra.dim(), (0, 3));
        Ok(())
    }

    #[test]
    fn rejects_bad_shapes_and_tokens() {
        let mut bad = sample_fms_bin();
        bad.energy_count = 4;
        assert!(matches!(
            fms_bin_string(&bad),
            Err(IoError::FmsBinShape {
                field: "gtr",
                actual,
                expected,
            }) if actual == vec![2, 3] && expected == vec![2, 4]
        ));

        assert!(matches!(
            parse_fms_bin("FMS rfms= nope\n       3       2       1       1       8       1\n"),
            Err(IoError::FmsBinParse { field: "rfms", .. })
        ));
    }

    fn sample_fms_bin() -> FmsBinData {
        FmsBinData {
            cluster_radius_angstrom: 5.5,
            energy_count: 3,
            main_energy_count: 2,
            auxiliary_energy_count: 1,
            highest_potential_index: 1,
            pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
            spectra: Array2::from_shape_fn((2, 3), |(spectrum, energy)| {
                Complex64::new(
                    0.25 * (energy + 1) as f64 + spectrum as f64,
                    -0.05 * (energy + 1) as f64 - spectrum as f64,
                )
            }),
        }
    }
}
