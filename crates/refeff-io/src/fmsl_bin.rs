//! FEFF `fmsl.bin` text/PAD FMS decomposition codec.
//!
//! The NRIXS/JAS path in `MKGTR/getgtrjas.f90` writes `fmsl.bin` when
//! `ldecmx >= 0`. Unlike `fms.bin`, this file has no header: FF2X already knows
//! the PAD width, energy count, and number of decomposition channels from
//! `fms.bin` and module input files. The payload is one complex PAD block per
//! energy, each containing the square `gtrl(lg2, lg1, ie)` channel matrix in
//! Fortran column-major order.

use std::path::Path;

use ndarray::{Array3, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::pad::{decode_complex, encode_complex};

/// FEFF `fmsl.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct FmslBinData {
    /// PAD field width, `npadx`.
    pub pad_width: usize,
    /// Highest decomposition angular-momentum channel, `ldecmx`.
    pub max_decomposition_channel: usize,
    /// Decomposed FMS traces as `(energy, lg2, lg1)`, matching FEFF `gtrl`.
    pub traces: Array3<Complex64>,
}

impl FmslBinData {
    /// Number of energy blocks in the file, `ne`.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.traces.len_of(Axis(0))
    }

    /// Number of decomposition channels, `ldecmx + 1`.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.max_decomposition_channel + 1
    }
}

/// Render FEFF `fmsl.bin` text.
pub fn fmsl_bin_string(data: &FmslBinData) -> Result<String> {
    validate_fmsl_bin(data)?;

    let mut out = String::new();
    for energy in 0..data.energy_count() {
        out.push_str(&encode_complex(
            &flatten_channel_matrix(data, energy),
            data.pad_width,
        )?);
    }
    Ok(out)
}

/// Parse FEFF `fmsl.bin` text.
pub fn parse_fmsl_bin(
    text: &str,
    pad_width: usize,
    energy_count: usize,
    max_decomposition_channel: usize,
) -> Result<FmslBinData> {
    let channel_count = max_decomposition_channel
        .checked_add(1)
        .ok_or_else(|| invalid_fmsl_bin("ldecmx", "channel count overflowed"))?;
    let values_per_energy = checked_square(channel_count)?;
    let expected = energy_count
        .checked_mul(values_per_energy)
        .ok_or_else(|| invalid_fmsl_bin("gtrl", "array element count overflowed"))?;
    let values = decode_complex(text, pad_width, expected)?;
    if values.len() != expected {
        return Err(IoError::FmslBinShape {
            field: "gtrl",
            actual: vec![values.len()],
            expected: vec![expected],
        });
    }

    let mut traces = Array3::from_elem(
        (energy_count, channel_count, channel_count),
        Complex64::new(0.0, 0.0),
    );
    for (energy, chunk) in values.chunks(values_per_energy).enumerate() {
        for lg1 in 0..channel_count {
            for lg2 in 0..channel_count {
                traces[(energy, lg2, lg1)] = chunk[lg1 * channel_count + lg2];
            }
        }
    }

    let data = FmslBinData {
        pad_width,
        max_decomposition_channel,
        traces,
    };
    validate_fmsl_bin(&data)?;
    Ok(data)
}

/// Write FEFF `fmsl.bin` text to a file.
pub fn write_fmsl_bin(path: impl AsRef<Path>, data: &FmslBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, fmsl_bin_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `fmsl.bin` text from a file.
pub fn read_fmsl_bin(
    path: impl AsRef<Path>,
    pad_width: usize,
    energy_count: usize,
    max_decomposition_channel: usize,
) -> Result<FmslBinData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_fmsl_bin(&text, pad_width, energy_count, max_decomposition_channel)
}

fn validate_fmsl_bin(data: &FmslBinData) -> Result<()> {
    let channel_count = data.channel_count();
    let expected = vec![data.energy_count(), channel_count, channel_count];
    let actual = data.traces.shape().to_vec();
    if actual != expected {
        return Err(IoError::FmslBinShape {
            field: "gtrl",
            actual,
            expected,
        });
    }
    if data.energy_count() == 0 {
        return Err(invalid_fmsl_bin("ne", "at least one energy is required"));
    }
    for value in data.traces.iter() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(invalid_fmsl_bin("gtrl", "all values must be finite"));
        }
    }
    Ok(())
}

fn flatten_channel_matrix(data: &FmslBinData, energy: usize) -> Vec<Complex64> {
    let channel_count = data.channel_count();
    let mut values = Vec::with_capacity(channel_count * channel_count);
    for lg1 in 0..channel_count {
        for lg2 in 0..channel_count {
            values.push(data.traces[(energy, lg2, lg1)]);
        }
    }
    values
}

fn checked_square(value: usize) -> Result<usize> {
    value
        .checked_mul(value)
        .ok_or_else(|| invalid_fmsl_bin("ldecmx", "channel matrix size overflowed"))
}

fn invalid_fmsl_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidFmslBin {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_complex_pad_payload() -> Result<()> {
        let data = sample_fmsl_bin();
        let text = fmsl_bin_string(&data)?;
        let lines = text.lines().collect::<Vec<_>>();
        assert!(!lines.is_empty());
        assert!(lines.iter().all(|line| line.starts_with('$')));
        assert_eq!(
            decode_complex(&text, data.pad_width, data.traces.len())?.len(),
            data.traces.len()
        );
        Ok(())
    }

    #[test]
    fn roundtrips_fmsl_bin_text_with_pad_tolerance() -> Result<()> {
        let data = sample_fmsl_bin();
        let parsed = parse_fmsl_bin(
            &fmsl_bin_string(&data)?,
            data.pad_width,
            data.energy_count(),
            data.max_decomposition_channel,
        )?;
        assert_eq!(parsed.pad_width, data.pad_width);
        assert_eq!(
            parsed.max_decomposition_channel,
            data.max_decomposition_channel
        );
        assert_eq!(parsed.traces.dim(), data.traces.dim());
        for (actual, expected) in parsed.traces.iter().zip(data.traces.iter()) {
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
    fn preserves_feff_column_major_channel_order() -> Result<()> {
        let data = sample_fmsl_bin();
        let text = fmsl_bin_string(&data)?;
        let values = decode_complex(&text, data.pad_width, data.traces.len())?;
        assert_close(values[0], data.traces[(0, 0, 0)]);
        assert_close(values[1], data.traces[(0, 1, 0)]);
        assert_close(values[2], data.traces[(0, 2, 0)]);
        assert_close(values[3], data.traces[(0, 0, 1)]);
        Ok(())
    }

    #[test]
    fn rejects_bad_shapes() {
        let mut bad = sample_fmsl_bin();
        bad.max_decomposition_channel = 1;
        assert!(matches!(
            fmsl_bin_string(&bad),
            Err(IoError::FmslBinShape {
                field: "gtrl",
                actual,
                expected,
            }) if actual == vec![2, 3, 3] && expected == vec![2, 2, 2]
        ));
    }

    fn sample_fmsl_bin() -> FmslBinData {
        FmslBinData {
            pad_width: 8,
            max_decomposition_channel: 2,
            traces: Array3::from_shape_fn((2, 3, 3), |(energy, lg2, lg1)| {
                Complex64::new(
                    energy as f64 + 0.1 * lg2 as f64 + 0.01 * lg1 as f64,
                    -(energy as f64) - 0.2 * lg2 as f64 - 0.02 * lg1 as f64,
                )
            }),
        }
    }

    fn assert_close(actual: Complex64, expected: Complex64) {
        assert!((actual.re - expected.re).abs() <= 1.0e-6);
        assert!((actual.im - expected.im).abs() <= 1.0e-6);
    }
}
