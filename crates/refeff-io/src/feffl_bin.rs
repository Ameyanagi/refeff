//! FEFF `feffl.bin` text/PAD path-decomposition companion codec.
//!
//! `GENFMT/genfmtjas.f90` writes this file alongside `feff.bin` when
//! `ldecmx >= 0`. The file has no header: it is a stream of real PAD blocks
//! keyed by the path order in `feff.bin`, the energy count and PAD width from
//! `feff.bin`, and the decomposition limit from `genfmt.inp`. For each path and
//! each `(lg2, lg1)` channel pair, FEFF writes an amplitude block followed by a
//! phase block.

use std::path::Path;

use ndarray::{Array4, Axis};
use refeff_core::{GenfmtDecomposedChiAmplitudePhase, GenfmtJasDriverOutput, GenfmtJasPathOutputs};

use crate::error::{IoError, Result};
use crate::pad::{decode_reals, encode_reals};

/// FEFF `feffl.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct FefflBinData {
    /// PAD field width, `mpadx`, supplied by `feff.bin`.
    pub pad_width: usize,
    /// Highest decomposition angular-momentum channel, `ldecmx`.
    pub max_decomposition_channel: usize,
    /// Decomposed path amplitudes as `(path, lg2, lg1, energy)`, matching
    /// FEFF `lgachi(energy, lg2, lg1, path)` with Rust path-first storage.
    pub amplitudes: Array4<f64>,
    /// Decomposed path phases as `(path, lg2, lg1, energy)`, matching FEFF
    /// `lgphchi(energy, lg2, lg1, path)` with Rust path-first storage.
    pub phases: Array4<f64>,
}

impl FefflBinData {
    /// Number of path decomposition records.
    #[must_use]
    pub fn path_count(&self) -> usize {
        self.amplitudes.len_of(Axis(0))
    }

    /// Number of decomposition channels, `ldecmx + 1`.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.max_decomposition_channel.saturating_add(1)
    }

    /// Number of energy points per PAD block, `ne`.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.amplitudes.len_of(Axis(3))
    }

    /// Build FEFF `feffl.bin` data from retained GENFMTJAS decomposition output.
    ///
    /// Path records are copied in caller-supplied order, matching the retained
    /// path order in the companion `feff.bin` file.
    pub fn from_genfmt_output(
        pad_width: usize,
        max_decomposition_channel: usize,
        decomposed_paths: &[GenfmtDecomposedChiAmplitudePhase],
    ) -> Result<Self> {
        let channel_count = checked_channel_count(max_decomposition_channel)?;
        let path_count = decomposed_paths.len();
        if path_count == 0 {
            return Err(invalid_feffl_bin(
                "path_count",
                "at least one decomposed path is required",
            ));
        }

        let energy_count = decomposed_paths[0].amplitudes.shape()[2];
        let expected = vec![channel_count, channel_count, energy_count];
        let mut amplitudes =
            Array4::zeros((path_count, channel_count, channel_count, energy_count));
        let mut phases = Array4::zeros((path_count, channel_count, channel_count, energy_count));

        for (path, decomposed) in decomposed_paths.iter().enumerate() {
            let amplitude_shape = decomposed.amplitudes.shape().to_vec();
            if amplitude_shape != expected {
                return Err(IoError::FefflBinShape {
                    field: "lgachi",
                    actual: amplitude_shape,
                    expected: expected.clone(),
                });
            }
            let phase_shape = decomposed.phases.shape().to_vec();
            if phase_shape != expected {
                return Err(IoError::FefflBinShape {
                    field: "lgphchi",
                    actual: phase_shape,
                    expected: expected.clone(),
                });
            }
            for lg2 in 0..channel_count {
                for lg1 in 0..channel_count {
                    for energy in 0..energy_count {
                        amplitudes[(path, lg2, lg1, energy)] =
                            decomposed.amplitudes[(lg2, lg1, energy)];
                        phases[(path, lg2, lg1, energy)] = decomposed.phases[(lg2, lg1, energy)];
                    }
                }
            }
        }

        let data = Self {
            pad_width,
            max_decomposition_channel,
            amplitudes,
            phases,
        };
        validate_feffl_bin(&data)?;
        Ok(data)
    }

    /// Build optional FEFF `feffl.bin` data from retained GENFMTJAS outputs.
    ///
    /// Returns `None` when the GENFMTJAS run did not request decomposition.
    pub fn from_genfmt_jas_outputs(
        pad_width: usize,
        max_decomposition_channel: usize,
        outputs: &GenfmtJasPathOutputs,
    ) -> Result<Option<Self>> {
        match outputs.decomposed_paths.as_deref() {
            Some(decomposed_paths) => {
                Self::from_genfmt_output(pad_width, max_decomposition_channel, decomposed_paths)
                    .map(Some)
            }
            None => Ok(None),
        }
    }

    /// Build optional FEFF `feffl.bin` data from GENFMTJAS driver output.
    ///
    /// Returns `None` when the GENFMTJAS run did not request decomposition.
    pub fn from_genfmt_jas_driver_output(
        max_decomposition_channel: usize,
        output: &GenfmtJasDriverOutput,
    ) -> Result<Option<Self>> {
        Self::from_genfmt_jas_outputs(
            output.header.pad_width,
            max_decomposition_channel,
            &output.path_sequence.outputs,
        )
    }
}

/// Render FEFF `feffl.bin` text.
pub fn feffl_bin_string(data: &FefflBinData) -> Result<String> {
    validate_feffl_bin(data)?;

    let channel_count = checked_channel_count(data.max_decomposition_channel)?;
    let mut out = String::new();
    for path in 0..data.path_count() {
        for lg1 in 0..channel_count {
            for lg2 in 0..channel_count {
                out.push_str(&encode_reals(
                    &energy_values(&data.amplitudes, path, lg2, lg1),
                    data.pad_width,
                )?);
                out.push_str(&encode_reals(
                    &energy_values(&data.phases, path, lg2, lg1),
                    data.pad_width,
                )?);
            }
        }
    }
    Ok(out)
}

/// Parse FEFF `feffl.bin` text.
pub fn parse_feffl_bin(
    text: &str,
    pad_width: usize,
    path_count: usize,
    energy_count: usize,
    max_decomposition_channel: usize,
) -> Result<FefflBinData> {
    let channel_count = checked_channel_count(max_decomposition_channel)?;
    let channel_pairs = checked_square(channel_count)?;
    let block_count = checked_product("lgachi", path_count, channel_pairs)?
        .checked_mul(2)
        .ok_or_else(|| invalid_feffl_bin("lgachi", "PAD block count overflowed"))?;
    let expected = checked_product("lgachi", block_count, energy_count)?;
    let values = decode_reals(text, pad_width, expected)?;
    if values.len() != expected {
        return Err(IoError::FefflBinShape {
            field: "lgachi/lgphchi",
            actual: vec![values.len()],
            expected: vec![expected],
        });
    }

    let mut amplitudes = Array4::zeros((path_count, channel_count, channel_count, energy_count));
    let mut phases = Array4::zeros((path_count, channel_count, channel_count, energy_count));
    let mut offset = 0_usize;
    for path in 0..path_count {
        for lg1 in 0..channel_count {
            for lg2 in 0..channel_count {
                for energy in 0..energy_count {
                    amplitudes[(path, lg2, lg1, energy)] = values[offset];
                    offset += 1;
                }
                for energy in 0..energy_count {
                    phases[(path, lg2, lg1, energy)] = values[offset];
                    offset += 1;
                }
            }
        }
    }

    let data = FefflBinData {
        pad_width,
        max_decomposition_channel,
        amplitudes,
        phases,
    };
    validate_feffl_bin(&data)?;
    Ok(data)
}

/// Write FEFF `feffl.bin` text to a file.
pub fn write_feffl_bin(path: impl AsRef<Path>, data: &FefflBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, feffl_bin_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `feffl.bin` text from a file.
pub fn read_feffl_bin(
    path: impl AsRef<Path>,
    pad_width: usize,
    path_count: usize,
    energy_count: usize,
    max_decomposition_channel: usize,
) -> Result<FefflBinData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_feffl_bin(
        &text,
        pad_width,
        path_count,
        energy_count,
        max_decomposition_channel,
    )
}

fn validate_feffl_bin(data: &FefflBinData) -> Result<()> {
    let channel_count = checked_channel_count(data.max_decomposition_channel)?;
    let expected = vec![
        data.path_count(),
        channel_count,
        channel_count,
        data.energy_count(),
    ];
    let amplitude_shape = data.amplitudes.shape().to_vec();
    if amplitude_shape != expected {
        return Err(IoError::FefflBinShape {
            field: "lgachi",
            actual: amplitude_shape,
            expected,
        });
    }
    let expected = vec![
        data.path_count(),
        channel_count,
        channel_count,
        data.energy_count(),
    ];
    let phase_shape = data.phases.shape().to_vec();
    if phase_shape != expected {
        return Err(IoError::FefflBinShape {
            field: "lgphchi",
            actual: phase_shape,
            expected,
        });
    }
    if data.energy_count() == 0 {
        return Err(invalid_feffl_bin("ne", "at least one energy is required"));
    }
    for value in data.amplitudes.iter().chain(data.phases.iter()) {
        if !value.is_finite() {
            return Err(invalid_feffl_bin(
                "lgachi/lgphchi",
                "all values must be finite",
            ));
        }
    }
    Ok(())
}

fn energy_values(array: &Array4<f64>, path: usize, lg2: usize, lg1: usize) -> Vec<f64> {
    (0..array.len_of(Axis(3)))
        .map(|energy| array[(path, lg2, lg1, energy)])
        .collect()
}

fn checked_channel_count(max_decomposition_channel: usize) -> Result<usize> {
    max_decomposition_channel
        .checked_add(1)
        .ok_or_else(|| invalid_feffl_bin("ldecmx", "channel count overflowed"))
}

fn checked_square(value: usize) -> Result<usize> {
    checked_product("ldecmx", value, value)
}

fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_feffl_bin(field, "array element count overflowed"))
}

fn invalid_feffl_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidFefflBin {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{Array1, Array3};
    use num_complex::Complex64;
    use refeff_core::{
        GenfmtDecomposedChiAmplitudePhase, GenfmtFeffBinHeader, GenfmtJasDriverOutput,
        GenfmtJasPathOutputs, GenfmtJasPathSequence,
    };

    #[test]
    fn writes_real_pad_payload() -> Result<()> {
        let data = sample_feffl_bin();
        let text = feffl_bin_string(&data)?;
        let lines = text.lines().collect::<Vec<_>>();
        assert!(!lines.is_empty());
        assert!(lines.iter().all(|line| line.starts_with('!')));
        let channel_count = data.channel_count();
        let expected = data.path_count() * channel_count * channel_count * 2 * data.energy_count();
        assert_eq!(
            decode_reals(&text, data.pad_width, expected)?.len(),
            expected
        );
        Ok(())
    }

    #[test]
    fn roundtrips_feffl_bin_text_with_pad_tolerance() -> Result<()> {
        let data = sample_feffl_bin();
        let parsed = parse_feffl_bin(
            &feffl_bin_string(&data)?,
            data.pad_width,
            data.path_count(),
            data.energy_count(),
            data.max_decomposition_channel,
        )?;
        assert_eq!(parsed.pad_width, data.pad_width);
        assert_eq!(
            parsed.max_decomposition_channel,
            data.max_decomposition_channel
        );
        assert_eq!(parsed.amplitudes.dim(), data.amplitudes.dim());
        assert_eq!(parsed.phases.dim(), data.phases.dim());
        for (actual, expected) in parsed.amplitudes.iter().zip(data.amplitudes.iter()) {
            assert!(
                (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-6,
                "{actual} != {expected}"
            );
        }
        for (actual, expected) in parsed.phases.iter().zip(data.phases.iter()) {
            assert!(
                (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-6,
                "{actual} != {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn preserves_feff_path_channel_energy_order() -> Result<()> {
        let data = sample_feffl_bin();
        let text = feffl_bin_string(&data)?;
        let values = decode_reals(&text, data.pad_width, data.energy_count() * 2)?;
        assert_close(values[0], data.amplitudes[(0, 0, 0, 0)]);
        assert_close(values[1], data.amplitudes[(0, 0, 0, 1)]);
        assert_close(values[2], data.phases[(0, 0, 0, 0)]);
        assert_close(values[3], data.phases[(0, 0, 0, 1)]);
        Ok(())
    }

    #[test]
    fn builds_feffl_from_genfmt_decomposed_outputs() -> Result<()> {
        let first = sample_genfmt_decomposed_path(0.0);
        let second = sample_genfmt_decomposed_path(1.0);
        let data = FefflBinData::from_genfmt_output(8, 1, &[first.clone(), second.clone()])?;

        assert_eq!(data.pad_width, 8);
        assert_eq!(data.max_decomposition_channel, 1);
        assert_eq!(data.path_count(), 2);
        assert_eq!(data.channel_count(), 2);
        assert_eq!(data.energy_count(), 3);
        assert_close(data.amplitudes[(0, 0, 0, 0)], first.amplitudes[(0, 0, 0)]);
        assert_close(data.amplitudes[(1, 1, 0, 2)], second.amplitudes[(1, 0, 2)]);
        assert_close(data.phases[(0, 0, 1, 1)], first.phases[(0, 1, 1)]);
        assert_close(data.phases[(1, 1, 1, 2)], second.phases[(1, 1, 2)]);

        let rendered = feffl_bin_string(&data)?;
        let parsed = parse_feffl_bin(&rendered, 8, 2, 3, 1)?;
        assert_eq!(parsed.path_count(), 2);
        assert_close(
            parsed.amplitudes[(1, 1, 0, 2)],
            second.amplitudes[(1, 0, 2)],
        );
        Ok(())
    }

    #[test]
    fn builds_feffl_from_genfmt_jas_outputs() -> Result<()> {
        let first = sample_genfmt_decomposed_path(0.0);
        let second = sample_genfmt_decomposed_path(1.0);
        let outputs = GenfmtJasPathOutputs {
            examined_path_count: 3,
            retained_path_count: 2,
            final_normalization: Some(5.0),
            path_summaries: Vec::new(),
            retained_paths: Vec::new(),
            decomposed_paths: Some(vec![first.clone(), second.clone()]),
        };

        let data =
            FefflBinData::from_genfmt_jas_outputs(8, 1, &outputs)?.expect("decomposition output");

        assert_eq!(data.path_count(), 2);
        assert_close(data.amplitudes[(0, 0, 0, 0)], first.amplitudes[(0, 0, 0)]);
        assert_close(data.phases[(1, 1, 1, 2)], second.phases[(1, 1, 2)]);

        let total_only = GenfmtJasPathOutputs {
            examined_path_count: 1,
            retained_path_count: 1,
            final_normalization: Some(1.0),
            path_summaries: Vec::new(),
            retained_paths: Vec::new(),
            decomposed_paths: None,
        };
        assert_eq!(
            FefflBinData::from_genfmt_jas_outputs(8, 1, &total_only)?,
            None
        );
        Ok(())
    }

    #[test]
    fn builds_feffl_from_genfmt_jas_driver_output() -> Result<()> {
        let first = sample_genfmt_decomposed_path(0.0);
        let second = sample_genfmt_decomposed_path(1.0);
        let output = GenfmtJasDriverOutput {
            header: sample_genfmt_feff_bin_header(),
            path_sequence: GenfmtJasPathSequence {
                evaluations: Vec::new(),
                outputs: GenfmtJasPathOutputs {
                    examined_path_count: 3,
                    retained_path_count: 2,
                    final_normalization: Some(5.0),
                    path_summaries: Vec::new(),
                    retained_paths: Vec::new(),
                    decomposed_paths: Some(vec![first.clone(), second.clone()]),
                },
            },
            nstar_rows: None,
        };

        let data =
            FefflBinData::from_genfmt_jas_driver_output(1, &output)?.expect("decomposition output");

        assert_eq!(data.pad_width, output.header.pad_width);
        assert_eq!(data.path_count(), 2);
        assert_close(data.amplitudes[(0, 0, 0, 0)], first.amplitudes[(0, 0, 0)]);
        assert_close(data.phases[(1, 1, 1, 2)], second.phases[(1, 1, 2)]);
        Ok(())
    }

    #[test]
    fn rejects_genfmt_decomposed_output_shape_mismatch() {
        let first = sample_genfmt_decomposed_path(0.0);
        let mut bad = sample_genfmt_decomposed_path(1.0);
        bad.phases = Array3::zeros((2, 1, 3));

        assert!(matches!(
            FefflBinData::from_genfmt_output(8, 1, &[first, bad]),
            Err(IoError::FefflBinShape {
                field: "lgphchi",
                actual,
                expected,
            }) if actual == vec![2, 1, 3] && expected == vec![2, 2, 3]
        ));

        assert!(matches!(
            FefflBinData::from_genfmt_output(8, 1, &[]),
            Err(IoError::InvalidFefflBin {
                field: "path_count",
                ..
            })
        ));
    }

    #[test]
    fn rejects_bad_shapes() {
        let mut bad = sample_feffl_bin();
        bad.max_decomposition_channel = 0;
        assert!(matches!(
            feffl_bin_string(&bad),
            Err(IoError::FefflBinShape {
                field: "lgachi",
                actual,
                expected,
            }) if actual == vec![2, 2, 2, 2] && expected == vec![2, 1, 1, 2]
        ));
    }

    fn sample_feffl_bin() -> FefflBinData {
        FefflBinData {
            pad_width: 8,
            max_decomposition_channel: 1,
            amplitudes: Array4::from_shape_fn((2, 2, 2, 2), |(path, lg2, lg1, energy)| {
                0.1 * (path + 1) as f64
                    + 0.01 * lg2 as f64
                    + 0.001 * lg1 as f64
                    + 0.0001 * energy as f64
            }),
            phases: Array4::from_shape_fn((2, 2, 2, 2), |(path, lg2, lg1, energy)| {
                -0.2 * (path + 1) as f64
                    - 0.02 * lg2 as f64
                    - 0.002 * lg1 as f64
                    - 0.0002 * energy as f64
            }),
        }
    }

    fn sample_genfmt_decomposed_path(offset: f64) -> GenfmtDecomposedChiAmplitudePhase {
        GenfmtDecomposedChiAmplitudePhase {
            amplitudes: Array3::from_shape_fn((2, 2, 3), |(lg2, lg1, energy)| {
                offset + 0.1 * lg2 as f64 + 0.01 * lg1 as f64 + 0.001 * energy as f64
            }),
            phases: Array3::from_shape_fn((2, 2, 3), |(lg2, lg1, energy)| {
                -offset - 0.2 * lg2 as f64 - 0.02 * lg1 as f64 - 0.002 * energy as f64
            }),
        }
    }

    fn sample_genfmt_feff_bin_header() -> GenfmtFeffBinHeader {
        GenfmtFeffBinHeader {
            version: "refeff-test".to_string(),
            pad_width: 8,
            core_hole: 1,
            order: 2,
            initial_angular_momentum: 0,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potentials: Vec::new(),
            central_phase_shifts: Array1::from_vec(vec![Complex64::new(0.1, -0.01)]),
            complex_momenta: Array1::from_vec(vec![Complex64::new(1.0, 0.1)]),
            wave_numbers: Array1::from_vec(vec![0.5]),
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
}
