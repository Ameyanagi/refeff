//! Aggregated FEFF GENFMT output payloads.
//!
//! The numerical GENFMT port produces driver-level Rust structures before they
//! are serialized as the FEFF `feff.bin`, `list.dat`, optional `nstar.dat`, and
//! optional `feffl.bin` files. This module keeps that conversion boundary in one
//! place so the CLI can hand off a single typed bundle once the solver path is
//! wired in.

use std::path::{Path, PathBuf};

use refeff_core::{GenfmtJasDriverOutput, GenfmtOrdinaryDriverOutput};

use crate::{
    FeffBinData, FefflBinData, ListDatData, NStarDatData, Result, write_feff_bin, write_feffl_bin,
    write_list_dat, write_nstar_dat,
};

/// Typed output bundle for one GENFMT run.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOutputData {
    /// Generated `feff.bin` payload.
    pub feff_bin: FeffBinData,
    /// Generated `list.dat` payload.
    pub list_dat: ListDatData,
    /// Optional `nstar.dat` payload when `wnstar` is enabled.
    pub nstar_dat: Option<NStarDatData>,
    /// Optional `feffl.bin` path-decomposition payload for GENFMTJAS.
    pub feffl_bin: Option<FefflBinData>,
}

impl GenfmtOutputData {
    /// Build a complete ordinary GENFMT file payload bundle from driver output.
    #[must_use]
    pub fn from_genfmt_ordinary_driver_output(
        titles: &[String],
        output: &GenfmtOrdinaryDriverOutput,
    ) -> Self {
        Self {
            feff_bin: FeffBinData::from_genfmt_ordinary_driver_output(output),
            list_dat: ListDatData::from_genfmt_ordinary_driver_output(titles, output),
            nstar_dat: NStarDatData::from_genfmt_ordinary_driver_output(output),
            feffl_bin: None,
        }
    }

    /// Build a complete GENFMTJAS file payload bundle from driver output.
    pub fn from_genfmt_jas_driver_output(
        titles: &[String],
        max_decomposition_channel: usize,
        output: &GenfmtJasDriverOutput,
    ) -> Result<Self> {
        Ok(Self {
            feff_bin: FeffBinData::from_genfmt_jas_driver_output(output),
            list_dat: ListDatData::from_genfmt_jas_driver_output(titles, output),
            nstar_dat: NStarDatData::from_genfmt_jas_driver_output(output),
            feffl_bin: FefflBinData::from_genfmt_jas_driver_output(
                max_decomposition_channel,
                output,
            )?,
        })
    }
}

/// Output file paths for a generated GENFMT payload bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenfmtOutputPaths {
    /// Destination path for `feff.bin`.
    pub feff_bin: PathBuf,
    /// Destination path for `list.dat`.
    pub list_dat: PathBuf,
    /// Destination path for optional `nstar.dat`.
    pub nstar_dat: Option<PathBuf>,
    /// Destination path for optional `feffl.bin`.
    pub feffl_bin: Option<PathBuf>,
}

impl GenfmtOutputPaths {
    /// Standard GENFMT output paths rooted in a working directory.
    #[must_use]
    pub fn standard(work_dir: impl AsRef<Path>) -> Self {
        let work_dir = work_dir.as_ref();
        Self {
            feff_bin: work_dir.join("feff.bin"),
            list_dat: work_dir.join("list.dat"),
            nstar_dat: Some(work_dir.join("nstar.dat")),
            feffl_bin: Some(work_dir.join("feffl.bin")),
        }
    }
}

/// Write a GENFMT output payload bundle to its configured files.
///
/// `feff.bin` and `list.dat` are always written. Optional outputs are written
/// only when both the payload and destination path are present.
pub fn write_genfmt_output_files(
    paths: &GenfmtOutputPaths,
    data: &GenfmtOutputData,
) -> Result<usize> {
    write_feff_bin(&paths.feff_bin, &data.feff_bin)?;
    write_list_dat(&paths.list_dat, &data.list_dat)?;
    let mut written = 2;

    if let (Some(path), Some(nstar_dat)) = (&paths.nstar_dat, &data.nstar_dat) {
        write_nstar_dat(path, nstar_dat)?;
        written += 1;
    }
    if let (Some(path), Some(feffl_bin)) = (&paths.feffl_bin, &data.feffl_bin) {
        write_feffl_bin(path, feffl_bin)?;
        written += 1;
    }

    Ok(written)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3};
    use num_complex::Complex64;
    use refeff_core::{
        GenfmtDecomposedChiAmplitudePhase, GenfmtFeffBinHeader, GenfmtFeffBinPotential,
        GenfmtJasDriverOutput, GenfmtJasPathOutputs, GenfmtJasPathSequence, GenfmtNStarRow,
        GenfmtNStarRows, GenfmtOrdinaryDriverOutput, GenfmtOrdinaryPathOutputs,
        GenfmtOrdinaryPathSequence, GenfmtRetainedPathOutput,
    };

    use super::*;
    use crate::{
        IoError, feff_bin_string, feffl_bin_string, list_dat_string, nstar_dat_string,
        read_feff_bin, read_feffl_bin, read_list_dat, read_nstar_dat, write_genfmt_output_files,
    };

    #[test]
    fn builds_ordinary_bundle_from_genfmt_driver_output() {
        let titles = sample_titles();
        let output = sample_ordinary_output(true);

        let data = GenfmtOutputData::from_genfmt_ordinary_driver_output(&titles, &output);

        assert_eq!(data.feff_bin.paths.len(), 2);
        assert_eq!(data.list_dat.titles, titles);
        assert_eq!(data.list_dat.entries.len(), 2);
        assert_eq!(
            data.nstar_dat.as_ref().map(|data| data.entries.len()),
            Some(2)
        );
        assert!(data.feffl_bin.is_none());
    }

    #[test]
    fn builds_jas_bundle_from_genfmt_driver_output() -> Result<()> {
        let titles = sample_titles();
        let output = sample_jas_output(true, true);

        let data = GenfmtOutputData::from_genfmt_jas_driver_output(&titles, 1, &output)?;

        assert_eq!(data.feff_bin.paths.len(), 2);
        assert_eq!(data.list_dat.titles, titles);
        assert_eq!(
            data.nstar_dat.as_ref().map(|data| data.entries.len()),
            Some(2)
        );
        assert_eq!(
            data.feffl_bin.as_ref().map(FefflBinData::path_count),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn skips_absent_optional_jas_outputs() -> Result<()> {
        let output = sample_jas_output(false, false);

        let data = GenfmtOutputData::from_genfmt_jas_driver_output(&sample_titles(), 1, &output)?;

        assert!(data.nstar_dat.is_none());
        assert!(data.feffl_bin.is_none());
        Ok(())
    }

    #[test]
    fn writes_and_roundtrips_generated_output_files() -> Result<()> {
        let output = sample_jas_output(true, true);
        let data = GenfmtOutputData::from_genfmt_jas_driver_output(&sample_titles(), 1, &output)?;
        let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
        let paths = GenfmtOutputPaths::standard(temp.path());

        let written = write_genfmt_output_files(&paths, &data)?;

        assert_eq!(written, 4);
        assert_eq!(
            feff_bin_string(&read_feff_bin(&paths.feff_bin)?)?,
            feff_bin_string(&data.feff_bin)?
        );
        assert_eq!(
            list_dat_string(&read_list_dat(&paths.list_dat)?)?,
            list_dat_string(&data.list_dat)?
        );

        let expected_nstar = data.nstar_dat.as_ref().expect("nstar output");
        assert_eq!(
            nstar_dat_string(&read_nstar_dat(paths.nstar_dat.as_ref().unwrap())?)?,
            nstar_dat_string(expected_nstar)?
        );

        let expected_feffl = data.feffl_bin.as_ref().expect("feffl output");
        let actual_feffl = read_feffl_bin(
            paths.feffl_bin.as_ref().unwrap(),
            expected_feffl.pad_width,
            expected_feffl.path_count(),
            expected_feffl.energy_count(),
            expected_feffl.max_decomposition_channel,
        )?;
        assert_eq!(
            feffl_bin_string(&actual_feffl)?,
            feffl_bin_string(expected_feffl)?
        );
        Ok(())
    }

    fn sample_titles() -> Vec<String> {
        vec!["PATH  Rmax= 6.000".to_string()]
    }

    fn sample_ordinary_output(with_nstar: bool) -> GenfmtOrdinaryDriverOutput {
        GenfmtOrdinaryDriverOutput {
            header: sample_genfmt_feff_bin_header(),
            path_sequence: GenfmtOrdinaryPathSequence {
                evaluations: Vec::new(),
                outputs: GenfmtOrdinaryPathOutputs {
                    examined_path_count: 2,
                    retained_path_count: 2,
                    final_normalization: Some(5.0),
                    path_summaries: Vec::new(),
                    retained_paths: vec![
                        sample_genfmt_retained_path_output(17),
                        sample_genfmt_retained_path_output(23),
                    ],
                },
            },
            nstar_rows: with_nstar.then(sample_nstar_rows),
        }
    }

    fn sample_jas_output(with_nstar: bool, with_decomposition: bool) -> GenfmtJasDriverOutput {
        GenfmtJasDriverOutput {
            header: sample_genfmt_feff_bin_header(),
            path_sequence: GenfmtJasPathSequence {
                evaluations: Vec::new(),
                outputs: GenfmtJasPathOutputs {
                    examined_path_count: 2,
                    retained_path_count: 2,
                    final_normalization: Some(5.0),
                    path_summaries: Vec::new(),
                    retained_paths: vec![
                        sample_genfmt_retained_path_output(17),
                        sample_genfmt_retained_path_output(23),
                    ],
                    decomposed_paths: with_decomposition.then(|| {
                        vec![
                            sample_genfmt_decomposed_path(0.0),
                            sample_genfmt_decomposed_path(1.0),
                        ]
                    }),
                },
            },
            nstar_rows: with_nstar.then(sample_nstar_rows),
        }
    }

    fn sample_genfmt_retained_path_output(path_index: usize) -> GenfmtRetainedPathOutput {
        GenfmtRetainedPathOutput {
            path_index,
            degeneracy: 4.0,
            criterion_percent: 12.5,
            effective_half_path_length_bohr: 2.4,
            effective_half_path_length_angstrom: 1.270_025_397_6,
            list_sigma2: 0.0,
            potential_indices: Array1::from_vec(vec![1, 2, 0]),
            positions: Array2::from_shape_fn((3, 3), |(leg, axis)| match (leg, axis) {
                (0, 0) => 1.0,
                (0, 1) => 0.5,
                (0, 2) => -0.25,
                (1, 0) => 0.4,
                (1, 1) => -0.3,
                (1, 2) => 1.2,
                _ => 0.0,
            }),
            beta_angles: Array1::from_vec(vec![0.10, 0.20, 0.30]),
            eta_angles: Array1::from_vec(vec![0.40, 0.50, 0.60]),
            leg_lengths: Array1::from_vec(vec![1.0, 1.1, 1.2]),
            amplitudes: Array1::from_vec(vec![0.2, 0.3, 0.4]),
            phases: Array1::from_vec(vec![0.1, 1.2, 2.3]),
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

    fn sample_nstar_rows() -> GenfmtNStarRows {
        GenfmtNStarRows {
            primary_polarization: [0.25, -0.5, 1.0],
            rows: vec![
                GenfmtNStarRow {
                    path_number: 1,
                    nstar: 2.345,
                },
                GenfmtNStarRow {
                    path_number: 2,
                    nstar: -0.125,
                },
            ],
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
            potentials: vec![
                GenfmtFeffBinPotential {
                    label: "Cu".to_string(),
                    atomic_number: 29,
                },
                GenfmtFeffBinPotential {
                    label: "O".to_string(),
                    atomic_number: 8,
                },
                GenfmtFeffBinPotential {
                    label: "C".to_string(),
                    atomic_number: 6,
                },
            ],
            central_phase_shifts: Array1::from_vec(vec![
                Complex64::new(0.1, -0.01),
                Complex64::new(0.2, -0.02),
                Complex64::new(0.3, -0.03),
            ]),
            complex_momenta: Array1::from_vec(vec![
                Complex64::new(1.0, 0.1),
                Complex64::new(1.1, 0.2),
                Complex64::new(1.2, 0.3),
            ]),
            wave_numbers: Array1::from_vec(vec![0.5, 0.6, 0.7]),
        }
    }
}
