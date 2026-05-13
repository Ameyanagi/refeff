//! FEFF RHORRP density-output selection and unit conversion.
//!
//! `RHORRP/rhorrp.f90` chooses text or Fortran-unformatted binary output from
//! the requested filename, then converts calculation-space Bohr data to the
//! Angstrom units used on disk. This module composes the text and binary codecs
//! around that final output boundary.

use std::path::Path;

use ndarray::{ArrayView1, ArrayView2};

use crate::error::Result;
use crate::{
    RhorrpDensityBinBohrInput, RhorrpDensityBinData, RhorrpDensityTextBohrInput,
    RhorrpDensityTextData, RhorrpNearestAtomColumns, rhorrp_density_bin_from_bohr,
    rhorrp_density_filename_is_binary, rhorrp_density_text_from_bohr, write_rhorrp_density_bin,
    write_rhorrp_density_text,
};

/// Bohr-unit RHORRP calculation output plus grid metadata.
#[derive(Debug, Clone)]
pub struct RhorrpDensityOutputBohrInput<'a> {
    /// Grid origin in Bohr.
    pub origin_bohr: [f64; 3],
    /// Grid axis vectors in Bohr as `(xyz, dimension)`.
    pub axes_bohr: ArrayView2<'a, f64>,
    /// Number of grid points along each active axis.
    pub points_per_axis: &'a [usize],
    /// Grid points in Bohr as `(xyz, point)`, matching FEFF `points(3, totpts)`.
    pub points_bohr: ArrayView2<'a, f64>,
    /// Density values in inverse cubic Bohr, in FEFF point traversal order.
    pub density_per_bohr3: ArrayView1<'a, f64>,
    /// Optional nearest-atom diagnostic columns for text output.
    pub nearest: Option<RhorrpNearestAtomColumns>,
}

/// RHORRP density output after FEFF-compatible mode selection.
#[derive(Debug, Clone, PartialEq)]
pub enum RhorrpDensityOutputData {
    /// ASCII output with Cartesian point rows.
    Text(RhorrpDensityTextData),
    /// Fortran-unformatted binary output with grid metadata and density.
    Binary(RhorrpDensityBinData),
}

impl RhorrpDensityOutputData {
    /// Whether this output should be written in RHORRP binary format.
    #[must_use]
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary(_))
    }
}

/// Convert Bohr-unit RHORRP calculation output using FEFF filename selection.
pub fn rhorrp_density_output_from_bohr(
    filename: &str,
    input: RhorrpDensityOutputBohrInput<'_>,
) -> Result<RhorrpDensityOutputData> {
    if rhorrp_density_filename_is_binary(filename) {
        Ok(RhorrpDensityOutputData::Binary(
            rhorrp_density_bin_from_bohr(RhorrpDensityBinBohrInput {
                origin_bohr: input.origin_bohr,
                axes_bohr: input.axes_bohr,
                points_per_axis: input.points_per_axis,
                density_per_bohr3: input.density_per_bohr3,
            })?,
        ))
    } else {
        Ok(RhorrpDensityOutputData::Text(
            rhorrp_density_text_from_bohr(RhorrpDensityTextBohrInput {
                points_bohr: input.points_bohr,
                density_per_bohr3: input.density_per_bohr3,
                nearest: input.nearest,
            })?,
        ))
    }
}

/// Write Bohr-unit RHORRP calculation output using FEFF filename selection.
pub fn write_rhorrp_density_output_from_bohr(
    path: impl AsRef<Path>,
    input: RhorrpDensityOutputBohrInput<'_>,
) -> Result<()> {
    let path = path.as_ref();
    let filename = path.to_string_lossy();
    match rhorrp_density_output_from_bohr(&filename, input)? {
        RhorrpDensityOutputData::Text(data) => write_rhorrp_density_text(path, &data),
        RhorrpDensityOutputData::Binary(data) => write_rhorrp_density_bin(path, &data),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use crate::{
        IoError, RhorrpDensityOutputBohrInput, RhorrpDensityOutputData, parse_rhorrp_density_bin,
        parse_rhorrp_density_text, rhorrp_density_output_from_bohr,
        write_rhorrp_density_output_from_bohr,
    };

    #[test]
    fn selects_text_density_output_from_filename() -> crate::Result<()> {
        let points_bohr = sample_points_bohr();
        let axes_bohr = sample_axes_bohr();
        let density_per_bohr3 = sample_density_per_bohr3();
        let output = rhorrp_density_output_from_bohr(
            "density.dat",
            RhorrpDensityOutputBohrInput {
                origin_bohr: [0.1, -0.2, 0.3],
                axes_bohr: axes_bohr.view(),
                points_per_axis: &[3, 2],
                points_bohr: points_bohr.view(),
                density_per_bohr3: density_per_bohr3.view(),
                nearest: None,
            },
        )?;

        let RhorrpDensityOutputData::Text(data) = output else {
            return Err(IoError::InvalidRhorrpDensity {
                field: "output",
                message: "expected text RHORRP density output".to_string(),
            });
        };
        assert_eq!(data.point_count(), 6);
        assert_close(data.points_angstrom[(1, 0)], 0.105_835_449_8);
        assert_close(data.density_per_angstrom3[1], 13.496_666_074_208_301);
        Ok(())
    }

    #[test]
    fn selects_binary_density_output_from_filename() -> crate::Result<()> {
        let points_bohr = sample_points_bohr();
        let axes_bohr = sample_axes_bohr();
        let density_per_bohr3 = sample_density_per_bohr3();
        let output = rhorrp_density_output_from_bohr(
            "density.BIN",
            RhorrpDensityOutputBohrInput {
                origin_bohr: [0.1, -0.2, 0.3],
                axes_bohr: axes_bohr.view(),
                points_per_axis: &[3, 2],
                points_bohr: points_bohr.view(),
                density_per_bohr3: density_per_bohr3.view(),
                nearest: None,
            },
        )?;

        let RhorrpDensityOutputData::Binary(data) = output else {
            return Err(IoError::InvalidRhorrpDensityBin {
                message: "expected binary RHORRP density output".to_string(),
            });
        };
        assert_eq!(data.point_count(), 6);
        assert_eq!(data.points_per_axis, [3, 2]);
        assert_close(data.origin_angstrom[0], 0.052_917_724_9);
        assert_close(data.axes_angstrom[(1, 1)], 0.661_471_561_25);
        assert_close(data.density_per_angstrom3[2], -0.843_541_629_638_018_8);
        Ok(())
    }

    #[test]
    fn writes_selected_density_output_files() -> crate::Result<()> {
        let dir = tempfile::tempdir().map_err(|source| IoError::Io {
            path: "rhorrp-density-output-tempdir".into(),
            source,
        })?;
        let points_bohr = sample_points_bohr();
        let axes_bohr = sample_axes_bohr();
        let density_per_bohr3 = sample_density_per_bohr3();

        let text_path = dir.path().join("density.dat");
        write_rhorrp_density_output_from_bohr(
            &text_path,
            RhorrpDensityOutputBohrInput {
                origin_bohr: [0.1, -0.2, 0.3],
                axes_bohr: axes_bohr.view(),
                points_per_axis: &[3, 2],
                points_bohr: points_bohr.view(),
                density_per_bohr3: density_per_bohr3.view(),
                nearest: None,
            },
        )?;
        let parsed_text = parse_rhorrp_density_text(
            &std::fs::read_to_string(&text_path)
                .map_err(|source| IoError::io(&text_path, source))?,
        )?;
        assert_eq!(parsed_text.point_count(), 6);

        let bin_path = dir.path().join("density.bin");
        write_rhorrp_density_output_from_bohr(
            &bin_path,
            RhorrpDensityOutputBohrInput {
                origin_bohr: [0.1, -0.2, 0.3],
                axes_bohr: axes_bohr.view(),
                points_per_axis: &[3, 2],
                points_bohr: points_bohr.view(),
                density_per_bohr3: density_per_bohr3.view(),
                nearest: None,
            },
        )?;
        let parsed_bin = parse_rhorrp_density_bin(
            &std::fs::read(&bin_path).map_err(|source| IoError::io(&bin_path, source))?,
        )?;
        assert_eq!(parsed_bin.point_count(), 6);
        Ok(())
    }

    fn sample_points_bohr() -> Array2<f64> {
        Array2::from_shape_fn((3, 6), |(axis, point)| {
            0.1 * (point + 1) as f64 + 0.25 * axis as f64
        })
    }

    fn sample_axes_bohr() -> Array2<f64> {
        ndarray::arr2(&[[1.0, -0.2], [0.5, 1.25], [-0.25, 0.75]])
    }

    fn sample_density_per_bohr3() -> Array1<f64> {
        ndarray::arr1(&[0.5, 2.0, -0.125, 0.0, 1.0, -2.0])
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}
