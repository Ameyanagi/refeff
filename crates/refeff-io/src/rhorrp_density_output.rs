//! FEFF RHORRP density-output selection and unit conversion.
//!
//! `RHORRP/rhorrp.f90` chooses text or Fortran-unformatted binary output from
//! the requested filename, then converts calculation-space Bohr data to the
//! Angstrom units used on disk. This module composes the text and binary codecs
//! around that final output boundary.

use std::path::{Path, PathBuf};

use ndarray::{Array1, ArrayView1, ArrayView2};
use refeff_core::{
    RhorrpDensityGridEvaluation, RhorrpError, RhorrpNearestAtomTableInput, Vector3,
    rhorrp_evaluate_density_grid, rhorrp_nearest_atom_table,
};

use crate::error::{IoError, Result};
use crate::{
    DensityGridBohr, RhorrpDensityBinBohrInput, RhorrpDensityBinData, RhorrpDensityTextBohrInput,
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

/// Parsed density-grid request plus optional text-output diagnostics.
#[derive(Debug, Clone)]
pub struct RhorrpDensityGridOutputInput<'a> {
    /// Density grid already converted to FEFF Bohr units.
    pub grid: &'a DensityGridBohr,
    /// Optional nearest-atom diagnostic columns for text output.
    pub nearest: Option<RhorrpNearestAtomColumns>,
}

/// Parsed density-grid request plus atom data for nearest-atom diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpDensityGridNearestOutputInput<'a> {
    /// Density grid already converted to FEFF Bohr units.
    pub grid: &'a DensityGridBohr,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions_bohr: ArrayView2<'a, f64>,
    /// Potential index for each atom.
    pub atom_potentials: &'a [usize],
    /// Optional leading atom count for FEFF FMS-limited nearest-atom searches.
    pub fms_atom_count: Option<usize>,
}

/// Input for building RHORRP text nearest-atom diagnostic columns.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpNearestAtomColumnsBohrInput<'a> {
    /// Grid points in Bohr as `(xyz, point)`.
    pub points_bohr: ArrayView2<'a, f64>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions_bohr: ArrayView2<'a, f64>,
    /// Potential index for each atom.
    pub atom_potentials: &'a [usize],
    /// Optional leading atom count for FEFF FMS-limited nearest-atom searches.
    pub fms_atom_count: Option<usize>,
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

/// Evaluate a parsed density grid and convert it using FEFF filename selection.
pub fn rhorrp_density_output_from_grid<F>(
    input: RhorrpDensityGridOutputInput<'_>,
    density_at: F,
) -> Result<RhorrpDensityOutputData>
where
    F: FnMut(Vector3) -> std::result::Result<f64, RhorrpError>,
{
    let evaluated = rhorrp_evaluate_density_grid(input.grid.as_rhorrp_input(), density_at)
        .map_err(|source| IoError::RhorrpDensityEvaluation { source })?;

    rhorrp_density_output_from_evaluated_grid(input.grid, &evaluated, input.nearest)
}

/// Evaluate a parsed density grid and include nearest-atom diagnostics for text output.
///
/// Binary RHORRP output does not carry nearest-atom columns, so atom diagnostics
/// are only evaluated when the grid filename selects text output.
pub fn rhorrp_density_output_from_grid_with_nearest<F>(
    input: RhorrpDensityGridNearestOutputInput<'_>,
    density_at: F,
) -> Result<RhorrpDensityOutputData>
where
    F: FnMut(Vector3) -> std::result::Result<f64, RhorrpError>,
{
    let evaluated = rhorrp_evaluate_density_grid(input.grid.as_rhorrp_input(), density_at)
        .map_err(|source| IoError::RhorrpDensityEvaluation { source })?;
    let nearest = if rhorrp_density_filename_is_binary(&input.grid.filename) {
        None
    } else {
        Some(rhorrp_nearest_atom_columns_from_bohr(
            RhorrpNearestAtomColumnsBohrInput {
                points_bohr: evaluated.points.view(),
                atom_positions_bohr: input.atom_positions_bohr,
                atom_potentials: input.atom_potentials,
                fms_atom_count: input.fms_atom_count,
            },
        )?)
    };

    rhorrp_density_output_from_evaluated_grid(input.grid, &evaluated, nearest)
}

/// Build FEFF text nearest-atom diagnostic columns from Bohr-space points.
pub fn rhorrp_nearest_atom_columns_from_bohr(
    input: RhorrpNearestAtomColumnsBohrInput<'_>,
) -> Result<RhorrpNearestAtomColumns> {
    let table = rhorrp_nearest_atom_table(RhorrpNearestAtomTableInput {
        points: input.points_bohr,
        atom_positions: input.atom_positions_bohr,
        atom_potentials: input.atom_potentials,
        fms_atom_count: input.fms_atom_count,
    })
    .map_err(|source| IoError::RhorrpDensityEvaluation { source })?;

    Ok(RhorrpNearestAtomColumns {
        displacement_bohr: table.displacement_bohr,
        atom_indices: Array1::from_vec(table.atom_indices),
        potential_indices: Array1::from_vec(table.potential_indices),
    })
}

fn rhorrp_density_output_from_evaluated_grid(
    grid: &DensityGridBohr,
    evaluated: &RhorrpDensityGridEvaluation,
    nearest: Option<RhorrpNearestAtomColumns>,
) -> Result<RhorrpDensityOutputData> {
    rhorrp_density_output_from_bohr(
        &grid.filename,
        RhorrpDensityOutputBohrInput {
            origin_bohr: grid.origin,
            axes_bohr: grid.axes.view(),
            points_per_axis: &grid.points_per_axis,
            points_bohr: evaluated.points.view(),
            density_per_bohr3: evaluated.density_per_bohr3.view(),
            nearest,
        },
    )
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

/// Evaluate and write a parsed density grid into the requested directory.
///
/// The output path is `directory/grid.filename`, matching `density.inp`, and
/// the text/binary mode is selected from that filename using FEFF's `.bin`
/// suffix rule.
pub fn write_rhorrp_density_grid_output<F>(
    directory: impl AsRef<Path>,
    input: RhorrpDensityGridOutputInput<'_>,
    density_at: F,
) -> Result<PathBuf>
where
    F: FnMut(Vector3) -> std::result::Result<f64, RhorrpError>,
{
    let path = directory.as_ref().join(&input.grid.filename);
    match rhorrp_density_output_from_grid(input, density_at)? {
        RhorrpDensityOutputData::Text(data) => write_rhorrp_density_text(&path, &data)?,
        RhorrpDensityOutputData::Binary(data) => write_rhorrp_density_bin(&path, &data)?,
    }
    Ok(path)
}

/// Evaluate and write a parsed density grid with nearest-atom text diagnostics.
pub fn write_rhorrp_density_grid_output_with_nearest<F>(
    directory: impl AsRef<Path>,
    input: RhorrpDensityGridNearestOutputInput<'_>,
    density_at: F,
) -> Result<PathBuf>
where
    F: FnMut(Vector3) -> std::result::Result<f64, RhorrpError>,
{
    let path = directory.as_ref().join(&input.grid.filename);
    match rhorrp_density_output_from_grid_with_nearest(input, density_at)? {
        RhorrpDensityOutputData::Text(data) => write_rhorrp_density_text(&path, &data)?,
        RhorrpDensityOutputData::Binary(data) => write_rhorrp_density_bin(&path, &data)?,
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use refeff_core::RhorrpError;

    use crate::{
        DensityGridBohr, DensityInput, FEFF_BOHR_ANGSTROM, IoError,
        RhorrpDensityGridNearestOutputInput, RhorrpDensityGridOutputInput,
        RhorrpDensityOutputBohrInput, RhorrpDensityOutputData, parse_rhorrp_density_bin,
        parse_rhorrp_density_text, rhorrp_density_output_from_bohr,
        rhorrp_density_output_from_grid, rhorrp_density_output_from_grid_with_nearest,
        write_rhorrp_density_grid_output, write_rhorrp_density_output_from_bohr,
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

    #[test]
    fn evaluates_parsed_grid_as_selected_text_output() -> crate::Result<()> {
        let grid = parsed_bohr_grid("density.dat")?;
        let output = rhorrp_density_output_from_grid(
            RhorrpDensityGridOutputInput {
                grid: &grid,
                nearest: None,
            },
            density_from_x,
        )?;

        let RhorrpDensityOutputData::Text(data) = output else {
            return Err(IoError::InvalidRhorrpDensity {
                field: "output",
                message: "expected text RHORRP density output".to_string(),
            });
        };
        let density_scale = 1.0 / FEFF_BOHR_ANGSTROM.powi(3);
        assert_eq!(data.point_count(), 3);
        assert_close(data.points_angstrom[(1, 0)], 0.264_588_624_5);
        assert_close(data.density_per_angstrom3[1], 1.5 * density_scale);
        Ok(())
    }

    #[test]
    fn evaluates_parsed_grid_as_selected_binary_output() -> crate::Result<()> {
        let grid = parsed_bohr_grid("density.bin")?;
        let output = rhorrp_density_output_from_grid(
            RhorrpDensityGridOutputInput {
                grid: &grid,
                nearest: None,
            },
            density_from_x,
        )?;

        let RhorrpDensityOutputData::Binary(data) = output else {
            return Err(IoError::InvalidRhorrpDensityBin {
                message: "expected binary RHORRP density output".to_string(),
            });
        };
        assert_eq!(data.point_count(), 3);
        assert_eq!(data.points_per_axis, [3]);
        assert_close(data.origin_angstrom[0], 0.0);
        assert_close(data.axes_angstrom[(0, 0)], FEFF_BOHR_ANGSTROM);
        assert_close(
            data.density_per_angstrom3[2],
            2.0 / FEFF_BOHR_ANGSTROM.powi(3),
        );
        Ok(())
    }

    #[test]
    fn writes_parsed_grid_output_to_requested_directory() -> crate::Result<()> {
        let dir = tempfile::tempdir().map_err(|source| IoError::Io {
            path: "rhorrp-density-grid-output-tempdir".into(),
            source,
        })?;
        let grid = parsed_bohr_grid("density.dat")?;

        let path = write_rhorrp_density_grid_output(
            dir.path(),
            RhorrpDensityGridOutputInput {
                grid: &grid,
                nearest: None,
            },
            density_from_x,
        )?;
        let parsed = parse_rhorrp_density_text(
            &std::fs::read_to_string(&path).map_err(|source| IoError::io(&path, source))?,
        )?;

        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("density.dat")
        );
        assert_eq!(parsed.point_count(), 3);
        Ok(())
    }

    #[test]
    fn evaluates_parsed_grid_with_nearest_text_diagnostics() -> crate::Result<()> {
        let grid = parsed_bohr_grid("density.dat")?;
        let atom_positions = nearest_atom_positions();
        let atom_potentials = [0, 2];
        let output = rhorrp_density_output_from_grid_with_nearest(
            RhorrpDensityGridNearestOutputInput {
                grid: &grid,
                atom_positions_bohr: atom_positions.view(),
                atom_potentials: &atom_potentials,
                fms_atom_count: None,
            },
            density_from_x,
        )?;

        let RhorrpDensityOutputData::Text(data) = output else {
            return Err(IoError::InvalidRhorrpDensity {
                field: "output",
                message: "expected text RHORRP density output".to_string(),
            });
        };
        let Some(nearest) = data.nearest else {
            return Err(IoError::InvalidRhorrpDensity {
                field: "nearest",
                message: "expected nearest-atom columns".to_string(),
            });
        };
        assert_eq!(nearest.atom_indices.to_vec(), vec![0, 0, 1]);
        assert_eq!(nearest.potential_indices.to_vec(), vec![0, 0, 2]);
        assert_close(nearest.displacement_bohr[(1, 0)], 0.5);
        assert_close(nearest.displacement_bohr[(2, 0)], 0.0);
        Ok(())
    }

    #[test]
    fn parsed_grid_output_wraps_evaluation_errors() -> crate::Result<()> {
        let grid = parsed_bohr_grid("density.dat")?;

        assert!(matches!(
            rhorrp_density_output_from_grid(
                RhorrpDensityGridOutputInput {
                    grid: &grid,
                    nearest: None,
                },
                |_| Err(RhorrpError::InvalidProcessCount),
            ),
            Err(IoError::RhorrpDensityEvaluation {
                source: RhorrpError::InvalidProcessCount,
            })
        ));
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

    fn nearest_atom_positions() -> Array2<f64> {
        ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    }

    fn parsed_bohr_grid(filename: &str) -> crate::Result<DensityGridBohr> {
        let density = DensityInput::parse_str(
            "density.inp",
            &format!("line {filename} 0.0 0.0 0.0\n{FEFF_BOHR_ANGSTROM} 0.0 0.0 3\n"),
        )?;
        let mut grids = density.to_bohr_grids()?;
        if grids.len() != 1 {
            return Err(IoError::InvalidRhorrpDensity {
                field: "density.inp",
                message: format!("expected one grid, got {}", grids.len()),
            });
        }
        Ok(grids.remove(0))
    }

    fn density_from_x(point: [f64; 3]) -> std::result::Result<f64, RhorrpError> {
        Ok(1.0 + point[0])
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}
