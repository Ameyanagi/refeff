//! FEFF RHORRP density-output selection and unit conversion.
//!
//! `RHORRP/rhorrp.f90` chooses text or Fortran-unformatted binary output from
//! the requested filename, then converts calculation-space Bohr data to the
//! Angstrom units used on disk. This module composes the text and binary codecs
//! around that final output boundary.

use std::path::{Path, PathBuf};

use ndarray::{Array1, ArrayView1, ArrayView2, ArrayView4};
use num_complex::Complex64;
use refeff_core::{
    RhorrpDensityGridEvaluation, RhorrpDensityGridFromTablesInput, RhorrpError,
    RhorrpNearestAtomTableInput, RhorrpWavefunctionTables, Vector3, rhorrp_evaluate_density_grid,
    rhorrp_evaluate_density_grid_from_tables, rhorrp_nearest_atom_table,
};

use crate::error::{IoError, Result};
use crate::{
    DensityGridBohr, RhorrpDensityBinBohrInput, RhorrpDensityBinData, RhorrpDensityTextBohrInput,
    RhorrpDensityTextData, RhorrpFmsInputHandoff, RhorrpGeomHandoff, RhorrpNearestAtomColumns,
    RhorrpPhaseBinHandoff, rhorrp_density_bin_from_bohr, rhorrp_density_filename_is_binary,
    rhorrp_density_text_from_bohr, write_rhorrp_density_bin, write_rhorrp_density_text,
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

/// Parsed density-grid request plus RHORRP `init_wavefunctions` handoff tables.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpDensityGridTablesOutputInput<'a> {
    /// Density grid already converted to FEFF Bohr units.
    pub grid: &'a DensityGridBohr,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions_bohr: ArrayView2<'a, f64>,
    /// Potential index for each atom.
    pub atom_potentials: &'a [usize],
    /// Optional leading atom count for FEFF FMS-limited density evaluation.
    pub fms_atom_count: Option<usize>,
    /// Optional leading atom count for text-output nearest-atom diagnostics.
    pub nearest_atom_count: Option<usize>,
    /// Complex contour energies `em`, in Hartree.
    pub energies_hartree: ArrayView1<'a, Complex64>,
    /// Final `init_wavefunctions` reference energy `eref0`, in Hartree.
    pub reference_energy_hartree: Complex64,
    /// All-potential wavefunction tables from FEFF `init_wavefunctions`.
    pub wavefunctions: &'a RhorrpWavefunctionTables,
    /// Optional FEFF `gg_diag.bin` blocks promoted to `(energy, atom, L, L')`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex64>>,
    /// FEFF logarithmic radial-grid offset `x0`.
    pub radial_x0: f64,
    /// FEFF logarithmic radial-grid step `dx`.
    pub radial_dx: f64,
    /// FEFF `ne1`: contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// FEFF chemical potential `xmu`, in Hartree.
    pub chemical_potential_hartree: f64,
    /// FEFF electronic temperature, in Hartree.
    pub temperature_hartree: f64,
    /// Optional COMPTON chemical-potential override, already in Hartree.
    pub chemical_potential_override_hartree: Option<f64>,
}

/// RHORRP density-grid tables plus the parsed `rhorrp_init` handoffs.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpDensityGridTablesHandoffInput<'a> {
    /// Density grid already converted to FEFF Bohr units.
    pub grid: &'a DensityGridBohr,
    /// RHORRP atom geometry from FEFF `geom.dat`.
    pub geometry: &'a RhorrpGeomHandoff,
    /// RHORRP FMS controls from FEFF `fms.inp`.
    pub fms: &'a RhorrpFmsInputHandoff,
    /// RHORRP contour and chemical-potential controls from FEFF `phase.bin`.
    pub phase: &'a RhorrpPhaseBinHandoff,
    /// Final `init_wavefunctions` reference energy `eref0`, in Hartree.
    pub reference_energy_hartree: Complex64,
    /// All-potential wavefunction tables from FEFF `init_wavefunctions`.
    pub wavefunctions: &'a RhorrpWavefunctionTables,
    /// Optional FEFF `gg_diag.bin` blocks promoted to `(energy, atom, L, L')`.
    pub diagonal_scattering_matrices: Option<ArrayView4<'a, Complex64>>,
    /// FEFF logarithmic radial-grid offset `x0`.
    pub radial_x0: f64,
    /// FEFF logarithmic radial-grid step `dx`.
    pub radial_dx: f64,
    /// FEFF electronic temperature, in Hartree.
    pub temperature_hartree: f64,
    /// Optional COMPTON chemical-potential override, already in Hartree.
    pub chemical_potential_override_hartree: Option<f64>,
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

/// Build table-backed RHORRP density-grid input from parsed initialization handoffs.
///
/// This applies FEFF's `rhorrp(point, point)` `nearest_atom(..., fmsF=.true.)`
/// rule by using `inclus(0)` from the composed `fms.inp` and `geom.dat`
/// handoffs. FEFF's final text-output diagnostic calls `nearest_atom` with
/// `fmsF=.false.`, so diagnostics still search all atoms.
pub fn rhorrp_density_grid_tables_input_from_handoffs<'a>(
    input: RhorrpDensityGridTablesHandoffInput<'a>,
) -> Result<RhorrpDensityGridTablesOutputInput<'a>> {
    Ok(RhorrpDensityGridTablesOutputInput {
        grid: input.grid,
        atom_positions_bohr: input.geometry.atom_positions_bohr.view(),
        atom_potentials: &input.geometry.atom_potentials,
        fms_atom_count: Some(input.fms.central_fms_atom_count(input.geometry)?),
        nearest_atom_count: None,
        energies_hartree: input.phase.energies_hartree.view(),
        reference_energy_hartree: input.reference_energy_hartree,
        wavefunctions: input.wavefunctions,
        diagonal_scattering_matrices: input.diagonal_scattering_matrices,
        radial_x0: input.radial_x0,
        radial_dx: input.radial_dx,
        real_axis_count: input.phase.real_axis_count,
        chemical_potential_hartree: input.phase.chemical_potential_hartree,
        temperature_hartree: input.temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })
}

/// Evaluate a parsed density grid from RHORRP handoff tables and convert it using FEFF output rules.
///
/// Text output includes the same nearest-atom diagnostics as FEFF
/// `calculate_density`; binary output stores only the grid metadata and
/// densities.
pub fn rhorrp_density_output_from_grid_tables(
    input: RhorrpDensityGridTablesOutputInput<'_>,
) -> Result<RhorrpDensityOutputData> {
    let evaluated = rhorrp_evaluate_density_grid_from_tables(RhorrpDensityGridFromTablesInput {
        grid: input.grid.as_rhorrp_input(),
        atom_positions: input.atom_positions_bohr,
        atom_potentials: input.atom_potentials,
        fms_atom_count: input.fms_atom_count,
        energies_hartree: input.energies_hartree,
        reference_energy_hartree: input.reference_energy_hartree,
        wavefunctions: input.wavefunctions,
        diagonal_scattering_matrices: input.diagonal_scattering_matrices,
        radial_x0: input.radial_x0,
        radial_dx: input.radial_dx,
        real_axis_count: input.real_axis_count,
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree: input.temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })
    .map_err(|source| IoError::RhorrpDensityEvaluation { source })?;
    let nearest = if rhorrp_density_filename_is_binary(&input.grid.filename) {
        None
    } else {
        Some(rhorrp_nearest_atom_columns_from_bohr(
            RhorrpNearestAtomColumnsBohrInput {
                points_bohr: evaluated.points.view(),
                atom_positions_bohr: input.atom_positions_bohr,
                atom_potentials: input.atom_potentials,
                fms_atom_count: input.nearest_atom_count,
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

/// Evaluate and write a parsed density grid from RHORRP handoff tables.
pub fn write_rhorrp_density_grid_output_from_tables(
    directory: impl AsRef<Path>,
    input: RhorrpDensityGridTablesOutputInput<'_>,
) -> Result<PathBuf> {
    let path = directory.as_ref().join(&input.grid.filename);
    match rhorrp_density_output_from_grid_tables(input)? {
        RhorrpDensityOutputData::Text(data) => write_rhorrp_density_text(&path, &data)?,
        RhorrpDensityOutputData::Binary(data) => write_rhorrp_density_bin(&path, &data)?,
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, Array4};
    use num_complex::Complex64;
    use refeff_core::{
        RhorrpDensityGridFromTablesInput, RhorrpError, RhorrpWavefunctionTables,
        rhorrp_evaluate_density_grid_from_tables,
    };

    use crate::{
        DensityGridBohr, DensityInput, FEFF_BOHR_ANGSTROM, IoError,
        RhorrpDensityGridNearestOutputInput, RhorrpDensityGridOutputInput,
        RhorrpDensityGridTablesHandoffInput, RhorrpDensityGridTablesOutputInput,
        RhorrpDensityOutputBohrInput, RhorrpDensityOutputData, RhorrpFmsInputHandoff,
        RhorrpGeomHandoff, RhorrpPhaseBinHandoff, parse_rhorrp_density_bin,
        parse_rhorrp_density_text, rhorrp_density_grid_tables_input_from_handoffs,
        rhorrp_density_output_from_bohr, rhorrp_density_output_from_grid,
        rhorrp_density_output_from_grid_tables, rhorrp_density_output_from_grid_with_nearest,
        write_rhorrp_density_grid_output, write_rhorrp_density_grid_output_from_tables,
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
    fn evaluates_parsed_grid_from_tables_with_nearest_text_diagnostics() -> crate::Result<()> {
        let grid = parsed_bohr_grid("density.dat")?;
        let source = sample_table_source();

        let output = rhorrp_density_output_from_grid_tables(table_output_input(&grid, &source))?;

        let RhorrpDensityOutputData::Text(data) = output else {
            return Err(IoError::InvalidRhorrpDensity {
                field: "output",
                message: "expected text RHORRP density output".to_string(),
            });
        };
        let Some(nearest) = data.nearest.as_ref() else {
            return Err(IoError::InvalidRhorrpDensity {
                field: "nearest",
                message: "expected nearest-atom columns".to_string(),
            });
        };
        let expected = rhorrp_evaluate_density_grid_from_tables(RhorrpDensityGridFromTablesInput {
            grid: grid.as_rhorrp_input(),
            atom_positions: source.atom_positions_bohr.view(),
            atom_potentials: &source.atom_potentials,
            fms_atom_count: Some(2),
            energies_hartree: source.energies_hartree.view(),
            reference_energy_hartree: source.reference_energy_hartree,
            wavefunctions: &source.wavefunctions,
            diagonal_scattering_matrices: Some(source.diagonal_scattering.view()),
            radial_x0: source.radial_x0,
            radial_dx: source.radial_dx,
            real_axis_count: source.real_axis_count,
            chemical_potential_hartree: source.chemical_potential_hartree,
            temperature_hartree: source.temperature_hartree,
            chemical_potential_override_hartree: None,
        })
        .map_err(|source| IoError::RhorrpDensityEvaluation { source })?;

        assert_eq!(data.point_count(), 3);
        assert_eq!(nearest.atom_indices.to_vec(), vec![0, 0, 1]);
        assert_eq!(nearest.potential_indices.to_vec(), vec![0, 0, 1]);
        assert_close(nearest.displacement_bohr[(2, 0)], 0.0);
        let density_scale = 1.0 / FEFF_BOHR_ANGSTROM.powi(3);
        for point in 0..data.point_count() {
            assert_close(
                data.density_per_angstrom3[point],
                expected.density_per_bohr3[point] * density_scale,
            );
        }
        Ok(())
    }

    #[test]
    fn builds_table_grid_input_from_rhorrp_handoffs() -> crate::Result<()> {
        let grid = parsed_bohr_line_grid("density.dat", 2.0, 5)?;
        let source = sample_table_source();
        let geometry = sample_table_geometry_with_outer_atom();
        let fms = sample_table_fms();
        let phase = sample_table_phase(&source);

        let input =
            rhorrp_density_grid_tables_input_from_handoffs(RhorrpDensityGridTablesHandoffInput {
                grid: &grid,
                geometry: &geometry,
                fms: &fms,
                phase: &phase,
                reference_energy_hartree: source.reference_energy_hartree,
                wavefunctions: &source.wavefunctions,
                diagonal_scattering_matrices: Some(source.diagonal_scattering.view()),
                radial_x0: source.radial_x0,
                radial_dx: source.radial_dx,
                temperature_hartree: source.temperature_hartree,
                chemical_potential_override_hartree: None,
            })?;

        assert_eq!(input.fms_atom_count, Some(2));
        assert_eq!(input.nearest_atom_count, None);
        assert_eq!(input.atom_potentials, &[0, 1, 1]);
        assert_eq!(
            input.energies_hartree.to_vec(),
            source.energies_hartree.to_vec()
        );
        assert_eq!(input.real_axis_count, source.real_axis_count);
        assert_eq!(
            input.chemical_potential_hartree,
            source.chemical_potential_hartree
        );

        let RhorrpDensityOutputData::Text(data) = rhorrp_density_output_from_grid_tables(input)?
        else {
            return Err(IoError::InvalidRhorrpDensity {
                field: "output",
                message: "expected text RHORRP density output".to_string(),
            });
        };
        let nearest = data
            .nearest
            .as_ref()
            .ok_or_else(|| IoError::InvalidRhorrpDensity {
                field: "nearest",
                message: "expected nearest-atom columns".to_string(),
            })?;
        assert_eq!(nearest.atom_indices.to_vec(), vec![0, 0, 1, 1, 2]);
        assert_close(nearest.displacement_bohr[(4, 0)], 0.0);
        Ok(())
    }

    #[test]
    fn rejects_table_grid_handoff_with_mismatched_fms_geometry() -> crate::Result<()> {
        let grid = parsed_bohr_grid("density.dat")?;
        let source = sample_table_source();
        let geometry = sample_table_geometry(&source);
        let fms = RhorrpFmsInputHandoff {
            potential_count: 1,
            ..sample_table_fms()
        };
        let phase = sample_table_phase(&source);

        let result =
            rhorrp_density_grid_tables_input_from_handoffs(RhorrpDensityGridTablesHandoffInput {
                grid: &grid,
                geometry: &geometry,
                fms: &fms,
                phase: &phase,
                reference_energy_hartree: source.reference_energy_hartree,
                wavefunctions: &source.wavefunctions,
                diagonal_scattering_matrices: Some(source.diagonal_scattering.view()),
                radial_x0: source.radial_x0,
                radial_dx: source.radial_dx,
                temperature_hartree: source.temperature_hartree,
                chemical_potential_override_hartree: None,
            });
        assert!(matches!(result, Err(IoError::Parse { .. })));
        Ok(())
    }

    #[test]
    fn writes_parsed_grid_output_from_tables_to_requested_directory() -> crate::Result<()> {
        let dir = tempfile::tempdir().map_err(|source| IoError::Io {
            path: "rhorrp-density-table-output-tempdir".into(),
            source,
        })?;
        let grid = parsed_bohr_grid("density.bin")?;
        let source = sample_table_source();

        let path = write_rhorrp_density_grid_output_from_tables(
            dir.path(),
            table_output_input(&grid, &source),
        )?;
        let parsed = parse_rhorrp_density_bin(
            &std::fs::read(&path).map_err(|source| IoError::io(&path, source))?,
        )?;

        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("density.bin")
        );
        assert_eq!(parsed.point_count(), 3);
        assert_eq!(parsed.points_per_axis, [3]);
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
        parsed_bohr_line_grid(filename, 1.0, 3)
    }

    fn parsed_bohr_line_grid(
        filename: &str,
        axis_length_bohr: f64,
        point_count: usize,
    ) -> crate::Result<DensityGridBohr> {
        let axis_length_angstrom = axis_length_bohr * FEFF_BOHR_ANGSTROM;
        let density = DensityInput::parse_str(
            "density.inp",
            &format!("line {filename} 0.0 0.0 0.0\n{axis_length_angstrom} 0.0 0.0 {point_count}\n"),
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

    struct TableSource {
        atom_positions_bohr: Array2<f64>,
        atom_potentials: Vec<usize>,
        energies_hartree: Array1<Complex64>,
        reference_energy_hartree: Complex64,
        wavefunctions: RhorrpWavefunctionTables,
        diagonal_scattering: Array4<Complex64>,
        radial_x0: f64,
        radial_dx: f64,
        real_axis_count: usize,
        chemical_potential_hartree: f64,
        temperature_hartree: f64,
    }

    fn table_output_input<'a>(
        grid: &'a DensityGridBohr,
        source: &'a TableSource,
    ) -> RhorrpDensityGridTablesOutputInput<'a> {
        RhorrpDensityGridTablesOutputInput {
            grid,
            atom_positions_bohr: source.atom_positions_bohr.view(),
            atom_potentials: &source.atom_potentials,
            fms_atom_count: Some(2),
            nearest_atom_count: Some(2),
            energies_hartree: source.energies_hartree.view(),
            reference_energy_hartree: source.reference_energy_hartree,
            wavefunctions: &source.wavefunctions,
            diagonal_scattering_matrices: Some(source.diagonal_scattering.view()),
            radial_x0: source.radial_x0,
            radial_dx: source.radial_dx,
            real_axis_count: source.real_axis_count,
            chemical_potential_hartree: source.chemical_potential_hartree,
            temperature_hartree: source.temperature_hartree,
            chemical_potential_override_hartree: None,
        }
    }

    fn sample_table_geometry(source: &TableSource) -> RhorrpGeomHandoff {
        RhorrpGeomHandoff {
            atom_positions_bohr: source.atom_positions_bohr.clone(),
            atom_potentials: source.atom_potentials.clone(),
            representative_atoms: vec![0, 1],
        }
    }

    fn sample_table_geometry_with_outer_atom() -> RhorrpGeomHandoff {
        RhorrpGeomHandoff {
            atom_positions_bohr: ndarray::arr2(&[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]),
            atom_potentials: vec![0, 1, 1],
            representative_atoms: vec![0, 1],
        }
    }

    fn sample_table_fms() -> RhorrpFmsInputHandoff {
        RhorrpFmsInputHandoff {
            fms_radius_bohr: 1.1,
            potential_count: 2,
            max_angular_momentum: 1,
            angular_momentum_count: 2,
        }
    }

    fn sample_table_phase(source: &TableSource) -> RhorrpPhaseBinHandoff {
        RhorrpPhaseBinHandoff {
            energies_hartree: source.energies_hartree.clone(),
            chemical_potential_hartree: source.chemical_potential_hartree,
            real_axis_count: source.real_axis_count,
            xsph_phase_shifts: source.wavefunctions.phase_shifts.clone(),
        }
    }

    fn sample_table_source() -> TableSource {
        let energies = Array1::from_vec(vec![
            Complex64::new(-0.030, 0.070),
            Complex64::new(-0.030, 0.035),
            Complex64::new(-0.030, 0.000),
            Complex64::new(0.010, 0.000),
            Complex64::new(0.065, 0.000),
            Complex64::new(0.130, 0.000),
            Complex64::new(0.045, 0.021_991_148_575_128_55),
            Complex64::new(0.045, 0.043_982_297_150_257_1),
        ]);
        let energy_count = energies.len();
        let angular_count = 2;
        let radial_count = 6;
        let potential_count = 2;
        let phase_shifts =
            Array3::from_shape_fn((energy_count, angular_count, potential_count), |index| {
                table_complex3(index, 0.012, -0.007)
            });
        let regular_large = wavefunction_component(energy_count, angular_count, radial_count, 0.0);
        let irregular_large =
            wavefunction_component(energy_count, angular_count, radial_count, 0.35);
        let regular_small = wavefunction_component(energy_count, angular_count, radial_count, 0.7);
        let irregular_small =
            wavefunction_component(energy_count, angular_count, radial_count, 1.05);
        let diagonal_scattering = Array4::from_shape_fn(
            (
                energy_count,
                2,
                angular_count * angular_count,
                angular_count * angular_count,
            ),
            |index| table_complex4(index, 0.002, -0.0015),
        );

        TableSource {
            atom_positions_bohr: ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            atom_potentials: vec![0, 1],
            reference_energy_hartree: Complex64::new(0.03, -0.01),
            wavefunctions: RhorrpWavefunctionTables {
                setups_by_potential: vec![Vec::new(); potential_count],
                wave_numbers: Array2::zeros((energy_count, potential_count)),
                phase_shifts,
                regular_large,
                irregular_large,
                regular_small,
                irregular_small,
                regular_iteration_count: 0,
                irregular_iteration_count: 0,
                difficult_iterations: 0,
            },
            diagonal_scattering,
            energies_hartree: energies,
            radial_x0: 0.7,
            radial_dx: 0.2,
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
        }
    }

    fn wavefunction_component(
        energy_count: usize,
        angular_count: usize,
        radial_count: usize,
        offset: f64,
    ) -> Array4<Complex64> {
        Array4::from_shape_fn(
            (energy_count, angular_count, radial_count, 2),
            |(energy, angular, radial, potential)| {
                let re = 0.10 * (energy + 1) as f64
                    + 0.03 * angular as f64
                    + 0.01 * (radial + 1) as f64
                    + 0.05 * potential as f64
                    + offset;
                let im = -0.06 * (energy + 1) as f64 + 0.02 * angular as f64
                    - 0.015 * (radial + 1) as f64
                    + 0.025 * potential as f64
                    - 0.5 * offset;
                Complex64::new(re, im)
            },
        )
    }

    fn table_complex3(
        (energy, first, second): (usize, usize, usize),
        re_scale: f64,
        im_scale: f64,
    ) -> Complex64 {
        table_complex4((energy, first, second, 0), re_scale, im_scale)
    }

    fn table_complex4(
        (energy, first, second, third): (usize, usize, usize, usize),
        re_scale: f64,
        im_scale: f64,
    ) -> Complex64 {
        let weighted = (energy + 1) as f64 + 0.5 * (first + 1) as f64 - 0.25 * (second + 1) as f64
            + 0.125 * (third + 1) as f64;
        Complex64::new(re_scale * weighted, im_scale * weighted)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}
