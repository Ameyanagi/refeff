//! FEFF RHORRP grid and atom-localization helpers.
//!
//! The full `RHORRP/m_rhorrp.f90` density-matrix calculation depends on the
//! potential, phase, and FMS handoff data. This module starts with the compact
//! support routines used by that calculation and by `RHORRP/rhorrp.f90` output:
//! FEFF-order density-grid traversal, nearest-atom selection, radial
//! wavefunction interpolation, and contour Fermi occupations.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3, Axis, ShapeBuilder, Slice};
use refeff_linalg::{LinalgError, complex_polyfit, complex_polyval};
use thiserror::Error;

use crate::{Complex, ComplexMat, ComplexVec, Real, RealMat, Vector3};

/// Input for FEFF `point_at_index` density-grid traversal.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpDensityGridInput<'a> {
    /// Grid origin in Bohr, FEFF `grid%origin`.
    pub origin: Vector3,
    /// Grid axes in Bohr with FEFF shape `(xyz, dimension)`.
    pub axes: ArrayView2<'a, Real>,
    /// Number of points along each active axis, FEFF `grid%npts`.
    pub points_per_axis: &'a [usize],
}

/// FEFF-order density-grid point table.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpDensityGridPoints {
    /// Points as `(xyz, point)` in Fortran-order storage, matching FEFF
    /// `points(3, totpts)`.
    pub points: RealMat,
}

/// Input for FEFF `nearest_atom`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpNearestAtomInput<'a> {
    /// Cartesian point in Bohr.
    pub point: Vector3,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// If set, only this many leading atoms are considered, matching FEFF's
    /// `fmsF` branch that loops over `inclus(0)`.
    pub fms_atom_count: Option<usize>,
}

/// Result of FEFF `nearest_atom`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpNearestAtom {
    /// Zero-based atom index for Rust callers.
    pub atom_index: usize,
    /// FEFF one-based atom index.
    pub atom_index_1based: usize,
    /// Potential index associated with the selected atom.
    pub potential_index: usize,
    /// Displacement `point - atom_position`.
    pub displacement: Vector3,
    /// Squared distance to the selected atom.
    pub squared_distance: Real,
}

/// Input for FEFF `interpwf` radial wavefunction interpolation.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpWavefunctionInterpolationInput<'a> {
    /// Wavefunction table as `(energy, angular_momentum, radial)`, matching
    /// FEFF `wf(ne, 0:lx, nr)`.
    pub wavefunctions: ArrayView3<'a, Complex>,
    /// FEFF one-based lower radial index `i`. Negative values return zero and
    /// zero selects the FEFF `wf(:,:,i+1) * f` branch.
    pub index_below_1based: isize,
    /// Fractional distance between the lower and upper radial samples.
    pub fraction: Real,
}

/// Input for FEFF `fermi_dist`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpFermiDistributionInput {
    /// Complex energy in Hartree.
    pub energy_hartree: Complex,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to
    /// Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
}

/// Input for FEFF `fix_irreg` irregular-solution smoothing.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpIrregularFixInput<'a> {
    /// Radial grid `ri`.
    pub radii: &'a [Real],
    /// Irregular solution samples `y0`.
    pub values: ArrayView1<'a, Complex>,
}

/// Error returned by RHORRP support helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum RhorrpError {
    /// FEFF density grids only support line, plane, and volume commands.
    #[error("RHORRP dimension count must be in 1..=3, got {dimensions}")]
    InvalidDimensionCount { dimensions: usize },
    /// Axis tables must have FEFF shape `(3, dimensions)`.
    #[error("RHORRP axes must have shape (3, {expected_columns}), got ({rows}, {columns})")]
    InvalidAxesShape {
        rows: usize,
        columns: usize,
        expected_columns: usize,
    },
    /// Atom coordinate tables must have shape `(atoms, 3)`.
    #[error("RHORRP atom positions must have shape (atoms, 3), got ({rows}, {columns})")]
    InvalidAtomPositionShape { rows: usize, columns: usize },
    /// Point counts must allow FEFF's `(npts - 1)` denominator.
    #[error("RHORRP points_per_axis[{axis}] must be at least 2, got {value}")]
    InvalidPointCount { axis: usize, value: usize },
    /// One-based FEFF indices must stay within their axis bounds.
    #[error("RHORRP index[{axis}]={index} is outside 1..={limit}")]
    InvalidGridIndex {
        axis: usize,
        index: usize,
        limit: usize,
    },
    /// Index vector length must match the active dimension count.
    #[error("RHORRP index length {index_len} does not match dimension count {dimensions}")]
    IndexLengthMismatch { index_len: usize, dimensions: usize },
    /// Atom-potential assignments must match atom coordinates.
    #[error("RHORRP atom potential length {potentials} does not match atom count {atoms}")]
    AtomPotentialLengthMismatch { potentials: usize, atoms: usize },
    /// At least one atom must be available for nearest-atom lookup.
    #[error("RHORRP nearest_atom requires at least one atom")]
    NoAtoms,
    /// The FEFF FMS atom limit must be in the atom table.
    #[error("RHORRP fms_atom_count must be in 1..={atoms}, got {fms_atom_count}")]
    InvalidFmsAtomCount { fms_atom_count: usize, atoms: usize },
    /// Floating-point inputs must be finite.
    #[error("RHORRP {name}[{index}] must be finite, got {value}")]
    NonFiniteValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// Wavefunction interpolation needs a non-empty `(energy, angular, radial)` table.
    #[error("RHORRP wavefunction table has invalid shape ({energy}, {angular}, {radial})")]
    InvalidWavefunctionShape {
        energy: usize,
        angular: usize,
        radial: usize,
    },
    /// FEFF wavefunction interpolation references both `i` and `i+1`.
    #[error("RHORRP wavefunction index {index} cannot interpolate radial count {radial}")]
    InvalidWavefunctionIndex { index: isize, radial: usize },
    /// `fix_irreg` requires matching radial and value vectors.
    #[error("RHORRP irregular fix length mismatch: radii={radii}, values={values}")]
    IrregularFixLengthMismatch { radii: usize, values: usize },
    /// `fix_irreg` fits points 50..=100 and replaces 1..=100.
    #[error("RHORRP irregular fix requires at least {required} points, got {points}")]
    InsufficientIrregularFixPoints { points: usize, required: usize },
    /// Polynomial fitting failed while smoothing the irregular solution.
    #[error("RHORRP irregular fix polynomial fit failed: {source}")]
    IrregularFixPolynomial {
        #[from]
        source: LinalgError,
    },
    /// Total point count overflowed `usize`.
    #[error("RHORRP density-grid point count overflows usize")]
    PointCountOverflow,
}

/// Generate all density-grid points in FEFF traversal order.
///
/// FEFF increments the first active index fastest, then wraps through later
/// dimensions. The output matrix stores each point as a column, matching
/// `points(3, totpts)` in `RHORRP/rhorrp.f90`.
pub fn rhorrp_density_grid_points(
    input: RhorrpDensityGridInput<'_>,
) -> Result<RhorrpDensityGridPoints, RhorrpError> {
    validate_density_grid_input(input)?;
    let total_points = checked_total_points(input.points_per_axis)?;
    let mut points = Array2::zeros((3, total_points).f());
    let mut index = vec![1; input.points_per_axis.len()];

    for point_index in 0..total_points {
        let point = point_at_index_unchecked(input, &index);
        for axis in 0..3 {
            points[(axis, point_index)] = point[axis];
        }
        next_index_unchecked(input.points_per_axis, &mut index);
    }

    Ok(RhorrpDensityGridPoints { points })
}

/// Port of FEFF `point_at_index` for a one-based grid index.
pub fn rhorrp_point_at_index(
    input: RhorrpDensityGridInput<'_>,
    index_1based: &[usize],
) -> Result<Vector3, RhorrpError> {
    validate_density_grid_input(input)?;
    validate_grid_index(input.points_per_axis, index_1based)?;

    Ok(point_at_index_unchecked(input, index_1based))
}

fn point_at_index_unchecked(input: RhorrpDensityGridInput<'_>, index_1based: &[usize]) -> Vector3 {
    let mut point = input.origin;
    for (dimension, (&count, &index)) in input
        .points_per_axis
        .iter()
        .zip(index_1based.iter())
        .enumerate()
    {
        let fraction = (index as Real - 1.0) / (count as Real - 1.0);
        for (axis, coordinate) in point.iter_mut().enumerate() {
            *coordinate += input.axes[(axis, dimension)] * fraction;
        }
    }
    point
}

/// Port of FEFF `next_index` for one-based density-grid indices.
///
/// Calling this on the final index wraps back to all ones, matching the FEFF
/// routine used after every generated point.
pub fn rhorrp_next_index_1based(
    points_per_axis: &[usize],
    index_1based: &mut [usize],
) -> Result<(), RhorrpError> {
    validate_dimension_count(points_per_axis.len())?;
    validate_point_counts(points_per_axis)?;
    validate_grid_index(points_per_axis, index_1based)?;

    for axis in 0..points_per_axis.len() {
        if index_1based[axis] < points_per_axis[axis] {
            index_1based[axis] += 1;
            return Ok(());
        }
        index_1based[axis] = 1;
    }
    Ok(())
}

fn next_index_unchecked(points_per_axis: &[usize], index_1based: &mut [usize]) {
    for axis in 0..points_per_axis.len() {
        if index_1based[axis] < points_per_axis[axis] {
            index_1based[axis] += 1;
            return;
        }
        index_1based[axis] = 1;
    }
}

/// Port of FEFF `nearest_atom`.
///
/// When `fms_atom_count` is set, only the leading `fms_atom_count` atoms are
/// considered, matching the `fmsF` branch in FEFF. Ties keep the first atom,
/// because the FEFF routine updates only on strictly smaller distance.
pub fn rhorrp_nearest_atom(
    input: RhorrpNearestAtomInput<'_>,
) -> Result<RhorrpNearestAtom, RhorrpError> {
    validate_nearest_atom_input(input)?;

    let atoms_to_search = input.fms_atom_count.unwrap_or(input.atom_positions.nrows());
    let mut best: Option<RhorrpNearestAtom> = None;
    for atom_index in 0..atoms_to_search {
        let displacement = [
            input.point[0] - input.atom_positions[(atom_index, 0)],
            input.point[1] - input.atom_positions[(atom_index, 1)],
            input.point[2] - input.atom_positions[(atom_index, 2)],
        ];
        let squared_distance = displacement.iter().map(|value| value * value).sum();
        let candidate = RhorrpNearestAtom {
            atom_index,
            atom_index_1based: atom_index + 1,
            potential_index: input.atom_potentials[atom_index],
            displacement,
            squared_distance,
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate.squared_distance < current.squared_distance)
        {
            best = Some(candidate);
        }
    }

    best.ok_or(RhorrpError::NoAtoms)
}

/// Port of FEFF `interpwf`.
///
/// The index is FEFF's one-based lower radial index. Negative indices return a
/// zero matrix, `0` returns `wf(:,:,1) * fraction`, and positive indices linearly
/// blend FEFF radial samples `i` and `i+1`.
pub fn rhorrp_interpolate_wavefunction(
    input: RhorrpWavefunctionInterpolationInput<'_>,
) -> Result<ComplexMat, RhorrpError> {
    validate_wavefunction_interpolation_input(input)?;

    let (energy_count, angular_count, radial_count) = input.wavefunctions.dim();
    let mut output = Array2::zeros((energy_count, angular_count).f());
    if input.index_below_1based < 0 {
        return Ok(output);
    }

    if input.index_below_1based == 0 {
        for energy in 0..energy_count {
            for angular in 0..angular_count {
                output[(energy, angular)] =
                    input.wavefunctions[(energy, angular, 0)] * input.fraction;
            }
        }
        return Ok(output);
    }

    let lower = usize::try_from(input.index_below_1based - 1).map_err(|_| {
        RhorrpError::InvalidWavefunctionIndex {
            index: input.index_below_1based,
            radial: radial_count,
        }
    })?;
    let upper = lower + 1;
    if upper >= radial_count {
        return Err(RhorrpError::InvalidWavefunctionIndex {
            index: input.index_below_1based,
            radial: radial_count,
        });
    }

    let lower_weight = 1.0 - input.fraction;
    for energy in 0..energy_count {
        for angular in 0..angular_count {
            output[(energy, angular)] = input.wavefunctions[(energy, angular, lower)]
                * lower_weight
                + input.wavefunctions[(energy, angular, upper)] * input.fraction;
        }
    }
    Ok(output)
}

/// Port of FEFF `fermi_dist`.
///
/// FEFF uses the override chemical potential when COMPTON asks for one, applies
/// a step function for temperatures below `1e-5` Hartree, and otherwise returns
/// `1 / (exp((E - mu) / T) + 1)` for complex contour energies.
pub fn rhorrp_fermi_distribution(
    input: RhorrpFermiDistributionInput,
) -> Result<Complex, RhorrpError> {
    validate_scalar("energy_hartree.real", 0, input.energy_hartree.re)?;
    validate_scalar("energy_hartree.imag", 0, input.energy_hartree.im)?;
    validate_scalar(
        "chemical_potential_hartree",
        0,
        input.chemical_potential_hartree,
    )?;
    validate_scalar("temperature_hartree", 0, input.temperature_hartree)?;

    let mu = if let Some(override_mu) = input.chemical_potential_override_hartree {
        validate_scalar("chemical_potential_override_hartree", 0, override_mu)?;
        override_mu
    } else {
        input.chemical_potential_hartree
    };

    let value = if input.temperature_hartree < 1.0e-5 {
        if input.energy_hartree.re < mu {
            Complex::new(1.0, 0.0)
        } else {
            Complex::new(0.0, 0.0)
        }
    } else {
        let exponent = (input.energy_hartree - Complex::new(mu, 0.0)) / input.temperature_hartree;
        Complex::new(1.0, 0.0) / (exponent.exp() + Complex::new(1.0, 0.0))
    };

    validate_scalar("fermi_distribution.real", 0, value.re)?;
    validate_scalar("fermi_distribution.imag", 0, value.im)?;
    Ok(value)
}

/// Port of FEFF `fix_irreg`.
///
/// FEFF fits a cubic polynomial to radial samples `50:100` and replaces samples
/// `1:100` with the polynomial evaluation. The tail after sample 100 is left
/// unchanged. This function returns the updated vector instead of mutating the
/// caller's data.
pub fn rhorrp_fix_irregular_origin(
    input: RhorrpIrregularFixInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    validate_irregular_fix_input(input)?;

    let coefficients = complex_polyfit(
        &input.radii[49..100],
        input.values.slice_axis(Axis(0), Slice::from(49..100)),
        3,
    )?;
    let smoothed = complex_polyval(coefficients.view(), &input.radii[..100]);
    let mut output = input.values.to_owned();
    output
        .slice_axis_mut(Axis(0), Slice::from(..100))
        .assign(&smoothed);
    Ok(output)
}

fn validate_density_grid_input(input: RhorrpDensityGridInput<'_>) -> Result<(), RhorrpError> {
    validate_dimension_count(input.points_per_axis.len())?;
    let (rows, columns) = input.axes.dim();
    if rows != 3 || columns != input.points_per_axis.len() {
        return Err(RhorrpError::InvalidAxesShape {
            rows,
            columns,
            expected_columns: input.points_per_axis.len(),
        });
    }
    validate_point_counts(input.points_per_axis)?;
    validate_vector("origin", input.origin)?;
    for (index, &value) in input.axes.iter().enumerate() {
        if !value.is_finite() {
            return Err(RhorrpError::NonFiniteValue {
                name: "axes",
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_nearest_atom_input(input: RhorrpNearestAtomInput<'_>) -> Result<(), RhorrpError> {
    validate_vector("point", input.point)?;
    let (rows, columns) = input.atom_positions.dim();
    if columns != 3 {
        return Err(RhorrpError::InvalidAtomPositionShape { rows, columns });
    }
    if rows == 0 {
        return Err(RhorrpError::NoAtoms);
    }
    if input.atom_potentials.len() != rows {
        return Err(RhorrpError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            atoms: rows,
        });
    }
    if let Some(fms_atom_count) = input.fms_atom_count
        && (fms_atom_count == 0 || fms_atom_count > rows)
    {
        return Err(RhorrpError::InvalidFmsAtomCount {
            fms_atom_count,
            atoms: rows,
        });
    }
    for (index, &value) in input.atom_positions.iter().enumerate() {
        if !value.is_finite() {
            return Err(RhorrpError::NonFiniteValue {
                name: "atom_positions",
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_vector(name: &'static str, vector: Vector3) -> Result<(), RhorrpError> {
    for (index, value) in vector.into_iter().enumerate() {
        validate_scalar(name, index, value)?;
    }
    Ok(())
}

fn validate_scalar(name: &'static str, index: usize, value: Real) -> Result<(), RhorrpError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RhorrpError::NonFiniteValue { name, index, value })
    }
}

fn validate_wavefunction_interpolation_input(
    input: RhorrpWavefunctionInterpolationInput<'_>,
) -> Result<(), RhorrpError> {
    let (energy, angular, radial) = input.wavefunctions.dim();
    if energy == 0 || angular == 0 || radial == 0 {
        return Err(RhorrpError::InvalidWavefunctionShape {
            energy,
            angular,
            radial,
        });
    }
    validate_scalar("wavefunction_fraction", 0, input.fraction)?;
    if input.index_below_1based >= 0 {
        let upper = if input.index_below_1based == 0 {
            0
        } else {
            usize::try_from(input.index_below_1based).map_err(|_| {
                RhorrpError::InvalidWavefunctionIndex {
                    index: input.index_below_1based,
                    radial,
                }
            })?
        };
        if upper >= radial {
            return Err(RhorrpError::InvalidWavefunctionIndex {
                index: input.index_below_1based,
                radial,
            });
        }
    }
    Ok(())
}

fn validate_irregular_fix_input(input: RhorrpIrregularFixInput<'_>) -> Result<(), RhorrpError> {
    if input.radii.len() != input.values.len() {
        return Err(RhorrpError::IrregularFixLengthMismatch {
            radii: input.radii.len(),
            values: input.values.len(),
        });
    }
    if input.radii.len() < 100 {
        return Err(RhorrpError::InsufficientIrregularFixPoints {
            points: input.radii.len(),
            required: 100,
        });
    }
    for (index, &radius) in input.radii.iter().enumerate() {
        validate_scalar("irregular_radii", index, radius)?;
    }
    for (index, value) in input.values.iter().enumerate() {
        validate_scalar("irregular_values.real", index, value.re)?;
        validate_scalar("irregular_values.imag", index, value.im)?;
    }
    Ok(())
}

fn validate_dimension_count(dimensions: usize) -> Result<(), RhorrpError> {
    if !(1..=3).contains(&dimensions) {
        return Err(RhorrpError::InvalidDimensionCount { dimensions });
    }
    Ok(())
}

fn validate_point_counts(points_per_axis: &[usize]) -> Result<(), RhorrpError> {
    for (axis, &value) in points_per_axis.iter().enumerate() {
        if value < 2 {
            return Err(RhorrpError::InvalidPointCount { axis, value });
        }
    }
    Ok(())
}

fn validate_grid_index(
    points_per_axis: &[usize],
    index_1based: &[usize],
) -> Result<(), RhorrpError> {
    if index_1based.len() != points_per_axis.len() {
        return Err(RhorrpError::IndexLengthMismatch {
            index_len: index_1based.len(),
            dimensions: points_per_axis.len(),
        });
    }
    for (axis, (&index, &limit)) in index_1based.iter().zip(points_per_axis.iter()).enumerate() {
        if index == 0 || index > limit {
            return Err(RhorrpError::InvalidGridIndex { axis, index, limit });
        }
    }
    Ok(())
}

fn checked_total_points(points_per_axis: &[usize]) -> Result<usize, RhorrpError> {
    points_per_axis
        .iter()
        .try_fold(1usize, |total, &count| total.checked_mul(count))
        .ok_or(RhorrpError::PointCountOverflow)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array3, arr2};

    use super::*;

    #[test]
    fn density_grid_points_match_feff_reference() -> Result<(), RhorrpError> {
        let axes = reference_axes();
        let input = reference_grid_input(&axes);
        let points = rhorrp_density_grid_points(input)?;

        assert_eq!(points.points.dim(), (3, 24));
        assert_vector_close(column(&points.points, 0), [0.1, -0.2, 0.3]);
        assert_vector_close(column(&points.points, 1), [0.7, -0.4, 0.4]);
        assert_vector_close(column(&points.points, 3), [-0.2, 0.7, 0.8]);
        assert_vector_close(
            column(&points.points, 6),
            [0.233333333333333, -0.166666666666667, 0.666666666666667],
        );
        assert_vector_close(column(&points.points, 23), [1.4, 0.4, 2.1]);
        Ok(())
    }

    #[test]
    fn point_and_next_index_match_feff_order() -> Result<(), RhorrpError> {
        let axes = reference_axes();
        let input = reference_grid_input(&axes);
        let mut index = vec![1, 1, 1];
        assert_vector_close(rhorrp_point_at_index(input, &index)?, [0.1, -0.2, 0.3]);
        rhorrp_next_index_1based(&[3, 2, 4], &mut index)?;
        assert_eq!(index, vec![2, 1, 1]);
        rhorrp_next_index_1based(&[3, 2, 4], &mut index)?;
        assert_eq!(index, vec![3, 1, 1]);
        rhorrp_next_index_1based(&[3, 2, 4], &mut index)?;
        assert_eq!(index, vec![1, 2, 1]);
        Ok(())
    }

    #[test]
    fn nearest_atom_matches_feff_reference() -> Result<(), RhorrpError> {
        let positions = reference_positions();
        let potentials = [0, 2, 1, 3];
        let first = rhorrp_nearest_atom(RhorrpNearestAtomInput {
            point: [0.7, 0.2, 0.1],
            atom_positions: positions.view(),
            atom_potentials: &potentials,
            fms_atom_count: Some(3),
        })?;
        assert_eq!(first.atom_index_1based, 2);
        assert_eq!(first.potential_index, 2);
        assert_vector_close(first.displacement, [-0.3, 0.2, 0.1]);

        let z_limited = rhorrp_nearest_atom(RhorrpNearestAtomInput {
            point: [0.0, 0.1, 0.8],
            atom_positions: positions.view(),
            atom_potentials: &potentials,
            fms_atom_count: Some(3),
        })?;
        assert_eq!(z_limited.atom_index_1based, 1);
        assert_eq!(z_limited.potential_index, 0);
        assert_vector_close(z_limited.displacement, [0.0, 0.1, 0.8]);

        let z_all = rhorrp_nearest_atom(RhorrpNearestAtomInput {
            point: [0.0, 0.1, 0.8],
            atom_positions: positions.view(),
            atom_potentials: &potentials,
            fms_atom_count: None,
        })?;
        assert_eq!(z_all.atom_index_1based, 4);
        assert_eq!(z_all.potential_index, 3);
        assert_vector_close(z_all.displacement, [0.0, 0.1, -0.2]);
        Ok(())
    }

    #[test]
    fn rhorrp_helpers_reject_invalid_inputs() {
        let axes = arr2(&[[1.0], [0.0], [0.0]]);
        assert!(matches!(
            rhorrp_density_grid_points(RhorrpDensityGridInput {
                origin: [0.0; 3],
                axes: axes.view(),
                points_per_axis: &[1],
            }),
            Err(RhorrpError::InvalidPointCount { axis: 0, value: 1 })
        ));
        assert!(matches!(
            rhorrp_point_at_index(
                RhorrpDensityGridInput {
                    origin: [0.0; 3],
                    axes: axes.view(),
                    points_per_axis: &[2],
                },
                &[3],
            ),
            Err(RhorrpError::InvalidGridIndex {
                axis: 0,
                index: 3,
                limit: 2,
            })
        ));

        let positions = reference_positions();
        assert!(matches!(
            rhorrp_nearest_atom(RhorrpNearestAtomInput {
                point: [0.0; 3],
                atom_positions: positions.view(),
                atom_potentials: &[0, 1],
                fms_atom_count: None,
            }),
            Err(RhorrpError::AtomPotentialLengthMismatch {
                potentials: 2,
                atoms: 4,
            })
        ));
        assert!(matches!(
            rhorrp_nearest_atom(RhorrpNearestAtomInput {
                point: [0.0; 3],
                atom_positions: positions.view(),
                atom_potentials: &[0, 1, 2, 3],
                fms_atom_count: Some(5),
            }),
            Err(RhorrpError::InvalidFmsAtomCount {
                fms_atom_count: 5,
                atoms: 4,
            })
        ));
    }

    #[test]
    fn wavefunction_interpolation_matches_feff_reference() -> Result<(), RhorrpError> {
        let wavefunctions = reference_wavefunctions();

        let negative = rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
            wavefunctions: wavefunctions.view(),
            index_below_1based: -1,
            fraction: 0.4,
        })?;
        assert_complex_close(negative[(1, 1)], Complex::new(0.0, 0.0));

        let zero = rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
            wavefunctions: wavefunctions.view(),
            index_below_1based: 0,
            fraction: 0.4,
        })?;
        assert_complex_close(zero[(1, 1)], Complex::new(4.48, -2.06));

        let two = rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
            wavefunctions: wavefunctions.view(),
            index_below_1based: 2,
            fraction: 0.35,
        })?;
        assert_complex_close(two[(0, 0)], Complex::new(23.6, -11.95));
        assert_complex_close(two[(2, 2)], Complex::new(25.799999999999997, -11.85));
        Ok(())
    }

    #[test]
    fn fermi_distribution_matches_feff_reference() -> Result<(), RhorrpError> {
        let complex = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
            energy_hartree: Complex::new(0.2, 0.05),
            chemical_potential_hartree: 0.1,
            temperature_hartree: 0.025,
            chemical_potential_override_hartree: None,
        })?;
        assert_complex_close(
            complex,
            Complex::new(-7.396_808_073_316_784e-3, -1.690_641_303_994_834_5e-2),
        );

        let override_mu = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
            energy_hartree: Complex::new(0.2, 0.05),
            chemical_potential_hartree: 0.1,
            temperature_hartree: 0.025,
            chemical_potential_override_hartree: Some(0.22),
        })?;
        assert_complex_close(
            override_mu,
            Complex::new(9.819_914_491_359_244e-1, -4.934_924_358_596_282_6e-1),
        );

        let zero_low = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
            energy_hartree: Complex::new(0.05, 0.0),
            chemical_potential_hartree: 0.1,
            temperature_hartree: 1.0e-6,
            chemical_potential_override_hartree: None,
        })?;
        assert_complex_close(zero_low, Complex::new(1.0, 0.0));

        let zero_high = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
            energy_hartree: Complex::new(0.15, 0.0),
            chemical_potential_hartree: 0.1,
            temperature_hartree: 1.0e-6,
            chemical_potential_override_hartree: None,
        })?;
        assert_complex_close(zero_high, Complex::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn wavefunction_and_fermi_helpers_reject_invalid_inputs() {
        let wavefunctions = reference_wavefunctions();
        assert!(matches!(
            rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
                wavefunctions: wavefunctions.view(),
                index_below_1based: 4,
                fraction: 0.0,
            }),
            Err(RhorrpError::InvalidWavefunctionIndex {
                index: 4,
                radial: 4,
            })
        ));

        let empty = Array3::<Complex>::zeros((1, 1, 0));
        assert!(matches!(
            rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
                wavefunctions: empty.view(),
                index_below_1based: 0,
                fraction: 0.0,
            }),
            Err(RhorrpError::InvalidWavefunctionShape {
                energy: 1,
                angular: 1,
                radial: 0,
            })
        ));

        assert!(matches!(
            rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
                energy_hartree: Complex::new(f64::NAN, 0.0),
                chemical_potential_hartree: 0.1,
                temperature_hartree: 0.025,
                chemical_potential_override_hartree: None,
            }),
            Err(RhorrpError::NonFiniteValue {
                name: "energy_hartree.real",
                ..
            })
        ));
    }

    #[test]
    fn fix_irregular_origin_matches_feff_reference() -> Result<(), RhorrpError> {
        let (radii, values) = reference_irregular_solution();
        let fixed = rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
            radii: &radii,
            values: values.view(),
        })?;

        assert_complex_close_tol(
            fixed[0],
            Complex::new(9.791_151_469_085_387, 3.741_459_448_683_99),
            1.0e-8,
        );
        assert_complex_close_tol(
            fixed[49],
            Complex::new(-2.047_179_619_930_901_1e-1, -8.434_737_680_311_137e-1),
            1.0e-8,
        );
        assert_complex_close_tol(
            fixed[74],
            Complex::new(-6.916_158_567_064_077e-1, -8.929_639_586_361_882e-1),
            1.0e-8,
        );
        assert_complex_close_tol(
            fixed[99],
            Complex::new(8.811_645_823_831e-1, 1.866_102_289_679_183_5e-1),
            1.0e-8,
        );
        assert_complex_close_tol(
            fixed[100],
            Complex::new(9.101_077_089_878_837e-1, 2.302_339_202_367_545e-1),
            1.0e-8,
        );
        assert_complex_close_tol(
            fixed[119],
            Complex::new(1.094_598_908_088_280_5, 8.401_702_866_503_66e-1),
            1.0e-8,
        );
        Ok(())
    }

    #[test]
    fn fix_irregular_origin_rejects_invalid_inputs() {
        let (radii, values) = reference_irregular_solution();
        assert!(matches!(
            rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
                radii: &radii[..99],
                values: values.slice_axis(Axis(0), Slice::from(..99)),
            }),
            Err(RhorrpError::InsufficientIrregularFixPoints {
                points: 99,
                required: 100,
            })
        ));

        assert!(matches!(
            rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
                radii: &radii[..100],
                values: values.view(),
            }),
            Err(RhorrpError::IrregularFixLengthMismatch {
                radii: 100,
                values: 120,
            })
        ));
    }

    fn reference_grid_input<'a>(axes: &'a Array2<Real>) -> RhorrpDensityGridInput<'a> {
        RhorrpDensityGridInput {
            origin: [0.1, -0.2, 0.3],
            axes: axes.view(),
            points_per_axis: &[3, 2, 4],
        }
    }

    fn reference_axes() -> Array2<Real> {
        arr2(&[[1.2, -0.3, 0.4], [-0.4, 0.9, 0.1], [0.2, 0.5, 1.1]])
    }

    fn reference_positions() -> Array2<Real> {
        arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }

    fn reference_wavefunctions() -> Array3<Complex> {
        Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(10.0 * ir + il + 0.1 * ie, -5.0 * ir + 0.25 * il - 0.2 * ie)
        })
    }

    fn reference_irregular_solution() -> (Vec<Real>, ComplexVec) {
        let radii = (1..=120)
            .map(|index| {
                let index = index as Real;
                0.02 * index + 0.0001 * index * index
            })
            .collect::<Vec<_>>();
        let values = ComplexVec::from_shape_fn(120, |index| {
            let one_based = (index + 1) as Real;
            Complex::new(
                (0.07 * one_based).sin() + 0.002 * one_based,
                (0.05 * one_based).cos() - 0.001 * one_based,
            )
        });
        (radii, values)
    }

    fn column(points: &RealMat, index: usize) -> Vector3 {
        [points[(0, index)], points[(1, index)], points[(2, index)]]
    }

    fn assert_vector_close(actual: Vector3, expected: Vector3) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}",
                (actual - expected).abs()
            );
        }
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert_complex_close_tol(actual, expected, 1.0e-12);
    }

    fn assert_complex_close_tol(actual: Complex, expected: Complex, tolerance: Real) {
        assert!(
            (actual.re - expected.re).abs() < tolerance,
            "real actual={:.17e}, expected={:.17e}, diff={:.17e}",
            actual.re,
            expected.re,
            (actual.re - expected.re).abs()
        );
        assert!(
            (actual.im - expected.im).abs() < tolerance,
            "imag actual={:.17e}, expected={:.17e}, diff={:.17e}",
            actual.im,
            expected.im,
            (actual.im - expected.im).abs()
        );
    }
}
