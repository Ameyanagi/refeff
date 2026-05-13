//! FEFF RHORRP grid and atom-localization helpers.
//!
//! The full `RHORRP/m_rhorrp.f90` density-matrix calculation depends on the
//! potential, phase, and FMS handoff data. This module starts with the compact
//! support routines used by that calculation and by `RHORRP/rhorrp.f90` output:
//! FEFF-order density-grid traversal and nearest-atom selection.

use ndarray::{Array2, ArrayView2, ShapeBuilder};
use thiserror::Error;

use crate::{Real, RealMat, Vector3};

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
        if !value.is_finite() {
            return Err(RhorrpError::NonFiniteValue { name, index, value });
        }
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
    use ndarray::arr2;

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
}
