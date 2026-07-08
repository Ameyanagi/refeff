use ndarray::{Array2, ShapeBuilder};

use crate::{Real, RealVec, Vector3};

use super::density::rhorrp_point_density_from_tables;
use super::types::{
    RhorrpDensityGridEvaluation, RhorrpDensityGridFromTablesInput, RhorrpDensityGridInput,
    RhorrpDensityGridPoints, RhorrpError, RhorrpPointDensityFromTablesInput, RhorrpProcessRange,
};
use super::validation::{
    checked_total_points, validate_density_grid_input, validate_dimension_count,
    validate_grid_index, validate_point_counts,
};

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

/// Evaluate a density callback at every density-grid point in FEFF order.
///
/// The returned points are in Bohr and the returned density values are expected
/// to be in inverse cubic Bohr, matching FEFF's internal RHORRP units before
/// any text or binary density-output conversion.
pub fn rhorrp_evaluate_density_grid<F>(
    input: RhorrpDensityGridInput<'_>,
    mut density_at: F,
) -> Result<RhorrpDensityGridEvaluation, RhorrpError>
where
    F: FnMut(Vector3) -> Result<Real, RhorrpError>,
{
    validate_density_grid_input(input)?;
    let total_points = checked_total_points(input.points_per_axis)?;
    let mut points = Array2::zeros((3, total_points).f());
    let mut density = Vec::with_capacity(total_points);
    let mut index = vec![1; input.points_per_axis.len()];

    for point_index in 0..total_points {
        let point = point_at_index_unchecked(input, &index);
        for axis in 0..3 {
            points[(axis, point_index)] = point[axis];
        }

        let value = density_at(point)?;
        if !value.is_finite() {
            return Err(RhorrpError::NonFiniteDensityValue {
                point: point_index,
                value,
            });
        }
        density.push(value);

        next_index_unchecked(input.points_per_axis, &mut index);
    }

    Ok(RhorrpDensityGridEvaluation {
        points,
        density_per_bohr3: RealVec::from_vec(density),
    })
}

/// Evaluate FEFF `calculate_density` for a non-core grid from wavefunction tables.
///
/// FEFF evaluates `rhorrp(point, point, rho)` at each generated grid point.
/// This helper preserves the same grid traversal and delegates each point to
/// the table-backed RHORRP same-point density path.
pub fn rhorrp_evaluate_density_grid_from_tables(
    input: RhorrpDensityGridFromTablesInput<'_>,
) -> Result<RhorrpDensityGridEvaluation, RhorrpError> {
    rhorrp_evaluate_density_grid(input.grid, |point| {
        rhorrp_point_density_from_tables(RhorrpPointDensityFromTablesInput {
            point,
            atom_positions: input.atom_positions,
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
    })
}

/// Port of FEFF `calculate_density` process range partitioning.
///
/// FEFF divides `totpts` density-grid points over `numprocs` ranks with the
/// first `totpts % numprocs` ranks receiving one extra point. Ranges are
/// one-based and inclusive to match `proc_i1`/`proc_i2`; when there are more
/// processes than points, tail ranks receive empty ranges with `end < start`.
pub fn rhorrp_process_ranges(
    total_points: usize,
    process_count: usize,
) -> Result<Vec<RhorrpProcessRange>, RhorrpError> {
    if process_count == 0 {
        return Err(RhorrpError::InvalidProcessCount);
    }

    let points_per_process = total_points / process_count;
    let extra_points = total_points % process_count;
    let mut next_start = 1usize;
    let ranges = (0..process_count)
        .map(|process| {
            let mut end = next_start + points_per_process - 1;
            if process < extra_points {
                end += 1;
            }
            let range = RhorrpProcessRange {
                process,
                start_1based: next_start,
                end_1based: end,
            };
            if process + 1 < process_count {
                next_start = end + 1;
            }
            range
        })
        .collect();
    Ok(ranges)
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
