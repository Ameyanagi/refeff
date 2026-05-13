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

use crate::interpolation::{
    InterpolationError, locate_below, polynomial_interpolate, polynomial_interpolate_complex,
};
use crate::{Complex, ComplexMat, ComplexVec, Real, RealMat, RealVec, Vector3};

const ATOMIC_DENSITY_CUTOFF_SQUARED: Real = 4.0;
const ATOMIC_DENSITY_MIN_RADIUS: Real = 1.0e-4;
const ATOMIC_DENSITY_INTERPOLATION_ORDER: usize = 2;
const DENSITY_INTEGRATION_HORIZONTAL_EPSILON: Real = 1.0e-15;
const DENSITY_INTEGRATION_INTERPOLATION_ORDER: usize = 2;
const DENSITY_INTEGRATION_SUBDIVISIONS: usize = 10;

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

/// FEFF-order RHORRP density-grid evaluation in Bohr units.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpDensityGridEvaluation {
    /// Points as `(xyz, point)` in Fortran-order storage, matching FEFF
    /// `points(3, totpts)`.
    pub points: RealMat,
    /// Density values in inverse cubic Bohr, matching the FEFF point order.
    pub density_per_bohr3: RealVec,
}

impl RhorrpDensityGridEvaluation {
    /// Number of evaluated grid points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.density_per_bohr3.len()
    }
}

/// One FEFF density-grid work range for a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RhorrpProcessRange {
    /// Zero-based process rank.
    pub process: usize,
    /// FEFF one-based inclusive first point.
    pub start_1based: usize,
    /// FEFF one-based inclusive last point. Empty ranges have `end < start`.
    pub end_1based: usize,
}

impl RhorrpProcessRange {
    /// Number of points in this range.
    #[must_use]
    pub fn len(self) -> usize {
        self.end_1based
            .checked_sub(self.start_1based)
            .map_or(0, |delta| delta + 1)
    }

    /// Whether this range contains no points.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
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

/// Input for nearest-atom diagnostics over a FEFF-order point table.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpNearestAtomTableInput<'a> {
    /// Cartesian points in Bohr as `(xyz, point)`.
    pub points: ArrayView2<'a, Real>,
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

/// Nearest-atom diagnostics for a density-grid point table.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpNearestAtomTable {
    /// Displacement `point - atom_position` in Bohr as `(point, xyz)`.
    pub displacement_bohr: RealMat,
    /// Zero-based atom index for each point.
    pub atom_indices: Vec<usize>,
    /// FEFF one-based atom index for each point.
    pub atom_indices_1based: Vec<usize>,
    /// Potential index associated with each selected atom.
    pub potential_indices: Vec<usize>,
}

impl RhorrpNearestAtomTable {
    /// Number of point rows in this diagnostic table.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.atom_indices.len()
    }
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

/// Input for FEFF `atomic_density`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpAtomicDensityInput<'a> {
    /// Cartesian point in Bohr.
    pub point: Vector3,
    /// FEFF one-based orbital/core-wavefunction column `il`.
    pub orbital_index_1based: usize,
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: &'a [usize],
    /// FEFF radial grid `ripot`.
    pub radii: &'a [Real],
    /// Large Dirac components `dgc` as `(radial, orbital, potential)`.
    pub large_components: ArrayView3<'a, Real>,
    /// Small Dirac components `dpc` as `(radial, orbital, potential)`.
    pub small_components: ArrayView3<'a, Real>,
}

/// Input for FEFF `rhorrp` contour integration after `rhoerrp`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpDensityIntegrationInput<'a> {
    /// FEFF complex energy contour `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Energy-dependent density matrix values `rhoe`.
    pub energy_density: ArrayView1<'a, Complex>,
    /// FEFF `ne1`: number of contour points through the real-axis segment.
    pub real_axis_count: usize,
    /// Default chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Electronic temperature in Hartree.
    pub temperature_hartree: Real,
    /// Optional COMPTON chemical-potential override, already converted to
    /// Hartree.
    pub chemical_potential_override_hartree: Option<Real>,
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
    /// Point tables must have shape `(3, points)`.
    #[error("RHORRP point table must have shape (3, points), got ({rows}, {columns})")]
    InvalidPointTableShape { rows: usize, columns: usize },
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
    /// Density callbacks must produce finite values.
    #[error("RHORRP density callback returned non-finite value at point {point}: {value}")]
    NonFiniteDensityValue { point: usize, value: Real },
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
    /// Large and small component tables must have identical dimensions.
    #[error(
        "RHORRP atomic density component shape mismatch: large=({large_radial}, {large_orbital}, {large_potential}), small=({small_radial}, {small_orbital}, {small_potential})"
    )]
    AtomicDensityComponentShapeMismatch {
        large_radial: usize,
        large_orbital: usize,
        large_potential: usize,
        small_radial: usize,
        small_orbital: usize,
        small_potential: usize,
    },
    /// Component tables must have non-empty radial, orbital, and potential axes.
    #[error(
        "RHORRP atomic density {table} table has invalid shape ({radial}, {orbital}, {potential})"
    )]
    InvalidAtomicDensityShape {
        table: &'static str,
        radial: usize,
        orbital: usize,
        potential: usize,
    },
    /// The radial grid must match component-table radial length.
    #[error("RHORRP atomic density radial length mismatch: radii={radii}, components={components}")]
    AtomicDensityRadialLengthMismatch { radii: usize, components: usize },
    /// FEFF `terp` with order 2 needs three radial samples.
    #[error("RHORRP atomic density requires at least {required} radial points, got {points}")]
    InsufficientAtomicDensityRadii { points: usize, required: usize },
    /// FEFF orbital/core-wavefunction columns are one-based.
    #[error("RHORRP atomic density orbital index {orbital} is outside 1..={orbital_count}")]
    InvalidAtomicDensityOrbital {
        orbital: usize,
        orbital_count: usize,
    },
    /// Atom potential indices must point into the component-table potential axis.
    #[error(
        "RHORRP atomic density atom {atom_index_1based} potential {potential} is outside 0..={max_potential}"
    )]
    InvalidAtomicDensityPotential {
        atom_index_1based: usize,
        potential: usize,
        max_potential: usize,
    },
    /// FEFF quadratic radial interpolation failed.
    #[error("RHORRP atomic density interpolation failed: {source}")]
    AtomicDensityInterpolation {
        #[from]
        source: InterpolationError,
    },
    /// Energy and density arrays must have identical lengths.
    #[error(
        "RHORRP density integration length mismatch: energies={energies}, densities={densities}"
    )]
    DensityIntegrationLengthMismatch { energies: usize, densities: usize },
    /// FEFF needs at least two points before Matsubara poles and no more than `ne`.
    #[error(
        "RHORRP density integration real_axis_count {real_axis_count} is outside 2..={energy_count}"
    )]
    InvalidDensityIntegrationRealAxisCount {
        real_axis_count: usize,
        energy_count: usize,
    },
    /// The contour must turn from the vertical leg onto the real axis.
    #[error("RHORRP density integration did not find a horizontal contour segment")]
    MissingDensityIntegrationCorner,
    /// Quadratic interpolation on the real-axis contour needs three points.
    #[error(
        "RHORRP density integration requires at least {required} horizontal points, got {points}"
    )]
    InsufficientDensityIntegrationPoints { points: usize, required: usize },
    /// FEFF complex interpolation failed while integrating the density contour.
    #[error("RHORRP density integration interpolation failed: {source}")]
    DensityIntegrationInterpolation { source: InterpolationError },
    /// FEFF density-grid work partitioning needs at least one process.
    #[error("RHORRP process count must be positive")]
    InvalidProcessCount,
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

/// Port of FEFF `nearest_atom`.
///
/// When `fms_atom_count` is set, only the leading `fms_atom_count` atoms are
/// considered, matching the `fmsF` branch in FEFF. Ties keep the first atom,
/// because the FEFF routine updates only on strictly smaller distance.
pub fn rhorrp_nearest_atom(
    input: RhorrpNearestAtomInput<'_>,
) -> Result<RhorrpNearestAtom, RhorrpError> {
    validate_vector("point", input.point)?;
    let atoms_to_search = validate_atom_search_input(
        input.atom_positions,
        input.atom_potentials,
        input.fms_atom_count,
    )?;
    nearest_atom_unchecked(
        input.point,
        input.atom_positions,
        input.atom_potentials,
        atoms_to_search,
    )
}

/// Evaluate nearest-atom diagnostics for every point in a FEFF-order table.
///
/// The point table uses the same `(xyz, point)` layout as
/// [`rhorrp_density_grid_points`]. The returned displacement table is
/// `(point, xyz)`, matching RHORRP text diagnostic rows.
pub fn rhorrp_nearest_atom_table(
    input: RhorrpNearestAtomTableInput<'_>,
) -> Result<RhorrpNearestAtomTable, RhorrpError> {
    let atoms_to_search = validate_nearest_atom_table_input(input)?;
    let point_count = input.points.ncols();
    let mut displacement_bohr = Array2::zeros((point_count, 3));
    let mut atom_indices = Vec::with_capacity(point_count);
    let mut atom_indices_1based = Vec::with_capacity(point_count);
    let mut potential_indices = Vec::with_capacity(point_count);

    for point_index in 0..point_count {
        let point = [
            input.points[(0, point_index)],
            input.points[(1, point_index)],
            input.points[(2, point_index)],
        ];
        let nearest = nearest_atom_unchecked(
            point,
            input.atom_positions,
            input.atom_potentials,
            atoms_to_search,
        )?;
        for axis in 0..3 {
            displacement_bohr[(point_index, axis)] = nearest.displacement[axis];
        }
        atom_indices.push(nearest.atom_index);
        atom_indices_1based.push(nearest.atom_index_1based);
        potential_indices.push(nearest.potential_index);
    }

    Ok(RhorrpNearestAtomTable {
        displacement_bohr,
        atom_indices,
        atom_indices_1based,
        potential_indices,
    })
}

fn nearest_atom_unchecked(
    point: Vector3,
    atom_positions: ArrayView2<'_, Real>,
    atom_potentials: &[usize],
    atoms_to_search: usize,
) -> Result<RhorrpNearestAtom, RhorrpError> {
    let mut best: Option<RhorrpNearestAtom> = None;
    for atom_index in 0..atoms_to_search {
        let displacement = [
            point[0] - atom_positions[(atom_index, 0)],
            point[1] - atom_positions[(atom_index, 1)],
            point[2] - atom_positions[(atom_index, 2)],
        ];
        let squared_distance = displacement.iter().map(|value| value * value).sum();
        let candidate = RhorrpNearestAtom {
            atom_index,
            atom_index_1based: atom_index + 1,
            potential_index: atom_potentials[atom_index],
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

/// Port of FEFF `atomic_density`.
///
/// FEFF sums core radial densities from atoms within two Bohr of the requested
/// point. Each contributing atom uses quadratic `terp` interpolation on `ripot`
/// for the requested core-wavefunction column and returns the spherical
/// density `(p^2 + q^2) / (4*pi*r^2)`.
pub fn rhorrp_atomic_density(input: RhorrpAtomicDensityInput<'_>) -> Result<Real, RhorrpError> {
    validate_atomic_density_input(input)?;

    let orbital = input.orbital_index_1based - 1;
    let mut density = 0.0;
    for atom in 0..input.atom_positions.nrows() {
        let displacement = [
            input.atom_positions[(atom, 0)] - input.point[0],
            input.atom_positions[(atom, 1)] - input.point[1],
            input.atom_positions[(atom, 2)] - input.point[2],
        ];
        let distance_squared: Real = displacement.iter().map(|value| value * value).sum();
        if distance_squared > ATOMIC_DENSITY_CUTOFF_SQUARED {
            continue;
        }

        let radius = distance_squared.sqrt().max(ATOMIC_DENSITY_MIN_RADIUS);
        let potential = input.atom_potentials[atom];
        let large = interpolate_atomic_component(
            input.radii,
            input.large_components,
            orbital,
            potential,
            radius,
        )?;
        let small = interpolate_atomic_component(
            input.radii,
            input.small_components,
            orbital,
            potential,
            radius,
        )?;
        density += (large * large + small * small) / (4.0 * std::f64::consts::PI * radius * radius);
    }

    validate_scalar("atomic_density", 0, density)?;
    Ok(density)
}

/// Port of FEFF `rhorrp` energy-contour integration.
///
/// This helper starts after `rhoerrp` has produced the energy-dependent density
/// matrix. It preserves FEFF's vertical trapezoid leg, hard-coded ten-way
/// horizontal subdivision with quadratic `terpc`, Matsubara pole sum, and final
/// imaginary-part extraction.
pub fn rhorrp_integrate_density(
    input: RhorrpDensityIntegrationInput<'_>,
) -> Result<Real, RhorrpError> {
    validate_density_integration_input(input)?;

    let mut fermi = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
        energy_hartree: input.energies_hartree[0],
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree: input.temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })?;
    let mut integrated = input.energies_hartree[0] * input.energy_density[0] * fermi;
    let mut previous_density = input.energy_density[0];
    let mut previous_fermi = fermi;
    let mut horizontal_start = None;

    for energy_index in 1..input.real_axis_count {
        let delta = input.energies_hartree[energy_index] - input.energies_hartree[energy_index - 1];
        if delta.re > DENSITY_INTEGRATION_HORIZONTAL_EPSILON {
            horizontal_start = Some(energy_index - 1);
            break;
        }

        fermi = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
            energy_hartree: input.energies_hartree[energy_index],
            chemical_potential_hartree: input.chemical_potential_hartree,
            temperature_hartree: input.temperature_hartree,
            chemical_potential_override_hartree: input.chemical_potential_override_hartree,
        })?;
        let density = input.energy_density[energy_index];
        integrated += (previous_density * previous_fermi + density * fermi) * 0.5 * delta;
        previous_density = density;
        previous_fermi = fermi;
    }

    let horizontal_start = horizontal_start.ok_or(RhorrpError::MissingDensityIntegrationCorner)?;
    let horizontal_points = input.real_axis_count - horizontal_start;
    let required = DENSITY_INTEGRATION_INTERPOLATION_ORDER + 1;
    if horizontal_points < required {
        return Err(RhorrpError::InsufficientDensityIntegrationPoints {
            points: horizontal_points,
            required,
        });
    }

    for energy_index in (horizontal_start + 1)..input.real_axis_count {
        let delta = (input.energies_hartree[energy_index]
            - input.energies_hartree[energy_index - 1])
            / DENSITY_INTEGRATION_SUBDIVISIONS as Real;
        for subdivision in 1..=DENSITY_INTEGRATION_SUBDIVISIONS {
            let energy = input.energies_hartree[energy_index - 1] + delta * subdivision as Real;
            fermi = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
                energy_hartree: energy,
                chemical_potential_hartree: input.chemical_potential_hartree,
                temperature_hartree: input.temperature_hartree,
                chemical_potential_override_hartree: input.chemical_potential_override_hartree,
            })?;
            let density = interpolate_density_contour(
                input.energies_hartree,
                input.energy_density,
                horizontal_start,
                input.real_axis_count,
                energy.re,
            )?;
            integrated += (previous_density * previous_fermi + density * fermi) * 0.5 * delta;
            previous_density = density;
            previous_fermi = fermi;
        }
    }

    for energy_index in input.real_axis_count..input.energies_hartree.len() {
        integrated += Complex::new(0.0, -2.0 * std::f64::consts::PI * input.temperature_hartree)
            * input.energy_density[energy_index];
    }

    validate_scalar("integrated_density", 0, integrated.im)?;
    Ok(integrated.im)
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

fn validate_nearest_atom_table_input(
    input: RhorrpNearestAtomTableInput<'_>,
) -> Result<usize, RhorrpError> {
    let (rows, columns) = input.points.dim();
    if rows != 3 {
        return Err(RhorrpError::InvalidPointTableShape { rows, columns });
    }
    for (index, &value) in input.points.iter().enumerate() {
        validate_scalar("nearest_atom_points", index, value)?;
    }
    validate_atom_search_input(
        input.atom_positions,
        input.atom_potentials,
        input.fms_atom_count,
    )
}

fn validate_atom_search_input(
    atom_positions: ArrayView2<'_, Real>,
    atom_potentials: &[usize],
    fms_atom_count: Option<usize>,
) -> Result<usize, RhorrpError> {
    let (rows, columns) = atom_positions.dim();
    if columns != 3 {
        return Err(RhorrpError::InvalidAtomPositionShape { rows, columns });
    }
    if rows == 0 {
        return Err(RhorrpError::NoAtoms);
    }
    if atom_potentials.len() != rows {
        return Err(RhorrpError::AtomPotentialLengthMismatch {
            potentials: atom_potentials.len(),
            atoms: rows,
        });
    }
    if let Some(fms_atom_count) = fms_atom_count
        && (fms_atom_count == 0 || fms_atom_count > rows)
    {
        return Err(RhorrpError::InvalidFmsAtomCount {
            fms_atom_count,
            atoms: rows,
        });
    }
    for (index, &value) in atom_positions.iter().enumerate() {
        validate_scalar("atom_positions", index, value)?;
    }
    Ok(fms_atom_count.unwrap_or(rows))
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

fn validate_atomic_density_input(input: RhorrpAtomicDensityInput<'_>) -> Result<(), RhorrpError> {
    validate_vector("atomic_density_point", input.point)?;
    let (atoms, columns) = input.atom_positions.dim();
    if columns != 3 {
        return Err(RhorrpError::InvalidAtomPositionShape {
            rows: atoms,
            columns,
        });
    }
    if atoms == 0 {
        return Err(RhorrpError::NoAtoms);
    }
    if input.atom_potentials.len() != atoms {
        return Err(RhorrpError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            atoms,
        });
    }

    let large_shape = input.large_components.dim();
    let small_shape = input.small_components.dim();
    if large_shape != small_shape {
        return Err(RhorrpError::AtomicDensityComponentShapeMismatch {
            large_radial: large_shape.0,
            large_orbital: large_shape.1,
            large_potential: large_shape.2,
            small_radial: small_shape.0,
            small_orbital: small_shape.1,
            small_potential: small_shape.2,
        });
    }
    let (radial, orbital, potential_count) = large_shape;
    if radial == 0 || orbital == 0 || potential_count == 0 {
        return Err(RhorrpError::InvalidAtomicDensityShape {
            table: "component",
            radial,
            orbital,
            potential: potential_count,
        });
    }
    if input.radii.len() != radial {
        return Err(RhorrpError::AtomicDensityRadialLengthMismatch {
            radii: input.radii.len(),
            components: radial,
        });
    }
    let required = ATOMIC_DENSITY_INTERPOLATION_ORDER + 1;
    if radial < required {
        return Err(RhorrpError::InsufficientAtomicDensityRadii {
            points: radial,
            required,
        });
    }
    if input.orbital_index_1based == 0 || input.orbital_index_1based > orbital {
        return Err(RhorrpError::InvalidAtomicDensityOrbital {
            orbital: input.orbital_index_1based,
            orbital_count: orbital,
        });
    }
    for (atom, &potential) in input.atom_potentials.iter().enumerate() {
        if potential >= potential_count {
            return Err(RhorrpError::InvalidAtomicDensityPotential {
                atom_index_1based: atom + 1,
                potential,
                max_potential: potential_count.saturating_sub(1),
            });
        }
    }
    for (index, &value) in input.atom_positions.iter().enumerate() {
        validate_scalar("atomic_density_atom_positions", index, value)?;
    }
    for (index, &radius) in input.radii.iter().enumerate() {
        validate_scalar("atomic_density_radii", index, radius)?;
    }
    for (index, &value) in input.large_components.iter().enumerate() {
        validate_scalar("atomic_density_large_components", index, value)?;
    }
    for (index, &value) in input.small_components.iter().enumerate() {
        validate_scalar("atomic_density_small_components", index, value)?;
    }
    Ok(())
}

fn validate_density_integration_input(
    input: RhorrpDensityIntegrationInput<'_>,
) -> Result<(), RhorrpError> {
    if input.energies_hartree.len() != input.energy_density.len() {
        return Err(RhorrpError::DensityIntegrationLengthMismatch {
            energies: input.energies_hartree.len(),
            densities: input.energy_density.len(),
        });
    }
    if input.real_axis_count < 2 || input.real_axis_count > input.energies_hartree.len() {
        return Err(RhorrpError::InvalidDensityIntegrationRealAxisCount {
            real_axis_count: input.real_axis_count,
            energy_count: input.energies_hartree.len(),
        });
    }
    validate_scalar(
        "density_integration_chemical_potential_hartree",
        0,
        input.chemical_potential_hartree,
    )?;
    validate_scalar(
        "density_integration_temperature_hartree",
        0,
        input.temperature_hartree,
    )?;
    if let Some(override_mu) = input.chemical_potential_override_hartree {
        validate_scalar(
            "density_integration_chemical_potential_override_hartree",
            0,
            override_mu,
        )?;
    }
    for (index, energy) in input.energies_hartree.iter().enumerate() {
        validate_scalar("density_integration_energy.real", index, energy.re)?;
        validate_scalar("density_integration_energy.imag", index, energy.im)?;
    }
    for (index, density) in input.energy_density.iter().enumerate() {
        validate_scalar("density_integration_density.real", index, density.re)?;
        validate_scalar("density_integration_density.imag", index, density.im)?;
    }
    Ok(())
}

fn interpolate_atomic_component(
    radii: &[Real],
    components: ArrayView3<'_, Real>,
    orbital: usize,
    potential: usize,
    radius: Real,
) -> Result<Real, RhorrpError> {
    let located = locate_below(radius, radii);
    let start_1based = (located.saturating_sub(ATOMIC_DENSITY_INTERPOLATION_ORDER / 2))
        .clamp(1, radii.len() - ATOMIC_DENSITY_INTERPOLATION_ORDER);
    let start = start_1based - 1;
    let values = [
        components[(start, orbital, potential)],
        components[(start + 1, orbital, potential)],
        components[(start + 2, orbital, potential)],
    ];
    Ok(polynomial_interpolate(
        &radii[start..start + ATOMIC_DENSITY_INTERPOLATION_ORDER + 1],
        &values,
        radius,
    )?
    .value)
}

fn interpolate_density_contour(
    energies: ArrayView1<'_, Complex>,
    density: ArrayView1<'_, Complex>,
    horizontal_start: usize,
    real_axis_count: usize,
    energy: Real,
) -> Result<Complex, RhorrpError> {
    let located = locate_density_contour_below(energies, horizontal_start, real_axis_count, energy);
    let local_len = real_axis_count - horizontal_start;
    let start_1based = (located.saturating_sub(DENSITY_INTEGRATION_INTERPOLATION_ORDER / 2))
        .clamp(1, local_len - DENSITY_INTEGRATION_INTERPOLATION_ORDER);
    let start = horizontal_start + start_1based - 1;
    let interpolation_energies = [
        energies[start].re,
        energies[start + 1].re,
        energies[start + 2].re,
    ];
    let values = [density[start], density[start + 1], density[start + 2]];
    Ok(
        polynomial_interpolate_complex(&interpolation_energies, &values, energy)
            .map_err(|source| RhorrpError::DensityIntegrationInterpolation { source })?
            .value,
    )
}

fn locate_density_contour_below(
    energies: ArrayView1<'_, Complex>,
    start: usize,
    end: usize,
    energy: Real,
) -> usize {
    let mut lower = 0;
    let mut upper = end - start + 1;

    while upper - lower > 1 {
        let middle = (upper + lower) / 2;
        let middle_value = energies[start + middle - 1].re;
        if energy < middle_value {
            upper = middle;
        } else {
            lower = middle;
        }
    }

    lower
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
    use ndarray::{Array1, Array3, arr2};

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
    fn evaluate_density_grid_matches_feff_reference() -> Result<(), RhorrpError> {
        let axes = reference_axes();
        let input = reference_grid_input(&axes);
        let evaluated = rhorrp_evaluate_density_grid(input, |point| Ok(sample_density(point)))?;

        assert_eq!(evaluated.point_count(), 24);
        assert_eq!(evaluated.points.dim(), (3, 24));
        assert_vector_close(column(&evaluated.points, 0), [0.1, -0.2, 0.3]);
        assert_real_close(evaluated.density_per_bohr3[0], -0.470_000_000_000_000_1);
        assert_vector_close(column(&evaluated.points, 1), [0.7, -0.4, 0.4]);
        assert_real_close(evaluated.density_per_bohr3[1], -0.580_000_000_000_000_1);
        assert_vector_close(column(&evaluated.points, 3), [-0.2, 0.7, 0.8]);
        assert_real_close(evaluated.density_per_bohr3[3], 0.659_999_999_999_999_9);
        assert_vector_close(
            column(&evaluated.points, 6),
            [0.233333333333333, -0.166666666666667, 0.666666666666667],
        );
        assert_real_close(evaluated.density_per_bohr3[6], -0.472_222_222_222_222_27);
        assert_vector_close(column(&evaluated.points, 23), [1.4, 0.4, 2.1]);
        assert_real_close(evaluated.density_per_bohr3[23], 1.709_999_999_999_999_5);
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
    fn process_ranges_match_feff_reference() -> Result<(), RhorrpError> {
        assert_eq!(
            rhorrp_process_ranges(10, 3)?,
            vec![
                RhorrpProcessRange {
                    process: 0,
                    start_1based: 1,
                    end_1based: 4,
                },
                RhorrpProcessRange {
                    process: 1,
                    start_1based: 5,
                    end_1based: 7,
                },
                RhorrpProcessRange {
                    process: 2,
                    start_1based: 8,
                    end_1based: 10,
                },
            ]
        );
        assert_eq!(
            rhorrp_process_ranges(3, 5)?,
            vec![
                RhorrpProcessRange {
                    process: 0,
                    start_1based: 1,
                    end_1based: 1,
                },
                RhorrpProcessRange {
                    process: 1,
                    start_1based: 2,
                    end_1based: 2,
                },
                RhorrpProcessRange {
                    process: 2,
                    start_1based: 3,
                    end_1based: 3,
                },
                RhorrpProcessRange {
                    process: 3,
                    start_1based: 4,
                    end_1based: 3,
                },
                RhorrpProcessRange {
                    process: 4,
                    start_1based: 4,
                    end_1based: 3,
                },
            ]
        );
        assert_eq!(rhorrp_process_ranges(24, 4)?[3].len(), 6);
        assert!(rhorrp_process_ranges(3, 5)?[3].is_empty());
        assert!(matches!(
            rhorrp_process_ranges(10, 0),
            Err(RhorrpError::InvalidProcessCount)
        ));
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
    fn nearest_atom_table_matches_feff_reference() -> Result<(), RhorrpError> {
        let positions = reference_positions();
        let potentials = [0, 2, 1, 3];
        let points = reference_nearest_points();
        let table = rhorrp_nearest_atom_table(RhorrpNearestAtomTableInput {
            points: points.view(),
            atom_positions: positions.view(),
            atom_potentials: &potentials,
            fms_atom_count: None,
        })?;

        assert_eq!(table.point_count(), 4);
        assert_vector_close(
            row(&table.displacement_bohr, 0),
            [-0.300_000_000_000_000_04, 0.2, 0.1],
        );
        assert_vector_close(row(&table.displacement_bohr, 1), [0.0, 0.1, -0.2]);
        assert_vector_close(row(&table.displacement_bohr, 2), [0.2, -0.1, 0.1]);
        assert_vector_close(row(&table.displacement_bohr, 3), [0.0, 0.5, 0.5]);
        assert_eq!(table.atom_indices, vec![1, 3, 2, 0]);
        assert_eq!(table.atom_indices_1based, vec![2, 4, 3, 1]);
        assert_eq!(table.potential_indices, vec![2, 3, 1, 0]);
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
        assert!(matches!(
            rhorrp_evaluate_density_grid(
                RhorrpDensityGridInput {
                    origin: [0.0; 3],
                    axes: axes.view(),
                    points_per_axis: &[2],
                },
                |_| Ok(f64::NAN),
            ),
            Err(RhorrpError::NonFiniteDensityValue { point: 0, .. })
        ));
        assert!(matches!(
            rhorrp_evaluate_density_grid(
                RhorrpDensityGridInput {
                    origin: [0.0; 3],
                    axes: axes.view(),
                    points_per_axis: &[2],
                },
                |_| Err(RhorrpError::InvalidProcessCount),
            ),
            Err(RhorrpError::InvalidProcessCount)
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
        let bad_points = arr2(&[[0.0, 1.0]]);
        assert!(matches!(
            rhorrp_nearest_atom_table(RhorrpNearestAtomTableInput {
                points: bad_points.view(),
                atom_positions: positions.view(),
                atom_potentials: &[0, 1, 2, 3],
                fms_atom_count: None,
            }),
            Err(RhorrpError::InvalidPointTableShape {
                rows: 1,
                columns: 2,
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

    #[test]
    fn atomic_density_matches_feff_reference() -> Result<(), RhorrpError> {
        let reference = reference_atomic_density_tables();

        assert_real_close_scaled(
            rhorrp_atomic_density(RhorrpAtomicDensityInput {
                point: [0.08, 0.04, -0.03],
                orbital_index_1based: 1,
                atom_positions: reference.positions.view(),
                atom_potentials: &reference.potentials,
                radii: &reference.radii,
                large_components: reference.large.view(),
                small_components: reference.small.view(),
            })?,
            9.746_265_921_948_757,
        );
        assert_real_close_scaled(
            rhorrp_atomic_density(RhorrpAtomicDensityInput {
                point: [0.72, -0.15, 0.18],
                orbital_index_1based: 2,
                atom_positions: reference.positions.view(),
                atom_potentials: &reference.potentials,
                radii: &reference.radii,
                large_components: reference.large.view(),
                small_components: reference.small.view(),
            })?,
            2.182_748_347_338_233e1,
        );
        assert_real_close_scaled(
            rhorrp_atomic_density(RhorrpAtomicDensityInput {
                point: [0.0, 0.0, 0.0],
                orbital_index_1based: 3,
                atom_positions: reference.positions.view(),
                atom_potentials: &reference.potentials,
                radii: &reference.radii,
                large_components: reference.large.view(),
                small_components: reference.small.view(),
            })?,
            7.107_185_239_762_148e6,
        );
        assert_real_close_scaled(
            rhorrp_atomic_density(RhorrpAtomicDensityInput {
                point: [4.2, 3.9, -2.5],
                orbital_index_1based: 1,
                atom_positions: reference.positions.view(),
                atom_potentials: &reference.potentials,
                radii: &reference.radii,
                large_components: reference.large.view(),
                small_components: reference.small.view(),
            })?,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn atomic_density_rejects_invalid_inputs() {
        let reference = reference_atomic_density_tables();
        assert!(matches!(
            rhorrp_atomic_density(RhorrpAtomicDensityInput {
                point: [0.0, 0.0, 0.0],
                orbital_index_1based: 0,
                atom_positions: reference.positions.view(),
                atom_potentials: &reference.potentials,
                radii: &reference.radii,
                large_components: reference.large.view(),
                small_components: reference.small.view(),
            }),
            Err(RhorrpError::InvalidAtomicDensityOrbital {
                orbital: 0,
                orbital_count: 3,
            })
        ));

        let bad_potentials = [0, 1, 3, 1];
        assert!(matches!(
            rhorrp_atomic_density(RhorrpAtomicDensityInput {
                point: [0.0, 0.0, 0.0],
                orbital_index_1based: 1,
                atom_positions: reference.positions.view(),
                atom_potentials: &bad_potentials,
                radii: &reference.radii,
                large_components: reference.large.view(),
                small_components: reference.small.view(),
            }),
            Err(RhorrpError::InvalidAtomicDensityPotential {
                atom_index_1based: 3,
                potential: 3,
                max_potential: 2,
            })
        ));

        assert!(matches!(
            rhorrp_atomic_density(RhorrpAtomicDensityInput {
                point: [0.0, 0.0, 0.0],
                orbital_index_1based: 1,
                atom_positions: reference.positions.view(),
                atom_potentials: &reference.potentials,
                radii: &reference.radii[..11],
                large_components: reference.large.view(),
                small_components: reference.small.view(),
            }),
            Err(RhorrpError::AtomicDensityRadialLengthMismatch {
                radii: 11,
                components: 12,
            })
        ));
    }

    #[test]
    fn integrate_density_matches_feff_reference() -> Result<(), RhorrpError> {
        let (energies, energy_density) = reference_density_integration_inputs();

        assert_real_close(
            rhorrp_integrate_density(RhorrpDensityIntegrationInput {
                energies_hartree: energies.view(),
                energy_density: energy_density.view(),
                real_axis_count: 6,
                chemical_potential_hartree: 0.045,
                temperature_hartree: 0.0035,
                chemical_potential_override_hartree: None,
            })?,
            -4.627_669_214_946_009e-2,
        );
        assert_real_close(
            rhorrp_integrate_density(RhorrpDensityIntegrationInput {
                energies_hartree: energies.view(),
                energy_density: energy_density.view(),
                real_axis_count: 6,
                chemical_potential_hartree: -0.010,
                temperature_hartree: 0.000_001,
                chemical_potential_override_hartree: None,
            })?,
            -1.115_611_780_024_965e-3,
        );
        Ok(())
    }

    #[test]
    fn integrate_density_rejects_invalid_inputs() {
        let (energies, energy_density) = reference_density_integration_inputs();

        assert!(matches!(
            rhorrp_integrate_density(RhorrpDensityIntegrationInput {
                energies_hartree: energies.slice_axis(Axis(0), Slice::from(..7)),
                energy_density: energy_density.view(),
                real_axis_count: 6,
                chemical_potential_hartree: 0.045,
                temperature_hartree: 0.0035,
                chemical_potential_override_hartree: None,
            }),
            Err(RhorrpError::DensityIntegrationLengthMismatch {
                energies: 7,
                densities: 8,
            })
        ));
        assert!(matches!(
            rhorrp_integrate_density(RhorrpDensityIntegrationInput {
                energies_hartree: energies.view(),
                energy_density: energy_density.view(),
                real_axis_count: 1,
                chemical_potential_hartree: 0.045,
                temperature_hartree: 0.0035,
                chemical_potential_override_hartree: None,
            }),
            Err(RhorrpError::InvalidDensityIntegrationRealAxisCount {
                real_axis_count: 1,
                energy_count: 8,
            })
        ));

        let vertical_only = Array1::from_vec(vec![
            Complex::new(-0.03, 0.09),
            Complex::new(-0.03, 0.06),
            Complex::new(-0.03, 0.03),
            Complex::new(-0.03, 0.00),
        ]);
        let vertical_density = Array1::from_vec(vec![Complex::new(0.3, 0.1); 4]);
        assert!(matches!(
            rhorrp_integrate_density(RhorrpDensityIntegrationInput {
                energies_hartree: vertical_only.view(),
                energy_density: vertical_density.view(),
                real_axis_count: 4,
                chemical_potential_hartree: 0.045,
                temperature_hartree: 0.0035,
                chemical_potential_override_hartree: None,
            }),
            Err(RhorrpError::MissingDensityIntegrationCorner)
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

    fn reference_nearest_points() -> Array2<Real> {
        arr2(&[
            [0.7, 0.0, 0.2, 0.0],
            [0.2, 0.1, 0.9, 0.5],
            [0.1, 0.8, 0.1, 0.5],
        ])
    }

    fn sample_density(point: Vector3) -> Real {
        point[0] + 2.0 * point[1] - 0.5 * point[2] + point[0] * point[1]
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

    struct ReferenceAtomicDensityTables {
        radii: Vec<Real>,
        positions: Array2<Real>,
        potentials: [usize; 4],
        large: Array3<Real>,
        small: Array3<Real>,
    }

    fn reference_atomic_density_tables() -> ReferenceAtomicDensityTables {
        let radii = (1..=12)
            .map(|index| 0.015 + 0.035 * index as Real + 0.001 * (index as Real - 1.0).powi(2))
            .collect::<Vec<_>>();
        let positions = arr2(&[
            [0.0, 0.0, 0.0],
            [0.7, -0.2, 0.15],
            [-0.5, 0.55, -0.25],
            [1.85, 0.2, -0.1],
        ]);
        let potentials = [0, 1, 2, 1];
        let large = Array3::from_shape_fn((12, 3, 3), |(radial, orbital, potential)| {
            let index = (radial + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.13 * index).sin()
                + 0.031 * orbital
                + 0.047 * potential as Real
                + 0.12 * radii[radial]
        });
        let small = Array3::from_shape_fn((12, 3, 3), |(radial, orbital, potential)| {
            let index = (radial + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.09 * index).cos() - 0.019 * orbital + 0.023 * potential as Real
                - 0.08 * radii[radial]
        });
        ReferenceAtomicDensityTables {
            radii,
            positions,
            potentials,
            large,
            small,
        }
    }

    fn reference_density_integration_inputs() -> (Array1<Complex>, Array1<Complex>) {
        let energies = Array1::from_vec(vec![
            Complex::new(-0.030, 0.070),
            Complex::new(-0.030, 0.035),
            Complex::new(-0.030, 0.000),
            Complex::new(0.010, 0.000),
            Complex::new(0.065, 0.000),
            Complex::new(0.130, 0.000),
            Complex::new(0.045, 0.021_991_148_575_128_55),
            Complex::new(0.045, 0.043_982_297_150_257_1),
        ]);
        let energy_density = Array1::from_shape_fn(8, |index| {
            let energy = energies[index];
            let one_based = (index + 1) as Real;
            Complex::new(
                0.40 + 0.07 * one_based + 0.02 * energy.re - 0.15 * energy.im,
                -0.25 + 0.04 * one_based + 0.18 * energy.re + 0.03 * energy.im,
            )
        });
        (energies, energy_density)
    }

    fn column(points: &RealMat, index: usize) -> Vector3 {
        [points[(0, index)], points[(1, index)], points[(2, index)]]
    }

    fn row(points: &RealMat, index: usize) -> Vector3 {
        [points[(index, 0)], points[(index, 1)], points[(index, 2)]]
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

    fn assert_real_close_scaled(actual: Real, expected: Real) {
        let tolerance = 1.0e-11 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() < tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}, tolerance={tolerance:.17e}",
            (actual - expected).abs()
        );
    }

    fn assert_real_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}",
            (actual - expected).abs()
        );
    }
}
