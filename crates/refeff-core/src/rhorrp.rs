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

use crate::angular::{AngularError, legendre_polynomials_into, spherical_harmonics};
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
const FEFF_FINE_STRUCTURE_ALPHA: Real = 1.0 / 137.03598956;
const RHORRP_ORIGIN_EPSILON: Real = 1.0e-3;

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

/// Input for FEFF `init_inclus` FMS-radius atom counts.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpFmsInclusionInput<'a> {
    /// Atomic coordinates in Bohr as `(atom, xyz)`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Zero-based representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: &'a [usize],
    /// FMS inclusion radius in Bohr, FEFF `rfms2` after unit conversion.
    pub fms_radius: Real,
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

/// Input for FEFF `rhoerrp` radial-grid interpolation location.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpRadialInterpolationInput {
    /// Distance from the selected atom center in Bohr.
    pub radius: Real,
    /// FEFF logarithmic-grid offset `x0`.
    pub x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub dx: Real,
    /// Number of available radial samples `nr`.
    pub radial_count: usize,
}

/// FEFF radial interpolation index and fraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhorrpRadialInterpolationLocation {
    /// FEFF one-based lower radial index to pass into `interpwf`.
    pub index_below_1based: isize,
    /// Fractional distance from the lower radial sample.
    pub fraction: Real,
}

/// Input for the FEFF `rhoerrp` per-energy density prefactor.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpEnergyPrefactorInput {
    /// Complex contour energy in Hartree, FEFF `em(ie)`.
    pub energy_hartree: Complex,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
}

/// Input for FEFF `rhoerrp` final energy-density scaling.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpEnergyDensityInput<'a> {
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Accumulated local plus scattering Green's-function values, FEFF `Ge`.
    pub green_function: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// Radius `r` from the nearest atom center in Bohr.
    pub radius: Real,
    /// Radius `r'` from the nearest atom center in Bohr.
    pub prime_radius: Real,
}

/// Input for the FEFF `rhoerrp` point-pair energy-density assembly.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPairEnergyDensityInput<'a> {
    /// Complex contour energies in Hartree, FEFF `em`.
    pub energies_hartree: ArrayView1<'a, Complex>,
    /// Reference potential energy in Hartree, FEFF `eref0`.
    pub reference_energy_hartree: Complex,
    /// Regular large Dirac component near `r`, `prel(:,:,:,iph)`.
    pub first_regular_large: ArrayView3<'a, Complex>,
    /// Irregular large Dirac component near `r`, `pnel(:,:,:,iph)`.
    pub first_irregular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component near `r`, `qrel(:,:,:,iph)`.
    pub first_regular_small: ArrayView3<'a, Complex>,
    /// Irregular small Dirac component near `r`, `qnel(:,:,:,iph)`.
    pub first_irregular_small: ArrayView3<'a, Complex>,
    /// Regular large Dirac component near `r'`, `prel(:,:,:,iphp)`.
    pub second_regular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component near `r'`, `qrel(:,:,:,iphp)`.
    pub second_regular_small: ArrayView3<'a, Complex>,
    /// Phase shifts for the first potential as `(energy, l)`, FEFF `ph2`.
    pub first_phase: ArrayView2<'a, Complex>,
    /// Phase shifts for the second potential as `(energy, l)`, FEFF `ph2`.
    pub second_phase: ArrayView2<'a, Complex>,
    /// FMS scattering matrix slice as `(energy, L, L')`; `None` skips scattering.
    pub scattering_matrix: Option<ArrayView3<'a, Complex>>,
    /// Whether `r` and `r'` are nearest to the same atom and need the local term.
    pub same_atom: bool,
    /// Displacement from the first nearest atom to `r`, FEFF `dv`.
    pub first_displacement: Vector3,
    /// Displacement from the second nearest atom to `r'`, FEFF `dvp`.
    pub second_displacement: Vector3,
    /// FEFF logarithmic-grid offset `x0`.
    pub radial_x0: Real,
    /// FEFF logarithmic-grid spacing `dx`.
    pub radial_dx: Real,
    /// Number of available radial samples `nr`.
    pub radial_count: usize,
}

/// Input for FEFF `rhorrp` after nearest-atom and FMS-matrix selection.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpPairDensityInput<'a> {
    /// Point-pair energy-density assembly input, matching FEFF `rhoerrp`.
    pub pair_energy: RhorrpPairEnergyDensityInput<'a>,
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

/// Input for the same-site local Green's-function term in FEFF `rhoerrp`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpSameSiteGreenInput<'a> {
    /// Regular large Dirac component `prel` as `(energy, l, radial)`.
    pub regular_large: ArrayView3<'a, Complex>,
    /// Irregular large Dirac component `pnel` as `(energy, l, radial)`.
    pub irregular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component `qrel` as `(energy, l, radial)`.
    pub regular_small: ArrayView3<'a, Complex>,
    /// Irregular small Dirac component `qnel` as `(energy, l, radial)`.
    pub irregular_small: ArrayView3<'a, Complex>,
    /// Radial interpolation location for `r`.
    pub first_location: RhorrpRadialInterpolationLocation,
    /// Radial interpolation location for `r'`.
    pub second_location: RhorrpRadialInterpolationLocation,
    /// Cosine of the angle between same-site displacement vectors.
    pub cosine_between: Real,
}

/// Input for the scattering Green's-function term in FEFF `rhoerrp`.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpScatteringGreenInput<'a> {
    /// Regular large Dirac component near `r`, `prel(:,:,:,iph)`.
    pub first_regular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component near `r`, `qrel(:,:,:,iph)`.
    pub first_regular_small: ArrayView3<'a, Complex>,
    /// Regular large Dirac component near `r'`, `prel(:,:,:,iphp)`.
    pub second_regular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac component near `r'`, `qrel(:,:,:,iphp)`.
    pub second_regular_small: ArrayView3<'a, Complex>,
    /// Phase shifts for the first potential as `(energy, l)`, FEFF `ph2`.
    pub first_phase: ArrayView2<'a, Complex>,
    /// Phase shifts for the second potential as `(energy, l)`, FEFF `ph2`.
    pub second_phase: ArrayView2<'a, Complex>,
    /// FMS scattering matrix slice as `(energy, L, L')`.
    pub scattering_matrix: ArrayView3<'a, Complex>,
    /// Radial interpolation location for `r`.
    pub first_location: RhorrpRadialInterpolationLocation,
    /// Radial interpolation location for `r'`.
    pub second_location: RhorrpRadialInterpolationLocation,
    /// Displacement from the first nearest atom to `r`, FEFF `dv`.
    pub first_displacement: Vector3,
    /// Displacement from the second nearest atom to `r'`, FEFF `dvp`.
    pub second_displacement: Vector3,
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
    /// FEFF representative atom indices must point into the atom table.
    #[error(
        "RHORRP representative atom for potential {potential} is outside 0..{atoms}, got {representative}"
    )]
    InvalidRepresentativeAtom {
        potential: usize,
        representative: usize,
        atoms: usize,
    },
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
    /// RHORRP wavefunction component arrays must share the same shape.
    #[error(
        "RHORRP {component} wavefunction shape ({actual_energy}, {actual_angular}, {actual_radial}) does not match ({expected_energy}, {expected_angular}, {expected_radial})"
    )]
    WavefunctionComponentShapeMismatch {
        component: &'static str,
        expected_energy: usize,
        expected_angular: usize,
        expected_radial: usize,
        actual_energy: usize,
        actual_angular: usize,
        actual_radial: usize,
    },
    /// RHORRP phase tables must align with wavefunction energy and angular axes.
    #[error(
        "RHORRP {component} phase shape ({actual_energy}, {actual_angular}) does not match ({expected_energy}, {expected_angular})"
    )]
    PhaseShapeMismatch {
        component: &'static str,
        expected_energy: usize,
        expected_angular: usize,
        actual_energy: usize,
        actual_angular: usize,
    },
    /// RHORRP scattering matrices use `(energy, L, L')`.
    #[error(
        "RHORRP scattering matrix shape ({actual_energy}, {actual_rows}, {actual_columns}) does not match ({expected_energy}, {expected_states}, {expected_states})"
    )]
    ScatteringMatrixShapeMismatch {
        expected_energy: usize,
        expected_states: usize,
        actual_energy: usize,
        actual_rows: usize,
        actual_columns: usize,
    },
    /// Final RHORRP energy-density scaling needs one Green's value per energy.
    #[error("RHORRP energy-density length mismatch: energies={energies}, green={green}")]
    EnergyDensityLengthMismatch { energies: usize, green: usize },
    /// Final RHORRP energy-density scaling divides by positive radii.
    #[error("RHORRP {name} must be positive, got {value}")]
    InvalidPositiveRadius { name: &'static str, value: Real },
    /// FEFF radial interpolation needs at least one radial sample.
    #[error("RHORRP radial_count must be positive, got {radial_count}")]
    InvalidRadialCount { radial_count: usize },
    /// FEFF radial interpolation uses a positive logarithmic-grid spacing.
    #[error("RHORRP radial dx must be positive, got {value}")]
    InvalidRadialStep { value: Real },
    /// FEFF radial interpolation receives radii from vector norms.
    #[error("RHORRP radial radius must be non-negative, got {value}")]
    InvalidRadius { value: Real },
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
    /// Spherical harmonic evaluation failed while building the scattering term.
    #[error("RHORRP spherical-harmonic evaluation failed: {source}")]
    SphericalHarmonics {
        #[from]
        source: AngularError,
    },
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

/// Port of FEFF `init_inclus` FMS-radius atom counting.
///
/// For each potential representative atom, FEFF counts all atoms with
/// `|r_atom - r_representative|^2 <= rfms2^2`. Coordinates and `fms_radius`
/// are already in Bohr, matching RHORRP after input-file unit conversion.
pub fn rhorrp_fms_inclusion_counts(
    input: RhorrpFmsInclusionInput<'_>,
) -> Result<Vec<usize>, RhorrpError> {
    validate_fms_inclusion_input(input)?;
    let radius_squared = input.fms_radius * input.fms_radius;

    Ok(input
        .representative_atoms
        .iter()
        .map(|&representative| {
            let center = [
                input.atom_positions[(representative, 0)],
                input.atom_positions[(representative, 1)],
                input.atom_positions[(representative, 2)],
            ];
            (0..input.atom_positions.nrows())
                .filter(|&atom| {
                    let dx = input.atom_positions[(atom, 0)] - center[0];
                    let dy = input.atom_positions[(atom, 1)] - center[1];
                    let dz = input.atom_positions[(atom, 2)] - center[2];
                    dx * dx + dy * dy + dz * dz <= radius_squared
                })
                .count()
        })
        .collect())
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

/// Port of FEFF `rhoerrp` radial interpolation index setup.
///
/// FEFF maps a radius to `f = (log(r) + x0) / dx + 1`, clamps `f` to
/// `1..=nr`, truncates it to the one-based lower index, then keeps the
/// fractional remainder for `interpwf`.
pub fn rhorrp_radial_interpolation_location(
    input: RhorrpRadialInterpolationInput,
) -> Result<RhorrpRadialInterpolationLocation, RhorrpError> {
    validate_radial_interpolation_input(input)?;

    let mut position = if input.radius == 0.0 {
        1.0
    } else {
        (input.radius.ln() + input.x0) / input.dx + 1.0
    };
    position = position.clamp(1.0, input.radial_count as Real);

    let index_below_1based = position.trunc() as isize;
    Ok(RhorrpRadialInterpolationLocation {
        index_below_1based,
        fraction: position - index_below_1based as Real,
    })
}

/// Port of FEFF `rhoerrp` final per-energy prefactor.
///
/// FEFF converts `p2 = E - eref0` to the relativistic wave number `ck`, derives
/// the small-component ratio `pu`, and multiplies the accumulated Green's
/// function by `4 * ck / (pi * (1 + pu^2))`.
pub fn rhorrp_energy_prefactor(input: RhorrpEnergyPrefactorInput) -> Result<Complex, RhorrpError> {
    validate_energy_prefactor_input(input)?;

    let one = Complex::new(1.0, 0.0);
    let p2 = input.energy_hartree - input.reference_energy_hartree;
    let alpha_p2 = p2 * FEFF_FINE_STRUCTURE_ALPHA;
    let ck = (p2 * 2.0 + alpha_p2 * alpha_p2).sqrt();
    let scaled_ck = ck * FEFF_FINE_STRUCTURE_ALPHA;
    let pu = -scaled_ck / (one + (one + scaled_ck * scaled_ck).sqrt());
    Ok(ck * (4.0 / std::f64::consts::PI) / (one + pu * pu))
}

/// Port of FEFF `rhoerrp` final energy-density scaling loop.
///
/// After local/scattering contributions are accumulated in `Ge`, FEFF applies
/// the relativistic per-energy prefactor and divides by `r * r'` to produce
/// `rhoe(ie)`.
pub fn rhorrp_finish_energy_density(
    input: RhorrpEnergyDensityInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    validate_energy_density_input(input)?;
    let radius_scale = input.radius * input.prime_radius;
    let mut density = ComplexVec::zeros(input.energies_hartree.len());
    for (index, (&energy, &green)) in input
        .energies_hartree
        .iter()
        .zip(input.green_function.iter())
        .enumerate()
    {
        let prefactor = rhorrp_energy_prefactor(RhorrpEnergyPrefactorInput {
            energy_hartree: energy,
            reference_energy_hartree: input.reference_energy_hartree,
        })?;
        density[index] = green * prefactor / radius_scale;
    }
    Ok(density)
}

/// Port of FEFF `rhoerrp` after atom and FMS-slice selection.
///
/// The caller supplies wavefunction/phase views for the selected potentials and
/// the already-selected scattering matrix for this point pair. The helper keeps
/// FEFF's near-origin displacement adjustment, logarithmic radial-grid lookup,
/// optional same-site local term, optional scattering term, and final
/// relativistic energy scaling in one composable operation.
pub fn rhorrp_pair_energy_density(
    input: RhorrpPairEnergyDensityInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    let energy_count = validate_pair_energy_density_input(input)?;
    let (first_displacement, first_radius) =
        regularize_density_displacement("first_displacement", input.first_displacement)?;
    let (second_displacement, second_radius) =
        regularize_density_displacement("second_displacement", input.second_displacement)?;
    let first_location = rhorrp_radial_interpolation_location(RhorrpRadialInterpolationInput {
        radius: first_radius,
        x0: input.radial_x0,
        dx: input.radial_dx,
        radial_count: input.radial_count,
    })?;
    let second_location = rhorrp_radial_interpolation_location(RhorrpRadialInterpolationInput {
        radius: second_radius,
        x0: input.radial_x0,
        dx: input.radial_dx,
        radial_count: input.radial_count,
    })?;

    let mut green = ComplexVec::zeros(energy_count);
    if input.same_atom {
        let same_site = rhorrp_same_site_green(RhorrpSameSiteGreenInput {
            regular_large: input.first_regular_large,
            irregular_large: input.first_irregular_large,
            regular_small: input.first_regular_small,
            irregular_small: input.first_irregular_small,
            first_location,
            second_location,
            cosine_between: cosine_between_vectors(first_displacement, second_displacement)?,
        })?;
        for (total, contribution) in green.iter_mut().zip(same_site.iter()) {
            *total += *contribution;
        }
    }
    if let Some(scattering_matrix) = input.scattering_matrix {
        let scattering = rhorrp_scattering_green(RhorrpScatteringGreenInput {
            first_regular_large: input.first_regular_large,
            first_regular_small: input.first_regular_small,
            second_regular_large: input.second_regular_large,
            second_regular_small: input.second_regular_small,
            first_phase: input.first_phase,
            second_phase: input.second_phase,
            scattering_matrix,
            first_location,
            second_location,
            first_displacement,
            second_displacement,
        })?;
        for (total, contribution) in green.iter_mut().zip(scattering.iter()) {
            *total += *contribution;
        }
    }

    rhorrp_finish_energy_density(RhorrpEnergyDensityInput {
        energies_hartree: input.energies_hartree,
        green_function: green.view(),
        reference_energy_hartree: input.reference_energy_hartree,
        radius: first_radius,
        prime_radius: second_radius,
    })
}

/// Port of FEFF `rhorrp` after point-pair setup.
///
/// This helper evaluates the energy-dependent density matrix with
/// [`rhorrp_pair_energy_density`] and immediately integrates it over the FEFF
/// occupied-state contour with [`rhorrp_integrate_density`].
pub fn rhorrp_pair_density(input: RhorrpPairDensityInput<'_>) -> Result<Real, RhorrpError> {
    let energy_density = rhorrp_pair_energy_density(input.pair_energy)?;
    rhorrp_integrate_density(RhorrpDensityIntegrationInput {
        energies_hartree: input.pair_energy.energies_hartree,
        energy_density: energy_density.view(),
        real_axis_count: input.real_axis_count,
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree: input.temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })
}

/// Port of FEFF `rhoerrp` same-site local Green's-function term.
///
/// This evaluates the branch used when `r` and `r'` are nearest to the same
/// atom. FEFF orders the two radial interpolation locations by lower radial
/// index, uses regular solutions at the lesser radius and irregular-minus-iR
/// solutions at the greater radius, then sums over `l` with `P_l(cos theta)`.
pub fn rhorrp_same_site_green(
    input: RhorrpSameSiteGreenInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    let (energy_count, angular_count, _) = validate_same_site_green_input(input)?;
    let (lesser, greater) = ordered_radial_locations(input.first_location, input.second_location);

    let regular_large_lesser = interpolate_component(input.regular_large, lesser)?;
    let regular_large_greater = interpolate_component(input.regular_large, greater)?;
    let irregular_large_greater = interpolate_component(input.irregular_large, greater)?;
    let regular_small_lesser = interpolate_component(input.regular_small, lesser)?;
    let regular_small_greater = interpolate_component(input.regular_small, greater)?;
    let irregular_small_greater = interpolate_component(input.irregular_small, greater)?;

    let mut legendre = vec![0.0; angular_count];
    legendre_polynomials_into(input.cosine_between, &mut legendre);

    let imaginary = Complex::new(0.0, 1.0);
    let mut green = ComplexVec::zeros(energy_count);
    for energy in 0..energy_count {
        for angular in 0..angular_count {
            let rho_l = -regular_large_lesser[(energy, angular)]
                * (irregular_large_greater[(energy, angular)]
                    - imaginary * regular_large_greater[(energy, angular)])
                - regular_small_lesser[(energy, angular)]
                    * (irregular_small_greater[(energy, angular)]
                        - imaginary * regular_small_greater[(energy, angular)]);
            let angular_factor =
                legendre[angular] * (2 * angular + 1) as Real / (4.0 * std::f64::consts::PI);
            green[energy] += rho_l * angular_factor;
        }
    }
    Ok(green)
}

/// Port of FEFF `rhoerrp` scattering Green's-function term.
///
/// This evaluates the branch below `call ylm` in FEFF. The `L`/`L'` state axes
/// use FEFF spherical-harmonic order, while the radial components are indexed
/// by their corresponding angular momentum `l`.
pub fn rhorrp_scattering_green(
    input: RhorrpScatteringGreenInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    let (energy_count, angular_count, state_count) = validate_scattering_green_input(input)?;
    let first_large = interpolate_component(input.first_regular_large, input.first_location)?;
    let first_small = interpolate_component(input.first_regular_small, input.first_location)?;
    let second_large = interpolate_component(input.second_regular_large, input.second_location)?;
    let second_small = interpolate_component(input.second_regular_small, input.second_location)?;
    let lmax = angular_count - 1;
    let first_harmonics = spherical_harmonics(input.first_displacement, lmax)?;
    let second_harmonics = spherical_harmonics(input.second_displacement, lmax)?;

    let imaginary = Complex::new(0.0, 1.0);
    let mut green = ComplexVec::zeros(energy_count);
    for first_state in 0..state_count {
        let first_l = angular_momentum_for_state_index(first_state);
        let first_factor = first_harmonics[first_state] * imaginary_power(first_l);
        for second_state in 0..state_count {
            let second_l = angular_momentum_for_state_index(second_state);
            let angular_factor = first_factor
                * second_harmonics[second_state].conj()
                * negative_imaginary_power(second_l);
            for energy in 0..energy_count {
                let radial = first_large[(energy, first_l)] * second_large[(energy, second_l)]
                    + first_small[(energy, first_l)] * second_small[(energy, second_l)];
                let phase = (imaginary
                    * (input.first_phase[(energy, first_l)]
                        + input.second_phase[(energy, second_l)]))
                    .exp();
                green[energy] += radial
                    * angular_factor
                    * phase
                    * input.scattering_matrix[(energy, first_state, second_state)];
            }
        }
    }
    Ok(green)
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

fn validate_fms_inclusion_input(input: RhorrpFmsInclusionInput<'_>) -> Result<(), RhorrpError> {
    let (rows, columns) = input.atom_positions.dim();
    if columns != 3 {
        return Err(RhorrpError::InvalidAtomPositionShape { rows, columns });
    }
    if rows == 0 {
        return Err(RhorrpError::NoAtoms);
    }
    validate_scalar("fms_radius", 0, input.fms_radius)?;
    for (index, &value) in input.atom_positions.iter().enumerate() {
        validate_scalar("atom_positions", index, value)?;
    }
    for (potential, &representative) in input.representative_atoms.iter().enumerate() {
        if representative >= rows {
            return Err(RhorrpError::InvalidRepresentativeAtom {
                potential,
                representative,
                atoms: rows,
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

fn validate_radial_interpolation_input(
    input: RhorrpRadialInterpolationInput,
) -> Result<(), RhorrpError> {
    validate_scalar("radius", 0, input.radius)?;
    validate_scalar("x0", 0, input.x0)?;
    validate_scalar("dx", 0, input.dx)?;
    if input.radius < 0.0 {
        return Err(RhorrpError::InvalidRadius {
            value: input.radius,
        });
    }
    if input.dx <= 0.0 {
        return Err(RhorrpError::InvalidRadialStep { value: input.dx });
    }
    if input.radial_count == 0 || input.radial_count > isize::MAX as usize {
        return Err(RhorrpError::InvalidRadialCount {
            radial_count: input.radial_count,
        });
    }
    Ok(())
}

fn validate_energy_prefactor_input(input: RhorrpEnergyPrefactorInput) -> Result<(), RhorrpError> {
    validate_scalar("energy_hartree.real", 0, input.energy_hartree.re)?;
    validate_scalar("energy_hartree.imag", 0, input.energy_hartree.im)?;
    validate_scalar(
        "reference_energy_hartree.real",
        0,
        input.reference_energy_hartree.re,
    )?;
    validate_scalar(
        "reference_energy_hartree.imag",
        0,
        input.reference_energy_hartree.im,
    )
}

fn validate_energy_density_input(input: RhorrpEnergyDensityInput<'_>) -> Result<(), RhorrpError> {
    if input.energies_hartree.len() != input.green_function.len() {
        return Err(RhorrpError::EnergyDensityLengthMismatch {
            energies: input.energies_hartree.len(),
            green: input.green_function.len(),
        });
    }
    validate_scalar("radius", 0, input.radius)?;
    validate_scalar("prime_radius", 0, input.prime_radius)?;
    validate_scalar(
        "reference_energy_hartree.real",
        0,
        input.reference_energy_hartree.re,
    )?;
    validate_scalar(
        "reference_energy_hartree.imag",
        0,
        input.reference_energy_hartree.im,
    )?;
    if input.radius <= 0.0 {
        return Err(RhorrpError::InvalidPositiveRadius {
            name: "radius",
            value: input.radius,
        });
    }
    if input.prime_radius <= 0.0 {
        return Err(RhorrpError::InvalidPositiveRadius {
            name: "prime_radius",
            value: input.prime_radius,
        });
    }
    for (index, &energy) in input.energies_hartree.iter().enumerate() {
        validate_scalar("energies_hartree.real", index, energy.re)?;
        validate_scalar("energies_hartree.imag", index, energy.im)?;
    }
    for (index, &green) in input.green_function.iter().enumerate() {
        validate_scalar("green_function.real", index, green.re)?;
        validate_scalar("green_function.imag", index, green.im)?;
    }
    Ok(())
}

fn validate_pair_energy_density_input(
    input: RhorrpPairEnergyDensityInput<'_>,
) -> Result<usize, RhorrpError> {
    let (energy, angular, radial) = input.first_regular_large.dim();
    if energy == 0 || angular == 0 || radial == 0 {
        return Err(RhorrpError::InvalidWavefunctionShape {
            energy,
            angular,
            radial,
        });
    }
    validate_wavefunction_component_shape(
        "first_irregular_large",
        input.first_regular_large,
        input.first_irregular_large,
    )?;
    validate_wavefunction_component_shape(
        "first_regular_small",
        input.first_regular_large,
        input.first_regular_small,
    )?;
    validate_wavefunction_component_shape(
        "first_irregular_small",
        input.first_regular_large,
        input.first_irregular_small,
    )?;
    validate_wavefunction_component_shape(
        "second_regular_large",
        input.first_regular_large,
        input.second_regular_large,
    )?;
    validate_wavefunction_component_shape(
        "second_regular_small",
        input.first_regular_large,
        input.second_regular_small,
    )?;
    validate_phase_shape("first_phase", input.first_phase, energy, angular)?;
    validate_phase_shape("second_phase", input.second_phase, energy, angular)?;
    if let Some(scattering_matrix) = input.scattering_matrix {
        let state_count = angular
            .checked_mul(angular)
            .ok_or(RhorrpError::PointCountOverflow)?;
        validate_scattering_matrix_shape(scattering_matrix, energy, state_count)?;
    }
    Ok(energy)
}

fn validate_same_site_green_input(
    input: RhorrpSameSiteGreenInput<'_>,
) -> Result<(usize, usize, usize), RhorrpError> {
    validate_scalar("cosine_between", 0, input.cosine_between)?;
    let (energy, angular, radial) = input.regular_large.dim();
    if energy == 0 || angular == 0 || radial == 0 {
        return Err(RhorrpError::InvalidWavefunctionShape {
            energy,
            angular,
            radial,
        });
    }
    validate_wavefunction_component_shape(
        "irregular_large",
        input.regular_large,
        input.irregular_large,
    )?;
    validate_wavefunction_component_shape(
        "regular_small",
        input.regular_large,
        input.regular_small,
    )?;
    validate_wavefunction_component_shape(
        "irregular_small",
        input.regular_large,
        input.irregular_small,
    )?;
    validate_wavefunction_interpolation_input(RhorrpWavefunctionInterpolationInput {
        wavefunctions: input.regular_large,
        index_below_1based: input.first_location.index_below_1based,
        fraction: input.first_location.fraction,
    })?;
    validate_wavefunction_interpolation_input(RhorrpWavefunctionInterpolationInput {
        wavefunctions: input.regular_large,
        index_below_1based: input.second_location.index_below_1based,
        fraction: input.second_location.fraction,
    })?;
    Ok((energy, angular, radial))
}

fn validate_scattering_green_input(
    input: RhorrpScatteringGreenInput<'_>,
) -> Result<(usize, usize, usize), RhorrpError> {
    validate_vector("first_displacement", input.first_displacement)?;
    validate_vector("second_displacement", input.second_displacement)?;
    let (energy, angular, radial) = input.first_regular_large.dim();
    if energy == 0 || angular == 0 || radial == 0 {
        return Err(RhorrpError::InvalidWavefunctionShape {
            energy,
            angular,
            radial,
        });
    }
    validate_wavefunction_component_shape(
        "first_regular_small",
        input.first_regular_large,
        input.first_regular_small,
    )?;
    validate_wavefunction_component_shape(
        "second_regular_large",
        input.first_regular_large,
        input.second_regular_large,
    )?;
    validate_wavefunction_component_shape(
        "second_regular_small",
        input.first_regular_large,
        input.second_regular_small,
    )?;
    validate_phase_shape("first_phase", input.first_phase, energy, angular)?;
    validate_phase_shape("second_phase", input.second_phase, energy, angular)?;
    let state_count = angular
        .checked_mul(angular)
        .ok_or(RhorrpError::PointCountOverflow)?;
    validate_scattering_matrix_shape(input.scattering_matrix, energy, state_count)?;
    validate_wavefunction_interpolation_input(RhorrpWavefunctionInterpolationInput {
        wavefunctions: input.first_regular_large,
        index_below_1based: input.first_location.index_below_1based,
        fraction: input.first_location.fraction,
    })?;
    validate_wavefunction_interpolation_input(RhorrpWavefunctionInterpolationInput {
        wavefunctions: input.second_regular_large,
        index_below_1based: input.second_location.index_below_1based,
        fraction: input.second_location.fraction,
    })?;
    for (index, value) in input.first_phase.iter().enumerate() {
        validate_scalar("first_phase.real", index, value.re)?;
        validate_scalar("first_phase.imag", index, value.im)?;
    }
    for (index, value) in input.second_phase.iter().enumerate() {
        validate_scalar("second_phase.real", index, value.re)?;
        validate_scalar("second_phase.imag", index, value.im)?;
    }
    for (index, value) in input.scattering_matrix.iter().enumerate() {
        validate_scalar("scattering_matrix.real", index, value.re)?;
        validate_scalar("scattering_matrix.imag", index, value.im)?;
    }
    Ok((energy, angular, state_count))
}

fn validate_wavefunction_component_shape(
    component: &'static str,
    reference: ArrayView3<'_, Complex>,
    actual: ArrayView3<'_, Complex>,
) -> Result<(), RhorrpError> {
    let (expected_energy, expected_angular, expected_radial) = reference.dim();
    let (actual_energy, actual_angular, actual_radial) = actual.dim();
    if actual.dim() != reference.dim() {
        return Err(RhorrpError::WavefunctionComponentShapeMismatch {
            component,
            expected_energy,
            expected_angular,
            expected_radial,
            actual_energy,
            actual_angular,
            actual_radial,
        });
    }
    Ok(())
}

fn validate_phase_shape(
    component: &'static str,
    actual: ArrayView2<'_, Complex>,
    expected_energy: usize,
    expected_angular: usize,
) -> Result<(), RhorrpError> {
    let (actual_energy, actual_angular) = actual.dim();
    if actual_energy != expected_energy || actual_angular != expected_angular {
        return Err(RhorrpError::PhaseShapeMismatch {
            component,
            expected_energy,
            expected_angular,
            actual_energy,
            actual_angular,
        });
    }
    Ok(())
}

fn validate_scattering_matrix_shape(
    actual: ArrayView3<'_, Complex>,
    expected_energy: usize,
    expected_states: usize,
) -> Result<(), RhorrpError> {
    let (actual_energy, actual_rows, actual_columns) = actual.dim();
    if actual_energy != expected_energy
        || actual_rows != expected_states
        || actual_columns != expected_states
    {
        return Err(RhorrpError::ScatteringMatrixShapeMismatch {
            expected_energy,
            expected_states,
            actual_energy,
            actual_rows,
            actual_columns,
        });
    }
    Ok(())
}

fn ordered_radial_locations(
    first: RhorrpRadialInterpolationLocation,
    second: RhorrpRadialInterpolationLocation,
) -> (
    RhorrpRadialInterpolationLocation,
    RhorrpRadialInterpolationLocation,
) {
    if first.index_below_1based > second.index_below_1based {
        (second, first)
    } else {
        (first, second)
    }
}

fn interpolate_component(
    wavefunctions: ArrayView3<'_, Complex>,
    location: RhorrpRadialInterpolationLocation,
) -> Result<ComplexMat, RhorrpError> {
    rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
        wavefunctions,
        index_below_1based: location.index_below_1based,
        fraction: location.fraction,
    })
}

fn regularize_density_displacement(
    name: &'static str,
    displacement: Vector3,
) -> Result<(Vector3, Real), RhorrpError> {
    validate_vector(name, displacement)?;
    let radius_squared: Real = displacement.iter().map(|value| value * value).sum();
    let radius = radius_squared.sqrt();
    if radius < RHORRP_ORIGIN_EPSILON {
        let mut adjusted = displacement;
        adjusted[2] += RHORRP_ORIGIN_EPSILON;
        Ok((adjusted, RHORRP_ORIGIN_EPSILON))
    } else {
        Ok((displacement, radius))
    }
}

fn cosine_between_vectors(first: Vector3, second: Vector3) -> Result<Real, RhorrpError> {
    let dot: Real = first
        .iter()
        .zip(second.iter())
        .map(|(left, right)| left * right)
        .sum();
    let first_norm = first.iter().map(|value| value * value).sum::<Real>().sqrt();
    let second_norm = second
        .iter()
        .map(|value| value * value)
        .sum::<Real>()
        .sqrt();
    let cosine = dot / (first_norm * second_norm);
    validate_scalar("cosine_between", 0, cosine)?;
    Ok(cosine)
}

fn angular_momentum_for_state_index(state: usize) -> usize {
    let mut angular = 0usize;
    while (angular + 1)
        .checked_mul(angular + 1)
        .is_some_and(|limit| limit <= state)
    {
        angular += 1;
    }
    angular
}

fn imaginary_power(exponent: usize) -> Complex {
    match exponent % 4 {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

fn negative_imaginary_power(exponent: usize) -> Complex {
    match exponent % 4 {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, -1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, 1.0),
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
mod tests;
