use ndarray::{ArrayView1, ArrayView2, ArrayView3};
use refeff_linalg::LinalgError;
use thiserror::Error;

use crate::angular::AngularError;
use crate::interpolation::InterpolationError;
use crate::{Complex, Real, RealMat, RealVec, Vector3};

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
