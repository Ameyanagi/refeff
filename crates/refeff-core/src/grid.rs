//! FEFF common energy and radial-grid helpers.
//!
//! These functions port the small common routines `getxk.f90`, `xx.f90`,
//! `m_ifuns.f90`, and radial resampling helpers from `COMMON/`. FEFF uses a
//! 1-based logarithmic radial grid with `x = -8.8 + (j - 1) * delta` and
//! `r = exp(x)`.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};
use num_complex::Complex32;
use refeff_linalg::{Complex32Lu, LinalgError, complex32_lu_factor};
use thiserror::Error;

use crate::interpolation::{InterpolationError, terp};
use crate::quadrature::{QuadratureError, somm2};
use crate::vector::distance_between;
use crate::{Complex, Real};

/// Default FEFF logarithmic radial-grid spacing.
pub const LOUCKS_DELTA: Real = 0.05;

/// Offset used by FEFF's Loucks radial grid.
pub const LOUCKS_X_OFFSET: Real = 8.8;

/// FEFF Hartree constant in eV, from `COMMON/m_constants.f90`.
pub const FEFF_HARTREE_EV: Real = 27.211_396;

/// FEFF Fermi-momentum factor `(9*pi/4)^(1/3)`, from `COMMON/m_constants.f90`.
pub const FEFF_FERMI_MOMENTUM_FACTOR: Real = 1.919_158_292_677_512_8;

const SPINOR_ZERO_THRESHOLD: Real = 1.0e-11;
const SUMAX_WIGNER_SEITZ_RADIUS: Real = 15.0;
const SUMAX_LITERAL_DELTA: Real = 0.05_f32 as Real;
const SUMAX_LITERAL_OFFSET: Real = 8.8_f32 as Real;
const SIDX_DENSITY_CUTOFF: Real = 1.0e-5;
const FRNRM_DENSITY_POINTS: usize = 251;
const FRNRM_NRPTX: usize = 1251;
const FRNRM_LITERAL_DELTA: Real = 0.05_f32 as Real;
const FRNRM_LITERAL_OFFSET: Real = 8.8_f32 as Real;
const FRNRM_CORRECTION_THRESHOLD: Real = 0.0001_f32 as Real;
const MOVRLP_NOVP: usize = 50;

/// Inputs for FEFF `COMMON/fixdsp.f90` spinor grid interpolation.
#[derive(Debug, Clone, Copy)]
pub struct DiracSpinorGridInput<'a> {
    /// Original FEFF logarithmic-grid spacing `dxorg`.
    pub original_delta: Real,
    /// Target FEFF logarithmic-grid spacing `dxnew`.
    pub new_delta: Real,
    /// Original large Dirac component on the source grid.
    pub large_component: ArrayView1<'a, Real>,
    /// Original small Dirac component on the source grid.
    pub small_component: ArrayView1<'a, Real>,
    /// Length of the target FEFF radial grid, equivalent to `nrptx`.
    pub output_len: usize,
}

/// FEFF `fixdsp` spinor components on a target logarithmic grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DiracSpinorGrid {
    /// Interpolated large Dirac component.
    pub large_component: Array1<Real>,
    /// Interpolated small Dirac component.
    pub small_component: Array1<Real>,
    /// Number of target-grid points filled before the zero tail.
    pub active_len: usize,
}

/// Inputs for FEFF `COMMON/fixdsx.f90` multi-orbital spinor interpolation.
#[derive(Debug, Clone, Copy)]
pub struct DiracSpinorOrbitalsGridInput<'a> {
    /// Original FEFF logarithmic-grid spacing `dxorg`.
    pub original_delta: Real,
    /// Target FEFF logarithmic-grid spacing `dxnew`.
    pub new_delta: Real,
    /// Original large Dirac components as `(source_radial, orbital)`.
    pub large_components: ArrayView2<'a, Real>,
    /// Original small Dirac components as `(source_radial, orbital)`.
    pub small_components: ArrayView2<'a, Real>,
    /// Length of the target FEFF radial grid, equivalent to `nrptx`.
    pub output_len: usize,
}

/// FEFF `fixdsx` spinor components on a target logarithmic grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DiracSpinorOrbitalsGrid {
    /// Interpolated large Dirac components as `(target_radial, orbital)`.
    pub large_components: Array2<Real>,
    /// Interpolated small Dirac components as `(target_radial, orbital)`.
    pub small_components: Array2<Real>,
    /// Per-orbital target-grid active lengths before zero tails.
    pub active_lengths: Array1<usize>,
}

/// Inputs for FEFF `COMMON/fixvar.f90` potential and density interpolation.
#[derive(Debug, Clone, Copy)]
pub struct PotentialGridInput<'a> {
    /// Muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
    /// Overlapping charge density on the source grid, matching FEFF `edens`.
    ///
    /// FEFF callers pass density multiplied by `4*pi`; [`fix_potential_grid`]
    /// divides the interpolated output by `4*pi`.
    pub electron_density: ArrayView1<'a, Real>,
    /// Total potential on the source grid, matching FEFF `vtot`.
    pub total_potential: ArrayView1<'a, Real>,
    /// Magnetization density on the source grid, matching FEFF `dmag`.
    pub magnetization: ArrayView1<'a, Real>,
    /// Interstitial potential `vint` used to fill the target-grid tail.
    pub interstitial_potential: Real,
    /// Interstitial charge density `rhoint` used to fill the target-grid tail.
    pub interstitial_density: Real,
    /// Original FEFF logarithmic-grid spacing `dxorg`.
    pub original_delta: Real,
    /// Target FEFF logarithmic-grid spacing `dxnew`.
    pub new_delta: Real,
    /// FEFF jump mode `jumprm`: `0` disables, `1` recomputes, `>0` applies.
    pub jump_mode: i32,
    /// Input potential jump `vjump`, or the initial value for `jump_mode == 1`.
    pub potential_jump: Real,
    /// Length of the target FEFF radial grid, equivalent to `nrptx`.
    pub output_len: usize,
}

/// FEFF `fixvar` potential, charge-density, and magnetization target grid.
#[derive(Debug, Clone, PartialEq)]
pub struct PotentialGrid {
    /// Target radial coordinates `ri`.
    pub radii: Array1<Real>,
    /// Target total potential `vtotph`.
    pub total_potential: Array1<Real>,
    /// Target charge density `rhoph`, after FEFF's `4*pi` normalization.
    pub charge_density: Array1<Real>,
    /// Target magnetization density `dmagx`.
    pub magnetization: Array1<Real>,
    /// 1-based target muffin-tin index `jmtnew`.
    pub muffin_tin_index: usize,
    /// 1-based first target interstitial index `jrinew`.
    pub interstitial_index: usize,
    /// Final potential jump after optional `jumprm == 1` recomputation.
    pub potential_jump: Real,
}

/// Inputs for FEFF `ATOM/potslw.f90` four-point Coulomb integration.
#[derive(Debug, Clone, Copy)]
pub struct CoulombPotentialSlwInput<'a> {
    /// Density-like radial source array `d`.
    pub density: ArrayView1<'a, Real>,
    /// Radial mesh `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid step `dpas`.
    pub delta: Real,
    /// Number of active radial points `np`.
    pub active_len: usize,
}

/// FEFF `potslw` Coulomb-potential result.
#[derive(Debug, Clone, PartialEq)]
pub struct CoulombPotentialSlw {
    /// Potential values `dv`, zero-filled after [`CoulombPotentialSlw::active_len`].
    pub potential: Array1<Real>,
    /// Number of active radial points integrated.
    pub active_len: usize,
}

/// Inputs for FEFF `POT/grids.f90` SCMT complex-energy mesh construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScmtEnergyGridInput {
    /// Core-valence separation energy `ecv`, in Hartrees.
    pub core_valence_energy: Real,
    /// Fermi energy `xmu`, in Hartrees.
    pub fermi_energy: Real,
    /// Length of the output complex-energy table, equivalent to `negx`.
    pub max_points: usize,
    /// FEFF step table length, equivalent to `nflrx`.
    pub step_count: usize,
}

/// FEFF SCMT complex-energy mesh and integration step table.
#[derive(Debug, Clone, PartialEq)]
pub struct ScmtEnergyGrid {
    /// Complex energies `emg`, zero-filled after [`ScmtEnergyGrid::active_len`].
    pub energies: Array1<Complex>,
    /// Integration step table `step`.
    pub steps: Array1<Real>,
    /// Number of active complex-energy points `neg`.
    pub active_len: usize,
    /// Number of initial off-axis points `neg1`.
    pub lower_imaginary_count: usize,
    /// Number of real-step bridge points `neg2`.
    pub real_axis_count: usize,
    /// Number of final off-axis points `neg3`.
    pub upper_imaginary_count: usize,
}

/// Inputs for FEFF `POT/sumax.f90` spherical overlap summation.
#[derive(Debug, Clone, Copy)]
pub struct LoucksSphericalOverlapInput<'a> {
    /// Distance from atom 1 to atom 2, `rn`, in Bohr.
    pub neighbor_distance: Real,
    /// Number of type-2 atoms to add to atom 1, `ann`.
    pub multiplicity: Real,
    /// Potential or density from atom 2 on FEFF's Loucks grid, `aa2`.
    pub source: ArrayView1<'a, Real>,
    /// Existing accumulated values, `aasum`, before this contribution is added.
    pub accumulated: ArrayView1<'a, Real>,
}

/// Updated FEFF Loucks-grid spherical overlap sum.
#[derive(Debug, Clone, PartialEq)]
pub struct LoucksSphericalOverlap {
    /// Accumulated values after adding this `sumax` contribution.
    pub accumulated: Array1<Real>,
    /// Last 1-based grid index updated by the overlap sum.
    pub active_len: usize,
}

/// Explicit neighbor entry for FEFF `POT/movrlp.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MuffinTinOverlapNeighbor {
    /// Neighbor potential index `iphovr`.
    pub source_potential: usize,
    /// Number of equivalent neighbors `nnovr`.
    pub multiplicity: usize,
    /// Neighbor distance `rovr`, in Bohr.
    pub distance: Real,
}

/// Inputs for FEFF `POT/movrlp.f90` overlap-matrix construction.
#[derive(Debug, Clone, Copy)]
pub struct MuffinTinOverlapMatrixInput<'a> {
    /// Highest unique potential index; potentials `0..=highest_potential_index` are included.
    pub highest_potential_index: usize,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, FEFF `rat`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: ArrayView1<'a, usize>,
    /// Multiplicity of each potential in the cluster, FEFF `xnatph`.
    pub potential_multiplicities: ArrayView1<'a, Real>,
    /// Explicit overlap lists by destination potential, FEFF `OVERLAP` data.
    ///
    /// An empty list for a potential uses the geometry-derived neighbor list.
    pub explicit_overlaps: &'a [&'a [MuffinTinOverlapNeighbor]],
    /// 1-based muffin-tin indices `imt`.
    pub muffin_tin_indices: ArrayView1<'a, usize>,
    /// Muffin-tin radii `rmt`.
    pub muffin_tin_radii: ArrayView1<'a, Real>,
    /// Norman radii `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// FEFF `lnear` flags; true forces `rav = ri(imt + 1)`.
    pub near_neighbor_flags: ArrayView1<'a, bool>,
    /// FEFF `inters` selector. Even values use all potentials in the final
    /// equation; odd values use only the absorber. Values `0`, `1`, and others
    /// choose `rnrm`, `(rmt + rnrm) / 2`, and `ri(imt + 1)` for `rav`.
    pub interstitial_selector: usize,
    /// Current interstitial volume before overlap corrections, FEFF `volint`.
    pub interstitial_volume: Real,
}

/// FEFF `movrlp` overlap matrix and corrected volume.
#[derive(Debug, Clone, PartialEq)]
pub struct MuffinTinOverlapMatrix {
    /// FEFF Loucks radial grid `ri`.
    pub radii: Array1<Real>,
    /// LU factors for the active `cmovp` matrix.
    pub lu: Complex32Lu,
    /// Corrected interstitial volume `volint`.
    pub interstitial_volume: Real,
    /// Active matrix order `ncp = novp * (nph + 1) + 1`.
    pub active_order: usize,
}

/// Inputs for FEFF `POT/istval.f90` interstitial shell averaging.
#[derive(Debug, Clone, Copy)]
pub struct InterstitialShellValuesInput<'a> {
    /// Total potential on the Loucks grid, `vtot`.
    pub total_potential: ArrayView1<'a, Real>,
    /// Overlapped density on the Loucks grid, `rholap`.
    pub overlapped_density: ArrayView1<'a, Real>,
    /// Muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
    /// 1-based grid index immediately below `rmt`, `imt`.
    pub muffin_tin_index: usize,
    /// Wigner-Seitz or Norman shell radius `rws`.
    pub wigner_seitz_radius: Real,
    /// 1-based grid index immediately below `rws`, `iws`.
    pub wigner_seitz_index: usize,
}

/// FEFF interstitial potential and density shell averages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterstitialShellValues {
    /// Interstitial potential average `vint`.
    pub interstitial_potential: Real,
    /// Interstitial density average `rhoint`.
    pub interstitial_density: Real,
    /// Shell volume without the common `4*pi` factor.
    pub shell_volume: Real,
}

/// Inputs for FEFF `POT/sidx.f90` overlapped-density index adjustment.
#[derive(Debug, Clone, Copy)]
pub struct OverlapDensityIndicesInput<'a> {
    /// Overlapped density on the Loucks grid, `rholap`.
    pub overlapped_density: ArrayView1<'a, Real>,
    /// Muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
    /// Norman radius `rnrm`.
    pub norman_radius: Real,
}

/// FEFF `sidx` indices and any adjusted radii.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlapDensityIndices {
    /// Last 1-based density index before the zero tail, `imax`.
    pub max_density_index: usize,
    /// 1-based grid index immediately below the final muffin-tin radius, `imt`.
    pub muffin_tin_index: usize,
    /// 1-based grid index immediately below the final Norman radius, `inrm`.
    pub norman_index: usize,
    /// Final muffin-tin radius, moved only when FEFF's valid density tail requires it.
    pub muffin_tin_radius: Real,
    /// Final Norman radius, moved inward when density ends before the input radius.
    pub norman_radius: Real,
    /// Whether `rnrm` was moved inward to the density tail.
    pub moved_norman_radius: bool,
}

/// Inputs for FEFF `POT/frnrm.f90` Norman-radius calculation.
#[derive(Debug, Clone, Copy)]
pub struct NormanRadiusInput<'a> {
    /// Overlapped density `rho` in FEFF's `4*pi*density` convention.
    ///
    /// FEFF `frnrm` only reads the first 251 values even though the surrounding
    /// potential module uses a longer radial grid.
    pub overlapped_density: ArrayView1<'a, Real>,
    /// Atomic number `iz`; the integrated charge inside the Norman sphere is
    /// matched to this value.
    pub atomic_number: usize,
}

/// FEFF Norman radius found from an overlapped density profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormanRadius {
    /// Norman radius `rnrm`, in Bohr.
    pub radius: Real,
    /// 1-based Loucks grid index immediately below the corrected radius.
    pub index: usize,
    /// Fractional FEFF linear interpolation offset from [`NormanRadius::index`].
    pub fraction: Real,
}

/// Inputs for FEFF `POT/fermi.f90` interstitial Fermi-level calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FermiLevelInput {
    /// Interstitial density `rhoint`, in FEFF's `4*pi*density` convention.
    pub interstitial_density: Real,
    /// Interstitial potential `vint`, including ground-state exchange-correlation.
    pub interstitial_potential: Real,
}

/// FEFF interstitial Fermi-level result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FermiLevel {
    /// Fermi level `xmu`, in Hartrees.
    pub chemical_potential: Real,
    /// Density parameter `rs`.
    pub density_parameter: Real,
    /// Interstitial Fermi momentum `xf`.
    pub fermi_momentum: Real,
}

/// Error returned by radial-grid indexing helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GridError {
    /// The radius must be positive and finite before `ln(r)` is meaningful.
    #[error("radius must be positive and finite, got {radius}")]
    InvalidRadius { radius: Real },
    /// The logarithmic grid spacing must be positive and finite.
    #[error("grid delta must be positive and finite, got {delta}")]
    InvalidDelta { delta: Real },
    /// The outer shell radius must be larger than the inner shell radius.
    #[error("outer radius {outer_radius} must be larger than inner radius {inner_radius}")]
    InvalidRadiusOrder {
        inner_radius: Real,
        outer_radius: Real,
    },
    /// FEFF radial-grid indices are 1-based.
    #[error("{name} grid index must be 1-based and positive, got {index}")]
    InvalidGridIndex { name: &'static str, index: usize },
    /// A derived 1-based radial-grid interval must be ordered.
    #[error("grid index range is inverted: lower={lower_index}, upper={upper_index}")]
    InvalidGridIndexRange {
        lower_index: usize,
        upper_index: usize,
    },
    /// Source spinor component arrays must have matching lengths.
    #[error("spinor component length mismatch: large={large_len}, small={small_len}")]
    SpinorLengthMismatch { large_len: usize, small_len: usize },
    /// Source spinor component tables must have matching shapes.
    #[error(
        "spinor component shape mismatch: large=({large_rows},{large_columns}), small=({small_rows},{small_columns})"
    )]
    SpinorShapeMismatch {
        large_rows: usize,
        large_columns: usize,
        small_rows: usize,
        small_columns: usize,
    },
    /// Source potential, density, and magnetization arrays must have matching lengths.
    #[error(
        "potential-grid length mismatch: density={density_len}, potential={potential_len}, magnetization={magnetization_len}"
    )]
    PotentialLengthMismatch {
        density_len: usize,
        potential_len: usize,
        magnetization_len: usize,
    },
    /// FEFF `potslw` density and radius arrays must have matching lengths.
    #[error("Coulomb-grid length mismatch: density={density_len}, radii={radii_len}")]
    CoulombLengthMismatch {
        density_len: usize,
        radii_len: usize,
    },
    /// FEFF overlap source and accumulated arrays must have matching lengths.
    #[error("overlap grid length mismatch: source={source_len}, accumulated={accumulated_len}")]
    OverlapLengthMismatch {
        source_len: usize,
        accumulated_len: usize,
    },
    /// A grid length must be positive.
    #[error("{name} length must be positive")]
    InvalidGridLength { name: &'static str },
    /// A grid length or derived table size overflowed Rust indexing.
    #[error("{name} grid length is too large")]
    GridLengthOverflow { name: &'static str },
    /// A scalar grid parameter must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteScalar { name: &'static str, value: Real },
    /// A scalar grid parameter must be positive and finite.
    #[error("{name} must be positive and finite, got {value}")]
    NonPositiveScalar { name: &'static str, value: Real },
    /// A scalar denominator must be finite and nonzero.
    #[error("{name} must be finite and nonzero, got {value}")]
    ZeroScalar { name: &'static str, value: Real },
    /// A vector must contain enough values for the FEFF loop bounds.
    #[error("{name} length {actual} is shorter than required length {required}")]
    LengthTooShort {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// A matrix must have enough rows and columns for the FEFF loop bounds.
    #[error(
        "{name} shape ({rows},{columns}) is smaller than required ({required_rows},{required_columns})"
    )]
    ShapeTooSmall {
        name: &'static str,
        rows: usize,
        columns: usize,
        required_rows: usize,
        required_columns: usize,
    },
    /// Coordinate rows and atom-potential assignments must have matching lengths.
    #[error("atom potential length {potentials} does not match position rows {positions}")]
    AtomPotentialLengthMismatch { potentials: usize, positions: usize },
    /// FEFF Cartesian coordinate tables must be shaped `(atoms, 3)`.
    #[error("atom_positions must have shape (atoms, 3), got ({rows}, {columns})")]
    InvalidPositionShape { rows: usize, columns: usize },
    /// A potential index referenced outside a potential-indexed table.
    #[error(
        "{name} references potential index {index}, but only {available} potentials are available"
    )]
    InvalidPotentialIndex {
        name: &'static str,
        index: usize,
        available: usize,
    },
    /// FEFF `movrlp` requires at least `novp` radial points inside each muffin tin.
    #[error("{name} for potential {potential} must be at least {minimum}, got {index}")]
    MuffinTinIndexTooSmall {
        name: &'static str,
        potential: usize,
        minimum: usize,
        index: usize,
    },
    /// FEFF `movrlp` detected overlap beyond the supported `novp` window.
    #[error("muffin-tin overlap window is too large for potential pair {left}->{right}")]
    MuffinTinOverlapTooLarge { left: usize, right: usize },
    /// FEFF `movrlp` requires the final pivot to remain in the final row.
    #[error("illegal final pivot in movrlp: expected {expected}, got {actual}")]
    IllegalFinalPivot { expected: usize, actual: usize },
    /// A source or interpolated grid value must be finite.
    #[error("{name}[{index}] must be finite, got {value}")]
    NonFiniteGridValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// FEFF `sidx` could not find a positive density tail from the start index.
    #[error("no density value above {threshold} at or after 1-based index {start_index}")]
    NoActiveDensityTail { start_index: usize, threshold: Real },
    /// Atomic number must be positive for charge-normalized radius searches.
    #[error("atomic number must be positive, got {atomic_number}")]
    InvalidAtomicNumber { atomic_number: usize },
    /// FEFF `frnrm` could not integrate enough charge to determine `rnrm`.
    #[error(
        "could not integrate enough charge for Z={atomic_number}: found {charge_found} by radius {max_radius}"
    )]
    InsufficientNormanCharge {
        atomic_number: usize,
        charge_found: Real,
        max_radius: Real,
    },
    /// The caller's output grid is too short for FEFF's active interpolation range.
    #[error("output grid length {available} is shorter than required active length {required}")]
    OutputGridTooShort { required: usize, available: usize },
    /// The source grid is too short for FEFF's muffin-tin interpolation range.
    #[error("{name} source length {available} is shorter than required length {required}")]
    SourceGridTooShort {
        name: &'static str,
        required: usize,
        available: usize,
    },
    /// FEFF `terp` failed while resampling grid data.
    #[error(transparent)]
    Interpolation(#[from] InterpolationError),
    /// FEFF `somm2` failed while applying a radial-grid endpoint correction.
    #[error(transparent)]
    Quadrature(#[from] QuadratureError),
    /// FEFF-compatible linear algebra failed.
    #[error(transparent)]
    Linalg(#[from] LinalgError),
}

/// Convert energy in Hartrees to FEFF's signed photoelectron wave number.
///
/// This ports `getxk`: `sqrt(2E)` above the edge and `-sqrt(-2E)` below it.
#[must_use]
pub fn wave_number_from_hartree(energy: Real) -> Real {
    let magnitude = (2.0 * energy).abs().sqrt();
    if energy < 0.0 { -magnitude } else { magnitude }
}

/// Return the logarithmic `x = ln(r)` grid coordinate for a 1-based index.
#[must_use]
pub fn loucks_x(index_1based: usize) -> Real {
    radial_x(index_1based, LOUCKS_DELTA)
}

/// Return the radial coordinate for a 1-based Loucks grid index.
#[must_use]
pub fn loucks_radius(index_1based: usize) -> Real {
    loucks_x(index_1based).exp()
}

/// Return the 1-based Loucks grid index immediately below `radius`.
pub fn loucks_index_below(radius: Real) -> Result<usize, GridError> {
    radial_index_below(radius, LOUCKS_DELTA)
}

/// Return the logarithmic `x = ln(r)` grid coordinate for a custom spacing.
#[must_use]
pub fn radial_x(index_1based: usize, delta: Real) -> Real {
    -LOUCKS_X_OFFSET + (index_1based as Real - 1.0) * delta
}

/// Return the radial coordinate for a custom logarithmic spacing.
#[must_use]
pub fn radial_radius(index_1based: usize, delta: Real) -> Real {
    radial_x(index_1based, delta).exp()
}

/// Return the 1-based grid index immediately below `radius` for a custom spacing.
pub fn radial_index_below(radius: Real, delta: Real) -> Result<usize, GridError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(GridError::InvalidRadius { radius });
    }
    if !(delta.is_finite() && delta > 0.0) {
        return Err(GridError::InvalidDelta { delta });
    }
    let index = ((radius.ln() + LOUCKS_X_OFFSET) / delta + 1.0).trunc();
    Ok(index as usize)
}

/// Interpolate one FEFF Dirac spinor pair from `dxorg` to `dxnew`.
///
/// This ports the deterministic numerical part of `COMMON/fixdsp.f90`. FEFF
/// finds the last nonzero source-grid spinor point, adds one source point as
/// the zero boundary, interpolates both components with cubic `terp` on the
/// logarithmic `x` grid, and zero-fills the target tail.
pub fn fix_dirac_spinor_grid(
    input: DiracSpinorGridInput<'_>,
) -> Result<DiracSpinorGrid, GridError> {
    validate_delta(input.original_delta)?;
    validate_delta(input.new_delta)?;
    validate_positive_grid_length("output", input.output_len)?;

    let source_len = input.large_component.len();
    if source_len != input.small_component.len() {
        return Err(GridError::SpinorLengthMismatch {
            large_len: source_len,
            small_len: input.small_component.len(),
        });
    }
    validate_positive_grid_length("source", source_len)?;
    validate_component_values("large_component", input.large_component)?;
    validate_component_values("small_component", input.small_component)?;

    let mut large_component = Array1::<Real>::zeros(input.output_len);
    let mut small_component = Array1::<Real>::zeros(input.output_len);
    let Some(last_nonzero) =
        last_nonzero_spinor_index(input.large_component, input.small_component)
    else {
        return Ok(DiracSpinorGrid {
            large_component,
            small_component,
            active_len: 0,
        });
    };

    let source_window_len = (last_nonzero + 2).min(source_len);
    let source_x = (1..=source_window_len)
        .map(|index| radial_x(index, input.original_delta))
        .collect::<Vec<_>>();
    let source_large = input
        .large_component
        .iter()
        .take(source_window_len)
        .copied()
        .collect::<Vec<_>>();
    let source_small = input
        .small_component
        .iter()
        .take(source_window_len)
        .copied()
        .collect::<Vec<_>>();

    let rmax = radial_radius(source_window_len, input.original_delta);
    let active_len = radial_index_below(rmax, input.new_delta)?;
    if active_len > input.output_len {
        return Err(GridError::OutputGridTooShort {
            required: active_len,
            available: input.output_len,
        });
    }

    for target_index in 1..=active_len {
        let x = radial_x(target_index, input.new_delta);
        let index = target_index - 1;
        large_component[index] = terp(&source_x, &source_large, 3, x)?.value;
        small_component[index] = terp(&source_x, &source_small, 3, x)?.value;
    }

    Ok(DiracSpinorGrid {
        large_component,
        small_component,
        active_len,
    })
}

/// Interpolate FEFF Dirac spinor orbital columns from `dxorg` to `dxnew`.
///
/// This ports the deterministic resampling behavior of `COMMON/fixdsx.f90`.
/// Each orbital column is treated independently with the same zero-tail
/// detection and cubic interpolation used by [`fix_dirac_spinor_grid`].
pub fn fix_dirac_spinor_orbitals_grid(
    input: DiracSpinorOrbitalsGridInput<'_>,
) -> Result<DiracSpinorOrbitalsGrid, GridError> {
    validate_delta(input.original_delta)?;
    validate_delta(input.new_delta)?;
    validate_positive_grid_length("output", input.output_len)?;

    let large_shape = input.large_components.shape();
    let small_shape = input.small_components.shape();
    if large_shape != small_shape {
        return Err(GridError::SpinorShapeMismatch {
            large_rows: large_shape[0],
            large_columns: large_shape[1],
            small_rows: small_shape[0],
            small_columns: small_shape[1],
        });
    }
    validate_positive_grid_length("source", large_shape[0])?;
    validate_positive_grid_length("orbital", large_shape[1])?;

    let orbital_count = large_shape[1];
    let mut large_components = Array2::<Real>::zeros((input.output_len, orbital_count).f());
    let mut small_components = Array2::<Real>::zeros((input.output_len, orbital_count).f());
    let mut active_lengths = Array1::<usize>::zeros(orbital_count);

    for orbital in 0..orbital_count {
        let spinor = fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: input.original_delta,
            new_delta: input.new_delta,
            large_component: input.large_components.column(orbital),
            small_component: input.small_components.column(orbital),
            output_len: input.output_len,
        })?;
        large_components
            .column_mut(orbital)
            .assign(&spinor.large_component);
        small_components
            .column_mut(orbital)
            .assign(&spinor.small_component);
        active_lengths[orbital] = spinor.active_len;
    }

    Ok(DiracSpinorOrbitalsGrid {
        large_components,
        small_components,
        active_lengths,
    })
}

/// Interpolate FEFF potential, charge density, and magnetization onto a target grid.
///
/// This ports the deterministic numerical behavior of `COMMON/fixvar.f90`.
/// Values through the first target interstitial point are cubic-interpolated on
/// FEFF's logarithmic `x` grid, optional potential jumps are applied exactly as
/// `jumprm` specifies, charge density is divided by `4*pi`, and the remaining
/// tail is filled with interstitial values.
pub fn fix_potential_grid(input: PotentialGridInput<'_>) -> Result<PotentialGrid, GridError> {
    validate_delta(input.original_delta)?;
    validate_delta(input.new_delta)?;
    validate_positive_grid_length("output", input.output_len)?;
    validate_finite_scalar("interstitial_potential", input.interstitial_potential)?;
    validate_finite_scalar("interstitial_density", input.interstitial_density)?;
    validate_finite_scalar("potential_jump", input.potential_jump)?;

    let density_len = input.electron_density.len();
    let potential_len = input.total_potential.len();
    let magnetization_len = input.magnetization.len();
    if density_len != potential_len || density_len != magnetization_len {
        return Err(GridError::PotentialLengthMismatch {
            density_len,
            potential_len,
            magnetization_len,
        });
    }
    validate_positive_grid_length("source", density_len)?;
    validate_component_values("electron_density", input.electron_density)?;
    validate_component_values("total_potential", input.total_potential)?;
    validate_component_values("magnetization", input.magnetization)?;

    let muffin_tin_index_source =
        radial_index_below(input.muffin_tin_radius, input.original_delta)?;
    let interstitial_index_source = muffin_tin_index_source + 1;
    let density_window_len = interstitial_index_source + 1;
    ensure_source_length("total_potential", interstitial_index_source, potential_len)?;
    ensure_source_length("electron_density", density_window_len, density_len)?;
    ensure_source_length("magnetization", density_window_len, magnetization_len)?;

    let muffin_tin_index = radial_index_below(input.muffin_tin_radius, input.new_delta)?;
    let interstitial_index = muffin_tin_index + 1;
    if interstitial_index > input.output_len {
        return Err(GridError::OutputGridTooShort {
            required: interstitial_index,
            available: input.output_len,
        });
    }

    let source_x = (1..=density_window_len)
        .map(|index| radial_x(index, input.original_delta))
        .collect::<Vec<_>>();
    let source_density = input
        .electron_density
        .iter()
        .take(density_window_len)
        .copied()
        .collect::<Vec<_>>();
    let source_potential = input
        .total_potential
        .iter()
        .take(interstitial_index_source)
        .copied()
        .collect::<Vec<_>>();
    let source_magnetization = input
        .magnetization
        .iter()
        .take(density_window_len)
        .copied()
        .collect::<Vec<_>>();

    let radii = (1..=input.output_len)
        .map(|index| radial_radius(index, input.new_delta))
        .collect::<Array1<_>>();
    let mut total_potential = Array1::<Real>::zeros(input.output_len);
    let mut charge_density = Array1::<Real>::zeros(input.output_len);
    let mut magnetization = Array1::<Real>::zeros(input.output_len);

    for target_index in 1..=interstitial_index {
        let x = radial_x(target_index, input.new_delta);
        let index = target_index - 1;
        total_potential[index] = terp(
            &source_x[..interstitial_index_source],
            &source_potential,
            3,
            x,
        )?
        .value;
        charge_density[index] = terp(&source_x, &source_density, 3, x)?.value;
        magnetization[index] = terp(&source_x, &source_magnetization, 3, x)?.value;
    }

    let mut potential_jump = input.potential_jump;
    if input.jump_mode == 1 {
        let muffin_tin_potential = terp(
            &source_x[..interstitial_index_source],
            &source_potential,
            3,
            input.muffin_tin_radius.ln(),
        )?
        .value;
        potential_jump = input.interstitial_potential - muffin_tin_potential;
    }
    if input.jump_mode > 0 {
        total_potential
            .iter_mut()
            .take(interstitial_index)
            .for_each(|value| *value += potential_jump);
    }

    charge_density
        .iter_mut()
        .take(interstitial_index)
        .for_each(|value| *value /= 4.0 * PI);

    total_potential
        .iter_mut()
        .zip(charge_density.iter_mut())
        .zip(magnetization.iter_mut())
        .skip(interstitial_index)
        .for_each(|((potential, density), moment)| {
            *potential = input.interstitial_potential;
            *density = input.interstitial_density / (4.0 * PI);
            *moment = 0.0;
        });

    Ok(PotentialGrid {
        radii,
        total_potential,
        charge_density,
        magnetization,
        muffin_tin_index,
        interstitial_index,
        potential_jump,
    })
}

/// Integrate a radial density into a Coulomb potential using FEFF `potslw`.
///
/// This ports `ATOM/potslw.f90`, a four-point integration stencil used by the
/// potential module's Coulomb update. FEFF only defines values through `np`; the
/// Rust result preserves the caller's grid length and zero-fills the inactive
/// tail.
pub fn coulomb_potential_slw(
    input: CoulombPotentialSlwInput<'_>,
) -> Result<CoulombPotentialSlw, GridError> {
    validate_delta(input.delta)?;

    let density_len = input.density.len();
    let radii_len = input.radii.len();
    if density_len != radii_len {
        return Err(GridError::CoulombLengthMismatch {
            density_len,
            radii_len,
        });
    }
    validate_positive_grid_length("density", density_len)?;
    validate_source_len_at_least("active", input.active_len, 3)?;
    ensure_source_length("density", input.active_len, density_len)?;
    validate_component_prefix_values("density", input.density, input.active_len)?;
    validate_positive_radii(input.radii, input.active_len)?;

    let mut potential = Array1::<Real>::zeros(density_len);
    let mut work = Array1::<Real>::zeros(density_len);
    let scale = input.delta / 24.0;
    for index in 0..input.active_len {
        potential[index] = input.density[index] * input.radii[index];
    }

    let grid_ratio = input.delta.exp();
    let grid_ratio2 = grid_ratio * grid_ratio;
    work[1] = input.radii[0] * (input.density[1] - input.density[0] * grid_ratio2)
        / (12.0 * (grid_ratio - 1.0));
    work[0] = potential[0] / 3.0 - work[1] / grid_ratio2;
    work[1] = potential[1] / 3.0 - work[1] * grid_ratio2;

    let last_inner = input.active_len - 2;
    for index in 2..=last_inner {
        work[index] = work[index - 1]
            + scale
                * (13.0 * (potential[index] + potential[index - 1])
                    - (potential[index - 2] + potential[index + 1]));
    }

    work[input.active_len - 1] = work[last_inner];
    potential[last_inner] = work[last_inner];
    potential[input.active_len - 1] = work[last_inner];
    for fortran_i in 3..=last_inner + 1 {
        let index = input.active_len - fortran_i;
        potential[index] = potential[index + 1] / grid_ratio
            + scale
                * (13.0 * (work[index + 1] / grid_ratio + work[index])
                    - (work[index + 2] / grid_ratio2 + work[index - 1] * grid_ratio));
    }
    potential[0] = potential[2] / grid_ratio2
        + input.delta * (work[0] + 4.0 * work[1] / grid_ratio + work[2] / grid_ratio2) / 3.0;

    potential
        .iter_mut()
        .zip(input.radii.iter())
        .take(input.active_len)
        .for_each(|(potential, radius)| *potential /= radius);

    Ok(CoulombPotentialSlw {
        potential,
        active_len: input.active_len,
    })
}

/// Build FEFF's SCMT complex-energy contour from `ecv` to `xmu`.
///
/// This ports `POT/grids.f90`. FEFF first creates a short vertical line above
/// `ecv`, then a real-axis bridge that retains the initial imaginary part, and
/// finally a descending set of points above `xmu`. The Rust version preserves
/// FEFF's count and rounding rules while validating that the caller-provided
/// table sizes are large enough.
pub fn scmt_energy_grid(input: ScmtEnergyGridInput) -> Result<ScmtEnergyGrid, GridError> {
    validate_finite_scalar("core_valence_energy", input.core_valence_energy)?;
    validate_finite_scalar("fermi_energy", input.fermi_energy)?;
    let energy_span = input.fermi_energy - input.core_valence_energy;
    validate_finite_scalar("energy_span", energy_span)?;
    validate_positive_grid_length("energy", input.max_points)?;
    validate_positive_grid_length("step", input.step_count)?;

    let lower_imaginary_count = input
        .step_count
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "step" })?
        / 2;
    let upper_imaginary_count = input.step_count - 1;
    let minimum_points = lower_imaginary_count
        .checked_mul(2)
        .and_then(|value| value.checked_add(upper_imaginary_count))
        .ok_or(GridError::OutputGridTooShort {
            required: usize::MAX,
            available: input.max_points,
        })?;
    if input.max_points < minimum_points {
        return Err(GridError::OutputGridTooShort {
            required: minimum_points,
            available: input.max_points,
        });
    }

    let real_axis_max = input.max_points - lower_imaginary_count - upper_imaginary_count;
    let minimum_imaginary = 0.05 / FEFF_HARTREE_EV;
    let mut energies = Array1::<Complex>::zeros(input.max_points);
    let mut steps = Array1::<Real>::zeros(input.step_count);

    for index in 1..=lower_imaginary_count {
        let imaginary = minimum_imaginary * square_index_as_real("step", index)?;
        energies[index - 1] = Complex::new(input.core_valence_energy, imaginary);
    }
    steps[input.step_count - 1] = energies[lower_imaginary_count - 1].im / 4.0;

    let bridge_step_guess = energies[lower_imaginary_count - 1].im / 4.0;
    let rounded_bridge_points = (energy_span / bridge_step_guess).round();
    let mut real_axis_count = if rounded_bridge_points <= 0.0 {
        0
    } else if rounded_bridge_points >= real_axis_max as Real {
        real_axis_max
    } else {
        rounded_bridge_points as usize
    };
    if real_axis_count < lower_imaginary_count {
        real_axis_count = lower_imaginary_count;
    }

    let real_step = energy_span / real_axis_count as Real;
    for index in lower_imaginary_count + 1..=lower_imaginary_count + real_axis_count {
        energies[index - 1] = energies[index - 2] + Complex::new(real_step, 0.0);
    }

    let active_len = lower_imaginary_count + real_axis_count + upper_imaginary_count;
    for index in 1..=upper_imaginary_count {
        let imaginary = minimum_imaginary * square_index_as_real("step", index + 1)? / 4.0;
        steps[index - 1] = imaginary / 4.0;
        energies[active_len - index] = Complex::new(input.fermi_energy, imaginary);
    }

    Ok(ScmtEnergyGrid {
        energies,
        steps,
        active_len,
        lower_imaginary_count,
        real_axis_count,
        upper_imaginary_count,
    })
}

/// Add one FEFF `sumax` spherical overlap contribution on the Loucks grid.
///
/// This ports `POT/sumax.f90`, used by FEFF's overlapped potential/density
/// setup. The input and accumulated arrays use the fixed Loucks spacing
/// `delta = 0.05`; only grid points through the neighbor distance are updated,
/// matching FEFF's `jtop = ii(rn)` behavior.
pub fn sum_loucks_spherical_overlap(
    input: LoucksSphericalOverlapInput<'_>,
) -> Result<LoucksSphericalOverlap, GridError> {
    if !(input.neighbor_distance.is_finite() && input.neighbor_distance > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.neighbor_distance,
        });
    }
    validate_finite_scalar("multiplicity", input.multiplicity)?;

    let source_len = input.source.len();
    let accumulated_len = input.accumulated.len();
    if source_len != accumulated_len {
        return Err(GridError::OverlapLengthMismatch {
            source_len,
            accumulated_len,
        });
    }
    validate_positive_grid_length("source", source_len)?;
    validate_component_values("source", input.source)?;
    validate_component_values("accumulated", input.accumulated)?;

    let cutoff_index = loucks_index_below(SUMAX_WIGNER_SEITZ_RADIUS)?;
    let active_len = loucks_index_below(input.neighbor_distance)?;
    ensure_source_length("source", cutoff_index, source_len)?;
    ensure_source_length("accumulated", active_len, accumulated_len)?;

    let source = input.source.iter().copied().collect::<Vec<_>>();
    let mut accumulated = input.accumulated.iter().copied().collect::<Array1<_>>();
    if active_len == 0 {
        return Ok(LoucksSphericalOverlap {
            accumulated,
            active_len,
        });
    }

    let top_x = loucks_x(cutoff_index);

    for index in 1..=active_len {
        let x = loucks_x(index);
        let radius = x.exp();
        let contribution = sumax_integral_contribution(
            input.neighbor_distance,
            input.multiplicity,
            &source,
            top_x,
            radius,
        )?;
        accumulated[index - 1] += contribution;
    }

    Ok(LoucksSphericalOverlap {
        accumulated,
        active_len,
    })
}

/// Construct FEFF's muffin-tin overlap matrix from `POT/movrlp.f90`.
///
/// FEFF stores only a moving `novp = 50` radial window for each potential and
/// appends one equation for the interstitial potential. This function builds
/// that active matrix, applies FEFF-compatible single-complex LU factorization,
/// and returns the factors for downstream `ovp2mt`-style solves.
pub fn muffin_tin_overlap_matrix(
    input: MuffinTinOverlapMatrixInput<'_>,
) -> Result<MuffinTinOverlapMatrix, GridError> {
    validate_muffin_tin_overlap_input(input)?;

    let potential_count = input.highest_potential_index + 1;
    let active_order = MOVRLP_NOVP
        .checked_mul(potential_count)
        .and_then(|value| value.checked_add(1))
        .ok_or(GridError::GridLengthOverflow { name: "movrlp" })?;
    let radii = (1..=251).map(loucks_radius).collect::<Array1<_>>();
    let grid_half_step = (LOUCKS_DELTA / 2.0).exp();
    let radius_mode = (input.interstitial_selector - (input.interstitial_selector % 2)) / 2;
    let absorber_only = input.interstitial_selector % 2 == 1;

    let mut matrix = Array2::<Complex32>::zeros((active_order, active_order));
    for row in 0..active_order {
        for column in 0..(active_order - 1) {
            matrix[(row, column)] = Complex32::new(0.0, 0.0);
        }
        matrix[(row, row)] = Complex32::new(1.0, 0.0);
        matrix[(row, active_order - 1)] = Complex32::new(0.01, 0.0);
    }

    let mut bmat = Array2::<f32>::zeros((potential_count, active_order - 1));
    let mut interstitial_volume = input.interstitial_volume;
    validate_finite_scalar("interstitial_volume", interstitial_volume)?;
    let mut atom_count = 0.0;

    for target in 0..potential_count {
        let rav = movrlp_average_radius(input, &radii, target, radius_mode)?;
        let neighbors = movrlp_neighbors(input, target)?;
        for neighbor in neighbors {
            let source = neighbor.source_potential;
            let distance = neighbor.distance;
            let multiplicity = neighbor.multiplicity as Real;
            let pair = MovrlpPair {
                target,
                source,
                distance,
                multiplicity,
            };

            if distance < input.muffin_tin_radii[target] + input.muffin_tin_radii[source] {
                interstitial_volume += input.potential_multiplicities[target]
                    * multiplicity
                    * sphere_overlap_cap_volume(
                        input.muffin_tin_radii[target],
                        input.muffin_tin_radii[source],
                        distance,
                    )?;
            }

            if rav + input.muffin_tin_radii[source] > distance {
                movrlp_fill_boundary_row(input, &radii, &mut bmat, pair, rav, grid_half_step)?;
            }

            if input.muffin_tin_radii[target] + input.muffin_tin_radii[source] > distance {
                movrlp_fill_overlap_matrix(input, &radii, &mut matrix, pair, grid_half_step)?;
            }
        }
        atom_count += input.potential_multiplicities[target];
    }
    validate_nonzero_finite_scalar("atom_count", atom_count)?;

    if absorber_only {
        for column in 0..(active_order - 1) {
            matrix[(active_order - 1, column)] += Complex32::new(bmat[(0, column)], 0.0);
        }
    } else {
        for potential in 0..potential_count {
            let weight = (input.potential_multiplicities[potential] / atom_count) as f32;
            for column in 0..(active_order - 1) {
                matrix[(active_order - 1, column)] +=
                    Complex32::new(weight * bmat[(potential, column)], 0.0);
            }
        }
    }

    let lu = complex32_lu_factor(matrix.view())?;
    let final_pivot =
        lu.pivots()
            .get(active_order - 1)
            .copied()
            .ok_or(GridError::LengthTooShort {
                name: "movrlp_pivots",
                required: active_order,
                actual: lu.pivots().len(),
            })?;
    if final_pivot != active_order {
        return Err(GridError::IllegalFinalPivot {
            expected: active_order,
            actual: final_pivot,
        });
    }

    Ok(MuffinTinOverlapMatrix {
        radii,
        lu,
        interstitial_volume,
        active_order,
    })
}

/// Volume of one FEFF spherical-overlap cap from `POT/istprm.f90` `calcvl`.
///
/// `sphere_radius` is the radius of the sphere whose cap is being measured,
/// `other_radius` is the radius of the overlapping sphere, and
/// `center_distance` is the distance between sphere centers. FEFF callers use
/// this only after confirming the spheres overlap; this function preserves the
/// algebraic `calcvl` formula and validates only finite positive inputs.
pub fn sphere_overlap_cap_volume(
    sphere_radius: Real,
    other_radius: Real,
    center_distance: Real,
) -> Result<Real, GridError> {
    validate_positive_finite_scalar("sphere_radius", sphere_radius)?;
    validate_positive_finite_scalar("other_radius", other_radius)?;
    validate_positive_finite_scalar("center_distance", center_distance)?;

    let plane_distance = (sphere_radius.powi(2) - other_radius.powi(2) + center_distance.powi(2))
        / (2.0 * center_distance);
    let cap_height = sphere_radius - plane_distance;
    let volume = PI / 3.0 * cap_height.powi(2) * (3.0 * sphere_radius - cap_height);
    validate_finite_scalar("sphere_overlap_cap_volume", volume)?;
    Ok(volume)
}

/// Total lens volume of two overlapping spheres using FEFF `calcvl` caps.
pub fn sphere_overlap_lens_volume(
    radius_a: Real,
    radius_b: Real,
    center_distance: Real,
) -> Result<Real, GridError> {
    Ok(
        sphere_overlap_cap_volume(radius_a, radius_b, center_distance)?
            + sphere_overlap_cap_volume(radius_b, radius_a, center_distance)?,
    )
}

/// Average FEFF potential and overlapped density over an interstitial shell.
///
/// This ports `POT/istval.f90`. FEFF integrates `r**3 * value` over the
/// logarithmic Loucks coordinate and divides by `(rws**3 - rmt**3) / 3`, leaving
/// out the common `4*pi` factor in both the integral and the shell volume.
pub fn interstitial_shell_values(
    input: InterstitialShellValuesInput<'_>,
) -> Result<InterstitialShellValues, GridError> {
    if !(input.muffin_tin_radius.is_finite() && input.muffin_tin_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.muffin_tin_radius,
        });
    }
    if !(input.wigner_seitz_radius.is_finite() && input.wigner_seitz_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.wigner_seitz_radius,
        });
    }
    if input.wigner_seitz_radius <= input.muffin_tin_radius {
        return Err(GridError::InvalidRadiusOrder {
            inner_radius: input.muffin_tin_radius,
            outer_radius: input.wigner_seitz_radius,
        });
    }
    validate_grid_index("muffin_tin", input.muffin_tin_index)?;
    validate_grid_index("wigner_seitz", input.wigner_seitz_index)?;
    if input.wigner_seitz_index < input.muffin_tin_index {
        return Err(GridError::InvalidGridIndexRange {
            lower_index: input.muffin_tin_index,
            upper_index: input.wigner_seitz_index,
        });
    }
    validate_positive_grid_length("total_potential", input.total_potential.len())?;
    validate_positive_grid_length("overlapped_density", input.overlapped_density.len())?;
    validate_component_values("total_potential", input.total_potential)?;
    validate_component_values("overlapped_density", input.overlapped_density)?;

    let required =
        input
            .wigner_seitz_index
            .checked_add(1)
            .ok_or(GridError::GridLengthOverflow {
                name: "interstitial",
            })?;
    ensure_source_length("total_potential", required, input.total_potential.len())?;
    ensure_source_length(
        "overlapped_density",
        required,
        input.overlapped_density.len(),
    )?;

    let shell_volume = (input.wigner_seitz_radius.powi(3) - input.muffin_tin_radius.powi(3)) / 3.0;
    let potential_integral = interstitial_shell_integral(
        input.total_potential,
        input.muffin_tin_radius,
        input.muffin_tin_index,
        input.wigner_seitz_radius,
        input.wigner_seitz_index,
    )?;
    let density_integral = interstitial_shell_integral(
        input.overlapped_density,
        input.muffin_tin_radius,
        input.muffin_tin_index,
        input.wigner_seitz_radius,
        input.wigner_seitz_index,
    )?;

    Ok(InterstitialShellValues {
        interstitial_potential: potential_integral / shell_volume,
        interstitial_density: density_integral / shell_volume,
        shell_volume,
    })
}

/// Locate FEFF overlapped-density tail indices and adjust radii when needed.
///
/// This ports the defined behavior of `POT/sidx.f90`. FEFF scans `rholap`
/// from `imt = ii(rmt)` until the first value at or below `1.0e-5`, then moves
/// the Norman radius inward if its index lies beyond the last positive-density
/// point. The original Fortran leaves `imax` undefined when the first scanned
/// density value is already below cutoff; Rust reports that case as
/// [`GridError::NoActiveDensityTail`].
pub fn overlap_density_indices(
    input: OverlapDensityIndicesInput<'_>,
) -> Result<OverlapDensityIndices, GridError> {
    if !(input.muffin_tin_radius.is_finite() && input.muffin_tin_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.muffin_tin_radius,
        });
    }
    if !(input.norman_radius.is_finite() && input.norman_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.norman_radius,
        });
    }
    validate_positive_grid_length("overlapped_density", input.overlapped_density.len())?;
    validate_component_values("overlapped_density", input.overlapped_density)?;

    let muffin_tin_index = feff_legacy_loucks_index_below(input.muffin_tin_radius)?;
    let initial_norman_index = feff_legacy_loucks_index_below(input.norman_radius)?;
    validate_grid_index("muffin_tin", muffin_tin_index)?;
    validate_grid_index("norman", initial_norman_index)?;
    ensure_source_length(
        "overlapped_density",
        muffin_tin_index,
        input.overlapped_density.len(),
    )?;

    let mut max_density_index = None;
    for index in muffin_tin_index..=input.overlapped_density.len() {
        if view_value(input.overlapped_density, index, "overlapped_density")? <= SIDX_DENSITY_CUTOFF
        {
            break;
        }
        max_density_index = Some(index);
    }
    let max_density_index = max_density_index.ok_or(GridError::NoActiveDensityTail {
        start_index: muffin_tin_index,
        threshold: SIDX_DENSITY_CUTOFF,
    })?;

    let (norman_index, norman_radius, moved_norman_radius) =
        if initial_norman_index > max_density_index {
            (
                max_density_index,
                feff_legacy_loucks_radius(max_density_index),
                true,
            )
        } else {
            (initial_norman_index, input.norman_radius, false)
        };

    Ok(OverlapDensityIndices {
        max_density_index,
        muffin_tin_index,
        norman_index,
        muffin_tin_radius: input.muffin_tin_radius,
        norman_radius,
        moved_norman_radius,
    })
}

/// Find FEFF's Norman radius from an overlapped density profile.
///
/// This ports `POT/frnrm.f90`. FEFF integrates `rho * r**2 dr`, with `rho`
/// already stored as `4*pi*density`, until the accumulated charge reaches the
/// atom's `Z`. The first pass follows FEFF's hand-coded Simpson recurrence, then
/// the returned radius is refined by the same `somm2` endpoint correction used
/// in the original routine. The radial grid intentionally preserves FEFF's
/// default-real `xx.f90` constants before widening to double precision.
pub fn norman_radius_from_density(input: NormanRadiusInput<'_>) -> Result<NormanRadius, GridError> {
    if input.atomic_number == 0 {
        return Err(GridError::InvalidAtomicNumber {
            atomic_number: input.atomic_number,
        });
    }
    ensure_source_length(
        "overlapped_density",
        FRNRM_DENSITY_POINTS,
        input.overlapped_density.len(),
    )?;
    let density = input
        .overlapped_density
        .iter()
        .take(FRNRM_DENSITY_POINTS)
        .copied()
        .collect::<Vec<_>>();
    validate_slice_values("overlapped_density", &density)?;
    let radii = (1..=FRNRM_DENSITY_POINTS)
        .map(feff_legacy_loucks_radius)
        .collect::<Vec<_>>();
    let density_moments = density
        .iter()
        .zip(radii.iter())
        .map(|(&rho, &radius)| rho * radius * radius * radius)
        .collect::<Vec<_>>();

    let target_charge = input.atomic_number as Real;
    let scan = frnrm_initial_scan(&density, &radii, &density_moments, target_charge)?;
    let (index, mut fraction) = scan.crossing.ok_or(GridError::InsufficientNormanCharge {
        atomic_number: input.atomic_number,
        charge_found: scan.charge,
        max_radius: radii[FRNRM_DENSITY_POINTS - 1],
    })?;

    let mut radius = radii[index - 1] * (1.0 + fraction * FRNRM_LITERAL_DELTA);
    let correction_len = frnrm_correction_len(radius)?;
    ensure_source_length("overlapped_density", correction_len, FRNRM_DENSITY_POINTS)?;
    ensure_source_length("norman_correction", index + 1, correction_len)?;

    let correction_radii = &radii[..correction_len];
    let correction_values = correction_radii
        .iter()
        .zip(density.iter())
        .map(|(&ri, &rho)| rho * ri * ri)
        .collect::<Vec<_>>();

    let first_charge = somm2(
        correction_radii,
        &correction_values,
        FRNRM_LITERAL_DELTA,
        2.0,
        radius,
        0,
    )?;
    let first_delta = first_charge - target_charge;
    let density_at_radius =
        (1.0 - fraction) * correction_values[index - 1] + fraction * correction_values[index];
    validate_nonzero_finite_scalar("norman_correction_density", density_at_radius)?;

    let second_fraction = fraction - first_delta / density_at_radius;
    if (second_fraction - fraction).abs() > FRNRM_CORRECTION_THRESHOLD {
        radius = radii[index - 1] * (1.0 + second_fraction * FRNRM_LITERAL_DELTA);
        let second_charge = somm2(
            correction_radii,
            &correction_values,
            FRNRM_LITERAL_DELTA,
            2.0,
            radius,
            0,
        )?;
        let second_delta = second_charge - target_charge;
        let delta_difference = second_delta - first_delta;
        validate_nonzero_finite_scalar("norman_correction_delta", delta_difference)?;
        fraction = second_fraction - second_delta * (second_fraction - fraction) / delta_difference;
    } else {
        fraction = second_fraction;
    }

    Ok(NormanRadius {
        radius: radii[index - 1] * (1.0 + fraction * FRNRM_LITERAL_DELTA),
        index,
        fraction,
    })
}

/// Calculate FEFF's interstitial Fermi level from density and potential.
///
/// This ports `POT/fermi.f90`. FEFF stores `rhoint` as `4*pi*density`, so the
/// density parameter is `rs = (3 / rhoint)^(1/3)`, the Fermi momentum is
/// `xf = fa / rs`, and the chemical potential is `xmu = vint + xf**2 / 2`.
pub fn interstitial_fermi_level(input: FermiLevelInput) -> Result<FermiLevel, GridError> {
    validate_positive_finite_scalar("interstitial_density", input.interstitial_density)?;
    validate_finite_scalar("interstitial_potential", input.interstitial_potential)?;

    let density_parameter = (3.0 / input.interstitial_density).powf(1.0 / 3.0);
    let fermi_momentum = FEFF_FERMI_MOMENTUM_FACTOR / density_parameter;
    let chemical_potential = input.interstitial_potential + fermi_momentum.powi(2) / 2.0;

    Ok(FermiLevel {
        chemical_potential,
        density_parameter,
        fermi_momentum,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FrnrmInitialScan {
    crossing: Option<(usize, Real)>,
    charge: Real,
}

fn frnrm_initial_scan(
    density: &[Real],
    radii: &[Real],
    density_moments: &[Real],
    target_charge: Real,
) -> Result<FrnrmInitialScan, GridError> {
    let mut charge =
        (9.0 * density_moments[0] + 28.0 * density_moments[1] + 23.0 * density_moments[2]) / 480.0;
    charge += frnrm_initial_origin_correction(density, radii)?;

    let mut left = density_moments[3];
    let mut center = density_moments[4];
    let mut right = density_moments[5];

    for index in 7..=FRNRM_NRPTX {
        let far_left = left;
        left = center;
        center = right;
        right = if index <= FRNRM_DENSITY_POINTS {
            density_moments[index - 1]
        } else {
            0.0
        };
        let previous_charge = charge;
        charge += (13.0 * (center + left) - far_left - right) / 480.0;
        if charge >= target_charge {
            let increment = charge - previous_charge;
            validate_nonzero_finite_scalar("norman_charge_increment", increment)?;
            return Ok(FrnrmInitialScan {
                crossing: Some((index - 2, (target_charge - previous_charge) / increment)),
                charge,
            });
        }
    }

    Ok(FrnrmInitialScan {
        crossing: None,
        charge,
    })
}

fn frnrm_initial_origin_correction(density: &[Real], radii: &[Real]) -> Result<Real, GridError> {
    let d1 = 3.0;
    let delta = FRNRM_LITERAL_DELTA.exp() - 1.0;
    let second_coefficient =
        radii[0] / (d1 * (d1 + 1.0) * delta * ((d1 - 1.0) * FRNRM_LITERAL_DELTA).exp());
    let first_coefficient = radii[0] * (1.0 + 1.0 / (delta * (d1 + 1.0))) / d1;
    let correction = first_coefficient * density[0] * radii[0] * radii[0]
        - second_coefficient * density[1] * radii[1] * radii[1];
    validate_finite_scalar("norman_origin_correction", correction)?;
    Ok(correction)
}

fn frnrm_correction_len(radius: Real) -> Result<usize, GridError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(GridError::InvalidRadius { radius });
    }
    let grid_index =
        fortran_truncated_index((radius.ln() + FRNRM_LITERAL_OFFSET) / FRNRM_LITERAL_DELTA + 2.0);
    grid_index
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow {
            name: "norman_correction",
        })
}

fn sumax_integral_contribution(
    neighbor_distance: Real,
    multiplicity: Real,
    source: &[Real],
    top_x: Real,
    radius: Real,
) -> Result<Real, GridError> {
    let lower_radius = neighbor_distance - radius;
    if lower_radius <= 0.0 {
        return Ok(0.0);
    }

    let lower_x = lower_radius.ln();
    if lower_x >= top_x {
        return Ok(0.0);
    }

    let mut integral = 0.0;
    let mut lower_index =
        fortran_truncated_index(2.0 + 20.0 * (lower_x + SUMAX_LITERAL_OFFSET)).max(1);
    let mut lower_grid_x = sumax_literal_x(lower_index);
    if lower_index >= 2 {
        let cap_width = lower_grid_x - lower_x;
        let lower_value = source_value(source, lower_index, "source")?;
        let previous_value = source_value(source, lower_index - 1, "source")?;
        integral += 0.5
            * cap_width
            * (lower_value * (2.0 - 20.0 * cap_width) * (2.0 * lower_grid_x).exp()
                + 20.0
                    * cap_width
                    * previous_value
                    * (2.0 * (lower_grid_x - SUMAX_LITERAL_DELTA)).exp());
    }

    let upper_x = (neighbor_distance + radius).ln();
    let upper_index = if upper_x >= top_x {
        radial_index_below(SUMAX_WIGNER_SEITZ_RADIUS, LOUCKS_DELTA)?
    } else {
        let index = fortran_truncated_index(1.0 + 20.0 * (upper_x + SUMAX_LITERAL_OFFSET));
        if index < lower_index {
            let near_zero = source_value(source, index, "source")?
                * (2.0 * (lower_grid_x - SUMAX_LITERAL_DELTA)).exp();
            let lower_value =
                source_value(source, lower_index, "source")? * (2.0 * lower_grid_x).exp();
            let upper_interp = near_zero
                + 20.0 * (lower_value - near_zero) * (upper_x - lower_grid_x + SUMAX_LITERAL_DELTA);
            let lower_interp = near_zero
                + 20.0 * (lower_value - near_zero) * (lower_x - lower_grid_x + SUMAX_LITERAL_DELTA);
            integral = 0.5 * (lower_interp + upper_interp) * (upper_x - lower_x);
            return Ok(0.5 * integral * multiplicity / (neighbor_distance * radius));
        }

        let upper_grid_x = sumax_literal_x(index);
        let cap_width = upper_x - upper_grid_x;
        let upper_value = source_value(source, index, "source")?;
        let next_value = source_value(source, index + 1, "source")?;
        integral += 0.5
            * cap_width
            * (upper_value * (2.0 - 20.0 * cap_width) * (2.0 * upper_grid_x).exp()
                + next_value
                    * 20.0
                    * cap_width
                    * (2.0 * (upper_grid_x + SUMAX_LITERAL_DELTA)).exp());
        index
    };

    while upper_index > lower_index {
        let current = source_value(source, lower_index, "source")? * (2.0 * lower_grid_x).exp();
        let next = source_value(source, lower_index + 1, "source")?
            * (2.0 * (lower_grid_x + SUMAX_LITERAL_DELTA)).exp();
        integral += 0.5 * (current + next) * SUMAX_LITERAL_DELTA;
        lower_index += 1;
        if lower_index < upper_index {
            lower_grid_x += SUMAX_LITERAL_DELTA;
        }
    }

    Ok(0.5 * integral * multiplicity / (neighbor_distance * radius))
}

fn movrlp_average_radius(
    input: MuffinTinOverlapMatrixInput<'_>,
    radii: &Array1<Real>,
    potential: usize,
    radius_mode: usize,
) -> Result<Real, GridError> {
    let radius = if radius_mode == 1 {
        (input.muffin_tin_radii[potential] + input.norman_radii[potential]) / 2.0
    } else if radius_mode == 0 {
        input.norman_radii[potential]
    } else {
        radii[movrlp_radii_index_after_muffin(input.muffin_tin_indices[potential])?]
    };

    if input.near_neighbor_flags[potential] {
        Ok(radii[movrlp_radii_index_after_muffin(input.muffin_tin_indices[potential])?])
    } else {
        Ok(radius)
    }
}

fn movrlp_radii_index_after_muffin(muffin_tin_index: usize) -> Result<usize, GridError> {
    muffin_tin_index
        .checked_add(1)
        .filter(|&index| index <= 251)
        .map(|index| index - 1)
        .ok_or(GridError::SourceGridTooShort {
            name: "radii",
            required: muffin_tin_index.saturating_add(1),
            available: 251,
        })
}

fn movrlp_neighbors(
    input: MuffinTinOverlapMatrixInput<'_>,
    target: usize,
) -> Result<Vec<MuffinTinOverlapNeighbor>, GridError> {
    let explicit = input.explicit_overlaps[target];
    if !explicit.is_empty() {
        return Ok(explicit.to_vec());
    }

    let representative = input.representative_atoms[target];
    let center = [
        input.atom_positions[(representative, 0)],
        input.atom_positions[(representative, 1)],
        input.atom_positions[(representative, 2)],
    ];
    let mut neighbors = Vec::new();
    for atom in 0..input.atom_positions.nrows() {
        if atom == representative {
            continue;
        }
        let position = [
            input.atom_positions[(atom, 0)],
            input.atom_positions[(atom, 1)],
            input.atom_positions[(atom, 2)],
        ];
        neighbors.push(MuffinTinOverlapNeighbor {
            source_potential: input.atom_potentials[atom],
            multiplicity: 1,
            distance: distance_between(center, position),
        });
    }
    Ok(neighbors)
}

#[derive(Debug, Clone, Copy)]
struct MovrlpPair {
    target: usize,
    source: usize,
    distance: Real,
    multiplicity: Real,
}

fn movrlp_fill_boundary_row(
    input: MuffinTinOverlapMatrixInput<'_>,
    radii: &Array1<Real>,
    bmat: &mut Array2<f32>,
    pair: MovrlpPair,
    average_radius: Real,
    grid_half_step: Real,
) -> Result<(), GridError> {
    let check_index = loucks_index_below(pair.distance - average_radius)?;
    if input.muffin_tin_indices[pair.source].saturating_sub(check_index) >= MOVRLP_NOVP - 1 {
        return Err(GridError::MuffinTinOverlapTooLarge {
            left: pair.target,
            right: pair.source,
        });
    }
    let start = movrlp_window_start(input.muffin_tin_indices[pair.source], pair.source)?;
    for radial in start..=input.muffin_tin_indices[pair.source] {
        let radius = radii[radial - 1];
        let mut r1 = radius / grid_half_step;
        let mut r2 = radius * grid_half_step;
        if radial == input.muffin_tin_indices[pair.source] {
            r2 = input.muffin_tin_radii[pair.source];
            r1 = (r1 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                - input.muffin_tin_radii[pair.source])
                / 2.0;
        }
        if radial + 1 == input.muffin_tin_indices[pair.source] {
            r2 = (r2 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                - input.muffin_tin_radii[pair.source])
                / 2.0;
        }
        if r2 + average_radius < pair.distance {
            continue;
        }

        if r1 + average_radius < pair.distance {
            let mut fraction = (pair.distance - average_radius - r1) / (r2 - r1);
            r1 = pair.distance - average_radius;
            let contribution = (r2.powi(2) - r1.powi(2)) / (4.0 * pair.distance * average_radius)
                * pair.multiplicity;
            let neighbor_index = if radial == input.muffin_tin_indices[pair.source] {
                radial - 1
            } else {
                radial + 1
            };
            fraction *= (r2 - radius) / (radii[neighbor_index - 1] - radius);
            let column = pair.source * MOVRLP_NOVP + radial - start;
            bmat[(pair.target, column)] += movrlp_real32("bmat", contribution * (1.0 - fraction))?;
            let column = pair.source * MOVRLP_NOVP + neighbor_index - start;
            bmat[(pair.target, column)] += movrlp_real32("bmat", contribution * fraction)?;
        } else {
            let contribution = (r2.powi(2) - r1.powi(2)) / (4.0 * pair.distance * average_radius)
                * pair.multiplicity;
            let column = pair.source * MOVRLP_NOVP + radial - start;
            bmat[(pair.target, column)] += movrlp_real32("bmat", contribution)?;
        }
    }
    Ok(())
}

fn movrlp_fill_overlap_matrix(
    input: MuffinTinOverlapMatrixInput<'_>,
    radii: &Array1<Real>,
    matrix: &mut Array2<Complex32>,
    pair: MovrlpPair,
    grid_half_step: Real,
) -> Result<(), GridError> {
    let check_target = loucks_index_below(pair.distance - input.muffin_tin_radii[pair.source])?;
    let check_source = loucks_index_below(pair.distance - input.muffin_tin_radii[pair.target])?;
    if input.muffin_tin_indices[pair.target].saturating_sub(check_target) >= MOVRLP_NOVP - 1
        || input.muffin_tin_indices[pair.source].saturating_sub(check_source) >= MOVRLP_NOVP - 1
    {
        return Err(GridError::MuffinTinOverlapTooLarge {
            left: pair.target,
            right: pair.source,
        });
    }

    let target_start = movrlp_window_start(input.muffin_tin_indices[pair.target], pair.target)?;
    let source_start = movrlp_window_start(input.muffin_tin_indices[pair.source], pair.source)?;
    for target_radial in target_start..=input.muffin_tin_indices[pair.target] {
        let target_radius = radii[target_radial - 1];
        let mut target_r1 = target_radius / grid_half_step;
        let mut target_r2 = target_radius * grid_half_step;
        if target_radial == input.muffin_tin_indices[pair.target] {
            target_r2 = input.muffin_tin_radii[pair.target];
            target_r1 = (target_r1 + 2.0 * radii[input.muffin_tin_indices[pair.target] - 1]
                - input.muffin_tin_radii[pair.target])
                / 2.0;
        }
        if target_radial + 1 == input.muffin_tin_indices[pair.target] {
            target_r2 = (target_r2 + 2.0 * radii[input.muffin_tin_indices[pair.target] - 1]
                - input.muffin_tin_radii[pair.target])
                / 2.0;
        }
        let target_column = pair.target * MOVRLP_NOVP + target_radial - target_start;

        for source_radial in source_start..=input.muffin_tin_indices[pair.source] {
            let source_radius = radii[source_radial - 1];
            let mut source_r1 = source_radius / grid_half_step;
            let mut source_r2 = source_radius * grid_half_step;
            if source_radial == input.muffin_tin_indices[pair.source] {
                source_r2 = input.muffin_tin_radii[pair.source];
                source_r1 = (source_r1 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                    - input.muffin_tin_radii[pair.source])
                    / 2.0;
            }
            if source_radial + 1 == input.muffin_tin_indices[pair.source] {
                source_r2 = (source_r2 + 2.0 * radii[input.muffin_tin_indices[pair.source] - 1]
                    - input.muffin_tin_radii[pair.source])
                    / 2.0;
            }
            if source_r2 + target_r2 < pair.distance {
                continue;
            }

            let mut contribution = sphere_overlap_lens_volume(target_r2, source_r2, pair.distance)?;
            if target_r1 + source_r2 > pair.distance {
                contribution -= sphere_overlap_lens_volume(target_r1, source_r2, pair.distance)?;
            }
            if target_r2 + source_r1 > pair.distance {
                contribution -= sphere_overlap_lens_volume(target_r2, source_r1, pair.distance)?;
            }
            if target_r1 + source_r1 > pair.distance {
                contribution += sphere_overlap_lens_volume(target_r1, source_r1, pair.distance)?;
            }
            contribution = contribution
                / (4.0 / 3.0 * PI * (target_r2.powi(3) - target_r1.powi(3)))
                * pair.multiplicity;

            if source_r1 + target_r2 < pair.distance {
                let mut fraction =
                    (pair.distance - target_radius - source_r1) / (source_r2 - source_r1);
                let neighbor_index = if source_radial == input.muffin_tin_indices[pair.source] {
                    source_radial - 1
                } else {
                    source_radial + 1
                };
                fraction *=
                    (source_r2 - source_radius) / (radii[neighbor_index - 1] - source_radius);
                let column = pair.source * MOVRLP_NOVP + source_radial - source_start;
                matrix[(target_column, column)] += Complex32::new(
                    movrlp_real32("cmovp", contribution * (1.0 - fraction))?,
                    0.0,
                );
                let column = pair.source * MOVRLP_NOVP + neighbor_index - source_start;
                matrix[(target_column, column)] +=
                    Complex32::new(movrlp_real32("cmovp", contribution * fraction)?, 0.0);
            } else {
                let column = pair.source * MOVRLP_NOVP + source_radial - source_start;
                matrix[(target_column, column)] +=
                    Complex32::new(movrlp_real32("cmovp", contribution)?, 0.0);
            }
        }
    }
    Ok(())
}

fn movrlp_window_start(muffin_tin_index: usize, potential: usize) -> Result<usize, GridError> {
    if muffin_tin_index < MOVRLP_NOVP {
        Err(GridError::MuffinTinIndexTooSmall {
            name: "muffin_tin_indices",
            potential,
            minimum: MOVRLP_NOVP,
            index: muffin_tin_index,
        })
    } else {
        Ok(muffin_tin_index - MOVRLP_NOVP + 1)
    }
}

fn movrlp_real32(name: &'static str, value: Real) -> Result<f32, GridError> {
    validate_finite_scalar(name, value)?;
    let narrowed = value as f32;
    if narrowed.is_finite() {
        Ok(narrowed)
    } else {
        Err(GridError::NonFiniteScalar { name, value })
    }
}

fn validate_muffin_tin_overlap_input(
    input: MuffinTinOverlapMatrixInput<'_>,
) -> Result<(), GridError> {
    let potential_count = input
        .highest_potential_index
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "potential" })?;
    ensure_len(
        "representative_atoms",
        input.representative_atoms.len(),
        potential_count,
    )?;
    ensure_len(
        "potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    ensure_len(
        "explicit_overlaps",
        input.explicit_overlaps.len(),
        potential_count,
    )?;
    ensure_len(
        "muffin_tin_indices",
        input.muffin_tin_indices.len(),
        potential_count,
    )?;
    ensure_len(
        "muffin_tin_radii",
        input.muffin_tin_radii.len(),
        potential_count,
    )?;
    ensure_len("norman_radii", input.norman_radii.len(), potential_count)?;
    ensure_len(
        "near_neighbor_flags",
        input.near_neighbor_flags.len(),
        potential_count,
    )?;
    validate_position_table(input.atom_positions)?;
    if input.atom_potentials.len() != input.atom_positions.nrows() {
        return Err(GridError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            positions: input.atom_positions.nrows(),
        });
    }
    validate_usize_potential_values("atom_potentials", input.atom_potentials, potential_count)?;
    validate_usize_potential_values(
        "representative_atoms",
        input.representative_atoms,
        input.atom_positions.nrows(),
    )?;
    validate_real_values("potential_multiplicities", input.potential_multiplicities)?;
    validate_real_values("muffin_tin_radii", input.muffin_tin_radii)?;
    validate_real_values("norman_radii", input.norman_radii)?;
    for potential in 0..potential_count {
        validate_positive_finite_scalar(
            "potential_multiplicities",
            input.potential_multiplicities[potential],
        )?;
        validate_positive_finite_scalar("muffin_tin_radii", input.muffin_tin_radii[potential])?;
        validate_positive_finite_scalar("norman_radii", input.norman_radii[potential])?;
        if input.muffin_tin_indices[potential] < MOVRLP_NOVP {
            return Err(GridError::MuffinTinIndexTooSmall {
                name: "muffin_tin_indices",
                potential,
                minimum: MOVRLP_NOVP,
                index: input.muffin_tin_indices[potential],
            });
        }
        if input.muffin_tin_indices[potential] >= 251 {
            return Err(GridError::SourceGridTooShort {
                name: "radii",
                required: input.muffin_tin_indices[potential] + 1,
                available: 251,
            });
        }
        for neighbor in input.explicit_overlaps[potential] {
            if neighbor.source_potential >= potential_count {
                return Err(GridError::InvalidPotentialIndex {
                    name: "explicit_overlaps.source_potential",
                    index: neighbor.source_potential,
                    available: potential_count,
                });
            }
            if neighbor.multiplicity == 0 {
                return Err(GridError::InvalidGridIndex {
                    name: "explicit_overlaps.multiplicity",
                    index: 0,
                });
            }
            validate_positive_finite_scalar("explicit_overlaps.distance", neighbor.distance)?;
        }
    }
    Ok(())
}

fn interstitial_shell_integral(
    values: ArrayView1<'_, Real>,
    muffin_tin_radius: Real,
    muffin_tin_index: usize,
    wigner_seitz_radius: Real,
    wigner_seitz_index: usize,
) -> Result<Real, GridError> {
    let trapezoid_sum = (muffin_tin_index..wigner_seitz_index).try_fold(0.0, |sum, index| {
        let right = radius_cubed_grid_value(values, index + 1, "grid")?;
        let left = radius_cubed_grid_value(values, index, "grid")?;
        Ok::<_, GridError>(sum + 0.5 * (right + left) * LOUCKS_DELTA)
    })?;
    let upper_cap = interstitial_shell_cap(values, wigner_seitz_radius, wigner_seitz_index)?;
    let lower_cap = interstitial_shell_cap(values, muffin_tin_radius, muffin_tin_index)?;
    Ok(trapezoid_sum + upper_cap - lower_cap)
}

fn interstitial_shell_cap(
    values: ArrayView1<'_, Real>,
    radius: Real,
    index: usize,
) -> Result<Real, GridError> {
    let cap_width = radius.ln() - loucks_x(index);
    let ratio = cap_width / LOUCKS_DELTA;
    let left = radius_cubed_grid_value(values, index, "grid")?;
    let right = radius_cubed_grid_value(values, index + 1, "grid")?;
    Ok(0.5 * cap_width * ((2.0 - ratio) * left + ratio * right))
}

fn validate_delta(delta: Real) -> Result<(), GridError> {
    if delta.is_finite() && delta > 0.0 {
        Ok(())
    } else {
        Err(GridError::InvalidDelta { delta })
    }
}

fn validate_positive_grid_length(name: &'static str, len: usize) -> Result<(), GridError> {
    if len > 0 {
        Ok(())
    } else {
        Err(GridError::InvalidGridLength { name })
    }
}

fn ensure_len(name: &'static str, actual: usize, required: usize) -> Result<(), GridError> {
    if actual >= required {
        Ok(())
    } else {
        Err(GridError::LengthTooShort {
            name,
            required,
            actual,
        })
    }
}

fn ensure_shape(
    name: &'static str,
    shape: &[usize],
    required_rows: usize,
    required_columns: usize,
) -> Result<(), GridError> {
    let rows = shape[0];
    let columns = shape[1];
    if rows >= required_rows && columns >= required_columns {
        Ok(())
    } else {
        Err(GridError::ShapeTooSmall {
            name,
            rows,
            columns,
            required_rows,
            required_columns,
        })
    }
}

fn validate_grid_index(name: &'static str, index: usize) -> Result<(), GridError> {
    if index > 0 {
        Ok(())
    } else {
        Err(GridError::InvalidGridIndex { name, index })
    }
}

fn validate_finite_scalar(name: &'static str, value: Real) -> Result<(), GridError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GridError::NonFiniteScalar { name, value })
    }
}

fn validate_positive_finite_scalar(name: &'static str, value: Real) -> Result<(), GridError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(GridError::NonPositiveScalar { name, value })
    }
}

fn validate_nonzero_finite_scalar(name: &'static str, value: Real) -> Result<(), GridError> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(GridError::ZeroScalar { name, value })
    }
}

fn validate_real_values(name: &'static str, values: ArrayView1<'_, Real>) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_component_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_position_table(positions: ArrayView2<'_, Real>) -> Result<(), GridError> {
    if positions.ncols() != 3 {
        return Err(GridError::InvalidPositionShape {
            rows: positions.nrows(),
            columns: positions.ncols(),
        });
    }
    ensure_shape("atom_positions", positions.shape(), positions.nrows(), 3)?;
    for ((atom_index, axis), &value) in positions.indexed_iter() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue {
                name: "atom_positions",
                index: atom_index * 3 + axis,
                value,
            });
        }
    }
    Ok(())
}

fn validate_usize_potential_values(
    name: &'static str,
    values: ArrayView1<'_, usize>,
    available: usize,
) -> Result<(), GridError> {
    for &index in values {
        if index >= available {
            return Err(GridError::InvalidPotentialIndex {
                name,
                index,
                available,
            });
        }
    }
    Ok(())
}

fn validate_component_prefix_values(
    name: &'static str,
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), GridError> {
    for (index, &value) in values.iter().take(active_len).enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_slice_values(name: &'static str, values: &[Real]) -> Result<(), GridError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GridError::NonFiniteGridValue { name, index, value });
        }
    }
    Ok(())
}

fn validate_positive_radii(
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), GridError> {
    for &radius in values.iter().take(active_len) {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(GridError::InvalidRadius { radius });
        }
    }
    Ok(())
}

fn validate_source_len_at_least(
    name: &'static str,
    available: usize,
    required: usize,
) -> Result<(), GridError> {
    if available >= required {
        Ok(())
    } else {
        Err(GridError::SourceGridTooShort {
            name,
            required,
            available,
        })
    }
}

fn ensure_source_length(
    name: &'static str,
    required: usize,
    available: usize,
) -> Result<(), GridError> {
    if available >= required {
        Ok(())
    } else {
        Err(GridError::SourceGridTooShort {
            name,
            required,
            available,
        })
    }
}

fn square_index_as_real(name: &'static str, index: usize) -> Result<Real, GridError> {
    index
        .checked_mul(index)
        .map(|value| value as Real)
        .ok_or(GridError::GridLengthOverflow { name })
}

fn fortran_truncated_index(value: Real) -> usize {
    if value <= 0.0 {
        0
    } else {
        value.trunc() as usize
    }
}

fn sumax_literal_x(index_1based: usize) -> Real {
    SUMAX_LITERAL_DELTA * (index_1based as Real - 1.0) - SUMAX_LITERAL_OFFSET
}

fn feff_legacy_loucks_x(index_1based: usize) -> Real {
    sumax_literal_x(index_1based)
}

fn feff_legacy_loucks_radius(index_1based: usize) -> Real {
    feff_legacy_loucks_x(index_1based).exp()
}

fn feff_legacy_loucks_index_below(radius: Real) -> Result<usize, GridError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(GridError::InvalidRadius { radius });
    }
    Ok(fortran_truncated_index(
        (radius.ln() + SUMAX_LITERAL_OFFSET) / SUMAX_LITERAL_DELTA + 1.0,
    ))
}

fn radius_cubed_grid_value(
    values: ArrayView1<'_, Real>,
    index_1based: usize,
    name: &'static str,
) -> Result<Real, GridError> {
    Ok(loucks_radius(index_1based).powi(3) * view_value(values, index_1based, name)?)
}

fn view_value(
    values: ArrayView1<'_, Real>,
    index_1based: usize,
    name: &'static str,
) -> Result<Real, GridError> {
    if index_1based == 0 || index_1based > values.len() {
        Err(GridError::SourceGridTooShort {
            name,
            required: index_1based.max(1),
            available: values.len(),
        })
    } else {
        Ok(values[index_1based - 1])
    }
}

fn source_value(
    values: &[Real],
    index_1based: usize,
    name: &'static str,
) -> Result<Real, GridError> {
    if index_1based == 0 || index_1based > values.len() {
        Err(GridError::SourceGridTooShort {
            name,
            required: index_1based.max(1),
            available: values.len(),
        })
    } else {
        Ok(values[index_1based - 1])
    }
}

fn last_nonzero_spinor_index(
    large_component: ArrayView1<'_, Real>,
    small_component: ArrayView1<'_, Real>,
) -> Option<usize> {
    large_component
        .iter()
        .zip(small_component.iter())
        .enumerate()
        .rev()
        .find_map(|(index, (&large, &small))| {
            (large.abs() >= SPINOR_ZERO_THRESHOLD || small.abs() >= SPINOR_ZERO_THRESHOLD)
                .then_some(index)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, ShapeBuilder};

    #[test]
    fn converts_energy_to_signed_wave_number() {
        assert_eq!(wave_number_from_hartree(2.0), 2.0);
        assert_eq!(wave_number_from_hartree(-2.0), -2.0);
        assert_eq!(wave_number_from_hartree(0.0), 0.0);
    }

    #[test]
    fn reproduces_loucks_log_grid_points() {
        assert!((loucks_x(1) + 8.8).abs() < 1.0e-12);
        assert!((loucks_x(2) + 8.75).abs() < 1.0e-12);
        assert!((loucks_radius(1) - (-8.8_f64).exp()).abs() < 1.0e-16);
    }

    #[test]
    fn maps_radius_to_index_below() -> Result<(), GridError> {
        let radius = loucks_radius(42);
        assert_eq!(loucks_index_below(radius)?, 42);

        let midpoint = (loucks_x(42) + 0.5 * LOUCKS_DELTA).exp();
        assert_eq!(loucks_index_below(midpoint)?, 42);
        Ok(())
    }

    #[test]
    fn rejects_invalid_radius_or_delta() {
        assert!(matches!(
            loucks_index_below(0.0),
            Err(GridError::InvalidRadius { .. })
        ));
        assert!(matches!(
            radial_index_below(1.0, 0.0),
            Err(GridError::InvalidDelta { .. })
        ));
    }

    #[test]
    fn fix_dirac_spinor_grid_matches_feff_fixdsp_reference() -> Result<(), GridError> {
        let mut large = vec![0.0; 251];
        let mut small = vec![0.0; 251];
        for i in 1..=80 {
            let i_real = i as Real;
            large[i - 1] = (0.1 * i_real).sin() * (-0.02 * i_real).exp() + 0.001 * i_real;
            small[i - 1] = (0.08 * i_real).cos() * (-0.015 * i_real).exp() - 0.0005 * i_real;
        }
        let large = Array1::from_vec(large);
        let small = Array1::from_vec(small);

        let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: large.view(),
            small_component: small.view(),
            output_len: 180,
        })?;

        assert_eq!(result.active_len, 161);
        assert_spinor_value(
            &result,
            1,
            0.098_856_582_548_901_49,
            0.981_461_262_295_415_9,
        );
        assert_spinor_value(&result, 2, 0.146_525_001_614_189, 0.969_970_868_040_543_4);
        assert_spinor_value(
            &result,
            3,
            0.192_879_394_911_354_22,
            0.957_050_307_749_104_5,
        );
        assert_spinor_value(&result, 10, 0.473_738_853_193_487_96, 0.830_355_320_320_026);
        assert_spinor_value(
            &result,
            80,
            -0.310_280_702_093_608_3,
            -0.562_325_207_440_241_6,
        );
        assert_spinor_value(
            &result,
            120,
            -0.008_407_166_503_866_128,
            0.021_105_137_955_943_806,
        );
        assert_spinor_value(
            &result,
            160,
            0.191_266_534_139_204_64,
            0.176_750_359_590_577_94,
        );
        assert_spinor_value(&result, 161, 0.0, 0.0);
        assert_spinor_value(&result, 180, 0.0, 0.0);
        Ok(())
    }

    #[test]
    fn fix_dirac_spinor_grid_zero_fills_empty_spinor() -> Result<(), GridError> {
        let large = Array1::zeros(251);
        let small = Array1::zeros(251);

        let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: large.view(),
            small_component: small.view(),
            output_len: 16,
        })?;

        assert_eq!(result.active_len, 0);
        assert!(result.large_component.iter().all(|&value| value == 0.0));
        assert!(result.small_component.iter().all(|&value| value == 0.0));
        Ok(())
    }

    #[test]
    fn fix_dirac_spinor_grid_rejects_invalid_inputs() {
        let large = Array1::zeros(4);
        let small = Array1::zeros(3);
        assert_eq!(
            fix_dirac_spinor_grid(DiracSpinorGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_component: large.view(),
                small_component: small.view(),
                output_len: 16,
            }),
            Err(GridError::SpinorLengthMismatch {
                large_len: 4,
                small_len: 3,
            })
        );

        let nonfinite = Array1::from_vec(vec![0.0, f64::NAN, 0.0, 0.0]);
        let zeros = Array1::zeros(4);
        assert!(matches!(
            fix_dirac_spinor_grid(DiracSpinorGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_component: nonfinite.view(),
                small_component: zeros.view(),
                output_len: 16,
            }),
            Err(GridError::NonFiniteGridValue {
                name: "large_component",
                index: 1,
                ..
            })
        ));

        assert_eq!(
            fix_dirac_spinor_grid(DiracSpinorGridInput {
                original_delta: 0.0,
                new_delta: 0.025,
                large_component: zeros.view(),
                small_component: zeros.view(),
                output_len: 16,
            }),
            Err(GridError::InvalidDelta { delta: 0.0 })
        );
    }

    #[test]
    fn fix_dirac_spinor_orbitals_grid_matches_feff_fixdsx_reference() -> Result<(), GridError> {
        let mut large = Array2::<Real>::zeros((251, 4).f());
        let mut small = Array2::<Real>::zeros((251, 4).f());
        for i in 1..=40 {
            let i_real = i as Real;
            large[(i - 1, 0)] = (0.07 * i_real).sin() * (-0.01 * i_real).exp();
            small[(i - 1, 0)] = (0.05 * i_real).cos() * (-0.02 * i_real).exp();
        }
        for i in 1..=75 {
            let i_real = i as Real;
            large[(i - 1, 2)] = 0.2 * (0.11 * i_real).sin() + 0.002 * i_real;
            small[(i - 1, 2)] = 0.3 * (0.09 * i_real).cos() - 0.001 * i_real;
        }
        for i in 1..=5 {
            let i_real = i as Real;
            large[(i - 1, 3)] = 0.05 * i_real;
            small[(i - 1, 3)] = -0.04 * i_real;
        }

        let result = fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_components: large.view(),
            small_components: small.view(),
            output_len: 260,
        })?;

        assert_eq!(result.large_components.shape(), &[260, 4]);
        assert_eq!(result.large_components.strides(), &[1, 260]);
        assert_eq!(result.active_lengths.to_vec(), vec![81, 0, 151, 11]);
        assert_orbital_value(
            &result,
            1,
            1,
            0.069_246_904_378_467_77,
            0.978_973_680_203_922_3,
        );
        assert_orbital_value(&result, 81, 1, 0.0, 0.0);
        assert_orbital_value(&result, 82, 1, 0.0, 0.0);
        assert_orbital_value(&result, 1, 2, 0.0, 0.0);
        assert_orbital_value(&result, 100, 2, 0.0, 0.0);
        assert_orbital_value(
            &result,
            1,
            3,
            0.023_955_660_167_434_965,
            0.297_785_819_903_598_26,
        );
        assert_orbital_value(
            &result,
            150,
            3,
            0.228_834_221_332_933_4,
            0.130_219_461_349_623_98,
        );
        assert_orbital_value(&result, 151, 3, 0.0, 0.0);
        assert_orbital_value(&result, 1, 4, 0.05, -0.04);
        assert_orbital_value(&result, 11, 4, 0.0, 0.0);
        assert_orbital_value(&result, 12, 4, 0.0, 0.0);
        Ok(())
    }

    #[test]
    fn fix_dirac_spinor_orbitals_grid_rejects_shape_mismatch() {
        let large = Array2::<Real>::zeros((4, 2));
        let small = Array2::<Real>::zeros((4, 3));

        assert_eq!(
            fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
                original_delta: 0.05,
                new_delta: 0.025,
                large_components: large.view(),
                small_components: small.view(),
                output_len: 16,
            }),
            Err(GridError::SpinorShapeMismatch {
                large_rows: 4,
                large_columns: 2,
                small_rows: 4,
                small_columns: 3,
            })
        );
    }

    #[test]
    fn fix_potential_grid_matches_feff_fixvar_nojump_reference() -> Result<(), GridError> {
        let result = run_sample_potential_grid(0, 0.125)?;

        assert_eq!(result.muffin_tin_index, 121);
        assert_eq!(result.interstitial_index, 122);
        assert_close(result.potential_jump, 0.125);
        assert_potential_value(
            &result,
            1,
            1.507_330_750_954_765e-4,
            -1.935_022_498_312_550_8,
            3.208_561_106_457_231e-2,
            6.991_469_396_917_269e-4,
        );
        assert_potential_value(
            &result,
            2,
            1.545_489_010_585_363e-4,
            -1.927_550_614_879_478_5,
            3.221_287_457_552_784e-2,
            1.047_124_998_462_492_8e-3,
        );
        assert_potential_value(
            &result,
            60,
            6.588_596_634_060_351e-4,
            -1.512_010_471_807_159,
            3.892_714_881_737_269e-2,
            3.404_343_790_471_498e-3,
        );
        assert_potential_value(
            &result,
            121,
            3.027_554_745_375_812_7e-3,
            -1.097_815_545_411_376,
            4.308_030_270_342_605e-2,
            -1.595_986_127_361_670_4e-2,
        );
        assert_potential_value(
            &result,
            122,
            3.104_197_658_649_308_7e-3,
            -1.091_039_022_689_942_5,
            4.312_310_486_424_463e-2,
            -1.593_525_191_056_929e-2,
        );
        assert_potential_value(
            &result,
            123,
            3.182_780_796_509_667e-3,
            -0.75,
            2.228_169_203_286_535e-2,
            0.0,
        );
        assert_potential_value(
            &result,
            180,
            1.323_355_009_654_092_8e-2,
            -0.75,
            2.228_169_203_286_535e-2,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn fix_potential_grid_matches_feff_fixvar_auto_jump_reference() -> Result<(), GridError> {
        let result = run_sample_potential_grid(1, 0.125)?;

        assert_close(result.potential_jump, 3.423_945_657_555_365e-1);
        assert_potential_value(
            &result,
            1,
            1.507_330_750_954_765e-4,
            -1.592_627_932_557_014_3,
            3.208_561_106_457_231e-2,
            6.991_469_396_917_269e-4,
        );
        assert_potential_value(
            &result,
            2,
            1.545_489_010_585_363e-4,
            -1.585_156_049_123_942,
            3.221_287_457_552_784e-2,
            1.047_124_998_462_492_8e-3,
        );
        assert_potential_value(
            &result,
            60,
            6.588_596_634_060_351e-4,
            -1.169_615_906_051_622_5,
            3.892_714_881_737_269e-2,
            3.404_343_790_471_498e-3,
        );
        assert_potential_value(
            &result,
            121,
            3.027_554_745_375_812_7e-3,
            -7.554_209_796_558_395e-1,
            4.308_030_270_342_605e-2,
            -1.595_986_127_361_670_4e-2,
        );
        assert_potential_value(
            &result,
            122,
            3.104_197_658_649_308_7e-3,
            -7.486_444_569_344_06e-1,
            4.312_310_486_424_463e-2,
            -1.593_525_191_056_929e-2,
        );
        assert_potential_value(
            &result,
            123,
            3.182_780_796_509_667e-3,
            -0.75,
            2.228_169_203_286_535e-2,
            0.0,
        );
        assert_potential_value(
            &result,
            180,
            1.323_355_009_654_092_8e-2,
            -0.75,
            2.228_169_203_286_535e-2,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn fix_potential_grid_rejects_invalid_inputs() {
        let density = Array1::<Real>::zeros(4);
        let potential = Array1::<Real>::zeros(5);
        let magnetization = Array1::<Real>::zeros(4);
        assert!(matches!(
            fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: radial_radius(2, 0.05),
                electron_density: density.view(),
                total_potential: potential.view(),
                magnetization: magnetization.view(),
                interstitial_potential: -0.75,
                interstitial_density: 0.28,
                original_delta: 0.05,
                new_delta: 0.025,
                jump_mode: 0,
                potential_jump: 0.125,
                output_len: 8,
            }),
            Err(GridError::PotentialLengthMismatch {
                density_len: 4,
                potential_len: 5,
                magnetization_len: 4,
            })
        ));

        assert!(matches!(
            fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: radial_radius(2, 0.05),
                electron_density: density.view(),
                total_potential: density.view(),
                magnetization: density.view(),
                interstitial_potential: -0.75,
                interstitial_density: 0.28,
                original_delta: 0.05,
                new_delta: 0.025,
                jump_mode: 0,
                potential_jump: f64::NAN,
                output_len: 8,
            }),
            Err(GridError::NonFiniteScalar {
                name: "potential_jump",
                ..
            })
        ));

        let nonfinite_density = Array1::from_vec(vec![0.0, f64::INFINITY, 0.0, 0.0]);
        assert!(matches!(
            fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: radial_radius(2, 0.05),
                electron_density: nonfinite_density.view(),
                total_potential: density.view(),
                magnetization: density.view(),
                interstitial_potential: -0.75,
                interstitial_density: 0.28,
                original_delta: 0.05,
                new_delta: 0.025,
                jump_mode: 0,
                potential_jump: 0.125,
                output_len: 8,
            }),
            Err(GridError::NonFiniteGridValue {
                name: "electron_density",
                index: 1,
                ..
            })
        ));

        assert_eq!(
            fix_potential_grid(PotentialGridInput {
                muffin_tin_radius: radial_radius(6, 0.05),
                electron_density: density.view(),
                total_potential: density.view(),
                magnetization: density.view(),
                interstitial_potential: -0.75,
                interstitial_density: 0.28,
                original_delta: 0.05,
                new_delta: 0.025,
                jump_mode: 0,
                potential_jump: 0.125,
                output_len: 16,
            }),
            Err(GridError::SourceGridTooShort {
                name: "total_potential",
                required: 7,
                available: 4,
            })
        );
    }

    #[test]
    fn coulomb_potential_slw_matches_feff_potslw_long_reference() -> Result<(), GridError> {
        let radii = (1..=251)
            .map(|index| (-8.8 + 0.05 * (index - 1) as Real).exp())
            .collect::<Array1<_>>();
        let density = (1..=251)
            .map(|index| {
                let radius = radii[index - 1];
                (0.015 * index as Real + 0.002 * (index % 5) as Real) * radius * radius
            })
            .collect::<Array1<_>>();

        let result = coulomb_potential_slw(CoulombPotentialSlwInput {
            density: density.view(),
            radii: radii.view(),
            delta: 0.05,
            active_len: 251,
        })?;

        assert_eq!(result.active_len, 251);
        assert_close(result.potential[0], 2.668_218_180_150_684e3);
        assert_close(result.potential[1], 2.668_218_180_150_706e3);
        assert_close(result.potential[2], 2.668_218_180_150_729e3);
        assert_close(result.potential[63], 2.668_218_178_677_377_6e3);
        assert_close(result.potential[127], 2.668_216_102_613_817_4e3);
        assert_close(result.potential[250], 1.715_191_594_675_573_5e3);
        Ok(())
    }

    #[test]
    fn coulomb_potential_slw_matches_feff_potslw_short_reference() -> Result<(), GridError> {
        let radii = (1..=251)
            .map(|index| 0.2 + 0.04 * index as Real)
            .collect::<Array1<_>>();
        let mut density = (1..=251)
            .map(|index| 0.03 * index as Real + 0.001 * (index * index) as Real)
            .collect::<Array1<_>>();
        density[8] = Real::NAN;

        let result = coulomb_potential_slw(CoulombPotentialSlwInput {
            density: density.view(),
            radii: radii.view(),
            delta: 0.07,
            active_len: 8,
        })?;

        let expected = [
            5.611_527_327_297_855e-2,
            5.254_100_056_797_561_5e-2,
            4.981_230_728_432_756e-2,
            4.749_189_715_919_542_6e-2,
            4.522_151_874_857_458e-2,
            4.271_756_035_420_61e-2,
            3.967_487_563_606_690_6e-2,
            3.662_296_212_560_022e-2,
        ];
        for (actual, expected) in result.potential.iter().take(8).zip(expected) {
            assert_close(*actual, expected);
        }
        assert_eq!(result.potential[8], 0.0);
        assert_eq!(result.potential[250], 0.0);
        Ok(())
    }

    #[test]
    fn coulomb_potential_slw_rejects_invalid_inputs() {
        let density = Array1::<Real>::zeros(4);
        let radii = Array1::<Real>::ones(4);
        let short_radii = Array1::<Real>::ones(3);
        assert!(matches!(
            coulomb_potential_slw(CoulombPotentialSlwInput {
                density: density.view(),
                radii: short_radii.view(),
                delta: 0.05,
                active_len: 3,
            }),
            Err(GridError::CoulombLengthMismatch {
                density_len: 4,
                radii_len: 3,
            })
        ));
        assert!(matches!(
            coulomb_potential_slw(CoulombPotentialSlwInput {
                density: density.view(),
                radii: radii.view(),
                delta: 0.05,
                active_len: 2,
            }),
            Err(GridError::SourceGridTooShort {
                name: "active",
                required: 3,
                available: 2,
            })
        ));
        assert!(matches!(
            coulomb_potential_slw(CoulombPotentialSlwInput {
                density: density.view(),
                radii: radii.view(),
                delta: 0.05,
                active_len: 5,
            }),
            Err(GridError::SourceGridTooShort {
                name: "density",
                required: 5,
                available: 4,
            })
        ));

        let invalid_radii = Array1::from_vec(vec![1.0, 1.1, 0.0, 1.3]);
        assert!(matches!(
            coulomb_potential_slw(CoulombPotentialSlwInput {
                density: density.view(),
                radii: invalid_radii.view(),
                delta: 0.05,
                active_len: 4,
            }),
            Err(GridError::InvalidRadius { radius: 0.0 })
        ));
    }

    #[test]
    fn scmt_energy_grid_matches_feff_grids_reference() -> Result<(), GridError> {
        let result = scmt_energy_grid(ScmtEnergyGridInput {
            core_valence_energy: -0.50,
            fermi_energy: 0.20,
            max_points: 120,
            step_count: 9,
        })?;

        assert_eq!(result.active_len, 74);
        assert_eq!(result.lower_imaginary_count, 5);
        assert_eq!(result.real_axis_count, 61);
        assert_eq!(result.upper_imaginary_count, 8);
        assert_energy(&result, 1, -0.5, 1.837_465_450_137_141e-3);
        assert_energy(&result, 2, -0.5, 7.349_861_800_548_564e-3);
        assert_energy(&result, 3, -0.5, 1.653_718_905_123_426_8e-2);
        assert_energy(&result, 4, -0.5, 2.939_944_720_219_425_7e-2);
        assert_energy(&result, 5, -0.5, 4.593_663_625_342_852e-2);
        assert_energy(
            &result,
            37,
            -1.327_868_852_459_011_8e-1,
            4.593_663_625_342_852e-2,
        );
        assert_energy(&result, 72, 0.2, 7.349_861_800_548_564e-3);
        assert_energy(&result, 73, 0.2, 4.134_297_262_808_567e-3);
        assert_energy(&result, 74, 0.2, 1.837_465_450_137_141e-3);
        assert_eq!(result.energies[74], Complex::new(0.0, 0.0));
        assert_step(&result, 1, 4.593_663_625_342_853e-4);
        assert_step(&result, 5, 4.134_297_262_808_567e-3);
        assert_step(&result, 9, 1.148_415_906_335_713_1e-2);
        Ok(())
    }

    #[test]
    fn scmt_energy_grid_matches_feff_grids_clamped_reference() -> Result<(), GridError> {
        let result = scmt_energy_grid(ScmtEnergyGridInput {
            core_valence_energy: -0.20,
            fermi_energy: 20.00,
            max_points: 42,
            step_count: 8,
        })?;

        assert_eq!(result.active_len, 42);
        assert_eq!(result.lower_imaginary_count, 4);
        assert_eq!(result.real_axis_count, 31);
        assert_eq!(result.upper_imaginary_count, 7);
        assert_energy(&result, 1, -0.2, 1.837_465_450_137_141e-3);
        assert_energy(&result, 4, -0.2, 2.939_944_720_219_425_7e-2);
        assert_energy(
            &result,
            5,
            4.516_129_032_258_064_4e-1,
            2.939_944_720_219_425_7e-2,
        );
        assert_energy(
            &result,
            21,
            1.087_741_935_483_871_1e1,
            2.939_944_720_219_425_7e-2,
        );
        assert_energy(&result, 40, 20.0, 7.349_861_800_548_564e-3);
        assert_energy(&result, 41, 20.0, 4.134_297_262_808_567e-3);
        assert_energy(&result, 42, 20.0, 1.837_465_450_137_141e-3);
        assert_step(&result, 1, 4.593_663_625_342_853e-4);
        assert_step(&result, 7, 7.349_861_800_548_564e-3);
        assert_step(&result, 8, 7.349_861_800_548_564e-3);
        Ok(())
    }

    #[test]
    fn scmt_energy_grid_rejects_invalid_inputs() {
        assert!(matches!(
            scmt_energy_grid(ScmtEnergyGridInput {
                core_valence_energy: f64::NAN,
                fermi_energy: 0.2,
                max_points: 120,
                step_count: 9,
            }),
            Err(GridError::NonFiniteScalar {
                name: "core_valence_energy",
                ..
            })
        ));
        assert_eq!(
            scmt_energy_grid(ScmtEnergyGridInput {
                core_valence_energy: -0.5,
                fermi_energy: 0.2,
                max_points: 120,
                step_count: 0,
            }),
            Err(GridError::InvalidGridLength { name: "step" })
        );
        assert_eq!(
            scmt_energy_grid(ScmtEnergyGridInput {
                core_valence_energy: -0.5,
                fermi_energy: 0.2,
                max_points: 14,
                step_count: 8,
            }),
            Err(GridError::OutputGridTooShort {
                required: 15,
                available: 14,
            })
        );
    }

    #[test]
    fn sum_loucks_spherical_overlap_matches_feff_sumax_wide_reference() -> Result<(), GridError> {
        let (source, base) = sample_sumax_grids();
        let result = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: 2.35,
            multiplicity: 1.75,
            source: source.view(),
            accumulated: base.view(),
        })?;

        assert_eq!(result.active_len, 194);
        assert_overlap_value(
            &result,
            &base,
            1,
            1.745_028_012_500_681_4,
            1.735_031_657_279_253,
        );
        assert_overlap_value(
            &result,
            &base,
            2,
            1.745_017_080_247_046_8,
            1.735_031_656_704_451_3,
        );
        assert_overlap_value(
            &result,
            &base,
            10,
            1.744_669_358_295_808_8,
            1.735_031_649_332_149_8,
        );
        assert_overlap_value(
            &result,
            &base,
            97,
            1.726_426_742_568_854_9,
            1.735_092_022_832_586,
        );
        assert_overlap_value(
            &result,
            &base,
            193,
            1.768_292_002_760_516,
            1.763_509_941_444_896,
        );
        assert_overlap_value(
            &result,
            &base,
            194,
            1.772_250_425_997_878,
            1.767_233_009_588_076_4,
        );
        assert_overlap_value(&result, &base, 195, 5.249_114_029_620_047e-3, 0.0);
        assert_overlap_value(&result, &base, 250, 8.930_063_446_890_768e-3, 0.0);
        Ok(())
    }

    #[test]
    fn sum_loucks_spherical_overlap_matches_feff_sumax_near_reference() -> Result<(), GridError> {
        let (source, base) = sample_sumax_grids();
        let result = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: 0.012,
            multiplicity: 0.60,
            source: source.view(),
            accumulated: base.view(),
        })?;

        assert_eq!(result.active_len, 88);
        assert_overlap_value(
            &result,
            &base,
            1,
            3.436_843_996_472_091_5e-1,
            3.336_880_444_257_808e-1,
        );
        assert_overlap_value(
            &result,
            &base,
            2,
            3.436_695_121_985_222e-1,
            3.336_840_886_559_266e-1,
        );
        assert_overlap_value(
            &result,
            &base,
            10,
            3.432_708_426_104_191e-1,
            3.336_331_336_467_602e-1,
        );
        assert_overlap_value(
            &result,
            &base,
            44,
            3.373_894_297_682_521_5e-1,
            3.336_542_711_118_750_7e-1,
        );
        assert_overlap_value(
            &result,
            &base,
            87,
            3.321_150_695_075_532_6e-1,
            3.391_350_820_293_816e-1,
        );
        assert_overlap_value(
            &result,
            &base,
            88,
            3.326_278_771_348_888e-1,
            3.398_375_950_972_270_5e-1,
        );
        assert_overlap_value(&result, &base, 89, -7.394_167_837_740_848e-3, 0.0);
        assert_overlap_value(&result, &base, 250, 8.930_063_446_890_768e-3, 0.0);
        Ok(())
    }

    #[test]
    fn sum_loucks_spherical_overlap_rejects_invalid_inputs() {
        let source = Array1::<Real>::zeros(250);
        let accumulated = Array1::<Real>::zeros(249);
        assert_eq!(
            sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
                neighbor_distance: 2.35,
                multiplicity: 1.0,
                source: source.view(),
                accumulated: accumulated.view(),
            }),
            Err(GridError::OverlapLengthMismatch {
                source_len: 250,
                accumulated_len: 249,
            })
        );

        let short = Array1::<Real>::zeros(16);
        assert!(matches!(
            sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
                neighbor_distance: 2.35,
                multiplicity: 1.0,
                source: short.view(),
                accumulated: short.view(),
            }),
            Err(GridError::SourceGridTooShort { name: "source", .. })
        ));

        assert!(matches!(
            sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
                neighbor_distance: 2.35,
                multiplicity: f64::NAN,
                source: source.view(),
                accumulated: source.view(),
            }),
            Err(GridError::NonFiniteScalar {
                name: "multiplicity",
                ..
            })
        ));
    }

    #[test]
    fn muffin_tin_overlap_matrix_matches_feff_movrlp_explicit_reference() -> Result<(), GridError> {
        let sample = sample_movrlp_state();
        let explicit = sample.explicit_overlaps();
        let result = muffin_tin_overlap_matrix(sample.input(&explicit))?;
        let factors = result.lu.factors();

        assert_eq!(result.active_order, 101);
        assert_close(result.interstitial_volume, 1.250_001_131_628_848e1);
        assert_close(result.radii[0], 1.507_330_750_954_765e-4);
        assert_close(result.radii[94], 1.657_267_540_176_123_7e-2);
        assert_close(result.radii[99], 2.127_973_643_837_715_8e-2);
        assert_eq!(
            [
                result.lu.pivots()[0],
                result.lu.pivots()[1],
                result.lu.pivots()[49],
                result.lu.pivots()[50],
                result.lu.pivots()[99],
                result.lu.pivots()[100],
                result.lu.pivots()[74],
                result.lu.pivots()[89],
            ],
            [1, 2, 50, 51, 100, 101, 75, 90]
        );

        assert_complex32_close(factors[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(factors[(0, 100)], Complex32::new(1.0e-2, 0.0));
        assert_complex32_close(factors[(100, 0)], Complex32::new(0.0, 0.0));
        assert_complex32_close(factors[(99, 99)], Complex32::new(9.738_406_5e-1, 0.0));
        assert_complex32_close(factors[(100, 100)], Complex32::new(8.354_477e-3, 0.0));
        assert_complex32_close(factors[(29, 98)], Complex32::new(-3.502_009_4e-2, 0.0));
        assert_complex32_close(factors[(29, 99)], Complex32::new(4.868_523_8e-2, 0.0));
        assert_complex32_close(factors[(34, 98)], Complex32::new(-2.731_694e-1, 0.0));
        assert_complex32_close(factors[(34, 99)], Complex32::new(4.531_623e-1, 0.0));
        Ok(())
    }

    #[test]
    fn muffin_tin_overlap_matrix_rejects_invalid_inputs() {
        let sample = sample_movrlp_state();
        let explicit = sample.explicit_overlaps();
        let bad_indices = Array1::from_vec(vec![49, 100]);
        assert_eq!(
            muffin_tin_overlap_matrix(MuffinTinOverlapMatrixInput {
                muffin_tin_indices: bad_indices.view(),
                ..sample.input(&explicit)
            }),
            Err(GridError::MuffinTinIndexTooSmall {
                name: "muffin_tin_indices",
                potential: 0,
                minimum: MOVRLP_NOVP,
                index: 49,
            })
        );

        let bad_positions = Array2::<Real>::zeros((2, 2));
        assert_eq!(
            muffin_tin_overlap_matrix(MuffinTinOverlapMatrixInput {
                atom_positions: bad_positions.view(),
                ..sample.input(&explicit)
            }),
            Err(GridError::InvalidPositionShape {
                rows: 2,
                columns: 2,
            })
        );
    }

    #[test]
    fn sphere_overlap_volumes_match_feff_calcvl_reference() -> Result<(), GridError> {
        let cases = [
            (
                1.25,
                0.95,
                1.10,
                5.612_978_874_413_764e-1,
                1.664_520_507_626_991_6,
            ),
            (
                2.40,
                1.70,
                2.15,
                2.962_352_981_526_981,
                9.622_705_147_348_121,
            ),
            (
                0.80,
                1.60,
                1.25,
                1.356_786_629_672_262_6,
                1.562_880_822_304_789_6,
            ),
            (3.10, 2.90, 4.80, 3.020_854_048_429_17, 6.324_026_011_676_25),
        ];

        for (radius_a, radius_b, distance, expected_cap, expected_lens) in cases {
            assert_close(
                sphere_overlap_cap_volume(radius_a, radius_b, distance)?,
                expected_cap,
            );
            assert_close(
                sphere_overlap_lens_volume(radius_a, radius_b, distance)?,
                expected_lens,
            );
        }
        Ok(())
    }

    #[test]
    fn sphere_overlap_volumes_reject_invalid_inputs() {
        assert_eq!(
            sphere_overlap_cap_volume(0.0, 1.0, 1.0),
            Err(GridError::NonPositiveScalar {
                name: "sphere_radius",
                value: 0.0,
            })
        );
        assert!(matches!(
            sphere_overlap_lens_volume(1.0, Real::NAN, 1.0),
            Err(GridError::NonPositiveScalar {
                name: "other_radius",
                ..
            })
        ));
    }

    #[test]
    fn interstitial_shell_values_match_feff_istval_wide_reference() -> Result<(), GridError> {
        let (potential, density) = sample_istval_grids();
        let muffin_tin_radius = (loucks_x(45) + 0.021).exp();
        let wigner_seitz_radius = (loucks_x(116) + 0.034).exp();
        let muffin_tin_index = loucks_index_below(muffin_tin_radius)?;
        let wigner_seitz_index = loucks_index_below(wigner_seitz_radius)?;

        assert_eq!(muffin_tin_index, 45);
        assert_eq!(wigner_seitz_index, 116);
        let result = interstitial_shell_values(InterstitialShellValuesInput {
            total_potential: potential.view(),
            overlapped_density: density.view(),
            muffin_tin_radius,
            muffin_tin_index,
            wigner_seitz_radius,
            wigner_seitz_index,
        })?;

        assert_interstitial_values(
            result,
            -1.294_131_834_592_241_2,
            8.430_358_921_763_391e-1,
            3.920_777_855_274_227_4e-5,
        );
        Ok(())
    }

    #[test]
    fn interstitial_shell_values_match_feff_istval_tight_reference() -> Result<(), GridError> {
        let (potential, density) = sample_istval_grids();
        let muffin_tin_radius = (loucks_x(70) + 0.010).exp();
        let wigner_seitz_radius = (loucks_x(70) + 0.037).exp();
        let muffin_tin_index = loucks_index_below(muffin_tin_radius)?;
        let wigner_seitz_index = loucks_index_below(wigner_seitz_radius)?;

        assert_eq!(muffin_tin_index, 70);
        assert_eq!(wigner_seitz_index, 70);
        let result = interstitial_shell_values(InterstitialShellValuesInput {
            total_potential: potential.view(),
            overlapped_density: density.view(),
            muffin_tin_radius,
            muffin_tin_index,
            wigner_seitz_radius,
            wigner_seitz_index,
        })?;

        assert_interstitial_values(
            result,
            -1.347_852_330_921_851,
            7.333_517_443_187_345e-1,
            3.102_227_388_939_98e-9,
        );
        Ok(())
    }

    #[test]
    fn interstitial_shell_values_rejects_invalid_inputs() {
        let values = Array1::<Real>::zeros(8);
        assert_eq!(
            interstitial_shell_values(InterstitialShellValuesInput {
                total_potential: values.view(),
                overlapped_density: values.view(),
                muffin_tin_radius: loucks_radius(4),
                muffin_tin_index: 4,
                wigner_seitz_radius: loucks_radius(4),
                wigner_seitz_index: 4,
            }),
            Err(GridError::InvalidRadiusOrder {
                inner_radius: loucks_radius(4),
                outer_radius: loucks_radius(4),
            })
        );

        assert_eq!(
            interstitial_shell_values(InterstitialShellValuesInput {
                total_potential: values.view(),
                overlapped_density: values.view(),
                muffin_tin_radius: loucks_radius(4),
                muffin_tin_index: 0,
                wigner_seitz_radius: loucks_radius(5),
                wigner_seitz_index: 5,
            }),
            Err(GridError::InvalidGridIndex {
                name: "muffin_tin",
                index: 0,
            })
        );

        assert_eq!(
            interstitial_shell_values(InterstitialShellValuesInput {
                total_potential: values.view(),
                overlapped_density: values.view(),
                muffin_tin_radius: loucks_radius(6),
                muffin_tin_index: 6,
                wigner_seitz_radius: loucks_radius(7),
                wigner_seitz_index: 5,
            }),
            Err(GridError::InvalidGridIndexRange {
                lower_index: 6,
                upper_index: 5,
            })
        );

        let short = Array1::<Real>::zeros(4);
        assert_eq!(
            interstitial_shell_values(InterstitialShellValuesInput {
                total_potential: short.view(),
                overlapped_density: short.view(),
                muffin_tin_radius: loucks_radius(3),
                muffin_tin_index: 3,
                wigner_seitz_radius: loucks_radius(4),
                wigner_seitz_index: 4,
            }),
            Err(GridError::SourceGridTooShort {
                name: "total_potential",
                required: 5,
                available: 4,
            })
        );
    }

    #[test]
    fn overlap_density_indices_match_feff_sidx_keep_reference() -> Result<(), GridError> {
        let density = sample_sidx_keep_density();
        let muffin_tin_radius = (feff_legacy_loucks_x(30) + 0.020).exp();
        let norman_radius = (feff_legacy_loucks_x(90) + 0.030).exp();

        let result = overlap_density_indices(OverlapDensityIndicesInput {
            overlapped_density: density.view(),
            muffin_tin_radius,
            norman_radius,
        })?;

        assert_eq!(result.max_density_index, 250);
        assert_eq!(result.muffin_tin_index, 30);
        assert_eq!(result.norman_index, 90);
        assert!(!result.moved_norman_radius);
        assert_close(result.muffin_tin_radius, 6.555_734_762_497_377e-4);
        assert_close(result.norman_radius, 1.329_988_188_760_991_2e-2);
        Ok(())
    }

    #[test]
    fn overlap_density_indices_match_feff_sidx_move_norman_reference() -> Result<(), GridError> {
        let density = sample_sidx_cutoff_density();
        let muffin_tin_radius = (feff_legacy_loucks_x(30) + 0.020).exp();
        let norman_radius = (feff_legacy_loucks_x(130) + 0.010).exp();

        let result = overlap_density_indices(OverlapDensityIndicesInput {
            overlapped_density: density.view(),
            muffin_tin_radius,
            norman_radius,
        })?;

        assert_eq!(result.max_density_index, 92);
        assert_eq!(result.muffin_tin_index, 30);
        assert_eq!(result.norman_index, 92);
        assert!(result.moved_norman_radius);
        assert_close(result.muffin_tin_radius, 6.555_734_762_497_377e-4);
        assert_close(result.norman_radius, 1.426_423_215_543_176_1e-2);
        Ok(())
    }

    #[test]
    fn overlap_density_indices_rejects_invalid_inputs() {
        let density = Array1::<Real>::from_elem(8, 0.1);
        assert_eq!(
            overlap_density_indices(OverlapDensityIndicesInput {
                overlapped_density: density.view(),
                muffin_tin_radius: 0.0,
                norman_radius: loucks_radius(4),
            }),
            Err(GridError::InvalidRadius { radius: 0.0 })
        );

        assert_eq!(
            overlap_density_indices(OverlapDensityIndicesInput {
                overlapped_density: density.view(),
                muffin_tin_radius: loucks_radius(9),
                norman_radius: loucks_radius(10),
            }),
            Err(GridError::SourceGridTooShort {
                name: "overlapped_density",
                required: 9,
                available: 8,
            })
        );

        let zero_tail = Array1::<Real>::from_elem(16, 1.0e-6);
        assert_eq!(
            overlap_density_indices(OverlapDensityIndicesInput {
                overlapped_density: zero_tail.view(),
                muffin_tin_radius: loucks_radius(4),
                norman_radius: loucks_radius(8),
            }),
            Err(GridError::NoActiveDensityTail {
                start_index: 4,
                threshold: SIDX_DENSITY_CUTOFF,
            })
        );

        let mut nonfinite = Array1::<Real>::from_elem(16, 0.1);
        nonfinite[2] = Real::NAN;
        assert!(matches!(
            overlap_density_indices(OverlapDensityIndicesInput {
                overlapped_density: nonfinite.view(),
                muffin_tin_radius: loucks_radius(4),
                norman_radius: loucks_radius(8),
            }),
            Err(GridError::NonFiniteGridValue {
                name: "overlapped_density",
                index: 2,
                ..
            })
        ));
    }

    #[test]
    fn norman_radius_matches_feff_frnrm_oxygen_like_reference() -> Result<(), GridError> {
        let density = sample_frnrm_oxygen_density();

        let result = norman_radius_from_density(NormanRadiusInput {
            overlapped_density: density.view(),
            atomic_number: 8,
        })?;

        assert_close(result.radius, 1.063_980_446_859_560_2);
        Ok(())
    }

    #[test]
    fn norman_radius_matches_feff_frnrm_iron_like_reference() -> Result<(), GridError> {
        let density = sample_frnrm_iron_density();

        let result = norman_radius_from_density(NormanRadiusInput {
            overlapped_density: density.view(),
            atomic_number: 26,
        })?;

        assert_close(result.radius, 8.688_945_443_598_616e-1);
        Ok(())
    }

    #[test]
    fn norman_radius_matches_feff_frnrm_gold_like_reference() -> Result<(), GridError> {
        let density = sample_frnrm_gold_density();

        let result = norman_radius_from_density(NormanRadiusInput {
            overlapped_density: density.view(),
            atomic_number: 79,
        })?;

        assert_close(result.radius, 6.973_687_583_509_427e-1);
        Ok(())
    }

    #[test]
    fn norman_radius_rejects_invalid_inputs() {
        let density = Array1::<Real>::from_elem(FRNRM_DENSITY_POINTS, 1.0);
        assert_eq!(
            norman_radius_from_density(NormanRadiusInput {
                overlapped_density: density.view(),
                atomic_number: 0,
            }),
            Err(GridError::InvalidAtomicNumber { atomic_number: 0 })
        );

        let short_density = Array1::<Real>::zeros(FRNRM_DENSITY_POINTS - 1);
        assert_eq!(
            norman_radius_from_density(NormanRadiusInput {
                overlapped_density: short_density.view(),
                atomic_number: 1,
            }),
            Err(GridError::SourceGridTooShort {
                name: "overlapped_density",
                required: FRNRM_DENSITY_POINTS,
                available: FRNRM_DENSITY_POINTS - 1,
            })
        );

        let zero_density = Array1::<Real>::zeros(FRNRM_DENSITY_POINTS);
        assert!(matches!(
            norman_radius_from_density(NormanRadiusInput {
                overlapped_density: zero_density.view(),
                atomic_number: 1,
            }),
            Err(GridError::InsufficientNormanCharge {
                atomic_number: 1,
                ..
            })
        ));
    }

    #[test]
    fn interstitial_fermi_level_matches_feff_fermi_reference() -> Result<(), GridError> {
        let shell = interstitial_fermi_level(FermiLevelInput {
            interstitial_density: 8.430_358_921_763_391e-1,
            interstitial_potential: -1.294_131_834_592_241_2,
        })?;
        assert_fermi_level(
            shell,
            -5.040_450_363_824_843e-1,
            1.526_716_490_479_997_5,
            1.257_049_560_049_051_4,
        );

        let dense = interstitial_fermi_level(FermiLevelInput {
            interstitial_density: 3.2,
            interstitial_potential: -0.42,
        })?;
        assert_fermi_level(
            dense,
            1.502_548_984_343_600_6,
            9.787_169_102_922_159e-1,
            1.960_892_135_913_447_2,
        );
        Ok(())
    }

    #[test]
    fn interstitial_fermi_level_rejects_invalid_inputs() {
        assert_eq!(
            interstitial_fermi_level(FermiLevelInput {
                interstitial_density: 0.0,
                interstitial_potential: -1.0,
            }),
            Err(GridError::NonPositiveScalar {
                name: "interstitial_density",
                value: 0.0,
            })
        );
        assert!(matches!(
            interstitial_fermi_level(FermiLevelInput {
                interstitial_density: 1.0,
                interstitial_potential: Real::NAN,
            }),
            Err(GridError::NonFiniteScalar {
                name: "interstitial_potential",
                ..
            })
        ));
    }

    fn assert_spinor_value(
        spinor: &DiracSpinorGrid,
        index_1based: usize,
        expected_large: Real,
        expected_small: Real,
    ) {
        let index = index_1based - 1;
        assert_close(spinor.large_component[index], expected_large);
        assert_close(spinor.small_component[index], expected_small);
    }

    fn assert_orbital_value(
        spinor: &DiracSpinorOrbitalsGrid,
        index_1based: usize,
        orbital_1based: usize,
        expected_large: Real,
        expected_small: Real,
    ) {
        let radial = index_1based - 1;
        let orbital = orbital_1based - 1;
        assert_close(spinor.large_components[(radial, orbital)], expected_large);
        assert_close(spinor.small_components[(radial, orbital)], expected_small);
    }

    fn assert_potential_value(
        grid: &PotentialGrid,
        index_1based: usize,
        expected_radius: Real,
        expected_potential: Real,
        expected_density: Real,
        expected_magnetization: Real,
    ) {
        let index = index_1based - 1;
        assert_close(grid.radii[index], expected_radius);
        assert_close(grid.total_potential[index], expected_potential);
        assert_close(grid.charge_density[index], expected_density);
        assert_close(grid.magnetization[index], expected_magnetization);
    }

    fn assert_energy(
        grid: &ScmtEnergyGrid,
        index_1based: usize,
        expected_real: Real,
        expected_imaginary: Real,
    ) {
        let value = grid.energies[index_1based - 1];
        assert_close(value.re, expected_real);
        assert_close(value.im, expected_imaginary);
    }

    fn assert_step(grid: &ScmtEnergyGrid, index_1based: usize, expected: Real) {
        assert_close(grid.steps[index_1based - 1], expected);
    }

    fn assert_overlap_value(
        overlap: &LoucksSphericalOverlap,
        base: &Array1<Real>,
        index_1based: usize,
        expected_total: Real,
        expected_contribution: Real,
    ) {
        let index = index_1based - 1;
        const SUMAX_ORACLE_TOLERANCE: Real = 5.0e-9;

        assert_close_with_tolerance(
            overlap.accumulated[index],
            expected_total,
            SUMAX_ORACLE_TOLERANCE,
        );
        assert_close_with_tolerance(
            overlap.accumulated[index] - base[index],
            expected_contribution,
            SUMAX_ORACLE_TOLERANCE,
        );
    }

    fn assert_interstitial_values(
        values: InterstitialShellValues,
        expected_potential: Real,
        expected_density: Real,
        expected_volume: Real,
    ) {
        const ISTVAL_ORACLE_TOLERANCE: Real = 5.0e-10;

        assert_close_with_tolerance(
            values.interstitial_potential,
            expected_potential,
            ISTVAL_ORACLE_TOLERANCE,
        );
        assert_close_with_tolerance(
            values.interstitial_density,
            expected_density,
            ISTVAL_ORACLE_TOLERANCE,
        );
        assert_close_with_tolerance(
            values.shell_volume,
            expected_volume,
            1.0e-15_f64.max(expected_volume.abs() * 5.0e-7),
        );
    }

    fn assert_fermi_level(
        value: FermiLevel,
        expected_chemical_potential: Real,
        expected_density_parameter: Real,
        expected_fermi_momentum: Real,
    ) {
        assert_close(value.chemical_potential, expected_chemical_potential);
        assert_close(value.density_parameter, expected_density_parameter);
        assert_close(value.fermi_momentum, expected_fermi_momentum);
    }

    fn run_sample_potential_grid(
        jump_mode: i32,
        potential_jump: Real,
    ) -> Result<PotentialGrid, GridError> {
        let (density, potential, magnetization) = sample_potential_sources();
        fix_potential_grid(PotentialGridInput {
            muffin_tin_radius: (-8.8 + 60.4 * 0.05_f64).exp(),
            electron_density: density.view(),
            total_potential: potential.view(),
            magnetization: magnetization.view(),
            interstitial_potential: -0.75,
            interstitial_density: 0.28,
            original_delta: 0.05,
            new_delta: 0.025,
            jump_mode,
            potential_jump,
            output_len: 180,
        })
    }

    fn sample_potential_sources() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
        let source_len = 251;
        let density = (1..=source_len)
            .map(|index| {
                let i = index as Real;
                0.4 + 0.002 * i + 0.03 * (0.04 * i).sin()
            })
            .collect::<Array1<_>>();
        let potential = (1..=source_len)
            .map(|index| {
                let i = index as Real;
                -2.0 + 0.015 * i + 0.05 * (0.03 * i).cos()
            })
            .collect::<Array1<_>>();
        let magnetization = (1..=source_len)
            .map(|index| {
                let i = index as Real;
                0.01 * (0.08 * i).sin() - 0.0001 * i
            })
            .collect::<Array1<_>>();
        (density, potential, magnetization)
    }

    #[derive(Debug, Clone)]
    struct MovrlpSample {
        atom_potentials: Array1<usize>,
        atom_positions: Array2<Real>,
        representative_atoms: Array1<usize>,
        potential_multiplicities: Array1<Real>,
        neighbors0: [MuffinTinOverlapNeighbor; 1],
        neighbors1: [MuffinTinOverlapNeighbor; 1],
        muffin_tin_indices: Array1<usize>,
        muffin_tin_radii: Array1<Real>,
        norman_radii: Array1<Real>,
        near_neighbor_flags: Array1<bool>,
    }

    impl MovrlpSample {
        fn explicit_overlaps(&self) -> [&[MuffinTinOverlapNeighbor]; 2] {
            [&self.neighbors0, &self.neighbors1]
        }

        fn input<'a>(
            &'a self,
            explicit_overlaps: &'a [&'a [MuffinTinOverlapNeighbor]],
        ) -> MuffinTinOverlapMatrixInput<'a> {
            MuffinTinOverlapMatrixInput {
                highest_potential_index: 1,
                atom_potentials: self.atom_potentials.view(),
                atom_positions: self.atom_positions.view(),
                representative_atoms: self.representative_atoms.view(),
                potential_multiplicities: self.potential_multiplicities.view(),
                explicit_overlaps,
                muffin_tin_indices: self.muffin_tin_indices.view(),
                muffin_tin_radii: self.muffin_tin_radii.view(),
                norman_radii: self.norman_radii.view(),
                near_neighbor_flags: self.near_neighbor_flags.view(),
                interstitial_selector: 0,
                interstitial_volume: 12.5,
            }
        }
    }

    fn sample_movrlp_state() -> MovrlpSample {
        let atom_potentials = Array1::from_vec(vec![0, 1]);
        let atom_positions = Array2::<Real>::zeros((2, 3));
        let representative_atoms = Array1::from_vec(vec![0, 1]);
        let potential_multiplicities = Array1::from_vec(vec![1.0, 2.0]);
        let neighbors0 = [MuffinTinOverlapNeighbor {
            source_potential: 1,
            multiplicity: 2,
            distance: 0.030,
        }];
        let neighbors1 = [MuffinTinOverlapNeighbor {
            source_potential: 0,
            multiplicity: 1,
            distance: 0.031,
        }];
        let muffin_tin_indices = Array1::from_vec(vec![95, 100]);
        let muffin_tin_radii = Array1::from_vec(vec![0.020, 0.024]);
        let norman_radii = Array1::from_vec(vec![0.015, 0.018]);
        let near_neighbor_flags = Array1::from_vec(vec![false, false]);
        MovrlpSample {
            atom_potentials,
            atom_positions,
            representative_atoms,
            potential_multiplicities,
            neighbors0,
            neighbors1,
            muffin_tin_indices,
            muffin_tin_radii,
            norman_radii,
            near_neighbor_flags,
        }
    }

    fn sample_sumax_grids() -> (Array1<Real>, Array1<Real>) {
        let len = 250;
        let source = (1..=len)
            .map(|index| {
                let i = index as Real;
                0.2 + 0.004 * i + 0.03 * (0.035 * i).sin()
            })
            .collect::<Array1<_>>();
        let base = (1..=len)
            .map(|index| {
                let i = index as Real;
                0.01 * (0.027 * i).cos()
            })
            .collect::<Array1<_>>();
        (source, base)
    }

    fn sample_istval_grids() -> (Array1<Real>, Array1<Real>) {
        let len = 1251;
        let potential = (1..=len)
            .map(|index| {
                let i = index as Real;
                -1.5 + 0.002 * i + 0.04 * (0.017 * i).cos()
            })
            .collect::<Array1<_>>();
        let density = (1..=len)
            .map(|index| {
                let i = index as Real;
                0.5 + 0.003 * i + 0.02 * (0.023 * i).sin()
            })
            .collect::<Array1<_>>();
        (potential, density)
    }

    fn sample_frnrm_oxygen_density() -> Array1<Real> {
        (1..=FRNRM_DENSITY_POINTS)
            .map(|index| {
                let radius = feff_legacy_loucks_radius(index);
                50.0 * (-1.2 * radius).exp() + 0.1 * (-0.05 * radius).exp()
            })
            .collect::<Array1<_>>()
    }

    fn sample_frnrm_iron_density() -> Array1<Real> {
        (1..=FRNRM_DENSITY_POINTS)
            .map(|index| {
                let radius = feff_legacy_loucks_radius(index);
                220.0 * (-0.85 * radius).exp() / (1.0 + 0.12 * radius)
            })
            .collect::<Array1<_>>()
    }

    fn sample_frnrm_gold_density() -> Array1<Real> {
        (1..=FRNRM_DENSITY_POINTS)
            .map(|index| {
                let radius = feff_legacy_loucks_radius(index);
                950.0 * (-0.55 * radius).exp() / (1.0 + 0.08 * radius * radius)
            })
            .collect::<Array1<_>>()
    }

    fn sample_sidx_keep_density() -> Array1<Real> {
        (1..=250)
            .map(|index| {
                let i = index as Real;
                0.08 + 0.0004 * i + 0.002 * (0.05 * i).sin()
            })
            .collect::<Array1<_>>()
    }

    fn sample_sidx_cutoff_density() -> Array1<Real> {
        (1..=250)
            .map(|index| {
                if index <= 92 {
                    0.04 + 0.0002 * index as Real
                } else {
                    1.0e-6
                }
            })
            .collect::<Array1<_>>()
    }

    fn assert_close(actual: Real, expected: Real) {
        let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
        assert_close_with_tolerance(actual, expected, tolerance);
    }

    fn assert_close_with_tolerance(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    fn assert_complex32_close(actual: Complex32, expected: Complex32) {
        assert_close_with_tolerance(actual.re as Real, expected.re as Real, 5.0e-6);
        assert_close_with_tolerance(actual.im as Real, expected.im as Real, 5.0e-6);
    }
}
