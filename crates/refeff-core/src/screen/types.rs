//! Public SCREEN data types.

use ndarray::{ArrayView1, ArrayView2};
use num_complex::Complex32;
use thiserror::Error;

use crate::{Complex, ComplexVec, Real, RealMat, RealVec};

/// Error returned by FEFF SCREEN helper kernels.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ScreenError {
    #[error("SCREEN input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    #[error("SCREEN complex input {name} must be finite, got {real}+{imaginary}i")]
    NonFiniteComplexInput {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    #[error("SCREEN input {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    #[error("SCREEN radial count must be positive")]
    EmptyRadialGrid,
    #[error("SCREEN active radial count {active_count} exceeds input length {len}")]
    ActiveCountOutOfRange { active_count: usize, len: usize },
    #[error("SCREEN atom positions must have exactly 3 coordinate columns, got {columns}")]
    AtomPositionColumnCount { columns: usize },
    #[error("SCREEN radial index is outside isize range: {value}")]
    RadialIndexOutOfRange { value: Real },
    #[error("SCREEN radial bound {name} must be positive after FEFF indexing, got {value}")]
    NonPositiveRadialBound { name: &'static str, value: isize },
    #[error("SCREEN radial bound {name}={value} exceeds capacity {capacity}")]
    RadialBoundOutOfRange {
        name: &'static str,
        value: usize,
        capacity: usize,
    },
    #[error("SCREEN {name} count {actual} is below minimum {minimum}")]
    CountTooSmall {
        name: &'static str,
        actual: usize,
        minimum: usize,
    },
    #[error("SCREEN input {upper_name} must exceed {lower_name}: {upper} <= {lower}")]
    NonIncreasingInput {
        lower_name: &'static str,
        upper_name: &'static str,
        lower: Real,
        upper: Real,
    },
    #[error("SCREEN energy grid requires {required} points but capacity is {available}")]
    EnergyGridTooLong { required: usize, available: usize },
    #[error("SCREEN energy grid size overflow for {name}")]
    EnergyGridSizeOverflow { name: &'static str },
    #[error("SCREEN index size overflow for {name}")]
    IndexSizeOverflow { name: &'static str },
    #[error("SCREEN energy grid unexpectedly has no points")]
    EmptyEnergyGrid,
    #[error("SCREEN energy index {index} is out of range for {len} energies")]
    EnergyIndexOutOfRange { index: usize, len: usize },
    #[error("SCREEN result {name} must be finite, got {value}")]
    NonFiniteResult { name: &'static str, value: Real },
    #[error("SCREEN complex result {name} must be finite, got {real}+{imaginary}i")]
    NonFiniteComplexResult {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    #[error("SCREEN complex result {name} must be nonzero")]
    ZeroComplexResult { name: &'static str },
    #[error("SCREEN result {name} must be positive, got {value}")]
    NonPositiveResult { name: &'static str, value: Real },
    #[error(
        "SCREEN matrix {name} must be at least {active_count}x{active_count}, got {rows}x{columns}"
    )]
    MatrixTooSmall {
        name: &'static str,
        rows: usize,
        columns: usize,
        active_count: usize,
    },
    #[error("SCREEN matrix {name}({row},{column}) must be finite, got {value}")]
    NonFiniteMatrixInput {
        name: &'static str,
        row: usize,
        column: usize,
        value: Real,
    },
    #[error("SCREEN complex matrix {name}({row},{column}) must be finite, got {real}+{imaginary}i")]
    NonFiniteComplexMatrixInput {
        name: &'static str,
        row: usize,
        column: usize,
        real: Real,
        imaginary: Real,
    },
    #[error("SCREEN linear solve failed: {0}")]
    Linalg(#[from] refeff_linalg::LinalgError),
}

/// Inputs for SCREEN `setegi`: rectangular complex-energy contour setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenContourEnergyGridInput {
    /// Lower real-axis energy `emin`.
    pub min_real_energy: Real,
    /// Upper real-axis energy `emax`.
    pub max_real_energy: Real,
    /// Maximum imaginary-axis energy `eimax`.
    pub max_imaginary_energy: Real,
    /// Minimum imaginary-axis offset `ermin`; FEFF clamps non-positive values to 0.05.
    pub min_imaginary_energy: Real,
    /// Number of real-axis divisions `ner`.
    pub real_points: usize,
    /// Number of imaginary-axis divisions `nei`.
    pub imaginary_points: usize,
    /// Capacity of the output energy table, equivalent to FEFF `nex`.
    pub max_points: usize,
}

/// SCREEN complex-energy contour with FEFF's active-length convention.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenContourEnergyGrid {
    /// Complex contour energies `em`, zero-filled after [`ScreenContourEnergyGrid::active_len`].
    pub energies: ComplexVec,
    /// Number of active contour points returned as FEFF `ne`.
    pub active_len: usize,
    /// Effective `ermin` after FEFF's non-positive clamp.
    pub effective_min_imaginary_energy: Real,
}

/// Inputs for SCREEN/CRPA radial active-prefix setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenRadialBoundsInput {
    /// Loucks-grid origin parameter `x0`.
    pub x0: Real,
    /// Loucks-grid spacing `dx`.
    pub dx: Real,
    /// Muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
    /// Norman radius `rnrm`.
    pub norman_radius: Real,
    /// FEFF tail extension `iend` used in `ilast = jnrm + 6 + iend`.
    pub tail_extension: isize,
    /// Radial wavefunction capacity, equivalent to FEFF `nrptx`.
    pub radial_capacity: usize,
    /// Response-array capacity, equivalent to FEFF `nrx`.
    pub response_capacity: usize,
}

/// SCREEN/CRPA radial bounds using FEFF's 1-based names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenRadialBounds {
    /// FEFF `jri = getiat(x0, dx, rmt) + 1`.
    pub muffin_tin_index_1based: usize,
    /// FEFF `jri1 = jri + 1`, checked against `nrptx`.
    pub muffin_tin_next_index_1based: usize,
    /// FEFF `jnrm = getiat(x0, dx, rnrm) + 1`.
    pub norman_index_1based: usize,
    /// FEFF `ilast = min(jnrm + 6 + iend, nrx)`.
    pub active_count: usize,
}

/// Inputs for SCREEN `getph.f90` radial integration bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenGetphRadialBoundsInput {
    /// Loucks-grid origin parameter `x0`.
    pub x0: Real,
    /// Loucks-grid spacing `dx`.
    pub dx: Real,
    /// Muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
    /// Norman radius `rnrm`.
    pub norman_radius: Real,
    /// Radial wavefunction capacity, equivalent to FEFF `nrptx`.
    pub radial_capacity: usize,
}

/// SCREEN `getph.f90` radial bounds using FEFF's 1-based names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenGetphRadialBounds {
    /// FEFF `jri = getiat(x0, dx, rmt) + 1`.
    pub muffin_tin_index_1based: usize,
    /// FEFF `jnrm = getiat(x0, dx, rnrm) + 1`.
    pub norman_index_1based: usize,
    /// FEFF `ilast = min(jnrm + 6, nrptx)`.
    pub active_count: usize,
}

/// Inputs for the SCREEN/CRPA per-energy state setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenEnergyStateInput {
    /// Complex contour energy `em(ie)`.
    pub energy: Complex,
    /// Complex reference potential `eref`.
    pub reference_energy: Complex,
    /// Muffin-tin radius `rmt` for `xkmt = rmt * ck`.
    pub muffin_tin_radius: Real,
    /// FEFF exchange selector `ixc0`; `mod(ixc0,10) >= 5` enables three cycles.
    pub exchange_selector: i32,
}

/// SCREEN/CRPA per-energy values shared by the response drivers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenEnergyState {
    /// FEFF `p2 = em(ie) - eref`.
    pub kinetic_energy: Complex,
    /// Relativistic complex wave number `ck`.
    pub wave_number: Complex,
    /// Single-precision FMS wave number `cks(1)`.
    pub fms_wave_number: Complex32,
    /// Muffin-tin wave argument `xkmt = rmt * ck`.
    pub muffin_tin_argument: Complex,
    /// FEFF `ncycle`: `0` for low exchange models, `3` otherwise.
    pub dirac_cycle_count: usize,
}

/// Inputs for SCREEN/CRPA regular-solution relativistic normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenSolutionNormalizationInput {
    /// Complex wave number `ck`.
    pub wave_number: Complex,
    /// FEFF `temp`, the `phamp` amplitude used to normalize the regular radial solution.
    pub phase_amplitude: Complex,
}

/// Relativistic normalization factors used by SCREEN/CRPA radial solutions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenSolutionNormalization {
    /// FEFF lower-component factor after `factor = -ck*alphfs/(1+sqrt(1+(ck*alphfs)**2))`.
    pub small_component_factor: Complex,
    /// FEFF `dum1 = 1/sqrt(1+factor**2)`.
    pub relativistic_scale: Complex,
    /// FEFF `xfnorm = dum1/temp`, or zero when `temp == 0`.
    pub regular_solution_scale: Complex,
}

/// Inputs for SCREEN irregular-solution muffin-tin boundary values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenIrregularInitialConditionInput {
    /// Muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
    /// Complex phase shift `ph0` from `phamp`.
    pub phase_shift: Complex,
    /// Complex photoelectron wave number `ck`.
    pub wave_number: Complex,
    /// Spherical Bessel value `jl`.
    pub bessel_j_l: Complex,
    /// Spherical Neumann value `nl`.
    pub neumann_l: Complex,
    /// Next-order spherical Bessel value `jlp1`.
    pub bessel_j_l_plus_1: Complex,
    /// Next-order spherical Neumann value `nlp1`.
    pub neumann_l_plus_1: Complex,
    /// FEFF Hankel value `bessh(l+1)`, used only when `use_hankel_boundary` is true.
    pub hankel_l: Complex,
    /// FEFF Hankel value `bessh(l+2)`, used only when `use_hankel_boundary` is true.
    pub hankel_l_plus_1: Complex,
    /// FEFF `irrh == 1` switch for outgoing-Hankel irregular boundary values.
    pub use_hankel_boundary: bool,
}

/// Irregular-solution initial values passed into `dfovrg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenIrregularInitialCondition {
    /// FEFF input `pu` for the irregular `dfovrg` call.
    pub large_component: Complex,
    /// FEFF input `qu` for the irregular `dfovrg` call.
    pub small_component: Complex,
}

/// Inputs for SCREEN irregular-solution Wronskian normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenIrregularWronskianScaleInput {
    /// Complex phase shift `ph0` from `phamp`.
    pub phase_shift: Complex,
    /// Complex photoelectron wave number `ck`.
    pub wave_number: Complex,
    /// Regular large radial solution at FEFF `jri`, `pr(jri)`.
    pub regular_large_at_match: Complex,
    /// Regular small radial solution at FEFF `jri`, `qr(jri)`.
    pub regular_small_at_match: Complex,
    /// Irregular large radial solution at FEFF `jri`, `pn(jri)`.
    pub irregular_large_at_match: Complex,
    /// Irregular small radial solution at FEFF `jri`, `qn(jri)`.
    pub irregular_small_at_match: Complex,
}

/// FEFF Wronskian scale for the irregular radial solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenIrregularWronskianScale {
    /// FEFF `temp = exp(i*ph0)`.
    pub phase_factor: Complex,
    /// FEFF denominator before reciprocal scaling:
    /// `2*alpinv*temp*(pn(jri)*qr(jri)-pr(jri)*qn(jri))`.
    pub denominator: Complex,
    /// FEFF overwritten `qu = 1 / denominator / ck`, or zero when the denominator is zero.
    pub reciprocal_wave_scale: Complex,
    /// Multiplier applied to both `pn` and `qn`: `temp * reciprocal_wave_scale`.
    pub irregular_solution_scale: Complex,
}

/// Inputs for one exact radial-continuation point outside the muffin-tin match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenExactRadialContinuationInput {
    /// Radial point `ri(j)`.
    pub radius: Real,
    /// Complex phase shift `ph0` from `phamp`.
    pub phase_shift: Complex,
    /// Complex photoelectron wave number `ck`.
    pub wave_number: Complex,
    /// Spherical Bessel value `jl` at `ck*ri(j)`.
    pub bessel_j_l: Complex,
    /// Spherical Neumann value `nl` at `ck*ri(j)`.
    pub neumann_l: Complex,
    /// Next-order spherical Bessel value `jlp1` at `ck*ri(j)`.
    pub bessel_j_l_plus_1: Complex,
    /// Next-order spherical Neumann value `nlp1` at `ck*ri(j)`.
    pub neumann_l_plus_1: Complex,
    /// FEFF Hankel value `bessh(l+1)` at `ck*ri(j)`.
    pub hankel_l: Complex,
    /// FEFF Hankel value `bessh(l+2)` at `ck*ri(j)`.
    pub hankel_l_plus_1: Complex,
}

/// Exact regular and irregular radial values used after `jri`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenExactRadialContinuation {
    /// FEFF exact continued regular large component `pr(j)`.
    pub regular_large_component: Complex,
    /// FEFF exact continued regular small component `qr(j)`.
    pub regular_small_component: Complex,
    /// FEFF exact continued irregular large component `pn(j)`.
    pub irregular_large_component: Complex,
    /// FEFF exact continued irregular small component `qn(j)`.
    pub irregular_small_component: Complex,
}

/// Inputs for SCREEN `rdgeom.f90` unit conversion.
#[derive(Debug, Clone, Copy)]
pub struct ScreenRdgeomAtomicUnitsInput<'a> {
    /// Atom Cartesian positions `rat`, stored as an `atoms x 3` table in Angstrom.
    pub atom_positions_angstrom: ArrayView2<'a, Real>,
    /// FEFF `rfms2` cluster radius in Angstrom.
    pub rfms2_angstrom: Real,
    /// FEFF `rdirec` direct radius in Angstrom.
    pub direct_radius_angstrom: Real,
    /// SCREEN lower real-energy bound `emin` in eV.
    pub min_real_energy_ev: Real,
    /// SCREEN upper real-energy bound `emax` in eV.
    pub max_real_energy_ev: Real,
    /// SCREEN upper imaginary-energy bound `eimax` in eV.
    pub max_imaginary_energy_ev: Real,
    /// SCREEN FMS radius `ScreenI%rfms` in Angstrom.
    pub screen_rfms_angstrom: Real,
    /// SCREEN minimum imaginary-energy offset `ScreenI%ermin` in eV.
    pub min_imaginary_energy_ev: Real,
    /// SCREEN maximum angular count `ScreenI%maxl`.
    pub max_l: usize,
    /// FEFF angular capacity `lx`.
    pub angular_capacity_lx: usize,
}

/// SCREEN setup values converted to FEFF atomic units.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenRdgeomAtomicUnits {
    /// Atom Cartesian positions in bohr, preserving the input `atoms x 3` layout.
    pub atom_positions_bohr: RealMat,
    /// FEFF `rfms2` in bohr.
    pub rfms2_bohr: Real,
    /// FEFF `rdirec` in bohr.
    pub direct_radius_bohr: Real,
    /// SCREEN lower real-energy bound in Hartree.
    pub min_real_energy_hartree: Real,
    /// SCREEN upper real-energy bound in Hartree.
    pub max_real_energy_hartree: Real,
    /// SCREEN upper imaginary-energy bound in Hartree.
    pub max_imaginary_energy_hartree: Real,
    /// SCREEN FMS radius in bohr.
    pub screen_rfms_bohr: Real,
    /// SCREEN minimum imaginary-energy offset in Hartree.
    pub min_imaginary_energy_hartree: Real,
    /// FEFF `ScreenI%maxl = min(ScreenI%maxl, lx + 1)`.
    pub max_l: usize,
}

/// Inputs for SCREEN `prep.f90` phase-potential reference shifting.
#[derive(Debug, Clone)]
pub struct ScreenPhasePotentialInput<'a> {
    /// FEFF `vtotph` after `fixvar`.
    pub total_potential: ArrayView1<'a, Real>,
    /// FEFF `vvalph` after `fixvar`.
    pub valence_potential: ArrayView1<'a, Real>,
    /// FEFF `jri1 = jri + 1`, used as a 1-based reference-potential index.
    pub muffin_tin_next_index_1based: usize,
    /// FEFF exchange selector `ixc`; values `>= 5` keep a separate valence potential.
    pub exchange_selector: i32,
}

/// Reference-shifted phase potentials prepared for `getph`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenPhasePotential {
    /// FEFF `eref(1) = vtotph(jri1)`.
    pub reference_energy: Real,
    /// Shifted `vtotph`; only `1:jri1` is modified.
    pub total_potential: RealVec,
    /// Shifted or copied `vvalph`; only `1:jri1` is modified.
    pub valence_potential: RealVec,
}

/// CRPA radial projection window from `chi_crpa.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenCrpaProjectionWindow {
    /// Lower clamp radius. FEFF uses `rcut0 = rcut - 1`.
    pub inner_radius: Real,
    /// Upper clamp radius. FEFF uses `rcut = rnrm * rcutin`.
    pub outer_radius: Real,
}

/// Normalized CRPA radial density and shell weights.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenCrpaDensityWeights {
    /// Density after optional projection and FEFF normalization.
    pub normalized_density: RealVec,
    /// FEFF `vch(i) = normalized_density(i) * dx * ri(i)` weights, with the
    /// tail after `jnrm` zeroed.
    pub shell_weights: RealVec,
    /// Pre-normalization integral `sum rho(i) * ri(i) * dx`.
    pub normalization: Real,
}

/// CRPA Hubbard-parameter accumulation result from `chi_crpa.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenCrpaHubbardSummary {
    /// FEFF final `vch(i) = wscrn(i) * den_CRPA(i,ie)` radial table.
    pub screened_density_potential: RealVec,
    /// Screened Hubbard interaction `U_Hub`, in the same Hartree units written
    /// to `crpa.dat`.
    pub hubbard_u: Real,
    /// FEFF occupation integral `n_occ`.
    pub occupation: Real,
    /// Bare Hubbard interaction `U_Bare`, in the same Hartree units written to
    /// `crpa.dat`.
    pub bare_u: Real,
}

/// Inputs for [`crate::screen::screen_fms_response_slice`].
#[derive(Debug, Clone)]
pub struct ScreenFmsResponseSliceInput<'a> {
    /// FEFF radial grid `ri`.
    pub radii: &'a [Real],
    /// Regular radial solution `pr(:,l)`.
    pub regular_solution: ArrayView1<'a, Complex>,
    /// Irregular radial solution `pn(:,l)`.
    pub irregular_solution: ArrayView1<'a, Complex>,
    /// FEFF cluster Green's function `gtrl(l,ie)`.
    pub cluster_green: Complex,
    /// Complex photoelectron wave number `ck`.
    pub wave_number: Complex,
    /// Loucks-grid spacing `dx`.
    pub dx: Real,
    /// Angular momentum `l`.
    pub angular_momentum: usize,
    /// Active radial count, FEFF `ilast`.
    pub active_count: usize,
    /// FMS correction prefix, FEFF `jnrm`.
    pub fms_count: usize,
}

/// Inputs for [`crate::screen::screen_crpa_response_slice`].
#[derive(Debug, Clone)]
pub struct ScreenCrpaResponseSliceInput<'a> {
    /// FEFF radial grid `ri`.
    pub radii: &'a [Real],
    /// Regular radial solution `pr(:,l)`.
    pub regular_solution: ArrayView1<'a, Complex>,
    /// Irregular radial solution `pn(:,l)`.
    pub irregular_solution: ArrayView1<'a, Complex>,
    /// Diagonal CRPA/FMS cluster Green's function `gtrl(l,l,ie)`.
    pub cluster_green: Complex,
    /// Complex photoelectron wave number `ck`.
    pub wave_number: Complex,
    /// Loucks-grid spacing `dx`.
    pub dx: Real,
    /// Angular momentum `l` for this response channel.
    pub angular_momentum: usize,
    /// Selected constrained-RPA channel `ll_CRPA`.
    pub crpa_angular_momentum: usize,
    /// Optional CRPA projection window. FEFF's default CRPA path applies this
    /// only when `angular_momentum == crpa_angular_momentum`.
    pub projection_window: Option<ScreenCrpaProjectionWindow>,
    /// Active radial count, FEFF `ilast`.
    pub active_count: usize,
}
