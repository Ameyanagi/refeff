//! FEFF SCREEN helper kernels.
//!
//! These routines cover small, self-contained pieces from `SCREEN/frgrid.f90`,
//! `SCREEN/fegrid.f90`, `SCREEN/fxc.f90`, and the response setup blocks in
//! `SCREEN/screensub.f90` and `CRPA/chi_crpa.f90`, plus the compact CRPA radial
//! density setup block. The full SCREEN/CRPA drivers also depend on phase,
//! potential, and FMS handoff state; keeping these kernels separate makes them
//! usable and testable while those drivers are ported incrementally.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};
use num_complex::Complex32;
use refeff_linalg::{real_lu_factor, real_lu_solve_vector};
use thiserror::Error;

use crate::{Complex, ComplexMat, ComplexVec, Real, RealMat, RealVec};

/// FEFF inverse fine-structure constant from `COMMON/m_constants.f90`.
pub const SCREEN_ALPHA_INVERSE: Real = 137.035_989_56;
/// FEFF fine-structure constant `alphfs`.
pub const SCREEN_FINE_STRUCTURE_ALPHA: Real = 1.0 / SCREEN_ALPHA_INVERSE;
/// FEFF Bohr radius in Angstrom, `bohr` from `COMMON/m_constants.f90`.
pub const SCREEN_BOHR_ANGSTROM: Real = 0.529_177_249;
/// FEFF Hartree energy in eV, `hart` from `COMMON/m_constants.f90`.
pub const SCREEN_HARTREE_EV: Real = 27.211_396;

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

/// Inputs for [`screen_fms_response_slice`].
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

/// Inputs for [`screen_crpa_response_slice`].
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

/// Port of SCREEN `setri`: build the logarithmic radial grid.
///
/// FEFF stores radial samples as `ri(i) = exp(-x0 + (i-1)*dx)` using 1-based
/// loop bounds. This helper returns the same values in Rust's zero-based
/// [`ndarray::Array1`] layout.
pub fn screen_radial_grid(dx: Real, x0: Real, count: usize) -> Result<RealVec, ScreenError> {
    validate_positive("dx", dx)?;
    validate_finite("x0", x0)?;
    if count == 0 {
        return Err(ScreenError::EmptyRadialGrid);
    }

    Ok(Array1::from_iter(
        (0..count).map(|index| (-x0 + index as Real * dx).exp()),
    ))
}

/// Port of SCREEN `SetEGrid`: exponential grid on the imaginary axis.
///
/// FEFF fills `em(ie) = i * (exp((ne-ie)*dx)-1)`, where
/// `dx = log(emax+1)/(ne-1)`. The resulting table runs from `i*emax`
/// down to zero, matching the reference routine's storage order.
pub fn screen_exponential_energy_grid(
    max_imaginary_energy: Real,
    count: usize,
) -> Result<ComplexVec, ScreenError> {
    validate_positive("max_imaginary_energy", max_imaginary_energy)?;
    validate_count_at_least("energy", count, 2)?;

    let denominator = (count - 1) as Real;
    let dx = (max_imaginary_energy + 1.0).ln() / denominator;
    Ok(Array1::from_iter((1..=count).map(|index_1based| {
        let scaled = (count - index_1based) as Real * dx;
        Complex::new(0.0, scaled.exp() - 1.0)
    })))
}

/// Port of SCREEN `setegi`: rectangular complex-energy contour.
///
/// FEFF starts at `emax + i*ermin`, climbs the imaginary branch to `eimax`,
/// steps across the top edge toward `emin`, descends back to `ermin`, and then
/// reverses the table. Non-positive `ermin` is clamped to `0.05` before any
/// step sizes are computed.
pub fn screen_contour_energy_grid(
    input: ScreenContourEnergyGridInput,
) -> Result<ScreenContourEnergyGrid, ScreenError> {
    validate_finite("min_real_energy", input.min_real_energy)?;
    validate_finite("max_real_energy", input.max_real_energy)?;
    validate_finite("max_imaginary_energy", input.max_imaginary_energy)?;
    validate_finite("min_imaginary_energy", input.min_imaginary_energy)?;
    validate_count_at_least("real_points", input.real_points, 2)?;
    validate_count_at_least("imaginary_points", input.imaginary_points, 2)?;
    validate_count_at_least("max_points", input.max_points, 1)?;
    validate_increasing(
        "min_real_energy",
        input.min_real_energy,
        "max_real_energy",
        input.max_real_energy,
    )?;

    let effective_min_imaginary_energy = if input.min_imaginary_energy <= 0.0 {
        0.05
    } else {
        input.min_imaginary_energy
    };
    validate_increasing(
        "min_imaginary_energy",
        effective_min_imaginary_energy,
        "max_imaginary_energy",
        input.max_imaginary_energy,
    )?;

    let max_iterations = input
        .max_points
        .checked_mul(input.max_points)
        .ok_or(ScreenError::EnergyGridSizeOverflow { name: "max_points" })?;
    let real_step =
        (input.max_real_energy - input.min_real_energy) / (input.real_points - 1) as Real;
    let imaginary_step = Complex::new(
        0.0,
        (input.max_imaginary_energy - effective_min_imaginary_energy)
            / (input.imaginary_points - 1) as Real,
    );

    let mut points = Vec::with_capacity(input.max_points.min(max_iterations));
    points.push(Complex::new(
        input.max_real_energy,
        effective_min_imaginary_energy,
    ));
    let mut accumulated_imaginary = effective_min_imaginary_energy;
    let mut delta = imaginary_step;

    for index_1based in 2..=max_iterations {
        let previous = points.last().copied().ok_or(ScreenError::EmptyEnergyGrid)?;
        if previous.re < input.min_real_energy {
            delta = -imaginary_step;
            if previous.im <= effective_min_imaginary_energy {
                let active_len = if previous.im <= 0.0 {
                    index_1based - 2
                } else {
                    index_1based - 1
                };
                points.truncate(active_len);
                break;
            }
        } else if accumulated_imaginary.abs() >= input.max_imaginary_energy {
            delta = Complex::new(-real_step, 0.0);
            accumulated_imaginary = 0.0;
        }

        accumulated_imaginary += delta.im.abs();
        points.push(previous + delta);
    }

    if points.len() > input.max_points {
        return Err(ScreenError::EnergyGridTooLong {
            required: points.len(),
            available: input.max_points,
        });
    }

    let active_len = points.len();
    let mut energies = Array1::<Complex>::zeros(input.max_points);
    for (index, energy) in points.into_iter().rev().enumerate() {
        energies[index] = energy;
    }

    Ok(ScreenContourEnergyGrid {
        energies,
        active_len,
        effective_min_imaginary_energy,
    })
}

/// Port of SCREEN `getiat`: map a radius to FEFF's 1-based radial index.
///
/// Fortran assigns the floating-point expression to an integer, which truncates
/// toward zero. Returning an `isize` preserves that behavior for callers that
/// need to handle out-of-grid locations explicitly. Values reconstructed from
/// the same logarithmic grid are snapped back to exact integer boundaries when
/// roundoff alone would move them just below the FEFF index.
pub fn screen_radial_index_1based(x0: Real, dx: Real, radius: Real) -> Result<isize, ScreenError> {
    validate_finite("x0", x0)?;
    validate_positive("dx", dx)?;
    validate_positive("radius", radius)?;

    let value = (radius.ln() + x0) / dx + 1.0;
    if value < isize::MIN as Real || value > isize::MAX as Real {
        return Err(ScreenError::RadialIndexOutOfRange { value });
    }
    Ok(feff_truncated_index(value))
}

/// Port the shared SCREEN/CRPA `jri`, `jnrm`, and `ilast` setup.
///
/// `screensub.f90` and `CRPA/chi_crpa.f90` both derive active radial bounds
/// from `getiat`: `jri = getiat(rmt) + 1`, `jri1 = jri + 1`,
/// `jnrm = getiat(rnrm) + 1`, and `ilast = min(jnrm + 6 + iend, nrx)`.
/// The returned indices keep those FEFF 1-based names so callers can mirror
/// the original handoff logic while converting to zero-based slices locally.
pub fn screen_radial_bounds(
    input: ScreenRadialBoundsInput,
) -> Result<ScreenRadialBounds, ScreenError> {
    validate_positive("dx", input.dx)?;
    validate_finite("x0", input.x0)?;
    validate_positive("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_positive("norman_radius", input.norman_radius)?;
    validate_count_at_least("radial_capacity", input.radial_capacity, 1)?;
    validate_count_at_least("response_capacity", input.response_capacity, 1)?;

    let muffin_tin_base = screen_radial_index_1based(input.x0, input.dx, input.muffin_tin_radius)?;
    let muffin_tin_value = checked_radial_add("muffin_tin_index_1based", muffin_tin_base, 1)?;
    let muffin_tin_index_1based =
        positive_radial_bound("muffin_tin_index_1based", muffin_tin_value)?;
    let muffin_tin_next_value =
        checked_radial_add("muffin_tin_next_index_1based", muffin_tin_value, 1)?;
    let muffin_tin_next_index_1based =
        positive_radial_bound("muffin_tin_next_index_1based", muffin_tin_next_value)?;
    if muffin_tin_next_index_1based > input.radial_capacity {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "muffin_tin_next_index_1based",
            value: muffin_tin_next_index_1based,
            capacity: input.radial_capacity,
        });
    }

    let norman_base = screen_radial_index_1based(input.x0, input.dx, input.norman_radius)?;
    let norman_value = checked_radial_add("norman_index_1based", norman_base, 1)?;
    let norman_index_1based = positive_radial_bound("norman_index_1based", norman_value)?;
    let active_tail_value = checked_radial_add("active_count", norman_value, 6)?;
    let active_value = checked_radial_add("active_count", active_tail_value, input.tail_extension)?;
    let unclamped_active_count = positive_radial_bound("active_count", active_value)?;
    let active_count = unclamped_active_count.min(input.response_capacity);

    Ok(ScreenRadialBounds {
        muffin_tin_index_1based,
        muffin_tin_next_index_1based,
        norman_index_1based,
        active_count,
    })
}

/// Port the radial-bound setup from SCREEN `getph.f90`.
///
/// `getph` uses the same Loucks-grid index helper as `screensub`, but its
/// bounds are slightly different: only `jri` is checked against `nrptx`, there
/// is no `jri + 1` reference-potential bound, and `ilast` is clamped to the
/// radial wavefunction capacity rather than a response workspace.
pub fn screen_getph_radial_bounds(
    input: ScreenGetphRadialBoundsInput,
) -> Result<ScreenGetphRadialBounds, ScreenError> {
    validate_positive("dx", input.dx)?;
    validate_finite("x0", input.x0)?;
    validate_positive("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_positive("norman_radius", input.norman_radius)?;
    validate_count_at_least("radial_capacity", input.radial_capacity, 1)?;

    let muffin_tin_base = screen_radial_index_1based(input.x0, input.dx, input.muffin_tin_radius)?;
    let muffin_tin_value = checked_radial_add("getph_muffin_tin_index_1based", muffin_tin_base, 1)?;
    let muffin_tin_index_1based =
        positive_radial_bound("getph_muffin_tin_index_1based", muffin_tin_value)?;
    if muffin_tin_index_1based > input.radial_capacity {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "getph_muffin_tin_index_1based",
            value: muffin_tin_index_1based,
            capacity: input.radial_capacity,
        });
    }

    let norman_base = screen_radial_index_1based(input.x0, input.dx, input.norman_radius)?;
    let norman_value = checked_radial_add("getph_norman_index_1based", norman_base, 1)?;
    let norman_index_1based = positive_radial_bound("getph_norman_index_1based", norman_value)?;
    let active_value = checked_radial_add("getph_active_count", norman_value, 6)?;
    let unclamped_active_count = positive_radial_bound("getph_active_count", active_value)?;
    let active_count = unclamped_active_count.min(input.radial_capacity);

    Ok(ScreenGetphRadialBounds {
        muffin_tin_index_1based,
        norman_index_1based,
        active_count,
    })
}

/// Port the per-energy setup shared by `screensub.f90` and `chi_crpa.f90`.
///
/// For each contour point, FEFF computes `p2 = em(ie) - eref`,
/// `ck = sqrt(2*p2 + (p2*alphfs)**2)`, converts `ck` to the single-precision
/// `cks(1)` value passed into FMS, forms `xkmt = rmt * ck`, and chooses the
/// number of Dirac correction cycles from `mod(ixc0, 10)`.
pub fn screen_energy_state(
    input: ScreenEnergyStateInput,
) -> Result<ScreenEnergyState, ScreenError> {
    validate_finite_complex_input("energy", input.energy)?;
    validate_finite_complex_input("reference_energy", input.reference_energy)?;
    validate_positive("muffin_tin_radius", input.muffin_tin_radius)?;

    let kinetic_energy = input.energy - input.reference_energy;
    validate_result_finite_complex("kinetic_energy", kinetic_energy)?;
    let alpha_scaled = kinetic_energy * SCREEN_FINE_STRUCTURE_ALPHA;
    let wave_number = (kinetic_energy * 2.0 + alpha_scaled * alpha_scaled).sqrt();
    validate_result_finite_complex("wave_number", wave_number)?;
    let muffin_tin_argument = wave_number * input.muffin_tin_radius;
    validate_result_finite_complex("muffin_tin_argument", muffin_tin_argument)?;
    let fms_wave_number = complex32_result("fms_wave_number", wave_number)?;
    let dirac_cycle_count = if input.exchange_selector % 10 < 5 {
        0
    } else {
        3
    };

    Ok(ScreenEnergyState {
        kinetic_energy,
        wave_number,
        fms_wave_number,
        muffin_tin_argument,
        dirac_cycle_count,
    })
}

/// Port the angular-momentum selector from SCREEN `getph.f90`.
///
/// FEFF starts from the requested phase-shift `lmaxsc`, caps it by the global
/// `lx`, then applies historical light-element overrides: elements up to Be
/// use `lmax = 2`, and H/He use `lmax = 1`. The overrides intentionally replace
/// the previous cap, matching the original assignment order.
pub fn screen_getph_lmax(
    atomic_number: usize,
    requested_lmax: usize,
    angular_capacity_lx: usize,
) -> Result<usize, ScreenError> {
    validate_count_at_least("atomic_number", atomic_number, 1)?;

    let mut lmax = requested_lmax.min(angular_capacity_lx);
    if atomic_number <= 4 {
        lmax = 2;
    }
    if atomic_number <= 2 {
        lmax = 1;
    }
    Ok(lmax)
}

/// Port the SCREEN/CRPA radial-solution normalization scalar setup.
///
/// After `phamp`, `screensub.f90` and `chi_crpa.f90` compute the relativistic
/// lower-component factor, `dum1`, and the regular-solution scale `xfnorm`.
/// FEFF sets `xfnorm` to zero when the phase amplitude is exactly zero; this
/// helper preserves that branch and validates all finite complex results.
pub fn screen_solution_normalization(
    input: ScreenSolutionNormalizationInput,
) -> Result<ScreenSolutionNormalization, ScreenError> {
    validate_finite_complex_input("wave_number", input.wave_number)?;
    validate_finite_complex_input("phase_amplitude", input.phase_amplitude)?;

    let one = Complex::new(1.0, 0.0);
    let zero = Complex::new(0.0, 0.0);
    let alpha_scaled = input.wave_number * SCREEN_FINE_STRUCTURE_ALPHA;
    let lower_denominator = one + (one + alpha_scaled * alpha_scaled).sqrt();
    validate_result_finite_complex("small_component_denominator", lower_denominator)?;
    if lower_denominator == zero {
        return Err(ScreenError::ZeroComplexResult {
            name: "small_component_denominator",
        });
    }

    let small_component_factor = -alpha_scaled / lower_denominator;
    validate_result_finite_complex("small_component_factor", small_component_factor)?;
    let scale_denominator = (one + small_component_factor * small_component_factor).sqrt();
    validate_result_finite_complex("relativistic_scale_denominator", scale_denominator)?;
    if scale_denominator == zero {
        return Err(ScreenError::ZeroComplexResult {
            name: "relativistic_scale_denominator",
        });
    }

    let relativistic_scale = one / scale_denominator;
    validate_result_finite_complex("relativistic_scale", relativistic_scale)?;
    let regular_solution_scale = if input.phase_amplitude == zero {
        zero
    } else {
        let scale = relativistic_scale / input.phase_amplitude;
        validate_result_finite_complex("regular_solution_scale", scale)?;
        scale
    };

    Ok(ScreenSolutionNormalization {
        small_component_factor,
        relativistic_scale,
        regular_solution_scale,
    })
}

/// Port the irregular muffin-tin boundary values from SCREEN `screensub.f90`.
///
/// Before calling `dfovrg` with `irr = 1`, FEFF initializes the irregular
/// solution from either the standing-wave `N*cos(ph0)+J*sin(ph0)` expression or
/// the outgoing-Hankel branch selected by `irrh == 1`. This helper computes the
/// two complex boundary values while reusing the same relativistic `factor` and
/// `dum1` terms as [`screen_solution_normalization`].
pub fn screen_irregular_initial_condition(
    input: ScreenIrregularInitialConditionInput,
) -> Result<ScreenIrregularInitialCondition, ScreenError> {
    validate_positive("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_finite_complex_input("phase_shift", input.phase_shift)?;
    validate_finite_complex_input("wave_number", input.wave_number)?;
    validate_finite_complex_input("bessel_j_l", input.bessel_j_l)?;
    validate_finite_complex_input("bessel_j_l_plus_1", input.bessel_j_l_plus_1)?;
    validate_finite_complex_input("neumann_l", input.neumann_l)?;
    validate_finite_complex_input("neumann_l_plus_1", input.neumann_l_plus_1)?;

    let normalization = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number: input.wave_number,
        phase_amplitude: Complex::new(1.0, 0.0),
    })?;
    let relativistic_scale = normalization.relativistic_scale;
    let small_component_factor = normalization.small_component_factor;

    let radius_scale = input.muffin_tin_radius * relativistic_scale;
    let (large_component, small_component) = if input.use_hankel_boundary {
        validate_finite_complex_input("hankel_l", input.hankel_l)?;
        validate_finite_complex_input("hankel_l_plus_1", input.hankel_l_plus_1)?;
        let phase_factor = (Complex::new(0.0, 1.0) * input.phase_shift).exp();
        validate_result_finite_complex("irregular_phase_factor", phase_factor)?;
        (
            input.hankel_l * phase_factor * radius_scale,
            input.hankel_l_plus_1 * phase_factor * radius_scale * small_component_factor,
        )
    } else {
        let cos_phase = input.phase_shift.cos();
        let sin_phase = input.phase_shift.sin();
        validate_result_finite_complex("irregular_cos_phase", cos_phase)?;
        validate_result_finite_complex("irregular_sin_phase", sin_phase)?;
        (
            (input.neumann_l * cos_phase + input.bessel_j_l * sin_phase) * radius_scale,
            (input.neumann_l_plus_1 * cos_phase + input.bessel_j_l_plus_1 * sin_phase)
                * radius_scale
                * small_component_factor,
        )
    };

    validate_result_finite_complex("irregular_large_component", large_component)?;
    validate_result_finite_complex("irregular_small_component", small_component)?;
    Ok(ScreenIrregularInitialCondition {
        large_component,
        small_component,
    })
}

/// Port the irregular-solution Wronskian rescaling from SCREEN `screensub.f90`.
///
/// After the irregular `dfovrg` pass, FEFF computes a complex Wronskian at
/// `jri`, inverts it with the photoelectron wave number, and multiplies both
/// irregular radial components by `exp(i*ph0) * qu`. A zero Wronskian follows
/// the original branch and yields a zero scale instead of dividing.
pub fn screen_irregular_wronskian_scale(
    input: ScreenIrregularWronskianScaleInput,
) -> Result<ScreenIrregularWronskianScale, ScreenError> {
    validate_finite_complex_input("phase_shift", input.phase_shift)?;
    validate_finite_complex_input("wave_number", input.wave_number)?;
    validate_finite_complex_input("regular_large_at_match", input.regular_large_at_match)?;
    validate_finite_complex_input("regular_small_at_match", input.regular_small_at_match)?;
    validate_finite_complex_input("irregular_large_at_match", input.irregular_large_at_match)?;
    validate_finite_complex_input("irregular_small_at_match", input.irregular_small_at_match)?;

    let phase_factor = (Complex::new(0.0, 1.0) * input.phase_shift).exp();
    validate_result_finite_complex("wronskian_phase_factor", phase_factor)?;
    let denominator = 2.0
        * SCREEN_ALPHA_INVERSE
        * phase_factor
        * (input.irregular_large_at_match * input.regular_small_at_match
            - input.regular_large_at_match * input.irregular_small_at_match);
    validate_result_finite_complex("wronskian_denominator", denominator)?;

    let zero = Complex::new(0.0, 0.0);
    let reciprocal_wave_scale = if denominator == zero {
        zero
    } else {
        if input.wave_number == zero {
            return Err(ScreenError::ZeroComplexResult {
                name: "wave_number",
            });
        }
        let value = Complex::new(1.0, 0.0) / denominator / input.wave_number;
        validate_result_finite_complex("wronskian_reciprocal_wave_scale", value)?;
        value
    };
    let irregular_solution_scale = phase_factor * reciprocal_wave_scale;
    validate_result_finite_complex(
        "wronskian_irregular_solution_scale",
        irregular_solution_scale,
    )?;

    Ok(ScreenIrregularWronskianScale {
        phase_factor,
        denominator,
        reciprocal_wave_scale,
        irregular_solution_scale,
    })
}

/// Port the exact radial continuation from SCREEN `screensub.f90`.
///
/// After the regular and irregular `dfovrg` solutions are normalized, FEFF
/// overwrites rows `jri:ilast` with exact free-particle combinations evaluated
/// at `xck = ck * ri(j)`. This scalar helper computes one such row from the
/// already-evaluated Bessel, Neumann, and Hankel values; the caller owns the
/// radial loop and special-function evaluation.
pub fn screen_exact_radial_continuation(
    input: ScreenExactRadialContinuationInput,
) -> Result<ScreenExactRadialContinuation, ScreenError> {
    validate_positive("radius", input.radius)?;
    validate_finite_complex_input("phase_shift", input.phase_shift)?;
    validate_finite_complex_input("wave_number", input.wave_number)?;
    validate_finite_complex_input("bessel_j_l", input.bessel_j_l)?;
    validate_finite_complex_input("bessel_j_l_plus_1", input.bessel_j_l_plus_1)?;
    validate_finite_complex_input("neumann_l", input.neumann_l)?;
    validate_finite_complex_input("neumann_l_plus_1", input.neumann_l_plus_1)?;
    validate_finite_complex_input("hankel_l", input.hankel_l)?;
    validate_finite_complex_input("hankel_l_plus_1", input.hankel_l_plus_1)?;

    let normalization = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number: input.wave_number,
        phase_amplitude: Complex::new(1.0, 0.0),
    })?;
    let relativistic_scale = normalization.relativistic_scale;
    let small_component_factor = normalization.small_component_factor;
    let radius_scale = input.radius * relativistic_scale;
    let cos_phase = input.phase_shift.cos();
    let sin_phase = input.phase_shift.sin();
    let phase_factor = (Complex::new(0.0, 1.0) * input.phase_shift).exp();
    validate_result_finite_complex("exact_continuation_cos_phase", cos_phase)?;
    validate_result_finite_complex("exact_continuation_sin_phase", sin_phase)?;
    validate_result_finite_complex("exact_continuation_phase_factor", phase_factor)?;

    let regular_large_component =
        (input.bessel_j_l * cos_phase - input.neumann_l * sin_phase) * radius_scale;
    let regular_small_component = (input.bessel_j_l_plus_1 * cos_phase
        - input.neumann_l_plus_1 * sin_phase)
        * radius_scale
        * small_component_factor;
    let irregular_large_component = input.hankel_l * phase_factor * radius_scale;
    let irregular_small_component =
        input.hankel_l_plus_1 * phase_factor * radius_scale * small_component_factor;

    validate_result_finite_complex("exact_regular_large_component", regular_large_component)?;
    validate_result_finite_complex("exact_regular_small_component", regular_small_component)?;
    validate_result_finite_complex("exact_irregular_large_component", irregular_large_component)?;
    validate_result_finite_complex("exact_irregular_small_component", irregular_small_component)?;

    Ok(ScreenExactRadialContinuation {
        regular_large_component,
        regular_small_component,
        irregular_large_component,
        irregular_small_component,
    })
}

/// Port the unit setup block from SCREEN `rdgeom.f90`.
///
/// FEFF clamps `ScreenI%maxl` to `lx + 1`, converts atomic coordinates and
/// FMS radii from Angstrom to bohr, and converts SCREEN contour energies from
/// eV to Hartree before the screening driver starts. This helper keeps that
/// setup separate from the full file-reading routine so callers can apply it
/// to already-parsed Rust inputs.
pub fn screen_rdgeom_atomic_units(
    input: ScreenRdgeomAtomicUnitsInput<'_>,
) -> Result<ScreenRdgeomAtomicUnits, ScreenError> {
    let (_, columns) = input.atom_positions_angstrom.dim();
    if columns != 3 {
        return Err(ScreenError::AtomPositionColumnCount { columns });
    }

    validate_finite("rfms2_angstrom", input.rfms2_angstrom)?;
    validate_finite("direct_radius_angstrom", input.direct_radius_angstrom)?;
    validate_finite("min_real_energy_ev", input.min_real_energy_ev)?;
    validate_finite("max_real_energy_ev", input.max_real_energy_ev)?;
    validate_finite("max_imaginary_energy_ev", input.max_imaginary_energy_ev)?;
    validate_finite("screen_rfms_angstrom", input.screen_rfms_angstrom)?;
    validate_finite("min_imaginary_energy_ev", input.min_imaginary_energy_ev)?;

    let mut atom_positions_bohr =
        Array2::zeros((input.atom_positions_angstrom.nrows(), columns).f());
    for ((row, column), value) in input.atom_positions_angstrom.indexed_iter() {
        validate_finite_matrix("atom_positions_angstrom", row, column, *value)?;
        let converted = *value / SCREEN_BOHR_ANGSTROM;
        validate_result_finite("atom_position_bohr", converted)?;
        atom_positions_bohr[(row, column)] = converted;
    }

    let angular_count_cap =
        input
            .angular_capacity_lx
            .checked_add(1)
            .ok_or(ScreenError::IndexSizeOverflow {
                name: "angular_capacity_lx",
            })?;
    let converted = ScreenRdgeomAtomicUnits {
        atom_positions_bohr,
        rfms2_bohr: input.rfms2_angstrom / SCREEN_BOHR_ANGSTROM,
        direct_radius_bohr: input.direct_radius_angstrom / SCREEN_BOHR_ANGSTROM,
        min_real_energy_hartree: input.min_real_energy_ev / SCREEN_HARTREE_EV,
        max_real_energy_hartree: input.max_real_energy_ev / SCREEN_HARTREE_EV,
        max_imaginary_energy_hartree: input.max_imaginary_energy_ev / SCREEN_HARTREE_EV,
        screen_rfms_bohr: input.screen_rfms_angstrom / SCREEN_BOHR_ANGSTROM,
        min_imaginary_energy_hartree: input.min_imaginary_energy_ev / SCREEN_HARTREE_EV,
        max_l: input.max_l.min(angular_count_cap),
    };
    validate_result_finite("rfms2_bohr", converted.rfms2_bohr)?;
    validate_result_finite("direct_radius_bohr", converted.direct_radius_bohr)?;
    validate_result_finite("min_real_energy_hartree", converted.min_real_energy_hartree)?;
    validate_result_finite("max_real_energy_hartree", converted.max_real_energy_hartree)?;
    validate_result_finite(
        "max_imaginary_energy_hartree",
        converted.max_imaginary_energy_hartree,
    )?;
    validate_result_finite("screen_rfms_bohr", converted.screen_rfms_bohr)?;
    validate_result_finite(
        "min_imaginary_energy_hartree",
        converted.min_imaginary_energy_hartree,
    )?;

    Ok(converted)
}

/// Port the phase-potential reference shift from SCREEN `prep.f90`.
///
/// After `fixvar`, FEFF chooses `eref(1) = vtotph(jri1)`, subtracts that
/// reference from `vtotph(1:jri1)`, and either subtracts it from
/// `vvalph(1:jri1)` (`ixc >= 5`) or copies the shifted total potential into
/// `vvalph(1:jri1)`. Entries after `jri1` are left untouched, matching the
/// Fortran loop bounds.
pub fn screen_phase_potential_reference_shift(
    input: ScreenPhasePotentialInput<'_>,
) -> Result<ScreenPhasePotential, ScreenError> {
    validate_count_at_least(
        "muffin_tin_next_index_1based",
        input.muffin_tin_next_index_1based,
        1,
    )?;
    if input.muffin_tin_next_index_1based > input.total_potential.len() {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "muffin_tin_next_index_1based",
            value: input.muffin_tin_next_index_1based,
            capacity: input.total_potential.len(),
        });
    }
    if input.muffin_tin_next_index_1based > input.valence_potential.len() {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "muffin_tin_next_index_1based",
            value: input.muffin_tin_next_index_1based,
            capacity: input.valence_potential.len(),
        });
    }

    let prefix_len = input.muffin_tin_next_index_1based;
    let reference_index = prefix_len - 1;
    let reference_energy = input.total_potential[reference_index];
    validate_finite("reference_potential", reference_energy)?;

    let mut total_potential = input.total_potential.to_owned();
    let mut valence_potential = input.valence_potential.to_owned();
    for index in 0..prefix_len {
        let total = input.total_potential[index];
        let valence = input.valence_potential[index];
        validate_finite("total_potential", total)?;
        validate_finite("valence_potential", valence)?;

        let shifted_total = total - reference_energy;
        validate_result_finite("shifted_total_potential", shifted_total)?;
        total_potential[index] = shifted_total;
        valence_potential[index] = if input.exchange_selector >= 5 {
            let shifted_valence = valence - reference_energy;
            validate_result_finite("shifted_valence_potential", shifted_valence)?;
            shifted_valence
        } else {
            shifted_total
        };
    }

    Ok(ScreenPhasePotential {
        reference_energy,
        total_potential,
        valence_potential,
    })
}

fn feff_truncated_index(value: Real) -> isize {
    let nearest = value.round();
    let tolerance = 1.0e-12 * nearest.abs().max(1.0);
    if value >= 0.0 && (value - nearest).abs() <= tolerance {
        nearest as isize
    } else {
        value.trunc() as isize
    }
}

/// Port of SCREEN `ldafxc`: local-density exchange-correlation kernel.
///
/// FEFF evaluates only the first `active_count` rows, sets non-positive
/// electron-density rows to zero, and uses a pure-exchange branch when
/// `exchange_selector == 2`.
pub fn screen_lda_exchange_correlation_kernel(
    radii: &[Real],
    electron_density: &[Real],
    exchange_selector: i32,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, electron_density.len())?;

    let mut output = Array1::zeros(active_count);
    for index in 0..active_count {
        let radius = radii[index];
        let density = electron_density[index];
        validate_positive("radius", radius)?;
        validate_finite("electron_density", density)?;
        if density <= 0.0 {
            continue;
        }

        let rs = (density / 3.0).powf(-1.0 / 3.0);
        let exchange = -1.222 / rs;
        let correlation = if exchange_selector == 2 {
            0.0
        } else {
            -0.75924 / (11.4 + rs)
        };
        output[index] = rs.powi(3) / radius.powi(2) / 6.0 * (exchange + correlation);
    }
    Ok(output)
}

/// Port the SCREEN/CRPA radial Coulomb response kernel setup.
///
/// FEFF fills the upper triangle as `K(m,n) = 4*pi/r(n)`, mirrors it into the
/// lower triangle, and optionally adds `4*pi*fxc(i)` to the diagonal for TDLDA
/// runs. Because the FEFF radial grid is monotonically increasing, the
/// symmetric result is `4*pi/max(r_i, r_j)` plus the optional diagonal local
/// exchange-correlation term. The returned matrix uses Fortran-order
/// [`ndarray::Array2`] storage so downstream solver code can preserve FEFF's
/// column-major traversal.
pub fn screen_coulomb_kernel_matrix(
    radii: &[Real],
    active_count: usize,
    local_kernel: Option<&[Real]>,
) -> Result<RealMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    if let Some(local_kernel) = local_kernel {
        validate_active_count(active_count, local_kernel.len())?;
    }

    for &radius in radii.iter().take(active_count) {
        validate_positive("radius", radius)?;
    }

    let scale = 4.0 * std::f64::consts::PI;
    let mut matrix = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        for column in row..active_count {
            let value = scale / radii[column];
            matrix[(row, column)] = value;
            matrix[(column, row)] = value;
        }
    }
    if let Some(local_kernel) = local_kernel {
        for index in 0..active_count {
            let value = local_kernel[index];
            validate_finite("local_kernel", value)?;
            matrix[(index, index)] += scale * value;
        }
    }
    Ok(matrix)
}

/// Port the SCREEN bare core-hole potential setup.
///
/// FEFF first forms shell weights
/// `(dgc0(i)^2 + dpc0(i)^2) * dx * r(i)`, then evaluates the radial Coulomb
/// potential `int rho(r') / max(r, r') dr'`. This helper returns FEFF's final
/// `vch = wscrn` vector. The implementation uses prefix and suffix reductions
/// instead of the original nested loops, preserving the same mathematical
/// expression with linear complexity.
pub fn screen_bare_core_hole_potential(
    radii: &[Real],
    large_component: &[Real],
    small_component: &[Real],
    dx: Real,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, large_component.len())?;
    validate_active_count(active_count, small_component.len())?;

    let mut shell_weight = Vec::with_capacity(active_count);
    for index in 0..active_count {
        let radius = radii[index];
        let large = large_component[index];
        let small = small_component[index];
        validate_positive("radius", radius)?;
        validate_finite("large_component", large)?;
        validate_finite("small_component", small)?;
        let radial_density = large.mul_add(large, small * small);
        let shell = radial_density * dx * radius;
        validate_result_finite("core_hole_shell_weight", shell)?;
        shell_weight.push(shell);
    }

    screen_radial_coulomb_potential(radii, &shell_weight, active_count)
}

/// Evaluate FEFF's radial Coulomb potential from shell weights.
///
/// Both `SCREEN/screensub.f90` and `CRPA/chi_crpa.f90` form radial shell
/// weights first and then evaluate `sum_j weight(j) / max(r_i, r_j)`. This
/// helper keeps that common loop available for core-hole and CRPA density
/// sources. Prefix and suffix reductions preserve the FEFF expression while
/// avoiding the original nested-loop cost.
pub fn screen_radial_coulomb_potential(
    radii: &[Real],
    shell_weights: &[Real],
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, shell_weights.len())?;

    let mut outer_weight = Vec::with_capacity(active_count);
    for index in 0..active_count {
        let radius = radii[index];
        let shell = shell_weights[index];
        validate_positive("radius", radius)?;
        validate_finite("shell_weight", shell)?;
        outer_weight.push(shell / radius);
    }

    let mut tail = vec![0.0; active_count + 1];
    for index in (0..active_count).rev() {
        tail[index] = tail[index + 1] + outer_weight[index];
        validate_result_finite("radial_coulomb_tail_weight", tail[index])?;
    }

    let mut prefix = 0.0;
    let mut output = Array1::zeros(active_count);
    for index in 0..active_count {
        prefix += shell_weights[index];
        validate_result_finite("radial_coulomb_prefix_weight", prefix)?;
        let value = prefix / radii[index] + tail[index + 1];
        validate_result_finite("radial_coulomb_potential", value)?;
        output[index] = value;
    }
    Ok(output)
}

/// Port the CRPA total-density projection and normalization setup.
///
/// `CRPA/chi_crpa.f90` optionally damps the total density by a
/// `cos(...)^4` radial window, normalizes `sum rho(r_i) * r_i * dx` to one,
/// and forms shell weights for the following Coulomb-potential loop. FEFF then
/// zeros `vch(jnrm+1:)`; pass `norman_count = jnrm` to preserve that active
/// prefix.
pub fn screen_crpa_density_weights(
    radii: &[Real],
    total_density: &[Real],
    dx: Real,
    active_count: usize,
    norman_count: usize,
    projection_window: Option<ScreenCrpaProjectionWindow>,
) -> Result<ScreenCrpaDensityWeights, ScreenError> {
    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_count_at_least("norman_count", norman_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, total_density.len())?;
    if norman_count > active_count {
        return Err(ScreenError::ActiveCountOutOfRange {
            active_count: norman_count,
            len: active_count,
        });
    }
    if let Some(window) = projection_window {
        validate_finite("projection_inner_radius", window.inner_radius)?;
        validate_finite("projection_outer_radius", window.outer_radius)?;
        validate_increasing(
            "projection_inner_radius",
            window.inner_radius,
            "projection_outer_radius",
            window.outer_radius,
        )?;
    }

    let mut projected_density = Vec::with_capacity(active_count);
    let mut normalization = 0.0;
    for index in 0..active_count {
        let radius = radii[index];
        let mut density = total_density[index];
        validate_positive("radius", radius)?;
        validate_finite("total_density", density)?;
        if let Some(window) = projection_window {
            let clamped_radius = radius.max(window.inner_radius).min(window.outer_radius);
            let scaled = (clamped_radius - window.inner_radius)
                / (window.outer_radius - window.inner_radius);
            density *= (scaled * std::f64::consts::FRAC_PI_2).cos().powi(4);
            validate_result_finite("projected_crpa_density", density)?;
        }
        normalization += density * radius * dx;
        validate_result_finite("crpa_density_normalization", normalization)?;
        projected_density.push(density);
    }
    validate_positive_result("crpa_density_normalization", normalization)?;

    let mut normalized_density = Array1::zeros(active_count);
    let mut shell_weights = Array1::zeros(active_count);
    for index in 0..active_count {
        let density = projected_density[index] / normalization;
        validate_result_finite("normalized_crpa_density", density)?;
        normalized_density[index] = density;
        if index < norman_count {
            let shell = density * dx * radii[index];
            validate_result_finite("crpa_shell_weight", shell)?;
            shell_weights[index] = shell;
        }
    }

    Ok(ScreenCrpaDensityWeights {
        normalized_density,
        shell_weights,
        normalization,
    })
}

/// Port the CRPA Hubbard-parameter accumulation loop.
///
/// After solving the screened response equation, FEFF stores
/// `vch(i) = wscrn(i) * den_CRPA(i,ie)` and accumulates screened and bare
/// Hubbard interactions with the normalized total CRPA density:
/// `sum potential(i) * totden_CRPA(i) * dx * ri(i)`. The scalar outputs are the
/// values FEFF writes to `crpa.dat`; no Hartree-to-eV conversion is applied.
pub fn screen_crpa_hubbard_summary(
    radii: &[Real],
    screened_potential: &[Real],
    bare_potential: &[Real],
    total_density: &[Real],
    orbital_density: &[Real],
    dx: Real,
    active_count: usize,
) -> Result<ScreenCrpaHubbardSummary, ScreenError> {
    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, screened_potential.len())?;
    validate_active_count(active_count, bare_potential.len())?;
    validate_active_count(active_count, total_density.len())?;
    validate_active_count(active_count, orbital_density.len())?;

    let mut screened_density_potential = Array1::zeros(active_count);
    let mut hubbard_u = 0.0;
    let mut bare_u = 0.0;
    let mut occupation = 0.0;
    for index in 0..active_count {
        let radius = radii[index];
        let screened = screened_potential[index];
        let bare = bare_potential[index];
        let total = total_density[index];
        let orbital = orbital_density[index];
        validate_positive("radius", radius)?;
        validate_finite("screened_potential", screened)?;
        validate_finite("bare_potential", bare)?;
        validate_finite("total_density", total)?;
        validate_finite("orbital_density", orbital)?;

        let density_potential = screened * orbital;
        validate_result_finite("crpa_screened_density_potential", density_potential)?;
        screened_density_potential[index] = density_potential;

        let weight = total * dx * radius;
        validate_result_finite("crpa_hubbard_weight", weight)?;
        hubbard_u += screened * weight;
        bare_u += bare * weight;
        occupation += weight;
        validate_result_finite("crpa_hubbard_u", hubbard_u)?;
        validate_result_finite("crpa_bare_u", bare_u)?;
        validate_result_finite("crpa_occupation", occupation)?;
    }

    Ok(ScreenCrpaHubbardSummary {
        screened_density_potential,
        hubbard_u,
        occupation,
        bare_u,
    })
}

/// Port the SCREEN/CRPA contour trapezoid energy step.
///
/// `screensub.f90` and `chi_crpa.f90` integrate each `chi0re(:,:,ie)` slice
/// with endpoint half-steps and centered interior steps:
/// `(em(ie+1) - em(ie-1)) / 2`. The `energy_index` argument is zero-based and
/// maps to FEFF's one-based `ie`.
pub fn screen_energy_integration_delta(
    energies: ArrayView1<'_, Complex>,
    energy_index: usize,
) -> Result<Complex, ScreenError> {
    validate_count_at_least("energies", energies.len(), 2)?;
    if energy_index >= energies.len() {
        return Err(ScreenError::EnergyIndexOutOfRange {
            index: energy_index,
            len: energies.len(),
        });
    }
    for &energy in energies {
        validate_finite_complex_input("energy", energy)?;
    }

    let delta = if energy_index == 0 {
        (energies[1] - energies[0]) / 2.0
    } else if energy_index + 1 == energies.len() {
        (energies[energy_index] - energies[energy_index - 1]) / 2.0
    } else {
        (energies[energy_index + 1] - energies[energy_index - 1]) / 2.0
    };
    validate_result_finite_complex("energy_integration_delta", delta)?;
    Ok(delta)
}

/// Accumulate one SCREEN/CRPA response slice into the contour integral.
///
/// FEFF stores only the active upper triangle during the energy loop:
/// `chi0r(ir1,i) += chi0re(ir1,i) * de` for `ir1 <= i`. This helper preserves
/// that convention and leaves the lower triangle from `accumulated` unchanged;
/// use [`screen_symmetrize_response_upper`] before building the response system.
pub fn screen_integrate_response_step(
    accumulated: ArrayView2<'_, Complex>,
    response_at_energy: ArrayView2<'_, Complex>,
    energy_delta: Complex,
    active_count: usize,
) -> Result<ComplexMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_matrix_shape(
        "accumulated_response",
        accumulated.nrows(),
        accumulated.ncols(),
        active_count,
    )?;
    validate_active_matrix_shape(
        "response_at_energy",
        response_at_energy.nrows(),
        response_at_energy.ncols(),
        active_count,
    )?;
    validate_finite_complex_input("energy_delta", energy_delta)?;

    let mut output = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        for column in 0..active_count {
            let value = accumulated[(row, column)];
            validate_finite_complex_matrix("accumulated_response", row, column, value)?;
            output[(row, column)] = value;
        }
    }
    for row in 0..active_count {
        for column in row..active_count {
            let response = response_at_energy[(row, column)];
            validate_finite_complex_matrix("response_at_energy", row, column, response)?;
            let value = output[(row, column)] + response * energy_delta;
            validate_result_finite_complex("integrated_response", value)?;
            output[(row, column)] = value;
        }
    }
    Ok(output)
}

/// Mirror FEFF's stored upper-triangle response matrix.
///
/// The original SCREEN/CRPA routines fill `chi0r(ir1,i)` only for `ir1 <= i`
/// during energy integration, then copy `chi0r(i,ir1)` into the lower triangle
/// before solving. This is a plain symmetric copy, not a Hermitian conjugate.
pub fn screen_symmetrize_response_upper(
    response_upper: ArrayView2<'_, Complex>,
    active_count: usize,
) -> Result<ComplexMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_matrix_shape(
        "response_upper",
        response_upper.nrows(),
        response_upper.ncols(),
        active_count,
    )?;

    let mut output = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        for column in row..active_count {
            let value = response_upper[(row, column)];
            validate_finite_complex_matrix("response_upper", row, column, value)?;
            output[(row, column)] = value;
            output[(column, row)] = value;
        }
    }
    Ok(output)
}

/// Port the CRPA angular-channel density row.
///
/// `chi_crpa.f90` stores
/// `DIMAG((pr*pn + pr**2*gtrl)*ck*4) * (2*l + 1) / pi` in `den_CRPA(:,ie)`
/// for the selected CRPA angular momentum. The regular and irregular radial
/// solutions are passed as `ndarray` views over FEFF's active radial prefix.
pub fn screen_crpa_orbital_density(
    regular_solution: ArrayView1<'_, Complex>,
    irregular_solution: ArrayView1<'_, Complex>,
    cluster_green: Complex,
    wave_number: Complex,
    angular_momentum: usize,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    validate_finite_complex_input("cluster_green", cluster_green)?;
    validate_finite_complex_input("wave_number", wave_number)?;

    let angular_scale = (2.0 * angular_momentum as Real + 1.0) / std::f64::consts::PI;
    let mut density = Array1::zeros(active_count);
    for index in 0..active_count {
        let regular = regular_solution[index];
        let irregular = irregular_solution[index];
        validate_finite_complex_input("regular_solution", regular)?;
        validate_finite_complex_input("irregular_solution", irregular)?;
        let response =
            (regular * irregular + regular * regular * cluster_green) * wave_number * 4.0;
        validate_result_finite_complex("crpa_orbital_density_response", response)?;
        let value = response.im * angular_scale;
        validate_result_finite("crpa_orbital_density", value)?;
        density[index] = value;
    }
    Ok(density)
}

/// Build one SCREEN atomic response slice.
///
/// In `screensub.f90`, each angular channel adds an upper-triangle contribution
/// `factor * r(m) * r(n) * pr(m)^2 * pn(n)^2`, where
/// `factor = -((2*l + 1) * (2*ck)^2 * dx^2) / (2*pi^2)`. The returned matrix
/// stores the active upper triangle in Fortran order; lower-triangle entries
/// remain zero until [`screen_symmetrize_response_upper`] is applied after
/// energy integration.
pub fn screen_atomic_response_slice(
    radii: &[Real],
    regular_solution: ArrayView1<'_, Complex>,
    irregular_solution: ArrayView1<'_, Complex>,
    wave_number: Complex,
    dx: Real,
    angular_momentum: usize,
    active_count: usize,
) -> Result<ComplexMat, ScreenError> {
    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    validate_finite_complex_input("wave_number", wave_number)?;

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let doubled_wave = wave_number * 2.0;
    let prefactor = doubled_wave
        * doubled_wave
        * (-(angular_weight * dx * dx) / (2.0 * std::f64::consts::PI.powi(2)));
    validate_result_finite_complex("atomic_response_prefactor", prefactor)?;

    let mut response = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        let row_radius = radii[row];
        let regular = regular_solution[row];
        validate_positive("radius", row_radius)?;
        validate_finite_complex_input("regular_solution", regular)?;
        for column in row..active_count {
            let column_radius = radii[column];
            let irregular = irregular_solution[column];
            validate_positive("radius", column_radius)?;
            validate_finite_complex_input("irregular_solution", irregular)?;
            let value =
                prefactor * row_radius * column_radius * regular * regular * irregular * irregular;
            validate_result_finite_complex("atomic_response_slice", value)?;
            response[(row, column)] = value;
        }
    }
    Ok(response)
}

/// Build one SCREEN FMS cluster response correction slice.
///
/// When the FMS cluster contains more than the absorber, `screensub.f90` adds a
/// `1:jnrm` upper-triangle correction to the atomic response slice:
/// `factor*r(m)*r(n)*(2*gtrl*pr(m)^2*pr(n)*pn(n) + gtrl^2*pr(m)^2*pr(n)^2)`.
/// `fms_count` is FEFF `jnrm`; entries outside that prefix remain zero.
pub fn screen_fms_response_slice(
    input: ScreenFmsResponseSliceInput<'_>,
) -> Result<ComplexMat, ScreenError> {
    let ScreenFmsResponseSliceInput {
        radii,
        regular_solution,
        irregular_solution,
        cluster_green,
        wave_number,
        dx,
        angular_momentum,
        active_count,
        fms_count,
    } = input;

    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_count_at_least("fms_count", fms_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    if fms_count > active_count {
        return Err(ScreenError::ActiveCountOutOfRange {
            active_count: fms_count,
            len: active_count,
        });
    }
    validate_finite_complex_input("cluster_green", cluster_green)?;
    validate_finite_complex_input("wave_number", wave_number)?;

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let doubled_wave = wave_number * 2.0;
    let prefactor = doubled_wave
        * doubled_wave
        * (-(angular_weight * dx * dx) / (2.0 * std::f64::consts::PI.powi(2)));
    validate_result_finite_complex("fms_response_prefactor", prefactor)?;
    let cluster_green_squared = cluster_green * cluster_green;

    let mut response = Array2::zeros((active_count, active_count).f());
    for row in 0..fms_count {
        let row_radius = radii[row];
        let regular_row = regular_solution[row];
        validate_positive("radius", row_radius)?;
        validate_finite_complex_input("regular_solution", regular_row)?;
        let regular_row_squared = regular_row * regular_row;
        for column in row..fms_count {
            let column_radius = radii[column];
            let regular_column = regular_solution[column];
            let irregular_column = irregular_solution[column];
            validate_positive("radius", column_radius)?;
            validate_finite_complex_input("regular_solution", regular_column)?;
            validate_finite_complex_input("irregular_solution", irregular_column)?;
            let cluster_term =
                2.0 * cluster_green * regular_row_squared * regular_column * irregular_column
                    + cluster_green_squared * regular_row_squared * regular_column * regular_column;
            let value = prefactor * row_radius * column_radius * cluster_term;
            validate_result_finite_complex("fms_response_slice", value)?;
            response[(row, column)] = value;
        }
    }
    Ok(response)
}

/// Build one CRPA response slice from `chi_crpa.f90`.
///
/// CRPA stores the same upper-triangle `chi0re(m,n)` workspace as SCREEN, but
/// separates the angular prefactor from the base factor and applies a
/// `sin(...)^4` radial projection to the selected constrained channel. Passing
/// `cluster_green = 0` yields the atomic part. A nonzero `cluster_green` adds
/// the diagonal FMS terms used by the CRPA driver.
pub fn screen_crpa_response_slice(
    input: ScreenCrpaResponseSliceInput<'_>,
) -> Result<ComplexMat, ScreenError> {
    let ScreenCrpaResponseSliceInput {
        radii,
        regular_solution,
        irregular_solution,
        cluster_green,
        wave_number,
        dx,
        angular_momentum,
        crpa_angular_momentum,
        projection_window,
        active_count,
    } = input;

    validate_positive("dx", dx)?;
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, regular_solution.len())?;
    validate_active_count(active_count, irregular_solution.len())?;
    validate_finite_complex_input("cluster_green", cluster_green)?;
    validate_finite_complex_input("wave_number", wave_number)?;
    let projection_window = projection_window.filter(|_| angular_momentum == crpa_angular_momentum);
    if let Some(window) = projection_window {
        validate_finite("projection_inner_radius", window.inner_radius)?;
        validate_finite("projection_outer_radius", window.outer_radius)?;
        validate_increasing(
            "projection_inner_radius",
            window.inner_radius,
            "projection_outer_radius",
            window.outer_radius,
        )?;
    }

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let doubled_wave = wave_number * 2.0;
    let prefactor =
        doubled_wave * doubled_wave * (-(dx * dx) / (2.0 * std::f64::consts::PI.powi(2)));
    validate_result_finite_complex("crpa_response_prefactor", prefactor)?;
    let cluster_green_squared = cluster_green * cluster_green;

    let mut projection_weights = Vec::with_capacity(active_count);
    for &radius in radii.iter().take(active_count) {
        validate_positive("radius", radius)?;
        let weight = match projection_window {
            Some(window) => crpa_response_projection_weight(radius, window)?,
            None => 1.0,
        };
        projection_weights.push(weight);
    }

    let mut response = Array2::zeros((active_count, active_count).f());
    for row in 0..active_count {
        let row_radius = radii[row];
        let regular_row = regular_solution[row];
        validate_finite_complex_input("regular_solution", regular_row)?;
        let row_factor = row_radius * projection_weights[row] * regular_row * regular_row;
        for column in row..active_count {
            let column_radius = radii[column];
            let regular_column = regular_solution[column];
            let irregular_column = irregular_solution[column];
            validate_finite_complex_input("regular_solution", regular_column)?;
            validate_finite_complex_input("irregular_solution", irregular_column)?;
            let response_column = irregular_column * irregular_column
                + 2.0 * cluster_green * regular_column * irregular_column
                + cluster_green_squared * regular_column * regular_column;
            let value = prefactor
                * angular_weight
                * row_factor
                * column_radius
                * projection_weights[column]
                * response_column;
            validate_result_finite_complex("crpa_response_slice", value)?;
            response[(row, column)] = value;
        }
    }
    Ok(response)
}

/// Convert an FMS diagonal scattering block into SCREEN/CRPA `gtrl(l,ie)`.
///
/// `screensub.f90` sums `gg(l^2+m,l^2+m,iph)` over the `2*l+1` magnetic
/// substates, widens the single-precision FMS result to double precision, and
/// applies the absorber phase factor `exp(2*i*ph_l)/(2*l+1)`. The CRPA
/// diagonal `gtrl(l,l,ie)` expression reduces to the same formula.
pub fn screen_fms_cluster_green_trace(
    scattering: ArrayView2<'_, Complex32>,
    phase_shift: Complex,
    angular_momentum: usize,
) -> Result<Complex, ScreenError> {
    validate_finite_complex_input("phase_shift", phase_shift)?;
    let start =
        angular_momentum
            .checked_mul(angular_momentum)
            .ok_or(ScreenError::IndexSizeOverflow {
                name: "angular_momentum",
            })?;
    let required_order = angular_momentum
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .ok_or(ScreenError::IndexSizeOverflow {
            name: "angular_momentum",
        })?;
    validate_active_matrix_shape(
        "fms_scattering",
        scattering.nrows(),
        scattering.ncols(),
        required_order,
    )?;

    let mut trace = Complex::new(0.0, 0.0);
    for state_index in start..required_order {
        let value = scattering[(state_index, state_index)];
        validate_finite_complex32_matrix("fms_scattering", state_index, state_index, value)?;
        trace += Complex::new(value.re as Real, value.im as Real);
    }

    let angular_weight = 2.0 * angular_momentum as Real + 1.0;
    let value = trace * (Complex::new(0.0, 2.0) * phase_shift).exp() / angular_weight;
    validate_result_finite_complex("fms_cluster_green_trace", value)?;
    Ok(value)
}

fn crpa_response_projection_weight(
    radius: Real,
    window: ScreenCrpaProjectionWindow,
) -> Result<Real, ScreenError> {
    let clamped = radius.max(window.inner_radius).min(window.outer_radius);
    let scaled = (clamped - window.inner_radius) / (window.outer_radius - window.inner_radius);
    let weight = (scaled * std::f64::consts::FRAC_PI_2).sin().powi(4);
    validate_result_finite("crpa_response_projection_weight", weight)?;
    Ok(weight)
}

/// Port the SCREEN/CRPA response-system matrix setup.
///
/// FEFF builds the real system matrix as `A = I - K * imag(chi0)`, then passes
/// that matrix to LAPACK `dgetrf`/`dgetrs`. The inputs are `ndarray` views so
/// callers can pass full FEFF work arrays and select the active `ilast` prefix.
/// The returned matrix uses Fortran-order storage to preserve the layout that
/// downstream FEFF-compatible linear algebra expects.
pub fn screen_response_system_matrix(
    kernel: ArrayView2<'_, Real>,
    susceptibility: ArrayView2<'_, Complex>,
    active_count: usize,
) -> Result<RealMat, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_matrix_shape("kernel", kernel.nrows(), kernel.ncols(), active_count)?;
    validate_active_matrix_shape(
        "susceptibility",
        susceptibility.nrows(),
        susceptibility.ncols(),
        active_count,
    )?;

    for row in 0..active_count {
        for column in 0..active_count {
            validate_finite_matrix("kernel", row, column, kernel[(row, column)])?;
            validate_finite_complex_matrix(
                "susceptibility",
                row,
                column,
                susceptibility[(row, column)],
            )?;
        }
    }

    let mut system = Array2::zeros((active_count, active_count).f());
    for index in 0..active_count {
        system[(index, index)] = 1.0;
    }
    for column in 0..active_count {
        for index in 0..active_count {
            let susceptibility_imaginary = susceptibility[(index, column)].im;
            if susceptibility_imaginary == 0.0 {
                continue;
            }
            for row in 0..active_count {
                system[(row, column)] -= kernel[(row, index)] * susceptibility_imaginary;
            }
        }
        for row in 0..active_count {
            validate_result_finite("response_system_matrix", system[(row, column)])?;
        }
    }
    Ok(system)
}

/// Solve FEFF's screened-core-hole response equation.
///
/// This is the matrix-inversion block shared by `SCREEN/screensub.f90` and
/// `CRPA/chi_crpa.f90`: build `A = I - K * imag(chi0)` and solve
/// `A * wscrn = v_ch` with FEFF-compatible real LU factorization. The result is
/// the screened potential vector that FEFF stores back into `wscrn`.
pub fn screen_solve_response_potential(
    kernel: ArrayView2<'_, Real>,
    susceptibility: ArrayView2<'_, Complex>,
    bare_potential: ArrayView1<'_, Real>,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_active_count(active_count, bare_potential.len())?;
    for &value in bare_potential.iter().take(active_count) {
        validate_finite("bare_potential", value)?;
    }

    let system = screen_response_system_matrix(kernel, susceptibility, active_count)?;
    let rhs = Array1::from_iter(bare_potential.iter().take(active_count).copied());
    let lu = real_lu_factor(system.view())?;
    let solution = real_lu_solve_vector(&lu, rhs.view())?;
    for &value in &solution {
        validate_result_finite("screened_response_potential", value)?;
    }
    Ok(solution)
}

fn validate_active_count(active_count: usize, len: usize) -> Result<(), ScreenError> {
    if active_count > len {
        Err(ScreenError::ActiveCountOutOfRange { active_count, len })
    } else {
        Ok(())
    }
}

fn checked_radial_add(
    name: &'static str,
    value: isize,
    increment: isize,
) -> Result<isize, ScreenError> {
    value
        .checked_add(increment)
        .ok_or(ScreenError::IndexSizeOverflow { name })
}

fn positive_radial_bound(name: &'static str, value: isize) -> Result<usize, ScreenError> {
    if value > 0 {
        Ok(value as usize)
    } else {
        Err(ScreenError::NonPositiveRadialBound { name, value })
    }
}

fn complex32_result(name: &'static str, value: Complex) -> Result<Complex32, ScreenError> {
    let single = Complex32::new(value.re as f32, value.im as f32);
    if single.re.is_finite() && single.im.is_finite() {
        Ok(single)
    } else {
        Err(ScreenError::NonFiniteComplexResult {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), ScreenError> {
    if actual < minimum {
        Err(ScreenError::CountTooSmall {
            name,
            actual,
            minimum,
        })
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteInput { name, value })
    }
}

fn validate_finite_complex_input(name: &'static str, value: Complex) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexInput {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_positive(name: &'static str, value: Real) -> Result<(), ScreenError> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreenError::NonPositiveInput { name, value })
    }
}

fn validate_result_finite_complex(name: &'static str, value: Complex) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexResult {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_increasing(
    lower_name: &'static str,
    lower: Real,
    upper_name: &'static str,
    upper: Real,
) -> Result<(), ScreenError> {
    if upper > lower {
        Ok(())
    } else {
        Err(ScreenError::NonIncreasingInput {
            lower_name,
            upper_name,
            lower,
            upper,
        })
    }
}

fn validate_result_finite(name: &'static str, value: Real) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteResult { name, value })
    }
}

fn validate_positive_result(name: &'static str, value: Real) -> Result<(), ScreenError> {
    validate_result_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreenError::NonPositiveResult { name, value })
    }
}

fn validate_active_matrix_shape(
    name: &'static str,
    rows: usize,
    columns: usize,
    active_count: usize,
) -> Result<(), ScreenError> {
    if rows < active_count || columns < active_count {
        Err(ScreenError::MatrixTooSmall {
            name,
            rows,
            columns,
            active_count,
        })
    } else {
        Ok(())
    }
}

fn validate_finite_matrix(
    name: &'static str,
    row: usize,
    column: usize,
    value: Real,
) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteMatrixInput {
            name,
            row,
            column,
            value,
        })
    }
}

fn validate_finite_complex_matrix(
    name: &'static str,
    row: usize,
    column: usize,
    value: Complex,
) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexMatrixInput {
            name,
            row,
            column,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_finite_complex32_matrix(
    name: &'static str,
    row: usize,
    column: usize,
    value: Complex32,
) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexMatrixInput {
            name,
            row,
            column,
            real: value.re as Real,
            imaginary: value.im as Real,
        })
    }
}

#[cfg(test)]
mod tests;
