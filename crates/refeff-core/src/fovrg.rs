//! FEFF FOVRG numerical helpers.
//!
//! These routines cover small pieces of the relativistic radial solver that can
//! be validated independently of the full `dfovrg` integration path.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use thiserror::Error;

use crate::{
    Complex, ComplexVec, Real, RealMat, RealVec,
    angular::{AngularError, wigner_3j},
    bessel::{BesselError, besjh, besjn},
};

// `diff.f90` uses unsuffixed Fortran real literals in these stencils. Preserve
// their default-real rounding before widening to the Rust `Real` type.
const F77_REAL_HALF: Real = 0.5_f32 as Real;
const F77_REAL_ONE_POINT_TWO: Real = 1.2_f32 as Real;
const F77_REAL_ONE_POINT_FIVE: Real = 1.5_f32 as Real;
const F77_REAL_TWO: Real = 2.0_f32 as Real;
const F77_REAL_TWO_POINT_FOUR_FIVE: Real = 2.45_f32 as Real;
const F77_REAL_THREE_POINT_THREE: Real = 3.3_f32 as Real;
const F77_REAL_THREE_POINT_SEVEN_FIVE: Real = 3.75_f32 as Real;
const F77_REAL_FOUR_POINT_TWO: Real = 4.2_f32 as Real;
const F77_REAL_SIX: Real = 6.0_f32 as Real;
const F77_REAL_SIX_AND_TWO_THIRDS: Real = 6.666_666_5_f32 as Real;
const F77_REAL_SEVEN_POINT_FIVE: Real = 7.5_f32 as Real;
const F77_REAL_SEVEN_POINT_EIGHT: Real = 7.8_f32 as Real;
const F77_REAL_EIGHT: Real = 8.0_f32 as Real;
const F77_REAL_TWELVE: Real = 12.0_f32 as Real;
const F77_REAL_ONE_SIXTH: Real = 0.166_666_67_f32 as Real;
const F77_REAL_FOURTEEN_OVER_FORTY_FIVE: Real = (14.0_f32 as Real) / (45.0_f32 as Real);
const F77_REAL_TWENTY_FOUR_OVER_FORTY_FIVE: Real = (24.0_f32 as Real) / (45.0_f32 as Real);
const F77_REAL_SIXTY_FOUR_OVER_FORTY_FIVE: Real = (64.0_f32 as Real) / (45.0_f32 as Real);
const FOVRG_INT_OUT_HISTORY: usize = 6;
const FOVRG_INT_OUT_TEST: Real = 1.0e5;
const FOVRG_ANGULAR_COEFFICIENT_SLOTS: usize = 5;
const FEFF_ALPHA_INVERSE: Real = 137.03598956;
const FEFF_FINE_STRUCTURE_ALPHA: Real = 1.0 / FEFF_ALPHA_INVERSE;

/// Inputs for FEFF `FOVRG/diff.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgC3DerivativeInput<'a> {
    /// Complex potential values `v`.
    pub potential: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Relativistic kappa `kap`.
    pub kappa: i32,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic grid step `dx`.
    pub delta: Real,
    /// Number of active radial rows `n`.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/yzktec.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgYkZkTransformInput<'a> {
    /// Tabulated source function `f`.
    pub source: ArrayView1<'a, Complex>,
    /// Origin development coefficients `af` for [`FovrgYkZkTransformInput::source`].
    pub source_coefficients: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Initial origin power `ap`; FEFF uses only its real part before overwriting it.
    pub initial_power: Complex,
    /// Logarithmic radial step `h`.
    pub step: Real,
    /// Multipole order `k`.
    pub angular_momentum: usize,
    /// Number of active origin coefficients `nd`.
    pub coefficient_count: usize,
    /// Number of active source samples `np`; FEFF clamps this to `idim - 1`.
    pub source_len: usize,
    /// Active radial capacity `idim`.
    pub active_len: usize,
    /// Optional tail correction `dyzk`.
    pub tail_correction: Complex,
}

/// Inputs for FEFF `FOVRG/yzkrdc.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgYkZkExchangeInput<'a> {
    /// Bound-orbital large radial component `cg(:, i)`.
    pub large_component: ArrayView1<'a, Real>,
    /// Bound-orbital small radial component `cp(:, i)`.
    pub small_component: ArrayView1<'a, Real>,
    /// Bound-orbital large origin coefficients `bg(:, i)`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Bound-orbital small origin coefficients `bp(:, i)`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Partner large radial component `ps`.
    pub partner_large_component: ArrayView1<'a, Complex>,
    /// Partner small radial component `qs`.
    pub partner_small_component: ArrayView1<'a, Complex>,
    /// Partner large origin coefficients `aps`.
    pub partner_large_coefficients: ArrayView1<'a, Complex>,
    /// Partner small origin coefficients `aqs`.
    pub partner_small_coefficients: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Bound-orbital origin power `fl(i)`.
    pub orbital_power: Real,
    /// Partner origin power `flps`.
    pub partner_power: Real,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// Multipole order `k`.
    pub angular_momentum: usize,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// Bound-orbital maximum tabulated row `nmax(i)`.
    pub orbital_len: usize,
    /// Global active source row count `np`.
    pub source_len: usize,
    /// Active radial capacity `idim`.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/dsordc.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgOverlapIntegralInput<'a> {
    /// Radial large-component integrand `dg`.
    pub large_integrand: ArrayView1<'a, Complex>,
    /// Radial small-component integrand `dp`.
    pub small_integrand: ArrayView1<'a, Complex>,
    /// Origin coefficients `ag` for [`FovrgOverlapIntegralInput::large_integrand`].
    pub large_integrand_coefficients: ArrayView1<'a, Complex>,
    /// Origin coefficients `ap` for [`FovrgOverlapIntegralInput::small_integrand`].
    pub small_integrand_coefficients: ArrayView1<'a, Complex>,
    /// Bound-orbital large radial component `cg(:, j)`.
    pub large_component: ArrayView1<'a, Real>,
    /// Bound-orbital small radial component `cp(:, j)`.
    pub small_component: ArrayView1<'a, Real>,
    /// Bound-orbital large origin coefficients `bg(:, j)`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Bound-orbital small origin coefficients `bp(:, j)`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Origin power `a` of the incoming integrand.
    pub integrand_power: Real,
    /// Bound-orbital origin power `fl(j)`.
    pub orbital_power: Real,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// Number of active radial rows `idim`.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/ortdac.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgOrthogonalizationInput<'a> {
    /// Target large radial component `ps`.
    pub target_large_component: ArrayView1<'a, Complex>,
    /// Target small radial component `qs`.
    pub target_small_component: ArrayView1<'a, Complex>,
    /// Target large origin coefficients `aps`.
    pub target_large_coefficients: ArrayView1<'a, Complex>,
    /// Target small origin coefficients `aqs`.
    pub target_small_coefficients: ArrayView1<'a, Complex>,
    /// Bound-orbital large radial components `cg(row, orbital)`.
    pub bound_large_components: ArrayView2<'a, Real>,
    /// Bound-orbital small radial components `cp(row, orbital)`.
    pub bound_small_components: ArrayView2<'a, Real>,
    /// Bound-orbital large origin coefficients `bg(coefficient, orbital)`.
    pub bound_large_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital small origin coefficients `bp(coefficient, orbital)`.
    pub bound_small_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital occupations `xnel`.
    pub electron_counts: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values `kap`.
    pub kappa: ArrayView1<'a, i32>,
    /// Bound-orbital origin powers `fl`.
    pub orbital_powers: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Target origin power `fl(norb)`.
    pub target_power: Real,
    /// Target relativistic kappa `ikap`.
    pub target_kappa: i32,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// Number of active radial rows `idim`.
    pub active_len: usize,
    /// Number of bound orbitals, equivalent to FEFF `norb - 1`.
    pub bound_orbital_count: usize,
}

/// Inputs for FEFF `FOVRG/muatcc.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgAngularCoefficientsInput<'a> {
    /// Bound-orbital occupations `xnel`.
    pub electron_counts: ArrayView1<'a, Real>,
    /// Valence occupations `xnval`; positive rows are skipped like FEFF.
    pub valence_counts: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values `kap`.
    pub kappa: ArrayView1<'a, i32>,
    /// Target photoelectron relativistic kappa `ikap`.
    pub target_kappa: i32,
    /// Number of bound orbitals, equivalent to FEFF `norb - 1`.
    pub bound_orbital_count: usize,
}

/// Inputs for FEFF `FOVRG/dfovrg.f90` `flatv`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgFlatPotentialInput {
    /// Initial radius `r1`.
    pub start_radius: Real,
    /// Target radius `r2`.
    pub end_radius: Real,
    /// Initial large radial component `p1`.
    pub large_component: Complex,
    /// Initial small radial component `q1`.
    pub small_component: Complex,
    /// Electron energy `en`.
    pub energy: Complex,
    /// Average flat potential `vav`.
    pub average_potential: Complex,
    /// Relativistic kappa `ikap`.
    pub kappa: i32,
}

/// Inputs for FEFF `FOVRG/intout.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgOutwardIntegrationInput<'a> {
    /// Initial large radial component `gg(i0)`.
    pub initial_large_component: Complex,
    /// Initial small radial component `gp(i0)`.
    pub initial_small_component: Complex,
    /// One-electron energy `en`.
    pub energy: Complex,
    /// Direct potential `dv`.
    pub potential: ArrayView1<'a, Complex>,
    /// Direct-potential origin coefficients `av`.
    pub potential_coefficients: ArrayView1<'a, Complex>,
    /// Large-component exchange potential `eg`.
    pub large_exchange: ArrayView1<'a, Complex>,
    /// Small-component exchange potential `ep`.
    pub small_exchange: ArrayView1<'a, Complex>,
    /// C3 correction potential `vm`.
    pub c3_potential: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic grid step `hx`.
    pub step: Real,
    /// Relativistic kappa `kap`.
    pub kappa: i32,
    /// FEFF `ic3` switch or scale for the C3 term.
    pub c3_scale: i32,
    /// Zero-based equivalent of FEFF `i0`.
    pub start_index: usize,
    /// Zero-based equivalent of FEFF `max0`.
    pub last_index: usize,
    /// FEFF `np`, the output rows retained before zero fill.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/solout.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgOutgoingSolutionInput<'a> {
    /// Initial large origin coefficient `agi`.
    pub initial_large_coefficient: Complex,
    /// Initial small origin coefficient `api`; FEFF may replace it for point nuclei.
    pub initial_small_coefficient: Complex,
    /// One-electron energy `en`.
    pub energy: Complex,
    /// First origin power `fl`.
    pub origin_power: Real,
    /// Relativistic kappa `kap`.
    pub kappa: i32,
    /// FEFF `rmt`; retained for call-shape compatibility with `solout`.
    pub muffin_tin_radius: Real,
    /// Direct potential `dv`.
    pub potential: ArrayView1<'a, Complex>,
    /// Direct-potential origin coefficients `av`.
    pub potential_coefficients: ArrayView1<'a, Complex>,
    /// Large-component exchange potential `eg`.
    pub large_exchange: ArrayView1<'a, Complex>,
    /// Small-component exchange potential `ep`.
    pub small_exchange: ArrayView1<'a, Complex>,
    /// Large-component exchange origin coefficients `ceg`.
    pub large_exchange_coefficients: ArrayView1<'a, Complex>,
    /// Small-component exchange origin coefficients `cep`.
    pub small_exchange_coefficients: ArrayView1<'a, Complex>,
    /// C3 correction potential `vm`.
    pub c3_potential: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic grid step `hx`.
    pub step: Real,
    /// FEFF `ic3` switch or scale for the C3 term.
    pub c3_scale: i32,
    /// Zero-based equivalent of FEFF `jri`.
    pub radial_match_index: usize,
    /// Zero-based equivalent of FEFF `max0`.
    pub last_index: usize,
    /// Zero-based equivalent of FEFF `iwkb`.
    pub wkb_index: usize,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// FEFF `np`, the output rows retained before zero fill.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/solin.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgInwardSolutionInput<'a> {
    /// Initial large origin coefficient `agi`; retained for `solin` call-shape compatibility.
    pub initial_large_coefficient: Complex,
    /// Initial small origin coefficient `api`; retained for `solin` call-shape compatibility.
    pub initial_small_coefficient: Complex,
    /// One-electron energy `en`.
    pub energy: Complex,
    /// First origin power `fl`.
    pub origin_power: Real,
    /// Relativistic kappa `kap`.
    pub kappa: i32,
    /// FEFF `rmt`; retained for call-shape compatibility with `solin`.
    pub muffin_tin_radius: Real,
    /// Direct potential `dv`.
    pub potential: ArrayView1<'a, Complex>,
    /// Large-component exchange potential `eg`.
    pub large_exchange: ArrayView1<'a, Complex>,
    /// Small-component exchange potential `ep`.
    pub small_exchange: ArrayView1<'a, Complex>,
    /// C3 correction potential `vm`.
    pub c3_potential: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic grid step `hx`.
    pub step: Real,
    /// FEFF `ic3` switch or scale for the C3 term.
    pub c3_scale: i32,
    /// Zero-based equivalent of FEFF `jri`.
    pub radial_match_index: usize,
    /// Zero-based equivalent of FEFF `imax`.
    pub last_index: usize,
    /// Zero-based equivalent of FEFF `iwkb`.
    pub wkb_index: usize,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// FEFF `np`, the output rows retained before zero fill.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/potex.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgExchangePotentialInput<'a> {
    /// Target large radial component `ps`.
    pub target_large_component: ArrayView1<'a, Complex>,
    /// Target small radial component `qs`.
    pub target_small_component: ArrayView1<'a, Complex>,
    /// Target large origin coefficients `aps`.
    pub target_large_coefficients: ArrayView1<'a, Complex>,
    /// Target small origin coefficients `aqs`.
    pub target_small_coefficients: ArrayView1<'a, Complex>,
    /// Bound-orbital large radial components `cg(row, orbital)`.
    pub bound_large_components: ArrayView2<'a, Real>,
    /// Bound-orbital small radial components `cp(row, orbital)`.
    pub bound_small_components: ArrayView2<'a, Real>,
    /// Bound-orbital large origin coefficients `bg(coefficient, orbital)`.
    pub bound_large_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital small origin coefficients `bp(coefficient, orbital)`.
    pub bound_small_coefficients: ArrayView2<'a, Real>,
    /// FEFF angular coefficients `afgkc(kap(target), orbital, index)`.
    pub angular_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital origin powers `fl`.
    pub orbital_powers: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values `kap`.
    pub kappa: ArrayView1<'a, i32>,
    /// Bound-orbital maximum tabulated rows `nmax`.
    pub orbital_lengths: ArrayView1<'a, usize>,
    /// Bound-orbital normalization factors `fix`.
    pub normalization: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Target origin power `fl(norb)`.
    pub target_power: Real,
    /// Target relativistic kappa `kap(norb)`.
    pub target_kappa: i32,
    /// Target normalization factor `fix(norb)`.
    pub target_normalization: Real,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// FEFF `np`, the source grid limit passed through `yzkrdc`.
    pub source_len: usize,
    /// Number of active radial rows `idim`.
    pub active_len: usize,
    /// FEFF `jri`, rows retained in the output potentials.
    pub radial_output_count: usize,
    /// Number of bound orbitals, equivalent to FEFF `norb - 1`.
    pub bound_orbital_count: usize,
}

/// Inputs for FEFF `FOVRG/potdvp.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgPotentialDevelopmentInput<'a> {
    /// Nuclear potential development coefficients `anoy`.
    pub nuclear_coefficients: ArrayView1<'a, Real>,
    /// Bound-orbital large-component coefficients `bg(coefficient, orbital)`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital small-component coefficients `bp(coefficient, orbital)`.
    pub small_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital occupations `xnel`.
    pub electron_counts: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values `kap`.
    pub kappa: ArrayView1<'a, i32>,
    /// FEFF normalization factors `fix`.
    pub normalization: ArrayView1<'a, Real>,
    /// Radial grid `dr`; only the first point enters this kernel.
    pub radii: ArrayView1<'a, Real>,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// FEFF `norb`; bound orbitals `1..norb-1` contribute.
    pub orbital_count: usize,
}

/// Inputs for FEFF `FOVRG/nucdec.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgNuclearPotentialInput {
    /// Nuclear charge `dz`.
    pub nuclear_charge: Real,
    /// Logarithmic radial step `hx`.
    pub step: Real,
    /// FEFF input/output `dr1`, the first tabulation radius multiplied by `dz`.
    pub first_radius_times_charge: Real,
    /// Number of radial tabulation points `np`.
    pub radial_count: usize,
    /// Number of origin development coefficients `ndor`.
    pub coefficient_count: usize,
}

/// Output from FEFF `FOVRG/yzktec.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgYkZkTransform {
    /// Transformed `yk` values, zero-filled after [`FovrgYkZkTransform::computed_len`].
    pub yk: ComplexVec,
    /// Intermediate `zk` values, zero-filled after [`FovrgYkZkTransform::computed_len`].
    pub zk: ComplexVec,
    /// Mutated `af` development coefficients for `yk`.
    pub yk_coefficients: ComplexVec,
    /// Development coefficients `ag` for `zk`.
    pub zk_coefficients: ComplexVec,
    /// FEFF output scalar `ap`, the leading origin constant for `yk`.
    pub origin_constant: Complex,
    /// Number of meaningful radial rows, equivalent to clamped `np + 1`.
    pub computed_len: usize,
}

/// Output from FEFF `FOVRG/nucdec.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgNuclearPotential {
    /// Origin development coefficients `av`.
    pub development_coefficients: RealVec,
    /// Radial grid `dr`.
    pub radii: RealVec,
    /// Nuclear potential `dv`.
    pub potential: RealVec,
    /// FEFF 1-based nuclear-radius index `nuc`.
    pub nucleus_index: usize,
    /// FEFF output `dr1`.
    pub first_radius_times_charge: Real,
}

/// Output from FEFF `FOVRG/ortdac.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgOrthogonalization {
    /// Orthogonalized target large radial component `ps`.
    pub large_component: ComplexVec,
    /// Orthogonalized target small radial component `qs`.
    pub small_component: ComplexVec,
    /// Orthogonalized target large origin coefficients `aps`.
    pub large_coefficients: ComplexVec,
    /// Orthogonalized target small origin coefficients `aqs`.
    pub small_coefficients: ComplexVec,
    /// Per-bound-orbital overlap coefficients used for subtraction.
    pub overlaps: ComplexVec,
}

/// Output from FEFF `FOVRG/potex.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgExchangePotential {
    /// Large-component exchange potential `eg`.
    pub large_potential: ComplexVec,
    /// Small-component exchange potential `ep`.
    pub small_potential: ComplexVec,
    /// Large-component origin coefficients `ceg`.
    pub large_coefficients: ComplexVec,
    /// Small-component origin coefficients `cep`.
    pub small_coefficients: ComplexVec,
}

/// Output from FEFF `FOVRG/potdvp.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgPotentialDevelopment {
    /// Potential development coefficients `av` after FEFF's division by `cl`.
    pub potential_coefficients: ComplexVec,
    /// Transformed density coefficients `ag` before division by `cl`.
    pub density_coefficients: RealVec,
    /// FEFF output `ap(1)` before division by `cl`.
    pub origin_correction: Real,
}

/// Output from FEFF `FOVRG/dfovrg.f90` `flatv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FovrgFlatPotentialPropagation {
    /// Propagated large radial component `p2`.
    pub large_component: Complex,
    /// Propagated small radial component `q2`.
    pub small_component: Complex,
}

/// Output from FEFF `FOVRG/intout.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgOutwardIntegration {
    /// Outward large radial component `gg`.
    pub large_component: ComplexVec,
    /// Outward small radial component `gp`.
    pub small_component: ComplexVec,
    /// Count of Milne corrector rows that did not converge within FEFF's retry limit.
    pub difficult_iterations: usize,
}

/// Output from FEFF `FOVRG/solout.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgOutgoingSolution {
    /// Regular outgoing large radial component `gg`.
    pub large_component: ComplexVec,
    /// Regular outgoing small radial component `gp`.
    pub small_component: ComplexVec,
    /// Updated large origin coefficients `ag`.
    pub large_coefficients: ComplexVec,
    /// Updated small origin coefficients `ap`.
    pub small_coefficients: ComplexVec,
    /// Count of difficult Milne iterations reported by the inner `intout` step.
    pub difficult_iterations: usize,
}

/// Output from FEFF `FOVRG/solin.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgInwardSolution {
    /// Irregular large radial component `gg`.
    pub large_component: ComplexVec,
    /// Irregular small radial component `gp`.
    pub small_component: ComplexVec,
    /// Updated large origin coefficients `ag`.
    pub large_coefficients: ComplexVec,
    /// Updated small origin coefficients `ap`.
    pub small_coefficients: ComplexVec,
    /// Count of difficult Milne iterations reported by the inward integration step.
    pub difficult_iterations: usize,
}

/// Error returned by FOVRG helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FovrgError {
    /// FEFF `diff` uses rows 1..=8 in the first two one-sided stencils.
    #[error("FOVRG {name} count {actual} is below minimum {minimum}")]
    CountTooSmall {
        name: &'static str,
        actual: usize,
        minimum: usize,
    },
    /// Active rows must fit in every input array.
    #[error("FOVRG active row count {active_len} exceeds {field} length {len}")]
    ActiveCountOutOfRange {
        field: &'static str,
        active_len: usize,
        len: usize,
    },
    /// Counts that are later converted to FEFF integer exponents must fit.
    #[error("FOVRG {name} count {actual} exceeds maximum {maximum}")]
    CountTooLarge {
        name: &'static str,
        actual: usize,
        maximum: usize,
    },
    /// Simpson integration in FEFF `dsordc` advances by two rows.
    #[error("FOVRG {name} count {actual} must be odd")]
    CountMustBeOdd { name: &'static str, actual: usize },
    /// Index ranges must be ordered.
    #[error("FOVRG {name} range start {start} exceeds end {end}")]
    InvalidRange {
        name: &'static str,
        start: usize,
        end: usize,
    },
    /// Scalar inputs must be finite.
    #[error("FOVRG {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Positive scalar inputs must be finite and greater than zero.
    #[error("FOVRG {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    /// Divisor-like scalar inputs must be nonzero.
    #[error("FOVRG {name} must be nonzero")]
    ZeroInput { name: &'static str },
    /// FEFF formulas with a zero denominator are reported instead of evaluated.
    #[error("FOVRG denominator {name} is zero")]
    ZeroDenominator { name: &'static str },
    /// Radii must be positive.
    #[error("FOVRG radius row {row} must be positive, got {value}")]
    NonPositiveRadius { row: usize, value: Real },
    /// Kappa values are nonzero quantum numbers in FEFF radial kernels.
    #[error("FOVRG {name} row {row} has invalid quantum number {value}")]
    InvalidQuantumNumber {
        name: &'static str,
        row: usize,
        value: i32,
    },
    /// Complex inputs must be finite.
    #[error("FOVRG {name} row {row} must be finite, got {value}")]
    NonFiniteComplexInput {
        name: &'static str,
        row: usize,
        value: Complex,
    },
    /// Real vector inputs must be finite.
    #[error("FOVRG {name} row {row} must be finite, got {value}")]
    NonFiniteRealInput {
        name: &'static str,
        row: usize,
        value: Real,
    },
    /// Complex potential values must be finite.
    #[error("FOVRG potential row {row} must be finite, got {value}")]
    NonFinitePotential { row: usize, value: Complex },
    /// Output values must remain finite.
    #[error("FOVRG derivative row {row} must be finite, got {value}")]
    NonFiniteResult { row: usize, value: Complex },
    /// Wigner 3j construction failed while building FEFF `muatcc` coefficients.
    #[error("FOVRG angular coefficient construction failed: {source}")]
    AngularCoefficient { source: AngularError },
    /// Spherical Bessel construction failed while evaluating FEFF `flatv`.
    #[error("FOVRG flat-potential propagation failed: {source}")]
    FlatPotentialBessel { source: BesselError },
}

/// Port of `FOVRG/diff.f90`: C3 radial derivative term.
///
/// FEFF first differentiates `v(r) * r^2` with one-sided boundary stencils and
/// a centered fourth-order interior stencil, then returns
/// `(d(v*r^2)/dx - 2*v*r^2) / r * (kap+1) / cl`.
pub fn fovrg_c3_derivative(input: FovrgC3DerivativeInput<'_>) -> Result<ComplexVec, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 8)?;
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_nonzero_finite("delta", input.delta)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;

    for row in 0..input.active_len {
        validate_radius(row, input.radii[row])?;
        validate_potential(row, input.potential[row])?;
    }

    let vt = Array1::from_iter(
        (0..input.active_len).map(|row| input.potential[row] * input.radii[row].powi(2)),
    );
    let mut derivative = Array1::<Complex>::zeros(input.active_len);

    derivative[0] = ((F77_REAL_SIX * vt[1]
        + F77_REAL_SIX_AND_TWO_THIRDS * vt[3]
        + F77_REAL_ONE_POINT_TWO * vt[5])
        - (F77_REAL_TWO_POINT_FOUR_FIVE * vt[0]
            + F77_REAL_SEVEN_POINT_FIVE * vt[2]
            + F77_REAL_THREE_POINT_SEVEN_FIVE * vt[4]
            + F77_REAL_ONE_SIXTH * vt[6]))
        / input.delta;
    derivative[1] = ((F77_REAL_SIX * vt[2]
        + F77_REAL_SIX_AND_TWO_THIRDS * vt[4]
        + F77_REAL_ONE_POINT_TWO * vt[6])
        - (F77_REAL_TWO_POINT_FOUR_FIVE * vt[1]
            + F77_REAL_SEVEN_POINT_FIVE * vt[3]
            + F77_REAL_THREE_POINT_SEVEN_FIVE * vt[5]
            + F77_REAL_ONE_SIXTH * vt[7]))
        / input.delta;

    for row in 2..input.active_len - 2 {
        derivative[row] = ((vt[row - 2] + F77_REAL_EIGHT * vt[row + 1])
            - (F77_REAL_EIGHT * vt[row - 1] + vt[row + 2]))
            / F77_REAL_TWELVE
            / input.delta;
    }

    let last = input.active_len - 1;
    derivative[last - 1] = (vt[last] - vt[last - 2]) / (F77_REAL_TWO * input.delta);
    derivative[last] = (F77_REAL_HALF * vt[last - 2] - F77_REAL_TWO * vt[last - 1]
        + F77_REAL_ONE_POINT_FIVE * vt[last])
        / input.delta;

    let scale = ((input.kappa as f32 + 1.0_f32) as Real) / input.speed_of_light;
    let mut output = Array1::<Complex>::zeros(input.active_len);
    for row in 0..input.active_len {
        let value = (derivative[row] - F77_REAL_TWO * vt[row]) / input.radii[row] * scale;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(FovrgError::NonFiniteResult { row, value });
        }
        output[row] = value;
    }
    Ok(output)
}

/// Port of `FOVRG/aprdep.f90`: real polynomial product coefficient.
///
/// Returns the coefficient for power `coefficient_count - 1` in the product of
/// two real origin-development polynomials.
pub fn fovrg_real_product_coefficient(
    left_coefficients: ArrayView1<'_, Real>,
    right_coefficients: ArrayView1<'_, Real>,
    coefficient_count: usize,
) -> Result<Real, FovrgError> {
    validate_count_at_least("coefficient_count", coefficient_count, 1)?;
    validate_active_len(
        "left_coefficients",
        coefficient_count,
        left_coefficients.len(),
    )?;
    validate_active_len(
        "right_coefficients",
        coefficient_count,
        right_coefficients.len(),
    )?;
    for coefficient in 0..coefficient_count {
        validate_real_input(
            "left_coefficients",
            coefficient,
            left_coefficients[coefficient],
        )?;
        validate_real_input(
            "right_coefficients",
            coefficient,
            right_coefficients[coefficient],
        )?;
    }

    let coefficient =
        real_product_coefficient(left_coefficients, right_coefficients, coefficient_count);
    validate_real_result("real_product_coefficient", 0, coefficient)?;
    Ok(coefficient)
}

/// Port of `FOVRG/aprdec.f90`: complex-real polynomial product coefficient.
///
/// Returns the coefficient for power `coefficient_count - 1` in the product of
/// a complex origin-development polynomial and a real one.
pub fn fovrg_complex_real_product_coefficient(
    complex_coefficients: ArrayView1<'_, Complex>,
    real_coefficients: ArrayView1<'_, Real>,
    coefficient_count: usize,
) -> Result<Complex, FovrgError> {
    validate_count_at_least("coefficient_count", coefficient_count, 1)?;
    validate_active_len(
        "complex_coefficients",
        coefficient_count,
        complex_coefficients.len(),
    )?;
    validate_active_len(
        "real_coefficients",
        coefficient_count,
        real_coefficients.len(),
    )?;
    for coefficient in 0..coefficient_count {
        validate_complex_input(
            "complex_coefficients",
            coefficient,
            complex_coefficients[coefficient],
        )?;
        validate_real_input(
            "real_coefficients",
            coefficient,
            real_coefficients[coefficient],
        )?;
    }

    let coefficient = complex_real_product_coefficient(
        complex_coefficients,
        real_coefficients,
        coefficient_count,
    );
    validate_complex_result("complex_real_product_coefficient", 0, coefficient)?;
    Ok(coefficient)
}

/// Port of `FOVRG/yzktec.f90`: build the radial `yk` and `zk` exchange kernels.
///
/// FEFF evaluates
/// `zk(r) = r^-k * integral(0..r, f(u) * u^k du)` and then
/// `yk(r) = zk(r) + r^(k+1) * integral(r..infinity, f(u) * u^(-k-1) du)`.
/// The first integration runs outward on the logarithmic radial mesh and the
/// second runs backward from FEFF's clamped `np + 1` endpoint.
pub fn fovrg_yk_zk_transform(
    input: FovrgYkZkTransformInput<'_>,
) -> Result<FovrgYkZkTransform, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len("source", input.active_len, input.source.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "source_coefficients",
        input.coefficient_count,
        input.source_coefficients.len(),
    )?;
    validate_positive_finite("step", input.step)?;
    validate_complex_input("initial_power", 0, input.initial_power)?;
    validate_complex_input("tail_correction", 0, input.tail_correction)?;
    if input.angular_momentum > i32::MAX as usize - 1 {
        return Err(FovrgError::CountTooLarge {
            name: "angular_momentum",
            actual: input.angular_momentum,
            maximum: i32::MAX as usize - 1,
        });
    }

    let source_len = input.source_len.min(input.active_len - 1);
    let computed_len = source_len + 1;
    for row in 0..source_len {
        validate_complex_input("source", row, input.source[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "source_coefficients",
            coefficient,
            input.source_coefficients[coefficient],
        )?;
    }

    let k = input.angular_momentum;
    let k_real = k as Real;
    let k_i32 = k as i32;
    let k_plus_one_i32 = (k + 1) as i32;
    let singular_tolerance = 1.0e-5_f32 as Real;
    let mut yk = Array1::<Complex>::zeros(input.active_len);
    let mut zk = Array1::<Complex>::zeros(input.active_len);
    let mut yk_coefficients = Array1::<Complex>::from_iter(
        (0..input.coefficient_count).map(|row| input.source_coefficients[row]),
    );
    let mut zk_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    for row in 0..source_len {
        yk[row] = input.source[row];
    }

    let mut power = input.initial_power.re;
    let mut origin_constant = Complex::new(0.0, 0.0);
    for coefficient in 0..input.coefficient_count {
        power += 1.0;
        let zk_denominator = power + k_real;
        validate_nonzero_denominator("yk_zk_origin_zk", zk_denominator)?;
        zk_coefficients[coefficient] = yk_coefficients[coefficient] / zk_denominator;

        if yk_coefficients[coefficient] != Complex::new(0.0, 0.0) {
            let radius_power = input.radii[0].powf(power);
            zk[0] += zk_coefficients[coefficient] * radius_power;

            let yk_denominator = power - k_real - 1.0;
            if yk_denominator.abs() <= singular_tolerance {
                yk_coefficients[coefficient] = Complex::new(0.0, 0.0);
                power -= 1.0;
            } else {
                yk_coefficients[coefficient] =
                    ((k + k + 1) as Real) * zk_coefficients[coefficient] / yk_denominator;
            }
            origin_constant += yk_coefficients[coefficient] * radius_power;
        }
    }

    for row in 0..source_len {
        yk[row] *= input.radii[row];
    }

    let hk = input.step * k_real;
    let e = (-input.step).exp();
    let ehk = e.powi(k_i32);
    let b1 = if k == 0 {
        input.step / 2.0
    } else {
        (ehk - 1.0 + hk) / (hk * k_real)
    };
    let b0 = input.step - (1.0 + hk) * b1;
    for row in 0..source_len {
        zk[row + 1] = zk[row] * ehk + b0 * yk[row] + b1 * yk[row + 1];
    }

    yk[source_len] = zk[source_len] + input.tail_correction;
    let backward_ehk = ehk * e;
    let backward_hk = hk + input.step;
    let backward_order = (k + k + 1) as Real;
    let backward_b1 =
        backward_order * (backward_ehk - 1.0 + backward_hk) / (backward_hk * (k_real + 1.0));
    let backward_b0 = backward_order * input.step - (1.0 + backward_hk) * backward_b1;
    for row in (0..source_len).rev() {
        yk[row] = yk[row + 1] * backward_ehk + backward_b0 * zk[row + 1] + backward_b1 * zk[row];
    }

    origin_constant = (origin_constant + yk[0]) / input.radii[0].powi(k_plus_one_i32);
    validate_complex_result("origin_constant", 0, origin_constant)?;
    for row in 0..computed_len {
        validate_complex_result("yk", row, yk[row])?;
        validate_complex_result("zk", row, zk[row])?;
    }
    for row in 0..input.coefficient_count {
        validate_complex_result("yk_coefficients", row, yk_coefficients[row])?;
        validate_complex_result("zk_coefficients", row, zk_coefficients[row])?;
    }

    Ok(FovrgYkZkTransform {
        yk,
        zk,
        yk_coefficients,
        zk_coefficients,
        origin_constant,
        computed_len,
    })
}

/// Port of `FOVRG/yzkrdc.f90`: construct exchange source terms and `yk/zk`.
///
/// FEFF forms `f = cg_i * ps + cp_i * qs`, builds origin coefficients from the
/// products of the large/small development polynomials, and then delegates the
/// radial integrations to `yzktec`.
pub fn fovrg_yk_zk_exchange(
    input: FovrgYkZkExchangeInput<'_>,
) -> Result<FovrgYkZkTransform, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("orbital_len", input.orbital_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len(
        "large_component",
        input.active_len,
        input.large_component.len(),
    )?;
    validate_active_len(
        "small_component",
        input.active_len,
        input.small_component.len(),
    )?;
    validate_active_len(
        "partner_large_component",
        input.active_len,
        input.partner_large_component.len(),
    )?;
    validate_active_len(
        "partner_small_component",
        input.active_len,
        input.partner_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_active_len(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_active_len(
        "partner_large_coefficients",
        input.coefficient_count,
        input.partner_large_coefficients.len(),
    )?;
    validate_active_len(
        "partner_small_coefficients",
        input.coefficient_count,
        input.partner_small_coefficients.len(),
    )?;
    validate_positive_finite("step", input.step)?;
    validate_finite("orbital_power", input.orbital_power)?;
    validate_finite("partner_power", input.partner_power)?;

    let source_len = input
        .orbital_len
        .min(input.source_len)
        .min(input.active_len - 1);
    for row in 0..source_len {
        validate_real_input("large_component", row, input.large_component[row])?;
        validate_real_input("small_component", row, input.small_component[row])?;
        validate_complex_input(
            "partner_large_component",
            row,
            input.partner_large_component[row],
        )?;
        validate_complex_input(
            "partner_small_component",
            row,
            input.partner_small_component[row],
        )?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_real_input(
            "large_coefficients",
            coefficient,
            input.large_coefficients[coefficient],
        )?;
        validate_real_input(
            "small_coefficients",
            coefficient,
            input.small_coefficients[coefficient],
        )?;
        validate_complex_input(
            "partner_large_coefficients",
            coefficient,
            input.partner_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "partner_small_coefficients",
            coefficient,
            input.partner_small_coefficients[coefficient],
        )?;
    }

    let source = Array1::from_iter((0..input.active_len).map(|row| {
        if row < source_len {
            input.large_component[row] * input.partner_large_component[row]
                + input.small_component[row] * input.partner_small_component[row]
        } else {
            Complex::new(0.0, 0.0)
        }
    }));
    let source_coefficients = Array1::from_iter((1..=input.coefficient_count).map(|count| {
        complex_real_product_coefficient(
            input.partner_large_coefficients,
            input.large_coefficients,
            count,
        ) + complex_real_product_coefficient(
            input.partner_small_coefficients,
            input.small_coefficients,
            count,
        )
    }));

    fovrg_yk_zk_transform(FovrgYkZkTransformInput {
        source: source.view(),
        source_coefficients: source_coefficients.view(),
        radii: input.radii,
        initial_power: Complex::new(input.orbital_power + input.partner_power, 0.0),
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count: input.coefficient_count,
        source_len,
        active_len: input.active_len,
        tail_correction: Complex::new(0.0, 0.0),
    })
}

/// Port of `FOVRG/dsordc.f90`: complex radial overlap integral.
///
/// FEFF forms `hg = dg * cg_j + dp * cp_j`, integrates `hg(r) * r` over the
/// logarithmic radial mesh with its Simpson stencil, and adds the analytic
/// origin contribution from the product of the large/small development
/// coefficients.
pub fn fovrg_overlap_integral(input: FovrgOverlapIntegralInput<'_>) -> Result<Complex, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 3)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.active_len.is_multiple_of(2) {
        return Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            actual: input.active_len,
        });
    }
    validate_active_len(
        "large_integrand",
        input.active_len,
        input.large_integrand.len(),
    )?;
    validate_active_len(
        "small_integrand",
        input.active_len,
        input.small_integrand.len(),
    )?;
    validate_active_len(
        "large_component",
        input.active_len,
        input.large_component.len(),
    )?;
    validate_active_len(
        "small_component",
        input.active_len,
        input.small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "large_integrand_coefficients",
        input.coefficient_count,
        input.large_integrand_coefficients.len(),
    )?;
    validate_active_len(
        "small_integrand_coefficients",
        input.coefficient_count,
        input.small_integrand_coefficients.len(),
    )?;
    validate_active_len(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_active_len(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite("integrand_power", input.integrand_power)?;
    validate_finite("orbital_power", input.orbital_power)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input("large_integrand", row, input.large_integrand[row])?;
        validate_complex_input("small_integrand", row, input.small_integrand[row])?;
        validate_real_input("large_component", row, input.large_component[row])?;
        validate_real_input("small_component", row, input.small_component[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "large_integrand_coefficients",
            coefficient,
            input.large_integrand_coefficients[coefficient],
        )?;
        validate_complex_input(
            "small_integrand_coefficients",
            coefficient,
            input.small_integrand_coefficients[coefficient],
        )?;
        validate_real_input(
            "large_coefficients",
            coefficient,
            input.large_coefficients[coefficient],
        )?;
        validate_real_input(
            "small_coefficients",
            coefficient,
            input.small_coefficients[coefficient],
        )?;
    }

    let mixed_integrand = Array1::from_iter((0..input.active_len).map(|row| {
        (input.large_integrand[row] * input.large_component[row]
            + input.small_integrand[row] * input.small_component[row])
            * input.radii[row]
    }));

    let simpson_sum = (1..input.active_len - 1)
        .step_by(2)
        .fold(Complex::new(0.0, 0.0), |sum, row| {
            sum + mixed_integrand[row] + mixed_integrand[row] + mixed_integrand[row + 1]
        });
    let mut integral = input.step
        * (simpson_sum + simpson_sum + mixed_integrand[0] - mixed_integrand[input.active_len - 1])
        / 3.0;

    let mut origin_power = input.integrand_power + input.orbital_power;
    for coefficient in 1..=input.coefficient_count {
        origin_power += 1.0;
        validate_nonzero_denominator("overlap_origin_power", origin_power)?;
        let origin_coefficient = complex_real_product_coefficient(
            input.large_integrand_coefficients,
            input.large_coefficients,
            coefficient,
        ) + complex_real_product_coefficient(
            input.small_integrand_coefficients,
            input.small_coefficients,
            coefficient,
        );
        integral += origin_coefficient * input.radii[0].powf(origin_power) / origin_power;
    }
    validate_complex_result("overlap_integral", 0, integral)?;
    Ok(integral)
}

/// Port of `FOVRG/ortdac.f90`: Schmidt orthogonalization against bound orbitals.
///
/// FEFF walks the bound orbitals in order, skips orbitals whose kappa differs
/// from `ikap` or whose occupation is not positive, computes the current
/// overlap with `dsordc`, and subtracts that overlap from both the radial
/// target arrays and their origin development coefficients.
pub fn fovrg_schmidt_orthogonalize(
    input: FovrgOrthogonalizationInput<'_>,
) -> Result<FovrgOrthogonalization, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 3)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.active_len.is_multiple_of(2) {
        return Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            actual: input.active_len,
        });
    }
    validate_active_len(
        "target_large_component",
        input.active_len,
        input.target_large_component.len(),
    )?;
    validate_active_len(
        "target_small_component",
        input.active_len,
        input.target_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "target_large_coefficients",
        input.coefficient_count,
        input.target_large_coefficients.len(),
    )?;
    validate_active_len(
        "target_small_coefficients",
        input.coefficient_count,
        input.target_small_coefficients.len(),
    )?;
    validate_matrix_rows(
        "bound_large_components",
        input.active_len,
        input.bound_large_components.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_components",
        input.active_len,
        input.bound_small_components.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_components",
        input.bound_orbital_count,
        input.bound_large_components.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_components",
        input.bound_orbital_count,
        input.bound_small_components.shape()[1],
    )?;
    validate_matrix_rows(
        "bound_large_coefficients",
        input.coefficient_count,
        input.bound_large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_coefficients",
        input.coefficient_count,
        input.bound_small_coefficients.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_coefficients",
        input.bound_orbital_count,
        input.bound_large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_coefficients",
        input.bound_orbital_count,
        input.bound_small_coefficients.shape()[1],
    )?;
    validate_active_len(
        "electron_counts",
        input.bound_orbital_count,
        input.electron_counts.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_active_len(
        "orbital_powers",
        input.bound_orbital_count,
        input.orbital_powers.len(),
    )?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;
    validate_finite("target_power", input.target_power)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input(
            "target_large_component",
            row,
            input.target_large_component[row],
        )?;
        validate_complex_input(
            "target_small_component",
            row,
            input.target_small_component[row],
        )?;
        validate_radius(row, input.radii[row])?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_components",
                row,
                input.bound_large_components[(row, orbital)],
            )?;
            validate_real_input(
                "bound_small_components",
                row,
                input.bound_small_components[(row, orbital)],
            )?;
        }
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "target_large_coefficients",
            coefficient,
            input.target_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "target_small_coefficients",
            coefficient,
            input.target_small_coefficients[coefficient],
        )?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_coefficients",
                coefficient,
                input.bound_large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "bound_small_coefficients",
                coefficient,
                input.bound_small_coefficients[(coefficient, orbital)],
            )?;
        }
    }
    for orbital in 0..input.bound_orbital_count {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        validate_real_input("orbital_powers", orbital, input.orbital_powers[orbital])?;
    }

    let mut large_component = input.target_large_component.to_owned();
    let mut small_component = input.target_small_component.to_owned();
    let mut large_coefficients = input.target_large_coefficients.to_owned();
    let mut small_coefficients = input.target_small_coefficients.to_owned();
    let mut overlaps = Array1::<Complex>::zeros(input.bound_orbital_count);

    for orbital in 0..input.bound_orbital_count {
        if input.kappa[orbital] != input.target_kappa || input.electron_counts[orbital] <= 0.0 {
            continue;
        }

        let overlap = fovrg_overlap_integral(FovrgOverlapIntegralInput {
            large_integrand: large_component.view(),
            small_integrand: small_component.view(),
            large_integrand_coefficients: large_coefficients.view(),
            small_integrand_coefficients: small_coefficients.view(),
            large_component: input.bound_large_components.column(orbital),
            small_component: input.bound_small_components.column(orbital),
            large_coefficients: input.bound_large_coefficients.column(orbital),
            small_coefficients: input.bound_small_coefficients.column(orbital),
            radii: input.radii,
            integrand_power: input.target_power,
            orbital_power: input.orbital_powers[orbital],
            step: input.step,
            coefficient_count: input.coefficient_count,
            active_len: input.active_len,
        })?;
        overlaps[orbital] = overlap;

        for row in 0..input.active_len {
            large_component[row] -= overlap * input.bound_large_components[(row, orbital)];
            small_component[row] -= overlap * input.bound_small_components[(row, orbital)];
        }
        for coefficient in 0..input.coefficient_count {
            large_coefficients[coefficient] -=
                overlap * input.bound_large_coefficients[(coefficient, orbital)];
            small_coefficients[coefficient] -=
                overlap * input.bound_small_coefficients[(coefficient, orbital)];
        }
    }

    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }
    for orbital in 0..input.bound_orbital_count {
        validate_complex_result("overlaps", orbital, overlaps[orbital])?;
    }

    Ok(FovrgOrthogonalization {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        overlaps,
    })
}

/// Port of `FOVRG/muatcc.f90`: angular coefficients for exchange coupling.
///
/// FEFF builds `afgkc(ikap, orbital, index)` for every target kappa. This
/// helper returns the single target-kappa row consumed by `potex`, indexed as
/// `(orbital, index)` with FEFF's fixed five coefficient slots.
pub fn fovrg_angular_coefficients(
    input: FovrgAngularCoefficientsInput<'_>,
) -> Result<RealMat, FovrgError> {
    validate_count_at_least("bound_orbital_count", input.bound_orbital_count, 1)?;
    validate_active_len(
        "electron_counts",
        input.bound_orbital_count,
        input.electron_counts.len(),
    )?;
    validate_active_len(
        "valence_counts",
        input.bound_orbital_count,
        input.valence_counts.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;

    for orbital in 0..input.bound_orbital_count {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_real_input("valence_counts", orbital, input.valence_counts[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
    }

    let target_j = target_j_value(input.target_kappa);
    let target_j_i32 = fovrg_usize_to_i32("target_j", target_j)?;
    let mut coefficients =
        Array2::<Real>::zeros((input.bound_orbital_count, FOVRG_ANGULAR_COEFFICIENT_SLOTS));

    for orbital in 0..input.bound_orbital_count {
        let bound_j = target_j_value(input.kappa[orbital]);
        let max_multipole = (target_j + bound_j) / 2;
        let mut min_multipole = target_j.abs_diff(bound_j) / 2;
        if (input.target_kappa < 0) != (input.kappa[orbital] < 0) {
            min_multipole += 1;
        }
        let required_slots = (max_multipole - min_multipole) / 2 + 1;
        if required_slots > FOVRG_ANGULAR_COEFFICIENT_SLOTS {
            return Err(FovrgError::CountTooLarge {
                name: "angular_coefficient_slots",
                actual: required_slots,
                maximum: FOVRG_ANGULAR_COEFFICIENT_SLOTS,
            });
        }
        if input.valence_counts[orbital] > 0.0 {
            continue;
        }

        let bound_j_i32 = fovrg_usize_to_i32("bound_j", bound_j)?;
        let mut multipole = min_multipole;
        while multipole <= max_multipole {
            let angular_index = (multipole - min_multipole) / 2;
            let doubled_multipole = multipole.checked_mul(2).ok_or(FovrgError::CountTooLarge {
                name: "doubled_multipole",
                actual: multipole,
                maximum: usize::MAX / 2,
            })?;
            let wigner = wigner_3j(
                target_j_i32,
                fovrg_usize_to_i32("doubled_multipole", doubled_multipole)?,
                bound_j_i32,
                1,
                0,
                2,
            )
            .map_err(|source| FovrgError::AngularCoefficient { source })?;
            let coefficient = input.electron_counts[orbital] * wigner * wigner;
            validate_real_result("angular_coefficients", orbital, coefficient)?;
            coefficients[(orbital, angular_index)] = coefficient;
            multipole += 2;
        }
    }

    Ok(coefficients)
}

/// Port of FEFF `FOVRG/dfovrg.f90` `flatv`: exact flat-potential propagation.
///
/// For a constant potential between two radii, FEFF solves the Dirac equation
/// analytically with spherical Bessel and Neumann functions. The returned
/// components are the values at `end_radius` implied by the initial components
/// at `start_radius`.
pub fn fovrg_flat_potential_propagate(
    input: FovrgFlatPotentialInput,
) -> Result<FovrgFlatPotentialPropagation, FovrgError> {
    validate_positive_finite("start_radius", input.start_radius)?;
    validate_positive_finite("end_radius", input.end_radius)?;
    validate_complex_input("large_component", 0, input.large_component)?;
    validate_complex_input("small_component", 0, input.small_component)?;
    validate_complex_input("energy", 0, input.energy)?;
    validate_complex_input("average_potential", 0, input.average_potential)?;
    validate_nonzero_kappa("kappa", 0, input.kappa)?;

    let energy_offset = input.energy - input.average_potential;
    let alpha_wave_offset = FEFF_FINE_STRUCTURE_ALPHA * energy_offset;
    let wave_number = (2.0 * energy_offset + alpha_wave_offset * alpha_wave_offset).sqrt();
    let start_argument = wave_number * input.start_radius;

    let (sign, large_l, small_l) = if input.kappa < 0 {
        let large_l = input.kappa.unsigned_abs() as usize - 1;
        (-1.0, large_l, large_l + 1)
    } else {
        let large_l = input.kappa as usize;
        (1.0, large_l, large_l - 1)
    };
    let max_l = large_l.max(small_l);
    let alpha_wave = wave_number * FEFF_FINE_STRUCTURE_ALPHA;
    let factor = sign * alpha_wave / (1.0 + (1.0 + alpha_wave * alpha_wave).sqrt());
    validate_nonzero_complex_denominator("flat_potential_factor", factor)?;

    let start_bessel = besjn(start_argument, max_l)
        .map_err(|source| FovrgError::FlatPotentialBessel { source })?;
    let amplitude_j = sign
        * wave_number
        * start_argument
        * (input.large_component * start_bessel.y[small_l]
            - input.small_component * start_bessel.y[large_l] / factor);
    let amplitude_y = sign
        * wave_number
        * start_argument
        * (input.small_component * start_bessel.j[large_l] / factor
            - input.large_component * start_bessel.j[small_l]);

    let end_argument = wave_number * input.end_radius;
    let end_bessel =
        besjn(end_argument, max_l).map_err(|source| FovrgError::FlatPotentialBessel { source })?;
    let large_component = input.end_radius
        * (end_bessel.j[large_l] * amplitude_j + end_bessel.y[large_l] * amplitude_y);
    let small_component = input.end_radius
        * factor
        * (end_bessel.j[small_l] * amplitude_j + end_bessel.y[small_l] * amplitude_y);

    validate_complex_result("flat_large_component", 0, large_component)?;
    validate_complex_result("flat_small_component", 0, small_component)?;
    Ok(FovrgFlatPotentialPropagation {
        large_component,
        small_component,
    })
}

/// Port of FEFF `FOVRG/intout.f90`: outward Dirac radial integration.
///
/// FEFF starts with a six-point Runge-Kutta bootstrap, converts those
/// derivatives to Milne history values, then advances the inhomogeneous Dirac
/// system with predictor-corrector iterations. Rows after `last_index` are
/// zero-filled like FEFF's `max0+1:np` cleanup.
pub fn fovrg_outward_integrate(
    input: FovrgOutwardIntegrationInput<'_>,
) -> Result<FovrgOutwardIntegration, FovrgError> {
    validate_outward_integration_input(&input)?;

    let ccl = input.speed_of_light + input.speed_of_light;
    let kappa = input.kappa as Real;
    let energy_over_light = input.energy / input.speed_of_light;
    let exp_half_step = (input.step / 2.0).exp();
    let mut large_component = Array1::<Complex>::zeros(input.active_len);
    let mut small_component = Array1::<Complex>::zeros(input.active_len);
    large_component[input.start_index] = input.initial_large_component;
    small_component[input.start_index] = input.initial_small_component;

    let mut large_derivative = [Complex::new(0.0, 0.0); FOVRG_INT_OUT_HISTORY];
    let mut small_derivative = [Complex::new(0.0, 0.0); FOVRG_INT_OUT_HISTORY];
    let mut current = input.start_index;
    let mut history_index = 0usize;
    let mut difficult_iterations = 0usize;

    let (f, g, c3) = fovrg_outward_grid_terms(&input, current, energy_over_light, ccl)?;
    large_derivative[history_index] = input.step
        * (g * small_component[current] - kappa * large_component[current]
            + input.small_exchange[current]);
    small_derivative[history_index] = input.step
        * (kappa * small_component[current]
            - (f - c3) * large_component[current]
            - input.large_exchange[current]);

    while current < input.last_index {
        let midpoint =
            fovrg_outward_midpoint_terms(&input, current, energy_over_light, ccl, exp_half_step)?;
        let mut large_trial = large_component[current] + 0.5 * large_derivative[history_index];
        let mut small_trial = small_component[current] + 0.5 * small_derivative[history_index];
        let large_derivative_2 =
            input.step * (midpoint.g * small_trial - kappa * large_trial + midpoint.small_exchange);
        let small_derivative_2 = input.step
            * (kappa * small_trial
                - (midpoint.f - midpoint.c3) * large_trial
                - midpoint.large_exchange);
        large_trial += F77_REAL_HALF * (large_derivative_2 - large_derivative[history_index]);
        small_trial += F77_REAL_HALF * (small_derivative_2 - small_derivative[history_index]);
        let large_derivative_3 =
            input.step * (midpoint.g * small_trial - kappa * large_trial + midpoint.small_exchange);
        let small_derivative_3 = input.step
            * (kappa * small_trial
                - (midpoint.f - midpoint.c3) * large_trial
                - midpoint.large_exchange);
        large_trial += large_derivative_3 - F77_REAL_HALF * large_derivative_2;
        small_trial += small_derivative_3 - F77_REAL_HALF * small_derivative_2;

        current += 1;
        history_index += 1;
        let (f, g, c3) = fovrg_outward_grid_terms(&input, current, energy_over_light, ccl)?;
        let large_derivative_4 =
            input.step * (g * small_trial - kappa * large_trial + input.small_exchange[current]);
        let small_derivative_4 = input.step
            * (kappa * small_trial - (f - c3) * large_trial - input.large_exchange[current]);
        large_component[current] = large_component[current - 1]
            + (large_derivative[history_index - 1]
                + F77_REAL_TWO * (large_derivative_2 + large_derivative_3)
                + large_derivative_4)
                / F77_REAL_SIX;
        small_component[current] = small_component[current - 1]
            + (small_derivative[history_index - 1]
                + F77_REAL_TWO * (small_derivative_2 + small_derivative_3)
                + small_derivative_4)
                / F77_REAL_SIX;
        large_derivative[history_index] = input.step
            * (g * small_component[current] - kappa * large_component[current]
                + input.small_exchange[current]);
        small_derivative[history_index] = input.step
            * (kappa * small_component[current]
                - (f - c3) * large_component[current]
                - input.large_exchange[current]);

        if history_index + 1 >= FOVRG_INT_OUT_HISTORY {
            break;
        }
    }

    if current < input.last_index {
        for row in 0..FOVRG_INT_OUT_HISTORY {
            large_derivative[row] /= input.step;
            small_derivative[row] /= input.step;
        }

        let a1 = input.step * F77_REAL_THREE_POINT_THREE;
        let a2 = -input.step * F77_REAL_FOUR_POINT_TWO;
        let a3 = input.step * F77_REAL_SEVEN_POINT_EIGHT;
        let a4 = input.step * F77_REAL_FOURTEEN_OVER_FORTY_FIVE;
        let a5 = input.step * F77_REAL_SIXTY_FOUR_OVER_FORTY_FIVE;
        let a6 = input.step * F77_REAL_TWENTY_FOUR_OVER_FORTY_FIVE;

        for row in (input.start_index + FOVRG_INT_OUT_HISTORY - 1)..input.last_index {
            let mut predicted_large = large_component[row - 5]
                + a1 * (large_derivative[5] + large_derivative[1])
                + a2 * (large_derivative[4] + large_derivative[2])
                + a3 * large_derivative[3];
            let mut predicted_small = small_component[row - 5]
                + a1 * (small_derivative[5] + small_derivative[1])
                + a2 * (small_derivative[4] + small_derivative[2])
                + a3 * small_derivative[3];
            let corrected_large_base = large_component[row - 3]
                + a4 * large_derivative[2]
                + a5 * (large_derivative[5] + large_derivative[3])
                + a6 * large_derivative[4];
            let corrected_small_base = small_component[row - 3]
                + a4 * small_derivative[2]
                + a5 * (small_derivative[5] + small_derivative[3])
                + a6 * small_derivative[4];

            large_derivative.copy_within(1..FOVRG_INT_OUT_HISTORY, 0);
            small_derivative.copy_within(1..FOVRG_INT_OUT_HISTORY, 0);

            let next = row + 1;
            let (f, g, c3) = fovrg_outward_grid_terms(&input, next, energy_over_light, ccl)?;
            let mut retry_count = 0usize;
            loop {
                large_derivative[5] =
                    g * predicted_small - kappa * predicted_large + input.small_exchange[next];
                small_derivative[5] = kappa * predicted_small
                    - (f - c3) * predicted_large
                    - input.large_exchange[next];
                large_component[next] = corrected_large_base + a4 * large_derivative[5];
                small_component[next] = corrected_small_base + a4 * small_derivative[5];

                let large_failed = (FOVRG_INT_OUT_TEST * (large_component[next] - predicted_large))
                    .norm()
                    > large_component[next].norm();
                let small_failed = (FOVRG_INT_OUT_TEST * (small_component[next] - predicted_small))
                    .norm()
                    > small_component[next].norm();
                if large_failed || small_failed {
                    if retry_count < 40 {
                        predicted_large = large_component[next];
                        predicted_small = small_component[next];
                        retry_count += 1;
                    } else {
                        difficult_iterations += 1;
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    for row in input.last_index + 1..input.active_len {
        large_component[row] = Complex::new(0.0, 0.0);
        small_component[row] = Complex::new(0.0, 0.0);
    }
    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }

    Ok(FovrgOutwardIntegration {
        large_component,
        small_component,
        difficult_iterations,
    })
}

/// Port of FEFF `FOVRG/solout.f90`: regular solution integrated outward.
///
/// FEFF builds the origin power-series coefficients, integrates from the
/// origin through `min(jri, iwkb)` with `intout`, then uses exact flat-potential
/// propagation to reach `max0`. The returned arrays follow FEFF's zero-fill
/// convention after `last_index`.
pub fn fovrg_outgoing_solution(
    input: FovrgOutgoingSolutionInput<'_>,
) -> Result<FovrgOutgoingSolution, FovrgError> {
    validate_outgoing_solution_input(&input)?;

    let mut large_coefficients = Array1::<Complex>::zeros(
        input
            .potential_coefficients
            .len()
            .max(input.coefficient_count),
    );
    let mut small_coefficients = Array1::<Complex>::zeros(
        input
            .potential_coefficients
            .len()
            .max(input.coefficient_count),
    );
    let mut initial_small_coefficient = input.initial_small_coefficient;
    if input.potential_coefficients[0].re < 0.0 {
        if input.kappa > 0 {
            validate_nonzero_complex_denominator(
                "solout_point_nucleus_large_denominator",
                input.potential_coefficients[0],
            )?;
            initial_small_coefficient = -input.initial_large_coefficient
                * (input.kappa as Real + input.origin_power)
                / input.potential_coefficients[0];
        } else if input.kappa < 0 {
            let denominator = input.kappa as Real - input.origin_power;
            validate_nonzero_denominator("solout_point_nucleus_small_denominator", denominator)?;
            initial_small_coefficient =
                -input.initial_large_coefficient * input.potential_coefficients[0] / denominator;
        }
    }

    large_coefficients[0] = input.initial_large_coefficient;
    small_coefficients[0] = initial_small_coefficient;
    for coefficient in 1..input.coefficient_count {
        large_coefficients[coefficient] = input.large_exchange_coefficients[coefficient - 1];
        small_coefficients[coefficient] = input.small_exchange_coefficients[coefficient - 1];
    }

    let energy_over_light = input.energy / input.speed_of_light;
    if input.c3_scale == 0 {
        fovrg_desclaux_origin_series(
            input,
            &mut large_coefficients,
            &mut small_coefficients,
            energy_over_light,
        )?;
    } else {
        fovrg_relativistic_origin_series(
            input,
            &mut large_coefficients,
            &mut small_coefficients,
            energy_over_light,
        )?;
    }

    let (initial_large_component, initial_small_component) =
        fovrg_origin_components(input, large_coefficients.view(), small_coefficients.view());
    let flat_start_index = input.radial_match_index.min(input.wkb_index);
    let integrated = fovrg_outward_integrate(FovrgOutwardIntegrationInput {
        initial_large_component,
        initial_small_component,
        energy: input.energy,
        potential: input.potential,
        potential_coefficients: input.potential_coefficients,
        large_exchange: input.large_exchange,
        small_exchange: input.small_exchange,
        c3_potential: input.c3_potential,
        radii: input.radii,
        speed_of_light: input.speed_of_light,
        step: input.step,
        kappa: input.kappa,
        c3_scale: input.c3_scale,
        start_index: 0,
        last_index: flat_start_index,
        active_len: input.active_len,
    })?;
    let mut large_component = integrated.large_component;
    let mut small_component = integrated.small_component;

    for row in flat_start_index..input.last_index {
        let average_potential = fovrg_solout_average_potential(input, row)?;
        let propagated = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            start_radius: input.radii[row],
            end_radius: input.radii[row + 1],
            large_component: large_component[row],
            small_component: small_component[row],
            energy: input.energy,
            average_potential,
            kappa: input.kappa,
        })?;
        large_component[row + 1] = propagated.large_component;
        small_component[row + 1] = propagated.small_component;
    }
    for row in input.last_index + 1..input.active_len {
        large_component[row] = Complex::new(0.0, 0.0);
        small_component[row] = Complex::new(0.0, 0.0);
    }

    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }

    Ok(FovrgOutgoingSolution {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        difficult_iterations: integrated.difficult_iterations,
    })
}

/// Port of FEFF `FOVRG/solin.f90`: irregular solution integrated inward.
///
/// FEFF seeds the outer region from spherical Hankel functions, propagates
/// through the flat-potential interval, then integrates the inhomogeneous
/// Dirac system inward to the first radial point. Coefficients after the first
/// origin term are zero-filled to match `solin`.
pub fn fovrg_inward_solution(
    input: FovrgInwardSolutionInput<'_>,
) -> Result<FovrgInwardSolution, FovrgError> {
    validate_inward_solution_input(&input)?;

    let ccl = input.speed_of_light + input.speed_of_light;
    let flat_start_index = input.radial_match_index.min(input.wkb_index);
    let mut derivative_energy = input.energy / input.speed_of_light;
    let mut large_component = Array1::<Complex>::zeros(input.active_len);
    let mut small_component = Array1::<Complex>::zeros(input.active_len);
    let mut large_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    let mut small_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    let mut large_derivative = [Complex::new(0.0, 0.0); FOVRG_INT_OUT_HISTORY];
    let mut small_derivative = [Complex::new(0.0, 0.0); FOVRG_INT_OUT_HISTORY];

    let match_potential = input.speed_of_light * input.potential[input.radial_match_index + 1];
    let energy_offset = input.energy - match_potential;
    let alpha_wave_offset = FEFF_FINE_STRUCTURE_ALPHA * energy_offset;
    let wave_number = (2.0 * energy_offset + alpha_wave_offset * alpha_wave_offset).sqrt();
    let large_l = if input.kappa < 0 {
        input.kappa.unsigned_abs() as usize - 1
    } else {
        input.kappa as usize
    };
    let small_l = if input.kappa < 0 {
        large_l + 1
    } else {
        large_l - 1
    };
    let max_l = large_l.max(small_l);
    let sign = if input.kappa > 0 { 1.0 } else { -1.0 };
    let alpha_wave = wave_number * FEFF_FINE_STRUCTURE_ALPHA;
    let factor = sign * alpha_wave / (1.0 + (1.0 + alpha_wave * alpha_wave).sqrt());
    let normalization_denominator = (1.0 + factor * factor).sqrt();
    validate_nonzero_complex_denominator("inward_hankel_normalization", normalization_denominator)?;
    let normalization = Complex::new(1.0, 0.0) / normalization_denominator;

    for row in input.radial_match_index..=input.last_index {
        let argument = wave_number * input.radii[row];
        let hankel =
            besjh(argument, max_l).map_err(|source| FovrgError::FlatPotentialBessel { source })?;
        large_component[row] = hankel.h[large_l] * input.radii[row] * normalization;
        small_component[row] = hankel.h[small_l] * input.radii[row] * normalization * factor;

        if let Some(history_slot) = fovrg_inward_history_slot(flat_start_index, row) {
            let (large, small) = fovrg_inward_derivatives(
                &input,
                row,
                derivative_energy,
                ccl,
                false,
                large_component[row],
                small_component[row],
            )?;
            large_derivative[history_slot] = large;
            small_derivative[history_slot] = small;
        }
    }

    for row in (flat_start_index..input.radial_match_index).rev() {
        let mut average_potential = fovrg_solin_average_potential(input, row)?;
        if input.c3_scale > 0 {
            let radius_average = (input.radii[row] + input.radii[row + 1]) / 2.0;
            derivative_energy = radius_average.powi(3)
                * (ccl + (input.energy - average_potential) / input.speed_of_light).powi(2);
            validate_nonzero_complex_denominator("solin_c3_flat_denominator", derivative_energy)?;
            average_potential += (input.c3_scale as Real) * input.speed_of_light
                / derivative_energy
                * (input.c3_potential[row] + input.c3_potential[row + 1])
                / 2.0;
        }

        let propagated = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            start_radius: input.radii[row + 1],
            end_radius: input.radii[row],
            large_component: large_component[row + 1],
            small_component: small_component[row + 1],
            energy: input.energy,
            average_potential,
            kappa: input.kappa,
        })?;
        large_component[row] = propagated.large_component;
        small_component[row] = propagated.small_component;

        if let Some(history_slot) = fovrg_inward_history_slot(flat_start_index, row) {
            let (large, small) = fovrg_inward_derivatives(
                &input,
                row,
                derivative_energy,
                ccl,
                true,
                large_component[row],
                small_component[row],
            )?;
            large_derivative[history_slot] = large;
            small_derivative[history_slot] = small;
        }
    }

    let a1 = input.step * F77_REAL_THREE_POINT_THREE;
    let a2 = -input.step * F77_REAL_FOUR_POINT_TWO;
    let a3 = input.step * F77_REAL_SEVEN_POINT_EIGHT;
    let a4 = input.step * F77_REAL_FOURTEEN_OVER_FORTY_FIVE;
    let a5 = input.step * F77_REAL_SIXTY_FOUR_OVER_FORTY_FIVE;
    let a6 = input.step * F77_REAL_TWENTY_FOUR_OVER_FORTY_FIVE;
    let mut difficult_iterations = 0usize;

    for row in (1..=flat_start_index).rev() {
        let mut predicted_large = large_component[row + 5]
            + a1 * (large_derivative[5] + large_derivative[1])
            + a2 * (large_derivative[4] + large_derivative[2])
            + a3 * large_derivative[3];
        let mut predicted_small = small_component[row + 5]
            + a1 * (small_derivative[5] + small_derivative[1])
            + a2 * (small_derivative[4] + small_derivative[2])
            + a3 * small_derivative[3];
        let corrected_large_base = large_component[row + 3]
            + a4 * large_derivative[2]
            + a5 * (large_derivative[5] + large_derivative[3])
            + a6 * large_derivative[4];
        let corrected_small_base = small_component[row + 3]
            + a4 * small_derivative[2]
            + a5 * (small_derivative[5] + small_derivative[3])
            + a6 * small_derivative[4];

        large_derivative.copy_within(1..FOVRG_INT_OUT_HISTORY, 0);
        small_derivative.copy_within(1..FOVRG_INT_OUT_HISTORY, 0);

        let next = row - 1;
        let mut retry_count = 0usize;
        loop {
            let (large, small) = fovrg_inward_derivatives(
                &input,
                next,
                derivative_energy,
                ccl,
                true,
                predicted_large,
                predicted_small,
            )?;
            large_derivative[5] = large;
            small_derivative[5] = small;
            large_component[next] = corrected_large_base + a4 * large_derivative[5];
            small_component[next] = corrected_small_base + a4 * small_derivative[5];

            let large_failed = (FOVRG_INT_OUT_TEST * (large_component[next] - predicted_large))
                .norm()
                > large_component[next].norm();
            let small_failed = (FOVRG_INT_OUT_TEST * (small_component[next] - predicted_small))
                .norm()
                > small_component[next].norm();
            if large_failed || small_failed {
                if retry_count < 40 {
                    predicted_large = large_component[next];
                    predicted_small = small_component[next];
                    retry_count += 1;
                } else {
                    difficult_iterations += 1;
                    break;
                }
            } else {
                break;
            }
        }
    }

    for row in input.last_index + 1..input.active_len {
        large_component[row] = Complex::new(0.0, 0.0);
        small_component[row] = Complex::new(0.0, 0.0);
    }

    let origin_scale = input.radii[0].powf(-input.origin_power);
    large_coefficients[0] = large_component[0] * origin_scale;
    small_coefficients[0] = small_component[0] * origin_scale;

    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }

    Ok(FovrgInwardSolution {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        difficult_iterations,
    })
}

/// Port of `FOVRG/potex.f90`: exchange-potential accumulation.
///
/// FEFF loops over bound orbitals and allowed multipoles, obtains the `yk`
/// exchange kernel from `yzkrdc`, accumulates the radial exchange potentials
/// `eg/ep`, updates their origin development coefficients `ceg/cep`, and
/// finally divides retained rows and coefficients by `cl`.
pub fn fovrg_exchange_potential(
    input: FovrgExchangePotentialInput<'_>,
) -> Result<FovrgExchangePotential, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len(
        "target_large_component",
        input.active_len,
        input.target_large_component.len(),
    )?;
    validate_active_len(
        "target_small_component",
        input.active_len,
        input.target_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "target_large_coefficients",
        input.coefficient_count,
        input.target_large_coefficients.len(),
    )?;
    validate_active_len(
        "target_small_coefficients",
        input.coefficient_count,
        input.target_small_coefficients.len(),
    )?;
    validate_matrix_rows(
        "bound_large_components",
        input.active_len,
        input.bound_large_components.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_components",
        input.active_len,
        input.bound_small_components.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_components",
        input.bound_orbital_count,
        input.bound_large_components.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_components",
        input.bound_orbital_count,
        input.bound_small_components.shape()[1],
    )?;
    validate_matrix_rows(
        "bound_large_coefficients",
        input.coefficient_count,
        input.bound_large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_coefficients",
        input.coefficient_count,
        input.bound_small_coefficients.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_coefficients",
        input.bound_orbital_count,
        input.bound_large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_coefficients",
        input.bound_orbital_count,
        input.bound_small_coefficients.shape()[1],
    )?;
    validate_matrix_rows(
        "angular_coefficients",
        input.bound_orbital_count,
        input.angular_coefficients.shape()[0],
    )?;
    validate_active_len(
        "orbital_powers",
        input.bound_orbital_count,
        input.orbital_powers.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_active_len(
        "orbital_lengths",
        input.bound_orbital_count,
        input.orbital_lengths.len(),
    )?;
    validate_active_len(
        "normalization",
        input.bound_orbital_count,
        input.normalization.len(),
    )?;
    validate_active_len(
        "radial_output_count",
        input.radial_output_count,
        input.active_len,
    )?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;
    validate_finite("target_power", input.target_power)?;
    validate_nonzero_finite("target_normalization", input.target_normalization)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input(
            "target_large_component",
            row,
            input.target_large_component[row],
        )?;
        validate_complex_input(
            "target_small_component",
            row,
            input.target_small_component[row],
        )?;
        validate_radius(row, input.radii[row])?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_components",
                row,
                input.bound_large_components[(row, orbital)],
            )?;
            validate_real_input(
                "bound_small_components",
                row,
                input.bound_small_components[(row, orbital)],
            )?;
        }
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "target_large_coefficients",
            coefficient,
            input.target_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "target_small_coefficients",
            coefficient,
            input.target_small_coefficients[coefficient],
        )?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_coefficients",
                coefficient,
                input.bound_large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "bound_small_coefficients",
                coefficient,
                input.bound_small_coefficients[(coefficient, orbital)],
            )?;
        }
    }
    for orbital in 0..input.bound_orbital_count {
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        validate_real_input("orbital_powers", orbital, input.orbital_powers[orbital])?;
        validate_real_input("normalization", orbital, input.normalization[orbital])?;
        validate_count_at_least("orbital_length", input.orbital_lengths[orbital], 1)?;
        for index in 0..input.angular_coefficients.shape()[1] {
            validate_real_input(
                "angular_coefficients",
                orbital,
                input.angular_coefficients[(orbital, index)],
            )?;
        }
    }

    let target_j = target_j_value(input.target_kappa);
    let mut large_potential = Array1::<Complex>::zeros(input.active_len);
    let mut small_potential = Array1::<Complex>::zeros(input.active_len);
    let mut large_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    let mut small_coefficients = Array1::<Complex>::zeros(input.coefficient_count);

    for orbital in 0..input.bound_orbital_count {
        let bound_j = target_j_value(input.kappa[orbital]);
        let max_multipole = (bound_j + target_j) / 2;
        let mut multipole = bound_j.abs_diff(max_multipole);
        if (input.kappa[orbital] < 0) != (input.target_kappa < 0) {
            multipole += 1;
        }
        let min_multipole = multipole;

        while multipole <= max_multipole {
            let angular_index = (multipole - min_multipole) / 2;
            validate_matrix_cols(
                "angular_coefficients",
                angular_index + 1,
                input.angular_coefficients.shape()[1],
            )?;
            let angular_coefficient = input.angular_coefficients[(orbital, angular_index)];
            if angular_coefficient != 0.0 {
                let transform = fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
                    large_component: input.bound_large_components.column(orbital),
                    small_component: input.bound_small_components.column(orbital),
                    large_coefficients: input.bound_large_coefficients.column(orbital),
                    small_coefficients: input.bound_small_coefficients.column(orbital),
                    partner_large_component: input.target_large_component,
                    partner_small_component: input.target_small_component,
                    partner_large_coefficients: input.target_large_coefficients,
                    partner_small_coefficients: input.target_small_coefficients,
                    radii: input.radii,
                    orbital_power: input.orbital_powers[orbital],
                    partner_power: input.target_power,
                    step: input.step,
                    angular_momentum: multipole,
                    coefficient_count: input.coefficient_count,
                    orbital_len: input.orbital_lengths[orbital],
                    source_len: input.source_len,
                    active_len: input.active_len,
                })?;

                for row in 0..input.active_len {
                    large_potential[row] += angular_coefficient
                        * transform.yk[row]
                        * input.bound_large_components[(row, orbital)];
                    small_potential[row] += angular_coefficient
                        * transform.yk[row]
                        * input.bound_small_components[(row, orbital)];
                }

                if let Some(coefficient_start) = exchange_coefficient_start(
                    multipole,
                    input.kappa[orbital],
                    input.target_kappa,
                    input.target_power,
                )
                .filter(|&start| start <= input.coefficient_count)
                {
                    for coefficient in coefficient_start..=input.coefficient_count {
                        let target_row = coefficient - 1;
                        let bound_row = coefficient - coefficient_start;
                        let scale = angular_coefficient
                            * transform.origin_constant
                            * input.normalization[orbital]
                            / input.target_normalization;
                        large_coefficients[target_row] +=
                            input.bound_large_coefficients[(bound_row, orbital)] * scale;
                        small_coefficients[target_row] +=
                            input.bound_small_coefficients[(bound_row, orbital)] * scale;
                    }
                }

                let product_start = 2 * input.kappa[orbital].unsigned_abs() as usize + 1;
                if product_start <= input.coefficient_count {
                    let scale = angular_coefficient * input.normalization[orbital].powi(2);
                    for coefficient in product_start..=input.coefficient_count {
                        let product_count = coefficient + 1 - product_start;
                        large_coefficients[coefficient - 1] -= scale
                            * complex_real_product_coefficient(
                                transform.yk_coefficients.view(),
                                input.bound_large_coefficients.column(orbital),
                                product_count,
                            );
                        small_coefficients[coefficient - 1] -= scale
                            * complex_real_product_coefficient(
                                transform.yk_coefficients.view(),
                                input.bound_small_coefficients.column(orbital),
                                product_count,
                            );
                    }
                }
            }
            multipole += 2;
        }
    }

    for coefficient in 0..input.coefficient_count {
        large_coefficients[coefficient] /= input.speed_of_light;
        small_coefficients[coefficient] /= input.speed_of_light;
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }
    for row in 0..input.active_len {
        if row < input.radial_output_count {
            large_potential[row] /= input.speed_of_light;
            small_potential[row] /= input.speed_of_light;
        } else {
            large_potential[row] = Complex::new(0.0, 0.0);
            small_potential[row] = Complex::new(0.0, 0.0);
        }
        validate_complex_result("large_potential", row, large_potential[row])?;
        validate_complex_result("small_potential", row, small_potential[row])?;
    }

    Ok(FovrgExchangePotential {
        large_potential,
        small_potential,
        large_coefficients,
        small_coefficients,
    })
}

/// Port of `FOVRG/nucdec.f90`: point-nucleus radial grid and potential.
///
/// FEFF10 currently resets the nuclear mass to zero inside `nucdec`, so the
/// active branch is the point-nucleus Coulomb potential:
/// `dr(i) = dr1 / dz * exp(hx * (i - 1))`, `dv(i) = -dz / dr(i)`, and
/// `av(1) = -dz` with all remaining development coefficients zero.
pub fn fovrg_nuclear_potential(
    input: FovrgNuclearPotentialInput,
) -> Result<FovrgNuclearPotential, FovrgError> {
    validate_positive_finite("nuclear_charge", input.nuclear_charge)?;
    validate_positive_finite("step", input.step)?;
    validate_positive_finite("first_radius_times_charge", input.first_radius_times_charge)?;
    validate_count_at_least("radial_count", input.radial_count, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 5)?;

    let first_radius = input.first_radius_times_charge / input.nuclear_charge;
    let mut radii = Array1::<Real>::zeros(input.radial_count);
    let mut potential = Array1::<Real>::zeros(input.radial_count);
    for row in 0..input.radial_count {
        radii[row] = first_radius * (input.step * row as Real).exp();
        validate_radius(row, radii[row])?;

        potential[row] = -input.nuclear_charge / radii[row];
        validate_real_result("nuclear_potential", row, potential[row])?;
    }

    let mut development_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    development_coefficients[0] = -input.nuclear_charge;
    validate_real_result("development_coefficients", 0, development_coefficients[0])?;

    Ok(FovrgNuclearPotential {
        development_coefficients,
        radii,
        potential,
        nucleus_index: 1,
        first_radius_times_charge: input.first_radius_times_charge,
    })
}

/// Port of `FOVRG/potdvp.f90`: potential development coefficients.
///
/// FEFF accumulates bound-orbital density development coefficients from
/// occupied large/small radial polynomials, integrates those coefficients into
/// a local potential expansion, adds the nuclear development, and divides the
/// resulting `av` coefficients by `cl`.
pub fn fovrg_potential_development(
    input: FovrgPotentialDevelopmentInput<'_>,
) -> Result<FovrgPotentialDevelopment, FovrgError> {
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_count_at_least("orbital_count", input.orbital_count, 1)?;
    validate_count_at_least("nuclear_coefficients", input.nuclear_coefficients.len(), 2)?;
    validate_count_at_least("radii", input.radii.len(), 1)?;
    validate_active_len(
        "nuclear_coefficients",
        input.coefficient_count,
        input.nuclear_coefficients.len(),
    )?;
    validate_matrix_rows(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.shape()[0],
    )?;
    let bound_orbitals = input.orbital_count - 1;
    validate_matrix_cols(
        "large_coefficients",
        bound_orbitals,
        input.large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "small_coefficients",
        bound_orbitals,
        input.small_coefficients.shape()[1],
    )?;
    validate_active_len(
        "electron_counts",
        bound_orbitals,
        input.electron_counts.len(),
    )?;
    validate_active_len("kappa", bound_orbitals, input.kappa.len())?;
    validate_active_len("normalization", bound_orbitals, input.normalization.len())?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_radius(0, input.radii[0])?;
    if input.coefficient_count > i32::MAX as usize - 1 {
        return Err(FovrgError::CountTooLarge {
            name: "coefficient_count",
            actual: input.coefficient_count,
            maximum: i32::MAX as usize - 1,
        });
    }

    for coefficient in 0..input.nuclear_coefficients.len() {
        validate_real_input(
            "nuclear_coefficients",
            coefficient,
            input.nuclear_coefficients[coefficient],
        )?;
    }
    for orbital in 0..bound_orbitals {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_real_input("normalization", orbital, input.normalization[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        for coefficient in 0..input.coefficient_count {
            validate_real_input(
                "large_coefficients",
                coefficient,
                input.large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "small_coefficients",
                coefficient,
                input.small_coefficients[(coefficient, orbital)],
            )?;
        }
    }

    let mut density_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    for orbital in 0..bound_orbitals {
        let kappa_abs = input.kappa[orbital].unsigned_abs() as usize;
        let leading_power = kappa_abs.saturating_mul(2);
        let product_count = input.coefficient_count + 2;
        if leading_power >= product_count {
            continue;
        }
        let max_product_order = product_count - leading_power;
        for product_order in 1..=max_product_order {
            let density_row = leading_power - 2 + product_order;
            density_coefficients[density_row - 1] += input.electron_counts[orbital]
                * (real_product_coefficient(
                    input.large_coefficients.column(orbital),
                    input.large_coefficients.column(orbital),
                    product_order,
                ) + real_product_coefficient(
                    input.small_coefficients.column(orbital),
                    input.small_coefficients.column(orbital),
                    product_order,
                ))
                * input.normalization[orbital].powi(2);
        }
    }

    let mut origin_correction = 0.0;
    for coefficient in 1..=input.coefficient_count {
        let row = coefficient - 1;
        density_coefficients[row] /= (coefficient + 2) as Real * (coefficient + 1) as Real;
        origin_correction +=
            density_coefficients[row] * input.radii[0].powi(coefficient as i32 + 1);
    }

    let mut potential_coefficients = Array1::from_iter(
        input
            .nuclear_coefficients
            .iter()
            .copied()
            .map(|value| Complex::new(value, 0.0)),
    );
    for coefficient in 1..=input.coefficient_count {
        let potential_row = coefficient + 3;
        if potential_row <= input.coefficient_count {
            potential_coefficients[potential_row - 1] -= density_coefficients[coefficient - 1];
        }
    }
    potential_coefficients[1] += origin_correction;
    for row in 0..potential_coefficients.len() {
        potential_coefficients[row] /= input.speed_of_light;
        validate_complex_result("potential_coefficients", row, potential_coefficients[row])?;
    }
    for row in 0..density_coefficients.len() {
        validate_real_result("density_coefficients", row, density_coefficients[row])?;
    }
    validate_real_result("origin_correction", 0, origin_correction)?;

    Ok(FovrgPotentialDevelopment {
        potential_coefficients,
        density_coefficients,
        origin_correction,
    })
}

#[derive(Debug, Clone, Copy)]
struct FovrgOutwardMidpoint {
    f: Complex,
    g: Complex,
    c3: Complex,
    large_exchange: Complex,
    small_exchange: Complex,
}

fn validate_outward_integration_input(
    input: &FovrgOutwardIntegrationInput<'_>,
) -> Result<(), FovrgError> {
    validate_count_at_least("active_len", input.active_len, 1)?;
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len(
        "large_exchange",
        input.active_len,
        input.large_exchange.len(),
    )?;
    validate_active_len(
        "small_exchange",
        input.active_len,
        input.small_exchange.len(),
    )?;
    validate_active_len("c3_potential", input.active_len, input.c3_potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_count_at_least(
        "potential_coefficients",
        input.potential_coefficients.len(),
        4,
    )?;
    if input.start_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "start_index",
            active_len: input.start_index + 1,
            len: input.active_len,
        });
    }
    if input.last_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "last_index",
            active_len: input.last_index + 1,
            len: input.active_len,
        });
    }
    if input.start_index > input.last_index {
        return Err(FovrgError::InvalidRange {
            name: "outward_integration",
            start: input.start_index,
            end: input.last_index,
        });
    }
    validate_nonzero_kappa("kappa", 0, input.kappa)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_nonzero_finite("step", input.step)?;
    validate_complex_input("initial_large_component", 0, input.initial_large_component)?;
    validate_complex_input("initial_small_component", 0, input.initial_small_component)?;
    validate_complex_input("energy", 0, input.energy)?;
    for row in 0..input.active_len {
        validate_complex_input("potential", row, input.potential[row])?;
        validate_complex_input("large_exchange", row, input.large_exchange[row])?;
        validate_complex_input("small_exchange", row, input.small_exchange[row])?;
        validate_complex_input("c3_potential", row, input.c3_potential[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for row in 0..input.potential_coefficients.len() {
        validate_complex_input(
            "potential_coefficients",
            row,
            input.potential_coefficients[row],
        )?;
    }
    Ok(())
}

fn fovrg_outward_grid_terms(
    input: &FovrgOutwardIntegrationInput<'_>,
    row: usize,
    energy_over_light: Complex,
    ccl: Real,
) -> Result<(Complex, Complex, Complex), FovrgError> {
    let f = (energy_over_light - input.potential[row]) * input.radii[row];
    let g = f + ccl * input.radii[row];
    let c3 = fovrg_outward_c3(input.c3_scale, input.c3_potential[row], g)?;
    Ok((f, g, c3))
}

fn fovrg_outward_midpoint_terms(
    input: &FovrgOutwardIntegrationInput<'_>,
    row: usize,
    energy_over_light: Complex,
    ccl: Real,
    exp_half_step: Real,
) -> Result<FovrgOutwardMidpoint, FovrgError> {
    let next = row + 1;
    let radius = input.radii[row];
    let next_radius = input.radii[next];
    let half_radius = radius * exp_half_step;
    let interval = next_radius - radius;
    validate_nonzero_denominator("outward_radius_interval", interval)?;
    let left_weight = (next_radius - half_radius) / interval;
    let right_weight = (half_radius - radius) / interval;

    let (potential, c3_potential) = if input.potential_coefficients[0].re < 0.0
        && input.start_index == 0
    {
        let left_potential = input.potential[row] - input.potential_coefficients[0] / radius;
        let right_potential = input.potential[next] - input.potential_coefficients[0] / next_radius;
        let potential = left_potential * left_weight
            + right_potential * right_weight
            + input.potential_coefficients[0] / half_radius;
        let c3_potential = (input.c3_potential[row] * left_weight * radius
            + input.c3_potential[next] * right_weight * next_radius)
            / half_radius;
        (potential, c3_potential)
    } else if input.start_index == 0 {
        let left_radius_sq = radius * radius;
        let right_radius_sq = next_radius * next_radius;
        let half_radius_sq = half_radius * half_radius;
        let left_potential =
            input.potential[row] - input.potential_coefficients[3] * left_radius_sq;
        let right_potential =
            input.potential[next] - input.potential_coefficients[3] * right_radius_sq;
        let potential = (left_potential * (next_radius - half_radius)
            + right_potential * (half_radius - radius))
            / interval
            + input.potential_coefficients[3] * half_radius_sq;
        let c3_potential = (input.c3_potential[row] * left_weight / left_radius_sq
            + input.c3_potential[next] * right_weight / right_radius_sq)
            * half_radius_sq;
        (potential, c3_potential)
    } else {
        (
            input.potential[row] * left_weight + input.potential[next] * right_weight,
            input.c3_potential[row] * left_weight + input.c3_potential[next] * right_weight,
        )
    };

    let large_exchange =
        input.large_exchange[row] * left_weight + input.large_exchange[next] * right_weight;
    let small_exchange =
        input.small_exchange[row] * left_weight + input.small_exchange[next] * right_weight;
    let f = (energy_over_light - potential) * half_radius;
    let g = f + ccl * half_radius;
    let c3 = fovrg_outward_c3(input.c3_scale, c3_potential, g)?;

    Ok(FovrgOutwardMidpoint {
        f,
        g,
        c3,
        large_exchange,
        small_exchange,
    })
}

fn fovrg_outward_c3(
    c3_scale: i32,
    c3_potential: Complex,
    denominator: Complex,
) -> Result<Complex, FovrgError> {
    if c3_scale == 0 {
        Ok(Complex::new(0.0, 0.0))
    } else {
        validate_nonzero_complex_denominator("outward_c3_denominator", denominator)?;
        Ok((c3_scale as Real) * c3_potential / denominator.powi(2))
    }
}

fn validate_outgoing_solution_input(
    input: &FovrgOutgoingSolutionInput<'_>,
) -> Result<(), FovrgError> {
    validate_count_at_least("active_len", input.active_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.c3_scale != 0 {
        validate_count_at_least("coefficient_count", input.coefficient_count, 3)?;
    }
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len(
        "large_exchange",
        input.active_len,
        input.large_exchange.len(),
    )?;
    validate_active_len(
        "small_exchange",
        input.active_len,
        input.small_exchange.len(),
    )?;
    validate_active_len("c3_potential", input.active_len, input.c3_potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "potential_coefficients",
        input.coefficient_count.max(4),
        input.potential_coefficients.len(),
    )?;
    validate_active_len(
        "large_exchange_coefficients",
        input.coefficient_count.saturating_sub(1),
        input.large_exchange_coefficients.len(),
    )?;
    validate_active_len(
        "small_exchange_coefficients",
        input.coefficient_count.saturating_sub(1),
        input.small_exchange_coefficients.len(),
    )?;
    validate_nonzero_kappa("kappa", 0, input.kappa)?;
    validate_finite("origin_power", input.origin_power)?;
    validate_finite("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_nonzero_finite("step", input.step)?;
    validate_complex_input(
        "initial_large_coefficient",
        0,
        input.initial_large_coefficient,
    )?;
    validate_complex_input(
        "initial_small_coefficient",
        0,
        input.initial_small_coefficient,
    )?;
    validate_complex_input("energy", 0, input.energy)?;

    if input.radial_match_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            active_len: input.radial_match_index + 1,
            len: input.active_len,
        });
    }
    if input.wkb_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "wkb_index",
            active_len: input.wkb_index + 1,
            len: input.active_len,
        });
    }
    if input.last_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "last_index",
            active_len: input.last_index + 1,
            len: input.active_len,
        });
    }
    let flat_start_index = input.radial_match_index.min(input.wkb_index);
    if flat_start_index > input.last_index {
        return Err(FovrgError::InvalidRange {
            name: "outgoing_solution",
            start: flat_start_index,
            end: input.last_index,
        });
    }
    if input.wkb_index < input.last_index && input.wkb_index + 2 >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "wkb_index",
            active_len: input.wkb_index + 3,
            len: input.active_len,
        });
    }

    for row in 0..input.active_len {
        validate_complex_input("potential", row, input.potential[row])?;
        validate_complex_input("large_exchange", row, input.large_exchange[row])?;
        validate_complex_input("small_exchange", row, input.small_exchange[row])?;
        validate_complex_input("c3_potential", row, input.c3_potential[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.potential_coefficients.len() {
        validate_complex_input(
            "potential_coefficients",
            coefficient,
            input.potential_coefficients[coefficient],
        )?;
    }
    for coefficient in 0..input.large_exchange_coefficients.len() {
        validate_complex_input(
            "large_exchange_coefficients",
            coefficient,
            input.large_exchange_coefficients[coefficient],
        )?;
    }
    for coefficient in 0..input.small_exchange_coefficients.len() {
        validate_complex_input(
            "small_exchange_coefficients",
            coefficient,
            input.small_exchange_coefficients[coefficient],
        )?;
    }

    Ok(())
}

fn fovrg_desclaux_origin_series(
    input: FovrgOutgoingSolutionInput<'_>,
    large_coefficients: &mut ComplexVec,
    small_coefficients: &mut ComplexVec,
    energy_over_light: Complex,
) -> Result<(), FovrgError> {
    let ccl = FEFF_ALPHA_INVERSE + FEFF_ALPHA_INVERSE;
    let kappa = input.kappa as Real;
    for coefficient in 1..input.coefficient_count {
        let k = coefficient as Real;
        let a = input.origin_power + kappa + k;
        let b = input.origin_power - kappa + k;
        let denominator = a * b + input.potential_coefficients[0].powi(2);
        validate_nonzero_complex_denominator("solout_desclaux_denominator", denominator)?;
        let mut f = (energy_over_light + ccl) * small_coefficients[coefficient - 1]
            + small_coefficients[coefficient];
        let mut g = energy_over_light * large_coefficients[coefficient - 1]
            + large_coefficients[coefficient];
        for term in 1..=coefficient {
            f -= input.potential_coefficients[term] * small_coefficients[coefficient - term];
            g -= input.potential_coefficients[term] * large_coefficients[coefficient - term];
        }
        large_coefficients[coefficient] =
            (b * f + input.potential_coefficients[0] * g) / denominator;
        small_coefficients[coefficient] =
            (input.potential_coefficients[0] * f - a * g) / denominator;
    }
    Ok(())
}

fn fovrg_relativistic_origin_series(
    input: FovrgOutgoingSolutionInput<'_>,
    large_coefficients: &mut ComplexVec,
    small_coefficients: &mut ComplexVec,
    energy_over_light: Complex,
) -> Result<(), FovrgError> {
    let ccl = FEFF_ALPHA_INVERSE + FEFF_ALPHA_INVERSE;
    let speed_squared = ccl * ccl;
    let two_z = -input.potential_coefficients[0].re * 2.0 * input.speed_of_light;
    let il = if input.kappa > 0 {
        input.kappa + 1
    } else {
        -input.kappa
    } as Real;
    let l0 = il - 1.0;
    large_coefficients[0] = input.initial_large_coefficient;
    if two_z <= 0.0 {
        let denominator = 2.0 * il + 1.0;
        validate_nonzero_denominator("solout_relativistic_il_denominator", denominator)?;
        small_coefficients[0] =
            -energy_over_light / denominator * input.radii[0] * large_coefficients[0];
        large_coefficients[1] = Complex::new(0.0, 0.0);
        small_coefficients[1] = Complex::new(0.0, 0.0);
        large_coefficients[2] = Complex::new(0.0, 0.0);
        small_coefficients[2] = Complex::new(0.0, 0.0);
    } else {
        let rat1 = two_z / ccl;
        let rat2 = rat1 * rat1;
        let rat3 = speed_squared / two_z;
        validate_nonzero_denominator(
            "solout_relativistic_fl2_denominator",
            2.0 * input.origin_power + 1.0,
        )?;
        validate_nonzero_denominator(
            "solout_relativistic_fl1_denominator",
            input.origin_power + 1.0,
        )?;
        small_coefficients[0] = (input.origin_power - il) * rat3 * large_coefficients[0];
        large_coefficients[1] = (3.0 * input.origin_power - rat2)
            / (2.0 * input.origin_power + 1.0)
            * large_coefficients[0];
        small_coefficients[1] = rat3
            * ((input.origin_power - l0) * large_coefficients[1] - large_coefficients[0])
            - small_coefficients[0];
        large_coefficients[2] = ((input.origin_power + 3.0 * il) * large_coefficients[1]
            - 3.0 * l0 * large_coefficients[0]
            + (input.origin_power + il + 3.0) / rat3 * small_coefficients[1])
            / (input.origin_power + 1.0)
            / 4.0;
        small_coefficients[2] = (rat3
            * (2.0 * l0 * (input.origin_power + 2.0 - il) - l0 - rat2)
            * large_coefficients[1]
            - 3.0 * l0 * rat3 * (input.origin_power + 2.0 - il) * large_coefficients[0]
            + (input.origin_power + 3.0 - 2.0 * il - rat2) * small_coefficients[1])
            / (input.origin_power + 1.0)
            / 4.0;
        small_coefficients[0] /= ccl;
        large_coefficients[1] *= rat3;
        small_coefficients[1] *= rat3 / ccl;
        large_coefficients[2] *= rat3 * rat3;
        small_coefficients[2] *= rat3 * rat3 / ccl;
    }
    Ok(())
}

fn fovrg_origin_components(
    input: FovrgOutgoingSolutionInput<'_>,
    large_coefficients: ArrayView1<'_, Complex>,
    small_coefficients: ArrayView1<'_, Complex>,
) -> (Complex, Complex) {
    if input.c3_scale == 0 {
        (0..input.coefficient_count).fold(
            (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
            |(large_sum, small_sum), coefficient| {
                let power = input.origin_power + coefficient as Real;
                let radius_power = input.radii[0].powf(power);
                (
                    large_sum + radius_power * large_coefficients[coefficient],
                    small_sum + radius_power * small_coefficients[coefficient],
                )
            },
        )
    } else {
        let radius = input.radii[0];
        let radius_power = radius.powf(input.origin_power);
        (
            radius_power
                * (large_coefficients[0]
                    + radius * (large_coefficients[1] + radius * large_coefficients[2])),
            radius_power
                * (small_coefficients[0]
                    + radius * (small_coefficients[1] + radius * small_coefficients[2])),
        )
    }
}

fn fovrg_solout_average_potential(
    input: FovrgOutgoingSolutionInput<'_>,
    row: usize,
) -> Result<Complex, FovrgError> {
    let mut average_potential = if row == input.wkb_index {
        let extrapolated = input.speed_of_light
            * (3.0 * input.potential[input.wkb_index + 1] - input.potential[input.wkb_index + 2])
            / 2.0;
        if input.wkb_index + 1 == input.radial_match_index {
            input.speed_of_light * (input.potential[row] + input.potential[row + 1]) / 2.0
        } else {
            extrapolated
        }
    } else {
        input.speed_of_light * (input.potential[row] + input.potential[row + 1]) / 2.0
    };

    if input.c3_scale > 0 && row < input.radial_match_index {
        let radius_average = (input.radii[row] + input.radii[row + 1]) / 2.0;
        let relativistic = FEFF_ALPHA_INVERSE
            + FEFF_ALPHA_INVERSE
            + (input.energy - average_potential) / input.speed_of_light;
        let denominator = radius_average.powi(3) * relativistic.powi(2);
        validate_nonzero_complex_denominator("solout_c3_flat_denominator", denominator)?;
        average_potential += (input.c3_scale as Real) * input.speed_of_light / denominator
            * (input.c3_potential[row] + input.c3_potential[row + 1])
            / 2.0;
    }

    Ok(average_potential)
}

fn validate_inward_solution_input(input: &FovrgInwardSolutionInput<'_>) -> Result<(), FovrgError> {
    validate_count_at_least("active_len", input.active_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len(
        "large_exchange",
        input.active_len,
        input.large_exchange.len(),
    )?;
    validate_active_len(
        "small_exchange",
        input.active_len,
        input.small_exchange.len(),
    )?;
    validate_active_len("c3_potential", input.active_len, input.c3_potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_nonzero_kappa("kappa", 0, input.kappa)?;
    validate_finite("origin_power", input.origin_power)?;
    validate_finite("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_nonzero_finite("step", input.step)?;
    validate_complex_input(
        "initial_large_coefficient",
        0,
        input.initial_large_coefficient,
    )?;
    validate_complex_input(
        "initial_small_coefficient",
        0,
        input.initial_small_coefficient,
    )?;
    validate_complex_input("energy", 0, input.energy)?;

    if input.radial_match_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            active_len: input.radial_match_index + 1,
            len: input.active_len,
        });
    }
    if input.radial_match_index + 1 >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            active_len: input.radial_match_index + 2,
            len: input.active_len,
        });
    }
    if input.wkb_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "wkb_index",
            active_len: input.wkb_index + 1,
            len: input.active_len,
        });
    }
    if input.last_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "last_index",
            active_len: input.last_index + 1,
            len: input.active_len,
        });
    }
    if input.radial_match_index > input.last_index {
        return Err(FovrgError::InvalidRange {
            name: "inward_solution",
            start: input.radial_match_index,
            end: input.last_index,
        });
    }
    if input.wkb_index < input.radial_match_index && input.wkb_index + 2 >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "wkb_index",
            active_len: input.wkb_index + 3,
            len: input.active_len,
        });
    }
    let flat_start_index = input.radial_match_index.min(input.wkb_index);
    if flat_start_index > 0 {
        let history_rows = input.last_index - flat_start_index + 1;
        validate_count_at_least("inward_history_rows", history_rows, FOVRG_INT_OUT_HISTORY)?;
    }

    for row in 0..input.active_len {
        validate_complex_input("potential", row, input.potential[row])?;
        validate_complex_input("large_exchange", row, input.large_exchange[row])?;
        validate_complex_input("small_exchange", row, input.small_exchange[row])?;
        validate_complex_input("c3_potential", row, input.c3_potential[row])?;
        validate_radius(row, input.radii[row])?;
    }

    Ok(())
}

fn fovrg_solin_average_potential(
    input: FovrgInwardSolutionInput<'_>,
    row: usize,
) -> Result<Complex, FovrgError> {
    let average_potential = if row == input.wkb_index {
        let extrapolated = input.speed_of_light
            * (3.0 * input.potential[input.wkb_index + 1] - input.potential[input.wkb_index + 2])
            / 2.0;
        if input.wkb_index + 1 == input.radial_match_index {
            input.speed_of_light * (input.potential[row] + input.potential[row + 1]) / 2.0
        } else {
            extrapolated
        }
    } else {
        input.speed_of_light * (input.potential[row] + input.potential[row + 1]) / 2.0
    };
    validate_complex_result("solin_average_potential", row, average_potential)?;
    Ok(average_potential)
}

fn fovrg_inward_history_slot(flat_start_index: usize, row: usize) -> Option<usize> {
    let slot = flat_start_index + FOVRG_INT_OUT_HISTORY - 1;
    (row <= slot).then_some(slot - row)
}

fn fovrg_inward_derivatives(
    input: &FovrgInwardSolutionInput<'_>,
    row: usize,
    energy_term: Complex,
    ccl: Real,
    include_exchange: bool,
    large_component: Complex,
    small_component: Complex,
) -> Result<(Complex, Complex), FovrgError> {
    let f = (energy_term - input.potential[row]) * input.radii[row];
    let g = f + ccl * input.radii[row];
    let c3 = fovrg_outward_c3(input.c3_scale, input.c3_potential[row], g)?;
    let (large_exchange, small_exchange) = if include_exchange {
        (input.large_exchange[row], input.small_exchange[row])
    } else {
        (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0))
    };
    let kappa = input.kappa as Real;
    let large_derivative = -(g * small_component - kappa * large_component + small_exchange);
    let small_derivative = -(kappa * small_component - (f - c3) * large_component - large_exchange);
    validate_complex_result("inward_large_derivative", row, large_derivative)?;
    validate_complex_result("inward_small_derivative", row, small_derivative)?;
    Ok((large_derivative, small_derivative))
}

fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), FovrgError> {
    if actual < minimum {
        Err(FovrgError::CountTooSmall {
            name,
            actual,
            minimum,
        })
    } else {
        Ok(())
    }
}

fn target_j_value(kappa: i32) -> usize {
    2 * kappa.unsigned_abs() as usize - 1
}

fn exchange_coefficient_start(
    multipole: usize,
    bound_kappa: i32,
    target_kappa: i32,
    target_power: Real,
) -> Option<usize> {
    let bound_abs = i64::from(bound_kappa.unsigned_abs());
    let target_abs = i64::from(target_kappa.unsigned_abs());
    let multipole = multipole as i64;
    let start = if target_power < 0.0 {
        multipole + 1 + bound_abs + target_abs
    } else {
        multipole + 1 + bound_abs - target_abs
    };
    (start >= 1).then_some(start as usize)
}

fn validate_active_len(
    field: &'static str,
    active_len: usize,
    len: usize,
) -> Result<(), FovrgError> {
    if active_len > len {
        Err(FovrgError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_matrix_rows(
    field: &'static str,
    required: usize,
    available: usize,
) -> Result<(), FovrgError> {
    if required > available {
        Err(FovrgError::ActiveCountOutOfRange {
            field,
            active_len: required,
            len: available,
        })
    } else {
        Ok(())
    }
}

fn validate_matrix_cols(
    field: &'static str,
    required: usize,
    available: usize,
) -> Result<(), FovrgError> {
    if required > available {
        Err(FovrgError::ActiveCountOutOfRange {
            field,
            active_len: required,
            len: available,
        })
    } else {
        Ok(())
    }
}

fn fovrg_usize_to_i32(name: &'static str, value: usize) -> Result<i32, FovrgError> {
    i32::try_from(value).map_err(|_| FovrgError::CountTooLarge {
        name,
        actual: value,
        maximum: i32::MAX as usize,
    })
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteInput { name, value })
    }
}

fn validate_nonzero_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput { name, value })
    } else if value == 0.0 {
        Err(FovrgError::ZeroInput { name })
    } else {
        Ok(())
    }
}

fn validate_positive_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput { name, value })
    } else if value <= 0.0 {
        Err(FovrgError::NonPositiveInput { name, value })
    } else {
        Ok(())
    }
}

fn validate_nonzero_denominator(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if value == 0.0 {
        Err(FovrgError::ZeroDenominator { name })
    } else {
        Ok(())
    }
}

fn validate_nonzero_complex_denominator(
    name: &'static str,
    value: Complex,
) -> Result<(), FovrgError> {
    if value == Complex::new(0.0, 0.0) {
        Err(FovrgError::ZeroDenominator { name })
    } else {
        Ok(())
    }
}

fn validate_radius(row: usize, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput {
            name: "radius",
            value,
        })
    } else if value <= 0.0 {
        Err(FovrgError::NonPositiveRadius { row, value })
    } else {
        Ok(())
    }
}

fn validate_nonzero_kappa(name: &'static str, row: usize, value: i32) -> Result<(), FovrgError> {
    if value == 0 {
        Err(FovrgError::InvalidQuantumNumber { name, row, value })
    } else {
        Ok(())
    }
}

fn validate_real_input(name: &'static str, row: usize, value: Real) -> Result<(), FovrgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteRealInput { name, row, value })
    }
}

fn validate_real_result(name: &'static str, row: usize, value: Real) -> Result<(), FovrgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteRealInput { name, row, value })
    }
}

fn validate_complex_input(
    name: &'static str,
    row: usize,
    value: Complex,
) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteComplexInput { name, row, value })
    }
}

fn validate_potential(row: usize, value: Complex) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFinitePotential { row, value })
    }
}

fn complex_real_product_coefficient(
    complex_coefficients: ArrayView1<'_, Complex>,
    real_coefficients: ArrayView1<'_, Real>,
    count: usize,
) -> Complex {
    (0..count).fold(Complex::new(0.0, 0.0), |sum, index| {
        sum + complex_coefficients[index] * real_coefficients[count - 1 - index]
    })
}

fn real_product_coefficient(
    left_coefficients: ArrayView1<'_, Real>,
    right_coefficients: ArrayView1<'_, Real>,
    count: usize,
) -> Real {
    (0..count).fold(0.0, |sum, index| {
        sum + left_coefficients[index] * right_coefficients[count - 1 - index]
    })
}

fn validate_complex_result(
    name: &'static str,
    row: usize,
    value: Complex,
) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteComplexInput { name, row, value })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use crate::{Complex, Real};

    use super::{
        FovrgAngularCoefficientsInput, FovrgC3DerivativeInput, FovrgError,
        FovrgExchangePotentialInput, FovrgFlatPotentialInput, FovrgInwardSolutionInput,
        FovrgNuclearPotentialInput, FovrgOrthogonalizationInput, FovrgOutgoingSolutionInput,
        FovrgOutwardIntegrationInput, FovrgOverlapIntegralInput, FovrgPotentialDevelopmentInput,
        FovrgYkZkExchangeInput, FovrgYkZkTransformInput, fovrg_angular_coefficients,
        fovrg_c3_derivative, fovrg_complex_real_product_coefficient, fovrg_exchange_potential,
        fovrg_flat_potential_propagate, fovrg_inward_solution, fovrg_nuclear_potential,
        fovrg_outgoing_solution, fovrg_outward_integrate, fovrg_overlap_integral,
        fovrg_potential_development, fovrg_real_product_coefficient, fovrg_schmidt_orthogonalize,
        fovrg_yk_zk_exchange, fovrg_yk_zk_transform,
    };

    #[test]
    fn c3_derivative_matches_feff_diff_reference() -> Result<(), FovrgError> {
        let (potential, radii) = diff_reference_inputs(10);

        let derivative = fovrg_c3_derivative(FovrgC3DerivativeInput {
            potential: potential.view(),
            radii: radii.view(),
            kappa: -2,
            speed_of_light: 137.035_999_084,
            delta: 0.0375,
            active_len: 10,
        })?;

        let expected = [
            (-0.011_975_827_006_405_27, -0.011_279_195_671_167_455),
            (-0.016_505_394_195_758_99, -0.008_884_114_730_822_418),
            (-0.020_242_542_448_345_43, -0.005_647_908_958_998_54),
            (-0.022_839_291_155_546_27, -0.001_659_964_058_354_706_8),
            (-0.024_047_315_082_090_202, 0.002_950_607_669_371_263_3),
            (-0.023_683_648_659_231_31, 0.008_014_885_042_325_136),
            (-0.021_663_526_338_827_583, 0.013_330_188_602_550_464),
            (-0.018_012_853_921_219_218, 0.018_667_556_473_840_063),
            (-0.012_457_714_462_626_513, 0.023_984_332_127_499_31),
            (-0.007_300_598_102_380_937, 0.028_056_048_903_698_883),
        ];
        for (actual, (expected_re, expected_im)) in derivative.iter().zip(expected) {
            assert_complex_close(*actual, expected_re, expected_im, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn c3_derivative_rejects_invalid_inputs() {
        let (potential, radii) = diff_reference_inputs(8);

        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 7,
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 9,
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "potential",
                ..
            })
        ));
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0,
                active_len: 8,
            }),
            Err(FovrgError::ZeroInput { name: "delta" })
        ));

        let mut bad_radii = radii.clone();
        bad_radii[3] = 0.0;
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: bad_radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 8,
            }),
            Err(FovrgError::NonPositiveRadius { row: 3, .. })
        ));

        let mut bad_potential = potential.clone();
        bad_potential[2] = Complex::new(f64::NAN, 0.0);
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: bad_potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 8,
            }),
            Err(FovrgError::NonFinitePotential { row: 2, .. })
        ));
    }

    #[test]
    fn polynomial_product_coefficients_match_feff_aprd_reference() -> Result<(), FovrgError> {
        let (real_left, real_right, complex_left) = aprd_reference_inputs(10);

        assert_close(
            fovrg_real_product_coefficient(real_left.view(), real_right.view(), 4)?,
            0.611_437_708_836_968_1,
            1.0e-14,
        );
        assert_close(
            fovrg_real_product_coefficient(real_left.view(), real_right.view(), 7)?,
            1.688_549_807_000_237_2,
            1.0e-14,
        );
        assert_complex_close(
            fovrg_complex_real_product_coefficient(complex_left.view(), real_right.view(), 4)?,
            0.615_721_272_049_818_1,
            0.159_539_410_440_073_47,
            1.0e-14,
        );
        assert_complex_close(
            fovrg_complex_real_product_coefficient(complex_left.view(), real_right.view(), 7)?,
            1.660_658_325_254_387,
            0.615_717_443_886_918,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn polynomial_product_coefficients_reject_invalid_inputs() {
        let (real_left, real_right, complex_left) = aprd_reference_inputs(10);

        assert!(matches!(
            fovrg_real_product_coefficient(real_left.view(), real_right.view(), 0),
            Err(FovrgError::CountTooSmall {
                name: "coefficient_count",
                ..
            })
        ));
        assert!(matches!(
            fovrg_real_product_coefficient(real_left.view(), real_right.view(), 11),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "left_coefficients",
                ..
            })
        ));
        assert!(matches!(
            fovrg_complex_real_product_coefficient(complex_left.view(), real_right.view(), 11),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "complex_coefficients",
                ..
            })
        ));

        let mut bad_real_right = real_right.clone();
        bad_real_right[2] = Real::NAN;
        assert!(matches!(
            fovrg_real_product_coefficient(real_left.view(), bad_real_right.view(), 4),
            Err(FovrgError::NonFiniteRealInput {
                name: "right_coefficients",
                row: 2,
                ..
            })
        ));

        let mut bad_complex_left = complex_left.clone();
        bad_complex_left[1] = Complex::new(0.0, Real::NAN);
        assert!(matches!(
            fovrg_complex_real_product_coefficient(bad_complex_left.view(), real_right.view(), 4),
            Err(FovrgError::NonFiniteComplexInput {
                name: "complex_coefficients",
                row: 1,
                ..
            })
        ));
    }

    #[test]
    fn angular_coefficients_match_feff_muatcc_reference() -> Result<(), FovrgError> {
        let (electron_counts, valence_counts, kappa) = muatcc_reference_inputs();

        let target_negative = fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
            electron_counts: electron_counts.view(),
            valence_counts: valence_counts.view(),
            kappa: kappa.view(),
            target_kappa: -2,
            bound_orbital_count: 5,
        })?;
        let expected_negative = [
            [0.333_333_333_333_333_54, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.625_000_000_000_000_2,
                0.125_000_000_000_000_08,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.016_666_666_666_666_684,
                0.064_285_714_285_714_21,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.299_999_999_999_999_9,
                0.085_714_285_714_285_62,
                0.0,
                0.0,
                0.0,
            ],
        ];
        assert_real_matrix_close(&target_negative, &expected_negative, 1.0e-14);

        let target_positive = fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
            electron_counts: electron_counts.view(),
            valence_counts: valence_counts.view(),
            kappa: kappa.view(),
            target_kappa: 3,
            bound_orbital_count: 5,
        })?;
        let expected_positive = [
            [0.142_857_142_857_142_74, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.035_714_285_714_285_67,
                0.119_047_619_047_618_57,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.099_999_999_999_999_94,
                0.028_571_428_571_428_54,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.014_285_714_285_714_28,
                0.038_095_238_095_237_96,
                0.108_225_108_225_107_97,
                0.0,
                0.0,
            ],
        ];
        assert_real_matrix_close(&target_positive, &expected_positive, 1.0e-14);

        Ok(())
    }

    #[test]
    fn angular_coefficients_reject_invalid_inputs() {
        let (electron_counts, valence_counts, kappa) = muatcc_reference_inputs();

        assert!(matches!(
            fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
                electron_counts: electron_counts.view(),
                valence_counts: valence_counts.view(),
                kappa: kappa.view(),
                target_kappa: -2,
                bound_orbital_count: 0,
            }),
            Err(FovrgError::CountTooSmall {
                name: "bound_orbital_count",
                ..
            })
        ));
        assert!(matches!(
            fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
                electron_counts: electron_counts.view(),
                valence_counts: valence_counts.view(),
                kappa: kappa.view(),
                target_kappa: 0,
                bound_orbital_count: 5,
            }),
            Err(FovrgError::InvalidQuantumNumber {
                name: "target_kappa",
                row: 0,
                ..
            })
        ));

        let mut bad_kappa = kappa.clone();
        bad_kappa[1] = 0;
        assert!(matches!(
            fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
                electron_counts: electron_counts.view(),
                valence_counts: valence_counts.view(),
                kappa: bad_kappa.view(),
                target_kappa: -2,
                bound_orbital_count: 5,
            }),
            Err(FovrgError::InvalidQuantumNumber {
                name: "kappa",
                row: 1,
                ..
            })
        ));

        let mut bad_electron_counts = electron_counts.clone();
        bad_electron_counts[3] = Real::NAN;
        assert!(matches!(
            fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
                electron_counts: bad_electron_counts.view(),
                valence_counts: valence_counts.view(),
                kappa: kappa.view(),
                target_kappa: -2,
                bound_orbital_count: 5,
            }),
            Err(FovrgError::NonFiniteRealInput {
                name: "electron_counts",
                row: 3,
                ..
            })
        ));

        let mut bad_valence_counts = valence_counts.clone();
        bad_valence_counts[2] = Real::NAN;
        assert!(matches!(
            fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
                electron_counts: electron_counts.view(),
                valence_counts: bad_valence_counts.view(),
                kappa: kappa.view(),
                target_kappa: -2,
                bound_orbital_count: 5,
            }),
            Err(FovrgError::NonFiniteRealInput {
                name: "valence_counts",
                row: 2,
                ..
            })
        ));

        assert!(matches!(
            fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
                electron_counts: Array1::from_vec(vec![1.0]).view(),
                valence_counts: Array1::from_vec(vec![0.0]).view(),
                kappa: Array1::from_vec(vec![-6]).view(),
                target_kappa: -6,
                bound_orbital_count: 1,
            }),
            Err(FovrgError::CountTooLarge {
                name: "angular_coefficient_slots",
                actual: 6,
                maximum: 5,
            })
        ));
    }

    #[test]
    fn flat_potential_propagation_matches_feff_flatv_reference() -> Result<(), FovrgError> {
        let propagated = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            start_radius: 0.8,
            end_radius: 1.35,
            large_component: Complex::new(0.32, -0.11),
            small_component: Complex::new(-0.08, 0.045),
            energy: Complex::new(0.85, 0.12),
            average_potential: Complex::new(-0.18, 0.025),
            kappa: -2,
        })?;
        assert_complex_close(
            propagated.large_component,
            -11.083_037_894_089_62,
            6.535_303_549_398_971_5,
            1.0e-12,
        );
        assert_complex_close(
            propagated.small_component,
            -0.009_973_201_918_406_406,
            0.007_263_491_015_424_047,
            1.0e-12,
        );

        let propagated = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            start_radius: 1.2,
            end_radius: 0.9,
            large_component: Complex::new(-0.14, 0.27),
            small_component: Complex::new(0.19, -0.06),
            energy: Complex::new(1.6, -0.05),
            average_potential: Complex::new(0.2, 0.01),
            kappa: 3,
        })?;
        assert_complex_close(
            propagated.large_component,
            -17.939_760_805_034_215,
            6.125_209_917_887_357,
            1.0e-12,
        );
        assert_complex_close(
            propagated.small_component,
            0.060_863_298_623_451_69,
            -0.017_588_891_061_652_855,
            1.0e-12,
        );
        Ok(())
    }

    #[test]
    fn flat_potential_propagation_rejects_invalid_inputs() {
        let input = FovrgFlatPotentialInput {
            start_radius: 0.8,
            end_radius: 1.35,
            large_component: Complex::new(0.32, -0.11),
            small_component: Complex::new(-0.08, 0.045),
            energy: Complex::new(0.85, 0.12),
            average_potential: Complex::new(-0.18, 0.025),
            kappa: -2,
        };

        assert!(matches!(
            fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
                start_radius: 0.0,
                ..input
            }),
            Err(FovrgError::NonPositiveInput {
                name: "start_radius",
                ..
            })
        ));
        assert!(matches!(
            fovrg_flat_potential_propagate(FovrgFlatPotentialInput { kappa: 0, ..input }),
            Err(FovrgError::InvalidQuantumNumber {
                name: "kappa",
                row: 0,
                ..
            })
        ));
        assert!(matches!(
            fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
                energy: Complex::new(Real::NAN, 0.0),
                ..input
            }),
            Err(FovrgError::NonFiniteComplexInput {
                name: "energy",
                row: 0,
                ..
            })
        ));
        assert!(matches!(
            fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
                energy: Complex::new(0.85, 0.12),
                average_potential: Complex::new(0.85, 0.12),
                ..input
            }),
            Err(FovrgError::ZeroDenominator {
                name: "flat_potential_factor"
            })
        ));
    }

    #[test]
    fn outward_integration_matches_feff_intout_reference() -> Result<(), FovrgError> {
        let tolerance = 2.0e-5;
        let case1 = intout_reference_inputs(1);
        let integrated = fovrg_outward_integrate(case1.to_input())?;
        assert_eq!(integrated.difficult_iterations, 0);
        assert_complex_close(
            integrated.large_component[1],
            0.017_463_805_053_776_79,
            -0.003_797_490_517_828_303_4,
            tolerance,
        );
        assert_complex_close(
            integrated.small_component[1],
            -0.008_406_932_541_754_055,
            0.003_782_923_630_515_854,
            tolerance,
        );
        assert_complex_close(
            integrated.large_component[5],
            -0.066_689_204_945_806_59,
            0.035_296_711_329_772_11,
            tolerance,
        );
        assert_complex_close(
            integrated.small_component[5],
            -0.006_557_738_302_850_965,
            0.002_754_941_000_224_632,
            tolerance,
        );
        assert_complex_close(
            integrated.large_component[11],
            -0.233_694_640_455_667_15,
            0.106_985_257_008_756_33,
            tolerance,
        );
        assert_complex_close(
            integrated.small_component[11],
            -0.002_738_717_599_426_290_6,
            0.000_956_279_014_355_418_1,
            tolerance,
        );
        assert_complex_close(integrated.large_component[12], 0.0, 0.0, 0.0);
        assert_complex_close(integrated.small_component[12], 0.0, 0.0, 0.0);

        let case2 = intout_reference_inputs(2);
        let integrated = fovrg_outward_integrate(case2.to_input())?;
        assert_eq!(integrated.difficult_iterations, 0);
        assert_complex_close(
            integrated.large_component[1],
            0.009_943_048_888_859_825,
            0.010_956_870_736_304_185,
            tolerance,
        );
        assert_complex_close(
            integrated.small_component[1],
            0.012_429_931_740_178_783,
            -0.006_764_974_141_679_289,
            tolerance,
        );
        assert_complex_close(
            integrated.large_component[12],
            0.586_154_840_562_188_5,
            -0.333_526_150_681_263_4,
            tolerance,
        );
        assert_complex_close(
            integrated.small_component[12],
            0.049_643_797_589_848_5,
            -0.028_715_924_595_140_7,
            tolerance,
        );
        assert_complex_close(integrated.large_component[13], 0.0, 0.0, 0.0);
        assert_complex_close(integrated.small_component[13], 0.0, 0.0, 0.0);

        let case3 = intout_reference_inputs(3);
        let integrated = fovrg_outward_integrate(case3.to_input())?;
        assert_eq!(integrated.difficult_iterations, 0);
        assert_complex_close(integrated.large_component[0], 0.0, 0.0, 0.0);
        assert_complex_close(integrated.small_component[0], 0.0, 0.0, 0.0);
        assert_complex_close(integrated.large_component[3], 0.026, 0.014, 1.0e-15);
        assert_complex_close(integrated.small_component[3], -0.008, 0.017, 1.0e-15);
        assert_complex_close(
            integrated.large_component[4],
            0.005_997_786_087_037_459,
            0.058_938_915_396_690_67,
            tolerance,
        );
        assert_complex_close(
            integrated.small_component[4],
            -0.007_911_306_018_542_903,
            0.016_250_490_408_196_89,
            tolerance,
        );
        assert_complex_close(
            integrated.large_component[10],
            -0.136_499_070_289_096_64,
            0.363_725_975_734_245_37,
            tolerance,
        );
        assert_complex_close(
            integrated.small_component[10],
            -0.005_440_281_025_366_89,
            0.011_164_144_227_578_115,
            tolerance,
        );
        assert_complex_close(integrated.large_component[11], 0.0, 0.0, 0.0);
        assert_complex_close(integrated.small_component[11], 0.0, 0.0, 0.0);

        Ok(())
    }

    #[test]
    fn outward_integration_rejects_invalid_inputs() {
        let mut input = intout_reference_inputs(1);

        assert!(matches!(
            fovrg_outward_integrate(FovrgOutwardIntegrationInput {
                active_len: 0,
                ..input.to_input()
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));

        assert!(matches!(
            fovrg_outward_integrate(FovrgOutwardIntegrationInput {
                start_index: 5,
                last_index: 4,
                ..input.to_input()
            }),
            Err(FovrgError::InvalidRange {
                name: "outward_integration",
                ..
            })
        ));

        assert!(matches!(
            fovrg_outward_integrate(FovrgOutwardIntegrationInput {
                kappa: 0,
                ..input.to_input()
            }),
            Err(FovrgError::InvalidQuantumNumber { name: "kappa", .. })
        ));

        assert!(matches!(
            fovrg_outward_integrate(FovrgOutwardIntegrationInput {
                step: 0.0,
                ..input.to_input()
            }),
            Err(FovrgError::ZeroInput { name: "step" })
        ));

        input.potential[2] = Complex::new(Real::NAN, 0.0);
        assert!(matches!(
            fovrg_outward_integrate(input.to_input()),
            Err(FovrgError::NonFiniteComplexInput {
                name: "potential",
                row: 2,
                ..
            })
        ));
    }

    #[test]
    fn outgoing_solution_matches_feff_solout_reference() -> Result<(), FovrgError> {
        let tolerance = 5.0e-5;

        let case1 = solout_reference_inputs(1);
        let solution = fovrg_outgoing_solution(case1.to_input())?;
        assert_eq!(solution.difficult_iterations, 0);
        assert_complex_close(solution.large_coefficients[0], 0.85, -0.13, 1.0e-14);
        assert_complex_close(
            solution.small_coefficients[0],
            -0.044_826_720_241_084_875,
            0.006_855_851_330_989_452,
            1.0e-13,
        );
        assert_complex_close(
            solution.large_coefficients[1],
            -12.399_643_967_721_534,
            1.897_735_483_431_073_6,
            tolerance,
        );
        assert_complex_close(
            solution.small_coefficients[5],
            18.506_377_176_380_628,
            -3.169_720_164_560_293,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[6],
            -0.013_570_707_559_336_159,
            0.004_971_932_036_426_257,
            tolerance,
        );
        assert_complex_close(
            solution.small_component[6],
            -0.000_934_704_975_260_855_8,
            0.000_196_765_336_633_890_28,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[11],
            -0.036_169_233_625_015_3,
            0.010_896_837_643_968_734,
            tolerance,
        );
        assert_complex_close(
            solution.small_component[11],
            -0.000_538_567_588_404_036_5,
            0.000_111_061_211_719_815_06,
            tolerance,
        );
        assert_complex_close(solution.large_component[12], 0.0, 0.0, 0.0);

        let case2 = solout_reference_inputs(2);
        let solution = fovrg_outgoing_solution(case2.to_input())?;
        assert_complex_close(solution.large_coefficients[0], -0.72, 0.21, 1.0e-14);
        assert_complex_close(
            solution.small_coefficients[0],
            0.000_037_001_955_937_810_44,
            -0.000_013_870_807_763_694_648,
            tolerance,
        );
        assert_complex_close(
            solution.large_coefficients[3],
            0.008_048_689_860_581_586,
            -0.004_087_598_669_440_412,
            1.0e-14,
        );
        assert_complex_close(
            solution.large_component[12],
            -0.046_463_364_356_391_396,
            0.013_612_530_210_438_293,
            tolerance,
        );
        assert_complex_close(
            solution.small_component[12],
            -0.003_952_067_939_391_788,
            0.001_101_615_849_936_607_4,
            tolerance,
        );
        assert_complex_close(solution.large_component[13], 0.0, 0.0, 0.0);

        let case3 = solout_reference_inputs(3);
        let solution = fovrg_outgoing_solution(case3.to_input())?;
        assert_complex_close(solution.large_coefficients[0], 0.64, 0.08, 1.0e-14);
        assert_complex_close(
            solution.small_coefficients[0],
            -0.124_444_435_795_557_43,
            -0.015_555_554_474_444_679,
            tolerance,
        );
        assert_complex_close(
            solution.large_coefficients[2],
            710_464.572_458_431_2,
            88_808.071_557_303_9,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[0],
            4_432.600_877_072_657,
            554.075_109_634_082_1,
            tolerance,
        );
        assert_complex_close(
            solution.small_component[0],
            -488.363_646_550_350_4,
            -61.045_455_818_793_8,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[10],
            -6_977.844_764_917_198,
            -863.171_547_536_417_2,
            tolerance,
        );
        assert_complex_close(solution.large_component[11], 0.0, 0.0, 0.0);

        Ok(())
    }

    #[test]
    fn outgoing_solution_rejects_invalid_inputs() {
        let mut input = solout_reference_inputs(1);

        assert!(matches!(
            fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
                active_len: 0,
                ..input.to_input()
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));

        assert!(matches!(
            fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
                coefficient_count: 0,
                ..input.to_input()
            }),
            Err(FovrgError::CountTooSmall {
                name: "coefficient_count",
                ..
            })
        ));

        assert!(matches!(
            fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
                radial_match_index: 14,
                wkb_index: 13,
                last_index: 12,
                ..input.to_input()
            }),
            Err(FovrgError::InvalidRange {
                name: "outgoing_solution",
                ..
            })
        ));

        let case2 = solout_reference_inputs(2);
        assert!(matches!(
            fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
                coefficient_count: 2,
                ..case2.to_input()
            }),
            Err(FovrgError::CountTooSmall {
                name: "coefficient_count",
                ..
            })
        ));

        input.large_exchange_coefficients[1] = Complex::new(Real::NAN, 0.0);
        assert!(matches!(
            fovrg_outgoing_solution(input.to_input()),
            Err(FovrgError::NonFiniteComplexInput {
                name: "large_exchange_coefficients",
                row: 1,
                ..
            })
        ));
    }

    #[test]
    fn inward_solution_matches_feff_solin_reference() -> Result<(), FovrgError> {
        let tolerance = 5.0e-5;

        let case1 = solin_reference_inputs(1);
        let solution = fovrg_inward_solution(case1.to_input())?;
        assert_eq!(solution.difficult_iterations, 0);
        assert_complex_close(
            solution.large_coefficients[0],
            13.035_518_197_636_561,
            0.349_850_489_417_380_23,
            tolerance,
        );
        assert_complex_close(
            solution.small_coefficients[0],
            -0.628_236_070_374_292_4,
            0.041_526_389_085_050_74,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[0],
            0.435_590_505_925_380_57,
            0.011_690_486_666_743_206,
            tolerance,
        );
        assert_complex_close(
            solution.small_component[0],
            -0.020_992_925_911_033_328,
            0.001_387_631_895_914_275,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[11],
            0.333_382_471_777_223,
            0.079_399_312_578_217_63,
            tolerance,
        );
        assert_complex_close(solution.large_component[12], 0.0, 0.0, 0.0);
        assert_complex_close(solution.large_coefficients[1], 0.0, 0.0, 0.0);

        let case2 = solin_reference_inputs(2);
        let solution = fovrg_inward_solution(case2.to_input())?;
        assert_complex_close(
            solution.large_coefficients[0],
            3_881.336_079_768_998_4,
            -425.269_741_140_675_76,
            tolerance,
        );
        assert_complex_close(
            solution.small_coefficients[0],
            17.080_439_990_261_297,
            -1.894_813_979_341_339,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[0],
            21.686_056_118_456_456,
            -2.376_095_056_526_779,
            tolerance,
        );
        assert_complex_close(
            solution.small_component[0],
            0.095_432_957_245_686_26,
            -0.010_586_829_237_543_801,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[9],
            6.955_715_550_949_966,
            -0.740_042_518_825_175_1,
            tolerance,
        );
        assert_complex_close(solution.large_component[13], 0.0, 0.0, 0.0);

        let case3 = solin_reference_inputs(3);
        let solution = fovrg_inward_solution(case3.to_input())?;
        assert_complex_close(
            solution.large_coefficients[0],
            1.010_225_566_356_747_7,
            0.771_472_351_197_525_2,
            tolerance,
        );
        assert_complex_close(
            solution.small_coefficients[0],
            -0.021_374_479_295_957_21,
            0.001_988_824_772_918_297,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[0],
            0.193_088_461_728_290_2,
            0.147_454_602_733_775_3,
            tolerance,
        );
        assert_complex_close(
            solution.small_component[7],
            -0.005_174_328_864_862_752,
            -0.000_796_579_407_773_487_3,
            tolerance,
        );
        assert_complex_close(
            solution.large_component[12],
            0.109_417_857_489_330_1,
            0.230_010_707_501_99,
            tolerance,
        );
        assert_complex_close(solution.large_component[13], 0.0, 0.0, 0.0);

        Ok(())
    }

    #[test]
    fn inward_solution_rejects_invalid_inputs() {
        let mut input = solin_reference_inputs(1);

        assert!(matches!(
            fovrg_inward_solution(FovrgInwardSolutionInput {
                active_len: 0,
                ..input.to_input()
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));

        assert!(matches!(
            fovrg_inward_solution(FovrgInwardSolutionInput {
                radial_match_index: 12,
                last_index: 11,
                ..input.to_input()
            }),
            Err(FovrgError::InvalidRange {
                name: "inward_solution",
                ..
            })
        ));

        assert!(matches!(
            fovrg_inward_solution(FovrgInwardSolutionInput {
                last_index: 8,
                ..input.to_input()
            }),
            Err(FovrgError::CountTooSmall {
                name: "inward_history_rows",
                ..
            })
        ));

        assert!(matches!(
            fovrg_inward_solution(FovrgInwardSolutionInput {
                kappa: 0,
                ..input.to_input()
            }),
            Err(FovrgError::InvalidQuantumNumber { name: "kappa", .. })
        ));

        input.potential[2] = Complex::new(Real::NAN, 0.0);
        assert!(matches!(
            fovrg_inward_solution(input.to_input()),
            Err(FovrgError::NonFiniteComplexInput {
                name: "potential",
                row: 2,
                ..
            })
        ));
    }

    struct SoloutReferenceInputs {
        initial_large_coefficient: Complex,
        initial_small_coefficient: Complex,
        energy: Complex,
        origin_power: Real,
        kappa: i32,
        muffin_tin_radius: Real,
        potential: Array1<Complex>,
        potential_coefficients: Array1<Complex>,
        large_exchange: Array1<Complex>,
        small_exchange: Array1<Complex>,
        large_exchange_coefficients: Array1<Complex>,
        small_exchange_coefficients: Array1<Complex>,
        c3_potential: Array1<Complex>,
        radii: Array1<Real>,
        c3_scale: i32,
        radial_match_index: usize,
        last_index: usize,
        wkb_index: usize,
        coefficient_count: usize,
        active_len: usize,
    }

    impl SoloutReferenceInputs {
        fn to_input(&self) -> FovrgOutgoingSolutionInput<'_> {
            FovrgOutgoingSolutionInput {
                initial_large_coefficient: self.initial_large_coefficient,
                initial_small_coefficient: self.initial_small_coefficient,
                energy: self.energy,
                origin_power: self.origin_power,
                kappa: self.kappa,
                muffin_tin_radius: self.muffin_tin_radius,
                potential: self.potential.view(),
                potential_coefficients: self.potential_coefficients.view(),
                large_exchange: self.large_exchange.view(),
                small_exchange: self.small_exchange.view(),
                large_exchange_coefficients: self.large_exchange_coefficients.view(),
                small_exchange_coefficients: self.small_exchange_coefficients.view(),
                c3_potential: self.c3_potential.view(),
                radii: self.radii.view(),
                speed_of_light: 137.035_999_084,
                step: 0.045,
                c3_scale: self.c3_scale,
                radial_match_index: self.radial_match_index,
                last_index: self.last_index,
                wkb_index: self.wkb_index,
                coefficient_count: self.coefficient_count,
                active_len: self.active_len,
            }
        }
    }

    fn solout_reference_inputs(case_id: usize) -> SoloutReferenceInputs {
        let active_len = 15;
        let coefficient_count = 6;
        let radii = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            0.18 * ((row - 1.0) * 0.045).exp()
        }));
        let potential = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(-0.18 + 0.013 * row, 0.004 * (0.37 * row).cos())
        }));
        let large_exchange = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(0.006 * (0.42 * row).sin(), -0.003 * (0.28 * row).cos())
        }));
        let small_exchange = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(-0.004 * (0.31 * row).cos(), 0.0025 * (0.53 * row).sin())
        }));
        let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
        }));
        let large_exchange_coefficients = Array1::from_iter((1..=coefficient_count).map(|row| {
            let row = row as Real;
            Complex::new(
                0.0025 * row + 0.001 * (0.33 * row).cos(),
                -0.0015 * row + 0.0007 * (0.21 * row).sin(),
            )
        }));
        let small_exchange_coefficients = Array1::from_iter((1..=coefficient_count).map(|row| {
            let row = row as Real;
            Complex::new(
                -0.0018 * row + 0.0008 * (0.27 * row).sin(),
                0.0012 * row + 0.0005 * (0.19 * row).cos(),
            )
        }));

        let mut potential_coefficients = Array1::<Complex>::zeros(coefficient_count);
        match case_id {
            1 => {
                potential_coefficients[0] = Complex::new(-0.21, 0.0);
                potential_coefficients[1] = Complex::new(0.013, -0.002);
                potential_coefficients[2] = Complex::new(-0.004, 0.001);
                potential_coefficients[3] = Complex::new(0.002, 0.0005);
                potential_coefficients[4] = Complex::new(-0.001, 0.0002);
                potential_coefficients[5] = Complex::new(0.0006, -0.0001);
                SoloutReferenceInputs {
                    initial_large_coefficient: Complex::new(0.85, -0.13),
                    initial_small_coefficient: Complex::new(-0.045, 0.018),
                    energy: Complex::new(-0.42, 0.018),
                    origin_power: 1.982,
                    kappa: -2,
                    muffin_tin_radius: 1.35,
                    potential,
                    potential_coefficients,
                    large_exchange,
                    small_exchange,
                    large_exchange_coefficients,
                    small_exchange_coefficients,
                    c3_potential,
                    radii,
                    c3_scale: 0,
                    radial_match_index: 8,
                    last_index: 11,
                    wkb_index: 6,
                    coefficient_count,
                    active_len,
                }
            }
            2 => {
                potential_coefficients[0] = Complex::new(0.11, 0.0);
                potential_coefficients[1] = Complex::new(-0.009, 0.002);
                potential_coefficients[2] = Complex::new(0.003, -0.001);
                potential_coefficients[3] = Complex::new(0.018, -0.004);
                potential_coefficients[4] = Complex::new(0.001, 0.0003);
                potential_coefficients[5] = Complex::new(-0.0004, 0.0002);
                SoloutReferenceInputs {
                    initial_large_coefficient: Complex::new(-0.72, 0.21),
                    initial_small_coefficient: Complex::new(0.037, -0.015),
                    energy: Complex::new(0.36, -0.027),
                    origin_power: 3.025,
                    kappa: 3,
                    muffin_tin_radius: 1.20,
                    potential,
                    potential_coefficients,
                    large_exchange,
                    small_exchange,
                    large_exchange_coefficients,
                    small_exchange_coefficients,
                    c3_potential,
                    radii,
                    c3_scale: 1,
                    radial_match_index: 9,
                    last_index: 12,
                    wkb_index: 7,
                    coefficient_count,
                    active_len,
                }
            }
            _ => {
                potential_coefficients[0] = Complex::new(-0.18, 0.0);
                potential_coefficients[1] = Complex::new(0.010, 0.001);
                potential_coefficients[2] = Complex::new(-0.003, 0.0008);
                potential_coefficients[3] = Complex::new(-0.015, 0.003);
                potential_coefficients[4] = Complex::new(0.0008, -0.0002);
                potential_coefficients[5] = Complex::new(-0.0003, 0.0001);
                SoloutReferenceInputs {
                    initial_large_coefficient: Complex::new(0.64, 0.08),
                    initial_small_coefficient: Complex::new(0.025, -0.011),
                    energy: Complex::new(0.22, 0.041),
                    origin_power: 0.965,
                    kappa: -1,
                    muffin_tin_radius: 1.40,
                    potential,
                    potential_coefficients,
                    large_exchange,
                    small_exchange,
                    large_exchange_coefficients,
                    small_exchange_coefficients,
                    c3_potential,
                    radii,
                    c3_scale: 1,
                    radial_match_index: 8,
                    last_index: 10,
                    wkb_index: 7,
                    coefficient_count,
                    active_len,
                }
            }
        }
    }

    struct SolinReferenceInputs {
        initial_large_coefficient: Complex,
        initial_small_coefficient: Complex,
        energy: Complex,
        origin_power: Real,
        kappa: i32,
        muffin_tin_radius: Real,
        potential: Array1<Complex>,
        large_exchange: Array1<Complex>,
        small_exchange: Array1<Complex>,
        c3_potential: Array1<Complex>,
        radii: Array1<Real>,
        c3_scale: i32,
        radial_match_index: usize,
        last_index: usize,
        wkb_index: usize,
        coefficient_count: usize,
        active_len: usize,
    }

    impl SolinReferenceInputs {
        fn to_input(&self) -> FovrgInwardSolutionInput<'_> {
            FovrgInwardSolutionInput {
                initial_large_coefficient: self.initial_large_coefficient,
                initial_small_coefficient: self.initial_small_coefficient,
                energy: self.energy,
                origin_power: self.origin_power,
                kappa: self.kappa,
                muffin_tin_radius: self.muffin_tin_radius,
                potential: self.potential.view(),
                large_exchange: self.large_exchange.view(),
                small_exchange: self.small_exchange.view(),
                c3_potential: self.c3_potential.view(),
                radii: self.radii.view(),
                speed_of_light: 137.035_999_084,
                step: 0.045,
                c3_scale: self.c3_scale,
                radial_match_index: self.radial_match_index,
                last_index: self.last_index,
                wkb_index: self.wkb_index,
                coefficient_count: self.coefficient_count,
                active_len: self.active_len,
            }
        }
    }

    fn solin_reference_inputs(case_id: usize) -> SolinReferenceInputs {
        let active_len = 15;
        let coefficient_count = 6;
        let radii = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            0.18 * ((row - 1.0) * 0.045).exp()
        }));
        let potential = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(-0.18 + 0.013 * row, 0.004 * (0.37 * row).cos())
        }));
        let large_exchange = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(0.006 * (0.42 * row).sin(), -0.003 * (0.28 * row).cos())
        }));
        let small_exchange = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(-0.004 * (0.31 * row).cos(), 0.0025 * (0.53 * row).sin())
        }));
        let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
        }));

        match case_id {
            1 => SolinReferenceInputs {
                initial_large_coefficient: Complex::new(0.85, -0.13),
                initial_small_coefficient: Complex::new(-0.045, 0.018),
                energy: Complex::new(0.42, 0.018),
                origin_power: 1.982,
                kappa: -2,
                muffin_tin_radius: 1.35,
                potential,
                large_exchange,
                small_exchange,
                c3_potential,
                radii,
                c3_scale: 0,
                radial_match_index: 8,
                last_index: 11,
                wkb_index: 6,
                coefficient_count,
                active_len,
            },
            2 => SolinReferenceInputs {
                initial_large_coefficient: Complex::new(-0.72, 0.21),
                initial_small_coefficient: Complex::new(0.037, -0.015),
                energy: Complex::new(0.36, -0.027),
                origin_power: 3.025,
                kappa: 3,
                muffin_tin_radius: 1.20,
                potential,
                large_exchange,
                small_exchange,
                c3_potential,
                radii,
                c3_scale: 0,
                radial_match_index: 9,
                last_index: 12,
                wkb_index: 7,
                coefficient_count,
                active_len,
            },
            _ => SolinReferenceInputs {
                initial_large_coefficient: Complex::new(0.64, 0.08),
                initial_small_coefficient: Complex::new(0.025, -0.011),
                energy: Complex::new(0.22, 0.041),
                origin_power: 0.965,
                kappa: -1,
                muffin_tin_radius: 1.40,
                potential,
                large_exchange,
                small_exchange,
                c3_potential,
                radii,
                c3_scale: 0,
                radial_match_index: 8,
                last_index: 12,
                wkb_index: 7,
                coefficient_count,
                active_len,
            },
        }
    }

    #[test]
    fn nuclear_potential_matches_feff_nucdec_point_reference() -> Result<(), FovrgError> {
        let potential = fovrg_nuclear_potential(FovrgNuclearPotentialInput {
            nuclear_charge: 29.0,
            step: 0.0725,
            first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
            radial_count: 8,
            coefficient_count: 6,
        })?;

        assert_eq!(potential.nucleus_index, 1);
        assert_close(
            potential.first_radius_times_charge,
            0.004_371_259_177_768_818_5,
            1.0e-15,
        );
        let expected_coefficients = [-29.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (actual, expected) in potential
            .development_coefficients
            .iter()
            .zip(expected_coefficients)
        {
            assert_close(*actual, expected, 1.0e-13);
        }

        let expected_rows = [
            (0.000_150_733_075_095_476_5, -192_393.076_182_058_78),
            (0.000_162_067_117_982_503_44, -178_938.210_051_534_35),
            (0.000_174_253_399_358_552_06, -166_424.299_937_634_06),
            (0.000_187_356_001_427_070_04, -154_785.540_783_909_73),
            (0.000_201_443_824_912_202_5, -143_960.729_561_402),
            (0.000_216_590_951_376_884_9, -133_892.943_429_283_77),
            (0.000_232_877_032_784_649_17, -124_529.240_403_099_25),
            (0.000_250_387_710_353_676, -115_820.380_956_545_8),
        ];
        for (row, (expected_radius, expected_potential)) in expected_rows.into_iter().enumerate() {
            assert_close(potential.radii[row], expected_radius, 1.0e-13);
            assert_close(potential.potential[row], expected_potential, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn nuclear_potential_rejects_invalid_inputs() {
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 29.0,
                step: 0.0725,
                first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
                radial_count: 0,
                coefficient_count: 6,
            }),
            Err(FovrgError::CountTooSmall {
                name: "radial_count",
                ..
            })
        ));
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 29.0,
                step: 0.0725,
                first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
                radial_count: 8,
                coefficient_count: 4,
            }),
            Err(FovrgError::CountTooSmall {
                name: "coefficient_count",
                ..
            })
        ));
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 0.0,
                step: 0.0725,
                first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
                radial_count: 8,
                coefficient_count: 6,
            }),
            Err(FovrgError::NonPositiveInput {
                name: "nuclear_charge",
                ..
            })
        ));
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 29.0,
                step: 0.0,
                first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
                radial_count: 8,
                coefficient_count: 6,
            }),
            Err(FovrgError::NonPositiveInput { name: "step", .. })
        ));
        assert!(matches!(
            fovrg_nuclear_potential(FovrgNuclearPotentialInput {
                nuclear_charge: 29.0,
                step: 0.0725,
                first_radius_times_charge: Real::NAN,
                radial_count: 8,
                coefficient_count: 6,
            }),
            Err(FovrgError::NonFiniteInput {
                name: "first_radius_times_charge",
                ..
            })
        ));
    }

    #[test]
    fn yk_zk_transform_matches_feff_yzktec_reference() -> Result<(), FovrgError> {
        let (source, coefficients, radii) = yzktec_reference_inputs(12);

        let transform = fovrg_yk_zk_transform(FovrgYkZkTransformInput {
            source: source.view(),
            source_coefficients: coefficients.view(),
            radii: radii.view(),
            initial_power: Complex::new(1.35, -0.25),
            step: 0.0725,
            angular_momentum: 2,
            coefficient_count: 6,
            source_len: 9,
            active_len: 12,
            tail_correction: Complex::new(0.011, -0.006),
        })?;

        assert_eq!(transform.computed_len, 10);
        assert_complex_close(
            transform.origin_constant,
            1_069.293_326_934_643,
            639.337_203_837_502_8,
            1.0e-12,
        );

        let expected_rows = [
            (
                0.006_376_970_423_953_328,
                0.003_747_109_936_537_645_4,
                0.000_019_115_876_398_023_115,
                0.000_002_615_603_860_575_636,
            ),
            (
                0.007_841_326_927_116_237,
                0.004_425_503_213_295_339,
                0.000_415_175_421_810_819_7,
                0.001_186_221_311_123_577_3,
            ),
            (
                0.009_454_062_278_996_728,
                0.004_817_609_696_528_203,
                0.001_052_233_690_642_138_8,
                0.002_225_420_005_754_274_7,
            ),
            (
                0.011_156_498_748_891_856,
                0.004_912_703_002_968_925,
                0.001_915_624_422_266_479,
                0.003_118_393_016_633_964_7,
            ),
            (
                0.012_883_154_525_001_68,
                0.004_698_896_965_378_377,
                0.002_982_924_829_137_327,
                0.003_859_683_726_441_837_7,
            ),
            (
                0.014_563_357_943_144_598,
                0.004_164_285_902_400_606,
                0.004_223_307_649_668_978,
                0.004_440_459_445_284_031,
            ),
            (
                0.016_123_447_951_845_19,
                0.003_298_256_791_962_156,
                0.005_597_243_449_987_172_5,
                0.004_848_768_666_445_236,
            ),
            (
                0.017_489_549_856_229_16,
                0.002_093_015_402_338_084,
                0.007_056_612_425_756_153,
                0.005_069_801_813_782_371,
            ),
            (
                0.018_590_890_204_374_55,
                0.000_545_375_511_912_808,
                0.008_545_277_387_035_53,
                0.005_086_162_856_970_115_5,
            ),
            (
                0.019_630_800_153_888_66,
                -0.001_305_325_902_639_564_2,
                0.008_630_800_153_888_66,
                0.004_694_674_097_360_436,
            ),
        ];
        for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_rows.into_iter().enumerate() {
            assert_complex_close(transform.yk[row], yk_re, yk_im, 1.0e-13);
            assert_complex_close(transform.zk[row], zk_re, zk_im, 1.0e-13);
        }

        let expected_coefficients = [
            (
                -1.824_158_963_244_542,
                -0.246_122_633_186_553_6,
                0.237_140_665_221_790_4,
                0.031_995_942_314_251_964,
            ),
            (
                2.794_098_740_012_050_3,
                0.730_272_609_187_755_4,
                0.195_586_911_800_843_6,
                0.051_119_082_643_142_88,
            ),
            (
                0.609_454_103_153_871_8,
                0.232_241_030_552_876_95,
                0.164_552_607_851_545_35,
                0.062_705_078_249_276_76,
            ),
            (
                0.297_530_519_518_787_1,
                0.147_284_129_112_308_2,
                0.139_839_344_173_829_93,
                0.069_223_540_682_784_84,
            ),
            (
                0.178_046_974_447_949_95,
                0.107_477_058_743_461_04,
                0.119_291_472_880_126_45,
                0.072_009_629_358_118_89,
            ),
            (
                0.116_898_830_661_045_85,
                0.082_624_380_349_051_94,
                0.101_701_982_675_109_88,
                0.071_883_210_903_675_18,
            ),
        ];
        for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_coefficients.into_iter().enumerate() {
            assert_complex_close(transform.yk_coefficients[row], yk_re, yk_im, 1.0e-13);
            assert_complex_close(transform.zk_coefficients[row], zk_re, zk_im, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn yk_zk_transform_rejects_invalid_inputs() {
        let (source, coefficients, radii) = yzktec_reference_inputs(12);

        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 1,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::NonPositiveInput { name: "step", .. })
        ));
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 11,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "source_coefficients",
                ..
            })
        ));

        let mut bad_source = source.clone();
        bad_source[3] = Complex::new(0.0, Real::NAN);
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: bad_source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::NonFiniteComplexInput {
                name: "source",
                row: 3,
                ..
            })
        ));

        let mut bad_radii = radii.clone();
        bad_radii[0] = -1.0;
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: bad_radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::NonPositiveRadius { row: 0, .. })
        ));
    }

    #[test]
    fn yk_zk_exchange_matches_feff_yzkrdc_reference() -> Result<(), FovrgError> {
        let input = yzkrdc_reference_inputs(12);

        let transform = fovrg_yk_zk_exchange(input.as_exchange_input())?;

        assert_eq!(transform.computed_len, 10);
        assert_complex_close(
            transform.origin_constant,
            1_321.269_761_542_853_5,
            1_058.551_269_340_285_2,
            1.0e-12,
        );

        let expected_rows = [
            (
                0.007_686_009_135_817_749,
                0.006_170_157_063_400_744,
                0.000_000_645_317_783_462_879_7,
                0.000_000_110_270_749_084_274_43,
            ),
            (
                0.009_300_746_624_727_518,
                0.007_544_419_441_270_886,
                0.001_294_275_945_600_778,
                0.000_639_802_281_166_626_1,
            ),
            (
                0.010_786_770_527_864_456,
                0.008_925_139_869_295_514,
                0.002_573_522_373_652_341,
                0.001_630_025_738_506_313,
            ),
            (
                0.012_109_032_230_448_815,
                0.010_184_928_348_947_297,
                0.003_887_582_939_633_221_6,
                0.002_904_232_874_818_797,
            ),
            (
                0.013_206_275_901_284_993,
                0.011_197_011_268_772_228,
                0.005_274_223_639_134_339,
                0.004_372_201_622_443_519_5,
            ),
            (
                0.013_990_089_034_721_83,
                0.011_844_365_308_609_77,
                0.006_755_737_128_168_105,
                0.005_923_123_611_939_633,
            ),
            (
                0.014_345_254_715_897_94,
                0.012_029_374_779_196_415,
                0.008_335_434_490_732_629,
                0.007_430_974_286_925_581,
            ),
            (
                0.014_131_414_050_294_111,
                0.011_683_264_170_573_946,
                0.009_995_141_713_724_128,
                0.008_761_915_452_378_507,
            ),
            (
                0.013_185_862_903_069_551,
                0.010_774_522_262_808_485,
                0.011_694_148_402_953_802,
                0.009_783_248_603_735_934,
            ),
            (
                0.011_660_651_859_152_821,
                0.009_488_268_367_479_21,
                0.011_660_651_859_152_821,
                0.009_488_268_367_479_21,
            ),
        ];
        for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_rows.into_iter().enumerate() {
            assert_complex_close(transform.yk[row], yk_re, yk_im, 1.0e-13);
            assert_complex_close(transform.zk[row], zk_re, zk_im, 1.0e-13);
        }

        let expected_coefficients = [
            (6.375_854_958_562_043, 1.073_871_387_646_292_4),
            (1.497_833_540_772_686, 0.370_169_086_848_655_57),
            (1.049_320_795_997_538_8, 0.338_218_568_996_506_2),
            (0.843_625_047_557_286_8, 0.332_360_660_760_794_2),
            (0.713_658_559_831_859_2, 0.329_689_349_293_459_3),
            (0.619_406_204_717_043, 0.325_898_372_715_123_5),
        ];
        for (row, (expected_re, expected_im)) in expected_coefficients.into_iter().enumerate() {
            assert_complex_close(
                transform.yk_coefficients[row],
                expected_re,
                expected_im,
                1.0e-13,
            );
        }
        Ok(())
    }

    #[test]
    fn yk_zk_exchange_rejects_invalid_inputs() {
        let mut input = yzkrdc_reference_inputs(12);
        input.large_component[2] = Real::NAN;

        assert!(matches!(
            fovrg_yk_zk_exchange(input.as_exchange_input()),
            Err(FovrgError::NonFiniteRealInput {
                name: "large_component",
                row: 2,
                ..
            })
        ));

        let mut input = yzkrdc_reference_inputs(12);
        input.partner_small_coefficients[1] = Complex::new(0.0, Real::INFINITY);
        assert!(matches!(
            fovrg_yk_zk_exchange(input.as_exchange_input()),
            Err(FovrgError::NonFiniteComplexInput {
                name: "partner_small_coefficients",
                row: 1,
                ..
            })
        ));

        let input = yzkrdc_reference_inputs(4);
        assert!(matches!(
            fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
                active_len: 5,
                ..input.as_exchange_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "large_component",
                ..
            })
        ));
    }

    #[test]
    fn overlap_integral_matches_feff_dsordc_reference() -> Result<(), FovrgError> {
        let input = dsordc_reference_inputs(9);

        let integral = fovrg_overlap_integral(input.as_overlap_input())?;

        assert_complex_close(
            integral,
            0.018_257_373_605_649_284,
            0.014_647_428_406_545_006,
            1.0e-13,
        );
        Ok(())
    }

    #[test]
    fn overlap_integral_rejects_invalid_inputs() {
        let input = dsordc_reference_inputs(9);

        assert!(matches!(
            fovrg_overlap_integral(FovrgOverlapIntegralInput {
                active_len: 8,
                ..input.as_overlap_input()
            }),
            Err(FovrgError::CountMustBeOdd {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_overlap_integral(FovrgOverlapIntegralInput {
                active_len: 2,
                ..input.as_overlap_input()
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_overlap_integral(FovrgOverlapIntegralInput {
                active_len: 11,
                ..input.as_overlap_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "large_integrand",
                ..
            })
        ));
        assert!(matches!(
            fovrg_overlap_integral(FovrgOverlapIntegralInput {
                step: 0.0,
                ..input.as_overlap_input()
            }),
            Err(FovrgError::NonPositiveInput { name: "step", .. })
        ));

        let mut input = dsordc_reference_inputs(9);
        input.radii[2] = 0.0;
        assert!(matches!(
            fovrg_overlap_integral(input.as_overlap_input()),
            Err(FovrgError::NonPositiveRadius { row: 2, .. })
        ));

        let mut input = dsordc_reference_inputs(9);
        input.large_integrand_coefficients[3] = Complex::new(Real::NAN, 0.0);
        assert!(matches!(
            fovrg_overlap_integral(input.as_overlap_input()),
            Err(FovrgError::NonFiniteComplexInput {
                name: "large_integrand_coefficients",
                row: 3,
                ..
            })
        ));
    }

    #[test]
    fn schmidt_orthogonalization_matches_feff_ortdac_reference() -> Result<(), FovrgError> {
        let input = ortdac_reference_inputs(9);

        let orthogonalized = fovrg_schmidt_orthogonalize(input.as_orthogonalization_input())?;

        assert_ne!(orthogonalized.overlaps[0], Complex::new(0.0, 0.0));
        assert_eq!(orthogonalized.overlaps[1], Complex::new(0.0, 0.0));
        assert_eq!(orthogonalized.overlaps[2], Complex::new(0.0, 0.0));
        assert_ne!(orthogonalized.overlaps[3], Complex::new(0.0, 0.0));

        let expected_rows = [
            (
                0.184_796_621_476_688_8,
                0.960_525_659_674_847_8,
                0.953_489_591_844_743_2,
                0.196_175_227_984_495_05,
            ),
            (
                0.364_943_848_030_108_26,
                0.909_210_209_413_431_4,
                0.932_155_457_421_994,
                0.411_067_158_250_690_3,
            ),
            (
                0.535_755_652_121_730_3,
                0.846_307_238_142_853_7,
                0.903_311_497_295_576_5,
                0.608_386_237_505_285_2,
            ),
            (
                0.692_898_032_849_261_3,
                0.772_271_807_926_033_1,
                0.867_091_885_664_384_1,
                0.780_141_644_613_100_9,
            ),
            (
                0.832_426_823_226_043_9,
                0.687_685_325_115_306_8,
                0.823_683_636_514_673_7,
                0.919_472_245_213_980_9,
            ),
            (
                0.950_900_258_284_096_4,
                0.593_246_291_500_718_2,
                0.773_325_760_496_256_5,
                1.020_947_524_294_445_2,
            ),
            (
                1.045_477_281_469_491_5,
                0.489_760_058_904_124,
                0.716_308_162_735_437_1,
                1.080_805_531_074_559_2,
            ),
            (
                1.113_998_793_619_971_6,
                0.378_127_783_523_524,
                0.652_970_269_213_538_8,
                1.097_117_398_093_875_3,
            ),
            (
                1.155_049_530_148_776_2,
                0.259_334_766_409_572_26,
                0.583_699_367_941_233_6,
                1.069_871_225_404_380_5,
            ),
        ];
        for (row, (large_re, large_im, small_re, small_im)) in expected_rows.into_iter().enumerate()
        {
            assert_complex_close(
                orthogonalized.large_component[row],
                large_re,
                large_im,
                1.0e-13,
            );
            assert_complex_close(
                orthogonalized.small_component[row],
                small_re,
                small_im,
                1.0e-13,
            );
        }

        let expected_coefficients = [
            (
                0.998_449_079_711_476_6,
                0.111_350_550_939_607_72,
                0.068_224_857_711_253_53,
                1.016_544_722_930_134,
            ),
            (
                1.013_026_259_606_995_7,
                0.245_410_754_538_065_13,
                0.135_740_121_285_759_1,
                1.018_823_567_734_752_8,
            ),
            (
                1.011_555_658_693_028_5,
                0.370_053_923_878_711_44,
                0.201_841_908_576_728_68,
                1.007_158_627_823_294_7,
            ),
            (
                0.994_728_243_334_449_6,
                0.480_813_496_862_218_4,
                0.265_837_715_515_805,
                0.982_072_667_678_714_5,
            ),
            (
                0.963_494_952_966_650_6,
                0.573_626_109_357_632,
                0.327_051_990_766_753_06,
                0.944_281_663_334_709_4,
            ),
            (
                0.919_050_620_050_682_9,
                0.644_948_608_319_962_8,
                0.384_831_574_144_536_1,
                0.894_684_562_223_998_2,
            ),
        ];
        for (coefficient, (large_re, large_im, small_re, small_im)) in
            expected_coefficients.into_iter().enumerate()
        {
            assert_complex_close(
                orthogonalized.large_coefficients[coefficient],
                large_re,
                large_im,
                1.0e-13,
            );
            assert_complex_close(
                orthogonalized.small_coefficients[coefficient],
                small_re,
                small_im,
                1.0e-13,
            );
        }
        Ok(())
    }

    #[test]
    fn schmidt_orthogonalization_rejects_invalid_inputs() {
        let input = ortdac_reference_inputs(9);

        assert!(matches!(
            fovrg_schmidt_orthogonalize(FovrgOrthogonalizationInput {
                target_kappa: 0,
                ..input.as_orthogonalization_input()
            }),
            Err(FovrgError::InvalidQuantumNumber {
                name: "target_kappa",
                value: 0,
                ..
            })
        ));
        assert!(matches!(
            fovrg_schmidt_orthogonalize(FovrgOrthogonalizationInput {
                active_len: 8,
                ..input.as_orthogonalization_input()
            }),
            Err(FovrgError::CountMustBeOdd {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_schmidt_orthogonalize(FovrgOrthogonalizationInput {
                bound_orbital_count: 5,
                ..input.as_orthogonalization_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "bound_large_components",
                ..
            })
        ));

        let mut input = ortdac_reference_inputs(9);
        input.electron_counts[0] = Real::NAN;
        assert!(matches!(
            fovrg_schmidt_orthogonalize(input.as_orthogonalization_input()),
            Err(FovrgError::NonFiniteRealInput {
                name: "electron_counts",
                row: 0,
                ..
            })
        ));

        let mut input = ortdac_reference_inputs(9);
        input.target_large_component[1] = Complex::new(0.0, Real::INFINITY);
        assert!(matches!(
            fovrg_schmidt_orthogonalize(input.as_orthogonalization_input()),
            Err(FovrgError::NonFiniteComplexInput {
                name: "target_large_component",
                row: 1,
                ..
            })
        ));
    }

    #[test]
    fn exchange_potential_matches_feff_potex_reference() -> Result<(), FovrgError> {
        let input = potex_reference_inputs(9);

        let potential = fovrg_exchange_potential(input.as_exchange_potential_input())?;

        let expected_rows = [
            (
                0.000_005_554_864_571_582_592,
                0.000_004_589_245_040_261_105,
                0.000_039_609_278_623_293_83,
                0.000_033_434_207_770_074_9,
            ),
            (
                0.000_011_841_826_673_104_183,
                0.000_009_794_866_583_804_325,
                0.000_042_053_026_297_685_21,
                0.000_035_634_329_767_400_91,
            ),
            (
                0.000_018_578_019_491_824_635,
                0.000_015_404_560_990_245_302,
                0.000_043_309_970_634_588_22,
                0.000_036_974_047_816_590_13,
            ),
            (
                0.000_025_293_374_225_649_448,
                0.000_020_974_401_383_246_206,
                0.000_043_220_277_722_069_02,
                0.000_037_209_351_789_692_02,
            ),
            (
                0.000_031_463_027_867_695_25,
                0.000_026_005_262_272_416_62,
                0.000_041_711_066_672_227_27,
                0.000_036_233_682_295_299_64,
            ),
            (
                0.000_036_572_642_292_251_735,
                0.000_030_066_055_448_048_974,
                0.000_038_831_760_336_099_916,
                0.000_034_098_776_520_822_48,
            ),
            (
                0.000_040_212_198_293_964_37,
                0.000_032_909_100_990_340_72,
                0.000_034_779_450_472_774_91,
                0.000_030_991_961_623_472_31,
            ),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ];
        for (row, (large_re, large_im, small_re, small_im)) in expected_rows.into_iter().enumerate()
        {
            assert_complex_close(potential.large_potential[row], large_re, large_im, 1.0e-13);
            assert_complex_close(potential.small_potential[row], small_re, small_im, 1.0e-13);
        }

        let expected_coefficients = [
            (
                0.056_004_531_605_744_41,
                0.046_663_043_007_772_96,
                0.000_603_453_997_835_984_7,
                0.000_503_278_831_814_845_3,
            ),
            (
                -1.349_603_830_126_038,
                -1.128_877_944_124_393_2,
                -0.045_179_344_996_730_146,
                -0.037_885_484_599_471_386,
            ),
            (
                -2.231_032_417_788_144_4,
                -1.854_555_246_757_578,
                -0.141_157_217_260_626_58,
                -0.117_915_434_665_414_93,
            ),
            (
                24.781_460_626_354_06,
                19.753_480_329_995_963,
                2.027_705_895_254_902,
                1.613_953_852_227_001_6,
            ),
            (
                24.726_993_882_200_276,
                19.712_367_835_956_03,
                4.170_773_455_085_641,
                3.325_408_244_119_067_5,
            ),
            (
                24.319_899_966_071_53,
                19.384_360_683_910_17,
                6.262_813_264_194_341,
                4.995_924_412_893_355,
            ),
        ];
        for (coefficient, (large_re, large_im, small_re, small_im)) in
            expected_coefficients.into_iter().enumerate()
        {
            assert_complex_close(
                potential.large_coefficients[coefficient],
                large_re,
                large_im,
                1.0e-13,
            );
            assert_complex_close(
                potential.small_coefficients[coefficient],
                small_re,
                small_im,
                1.0e-13,
            );
        }
        Ok(())
    }

    #[test]
    fn exchange_potential_rejects_invalid_inputs() {
        let input = potex_reference_inputs(9);

        assert!(matches!(
            fovrg_exchange_potential(FovrgExchangePotentialInput {
                target_kappa: 0,
                ..input.as_exchange_potential_input()
            }),
            Err(FovrgError::InvalidQuantumNumber {
                name: "target_kappa",
                value: 0,
                ..
            })
        ));
        assert!(matches!(
            fovrg_exchange_potential(FovrgExchangePotentialInput {
                radial_output_count: 10,
                ..input.as_exchange_potential_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "radial_output_count",
                ..
            })
        ));
        assert!(matches!(
            fovrg_exchange_potential(FovrgExchangePotentialInput {
                speed_of_light: 0.0,
                ..input.as_exchange_potential_input()
            }),
            Err(FovrgError::ZeroInput {
                name: "speed_of_light"
            })
        ));
        assert!(matches!(
            fovrg_exchange_potential(FovrgExchangePotentialInput {
                bound_orbital_count: 5,
                ..input.as_exchange_potential_input()
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "bound_large_components",
                ..
            })
        ));

        let mut input = potex_reference_inputs(9);
        input.orbital_lengths[2] = 0;
        assert!(matches!(
            fovrg_exchange_potential(input.as_exchange_potential_input()),
            Err(FovrgError::CountTooSmall {
                name: "orbital_length",
                ..
            })
        ));

        let mut input = potex_reference_inputs(9);
        input.angular_coefficients[(1, 0)] = Real::NAN;
        assert!(matches!(
            fovrg_exchange_potential(input.as_exchange_potential_input()),
            Err(FovrgError::NonFiniteRealInput {
                name: "angular_coefficients",
                row: 1,
                ..
            })
        ));
    }

    #[test]
    fn potential_development_matches_feff_potdvp_reference() -> Result<(), FovrgError> {
        let input = potdvp_reference_inputs(12);

        let development = fovrg_potential_development(input.as_potential_input())?;

        assert_close(
            development.origin_correction,
            0.000_092_381_409_682_418_76,
            1.0e-13,
        );
        let expected_potential = [
            -0.002_211_097_828_492_991_6,
            -0.001_838_258_707_742_217_9,
            -0.001_437_578_456_148_908_5,
            -0.003_049_520_002_144_625,
            -0.002_623_511_736_279_590_5,
            -0.002_546_330_557_249_715,
            -0.002_045_957_521_005_020_5,
            -0.001_773_999_888_200_908_3,
            0.001_583_525_507_534_584_8,
            0.002_189_205_770_785_14,
        ];
        for (actual, expected) in development
            .potential_coefficients
            .iter()
            .zip(expected_potential)
        {
            assert_complex_close(*actual, expected, 0.0, 1.0e-13);
        }

        let expected_density = [
            0.279_894_020_220_530_5,
            0.284_515_551_889_673_2,
            0.340_938_951_910_833_2,
            0.343_369_832_974_347,
            0.381_101_847_054_515_8,
            0.388_553_939_183_866_9,
            0.381_768_833_467_862,
            0.368_012_415_945_436_16,
        ];
        for (actual, expected) in development
            .density_coefficients
            .iter()
            .zip(expected_density)
        {
            assert_close(*actual, expected, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn potential_development_rejects_invalid_inputs() {
        let mut input = potdvp_reference_inputs(12);
        input.nuclear_coefficients[0] = Real::NAN;
        assert!(matches!(
            fovrg_potential_development(input.as_potential_input()),
            Err(FovrgError::NonFiniteRealInput {
                name: "nuclear_coefficients",
                row: 0,
                ..
            })
        ));

        let mut input = potdvp_reference_inputs(12);
        input.kappa[1] = 0;
        assert!(matches!(
            fovrg_potential_development(input.as_potential_input()),
            Err(FovrgError::InvalidQuantumNumber {
                name: "kappa",
                row: 1,
                value: 0,
            })
        ));

        let mut input = potdvp_reference_inputs(12);
        input.large_coefficients = Array2::zeros((7, 4));
        assert!(matches!(
            fovrg_potential_development(input.as_potential_input()),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "large_coefficients",
                ..
            })
        ));

        let input = potdvp_reference_inputs(12);
        assert!(matches!(
            fovrg_potential_development(FovrgPotentialDevelopmentInput {
                speed_of_light: 0.0,
                ..input.as_potential_input()
            }),
            Err(FovrgError::ZeroInput {
                name: "speed_of_light"
            })
        ));
    }

    struct IntoutReferenceInputs {
        initial_large_component: Complex,
        initial_small_component: Complex,
        energy: Complex,
        potential: Array1<Complex>,
        potential_coefficients: Array1<Complex>,
        large_exchange: Array1<Complex>,
        small_exchange: Array1<Complex>,
        c3_potential: Array1<Complex>,
        radii: Array1<Real>,
        kappa: i32,
        c3_scale: i32,
        start_index: usize,
        last_index: usize,
        active_len: usize,
    }

    impl IntoutReferenceInputs {
        fn to_input(&self) -> FovrgOutwardIntegrationInput<'_> {
            FovrgOutwardIntegrationInput {
                initial_large_component: self.initial_large_component,
                initial_small_component: self.initial_small_component,
                energy: self.energy,
                potential: self.potential.view(),
                potential_coefficients: self.potential_coefficients.view(),
                large_exchange: self.large_exchange.view(),
                small_exchange: self.small_exchange.view(),
                c3_potential: self.c3_potential.view(),
                radii: self.radii.view(),
                speed_of_light: 137.035_999_084,
                step: 0.045,
                kappa: self.kappa,
                c3_scale: self.c3_scale,
                start_index: self.start_index,
                last_index: self.last_index,
                active_len: self.active_len,
            }
        }
    }

    fn intout_reference_inputs(case_id: usize) -> IntoutReferenceInputs {
        let active_len = 15;
        let mut potential_coefficients = Array1::<Complex>::zeros(10);
        let radii = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            0.18 * ((row - 1.0) * 0.045).exp()
        }));
        let potential = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(-0.18 + 0.013 * row, 0.004 * (0.37 * row).cos())
        }));
        let large_exchange = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(0.006 * (0.42 * row).sin(), -0.003 * (0.28 * row).cos())
        }));
        let small_exchange = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(-0.004 * (0.31 * row).cos(), 0.0025 * (0.53 * row).sin())
        }));
        let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
            let row = row as Real;
            Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
        }));

        match case_id {
            1 => {
                potential_coefficients[0] = Complex::new(-0.21, 0.0);
                IntoutReferenceInputs {
                    initial_large_component: Complex::new(0.035, -0.012),
                    initial_small_component: Complex::new(-0.009, 0.004),
                    energy: Complex::new(-0.42, 0.018),
                    potential,
                    potential_coefficients,
                    large_exchange,
                    small_exchange,
                    c3_potential,
                    radii,
                    kappa: -2,
                    c3_scale: 0,
                    start_index: 0,
                    last_index: 11,
                    active_len,
                }
            }
            2 => {
                potential_coefficients[0] = Complex::new(0.11, 0.0);
                potential_coefficients[3] = Complex::new(0.018, -0.004);
                IntoutReferenceInputs {
                    initial_large_component: Complex::new(-0.017, 0.028),
                    initial_small_component: Complex::new(0.011, -0.006),
                    energy: Complex::new(0.36, -0.027),
                    potential,
                    potential_coefficients,
                    large_exchange,
                    small_exchange,
                    c3_potential,
                    radii,
                    kappa: 3,
                    c3_scale: 1,
                    start_index: 0,
                    last_index: 12,
                    active_len,
                }
            }
            _ => {
                potential_coefficients[0] = Complex::new(0.09, 0.0);
                potential_coefficients[3] = Complex::new(-0.015, 0.003);
                IntoutReferenceInputs {
                    initial_large_component: Complex::new(0.026, 0.014),
                    initial_small_component: Complex::new(-0.008, 0.017),
                    energy: Complex::new(0.22, 0.041),
                    potential,
                    potential_coefficients,
                    large_exchange,
                    small_exchange,
                    c3_potential,
                    radii,
                    kappa: -1,
                    c3_scale: 1,
                    start_index: 3,
                    last_index: 10,
                    active_len,
                }
            }
        }
    }

    fn diff_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Real>) {
        let potential = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            Complex::new(
                (0.21 * index).sin() + 0.03 * index,
                (0.17 * index).cos() - 0.02 * index,
            )
        }));
        let radii = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            0.15 + 0.04 * index + 0.001 * index * index
        }));
        (potential, radii)
    }

    fn aprd_reference_inputs(count: usize) -> (Array1<Real>, Array1<Real>, Array1<Complex>) {
        let real_left = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.02 * row + (0.03 * row * 2.0).cos()
        }));
        let real_right = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            -0.015 * row + (0.025 * row * 3.0).sin()
        }));
        let complex_left = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                0.04 * row + (0.13 * row).cos(),
                -0.03 * row + (0.17 * row).sin(),
            )
        }));
        (real_left, real_right, complex_left)
    }

    fn muatcc_reference_inputs() -> (Array1<Real>, Array1<Real>, Array1<i32>) {
        (
            Array1::from_vec(vec![2.0, 1.5, 2.5, 1.0, 3.0]),
            Array1::from_vec(vec![0.0, 0.25, -0.10, 0.0, -0.20]),
            Array1::from_vec(vec![-1, 1, -2, 2, -3]),
        )
    }

    fn yzktec_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Complex>, Array1<Real>) {
        let step = 0.0725;
        let source = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            Complex::new(
                (0.19 * index).sin() + 0.02 * index,
                (0.11 * index).cos() - 0.03 * index,
            )
        }));
        let coefficients = Array1::from_iter((1..=10).map(|index| {
            let index = index as Real;
            Complex::new(
                0.04 * index + (0.13 * index).cos(),
                -0.03 * index + (0.17 * index).sin(),
            )
        }));
        let radii = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            0.018 * (step * (index - 1.0)).exp()
        }));
        (source, coefficients, radii)
    }

    struct YzkrdcReferenceInputs {
        large_component: Array1<Real>,
        small_component: Array1<Real>,
        large_coefficients: Array1<Real>,
        small_coefficients: Array1<Real>,
        partner_large_component: Array1<Complex>,
        partner_small_component: Array1<Complex>,
        partner_large_coefficients: Array1<Complex>,
        partner_small_coefficients: Array1<Complex>,
        radii: Array1<Real>,
        orbital_power: Real,
        partner_power: Real,
        step: Real,
        angular_momentum: usize,
        coefficient_count: usize,
        orbital_len: usize,
        source_len: usize,
        active_len: usize,
    }

    impl YzkrdcReferenceInputs {
        fn as_exchange_input(&self) -> FovrgYkZkExchangeInput<'_> {
            FovrgYkZkExchangeInput {
                large_component: self.large_component.view(),
                small_component: self.small_component.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                partner_large_component: self.partner_large_component.view(),
                partner_small_component: self.partner_small_component.view(),
                partner_large_coefficients: self.partner_large_coefficients.view(),
                partner_small_coefficients: self.partner_small_coefficients.view(),
                radii: self.radii.view(),
                orbital_power: self.orbital_power,
                partner_power: self.partner_power,
                step: self.step,
                angular_momentum: self.angular_momentum,
                coefficient_count: self.coefficient_count,
                orbital_len: self.orbital_len,
                source_len: self.source_len,
                active_len: self.active_len,
            }
        }
    }

    fn yzkrdc_reference_inputs(count: usize) -> YzkrdcReferenceInputs {
        let step = 0.0725;
        let orbital_column = 2.0;
        let large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            (0.05 * row * orbital_column).sin() + 0.001 * (row + orbital_column)
        }));
        let small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            (0.04 * row * orbital_column).cos() - 0.002 * (row - orbital_column)
        }));
        let large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            0.02 * row + (0.03 * row * orbital_column).cos()
        }));
        let small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            -0.015 * row + (0.025 * row * orbital_column).sin()
        }));
        let partner_large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.19 * row).sin() + 0.02 * row,
                (0.11 * row).cos() - 0.03 * row,
            )
        }));
        let partner_small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.07 * row).cos() - 0.01 * row,
                (0.23 * row).sin() + 0.015 * row,
            )
        }));
        let partner_large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                0.04 * row + (0.13 * row).cos(),
                -0.03 * row + (0.17 * row).sin(),
            )
        }));
        let partner_small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                -0.02 * row + (0.09 * row).sin(),
                0.025 * row + (0.12 * row).cos(),
            )
        }));
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        YzkrdcReferenceInputs {
            large_component,
            small_component,
            large_coefficients,
            small_coefficients,
            partner_large_component,
            partner_small_component,
            partner_large_coefficients,
            partner_small_coefficients,
            radii,
            orbital_power: 0.65 + 0.08 * orbital_column,
            partner_power: 1.35,
            step,
            angular_momentum: 2,
            coefficient_count: 6,
            orbital_len: 9,
            source_len: 9,
            active_len: count,
        }
    }

    struct DsordcReferenceInputs {
        large_integrand: Array1<Complex>,
        small_integrand: Array1<Complex>,
        large_integrand_coefficients: Array1<Complex>,
        small_integrand_coefficients: Array1<Complex>,
        large_component: Array1<Real>,
        small_component: Array1<Real>,
        large_coefficients: Array1<Real>,
        small_coefficients: Array1<Real>,
        radii: Array1<Real>,
        integrand_power: Real,
        orbital_power: Real,
        step: Real,
        coefficient_count: usize,
        active_len: usize,
    }

    impl DsordcReferenceInputs {
        fn as_overlap_input(&self) -> FovrgOverlapIntegralInput<'_> {
            FovrgOverlapIntegralInput {
                large_integrand: self.large_integrand.view(),
                small_integrand: self.small_integrand.view(),
                large_integrand_coefficients: self.large_integrand_coefficients.view(),
                small_integrand_coefficients: self.small_integrand_coefficients.view(),
                large_component: self.large_component.view(),
                small_component: self.small_component.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                radii: self.radii.view(),
                integrand_power: self.integrand_power,
                orbital_power: self.orbital_power,
                step: self.step,
                coefficient_count: self.coefficient_count,
                active_len: self.active_len,
            }
        }
    }

    fn dsordc_reference_inputs(count: usize) -> DsordcReferenceInputs {
        let step = 0.0725;
        let orbital = 3.0;
        let large_integrand = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.17 * row).sin() + 0.02 * row,
                (0.11 * row).cos() - 0.03 * row,
            )
        }));
        let small_integrand = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.09 * row).cos() - 0.01 * row,
                (0.21 * row).sin() + 0.015 * row,
            )
        }));
        let large_integrand_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                0.04 * row + (0.13 * row).cos(),
                -0.03 * row + (0.17 * row).sin(),
            )
        }));
        let small_integrand_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                -0.02 * row + (0.09 * row).sin(),
                0.025 * row + (0.12 * row).cos(),
            )
        }));
        let large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
        }));
        let small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
        }));
        let large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            0.02 * row + (0.03 * row * orbital).cos()
        }));
        let small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            -0.015 * row + (0.025 * row * orbital).sin()
        }));
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        DsordcReferenceInputs {
            large_integrand,
            small_integrand,
            large_integrand_coefficients,
            small_integrand_coefficients,
            large_component,
            small_component,
            large_coefficients,
            small_coefficients,
            radii,
            integrand_power: 1.35,
            orbital_power: 0.45 + 0.06 * orbital,
            step,
            coefficient_count: 6,
            active_len: count,
        }
    }

    struct OrtdacReferenceInputs {
        target_large_component: Array1<Complex>,
        target_small_component: Array1<Complex>,
        target_large_coefficients: Array1<Complex>,
        target_small_coefficients: Array1<Complex>,
        bound_large_components: Array2<Real>,
        bound_small_components: Array2<Real>,
        bound_large_coefficients: Array2<Real>,
        bound_small_coefficients: Array2<Real>,
        electron_counts: Array1<Real>,
        kappa: Array1<i32>,
        orbital_powers: Array1<Real>,
        radii: Array1<Real>,
        target_power: Real,
        target_kappa: i32,
        step: Real,
        coefficient_count: usize,
        active_len: usize,
        bound_orbital_count: usize,
    }

    impl OrtdacReferenceInputs {
        fn as_orthogonalization_input(&self) -> FovrgOrthogonalizationInput<'_> {
            FovrgOrthogonalizationInput {
                target_large_component: self.target_large_component.view(),
                target_small_component: self.target_small_component.view(),
                target_large_coefficients: self.target_large_coefficients.view(),
                target_small_coefficients: self.target_small_coefficients.view(),
                bound_large_components: self.bound_large_components.view(),
                bound_small_components: self.bound_small_components.view(),
                bound_large_coefficients: self.bound_large_coefficients.view(),
                bound_small_coefficients: self.bound_small_coefficients.view(),
                electron_counts: self.electron_counts.view(),
                kappa: self.kappa.view(),
                orbital_powers: self.orbital_powers.view(),
                radii: self.radii.view(),
                target_power: self.target_power,
                target_kappa: self.target_kappa,
                step: self.step,
                coefficient_count: self.coefficient_count,
                active_len: self.active_len,
                bound_orbital_count: self.bound_orbital_count,
            }
        }
    }

    fn ortdac_reference_inputs(count: usize) -> OrtdacReferenceInputs {
        let step = 0.0725;
        let bound_orbitals = 4;
        let target_large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.17 * row).sin() + 0.02 * row,
                (0.11 * row).cos() - 0.03 * row,
            )
        }));
        let target_small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.09 * row).cos() - 0.01 * row,
                (0.21 * row).sin() + 0.015 * row,
            )
        }));
        let target_large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                0.04 * row + (0.13 * row).cos(),
                -0.03 * row + (0.17 * row).sin(),
            )
        }));
        let target_small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                -0.02 * row + (0.09 * row).sin(),
                0.025 * row + (0.12 * row).cos(),
            )
        }));
        let bound_large_components =
            Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
            });
        let bound_small_components =
            Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
            });
        let bound_large_coefficients =
            Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                0.02 * row + (0.03 * row * orbital).cos()
            });
        let bound_small_coefficients =
            Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                -0.015 * row + (0.025 * row * orbital).sin()
            });
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        OrtdacReferenceInputs {
            target_large_component,
            target_small_component,
            target_large_coefficients,
            target_small_coefficients,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.2, 1.4, 0.0, 2.0]),
            kappa: Array1::from_vec(vec![-2, 1, -2, -2]),
            orbital_powers: Array1::from_iter((1..=bound_orbitals).map(|orbital| {
                let orbital = orbital as Real;
                0.45 + 0.06 * orbital
            })),
            radii,
            target_power: 0.45 + 0.06 * 5.0,
            target_kappa: -2,
            step,
            coefficient_count: 6,
            active_len: count,
            bound_orbital_count: bound_orbitals,
        }
    }

    struct PotexReferenceInputs {
        target_large_component: Array1<Complex>,
        target_small_component: Array1<Complex>,
        target_large_coefficients: Array1<Complex>,
        target_small_coefficients: Array1<Complex>,
        bound_large_components: Array2<Real>,
        bound_small_components: Array2<Real>,
        bound_large_coefficients: Array2<Real>,
        bound_small_coefficients: Array2<Real>,
        angular_coefficients: Array2<Real>,
        orbital_powers: Array1<Real>,
        kappa: Array1<i32>,
        orbital_lengths: Array1<usize>,
        normalization: Array1<Real>,
        radii: Array1<Real>,
        target_power: Real,
        target_kappa: i32,
        target_normalization: Real,
        speed_of_light: Real,
        step: Real,
        coefficient_count: usize,
        source_len: usize,
        active_len: usize,
        radial_output_count: usize,
        bound_orbital_count: usize,
    }

    impl PotexReferenceInputs {
        fn as_exchange_potential_input(&self) -> FovrgExchangePotentialInput<'_> {
            FovrgExchangePotentialInput {
                target_large_component: self.target_large_component.view(),
                target_small_component: self.target_small_component.view(),
                target_large_coefficients: self.target_large_coefficients.view(),
                target_small_coefficients: self.target_small_coefficients.view(),
                bound_large_components: self.bound_large_components.view(),
                bound_small_components: self.bound_small_components.view(),
                bound_large_coefficients: self.bound_large_coefficients.view(),
                bound_small_coefficients: self.bound_small_coefficients.view(),
                angular_coefficients: self.angular_coefficients.view(),
                orbital_powers: self.orbital_powers.view(),
                kappa: self.kappa.view(),
                orbital_lengths: self.orbital_lengths.view(),
                normalization: self.normalization.view(),
                radii: self.radii.view(),
                target_power: self.target_power,
                target_kappa: self.target_kappa,
                target_normalization: self.target_normalization,
                speed_of_light: self.speed_of_light,
                step: self.step,
                coefficient_count: self.coefficient_count,
                source_len: self.source_len,
                active_len: self.active_len,
                radial_output_count: self.radial_output_count,
                bound_orbital_count: self.bound_orbital_count,
            }
        }
    }

    fn potex_reference_inputs(count: usize) -> PotexReferenceInputs {
        let step = 0.0725;
        let bound_orbitals = 4;
        let target_large_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.17 * row).sin() + 0.02 * row,
                (0.11 * row).cos() - 0.03 * row,
            )
        }));
        let target_small_component = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            Complex::new(
                (0.09 * row).cos() - 0.01 * row,
                (0.21 * row).sin() + 0.015 * row,
            )
        }));
        let target_large_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                0.04 * row + (0.13 * row).cos(),
                -0.03 * row + (0.17 * row).sin(),
            )
        }));
        let target_small_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            Complex::new(
                -0.02 * row + (0.09 * row).sin(),
                0.025 * row + (0.12 * row).cos(),
            )
        }));
        let bound_large_components =
            Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
            });
        let bound_small_components =
            Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
            });
        let bound_large_coefficients =
            Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                0.02 * row + (0.03 * row * orbital).cos()
            });
        let bound_small_coefficients =
            Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
                let row = (row + 1) as Real;
                let orbital = (orbital + 1) as Real;
                -0.015 * row + (0.025 * row * orbital).sin()
            });
        let mut angular_coefficients = Array2::zeros((bound_orbitals, 5));
        angular_coefficients[(0, 0)] = 0.31;
        angular_coefficients[(1, 0)] = -0.18;
        angular_coefficients[(2, 0)] = 0.27;
        angular_coefficients[(2, 1)] = -0.11;
        angular_coefficients[(3, 0)] = 0.19;
        angular_coefficients[(3, 1)] = 0.07;
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        PotexReferenceInputs {
            target_large_component,
            target_small_component,
            target_large_coefficients,
            target_small_coefficients,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            angular_coefficients,
            orbital_powers: Array1::from_vec(vec![0.51, 0.57, 0.63, 0.69]),
            kappa: Array1::from_vec(vec![-1, 1, -2, 2]),
            orbital_lengths: Array1::from_vec(vec![9, 8, 7, 9]),
            normalization: Array1::from_vec(vec![1.01, 1.02, 1.03, 1.04]),
            radii,
            target_power: 0.75,
            target_kappa: -2,
            target_normalization: 1.08,
            speed_of_light: 137.035_999_084,
            step,
            coefficient_count: 6,
            source_len: 9,
            active_len: count,
            radial_output_count: 7,
            bound_orbital_count: bound_orbitals,
        }
    }

    struct PotdvpReferenceInputs {
        nuclear_coefficients: Array1<Real>,
        large_coefficients: Array2<Real>,
        small_coefficients: Array2<Real>,
        electron_counts: Array1<Real>,
        kappa: Array1<i32>,
        normalization: Array1<Real>,
        radii: Array1<Real>,
        speed_of_light: Real,
        coefficient_count: usize,
        orbital_count: usize,
    }

    impl PotdvpReferenceInputs {
        fn as_potential_input(&self) -> FovrgPotentialDevelopmentInput<'_> {
            FovrgPotentialDevelopmentInput {
                nuclear_coefficients: self.nuclear_coefficients.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                electron_counts: self.electron_counts.view(),
                kappa: self.kappa.view(),
                normalization: self.normalization.view(),
                radii: self.radii.view(),
                speed_of_light: self.speed_of_light,
                coefficient_count: self.coefficient_count,
                orbital_count: self.orbital_count,
            }
        }
    }

    fn potdvp_reference_inputs(count: usize) -> PotdvpReferenceInputs {
        let step = 0.0725;
        let bound_orbitals = 4;
        let large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            0.02 * row + (0.03 * row * orbital).cos()
        });
        let small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            -0.015 * row + (0.025 * row * orbital).sin()
        });
        let nuclear_coefficients = Array1::from_iter((1..=10).map(|row| {
            let row = row as Real;
            -0.35 + 0.045 * row + 0.002 * row * row
        }));
        let electron_counts = Array1::from_iter((1..=bound_orbitals).map(|orbital| {
            let orbital = orbital as Real;
            0.45 * orbital + 0.1
        }));
        let kappa = Array1::from_vec(vec![-1, 1, -2, 3]);
        let normalization = Array1::from_iter((1..=bound_orbitals).map(|orbital| {
            let orbital = orbital as Real;
            1.0 + 0.013 * orbital
        }));
        let radii = Array1::from_iter((1..=count).map(|row| {
            let row = row as Real;
            0.018 * (step * (row - 1.0)).exp()
        }));

        PotdvpReferenceInputs {
            nuclear_coefficients,
            large_coefficients,
            small_coefficients,
            electron_counts,
            kappa,
            normalization,
            radii,
            speed_of_light: 137.035_999_084,
            coefficient_count: 8,
            orbital_count: 5,
        }
    }

    fn assert_complex_close(
        actual: Complex,
        expected_re: Real,
        expected_im: Real,
        tolerance: Real,
    ) {
        assert_close(actual.re, expected_re, tolerance);
        assert_close(actual.im, expected_im, tolerance);
    }

    fn assert_real_matrix_close<const ROWS: usize, const COLS: usize>(
        actual: &Array2<Real>,
        expected: &[[Real; COLS]; ROWS],
        tolerance: Real,
    ) {
        assert_eq!(actual.shape(), &[ROWS, COLS]);
        for row in 0..ROWS {
            for column in 0..COLS {
                assert_close(actual[(row, column)], expected[row][column], tolerance);
            }
        }
    }

    fn assert_close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}
