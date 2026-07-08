use ndarray::{Array1, ArrayView1, ArrayView2};
use thiserror::Error;

use crate::{Complex, ComplexVec, Real, RealVec, angular::AngularError, bessel::BesselError};

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

/// Inputs for the FEFF `dfovrg` C3 correction potential `vm`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgC3PotentialInput<'a> {
    /// Interstitial-flattened exchange-correlation potential.
    pub exchange_correlation_potential: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Target photoelectron kappa `kap(norb)`.
    pub target_kappa: i32,
    /// Logarithmic grid step `dx`.
    pub step: Real,
    /// Zero-based equivalent of FEFF `jri`.
    pub radial_match_index: usize,
    /// Number of active `wfirdc` radial rows.
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

/// Inputs for FEFF `FOVRG/wfirdc.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgInitialPhotoelectronInput<'a> {
    /// One-electron photoelectron energy `eph`.
    pub energy: Complex,
    /// Bound-orbital large origin coefficients `bg(coefficient, orbital)`.
    pub bound_large_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital small origin coefficients `bp(coefficient, orbital)`.
    pub bound_small_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital occupations `xnel`; target photoelectron is excluded.
    pub electron_counts: ArrayView1<'a, Real>,
    /// Relativistic kappa values `kap`; the target photoelectron is the last active row.
    pub kappa: ArrayView1<'a, i32>,
    /// FEFF `nmax` tabulation lengths; the target row is clamped to the retained mesh length.
    pub orbital_lengths: ArrayView1<'a, usize>,
    /// Initial LDA potential `vxc` for the photoelectron.
    pub exchange_correlation_potential: ArrayView1<'a, Complex>,
    /// C3 correction potential `vm`.
    pub c3_potential: ArrayView1<'a, Complex>,
    /// Initial large coefficient `aps(1)` for irregular solutions.
    pub initial_large_coefficient: Complex,
    /// Initial small coefficient `aqs(1)` for irregular solutions.
    pub initial_small_coefficient: Complex,
    /// Nuclear charge `nz`.
    pub nuclear_charge: Real,
    /// Muffin-tin radius `rmt`; retained for call-shape compatibility.
    pub muffin_tin_radius: Real,
    /// Logarithmic grid step `hx`.
    pub step: Real,
    /// Speed of light `cl`; FEFF `wfirdc` uses `137.0373`.
    pub speed_of_light: Real,
    /// FEFF `ic3` switch or scale for the C3 term.
    pub c3_scale: i32,
    /// Whether to compute the irregular inward solution (`irr > 0`).
    pub irregular: bool,
    /// Zero-based equivalent of FEFF `jri`.
    pub radial_match_index: usize,
    /// Zero-based equivalent of FEFF `iwkb`.
    pub wkb_index: usize,
    /// Number of active origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// Number of active orbitals `norb`; the target photoelectron is `orbital_count - 1`.
    pub orbital_count: usize,
    /// Number of radial rows `idim`.
    pub active_len: usize,
}

/// Inputs for FEFF `FOVRG/dfovrg.f90`.
///
/// This high-level driver keeps FEFF's radial-solver behavior but makes the
/// hidden atomic common-block state explicit. Bound orbitals and occupations
/// are supplied by the caller; the target photoelectron orbital is appended
/// internally with [`FovrgDiracSolverInput::target_kappa`].
#[derive(Debug, Clone, Copy)]
pub struct FovrgDiracSolverInput<'a> {
    /// FEFF `ncycle`: exchange iterations run `ncycle + 1` times when nonzero.
    pub exchange_cycle_count: usize,
    /// Photoelectron relativistic kappa `ikap`.
    pub target_kappa: i32,
    /// Muffin-tin radius `rmt`.
    pub muffin_tin_radius: Real,
    /// Zero-based equivalent of incoming FEFF `jlast`.
    pub target_last_index: usize,
    /// FEFF `p2`, the complex one-electron energy.
    pub energy: Complex,
    /// Logarithmic Loucks grid step `dx`.
    pub step: Real,
    /// Loucks radial grid `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// Total-density Coulomb plus exchange-correlation potential `vxc`.
    pub exchange_correlation_potential: ArrayView1<'a, Complex>,
    /// Valence-density Coulomb plus exchange-correlation potential `vxcval`.
    pub valence_exchange_correlation_potential: ArrayView1<'a, Complex>,
    /// Bound-orbital large components `dgcn(row, orbital)`.
    pub bound_large_components: ArrayView2<'a, Real>,
    /// Bound-orbital small components `dpcn(row, orbital)`.
    pub bound_small_components: ArrayView2<'a, Real>,
    /// Bound-orbital large origin coefficients `adgc(coefficient, orbital)`.
    pub bound_large_coefficients: ArrayView2<'a, Real>,
    /// Bound-orbital small origin coefficients `adpc(coefficient, orbital)`.
    pub bound_small_coefficients: ArrayView2<'a, Real>,
    /// Total bound-orbital occupations before FEFF subtracts valence counts.
    pub electron_counts: ArrayView1<'a, Real>,
    /// Valence occupations `xnval`; positive rows are skipped by exchange.
    pub valence_counts: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values.
    pub kappa: ArrayView1<'a, i32>,
    /// Input/output regular muffin-tin large component `pu`.
    pub muffin_tin_large_component: Complex,
    /// Input/output regular muffin-tin small component `qu`.
    pub muffin_tin_small_component: Complex,
    /// Nuclear charge `iz`.
    pub atomic_number: Real,
    /// Whether to compute the irregular inward solution (`irr > 0`).
    pub irregular: bool,
    /// FEFF `ic3` switch or scale for the C3 term.
    pub c3_scale: i32,
    /// Zero-based equivalent of FEFF `jri`.
    pub radial_match_index: usize,
    /// Number of explicitly supplied bound orbitals.
    pub bound_orbital_count: usize,
}

/// Inputs for the orbital bookkeeping portion of FEFF `FOVRG/inmuac.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgOrbitalSetupInput<'a> {
    /// Bound-orbital large radial components `cg(row, orbital)`.
    pub bound_large_components: ArrayView2<'a, Real>,
    /// Bound-orbital small radial components `cp(row, orbital)`.
    pub bound_small_components: ArrayView2<'a, Real>,
    /// Total bound-orbital occupations `xnel`.
    pub electron_counts: ArrayView1<'a, Real>,
    /// Valence occupations `xnval`.
    pub valence_counts: ArrayView1<'a, Real>,
    /// Bound-orbital relativistic kappa values `kap`.
    pub kappa: ArrayView1<'a, i32>,
    /// Target photoelectron kappa appended as FEFF `kap(norb)`.
    pub target_kappa: i32,
    /// FEFF `idim`, the active radial block length.
    pub active_len: usize,
    /// Number of explicitly supplied bound orbitals.
    pub bound_orbital_count: usize,
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

/// Output from FEFF `FOVRG/wfirdc.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgInitialPhotoelectron {
    /// Photoelectron large radial component `ps`.
    pub large_component: ComplexVec,
    /// Photoelectron small radial component `qs`.
    pub small_component: ComplexVec,
    /// Photoelectron large origin coefficients `aps`.
    pub large_coefficients: ComplexVec,
    /// Photoelectron small origin coefficients `aqs`.
    pub small_coefficients: ComplexVec,
    /// Origin powers `fl` for all active orbitals.
    pub origin_powers: RealVec,
    /// FEFF normalization factors `fix` for all active orbitals.
    pub normalization: RealVec,
    /// Clamped `nmax` values for all active orbitals.
    pub orbital_lengths: Array1<usize>,
    /// Nuclear radial mesh and potential from `nucdec`.
    pub nuclear_potential: FovrgNuclearPotential,
    /// Direct photoelectron potential `dv` after FEFF's division by `cl`.
    pub direct_potential: ComplexVec,
    /// Direct-potential origin coefficients `av`.
    pub potential_coefficients: ComplexVec,
    /// FEFF `np`, the retained radial mesh length.
    pub retained_len: usize,
    /// Zero-based target tabulation endpoint after `nmax(norb)` clamping.
    pub target_last_index: usize,
    /// Count of difficult Milne iterations reported by the selected radial solver.
    pub difficult_iterations: usize,
}

/// Output from FEFF `FOVRG/dfovrg.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgDiracSolution {
    /// Photoelectron large radial component `ps`.
    pub large_component: ComplexVec,
    /// Photoelectron small radial component `qs`.
    pub small_component: ComplexVec,
    /// Photoelectron large origin coefficients `aps`.
    pub large_coefficients: ComplexVec,
    /// Photoelectron small origin coefficients `aqs`.
    pub small_coefficients: ComplexVec,
    /// Muffin-tin large component `pu`.
    pub muffin_tin_large_component: Complex,
    /// Muffin-tin small component `qu`.
    pub muffin_tin_small_component: Complex,
    /// Mutated total-density potential after FEFF's interstitial flattening.
    pub exchange_correlation_potential: ComplexVec,
    /// Mutated valence potential after FEFF's interstitial flattening.
    pub valence_exchange_correlation_potential: ComplexVec,
    /// Direct photoelectron potential `dv`.
    pub direct_potential: ComplexVec,
    /// Direct-potential origin coefficients `av`.
    pub potential_coefficients: ComplexVec,
    /// Large-component exchange potential `eg`.
    pub large_exchange: ComplexVec,
    /// Small-component exchange potential `ep`.
    pub small_exchange: ComplexVec,
    /// Large-component exchange origin coefficients `ceg`.
    pub large_exchange_coefficients: ComplexVec,
    /// Small-component exchange origin coefficients `cep`.
    pub small_exchange_coefficients: ComplexVec,
    /// C3 correction potential `vm`.
    pub c3_potential: ComplexVec,
    /// Origin powers `fl` for bound orbitals plus the photoelectron.
    pub origin_powers: RealVec,
    /// FEFF normalization factors `fix`.
    pub normalization: RealVec,
    /// Clamped `nmax` tabulation lengths.
    pub orbital_lengths: Array1<usize>,
    /// FEFF `idim`, the active radial block length.
    pub active_len: usize,
    /// FEFF `np`, the retained radial mesh length.
    pub retained_len: usize,
    /// Zero-based equivalent of FEFF `iwkb`.
    pub wkb_index: usize,
    /// Zero-based target endpoint after FEFF `nmax(norb)` clamping.
    pub target_last_index: usize,
    /// Number of nonlocal-exchange resolution iterations that were run.
    pub iteration_count: usize,
    /// Count of difficult Milne iterations reported by nested solvers.
    pub difficult_iterations: usize,
}

/// Output from the orbital bookkeeping portion of FEFF `FOVRG/inmuac.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgOrbitalSetup {
    /// Bound-orbital lengths plus a zero placeholder for the target orbital.
    pub orbital_lengths: Array1<usize>,
    /// Bound kappa values plus the appended target photoelectron kappa.
    pub kappa: Array1<i32>,
    /// Core occupations after subtracting valence occupations.
    pub core_counts: RealVec,
    /// FEFF `nre > 0` open-shell flags for each bound orbital.
    pub open_shell: Array1<bool>,
    /// FEFF `ipl`, the count of bound orbitals with the target kappa.
    pub matching_kappa_count: usize,
}

/// Error returned by FOVRG helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
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
    /// Square-root radicands must be nonnegative.
    #[error("FOVRG {name} row {row} radicand must be nonnegative, got {value}")]
    NegativeRadicand {
        name: &'static str,
        row: usize,
        value: Real,
    },
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
