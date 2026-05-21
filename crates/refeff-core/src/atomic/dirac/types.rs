use ndarray::{Array1, ArrayView1};

use crate::Real;

/// Inputs for FEFF `ATOM/soldir.f90` entry-state setup.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracEntryStateInput {
    /// Asymptotic large-component seed `ainf`.
    pub asymptotic_large_component: Real,
    /// Requested FEFF solution method.
    pub method: i32,
}

/// FEFF `soldir` state initialized before label `101`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracEntryState {
    /// Initial previous-energy reference, FEFF `enav = 1`.
    pub previous_energy: Real,
    /// Nonnegative asymptotic large-component seed, FEFF `ainf = abs(ainf)`.
    pub asymptotic_large_component: Real,
    /// Original method request, FEFF `iex`.
    pub requested_method: i32,
    /// Effective method after FEFF maps `method <= 0` to method 1.
    pub method: i32,
}

/// Inputs for FEFF `ATOM/soldir.f90` `norm`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracNormalizationInput<'a> {
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// FEFF solution method selector; only `1` applies the matching correction.
    pub method: i32,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Inward small-component value at the matching point, FEFF `gpmat`.
    pub matching_small_component: Real,
    /// First origin-development power `fl`.
    pub origin_power: Real,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` normalization integral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracNormalization {
    /// Normalization integral `b` before `soldir` takes its square root.
    pub norm: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` final wavefunction normalization.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracSolutionNormalizationInput<'a> {
    /// Normalization integral `b` before FEFF takes its square root.
    pub norm: Real,
    /// Initial large origin coefficient `agi`, used only for FEFF's sign rule.
    pub initial_large_coefficient: Real,
    /// Initial small origin coefficient `api`, used only for FEFF's sign rule.
    pub initial_small_coefficient: Real,
    /// Large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
}

/// FEFF `soldir` normalized Dirac solution.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracSolutionNormalization {
    /// Normalized large component `gg`; rows after `active_len` are zeroed.
    pub large_component: Array1<Real>,
    /// Normalized small component `gp`; rows after `active_len` are zeroed.
    pub small_component: Array1<Real>,
    /// Normalized large origin-development coefficients `ag`.
    pub large_coefficients: Array1<Real>,
    /// Normalized small origin-development coefficients `ap`.
    pub small_coefficients: Array1<Real>,
    /// Signed divisor used for radial components, FEFF `b` after sign adjustment.
    pub component_divisor: Real,
    /// Signed divisor used for origin coefficients, FEFF `c`.
    pub coefficient_divisor: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` radial node counting.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracNodeCountInput<'a> {
    /// Large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
    /// One-based candidate scan limit `j`; FEFF scans through `max(j, mat)`.
    pub scan_index_1based: usize,
}

/// FEFF `soldir` radial node count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDiracNodeCount {
    /// FEFF `nd`, starting from one and counting sign changes through the scan.
    pub node_count: usize,
    /// Effective one-based scan limit, `max(j, mat)`.
    pub scan_index_1based: usize,
}

/// Inputs for FEFF `ATOM/soldir.f90` node-count energy search.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracNodeEnergySearchInput {
    /// Current trial energy `en`.
    pub energy: Real,
    /// Current node count `nd`.
    pub node_count: usize,
    /// Target FEFF node count `node`.
    pub target_node_count: usize,
    /// FEFF upper bracket variable `esup`.
    pub energy_sup: Real,
    /// FEFF lower bracket variable `einf`.
    pub energy_inf: Real,
    /// Apparent-potential minimum `emin`.
    pub energy_floor: Real,
    /// Energy precision cutoff `test1`.
    pub energy_precision: Real,
    /// Current node-search attempt count `jes`.
    pub search_attempt_count: usize,
    /// Maximum node-search attempts `nes`.
    pub max_attempt_count: usize,
}

/// Result of FEFF `soldir` node-count energy search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracNodeEnergySearch {
    /// Updated trial energy `en`.
    pub energy: Real,
    /// Updated FEFF upper bracket variable `esup`.
    pub energy_sup: Real,
    /// Updated FEFF lower bracket variable `einf`.
    pub energy_inf: Real,
    /// Updated attempt count `jes`.
    pub search_attempt_count: usize,
    /// Whether FEFF would return to the integration loop at label `106`.
    pub needs_reintegration: bool,
    /// Whether FEFF would set `ifail = 1` and continue with the current solution.
    pub attempts_exhausted: bool,
}

/// Inputs for FEFF `ATOM/soldir.f90` restart state at labels `101` and `105`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracIterationResetInput {
    /// Current effective FEFF method.
    pub method: i32,
    /// Primary small-component matching precision `test1`.
    pub primary_matching_precision: Real,
    /// Secondary normalization precision `test2`.
    pub secondary_matching_precision: Real,
    /// Apparent-potential minimum `emin`.
    pub energy_floor: Real,
    /// Stored trial energy restored at label `101`, FEFF `edep`.
    pub reference_energy: Real,
}

/// FEFF `soldir` state after restarting an iteration block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracIterationReset {
    /// Active mismatch tolerance `test`.
    pub mismatch_precision: Real,
    /// Restored trial energy `en`.
    pub energy: Real,
    /// Reset lower energy bracket `einf`.
    pub energy_inf: Real,
    /// Reset upper energy bracket `esup`.
    pub energy_sup: Real,
    /// Reset small-component matching attempt count `ies`.
    pub match_attempt_count: usize,
    /// Reset node count `nd`.
    pub node_count: usize,
    /// Reset node-search attempt count `jes`.
    pub search_attempt_count: usize,
}

/// Inputs for FEFF `ATOM/soldir.f90` abnormal-exit recovery at label `899`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracAbnormalExitRecoveryInput {
    /// Originally requested method, FEFF `iex`.
    pub requested_method: i32,
    /// Current effective method.
    pub method: i32,
}

/// FEFF `soldir` abnormal-exit recovery decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDiracAbnormalExitRecovery {
    /// Method after the recovery branch.
    pub method: i32,
    /// Whether FEFF would jump back to label `101`.
    pub needs_restart: bool,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-1 energy correction.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracMethodOneEnergyCorrectionInput<'a> {
    /// FEFF speed of light `cl`.
    pub speed_of_light: Real,
    /// FEFF normalization integral `b` before the final square root.
    pub norm: Real,
    /// Large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Inward small-component value at the matching point, FEFF `gpmat`.
    pub matching_small_component: Real,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` energy correction scalars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracEnergyCorrection {
    /// Additive energy correction `f`.
    pub correction: Real,
    /// Small-component mismatch `c`, relative to `gpmat` when FEFF scales it.
    pub mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` energy-correction backtracking.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracEnergyStepInput {
    /// Trial energy before applying `correction`, FEFF `en`.
    pub energy: Real,
    /// Additive energy correction `f`.
    pub correction: Real,
    /// Small-component or normalization mismatch `c`.
    pub mismatch: Real,
    /// FEFF `esup` search bracket value.
    pub energy_sup: Real,
    /// FEFF `einf` search bracket value.
    pub energy_inf: Real,
    /// Active mismatch tolerance, FEFF `test`.
    pub mismatch_precision: Real,
    /// Lower relative-step cutoff, FEFF `test1`.
    pub zero_energy_precision: Real,
}

/// Result of FEFF `soldir` energy-correction backtracking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracEnergyStep {
    /// Updated trial energy `en`.
    pub energy: Real,
    /// Possibly halved correction `f`.
    pub correction: Real,
    /// FEFF relative step `g`.
    pub relative_step: Real,
    /// Whether FEFF would continue matching the small component.
    pub needs_rematch: bool,
}

/// Inputs for FEFF `ATOM/soldir.f90` small-component rematch attempt handling.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracRematchAttemptInput {
    /// Small-component or normalization mismatch `c`.
    pub mismatch: Real,
    /// Active mismatch tolerance, FEFF `test`.
    pub mismatch_precision: Real,
    /// Current small-component matching attempt count `ies`.
    pub match_attempt_count: usize,
    /// Maximum matching attempts `nes`.
    pub max_attempt_count: usize,
}

/// FEFF `soldir` small-component rematch attempt result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDiracRematchAttempt {
    /// Updated small-component matching attempt count `ies`.
    pub match_attempt_count: usize,
    /// Whether FEFF would jump back to label `105`.
    pub needs_rematch: bool,
    /// Whether FEFF would set `ifail = 1` and continue with the current solution.
    pub attempts_exhausted: bool,
}

/// Inputs for FEFF `ATOM/soldir.f90` homogeneous-system tail matching.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracHomogeneousMatchInput<'a> {
    /// Homogeneous large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Homogeneous small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Inward large-component value at the matching point, FEFF `ggmat`.
    pub matching_large_component: Real,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` homogeneous-system tail matching result.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracHomogeneousMatch {
    /// Matched large component `gg`.
    pub large_component: Array1<Real>,
    /// Matched small component `gp`.
    pub small_component: Array1<Real>,
    /// Scale applied from `mat` through `max0`, FEFF `a`.
    pub tail_scale: Real,
    /// One-based scan index set for later node counting, FEFF `j = mat`.
    pub scan_index_1based: usize,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-1 large-component matching.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracLargeComponentMatchInput<'a> {
    /// Inhomogeneous large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Inhomogeneous small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Homogeneous large radial component `hg`.
    pub homogeneous_large_component: ArrayView1<'a, Real>,
    /// Homogeneous small radial component `hp`.
    pub homogeneous_small_component: ArrayView1<'a, Real>,
    /// Inward large-component value at the matching point, FEFF `ggmat`.
    pub matching_large_component: Real,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` large-component matching result.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracLargeComponentMatch {
    /// Matched large component `gg`.
    pub large_component: Array1<Real>,
    /// Matched small component `gp`.
    pub small_component: Array1<Real>,
    /// Homogeneous tail scale, FEFF `b`.
    pub tail_scale: Real,
    /// Large-component mismatch before matching, FEFF `a`.
    pub large_mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-2 two-component matching.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracTwoComponentMatchInput<'a> {
    /// Inhomogeneous large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Inhomogeneous small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Inhomogeneous large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Inhomogeneous small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Homogeneous large radial component `hg`.
    pub homogeneous_large_component: ArrayView1<'a, Real>,
    /// Homogeneous small radial component `hp`.
    pub homogeneous_small_component: ArrayView1<'a, Real>,
    /// Homogeneous large origin-development coefficients `agh`.
    pub homogeneous_large_coefficients: ArrayView1<'a, Real>,
    /// Homogeneous small origin-development coefficients `aph`.
    pub homogeneous_small_coefficients: ArrayView1<'a, Real>,
    /// Inward large-component value at the matching point, FEFF `ggmat`.
    pub matching_large_component: Real,
    /// Inward small-component value at the matching point, FEFF `gpmat`.
    pub matching_small_component: Real,
    /// Homogeneous inward large-component value at the matching point, FEFF `hgmat`.
    pub homogeneous_matching_large_component: Real,
    /// Homogeneous inward small-component value at the matching point, FEFF `hpmat`.
    pub homogeneous_matching_small_component: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` two-component matching result.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracTwoComponentMatch {
    /// Matched large component `gg`.
    pub large_component: Array1<Real>,
    /// Matched small component `gp`.
    pub small_component: Array1<Real>,
    /// Matched large origin-development coefficients `ag`.
    pub large_coefficients: Array1<Real>,
    /// Matched small origin-development coefficients `ap`.
    pub small_coefficients: Array1<Real>,
    /// Determinant of the matching system, FEFF `ah`.
    pub determinant: Real,
    /// Homogeneous scale applied before `mat`, FEFF `c`.
    pub prefix_scale: Real,
    /// Homogeneous scale applied from `mat` through `max0`, FEFF `b`.
    pub tail_scale: Real,
    /// Large-component mismatch before matching, FEFF `a`.
    pub large_mismatch: Real,
    /// Small-component mismatch before matching, FEFF `b` before reuse.
    pub small_mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-2 energy-disagreement matching.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracEnergyDisagreementMatchInput<'a> {
    /// Energy-derivative large radial component `bg`.
    pub large_derivative: ArrayView1<'a, Real>,
    /// Energy-derivative small radial component `bp`.
    pub small_derivative: ArrayView1<'a, Real>,
    /// Energy-derivative large origin-development coefficients `bgh`.
    pub large_derivative_coefficients: ArrayView1<'a, Real>,
    /// Energy-derivative small origin-development coefficients `bph`.
    pub small_derivative_coefficients: ArrayView1<'a, Real>,
    /// Homogeneous large radial component `hg`.
    pub homogeneous_large_component: ArrayView1<'a, Real>,
    /// Homogeneous small radial component `hp`.
    pub homogeneous_small_component: ArrayView1<'a, Real>,
    /// Homogeneous large origin-development coefficients `agh`.
    pub homogeneous_large_coefficients: ArrayView1<'a, Real>,
    /// Homogeneous small origin-development coefficients `aph`.
    pub homogeneous_small_coefficients: ArrayView1<'a, Real>,
    /// Inward large derivative value at the matching point, FEFF `bgmat`.
    pub matching_large_derivative: Real,
    /// Inward small derivative value at the matching point, FEFF `bpmat`.
    pub matching_small_derivative: Real,
    /// Homogeneous inward large-component value at the matching point, FEFF `hgmat`.
    pub homogeneous_matching_large_component: Real,
    /// Homogeneous inward small-component value at the matching point, FEFF `hpmat`.
    pub homogeneous_matching_small_component: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` matched method-2 energy-derivative solution.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracEnergyDisagreementMatch {
    /// Matched large derivative component `bg`.
    pub large_derivative: Array1<Real>,
    /// Matched small derivative component `bp`.
    pub small_derivative: Array1<Real>,
    /// Matched large derivative coefficients `bgh`.
    pub large_derivative_coefficients: Array1<Real>,
    /// Matched small derivative coefficients `bph`.
    pub small_derivative_coefficients: Array1<Real>,
    /// Determinant of the matching system, FEFF `ah`.
    pub determinant: Real,
    /// Homogeneous scale applied before `mat`, FEFF `a`.
    pub prefix_scale: Real,
    /// Homogeneous scale applied from `mat` through `max0`, FEFF reused `g`.
    pub tail_scale: Real,
    /// Large derivative mismatch before matching, FEFF `f`.
    pub large_mismatch: Real,
    /// Small derivative mismatch before matching, FEFF initial `g`.
    pub small_mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-2 energy-disagreement source.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracEnergyDisagreementSourceInput<'a> {
    /// Current large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Current small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Current large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Current small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// FEFF speed of light `cl`.
    pub speed_of_light: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
}

/// FEFF `soldir` method-2 energy-disagreement source arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracEnergyDisagreementSource {
    /// Large derivative source `bg`.
    pub large_source: Array1<Real>,
    /// Small derivative source `bp`.
    pub small_source: Array1<Real>,
    /// Large derivative source coefficients `bgh`.
    pub large_coefficients: Array1<Real>,
    /// Small derivative source coefficients `bph`.
    pub small_coefficients: Array1<Real>,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-2 energy correction.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracEnergyDisagreementCorrectionInput<'a> {
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Current large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Current small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Matched large energy-derivative solution `bg`.
    pub large_derivative: ArrayView1<'a, Real>,
    /// Matched small energy-derivative solution `bp`.
    pub small_derivative: ArrayView1<'a, Real>,
    /// Current large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Current small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Matched large derivative coefficients `bgh`.
    pub large_derivative_coefficients: ArrayView1<'a, Real>,
    /// Matched small derivative coefficients `bph`.
    pub small_derivative_coefficients: ArrayView1<'a, Real>,
    /// FEFF normalization integral `b`.
    pub norm: Real,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// First origin-development power `fl`.
    pub origin_power: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
}

/// FEFF `soldir` method-2 energy-corrected Dirac solution.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracEnergyDisagreementCorrection {
    /// Corrected large component `gg`.
    pub large_component: Array1<Real>,
    /// Corrected small component `gp`.
    pub small_component: Array1<Real>,
    /// Corrected large origin-development coefficients `ag`.
    pub large_coefficients: Array1<Real>,
    /// Corrected small origin-development coefficients `ap`.
    pub small_coefficients: Array1<Real>,
    /// Cross integral `ah` used by FEFF's method-2 energy correction.
    pub overlap_integral: Real,
    /// Additive energy correction `f`.
    pub correction: Real,
    /// Normalization mismatch `c = 1 - b`.
    pub normalization_mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` matching-point relocation.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracMatchingPointUpdateInput<'a> {
    /// Current large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// Current one-based matching-point index `mat`.
    pub matching_index_1based: usize,
    /// Whether FEFF has already relocated `mat`, FEFF `modmat != 0`.
    pub already_relocated: bool,
}

/// FEFF `soldir` matching-point relocation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDiracMatchingPointUpdate {
    /// Updated one-based matching-point index `mat`.
    pub matching_index_1based: usize,
    /// One-based peak index found by scanning `gg(i)^2`.
    pub peak_index_1based: usize,
    /// One-based scan limit that later node counting uses, FEFF `max(j, mat)`.
    pub scan_index_1based: usize,
    /// Final relocation flag, FEFF `modmat != 0`.
    pub relocated: bool,
    /// Whether FEFF would reintegrate with the updated matching point.
    pub needs_reintegration: bool,
}

/// Inputs for FEFF `ATOM/soldir.f90` inhomogeneous `intdir` seed setup.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracInhomogeneousSeedInput<'a> {
    /// Large-component exchange/source rows, FEFF `eg`.
    pub large_source: ArrayView1<'a, Real>,
    /// Small-component exchange/source rows, FEFF `ep`.
    pub small_source: ArrayView1<'a, Real>,
    /// Large-source origin coefficients, FEFF `ceg`.
    pub large_source_coefficients: ArrayView1<'a, Real>,
    /// Small-source origin coefficients, FEFF `cep`.
    pub small_source_coefficients: ArrayView1<'a, Real>,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
}

/// Inputs for FEFF `ATOM/soldir.f90` homogeneous `intdir` seed setup.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracHomogeneousSeedInput {
    /// Number of radial rows in `hg/hp`.
    pub radial_len: usize,
    /// Number of origin-development coefficient slots in `agh/aph`.
    pub coefficient_len: usize,
}

/// Seed arrays passed to FEFF `intdir` from `soldir`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracIntegrationSeed {
    /// Initial large component/source array.
    pub large_source: Array1<Real>,
    /// Initial small component/source array.
    pub small_source: Array1<Real>,
    /// Initial large origin-development/source coefficients.
    pub large_coefficients: Array1<Real>,
    /// Initial small origin-development/source coefficients.
    pub small_coefficients: Array1<Real>,
}

/// FEFF `ATOM/soldir.f90` branch after the first inhomogeneous `intdir` pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicDiracInhomogeneousBranchAction {
    /// Original homogeneous request (`iex == 0`): match the integrated tail.
    MatchHomogeneousTail,
    /// Original inhomogeneous request (`iex != 0`): integrate a homogeneous correction.
    IntegrateHomogeneousSystem,
}

/// Inputs for FEFF `ATOM/soldir.f90` branch after the inhomogeneous pass.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracInhomogeneousBranchInput {
    /// Originally requested method, FEFF `iex`.
    pub requested_method: i32,
}

/// FEFF `soldir` action after the inhomogeneous `intdir` pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDiracInhomogeneousBranch {
    /// Next high-level `soldir` action.
    pub action: AtomicDiracInhomogeneousBranchAction,
}

/// FEFF `ATOM/intdir.f90` integration branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicDiracIntegrationMode {
    /// FEFF `imm = 0`: search the matching point, then integrate outward and inward.
    SearchMatchingPoint,
    /// FEFF `imm > 0`: use the supplied matching point and integrate outward and inward.
    FixedMatchingPoint,
    /// FEFF `imm < 0`: skip the outward pass and integrate inward only.
    InwardOnly,
}

/// Inputs for FEFF `ATOM/soldir.f90` homogeneous `intdir` pass setup.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracHomogeneousPassSetupInput {
    /// Current effective FEFF method.
    pub method: i32,
}

/// FEFF `soldir` homogeneous `intdir` pass setup state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDiracHomogeneousPassSetup {
    /// Integration mode selected from FEFF `imm`.
    pub integration_mode: AtomicDiracIntegrationMode,
    /// Raw FEFF integration flag, `imm`.
    pub raw_integration_flag: i32,
}

/// Inputs for FEFF `ATOM/soldir.f90` shooting-pass setup at label `106`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracShootingPassSetupInput {
    /// Current trial energy `en`.
    pub energy: Real,
    /// Previous reference energy used by FEFF's `imm` heuristic, `enav`.
    pub previous_energy: Real,
}

/// FEFF `soldir` shooting-pass setup state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracShootingPassSetup {
    /// Integration mode selected from FEFF `imm`.
    pub integration_mode: AtomicDiracIntegrationMode,
    /// Updated reference energy, FEFF `enav = en`.
    pub reference_energy: Real,
    /// Relative energy change used by FEFF's fixed-matching heuristic.
    pub relative_energy_change: Real,
    /// Matching-point relocation flag reset at label `106`, FEFF `modmat = 0`.
    pub relocated: bool,
}

/// Inputs for FEFF `ATOM/intdir.f90`.
///
/// Component and potential vectors use FEFF radial order. Only the first
/// [`AtomicDiracIntegrationInput::active_len`] rows participate in the
/// integration, but the returned arrays preserve the input capacity and update
/// the rows touched by FEFF's predictor-corrector sweep.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracIntegrationInput<'a> {
    /// Initial large-component source `gg`; overwritten with the integrated solution.
    pub large_source: ArrayView1<'a, Real>,
    /// Initial small-component source `gp`; overwritten with the integrated solution.
    pub small_source: ArrayView1<'a, Real>,
    /// Initial large origin-development/source coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Initial small origin-development/source coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Direct potential `dv` in FEFF `intdir` units.
    pub potential: ArrayView1<'a, Real>,
    /// Potential origin-development coefficients `av`.
    pub potential_coefficients: ArrayView1<'a, Real>,
    /// One-electron energy `en`.
    pub energy: Real,
    /// First origin-development power `fl`.
    pub origin_power: Real,
    /// Initial large origin coefficient `agi`.
    pub initial_large_coefficient: Real,
    /// Initial small origin coefficient `api`.
    pub initial_small_coefficient: Real,
    /// Initial large-component tail amplitude `ainf`.
    pub asymptotic_large_component: Real,
    /// Relativistic kappa `kap`; FEFF stores this as `fk`.
    pub kappa: i32,
    /// Speed of light `cl` in atomic units.
    pub speed_of_light: Real,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Matching-point/tail precision `test1`.
    pub matching_precision: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Active radial row count `np`.
    pub active_len: usize,
    /// Integration branch corresponding to FEFF `imm`.
    pub mode: AtomicDiracIntegrationMode,
    /// One-based matching point `mat` for fixed and inward-only modes.
    pub matching_index_1based: usize,
    /// One-based inward start `max0` for fixed and inward-only modes.
    pub max_index_1based: usize,
}

/// Result of FEFF `ATOM/intdir.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracIntegration {
    /// Integrated large component `gg`.
    pub large_component: Array1<Real>,
    /// Integrated small component `gp`.
    pub small_component: Array1<Real>,
    /// Updated large origin-development coefficients `ag`.
    pub large_coefficients: Array1<Real>,
    /// Updated small origin-development coefficients `ap`.
    pub small_coefficients: Array1<Real>,
    /// Outward large-component value at the matching point, FEFF `ggmat`.
    pub matching_large_component: Option<Real>,
    /// Outward small-component value at the matching point, FEFF `gpmat`.
    pub matching_small_component: Option<Real>,
    /// Final one-based matching-point index `mat`.
    pub matching_index_1based: usize,
    /// Final one-based inward-start index `max0`.
    pub max_index_1based: usize,
}

/// Inputs for FEFF `ATOM/soldir.f90` setup before the first integration pass.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracSolverSetupInput<'a> {
    /// Trial one-electron energy `en`.
    pub energy: Real,
    /// First origin-development power `fl`.
    pub origin_power: Real,
    /// Initial large origin coefficient `agi`.
    pub initial_large_coefficient: Real,
    /// Initial small origin coefficient `api`.
    pub initial_small_coefficient: Real,
    /// Principal quantum number `nq`.
    pub principal_quantum_number: usize,
    /// Relativistic kappa `kap`.
    pub kappa: i32,
    /// Speed of light `cl` in atomic units.
    pub speed_of_light: Real,
    /// Initial FEFF method selector; `<= 0` falls back to method `1`.
    pub method: i32,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Direct potential `dv`.
    pub potential: ArrayView1<'a, Real>,
    /// Potential origin coefficients `av`; only `av(1)` participates here.
    pub potential_coefficients: ArrayView1<'a, Real>,
    /// Number of active radial rows `np`.
    pub active_len: usize,
}

/// Deterministic `soldir` setup values shared by later integration passes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracSolverSetup {
    /// FEFF `iex`: the originally requested method before fallback.
    pub requested_method: i32,
    /// Effective FEFF `method` after `method <= 0` fallback.
    pub method: i32,
    /// Trial energy after clamping below the apparent-potential floor.
    pub energy: Real,
    /// FEFF `emin`, the minimum apparent potential energy.
    pub energy_floor: Real,
    /// Initial small origin coefficient after point-nucleus correction.
    pub initial_small_coefficient: Real,
    /// Angular coefficient `ell = kappa * (kappa + 1) / (2 * cl)`.
    pub angular_term: Real,
    /// Target radial node count.
    pub target_nodes: i32,
    /// Twice the speed of light, FEFF `ccl`.
    pub doubled_speed_of_light: Real,
}
