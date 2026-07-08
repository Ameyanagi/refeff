use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use refeff_linalg::LinalgError;
use thiserror::Error;

use crate::{
    AngularError, BesselError, Complex, ConvolutionError, ExcitationPole, FovrgDiracSolution,
    FovrgDiracSolverInput, FovrgError, FovrgInitialPhotoelectron, FovrgInitialPhotoelectronInput,
    GridError, InterpolationError, PhaseError, QuadratureError, Real,
};

/// Number of columns returned by [`crate::xsph::xsph_axafs`].
pub const XSPH_AXAFS_COLUMN_COUNT: usize = 6;

/// Shared final-state calculation plan returned by [`crate::xsph::xsph_minimize_calculations`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsphCalculationPlan {
    /// Maximum `lj` encountered in the active final-state index list, FEFF `ljj`.
    pub max_lj: i32,
    /// Rows `[kind, max_lj_for_kind, representative_l]`, FEFF `indcalc`.
    pub calculations: Array2<i32>,
    /// Per-final-state map to a calculation row, FEFF `indmap`.
    ///
    /// Positive values mark the first occurrence of a final-state `kind`.
    /// Negative values reuse the absolute calculation index from an earlier
    /// occurrence, matching FEFF's convention.
    pub index_map: Array1<i32>,
}

/// One FEFF NRIXS final-state transition index from `nrixs_inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphNrixsTransitionIndex {
    /// Relativistic final-state kappa index, FEFF `kind`.
    pub final_state_kappa: i32,
    /// Angular-decomposition channel, FEFF `lgind`.
    pub decomposition_channel: i32,
    /// Spherical transition multipole, FEFF `ljind`.
    pub total_angular_momentum_channel: i32,
    /// Final-state orbital angular momentum, FEFF `lind`.
    pub orbital_angular_momentum: i32,
}

/// Inputs for reconstructing FEFF `nrixs_inp` transition-index workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphNrixsTransitionIndicesInput {
    /// FEFF `kinit`, the relativistic kappa of the initial orbital.
    pub initial_kappa: i32,
    /// FEFF `le2`; `abs(le2)` is the `ljmax` multipole limit.
    pub multipole: i32,
    /// FEFF `ltot`, the maximum generated final-state orbital momentum.
    pub max_angular_momentum: usize,
}

/// FEFF `nrixs_inp` transition-index workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsphNrixsTransitionIndices {
    /// FEFF `jinit`, doubled initial-state angular momentum.
    pub initial_j2: i32,
    /// FEFF `jmax`, largest doubled final-state angular momentum.
    pub final_j2_max: i32,
    /// FEFF `ljmax`, largest spherical transition multipole.
    pub final_lj_max: usize,
    /// FEFF `kfinmax`, final-state array capacity.
    pub final_state_capacity: usize,
    /// Active `kind/lgind/ljind/lind` rows in FEFF traversal order.
    pub transitions: Vec<XsphNrixsTransitionIndex>,
}

/// Inputs for FEFF `MATH/bcoef.f90` ordinary XSPH transition-index setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphBcoefTransitionIndicesInput {
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// FEFF `le2`: `0` for dipole only, `1` to include M1, `2` to include E2.
    pub higher_multipole_selector: i32,
    /// FEFF `lx`, the maximum retained final-state orbital momentum.
    pub max_angular_momentum: usize,
}

/// One FEFF `bcoef` ordinary transition slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphBcoefTransitionIndex {
    /// One-based FEFF slot, `1..=8`.
    pub slot_1based: usize,
    /// Relativistic final-state kappa, FEFF `kiind(slot)`.
    pub final_kappa: i32,
    /// FEFF `jind(slot) = abs(final_kappa)` for physical slots.
    pub j_index: i32,
    /// Final-state orbital angular momentum, FEFF `lind(slot)`.
    pub orbital_l: i32,
}

/// FEFF `bcoef` ordinary transition-index workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsphBcoefTransitionIndices {
    /// FEFF `kiind(1:8)`.
    pub final_kappas: Array1<i32>,
    /// FEFF `jind(1:8)`.
    pub j_indices: Array1<i32>,
    /// FEFF `lind(1:8)`.
    pub orbital_l: Array1<i32>,
    /// Slots in FEFF order.
    pub transitions: Vec<XsphBcoefTransitionIndex>,
}

/// Inputs for extracting FEFF `XSPH/xsect.f90` angular weights from `bcoef`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectBcoefWeightsInput {
    /// FEFF `lx`, the maximum retained final-state orbital momentum.
    pub max_angular_momentum: usize,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// FEFF polarization selector, `ipol`.
    pub polarization: i32,
    /// Polarization tensor from FEFF `ptz(-1:1,-1:1)`.
    pub polarization_tensor: [[Complex; 3]; 3],
    /// FEFF `le2`: `0` for dipole only, `1` to include M1, `2` to include E2.
    pub higher_multipole_selector: i32,
    /// FEFF spin selector, `ispin`.
    pub spin: i32,
    /// Compiled spin-channel count used by FEFF `nspx`/`nspu`.
    pub spin_channels: usize,
    /// Angle between the x-ray k-vector and spin vector.
    pub spin_vector_angle: Real,
}

/// Traced FEFF `bcoef` weights consumed by `XSPH/xsect.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectBcoefWeights {
    /// FEFF `isp`: `0`, except `ispin == 1` selects `nspx - 1`.
    pub selected_spin_index: usize,
    /// FEFF `kiind(1:8)`.
    pub final_kappas: Array1<i32>,
    /// FEFF `lind(1:8)`.
    pub orbital_l: Array1<i32>,
    /// FEFF `bmat(0,isp,k2,0,isp,k1)` for all transition-slot pairs.
    pub trace_weights: Array2<Complex>,
    /// Diagonal `bmat(0,isp,k,0,isp,k)` entries used by direct transitions.
    pub diagonal_weights: Array1<Complex>,
}

/// FEFF `XSPH/xmult.f90` relativistic multipole prefactors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphRelativisticMultipoleFactors {
    /// FEFF `xm1`, multiplying the radial `P_k * Q_k'` contribution.
    pub p_q_prime: Complex,
    /// FEFF `xm2`, multiplying the radial `Q_k * P_k'` contribution.
    pub q_p_prime: Complex,
}

/// FEFF `radint.f90` reduced matrix-element branch selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphRadialIntegralMode {
    /// FEFF `ifl = 1`, relativistic reduced matrix element.
    RelativisticMatrixElement,
    /// FEFF `ifl = -1`, nonrelativistic reduced matrix element.
    NonRelativisticMatrixElement,
}

/// Transition kind used by FEFF `radint.f90` `mult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphTransitionMultipole {
    /// FEFF `mult = 0`, electric dipole with `j_0` and `j_2` Bessel terms.
    ElectricDipole,
    /// FEFF `mult = 1`, magnetic dipole.
    MagneticDipole,
    /// FEFF `mult = 2`, electric quadrupole.
    ElectricQuadrupole,
}

/// Inputs for FEFF `XSPH/xsect.f90` photon Bessel table generation.
#[derive(Debug, Clone, Copy)]
pub struct XsphXrayBesselTableInput<'a> {
    /// FEFF `xk0 = omega * alphfs`, the photon wave number.
    pub photon_wave_number: Real,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// Photon Bessel table used by FEFF `XSPH/radint.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXrayBesselTable {
    /// FEFF `bf(0:2, 1:ilast)`, with rows `j_0`, `j_1`, and `j_2`.
    pub values: Array2<Real>,
}

/// Inputs for FEFF `XSPH/xsect.f90` initial hole-orbital normalization check.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectHoleNormalizationInput<'a> {
    /// Initial-state orbital angular momentum, FEFF `linit`.
    pub initial_l: usize,
    /// Loucks logarithmic grid step, FEFF `dx`.
    pub log_step: Real,
    /// Radial grid, FEFF `ri(1:jnrm)`.
    pub radii: ArrayView1<'a, Real>,
    /// Large core-hole component, FEFF `dgc0(1:jnrm)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Small core-hole component, FEFF `dpc0(1:jnrm)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Norman-radius integration endpoint, FEFF one-based `jnrm`.
    pub norman_index_1based: usize,
}

/// FEFF `XSPH/xsect.f90` initial hole-orbital normalization check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectHoleNormalization {
    /// Near-origin power passed to FEFF `somm`, `2*linit + 2`.
    pub near_origin_power: Real,
    /// Normalization integral FEFF stores in `xinorm`.
    pub normalization: Real,
    /// FEFF `abs(abs(xinorm) - 1)`.
    pub deviation: Real,
    /// Whether FEFF would log the normalization warning.
    pub warning_required: bool,
}

/// FEFF `XSPH/xsect.f90` per-energy row decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphXsectEnergyDecision {
    /// FEFF skips rows with `dble(em(ie)) < -10`.
    BelowEnergyWindow,
    /// FEFF skips rows with nonpositive real and imaginary `p2`.
    NonPositiveMomentum,
    /// The row proceeds to the transition/radial-integral loops.
    Active,
}

/// Inputs for FEFF `XSPH/xsect.f90` per-energy setup before radial solves.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectEnergySetupInput {
    /// Complex energy-grid row, FEFF `em(ie)`.
    pub energy: Complex,
    /// Energy-dependent reference returned by `xcpot`, FEFF `eref`.
    pub reference_energy: Complex,
    /// Edge energy, FEFF `edge`.
    pub edge_energy: Real,
    /// Chemical-potential position in the absorption spectrum, FEFF `emu`.
    pub chemical_potential: Real,
    /// Muffin-tin radius, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Exchange/self-energy selector used for FEFF `ncycle`, FEFF `index`.
    pub exchange_selector: i32,
    /// Norman-radius radial row, FEFF one-based `jnrm`.
    pub norman_index_1based: usize,
    /// Core-orbital output row count, FEFF `jnew`.
    pub new_grid_index_1based: usize,
    /// Radial array capacity, FEFF `nrptx`.
    pub radial_capacity: usize,
}

/// FEFF `XSPH/xsect.f90` per-energy values reused by cross-section loops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectEnergySetup {
    /// Whether FEFF continues to the transition/radial-integral loops.
    pub decision: XsphXsectEnergyDecision,
    /// Momentum squared referenced to the energy-dependent self-energy, FEFF `p2`.
    pub momentum_squared: Complex,
    /// Edge row referenced to the real part of `eref`, FEFF `p2f`.
    pub edge_momentum_squared: Real,
    /// Relativistic photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Muffin-tin Bessel argument, FEFF `xkmt`.
    pub muffin_tin_argument: Complex,
    /// Differential-equation cycle count, FEFF `ncycle`.
    pub cycle_count: usize,
    /// Floored photon energy above edge, FEFF `omega`.
    pub photon_energy: Real,
    /// Photon wave number, FEFF `xk0 = omega * alphfs`.
    pub photon_wave_number: Real,
    /// Active radial prefix for Bessel tables and radial integrals, FEFF `ilast`.
    pub active_radial_len: usize,
}

/// Inputs for FEFF `XSPH/xsect.f90` transition-loop planning.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectTransitionPlanInput<'a> {
    /// Photon energy above edge, FEFF `omega`.
    pub photon_energy: Real,
    /// Active higher multipole requested by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// Transition-direction selector, FEFF `l2lp`.
    pub transition_direction: i32,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state kappa table, FEFF `kiind(1:8)`.
    pub final_kappas: ArrayView1<'a, i32>,
    /// Final-state orbital momentum table, FEFF `lind(1:8)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Active final-state table prefix.
    pub active_len: usize,
}

/// One FEFF `XSPH/xsect.f90` transition selected for radial solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphXsectTransition {
    /// Active transition multipole, FEFF `mult`.
    pub multipole: XsphTransitionMultipole,
    /// Transition-difference loop value, FEFF `kdif`.
    pub transition_delta: i32,
    /// One-based transition table index, FEFF `ind = kdif + ks`.
    pub transition_index_1based: usize,
    /// Final-state kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Final-state orbital momentum, FEFF `lfin`.
    pub final_l: i32,
    /// FEFF `kx`, used in `2*kx+1` transition normalization.
    pub multipole_order: usize,
}

/// FEFF `XSPH/xsect.f90` transition-loop plan for one energy row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsphXsectTransitionPlan {
    /// Selected transitions in FEFF traversal order.
    pub transitions: Vec<XsphXsectTransition>,
}

/// FEFF `XSPH/xsect.f90` screened-dipole field branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphXsectScreenedFieldMode {
    /// FEFF `else`: `fscf(:)=1` and `wse = ww`.
    UnityField,
    /// FEFF `mult.eq.0 .and. izstd.gt.0`: call `phiscf`.
    ScreenedDipole,
}

/// FEFF `phiscf` workspace constants from `XSPH/xsect.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectPhiscfWorkspace {
    /// FEFF `maxsize = 1`.
    pub max_size: usize,
    /// FEFF `matsize = 0`.
    pub matrix_size: usize,
    /// FEFF `sfun = 1.d0`.
    pub scale_function: Real,
}

/// Inputs for FEFF `XSPH/xsect.f90` screened-dipole field setup.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectScreenedFieldInput {
    /// Active transition multipole, FEFF `mult`.
    pub multipole: XsphTransitionMultipole,
    /// Whether FEFF `izstd.gt.0`.
    pub standard_potential: bool,
    /// Whether FEFF `CorrectOrbitalEnergies` is still true.
    pub orbital_correction_pending: bool,
    /// Momentum squared referenced to the energy-dependent self-energy, FEFF `p2`.
    pub momentum_squared: Complex,
    /// Edge energy, FEFF `edge`.
    pub edge_energy: Real,
    /// Chemical-potential position in the absorption spectrum, FEFF `emu`.
    pub chemical_potential: Real,
    /// Corrected hole-orbital energy, FEFF `eng(1,ihole)`, used only for screened dipoles.
    pub screened_orbital_energy: Real,
}

/// FEFF `XSPH/xsect.f90` screened-dipole field setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectScreenedField {
    /// Selected FEFF branch.
    pub mode: XsphXsectScreenedFieldMode,
    /// FEFF initial `ww = dble(emu+p2-edge)`.
    pub work_energy: Real,
    /// FEFF `wse`, equal to `ww` for unity fields.
    pub screened_transition_energy: Real,
    /// FEFF final `ww = sqrt(wse/ww)`.
    pub field_scale: Real,
    /// Whether the driver should set `fscf(:)=1`.
    pub unity_fscf: bool,
    /// Whether FEFF would call `correorb` before `phiscf`.
    pub orbital_correction_required: bool,
    /// FEFF `CorrectOrbitalEnergies` state after this setup block.
    pub orbital_correction_pending_after: bool,
    /// FEFF `phiscf` workspace constants when `mode` is `ScreenedDipole`.
    pub phiscf_workspace: Option<XsphXsectPhiscfWorkspace>,
}

/// Inputs for FEFF `TDLDA/phiscf.f90` local exchange-field setup.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfLocalFieldInput<'a> {
    /// FEFF `ifxc`; zero selects RPA and forces the local field to zero.
    pub exchange_correlation_selector: i32,
    /// Loucks radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Total electron density, FEFF `edens(1:ilast)`.
    pub electron_density: ArrayView1<'a, Real>,
    /// Active radial prefix, FEFF `ilast`.
    pub active_len: usize,
}

/// FEFF `TDLDA/phiscf.f90` local exchange-field setup.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfLocalField {
    /// FEFF `fxc(1:ilast)`.
    pub values: Array1<Real>,
}

/// Inputs for FEFF `TDLDA/chiklu.f90` screened-field linear solve.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfLinearSolveInput<'a> {
    /// Number of coarse 0.05-grid points, FEFF `nx`.
    pub coarse_count: usize,
    /// Fine radial grid, FEFF `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Coarse `K*chi0` matrix, FEFF `chik(1:nx,1:nx)`.
    pub response: ArrayView2<'a, Complex>,
    /// Fine-grid source columns, FEFF `yvec(1:nrptx,1:matsize)`.
    pub basis_fields: ArrayView2<'a, Complex>,
    /// Number of active `yvec` columns, FEFF `matsize`.
    pub basis_count: usize,
}

/// FEFF `TDLDA/chiklu.f90` screened-field linear solve.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfLinearSolve {
    /// FEFF `fnew`, the solved self-consistent radiation field.
    pub screened_field: Array1<Complex>,
    /// FEFF updated `yvec` columns after the same screened solve.
    pub screened_basis_fields: Array2<Complex>,
}

/// Inputs for FEFF `TDLDA/lipman.f90` `K*chi0` response assembly.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfLipmanInput<'a> {
    /// Number of coarse 0.05-grid rows to emit.
    pub coarse_count: usize,
    /// Fine radial-grid prefix used by FEFF `lipman`, FEFF `imx0`.
    pub active_len: usize,
    /// FEFF one-based `jri` where tail integrals are zeroed.
    pub match_index_1based: usize,
    /// Fine radial grid, FEFF `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Selected occupied orbital large component, FEFF `cg(:,j1)`.
    pub orbital_large: ArrayView1<'a, Real>,
    /// Selected occupied orbital small component, FEFF `cp(:,j1)`.
    pub orbital_small: ArrayView1<'a, Real>,
    /// Homogeneous regular large component, FEFF `ph`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Homogeneous regular small component, FEFF `qh`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Irregular large component, FEFF `pir`.
    pub irregular_large: ArrayView1<'a, Complex>,
    /// Irregular small component, FEFF `qir`.
    pub irregular_small: ArrayView1<'a, Complex>,
    /// Local exchange-field term, FEFF `fxc`.
    pub local_field: ArrayView1<'a, Real>,
}

/// FEFF `TDLDA/lipman.f90` `K*chi0` response assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfLipman {
    /// Coarse response matrix, FEFF `chik`.
    pub response: Array2<Complex>,
}

/// One FEFF `TDLDA/phiscf.f90` contribution to the accumulated `cchik` matrix.
#[derive(Debug, Clone)]
pub struct XsphXsectPhiscfResponseContributionInput<'a> {
    /// Coarse `lipman` response contribution, FEFF `chik`.
    pub response: ArrayView2<'a, Complex>,
    /// Real prefactor applied to this contribution, FEFF `real(aa)`.
    pub scale: Real,
    /// Whether FEFF keeps `dimag(chik)` for this pole.
    pub include_imaginary: bool,
}

/// Inputs for accumulating FEFF `TDLDA/phiscf.f90` `cchik`.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfAccumulatedResponseInput<'a> {
    /// Number of coarse 0.05-grid rows to emit.
    pub coarse_count: usize,
    /// FEFF `lipman` contributions in traversal order.
    pub contributions: &'a [XsphXsectPhiscfResponseContributionInput<'a>],
}

/// FEFF `TDLDA/phiscf.f90` accumulated `cchik` response.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfAccumulatedResponse {
    /// Coarse accumulated response matrix, FEFF `cchik`.
    pub response: Array2<Complex>,
}

/// Inputs for FEFF `TDLDA/phiscf.f90` `aa` contribution scaling.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfContributionRuleInput {
    /// Initial occupied-orbital relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state relativistic kappa, FEFF `kfin`.
    pub final_kappa: i32,
    /// Fractional shell occupation, FEFF `xx`.
    pub shell_occupation_fraction: Real,
    /// Photon-energy correction, FEFF `wp`.
    pub photon_energy_correction: Real,
    /// Separation function between ZS and PM, FEFF `sfun`.
    pub scale_function: Real,
    /// One-based pole index, FEFF `ind`.
    pub pole_index_1based: usize,
    /// Pole energy used for the production imaginary branch, FEFF `p2p`.
    pub pole_energy: Complex,
    /// Shifted Fermi level, FEFF `edge`.
    pub edge_energy: Real,
}

/// FEFF `TDLDA/phiscf.f90` contribution scaling and imaginary-branch decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectPhiscfContributionRule {
    /// Angular prefactor before occupation/photon/scaling terms.
    pub angular_scale: Real,
    /// Real FEFF `aa` scale applied during `cchik` accumulation.
    pub scale: Real,
    /// Whether FEFF keeps `dimag(chik)` for this contribution.
    pub include_imaginary: bool,
}

/// Inputs for FEFF `TDLDA/phiscf.f90` pole-energy setup.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfPoleEnergyInput {
    /// Current complex energy, FEFF `p2`.
    pub momentum_squared: Complex,
    /// Shifted Fermi level, FEFF `edge`.
    pub edge_energy: Real,
    /// Chemical-potential position in the absorption spectrum, FEFF `emu`.
    pub chemical_potential: Real,
    /// Corrected hole-orbital energy, FEFF `eng(1,ihole)`.
    pub hole_orbital_energy: Real,
    /// Occupied-orbital DOS energy, FEFF `eng(ieg,iorb)`.
    pub occupied_orbital_energy: Real,
    /// One-based pole index, FEFF `ind`.
    pub pole_index_1based: usize,
}

/// FEFF `TDLDA/phiscf.f90` pole-energy setup for one occupied-state pole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectPhiscfPoleEnergy {
    /// True x-ray photon energy, FEFF `dble(p2 + emu - edge)`.
    pub photon_energy: Real,
    /// Single-electron response energy, FEFF `ww = p2 - eng(1,ihole)`.
    pub response_energy: Complex,
    /// FEFF `wp = dble(ww) / dble(p2 + emu - edge)`.
    pub photon_energy_correction: Real,
    /// Pole energy before below-edge broadening adjustment.
    pub raw_pole_energy: Complex,
    /// Final pole energy used to solve radial equations, FEFF `p2p`.
    pub pole_energy: Complex,
    /// Whether FEFF replaced the imaginary part with the large-broadening branch.
    pub below_edge_broadening_applied: bool,
    /// Final imaginary broadening, FEFF `gamb` when adjusted.
    pub broadening: Real,
}

/// Inputs for FEFF `TDLDA/getmat.f90` channel-basis construction.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaChannelBasisInput<'a> {
    /// FEFF `ihole`, the one-based core-hole orbital index.
    pub core_hole_index_1based: usize,
    /// Initial-state orbital angular momentum, FEFF `lin`.
    pub initial_l: i32,
    /// Requested number of `lin + 1` projector orbitals, FEFF `nlp1`.
    pub plus_basis_count: i32,
    /// Requested number of `lin - 1` projector orbitals, FEFF `nlm1`.
    pub minus_basis_count: i32,
    /// FEFF orbital kappa table, `kappa(1:norbp)`.
    pub orbital_kappas: ArrayView1<'a, i32>,
    /// Valence occupations used by `ibasis = 0`, FEFF `xnval(1:norbp)`.
    pub valence_occupations: ArrayView1<'a, Real>,
    /// FEFF `ibasis`; only `0` remaps projector slots to occupied orbitals.
    pub basis_selector: i32,
}

/// One row of the FEFF `TDLDA/getmat.f90` channel basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphTdldaChannelBasisRow {
    /// Doubled initial-state total angular momentum, FEFF `jinit(im)`.
    pub initial_j2: i32,
    /// Doubled initial-state magnetic quantum number, FEFF `minit(im)`.
    pub initial_m2: i32,
    /// Initial-state relativistic kappa, FEFF `kinit(im)`.
    pub initial_kappa: i32,
    /// Doubled final-state total angular momentum, FEFF `jfin(im)`.
    pub final_j2: i32,
    /// Doubled final-state magnetic quantum number, FEFF `mfin(im)`.
    pub final_m2: i32,
    /// Final-state relativistic kappa, FEFF `kfin(im)`.
    pub final_kappa: i32,
    /// One-based core orbital slot, FEFF `ncore(im)`.
    pub core_orbital_index_1based: i32,
    /// Projector orbital selector, FEFF `nph(im)`.
    pub projector_orbital_selector: i32,
}

/// Decoded FEFF `TDLDA/getmat.f90` projector selector, FEFF `nph(im)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphTdldaProjectorSelector {
    /// `ibasis = 0` selected an occupied bound-orbital row.
    ///
    /// The index is Rust zero-based; FEFF stores it as a positive one-based
    /// `nph(im)` selector.
    OccupiedOrbital { orbital_index: usize },
    /// FEFF-generated projector slot selected by the default negative `nph`.
    ///
    /// Negative selectors are encoded as pairs: `-1/-2` for basis slot zero,
    /// `-3/-4` for basis slot one, and so on. Even negative selectors mark the
    /// positive-final-kappa partner.
    GeneratedBasis {
        basis_index: usize,
        positive_final_kappa: bool,
    },
}

/// FEFF `TDLDA/getmat.f90` channel-basis workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsphTdldaChannelBasis {
    /// FEFF-adjusted `nlp1`; values below one are raised to one.
    pub plus_basis_count: usize,
    /// FEFF-adjusted `nlm1`; negative values are clamped to zero.
    pub minus_basis_count: usize,
    /// FEFF `matsize`.
    pub matrix_size: usize,
    /// Active channel rows in FEFF traversal order.
    pub rows: Vec<XsphTdldaChannelBasisRow>,
}

/// Inputs for the FEFF `TDLDA/phiscf.f90` occupied-state contribution traversal.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfContributionPlanInput<'a> {
    /// Current complex energy, FEFF `p2`.
    pub momentum_squared: Complex,
    /// Shifted Fermi level, FEFF `edge`.
    pub edge_energy: Real,
    /// Chemical-potential position in the absorption spectrum, FEFF `emu`.
    pub chemical_potential: Real,
    /// Corrected hole-orbital energy, FEFF `eng(1,ihole)`.
    pub hole_orbital_energy: Real,
    /// Separation function between ZS and PM, FEFF `sfun`.
    pub scale_function: Real,
    /// Occupied-orbital kappa values, FEFF `kappa(1:norbp)`.
    pub orbital_kappas: ArrayView1<'a, i32>,
    /// Energy-row counts for each occupied orbital, FEFF `neg(1:norbp)`.
    pub orbital_energy_counts: ArrayView1<'a, usize>,
    /// Occupied-orbital DOS energies, FEFF `eng(1:nex,1:norbp)`.
    pub occupied_energies: ArrayView2<'a, Real>,
    /// Fractional shell occupations, FEFF `rhoj(1:nex,1:norbp)`.
    pub occupation_fractions: ArrayView2<'a, Real>,
    /// Active occupied-orbital prefix, FEFF `norbp`.
    pub active_orbital_count: usize,
}

/// One planned FEFF `TDLDA/phiscf.f90` `lipman` response contribution.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfContributionPlanRow {
    /// One-based occupied-orbital index, FEFF `iorb`.
    pub orbital_index_1based: usize,
    /// One-based energy-row index, FEFF `ieg`.
    pub energy_index_1based: usize,
    /// One-based pole index, FEFF `ind`.
    pub pole_index_1based: usize,
    /// Dipole selection-loop value, FEFF `ik`.
    pub dipole_delta: i32,
    /// Initial occupied-orbital relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state relativistic kappa, FEFF `kfin`.
    pub final_kappa: i32,
    /// Occupied-orbital DOS energy, FEFF `eng(ieg,iorb)`.
    pub occupied_orbital_energy: Real,
    /// Fractional shell occupation, FEFF `xx`.
    pub shell_occupation_fraction: Real,
    /// Pole-energy setup for this occupied state and pole.
    pub pole: XsphXsectPhiscfPoleEnergy,
    /// FEFF `aa` scaling and imaginary-branch decision for this response.
    pub rule: XsphXsectPhiscfContributionRule,
}

/// FEFF `TDLDA/phiscf.f90` contribution traversal plan.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfContributionPlan {
    /// Contributions in FEFF `iorb`/`ieg`/`ind`/`ik` traversal order.
    pub rows: Vec<XsphXsectPhiscfContributionPlanRow>,
}

/// Inputs for FEFF `TDLDA/phiscf.f90` radial solver setup before `wfirdc`.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfRadialSolverSetupInput<'a> {
    /// Pole energy used for radial equations, FEFF `p2p`.
    pub pole_energy: Complex,
    /// Muffin-tin radius, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Loucks radial grid, FEFF `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Loucks grid origin shift, FEFF `x0`.
    pub origin_shift: Real,
    /// Active radial solver prefix, FEFF `idm`.
    pub active_len: usize,
    /// One-based last tabulated row for the target orbital, FEFF `jlast`.
    pub target_last_index_1based: usize,
}

/// FEFF `TDLDA/phiscf.f90` radial solver setup before `wfirdc`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectPhiscfRadialSolverSetup {
    /// Complex electron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Effective matching radius before selecting grid row, FEFF `rmtx`.
    pub matching_radius_limit: Real,
    /// FEFF one-based matching row, `jrip`.
    pub match_index_1based: usize,
    /// Rust zero-based matching row.
    pub match_index: usize,
    /// Radius passed to `wfirdc`, FEFF `rmtp = ri(jrip)-1.d-20`.
    pub match_radius: Real,
    /// FEFF one-based WKB row, `iwkb`.
    pub wkb_index_1based: usize,
    /// Rust zero-based WKB row.
    pub wkb_index: usize,
    /// FEFF `ck * rmtp`, used for Bessel/Hankel setup.
    pub match_argument_inside: Complex,
    /// FEFF `ck * ri(jrip)`, used for outside-region Hankel setup.
    pub match_argument_grid: Complex,
}

/// FEFF `TDLDA/phiscf.f90` large/small angular channels for `kfin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphXsectPhiscfAngularChannels {
    /// Large-component orbital angular momentum, FEFF `il`.
    pub large_l: usize,
    /// Small-component orbital angular momentum, FEFF `ilp`.
    pub small_l: usize,
}

/// Inputs for FEFF `TDLDA/phiscf.f90` irregular `wfirdc` seed setup.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfIrregularSeedInput {
    /// Final-state relativistic kappa, FEFF `kfin`.
    pub final_kappa: i32,
    /// Complex electron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Radius passed to `wfirdc`, FEFF `rmtp`.
    pub match_radius: Real,
}

/// FEFF `TDLDA/phiscf.f90` irregular `wfirdc` seed coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectPhiscfIrregularSeed {
    /// Large/small angular channels selected from `kfin`.
    pub channels: XsphXsectPhiscfAngularChannels,
    /// FEFF lower-component factor.
    pub small_component_factor: Complex,
    /// FEFF relativistic scale `dum1`.
    pub relativistic_scale: Complex,
    /// FEFF `api(1)` at `rmtp`.
    pub large_coefficient: Complex,
    /// FEFF `aqi(1)` at `rmtp`.
    pub small_coefficient: Complex,
}

/// Inputs for FEFF `TDLDA/phiscf.f90` post-`wfirdc` field assembly.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfFieldAssemblyInput<'a> {
    /// Final-state relativistic kappa, FEFF `kfin`.
    pub final_kappa: i32,
    /// Complex electron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Loucks radial grid, FEFF `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// Regular large component copied from FEFF `ps` to `ph`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Regular small component copied from FEFF `qs` to `qh`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Irregular large component from `wfirdc`, FEFF `pir`.
    pub irregular_large: ArrayView1<'a, Complex>,
    /// Irregular small component from `wfirdc`, FEFF `qir`.
    pub irregular_small: ArrayView1<'a, Complex>,
    /// Active radial prefix, FEFF `idm`.
    pub active_len: usize,
    /// One-based matching row, FEFF `jrip`.
    pub match_index_1based: usize,
}

/// FEFF `TDLDA/phiscf.f90` regular/irregular fields after Wronskian matching.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfFields {
    /// Large/small angular channels selected from `kfin`.
    pub channels: XsphXsectPhiscfAngularChannels,
    /// FEFF Wronskian normalization applied to `ph/qh` through `jrip`.
    pub wronskian_scale: Complex,
    /// FEFF outside-region coefficient `tl`.
    pub tail_coefficient: Complex,
    /// Regular large component, FEFF `ph`.
    pub regular_large: Array1<Complex>,
    /// Regular small component, FEFF `qh`.
    pub regular_small: Array1<Complex>,
    /// Irregular large component, FEFF `pir`.
    pub irregular_large: Array1<Complex>,
    /// Irregular small component, FEFF `qir`.
    pub irregular_small: Array1<Complex>,
}

/// Inputs for one FEFF `TDLDA/phiscf.f90` `lipman` radial contribution.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfRadialContributionInput<'a> {
    /// Number of coarse 0.05-grid rows to emit, FEFF `nx`.
    pub coarse_count: usize,
    /// Fine radial-grid prefix used by FEFF `lipman`, FEFF `idm`.
    pub active_len: usize,
    /// One-based matching row, FEFF `jrip`.
    pub match_index_1based: usize,
    /// Final-state relativistic kappa, FEFF `kfin`.
    pub final_kappa: i32,
    /// Complex electron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Loucks radial grid, FEFF `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// Selected occupied orbital large component, FEFF `cg(:,j1)`.
    pub orbital_large: ArrayView1<'a, Real>,
    /// Selected occupied orbital small component, FEFF `cp(:,j1)`.
    pub orbital_small: ArrayView1<'a, Real>,
    /// Regular large component returned by `wfirdc`, FEFF `ps`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Regular small component returned by `wfirdc`, FEFF `qs`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Irregular large component returned by `wfirdc`, FEFF `pir`.
    pub irregular_large: ArrayView1<'a, Complex>,
    /// Irregular small component returned by `wfirdc`, FEFF `qir`.
    pub irregular_small: ArrayView1<'a, Complex>,
    /// Local exchange-field term, FEFF `fxc`.
    pub local_field: ArrayView1<'a, Real>,
    /// Real prefactor applied to this contribution, FEFF `real(aa)`.
    pub response_scale: Real,
    /// Whether FEFF keeps `dimag(chik)` for this pole.
    pub include_response_imaginary: bool,
}

/// One owned FEFF `TDLDA/phiscf.f90` radial response contribution.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfRadialContribution {
    /// Regular/irregular fields after FEFF Wronskian/outside matching.
    pub fields: XsphXsectPhiscfFields,
    /// Coarse response matrix, FEFF `chik`.
    pub response: XsphXsectPhiscfLipman,
    /// Real prefactor applied to this contribution, FEFF `real(aa)`.
    pub scale: Real,
    /// Whether FEFF keeps `dimag(chik)` for this pole.
    pub include_imaginary: bool,
}

/// Inputs for one source-backed FEFF `phiscf` contribution from `wfirdc`.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfWfirdcContributionInput<'a> {
    /// Number of coarse 0.05-grid rows to emit, FEFF `nx`.
    pub coarse_count: usize,
    /// Complex electron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Prepared base input for FEFF `wfirdc`.
    pub wfirdc_input: FovrgInitialPhotoelectronInput<'a>,
    /// Selected occupied orbital large component, FEFF `cg(:,j1)`.
    pub orbital_large: ArrayView1<'a, Real>,
    /// Selected occupied orbital small component, FEFF `cp(:,j1)`.
    pub orbital_small: ArrayView1<'a, Real>,
    /// Local exchange-field term, FEFF `fxc`.
    pub local_field: ArrayView1<'a, Real>,
    /// Real prefactor applied to this contribution, FEFF `real(aa)`.
    pub response_scale: Real,
    /// Whether FEFF keeps `dimag(chik)` for this pole.
    pub include_response_imaginary: bool,
}

/// One FEFF `phiscf` contribution generated from regular/irregular `wfirdc`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfWfirdcContribution {
    /// Regular `wfirdc` solution, FEFF `ps/qs`.
    pub regular_solution: FovrgInitialPhotoelectron,
    /// Irregular Hankel seed passed to `wfirdc`, FEFF `api/aqi`.
    pub irregular_seed: XsphXsectPhiscfIrregularSeed,
    /// Irregular `wfirdc` solution, FEFF `pir/qir`.
    pub irregular_solution: FovrgInitialPhotoelectron,
    /// Matched field and `lipman` response contribution.
    pub contribution: XsphXsectPhiscfRadialContribution,
}

/// Inputs for collecting source-backed FEFF `phiscf` contributions from `wfirdc`.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfWfirdcContributionsInput<'a> {
    /// Number of coarse 0.05-grid rows to emit, FEFF `nx`.
    pub coarse_count: usize,
    /// Fine radial grid used by FEFF `chiklu`, FEFF `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Per-pole `wfirdc` contribution inputs in FEFF traversal order.
    pub contribution_inputs: &'a [XsphXsectPhiscfWfirdcContributionInput<'a>],
    /// Fine-grid source columns, FEFF `yvec(1:nrptx,1:matsize)`.
    pub basis_fields: ArrayView2<'a, Complex>,
    /// Number of active `yvec` columns, FEFF `matsize`.
    pub basis_count: usize,
}

/// Source-backed FEFF `phiscf` contributions and solved screened field.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfWfirdcContributions {
    /// Generated regular/irregular `wfirdc` contributions in FEFF traversal order.
    pub contributions: Vec<XsphXsectPhiscfWfirdcContribution>,
    /// Accumulated `cchik` solve that yields FEFF `fscf`.
    pub screened_solution: XsphXsectPhiscfScreenedSolution,
}

/// Inputs for accumulating FEFF `phiscf` contributions and solving `fscf`.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfScreenedContributionsInput<'a> {
    /// Number of coarse 0.05-grid rows to emit, FEFF `nx`.
    pub coarse_count: usize,
    /// Fine radial grid used by FEFF `chiklu`, FEFF `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Owned radial contributions in FEFF traversal order.
    pub contributions: &'a [XsphXsectPhiscfRadialContribution],
    /// Fine-grid source columns, FEFF `yvec(1:nrptx,1:matsize)`.
    pub basis_fields: ArrayView2<'a, Complex>,
    /// Number of active `yvec` columns, FEFF `matsize`.
    pub basis_count: usize,
}

/// Inputs for the FEFF `TDLDA/phiscf.f90` `lipman` + `chiklu` solve chain.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectPhiscfScreenedSolutionInput<'a> {
    /// Number of coarse 0.05-grid points, FEFF `nx`.
    pub coarse_count: usize,
    /// Fine radial-grid prefix used by FEFF `lipman`, FEFF `imx0`.
    pub active_len: usize,
    /// FEFF one-based `jri` where tail integrals are zeroed.
    pub match_index_1based: usize,
    /// Fine radial grid, FEFF `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Selected occupied orbital large component, FEFF `cg(:,j1)`.
    pub orbital_large: ArrayView1<'a, Real>,
    /// Selected occupied orbital small component, FEFF `cp(:,j1)`.
    pub orbital_small: ArrayView1<'a, Real>,
    /// Homogeneous regular large component, FEFF `ph`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Homogeneous regular small component, FEFF `qh`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Irregular large component, FEFF `pir`.
    pub irregular_large: ArrayView1<'a, Complex>,
    /// Irregular small component, FEFF `qir`.
    pub irregular_small: ArrayView1<'a, Complex>,
    /// Local exchange-field term, FEFF `fxc`.
    pub local_field: ArrayView1<'a, Real>,
    /// Real prefactor applied to this contribution, FEFF `real(aa)`.
    pub response_scale: Real,
    /// Whether FEFF keeps `dimag(chik)` for this pole.
    pub include_response_imaginary: bool,
    /// Fine-grid source columns, FEFF `yvec(1:nrptx,1:matsize)`.
    pub basis_fields: ArrayView2<'a, Complex>,
    /// Number of active `yvec` columns, FEFF `matsize`.
    pub basis_count: usize,
}

/// FEFF `TDLDA/phiscf.f90` screened radiation-field solve.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectPhiscfScreenedSolution {
    /// Coarse response matrix, FEFF `chik`.
    pub response: Array2<Complex>,
    /// FEFF `fscf`, the self-consistent radiation field.
    pub screened_field: Array1<Complex>,
    /// FEFF updated `yvec` columns after the screened solve.
    pub screened_basis_fields: Array2<Complex>,
}

/// FEFF `XSPH/xsect.f90` `fscf` component used in radial-integral passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphXsectFscfComponentPart {
    /// FEFF `id.eq.1`, use `dble(fscf)`.
    Real,
    /// FEFF `id.eq.2`, use `dimag(fscf)`.
    Imaginary,
}

/// Inputs for FEFF `XSPH/xsect.f90` `fscf` component pass setup.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectFscfWeightsInput<'a> {
    /// Whether FEFF `izstd.gt.0`.
    pub standard_potential: bool,
    /// Screened dipole field, FEFF `fscf(1:ilast)`.
    pub fscf: ArrayView1<'a, Complex>,
    /// Active radial prefix, FEFF `ilast`.
    pub active_len: usize,
}

/// One FEFF `fscf` weighting pass.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectFscfWeight {
    /// FEFF `id`, either `1` for real or `2` for imaginary.
    pub component_id: usize,
    /// Typed component selected by `id`.
    pub part: XsphXsectFscfComponentPart,
    /// FEFF `dble(fscf(1:ilast))` or `dimag(fscf(1:ilast))`.
    pub weights: Array1<Real>,
}

/// FEFF `XSPH/xsect.f90` `fscf` weighting passes.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectFscfWeights {
    /// Passes in FEFF `id` traversal order.
    pub components: Vec<XsphXsectFscfWeight>,
}

/// FEFF `XSPH/xsect.f90` radial-integral section using `fscf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphXsectRadialPassKind {
    /// Reduced matrix-element section, FEFF `ifl = +/-1`.
    ReducedMatrixElement,
    /// Central cross-section section, FEFF `ifl = +/-2`.
    CentralCrossSection,
}

/// Inputs for FEFF `XSPH/xsect.f90` `ifl` and post-`radint` scale setup.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectRadialPassInput {
    /// Section using `radint`.
    pub kind: XsphXsectRadialPassKind,
    /// Whether FEFF `izstd.gt.0`.
    pub standard_potential: bool,
    /// Photon wave number, FEFF `xk0`.
    pub photon_wave_number: Real,
    /// Screened-field scale, FEFF final `ww`.
    pub screened_field_scale: Real,
}

/// FEFF `XSPH/xsect.f90` `radint` mode and scale for one section.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectRadialPass {
    /// FEFF `ifl` passed to `radint`.
    pub feff_ifl: i32,
    /// Multiplicative scale applied to `xirf1` after `radint`.
    pub post_radint_scale: Real,
}

/// Inputs for FEFF `XSPH/xsect.f90` regular-solution normalization.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectRegularSolutionInput<'a> {
    /// Photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Phase amplitude returned by `phamp`, FEFF `temp`.
    pub phase_amplitude: Complex,
    /// Relativistic final-state kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Large regular radial component returned by `dfovrg`, FEFF `p(1:ilast)`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Small regular radial component returned by `dfovrg`, FEFF `q(1:ilast)`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// FEFF `XSPH/xsect.f90` regular solution after the `xfnorm` scale.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectRegularSolution {
    /// Relativistic lower-component factor, FEFF `factor`.
    pub small_component_factor: Complex,
    /// Relativistic normalization correction, FEFF `dum1`.
    pub relativistic_scale: Complex,
    /// Regular-solution scale, FEFF `xfnorm = dum1 / temp`.
    pub regular_solution_scale: Complex,
    /// Scaled large regular radial component.
    pub regular_large: Array1<Complex>,
    /// Scaled small regular radial component.
    pub regular_small: Array1<Complex>,
}

/// Inputs for one regular FEFF `XSPH/xsect.f90` FOVRG channel.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectRegularChannelInput<'a> {
    /// Prepared regular FOVRG solver input for the transition channel.
    pub solver: FovrgDiracSolverInput<'a>,
    /// Photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
}

/// Regular FOVRG channel after the FEFF `phamp` match and `xfnorm` scale.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectRegularChannel {
    /// Raw regular `dfovrg` output for this transition channel.
    pub regular_solution: FovrgDiracSolution,
    /// Muffin-tin phase/amplitude recovered from the regular solution.
    pub phase: XsphRegularPhase,
    /// Regular solution scaled as FEFF `XSPH/xsect.f90` uses for radial integrals.
    pub normalized_solution: XsphXsectRegularSolution,
}

/// Inputs for FEFF `XSPH/xsect.f90` irregular `dfovrg` boundary values.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectIrregularInitialConditionInput {
    /// Muffin-tin radius, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Regular phase shift, FEFF `ph0`.
    pub phase_shift: Complex,
    /// Photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Relativistic final-state kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Spherical Bessel `j_l(xkmt)`, FEFF `jl`.
    pub bessel_j_l: Complex,
    /// Spherical Neumann `n_l(xkmt)`, FEFF `nl`.
    pub neumann_l: Complex,
    /// Spherical Bessel for the coupled small-component channel, FEFF `jlp1`.
    pub bessel_j_l_plus_1: Complex,
    /// Spherical Neumann for the coupled small-component channel, FEFF `nlp1`.
    pub neumann_l_plus_1: Complex,
}

/// FEFF `XSPH/xsect.f90` irregular boundary values passed to `dfovrg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectIrregularInitialCondition {
    /// Irregular large component at `rmt`, FEFF `pu`.
    pub large_component: Complex,
    /// Irregular small component at `rmt`, FEFF `qu`.
    pub small_component: Complex,
    /// Relativistic lower-component factor, FEFF `factor`.
    pub small_component_factor: Complex,
    /// Relativistic normalization correction, FEFF `dum1`.
    pub relativistic_scale: Complex,
}

/// Inputs for FEFF `XSPH/xsect.f90` irregular post-`dfovrg` transform.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectIrregularTransformInput<'a> {
    /// Regular phase shift, FEFF `ph0`.
    pub phase_shift: Complex,
    /// Scaled large regular radial component, FEFF `p(1:ilast)`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Scaled small regular radial component, FEFF `q(1:ilast)`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Large irregular component returned by `dfovrg`, FEFF `pn(1:ilast)`.
    pub irregular_large: ArrayView1<'a, Complex>,
    /// Small irregular component returned by `dfovrg`, FEFF `qn(1:ilast)`.
    pub irregular_small: ArrayView1<'a, Complex>,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// FEFF `XSPH/xsect.f90` transformed irregular solution `N = iR - H exp(i*ph0)`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectIrregularTransform {
    /// FEFF `temp = exp(coni * ph0)`.
    pub phase_factor: Complex,
    /// Transformed large irregular component, FEFF `pn(1:ilast)`.
    pub irregular_large: Array1<Complex>,
    /// Transformed small irregular component, FEFF `qn(1:ilast)`.
    pub irregular_small: Array1<Complex>,
}

/// Inputs for one irregular FEFF `XSPH/xsect.f90` FOVRG channel.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectIrregularChannelInput<'a, 'b> {
    /// Prepared FOVRG solver input for the transition channel.
    pub solver: FovrgDiracSolverInput<'a>,
    /// Photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Regular channel carrying the FEFF `phamp` match and normalized regular rows.
    pub regular_channel: &'b XsphXsectRegularChannel,
}

/// Irregular FOVRG channel after FEFF boundary setup and outgoing transform.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectIrregularChannel {
    /// Irregular boundary values passed into `dfovrg`.
    pub initial_condition: XsphXsectIrregularInitialCondition,
    /// Raw irregular `dfovrg` output for this transition channel.
    pub irregular_solution: FovrgDiracSolution,
    /// Outgoing irregular rows transformed as FEFF `XSPH/xsect.f90` uses them.
    pub transformed_solution: XsphXsectIrregularTransform,
}

/// Inputs for FEFF `XSPH/xsect.f90` positive-omega output finalization.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectOutputNormalizationInput<'a> {
    /// Photon energy above edge, FEFF `omega`.
    pub photon_energy: Real,
    /// Photoelectron wave number, FEFF `ck` before the reduced-matrix square root.
    pub wave_number: Complex,
    /// Accumulated atomic background before the FEFF `prefac` scale, FEFF `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// Accumulated central-atom cross section before the FEFF `prefac` scale, FEFF `xsec(ie)`.
    pub cross_section: Complex,
    /// Reduced matrix elements, FEFF `rkk(ie, 1:8)`.
    pub reduced_matrix_elements: ArrayView1<'a, Complex>,
    /// Central-atom phase shifts added to reduced matrix elements, FEFF `phx(1:8)`.
    pub phase_shifts: ArrayView1<'a, Complex>,
    /// Number of active reduced-matrix channels to update.
    pub active_channel_count: usize,
}

/// FEFF `XSPH/xsect.f90` positive-omega output finalization result.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectOutputNormalization {
    /// FEFF `prefac = 4*pi*alpinv/omega*bohr**2`.
    pub prefactor: Real,
    /// FEFF normalized `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// FEFF `xnorm = sqrt(xsnorm(ie))` after the normalization scale.
    pub spectrum_norm_sqrt: Real,
    /// FEFF normalized `xsec(ie)`.
    pub cross_section: Complex,
    /// Complex square-root scale before division by `xnorm`.
    pub reduced_matrix_root_scale: Complex,
    /// Full reduced-matrix scale after dividing by `xnorm`.
    pub reduced_matrix_scale: Complex,
    /// Normalized reduced matrix elements after the `exp(i*phx)` phase factor.
    pub reduced_matrix_elements: Array1<Complex>,
}

/// Inputs for FEFF `XSPH/xsect.f90` direct transition accumulation.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectDirectTransitionInput {
    /// Active transition multipole, FEFF `mult`.
    pub multipole: XsphTransitionMultipole,
    /// Active higher multipole selected by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// Combined radial integral, FEFF `xirf`.
    pub radial_integral: Complex,
    /// Central-atom phase shift stored with `rkk`, FEFF `ph0`.
    pub phase_shift: Complex,
    /// Diagonal angular coefficient, FEFF `bmat(0,isp,ind,0,isp,ind)`.
    pub angular_weight: Complex,
    /// Current unnormalized cross-section norm, FEFF `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
}

/// Inputs for direct transition accumulation using traced FEFF `bcoef` weights.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefDirectTransitionInput<'a> {
    /// Active transition multipole, FEFF `mult`.
    pub multipole: XsphTransitionMultipole,
    /// Active higher multipole selected by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// One-based FEFF transition slot, `ind`.
    pub transition_index_1based: usize,
    /// Traced diagonal weights, FEFF `bmat(0,isp,k,0,isp,k)`.
    pub diagonal_weights: ArrayView1<'a, Complex>,
    /// Combined radial integral, FEFF `xirf`.
    pub radial_integral: Complex,
    /// Central-atom phase shift stored with `rkk`, FEFF `ph0`.
    pub phase_shift: Complex,
    /// Current unnormalized cross-section norm, FEFF `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
}

/// Inputs for a direct transition update of FEFF `xsect` row work arrays.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefDirectTransitionUpdateInput<'a> {
    /// Active transition multipole, FEFF `mult`.
    pub multipole: XsphTransitionMultipole,
    /// Active higher multipole selected by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// One-based FEFF transition slot, `ind`.
    pub transition_index_1based: usize,
    /// Traced diagonal weights, FEFF `bmat(0,isp,k,0,isp,k)`.
    pub diagonal_weights: ArrayView1<'a, Complex>,
    /// Combined radial integral, FEFF `xirf`.
    pub radial_integral: Complex,
    /// Central-atom phase shift stored with `rkk`, FEFF `ph0`.
    pub phase_shift: Complex,
    /// Current unnormalized cross-section norm, FEFF `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
    /// Current eight-slot reduced-matrix workspace, FEFF `rkk(ie,1:8)`.
    pub reduced_matrix_elements: ArrayView1<'a, Complex>,
    /// Current eight-slot phase workspace, FEFF `phx(1:8)`.
    pub phase_shifts: ArrayView1<'a, Complex>,
}

/// FEFF `XSPH/xsect.f90` direct transition accumulation result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectDirectTransition {
    /// Whether FEFF stores `rkk(ie,ind)` and `phx(ind)` for this multipole.
    pub store_reduced_matrix: bool,
    /// Reduced matrix element to store when `store_reduced_matrix` is true.
    pub reduced_matrix_element: Option<Complex>,
    /// Phase shift to store when `store_reduced_matrix` is true.
    pub phase_shift: Option<Complex>,
    /// FEFF `xsnorm` contribution from this transition.
    pub spectrum_norm_increment: Real,
    /// Updated unnormalized cross-section norm.
    pub spectrum_norm: Real,
    /// FEFF `xsec` contribution from `-(-i*xirf**2)*bmat`.
    pub cross_section_increment: Complex,
    /// Updated unnormalized central-atom cross section.
    pub cross_section: Complex,
}

/// Direct transition update of FEFF `xsect` row work arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectBcoefDirectTransitionUpdate {
    /// Raw transition accumulation result before workspace storage.
    pub transition: XsphXsectDirectTransition,
    /// Updated unnormalized cross-section norm, FEFF `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// Updated unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
    /// Updated eight-slot reduced-matrix workspace, FEFF `rkk(ie,1:8)`.
    pub reduced_matrix_elements: Array1<Complex>,
    /// Updated eight-slot phase workspace, FEFF `phx(1:8)`.
    pub phase_shifts: Array1<Complex>,
}

/// Inputs for FEFF `XSPH/xsect.f90` central cross-section accumulation.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectCentralCrossSectionInput {
    /// Whether this is the spin-orbit-removed pass, FEFF `ic3 != 0`.
    pub spin_orbit_removed_pass: bool,
    /// Combined central cross-section radial integral, FEFF `xirf`.
    pub radial_integral: Complex,
    /// Diagonal angular coefficient, FEFF `bmat(0,isp,ind,0,isp,ind)`.
    pub angular_weight: Complex,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
}

/// Inputs for central cross-section accumulation using traced FEFF `bcoef`.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefCentralCrossSectionInput<'a> {
    /// Whether this is the spin-orbit-removed pass, FEFF `ic3 != 0`.
    pub spin_orbit_removed_pass: bool,
    /// One-based FEFF transition slot, `ind`.
    pub transition_index_1based: usize,
    /// Traced diagonal weights, FEFF `bmat(0,isp,k,0,isp,k)`.
    pub diagonal_weights: ArrayView1<'a, Complex>,
    /// Combined central cross-section radial integral, FEFF `xirf`.
    pub radial_integral: Complex,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
}

/// FEFF `XSPH/xsect.f90` central cross-section accumulation result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectCentralCrossSection {
    /// Increment from FEFF `-xirf * bmat(0,isp,ind,0,isp,ind)`.
    pub cross_section_increment: Complex,
    /// Updated unnormalized central-atom cross section.
    pub cross_section: Complex,
}

/// Inputs for one ordinary FEFF `XSPH/xsect.f90` transition row.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefOrdinaryRowInput<'a> {
    /// Active transition multipole, FEFF `mult`.
    pub multipole: XsphTransitionMultipole,
    /// Active higher multipole selected by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// One-based FEFF transition slot, `ind`.
    pub transition_index_1based: usize,
    /// Traced diagonal weights, FEFF `bmat(0,isp,k,0,isp,k)`.
    pub diagonal_weights: ArrayView1<'a, Complex>,
    /// Reduced matrix-element radial integral from FEFF `radint(ifl=1)`.
    pub reduced_matrix_integral: Complex,
    /// Central cross-section radial integral from FEFF `radint(ifl=2)`.
    pub central_cross_integral: Complex,
    /// Central-atom phase shift stored with `rkk`, FEFF `ph0`.
    pub phase_shift: Complex,
    /// Current unnormalized cross-section norm, FEFF `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
    /// Current eight-slot reduced-matrix workspace, FEFF `rkk(ie,1:8)`.
    pub reduced_matrix_elements: ArrayView1<'a, Complex>,
    /// Current eight-slot phase workspace, FEFF `phx(1:8)`.
    pub phase_shifts: ArrayView1<'a, Complex>,
}

/// Inputs for one nonstandard-potential ordinary `XSPH/xsect.f90` transition row.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefNonstandardChannelRowInput<'a> {
    /// Active transition selected by the FEFF `xsect` transition-loop planner.
    pub transition: XsphXsectTransition,
    /// Active higher multipole selected by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Regular final-state channel after the FEFF `xfnorm` scale.
    pub regular_channel: &'a XsphXsectRegularChannel,
    /// Irregular final-state channel after the FEFF outgoing transform.
    pub irregular_channel: &'a XsphXsectIrregularChannel,
    /// X-ray Bessel functions `bf(0:2, 1:ilast)`.
    pub xray_bessel: ArrayView2<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Traced diagonal weights, FEFF `bmat(0,isp,k,0,isp,k)`.
    pub diagonal_weights: ArrayView1<'a, Complex>,
    /// Current unnormalized cross-section norm, FEFF `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
    /// Current eight-slot reduced-matrix workspace, FEFF `rkk(ie,1:8)`.
    pub reduced_matrix_elements: ArrayView1<'a, Complex>,
    /// Current eight-slot phase workspace, FEFF `phx(1:8)`.
    pub phase_shifts: ArrayView1<'a, Complex>,
}

/// Inputs for one standard-potential ordinary `XSPH/xsect.f90` transition row.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefStandardChannelRowInput<'a> {
    /// Active transition selected by the FEFF `xsect` transition-loop planner.
    pub transition: XsphXsectTransition,
    /// Active higher multipole selected by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Regular final-state channel after the FEFF `xfnorm` scale.
    pub regular_channel: &'a XsphXsectRegularChannel,
    /// Irregular final-state channel after the FEFF outgoing transform.
    pub irregular_channel: &'a XsphXsectIrregularChannel,
    /// X-ray Bessel functions `bf(0:2, 1:ilast)`.
    pub xray_bessel: ArrayView2<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Photon wave number, FEFF `xk0`.
    pub photon_wave_number: Real,
    /// Screened-field scale from the standard-atom branch, FEFF final `ww`.
    pub screened_field_scale: Real,
    /// Complex screened dipole field, FEFF `fscf(1:ilast)`.
    pub fscf: ArrayView1<'a, Complex>,
    /// Traced diagonal weights, FEFF `bmat(0,isp,k,0,isp,k)`.
    pub diagonal_weights: ArrayView1<'a, Complex>,
    /// Current unnormalized cross-section norm, FEFF `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
    /// Current eight-slot reduced-matrix workspace, FEFF `rkk(ie,1:8)`.
    pub reduced_matrix_elements: ArrayView1<'a, Complex>,
    /// Current eight-slot phase workspace, FEFF `phx(1:8)`.
    pub phase_shifts: ArrayView1<'a, Complex>,
}

/// Computed nonstandard-potential ordinary `XSPH/xsect.f90` transition row.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectBcoefNonstandardChannelRow {
    /// FEFF `radint(ifl=1)` branch selection for the reduced matrix element.
    pub reduced_radial_pass: XsphXsectRadialPass,
    /// FEFF `radint(ifl=2)` branch selection for the central cross section.
    pub central_radial_pass: XsphXsectRadialPass,
    /// Reduced radial matrix element before FEFF `xsect` row accumulation.
    pub reduced_radial_integral: XsphRadialIntegral,
    /// Central-atom cross-section radial integral before row accumulation.
    pub central_cross_integral: XsphRadialCrossIntegral,
    /// Updated FEFF ordinary row workspaces and unnormalized spectrum values.
    pub row: XsphXsectBcoefOrdinaryRow,
}

/// Computed standard-potential ordinary `XSPH/xsect.f90` transition row.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectBcoefStandardChannelRow {
    /// FEFF real/imaginary `fscf` component passes in traversal order.
    pub fscf_weights: XsphXsectFscfWeights,
    /// FEFF `radint(ifl=-1)` branch selection for the reduced matrix element.
    pub reduced_radial_pass: XsphXsectRadialPass,
    /// FEFF `radint(ifl=-2)` branch selection for the central cross section.
    pub central_radial_pass: XsphXsectRadialPass,
    /// Per-`fscf` reduced radial matrix elements before component combination.
    pub reduced_component_integrals: Vec<XsphXsectWeightedRadialIntegral>,
    /// FEFF magnitude-combination trace for the reduced matrix element.
    pub reduced_fscf_integrals: Vec<XsphXsectFscfIntegral>,
    /// Per-`fscf` central cross-section radial integrals before combination.
    pub central_component_integrals: Vec<XsphXsectWeightedRadialCrossIntegral>,
    /// FEFF magnitude-combination trace for the central cross-section integral.
    pub central_fscf_integrals: Vec<XsphXsectFscfIntegral>,
    /// Updated FEFF ordinary row workspaces and unnormalized spectrum values.
    pub row: XsphXsectBcoefOrdinaryRow,
}

/// Inputs for one standard-potential ordinary `XSPH/xsect.f90` energy row.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefStandardEnergyRowInput<'a> {
    /// Active transitions selected by the FEFF `xsect` transition-loop planner.
    pub transitions: &'a [XsphXsectTransition],
    /// Regular final-state channel for each transition.
    pub regular_channels: &'a [XsphXsectRegularChannel],
    /// Irregular final-state channel for each transition.
    pub irregular_channels: &'a [XsphXsectIrregularChannel],
    /// Active higher multipole selected by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// X-ray Bessel functions `bf(0:2, 1:ilast)`.
    pub xray_bessel: ArrayView2<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Photon wave number, FEFF `xk0`.
    pub photon_wave_number: Real,
    /// Screened-field scale from the standard-atom branch, FEFF final `ww`.
    pub screened_field_scale: Real,
    /// Complex screened dipole field, FEFF `fscf(1:ilast)`.
    pub fscf: ArrayView1<'a, Complex>,
    /// Traced diagonal weights, FEFF `bmat(0,isp,k,0,isp,k)`.
    pub diagonal_weights: ArrayView1<'a, Complex>,
    /// Whether to run FEFF's spin-polarized adjacent-channel cross-term retry.
    pub spin_polarized_cross_terms: bool,
    /// Final-state orbital momentum table, FEFF `lind(1:8)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Traced angular coefficients, FEFF `bmat(0,isp,k2,0,isp,k1)`.
    pub trace_weights: ArrayView2<'a, Complex>,
    /// Spin-orbit-removed regular channels from the FEFF `ic3 = 1` retry pass.
    pub spin_orbit_removed_regular_channels: Option<&'a [XsphXsectRegularChannel]>,
    /// Spin-orbit-removed irregular channels from the FEFF `ic3 = 1` retry pass.
    pub spin_orbit_removed_irregular_channels: Option<&'a [XsphXsectIrregularChannel]>,
    /// Photon energy above edge, FEFF `omega`.
    pub photon_energy: Real,
    /// Photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Number of reduced-matrix channels to normalize into FEFF `rkk`.
    pub active_channel_count: usize,
}

/// Screened-field data for one standard-potential transition row.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefStandardTransitionField<'a> {
    /// Screened-field scale for this transition, FEFF final `ww`.
    pub screened_field_scale: Real,
    /// Complex screened field for this transition, FEFF `fscf(1:ilast)`.
    pub fscf: ArrayView1<'a, Complex>,
}

/// Inputs for a standard-potential ordinary `XSPH/xsect.f90` energy row when
/// FEFF uses different `fscf` branches for different transition multipoles.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefStandardEnergyRowFieldsInput<'a> {
    /// Active transitions selected by the FEFF `xsect` transition-loop planner.
    pub transitions: &'a [XsphXsectTransition],
    /// Regular final-state channel for each transition.
    pub regular_channels: &'a [XsphXsectRegularChannel],
    /// Irregular final-state channel for each transition.
    pub irregular_channels: &'a [XsphXsectIrregularChannel],
    /// Per-transition screened-field branch in the same order as `transitions`.
    pub transition_fields: &'a [XsphXsectBcoefStandardTransitionField<'a>],
    /// Active higher multipole selected by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// X-ray Bessel functions `bf(0:2, 1:ilast)`.
    pub xray_bessel: ArrayView2<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Photon wave number, FEFF `xk0`.
    pub photon_wave_number: Real,
    /// Traced diagonal weights, FEFF `bmat(0,isp,k,0,isp,k)`.
    pub diagonal_weights: ArrayView1<'a, Complex>,
    /// Whether to run FEFF's spin-polarized adjacent-channel cross-term retry.
    pub spin_polarized_cross_terms: bool,
    /// Final-state orbital momentum table, FEFF `lind(1:8)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Traced angular coefficients, FEFF `bmat(0,isp,k2,0,isp,k1)`.
    pub trace_weights: ArrayView2<'a, Complex>,
    /// Spin-orbit-removed regular channels from the FEFF `ic3 = 1` retry pass.
    pub spin_orbit_removed_regular_channels: Option<&'a [XsphXsectRegularChannel]>,
    /// Spin-orbit-removed irregular channels from the FEFF `ic3 = 1` retry pass.
    pub spin_orbit_removed_irregular_channels: Option<&'a [XsphXsectIrregularChannel]>,
    /// Photon energy above edge, FEFF `omega`.
    pub photon_energy: Real,
    /// Photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Number of reduced-matrix channels to normalize into FEFF `rkk`.
    pub active_channel_count: usize,
}

/// Computed standard-potential ordinary `XSPH/xsect.f90` energy row.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectBcoefStandardEnergyRow {
    /// Per-transition standard radial-integral and bcoef row details in FEFF order.
    pub transition_rows: Vec<XsphXsectBcoefStandardChannelRow>,
    /// Active FEFF XMCD adjacent-channel cross-term updates, in traversal order.
    pub cross_term_updates: Vec<XsphXsectCrossTermAccumulation>,
    /// Unnormalized FEFF `xsnorm(ie)` before the final `prefac` scale.
    pub unnormalized_spectrum_norm: Real,
    /// Unnormalized FEFF `xsec(ie)` before the final `prefac` scale.
    pub unnormalized_cross_section: Complex,
    /// Unnormalized reduced-matrix workspace before the final `rkk` scale.
    pub unnormalized_reduced_matrix_elements: Array1<Complex>,
    /// Central-atom phase workspace, FEFF `phx`.
    pub phase_shifts: Array1<Complex>,
    /// Final FEFF positive-omega normalization for this energy row.
    pub output_normalization: XsphXsectOutputNormalization,
}

/// Inputs for one nonstandard-potential ordinary `XSPH/xsect.f90` energy row.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefNonstandardEnergyRowInput<'a> {
    /// Active transitions selected by the FEFF `xsect` transition-loop planner.
    pub transitions: &'a [XsphXsectTransition],
    /// Regular final-state channel for each transition.
    pub regular_channels: &'a [XsphXsectRegularChannel],
    /// Irregular final-state channel for each transition.
    pub irregular_channels: &'a [XsphXsectIrregularChannel],
    /// Active higher multipole selected by FEFF `le2`, if any.
    pub selected_higher_multipole: Option<XsphTransitionMultipole>,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// X-ray Bessel functions `bf(0:2, 1:ilast)`.
    pub xray_bessel: ArrayView2<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Traced diagonal weights, FEFF `bmat(0,isp,k,0,isp,k)`.
    pub diagonal_weights: ArrayView1<'a, Complex>,
    /// Whether to run FEFF's spin-polarized adjacent-channel cross-term retry.
    pub spin_polarized_cross_terms: bool,
    /// Final-state orbital momentum table, FEFF `lind(1:8)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Traced angular coefficients, FEFF `bmat(0,isp,k2,0,isp,k1)`.
    pub trace_weights: ArrayView2<'a, Complex>,
    /// Spin-orbit-removed regular channels from the FEFF `ic3 = 1` retry pass.
    pub spin_orbit_removed_regular_channels: Option<&'a [XsphXsectRegularChannel]>,
    /// Spin-orbit-removed irregular channels from the FEFF `ic3 = 1` retry pass.
    pub spin_orbit_removed_irregular_channels: Option<&'a [XsphXsectIrregularChannel]>,
    /// Photon energy above edge, FEFF `omega`.
    pub photon_energy: Real,
    /// Photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Number of reduced-matrix channels to normalize into FEFF `rkk`.
    pub active_channel_count: usize,
}

/// Computed nonstandard-potential ordinary `XSPH/xsect.f90` energy row.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectBcoefNonstandardEnergyRow {
    /// Per-transition radial-integral and bcoef row details in FEFF order.
    pub transition_rows: Vec<XsphXsectBcoefNonstandardChannelRow>,
    /// Active FEFF XMCD adjacent-channel cross-term updates, in traversal order.
    pub cross_term_updates: Vec<XsphXsectCrossTermAccumulation>,
    /// Unnormalized FEFF `xsnorm(ie)` before the final `prefac` scale.
    pub unnormalized_spectrum_norm: Real,
    /// Unnormalized FEFF `xsec(ie)` before the final `prefac` scale.
    pub unnormalized_cross_section: Complex,
    /// Unnormalized reduced-matrix workspace before the final `rkk` scale.
    pub unnormalized_reduced_matrix_elements: Array1<Complex>,
    /// Central-atom phase workspace, FEFF `phx`.
    pub phase_shifts: Array1<Complex>,
    /// Final FEFF positive-omega normalization for this energy row.
    pub output_normalization: XsphXsectOutputNormalization,
}

/// One ordinary FEFF `XSPH/xsect.f90` transition row after bcoef accumulation.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectBcoefOrdinaryRow {
    /// Direct reduced-matrix and `xsnorm/xsec` update.
    pub direct_transition: XsphXsectBcoefDirectTransitionUpdate,
    /// Diagonal central cross-section update from `radint(ifl=2)`.
    pub central_cross_section: XsphXsectCentralCrossSection,
    /// Updated unnormalized cross-section norm, FEFF `xsnorm(ie)`.
    pub spectrum_norm: Real,
    /// Updated unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
    /// Updated eight-slot reduced-matrix workspace, FEFF `rkk(ie,1:8)`.
    pub reduced_matrix_elements: Array1<Complex>,
    /// Updated eight-slot phase workspace, FEFF `phx(1:8)`.
    pub phase_shifts: Array1<Complex>,
}

/// Inputs for FEFF `XSPH/xsect.f90` embedded central density `xrhoce`.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectEmbeddedDensityInput<'a> {
    /// Final-state orbital angular momentum, FEFF `lfin`.
    pub final_l: usize,
    /// Relativistic final-state kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Scaled large regular radial component, FEFF `p(1:ilast)`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Scaled small regular radial component, FEFF `q(1:ilast)`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Transformed large irregular radial component, FEFF `pn(1:ilast)`.
    pub irregular_large: ArrayView1<'a, Complex>,
    /// Transformed small irregular radial component, FEFF `qn(1:ilast)`.
    pub irregular_small: ArrayView1<'a, Complex>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Loucks logarithmic grid step, FEFF `dx`.
    pub log_step: Real,
    /// Norman radius used as the integration endpoint, FEFF `rnrm`.
    pub norman_radius: Real,
    /// Number of radial samples prepared by XSPH, FEFF `ilast`.
    pub active_len: usize,
    /// Number of samples integrated through Norman radius, FEFF `i0 = jnrm + 1`.
    pub integration_len: usize,
}

/// FEFF `XSPH/xsect.f90` embedded central density contribution.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectEmbeddedDensity {
    /// Density prefactor, FEFF `temp`.
    pub prefactor: Complex,
    /// Integrand samples, FEFF `xrc(1:ilast)`.
    pub density_samples: Array1<Complex>,
    /// Norman-radius integral before applying `-temp`, FEFF `xirf`.
    pub integral: Complex,
    /// Embedded central density, FEFF `xrhoce(ie)`.
    pub density: Complex,
}

/// Inputs for FEFF `XSPH/xsect.f90` projected density `xrhopr`.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectProjectedDensityInput<'a> {
    /// Final-state orbital angular momentum, FEFF `lfin`.
    pub final_l: usize,
    /// Relativistic final-state kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Photoelectron wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Scaled large regular radial component, FEFF `p(1:ilast)`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Scaled small regular radial component, FEFF `q(1:ilast)`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Transformed large irregular radial component, FEFF `pn(1:ilast)`.
    pub irregular_large: ArrayView1<'a, Complex>,
    /// Transformed small irregular radial component, FEFF `qn(1:ilast)`.
    pub irregular_small: ArrayView1<'a, Complex>,
    /// Projector large atomic component, FEFF `dgcn(1:ilast,jproj)`.
    pub atomic_large: ArrayView1<'a, Real>,
    /// Projector small atomic component, FEFF `dpcn(1:ilast,jproj)`.
    pub atomic_small: ArrayView1<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Loucks logarithmic grid step, FEFF `dx`.
    pub log_step: Real,
    /// Norman radius used as the integration endpoint, FEFF `rnrm`.
    pub norman_radius: Real,
    /// Number of radial samples prepared by XSPH, FEFF `ilast`.
    pub active_len: usize,
    /// Number of samples integrated through Norman radius, FEFF `i0 = jnrm + 1`.
    pub integration_len: usize,
}

/// FEFF `XSPH/xsect.f90` projected density contribution.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectProjectedDensity {
    /// Density prefactor, FEFF `temp`.
    pub prefactor: Complex,
    /// Atomic projector normalization integral before square root, FEFF `xinorm`.
    pub atomic_norm_integral: Real,
    /// Atomic projector normalization factor, FEFF `sqrt(xinorm)`.
    pub atomic_norm_sqrt: Real,
    /// Normalized projector large component, FEFF `pat(1:ilast)`.
    pub normalized_atomic_large: Array1<Real>,
    /// Normalized projector small component, FEFF `qat(1:ilast)`.
    pub normalized_atomic_small: Array1<Real>,
    /// Cumulative trapezoid overlaps, FEFF `intr(1:ilast)`.
    pub cumulative_overlap: Array1<Complex>,
    /// Integrand samples, FEFF `xrc(1:ilast)`.
    pub density_samples: Array1<Complex>,
    /// Norman-radius integral before applying `-temp`, FEFF `xirf`.
    pub integral: Complex,
    /// Projected density, FEFF `xrhopr(ie)`.
    pub density: Complex,
}

/// Inputs for the FEFF `XSPH/xsect.f90` `xrhoce`/`xrhopr` branch predicate.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectDensityBranchInput<'a> {
    /// Initial-state kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Transition-difference loop value, FEFF `kdif`.
    pub transition_delta: i32,
    /// Whether this is the spin-orbit-removed cross-term pass, FEFF `ic3 != 0`.
    pub spin_orbit_removed_pass: bool,
    /// Projector orbital map, FEFF `iorb(kappa)`.
    pub orbital_projector_map: ArrayView1<'a, i32>,
    /// Kappa value represented by `orbital_projector_map[0]`, FEFF `-5`.
    pub min_projector_kappa: i32,
}

/// Active FEFF `XSPH/xsect.f90` density branch selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphXsectDensityBranch {
    /// FEFF `kdif1`, the only transition delta that computes ratio densities.
    pub required_transition_delta: i32,
    /// One-based projector orbital index, FEFF `jproj`.
    pub projector_index_1based: usize,
}

/// FEFF `XSPH/xsect.f90` `fscf` radial-integral combination branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphXsectFscfSelection {
    /// FEFF `id.eq.1`: initialize `xirf` from `xirf1`.
    FirstComponent,
    /// FEFF `abs(xirf).eq.0`: replace the accumulated value with `xirf1`.
    AccumulatedZero,
    /// FEFF `abs(xirf1).eq.0`: keep the accumulated value.
    ContributionZero,
    /// FEFF `abs(xirf1).lt.abs(xirf)`: scale the accumulated value.
    AccumulatedDominant,
    /// FEFF fallback: scale the new contribution.
    ContributionDominant,
}

/// Inputs for FEFF `XSPH/xsect.f90` real/imaginary `fscf` integral combination.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectFscfIntegralInput {
    /// Current accumulated radial integral, FEFF `xirf`.
    pub accumulated: Complex,
    /// Current real- or imaginary-`fscf` radial integral, FEFF `xirf1`.
    pub contribution: Complex,
    /// Whether this is FEFF `id.eq.1`.
    pub first_component: bool,
}

/// FEFF `XSPH/xsect.f90` combined `fscf` radial integral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectFscfIntegral {
    /// Combined radial integral, FEFF `xirf`.
    pub value: Complex,
    /// Branch selected by the FEFF magnitude logic.
    pub selection: XsphXsectFscfSelection,
    /// Real scale applied to the selected complex value.
    pub scale: Real,
}

/// FEFF `XSPH/xsect.f90` spin-orbit-removal cross-term mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphXsectCrossTermMode {
    /// FEFF `iold = 1`: save current radial samples for the next channel.
    SaveCurrentForNext,
    /// FEFF `iold = 2`: reuse previous radial samples for the current channel.
    UsePreviousForCurrent,
}

/// Inputs for FEFF `XSPH/xsect.f90` `iold` cross-term planning.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectCrossTermPlanInput<'a> {
    /// Whether `abs(ispin).eq.1`.
    pub spin_polarized: bool,
    /// Whether this is the spin-orbit-removed pass, FEFF `ic3 != 0`.
    pub spin_orbit_removed_pass: bool,
    /// One-based final-state transition index, FEFF `ind`.
    pub transition_index_1based: usize,
    /// Final-state orbital momenta, FEFF `lind(1:indmax)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Number of active transitions, FEFF `indmax`.
    pub active_len: usize,
}

/// FEFF `XSPH/xsect.f90` plan for repeating a channel with spin-orbit removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphXsectCrossTermPlan {
    /// FEFF `iold` value, either `1` or `2`.
    pub iold: i32,
    /// Typed meaning of `iold`.
    pub mode: XsphXsectCrossTermMode,
    /// One-based partner transition index, FEFF `k1`.
    pub partner_index_1based: usize,
}

/// Inputs for saving FEFF `XSPH/xsect.f90` cross-term retry state.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectCrossTermStateSaveInput<'a> {
    /// FEFF `iold` retry plan for the active transition.
    pub plan: XsphXsectCrossTermPlan,
    /// One-based current transition index, FEFF `ind`.
    pub transition_index_1based: usize,
    /// Reduced matrix element saved as FEFF `rkk1`.
    pub radial_integral: Complex,
    /// Phase shift saved as FEFF `phold`.
    pub phase_shift: Complex,
    /// Regular radial coupling saved as FEFF `xrcold(1:ilast)`.
    pub regular_coupling: ArrayView1<'a, Complex>,
    /// Irregular radial coupling saved as FEFF `xncold(1:ilast)`.
    pub irregular_coupling: ArrayView1<'a, Complex>,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// Saved FEFF `XSPH/xsect.f90` adjacent same-`l` cross-term state.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectCrossTermState {
    /// One-based transition that supplied `rkk1/phold/xrcold/xncold`.
    pub transition_index_1based: usize,
    /// One-based transition expected to consume this state.
    pub partner_index_1based: usize,
    /// Saved reduced matrix element, FEFF `rkk1`.
    pub radial_integral: Complex,
    /// Saved phase shift, FEFF `phold`.
    pub phase_shift: Complex,
    /// Saved regular radial coupling, FEFF `xrcold(1:ilast)`.
    pub regular_coupling: Array1<Complex>,
    /// Saved irregular radial coupling, FEFF `xncold(1:ilast)`.
    pub irregular_coupling: Array1<Complex>,
}

/// Inputs for reusing saved FEFF cross-term retry state.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectCrossTermStateReuseInput<'a> {
    /// FEFF `iold` retry plan for the active transition.
    pub plan: XsphXsectCrossTermPlan,
    /// One-based current transition index, FEFF `ind`.
    pub transition_index_1based: usize,
    /// State saved from the adjacent previous transition.
    pub state: &'a XsphXsectCrossTermState,
}

/// Saved cross-term state projected into FEFF `radint` branches `3` and `4`.
#[derive(Debug, Clone)]
pub struct XsphXsectCrossTermStateReuse<'a> {
    /// One-based transition that supplied the saved state.
    pub saved_transition_index_1based: usize,
    /// Saved reduced matrix element to pass as FEFF `rkk1`.
    pub saved_radial_integral: Complex,
    /// Saved phase shift to pass as FEFF `phold`.
    pub saved_phase_shift: Complex,
    /// FEFF `radint(ifl=3, iold=2)` branch using saved regular coupling.
    pub radint3_branch: XsphRadialCrossIntegralBranch<'a>,
    /// FEFF `radint(ifl=4, iold=2)` branch using saved irregular coupling.
    pub radint4_branch: XsphRadialCrossIntegralBranch<'a>,
}

/// Inputs for FEFF `XSPH/xsect.f90` XMCD cross-term accumulation.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectCrossTermAccumulationInput<'a> {
    /// One-based current transition index, FEFF `ind`.
    pub transition_index_1based: usize,
    /// Final-state orbital momenta, FEFF `lind(1:indmax)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Number of active transitions, FEFF `indmax`.
    pub active_len: usize,
    /// Saved first same-`l` radial integral, FEFF `rkk1`.
    pub saved_radial_integral: Complex,
    /// Current same-`l` radial integral, FEFF `rkk0`.
    pub current_radial_integral: Complex,
    /// Saved first same-`l` phase shift, FEFF `phold`.
    pub saved_phase_shift: Complex,
    /// Current phase shift, FEFF `ph0`.
    pub current_phase_shift: Complex,
    /// Off-diagonal angular coefficient, FEFF `bmat(...,k1,...,ind)`.
    pub partner_current_weight: Complex,
    /// Off-diagonal angular coefficient, FEFF `bmat(...,ind,...,k1)`.
    pub current_partner_weight: Complex,
    /// Radial cross term from FEFF `radint(ifl=3)`.
    pub radint3_integral: Complex,
    /// Radial cross term from FEFF `radint(ifl=4)`.
    pub radint4_integral: Complex,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
}

/// Inputs for XMCD cross-term accumulation using traced FEFF `bcoef` weights.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefCrossTermAccumulationInput<'a> {
    /// One-based current transition index, FEFF `ind`.
    pub transition_index_1based: usize,
    /// Final-state orbital momenta, FEFF `lind(1:indmax)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Number of active transitions, FEFF `indmax`.
    pub active_len: usize,
    /// Traced angular coefficients, FEFF `bmat(0,isp,k2,0,isp,k1)`.
    pub trace_weights: ArrayView2<'a, Complex>,
    /// Saved first same-`l` radial integral, FEFF `rkk1`.
    pub saved_radial_integral: Complex,
    /// Current same-`l` radial integral, FEFF `rkk0`.
    pub current_radial_integral: Complex,
    /// Saved first same-`l` phase shift, FEFF `phold`.
    pub saved_phase_shift: Complex,
    /// Current phase shift, FEFF `ph0`.
    pub current_phase_shift: Complex,
    /// Radial cross term from FEFF `radint(ifl=3)`.
    pub radint3_integral: Complex,
    /// Radial cross term from FEFF `radint(ifl=4)`.
    pub radint4_integral: Complex,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
}

/// Inputs for bcoef-weighted XMCD accumulation from saved FEFF retry state.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectBcoefCrossTermStateAccumulationInput<'a> {
    /// One-based current transition index, FEFF `ind`.
    pub transition_index_1based: usize,
    /// Final-state orbital momenta, FEFF `lind(1:indmax)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Number of active transitions, FEFF `indmax`.
    pub active_len: usize,
    /// Traced angular coefficients, FEFF `bmat(0,isp,k2,0,isp,k1)`.
    pub trace_weights: ArrayView2<'a, Complex>,
    /// Saved `iold = 2` state projected from FEFF `rkk1/phold/xrcold/xncold`.
    pub state_reuse: &'a XsphXsectCrossTermStateReuse<'a>,
    /// Current same-`l` radial integral, FEFF `rkk0`.
    pub current_radial_integral: Complex,
    /// Current phase shift, FEFF `ph0`.
    pub current_phase_shift: Complex,
    /// Radial cross term from FEFF `radint(ifl=3)`.
    pub radint3_integral: Complex,
    /// Radial cross term from FEFF `radint(ifl=4)`.
    pub radint4_integral: Complex,
    /// Current unnormalized central-atom cross section, FEFF `xsec(ie)`.
    pub cross_section: Complex,
}

/// FEFF `XSPH/xsect.f90` XMCD cross-term accumulation result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphXsectCrossTermAccumulation {
    /// One-based previous transition index, FEFF `k1 = ind - 1`.
    pub partner_index_1based: usize,
    /// FEFF `aa = exp(i*(ph0-phold))`.
    pub phase_factor: Complex,
    /// FEFF `bb = 1/aa`.
    pub inverse_phase_factor: Complex,
    /// Symmetrized angular coupling, FEFF `cc`.
    pub angular_coupling: Complex,
    /// Increment from `-i*rkk1*rkk0*(bb+aa)*cc`.
    pub matrix_cross_term_increment: Complex,
    /// Increment from FEFF `radint(ifl=3) * cc * bb`.
    pub radint3_increment: Complex,
    /// Increment from FEFF `radint(ifl=4) * cc * aa`.
    pub radint4_increment: Complex,
    /// Total FEFF `xsec` increment from the cross-term block.
    pub cross_section_increment: Complex,
    /// Updated unnormalized central-atom cross section.
    pub cross_section: Complex,
}

/// Inputs for the FEFF `XSPH/radjas.f90` `getcorrection` helper.
#[derive(Debug, Clone, Copy)]
pub struct XsphJasOrthogonalityCorrectionInput<'a> {
    /// Doubled initial angular momentum limit, FEFF `jinit`.
    pub initial_j: usize,
    /// Initial-state orbital angular momentum, FEFF `linit`.
    pub initial_l: usize,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub large_component: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub small_component: ArrayView1<'a, Real>,
    /// NRIXS q-Bessel table `qjbess(1:ilast, 0:ljmax, 1:nq)`.
    pub q_bessel: ArrayView3<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Largest active NRIXS multipole, FEFF `ljmax`.
    pub ljmax: usize,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// Orthogonality correction table from FEFF `XSPH/radjas.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphJasOrthogonalityCorrection {
    /// FEFF `ortcor(0:ljmax, 1:nq)`.
    pub corrections: Array2<Complex>,
    /// Bound-spinor normalization integral used as the correction denominator.
    pub normalization: Complex,
}

/// Inputs for FEFF `XSPH/radjas.f90` `getorthg`.
#[derive(Debug, Clone, Copy)]
pub struct XsphJasOverlapInput<'a> {
    /// Initial-state orbital angular momentum, FEFF `linit`.
    pub initial_l: usize,
    /// Final-state orbital/angular selector, FEFF `lfin`.
    pub final_l: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:np)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:np)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Final-state large Dirac component, FEFF `p(1:np)`.
    pub final_large: ArrayView1<'a, Complex>,
    /// Final-state small Dirac component, FEFF `q(1:np)`.
    pub final_small: ArrayView1<'a, Complex>,
    /// Radial grid, FEFF `ri(1:np)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// FEFF `m`, the radial power offset before the internal `m + 1`.
    pub radial_power: usize,
    /// Number of active radial points, FEFF `np`.
    pub active_len: usize,
}

/// Separate large/small component overlaps from FEFF `XSPH/radjas.f90` `getorthg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphJasOverlap {
    /// Large-component overlap, FEFF `ap`.
    pub large_overlap: Complex,
    /// Small-component overlap, FEFF `aq`.
    pub small_overlap: Complex,
    /// Sum of the large and small component overlaps.
    pub total_overlap: Complex,
    /// Near-origin power passed to the corrected Simpson helper.
    pub near_origin_power: Real,
}

/// Inputs for FEFF `XSPH/radjas.f90` `ifl = 1`.
#[derive(Debug, Clone, Copy)]
pub struct XsphJasRadialIntegralInput<'a> {
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state relativistic kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Regular final-state large Dirac component, FEFF `p(1:ilast)`.
    pub final_large_regular: ArrayView1<'a, Complex>,
    /// Regular final-state small Dirac component, FEFF `q(1:ilast)`.
    pub final_small_regular: ArrayView1<'a, Complex>,
    /// Which NRIXS multipoles to evaluate, FEFF `ljneeded(0:ljmax)`.
    pub needed_multipoles: ArrayView1<'a, i32>,
    /// NRIXS q-Bessel table `jbess(1:ilast, 0:ljmax)`.
    pub q_bessel: ArrayView2<'a, Real>,
    /// Orthogonality correction `ortcor(0:ljmax)` for same-kappa channels.
    pub orthogonality_correction: ArrayView1<'a, Complex>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Largest active NRIXS multipole, FEFF `ljmax`.
    pub ljmax: usize,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// FEFF `radjas.f90` reduced radial matrix elements and coupling workspace.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphJasRadialIntegral {
    /// Integrated reduced matrix elements, FEFF `xirf(0:ljmax)`.
    pub radial_integrals: Array1<Complex>,
    /// Radial coupling samples before `csommjas`, FEFF `xrc(1:ilast, 0:ljmax)`.
    pub regular_coupling: Array2<Complex>,
    /// Near-origin powers used for each active multipole.
    pub near_origin_powers: Array1<Real>,
}

/// Inputs for FEFF `XSPH/radjas.f90` `ifl = 2`.
#[derive(Debug, Clone, Copy)]
pub struct XsphJasRadialCrossIntegralInput<'a> {
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state relativistic kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Irregular final-state large Dirac component, FEFF `pn(1:ilast)`.
    pub final_large_irregular: ArrayView1<'a, Complex>,
    /// Irregular final-state small Dirac component, FEFF `qn(1:ilast)`.
    pub final_small_irregular: ArrayView1<'a, Complex>,
    /// Regular coupling from the previous `ifl = 1` call, FEFF `xrc`.
    pub regular_coupling: ArrayView2<'a, Complex>,
    /// Which NRIXS multipoles to evaluate, FEFF `ljneeded(0:ljmax)`.
    pub needed_multipoles: ArrayView1<'a, i32>,
    /// NRIXS q-Bessel table `jbess(1:ilast, 0:ljmax)`.
    pub q_bessel: ArrayView2<'a, Real>,
    /// Orthogonality correction `ortcor(0:ljmax)` for same-kappa channels.
    pub orthogonality_correction: ArrayView1<'a, Complex>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Largest active NRIXS multipole, FEFF `ljmax`.
    pub ljmax: usize,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// FEFF `radjas.f90` central-atom double radial integral and work arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphJasRadialCrossIntegral {
    /// Integrated central-atom cross-section terms, FEFF `xirf(0:ljmax)`.
    pub radial_integrals: Array1<Complex>,
    /// Irregular coupling before prefix weighting, FEFF `xnc`.
    pub irregular_coupling: Array2<Complex>,
    /// Prefix integral of the regular coupling, FEFF's first radial integration.
    pub regular_prefix_integral: Array2<Complex>,
    /// Irregular coupling multiplied by the prefix integral, FEFF's output `xnc`.
    pub weighted_irregular_coupling: Array2<Complex>,
    /// Near-origin power of the regular coupling used by the prefix integral.
    pub first_near_origin_powers: Array1<Real>,
    /// Near-origin powers passed to FEFF `csommjas` for the second integral.
    pub second_near_origin_powers: Array1<Real>,
}

/// Inputs for FEFF `XSPH/radint.f90` `abs(ifl) == 1`.
#[derive(Debug, Clone, Copy)]
pub struct XsphRadialIntegralInput<'a> {
    /// FEFF `ifl` branch exposed by this safe wrapper.
    pub mode: XsphRadialIntegralMode,
    /// FEFF `mult` transition kind.
    pub multipole: XsphTransitionMultipole,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state relativistic kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Regular final-state large Dirac component, FEFF `p(1:ilast)`.
    pub final_large_regular: ArrayView1<'a, Complex>,
    /// Regular final-state small Dirac component, FEFF `q(1:ilast)`.
    pub final_small_regular: ArrayView1<'a, Complex>,
    /// X-ray Bessel functions `bf(0:2, 1:ilast)`.
    pub xray_bessel: ArrayView2<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// FEFF `radint.f90` cross-section branch selector.
#[derive(Debug, Clone)]
pub enum XsphRadialCrossIntegralBranch<'a> {
    /// FEFF `ifl = 2, iold = 0`: compute current regular and irregular couplings.
    CurrentRegularAndIrregular,
    /// FEFF `ifl = 3, iold = 2`: reuse a stored regular coupling.
    StoredRegularCurrentIrregular {
        /// FEFF `xrcold(1:ilast)` from a previous same-`l` final state.
        stored_regular_coupling: ArrayView1<'a, Complex>,
    },
    /// FEFF `ifl = 4, iold = 2`: reuse a stored irregular coupling.
    CurrentRegularStoredIrregular {
        /// FEFF `xncold(1:ilast)` from a previous same-`l` final state.
        stored_irregular_coupling: ArrayView1<'a, Complex>,
    },
}

/// Inputs for FEFF `XSPH/radint.f90` cross-section branches.
#[derive(Debug, Clone)]
pub struct XsphRadialCrossIntegralInput<'a> {
    /// Relativistic or nonrelativistic coupling equations.
    pub mode: XsphRadialIntegralMode,
    /// FEFF `ifl`/`iold` cross-section branch.
    pub branch: XsphRadialCrossIntegralBranch<'a>,
    /// FEFF `mult` transition kind.
    pub multipole: XsphTransitionMultipole,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state relativistic kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Regular final-state large Dirac component, FEFF `p(1:ilast)`.
    pub final_large_regular: ArrayView1<'a, Complex>,
    /// Regular final-state small Dirac component, FEFF `q(1:ilast)`.
    pub final_small_regular: ArrayView1<'a, Complex>,
    /// Irregular final-state large Dirac component, FEFF `pn(1:ilast)`.
    pub final_large_irregular: ArrayView1<'a, Complex>,
    /// Irregular final-state small Dirac component, FEFF `qn(1:ilast)`.
    pub final_small_irregular: ArrayView1<'a, Complex>,
    /// X-ray Bessel functions `bf(0:2, 1:ilast)`.
    pub xray_bessel: ArrayView2<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// FEFF `radint.f90` reduced radial matrix element and coupling workspace.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphRadialIntegral {
    /// Integrated reduced matrix element, FEFF `xirf`.
    pub value: Complex,
    /// Radial coupling samples before Simpson integration, FEFF `xrc`.
    pub coupling: Array1<Complex>,
    /// Near-origin power passed to FEFF `csomm`.
    pub near_origin_power: Real,
}

/// Inputs for FEFF `XSPH/xsect.f90` radial integrals weighted by `fscf`.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectWeightedRadialIntegralInput<'a> {
    /// FEFF `radint` matrix-element branch.
    pub mode: XsphRadialIntegralMode,
    /// FEFF `mult` transition kind.
    pub multipole: XsphTransitionMultipole,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state relativistic kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Regular final-state large Dirac component, FEFF `p(1:ilast)`.
    pub final_large_regular: ArrayView1<'a, Complex>,
    /// Regular final-state small Dirac component, FEFF `q(1:ilast)`.
    pub final_small_regular: ArrayView1<'a, Complex>,
    /// X-ray Bessel functions `bf(0:2, 1:ilast)`.
    pub xray_bessel: ArrayView2<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Real `fscf` component for the active pass.
    pub radial_weights: ArrayView1<'a, Real>,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// FEFF `XSPH/xsect.f90` weighted reduced radial integral.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectWeightedRadialIntegral {
    /// Weighted radial integral. Its `coupling` field includes `radial_weights`.
    pub integral: XsphRadialIntegral,
    /// Unweighted radial coupling before the `fscf` component is applied.
    pub unweighted_coupling: Array1<Complex>,
}

/// FEFF `radint.f90` cross-section radial integral and work arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphRadialCrossIntegral {
    /// Integrated central-atom cross-section term, FEFF `xirf`.
    pub value: Complex,
    /// Regular radial coupling before FEFF zeroes `xrc` for the second integral.
    pub regular_coupling: Array1<Complex>,
    /// Irregular radial coupling before prefix weighting.
    pub irregular_coupling: Array1<Complex>,
    /// Prefix integral of the regular coupling, FEFF's first radial integration.
    pub regular_prefix_integral: Array1<Complex>,
    /// Irregular coupling multiplied by the prefix integral, FEFF `xnc`.
    pub weighted_irregular_coupling: Array1<Complex>,
    /// Near-origin power of the regular coupling used by the prefix integral.
    pub first_near_origin_power: Real,
    /// Near-origin power passed to FEFF `csomm` for the second integral.
    pub second_near_origin_power: Real,
}

/// Inputs for FEFF `XSPH/xsect.f90` cross integrals weighted by `fscf`.
#[derive(Debug, Clone)]
pub struct XsphXsectWeightedRadialCrossIntegralInput<'a, 'b> {
    /// FEFF `radint` matrix-element branch.
    pub mode: XsphRadialIntegralMode,
    /// FEFF `iold` branch for current/stored regular and irregular couplings.
    pub branch: XsphRadialCrossIntegralBranch<'b>,
    /// FEFF `mult` transition kind.
    pub multipole: XsphTransitionMultipole,
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Final-state relativistic kappa, FEFF `ikap`.
    pub final_kappa: i32,
    /// Initial-state large Dirac component, FEFF `dgc0(1:ilast)`.
    pub initial_large: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0(1:ilast)`.
    pub initial_small: ArrayView1<'a, Real>,
    /// Regular final-state large Dirac component, FEFF `p(1:ilast)`.
    pub final_large_regular: ArrayView1<'a, Complex>,
    /// Regular final-state small Dirac component, FEFF `q(1:ilast)`.
    pub final_small_regular: ArrayView1<'a, Complex>,
    /// Irregular final-state large Dirac component, FEFF `pn(1:ilast)`.
    pub final_large_irregular: ArrayView1<'a, Complex>,
    /// Irregular final-state small Dirac component, FEFF `qn(1:ilast)`.
    pub final_small_irregular: ArrayView1<'a, Complex>,
    /// X-ray Bessel functions `bf(0:2, 1:ilast)`.
    pub xray_bessel: ArrayView2<'a, Real>,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Real `fscf` component applied to the regular coupling.
    pub regular_weights: ArrayView1<'a, Real>,
    /// Real `fscf` component applied to the irregular coupling.
    pub irregular_weights: ArrayView1<'a, Real>,
    /// Number of active radial points, FEFF `ilast`.
    pub active_len: usize,
}

/// FEFF `XSPH/xsect.f90` weighted central cross-section radial integral.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectWeightedRadialCrossIntegral {
    /// Weighted radial cross integral. Its coupling fields include weights.
    pub integral: XsphRadialCrossIntegral,
    /// Unweighted regular coupling before `regular_weights` are applied.
    pub unweighted_regular_coupling: Array1<Complex>,
    /// Unweighted irregular coupling before `irregular_weights` are applied.
    pub unweighted_irregular_coupling: Array1<Complex>,
}

/// Inputs for the empty-cell branch in FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphEmptyCellPhaseInput {
    /// Muffin-tin radius, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Complex wave number for the active potential, FEFF `ck`.
    pub wave_number: Complex,
    /// Complex wave number for the empty-cell reference, FEFF `ckEC`.
    pub empty_cell_wave_number: Complex,
    /// Relativistic final-state kappa, FEFF `ikap`.
    pub kappa: i32,
}

/// Inputs for the normal-potential phase match in FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphRegularPhaseInput {
    /// Muffin-tin radius, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Complex wave number for the active potential, FEFF `ck`.
    pub wave_number: Complex,
    /// Regular large radial component at `rmt` returned by `dfovrg`, FEFF `pu`.
    pub regular_large_at_muffin_tin: Complex,
    /// Regular small radial component at `rmt` returned by `dfovrg`, FEFF `qu`.
    pub regular_small_at_muffin_tin: Complex,
    /// Relativistic final-state kappa, FEFF `ikap`.
    pub kappa: i32,
}

/// Inputs for preparing FEFF `XSPH/phase.f90` potential-local grids.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhaseGridPreparationInput<'a> {
    /// Muffin-tin radii `rmt(0:nph)` in Bohr.
    pub muffin_tin_radii: &'a [Real],
    /// Ground-state total electron density from `pot.bin`, FEFF `edens`.
    pub electron_density: ArrayView2<'a, Real>,
    /// Ground-state total potential from `pot.bin`, FEFF `vtot`.
    pub total_potential: ArrayView2<'a, Real>,
    /// Ground-state valence electron density from `pot.bin`, FEFF `edenvl`.
    pub valence_density: ArrayView2<'a, Real>,
    /// Ground-state valence potential from `pot.bin`, FEFF `vvalgs`.
    pub valence_potential: ArrayView2<'a, Real>,
    /// Density magnetization from `pot.bin`, FEFF `dmag`.
    pub magnetization: ArrayView2<'a, Real>,
    /// Bound large Dirac components, FEFF `dgc`.
    pub bound_large_components: ArrayView3<'a, Real>,
    /// Bound small Dirac components, FEFF `dpc`.
    pub bound_small_components: ArrayView3<'a, Real>,
    /// Interstitial potential `vint`.
    pub interstitial_potential: Real,
    /// Interstitial density `rhoint`.
    pub interstitial_density: Real,
    /// Source-grid logarithmic step, FEFF `dx = 0.05`.
    pub original_radial_dx: Real,
    /// Target phase-grid logarithmic step, FEFF `dxnew = rgrd`.
    pub target_radial_dx: Real,
    /// FEFF jump mode `jumprm`.
    pub jump_mode: i32,
    /// Initial `vjump`.
    pub potential_jump: Real,
    /// Exchange selector `ixc`.
    pub exchange_selector: i32,
    /// Target radial table length, normally FEFF `nrptx`.
    pub radial_count: usize,
}

/// FEFF `XSPH/phase.f90` grids after `fixvar` and `fixdsx`, before `xcpot`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphPhaseGridPreparation {
    /// Target radial grid `ri`.
    pub radii: Array1<Real>,
    /// Target logarithmic radial spacing, FEFF `dxnew`.
    pub radial_dx: Real,
    /// Per-potential total-potential jump after the total `fixvar` pass.
    pub potential_jumps: Array1<Real>,
    /// Unreferenced total potentials, FEFF `vtotph(row, iph)`.
    pub total_potential: Array2<Real>,
    /// Unreferenced valence potentials, FEFF `vvalph(row, iph)`.
    pub valence_potential: Array2<Real>,
    /// Prepared total charge density, FEFF `rhoph(row, iph)`.
    pub electron_density: Array2<Real>,
    /// Prepared valence charge density, FEFF `rhphvl(row, iph)`.
    pub valence_density: Array2<Real>,
    /// Prepared magnetization density, FEFF `dmagx(row, iph)`.
    pub magnetization: Array2<Real>,
    /// Resampled bound large Dirac components `dgcn(row, orbital, iph)`.
    pub bound_large_components: Array3<Real>,
    /// Resampled bound small Dirac components `dpcn(row, orbital, iph)`.
    pub bound_small_components: Array3<Real>,
    /// Per-orbital active lengths from FEFF `fixdsx`, shaped `(orbital, iph)`.
    pub bound_active_lengths: Array2<usize>,
}

/// Inputs for the angular cutoff planning in FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhaseAngularLimitInput<'a> {
    /// FEFF `em(1:ne)`, the complex phase-energy mesh.
    pub energies: ArrayView1<'a, Complex>,
    /// Total number of energy points, FEFF `ne`.
    pub energy_count: usize,
    /// Auxiliary horizontal point count excluded from the `kmax` scan, FEFF `ne3`.
    pub auxiliary_count: usize,
    /// Muffin-tin radius, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Maximum available angular momentum, FEFF `ltot`.
    pub max_angular_momentum: usize,
}

/// Angular cutoff selected by FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseAngularLimit {
    /// Final capped phase-shift limit, FEFF `lmax`.
    pub angular_limit: usize,
    /// Limit after FEFF's `max(lmax, 5)` floor but before the `ltot` cap.
    pub uncapped_limit: usize,
    /// Maximum wave number derived from the scanned real-energy prefix.
    pub max_wave_number: Real,
    /// FEFF's diagnostic `k` value when the requested limit exceeds `ltot`.
    pub accuracy_warning_wave_number: Option<i32>,
}

/// Inputs for the per-energy setup block in FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhaseEnergySetupInput {
    /// Current complex energy, FEFF `em(ie)`.
    pub energy: Complex,
    /// Energy-dependent reference potential, FEFF `eref(ie)`.
    pub reference_energy: Complex,
    /// Muffin-tin potential at the first radial point, FEFF `vtot(1)`.
    pub muffin_tin_potential: Real,
    /// FEFF `lreal`; values greater than one force real `p2` on the real mesh.
    pub lreal: i32,
    /// Zero-based energy index, equal to FEFF `ie - 1`.
    pub energy_index: usize,
    /// Number of real-mesh energies, FEFF `ne1`.
    pub real_mesh_count: usize,
    /// Muffin-tin radius, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Exchange selector, FEFF `ixc`.
    pub exchange_selector: i32,
}

/// FEFF branch selected by the per-energy phase setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphPhaseEnergyDecision {
    /// Continue into Bessel functions and radial matching for this energy.
    Active,
    /// Skip because `Re(em)` is outside FEFF's `[-10, 300]` phase window.
    OutsideEnergyWindow,
    /// Skip after computing momenta because `Re(p2) <= 0` and `Im(p2) <= 0`.
    NonPositiveMomentum,
}

/// Momentum quantities computed by the per-energy phase setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseEnergyDynamics {
    /// Complex momentum squared, FEFF `p2`.
    pub momentum_squared: Complex,
    /// Empty-cell momentum squared, FEFF `p2EC`.
    pub empty_cell_momentum_squared: Complex,
    /// Relativistic wave number, FEFF `ck`.
    pub wave_number: Complex,
    /// Empty-cell relativistic wave number, FEFF `ckEC`.
    pub empty_cell_wave_number: Complex,
    /// Muffin-tin Bessel argument, FEFF `xkmt`.
    pub muffin_tin_argument: Complex,
    /// Empty-cell muffin-tin Bessel argument, FEFF `xkmtEC`.
    pub empty_cell_muffin_tin_argument: Complex,
}

/// Result of the per-energy setup block in FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseEnergySetup {
    /// FEFF branch decision for this energy.
    pub decision: XsphPhaseEnergyDecision,
    /// Computed momentum quantities, absent when FEFF skips before `p2`.
    pub dynamics: Option<XsphPhaseEnergyDynamics>,
    /// FEFF `ncycle`, present only when the energy continues past Bessel setup.
    pub cycle_count: Option<usize>,
}

/// Inputs for the per-angular-channel setup loop in FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhaseChannelPlanInput {
    /// Largest absolute FEFF angular channel, FEFF `lmax`.
    pub angular_limit: usize,
    /// Logarithmic radial-grid spacing, FEFF `dx`.
    pub log_step: Real,
    /// Cycle count selected before the angular-channel loop, FEFF `ncycle`.
    pub initial_cycle_count: usize,
    /// Spin channel count, FEFF `nsp`.
    pub spin_channels: i32,
    /// Spin selector, FEFF `ispin`.
    pub spin: i32,
}

/// One per-`ll` channel setup row from FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphPhaseChannel {
    /// Signed angular channel, FEFF `ll`.
    pub angular_channel: i32,
    /// FEFF one-based channel index `il = abs(ll) + 1`.
    pub orbital_index: usize,
    /// Partner one-based channel index, FEFF `ilp`.
    pub partner_orbital_index: usize,
    /// Relativistic final-state kappa, FEFF `ikap`.
    pub kappa: i32,
    /// FEFF spin-orbit-removal derivative selector `ic3`.
    pub c3_derivative: i32,
    /// Effective `dfovrg` cycle count at this channel, FEFF `ncycle`.
    pub cycle_count: usize,
    /// Whether this channel directly triggered FEFF's local-exchange fallback.
    pub forces_local_exchange: bool,
}

/// Per-angular-channel setup plan from FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsphPhaseChannelPlan {
    /// Rows in FEFF traversal order, `ll = -lmax..lmax`.
    pub channels: Vec<XsphPhaseChannel>,
}

/// Inputs for the small-phase cutoff block after FEFF `phamp`.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhaseCutoffInput {
    /// Signed angular channel, FEFF `ll`.
    pub angular_channel: i32,
    /// Complex phase shift for this energy/channel, FEFF `ph(ie,ll)`.
    pub phase_shift: Complex,
}

/// Result of FEFF `XSPH/phase.f90` small-phase cutoff handling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseCutoff {
    /// Phase shift after FEFF's optional zeroing branch.
    pub phase_shift: Complex,
    /// Whether FEFF set the phase shift to zero via `exp(2i*ph)-1`.
    pub zeroed: bool,
    /// Whether FEFF would `goto 220` and stop this energy's angular loop.
    pub terminate_energy: bool,
}

/// Result of FEFF XSPH phase reference-energy tail finalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XsphPhaseReferenceTail {
    /// One-based loop start used by the FEFF tail-copy branch.
    pub start_index_1based: usize,
    /// Number of active reference-energy entries overwritten.
    pub filled_count: usize,
}

/// Inputs for FEFF `XSPH/phase.f90` muffin-tin radial-index setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseRadialIndicesInput {
    /// Muffin-tin radius, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Loucks-grid origin, FEFF `x0`.
    pub grid_origin: Real,
    /// Loucks logarithmic grid step, FEFF `dx`.
    pub log_step: Real,
    /// Available radial rows, FEFF `nrptx`.
    pub radial_capacity: usize,
}

/// FEFF `imt`/`jri`/`jri1` indices for the XSPH phase radial grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseRadialIndices {
    /// Raw real value assigned to integer `imt`.
    pub raw_muffin_tin_index: Real,
    /// FEFF integer `imt`. FEFF's real-to-integer assignment truncates toward zero.
    pub muffin_tin_index: i32,
    /// FEFF one-based `jri = imt + 1`.
    pub radial_match_index_1based: usize,
    /// FEFF one-based `jri1 = jri + 1`.
    pub reference_index_1based: usize,
}

/// Inputs for the FEFF `XSPH/phase.f90` `mpse.dat` self-energy summary row.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhaseSelfEnergySummaryInput<'a> {
    /// Total electron density on the Loucks radial grid, FEFF `edens`.
    pub electron_density: ArrayView1<'a, Real>,
    /// FEFF one-based index `jri + 1` used for the interstitial density sample.
    pub reference_index_1based: usize,
}

/// FEFF `mpse.dat` header values derived from `edens(jri+1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseSelfEnergySummary {
    /// Electron density sampled at FEFF `edens(jri+1)`.
    pub electron_density: Real,
    /// Wigner-Seitz radius, FEFF `(3/(4*pi*edens(jri+1)))**third`.
    pub wigner_seitz_radius: Real,
    /// Plasma frequency in eV, FEFF `sqrt(3/rs**3)*hart`.
    pub plasma_frequency_ev: Real,
}

/// Inputs for the FEFF `XSPH/phase.f90` MPSE plasmon-pole setup branch.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhasePlasmonPoleSetupInput<'a> {
    /// FEFF `iPl`; values greater than zero enable many-pole setup.
    pub plasmon_selector: i32,
    /// FEFF `ixc`; only Hedin-Lundqvist (`ixc == 0`) enters this branch.
    pub exchange_selector: i32,
    /// Total electron density on the Loucks radial grid, FEFF `edens`.
    pub electron_density: ArrayView1<'a, Real>,
    /// FEFF one-based index `jri + 1` used for the interstitial density sample.
    pub reference_index_1based: usize,
    /// FEFF `MkExc` pole rows before `phase.f90` rescales `Wi` and `Gamma`.
    pub excitation_poles: &'a [ExcitationPole],
}

/// One many-pole row after FEFF `phase.f90` MPSE rescaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhasePlasmonPole {
    /// FEFF `WpCorr`: pole energy in Hartree divided by the local plasma frequency.
    pub energy_over_plasma: Real,
    /// FEFF `Gamma`: pole width converted from eV to Hartree.
    pub width_hartree: Real,
    /// FEFF `AmpFac`: pole amplitude carried through unchanged from `MkExc`.
    pub amplitude: Real,
}

/// FEFF MPSE plasmon-pole setup data passed into `xcpot`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphPhasePlasmonPoleSetup {
    /// Electron density sampled at FEFF `edens(jri+1)`.
    pub electron_density: Real,
    /// Wigner-Seitz radius, FEFF `(3/(4*pi*edens(jri+1)))**third`.
    pub wigner_seitz_radius: Real,
    /// Local plasma frequency in Hartree, FEFF `sqrt(3/rs**3)`.
    pub plasma_frequency_hartree: Real,
    /// Local plasma frequency in eV.
    pub plasma_frequency_ev: Real,
    /// Active pole rows in FEFF `MkExc` order.
    pub poles: Vec<XsphPhasePlasmonPole>,
}

/// Inputs for the FEFF `XSPH/phase.f90` `PrintRl` header branch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseRadialHeaderInput {
    /// FEFF `PrintRl` flag.
    pub print_radial: bool,
    /// Potential index, FEFF `iph`.
    pub potential_index: i32,
    /// Muffin-tin radius written to the first `rl.dat` header row, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Active angular cutoff written as FEFF `lmax`.
    pub angular_limit: usize,
    /// FEFF one-based radial match row written as `jri`.
    pub radial_match_index_1based: usize,
    /// Loucks logarithmic grid step written to the second header row, FEFF `dx`.
    pub log_step: Real,
    /// Loucks-grid origin written to the second header row, FEFF `x0`.
    pub grid_origin: Real,
}

/// FEFF `rl.dat` header values emitted before `PrintRl` radial rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseRadialHeader {
    /// Muffin-tin radius, FEFF `rmt`.
    pub muffin_tin_radius: Real,
    /// Active angular cutoff, FEFF `lmax`.
    pub angular_limit: usize,
    /// FEFF one-based radial match row, `jri`.
    pub radial_match_index_1based: usize,
    /// Loucks logarithmic grid step, FEFF `dx`.
    pub log_step: Real,
    /// Loucks-grid origin, FEFF `x0`.
    pub grid_origin: Real,
}

/// Inputs for FEFF `XSPH/phase_h.f90` Hubbard phase potential shifts.
#[derive(Debug, Clone, Copy)]
pub struct XsphHubbardPhasePotentialInput<'a> {
    /// Nonnegative angular channel, FEFF `ll`.
    pub angular_channel: i32,
    /// Hubbard spin selector, FEFF `is_p`.
    pub spin_projection: i32,
    /// Total-potential channel, FEFF `v(1:nrptx)`.
    pub total_potential: ArrayView1<'a, Complex>,
    /// Valence-potential channel, FEFF `vval(1:nrptx)`.
    pub valence_potential: ArrayView1<'a, Complex>,
    /// Hubbard shifts, FEFF `Vnlm(0:lx, 1:(lx+1)**2)` with zero-based columns.
    pub hubbard_potential: ArrayView2<'a, Real>,
    /// Active potential prefix to shift.
    pub active_len: usize,
}

/// One magnetic Hubbard potential shift from FEFF `XSPH/phase_h.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphHubbardPhasePotentialShift {
    /// Zero-based magnetic channel corresponding to FEFF `imm - 1`.
    pub magnetic_channel: usize,
    /// Signed scalar shift applied to both potential arrays.
    pub shift: Real,
    /// Shifted total-potential prefix, FEFF `vtotc_tmp`.
    pub total_potential: Array1<Complex>,
    /// Shifted valence-potential prefix, FEFF `vvalc_tmp`.
    pub valence_potential: Array1<Complex>,
}

/// Inputs for FEFF `XSPH/phase_h.f90` Hubbard `aph` phase assignments.
#[derive(Debug, Clone, Copy)]
pub struct XsphHubbardPhaseAssignmentInput<'a> {
    /// Zero-based energy row corresponding to FEFF `ie - 1`.
    pub energy_index: usize,
    /// Relativistic angular channel, FEFF `ll`.
    pub angular_channel: i32,
    /// Maximum Hubbard angular channel, FEFF `lx`.
    pub hubbard_angular_limit: usize,
    /// Per-magnetic-channel perturbed phase shifts from the inner `imm` loop.
    pub magnetic_phase_shifts: ArrayView1<'a, Complex>,
}

/// One FEFF `aph(ie,ll+1,imm)=ph_m(ie,ll)` workspace assignment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphHubbardPhaseAssignment {
    /// Zero-based energy row corresponding to FEFF `ie - 1`.
    pub energy_index: usize,
    /// Nonnegative FEFF `ll` channel used as the Rust angular index.
    pub angular_channel: usize,
    /// Zero-based magnetic channel corresponding to FEFF `imm - 1`.
    pub magnetic_channel: usize,
    /// Perturbed phase shift stored in FEFF `aph`.
    pub phase_shift: Complex,
}

/// Inputs for the FEFF `XSPH/phase.f90` `PrintRl` radial-output branch.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhaseRadialOutputInput<'a> {
    /// FEFF `PrintRl` flag.
    pub print_radial: bool,
    /// Potential index, FEFF `iph`.
    pub potential_index: i32,
    /// Relativistic angular channel, FEFF `ll`.
    pub angular_channel: i32,
    /// Active angular cutoff, FEFF `lmax`.
    pub angular_limit: usize,
    /// Current phase-grid energy, FEFF `em(ie)`.
    pub energy: Complex,
    /// Current phase shift, FEFF `ph(ie,ll)`.
    pub phase_shift: Complex,
    /// Phase amplitude returned by `phamp`, FEFF `temp`.
    pub phase_amplitude: Complex,
    /// Large radial component, FEFF `p(1:jri)`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Small radial component, FEFF `q(1:jri)`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Active radial prefix, FEFF `jri`.
    pub active_len: usize,
}

/// Normalized radial data FEFF writes into `rl.dat` from `XSPH/phase.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphPhaseRadialOutput {
    /// Current phase-grid energy, FEFF `em(ie)`.
    pub energy: Complex,
    /// Original FEFF `ll` channel that passed the `PrintRl` predicate.
    pub angular_channel: i32,
    /// Angular momentum written as FEFF `Int2 = -ll`.
    pub output_angular_momentum: usize,
    /// Current phase shift, FEFF `ph(ie,ll)`.
    pub phase_shift: Complex,
    /// Large radial component after FEFF divides `p(1:jri)` by `temp`.
    pub regular_large: Array1<Complex>,
    /// Small radial component after FEFF divides `q(1:jri)` by `temp`.
    pub regular_small: Array1<Complex>,
}

/// Empty-cell phase-matching result from FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphEmptyCellPhase {
    /// Complex scattering phase shift returned by `phamp`.
    pub phase_shift: Complex,
    /// Complex phase amplitude returned by `phamp`.
    pub phase_amplitude: Complex,
    /// Empty-cell large radial component at `rmt`, FEFF `pu`.
    pub regular_large_at_muffin_tin: Complex,
    /// Empty-cell small radial component at `rmt`, FEFF `qu`.
    pub regular_small_at_muffin_tin: Complex,
    /// Orbital angular momentum used for the large component.
    pub large_l: usize,
    /// Orbital angular momentum used for the small component.
    pub small_l: usize,
    /// Active-potential spherical Bessel `j_l(ck*rmt)`.
    pub bessel_j_large: Complex,
    /// Active-potential spherical Neumann `y_l(ck*rmt)`.
    pub neumann_large: Complex,
    /// Active-potential spherical Bessel `j_l'(ck*rmt)` for the small component.
    pub bessel_j_small: Complex,
    /// Active-potential spherical Neumann `y_l'(ck*rmt)` for the small component.
    pub neumann_small: Complex,
    /// Empty-cell spherical Bessel for the large component, `j_l(ckEC*rmt)`.
    pub empty_cell_bessel_j_large: Complex,
    /// Empty-cell spherical Bessel for the small component, `j_l'(ckEC*rmt)`.
    pub empty_cell_bessel_j_small: Complex,
}

/// Normal-potential phase-matching result from FEFF `XSPH/phase.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphRegularPhase {
    /// Complex scattering phase shift returned by `phamp`.
    pub phase_shift: Complex,
    /// Complex phase amplitude returned by `phamp`.
    pub phase_amplitude: Complex,
    /// FOVRG large radial component at `rmt`, FEFF `pu`.
    pub regular_large_at_muffin_tin: Complex,
    /// FOVRG small radial component at `rmt`, FEFF `qu`.
    pub regular_small_at_muffin_tin: Complex,
    /// Orbital angular momentum used for the large component.
    pub large_l: usize,
    /// Orbital angular momentum used for the small component.
    pub small_l: usize,
    /// Active-potential spherical Bessel `j_l(ck*rmt)`.
    pub bessel_j_large: Complex,
    /// Active-potential spherical Neumann `y_l(ck*rmt)`.
    pub neumann_large: Complex,
    /// Active-potential spherical Bessel `j_l'(ck*rmt)` for the small component.
    pub bessel_j_small: Complex,
    /// Active-potential spherical Neumann `y_l'(ck*rmt)` for the small component.
    pub neumann_small: Complex,
}

/// Regular FOVRG channel solution and matched XSPH phase shift.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphRegularPhaseChannel {
    /// Regular `dfovrg` output for this XSPH phase channel.
    pub regular_solution: FovrgDiracSolution,
    /// Phase/amplitude recovered from the regular solution at `rmt`.
    pub phase: XsphRegularPhase,
}

/// FEFF `specupdlg` branch for regular or irregular radial contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphSpectrumUpdateMode {
    /// FEFF `imode = 1`, regular radial-integral branch.
    Regular,
    /// FEFF `imode = 2`, irregular radial-integral branch.
    Irregular,
}

/// Inputs for the FEFF `XSPH/xsphsub.f90` final `xsect.dat` spin merge.
#[derive(Debug, Clone, Copy)]
pub struct XsphXsectSpinMergeInput<'a> {
    /// Whether FEFF takes the two-spin XMCD branch, `abs(ispin).eq.1 .and. nspx.ne.1`.
    pub spin_polarized: bool,
    /// Per-spin normalized backgrounds for one energy, FEFF `xsnorm(ie,1:nspx)`.
    pub spectrum_norms: ArrayView1<'a, Real>,
    /// Per-spin atomic cross sections for one energy, FEFF `xsec(ie,1:nspx)`.
    pub cross_sections: ArrayView1<'a, Complex>,
    /// Active reduced matrix row, FEFF `rkk(ie,1:nq,1:kfinmax,1:nspx)`.
    pub reduced_matrix_elements: ArrayView3<'a, Complex>,
    /// Number of active q rows, FEFF `nq`.
    pub q_count: usize,
    /// Number of active transition columns, FEFF `kfinmax`.
    pub transition_count: usize,
}

/// FEFF `XSPH/xsphsub.f90` final `xsect.dat` spin-merged row.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXsectSpinMerge {
    /// Value written to the `xsnorm` column of `xsect.dat`.
    pub spectrum_norm: Real,
    /// Value written to the complex `xsec` columns of `xsect.dat`.
    pub cross_section: Complex,
    /// FEFF `xnorm1`/`xnorm2` when the two-spin `rkk` normalization branch runs.
    pub spin_scales: Option<[Real; 2]>,
    /// Reduced matrix row after the FEFF spin-average normalization rule.
    pub reduced_matrix_elements: Array3<Complex>,
}

/// Inputs for FEFF `TDLDA/dmscf.f90` screened dipole solve.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaScreenedDipoleInput<'a> {
    /// Number of active energy rows, FEFF `ne`.
    pub energy_count: usize,
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Response matrix, FEFF `chi0(1:ne,1:matsize,1:matsize)`.
    pub response: ArrayView3<'a, Complex>,
    /// Interaction kernel, FEFF `xkmat(1:ne,1:matsize,1:matsize)`.
    pub kernel: ArrayView3<'a, Complex>,
    /// Localized dipole matrix elements, FEFF `dipmat(1:ne,1:matsize)`.
    pub dipole_matrix: ArrayView2<'a, Real>,
}

/// FEFF `TDLDA/dmscf.f90` screened dipole rows.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaScreenedDipole {
    /// Screened matrix elements, FEFF `dipscf(1:ne,1:matsize)`.
    pub screened_dipoles: Array2<Complex>,
}

/// Inputs for FEFF `TDLDA/xsectd.f90` per-energy setup before `getchi0`.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaEnergyRowsInput<'a> {
    /// Number of active energy rows, FEFF `ne1`.
    pub energy_count: usize,
    /// TDLDA energy grid, FEFF `emr(1:ne1)`, in Hartree.
    pub energy_hartree: ArrayView1<'a, Real>,
    /// Energy-dependent reference potential from `xcpot`, FEFF `eref(ie)`.
    pub reference_energy: ArrayView1<'a, Complex>,
    /// Plus-channel onset, FEFF `edge`, in Hartree.
    pub edge_energy: Real,
    /// Chemical-potential position in the absorption spectrum, FEFF `emu`.
    pub chemical_potential: Real,
    /// Spin-orbit split between `l -> l + 1` edges, FEFF `deltaso`.
    pub spin_orbit_split: Real,
    /// FEFF `ipmbse`, used to build the TDLDA/PMBSE separation function.
    pub ipmbse: i32,
}

/// FEFF `TDLDA/xsectd.f90` per-energy arrays before raw `getchi0` generation.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaEnergyRows {
    /// Photon energies, FEFF `omega(1:ne1)`, in Hartree.
    pub photon_energy: Array1<Real>,
    /// Real part of FEFF `ckl3(1:ne1)`.
    pub plus_wave_number: Array1<Real>,
    /// Real part of FEFF `ckl2(1:ne1)`.
    pub minus_wave_number: Array1<Real>,
    /// FEFF PMBSE/TDLDA separation function, `sfun(1:ne1)`.
    pub separation_function: Array1<Real>,
    /// Rows that proceed past FEFF's `emr(ie) < -10` skip.
    pub active_rows: Array1<bool>,
}

/// Inputs for FEFF `TDLDA/getchi0.f90` per-channel wave-number setup.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaRowWaveNumbersInput<'a> {
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Current energy row, FEFF `em`, in Hartree.
    pub energy_hartree: Real,
    /// Energy-dependent reference potential, FEFF `eref`.
    ///
    /// `getchi0` uses only `dble(eref)` for this row setup.
    pub reference_energy: Complex,
    /// Per-channel reference shifts, FEFF `refsh(1:matsize)`.
    pub reference_shifts: ArrayView1<'a, Real>,
}

/// FEFF `TDLDA/getchi0.f90` per-channel momentum and real wave numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaRowWaveNumbers {
    /// Real momentum-squared values, FEFF `p2m(1:matsize)` before the radial solve.
    pub momentum_squared: Array1<Real>,
    /// Real part of `ck = sqrt(2*p2 + (p2*alphfs)**2)`, FEFF `dble(ck)`.
    pub row_wave_numbers: Array1<Real>,
    /// Rows satisfying FEFF's first propagation check, `dble(p2) > 0`.
    pub positive_momentum_rows: Array1<bool>,
}

/// Inputs for FEFF `TDLDA/getchi0.f90` raw overlap response assembly.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaRawResponseInput<'a> {
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Number of `lin + 1` projector orbitals, FEFF `nfo`.
    pub plus_basis_count: usize,
    /// Number of `lin - 1` projector orbitals, FEFF `npo`.
    pub minus_basis_count: usize,
    /// Initial-state orbital angular momentum, FEFF `lin`.
    pub initial_l: i32,
    /// Current energy row, FEFF `em`, in Hartree.
    pub energy_hartree: Real,
    /// Shifted Fermi level, FEFF `edge`, in Hartree.
    pub edge_energy: Real,
    /// Per-channel reference shifts, FEFF `refsh(1:matsize)`.
    pub reference_shifts: ArrayView1<'a, Real>,
    /// Per-row real photoelectron wave number, FEFF `dble(ck)`.
    pub row_wave_numbers: ArrayView1<'a, Real>,
    /// Localized overlap integrals, FEFF `ovrl(1:matsize)`.
    pub overlaps: ArrayView1<'a, Real>,
    /// Localized dipoles before thresholding, FEFF `dipmatl(1:matsize)`.
    pub localized_dipoles: ArrayView1<'a, Real>,
    /// Full dipoles before thresholding, FEFF `dipmat(1:matsize)`.
    pub full_dipoles: ArrayView1<'a, Real>,
}

/// FEFF `TDLDA/getchi0.f90` raw imaginary response for one energy row.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaRawResponse {
    /// Imaginary response matrix, FEFF local `chi0im(1:matsize,1:matsize)`.
    pub raw_imaginary_response: Array2<Real>,
    /// Localized dipoles after FEFF unoccupied-row thresholding.
    pub localized_dipoles: Array1<Real>,
    /// Full dipoles after FEFF unoccupied-row thresholding.
    pub full_dipoles: Array1<Real>,
    /// Rows satisfying FEFF `em >= edge - refsh(im)`.
    pub occupied_rows: Array1<bool>,
}

/// Inputs for FEFF `TDLDA/getchi0.f90` projected-kernel row folding.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaProjectedKernelInput<'a> {
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Number of `lin + 1` projector orbitals, FEFF `nfo`.
    pub plus_basis_count: usize,
    /// Number of `lin - 1` projector orbitals, FEFF `npo`.
    pub minus_basis_count: usize,
    /// Initial-state orbital angular momentum, FEFF `lin`.
    pub initial_l: i32,
    /// Projected interaction kernel before FEFF's row folding, FEFF `xkmatp`.
    pub projected_kernel: ArrayView2<'a, Complex>,
}

/// FEFF `TDLDA/getchi0.f90` projected interaction kernel after row folding.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaProjectedKernel {
    /// Folded projected kernel, FEFF `xkmatp(1:matsize,1:matsize)`.
    pub projected_kernel: Array2<Complex>,
}

/// Inputs for FEFF `TDLDA/getchi0.f90` direct core-hole potential kernel terms.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaDirectKernelInput<'a> {
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Number of `lin + 1` projector orbitals, FEFF `nfo`.
    pub plus_basis_count: usize,
    /// Number of `lin - 1` projector orbitals, FEFF `npo`.
    pub minus_basis_count: usize,
    /// Initial-state orbital angular momentum, FEFF `lin`.
    pub initial_l: i32,
    /// Active radial integration prefix, FEFF `jint`.
    pub active_len: usize,
    /// Current energy row, FEFF `em`, in Hartree.
    pub energy_hartree: Real,
    /// Shifted Fermi level, FEFF `edge`, in Hartree.
    pub edge_energy: Real,
    /// FEFF PMBSE/TDLDA separation function, `sfun`.
    pub separation_function: Real,
    /// Per-channel reference shifts, FEFF `refsh(1:matsize)`.
    pub reference_shifts: ArrayView1<'a, Real>,
    /// Real momentum-squared rows, FEFF `p2m(1:matsize)`.
    pub momentum_squared: ArrayView1<'a, Real>,
    /// Radial grid, FEFF `ri(1:jint)`.
    pub radii: ArrayView1<'a, Real>,
    /// Core-hole potential, FEFF `vch(1:jint)`.
    pub core_hole_potential: ArrayView1<'a, Real>,
    /// Localized large components, FEFF `pf(1:jint,1:matsize)`.
    pub localized_large: ArrayView2<'a, Real>,
    /// Localized small components, FEFF `qf(1:jint,1:matsize)`.
    pub localized_small: ArrayView2<'a, Real>,
    /// Full large components, FEFF `ptot(1:jint,1:matsize)`.
    pub full_large: ArrayView2<'a, Real>,
    /// Full small components, FEFF `qtot(1:jint,1:matsize)`.
    pub full_small: ArrayView2<'a, Real>,
}

/// FEFF `TDLDA/getchi0.f90` direct core-hole potential kernel contribution.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaDirectKernel {
    /// Direct contribution to the interaction kernel, FEFF `xkmat`.
    pub kernel: Array2<Complex>,
    /// Direct contribution to the projected kernel before final row folding, FEFF `xkmatp`.
    pub projected_kernel: Array2<Complex>,
}

/// Inputs for FEFF `TDLDA/yzktd.f90` Coulomb field generation.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaCoulombFieldsInput<'a> {
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Active radial capacity, FEFF `idim`.
    pub active_len: usize,
    /// Global active source row count, FEFF `np`.
    pub source_len: usize,
    /// Active origin coefficient count, FEFF `ndor`.
    pub coefficient_count: usize,
    /// Loucks logarithmic grid step, FEFF `hx`.
    pub step: Real,
    /// Multipole order passed to `yzktd`, FEFF `nu`.
    pub multipole: usize,
    /// Radial grid, FEFF `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Per-row bound/core large components, FEFF `cg(1:idim,ncore(row))`.
    pub core_large: ArrayView2<'a, Real>,
    /// Per-row bound/core small components, FEFF `cp(1:idim,ncore(row))`.
    pub core_small: ArrayView2<'a, Real>,
    /// Per-row bound/core large origin coefficients, FEFF `bg(:,ncore(row))`.
    pub core_large_coefficients: ArrayView2<'a, Real>,
    /// Per-row bound/core small origin coefficients, FEFF `bp(:,ncore(row))`.
    pub core_small_coefficients: ArrayView2<'a, Real>,
    /// Per-row bound/core origin powers, FEFF `fl(ncore(row))`.
    pub core_powers: ArrayView1<'a, Real>,
    /// Per-row bound/core maximum tabulated row, FEFF `nmax(ncore(row))`.
    pub core_lengths: ArrayView1<'a, usize>,
    /// Target large components, FEFF `ps`/`pf(1:idim,row)`.
    pub target_large: ArrayView2<'a, Complex>,
    /// Target small components, FEFF `qs`/`qf(1:idim,row)`.
    pub target_small: ArrayView2<'a, Complex>,
    /// Target large origin coefficients, FEFF `apsm(:,row)`.
    pub target_large_coefficients: ArrayView2<'a, Complex>,
    /// Target small origin coefficients, FEFF `aqsm(:,row)`.
    pub target_small_coefficients: ArrayView2<'a, Complex>,
    /// Per-row target origin powers, FEFF `flps`.
    pub target_powers: ArrayView1<'a, Real>,
}

/// FEFF `TDLDA/yzktd.f90` Coulomb fields for TDLDA `getchi0`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaCoulombFields {
    /// Per-row `ykgr(1:active_len,row)` fields returned by `yzktd`.
    pub fields: Array2<Complex>,
    /// Meaningful transform length for each row, equivalent to clamped `np + 1`.
    pub computed_lengths: Array1<usize>,
    /// FEFF output origin constant `ap(1)` for each row.
    pub origin_constants: Array1<Complex>,
}

/// Inputs for FEFF `TDLDA/getchi0.f90` PMBSE nonlocal exchange radial integrals.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaNonlocalExchangeInput<'a> {
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Active radial integration prefix, FEFF `ilast`.
    pub active_len: usize,
    /// Global active source row count, FEFF `np`.
    pub source_len: usize,
    /// Active origin coefficient count, FEFF `ndor`.
    pub coefficient_count: usize,
    /// Loucks logarithmic grid step, FEFF `hx`.
    pub step: Real,
    /// Multipole order passed to `yzktd`, FEFF `nu = 2` for this branch.
    pub multipole: usize,
    /// FEFF direct/PMBSE scale `sfx = 1 - sfun`.
    pub direct_scale: Real,
    /// Rows satisfying FEFF's `p2m > 0` propagation check.
    pub positive_momentum_rows: ArrayView1<'a, bool>,
    /// Initial-state relativistic kappa, FEFF `kinitm(1:matsize)`.
    pub initial_kappas: ArrayView1<'a, i32>,
    /// Radial grid, FEFF `dr`/`ri`.
    pub radii: ArrayView1<'a, Real>,
    /// Per-row bound/core large components, FEFF `cg(1:idim,ncore(row))`.
    pub core_large: ArrayView2<'a, Real>,
    /// Per-row bound/core small components, FEFF `cp(1:idim,ncore(row))`.
    pub core_small: ArrayView2<'a, Real>,
    /// Per-row bound/core large origin coefficients, FEFF `bg(:,ncore(row))`.
    pub core_large_coefficients: ArrayView2<'a, Real>,
    /// Per-row bound/core small origin coefficients, FEFF `bp(:,ncore(row))`.
    pub core_small_coefficients: ArrayView2<'a, Real>,
    /// Per-row bound/core origin powers, FEFF `fl(ncore(row))`.
    pub core_powers: ArrayView1<'a, Real>,
    /// Per-row bound/core maximum tabulated row, FEFF `nmax(ncore(row))`.
    pub core_lengths: ArrayView1<'a, usize>,
    /// FEFF `pf(1:ilast,1:matsize)` localized large components.
    pub localized_large: ArrayView2<'a, Complex>,
    /// FEFF `qf(1:ilast,1:matsize)` localized small components.
    pub localized_small: ArrayView2<'a, Complex>,
    /// FEFF `ptot(1:ilast,1:matsize)` full large components.
    pub full_large: ArrayView2<'a, Complex>,
    /// FEFF `qtot(1:ilast,1:matsize)` full small components.
    pub full_small: ArrayView2<'a, Complex>,
}

/// Inputs for FEFF `TDLDA/getwf.f90` projector orthogonalization/normalization.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaProjectorOrthogonalizationInput<'a> {
    /// Active radial integration prefix, FEFF `ilast`.
    pub active_len: usize,
    /// Loucks logarithmic grid step, FEFF `dx`.
    pub log_step: Real,
    /// Norman-sphere radius used by FEFF `somm2`, FEFF `rint`.
    pub norman_radius: Real,
    /// Final-state orbital angular momentum, FEFF `lfin`.
    pub final_l: usize,
    /// Radial grid, FEFF `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// Candidate projector large component, FEFF `pat`.
    pub candidate_large: ArrayView1<'a, Real>,
    /// Candidate projector small component, FEFF `qat`.
    pub candidate_small: ArrayView1<'a, Real>,
    /// Previously stored projector large components, FEFF `dgcnp(:,ifp)`.
    pub previous_large: ArrayView2<'a, Real>,
    /// Previously stored projector small components, FEFF `dpcnp(:,ifp)`.
    pub previous_small: ArrayView2<'a, Real>,
}

/// FEFF `TDLDA/getwf.f90` normalized projector after Gram-Schmidt cleanup.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaProjectorOrthogonalization {
    /// Orthogonalized and normalized large component, FEFF final `pat`.
    pub large: Array1<Real>,
    /// Orthogonalized and normalized small component, FEFF final `qat`.
    pub small: Array1<Real>,
    /// FEFF `xinorm` overlap subtractions applied to each previous projector.
    pub overlaps: Array1<Real>,
    /// Pre-square-root normalization integral inside the Norman sphere.
    pub norm_integral: Real,
    /// Square root of `norm_integral`, the divisor applied to both components.
    pub norm_sqrt: Real,
}

/// Inputs for FEFF `TDLDA/getchi0.f90` Coulomb/xc radial kernel integrals.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaRadialKernelInput<'a> {
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Active radial integration prefix, FEFF `ilast`.
    pub active_len: usize,
    /// Rows satisfying FEFF's `p2m > 0` propagation check.
    pub positive_momentum_rows: ArrayView1<'a, bool>,
    /// Initial-state relativistic kappa, FEFF `kinitm(1:matsize)`.
    pub initial_kappas: ArrayView1<'a, i32>,
    /// FEFF exchange-correlation selector, `ifxc`.
    pub exchange_correlation_selector: i32,
    /// FEFF direct/PMBSE scale `sfx = 1 - sfun`.
    pub direct_scale: Real,
    /// Radial grid, FEFF `ri(1:ilast)`.
    pub radii: ArrayView1<'a, Real>,
    /// FEFF local same-edge exchange-correlation kernel, `fxc0(1:ilast)`.
    pub exchange_correlation_same_edge: ArrayView1<'a, Real>,
    /// FEFF local exchange-correlation kernel real part, `fxc(1:ilast)`.
    pub exchange_correlation_real: ArrayView1<'a, Real>,
    /// FEFF local exchange-correlation kernel imaginary part, `fxcim(1:ilast)`.
    pub exchange_correlation_imaginary: ArrayView1<'a, Real>,
    /// FEFF `pc(1:ilast,1:matsize)` response large components.
    pub response_large: ArrayView2<'a, Complex>,
    /// FEFF `qc(1:ilast,1:matsize)` response small components.
    pub response_small: ArrayView2<'a, Complex>,
    /// FEFF `pf(1:ilast,1:matsize)` localized large components.
    pub localized_large: ArrayView2<'a, Complex>,
    /// FEFF `qf(1:ilast,1:matsize)` localized small components.
    pub localized_small: ArrayView2<'a, Complex>,
    /// FEFF `ptot(1:ilast,1:matsize)` full large components.
    pub full_large: ArrayView2<'a, Complex>,
    /// FEFF `qtot(1:ilast,1:matsize)` full small components.
    pub full_small: ArrayView2<'a, Complex>,
    /// FEFF `ykgr(1:ilast,1:matsize)` Coulomb fields from `yzktd`.
    pub coulomb_fields: ArrayView2<'a, Complex>,
}

/// FEFF `TDLDA/getchi0.f90` radial Coulomb/xc integrals before angular weights.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaRadialKernel {
    /// Localized radial Coulomb/xc integrals, FEFF `rabcd`.
    pub radial_integrals: Array2<Complex>,
    /// Projected radial Coulomb/xc integrals, FEFF `rabcdp`.
    pub projected_radial_integrals: Array2<Complex>,
}

/// Inputs for FEFF `TDLDA/getchi0.f90` Coulomb/xc angular kernel accumulation.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaAngularKernelInput<'a> {
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Initial-state doubled total angular momentum, FEFF `jinit(1:matsize)`.
    pub initial_j2: ArrayView1<'a, i32>,
    /// Initial-state doubled magnetic quantum number, FEFF `minit(1:matsize)`.
    pub initial_m2: ArrayView1<'a, i32>,
    /// Initial-state relativistic kappa, FEFF `kinitm(1:matsize)`.
    pub initial_kappas: ArrayView1<'a, i32>,
    /// Final-state doubled total angular momentum, FEFF `jfin(1:matsize)`.
    pub final_j2: ArrayView1<'a, i32>,
    /// Final-state doubled magnetic quantum number, FEFF `mfin(1:matsize)`.
    pub final_m2: ArrayView1<'a, i32>,
    /// Rows satisfying FEFF's `p2m > 0` propagation check.
    pub positive_momentum_rows: ArrayView1<'a, bool>,
    /// Localized radial Coulomb/xc integrals, FEFF `rabcd`.
    pub radial_integrals: ArrayView2<'a, Complex>,
    /// Projected radial Coulomb/xc integrals, FEFF `rabcdp`.
    pub projected_radial_integrals: ArrayView2<'a, Complex>,
    /// Optional PMBSE nonlocal-exchange radial integrals, FEFF `rabcd` for `nu = 2`.
    pub nonlocal_radial_integrals: Option<ArrayView2<'a, Complex>>,
    /// Optional projected PMBSE nonlocal-exchange radial integrals, FEFF `rabcdp` for `nu = 2`.
    pub nonlocal_projected_radial_integrals: Option<ArrayView2<'a, Complex>>,
}

/// FEFF `TDLDA/getchi0.f90` Coulomb/xc angular kernel contribution.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaAngularKernel {
    /// Angular-weighted contribution to the interaction kernel, FEFF `xkmat`.
    pub kernel: Array2<Complex>,
    /// Angular-weighted contribution to projected kernel before row folding, FEFF `xkmatp`.
    pub projected_kernel: Array2<Complex>,
    /// Main `nu = 1` Wigner prefactors applied to `radial_integrals`.
    pub prefactors: Array2<Real>,
    /// PMBSE nonlocal `nu = 2` Wigner prefactors subtracted from the kernel.
    pub nonlocal_prefactors: Array2<Real>,
}

/// Inputs for FEFF `TDLDA/kkchi.f90` Kramers-Kronig response transform.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaKramersKronigInput<'a> {
    /// Number of active energy rows, FEFF `ne1`.
    pub energy_count: usize,
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Energy grid, FEFF `emr(1:ne1)`, in Hartree.
    pub energy_hartree: ArrayView1<'a, Real>,
    /// Chemical-potential position in the absorption spectrum, FEFF `emu`.
    pub chemical_potential: Real,
    /// Shifted Fermi level, FEFF `edge`.
    pub edge_energy: Real,
    /// Per-channel reference shifts, FEFF `refsh(1:matsize)`.
    pub reference_shifts: ArrayView1<'a, Real>,
    /// Imaginary response matrix, FEFF `chi0im(1:ne1,1:matsize,1:matsize)`.
    pub imaginary_response: ArrayView3<'a, Real>,
}

/// FEFF `TDLDA/kkchi.f90` real response matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaKramersKronigResponse {
    /// Real response matrix, FEFF `chi0r(1:ne1,1:matsize,1:matsize)`.
    pub real_response: Array3<Real>,
}

/// Inputs for FEFF `TDLDA/xsectd.f90` response broadening and complex assembly.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaResponseConditioningInput<'a> {
    /// Number of active energy rows, FEFF `ne1`.
    pub energy_count: usize,
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Energy grid, FEFF `emr(1:ne1)`, in Hartree.
    pub energy_hartree: ArrayView1<'a, Real>,
    /// Chemical-potential position in the absorption spectrum, FEFF `emu`.
    pub chemical_potential: Real,
    /// Shifted Fermi level, FEFF `edge`.
    pub edge_energy: Real,
    /// Per-channel reference shifts, FEFF `refsh(1:matsize)`.
    pub reference_shifts: ArrayView1<'a, Real>,
    /// Per-row Lorentzian broadening values, FEFF `gammab(1:matsize)`.
    pub row_broadenings: ArrayView1<'a, Real>,
    /// Raw imaginary response matrix, FEFF `chi0im(1:ne1,1:matsize,1:matsize)`.
    pub imaginary_response: ArrayView3<'a, Real>,
}

/// FEFF `TDLDA/xsectd.f90` response after broadening and KK transform.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaConditionedResponse {
    /// Broadened imaginary response, FEFF `chi0im` after `conv`.
    pub broadened_imaginary_response: Array3<Real>,
    /// Real response from `kkchi`, FEFF `chi0r`.
    pub real_response: Array3<Real>,
    /// Complex response, FEFF `chi0 = chi0r + i*chi0im`.
    pub response: Array3<Complex>,
}

/// One `xmu.dat` channel table consumed by FEFF `TDLDA/ridxmu.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaXmuChannelInput<'a> {
    /// Number of active `xmu.dat` rows.
    pub point_count: usize,
    /// FEFF `xmu.dat` photon-energy column, in eV.
    pub photon_energy_ev: ArrayView1<'a, Real>,
    /// FEFF `xmu.dat` edge-relative energy column, in eV.
    pub relative_energy_ev: ArrayView1<'a, Real>,
    /// FEFF `xmu.dat` photoelectron wave-number column.
    pub wave_number: ArrayView1<'a, Real>,
    /// FEFF `xmu.dat` normalized background column, `mu0`.
    pub background: ArrayView1<'a, Real>,
    /// FEFF `xmu.dat` fine-structure column, `chi`.
    pub fine_structure: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `TDLDA/ridxmu.f90` channel multiplier interpolation.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaChannelMultipliersInput<'a> {
    /// Initial-state relativistic kappa, FEFF `kinit`.
    pub initial_kappa: i32,
    /// Maximum retained output rows, FEFF `nex`.
    pub energy_capacity: usize,
    /// Dominant `l -> l + 1` channel, FEFF `Oddp1/xmu.dat`.
    pub dominant_plus: XsphTdldaXmuChannelInput<'a>,
    /// Split-edge `l -> l + 1` channel, FEFF `Evenp1/xmu.dat`.
    pub split_plus: Option<XsphTdldaXmuChannelInput<'a>>,
    /// Dominant `l -> l - 1` channel, FEFF `Oddm1/xmu.dat`.
    pub dominant_minus: Option<XsphTdldaXmuChannelInput<'a>>,
    /// Split-edge `l -> l - 1` channel, FEFF `Evenm1/xmu.dat`.
    pub split_minus: Option<XsphTdldaXmuChannelInput<'a>>,
}

/// FEFF `TDLDA/ridxmu.f90` energy grid and channel multipliers.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaChannelMultipliers {
    /// Output grid, FEFF `ee(1:ne)`, in Hartree.
    pub energy_hartree: Array1<Real>,
    /// Spin-orbit split between the two `l -> l + 1` edges, FEFF `deltaso`.
    pub spin_orbit_split: Real,
    /// Channel multipliers ordered as FEFF `chil3`, `chil2`, `chil5`, `chil4`.
    pub channel_multipliers: Array2<Real>,
}

/// Inputs for FEFF `TDLDA/xsectd.f90` PMBSE channel weighting of raw `getchi0` rows.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaWeightedResponseInput<'a> {
    /// Number of active energy rows, FEFF `ne1` after `ridxmu`.
    pub energy_count: usize,
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// FEFF `kinitm(1:matsize)`.
    pub initial_kappas: ArrayView1<'a, i32>,
    /// FEFF `kfinm(1:matsize)`.
    pub final_kappas: ArrayView1<'a, i32>,
    /// Raw `getchi0` imaginary response, FEFF local `chi(1:matsize,1:matsize)`.
    pub raw_imaginary_response: ArrayView3<'a, Real>,
    /// PMBSE channel multipliers ordered as FEFF `chil3`, `chil2`, `chil5`, `chil4`.
    pub channel_multipliers: ArrayView2<'a, Real>,
}

/// FEFF `TDLDA/xsectd.f90` raw response after PMBSE channel multipliers.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaWeightedResponse {
    /// Weighted imaginary response, FEFF `chi0im(1:ne1,1:matsize,1:matsize)`.
    pub imaginary_response: Array3<Real>,
    /// Per-row channel indices in `chil3`, `chil2`, `chil5`, `chil4` order.
    pub row_channels: Array1<usize>,
}

/// Inputs for FEFF `TDLDA/xsectd.f90` channel-spectrum accumulation.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaChannelSpectraInput<'a> {
    /// Number of active energy rows, FEFF `nelast`.
    pub energy_count: usize,
    /// Active `getmat` matrix size, FEFF `matsize`.
    pub matrix_size: usize,
    /// Number of leading rows assigned to `l3`/`l2`; FEFF uses `im <= 15`.
    pub primary_channel_count: usize,
    /// Active spin-orbit channel count, FEFF `nch`; supported FEFF cases are 1, 2, and 4.
    pub channel_count: usize,
    /// Photon energies, FEFF `omega(1:nelast)`, in Hartree.
    pub photon_energy: ArrayView1<'a, Real>,
    /// Real part of FEFF `ckl3(1:nelast)`.
    pub plus_wave_number: ArrayView1<'a, Real>,
    /// Real part of FEFF `ckl2(1:nelast)`.
    pub minus_wave_number: ArrayView1<'a, Real>,
    /// Initial-state kappa sign map, FEFF `kinitm(1:matsize)`.
    pub initial_kappas: ArrayView1<'a, i32>,
    /// Localized dipole matrix elements, FEFF `dipmat(1:nelast,1:matsize)`.
    pub dipole_matrix: ArrayView2<'a, Real>,
    /// Response matrix, FEFF `chi0(1:nelast,1:matsize,1:matsize)`.
    pub response: ArrayView3<'a, Complex>,
    /// Projected interaction kernel, FEFF `xkmatp(1:matsize,1:matsize,1:nelast)`.
    pub projected_kernel: ArrayView3<'a, Complex>,
    /// Screened matrix elements, FEFF `dipscf(1:nelast,1:matsize)`.
    pub screened_dipoles: ArrayView2<'a, Complex>,
}

/// FEFF `TDLDA/xsectd.f90` single-particle and TDLDA-screened channel spectra.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaChannelSpectra {
    /// Channel spectra from `dipmat**2`, ordered as FEFF `l3`, `l2`, `l5`, `l4`.
    pub single_particle_channels: Array2<Real>,
    /// Channel spectra from `abs(dipmat + K*chi0*dipscf)**2`, ordered as FEFF `l3`, `l2`, `l5`, `l4`.
    pub screened_channels: Array2<Real>,
    /// FEFF `prefacl3` per active energy row.
    pub plus_prefactors: Array1<Real>,
    /// FEFF `prefacl2` per active energy row.
    pub minus_prefactors: Array1<Real>,
}

/// Inputs for FEFF `TDLDA/xsectd.f90` channel broadening before `xsedge.dat`.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaChannelBroadeningInput<'a> {
    /// Number of active energy rows, FEFF `ne1`.
    pub energy_count: usize,
    /// Active spin-orbit channel count, FEFF `nch`; supported FEFF cases are 1, 2, and 4.
    pub channel_count: usize,
    /// Energy grid, FEFF `emr(1:ne1)`, in Hartree.
    pub energy_hartree: ArrayView1<'a, Real>,
    /// Plus-channel onset, FEFF `edge`, in Hartree.
    pub edge_energy: Real,
    /// Spin-orbit split, FEFF `deltaso`, in Hartree.
    pub spin_orbit_split: Real,
    /// Plus-channel broadening, FEFF `gaml3`, in Hartree.
    pub plus_broadening: Real,
    /// Minus-channel broadening, FEFF `gaml2`, in Hartree.
    pub minus_broadening: Real,
    /// Single-particle channel spectra ordered as FEFF `l3`, `l2`, `l5`, `l4`.
    pub single_particle_channels: ArrayView2<'a, Real>,
    /// TDLDA-screened channel spectra ordered as FEFF `l3`, `l2`, `l5`, `l4`.
    pub screened_channels: ArrayView2<'a, Real>,
}

/// FEFF `TDLDA/xsectd.f90` broadened channel spectra before `xsedge.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaBroadenedChannelSpectra {
    /// Broadened single-particle channel spectra ordered as FEFF `l3`, `l2`, `l5`, `l4`.
    pub single_particle_channels: Array2<Real>,
    /// Broadened TDLDA-screened channel spectra ordered as FEFF `l3`, `l2`, `l5`, `l4`.
    pub screened_channels: Array2<Real>,
}

/// Inputs for FEFF `TDLDA/xsectd.f90` `xsedge.dat` final row assembly.
#[derive(Debug, Clone, Copy)]
pub struct XsphTdldaXsedgeRowsInput<'a> {
    /// Number of active energy rows, FEFF `nelast`.
    pub energy_count: usize,
    /// Active spin-orbit channel count, FEFF `nch`; supported FEFF output cases are 1, 2, and 4.
    pub channel_count: usize,
    /// Output energies in Hartree, FEFF writes `(emr(ie) + emu) * hart`.
    pub energy_hartree: ArrayView1<'a, Real>,
    /// Single-particle channel spectra ordered as FEFF `l3`, `l2`, `l5`, `l4`.
    pub single_particle_channels: ArrayView2<'a, Real>,
    /// TDLDA-screened channel spectra ordered as FEFF `l3`, `l2`, `l5`, `l4`.
    pub screened_channels: ArrayView2<'a, Real>,
    /// Channel multipliers ordered as FEFF `chil3`, `chil2`, `chil5`, `chil4`.
    pub channel_multipliers: ArrayView2<'a, Real>,
}

/// FEFF `TDLDA/xsectd.f90` `xsedge.dat` row data.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphTdldaXsedgeRows {
    /// Output energy column, converted to eV.
    pub energy_ev: Array1<Real>,
    /// Sum of active single-particle channel spectra.
    pub total_single_particle: Array1<Real>,
    /// Sum of active TDLDA-screened channel spectra.
    pub total_screened: Array1<Real>,
    /// Sum of the `l3` and optional `l5` channels.
    pub plus_branch_single_particle: Array1<Real>,
    /// Sum of the `l2` and optional `l4` channels.
    pub minus_branch_single_particle: Array1<Real>,
    /// Screened sum of the `l3` and optional `l5` channels.
    pub plus_branch_screened: Array1<Real>,
    /// Screened sum of the `l2` and optional `l4` channels.
    pub minus_branch_screened: Array1<Real>,
}

/// Inputs for FEFF `XSPH/specupdlg.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphLgSpectrumUpdateInput<'a> {
    /// One-based shared calculation index, FEFF `icalc`.
    pub calculation_index: i32,
    /// Spin component, FEFF `isp` in the range `0..=1`.
    pub spin_index: usize,
    /// Per-final-state calculation map, FEFF `indmap(1:indmax)`.
    pub index_map: ArrayView1<'a, i32>,
    /// Angular-decomposition output index, FEFF `lind(1:indmax)`.
    pub orbital_l: ArrayView1<'a, i32>,
    /// Radial-integral/Legendre index, FEFF `ljind(1:indmax)`.
    pub final_lj: ArrayView1<'a, i32>,
    /// Doubled initial-state angular momentum, FEFF `jinit`.
    pub initial_j2: i32,
    /// Transition weights with compact magnetic index, FEFF `hbmat(0:1,ii,mjinit)`.
    ///
    /// Shape must be at least `(2, active_len, initial_j2 + 1)`, where the
    /// compact magnetic column is `(mjinit + initial_j2) / 2`.
    pub transition_weights: ArrayView3<'a, Real>,
    /// Radial integrals `xirflj(0:ljmax)`.
    pub radial_integrals: ArrayView1<'a, Complex>,
    /// MDFF q-vector weights, FEFF `qw(1:nq)`.
    pub q_weights: ArrayView1<'a, Complex>,
    /// Cosines between q-vector pairs, FEFF `cosmdff(1:nq,1:nq)`.
    pub q_cosines: ArrayView2<'a, Real>,
    /// Whether FEFF `mixdff` is enabled.
    pub mix_dff: bool,
    /// FEFF `imdff` selector when `mix_dff` is enabled.
    pub mdff_mode: i32,
    /// Largest active `lj`/`lg` index.
    pub ljmax: usize,
    /// Number of active final states, FEFF `indmax`.
    pub active_len: usize,
    /// FEFF regular/irregular update mode.
    pub mode: XsphSpectrumUpdateMode,
}

/// Inputs for FEFF `XSPH/specupd.f90` and `XSPH/specupdatom.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphLjSpectrumUpdateInput<'a> {
    /// One-based shared calculation index, FEFF `icalc`.
    pub calculation_index: i32,
    /// Spin component, FEFF `isp` in the range `0..=1`.
    pub spin_index: usize,
    /// Per-final-state calculation map, FEFF `indmap(1:indmax)`.
    pub index_map: ArrayView1<'a, i32>,
    /// Radial-integral/Legendre index, FEFF `ljind(1:indmax)`.
    pub final_lj: ArrayView1<'a, i32>,
    /// Doubled initial-state angular momentum, FEFF `jinit`.
    pub initial_j2: i32,
    /// Transition weights with compact magnetic index, FEFF `hbmat(0:1,ii,mjinit)`.
    ///
    /// Shape must be at least `(2, active_len, initial_j2 + 1)`, where the
    /// compact magnetic column is `(mjinit + initial_j2) / 2`.
    pub transition_weights: ArrayView3<'a, Real>,
    /// Radial integrals `xirflj(0:ljmax)`.
    pub radial_integrals: ArrayView1<'a, Complex>,
    /// MDFF q-vector weights, FEFF `qw(1:nq)`.
    pub q_weights: ArrayView1<'a, Complex>,
    /// Cosines between q-vector pairs, FEFF `cosmdff(1:nq,1:nq)`.
    pub q_cosines: ArrayView2<'a, Real>,
    /// Whether FEFF `mixdff` is enabled.
    pub mix_dff: bool,
    /// FEFF `imdff` selector when `mixdff` is enabled.
    pub mdff_mode: i32,
    /// Largest active `lj` index.
    pub ljmax: usize,
    /// Number of active final states, FEFF `indmax`.
    pub active_len: usize,
    /// FEFF regular/irregular update mode.
    pub mode: XsphSpectrumUpdateMode,
}

/// Inputs for FEFF `XSPH/axafs.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphAxafsInput<'a> {
    /// Complex energy grid `em(1:ne)` in Hartree.
    pub energies: ArrayView1<'a, Complex>,
    /// Complex atomic cross section `xsec(1:ne)`.
    pub cross_section: ArrayView1<'a, Complex>,
    /// FEFF `emu`, the Fermi/edge reference energy in Hartree.
    pub fermi_energy: Real,
    /// Number of horizontal grid points, FEFF `ne1`.
    pub horizontal_count: usize,
    /// Zero-wave grid point as a Rust zero-based index, FEFF `ik0 - 1`.
    pub zero_wave_index: usize,
}

/// AXAFS table generated by FEFF `XSPH/axafs.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphAxafs {
    /// Output rows with columns `e`, `e(wrt edge)`, `k`, `mu_at`, `mu0_at`,
    /// and `chi_at`, matching FEFF `axafs.dat`.
    pub rows: Array2<Real>,
    /// Quadratic background coefficients `(aa, bb, cc)` in Hartree units.
    pub coefficients: [Real; 3],
    /// FEFF normalization at the first output energy plus 100 eV.
    pub normalization: Real,
}

/// Inputs for FEFF `XSPH/getholeorb0.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphHoleOrbitalInput<'a> {
    /// Large radial spinor component for the compacted hole orbital on the
    /// original logarithmic grid, FEFF `dgc(1:251, iholep, 0)`.
    pub large_component: ArrayView1<'a, Real>,
    /// Small radial spinor component for the compacted hole orbital on the
    /// original logarithmic grid, FEFF `dpc(1:251, iholep, 0)`.
    pub small_component: ArrayView1<'a, Real>,
    /// Original logarithmic grid spacing, FEFF `dx`.
    pub original_step: Real,
    /// New logarithmic grid spacing, FEFF `dxnew`.
    pub new_step: Real,
    /// Number of output points to interpolate, FEFF `jnew`.
    pub output_count: usize,
    /// Full output capacity, FEFF `nrptx`. Values past `output_count` are zero.
    pub output_capacity: usize,
}

/// Initial-state hole orbital interpolated onto the XSPH radial grid.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphHoleOrbital {
    /// Large radial spinor component on the new grid, FEFF `dgcx0`.
    pub large_component: Array1<Real>,
    /// Small radial spinor component on the new grid, FEFF `dpcx0`.
    pub small_component: Array1<Real>,
    /// Number of interpolated points before the zero-filled tail.
    pub active_count: usize,
    /// Source prefix length used for FEFF cubic interpolation, FEFF `jmax`.
    pub source_count: usize,
}

/// FEFF phase-energy grid after sorting and near-duplicate removal.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphSortedEnergyGrid {
    /// Sorted real energy points with zero imaginary parts, FEFF `em`.
    pub energies: Array1<Complex>,
    /// Rust zero-based index of FEFF `ik0`, the point closest to zero.
    pub zero_index: usize,
}

/// FEFF84 horizontal XANES/DANES phase mesh and its zero-energy index.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXanesEnergyGrid84 {
    /// Horizontal FEFF84 energy mesh, FEFF `em(1:ne)`.
    pub energies: Array1<Complex>,
    /// Rust zero-based index of FEFF `ik0`, the Fermi-level point.
    pub zero_index: usize,
}

/// FEFF84 FPRIME phase mesh with its regular and KK-extension counts.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphFprimeEnergyGrid84 {
    /// FEFF84 FPRIME energy mesh, FEFF `em(1:ne)`.
    pub energies: Array1<Complex>,
    /// Number of points in the regular FPRIME grid, FEFF `ne1`.
    pub regular_count: usize,
    /// Number of points in the KK-transform extension, FEFF `ne3`.
    pub kk_count: usize,
}

/// Inputs for the default FEFF84 branch of `XSPH/phmesh2.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhaseEnergyMesh84Input {
    /// FEFF `ispec` selector: negative no-FMS EXAFS/DANES, `0` EXAFS,
    /// `1` XANES, `2` XES, `3` DANES, or `4` FPRIME.
    pub spectroscopy: i32,
    /// FEFF `edge`, the `xmu - vr0` offset in Hartree.
    pub edge: Real,
    /// FEFF `emu`, the Fermi/reference energy in Hartree.
    pub reference_energy: Real,
    /// FEFF `vi0`, the constant imaginary potential in Hartree.
    pub constant_imaginary: Real,
    /// FEFF `gamach`, the core-hole broadening in Hartree.
    pub core_hole_broadening: Real,
    /// FEFF `ecv`, retained for signature compatibility with `phmesh2`.
    pub core_valence_separation: Real,
    /// FEFF `xkmax`; for XES/FPRIME this is the lower energy bound.
    pub max_wave_number: Real,
    /// FEFF `xkstep`; for XES/FPRIME this is the upper energy bound.
    pub wave_number_step: Real,
    /// FEFF `vixan`; positive values override the near-edge step.
    pub xanes_energy_step: Real,
    /// Output capacity, FEFF `nex`.
    pub capacity: usize,
}

/// FEFF `XSPH/phmesh2.f90` NRIXS/RHORRP contour from `mk_rhorrp_grid`.
#[derive(Debug, Clone, Copy)]
pub struct XsphRhorrpPhaseEnergyMeshInput {
    /// FEFF `edge`, the `xmu - vr0` offset in Hartree.
    pub edge: Real,
    /// FEFF `ecv`, the core-valence separation in Hartree.
    pub core_valence_separation: Real,
    /// FEFF `potential_inp::scf_temperature`, in eV.
    pub scf_temperature: Real,
    /// Output capacity, FEFF `nex`.
    pub capacity: usize,
}

/// Inputs for the JAS/NRIXS constant-step branch of `XSPH/phmeshjas.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphJasPhaseEnergyMeshInput {
    /// FEFF `ispec` selector. FEFF falls back to `phmesh` for `2` and `>= 3`;
    /// this safe wrapper reports those selectors as unsupported.
    pub spectroscopy: i32,
    /// FEFF `edge`, the `xmu - vr0` offset in Hartree.
    pub edge: Real,
    /// FEFF `vi0`, the constant imaginary potential in Hartree.
    pub constant_imaginary: Real,
    /// FEFF `gamach`, the core-hole broadening in Hartree.
    pub core_hole_broadening: Real,
    /// FEFF `ecv`, retained for signature compatibility with `phmeshjas`.
    pub core_valence_separation: Real,
    /// FEFF `xkmax`, in internal inverse-Bohr wave-number units.
    pub max_wave_number: Real,
    /// FEFF `xkstep`, in internal inverse-Bohr wave-number units.
    pub wave_number_step: Real,
    /// FEFF `vixan`; positive values above `1e-4` override the near-edge step.
    pub xanes_energy_step: Real,
    /// FEFF module `nex`. `phmeshjas` treats this as a horizontal grid budget
    /// for XANES-like spectra and then appends the vertical contour.
    pub horizontal_capacity: usize,
}

/// FEFF `grid.inp` regular-grid kind for the XSPH `phmesh2` user-grid branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphPhaseUserGridKind {
    /// `e_grid`: regular in energy, with values in eV.
    Energy,
    /// `k_grid`: regular in wave number, with values in inverse Angstrom.
    WaveNumber,
    /// `exp_grid`: FEFF exponential energy grid, with energy values in eV.
    Exponential,
}

/// FEFF `grid.inp` minimum field for a regular XSPH phase grid record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum XsphPhaseUserGridMinimum {
    /// Explicit minimum value, in eV for energy grids and inverse Angstrom for k grids.
    Value(Real),
    /// FEFF `last` marker, resolved from the previous grid's maximum and this grid's step.
    Last,
}

/// Regular generated grid record from FEFF `grid.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphPhaseUserRegularGrid {
    /// Grid generator kind.
    pub kind: XsphPhaseUserGridKind,
    /// Minimum grid value or FEFF's `last` continuation marker.
    pub minimum: XsphPhaseUserGridMinimum,
    /// Maximum grid value, in eV for energy grids and inverse Angstrom for k grids.
    pub maximum: Real,
    /// Grid step, in eV for energy grids and inverse Angstrom for k grids.
    pub step: Real,
}

/// One FEFF `grid.inp` record for the XSPH `phmesh2` user-grid branch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum XsphPhaseUserGridRecord<'a> {
    /// Regular generated grid.
    Regular(XsphPhaseUserRegularGrid),
    /// User-specified complex energy points in eV.
    ///
    /// FEFF sorts the horizontal grid by real energy before shifting, so any
    /// supplied imaginary parts are accepted for input compatibility but are
    /// discarded by the `SortE` step.
    User(ArrayView1<'a, Complex>),
}

/// Inputs for the FEFF `XSPH/phmesh2.f90` `iGrid != 0` `grid.inp` branch.
#[derive(Debug, Clone, Copy)]
pub struct XsphPhaseUserGridInput<'a> {
    /// FEFF `ispec` selector; `abs(ispec) <= 5` follows the user-grid
    /// `phmesh2` path.
    pub spectroscopy: i32,
    /// FEFF `edge`, the `xmu - vr0` offset in Hartree.
    pub edge: Real,
    /// FEFF `vi0`, the constant imaginary potential in Hartree.
    pub constant_imaginary: Real,
    /// FEFF `gamach`, the core-hole broadening in Hartree.
    pub core_hole_broadening: Real,
    /// Parsed `grid.inp` records in file order.
    pub records: &'a [XsphPhaseUserGridRecord<'a>],
    /// Output capacity, FEFF `nex`.
    pub capacity: usize,
}

/// Inputs for the normal finite-temperature branch of `XSPH/phmesh2T.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XsphThermalPhaseEnergyMeshInput<'a> {
    /// FEFF `edge`, the `xmu - vr0` offset in Hartree.
    pub edge: Real,
    /// FEFF `vi0`, the constant imaginary potential in Hartree.
    pub constant_imaginary: Real,
    /// FEFF `gamach`, the core-hole broadening in Hartree.
    pub core_hole_broadening: Real,
    /// FEFF `ecv`, the core-valence separation in Hartree.
    pub core_valence_separation: Real,
    /// FEFF `electronic_temperature` in eV.
    pub electronic_temperature: Real,
    /// Optional parsed `grid.inp` records. `None` selects the default thermal grid.
    pub user_records: Option<&'a [XsphPhaseUserGridRecord<'a>]>,
    /// Output capacity, FEFF `nex`.
    pub capacity: usize,
}

/// Combined FEFF84 phase-energy mesh from `XSPH/phmesh2.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphPhaseEnergyMesh84 {
    /// Combined FEFF84 energy mesh, FEFF `em(1:ne)`.
    pub energies: Array1<Complex>,
    /// Number of horizontal points before the vertical contour, FEFF `ne1`.
    pub horizontal_count: usize,
    /// FEFF `ne3`: FPRIME KK-extension count, or DANES high-energy extension count.
    pub extension_count: usize,
    /// Rust zero-based index of FEFF `ik0`.
    pub zero_index: usize,
    /// Constant imaginary broadening applied to horizontal non-FPRIME meshes.
    pub xloss: Real,
}

/// NRIXS/RHORRP phase-energy mesh from FEFF `mk_rhorrp_grid`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphRhorrpPhaseEnergyMesh {
    /// Combined contour plus Matsubara poles, FEFF `em(1:ne)`.
    pub energies: Array1<Complex>,
    /// Number of vertical plus horizontal contour points, FEFF `ne1`.
    pub contour_count: usize,
    /// Number of Matsubara poles, FEFF `ne - ne1`.
    pub pole_count: usize,
    /// FEFF floor-adjusted electronic temperature in Hartree.
    pub temperature: Real,
    /// Imaginary height of the contour.
    pub upper_imaginary: Real,
}

/// Finite-temperature phase-energy mesh from `XSPH/phmesh2T.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphThermalPhaseEnergyMesh {
    /// Combined thermal contour, FEFF `em(1:ne)`.
    pub energies: Array1<Complex>,
    /// Number of points on each horizontal leg, FEFF `ne1`.
    pub horizontal_count: usize,
    /// Number of Matsubara poles enclosed by the contour.
    pub pole_count: usize,
    /// Rust zero-based index of FEFF `ik0`.
    pub zero_index: usize,
    /// Constant imaginary broadening applied to the lower horizontal leg.
    pub xloss: Real,
    /// Imaginary height of the upper horizontal leg.
    pub upper_imaginary: Real,
}

/// JAS/NRIXS phase-energy mesh from `XSPH/phmeshjas.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphJasPhaseEnergyMesh {
    /// Combined JAS phase mesh, FEFF `em(1:ne)`.
    pub energies: Array1<Complex>,
    /// Number of horizontal points before the vertical contour, FEFF `ne1`.
    pub horizontal_count: usize,
    /// Number of appended vertical-contour points.
    pub vertical_count: usize,
    /// Rust zero-based index of FEFF `ik0`.
    pub zero_index: usize,
    /// Constant imaginary broadening applied to horizontal points.
    pub xloss: Real,
}

/// FEFF84 XES phase mesh and its zero-energy index.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphXesEnergyGrid84 {
    /// Horizontal FEFF84 XES energy mesh, FEFF `em(1:ne)`.
    pub energies: Array1<Complex>,
    /// Rust zero-based index of FEFF `ik0`, the closest point to zero.
    pub zero_index: usize,
}

/// Error returned by XSPH planning helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum XsphError {
    /// FEFF `mincalc` expects at least one active final-state index.
    #[error("XSPH calculation planning requires at least one active index")]
    EmptyIndexSet,
    /// A supplied index row is shorter than the requested active prefix.
    #[error("{name} length {actual} is shorter than active length {required}")]
    LengthTooShort {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// One-based FEFF indices must stay inside the active prefix.
    #[error("{name} one-based index {index_1based} is outside 1..={active_len}")]
    InvalidOneBasedIndex {
        name: &'static str,
        index_1based: usize,
        active_len: usize,
    },
    /// Angular momentum indices used as output slots must be non-negative.
    #[error("{name} entry {index} must be non-negative, got {value}")]
    NegativeAngularMomentum {
        name: &'static str,
        index: usize,
        value: i32,
    },
    /// FEFF `ljneeded0` would stop when an `lj` index exceeds `ljmax`.
    #[error("XSPH angular momentum {angular_momentum} exceeds ljmax {ljmax}")]
    AngularMomentumOutOfRange {
        angular_momentum: usize,
        ljmax: usize,
    },
    /// Shared calculation indices are one-based in FEFF.
    #[error("XSPH calculation index must be positive, got {calculation_index}")]
    NonPositiveCalculationIndex { calculation_index: i32 },
    /// The FEFF map convention cannot represent `abs(i32::MIN)`.
    #[error("XSPH index map entry {index} cannot be negated: {value}")]
    IndexMapOverflow { index: usize, value: i32 },
    /// Requested output size overflows `usize`.
    #[error("XSPH ljmax {ljmax} cannot be represented as an output vector length")]
    AngularMomentumCapacityOverflow { ljmax: usize },
    /// XSPH scalar inputs must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteScalar { name: &'static str, value: Real },
    /// Positive scalar inputs must be finite and greater than zero.
    #[error("{name} must be finite and positive, got {value}")]
    InvalidPositiveScalar { name: &'static str, value: Real },
    /// XSPH complex inputs must have finite real and imaginary parts.
    #[error("{name} entry {index} must be finite, got ({real}, {imaginary})")]
    NonFiniteComplex {
        name: &'static str,
        index: usize,
        real: Real,
        imaginary: Real,
    },
    /// Spherical Bessel evaluation failed.
    #[error(transparent)]
    Bessel(#[from] BesselError),
    /// Wigner-symbol evaluation failed.
    #[error(transparent)]
    Angular(#[from] AngularError),
    /// FEFF radial-grid quadrature failed.
    #[error(transparent)]
    Quadrature(#[from] QuadratureError),
    /// FEFF phase-amplitude evaluation failed.
    #[error(transparent)]
    Phase(#[from] PhaseError),
    /// FOVRG radial Dirac solve failed while building XSPH phase data.
    #[error(transparent)]
    Fovrg(#[from] FovrgError),
    /// FEFF radial-grid resampling failed while preparing XSPH phase data.
    #[error(transparent)]
    Grid(#[from] GridError),
    /// FEFF linear solve failed while building XSPH screened fields.
    #[error(transparent)]
    Linalg(#[from] LinalgError),
    /// FEFF analytic convolution failed while broadening XSPH spectra.
    #[error(transparent)]
    Convolution(#[from] ConvolutionError),
    /// FEFF TDLDA/PMBSE channel generation requires a listed channel file.
    #[error("XSPH TDLDA/PMBSE channel input {name} is missing")]
    MissingTdldaChannel { name: &'static str },
    /// FEFF `TDLDA/ridxmu.f90` could not identify an edge row from xmu.dat.
    #[error("XSPH TDLDA/PMBSE channel input {name} has no near-zero wave-number edge row")]
    MissingTdldaEdge { name: &'static str },
    /// Relativistic kappa values must be nonzero.
    #[error("XSPH relativistic kappa must be nonzero")]
    ZeroKappa,
    /// XSPH radius-like inputs must be finite and positive.
    #[error("{name} must be finite and positive, got {value}")]
    InvalidPositiveRadius { name: &'static str, value: Real },
    /// FEFF only defines some radial-integral multipoles for each branch.
    #[error("XSPH radial-integral mode {mode:?} does not support transition {multipole:?}")]
    UnsupportedRadialMultipole {
        mode: XsphRadialIntegralMode,
        multipole: XsphTransitionMultipole,
    },
    /// Integer angular inputs must stay in the supported FEFF range.
    #[error("{name} value {value} is outside the supported XSPH integer range")]
    IntegerOutOfRange { name: &'static str, value: i32 },
    /// Real-valued FEFF integer assignments must fit an `i32`.
    #[error("{name} value {value} cannot be represented as a FEFF integer")]
    RealIntegerOutOfRange { name: &'static str, value: Real },
    /// Rust-sized inputs must fit the FEFF integer helper range.
    #[error("{name} size {value} is outside the supported XSPH integer range")]
    SizeOutOfRange { name: &'static str, value: usize },
    /// FEFF `bcoefjas` generated too few final-state rows for `indmax`.
    #[error("XSPH generated {generated} NRIXS final states, fewer than active length {required}")]
    InsufficientGeneratedStates { required: usize, generated: usize },
    /// FEFF `specupd*` spin indices are limited to two spin components.
    #[error("XSPH spin index must be 0 or 1, got {spin_index}")]
    InvalidSpinIndex { spin_index: usize },
    /// FEFF `specupd*` received an unsupported MDFF selector.
    #[error("XSPH unsupported MDFF mode {mdff_mode}")]
    InvalidMdffMode { mdff_mode: i32 },
    /// Multidimensional arrays must have enough rows for the FEFF active shape.
    #[error("{name} shape {actual:?} is smaller than required shape {required:?}")]
    ShapeTooSmall {
        name: &'static str,
        required: [usize; 3],
        actual: [usize; 3],
    },
    /// Two-dimensional q-pair tables must cover every active q weight.
    #[error("{name} shape {actual:?} is smaller than required shape {required:?}")]
    MatrixTooSmall {
        name: &'static str,
        required: [usize; 2],
        actual: [usize; 2],
    },
    /// FEFF `axafs` requires at least three points after `ik0`.
    #[error("XSPH AXAFS requires at least three points after ik0, got {point_count}")]
    InsufficientAxafsPoints { point_count: usize },
    /// AXAFS grid indices must select a nonempty horizontal tail.
    #[error(
        "XSPH AXAFS zero-wave index {zero_wave_index} is invalid for horizontal count {horizontal_count}"
    )]
    InvalidAxafsGridIndex {
        zero_wave_index: usize,
        horizontal_count: usize,
    },
    /// The quadratic AXAFS background fit is singular.
    #[error("XSPH AXAFS quadratic background fit is singular")]
    SingularAxafsFit,
    /// AXAFS normalization must be nonzero.
    #[error("XSPH AXAFS normalization is zero")]
    ZeroAxafsNormalization,
    /// JAS orthogonality correction normalization must be nonzero.
    #[error("XSPH JAS orthogonality normalization is zero")]
    ZeroJasOrthogonalityNormalization,
    /// AXAFS background rows must be nonzero to compute `chi_at`.
    #[error("XSPH AXAFS background row {index} is zero")]
    ZeroAxafsBackground { index: usize },
    /// FEFF `GetOccNorm` has default rows for elements 1 through 100.
    #[error(
        "XSPH occupation normalization atomic number {atomic_number} is outside 1..={max_atomic_number}"
    )]
    InvalidOccupationNormAtomicNumber {
        atomic_number: usize,
        max_atomic_number: usize,
    },
    /// FEFF `GetOccNorm` uses one-based hole selectors in `1..=29`.
    #[error(
        "XSPH occupation normalization hole index {hole_index} is outside 1..={max_hole_index}"
    )]
    InvalidOccupationNormHoleIndex {
        hole_index: usize,
        max_hole_index: usize,
    },
    /// Some FEFF `GetOccNorm` denominator entries are zero for unsupported holes.
    #[error("XSPH occupation normalization denominator is zero for hole index {hole_index}")]
    ZeroOccupationNormDenominator { hole_index: usize },
    /// Hole-orbital spinor components must have matching source-grid lengths.
    #[error("XSPH hole-orbital length mismatch: large={large_len}, small={small_len}")]
    HoleOrbitalLengthMismatch { large_len: usize, small_len: usize },
    /// FEFF `jnew` must fit inside `nrptx`.
    #[error("XSPH hole-orbital output count {output_count} exceeds capacity {output_capacity}")]
    InvalidHoleOrbitalOutputCount {
        output_count: usize,
        output_capacity: usize,
    },
    /// At least one nonzero source sample is needed before interpolation.
    #[error("XSPH hole-orbital source components are zero below the FEFF tail cutoff")]
    EmptyHoleOrbital,
    /// FEFF phase-grid helpers need sufficient output capacity.
    #[error("XSPH phase mesh capacity is too small: {capacity}")]
    InvalidPhaseMeshCapacity { capacity: usize },
    /// FEFF phase-energy mesh counters must form a valid prefix.
    #[error(
        "XSPH auxiliary energy count {auxiliary_count} exceeds total energy count {energy_count}"
    )]
    InvalidAuxiliaryEnergyCount {
        auxiliary_count: usize,
        energy_count: usize,
    },
    /// FEFF phase finalization needs `ne1` to name an active reference energy.
    #[error("XSPH real energy count {real_mesh_count} is outside 1..={energy_count}")]
    InvalidRealEnergyCount {
        real_mesh_count: usize,
        energy_count: usize,
    },
    /// FEFF phase radial reference indices are one-based.
    #[error("XSPH phase radial reference index must be positive, got {index_1based}")]
    InvalidPhaseRadialReferenceIndex { index_1based: usize },
    /// FEFF phase radial match indices are one-based.
    #[error("XSPH phase radial match index must be positive, got {index_1based}")]
    InvalidPhaseRadialMatchIndex { index_1based: usize },
    /// FEFF `phase_h` Hubbard branch only has spin selectors 1 and 2.
    #[error("XSPH Hubbard phase spin selector must be 1 or 2, got {spin_projection}")]
    InvalidHubbardSpinProjection { spin_projection: i32 },
    /// FEFF would divide `PrintRl` radial components by a zero `phamp` amplitude.
    #[error("XSPH phase radial output amplitude is zero")]
    ZeroPhaseAmplitude,
    /// A denominator or scale that FEFF divides by evaluated to zero.
    #[error("XSPH {name} complex result is zero")]
    ZeroComplexResult { name: &'static str },
    /// This safe wrapper only exposes the default FEFF84 `phmesh2` branches.
    #[error("XSPH FEFF84 phase mesh does not support spectroscopy selector {spectroscopy}")]
    UnsupportedPhaseMeshSpectroscopy { spectroscopy: i32 },
    /// FEFF user-grid branch requires at least one `grid.inp` record.
    #[error("XSPH user phase mesh requires at least one grid record")]
    EmptyPhaseGridRecords,
    /// FEFF `rdgrid.f90` stores at most ten `grid.inp` records.
    #[error("XSPH user phase mesh supports at most {max} grid records, got {count}")]
    TooManyPhaseGridRecords { count: usize, max: usize },
    /// FEFF phase-grid helpers need a nonzero finite step.
    #[error("XSPH phase mesh step {name} must be finite and nonzero, got {value}")]
    InvalidPhaseMeshStep { name: &'static str, value: Real },
    /// Exponential phase-grid endpoints must be finite and positive.
    #[error("XSPH exponential phase mesh endpoint {name} must be finite and positive, got {value}")]
    InvalidPhaseMeshEndpoint { name: &'static str, value: Real },
    /// FEFF `SortE` expects at least one energy point.
    #[error("XSPH phase mesh sorting requires at least one energy point")]
    EmptyPhaseMesh,
    /// FEFF interpolation helper failed.
    #[error(transparent)]
    Interpolation(#[from] InterpolationError),
}
