//! Public data structures for FEFF `GENFMT` helper routines.

use ndarray::{
    Array1, Array2, Array3, Array4, Array5, ArrayView1, ArrayView2, ArrayView3, ArrayView4,
    ArrayView5, ArrayView6,
};
use thiserror::Error;

use crate::{AngularError, AtomicError, Complex, PhaseError, QuadratureError, Real};

/// Inputs for FEFF `GENFMT/rot3i.f90` initial-state rotation matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialStateRotationInput {
    /// FEFF `lxp1`, equal to `lmax + 1`.
    pub lmaxp1: usize,
    /// FEFF `mxp1`, equal to `mmax + 1`.
    pub mmaxp1: usize,
    /// FEFF `beta(ileg)` scattering angle in radians.
    pub beta_angle: Real,
}

/// Inputs for the FEFF GENFMT path-local `rot3i` loop.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtPathRotationTablesInput<'a> {
    /// FEFF `beta(1:nangle)` scattering angles from `rdpath`.
    pub beta_angles: ArrayView1<'a, Real>,
    /// FEFF `nleg`, the count of real path legs.
    pub leg_count: usize,
    /// FEFF `lmaxp1` for ordinary path-leg rotations.
    pub lmaxp1: usize,
    /// FEFF `mmaxp1` for ordinary path-leg rotations.
    pub mmaxp1: usize,
    /// Optional dimensions for FEFF's polarized pseudo-leg at `nleg + 1`.
    ///
    /// Ordinary GENFMT passes `(ilinit + 1, ilinit + 1)` here, while the
    /// current GENFMTJAS source uses `(lmaxp1, mmaxp1)`.
    pub polarized_extra: Option<(usize, usize)>,
}

/// Inputs for FEFF `GENFMT/rdpath.f90` path angle construction.
#[derive(Debug, Clone, Copy)]
pub struct PathRotationInput<'a> {
    /// Path atom coordinates as `(nleg, 3)`.
    ///
    /// Row `0` is FEFF `rat(:,1)`, and row `nleg - 1` is the absorber row
    /// used as FEFF `rat(:,nleg)`. Coordinates are used as supplied; callers
    /// should perform any Angstrom/Bohr conversion before calling this helper.
    pub positions: ArrayView2<'a, Real>,
    /// Whether to include FEFF's extra z-axis polarization pseudo-leg.
    pub polarized: bool,
}

/// Inputs for ordinary FEFF `GENFMT/genfmtsub.f90` path setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathSetupInput<'a> {
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)`.
    pub positions: ArrayView2<'a, Real>,
    /// Whether FEFF `ipol > 0`, enabling the polarization pseudo-leg.
    pub polarized: bool,
    /// FEFF `icalc` selector passed to `setlam`.
    pub calculation: i32,
    /// FEFF `ilinit`, the initial-state angular momentum.
    pub initial_l: usize,
    /// FEFF `lamtot`, the capacity of `mlam` and `nlam`.
    pub lambda_capacity: usize,
    /// FEFF `mtot`, the maximum magnetic index dimension.
    pub max_m: usize,
    /// FEFF `ntot`, the maximum order index dimension.
    pub max_n: usize,
    /// FEFF `lmaxp1` for real-leg `rot3i` calls.
    pub lmaxp1: usize,
}

/// Inputs for FEFF `GENFMT/genfmtjas.f90` path setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathSetupInput<'a> {
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)`.
    pub positions: ArrayView2<'a, Real>,
    /// Whether FEFF `ipol > 0`, enabling the polarization pseudo-leg.
    pub polarized: bool,
    /// FEFF `icalc` selector passed to `setlam`.
    pub calculation: i32,
    /// FEFF `ilinit`, the initial-state angular momentum.
    pub initial_l: usize,
    /// FEFF `lamtot`, the capacity of `mlam` and `nlam`.
    pub lambda_capacity: usize,
    /// FEFF `mtot`, the maximum magnetic index dimension.
    pub max_m: usize,
    /// FEFF `ntot`, the maximum order index dimension.
    pub max_n: usize,
    /// FEFF `lmaxp1` for real-leg and polarized pseudo-leg `rot3i` calls.
    pub lmaxp1: usize,
}

/// Inputs for FEFF `GENFMT/setlam.f90` lambda-index selection.
#[derive(Debug, Clone, Copy)]
pub struct LambdaIndexInput<'a> {
    /// FEFF `icalc` selector: `0..=9` for exact order, `10` for the cute
    /// heuristic, or a negative encoded `(nmax, mmax, iord)` request.
    pub calculation: i32,
    /// FEFF one-based energy index `ie`; the cute heuristic raises `nmax` for
    /// `ie >= 42`.
    pub energy_index: usize,
    /// FEFF `nsc`, used to detect single-scattering paths.
    pub scattering_count: usize,
    /// FEFF `ilinit`, the initial-state angular momentum.
    pub initial_l: usize,
    /// FEFF `beta(1:nleg)` path scattering angles in radians.
    pub beta_angles: &'a [Real],
    /// FEFF `lamtot`, the capacity of `mlam` and `nlam`.
    pub lambda_capacity: usize,
    /// FEFF `mtot`, the maximum magnetic index dimension.
    pub max_m: usize,
    /// FEFF `ntot`, the maximum order index dimension.
    pub max_n: usize,
}

/// Inputs for FEFF `GENFMT/xstar.f90` central-atom plane-wave factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XStarInput {
    /// FEFF `eps1`: primary polarization vector.
    pub primary_polarization: [Real; 3],
    /// FEFF `eps2`: secondary polarization vector for elliptic polarization.
    pub secondary_polarization: [Real; 3],
    /// FEFF `vec1`: direction to the first atom in the path.
    pub first_leg: [Real; 3],
    /// FEFF `vec2`: direction to the last atom in the path.
    pub last_leg: [Real; 3],
    /// FEFF `ndeg`, the path degeneracy used for this approximation.
    pub degeneracy: Real,
    /// FEFF `ilinit`, supported by the embedded Legendre table for `1..=4`.
    pub initial_l: usize,
    /// FEFF `elpty`, the ellipticity ratio.
    pub ellipticity: Real,
}

/// Inputs for one FEFF GENFMT `nstar.dat` path row.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtNStarInput<'a> {
    /// Sequential FEFF `npath` written to `nstar.dat`.
    pub path_number: usize,
    /// Path atom coordinates as `(nleg, 3)`, matching [`PathRotationInput`].
    pub positions: ArrayView2<'a, Real>,
    /// FEFF `evec`, the primary polarization vector.
    pub primary_polarization: [Real; 3],
    /// FEFF `xivec`; GENFMT forms `eps2=xivec cross evec` when `elpty != 0`.
    pub ellipticity_vector: [Real; 3],
    /// FEFF path degeneracy `deg`, rounded with `nint` before calling `xstar`.
    pub degeneracy: Real,
    /// FEFF `ilinit`, supported by the embedded Legendre table for `1..=4`.
    pub initial_l: usize,
    /// FEFF `elpty`, the ellipticity ratio.
    pub ellipticity: Real,
}

/// One FEFF GENFMT `nstar.dat` path row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtNStarRow {
    /// Sequential FEFF `npath`.
    pub path_number: usize,
    /// FEFF `xstar(...)` value written as `n*`.
    pub nstar: Real,
}

/// Path-local inputs for one FEFF GENFMT `nstar.dat` row in driver order.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtNStarPathInput<'a> {
    /// Path atom coordinates as `(nleg, 3)`, matching [`PathRotationInput`].
    pub positions: ArrayView2<'a, Real>,
    /// FEFF path degeneracy `deg`, rounded with `nint` before calling `xstar`.
    pub degeneracy: Real,
}

/// Inputs for FEFF GENFMT `nstar.dat` row generation.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtNStarRowsInput<'a> {
    /// FEFF `evec`, the primary polarization vector written in the file header.
    pub primary_polarization: [Real; 3],
    /// FEFF `xivec`; GENFMT forms `eps2=xivec cross evec` when `elpty != 0`.
    pub ellipticity_vector: [Real; 3],
    /// FEFF `ilinit`, supported by the embedded Legendre table for `1..=4`.
    pub initial_l: usize,
    /// FEFF `elpty`, the ellipticity ratio.
    pub ellipticity: Real,
    /// Path rows in FEFF `paths.dat` traversal order.
    pub path_inputs: &'a [GenfmtNStarPathInput<'a>],
}

/// FEFF GENFMT `nstar.dat` rows plus the shared polarization header.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtNStarRows {
    /// FEFF `evec`, the primary polarization vector written in the file header.
    pub primary_polarization: [Real; 3],
    /// Rows in FEFF `paths.dat` traversal order.
    pub rows: Vec<GenfmtNStarRow>,
}

/// Shared FEFF `nstar.dat` controls for driver-level GENFMT output assembly.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtNStarDriverInput {
    /// FEFF `evec`, the primary polarization vector written in the file header.
    pub primary_polarization: [Real; 3],
    /// FEFF `xivec`; GENFMT forms `eps2=xivec cross evec` when `elpty != 0`.
    pub ellipticity_vector: [Real; 3],
    /// FEFF `ilinit`, supported by the embedded Legendre table for `1..=4`.
    pub initial_l: usize,
    /// FEFF `elpty`, the ellipticity ratio.
    pub ellipticity: Real,
}

/// Inputs for FEFF `GENFMT/genfmtjas.f90` q-vector angle preparation.
#[derive(Debug, Clone, Copy)]
pub struct JasQAngleInput<'a> {
    /// FEFF `qaverage`; spherical q averaging uses a single unrotated q entry.
    pub qaverage: bool,
    /// FEFF `qtrig(1:nq,1:4)` with columns `(cos theta, sin theta, cos phi, sin phi)`.
    pub q_trig: ArrayView2<'a, Real>,
    /// FEFF `qw(1:nq)` q weights. Non-averaged preparation preserves these values.
    pub q_weights: ArrayView1<'a, Complex>,
}

/// Prepared q-vector angles for FEFF JAS/NRIXS GENFMT paths.
#[derive(Debug, Clone, PartialEq)]
pub struct JasQAngles {
    /// FEFF `pha(iq)`, the conjugated azimuthal phase `cos(phi)-i sin(phi)`.
    pub phases: Array1<Complex>,
    /// FEFF `beta(iq)=atan2(sin(theta), cos(theta))`.
    pub beta_angles: Array1<Real>,
    /// FEFF q weights after the q-average override.
    pub weights: Array1<Complex>,
}

/// Inputs for FEFF GENFMT ordinary scattering-matrix call planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtScatteringMatrixPlanInput {
    /// FEFF `nleg`, equal to `nsc + 1`.
    pub leg_count: usize,
    /// FEFF `lamx`, the active full lambda count.
    pub full_lambda_count: usize,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub initial_lambda_count: usize,
}

/// Role of one FEFF ordinary `fmtrxi` scattering-matrix task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenfmtScatteringMatrixRole {
    /// FEFF `call fmtrxi(lamx,laml0x,ie,2,1)`.
    First,
    /// FEFF `call fmtrxi(laml0x,lamx,ie,nleg,nleg-1)`.
    LastOrdinary,
    /// FEFF `call fmtrxi(lamx,lamx,ie,ileg,ilegp)` for intermediate legs.
    Intermediate,
}

/// One ordinary FEFF `fmtrxi` task in GENFMT call order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtScatteringMatrixTask {
    /// The FEFF task role.
    pub role: GenfmtScatteringMatrixRole,
    /// Zero-based Rust index for FEFF `ileg`.
    pub current_leg_index: usize,
    /// Zero-based Rust index for FEFF `ilegp`.
    pub previous_leg_index: usize,
    /// Zero-based Rust output slot corresponding to FEFF `fmati(:,:,ilegp)`.
    pub matrix_slot_index: usize,
    /// FEFF `lam1x` passed to `fmtrxi`.
    pub left_lambda_count: usize,
    /// FEFF `lam2x` passed to `fmtrxi`.
    pub right_lambda_count: usize,
}

/// FEFF ordinary scattering-matrix tasks for one path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenfmtScatteringMatrixPlan {
    /// FEFF `nsc=nleg-1`.
    pub scattering_count: usize,
    /// Tasks in the same order FEFF calls `fmtrxi`.
    pub tasks: Vec<GenfmtScatteringMatrixTask>,
}

/// Inputs for the FEFF `GENFMT/genfmtsub.f90` path F-matrix product trace.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtPathMatrixTraceInput<'a> {
    /// FEFF `fmati(:,:,1)`, the first scattering matrix `f(2,1)`.
    ///
    /// Rust axes are `(lambda, initial_lambda)` and only the prefix selected by
    /// [`Self::full_lambda_count`] and [`Self::initial_lambda_count`] is used.
    pub first_scattering: ArrayView2<'a, Complex>,
    /// FEFF intermediate matrices `fmati(:,:,2:nleg-1)`.
    ///
    /// Rust axes are `(intermediate_leg, lambda, lambda)`. Single-scattering
    /// paths pass a zero-length first axis.
    pub intermediate_scattering: ArrayView3<'a, Complex>,
    /// FEFF termination matrix `fmati(:,:,nleg)`.
    ///
    /// Rust axes are `(initial_lambda, initial_lambda)`.
    pub termination_matrix: ArrayView2<'a, Complex>,
    /// FEFF `lamx`, the active full lambda count.
    pub full_lambda_count: usize,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub initial_lambda_count: usize,
}

/// Inputs for the FEFF `pmati` path F-matrix product.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtPathMatrixProductInput<'a> {
    /// FEFF `fmati(:,:,1)`, the first scattering matrix `f(2,1)`.
    ///
    /// Rust axes are `(lambda, initial_lambda)`.
    pub first_scattering: ArrayView2<'a, Complex>,
    /// FEFF intermediate matrices `fmati(:,:,2:nleg-1)`.
    ///
    /// Rust axes are `(intermediate_leg, lambda, lambda)`.
    pub intermediate_scattering: ArrayView3<'a, Complex>,
    /// FEFF `lamx`, the active full lambda count.
    pub full_lambda_count: usize,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub initial_lambda_count: usize,
}

/// FEFF alternating `pmati(:,:,indp)` path product.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtPathMatrixProduct {
    /// FEFF product after all intermediate scattering matrices.
    ///
    /// Rust axes are `(lambda, initial_lambda)`.
    pub product_matrix: Array2<Complex>,
}

/// FEFF path F-matrix product and final trace.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtPathMatrixTrace {
    /// FEFF alternating `pmati(:,:,indp)` product after all intermediate legs.
    ///
    /// Rust axes are `(lambda, initial_lambda)`.
    pub product_matrix: Array2<Complex>,
    /// FEFF `ptrac` after contracting the termination matrix with
    /// [`Self::product_matrix`].
    pub trace: Complex,
}

/// Inputs for the ordinary FEFF GENFMT per-energy path trace.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathTraceInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lamx`, the active full lambda count.
    pub full_lambda_count: usize,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub initial_lambda_count: usize,
    /// FEFF `ipot(1:nleg)` path potential indices.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(ie,0:npot)` inclusive angular limits for this energy.
    pub angular_limits: ArrayView1<'a, usize>,
    /// FEFF `ph(ie,-ltot:ltot,0:npot)` for this energy.
    ///
    /// Rust axes are `(signed_l + signed_angular_offset, potential)`.
    pub phase_shifts: ArrayView2<'a, Complex>,
    /// Offset used to map signed angular momentum onto the first phase axis.
    pub signed_angular_offset: usize,
    /// FEFF `clmi(:,:,1:nleg)` curved-wave polynomial tables.
    ///
    /// Rust axes are `(l, mixed_order, leg)`.
    pub curved_wave_polynomials: ArrayView3<'a, Complex>,
    /// FEFF `dri(:,:,:,1:nleg)` rotation matrices.
    ///
    /// Rust axes are `(leg, l, m1 + rotation_magnetic_offset,
    /// m2 + rotation_magnetic_offset)`.
    pub rotations: ArrayView4<'a, Real>,
    /// Magnetic-index offset for the rotation matrix axes.
    pub rotation_magnetic_offset: usize,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(1:nleg)` phase factors.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF transition angular momenta `lind(1:8)`.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF radial transition factors `rkk(ie,1:8)` for this energy and spin.
    pub radial_factors: ArrayView1<'a, Complex>,
    /// FEFF `bmati(-mtot:mtot,1:8,-mtot:mtot,1:8)`.
    ///
    /// Rust indices are `(m1 + transition_magnetic_offset, k1,
    /// m2 + transition_magnetic_offset, k2)`.
    pub transition_matrix: ArrayView4<'a, Complex>,
    /// Magnetic-index offset for the first and third transition matrix axes.
    pub transition_magnetic_offset: usize,
}

/// Ordinary FEFF GENFMT path trace for one energy and spin channel.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinaryPathTrace {
    /// FEFF `fmati(:,:,1:nleg-1)` scattering matrices in matrix-slot order.
    ///
    /// Each matrix keeps the actual `fmtrxi` dimensions for its slot. The final
    /// pre-termination matrix can therefore have only `laml0x` rows, matching
    /// FEFF's `call fmtrxi(laml0x,lamx,...)`.
    pub scattering_matrices: Vec<Array2<Complex>>,
    /// FEFF `fmati(:,:,nleg)` termination matrix from `mmtrxi`.
    pub termination_matrix: Array2<Complex>,
    /// FEFF product matrix and final `ptrac` contraction.
    pub matrix_trace: GenfmtPathMatrixTrace,
}

/// Inputs for building FEFF ordinary/JAS scattering matrices and `pmati`.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtScatteringPathProductInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lamx`, the active full lambda count.
    pub full_lambda_count: usize,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub initial_lambda_count: usize,
    /// FEFF `ipot(1:nleg)` path potential indices.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(ie,0:npot)` inclusive angular limits for this energy.
    pub angular_limits: ArrayView1<'a, usize>,
    /// FEFF `ph(ie,-ltot:ltot,0:npot)` for this energy.
    ///
    /// Rust axes are `(signed_l + signed_angular_offset, potential)`.
    pub phase_shifts: ArrayView2<'a, Complex>,
    /// Offset used to map signed angular momentum onto the first phase axis.
    pub signed_angular_offset: usize,
    /// FEFF `clmi(:,:,1:nleg)` curved-wave polynomial tables.
    ///
    /// Rust axes are `(l, mixed_order, leg)`.
    pub curved_wave_polynomials: ArrayView3<'a, Complex>,
    /// FEFF `dri(:,:,:,1:nleg)` rotation matrices.
    ///
    /// Rust axes are `(leg, l, m1 + rotation_magnetic_offset,
    /// m2 + rotation_magnetic_offset)`.
    pub rotations: ArrayView4<'a, Real>,
    /// Magnetic-index offset for the rotation matrix axes.
    pub rotation_magnetic_offset: usize,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(1:nleg)` phase factors.
    pub eta_angles: ArrayView1<'a, Real>,
}

/// FEFF scattering matrices and their `pmati` path product.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtScatteringPathProduct {
    /// FEFF `fmati(:,:,1:nleg-1)` scattering matrices in matrix-slot order.
    pub scattering_matrices: Vec<Array2<Complex>>,
    /// FEFF `pmati(:,:,indp)` after all intermediate scattering matrices.
    pub matrix_product: GenfmtPathMatrixProduct,
}

/// Inputs for the ordinary FEFF GENFMT work at one energy and spin channel.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathEnergyPointInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lamx`, the active full lambda count.
    pub full_lambda_count: usize,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub initial_lambda_count: usize,
    /// FEFF zero-based energy index corresponding to `ie`.
    pub energy_index: usize,
    /// FEFF `ipot(1:nleg)` path potential indices.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// FEFF `ph(1:ne,-ltot:ltot,0:npot)` for this spin channel.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, potential)`.
    pub phase_shifts: ArrayView3<'a, Complex>,
    /// Offset used to map signed angular momentum onto the phase axis.
    pub signed_angular_offset: usize,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF `ck(ie)`, the complex momentum for this energy and spin.
    pub complex_momentum: Complex,
    /// FEFF `xk(ie)`, the real photoelectron wave number for this energy.
    pub wave_number: Real,
    /// FEFF `eps`, the zero-momentum threshold.
    pub momentum_zero_epsilon: Real,
    /// FEFF `mmaxp1`, equal to `mmax + 1`.
    pub max_m_plus_one: usize,
    /// FEFF `nmax`, the current Rehr-Albers order limit.
    pub max_n: usize,
    /// FEFF `dri(:,:,:,1:nleg)` rotation matrices.
    ///
    /// Rust axes are `(leg, l, m1 + rotation_magnetic_offset,
    /// m2 + rotation_magnetic_offset)`.
    pub rotations: ArrayView4<'a, Real>,
    /// Magnetic-index offset for the rotation matrix axes.
    pub rotation_magnetic_offset: usize,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(1:nleg)` phase factors.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF transition angular momenta `lind(1:8)`.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `rkk(1:ne,1:8)` radial transition factors for this spin channel.
    pub radial_factors: ArrayView2<'a, Complex>,
    /// FEFF `bmati(-mtot:mtot,1:8,-mtot:mtot,1:8)`.
    ///
    /// Rust indices are `(m1 + transition_magnetic_offset, k1,
    /// m2 + transition_magnetic_offset, k2)`.
    pub transition_matrix: ArrayView4<'a, Complex>,
    /// Magnetic-index offset for the first and third transition matrix axes.
    pub transition_magnetic_offset: usize,
    /// Current FEFF `cchi(ie)` before this spin contribution is added.
    pub accumulated_chi: Complex,
    /// FEFF `nsp`; supported values are one or two spin channels.
    pub spin_channel_count: usize,
    /// Zero-based Rust spin index. FEFF `is=1` maps to `0`.
    pub spin_index: usize,
}

/// Ordinary FEFF GENFMT calculation products for one energy and spin channel.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinaryPathEnergyPoint {
    /// FEFF `rho`, `reff`, and zero-momentum branch state.
    pub geometry: GenfmtPathGeometry,
    /// FEFF per-leg `sclmz` limits, absent for skipped zero-momentum energies.
    pub leg_limits: Option<GenfmtCurvedWaveLegLimits>,
    /// FEFF `clmi` curved-wave polynomial table, absent when skipped.
    pub curved_wave_polynomials: Option<GenfmtCurvedWavePolynomialTables>,
    /// FEFF scattering matrices and `ptrac`, absent when skipped.
    pub path_trace: Option<GenfmtOrdinaryPathTrace>,
    /// FEFF `cfac`, absent when skipped.
    pub path_factor: Option<GenfmtCurvedWavePathFactor>,
    /// FEFF contribution to `cchi(ie)`, absent when skipped.
    pub signal: Option<GenfmtPathSignalContribution>,
}

/// Inputs for wiring ordinary GENFMT path setup into the spin/energy loop.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathEnergyGridFromSetupInput<'a> {
    /// Checked path-local setup from `rdpath`, `setlam`, and `rot3i`.
    pub path_setup: &'a GenfmtPathSetup,
    /// FEFF `ipot(1:nleg)` path potential indices.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// FEFF `ph4(1:ne,-ltot:ltot,1:nspx,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, spin, potential)`.
    pub spin_phase_shifts: ArrayView4<'a, Complex>,
    /// Offset used to map signed angular momentum onto the phase axis.
    pub signed_angular_offset: usize,
    /// FEFF `ck(1:ne,1:nsp)`, the spin-resolved complex momentum grid.
    ///
    /// Rust axes are `(energy, spin)`.
    pub complex_momenta: ArrayView2<'a, Complex>,
    /// FEFF `xk(1:ne)`, the real photoelectron wave-number grid.
    pub wave_numbers: ArrayView1<'a, Real>,
    /// FEFF `eps`, the zero-momentum threshold.
    pub momentum_zero_epsilon: Real,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF transition angular momenta `lind(1:8)`.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `rkk2(1:ne,1:8,1:nspx)`.
    ///
    /// Rust axes are `(energy, transition, spin)`.
    pub spin_radial_factors: ArrayView3<'a, Complex>,
    /// FEFF `bmati` for each active spin channel.
    ///
    /// Rust axes are `(spin, m1 + transition_magnetic_offset, k1,
    /// m2 + transition_magnetic_offset, k2)`.
    pub transition_matrices: ArrayView5<'a, Complex>,
    /// Magnetic-index offset for transition matrix magnetic axes.
    pub transition_magnetic_offset: usize,
    /// FEFF `nsp`; supported values are one or two spin channels.
    pub spin_channel_count: usize,
}

/// Inputs for wiring ordinary GENFMT checked driver/path setup into the spin loop.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathEnergyGridFromDriverSetupInput<'a> {
    /// Checked ordinary GENFMT driver setup.
    pub driver_setup: &'a GenfmtDriverSetup,
    /// Checked path-local setup from `rdpath`, `setlam`, and `rot3i`.
    pub path_setup: &'a GenfmtPathSetup,
    /// FEFF `ipot(1:nleg)` path potential indices.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// FEFF `ph4(1:ne,-ltot:ltot,1:nspx,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, spin, potential)`.
    pub spin_phase_shifts: ArrayView4<'a, Complex>,
    /// Offset used to map signed angular momentum onto the phase axis.
    pub signed_angular_offset: usize,
    /// FEFF `eps`, the zero-momentum threshold.
    pub momentum_zero_epsilon: Real,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF transition angular momenta `lind(1:8)`.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `rkk2(1:ne,1:8,1:nspx)`.
    ///
    /// Rust axes are `(energy, transition, spin)`.
    pub spin_radial_factors: ArrayView3<'a, Complex>,
    /// FEFF `bmati` for each active spin channel.
    ///
    /// Rust axes are `(spin, m1 + transition_magnetic_offset, k1,
    /// m2 + transition_magnetic_offset, k2)`.
    pub transition_matrices: ArrayView5<'a, Complex>,
    /// Magnetic-index offset for transition matrix magnetic axes.
    pub transition_magnetic_offset: usize,
}

/// Inputs for the ordinary FEFF GENFMT spin and energy loop for one path.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathEnergyGridInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lamx`, the active full lambda count.
    pub full_lambda_count: usize,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub initial_lambda_count: usize,
    /// FEFF `ipot(1:nleg)` path potential indices.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// FEFF `ph4(1:ne,-ltot:ltot,1:nspx,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, spin, potential)`.
    pub spin_phase_shifts: ArrayView4<'a, Complex>,
    /// Offset used to map signed angular momentum onto the phase axis.
    pub signed_angular_offset: usize,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF `ck(1:ne,1:nsp)`, the spin-resolved complex momentum grid.
    ///
    /// Rust axes are `(energy, spin)`.
    pub complex_momenta: ArrayView2<'a, Complex>,
    /// FEFF `xk(1:ne)`, the real photoelectron wave-number grid.
    pub wave_numbers: ArrayView1<'a, Real>,
    /// FEFF `eps`, the zero-momentum threshold.
    pub momentum_zero_epsilon: Real,
    /// FEFF `mmaxp1`, equal to `mmax + 1`.
    pub max_m_plus_one: usize,
    /// FEFF `nmax`, the current Rehr-Albers order limit.
    pub max_n: usize,
    /// FEFF `dri(:,:,:,1:nleg)` rotation matrices.
    ///
    /// Rust axes are `(leg, l, m1 + rotation_magnetic_offset,
    /// m2 + rotation_magnetic_offset)`.
    pub rotations: ArrayView4<'a, Real>,
    /// Magnetic-index offset for the rotation matrix axes.
    pub rotation_magnetic_offset: usize,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(1:nleg)` phase factors.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF transition angular momenta `lind(1:8)`.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `rkk2(1:ne,1:8,1:nspx)`.
    ///
    /// Rust axes are `(energy, transition, spin)`.
    pub spin_radial_factors: ArrayView3<'a, Complex>,
    /// FEFF `bmati` for each active spin channel.
    ///
    /// Rust axes are `(spin, m1 + transition_magnetic_offset, k1,
    /// m2 + transition_magnetic_offset, k2)`.
    pub transition_matrices: ArrayView5<'a, Complex>,
    /// Magnetic-index offset for transition matrix magnetic axes.
    pub transition_magnetic_offset: usize,
    /// FEFF `nsp`; supported values are one or two spin channels.
    pub spin_channel_count: usize,
}

/// Ordinary FEFF GENFMT spin and energy loop products for one path.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinaryPathEnergyGrid {
    /// Whether each FEFF spin/energy point reached the signal calculation.
    ///
    /// Rust axes are `(spin, energy)`.
    pub active: Array2<bool>,
    /// FEFF `ptrac(is,ie)` before multiplication by `cfac`.
    pub path_traces: Array2<Complex>,
    /// FEFF `cfac(is,ie)` before the two-spin sign flip.
    pub path_factors: Array2<Complex>,
    /// FEFF per-spin contributions and spin-summed `cchi(1:ne)`.
    pub signals: GenfmtPathSignals,
}

/// Inputs for evaluating one ordinary FEFF GENFMT path.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathEvaluationInput<'a> {
    /// Ordinary GENFMT spin and energy loop inputs.
    pub energy_grid: GenfmtOrdinaryPathEnergyGridInput<'a>,
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF `beta(1:nleg)`.
    pub beta_angles: ArrayView1<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// Inputs for evaluating one ordinary FEFF GENFMT path from checked setup products.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathEvaluationFromSetupInput<'a> {
    /// Checked path setup plus ordinary spin/energy-loop tables.
    pub energy_grid: GenfmtOrdinaryPathEnergyGridFromSetupInput<'a>,
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// Inputs for evaluating one ordinary FEFF GENFMT path from checked driver/path setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathEvaluationFromDriverSetupInput<'a> {
    /// Checked driver/path setup plus ordinary spin-loop tables.
    pub energy_grid: GenfmtOrdinaryPathEnergyGridFromDriverSetupInput<'a>,
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// Ordinary FEFF GENFMT products for one evaluated path.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinaryPathEvaluation {
    /// FEFF spin and energy loop result.
    pub energy_grid: GenfmtOrdinaryPathEnergyGrid,
    /// FEFF path importance, retention, and output payload.
    pub finalization: GenfmtOrdinaryPathFinalization,
}

/// Inputs for evaluating ordinary FEFF GENFMT paths in driver order.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathSequenceInput<'a> {
    /// Path inputs in FEFF `paths.dat` traversal order.
    pub path_inputs: &'a [GenfmtOrdinaryPathEvaluationInput<'a>],
    /// Initial FEFF `xportx`; the drivers initialize this to `-1`.
    pub initial_normalization: Real,
}

/// Inputs for evaluating setup-based ordinary FEFF GENFMT paths in driver order.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathSequenceFromSetupInput<'a> {
    /// Setup-based path inputs in FEFF `paths.dat` traversal order.
    pub path_inputs: &'a [GenfmtOrdinaryPathEvaluationFromSetupInput<'a>],
    /// Initial FEFF `xportx`; the drivers initialize this to `-1`.
    pub initial_normalization: Real,
}

/// Inputs for evaluating driver-backed ordinary FEFF GENFMT paths in driver order.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathSequenceFromDriverSetupInput<'a> {
    /// Driver-backed path inputs in FEFF `paths.dat` traversal order.
    pub path_inputs: &'a [GenfmtOrdinaryPathEvaluationFromDriverSetupInput<'a>],
    /// Initial FEFF `xportx`; the drivers initialize this to `-1`.
    pub initial_normalization: Real,
}

/// Ordinary FEFF GENFMT path-loop products.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinaryPathSequence {
    /// Evaluated paths in traversal order.
    pub evaluations: Vec<GenfmtOrdinaryPathEvaluation>,
    /// Retained ordinary path outputs and counters.
    pub outputs: GenfmtOrdinaryPathOutputs,
}

/// Inputs for assembling ordinary GENFMT driver outputs after checked setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryDriverOutputInput<'a> {
    /// Checked ordinary GENFMT driver setup with the prepared `feff.bin` header.
    pub driver_setup: &'a GenfmtDriverSetup,
    /// Driver-backed path inputs in FEFF `paths.dat` traversal order.
    pub path_inputs: &'a [GenfmtOrdinaryPathEvaluationFromDriverSetupInput<'a>],
    /// Initial FEFF `xportx`; the drivers initialize this to `-1`.
    pub initial_normalization: Real,
    /// Optional FEFF `wnstar` bookkeeping controls.
    pub nstar: Option<GenfmtNStarDriverInput>,
}

/// Ordinary GENFMT driver-level output payloads before text serialization.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinaryDriverOutput {
    /// Prepared FEFF `feff.bin` header payload.
    pub header: GenfmtFeffBinHeader,
    /// Evaluated path sequence and retained ordinary path outputs.
    pub path_sequence: GenfmtOrdinaryPathSequence,
    /// Optional FEFF `nstar.dat` rows for all examined paths.
    pub nstar_rows: Option<GenfmtNStarRows>,
}

/// Inputs for the FEFF `GENFMT/genfmtjas.f90` left/right JAS trace contraction.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasLeftRightPathTraceInput<'a> {
    /// FEFF `pmati(:,:,indp)` product after intermediate scattering matrices.
    ///
    /// Rust axes are `(lambda, initial_lambda)`; this contraction uses the
    /// `lambda_count` square prefix as FEFF `pmati(lmp,lm,indp)`.
    pub path_product: ArrayView2<'a, Complex>,
    /// FEFF `fmatl(mj,iq,lambda)` with compact doubled-`j` rows.
    pub left_amplitudes: ArrayView3<'a, Complex>,
    /// FEFF `fmatr(mj,iq,lambda)` with compact doubled-`j` rows.
    pub right_amplitudes: ArrayView3<'a, Complex>,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub lambda_count: usize,
    /// FEFF `lgfmatl(mj,iq,ll,lambda)` when angular decomposition is active.
    pub decomposed_left_amplitudes: Option<ArrayView4<'a, Complex>>,
    /// FEFF `lgfmatr(mj,iq,ll,lambda)` when angular decomposition is active.
    pub decomposed_right_amplitudes: Option<ArrayView4<'a, Complex>>,
}

/// FEFF JAS left/right path trace and optional decomposition trace table.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasLeftRightPathTrace {
    /// FEFF total `ptrac`.
    pub trace: Complex,
    /// FEFF `pgtrl(lg2,lg1)` before multiplication by the curved-wave factor.
    ///
    /// Rust axes preserve the FEFF order `(left_decomposition, right_decomposition)`.
    pub decomposed_traces: Option<Array2<Complex>>,
}

/// Inputs for the FEFF `GENFMT/genfmtjas.f90` spherical JAS trace contraction.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasSphericalPathTraceInput<'a> {
    /// FEFF `pmati(:,:,indp)` product after intermediate scattering matrices.
    ///
    /// Rust axes are `(lambda, initial_lambda)`; this contraction uses the
    /// `lambda_count` square prefix as FEFF `pmati(lmp,lm,indp)`.
    pub path_product: ArrayView2<'a, Complex>,
    /// FEFF `fmats(mj,is2,lmp,lm)` with compact doubled-`j` rows.
    pub amplitudes: ArrayView4<'a, Complex>,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub lambda_count: usize,
    /// FEFF `lgfmats(mj,is2,ll,lmp,lm)` when angular decomposition is active.
    pub decomposed_amplitudes: Option<ArrayView5<'a, Complex>>,
}

/// FEFF spherical JAS path trace and optional diagonal decomposition table.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasSphericalPathTrace {
    /// FEFF total `ptrac`.
    pub trace: Complex,
    /// FEFF `pgtrl(ll,ll)` before multiplication by the curved-wave factor.
    ///
    /// Off-diagonal entries remain zero, matching the spherical-averaging
    /// branch in `genfmtjas.f90`.
    pub decomposed_traces: Option<Array2<Complex>>,
}

/// Inputs for the FEFF GENFMTJAS per-energy path trace branch.
#[derive(Debug, Clone, Copy)]
pub enum GenfmtJasPathTraceInput<'a> {
    /// FEFF `elpty >= 0`: left/right JAS amplitudes from `mmtrxijas`.
    LeftRight {
        /// FEFF `pmati(:,:,indp)` product after intermediate scattering.
        path_product: ArrayView2<'a, Complex>,
        /// Inputs for the `mmtrxijas` amplitude folding.
        amplitude_input: JasLeftRightAmplitudeInput<'a>,
    },
    /// FEFF spherical-averaging branch using `mmtrxijas0`.
    Spherical {
        /// FEFF `pmati(:,:,indp)` product after intermediate scattering.
        path_product: ArrayView2<'a, Complex>,
        /// Inputs for the `mmtrxijas0` amplitude folding.
        amplitude_input: JasScatteringAmplitudeInput<'a>,
    },
}

/// FEFF GENFMTJAS path trace for one energy point.
#[derive(Debug, Clone, PartialEq)]
pub enum GenfmtJasPathTrace {
    /// Left/right `mmtrxijas` branch.
    LeftRight {
        /// FEFF `fmatl`/`fmatr` and optional decomposition amplitudes.
        amplitudes: JasLeftRightAmplitudeMatrices,
        /// FEFF total `ptrac` and optional `pgtrl(lg2,lg1)`.
        trace: GenfmtJasLeftRightPathTrace,
    },
    /// Spherical `mmtrxijas0` branch.
    Spherical {
        /// FEFF `fmats` and optional decomposition amplitudes.
        amplitudes: JasScatteringAmplitudeMatrices,
        /// FEFF total `ptrac` and optional diagonal decomposition traces.
        trace: GenfmtJasSphericalPathTrace,
    },
}

/// Termination branch inputs for one FEFF GENFMTJAS energy point.
#[derive(Debug, Clone, Copy)]
pub enum GenfmtJasPathEnergyBranchInput<'a> {
    /// FEFF `elpty >= 0`: left/right JAS amplitudes from `mmtrxijas`.
    LeftRight {
        /// FEFF `lind(1:indmax)` final-state orbital angular momenta.
        transition_angular_momenta: ArrayView1<'a, i32>,
        /// FEFF `rkk(ie,1:nq,1:indmax)` for one energy point.
        radial_factors: ArrayView2<'a, Complex>,
        /// FEFF complex q-vector weights `qw(1:nq)`.
        q_weights: ArrayView1<'a, Complex>,
        /// FEFF `hbmatl(mj,mu,iq,k1)`.
        left_transition_matrix: ArrayView4<'a, Complex>,
        /// FEFF `hbmatr(mj,mu,iq,k1)`.
        right_transition_matrix: ArrayView4<'a, Complex>,
        /// FEFF `jinit`, doubled initial-state angular momentum.
        initial_j2: i32,
        /// Magnetic-index offset for transition matrix magnetic axes.
        transition_magnetic_offset: usize,
        /// FEFF `l2lp`, the inclusive angular limit for this leg pair.
        max_angular_momentum: usize,
        /// FEFF `ldecmx`; `None` disables angular decomposition.
        decomposition_l_max: Option<usize>,
    },
    /// FEFF spherical-averaging branch using `mmtrxijas0`.
    Spherical {
        /// FEFF `lind(1:indmax)` final-state orbital angular momenta.
        transition_angular_momenta: ArrayView1<'a, i32>,
        /// FEFF `rkk(ie,1:nq,1:indmax)` for one energy point.
        radial_factors: ArrayView2<'a, Complex>,
        /// FEFF complex q-vector weights `qw(1:nq)`.
        q_weights: ArrayView1<'a, Complex>,
        /// FEFF `hbmatrs(mj,is2,mu2,mu1,k1)`.
        transition_matrix: ArrayView5<'a, Complex>,
        /// FEFF `jinit`, doubled initial-state angular momentum.
        initial_j2: i32,
        /// Magnetic-index offset for transition matrix magnetic axes.
        transition_magnetic_offset: usize,
        /// FEFF `l2lp`, the inclusive angular limit for this leg pair.
        max_angular_momentum: usize,
        /// FEFF `ldecmx`; `None` disables angular decomposition.
        decomposition_l_max: Option<usize>,
    },
}

/// Inputs for the FEFF GENFMTJAS work at one energy point.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathEnergyPointInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lamx`, the active full lambda count.
    pub full_lambda_count: usize,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub initial_lambda_count: usize,
    /// FEFF zero-based energy index corresponding to `ie`.
    pub energy_index: usize,
    /// FEFF `ipot(1:nleg)` path potential indices.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// FEFF `ph(1:ne,-ltot:ltot,0:npot)` for this spin channel.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, potential)`.
    pub phase_shifts: ArrayView3<'a, Complex>,
    /// Offset used to map signed angular momentum onto the phase axis.
    pub signed_angular_offset: usize,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF `ck(ie)`, the complex momentum for this energy.
    pub complex_momentum: Complex,
    /// FEFF `xk(ie)`, the real photoelectron wave number for this energy.
    pub wave_number: Real,
    /// FEFF `eps`, the zero-momentum threshold.
    pub momentum_zero_epsilon: Real,
    /// FEFF `mmaxp1`, equal to `mmax + 1`.
    pub max_m_plus_one: usize,
    /// FEFF `nmax`, the current Rehr-Albers order limit.
    pub max_n: usize,
    /// FEFF `dri(:,:,:,1:nleg)` rotation matrices.
    ///
    /// Rust axes are `(leg, l, m1 + rotation_magnetic_offset,
    /// m2 + rotation_magnetic_offset)`.
    pub rotations: ArrayView4<'a, Real>,
    /// Magnetic-index offset for the rotation matrix axes.
    pub rotation_magnetic_offset: usize,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(1:nleg)` phase factors.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF JAS termination branch selected by `elpty`.
    pub branch: GenfmtJasPathEnergyBranchInput<'a>,
}

/// FEFF GENFMTJAS calculation products for one energy point.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasPathEnergyPoint {
    /// FEFF `rho`, `reff`, and zero-momentum branch state.
    pub geometry: GenfmtPathGeometry,
    /// FEFF per-leg `sclmz` limits, absent for skipped zero-momentum energies.
    pub leg_limits: Option<GenfmtCurvedWaveLegLimits>,
    /// FEFF `clmi` curved-wave polynomial table, absent when skipped.
    pub curved_wave_polynomials: Option<GenfmtCurvedWavePolynomialTables>,
    /// FEFF scattering matrices and `pmati`, absent when skipped.
    pub scattering_product: Option<GenfmtScatteringPathProduct>,
    /// FEFF JAS termination trace, absent when skipped.
    pub path_trace: Option<GenfmtJasPathTrace>,
    /// FEFF `cfac`, absent when skipped.
    pub path_factor: Option<GenfmtCurvedWavePathFactor>,
    /// FEFF `cchi` and optional `pgtrl` values, absent when skipped.
    pub signal: Option<GenfmtJasPathSignal>,
}

/// Termination branch inputs for a FEFF GENFMTJAS energy-grid path loop.
#[derive(Debug, Clone, Copy)]
pub enum GenfmtJasPathEnergyGridBranchInput<'a> {
    /// FEFF `elpty >= 0`: left/right JAS amplitudes from `mmtrxijas`.
    LeftRight {
        /// FEFF `lind(1:indmax)` final-state orbital angular momenta.
        transition_angular_momenta: ArrayView1<'a, i32>,
        /// FEFF `rkk(1:ne,1:nq,1:indmax)` for the active spin channel.
        radial_factors: ArrayView3<'a, Complex>,
        /// FEFF complex q-vector weights `qw(1:nq)`.
        q_weights: ArrayView1<'a, Complex>,
        /// FEFF `hbmatl(mj,mu,iq,k1)`.
        left_transition_matrix: ArrayView4<'a, Complex>,
        /// FEFF `hbmatr(mj,mu,iq,k1)`.
        right_transition_matrix: ArrayView4<'a, Complex>,
        /// FEFF `jinit`, doubled initial-state angular momentum.
        initial_j2: i32,
        /// Magnetic-index offset for transition matrix magnetic axes.
        transition_magnetic_offset: usize,
        /// FEFF `l2lp`, the inclusive angular limit for this leg pair.
        max_angular_momentum: usize,
        /// FEFF `ldecmx`; `None` disables angular decomposition.
        decomposition_l_max: Option<usize>,
    },
    /// FEFF spherical-averaging branch using `mmtrxijas0`.
    Spherical {
        /// FEFF `lind(1:indmax)` final-state orbital angular momenta.
        transition_angular_momenta: ArrayView1<'a, i32>,
        /// FEFF `rkk(1:ne,1:nq,1:indmax)` for the active spin channel.
        radial_factors: ArrayView3<'a, Complex>,
        /// FEFF complex q-vector weights `qw(1:nq)`.
        q_weights: ArrayView1<'a, Complex>,
        /// FEFF `hbmatrs(mj,is2,mu2,mu1,k1)`.
        transition_matrix: ArrayView5<'a, Complex>,
        /// FEFF `jinit`, doubled initial-state angular momentum.
        initial_j2: i32,
        /// Magnetic-index offset for transition matrix magnetic axes.
        transition_magnetic_offset: usize,
        /// FEFF `l2lp`, the inclusive angular limit for this leg pair.
        max_angular_momentum: usize,
        /// FEFF `ldecmx`; `None` disables angular decomposition.
        decomposition_l_max: Option<usize>,
    },
}

/// Inputs for wiring checked GENFMTJAS transition setup into the path energy loop.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasEnergyGridBranchFromTransitionSetupInput<'a> {
    /// Checked transition setup from the FEFF `genfmtjas.f90` pre-loop block.
    pub transition_setup: &'a GenfmtJasTransitionSetup,
    /// FEFF `lind(1:indmax)` final-state orbital angular momenta.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `rkk(1:ne,1:nq,1:indmax)` for the active spin channel.
    pub radial_factors: ArrayView3<'a, Complex>,
    /// FEFF complex q-vector weights `qw(1:nq)`.
    pub q_weights: ArrayView1<'a, Complex>,
    /// Magnetic-index offset for transition matrix magnetic axes.
    pub transition_magnetic_offset: usize,
    /// FEFF `l2lp`, the inclusive angular limit for this leg pair.
    pub max_angular_momentum: usize,
    /// FEFF `ldecmx`; `None` disables angular decomposition.
    pub decomposition_l_max: Option<usize>,
}

/// Inputs for wiring checked GENFMTJAS driver/path setup into the energy loop.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathEnergyGridFromSetupInput<'a> {
    /// Checked GENFMTJAS driver setup from the pre-path-loop block.
    pub driver_setup: &'a GenfmtJasDriverSetup,
    /// Checked path-local setup from `rdpath`, `setlam`, and `rot3i`.
    pub path_setup: &'a GenfmtPathSetup,
    /// FEFF `ipot(1:nleg)` path potential indices.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// Offset used to map signed angular momentum onto the phase axis.
    pub signed_angular_offset: usize,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF JAS termination branch selected by `elpty`.
    pub branch: GenfmtJasPathEnergyGridBranchInput<'a>,
    /// FEFF `eps`, the zero-momentum threshold.
    pub momentum_zero_epsilon: Real,
}

/// Inputs for the FEFF GENFMTJAS energy loop for one path.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathEnergyGridInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lamx`, the active full lambda count.
    pub full_lambda_count: usize,
    /// FEFF `laml0x`, the active initial-state lambda count.
    pub initial_lambda_count: usize,
    /// FEFF `ipot(1:nleg)` path potential indices.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// FEFF `ph(1:ne,-ltot:ltot,0:npot)` for the active spin channel.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, potential)`.
    pub phase_shifts: ArrayView3<'a, Complex>,
    /// Offset used to map signed angular momentum onto the phase axis.
    pub signed_angular_offset: usize,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF `ck(1:ne)`, the complex momentum grid.
    pub complex_momenta: ArrayView1<'a, Complex>,
    /// FEFF `xk(1:ne)`, the real photoelectron wave-number grid.
    pub wave_numbers: ArrayView1<'a, Real>,
    /// FEFF `eps`, the zero-momentum threshold.
    pub momentum_zero_epsilon: Real,
    /// FEFF `mmaxp1`, equal to `mmax + 1`.
    pub max_m_plus_one: usize,
    /// FEFF `nmax`, the current Rehr-Albers order limit.
    pub max_n: usize,
    /// FEFF `dri(:,:,:,1:nleg)` rotation matrices.
    ///
    /// Rust axes are `(leg, l, m1 + rotation_magnetic_offset,
    /// m2 + rotation_magnetic_offset)`.
    pub rotations: ArrayView4<'a, Real>,
    /// Magnetic-index offset for the rotation matrix axes.
    pub rotation_magnetic_offset: usize,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(1:nleg)` phase factors.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF JAS termination branch selected by `elpty`.
    pub branch: GenfmtJasPathEnergyGridBranchInput<'a>,
}

/// FEFF GENFMTJAS energy-loop products for one path.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasPathEnergyGrid {
    /// Whether each FEFF energy point reached the signal calculation.
    pub active: Array1<bool>,
    /// FEFF `ptrac(1:ne)` before multiplication by `cfac`.
    pub path_traces: Array1<Complex>,
    /// FEFF `cfac(1:ne)`.
    pub path_factors: Array1<Complex>,
    /// Optional FEFF `ptrac(lg2,lg1,1:ne)` before multiplication by `cfac`.
    pub decomposed_traces: Option<Array3<Complex>>,
    /// FEFF `cchi(1:ne)` and optional `pgtrl(lg2,lg1,1:ne)`.
    pub signals: GenfmtJasPathSignals,
}

/// Inputs for evaluating one FEFF GENFMTJAS path.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathEvaluationInput<'a> {
    /// GENFMTJAS energy loop inputs.
    pub energy_grid: GenfmtJasPathEnergyGridInput<'a>,
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF `beta(1:nleg)`.
    pub beta_angles: ArrayView1<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// Inputs for evaluating one FEFF GENFMTJAS path from checked setup products.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathEvaluationFromSetupInput<'a> {
    /// Checked driver/path setup plus selected JAS termination branch.
    pub energy_grid: GenfmtJasPathEnergyGridFromSetupInput<'a>,
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// Inputs for evaluating one FEFF GENFMTJAS path from checked driver/path setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathEvaluationFromDriverSetupInput<'a> {
    /// Checked driver/path setup plus selected JAS termination branch.
    pub energy_grid: GenfmtJasPathEnergyGridFromSetupInput<'a>,
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// FEFF GENFMTJAS products for one evaluated path.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasPathEvaluation {
    /// FEFF energy loop result.
    pub energy_grid: GenfmtJasPathEnergyGrid,
    /// FEFF path importance, retention, and optional decomposition output.
    pub finalization: GenfmtJasPathFinalization,
}

/// Inputs for evaluating FEFF GENFMTJAS paths in driver order.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathSequenceInput<'a> {
    /// Path inputs in FEFF `paths.dat` traversal order.
    pub path_inputs: &'a [GenfmtJasPathEvaluationInput<'a>],
    /// Initial FEFF `xportx`; the drivers initialize this to `-1`.
    pub initial_normalization: Real,
}

/// Inputs for evaluating FEFF GENFMTJAS setup-based paths in driver order.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathSequenceFromSetupInput<'a> {
    /// Setup-based path inputs in FEFF `paths.dat` traversal order.
    pub path_inputs: &'a [GenfmtJasPathEvaluationFromSetupInput<'a>],
    /// Initial FEFF `xportx`; the drivers initialize this to `-1`.
    pub initial_normalization: Real,
}

/// Inputs for evaluating driver-backed FEFF GENFMTJAS paths in driver order.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathSequenceFromDriverSetupInput<'a> {
    /// Driver-backed path inputs in FEFF `paths.dat` traversal order.
    pub path_inputs: &'a [GenfmtJasPathEvaluationFromDriverSetupInput<'a>],
    /// Initial FEFF `xportx`; the drivers initialize this to `-1`.
    pub initial_normalization: Real,
}

/// FEFF GENFMTJAS path-loop products.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasPathSequence {
    /// Evaluated paths in traversal order.
    pub evaluations: Vec<GenfmtJasPathEvaluation>,
    /// Retained JAS path outputs, optional decomposition outputs, and counters.
    pub outputs: GenfmtJasPathOutputs,
}

/// Inputs for assembling GENFMTJAS driver outputs after checked setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasDriverOutputInput<'a> {
    /// Checked GENFMTJAS driver setup with the prepared `feff.bin` header.
    pub driver_setup: &'a GenfmtJasDriverSetup,
    /// Driver-backed path inputs in FEFF `paths.dat` traversal order.
    pub path_inputs: &'a [GenfmtJasPathEvaluationFromDriverSetupInput<'a>],
    /// Initial FEFF `xportx`; the drivers initialize this to `-1`.
    pub initial_normalization: Real,
    /// Optional FEFF `wnstar` bookkeeping controls.
    pub nstar: Option<GenfmtNStarDriverInput>,
}

/// GENFMTJAS driver-level output payloads before text serialization.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasDriverOutput {
    /// Prepared FEFF `feff.bin` header payload.
    pub header: GenfmtFeffBinHeader,
    /// Evaluated path sequence and retained JAS path outputs.
    pub path_sequence: GenfmtJasPathSequence,
    /// Optional FEFF `nstar.dat` rows for all examined paths.
    pub nstar_rows: Option<GenfmtNStarRows>,
}

/// Inputs for FEFF GENFMT active spin-channel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtSpinChannelCountInput {
    /// FEFF `ispin`; only `ispin == 1` activates both spin channels.
    pub spin_selector: i32,
    /// FEFF `nspx`, the number of spin channels available from XSPH.
    pub available_spin_channels: usize,
}

/// FEFF GENFMT reference-energy selection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenfmtReferenceEnergyMode {
    /// FEFF header/setup branch before the path loop.
    ///
    /// One-spin data uses the first spin slot; two-spin data averages the
    /// first and last active spin slots.
    Header,
    /// FEFF per-spin path loop branch.
    ///
    /// The zero-based Rust index maps to FEFF `is`.
    SpinChannel { spin_index: usize },
}

/// Inputs for FEFF GENFMT spin reference-energy preparation.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtSpinReferenceEnergyInput<'a> {
    /// FEFF `eref2(1:ne,1:nspx)` as `(energy, spin)`.
    pub spin_reference_energies: ArrayView2<'a, Complex>,
    /// FEFF active `nsp` after applying `ispin`.
    pub spin_channel_count: usize,
    /// Header averaging or one per-spin channel selection.
    pub mode: GenfmtReferenceEnergyMode,
}

/// FEFF GENFMT reference-energy vector for a header or spin-loop branch.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtSpinReferenceEnergies {
    /// FEFF `eref(1:ne)`.
    pub reference_energies: Array1<Complex>,
}

/// Inputs for FEFF GENFMT spin phase-shift preparation.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtSpinPhaseShiftInput<'a> {
    /// FEFF `ph4(1:ne,-ltot:ltot,1:nspx,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, spin, potential)`.
    pub spin_phase_shifts: ArrayView4<'a, Complex>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// Offset used to map signed angular momentum onto the second table axis.
    pub signed_angular_offset: usize,
    /// FEFF active `nsp` after applying `ispin`.
    pub spin_channel_count: usize,
    /// Header averaging or one per-spin channel selection.
    pub mode: GenfmtReferenceEnergyMode,
}

/// FEFF GENFMT phase-shift table for a header or spin-loop branch.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtSpinPhaseShifts {
    /// FEFF `ph(1:ne,-ltot:ltot,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, potential)`.
    pub phase_shifts: Array3<Complex>,
}

/// Inputs for FEFF GENFMT central-atom phase-shift header selection.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtCentralPhaseShiftInput<'a> {
    /// FEFF `ph(1:ne,-ltot:ltot,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, potential)`.
    pub phase_shifts: ArrayView3<'a, Complex>,
    /// Offset used to map signed angular momentum onto the second table axis.
    pub signed_angular_offset: usize,
    /// FEFF `linit`, the initial core orbital angular momentum.
    pub initial_orbital_l: usize,
    /// FEFF `kinit`, the relativistic kappa of the initial orbital.
    pub initial_kappa: i32,
}

/// FEFF GENFMT central-atom phase shifts written to the `feff.bin` header.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtCentralPhaseShifts {
    /// Signed angular momentum channel selected by FEFF as `ll`.
    pub signed_angular_momentum: i32,
    /// FEFF `ph(1:ne,ll,0)`.
    pub phase_shifts: Array1<Complex>,
}

/// Inputs for FEFF GENFMT per-spin radial transition factors.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtSpinRadialFactorInput<'a> {
    /// FEFF `rkk2(1:ne,1:8,1:nspx)` as `(energy, transition, spin)`.
    pub spin_radial_factors: ArrayView3<'a, Complex>,
    /// FEFF active `nsp` after applying `ispin`.
    pub spin_channel_count: usize,
    /// Zero-based Rust spin index. FEFF `is=1` maps to `0`.
    pub spin_index: usize,
}

/// FEFF GENFMT radial transition factors for one active spin channel.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtSpinRadialFactors {
    /// FEFF `rkk(1:ne,1:8)`.
    pub radial_factors: Array2<Complex>,
}

/// Inputs for FEFF GENFMT momentum-grid setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtMomentumGridInput<'a> {
    /// FEFF `em(1:ne)`, the complex energy grid.
    pub energies: ArrayView1<'a, Complex>,
    /// FEFF `eref(1:ne)`, the complex reference-energy grid.
    pub reference_energies: ArrayView1<'a, Complex>,
    /// FEFF `edge`, used with `getxk(dble(em(ie))-edge)`.
    pub edge: Real,
}

/// FEFF GENFMT momentum arrays derived from `em`, `eref`, and `edge`.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtMomentumGrid {
    /// FEFF `xk(1:ne)`, the signed real photoelectron wave number.
    pub wave_numbers: Array1<Real>,
    /// FEFF `ck(1:ne)=sqrt(2*(em(ie)-eref(ie)))`.
    pub complex_momenta: Array1<Complex>,
    /// FEFF `ckmag(1:ne)=abs(ck(ie))`.
    pub complex_momentum_magnitudes: Array1<Real>,
    /// FEFF `xkr(1:ne)=real(xk(ie))`, written by `genfmtjas`.
    pub output_wave_numbers: Array1<Real>,
}

/// Inputs for ordinary FEFF GENFMT per-spin momentum setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinarySpinMomentumGridInput<'a> {
    /// FEFF `em(1:ne)`, the complex energy grid.
    pub energies: ArrayView1<'a, Complex>,
    /// FEFF `eref2(1:ne,1:nspx)` as `(energy, spin)`.
    pub spin_reference_energies: ArrayView2<'a, Complex>,
    /// FEFF `edge`, used with `getxk(dble(em(ie))-edge)`.
    pub edge: Real,
    /// FEFF active `nsp` after applying `ispin`.
    pub spin_channel_count: usize,
}

/// Ordinary FEFF GENFMT per-spin momentum arrays for the path loop.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinarySpinMomentumGrid {
    /// FEFF `xk(1:ne)`, shared by all spin channels.
    pub wave_numbers: Array1<Real>,
    /// FEFF `ck(1:ne,1:nsp)` for the ordinary spin loop.
    ///
    /// Rust axes are `(energy, spin)`.
    pub complex_momenta: Array2<Complex>,
    /// Magnitudes of [`Self::complex_momenta`], with the same axes.
    pub complex_momentum_magnitudes: Array2<Real>,
}

/// Inputs for the FEFF GENFMT `feff.bin` header block.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtFeffBinHeaderInput<'a> {
    /// Version suffix written after `#_feff.bin v03:`.
    pub version: &'a str,
    /// FEFF PAD field width, `mpadx`.
    pub pad_width: usize,
    /// FEFF core-hole index, `ihole`.
    pub core_hole: i32,
    /// FEFF GENFMT matrix order, `iorder`.
    pub order: i32,
    /// FEFF `ilinit`, written directly to the header.
    pub initial_angular_momentum: i32,
    /// FEFF average Norman radius, `rnrmav`.
    pub average_norman_radius: Real,
    /// FEFF Fermi level, `xmu`.
    pub fermi_level: Real,
    /// FEFF edge energy.
    pub edge_energy: Real,
    /// FEFF `potlbl(0:npot)` labels. Blank labels fall back to `atsym(iz)`.
    pub potential_labels: &'a [&'a str],
    /// FEFF `iz(0:npot)` atomic numbers.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// FEFF central atom phase shifts, `ph(1:ne,ll,0)`.
    pub central_phase_shifts: ArrayView1<'a, Complex>,
    /// FEFF complex momenta, `ck(1:ne)`.
    pub complex_momenta: ArrayView1<'a, Complex>,
    /// FEFF real momenta, `xk(1:ne)`.
    pub wave_numbers: ArrayView1<'a, Real>,
}

/// Potential label and atomic number written in a GENFMT `feff.bin` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenfmtFeffBinPotential {
    /// FEFF six-character potential label.
    pub label: String,
    /// FEFF `iz`.
    pub atomic_number: usize,
}

/// Prepared FEFF GENFMT `feff.bin` header data.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtFeffBinHeader {
    /// Version suffix written after `#_feff.bin v03:`.
    pub version: String,
    /// FEFF PAD field width, `mpadx`.
    pub pad_width: usize,
    /// FEFF core-hole index, `ihole`.
    pub core_hole: i32,
    /// FEFF GENFMT matrix order, `iorder`.
    pub order: i32,
    /// FEFF `ilinit`, written directly to the header.
    pub initial_angular_momentum: i32,
    /// FEFF average Norman radius, `rnrmav`.
    pub average_norman_radius: Real,
    /// FEFF Fermi level, `xmu`.
    pub fermi_level: Real,
    /// FEFF edge energy.
    pub edge_energy: Real,
    /// Potential table for FEFF indices `0:npot`.
    pub potentials: Vec<GenfmtFeffBinPotential>,
    /// FEFF central atom phase shifts, `ph(1:ne,ll,0)`.
    pub central_phase_shifts: Array1<Complex>,
    /// FEFF complex momenta, `ck(1:ne)`.
    pub complex_momenta: Array1<Complex>,
    /// FEFF real momenta, `xk(1:ne)`.
    pub wave_numbers: Array1<Real>,
}

/// Inputs for the common FEFF GENFMT driver setup before the path loop.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtDriverSetupInput<'a> {
    /// Version suffix written after `#_feff.bin v03:`.
    pub version: &'a str,
    /// FEFF PAD field width, `mpadx`.
    pub pad_width: usize,
    /// FEFF core-hole index, `ihole`.
    pub core_hole: i32,
    /// FEFF GENFMT matrix order, `iorder`.
    pub order: i32,
    /// FEFF average Norman radius, `rnrmav`.
    pub average_norman_radius: Real,
    /// FEFF Fermi level, `xmu`.
    pub fermi_level: Real,
    /// FEFF edge energy.
    pub edge_energy: Real,
    /// FEFF `ispin`; only `ispin == 1` activates both spin channels.
    pub spin_selector: i32,
    /// FEFF `nspx`, the number of spin channels available from XSPH.
    pub available_spin_channels: usize,
    /// FEFF `em(1:ne)`, the complex energy grid.
    pub energies: ArrayView1<'a, Complex>,
    /// FEFF `eref2(1:ne,1:nspx)` as `(energy, spin)`.
    pub spin_reference_energies: ArrayView2<'a, Complex>,
    /// FEFF `ph4(1:ne,-ltot:ltot,1:nspx,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, spin, potential)`.
    pub spin_phase_shifts: ArrayView4<'a, Complex>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// Offset used to map signed angular momentum onto the phase-shift table axis.
    pub signed_angular_offset: usize,
    /// FEFF `linit`, the initial core orbital angular momentum.
    pub initial_orbital_l: usize,
    /// FEFF `kinit`, the relativistic kappa of the initial orbital.
    pub initial_kappa: i32,
    /// FEFF `potlbl(0:npot)` labels. Blank labels fall back to `atsym(iz)`.
    pub potential_labels: &'a [&'a str],
    /// FEFF `iz(0:npot)` atomic numbers.
    pub atomic_numbers: ArrayView1<'a, usize>,
}

/// Common FEFF GENFMT setup products before evaluating paths.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtDriverSetup {
    /// FEFF active `nsp` after applying `ispin`.
    pub spin_channel_count: usize,
    /// FEFF header/setup reference energies.
    pub reference_energies: GenfmtSpinReferenceEnergies,
    /// FEFF header/setup phase shifts.
    pub phase_shifts: GenfmtSpinPhaseShifts,
    /// FEFF central-atom phase shifts written to the `feff.bin` header.
    pub central_phase_shifts: GenfmtCentralPhaseShifts,
    /// FEFF ordinary path-loop `ck(ie,is)` for each active spin channel.
    pub spin_momentum_grid: GenfmtOrdinarySpinMomentumGrid,
    /// FEFF `xk`, `ck`, and `ckmag` arrays for the header and path loops.
    pub momentum_grid: GenfmtMomentumGrid,
    /// Prepared FEFF `feff.bin` header payload.
    pub header: GenfmtFeffBinHeader,
}

/// Inputs for FEFF GENFMTJAS active spin-channel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtJasSpinSelectionInput {
    /// FEFF `ispin`; `ispin == 1` selects the last available spin channel.
    pub spin_selector: i32,
    /// FEFF `nspx`, the number of spin channels available from XSPH.
    pub available_spin_channels: usize,
}

/// FEFF GENFMTJAS selected spin slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtJasSpinSelection {
    /// Zero-based spin index selected by `genfmtjas.f90`.
    pub spin_index: usize,
}

/// Inputs for FEFF GENFMTJAS spin-resolved radial transition factors.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasSpinRadialFactorInput<'a> {
    /// FEFF `rkk2(1:ne,1:nq,1:indmax,1:nspx)`.
    ///
    /// Rust axes are `(energy, q, transition, spin)`.
    pub spin_radial_factors: ArrayView4<'a, Complex>,
    /// Zero-based spin index selected by `genfmtjas.f90`.
    pub spin_index: usize,
}

/// FEFF GENFMTJAS radial transition factors for the selected spin channel.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasSpinRadialFactors {
    /// FEFF `rkk(1:ne,1:nq,1:indmax)`.
    pub radial_factors: Array3<Complex>,
}

/// Inputs for FEFF GENFMTJAS initial-state angular-momentum setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtJasEffectiveInitialJInput {
    /// FEFF `elpty`, the ellipticity ratio.
    pub ellipticity: Real,
    /// FEFF `jinit`, the doubled initial-state angular momentum before `regenf`.
    pub initial_j2: i32,
    /// FEFF `jmax`, the largest doubled final-state angular momentum.
    pub final_j2_max: i32,
}

/// FEFF GENFMTJAS effective initial-state angular momentum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtJasEffectiveInitialJ {
    /// FEFF effective `jinit` after the spherical NRIXS override.
    pub initial_j2: i32,
    /// Whether FEFF promoted `jinit` to `jmax`.
    pub promoted_to_final_j2_max: bool,
}

/// Inputs for the FEFF GENFMTJAS `indmaxt`/`indmax` consistency check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtJasTransitionCountInput {
    /// FEFF `indmaxt`, read from `phase.bin`.
    pub phase_transition_count: usize,
    /// FEFF `indmax`, requested by NRIXS input.
    pub requested_transition_count: usize,
}

/// Checked FEFF GENFMTJAS transition count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtJasTransitionCount {
    /// Shared transition count after confirming `indmaxt == indmax`.
    pub transition_count: usize,
}

/// Inputs for the FEFF GENFMTJAS driver setup before the path loop.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasDriverSetupInput<'a> {
    /// Version suffix written after `#_feff.bin v03:`.
    pub version: &'a str,
    /// FEFF PAD field width, `mpadx`.
    pub pad_width: usize,
    /// FEFF core-hole index, `ihole`.
    pub core_hole: i32,
    /// FEFF GENFMT matrix order, `iorder`.
    pub order: i32,
    /// FEFF average Norman radius, `rnrmav`.
    pub average_norman_radius: Real,
    /// FEFF Fermi level, `xmu`.
    pub fermi_level: Real,
    /// FEFF edge energy.
    pub edge_energy: Real,
    /// FEFF `ispin`; `ispin == 1` selects the last available spin channel.
    pub spin_selector: i32,
    /// FEFF `nspx`, the number of spin channels available from XSPH.
    pub available_spin_channels: usize,
    /// FEFF `em(1:ne)`, the complex energy grid.
    pub energies: ArrayView1<'a, Complex>,
    /// FEFF `eref2(1:ne,1:nspx)` as `(energy, spin)`.
    pub spin_reference_energies: ArrayView2<'a, Complex>,
    /// FEFF `ph4(1:ne,-ltot:ltot,1:nspx,0:npot)`.
    ///
    /// Rust axes are `(energy, signed_l + signed_angular_offset, spin, potential)`.
    pub spin_phase_shifts: ArrayView4<'a, Complex>,
    /// FEFF `rkk2(1:ne,1:nq,1:indmax,1:nspx)`.
    ///
    /// Rust axes are `(energy, q, transition, spin)`.
    pub spin_radial_factors: ArrayView4<'a, Complex>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// Offset used to map signed angular momentum onto the phase-shift table axis.
    pub signed_angular_offset: usize,
    /// FEFF `linit`, the initial core orbital angular momentum.
    pub initial_orbital_l: usize,
    /// FEFF `kinit`, the relativistic kappa of the initial orbital.
    pub initial_kappa: i32,
    /// FEFF `potlbl(0:npot)` labels. Blank labels fall back to `atsym(iz)`.
    pub potential_labels: &'a [&'a str],
    /// FEFF `iz(0:npot)` atomic numbers.
    pub atomic_numbers: ArrayView1<'a, usize>,
}

/// FEFF GENFMTJAS setup products before evaluating paths.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasDriverSetup {
    /// FEFF spin slot selected by the JAS driver.
    pub spin_selection: GenfmtJasSpinSelection,
    /// FEFF JAS reference energies for the selected spin channel.
    pub reference_energies: GenfmtSpinReferenceEnergies,
    /// FEFF JAS phase shifts for the selected spin channel.
    pub phase_shifts: GenfmtSpinPhaseShifts,
    /// FEFF JAS radial transition factors for the selected spin channel.
    pub radial_factors: GenfmtJasSpinRadialFactors,
    /// FEFF central-atom phase shifts written to the `feff.bin` header.
    pub central_phase_shifts: GenfmtCentralPhaseShifts,
    /// FEFF `xk`, `ck`, and `ckmag` arrays for the header and path loops.
    pub momentum_grid: GenfmtMomentumGrid,
    /// Prepared FEFF `feff.bin` header payload.
    pub header: GenfmtFeffBinHeader,
}

/// Inputs for FEFF GENFMT per-leg curved-wave polynomial limits.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtCurvedWaveLegLimitsInput<'a> {
    /// FEFF `ipot(1:nleg)`, the potential index for each path leg.
    ///
    /// The final entry should be the absorber potential. FEFF also mirrors it
    /// into `ipot(0)`; Rust derives that wraparound from this slice.
    pub path_potential_indices: ArrayView1<'a, usize>,
    /// FEFF `lmax(1:ne,0:npot)` inclusive angular limits.
    pub angular_limits: ArrayView2<'a, usize>,
    /// Zero-based Rust energy index corresponding to FEFF `ie`.
    pub energy_index: usize,
    /// FEFF `mmaxp1`, equal to `mmax + 1`.
    pub max_m_plus_one: usize,
    /// FEFF `nmax`, the current Rehr-Albers order limit.
    pub max_n: usize,
}

/// FEFF `sclmz` limit selection for one path leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtCurvedWaveLegLimit {
    /// FEFF `ipot(isc0)`, where leg 1 wraps to `ipot(nleg)`.
    pub previous_potential_index: usize,
    /// FEFF `ipot(isc1)`, the current leg potential index.
    pub current_potential_index: usize,
    /// FEFF `lxp1=max(lmax(ie,ipot(isc0))+1,lmax(ie,ipot(isc1))+1)`.
    pub angular_count: usize,
    /// FEFF `mnp1=min(lxp1,mmaxp1+nmax)`.
    pub mixed_order_count: usize,
}

/// FEFF curved-wave polynomial limits for all path legs at one energy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenfmtCurvedWaveLegLimits {
    /// FEFF `mnmxp1=mmaxp1+nmax`.
    pub mixed_order_capacity: usize,
    /// Per-leg `sclmz` limits in FEFF leg order.
    pub limits: Vec<GenfmtCurvedWaveLegLimit>,
}

/// Inputs for FEFF GENFMT all-leg curved-wave polynomial preparation.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtCurvedWavePolynomialTablesInput<'a> {
    /// FEFF `rho(1:nleg)=ck(ie)*ri(1:nleg)` for one energy.
    pub leg_rhos: ArrayView1<'a, Complex>,
    /// Per-leg `sclmz` limits, in the same order as [`Self::leg_rhos`].
    pub leg_limits: &'a [GenfmtCurvedWaveLegLimit],
    /// FEFF `mnmxp1=mmaxp1+nmax`, the global `clmi` mixed-order capacity.
    pub mixed_order_capacity: usize,
}

/// FEFF `clmi(il,im,ileg)` curved-wave polynomial table for one energy.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtCurvedWavePolynomialTables {
    /// Zero-filled `clmi`-style table with Rust axes `(l, m, leg)`.
    pub tables: Array3<Complex>,
}

/// Inputs for FEFF GENFMT per-energy path geometry setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtPathGeometryInput<'a> {
    /// FEFF `ri(1:nleg)` leg lengths.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF `ck(ie)`, the complex momentum for one energy.
    pub complex_momentum: Complex,
    /// FEFF `eps`, the zero-momentum threshold.
    pub momentum_zero_epsilon: Real,
}

/// FEFF GENFMT path geometry values used in the energy loop.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtPathGeometry {
    /// FEFF `rho(ileg)=ck(ie)*ri(ileg)`.
    pub leg_rhos: Array1<Complex>,
    /// FEFF `reff=sum(ri)/2`.
    pub effective_path_length: Real,
    /// Whether FEFF continues the scattering calculation for this energy.
    ///
    /// FEFF skips the calculation when `abs(ck(ie)) <= eps`.
    pub active: bool,
}

/// Inputs for FEFF GENFMT path-signal accumulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtPathSignalContributionInput {
    /// Current FEFF `cchi(ie)` before this spin/path contribution is added.
    pub accumulated_chi: Complex,
    /// FEFF `ptrac` for one energy.
    pub path_trace: Complex,
    /// FEFF curved-wave path factor `cfac`.
    pub path_factor: Complex,
    /// FEFF `nsp`; supported values are one or two spin channels.
    pub spin_channel_count: usize,
    /// Zero-based Rust spin index. FEFF `is=1` maps to `0`.
    pub spin_index: usize,
}

/// FEFF GENFMT path-signal contribution for one energy and spin channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtPathSignalContribution {
    /// FEFF contribution added to `cchi(ie)` after any spin sign is applied.
    pub contribution: Complex,
    /// Updated FEFF `cchi(ie)`.
    pub accumulated_chi: Complex,
}

/// Inputs for the ordinary FEFF GENFMT spin/path signal energy loop.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtPathSignalsInput<'a> {
    /// FEFF `ptrac` values for each active spin and energy.
    ///
    /// Rust axes are `(spin, energy)`.
    pub path_traces: ArrayView2<'a, Complex>,
    /// FEFF `cfac(1:ne)` curved-wave path factors.
    pub path_factors: ArrayView1<'a, Complex>,
    /// Whether the FEFF energy point reached the signal calculation.
    ///
    /// Energies where `abs(ck(ie)) <= eps` keep zero signal, matching the
    /// `goto 4990` branch in `genfmtsub.f90`.
    pub active: ArrayView1<'a, bool>,
    /// FEFF `nsp`; supported values are one or two spin channels.
    pub spin_channel_count: usize,
}

/// Ordinary FEFF GENFMT path signals over the energy grid.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtPathSignals {
    /// Per-spin FEFF contributions after the two-spin sign rule.
    ///
    /// Rust axes are `(spin, energy)`.
    pub contributions: Array2<Complex>,
    /// FEFF `cchi(1:ne)` after summing spin contributions.
    pub chi: Array1<Complex>,
}

/// Inputs for ordinary FEFF GENFMT path finalization after trace evaluation.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathFinalizationInput<'a> {
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `ptrac(is,ie)` values for each active spin and energy.
    pub path_traces: ArrayView2<'a, Complex>,
    /// FEFF `cfac(1:ne)` curved-wave path factors.
    pub path_factors: ArrayView1<'a, Complex>,
    /// Whether the FEFF energy point reached the signal calculation.
    pub active: ArrayView1<'a, bool>,
    /// FEFF `nsp`; supported values are one or two spin channels.
    pub spin_channel_count: usize,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `reff`, the effective half path length in bohr.
    pub effective_half_path_length_bohr: Real,
    /// FEFF `ipot(1:nleg)`.
    pub potential_indices: ArrayView1<'a, usize>,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF `beta(1:nleg)`.
    pub beta_angles: ArrayView1<'a, Real>,
    /// FEFF `eta(1:nleg)` values written to `feff.bin`.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// Ordinary FEFF GENFMT path finalization result.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinaryPathFinalization {
    /// FEFF spin-summed `cchi` and per-spin contributions.
    pub signals: GenfmtPathSignals,
    /// FEFF path importance, retention, and retained output payload.
    pub output_decision: GenfmtPathOutputDecision,
}

/// Inputs for finalizing an ordinary FEFF GENFMT path from a full energy grid.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathEnergyGridFinalizationInput<'a> {
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// Ordinary GENFMT spin and energy loop result.
    pub energy_grid: &'a GenfmtOrdinaryPathEnergyGrid,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `reff`, the effective half path length in bohr.
    pub effective_half_path_length_bohr: Real,
    /// FEFF `ipot(1:nleg)`.
    pub potential_indices: ArrayView1<'a, usize>,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF `beta(1:nleg)`.
    pub beta_angles: ArrayView1<'a, Real>,
    /// FEFF `eta(1:nleg)` values written to `feff.bin`.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// Inputs for collecting ordinary FEFF GENFMT finalized path outputs.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryPathOutputsInput<'a> {
    /// Finalized paths in FEFF `paths.dat` traversal order.
    pub path_finalizations: &'a [GenfmtOrdinaryPathFinalization],
}

/// Retained ordinary FEFF GENFMT path outputs in driver order.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinaryPathOutputs {
    /// Number of paths examined by the driver loop.
    pub examined_path_count: usize,
    /// Number of paths retained for output.
    pub retained_path_count: usize,
    /// FEFF `xportx` after the final examined path, if any path was examined.
    pub final_normalization: Option<Real>,
    /// Per-path summaries in FEFF `paths.dat` traversal order.
    pub path_summaries: Vec<GenfmtPathOutputSummary>,
    /// Retained path payloads in FEFF output order.
    pub retained_paths: Vec<GenfmtRetainedPathOutput>,
}

/// Inputs for the FEFF GENFMTJAS total and decomposed path-signal update.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathSignalInput<'a> {
    /// FEFF `ptrac` for one energy.
    pub path_trace: Complex,
    /// FEFF curved-wave path factor `cfac`.
    pub path_factor: Complex,
    /// Optional FEFF `ptrac` table for `pgtrl(lg2,lg1,ie)`.
    ///
    /// Rust axes are `(decomposition_row, decomposition_column)`.
    pub decomposed_traces: Option<ArrayView2<'a, Complex>>,
}

/// FEFF GENFMTJAS total and decomposed path signal for one energy.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasPathSignal {
    /// FEFF `cchi(ie)=ptrac*cfac`.
    pub chi: Complex,
    /// Optional FEFF `pgtrl(lg2,lg1,ie)=ptrac(lg2,lg1)*cfac`.
    pub decomposed_chi: Option<Array2<Complex>>,
    /// FEFF `lgcchi`, the sum of the decomposed `pgtrl` entries.
    pub decomposed_sum: Option<Complex>,
}

/// Inputs for the FEFF GENFMTJAS path signal energy loop.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathSignalsInput<'a> {
    /// FEFF `ptrac(1:ne)` values for a path after trace contraction.
    pub path_traces: ArrayView1<'a, Complex>,
    /// FEFF `cfac(1:ne)` curved-wave path factors.
    pub path_factors: ArrayView1<'a, Complex>,
    /// Whether the FEFF energy point reached the signal calculation.
    ///
    /// Energies where `abs(ck(ie)) <= eps` stay zero, matching the
    /// `goto 4990` branch in `genfmtjas.f90`.
    pub active: ArrayView1<'a, bool>,
    /// Optional FEFF decomposition traces before multiplying by `cfac`.
    ///
    /// Rust axes are `(decomposition_row, decomposition_column, energy)`.
    pub decomposed_traces: Option<ArrayView3<'a, Complex>>,
}

/// FEFF GENFMTJAS path signals over the energy grid.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasPathSignals {
    /// FEFF `cchi(1:ne)`.
    pub chi: Array1<Complex>,
    /// Optional FEFF `pgtrl(lg2,lg1,1:ne)`.
    pub decomposed_chi: Option<Array3<Complex>>,
    /// Optional FEFF `lgcchi(1:ne)` sums.
    pub decomposed_sums: Option<Array1<Complex>>,
}

/// Inputs for FEFF GENFMTJAS path finalization after trace evaluation.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathFinalizationInput<'a> {
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `ptrac(1:ne)` values for a path after trace contraction.
    pub path_traces: ArrayView1<'a, Complex>,
    /// FEFF `cfac(1:ne)` curved-wave path factors.
    pub path_factors: ArrayView1<'a, Complex>,
    /// Whether the FEFF energy point reached the signal calculation.
    pub active: ArrayView1<'a, bool>,
    /// Optional FEFF decomposition traces before multiplying by `cfac`.
    pub decomposed_traces: Option<ArrayView3<'a, Complex>>,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `reff`, the effective half path length in bohr.
    pub effective_half_path_length_bohr: Real,
    /// FEFF `ipot(1:nleg)`.
    pub potential_indices: ArrayView1<'a, usize>,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF `beta(1:nleg)`.
    pub beta_angles: ArrayView1<'a, Real>,
    /// FEFF `eta(1:nleg)` values written to `feff.bin`.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// FEFF GENFMTJAS path finalization result.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasPathFinalization {
    /// FEFF total and optional decomposed path signals.
    pub signals: GenfmtJasPathSignals,
    /// FEFF path importance, retention, and retained output payload.
    pub output_decision: GenfmtPathOutputDecision,
    /// Optional FEFF `feffl.bin` amplitude/phase payload for retained paths.
    pub decomposed_output: Option<GenfmtDecomposedChiAmplitudePhase>,
}

/// Inputs for finalizing a FEFF GENFMTJAS path from a full energy grid.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathEnergyGridFinalizationInput<'a> {
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// GENFMTJAS energy loop result.
    pub energy_grid: &'a GenfmtJasPathEnergyGrid,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `reff`, the effective half path length in bohr.
    pub effective_half_path_length_bohr: Real,
    /// FEFF `ipot(1:nleg)`.
    pub potential_indices: ArrayView1<'a, usize>,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF `beta(1:nleg)`.
    pub beta_angles: ArrayView1<'a, Real>,
    /// FEFF `eta(1:nleg)` values written to `feff.bin`.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// Inputs for collecting FEFF GENFMTJAS finalized path outputs.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasPathOutputsInput<'a> {
    /// Finalized paths in FEFF `paths.dat` traversal order.
    pub path_finalizations: &'a [GenfmtJasPathFinalization],
}

/// Retained FEFF GENFMTJAS path outputs in driver order.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasPathOutputs {
    /// Number of paths examined by the driver loop.
    pub examined_path_count: usize,
    /// Number of paths retained for output.
    pub retained_path_count: usize,
    /// FEFF `xportx` after the final examined path, if any path was examined.
    pub final_normalization: Option<Real>,
    /// Per-path summaries in FEFF `paths.dat` traversal order.
    pub path_summaries: Vec<GenfmtPathOutputSummary>,
    /// Retained path payloads in FEFF output order.
    pub retained_paths: Vec<GenfmtRetainedPathOutput>,
    /// Retained `feffl.bin` payloads, when JAS decomposition is enabled.
    pub decomposed_paths: Option<Vec<GenfmtDecomposedChiAmplitudePhase>>,
}

/// Inputs for the FEFF GENFMT curved-wave propagation factor.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtCurvedWavePathFactorInput<'a> {
    /// FEFF `rho(1:nleg)=ck(ie)*ri(1:nleg)`.
    pub leg_rhos: ArrayView1<'a, Complex>,
    /// FEFF real wave number `xk(ie)`.
    pub wave_number: Real,
    /// FEFF `reff`, half the total path length.
    pub effective_path_length: Real,
}

/// FEFF GENFMT curved-wave propagation factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtCurvedWavePathFactor {
    /// FEFF `srho=sum(rho)`.
    pub rho_sum: Complex,
    /// FEFF `prho=product(rho)`.
    pub rho_product: Complex,
    /// FEFF `cfac=exp(i*(srho-2*xk*reff))/prho`.
    pub factor: Complex,
}

/// Inputs for the FEFF GENFMT path importance calculation.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtPathImportanceInput<'a> {
    /// FEFF `cchi(1:ne)`, the complex path contribution after the trace and
    /// curved-wave factor have been applied.
    pub chi: ArrayView1<'a, Complex>,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`, the path degeneracy.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current normalization. Values `<= 0` initialize the
    /// normalization from this path's raw importance, matching FEFF.
    pub current_normalization: Real,
}

/// FEFF GENFMT path importance values.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtPathImportance {
    /// FEFF `ffmag(1:ne1)`.
    pub magnitudes: Array1<Real>,
    /// FEFF `xport=abs(deg*trap(...))`.
    pub raw_importance: Real,
    /// Updated FEFF `xportx`.
    pub normalization: Real,
    /// FEFF `crit=100*xport/xportx`.
    pub percent: Real,
}

/// Inputs for the FEFF GENFMT path-output retention decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtPathRetentionInput {
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `crit`, the path importance percentage.
    pub path_importance_percent: Real,
}

/// FEFF GENFMT decision for writing or discarding one path output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtPathRetention {
    /// FEFF `crit0=2*critcw/3` when `ipr3 <= 0`; unused when output is forced.
    pub discard_threshold_percent: Option<Real>,
    /// Whether FEFF writes this path to `feff.bin`/`feffl.bin`.
    pub keep: bool,
}

/// FEFF GENFMT per-path output summary for driver logs and counters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtPathOutputSummary {
    /// FEFF `ipath`.
    pub path_index: usize,
    /// Whether FEFF retained this path for output.
    pub retained: bool,
    /// FEFF `crit`, the curved-wave amplitude ratio in percent.
    pub criterion_percent: Real,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `nleg`.
    pub leg_count: usize,
    /// FEFF `reff`, the effective half path length in bohr.
    pub effective_half_path_length_bohr: Real,
    /// FEFF `reff*bohr`, printed in Angstrom.
    pub effective_half_path_length_angstrom: Real,
}

/// Inputs for the FEFF GENFMT post-energy path output decision.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtPathOutputDecisionInput<'a> {
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `ipr3`/`ipr5` print level.
    pub print_level: i32,
    /// FEFF `critcw`, the requested curved-wave chi amplitude ratio in percent.
    pub curved_wave_criterion_percent: Real,
    /// FEFF `cchi(1:ne)`.
    pub chi: ArrayView1<'a, Complex>,
    /// FEFF `ckmag(1:ne)`, the momentum magnitudes used as integration points.
    pub momentum_magnitudes: ArrayView1<'a, Real>,
    /// FEFF `ik0`, represented as a zero-based Rust index.
    pub edge_start_index: usize,
    /// FEFF `ne1`, the number of energy points active in the importance integral.
    pub active_energy_count: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `xportx`, the current path-importance normalization.
    pub current_normalization: Real,
    /// FEFF `reff`, the effective half path length in bohr.
    pub effective_half_path_length_bohr: Real,
    /// FEFF `ipot(1:nleg)`.
    pub potential_indices: ArrayView1<'a, usize>,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF `beta(1:nleg)`.
    pub beta_angles: ArrayView1<'a, Real>,
    /// FEFF `eta(1:nleg)` values written to `feff.bin`.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// FEFF GENFMT post-energy path output decision.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtPathOutputDecision {
    /// FEFF per-path summary retained even when payload output is discarded.
    pub summary: GenfmtPathOutputSummary,
    /// FEFF `ffmag`, `xport`, `xportx`, and `crit` values for the path.
    pub importance: GenfmtPathImportance,
    /// FEFF keep/discard branch selected from `ipr3`, `critcw`, and `crit`.
    pub retention: GenfmtPathRetention,
    /// Data written for retained paths, or `None` when FEFF discards the path.
    pub retained_output: Option<GenfmtRetainedPathOutput>,
}

/// Inputs for the FEFF GENFMT `feff.bin` amplitude/phase table.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtChiAmplitudePhaseInput<'a> {
    /// FEFF `cchi(1:ne)`.
    pub chi: ArrayView1<'a, Complex>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// FEFF GENFMT path amplitude and unwrapped phase arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtChiAmplitudePhase {
    /// FEFF `amff(1:ne)=abs(cchi)`.
    pub amplitudes: Array1<Real>,
    /// FEFF `phff(1:ne)` after `pijump`.
    pub phases: Array1<Real>,
}

/// Inputs for a retained FEFF GENFMT path output block.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtRetainedPathOutputInput<'a> {
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `crit`.
    pub criterion_percent: Real,
    /// FEFF `reff`, the effective half path length in bohr.
    pub effective_half_path_length_bohr: Real,
    /// FEFF `ipot(1:nleg)`.
    pub potential_indices: ArrayView1<'a, usize>,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: ArrayView2<'a, Real>,
    /// FEFF `beta(1:nleg)`.
    pub beta_angles: ArrayView1<'a, Real>,
    /// FEFF `eta(1:nleg)` values written to `feff.bin`.
    pub eta_angles: ArrayView1<'a, Real>,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: ArrayView1<'a, Real>,
    /// FEFF `cchi(1:ne)`.
    pub chi: ArrayView1<'a, Complex>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// Prepared data for a retained FEFF GENFMT path output block.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtRetainedPathOutput {
    /// FEFF `ipath`.
    pub path_index: usize,
    /// FEFF `deg`.
    pub degeneracy: Real,
    /// FEFF `crit`.
    pub criterion_percent: Real,
    /// FEFF `reff`, the effective half path length in bohr.
    pub effective_half_path_length_bohr: Real,
    /// FEFF `reff*bohr`, written to `list.dat` and `feff.bin` headers.
    pub effective_half_path_length_angstrom: Real,
    /// FEFF list.dat Debye-Waller column, hard-coded as zero in GENFMT.
    pub list_sigma2: Real,
    /// FEFF `ipot(1:nleg)`.
    pub potential_indices: Array1<usize>,
    /// FEFF `rat(1:3,1:nleg)` as Rust `(leg, xyz)` in bohr.
    pub positions: Array2<Real>,
    /// FEFF `beta(1:nleg)`.
    pub beta_angles: Array1<Real>,
    /// FEFF `eta(1:nleg)`.
    pub eta_angles: Array1<Real>,
    /// FEFF `ri(1:nleg)` leg distances in bohr.
    pub leg_lengths: Array1<Real>,
    /// FEFF `amff(1:ne)`.
    pub amplitudes: Array1<Real>,
    /// FEFF `phff(1:ne)`.
    pub phases: Array1<Real>,
}

/// Inputs for FEFF GENFMTJAS decomposition amplitude/phase output.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtDecomposedChiAmplitudePhaseInput<'a> {
    /// FEFF `pgtrl(0:ldecmx,0:ldecmx,1:ne)`.
    ///
    /// Rust axes are `(decomposition_row, decomposition_column, energy)`.
    pub decomposed_chi: ArrayView3<'a, Complex>,
    /// FEFF small-value threshold used before `atan2`.
    pub phase_epsilon: Real,
}

/// FEFF GENFMTJAS decomposition amplitude and unwrapped phase tables.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtDecomposedChiAmplitudePhase {
    /// FEFF `amff(1:ne)` generated for each decomposition channel.
    pub amplitudes: Array3<Real>,
    /// FEFF `phff(1:ne)` generated for each decomposition channel.
    pub phases: Array3<Real>,
}

/// Inputs for FEFF `GENFMT/sclmz.f90` curved-wave polynomial tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvedWavePolynomialInput {
    /// FEFF `lmaxp1`, equal to `lmax + 1`.
    pub lmaxp1: usize,
    /// FEFF `mmaxp1`; columns above `lmaxp1` are retained as zeroes.
    pub mmaxp1: usize,
    /// FEFF complex path length `rho(ileg)`.
    pub rho: Complex,
}

/// Inputs for FEFF `GENFMT/snlm.f90` Legendre-normalization tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenfmtLegendreNormalizationInput {
    /// FEFF `lmaxp1`, equal to `lmax + 1`.
    pub lmaxp1: usize,
    /// FEFF `mmaxp1`, equal to `mmax + 1`.
    pub mmaxp1: usize,
}

/// Inputs for FEFF `GENFMT/fmtrxi.f90` scattering-amplitude F matrices.
#[derive(Debug, Clone, Copy)]
pub struct ScatteringAmplitudeMatrixInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lam1x`, the active row lambda count.
    pub left_lambda_count: usize,
    /// FEFF `lam2x`, the active column lambda count.
    pub right_lambda_count: usize,
    /// FEFF signed phase vector for one energy and potential.
    ///
    /// The vector length must be odd. Rust index `phase_offset + l` stores
    /// FEFF `ph(ie,l,ipot)`, and `phase_offset - l` stores `ph(ie,-l,ipot)`.
    pub phase_shifts: ArrayView1<'a, Complex>,
    /// FEFF `lmax(ie,ipot)`, inclusive.
    pub angular_limit: usize,
    /// FEFF `clmi(:,:,ileg)` table for the first leg.
    pub first_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF `clmi(:,:,ilegp)` table for the following leg.
    pub second_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF `dri(:,:,:,ilegp)` rotation matrix.
    ///
    /// Rust indices are `(l, m1 + rotation_magnetic_offset,
    /// m2 + rotation_magnetic_offset)`.
    pub rotation: ArrayView3<'a, Real>,
    /// Magnetic-index offset for the second and third axes of `rotation`.
    pub rotation_magnetic_offset: usize,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(ileg)` phase factor.
    pub eta: Real,
}

/// Inputs for FEFF `GENFMT/mmtrxi.f90` polarized scattering-amplitude matrices.
#[derive(Debug, Clone, Copy)]
pub struct PolarizedScatteringAmplitudeInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lam1x`, the active square lambda dimension.
    pub lambda_count: usize,
    /// FEFF transition angular momenta `lind(1:8)`.
    ///
    /// Negative entries are ignored, matching FEFF transition slots that are
    /// not active for the selected edge and polarization.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF radial transition factors `rkk(ie,1:8)` for one energy.
    pub radial_factors: ArrayView1<'a, Complex>,
    /// FEFF `bmati(-mtot:mtot,1:8,-mtot:mtot,1:8)`.
    ///
    /// Rust indices are `(m1 + transition_magnetic_offset, k1,
    /// m2 + transition_magnetic_offset, k2)`.
    pub transition_matrix: ArrayView4<'a, Complex>,
    /// Magnetic-index offset for the first and third axes of `transition_matrix`.
    pub transition_magnetic_offset: usize,
    /// FEFF `clmi(:,:,ileg)` table for the first leg.
    pub first_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF `clmi(:,:,ilegp)` table for the following leg.
    pub second_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(ileg)` phase factor.
    pub eta: Real,
}

/// Inputs for FEFF `GENFMT/mmtrxijas0.f90` JAS/NRIXS amplitude folding.
#[derive(Debug, Clone, Copy)]
pub struct JasScatteringAmplitudeInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lam1x`, the active square lambda dimension.
    pub lambda_count: usize,
    /// FEFF `lind(1:indmax)` final-state orbital angular momenta.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `rkk(ie,1:nq,1:indmax)` for one energy point.
    ///
    /// Rust indices are `(q, final_state)`.
    pub radial_factors: ArrayView2<'a, Complex>,
    /// FEFF complex q-vector weights `qw(1:nq)`.
    pub q_weights: ArrayView1<'a, Complex>,
    /// FEFF `hbmatrs(mj,is2,mu2,mu1,k1)`.
    ///
    /// The first axis uses compact doubled-`j` rows, where
    /// `mj = -initial_j2 + 2 * row`. The magnetic axes are shifted by
    /// [`Self::transition_magnetic_offset`].
    pub transition_matrix: ArrayView5<'a, Complex>,
    /// FEFF `jinit`, doubled initial-state angular momentum.
    pub initial_j2: i32,
    /// Magnetic-index offset for the third and fourth axes of
    /// [`Self::transition_matrix`].
    pub transition_magnetic_offset: usize,
    /// FEFF `clmi(:,:,ileg)` table for the first leg.
    pub first_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF `clmi(:,:,ilegp)` table for the following leg.
    pub second_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(ileg)` phase factor applied to the decomposed amplitudes.
    pub eta: Real,
    /// FEFF `l2lp`, the inclusive angular limit for this leg pair.
    pub max_angular_momentum: usize,
    /// FEFF `ldecmx`; `None` disables the `lgfmats` angular-decomposition table.
    pub decomposition_l_max: Option<usize>,
}

/// FEFF `GENFMT/mmtrxijas0.f90` JAS/NRIXS amplitude matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct JasScatteringAmplitudeMatrices {
    /// FEFF `fmats(mj,is2,lam2,lam1)` with compact doubled-`j` rows.
    pub amplitudes: Array4<Complex>,
    /// FEFF `lgfmats(mj,is2,ll,lam2,lam1)` when angular decomposition is requested.
    pub decomposed_amplitudes: Option<Array5<Complex>>,
}

/// Inputs for FEFF `GENFMT/mmtrxijas.f90` left/right JAS amplitude folding.
#[derive(Debug, Clone, Copy)]
pub struct JasLeftRightAmplitudeInput<'a> {
    /// FEFF `mlam(1:lamx)` magnetic lambda indices.
    pub m_indices: ArrayView1<'a, i32>,
    /// FEFF `nlam(1:lamx)` order lambda indices.
    pub n_indices: ArrayView1<'a, i32>,
    /// FEFF `lam1x`, the active lambda dimension.
    pub lambda_count: usize,
    /// FEFF `lind(1:indmax)` final-state orbital angular momenta.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `rkk(ie,1:nq,1:indmax)` for one energy point.
    ///
    /// Rust indices are `(q, final_state)`.
    pub radial_factors: ArrayView2<'a, Complex>,
    /// FEFF complex q-vector weights `qw(1:nq)`.
    pub q_weights: ArrayView1<'a, Complex>,
    /// FEFF `hbmatl(mj,mu,iq,k1)`.
    ///
    /// The first axis uses compact doubled-`j` rows, where
    /// `mj = -initial_j2 + 2 * row`; the magnetic axis is shifted by
    /// [`Self::transition_magnetic_offset`].
    pub left_transition_matrix: ArrayView4<'a, Complex>,
    /// FEFF `hbmatr(mj,mu,iq,k1)`, with the same axis layout as
    /// [`Self::left_transition_matrix`].
    pub right_transition_matrix: ArrayView4<'a, Complex>,
    /// FEFF `jinit`, doubled initial-state angular momentum.
    pub initial_j2: i32,
    /// Magnetic-index offset for the second axis of the transition matrices.
    pub transition_magnetic_offset: usize,
    /// FEFF `clmi(:,:,ileg)` table for the first leg.
    pub first_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF `clmi(:,:,ilegp)` table for the following leg.
    pub second_leg_polynomials: ArrayView2<'a, Complex>,
    /// FEFF associated-Legendre normalization table, indexed as `(m, l)`.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `eta(ileg)` phase factor applied to the left amplitudes.
    pub eta: Real,
    /// FEFF `l2lp`, the inclusive angular limit for this leg pair.
    pub max_angular_momentum: usize,
    /// FEFF `ldecmx`; `None` disables the `lgfmatl`/`lgfmatr` tables.
    pub decomposition_l_max: Option<usize>,
}

/// FEFF `GENFMT/mmtrxijas.f90` left/right JAS amplitude matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct JasLeftRightAmplitudeMatrices {
    /// FEFF `fmatl(mj,iq,lam)` with compact doubled-`j` rows.
    pub left_amplitudes: Array3<Complex>,
    /// FEFF `fmatr(mj,iq,lam)` with compact doubled-`j` rows.
    pub right_amplitudes: Array3<Complex>,
    /// FEFF `lgfmatl(mj,iq,ll,lam)` when angular decomposition is requested.
    pub decomposed_left_amplitudes: Option<Array4<Complex>>,
    /// FEFF `lgfmatr(mj,iq,ll,lam)` when angular decomposition is requested.
    pub decomposed_right_amplitudes: Option<Array4<Complex>>,
}

/// Inputs for FEFF `GENFMT/mmtrjas0.f90` spherical JAS/NRIXS transition tensor.
#[derive(Debug, Clone, Copy)]
pub struct JasSpinTransitionInput<'a> {
    /// FEFF `kinit`, the relativistic kappa of the initial orbital.
    pub initial_kappa: i32,
    /// FEFF `jinit`, the doubled angular-momentum row limit for `hbmatrs`.
    ///
    /// In spherically averaged NRIXS, FEFF may promote this value to `jmax`
    /// before calling `mmtrjas0`.
    pub initial_j2: i32,
    /// FEFF `nsp`; valid values are one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `lind(1:indmax)` final-state orbital angular momenta.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `ljmax`, the largest spherical transition multipole.
    pub final_lj_max: usize,
    /// FEFF `jmax`, the largest doubled final-state angular momentum.
    pub final_j2_max: i32,
    /// FEFF `lx1`, the largest orbital angular momentum retained in the path.
    pub max_angular_momentum: usize,
    /// FEFF `dri(:,:,:,nsc+2)`, rotating from the polarization axis to the first leg.
    pub first_rotation: ArrayView3<'a, Real>,
    /// FEFF `dri(:,:,:,nleg)`, rotating from the last leg to the polarization axis.
    pub last_rotation: ArrayView3<'a, Real>,
    /// Magnetic-index offset for the second and third axes of the rotation tables.
    pub rotation_magnetic_offset: usize,
    /// FEFF `eta(0)`, the first-leg azimuthal phase.
    pub first_eta: Real,
    /// FEFF `eta(nsc+2)`, the last-leg azimuthal phase.
    pub last_eta: Real,
}

/// FEFF `GENFMT/mmtrjas0.f90` spherical JAS/NRIXS transition tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct JasSpinTransitionMatrix {
    /// FEFF `hbmatrs(mj,is2,mu2,mu1,k1)` with compact doubled-`j` rows.
    pub matrix: Array5<Complex>,
    /// FEFF-generated `jind(1:indmax)` doubled final-state angular momenta.
    pub generated_final_j2: Vec<i32>,
}

/// Inputs for FEFF `GENFMT/mmtrjas.f90` one-sided JAS/NRIXS transition tensors.
#[derive(Debug, Clone, Copy)]
pub struct JasOneSidedTransitionInput<'a> {
    /// FEFF `kinit`, the relativistic kappa of the initial orbital.
    pub initial_kappa: i32,
    /// FEFF `jinit`, the doubled angular-momentum row limit for `hbmatl`/`hbmatr`.
    pub initial_j2: i32,
    /// FEFF `nsp`; valid values are one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `lind(1:indmax)` final-state orbital angular momenta.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `lgind(1:indmax)` final-state orbital angular momenta for `bcoefjas`.
    pub final_lg_momenta: ArrayView1<'a, i32>,
    /// FEFF `ljind(1:indmax)` spherical transition multipoles.
    pub final_lj_momenta: ArrayView1<'a, i32>,
    /// FEFF `ljmax`, the largest spherical transition multipole.
    pub final_lj_max: usize,
    /// FEFF `jmax`, the largest doubled final-state angular momentum.
    pub final_j2_max: i32,
    /// FEFF `lx`, the largest orbital angular momentum retained in `bcoefjas`.
    pub max_angular_momentum: usize,
    /// FEFF `pha(1:nq)` q-vector azimuthal phases.
    pub q_phases: ArrayView1<'a, Complex>,
    /// FEFF `qbeta(1:nq)` q-vector polar rotation angles.
    pub q_beta_angles: ArrayView1<'a, Real>,
    /// FEFF `dri(:,:,:,nsc+2)`, rotating from the polarization axis to the first leg.
    pub first_rotation: ArrayView3<'a, Real>,
    /// FEFF `dri(:,:,:,nleg)`, rotating from the last leg to the polarization axis.
    pub last_rotation: ArrayView3<'a, Real>,
    /// Magnetic-index offset for rotation and output magnetic axes.
    pub rotation_magnetic_offset: usize,
    /// FEFF `eta(0)`, the first-leg azimuthal phase.
    pub first_eta: Real,
    /// FEFF `eta(nsc+2)`, the last-leg azimuthal phase.
    pub last_eta: Real,
}

/// FEFF `GENFMT/mmtrjas.f90` one-sided JAS/NRIXS transition tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct JasOneSidedTransitionMatrices {
    /// FEFF `hbmatl(mj,iq,mu,k1)` stored as `(mj, mu, q, k1)`.
    pub left_matrix: Array4<Complex>,
    /// FEFF `hbmatr(mj,iq,mu,k1)` stored as `(mj, mu, q, k1)`.
    pub right_matrix: Array4<Complex>,
    /// FEFF-generated `jind(1:indmax)` doubled final-state angular momenta.
    pub generated_final_j2: Vec<i32>,
}

/// Inputs for the FEFF `GENFMT/genfmtjas.f90` transition setup branch.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasTransitionMatricesInput<'a> {
    /// FEFF `elpty`; `elpty >= 0` selects `mmtrjas`, otherwise `mmtrjas0`.
    pub ellipticity: Real,
    /// Inputs for FEFF `mmtrjas`, used when [`Self::ellipticity`] is nonnegative.
    pub left_right: JasOneSidedTransitionInput<'a>,
    /// Inputs for FEFF `mmtrjas0`, used when [`Self::ellipticity`] is negative.
    pub spherical: JasSpinTransitionInput<'a>,
}

/// FEFF GENFMTJAS transition setup result.
#[derive(Debug, Clone, PartialEq)]
pub enum GenfmtJasTransitionMatrices {
    /// FEFF `elpty >= 0`: q-resolved left/right transition matrices.
    LeftRight(JasOneSidedTransitionMatrices),
    /// FEFF `elpty < 0`: spherical-averaging transition tensor.
    Spherical(JasSpinTransitionMatrix),
}

/// Inputs for the checked FEFF `GENFMT/genfmtjas.f90` transition setup block.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtJasTransitionSetupInput<'a> {
    /// FEFF `elpty`; `elpty >= 0` selects `mmtrjas`, otherwise `mmtrjas0`.
    pub ellipticity: Real,
    /// FEFF `indmaxt`, read from `phase.bin`.
    pub phase_transition_count: usize,
    /// FEFF `indmax`, requested by NRIXS input.
    pub requested_transition_count: usize,
    /// Inputs for FEFF `mmtrjas`, used when [`Self::ellipticity`] is nonnegative.
    pub left_right: JasOneSidedTransitionInput<'a>,
    /// Inputs for FEFF `mmtrjas0`, used when [`Self::ellipticity`] is negative.
    pub spherical: JasSpinTransitionInput<'a>,
}

/// Checked FEFF GENFMTJAS transition setup result.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtJasTransitionSetup {
    /// FEFF effective `jinit` used by the selected transition branch.
    pub effective_initial_j: GenfmtJasEffectiveInitialJ,
    /// Checked `indmaxt == indmax` transition count.
    pub transition_count: GenfmtJasTransitionCount,
    /// FEFF energy-independent transition matrices for the selected branch.
    pub matrices: GenfmtJasTransitionMatrices,
}

/// Rotation inputs for FEFF `GENFMT/mmtr.f90` matrix assembly.
#[derive(Debug, Clone, Copy)]
pub enum TransitionRotationInput<'a> {
    /// FEFF `ipol != 0`: use separate rotations from polarization to first
    /// leg and last leg to polarization, plus the two azimuthal phase factors.
    Polarized {
        /// FEFF `dri(:,:,:,nsc+2)`, angle between z and first leg.
        first_rotation: ArrayView3<'a, Real>,
        /// FEFF `dri(:,:,:,nleg)`, angle between last leg and z.
        last_rotation: ArrayView3<'a, Real>,
        /// FEFF `eta(0)`, gamma between polarization and first leg.
        first_eta: Real,
        /// FEFF `eta(nsc+2)`, alpha between last leg and polarization.
        last_eta: Real,
    },
    /// FEFF `ipol == 0`: use the precombined first-to-last-leg rotation.
    Unpolarized {
        /// FEFF `dri(:,:,:,nsc+1)`, angle between last leg and first leg.
        combined_rotation: ArrayView3<'a, Real>,
    },
}

/// Inputs for FEFF `GENFMT/mmtr.f90` energy-independent transition matrix.
#[derive(Debug, Clone, Copy)]
pub struct EnergyIndependentMatrixInput<'a> {
    /// FEFF transition angular momenta `lind(1:8)`.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `bmat(-lx:lx,0:1,1:8,-lx:lx,0:1,1:8)`.
    ///
    /// Rust indices are `(m1 + transition_magnetic_offset, spin1, k1,
    /// m2 + transition_magnetic_offset, spin2, k2)`.
    pub transition_b_matrix: ArrayView6<'a, Complex>,
    /// Magnetic-index offset for the first and fourth `transition_b_matrix`
    /// axes.
    pub transition_magnetic_offset: usize,
    /// FEFF selected spin index `is`.
    pub spin_index: usize,
    /// FEFF `ilinit`, the initial orbital angular-momentum limit.
    pub initial_l: usize,
    /// FEFF `mtot`, the output magnetic-index limit.
    pub magnetic_limit: usize,
    /// Magnetic-index offset for all rotation matrices.
    pub rotation_magnetic_offset: usize,
    /// Polarized or unpolarized FEFF rotation branch.
    pub rotations: TransitionRotationInput<'a>,
}

/// Inputs for the ordinary FEFF GENFMT spin-loop `mmtr` setup.
#[derive(Debug, Clone, Copy)]
pub struct GenfmtOrdinaryTransitionMatricesInput<'a> {
    /// FEFF `ispin` selector from the driver.
    pub spin_selector: i32,
    /// FEFF active `nsp` after applying the driver spin-channel rule.
    pub active_spin_channel_count: usize,
    /// FEFF `nspx`, the number of spin channels available to `mmtr`.
    pub available_spin_channels: usize,
    /// FEFF transition angular momenta `lind(1:8)`.
    pub transition_angular_momenta: ArrayView1<'a, i32>,
    /// FEFF `bmat(-lx:lx,0:1,1:8,-lx:lx,0:1,1:8)`.
    pub transition_b_matrix: ArrayView6<'a, Complex>,
    /// Magnetic-index offset for the first and fourth `transition_b_matrix`
    /// axes.
    pub transition_magnetic_offset: usize,
    /// FEFF `ilinit`, the initial orbital angular-momentum limit.
    pub initial_l: usize,
    /// FEFF `mtot`, the output magnetic-index limit.
    pub magnetic_limit: usize,
    /// Magnetic-index offset for all rotation matrices.
    pub rotation_magnetic_offset: usize,
    /// Polarized or unpolarized FEFF rotation branch.
    pub rotations: TransitionRotationInput<'a>,
}

/// Ordinary FEFF GENFMT `bmati` matrices for every active spin channel.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtOrdinaryTransitionMatrices {
    /// FEFF `bmati` for each active spin channel.
    ///
    /// Rust axes are `(active_spin, m1 + magnetic_limit, transition1,
    /// m2 + magnetic_limit, transition2)`.
    pub matrices: Array5<Complex>,
    /// Zero-based `bmat` spin slots selected by FEFF's `mmtr` wrapper logic.
    pub b_matrix_spin_indices: Vec<usize>,
}

/// Compact FEFF `rot3i` rotation table for one path leg.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialStateRotation {
    /// FEFF `dri(il,m1+mtot+1,m2+mtot+1,ileg)` without unused global padding.
    ///
    /// Rust indices are `(il - 1, m1 + magnetic_offset, m2 + magnetic_offset)`.
    pub matrix: Array3<Real>,
    /// Offset added to signed magnetic indices before indexing `matrix`.
    pub magnetic_offset: usize,
}

/// FEFF-shaped `rot3i` rotation tables for one path.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtPathRotationTables {
    /// FEFF `dri(:,:,:,ileg)` tables in call order.
    ///
    /// Rust axes are `(rotation, il - 1, m1 + rotation_magnetic_offset,
    /// m2 + rotation_magnetic_offset)`. The first [`Self::real_leg_count`]
    /// entries are real path legs; any remaining entry is FEFF's polarized
    /// pseudo-leg.
    pub rotations: Array4<Real>,
    /// Count of real path-leg rotations at the start of [`Self::rotations`].
    pub real_leg_count: usize,
    /// Shared FEFF magnetic-index offset for all padded rotation tables.
    pub rotation_magnetic_offset: usize,
}

/// FEFF path-local setup before the GENFMT energy loop.
#[derive(Debug, Clone, PartialEq)]
pub struct GenfmtPathSetup {
    /// FEFF `rdpath` beta, eta, and leg-length tables.
    pub angles: PathRotationAngles,
    /// FEFF `reff=sum(ri)/2`.
    pub effective_half_path_length: Real,
    /// FEFF `setlam(icalc, 1)` lambda selection.
    pub lambda: LambdaIndexSet,
    /// FEFF `rot3i` rotation tables in `dri` layout.
    pub rotations: GenfmtPathRotationTables,
}

/// FEFF `rdpath` angle tables for one path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathRotationAngles {
    /// FEFF `beta(1:nangle)` scattering angles in radians.
    pub beta_angles: Array1<Real>,
    /// FEFF `eta(0:nleg+1)` azimuthal phase factors.
    ///
    /// Rust index `j` intentionally maps to FEFF `eta(j)` so the polarized
    /// endpoints remain directly addressable as `eta_values[0]` and
    /// `eta_values[nleg + 1]`.
    pub eta_values: Array1<Real>,
    /// FEFF `ri(1:nleg)` leg lengths in the same units as the input positions.
    pub leg_lengths: Array1<Real>,
}

/// FEFF lambda index arrays and associated `setlam` metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaIndexSet {
    /// FEFF `mlam(1:lamx)` magnetic indices.
    pub m_indices: Array1<i32>,
    /// FEFF `nlam(1:lamx)` order indices.
    pub n_indices: Array1<i32>,
    /// FEFF `laml0x`: prefix count whose entries are within `ilinit`.
    pub initial_l_prefix_len: usize,
    /// FEFF `mmaxp1`, computed after capacity truncation and ordering.
    pub max_m_plus_one: usize,
    /// FEFF final `nmax`, computed after capacity truncation and ordering.
    pub max_n: usize,
    /// FEFF `iord`, the requested Rehr-Albers order.
    pub order: i32,
    /// Requested `nmax` before lambda-capacity truncation.
    pub requested_n_max: usize,
    /// Requested `mmax` before lambda-capacity truncation.
    pub requested_m_max: usize,
    /// Whether FEFF would have logged `Lambda array filled, some order lost`.
    pub truncated: bool,
}

/// Error returned by FEFF `GENFMT` helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum GenfmtError {
    /// FEFF only defines nonnegative `icalc` values through `10`.
    #[error("undefined FEFF lambda calculation {calculation}")]
    UndefinedLambdaCalculation { calculation: i32 },
    /// A negative `icalc` could not be decoded safely.
    #[error("lambda calculation code {calculation} cannot be decoded safely")]
    LambdaCodeOverflow { calculation: i32 },
    /// The cute heuristic needs finite beta angles.
    #[error("beta angle at index {index} must be finite, got {value}")]
    NonFiniteBetaAngle { index: usize, value: Real },
    /// A generated FEFF integer field would overflow.
    #[error("lambda field {field}={value} does not fit in i32")]
    IntegerOverflow { field: &'static str, value: usize },
    /// GENFMT angular limits must be positive and fit index calculations.
    #[error("invalid GENFMT angular limit {name}={value}")]
    InvalidAngularLimit { name: &'static str, value: usize },
    /// FEFF `rot3i` requires a finite beta angle.
    #[error("rotation beta angle must be finite")]
    NonFiniteRotationAngle,
    /// FEFF path angle construction needs at least one path row.
    #[error("path positions must contain at least one leg")]
    EmptyPath,
    /// FEFF path angle construction uses three Cartesian coordinates per row.
    #[error("path positions must have exactly 3 coordinate columns, got {columns}")]
    InvalidPathCoordinateColumns { columns: usize },
    /// FEFF path coordinates must be finite.
    #[error(
        "path position leg index {leg_index} component {component} must be finite, got {value}"
    )]
    NonFinitePathCoordinate {
        leg_index: usize,
        component: usize,
        value: Real,
    },
    /// FEFF `sclmz` needs a finite complex path length.
    #[error("{field} must be finite, got ({real}, {imaginary})")]
    NonFiniteComplex {
        field: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// FEFF `sclmz` divides by the complex path length.
    #[error("{field} must be nonzero")]
    ZeroComplex { field: &'static str },
    /// FEFF `xstar` only tabulates Legendre coefficients through `ilinit=4`.
    #[error("initial angular momentum {initial_l} is outside GENFMT xstar table range 1..=4")]
    InvalidInitialAngularMomentum { initial_l: usize },
    /// Scalar GENFMT inputs must be finite.
    #[error("{field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// Scalar GENFMT inputs must satisfy FEFF-compatible sign constraints.
    #[error("{field} must be nonnegative, got {value}")]
    NegativeScalar { field: &'static str, value: Real },
    /// FEFF percentage calculations require a nonzero normalization.
    #[error("{field} must be nonzero")]
    ZeroScalar { field: &'static str },
    /// Vector GENFMT inputs must have finite components.
    #[error("{field}[{index}] must be finite, got {value}")]
    NonFiniteVector {
        field: &'static str,
        index: usize,
        value: Real,
    },
    /// FEFF `xxcos` is undefined for zero-length vectors.
    #[error("{field} must have nonzero length")]
    ZeroVector { field: &'static str },
    /// Generated lambda indices exceed the caller's FEFF dimensions.
    #[error(
        "lambda selection exceeded dimensions: mmaxp1={max_m_plus_one}, nmax={max_n}, mtot={max_m}, ntot={max_n_limit}"
    )]
    DimensionExceeded {
        max_m_plus_one: usize,
        max_n: usize,
        max_m: usize,
        max_n_limit: usize,
    },
    /// A lambda count exceeds the supplied lambda-index arrays.
    #[error("{name}={requested} exceeds lambda array length {available}")]
    LambdaCountOutOfRange {
        name: &'static str,
        requested: usize,
        available: usize,
    },
    /// FEFF signed phase vectors must cover `-lmax..=lmax`.
    #[error("signed phase vector length {length} must be odd and nonzero")]
    InvalidSignedPhaseShape { length: usize },
    /// A FEFF lambda index cannot be represented safely.
    #[error("lambda {field} at index {index} has invalid value {value}")]
    InvalidLambdaIndex {
        index: usize,
        field: &'static str,
        value: i32,
    },
    /// An ndarray axis is too short for FEFF-compatible indexing.
    #[error("{table} axis {axis} length {length} is smaller than required {required}")]
    TableAxisTooShort {
        table: &'static str,
        axis: &'static str,
        length: usize,
        required: usize,
    },
    /// A complex table entry must be finite.
    #[error("{table}({row},{column}) must be finite, got ({real}, {imaginary})")]
    NonFiniteTableComplex {
        table: &'static str,
        row: usize,
        column: usize,
        real: Real,
        imaginary: Real,
    },
    /// A three-dimensional complex tensor entry must be finite.
    #[error("{table}({i0},{i1},{i2}) must be finite, got ({real}, {imaginary})")]
    NonFiniteTensor3Complex {
        table: &'static str,
        i0: usize,
        i1: usize,
        i2: usize,
        real: Real,
        imaginary: Real,
    },
    /// A complex tensor entry must be finite.
    #[error("{table}({i0},{i1},{i2},{i3}) must be finite, got ({real}, {imaginary})")]
    NonFiniteTensorComplex {
        table: &'static str,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        real: Real,
        imaginary: Real,
    },
    /// A five-dimensional complex tensor entry must be finite.
    #[error("{table}({i0},{i1},{i2},{i3},{i4}) must be finite, got ({real}, {imaginary})")]
    NonFiniteTensor5Complex {
        table: &'static str,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        real: Real,
        imaginary: Real,
    },
    /// A six-dimensional complex tensor entry must be finite.
    #[error("{table}({i0},{i1},{i2},{i3},{i4},{i5}) must be finite, got ({real}, {imaginary})")]
    NonFiniteTensor6Complex {
        table: &'static str,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        i5: usize,
        real: Real,
        imaginary: Real,
    },
    /// A real table entry must be finite.
    #[error("{table}({row},{column}) must be finite, got {value}")]
    NonFiniteTableScalar {
        table: &'static str,
        row: usize,
        column: usize,
        value: Real,
    },
    /// FEFF divides by `xnlm(m,l)` in `fmtrxi`.
    #[error("xnlm({magnetic},{angular_momentum}) must be nonzero")]
    ZeroLegendreNormalization {
        angular_momentum: usize,
        magnetic: usize,
    },
    /// FEFF JAS doubled angular momentum must be nonnegative.
    #[error("{field} doubled angular momentum must be nonnegative, got {value}")]
    InvalidDoubledAngularMomentum { field: &'static str, value: i32 },
    /// Left/right JAS decomposition tables must be supplied as a pair.
    #[error("left/right JAS decomposition tables must both be present or both be absent")]
    MismatchedJasDecompositionTables,
    /// Retained JAS path decomposition output must be consistently present.
    #[error("retained GENFMTJAS path decomposition output must be consistently present or absent")]
    MismatchedJasFinalizationDecomposition,
    /// FEFF `genfmtjas` requires the phase-file transition count to match NRIXS input.
    #[error(
        "GENFMTJAS phase transition count {phase_transition_count} does not match requested indmax {requested_transition_count}"
    )]
    MismatchedJasTransitionCount {
        phase_transition_count: usize,
        requested_transition_count: usize,
    },
    /// FEFF relativistic kappa values are nonzero.
    #[error("invalid initial kappa {kappa}; expected nonzero relativistic kappa")]
    InvalidInitialKappa { kappa: i32 },
    /// Text fields copied to FEFF output must be non-empty ASCII text.
    #[error("{field} must be non-empty ASCII text")]
    InvalidTextField { field: &'static str },
    /// FEFF `feff.bin` potential labels are non-empty ASCII labels up to 6 bytes.
    #[error("potential label at index {index} must be non-empty ASCII text up to 6 bytes")]
    InvalidPotentialLabel { index: usize },
    /// FEFF did not generate enough JAS final-state slots for `indmax`.
    #[error(
        "generated {generated} JAS final states, but {required} transition slots were requested"
    )]
    InsufficientGeneratedTransitions { required: usize, generated: usize },
    /// Atomic lookup failure while filling GENFMT output labels.
    #[error(transparent)]
    Atomic(#[from] AtomicError),
    /// Angular helper failure while evaluating FEFF coupling coefficients.
    #[error(transparent)]
    Angular(#[from] AngularError),
    /// Quadrature helper failure while integrating GENFMT path importance.
    #[error(transparent)]
    Quadrature(#[from] QuadratureError),
    /// Phase helper failure while unwrapping GENFMT output phases.
    #[error(transparent)]
    Phase(#[from] PhaseError),
}
