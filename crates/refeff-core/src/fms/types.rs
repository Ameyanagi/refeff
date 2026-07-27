use ndarray::{
    Array2, Array3, Array4, Array5, Array6, ArrayView2, ArrayView3, ArrayView4, ArrayView5,
    ArrayView6,
};
use num_complex::Complex32;
use refeff_linalg::LinalgError;
use thiserror::Error;

use crate::{
    Real,
    angular::SpinOrbitCouplingTables,
    state::{StateKet, StateKetSet},
};

/// Atom record used by FEFF FMS cluster preparation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmsAtom {
    /// Cartesian position in FEFF FMS single-precision arithmetic.
    pub position: [f32; 3],
    /// FEFF potential index for this atom.
    pub potential: i32,
}

/// Inputs for the FEFF `fmspack.f90` FMS setup prelude.
#[derive(Debug, Clone)]
pub struct FmsDriverSetupInput<'a> {
    /// FEFF `lfms` selector; `0` packs only the absorber potential.
    pub lfms: i32,
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FMS cluster atoms in FEFF `iphx` order.
    pub atoms: &'a [FmsAtom],
    /// Inclusive FEFF `npot` maximum potential index.
    pub max_potential: usize,
    /// FEFF global `lx` angular momentum limit.
    pub global_lmax: usize,
    /// Raw FEFF `lipotx(0:nphx)` values before `fmspack` clamps them.
    pub raw_potential_lmax: &'a [i32],
    /// Optional `istatx`-style state-ket capacity.
    pub state_capacity: Option<usize>,
}

/// FEFF-compatible state and potential-range setup for FMS solvers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FmsDriverSetup {
    /// Clamped FEFF `lipotx(0:npot)` values used by FMS solvers.
    pub potential_lmax: Vec<usize>,
    /// First potential index to pack into `gg`.
    pub potential_start: usize,
    /// Final potential index to pack into `gg`.
    pub potential_end: usize,
    /// FEFF `getkts` state table and representative offsets.
    pub state_kets: StateKetSet,
}

/// Inputs for one real-space FEFF FMS energy point.
#[derive(Debug, Clone)]
pub struct FmsRealSpaceEnergyInput<'a> {
    /// FEFF `lfms` selector; `0` packs only the absorber potential.
    pub lfms: i32,
    /// Raw FEFF `minv` solver selector.
    pub minv: i32,
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// FMS cluster atoms in FEFF `iphx` order.
    pub atoms: &'a [FmsAtom],
    /// Inclusive FEFF `npot` maximum potential index.
    pub max_potential: usize,
    /// FEFF global `lx` angular momentum limit.
    pub global_lmax: usize,
    /// Raw FEFF `lipotx(0:nphx)` values before `fmspack` clamps them.
    pub raw_potential_lmax: &'a [i32],
    /// Optional `istatx`-style state-ket capacity.
    pub state_capacity: Option<usize>,
    /// Complex wave numbers `ck(spin)` in inverse Angstrom.
    pub wave_numbers: &'a [Complex32],
    /// FEFF `xphase(spin,l,potential)` table with signed `l` centered.
    pub phase_shifts: ArrayView3<'a, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
    /// Direct-space cutoff `rdirec` in Angstrom.
    pub direct_cutoff: f32,
    /// FEFF `sigsqr(atom2,atom1)` mean-square displacement table in Angstrom squared.
    pub mean_square_displacements: ArrayView2<'a, f32>,
    /// FEFF `xnlm(mu,l)` normalization table.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `drix(m2,m1,l,k,atom2,atom1)` rotation table.
    pub rotations: ArrayView6<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for iterative angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance for iterative branches.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff for iterative system-matrix construction.
    pub zero_tolerance: f32,
    /// Whether FEFF `gg_full` output is requested.
    pub full_scattering_matrix_requested: bool,
}

/// Inputs for building a reusable [`FmsRealSpacePlan`] via
/// [`crate::fms::fms_real_space_plan`].
///
/// This collects every FEFF FMS real-space input that stays fixed across an
/// energy sweep: cluster geometry, angular-momentum limits, and the
/// once-per-run tables (`sigsqr`, `xnlm`, `drix`, spin-orbit coefficients).
/// Only the complex wave number and phase-shift tables vary per energy point;
/// those are supplied separately through [`FmsRealSpaceEnergyPoint`].
#[derive(Debug, Clone)]
pub struct FmsRealSpacePlanInput<'a> {
    /// FEFF `lfms` selector; `0` packs only the absorber potential.
    pub lfms: i32,
    /// Raw FEFF `minv` solver selector.
    pub minv: i32,
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// FMS cluster atoms in FEFF `iphx` order.
    pub atoms: &'a [FmsAtom],
    /// Inclusive FEFF `npot` maximum potential index.
    pub max_potential: usize,
    /// FEFF global `lx` angular momentum limit.
    pub global_lmax: usize,
    /// Raw FEFF `lipotx(0:nphx)` values before `fmspack` clamps them.
    pub raw_potential_lmax: &'a [i32],
    /// Optional `istatx`-style state-ket capacity.
    pub state_capacity: Option<usize>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
    /// Direct-space cutoff `rdirec` in Angstrom.
    pub direct_cutoff: f32,
    /// FEFF `sigsqr(atom2,atom1)` mean-square displacement table in Angstrom squared.
    pub mean_square_displacements: ArrayView2<'a, f32>,
    /// FEFF `xnlm(mu,l)` normalization table.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `drix(m2,m1,l,k,atom2,atom1)` rotation table.
    pub rotations: ArrayView6<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for iterative angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance for iterative branches.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff for iterative system-matrix construction.
    pub zero_tolerance: f32,
    /// Whether FEFF `gg_full` output is requested.
    pub full_scattering_matrix_requested: bool,
    /// Whether to keep [`FmsDriverSetup`] on each [`FmsRealSpaceEnergyResult`].
    pub retain_setup: bool,
    /// Whether to keep spin-resolved pair tables on each result.
    pub retain_pair_tables: bool,
    /// Whether to keep the free-propagator matrix on each result.
    pub retain_free_propagator: bool,
    /// Whether to keep the compact T-matrix on each result.
    pub retain_t_matrix: bool,
    /// Whether to keep the assembled scattering system matrix on each result.
    pub retain_system_matrix: bool,
}

/// Energy-independent FEFF FMS real-space setup, built once via
/// [`crate::fms::fms_real_space_plan`] and reused across an energy sweep.
///
/// This is `Sync` (every field is either `Copy`, an immutable reference, or
/// owned data behind a shared reference) so it can be shared across worker
/// threads, e.g. with `rayon`'s `into_par_iter`.
#[derive(Debug, Clone)]
pub struct FmsRealSpacePlan<'a> {
    pub(super) setup: FmsDriverSetup,
    pub(super) minv: i32,
    pub(super) spin_channels: usize,
    pub(super) spin_selector: i32,
    pub(super) atoms: &'a [FmsAtom],
    pub(super) global_lmax: usize,
    pub(super) spin_orbit: &'a SpinOrbitCouplingTables,
    pub(super) direct_cutoff: f32,
    pub(super) mean_square_displacements: ArrayView2<'a, f32>,
    pub(super) xnlm: ArrayView2<'a, Real>,
    pub(super) rotations: ArrayView6<'a, Complex32>,
    pub(super) calculated_l: &'a [bool],
    pub(super) convergence_tolerance: f32,
    pub(super) zero_tolerance: f32,
    pub(super) full_scattering_matrix_requested: bool,
    pub(super) retain_setup: bool,
    pub(super) retain_pair_tables: bool,
    pub(super) retain_free_propagator: bool,
    pub(super) retain_t_matrix: bool,
    pub(super) retain_system_matrix: bool,
}

/// Per-energy inputs consumed against a shared [`FmsRealSpacePlan`].
///
/// Every other FEFF FMS real-space input is fixed for the sweep and lives on
/// the plan; only the complex wave number and phase-shift tables change from
/// one energy point to the next.
#[derive(Debug, Clone, Copy)]
pub struct FmsRealSpaceEnergyPoint<'a> {
    /// Complex wave numbers `ck(spin)` in inverse Angstrom.
    pub wave_numbers: &'a [Complex32],
    /// FEFF `xphase(spin,l,potential)` table with signed `l` centered.
    pub phase_shifts: ArrayView3<'a, Complex32>,
}

/// Result for one real-space FEFF FMS energy point.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsRealSpaceEnergyResult {
    /// FEFF setup prelude result, including clamped `lipotx` and state kets.
    ///
    /// `None` when the plan or caller did not request it retained; see
    /// [`FmsRealSpacePlanInput::retain_setup`].
    pub setup: Option<FmsDriverSetup>,
    /// Effective solver method after FEFF compatibility adjustments.
    pub method_selection: FmsScatteringMethodSelection,
    /// Spin-resolved `xrho` and `xclm` tables for this energy.
    ///
    /// `None` when not retained; see
    /// [`FmsRealSpacePlanInput::retain_pair_tables`].
    pub pair_tables: Option<FmsSpinPairTables>,
    /// FEFF `g0(state,state)` free-propagator matrix.
    ///
    /// `None` when not retained; see
    /// [`FmsRealSpacePlanInput::retain_free_propagator`].
    pub free_propagator: Option<Array2<Complex32>>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    ///
    /// `None` when not retained; see [`FmsRealSpacePlanInput::retain_t_matrix`].
    pub t_matrix: Option<Array2<Complex32>>,
    /// Solver output and packed `gg` matrices.
    pub scattering: FmsScatteringResult,
}

/// Inputs for FEFF `FMS/yprep.f90` absorber-centered cluster selection.
#[derive(Debug, Clone)]
pub struct FmsYprepClusterInput<'a> {
    /// Central potential `iph0`; `0` is the absorbing atom.
    pub central_potential: i32,
    /// FEFF potential index `iphat(i)` for each atom.
    pub potentials: &'a [i32],
    /// Cartesian atom positions `rat(:,i)` as an `atoms x 3` table.
    pub positions: ArrayView2<'a, f32>,
    /// FMS cluster cutoff radius `rmax`.
    pub cluster_radius: f32,
    /// Hard cluster capacity, equivalent to FEFF `nclusx`.
    pub cluster_capacity: usize,
}

/// Result from FEFF `yprep` cluster-prefix construction.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsYprepCluster {
    /// Atom index used as the center before FEFF shifts coordinates to the absorber.
    pub central_atom: usize,
    /// Number of atoms within `cluster_radius` before capacity truncation.
    pub untruncated_count: usize,
    /// Absorber-centered, radius-sorted cluster prefix copied into FEFF `xrat`/`iphx`.
    pub atoms: Vec<FmsAtom>,
}

/// Pair-angle and rotation tables prepared by FEFF `yprep`.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsYprepGeometry {
    /// FEFF `xphi(i,j)` azimuth table for the vector `R_i - R_j`.
    pub phi: Array2<f32>,
    /// FEFF `drix(m2,m1,l,k,j,i)` forward/backward rotation table.
    pub rotations: Array6<Complex32>,
}

/// Direction branch used by FEFF `rotxan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmsRotationDirection {
    /// FEFF `k=0`, used for forward rotations.
    Forward,
    /// FEFF `k=1`, used for backward rotations.
    Backward,
}

/// FEFF FMS pair tables for one energy point and spin channel.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsPairTables {
    /// `xrho(atom2, atom1)` complex distance table.
    pub rho: Array2<Complex32>,
    /// `xclm(m, l, atom2, atom1)` Rehr-Albers polynomial table.
    pub polynomials: Array4<Complex32>,
}

/// FEFF FMS pair tables with an explicit spin axis.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsSpinPairTables {
    /// `xrho(atom2, atom1, spin)` complex distance table.
    pub rho: Array3<Complex32>,
    /// `xclm(m, l, atom2, atom1, spin)` Rehr-Albers polynomial table.
    pub polynomials: Array5<Complex32>,
}

/// FEFF FMS scattering branch selected by `minv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmsScatteringMethod {
    /// FEFF `minv=0`, direct LU factorization via `gglu`.
    Lu,
    /// FEFF `minv=1`, BiCGStab/VdV branch via `ggbi`.
    BiCgStab,
    /// FEFF `minv=2`, Lanczos/recursion branch via `ggrm`.
    Recursion,
    /// FEFF `minv=3`, Graves-Morris/Salam branch via `gggm`.
    GravesMorris,
    /// FEFF fallback branch for all other `minv` values via `ggtf`.
    Tfqmr,
}

impl FmsScatteringMethod {
    /// Return FEFF's three-character runtime label for this branch.
    pub fn feff_label(self) -> &'static str {
        match self {
            Self::Lu => "LUD",
            Self::BiCgStab => "VdV",
            Self::Recursion => "LLU",
            Self::GravesMorris => "GMS",
            Self::Tfqmr => "TF",
        }
    }
}

/// FEFF `minv` normalization and method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FmsScatteringMethodSelection {
    /// Effective FEFF `minv` value after compatibility adjustments.
    pub effective_minv: i32,
    /// Scattering branch selected from `effective_minv`.
    pub method: FmsScatteringMethod,
    /// Whether a requested full scattering matrix forced LU inversion.
    pub forced_lu_for_full_scattering: bool,
}

/// Inputs for one FEFF FMS free-propagator matrix element.
#[derive(Debug, Clone)]
pub struct FmsFreePropagatorInput<'a> {
    /// Bra-side FEFF state.
    pub first: StateKet,
    /// Ket-side FEFF state.
    pub second: StateKet,
    /// Pair `rho = ck * |R_i - R_j|`.
    pub rho: Complex32,
    /// Complex wave number `ck` in inverse Angstrom.
    pub wave_number: Complex32,
    /// Pair mean-square displacement in Angstrom squared.
    pub mean_square_displacement: f32,
    /// FEFF `xclm(m,l,atom2,atom1)` table.
    pub xclm: ArrayView4<'a, Complex32>,
    /// FEFF `xnlm(mu,l)` normalization table.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `drix(...,k=1,atom2,atom1)` backward rotation table.
    pub backward_rotation: ArrayView3<'a, Complex32>,
    /// FEFF `drix(...,k=0,atom2,atom1)` forward rotation table.
    pub forward_rotation: ArrayView3<'a, Complex32>,
}

/// Inputs for building the FEFF FMS free-propagator matrix.
#[derive(Debug, Clone)]
pub struct FmsFreePropagatorMatrixInput<'a> {
    /// FEFF state kets in matrix row/column order.
    pub states: &'a [StateKet],
    /// FMS cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// Direct-space cutoff `rdirec` in Angstrom.
    pub direct_cutoff: f32,
    /// FEFF `xrho(atom2,atom1)` table.
    pub rho: ArrayView2<'a, Complex32>,
    /// Complex wave number `ck` in inverse Angstrom.
    pub wave_number: Complex32,
    /// FEFF `sigsqr(atom2,atom1)` mean-square displacement table in Angstrom squared.
    pub mean_square_displacements: ArrayView2<'a, f32>,
    /// FEFF `xclm(m,l,atom2,atom1)` table.
    pub xclm: ArrayView4<'a, Complex32>,
    /// FEFF `xnlm(mu,l)` normalization table.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `drix(m2,m1,l,k,atom2,atom1)` rotation table.
    pub rotations: ArrayView6<'a, Complex32>,
}

/// Inputs for building FEFF's spin-resolved FMS free-propagator matrix.
#[derive(Debug, Clone)]
pub struct FmsSpinFreePropagatorMatrixInput<'a> {
    /// FEFF state kets in matrix row/column order.
    pub states: &'a [StateKet],
    /// FMS cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// Direct-space cutoff `rdirec` in Angstrom.
    pub direct_cutoff: f32,
    /// FEFF `xrho(atom2,atom1,spin)` table.
    pub rho: ArrayView3<'a, Complex32>,
    /// Complex wave numbers `ck(spin)` in inverse Angstrom.
    pub wave_numbers: &'a [Complex32],
    /// FEFF `sigsqr(atom2,atom1)` mean-square displacement table in Angstrom squared.
    pub mean_square_displacements: ArrayView2<'a, f32>,
    /// FEFF `xclm(m,l,atom2,atom1,spin)` table.
    pub xclm: ArrayView5<'a, Complex32>,
    /// FEFF `xnlm(mu,l)` normalization table.
    pub xnlm: ArrayView2<'a, Real>,
    /// FEFF `drix(m2,m1,l,k,atom2,atom1)` rotation table.
    pub rotations: ArrayView6<'a, Complex32>,
}

/// Inputs for one FEFF FMS single-site T-matrix element.
#[derive(Debug, Clone)]
pub struct FmsTMatrixInput<'a> {
    /// Bra-side FEFF state.
    pub first: StateKet,
    /// Ket-side FEFF state.
    pub second: StateKet,
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// Zero-based potential index for the shared atom.
    pub potential: usize,
    /// FEFF `xphase(spin,l,potential)` table with signed `l` centered.
    pub phase_shifts: ArrayView3<'a, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
}

/// Inputs for building FEFF's compact `tmatrx(spin_band,state)` table.
#[derive(Debug, Clone)]
pub struct FmsTMatrixTableInput<'a> {
    /// FEFF state kets in compact-table column order.
    pub states: &'a [StateKet],
    /// FMS cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// FEFF `xphase(spin,l,potential)` table with signed `l` centered.
    pub phase_shifts: ArrayView3<'a, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
}

/// Inputs for one FEFF `fms_h` Hubbard magnetic T-matrix element.
#[derive(Debug, Clone)]
pub struct FmsHubbardTMatrixInput<'a> {
    /// Bra-side FEFF state.
    pub first: StateKet,
    /// Ket-side FEFF state.
    pub second: StateKet,
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// Zero-based potential index for the shared atom.
    pub potential: usize,
    /// FEFF `xphase_m(spin,l,imm,potential)` table with signed `l` centered.
    pub magnetic_phase_shifts: ArrayView4<'a, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
}

/// Inputs for building FEFF `fms_h` full Hubbard `tmatrxfull(state,state)`.
#[derive(Debug, Clone)]
pub struct FmsHubbardTMatrixTableInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FMS cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// FEFF `xphase_m(spin,l,imm,potential)` table with signed `l` centered.
    pub magnetic_phase_shifts: ArrayView4<'a, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
}

/// Inputs for FEFF `fms_h` selected T-matrix block transformation.
#[derive(Debug, Clone)]
pub struct FmsHubbardTMatrixTransformInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FMS cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `UseTFrm(l,potential)` selector.
    pub use_transform: ArrayView2<'a, bool>,
    /// FEFF `TFrm(spin,row,column,l,potential)` transform matrix.
    pub transform: ArrayView5<'a, Complex32>,
    /// FEFF `TFrmInv(spin,row,column,l,potential)` inverse transform matrix.
    pub inverse: ArrayView5<'a, Complex32>,
    /// Full FEFF `tmatrxfull(state,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
}

/// Inputs for FEFF `fms_h` selected `gg` block back-transformation.
#[derive(Debug, Clone)]
pub struct FmsHubbardScatteringTransformInput<'a> {
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// FEFF `UseTFrm(l,potential)` selector.
    pub use_transform: ArrayView2<'a, bool>,
    /// FEFF `TFrm(spin,row,column,l,potential)` transform matrix.
    pub transform: ArrayView5<'a, Complex32>,
    /// FEFF `TFrmInv(spin,row,column,l,potential)` inverse transform matrix.
    pub inverse: ArrayView5<'a, Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: ArrayView3<'a, Complex32>,
}

/// Inputs for FEFF `fms_h` selected full `gg` matrix back-transformation.
#[derive(Debug, Clone)]
pub struct FmsHubbardFullScatteringTransformInput<'a> {
    /// FEFF state kets in full-matrix order.
    pub states: &'a [StateKet],
    /// FMS cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// FEFF `UseTFrm(l,potential)` selector.
    pub use_transform: ArrayView2<'a, bool>,
    /// FEFF `TFrm(spin,row,column,l,potential)` transform matrix.
    pub transform: ArrayView5<'a, Complex32>,
    /// FEFF `TFrmInv(spin,row,column,l,potential)` inverse transform matrix.
    pub inverse: ArrayView5<'a, Complex32>,
    /// Full `gg(state,state)` scattering matrix.
    pub full_scattering: ArrayView2<'a, Complex32>,
}

/// Inputs for FEFF iterative FMS matrix assembly.
#[derive(Debug, Clone)]
pub struct FmsIterativeSystemInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `toler2` cutoff applied to `abs(g0)` terms.
    pub zero_tolerance: f32,
}

/// Inputs for dispatching FEFF's compact FMS scattering branches.
#[derive(Debug, Clone)]
pub struct FmsScatteringInput<'a> {
    /// FEFF solver branch selected from `minv`.
    pub method: FmsScatteringMethod,
    /// Request FEFF's `gg_full` matrix; this is only available through LU.
    pub calculate_full_scattering: bool,
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for iterative angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance for iterative branches.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff for iterative system-matrix construction.
    pub zero_tolerance: f32,
}

/// Result from FEFF compact FMS scattering dispatch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsScatteringResult {
    /// Solver branch used for this result.
    pub method: FmsScatteringMethod,
    /// Branch-specific work matrix assembled before solving.
    ///
    /// FEFF FMS always assembles this matrix as part of solving, so it is
    /// unconditionally `Some` when returned directly from
    /// [`fms_scattering`](crate::fms::fms_scattering);
    /// callers that do not need it (e.g. [`crate::fms::fms_real_space_plan`]
    /// consumers with `retain_system_matrix: false`) may drop it to `None`.
    pub system_matrix: Option<Array2<Complex32>>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `gg_full = (1 - G0*T)^-1 * G0` when requested for LU.
    pub full_scattering: Option<Array2<Complex32>>,
    /// FEFF `msord` for iterative branches; LU does not report one.
    pub multiple_scattering_order: Option<usize>,
}

/// Inputs for FEFF's BiCGStab FMS branch, `ggbi`.
#[derive(Debug, Clone)]
pub struct FmsBiCgStabInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff applied while building `1 - T*G0`.
    pub zero_tolerance: f32,
}

/// Result from FEFF's BiCGStab FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsBiCgStabResult {
    /// The assembled `1 - T*G0` matrix.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `msord` value from the last solved source channel.
    pub multiple_scattering_order: usize,
}

/// Inputs for FEFF's recursion-method FMS branch, `ggrm`.
#[derive(Debug, Clone)]
pub struct FmsRecursionInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff applied while building `1 - T*G0`.
    pub zero_tolerance: f32,
}

/// Result from FEFF's recursion-method FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsRecursionResult {
    /// The assembled `1 - T*G0` matrix.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `msord` value from the last solved source channel.
    pub multiple_scattering_order: usize,
}

/// Inputs for FEFF's Graves-Morris/Salam FMS branch, `gggm`.
#[derive(Debug, Clone)]
pub struct FmsGravesMorrisInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff applied while building `T*G0`.
    pub zero_tolerance: f32,
}

/// Result from FEFF's Graves-Morris/Salam FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsGravesMorrisResult {
    /// The assembled FEFF `T*G0` work matrix.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `msord` value from the last solved source channel.
    pub multiple_scattering_order: usize,
}

/// Inputs for FEFF's TFQMR FMS branch, `ggtf`.
#[derive(Debug, Clone)]
pub struct FmsTfqmrInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// FEFF `lcalc(l)` mask for angular-momentum channels.
    pub calculated_l: &'a [bool],
    /// FEFF `toler1` convergence tolerance.
    pub convergence_tolerance: f32,
    /// FEFF `toler2` cutoff applied while building `1 - T*G0`.
    pub zero_tolerance: f32,
}

/// Result from FEFF's TFQMR FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsTfqmrResult {
    /// The assembled `1 - T*G0` matrix.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `msord` value from the last solved source channel.
    pub multiple_scattering_order: usize,
}

/// Inputs for FEFF's LU FMS branch, `gglu`.
#[derive(Debug, Clone)]
pub struct FmsLuInput<'a> {
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// Request FEFF's `gg_full` matrix in addition to packed `gg`.
    pub calculate_full_scattering: bool,
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// FEFF compact `tmatrx(spin_band,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
}

/// Result from FEFF's LU FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsLuResult {
    /// The assembled `1 - G0*T` matrix before LU factorization.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `gg_full = (1 - G0*T)^-1 * G0` when requested.
    pub full_scattering: Option<Array2<Complex32>>,
}

/// Inputs for FEFF's full-potential LU FMS branch, `gglufullpot`.
#[derive(Debug, Clone)]
pub struct FmsFullPotentialLuInput<'a> {
    /// Request FEFF's full `gg(state,state)` matrix in addition to packed `gg`.
    pub calculate_full_scattering: bool,
    /// FEFF state kets in matrix order.
    pub states: &'a [StateKet],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// Global FEFF `lx`, used for output channel dimensions.
    pub global_lmax: usize,
    /// FEFF `lipotx` maximum angular momentum per potential.
    pub potential_lmax: &'a [usize],
    /// Representative state offsets `i0(ip)` from FEFF `getkts`.
    pub representative_offsets: &'a [Option<usize>],
    /// First potential index to pack.
    pub potential_start: usize,
    /// Final potential index to pack.
    pub potential_end: usize,
    /// FEFF `g0(state,state)` free-propagator matrix.
    pub free_propagator: ArrayView2<'a, Complex32>,
    /// Full FEFF `tmatrx(state,state)` table.
    pub t_matrix: ArrayView2<'a, Complex32>,
}

/// Result from FEFF's full-potential LU FMS branch.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsFullPotentialLuResult {
    /// The assembled full-potential `1 - G0*T` matrix before LU factorization.
    pub system_matrix: Array2<Complex32>,
    /// Packed `gg(channel1,channel2,potential)` scattering matrices.
    pub scattering: Array3<Complex32>,
    /// FEFF `gg_full = (1 - G0*T)^-1 * G0` when requested.
    pub full_scattering: Option<Array2<Complex32>>,
}

/// Error returned by FEFF FMS helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FmsError {
    /// An angular-coupling or Wigner-rotation helper rejected an MKGTR input.
    #[error("angular coupling failure: {0}")]
    Angular(#[from] crate::AngularError),
    /// NRIXS transition-index or `bcoefjas` construction failed.
    #[error("NRIXS transition failure: {0}")]
    Xsph(#[from] crate::XsphError),
    /// FEFF angular limits must fit the allocated `clm(lx+2, 2*lx+3)` table.
    #[error("{name}={value} is invalid for lx={lx}")]
    InvalidAngularLimit {
        name: &'static str,
        value: usize,
        lx: usize,
    },
    /// FEFF state-ket atom indices are one-based.
    #[error("state atom index must be one-based, got {atom}")]
    InvalidStateAtom { atom: usize },
    /// A zero-based Rust atom index was outside the supplied cluster table.
    #[error("atom index {index} is outside cluster length {len}")]
    AtomIndexOutOfRange { index: usize, len: usize },
    /// FMS atom positions must be an `atoms x 3` table.
    #[error("atom position table must have 3 columns, got {columns}")]
    AtomPositionColumnCount { columns: usize },
    /// Potential and position inputs must describe the same atoms.
    #[error("potential count {potentials} does not match atom position count {positions}")]
    AtomCountMismatch { potentials: usize, positions: usize },
    /// FEFF `yprep` needs a finite, nonnegative FMS cluster radius.
    #[error("cluster radius must be finite and nonnegative")]
    InvalidClusterRadius,
    /// FEFF `yprep` needs positive `nclusx` capacity.
    #[error("cluster capacity must be positive")]
    EmptyClusterCapacity,
    /// FEFF `yprep` did not find the requested central potential.
    #[error("central potential {potential} is not present in the atom table")]
    MissingCentralAtom { potential: i32 },
    /// FEFF stops when more than one absorbing atom with `ipot=0` is present.
    #[error("absorber potential 0 appears more than once")]
    DuplicateAbsorber,
    /// FMS cluster coordinates must be finite.
    #[error("atom {atom} coordinate axis {axis} must be finite")]
    NonFiniteCoordinate { atom: usize, axis: usize },
    /// FEFF FMS rotation angles must be finite.
    #[error("rotation angle {name} must be finite")]
    NonFiniteRotationAngle { name: &'static str },
    /// FMS potential indices must fit the caller-provided potential range.
    #[error("potential {potential} is outside 0..={max_potential}")]
    PotentialOutOfRange {
        potential: i32,
        max_potential: usize,
    },
    /// FEFF `sortat` requires the first atom to be the central potential.
    #[error("first atom potential {actual} does not match central potential {expected}")]
    CentralAtomMismatch { expected: i32, actual: i32 },
    /// FEFF `xgllm` is called with `mu <= l1`.
    #[error("mu={mu} is invalid for angular momentum l={angular_momentum}")]
    MuOutOfRange { mu: usize, angular_momentum: usize },
    /// An input table is too small for a required FEFF index.
    #[error("{table} table is too small for {axis} index {index}")]
    TableIndexOutOfRange {
        table: &'static str,
        axis: &'static str,
        index: usize,
    },
    /// FEFF `xnlm(mu,l)` must be finite and nonzero when used as a divisor.
    #[error("xnlm({mu},{angular_momentum}) must be finite and nonzero")]
    InvalidNormalization { mu: usize, angular_momentum: usize },
    /// `rho` appears in the denominator of FEFF `xclmz`.
    #[error("rho must be nonzero")]
    ZeroRho,
    /// `rho` must contain finite real and imaginary parts.
    #[error("rho must be finite")]
    NonFiniteRho,
    /// The complex wave number used for FMS pair tables must be finite.
    #[error("wave number must be finite")]
    NonFiniteWaveNumber,
    /// Pair mean-square displacement must be finite.
    #[error("mean-square displacement must be finite")]
    NonFiniteMeanSquareDisplacement,
    /// The direct FMS cutoff must be finite and nonnegative.
    #[error("direct FMS cutoff must be finite and nonnegative")]
    InvalidDirectCutoff,
    /// Iterative FMS tolerances must be finite and nonnegative.
    #[error("{name} tolerance must be finite and nonnegative, got {value}")]
    InvalidTolerance { name: &'static str, value: f32 },
    /// FEFF FMS supports one or two spin channels.
    #[error("invalid spin channel count {value}; expected 1 or 2")]
    InvalidSpinChannelCount { value: usize },
    /// FEFF `fms_h` transform support currently follows the one-spin branch.
    #[error("Hubbard FMS transform requires one spin channel, got {spin_channels}")]
    HubbardTransformSpinUnsupported { spin_channels: usize },
    /// FEFF FMS requires at least one cluster atom for `iphx(1)`.
    #[error("FMS cluster must contain at least one atom")]
    EmptyCluster,
    /// FEFF state spins are one-based and must fit the active spin channels.
    #[error("state spin {spin} is outside 1..={spin_channels}")]
    InvalidStateSpin { spin: usize, spin_channels: usize },
    /// State-ket construction saw an atom potential outside the `lipotx` table.
    #[error(
        "state atom {atom} references potential {potential}, but only {potential_count} potentials are available"
    )]
    StateKetPotentialOutOfRange {
        atom: usize,
        potential: usize,
        potential_count: usize,
    },
    /// FEFF `getkts` exceeded the caller-provided `istatx` capacity.
    #[error("state-ket count exceeded capacity {capacity}")]
    StateCapacityExceeded { capacity: usize },
    /// A generated FEFF state field could not be represented in a legacy integer.
    #[error("state field {field}={value} does not fit in a FEFF integer")]
    IntegerOverflow { field: &'static str, value: usize },
    /// FEFF phase shifts used by the FMS T-matrix must be finite.
    #[error("xphase(spin={spin}, l={angular_momentum}, potential={potential}) must be finite")]
    NonFinitePhaseShift {
        spin: usize,
        angular_momentum: isize,
        potential: usize,
    },
    /// A complex input or result table contains a non-finite value.
    #[error("{table} complex value at flat index {index} must be finite")]
    NonFiniteComplexValue { table: &'static str, index: usize },
    /// A requested potential has no representative state offset.
    #[error("missing representative state offset for potential {potential}")]
    MissingRepresentativePotential { potential: usize },
    /// FEFF-compatible LU factorization or solve failed.
    #[error("linear algebra failure: {0}")]
    LinearAlgebra(#[from] LinalgError),
    /// Iterative FMS solver encountered a zero denominator.
    #[error("{solver} solver breakdown at {step}")]
    IterativeSolverBreakdown {
        solver: &'static str,
        step: &'static str,
    },
    /// Iterative FMS solver did not converge before the Rust safety limit.
    #[error("{solver} solver did not converge after {restarts} restarts")]
    IterativeSolverNoConvergence {
        solver: &'static str,
        restarts: usize,
    },
    /// A spin-indexed input table did not match FEFF `nsp`.
    #[error("{table} spin channel count {actual} does not match nsp={expected}")]
    SpinChannelCountMismatch {
        table: &'static str,
        expected: usize,
        actual: usize,
    },
    /// FEFF only computes the full FMS scattering matrix through LU inversion.
    #[error("full FMS scattering matrix requires LU inversion, got {method:?}")]
    FullScatteringRequiresLu { method: FmsScatteringMethod },
}
