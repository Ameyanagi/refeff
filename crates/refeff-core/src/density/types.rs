use super::*;
use thiserror::Error;

/// Persistent FEFF `broydn` workspace.
///
/// FEFF stores Broyden history in the `broydn_workspace` module. This Rust
/// type makes that state explicit so callers can keep SCF iterations pure and
/// testable.
#[derive(Debug, Clone, PartialEq)]
pub struct BroydenWorkspace {
    /// FEFF `cmi(iteration, history)` coefficients.
    pub coefficients: Array2<Real>,
    /// FEFF `frho(radial, potential, iteration)` residual history.
    pub residuals: Array3<Real>,
    /// FEFF `urho(radial, potential, iteration)` multiplier history.
    pub multipliers: Array3<Real>,
    /// FEFF `xnorm(iteration)` normalization factors.
    pub norms: Array1<Real>,
    /// FEFF `wt(radial)` radial weights.
    pub weights: Array1<Real>,
    /// FEFF `rhoold(radial, potential)` previous overlapped valence density.
    pub previous_density: Array2<Real>,
    /// FEFF `ri05(radial)` radial grid.
    pub radii: Array1<Real>,
}

impl BroydenWorkspace {
    /// Allocate zero-initialized Broyden history for `max_iterations` and potentials.
    #[must_use]
    pub fn zeros(max_iterations: usize, potential_count: usize) -> Self {
        Self {
            coefficients: Array2::zeros((max_iterations, max_iterations)),
            residuals: Array3::zeros((OVRLP_DENSITY_POINTS, potential_count, max_iterations)),
            multipliers: Array3::zeros((OVRLP_DENSITY_POINTS, potential_count, max_iterations)),
            norms: Array1::zeros(max_iterations),
            weights: Array1::zeros(OVRLP_DENSITY_POINTS),
            previous_density: Array2::zeros((OVRLP_DENSITY_POINTS, potential_count)),
            radii: Array1::zeros(OVRLP_DENSITY_POINTS),
        }
    }
}

/// Inputs for FEFF `POT/broydn.f90` valence-density mixing.
#[derive(Debug, Clone, Copy)]
pub struct BroydenMixInput<'a> {
    /// One-based SCF iteration `iscmt`.
    pub iteration: usize,
    /// Convergence accelerator factor `ca`.
    pub accelerator: Real,
    /// Highest unique potential index; potentials `0..=highest_potential_index` are mixed.
    pub highest_potential_index: usize,
    /// Valence electron counts by `(l, potential)`, FEFF `xnvmu`.
    pub valence_occupancy: ArrayView2<'a, Real>,
    /// Last active radial index for each potential, FEFF `ilast`.
    pub last_indices: ArrayView1<'a, usize>,
    /// Multiplicity of each potential in the cluster, FEFF `xnatph`.
    pub potential_multiplicities: ArrayView1<'a, Real>,
    /// Norman radii, FEFF `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// Current charge inside each Norman sphere, FEFF `qnrm`.
    pub norman_charges: ArrayView1<'a, Real>,
    /// Previous overlapped valence density, FEFF `edenvl`.
    pub overlapped_valence_density: ArrayView2<'a, Real>,
    /// Newly integrated valence density, FEFF `rhoval` on input.
    pub valence_density: ArrayView2<'a, Real>,
    /// Persistent Broyden history from prior iterations.
    pub workspace: &'a BroydenWorkspace,
}

/// Result of one FEFF Broyden density-mixing iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct BroydenMix {
    /// Mixed valence density, FEFF `rhoval` on output.
    pub valence_density: Array2<Real>,
    /// Charge-transfer deltas, FEFF `dq`.
    pub charge_deltas: Array1<Real>,
    /// Updated Norman-sphere charges, FEFF `qnrm`.
    pub norman_charges: Array1<Real>,
    /// Updated persistent Broyden history.
    pub workspace: BroydenWorkspace,
}

/// FEFF `POT/coulom.f90` normalization branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoulombUpdateMode {
    /// FEFF default branch (`icoul != 1`) using Norman-sphere charge matching.
    Norman,
    /// FEFF long-range branch (`icoul == 1`) using explicit cluster charge deltas.
    LongRange,
}

/// Inputs for FEFF `POT/coulom.f90` Coulomb-potential correction.
#[derive(Debug, Clone, Copy)]
pub struct CoulombPotentialUpdateInput<'a> {
    /// Normalization branch matching FEFF `icoul`.
    pub mode: CoulombUpdateMode,
    /// Highest unique potential index; potentials `0..=highest_potential_index` are updated.
    pub highest_potential_index: usize,
    /// Last active radial index for each potential, FEFF `ilast`.
    pub last_indices: ArrayView1<'a, usize>,
    /// Valence density `rhoval(radial, potential)`.
    pub valence_density: ArrayView2<'a, Real>,
    /// Overlapped valence density `edenvl(radial, potential)`.
    pub overlapped_valence_density: ArrayView2<'a, Real>,
    /// Overlapped electron density `edens(radial, potential)`.
    pub overlapped_density: ArrayView2<'a, Real>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, FEFF `rat`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: ArrayView1<'a, usize>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Norman radii `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// Charge deltas `dq`.
    pub charge_deltas: ArrayView1<'a, Real>,
    /// Atomic numbers `iz`.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// Coulomb potential to correct, FEFF `vclap`.
    pub coulomb_potential: ArrayView2<'a, Real>,
}

/// Corrected FEFF Coulomb potentials from `coulom`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoulombPotentialUpdate {
    /// Updated Coulomb potential `vclap(radial, potential)`.
    pub coulomb_potential: Array2<Real>,
}

/// Inputs for one FEFF POT SCF density/coulomb update after valence integration.
#[derive(Debug, Clone, Copy)]
pub struct ScfDensityStepInput<'a> {
    /// One-based SCF iteration `iscmt`.
    pub iteration: usize,
    /// Convergence accelerator factor `ca`.
    pub accelerator: Real,
    /// Normalization branch matching FEFF `icoul`.
    pub coulomb_mode: CoulombUpdateMode,
    /// Highest unique potential index; potentials `0..=highest_potential_index` are updated.
    pub highest_potential_index: usize,
    /// Valence electron counts by `(l, potential)`, FEFF `xnvmu`.
    pub valence_occupancy: ArrayView2<'a, Real>,
    /// Last active radial index for each potential, FEFF `ilast`.
    pub last_indices: ArrayView1<'a, usize>,
    /// Multiplicity of each potential in the cluster, FEFF `xnatph`.
    pub potential_multiplicities: ArrayView1<'a, Real>,
    /// Norman radii, FEFF `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// Current charge inside each Norman sphere, FEFF `qnrm`.
    pub norman_charges: ArrayView1<'a, Real>,
    /// Previous overlapped valence density, FEFF `edenvl`.
    pub overlapped_valence_density: ArrayView2<'a, Real>,
    /// Newly integrated valence density from `ff2g`, FEFF `rhoval`.
    pub integrated_valence_density: ArrayView2<'a, Real>,
    /// Persistent Broyden history from prior iterations.
    pub workspace: &'a BroydenWorkspace,
    /// Overlapped electron density `edens(radial, potential)`.
    pub overlapped_density: ArrayView2<'a, Real>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, FEFF `rat`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: ArrayView1<'a, usize>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Atomic numbers `iz`.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// Coulomb potential to correct, FEFF `vclap`.
    pub coulomb_potential: ArrayView2<'a, Real>,
}

/// Result of one FEFF POT SCF density/coulomb update.
#[derive(Debug, Clone, PartialEq)]
pub struct ScfDensityStep {
    /// Mixed valence density, FEFF `rhoval` after `broydn`.
    pub valence_density: Array2<Real>,
    /// Charge-transfer deltas, FEFF `dq`.
    pub charge_deltas: Array1<Real>,
    /// Updated Norman-sphere charges, FEFF `qnrm`.
    pub norman_charges: Array1<Real>,
    /// Updated Coulomb potential, FEFF `vclap` after `coulom`.
    pub coulomb_potential: Array2<Real>,
    /// Updated persistent Broyden history.
    pub workspace: BroydenWorkspace,
}

/// Result kind for one source-backed FEFF POT SCF iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PotScfIterationStatus {
    /// The contour driver consumed supplied rows before finding the Fermi bracket.
    NeedsMoreSourcePoints,
    /// FEFF occupation-count consistency checks request a repeated iteration.
    RepeatRequired,
    /// The contour, Broyden mix, Coulomb update, and density update completed.
    Updated,
}

/// Inputs for one FEFF `POT/scmt.f90` source-backed SCF iteration.
#[derive(Debug, Clone, Copy)]
pub struct PotScfIterationInput<'a> {
    /// Source-backed contour integration input.
    pub contour: PotScfContourRunInput<'a>,
    /// One-based SCF iteration `iscmt`.
    pub iteration: usize,
    /// Convergence accelerator factor `ca`.
    pub accelerator: Real,
    /// Normalization branch matching FEFF `icoul`.
    pub coulomb_mode: CoulombUpdateMode,
    /// Whether bad occupation counts should stop this iteration for a repeat.
    pub repeat_on_bad_counts: bool,
    /// Expected valence occupation from `getorb`, FEFF `xnvmu`.
    pub expected_valence_occupancy: ArrayView2<'a, Real>,
    /// Norman radii, FEFF `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// Current charge inside each Norman sphere, FEFF `qnrm`.
    pub norman_charges: ArrayView1<'a, Real>,
    /// Previous overlapped valence density, FEFF `edenvl`.
    pub overlapped_valence_density: ArrayView2<'a, Real>,
    /// Persistent Broyden history from prior iterations.
    pub workspace: &'a BroydenWorkspace,
    /// Overlapped electron density, FEFF `edens`.
    pub overlapped_density: ArrayView2<'a, Real>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, FEFF `rat`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: ArrayView1<'a, usize>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Atomic numbers `iz`.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// Coulomb potential to correct, FEFF `vclap`.
    pub coulomb_potential: ArrayView2<'a, Real>,
}

/// Result of one source-backed FEFF POT SCF iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfIteration {
    /// Iteration status.
    pub status: PotScfIterationStatus,
    /// Contour integration result.
    pub contour: PotScfContourRun,
    /// Density/Coulomb update when [`PotScfIterationStatus::Updated`].
    pub density_step: Option<ScfDensityStep>,
    /// Number of occupation channels that exceeded FEFF's repeat thresholds.
    pub bad_occupation_count: usize,
    /// Updated overlapped electron density `edens`.
    pub overlapped_density: Array2<Real>,
    /// Updated overlapped valence density `edenvl` with inactive tails zero-filled.
    pub overlapped_valence_density: Array2<Real>,
}

/// Result kind for FEFF `POT/potsub.f90` after one SCF iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PotScfOuterIterationStatus {
    /// The contour driver consumed supplied rows before finding the Fermi bracket.
    NeedsMoreSourcePoints,
    /// FEFF occupation-count consistency checks request a repeated iteration.
    RepeatRequired,
    /// This iteration did not pass convergence and the caller should run `istprm` before the next one.
    NeedsNextIteration,
    /// Self-consistency passed and the SCF loop can exit cleanly.
    Converged,
    /// The configured maximum SCF iteration count was reached.
    ReachedIterationLimit,
}

/// Inputs for FEFF `POT/potsub.f90` outer SCF convergence/state transition.
#[derive(Debug, Clone, Copy)]
pub struct PotScfOuterIterationInput<'a> {
    /// Completed or partial source-backed SCF iteration.
    pub iteration_result: &'a PotScfIteration,
    /// One-based SCF iteration `iscmt`.
    pub iteration: usize,
    /// Maximum configured SCF iterations `nscmt`.
    pub max_iterations: usize,
    /// FEFF minimum iteration floor `nscmt_min`.
    pub minimum_iterations: usize,
    /// Previous Fermi level `xmu` before this iteration.
    pub previous_fermi_energy: Real,
    /// Previous Norman-sphere charges `qold`.
    pub previous_norman_charges: ArrayView1<'a, Real>,
    /// Previous angular occupations `xnmues_old`.
    pub previous_occupancy_by_l: ArrayView2<'a, Real>,
    /// Expected valence occupations `xnvmu`.
    pub expected_valence_occupancy: ArrayView2<'a, Real>,
    /// Ion charges `xion`, used for reported charge transfer.
    pub ion_charges: ArrayView1<'a, Real>,
    /// Coulomb potential saved before `scmt`, FEFF `vclapp`.
    pub previous_coulomb_potential: ArrayView2<'a, Real>,
    /// Fermi-level convergence tolerance `tolmu`.
    pub fermi_tolerance: Real,
    /// Norman charge convergence tolerance `tolq`.
    pub charge_tolerance: Real,
    /// Total valence charge consistency tolerance `tolsum`.
    pub charge_sum_tolerance: Real,
    /// Local partial valence charge convergence tolerance `tolqp`.
    pub partial_charge_tolerance: Real,
}

/// FEFF `POT/potsub.f90` outer SCF convergence/state transition result.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfOuterIteration {
    /// Transition status.
    pub status: PotScfOuterIterationStatus,
    /// Updated Fermi level `xmu`.
    pub fermi_energy: Real,
    /// Maximum Norman-charge movement this iteration.
    pub charge_distance: Real,
    /// Maximum partial-charge movement this iteration.
    pub partial_charge_distance: Real,
    /// Updated Norman-charge reference `qold` for the next iteration.
    pub norman_charge_reference: Array1<Real>,
    /// Reported charge transfer, FEFF `-qnrm + xion`.
    pub reported_charge_transfer: Array1<Real>,
    /// Overlapped electron density `edens` after FEFF's outer transition.
    pub overlapped_density: Array2<Real>,
    /// Overlapped valence density `edenvl` after FEFF's outer transition.
    pub overlapped_valence_density: Array2<Real>,
    /// Coulomb potential `vclap` after FEFF's outer transition.
    pub coulomb_potential: Array2<Real>,
}

/// Working state carried between FEFF POT SCF iterations.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfState {
    /// Current Fermi level `xmu`.
    pub fermi_energy: Real,
    /// Current Norman-sphere charges `qnrm`.
    pub norman_charges: Array1<Real>,
    /// Charge reference from the previous convergence test, FEFF `qold`.
    pub norman_charge_reference: Array1<Real>,
    /// Previous angular occupations `xnmues_old` for the next convergence test.
    pub occupancy_by_l: Array2<Real>,
    /// Current overlapped electron density `edens`.
    pub overlapped_density: Array2<Real>,
    /// Current overlapped valence density `edenvl`.
    pub overlapped_valence_density: Array2<Real>,
    /// Current Coulomb potential `vclap`.
    pub coulomb_potential: Array2<Real>,
    /// Persistent Broyden workspace.
    pub workspace: BroydenWorkspace,
}

/// Inputs for advancing one supplied FEFF POT SCF iteration from a working state.
#[derive(Debug, Clone, Copy)]
pub struct PotScfStateAdvanceInput<'a> {
    /// Source-backed contour integration input for this iteration.
    pub contour: PotScfContourRunInput<'a>,
    /// Working state before this SCF iteration.
    pub state: &'a PotScfState,
    /// One-based SCF iteration `iscmt`.
    pub iteration: usize,
    /// Maximum configured SCF iterations `nscmt`.
    pub max_iterations: usize,
    /// FEFF minimum iteration floor `nscmt_min`.
    pub minimum_iterations: usize,
    /// Convergence accelerator factor `ca`.
    pub accelerator: Real,
    /// Normalization branch matching FEFF `icoul`.
    pub coulomb_mode: CoulombUpdateMode,
    /// Whether bad occupation counts should stop this iteration for a repeat.
    pub repeat_on_bad_counts: bool,
    /// Expected valence occupations `xnvmu`.
    pub expected_valence_occupancy: ArrayView2<'a, Real>,
    /// Norman radii, FEFF `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// Ion charges `xion`, used for reported charge transfer.
    pub ion_charges: ArrayView1<'a, Real>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, FEFF `rat`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: ArrayView1<'a, usize>,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Atomic numbers `iz`.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// Fermi-level convergence tolerance `tolmu`.
    pub fermi_tolerance: Real,
    /// Norman charge convergence tolerance `tolq`.
    pub charge_tolerance: Real,
    /// Total valence charge consistency tolerance `tolsum`.
    pub charge_sum_tolerance: Real,
    /// Local partial valence charge convergence tolerance `tolqp`.
    pub partial_charge_tolerance: Real,
}

/// Result of advancing one supplied FEFF POT SCF iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfStateAdvance {
    /// Source-backed SCMT iteration result.
    pub iteration: PotScfIteration,
    /// FEFF `potsub` outer transition result.
    pub outer: PotScfOuterIteration,
    /// Working state after applying the outer transition.
    pub state: PotScfState,
}

/// One FEFF `POT/ovrlp.f90` explicit overlap contribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotentialOverlapNeighbor {
    /// Source potential index `infr` whose free-atom grids are overlapped in.
    pub source_potential: usize,
    /// Number of equivalent neighbors `ann`, from FEFF `nnovr`.
    pub multiplicity: Real,
    /// Neighbor distance `rnn`, in Bohr.
    pub distance: Real,
}

/// Inputs for overlapping one FEFF potential's Coulomb potential and densities.
#[derive(Debug, Clone, Copy)]
pub struct PotentialOverlapInput<'a> {
    /// Potential index `iph` to overlap.
    pub potential_index: usize,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, FEFF `rat`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: ArrayView1<'a, usize>,
    /// Atomic number for each potential, FEFF `iz`.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// Explicit overlap list for `potential_index`; empty means geometry mode.
    pub explicit_overlaps: &'a [PotentialOverlapNeighbor],
    /// Free-atom electron density `rho(radial, potential)`.
    pub electron_density: ArrayView2<'a, Real>,
    /// Free-atom spin-density or magnetization `dmag(radial, potential)`.
    pub spin_density: ArrayView2<'a, Real>,
    /// Free-atom valence density `rhoval(radial, potential)`.
    pub valence_density: ArrayView2<'a, Real>,
    /// Free-atom Coulomb potential `vcoul(radial, potential)`.
    pub coulomb_potential: ArrayView2<'a, Real>,
}

/// FEFF `ovrlp` output for one potential.
#[derive(Debug, Clone, PartialEq)]
pub struct PotentialOverlap {
    /// Overlapped Coulomb potential `vclap(:,iph)`.
    pub coulomb_potential: Array1<Real>,
    /// Overlapped electron density `edens(:,iph)`.
    pub electron_density: Array1<Real>,
    /// Overlapped valence-density accumulator `edenvl(:,iph)`.
    pub valence_density: Array1<Real>,
    /// FEFF `dmag(:,iph)` after conversion to `dmag / edens`.
    pub spin_density_ratio: Array1<Real>,
    /// Norman radius computed from the overlapped density.
    pub norman_radius: NormanRadius,
}

/// Inputs for FEFF `POT/ff2g.f90` valence-density accumulation.
#[derive(Debug, Clone, Copy)]
pub struct ValenceDensityUpdateInput<'a> {
    /// Single-precision FMS scattering trace `gtr(0:lx)`.
    pub scattering_trace: ArrayView1<'a, Complex32>,
    /// Zero-based potential column corresponding to FEFF `iph`.
    pub potential_index: usize,
    /// One-based energy index `ie`; `ie == 1` initializes previous-energy work arrays.
    pub energy_index: usize,
    /// One-based last radial point to update, FEFF `ilast`.
    pub last_radial_index: usize,
    /// Scattering contribution to angular-momentum LDOS, `xrhole(0:lx)`.
    pub scattering_ldos: ArrayView1<'a, Complex>,
    /// Embedded-atom LDOS table, indexed as `(l, potential)`, FEFF `xrhoce`.
    pub embedded_ldos: ArrayView2<'a, Complex>,
    /// Previous-energy LDOS table, indexed as `(l, potential)`, FEFF `xrhocp`.
    pub previous_ldos: ArrayView2<'a, Complex>,
    /// Scattering radial density table, indexed as `(radial, l)`, FEFF `yrhole`.
    pub scattering_density: ArrayView2<'a, Complex>,
    /// Embedded radial density for the current potential, FEFF `yrhoce`.
    pub embedded_density: ArrayView1<'a, Complex>,
    /// Previous-energy radial density, FEFF `yrhocp`.
    pub previous_density: ArrayView1<'a, Complex>,
    /// Energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: ArrayView1<'a, Real>,
    /// Energy-integrated electron count per angular momentum, FEFF `xnmues`.
    pub occupancy_by_l: ArrayView1<'a, Real>,
    /// Current complex energy `ee`.
    pub current_energy: Complex,
    /// Previous complex energy `ep`.
    pub previous_energy: Complex,
    /// Number of atoms with this potential type, FEFF `xnatph`.
    pub potential_multiplicity: Real,
    /// Current contour-floor flag `iflr`.
    pub current_floor: i32,
    /// Previous contour-floor flag `iflrp`.
    pub previous_floor: i32,
    /// Running left endpoint spectrum sum `fl`.
    pub left_sum: Complex,
    /// Running right endpoint spectrum sum `fr`.
    pub right_sum: Complex,
    /// Running total valence electron count `xntot`.
    pub total_electron_count: Real,
    /// Include f and higher states, FEFF `iunf != 0`.
    pub include_high_l: bool,
}

/// Updated FEFF `ff2g` density and LDOS state.
#[derive(Debug, Clone, PartialEq)]
pub struct ValenceDensityUpdate {
    /// Updated embedded-atom LDOS table, FEFF `xrhoce`.
    pub embedded_ldos: Array2<Complex>,
    /// Updated previous-energy LDOS table, FEFF `xrhocp`.
    pub previous_ldos: Array2<Complex>,
    /// Updated embedded radial density, FEFF `yrhoce`.
    pub embedded_density: Array1<Complex>,
    /// Updated previous-energy radial density, FEFF `yrhocp`.
    pub previous_density: Array1<Complex>,
    /// Updated energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: Array1<Real>,
    /// Updated angular-momentum electron counts, FEFF `xnmues`.
    pub occupancy_by_l: Array1<Real>,
    /// Updated left endpoint spectrum sum `fl`.
    pub left_sum: Complex,
    /// Updated right endpoint spectrum sum `fr`.
    pub right_sum: Complex,
    /// Updated total valence electron count `xntot`.
    pub total_electron_count: Real,
}

/// Inputs for FEFF `LDOS/ff2rho.f90` non-full-potential table assembly.
#[derive(Debug, Clone, Copy)]
pub struct LdosFf2rhoInput<'a> {
    /// Complex LDOS energy grid, FEFF `em`.
    pub energy_grid_hartree: ArrayView1<'a, Complex>,
    /// Embedded-atom LDOS, FEFF `xrhoce(l,ie)`.
    pub embedded_ldos: ArrayView2<'a, Real>,
    /// Scattering LDOS, FEFF `xrhole(l,ie)`.
    pub scattering_ldos: ArrayView2<'a, Complex>,
    /// FMS trace copied into FEFF `cchi(l,ie)`.
    pub scattering_trace: ArrayView2<'a, Complex>,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
    /// Whether FEFF applies the `msapp.ne.1` scattering correction.
    pub apply_scattering: bool,
}

/// Inputs for FEFF `LDOS/fmsdos.f90` non-full-potential trace projection.
#[derive(Debug, Clone, Copy)]
pub struct LdosFmsdosTraceInput<'a> {
    /// Packed FMS `gg(channel, channel, potential)` scattering matrices.
    pub scattering_matrices: ArrayView3<'a, Complex32>,
    /// Signed-`l` phase table `xphase(spin, -lx:lx, potential)`.
    pub phase_shifts: ArrayView3<'a, Complex32>,
    /// Spin row to read from `phase_shifts`.
    pub spin_index: usize,
    /// Number of ordinary angular-momentum channels, `l = 0..angular_count-1`.
    pub angular_count: usize,
}

/// Energy-grid inputs for FEFF `LDOS/fmsdos.f90` non-full-potential projection.
#[derive(Debug, Clone, Copy)]
pub struct LdosFmsdosTraceGridInput<'a> {
    /// Packed FMS `gg(energy, channel, channel, potential)` matrices.
    pub scattering_matrices: ArrayView4<'a, Complex32>,
    /// Signed-`l` phase table `xphase(energy, spin, -lx:lx, potential)`.
    pub phase_shifts: ArrayView4<'a, Complex32>,
    /// Spin row to read from `phase_shifts`.
    pub spin_index: usize,
    /// Number of ordinary angular-momentum channels, `l = 0..angular_count-1`.
    pub angular_count: usize,
}

/// Inputs for the post-radial-solver LDOS density integrals in FEFF `LDOS/rhol.f90`.
#[derive(Debug, Clone, Copy)]
pub struct LdosRholDensityInput<'a> {
    /// Radial grid `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// Regular large Dirac component, FEFF `pr`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Regular small Dirac component, FEFF `qr`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Irregular large Dirac component, FEFF `pn`.
    pub irregular_large: ArrayView1<'a, Complex>,
    /// Irregular small Dirac component, FEFF `qn`.
    pub irregular_small: ArrayView1<'a, Complex>,
    /// Log-grid step `dx`.
    pub radial_step: Real,
    /// Norman radius `rnrm`.
    pub norman_radius: Real,
    /// Complex momentum `ck`.
    pub wave_number: Complex,
    /// Angular momentum channel `lll`.
    pub angular_momentum: usize,
}

/// FEFF `rhol` angular-channel LDOS values for one energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LdosRholDensity {
    /// Scattering LDOS, FEFF `xrhole(l,ie)`.
    pub scattering_ldos: Complex,
    /// Embedded central-atom LDOS, FEFF `xrhoce(l,ie)`.
    pub embedded_ldos: Real,
    /// Combined FEFF normalization factor applied to the radial integrals.
    pub density_scale: Complex,
}

/// Energy-grid inputs for FEFF `LDOS/rhol.f90` post-solver density integrals.
#[derive(Debug, Clone, Copy)]
pub struct LdosRholDensityGridInput<'a> {
    /// Radial grid `ri` shared by all energy/angular slices.
    pub radii: ArrayView1<'a, Real>,
    /// Regular large Dirac components, shaped `(energy, angular, radial)`.
    pub regular_large: ArrayView3<'a, Complex>,
    /// Regular small Dirac components, shaped `(energy, angular, radial)`.
    pub regular_small: ArrayView3<'a, Complex>,
    /// Irregular large Dirac components, shaped `(energy, angular, radial)`.
    pub irregular_large: ArrayView3<'a, Complex>,
    /// Irregular small Dirac components, shaped `(energy, angular, radial)`.
    pub irregular_small: ArrayView3<'a, Complex>,
    /// Complex momenta `ck(energy)`.
    pub wave_numbers: ArrayView1<'a, Complex>,
    /// Log-grid step `dx`.
    pub radial_step: Real,
    /// Norman radius `rnrm`.
    pub norman_radius: Real,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
}

/// FEFF `rhol` LDOS work arrays for a full energy grid.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosRholDensityGrid {
    /// Scattering LDOS `xrhole(l,ie)`, shaped `(angular, energy)`.
    pub scattering_ldos: Array2<Complex>,
    /// Embedded central-atom LDOS `xrhoce(l,ie)`, shaped `(angular, energy)`.
    pub embedded_ldos: Array2<Real>,
    /// Per-channel normalization factors, shaped `(angular, energy)`.
    pub density_scale: Array2<Complex>,
}

/// Inputs for the post-radial-solver density work arrays in FEFF `POT/rholie.f90`.
#[derive(Debug, Clone, Copy)]
pub struct PotRholieDensityInput<'a> {
    /// Source Loucks radial grid `ri`.
    pub source_radii: ArrayView1<'a, Real>,
    /// Target 0.05-grid radii `ri05` used by POT.
    pub output_radii: ArrayView1<'a, Real>,
    /// Regular large Dirac component, FEFF `pr`.
    pub regular_large: ArrayView1<'a, Complex>,
    /// Regular small Dirac component, FEFF `qr`.
    pub regular_small: ArrayView1<'a, Complex>,
    /// Irregular large Dirac component, FEFF `pn`.
    pub irregular_large: ArrayView1<'a, Complex>,
    /// Irregular small Dirac component, FEFF `qn`.
    pub irregular_small: ArrayView1<'a, Complex>,
    /// Source log-grid step `dx`.
    pub radial_step: Real,
    /// Norman radius `rnrm`.
    pub norman_radius: Real,
    /// Complex momentum `ck`.
    pub wave_number: Complex,
    /// Angular momentum channel `lll`.
    pub angular_momentum: usize,
}

/// FEFF `POT/rholie.f90` density work arrays for one angular channel.
#[derive(Debug, Clone, PartialEq)]
pub struct PotRholieDensity {
    /// Scattering LDOS integral, FEFF `xrhole(l)`.
    pub scattering_ldos: Complex,
    /// Embedded-atom LDOS integral, FEFF `xrhoce(l)`.
    pub embedded_ldos: Complex,
    /// Scattering radial density, FEFF `yrhole(:,l)`.
    pub scattering_density: Array1<Complex>,
    /// Embedded radial-density contribution accumulated into FEFF `yrhoce`.
    pub embedded_density: Array1<Complex>,
    /// POT normalization factor applied to radial density terms.
    pub density_scale: Complex,
}

/// Inputs for assembling one FEFF `POT/rholie.f90` energy/potential slice.
#[derive(Debug, Clone, Copy)]
pub struct PotRholieDensityGridInput<'a> {
    /// Source Loucks radial grid `ri`.
    pub source_radii: ArrayView1<'a, Real>,
    /// Target 0.05-grid radii `ri05` used by POT.
    pub output_radii: ArrayView1<'a, Real>,
    /// Regular large Dirac components, shaped `(angular, source_radial)`.
    pub regular_large: ArrayView2<'a, Complex>,
    /// Regular small Dirac components, shaped `(angular, source_radial)`.
    pub regular_small: ArrayView2<'a, Complex>,
    /// Irregular large Dirac components, shaped `(angular, source_radial)`.
    pub irregular_large: ArrayView2<'a, Complex>,
    /// Irregular small Dirac components, shaped `(angular, source_radial)`.
    pub irregular_small: ArrayView2<'a, Complex>,
    /// Source log-grid step `dx`.
    pub radial_step: Real,
    /// Norman radius `rnrm`.
    pub norman_radius: Real,
    /// Complex momentum `ck`.
    pub wave_number: Complex,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
}

/// FEFF `POT/rholie.f90` work arrays for one energy/potential slice.
#[derive(Debug, Clone, PartialEq)]
pub struct PotRholieDensityGrid {
    /// Scattering LDOS `xrhole(l)`, shaped `(angular)`.
    pub scattering_ldos: Array1<Complex>,
    /// Embedded-atom LDOS `xrhoce(l)`, shaped `(angular)`.
    pub embedded_ldos: Array1<Complex>,
    /// Scattering radial density `yrhole(ir,l)`, shaped `(output_radial, angular)`.
    pub scattering_density: Array2<Complex>,
    /// Embedded radial density `yrhoce(ir)` accumulated over angular channels.
    pub embedded_density: Array1<Complex>,
    /// Per-channel POT normalization factors, shaped `(angular)`.
    pub density_scale: Array1<Complex>,
}

/// Inputs for building FEFF `POT/scmt.f90` source rows from radial/FMS work arrays.
#[derive(Debug, Clone, Copy)]
pub struct PotScfContourSourceRowsInput<'a> {
    /// Complex energy for each source row in FEFF loop order.
    pub source_energies: ArrayView1<'a, Complex>,
    /// Source Loucks radial grid `ri`.
    pub source_radii: ArrayView1<'a, Real>,
    /// Target 0.05-grid radii `ri05` used by POT.
    pub output_radii: ArrayView1<'a, Real>,
    /// Source log-grid step `dx`.
    pub radial_step: Real,
    /// Highest unique potential index; potentials `0..=highest_potential_index` are assembled.
    pub highest_potential_index: usize,
    /// Norman radius for each potential, FEFF `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// Complex momentum for each `(source_point, potential)`.
    pub wave_numbers: ArrayView2<'a, Complex>,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
    /// Single-precision FMS scattering trace, indexed as `(source_point, l, potential)`.
    pub scattering_trace: ArrayView3<'a, Complex32>,
    /// Regular large Dirac components, indexed as `(source_point, potential, l, source_radial)`.
    pub regular_large: ArrayView4<'a, Complex>,
    /// Regular small Dirac components, indexed as `(source_point, potential, l, source_radial)`.
    pub regular_small: ArrayView4<'a, Complex>,
    /// Irregular large Dirac components, indexed as `(source_point, potential, l, source_radial)`.
    pub irregular_large: ArrayView4<'a, Complex>,
    /// Irregular small Dirac components, indexed as `(source_point, potential, l, source_radial)`.
    pub irregular_small: ArrayView4<'a, Complex>,
}

/// FEFF `POT/scmt.f90` source rows ready for [`PotScfContourRunInput`].
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfContourSourceRows {
    /// Complex source-row energies in FEFF loop order.
    pub source_energies: Array1<Complex>,
    /// Single-precision FMS scattering trace, indexed as `(source_point, l, potential)`.
    pub scattering_trace: Array3<Complex32>,
    /// Scattering angular LDOS `xrhole`, indexed as `(source_point, l, potential)`.
    pub scattering_ldos: Array3<Complex>,
    /// Embedded angular LDOS `xrhoce` before FMS folding, indexed as `(source_point, l, potential)`.
    pub embedded_ldos_source: Array3<Complex>,
    /// Scattering radial density `yrhole`, indexed as `(source_point, radial, l, potential)`.
    pub scattering_density: Array4<Complex>,
    /// Embedded radial density `yrhoce` before FMS folding, indexed as `(source_point, radial, potential)`.
    pub embedded_density_source: Array3<Complex>,
    /// Per-channel POT normalization factors, indexed as `(source_point, l, potential)`.
    pub density_scale: Array3<Complex>,
}

/// Inputs for one FEFF `POT/scmt.f90` energy/potential density update.
#[derive(Debug, Clone, Copy)]
pub struct PotScfEnergyDensityInput<'a> {
    /// Source Loucks radial grid `ri`.
    pub source_radii: ArrayView1<'a, Real>,
    /// Target 0.05-grid radii `ri05` used by POT.
    pub output_radii: ArrayView1<'a, Real>,
    /// Regular large Dirac components, shaped `(angular, source_radial)`.
    pub regular_large: ArrayView2<'a, Complex>,
    /// Regular small Dirac components, shaped `(angular, source_radial)`.
    pub regular_small: ArrayView2<'a, Complex>,
    /// Irregular large Dirac components, shaped `(angular, source_radial)`.
    pub irregular_large: ArrayView2<'a, Complex>,
    /// Irregular small Dirac components, shaped `(angular, source_radial)`.
    pub irregular_small: ArrayView2<'a, Complex>,
    /// Source log-grid step `dx`.
    pub radial_step: Real,
    /// Norman radius `rnrm`.
    pub norman_radius: Real,
    /// Complex momentum `ck`.
    pub wave_number: Complex,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
    /// Single-precision FMS scattering trace `gtr(0:lx,iph)`.
    pub scattering_trace: ArrayView1<'a, Complex32>,
    /// Zero-based potential column corresponding to FEFF `iph`.
    pub potential_index: usize,
    /// One-based energy index `ie`; `ie == 1` initializes previous-energy work arrays.
    pub energy_index: usize,
    /// One-based last radial point to update, FEFF `nr05(iph)`.
    pub last_radial_index: usize,
    /// Embedded-atom LDOS table, indexed as `(l, potential)`, FEFF `xrhoce`.
    pub embedded_ldos: ArrayView2<'a, Complex>,
    /// Previous-energy LDOS table, indexed as `(l, potential)`, FEFF `xrhocp`.
    pub previous_ldos: ArrayView2<'a, Complex>,
    /// Previous-energy radial density, FEFF `yrhocp`.
    pub previous_density: ArrayView1<'a, Complex>,
    /// Energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: ArrayView1<'a, Real>,
    /// Energy-integrated electron count per angular momentum, FEFF `xnmues`.
    pub occupancy_by_l: ArrayView1<'a, Real>,
    /// Current complex energy `ee`.
    pub current_energy: Complex,
    /// Previous complex energy `ep`.
    pub previous_energy: Complex,
    /// Number of atoms with this potential type, FEFF `xnatph`.
    pub potential_multiplicity: Real,
    /// Current contour-floor flag `iflr`.
    pub current_floor: i32,
    /// Previous contour-floor flag `iflrp`.
    pub previous_floor: i32,
    /// Running left endpoint spectrum sum `fl`.
    pub left_sum: Complex,
    /// Running right endpoint spectrum sum `fr`.
    pub right_sum: Complex,
    /// Running total valence electron count `xntot`.
    pub total_electron_count: Real,
    /// Include f and higher states, FEFF `iunf != 0`.
    pub include_high_l: bool,
}

/// FEFF `POT/scmt.f90` one-energy density update from `rholie` through `ff2g`.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfEnergyDensity {
    /// Raw FEFF `rholie` work arrays before FMS trace folding.
    pub rholie: PotRholieDensityGrid,
    /// Updated FEFF `ff2g` valence-density state.
    pub valence: ValenceDensityUpdate,
}

/// Inputs for the FEFF `POT/scmt.f90` per-energy `ff2g` potential loop.
#[derive(Debug, Clone, Copy)]
pub struct PotScfEnergyPointInput<'a> {
    /// One-based energy index `ie`.
    pub energy_index: usize,
    /// Current complex energy `ee`.
    pub current_energy: Complex,
    /// Previous complex energy `ep`.
    pub previous_energy: Complex,
    /// Current contour floor `iflr`.
    pub current_floor: i32,
    /// Previous contour floor `iflrp`.
    pub previous_floor: i32,
    /// Highest unique potential index; potentials `0..=highest_potential_index` are updated.
    pub highest_potential_index: usize,
    /// One-based last radial point for each potential, FEFF `nr05`.
    pub last_indices: ArrayView1<'a, usize>,
    /// Multiplicity of each potential in the cluster, FEFF `xnatph`.
    pub potential_multiplicities: ArrayView1<'a, Real>,
    /// Single-precision FMS scattering trace `gtr(l,iph)`.
    pub scattering_trace: ArrayView2<'a, Complex32>,
    /// Scattering contribution to angular-momentum LDOS, `xrhole(l,iph)`.
    pub scattering_ldos: ArrayView2<'a, Complex>,
    /// Current embedded-atom LDOS table, indexed as `(l, potential)`, FEFF `xrhoce`.
    pub embedded_ldos: ArrayView2<'a, Complex>,
    /// Previous-energy LDOS table, indexed as `(l, potential)`, FEFF `xrhocp`.
    pub previous_ldos: ArrayView2<'a, Complex>,
    /// Scattering radial density table, indexed as `(radial, l, potential)`, FEFF `yrhole`.
    pub scattering_density: ArrayView3<'a, Complex>,
    /// Current embedded radial density table, indexed as `(radial, potential)`, FEFF `yrhoce`.
    pub embedded_density: ArrayView2<'a, Complex>,
    /// Previous embedded radial density table, indexed as `(radial, potential)`, FEFF `yrhocp`.
    pub previous_density: ArrayView2<'a, Complex>,
    /// Energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: ArrayView2<'a, Real>,
    /// Energy-integrated electron count per `(l, potential)`, FEFF `xnmues`.
    pub occupancy_by_l: ArrayView2<'a, Real>,
    /// Include f and higher states, FEFF `iunf != 0`.
    pub include_high_l: bool,
}

/// FEFF `POT/scmt.f90` per-energy accumulation result.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfEnergyPoint {
    /// Updated embedded-atom LDOS table, FEFF `xrhoce`.
    pub embedded_ldos: Array2<Complex>,
    /// Updated previous-energy LDOS table, FEFF `xrhocp`.
    pub previous_ldos: Array2<Complex>,
    /// Updated embedded radial density, FEFF `yrhoce`.
    pub embedded_density: Array2<Complex>,
    /// Updated previous-energy radial density, FEFF `yrhocp`.
    pub previous_density: Array2<Complex>,
    /// Updated energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: Array2<Real>,
    /// Updated angular-momentum electron counts, FEFF `xnmues`.
    pub occupancy_by_l: Array2<Real>,
    /// FEFF `xntot` after summing all potentials for this energy point.
    pub total_electron_count: Real,
    /// Accumulated left endpoint sum `fl`.
    pub left_sum: Complex,
    /// Accumulated right endpoint sum `fr`.
    pub right_sum: Complex,
}

/// Result kind for a source-backed FEFF `POT/scmt.f90` contour run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PotScfContourRunStatus {
    /// The lowest-floor Fermi bracket was found and endpoint corrections were applied.
    Bracketed,
    /// The supplied source work arrays were consumed before a bracket was found.
    NeedsMoreSourcePoints,
}

/// Inputs for the FEFF `POT/scmt.f90` contour loop over source-backed work arrays.
#[derive(Debug, Clone, Copy)]
pub struct PotScfContourRunInput<'a> {
    /// Whether this is the first `scmt` call, matching FEFF `ient == 1`.
    pub first_scmt_call: bool,
    /// Target valence electron count `xnferm`.
    pub electron_count_target: Real,
    /// Active FEFF contour grid length `neg`.
    pub active_energy_count: usize,
    /// FEFF floor count `nflrx`, also the active length of `steps`.
    pub floor_count: usize,
    /// Prebuilt FEFF `emg` contour grid.
    pub energy_grid: ArrayView1<'a, Complex>,
    /// FEFF floor steps `step(1:nflrx)`.
    pub steps: ArrayView1<'a, Real>,
    /// Complex energy for each supplied source row, in loop order.
    pub source_energies: ArrayView1<'a, Complex>,
    /// Highest unique potential index; potentials `0..=highest_potential_index` are updated.
    pub highest_potential_index: usize,
    /// One-based last radial point for each potential, FEFF `nr05`.
    pub last_indices: ArrayView1<'a, usize>,
    /// Multiplicity of each potential in the cluster, FEFF `xnatph`.
    pub potential_multiplicities: ArrayView1<'a, Real>,
    /// Single-precision FMS scattering traces, indexed as `(source_point, l, potential)`.
    pub scattering_trace: ArrayView3<'a, Complex32>,
    /// Scattering angular LDOS from `rholie`, indexed as `(source_point, l, potential)`.
    pub scattering_ldos: ArrayView3<'a, Complex>,
    /// Embedded LDOS from `rholie` before FMS folding, indexed as `(source_point, l, potential)`.
    pub embedded_ldos_source: ArrayView3<'a, Complex>,
    /// Scattering radial density from `rholie`, indexed as `(source_point, radial, l, potential)`.
    pub scattering_density: ArrayView4<'a, Complex>,
    /// Embedded radial density from `rholie`, indexed as `(source_point, radial, potential)`.
    pub embedded_density_source: ArrayView3<'a, Complex>,
    /// Include f and higher states, FEFF `iunf != 0`.
    pub include_high_l: bool,
}

/// Source-backed FEFF `POT/scmt.f90` contour-loop result.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfContourRun {
    /// Whether the run reached a Fermi bracket or needs more source rows.
    pub status: PotScfContourRunStatus,
    /// Number of source energy rows consumed.
    pub energy_points_used: usize,
    /// Current complex energy `ee`, or the next requested energy when more source rows are needed.
    pub current_energy: Complex,
    /// Previous complex energy `ep`.
    pub previous_energy: Complex,
    /// Current contour floor `iflr`, one-based.
    pub current_floor: usize,
    /// Previous contour floor `iflrp`, one-based.
    pub previous_floor: usize,
    /// Horizontal search direction `idir`.
    pub direction: i32,
    /// Whether FEFF allows a later upward floor move, `upok`.
    pub can_step_up: bool,
    /// Last electron-count residual `xndif`.
    pub current_electron_delta: Real,
    /// Previous electron-count residual `xndifp`.
    pub previous_electron_delta: Real,
    /// Last FEFF `xntot` after summing all potentials.
    pub total_electron_count: Real,
    /// Last accumulated left endpoint sum `fl`.
    pub left_sum: Complex,
    /// Last accumulated right endpoint sum `fr`.
    pub right_sum: Complex,
    /// New Fermi level `xmunew` when bracketed.
    pub fermi_energy: Option<Real>,
    /// Endpoint interpolation fraction `a` when bracketed.
    pub interpolation_fraction: Option<Real>,
    /// Updated embedded-atom LDOS table, FEFF `xrhoce`.
    pub embedded_ldos: Array2<Complex>,
    /// Updated previous-energy LDOS table, FEFF `xrhocp`.
    pub previous_ldos: Array2<Complex>,
    /// Updated embedded radial density, FEFF `yrhoce`.
    pub embedded_density: Array2<Complex>,
    /// Updated previous-energy radial density, FEFF `yrhocp`.
    pub previous_density: Array2<Complex>,
    /// Energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: Array2<Real>,
    /// Energy-integrated electron count per `(l, potential)`, FEFF `xnmues`.
    pub occupancy_by_l: Array2<Real>,
}

/// Inputs for FEFF `POT/scmt.f90` Fermi end-cap correction after contour bracketing.
#[derive(Debug, Clone, Copy)]
pub struct PotScfFermiEndpointInput<'a> {
    /// Current complex energy `ee`.
    pub current_energy: Complex,
    /// Previous complex energy `ep`.
    pub previous_energy: Complex,
    /// Current electron-count residual `xndif = xntot - xnferm`.
    pub current_electron_delta: Real,
    /// Previous electron-count residual `xndifp`.
    pub previous_electron_delta: Real,
    /// Previous endpoint global spectrum sum `fl`.
    pub left_sum: Complex,
    /// Current endpoint global spectrum sum `fr`.
    pub right_sum: Complex,
    /// Highest unique potential index; potentials `0..=highest_potential_index` are corrected.
    pub highest_potential_index: usize,
    /// One-based last radial point to correct for each potential, FEFF `nr05`.
    pub last_indices: ArrayView1<'a, usize>,
    /// Current embedded-atom LDOS table, indexed as `(l, potential)`, FEFF `xrhoce`.
    pub embedded_ldos: ArrayView2<'a, Complex>,
    /// Previous-energy LDOS table, indexed as `(l, potential)`, FEFF `xrhocp`.
    pub previous_ldos: ArrayView2<'a, Complex>,
    /// Current embedded radial density table, indexed as `(radial, potential)`, FEFF `yrhoce`.
    pub embedded_density: ArrayView2<'a, Complex>,
    /// Previous embedded radial density table, indexed as `(radial, potential)`, FEFF `yrhocp`.
    pub previous_density: ArrayView2<'a, Complex>,
    /// Energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: ArrayView2<'a, Real>,
    /// Energy-integrated electron count per `(l, potential)`, FEFF `xnmues`.
    pub occupancy_by_l: ArrayView2<'a, Real>,
    /// Include f and higher states, FEFF `iunf != 0`.
    pub include_high_l: bool,
}

/// FEFF `POT/scmt.f90` Fermi end-cap correction result.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfFermiEndpoint {
    /// New Fermi level `xmunew`.
    pub fermi_energy: Real,
    /// FEFF interpolation fraction `a` along `ee` to `ep`.
    pub interpolation_fraction: Real,
    /// Corrected energy-integrated valence radial density, FEFF `rhoval`.
    pub valence_density: Array2<Real>,
    /// Corrected angular-momentum electron counts, FEFF `xnmues`.
    pub occupancy_by_l: Array2<Real>,
}

/// FEFF `POT/scmt.f90` contour search transition result kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PotScfContourStepStatus {
    /// Continue the SCMT energy loop at the returned current energy.
    Continue,
    /// The lowest-floor bracket is found; run the Fermi endpoint correction.
    Bracketed,
}

/// Inputs for one FEFF `POT/scmt.f90` contour-search transition.
#[derive(Debug, Clone, Copy)]
pub struct PotScfContourStepInput<'a> {
    /// Whether this is the first `scmt` call, matching FEFF `ient == 1`.
    pub first_scmt_call: bool,
    /// One-based current energy index `ie`.
    pub energy_index: usize,
    /// Active FEFF contour grid length `neg`.
    pub active_energy_count: usize,
    /// FEFF floor count `nflrx`, also the active length of `steps`.
    pub floor_count: usize,
    /// Prebuilt FEFF `emg` contour grid.
    pub energy_grid: ArrayView1<'a, Complex>,
    /// FEFF floor steps `step(1:nflrx)`.
    pub steps: ArrayView1<'a, Real>,
    /// Current complex energy `ee`.
    pub current_energy: Complex,
    /// Previous complex energy `ep`.
    pub previous_energy: Complex,
    /// Current contour floor `iflr`, one-based.
    pub current_floor: usize,
    /// Previous contour floor `iflrp`, one-based.
    pub previous_floor: usize,
    /// Current horizontal direction `idir`; FEFF uses `-1` or `1`.
    pub direction: i32,
    /// Whether FEFF allows a later upward floor move, `upok`.
    pub can_step_up: bool,
    /// Current electron-count residual `xndif = xntot - xnferm`.
    pub current_electron_delta: Real,
    /// Previous electron-count residual `xndifp`.
    pub previous_electron_delta: Real,
}

/// Output from one FEFF `POT/scmt.f90` contour-search transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotScfContourStep {
    /// Whether the caller should continue the energy loop or finish the Fermi endpoint.
    pub status: PotScfContourStepStatus,
    /// Updated previous complex energy `ep`.
    pub previous_energy: Complex,
    /// Updated current complex energy `ee`.
    pub current_energy: Complex,
    /// Updated contour floor `iflr`, one-based.
    pub current_floor: usize,
    /// Updated previous contour floor `iflrp`, one-based.
    pub previous_floor: usize,
    /// Updated horizontal direction `idir`.
    pub direction: i32,
    /// Updated upward-floor allowance `upok`.
    pub can_step_up: bool,
}

/// Inputs for the exact free-particle radial tail in FEFF `LDOS/rhol.f90`.
#[derive(Debug, Clone, Copy)]
pub struct LdosRholExactRadialTailInput<'a> {
    /// Radial grid `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// FEFF one-based first row overwritten by the exact tail, `jri`.
    pub start_index_1based: usize,
    /// Angular momentum channel `lll`.
    pub angular_momentum: usize,
    /// Complex phase shift `phx`.
    pub phase_shift: Complex,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
}

/// Exact free-particle radial rows for FEFF `rhol` samples `jri:ilast`.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosRholExactRadialTail {
    /// FEFF one-based first row represented by the returned arrays.
    pub start_index_1based: usize,
    /// Regular large component `pr(i)`.
    pub regular_large: Array1<Complex>,
    /// Regular small component `qr(i)`.
    pub regular_small: Array1<Complex>,
    /// Irregular large component `pn(i)`.
    pub irregular_large: Array1<Complex>,
    /// Irregular small component `qn(i)`.
    pub irregular_small: Array1<Complex>,
}

impl LdosRholExactRadialTail {
    /// Number of radial rows represented by this tail.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.regular_large.len()
    }
}

/// Raw `dfovrg` outputs and matching data for FEFF `LDOS/rhol.f90`.
#[derive(Debug, Clone, Copy)]
pub struct LdosRholRadialAssemblyInput<'a> {
    /// Radial grid `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// Raw regular large component from the regular `dfovrg` pass, FEFF `pn`.
    pub raw_regular_large: ArrayView1<'a, Complex>,
    /// Raw regular small component from the regular `dfovrg` pass, FEFF `qn`.
    pub raw_regular_small: ArrayView1<'a, Complex>,
    /// Raw irregular large component from the irregular `dfovrg` pass, FEFF `pn`.
    pub raw_irregular_large: ArrayView1<'a, Complex>,
    /// Raw irregular small component from the irregular `dfovrg` pass, FEFF `qn`.
    pub raw_irregular_small: ArrayView1<'a, Complex>,
    /// Complex phase shift `phx` returned by FEFF `phamp`.
    pub phase_shift: Complex,
    /// FEFF `temp`, the phase amplitude returned by `phamp`.
    pub phase_amplitude: Complex,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
    /// Angular momentum channel `lll`.
    pub angular_momentum: usize,
    /// FEFF one-based Wronskian match row, `jri`.
    pub match_index_1based: usize,
    /// FEFF one-based first exact-tail row, `jri`.
    pub exact_tail_start_index_1based: usize,
}

/// Normalized radial components used by FEFF `LDOS/rhol.f90` density integrals.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosRholRadialAssembly {
    /// FEFF `xfnorm = 1 / temp`.
    pub regular_solution_scale: Complex,
    /// FEFF `exp(i*phx)` used in the irregular transform.
    pub irregular_phase_factor: Complex,
    /// FEFF reciprocal Wronskian wave scale.
    pub irregular_reciprocal_wave_scale: Complex,
    /// Normalized regular large component `pr`.
    pub regular_large: Array1<Complex>,
    /// Normalized regular small component `qr`.
    pub regular_small: Array1<Complex>,
    /// Transformed irregular large component `pn`.
    pub irregular_large: Array1<Complex>,
    /// Transformed irregular small component `qn`.
    pub irregular_small: Array1<Complex>,
}

impl LdosRholRadialAssembly {
    /// Number of radial rows represented by this assembly.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.regular_large.len()
    }
}

/// Inputs for one FEFF `LDOS/rhol.f90` regular/irregular radial channel.
#[derive(Debug, Clone, Copy)]
pub struct LdosRholChannelInput<'a> {
    /// Base FOVRG solver input. LDOS overwrites the `irregular` flag and
    /// muffin-tin initial values for the regular and irregular passes.
    pub solver: FovrgDiracSolverInput<'a>,
    /// Angular momentum channel `lll`.
    pub angular_momentum: usize,
    /// Relativistic wave number `ck`.
    pub wave_number: Complex,
}

/// Solved FEFF `rhol` radial channel for one energy and angular momentum.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosRholChannel {
    /// Complex phase shift `phx` returned by `phamp`.
    pub phase_shift: Complex,
    /// FEFF `temp`, the phase amplitude returned by `phamp`.
    pub phase_amplitude: Complex,
    /// Irregular `dfovrg` input large component at the muffin-tin radius.
    pub irregular_initial_large: Complex,
    /// Irregular `dfovrg` input small component at the muffin-tin radius.
    pub irregular_initial_small: Complex,
    /// Final normalized regular/irregular radial components for LDOS.
    pub radial_components: LdosRholRadialAssembly,
    /// Active radial row count from the regular FOVRG pass.
    pub regular_active_len: usize,
    /// Active radial row count from the irregular FOVRG pass.
    pub irregular_active_len: usize,
    /// Regular FOVRG nonlocal-exchange iteration count.
    pub regular_iteration_count: usize,
    /// Irregular FOVRG nonlocal-exchange iteration count.
    pub irregular_iteration_count: usize,
    /// Total difficult Milne iterations reported by both FOVRG passes.
    pub difficult_iterations: usize,
}

/// Inputs for a source-backed non-spin FEFF `LDOS/rhol.f90` table driver.
#[derive(Debug, Clone, Copy)]
pub struct LdosRholTableDriverInput<'a> {
    /// Per-channel FOVRG inputs in FEFF loop order `(energy, angular)`.
    ///
    /// Each solver carries the prepared source potential, radial mesh, bound
    /// orbitals, and channel kappa. The driver overwrites only the regular /
    /// irregular FOVRG controls already handled by [`LdosRholChannelInput`].
    pub solvers: &'a [FovrgDiracSolverInput<'a>],
    /// Complex LDOS energy grid, FEFF `em`.
    pub energy_grid_hartree: ArrayView1<'a, Complex>,
    /// Relativistic wave numbers `ck(energy)`.
    pub wave_numbers: ArrayView1<'a, Complex>,
    /// FMS trace copied into FEFF `cchi(l,ie)`.
    pub scattering_trace: ArrayView2<'a, Complex>,
    /// Log-grid step `dx`.
    pub radial_step: Real,
    /// Norman radius `rnrm`.
    pub norman_radius: Real,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
    /// Whether FEFF applies the `msapp.ne.1` scattering correction.
    pub apply_scattering: bool,
}

/// Source-backed non-spin FEFF `LDOS/rhol.f90` and `ff2rho` outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosRholTableDriver {
    /// Final `ff2rho` table payloads.
    pub tables: LdosFf2rhoTables,
    /// Intermediate `rhol` density work arrays.
    pub density_grid: LdosRholDensityGrid,
    /// Complex phase shifts `phx(l,ie)`, shaped `(angular, energy)`.
    pub phase_shifts: Array2<Complex>,
    /// FEFF `temp(l,ie)` phase amplitudes, shaped `(angular, energy)`.
    pub phase_amplitudes: Array2<Complex>,
    /// Regular FOVRG iteration counts, shaped `(angular, energy)`.
    pub regular_iteration_counts: Array2<usize>,
    /// Irregular FOVRG iteration counts, shaped `(angular, energy)`.
    pub irregular_iteration_counts: Array2<usize>,
    /// Total difficult Milne iterations per channel, shaped `(angular, energy)`.
    pub difficult_iterations: Array2<usize>,
}

/// Inputs for a non-spin LDOS table assembled from source RHORRP wavefunctions.
#[derive(Debug, Clone, Copy)]
pub struct LdosRholWavefunctionTablesInput<'a> {
    /// All-potential source-backed wavefunctions from RHORRP `init_wavefunctions`.
    pub wavefunctions: &'a RhorrpWavefunctionTables,
    /// Shared radial grid `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// FEFF potential index `iph`.
    pub potential_index: usize,
    /// Complex LDOS energy grid, FEFF `em`.
    pub energy_grid_hartree: ArrayView1<'a, Complex>,
    /// FMS trace copied into FEFF `cchi(l,ie)`.
    pub scattering_trace: ArrayView2<'a, Complex>,
    /// Log-grid step `dx`.
    pub radial_step: Real,
    /// Norman radius `rnrm(iph)`.
    pub norman_radius: Real,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
    /// Whether FEFF applies the `msapp.ne.1` scattering correction.
    pub apply_scattering: bool,
}

/// Non-spin LDOS table payloads assembled from source RHORRP wavefunctions.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosRholWavefunctionTables {
    /// Final `ff2rho` table payloads.
    pub tables: LdosFf2rhoTables,
    /// Intermediate `rhol` density work arrays.
    pub density_grid: LdosRholDensityGrid,
    /// Relativistic wave numbers `ck(energy)` selected for this potential.
    pub wave_numbers: Array1<Complex>,
}

/// FEFF `ff2rho` table payloads before text rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosFf2rhoTables {
    /// Real energy grid in eV, FEFF `dble(em(ik))*hart`.
    pub energy_ev: Array1<Real>,
    /// Final LDOS table written to `ldosNN.dat`, shaped `(energy, angular)`.
    pub ldos_density: Array2<Real>,
    /// Embedded-density table written to `rhocNN.dat`, shaped `(energy, angular)`.
    pub rhoc_density: Array2<Real>,
}

/// Inputs for FEFF `LDOS/ff2rho_h.f90` spin-resolved table assembly.
#[derive(Debug, Clone, Copy)]
pub struct LdosSpinFf2rhoInput<'a> {
    /// Complex LDOS energy grid, FEFF `em`.
    pub energy_grid_hartree: ArrayView1<'a, Complex>,
    /// Embedded-atom LDOS, FEFF `xrhoce(l,is,ie)`.
    pub embedded_ldos: ArrayView3<'a, Real>,
    /// Scattering LDOS, FEFF `xrhole(l,is,ie)`.
    pub scattering_ldos: ArrayView3<'a, Complex>,
    /// FMS trace copied into FEFF `cchi(l,is,ie)`.
    pub scattering_trace: ArrayView3<'a, Complex>,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
    /// Whether FEFF applies the `msapp.ne.1` scattering correction.
    pub apply_scattering: bool,
}

/// FEFF spin-resolved `ff2rho_h` table payloads before text rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosSpinFf2rhoTables {
    /// Real energy grid in eV, FEFF `dble(em(ik))*hart`.
    pub energy_ev: Array1<Real>,
    /// Final spin-resolved LDOS table, shaped `(energy, 2 * angular)`.
    pub ldos_density: Array2<Real>,
    /// Embedded spin-resolved density table, shaped `(energy, 2 * angular)`.
    pub rhoc_density: Array2<Real>,
}

/// Inputs for FEFF `LDOS/ff2rho_h_step2.f90` magnetic-orbital table assembly.
#[derive(Debug, Clone, Copy)]
pub struct LdosHubbardMagneticFf2rhoInput<'a> {
    /// Complex LDOS energy grid, FEFF `em`.
    pub energy_grid_hartree: ArrayView1<'a, Complex>,
    /// Embedded magnetic LDOS, FEFF `xmrhoce(l,im,is,ie)`.
    pub embedded_magnetic_ldos: ArrayView4<'a, Real>,
    /// Scattering magnetic LDOS, FEFF `xmrhole(l,im,is,ie)`.
    pub scattering_magnetic_ldos: ArrayView4<'a, Complex>,
    /// Magnetic FMS trace, FEFF `gtr_m(l,im,is,iph,ie)` after selecting `iph`.
    pub magnetic_scattering_trace: ArrayView4<'a, Complex>,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
}

/// FEFF magnetic-orbital `ff2rho_h_step2` table payloads before text rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosHubbardMagneticFf2rhoTables {
    /// Real energy grid in eV, FEFF `dble(em(ik))*hart`.
    pub energy_ev: Array1<Real>,
    /// Final magnetic LDOS table written to `lmdosNN.dat`, shaped `(energy, 2 * (lx + 1)^2)`.
    pub lmdos_density: Array2<Real>,
    /// Embedded magnetic density table written to `rhocmNN.dat`, shaped `(energy, 2 * (lx + 1)^2)`.
    pub rhocm_density: Array2<Real>,
}

/// Inputs for the first pass of FEFF `LDOS/ff2rho_h_step1.f90`.
///
/// The first Hubbard pass converts ordinary spin-resolved radial densities and
/// diagonal/off-diagonal FMS traces into magnetic-orbital occupations. For the
/// active Hubbard potential it diagonalizes the occupation matrix and builds
/// the per-orbital potential shifts and basis transforms used by the second
/// radial/FMS pass.
#[derive(Debug, Clone, Copy)]
pub struct LdosHubbardStep1Input<'a> {
    /// Complex LDOS energy grid, FEFF `em`.
    pub energy_grid_hartree: ArrayView1<'a, Complex>,
    /// Embedded ordinary spin LDOS, FEFF `xrhoce(l,is,ie)`.
    pub embedded_ldos: ArrayView3<'a, Real>,
    /// Scattering ordinary spin LDOS, FEFF `xrhole(l,is,ie)`.
    pub scattering_ldos: ArrayView3<'a, Complex>,
    /// Diagonal magnetic FMS trace, FEFF `gtr_m(l,im,is,iph,ie)` after selecting `iph`.
    pub magnetic_scattering_trace: ArrayView4<'a, Complex>,
    /// Off-diagonal Hubbard trace, FEFF `gtr_off(l,im1,im2,is,iph,ie)` after selecting `iph`.
    pub off_diagonal_scattering_trace: ArrayView5<'a, Complex>,
    /// Chemical potential in Hartree, FEFF `xmu`.
    pub chemical_potential_hartree: Real,
    /// Hubbard Fermi shift in eV, FEFF `fermi_shift`.
    pub fermi_shift_ev: Real,
    /// Hubbard U in eV, FEFF `U_hubbard`.
    pub hubbard_u_ev: Real,
    /// Hubbard J in eV, FEFF `J_hubbard`.
    pub hubbard_j_ev: Real,
    /// Active Hubbard angular momentum, FEFF `l_hubbard`.
    pub hubbard_l: usize,
    /// Selected FEFF potential index, `iph`.
    pub potential_index: usize,
    /// Active angular-momentum channel count, FEFF `lx + 1`.
    pub angular_count: usize,
}

/// First-pass FEFF Hubbard LDOS work arrays for one potential.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosHubbardStep1 {
    /// First-pass magnetic embedded density, FEFF `xmrhoce(l,im,is,ie)`.
    pub embedded_magnetic_ldos: Array4<Real>,
    /// Integrated and, for potential 1, diagonalized occupations as `(l,im,spin)`.
    pub occupations: Array3<Real>,
    /// Hubbard shifts as `(spin,l,im)`, ready for one `v_hubbard.bin` potential block.
    pub hubbard_potential: Array3<Real>,
    /// FEFF `TFrm` as `(spin,l,row,column)`.
    pub transform: Array4<Complex>,
    /// FEFF `TFrmInv` as `(spin,l,row,column)`.
    pub inverse_transform: Array4<Complex>,
}

/// Error returned by density accumulation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum DensityError {
    /// FEFF indices that select a point count or energy point are 1-based.
    #[error("{name} must be 1-based and positive, got {index}")]
    InvalidIndex { name: &'static str, index: usize },
    /// A vector must contain enough values for the FEFF loop bounds.
    #[error("{name} length {actual} is shorter than required length {required}")]
    LengthTooShort {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// A pair of angular-momentum vectors must have matching lengths.
    #[error("{left_name} length {left_len} does not match {right_name} length {right_len}")]
    LengthMismatch {
        left_name: &'static str,
        left_len: usize,
        right_name: &'static str,
        right_len: usize,
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
    /// A 3-D table must have enough rows, columns, and pages for the FEFF loop bounds.
    #[error(
        "{name} shape ({rows},{columns},{depth}) is smaller than required ({required_rows},{required_columns},{required_depth})"
    )]
    CubeShapeTooSmall {
        name: &'static str,
        rows: usize,
        columns: usize,
        depth: usize,
        required_rows: usize,
        required_columns: usize,
        required_depth: usize,
    },
    /// A real scalar must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteScalar { name: &'static str, value: Real },
    /// A real scalar must be positive and finite.
    #[error("{name} must be positive and finite, got {value}")]
    NonPositiveScalar { name: &'static str, value: Real },
    /// A real scalar denominator must be finite and nonzero.
    #[error("{name} must be finite and nonzero, got {value}")]
    ZeroScalar { name: &'static str, value: Real },
    /// A complex scalar must have finite components.
    #[error("{name} must be finite, got ({real},{imaginary})")]
    NonFiniteComplex {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// A real vector entry must be finite.
    #[error("{name}[{index}] must be finite, got {value}")]
    NonFiniteValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// A complex vector or matrix entry must have finite components.
    #[error("{name}[{index}] must be finite, got ({real},{imaginary})")]
    NonFiniteComplexValue {
        name: &'static str,
        index: usize,
        real: Real,
        imaginary: Real,
    },
    /// A supplied SCMT source row does not match the generated contour energy.
    #[error(
        "pot_scf_contour_run_source_energies[{index}] is ({actual_real},{actual_imaginary}), expected ({expected_real},{expected_imaginary})"
    )]
    ContourEnergyMismatch {
        index: usize,
        expected_real: Real,
        expected_imaginary: Real,
        actual_real: Real,
        actual_imaginary: Real,
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
    /// FEFF radial-grid overlap or Norman-radius calculation failed.
    #[error(transparent)]
    Grid(#[from] GridError),
    /// FEFF radial quadrature failed.
    #[error(transparent)]
    Quadrature(#[from] QuadratureError),
    /// FEFF radial interpolation failed.
    #[error(transparent)]
    Interpolation(#[from] InterpolationError),
    /// FEFF radial solver failed.
    #[error(transparent)]
    Fovrg(#[from] FovrgError),
    /// Shared RHORRP radial wavefunction helper failed.
    #[error(transparent)]
    Rhorrp(#[from] RhorrpError),
    /// A Hubbard occupation-matrix eigensystem failed.
    #[error(transparent)]
    Linalg(#[from] refeff_linalg::LinalgError),
    /// Source-backed grids expected to share FEFF radial samples disagreed.
    #[error("{name}[{index}] value {actual} does not match expected {expected}")]
    ValueMismatch {
        name: &'static str,
        index: usize,
        expected: Real,
        actual: Real,
    },
}
