use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use refeff_linalg::{Complex32Lu, LinalgError};
use thiserror::Error;

use crate::exchange::ExchangeError;
use crate::interpolation::InterpolationError;
use crate::quadrature::QuadratureError;
use crate::{Complex, Real};

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
    /// FEFF callers pass density multiplied by `4*pi`; [`crate::grid::fix_potential_grid`]
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

/// Inputs for FEFF `ATOM/scfdat.f90` `FixAtomicQuantities` resampling.
#[derive(Debug, Clone, Copy)]
pub struct AtomicQuantitiesGridInput<'a> {
    /// Source ATOM radial mesh `dr`.
    pub source_radii: ArrayView1<'a, Real>,
    /// Source Coulomb potential `vcoul`.
    pub coulomb_potential: ArrayView1<'a, Real>,
    /// Source total density `srho`, already in the caller's FEFF convention.
    pub charge_density: ArrayView1<'a, Real>,
    /// Source spin magnetization density `dmag`.
    pub magnetization: ArrayView1<'a, Real>,
    /// Source valence density `srhovl`.
    pub valence_density: ArrayView1<'a, Real>,
    /// Initial-state large Dirac component `dgc0`.
    pub initial_large_component: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component `dpc0`.
    pub initial_small_component: ArrayView1<'a, Real>,
    /// Large Dirac components as `(source_radial, orbital)`, FEFF `dgc`.
    pub large_components: ArrayView2<'a, Real>,
    /// Small Dirac components as `(source_radial, orbital)`, FEFF `dpc`.
    pub small_components: ArrayView2<'a, Real>,
    /// Length of the regular FEFF target grid. FEFF `scfdat` uses `251`.
    pub output_len: usize,
}

/// FEFF `FixAtomicQuantities` values on the regular logarithmic radial grid.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicQuantitiesGrid {
    /// Target radial coordinates `exp(xx(i))`.
    pub radii: Array1<Real>,
    /// Interpolated Coulomb potential `vcoul`.
    pub coulomb_potential: Array1<Real>,
    /// Interpolated total density `srho`.
    pub charge_density: Array1<Real>,
    /// Interpolated spin magnetization density `dmag`.
    pub magnetization: Array1<Real>,
    /// Interpolated valence density `srhovl`.
    pub valence_density: Array1<Real>,
    /// Interpolated initial-state large Dirac component `dgc0`.
    pub initial_large_component: Array1<Real>,
    /// Interpolated initial-state small Dirac component `dpc0`.
    pub initial_small_component: Array1<Real>,
    /// Interpolated large Dirac components as `(target_radial, orbital)`.
    pub large_components: Array2<Real>,
    /// Interpolated small Dirac components as `(target_radial, orbital)`.
    pub small_components: Array2<Real>,
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

/// Inputs for FEFF `POT/istprm.f90` first-call muffin-tin radius setup.
#[derive(Debug, Clone, Copy)]
pub struct MuffinTinRadiusParametersInput<'a> {
    /// Highest unique potential index; potentials `0..=highest_potential_index` are included.
    pub highest_potential_index: usize,
    /// Potential index for each atom, FEFF `iphat`.
    pub atom_potentials: ArrayView1<'a, usize>,
    /// Atomic coordinates in Bohr as `(atom, xyz)`, FEFF `rat`.
    pub atom_positions: ArrayView2<'a, Real>,
    /// Representative atom index for each potential, FEFF `iatph`.
    pub representative_atoms: ArrayView1<'a, usize>,
    /// Explicit overlap lists by destination potential, FEFF `OVERLAP` data.
    ///
    /// An empty list for a potential uses the geometry-derived neighbor list.
    pub explicit_overlaps: &'a [&'a [MuffinTinOverlapNeighbor]],
    /// Norman radii `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// Muffin-tin overlap factors `folp`.
    pub overlap_factors: ArrayView1<'a, Real>,
    /// Maximum overlap factors before the `istprm` first-call reduction, `folpx`.
    pub max_overlap_factors: ArrayView1<'a, Real>,
    /// Coulomb potential table as `(radial, potential)`, FEFF `vclap`.
    pub coulomb_potential: ArrayView2<'a, Real>,
    /// Whether AFOLP is active, FEFF `iafolp > 0`.
    pub afolp_enabled: bool,
    /// FEFF `inters` selector. Explicit `OVERLAP` cards force this to `inters mod 6`.
    pub interstitial_selector: usize,
}

/// FEFF `istprm` first-call muffin-tin radius state.
#[derive(Debug, Clone, PartialEq)]
pub struct MuffinTinRadiusParameters {
    /// Muffin-tin radii `rmt`.
    pub muffin_tin_radii: Array1<Real>,
    /// Norman radii copied from input `rnrm`.
    pub norman_radii: Array1<Real>,
    /// 1-based Norman-radius indices `inrm`.
    pub norman_indices: Array1<usize>,
    /// Reduced maximum overlap factors `folpx`.
    pub max_overlap_factors: Array1<Real>,
    /// FEFF `lnear` flags for Norman radii at or beyond the nearest-neighbor distance.
    pub near_neighbor_flags: Array1<bool>,
    /// Nearest-neighbor distance per potential, FEFF `rnnmin`.
    pub nearest_neighbor_distances: Array1<Real>,
    /// Nearest-neighbor potential per potential, FEFF `inn`.
    pub nearest_neighbor_potentials: Array1<usize>,
    /// Whether FEFF's "set Rmt to Rnorman" fallback was used for each potential.
    pub norman_radius_fallbacks: Array1<bool>,
    /// Final `inters` selector after FEFF's explicit-overlap normalization.
    pub interstitial_selector: usize,
}

/// Inputs for the density/exchange/projection block of FEFF `POT/istprm.f90`.
#[derive(Debug, Clone, Copy)]
pub struct MuffinTinInterstitialParametersInput<'a> {
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
    pub explicit_overlaps: &'a [&'a [MuffinTinOverlapNeighbor]],
    /// Total electron density table `(radial, potential)`, FEFF `edens`.
    pub electron_density: ArrayView2<'a, Real>,
    /// Valence electron density table `(radial, potential)`, FEFF `edenvl`.
    pub valence_density: ArrayView2<'a, Real>,
    /// Spin magnetization table `(radial, potential)`, FEFF `dmag`.
    pub magnetization: ArrayView2<'a, Real>,
    /// Coulomb potential table `(radial, potential)`, FEFF `vclap`.
    pub coulomb_potential: ArrayView2<'a, Real>,
    /// Muffin-tin radii `rmt`.
    pub muffin_tin_radii: ArrayView1<'a, Real>,
    /// Norman radii `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// FEFF `lnear` flags.
    pub near_neighbor_flags: ArrayView1<'a, bool>,
    /// FEFF exchange selector `ixc`.
    pub exchange_selector: i32,
    /// FEFF SCF exchange-correlation selector `iscfxc` (`11`, `12`, `21`, or `22`).
    pub scf_exchange_selector: i32,
    /// FEFF `idmag` multiplier for spin-polarized paths.
    pub spin_polarization: i32,
    /// FEFF `scf_temperature / hart`, passed to finite-temperature XC selectors.
    pub scf_temperature_hartree: Real,
    /// Total cluster electron charge `qtotel`.
    pub total_charge: Real,
    /// Current Fermi level `xmu`.
    pub fermi_level: Real,
    /// Input total cell volume `totvol`; nonpositive uses Norman-sphere volume.
    pub total_volume: Real,
    /// FEFF `inters` selector.
    pub interstitial_selector: usize,
}

/// Result of the density/exchange/projection block of FEFF `POT/istprm.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct MuffinTinInterstitialParameters {
    /// Projected total potential table `vtot`.
    pub total_potential: Array2<Real>,
    /// Projected valence potential table `vvalgs`; zero for `ixc mod 10 < 5`.
    pub valence_potential: Array2<Real>,
    /// 1-based last active density index per potential, FEFF `imax`.
    pub max_density_indices: Array1<usize>,
    /// 1-based muffin-tin indices `imt`.
    pub muffin_tin_indices: Array1<usize>,
    /// Muffin-tin radii `rmt` used by the projection.
    pub muffin_tin_radii: Array1<Real>,
    /// 1-based Norman-radius indices `inrm`.
    pub norman_indices: Array1<usize>,
    /// Norman radii after FEFF `sidx` tail adjustment.
    pub norman_radii: Array1<Real>,
    /// Average Norman radius `rnrmav`.
    pub average_norman_radius: Real,
    /// Interstitial volume after `movrlp` overlap corrections, FEFF `volint`.
    pub interstitial_volume: Real,
    /// Interstitial potential `vint`.
    pub interstitial_potential: Real,
    /// Interstitial density in FEFF `4*pi*density` convention, `rhoint`.
    pub interstitial_density: Real,
    /// New Fermi-level result from FEFF `fermi`.
    pub fermi: FermiLevel,
    /// Whether FEFF's `vint >= xmu` fixed-potential retry was applied.
    pub interstitial_potential_limited: bool,
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

/// FEFF `POT/ovp2mt.f90` projection mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MuffinTinOverlapProjectionMode {
    /// Density calculation, FEFF `lrewr = 0`; the returned interstitial value is
    /// the charge outside muffin-tin spheres.
    Density {
        /// Total cluster charge `qtot`.
        total_charge: Real,
    },
    /// Potential calculation with FEFF estimating the muffin-tin zero level,
    /// `lrewr = 1`.
    PotentialEstimateInterstitial,
    /// Potential calculation with fixed interstitial potential, `lrewr = 2`.
    PotentialFixedInterstitial,
}

/// Inputs for FEFF `POT/ovp2mt.f90`.
#[derive(Debug, Clone, Copy)]
pub struct MuffinTinOverlapProjectionInput<'a> {
    /// Highest unique potential index; potentials `0..=highest_potential_index` are included.
    pub highest_potential_index: usize,
    /// Potential or density table as `(radial, potential)`, FEFF `vtot(251,0:nphx)`.
    pub values: ArrayView2<'a, Real>,
    /// FEFF Loucks radial grid `ri`.
    pub radii: ArrayView1<'a, Real>,
    /// Multiplicity of each potential in the cluster, FEFF `xnatph`.
    pub potential_multiplicities: ArrayView1<'a, Real>,
    /// 1-based Norman-radius indices `inrm`.
    pub norman_indices: ArrayView1<'a, usize>,
    /// 1-based muffin-tin indices `imt`.
    pub muffin_tin_indices: ArrayView1<'a, usize>,
    /// Muffin-tin radii `rmt`.
    pub muffin_tin_radii: ArrayView1<'a, Real>,
    /// Norman radii `rnrm`.
    pub norman_radii: ArrayView1<'a, Real>,
    /// FEFF `lnear` flags; true forces `rav = ri(imt + 1)`.
    pub near_neighbor_flags: ArrayView1<'a, bool>,
    /// LU-decomposed overlap matrix from [`crate::grid::muffin_tin_overlap_matrix`].
    pub overlap_matrix: &'a MuffinTinOverlapMatrix,
    /// FEFF `inters` selector, using the same interpretation as
    /// [`MuffinTinOverlapMatrixInput::interstitial_selector`].
    pub interstitial_selector: usize,
    /// Input interstitial value `vint`. This is fixed for
    /// [`MuffinTinOverlapProjectionMode::PotentialFixedInterstitial`] and
    /// ignored after solving in density mode.
    pub interstitial_value: Real,
    /// Projection branch matching FEFF `lrewr`.
    pub mode: MuffinTinOverlapProjectionMode,
}

/// FEFF `ovp2mt` projected muffin-tin table.
#[derive(Debug, Clone, PartialEq)]
pub struct MuffinTinOverlapProjection {
    /// Output table matching FEFF `vtot`. Density mode leaves this unchanged.
    pub values: Array2<Real>,
    /// FEFF `vint`: potential zero level for potential modes, or charge outside
    /// muffin-tin spheres for density mode.
    pub interstitial_value: Real,
    /// Solved moving-window values before adding `vint` in potential modes.
    pub window_values: Array1<Real>,
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
#[non_exhaustive]
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
    /// FEFF `FixAtomicQuantities` resamples same-length scalar radial tables.
    #[error(
        "atomic quantity length mismatch: radii={radii_len}, vcoul={coulomb_len}, srho={density_len}, dmag={magnetization_len}, srhovl={valence_len}, dgc0={large_len}, dpc0={small_len}"
    )]
    AtomicQuantitiesLengthMismatch {
        radii_len: usize,
        coulomb_len: usize,
        density_len: usize,
        magnetization_len: usize,
        valence_len: usize,
        large_len: usize,
        small_len: usize,
    },
    /// FEFF `FixAtomicQuantities` spinor tables are radial-row aligned.
    #[error(
        "atomic spinor table shape ({rows},{columns}) does not match source radial length {radial_len}"
    )]
    AtomicQuantitiesSpinorRowMismatch {
        radial_len: usize,
        rows: usize,
        columns: usize,
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
    /// FEFF `ovp2mt` needs an overlap matrix built for the same potential count.
    #[error("overlap matrix order {actual} does not match required order {required}")]
    OverlapMatrixOrderMismatch { required: usize, actual: usize },
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
    /// FEFF `istprm` needs at least one neighbor for each potential.
    #[error("no muffin-tin neighbor found for potential {potential}")]
    NoMuffinTinNeighbor { potential: usize },
    /// FEFF `istprm` matching-point prescription could not locate a crossing.
    #[error(
        "no muffin-tin matching point found for potential pair {target}->{source_potential} at distance {distance}"
    )]
    NoMuffinTinMatchingPoint {
        target: usize,
        source_potential: usize,
        distance: Real,
    },
    /// FEFF `istprm` calculated no positive interstitial volume.
    #[error("no interstitial density volume after muffin-tin overlap correction: {volume}")]
    NoInterstitialVolume { volume: Real },
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
    /// FEFF exchange-correlation helper failed while constructing grid data.
    #[error(transparent)]
    Exchange(#[from] ExchangeError),
}
