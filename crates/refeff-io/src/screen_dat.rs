//! FEFF screened-core-hole radial diagnostic table support.
//!
//! The SCREEN module writes `wscrn.dat` with the radial grid, screened
//! potential, and core-hole potential. XSPH can then write `vtot.dat` when it
//! folds the screened core-hole potential into the central-atom total
//! potential. Both files are simple FEFF three-column radial tables.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, Axis, Slice,
};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
use refeff_core::{
    ComplexAmplitudePhase, ComplexVec, FovrgDiracSolution, FovrgDiracSolverInput,
    LdosRholRadialAssemblyInput, PhaseError, PotRholieDensityInput, RhorrpExactRadialTail,
    RhorrpExactRadialTailInput, RhorrpIrregularInitialConditionInput,
    RhorrpIrregularSolutionTransformInput, RhorrpIrregularWronskianScaleInput,
    RhorrpWavefunctionGridPreparation, ScreenClusterResponseSlicesInput, ScreenEnergyStateInput,
    ScreenFovrgCubeAssembly, ScreenFovrgMatchedCubeAssembly, ScreenFovrgMatchedCubeAssemblyInput,
    ScreenIntegratedResponseInput, ScreenRadialBounds, ScreenRadialBoundsInput,
    ScreenRadialCubeAssembly, ScreenSolvedCoreHoleResponseInput, XsphRegularPhaseInput, exjlnl,
    fovrg_dirac_solver, fovrg_dirac_solver_c3_potential, fovrg_dirac_solver_with_c3_potential,
    ldos_rhol_assemble_radial_components, muffin_tin_phase_amplitude, pot_rholie_density,
    rhorrp_c3_scale_for_angular_momentum, rhorrp_exact_radial_tail,
    rhorrp_irregular_initial_condition, rhorrp_irregular_solution_transform,
    rhorrp_irregular_wronskian_scale, rhorrp_photoelectron_kappa,
    rhorrp_prepare_wavefunction_grids, screen_cluster_response_slices,
    screen_coulomb_kernel_matrix, screen_energy_state, screen_fms_cluster_green_trace,
    screen_fovrg_matched_cube_assembly, screen_integrated_response,
    screen_lda_exchange_correlation_kernel, screen_radial_bounds, screen_radial_grid,
    screen_solve_response_potential, screen_solved_core_hole_response, xsph_regular_phase,
};

use crate::config_dat::{
    ConfigDatData, RhorrpConfigOrbitalTables, rhorrp_orbital_tables_from_config_dat,
};
use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;
use crate::gg_dat::GgDatData;
use crate::phase_bin::PhaseBinData;
use crate::pot_bin::{POT_BIN_RADIAL_POINTS, PotBinData, rhorrp_wavefunction_handoff_from_pot_bin};
use crate::screen_input::ScreenInput;

const WSCRN_DAT_PATH: &str = "wscrn.dat";
const VTOT_DAT_PATH: &str = "vtot.dat";
const SCREEN_FMS_CLUSTER_GREEN_PATH: &str = "screen-fms-cluster-greens";
const SCREEN_POTENTIAL_KERNEL_PATH: &str = "screen-potential-kernel";
const SCREEN_FOVRG_RADIAL_PATH: &str = "screen-fovrg-radial";
const POT_SCF_FOVRG_SOURCE_GRID_PATH: &str = "pot-scf-fovrg-source-grid";
const POT_SCF_FMS_SOURCE_GRID_PATH: &str = "pot-scf-fms-source-grid";
const SCREEN_RESPONSE_ASSEMBLY_PATH: &str = "screen-response-assembly";
const WSCRN_DEFAULT_HEADER: &str = "# r       w_scrn(r)      v_ch(r)";
const SCREEN_POT_RADIAL_GRID_STEP: f64 = 0.05;
const SCREEN_POT_RADIAL_GRID_ORIGIN: f64 = 8.8;
const POT_SCF_FOVRG_MIN_INWARD_HISTORY_ROWS: usize = 6;
const POT_SCF_FOVRG_BOUND_ORBITAL_THRESHOLD: f64 = 1.0e-11;
const POT_SCF_FOVRG_CORE_COUNT_TOLERANCE: f64 = 1.0e-10;
const POT_SCF_RECOVERED_EMBEDDED_NORM_LIMIT: f64 = 1.0e18;

/// Parsed contents of FEFF `wscrn.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct WscrnDatData {
    /// Header/comment lines before the numeric radial table.
    pub header_lines: Vec<String>,
    /// Radial grid in bohr.
    pub radius_bohr: Array1<f64>,
    /// Screened potential `w_scrn(r)` in atomic units.
    pub screened_potential: Array1<f64>,
    /// Core-hole potential `v_ch(r)` in atomic units.
    pub core_hole_potential: Array1<f64>,
}

/// Inputs for building FEFF `wscrn.dat` from a solved SCREEN response system.
#[derive(Debug, Clone, Copy)]
pub struct WscrnDatFromScreenResponseInput<'a> {
    /// Header/comment lines to preserve before the radial table.
    pub header_lines: &'a [String],
    /// Full radial grid in bohr.
    pub radius_bohr: ArrayView1<'a, f64>,
    /// Bare core-hole potential `v_ch(r)` in atomic units.
    pub core_hole_potential: ArrayView1<'a, f64>,
    /// Screen/CRPA Coulomb response kernel, FEFF `vint`.
    pub response_kernel: ArrayView2<'a, f64>,
    /// Irreducible response function, FEFF `chi0`.
    pub susceptibility: ArrayView2<'a, Complex64>,
    /// Active radial prefix, FEFF `ilast`.
    pub active_count: usize,
}

/// Inputs for building FEFF `wscrn.dat` from per-energy SCREEN response slices.
#[derive(Debug, Clone, Copy)]
pub struct WscrnDatFromScreenResponseSlicesInput<'a> {
    /// Header/comment lines to preserve before the radial table.
    pub header_lines: &'a [String],
    /// Full radial grid in bohr.
    pub radius_bohr: ArrayView1<'a, f64>,
    /// Bare core-hole potential `v_ch(r)` in atomic units.
    pub core_hole_potential: ArrayView1<'a, f64>,
    /// Screen/CRPA Coulomb response kernel, FEFF `vint`.
    pub response_kernel: ArrayView2<'a, f64>,
    /// Complex contour energy grid, FEFF `em`.
    pub energies: ArrayView1<'a, Complex64>,
    /// Per-energy upper-triangle response slices, FEFF `chi0re(:,:,ie)`.
    pub response_slices: ArrayView3<'a, Complex64>,
    /// Active radial prefix, FEFF `ilast`.
    pub active_count: usize,
}

/// Inputs for building FEFF `wscrn.dat` from core-hole radial components and a
/// solved SCREEN response system.
#[derive(Debug, Clone, Copy)]
pub struct WscrnDatFromCoreHoleResponseInput<'a> {
    /// Header/comment lines to preserve before the radial table.
    pub header_lines: &'a [String],
    /// Full radial grid in bohr.
    pub radius_bohr: ArrayView1<'a, f64>,
    /// Core orbital large component, FEFF `dgc0`.
    pub large_component: ArrayView1<'a, f64>,
    /// Core orbital small component, FEFF `dpc0`.
    pub small_component: ArrayView1<'a, f64>,
    /// Screen/CRPA Coulomb response kernel, FEFF `vint`.
    pub response_kernel: ArrayView2<'a, f64>,
    /// Irreducible response function, FEFF `chi0`.
    pub susceptibility: ArrayView2<'a, Complex64>,
    /// Loucks radial-grid step `dx`.
    pub radial_step: f64,
    /// Active radial prefix, FEFF `ilast`.
    pub active_count: usize,
}

/// Inputs for deriving SCREEN FMS cluster Green traces from FEFF handoffs.
#[derive(Debug, Clone, Copy)]
pub struct ScreenFmsClusterGreenHandoffInput<'a> {
    /// Parsed XSPH `phase.bin` handoff.
    pub phase: &'a PhaseBinData,
    /// Parsed FMS `gg.bin`/`gg.dat` Green-function sections.
    pub green: &'a GgDatData,
    /// FEFF potential index providing the phase shifts, normally absorber `iph=0`.
    pub potential_index: usize,
    /// Spin channel to use from `phase.bin`, normally the first spin channel.
    pub spin_index: usize,
    /// Number of non-negative angular channels to assemble.
    pub angular_count: usize,
}

/// SCREEN-ready `gtrl(l,ie)` cluster Green traces from `phase.bin` and `gg.bin`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenFmsClusterGreenHandoff {
    /// FEFF contour/energy rows selected from `phase.bin`.
    pub energies_hartree: Array1<Complex64>,
    /// Cluster Green traces as `(energy, angular_momentum)`.
    pub cluster_greens: Array2<Complex64>,
    /// FEFF potential index used for the phase shifts.
    pub potential_index: usize,
    /// FEFF spin channel used for the phase shifts.
    pub spin_index: usize,
}

/// Inputs for deriving SCREEN radial/kernel state from `screen.inp` and `pot.bin`.
#[derive(Debug, Clone, Copy)]
pub struct ScreenPotentialKernelHandoffInput<'a> {
    /// Parsed `screen.inp` controls.
    pub screen: &'a ScreenInput,
    /// Parsed `pot.bin` potential state.
    pub pot: &'a PotBinData,
    /// FEFF potential index to use, normally absorber `iph=0`.
    pub potential_index: usize,
}

/// SCREEN-ready radial bounds and response kernel from typed potential state.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenPotentialKernelHandoff {
    /// Loucks radial grid in bohr, FEFF `ri`.
    pub radius_bohr: Array1<f64>,
    /// Shared SCREEN/CRPA radial bounds.
    pub bounds: ScreenRadialBounds,
    /// Optional local exchange-correlation kernel `fxc`; zero-width for RPA.
    pub local_kernel: Option<Array1<f64>>,
    /// Screen/CRPA response kernel, FEFF `Kmat`.
    pub response_kernel: Array2<f64>,
    /// Core-hole large component, FEFF `dgc0`.
    pub core_large_component: Array1<f64>,
    /// Core-hole small component, FEFF `dpc0`.
    pub core_small_component: Array1<f64>,
    /// FEFF potential index used for density/radii setup.
    pub potential_index: usize,
    /// Muffin-tin radius for the selected absorber potential, FEFF `rmt`.
    pub muffin_tin_radius_bohr: f64,
    /// Norman radius for the selected absorber potential, FEFF `rnrm`.
    pub norman_radius_bohr: f64,
    /// SCREEN exchange-correlation selector used for per-energy setup.
    pub exchange_selector: i32,
    /// Loucks radial-grid step `dx`.
    pub radial_step: f64,
}

/// Inputs for assembling a complete SCREEN response from radial solution cubes.
#[derive(Debug, Clone, Copy)]
pub struct ScreenResponseAssemblyHandoffInput<'a> {
    /// Radial bounds, kernel, and core-hole components from `screen.inp`/`pot.bin`.
    pub potential: &'a ScreenPotentialKernelHandoff,
    /// FMS trace table from `phase.bin` and `gg.bin`.
    pub fms: &'a ScreenFmsClusterGreenHandoff,
    /// FEFF `eref(ie)` values used to compute `ck(ie)`.
    pub reference_energies_hartree: ArrayView1<'a, Complex64>,
    /// Regular radial solutions `pr(energy,r,l)`.
    pub regular_solutions: ArrayView3<'a, Complex64>,
    /// Irregular radial solutions `pn(energy,r,l)`.
    pub irregular_solutions: ArrayView3<'a, Complex64>,
    /// Header/comment lines to write before `wscrn.dat`.
    pub header_lines: &'a [String],
}

/// Inputs for solving absorber SCREEN radial cubes from typed handoff files.
#[derive(Debug, Clone, Copy)]
pub struct ScreenFovrgRadialHandoffInput<'a> {
    /// Radial/kernel handoff built from `screen.inp` and `pot.bin`.
    pub potential: &'a ScreenPotentialKernelHandoff,
    /// Parsed `pot.bin` potential and bound-spinor state.
    pub pot: &'a PotBinData,
    /// Parsed `config.dat` compactable orbital occupations.
    pub config: &'a ConfigDatData,
    /// SCREEN contour energy grid in Hartree.
    pub energies_hartree: ArrayView1<'a, Complex64>,
    /// Reference/self-energy values used with the contour grid.
    pub reference_energies_hartree: ArrayView1<'a, Complex64>,
    /// Number of non-negative angular channels to solve.
    pub angular_count: usize,
    /// FEFF `irrh == 1` switch for outgoing-Hankel irregular boundary values.
    pub use_hankel_boundary: bool,
}

/// Solved SCREEN radial cubes produced from `pot.bin`/`config.dat` state.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenFovrgRadialHandoff {
    /// Reference/self-energy values used to prepare each radial energy row.
    pub reference_energies_hartree: Array1<Complex64>,
    /// Relativistic SCREEN wave numbers, FEFF `ck(ie)`.
    pub wave_numbers: Array1<Complex64>,
    /// Matched FOVRG radial cubes and per-channel phase data.
    pub matched: ScreenFovrgMatchedCubeAssembly,
    /// FEFF potential index used for the absorber solve.
    pub potential_index: usize,
}

/// SCREEN phase shifts produced from regular FOVRG solves only.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenFovrgPhaseHandoff {
    /// Reference/self-energy values used to prepare each radial energy row.
    pub reference_energies_hartree: Array1<Complex64>,
    /// Relativistic SCREEN wave numbers, FEFF `ck(ie)`.
    pub wave_numbers: Array1<Complex64>,
    /// Recovered FEFF `ph0(ie,l)` phase shifts.
    pub phase_shifts: Array2<Complex64>,
    /// Recovered FEFF `temp(ie,l)` phase amplitudes.
    pub phase_amplitudes: Array2<Complex64>,
    /// FEFF potential index used for the solve.
    pub potential_index: usize,
}

/// Inputs for building absorber radial cubes plus all-potential SCREEN phases.
#[derive(Debug, Clone, Copy)]
pub struct ScreenFovrgPhaseGridHandoffInput<'a> {
    /// Potential/kernel handoffs indexed by FEFF potential number.
    pub potentials: &'a [ScreenPotentialKernelHandoff],
    /// FEFF absorber potential index, normally `iph=0`.
    pub absorber_potential_index: usize,
    /// Parsed `pot.bin` potential and bound-spinor state.
    pub pot: &'a PotBinData,
    /// Parsed `config.dat` compactable orbital occupations.
    pub config: &'a ConfigDatData,
    /// SCREEN contour energy grid in Hartree.
    pub energies_hartree: ArrayView1<'a, Complex64>,
    /// Reference/self-energy values used with the contour grid.
    pub reference_energies_hartree: ArrayView1<'a, Complex64>,
    /// Number of non-negative angular channels to solve.
    pub angular_count: usize,
    /// FEFF `irrh == 1` switch for outgoing-Hankel irregular boundary values.
    pub use_hankel_boundary: bool,
}

/// Absorber radial cubes plus all-potential source FMS phase shifts.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenFovrgPhaseGridHandoff {
    /// Full absorber radial handoff consumed by SCREEN response assembly.
    pub absorber_radial: ScreenFovrgRadialHandoff,
    /// Recovered FEFF `ph0(ie,l,iph)` phase shifts.
    pub phase_shifts: Array3<Complex64>,
    /// Recovered FEFF `temp(ie,l,iph)` phase amplitudes.
    pub phase_amplitudes: Array3<Complex64>,
}

/// Inputs for source-backed POT SCF radial/channel rows from `pot.bin`.
#[derive(Debug, Clone, Copy)]
pub struct PotScfFovrgSourceGridHandoffInput<'a> {
    /// Parsed `pot.bin` potential and bound-spinor state.
    pub pot: &'a PotBinData,
    /// Parsed `config.dat` compactable orbital occupations.
    pub config: &'a ConfigDatData,
    /// POT `scmt` contour energy grid in Hartree.
    pub energies_hartree: ArrayView1<'a, Complex64>,
    /// FEFF exchange selector `ixc`.
    pub exchange_selector: i32,
    /// Number of non-negative angular channels to solve.
    pub angular_count: usize,
    /// FEFF `irrh == 1` switch for outgoing-Hankel irregular boundary values.
    pub use_hankel_boundary: bool,
}

/// Inputs for preparing reusable POT SCF FOVRG source-grid state.
#[derive(Debug, Clone, Copy)]
pub struct PotScfFovrgSourceGridPlanInput<'a> {
    /// Parsed `pot.bin` potential and bound-spinor state.
    pub pot: &'a PotBinData,
    /// Parsed `config.dat` compactable orbital occupations.
    pub config: &'a ConfigDatData,
    /// FEFF exchange selector `ixc`.
    pub exchange_selector: i32,
    /// Number of non-negative angular channels to solve.
    pub angular_count: usize,
    /// FEFF `irrh == 1` switch for outgoing-Hankel irregular boundary values.
    pub use_hankel_boundary: bool,
}

/// Reusable preparation for POT SCF FOVRG source-grid rows.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfFovrgSourceGridPlan {
    pot: PotBinData,
    config: ConfigDatData,
    angular_count: usize,
    use_hankel_boundary: bool,
    potential_handoffs: Vec<ScreenPotentialKernelHandoff>,
    prepared: RhorrpWavefunctionGridPreparation,
    orbital_tables: RhorrpConfigOrbitalTables,
}

/// Inputs for building POT SCF FOVRG rows from a reusable plan.
#[derive(Debug, Clone, Copy)]
pub struct PotScfFovrgSourceGridFromPlanInput<'a> {
    /// Reusable source-grid preparation.
    pub plan: &'a PotScfFovrgSourceGridPlan,
    /// POT `scmt` contour energy grid in Hartree.
    pub energies_hartree: ArrayView1<'a, Complex64>,
}

/// All-potential FOVRG source grid needed by POT `scmt` contour rows.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfFovrgSourceGridHandoff {
    /// Source Loucks radial grid prefix shared by the packed wavefunction cubes.
    pub source_radii: Array1<f64>,
    /// POT `scmt` contour energy grid in Hartree.
    pub energies_hartree: Array1<Complex64>,
    /// Per-energy, per-potential reference energies used by the radial solves.
    pub reference_energies_hartree: Array2<Complex64>,
    /// Relativistic wave numbers, shaped `(energy, potential)`.
    pub wave_numbers: Array2<Complex64>,
    /// Regular large Dirac components, shaped `(energy, potential, l, radial)`.
    pub regular_large: Array4<Complex64>,
    /// Regular small Dirac components, shaped `(energy, potential, l, radial)`.
    pub regular_small: Array4<Complex64>,
    /// Irregular large Dirac components, shaped `(energy, potential, l, radial)`.
    pub irregular_large: Array4<Complex64>,
    /// Irregular small Dirac components, shaped `(energy, potential, l, radial)`.
    pub irregular_small: Array4<Complex64>,
    /// Recovered FEFF `ph0(ie,l,iph)` phase shifts.
    pub phase_shifts: Array3<Complex64>,
    /// Recovered FEFF `temp(ie,l,iph)` phase amplitudes.
    pub phase_amplitudes: Array3<Complex64>,
    /// Active FOVRG radial prefix for each potential, FEFF `ilast`.
    pub radial_active_counts: Array1<usize>,
    /// Active POT density prefix for each potential, FEFF `rholie` `nr05`.
    pub rholie_active_counts: Array1<usize>,
    /// One-based muffin-tin match row for each potential.
    pub muffin_tin_indices_1based: Array1<usize>,
    /// One-based Norman-radius row for each potential.
    pub norman_indices_1based: Array1<usize>,
    /// Per-potential radial handoffs before packing into POT source shapes.
    pub radial_handoffs: Vec<ScreenFovrgRadialHandoff>,
}

/// Inputs for FEFF `POT/corval.f90` embedded-LDOS peak scans.
#[derive(Debug, Clone, Copy)]
pub struct PotScfCorvalLdosHandoffInput<'a> {
    /// Parsed `pot.bin` potential and bound-spinor state.
    pub pot: &'a PotBinData,
    /// Parsed `config.dat` compactable orbital occupations.
    pub config: &'a ConfigDatData,
    /// CORVAL scan energy grid in Hartree.
    pub energies_hartree: ArrayView1<'a, Complex64>,
    /// FEFF exchange selector `ixc`.
    pub exchange_selector: i32,
    /// Requested `(l, potential)` channels to solve.
    pub requested_channels: ArrayView2<'a, bool>,
    /// FEFF `irrh == 1` switch for outgoing-Hankel irregular boundary values.
    pub use_hankel_boundary: bool,
}

/// Embedded angular LDOS rows needed by FEFF `POT/corval.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfCorvalLdosHandoff {
    /// CORVAL scan energy grid in Hartree.
    pub energies_hartree: Array1<Complex64>,
    /// Embedded angular LDOS `xrhoce`, indexed as `(energy, l, potential)`.
    pub embedded_ldos_source: Array3<Complex64>,
}

/// Inputs for projecting POT SCF FMS scattering matrices to `gtr(l,iph)` rows.
#[derive(Debug, Clone, Copy)]
pub struct PotScfFmsSourceGridHandoffInput<'a> {
    /// POT `scmt` contour energy grid in Hartree.
    pub energies_hartree: ArrayView1<'a, Complex64>,
    /// Source all-potential phase shifts, shaped `(energy, l, potential)`.
    pub phase_shifts: ArrayView3<'a, Complex64>,
    /// FMS scattering matrices, shaped `(energy, channel, channel, potential)`.
    pub scattering_matrices: ArrayView4<'a, Complex32>,
    /// Number of non-negative angular channels to project.
    pub angular_count: usize,
}

/// All-potential FMS trace grid needed by POT `scmt` contour rows.
#[derive(Debug, Clone, PartialEq)]
pub struct PotScfFmsSourceGridHandoff {
    /// POT `scmt` contour energy grid in Hartree.
    pub energies_hartree: Array1<Complex64>,
    /// FEFF `gtr(l,iph)` traces, shaped `(energy, l, potential)`.
    pub scattering_trace: Array3<Complex64>,
}

/// SCREEN response handoff assembled from source radial, phase, and FMS state.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenResponseAssemblyHandoff {
    /// Complex photoelectron wave numbers as `ck(ie)`.
    pub wave_numbers: Array1<Complex64>,
    /// Per-energy upper-triangle response slices, FEFF `chi0re(:,:,ie)`.
    pub response_slices: Array3<Complex64>,
    /// Integrated symmetric susceptibility, FEFF `chi0r`.
    pub susceptibility: Array2<Complex64>,
    /// FEFF-compatible screened-core-hole table.
    pub wscrn: WscrnDatData,
}

impl WscrnDatData {
    /// Number of radial grid rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.radius_bohr.len()
    }
}

/// Parsed contents of FEFF `vtot.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct VtotDatData {
    /// Header/comment lines before the numeric radial table.
    pub header_lines: Vec<String>,
    /// Radial grid in bohr.
    pub radius_bohr: Array1<f64>,
    /// Original total potential before the screened core-hole update.
    pub total_potential: Array1<f64>,
    /// Screened core-hole potential read from `wscrn.dat`.
    pub screened_core_hole_potential: Array1<f64>,
}

impl VtotDatData {
    /// Number of radial grid rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.radius_bohr.len()
    }
}

/// Parse FEFF `wscrn.dat` text.
pub fn parse_wscrn_dat(text: &str) -> Result<WscrnDatData> {
    let table = parse_three_column_table(text, WSCRN_DAT_PATH)?;
    let data = WscrnDatData {
        header_lines: table.header_lines,
        radius_bohr: Array1::from_vec(table.first),
        screened_potential: Array1::from_vec(table.second),
        core_hole_potential: Array1::from_vec(table.third),
    };
    validate_wscrn_dat(&data)?;
    Ok(data)
}

/// Parse FEFF `vtot.dat` text.
pub fn parse_vtot_dat(text: &str) -> Result<VtotDatData> {
    let table = parse_three_column_table(text, VTOT_DAT_PATH)?;
    let data = VtotDatData {
        header_lines: table.header_lines,
        radius_bohr: Array1::from_vec(table.first),
        total_potential: Array1::from_vec(table.second),
        screened_core_hole_potential: Array1::from_vec(table.third),
    };
    validate_vtot_dat(&data)?;
    Ok(data)
}

/// Build FEFF `wscrn.dat` data from the SCREEN response equation.
///
/// This ports the final `SCREEN/screensub.f90` output handoff after FEFF solves
/// `A * w_scrn = v_ch`, where `A = I - K * imag(chi0)`. The input arrays may be
/// larger FEFF work buffers; only the active radial prefix is written.
pub fn wscrn_dat_from_screen_response(
    input: WscrnDatFromScreenResponseInput<'_>,
) -> Result<WscrnDatData> {
    validate_wscrn_dat_from_screen_response_input(&input)?;

    let screened_potential = screen_solve_response_potential(
        input.response_kernel,
        input.susceptibility,
        input.core_hole_potential,
        input.active_count,
    )
    .map_err(|source| parse_error_value(WSCRN_DAT_PATH, 0, source.to_string()))?;

    let data = WscrnDatData {
        header_lines: input.header_lines.to_vec(),
        radius_bohr: Array1::from_iter(input.radius_bohr.iter().take(input.active_count).copied()),
        screened_potential,
        core_hole_potential: Array1::from_iter(
            input
                .core_hole_potential
                .iter()
                .take(input.active_count)
                .copied(),
        ),
    };
    validate_wscrn_dat(&data)?;
    Ok(data)
}

/// Build FEFF `wscrn.dat` data from per-energy SCREEN response slices.
///
/// This starts one step before [`wscrn_dat_from_screen_response`]: it applies
/// FEFF's contour trapezoid accumulation to `chi0re(:,:,ie)`, mirrors the
/// upper triangle into symmetric `chi0r`, and then solves the screened
/// response equation.
pub fn wscrn_dat_from_screen_response_slices(
    input: WscrnDatFromScreenResponseSlicesInput<'_>,
) -> Result<WscrnDatData> {
    let susceptibility = screen_integrated_response(ScreenIntegratedResponseInput {
        energies: input.energies,
        response_slices: input.response_slices,
        active_count: input.active_count,
    })
    .map_err(|source| parse_error_value(WSCRN_DAT_PATH, 0, source.to_string()))?;

    wscrn_dat_from_screen_response(WscrnDatFromScreenResponseInput {
        header_lines: input.header_lines,
        radius_bohr: input.radius_bohr,
        core_hole_potential: input.core_hole_potential,
        response_kernel: input.response_kernel,
        susceptibility: susceptibility.view(),
        active_count: input.active_count,
    })
}

/// Build FEFF `wscrn.dat` data from `SCREEN/screensub.f90` core-hole response.
///
/// This starts before [`wscrn_dat_from_screen_response`]: it forms the bare
/// core-hole Coulomb potential from `dgc0`/`dpc0`, solves the screened response
/// equation, and writes both `wscrn` and `vch` columns.
pub fn wscrn_dat_from_core_hole_response(
    input: WscrnDatFromCoreHoleResponseInput<'_>,
) -> Result<WscrnDatData> {
    validate_wscrn_dat_from_core_hole_response_input(&input)?;
    let radius = active_prefix(input.radius_bohr, input.active_count);
    let large = active_prefix(input.large_component, input.active_count);
    let small = active_prefix(input.small_component, input.active_count);
    let response = screen_solved_core_hole_response(ScreenSolvedCoreHoleResponseInput {
        radii: &radius,
        large_component: &large,
        small_component: &small,
        response_kernel: input.response_kernel,
        susceptibility: input.susceptibility,
        dx: input.radial_step,
        active_count: input.active_count,
    })
    .map_err(|source| parse_error_value(WSCRN_DAT_PATH, 0, source.to_string()))?;

    let data = WscrnDatData {
        header_lines: input.header_lines.to_vec(),
        radius_bohr: Array1::from_vec(radius),
        screened_potential: response.screened_potential,
        core_hole_potential: response.bare_potential,
    };
    validate_wscrn_dat(&data)?;
    Ok(data)
}

/// Build SCREEN `gtrl(l,ie)` traces from FEFF `phase.bin` and FMS `gg.bin`.
///
/// `SCREEN/screensub.f90` reduces the FMS scattering matrix for each energy and
/// angular channel by summing the diagonal magnetic substates
/// `l^2..(l+1)^2-1`, then multiplying by `exp(2*i*ph_l)/(2*l+1)`. Parsed
/// `phase.bin` stores signed angular columns as `-lmax..lmax`, so this adapter
/// selects the positive `+l` slot `lmax + l` for the absorber phase factor.
pub fn screen_fms_cluster_green_handoff(
    input: ScreenFmsClusterGreenHandoffInput<'_>,
) -> Result<ScreenFmsClusterGreenHandoff> {
    validate_screen_fms_cluster_green_handoff_input(&input)?;

    let energy_count = input.green.section_count();
    let potential = &input.phase.potentials[input.potential_index];
    let mut cluster_greens = Array2::zeros((energy_count, input.angular_count));

    for (energy_index, section) in input.green.sections.iter().enumerate() {
        let scattering =
            section_values_as_complex32(section.values.view(), section.section_number)?;
        for angular_momentum in 0..input.angular_count {
            let phase_slot = potential
                .lmax
                .checked_add(angular_momentum)
                .ok_or_else(|| {
                    parse_error_value(
                        SCREEN_FMS_CLUSTER_GREEN_PATH,
                        section.section_number,
                        "positive phase angular slot overflowed",
                    )
                })?;
            let phase_shift = potential.phase_shifts[(energy_index, phase_slot, input.spin_index)];
            cluster_greens[(energy_index, angular_momentum)] =
                screen_fms_cluster_green_trace(scattering.view(), phase_shift, angular_momentum)
                    .map_err(|source| {
                        parse_error_value(
                            SCREEN_FMS_CLUSTER_GREEN_PATH,
                            section.section_number,
                            source.to_string(),
                        )
                    })?;
        }
    }

    Ok(ScreenFmsClusterGreenHandoff {
        energies_hartree: Array1::from_iter(
            input.phase.energy_grid.iter().take(energy_count).copied(),
        ),
        cluster_greens,
        potential_index: input.potential_index,
        spin_index: input.spin_index,
    })
}

/// Project POT SCF FMS scattering matrices into FEFF `gtr(l,iph)` traces.
///
/// POT uses the same angular reduction as SCREEN: sum each diagonal magnetic
/// substate block for angular momentum `l`, then multiply by
/// `exp(2*i*ph_l)/(2*l+1)`. Unlike SCREEN, POT retains a trace for every
/// potential so the output is shaped `(energy, l, potential)`.
pub fn pot_scf_fms_source_grid_handoff(
    input: PotScfFmsSourceGridHandoffInput<'_>,
) -> Result<PotScfFmsSourceGridHandoff> {
    validate_pot_scf_fms_source_grid_handoff_input(&input)?;

    let energy_count = input.energies_hartree.len();
    let potential_count = input.phase_shifts.dim().2;
    let mut scattering_trace =
        Array3::<Complex64>::zeros((energy_count, input.angular_count, potential_count));

    for energy_index in 0..energy_count {
        let scattering_energy = input.scattering_matrices.index_axis(Axis(0), energy_index);
        for potential_index in 0..potential_count {
            let scattering = scattering_energy.index_axis(Axis(2), potential_index);
            for angular_momentum in 0..input.angular_count {
                let phase_shift =
                    input.phase_shifts[(energy_index, angular_momentum, potential_index)];
                scattering_trace[(energy_index, angular_momentum, potential_index)] =
                    screen_fms_cluster_green_trace(scattering, phase_shift, angular_momentum)
                        .map_err(|source| {
                            parse_error_value(
                                POT_SCF_FMS_SOURCE_GRID_PATH,
                                energy_index + 1,
                                source.to_string(),
                            )
                        })?;
            }
        }
    }

    Ok(PotScfFmsSourceGridHandoff {
        energies_hartree: input.energies_hartree.to_owned(),
        scattering_trace,
    })
}

/// Build SCREEN radial bounds and `Kmat` from typed `screen.inp`/`pot.bin`.
///
/// This covers the production setup needed after `POT/wrpot.f90` and before
/// the SCREEN energy loop: reconstruct the standard 251-point Loucks radial
/// grid, derive `jri`, `jnrm`, and `ilast` from absorber `rmt`/`rnrm`, build
/// the Coulomb response matrix, and add FEFF's local-density `fxc` diagonal
/// term when `screen.inp` requests a nonzero local-field branch.
pub fn screen_potential_kernel_handoff(
    input: ScreenPotentialKernelHandoffInput<'_>,
) -> Result<ScreenPotentialKernelHandoff> {
    validate_screen_potential_kernel_handoff_input(&input)?;

    let radius_bohr = screen_radial_grid(
        SCREEN_POT_RADIAL_GRID_STEP,
        SCREEN_POT_RADIAL_GRID_ORIGIN,
        POT_BIN_RADIAL_POINTS,
    )
    .map_err(|source| {
        parse_error_value(
            SCREEN_POTENTIAL_KERNEL_PATH,
            0,
            format!("radial grid setup failed: {source}"),
        )
    })?;
    let tail_extension = isize::try_from(input.screen.iend).map_err(|_| {
        parse_error_value(
            SCREEN_POTENTIAL_KERNEL_PATH,
            0,
            format!("screen.inp iend {} does not fit isize", input.screen.iend),
        )
    })?;
    let bounds = screen_radial_bounds(ScreenRadialBoundsInput {
        x0: SCREEN_POT_RADIAL_GRID_ORIGIN,
        dx: SCREEN_POT_RADIAL_GRID_STEP,
        muffin_tin_radius: input.pot.muffin_tin_radii[input.potential_index],
        norman_radius: input.pot.norman_radii[input.potential_index],
        tail_extension,
        radial_capacity: POT_BIN_RADIAL_POINTS,
        response_capacity: POT_BIN_RADIAL_POINTS,
    })
    .map_err(|source| {
        parse_error_value(
            SCREEN_POTENTIAL_KERNEL_PATH,
            0,
            format!("radial bounds setup failed: {source}"),
        )
    })?;

    let density = input
        .pot
        .electron_density
        .column(input.potential_index)
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let radius_slice = radius_bohr.as_slice().ok_or_else(|| {
        parse_error_value(
            SCREEN_POTENTIAL_KERNEL_PATH,
            0,
            "SCREEN radial grid is not contiguous",
        )
    })?;
    let local_kernel = if input.screen.lfxc == 0 {
        None
    } else {
        Some(
            screen_lda_exchange_correlation_kernel(
                radius_slice,
                &density,
                input.screen.lfxc,
                bounds.active_count,
            )
            .map_err(|source| {
                parse_error_value(
                    SCREEN_POTENTIAL_KERNEL_PATH,
                    0,
                    format!("local exchange-correlation kernel setup failed: {source}"),
                )
            })?,
        )
    };
    let response_kernel = screen_coulomb_kernel_matrix(
        radius_slice,
        bounds.active_count,
        local_kernel.as_ref().and_then(|kernel| kernel.as_slice()),
    )
    .map_err(|source| {
        parse_error_value(
            SCREEN_POTENTIAL_KERNEL_PATH,
            0,
            format!("response kernel setup failed: {source}"),
        )
    })?;

    Ok(ScreenPotentialKernelHandoff {
        radius_bohr,
        bounds,
        local_kernel,
        response_kernel,
        core_large_component: input.pot.initial_large_component.clone(),
        core_small_component: input.pot.initial_small_component.clone(),
        potential_index: input.potential_index,
        muffin_tin_radius_bohr: input.pot.muffin_tin_radii[input.potential_index],
        norman_radius_bohr: input.pot.norman_radii[input.potential_index],
        exchange_selector: input.screen.lfxc,
        radial_step: SCREEN_POT_RADIAL_GRID_STEP,
    })
}

/// Build SCREEN radial cubes by solving FOVRG from `pot.bin` and `config.dat`.
///
/// This is the typed source bridge between the potential/config handoffs and
/// the SCREEN response assembly: it prepares the absorber radial grids through
/// the shared FEFF `fixvar`/`fixdsx` path, builds regular/irregular FOVRG
/// solver templates in `(energy,l)` order, runs the matched SCREEN radial cube
/// helper, and returns the normalized cubes consumed by
/// [`screen_response_assembly_handoff`].
pub fn screen_fovrg_radial_handoff(
    input: ScreenFovrgRadialHandoffInput<'_>,
) -> Result<ScreenFovrgRadialHandoff> {
    validate_screen_fovrg_radial_handoff_input(&input)?;
    let prepared = screen_fovrg_prepared_state(input)?;
    screen_fovrg_radial_handoff_from_prepared(input, &prepared.prepared, &prepared.orbital_tables)
}

/// Build SCREEN phase shifts by solving only regular FOVRG channels.
///
/// This is the source handoff needed for non-absorber potentials when SCREEN
/// recomputes inline FMS traces: FMS needs `getph`/`phamp` phase data for every
/// potential, but SCREEN response assembly only consumes absorber radial cubes.
pub fn screen_fovrg_phase_handoff(
    input: ScreenFovrgRadialHandoffInput<'_>,
) -> Result<ScreenFovrgPhaseHandoff> {
    validate_screen_fovrg_radial_handoff_input(&input)?;
    let prepared = screen_fovrg_prepared_state(input)?;
    screen_fovrg_phase_handoff_from_prepared(input, &prepared.prepared, &prepared.orbital_tables)
}

/// Build absorber SCREEN radial cubes and all-potential FOVRG phases from one
/// prepared `pot.bin`/`config.dat` grid.
pub fn screen_fovrg_phase_grid_handoff(
    input: ScreenFovrgPhaseGridHandoffInput<'_>,
) -> Result<ScreenFovrgPhaseGridHandoff> {
    validate_screen_fovrg_phase_grid_handoff_input(&input)?;

    let absorber = &input.potentials[input.absorber_potential_index];
    let base = ScreenFovrgRadialHandoffInput {
        potential: absorber,
        pot: input.pot,
        config: input.config,
        energies_hartree: input.energies_hartree,
        reference_energies_hartree: input.reference_energies_hartree,
        angular_count: input.angular_count,
        use_hankel_boundary: input.use_hankel_boundary,
    };
    let prepared = screen_fovrg_prepared_state(base)?;
    let energy_count = input.energies_hartree.len();
    let potential_count = input.potentials.len();
    let mut phase_shifts =
        Array3::<Complex64>::zeros((energy_count, input.angular_count, potential_count));
    let mut phase_amplitudes =
        Array3::<Complex64>::zeros((energy_count, input.angular_count, potential_count));
    let mut absorber_radial = None;

    for potential in input.potentials {
        let per_potential = ScreenFovrgRadialHandoffInput {
            potential,
            pot: input.pot,
            config: input.config,
            energies_hartree: input.energies_hartree,
            reference_energies_hartree: input.reference_energies_hartree,
            angular_count: input.angular_count,
            use_hankel_boundary: input.use_hankel_boundary,
        };
        if potential.potential_index == input.absorber_potential_index {
            let radial = screen_fovrg_radial_handoff_from_prepared(
                per_potential,
                &prepared.prepared,
                &prepared.orbital_tables,
            )?;
            for energy in 0..energy_count {
                for angular in 0..input.angular_count {
                    phase_shifts[(energy, angular, potential.potential_index)] =
                        radial.matched.phase_shifts[(energy, angular)];
                    phase_amplitudes[(energy, angular, potential.potential_index)] =
                        radial.matched.phase_amplitudes[(energy, angular)];
                }
            }
            absorber_radial = Some(radial);
        } else {
            let phases = screen_fovrg_phase_handoff_from_prepared(
                per_potential,
                &prepared.prepared,
                &prepared.orbital_tables,
            )?;
            for energy in 0..energy_count {
                for angular in 0..input.angular_count {
                    phase_shifts[(energy, angular, potential.potential_index)] =
                        phases.phase_shifts[(energy, angular)];
                    phase_amplitudes[(energy, angular, potential.potential_index)] =
                        phases.phase_amplitudes[(energy, angular)];
                }
            }
        }
    }

    Ok(ScreenFovrgPhaseGridHandoff {
        absorber_radial: absorber_radial.ok_or_else(|| {
            parse_error_value(
                SCREEN_FOVRG_RADIAL_PATH,
                0,
                "SCREEN absorber radial handoff was not built",
            )
        })?,
        phase_shifts,
        phase_amplitudes,
    })
}

/// Build all-potential POT `scmt` FOVRG source rows from typed handoff state.
///
/// This is the POT-facing lift of the SCREEN FOVRG bridge: it uses the same
/// `fixvar`/`fixdsx` preparation and matched regular/irregular solves, but
/// keeps radial cubes for every potential and packs them as
/// `(energy, potential, l, radial)` for `pot_scf_contour_source_rows`.
pub fn pot_scf_fovrg_source_grid_handoff(
    input: PotScfFovrgSourceGridHandoffInput<'_>,
) -> Result<PotScfFovrgSourceGridHandoff> {
    validate_pot_scf_fovrg_source_grid_handoff_input(&input)?;

    let plan = pot_scf_fovrg_source_grid_plan(PotScfFovrgSourceGridPlanInput {
        pot: input.pot,
        config: input.config,
        exchange_selector: input.exchange_selector,
        angular_count: input.angular_count,
        use_hankel_boundary: input.use_hankel_boundary,
    })?;
    pot_scf_fovrg_source_grid_handoff_from_plan(PotScfFovrgSourceGridFromPlanInput {
        plan: &plan,
        energies_hartree: input.energies_hartree,
    })
}

/// Prepare reusable POT SCF FOVRG source-grid state.
pub fn pot_scf_fovrg_source_grid_plan(
    input: PotScfFovrgSourceGridPlanInput<'_>,
) -> Result<PotScfFovrgSourceGridPlan> {
    validate_pot_scf_fovrg_source_grid_plan_input(&input)?;

    let potential_handoffs = pot_scf_fovrg_potential_handoffs(input.pot, input.exchange_selector)?;
    let reference_placeholder = Array1::<Complex64>::zeros(1);
    let base = ScreenFovrgRadialHandoffInput {
        potential: &potential_handoffs[0],
        pot: input.pot,
        config: input.config,
        energies_hartree: reference_placeholder.view(),
        reference_energies_hartree: reference_placeholder.view(),
        angular_count: input.angular_count,
        use_hankel_boundary: input.use_hankel_boundary,
    };
    let prepared = screen_fovrg_prepared_state(base).map_err(|source| {
        parse_error_value(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            format!("wavefunction grid preparation failed: {source}"),
        )
    })?;

    Ok(PotScfFovrgSourceGridPlan {
        pot: input.pot.clone(),
        config: input.config.clone(),
        angular_count: input.angular_count,
        use_hankel_boundary: input.use_hankel_boundary,
        potential_handoffs,
        prepared: prepared.prepared,
        orbital_tables: prepared.orbital_tables,
    })
}

/// Build all-potential POT `scmt` FOVRG source rows from a reusable plan.
pub fn pot_scf_fovrg_source_grid_handoff_from_plan(
    input: PotScfFovrgSourceGridFromPlanInput<'_>,
) -> Result<PotScfFovrgSourceGridHandoff> {
    validate_pot_scf_fovrg_source_grid_from_plan_input(&input)?;

    let potential_handoffs = &input.plan.potential_handoffs;
    let energy_count = input.energies_hartree.len();
    let potential_count = potential_handoffs.len();
    let reference_placeholder = Array1::<Complex64>::zeros(energy_count);

    let max_active_count = potential_handoffs
        .iter()
        .map(|potential| potential.bounds.active_count)
        .max()
        .ok_or_else(|| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                0,
                "at least one POT SCF potential is required",
            )
        })?;
    let source_radii = Array1::from_iter(
        potential_handoffs[0]
            .radius_bohr
            .iter()
            .take(max_active_count)
            .copied(),
    );

    let mut reference_energies = Array2::<Complex64>::zeros((energy_count, potential_count));
    let mut wave_numbers = Array2::<Complex64>::zeros((energy_count, potential_count));
    let mut regular_large = Array4::<Complex64>::zeros((
        energy_count,
        potential_count,
        input.plan.angular_count,
        max_active_count,
    ));
    let mut regular_small = Array4::<Complex64>::zeros(regular_large.dim());
    let mut irregular_large = Array4::<Complex64>::zeros(regular_large.dim());
    let mut irregular_small = Array4::<Complex64>::zeros(regular_large.dim());
    let mut phase_shifts =
        Array3::<Complex64>::zeros((energy_count, input.plan.angular_count, potential_count));
    let mut phase_amplitudes =
        Array3::<Complex64>::zeros((energy_count, input.plan.angular_count, potential_count));
    let mut radial_active_counts = Array1::<usize>::zeros(potential_count);
    let mut rholie_active_counts = Array1::<usize>::zeros(potential_count);
    let mut muffin_tin_indices_1based = Array1::<usize>::zeros(potential_count);
    let mut norman_indices_1based = Array1::<usize>::zeros(potential_count);
    let mut radial_handoffs = Vec::with_capacity(potential_count);

    for potential in potential_handoffs {
        let potential_index = potential.potential_index;
        let per_potential = ScreenFovrgRadialHandoffInput {
            potential,
            pot: &input.plan.pot,
            config: &input.plan.config,
            energies_hartree: input.energies_hartree,
            reference_energies_hartree: reference_placeholder.view(),
            angular_count: input.plan.angular_count,
            use_hankel_boundary: input.plan.use_hankel_boundary,
        };
        let radial = pot_scf_fovrg_radial_handoff_from_prepared(
            per_potential,
            &input.plan.prepared,
            &input.plan.orbital_tables,
        )
        .map_err(|source| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                potential_index + 1,
                format!("FOVRG radial solve failed: {source}"),
            )
        })?;

        radial_active_counts[potential_index] = potential.bounds.active_count;
        rholie_active_counts[potential_index] = pot_scf_rholie_active_count(
            input.plan.pot.norman_radii[potential_index],
            potential_index,
        )?;
        muffin_tin_indices_1based[potential_index] = potential.bounds.muffin_tin_index_1based;
        norman_indices_1based[potential_index] = potential.bounds.norman_index_1based;

        for energy in 0..energy_count {
            reference_energies[(energy, potential_index)] =
                radial.reference_energies_hartree[energy];
            wave_numbers[(energy, potential_index)] = radial.wave_numbers[energy];
            for angular in 0..input.plan.angular_count {
                phase_shifts[(energy, angular, potential_index)] =
                    radial.matched.phase_shifts[(energy, angular)];
                phase_amplitudes[(energy, angular, potential_index)] =
                    radial.matched.phase_amplitudes[(energy, angular)];
                for radial_index in 0..potential.bounds.active_count {
                    regular_large[(energy, potential_index, angular, radial_index)] =
                        radial.matched.solved.radial_cubes.regular_large
                            [(energy, radial_index, angular)];
                    regular_small[(energy, potential_index, angular, radial_index)] =
                        radial.matched.solved.radial_cubes.regular_small
                            [(energy, radial_index, angular)];
                    irregular_large[(energy, potential_index, angular, radial_index)] =
                        radial.matched.solved.radial_cubes.irregular_large
                            [(energy, radial_index, angular)];
                    irregular_small[(energy, potential_index, angular, radial_index)] =
                        radial.matched.solved.radial_cubes.irregular_small
                            [(energy, radial_index, angular)];
                }
            }
        }

        radial_handoffs.push(radial);
    }

    Ok(PotScfFovrgSourceGridHandoff {
        source_radii,
        energies_hartree: input.energies_hartree.to_owned(),
        reference_energies_hartree: reference_energies,
        wave_numbers,
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
        phase_shifts,
        phase_amplitudes,
        radial_active_counts,
        rholie_active_counts,
        muffin_tin_indices_1based,
        norman_indices_1based,
        radial_handoffs,
    })
}

/// Build the embedded LDOS rows needed by FEFF `POT/corval.f90`.
///
/// This follows the same `fixvar`/`fixdsx` and POT `rholie` radial solve path as
/// [`pot_scf_fovrg_source_grid_handoff`], but only evaluates requested
/// `(l, potential)` channels and only keeps `xrhoce`.
pub fn pot_scf_corval_ldos_handoff(
    input: PotScfCorvalLdosHandoffInput<'_>,
) -> Result<PotScfCorvalLdosHandoff> {
    validate_pot_scf_corval_ldos_handoff_input(&input)?;

    let potential_handoffs = pot_scf_fovrg_potential_handoffs(input.pot, input.exchange_selector)?;
    let energy_count = input.energies_hartree.len();
    let potential_count = potential_handoffs.len();
    let angular_count = input.requested_channels.nrows();
    let reference_placeholder = Array1::<Complex64>::zeros(energy_count);
    let base = ScreenFovrgRadialHandoffInput {
        potential: &potential_handoffs[0],
        pot: input.pot,
        config: input.config,
        energies_hartree: input.energies_hartree,
        reference_energies_hartree: reference_placeholder.view(),
        angular_count,
        use_hankel_boundary: input.use_hankel_boundary,
    };
    let prepared = screen_fovrg_prepared_state(base).map_err(|source| {
        parse_error_value(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            format!("CORVAL wavefunction grid preparation failed: {source}"),
        )
    })?;

    let mut embedded_ldos_source =
        Array3::<Complex64>::zeros((energy_count, angular_count, potential_count));
    for potential in &potential_handoffs {
        let potential_index = potential.potential_index;
        if !(0..angular_count).any(|angular| input.requested_channels[(angular, potential_index)]) {
            continue;
        }
        pot_scf_corval_ldos_for_potential_from_prepared(
            &input,
            potential,
            &prepared.prepared,
            &prepared.orbital_tables,
            &mut embedded_ldos_source,
        )?;
    }

    for (index, value) in embedded_ldos_source.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                index + 1,
                format!("POT CORVAL embedded LDOS value is non-finite: {value}"),
            ));
        }
    }

    Ok(PotScfCorvalLdosHandoff {
        energies_hartree: input.energies_hartree.to_owned(),
        embedded_ldos_source,
    })
}

struct ScreenFovrgPreparedState {
    prepared: RhorrpWavefunctionGridPreparation,
    orbital_tables: RhorrpConfigOrbitalTables,
}

fn screen_fovrg_prepared_state(
    input: ScreenFovrgRadialHandoffInput<'_>,
) -> Result<ScreenFovrgPreparedState> {
    let pot_handoff = rhorrp_wavefunction_handoff_from_pot_bin(input.pot)?;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(input.config)?;
    let prepared = rhorrp_prepare_wavefunction_grids(pot_handoff.grid_preparation_input(
        input.potential.radial_step,
        input.potential.exchange_selector,
        POT_BIN_RADIAL_POINTS,
    ))
    .map_err(|source| {
        parse_error_value(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!("wavefunction grid preparation failed: {source}"),
        )
    })?;
    Ok(ScreenFovrgPreparedState {
        prepared,
        orbital_tables,
    })
}

fn screen_fovrg_radial_handoff_from_prepared(
    input: ScreenFovrgRadialHandoffInput<'_>,
    prepared: &RhorrpWavefunctionGridPreparation,
    orbital_tables: &RhorrpConfigOrbitalTables,
) -> Result<ScreenFovrgRadialHandoff> {
    validate_screen_fovrg_radial_handoff_input(&input)?;
    validate_screen_fovrg_prepared_grid(prepared, input.potential)?;

    let zero = Complex64::new(0.0, 0.0);
    let potential_index = input.potential.potential_index;
    let energy_count = input.energies_hartree.len();
    let channel_count = energy_count
        .checked_mul(input.angular_count)
        .ok_or_else(|| {
            parse_error_value(
                SCREEN_FOVRG_RADIAL_PATH,
                0,
                "SCREEN FOVRG channel count overflowed",
            )
        })?;
    let radial_match_index_1based = input.potential.bounds.muffin_tin_index_1based;
    let radial_match_index = radial_match_index_1based - 1;
    let active_count = input.potential.bounds.active_count;
    let target_last_index = active_count - 1;
    let reference_energy = prepared.reference_energies_hartree[potential_index];
    let total_potential = prepared
        .total_potential
        .index_axis(Axis(1), potential_index);
    let valence_potential = prepared
        .valence_potential
        .index_axis(Axis(1), potential_index);
    let bound_large_components = prepared
        .bound_large_components
        .index_axis(Axis(2), potential_index);
    let bound_small_components = prepared
        .bound_small_components
        .index_axis(Axis(2), potential_index);
    let bound_large_coefficients = input
        .pot
        .large_coefficients
        .index_axis(Axis(2), potential_index);
    let bound_small_coefficients = input
        .pot
        .small_coefficients
        .index_axis(Axis(2), potential_index);
    let electron_counts = orbital_tables
        .electron_counts_by_potential
        .index_axis(Axis(1), potential_index);
    let valence_counts = input
        .pot
        .orbital_occupancy
        .index_axis(Axis(1), potential_index);
    let config_valence_counts = orbital_tables
        .valence_counts_by_potential
        .index_axis(Axis(1), potential_index);
    let kappa = orbital_tables
        .kappa_by_potential
        .index_axis(Axis(1), potential_index);
    let atomic_number = input.pot.atomic_numbers[potential_index] as f64;
    let bound_orbital_count = pot_scf_fovrg_effective_bound_orbital_count(
        bound_large_components,
        bound_small_components,
        electron_counts,
        config_valence_counts,
        orbital_tables.bound_orbital_counts[potential_index],
    )?;

    let mut regular_solvers = Vec::with_capacity(channel_count);
    let mut irregular_solvers = Vec::with_capacity(channel_count);
    let mut wave_numbers = Array1::<Complex64>::zeros(energy_count);
    for energy_index in 0..energy_count {
        let state = screen_energy_state(ScreenEnergyStateInput {
            energy: input.energies_hartree[energy_index],
            reference_energy,
            muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
            exchange_selector: input.potential.exchange_selector,
        })
        .map_err(|source| {
            parse_error_value(
                SCREEN_FOVRG_RADIAL_PATH,
                energy_index + 1,
                format!("energy-state setup failed: {source}"),
            )
        })?;
        wave_numbers[energy_index] = state.wave_number;

        for angular in 0..input.angular_count {
            let target_kappa = rhorrp_photoelectron_kappa(angular).map_err(|source| {
                parse_error_value(
                    SCREEN_FOVRG_RADIAL_PATH,
                    energy_index + 1,
                    format!("photoelectron kappa setup failed: {source}"),
                )
            })?;
            let base = FovrgDiracSolverInput {
                exchange_cycle_count: state.dirac_cycle_count,
                target_kappa,
                muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
                target_last_index,
                energy: state.kinetic_energy,
                step: prepared.radial_dx,
                radii: prepared.radii.view(),
                exchange_correlation_potential: total_potential,
                valence_exchange_correlation_potential: valence_potential,
                bound_large_components,
                bound_small_components,
                bound_large_coefficients,
                bound_small_coefficients,
                electron_counts,
                valence_counts,
                kappa,
                muffin_tin_large_component: zero,
                muffin_tin_small_component: zero,
                atomic_number,
                irregular: false,
                c3_scale: rhorrp_c3_scale_for_angular_momentum(angular),
                radial_match_index,
                bound_orbital_count,
            };
            regular_solvers.push(base);
            irregular_solvers.push(FovrgDiracSolverInput {
                irregular: true,
                ..base
            });
        }
    }

    let matched = screen_fovrg_matched_cube_assembly(ScreenFovrgMatchedCubeAssemblyInput {
        regular_solvers: &regular_solvers,
        irregular_solvers: &irregular_solvers,
        wave_numbers: wave_numbers.view(),
        angular_count: input.angular_count,
        radial_match_index_1based,
        active_count,
        use_hankel_boundary: input.use_hankel_boundary,
    })
    .map_err(|source| {
        parse_error_value(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!("matched FOVRG cube assembly failed: {source}"),
        )
    })?;

    Ok(ScreenFovrgRadialHandoff {
        reference_energies_hartree: Array1::from_elem(energy_count, reference_energy),
        wave_numbers,
        matched,
        potential_index,
    })
}

fn pot_scf_fovrg_radial_handoff_from_prepared(
    input: ScreenFovrgRadialHandoffInput<'_>,
    prepared: &RhorrpWavefunctionGridPreparation,
    orbital_tables: &RhorrpConfigOrbitalTables,
) -> Result<ScreenFovrgRadialHandoff> {
    let energy_count = input.energies_hartree.len();
    let potential_index = input.potential.potential_index;
    let channel_count = energy_count
        .checked_mul(input.angular_count)
        .ok_or_else(|| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                0,
                "POT SCF FOVRG channel count overflowed",
            )
        })?;
    let radial_match_index = input.potential.bounds.muffin_tin_index_1based - 1;
    let active_count = input.potential.bounds.active_count;
    let target_last_index = active_count - 1;
    let reference_energy = prepared.reference_energies_hartree[potential_index];
    let total_potential = prepared
        .total_potential
        .index_axis(Axis(1), potential_index);
    let valence_potential = prepared
        .valence_potential
        .index_axis(Axis(1), potential_index);
    let bound_large_components = prepared
        .bound_large_components
        .index_axis(Axis(2), potential_index);
    let bound_small_components = prepared
        .bound_small_components
        .index_axis(Axis(2), potential_index);
    let bound_large_coefficients = input
        .pot
        .large_coefficients
        .index_axis(Axis(2), potential_index);
    let bound_small_coefficients = input
        .pot
        .small_coefficients
        .index_axis(Axis(2), potential_index);
    let electron_counts = orbital_tables
        .electron_counts_by_potential
        .index_axis(Axis(1), potential_index);
    let valence_counts = input
        .pot
        .orbital_occupancy
        .index_axis(Axis(1), potential_index);
    let config_valence_counts = orbital_tables
        .valence_counts_by_potential
        .index_axis(Axis(1), potential_index);
    let kappa = orbital_tables
        .kappa_by_potential
        .index_axis(Axis(1), potential_index);
    let atomic_number = input.pot.atomic_numbers[potential_index] as f64;
    let bound_orbital_count = pot_scf_fovrg_effective_bound_orbital_count(
        bound_large_components,
        bound_small_components,
        electron_counts,
        config_valence_counts,
        orbital_tables.bound_orbital_counts[potential_index],
    )?;

    let zero = Complex64::new(0.0, 0.0);
    let mut wave_numbers = Array1::<Complex64>::zeros(energy_count);
    let mut regular_large =
        Array3::<Complex64>::zeros((energy_count, active_count, input.angular_count));
    let mut regular_small = Array3::<Complex64>::zeros(regular_large.dim());
    let mut irregular_large = Array3::<Complex64>::zeros(regular_large.dim());
    let mut irregular_small = Array3::<Complex64>::zeros(regular_large.dim());
    let mut irregular_initial_large =
        Array2::<Complex64>::zeros((energy_count, input.angular_count));
    let mut irregular_initial_small =
        Array2::<Complex64>::zeros((energy_count, input.angular_count));
    let mut phase_shifts = Array2::<Complex64>::zeros((energy_count, input.angular_count));
    let mut phase_amplitudes = Array2::<Complex64>::zeros((energy_count, input.angular_count));
    let mut regular_iteration_counts = Array2::<usize>::zeros((energy_count, input.angular_count));
    let mut irregular_iteration_counts =
        Array2::<usize>::zeros((energy_count, input.angular_count));
    let mut difficult_iterations = Array2::<usize>::zeros((energy_count, input.angular_count));

    let mut energy_states = Vec::with_capacity(energy_count);
    for energy_index in 0..energy_count {
        let state = screen_energy_state(ScreenEnergyStateInput {
            energy: input.energies_hartree[energy_index],
            reference_energy,
            muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
            exchange_selector: input.potential.exchange_selector,
        })
        .map_err(|source| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                energy_index + 1,
                format!("energy-state setup failed: {source}"),
            )
        })?;
        wave_numbers[energy_index] = state.wave_number;
        energy_states.push(state);
    }

    let c3_state = *energy_states.first().ok_or_else(|| {
        parse_error_value(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "POT SCF FOVRG source grid requires at least one prepared energy state",
        )
    })?;
    let mut c3_potentials = Vec::with_capacity(input.angular_count);
    for angular in 0..input.angular_count {
        let c3_scale = rhorrp_c3_scale_for_angular_momentum(angular);
        if c3_scale == 0 {
            c3_potentials.push(None);
            continue;
        }
        let target_kappa = rhorrp_photoelectron_kappa(angular).map_err(|source| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                0,
                format!("C3 photoelectron kappa setup failed: {source}"),
            )
        })?;
        let solver = FovrgDiracSolverInput {
            exchange_cycle_count: c3_state.dirac_cycle_count,
            target_kappa,
            muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
            target_last_index,
            energy: c3_state.kinetic_energy,
            step: prepared.radial_dx,
            radii: prepared.radii.view(),
            exchange_correlation_potential: total_potential,
            valence_exchange_correlation_potential: valence_potential,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts,
            valence_counts,
            kappa,
            muffin_tin_large_component: zero,
            muffin_tin_small_component: zero,
            atomic_number,
            irregular: false,
            c3_scale,
            radial_match_index,
            bound_orbital_count,
        };
        let c3_potential = fovrg_dirac_solver_c3_potential(solver).map_err(|source| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                0,
                format!("POT FOVRG C3 setup failed for l={angular}: {source}"),
            )
        })?;
        c3_potentials.push(Some(c3_potential));
    }

    let solved_channels = (0..channel_count)
        .into_par_iter()
        .map(|channel_index| {
            let energy_index = channel_index / input.angular_count;
            let angular = channel_index % input.angular_count;
            let state = energy_states[energy_index];
            let target_kappa = rhorrp_photoelectron_kappa(angular).map_err(|source| {
                parse_error_value(
                    POT_SCF_FOVRG_SOURCE_GRID_PATH,
                    energy_index + 1,
                    format!("photoelectron kappa setup failed: {source}"),
                )
            })?;
            let solver = FovrgDiracSolverInput {
                exchange_cycle_count: state.dirac_cycle_count,
                target_kappa,
                muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
                target_last_index,
                energy: state.kinetic_energy,
                step: prepared.radial_dx,
                radii: prepared.radii.view(),
                exchange_correlation_potential: total_potential,
                valence_exchange_correlation_potential: valence_potential,
                bound_large_components,
                bound_small_components,
                bound_large_coefficients,
                bound_small_coefficients,
                electron_counts,
                valence_counts,
                kappa,
                muffin_tin_large_component: zero,
                muffin_tin_small_component: zero,
                atomic_number,
                irregular: false,
                c3_scale: rhorrp_c3_scale_for_angular_momentum(angular),
                radial_match_index,
                bound_orbital_count,
            };
            let channel = pot_scf_rholie_channel(
                solver,
                angular,
                state.wave_number,
                active_count,
                c3_potentials[angular].as_ref(),
            )
            .map_err(|source| {
                parse_error_value(
                    POT_SCF_FOVRG_SOURCE_GRID_PATH,
                    energy_index + 1,
                    format!("POT rholie radial channel solve failed: {source}"),
                )
            })?;

            if channel.regular_large.len() != active_count {
                return Err(parse_error_value(
                    POT_SCF_FOVRG_SOURCE_GRID_PATH,
                    energy_index + 1,
                    format!(
                        "POT rholie radial channel returned {} row(s), expected {active_count}",
                        channel.regular_large.len()
                    ),
                ));
            }

            Ok(PotScfSolvedRholieChannel {
                energy_index,
                angular,
                channel,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let solved_count = solved_channels.len();
    for solved in solved_channels {
        let energy_index = solved.energy_index;
        let angular = solved.angular;
        let channel = solved.channel;

        phase_shifts[(energy_index, angular)] = channel.phase_shift;
        phase_amplitudes[(energy_index, angular)] = channel.phase_amplitude;
        irregular_initial_large[(energy_index, angular)] = channel.irregular_initial_large;
        irregular_initial_small[(energy_index, angular)] = channel.irregular_initial_small;
        regular_iteration_counts[(energy_index, angular)] = channel.regular_iteration_count;
        irregular_iteration_counts[(energy_index, angular)] = channel.irregular_iteration_count;
        difficult_iterations[(energy_index, angular)] = channel.difficult_iterations;
        for radial in 0..active_count {
            regular_large[(energy_index, radial, angular)] = channel.regular_large[radial];
            regular_small[(energy_index, radial, angular)] = channel.regular_small[radial];
            irregular_large[(energy_index, radial, angular)] = channel.irregular_large[radial];
            irregular_small[(energy_index, radial, angular)] = channel.irregular_small[radial];
        }
    }
    if solved_count != channel_count {
        return Err(parse_error_value(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            format!("solved {solved_count} POT FOVRG channel(s), expected {channel_count}"),
        ));
    }

    Ok(ScreenFovrgRadialHandoff {
        reference_energies_hartree: Array1::from_elem(energy_count, reference_energy),
        wave_numbers,
        matched: ScreenFovrgMatchedCubeAssembly {
            solved: ScreenFovrgCubeAssembly {
                radial_cubes: ScreenRadialCubeAssembly {
                    regular_large,
                    regular_small,
                    irregular_large,
                    irregular_small,
                },
                irregular_initial_large,
                irregular_initial_small,
                regular_iteration_counts,
                irregular_iteration_counts,
                difficult_iterations,
            },
            phase_shifts,
            phase_amplitudes,
        },
        potential_index,
    })
}

fn pot_scf_corval_ldos_for_potential_from_prepared(
    input: &PotScfCorvalLdosHandoffInput<'_>,
    potential: &ScreenPotentialKernelHandoff,
    prepared: &RhorrpWavefunctionGridPreparation,
    orbital_tables: &RhorrpConfigOrbitalTables,
    embedded_ldos_source: &mut Array3<Complex64>,
) -> Result<()> {
    let energy_count = input.energies_hartree.len();
    let potential_index = potential.potential_index;
    let radial_match_index = potential.bounds.muffin_tin_index_1based - 1;
    let active_count = potential.bounds.active_count;
    let target_last_index = active_count - 1;
    let reference_energy = prepared.reference_energies_hartree[potential_index];
    let total_potential = prepared
        .total_potential
        .index_axis(Axis(1), potential_index);
    let valence_potential = prepared
        .valence_potential
        .index_axis(Axis(1), potential_index);
    let bound_large_components = prepared
        .bound_large_components
        .index_axis(Axis(2), potential_index);
    let bound_small_components = prepared
        .bound_small_components
        .index_axis(Axis(2), potential_index);
    let bound_large_coefficients = input
        .pot
        .large_coefficients
        .index_axis(Axis(2), potential_index);
    let bound_small_coefficients = input
        .pot
        .small_coefficients
        .index_axis(Axis(2), potential_index);
    let electron_counts = orbital_tables
        .electron_counts_by_potential
        .index_axis(Axis(1), potential_index);
    let valence_counts = input
        .pot
        .orbital_occupancy
        .index_axis(Axis(1), potential_index);
    let config_valence_counts = orbital_tables
        .valence_counts_by_potential
        .index_axis(Axis(1), potential_index);
    let kappa = orbital_tables
        .kappa_by_potential
        .index_axis(Axis(1), potential_index);
    let atomic_number = input.pot.atomic_numbers[potential_index] as f64;
    let bound_orbital_count = pot_scf_fovrg_effective_bound_orbital_count(
        bound_large_components,
        bound_small_components,
        electron_counts,
        config_valence_counts,
        orbital_tables.bound_orbital_counts[potential_index],
    )?;
    let source_radii = Array1::from_iter(potential.radius_bohr.iter().take(active_count).copied());
    let output_radii = Array1::from_vec(vec![input.pot.norman_radii[potential_index]]);
    let zero = Complex64::new(0.0, 0.0);

    let c3_state = screen_energy_state(ScreenEnergyStateInput {
        energy: input.energies_hartree[0],
        reference_energy,
        muffin_tin_radius: potential.muffin_tin_radius_bohr,
        exchange_selector: potential.exchange_selector,
    })
    .map_err(|source| {
        parse_error_value(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            1,
            format!("CORVAL C3 energy-state setup failed: {source}"),
        )
    })?;
    let mut c3_potentials = Vec::with_capacity(input.requested_channels.nrows());
    for angular in 0..input.requested_channels.nrows() {
        if !input.requested_channels[(angular, potential_index)] {
            c3_potentials.push(None);
            continue;
        }
        let c3_scale = rhorrp_c3_scale_for_angular_momentum(angular);
        if c3_scale == 0 {
            c3_potentials.push(None);
            continue;
        }
        let target_kappa = rhorrp_photoelectron_kappa(angular).map_err(|source| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                0,
                format!("CORVAL C3 photoelectron kappa setup failed: {source}"),
            )
        })?;
        let solver = FovrgDiracSolverInput {
            exchange_cycle_count: c3_state.dirac_cycle_count,
            target_kappa,
            muffin_tin_radius: potential.muffin_tin_radius_bohr,
            target_last_index,
            energy: c3_state.kinetic_energy,
            step: prepared.radial_dx,
            radii: prepared.radii.view(),
            exchange_correlation_potential: total_potential,
            valence_exchange_correlation_potential: valence_potential,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts,
            valence_counts,
            kappa,
            muffin_tin_large_component: zero,
            muffin_tin_small_component: zero,
            atomic_number,
            irregular: false,
            c3_scale,
            radial_match_index,
            bound_orbital_count,
        };
        let c3_potential = fovrg_dirac_solver_c3_potential(solver).map_err(|source| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                0,
                format!("POT CORVAL FOVRG C3 setup failed for l={angular}: {source}"),
            )
        })?;
        c3_potentials.push(Some(c3_potential));
    }

    for energy_index in 0..energy_count {
        let state = screen_energy_state(ScreenEnergyStateInput {
            energy: input.energies_hartree[energy_index],
            reference_energy,
            muffin_tin_radius: potential.muffin_tin_radius_bohr,
            exchange_selector: potential.exchange_selector,
        })
        .map_err(|source| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                energy_index + 1,
                format!("CORVAL energy-state setup failed: {source}"),
            )
        })?;

        for angular in 0..input.requested_channels.nrows() {
            if !input.requested_channels[(angular, potential_index)] {
                continue;
            }
            let target_kappa = rhorrp_photoelectron_kappa(angular).map_err(|source| {
                parse_error_value(
                    POT_SCF_FOVRG_SOURCE_GRID_PATH,
                    energy_index + 1,
                    format!("CORVAL photoelectron kappa setup failed: {source}"),
                )
            })?;
            let solver = FovrgDiracSolverInput {
                exchange_cycle_count: state.dirac_cycle_count,
                target_kappa,
                muffin_tin_radius: potential.muffin_tin_radius_bohr,
                target_last_index,
                energy: state.kinetic_energy,
                step: prepared.radial_dx,
                radii: prepared.radii.view(),
                exchange_correlation_potential: total_potential,
                valence_exchange_correlation_potential: valence_potential,
                bound_large_components,
                bound_small_components,
                bound_large_coefficients,
                bound_small_coefficients,
                electron_counts,
                valence_counts,
                kappa,
                muffin_tin_large_component: zero,
                muffin_tin_small_component: zero,
                atomic_number,
                irregular: false,
                c3_scale: rhorrp_c3_scale_for_angular_momentum(angular),
                radial_match_index,
                bound_orbital_count,
            };
            let channel = pot_scf_rholie_channel(
                solver,
                angular,
                state.wave_number,
                active_count,
                c3_potentials[angular].as_ref(),
            )
            .map_err(|source| {
                parse_error_value(
                    POT_SCF_FOVRG_SOURCE_GRID_PATH,
                    energy_index + 1,
                    format!("POT CORVAL rholie radial channel solve failed: {source}"),
                )
            })?;
            let density = pot_rholie_density(PotRholieDensityInput {
                source_radii: source_radii.view(),
                output_radii: output_radii.view(),
                regular_large: channel.regular_large.view(),
                regular_small: channel.regular_small.view(),
                irregular_large: channel.irregular_large.view(),
                irregular_small: channel.irregular_small.view(),
                radial_step: SCREEN_POT_RADIAL_GRID_STEP,
                norman_radius: input.pot.norman_radii[potential_index],
                wave_number: state.wave_number,
                angular_momentum: angular,
            })
            .map_err(|source| {
                parse_error_value(
                    POT_SCF_FOVRG_SOURCE_GRID_PATH,
                    energy_index + 1,
                    format!("POT CORVAL embedded LDOS assembly failed: {source}"),
                )
            })?;
            embedded_ldos_source[(energy_index, angular, potential_index)] = density.embedded_ldos;
        }
    }

    Ok(())
}

fn pot_scf_fovrg_effective_bound_orbital_count(
    bound_large_components: ArrayView2<'_, f64>,
    bound_small_components: ArrayView2<'_, f64>,
    electron_counts: ArrayView1<'_, f64>,
    valence_counts: ArrayView1<'_, f64>,
    configured_count: usize,
) -> Result<usize> {
    if configured_count > bound_large_components.ncols()
        || configured_count > bound_small_components.ncols()
        || configured_count > electron_counts.len()
        || configured_count > valence_counts.len()
    {
        return Err(parse_error_value(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            format!(
                "POT SCF FOVRG bound orbital count {configured_count} exceeds component/count shapes large={:?}, small={:?}, xnel={}, xnval={}",
                bound_large_components.dim(),
                bound_small_components.dim(),
                electron_counts.len(),
                valence_counts.len()
            ),
        ));
    }

    let mut effective_count = 0usize;
    let mut active_by_orbital = Vec::with_capacity(configured_count);
    for orbital in 0..configured_count {
        let has_component = (0..bound_large_components.nrows()).any(|row| {
            bound_large_components[(row, orbital)].abs() >= POT_SCF_FOVRG_BOUND_ORBITAL_THRESHOLD
                || bound_small_components[(row, orbital)].abs()
                    >= POT_SCF_FOVRG_BOUND_ORBITAL_THRESHOLD
        });
        active_by_orbital.push(has_component);
        if has_component {
            effective_count = orbital + 1;
            continue;
        }

        let core_count = electron_counts[orbital] - valence_counts[orbital];
        if core_count.abs() > POT_SCF_FOVRG_CORE_COUNT_TOLERANCE {
            return Err(parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                orbital + 1,
                format!(
                    "POT SCF FOVRG core orbital {orbital} has no radial component but core count {core_count}"
                ),
            ));
        }
    }

    if effective_count == 0 {
        return Err(parse_error_value(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "POT SCF FOVRG source grid has no active bound orbital",
        ));
    }

    for (orbital, &active) in active_by_orbital.iter().take(effective_count).enumerate() {
        if !active {
            return Err(parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                orbital + 1,
                format!(
                    "POT SCF FOVRG zero-component orbital {orbital} is not trailing and cannot be omitted"
                ),
            ));
        }
    }

    Ok(effective_count)
}

struct PotScfRholieChannel {
    phase_shift: Complex64,
    phase_amplitude: Complex64,
    irregular_initial_large: Complex64,
    irregular_initial_small: Complex64,
    regular_large: Array1<Complex64>,
    regular_small: Array1<Complex64>,
    irregular_large: Array1<Complex64>,
    irregular_small: Array1<Complex64>,
    regular_iteration_count: usize,
    irregular_iteration_count: usize,
    difficult_iterations: usize,
}

struct PotScfSolvedRholieChannel {
    energy_index: usize,
    angular: usize,
    channel: PotScfRholieChannel,
}

#[derive(Debug, Clone, Copy)]
struct PotScfRecoveredPhase {
    row_index: usize,
    phase_shift: Complex64,
    phase_amplitude: Complex64,
}

fn pot_scf_rholie_channel(
    solver: FovrgDiracSolverInput<'_>,
    angular_momentum: usize,
    wave_number: Complex64,
    active_count: usize,
    c3_potential: Option<&ComplexVec>,
) -> std::result::Result<PotScfRholieChannel, String> {
    if active_count == 0 {
        return Err("active radial count is zero".to_string());
    }
    if active_count > solver.radii.len() {
        return Err(format!(
            "active radial count {active_count} exceeds radial grid length {}",
            solver.radii.len()
        ));
    }
    if solver.radial_match_index >= active_count {
        return Err(format!(
            "radial match index {} is outside active radial count {active_count}",
            solver.radial_match_index
        ));
    }

    let zero = Complex64::new(0.0, 0.0);
    let regular_solution = pot_scf_fovrg_dirac_solver(
        FovrgDiracSolverInput {
            irregular: false,
            muffin_tin_large_component: zero,
            muffin_tin_small_component: zero,
            ..solver
        },
        c3_potential,
        "regular",
    )?;
    ensure_pot_scf_rholie_component_len(
        "regular large component",
        regular_solution.large_component.len(),
        active_count,
    )?;
    ensure_pot_scf_rholie_component_len(
        "regular small component",
        regular_solution.small_component.len(),
        active_count,
    )?;

    let muffin_tin_wave_number = wave_number * solver.muffin_tin_radius;
    let current = exjlnl(muffin_tin_wave_number, angular_momentum)
        .map_err(|source| format!("Bessel setup failed for l={angular_momentum}: {source}"))?;
    let next_angular_momentum = angular_momentum
        .checked_add(1)
        .ok_or_else(|| "angular momentum overflowed when requesting l+1".to_string())?;
    let next = exjlnl(muffin_tin_wave_number, next_angular_momentum)
        .map_err(|source| format!("Bessel setup failed for l={next_angular_momentum}: {source}"))?;
    let boundary_phase = match muffin_tin_phase_amplitude(
        solver.muffin_tin_radius,
        regular_solution.muffin_tin_large_component,
        regular_solution.muffin_tin_small_component,
        wave_number,
        current.j,
        current.y,
        next.j,
        next.y,
        solver.target_kappa,
    ) {
        Ok(phase) => Some(phase),
        Err(PhaseError::SingularComplexArctangent { .. }) => None,
        Err(source) => return Err(format!("muffin-tin phase/amplitude setup failed: {source}")),
    };
    let active_radii = Array1::from_iter(solver.radii.iter().take(active_count).copied());
    let recovered_phase = if boundary_phase
        .as_ref()
        .is_none_or(|phase| phase.amplitude == zero)
    {
        pot_scf_recover_zero_boundary_phase(
            regular_solution
                .large_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            regular_solution
                .small_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            active_radii.view(),
            regular_solution.wkb_index,
            solver.radial_match_index,
            angular_momentum,
            wave_number,
            solver.target_kappa,
        )?
    } else {
        None
    };
    let boundary_phase = boundary_phase.unwrap_or(ComplexAmplitudePhase {
        amplitude: zero,
        phase: zero,
    });
    let (phase_shift, phase_amplitude) = if let Some(recovered) = recovered_phase {
        (recovered.phase_shift, recovered.phase_amplitude)
    } else {
        (boundary_phase.phase, boundary_phase.amplitude)
    };
    let irregular_initial =
        rhorrp_irregular_initial_condition(RhorrpIrregularInitialConditionInput {
            muffin_tin_radius: solver.muffin_tin_radius,
            phase_shift,
            wave_number,
            bessel_j_l: current.j,
            neumann_l: current.y,
            bessel_j_l_plus_1: next.j,
            neumann_l_plus_1: next.y,
        })
        .map_err(|source| format!("irregular boundary setup failed: {source}"))?;

    let match_index_1based = solver
        .radial_match_index
        .checked_add(1)
        .ok_or_else(|| "radial match index overflowed".to_string())?;

    let mut output_phase_shift = phase_shift;
    let mut output_phase_amplitude = phase_amplitude;
    let mut output_irregular_initial_large = irregular_initial.large_component;
    let mut output_irregular_initial_small = irregular_initial.small_component;

    let (
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
        irregular_iteration_count,
        difficult_iterations,
    ) = if phase_amplitude == zero {
        let exact_all =
            pot_scf_exact_tail(&active_radii, 1, angular_momentum, phase_shift, wave_number)?;
        let exact_tail = pot_scf_exact_tail(
            &active_radii,
            match_index_1based,
            angular_momentum,
            phase_shift,
            wave_number,
        )?;
        let match_index = match_index_1based - 1;
        if let Some(regular_solution_scale) = pot_scf_zero_amplitude_regular_scale(
            regular_solution
                .large_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            regular_solution
                .small_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            &exact_all,
        )? {
            let mut regular_large = regular_solution
                .large_component
                .slice_axis(Axis(0), Slice::from(..active_count))
                .mapv(|value| value * regular_solution_scale);
            let mut regular_small = regular_solution
                .small_component
                .slice_axis(Axis(0), Slice::from(..active_count))
                .mapv(|value| value * regular_solution_scale);

            let irregular_solution = pot_scf_fovrg_dirac_solver(
                FovrgDiracSolverInput {
                    irregular: true,
                    muffin_tin_large_component: irregular_initial.large_component,
                    muffin_tin_small_component: irregular_initial.small_component,
                    ..solver
                },
                c3_potential,
                "irregular",
            )?;
            ensure_pot_scf_rholie_component_len(
                "irregular large component",
                irregular_solution.large_component.len(),
                active_count,
            )?;
            ensure_pot_scf_rholie_component_len(
                "irregular small component",
                irregular_solution.small_component.len(),
                active_count,
            )?;

            let wronskian =
                match rhorrp_irregular_wronskian_scale(RhorrpIrregularWronskianScaleInput {
                    phase_shift,
                    wave_number,
                    regular_large_at_match: regular_large[match_index],
                    regular_small_at_match: regular_small[match_index],
                    irregular_large_at_match: irregular_solution.large_component[match_index],
                    irregular_small_at_match: irregular_solution.small_component[match_index],
                }) {
                    Ok(wronskian) => wronskian,
                    Err(_) => {
                        return Ok(PotScfRholieChannel {
                            phase_shift,
                            phase_amplitude,
                            irregular_initial_large: irregular_initial.large_component,
                            irregular_initial_small: irregular_initial.small_component,
                            regular_large: exact_all.regular_large_components,
                            regular_small: exact_all.regular_small_components,
                            irregular_large: exact_all.irregular_large_components,
                            irregular_small: exact_all.irregular_small_components,
                            regular_iteration_count: regular_solution.iteration_count,
                            irregular_iteration_count: irregular_solution.iteration_count,
                            difficult_iterations: regular_solution.difficult_iterations
                                + irregular_solution.difficult_iterations,
                        });
                    }
                };
            let mut irregular_large = Array1::<Complex64>::zeros(active_count);
            let mut irregular_small = Array1::<Complex64>::zeros(active_count);
            for row in 0..active_count {
                let transformed =
                    rhorrp_irregular_solution_transform(RhorrpIrregularSolutionTransformInput {
                        phase_factor: wronskian.phase_factor,
                        reciprocal_wave_scale: wronskian.reciprocal_wave_scale,
                        regular_large_component: regular_large[row],
                        regular_small_component: regular_small[row],
                        irregular_large_component: irregular_solution.large_component[row],
                        irregular_small_component: irregular_solution.small_component[row],
                    })
                    .map_err(|source| format!("irregular row transform failed: {source}"))?;
                irregular_large[row] = transformed.large_component;
                irregular_small[row] = transformed.small_component;
            }

            pot_scf_apply_exact_tail_rows(
                &mut regular_large,
                &mut regular_small,
                &mut irregular_large,
                &mut irregular_small,
                &exact_tail,
            )?;

            (
                regular_large,
                regular_small,
                irregular_large,
                irregular_small,
                irregular_solution.iteration_count,
                regular_solution.difficult_iterations + irregular_solution.difficult_iterations,
            )
        } else {
            (
                exact_all.regular_large_components,
                exact_all.regular_small_components,
                exact_all.irregular_large_components,
                exact_all.irregular_small_components,
                0,
                regular_solution.difficult_iterations,
            )
        }
    } else if let Some(recovered) = recovered_phase {
        if let Some(recovered_rows) = pot_scf_recovered_zero_boundary_rows(
            solver,
            c3_potential,
            active_count,
            &active_radii,
            angular_momentum,
            wave_number,
            recovered,
            irregular_initial.large_component,
            irregular_initial.small_component,
            regular_solution
                .large_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            regular_solution
                .small_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            regular_solution.difficult_iterations,
        )? {
            recovered_rows
        } else {
            let fallback_initial =
                rhorrp_irregular_initial_condition(RhorrpIrregularInitialConditionInput {
                    muffin_tin_radius: solver.muffin_tin_radius,
                    phase_shift: boundary_phase.phase,
                    wave_number,
                    bessel_j_l: current.j,
                    neumann_l: current.y,
                    bessel_j_l_plus_1: next.j,
                    neumann_l_plus_1: next.y,
                })
                .map_err(|source| format!("fallback irregular boundary setup failed: {source}"))?;
            output_phase_shift = boundary_phase.phase;
            output_phase_amplitude = boundary_phase.amplitude;
            output_irregular_initial_large = fallback_initial.large_component;
            output_irregular_initial_small = fallback_initial.small_component;

            let exact_all = pot_scf_exact_tail(
                &active_radii,
                1,
                angular_momentum,
                boundary_phase.phase,
                wave_number,
            )?;
            (
                exact_all.regular_large_components,
                exact_all.regular_small_components,
                exact_all.irregular_large_components,
                exact_all.irregular_small_components,
                0,
                regular_solution.difficult_iterations,
            )
        }
    } else {
        let irregular_solution = pot_scf_fovrg_dirac_solver(
            FovrgDiracSolverInput {
                irregular: true,
                muffin_tin_large_component: irregular_initial.large_component,
                muffin_tin_small_component: irregular_initial.small_component,
                ..solver
            },
            c3_potential,
            "irregular",
        )?;
        ensure_pot_scf_rholie_component_len(
            "irregular large component",
            irregular_solution.large_component.len(),
            active_count,
        )?;
        ensure_pot_scf_rholie_component_len(
            "irregular small component",
            irregular_solution.small_component.len(),
            active_count,
        )?;

        let radial_components = ldos_rhol_assemble_radial_components(LdosRholRadialAssemblyInput {
            radii: active_radii.view(),
            raw_regular_large: regular_solution
                .large_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            raw_regular_small: regular_solution
                .small_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            raw_irregular_large: irregular_solution
                .large_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            raw_irregular_small: irregular_solution
                .small_component
                .slice_axis(Axis(0), Slice::from(..active_count)),
            phase_shift,
            phase_amplitude,
            wave_number,
            angular_momentum,
            match_index_1based,
            exact_tail_start_index_1based: match_index_1based,
        })
        .map_err(|source| format!("radial component assembly failed: {source}"))?;

        (
            radial_components.regular_large,
            radial_components.regular_small,
            radial_components.irregular_large,
            radial_components.irregular_small,
            irregular_solution.iteration_count,
            regular_solution.difficult_iterations + irregular_solution.difficult_iterations,
        )
    };

    Ok(PotScfRholieChannel {
        phase_shift: output_phase_shift,
        phase_amplitude: output_phase_amplitude,
        irregular_initial_large: output_irregular_initial_large,
        irregular_initial_small: output_irregular_initial_small,
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
        regular_iteration_count: regular_solution.iteration_count,
        irregular_iteration_count,
        difficult_iterations,
    })
}

fn pot_scf_fovrg_dirac_solver(
    solver: FovrgDiracSolverInput<'_>,
    c3_potential: Option<&ComplexVec>,
    label: &str,
) -> std::result::Result<FovrgDiracSolution, String> {
    let solution = if let Some(c3_potential) = c3_potential {
        fovrg_dirac_solver_with_c3_potential(solver, c3_potential.view())
    } else {
        fovrg_dirac_solver(solver)
    };
    solution.map_err(|source| format!("{label} FOVRG solve failed: {source}"))
}

fn ensure_pot_scf_rholie_component_len(
    name: &str,
    len: usize,
    active_count: usize,
) -> std::result::Result<(), String> {
    if len < active_count {
        return Err(format!(
            "{name} length {len} is below active radial count {active_count}"
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn pot_scf_recover_zero_boundary_phase(
    raw_large: ArrayView1<'_, Complex64>,
    raw_small: ArrayView1<'_, Complex64>,
    radii: ArrayView1<'_, f64>,
    flat_start_index: usize,
    radial_match_index: usize,
    angular_momentum: usize,
    wave_number: Complex64,
    target_kappa: i32,
) -> std::result::Result<Option<PotScfRecoveredPhase>, String> {
    let zero = Complex64::new(0.0, 0.0);
    if raw_small.len() != raw_large.len() || radii.len() != raw_large.len() {
        return Err(format!(
            "zero-boundary phase recovery shape mismatch: large={} small={} radii={}",
            raw_large.len(),
            raw_small.len(),
            radii.len()
        ));
    }
    if raw_large.is_empty() {
        return Ok(None);
    }

    let start = flat_start_index.min(raw_large.len() - 1);
    let end = radial_match_index.min(raw_large.len() - 1);
    if start > end {
        return Ok(None);
    }

    let mut candidates = Vec::new();
    for row in start..=end {
        let norm = raw_large[row].norm() + raw_small[row].norm();
        if norm > 0.0 && (raw_large[row] != zero || raw_small[row] != zero) {
            candidates.push((row, norm));
        }
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    candidates.sort_by(|(_, left), (_, right)| {
        right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal)
    });

    let next_angular_momentum = angular_momentum
        .checked_add(1)
        .ok_or_else(|| "angular momentum overflowed when requesting stable-row l+1".to_string())?;
    for (row, _) in candidates {
        let radius = radii[row];
        let argument = wave_number * radius;
        let Ok(current) = exjlnl(argument, angular_momentum) else {
            continue;
        };
        let Ok(next) = exjlnl(argument, next_angular_momentum) else {
            continue;
        };
        let Ok(phase) = muffin_tin_phase_amplitude(
            radius,
            raw_large[row],
            raw_small[row],
            wave_number,
            current.j,
            current.y,
            next.j,
            next.y,
            target_kappa,
        ) else {
            continue;
        };
        if phase.amplitude != zero {
            return Ok(Some(PotScfRecoveredPhase {
                row_index: row,
                phase_shift: phase.phase,
                phase_amplitude: phase.amplitude,
            }));
        }
    }
    Ok(None)
}

type PotScfRecoveredZeroBoundaryRows = Option<(
    Array1<Complex64>,
    Array1<Complex64>,
    Array1<Complex64>,
    Array1<Complex64>,
    usize,
    usize,
)>;

#[allow(clippy::too_many_arguments)]
fn pot_scf_recovered_zero_boundary_rows(
    solver: FovrgDiracSolverInput<'_>,
    c3_potential: Option<&ComplexVec>,
    active_count: usize,
    active_radii: &Array1<f64>,
    angular_momentum: usize,
    wave_number: Complex64,
    recovered: PotScfRecoveredPhase,
    irregular_initial_large: Complex64,
    irregular_initial_small: Complex64,
    raw_regular_large: ArrayView1<'_, Complex64>,
    raw_regular_small: ArrayView1<'_, Complex64>,
    regular_difficult_iterations: usize,
) -> std::result::Result<PotScfRecoveredZeroBoundaryRows, String> {
    let zero = Complex64::new(0.0, 0.0);
    if recovered.phase_amplitude == zero {
        return Ok(None);
    }
    if recovered.row_index >= active_count {
        return Err(format!(
            "recovered zero-boundary match row {} is outside active radial count {active_count}",
            recovered.row_index
        ));
    }

    let regular_solution_scale = Complex64::new(1.0, 0.0) / recovered.phase_amplitude;
    if !(regular_solution_scale.re.is_finite() && regular_solution_scale.im.is_finite()) {
        return Ok(None);
    }
    let mut regular_large = raw_regular_large.mapv(|value| value * regular_solution_scale);
    let mut regular_small = raw_regular_small.mapv(|value| value * regular_solution_scale);

    let irregular_solution = pot_scf_fovrg_dirac_solver(
        FovrgDiracSolverInput {
            irregular: true,
            muffin_tin_large_component: irregular_initial_large,
            muffin_tin_small_component: irregular_initial_small,
            ..solver
        },
        c3_potential,
        "irregular",
    )?;
    ensure_pot_scf_rholie_component_len(
        "irregular large component",
        irregular_solution.large_component.len(),
        active_count,
    )?;
    ensure_pot_scf_rholie_component_len(
        "irregular small component",
        irregular_solution.small_component.len(),
        active_count,
    )?;

    let match_index = recovered.row_index;
    let wronskian = match rhorrp_irregular_wronskian_scale(RhorrpIrregularWronskianScaleInput {
        phase_shift: recovered.phase_shift,
        wave_number,
        regular_large_at_match: regular_large[match_index],
        regular_small_at_match: regular_small[match_index],
        irregular_large_at_match: irregular_solution.large_component[match_index],
        irregular_small_at_match: irregular_solution.small_component[match_index],
    }) {
        Ok(wronskian) => wronskian,
        Err(_) => return Ok(None),
    };

    let mut irregular_large = Array1::<Complex64>::zeros(active_count);
    let mut irregular_small = Array1::<Complex64>::zeros(active_count);
    for row in 0..active_count {
        let transformed =
            match rhorrp_irregular_solution_transform(RhorrpIrregularSolutionTransformInput {
                phase_factor: wronskian.phase_factor,
                reciprocal_wave_scale: wronskian.reciprocal_wave_scale,
                regular_large_component: regular_large[row],
                regular_small_component: regular_small[row],
                irregular_large_component: irregular_solution.large_component[row],
                irregular_small_component: irregular_solution.small_component[row],
            }) {
                Ok(transformed) => transformed,
                Err(_) => return Ok(None),
            };
        irregular_large[row] = transformed.large_component;
        irregular_small[row] = transformed.small_component;
    }

    let exact_tail_start_index_1based = solver
        .radial_match_index
        .checked_add(1)
        .ok_or_else(|| "radial match index overflowed".to_string())?;
    let exact_tail = pot_scf_exact_tail(
        active_radii,
        exact_tail_start_index_1based,
        angular_momentum,
        recovered.phase_shift,
        wave_number,
    )?;
    pot_scf_apply_exact_tail_rows(
        &mut regular_large,
        &mut regular_small,
        &mut irregular_large,
        &mut irregular_small,
        &exact_tail,
    )?;

    if !pot_scf_recovered_rows_are_usable(
        regular_large.view(),
        regular_small.view(),
        irregular_large.view(),
        irregular_small.view(),
    ) {
        return Ok(None);
    }

    Ok(Some((
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
        irregular_solution.iteration_count,
        regular_difficult_iterations + irregular_solution.difficult_iterations,
    )))
}

fn pot_scf_recovered_rows_are_usable(
    regular_large: ArrayView1<'_, Complex64>,
    regular_small: ArrayView1<'_, Complex64>,
    irregular_large: ArrayView1<'_, Complex64>,
    irregular_small: ArrayView1<'_, Complex64>,
) -> bool {
    if regular_small.len() != regular_large.len()
        || irregular_large.len() != regular_large.len()
        || irregular_small.len() != regular_large.len()
    {
        return false;
    }

    let imaginary = Complex64::new(0.0, 1.0);
    let mut embedded_norm = 0.0;
    let mut regular_norm = 0.0;
    let mut irregular_norm = 0.0;
    for row in 0..regular_large.len() {
        let values = [
            regular_large[row],
            regular_small[row],
            irregular_large[row],
            irregular_small[row],
        ];
        if values
            .iter()
            .any(|value| !(value.re.is_finite() && value.im.is_finite()))
        {
            return false;
        }
        regular_norm += regular_large[row].norm() + regular_small[row].norm();
        irregular_norm += irregular_large[row].norm() + irregular_small[row].norm();
        embedded_norm += (irregular_large[row] * regular_large[row]
            - imaginary * regular_large[row] * regular_large[row]
            + irregular_small[row] * regular_small[row]
            - imaginary * regular_small[row] * regular_small[row])
            .norm();
    }

    regular_norm > 0.0
        && irregular_norm > 0.0
        && embedded_norm > 0.0
        && embedded_norm < POT_SCF_RECOVERED_EMBEDDED_NORM_LIMIT
}

fn pot_scf_zero_amplitude_regular_scale(
    raw_large: ArrayView1<'_, Complex64>,
    raw_small: ArrayView1<'_, Complex64>,
    exact_rows: &RhorrpExactRadialTail,
) -> std::result::Result<Option<Complex64>, String> {
    let zero = Complex64::new(0.0, 0.0);
    if exact_rows.start_index_1based != 1 || exact_rows.row_count() != raw_large.len() {
        return Err(format!(
            "zero phase-amplitude exact row count {} starting at {} does not match raw row count {}",
            exact_rows.row_count(),
            exact_rows.start_index_1based,
            raw_large.len()
        ));
    }
    if raw_small.len() != raw_large.len() {
        return Err(format!(
            "zero phase-amplitude raw small row count {} does not match raw large row count {}",
            raw_small.len(),
            raw_large.len()
        ));
    }

    let mut best_raw = zero;
    let mut best_target = zero;
    let mut best_norm = 0.0;
    for row in 0..raw_large.len() {
        let large_norm = raw_large[row].norm();
        if raw_large[row] != zero && large_norm > best_norm {
            best_raw = raw_large[row];
            best_target = exact_rows.regular_large_components[row];
            best_norm = large_norm;
        }
        let small_norm = raw_small[row].norm();
        if raw_small[row] != zero && small_norm > best_norm {
            best_raw = raw_small[row];
            best_target = exact_rows.regular_small_components[row];
            best_norm = small_norm;
        }
    }
    if best_raw == zero {
        return Ok(None);
    }

    let scale = best_target / best_raw;
    if !(scale.re.is_finite() && scale.im.is_finite()) {
        return Err(format!(
            "zero phase-amplitude regular match scale is non-finite: {scale:?}"
        ));
    }
    Ok(Some(scale))
}

fn pot_scf_exact_tail(
    radii: &Array1<f64>,
    start_index_1based: usize,
    angular_momentum: usize,
    phase_shift: Complex64,
    wave_number: Complex64,
) -> std::result::Result<RhorrpExactRadialTail, String> {
    let radii_slice = radii
        .as_slice()
        .ok_or_else(|| "active radial grid is not contiguous".to_string())?;
    rhorrp_exact_radial_tail(RhorrpExactRadialTailInput {
        radii: radii_slice,
        start_index_1based,
        angular_momentum,
        phase_shift,
        wave_number,
    })
    .map_err(|source| format!("exact radial tail setup failed: {source}"))
}

fn pot_scf_apply_exact_tail_rows(
    regular_large: &mut Array1<Complex64>,
    regular_small: &mut Array1<Complex64>,
    irregular_large: &mut Array1<Complex64>,
    irregular_small: &mut Array1<Complex64>,
    exact_tail: &RhorrpExactRadialTail,
) -> std::result::Result<(), String> {
    let tail_start = exact_tail.start_index_1based - 1;
    for offset in 0..exact_tail.row_count() {
        let row = tail_start + offset;
        regular_large[row] = exact_tail.regular_large_components[offset];
        regular_small[row] = exact_tail.regular_small_components[offset];
        irregular_large[row] = exact_tail.irregular_large_components[offset];
        irregular_small[row] = exact_tail.irregular_small_components[offset];
    }
    Ok(())
}

fn screen_fovrg_phase_handoff_from_prepared(
    input: ScreenFovrgRadialHandoffInput<'_>,
    prepared: &RhorrpWavefunctionGridPreparation,
    orbital_tables: &RhorrpConfigOrbitalTables,
) -> Result<ScreenFovrgPhaseHandoff> {
    validate_screen_fovrg_radial_handoff_input(&input)?;
    validate_screen_fovrg_prepared_grid(prepared, input.potential)?;

    let zero = Complex64::new(0.0, 0.0);
    let potential_index = input.potential.potential_index;
    let energy_count = input.energies_hartree.len();
    let radial_match_index = input.potential.bounds.muffin_tin_index_1based - 1;
    let active_count = input.potential.bounds.active_count;
    let target_last_index = active_count - 1;
    let reference_energy = prepared.reference_energies_hartree[potential_index];
    let total_potential = prepared
        .total_potential
        .index_axis(Axis(1), potential_index);
    let valence_potential = prepared
        .valence_potential
        .index_axis(Axis(1), potential_index);
    let bound_large_components = prepared
        .bound_large_components
        .index_axis(Axis(2), potential_index);
    let bound_small_components = prepared
        .bound_small_components
        .index_axis(Axis(2), potential_index);
    let bound_large_coefficients = input
        .pot
        .large_coefficients
        .index_axis(Axis(2), potential_index);
    let bound_small_coefficients = input
        .pot
        .small_coefficients
        .index_axis(Axis(2), potential_index);
    let electron_counts = orbital_tables
        .electron_counts_by_potential
        .index_axis(Axis(1), potential_index);
    let valence_counts = input
        .pot
        .orbital_occupancy
        .index_axis(Axis(1), potential_index);
    let config_valence_counts = orbital_tables
        .valence_counts_by_potential
        .index_axis(Axis(1), potential_index);
    let kappa = orbital_tables
        .kappa_by_potential
        .index_axis(Axis(1), potential_index);
    let atomic_number = input.pot.atomic_numbers[potential_index] as f64;
    let bound_orbital_count = pot_scf_fovrg_effective_bound_orbital_count(
        bound_large_components,
        bound_small_components,
        electron_counts,
        config_valence_counts,
        orbital_tables.bound_orbital_counts[potential_index],
    )?;
    let mut wave_numbers = Array1::<Complex64>::zeros(energy_count);
    let mut phase_shifts = Array2::<Complex64>::zeros((energy_count, input.angular_count));
    let mut phase_amplitudes = Array2::<Complex64>::zeros((energy_count, input.angular_count));

    for energy_index in 0..energy_count {
        let state = screen_energy_state(ScreenEnergyStateInput {
            energy: input.energies_hartree[energy_index],
            reference_energy,
            muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
            exchange_selector: input.potential.exchange_selector,
        })
        .map_err(|source| {
            parse_error_value(
                SCREEN_FOVRG_RADIAL_PATH,
                energy_index + 1,
                format!("energy-state setup failed: {source}"),
            )
        })?;
        wave_numbers[energy_index] = state.wave_number;

        for angular in 0..input.angular_count {
            let target_kappa = rhorrp_photoelectron_kappa(angular).map_err(|source| {
                parse_error_value(
                    SCREEN_FOVRG_RADIAL_PATH,
                    energy_index + 1,
                    format!("photoelectron kappa setup failed: {source}"),
                )
            })?;
            let regular = fovrg_dirac_solver(FovrgDiracSolverInput {
                exchange_cycle_count: state.dirac_cycle_count,
                target_kappa,
                muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
                target_last_index,
                energy: state.kinetic_energy,
                step: prepared.radial_dx,
                radii: prepared.radii.view(),
                exchange_correlation_potential: total_potential,
                valence_exchange_correlation_potential: valence_potential,
                bound_large_components,
                bound_small_components,
                bound_large_coefficients,
                bound_small_coefficients,
                electron_counts,
                valence_counts,
                kappa,
                muffin_tin_large_component: zero,
                muffin_tin_small_component: zero,
                atomic_number,
                irregular: false,
                c3_scale: rhorrp_c3_scale_for_angular_momentum(angular),
                radial_match_index,
                bound_orbital_count,
            })
            .map_err(|source| {
                parse_error_value(
                    SCREEN_FOVRG_RADIAL_PATH,
                    energy_index + 1,
                    format!("regular FOVRG solve failed: {source}"),
                )
            })?;
            let phase = xsph_regular_phase(XsphRegularPhaseInput {
                muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
                wave_number: state.wave_number,
                regular_large_at_muffin_tin: regular.muffin_tin_large_component,
                regular_small_at_muffin_tin: regular.muffin_tin_small_component,
                kappa: target_kappa,
            })
            .map_err(|source| {
                parse_error_value(
                    SCREEN_FOVRG_RADIAL_PATH,
                    energy_index + 1,
                    format!("phase recovery failed: {source}"),
                )
            })?;
            phase_shifts[(energy_index, angular)] = phase.phase_shift;
            phase_amplitudes[(energy_index, angular)] = phase.phase_amplitude;
        }
    }

    Ok(ScreenFovrgPhaseHandoff {
        reference_energies_hartree: Array1::from_elem(energy_count, reference_energy),
        wave_numbers,
        phase_shifts,
        phase_amplitudes,
        potential_index,
    })
}

/// Assemble the SCREEN response and solved `wscrn.dat` handoff from radial cubes.
///
/// This is the typed production boundary immediately after the still-missing
/// radial-solution driver: it consumes regular/irregular radial solutions,
/// source FMS traces, and the potential/kernel handoff, then runs the ported
/// response-slice assembly, contour integration, and screened-core-hole solve.
pub fn screen_response_assembly_handoff(
    input: ScreenResponseAssemblyHandoffInput<'_>,
) -> Result<ScreenResponseAssemblyHandoff> {
    validate_screen_response_assembly_handoff_input(&input)?;

    let energy_count = input.fms.cluster_greens.nrows();
    let angular_count = input.fms.cluster_greens.ncols();
    let active_count = input.potential.bounds.active_count;
    let mut wave_numbers = Array1::zeros(energy_count);
    for energy_index in 0..energy_count {
        let state = screen_energy_state(ScreenEnergyStateInput {
            energy: input.fms.energies_hartree[energy_index],
            reference_energy: input.reference_energies_hartree[energy_index],
            muffin_tin_radius: input.potential.muffin_tin_radius_bohr,
            exchange_selector: input.potential.exchange_selector,
        })
        .map_err(|source| {
            parse_error_value(
                SCREEN_RESPONSE_ASSEMBLY_PATH,
                energy_index + 1,
                format!("energy-state setup failed: {source}"),
            )
        })?;
        wave_numbers[energy_index] = state.wave_number;
    }

    let radii = input.potential.radius_bohr.as_slice().ok_or_else(|| {
        parse_error_value(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            "SCREEN radial grid is not contiguous",
        )
    })?;
    let response_slices = screen_cluster_response_slices(ScreenClusterResponseSlicesInput {
        radii,
        regular_solutions: input.regular_solutions,
        irregular_solutions: input.irregular_solutions,
        cluster_greens: input.fms.cluster_greens.view(),
        wave_numbers: wave_numbers.view(),
        dx: input.potential.radial_step,
        angular_momentum_count: angular_count,
        active_count,
        fms_count: input.potential.bounds.norman_index_1based,
    })
    .map_err(|source| {
        parse_error_value(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!("response-slice assembly failed: {source}"),
        )
    })?;

    let susceptibility = screen_integrated_response(ScreenIntegratedResponseInput {
        energies: input.fms.energies_hartree.view(),
        response_slices: response_slices.view(),
        active_count,
    })
    .map_err(|source| {
        parse_error_value(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!("response contour integration failed: {source}"),
        )
    })?;

    let wscrn = wscrn_dat_from_core_hole_response(WscrnDatFromCoreHoleResponseInput {
        header_lines: input.header_lines,
        radius_bohr: input.potential.radius_bohr.view(),
        large_component: input.potential.core_large_component.view(),
        small_component: input.potential.core_small_component.view(),
        response_kernel: input.potential.response_kernel.view(),
        susceptibility: susceptibility.view(),
        radial_step: input.potential.radial_step,
        active_count,
    })
    .map_err(|source| {
        parse_error_value(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!("core-hole response solve failed: {source}"),
        )
    })?;

    Ok(ScreenResponseAssemblyHandoff {
        wave_numbers,
        response_slices,
        susceptibility,
        wscrn,
    })
}

/// Build FEFF `vtot.dat` data from a screened-core-hole table and `pot.bin`.
///
/// XSPH writes this sidecar after combining the absorber total potential with
/// the SCREEN `w_scrn(r)` column. This adapter keeps the handoff typed for
/// cached SCREEN/POT directories and for the future uncached SCREEN driver.
pub fn vtot_dat_from_wscrn_and_pot_bin(
    wscrn: &WscrnDatData,
    pot: &PotBinData,
) -> Result<VtotDatData> {
    if pot.total_potential.ncols() == 0 {
        return parse_error(
            VTOT_DAT_PATH,
            0,
            "pot.bin contains no absorber total-potential column for vtot.dat",
        );
    }
    vtot_dat_from_wscrn_and_total_potential(wscrn, pot.total_potential.column(0))
}

/// Build FEFF `vtot.dat` data from a screened-core-hole table and total potential.
pub fn vtot_dat_from_wscrn_and_total_potential(
    wscrn: &WscrnDatData,
    total_potential: ArrayView1<'_, f64>,
) -> Result<VtotDatData> {
    let row_count = wscrn.row_count().min(total_potential.len());
    if row_count == 0 {
        return parse_error(
            VTOT_DAT_PATH,
            0,
            "vtot.dat requires at least one shared wscrn/total-potential row",
        );
    }

    let data = VtotDatData {
        header_lines: Vec::new(),
        radius_bohr: Array1::from_iter(wscrn.radius_bohr.iter().copied().take(row_count)),
        total_potential: Array1::from_iter(total_potential.iter().copied().take(row_count)),
        screened_core_hole_potential: Array1::from_iter(
            wscrn.screened_potential.iter().copied().take(row_count),
        ),
    };
    validate_vtot_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `wscrn.dat` text.
pub fn wscrn_dat_string(data: &WscrnDatData) -> Result<String> {
    validate_wscrn_dat(data)?;
    if data.header_lines.is_empty() {
        three_column_string(
            &[WSCRN_DEFAULT_HEADER],
            &data.radius_bohr,
            &data.screened_potential,
            &data.core_hole_potential,
        )
    } else {
        three_column_string(
            &data.header_lines,
            &data.radius_bohr,
            &data.screened_potential,
            &data.core_hole_potential,
        )
    }
}

/// Render FEFF-compatible `vtot.dat` text.
pub fn vtot_dat_string(data: &VtotDatData) -> Result<String> {
    validate_vtot_dat(data)?;
    three_column_string(
        &data.header_lines,
        &data.radius_bohr,
        &data.total_potential,
        &data.screened_core_hole_potential,
    )
}

/// Read FEFF `wscrn.dat` text from a file.
pub fn read_wscrn_dat(path: impl AsRef<Path>) -> Result<WscrnDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_wscrn_dat(&text)
}

/// Read FEFF `vtot.dat` text from a file.
pub fn read_vtot_dat(path: impl AsRef<Path>) -> Result<VtotDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_vtot_dat(&text)
}

/// Write FEFF `wscrn.dat` text to a file.
pub fn write_wscrn_dat(path: impl AsRef<Path>, data: &WscrnDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, wscrn_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF `vtot.dat` text to a file.
pub fn write_vtot_dat(path: impl AsRef<Path>, data: &VtotDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, vtot_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

#[derive(Debug)]
struct ThreeColumnTable {
    header_lines: Vec<String>,
    first: Vec<f64>,
    second: Vec<f64>,
    third: Vec<f64>,
}

fn parse_three_column_table(text: &str, path: &'static str) -> Result<ThreeColumnTable> {
    let mut header_lines = Vec::new();
    let mut first = Vec::new();
    let mut second = Vec::new();
    let mut third = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            header_lines.push(raw.trim_end().to_owned());
            continue;
        }

        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 3 {
            return parse_error(
                path,
                line_number,
                format!("radial table row has {} token(s), expected 3", tokens.len()),
            );
        }
        first.push(parse_f64(path, line_number, "first", tokens[0])?);
        second.push(parse_f64(path, line_number, "second", tokens[1])?);
        third.push(parse_f64(path, line_number, "third", tokens[2])?);
    }

    if first.is_empty() {
        return parse_error(path, 0, "at least one radial table row is required");
    }
    Ok(ThreeColumnTable {
        header_lines,
        first,
        second,
        third,
    })
}

fn three_column_string(
    header_lines: &[impl AsRef<str>],
    first: &Array1<f64>,
    second: &Array1<f64>,
    third: &Array1<f64>,
) -> Result<String> {
    let mut out = String::new();
    for line in header_lines {
        writeln!(out, "{}", line.as_ref())?;
    }
    for ((first, second), third) in first.iter().zip(second.iter()).zip(third.iter()) {
        write_fortran_zero_scaled_exp(&mut out, *first, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, *second, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, *third, 20, 10)?;
        out.push('\n');
    }
    Ok(out)
}

fn validate_wscrn_dat(data: &WscrnDatData) -> Result<()> {
    validate_three_columns(
        WSCRN_DAT_PATH,
        "wscrn",
        &data.radius_bohr,
        &data.screened_potential,
        &data.core_hole_potential,
    )
}

fn validate_wscrn_dat_from_screen_response_input(
    input: &WscrnDatFromScreenResponseInput<'_>,
) -> Result<()> {
    if input.active_count == 0 {
        return parse_error(WSCRN_DAT_PATH, 0, "active_count must be at least 1");
    }
    validate_active_vector_len("radius_bohr", input.radius_bohr.len(), input.active_count)?;
    validate_active_vector_len(
        "core_hole_potential",
        input.core_hole_potential.len(),
        input.active_count,
    )?;
    validate_active_matrix_shape(
        "response_kernel",
        input.response_kernel.nrows(),
        input.response_kernel.ncols(),
        input.active_count,
    )?;
    validate_active_matrix_shape(
        "susceptibility",
        input.susceptibility.nrows(),
        input.susceptibility.ncols(),
        input.active_count,
    )?;

    for (index, (&radius, &core_hole)) in input
        .radius_bohr
        .iter()
        .zip(input.core_hole_potential.iter())
        .take(input.active_count)
        .enumerate()
    {
        let line = index + 1;
        validate_finite(WSCRN_DAT_PATH, line, "radius_bohr", radius)?;
        validate_finite(WSCRN_DAT_PATH, line, "core_hole_potential", core_hole)?;
    }
    Ok(())
}

fn validate_wscrn_dat_from_core_hole_response_input(
    input: &WscrnDatFromCoreHoleResponseInput<'_>,
) -> Result<()> {
    if input.active_count == 0 {
        return parse_error(WSCRN_DAT_PATH, 0, "active_count must be at least 1");
    }
    validate_active_vector_len("radius_bohr", input.radius_bohr.len(), input.active_count)?;
    validate_active_vector_len(
        "large_component",
        input.large_component.len(),
        input.active_count,
    )?;
    validate_active_vector_len(
        "small_component",
        input.small_component.len(),
        input.active_count,
    )?;
    validate_active_matrix_shape(
        "response_kernel",
        input.response_kernel.nrows(),
        input.response_kernel.ncols(),
        input.active_count,
    )?;
    validate_active_matrix_shape(
        "susceptibility",
        input.susceptibility.nrows(),
        input.susceptibility.ncols(),
        input.active_count,
    )?;

    for (index, ((&radius, &large), &small)) in input
        .radius_bohr
        .iter()
        .zip(input.large_component.iter())
        .zip(input.small_component.iter())
        .take(input.active_count)
        .enumerate()
    {
        let line = index + 1;
        validate_finite(WSCRN_DAT_PATH, line, "radius_bohr", radius)?;
        validate_finite(WSCRN_DAT_PATH, line, "large_component", large)?;
        validate_finite(WSCRN_DAT_PATH, line, "small_component", small)?;
    }
    Ok(())
}

fn validate_screen_fms_cluster_green_handoff_input(
    input: &ScreenFmsClusterGreenHandoffInput<'_>,
) -> Result<()> {
    if input.angular_count == 0 {
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            0,
            "angular_count must be at least 1",
        );
    }
    if input.green.section_count() == 0 {
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            0,
            "at least one gg.bin section is required",
        );
    }
    if input.potential_index >= input.phase.potentials.len() {
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            0,
            format!(
                "potential_index {} exceeds phase.bin potential count {}",
                input.potential_index,
                input.phase.potentials.len()
            ),
        );
    }
    if input.spin_index >= input.phase.spin_count {
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            0,
            format!(
                "spin_index {} exceeds phase.bin spin count {}",
                input.spin_index, input.phase.spin_count
            ),
        );
    }

    let energy_count = input.green.section_count();
    if energy_count != input.phase.main_energy_count && energy_count != input.phase.energy_count {
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            0,
            format!(
                "gg.bin section count {energy_count} does not match phase.bin ne1 {} or ne {}",
                input.phase.main_energy_count, input.phase.energy_count
            ),
        );
    }
    if input.phase.energy_grid.len() < energy_count {
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            0,
            format!(
                "phase.bin energy grid has {} row(s), shorter than gg.bin section count {energy_count}",
                input.phase.energy_grid.len()
            ),
        );
    }

    let potential = &input.phase.potentials[input.potential_index];
    let expected_phase_shape = (
        input.phase.energy_count,
        potential
            .lmax
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                parse_error_value(
                    SCREEN_FMS_CLUSTER_GREEN_PATH,
                    0,
                    "phase.bin lmax channel count overflowed",
                )
            })?,
        input.phase.spin_count,
    );
    if potential.phase_shifts.dim() != expected_phase_shape {
        let (energies, angular, spin) = potential.phase_shifts.dim();
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            0,
            format!(
                "phase.bin potential {} phase-shift shape {energies}x{angular}x{spin} does not match expected {}x{}x{}",
                input.potential_index,
                expected_phase_shape.0,
                expected_phase_shape.1,
                expected_phase_shape.2
            ),
        );
    }
    if input.angular_count > potential.lmax + 1 {
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            0,
            format!(
                "angular_count {} exceeds phase.bin potential {} lmax+1 {}",
                input.angular_count,
                input.potential_index,
                potential.lmax + 1
            ),
        );
    }

    let required_order = input
        .angular_count
        .checked_mul(input.angular_count)
        .ok_or_else(|| {
            parse_error_value(
                SCREEN_FMS_CLUSTER_GREEN_PATH,
                0,
                "required FMS scattering matrix order overflowed",
            )
        })?;
    for section in &input.green.sections {
        let (rows, columns) = section.shape();
        if rows != columns {
            return parse_error(
                SCREEN_FMS_CLUSTER_GREEN_PATH,
                section.section_number,
                format!("gg.bin SCREEN section must be square, got {rows}x{columns}"),
            );
        }
        if rows < required_order {
            return parse_error(
                SCREEN_FMS_CLUSTER_GREEN_PATH,
                section.section_number,
                format!(
                    "gg.bin SCREEN matrix order {rows} is smaller than required order {required_order} for angular_count {}",
                    input.angular_count
                ),
            );
        }
    }
    Ok(())
}

fn validate_pot_scf_fms_source_grid_handoff_input(
    input: &PotScfFmsSourceGridHandoffInput<'_>,
) -> Result<()> {
    let energy_count = input.energies_hartree.len();
    if energy_count == 0 {
        return parse_error(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            0,
            "at least one POT SCF FMS energy row is required",
        );
    }
    if input.angular_count == 0 {
        return parse_error(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            0,
            "at least one POT SCF FMS angular channel is required",
        );
    }
    for (energy_index, energy) in input.energies_hartree.iter().enumerate() {
        validate_finite(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            energy_index + 1,
            "energy.real",
            energy.re,
        )?;
        validate_finite(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            energy_index + 1,
            "energy.imag",
            energy.im,
        )?;
    }

    let (phase_energy_count, phase_angular_count, phase_potential_count) = input.phase_shifts.dim();
    if phase_energy_count < energy_count {
        return parse_error(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            0,
            format!(
                "POT SCF phase table has {phase_energy_count} energy row(s), expected at least {energy_count}"
            ),
        );
    }
    if phase_angular_count < input.angular_count {
        return parse_error(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            0,
            format!(
                "POT SCF phase table has {phase_angular_count} angular channel(s), expected at least {}",
                input.angular_count
            ),
        );
    }
    if phase_potential_count == 0 {
        return parse_error(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            0,
            "at least one POT SCF FMS potential block is required",
        );
    }

    let required_order = input
        .angular_count
        .checked_mul(input.angular_count)
        .ok_or_else(|| {
            parse_error_value(
                POT_SCF_FMS_SOURCE_GRID_PATH,
                0,
                "required FMS scattering matrix order overflowed",
            )
        })?;
    let (scattering_energy_count, scattering_rows, scattering_columns, scattering_potential_count) =
        input.scattering_matrices.dim();
    if scattering_energy_count < energy_count {
        return parse_error(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            0,
            format!(
                "POT SCF scattering table has {scattering_energy_count} energy row(s), expected at least {energy_count}"
            ),
        );
    }
    if scattering_rows != scattering_columns {
        return parse_error(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            0,
            format!(
                "POT SCF scattering matrices must be square, got {scattering_rows}x{scattering_columns}"
            ),
        );
    }
    if scattering_rows < required_order {
        return parse_error(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            0,
            format!(
                "POT SCF scattering matrix order {scattering_rows} is smaller than required order {required_order} for angular_count {}",
                input.angular_count
            ),
        );
    }
    if scattering_potential_count != phase_potential_count {
        return parse_error(
            POT_SCF_FMS_SOURCE_GRID_PATH,
            0,
            format!(
                "POT SCF scattering table has {scattering_potential_count} potential block(s), expected {phase_potential_count}"
            ),
        );
    }
    Ok(())
}

fn validate_screen_potential_kernel_handoff_input(
    input: &ScreenPotentialKernelHandoffInput<'_>,
) -> Result<()> {
    if input.potential_index >= input.pot.potential_count() {
        return parse_error(
            SCREEN_POTENTIAL_KERNEL_PATH,
            0,
            format!(
                "potential_index {} exceeds pot.bin potential count {}",
                input.potential_index,
                input.pot.potential_count()
            ),
        );
    }
    validate_pot_vector_len(
        "rmt",
        input.pot.muffin_tin_radii.len(),
        input.pot.potential_count(),
    )?;
    validate_pot_vector_len(
        "rnrm",
        input.pot.norman_radii.len(),
        input.pot.potential_count(),
    )?;
    validate_pot_vector_len(
        "dgc0",
        input.pot.initial_large_component.len(),
        POT_BIN_RADIAL_POINTS,
    )?;
    validate_pot_vector_len(
        "dpc0",
        input.pot.initial_small_component.len(),
        POT_BIN_RADIAL_POINTS,
    )?;
    let (density_rows, density_columns) = input.pot.electron_density.dim();
    if density_rows < POT_BIN_RADIAL_POINTS || density_columns <= input.potential_index {
        return parse_error(
            SCREEN_POTENTIAL_KERNEL_PATH,
            0,
            format!(
                "pot.bin electron_density shape {density_rows}x{density_columns} cannot supply {} radial rows for potential {}",
                POT_BIN_RADIAL_POINTS, input.potential_index
            ),
        );
    }
    Ok(())
}

fn validate_screen_fovrg_radial_handoff_input(
    input: &ScreenFovrgRadialHandoffInput<'_>,
) -> Result<()> {
    let energy_count = input.energies_hartree.len();
    if energy_count == 0 {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            "at least one SCREEN FOVRG energy row is required",
        );
    }
    if input.reference_energies_hartree.len() != energy_count {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "reference energy length {} does not match energy count {energy_count}",
                input.reference_energies_hartree.len()
            ),
        );
    }
    if input.angular_count == 0 {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            "at least one SCREEN FOVRG angular channel is required",
        );
    }
    let potential_index = input.potential.potential_index;
    if potential_index >= input.pot.potential_count() {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "potential_index {potential_index} exceeds pot.bin potential count {}",
                input.pot.potential_count()
            ),
        );
    }
    if potential_index >= input.config.potential_count() {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "potential_index {potential_index} exceeds config.dat potential count {}",
                input.config.potential_count()
            ),
        );
    }
    if input.potential.bounds.active_count == 0 {
        return parse_error(SCREEN_FOVRG_RADIAL_PATH, 0, "active_count must be positive");
    }
    if input.potential.bounds.muffin_tin_index_1based == 0 {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            "muffin_tin_index_1based must be positive",
        );
    }
    if input.potential.bounds.muffin_tin_index_1based > input.potential.bounds.active_count {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "muffin_tin_index_1based {} exceeds active_count {}",
                input.potential.bounds.muffin_tin_index_1based, input.potential.bounds.active_count
            ),
        );
    }
    Ok(())
}

fn validate_screen_fovrg_phase_grid_handoff_input(
    input: &ScreenFovrgPhaseGridHandoffInput<'_>,
) -> Result<()> {
    let potential_count = input.potentials.len();
    if potential_count == 0 {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            "at least one SCREEN FOVRG potential is required",
        );
    }
    if input.absorber_potential_index >= potential_count {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "absorber potential index {} exceeds potential count {potential_count}",
                input.absorber_potential_index
            ),
        );
    }
    if potential_count != input.pot.potential_count() {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "SCREEN FOVRG potential count {potential_count} does not match pot.bin potential count {}",
                input.pot.potential_count()
            ),
        );
    }
    if input.config.potential_count() < potential_count {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "config.dat potential count {} is smaller than SCREEN FOVRG potential count {potential_count}",
                input.config.potential_count()
            ),
        );
    }

    let absorber = &input.potentials[input.absorber_potential_index];
    if absorber.potential_index != input.absorber_potential_index {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "absorber handoff potential_index {} does not match absorber slot {}",
                absorber.potential_index, input.absorber_potential_index
            ),
        );
    }

    for (expected_index, potential) in input.potentials.iter().enumerate() {
        if potential.potential_index != expected_index {
            return parse_error(
                SCREEN_FOVRG_RADIAL_PATH,
                0,
                format!(
                    "potential handoff at slot {expected_index} has potential_index {}",
                    potential.potential_index
                ),
            );
        }
        if potential.radial_step != absorber.radial_step {
            return parse_error(
                SCREEN_FOVRG_RADIAL_PATH,
                0,
                format!(
                    "potential {expected_index} radial_step {} does not match absorber radial_step {}",
                    potential.radial_step, absorber.radial_step
                ),
            );
        }
        if potential.exchange_selector != absorber.exchange_selector {
            return parse_error(
                SCREEN_FOVRG_RADIAL_PATH,
                0,
                format!(
                    "potential {expected_index} exchange selector {} does not match absorber selector {}",
                    potential.exchange_selector, absorber.exchange_selector
                ),
            );
        }
        validate_screen_fovrg_radial_handoff_input(&ScreenFovrgRadialHandoffInput {
            potential,
            pot: input.pot,
            config: input.config,
            energies_hartree: input.energies_hartree,
            reference_energies_hartree: input.reference_energies_hartree,
            angular_count: input.angular_count,
            use_hankel_boundary: input.use_hankel_boundary,
        })?;
    }

    Ok(())
}

fn validate_pot_scf_fovrg_source_grid_handoff_input(
    input: &PotScfFovrgSourceGridHandoffInput<'_>,
) -> Result<()> {
    validate_pot_scf_fovrg_source_grid_plan_input(&PotScfFovrgSourceGridPlanInput {
        pot: input.pot,
        config: input.config,
        exchange_selector: input.exchange_selector,
        angular_count: input.angular_count,
        use_hankel_boundary: input.use_hankel_boundary,
    })?;
    validate_pot_scf_fovrg_source_grid_energies(input.energies_hartree)
}

fn validate_pot_scf_fovrg_source_grid_plan_input(
    input: &PotScfFovrgSourceGridPlanInput<'_>,
) -> Result<()> {
    let potential_count = input.pot.potential_count();
    if potential_count == 0 {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "at least one POT SCF potential is required",
        );
    }
    if input.angular_count == 0 {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "at least one POT SCF angular channel is required",
        );
    }
    if input.config.potential_count() < potential_count {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            format!(
                "config.dat potential count {} is smaller than POT SCF potential count {potential_count}",
                input.config.potential_count()
            ),
        );
    }

    let screen = ScreenInput {
        lfxc: input.exchange_selector,
        iend: 0,
        ..ScreenInput::default()
    };
    for potential_index in 0..potential_count {
        validate_screen_potential_kernel_handoff_input(&ScreenPotentialKernelHandoffInput {
            screen: &screen,
            pot: input.pot,
            potential_index,
        })?;
    }

    Ok(())
}

fn validate_pot_scf_fovrg_source_grid_from_plan_input(
    input: &PotScfFovrgSourceGridFromPlanInput<'_>,
) -> Result<()> {
    if input.plan.potential_handoffs.is_empty() {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "POT SCF FOVRG source-grid plan has no potential handoffs",
        );
    }
    if input.plan.angular_count == 0 {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "POT SCF FOVRG source-grid plan has no angular channels",
        );
    }
    validate_pot_scf_fovrg_source_grid_energies(input.energies_hartree)
}

fn validate_pot_scf_fovrg_source_grid_energies(
    energies_hartree: ArrayView1<'_, Complex64>,
) -> Result<()> {
    let energy_count = energies_hartree.len();
    if energy_count == 0 {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "at least one POT SCF energy row is required",
        );
    }
    for (energy_index, energy) in energies_hartree.iter().enumerate() {
        validate_finite(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            energy_index + 1,
            "energy.real",
            energy.re,
        )?;
        validate_finite(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            energy_index + 1,
            "energy.imag",
            energy.im,
        )?;
    }
    Ok(())
}

fn validate_pot_scf_corval_ldos_handoff_input(
    input: &PotScfCorvalLdosHandoffInput<'_>,
) -> Result<()> {
    let potential_count = input.pot.potential_count();
    if potential_count == 0 {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "at least one POT CORVAL potential is required",
        );
    }
    let energy_count = input.energies_hartree.len();
    if energy_count == 0 {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "at least one POT CORVAL energy row is required",
        );
    }
    let (angular_count, requested_potential_count) = input.requested_channels.dim();
    if angular_count == 0 {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            "at least one POT CORVAL angular channel is required",
        );
    }
    if requested_potential_count < potential_count {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            format!(
                "POT CORVAL request mask has {requested_potential_count} potential column(s), expected at least {potential_count}"
            ),
        );
    }
    if input.config.potential_count() < potential_count {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            format!(
                "config.dat potential count {} is smaller than POT CORVAL potential count {potential_count}",
                input.config.potential_count()
            ),
        );
    }
    for (energy_index, energy) in input.energies_hartree.iter().enumerate() {
        validate_finite(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            energy_index + 1,
            "corval_energy.real",
            energy.re,
        )?;
        validate_finite(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            energy_index + 1,
            "corval_energy.imag",
            energy.im,
        )?;
    }

    let screen = ScreenInput {
        lfxc: input.exchange_selector,
        iend: 0,
        ..ScreenInput::default()
    };
    for potential_index in 0..potential_count {
        validate_screen_potential_kernel_handoff_input(&ScreenPotentialKernelHandoffInput {
            screen: &screen,
            pot: input.pot,
            potential_index,
        })?;
    }

    Ok(())
}

fn pot_scf_fovrg_potential_handoffs(
    pot: &PotBinData,
    exchange_selector: i32,
) -> Result<Vec<ScreenPotentialKernelHandoff>> {
    let radius_bohr = screen_radial_grid(
        SCREEN_POT_RADIAL_GRID_STEP,
        SCREEN_POT_RADIAL_GRID_ORIGIN,
        POT_BIN_RADIAL_POINTS,
    )
    .map_err(|source| {
        parse_error_value(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            0,
            format!("radial grid setup failed: {source}"),
        )
    })?;

    let mut potentials = Vec::with_capacity(pot.potential_count());
    for potential_index in 0..pot.potential_count() {
        let mut bounds = screen_radial_bounds(ScreenRadialBoundsInput {
            x0: SCREEN_POT_RADIAL_GRID_ORIGIN,
            dx: SCREEN_POT_RADIAL_GRID_STEP,
            muffin_tin_radius: pot.muffin_tin_radii[potential_index],
            norman_radius: pot.norman_radii[potential_index],
            tail_extension: 0,
            radial_capacity: POT_BIN_RADIAL_POINTS,
            response_capacity: POT_BIN_RADIAL_POINTS,
        })
        .map_err(|source| {
            parse_error_value(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                potential_index + 1,
                format!("radial bounds setup failed: {source}"),
            )
        })?;
        let minimum_active_count = bounds
            .muffin_tin_index_1based
            .checked_add(POT_SCF_FOVRG_MIN_INWARD_HISTORY_ROWS)
            .ok_or_else(|| {
                parse_error_value(
                    POT_SCF_FOVRG_SOURCE_GRID_PATH,
                    potential_index + 1,
                    "minimum active radial count overflowed",
                )
            })?;
        if bounds.active_count < minimum_active_count {
            let tail_extension = isize::try_from(minimum_active_count - bounds.active_count)
                .map_err(|_| {
                    parse_error_value(
                        POT_SCF_FOVRG_SOURCE_GRID_PATH,
                        potential_index + 1,
                        "POT SCF radial tail extension does not fit isize",
                    )
                })?;
            bounds = screen_radial_bounds(ScreenRadialBoundsInput {
                x0: SCREEN_POT_RADIAL_GRID_ORIGIN,
                dx: SCREEN_POT_RADIAL_GRID_STEP,
                muffin_tin_radius: pot.muffin_tin_radii[potential_index],
                norman_radius: pot.norman_radii[potential_index],
                tail_extension,
                radial_capacity: POT_BIN_RADIAL_POINTS,
                response_capacity: POT_BIN_RADIAL_POINTS,
            })
            .map_err(|source| {
                parse_error_value(
                    POT_SCF_FOVRG_SOURCE_GRID_PATH,
                    potential_index + 1,
                    format!("radial tail-extension setup failed: {source}"),
                )
            })?;
        }
        if bounds.active_count < minimum_active_count {
            return parse_error(
                POT_SCF_FOVRG_SOURCE_GRID_PATH,
                potential_index + 1,
                format!(
                    "active_count {} is smaller than required FOVRG count {minimum_active_count}",
                    bounds.active_count
                ),
            );
        }
        let active_count = bounds.active_count;

        potentials.push(ScreenPotentialKernelHandoff {
            radius_bohr: radius_bohr.clone(),
            bounds,
            local_kernel: None,
            response_kernel: Array2::zeros((active_count, active_count)),
            core_large_component: pot.initial_large_component.clone(),
            core_small_component: pot.initial_small_component.clone(),
            potential_index,
            muffin_tin_radius_bohr: pot.muffin_tin_radii[potential_index],
            norman_radius_bohr: pot.norman_radii[potential_index],
            exchange_selector,
            radial_step: SCREEN_POT_RADIAL_GRID_STEP,
        });
    }

    Ok(potentials)
}

fn pot_scf_rholie_active_count(norman_radius: f64, potential_index: usize) -> Result<usize> {
    if !norman_radius.is_finite() || norman_radius <= 0.0 {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            potential_index + 1,
            format!(
                "POT SCF rholie Norman radius must be positive and finite, got {norman_radius}"
            ),
        );
    }
    let raw_index =
        (norman_radius.ln() + SCREEN_POT_RADIAL_GRID_ORIGIN) / SCREEN_POT_RADIAL_GRID_STEP + 5.0;
    if !raw_index.is_finite() || raw_index < 1.0 || raw_index > usize::MAX as f64 {
        return parse_error(
            POT_SCF_FOVRG_SOURCE_GRID_PATH,
            potential_index + 1,
            format!("POT SCF rholie active_count is out of range: {raw_index}"),
        );
    }
    Ok((raw_index.trunc() as usize).min(POT_BIN_RADIAL_POINTS))
}

fn validate_screen_fovrg_prepared_grid(
    prepared: &RhorrpWavefunctionGridPreparation,
    potential: &ScreenPotentialKernelHandoff,
) -> Result<()> {
    let potential_index = potential.potential_index;
    if potential_index >= prepared.potential_count() {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "potential_index {potential_index} exceeds prepared grid potential count {}",
                prepared.potential_count()
            ),
        );
    }
    if prepared.radii.len() < potential.bounds.active_count {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "prepared radial count {} is smaller than active_count {}",
                prepared.radii.len(),
                potential.bounds.active_count
            ),
        );
    }
    let prepared_reference = prepared.reference_indices_1based[potential_index];
    if prepared_reference < potential.bounds.muffin_tin_index_1based
        || prepared_reference > potential.bounds.muffin_tin_next_index_1based
    {
        return parse_error(
            SCREEN_FOVRG_RADIAL_PATH,
            0,
            format!(
                "prepared reference index {prepared_reference} does not cover SCREEN match interval {}..={}",
                potential.bounds.muffin_tin_index_1based,
                potential.bounds.muffin_tin_next_index_1based
            ),
        );
    }
    Ok(())
}

fn validate_screen_response_assembly_handoff_input(
    input: &ScreenResponseAssemblyHandoffInput<'_>,
) -> Result<()> {
    let active_count = input.potential.bounds.active_count;
    if active_count == 0 {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            "active_count must be at least 1",
        );
    }
    let energy_count = input.fms.cluster_greens.nrows();
    let angular_count = input.fms.cluster_greens.ncols();
    if energy_count == 0 {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            "at least one FMS energy row is required",
        );
    }
    if angular_count == 0 {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            "at least one angular channel is required",
        );
    }
    if input.fms.energies_hartree.len() != energy_count {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!(
                "FMS energy grid length {} does not match cluster Green row count {energy_count}",
                input.fms.energies_hartree.len()
            ),
        );
    }
    validate_response_active_vector_len(
        "reference_energies_hartree",
        input.reference_energies_hartree.len(),
        energy_count,
    )?;
    validate_response_active_vector_len(
        "radius_bohr",
        input.potential.radius_bohr.len(),
        active_count,
    )?;
    validate_response_active_vector_len(
        "core_large_component",
        input.potential.core_large_component.len(),
        active_count,
    )?;
    validate_response_active_vector_len(
        "core_small_component",
        input.potential.core_small_component.len(),
        active_count,
    )?;
    validate_response_active_matrix_shape(
        "response_kernel",
        input.potential.response_kernel.nrows(),
        input.potential.response_kernel.ncols(),
        active_count,
    )?;
    if input.potential.bounds.norman_index_1based > active_count {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!(
                "norman_index_1based {} exceeds active_count {active_count}",
                input.potential.bounds.norman_index_1based
            ),
        );
    }
    validate_response_positive_finite(
        "muffin_tin_radius_bohr",
        input.potential.muffin_tin_radius_bohr,
    )?;
    validate_response_positive_finite("radial_step", input.potential.radial_step)?;

    let regular_shape = input.regular_solutions.dim();
    if regular_shape.0 < energy_count
        || regular_shape.1 < active_count
        || regular_shape.2 < angular_count
    {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!(
                "regular_solutions shape {}x{}x{} cannot supply {energy_count}x{active_count}x{angular_count}",
                regular_shape.0, regular_shape.1, regular_shape.2
            ),
        );
    }
    let irregular_shape = input.irregular_solutions.dim();
    if irregular_shape.0 < energy_count
        || irregular_shape.1 < active_count
        || irregular_shape.2 < angular_count
    {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!(
                "irregular_solutions shape {}x{}x{} cannot supply {energy_count}x{active_count}x{angular_count}",
                irregular_shape.0, irregular_shape.1, irregular_shape.2
            ),
        );
    }
    Ok(())
}

fn validate_response_active_vector_len(
    field: &'static str,
    len: usize,
    expected_count: usize,
) -> Result<()> {
    if len < expected_count {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!("{field} length {len} is shorter than required count {expected_count}"),
        );
    }
    Ok(())
}

fn validate_response_active_matrix_shape(
    field: &'static str,
    rows: usize,
    cols: usize,
    active_count: usize,
) -> Result<()> {
    if rows < active_count || cols < active_count {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!("{field} shape {rows}x{cols} is smaller than active_count {active_count}"),
        );
    }
    Ok(())
}

fn validate_response_positive_finite(field: &'static str, value: f64) -> Result<()> {
    validate_finite(SCREEN_RESPONSE_ASSEMBLY_PATH, 0, field, value)?;
    if value <= 0.0 {
        return parse_error(
            SCREEN_RESPONSE_ASSEMBLY_PATH,
            0,
            format!("{field} must be positive"),
        );
    }
    Ok(())
}

fn validate_pot_vector_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual < expected {
        return parse_error(
            SCREEN_POTENTIAL_KERNEL_PATH,
            0,
            format!("pot.bin {field} length {actual} is shorter than expected {expected}"),
        );
    }
    Ok(())
}

fn section_values_as_complex32(
    values: ArrayView2<'_, Complex64>,
    section_number: usize,
) -> Result<Array2<Complex32>> {
    let mut output = Array2::zeros(values.dim());
    for ((row, column), &value) in values.indexed_iter() {
        output[(row, column)] =
            narrow_complex64_to_complex32(value, section_number, row + 1, column + 1)?;
    }
    Ok(output)
}

fn narrow_complex64_to_complex32(
    value: Complex64,
    section_number: usize,
    row: usize,
    column: usize,
) -> Result<Complex32> {
    Ok(Complex32::new(
        narrow_f64_to_f32(value.re, section_number, row, column, "real")?,
        narrow_f64_to_f32(value.im, section_number, row, column, "imaginary")?,
    ))
}

fn narrow_f64_to_f32(
    value: f64,
    section_number: usize,
    row: usize,
    column: usize,
    component: &'static str,
) -> Result<f32> {
    if !value.is_finite() {
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            section_number,
            format!("gg.bin value ({row},{column}) {component} component must be finite"),
        );
    }
    if value.abs() > f32::MAX as f64 {
        return parse_error(
            SCREEN_FMS_CLUSTER_GREEN_PATH,
            section_number,
            format!(
                "gg.bin value ({row},{column}) {component} component does not fit in FEFF default complex precision"
            ),
        );
    }
    Ok(value as f32)
}

fn validate_vtot_dat(data: &VtotDatData) -> Result<()> {
    validate_three_columns(
        VTOT_DAT_PATH,
        "vtot",
        &data.radius_bohr,
        &data.total_potential,
        &data.screened_core_hole_potential,
    )
}

fn validate_active_vector_len(field: &'static str, len: usize, active_count: usize) -> Result<()> {
    if len < active_count {
        return parse_error(
            WSCRN_DAT_PATH,
            0,
            format!("{field} length {len} is shorter than active_count {active_count}"),
        );
    }
    Ok(())
}

fn active_prefix(values: ArrayView1<'_, f64>, active_count: usize) -> Vec<f64> {
    values.iter().take(active_count).copied().collect()
}

fn validate_active_matrix_shape(
    field: &'static str,
    rows: usize,
    cols: usize,
    active_count: usize,
) -> Result<()> {
    if rows < active_count || cols < active_count {
        return parse_error(
            WSCRN_DAT_PATH,
            0,
            format!("{field} shape {rows}x{cols} is smaller than active_count {active_count}"),
        );
    }
    Ok(())
}

fn validate_three_columns(
    path: &'static str,
    table: &'static str,
    first: &Array1<f64>,
    second: &Array1<f64>,
    third: &Array1<f64>,
) -> Result<()> {
    if first.is_empty() {
        return parse_error(
            path,
            0,
            format!("{table} table must contain at least one row"),
        );
    }
    if second.len() != first.len() {
        return parse_error(
            path,
            0,
            format!(
                "second column length {} does not match radius length {}",
                second.len(),
                first.len()
            ),
        );
    }
    if third.len() != first.len() {
        return parse_error(
            path,
            0,
            format!(
                "third column length {} does not match radius length {}",
                third.len(),
                first.len()
            ),
        );
    }

    for (row, ((first, second), third)) in first
        .iter()
        .zip(second.iter())
        .zip(third.iter())
        .enumerate()
    {
        let line = row + 1;
        validate_finite(path, line, "first", *first)?;
        validate_finite(path, line, "second", *second)?;
        validate_finite(path, line, "third", *third)?;
    }
    Ok(())
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    let value = token.replace(['D', 'd'], "E").parse::<f64>().map_err(|_| {
        parse_error_value(
            path,
            line,
            format!("could not parse {field} from {token:?}"),
        )
    })?;
    validate_finite(path, line, field, value)?;
    Ok(value)
}

fn validate_finite(path: &'static str, line: usize, field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(path, line, format!("{field} must be finite"))
    }
}

fn parse_error<T>(path: &'static str, line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(path, line, message))
}

fn parse_error_value(path: &'static str, line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: path.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, Array4, Axis, array};
    use num_complex::{Complex32, Complex64};
    use refeff_core::{
        PotScfContourSourceRowsInput, ScreenClusterResponseSlicesInput, ScreenEnergyStateInput,
        ScreenIntegratedResponseInput, pot_scf_contour_source_rows, screen_cluster_response_slices,
        screen_coulomb_kernel_matrix, screen_energy_state, screen_fms_cluster_green_trace,
        screen_integrated_response, screen_lda_exchange_correlation_kernel, screen_radial_bounds,
        screen_radial_grid,
    };

    use crate::config_dat::{CONFIG_DAT_ORBITAL_COUNT, ConfigDatPotential};
    use crate::gg_dat::{GgDatData, GgDatSection};
    use crate::phase_bin::{PhaseBinData, PhaseBinPotential, PhaseBinScalars};
    use crate::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS,
        PotBinScalars,
    };

    use super::*;

    #[test]
    fn pot_scf_fovrg_trims_trailing_valence_only_zero_component_orbital() -> Result<()> {
        let large = array![
            [1.0, 2.0, 3.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        let small = Array2::zeros((3, 4));
        let electron_counts = array![2.0, 2.0, 1.0, 1.0];
        let valence_counts = array![0.0, 0.0, 0.0, 1.0];

        let count = pot_scf_fovrg_effective_bound_orbital_count(
            large.view(),
            small.view(),
            electron_counts.view(),
            valence_counts.view(),
            4,
        )?;

        assert_eq!(count, 3);
        Ok(())
    }

    #[test]
    fn pot_scf_fovrg_rejects_zero_component_core_orbital() {
        let large = array![
            [1.0, 2.0, 3.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        let small = Array2::zeros((3, 4));
        let electron_counts = array![2.0, 2.0, 1.0, 1.0];
        let valence_counts = array![0.0, 0.0, 0.0, 0.0];

        let err = pot_scf_fovrg_effective_bound_orbital_count(
            large.view(),
            small.view(),
            electron_counts.view(),
            valence_counts.view(),
            4,
        )
        .expect_err("zero-component core orbital should be rejected");

        assert!(err.to_string().contains("has no radial component"), "{err}");
    }

    #[test]
    fn pot_scf_fovrg_rejects_non_trailing_zero_component_orbital() {
        let large = array![[1.0, 0.0, 3.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]];
        let small = Array2::zeros((3, 3));
        let electron_counts = array![2.0, 1.0, 1.0];
        let valence_counts = array![0.0, 1.0, 0.0];

        let err = pot_scf_fovrg_effective_bound_orbital_count(
            large.view(),
            small.view(),
            electron_counts.view(),
            valence_counts.view(),
            3,
        )
        .expect_err("non-trailing zero-component orbital should be rejected");

        assert!(err.to_string().contains("is not trailing"), "{err}");
    }

    #[test]
    fn parses_wscrn_dat() -> Result<()> {
        let parsed = parse_wscrn_dat(WSCRN_DAT)?;
        assert_eq!(
            parsed.header_lines,
            vec![" # r       w_scrn(r)      v_ch(r)"]
        );
        assert_eq!(parsed.row_count(), 3);
        assert_eq!(parsed.radius_bohr[0], 0.150_733_046_3E-03);
        assert_eq!(parsed.screened_potential[1], 0.267_288_167_8E+02);
        assert_eq!(parsed.core_hole_potential[2], 0.291_616_320_4E+02);

        let rendered = wscrn_dat_string(&parsed)?;
        assert_eq!(rendered, WSCRN_DAT);
        assert_eq!(parse_wscrn_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_vtot_dat() -> Result<()> {
        let parsed = parse_vtot_dat(VTOT_DAT)?;
        assert!(parsed.header_lines.is_empty());
        assert_eq!(parsed.row_count(), 3);
        assert_eq!(parsed.radius_bohr[0], 0.150_733_046_3E-03);
        assert_eq!(parsed.total_potential[1], -0.182_900_133_6E+06);
        assert_eq!(parsed.screened_core_hole_potential[2], 0.267_288_030_6E+02);

        let rendered = vtot_dat_string(&parsed)?;
        assert_eq!(rendered, VTOT_DAT);
        assert_eq!(parse_vtot_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents() -> Result<()> {
        let parsed = parse_wscrn_dat("# h\n1.0D+00 2.0D+00 3.0D+00\n")?;
        assert_eq!(parsed.radius_bohr[0], 1.0);
        assert_eq!(parsed.screened_potential[0], 2.0);
        assert_eq!(parsed.core_hole_potential[0], 3.0);
        Ok(())
    }

    #[test]
    fn builds_wscrn_dat_from_screen_response() -> Result<()> {
        let header_lines = vec![" # r       w_scrn(r)      v_ch(r)".to_string()];
        let radius = array![0.1, 0.2, 0.3];
        let bare = array![0.8, 0.2, 9.0];
        let kernel = array![[2.0, 0.5], [0.5, 1.0]];
        let susceptibility = array![
            [Complex64::new(1.0, 0.1), Complex64::new(2.0, 0.2)],
            [Complex64::new(3.0, 0.3), Complex64::new(4.0, 0.05)]
        ];

        let data = wscrn_dat_from_screen_response(WscrnDatFromScreenResponseInput {
            header_lines: &header_lines,
            radius_bohr: radius.view(),
            core_hole_potential: bare.view(),
            response_kernel: kernel.view(),
            susceptibility: susceptibility.view(),
            active_count: 2,
        })?;

        assert_eq!(data.header_lines, header_lines);
        assert_eq!(data.row_count(), 2);
        assert_eq!(data.radius_bohr.to_vec(), vec![0.1, 0.2]);
        assert_eq!(data.core_hole_potential.to_vec(), vec![0.8, 0.2]);
        assert_close(data.screened_potential[0], 612.0 / 323.0, 1.0e-14);
        assert_close(data.screened_potential[1], 328.0 / 323.0, 1.0e-14);

        let rendered = wscrn_dat_string(&data)?;
        let parsed = parse_wscrn_dat(&rendered)?;
        assert_eq!(parsed.header_lines, data.header_lines);
        assert_eq!(parsed.radius_bohr, data.radius_bohr);
        assert_eq!(parsed.core_hole_potential, data.core_hole_potential);
        assert_close(
            parsed.screened_potential[0],
            data.screened_potential[0],
            1.0e-9,
        );
        assert_close(
            parsed.screened_potential[1],
            data.screened_potential[1],
            1.0e-9,
        );
        Ok(())
    }

    #[test]
    fn builds_wscrn_dat_from_screen_response_slices() -> Result<()> {
        let header_lines = vec![" # r       w_scrn(r)      v_ch(r)".to_string()];
        let radius = array![0.1, 0.2, 0.3];
        let bare = array![0.8, 0.2, 9.0];
        let kernel = array![[2.0, 0.5], [0.5, 1.0]];
        let susceptibility = array![
            [Complex64::new(1.0, 0.1), Complex64::new(2.0, 0.2)],
            [Complex64::new(2.0, 0.2), Complex64::new(4.0, 0.05)]
        ];
        let energies = array![
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0)
        ];
        let response_slices = Array1::from_iter(susceptibility.iter().copied())
            .into_shape_with_order((1, 2, 2))
            .expect("susceptibility shape")
            .broadcast((3, 2, 2))
            .expect("broadcast response slices")
            .mapv(|value| value / 4.0);

        let expected = wscrn_dat_from_screen_response(WscrnDatFromScreenResponseInput {
            header_lines: &header_lines,
            radius_bohr: radius.view(),
            core_hole_potential: bare.view(),
            response_kernel: kernel.view(),
            susceptibility: susceptibility.view(),
            active_count: 2,
        })?;
        let actual =
            wscrn_dat_from_screen_response_slices(WscrnDatFromScreenResponseSlicesInput {
                header_lines: &header_lines,
                radius_bohr: radius.view(),
                core_hole_potential: bare.view(),
                response_kernel: kernel.view(),
                energies: energies.view(),
                response_slices: response_slices.view(),
                active_count: 2,
            })?;

        assert_eq!(actual.header_lines, expected.header_lines);
        assert_eq!(actual.radius_bohr, expected.radius_bohr);
        assert_eq!(actual.core_hole_potential, expected.core_hole_potential);
        assert_close(actual.screened_potential[0], 612.0 / 391.0, 1.0e-14);
        assert_close(actual.screened_potential[1], 272.0 / 391.0, 1.0e-14);
        assert_close(
            actual.screened_potential[0],
            expected.screened_potential[0],
            1.0e-14,
        );
        assert_close(
            actual.screened_potential[1],
            expected.screened_potential[1],
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn builds_wscrn_dat_from_core_hole_response() -> Result<()> {
        let header_lines = vec!["# screen core-hole response".to_string()];
        let radius = array![1.0, 2.0, 9.0];
        let large = array![1.0, 2.0, 9.0];
        let small = array![0.0, 0.0, 9.0];
        let kernel = array![[2.0, 0.5], [0.5, 1.0]];
        let susceptibility = array![
            [Complex64::new(1.0, 0.1), Complex64::new(2.0, 0.2)],
            [Complex64::new(3.0, 0.3), Complex64::new(4.0, 0.05)]
        ];

        let data = wscrn_dat_from_core_hole_response(WscrnDatFromCoreHoleResponseInput {
            header_lines: &header_lines,
            radius_bohr: radius.view(),
            large_component: large.view(),
            small_component: small.view(),
            response_kernel: kernel.view(),
            susceptibility: susceptibility.view(),
            radial_step: 0.1,
            active_count: 2,
        })?;

        assert_eq!(data.header_lines, header_lines);
        assert_eq!(data.row_count(), 2);
        assert_eq!(data.radius_bohr.to_vec(), vec![1.0, 2.0]);
        assert_close(data.core_hole_potential[0], 0.5, 1.0e-14);
        assert_close(data.core_hole_potential[1], 0.45, 1.0e-14);
        assert_close(data.screened_potential[0], 493.0 / 323.0, 1.0e-14);
        assert_close(data.screened_potential[1], 374.0 / 323.0, 1.0e-14);

        let rendered = wscrn_dat_string(&data)?;
        let parsed = parse_wscrn_dat(&rendered)?;
        assert_eq!(parsed.header_lines, data.header_lines);
        assert_eq!(parsed.radius_bohr, data.radius_bohr);
        assert_close(
            parsed.core_hole_potential[0],
            data.core_hole_potential[0],
            1.0e-9,
        );
        assert_close(
            parsed.screened_potential[1],
            data.screened_potential[1],
            1.0e-9,
        );
        Ok(())
    }

    #[test]
    fn builds_screen_fms_cluster_green_handoff_from_phase_and_gg() -> Result<()> {
        let phase = sample_screen_phase_bin(2, 1);
        let green = sample_screen_gg_bin(2, 4);

        let handoff = screen_fms_cluster_green_handoff(ScreenFmsClusterGreenHandoffInput {
            phase: &phase,
            green: &green,
            potential_index: 0,
            spin_index: 0,
            angular_count: 2,
        })?;

        assert_eq!(handoff.energies_hartree, phase.energy_grid);
        assert_eq!(handoff.cluster_greens.dim(), (2, 2));
        assert_eq!(handoff.potential_index, 0);
        assert_eq!(handoff.spin_index, 0);

        for energy in 0..2 {
            let scattering = green.sections[energy]
                .values
                .mapv(|value| num_complex::Complex32::new(value.re as f32, value.im as f32));
            for angular in 0..2 {
                let phase_slot = phase.potentials[0].lmax + angular;
                let expected = screen_fms_cluster_green_trace(
                    scattering.view(),
                    phase.potentials[0].phase_shifts[(energy, phase_slot, 0)],
                    angular,
                )
                .expect("expected SCREEN FMS trace");
                assert_complex_close(handoff_value(&handoff, energy, angular), expected, 1.0e-12);
            }
        }

        let negative_slot_phase = phase.potentials[0].phase_shifts[(0, 0, 0)];
        let positive_slot_phase = phase.potentials[0].phase_shifts[(0, 2, 0)];
        assert_ne!(negative_slot_phase, positive_slot_phase);
        Ok(())
    }

    #[test]
    fn builds_pot_scf_fms_source_grid_handoff_from_scattering_matrices() -> Result<()> {
        let energies = Array1::from_vec(vec![Complex64::new(0.2, 0.01), Complex64::new(0.4, 0.02)]);
        let angular_count = 2;
        let potential_count = 2;
        let channel_count = angular_count * angular_count;
        let phase_shifts = Array3::from_shape_fn(
            (energies.len(), angular_count, potential_count),
            |(energy, angular, potential)| {
                Complex64::new(
                    0.05 * (energy + 1) as f64 + 0.02 * angular as f64,
                    -0.01 * potential as f64,
                )
            },
        );
        let scattering_matrices = Array4::from_shape_fn(
            (
                energies.len(),
                channel_count,
                channel_count,
                potential_count,
            ),
            |(energy, row, column, potential)| {
                let base = 1.0
                    + energy as f32
                    + 0.1 * row as f32
                    + 0.03 * column as f32
                    + potential as f32;
                Complex32::new(base, -0.2 * base)
            },
        );

        let handoff = pot_scf_fms_source_grid_handoff(PotScfFmsSourceGridHandoffInput {
            energies_hartree: energies.view(),
            phase_shifts: phase_shifts.view(),
            scattering_matrices: scattering_matrices.view(),
            angular_count,
        })?;

        assert_eq!(handoff.energies_hartree, energies);
        assert_eq!(
            handoff.scattering_trace.dim(),
            (energies.len(), angular_count, potential_count)
        );
        for energy in 0..energies.len() {
            let scattering_energy = scattering_matrices.index_axis(Axis(0), energy);
            for potential in 0..potential_count {
                let scattering = scattering_energy.index_axis(Axis(2), potential);
                for angular in 0..angular_count {
                    let expected = screen_fms_cluster_green_trace(
                        scattering,
                        phase_shifts[(energy, angular, potential)],
                        angular,
                    )
                    .expect("expected POT SCF FMS trace");
                    assert_complex_close(
                        handoff.scattering_trace[(energy, angular, potential)],
                        expected,
                        1.0e-12,
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn screen_fms_cluster_green_handoff_rejects_short_scattering_matrix() {
        let phase = sample_screen_phase_bin(2, 1);
        let green = sample_screen_gg_bin(2, 3);

        let error = screen_fms_cluster_green_handoff(ScreenFmsClusterGreenHandoffInput {
            phase: &phase,
            green: &green,
            potential_index: 0,
            spin_index: 0,
            angular_count: 2,
        })
        .unwrap_err();

        assert!(error.to_string().contains("required order 4"));
    }

    #[test]
    fn builds_screen_potential_kernel_handoff_for_rpa_branch() -> Result<()> {
        let mut pot = sample_screen_pot_bin(1);
        let radii = screen_radial_grid(
            SCREEN_POT_RADIAL_GRID_STEP,
            SCREEN_POT_RADIAL_GRID_ORIGIN,
            POT_BIN_RADIAL_POINTS,
        )
        .expect("sample radii");
        pot.muffin_tin_radii[0] = radii[3];
        pot.norman_radii[0] = radii[6];
        let screen = ScreenInput {
            lfxc: 0,
            iend: 1,
            ..ScreenInput::default()
        };

        let handoff = screen_potential_kernel_handoff(ScreenPotentialKernelHandoffInput {
            screen: &screen,
            pot: &pot,
            potential_index: 0,
        })?;
        let expected_bounds = screen_radial_bounds(ScreenRadialBoundsInput {
            x0: SCREEN_POT_RADIAL_GRID_ORIGIN,
            dx: SCREEN_POT_RADIAL_GRID_STEP,
            muffin_tin_radius: pot.muffin_tin_radii[0],
            norman_radius: pot.norman_radii[0],
            tail_extension: 1,
            radial_capacity: POT_BIN_RADIAL_POINTS,
            response_capacity: POT_BIN_RADIAL_POINTS,
        })
        .expect("sample bounds");
        let expected_kernel = screen_coulomb_kernel_matrix(
            radii.as_slice().expect("radii storage"),
            expected_bounds.active_count,
            None,
        )
        .expect("sample kernel");

        assert_eq!(handoff.radius_bohr, radii);
        assert_eq!(handoff.bounds, expected_bounds);
        assert!(handoff.local_kernel.is_none());
        assert_eq!(handoff.response_kernel, expected_kernel);
        assert_eq!(handoff.core_large_component, pot.initial_large_component);
        assert_eq!(handoff.core_small_component, pot.initial_small_component);
        assert_eq!(handoff.potential_index, 0);
        assert_eq!(handoff.radial_step, SCREEN_POT_RADIAL_GRID_STEP);
        Ok(())
    }

    #[test]
    fn builds_screen_potential_kernel_handoff_with_local_field() -> Result<()> {
        let mut pot = sample_screen_pot_bin(1);
        let radii = screen_radial_grid(
            SCREEN_POT_RADIAL_GRID_STEP,
            SCREEN_POT_RADIAL_GRID_ORIGIN,
            POT_BIN_RADIAL_POINTS,
        )
        .expect("sample radii");
        pot.muffin_tin_radii[0] = radii[2];
        pot.norman_radii[0] = radii[5];
        let screen = ScreenInput {
            lfxc: 2,
            ..ScreenInput::default()
        };

        let handoff = screen_potential_kernel_handoff(ScreenPotentialKernelHandoffInput {
            screen: &screen,
            pot: &pot,
            potential_index: 0,
        })?;
        let density = pot
            .electron_density
            .column(0)
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let expected_local = screen_lda_exchange_correlation_kernel(
            radii.as_slice().expect("radii storage"),
            &density,
            2,
            handoff.bounds.active_count,
        )
        .expect("sample local field");
        let expected_kernel = screen_coulomb_kernel_matrix(
            radii.as_slice().expect("radii storage"),
            handoff.bounds.active_count,
            Some(expected_local.as_slice().expect("local storage")),
        )
        .expect("sample kernel");

        assert_eq!(
            handoff.local_kernel.as_ref().expect("local kernel"),
            &expected_local
        );
        assert_eq!(handoff.response_kernel, expected_kernel);
        assert!(
            handoff.response_kernel[(0, 0)]
                != screen_coulomb_kernel_matrix(
                    radii.as_slice().expect("radii storage"),
                    handoff.bounds.active_count,
                    None,
                )
                .expect("rpa kernel")[(0, 0)]
        );
        Ok(())
    }

    #[test]
    fn screen_potential_kernel_handoff_rejects_bad_potential_index() {
        let pot = sample_screen_pot_bin(1);
        let screen = ScreenInput::default();

        let error = screen_potential_kernel_handoff(ScreenPotentialKernelHandoffInput {
            screen: &screen,
            pot: &pot,
            potential_index: 1,
        })
        .unwrap_err();

        assert!(error.to_string().contains("potential_index 1"));
    }

    #[test]
    fn builds_screen_response_assembly_handoff_from_radial_cubes() -> Result<()> {
        let (potential, fms, references, regular, irregular) = sample_screen_response_source()?;
        let header_lines = vec!["# assembled from source radial cubes".to_string()];

        let handoff = screen_response_assembly_handoff(ScreenResponseAssemblyHandoffInput {
            potential: &potential,
            fms: &fms,
            reference_energies_hartree: references.view(),
            regular_solutions: regular.view(),
            irregular_solutions: irregular.view(),
            header_lines: &header_lines,
        })?;

        let expected_wave_numbers =
            Array1::from_iter(fms.energies_hartree.iter().zip(references.iter()).map(
                |(&energy, &reference)| {
                    screen_energy_state(ScreenEnergyStateInput {
                        energy,
                        reference_energy: reference,
                        muffin_tin_radius: potential.muffin_tin_radius_bohr,
                        exchange_selector: potential.exchange_selector,
                    })
                    .expect("sample energy state")
                    .wave_number
                },
            ));
        let expected_slices = screen_cluster_response_slices(ScreenClusterResponseSlicesInput {
            radii: potential.radius_bohr.as_slice().expect("radii storage"),
            regular_solutions: regular.view(),
            irregular_solutions: irregular.view(),
            cluster_greens: fms.cluster_greens.view(),
            wave_numbers: expected_wave_numbers.view(),
            dx: potential.radial_step,
            angular_momentum_count: fms.cluster_greens.ncols(),
            active_count: potential.bounds.active_count,
            fms_count: potential.bounds.norman_index_1based,
        })
        .expect("sample response slices");
        let expected_susceptibility = screen_integrated_response(ScreenIntegratedResponseInput {
            energies: fms.energies_hartree.view(),
            response_slices: expected_slices.view(),
            active_count: potential.bounds.active_count,
        })
        .expect("sample susceptibility");
        let expected_wscrn =
            wscrn_dat_from_core_hole_response(WscrnDatFromCoreHoleResponseInput {
                header_lines: &header_lines,
                radius_bohr: potential.radius_bohr.view(),
                large_component: potential.core_large_component.view(),
                small_component: potential.core_small_component.view(),
                response_kernel: potential.response_kernel.view(),
                susceptibility: expected_susceptibility.view(),
                radial_step: potential.radial_step,
                active_count: potential.bounds.active_count,
            })?;

        assert_eq!(handoff.wave_numbers, expected_wave_numbers);
        assert_eq!(handoff.response_slices, expected_slices);
        assert_eq!(handoff.susceptibility, expected_susceptibility);
        assert_eq!(handoff.wscrn, expected_wscrn);
        assert_eq!(handoff.wscrn.header_lines, header_lines);
        assert!(
            handoff
                .wscrn
                .core_hole_potential
                .iter()
                .any(|value| value.abs() > 0.0)
        );
        assert_eq!(handoff.wscrn.row_count(), potential.bounds.active_count);
        Ok(())
    }

    #[test]
    fn builds_screen_fovrg_radial_handoff_from_pot_config_energy_state() -> Result<()> {
        let mut pot = sample_screen_pot_bin(1);
        let radii = screen_radial_grid(
            SCREEN_POT_RADIAL_GRID_STEP,
            SCREEN_POT_RADIAL_GRID_ORIGIN,
            POT_BIN_RADIAL_POINTS,
        )
        .expect("sample radii");
        pot.muffin_tin_radii[0] = radii[12];
        pot.norman_radii[0] = radii[20];
        let config = sample_screen_config_dat(1);
        let screen = ScreenInput {
            lfxc: 0,
            iend: 0,
            ..ScreenInput::default()
        };
        let potential = screen_potential_kernel_handoff(ScreenPotentialKernelHandoffInput {
            screen: &screen,
            pot: &pot,
            potential_index: 0,
        })?;
        let phase = sample_screen_phase_bin(2, 0);
        let references = phase.reference_energy.column(0).to_owned();

        let handoff = screen_fovrg_radial_handoff(ScreenFovrgRadialHandoffInput {
            potential: &potential,
            pot: &pot,
            config: &config,
            energies_hartree: phase.energy_grid.view(),
            reference_energies_hartree: references.view(),
            angular_count: 1,
            use_hankel_boundary: true,
        })?;

        let expected_reference = handoff.reference_energies_hartree[0];
        let expected_references = Array1::from_elem(phase.energy_grid.len(), expected_reference);
        let expected_state = screen_energy_state(ScreenEnergyStateInput {
            energy: phase.energy_grid[0],
            reference_energy: expected_reference,
            muffin_tin_radius: potential.muffin_tin_radius_bohr,
            exchange_selector: potential.exchange_selector,
        })
        .expect("sample energy state");
        assert_eq!(handoff.potential_index, 0);
        assert_eq!(handoff.reference_energies_hartree, expected_references);
        assert_complex_close(handoff.wave_numbers[0], expected_state.wave_number, 1.0e-12);
        assert_eq!(
            handoff.matched.solved.radial_cubes.regular_large.dim(),
            (2, potential.bounds.active_count, 1)
        );
        assert_eq!(
            handoff.matched.solved.radial_cubes.irregular_large.dim(),
            (2, potential.bounds.active_count, 1)
        );
        assert!(handoff.matched.phase_shifts[(0, 0)].re.is_finite());
        assert!(handoff.matched.phase_amplitudes[(0, 0)].re.is_finite());

        let phases = screen_fovrg_phase_handoff(ScreenFovrgRadialHandoffInput {
            potential: &potential,
            pot: &pot,
            config: &config,
            energies_hartree: phase.energy_grid.view(),
            reference_energies_hartree: references.view(),
            angular_count: 1,
            use_hankel_boundary: true,
        })?;
        assert_eq!(phases.potential_index, handoff.potential_index);
        assert_eq!(
            phases.reference_energies_hartree,
            handoff.reference_energies_hartree
        );
        assert_eq!(phases.wave_numbers, handoff.wave_numbers);
        assert_eq!(phases.phase_shifts, handoff.matched.phase_shifts);
        assert_eq!(phases.phase_amplitudes, handoff.matched.phase_amplitudes);

        let potentials = vec![potential.clone()];
        let grid = screen_fovrg_phase_grid_handoff(ScreenFovrgPhaseGridHandoffInput {
            potentials: &potentials,
            absorber_potential_index: 0,
            pot: &pot,
            config: &config,
            energies_hartree: phase.energy_grid.view(),
            reference_energies_hartree: references.view(),
            angular_count: 1,
            use_hankel_boundary: true,
        })?;
        assert_eq!(grid.absorber_radial, handoff);
        assert_eq!(grid.phase_shifts.dim(), (2, 1, 1));
        assert_eq!(grid.phase_amplitudes.dim(), (2, 1, 1));
        for energy in 0..phase.energy_grid.len() {
            assert_eq!(
                grid.phase_shifts[(energy, 0, 0)],
                handoff.matched.phase_shifts[(energy, 0)]
            );
            assert_eq!(
                grid.phase_amplitudes[(energy, 0, 0)],
                handoff.matched.phase_amplitudes[(energy, 0)]
            );
        }

        let mut pot_grid = sample_screen_pot_bin(2);
        pot_grid.muffin_tin_radii[0] = radii[12];
        pot_grid.muffin_tin_radii[1] = radii[14];
        pot_grid.norman_radii[0] = radii[20];
        pot_grid.norman_radii[1] = radii[22];
        let config_grid = sample_screen_config_dat(2);
        let source_grid = pot_scf_fovrg_source_grid_handoff(PotScfFovrgSourceGridHandoffInput {
            pot: &pot_grid,
            config: &config_grid,
            energies_hartree: phase.energy_grid.view(),
            exchange_selector: 0,
            angular_count: 1,
            use_hankel_boundary: true,
        })?;
        let source_plan = pot_scf_fovrg_source_grid_plan(PotScfFovrgSourceGridPlanInput {
            pot: &pot_grid,
            config: &config_grid,
            exchange_selector: 0,
            angular_count: 1,
            use_hankel_boundary: true,
        })?;
        let planned_source_grid =
            pot_scf_fovrg_source_grid_handoff_from_plan(PotScfFovrgSourceGridFromPlanInput {
                plan: &source_plan,
                energies_hartree: phase.energy_grid.view(),
            })?;
        assert_eq!(planned_source_grid, source_grid);
        assert_eq!(source_grid.energies_hartree, phase.energy_grid);
        assert_eq!(source_grid.radial_handoffs.len(), 2);
        assert_eq!(source_grid.wave_numbers.dim(), (2, 2));
        assert_eq!(source_grid.phase_shifts.dim(), (2, 1, 2));
        assert_eq!(source_grid.phase_amplitudes.dim(), (2, 1, 2));
        let expected_rholie_counts = Array1::from_vec(vec![
            pot_scf_rholie_active_count(pot_grid.norman_radii[0], 0)?,
            pot_scf_rholie_active_count(pot_grid.norman_radii[1], 1)?,
        ]);
        assert_eq!(source_grid.rholie_active_counts, expected_rholie_counts);
        assert!(
            source_grid
                .radial_active_counts
                .iter()
                .zip(source_grid.rholie_active_counts.iter())
                .all(|(radial_count, rholie_count)| radial_count >= rholie_count)
        );
        let expected_active_count = source_grid
            .radial_active_counts
            .iter()
            .copied()
            .max()
            .unwrap();
        assert_eq!(
            source_grid.regular_large.dim(),
            (2, 2, 1, expected_active_count)
        );
        assert_eq!(
            source_grid.regular_small.dim(),
            source_grid.regular_large.dim()
        );
        assert_eq!(
            source_grid.irregular_large.dim(),
            source_grid.regular_large.dim()
        );
        assert_eq!(
            source_grid.irregular_small.dim(),
            source_grid.regular_large.dim()
        );
        for potential_index in 0..2 {
            let radial = &source_grid.radial_handoffs[potential_index];
            assert_eq!(
                source_grid.radial_active_counts[potential_index],
                radial.matched.solved.radial_cubes.regular_large.dim().1
            );
            assert_eq!(
                source_grid.wave_numbers[(0, potential_index)],
                radial.wave_numbers[0]
            );
            assert_eq!(
                source_grid.phase_shifts[(0, 0, potential_index)],
                radial.matched.phase_shifts[(0, 0)]
            );
            assert_eq!(
                source_grid.regular_large[(0, potential_index, 0, 0)],
                radial.matched.solved.radial_cubes.regular_large[(0, 0, 0)]
            );
        }

        let fms = screen_fms_cluster_green_handoff(ScreenFmsClusterGreenHandoffInput {
            phase: &phase,
            green: &sample_screen_gg_bin(2, 1),
            potential_index: 0,
            spin_index: 0,
            angular_count: 1,
        })?;
        let response = screen_response_assembly_handoff(ScreenResponseAssemblyHandoffInput {
            potential: &potential,
            fms: &fms,
            reference_energies_hartree: handoff.reference_energies_hartree.view(),
            regular_solutions: handoff.matched.solved.radial_cubes.regular_large.view(),
            irregular_solutions: handoff.matched.solved.radial_cubes.irregular_large.view(),
            header_lines: &[],
        })?;
        assert_eq!(response.wscrn.row_count(), potential.bounds.active_count);
        Ok(())
    }

    #[test]
    fn pot_scf_corval_ldos_handoff_matches_requested_full_source_rows() -> Result<()> {
        let radii = screen_radial_grid(
            SCREEN_POT_RADIAL_GRID_STEP,
            SCREEN_POT_RADIAL_GRID_ORIGIN,
            POT_BIN_RADIAL_POINTS,
        )
        .expect("sample radii");
        let mut pot = sample_screen_pot_bin(2);
        pot.muffin_tin_radii[0] = radii[12];
        pot.muffin_tin_radii[1] = radii[14];
        pot.norman_radii[0] = radii[20];
        pot.norman_radii[1] = radii[22];
        let config = sample_screen_config_dat(2);
        let phase = sample_screen_phase_bin(2, 0);
        let requested = array![[true, false], [false, true]];

        let narrow = pot_scf_corval_ldos_handoff(PotScfCorvalLdosHandoffInput {
            pot: &pot,
            config: &config,
            energies_hartree: phase.energy_grid.view(),
            exchange_selector: 0,
            requested_channels: requested.view(),
            use_hankel_boundary: true,
        })?;
        let full_grid = pot_scf_fovrg_source_grid_handoff(PotScfFovrgSourceGridHandoffInput {
            pot: &pot,
            config: &config,
            energies_hartree: phase.energy_grid.view(),
            exchange_selector: 0,
            angular_count: 2,
            use_hankel_boundary: true,
        })?;
        let scattering_trace = Array3::<Complex32>::zeros((phase.energy_grid.len(), 2, 2));
        let full_rows = pot_scf_contour_source_rows(PotScfContourSourceRowsInput {
            source_energies: full_grid.energies_hartree.view(),
            source_radii: full_grid.source_radii.view(),
            output_radii: full_grid.source_radii.view(),
            radial_step: SCREEN_POT_RADIAL_GRID_STEP,
            highest_potential_index: 1,
            norman_radii: pot.norman_radii.view(),
            wave_numbers: full_grid.wave_numbers.view(),
            angular_count: 2,
            scattering_trace: scattering_trace.view(),
            regular_large: full_grid.regular_large.view(),
            regular_small: full_grid.regular_small.view(),
            irregular_large: full_grid.irregular_large.view(),
            irregular_small: full_grid.irregular_small.view(),
        })
        .expect("full source rows");

        assert_eq!(narrow.energies_hartree, phase.energy_grid);
        assert_eq!(narrow.embedded_ldos_source.dim(), (2, 2, 2));
        for energy in 0..phase.energy_grid.len() {
            assert_complex_close(
                narrow.embedded_ldos_source[(energy, 0, 0)],
                full_rows.embedded_ldos_source[(energy, 0, 0)],
                1.0e-10,
            );
            assert_complex_close(
                narrow.embedded_ldos_source[(energy, 1, 1)],
                full_rows.embedded_ldos_source[(energy, 1, 1)],
                1.0e-10,
            );
            assert_eq!(
                narrow.embedded_ldos_source[(energy, 1, 0)],
                Complex64::new(0.0, 0.0)
            );
            assert_eq!(
                narrow.embedded_ldos_source[(energy, 0, 1)],
                Complex64::new(0.0, 0.0)
            );
        }
        Ok(())
    }

    #[test]
    fn screen_response_assembly_handoff_rejects_short_radial_cubes() -> Result<()> {
        let (potential, fms, references, _regular, irregular) = sample_screen_response_source()?;
        let short_regular = Array3::zeros((
            fms.cluster_greens.nrows(),
            potential.bounds.active_count - 1,
            fms.cluster_greens.ncols(),
        ));
        let header_lines = Vec::new();

        let error = screen_response_assembly_handoff(ScreenResponseAssemblyHandoffInput {
            potential: &potential,
            fms: &fms,
            reference_energies_hartree: references.view(),
            regular_solutions: short_regular.view(),
            irregular_solutions: irregular.view(),
            header_lines: &header_lines,
        })
        .unwrap_err();

        assert!(error.to_string().contains("regular_solutions shape"));
        Ok(())
    }

    #[test]
    fn builds_vtot_dat_from_wscrn_and_total_potential() -> Result<()> {
        let wscrn = parse_wscrn_dat(WSCRN_DAT)?;
        let total_potential = array![-1000.0, -1000.25, -1000.5, -1000.75];

        let data = vtot_dat_from_wscrn_and_total_potential(&wscrn, total_potential.view())?;

        assert_eq!(data.header_lines, Vec::<String>::new());
        assert_eq!(data.row_count(), 3);
        assert_eq!(data.radius_bohr, wscrn.radius_bohr);
        assert_eq!(
            data.total_potential.to_vec(),
            vec![-1000.0, -1000.25, -1000.5]
        );
        assert_eq!(data.screened_core_hole_potential, wscrn.screened_potential);
        let rendered = vtot_dat_string(&data)?;
        assert_eq!(parse_vtot_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn vtot_dat_from_wscrn_and_total_potential_rejects_empty_overlap() {
        let wscrn = parse_wscrn_dat(WSCRN_DAT).expect("sample wscrn.dat");
        let total_potential = Array1::<f64>::zeros(0);

        let error =
            vtot_dat_from_wscrn_and_total_potential(&wscrn, total_potential.view()).unwrap_err();

        assert!(error.to_string().contains("at least one shared"));
    }

    #[test]
    fn wscrn_dat_from_screen_response_rejects_short_active_inputs() {
        let header_lines = Vec::new();
        let radius = array![0.1];
        let bare = array![0.8, 0.2];
        let kernel = array![[1.0, 0.0], [0.0, 1.0]];
        let susceptibility = array![
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
        ];

        let error = wscrn_dat_from_screen_response(WscrnDatFromScreenResponseInput {
            header_lines: &header_lines,
            radius_bohr: radius.view(),
            core_hole_potential: bare.view(),
            response_kernel: kernel.view(),
            susceptibility: susceptibility.view(),
            active_count: 2,
        })
        .unwrap_err();

        assert!(error.to_string().contains("radius_bohr length 1"));
    }

    #[test]
    fn wscrn_dat_from_core_hole_response_rejects_short_component_inputs() {
        let header_lines = Vec::new();
        let radius = array![1.0, 2.0];
        let large = array![1.0];
        let small = array![0.0, 0.0];
        let kernel = array![[1.0, 0.0], [0.0, 1.0]];
        let susceptibility = array![
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
        ];

        let error = wscrn_dat_from_core_hole_response(WscrnDatFromCoreHoleResponseInput {
            header_lines: &header_lines,
            radius_bohr: radius.view(),
            large_component: large.view(),
            small_component: small.view(),
            response_kernel: kernel.view(),
            susceptibility: susceptibility.view(),
            radial_step: 0.1,
            active_count: 2,
        })
        .unwrap_err();

        assert!(error.to_string().contains("large_component length 1"));
    }

    #[test]
    fn rejects_bad_screen_tables() {
        assert!(parse_wscrn_dat("").is_err());
        assert!(parse_wscrn_dat("# only a header\n").is_err());
        assert!(parse_wscrn_dat("1 2\n").is_err());
        assert!(parse_wscrn_dat("1 2 3 4\n").is_err());
        assert!(parse_wscrn_dat("1 NaN 3\n").is_err());
        assert!(
            wscrn_dat_string(&WscrnDatData {
                header_lines: Vec::new(),
                radius_bohr: array![1.0, 2.0],
                screened_potential: array![3.0],
                core_hole_potential: array![4.0, 5.0],
            })
            .is_err()
        );
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
        );
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        assert_close(actual.re, expected.re, tolerance);
        assert_close(actual.im, expected.im, tolerance);
    }

    fn handoff_value(
        handoff: &ScreenFmsClusterGreenHandoff,
        energy: usize,
        angular: usize,
    ) -> Complex64 {
        handoff.cluster_greens[(energy, angular)]
    }

    type SampleScreenResponseSource = (
        ScreenPotentialKernelHandoff,
        ScreenFmsClusterGreenHandoff,
        Array1<Complex64>,
        Array3<Complex64>,
        Array3<Complex64>,
    );

    fn sample_screen_response_source() -> Result<SampleScreenResponseSource> {
        let mut pot = sample_screen_pot_bin(1);
        let radii = screen_radial_grid(
            SCREEN_POT_RADIAL_GRID_STEP,
            SCREEN_POT_RADIAL_GRID_ORIGIN,
            POT_BIN_RADIAL_POINTS,
        )
        .expect("sample radii");
        pot.muffin_tin_radii[0] = radii[2];
        pot.norman_radii[0] = radii[4];
        let screen = ScreenInput {
            lfxc: 0,
            iend: 0,
            ..ScreenInput::default()
        };
        let potential = screen_potential_kernel_handoff(ScreenPotentialKernelHandoffInput {
            screen: &screen,
            pot: &pot,
            potential_index: 0,
        })?;
        let phase = sample_screen_phase_bin(2, 1);
        let green = sample_screen_gg_bin(2, 4);
        let fms = screen_fms_cluster_green_handoff(ScreenFmsClusterGreenHandoffInput {
            phase: &phase,
            green: &green,
            potential_index: 0,
            spin_index: 0,
            angular_count: 2,
        })?;
        let references = phase.reference_energy.column(0).to_owned();
        let active_count = potential.bounds.active_count;
        let energy_count = fms.cluster_greens.nrows();
        let angular_count = fms.cluster_greens.ncols();
        let regular =
            Array3::from_shape_fn((energy_count, active_count, angular_count), |(e, r, l)| {
                let scale = (e + 1) as f64 * (l + 1) as f64 * (r + 1) as f64;
                Complex64::new(1.0e-4 * scale, 2.0e-5 * scale)
            });
        let irregular =
            Array3::from_shape_fn((energy_count, active_count, angular_count), |(e, r, l)| {
                let scale = (e + 1) as f64 * (l + 1) as f64 * (r + 1) as f64;
                Complex64::new(7.5e-5 * scale, -1.5e-5 * scale)
            });

        Ok((potential, fms, references, regular, irregular))
    }

    fn sample_screen_pot_bin(potential_count: usize) -> PotBinData {
        PotBinData {
            titles: vec!["SCREEN potential kernel sample".to_string()],
            pad_width: POT_BIN_DEFAULT_PAD_WIDTH,
            nohole: 2,
            ihole: 1,
            interstitial_selector: 0,
            automatic_folp: 0,
            jump_mode: 0,
            unfreeze_f: 0,
            scalars: PotBinScalars {
                average_norman_radius: 1.25,
                fermi_level: -0.4,
                interstitial_potential: -1.2,
                interstitial_density: 0.03,
                edge_position: 9.1,
                amplitude_reduction: 0.85,
                relaxation_energy: 0.15,
                plasmon_frequency: 2.4,
                core_valence_energy: -3.0,
                density_radius: 1.7,
                fermi_momentum: 0.9,
                total_charge: 42.0,
                total_volume: 11.0,
            },
            muffin_tin_indices: Array1::from_elem(potential_count, 12),
            muffin_tin_radii: Array1::from_elem(potential_count, 1.1),
            norman_indices: Array1::from_elem(potential_count, 20),
            atomic_numbers: Array1::from_elem(potential_count, 29),
            kappa: Array1::from_iter(-20..=20),
            norman_radii: Array1::from_elem(potential_count, 2.1),
            overlap_factors: Array1::from_elem(potential_count, 0.9),
            max_overlap_factors: Array1::from_elem(potential_count, 1.3),
            potential_multiplicities: Array1::from_elem(potential_count, 1.0),
            ionization: Array1::zeros(potential_count),
            initial_large_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
                0.001 * (row + 1) as f64
            }),
            initial_small_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
                -0.0005 * (row + 1) as f64
            }),
            large_components: Array3::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potential_count),
                |(row, orbital, potential)| {
                    0.0001 * (row + 1) as f64 + 0.01 * orbital as f64 + 0.1 * potential as f64
                },
            ),
            small_components: Array3::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potential_count),
                |(row, orbital, potential)| {
                    -0.0001 * (row + 1) as f64 - 0.01 * orbital as f64 - 0.1 * potential as f64
                },
            ),
            large_coefficients: Array3::from_shape_fn(
                (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potential_count),
                |(coefficient, orbital, potential)| {
                    0.01 * (coefficient + 1) as f64
                        + 0.001 * orbital as f64
                        + 0.1 * potential as f64
                },
            ),
            small_coefficients: Array3::from_shape_fn(
                (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potential_count),
                |(coefficient, orbital, potential)| {
                    -0.01 * (coefficient + 1) as f64
                        - 0.001 * orbital as f64
                        - 0.1 * potential as f64
                },
            ),
            electron_density: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potential_count),
                |(row, potential)| 0.01 * (row + 1) as f64 + potential as f64 * 0.05,
            ),
            coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potential_count)),
            total_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potential_count)),
            valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potential_count)),
            valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potential_count)),
            magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potential_count)),
            orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potential_count)),
            orbital_energies: Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| {
                -10.0 + orbital as f64 * 0.25
            }),
            occupied_orbital_indices: Array2::from_shape_fn(
                (POT_BIN_IORB_SLOTS, potential_count),
                |(slot, _)| slot as i32 - 5,
            ),
            norman_charges: Array1::from_elem(potential_count, 28.5),
            valence_occupancy: Array2::zeros((4, potential_count)),
            raw_text: None,
        }
    }

    fn sample_screen_config_dat(potential_count: usize) -> ConfigDatData {
        ConfigDatData {
            header_lines: Vec::new(),
            potentials: (0..potential_count)
                .map(|potential| {
                    let mut occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
                    let valence_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
                    occupations[0] = 2.0;
                    ConfigDatPotential {
                        potential_index: potential as i32,
                        atomic_number: 29,
                        element: "Cu".to_string(),
                        occupations,
                        valence_occupations,
                        spin_occupations: None,
                    }
                })
                .collect(),
        }
    }

    fn sample_screen_phase_bin(energy_count: usize, lmax: usize) -> PhaseBinData {
        let l_count = 2 * lmax + 1;
        PhaseBinData {
            spin_count: 1,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 1,
            fermi_index: 0,
            pad_width: 8,
            final_state_count: 8,
            transition_count: 8,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 2.0,
                fermi_level: 0.5,
                edge_energy: 1.0,
            },
            energy_grid: Array1::from_iter(
                (0..energy_count).map(|energy| Complex64::new(0.1 * energy as f64, 0.02)),
            ),
            reference_energy: Array2::zeros((energy_count, 1)),
            potentials: vec![PhaseBinPotential {
                lmax,
                atomic_number: 29,
                label: "Cu".to_string(),
                phase_shifts: Array3::from_shape_fn(
                    (energy_count, l_count, 1),
                    |(energy, l_slot, _)| {
                        Complex64::new(
                            0.1 + energy as f64 * 0.01 + l_slot as f64 * 0.05,
                            -0.03 * l_slot as f64,
                        )
                    },
                ),
            }],
            transition_moments: Array4::zeros((energy_count, 1, 8, 1)),
            raw_pads: None,
        }
    }

    fn sample_screen_gg_bin(energy_count: usize, order: usize) -> GgDatData {
        GgDatData {
            sections: (0..energy_count)
                .map(|energy| GgDatSection {
                    section_number: energy + 1,
                    values: Array2::from_shape_fn((order, order), |(row, column)| {
                        let base = (energy + 1) as f64 * 10.0 + row as f64 + column as f64 * 0.25;
                        Complex64::new(base, -base * 0.1)
                    }),
                    raw_prefix_lines: None,
                })
                .collect(),
        }
    }

    const WSCRN_DAT: &str = r#" # r       w_scrn(r)      v_ch(r)
    0.1507330463E-03    0.2672882346E+02    0.2916165244E+02
    0.1584612949E-03    0.2672881678E+02    0.2916164576E+02
    0.1665857792E-03    0.2672880306E+02    0.2916163204E+02
"#;

    const VTOT_DAT: &str = r#"    0.1507330463E-03   -0.1922832821E+06    0.2672882346E+02
    0.1584612949E-03   -0.1829001336E+06    0.2672881678E+02
    0.1665857792E-03   -0.1739746063E+06    0.2672880306E+02
"#;
}
