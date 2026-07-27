//! FEFF `BAND` band-structure output support.
//!
//! `BAND/bandtot.f90` writes `bandstructure.dat` as one row per k-point with a
//! variable number of band energies. `KSPACE/kmesh.f90` can also write
//! `kmesh.dat`, where the first row carries mesh metadata and later rows only
//! carry k-point coordinates and weights.

use std::fmt::Write as _;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ShapeBuilder};
use num_complex::Complex32;
use refeff_core::{
    BandEnergiesFromPositiveCounts, BandEnergySearchMesh, BandFreePropagationBandEnergiesInput,
    BandKPathMesh, BandKkrBandEnergies, BandKkrBandEnergiesInput, BandLatticeTMatrixGridInput,
    BandLatticeTMatrixInput, BandStructureFactorFromKspace, BandStructureFactorFromKspaceGrid,
    BandStructureFactorFromKspaceNonRelInput, BandStructureFactorFromKspaceRelInput,
    BasisTransformMatrices, BravaisLattice, Complex, FEFF_BOHR_ANGSTROM, FmsAtom, KPath,
    KSPACE_Q_PAIR_TOLERANCE, KSpaceAngularTables, KSpaceDirectLatticeSetup,
    KSpaceDirectLatticeTermsInput, KSpaceEwaldEnergyTables, KSpaceEwaldEnergyTablesInput,
    KSpaceInitialEwaldTables, KSpaceQPairGroups, KSpaceReciprocalLatticeSetup,
    KSpaceReciprocalPairPhasesInput, KSpaceStrbbddInput, KSpaceStrsetNonRelFromLatticeSumInput,
    KSpaceStrsetRelFromLatticeSumInput, SpinOrbitCouplingTables, StateKet,
    band_free_propagation_band_energies, band_k_path_mesh, band_kkr_band_energies,
    band_lattice_t_matrix, band_lattice_t_matrix_grid, band_structure_factor_from_kspace_non_rel,
    basis_transform_matrices, bravais_lattice, construct_state_kets, define_k_path,
    kmesh_arbitrary_mesh, kmesh_bravais_basis, kspace_angular_tables, kspace_direct_lattice_setup,
    kspace_direct_lattice_terms, kspace_ewald_energy_tables,
    kspace_ewald_energy_tables_from_initial, kspace_q_pair_groups, kspace_reciprocal_lattice_setup,
    kspace_reciprocal_pair_phases, reciprocal_lattice_vectors, spin_orbit_coupling_tables,
};

use crate::control_input::{BandInput, ReciprocalCell};
use crate::error::{IoError, Result};
use crate::{
    PhaseBinBandSearchSetup, PhaseBinData, band_search_setup_from_handoffs,
    band_search_setup_from_handoffs_with_lmaxph,
};

const BANDSTRUCTURE_DAT_PATH: &str = "bandstructure.dat";
const BAND_INP_PATH: &str = "band.inp";
const KMESH_DAT_PATH: &str = "kmesh.dat";
const RECIPROCAL_INP_PATH: &str = "reciprocal.inp";
/// FEFF `KSPACE/m_boundaries.f90` continued-fraction order for `STRCC`.
pub const BAND_KSPACE_J22MAX: usize = 100;
const BAND_KSPACE_DEFAULT_SPIN_SELECTOR: i32 = 0;
const BAND_REL_COMPONENT_TOLERANCE_SQUARED: f64 = 1.0e-28;

/// One k-point row from FEFF `bandstructure.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct BandstructureRow {
    /// One-based k-point index written by FEFF.
    pub index: i32,
    /// Cartesian k-point coordinates.
    pub k_point: [f64; 3],
    /// Band energies at this k-point.
    pub bands: Array1<f64>,
}

/// Parsed FEFF `bandstructure.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct BandstructureDatData {
    /// Comment/header lines before the k-point rows.
    pub header_lines: Vec<String>,
    /// K-point rows in FEFF file order.
    pub rows: Vec<BandstructureRow>,
}

/// Solved BAND eigenvalue rows ready for FEFF `bandstructure.dat` assembly.
///
/// This mirrors the final output loop in `BAND/bandtot.f90`: one Cartesian
/// k-point row followed by the band count and that k-point's band energies.
/// When no custom header is supplied, FEFF-style k-grid, energy-grid, and
/// band-count summary lines are generated from the provided metadata.
#[derive(Debug, Clone, Copy)]
pub struct BandstructureDatFromEigenvaluesInput<'a> {
    /// Optional comment/header lines. If empty, FEFF-style summary headers are generated.
    pub header_lines: &'a [String],
    /// Cartesian k-point coordinates, shape `(nkp, 3)`.
    pub k_points: ArrayView2<'a, f64>,
    /// Variable-length band-energy rows, one row per k-point.
    pub band_energies: &'a [ArrayView1<'a, f64>],
    /// Number of energy grid points used by the BAND solver.
    pub energy_count: Option<usize>,
    /// Minimum BAND energy-grid value.
    pub energy_min: Option<f64>,
    /// Maximum BAND energy-grid value.
    pub energy_max: Option<f64>,
    /// BAND energy-grid step.
    pub energy_step: Option<f64>,
}

/// Typed BAND result handoff ready for `bandstructure.dat` assembly.
#[derive(Debug, Clone, Copy)]
pub struct BandstructureDatFromBandResultInput<'a> {
    /// Optional comment/header lines. If empty, FEFF-style summary headers are generated.
    pub header_lines: &'a [String],
    /// Sampled high-symmetry BAND k-path setup.
    pub k_path: &'a BandKPathHandoffSetup,
    /// FEFF `bandtot.f90` clipped search-energy mesh.
    pub energy_mesh: &'a BandEnergySearchMesh,
    /// FEFF `ne`, the total phase-energy count printed in the summary header.
    pub phase_energy_count: Option<usize>,
    /// Variable-length band-energy rows from the BAND solver, in Hartree.
    pub band_energies: &'a BandEnergiesFromPositiveCounts,
}

/// Optional mesh metadata written on the first FEFF `kmesh.dat` row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KmeshMetadata {
    /// Requested k-point count.
    pub requested_points: i32,
    /// Irreducible k-point count.
    pub irreducible_points: i32,
    /// K-mesh subdivisions.
    pub divisions: [i32; 3],
}

/// One FEFF `kmesh.dat` row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KmeshRow {
    /// One-based k-point index written by FEFF.
    pub index: i32,
    /// Irreducible Brillouin-zone k-point coordinates.
    pub k_point: [f64; 3],
    /// Integration weight.
    pub weight: f64,
    /// Mesh metadata, usually present only on the first row.
    pub metadata: Option<KmeshMetadata>,
}

/// Parsed FEFF `kmesh.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct KmeshDatData {
    /// K-point rows in FEFF file order.
    pub rows: Vec<KmeshRow>,
}

/// FEFF BAND high-symmetry K-path setup from parsed handoff inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKPathHandoffSetup {
    /// FEFF Bravais selector from `BAND/ibravais.f90`.
    pub bravais: BravaisLattice,
    /// Reciprocal basis vectors after FEFF `alat(1)/(2*pi)` scaling.
    pub reciprocal_basis: [[f64; 3]; 3],
    /// High-symmetry path segments from `BAND/kpath.f90`.
    pub path: KPath,
    /// Sampled `bk`/`KP` path mesh from `BAND/bandtot.f90`.
    pub mesh: BandKPathMesh,
}

/// FEFF KSPACE lattice setup derived from BAND source handoffs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceLatticeHandoffSetup {
    /// FEFF `ALAT`, the first lattice-vector length, in Bohr.
    pub alat_bohr: f64,
    /// FEFF `BRX/BRY/BRZ` row-vector basis, scaled by `ALAT`.
    pub direct_basis: [[f64; 3]; 3],
    /// FEFF `BGX/BGY/BGZ` reciprocal basis, inverse direct basis without `2*pi`.
    pub reciprocal_basis: [[f64; 3]; 3],
    /// FEFF Ewald parameter after `STRINIT` defaulting.
    pub eta: f64,
    /// FEFF real-space convergence radius after `STRINIT` defaulting.
    pub rmax: f64,
    /// FEFF reciprocal-space convergence radius after `STRINIT` defaulting.
    pub gmax: f64,
    /// Lower reduced-energy probe used by `STRVECGEN`.
    pub energy_min_reduced: f64,
    /// Upper reduced-energy probe used by `STRVECGEN`.
    pub energy_max_reduced: f64,
    /// FEFF `STRVECGEN` q-pair grouping.
    pub q_pairs: KSpaceQPairGroups,
    /// FEFF direct-lattice vector list and per-q-pair indirection.
    pub direct_lattice: KSpaceDirectLatticeSetup,
    /// FEFF reciprocal-lattice vector list for the BAND search-energy probe.
    pub reciprocal_lattice: KSpaceReciprocalLatticeSetup,
}

/// FEFF KSPACE angular setup derived from BAND source handoffs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceAngularHandoffSetup {
    /// Maximum scattering angular momentum, FEFF `LMAX = NL - 1`.
    pub angular_lmax: usize,
    /// Maximum harmonic-polynomial angular momentum, FEFF `LLMAX = 2 * LMAX`.
    pub harmonic_lmax: usize,
    /// FEFF `NLM = (LMAX + 1)**2`, the non-relativistic states per site.
    pub angular_state_count: usize,
    /// Total non-relativistic STRSET matrix order across all sites.
    pub matrix_order_non_rel: usize,
    /// Per-site matrix offsets, FEFF `IND0Q`, converted to zero-based offsets.
    pub site_offsets: Vec<usize>,
    /// Per-site non-relativistic state counts, FEFF `NKMQ`.
    pub site_state_counts: Vec<usize>,
    /// Total relativistic STRSET matrix order across all sites for BAND's active FEFF-basis block.
    pub matrix_order_rel: usize,
    /// Per-site relativistic matrix offsets, FEFF `IND0Q`, converted to zero-based offsets.
    pub rel_site_offsets: Vec<usize>,
    /// Per-site relativistic state counts, FEFF `NKMQ`, for BAND's active FEFF-basis block.
    pub rel_site_state_counts: Vec<usize>,
    /// FEFF `STRGAUNT`/`STRAA` tables used by `STRBBDD -> STRSET`.
    pub angular_tables: KSpaceAngularTables,
    /// FEFF `BASTRMAT` basis transformations for future relativistic solves.
    pub basis_transforms: BasisTransformMatrices,
    /// Sparse FEFF `NRREL`/`IRREL`/`SRREL` transform tables for relativistic `STRSET`.
    pub rel_components: BandKspaceRelComponentHandoffSetup,
    /// FEFF spin-orbit Clebsch-Gordan tables for future relativistic solves.
    pub spin_orbit: SpinOrbitCouplingTables,
}

/// Sparse FEFF relativistic transform tables derived from `BASTRMAT`.
///
/// Axes match `refeff_core::KSpaceStrsetRelInput`: counts use
/// `(spin, relativistic_state)`, while indices and coefficients use
/// `(term, spin, relativistic_state)`.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceRelComponentHandoffSetup {
    /// Number of non-rel components for each spin/state, FEFF `NRREL(IS,IKM)`.
    pub component_counts: Array2<usize>,
    /// Non-rel angular indices for each spin/state component, zero-based FEFF `IRREL`.
    pub component_indices: Array3<usize>,
    /// Relativistic transform coefficients, FEFF `SRREL`.
    pub component_coefficients: Array3<Complex>,
}

/// FEFF `STRCC` energy-dependent setup derived from BAND source handoffs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceEnergyHandoffSetup {
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of spin channels carried by the interpolated phase handoff.
    pub spin_count: usize,
    /// FEFF continued-fraction order used for `STRCC`, `J22MAX`.
    pub j22max: usize,
    /// Conversion denominator `(2*pi/ALAT)^2` used by `STRCC`.
    pub reduced_energy_scale: f64,
    /// Complex BAND momentum `p = sqrt(2*(E-eref))`, as `(energy, spin)`.
    pub wave_numbers: Array2<Complex>,
    /// Reduced `ERYD/(2*pi/ALAT)^2` values consumed by `STRCC`.
    pub reduced_energies: Array2<Complex>,
}

impl BandKspaceEnergyHandoffSetup {
    /// Return one reduced energy using zero-based indices.
    #[must_use]
    pub fn reduced_energy(&self, energy_index: usize, spin: usize) -> Option<Complex> {
        if energy_index >= self.energy_count || spin >= self.spin_count {
            return None;
        }
        Some(self.reduced_energies[(energy_index, spin)])
    }

    /// Return one complex momentum using zero-based indices.
    #[must_use]
    pub fn wave_number(&self, energy_index: usize, spin: usize) -> Option<Complex> {
        if energy_index >= self.energy_count || spin >= self.spin_count {
            return None;
        }
        Some(self.wave_numbers[(energy_index, spin)])
    }
}

/// FEFF BAND solver basis derived from source handoffs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceSolverBasisHandoffSetup {
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// BAND cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: Vec<FmsAtom>,
    /// FEFF BAND state kets in matrix order.
    pub states: Vec<StateKet>,
    /// First state offset for each representative potential.
    pub representative_offsets: Vec<Option<usize>>,
    /// Full FEFF BAND lattice matrix order.
    pub matrix_order: usize,
}

/// Prepared source handoffs for a streamed reciprocal-space FMS solve.
///
/// Unlike the BAND setup, this carries FEFF's full integration mesh and uses
/// one uniform `nsp * (lmax + 1)^2` state block for every unit-cell site.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsKspaceStaticHandoffSetup {
    /// Full-precision Cartesian integration points in FEFF file order.
    pub k_points: Array2<f64>,
    /// Integration weights in the same order as `k_points`.
    pub k_weights: Array1<f64>,
    /// Source-backed KSPACE lattice lists.
    pub kspace_lattice: BandKspaceLatticeHandoffSetup,
    /// Source-backed angular and STRSET tables.
    pub kspace_angular: BandKspaceAngularHandoffSetup,
    /// Uniform-site FEFF state basis.
    pub kspace_solver_basis: BandKspaceSolverBasisHandoffSetup,
    /// Energy-independent initial-`ETA` FEFF `STRAA` tables.
    pub initial_ewald_tables: KSpaceInitialEwaldTables,
}

/// Reciprocal FMS setup with shared static geometry and fresh energy terms.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsKspaceHandoffSetup {
    /// Pipeline-scoped immutable geometry and initial-`ETA` `STRAA` tables.
    pub static_setup: Arc<FmsKspaceStaticHandoffSetup>,
    /// Per-energy `STRCC` products.
    pub kspace_energy: BandKspaceEnergyHandoffSetup,
}

impl Deref for FmsKspaceHandoffSetup {
    type Target = FmsKspaceStaticHandoffSetup;

    fn deref(&self) -> &Self::Target {
        &self.static_setup
    }
}

/// Source-backed ordinary non-relativistic BAND solve from handoff inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceNonRelSolveHandoffResult {
    /// KSPACE-backed FEFF-basis structure-factor grid and point diagnostics.
    pub structure_factors: BandStructureFactorFromKspaceGrid,
    /// Per-energy FEFF lattice T-matrices used by the KKR solve.
    pub t_matrices: Array3<Complex32>,
    /// Solved KKR eigenvalue/count grid and final BAND rows, in Hartree.
    pub solved: BandKkrBandEnergies,
}

/// Source-backed non-relativistic `freeprop` BAND solve from handoff inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceFreePropagationNonRelSolveHandoffResult {
    /// KSPACE-backed FEFF-basis structure-factor grid and point diagnostics.
    pub structure_factors: BandStructureFactorFromKspaceGrid,
    /// Solved raw-G eigenvalue/count grid and final BAND rows, in Hartree.
    pub solved: BandKkrBandEnergies,
}

/// Source-backed spin-degenerate multi-spin BAND solve from handoff inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceSpinDegenerateSolveHandoffResult {
    /// Relativistic KSPACE source factors used by the KKR solve.
    pub source_structure_factors: BandStructureFactorFromKspaceGrid,
    /// FEFF-basis structure-factor grid used by the KKR solve.
    pub structure_factors: Array4<Complex32>,
    /// Per-energy FEFF lattice T-matrices used by the KKR solve.
    pub t_matrices: Array3<Complex32>,
    /// Solved KKR eigenvalue/count grid and final BAND rows, in Hartree.
    pub solved: BandKkrBandEnergies,
}

/// Source-backed spin-degenerate multi-spin `freeprop` BAND solve from handoff inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceFreePropagationSpinDegenerateSolveHandoffResult {
    /// Relativistic KSPACE source factors used by the raw-G solve.
    pub source_structure_factors: BandStructureFactorFromKspaceGrid,
    /// FEFF-basis structure-factor grid used by the raw-G solve.
    pub structure_factors: Array4<Complex32>,
    /// Solved raw-G eigenvalue/count grid and final BAND rows, in Hartree.
    pub solved: BandKkrBandEnergies,
}

/// Source-backed spin-resolved multi-spin BAND solve from handoff inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceSpinResolvedSolveHandoffResult {
    /// Relativistic KSPACE source factors used by the KKR solve.
    pub source_structure_factors: BandStructureFactorFromKspaceGrid,
    /// FEFF-basis structure-factor grid used by the KKR solve.
    pub structure_factors: Array4<Complex32>,
    /// Per-energy FEFF lattice T-matrices used by the KKR solve.
    pub t_matrices: Array3<Complex32>,
    /// Solved KKR eigenvalue/count grid and final BAND rows, in Hartree.
    pub solved: BandKkrBandEnergies,
}

/// Source-backed spin-resolved multi-spin `freeprop` BAND solve from handoff inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceFreePropagationSpinResolvedSolveHandoffResult {
    /// Relativistic KSPACE source factors used by the raw-G solve.
    pub source_structure_factors: BandStructureFactorFromKspaceGrid,
    /// FEFF-basis structure-factor grid used by the raw-G solve.
    pub structure_factors: Array4<Complex32>,
    /// Solved raw-G eigenvalue/count grid and final BAND rows, in Hartree.
    pub solved: BandKkrBandEnergies,
}

/// Source-backed ordinary relativistic BAND solve from handoff inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceRelSolveHandoffResult {
    /// Relativistic KSPACE-backed FEFF-basis structure-factor grid and point diagnostics.
    pub structure_factors: BandStructureFactorFromKspaceGrid,
    /// Per-energy FEFF lattice T-matrices used by the KKR solve.
    pub t_matrices: Array3<Complex32>,
    /// Solved KKR eigenvalue/count grid and final BAND rows, in Hartree.
    pub solved: BandKkrBandEnergies,
}

/// Source-backed relativistic `freeprop` BAND solve from handoff inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKspaceFreePropagationRelSolveHandoffResult {
    /// Relativistic KSPACE-backed FEFF-basis structure-factor grid and point diagnostics.
    pub structure_factors: BandStructureFactorFromKspaceGrid,
    /// Solved raw-G eigenvalue/count grid and final BAND rows, in Hartree.
    pub solved: BandKkrBandEnergies,
}

/// Combined BAND pre-solver setup from FEFF handoff files.
#[derive(Debug, Clone, PartialEq)]
pub struct BandPreSolverHandoffSetup {
    /// Search mesh and interpolated phase/reference tables from `phase.bin`.
    pub search: PhaseBinBandSearchSetup,
    /// High-symmetry sampled k-path from `reciprocal.inp`.
    pub k_path: BandKPathHandoffSetup,
    /// Source-backed KSPACE lattice lists from `reciprocal.inp`.
    pub kspace_lattice: BandKspaceLatticeHandoffSetup,
    /// Source-backed KSPACE angular tables and STRSET site layout.
    pub kspace_angular: BandKspaceAngularHandoffSetup,
    /// Source-backed BAND solver atoms and state kets.
    pub kspace_solver_basis: BandKspaceSolverBasisHandoffSetup,
    /// Source-backed per-energy `STRCC` products.
    pub kspace_energy: BandKspaceEnergyHandoffSetup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KmeshRows {
    Full,
    Irreducible,
}

impl BandstructureDatData {
    /// Number of k-point rows.
    #[must_use]
    pub fn k_point_count(&self) -> usize {
        self.rows.len()
    }

    /// Minimum number of bands found on any k-point row.
    #[must_use]
    pub fn min_band_count(&self) -> usize {
        self.rows
            .iter()
            .map(|row| row.bands.len())
            .min()
            .unwrap_or(0)
    }

    /// Maximum number of bands found on any k-point row.
    #[must_use]
    pub fn max_band_count(&self) -> usize {
        self.rows
            .iter()
            .map(|row| row.bands.len())
            .max()
            .unwrap_or(0)
    }
}

impl KmeshDatData {
    /// Number of k-point rows.
    #[must_use]
    pub fn k_point_count(&self) -> usize {
        self.rows.len()
    }
}

/// Parse FEFF `bandstructure.dat` text.
pub fn parse_bandstructure_dat(text: &str) -> Result<BandstructureDatData> {
    let mut header_lines = Vec::new();
    let mut rows = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            header_lines.push(raw.strip_suffix('\r').unwrap_or(raw).to_string());
            continue;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 5 {
            return bandstructure_parse_error(
                line_number,
                format!("row has {} token(s), expected at least 5", tokens.len()),
            );
        }
        let band_count = parse_usize(BANDSTRUCTURE_DAT_PATH, line_number, "band count", tokens[4])?;
        if tokens.len() != 5 + band_count {
            return bandstructure_parse_error(
                line_number,
                format!(
                    "row declares {band_count} band(s) but has {} band value token(s)",
                    tokens.len().saturating_sub(5)
                ),
            );
        }
        let bands = tokens[5..]
            .iter()
            .map(|token| parse_f64(BANDSTRUCTURE_DAT_PATH, line_number, "band energy", token))
            .collect::<Result<Vec<_>>>()?;
        rows.push(BandstructureRow {
            index: parse_i32(
                BANDSTRUCTURE_DAT_PATH,
                line_number,
                "k-point index",
                tokens[0],
            )?,
            k_point: [
                parse_f64(BANDSTRUCTURE_DAT_PATH, line_number, "kx", tokens[1])?,
                parse_f64(BANDSTRUCTURE_DAT_PATH, line_number, "ky", tokens[2])?,
                parse_f64(BANDSTRUCTURE_DAT_PATH, line_number, "kz", tokens[3])?,
            ],
            bands: Array1::from_vec(bands),
        });
    }

    let data = BandstructureDatData { header_lines, rows };
    validate_bandstructure_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `bandstructure.dat` text.
pub fn bandstructure_dat_string(data: &BandstructureDatData) -> Result<String> {
    validate_bandstructure_dat(data)?;
    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for row in &data.rows {
        write!(
            out,
            "{:5} {:8.4} {:8.4} {:8.4} {:4}",
            row.index,
            row.k_point[0],
            row.k_point[1],
            row.k_point[2],
            row.bands.len()
        )?;
        for band in &row.bands {
            write!(out, " {band:8.4}")?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Build FEFF `bandstructure.dat` rows from solved BAND eigenvalues.
///
/// The caller supplies the numerical result of the band-structure calculation:
/// Cartesian k-points and one variable-length band-energy row per k-point. This
/// adapter handles FEFF row numbering, validation, and the standard
/// `bandtot.f90` summary header.
pub fn bandstructure_dat_from_eigenvalues(
    input: BandstructureDatFromEigenvaluesInput<'_>,
) -> Result<BandstructureDatData> {
    if input.k_points.ncols() != 3 {
        return invalid_bandstructure_dat(
            "k_points",
            format!(
                "expected k-point coordinate shape (n, 3), got ({}, {})",
                input.k_points.nrows(),
                input.k_points.ncols()
            ),
        );
    }
    if input.k_points.nrows() == 0 {
        return invalid_bandstructure_dat("k_points", "at least one k-point row is required");
    }
    if input.band_energies.len() != input.k_points.nrows() {
        return invalid_bandstructure_dat(
            "band_energies",
            format!(
                "expected {} band-energy row(s), got {}",
                input.k_points.nrows(),
                input.band_energies.len()
            ),
        );
    }

    let energy_metadata = validate_bandstructure_energy_metadata(&input)?;
    let mut rows = Vec::with_capacity(input.k_points.nrows());
    for row_index in 0..input.k_points.nrows() {
        rows.push(BandstructureRow {
            index: usize_to_i32_bandstructure(row_index + 1, "k-point index")?,
            k_point: [
                input.k_points[(row_index, 0)],
                input.k_points[(row_index, 1)],
                input.k_points[(row_index, 2)],
            ],
            bands: input.band_energies[row_index].to_owned(),
        });
    }

    let header_lines = if input.header_lines.is_empty() {
        generated_bandstructure_header_lines(rows.len(), &rows, energy_metadata)
    } else {
        input.header_lines.to_vec()
    };
    let data = BandstructureDatData { header_lines, rows };
    validate_bandstructure_dat(&data)?;
    Ok(data)
}

/// Build FEFF `bandstructure.dat` from typed BAND setup and solver results.
///
/// Core BAND helpers keep energies in Hartree. FEFF `bandtot.f90` mutates the
/// eV-facing `BANDSTRUCTURE` controls into the clipped Hartree search mesh
/// before writing `bandstructure.dat`, so this adapter preserves Hartree values.
pub fn bandstructure_dat_from_band_result(
    input: BandstructureDatFromBandResultInput<'_>,
) -> Result<BandstructureDatData> {
    let band_energy_views = input
        .band_energies
        .band_energies_hartree
        .iter()
        .map(|row| row.view())
        .collect::<Vec<_>>();
    let energy_count = input
        .phase_energy_count
        .unwrap_or_else(|| input.energy_mesh.point_count());

    bandstructure_dat_from_eigenvalues(BandstructureDatFromEigenvaluesInput {
        header_lines: input.header_lines,
        k_points: input.k_path.mesh.k_points.view(),
        band_energies: &band_energy_views,
        energy_count: Some(energy_count),
        energy_min: Some(input.energy_mesh.min_hartree),
        energy_max: Some(input.energy_mesh.max_hartree),
        energy_step: Some(input.energy_mesh.step_hartree),
    })
}

/// Build the deterministic BAND setup available before the KKR numerical solve.
///
/// This combines the `bandtot.f90` search/phase setup from `phase.bin` with
/// the high-symmetry k-path setup from `reciprocal.inp`. The remaining solver
/// driver consumes this combined handoff before constructing KSPACE point
/// inputs.
pub fn band_pre_solver_setup_from_handoffs(
    band: &BandInput,
    phase: &PhaseBinData,
    cell: &ReciprocalCell,
) -> Result<BandPreSolverHandoffSetup> {
    let search = band_search_setup_from_handoffs(band, phase)?;
    band_pre_solver_setup_from_search_and_lmaxph(
        band,
        search,
        cell,
        &phase_bin_potential_lmax(phase),
    )
}

/// Build deterministic BAND setup using FEFF `fms.inp` `lmaxph(0:nph)`.
///
/// FEFF `band` calls `reafms` before `bandtot`, so KSPACE `kprep` derives
/// `maxl`/`msize` from `fms.inp`'s active `lmaxph` cutoffs. The raw
/// `phase.bin` write range can be larger; this source handoff mirrors FEFF by
/// using `lmaxph` for phase interpolation, KSPACE angular dimensions, and the
/// `fmsband` solver basis.
pub fn band_pre_solver_setup_from_handoffs_with_lmaxph(
    band: &BandInput,
    phase: &PhaseBinData,
    cell: &ReciprocalCell,
    lmaxph: &[usize],
) -> Result<BandPreSolverHandoffSetup> {
    let search = band_search_setup_from_handoffs_with_lmaxph(band, phase, lmaxph)?;
    band_pre_solver_setup_from_search_and_lmaxph(band, search, cell, lmaxph)
}

fn band_pre_solver_setup_from_search_and_lmaxph(
    band: &BandInput,
    search: PhaseBinBandSearchSetup,
    cell: &ReciprocalCell,
    lmaxph: &[usize],
) -> Result<BandPreSolverHandoffSetup> {
    let active_lmax = validate_band_lmaxph("lmaxph", lmaxph, &search)?;
    let k_path = band_k_path_setup_from_handoffs(band, cell)?;
    let kspace_lattice = band_kspace_lattice_setup_from_handoffs(&search.energy_mesh, cell)?;
    let kspace_angular =
        band_kspace_angular_setup_from_lmaxph(&search, &kspace_lattice, cell, &active_lmax)?;
    let kspace_solver_basis =
        band_kspace_solver_basis_setup_from_lmaxph(&search, cell, &active_lmax)?;
    let kspace_energy = band_kspace_energy_setup_from_handoffs(&search, &kspace_lattice)?;
    Ok(BandPreSolverHandoffSetup {
        search,
        k_path,
        kspace_lattice,
        kspace_angular,
        kspace_solver_basis,
        kspace_energy,
    })
}

fn phase_bin_potential_lmax(phase: &PhaseBinData) -> Vec<usize> {
    phase
        .potentials
        .iter()
        .map(|potential| potential.lmax)
        .collect()
}

/// Read FEFF `bandstructure.dat` text from a file.
pub fn read_bandstructure_dat(path: impl AsRef<Path>) -> Result<BandstructureDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_bandstructure_dat(&text)
}

/// Write FEFF `bandstructure.dat` text to a file.
pub fn write_bandstructure_dat(path: impl AsRef<Path>, data: &BandstructureDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, bandstructure_dat_string(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `kmesh.dat` text.
pub fn parse_kmesh_dat(text: &str) -> Result<KmeshDatData> {
    let mut rows = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 5 && tokens.len() != 10 {
            return kmesh_parse_error(
                line_number,
                format!("row has {} token(s), expected 5 or 10", tokens.len()),
            );
        }
        let metadata = if tokens.len() == 10 {
            Some(KmeshMetadata {
                requested_points: parse_i32(
                    KMESH_DAT_PATH,
                    line_number,
                    "requested k-points",
                    tokens[5],
                )?,
                irreducible_points: parse_i32(
                    KMESH_DAT_PATH,
                    line_number,
                    "irreducible k-points",
                    tokens[6],
                )?,
                divisions: [
                    parse_i32(KMESH_DAT_PATH, line_number, "k-division x", tokens[7])?,
                    parse_i32(KMESH_DAT_PATH, line_number, "k-division y", tokens[8])?,
                    parse_i32(KMESH_DAT_PATH, line_number, "k-division z", tokens[9])?,
                ],
            })
        } else {
            None
        };
        rows.push(KmeshRow {
            index: parse_i32(KMESH_DAT_PATH, line_number, "k-point index", tokens[0])?,
            k_point: [
                parse_f64(KMESH_DAT_PATH, line_number, "kx", tokens[1])?,
                parse_f64(KMESH_DAT_PATH, line_number, "ky", tokens[2])?,
                parse_f64(KMESH_DAT_PATH, line_number, "kz", tokens[3])?,
            ],
            weight: parse_f64(KMESH_DAT_PATH, line_number, "weight", tokens[4])?,
            metadata,
        });
    }
    let data = KmeshDatData { rows };
    validate_kmesh_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `kmesh.dat` text.
pub fn kmesh_dat_string(data: &KmeshDatData) -> Result<String> {
    validate_kmesh_dat(data)?;
    let mut out = String::new();
    for row in &data.rows {
        write!(
            out,
            "{:10}{:9.4}{:9.4}{:9.4}{:9.4}",
            row.index, row.k_point[0], row.k_point[1], row.k_point[2], row.weight
        )?;
        if let Some(metadata) = row.metadata {
            write!(
                out,
                "{:7}{:7}{:7}{:7}{:7}",
                metadata.requested_points,
                metadata.irreducible_points,
                metadata.divisions[0],
                metadata.divisions[1],
                metadata.divisions[2]
            )?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Build FEFF `kmesh.dat` rows from a parsed reciprocal-cell handoff.
///
/// This is the shared source-backed handoff used before the remaining BAND and
/// reciprocal-space LDOS solvers. It follows FEFF `KSPACE/kmesh.f90` for
/// Bravais-basis setup and arbitrary-mesh division. It writes the full
/// no-symmetry mesh because FEFF's crystallographic symmetry matrices are not
/// carried by `reciprocal.inp`; callers that already have those operations can
/// use [`kmesh_dat_from_reciprocal_cell_with_operations`].
pub fn kmesh_dat_from_reciprocal_cell(cell: &ReciprocalCell) -> Result<KmeshDatData> {
    let operations = identity_kmesh_operations();
    kmesh_dat_from_reciprocal_cell_with_operation_rows(cell, operations.view(), KmeshRows::Full)
}

/// Build FEFF `kmesh.dat` rows with explicit FEFF-compatible symmetry operations.
///
/// The operation array is passed directly to the Rust `KSPACE` arbitrary-mesh
/// reducer and must use FEFF's integer operation convention. The resulting rows
/// are the irreducible mesh.
pub fn kmesh_dat_from_reciprocal_cell_with_operations(
    cell: &ReciprocalCell,
    operations: ArrayView3<'_, i32>,
) -> Result<KmeshDatData> {
    kmesh_dat_from_reciprocal_cell_with_operation_rows(cell, operations, KmeshRows::Irreducible)
}

/// Build FEFF BAND high-symmetry K-path setup from parsed handoff files.
///
/// FEFF `bandtot.f90` obtains the Bravais class from `sgroup` and the lattice
/// centering, scales reciprocal basis vectors by `alat(1)/(2*pi)`, calls
/// `define_kpath`, and then samples the requested `nkp` points along those
/// segments. This adapter keeps that deterministic setup available to the Rust
/// BAND driver before the numerical KKR solve.
pub fn band_k_path_setup_from_handoffs(
    band: &BandInput,
    cell: &ReciprocalCell,
) -> Result<BandKPathHandoffSetup> {
    let requested_point_count = usize::try_from(band.nkp).map_err(|_| {
        invalid_band_input_error("nkp", format!("BAND nkp {} does not fit usize", band.nkp))
    })?;
    let lattice = lattice_centering(&cell.lattice_name)?;
    let bravais = bravais_lattice(cell.space_group, lattice)
        .map_err(|source| invalid_reciprocal_error("bravais", source.to_string()))?;
    let reciprocal_basis = band_scaled_reciprocal_basis(cell)?;
    let path = define_k_path(bravais, band.ikpath, reciprocal_basis)
        .map_err(|source| invalid_reciprocal_error("kpath", source.to_string()))?;
    let mesh = band_k_path_mesh(&path, requested_point_count)
        .map_err(|source| invalid_band_input_error("nkp", source.to_string()))?;

    Ok(BandKPathHandoffSetup {
        bravais,
        reciprocal_basis,
        path,
        mesh,
    })
}

/// Build FEFF `STRVECGEN` lattice setup from BAND handoff files.
///
/// This is the source-backed structure-factor setup available before the
/// remaining BAND KKR solver driver. It mirrors `KSPACE/kprep.f90` and
/// `STRINIT`: lattice vectors are scaled by `ALAT`, positions are consumed in
/// the dimensionless Cartesian units stored in `reciprocal.inp`, and the
/// reciprocal search probe is reduced by `(2*pi/ALAT)^2`.
pub fn band_kspace_lattice_setup_from_handoffs(
    search: &BandEnergySearchMesh,
    cell: &ReciprocalCell,
) -> Result<BandKspaceLatticeHandoffSetup> {
    let alat_angstrom = reciprocal_cell_first_lattice_length(cell)?;
    let alat_bohr = alat_angstrom / FEFF_BOHR_ANGSTROM;
    validate_positive_finite_reciprocal("alat", alat_bohr)?;
    let direct_basis = band_kspace_direct_basis(cell, alat_angstrom)?;
    let reciprocal_basis = inverse_direct_basis_without_pi2(direct_basis)?;
    let q_positions = band_kspace_q_positions(cell)?;
    let q_pairs = kspace_q_pair_groups(q_positions.view(), KSPACE_Q_PAIR_TOLERANCE)
        .map_err(|source| invalid_reciprocal_error("kspace_q_pairs", source.to_string()))?;
    let (eta, rmax, gmax) = band_kspace_strinit_limits(search, cell, direct_basis)?;
    let (energy_min_reduced, energy_max_reduced) =
        band_kspace_reduced_energy_probe(search, alat_bohr)?;

    let direct_lattice = kspace_direct_lattice_setup(
        direct_basis,
        q_pairs.offsets.view(),
        rmax,
        q_pairs.max_offset_norm,
    )
    .map_err(|source| invalid_reciprocal_error("kspace_direct_lattice", source.to_string()))?;
    let reciprocal_lattice = kspace_reciprocal_lattice_setup(
        reciprocal_basis,
        gmax,
        energy_min_reduced,
        energy_max_reduced,
    )
    .map_err(|source| invalid_reciprocal_error("kspace_reciprocal_lattice", source.to_string()))?;

    Ok(BandKspaceLatticeHandoffSetup {
        alat_bohr,
        direct_basis,
        reciprocal_basis,
        eta,
        rmax,
        gmax,
        energy_min_reduced,
        energy_max_reduced,
        q_pairs,
        direct_lattice,
        reciprocal_lattice,
    })
}

/// Build FEFF `STRGAUNT`/`STRAA` angular setup from BAND handoff files.
///
/// The angular cutoff comes from the normalized `phase.bin` BAND view, while
/// FEFF's Gaunt scaling uses the `ALAT` value already derived for KSPACE. Site
/// offsets are generated in the same order as `reciprocal.inp` positions.
pub fn band_kspace_angular_setup_from_handoffs(
    search: &PhaseBinBandSearchSetup,
    lattice: &BandKspaceLatticeHandoffSetup,
    cell: &ReciprocalCell,
) -> Result<BandKspaceAngularHandoffSetup> {
    band_kspace_angular_setup_from_lmaxph(
        search,
        lattice,
        cell,
        &search.phase_handoff.potential_lmax,
    )
}

/// Build FEFF `STRGAUNT`/`STRAA` angular setup using active `lmaxph` cutoffs.
pub fn band_kspace_angular_setup_from_lmaxph(
    search: &PhaseBinBandSearchSetup,
    lattice: &BandKspaceLatticeHandoffSetup,
    cell: &ReciprocalCell,
    lmaxph: &[usize],
) -> Result<BandKspaceAngularHandoffSetup> {
    let active_lmax = validate_band_lmaxph("lmaxph", lmaxph, search)?;
    let angular_lmax =
        active_lmax.iter().copied().max().ok_or_else(|| {
            invalid_band_input_error("lmaxph", "BAND requires at least one lmaxph")
        })?;
    let angular_tables = kspace_angular_tables(angular_lmax, lattice.alat_bohr)
        .map_err(|source| invalid_reciprocal_error("kspace_angular_tables", source.to_string()))?;
    let basis_transforms = basis_transform_matrices(angular_lmax)
        .map_err(|source| invalid_band_input_error("basis_transforms", source.to_string()))?;
    let rel_components = band_kspace_rel_component_setup_from_basis_transforms(
        &basis_transforms,
        angular_tables.angular_state_count,
    )?;
    let spin_orbit = spin_orbit_coupling_tables(angular_lmax)
        .map_err(|source| invalid_band_input_error("spin_orbit", source.to_string()))?;
    let spin_count = search
        .phase_interpolation
        .reference_energies_hartree
        .ncols();
    if spin_count == 0 {
        return Err(invalid_band_input_error(
            "kspace_angular",
            "BAND KSPACE angular setup requires at least one spin channel",
        ));
    }
    let rel_state_count = angular_tables
        .angular_state_count
        .checked_mul(spin_count)
        .ok_or_else(|| {
            invalid_band_input_error("kspace_angular", "relativistic matrix order overflowed")
        })?;
    let (site_offsets, site_state_counts, matrix_order_non_rel) =
        band_kspace_non_rel_site_layout(cell, angular_tables.angular_state_count)?;
    let (rel_site_offsets, rel_site_state_counts, matrix_order_rel) =
        band_kspace_rel_site_layout(cell, rel_state_count, &rel_components)?;

    Ok(BandKspaceAngularHandoffSetup {
        angular_lmax,
        harmonic_lmax: angular_tables.harmonic_lmax,
        angular_state_count: angular_tables.angular_state_count,
        matrix_order_non_rel,
        site_offsets,
        site_state_counts,
        matrix_order_rel,
        rel_site_offsets,
        rel_site_state_counts,
        angular_tables,
        basis_transforms,
        rel_components,
        spin_orbit,
    })
}

fn validate_band_lmaxph(
    field: &'static str,
    lmaxph: &[usize],
    search: &PhaseBinBandSearchSetup,
) -> Result<Vec<usize>> {
    let potential_count = search.phase_handoff.potential_lmax.len();
    if lmaxph.len() != potential_count {
        return Err(invalid_band_input_error(
            field,
            format!(
                "BAND {field} has {} value(s), expected {potential_count}",
                lmaxph.len()
            ),
        ));
    }
    if potential_count == 0 {
        return Err(invalid_band_input_error(
            field,
            "BAND requires at least one potential cutoff",
        ));
    }
    for (potential, (&active, &available)) in lmaxph
        .iter()
        .zip(search.phase_handoff.potential_lmax.iter())
        .enumerate()
    {
        if active > available {
            return Err(invalid_band_input_error(
                field,
                format!("{field}({potential})={active} exceeds phase.bin lmax {available}"),
            ));
        }
        if active > search.phase_handoff.signed_angular_offset {
            return Err(invalid_band_input_error(
                field,
                format!(
                    "{field}({potential})={active} exceeds signed phase axis {}",
                    search.phase_handoff.signed_angular_offset
                ),
            ));
        }
    }
    Ok(lmaxph.to_vec())
}

/// Build FEFF `NRREL`/`IRREL`/`SRREL` tables from `BASTRMAT` handoff data.
pub fn band_kspace_rel_component_setup_from_basis_transforms(
    transforms: &BasisTransformMatrices,
    angular_state_count: usize,
) -> Result<BandKspaceRelComponentHandoffSetup> {
    let expected_order = angular_state_count.checked_mul(2).ok_or_else(|| {
        invalid_band_input_error("rel_components", "relativistic matrix order overflowed")
    })?;
    if transforms.order != expected_order {
        return Err(invalid_band_input_error(
            "rel_components",
            format!(
                "basis transform order {} does not match 2 * angular state count {}",
                transforms.order, angular_state_count
            ),
        ));
    }
    if transforms.real_to_relativistic.dim() != (transforms.order, transforms.order) {
        return Err(invalid_band_input_error(
            "rel_components",
            format!(
                "real-to-relativistic matrix shape {:?} does not match order {}",
                transforms.real_to_relativistic.dim(),
                transforms.order
            ),
        ));
    }

    let mut component_counts = Array2::<usize>::zeros((2, transforms.order));
    let mut component_indices = Array3::<usize>::zeros((angular_state_count, 2, transforms.order));
    let mut component_coefficients =
        Array3::<Complex>::zeros((angular_state_count, 2, transforms.order));

    for state in 0..transforms.order {
        let mut state_component_count = 0_usize;
        for spin in 0..2 {
            let mut spin_component_count = 0_usize;
            let spin_row_offset = spin * angular_state_count;
            for angular_index in 0..angular_state_count {
                let coefficient =
                    transforms.real_to_relativistic[(spin_row_offset + angular_index, state)];
                if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                    return Err(invalid_band_input_error(
                        "rel_components",
                        format!(
                            "real-to-relativistic coefficient ({}, {state}) is not finite",
                            spin_row_offset + angular_index
                        ),
                    ));
                }
                if coefficient.norm_sqr() <= BAND_REL_COMPONENT_TOLERANCE_SQUARED {
                    continue;
                }
                component_indices[(spin_component_count, spin, state)] = angular_index;
                component_coefficients[(spin_component_count, spin, state)] = coefficient;
                spin_component_count += 1;
            }
            component_counts[(spin, state)] = spin_component_count;
            state_component_count += spin_component_count;
        }
        if state_component_count == 0 {
            return Err(invalid_band_input_error(
                "rel_components",
                format!("relativistic state {state} has no real-basis components"),
            ));
        }
    }

    Ok(BandKspaceRelComponentHandoffSetup {
        component_counts,
        component_indices,
        component_coefficients,
    })
}

/// Build per-energy FEFF `STRCC` products from BAND source handoffs.
///
/// BAND passes `em = E - eref` into `fmsband`, which then calls
/// `STRCC` with `ERYD = 2*em`. This builder preserves that convention and
/// reduces `ERYD` by `(2*pi/ALAT)^2` for the Rust KSPACE helpers.
pub fn band_kspace_energy_setup_from_handoffs(
    search: &PhaseBinBandSearchSetup,
    lattice: &BandKspaceLatticeHandoffSetup,
) -> Result<BandKspaceEnergyHandoffSetup> {
    let (energy_count, spin_count) = search.phase_interpolation.reference_energies_hartree.dim();
    if energy_count == 0 || spin_count == 0 {
        return Err(invalid_band_input_error(
            "kspace_energy",
            "BAND KSPACE energy setup requires at least one energy and spin channel",
        ));
    }
    if search.energy_mesh.energies_hartree.len() != energy_count {
        return Err(invalid_band_input_error(
            "kspace_energy",
            format!(
                "search energy count {} does not match reference table energy count {energy_count}",
                search.energy_mesh.energies_hartree.len()
            ),
        ));
    }

    let reciprocal_unit = std::f64::consts::TAU / lattice.alat_bohr;
    let reduced_energy_scale = reciprocal_unit * reciprocal_unit;
    validate_positive_finite_reciprocal("reduced_energy_scale", reduced_energy_scale)?;

    let mut wave_numbers = Array2::<Complex>::zeros((energy_count, spin_count));
    let mut reduced_energies = Array2::<Complex>::zeros((energy_count, spin_count));

    for energy_index in 0..energy_count {
        let search_energy = search.energy_mesh.energies_hartree[energy_index];
        validate_finite_band("search_energy_hartree", search_energy)?;
        for spin in 0..spin_count {
            let reference =
                search.phase_interpolation.reference_energies_hartree[(energy_index, spin)];
            let eryd = Complex::new(2.0, 0.0) * (Complex::new(search_energy, 0.0) - reference);
            validate_complex_finite_reciprocal("eryd", eryd)?;
            let wave_number = eryd.sqrt();
            validate_complex_finite_reciprocal("wave_number", wave_number)?;
            let reduced_energy = eryd / reduced_energy_scale;
            validate_complex_finite_reciprocal("reduced_energy", reduced_energy)?;
            wave_numbers[(energy_index, spin)] = wave_number;
            reduced_energies[(energy_index, spin)] = reduced_energy;
        }
    }

    Ok(BandKspaceEnergyHandoffSetup {
        energy_count,
        spin_count,
        j22max: BAND_KSPACE_J22MAX,
        reduced_energy_scale,
        wave_numbers,
        reduced_energies,
    })
}

/// Build FEFF BAND solver atoms and state kets from source handoffs.
///
/// This is the source-backed `fmsband.f90` atom/state setup needed before
/// per-energy lattice T-matrix assembly. The default wrapper uses the raw
/// `phase.bin` potential lmax values; callers with `fms.inp` should use
/// `band_kspace_solver_basis_setup_from_lmaxph` so FEFF's active KSPACE
/// `lpot`/`maxl` cutoffs drive `msize`. The current BAND handoff does not yet
/// carry global `ispin`, so the selector follows FEFF's default value `0`.
pub fn band_kspace_solver_basis_setup_from_handoffs(
    search: &PhaseBinBandSearchSetup,
    cell: &ReciprocalCell,
) -> Result<BandKspaceSolverBasisHandoffSetup> {
    band_kspace_solver_basis_setup_from_lmaxph(search, cell, &search.phase_handoff.potential_lmax)
}

/// Build FEFF BAND solver atoms and state kets using active `lmaxph` cutoffs.
pub fn band_kspace_solver_basis_setup_from_lmaxph(
    search: &PhaseBinBandSearchSetup,
    cell: &ReciprocalCell,
    lmaxph: &[usize],
) -> Result<BandKspaceSolverBasisHandoffSetup> {
    let (energy_count, spin_channels) = search.phase_interpolation.reference_energies_hartree.dim();
    if energy_count == 0 || spin_channels == 0 {
        return Err(invalid_band_input_error(
            "kspace_solver_basis",
            "BAND solver basis requires at least one energy and spin channel",
        ));
    }

    let atoms = band_kspace_solver_atoms(cell)?;
    let atom_potentials = cell
        .potentials
        .iter()
        .copied()
        .map(|potential| {
            usize::try_from(potential).map_err(|_| {
                invalid_reciprocal_error(
                    "potentials",
                    format!("potential index must be non-negative, got {potential}"),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let active_lmax = validate_band_lmaxph("lmaxph", lmaxph, search)?;
    let angular_lmax =
        active_lmax.iter().copied().max().ok_or_else(|| {
            invalid_band_input_error("lmaxph", "BAND requires at least one lmaxph")
        })?;
    let state_kets =
        construct_state_kets(spin_channels, &atom_potentials, &active_lmax, angular_lmax).map_err(
            |source| invalid_band_input_error("kspace_solver_basis", source.to_string()),
        )?;
    let matrix_order = state_kets.states.len();

    Ok(BandKspaceSolverBasisHandoffSetup {
        spin_channels,
        spin_selector: BAND_KSPACE_DEFAULT_SPIN_SELECTOR,
        atoms,
        states: state_kets.states,
        representative_offsets: state_kets.representative_offsets,
        matrix_order,
    })
}

/// Prepare reusable energy-independent KSPACE handoffs for reciprocal FMS.
///
/// `energy_probe_hartree` controls only FEFF `STRVECGEN` lattice bounds.
/// POT passes its native fixed `[-3,+3]` Hartree probe, while ordinary FMS
/// passes the active phase-energy range.  The numerical k mesh is regenerated
/// at full precision from `reciprocal.inp`; rounded `kmesh.dat` rows are never
/// used as solver input.
pub fn fms_kspace_static_setup_from_handoffs(
    cell: &ReciprocalCell,
    energy_probe_hartree: ArrayView1<'_, f64>,
    global_lmax: usize,
    spin_channels: usize,
    spin_selector: i32,
) -> Result<FmsKspaceStaticHandoffSetup> {
    if cell.k_mesh.use_symmetry {
        return Err(invalid_reciprocal_error(
            "use_symmetry",
            "reciprocal FMS symmetry reduction is unsupported without FEFF rotation tables",
        ));
    }
    let lattice_name = cell.lattice_name.trim().to_ascii_uppercase();
    if !matches!(lattice_name.as_str(), "P" | "H") {
        return Err(invalid_reciprocal_error(
            "lattice_name",
            format!(
                "reciprocal FMS currently supports only primitive P/H cells, got {:?}",
                cell.lattice_name
            ),
        ));
    }
    if !cell.volume_scale.is_finite() || cell.volume_scale > 0.0 {
        return Err(invalid_reciprocal_error(
            "volume_scale",
            "reciprocal-cell volume scaling must be finite and non-positive",
        ));
    }
    if cell.imaginary_energy != 0.0 {
        return Err(invalid_reciprocal_error(
            "imaginary_energy",
            format!(
                "nonzero reciprocal STRCC imaginary-energy override {} is unsupported",
                cell.imaginary_energy
            ),
        ));
    }
    if spin_channels != 1 {
        return Err(invalid_reciprocal_error(
            "spin_channels",
            format!(
                "reciprocal FMS currently supports exactly one spin channel, got {spin_channels}"
            ),
        ));
    }
    if spin_selector != 0 {
        return Err(invalid_reciprocal_error(
            "spin_selector",
            format!("reciprocal FMS spin selector {spin_selector} is unsupported"),
        ));
    }
    if energy_probe_hartree.is_empty() {
        return Err(invalid_reciprocal_error(
            "energy_probe",
            "reciprocal FMS requires a nonempty STRVECGEN energy probe",
        ));
    }

    let probe = BandEnergySearchMesh {
        min_hartree: energy_probe_hartree
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        max_hartree: energy_probe_hartree
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        step_hartree: 0.0,
        energies_hartree: energy_probe_hartree.to_owned(),
    };
    let kspace_lattice = band_kspace_lattice_setup_from_handoffs(&probe, cell)?;
    let kspace_angular =
        fms_kspace_angular_setup(global_lmax, spin_channels, &kspace_lattice, cell)?;
    let kspace_solver_basis =
        fms_kspace_uniform_solver_basis(cell, global_lmax, spin_channels, spin_selector)?;
    if kspace_solver_basis.matrix_order != kspace_angular.matrix_order_non_rel * spin_channels {
        return Err(invalid_reciprocal_error(
            "matrix_order",
            format!(
                "uniform FMS state order {} does not match angular order {} times {} spin channel(s)",
                kspace_solver_basis.matrix_order,
                kspace_angular.matrix_order_non_rel,
                spin_channels
            ),
        ));
    }
    let mesh = kmesh_dat_from_reciprocal_cell(cell)?;
    let mut k_points = Array2::<f64>::zeros((mesh.rows.len(), 3));
    let mut k_weights = Array1::<f64>::zeros(mesh.rows.len());
    for (point, row) in mesh.rows.iter().enumerate() {
        for axis in 0..3 {
            k_points[(point, axis)] = row.k_point[axis];
        }
        k_weights[point] = row.weight;
    }

    let initial_ewald_tables = KSpaceInitialEwaldTables {
        eta: kspace_lattice.eta,
        reciprocal_pair_phases: kspace_reciprocal_pair_phases(KSpaceReciprocalPairPhasesInput {
            direct_basis: kspace_lattice.direct_basis,
            reciprocal_basis: kspace_lattice.reciprocal_basis,
            reciprocal_indices: kspace_lattice.reciprocal_lattice.reciprocal_indices.view(),
            q_pair_offsets: kspace_lattice.q_pairs.offsets.view(),
            eta: kspace_lattice.eta,
        })
        .map_err(|source| {
            invalid_reciprocal_error("kspace_initial_reciprocal_phases", source.to_string())
        })?,
        direct_lattice_terms: kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
            direct_basis: kspace_lattice.direct_basis,
            direct_indices: kspace_lattice.direct_lattice.direct_indices.view(),
            direct_index_by_pair: kspace_lattice.direct_lattice.direct_index_by_pair.view(),
            direct_counts: &kspace_lattice.direct_lattice.direct_counts,
            q_pair_offsets: kspace_lattice.q_pairs.offsets.view(),
            lmax: kspace_angular.harmonic_lmax,
            j22max: BAND_KSPACE_J22MAX,
            qjltab: kspace_angular.angular_tables.qjltab.view(),
            eta: kspace_lattice.eta,
        })
        .map_err(|source| {
            invalid_reciprocal_error("kspace_initial_direct_terms", source.to_string())
        })?,
    };

    Ok(FmsKspaceStaticHandoffSetup {
        k_points,
        k_weights,
        kspace_lattice,
        kspace_angular,
        kspace_solver_basis,
        initial_ewald_tables,
    })
}

/// Attach fresh per-energy products to reusable reciprocal FMS geometry.
pub fn fms_kspace_setup_from_static_handoffs(
    static_setup: Arc<FmsKspaceStaticHandoffSetup>,
    energies_hartree: ArrayView1<'_, Complex>,
    reference_energies_hartree: ArrayView2<'_, Complex>,
) -> Result<FmsKspaceHandoffSetup> {
    let kspace_energy = fms_kspace_energy_setup(
        energies_hartree,
        reference_energies_hartree,
        &static_setup.kspace_lattice,
    )?;
    if kspace_energy.spin_count != static_setup.kspace_solver_basis.spin_channels {
        return Err(invalid_reciprocal_error(
            "spin_channels",
            format!(
                "reference-energy table has {} spin column(s), expected {}",
                kspace_energy.spin_count, static_setup.kspace_solver_basis.spin_channels
            ),
        ));
    }

    Ok(FmsKspaceHandoffSetup {
        static_setup,
        kspace_energy,
    })
}

/// Prepare the complete KSPACE handoff used by one reciprocal FMS solve.
pub fn fms_kspace_setup_from_handoffs(
    cell: &ReciprocalCell,
    energies_hartree: ArrayView1<'_, Complex>,
    reference_energies_hartree: ArrayView2<'_, Complex>,
    energy_probe_hartree: ArrayView1<'_, f64>,
    global_lmax: usize,
    spin_channels: usize,
    spin_selector: i32,
) -> Result<FmsKspaceHandoffSetup> {
    let static_setup = Arc::new(fms_kspace_static_setup_from_handoffs(
        cell,
        energy_probe_hartree,
        global_lmax,
        spin_channels,
        spin_selector,
    )?);
    fms_kspace_setup_from_static_handoffs(
        static_setup,
        energies_hartree,
        reference_energies_hartree,
    )
}

fn fms_kspace_angular_setup(
    angular_lmax: usize,
    spin_count: usize,
    lattice: &BandKspaceLatticeHandoffSetup,
    cell: &ReciprocalCell,
) -> Result<BandKspaceAngularHandoffSetup> {
    let angular_tables = kspace_angular_tables(angular_lmax, lattice.alat_bohr)
        .map_err(|source| invalid_reciprocal_error("kspace_angular_tables", source.to_string()))?;
    let basis_transforms = basis_transform_matrices(angular_lmax)
        .map_err(|source| invalid_reciprocal_error("basis_transforms", source.to_string()))?;
    let rel_components = band_kspace_rel_component_setup_from_basis_transforms(
        &basis_transforms,
        angular_tables.angular_state_count,
    )?;
    let spin_orbit = spin_orbit_coupling_tables(angular_lmax)
        .map_err(|source| invalid_reciprocal_error("spin_orbit", source.to_string()))?;
    let rel_state_count = angular_tables
        .angular_state_count
        .checked_mul(spin_count)
        .ok_or_else(|| {
            invalid_reciprocal_error("kspace_angular", "relativistic matrix order overflowed")
        })?;
    let (site_offsets, site_state_counts, matrix_order_non_rel) =
        band_kspace_non_rel_site_layout(cell, angular_tables.angular_state_count)?;
    let (rel_site_offsets, rel_site_state_counts, matrix_order_rel) =
        band_kspace_rel_site_layout(cell, rel_state_count, &rel_components)?;

    Ok(BandKspaceAngularHandoffSetup {
        angular_lmax,
        harmonic_lmax: angular_tables.harmonic_lmax,
        angular_state_count: angular_tables.angular_state_count,
        matrix_order_non_rel,
        site_offsets,
        site_state_counts,
        matrix_order_rel,
        rel_site_offsets,
        rel_site_state_counts,
        angular_tables,
        basis_transforms,
        rel_components,
        spin_orbit,
    })
}

fn fms_kspace_uniform_solver_basis(
    cell: &ReciprocalCell,
    global_lmax: usize,
    spin_channels: usize,
    spin_selector: i32,
) -> Result<BandKspaceSolverBasisHandoffSetup> {
    let atoms = band_kspace_solver_atoms(cell)?;
    let atom_potentials = cell
        .potentials
        .iter()
        .copied()
        .map(|potential| {
            usize::try_from(potential).map_err(|_| {
                invalid_reciprocal_error(
                    "potentials",
                    format!("potential index must be non-negative, got {potential}"),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let potential_count = atom_potentials
        .iter()
        .copied()
        .max()
        .and_then(|maximum| maximum.checked_add(1))
        .ok_or_else(|| invalid_reciprocal_error("potentials", "unit cell has no potentials"))?;
    let uniform_lmax = vec![global_lmax; potential_count];
    let state_kets =
        construct_state_kets(spin_channels, &atom_potentials, &uniform_lmax, global_lmax).map_err(
            |source| invalid_reciprocal_error("kspace_solver_basis", source.to_string()),
        )?;
    let matrix_order = state_kets.states.len();
    Ok(BandKspaceSolverBasisHandoffSetup {
        spin_channels,
        spin_selector,
        atoms,
        states: state_kets.states,
        representative_offsets: state_kets.representative_offsets,
        matrix_order,
    })
}

fn fms_kspace_energy_setup(
    energies_hartree: ArrayView1<'_, Complex>,
    reference_energies_hartree: ArrayView2<'_, Complex>,
    lattice: &BandKspaceLatticeHandoffSetup,
) -> Result<BandKspaceEnergyHandoffSetup> {
    let energy_count = energies_hartree.len();
    let (reference_energy_count, spin_count) = reference_energies_hartree.dim();
    if energy_count == 0 || spin_count == 0 {
        return Err(invalid_reciprocal_error(
            "kspace_energy",
            "reciprocal FMS requires at least one energy and spin channel",
        ));
    }
    if reference_energy_count != energy_count {
        return Err(invalid_reciprocal_error(
            "kspace_energy",
            format!(
                "energy count {energy_count} does not match reference table energy count {reference_energy_count}"
            ),
        ));
    }

    let reciprocal_unit = std::f64::consts::TAU / lattice.alat_bohr;
    let reduced_energy_scale = reciprocal_unit * reciprocal_unit;
    validate_positive_finite_reciprocal("reduced_energy_scale", reduced_energy_scale)?;
    let mut wave_numbers = Array2::<Complex>::zeros((energy_count, spin_count));
    let mut reduced_energies = Array2::<Complex>::zeros((energy_count, spin_count));
    for energy_index in 0..energy_count {
        validate_complex_finite_reciprocal(
            "reciprocal_fms_energy",
            energies_hartree[energy_index],
        )?;
        for spin in 0..spin_count {
            let reference = reference_energies_hartree[(energy_index, spin)];
            validate_complex_finite_reciprocal("reciprocal_fms_reference", reference)?;
            let eryd = Complex::new(2.0, 0.0) * (energies_hartree[energy_index] - reference);
            validate_complex_finite_reciprocal("eryd", eryd)?;
            let wave_number = eryd.sqrt();
            validate_complex_finite_reciprocal("wave_number", wave_number)?;
            let reduced_energy = eryd / reduced_energy_scale;
            validate_complex_finite_reciprocal("reduced_energy", reduced_energy)?;
            wave_numbers[(energy_index, spin)] = wave_number;
            reduced_energies[(energy_index, spin)] = reduced_energy;
        }
    }
    Ok(BandKspaceEnergyHandoffSetup {
        energy_count,
        spin_count,
        j22max: BAND_KSPACE_J22MAX,
        reduced_energy_scale,
        wave_numbers,
        reduced_energies,
    })
}

/// Build one reciprocal FMS lattice T matrix.
pub fn fms_kspace_t_matrix(
    setup: &FmsKspaceHandoffSetup,
    phase_shifts: ArrayView3<'_, Complex32>,
) -> Result<Array2<Complex32>> {
    band_lattice_t_matrix(BandLatticeTMatrixInput {
        states: &setup.kspace_solver_basis.states,
        atoms: &setup.kspace_solver_basis.atoms,
        spin_channels: setup.kspace_solver_basis.spin_channels,
        spin_selector: setup.kspace_solver_basis.spin_selector,
        phase_shifts,
        spin_orbit: &setup.kspace_angular.spin_orbit,
    })
    .map_err(|source| invalid_reciprocal_error("kspace_t_matrix", source.to_string()))
}

/// Build FEFF `fmsband.f90` lattice T-matrices across all BAND search energies.
pub fn band_kspace_t_matrix_grid_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<Array3<Complex32>> {
    band_lattice_t_matrix_grid(BandLatticeTMatrixGridInput {
        states: &setup.kspace_solver_basis.states,
        atoms: &setup.kspace_solver_basis.atoms,
        spin_channels: setup.kspace_solver_basis.spin_channels,
        spin_selector: setup.kspace_solver_basis.spin_selector,
        phase_shifts: setup.search.phase_interpolation.phase_shifts.view(),
        spin_orbit: &setup.kspace_angular.spin_orbit,
    })
    .map_err(|source| invalid_band_input_error("kspace_t_matrix", source.to_string()))
}

fn band_kspace_non_rel_structure_factor_grid_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandStructureFactorFromKspaceGrid> {
    band_kspace_non_rel_structure_factor_grid_from_handoffs_for_spin(setup, 0)
}

fn band_kspace_non_rel_structure_factor_grid_from_handoffs_for_spin(
    setup: &BandPreSolverHandoffSetup,
    spin: usize,
) -> Result<BandStructureFactorFromKspaceGrid> {
    let energy_count = setup.kspace_energy.energy_count;
    let k_point_count = setup.k_path.mesh.point_count();
    let matrix_order = setup.kspace_angular.matrix_order_non_rel;
    if spin >= setup.kspace_energy.spin_count {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "structure-factor spin index {spin} is outside {} spin column(s)",
                setup.kspace_energy.spin_count
            ),
        ));
    }
    let expected_points = energy_count.checked_mul(k_point_count).ok_or_else(|| {
        invalid_band_input_error(
            "kspace_grid",
            "energy/k-point grid size overflowed while building BAND solve",
        )
    })?;

    let mut point_solves = Vec::with_capacity(expected_points);
    let mut structure_factors =
        Array4::<Complex32>::zeros((energy_count, k_point_count, matrix_order, matrix_order).f());
    for energy_index in 0..energy_count {
        let tables = band_kspace_ewald_energy_tables_from_handoff(setup, energy_index, spin)?;
        for k_point_index in 0..k_point_count {
            let point_input = band_kspace_non_rel_structure_factor_input(
                setup,
                &tables,
                energy_index,
                spin,
                k_point_index,
            )?;
            let point_solve =
                band_structure_factor_from_kspace_non_rel(point_input).map_err(|source| {
                    invalid_band_input_error("kspace_structure_factor", source.to_string())
                })?;
            validate_band_kspace_structure_factor_order(&point_solve, matrix_order)?;
            for row in 0..matrix_order {
                for column in 0..matrix_order {
                    structure_factors[(energy_index, k_point_index, row, column)] =
                        point_solve.structure_factor[(row, column)];
                }
            }
            point_solves.push(point_solve);
        }
    }

    Ok(BandStructureFactorFromKspaceGrid {
        point_solves,
        structure_factors,
    })
}

fn band_kspace_rel_structure_factor_grid_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandStructureFactorFromKspaceGrid> {
    band_kspace_rel_structure_factor_grid_from_handoffs_for_spin(setup, 0)
}

fn band_kspace_rel_structure_factor_grid_from_handoffs_for_spin(
    setup: &BandPreSolverHandoffSetup,
    spin: usize,
) -> Result<BandStructureFactorFromKspaceGrid> {
    let energy_count = setup.kspace_energy.energy_count;
    let k_point_count = setup.k_path.mesh.point_count();
    let matrix_order = setup.kspace_angular.matrix_order_rel;
    let expected_points = energy_count.checked_mul(k_point_count).ok_or_else(|| {
        invalid_band_input_error(
            "kspace_rel_grid",
            "energy/k-point grid size overflowed while building BAND rel solve",
        )
    })?;

    let mut point_solves = Vec::with_capacity(expected_points);
    let mut structure_factors =
        Array4::<Complex32>::zeros((energy_count, k_point_count, matrix_order, matrix_order).f());
    for energy_index in 0..energy_count {
        let tables = band_kspace_ewald_energy_tables_from_handoff(setup, energy_index, spin)?;
        for k_point_index in 0..k_point_count {
            let point_input = band_kspace_rel_structure_factor_input(
                setup,
                &tables,
                energy_index,
                spin,
                k_point_index,
            )?;
            let point_solve = refeff_core::band_structure_factor_from_kspace_rel(point_input)
                .map_err(|source| {
                    invalid_band_input_error("kspace_rel_structure_factor", source.to_string())
                })?;
            validate_band_kspace_structure_factor_order(&point_solve, matrix_order)?;
            for row in 0..matrix_order {
                for column in 0..matrix_order {
                    structure_factors[(energy_index, k_point_index, row, column)] =
                        point_solve.structure_factor[(row, column)];
                }
            }
            point_solves.push(point_solve);
        }
    }

    Ok(BandStructureFactorFromKspaceGrid {
        point_solves,
        structure_factors,
    })
}

/// Solve ordinary non-relativistic BAND rows from parsed source handoffs.
///
/// This composes the current source-backed BAND boundary: build one-energy
/// `STRCC` tables on demand, assemble every k-path structure factor in
/// FEFF loop order, combine the result with per-energy `fmsband` T-matrices,
/// then run the Rust KKR eigenvalue/count and band-row identification helpers.
pub fn band_kspace_non_rel_solve_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandKspaceNonRelSolveHandoffResult> {
    validate_band_kspace_non_rel_solve_setup(setup)?;

    let structure_factors = band_kspace_non_rel_structure_factor_grid_from_handoffs(setup)?;
    let t_matrices = band_kspace_t_matrix_grid_from_handoffs(setup)?;
    let solved = band_kkr_band_energies(BandKkrBandEnergiesInput {
        structure_factors: structure_factors.structure_factors.view(),
        t_matrices: t_matrices.view(),
        wave_numbers: setup.search.phase_interpolation.wave_numbers.column(0),
        energy_min_hartree: setup.search.energy_mesh.min_hartree,
        energy_step_hartree: setup.search.energy_mesh.step_hartree,
    })
    .map_err(|source| invalid_band_input_error("kspace_kkr", source.to_string()))?;

    Ok(BandKspaceNonRelSolveHandoffResult {
        structure_factors,
        t_matrices,
        solved,
    })
}

/// Solve non-relativistic `freeprop` BAND rows from parsed source handoffs.
///
/// This shares the ordinary KSPACE source assembly but follows FEFF
/// `kkrband.f90`'s `freeprop` branch by diagonalizing raw `G * p` without
/// subtracting `Tmat^-1`.
pub fn band_kspace_free_propagation_non_rel_solve_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandKspaceFreePropagationNonRelSolveHandoffResult> {
    validate_band_kspace_non_rel_solve_setup(setup)?;

    let structure_factors = band_kspace_non_rel_structure_factor_grid_from_handoffs(setup)?;
    let solved = band_free_propagation_band_energies(BandFreePropagationBandEnergiesInput {
        structure_factors: structure_factors.structure_factors.view(),
        wave_numbers: setup.search.phase_interpolation.wave_numbers.column(0),
        energy_min_hartree: setup.search.energy_mesh.min_hartree,
        energy_step_hartree: setup.search.energy_mesh.step_hartree,
    })
    .map_err(|source| invalid_band_input_error("kspace_freeprop", source.to_string()))?;

    Ok(BandKspaceFreePropagationNonRelSolveHandoffResult {
        structure_factors,
        solved,
    })
}

/// Solve ordinary relativistic BAND rows from parsed source handoffs.
///
/// This is a guarded source-backed `IREL >= 2` companion to the ordinary
/// non-relativistic path. It prepares relativistic KSPACE `STRSET` point
/// inputs from the same source handoffs, then reuses the existing FEFF-basis
/// KKR eigenvalue/count and band-row identification helpers.
pub fn band_kspace_rel_solve_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandKspaceRelSolveHandoffResult> {
    validate_band_kspace_rel_solve_setup(setup)?;

    let structure_factors = band_kspace_rel_structure_factor_grid_from_handoffs(setup)?;
    let t_matrices = band_kspace_t_matrix_grid_from_handoffs(setup)?;
    let solved = band_kkr_band_energies(BandKkrBandEnergiesInput {
        structure_factors: structure_factors.structure_factors.view(),
        t_matrices: t_matrices.view(),
        wave_numbers: setup.search.phase_interpolation.wave_numbers.column(0),
        energy_min_hartree: setup.search.energy_mesh.min_hartree,
        energy_step_hartree: setup.search.energy_mesh.step_hartree,
    })
    .map_err(|source| invalid_band_input_error("kspace_rel_kkr", source.to_string()))?;

    Ok(BandKspaceRelSolveHandoffResult {
        structure_factors,
        t_matrices,
        solved,
    })
}

/// Solve relativistic `freeprop` BAND rows from parsed source handoffs.
pub fn band_kspace_free_propagation_rel_solve_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandKspaceFreePropagationRelSolveHandoffResult> {
    validate_band_kspace_rel_solve_setup(setup)?;

    let structure_factors = band_kspace_rel_structure_factor_grid_from_handoffs(setup)?;
    let solved = band_free_propagation_band_energies(BandFreePropagationBandEnergiesInput {
        structure_factors: structure_factors.structure_factors.view(),
        wave_numbers: setup.search.phase_interpolation.wave_numbers.column(0),
        energy_min_hartree: setup.search.energy_mesh.min_hartree,
        energy_step_hartree: setup.search.energy_mesh.step_hartree,
    })
    .map_err(|source| invalid_band_input_error("kspace_rel_freeprop", source.to_string()))?;

    Ok(BandKspaceFreePropagationRelSolveHandoffResult {
        structure_factors,
        solved,
    })
}

/// Solve spin-degenerate multi-spin BAND rows from parsed source handoffs.
///
/// KSPACE structure constants are spin-independent when all spin reference
/// energies are degenerate. This path builds the non-relativistic source-backed
/// `G(energy,kpoint)` grid once, expands it over FEFF's spin-resolved state
/// order, and reuses the existing multi-spin T-matrix/eigenvalue solver.
pub fn band_kspace_spin_degenerate_solve_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandKspaceSpinDegenerateSolveHandoffResult> {
    validate_band_kspace_spin_degenerate_solve_setup(setup)?;

    let source_structure_factors =
        band_kspace_rel_structure_factor_grid_from_handoffs_for_spin(setup, 0)?;
    let structure_factors = source_structure_factors.structure_factors.clone();
    let t_matrices = band_kspace_t_matrix_grid_from_handoffs(setup)?;
    let solved = band_kkr_band_energies(BandKkrBandEnergiesInput {
        structure_factors: structure_factors.view(),
        t_matrices: t_matrices.view(),
        wave_numbers: setup.search.phase_interpolation.wave_numbers.column(0),
        energy_min_hartree: setup.search.energy_mesh.min_hartree,
        energy_step_hartree: setup.search.energy_mesh.step_hartree,
    })
    .map_err(|source| invalid_band_input_error("kspace_spin_degenerate_kkr", source.to_string()))?;

    Ok(BandKspaceSpinDegenerateSolveHandoffResult {
        source_structure_factors,
        structure_factors,
        t_matrices,
        solved,
    })
}

/// Solve spin-degenerate multi-spin `freeprop` BAND rows from source handoffs.
pub fn band_kspace_free_propagation_spin_degenerate_solve_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandKspaceFreePropagationSpinDegenerateSolveHandoffResult> {
    validate_band_kspace_spin_degenerate_solve_setup(setup)?;

    let source_structure_factors =
        band_kspace_rel_structure_factor_grid_from_handoffs_for_spin(setup, 0)?;
    let structure_factors = source_structure_factors.structure_factors.clone();
    let solved = band_free_propagation_band_energies(BandFreePropagationBandEnergiesInput {
        structure_factors: structure_factors.view(),
        wave_numbers: setup.search.phase_interpolation.wave_numbers.column(0),
        energy_min_hartree: setup.search.energy_mesh.min_hartree,
        energy_step_hartree: setup.search.energy_mesh.step_hartree,
    })
    .map_err(|source| {
        invalid_band_input_error("kspace_spin_degenerate_freeprop", source.to_string())
    })?;

    Ok(BandKspaceFreePropagationSpinDegenerateSolveHandoffResult {
        source_structure_factors,
        structure_factors,
        solved,
    })
}

/// Solve spin-resolved multi-spin BAND rows from parsed source handoffs.
///
/// FEFF `bandtot.f90` interpolates all spin phase columns, but the scalar
/// energy passed to `fmsband` is the last spin's `ene - refene`. This branch
/// mirrors that behavior by assembling the KSPACE `G(energy,kpoint)` grid from
/// the last spin column before expanding it over FEFF's spin-resolved state
/// order.
pub fn band_kspace_spin_resolved_solve_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandKspaceSpinResolvedSolveHandoffResult> {
    let structure_spin = validate_band_kspace_spin_resolved_solve_setup(setup)?;

    let source_structure_factors =
        band_kspace_rel_structure_factor_grid_from_handoffs_for_spin(setup, structure_spin)?;
    let structure_factors = source_structure_factors.structure_factors.clone();
    let t_matrices = band_kspace_t_matrix_grid_from_handoffs(setup)?;
    let solved = band_kkr_band_energies(BandKkrBandEnergiesInput {
        structure_factors: structure_factors.view(),
        t_matrices: t_matrices.view(),
        wave_numbers: setup
            .search
            .phase_interpolation
            .wave_numbers
            .column(structure_spin),
        energy_min_hartree: setup.search.energy_mesh.min_hartree,
        energy_step_hartree: setup.search.energy_mesh.step_hartree,
    })
    .map_err(|source| invalid_band_input_error("kspace_spin_resolved_kkr", source.to_string()))?;

    Ok(BandKspaceSpinResolvedSolveHandoffResult {
        source_structure_factors,
        structure_factors,
        t_matrices,
        solved,
    })
}

/// Solve spin-resolved multi-spin `freeprop` BAND rows from parsed source handoffs.
pub fn band_kspace_free_propagation_spin_resolved_solve_from_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandKspaceFreePropagationSpinResolvedSolveHandoffResult> {
    let structure_spin = validate_band_kspace_spin_resolved_solve_setup(setup)?;

    let source_structure_factors =
        band_kspace_rel_structure_factor_grid_from_handoffs_for_spin(setup, structure_spin)?;
    let structure_factors = source_structure_factors.structure_factors.clone();
    let solved = band_free_propagation_band_energies(BandFreePropagationBandEnergiesInput {
        structure_factors: structure_factors.view(),
        wave_numbers: setup
            .search
            .phase_interpolation
            .wave_numbers
            .column(structure_spin),
        energy_min_hartree: setup.search.energy_mesh.min_hartree,
        energy_step_hartree: setup.search.energy_mesh.step_hartree,
    })
    .map_err(|source| {
        invalid_band_input_error("kspace_spin_resolved_freeprop", source.to_string())
    })?;

    Ok(BandKspaceFreePropagationSpinResolvedSolveHandoffResult {
        source_structure_factors,
        structure_factors,
        solved,
    })
}

/// Build FEFF `bandstructure.dat` rows from ordinary non-rel source handoffs.
pub fn bandstructure_dat_from_kspace_non_rel_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandstructureDatData> {
    let solved = band_kspace_non_rel_solve_from_handoffs(setup)?;
    bandstructure_dat_from_band_result(BandstructureDatFromBandResultInput {
        header_lines: &[],
        k_path: &setup.k_path,
        energy_mesh: &setup.search.energy_mesh,
        phase_energy_count: Some(setup.search.phase_handoff.energies_hartree.len()),
        band_energies: &solved.solved.band_energies,
    })
}

/// Build FEFF `bandstructure.dat` rows from non-rel `freeprop` source handoffs.
pub fn bandstructure_dat_from_kspace_free_propagation_non_rel_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandstructureDatData> {
    let solved = band_kspace_free_propagation_non_rel_solve_from_handoffs(setup)?;
    bandstructure_dat_from_band_result(BandstructureDatFromBandResultInput {
        header_lines: &[],
        k_path: &setup.k_path,
        energy_mesh: &setup.search.energy_mesh,
        phase_energy_count: Some(setup.search.phase_handoff.energies_hartree.len()),
        band_energies: &solved.solved.band_energies,
    })
}

/// Build FEFF `bandstructure.dat` rows from ordinary relativistic source handoffs.
pub fn bandstructure_dat_from_kspace_rel_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandstructureDatData> {
    let solved = band_kspace_rel_solve_from_handoffs(setup)?;
    bandstructure_dat_from_band_result(BandstructureDatFromBandResultInput {
        header_lines: &[],
        k_path: &setup.k_path,
        energy_mesh: &setup.search.energy_mesh,
        phase_energy_count: Some(setup.search.phase_handoff.energies_hartree.len()),
        band_energies: &solved.solved.band_energies,
    })
}

/// Build FEFF `bandstructure.dat` rows from relativistic `freeprop` source handoffs.
pub fn bandstructure_dat_from_kspace_free_propagation_rel_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandstructureDatData> {
    let solved = band_kspace_free_propagation_rel_solve_from_handoffs(setup)?;
    bandstructure_dat_from_band_result(BandstructureDatFromBandResultInput {
        header_lines: &[],
        k_path: &setup.k_path,
        energy_mesh: &setup.search.energy_mesh,
        phase_energy_count: Some(setup.search.phase_handoff.energies_hartree.len()),
        band_energies: &solved.solved.band_energies,
    })
}

/// Build FEFF `bandstructure.dat` rows from spin-degenerate multi-spin handoffs.
pub fn bandstructure_dat_from_kspace_spin_degenerate_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandstructureDatData> {
    let solved = band_kspace_spin_degenerate_solve_from_handoffs(setup)?;
    bandstructure_dat_from_band_result(BandstructureDatFromBandResultInput {
        header_lines: &[],
        k_path: &setup.k_path,
        energy_mesh: &setup.search.energy_mesh,
        phase_energy_count: Some(setup.search.phase_handoff.energies_hartree.len()),
        band_energies: &solved.solved.band_energies,
    })
}

/// Build FEFF `bandstructure.dat` rows from spin-degenerate multi-spin `freeprop`.
pub fn bandstructure_dat_from_kspace_free_propagation_spin_degenerate_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandstructureDatData> {
    let solved = band_kspace_free_propagation_spin_degenerate_solve_from_handoffs(setup)?;
    bandstructure_dat_from_band_result(BandstructureDatFromBandResultInput {
        header_lines: &[],
        k_path: &setup.k_path,
        energy_mesh: &setup.search.energy_mesh,
        phase_energy_count: Some(setup.search.phase_handoff.energies_hartree.len()),
        band_energies: &solved.solved.band_energies,
    })
}

/// Build FEFF `bandstructure.dat` rows from spin-resolved multi-spin handoffs.
pub fn bandstructure_dat_from_kspace_spin_resolved_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandstructureDatData> {
    let solved = band_kspace_spin_resolved_solve_from_handoffs(setup)?;
    bandstructure_dat_from_band_result(BandstructureDatFromBandResultInput {
        header_lines: &[],
        k_path: &setup.k_path,
        energy_mesh: &setup.search.energy_mesh,
        phase_energy_count: Some(setup.search.phase_handoff.energies_hartree.len()),
        band_energies: &solved.solved.band_energies,
    })
}

/// Build FEFF `bandstructure.dat` rows from spin-resolved multi-spin `freeprop`.
pub fn bandstructure_dat_from_kspace_free_propagation_spin_resolved_handoffs(
    setup: &BandPreSolverHandoffSetup,
) -> Result<BandstructureDatData> {
    let solved = band_kspace_free_propagation_spin_resolved_solve_from_handoffs(setup)?;
    bandstructure_dat_from_band_result(BandstructureDatFromBandResultInput {
        header_lines: &[],
        k_path: &setup.k_path,
        energy_mesh: &setup.search.energy_mesh,
        phase_energy_count: Some(setup.search.phase_handoff.energies_hartree.len()),
        band_energies: &solved.solved.band_energies,
    })
}

/// Build FEFF `STRCC` Ewald tables for one BAND search energy and spin.
pub fn band_kspace_ewald_energy_tables_from_handoff(
    setup: &BandPreSolverHandoffSetup,
    energy_index: usize,
    spin: usize,
) -> Result<KSpaceEwaldEnergyTables> {
    let reduced_energy = setup
        .kspace_energy
        .reduced_energy(energy_index, spin)
        .ok_or_else(|| {
            invalid_band_input_error(
                "kspace_energy",
                format!(
                    "energy/spin index ({energy_index}, {spin}) is outside ({}, {})",
                    setup.kspace_energy.energy_count, setup.kspace_energy.spin_count
                ),
            )
        })?;

    kspace_ewald_energy_tables(KSpaceEwaldEnergyTablesInput {
        energy: reduced_energy,
        initial_eta: setup.kspace_lattice.eta,
        lmax: setup.kspace_angular.harmonic_lmax,
        j22max: setup.kspace_energy.j22max,
        direct_basis: setup.kspace_lattice.direct_basis,
        reciprocal_basis: setup.kspace_lattice.reciprocal_basis,
        reciprocal_indices: setup
            .kspace_lattice
            .reciprocal_lattice
            .reciprocal_indices
            .view(),
        direct_indices: setup.kspace_lattice.direct_lattice.direct_indices.view(),
        direct_index_by_pair: setup
            .kspace_lattice
            .direct_lattice
            .direct_index_by_pair
            .view(),
        direct_counts: &setup.kspace_lattice.direct_lattice.direct_counts,
        q_pair_offsets: setup.kspace_lattice.q_pairs.offsets.view(),
        qjltab: setup.kspace_angular.angular_tables.qjltab.view(),
    })
    .map_err(|source| invalid_band_input_error("kspace_strcc", source.to_string()))
}

/// Build FEFF `STRCC` Ewald tables for one reciprocal FMS energy and spin.
pub fn fms_kspace_ewald_energy_tables_from_handoff(
    setup: &FmsKspaceHandoffSetup,
    energy_index: usize,
    spin: usize,
) -> Result<KSpaceEwaldEnergyTables> {
    let reduced_energy = setup
        .kspace_energy
        .reduced_energy(energy_index, spin)
        .ok_or_else(|| {
            invalid_reciprocal_error(
                "kspace_energy",
                format!(
                    "energy/spin index ({energy_index}, {spin}) is outside ({}, {})",
                    setup.kspace_energy.energy_count, setup.kspace_energy.spin_count
                ),
            )
        })?;
    kspace_ewald_energy_tables_from_initial(
        KSpaceEwaldEnergyTablesInput {
            energy: reduced_energy,
            initial_eta: setup.kspace_lattice.eta,
            lmax: setup.kspace_angular.harmonic_lmax,
            j22max: setup.kspace_energy.j22max,
            direct_basis: setup.kspace_lattice.direct_basis,
            reciprocal_basis: setup.kspace_lattice.reciprocal_basis,
            reciprocal_indices: setup
                .kspace_lattice
                .reciprocal_lattice
                .reciprocal_indices
                .view(),
            direct_indices: setup.kspace_lattice.direct_lattice.direct_indices.view(),
            direct_index_by_pair: setup
                .kspace_lattice
                .direct_lattice
                .direct_index_by_pair
                .view(),
            direct_counts: &setup.kspace_lattice.direct_lattice.direct_counts,
            q_pair_offsets: setup.kspace_lattice.q_pairs.offsets.view(),
            qjltab: setup.kspace_angular.angular_tables.qjltab.view(),
        },
        &setup.initial_ewald_tables,
    )
    .map_err(|source| invalid_reciprocal_error("kspace_strcc", source.to_string()))
}

/// Assemble one full-precision reciprocal FMS structure factor.
pub fn fms_kspace_non_rel_structure_factor(
    setup: &FmsKspaceHandoffSetup,
    tables: &KSpaceEwaldEnergyTables,
    energy_index: usize,
    spin: usize,
    k_point_index: usize,
) -> Result<BandStructureFactorFromKspace> {
    let reduced_energy = setup
        .kspace_energy
        .reduced_energy(energy_index, spin)
        .ok_or_else(|| {
            invalid_reciprocal_error(
                "kspace_point",
                format!(
                    "energy/spin index ({energy_index}, {spin}) is outside ({}, {})",
                    setup.kspace_energy.energy_count, setup.kspace_energy.spin_count
                ),
            )
        })?;
    let wave_number = setup
        .kspace_energy
        .wave_number(energy_index, spin)
        .ok_or_else(|| {
            invalid_reciprocal_error(
                "kspace_point",
                format!(
                    "energy/spin index ({energy_index}, {spin}) is outside ({}, {})",
                    setup.kspace_energy.energy_count, setup.kspace_energy.spin_count
                ),
            )
        })?;
    if k_point_index >= setup.k_points.nrows() {
        return Err(invalid_reciprocal_error(
            "kspace_point",
            format!(
                "k-point index {k_point_index} is outside {} integration points",
                setup.k_points.nrows()
            ),
        ));
    }
    let k = [
        setup.k_points[(k_point_index, 0)],
        setup.k_points[(k_point_index, 1)],
        setup.k_points[(k_point_index, 2)],
    ];
    band_structure_factor_from_kspace_non_rel(BandStructureFactorFromKspaceNonRelInput {
        kspace: KSpaceStrsetNonRelFromLatticeSumInput {
            lattice_sum: KSpaceStrbbddInput {
                k,
                lmax: setup.kspace_angular.harmonic_lmax,
                eta: tables.eta,
                energy: reduced_energy,
                gmax_squared: setup.kspace_lattice.reciprocal_lattice.gmax_squared,
                reciprocal_basis: setup.kspace_lattice.reciprocal_basis,
                reciprocal_indices: setup
                    .kspace_lattice
                    .reciprocal_lattice
                    .reciprocal_indices
                    .view(),
                reciprocal_pair_phases: tables.reciprocal_pair_phases.reciprocal_pair_phases.view(),
                d1term3: tables.energy_dependent_terms.d1term3.view(),
                qjltab: setup.kspace_angular.angular_tables.qjltab.view(),
                q_pair_offsets: setup.kspace_lattice.q_pairs.offsets.view(),
                direct_basis: setup.kspace_lattice.direct_basis,
                direct_indices: setup.kspace_lattice.direct_lattice.direct_indices.view(),
                direct_index_by_pair: setup
                    .kspace_lattice
                    .direct_lattice
                    .direct_index_by_pair
                    .view(),
                direct_counts: &setup.kspace_lattice.direct_lattice.direct_counts,
                direct_terms: tables.energy_dependent_terms.direct_terms.view(),
                d300: tables.energy_dependent_terms.d300,
            },
            angular_state_count: setup.kspace_angular.angular_state_count,
            q_pair_sites: setup.kspace_lattice.q_pairs.sites.view(),
            q_pair_counts: &setup.kspace_lattice.q_pairs.counts,
            site_offsets: &setup.kspace_angular.site_offsets,
            site_state_counts: &setup.kspace_angular.site_state_counts,
            gaunt_counts: &setup.kspace_angular.angular_tables.gaunt_counts,
            gaunt_indices: &setup.kspace_angular.angular_tables.gaunt_indices,
            gaunt_values: &setup.kspace_angular.angular_tables.gaunt_values,
            cipwl: setup.kspace_angular.angular_tables.cipwl.view(),
            wave_number,
        },
        atom_count: setup.kspace_angular.site_state_counts.len(),
        angular_lmax: setup.kspace_angular.angular_lmax,
        basis_transforms: &setup.kspace_angular.basis_transforms,
    })
    .map_err(|source| invalid_reciprocal_error("kspace_structure_factor", source.to_string()))
}

/// Build one borrowed non-relativistic KSPACE structure-factor input.
///
/// The returned value borrows the owned pre-solver handoff and is ready for
/// `refeff_core::band_structure_factor_from_kspace_non_rel`.
pub fn band_kspace_non_rel_structure_factor_input<'a>(
    setup: &'a BandPreSolverHandoffSetup,
    tables: &'a KSpaceEwaldEnergyTables,
    energy_index: usize,
    spin: usize,
    k_point_index: usize,
) -> Result<BandStructureFactorFromKspaceNonRelInput<'a>> {
    let (reduced_energy, wave_number, k) =
        band_kspace_point_terms(setup, energy_index, spin, k_point_index)?;

    Ok(BandStructureFactorFromKspaceNonRelInput {
        kspace: KSpaceStrsetNonRelFromLatticeSumInput {
            lattice_sum: KSpaceStrbbddInput {
                k,
                lmax: setup.kspace_angular.harmonic_lmax,
                eta: tables.eta,
                energy: reduced_energy,
                gmax_squared: setup.kspace_lattice.reciprocal_lattice.gmax_squared,
                reciprocal_basis: setup.kspace_lattice.reciprocal_basis,
                reciprocal_indices: setup
                    .kspace_lattice
                    .reciprocal_lattice
                    .reciprocal_indices
                    .view(),
                reciprocal_pair_phases: tables.reciprocal_pair_phases.reciprocal_pair_phases.view(),
                d1term3: tables.energy_dependent_terms.d1term3.view(),
                qjltab: setup.kspace_angular.angular_tables.qjltab.view(),
                q_pair_offsets: setup.kspace_lattice.q_pairs.offsets.view(),
                direct_basis: setup.kspace_lattice.direct_basis,
                direct_indices: setup.kspace_lattice.direct_lattice.direct_indices.view(),
                direct_index_by_pair: setup
                    .kspace_lattice
                    .direct_lattice
                    .direct_index_by_pair
                    .view(),
                direct_counts: &setup.kspace_lattice.direct_lattice.direct_counts,
                direct_terms: tables.energy_dependent_terms.direct_terms.view(),
                d300: tables.energy_dependent_terms.d300,
            },
            angular_state_count: setup.kspace_angular.angular_state_count,
            q_pair_sites: setup.kspace_lattice.q_pairs.sites.view(),
            q_pair_counts: &setup.kspace_lattice.q_pairs.counts,
            site_offsets: &setup.kspace_angular.site_offsets,
            site_state_counts: &setup.kspace_angular.site_state_counts,
            gaunt_counts: &setup.kspace_angular.angular_tables.gaunt_counts,
            gaunt_indices: &setup.kspace_angular.angular_tables.gaunt_indices,
            gaunt_values: &setup.kspace_angular.angular_tables.gaunt_values,
            cipwl: setup.kspace_angular.angular_tables.cipwl.view(),
            wave_number,
        },
        atom_count: setup.kspace_angular.site_state_counts.len(),
        angular_lmax: setup.kspace_angular.angular_lmax,
        basis_transforms: &setup.kspace_angular.basis_transforms,
    })
}

/// Build one borrowed relativistic KSPACE structure-factor input.
///
/// This prepares the `IREL >= 2` `STRBBDD -> STRSET` branch using the sparse
/// `NRREL`/`IRREL`/`SRREL` tables already derived in the BAND angular handoff.
/// It does not select or run the production BAND relativistic solve branch.
pub fn band_kspace_rel_structure_factor_input<'a>(
    setup: &'a BandPreSolverHandoffSetup,
    tables: &'a KSpaceEwaldEnergyTables,
    energy_index: usize,
    spin: usize,
    k_point_index: usize,
) -> Result<BandStructureFactorFromKspaceRelInput<'a>> {
    let (reduced_energy, wave_number, k) =
        band_kspace_point_terms(setup, energy_index, spin, k_point_index)?;

    Ok(BandStructureFactorFromKspaceRelInput {
        kspace: KSpaceStrsetRelFromLatticeSumInput {
            lattice_sum: KSpaceStrbbddInput {
                k,
                lmax: setup.kspace_angular.harmonic_lmax,
                eta: tables.eta,
                energy: reduced_energy,
                gmax_squared: setup.kspace_lattice.reciprocal_lattice.gmax_squared,
                reciprocal_basis: setup.kspace_lattice.reciprocal_basis,
                reciprocal_indices: setup
                    .kspace_lattice
                    .reciprocal_lattice
                    .reciprocal_indices
                    .view(),
                reciprocal_pair_phases: tables.reciprocal_pair_phases.reciprocal_pair_phases.view(),
                d1term3: tables.energy_dependent_terms.d1term3.view(),
                qjltab: setup.kspace_angular.angular_tables.qjltab.view(),
                q_pair_offsets: setup.kspace_lattice.q_pairs.offsets.view(),
                direct_basis: setup.kspace_lattice.direct_basis,
                direct_indices: setup.kspace_lattice.direct_lattice.direct_indices.view(),
                direct_index_by_pair: setup
                    .kspace_lattice
                    .direct_lattice
                    .direct_index_by_pair
                    .view(),
                direct_counts: &setup.kspace_lattice.direct_lattice.direct_counts,
                direct_terms: tables.energy_dependent_terms.direct_terms.view(),
                d300: tables.energy_dependent_terms.d300,
            },
            angular_state_count: setup.kspace_angular.angular_state_count,
            q_pair_sites: setup.kspace_lattice.q_pairs.sites.view(),
            q_pair_counts: &setup.kspace_lattice.q_pairs.counts,
            site_offsets: &setup.kspace_angular.rel_site_offsets,
            site_state_counts: &setup.kspace_angular.rel_site_state_counts,
            gaunt_counts: &setup.kspace_angular.angular_tables.gaunt_counts,
            gaunt_indices: &setup.kspace_angular.angular_tables.gaunt_indices,
            gaunt_values: &setup.kspace_angular.angular_tables.gaunt_values,
            cipwl: setup.kspace_angular.angular_tables.cipwl.view(),
            rel_component_counts: setup.kspace_angular.rel_components.component_counts.view(),
            rel_component_indices: setup.kspace_angular.rel_components.component_indices.view(),
            rel_component_coefficients: setup
                .kspace_angular
                .rel_components
                .component_coefficients
                .view(),
            wave_number,
        },
        atom_count: setup.kspace_angular.rel_site_state_counts.len(),
        angular_lmax: setup.kspace_angular.angular_lmax,
        basis_transforms: &setup.kspace_angular.basis_transforms,
    })
}

fn band_kspace_point_terms(
    setup: &BandPreSolverHandoffSetup,
    energy_index: usize,
    spin: usize,
    k_point_index: usize,
) -> Result<(Complex, Complex, [f64; 3])> {
    let reduced_energy = setup
        .kspace_energy
        .reduced_energy(energy_index, spin)
        .ok_or_else(|| {
            invalid_band_input_error(
                "kspace_point",
                format!(
                    "energy/spin index ({energy_index}, {spin}) is outside ({}, {})",
                    setup.kspace_energy.energy_count, setup.kspace_energy.spin_count
                ),
            )
        })?;
    let wave_number = setup
        .kspace_energy
        .wave_number(energy_index, spin)
        .ok_or_else(|| {
            invalid_band_input_error(
                "kspace_point",
                format!(
                    "energy/spin index ({energy_index}, {spin}) is outside ({}, {})",
                    setup.kspace_energy.energy_count, setup.kspace_energy.spin_count
                ),
            )
        })?;
    if k_point_index >= setup.k_path.mesh.point_count() {
        return Err(invalid_band_input_error(
            "kspace_point",
            format!(
                "k-point index {k_point_index} is outside {} sampled points",
                setup.k_path.mesh.point_count()
            ),
        ));
    }
    let k = [
        setup.k_path.mesh.k_points[(k_point_index, 0)],
        setup.k_path.mesh.k_points[(k_point_index, 1)],
        setup.k_path.mesh.k_points[(k_point_index, 2)],
    ];
    Ok((reduced_energy, wave_number, k))
}

fn kmesh_dat_from_reciprocal_cell_with_operation_rows(
    cell: &ReciprocalCell,
    operations: ArrayView3<'_, i32>,
    rows_kind: KmeshRows,
) -> Result<KmeshDatData> {
    if cell.k_mesh.total <= 0 {
        return invalid_kmesh_dat(
            "total",
            format!(
                "FEFF kmesh.dat generation requires a positive total k-point request, got {}",
                cell.k_mesh.total
            ),
        );
    }

    let direct = reciprocal_cell_direct_vectors(cell);
    let (lengths, angles) = reciprocal_cell_lengths_angles(cell)?;
    let bravais = kmesh_bravais_basis(&cell.lattice_name, lengths, angles)
        .map_err(|source| invalid_kmesh_error("bravais", source.to_string()))?;
    let reciprocal = reciprocal_lattice_vectors(bravais.direct_vectors.view())
        .or_else(|_| reciprocal_lattice_vectors(direct.view()))
        .map_err(|source| invalid_kmesh_error("reciprocal_vectors", source.to_string()))?;
    let requested_points = usize::try_from(cell.k_mesh.total).map_err(|_| {
        invalid_kmesh_error(
            "total",
            "FEFF kmesh.dat total k-point request does not fit usize",
        )
    })?;
    let mesh = kmesh_arbitrary_mesh(
        reciprocal.view(),
        operations,
        requested_points,
        bravais.dependencies,
        false,
    )
    .map_err(|source| invalid_kmesh_error("arbitrary_mesh", source.to_string()))?;
    let coordinate_scale = FEFF_BOHR_ANGSTROM;
    let divisions = [
        usize_to_i32(mesh.divisions[0], "kmesh x division")?,
        usize_to_i32(mesh.divisions[1], "kmesh y division")?,
        usize_to_i32(mesh.divisions[2], "kmesh z division")?,
    ];

    let row_count = if rows_kind == KmeshRows::Irreducible {
        mesh.irreducible_point_count
    } else {
        mesh.full_point_count
    };
    let irreducible_points = usize_to_i32(row_count, "irreducible k-point count")?;
    let mut rows = Vec::with_capacity(row_count);
    for point in 0..row_count {
        let (k_point, weight) = if rows_kind == KmeshRows::Irreducible {
            (
                [
                    mesh.reduction.irreducible_vectors[(point, 0)] * coordinate_scale,
                    mesh.reduction.irreducible_vectors[(point, 1)] * coordinate_scale,
                    mesh.reduction.irreducible_vectors[(point, 2)] * coordinate_scale,
                ],
                mesh.reduction.irreducible_weights[point] * mesh.total_weight / 2.0,
            )
        } else {
            (
                [
                    mesh.reduction.full_vectors[(point, 0)] * coordinate_scale,
                    mesh.reduction.full_vectors[(point, 1)] * coordinate_scale,
                    mesh.reduction.full_vectors[(point, 2)] * coordinate_scale,
                ],
                mesh.reduction.full_weights[point] * mesh.total_weight / 2.0,
            )
        };
        rows.push(KmeshRow {
            index: usize_to_i32(point + 1, "kmesh row index")?,
            k_point,
            weight,
            metadata: (point == 0).then_some(KmeshMetadata {
                requested_points: cell.k_mesh.total,
                irreducible_points,
                divisions,
            }),
        });
    }

    let data = KmeshDatData { rows };
    validate_kmesh_dat(&data)?;
    Ok(data)
}

fn band_scaled_reciprocal_basis(cell: &ReciprocalCell) -> Result<[[f64; 3]; 3]> {
    let direct = reciprocal_cell_direct_vectors(cell);
    let (lengths, angles) = reciprocal_cell_lengths_angles(cell)?;
    let bravais = kmesh_bravais_basis(&cell.lattice_name, lengths, angles)
        .map_err(|source| invalid_reciprocal_error("bravais_basis", source.to_string()))?;
    let reciprocal = reciprocal_lattice_vectors(bravais.direct_vectors.view())
        .or_else(|_| reciprocal_lattice_vectors(direct.view()))
        .map_err(|source| invalid_reciprocal_error("reciprocal_vectors", source.to_string()))?;
    let scale = lengths[0] / (2.0 * std::f64::consts::PI);
    let mut basis = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            basis[row][column] = reciprocal[(row, column)] * scale;
        }
    }
    Ok(basis)
}

fn band_kspace_direct_basis(cell: &ReciprocalCell, alat_angstrom: f64) -> Result<[[f64; 3]; 3]> {
    validate_positive_finite_reciprocal("alat", alat_angstrom)?;
    let mut basis = [[0.0; 3]; 3];
    for (row, basis_row) in basis.iter_mut().enumerate() {
        for (column, basis_value) in basis_row.iter_mut().enumerate() {
            let value = cell.lattice_vectors[row][column] / alat_angstrom;
            validate_finite_reciprocal("direct_basis", value)?;
            *basis_value = value;
        }
    }
    Ok(basis)
}

fn inverse_direct_basis_without_pi2(direct_basis: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let cross = cross(direct_basis[1], direct_basis[2]);
    let determinant = dot(direct_basis[0], cross);
    validate_positive_finite_reciprocal("direct_lattice_determinant", determinant.abs())?;

    let mut reciprocal = [[0.0; 3]; 3];
    reciprocal[0][0] = (direct_basis[1][1] * direct_basis[2][2]
        - direct_basis[2][1] * direct_basis[1][2])
        / determinant;
    reciprocal[1][0] = (direct_basis[2][1] * direct_basis[0][2]
        - direct_basis[0][1] * direct_basis[2][2])
        / determinant;
    reciprocal[2][0] = (direct_basis[0][1] * direct_basis[1][2]
        - direct_basis[1][1] * direct_basis[0][2])
        / determinant;
    reciprocal[0][1] = (direct_basis[1][2] * direct_basis[2][0]
        - direct_basis[2][2] * direct_basis[1][0])
        / determinant;
    reciprocal[1][1] = (direct_basis[2][2] * direct_basis[0][0]
        - direct_basis[0][2] * direct_basis[2][0])
        / determinant;
    reciprocal[2][1] = (direct_basis[0][2] * direct_basis[1][0]
        - direct_basis[1][2] * direct_basis[0][0])
        / determinant;
    reciprocal[0][2] = (direct_basis[1][0] * direct_basis[2][1]
        - direct_basis[2][0] * direct_basis[1][1])
        / determinant;
    reciprocal[1][2] = (direct_basis[2][0] * direct_basis[0][1]
        - direct_basis[0][0] * direct_basis[2][1])
        / determinant;
    reciprocal[2][2] = (direct_basis[0][0] * direct_basis[1][1]
        - direct_basis[1][0] * direct_basis[0][1])
        / determinant;

    for row in reciprocal {
        for value in row {
            validate_finite_reciprocal("reciprocal_basis", value)?;
        }
    }
    Ok(reciprocal)
}

fn band_kspace_q_positions(cell: &ReciprocalCell) -> Result<Array2<f64>> {
    if cell.positions.len() != cell.atom_count {
        return Err(invalid_reciprocal_error(
            "positions",
            format!(
                "reciprocal.inp atom_count is {} but has {} positions",
                cell.atom_count,
                cell.positions.len()
            ),
        ));
    }
    if cell.atom_count == 0 {
        return Err(invalid_reciprocal_error(
            "atom_count",
            "value must be positive",
        ));
    }

    let mut positions = Array2::<f64>::zeros((cell.atom_count, 3));
    for (row, position) in cell.positions.iter().enumerate() {
        for column in 0..3 {
            let value = position[column];
            validate_finite_reciprocal("positions", value)?;
            positions[(row, column)] = value;
        }
    }
    Ok(positions)
}

fn band_kspace_solver_atoms(cell: &ReciprocalCell) -> Result<Vec<FmsAtom>> {
    let positions = band_kspace_q_positions(cell)?;
    if cell.potentials.len() != cell.atom_count {
        return Err(invalid_reciprocal_error(
            "potentials",
            format!(
                "reciprocal.inp atom_count is {} but has {} potentials",
                cell.atom_count,
                cell.potentials.len()
            ),
        ));
    }

    let mut atoms = Vec::with_capacity(cell.atom_count);
    for atom_index in 0..cell.atom_count {
        let potential = cell.potentials[atom_index];
        if potential < 0 {
            return Err(invalid_reciprocal_error(
                "potentials",
                format!("potential index must be non-negative, got {potential}"),
            ));
        }
        atoms.push(FmsAtom {
            position: [
                f64_to_f32_reciprocal("positions", positions[(atom_index, 0)])?,
                f64_to_f32_reciprocal("positions", positions[(atom_index, 1)])?,
                f64_to_f32_reciprocal("positions", positions[(atom_index, 2)])?,
            ],
            potential,
        });
    }
    Ok(atoms)
}

fn band_kspace_strinit_limits(
    search: &BandEnergySearchMesh,
    cell: &ReciprocalCell,
    direct_basis: [[f64; 3]; 3],
) -> Result<(f64, f64, f64)> {
    let volume = dot(direct_basis[0], cross(direct_basis[1], direct_basis[2])).abs();
    validate_positive_finite_reciprocal("direct_lattice_volume", volume)?;
    let etop_rydberg =
        search
            .energies_hartree
            .iter()
            .copied()
            .try_fold(0.0_f64, |maximum, energy| {
                validate_finite_band("energy_mesh", energy)?;
                Ok::<f64, IoError>(maximum.max(2.0 * energy.abs()))
            })?;

    let b = -1.0e-15_f64.ln();
    let preta = 0.75 / volume.powf(2.0 / 3.0) / std::f64::consts::PI;
    let preta2 = preta.max(etop_rydberg / 225.0);
    validate_positive_finite_reciprocal("strinit_preta2", preta2)?;

    let mut eta = cell.stretch[0];
    validate_finite_reciprocal("streta", eta)?;
    if eta < 1.0e-3 {
        eta = 0.5 * preta2.sqrt();
    }
    let rmax = (b / preta2).sqrt() / std::f64::consts::PI;
    let gmax = (b * preta2).sqrt();
    validate_positive_finite_reciprocal("eta", eta)?;
    validate_positive_finite_reciprocal("rmax", rmax)?;
    validate_positive_finite_reciprocal("gmax", gmax)?;
    Ok((eta, rmax, gmax))
}

fn band_kspace_reduced_energy_probe(
    search: &BandEnergySearchMesh,
    alat_bohr: f64,
) -> Result<(f64, f64)> {
    validate_positive_finite_reciprocal("alat", alat_bohr)?;
    if search.energies_hartree.is_empty() {
        return Err(invalid_band_input_error(
            "energy_mesh",
            "search mesh is empty",
        ));
    }
    let scale = (2.0 * std::f64::consts::PI / alat_bohr).powi(2);
    validate_positive_finite_reciprocal("energy_reduction_scale", scale)?;
    let mut min = f64::INFINITY;
    let mut max = 0.0_f64;
    for energy in search.energies_hartree.iter().copied() {
        validate_finite_band("energy_mesh", energy)?;
        let reduced = (2.0 * energy).abs() / scale;
        validate_finite_reciprocal("reduced_energy_probe", reduced)?;
        min = min.min(reduced);
        max = max.max(reduced);
    }
    validate_finite_reciprocal("energy_min_reduced", min)?;
    validate_finite_reciprocal("energy_max_reduced", max)?;
    Ok((min, max))
}

fn reciprocal_cell_first_lattice_length(cell: &ReciprocalCell) -> Result<f64> {
    let length = vector_norm(cell.lattice_vectors[0]);
    validate_positive_finite_reciprocal("lattice_vectors", length)?;
    Ok(length)
}

/// Read FEFF `kmesh.dat` text from a file.
pub fn read_kmesh_dat(path: impl AsRef<Path>) -> Result<KmeshDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_kmesh_dat(&text)
}

/// Write FEFF `kmesh.dat` text to a file.
pub fn write_kmesh_dat(path: impl AsRef<Path>, data: &KmeshDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, kmesh_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

fn validate_bandstructure_dat(data: &BandstructureDatData) -> Result<()> {
    if data.rows.is_empty() {
        return invalid_bandstructure_dat("rows", "at least one k-point row is required");
    }
    for (row_index, row) in data.rows.iter().enumerate() {
        let row_number = row_index + 1;
        validate_positive_i32(
            BANDSTRUCTURE_DAT_PATH,
            "k-point index",
            row.index,
            row_number,
        )?;
        validate_finite_array(BANDSTRUCTURE_DAT_PATH, "k-point", &row.k_point, row_number)?;
        for value in &row.bands {
            validate_finite_value(BANDSTRUCTURE_DAT_PATH, "band energy", *value, row_number)?;
        }
    }
    Ok(())
}

fn validate_kmesh_dat(data: &KmeshDatData) -> Result<()> {
    if data.rows.is_empty() {
        return invalid_kmesh_dat("rows", "at least one k-point row is required");
    }
    for (row_index, row) in data.rows.iter().enumerate() {
        let row_number = row_index + 1;
        validate_positive_i32(KMESH_DAT_PATH, "k-point index", row.index, row_number)?;
        validate_finite_array(KMESH_DAT_PATH, "k-point", &row.k_point, row_number)?;
        validate_finite_value(KMESH_DAT_PATH, "weight", row.weight, row_number)?;
        if let Some(metadata) = row.metadata {
            validate_positive_i32(
                KMESH_DAT_PATH,
                "requested k-points",
                metadata.requested_points,
                row_number,
            )?;
            validate_positive_i32(
                KMESH_DAT_PATH,
                "irreducible k-points",
                metadata.irreducible_points,
                row_number,
            )?;
            for division in metadata.divisions {
                validate_positive_i32(KMESH_DAT_PATH, "k-division", division, row_number)?;
            }
        }
    }
    Ok(())
}

fn validate_bandstructure_energy_metadata(
    input: &BandstructureDatFromEigenvaluesInput<'_>,
) -> Result<Option<(usize, f64, f64, f64)>> {
    let present = [
        input.energy_count.is_some(),
        input.energy_min.is_some(),
        input.energy_max.is_some(),
        input.energy_step.is_some(),
    ]
    .into_iter()
    .filter(|is_present| *is_present)
    .count();
    if present == 0 {
        return Ok(None);
    }
    if present != 4 {
        return invalid_bandstructure_dat(
            "energy_metadata",
            "energy_count, energy_min, energy_max, and energy_step must be supplied together",
        );
    }

    let (Some(energy_count), Some(energy_min), Some(energy_max), Some(energy_step)) = (
        input.energy_count,
        input.energy_min,
        input.energy_max,
        input.energy_step,
    ) else {
        return invalid_bandstructure_dat(
            "energy_metadata",
            "energy_count, energy_min, energy_max, and energy_step must be supplied together",
        );
    };
    if energy_count == 0 {
        return invalid_bandstructure_dat("energy_count", "value must be positive");
    }
    for (field, value) in [
        ("energy_min", energy_min),
        ("energy_max", energy_max),
        ("energy_step", energy_step),
    ] {
        if !value.is_finite() {
            return invalid_bandstructure_dat(field, "value must be finite");
        }
    }
    if energy_step <= 0.0 {
        return invalid_bandstructure_dat("energy_step", "value must be positive");
    }
    if energy_max < energy_min {
        return invalid_bandstructure_dat(
            "energy_max",
            "value must be greater than or equal to energy_min",
        );
    }
    Ok(Some((energy_count, energy_min, energy_max, energy_step)))
}

fn generated_bandstructure_header_lines(
    k_point_count: usize,
    rows: &[BandstructureRow],
    energy_metadata: Option<(usize, f64, f64, f64)>,
) -> Vec<String> {
    let mut header_lines = vec![format!(" # grid of {k_point_count:12}  k-points.")];
    if let Some((energy_count, energy_min, energy_max, energy_step)) = energy_metadata {
        header_lines.push(format!(
            " # grid of {energy_count:12}  energy points  emin= {energy_min:21.17}       , emax= {energy_max:21.17}       , estep= {energy_step:21.17}     "
        ));
    }
    header_lines.push(format!(
        " # Found between {:12}  and {:12}  number of bands.",
        rows.iter().map(|row| row.bands.len()).min().unwrap_or(0),
        rows.iter().map(|row| row.bands.len()).max().unwrap_or(0)
    ));
    header_lines
}

fn reciprocal_cell_direct_vectors(cell: &ReciprocalCell) -> Array2<f64> {
    Array2::from_shape_fn((3, 3), |(row, column)| cell.lattice_vectors[row][column])
}

fn reciprocal_cell_lengths_angles(cell: &ReciprocalCell) -> Result<([f64; 3], [f64; 3])> {
    let vectors = cell.lattice_vectors;
    let lengths = [
        vector_norm(vectors[0]),
        vector_norm(vectors[1]),
        vector_norm(vectors[2]),
    ];
    for (index, length) in lengths.into_iter().enumerate() {
        if !length.is_finite() || length <= 0.0 {
            return invalid_kmesh_dat(
                "lattice_vectors",
                format!("FEFF reciprocal lattice vector {index} has invalid length {length}"),
            );
        }
    }

    let angles = [
        vector_angle(vectors[1], vectors[2], "alpha")?,
        vector_angle(vectors[0], vectors[2], "beta")?,
        vector_angle(vectors[0], vectors[1], "gamma")?,
    ];
    Ok((lengths, angles))
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    vector[0].hypot(vector[1]).hypot(vector[2])
}

fn vector_angle(left: [f64; 3], right: [f64; 3], name: &'static str) -> Result<f64> {
    let left_norm = vector_norm(left);
    let right_norm = vector_norm(right);
    if left_norm <= 0.0 || right_norm <= 0.0 {
        return invalid_kmesh_dat(
            "lattice_vectors",
            format!("FEFF reciprocal lattice angle {name} has a degenerate vector"),
        );
    }
    let cosine = dot(left, right) / (left_norm * right_norm);
    if !cosine.is_finite() {
        return invalid_kmesh_dat(
            "lattice_vectors",
            format!("FEFF reciprocal lattice angle {name} is non-finite"),
        );
    }
    Ok(cosine.clamp(-1.0, 1.0).acos())
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn identity_kmesh_operations() -> Array3<i32> {
    let mut operations = Array3::zeros((1, 3, 3));
    for axis in 0..3 {
        operations[(0, axis, axis)] = 1;
    }
    operations
}

fn lattice_centering(lattice_name: &str) -> Result<char> {
    lattice_name
        .chars()
        .find(|character| !character.is_whitespace())
        .map(|character| character.to_ascii_uppercase())
        .ok_or_else(|| invalid_reciprocal_error("lattice", "lattice name is empty"))
}

fn usize_to_i32(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid_kmesh_error(name, format!("{value} does not fit i32")))
}

fn usize_to_i32_bandstructure(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| IoError::Parse {
        path: BANDSTRUCTURE_DAT_PATH.into(),
        line: 0,
        message: format!("{name}: {value} does not fit i32"),
    })
}

fn validate_positive_i32(
    path: &'static str,
    field: &'static str,
    value: i32,
    row: usize,
) -> Result<()> {
    if value > 0 {
        Ok(())
    } else {
        invalid_dat(path, field, format!("row {row} value must be positive"))
    }
}

fn validate_finite_array(
    path: &'static str,
    field: &'static str,
    values: &[f64],
    row: usize,
) -> Result<()> {
    for value in values {
        validate_finite_value(path, field, *value, row)?;
    }
    Ok(())
}

fn validate_finite_value(
    path: &'static str,
    field: &'static str,
    value: f64,
    row: usize,
) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_dat(path, field, format!("row {row} value must be finite"))
    }
}

fn parse_i32(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<i32> {
    token
        .parse::<i32>()
        .map_err(|_| parse_error_value(path, line, format!("invalid {field} value {token:?}")))
}

fn parse_usize(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<usize> {
    token
        .parse::<usize>()
        .map_err(|_| parse_error_value(path, line, format!("invalid {field} value {token:?}")))
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(path, line, format!("invalid {field} value {token:?}")))
}

fn invalid_bandstructure_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    invalid_dat(BANDSTRUCTURE_DAT_PATH, field, message)
}

fn invalid_kmesh_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    invalid_dat(KMESH_DAT_PATH, field, message)
}

fn invalid_kmesh_error(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: KMESH_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

fn invalid_band_input_error(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: BAND_INP_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

fn validate_finite_reciprocal(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_reciprocal_error(field, "value must be finite"))
    }
}

fn validate_band_kspace_non_rel_solve_setup(setup: &BandPreSolverHandoffSetup) -> Result<()> {
    if setup.kspace_solver_basis.spin_channels != 1 || setup.kspace_energy.spin_count != 1 {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "ordinary non-rel BAND grid currently requires one spin channel, got solver={} energy={}",
                setup.kspace_solver_basis.spin_channels, setup.kspace_energy.spin_count
            ),
        ));
    }
    if setup.kspace_solver_basis.matrix_order != setup.kspace_angular.matrix_order_non_rel {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "solver matrix order {} does not match non-rel KSPACE order {}",
                setup.kspace_solver_basis.matrix_order, setup.kspace_angular.matrix_order_non_rel
            ),
        ));
    }
    if setup.search.phase_interpolation.wave_numbers.ncols() == 0 {
        return Err(invalid_band_input_error(
            "kspace_grid",
            "BAND solve requires at least one wave-number column",
        ));
    }
    Ok(())
}

fn validate_band_kspace_rel_solve_setup(setup: &BandPreSolverHandoffSetup) -> Result<()> {
    if setup.kspace_solver_basis.spin_channels != 1 || setup.kspace_energy.spin_count != 1 {
        return Err(invalid_band_input_error(
            "kspace_rel_grid",
            format!(
                "ordinary relativistic BAND grid currently requires one spin channel, got solver={} energy={}",
                setup.kspace_solver_basis.spin_channels, setup.kspace_energy.spin_count
            ),
        ));
    }
    if setup.kspace_solver_basis.matrix_order != setup.kspace_angular.matrix_order_rel {
        return Err(invalid_band_input_error(
            "kspace_rel_grid",
            format!(
                "solver matrix order {} does not match relativistic KSPACE order {}",
                setup.kspace_solver_basis.matrix_order, setup.kspace_angular.matrix_order_rel
            ),
        ));
    }
    if setup.search.phase_interpolation.wave_numbers.ncols() == 0 {
        return Err(invalid_band_input_error(
            "kspace_rel_grid",
            "BAND rel solve requires at least one wave-number column",
        ));
    }
    Ok(())
}

fn validate_band_kspace_spin_degenerate_solve_setup(
    setup: &BandPreSolverHandoffSetup,
) -> Result<()> {
    let spin_channels = setup.kspace_solver_basis.spin_channels;
    if spin_channels <= 1 || setup.kspace_energy.spin_count <= 1 {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "spin-degenerate BAND grid requires multiple spin channels, got solver={} energy={}",
                spin_channels, setup.kspace_energy.spin_count
            ),
        ));
    }
    if spin_channels != setup.kspace_energy.spin_count {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "solver spin channel count {spin_channels} does not match energy spin count {}",
                setup.kspace_energy.spin_count
            ),
        ));
    }
    let expected_order = setup.kspace_angular.matrix_order_rel;
    if setup.kspace_solver_basis.matrix_order != expected_order {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "solver matrix order {} does not match relativistic spin KSPACE order {expected_order}",
                setup.kspace_solver_basis.matrix_order
            ),
        ));
    }
    if setup.search.phase_interpolation.wave_numbers.ncols() < spin_channels {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "BAND solve requires {spin_channels} wave-number columns, got {}",
                setup.search.phase_interpolation.wave_numbers.ncols()
            ),
        ));
    }
    validate_spin_degenerate_kspace_energies(setup)
}

fn validate_band_kspace_spin_resolved_solve_setup(
    setup: &BandPreSolverHandoffSetup,
) -> Result<usize> {
    let spin_channels = setup.kspace_solver_basis.spin_channels;
    if spin_channels <= 1 || setup.kspace_energy.spin_count <= 1 {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "spin-resolved BAND grid requires multiple spin channels, got solver={} energy={}",
                spin_channels, setup.kspace_energy.spin_count
            ),
        ));
    }
    if spin_channels != setup.kspace_energy.spin_count {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "solver spin channel count {spin_channels} does not match energy spin count {}",
                setup.kspace_energy.spin_count
            ),
        ));
    }
    let expected_order = setup.kspace_angular.matrix_order_rel;
    if setup.kspace_solver_basis.matrix_order != expected_order {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "solver matrix order {} does not match relativistic spin KSPACE order {expected_order}",
                setup.kspace_solver_basis.matrix_order
            ),
        ));
    }
    if setup.search.phase_interpolation.wave_numbers.ncols() < spin_channels {
        return Err(invalid_band_input_error(
            "kspace_grid",
            format!(
                "BAND solve requires {spin_channels} wave-number columns, got {}",
                setup.search.phase_interpolation.wave_numbers.ncols()
            ),
        ));
    }
    Ok(spin_channels - 1)
}

fn validate_spin_degenerate_kspace_energies(setup: &BandPreSolverHandoffSetup) -> Result<()> {
    const TOLERANCE: f64 = 1.0e-10;

    for energy_index in 0..setup.kspace_energy.energy_count {
        let base_wave = setup.kspace_energy.wave_numbers[(energy_index, 0)];
        let base_reduced = setup.kspace_energy.reduced_energies[(energy_index, 0)];
        for spin in 1..setup.kspace_energy.spin_count {
            let wave = setup.kspace_energy.wave_numbers[(energy_index, spin)];
            let reduced = setup.kspace_energy.reduced_energies[(energy_index, spin)];
            if complex_abs_diff(base_wave, wave) > TOLERANCE {
                return Err(invalid_band_input_error(
                    "kspace_grid",
                    format!(
                        "spin-degenerate BAND grid requires matching wave numbers; energy {energy_index} spin {spin} differs"
                    ),
                ));
            }
            if complex_abs_diff(base_reduced, reduced) > TOLERANCE {
                return Err(invalid_band_input_error(
                    "kspace_grid",
                    format!(
                        "spin-degenerate BAND grid requires matching reduced energies; energy {energy_index} spin {spin} differs"
                    ),
                ));
            }
        }
    }

    Ok(())
}

fn complex_abs_diff(left: Complex, right: Complex) -> f64 {
    (left - right).norm()
}

fn validate_band_kspace_structure_factor_order(
    point_solve: &BandStructureFactorFromKspace,
    expected_order: usize,
) -> Result<()> {
    let (rows, columns) = point_solve.structure_factor.dim();
    if rows == expected_order && columns == expected_order {
        Ok(())
    } else {
        Err(invalid_band_input_error(
            "kspace_structure_factor",
            format!(
                "structure-factor order ({rows}, {columns}) does not match solver order {expected_order}"
            ),
        ))
    }
}

fn f64_to_f32_reciprocal(field: &'static str, value: f64) -> Result<f32> {
    validate_finite_reciprocal(field, value)?;
    if value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err(invalid_reciprocal_error(
            field,
            format!("value {value} does not fit f32"),
        ));
    }
    Ok(value as f32)
}

fn validate_positive_finite_reciprocal(field: &'static str, value: f64) -> Result<()> {
    validate_finite_reciprocal(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(invalid_reciprocal_error(
            field,
            format!("value must be positive, got {value}"),
        ))
    }
}

fn validate_complex_finite_reciprocal(field: &'static str, value: Complex) -> Result<()> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(invalid_reciprocal_error(
            field,
            format!("complex value must be finite, got {value:?}"),
        ))
    }
}

fn validate_finite_band(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_band_input_error(field, "value must be finite"))
    }
}

fn band_kspace_non_rel_site_layout(
    cell: &ReciprocalCell,
    angular_state_count: usize,
) -> Result<(Vec<usize>, Vec<usize>, usize)> {
    band_kspace_site_layout(cell, angular_state_count)
}

fn band_kspace_rel_site_layout(
    cell: &ReciprocalCell,
    active_rel_state_count: usize,
    rel_components: &BandKspaceRelComponentHandoffSetup,
) -> Result<(Vec<usize>, Vec<usize>, usize)> {
    if rel_components.component_counts.ncols() < active_rel_state_count {
        return Err(invalid_band_input_error(
            "rel_components",
            format!(
                "component table has {} relativistic state columns, but BAND needs {active_rel_state_count}",
                rel_components.component_counts.ncols()
            ),
        ));
    }
    band_kspace_site_layout(cell, active_rel_state_count)
}

fn band_kspace_site_layout(
    cell: &ReciprocalCell,
    state_count: usize,
) -> Result<(Vec<usize>, Vec<usize>, usize)> {
    if cell.positions.is_empty() {
        return Err(invalid_reciprocal_error(
            "positions",
            "KSPACE angular setup requires at least one site",
        ));
    }
    if state_count == 0 {
        return Err(invalid_reciprocal_error(
            "state_count",
            "KSPACE angular setup requires at least one angular state",
        ));
    }

    let mut site_offsets = Vec::with_capacity(cell.positions.len());
    let mut matrix_order = 0usize;
    for _ in &cell.positions {
        site_offsets.push(matrix_order);
        matrix_order = matrix_order
            .checked_add(state_count)
            .ok_or_else(|| invalid_reciprocal_error("site_offsets", "matrix order overflowed"))?;
    }
    let site_state_counts = vec![state_count; cell.positions.len()];
    Ok((site_offsets, site_state_counts, matrix_order))
}

fn invalid_reciprocal_error(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: RECIPROCAL_INP_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

fn invalid_dat<T>(
    path: &'static str,
    field: &'static str,
    message: impl Into<String>,
) -> Result<T> {
    Err(IoError::Parse {
        path: path.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    })
}

fn bandstructure_parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(BANDSTRUCTURE_DAT_PATH, line, message))
}

fn kmesh_parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(KMESH_DAT_PATH, line, message))
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
    use super::*;
    use ndarray::{Array2, Array3, Array4, array};
    use num_complex::{Complex32, Complex64};
    use refeff_core::FEFF_HARTREE_EV;

    use crate::phase_bin::PHASE_BIN_DEFAULT_PAD_WIDTH;
    use crate::{PhaseBinPotential, PhaseBinScalars};

    #[test]
    fn parses_bandstructure_dat() -> Result<()> {
        let data = parse_bandstructure_dat(SAMPLE_BANDSTRUCTURE)?;

        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.k_point_count(), 2);
        assert_eq!(data.min_band_count(), 1);
        assert_eq!(data.max_band_count(), 2);
        assert_eq!(data.rows[0].index, 1);
        assert_eq!(data.rows[0].k_point, [0.0, 0.5, 0.25]);
        assert_eq!(data.rows[0].bands.as_slice(), Some(&[-5.0, 1.25][..]));
        assert_eq!(data.rows[1].bands.as_slice(), Some(&[0.75][..]));
        Ok(())
    }

    #[test]
    fn roundtrips_bandstructure_dat() -> Result<()> {
        let data = parse_bandstructure_dat(SAMPLE_BANDSTRUCTURE)?;
        let rendered = bandstructure_dat_string(&data)?;
        let reparsed = parse_bandstructure_dat(&rendered)?;

        assert_eq!(reparsed, data);
        Ok(())
    }

    #[test]
    fn builds_bandstructure_dat_from_eigenvalues() -> Result<()> {
        let k_points = array![[0.0, 0.5, 0.25], [0.125, 0.0, 0.5]];
        let band_rows = [array![-5.0, 1.25], array![0.75]];
        let band_views = band_rows.iter().map(|row| row.view()).collect::<Vec<_>>();
        let data = bandstructure_dat_from_eigenvalues(BandstructureDatFromEigenvaluesInput {
            header_lines: &[],
            k_points: k_points.view(),
            band_energies: &band_views,
            energy_count: Some(3),
            energy_min: Some(-1.0),
            energy_max: Some(1.0),
            energy_step: Some(1.0),
        })?;

        assert_eq!(
            data.header_lines,
            vec![
                " # grid of            2  k-points.",
                " # grid of            3  energy points  emin=  -1.00000000000000000       , emax=   1.00000000000000000       , estep=   1.00000000000000000     ",
                " # Found between            1  and            2  number of bands.",
            ]
        );
        assert_eq!(data.rows[0].index, 1);
        assert_eq!(data.rows[1].index, 2);
        assert_eq!(data.min_band_count(), 1);
        assert_eq!(data.max_band_count(), 2);
        assert_eq!(
            parse_bandstructure_dat(&bandstructure_dat_string(&data)?)?,
            data
        );
        Ok(())
    }

    #[test]
    fn builds_bandstructure_dat_from_typed_band_result() -> Result<()> {
        let k_path =
            band_k_path_setup_from_handoffs(&sample_band_input(6, 5), &sample_reciprocal_cell(8))?;
        let energy_mesh = BandEnergySearchMesh {
            min_hartree: -1.0,
            max_hartree: 1.0,
            step_hartree: 0.5,
            energies_hartree: array![-1.0, -0.5, 0.0, 0.5, 1.0],
        };
        let band_energies = BandEnergiesFromPositiveCounts {
            band_energies_hartree: (0..k_path.mesh.point_count())
                .map(|index| array![index as f64 + 1.0])
                .collect(),
        };

        let data = bandstructure_dat_from_band_result(BandstructureDatFromBandResultInput {
            header_lines: &[],
            k_path: &k_path,
            energy_mesh: &energy_mesh,
            phase_energy_count: None,
            band_energies: &band_energies,
        })?;

        assert_eq!(data.rows.len(), k_path.mesh.point_count());
        assert!(data.header_lines[1].contains(" # grid of            5  energy points"));
        assert_close(data.rows[0].k_point[0], k_path.mesh.k_points[(0, 0)]);
        assert_close(data.rows[0].bands[0], 1.0);
        assert_close(data.rows[5].bands[0], 6.0);
        assert_eq!(
            parse_bandstructure_dat(&bandstructure_dat_string(&data)?)?,
            data
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_bandstructure_dat() {
        assert!(parse_bandstructure_dat("    1 0.0 0.0 0.0 2 1.0\n").is_err());
        assert!(parse_bandstructure_dat("# header only\n").is_err());
    }

    #[test]
    fn rejects_bad_bandstructure_eigenvalue_inputs() {
        let k_points = array![[0.0, 0.5, 0.25], [0.125, 0.0, 0.5]];
        let band_rows = [array![-5.0, 1.25]];
        let band_views = band_rows.iter().map(|row| row.view()).collect::<Vec<_>>();

        assert!(
            bandstructure_dat_from_eigenvalues(BandstructureDatFromEigenvaluesInput {
                header_lines: &[],
                k_points: k_points.view(),
                band_energies: &band_views,
                energy_count: Some(3),
                energy_min: Some(-1.0),
                energy_max: Some(1.0),
                energy_step: Some(1.0),
            })
            .is_err()
        );
        assert!(
            bandstructure_dat_from_eigenvalues(BandstructureDatFromEigenvaluesInput {
                header_lines: &[],
                k_points: k_points.view(),
                band_energies: &band_views,
                energy_count: Some(3),
                energy_min: None,
                energy_max: Some(1.0),
                energy_step: Some(1.0),
            })
            .is_err()
        );
    }

    #[test]
    fn parses_kmesh_dat() -> Result<()> {
        let data = parse_kmesh_dat(SAMPLE_KMESH)?;

        assert_eq!(data.k_point_count(), 2);
        assert_eq!(data.rows[0].index, 1);
        assert_eq!(data.rows[0].k_point, [0.0, 0.5, 0.25]);
        assert_eq!(data.rows[0].weight, 0.75);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 100,
                irreducible_points: 2,
                divisions: [4, 5, 6],
            })
        );
        assert_eq!(data.rows[1].metadata, None);
        Ok(())
    }

    #[test]
    fn roundtrips_kmesh_dat() -> Result<()> {
        let data = parse_kmesh_dat(SAMPLE_KMESH)?;
        let rendered = kmesh_dat_string(&data)?;
        let reparsed = parse_kmesh_dat(&rendered)?;

        assert_eq!(reparsed, data);
        Ok(())
    }

    #[test]
    fn builds_kmesh_dat_from_reciprocal_cell() -> Result<()> {
        let data = kmesh_dat_from_reciprocal_cell(&sample_reciprocal_cell(8))?;

        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 8,
                irreducible_points: 8,
                divisions: [2, 2, 2],
            })
        );
        assert_close(data.rows[0].k_point[0], 0.831_446_454_055_273_6);
        assert_close(data.rows[0].k_point[1], 0.831_446_454_055_273_6);
        assert_close(data.rows[0].k_point[2], 0.831_446_454_055_273_6);
        assert_eq!(data.rows[0].weight, 0.5);
        Ok(())
    }

    #[test]
    fn builds_kmesh_dat_from_reciprocal_cell_with_explicit_operations() -> Result<()> {
        let operations = array![
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        ];
        let full = kmesh_dat_from_reciprocal_cell(&sample_reciprocal_cell(8))?;
        let data = kmesh_dat_from_reciprocal_cell_with_operations(
            &sample_reciprocal_cell(8),
            operations.view(),
        )?;

        assert!(!data.rows.is_empty());
        assert!(data.rows.len() < full.rows.len());
        let metadata = data.rows[0].metadata.expect("metadata");
        assert_eq!(metadata.requested_points, 8);
        assert_eq!(metadata.irreducible_points, data.rows.len() as i32);
        let rendered = kmesh_dat_string(&data)?;
        assert_eq!(parse_kmesh_dat(&rendered)?.rows.len(), data.rows.len());
        Ok(())
    }

    #[test]
    fn reciprocal_fms_fails_closed_for_multi_spin_or_spin_selector() {
        let cell = sample_reciprocal_cell(8);
        let energies = array![Complex64::new(0.1, 0.0)];
        let references = Array2::from_elem((1, 2), Complex64::new(0.0, 0.0));
        let probe = array![-3.0, 3.0];
        let spin_error = fms_kspace_setup_from_handoffs(
            &cell,
            energies.view(),
            references.view(),
            probe.view(),
            1,
            2,
            0,
        )
        .expect_err("two-spin reciprocal FMS must fail closed");
        assert!(spin_error.to_string().contains("exactly one spin channel"));

        let references = Array2::from_elem((1, 1), Complex64::new(0.0, 0.0));
        let selector_error = fms_kspace_setup_from_handoffs(
            &cell,
            energies.view(),
            references.view(),
            probe.view(),
            1,
            1,
            1,
        )
        .expect_err("nonzero reciprocal spin selector must fail closed");
        assert!(selector_error.to_string().contains("spin selector"));
    }

    #[test]
    fn reciprocal_fms_fails_closed_for_unavailable_symmetry_rotations() {
        let mut cell = sample_reciprocal_cell(8);
        cell.k_mesh.use_symmetry = true;
        let energies = array![Complex64::new(0.1, 0.0)];
        let references = Array2::from_elem((1, 1), Complex64::new(0.0, 0.0));
        let probe = array![-3.0, 3.0];
        let error = fms_kspace_setup_from_handoffs(
            &cell,
            energies.view(),
            references.view(),
            probe.view(),
            1,
            1,
            0,
        )
        .expect_err("symmetry-reduced reciprocal FMS must fail closed");
        assert!(error.to_string().contains("symmetry reduction"));

        cell.k_mesh.use_symmetry = false;
        cell.lattice_name = "I".to_string();
        let centered_error = fms_kspace_setup_from_handoffs(
            &cell,
            energies.view(),
            references.view(),
            probe.view(),
            1,
            1,
            0,
        )
        .expect_err("centered reciprocal FMS must fail closed");
        assert!(centered_error.to_string().contains("only primitive P/H"));

        cell.lattice_name = "P".to_string();
        cell.imaginary_energy = 0.01;
        let imaginary_error = fms_kspace_setup_from_handoffs(
            &cell,
            energies.view(),
            references.view(),
            probe.view(),
            1,
            1,
            0,
        )
        .expect_err("nonzero reciprocal STRCC imaginary override must fail closed");
        assert!(imaginary_error.to_string().contains("imaginary-energy"));

        cell.imaginary_energy = 0.0;
        cell.volume_scale = f64::NAN;
        let volume_error = fms_kspace_setup_from_handoffs(
            &cell,
            energies.view(),
            references.view(),
            probe.view(),
            1,
            1,
            0,
        )
        .expect_err("non-finite reciprocal volume scaling must fail closed");
        assert!(volume_error.to_string().contains("finite and non-positive"));
    }

    #[test]
    fn builds_band_k_path_setup_from_handoffs() -> Result<()> {
        let setup =
            band_k_path_setup_from_handoffs(&sample_band_input(6, 5), &sample_reciprocal_cell(8))?;

        assert_eq!(setup.bravais, BravaisLattice::CubicPrimitive);
        assert_eq!(setup.path.labels, ["GG-GD-X ", "GG-GD-Y ", "GG-GD-Z "]);
        assert_eq!(setup.reciprocal_basis, identity_basis());
        assert_eq!(setup.mesh.segment_point_counts, [2, 2, 2]);
        assert_eq!(setup.mesh.segment_end_indices, [2, 4, 6]);
        assert_eq!(setup.mesh.point_count(), 6);
        assert_close(setup.mesh.k_points[(1, 0)], 0.5);
        assert_close(setup.mesh.k_points[(3, 1)], 0.5);
        assert_close(setup.mesh.k_points[(5, 2)], 0.5);
        assert_close(setup.mesh.path_distances[2], setup.mesh.path_distances[1]);
        Ok(())
    }

    #[test]
    fn builds_band_kspace_lattice_setup_from_handoffs() -> Result<()> {
        let search =
            band_search_setup_from_handoffs(&sample_band_input(6, 5), &sample_band_phase_bin())?;
        let setup = band_kspace_lattice_setup_from_handoffs(
            &search.energy_mesh,
            &sample_reciprocal_cell(8),
        )?;

        assert_close(setup.alat_bohr, 1.0 / FEFF_BOHR_ANGSTROM);
        assert_eq!(setup.direct_basis, identity_basis());
        assert_eq!(setup.reciprocal_basis, identity_basis());
        assert!(setup.eta > 0.0);
        assert!(setup.rmax > 0.0);
        assert!(setup.gmax > 0.0);
        assert!(setup.energy_min_reduced <= setup.energy_max_reduced);
        assert_eq!(setup.q_pairs.len(), 1);
        assert_eq!(setup.q_pairs.counts, [1]);
        assert_eq!(setup.q_pairs.offsets.shape(), &[1, 3]);
        assert_eq!(setup.direct_lattice.direct_counts.len(), 1);
        assert_eq!(
            setup.direct_lattice.direct_counts[0] + 1,
            setup.direct_lattice.direct_indices.nrows()
        );
        assert!(setup.reciprocal_lattice.reciprocal_indices.nrows() > 1);
        assert_close(
            setup.reciprocal_lattice.gmax_squared,
            setup.gmax * setup.gmax,
        );
        Ok(())
    }

    #[test]
    fn builds_band_kspace_rel_component_setup_for_s_state() -> Result<()> {
        let transforms = basis_transform_matrices(0)
            .map_err(|source| invalid_band_input_error("basis_transforms", source.to_string()))?;
        let setup = band_kspace_rel_component_setup_from_basis_transforms(&transforms, 1)?;

        assert_eq!(setup.component_counts.dim(), (2, 2));
        assert_eq!(setup.component_indices.dim(), (1, 2, 2));
        assert_eq!(setup.component_coefficients.dim(), (1, 2, 2));
        assert_eq!(setup.component_counts[(0, 0)], 1);
        assert_eq!(setup.component_counts[(1, 0)], 0);
        assert_eq!(setup.component_indices[(0, 0, 0)], 0);
        assert_complex_close(
            setup.component_coefficients[(0, 0, 0)],
            Complex::new(1.0, 0.0),
        );
        for state in 0..transforms.order {
            assert!(
                setup.component_counts[(0, state)] + setup.component_counts[(1, state)] > 0,
                "relativistic state {state} should have at least one component"
            );
        }
        Ok(())
    }

    #[test]
    fn builds_band_kspace_angular_setup_from_handoffs() -> Result<()> {
        let search =
            band_search_setup_from_handoffs(&sample_band_input(6, 5), &sample_band_phase_bin())?;
        let lattice = band_kspace_lattice_setup_from_handoffs(
            &search.energy_mesh,
            &sample_reciprocal_cell(8),
        )?;
        let setup =
            band_kspace_angular_setup_from_handoffs(&search, &lattice, &sample_reciprocal_cell(8))?;

        assert_eq!(setup.angular_lmax, 1);
        assert_eq!(setup.harmonic_lmax, 2);
        assert_eq!(setup.angular_state_count, 4);
        assert_eq!(setup.matrix_order_non_rel, 4);
        assert_eq!(setup.site_offsets, [0]);
        assert_eq!(setup.site_state_counts, [4]);
        assert_eq!(setup.matrix_order_rel, 4);
        assert_eq!(setup.rel_site_offsets, [0]);
        assert_eq!(setup.rel_site_state_counts, [4]);
        assert_eq!(setup.angular_tables.qjltab.shape(), &[3, 3]);
        assert_eq!(setup.angular_tables.gaunt_counts.len(), 10);
        assert_eq!(setup.angular_tables.cipwl.len(), 9);
        assert_eq!(setup.basis_transforms.order, 8);
        assert_eq!(setup.rel_components.component_counts.dim(), (2, 8));
        assert_eq!(setup.rel_components.component_indices.dim(), (4, 2, 8));
        assert_eq!(setup.rel_components.component_coefficients.dim(), (4, 2, 8));
        for state in 0..setup.basis_transforms.order {
            assert!(
                setup.rel_components.component_counts[(0, state)]
                    + setup.rel_components.component_counts[(1, state)]
                    > 0,
                "relativistic state {state} should have at least one component"
            );
            for spin in 0..2 {
                let count = setup.rel_components.component_counts[(spin, state)];
                assert!(count <= setup.angular_state_count);
                for term in 0..count {
                    assert!(
                        setup.rel_components.component_indices[(term, spin, state)]
                            < setup.angular_state_count
                    );
                    let coefficient =
                        setup.rel_components.component_coefficients[(term, spin, state)];
                    assert!(coefficient.re.is_finite());
                    assert!(coefficient.im.is_finite());
                }
            }
        }
        assert_eq!(setup.spin_orbit.plus.dim(), (2, 3, 2));
        Ok(())
    }

    #[test]
    fn builds_band_kspace_angular_setup_with_full_two_spin_rel_order() -> Result<()> {
        let search = band_search_setup_from_handoffs(
            &sample_band_input(6, 5),
            &sample_two_spin_degenerate_band_phase_bin(),
        )?;
        let lattice = band_kspace_lattice_setup_from_handoffs(
            &search.energy_mesh,
            &sample_reciprocal_cell(8),
        )?;
        let setup =
            band_kspace_angular_setup_from_handoffs(&search, &lattice, &sample_reciprocal_cell(8))?;

        assert_eq!(setup.angular_state_count, 4);
        assert_eq!(setup.matrix_order_non_rel, 4);
        assert_eq!(setup.site_state_counts, [4]);
        assert_eq!(setup.matrix_order_rel, 8);
        assert_eq!(setup.rel_site_offsets, [0]);
        assert_eq!(setup.rel_site_state_counts, [8]);
        assert_eq!(setup.basis_transforms.order, 8);
        assert_eq!(setup.rel_components.component_counts.dim(), (2, 8));
        Ok(())
    }

    #[test]
    fn builds_band_kspace_energy_setup_from_handoffs() -> Result<()> {
        let band = sample_sparse_band_input(6, 5);
        let search = band_search_setup_from_handoffs(&band, &sample_band_phase_bin())?;
        let lattice = band_kspace_lattice_setup_from_handoffs(
            &search.energy_mesh,
            &sample_reciprocal_cell(8),
        )?;
        let setup = band_kspace_energy_setup_from_handoffs(&search, &lattice)?;

        assert_eq!(setup.energy_count, 4);
        assert_eq!(setup.spin_count, 1);
        assert_eq!(setup.j22max, BAND_KSPACE_J22MAX);
        assert_close(
            setup.reduced_energy_scale,
            (std::f64::consts::TAU / lattice.alat_bohr).powi(2),
        );
        let expected_eryd = Complex::new(2.0, 0.0)
            * (Complex::new(search.energy_mesh.energies_hartree[0], 0.0)
                - search.phase_interpolation.reference_energies_hartree[(0, 0)]);
        assert_complex_close(setup.wave_numbers[(0, 0)], expected_eryd.sqrt());
        assert_complex_close(
            setup.reduced_energies[(0, 0)],
            expected_eryd / setup.reduced_energy_scale,
        );
        assert!(setup.reduced_energy(4, 0).is_none());
        assert!(setup.wave_number(0, 1).is_none());
        Ok(())
    }

    #[test]
    fn builds_band_kspace_solver_basis_setup_from_handoffs() -> Result<()> {
        let search =
            band_search_setup_from_handoffs(&sample_band_input(6, 5), &sample_band_phase_bin())?;
        let setup =
            band_kspace_solver_basis_setup_from_handoffs(&search, &sample_reciprocal_cell(8))?;

        assert_eq!(setup.spin_channels, 1);
        assert_eq!(setup.spin_selector, 0);
        assert_eq!(setup.atoms.len(), 1);
        assert_eq!(setup.atoms[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(setup.atoms[0].potential, 0);
        assert_eq!(setup.states.len(), 4);
        assert_eq!(setup.matrix_order, 4);
        assert_eq!(setup.representative_offsets, [Some(0)]);
        assert_eq!(setup.states[0].atom, 1);
        assert_eq!(setup.states[0].angular_momentum, 0);
        assert_eq!(setup.states[0].magnetic, 0);
        assert_eq!(setup.states[0].spin, 1);
        assert_eq!(setup.states[3].angular_momentum, 1);
        assert_eq!(setup.states[3].magnetic, 1);
        Ok(())
    }

    #[test]
    fn builds_band_kspace_t_matrix_grid_from_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let grid = band_kspace_t_matrix_grid_from_handoffs(&setup)?;

        assert_eq!(grid.dim(), (4, 4, 4));
        assert!(grid[(0, 0, 0)].re.is_finite());
        assert!(grid[(0, 0, 0)].im.is_finite());
        assert_eq!(grid[(0, 0, 1)], Complex32::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn solves_band_kspace_non_rel_grid_from_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let solved = band_kspace_non_rel_solve_from_handoffs(&setup)?;

        assert_eq!(solved.structure_factors.point_solves.len(), 24);
        assert_eq!(
            solved.structure_factors.structure_factors.dim(),
            (4, 6, 4, 4)
        );
        assert_eq!(solved.t_matrices.dim(), (4, 4, 4));
        assert_eq!(solved.solved.eigenvalues.dim(), (4, 6, 4));
        assert_eq!(solved.solved.positive_counts.dim(), (4, 6));
        assert_eq!(solved.solved.band_energies.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn solves_band_kspace_free_propagation_non_rel_grid_from_handoffs() -> Result<()> {
        let mut band = sample_sparse_band_input(6, 5);
        band.freeprop = true;
        let setup = band_pre_solver_setup_from_handoffs(
            &band,
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let solved = band_kspace_free_propagation_non_rel_solve_from_handoffs(&setup)?;

        assert_eq!(solved.structure_factors.point_solves.len(), 24);
        assert_eq!(
            solved.structure_factors.structure_factors.dim(),
            (4, 6, 4, 4)
        );
        assert_eq!(solved.solved.eigenvalues.dim(), (4, 6, 4));
        assert_eq!(solved.solved.positive_counts.dim(), (4, 6));
        assert_eq!(solved.solved.band_energies.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn solves_band_kspace_rel_grid_from_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let solved = band_kspace_rel_solve_from_handoffs(&setup)?;

        assert_eq!(
            solved.structure_factors.structure_factors.dim(),
            (4, 6, 4, 4)
        );
        assert_eq!(solved.t_matrices.dim(), (4, 4, 4));
        assert_eq!(solved.solved.eigenvalues.dim(), (4, 6, 4));
        assert_eq!(solved.solved.positive_counts.dim(), (4, 6));
        assert_eq!(solved.solved.band_energies.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn solves_band_kspace_free_propagation_rel_grid_from_handoffs() -> Result<()> {
        let mut band = sample_sparse_band_input(6, 5);
        band.freeprop = true;
        let setup = band_pre_solver_setup_from_handoffs(
            &band,
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let solved = band_kspace_free_propagation_rel_solve_from_handoffs(&setup)?;

        assert_eq!(
            solved.structure_factors.structure_factors.dim(),
            (4, 6, 4, 4)
        );
        assert_eq!(solved.solved.eigenvalues.dim(), (4, 6, 4));
        assert_eq!(solved.solved.positive_counts.dim(), (4, 6));
        assert_eq!(solved.solved.band_energies.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn solves_band_kspace_spin_degenerate_grid_from_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_band_input(2, 1),
            &sample_two_spin_degenerate_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let solved = band_kspace_spin_degenerate_solve_from_handoffs(&setup)?;

        assert_eq!(
            solved.source_structure_factors.structure_factors.dim(),
            (61, 10, 8, 8)
        );
        assert_eq!(
            solved.source_structure_factors.point_solves[0]
                .kspace
                .taukinv
                .dim(),
            (8, 8)
        );
        assert_eq!(solved.structure_factors.dim(), (61, 10, 8, 8));
        assert_eq!(solved.t_matrices.dim(), (61, 8, 8));
        assert_eq!(solved.solved.eigenvalues.dim(), (61, 10, 8));
        assert_eq!(solved.solved.positive_counts.dim(), (61, 10));
        assert_eq!(solved.solved.band_energies.k_point_count(), 10);
        Ok(())
    }

    #[test]
    fn solves_band_kspace_free_propagation_spin_degenerate_grid_from_handoffs() -> Result<()> {
        let mut band = sample_sparse_band_input(6, 5);
        band.freeprop = true;
        let setup = band_pre_solver_setup_from_handoffs(
            &band,
            &sample_two_spin_degenerate_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let solved = band_kspace_free_propagation_spin_degenerate_solve_from_handoffs(&setup)?;

        assert_eq!(
            solved.source_structure_factors.structure_factors.dim(),
            (4, 6, 8, 8)
        );
        assert_eq!(
            solved.source_structure_factors.point_solves[0]
                .kspace
                .taukinv
                .dim(),
            (8, 8)
        );
        assert_eq!(solved.structure_factors.dim(), (4, 6, 8, 8));
        assert_eq!(solved.solved.eigenvalues.dim(), (4, 6, 8));
        assert_eq!(solved.solved.positive_counts.dim(), (4, 6));
        assert_eq!(solved.solved.band_energies.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn solves_band_kspace_spin_resolved_non_degenerate_grid_from_handoffs() -> Result<()> {
        let mut phase = sample_two_spin_degenerate_band_phase_bin();
        phase.reference_energy[(0, 1)] += Complex64::new(0.01, 0.0);
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &phase,
            &sample_reciprocal_cell(8),
        )?;
        let solved = band_kspace_spin_resolved_solve_from_handoffs(&setup)?;

        assert_eq!(
            solved.source_structure_factors.structure_factors.dim(),
            (4, 6, 8, 8)
        );
        assert_eq!(
            solved.source_structure_factors.point_solves[0]
                .kspace
                .taukinv
                .dim(),
            (8, 8)
        );
        assert_eq!(solved.structure_factors.dim(), (4, 6, 8, 8));
        assert_eq!(solved.t_matrices.dim(), (4, 8, 8));
        assert_eq!(solved.solved.eigenvalues.dim(), (4, 6, 8));
        assert_eq!(solved.solved.positive_counts.dim(), (4, 6));
        assert_eq!(solved.solved.band_energies.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn solves_band_kspace_free_propagation_spin_resolved_non_degenerate_grid_from_handoffs()
    -> Result<()> {
        let mut band = sample_sparse_band_input(6, 5);
        band.freeprop = true;
        let mut phase = sample_two_spin_degenerate_band_phase_bin();
        phase.reference_energy[(0, 1)] += Complex64::new(0.01, 0.0);
        let setup = band_pre_solver_setup_from_handoffs(&band, &phase, &sample_reciprocal_cell(8))?;
        let solved = band_kspace_free_propagation_spin_resolved_solve_from_handoffs(&setup)?;

        assert_eq!(
            solved.source_structure_factors.structure_factors.dim(),
            (4, 6, 8, 8)
        );
        assert_eq!(
            solved.source_structure_factors.point_solves[0]
                .kspace
                .taukinv
                .dim(),
            (8, 8)
        );
        assert_eq!(solved.structure_factors.dim(), (4, 6, 8, 8));
        assert_eq!(solved.solved.eigenvalues.dim(), (4, 6, 8));
        assert_eq!(solved.solved.positive_counts.dim(), (4, 6));
        assert_eq!(solved.solved.band_energies.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn spin_degenerate_solver_rejects_non_degenerate_spin_kspace_grid_from_handoffs() -> Result<()>
    {
        let mut phase = sample_two_spin_degenerate_band_phase_bin();
        phase.reference_energy[(0, 1)] += Complex64::new(0.01, 0.0);
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &phase,
            &sample_reciprocal_cell(8),
        )?;

        assert!(band_kspace_spin_degenerate_solve_from_handoffs(&setup).is_err());
        Ok(())
    }

    #[test]
    fn builds_bandstructure_dat_from_kspace_non_rel_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let data = bandstructure_dat_from_kspace_non_rel_handoffs(&setup)?;

        assert_eq!(data.k_point_count(), 6);
        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.rows[0].index, 1);
        let rendered = bandstructure_dat_string(&data)?;
        assert_eq!(parse_bandstructure_dat(&rendered)?.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn builds_bandstructure_dat_from_kspace_free_propagation_non_rel_handoffs() -> Result<()> {
        let mut band = sample_sparse_band_input(6, 5);
        band.freeprop = true;
        let setup = band_pre_solver_setup_from_handoffs(
            &band,
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let data = bandstructure_dat_from_kspace_free_propagation_non_rel_handoffs(&setup)?;

        assert_eq!(data.k_point_count(), 6);
        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.rows[0].index, 1);
        let rendered = bandstructure_dat_string(&data)?;
        assert_eq!(parse_bandstructure_dat(&rendered)?.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn builds_bandstructure_dat_from_kspace_rel_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let data = bandstructure_dat_from_kspace_rel_handoffs(&setup)?;

        assert_eq!(data.k_point_count(), 6);
        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.rows[0].index, 1);
        let rendered = bandstructure_dat_string(&data)?;
        assert_eq!(parse_bandstructure_dat(&rendered)?.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn builds_bandstructure_dat_from_kspace_free_propagation_rel_handoffs() -> Result<()> {
        let mut band = sample_sparse_band_input(6, 5);
        band.freeprop = true;
        let setup = band_pre_solver_setup_from_handoffs(
            &band,
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let data = bandstructure_dat_from_kspace_free_propagation_rel_handoffs(&setup)?;

        assert_eq!(data.k_point_count(), 6);
        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.rows[0].index, 1);
        let rendered = bandstructure_dat_string(&data)?;
        assert_eq!(parse_bandstructure_dat(&rendered)?.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn builds_bandstructure_dat_from_kspace_spin_degenerate_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_band_input(2, 1),
            &sample_two_spin_degenerate_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let data = bandstructure_dat_from_kspace_spin_degenerate_handoffs(&setup)?;

        assert_eq!(data.k_point_count(), 10);
        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.rows[0].index, 1);
        let rendered = bandstructure_dat_string(&data)?;
        assert_eq!(parse_bandstructure_dat(&rendered)?.k_point_count(), 10);
        Ok(())
    }

    #[test]
    fn builds_bandstructure_dat_from_kspace_spin_resolved_handoffs() -> Result<()> {
        let mut phase = sample_two_spin_degenerate_band_phase_bin();
        phase.reference_energy[(0, 1)] += Complex64::new(0.01, 0.0);
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &phase,
            &sample_reciprocal_cell(8),
        )?;
        let data = bandstructure_dat_from_kspace_spin_resolved_handoffs(&setup)?;

        assert_eq!(data.k_point_count(), 6);
        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.rows[0].index, 1);
        let rendered = bandstructure_dat_string(&data)?;
        assert_eq!(parse_bandstructure_dat(&rendered)?.k_point_count(), 6);
        Ok(())
    }

    #[test]
    fn builds_band_kspace_non_rel_point_input_from_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let tables = band_kspace_ewald_energy_tables_from_handoff(&setup, 0, 0)?;
        assert_eq!(tables.energy_dependent_terms.d1term3.len(), 3);
        assert_eq!(
            tables.direct_lattice_terms.radial_terms.shape()[0],
            BAND_KSPACE_J22MAX + 1
        );

        let input = band_kspace_non_rel_structure_factor_input(&setup, &tables, 0, 0, 0)?;
        assert_eq!(input.kspace.lattice_sum.lmax, 2);
        assert_eq!(input.kspace.angular_state_count, 4);
        assert_eq!(input.atom_count, 1);
        let solved = refeff_core::band_structure_factor_from_kspace_non_rel(input)
            .map_err(|source| invalid_band_input_error("structure_factor", source.to_string()))?;
        assert_eq!(solved.kspace.dllmmke.nrows(), 9);
        assert_eq!(solved.kspace.taukinv.dim(), (4, 4));
        assert_eq!(solved.structure_factor.dim(), (4, 4));
        Ok(())
    }

    #[test]
    fn builds_band_kspace_rel_point_input_from_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_sparse_band_input(6, 5),
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;
        let tables = band_kspace_ewald_energy_tables_from_handoff(&setup, 0, 0)?;

        let input = band_kspace_rel_structure_factor_input(&setup, &tables, 0, 0, 0)?;
        assert_eq!(input.kspace.lattice_sum.lmax, 2);
        assert_eq!(input.kspace.angular_state_count, 4);
        assert_eq!(input.kspace.site_offsets, &[0]);
        assert_eq!(input.kspace.site_state_counts, &[4]);
        assert_eq!(input.kspace.rel_component_counts.dim(), (2, 8));
        assert_eq!(input.kspace.rel_component_indices.dim(), (4, 2, 8));
        assert_eq!(input.atom_count, 1);

        let solved = refeff_core::band_structure_factor_from_kspace_rel(input)
            .map_err(|source| invalid_band_input_error("structure_factor", source.to_string()))?;
        assert_eq!(solved.kspace.dllmmke.nrows(), 9);
        assert_eq!(solved.kspace.taukinv.dim(), (4, 4));
        assert_eq!(solved.structure_factor.dim(), (4, 4));
        Ok(())
    }

    #[test]
    fn builds_band_pre_solver_setup_from_handoffs() -> Result<()> {
        let setup = band_pre_solver_setup_from_handoffs(
            &sample_band_input(6, 5),
            &sample_band_phase_bin(),
            &sample_reciprocal_cell(8),
        )?;

        assert_eq!(setup.search.energy_mesh.point_count(), 61);
        assert_eq!(
            setup.search.phase_interpolation.phase_shifts.dim(),
            (61, 1, 3, 1)
        );
        assert_eq!(setup.k_path.mesh.point_count(), 6);
        assert_eq!(setup.k_path.mesh.segment_point_counts, [2, 2, 2]);
        assert_eq!(setup.kspace_lattice.q_pairs.len(), 1);
        assert!(setup.kspace_lattice.direct_lattice.direct_indices.nrows() > 1);
        assert!(
            setup
                .kspace_lattice
                .reciprocal_lattice
                .reciprocal_indices
                .nrows()
                > 1
        );
        assert_eq!(setup.kspace_angular.angular_lmax, 1);
        assert_eq!(setup.kspace_angular.matrix_order_non_rel, 4);
        assert_eq!(setup.kspace_angular.angular_tables.gaunt_counts.len(), 10);
        assert_eq!(setup.kspace_solver_basis.matrix_order, 4);
        assert_eq!(setup.kspace_solver_basis.states.len(), 4);
        assert_eq!(setup.kspace_solver_basis.atoms.len(), 1);
        assert_eq!(setup.kspace_energy.energy_count, 61);
        assert_eq!(setup.kspace_energy.spin_count, 1);
        assert_eq!(setup.kspace_energy.j22max, BAND_KSPACE_J22MAX);
        Ok(())
    }

    #[test]
    fn band_pre_solver_setup_uses_fms_lmaxph_when_phase_bin_is_wider() -> Result<()> {
        let mut phase = sample_band_phase_bin();
        phase.potentials[0].lmax = 2;
        phase.potentials[0].phase_shifts = Array3::from_shape_fn(
            (phase.energy_count, 5, phase.spin_count),
            |(energy, l_slot, _)| Complex64::new(0.01 * energy as f64 + 0.1 * l_slot as f64, 0.0),
        );

        let setup = band_pre_solver_setup_from_handoffs_with_lmaxph(
            &sample_band_input(6, 5),
            &phase,
            &sample_reciprocal_cell(8),
            &[1],
        )?;

        assert_eq!(setup.search.phase_handoff.signed_angular_offset, 2);
        assert_eq!(
            setup.search.phase_interpolation.phase_shifts.dim(),
            (61, 1, 5, 1)
        );
        assert_eq!(setup.kspace_angular.angular_lmax, 1);
        assert_eq!(setup.kspace_angular.matrix_order_non_rel, 4);
        assert_eq!(setup.kspace_solver_basis.matrix_order, 4);
        assert_eq!(
            setup.search.phase_interpolation.phase_shifts[(0, 0, 0, 0)],
            Complex32::new(0.0, 0.0)
        );
        assert_eq!(
            setup.search.phase_interpolation.phase_shifts[(0, 0, 4, 0)],
            Complex32::new(0.0, 0.0)
        );
        Ok(())
    }

    #[test]
    fn band_k_path_setup_rejects_invalid_point_count() {
        assert!(matches!(
            band_k_path_setup_from_handoffs(&sample_band_input(1, 5), &sample_reciprocal_cell(8)),
            Err(IoError::Parse { path, .. }) if path == Path::new(BAND_INP_PATH)
        ));
    }

    #[test]
    fn rejects_bad_kmesh_dat() {
        assert!(parse_kmesh_dat("1 0.0 0.0\n").is_err());
        assert!(parse_kmesh_dat("").is_err());
    }

    fn sample_band_input(nkp: i32, ikpath: i32) -> BandInput {
        BandInput {
            mband: 1,
            energy_mesh: crate::BandEnergyMesh {
                emin: -5.0,
                emax: 10.0,
                estep: 0.25,
            },
            nkp,
            ikpath,
            freeprop: false,
        }
    }

    fn sample_sparse_band_input(nkp: i32, ikpath: i32) -> BandInput {
        let mut input = sample_band_input(nkp, ikpath);
        input.energy_mesh.estep = 5.0;
        input
    }

    fn sample_band_phase_bin() -> PhaseBinData {
        let spin_count = 1;
        let energy_count = 4;
        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 4,
            fermi_index: 1,
            pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
            final_state_count: 1,
            transition_count: 1,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.0,
                fermi_level: 0.0,
                edge_energy: 0.0,
            },
            energy_grid: array![
                Complex64::new(-5.0 / FEFF_HARTREE_EV, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(5.0 / FEFF_HARTREE_EV, 0.0),
                Complex64::new(10.0 / FEFF_HARTREE_EV, 0.0)
            ],
            reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, _)| {
                Complex64::new(0.01 * energy as f64, 0.0)
            }),
            potentials: vec![PhaseBinPotential {
                lmax: 1,
                atomic_number: 29,
                label: "Cu".to_string(),
                phase_shifts: Array3::from_shape_fn(
                    (energy_count, 3, spin_count),
                    |(energy, l_slot, _)| {
                        Complex64::new(0.01 * energy as f64 + 0.1 * l_slot as f64, 0.0)
                    },
                ),
            }],
            transition_moments: Array4::zeros((energy_count, 1, 1, spin_count)),
            raw_pads: None,
        }
    }

    fn sample_two_spin_degenerate_band_phase_bin() -> PhaseBinData {
        let spin_count = 2;
        let energy_count = 4;
        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 4,
            fermi_index: 1,
            pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
            final_state_count: 1,
            transition_count: 1,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.0,
                fermi_level: 0.0,
                edge_energy: 0.0,
            },
            energy_grid: array![
                Complex64::new(-0.2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(0.4, 0.0)
            ],
            reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, _)| {
                Complex64::new(-0.1 + 0.01 * energy as f64, 0.0)
            }),
            potentials: vec![PhaseBinPotential {
                lmax: 1,
                atomic_number: 29,
                label: "Cu".to_string(),
                phase_shifts: Array3::from_shape_fn(
                    (energy_count, 3, spin_count),
                    |(energy, l_slot, _)| {
                        Complex64::new(0.01 * energy as f64 + 0.1 * l_slot as f64, 0.0)
                    },
                ),
            }],
            transition_moments: Array4::zeros((energy_count, 1, 1, spin_count)),
            raw_pads: None,
        }
    }

    fn sample_reciprocal_cell(total_kpoints: i32) -> ReciprocalCell {
        ReciprocalCell {
            lattice_vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            volume_scale: -1.0,
            imaginary_energy: 0.0,
            core_hole_strength: 1.0,
            lattice_name: "P".to_string(),
            space_group_hm: "Pm-3m".to_string(),
            space_group: 221,
            atom_count: 1,
            absorber: 1,
            core_hole: 1,
            k_mesh: crate::control_input::ReciprocalKMesh {
                total: total_kpoints,
                x: total_kpoints,
                y: 0,
                z: 0,
                kind: 3,
                use_symmetry: false,
            },
            positions: vec![[0.0, 0.0, 0.0]],
            potentials: vec![0],
            labels: vec!["Cu".to_string()],
            stretch: [0.0, 0.0, 0.0],
        }
    }

    fn identity_basis() -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 5.0e-4,
            "actual={actual:?} expected={expected:?}"
        );
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert_close(actual.re, expected.re);
        assert_close(actual.im, expected.im);
    }

    const SAMPLE_BANDSTRUCTURE: &str = concat!(
        " # grid of            2  k-points.\n",
        " # grid of            4  energy points  emin=   -5.0000000000000000       , emax=    10.000000000000000       , estep=   0.25000000000000000\n",
        " # Found between            1  and            2  number of bands.\n",
        "    1   0.0000   0.5000   0.2500    2  -5.0000   1.2500\n",
        "    2   0.5000   0.2500   0.0000    1   0.7500\n",
    );

    const SAMPLE_KMESH: &str = concat!(
        "         1   0.0000   0.5000   0.2500   0.7500    100      2      4      5      6\n",
        "         2   0.5000   0.2500   0.0000   0.2500\n",
    );
}
