//! FEFF BAND numerical setup helpers.
//!
//! The routines here cover deterministic `BAND/bandtot.f90` setup and
//! KKR/raw-`G` eigenvalue-counting post-processing for `bandstructure.dat`.

use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, Axis,
    ShapeBuilder,
};
use num_complex::Complex32;
use refeff_linalg::{
    LinalgError, complex_matmul, complex32_general_eigenvalues, complex32_lu_factor,
    complex32_lu_solve,
};
use thiserror::Error;

use crate::{
    Complex, FEFF_HARTREE_EV, Real,
    angular::{BasisTransformMatrices, SpinOrbitCouplingTables},
    fms::{FmsAtom, FmsError, FmsTMatrixInput, fms_t_matrix_element},
    interpolation::{InterpolationError, terpc},
    kspace::{
        KSpaceError, KSpaceStrsetMatrices, KSpaceStrsetNonRelFromLatticeSumInput,
        KSpaceStrsetRelFromLatticeSumInput, kspace_strset_non_rel_from_lattice_sum,
        kspace_strset_rel_from_lattice_sum,
    },
    state::StateKet,
};

/// Inputs for FEFF `BAND/bandtot.f90` energy-search mesh setup.
#[derive(Debug, Clone, Copy)]
pub struct BandEnergySearchMeshInput<'a> {
    /// Requested minimum energy from `band.inp`, in eV.
    pub requested_min_ev: Real,
    /// Requested maximum energy from `band.inp`, in eV.
    pub requested_max_ev: Real,
    /// Requested energy step from `band.inp`, in eV.
    pub requested_step_ev: Real,
    /// FEFF XSPH complex energy mesh `em`.
    pub phase_energies_hartree: ArrayView1<'a, Complex>,
    /// FEFF `ne1`, the active real-energy prefix available for interpolation.
    pub phase_active_len: usize,
    /// FEFF Fermi level `xmu`, in Hartree.
    pub fermi_level_hartree: Real,
}

/// FEFF BAND energy-search mesh after clipping to phase-shift coverage.
#[derive(Debug, Clone, PartialEq)]
pub struct BandEnergySearchMesh {
    /// Lower energy used by the BAND search, in Hartree.
    pub min_hartree: Real,
    /// Upper energy used by the BAND search, in Hartree.
    pub max_hartree: Real,
    /// Recomputed uniform step, in Hartree.
    pub step_hartree: Real,
    /// Uniform search energies, in Hartree.
    pub energies_hartree: Array1<Real>,
}

impl BandEnergySearchMesh {
    /// Number of FEFF BAND search energy points, `nep`.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energies_hartree.len()
    }
}

/// Inputs for FEFF `bandtot.f90` phase/reference interpolation on the BAND mesh.
#[derive(Debug, Clone, Copy)]
pub struct BandPhaseSearchInterpolationInput<'a> {
    /// BAND search energies, in Hartree.
    pub search_energies_hartree: ArrayView1<'a, Real>,
    /// Source XSPH energy abscissae, in Hartree.
    pub source_energies_hartree: ArrayView1<'a, Real>,
    /// FEFF `eref(source_energy,spin)`.
    pub source_reference_energies_hartree: ArrayView2<'a, Complex>,
    /// FEFF `ph(source_energy,signed_l,spin,potential)` on a shared signed-`l` axis.
    pub source_phase_shifts: ArrayView4<'a, Complex>,
    /// Inclusive active `lmax` for each potential in `source_phase_shifts`.
    pub potential_lmax: &'a [usize],
    /// FEFF polynomial interpolation order. BAND uses cubic order `3`.
    pub interpolation_order: usize,
}

/// FEFF BAND phase/reference tables interpolated onto the search mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct BandPhaseSearchInterpolation {
    /// Offset mapping `signed_l` to `signed_l + signed_l_offset`.
    pub signed_l_offset: usize,
    /// Interpolated reference energies as `(search_energy,spin)`.
    pub reference_energies_hartree: Array2<Complex>,
    /// Complex momenta `sqrt(2*(E-eref))` as `(search_energy,spin)`.
    pub wave_numbers: Array2<Complex32>,
    /// Interpolated phase shifts as `(search_energy,spin,signed_l,potential)`.
    pub phase_shifts: Array4<Complex32>,
}

/// Inputs for FEFF `bandtot.f90` band-energy identification.
#[derive(Debug, Clone, Copy)]
pub struct BandEnergiesFromPositiveCountsInput<'a> {
    /// FEFF `n_pos(ie,ik)`: positive eigenvalue counts as `(energy, kpoint)`.
    pub positive_counts: ArrayView2<'a, usize>,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// Variable-length band-energy rows identified at each k-point.
#[derive(Debug, Clone, PartialEq)]
pub struct BandEnergiesFromPositiveCounts {
    /// Band energies as one row per k-point, in Hartree.
    pub band_energies_hartree: Vec<Array1<Real>>,
}

impl BandEnergiesFromPositiveCounts {
    /// Number of k-point rows.
    #[must_use]
    pub fn k_point_count(&self) -> usize {
        self.band_energies_hartree.len()
    }

    /// Minimum number of bands found on any k-point row.
    #[must_use]
    pub fn min_band_count(&self) -> usize {
        self.band_energies_hartree
            .iter()
            .map(Array1::len)
            .min()
            .unwrap_or(0)
    }

    /// Maximum number of bands found on any k-point row.
    #[must_use]
    pub fn max_band_count(&self) -> usize {
        self.band_energies_hartree
            .iter()
            .map(Array1::len)
            .max()
            .unwrap_or(0)
    }
}

/// Inputs for FEFF `BAND/fmsband.f90` full lattice T-matrix assembly.
#[derive(Debug, Clone)]
pub struct BandLatticeTMatrixInput<'a> {
    /// FEFF BAND state kets in matrix order.
    pub states: &'a [StateKet],
    /// BAND cluster atoms addressed by one-based [`StateKet::atom`] values.
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

/// Inputs for FEFF `BAND/fmsband.f90` T-matrix assembly over the search mesh.
#[derive(Debug, Clone)]
pub struct BandLatticeTMatrixGridInput<'a> {
    /// FEFF BAND state kets in matrix order.
    pub states: &'a [StateKet],
    /// BAND cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// FEFF `xphase(energy,spin,l,potential)` tables with signed `l` centered.
    pub phase_shifts: ArrayView4<'a, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
}

/// Inputs for FEFF `BAND/kkrband.f90` KKR work-matrix setup.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrMatrixInput<'a> {
    /// FEFF `structurefactor` matrix `G` after applying FEFF sign convention.
    pub structure_factor: ArrayView2<'a, Complex32>,
    /// Full FEFF BAND lattice T-matrix `Tmat`.
    pub t_matrix: ArrayView2<'a, Complex32>,
}

/// Inputs for FEFF `BAND/fmsband.f90` KKR eigenvalue extraction.
#[derive(Debug, Clone, Copy)]
pub struct BandSortedKkrEigenvaluesInput<'a> {
    /// FEFF `kkrband` matrix `Gfms`.
    pub kkr_matrix: ArrayView2<'a, Complex32>,
    /// Complex momentum `p`; FEFF diagonalizes `Gfms * p`.
    pub wave_number: Complex32,
}

/// Inputs for one FEFF `fmsband -> kkrband -> cgees` solve.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrEigenvaluesFromStructureFactorInput<'a> {
    /// FEFF `structurefactor` matrix `G`.
    pub structure_factor: ArrayView2<'a, Complex32>,
    /// Full FEFF BAND lattice T-matrix `Tmat`.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// Complex momentum `p`; FEFF diagonalizes `(G - T^-1) * p`.
    pub wave_number: Complex32,
}

/// Inputs for FEFF `kkrband.f90`'s `freeprop` branch.
#[derive(Debug, Clone, Copy)]
pub struct BandFreePropagationEigenvaluesFromStructureFactorInput<'a> {
    /// FEFF `structurefactor` matrix `G`.
    pub structure_factor: ArrayView2<'a, Complex32>,
    /// Complex momentum `p`; FEFF diagonalizes `G * p` in `freeprop`.
    pub wave_number: Complex32,
}

/// Inputs for FEFF `bandtot.f90` eigenvalue solves over `(energy,kpoint)`.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrEigenvalueGridInput<'a> {
    /// Structure-factor matrices as `(energy, kpoint, row, column)`.
    pub structure_factors: ArrayView4<'a, Complex32>,
    /// Per-energy full FEFF BAND lattice T-matrices as `(energy, row, column)`.
    pub t_matrices: ArrayView3<'a, Complex32>,
    /// Per-energy complex momenta `p`.
    pub wave_numbers: ArrayView1<'a, Complex32>,
}

/// Inputs for FEFF `freeprop` eigenvalue solves over `(energy,kpoint)`.
#[derive(Debug, Clone, Copy)]
pub struct BandFreePropagationEigenvalueGridInput<'a> {
    /// Structure-factor matrices as `(energy, kpoint, row, column)`.
    pub structure_factors: ArrayView4<'a, Complex32>,
    /// Per-energy complex momenta `p`.
    pub wave_numbers: ArrayView1<'a, Complex32>,
}

/// FEFF `bandtot` eigenvalue and positive-count tables.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKkrEigenvalueGrid {
    /// Sorted KKR eigenvalues as `(energy, kpoint, eigenvalue)`.
    pub eigenvalues: Array3<Complex32>,
    /// FEFF `n_pos(energy,kpoint)` positive-real-eigenvalue count table.
    pub positive_counts: Array2<usize>,
}

/// Inputs for FEFF BAND KKR solves through final band-energy identification.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrBandEnergiesInput<'a> {
    /// Structure-factor matrices as `(energy, kpoint, row, column)`.
    pub structure_factors: ArrayView4<'a, Complex32>,
    /// Per-energy full FEFF BAND lattice T-matrices as `(energy, row, column)`.
    pub t_matrices: ArrayView3<'a, Complex32>,
    /// Per-energy complex momenta `p`.
    pub wave_numbers: ArrayView1<'a, Complex32>,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// Inputs for FEFF `freeprop` grid solves through final band-energy rows.
#[derive(Debug, Clone, Copy)]
pub struct BandFreePropagationBandEnergiesInput<'a> {
    /// Structure-factor matrices as `(energy, kpoint, row, column)`.
    pub structure_factors: ArrayView4<'a, Complex32>,
    /// Per-energy complex momenta `p`.
    pub wave_numbers: ArrayView1<'a, Complex32>,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// Inputs for ordinary BAND solve composition after `G(energy,kpoint)` is known.
#[derive(Debug, Clone)]
pub struct BandKkrBandEnergiesFromPhaseStructureGridInput<'a> {
    /// Structure-factor matrices as `(energy, kpoint, row, column)`.
    pub structure_factors: ArrayView4<'a, Complex32>,
    /// FEFF BAND state kets in matrix order.
    pub states: &'a [StateKet],
    /// BAND cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'a [FmsAtom],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// FEFF `xphase(energy,spin,l,potential)` tables with signed `l` centered.
    pub phase_shifts: ArrayView4<'a, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'a SpinOrbitCouplingTables,
    /// Per-energy complex momenta `p`.
    pub wave_numbers: ArrayView1<'a, Complex32>,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// Inputs for ordinary non-relativistic KSPACE-backed BAND solve composition.
#[derive(Debug, Clone)]
pub struct BandKkrBandEnergiesFromKspacePhaseNonRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order structure-factor inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandStructureFactorFromKspaceNonRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
    /// FEFF BAND state kets in matrix order.
    pub states: &'data [StateKet],
    /// BAND cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'data [FmsAtom],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// FEFF `xphase(energy,spin,l,potential)` tables with signed `l` centered.
    pub phase_shifts: ArrayView4<'data, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'data SpinOrbitCouplingTables,
    /// Per-energy complex momenta `p`.
    pub wave_numbers: ArrayView1<'data, Complex32>,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// Inputs for ordinary relativistic KSPACE-backed BAND solve composition.
#[derive(Debug, Clone)]
pub struct BandKkrBandEnergiesFromKspacePhaseRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order structure-factor inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandStructureFactorFromKspaceRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
    /// FEFF BAND state kets in matrix order.
    pub states: &'data [StateKet],
    /// BAND cluster atoms addressed by one-based [`StateKet::atom`] values.
    pub atoms: &'data [FmsAtom],
    /// FEFF `nsp`: one or two spin channels.
    pub spin_channels: usize,
    /// FEFF `ispin` selector used by the one-spin spin-orbit branch.
    pub spin_selector: i32,
    /// FEFF `xphase(energy,spin,l,potential)` tables with signed `l` centered.
    pub phase_shifts: ArrayView4<'data, Complex32>,
    /// FEFF `t3jp`/`t3jm` spin-orbit coupling coefficients.
    pub spin_orbit: &'data SpinOrbitCouplingTables,
    /// Per-energy complex momenta `p`.
    pub wave_numbers: ArrayView1<'data, Complex32>,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// FEFF BAND solved KKR grid plus identified band energies.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKkrBandEnergies {
    /// Sorted KKR eigenvalues as `(energy, kpoint, eigenvalue)`.
    pub eigenvalues: Array3<Complex32>,
    /// FEFF `n_pos(energy,kpoint)` positive-real-eigenvalue count table.
    pub positive_counts: Array2<usize>,
    /// Variable-length band-energy rows identified at each k-point.
    pub band_energies: BandEnergiesFromPositiveCounts,
}

/// FEFF BAND final rows plus the per-energy lattice T-matrices used to solve them.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKkrBandEnergiesFromPhaseStructureGrid {
    /// Per-energy full FEFF BAND lattice T-matrices as `(energy, row, column)`.
    pub t_matrices: Array3<Complex32>,
    /// Solved KKR grid and final variable-length band-energy rows.
    pub solved: BandKkrBandEnergies,
}

/// FEFF BAND final rows from KSPACE structure factors and interpolated phases.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKkrBandEnergiesFromKspacePhaseGrid {
    /// KSPACE-backed FEFF-basis structure-factor grid and point diagnostics.
    pub structure_factors: BandStructureFactorFromKspaceGrid,
    /// Per-energy T-matrices and solved final band rows.
    pub solved: BandKkrBandEnergiesFromPhaseStructureGrid,
}

/// Inputs for the FEFF `KSPACE/structurefactor.f90` FEFF-basis tail.
#[derive(Debug, Clone, Copy)]
pub struct BandStructureFactorFeffBasisInput<'a> {
    /// SPRKKR-basis `tauk` matrix produced by `strset`.
    pub tauk_sprkkr: ArrayView2<'a, Complex>,
    /// Complex momentum `p` for this energy.
    pub wave_number: Complex,
    /// Number of lattice atoms, FEFF `nats`.
    pub atom_count: usize,
    /// Maximum angular momentum, FEFF `maxl`.
    pub angular_lmax: usize,
    /// FEFF `BASTRMAT` basis-transform bundle.
    pub basis_transforms: &'a BasisTransformMatrices,
}

/// Inputs for FEFF-basis conversion over a search-energy/k-point `tauk` grid.
#[derive(Debug, Clone, Copy)]
pub struct BandStructureFactorFeffBasisGridInput<'a> {
    /// SPRKKR-basis `tauk` matrices as `(energy, kpoint, row, column)`.
    pub tauk_sprkkr: ArrayView4<'a, Complex>,
    /// Per-energy complex momenta `p`.
    pub wave_numbers: ArrayView1<'a, Complex>,
    /// Number of lattice atoms, FEFF `nats`.
    pub atom_count: usize,
    /// Maximum angular momentum, FEFF `maxl`.
    pub angular_lmax: usize,
    /// FEFF `BASTRMAT` basis-transform bundle.
    pub basis_transforms: &'a BasisTransformMatrices,
}

/// Inputs for composing non-relativistic KSPACE `STRBBDD -> STRSET` into FEFF-basis `G`.
#[derive(Debug, Clone, Copy)]
pub struct BandStructureFactorFromKspaceNonRelInput<'a> {
    /// Full KSPACE lattice-sum and non-relativistic `STRSET` input.
    pub kspace: KSpaceStrsetNonRelFromLatticeSumInput<'a>,
    /// Number of lattice atoms, FEFF `nats`.
    pub atom_count: usize,
    /// Maximum angular momentum, FEFF `maxl`.
    pub angular_lmax: usize,
    /// FEFF `BASTRMAT` basis-transform bundle.
    pub basis_transforms: &'a BasisTransformMatrices,
}

/// Inputs for composing relativistic KSPACE `STRBBDD -> STRSET` into FEFF-basis `G`.
#[derive(Debug, Clone, Copy)]
pub struct BandStructureFactorFromKspaceRelInput<'a> {
    /// Full KSPACE lattice-sum and relativistic `STRSET` input.
    pub kspace: KSpaceStrsetRelFromLatticeSumInput<'a>,
    /// Number of lattice atoms, FEFF `nats`.
    pub atom_count: usize,
    /// Maximum angular momentum, FEFF `maxl`.
    pub angular_lmax: usize,
    /// FEFF `BASTRMAT` basis-transform bundle.
    pub basis_transforms: &'a BasisTransformMatrices,
}

/// Completed FEFF BAND structure-factor composition for one energy/k-point.
#[derive(Debug, Clone, PartialEq)]
pub struct BandStructureFactorFromKspace {
    /// Intermediate KSPACE `DLLMMKE` and SPRKKR-basis `TAUKINV` matrices.
    pub kspace: KSpaceStrsetMatrices,
    /// FEFF-basis single-complex structure-factor matrix `G`.
    pub structure_factor: Array2<Complex32>,
}

/// Inputs for non-relativistic KSPACE `STRBBDD -> STRSET -> G` grid assembly.
#[derive(Debug, Clone, Copy)]
pub struct BandStructureFactorFromKspaceNonRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandStructureFactorFromKspaceNonRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
}

/// Inputs for relativistic KSPACE `STRBBDD -> STRSET -> G` grid assembly.
#[derive(Debug, Clone, Copy)]
pub struct BandStructureFactorFromKspaceRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandStructureFactorFromKspaceRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
}

/// Completed KSPACE-backed FEFF-basis structure-factor grid.
#[derive(Debug, Clone, PartialEq)]
pub struct BandStructureFactorFromKspaceGrid {
    /// One-point structure-factor assemblies in flat FEFF loop order.
    pub point_solves: Vec<BandStructureFactorFromKspace>,
    /// FEFF-basis structure-factor matrices as `(energy, kpoint, row, column)`.
    pub structure_factors: Array4<Complex32>,
}

/// Inputs for a one-point non-relativistic KSPACE-to-KKR BAND solve.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrFromKspaceNonRelInput<'a> {
    /// KSPACE-to-FEFF-basis structure-factor input.
    pub structure_factor: BandStructureFactorFromKspaceNonRelInput<'a>,
    /// Full FEFF BAND lattice T-matrix `Tmat`.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// Complex momentum `p`; FEFF diagonalizes `(G - T^-1) * p`.
    pub wave_number: Complex32,
}

/// Inputs for a one-point relativistic KSPACE-to-KKR BAND solve.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrFromKspaceRelInput<'a> {
    /// KSPACE-to-FEFF-basis structure-factor input.
    pub structure_factor: BandStructureFactorFromKspaceRelInput<'a>,
    /// Full FEFF BAND lattice T-matrix `Tmat`.
    pub t_matrix: ArrayView2<'a, Complex32>,
    /// Complex momentum `p`; FEFF diagonalizes `(G - T^-1) * p`.
    pub wave_number: Complex32,
}

/// Inputs for one FEFF `freeprop` non-relativistic KSPACE solve.
#[derive(Debug, Clone, Copy)]
pub struct BandFreePropagationFromKspaceNonRelInput<'a> {
    /// KSPACE-to-FEFF-basis structure-factor input.
    pub structure_factor: BandStructureFactorFromKspaceNonRelInput<'a>,
    /// Complex momentum `p`; FEFF diagonalizes `G * p`.
    pub wave_number: Complex32,
}

/// Inputs for one FEFF `freeprop` relativistic KSPACE solve.
#[derive(Debug, Clone, Copy)]
pub struct BandFreePropagationFromKspaceRelInput<'a> {
    /// KSPACE-to-FEFF-basis structure-factor input.
    pub structure_factor: BandStructureFactorFromKspaceRelInput<'a>,
    /// Complex momentum `p`; FEFF diagonalizes `G * p`.
    pub wave_number: Complex32,
}

/// Completed one-point FEFF BAND KSPACE-to-KKR solve.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKkrFromKspace {
    /// KSPACE intermediates and FEFF-basis structure factor.
    pub structure_factor: BandStructureFactorFromKspace,
    /// FEFF `kkrband.f90` work matrix: `G - T^-1`, or raw `G` in `freeprop`.
    pub kkr_matrix: Array2<Complex32>,
    /// Sorted FEFF KKR eigenvalues.
    pub eigenvalues: Array1<Complex32>,
    /// FEFF positive-real-eigenvalue count for this energy/k-point.
    pub positive_count: usize,
}

/// Inputs for KSPACE-backed FEFF BAND KKR solves over `(energy,kpoint)`.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrFromKspaceNonRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandKkrFromKspaceNonRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
}

/// Inputs for relativistic KSPACE-backed FEFF BAND KKR grid solves.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrFromKspaceRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandKkrFromKspaceRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
}

/// Inputs for non-relativistic KSPACE-backed FEFF `freeprop` grid solves.
#[derive(Debug, Clone, Copy)]
pub struct BandFreePropagationFromKspaceNonRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandFreePropagationFromKspaceNonRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
}

/// Inputs for relativistic KSPACE-backed FEFF `freeprop` grid solves.
#[derive(Debug, Clone, Copy)]
pub struct BandFreePropagationFromKspaceRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandFreePropagationFromKspaceRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
}

/// Completed KSPACE-backed FEFF BAND KKR grid solve.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKkrFromKspaceGrid {
    /// One-point solves in flat FEFF loop order.
    pub point_solves: Vec<BandKkrFromKspace>,
    /// Sorted KKR eigenvalues as `(energy, kpoint, eigenvalue)`.
    pub eigenvalues: Array3<Complex32>,
    /// FEFF `n_pos(energy,kpoint)` positive-real-eigenvalue count table.
    pub positive_counts: Array2<usize>,
}

/// Inputs for non-relativistic KSPACE-backed BAND grid solves through final rows.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrBandEnergiesFromKspaceNonRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandKkrFromKspaceNonRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// Inputs for relativistic KSPACE-backed BAND grid solves through final rows.
#[derive(Debug, Clone, Copy)]
pub struct BandKkrBandEnergiesFromKspaceRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandKkrFromKspaceRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// Inputs for non-relativistic KSPACE-backed `freeprop` solves through final rows.
#[derive(Debug, Clone, Copy)]
pub struct BandFreePropagationBandEnergiesFromKspaceNonRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandFreePropagationFromKspaceNonRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// Inputs for relativistic KSPACE-backed `freeprop` solves through final rows.
#[derive(Debug, Clone, Copy)]
pub struct BandFreePropagationBandEnergiesFromKspaceRelGridInput<'points, 'data> {
    /// Flat FEFF loop-order point inputs: energy-major, then k-point.
    pub point_inputs: &'points [BandFreePropagationFromKspaceRelInput<'data>],
    /// Number of BAND search energies.
    pub energy_count: usize,
    /// Number of sampled k-points.
    pub k_point_count: usize,
    /// BAND search mesh lower energy, in Hartree.
    pub energy_min_hartree: Real,
    /// BAND search mesh step, in Hartree.
    pub energy_step_hartree: Real,
}

/// Completed KSPACE-backed FEFF BAND grid solve plus final band-energy rows.
#[derive(Debug, Clone, PartialEq)]
pub struct BandKkrBandEnergiesFromKspaceGrid {
    /// One-point solves in flat FEFF loop order.
    pub point_solves: Vec<BandKkrFromKspace>,
    /// Sorted KKR eigenvalues as `(energy, kpoint, eigenvalue)`.
    pub eigenvalues: Array3<Complex32>,
    /// FEFF `n_pos(energy,kpoint)` positive-real-eigenvalue count table.
    pub positive_counts: Array2<usize>,
    /// Variable-length band-energy rows identified at each k-point.
    pub band_energies: BandEnergiesFromPositiveCounts,
}

/// Inputs for FEFF `bandtot.f90` positive-eigenvalue counting.
#[derive(Debug, Clone, Copy)]
pub struct BandPositiveCountsFromEigenvaluesInput<'a> {
    /// FEFF `eigen(ie,ik,state)` eigenvalue cube.
    pub eigenvalues: ArrayView3<'a, Complex32>,
}

/// Errors returned by FEFF BAND helpers.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum BandError {
    /// Scalar BAND setup inputs must be finite.
    #[error("{name}[{index}] must be finite, got {value}")]
    NonFiniteValue {
        name: &'static str,
        index: usize,
        value: Real,
    },
    /// FEFF `bandtot` needs a positive requested energy step.
    #[error("BAND requested energy step must be positive, got {step_ev}")]
    InvalidRequestedEnergyStep { step_ev: Real },
    /// FEFF `bandtot` reads `em(1)` and `em(ne1)`.
    #[error("BAND phase active length must be in 1..={available}, got {active_len}")]
    InvalidPhaseActiveLength { active_len: usize, available: usize },
    /// The requested range and phase range have no valid overlap.
    #[error("BAND clipped energy range is invalid: min={min_hartree}, max={max_hartree}")]
    InvalidClippedEnergyRange {
        min_hartree: Real,
        max_hartree: Real,
    },
    /// FEFF-compatible BAND energy point counts must fit addressable memory.
    #[error("BAND energy point count overflowed")]
    EnergyPointCountOverflow,
    /// FEFF `bandtot` needs at least two search energy points.
    #[error("BAND energy point count must be at least 2, got {point_count}")]
    InvalidEnergyPointCount { point_count: usize },
    /// Positive-eigenvalue count tables are `(energy, kpoint)` and need data.
    #[error(
        "BAND positive-count table must have shape (energy >= 2, kpoint >= 1), got ({rows}, {columns})"
    )]
    InvalidPositiveCountShape { rows: usize, columns: usize },
    /// BAND energy identification requires a finite, positive search step.
    #[error("BAND search energy step must be finite and positive, got {step_hartree}")]
    InvalidSearchEnergyStep { step_hartree: Real },
    /// FEFF positive-eigenvalue counting needs at least one eigenvalue.
    #[error("BAND eigenvalue list must not be empty")]
    EmptyEigenvalueList,
    /// FEFF `eigen(energy,kpoint,state)` tables need all three dimensions.
    #[error(
        "BAND eigenvalue cube must have shape (energy >= 1, kpoint >= 1, eigenvalue >= 1), got ({energy_count}, {k_point_count}, {eigenvalue_count})"
    )]
    InvalidEigenvalueCubeShape {
        energy_count: usize,
        k_point_count: usize,
        eigenvalue_count: usize,
    },
    /// FEFF BAND matrix helpers need at least one state.
    #[error("BAND state list must not be empty")]
    EmptyStateList,
    /// FEFF structure-factor helpers need at least one lattice atom.
    #[error("BAND lattice atom list must not be empty")]
    EmptyAtomList,
    /// FEFF state-ket atom indices must be one-based and address the atom table.
    #[error("state atom {atom} is outside one-based atom table length {atom_count}")]
    StateAtomOutOfRange { atom: usize, atom_count: usize },
    /// FEFF `structurefactor` matrix dimensions must match `nats*(maxl+1)^2`.
    #[error("BAND structure-factor matrix must be {expected}x{expected}, got {rows}x{columns}")]
    InvalidStructureFactorShape {
        rows: usize,
        columns: usize,
        expected: usize,
    },
    /// FEFF `BASTRMAT` transforms must match the requested angular cutoff.
    #[error(
        "BAND basis transform for lmax={lmax} must have order {expected_order}, got lmax={actual_lmax}, order={actual_order}, shape={rows}x{columns}"
    )]
    InvalidBasisTransform {
        lmax: usize,
        expected_order: usize,
        actual_lmax: usize,
        actual_order: usize,
        rows: usize,
        columns: usize,
    },
    /// FEFF `structurefactor` divides by complex momentum `p`.
    #[error("BAND structure-factor wave number must be nonzero")]
    ZeroWaveNumber,
    /// Complex BAND scalar inputs must be finite.
    #[error("{name} must be finite, got ({real}, {imaginary})")]
    NonFiniteComplexValue {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// Matrix order arithmetic overflowed.
    #[error("BAND matrix order overflowed")]
    MatrixOrderOverflow,
    /// FEFF BAND matrices are square.
    #[error("{name} must be square, got {rows}x{columns}")]
    NonSquareMatrix {
        name: &'static str,
        rows: usize,
        columns: usize,
    },
    /// FEFF `kkrband` subtracts two matrices with identical shape.
    #[error(
        "BAND matrix shape mismatch: {left_name} is {left_rows}x{left_columns}, {right_name} is {right_rows}x{right_columns}"
    )]
    MatrixShapeMismatch {
        left_name: &'static str,
        left_rows: usize,
        left_columns: usize,
        right_name: &'static str,
        right_rows: usize,
        right_columns: usize,
    },
    /// FEFF BAND grid solves need `(energy >= 1, kpoint >= 1, state >= 1, state)`.
    #[error(
        "BAND KKR grid must have shape (energy >= 1, kpoint >= 1, state >= 1, state), got ({energy_count}, {k_point_count}, {rows}, {columns})"
    )]
    InvalidKkrEigenvalueGridShape {
        energy_count: usize,
        k_point_count: usize,
        rows: usize,
        columns: usize,
    },
    /// FEFF `tauk` grids need non-empty square matrices matching `nats*(maxl+1)^2`.
    #[error(
        "BAND structure-factor grid must have shape (energy >= 1, kpoint >= 1, {expected}x{expected}), got ({energy_count}, {k_point_count}, {rows}, {columns})"
    )]
    InvalidStructureFactorGridShape {
        energy_count: usize,
        k_point_count: usize,
        rows: usize,
        columns: usize,
        expected: usize,
    },
    /// FEFF BAND per-energy tables must match the structure-factor energy axis.
    #[error("{name} length must be {expected}, got {actual}")]
    InvalidBandTableLength {
        name: &'static str,
        actual: usize,
        expected: usize,
    },
    /// FEFF BAND phase interpolation needs non-empty matching source tables.
    #[error(
        "BAND phase-search table must have shape (source_energy >= 1, signed_l odd >= 1, spin >= 1, potential >= 1), got ({source_energy_count}, {signed_l_count}, {spin_count}, {potential_count})"
    )]
    InvalidPhaseSearchShape {
        source_energy_count: usize,
        signed_l_count: usize,
        spin_count: usize,
        potential_count: usize,
    },
    /// A potential requested signed-`l` slots outside the shared phase table.
    #[error(
        "BAND potential {potential} lmax {lmax} exceeds shared signed-l offset {signed_l_offset}"
    )]
    InvalidPhaseSearchPotentialLmax {
        potential: usize,
        lmax: usize,
        signed_l_offset: usize,
    },
    /// FEFF BAND T-matrix grid needs a non-empty phase table.
    #[error(
        "BAND lattice T-matrix grid phase table must have shape (energy >= 1, spin >= 1, signed_l >= 1, potential >= 1), got ({energy_count}, {spin_count}, {signed_l_count}, {potential_count})"
    )]
    InvalidLatticeTMatrixGridShape {
        energy_count: usize,
        spin_count: usize,
        signed_l_count: usize,
        potential_count: usize,
    },
    /// FEFF-compatible phase/reference interpolation failed.
    #[error("BAND phase interpolation failure: {0}")]
    Interpolation(#[from] InterpolationError),
    /// FEFF FMS T-matrix element evaluation failed.
    #[error("FMS T-matrix failure: {0}")]
    Fms(#[from] FmsError),
    /// KSPACE structure-factor assembly failed.
    #[error("KSPACE structure-factor failure: {0}")]
    KSpace(#[from] KSpaceError),
    /// FEFF-compatible LU factorization or solve failed.
    #[error("linear algebra failure: {0}")]
    LinearAlgebra(#[from] LinalgError),
}

/// Build FEFF `BAND/bandtot.f90` search energies.
///
/// FEFF converts the requested eV range to Hartree, clips it to the real part
/// of the active XSPH phase mesh shifted by `xmu`, computes
/// `nep = nint((emax-emin)/estep)+1`, and then recomputes `estep` so the final
/// point lands exactly on the clipped maximum.
pub fn band_energy_search_mesh(
    input: BandEnergySearchMeshInput<'_>,
) -> Result<BandEnergySearchMesh, BandError> {
    validate_finite("requested_min_ev", 0, input.requested_min_ev)?;
    validate_finite("requested_max_ev", 0, input.requested_max_ev)?;
    validate_finite("requested_step_ev", 0, input.requested_step_ev)?;
    validate_finite("fermi_level_hartree", 0, input.fermi_level_hartree)?;
    if input.requested_step_ev <= 0.0 {
        return Err(BandError::InvalidRequestedEnergyStep {
            step_ev: input.requested_step_ev,
        });
    }
    validate_phase_active_len(input.phase_active_len, input.phase_energies_hartree.len())?;

    let phase_min = input.phase_energies_hartree[0].re - input.fermi_level_hartree;
    let phase_max =
        input.phase_energies_hartree[input.phase_active_len - 1].re - input.fermi_level_hartree;
    validate_finite("phase_energy_hartree", 0, phase_min)?;
    validate_finite(
        "phase_energy_hartree",
        input.phase_active_len - 1,
        phase_max,
    )?;

    let mut min_hartree = input.requested_min_ev / FEFF_HARTREE_EV;
    let mut max_hartree = input.requested_max_ev / FEFF_HARTREE_EV;
    let requested_step_hartree = input.requested_step_ev / FEFF_HARTREE_EV;
    max_hartree = max_hartree.min(phase_max);
    min_hartree = min_hartree.max(phase_min);
    if !min_hartree.is_finite() || !max_hartree.is_finite() || max_hartree <= min_hartree {
        return Err(BandError::InvalidClippedEnergyRange {
            min_hartree,
            max_hartree,
        });
    }

    let ratio = (max_hartree - min_hartree) / requested_step_hartree;
    validate_finite("energy_point_ratio", 0, ratio)?;
    let intervals = nint_positive_to_usize(ratio)?;
    let point_count = intervals
        .checked_add(1)
        .ok_or(BandError::EnergyPointCountOverflow)?;
    if point_count < 2 {
        return Err(BandError::InvalidEnergyPointCount { point_count });
    }
    let step_hartree = (max_hartree - min_hartree) / ((point_count - 1) as Real);
    validate_finite("step_hartree", 0, step_hartree)?;

    let energies_hartree = Array1::from_iter(
        (0..point_count).map(|index| min_hartree + (index as Real) * step_hartree),
    );
    Ok(BandEnergySearchMesh {
        min_hartree,
        max_hartree,
        step_hartree,
        energies_hartree,
    })
}

/// Interpolate XSPH reference energies and phase shifts onto the BAND search mesh.
///
/// This ports the `bandtot.f90` loop that calls `terp` for `eref` and
/// `ph(:,ill,isp,ipp)` at every search energy before `fmsband`. The caller
/// supplies `source_phase_shifts` on a shared signed-`l` axis, with slots
/// indexed by `signed_l + signed_l_offset`; only the active
/// `-lmax..=lmax` range for each potential is interpolated.
pub fn band_phase_search_interpolation(
    input: BandPhaseSearchInterpolationInput<'_>,
) -> Result<BandPhaseSearchInterpolation, BandError> {
    let source_energy_count = input.source_energies_hartree.len();
    let (reference_energy_count, spin_count) = input.source_reference_energies_hartree.dim();
    let (phase_energy_count, signed_l_count, phase_spin_count, potential_count) =
        input.source_phase_shifts.dim();
    if source_energy_count == 0
        || signed_l_count == 0
        || signed_l_count % 2 == 0
        || spin_count == 0
        || potential_count == 0
    {
        return Err(BandError::InvalidPhaseSearchShape {
            source_energy_count,
            signed_l_count,
            spin_count,
            potential_count,
        });
    }
    if input.search_energies_hartree.is_empty() {
        return Err(BandError::InvalidBandTableLength {
            name: "search_energies_hartree",
            actual: 0,
            expected: 1,
        });
    }
    validate_band_table_len(
        "source_reference_energies",
        reference_energy_count,
        source_energy_count,
    )?;
    validate_band_table_len(
        "source_phase_shifts",
        phase_energy_count,
        source_energy_count,
    )?;
    validate_band_table_len("source_phase_shifts spin", phase_spin_count, spin_count)?;
    validate_band_table_len(
        "potential_lmax",
        input.potential_lmax.len(),
        potential_count,
    )?;

    let signed_l_offset = signed_l_count / 2;
    for (potential, &lmax) in input.potential_lmax.iter().enumerate() {
        if lmax > signed_l_offset {
            return Err(BandError::InvalidPhaseSearchPotentialLmax {
                potential,
                lmax,
                signed_l_offset,
            });
        }
    }

    let source_energies = input.source_energies_hartree.to_vec();
    let search_count = input.search_energies_hartree.len();
    let mut reference_energies_hartree = Array2::zeros((search_count, spin_count).f());
    let mut wave_numbers = Array2::zeros((search_count, spin_count).f());
    let mut phase_shifts =
        Array4::zeros((search_count, spin_count, signed_l_count, potential_count).f());

    let signed_l_offset_isize =
        isize::try_from(signed_l_offset).map_err(|_| BandError::MatrixOrderOverflow)?;

    for (search_index, &energy) in input.search_energies_hartree.iter().enumerate() {
        validate_finite("search_energy_hartree", search_index, energy)?;

        for spin in 0..spin_count {
            let reference_values = (0..source_energy_count)
                .map(|source| input.source_reference_energies_hartree[(source, spin)])
                .collect::<Vec<_>>();
            let reference = terpc(
                &source_energies,
                &reference_values,
                input.interpolation_order,
                energy,
            )?
            .value;
            validate_complex_finite("reference_energy_hartree", reference)?;
            reference_energies_hartree[(search_index, spin)] = reference;
            let wave_number =
                (Complex::new(2.0, 0.0) * (Complex::new(energy, 0.0) - reference)).sqrt();
            validate_complex_finite("wave_number", wave_number)?;
            let wave_number = Complex32::new(wave_number.re as f32, wave_number.im as f32);
            validate_complex32_finite("wave_number", wave_number)?;
            wave_numbers[(search_index, spin)] = wave_number;
        }

        for potential in 0..potential_count {
            let lmax = input.potential_lmax[potential];
            for signed_l in -(lmax as isize)..=(lmax as isize) {
                let slot = usize::try_from(signed_l + signed_l_offset_isize)
                    .map_err(|_| BandError::MatrixOrderOverflow)?;
                for spin in 0..spin_count {
                    let phase_values = (0..source_energy_count)
                        .map(|source| input.source_phase_shifts[(source, slot, spin, potential)])
                        .collect::<Vec<_>>();
                    let phase = terpc(
                        &source_energies,
                        &phase_values,
                        input.interpolation_order,
                        energy,
                    )?
                    .value;
                    validate_complex_finite("phase_shift", phase)?;
                    let phase = Complex32::new(phase.re as f32, phase.im as f32);
                    validate_complex32_finite("phase_shift", phase)?;
                    phase_shifts[(search_index, spin, slot, potential)] = phase;
                }
            }
        }
    }

    Ok(BandPhaseSearchInterpolation {
        signed_l_offset,
        reference_energies_hartree,
        wave_numbers,
        phase_shifts,
    })
}

/// Identify BAND energies from positive-eigenvalue count changes.
///
/// This ports the `bandtot.f90` loop that scans `n_pos(ie,ik)`. Each increase
/// from one energy row to the next emits one band energy at
/// `emin + (ie-1)*estep`; increases larger than one emit repeated energies,
/// matching FEFF's degeneracy handling.
pub fn band_energies_from_positive_counts(
    input: BandEnergiesFromPositiveCountsInput<'_>,
) -> Result<BandEnergiesFromPositiveCounts, BandError> {
    let rows = input.positive_counts.nrows();
    let columns = input.positive_counts.ncols();
    if rows < 2 || columns == 0 {
        return Err(BandError::InvalidPositiveCountShape { rows, columns });
    }
    validate_finite("energy_min_hartree", 0, input.energy_min_hartree)?;
    if !input.energy_step_hartree.is_finite() || input.energy_step_hartree <= 0.0 {
        return Err(BandError::InvalidSearchEnergyStep {
            step_hartree: input.energy_step_hartree,
        });
    }

    let mut band_energies_hartree = Vec::with_capacity(columns);
    for kpoint in 0..columns {
        let mut row = Vec::new();
        for energy_index in 1..rows {
            let previous = input.positive_counts[(energy_index - 1, kpoint)];
            let current = input.positive_counts[(energy_index, kpoint)];
            if current > previous {
                let energy =
                    input.energy_min_hartree + (energy_index as Real) * input.energy_step_hartree;
                for _ in 0..(current - previous) {
                    row.push(energy);
                }
            }
        }
        band_energies_hartree.push(Array1::from_vec(row));
    }

    Ok(BandEnergiesFromPositiveCounts {
        band_energies_hartree,
    })
}

/// Assemble FEFF `BAND/fmsband.f90`'s full lattice T-matrix.
///
/// FEFF first builds the compact FMS `tmatrx` table from phase shifts and
/// spin-orbit coupling coefficients, then expands the same-site diagonal and
/// spin-mixing entries into the full `Tmat(state,state)` lattice matrix. This
/// helper mirrors that expansion: every row gets its same-site diagonal entry
/// and, for two-spin runs, only the adjacent spin-mixing partner implied by
/// FEFF state order is considered.
pub fn band_lattice_t_matrix(
    input: BandLatticeTMatrixInput<'_>,
) -> Result<Array2<Complex32>, BandError> {
    if input.states.is_empty() {
        return Err(BandError::EmptyStateList);
    }

    let mut matrix = Array2::zeros((input.states.len(), input.states.len()).f());
    for (row, &first) in input.states.iter().enumerate() {
        let atom = state_atom_index(first.atom, input.atoms.len())?;
        let potential = usize::try_from(input.atoms[atom].potential).map_err(|_| {
            FmsError::PotentialOutOfRange {
                potential: input.atoms[atom].potential,
                max_potential: input
                    .phase_shifts
                    .len_of(ndarray::Axis(2))
                    .saturating_sub(1),
            }
        })?;

        matrix[(row, row)] = fms_t_matrix_element(FmsTMatrixInput {
            first,
            second: first,
            spin_channels: input.spin_channels,
            spin_selector: input.spin_selector,
            potential,
            phase_shifts: input.phase_shifts,
            spin_orbit: input.spin_orbit,
        })?;

        if input.spin_channels == 2
            && let Some(column) = fmsband_spin_mixing_column(row, first.spin, input.states.len())
        {
            let second = input.states[column];
            matrix[(row, column)] = fms_t_matrix_element(FmsTMatrixInput {
                first,
                second,
                spin_channels: input.spin_channels,
                spin_selector: input.spin_selector,
                potential,
                phase_shifts: input.phase_shifts,
                spin_orbit: input.spin_orbit,
            })?;
        }
    }

    Ok(matrix)
}

/// Assemble FEFF `BAND/fmsband.f90` lattice T-matrices over the search mesh.
///
/// `bandtot.f90` interpolates `xphase(energy,spin,l,potential)` before each
/// `fmsband` call. This helper builds the corresponding full `Tmat` for every
/// search energy, producing the `(energy,state,state)` table consumed by the
/// source-backed KKR grid helpers.
pub fn band_lattice_t_matrix_grid(
    input: BandLatticeTMatrixGridInput<'_>,
) -> Result<Array3<Complex32>, BandError> {
    if input.states.is_empty() {
        return Err(BandError::EmptyStateList);
    }
    let (energy_count, spin_count, signed_l_count, potential_count) = input.phase_shifts.dim();
    if energy_count == 0 || spin_count == 0 || signed_l_count == 0 || potential_count == 0 {
        return Err(BandError::InvalidLatticeTMatrixGridShape {
            energy_count,
            spin_count,
            signed_l_count,
            potential_count,
        });
    }
    if spin_count != input.spin_channels {
        return Err(BandError::InvalidBandTableLength {
            name: "phase_shifts spin",
            actual: spin_count,
            expected: input.spin_channels,
        });
    }

    let state_count = input.states.len();
    let mut matrices = Array3::zeros((energy_count, state_count, state_count).f());
    for energy in 0..energy_count {
        let phase_shifts = input.phase_shifts.index_axis(Axis(0), energy);
        let matrix = band_lattice_t_matrix(BandLatticeTMatrixInput {
            states: input.states,
            atoms: input.atoms,
            spin_channels: input.spin_channels,
            spin_selector: input.spin_selector,
            phase_shifts,
            spin_orbit: input.spin_orbit,
        })?;
        for row in 0..state_count {
            for column in 0..state_count {
                matrices[(energy, row, column)] = matrix[(row, column)];
            }
        }
    }

    Ok(matrices)
}

fn fmsband_spin_mixing_column(row: usize, spin: usize, state_count: usize) -> Option<usize> {
    match spin {
        1 => row.checked_sub(1),
        2 => row
            .checked_add(1)
            .and_then(|column| (column < state_count).then_some(column)),
        _ => None,
    }
}

/// Build FEFF `BAND/kkrband.f90`'s KKR work matrix `G - T^-1`.
///
/// The Fortran routine calls `structurefactor` into `G`, inverts the full
/// `Tmat`, and replaces `G` with `G - Tmat^{-1}` before the BAND eigenvalue
/// count is evaluated.
pub fn band_kkr_matrix_from_structure_factor(
    input: BandKkrMatrixInput<'_>,
) -> Result<Array2<Complex32>, BandError> {
    validate_square_matrix("structure_factor", input.structure_factor)?;
    validate_square_matrix("t_matrix", input.t_matrix)?;
    if input.structure_factor.dim() != input.t_matrix.dim() {
        return Err(BandError::MatrixShapeMismatch {
            left_name: "structure_factor",
            left_rows: input.structure_factor.nrows(),
            left_columns: input.structure_factor.ncols(),
            right_name: "t_matrix",
            right_rows: input.t_matrix.nrows(),
            right_columns: input.t_matrix.ncols(),
        });
    }

    let inverse = invert_complex32_matrix(input.t_matrix)?;
    let mut matrix = input.structure_factor.to_owned();
    matrix -= &inverse;
    Ok(matrix)
}

/// Compute sorted FEFF BAND KKR eigenvalues for one energy/k-point.
///
/// FEFF `fmsband.f90` multiplies `Gfms` by complex momentum `p`, calls LAPACK
/// `CGEES` without eigenvectors or sorting, then orders the returned
/// eigenvalues from largest to smallest real part. This helper performs the
/// same FEFF-specific pre- and post-processing around the Rust linalg wrapper.
pub fn band_sorted_kkr_eigenvalues(
    input: BandSortedKkrEigenvaluesInput<'_>,
) -> Result<Array1<Complex32>, BandError> {
    validate_complex32_finite("wave_number", input.wave_number)?;
    validate_square_matrix("kkr_matrix", input.kkr_matrix)?;

    let scaled = Array2::from_shape_fn(input.kkr_matrix.dim().f(), |(row, column)| {
        input.kkr_matrix[(row, column)] * input.wave_number
    });
    let eigenvalues = complex32_general_eigenvalues(scaled.view())?;
    let mut eigenvalues = eigenvalues.to_vec();
    eigenvalues.sort_by(|left, right| right.re.total_cmp(&left.re));
    Ok(Array1::from_vec(eigenvalues))
}

/// Compute sorted FEFF BAND KKR eigenvalues directly from a structure factor.
///
/// This composes the deterministic `fmsband.f90` path after `structurefactor`:
/// build `Gfms = G - Tmat^-1`, scale by complex momentum, diagonalize, and
/// sort the returned eigenvalues by descending real part.
pub fn band_kkr_eigenvalues_from_structure_factor(
    input: BandKkrEigenvaluesFromStructureFactorInput<'_>,
) -> Result<Array1<Complex32>, BandError> {
    let kkr_matrix = band_kkr_matrix_from_structure_factor(BandKkrMatrixInput {
        structure_factor: input.structure_factor,
        t_matrix: input.t_matrix,
    })?;
    band_sorted_kkr_eigenvalues(BandSortedKkrEigenvaluesInput {
        kkr_matrix: kkr_matrix.view(),
        wave_number: input.wave_number,
    })
}

/// Compute sorted FEFF BAND eigenvalues for `freeprop`.
///
/// This ports the `kkrband.f90` early-return path used when `freeprop` is set:
/// `structurefactor` fills `G`, `T^-1` is not subtracted, and `fmsband.f90`
/// diagonalizes `G * p`.
pub fn band_free_propagation_eigenvalues_from_structure_factor(
    input: BandFreePropagationEigenvaluesFromStructureFactorInput<'_>,
) -> Result<Array1<Complex32>, BandError> {
    band_sorted_kkr_eigenvalues(BandSortedKkrEigenvaluesInput {
        kkr_matrix: input.structure_factor,
        wave_number: input.wave_number,
    })
}

/// Solve FEFF BAND KKR eigenvalues over a `(energy,kpoint)` structure-factor grid.
///
/// The returned eigenvalue cube and positive-count table are the source-backed
/// Rust equivalent of the `bandtot.f90` loop that calls `fmsband` at every
/// search energy and k-point before scanning `n_pos`.
pub fn band_kkr_eigenvalue_grid(
    input: BandKkrEigenvalueGridInput<'_>,
) -> Result<BandKkrEigenvalueGrid, BandError> {
    let (energy_count, k_point_count, rows, columns) = input.structure_factors.dim();
    if energy_count == 0 || k_point_count == 0 || rows == 0 || rows != columns {
        return Err(BandError::InvalidKkrEigenvalueGridShape {
            energy_count,
            k_point_count,
            rows,
            columns,
        });
    }

    let (t_energy_count, t_rows, t_columns) = input.t_matrices.dim();
    if t_energy_count != energy_count {
        return Err(BandError::InvalidBandTableLength {
            name: "t_matrices",
            actual: t_energy_count,
            expected: energy_count,
        });
    }
    if t_rows != rows || t_columns != columns {
        return Err(BandError::MatrixShapeMismatch {
            left_name: "structure_factors",
            left_rows: rows,
            left_columns: columns,
            right_name: "t_matrices",
            right_rows: t_rows,
            right_columns: t_columns,
        });
    }
    if input.wave_numbers.len() != energy_count {
        return Err(BandError::InvalidBandTableLength {
            name: "wave_numbers",
            actual: input.wave_numbers.len(),
            expected: energy_count,
        });
    }

    let mut eigenvalues = Array3::zeros((energy_count, k_point_count, rows).f());
    let mut positive_counts = Array2::zeros((energy_count, k_point_count).f());
    for energy_index in 0..energy_count {
        let t_matrix = input.t_matrices.index_axis(Axis(0), energy_index);
        let structure_factors_for_energy =
            input.structure_factors.index_axis(Axis(0), energy_index);
        for k_point_index in 0..k_point_count {
            let structure_factor = structure_factors_for_energy.index_axis(Axis(0), k_point_index);
            let row = band_kkr_eigenvalues_from_structure_factor(
                BandKkrEigenvaluesFromStructureFactorInput {
                    structure_factor,
                    t_matrix,
                    wave_number: input.wave_numbers[energy_index],
                },
            )?;
            positive_counts[(energy_index, k_point_index)] =
                band_positive_eigenvalue_count(row.view())?;
            for eigenvalue_index in 0..rows {
                eigenvalues[(energy_index, k_point_index, eigenvalue_index)] =
                    row[eigenvalue_index];
            }
        }
    }

    Ok(BandKkrEigenvalueGrid {
        eigenvalues,
        positive_counts,
    })
}

/// Solve FEFF `freeprop` eigenvalues over a `(energy,kpoint)` structure-factor grid.
///
/// This mirrors the ordinary BAND grid loop but preserves the `kkrband.f90`
/// `freeprop` behavior, diagonalizing each raw structure-factor matrix `G`
/// after multiplication by the corresponding complex momentum.
pub fn band_free_propagation_eigenvalue_grid(
    input: BandFreePropagationEigenvalueGridInput<'_>,
) -> Result<BandKkrEigenvalueGrid, BandError> {
    let (energy_count, k_point_count, rows, columns) = input.structure_factors.dim();
    if energy_count == 0 || k_point_count == 0 || rows == 0 || rows != columns {
        return Err(BandError::InvalidKkrEigenvalueGridShape {
            energy_count,
            k_point_count,
            rows,
            columns,
        });
    }
    if input.wave_numbers.len() != energy_count {
        return Err(BandError::InvalidBandTableLength {
            name: "wave_numbers",
            actual: input.wave_numbers.len(),
            expected: energy_count,
        });
    }

    let mut eigenvalues = Array3::zeros((energy_count, k_point_count, rows).f());
    let mut positive_counts = Array2::zeros((energy_count, k_point_count).f());
    for energy_index in 0..energy_count {
        let structure_factors_for_energy =
            input.structure_factors.index_axis(Axis(0), energy_index);
        for k_point_index in 0..k_point_count {
            let structure_factor = structure_factors_for_energy.index_axis(Axis(0), k_point_index);
            let row = band_free_propagation_eigenvalues_from_structure_factor(
                BandFreePropagationEigenvaluesFromStructureFactorInput {
                    structure_factor,
                    wave_number: input.wave_numbers[energy_index],
                },
            )?;
            positive_counts[(energy_index, k_point_index)] =
                band_positive_eigenvalue_count(row.view())?;
            for eigenvalue_index in 0..rows {
                eigenvalues[(energy_index, k_point_index, eigenvalue_index)] =
                    row[eigenvalue_index];
            }
        }
    }

    Ok(BandKkrEigenvalueGrid {
        eigenvalues,
        positive_counts,
    })
}

/// Solve a FEFF BAND KKR search grid and identify final band-energy rows.
///
/// This composes the `bandtot.f90` loop after structure-factor generation:
/// solve every `(energy,kpoint)` KKR matrix, build `n_pos`, then scan count
/// increases into the variable-length band-energy rows written by FEFF.
pub fn band_kkr_band_energies(
    input: BandKkrBandEnergiesInput<'_>,
) -> Result<BandKkrBandEnergies, BandError> {
    let grid = band_kkr_eigenvalue_grid(BandKkrEigenvalueGridInput {
        structure_factors: input.structure_factors,
        t_matrices: input.t_matrices,
        wave_numbers: input.wave_numbers,
    })?;
    let band_energies = band_energies_from_positive_counts(BandEnergiesFromPositiveCountsInput {
        positive_counts: grid.positive_counts.view(),
        energy_min_hartree: input.energy_min_hartree,
        energy_step_hartree: input.energy_step_hartree,
    })?;
    let BandKkrEigenvalueGrid {
        eigenvalues,
        positive_counts,
    } = grid;

    Ok(BandKkrBandEnergies {
        eigenvalues,
        positive_counts,
        band_energies,
    })
}

/// Solve a FEFF `freeprop` search grid and identify final band-energy rows.
pub fn band_free_propagation_band_energies(
    input: BandFreePropagationBandEnergiesInput<'_>,
) -> Result<BandKkrBandEnergies, BandError> {
    let grid = band_free_propagation_eigenvalue_grid(BandFreePropagationEigenvalueGridInput {
        structure_factors: input.structure_factors,
        wave_numbers: input.wave_numbers,
    })?;
    let band_energies = band_energies_from_positive_counts(BandEnergiesFromPositiveCountsInput {
        positive_counts: grid.positive_counts.view(),
        energy_min_hartree: input.energy_min_hartree,
        energy_step_hartree: input.energy_step_hartree,
    })?;
    let BandKkrEigenvalueGrid {
        eigenvalues,
        positive_counts,
    } = grid;

    Ok(BandKkrBandEnergies {
        eigenvalues,
        positive_counts,
        band_energies,
    })
}

/// Solve ordinary FEFF BAND rows from `G(energy,kpoint)` and interpolated phases.
///
/// This composes the post-KSPACE `bandtot.f90` ordinary branch: assemble the
/// per-search-energy lattice T-matrices from `xphase`, solve every KKR matrix,
/// build `n_pos`, and scan the final variable-length band rows.
pub fn band_kkr_band_energies_from_phase_structure_grid(
    input: BandKkrBandEnergiesFromPhaseStructureGridInput<'_>,
) -> Result<BandKkrBandEnergiesFromPhaseStructureGrid, BandError> {
    let t_matrices = band_lattice_t_matrix_grid(BandLatticeTMatrixGridInput {
        states: input.states,
        atoms: input.atoms,
        spin_channels: input.spin_channels,
        spin_selector: input.spin_selector,
        phase_shifts: input.phase_shifts,
        spin_orbit: input.spin_orbit,
    })?;
    let solved = band_kkr_band_energies(BandKkrBandEnergiesInput {
        structure_factors: input.structure_factors,
        t_matrices: t_matrices.view(),
        wave_numbers: input.wave_numbers,
        energy_min_hartree: input.energy_min_hartree,
        energy_step_hartree: input.energy_step_hartree,
    })?;

    Ok(BandKkrBandEnergiesFromPhaseStructureGrid { t_matrices, solved })
}

/// Solve ordinary non-relativistic KSPACE-backed FEFF BAND rows from phases.
///
/// This is the source-backed ordinary `bandtot.f90` composition after the
/// search mesh is known: build FEFF-basis KSPACE structure-factor grids, expand
/// interpolated phases into per-energy lattice T-matrices, then solve KKR and
/// identify final band rows.
pub fn band_kkr_band_energies_from_kspace_phase_non_rel_grid(
    input: BandKkrBandEnergiesFromKspacePhaseNonRelGridInput<'_, '_>,
) -> Result<BandKkrBandEnergiesFromKspacePhaseGrid, BandError> {
    let structure_factors = band_structure_factor_from_kspace_non_rel_grid(
        BandStructureFactorFromKspaceNonRelGridInput {
            point_inputs: input.point_inputs,
            energy_count: input.energy_count,
            k_point_count: input.k_point_count,
        },
    )?;
    let solved = band_kkr_band_energies_from_phase_structure_grid(
        BandKkrBandEnergiesFromPhaseStructureGridInput {
            structure_factors: structure_factors.structure_factors.view(),
            states: input.states,
            atoms: input.atoms,
            spin_channels: input.spin_channels,
            spin_selector: input.spin_selector,
            phase_shifts: input.phase_shifts,
            spin_orbit: input.spin_orbit,
            wave_numbers: input.wave_numbers,
            energy_min_hartree: input.energy_min_hartree,
            energy_step_hartree: input.energy_step_hartree,
        },
    )?;

    Ok(BandKkrBandEnergiesFromKspacePhaseGrid {
        structure_factors,
        solved,
    })
}

/// Solve ordinary relativistic KSPACE-backed FEFF BAND rows from phases.
pub fn band_kkr_band_energies_from_kspace_phase_rel_grid(
    input: BandKkrBandEnergiesFromKspacePhaseRelGridInput<'_, '_>,
) -> Result<BandKkrBandEnergiesFromKspacePhaseGrid, BandError> {
    let structure_factors =
        band_structure_factor_from_kspace_rel_grid(BandStructureFactorFromKspaceRelGridInput {
            point_inputs: input.point_inputs,
            energy_count: input.energy_count,
            k_point_count: input.k_point_count,
        })?;
    let solved = band_kkr_band_energies_from_phase_structure_grid(
        BandKkrBandEnergiesFromPhaseStructureGridInput {
            structure_factors: structure_factors.structure_factors.view(),
            states: input.states,
            atoms: input.atoms,
            spin_channels: input.spin_channels,
            spin_selector: input.spin_selector,
            phase_shifts: input.phase_shifts,
            spin_orbit: input.spin_orbit,
            wave_numbers: input.wave_numbers,
            energy_min_hartree: input.energy_min_hartree,
            energy_step_hartree: input.energy_step_hartree,
        },
    )?;

    Ok(BandKkrBandEnergiesFromKspacePhaseGrid {
        structure_factors,
        solved,
    })
}

/// Solve one non-relativistic KSPACE-backed BAND KKR point.
///
/// This composes source-backed KSPACE structure constants, FEFF-basis
/// `structurefactor` conversion, `kkrband` matrix setup, `fmsband`
/// diagonalization, sorting, and positive-eigenvalue counting for one
/// `(energy,kpoint)`.
pub fn band_kkr_from_kspace_non_rel(
    input: BandKkrFromKspaceNonRelInput<'_>,
) -> Result<BandKkrFromKspace, BandError> {
    let structure_factor = band_structure_factor_from_kspace_non_rel(input.structure_factor)?;
    band_kkr_from_kspace_structure_factor(structure_factor, input.t_matrix, input.wave_number)
}

/// Solve one relativistic KSPACE-backed BAND KKR point.
///
/// This is the `IREL >= 2` companion to [`band_kkr_from_kspace_non_rel`],
/// preserving the same downstream KKR solve and positive-count behavior after
/// the relativistic KSPACE structure-factor branch.
pub fn band_kkr_from_kspace_rel(
    input: BandKkrFromKspaceRelInput<'_>,
) -> Result<BandKkrFromKspace, BandError> {
    let structure_factor = band_structure_factor_from_kspace_rel(input.structure_factor)?;
    band_kkr_from_kspace_structure_factor(structure_factor, input.t_matrix, input.wave_number)
}

/// Solve one non-relativistic KSPACE-backed FEFF `freeprop` point.
///
/// This composes KSPACE structure-factor generation with the `kkrband.f90`
/// `freeprop` early-return branch, so the raw `G` matrix is diagonalized after
/// multiplication by the complex momentum.
pub fn band_free_propagation_from_kspace_non_rel(
    input: BandFreePropagationFromKspaceNonRelInput<'_>,
) -> Result<BandKkrFromKspace, BandError> {
    let structure_factor = band_structure_factor_from_kspace_non_rel(input.structure_factor)?;
    band_free_propagation_from_kspace_structure_factor(structure_factor, input.wave_number)
}

/// Solve one relativistic KSPACE-backed FEFF `freeprop` point.
pub fn band_free_propagation_from_kspace_rel(
    input: BandFreePropagationFromKspaceRelInput<'_>,
) -> Result<BandKkrFromKspace, BandError> {
    let structure_factor = band_structure_factor_from_kspace_rel(input.structure_factor)?;
    band_free_propagation_from_kspace_structure_factor(structure_factor, input.wave_number)
}

/// Solve a non-relativistic KSPACE-backed FEFF BAND KKR grid.
///
/// `point_inputs` are consumed in FEFF `bandtot.f90` loop order: all k-points
/// for the first search energy, then all k-points for the next search energy.
/// The returned arrays are shaped as `(energy,kpoint,state)`.
pub fn band_kkr_from_kspace_non_rel_grid(
    input: BandKkrFromKspaceNonRelGridInput<'_, '_>,
) -> Result<BandKkrFromKspaceGrid, BandError> {
    let expected_points = validate_kspace_grid_input(
        "point_inputs",
        input.point_inputs.len(),
        input.energy_count,
        input.k_point_count,
    )?;
    let mut point_solves = Vec::with_capacity(expected_points);
    for point_index in 0..expected_points {
        point_solves.push(band_kkr_from_kspace_non_rel(
            input.point_inputs[point_index],
        )?);
    }
    band_kkr_from_kspace_point_solves_grid(point_solves, input.energy_count, input.k_point_count)
}

/// Solve a non-relativistic KSPACE-backed FEFF `freeprop` grid.
pub fn band_free_propagation_from_kspace_non_rel_grid(
    input: BandFreePropagationFromKspaceNonRelGridInput<'_, '_>,
) -> Result<BandKkrFromKspaceGrid, BandError> {
    let expected_points = validate_kspace_grid_input(
        "point_inputs",
        input.point_inputs.len(),
        input.energy_count,
        input.k_point_count,
    )?;
    let mut point_solves = Vec::with_capacity(expected_points);
    for point_index in 0..expected_points {
        point_solves.push(band_free_propagation_from_kspace_non_rel(
            input.point_inputs[point_index],
        )?);
    }
    band_kkr_from_kspace_point_solves_grid(point_solves, input.energy_count, input.k_point_count)
}

/// Solve a relativistic KSPACE-backed FEFF BAND KKR grid.
///
/// This is the `IREL >= 2` companion to
/// [`band_kkr_from_kspace_non_rel_grid`] with the same FEFF loop-order
/// contract and output shapes.
pub fn band_kkr_from_kspace_rel_grid(
    input: BandKkrFromKspaceRelGridInput<'_, '_>,
) -> Result<BandKkrFromKspaceGrid, BandError> {
    let expected_points = validate_kspace_grid_input(
        "point_inputs",
        input.point_inputs.len(),
        input.energy_count,
        input.k_point_count,
    )?;
    let mut point_solves = Vec::with_capacity(expected_points);
    for point_index in 0..expected_points {
        point_solves.push(band_kkr_from_kspace_rel(input.point_inputs[point_index])?);
    }
    band_kkr_from_kspace_point_solves_grid(point_solves, input.energy_count, input.k_point_count)
}

/// Solve a relativistic KSPACE-backed FEFF `freeprop` grid.
pub fn band_free_propagation_from_kspace_rel_grid(
    input: BandFreePropagationFromKspaceRelGridInput<'_, '_>,
) -> Result<BandKkrFromKspaceGrid, BandError> {
    let expected_points = validate_kspace_grid_input(
        "point_inputs",
        input.point_inputs.len(),
        input.energy_count,
        input.k_point_count,
    )?;
    let mut point_solves = Vec::with_capacity(expected_points);
    for point_index in 0..expected_points {
        point_solves.push(band_free_propagation_from_kspace_rel(
            input.point_inputs[point_index],
        )?);
    }
    band_kkr_from_kspace_point_solves_grid(point_solves, input.energy_count, input.k_point_count)
}

/// Solve a non-relativistic KSPACE-backed BAND KKR grid through final band rows.
pub fn band_kkr_band_energies_from_kspace_non_rel_grid(
    input: BandKkrBandEnergiesFromKspaceNonRelGridInput<'_, '_>,
) -> Result<BandKkrBandEnergiesFromKspaceGrid, BandError> {
    let grid = band_kkr_from_kspace_non_rel_grid(BandKkrFromKspaceNonRelGridInput {
        point_inputs: input.point_inputs,
        energy_count: input.energy_count,
        k_point_count: input.k_point_count,
    })?;
    band_kkr_band_energies_from_kspace_grid(
        grid,
        input.energy_min_hartree,
        input.energy_step_hartree,
    )
}

/// Solve a non-relativistic KSPACE-backed `freeprop` grid through final rows.
pub fn band_free_propagation_band_energies_from_kspace_non_rel_grid(
    input: BandFreePropagationBandEnergiesFromKspaceNonRelGridInput<'_, '_>,
) -> Result<BandKkrBandEnergiesFromKspaceGrid, BandError> {
    let grid = band_free_propagation_from_kspace_non_rel_grid(
        BandFreePropagationFromKspaceNonRelGridInput {
            point_inputs: input.point_inputs,
            energy_count: input.energy_count,
            k_point_count: input.k_point_count,
        },
    )?;
    band_kkr_band_energies_from_kspace_grid(
        grid,
        input.energy_min_hartree,
        input.energy_step_hartree,
    )
}

/// Solve a relativistic KSPACE-backed BAND KKR grid through final band rows.
pub fn band_kkr_band_energies_from_kspace_rel_grid(
    input: BandKkrBandEnergiesFromKspaceRelGridInput<'_, '_>,
) -> Result<BandKkrBandEnergiesFromKspaceGrid, BandError> {
    let grid = band_kkr_from_kspace_rel_grid(BandKkrFromKspaceRelGridInput {
        point_inputs: input.point_inputs,
        energy_count: input.energy_count,
        k_point_count: input.k_point_count,
    })?;
    band_kkr_band_energies_from_kspace_grid(
        grid,
        input.energy_min_hartree,
        input.energy_step_hartree,
    )
}

/// Solve a relativistic KSPACE-backed `freeprop` grid through final rows.
pub fn band_free_propagation_band_energies_from_kspace_rel_grid(
    input: BandFreePropagationBandEnergiesFromKspaceRelGridInput<'_, '_>,
) -> Result<BandKkrBandEnergiesFromKspaceGrid, BandError> {
    let grid =
        band_free_propagation_from_kspace_rel_grid(BandFreePropagationFromKspaceRelGridInput {
            point_inputs: input.point_inputs,
            energy_count: input.energy_count,
            k_point_count: input.k_point_count,
        })?;
    band_kkr_band_energies_from_kspace_grid(
        grid,
        input.energy_min_hartree,
        input.energy_step_hartree,
    )
}

/// Count FEFF `bandtot.f90` eigenvalues with positive real part.
///
/// FEFF increments `n_pos_eigenval` for each eigenvalue whose real part is
/// strictly greater than zero. Imaginary parts do not affect the count.
pub fn band_positive_eigenvalue_count(
    eigenvalues: ArrayView1<'_, Complex32>,
) -> Result<usize, BandError> {
    if eigenvalues.is_empty() {
        return Err(BandError::EmptyEigenvalueList);
    }

    let mut count = 0;
    for (index, &value) in eigenvalues.iter().enumerate() {
        validate_complex32_finite_index("eigenvalue", index, value)?;
        if value.re > 0.0 {
            count += 1;
        }
    }
    Ok(count)
}

/// Build FEFF `n_pos(ie,ik)` from solved BAND eigenvalues.
///
/// This ports the `bandtot.f90` loop immediately after `fmsband`: each
/// `(energy,kpoint)` row counts how many sorted eigenvalues have positive real
/// part. The resulting table is the input to
/// [`band_energies_from_positive_counts`].
pub fn band_positive_counts_from_eigenvalues(
    input: BandPositiveCountsFromEigenvaluesInput<'_>,
) -> Result<Array2<usize>, BandError> {
    let (energy_count, k_point_count, eigenvalue_count) = input.eigenvalues.dim();
    if energy_count == 0 || k_point_count == 0 || eigenvalue_count == 0 {
        return Err(BandError::InvalidEigenvalueCubeShape {
            energy_count,
            k_point_count,
            eigenvalue_count,
        });
    }

    let mut counts = Array2::zeros((energy_count, k_point_count).f());
    for energy_index in 0..energy_count {
        for k_point_index in 0..k_point_count {
            let mut count = 0;
            for eigenvalue_index in 0..eigenvalue_count {
                let value = input.eigenvalues[(energy_index, k_point_index, eigenvalue_index)];
                validate_complex32_finite_index("eigenvalue", eigenvalue_index, value)?;
                if value.re > 0.0 {
                    count += 1;
                }
            }
            counts[(energy_index, k_point_index)] = count;
        }
    }
    Ok(counts)
}

/// Convert FEFF `structurefactor.f90`'s SPRKKR `tauk` blocks into FEFF basis.
///
/// This ports the deterministic tail after `strset`: every lattice-atom block
/// is divided by `p`, transformed from real spherical harmonics, including the
/// spin index when present, to complex spherical harmonics with FEFF
/// `CHANGEREP` mode `RLM>CLM`, multiplied by `i^(l_row-l_column)`, and finally
/// stored as single-complex `G`.
pub fn band_structure_factor_feff_basis(
    input: BandStructureFactorFeffBasisInput<'_>,
) -> Result<Array2<Complex32>, BandError> {
    validate_complex_finite("wave_number", input.wave_number)?;
    if input.wave_number == Complex::new(0.0, 0.0) {
        return Err(BandError::ZeroWaveNumber);
    }
    if input.atom_count == 0 {
        return Err(BandError::EmptyAtomList);
    }

    let non_spin_order = angular_block_order(input.angular_lmax)?;
    let block_order = structure_factor_block_order(&input, non_spin_order)?;
    let expected_order = input
        .atom_count
        .checked_mul(block_order)
        .ok_or(BandError::MatrixOrderOverflow)?;
    let real_to_complex = real_to_complex_block(
        input.basis_transforms,
        input.angular_lmax,
        non_spin_order,
        block_order,
    )?;
    let real_to_complex_adjoint = conjugate_transpose_complex(real_to_complex.view());
    let orbital_momenta =
        spherical_index_orbital_momenta_for_block(input.angular_lmax, non_spin_order, block_order)?;

    let mut output = Array2::zeros((expected_order, expected_order).f());
    for atom_row in 0..input.atom_count {
        for atom_column in 0..input.atom_count {
            let row_offset = atom_row * block_order;
            let column_offset = atom_column * block_order;
            let mut block = Array2::zeros((block_order, block_order).f());
            for row in 0..block_order {
                for column in 0..block_order {
                    block[(row, column)] = input.tauk_sprkkr
                        [(row_offset + row, column_offset + column)]
                        / input.wave_number;
                }
            }

            let work = complex_matmul(real_to_complex_adjoint.view(), block.view());
            let mut converted = complex_matmul(work.view(), real_to_complex.view());
            for row in 0..block_order {
                for column in 0..block_order {
                    let exponent = isize::try_from(orbital_momenta[row])
                        .map_err(|_| BandError::MatrixOrderOverflow)?
                        - isize::try_from(orbital_momenta[column])
                            .map_err(|_| BandError::MatrixOrderOverflow)?;
                    converted[(row, column)] *= complex_i_power(exponent);
                    let value = converted[(row, column)];
                    output[(row_offset + row, column_offset + column)] =
                        Complex32::new(value.re as f32, value.im as f32);
                }
            }
        }
    }

    Ok(output)
}

/// Assemble one non-relativistic KSPACE structure factor in FEFF basis.
///
/// This composes the Rust-backed `STRBBDD -> STRSET` path with the
/// `structurefactor.f90` FEFF-basis tail, returning both the KSPACE
/// intermediates and the `G` matrix consumed by the BAND KKR solve.
pub fn band_structure_factor_from_kspace_non_rel(
    input: BandStructureFactorFromKspaceNonRelInput<'_>,
) -> Result<BandStructureFactorFromKspace, BandError> {
    let BandStructureFactorFromKspaceNonRelInput {
        kspace,
        atom_count,
        angular_lmax,
        basis_transforms,
    } = input;
    let wave_number = kspace.wave_number;
    let kspace = kspace_strset_non_rel_from_lattice_sum(kspace)?;
    let structure_factor = band_structure_factor_feff_basis(BandStructureFactorFeffBasisInput {
        tauk_sprkkr: kspace.taukinv.view(),
        wave_number,
        atom_count,
        angular_lmax,
        basis_transforms,
    })?;

    Ok(BandStructureFactorFromKspace {
        kspace,
        structure_factor,
    })
}

/// Assemble one relativistic KSPACE structure factor in FEFF basis.
///
/// This composes the Rust-backed `STRBBDD -> STRSET` relativistic branch with
/// the `structurefactor.f90` FEFF-basis tail, preserving the KSPACE
/// intermediates needed for diagnostics and reference comparisons.
pub fn band_structure_factor_from_kspace_rel(
    input: BandStructureFactorFromKspaceRelInput<'_>,
) -> Result<BandStructureFactorFromKspace, BandError> {
    let BandStructureFactorFromKspaceRelInput {
        kspace,
        atom_count,
        angular_lmax,
        basis_transforms,
    } = input;
    let wave_number = kspace.wave_number;
    let kspace = kspace_strset_rel_from_lattice_sum(kspace)?;
    let structure_factor = band_structure_factor_feff_basis(BandStructureFactorFeffBasisInput {
        tauk_sprkkr: kspace.taukinv.view(),
        wave_number,
        atom_count,
        angular_lmax,
        basis_transforms,
    })?;

    Ok(BandStructureFactorFromKspace {
        kspace,
        structure_factor,
    })
}

/// Assemble non-relativistic KSPACE structure factors over a BAND search grid.
///
/// `point_inputs` are consumed in FEFF `bandtot.f90` loop order: all k-points
/// for one search energy before moving to the next energy. The returned
/// `structure_factors` grid is the FEFF-basis `G(energy,kpoint)` table used by
/// the downstream BAND KKR solvers.
pub fn band_structure_factor_from_kspace_non_rel_grid(
    input: BandStructureFactorFromKspaceNonRelGridInput<'_, '_>,
) -> Result<BandStructureFactorFromKspaceGrid, BandError> {
    let expected_points = validate_kspace_grid_input(
        "point_inputs",
        input.point_inputs.len(),
        input.energy_count,
        input.k_point_count,
    )?;
    let mut point_solves = Vec::with_capacity(expected_points);
    for point_index in 0..expected_points {
        point_solves.push(band_structure_factor_from_kspace_non_rel(
            input.point_inputs[point_index],
        )?);
    }
    band_structure_factor_from_kspace_point_grid(
        point_solves,
        input.energy_count,
        input.k_point_count,
    )
}

/// Assemble relativistic KSPACE structure factors over a BAND search grid.
pub fn band_structure_factor_from_kspace_rel_grid(
    input: BandStructureFactorFromKspaceRelGridInput<'_, '_>,
) -> Result<BandStructureFactorFromKspaceGrid, BandError> {
    let expected_points = validate_kspace_grid_input(
        "point_inputs",
        input.point_inputs.len(),
        input.energy_count,
        input.k_point_count,
    )?;
    let mut point_solves = Vec::with_capacity(expected_points);
    for point_index in 0..expected_points {
        point_solves.push(band_structure_factor_from_kspace_rel(
            input.point_inputs[point_index],
        )?);
    }
    band_structure_factor_from_kspace_point_grid(
        point_solves,
        input.energy_count,
        input.k_point_count,
    )
}

/// Convert a FEFF search-energy/k-point `tauk` grid into FEFF-basis `G` grids.
///
/// Each `(energy,kpoint)` matrix is passed through the same
/// `structurefactor.f90` tail as [`band_structure_factor_feff_basis`], producing
/// the structure-factor grid expected by [`band_kkr_eigenvalue_grid`] and
/// [`band_kkr_band_energies`].
pub fn band_structure_factor_feff_basis_grid(
    input: BandStructureFactorFeffBasisGridInput<'_>,
) -> Result<Array4<Complex32>, BandError> {
    if input.atom_count == 0 {
        return Err(BandError::EmptyAtomList);
    }
    let block_order = angular_block_order(input.angular_lmax)?;
    let expected_order = input
        .atom_count
        .checked_mul(block_order)
        .ok_or(BandError::MatrixOrderOverflow)?;
    let (energy_count, k_point_count, rows, columns) = input.tauk_sprkkr.dim();
    if energy_count == 0
        || k_point_count == 0
        || rows != expected_order
        || columns != expected_order
    {
        return Err(BandError::InvalidStructureFactorGridShape {
            energy_count,
            k_point_count,
            rows,
            columns,
            expected: expected_order,
        });
    }
    if input.wave_numbers.len() != energy_count {
        return Err(BandError::InvalidBandTableLength {
            name: "wave_numbers",
            actual: input.wave_numbers.len(),
            expected: energy_count,
        });
    }

    let mut structure_factors =
        Array4::zeros((energy_count, k_point_count, expected_order, expected_order).f());
    for energy_index in 0..energy_count {
        let tauk_for_energy = input.tauk_sprkkr.index_axis(Axis(0), energy_index);
        for k_point_index in 0..k_point_count {
            let tauk_sprkkr = tauk_for_energy.index_axis(Axis(0), k_point_index);
            let structure_factor =
                band_structure_factor_feff_basis(BandStructureFactorFeffBasisInput {
                    tauk_sprkkr,
                    wave_number: input.wave_numbers[energy_index],
                    atom_count: input.atom_count,
                    angular_lmax: input.angular_lmax,
                    basis_transforms: input.basis_transforms,
                })?;
            for row in 0..expected_order {
                for column in 0..expected_order {
                    structure_factors[(energy_index, k_point_index, row, column)] =
                        structure_factor[(row, column)];
                }
            }
        }
    }

    Ok(structure_factors)
}

fn band_kkr_from_kspace_structure_factor(
    structure_factor: BandStructureFactorFromKspace,
    t_matrix: ArrayView2<'_, Complex32>,
    wave_number: Complex32,
) -> Result<BandKkrFromKspace, BandError> {
    let kkr_matrix = band_kkr_matrix_from_structure_factor(BandKkrMatrixInput {
        structure_factor: structure_factor.structure_factor.view(),
        t_matrix,
    })?;
    let eigenvalues = band_sorted_kkr_eigenvalues(BandSortedKkrEigenvaluesInput {
        kkr_matrix: kkr_matrix.view(),
        wave_number,
    })?;
    let positive_count = band_positive_eigenvalue_count(eigenvalues.view())?;

    Ok(BandKkrFromKspace {
        structure_factor,
        kkr_matrix,
        eigenvalues,
        positive_count,
    })
}

fn band_free_propagation_from_kspace_structure_factor(
    structure_factor: BandStructureFactorFromKspace,
    wave_number: Complex32,
) -> Result<BandKkrFromKspace, BandError> {
    let kkr_matrix = structure_factor.structure_factor.to_owned();
    let eigenvalues = band_free_propagation_eigenvalues_from_structure_factor(
        BandFreePropagationEigenvaluesFromStructureFactorInput {
            structure_factor: kkr_matrix.view(),
            wave_number,
        },
    )?;
    let positive_count = band_positive_eigenvalue_count(eigenvalues.view())?;

    Ok(BandKkrFromKspace {
        structure_factor,
        kkr_matrix,
        eigenvalues,
        positive_count,
    })
}

fn band_structure_factor_from_kspace_point_grid(
    point_solves: Vec<BandStructureFactorFromKspace>,
    energy_count: usize,
    k_point_count: usize,
) -> Result<BandStructureFactorFromKspaceGrid, BandError> {
    let first = point_solves
        .first()
        .ok_or(BandError::InvalidKkrEigenvalueGridShape {
            energy_count,
            k_point_count,
            rows: 0,
            columns: 0,
        })?;
    let rows = first.structure_factor.nrows();
    let columns = first.structure_factor.ncols();
    if rows == 0 || rows != columns {
        return Err(BandError::InvalidKkrEigenvalueGridShape {
            energy_count,
            k_point_count,
            rows,
            columns,
        });
    }

    let mut structure_factors = Array4::zeros((energy_count, k_point_count, rows, columns).f());
    for (point_index, solve) in point_solves.iter().enumerate() {
        let (actual_rows, actual_columns) = solve.structure_factor.dim();
        if actual_rows != rows || actual_columns != columns {
            return Err(BandError::MatrixShapeMismatch {
                left_name: "first_structure_factor",
                left_rows: rows,
                left_columns: columns,
                right_name: "point_structure_factor",
                right_rows: actual_rows,
                right_columns: actual_columns,
            });
        }
        let energy_index = point_index / k_point_count;
        let k_point_index = point_index % k_point_count;
        for row in 0..rows {
            for column in 0..columns {
                structure_factors[(energy_index, k_point_index, row, column)] =
                    solve.structure_factor[(row, column)];
            }
        }
    }

    Ok(BandStructureFactorFromKspaceGrid {
        point_solves,
        structure_factors,
    })
}

fn band_kkr_from_kspace_point_solves_grid(
    point_solves: Vec<BandKkrFromKspace>,
    energy_count: usize,
    k_point_count: usize,
) -> Result<BandKkrFromKspaceGrid, BandError> {
    let state_count = point_solves
        .first()
        .map(|solve| solve.eigenvalues.len())
        .ok_or(BandError::InvalidKkrEigenvalueGridShape {
            energy_count,
            k_point_count,
            rows: 0,
            columns: 0,
        })?;
    let mut eigenvalues = Array3::zeros((energy_count, k_point_count, state_count).f());
    let mut positive_counts = Array2::zeros((energy_count, k_point_count).f());

    for (point_index, solve) in point_solves.iter().enumerate() {
        if solve.eigenvalues.len() != state_count {
            return Err(BandError::MatrixShapeMismatch {
                left_name: "first_eigenvalues",
                left_rows: state_count,
                left_columns: 1,
                right_name: "point_eigenvalues",
                right_rows: solve.eigenvalues.len(),
                right_columns: 1,
            });
        }
        let energy_index = point_index / k_point_count;
        let k_point_index = point_index % k_point_count;
        positive_counts[(energy_index, k_point_index)] = solve.positive_count;
        for eigenvalue_index in 0..state_count {
            eigenvalues[(energy_index, k_point_index, eigenvalue_index)] =
                solve.eigenvalues[eigenvalue_index];
        }
    }

    Ok(BandKkrFromKspaceGrid {
        point_solves,
        eigenvalues,
        positive_counts,
    })
}

fn band_kkr_band_energies_from_kspace_grid(
    grid: BandKkrFromKspaceGrid,
    energy_min_hartree: Real,
    energy_step_hartree: Real,
) -> Result<BandKkrBandEnergiesFromKspaceGrid, BandError> {
    let band_energies = band_energies_from_positive_counts(BandEnergiesFromPositiveCountsInput {
        positive_counts: grid.positive_counts.view(),
        energy_min_hartree,
        energy_step_hartree,
    })?;
    let BandKkrFromKspaceGrid {
        point_solves,
        eigenvalues,
        positive_counts,
    } = grid;

    Ok(BandKkrBandEnergiesFromKspaceGrid {
        point_solves,
        eigenvalues,
        positive_counts,
        band_energies,
    })
}

fn validate_kspace_grid_input(
    name: &'static str,
    actual_points: usize,
    energy_count: usize,
    k_point_count: usize,
) -> Result<usize, BandError> {
    if energy_count == 0 || k_point_count == 0 {
        return Err(BandError::InvalidKkrEigenvalueGridShape {
            energy_count,
            k_point_count,
            rows: 0,
            columns: 0,
        });
    }
    let expected_points = energy_count.checked_mul(k_point_count).ok_or(
        BandError::InvalidKkrEigenvalueGridShape {
            energy_count,
            k_point_count,
            rows: 0,
            columns: 0,
        },
    )?;
    if actual_points != expected_points {
        return Err(BandError::InvalidBandTableLength {
            name,
            actual: actual_points,
            expected: expected_points,
        });
    }
    Ok(expected_points)
}

fn validate_band_table_len(
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), BandError> {
    if actual == expected {
        Ok(())
    } else {
        Err(BandError::InvalidBandTableLength {
            name,
            actual,
            expected,
        })
    }
}

fn invert_complex32_matrix(
    matrix: ArrayView2<'_, Complex32>,
) -> Result<Array2<Complex32>, BandError> {
    let order = matrix.nrows();
    let mut identity = Array2::zeros((order, order).f());
    for index in 0..order {
        identity[(index, index)] = Complex32::new(1.0, 0.0);
    }

    let lu = complex32_lu_factor(matrix)?;
    Ok(complex32_lu_solve(&lu, identity.view())?)
}

fn validate_square_matrix(
    name: &'static str,
    matrix: ArrayView2<'_, Complex32>,
) -> Result<(), BandError> {
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        Err(BandError::NonSquareMatrix {
            name,
            rows: matrix.nrows(),
            columns: matrix.ncols(),
        })
    } else {
        Ok(())
    }
}

fn state_atom_index(atom: usize, atom_count: usize) -> Result<usize, BandError> {
    match atom.checked_sub(1) {
        Some(index) if index < atom_count => Ok(index),
        _ => Err(BandError::StateAtomOutOfRange { atom, atom_count }),
    }
}

fn angular_block_order(lmax: usize) -> Result<usize, BandError> {
    lmax.checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .ok_or(BandError::MatrixOrderOverflow)
}

fn structure_factor_block_order(
    input: &BandStructureFactorFeffBasisInput<'_>,
    non_spin_order: usize,
) -> Result<usize, BandError> {
    let expected_non_spin = input
        .atom_count
        .checked_mul(non_spin_order)
        .ok_or(BandError::MatrixOrderOverflow)?;
    let (rows, columns) = input.tauk_sprkkr.dim();
    if rows == expected_non_spin && columns == expected_non_spin {
        return Ok(non_spin_order);
    }

    let expected_spin_indexed = input
        .atom_count
        .checked_mul(input.basis_transforms.order)
        .ok_or(BandError::MatrixOrderOverflow)?;
    if input.basis_transforms.order != non_spin_order
        && rows == expected_spin_indexed
        && columns == expected_spin_indexed
    {
        return Ok(input.basis_transforms.order);
    }

    Err(BandError::InvalidStructureFactorShape {
        rows,
        columns,
        expected: expected_non_spin,
    })
}

fn real_to_complex_block(
    transforms: &BasisTransformMatrices,
    lmax: usize,
    non_spin_order: usize,
    block_order: usize,
) -> Result<Array2<Complex>, BandError> {
    let expected_order = non_spin_order
        .checked_mul(2)
        .ok_or(BandError::MatrixOrderOverflow)?;
    if transforms.lmax != lmax
        || transforms.order != expected_order
        || transforms.real_to_complex.dim() != (expected_order, expected_order)
    {
        return Err(BandError::InvalidBasisTransform {
            lmax,
            expected_order,
            actual_lmax: transforms.lmax,
            actual_order: transforms.order,
            rows: transforms.real_to_complex.nrows(),
            columns: transforms.real_to_complex.ncols(),
        });
    }

    if block_order == non_spin_order || block_order == transforms.order {
        Ok(Array2::from_shape_fn(
            (block_order, block_order).f(),
            |(row, column)| transforms.real_to_complex[(row, column)],
        ))
    } else {
        Err(BandError::InvalidStructureFactorShape {
            rows: block_order,
            columns: block_order,
            expected: non_spin_order,
        })
    }
}

fn conjugate_transpose_complex(matrix: ArrayView2<'_, Complex>) -> Array2<Complex> {
    Array2::from_shape_fn((matrix.ncols(), matrix.nrows()).f(), |(row, column)| {
        matrix[(column, row)].conj()
    })
}

fn spherical_index_orbital_momenta(lmax: usize) -> Vec<usize> {
    let mut momenta = Vec::new();
    for orbital in 0..=lmax {
        for _ in 0..(2 * orbital + 1) {
            momenta.push(orbital);
        }
    }
    momenta
}

fn spherical_index_orbital_momenta_for_block(
    lmax: usize,
    non_spin_order: usize,
    block_order: usize,
) -> Result<Vec<usize>, BandError> {
    let base = spherical_index_orbital_momenta(lmax);
    if block_order == non_spin_order {
        return Ok(base);
    }
    if block_order
        == non_spin_order
            .checked_mul(2)
            .ok_or(BandError::MatrixOrderOverflow)?
    {
        let mut momenta = Vec::with_capacity(block_order);
        momenta.extend_from_slice(&base);
        momenta.extend_from_slice(&base);
        return Ok(momenta);
    }
    Err(BandError::InvalidStructureFactorShape {
        rows: block_order,
        columns: block_order,
        expected: non_spin_order,
    })
}

fn complex_i_power(exponent: isize) -> Complex {
    match exponent.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

fn validate_complex_finite(name: &'static str, value: Complex) -> Result<(), BandError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(BandError::NonFiniteComplexValue {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_complex32_finite(name: &'static str, value: Complex32) -> Result<(), BandError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(BandError::NonFiniteComplexValue {
            name,
            real: value.re as Real,
            imaginary: value.im as Real,
        })
    }
}

fn validate_complex32_finite_index(
    name: &'static str,
    index: usize,
    value: Complex32,
) -> Result<(), BandError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(BandError::NonFiniteValue {
            name,
            index,
            value: value.re as Real,
        })
    }
}

fn validate_phase_active_len(active_len: usize, available: usize) -> Result<(), BandError> {
    if active_len == 0 || active_len > available {
        Err(BandError::InvalidPhaseActiveLength {
            active_len,
            available,
        })
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, index: usize, value: Real) -> Result<(), BandError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(BandError::NonFiniteValue { name, index, value })
    }
}

fn nint_positive_to_usize(value: Real) -> Result<usize, BandError> {
    if value < 0.0 {
        return Err(BandError::InvalidClippedEnergyRange {
            min_hartree: 0.0,
            max_hartree: value,
        });
    }
    let rounded = value.round();
    if rounded > (usize::MAX as Real) {
        return Err(BandError::EnergyPointCountOverflow);
    }
    Ok(rounded as usize)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, Array4, arr1, arr2};

    use super::*;
    use crate::kspace::KSpaceStrbbddInput;

    #[test]
    fn band_energy_search_mesh_matches_feff_bandtot_clipping() -> Result<(), BandError> {
        let xmu = 0.2;
        let phase = arr1(&[
            Complex::new(-5.0 / FEFF_HARTREE_EV + xmu, 0.25),
            Complex::new(10.0 / FEFF_HARTREE_EV + xmu, 0.10),
            Complex::new(25.0 / FEFF_HARTREE_EV + xmu, 0.05),
        ]);
        let mesh = band_energy_search_mesh(BandEnergySearchMeshInput {
            requested_min_ev: -10.0,
            requested_max_ev: 100.0,
            requested_step_ev: 10.0,
            phase_energies_hartree: phase.view(),
            phase_active_len: 3,
            fermi_level_hartree: xmu,
        })?;

        assert_eq!(mesh.point_count(), 4);
        assert_close(mesh.min_hartree, -5.0 / FEFF_HARTREE_EV);
        assert_close(mesh.max_hartree, 25.0 / FEFF_HARTREE_EV);
        assert_close(mesh.step_hartree, 10.0 / FEFF_HARTREE_EV);
        assert_array_close(
            mesh.energies_hartree.view(),
            arr1(&[
                -5.0 / FEFF_HARTREE_EV,
                5.0 / FEFF_HARTREE_EV,
                15.0 / FEFF_HARTREE_EV,
                25.0 / FEFF_HARTREE_EV,
            ])
            .view(),
        );
        Ok(())
    }

    #[test]
    fn band_energy_search_mesh_recomputes_step_after_nint() -> Result<(), BandError> {
        let phase = arr1(&[
            Complex::new(0.0, 0.0),
            Complex::new(10.0 / FEFF_HARTREE_EV, 0.0),
        ]);
        let mesh = band_energy_search_mesh(BandEnergySearchMeshInput {
            requested_min_ev: 0.0,
            requested_max_ev: 10.0,
            requested_step_ev: 4.0,
            phase_energies_hartree: phase.view(),
            phase_active_len: 2,
            fermi_level_hartree: 0.0,
        })?;

        assert_eq!(mesh.point_count(), 4);
        assert_close(mesh.step_hartree, (10.0 / 3.0) / FEFF_HARTREE_EV);
        assert_close(mesh.energies_hartree[3], 10.0 / FEFF_HARTREE_EV);
        Ok(())
    }

    #[test]
    fn band_phase_search_interpolation_matches_feff_terp_loop() -> Result<(), BandError> {
        let source_energies = arr1(&[0.0, 1.0, 2.0, 3.0]);
        let search_energies = arr1(&[2.0]);
        let reference_energies = Array2::from_shape_fn((4, 2).f(), |(_, spin)| {
            Complex::new(0.25 * spin as Real, 0.0)
        });
        let mut phase_shifts = Array4::zeros((4, 3, 2, 2).f());
        for source in 0..4 {
            let x: Real = source_energies[source];
            for slot in 0..3 {
                for spin in 0..2 {
                    for potential in 0..2 {
                        phase_shifts[(source, slot, spin, potential)] = Complex::new(
                            x.powi(3) + slot as Real + 10.0 * spin as Real,
                            -x.powi(2) - 100.0 * potential as Real,
                        );
                    }
                }
            }
        }

        let interpolated = band_phase_search_interpolation(BandPhaseSearchInterpolationInput {
            search_energies_hartree: search_energies.view(),
            source_energies_hartree: source_energies.view(),
            source_reference_energies_hartree: reference_energies.view(),
            source_phase_shifts: phase_shifts.view(),
            potential_lmax: &[0, 1],
            interpolation_order: 3,
        })?;

        assert_eq!(interpolated.signed_l_offset, 1);
        assert_eq!(interpolated.reference_energies_hartree.dim(), (1, 2));
        assert_complex_close(
            interpolated.reference_energies_hartree[(0, 1)],
            Complex::new(0.25, 0.0),
        );
        assert_complex32_close(interpolated.wave_numbers[(0, 0)], Complex32::new(2.0, 0.0));
        assert_complex32_close(
            interpolated.wave_numbers[(0, 1)],
            Complex32::new(3.5_f32.sqrt(), 0.0),
        );

        assert_complex32_close(
            interpolated.phase_shifts[(0, 0, 1, 0)],
            Complex32::new(9.0, -4.0),
        );
        assert_complex32_close(
            interpolated.phase_shifts[(0, 1, 2, 1)],
            Complex32::new(20.0, -104.0),
        );
        assert_complex32_close(
            interpolated.phase_shifts[(0, 0, 0, 0)],
            Complex32::new(0.0, 0.0),
        );
        assert_complex32_close(
            interpolated.phase_shifts[(0, 0, 2, 0)],
            Complex32::new(0.0, 0.0),
        );
        Ok(())
    }

    #[test]
    fn band_phase_search_interpolation_rejects_invalid_shapes() {
        let source_energies = arr1(&[0.0, 1.0, 2.0, 3.0]);
        let search_energies = arr1(&[1.0]);
        let reference_energies = Array2::zeros((4, 1).f());
        let even_signed_l = Array4::zeros((4, 2, 1, 1).f());

        assert_eq!(
            band_phase_search_interpolation(BandPhaseSearchInterpolationInput {
                search_energies_hartree: search_energies.view(),
                source_energies_hartree: source_energies.view(),
                source_reference_energies_hartree: reference_energies.view(),
                source_phase_shifts: even_signed_l.view(),
                potential_lmax: &[0],
                interpolation_order: 3,
            }),
            Err(BandError::InvalidPhaseSearchShape {
                source_energy_count: 4,
                signed_l_count: 2,
                spin_count: 1,
                potential_count: 1,
            })
        );

        let phase_shifts = Array4::zeros((4, 3, 1, 1).f());
        assert_eq!(
            band_phase_search_interpolation(BandPhaseSearchInterpolationInput {
                search_energies_hartree: search_energies.view(),
                source_energies_hartree: source_energies.view(),
                source_reference_energies_hartree: reference_energies.view(),
                source_phase_shifts: phase_shifts.view(),
                potential_lmax: &[2],
                interpolation_order: 3,
            }),
            Err(BandError::InvalidPhaseSearchPotentialLmax {
                potential: 0,
                lmax: 2,
                signed_l_offset: 1,
            })
        );
    }

    #[test]
    fn band_energies_from_positive_counts_matches_feff_loop() -> Result<(), BandError> {
        let counts = arr2(&[[0_usize, 1], [1, 1], [3, 0], [2, 2]]);
        let bands = band_energies_from_positive_counts(BandEnergiesFromPositiveCountsInput {
            positive_counts: counts.view(),
            energy_min_hartree: 1.0,
            energy_step_hartree: 0.5,
        })?;

        assert_eq!(bands.k_point_count(), 2);
        assert_eq!(bands.min_band_count(), 2);
        assert_eq!(bands.max_band_count(), 3);
        assert_array_close(
            bands.band_energies_hartree[0].view(),
            arr1(&[1.5, 2.0, 2.0]).view(),
        );
        assert_array_close(
            bands.band_energies_hartree[1].view(),
            arr1(&[2.5, 2.5]).view(),
        );
        Ok(())
    }

    #[test]
    fn band_lattice_t_matrix_matches_feff_fmsband_expansion()
    -> Result<(), Box<dyn std::error::Error>> {
        let phase_shifts = reference_phase_shifts();
        let spin_orbit = crate::spin_orbit_coupling_tables(2)?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        };
        let second = StateKet {
            magnetic: 0,
            spin: 2,
            ..first
        };
        let atoms = [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 1,
        }];
        let states = [second, first];

        let matrix = band_lattice_t_matrix(BandLatticeTMatrixInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 2,
            spin_selector: 0,
            phase_shifts: phase_shifts.view(),
            spin_orbit: &spin_orbit,
        })?;

        assert_eq!(matrix.dim(), (2, 2));
        assert_complex32_close(matrix[(0, 0)], Complex32::new(0.100_160_494, 0.026_909_722));
        assert_complex32_close(
            matrix[(0, 1)],
            Complex32::new(-0.087_964_38, -0.001_144_098_1),
        );
        Ok(())
    }

    #[test]
    fn band_lattice_t_matrix_grid_builds_each_search_energy()
    -> Result<(), Box<dyn std::error::Error>> {
        let phase_shifts = reference_phase_shifts();
        let mut shifted_phase_shifts = phase_shifts.clone();
        shifted_phase_shifts[(0, 4, 1)] += Complex32::new(0.01, -0.02);
        let mut phase_grid = Array4::zeros((2, 2, 5, 2).f());
        for spin in 0..2 {
            for signed_l in 0..5 {
                for potential in 0..2 {
                    phase_grid[(0, spin, signed_l, potential)] =
                        phase_shifts[(spin, signed_l, potential)];
                    phase_grid[(1, spin, signed_l, potential)] =
                        shifted_phase_shifts[(spin, signed_l, potential)];
                }
            }
        }
        let spin_orbit = crate::spin_orbit_coupling_tables(2)?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        };
        let second = StateKet {
            magnetic: 0,
            spin: 2,
            ..first
        };
        let atoms = [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 1,
        }];
        let states = [second, first];

        let matrices = band_lattice_t_matrix_grid(BandLatticeTMatrixGridInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 2,
            spin_selector: 0,
            phase_shifts: phase_grid.view(),
            spin_orbit: &spin_orbit,
        })?;
        let expected_first = band_lattice_t_matrix(BandLatticeTMatrixInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 2,
            spin_selector: 0,
            phase_shifts: phase_shifts.view(),
            spin_orbit: &spin_orbit,
        })?;
        let expected_second = band_lattice_t_matrix(BandLatticeTMatrixInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 2,
            spin_selector: 0,
            phase_shifts: shifted_phase_shifts.view(),
            spin_orbit: &spin_orbit,
        })?;

        assert_eq!(matrices.dim(), (2, 2, 2));
        for row in 0..2 {
            for column in 0..2 {
                assert_complex32_close(matrices[(0, row, column)], expected_first[(row, column)]);
                assert_complex32_close(matrices[(1, row, column)], expected_second[(row, column)]);
            }
        }
        Ok(())
    }

    #[test]
    fn band_kkr_matrix_from_structure_factor_matches_feff_kkrband_subtraction()
    -> Result<(), BandError> {
        let structure_factor = arr2(&[
            [Complex32::new(1.0, 1.0), Complex32::new(0.2, -0.1)],
            [Complex32::new(0.0, 0.3), Complex32::new(2.0, -0.5)],
        ]);
        let t_matrix = arr2(&[
            [Complex32::new(2.0, 0.0), Complex32::new(1.0, 0.0)],
            [Complex32::new(0.0, 0.0), Complex32::new(4.0, 0.0)],
        ]);

        let matrix = band_kkr_matrix_from_structure_factor(BandKkrMatrixInput {
            structure_factor: structure_factor.view(),
            t_matrix: t_matrix.view(),
        })?;

        assert_complex32_close(matrix[(0, 0)], Complex32::new(0.5, 1.0));
        assert_complex32_close(matrix[(0, 1)], Complex32::new(0.325, -0.1));
        assert_complex32_close(matrix[(1, 0)], Complex32::new(0.0, 0.3));
        assert_complex32_close(matrix[(1, 1)], Complex32::new(1.75, -0.5));
        Ok(())
    }

    #[test]
    fn band_sorted_kkr_eigenvalues_match_feff_fmsband_ordering() -> Result<(), BandError> {
        let kkr_matrix = arr2(&[
            [Complex32::new(-1.0, 0.5), Complex32::new(4.0, -1.0)],
            [Complex32::new(0.0, 0.0), Complex32::new(2.0, -0.25)],
        ]);

        let eigenvalues = band_sorted_kkr_eigenvalues(BandSortedKkrEigenvaluesInput {
            kkr_matrix: kkr_matrix.view(),
            wave_number: Complex32::new(2.0, 0.0),
        })?;

        assert_eq!(eigenvalues.len(), 2);
        assert_complex32_close(eigenvalues[0], Complex32::new(4.0, -0.5));
        assert_complex32_close(eigenvalues[1], Complex32::new(-2.0, 1.0));
        Ok(())
    }

    #[test]
    fn band_kkr_eigenvalues_from_structure_factor_composes_fmsband_path() -> Result<(), BandError> {
        let structure_factor = arr2(&[
            [Complex32::new(1.0, 1.0), Complex32::new(0.2, -0.1)],
            [Complex32::new(0.0, 0.0), Complex32::new(2.0, -0.5)],
        ]);
        let t_matrix = arr2(&[
            [Complex32::new(2.0, 0.0), Complex32::new(1.0, 0.0)],
            [Complex32::new(0.0, 0.0), Complex32::new(4.0, 0.0)],
        ]);

        let eigenvalues = band_kkr_eigenvalues_from_structure_factor(
            BandKkrEigenvaluesFromStructureFactorInput {
                structure_factor: structure_factor.view(),
                t_matrix: t_matrix.view(),
                wave_number: Complex32::new(2.0, 0.0),
            },
        )?;

        assert_eq!(eigenvalues.len(), 2);
        assert_complex32_close(eigenvalues[0], Complex32::new(3.5, -1.0));
        assert_complex32_close(eigenvalues[1], Complex32::new(1.0, 2.0));
        Ok(())
    }

    #[test]
    fn band_free_propagation_eigenvalues_skip_t_inverse_subtraction() -> Result<(), BandError> {
        let structure_factor = arr2(&[
            [Complex32::new(1.0, 1.0), Complex32::new(0.2, -0.1)],
            [Complex32::new(0.0, 0.0), Complex32::new(2.0, -0.5)],
        ]);

        let eigenvalues = band_free_propagation_eigenvalues_from_structure_factor(
            BandFreePropagationEigenvaluesFromStructureFactorInput {
                structure_factor: structure_factor.view(),
                wave_number: Complex32::new(2.0, 0.0),
            },
        )?;

        assert_eq!(eigenvalues.len(), 2);
        assert_complex32_close(eigenvalues[0], Complex32::new(4.0, -1.0));
        assert_complex32_close(eigenvalues[1], Complex32::new(2.0, 2.0));
        Ok(())
    }

    #[test]
    fn band_kkr_eigenvalue_grid_solves_counts_for_each_energy_kpoint() -> Result<(), BandError> {
        let mut structure_factors = Array4::zeros((2, 2, 2, 2).f());
        structure_factors[(0, 0, 0, 0)] = Complex32::new(3.0, 0.0);
        structure_factors[(0, 0, 1, 1)] = Complex32::new(0.0, 0.0);
        structure_factors[(0, 1, 0, 0)] = Complex32::new(0.0, 0.0);
        structure_factors[(0, 1, 1, 1)] = Complex32::new(4.0, 0.0);
        structure_factors[(1, 0, 0, 0)] = Complex32::new(2.0, 0.0);
        structure_factors[(1, 0, 1, 1)] = Complex32::new(3.0, 0.0);
        structure_factors[(1, 1, 0, 0)] = Complex32::new(-1.0, 0.0);
        structure_factors[(1, 1, 1, 1)] = Complex32::new(0.5, 0.0);

        let mut t_matrices = Array3::zeros((2, 2, 2).f());
        for energy_index in 0..2 {
            t_matrices[(energy_index, 0, 0)] = Complex32::new(1.0, 0.0);
            t_matrices[(energy_index, 1, 1)] = Complex32::new(1.0, 0.0);
        }
        let wave_numbers = arr1(&[Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)]);

        let grid = band_kkr_eigenvalue_grid(BandKkrEigenvalueGridInput {
            structure_factors: structure_factors.view(),
            t_matrices: t_matrices.view(),
            wave_numbers: wave_numbers.view(),
        })?;

        assert_eq!(grid.eigenvalues.dim(), (2, 2, 2));
        assert_eq!(grid.positive_counts, arr2(&[[1_usize, 1], [2, 0]]));
        assert_complex32_close(grid.eigenvalues[(0, 0, 0)], Complex32::new(2.0, 0.0));
        assert_complex32_close(grid.eigenvalues[(0, 0, 1)], Complex32::new(-1.0, 0.0));
        assert_complex32_close(grid.eigenvalues[(0, 1, 0)], Complex32::new(3.0, 0.0));
        assert_complex32_close(grid.eigenvalues[(0, 1, 1)], Complex32::new(-1.0, 0.0));
        assert_complex32_close(grid.eigenvalues[(1, 0, 0)], Complex32::new(4.0, 0.0));
        assert_complex32_close(grid.eigenvalues[(1, 0, 1)], Complex32::new(2.0, 0.0));
        assert_complex32_close(grid.eigenvalues[(1, 1, 0)], Complex32::new(-1.0, 0.0));
        assert_complex32_close(grid.eigenvalues[(1, 1, 1)], Complex32::new(-4.0, 0.0));
        Ok(())
    }

    #[test]
    fn band_free_propagation_eigenvalue_grid_solves_raw_structure_factors() -> Result<(), BandError>
    {
        let mut structure_factors = Array4::zeros((2, 2, 1, 1).f());
        structure_factors[(0, 0, 0, 0)] = Complex32::new(-1.0, 0.0);
        structure_factors[(0, 1, 0, 0)] = Complex32::new(2.0, 0.0);
        structure_factors[(1, 0, 0, 0)] = Complex32::new(3.0, 0.0);
        structure_factors[(1, 1, 0, 0)] = Complex32::new(-4.0, 0.0);
        let wave_numbers = arr1(&[Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)]);

        let grid = band_free_propagation_eigenvalue_grid(BandFreePropagationEigenvalueGridInput {
            structure_factors: structure_factors.view(),
            wave_numbers: wave_numbers.view(),
        })?;

        assert_eq!(grid.eigenvalues.dim(), (2, 2, 1));
        assert_eq!(grid.positive_counts, arr2(&[[0_usize, 1], [1, 0]]));
        assert_complex32_close(grid.eigenvalues[(1, 0, 0)], Complex32::new(6.0, 0.0));
        assert_complex32_close(grid.eigenvalues[(1, 1, 0)], Complex32::new(-8.0, 0.0));
        Ok(())
    }

    #[test]
    fn band_kkr_band_energies_solves_grid_and_identifies_crossings() -> Result<(), BandError> {
        let mut structure_factors = Array4::zeros((3, 2, 2, 2).f());
        structure_factors[(0, 0, 0, 0)] = Complex32::new(-1.0, 0.0);
        structure_factors[(0, 0, 1, 1)] = Complex32::new(0.0, 0.0);
        structure_factors[(1, 0, 0, 0)] = Complex32::new(2.0, 0.0);
        structure_factors[(1, 0, 1, 1)] = Complex32::new(0.0, 0.0);
        structure_factors[(2, 0, 0, 0)] = Complex32::new(3.0, 0.0);
        structure_factors[(2, 0, 1, 1)] = Complex32::new(2.0, 0.0);
        structure_factors[(0, 1, 0, 0)] = Complex32::new(3.0, 0.0);
        structure_factors[(0, 1, 1, 1)] = Complex32::new(0.0, 0.0);
        structure_factors[(1, 1, 0, 0)] = Complex32::new(1.5, 0.0);
        structure_factors[(1, 1, 1, 1)] = Complex32::new(0.0, 0.0);
        structure_factors[(2, 1, 0, 0)] = Complex32::new(2.0, 0.0);
        structure_factors[(2, 1, 1, 1)] = Complex32::new(1.5, 0.0);

        let mut t_matrices = Array3::zeros((3, 2, 2).f());
        for energy_index in 0..3 {
            t_matrices[(energy_index, 0, 0)] = Complex32::new(1.0, 0.0);
            t_matrices[(energy_index, 1, 1)] = Complex32::new(1.0, 0.0);
        }
        let wave_numbers = arr1(&[
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 0.0),
        ]);

        let solved = band_kkr_band_energies(BandKkrBandEnergiesInput {
            structure_factors: structure_factors.view(),
            t_matrices: t_matrices.view(),
            wave_numbers: wave_numbers.view(),
            energy_min_hartree: 1.0,
            energy_step_hartree: 0.5,
        })?;

        assert_eq!(solved.eigenvalues.dim(), (3, 2, 2));
        assert_eq!(
            solved.positive_counts,
            arr2(&[[0_usize, 1], [1, 1], [2, 2]])
        );
        assert_array_close(
            solved.band_energies.band_energies_hartree[0].view(),
            arr1(&[1.5, 2.0]).view(),
        );
        assert_array_close(
            solved.band_energies.band_energies_hartree[1].view(),
            arr1(&[2.0]).view(),
        );
        Ok(())
    }

    #[test]
    fn band_kkr_band_energies_from_phase_structure_grid_composes_t_matrix_grid()
    -> Result<(), Box<dyn std::error::Error>> {
        let phase_shifts = reference_phase_shifts();
        let mut shifted_phase_shifts = phase_shifts.clone();
        shifted_phase_shifts[(0, 4, 1)] += Complex32::new(0.02, 0.01);
        let mut phase_grid = Array4::zeros((2, 2, 5, 2).f());
        for spin in 0..2 {
            for signed_l in 0..5 {
                for potential in 0..2 {
                    phase_grid[(0, spin, signed_l, potential)] =
                        phase_shifts[(spin, signed_l, potential)];
                    phase_grid[(1, spin, signed_l, potential)] =
                        shifted_phase_shifts[(spin, signed_l, potential)];
                }
            }
        }
        let spin_orbit = crate::spin_orbit_coupling_tables(2)?;
        let first = StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        };
        let second = StateKet {
            magnetic: 0,
            spin: 2,
            ..first
        };
        let states = [second, first];
        let atoms = [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 1,
        }];
        let mut structure_factors = Array4::zeros((2, 2, 2, 2).f());
        structure_factors[(0, 0, 0, 0)] = Complex32::new(0.4, 0.1);
        structure_factors[(0, 0, 1, 1)] = Complex32::new(0.1, -0.2);
        structure_factors[(0, 1, 0, 0)] = Complex32::new(-0.3, 0.0);
        structure_factors[(0, 1, 1, 1)] = Complex32::new(0.6, 0.0);
        structure_factors[(1, 0, 0, 0)] = Complex32::new(0.2, -0.1);
        structure_factors[(1, 0, 1, 1)] = Complex32::new(0.9, 0.0);
        structure_factors[(1, 1, 0, 0)] = Complex32::new(-0.5, 0.2);
        structure_factors[(1, 1, 1, 1)] = Complex32::new(0.3, -0.1);
        let wave_numbers = arr1(&[Complex32::new(1.0, 0.0), Complex32::new(1.5, 0.0)]);

        let composed = band_kkr_band_energies_from_phase_structure_grid(
            BandKkrBandEnergiesFromPhaseStructureGridInput {
                structure_factors: structure_factors.view(),
                states: &states,
                atoms: &atoms,
                spin_channels: 2,
                spin_selector: 0,
                phase_shifts: phase_grid.view(),
                spin_orbit: &spin_orbit,
                wave_numbers: wave_numbers.view(),
                energy_min_hartree: 1.0,
                energy_step_hartree: 0.5,
            },
        )?;
        let expected_t_matrices = band_lattice_t_matrix_grid(BandLatticeTMatrixGridInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 2,
            spin_selector: 0,
            phase_shifts: phase_grid.view(),
            spin_orbit: &spin_orbit,
        })?;
        let expected_solved = band_kkr_band_energies(BandKkrBandEnergiesInput {
            structure_factors: structure_factors.view(),
            t_matrices: expected_t_matrices.view(),
            wave_numbers: wave_numbers.view(),
            energy_min_hartree: 1.0,
            energy_step_hartree: 0.5,
        })?;

        assert_eq!(composed.t_matrices.dim(), expected_t_matrices.dim());
        for energy_index in 0..2 {
            for row in 0..2 {
                for column in 0..2 {
                    assert_complex32_close(
                        composed.t_matrices[(energy_index, row, column)],
                        expected_t_matrices[(energy_index, row, column)],
                    );
                }
            }
        }
        assert_eq!(
            composed.solved.positive_counts,
            expected_solved.positive_counts
        );
        for energy_index in 0..2 {
            for k_point_index in 0..2 {
                for eigenvalue_index in 0..2 {
                    assert_complex32_close(
                        composed.solved.eigenvalues
                            [(energy_index, k_point_index, eigenvalue_index)],
                        expected_solved.eigenvalues
                            [(energy_index, k_point_index, eigenvalue_index)],
                    );
                }
            }
        }
        for k_point_index in 0..2 {
            assert_array_close(
                composed.solved.band_energies.band_energies_hartree[k_point_index].view(),
                expected_solved.band_energies.band_energies_hartree[k_point_index].view(),
            );
        }
        Ok(())
    }

    #[test]
    fn band_free_propagation_band_energies_identifies_crossings() -> Result<(), BandError> {
        let mut structure_factors = Array4::zeros((3, 2, 1, 1).f());
        structure_factors[(0, 0, 0, 0)] = Complex32::new(-1.0, 0.0);
        structure_factors[(1, 0, 0, 0)] = Complex32::new(1.0, 0.0);
        structure_factors[(2, 0, 0, 0)] = Complex32::new(2.0, 0.0);
        structure_factors[(0, 1, 0, 0)] = Complex32::new(2.0, 0.0);
        structure_factors[(1, 1, 0, 0)] = Complex32::new(-1.0, 0.0);
        structure_factors[(2, 1, 0, 0)] = Complex32::new(3.0, 0.0);
        let wave_numbers = arr1(&[
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 0.0),
        ]);

        let solved = band_free_propagation_band_energies(BandFreePropagationBandEnergiesInput {
            structure_factors: structure_factors.view(),
            wave_numbers: wave_numbers.view(),
            energy_min_hartree: 1.0,
            energy_step_hartree: 0.5,
        })?;

        assert_eq!(
            solved.positive_counts,
            arr2(&[[0_usize, 1], [1, 0], [1, 1]])
        );
        assert_array_close(
            solved.band_energies.band_energies_hartree[0].view(),
            arr1(&[1.5]).view(),
        );
        assert_array_close(
            solved.band_energies.band_energies_hartree[1].view(),
            arr1(&[2.0]).view(),
        );
        Ok(())
    }

    #[test]
    fn band_positive_counts_from_eigenvalues_matches_feff_bandtot_loop() -> Result<(), BandError> {
        let row = arr1(&[
            Complex32::new(2.0, 10.0),
            Complex32::new(0.0, -3.0),
            Complex32::new(-1.0, 0.0),
        ]);
        assert_eq!(band_positive_eigenvalue_count(row.view())?, 1);

        let mut eigenvalues = Array3::zeros((2, 2, 3).f());
        eigenvalues[(0, 0, 0)] = Complex32::new(2.0, 10.0);
        eigenvalues[(0, 0, 1)] = Complex32::new(0.0, -3.0);
        eigenvalues[(0, 0, 2)] = Complex32::new(-1.0, 0.0);
        eigenvalues[(0, 1, 0)] = Complex32::new(-0.25, 0.0);
        eigenvalues[(0, 1, 1)] = Complex32::new(4.0, 0.0);
        eigenvalues[(0, 1, 2)] = Complex32::new(1.0, 0.0);
        eigenvalues[(1, 0, 0)] = Complex32::new(3.0, 0.0);
        eigenvalues[(1, 0, 1)] = Complex32::new(2.0, 0.0);
        eigenvalues[(1, 0, 2)] = Complex32::new(1.0, 0.0);
        eigenvalues[(1, 1, 0)] = Complex32::new(-3.0, 0.0);
        eigenvalues[(1, 1, 1)] = Complex32::new(-2.0, 0.0);
        eigenvalues[(1, 1, 2)] = Complex32::new(-1.0, 0.0);

        let counts =
            band_positive_counts_from_eigenvalues(BandPositiveCountsFromEigenvaluesInput {
                eigenvalues: eigenvalues.view(),
            })?;

        assert_eq!(counts, arr2(&[[1_usize, 2], [3, 0]]));
        Ok(())
    }

    #[test]
    fn band_structure_factor_feff_basis_matches_feff_tail() -> Result<(), Box<dyn std::error::Error>>
    {
        let transforms = crate::basis_transform_matrices(1)?;
        let wave_number = Complex::new(2.0, 0.0);
        let mut tauk = Array2::zeros((4, 4).f());
        tauk[(0, 0)] = wave_number;
        tauk[(2, 2)] = wave_number * Complex::new(3.0, 0.0);
        tauk[(0, 2)] = wave_number;
        tauk[(2, 0)] = wave_number;

        let structure_factor =
            band_structure_factor_feff_basis(BandStructureFactorFeffBasisInput {
                tauk_sprkkr: tauk.view(),
                wave_number,
                atom_count: 1,
                angular_lmax: 1,
                basis_transforms: &transforms,
            })?;

        assert_eq!(structure_factor.dim(), (4, 4));
        assert_complex32_close(structure_factor[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(structure_factor[(2, 2)], Complex32::new(3.0, 0.0));
        assert_complex32_close(structure_factor[(0, 2)], Complex32::new(0.0, -1.0));
        assert_complex32_close(structure_factor[(2, 0)], Complex32::new(0.0, 1.0));
        Ok(())
    }

    #[test]
    fn band_structure_factor_feff_basis_accepts_full_spin_block()
    -> Result<(), Box<dyn std::error::Error>> {
        let transforms = crate::basis_transform_matrices(0)?;
        let wave_number = Complex::new(2.0, 0.0);
        let tauk = arr2(&[
            [Complex::new(2.0, 0.0), Complex::new(0.5, -1.0)],
            [Complex::new(-0.5, 1.0), Complex::new(4.0, 0.0)],
        ]);

        let structure_factor =
            band_structure_factor_feff_basis(BandStructureFactorFeffBasisInput {
                tauk_sprkkr: tauk.view(),
                wave_number,
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            })?;

        assert_eq!(structure_factor.dim(), (2, 2));
        assert_complex32_close(structure_factor[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(structure_factor[(0, 1)], Complex32::new(0.25, -0.5));
        assert_complex32_close(structure_factor[(1, 0)], Complex32::new(-0.25, 0.5));
        assert_complex32_close(structure_factor[(1, 1)], Complex32::new(2.0, 0.0));
        Ok(())
    }

    #[test]
    fn band_structure_factor_feff_basis_grid_converts_each_energy_kpoint()
    -> Result<(), Box<dyn std::error::Error>> {
        let transforms = crate::basis_transform_matrices(0)?;
        let wave_numbers = arr1(&[Complex::new(2.0, 0.0), Complex::new(1.0, 0.0)]);
        let mut tauk = Array4::zeros((2, 2, 1, 1).f());
        tauk[(0, 0, 0, 0)] = Complex::new(4.0, 0.0);
        tauk[(0, 1, 0, 0)] = Complex::new(6.0, 0.0);
        tauk[(1, 0, 0, 0)] = Complex::new(5.0, 0.0);
        tauk[(1, 1, 0, 0)] = Complex::new(1.0, 1.0);

        let structure_factors =
            band_structure_factor_feff_basis_grid(BandStructureFactorFeffBasisGridInput {
                tauk_sprkkr: tauk.view(),
                wave_numbers: wave_numbers.view(),
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            })?;

        assert_eq!(structure_factors.dim(), (2, 2, 1, 1));
        assert_complex32_close(structure_factors[(0, 0, 0, 0)], Complex32::new(2.0, 0.0));
        assert_complex32_close(structure_factors[(0, 1, 0, 0)], Complex32::new(3.0, 0.0));
        assert_complex32_close(structure_factors[(1, 0, 0, 0)], Complex32::new(5.0, 0.0));
        assert_complex32_close(structure_factors[(1, 1, 0, 0)], Complex32::new(1.0, 1.0));
        Ok(())
    }

    #[test]
    fn band_structure_factor_from_kspace_non_rel_composes_structurefactor_path()
    -> Result<(), Box<dyn std::error::Error>> {
        let transforms = crate::basis_transform_matrices(0)?;
        let reciprocal_indices = Array2::<i32>::zeros((0, 3));
        let reciprocal_pair_phases = Array2::<Complex>::zeros((0, 1));
        let d1term3 = arr1(&[Complex::new(1.0, 0.0)]);
        let qjltab = arr2(&[[1.0]]);
        let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
        let direct_indices = Array2::<i32>::zeros((0, 3));
        let direct_index_by_pair = Array2::<usize>::zeros((0, 1));
        let direct_counts = [0_usize];
        let direct_terms = Array3::<Complex>::zeros((1, 0, 1));
        let mut q_pair_sites = Array3::<usize>::zeros((1, 1, 2));
        q_pair_sites[(0, 0, 0)] = 0;
        q_pair_sites[(0, 0, 1)] = 0;
        let q_pair_counts = [1_usize];
        let site_offsets = [0_usize];
        let site_state_counts = [1_usize];
        let gaunt_counts = [1_usize];
        let gaunt_indices = [0_usize];
        let gaunt_values = [1.0];
        let cipwl = arr1(&[Complex::new(1.0, 0.0)]);

        let solved =
            band_structure_factor_from_kspace_non_rel(BandStructureFactorFromKspaceNonRelInput {
                kspace: KSpaceStrsetNonRelFromLatticeSumInput {
                    lattice_sum: KSpaceStrbbddInput {
                        k: [0.0, 0.0, 0.0],
                        lmax: 0,
                        eta: 1.0,
                        energy: Complex::new(0.0, 0.0),
                        gmax_squared: 1.0,
                        reciprocal_basis: identity_basis(),
                        reciprocal_indices: reciprocal_indices.view(),
                        reciprocal_pair_phases: reciprocal_pair_phases.view(),
                        d1term3: d1term3.view(),
                        qjltab: qjltab.view(),
                        q_pair_offsets: q_pair_offsets.view(),
                        direct_basis: identity_basis(),
                        direct_indices: direct_indices.view(),
                        direct_index_by_pair: direct_index_by_pair.view(),
                        direct_counts: &direct_counts,
                        direct_terms: direct_terms.view(),
                        d300: Complex::new(1.0, 0.0),
                    },
                    angular_state_count: 1,
                    q_pair_sites: q_pair_sites.view(),
                    q_pair_counts: &q_pair_counts,
                    site_offsets: &site_offsets,
                    site_state_counts: &site_state_counts,
                    gaunt_counts: &gaunt_counts,
                    gaunt_indices: &gaunt_indices,
                    gaunt_values: &gaunt_values,
                    cipwl: cipwl.view(),
                    wave_number: Complex::new(2.0, 0.0),
                },
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            })?;

        assert_eq!(solved.kspace.dllmmke, arr2(&[[Complex::new(1.0, 0.0)]]));
        assert_eq!(solved.kspace.taukinv, arr2(&[[Complex::new(-1.0, -2.0)]]));
        assert_complex32_close(solved.structure_factor[(0, 0)], Complex32::new(-0.5, -1.0));
        Ok(())
    }

    #[test]
    fn band_structure_factor_from_kspace_rel_composes_structurefactor_path()
    -> Result<(), Box<dyn std::error::Error>> {
        let transforms = crate::basis_transform_matrices(0)?;
        let reciprocal_indices = Array2::<i32>::zeros((0, 3));
        let reciprocal_pair_phases = Array2::<Complex>::zeros((0, 1));
        let d1term3 = arr1(&[Complex::new(1.0, 0.0)]);
        let qjltab = arr2(&[[1.0]]);
        let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
        let direct_indices = Array2::<i32>::zeros((0, 3));
        let direct_index_by_pair = Array2::<usize>::zeros((0, 1));
        let direct_counts = [0_usize];
        let direct_terms = Array3::<Complex>::zeros((1, 0, 1));
        let mut q_pair_sites = Array3::<usize>::zeros((1, 1, 2));
        q_pair_sites[(0, 0, 0)] = 0;
        q_pair_sites[(0, 0, 1)] = 0;
        let q_pair_counts = [1_usize];
        let site_offsets = [0_usize];
        let site_state_counts = [1_usize];
        let gaunt_counts = [1_usize];
        let gaunt_indices = [0_usize];
        let gaunt_values = [1.0];
        let cipwl = arr1(&[Complex::new(1.0, 0.0)]);
        let rel_component_counts = arr2(&[[1_usize], [0]]);
        let mut rel_component_indices = Array3::<usize>::zeros((1, 2, 1));
        rel_component_indices[(0, 0, 0)] = 0;
        let mut rel_component_coefficients = Array3::<Complex>::zeros((1, 2, 1));
        rel_component_coefficients[(0, 0, 0)] = Complex::new(1.0, 0.0);

        let solved =
            band_structure_factor_from_kspace_rel(BandStructureFactorFromKspaceRelInput {
                kspace: KSpaceStrsetRelFromLatticeSumInput {
                    lattice_sum: KSpaceStrbbddInput {
                        k: [0.0, 0.0, 0.0],
                        lmax: 0,
                        eta: 1.0,
                        energy: Complex::new(0.0, 0.0),
                        gmax_squared: 1.0,
                        reciprocal_basis: identity_basis(),
                        reciprocal_indices: reciprocal_indices.view(),
                        reciprocal_pair_phases: reciprocal_pair_phases.view(),
                        d1term3: d1term3.view(),
                        qjltab: qjltab.view(),
                        q_pair_offsets: q_pair_offsets.view(),
                        direct_basis: identity_basis(),
                        direct_indices: direct_indices.view(),
                        direct_index_by_pair: direct_index_by_pair.view(),
                        direct_counts: &direct_counts,
                        direct_terms: direct_terms.view(),
                        d300: Complex::new(1.0, 0.0),
                    },
                    angular_state_count: 1,
                    q_pair_sites: q_pair_sites.view(),
                    q_pair_counts: &q_pair_counts,
                    site_offsets: &site_offsets,
                    site_state_counts: &site_state_counts,
                    gaunt_counts: &gaunt_counts,
                    gaunt_indices: &gaunt_indices,
                    gaunt_values: &gaunt_values,
                    cipwl: cipwl.view(),
                    rel_component_counts: rel_component_counts.view(),
                    rel_component_indices: rel_component_indices.view(),
                    rel_component_coefficients: rel_component_coefficients.view(),
                    wave_number: Complex::new(2.0, 0.0),
                },
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            })?;

        assert_eq!(solved.kspace.dllmmke, arr2(&[[Complex::new(1.0, 0.0)]]));
        assert_eq!(solved.kspace.taukinv, arr2(&[[Complex::new(-1.0, -2.0)]]));
        assert_complex32_close(solved.structure_factor[(0, 0)], Complex32::new(-0.5, -1.0));
        Ok(())
    }

    #[test]
    fn band_structure_factor_from_kspace_non_rel_grid_assembles_feff_loop_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.non_rel_point(-4.0).structure_factor,
            fixture.non_rel_point(0.0).structure_factor,
            fixture.non_rel_point(-6.0).structure_factor,
            fixture.non_rel_point(-1.0).structure_factor,
        ];

        let grid = band_structure_factor_from_kspace_non_rel_grid(
            BandStructureFactorFromKspaceNonRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
            },
        )?;

        assert_eq!(grid.point_solves.len(), 4);
        assert_eq!(grid.structure_factors.dim(), (2, 2, 1, 1));
        assert_complex32_close(
            grid.structure_factors[(0, 0, 0, 0)],
            grid.point_solves[0].structure_factor[(0, 0)],
        );
        assert_complex32_close(
            grid.structure_factors[(0, 1, 0, 0)],
            grid.point_solves[1].structure_factor[(0, 0)],
        );
        assert_complex32_close(
            grid.structure_factors[(1, 0, 0, 0)],
            grid.point_solves[2].structure_factor[(0, 0)],
        );
        assert_complex32_close(
            grid.structure_factors[(1, 1, 0, 0)],
            grid.point_solves[3].structure_factor[(0, 0)],
        );
        Ok(())
    }

    #[test]
    fn band_structure_factor_from_kspace_rel_grid_assembles_feff_loop_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.rel_point(-4.0).structure_factor,
            fixture.rel_point(0.0).structure_factor,
            fixture.rel_point(-6.0).structure_factor,
            fixture.rel_point(-1.0).structure_factor,
        ];

        let grid = band_structure_factor_from_kspace_rel_grid(
            BandStructureFactorFromKspaceRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
            },
        )?;

        assert_eq!(grid.point_solves.len(), 4);
        assert_eq!(grid.structure_factors.dim(), (2, 2, 1, 1));
        assert_complex32_close(
            grid.structure_factors[(0, 0, 0, 0)],
            grid.point_solves[0].structure_factor[(0, 0)],
        );
        assert_complex32_close(
            grid.structure_factors[(0, 1, 0, 0)],
            grid.point_solves[1].structure_factor[(0, 0)],
        );
        assert_complex32_close(
            grid.structure_factors[(1, 0, 0, 0)],
            grid.point_solves[2].structure_factor[(0, 0)],
        );
        assert_complex32_close(
            grid.structure_factors[(1, 1, 0, 0)],
            grid.point_solves[3].structure_factor[(0, 0)],
        );
        Ok(())
    }

    #[test]
    fn band_kkr_band_energies_from_kspace_phase_non_rel_grid_composes_full_ordinary_path()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.non_rel_point(-4.0).structure_factor,
            fixture.non_rel_point(0.0).structure_factor,
            fixture.non_rel_point(-6.0).structure_factor,
            fixture.non_rel_point(-1.0).structure_factor,
        ];
        let phase_grid = one_state_phase_grid();
        let spin_orbit = crate::spin_orbit_coupling_tables(0)?;
        let states = one_state_band_states();
        let atoms = one_state_band_atoms();
        let wave_numbers = arr1(&[Complex32::new(2.0, 0.0), Complex32::new(2.0, 0.0)]);

        let composed = band_kkr_band_energies_from_kspace_phase_non_rel_grid(
            BandKkrBandEnergiesFromKspacePhaseNonRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
                states: &states,
                atoms: &atoms,
                spin_channels: 1,
                spin_selector: 0,
                phase_shifts: phase_grid.view(),
                spin_orbit: &spin_orbit,
                wave_numbers: wave_numbers.view(),
                energy_min_hartree: 1.0,
                energy_step_hartree: 0.5,
            },
        )?;
        let expected_structure_factors = band_structure_factor_from_kspace_non_rel_grid(
            BandStructureFactorFromKspaceNonRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
            },
        )?;
        let expected_solved = band_kkr_band_energies_from_phase_structure_grid(
            BandKkrBandEnergiesFromPhaseStructureGridInput {
                structure_factors: expected_structure_factors.structure_factors.view(),
                states: &states,
                atoms: &atoms,
                spin_channels: 1,
                spin_selector: 0,
                phase_shifts: phase_grid.view(),
                spin_orbit: &spin_orbit,
                wave_numbers: wave_numbers.view(),
                energy_min_hartree: 1.0,
                energy_step_hartree: 0.5,
            },
        )?;

        assert_eq!(composed.structure_factors.point_solves.len(), 4);
        assert_eq!(
            composed.structure_factors.structure_factors.dim(),
            (2, 2, 1, 1)
        );
        for energy_index in 0..2 {
            for k_point_index in 0..2 {
                assert_complex32_close(
                    composed.structure_factors.structure_factors
                        [(energy_index, k_point_index, 0, 0)],
                    expected_structure_factors.structure_factors
                        [(energy_index, k_point_index, 0, 0)],
                );
                assert_complex32_close(
                    composed.solved.solved.eigenvalues[(energy_index, k_point_index, 0)],
                    expected_solved.solved.eigenvalues[(energy_index, k_point_index, 0)],
                );
            }
        }
        for energy_index in 0..2 {
            assert_complex32_close(
                composed.solved.t_matrices[(energy_index, 0, 0)],
                expected_solved.t_matrices[(energy_index, 0, 0)],
            );
        }
        assert_eq!(
            composed.solved.solved.positive_counts,
            expected_solved.solved.positive_counts
        );
        Ok(())
    }

    #[test]
    fn band_kkr_band_energies_from_kspace_phase_rel_grid_composes_full_ordinary_path()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.rel_point(-4.0).structure_factor,
            fixture.rel_point(0.0).structure_factor,
            fixture.rel_point(-6.0).structure_factor,
            fixture.rel_point(-1.0).structure_factor,
        ];
        let phase_grid = one_state_phase_grid();
        let spin_orbit = crate::spin_orbit_coupling_tables(0)?;
        let states = one_state_band_states();
        let atoms = one_state_band_atoms();
        let wave_numbers = arr1(&[Complex32::new(2.0, 0.0), Complex32::new(2.0, 0.0)]);

        let composed = band_kkr_band_energies_from_kspace_phase_rel_grid(
            BandKkrBandEnergiesFromKspacePhaseRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
                states: &states,
                atoms: &atoms,
                spin_channels: 1,
                spin_selector: 0,
                phase_shifts: phase_grid.view(),
                spin_orbit: &spin_orbit,
                wave_numbers: wave_numbers.view(),
                energy_min_hartree: 1.0,
                energy_step_hartree: 0.5,
            },
        )?;
        let expected_structure_factors = band_structure_factor_from_kspace_rel_grid(
            BandStructureFactorFromKspaceRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
            },
        )?;
        let expected_solved = band_kkr_band_energies_from_phase_structure_grid(
            BandKkrBandEnergiesFromPhaseStructureGridInput {
                structure_factors: expected_structure_factors.structure_factors.view(),
                states: &states,
                atoms: &atoms,
                spin_channels: 1,
                spin_selector: 0,
                phase_shifts: phase_grid.view(),
                spin_orbit: &spin_orbit,
                wave_numbers: wave_numbers.view(),
                energy_min_hartree: 1.0,
                energy_step_hartree: 0.5,
            },
        )?;

        assert_eq!(composed.structure_factors.point_solves.len(), 4);
        assert_eq!(
            composed.structure_factors.structure_factors.dim(),
            (2, 2, 1, 1)
        );
        for energy_index in 0..2 {
            for k_point_index in 0..2 {
                assert_complex32_close(
                    composed.structure_factors.structure_factors
                        [(energy_index, k_point_index, 0, 0)],
                    expected_structure_factors.structure_factors
                        [(energy_index, k_point_index, 0, 0)],
                );
                assert_complex32_close(
                    composed.solved.solved.eigenvalues[(energy_index, k_point_index, 0)],
                    expected_solved.solved.eigenvalues[(energy_index, k_point_index, 0)],
                );
            }
        }
        assert_eq!(
            composed.solved.solved.positive_counts,
            expected_solved.solved.positive_counts
        );
        Ok(())
    }

    #[test]
    fn band_kkr_from_kspace_non_rel_solves_one_fmsband_point()
    -> Result<(), Box<dyn std::error::Error>> {
        let transforms = crate::basis_transform_matrices(0)?;
        let reciprocal_indices = Array2::<i32>::zeros((0, 3));
        let reciprocal_pair_phases = Array2::<Complex>::zeros((0, 1));
        let d1term3 = arr1(&[Complex::new(1.0, 0.0)]);
        let qjltab = arr2(&[[1.0]]);
        let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
        let direct_indices = Array2::<i32>::zeros((0, 3));
        let direct_index_by_pair = Array2::<usize>::zeros((0, 1));
        let direct_counts = [0_usize];
        let direct_terms = Array3::<Complex>::zeros((1, 0, 1));
        let mut q_pair_sites = Array3::<usize>::zeros((1, 1, 2));
        q_pair_sites[(0, 0, 0)] = 0;
        q_pair_sites[(0, 0, 1)] = 0;
        let q_pair_counts = [1_usize];
        let site_offsets = [0_usize];
        let site_state_counts = [1_usize];
        let gaunt_counts = [1_usize];
        let gaunt_indices = [0_usize];
        let gaunt_values = [1.0];
        let cipwl = arr1(&[Complex::new(1.0, 0.0)]);
        let t_matrix = arr2(&[[Complex32::new(1.0, 0.0)]]);

        let solved = band_kkr_from_kspace_non_rel(BandKkrFromKspaceNonRelInput {
            structure_factor: BandStructureFactorFromKspaceNonRelInput {
                kspace: KSpaceStrsetNonRelFromLatticeSumInput {
                    lattice_sum: KSpaceStrbbddInput {
                        k: [0.0, 0.0, 0.0],
                        lmax: 0,
                        eta: 1.0,
                        energy: Complex::new(0.0, 0.0),
                        gmax_squared: 1.0,
                        reciprocal_basis: identity_basis(),
                        reciprocal_indices: reciprocal_indices.view(),
                        reciprocal_pair_phases: reciprocal_pair_phases.view(),
                        d1term3: d1term3.view(),
                        qjltab: qjltab.view(),
                        q_pair_offsets: q_pair_offsets.view(),
                        direct_basis: identity_basis(),
                        direct_indices: direct_indices.view(),
                        direct_index_by_pair: direct_index_by_pair.view(),
                        direct_counts: &direct_counts,
                        direct_terms: direct_terms.view(),
                        d300: Complex::new(-4.0, 0.0),
                    },
                    angular_state_count: 1,
                    q_pair_sites: q_pair_sites.view(),
                    q_pair_counts: &q_pair_counts,
                    site_offsets: &site_offsets,
                    site_state_counts: &site_state_counts,
                    gaunt_counts: &gaunt_counts,
                    gaunt_indices: &gaunt_indices,
                    gaunt_values: &gaunt_values,
                    cipwl: cipwl.view(),
                    wave_number: Complex::new(2.0, 0.0),
                },
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            },
            t_matrix: t_matrix.view(),
            wave_number: Complex32::new(2.0, 0.0),
        })?;

        assert_eq!(
            solved.structure_factor.kspace.dllmmke,
            arr2(&[[Complex::new(-4.0, 0.0)]])
        );
        assert_eq!(
            solved.structure_factor.kspace.taukinv,
            arr2(&[[Complex::new(4.0, -2.0)]])
        );
        assert_complex32_close(
            solved.structure_factor.structure_factor[(0, 0)],
            Complex32::new(2.0, -1.0),
        );
        assert_complex32_close(solved.kkr_matrix[(0, 0)], Complex32::new(1.0, -1.0));
        assert_complex32_close(solved.eigenvalues[0], Complex32::new(2.0, -2.0));
        assert_eq!(solved.positive_count, 1);
        Ok(())
    }

    #[test]
    fn band_kkr_from_kspace_rel_solves_one_fmsband_point() -> Result<(), Box<dyn std::error::Error>>
    {
        let transforms = crate::basis_transform_matrices(0)?;
        let reciprocal_indices = Array2::<i32>::zeros((0, 3));
        let reciprocal_pair_phases = Array2::<Complex>::zeros((0, 1));
        let d1term3 = arr1(&[Complex::new(1.0, 0.0)]);
        let qjltab = arr2(&[[1.0]]);
        let q_pair_offsets = arr2(&[[0.0, 0.0, 0.0]]);
        let direct_indices = Array2::<i32>::zeros((0, 3));
        let direct_index_by_pair = Array2::<usize>::zeros((0, 1));
        let direct_counts = [0_usize];
        let direct_terms = Array3::<Complex>::zeros((1, 0, 1));
        let mut q_pair_sites = Array3::<usize>::zeros((1, 1, 2));
        q_pair_sites[(0, 0, 0)] = 0;
        q_pair_sites[(0, 0, 1)] = 0;
        let q_pair_counts = [1_usize];
        let site_offsets = [0_usize];
        let site_state_counts = [1_usize];
        let gaunt_counts = [1_usize];
        let gaunt_indices = [0_usize];
        let gaunt_values = [1.0];
        let cipwl = arr1(&[Complex::new(1.0, 0.0)]);
        let rel_component_counts = arr2(&[[1_usize], [0]]);
        let mut rel_component_indices = Array3::<usize>::zeros((1, 2, 1));
        rel_component_indices[(0, 0, 0)] = 0;
        let mut rel_component_coefficients = Array3::<Complex>::zeros((1, 2, 1));
        rel_component_coefficients[(0, 0, 0)] = Complex::new(1.0, 0.0);
        let t_matrix = arr2(&[[Complex32::new(1.0, 0.0)]]);

        let solved = band_kkr_from_kspace_rel(BandKkrFromKspaceRelInput {
            structure_factor: BandStructureFactorFromKspaceRelInput {
                kspace: KSpaceStrsetRelFromLatticeSumInput {
                    lattice_sum: KSpaceStrbbddInput {
                        k: [0.0, 0.0, 0.0],
                        lmax: 0,
                        eta: 1.0,
                        energy: Complex::new(0.0, 0.0),
                        gmax_squared: 1.0,
                        reciprocal_basis: identity_basis(),
                        reciprocal_indices: reciprocal_indices.view(),
                        reciprocal_pair_phases: reciprocal_pair_phases.view(),
                        d1term3: d1term3.view(),
                        qjltab: qjltab.view(),
                        q_pair_offsets: q_pair_offsets.view(),
                        direct_basis: identity_basis(),
                        direct_indices: direct_indices.view(),
                        direct_index_by_pair: direct_index_by_pair.view(),
                        direct_counts: &direct_counts,
                        direct_terms: direct_terms.view(),
                        d300: Complex::new(-4.0, 0.0),
                    },
                    angular_state_count: 1,
                    q_pair_sites: q_pair_sites.view(),
                    q_pair_counts: &q_pair_counts,
                    site_offsets: &site_offsets,
                    site_state_counts: &site_state_counts,
                    gaunt_counts: &gaunt_counts,
                    gaunt_indices: &gaunt_indices,
                    gaunt_values: &gaunt_values,
                    cipwl: cipwl.view(),
                    rel_component_counts: rel_component_counts.view(),
                    rel_component_indices: rel_component_indices.view(),
                    rel_component_coefficients: rel_component_coefficients.view(),
                    wave_number: Complex::new(2.0, 0.0),
                },
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            },
            t_matrix: t_matrix.view(),
            wave_number: Complex32::new(2.0, 0.0),
        })?;

        assert_eq!(
            solved.structure_factor.kspace.dllmmke,
            arr2(&[[Complex::new(-4.0, 0.0)]])
        );
        assert_eq!(
            solved.structure_factor.kspace.taukinv,
            arr2(&[[Complex::new(4.0, -2.0)]])
        );
        assert_complex32_close(
            solved.structure_factor.structure_factor[(0, 0)],
            Complex32::new(2.0, -1.0),
        );
        assert_complex32_close(solved.kkr_matrix[(0, 0)], Complex32::new(1.0, -1.0));
        assert_complex32_close(solved.eigenvalues[0], Complex32::new(2.0, -2.0));
        assert_eq!(solved.positive_count, 1);
        Ok(())
    }

    #[test]
    fn band_kkr_from_kspace_non_rel_grid_solves_feff_loop_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.non_rel_point(-4.0),
            fixture.non_rel_point(0.0),
            fixture.non_rel_point(-6.0),
            fixture.non_rel_point(-1.0),
        ];

        let grid = band_kkr_from_kspace_non_rel_grid(BandKkrFromKspaceNonRelGridInput {
            point_inputs: &point_inputs,
            energy_count: 2,
            k_point_count: 2,
        })?;

        assert_eq!(grid.point_solves.len(), 4);
        assert_eq!(grid.eigenvalues.dim(), (2, 2, 1));
        assert_eq!(grid.positive_counts, arr2(&[[1_usize, 0], [1, 0]]));
        assert_complex32_close(grid.eigenvalues[(0, 0, 0)], Complex32::new(2.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(0, 1, 0)], Complex32::new(-2.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(1, 0, 0)], Complex32::new(4.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(1, 1, 0)], Complex32::new(-1.0, -2.0));
        Ok(())
    }

    #[test]
    fn band_kkr_from_kspace_rel_grid_solves_feff_loop_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.rel_point(-4.0),
            fixture.rel_point(0.0),
            fixture.rel_point(-6.0),
            fixture.rel_point(-1.0),
        ];

        let grid = band_kkr_from_kspace_rel_grid(BandKkrFromKspaceRelGridInput {
            point_inputs: &point_inputs,
            energy_count: 2,
            k_point_count: 2,
        })?;

        assert_eq!(grid.point_solves.len(), 4);
        assert_eq!(grid.eigenvalues.dim(), (2, 2, 1));
        assert_eq!(grid.positive_counts, arr2(&[[1_usize, 0], [1, 0]]));
        assert_complex32_close(grid.eigenvalues[(0, 0, 0)], Complex32::new(2.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(0, 1, 0)], Complex32::new(-2.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(1, 0, 0)], Complex32::new(4.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(1, 1, 0)], Complex32::new(-1.0, -2.0));
        Ok(())
    }

    #[test]
    fn band_free_propagation_from_kspace_non_rel_grid_solves_feff_loop_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.non_rel_freeprop_point(-4.0),
            fixture.non_rel_freeprop_point(0.0),
            fixture.non_rel_freeprop_point(-6.0),
            fixture.non_rel_freeprop_point(1.0),
        ];

        let grid = band_free_propagation_from_kspace_non_rel_grid(
            BandFreePropagationFromKspaceNonRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
            },
        )?;

        assert_eq!(grid.point_solves.len(), 4);
        assert_eq!(grid.eigenvalues.dim(), (2, 2, 1));
        assert_eq!(grid.positive_counts, arr2(&[[1_usize, 0], [1, 0]]));
        assert_complex32_close(grid.eigenvalues[(0, 0, 0)], Complex32::new(4.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(0, 1, 0)], Complex32::new(0.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(1, 0, 0)], Complex32::new(6.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(1, 1, 0)], Complex32::new(-1.0, -2.0));
        Ok(())
    }

    #[test]
    fn band_free_propagation_from_kspace_rel_grid_solves_feff_loop_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.rel_freeprop_point(-4.0),
            fixture.rel_freeprop_point(0.0),
            fixture.rel_freeprop_point(-6.0),
            fixture.rel_freeprop_point(1.0),
        ];

        let grid = band_free_propagation_from_kspace_rel_grid(
            BandFreePropagationFromKspaceRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
            },
        )?;

        assert_eq!(grid.point_solves.len(), 4);
        assert_eq!(grid.eigenvalues.dim(), (2, 2, 1));
        assert_eq!(grid.positive_counts, arr2(&[[1_usize, 0], [1, 0]]));
        assert_complex32_close(grid.eigenvalues[(0, 0, 0)], Complex32::new(4.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(0, 1, 0)], Complex32::new(0.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(1, 0, 0)], Complex32::new(6.0, -2.0));
        assert_complex32_close(grid.eigenvalues[(1, 1, 0)], Complex32::new(-1.0, -2.0));
        Ok(())
    }

    #[test]
    fn band_kkr_band_energies_from_kspace_non_rel_grid_identifies_rows()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.non_rel_point(0.0),
            fixture.non_rel_point(-4.0),
            fixture.non_rel_point(-4.0),
            fixture.non_rel_point(-6.0),
        ];

        let solved = band_kkr_band_energies_from_kspace_non_rel_grid(
            BandKkrBandEnergiesFromKspaceNonRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
                energy_min_hartree: 1.0,
                energy_step_hartree: 0.5,
            },
        )?;

        assert_eq!(solved.positive_counts, arr2(&[[0_usize, 1], [1, 1]]));
        assert_array_close(
            solved.band_energies.band_energies_hartree[0].view(),
            arr1(&[1.5]).view(),
        );
        assert!(solved.band_energies.band_energies_hartree[1].is_empty());
        Ok(())
    }

    #[test]
    fn band_kkr_band_energies_from_kspace_rel_grid_identifies_rows()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.rel_point(0.0),
            fixture.rel_point(-4.0),
            fixture.rel_point(-4.0),
            fixture.rel_point(-6.0),
        ];

        let solved = band_kkr_band_energies_from_kspace_rel_grid(
            BandKkrBandEnergiesFromKspaceRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
                energy_min_hartree: 1.0,
                energy_step_hartree: 0.5,
            },
        )?;

        assert_eq!(solved.positive_counts, arr2(&[[0_usize, 1], [1, 1]]));
        assert_array_close(
            solved.band_energies.band_energies_hartree[0].view(),
            arr1(&[1.5]).view(),
        );
        assert!(solved.band_energies.band_energies_hartree[1].is_empty());
        Ok(())
    }

    #[test]
    fn band_free_propagation_band_energies_from_kspace_non_rel_grid_identifies_rows()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.non_rel_freeprop_point(0.0),
            fixture.non_rel_freeprop_point(-4.0),
            fixture.non_rel_freeprop_point(-4.0),
            fixture.non_rel_freeprop_point(-6.0),
        ];

        let solved = band_free_propagation_band_energies_from_kspace_non_rel_grid(
            BandFreePropagationBandEnergiesFromKspaceNonRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
                energy_min_hartree: 1.0,
                energy_step_hartree: 0.5,
            },
        )?;

        assert_eq!(solved.positive_counts, arr2(&[[0_usize, 1], [1, 1]]));
        assert_array_close(
            solved.band_energies.band_energies_hartree[0].view(),
            arr1(&[1.5]).view(),
        );
        assert!(solved.band_energies.band_energies_hartree[1].is_empty());
        Ok(())
    }

    #[test]
    fn band_free_propagation_band_energies_from_kspace_rel_grid_identifies_rows()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture = SingleStateKspaceFixture::new()?;
        let point_inputs = [
            fixture.rel_freeprop_point(0.0),
            fixture.rel_freeprop_point(-4.0),
            fixture.rel_freeprop_point(-4.0),
            fixture.rel_freeprop_point(-6.0),
        ];

        let solved = band_free_propagation_band_energies_from_kspace_rel_grid(
            BandFreePropagationBandEnergiesFromKspaceRelGridInput {
                point_inputs: &point_inputs,
                energy_count: 2,
                k_point_count: 2,
                energy_min_hartree: 1.0,
                energy_step_hartree: 0.5,
            },
        )?;

        assert_eq!(solved.positive_counts, arr2(&[[0_usize, 1], [1, 1]]));
        assert_array_close(
            solved.band_energies.band_energies_hartree[0].view(),
            arr1(&[1.5]).view(),
        );
        assert!(solved.band_energies.band_energies_hartree[1].is_empty());
        Ok(())
    }

    #[test]
    fn band_helpers_reject_invalid_inputs() -> Result<(), Box<dyn std::error::Error>> {
        let phase = arr1(&[Complex::new(0.0, 0.0)]);
        assert_eq!(
            band_energy_search_mesh(BandEnergySearchMeshInput {
                requested_min_ev: 0.0,
                requested_max_ev: 1.0,
                requested_step_ev: 0.0,
                phase_energies_hartree: phase.view(),
                phase_active_len: 1,
                fermi_level_hartree: 0.0,
            }),
            Err(BandError::InvalidRequestedEnergyStep { step_ev: 0.0 })
        );
        assert_eq!(
            band_energy_search_mesh(BandEnergySearchMeshInput {
                requested_min_ev: 0.0,
                requested_max_ev: 1.0,
                requested_step_ev: 1.0,
                phase_energies_hartree: phase.view(),
                phase_active_len: 0,
                fermi_level_hartree: 0.0,
            }),
            Err(BandError::InvalidPhaseActiveLength {
                active_len: 0,
                available: 1
            })
        );

        let counts = arr2(&[[0_usize], [1]]);
        let short_counts = arr2(&[[0_usize]]);
        assert_eq!(
            band_energies_from_positive_counts(BandEnergiesFromPositiveCountsInput {
                positive_counts: short_counts.view(),
                energy_min_hartree: 0.0,
                energy_step_hartree: 1.0,
            }),
            Err(BandError::InvalidPositiveCountShape {
                rows: 1,
                columns: 1
            })
        );
        assert_eq!(
            band_energies_from_positive_counts(BandEnergiesFromPositiveCountsInput {
                positive_counts: counts.view(),
                energy_min_hartree: 0.0,
                energy_step_hartree: 0.0,
            }),
            Err(BandError::InvalidSearchEnergyStep { step_hartree: 0.0 })
        );

        let phase_shifts = reference_phase_shifts();
        let spin_orbit = crate::spin_orbit_coupling_tables(2)?;
        assert_eq!(
            band_lattice_t_matrix(BandLatticeTMatrixInput {
                states: &[],
                atoms: &[],
                spin_channels: 2,
                spin_selector: 0,
                phase_shifts: phase_shifts.view(),
                spin_orbit: &spin_orbit,
            }),
            Err(BandError::EmptyStateList)
        );
        assert_eq!(
            band_lattice_t_matrix(BandLatticeTMatrixInput {
                states: &[StateKet {
                    atom: 2,
                    angular_momentum: 0,
                    magnetic: 0,
                    spin: 1,
                }],
                atoms: &[FmsAtom {
                    position: [0.0, 0.0, 0.0],
                    potential: 1,
                }],
                spin_channels: 2,
                spin_selector: 0,
                phase_shifts: phase_shifts.view(),
                spin_orbit: &spin_orbit,
            }),
            Err(BandError::StateAtomOutOfRange {
                atom: 2,
                atom_count: 1
            })
        );
        let valid_state = [StateKet {
            atom: 1,
            angular_momentum: 0,
            magnetic: 0,
            spin: 1,
        }];
        let valid_atom = [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        }];
        let empty_phase_grid = Array4::zeros((0, 1, 1, 1).f());
        assert_eq!(
            band_lattice_t_matrix_grid(BandLatticeTMatrixGridInput {
                states: &valid_state,
                atoms: &valid_atom,
                spin_channels: 1,
                spin_selector: 0,
                phase_shifts: empty_phase_grid.view(),
                spin_orbit: &spin_orbit,
            }),
            Err(BandError::InvalidLatticeTMatrixGridShape {
                energy_count: 0,
                spin_count: 1,
                signed_l_count: 1,
                potential_count: 1,
            })
        );
        let wrong_spin_phase_grid = Array4::zeros((1, 2, 1, 1).f());
        assert_eq!(
            band_lattice_t_matrix_grid(BandLatticeTMatrixGridInput {
                states: &valid_state,
                atoms: &valid_atom,
                spin_channels: 1,
                spin_selector: 0,
                phase_shifts: wrong_spin_phase_grid.view(),
                spin_orbit: &spin_orbit,
            }),
            Err(BandError::InvalidBandTableLength {
                name: "phase_shifts spin",
                actual: 2,
                expected: 1,
            })
        );

        let non_square = arr2(&[[Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)]]);
        assert_eq!(
            band_kkr_matrix_from_structure_factor(BandKkrMatrixInput {
                structure_factor: non_square.view(),
                t_matrix: non_square.view(),
            }),
            Err(BandError::NonSquareMatrix {
                name: "structure_factor",
                rows: 1,
                columns: 2
            })
        );
        assert_eq!(
            band_sorted_kkr_eigenvalues(BandSortedKkrEigenvaluesInput {
                kkr_matrix: non_square.view(),
                wave_number: Complex32::new(1.0, 0.0),
            }),
            Err(BandError::NonSquareMatrix {
                name: "kkr_matrix",
                rows: 1,
                columns: 2
            })
        );
        let empty_grid = Array4::zeros((0, 1, 1, 1).f());
        let one_energy_t = Array3::zeros((1, 1, 1).f());
        let one_wave_number = arr1(&[Complex32::new(1.0, 0.0)]);
        assert_eq!(
            band_kkr_eigenvalue_grid(BandKkrEigenvalueGridInput {
                structure_factors: empty_grid.view(),
                t_matrices: one_energy_t.view(),
                wave_numbers: one_wave_number.view(),
            }),
            Err(BandError::InvalidKkrEigenvalueGridShape {
                energy_count: 0,
                k_point_count: 1,
                rows: 1,
                columns: 1
            })
        );
        let structure_factors = Array4::zeros((2, 1, 1, 1).f());
        assert_eq!(
            band_kkr_eigenvalue_grid(BandKkrEigenvalueGridInput {
                structure_factors: structure_factors.view(),
                t_matrices: one_energy_t.view(),
                wave_numbers: one_wave_number.view(),
            }),
            Err(BandError::InvalidBandTableLength {
                name: "t_matrices",
                actual: 1,
                expected: 2
            })
        );
        let two_energy_t = Array3::zeros((2, 1, 1).f());
        assert_eq!(
            band_kkr_eigenvalue_grid(BandKkrEigenvalueGridInput {
                structure_factors: structure_factors.view(),
                t_matrices: two_energy_t.view(),
                wave_numbers: one_wave_number.view(),
            }),
            Err(BandError::InvalidBandTableLength {
                name: "wave_numbers",
                actual: 1,
                expected: 2
            })
        );
        let empty_eigenvalues = arr1(&[] as &[Complex32]);
        assert_eq!(
            band_positive_eigenvalue_count(empty_eigenvalues.view()),
            Err(BandError::EmptyEigenvalueList)
        );
        let empty_cube = Array3::zeros((1, 0, 1).f());
        assert_eq!(
            band_positive_counts_from_eigenvalues(BandPositiveCountsFromEigenvaluesInput {
                eigenvalues: empty_cube.view(),
            }),
            Err(BandError::InvalidEigenvalueCubeShape {
                energy_count: 1,
                k_point_count: 0,
                eigenvalue_count: 1
            })
        );

        let transforms = crate::basis_transform_matrices(0)?;
        let tauk = arr2(&[[Complex::new(1.0, 0.0)]]);
        assert_eq!(
            band_structure_factor_feff_basis(BandStructureFactorFeffBasisInput {
                tauk_sprkkr: tauk.view(),
                wave_number: Complex::new(0.0, 0.0),
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            }),
            Err(BandError::ZeroWaveNumber)
        );
        let wrong_tauk = Array2::zeros((1, 2).f());
        assert_eq!(
            band_structure_factor_feff_basis(BandStructureFactorFeffBasisInput {
                tauk_sprkkr: wrong_tauk.view(),
                wave_number: Complex::new(1.0, 0.0),
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            }),
            Err(BandError::InvalidStructureFactorShape {
                rows: 1,
                columns: 2,
                expected: 1
            })
        );
        let empty_tauk_grid = Array4::zeros((0, 1, 1, 1).f());
        assert_eq!(
            band_structure_factor_feff_basis_grid(BandStructureFactorFeffBasisGridInput {
                tauk_sprkkr: empty_tauk_grid.view(),
                wave_numbers: arr1(&[] as &[Complex]).view(),
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            }),
            Err(BandError::InvalidStructureFactorGridShape {
                energy_count: 0,
                k_point_count: 1,
                rows: 1,
                columns: 1,
                expected: 1
            })
        );
        let tauk_grid = Array4::zeros((2, 1, 1, 1).f());
        assert_eq!(
            band_structure_factor_feff_basis_grid(BandStructureFactorFeffBasisGridInput {
                tauk_sprkkr: tauk_grid.view(),
                wave_numbers: arr1(&[Complex::new(1.0, 0.0)]).view(),
                atom_count: 1,
                angular_lmax: 0,
                basis_transforms: &transforms,
            }),
            Err(BandError::InvalidBandTableLength {
                name: "wave_numbers",
                actual: 1,
                expected: 2
            })
        );
        Ok(())
    }

    fn assert_array_close(
        actual: ndarray::ArrayView1<'_, Real>,
        expected: ndarray::ArrayView1<'_, Real>,
    ) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_close(*actual, *expected);
        }
    }

    fn assert_close(actual: Real, expected: Real) {
        let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        let tolerance = 1.0e-12_f64.max(expected.norm() * 1.0e-12);
        assert!(
            (actual - expected).norm() <= tolerance,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn assert_complex32_close(actual: Complex32, expected: Complex32) {
        let tolerance = 1.0e-6_f32.max(expected.norm() * 1.0e-6);
        assert!(
            (actual - expected).norm() <= tolerance,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn identity_basis() -> [[Real; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    fn one_state_band_states() -> [StateKet; 1] {
        [StateKet {
            atom: 1,
            angular_momentum: 0,
            magnetic: 0,
            spin: 1,
        }]
    }

    fn one_state_band_atoms() -> [FmsAtom; 1] {
        [FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        }]
    }

    fn one_state_phase_grid() -> Array4<Complex32> {
        let mut phase_grid = Array4::zeros((2, 1, 1, 1).f());
        phase_grid[(0, 0, 0, 0)] = Complex32::new(0.25, 0.0);
        phase_grid[(1, 0, 0, 0)] = Complex32::new(0.35, 0.0);
        phase_grid
    }

    struct SingleStateKspaceFixture {
        transforms: BasisTransformMatrices,
        reciprocal_indices: Array2<i32>,
        reciprocal_pair_phases: Array2<Complex>,
        d1term3: Array1<Complex>,
        qjltab: Array2<Real>,
        q_pair_offsets: Array2<Real>,
        direct_indices: Array2<i32>,
        direct_index_by_pair: Array2<usize>,
        direct_counts: [usize; 1],
        direct_terms: Array3<Complex>,
        q_pair_sites: Array3<usize>,
        q_pair_counts: [usize; 1],
        site_offsets: [usize; 1],
        site_state_counts: [usize; 1],
        gaunt_counts: [usize; 1],
        gaunt_indices: [usize; 1],
        gaunt_values: [Real; 1],
        cipwl: Array1<Complex>,
        rel_component_counts: Array2<usize>,
        rel_component_indices: Array3<usize>,
        rel_component_coefficients: Array3<Complex>,
        t_matrix: Array2<Complex32>,
    }

    impl SingleStateKspaceFixture {
        fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let mut q_pair_sites = Array3::<usize>::zeros((1, 1, 2));
            q_pair_sites[(0, 0, 0)] = 0;
            q_pair_sites[(0, 0, 1)] = 0;
            let mut rel_component_indices = Array3::<usize>::zeros((1, 2, 1));
            rel_component_indices[(0, 0, 0)] = 0;
            let mut rel_component_coefficients = Array3::<Complex>::zeros((1, 2, 1));
            rel_component_coefficients[(0, 0, 0)] = Complex::new(1.0, 0.0);

            Ok(Self {
                transforms: crate::basis_transform_matrices(0)?,
                reciprocal_indices: Array2::<i32>::zeros((0, 3)),
                reciprocal_pair_phases: Array2::<Complex>::zeros((0, 1)),
                d1term3: arr1(&[Complex::new(1.0, 0.0)]),
                qjltab: arr2(&[[1.0]]),
                q_pair_offsets: arr2(&[[0.0, 0.0, 0.0]]),
                direct_indices: Array2::<i32>::zeros((0, 3)),
                direct_index_by_pair: Array2::<usize>::zeros((0, 1)),
                direct_counts: [0],
                direct_terms: Array3::<Complex>::zeros((1, 0, 1)),
                q_pair_sites,
                q_pair_counts: [1],
                site_offsets: [0],
                site_state_counts: [1],
                gaunt_counts: [1],
                gaunt_indices: [0],
                gaunt_values: [1.0],
                cipwl: arr1(&[Complex::new(1.0, 0.0)]),
                rel_component_counts: arr2(&[[1_usize], [0]]),
                rel_component_indices,
                rel_component_coefficients,
                t_matrix: arr2(&[[Complex32::new(1.0, 0.0)]]),
            })
        }

        fn non_rel_point<'a>(&'a self, d300: Real) -> BandKkrFromKspaceNonRelInput<'a> {
            BandKkrFromKspaceNonRelInput {
                structure_factor: BandStructureFactorFromKspaceNonRelInput {
                    kspace: self.non_rel_kspace(d300),
                    atom_count: 1,
                    angular_lmax: 0,
                    basis_transforms: &self.transforms,
                },
                t_matrix: self.t_matrix.view(),
                wave_number: Complex32::new(2.0, 0.0),
            }
        }

        fn rel_point<'a>(&'a self, d300: Real) -> BandKkrFromKspaceRelInput<'a> {
            BandKkrFromKspaceRelInput {
                structure_factor: BandStructureFactorFromKspaceRelInput {
                    kspace: self.rel_kspace(d300),
                    atom_count: 1,
                    angular_lmax: 0,
                    basis_transforms: &self.transforms,
                },
                t_matrix: self.t_matrix.view(),
                wave_number: Complex32::new(2.0, 0.0),
            }
        }

        fn non_rel_freeprop_point<'a>(
            &'a self,
            d300: Real,
        ) -> BandFreePropagationFromKspaceNonRelInput<'a> {
            BandFreePropagationFromKspaceNonRelInput {
                structure_factor: BandStructureFactorFromKspaceNonRelInput {
                    kspace: self.non_rel_kspace(d300),
                    atom_count: 1,
                    angular_lmax: 0,
                    basis_transforms: &self.transforms,
                },
                wave_number: Complex32::new(2.0, 0.0),
            }
        }

        fn rel_freeprop_point<'a>(
            &'a self,
            d300: Real,
        ) -> BandFreePropagationFromKspaceRelInput<'a> {
            BandFreePropagationFromKspaceRelInput {
                structure_factor: BandStructureFactorFromKspaceRelInput {
                    kspace: self.rel_kspace(d300),
                    atom_count: 1,
                    angular_lmax: 0,
                    basis_transforms: &self.transforms,
                },
                wave_number: Complex32::new(2.0, 0.0),
            }
        }

        fn non_rel_kspace<'a>(&'a self, d300: Real) -> KSpaceStrsetNonRelFromLatticeSumInput<'a> {
            KSpaceStrsetNonRelFromLatticeSumInput {
                lattice_sum: self.lattice_sum(d300),
                angular_state_count: 1,
                q_pair_sites: self.q_pair_sites.view(),
                q_pair_counts: &self.q_pair_counts,
                site_offsets: &self.site_offsets,
                site_state_counts: &self.site_state_counts,
                gaunt_counts: &self.gaunt_counts,
                gaunt_indices: &self.gaunt_indices,
                gaunt_values: &self.gaunt_values,
                cipwl: self.cipwl.view(),
                wave_number: Complex::new(2.0, 0.0),
            }
        }

        fn rel_kspace<'a>(&'a self, d300: Real) -> KSpaceStrsetRelFromLatticeSumInput<'a> {
            KSpaceStrsetRelFromLatticeSumInput {
                lattice_sum: self.lattice_sum(d300),
                angular_state_count: 1,
                q_pair_sites: self.q_pair_sites.view(),
                q_pair_counts: &self.q_pair_counts,
                site_offsets: &self.site_offsets,
                site_state_counts: &self.site_state_counts,
                gaunt_counts: &self.gaunt_counts,
                gaunt_indices: &self.gaunt_indices,
                gaunt_values: &self.gaunt_values,
                cipwl: self.cipwl.view(),
                rel_component_counts: self.rel_component_counts.view(),
                rel_component_indices: self.rel_component_indices.view(),
                rel_component_coefficients: self.rel_component_coefficients.view(),
                wave_number: Complex::new(2.0, 0.0),
            }
        }

        fn lattice_sum<'a>(&'a self, d300: Real) -> KSpaceStrbbddInput<'a> {
            KSpaceStrbbddInput {
                k: [0.0, 0.0, 0.0],
                lmax: 0,
                eta: 1.0,
                energy: Complex::new(0.0, 0.0),
                gmax_squared: 1.0,
                reciprocal_basis: identity_basis(),
                reciprocal_indices: self.reciprocal_indices.view(),
                reciprocal_pair_phases: self.reciprocal_pair_phases.view(),
                d1term3: self.d1term3.view(),
                qjltab: self.qjltab.view(),
                q_pair_offsets: self.q_pair_offsets.view(),
                direct_basis: identity_basis(),
                direct_indices: self.direct_indices.view(),
                direct_index_by_pair: self.direct_index_by_pair.view(),
                direct_counts: &self.direct_counts,
                direct_terms: self.direct_terms.view(),
                d300: Complex::new(d300, 0.0),
            }
        }
    }

    fn reference_phase_shifts() -> Array3<Complex32> {
        let mut phases = Array3::zeros((2, 5, 2).f());
        phases[(0, 4, 1)] = Complex32::new(0.2, 0.05);
        phases[(0, 0, 1)] = Complex32::new(-0.1, 0.03);
        phases[(1, 4, 1)] = Complex32::new(0.15, -0.02);
        phases[(1, 0, 1)] = Complex32::new(0.07, 0.04);
        phases
    }
}
