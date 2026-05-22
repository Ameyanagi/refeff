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

/// Error returned by density accumulation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
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
}
