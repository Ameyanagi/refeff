use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use thiserror::Error;

use crate::{AngularError, BesselError, Complex, InterpolationError, Real};

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

/// FEFF `XSPH/xmult.f90` relativistic multipole prefactors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphRelativisticMultipoleFactors {
    /// FEFF `xm1`, multiplying the radial `P_k * Q_k'` contribution.
    pub p_q_prime: Complex,
    /// FEFF `xm2`, multiplying the radial `Q_k * P_k'` contribution.
    pub q_p_prime: Complex,
}

/// FEFF `specupdlg` branch for regular or irregular radial contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XsphSpectrumUpdateMode {
    /// FEFF `imode = 1`, regular radial-integral branch.
    Regular,
    /// FEFF `imode = 2`, irregular radial-integral branch.
    Irregular,
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
    /// FEFF `ispec` selector; `-3..=4` follows the user-grid `phmesh2` path.
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
    /// Relativistic kappa values must be nonzero.
    #[error("XSPH relativistic kappa must be nonzero")]
    ZeroKappa,
    /// Integer angular inputs must stay in the supported FEFF range.
    #[error("{name} value {value} is outside the supported XSPH integer range")]
    IntegerOutOfRange { name: &'static str, value: i32 },
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
    /// Linear algebra helper failed.
    #[error(transparent)]
    Linalg(#[from] refeff_linalg::LinalgError),
}
