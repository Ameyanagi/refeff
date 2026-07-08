//! Public FULLSPECTRUM data types.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::elam::ElamError;
use crate::interpolation::InterpolationError;
use crate::{Complex, Real};

/// Inputs for FEFF `FULLSPECTRUM/qsum.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumQSumInput<'a> {
    /// Number density `numden` used in the oscillator-strength sum rule.
    pub number_density: Real,
    /// Imaginary dielectric function `eps2`.
    pub epsilon2: ArrayView1<'a, Real>,
    /// Energy grid `omega`.
    pub omega: ArrayView1<'a, Real>,
    /// Number of active rows, equivalent to FEFF `iepts`.
    pub active_len: usize,
}

/// Inputs for FEFF `FULLSPECTRUM/drdtrm.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumDrudeInput<'a> {
    /// Energy grid `omega` in Hartree.
    pub omega: ArrayView1<'a, Real>,
    /// Drude lifetime `tau`, in seconds.
    pub lifetime_seconds: Real,
    /// Free-electron density `numden`, in FEFF atomic units.
    pub number_density: Real,
}

/// Drude free-electron dielectric contribution.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumDrudeTerm {
    /// Drude width in eV, matching the first `drude.dat` header value.
    pub gamma_ev: Real,
    /// Plasma frequency in eV, matching the second `drude.dat` header value.
    pub plasma_frequency_ev: Real,
    /// Energy grid `omega` in Hartree.
    pub omega: Array1<Real>,
    /// Complex Drude dielectric contribution on `omega`.
    pub epsilon: Array1<Complex>,
}

impl FullSpectrumDrudeTerm {
    /// Number of Drude samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.omega.len()
    }
}

/// Inputs for FEFF `FULLSPECTRUM/rdval.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumValenceInput<'a> {
    /// Number density `numden` in FEFF atomic units.
    pub number_density: Real,
    /// Output photon energy grid `omega`, in Hartree.
    pub omega: ArrayView1<'a, Real>,
    /// Source `xmu.dat` photon energy column, in eV.
    pub source_energy_ev: ArrayView1<'a, Real>,
    /// Absolute valence absorption cross section, in square Angstroms.
    pub source_absorption_angstrom2: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `FULLSPECTRUM/rddens.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumNumberDensityInput<'a> {
    /// Atomic number whose species density should be estimated.
    pub target_atomic_number: usize,
    /// Atomic numbers for FEFF potential slots, `iz(0:nph)`.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// Multiplicity of each potential slot, FEFF `xnatph(0:nph)`.
    pub potential_multiplicities: ArrayView1<'a, Real>,
    /// Norman radii in Bohr, FEFF `rnrm(0:nph)`.
    pub norman_radii: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `FULLSPECTRUM/gtedgs.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumEdgeSelectionInput<'a> {
    /// FEFF `getorb` occupation row for the 40 core-hole slots.
    pub occupations: ArrayView1<'a, Real>,
    /// Edge onsets in Hartree, indexed by the same 40 core-hole slots.
    pub edge_onsets_hartree: ArrayView1<'a, Real>,
}

/// One edge selected by FEFF `FULLSPECTRUM/gtedgs.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullSpectrumSelectedEdge {
    /// One-based FEFF core-hole slot.
    pub hole_index: usize,
    /// FEFF edge label, such as `K`, `L3`, or `M5`.
    pub label: &'static str,
    /// Occupation of the selected initial state.
    pub occupation: Real,
    /// True when FEFF would convolve this edge with the density of states.
    pub convolve: bool,
}

/// Edge list selected for one FULLSPECTRUM component.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumEdgeSelection {
    /// Edges with occupation at least one, in FEFF hole-index order.
    pub edges: Vec<FullSpectrumSelectedEdge>,
}

impl FullSpectrumEdgeSelection {
    /// Number of selected edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

/// Inputs for FEFF `FULLSPECTRUM/sumrules.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumSumRulesInput<'a> {
    /// Number density `numden` in FEFF atomic units.
    pub number_density: Real,
    /// Photon energy grid in eV, as read from `opconsKK.dat`.
    pub energy_ev: ArrayView1<'a, Real>,
    /// Dielectric function minus one, using the columns written by `opcons.f90`.
    pub epsilon_minus_one: ArrayView1<'a, Complex>,
    /// Refractive index minus one, using the columns written by `opcons.f90`.
    pub refractive_index_minus_one: ArrayView1<'a, Complex>,
    /// FEFF `mu` absorption coefficient column, in `cm^(-1)`.
    pub absorption_coefficient: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `FULLSPECTRUM/egrid_lin.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumLinearGridInput {
    /// Number of energy points to generate.
    pub point_count: usize,
    /// Requested lower energy bound in Hartree.
    pub min_energy: Real,
    /// Upper energy bound in Hartree.
    pub max_energy: Real,
}

/// Inputs for FEFF `FULLSPECTRUM/egrid.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumEdgeGridInput<'a> {
    /// Requested lower energy bound in Hartree.
    pub min_energy: Real,
    /// Upper energy bound in Hartree.
    pub max_energy: Real,
    /// Edge energies in Hartree. This is the table FEFF obtains from Elam data.
    pub edge_energies: ArrayView1<'a, Real>,
    /// Momentum spacing, FEFF `xkstep`.
    pub wave_number_step: Real,
    /// Maximum number of points to emit, FEFF `fullpts`.
    pub max_points: usize,
}

/// One selected edge considered by `FULLSPECTRUM/rdop.f90` grid defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullSpectrumDefaultGridEdge {
    /// Component atomic number.
    pub atomic_number: i32,
    /// One-based FEFF core-hole index.
    pub hole_index: i32,
    /// True when this edge will include fine structure.
    pub fine_structure: bool,
}

/// Energy-grid defaults inferred by `FULLSPECTRUM/rdop.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullSpectrumDefaultEnergyGrid {
    /// Lower photon-energy bound in Hartree.
    pub min_energy: Real,
    /// Upper photon-energy bound in Hartree.
    pub max_energy: Real,
    /// FEFF's 0.5 eV point-count estimate.
    pub point_count: usize,
    /// True when bounds came from the fine-structure edge subset.
    pub used_fine_structure_edges: bool,
}

/// Energy grid generated by FEFF `FULLSPECTRUM/egrid.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumEdgeGrid {
    /// Generated energy grid in Hartree.
    pub energy: Array1<Real>,
    /// True when `max_points` was exhausted before the upper bound was reached.
    pub clipped: bool,
}

impl FullSpectrumEdgeGrid {
    /// Number of generated grid points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy.len()
    }
}

/// One FPRIME background segment consumed by `FULLSPECTRUM/rdbkg.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumBackgroundSegmentInput<'a> {
    /// Photon energy grid from `xmu.dat`, in eV.
    pub photon_energy_ev: ArrayView1<'a, Real>,
    /// Real scattering factor column from the normalized FPRIME `xmu.dat`.
    pub f_prime: ArrayView1<'a, Real>,
    /// Imaginary scattering factor column from the normalized FPRIME `xmu.dat`.
    pub f_double_prime: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `FULLSPECTRUM/rdbkg.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumBackgroundInput<'a> {
    /// Output photon-energy grid `omega`, in Hartree.
    pub omega: ArrayView1<'a, Real>,
    /// FPRIME segments in FEFF file priority order (`fprime1`, `fprime2`, ...).
    pub segments: &'a [FullSpectrumBackgroundSegmentInput<'a>],
}

/// Background scattering factor assembled from FPRIME calculations.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumBackground {
    /// Complex scalar scattering factor on the requested Hartree grid.
    pub scattering_factor: Array1<Complex>,
    /// FEFF `qsum` effective electron count estimate for this edge.
    pub effective_electron_count: Real,
    /// Lowest-energy real scattering factor used as FEFF `fp0`.
    pub zero_energy_fprime: Real,
}

impl FullSpectrumBackground {
    /// Number of output energy-grid rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.scattering_factor.len()
    }
}

/// One FMS/path-expansion segment consumed by `FULLSPECTRUM/rdst.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumFineStructureSegmentInput<'a> {
    /// Photon energy grid from `xmu.dat`, in eV.
    pub photon_energy_ev: ArrayView1<'a, Real>,
    /// Photoelectron wave number from `xmu.dat`, in inverse Angstrom.
    pub wave_number_inverse_angstrom: ArrayView1<'a, Real>,
    /// Scattering-factor component on this segment.
    ///
    /// For real segments this is `f'`; for imaginary segments it is `f''`
    /// after any `rdxmu.f90` cross-section conversion.
    pub scattering_factor: ArrayView1<'a, Real>,
    /// Atomic-background component matching [`Self::scattering_factor`].
    pub background: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `FULLSPECTRUM/rdst.f90` fine-structure interpolation.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumFineStructureInput<'a> {
    /// Output photon-energy grid `omega`, in Hartree.
    pub omega: ArrayView1<'a, Real>,
    /// FMS/DANES real-part segment.
    pub real_fms: FullSpectrumFineStructureSegmentInput<'a>,
    /// Path-expansion/DANES real-part segment.
    pub real_path: FullSpectrumFineStructureSegmentInput<'a>,
    /// FMS/XANES imaginary-part segment.
    pub imaginary_fms: FullSpectrumFineStructureSegmentInput<'a>,
    /// Path-expansion/EXAFS imaginary-part segment.
    pub imaginary_path: FullSpectrumFineStructureSegmentInput<'a>,
    /// Lowest wave number used to start the path-expansion transition.
    pub low_wave_number: Real,
    /// Highest wave number used to end the FMS transition.
    pub high_wave_number: Real,
}

/// Fine-structure scattering factor and near-edge atomic background.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumFineStructure {
    /// Complex fine-structure scattering factor on the requested Hartree grid.
    pub scattering_factor: Array1<Complex>,
    /// Complex near-edge atomic background on the requested Hartree grid.
    pub background: Array1<Complex>,
    /// Real-part source interval `[elo(1), ehi(1)]`, in Hartree.
    pub real_energy_interval: [Real; 2],
    /// Imaginary-part source interval `[elo(2), ehi(2)]`, in Hartree.
    pub imaginary_energy_interval: [Real; 2],
    /// Real-part FMS/path transition interval, in Hartree.
    pub real_transition_interval: [Real; 2],
    /// Imaginary-part FMS/path transition interval, in Hartree.
    pub imaginary_transition_interval: [Real; 2],
}

impl FullSpectrumFineStructure {
    /// Number of output energy-grid rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.scattering_factor.len()
    }
}

/// Inputs for FEFF `FULLSPECTRUM/addedg.f90` edge assembly.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumEdgeAssemblyInput<'a> {
    /// Output photon-energy grid `omega`, in Hartree.
    pub omega: ArrayView1<'a, Real>,
    /// FPRIME background produced by [`crate::fullspectrum::full_spectrum_background_from_fprime`].
    pub background: &'a FullSpectrumBackground,
    /// FMS/path fine structure produced by [`crate::fullspectrum::full_spectrum_fine_structure_from_segments`].
    pub fine_structure: &'a FullSpectrumFineStructure,
    /// Width used to choose the entry transition overlap, FEFF `trsize`.
    pub transition_size: Real,
}

/// Edge contribution after combining FPRIME background and fine structure.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumEdgeAssembly {
    /// Full complex scattering factor contribution from one edge.
    pub scattering_factor: Array1<Complex>,
    /// Atomic-background contribution from one edge.
    pub background: Array1<Complex>,
    /// Effective electron count carried through from `rdbkg`.
    pub effective_electron_count: Real,
    /// FEFF `fp0` shift applied to both returned arrays.
    pub zero_energy_fprime: Real,
    /// Number of output-grid points used for FEFF's main edge transition.
    pub overlap_points: usize,
}

impl FullSpectrumEdgeAssembly {
    /// Number of output energy-grid rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.scattering_factor.len()
    }
}

/// Inputs for the dielectric conversion in `FULLSPECTRUM/fullspectrum.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumScatteringDielectricInput<'a> {
    /// Component number density `numden`, in FEFF atomic units.
    pub number_density: Real,
    /// Energy grid `omega` in Hartree.
    pub omega: ArrayView1<'a, Real>,
    /// Full complex scattering factor `f` assembled for one edge.
    pub scattering_factor: ArrayView1<'a, Complex>,
    /// Atomic-background scattering factor `f0` assembled for one edge.
    pub background_scattering_factor: ArrayView1<'a, Complex>,
}

/// Dielectric contribution and `sigma` column derived from edge scattering.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumScatteringDielectric {
    /// Energy grid `omega` in Hartree.
    pub omega: Array1<Real>,
    /// Complex dielectric contribution, stored by FEFF as `eps - 1`.
    pub epsilon_minus_one: Array1<Complex>,
    /// Atomic-background dielectric contribution, stored as `eps0 - 1`.
    pub background_epsilon_minus_one: Array1<Complex>,
    /// Conductivity-like `sigma` contribution written to `eps.dat`/`xmu.dat`.
    pub sigma: Array1<Real>,
}

impl FullSpectrumScatteringDielectric {
    /// Number of dielectric-contribution samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.omega.len()
    }
}

/// Inputs for FEFF `FULLSPECTRUM/kk.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumKramersKronigInput<'a> {
    /// Monotonic energy grid `omega`.
    pub omega: ArrayView1<'a, Real>,
    /// Imaginary dielectric function `eps2` tabulated on `omega`.
    pub epsilon2: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `FULLSPECTRUM/hamaker.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumHamakerInput<'a> {
    /// Monotonic energy grid `omega`.
    pub omega: ArrayView1<'a, Real>,
    /// Complex dielectric function `eps`; FEFF uses its imaginary part.
    pub epsilon: ArrayView1<'a, Complex>,
}

/// Inputs for FEFF `FULLSPECTRUM/opcons.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumOpticalConstantsInput<'a> {
    /// Energy grid `omega` in Hartree.
    pub omega: ArrayView1<'a, Real>,
    /// Complex dielectric response minus one, matching FEFF's `eps` variable.
    pub epsilon_minus_one: ArrayView1<'a, Complex>,
}

/// Optical constants derived from a FULLSPECTRUM dielectric response.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumOpticalConstants {
    /// Energy grid `omega` in Hartree.
    pub omega: Array1<Real>,
    /// Complex dielectric response minus one.
    pub epsilon_minus_one: Array1<Complex>,
    /// Complex refractive index minus one.
    pub refractive_index_minus_one: Array1<Complex>,
    /// Absorption coefficient, matching `FULLSPECTRUM/opcons.f90`.
    pub absorption_coefficient: Array1<Real>,
    /// Normal-incidence reflectivity.
    pub reflectivity: Array1<Real>,
    /// Energy-loss function.
    pub loss: Array1<Real>,
}

impl FullSpectrumOpticalConstants {
    /// Number of optical-constant samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.omega.len()
    }
}

/// Cumulative rows written to FEFF `sumrules.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumSumRules {
    /// Photon energy grid in eV.
    pub energy_ev: Array1<Real>,
    /// Cumulative `epsilon_2` sum-rule effective electron count.
    pub epsilon2_effective_electrons: Array1<Real>,
    /// Cumulative absorption-coefficient sum-rule effective electron count.
    pub absorption_effective_electrons: Array1<Real>,
    /// Cumulative loss-function sum-rule effective electron count.
    pub loss_effective_electrons: Array1<Real>,
    /// Cumulative `mu * (n - 1)` sum-rule column.
    pub absorption_refractive_sum: Array1<Real>,
    /// Cumulative `(n - 1)` signed-to-absolute integral ratio.
    pub refractive_index_sum_ratio: Array1<Real>,
    /// Cumulative logarithmic loss-function moment ratio.
    pub log_loss_moment_ratio: Array1<Real>,
}

impl FullSpectrumSumRules {
    /// Number of cumulative sum-rule samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

/// Error returned by FULLSPECTRUM helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum FullSpectrumError {
    /// Number density must be positive.
    #[error("FULLSPECTRUM {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    /// Scalar inputs must be finite.
    #[error("FULLSPECTRUM {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Active rows must fit in both input arrays.
    #[error("FULLSPECTRUM active row count {active_len} exceeds {field} length {len}")]
    ActiveCountOutOfRange {
        field: &'static str,
        active_len: usize,
        len: usize,
    },
    /// Array values must be finite.
    #[error("FULLSPECTRUM {field} row {row} must be finite, got {value}")]
    NonFiniteValue {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// Energy rows are expected in nondecreasing order for the trapezoid rule.
    #[error("FULLSPECTRUM omega row {row} must not decrease, got {current} after {previous}")]
    DecreasingOmega {
        row: usize,
        previous: Real,
        current: Real,
    },
    /// Array values must be positive.
    #[error("FULLSPECTRUM {field} row {row} must be positive, got {value}")]
    NonPositiveValue {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// Tabulated sum-rule inputs require at least one row.
    #[error("FULLSPECTRUM {name} requires at least one row")]
    EmptyTable { name: &'static str },
    /// Array lengths must agree for a tabulated calculation.
    #[error("FULLSPECTRUM {field} length {actual} does not match energy length {expected}")]
    LengthMismatch {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// Each cumulative sum-rule value must be finite.
    #[error("FULLSPECTRUM sum-rule row {row} {field} must be finite, got {value}")]
    NonFiniteSumRule {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// The final sum-rule value must be finite.
    #[error("FULLSPECTRUM neff must be finite, got {value}")]
    NonFiniteResult { value: Real },
    /// Kramers-Kronig style transforms need at least two rows.
    #[error("FULLSPECTRUM {name} requires at least two rows, got {len}")]
    TooFewRows { name: &'static str, len: usize },
    /// Energy rows are expected in strictly increasing order for transforms.
    #[error("FULLSPECTRUM omega row {row} must increase, got {current} after {previous}")]
    NonIncreasingOmega {
        row: usize,
        previous: Real,
        current: Real,
    },
    /// FEFF `lint` failed while projecting midpoint transform values.
    #[error("FULLSPECTRUM midpoint interpolation failed: {source}")]
    Interpolation { source: InterpolationError },
    /// A generated energy grid must have an increasing range.
    #[error("FULLSPECTRUM {name} upper bound {max} must exceed lower bound {min}")]
    InvalidEnergyRange {
        name: &'static str,
        min: Real,
        max: Real,
    },
    /// Atomic numbers must be positive element identifiers.
    #[error("FULLSPECTRUM atomic number must be positive, got {atomic_number}")]
    InvalidAtomicNumber { atomic_number: usize },
    /// FEFF occupation-like table values cannot be negative.
    #[error("FULLSPECTRUM {field} row {row} must be non-negative, got {value}")]
    NegativeValue {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// A background segment expected at least two increasing energy rows.
    #[error("FULLSPECTRUM {name} segment {segment} requires at least two rows, got {len}")]
    SegmentTooShort {
        name: &'static str,
        segment: usize,
        len: usize,
    },
    /// Segment columns must match the segment energy length.
    #[error(
        "FULLSPECTRUM {field} segment {segment} length {actual} does not match energy length {expected}"
    )]
    SegmentLengthMismatch {
        field: &'static str,
        segment: usize,
        actual: usize,
        expected: usize,
    },
    /// Segment energy rows are expected to increase.
    #[error(
        "FULLSPECTRUM segment {segment} energy row {row} must increase, got {current} after {previous}"
    )]
    SegmentNonIncreasingEnergy {
        segment: usize,
        row: usize,
        previous: Real,
        current: Real,
    },
    /// FMS segment did not provide an energy at a required transition k.
    #[error("FULLSPECTRUM {name} did not cross wave-number threshold {threshold}")]
    MissingTransitionThreshold { name: &'static str, threshold: Real },
    /// FEFF Elam edge-table lookup failed while assembling the energy grid.
    #[error("FULLSPECTRUM component {component} Elam edge lookup failed: {source}")]
    ElamEdgeTable { component: usize, source: ElamError },
    /// A selected edge was valid but absent from FEFF's Elam table.
    #[error("FULLSPECTRUM missing Elam edge for Z={atomic_number}, hole {hole_index}")]
    MissingElamEdge { atomic_number: i32, hole_index: i32 },
}
