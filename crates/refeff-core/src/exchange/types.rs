use super::*;
use ndarray::{Array1, ArrayView1};

/// Error returned by exchange-potential helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum ExchangeError {
    /// Inputs must be finite real values.
    #[error("exchange input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Complex inputs must have finite real and imaginary parts.
    #[error("exchange input {name} must be finite, got {value:?}")]
    NonFiniteComplex { name: &'static str, value: Complex },
    /// Inputs used as positive physical scales must be strictly positive.
    #[error("exchange input {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    /// Inputs used as nonnegative physical factors must be zero or positive.
    #[error("exchange input {name} must be nonnegative, got {value}")]
    NegativeInput { name: &'static str, value: Real },
    /// A square-root radicand fell outside the real branch used by FEFF.
    #[error("exchange radicand {name} must be nonnegative, got {value}")]
    NegativeRadicand { name: &'static str, value: Real },
    /// A logarithm argument fell outside the real branch used by FEFF.
    #[error("exchange logarithm argument {name} must be positive, got {value}")]
    NonPositiveLogArgument { name: &'static str, value: Real },
    /// One-based FEFF indices must be positive and in range.
    #[error("exchange input {name} index {index} is invalid")]
    InvalidIndex { name: &'static str, index: usize },
    /// Integer selectors must identify a branch supported by the FEFF helper.
    #[error("exchange input {name} selector {value} is invalid")]
    InvalidSelector { name: &'static str, value: i32 },
    /// Array inputs must cover the FEFF active prefix.
    #[error("exchange input {name} requires at least {required} values, got {actual}")]
    LengthTooShort {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// FEFF many-pole arrays must contain the `< -1` terminator before use.
    #[error("exchange input {name} has no many-pole sentinel in {len} values")]
    MissingManyPoleSentinel { name: &'static str, len: usize },
    /// Interpolation failed while evaluating a FEFF exchange helper.
    #[error("exchange interpolation failed: {0}")]
    Interpolation(crate::interpolation::InterpolationError),
    /// FEFF formula denominator is singular for this input.
    #[error("exchange denominator {name} is zero")]
    ZeroDenominator { name: &'static str },
    /// A lower-level self-energy helper failed while evaluating exchange.
    #[error("exchange self-energy helper {routine} failed: {detail}")]
    SelfEnergyFailure {
        routine: &'static str,
        detail: &'static str,
    },
    /// A FEFF branch depends on reference data that the caller did not supply.
    #[error("exchange input {name} selector {value} requires reference data {data}")]
    MissingReferenceData {
        name: &'static str,
        value: i32,
        data: &'static str,
    },
    /// A caller-provided FEFF reference table has the wrong shape.
    #[error(
        "exchange reference data {data} field {field} has length {actual}, expected {expected}"
    )]
    ReferenceDataLength {
        data: &'static str,
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// A caller-provided FEFF reference mesh is not strictly increasing.
    #[error("exchange reference data {data} field {field} is not increasing at index {index}")]
    NonIncreasingReferenceMesh {
        data: &'static str,
        field: &'static str,
        index: usize,
    },
    /// A branch-specific input was not supplied.
    #[error("exchange input {name} is required for selector {value}")]
    MissingRequiredInput { name: &'static str, value: i32 },
}

/// Exchange-correlation energy and potential from FEFF LDA helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExchangeCorrelation {
    /// Exchange-correlation energy per particle in Hartrees.
    pub energy_per_particle: Real,
    /// Exchange-correlation potential in Hartrees.
    pub potential: Real,
}

/// Spin branch used by FEFF `fxc_ksdt_01` and `exc_ksdt_01`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KsdTSpin {
    /// FEFF `iz = 0`, spin-unpolarized.
    Unpolarized,
    /// FEFF `iz = 1`, fully spin-polarized.
    FullyPolarized,
}

/// KSDT exchange-correlation free energy and potential.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KsdTFreeEnergy {
    /// Exchange-correlation free energy per particle in Hartrees.
    pub free_energy_per_particle: Real,
    /// Exchange-correlation potential in Hartrees.
    pub potential: Real,
}

/// Result from FEFF `imhl`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HedinLundqvistImaginary {
    /// Imaginary self-energy returned by FEFF `imhl`.
    pub value: Real,
    /// FEFF `icusp` flag, true at the beginning of the imaginary branch cusp.
    pub cusp: bool,
}

/// Result from FEFF `rhl`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HedinLundqvistSelfEnergy {
    /// Real Hedin-Lundqvist self-energy from FEFF `rhl`.
    pub real: Real,
    /// Imaginary self-energy from FEFF `imhl`, as returned by `rhl`.
    pub imaginary: Real,
    /// FEFF `imhl` cusp flag used to choose the real-branch interpolation.
    pub cusp: bool,
}

/// Number of density-radius rows in FEFF `bphl.dat`.
pub const BPHL_RADIUS_COUNT: usize = 21;

/// Number of reduced-energy columns in FEFF `rhlbp`, including the implicit
/// zero-valued first column that is absent from `bphl.dat`.
pub const BPHL_REDUCED_ENERGY_COUNT: usize = 51;

/// Number of explicit four-column records in FEFF `bphl.dat`.
pub const BPHL_RECORD_COUNT: usize = BPHL_RADIUS_COUNT * (BPHL_REDUCED_ENERGY_COUNT - 1);

/// Parsed reference table used by FEFF's broadened-plasmon Hedin-Lundqvist
/// routine `rhlbp`.
#[derive(Debug, Clone, PartialEq)]
pub struct BroadenedHedinLundqvistTable {
    pub(super) radius_mesh: Vec<Real>,
    pub(super) reduced_energy_mesh: Vec<Real>,
    pub(super) real: Vec<Real>,
    pub(super) imaginary: Vec<Real>,
}

impl BroadenedHedinLundqvistTable {
    /// Build and validate a table in radius-major order.
    ///
    /// `real` and `imaginary` contain all 21×51 values, including the
    /// implicit zero-valued reduced-energy column at index zero.
    pub fn new(
        radius_mesh: Vec<Real>,
        reduced_energy_mesh: Vec<Real>,
        real: Vec<Real>,
        imaginary: Vec<Real>,
    ) -> Result<Self, ExchangeError> {
        validate_bphl_length("radius_mesh", radius_mesh.len(), BPHL_RADIUS_COUNT)?;
        validate_bphl_length(
            "reduced_energy_mesh",
            reduced_energy_mesh.len(),
            BPHL_REDUCED_ENERGY_COUNT,
        )?;
        let value_count = BPHL_RADIUS_COUNT * BPHL_REDUCED_ENERGY_COUNT;
        validate_bphl_length("real", real.len(), value_count)?;
        validate_bphl_length("imaginary", imaginary.len(), value_count)?;

        validate_bphl_mesh("radius_mesh", &radius_mesh)?;
        validate_bphl_mesh("reduced_energy_mesh", &reduced_energy_mesh)?;
        for &value in &real {
            ensure_finite("bphl.dat real", value)?;
        }
        for &value in &imaginary {
            ensure_finite("bphl.dat imaginary", value)?;
        }

        Ok(Self {
            radius_mesh,
            reduced_energy_mesh,
            real,
            imaginary,
        })
    }

    /// FEFF `bphl.dat` density-radius mesh.
    #[must_use]
    pub fn radius_mesh(&self) -> &[Real] {
        &self.radius_mesh
    }

    /// FEFF `bphl.dat` reduced-energy mesh, including its implicit leading
    /// zero.
    #[must_use]
    pub fn reduced_energy_mesh(&self) -> &[Real] {
        &self.reduced_energy_mesh
    }

    /// Radius-major real self-energy table, including implicit leading zeros.
    #[must_use]
    pub fn real_values(&self) -> &[Real] {
        &self.real
    }

    /// Radius-major imaginary self-energy table, including implicit leading
    /// zeros.
    #[must_use]
    pub fn imaginary_values(&self) -> &[Real] {
        &self.imaginary
    }

    pub(super) fn flat_index(&self, radius_index: usize, energy_index: usize) -> usize {
        radius_index * BPHL_REDUCED_ENERGY_COUNT + energy_index
    }
}

fn validate_bphl_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ExchangeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ExchangeError::ReferenceDataLength {
            data: "bphl.dat",
            field,
            actual,
            expected,
        })
    }
}

fn validate_bphl_mesh(field: &'static str, values: &[Real]) -> Result<(), ExchangeError> {
    for (index, &value) in values.iter().enumerate() {
        ensure_finite("bphl.dat mesh", value)?;
        if index > 0 && value <= values[index - 1] {
            return Err(ExchangeError::NonIncreasingReferenceMesh {
                data: "bphl.dat",
                field,
                index,
            });
        }
    }
    Ok(())
}

/// Inputs for the MPSE density grid in FEFF `EXCH/xcpot.f90`.
#[derive(Debug, Clone, Copy)]
pub struct XcpotManyPoleDensityGridInput<'a> {
    /// FEFF `iPl`; selector `2` enables the radial many-pole grid.
    pub plasmon_selector: i32,
    /// Total electron density on the Loucks radial grid, FEFF `densty`.
    pub density: ArrayView1<'a, Real>,
    /// FEFF one-based `jri`; `xcpot` samples the interstitial row at `jri + 1`.
    pub radial_match_index_1based: usize,
}

/// MPSE Wigner-Seitz radius grid from FEFF `EXCH/xcpot.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotManyPoleDensityGrid {
    /// Interstitial radius `RsInt` derived from `densty(jri+1)`.
    pub interstitial_radius: Real,
    /// Core radius `rscore` derived from `densty(1)`.
    pub core_radius: Real,
    /// Lower radius bound `RsMin` from the maximum active density.
    pub min_radius: Real,
    /// Upper radius bound `RsMax` from the minimum active density.
    pub max_radius: Real,
    /// FEFF `DRs = (RsMax - RsMin) / (NRPts - 2)`.
    pub radius_step: Real,
    /// FEFF `Rs1(1:NRPts)` interpolation/sample radii.
    pub radii: [Real; XCPOT_MPSE_GRID_POINTS],
}

/// FEFF fixed `NRPts` value for the `xcpot` MPSE radius grid.
pub const XCPOT_MPSE_GRID_POINTS: usize = 10;

/// Inputs for FEFF `EXCH/xcpot.f90` MPSE enable and pole-count setup.
#[derive(Debug, Clone, Copy)]
pub struct XcpotManyPoleControlInput<'a> {
    /// FEFF `iPl`; many-pole self energy is enabled only when this is positive.
    pub plasmon_selector: i32,
    /// FEFF `index`; this helper applies `mod(index, 10)` before branch tests.
    pub exchange_selector: i32,
    /// FEFF `WpCorr` pole-frequency table, terminated by the first value `< -1`.
    pub pole_frequencies: ArrayView1<'a, Real>,
}

/// FEFF `EXCH/xcpot.f90` MPSE control state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XcpotManyPoleControl {
    /// FEFF `csig`, true when the many-pole self-energy path is active.
    pub enabled: bool,
    /// FEFF `NPoles`, the count before the first `< -1` sentinel.
    pub active_pole_count: usize,
}

/// Inputs for FEFF `EXCH/xcpot.f90` MPSE delta-self-energy table shaping.
#[derive(Debug, Clone, Copy)]
pub struct XcpotManyPoleDeltaTableInput<'a> {
    /// FEFF `iPl`; selector `2` keeps radial samples, positive others use bulk.
    pub plasmon_selector: i32,
    /// FEFF `SigF` values returned by `CSigZ` at the Fermi level.
    pub fermi_self_energy: ArrayView1<'a, Complex>,
    /// Raw FEFF `deltaHL` values returned by `CSigZ` at the current energy.
    pub energy_self_energy: ArrayView1<'a, Complex>,
    /// FEFF complex renormalization factor `ZRnrm`.
    pub renormalization: Complex,
}

/// MPSE delta-self-energy table from FEFF `EXCH/xcpot.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotManyPoleDeltaTable {
    /// Shaped Fermi-level self-energy cache, FEFF `SigF(1:NRPts)`.
    pub fermi_self_energy: [Complex; XCPOT_MPSE_GRID_POINTS],
    /// Renormalized current-energy delta table, FEFF `deltaHL(1:NRPts)`.
    pub delta_self_energy: [Complex; XCPOT_MPSE_GRID_POINTS],
    /// Real interpolation table, FEFF `delrHL(1:NRPts)`.
    pub real: [Real; XCPOT_MPSE_GRID_POINTS],
    /// Imaginary interpolation table, FEFF `deliHL(1:NRPts)`.
    pub imaginary: [Real; XCPOT_MPSE_GRID_POINTS],
}

/// Inputs for calculating the FEFF `CSigZ` MPSE delta table inside `xcpot`.
#[derive(Debug, Clone, Copy)]
pub struct XcpotManyPoleSelfEnergyTableInput<'a> {
    /// FEFF `iPl`; selector `2` computes radial samples, positive others use bulk.
    pub plasmon_selector: i32,
    /// Current complex energy, FEFF `em`.
    pub energy: Complex,
    /// Fermi level, FEFF `xmu`.
    pub fermi_level: Real,
    /// FEFF `Rs1` density-radius grid.
    pub density_grid: XcpotManyPoleDensityGrid,
    /// FEFF `WpCorr`, terminated by the first value `< -1`.
    pub pole_frequencies: ArrayView1<'a, Real>,
    /// FEFF `Gamma`.
    pub pole_widths: ArrayView1<'a, Real>,
    /// FEFF `AmpFac`.
    pub amplitudes: ArrayView1<'a, Real>,
    /// FEFF `EGap`.
    pub gap_energy: Real,
    /// FEFF `NPoles`, usually from [`xcpot_many_pole_control`].
    pub active_pole_count: usize,
    /// FEFF `UseBP`: select broadened-pole BPR self-energy integrands.
    ///
    /// FEFF `EXCH/xcpot.f90` currently passes `.FALSE.` for the ordinary XSPH
    /// MPSE path, but this field keeps the lower-level `CSigZ` selector
    /// available to source-backed callers that explicitly request BPR.
    pub use_broadened_pole: bool,
}

/// Raw FEFF MPSE pole data supplied to the composed `xcpot` driver.
#[derive(Debug, Clone, Copy)]
pub struct XcpotManyPoleSelfEnergyInput<'a> {
    /// FEFF `WpCorr`, terminated by the first value `< -1`.
    pub pole_frequencies: ArrayView1<'a, Real>,
    /// FEFF `Gamma`.
    pub pole_widths: ArrayView1<'a, Real>,
    /// FEFF `AmpFac`.
    pub amplitudes: ArrayView1<'a, Real>,
    /// FEFF `EGap`.
    pub gap_energy: Real,
    /// FEFF `UseBP`: select broadened-pole BPR self-energy integrands.
    pub use_broadened_pole: bool,
}

/// Inputs for FEFF `EXCH/xcpot.f90` MPSE row-delta selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotManyPoleRowDeltaInput {
    /// FEFF `iPl`; selector `2` enables radial interpolation.
    pub plasmon_selector: i32,
    /// Current density radius, FEFF `rs`.
    pub radius: Real,
    /// MPSE radius grid and bounds from FEFF `Rs1`, `RsMin`, and `RsMax`.
    pub density_grid: XcpotManyPoleDensityGrid,
    /// Real/imaginary MPSE delta tables, FEFF `delrHL`/`deliHL`.
    pub delta_table: XcpotManyPoleDeltaTable,
}

/// Inputs for applying FEFF `EXCH/xcpot.f90` self-energy deltas.
#[derive(Debug, Clone, Copy)]
pub struct XcpotSelfEnergyApplicationInput<'a> {
    /// FEFF `index`; this helper applies `mod(index, 10)` before branch tests.
    pub exchange_selector: i32,
    /// Ground-state total potential, FEFF `vtot(1:jri+1)`.
    pub total_potential: ArrayView1<'a, Real>,
    /// Ground-state valence potential, FEFF `vvalgs(1:jri+1)`.
    pub valence_potential: ArrayView1<'a, Real>,
    /// Real part of the total self-energy correction, FEFF `delr`.
    pub delta_real: ArrayView1<'a, Real>,
    /// Imaginary part of the total self-energy correction, FEFF `deli`.
    pub delta_imaginary: ArrayView1<'a, Real>,
    /// Real part of the valence self-energy correction, FEFF `delvr`.
    pub valence_delta_real: ArrayView1<'a, Real>,
    /// Imaginary part of the valence self-energy correction, FEFF `delvi`.
    pub valence_delta_imaginary: ArrayView1<'a, Real>,
    /// Active prefix length, FEFF `jri1 = jri + 1`.
    pub active_len: usize,
}

/// Complex work arrays after FEFF `EXCH/xcpot.f90` delta application.
#[derive(Debug, Clone, PartialEq)]
pub struct XcpotSelfEnergyApplication {
    /// Complex total potential workspace, FEFF `v(1:jri+1)`.
    pub total_potential: Array1<Complex>,
    /// Complex valence potential workspace, FEFF `vval(1:jri+1)` for `ixc >= 5`.
    pub valence_potential: Array1<Complex>,
}

/// Inputs for FEFF `EXCH/xcpot.f90` local density and momentum scales.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotLocalScalesInput {
    /// FEFF `index`; this helper applies `mod(index, 10)` before branch tests.
    pub exchange_selector: i32,
    /// Total electron density for one radial row, FEFF `densty(i)`.
    pub density: Real,
    /// Density magnetization for one radial row, FEFF `dmag(i)`.
    pub magnetization: Real,
    /// Valence electron density for one radial row, FEFF `denval(i)`.
    pub valence_density: Real,
}

/// Local radius and momentum scales from FEFF `EXCH/xcpot.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotLocalScales {
    /// FEFF `rs` derived from `densty(i)`, with the `10` nonpositive fallback.
    pub radius: Real,
    /// FEFF `xf = fa / rs`.
    pub fermi_momentum: Real,
    /// FEFF `rsm = rs / (1 + dmag(i))**third`.
    pub magnetized_radius: Real,
    /// FEFF `xfm = fa / rsm`.
    pub magnetized_fermi_momentum: Real,
    /// FEFF `rsval`, present only for `ixc == 5`.
    pub valence_radius: Option<Real>,
    /// FEFF `xfval`, present only for `ixc == 5`.
    pub valence_fermi_momentum: Option<Real>,
    /// FEFF `rscore`, recomputed in this block only for `ixc >= 6`.
    pub core_radius: Option<Real>,
}

/// Inputs for FEFF `EXCH/xcpot.f90` nested `sigma` dispatcher.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotSigmaInput {
    /// FEFF `index`; this helper applies `mod(index, 10)` and `index / 10`.
    pub exchange_selector: i32,
    /// Density radius, FEFF `rs`.
    pub radius: Real,
    /// Core-density radius used for `ixc >= 6`, FEFF `rscore`.
    pub core_radius: Real,
    /// Local momentum, FEFF `xk`.
    pub momentum: Real,
}

/// Real and imaginary self-energy from FEFF `EXCH/xcpot.f90` `sigma`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotSigma {
    /// FEFF `vr`.
    pub real: Real,
    /// FEFF `vi`.
    pub imaginary: Real,
}

/// Inputs for FEFF `EXCH/xcpot.f90` Fermi-level self-energy cache setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotFermiCacheInput {
    /// FEFF `index`; this helper applies `mod(index, 10)` and `index / 10`.
    pub exchange_selector: i32,
    /// Density radius, FEFF `rs`.
    pub radius: Real,
    /// Core-density radius, FEFF `rscore`.
    pub core_radius: Real,
    /// Valence density radius, FEFF `rsval`, required only for `ixc == 5`.
    pub valence_radius: Option<Real>,
    /// True for FEFF row `i == jri1`, where some valence cache values are copied.
    pub interstitial: bool,
}

/// Per-row Fermi-level cache values from FEFF `EXCH/xcpot.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotFermiCache {
    /// FEFF `vxcrmu(i)`/`vxcimu(i)`.
    pub total_self_energy: XcpotSigma,
    /// FEFF `vvxcrm(i)`/`vvxcim(i)`.
    pub valence_self_energy: XcpotSigma,
    /// FEFF `gsrel(i)`, currently left at the FEFF default `1`.
    pub ground_state_ratio: Real,
}

/// Inputs for the composed FEFF `EXCH/xcpot.f90` potential update.
#[derive(Debug, Clone)]
pub struct XcpotInput<'a> {
    /// FEFF `index`; this helper applies `mod(index, 10)` and `index / 10`.
    pub exchange_selector: i32,
    /// FEFF `lreal`; positive values force real self energy after referencing.
    pub lreal: i32,
    /// Current complex energy, FEFF `em`.
    pub energy: Complex,
    /// Fermi level, FEFF `xmu`.
    pub fermi_level: Real,
    /// Ground-state total potential, FEFF `vtot(1:jri+1)`.
    pub total_potential: ArrayView1<'a, Real>,
    /// Ground-state valence potential, FEFF `vvalgs(1:jri+1)`.
    pub valence_potential: ArrayView1<'a, Real>,
    /// Total electron density on the Loucks radial grid, FEFF `densty`.
    pub density: ArrayView1<'a, Real>,
    /// Density magnetization on the Loucks radial grid, FEFF `dmag`.
    pub magnetization: ArrayView1<'a, Real>,
    /// Valence electron density on the Loucks radial grid, FEFF `denval`.
    pub valence_density: ArrayView1<'a, Real>,
    /// Active prefix length, FEFF `jri1 = jri + 1`.
    pub active_len: usize,
    /// FEFF `iPl`; positive with `ixc == 0` enables the MPSE branch.
    pub plasmon_selector: i32,
    /// Optional shaped MPSE delta table, used while the `CSigZ` calculation is
    /// bypassed or supplied from a caller-owned cache.
    pub many_pole_delta_table: Option<XcpotManyPoleDeltaTable>,
    /// Raw MPSE pole data used to calculate the FEFF `CSigZ` delta table when
    /// a prepared table is not supplied.
    pub many_pole_self_energy: Option<XcpotManyPoleSelfEnergyInput<'a>>,
    /// Optional cached FEFF `vxcrmu/vxcimu`, `vvxcrm/vvxcim`, and `gsrel`
    /// values for calls after the first entry.
    pub fermi_cache: Option<ArrayView1<'a, XcpotFermiCache>>,
}

/// Composed FEFF `EXCH/xcpot.f90` potential update.
#[derive(Debug, Clone, PartialEq)]
pub struct XcpotResult {
    /// Complex reference potential, FEFF `eref`.
    pub reference_energy: Complex,
    /// Referenced total potential, FEFF `v(1:jri+1)`.
    pub total_potential: Array1<Complex>,
    /// Referenced valence potential, FEFF `vval(1:jri+1)`.
    pub valence_potential: Array1<Complex>,
    /// Per-row Fermi-level cache values used or generated by the non-MPSE path.
    pub fermi_cache: Array1<XcpotFermiCache>,
    /// Density-radius grid built for dynamic calls, FEFF `Rs1`.
    pub density_grid: Option<XcpotManyPoleDensityGrid>,
}

/// Inputs for FEFF `EXCH/xcpot.f90` Dyson self-energy correction block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotSelfEnergyCorrectionInput {
    /// FEFF `index`; this helper applies `mod(index, 10)` and `index / 10`.
    pub exchange_selector: i32,
    /// Real current energy, FEFF `dble(em)`.
    pub energy: Real,
    /// Fermi level, FEFF `xmu`.
    pub fermi_level: Real,
    /// Density radius, FEFF `rs`.
    pub radius: Real,
    /// Core-density radius, FEFF `rscore`.
    pub core_radius: Real,
    /// Fermi momentum, FEFF `xf`.
    pub fermi_momentum: Real,
    /// Magnetized Fermi momentum, FEFF `xfm`.
    pub magnetized_fermi_momentum: Real,
    /// Valence density radius, FEFF `rsval`, required only for `ixc == 5`.
    pub valence_radius: Option<Real>,
    /// Valence Fermi momentum, FEFF `xfval`, required only for `ixc == 5`.
    pub valence_fermi_momentum: Option<Real>,
    /// True for FEFF row `i == jri1`.
    pub interstitial: bool,
    /// FEFF `vxcrmu/vxcimu`, `vvxcrm/vvxcim`, and `gsrel` cache values.
    pub fermi_cache: XcpotFermiCache,
}

/// Per-row self-energy correction from FEFF `EXCH/xcpot.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcpotSelfEnergyCorrection {
    /// FEFF `xkm`, retained for the magnetized-momentum branch.
    pub magnetized_momentum: Real,
    /// Final corrected local momentum, FEFF `xk`.
    pub corrected_momentum: Real,
    /// FEFF `delr/deli`.
    pub total_delta: XcpotSigma,
    /// FEFF `delvr/delvi` when the branch assigns it.
    pub valence_delta: Option<XcpotSigma>,
}

/// Inputs for FEFF `EXCH/xcpot.f90` final potential referencing.
#[derive(Debug, Clone, Copy)]
pub struct XcpotReferenceShiftInput<'a> {
    /// FEFF `index`; this helper applies `mod(index, 10)` before branch tests.
    pub exchange_selector: i32,
    /// FEFF `lreal`; positive values force real self energy.
    pub lreal: i32,
    /// FEFF complex total potential workspace `v(1:jri+1)` before referencing.
    pub total_potential: ArrayView1<'a, Complex>,
    /// FEFF complex valence potential workspace `vval(1:jri+1)` before referencing.
    pub valence_potential: ArrayView1<'a, Complex>,
    /// Active prefix length, FEFF `jri1 = jri + 1`.
    pub active_len: usize,
}

/// Final referenced potentials from FEFF `EXCH/xcpot.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct XcpotReferenceShift {
    /// Complex reference potential, FEFF `eref`.
    pub reference_energy: Complex,
    /// Referenced total potential, FEFF `v(1:jri+1)`.
    pub total_potential: Array1<Complex>,
    /// Referenced valence potential, FEFF `vval(1:jri+1)`.
    pub valence_potential: Array1<Complex>,
}

/// Inputs for FEFF `EXCH/xcpot.f90` ground-state/static-potential branch.
#[derive(Debug, Clone, Copy)]
pub struct XcpotGroundStateBranchInput<'a> {
    /// FEFF `index`; the branch uses `mod(index, 10)` as `ixc`.
    pub exchange_selector: i32,
    /// FEFF `lreal`; positive values force real self energy after referencing.
    pub lreal: i32,
    /// Current complex energy, FEFF `em`.
    pub energy: Complex,
    /// Fermi level, FEFF `xmu`.
    pub fermi_level: Real,
    /// Ground-state total potential, FEFF `vtot(1:jri+1)`.
    pub total_potential: ArrayView1<'a, Real>,
    /// Ground-state valence potential, FEFF `vvalgs(1:jri+1)`.
    pub valence_potential: ArrayView1<'a, Real>,
    /// Active prefix length, FEFF `jri1 = jri + 1`.
    pub active_len: usize,
}
