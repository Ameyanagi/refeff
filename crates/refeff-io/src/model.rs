//! Typed extraction of common FEFF input cards.
//!
//! This layer intentionally starts with stable structural cards and grows as
//! each FEFF module is ported. Unknown or module-specific cards remain
//! available in [`crate::FeffInput`] so no information is lost.

use std::path::{Component, Path, PathBuf};

use crate::cif::{
    CifCluster, CifEquivalence, expand_cif_cluster_with_equivalence,
    expand_cif_structure_with_equivalence, read_cif,
};
use crate::control_input::{
    BandEnergyMesh, BandInput, DensityInput, FullSpectrumInput, OpconsInput, ReciprocalCell,
    ReciprocalInput, ReciprocalKMesh,
};
use crate::dym::parse_dym;
use crate::error::{IoError, Result};
use crate::global_input::CfAverage;
use crate::grid_input::parse_grid_inp;
use crate::input::{FeffInput, FeffLine, LineKind};
use crate::screen_input::ScreenInput;
use crate::sfconv_input::{SfconvControl, SfconvInput, SfconvSpectrum, SfconvWindow};
use crate::spring_input::parse_spring_inp;
use crate::xsph_input::XsphAdvanced;

/// FEFF input projected into typed structures used by the Rust modules.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffDocument {
    /// Root input file.
    pub source: PathBuf,
    /// Active FEFF card names in FEFF token order, using canonical output names
    /// from `itoken_reverse`.
    pub active_cards: Vec<String>,
    /// FEFF card names in parsed input order, using canonical output names from
    /// `itoken_reverse` and preserving repeated cards.
    pub input_cards: Vec<String>,
    /// All `TITLE` lines in read order.
    pub titles: Vec<String>,
    /// Selected absorption edge, when present.
    pub edge: Option<Edge>,
    /// Numeric core-hole selector from legacy `HOLE`.
    pub hole: Option<i32>,
    /// Amplitude reduction factor from `S02`, when present.
    pub s02: Option<f64>,
    /// Real and imaginary final-state corrections from `CORRECTIONS`.
    pub corrections: [f64; 2],
    /// Chemical-shift correction mode from `CHSHIFT`.
    pub chsh_type: i32,
    /// Advanced XSPH/FF2X handoff controls from XSPH-related cards.
    pub xsph_handoff: XsphHandoffControls,
    /// TDLDA and PMBSE advanced XSPH controls.
    pub xsph_advanced: XsphAdvanced,
    /// Configuration-average controls from `CFAVERAGE`.
    pub cfaverage: CfAverage,
    /// Lower bound for core-valence separation search from `CORVAL`, in eV.
    pub corval_emin: f64,
    /// Six execution switches from `CONTROL`, when present.
    pub control: Option<[i32; 6]>,
    /// Six print switches from the common `PRINT` card, when present.
    pub print: Option<[i32; 6]>,
    /// Self-consistent-field settings from `SCF`, when present.
    pub scf: Option<Scf>,
    /// Exchange-correlation settings from `EXCHANGE`, when present.
    pub exchange: Option<Exchange>,
    /// EXAFS energy-grid settings from `EXAFS`, when present.
    pub exafs: Option<Exafs>,
    /// Spectroscopy energy-grid settings used by `xsph`.
    pub spectrum_grid: SpectrumGrid,
    /// Whether the input requests reciprocal-space processing.
    pub reciprocal: bool,
    /// FEFF CIF potential-equivalence selector from `EQUIVALENCE`.
    pub cif_equivalence: i32,
    /// FEFF reciprocal-lattice atom coordinate selector from `COORDINATES`.
    pub coordinate_mode: i32,
    /// Generated reciprocal-space handoff, when direct lattice data is present.
    pub reciprocal_input: Option<ReciprocalInput>,
    /// Band-structure module handoff from `BANDSTRUCTURE`/`BAND`.
    pub band_input: BandInput,
    /// Full-spectrum module handoff from `FULLSPECTRUM`.
    pub full_spectrum_input: FullSpectrumInput,
    /// Screening module handoff from repeated `SCREEN` cards.
    pub screen_input: ScreenInput,
    /// Explicit `EGRID` switch used by `xsph`.
    pub i_grid: i32,
    /// Raw `EGRID` payload rows copied by RDINP into `grid.inp`.
    pub egrid_records: Vec<String>,
    /// Raw `DENSITY` payload rows copied by RDINP into `density.inp`.
    pub density_records: Vec<String>,
    /// Electronic temperature from `TEMP`, in eV.
    pub electronic_temperature: f64,
    /// SCF exchange-correlation selector from `TEMP`/`SCXC`.
    pub iscfxc: i32,
    /// Radial grid spacing from `RGRID`.
    pub rgrid: f64,
    /// Curved-wave amplitude criterion from `CRITERIA`.
    pub critcw: f64,
    /// Plane-wave path criterion from `CRITERIA`.
    pub critpw: f64,
    /// Keep criterion from `PCRITERIA`.
    pub pcritk: f64,
    /// Heap criterion from `PCRITERIA`.
    pub pcrith: f64,
    /// Real-self-energy/real-phase switch from `RSIGMA` or `RPHASES`.
    pub lreal: i32,
    /// Curved-wave expansion order from `IORDER`/`IORD`.
    pub iorder: i32,
    /// Whether GENFMT should write n-star data for collinear polarization.
    pub nstar: bool,
    /// Multiple-pole self-energy model selector from `MPSE`/`PLASMON`.
    pub i_plsmn: i32,
    /// Number of poles for MPSE.
    pub n_poles: i32,
    /// Whether `OPCONS` should build optical constants from the database.
    pub opcons: bool,
    /// Optical-constants database handoff from `OPCONS`/`NUMDENS`/`PREPS`.
    pub opcons_input: OpconsInput,
    /// Whether spectral-function convolution is requested.
    pub sfconv: bool,
    /// Spectral-function convolution handoff from `SFCONV`/`SELF`/`SFSE`/`RCONV`.
    pub sfconv_input: SfconvInput,
    /// Whether FF2X should convolve with an excitation spectrum.
    pub many_body_convolution: bool,
    /// Global fine-structure damping and cumulant controls for FF2X/FMS.
    pub fine_structure_damping: FineStructureDamping,
    /// Unfreeze f-electrons in the potential stage.
    pub unfreezef: bool,
    /// Use external muffin-tin potentials from `EXTPOT`.
    pub external_pot: bool,
    /// Restart potential generation from a prior `pot.bin` via `RESTART`.
    pub restart_from_pot_bin: bool,
    /// Atomic-configuration source selector for `pot.inp`.
    pub config_type: i32,
    /// Raw `CONFIG card` payload rows copied into `config.inp`.
    pub config_records: Vec<String>,
    /// Whether ionicity warnings are requested.
    pub warn_ion: bool,
    /// Use finite-nucleus atomic wavefunctions for high-Z calculations.
    pub finite_nucleus: bool,
    /// Thermal-SCF integration controls from `SCFTH`.
    pub scf_thermal: ScfThermal,
    /// Optional SCF radius ramp from `SCFR`.
    pub scf_ramp: ScfRamp,
    /// POT SCF convergence tolerances from `TOLS`.
    pub scf_tolerances: ScfTolerances,
    /// FEFF core-hole treatment selector (`nohole`) from `NOHOLE`/`COREHOLE`.
    pub nohole: i32,
    /// Remove potential jumps at muffin-tin radii when `JUMPRM` is present.
    pub jump_removal: bool,
    /// FEFF spectroscopy selector (`ispec`) derived from spectroscopy cards.
    pub ispec: i32,
    /// FEFF polarization mode (`ipol`) for dichroism and polarization cards.
    pub ipol: i32,
    /// Multipole transition selector from `MULTIPOLE`.
    pub le2: i32,
    /// Secondary multipole selector.
    pub l2lp: i32,
    /// Ellipticity from `ELLIPTICITY`.
    pub ellipticity: f64,
    /// Linear polarization vector from `POLARIZATION`.
    pub polarization_vector: [f64; 3],
    /// Incident propagation vector from `ELLIPTICITY`.
    pub incidence_vector: [f64; 3],
    /// Spin selector from `SPIN`.
    pub spin: i32,
    /// Spin vector from `SPIN`, defaulting to z for spin-polarized runs.
    pub spin_vector: [f64; 3],
    /// Disable spectrum normalization for `ABSOLUTE`, ELNES, and EXELFS runs.
    pub absolute: bool,
    /// Full multiple-scattering settings from `FMS`, when present.
    pub fms: Option<Fms>,
    /// Constrained random phase approximation settings from `CRPA`.
    pub crpa: Crpa,
    /// Compton-profile settings from `COMPTON`, `RHOZZP`, and `CGRID`.
    pub compton: Compton,
    /// Hubbard correction settings from `HUBBARD`.
    pub hubbard: Hubbard,
    /// EELS/ELNES/EXELFS settings from the EELS card family.
    pub eels: Eels,
    /// RIXS module settings from `RIXS` and multi-edge `EDGE` cards.
    pub rixs: Rixs,
    /// NRIXS momentum-transfer settings.
    pub nrixs: Option<Nrixs>,
    /// Mixed dynamic form-factor settings from `MDFF`.
    pub mdff: Mdff,
    /// Debye-Waller settings from `DEBYE`, when present.
    pub debye: Option<Debye>,
    /// Original auxiliary `spring.inp` text required by DEBYE EMM/RM runs.
    pub spring_input_text: Option<String>,
    /// Validated auxiliary dynamical-matrix text required by DMDW runs.
    pub dym_input: Option<AuxiliaryTextFile>,
    /// Path expansion radius from `RPATH`/`RMAX`, when present.
    pub rpath: Option<f64>,
    /// Maximum path leg count from `NLEG`.
    pub nleg: Option<i32>,
    /// PATH symmetry case from `SYMMETRY`, or FEFF's default `-1`.
    pub path_symmetry: i32,
    /// Suppress `geom.dat` output when `NOGEOM` is present.
    pub no_geom: bool,
    /// Coordinate scale factor from `RMULTIPLIER`.
    pub r_multiplier: f64,
    /// Dynamic allocation limits from `DIMS`, when present.
    pub dims: Option<DimensionLimits>,
    /// Local density of states settings from `LDOS`, when present.
    pub ldos: Option<Ldos>,
    /// Interstitial-potential settings from `INTERSTITIAL`, when present.
    pub interstitial: Option<Interstitial>,
    /// Automatic overlap factor from `AFOLP`.
    pub afolp: f64,
    /// Manual overlap factors from `FOLP` cards.
    pub overlap_factors: Vec<OverlapFactor>,
    /// Per-potential ionization values from `ION` cards.
    pub ionizations: Vec<Ionization>,
    /// Approximate overlap-shell geometry from `OVERLAP` cards.
    pub overlap_shells: Vec<OverlapShell>,
    /// Explicit single-scattering paths from `SS` cards.
    pub single_scattering_paths: Vec<SingleScatteringPath>,
    /// Rows from `POTENTIALS`/`POTENTIAL`.
    pub potentials: Vec<Potential>,
    /// Rows from `ATOMS`/`ATOM`.
    pub atoms: Vec<Atom>,
}

/// Absorption edge label, normalized to uppercase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    pub label: String,
}

/// Self-consistent-field control values from the `SCF` card.
#[derive(Debug, Clone, PartialEq)]
pub struct Scf {
    /// SCF cluster radius in Angstrom.
    pub radius: f64,
    /// FMS switch for the SCF cycle.
    pub lfms: i32,
    /// Maximum SCF iterations.
    pub iterations: i32,
    /// Broyden convergence accelerator.
    pub ca: f64,
    /// Broyden mixing history length.
    pub nmix: i32,
    /// Core-valence separation energy.
    pub ecv: f64,
    /// Coulomb potential mode.
    pub icoul: i32,
}

/// Exchange-correlation control values from the `EXCHANGE` card.
#[derive(Debug, Clone, PartialEq)]
pub struct Exchange {
    /// FEFF exchange-correlation model selector.
    pub ixc: i32,
    /// Real potential shift.
    pub vr0: f64,
    /// Imaginary potential shift.
    pub vi0: f64,
    /// Optional exchange model used for the initial state.
    pub ixc0: Option<i32>,
}

/// EXAFS control values from the `EXAFS` card.
#[derive(Debug, Clone, PartialEq)]
pub struct Exafs {
    /// Maximum photoelectron wave number used for the high-energy grid.
    pub xkmax: f64,
}

/// Energy-grid controls shared by near-edge spectroscopy cards.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectrumGrid {
    /// Initial-state exchange selector written to `xsph.inp`.
    pub ixc0: i32,
    /// High-energy k-grid step.
    pub xkstep: f64,
    /// Maximum k value, or an energy bound for emission-style cards.
    pub xkmax: f64,
    /// Near-edge imaginary energy step/broadening.
    pub vixan: f64,
}

impl Default for SpectrumGrid {
    fn default() -> Self {
        Self {
            ixc0: 0,
            xkstep: 0.07,
            xkmax: 20.0,
            vixan: 0.0,
        }
    }
}

/// Global fine-structure damping and cumulant controls for FF2X/FMS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FineStructureDamping {
    /// First/third-cumulant expansion factor from `SIG3`.
    pub alphat: f64,
    /// Einstein temperature used with `SIG3`.
    pub thetae: f64,
    /// Global mean-square Debye-Waller factor from `SIG2`, in Angstrom^2.
    pub sig2g: f64,
    /// Global k-dependent Debye-Waller factor from `SIGGK`, in Angstrom.
    pub sig_gk: f64,
}

/// Advanced XSPH/FF2X handoff controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsphHandoffControls {
    /// Core-hole broadening mode from `CHBROADENING`.
    pub core_hole_broadening: i32,
    /// Matrix-element core-state override from `ICORE`.
    pub core_state: i32,
    /// Static dielectric constant from `EPS0`.
    pub eps0: f64,
    /// Band gap from `EGAP`, in eV.
    pub egap: f64,
    /// Manual core-hole lifetime from `CHWIDTH`, in eV.
    pub core_hole_width: Option<f64>,
    /// Whether `SETEDGE` should use tabulated excitation energies.
    pub set_edge: bool,
    /// Whether `RLPRINT` should make XSPH print radial wavefunctions.
    pub print_radial_wavefunctions: bool,
}

/// Full multiple-scattering control values from the `FMS` card.
#[derive(Debug, Clone, PartialEq)]
pub struct Fms {
    /// Cluster radius for the FMS calculation in Angstrom.
    pub radius: f64,
    /// FMS angular-momentum convergence switch.
    pub lfms: i32,
    /// Matrix inversion strategy selector.
    pub minv: i32,
    /// First FMS convergence tolerance.
    pub toler1: f64,
    /// Second FMS convergence tolerance.
    pub toler2: f64,
    /// Direct-space cutoff radius.
    pub rdirec: f64,
}

/// Constrained random phase approximation control values from `CRPA`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crpa {
    /// Whether to run the CRPA module.
    pub enabled: bool,
    /// Angular momentum channel used by CRPA.
    pub l: i32,
    /// Real-space cutoff radius.
    pub rcut: f64,
}

impl Default for Crpa {
    fn default() -> Self {
        Self {
            enabled: false,
            l: 3,
            rcut: 1.600_000_023_841_858,
        }
    }
}

/// Compton-profile control values from the `COMPTON` card family.
#[derive(Debug, Clone, PartialEq)]
pub struct Compton {
    /// Whether to calculate the Compton profile.
    pub do_compton: bool,
    /// Whether to calculate the rho(z,z') slice.
    pub do_rhozzp: bool,
    /// Whether to force recalculation of j(z,z').
    pub force_jzzp: bool,
    /// Maximum momentum transfer.
    pub pqmax: f64,
    /// Number of momentum-grid points.
    pub npq: i32,
    /// Radial spatial grid size.
    pub ns: i32,
    /// Angular grid size.
    pub nphi: i32,
    /// z-grid size.
    pub nz: i32,
    /// z'-grid size.
    pub nzp: i32,
    /// Maximum z' coordinate.
    pub zpmax: f64,
}

/// Hubbard correction controls written to `hubbard.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hubbard {
    /// Module switch used by the Hubbard reader.
    pub i_hubbard: i32,
    /// LDOS Hubbard switch.
    pub mldos_hubb: i32,
    /// Hubbard U parameter in eV.
    pub u: f64,
    /// Hubbard J parameter in eV.
    pub j: f64,
    /// Fermi-level shift in eV.
    pub fermi_shift: f64,
    /// Angular momentum channel for the correction.
    pub l: i32,
}

impl Default for Hubbard {
    fn default() -> Self {
        Self {
            i_hubbard: 1,
            mldos_hubb: 1,
            u: 0.0,
            j: 0.0,
            fermi_shift: 0.0,
            l: 0,
        }
    }
}

/// EELS/ELNES/EXELFS control values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eels {
    /// Whether to calculate ELNES/EXELFS.
    pub enabled: bool,
    /// Average over sample orientation.
    pub average: i32,
    /// Relativistic q-vector switch.
    pub relativistic: i32,
    /// Cross-term switch.
    pub cross_terms: i32,
    /// Input source selector.
    pub input: i32,
    /// Spectrum column selector.
    pub spectrum_column: i32,
    /// First polarization index.
    pub polarization_min: i32,
    /// Polarization index step.
    pub polarization_step: i32,
    /// Last polarization index.
    pub polarization_max: i32,
    /// Beam energy in eV.
    pub beam_energy: f64,
    /// Normalized incident beam vector.
    pub beam_direction: [f64; 3],
    /// Collection semiangle in radians.
    pub collection_angle: f64,
    /// Convergence semiangle in radians.
    pub convergence_angle: f64,
    /// Radial q-mesh size.
    pub qmesh_radial: i32,
    /// Angular q-mesh size.
    pub qmesh_angular: i32,
    /// Detector position angles in radians.
    pub detector: [f64; 2],
    /// Magic-angle calculation switch.
    pub magic: i32,
    /// Magic-angle energy above threshold.
    pub magic_energy: f64,
}

/// Resonant inelastic x-ray scattering controls from `RIXS`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rixs {
    /// Whether the RIXS module should run.
    pub run: bool,
    /// Optional experimental broadening values from the `RIXS` card, in eV.
    pub gamma_exp: [Option<f64>; 2],
    /// Optional Fermi level from the `RIXS` card, in eV.
    pub xmu: Option<f64>,
    /// Many-body convolution switch, enabled by a `VAL` edge.
    pub mbconv: bool,
    /// RIXS edge labels in FEFF order.
    pub edges: Vec<String>,
}

/// Non-resonant inelastic x-ray scattering controls from `NRIXS`.
#[derive(Debug, Clone, PartialEq)]
pub struct Nrixs {
    /// Number of q vectors.
    pub nq: i32,
    /// Whether FEFF should average over q directions.
    pub qaverage: bool,
    /// First momentum-transfer vector, retained for scalar handoff fields.
    pub qvec: [f64; 3],
    /// First q-vector norm, retained for scalar handoff fields.
    pub qnorm: f64,
    /// Full q-vector table written to `global.inp`.
    pub q_vectors: Vec<NrixsQVector>,
    /// Decomposition limit from `LDEC`.
    pub ldecmx: i32,
    /// Angular momentum limit from `LJMAX`.
    pub lj: i32,
}

/// One `NRIXS` q-vector row with FEFF's complex list weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NrixsQVector {
    /// Momentum-transfer vector components.
    pub vector: [f64; 3],
    /// Stored q-vector norm. FEFF keeps this separate from the vector for `MDFF 2`.
    pub norm: f64,
    /// Complex q weight as `[real, imaginary]`.
    pub weight: [f64; 2],
}

/// Mixed dynamic form-factor controls from `MDFF`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mdff {
    /// FEFF MDFF selector.
    pub imdff: i32,
    /// Requested generated q-prime norm for `MDFF 2`; -1 reuses the q-list.
    pub qqmdff: f64,
    /// Requested q/q-prime angle in degrees for `MDFF 2`.
    pub cosmdff_angle: f64,
}

impl Default for Eels {
    fn default() -> Self {
        Self {
            enabled: false,
            average: 0,
            relativistic: 1,
            cross_terms: 1,
            input: 1,
            spectrum_column: 4,
            polarization_min: 1,
            polarization_step: 1,
            polarization_max: 1,
            beam_energy: 0.0,
            beam_direction: [0.0; 3],
            collection_angle: 0.0,
            convergence_angle: 0.0,
            qmesh_radial: 0,
            qmesh_angular: 0,
            detector: [0.0; 2],
            magic: 0,
            magic_energy: 0.0,
        }
    }
}

impl Default for Rixs {
    fn default() -> Self {
        Self {
            run: false,
            gamma_exp: [None, None],
            xmu: None,
            mbconv: false,
            edges: vec!["NULL".to_string()],
        }
    }
}

impl Default for Compton {
    fn default() -> Self {
        Self {
            do_compton: false,
            do_rhozzp: false,
            force_jzzp: false,
            pqmax: 5.0,
            npq: 1000,
            ns: 32,
            nphi: 32,
            nz: 32,
            nzp: 144,
            zpmax: 10.0,
        }
    }
}

/// Debye-Waller control values from the `DEBYE` card.
#[derive(Debug, Clone, PartialEq)]
pub struct Debye {
    /// Sample temperature in Kelvin.
    pub temperature: f64,
    /// Debye temperature in Kelvin.
    pub debye_temperature: f64,
    /// Debye-Waller calculation mode.
    pub idwopt: i32,
    /// Dynamical matrix filename used when `idwopt == 5`.
    pub dym_file: Option<String>,
    /// Lanczos recursion order for FEFF's standalone `dmdw` handoff.
    pub dmdw_order: i32,
    /// Dynamical-matrix calculation type selector.
    pub dmdw_type: i32,
    /// Path-selection route for the standalone `dmdw` run.
    pub dmdw_route: i32,
}

/// Text input file that must be carried into the FEFF handoff directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuxiliaryTextFile {
    /// Relative output path used by downstream FEFF-compatible modules.
    pub output_name: String,
    /// Original file text, preserved byte-for-byte apart from Rust `String`
    /// UTF-8 validation.
    pub text: String,
}

/// Local-density-of-states control values from the `LDOS` card.
#[derive(Debug, Clone, PartialEq)]
pub struct Ldos {
    /// Lower energy bound.
    pub emin: f64,
    /// Upper energy bound.
    pub emax: f64,
    /// Imaginary energy broadening.
    pub eimag: f64,
    /// Number of energy mesh points.
    pub neldos: i32,
    /// LDOS output type selector.
    pub ldostype: i32,
}

/// Interstitial calculation controls from the `INTERSTITIAL` card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interstitial {
    /// Interstitial mode selector.
    pub mode: i32,
    /// FEFF volume scale before normalization by the atom statistics.
    pub volume_scale: f64,
}

/// Thermal-SCF integration controls from the `SCFTH` card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScfThermal {
    /// Thermal-SCF algorithm selector.
    pub iscfth: i32,
    /// Electron-count tolerance.
    pub xntol: f64,
    /// Number of chemical-potential iterations.
    pub nmu: i32,
    /// Number of energy-grid points.
    pub negrid: i32,
    /// Upper energy-grid bound in eV.
    pub emaxscf: f64,
}

/// SCF radius ramp requested by the `SCFR` card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScfRamp {
    /// Whether SCF radius ramping is enabled.
    pub enabled: bool,
    /// Starting SCF radius.
    pub rfms_start: f64,
    /// Number of ramp steps.
    pub nramp: i32,
}

/// POT self-consistency tolerances from the `TOLS` card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScfTolerances {
    /// Chemical-potential convergence tolerance.
    pub tolmu: f64,
    /// Charge convergence tolerance.
    pub tolq: f64,
    /// Potential-charge convergence tolerance.
    pub tolqp: f64,
}

/// User-requested dynamic allocation caps from the `DIMS` card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionLimits {
    /// Maximum FMS cluster size requested by the input.
    pub nclusx: i32,
    /// Maximum angular momentum requested by the input.
    pub lx: i32,
}

/// Manual muffin-tin overlap factor requested by a FEFF `FOLP` card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlapFactor {
    /// Potential index affected by the manual factor.
    pub potential_index: i32,
    /// Multiplicative muffin-tin overlap factor.
    pub factor: f64,
}

/// Ionization requested by a FEFF `ION` card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ionization {
    /// Potential index affected by the ionization value.
    pub potential_index: i32,
    /// Effective ionization value.
    pub value: f64,
}

/// One shell row from a FEFF `OVERLAP` block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlapShell {
    /// Potential index being overlapped, from the `OVERLAP iph` card.
    pub potential_index: i32,
    /// Potential index of atoms in this overlap shell.
    pub neighbor_potential_index: i32,
    /// Number of atoms in the shell.
    pub count: i32,
    /// Shell distance in Angstroms.
    pub distance: f64,
}

/// Explicit single-scattering path requested by a FEFF `SS` card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingleScatteringPath {
    /// FEFF path index.
    pub index: i32,
    /// Unique potential index for the scatterer.
    pub potential_index: i32,
    /// Path degeneracy/multiplicity.
    pub degeneracy: f64,
    /// Half path length in Angstroms.
    pub distance: f64,
}

/// One row of the FEFF `POTENTIALS` table.
#[derive(Debug, Clone, PartialEq)]
pub struct Potential {
    /// FEFF potential index.
    pub ipot: i32,
    /// Parsed atomic number.
    pub z: Option<i32>,
    /// Original Z token, preserved for diagnostics and round-trip context.
    pub z_token: String,
    /// Element or user tag.
    pub tag: Option<String>,
    /// Optional phase-shift angular momentum limit.
    pub lmax1: Option<i32>,
    /// Optional FMS angular momentum limit.
    pub lmax2: Option<i32>,
    /// Optional stoichiometry/count field.
    pub xnatph: Option<f64>,
    /// Optional spin field.
    pub spinph: Option<f64>,
}

/// One row of the FEFF `ATOMS` table.
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// Cartesian x coordinate in Angstrom.
    pub x: f64,
    /// Cartesian y coordinate in Angstrom.
    pub y: f64,
    /// Cartesian z coordinate in Angstrom.
    pub z: f64,
    /// Potential index for this atom.
    pub ipot: i32,
    /// Optional atom tag.
    pub tag: Option<String>,
    /// Optional distance field from the input; generated if absent.
    pub distance: Option<f64>,
    /// Optional trailing index.
    pub index: Option<usize>,
}

impl FeffDocument {
    /// Extract the currently supported typed card subset from parsed input.
    pub fn from_input(input: &FeffInput) -> Result<Self> {
        let active_cards = parse_active_cards(input);
        let input_cards = parse_input_cards(input);
        validate_feff_consistency(input, &active_cards)?;
        let titles = parse_titles(input)?;
        let edge = parse_edge(input)?;
        let (hole, hole_s02) = parse_hole(input)?;
        let s02 = parse_scalar_card(input, "S02")?.or(hole_s02);
        let corrections = parse_corrections(input)?;
        let chsh_type = parse_chsh_type(input)?;
        let xsph_handoff = parse_xsph_handoff(input)?;
        let xsph_advanced = parse_xsph_advanced(input)?;
        let (mut cfaverage, cfaverage_requested) = parse_cfaverage(input)?;
        let corval_emin = parse_corval_emin(input)?;
        let control = parse_i32_6(input, "CONTROL")?;
        let print = parse_i32_6(input, "PRINT")?;
        let mut scf = parse_scf(input)?;
        let exchange = parse_exchange(input)?;
        let exafs = parse_exafs(input)?;
        let ispec = parse_ispec(input);
        let ipol = parse_ipol(input);
        let (le2, l2lp) = parse_multipole(input)?;
        let (ellipticity, incidence_vector) = parse_ellipticity(input)?;
        let polarization_vector = parse_polarization_vector(input)?;
        let (spin, spin_vector) = parse_spin(input)?;
        let spectrum_grid = parse_spectrum_grid(input, exchange.as_ref(), ispec)?;
        let reciprocal = parse_reciprocal_space(input);
        let cif_equivalence = parse_cif_equivalence(input)?;
        let coordinate_mode = parse_coordinate_mode(input)?;
        let band_input = parse_band_input(input)?;
        let full_spectrum_input = parse_full_spectrum_input(&active_cards);
        let screen_input = parse_screen_input(input)?;
        let i_grid = i32::from(card_by_feff_name(input, "EGRID").is_some());
        let egrid_records = parse_egrid_records(input)?;
        let density_records = parse_density_records(input)?;
        let (electronic_temperature, iscfxc) = parse_temp(input)?;
        let rgrid = parse_scalar_card(input, "RGRID")?.unwrap_or(0.05);
        let (critcw, critpw) = parse_criteria(input)?;
        let (pcritk, pcrith) = parse_pcriteria(input)?;
        let lreal = parse_lreal(input);
        let iorder = parse_iorder(input)?;
        let nstar = active_cards.iter().any(|card| card == "NSTAR") && ipol == 1;
        let (i_plsmn, n_poles) = parse_mpse(input)?;
        let opcons = active_cards.iter().any(|card| card == "OPCONS");
        let many_body_convolution = active_cards.iter().any(|card| card == "MBCONV");
        let fine_structure_damping = parse_fine_structure_damping(input)?;
        let unfreezef = card_by_feff_name(input, "UNFREEZEF").is_some();
        let external_pot = active_cards.iter().any(|card| card == "EXTPOT");
        let restart_from_pot_bin = active_cards.iter().any(|card| card == "RESTART");
        let config_type = parse_config_type(input)?;
        let config_records = parse_config_records(input)?;
        let warn_ion = active_cards.iter().any(|card| card == "WARN");
        let finite_nucleus = active_cards.iter().any(|card| card == "HIGHZ");
        let scf_thermal = parse_scf_thermal(input)?;
        let scf_ramp = parse_scf_ramp(input)?;
        let scf_tolerances = parse_scf_tolerances(input)?;
        let nohole = parse_nohole(input)?;
        let jump_removal = active_cards.iter().any(|card| card == "JUMPRM");
        let absolute = card_by_feff_name(input, "ABSOLUTE").is_some();
        let mut fms = parse_fms(input)?;
        let crpa = parse_crpa(input)?;
        let compton = parse_compton(input)?;
        let hubbard = parse_hubbard(input)?;
        let eels = parse_eels(input)?;
        let rixs = parse_rixs(input)?;
        let mut nrixs = parse_nrixs(input)?;
        let mdff = parse_mdff(input, &mut nrixs, &eels)?;
        let nohole = if compton.do_compton || compton.do_rhozzp {
            0
        } else {
            nohole
        };
        let debye = parse_debye(input)?;
        let spring_input_text = parse_spring_input_text(input, debye.as_ref())?;
        let dym_input = parse_dym_input(input, debye.as_ref())?;
        let mut rpath = parse_rpath(input)?;
        let mut overlap_shells = parse_overlap_shells(input)?;
        let mut single_scattering_paths = parse_single_scattering_paths(input)?;
        if !single_scattering_paths.is_empty() && card_by_feff_name(input, "OVERLAP").is_none() {
            return Err(IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "SS cards require an OVERLAP card".to_string(),
            });
        }
        if !single_scattering_paths.is_empty() && overlap_shells.is_empty() {
            return Err(IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "SS cards require OVERLAP shell rows".to_string(),
            });
        }
        let cif_cluster_radius = cif_cluster_radius(scf.as_ref(), fms.as_ref(), rpath);
        let nleg = parse_nleg(input)?;
        let path_symmetry = parse_path_symmetry(input)?;
        let no_geom = active_cards.iter().any(|card| card == "NOGEOM");
        let r_multiplier = parse_scalar_card(input, "RMULT")?.unwrap_or(1.0);
        if r_multiplier != 1.0 {
            if let Some(scf) = &mut scf {
                scf.radius *= r_multiplier;
            }
            if let Some(fms) = &mut fms {
                fms.radius *= r_multiplier;
            }
            if let Some(rpath) = &mut rpath {
                *rpath *= r_multiplier;
            }
            for shell in &mut overlap_shells {
                shell.distance *= r_multiplier;
            }
            for path in &mut single_scattering_paths {
                path.distance *= r_multiplier;
            }
        }
        let dims = parse_dims(input)?;
        let ldos = parse_ldos(input)?;
        let interstitial = parse_interstitial(input)?;
        let afolp = parse_afolp(input)?;
        let overlap_factors = parse_overlap_factors(input)?;
        let ionizations = parse_ionizations(input)?;
        let mut potentials = parse_potentials(input)?;
        let input_atoms = parse_atoms(input)?;
        let mut atoms = input_atoms.clone();
        let cif_cluster = parse_cif_cluster(
            input,
            cif_cluster_radius,
            potentials.is_empty() || atoms.is_empty(),
            cif_equivalence,
        )?;
        if potentials.is_empty() {
            potentials = cif_cluster
                .as_ref()
                .map(cif_cluster_potentials)
                .transpose()?
                .unwrap_or_default();
        }
        if atoms.is_empty() {
            atoms = cif_cluster
                .as_ref()
                .map(cif_cluster_atoms)
                .unwrap_or_default();
        } else if let Some(lattice_atoms) =
            parse_lattice_cluster_atoms(input, &input_atoms, cif_cluster_radius, reciprocal)?
        {
            atoms = lattice_atoms;
        }
        let atoms: Vec<Atom> = atoms
            .into_iter()
            .map(|mut atom| {
                atom.x *= r_multiplier;
                atom.y *= r_multiplier;
                atom.z *= r_multiplier;
                atom.distance = atom.distance.map(|distance| distance * r_multiplier);
                atom
            })
            .collect();
        if !overlap_shells.is_empty() && !atoms.is_empty() {
            return Err(IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "cannot use ATOMS and OVERLAP in the same input".to_string(),
            });
        }
        if overlap_shells
            .iter()
            .any(|shell| shell.potential_index == 0)
        {
            cfaverage = default_cfaverage();
        } else {
            cfaverage = effective_cfaverage(cfaverage, cfaverage_requested, &atoms);
        }
        ensure_cfaverage_absorber_potential(input, &mut potentials, cfaverage)?;
        let reciprocal_input =
            parse_reciprocal_input(input, nohole, &input_atoms, reciprocal, cif_equivalence)?;
        let opcons_input = parse_opcons_input(input, opcons, &potentials)?;
        let sfconv_input = parse_sfconv_input(input, ispec, ipol, spin, print)?;
        let sfconv = sfconv_input.control.msfconv != 0;

        Ok(Self {
            source: input.source.clone(),
            active_cards,
            input_cards,
            titles,
            edge,
            hole,
            s02,
            corrections,
            chsh_type,
            xsph_handoff,
            xsph_advanced,
            cfaverage,
            corval_emin,
            control,
            print,
            scf,
            exchange,
            exafs,
            spectrum_grid,
            reciprocal,
            cif_equivalence,
            coordinate_mode,
            reciprocal_input,
            band_input,
            full_spectrum_input,
            screen_input,
            i_grid,
            egrid_records,
            density_records,
            electronic_temperature,
            iscfxc,
            rgrid,
            critcw,
            critpw,
            pcritk,
            pcrith,
            lreal,
            iorder,
            nstar,
            i_plsmn,
            n_poles,
            opcons,
            opcons_input,
            sfconv,
            sfconv_input,
            many_body_convolution,
            fine_structure_damping,
            unfreezef,
            external_pot,
            restart_from_pot_bin,
            config_type,
            config_records,
            warn_ion,
            finite_nucleus,
            scf_thermal,
            scf_ramp,
            scf_tolerances,
            nohole,
            jump_removal,
            ispec,
            ipol,
            le2,
            l2lp,
            ellipticity,
            polarization_vector,
            incidence_vector,
            spin,
            spin_vector,
            absolute,
            fms,
            crpa,
            compton,
            hubbard,
            eels,
            rixs,
            nrixs,
            mdff,
            debye,
            spring_input_text,
            dym_input,
            rpath,
            nleg,
            path_symmetry,
            no_geom,
            r_multiplier,
            dims,
            ldos,
            interstitial,
            afolp,
            overlap_factors,
            ionizations,
            overlap_shells,
            single_scattering_paths,
            potentials,
            atoms,
        })
    }
}

fn parse_active_cards(input: &FeffInput) -> Vec<String> {
    let mut cards = input
        .cards()
        .filter_map(|line| match &line.kind {
            LineKind::Card { keyword, .. } => feff_card_token(keyword),
            LineKind::SectionData { .. } => None,
        })
        .collect::<Vec<_>>();
    cards.sort_by_key(|(token, _)| *token);
    cards.dedup_by_key(|(token, _)| *token);
    cards
        .into_iter()
        .map(|(_, display)| display.to_string())
        .collect()
}

fn parse_input_cards(input: &FeffInput) -> Vec<String> {
    input
        .cards()
        .filter_map(|line| match &line.kind {
            LineKind::Card { keyword, .. } => {
                feff_card_token(keyword).map(|(_, display)| display.to_string())
            }
            LineKind::SectionData { .. } => None,
        })
        .collect()
}

fn validate_feff_consistency(input: &FeffInput, active_cards: &[String]) -> Result<()> {
    let spectroscopy_cards = [
        "XANES", "EXAFS", "XES", "DANES", "FPRIME", "ELNES", "EXELFS",
    ];
    let active_spectroscopy = spectroscopy_cards
        .iter()
        .filter(|card| active_card(active_cards, card))
        .count();
    if active_spectroscopy > 1
        && let Some(card) = spectroscopy_cards
            .iter()
            .find(|card| active_card(active_cards, card))
    {
        return Err(parse_error(
            required_card_line(input, card)?,
            "ERROR more than one type of spectroscopy selected",
        ));
    }

    if active_card(active_cards, "NRIXS") {
        let nrixs_line = required_card_line(input, "NRIXS")?;
        let xanes_or_exafs = ["XANES", "EXAFS"]
            .iter()
            .filter(|card| active_card(active_cards, card))
            .count();
        if xanes_or_exafs != 1 {
            return Err(parse_error(
                nrixs_line,
                "NRIXS must be combined with XANES or EXAFS",
            ));
        }
        if let Some(card) = ["FPRIME", "XES", "DANES", "ELNES", "EXELFS"]
            .iter()
            .find(|card| active_card(active_cards, card))
        {
            return Err(parse_error(
                required_card_line(input, card)?,
                "NRIXS combined with incompatible spectroscopy card",
            ));
        }
        if active_card(active_cards, "MULT") {
            return Err(parse_error(
                required_card_line(input, "MULT")?,
                "you cannot combine NRIXS and MULTIPOLE",
            ));
        }
        if let Some(card) = [
            "ELLIPTICITY",
            "POLARIZATION",
            "NSTAR",
            "SPIN",
            "CFAVERAGE",
            "XMCD",
            "RPHASES",
            "TDLDA",
            "PMBSE",
            "HUBBARD",
        ]
        .iter()
        .find(|card| active_card(active_cards, card))
        {
            return Err(parse_error(
                required_card_line(input, card)?,
                "card is explicitly forbidden for NRIXS",
            ));
        }
    } else if active_card(active_cards, "LJMAX") || active_card(active_cards, "LDECMX") {
        let line = card_by_feff_name(input, "LJMAX")
            .or_else(|| card_by_feff_name(input, "LDECMX"))
            .ok_or_else(|| IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "LDEC/LJMAX card not found".to_string(),
            })?;
        return Err(parse_error(
            line,
            "LDEC and LJMAX cards only allowed with NRIXS",
        ));
    }

    if active_card(active_cards, "RECIPROCAL") {
        let reciprocal_line = required_card_line(input, "RECIPROCAL")?;
        if !(active_card(active_cards, "KMESH") && active_card(active_cards, "TARGET")) {
            return Err(parse_error(
                reciprocal_line,
                "KMESH and TARGET are required for RECIPROCAL card",
            ));
        }

        let structure_source_count = ["LATTICE", "CIF"]
            .iter()
            .filter(|card| active_card(active_cards, card))
            .count();
        if structure_source_count != 1 {
            return Err(parse_error(
                reciprocal_line,
                "use either LATTICE or CIF with RECIPROCAL card",
            ));
        }
    }

    if active_card(active_cards, "CGRID")
        && !(active_card(active_cards, "COMPTON") || active_card(active_cards, "RHOZZP"))
    {
        return Err(parse_error(
            required_card_line(input, "CGRID")?,
            "Cannot use CGRID without COMPTON or RHOZZP.  Exiting.",
        ));
    }

    if active_card(active_cards, "HUBBARD") && active_card(active_cards, "RECIPROCAL") {
        return Err(parse_error(
            required_card_line(input, "HUBBARD")?,
            "Cannot use RECIPROCAL with HUBBARD.",
        ));
    }

    Ok(())
}

fn active_card(active_cards: &[String], canonical: &str) -> bool {
    active_cards.iter().any(|card| card == canonical)
}

fn required_card_line<'a>(input: &'a FeffInput, canonical: &str) -> Result<&'a FeffLine> {
    card_by_feff_name(input, canonical).ok_or_else(|| IoError::Parse {
        path: input.source.clone(),
        line: 0,
        message: format!("{canonical} card not found"),
    })
}

fn card_by_feff_name<'a>(input: &'a FeffInput, canonical: &str) -> Option<&'a FeffLine> {
    input.cards().find(|line| {
        if let LineKind::Card { keyword, .. } = &line.kind {
            return keyword == canonical
                || feff_card_token(keyword)
                    .map(|(_, display)| display == canonical)
                    .unwrap_or(false);
        }
        false
    })
}

fn feff_card_token(keyword: &str) -> Option<(i32, &'static str)> {
    let upper = keyword.to_ascii_uppercase();
    let w = upper.get(..upper.len().min(4)).unwrap_or("");
    match w {
        "ATOM" => Some((1, "ATOMS")),
        "HOLE" => Some((2, "HOLE")),
        "OVER" => Some((3, "OVERLAP")),
        "CONT" => Some((4, "CONTROL")),
        "EXCH" => Some((5, "EXCHANGE")),
        "ION" => Some((6, "ION")),
        "TITL" => Some((7, "TITLE")),
        "FOLP" => Some((8, "FOLP")),
        "RPAT" | "RMAX" => Some((9, "RPATH")),
        "DEBY" => Some((10, "DEBYE")),
        "RMUL" => Some((11, "RMULT")),
        "SS" => Some((12, "SS")),
        "PRIN" => Some((13, "PRINT")),
        "POTE" => Some((14, "POTENTIALS")),
        "NLEG" => Some((15, "NLEG")),
        "CRIT" => Some((16, "CRITERIA")),
        "NOGE" => Some((17, "NOGEOM")),
        "IORD" => Some((18, "IORD")),
        "PCRI" => Some((19, "PCRITERIA")),
        "SIG2" => Some((20, "SIG2")),
        "XANE" => Some((21, "XANES")),
        "CORR" => Some((22, "CORRECTIONS")),
        "AFOL" => Some((23, "AFOLP")),
        "EXAF" => Some((24, "EXAFS")),
        "POLA" => Some((25, "POLARIZATION")),
        "ELLI" => Some((26, "ELLIPTICITY")),
        "RGRI" => Some((27, "RGRID")),
        "RPHA" => Some((28, "RPHASES")),
        "NSTA" => Some((29, "NSTAR")),
        "NOHO" => Some((30, "NOHOLE")),
        "SIG3" => Some((31, "SIG3")),
        "JUMP" => Some((32, "JUMPRM")),
        "MBCO" => Some((33, "MBCONV")),
        "SPIN" => Some((34, "SPIN")),
        "EDGE" => Some((35, "EDGE")),
        "SCF" => Some((36, "SCF")),
        "FMS" => Some((37, "FMS")),
        "LDOS" => Some((38, "LDOS")),
        "INTE" => Some((39, "INTERSTITIAL")),
        "CFAV" => Some((40, "CFAVERAGE")),
        "S02" => Some((41, "S02")),
        "XES" => Some((42, "XES")),
        "DANE" => Some((43, "DANES")),
        "FPRI" => Some((44, "FPRIME")),
        "RSIG" => Some((45, "RSIGMA")),
        "XNCD" | "XMCD" => Some((46, "XMCD")),
        "MULT" => Some((47, "MULT")),
        "UNFR" => Some((48, "UNFREEZEF")),
        "TDLD" => Some((49, "TDLDA")),
        "PMBS" => Some((50, "PMBSE")),
        "PLAS" | "MPSE" => Some((51, "MPSE")),
        "SO2C" | "SFCO" => Some((52, "SFCONV")),
        "SELF" => Some((53, "SELF")),
        "SFSE" => Some((54, "SFSE")),
        "RCON" => Some((55, "RCONV")),
        "ELNE" => Some((56, "ELNES")),
        "EXEL" => Some((57, "EXELFS")),
        "MAGI" => Some((58, "MAGIC")),
        "ABSO" => Some((59, "ABSOLUTE")),
        "SYMM" => Some((60, "SYMMETRY")),
        "REAL" => Some((61, "REAL")),
        "RECI" => Some((62, "RECIPROCAL")),
        "SGRO" => Some((63, "SGROUP")),
        "LATT" => Some((64, "LATTICE")),
        "KMES" => Some((65, "KMESH")),
        "STRF" => Some((66, "STRFAC")),
        "BAND" => Some((67, "BAND")),
        "CORE" => Some((68, "COREHOLE")),
        "MARK" | "TARG" => Some((71, "TARGET")),
        "EGRI" => Some((72, "EGRID")),
        "COOR" => Some((73, "COORDINATES")),
        "EXTP" => Some((74, "EXTPOT")),
        "CHBR" => Some((75, "CHBROADENING")),
        "CHSH" => Some((76, "CHSHIFT")),
        "DIMS" => Some((77, "DIMS")),
        "NRIX" => Some((78, "NRIXS")),
        "LJMA" => Some((79, "LJMAX")),
        "LDEC" => Some((80, "LDECMX")),
        "SETE" => Some((81, "SETE")),
        "EPS0" => Some((82, "EPS0")),
        "OPCO" => Some((83, "OPCONS")),
        "NUMD" => Some((84, "NUMD")),
        "PREP" => Some((85, "PREP")),
        "EGAP" => Some((86, "EGAP")),
        "CHWI" => Some((87, "CHWIDTH")),
        "MDFF" => Some((88, "MDFF")),
        "REST" => Some((89, "RESTART")),
        "CONF" => Some((90, "CONFIGURATION")),
        "SCRE" => Some((91, "SCREEN")),
        "CIF" => Some((92, "CIF")),
        "EQUI" => Some((93, "EQUIVALENCE")),
        "COMP" => Some((94, "COMPTON")),
        "RHOZ" => Some((95, "RHOZZP")),
        "CGRI" => Some((96, "CGRID")),
        "CORV" => Some((97, "CORVAL")),
        "SIGG" => Some((98, "SIGGK")),
        "TEMP" => Some((99, "TEMP")),
        "DENS" => Some((100, "DENS")),
        "RIXS" => Some((101, "RIXS")),
        "RLPR" => Some((102, "RLPR")),
        "ICOR" => Some((103, "ICOR")),
        "HUBB" => Some((104, "HUBBARD")),
        "CRPA" => Some((105, "CRPA")),
        "FULL" => Some((106, "FULLSPECTRUM")),
        "SCXC" => Some((107, "SCXC")),
        "HIGH" => Some((108, "HIGHZ")),
        "SCFT" => Some((109, "SCFTH")),
        "WARN" => Some((110, "WARN")),
        "SCFR" => Some((111, "SCFR")),
        "TOLS" => Some((112, "TOLS")),
        _ => None,
    }
}

fn parse_titles(input: &FeffInput) -> Result<Vec<String>> {
    let mut titles = Vec::new();
    for line in input.cards() {
        if let LineKind::Card {
            keyword, raw_args, ..
        } = &line.kind
            && feff_card_token(keyword)
                .map(|(_, display)| display == "TITLE")
                .unwrap_or(false)
        {
            titles.push(raw_args.clone());
        }
    }
    Ok(titles)
}

fn parse_edge(input: &FeffInput) -> Result<Option<Edge>> {
    let Some(line) = card_by_feff_name(input, "EDGE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(label) = args.first() else {
        return Err(parse_error(line, "EDGE requires a label"));
    };
    Ok(Some(Edge {
        label: label.to_ascii_uppercase(),
    }))
}

fn parse_hole(input: &FeffInput) -> Result<(Option<i32>, Option<f64>)> {
    let Some(line) = card_by_feff_name(input, "HOLE") else {
        return Ok((None, None));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_i32(line, args.first())?,
        parse_optional_f64(line, args.get(1))?,
    ))
}

fn parse_scalar_card(input: &FeffInput, keyword: &str) -> Result<Option<f64>> {
    let Some(line) = card_by_feff_name(input, keyword) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(
            line,
            format!("{keyword} requires a numeric value"),
        ));
    };
    Ok(Some(parse_f64(line, value)?))
}

fn parse_optional_i32_card_by_feff_name(
    input: &FeffInput,
    canonical: &str,
    label: &str,
) -> Result<Option<i32>> {
    let Some(line) = card_by_feff_name(input, canonical) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(
            line,
            format!("{label} requires an integer value"),
        ));
    };
    Ok(Some(parse_i32(line, value)?))
}

fn parse_optional_f64_card_by_feff_name(
    input: &FeffInput,
    canonical: &str,
    label: &str,
) -> Result<Option<f64>> {
    let Some(line) = card_by_feff_name(input, canonical) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(
            line,
            format!("{label} requires a numeric value"),
        ));
    };
    let value = parse_f64(line, value)?;
    if !value.is_finite() {
        return Err(parse_error(line, format!("{label} value must be finite")));
    }
    Ok(Some(value))
}

fn parse_single_scattering_paths(input: &FeffInput) -> Result<Vec<SingleScatteringPath>> {
    let mut paths = Vec::new();
    for line in input.cards() {
        if let LineKind::Card { keyword, .. } = &line.kind
            && keyword == "SS"
        {
            let args = card_args(line)?;
            if args.len() < 4 {
                return Err(parse_error(
                    line,
                    "SS requires index, ipot, degeneracy, and rss",
                ));
            }
            let degeneracy = parse_f64(line, &args[2])?;
            let distance = parse_f64(line, &args[3])?;
            if !degeneracy.is_finite() || !distance.is_finite() {
                return Err(parse_error(
                    line,
                    "SS degeneracy and distance must be finite",
                ));
            }
            paths.push(SingleScatteringPath {
                index: parse_i32(line, &args[0])?,
                potential_index: parse_i32(line, &args[1])?,
                degeneracy,
                distance,
            });
        }
    }
    Ok(paths)
}

fn parse_overlap_shells(input: &FeffInput) -> Result<Vec<OverlapShell>> {
    let mut shells = Vec::new();
    let mut current_potential_index = None;
    for line in &input.lines {
        match &line.kind {
            LineKind::Card { keyword, args, .. } if keyword == "OVERLAP" => {
                let Some(value) = args.first() else {
                    return Err(parse_error(line, "OVERLAP requires a potential index"));
                };
                current_potential_index = Some(parse_i32(line, value)?);
            }
            LineKind::SectionData { section, fields } if section == "OVERLAP" => {
                let Some(potential_index) = current_potential_index else {
                    return Err(parse_error(line, "OVERLAP row without an OVERLAP card"));
                };
                if fields.len() < 3 {
                    return Err(parse_error(
                        line,
                        "OVERLAP rows require iphovr, nnovr, and rovr",
                    ));
                }
                let distance = parse_f64(line, &fields[2])?;
                if !distance.is_finite() {
                    return Err(parse_error(line, "OVERLAP distance must be finite"));
                }
                shells.push(OverlapShell {
                    potential_index,
                    neighbor_potential_index: parse_i32(line, &fields[0])?,
                    count: parse_i32(line, &fields[1])?,
                    distance,
                });
            }
            LineKind::Card { .. } | LineKind::SectionData { .. } => {}
        }
    }
    Ok(shells)
}

fn parse_corrections(input: &FeffInput) -> Result<[f64; 2]> {
    let Some(line) = card_by_feff_name(input, "CORRECTIONS") else {
        return Ok([0.0, 0.0]);
    };
    let args = card_args(line)?;
    if args.len() < 2 {
        return Err(parse_error(
            line,
            "CORRECTIONS requires real and imaginary shifts",
        ));
    }
    Ok([parse_f64(line, &args[0])?, parse_f64(line, &args[1])?])
}

fn parse_chsh_type(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "CHSHIFT") else {
        return Ok(0);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "CHSHIFT requires ChSh_Type"));
    };
    parse_i32(line, value)
}

fn parse_xsph_handoff(input: &FeffInput) -> Result<XsphHandoffControls> {
    Ok(XsphHandoffControls {
        core_hole_broadening: parse_optional_i32_card_by_feff_name(
            input,
            "CHBROADENING",
            "CHBROADENING",
        )?
        .unwrap_or(0),
        core_state: parse_optional_i32_card_by_feff_name(input, "ICOR", "ICORE")?.unwrap_or(-1),
        eps0: parse_optional_f64_card_by_feff_name(input, "EPS0", "EPS0")?.unwrap_or(0.0),
        egap: parse_optional_f64_card_by_feff_name(input, "EGAP", "EGAP")?.unwrap_or(0.0),
        core_hole_width: parse_optional_f64_card_by_feff_name(input, "CHWIDTH", "CHWIDTH")?,
        set_edge: card_by_feff_name(input, "SETE").is_some(),
        print_radial_wavefunctions: card_by_feff_name(input, "RLPR").is_some(),
    })
}

fn parse_xsph_advanced(input: &FeffInput) -> Result<XsphAdvanced> {
    let mut advanced = XsphAdvanced {
        izstd: 0,
        ifxc: 0,
        ipmbse: 0,
        itdlda: 0,
        nonlocal: 0,
        ibasis: 0,
    };

    for line in input.cards() {
        let LineKind::Card { keyword, args, .. } = &line.kind else {
            continue;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("TDLDA") => {
                advanced.izstd = 1;
                if let Some(value) = args.first() {
                    advanced.ifxc = parse_i32(line, value)?;
                }
            }
            Some("PMBSE") => {
                advanced.itdlda = 2;
                if let Some(value) = args.first() {
                    advanced.ipmbse = parse_i32(line, value)?;
                }
                if let Some(value) = args.get(1) {
                    advanced.nonlocal = parse_i32(line, value)?;
                }
                if let Some(value) = args.get(2)
                    && advanced.izstd == 0
                {
                    advanced.ifxc = parse_i32(line, value)?;
                }
                if let Some(value) = args.get(3) {
                    advanced.ibasis = parse_i32(line, value)?;
                }
            }
            _ => {}
        }
    }

    Ok(advanced)
}

fn parse_cfaverage(input: &FeffInput) -> Result<(CfAverage, bool)> {
    let mut cfaverage = default_cfaverage();
    let mut found = false;

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        if feff_card_token(keyword).map(|(_, display)| display) != Some("CFAVERAGE") {
            continue;
        }

        found = true;
        let args = card_args(line)?;
        if args.len() < 3 {
            return Err(parse_error(line, "CFAVERAGE requires iphabs nabs rclabs"));
        }
        let mut rclabs = parse_f64(line, &args[2])?;
        if !rclabs.is_finite() {
            return Err(parse_error(line, "CFAVERAGE rclabs must be finite"));
        }
        if rclabs < 0.5 {
            rclabs = default_cfaverage().rclabs;
        }
        cfaverage = CfAverage {
            iphabs: parse_i32(line, &args[0])?,
            nabs: parse_i32(line, &args[1])?,
            rclabs,
        };
    }

    Ok((cfaverage, found))
}

fn default_cfaverage() -> CfAverage {
    CfAverage {
        nabs: 1,
        iphabs: 0,
        rclabs: 100000.0,
    }
}

fn effective_cfaverage(
    mut cfaverage: CfAverage,
    cfaverage_requested: bool,
    atoms: &[Atom],
) -> CfAverage {
    if !cfaverage_requested {
        return cfaverage;
    }

    let absorber_count = atoms
        .iter()
        .filter(|atom| atom.ipot == cfaverage.iphabs || atom.ipot == 0)
        .count() as i32;
    if absorber_count > 0 && (cfaverage.nabs <= 0 || cfaverage.nabs > absorber_count) {
        cfaverage.nabs = absorber_count;
    }
    cfaverage
}

fn ensure_cfaverage_absorber_potential(
    input: &FeffInput,
    potentials: &mut Vec<Potential>,
    cfaverage: CfAverage,
) -> Result<()> {
    if cfaverage.iphabs <= 0 || potentials.iter().any(|potential| potential.ipot == 0) {
        return Ok(());
    }

    let Some(mut absorber) = potentials
        .iter()
        .find(|potential| potential.ipot == cfaverage.iphabs)
        .cloned()
    else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: format!(
                "CFAVERAGE absorber potential {} is missing",
                cfaverage.iphabs
            ),
        });
    };
    absorber.ipot = 0;
    potentials.push(absorber);
    potentials.sort_by_key(|potential| potential.ipot);
    Ok(())
}

fn parse_opcons_input(
    input: &FeffInput,
    run_opcons: bool,
    potentials: &[Potential],
) -> Result<OpconsInput> {
    let mut number_densities = vec![-1.0; opcons_density_count(potentials)];
    let mut print_eps = false;

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("NUMD") => {
                let args = card_args(line)?;
                if args.len() < 2 {
                    return Err(parse_error(line, "NUMDENS requires ipot and numdens"));
                }
                let ipot = parse_i32(line, &args[0])?;
                if ipot < 0 {
                    return Err(parse_error(line, "NUMDENS ipot must be non-negative"));
                }
                let density = parse_f64(line, &args[1])?;
                if !density.is_finite() {
                    return Err(parse_error(line, "NUMDENS value must be finite"));
                }
                let index = usize::try_from(ipot)
                    .map_err(|_| parse_error(line, "NUMDENS ipot is out of range"))?;
                if index >= number_densities.len() {
                    number_densities.resize(index + 1, -1.0);
                }
                number_densities[index] = density;
            }
            Some("PREP") => print_eps = true,
            _ => {}
        }
    }

    Ok(OpconsInput {
        run_opcons,
        print_eps,
        number_densities,
    })
}

fn opcons_density_count(potentials: &[Potential]) -> usize {
    let nph = potentials
        .iter()
        .map(|potential| potential.ipot)
        .max()
        .unwrap_or(1)
        .max(1);
    usize::try_from(nph).map_or(2, |nph| nph + 1)
}

fn parse_sfconv_input(
    input: &FeffInput,
    ispec: i32,
    ipol: i32,
    spin: i32,
    print: Option<[i32; 6]>,
) -> Result<SfconvInput> {
    let mut sfconv = SfconvInput {
        control: SfconvControl {
            msfconv: 0,
            ipse: 0,
            ipsk: 0,
        },
        window: SfconvWindow {
            wsigk: 0.0,
            cen: 0.0,
        },
        spectrum: SfconvSpectrum {
            ispec: output_ispec_for_handoff(ispec, ipol, spin),
            ipr6: print.and_then(|values| values.get(5).copied()).unwrap_or(0),
        },
        cfname: "NULL".to_string(),
    };

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("SFCONV") => sfconv.control.msfconv = 1,
            Some("SELF") => sfconv.control.ipse = 1,
            Some("SFSE") => {
                let args = card_args(line)?;
                let Some(wsigk) = args.first() else {
                    return Err(parse_error(line, "SFSE requires wsigk"));
                };
                sfconv.control.ipsk = 1;
                sfconv.window.wsigk = parse_f64(line, wsigk)?;
                if !sfconv.window.wsigk.is_finite() {
                    return Err(parse_error(line, "SFSE wsigk must be finite"));
                }
            }
            Some("RCONV") => {
                let args = card_args(line)?;
                if args.len() < 2 {
                    return Err(parse_error(line, "RCONV requires cen and cfname"));
                }
                sfconv.window.cen = parse_f64(line, &args[0])?;
                if !sfconv.window.cen.is_finite() {
                    return Err(parse_error(line, "RCONV cen must be finite"));
                }
                sfconv.cfname = fortran_fixed_string(&args[1], 12);
            }
            _ => {}
        }
    }

    Ok(sfconv)
}

fn output_ispec_for_handoff(ispec: i32, ipol: i32, spin: i32) -> i32 {
    if ipol == 2 && spin != 0 { -1 } else { ispec }
}

fn fortran_fixed_string(value: &str, width: usize) -> String {
    let mut end = 0;
    for (index, character) in value.char_indices() {
        let next = index + character.len_utf8();
        if next > width {
            break;
        }
        end = next;
    }
    value[..end].to_string()
}

fn parse_corval_emin(input: &FeffInput) -> Result<f64> {
    let Some(line) = card_by_feff_name(input, "CORVAL") else {
        return Ok(-70.0);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "CORVAL requires emin"));
    };
    let value = parse_f64(line, value)?;
    if !value.is_finite() {
        return Err(parse_error(line, "CORVAL emin must be finite"));
    }
    Ok(value)
}

fn parse_scf_thermal(input: &FeffInput) -> Result<ScfThermal> {
    let Some(line) = card_by_feff_name(input, "SCFTH") else {
        return Ok(default_scf_thermal());
    };
    let args = card_args(line)?;
    let Some(iscfth) = args.first() else {
        return Err(parse_error(line, "SCFTH requires iscfth"));
    };
    let emaxscf = parse_optional_f64(line, args.get(1))?.unwrap_or(5.0);
    let xntol = parse_optional_f64(line, args.get(4))?.unwrap_or(1.0e-4);
    if !emaxscf.is_finite() || !xntol.is_finite() {
        return Err(parse_error(line, "SCFTH values must be finite"));
    }
    Ok(ScfThermal {
        iscfth: parse_i32(line, iscfth)?,
        emaxscf,
        negrid: parse_optional_i32(line, args.get(2))?.unwrap_or(400),
        nmu: parse_optional_i32(line, args.get(3))?.unwrap_or(100),
        xntol,
    })
}

fn default_scf_thermal() -> ScfThermal {
    ScfThermal {
        iscfth: 2,
        xntol: 1.0e-4,
        nmu: 100,
        negrid: 400,
        emaxscf: 5.0,
    }
}

fn parse_scf_ramp(input: &FeffInput) -> Result<ScfRamp> {
    let Some(line) = card_by_feff_name(input, "SCFR") else {
        return Ok(ScfRamp {
            enabled: false,
            rfms_start: 0.0,
            nramp: 1,
        });
    };
    let args = card_args(line)?;
    let rfms_start = parse_optional_f64(line, args.first())?.unwrap_or(0.0);
    if !rfms_start.is_finite() {
        return Err(parse_error(line, "SCFR rfms_start must be finite"));
    }
    Ok(ScfRamp {
        enabled: true,
        rfms_start,
        nramp: parse_optional_i32(line, args.get(1))?.unwrap_or(1),
    })
}

fn parse_scf_tolerances(input: &FeffInput) -> Result<ScfTolerances> {
    let default = ScfTolerances {
        tolmu: 0.001,
        tolq: 0.001,
        tolqp: 0.0002,
    };
    let Some(line) = card_by_feff_name(input, "TOLS") else {
        return Ok(default);
    };
    let args = card_args(line)?;
    let Some(first) = args.first() else {
        return Err(parse_error(line, "TOLS requires a tolerance value"));
    };
    let first = parse_f64(line, first)?;
    if !first.is_finite() {
        return Err(parse_error(line, "TOLS values must be finite"));
    }
    if first < 0.0 {
        return Ok(ScfTolerances {
            tolmu: default.tolmu * first,
            tolq: default.tolq * first,
            tolqp: default.tolqp * first,
        });
    }
    let tolq = parse_optional_f64(line, args.get(1))?.unwrap_or(default.tolq);
    let tolqp = parse_optional_f64(line, args.get(2))?.unwrap_or(default.tolqp);
    if !tolq.is_finite() || !tolqp.is_finite() {
        return Err(parse_error(line, "TOLS values must be finite"));
    }
    Ok(ScfTolerances {
        tolmu: default.tolmu,
        tolq,
        tolqp,
    })
}

fn parse_fine_structure_damping(input: &FeffInput) -> Result<FineStructureDamping> {
    let mut damping = FineStructureDamping {
        alphat: 0.0,
        thetae: 0.0,
        sig2g: 0.0,
        sig_gk: 0.0,
    };

    if let Some(line) = card_by_feff_name(input, "SIG2") {
        let args = card_args(line)?;
        let Some(sig2g) = args.first() else {
            return Err(parse_error(line, "SIG2 requires sig2g"));
        };
        damping.sig2g = parse_f64(line, sig2g)?;
        if !damping.sig2g.is_finite() {
            return Err(parse_error(line, "SIG2 sig2g must be finite"));
        }
    }

    if let Some(line) = card_by_feff_name(input, "SIG3") {
        let args = card_args(line)?;
        let Some(alphat) = args.first() else {
            return Err(parse_error(line, "SIG3 requires alphat"));
        };
        damping.alphat = parse_f64(line, alphat)?;
        damping.thetae = parse_optional_f64(line, args.get(1))?.unwrap_or(0.0);
        if !damping.alphat.is_finite() || !damping.thetae.is_finite() {
            return Err(parse_error(line, "SIG3 values must be finite"));
        }
    }

    if let Some(line) = card_by_feff_name(input, "SIGGK") {
        let args = card_args(line)?;
        if let Some(sig_gk) = args.first() {
            damping.sig_gk = parse_f64(line, sig_gk)?;
            if !damping.sig_gk.is_finite() {
                return Err(parse_error(line, "SIGGK sig_gk must be finite"));
            }
        }
    }

    Ok(damping)
}

fn parse_config_type(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "CONFIGURATION") else {
        return Ok(1);
    };
    let args = card_args(line)?;
    Ok(match args.first().map(|arg| arg.to_ascii_lowercase()) {
        Some(kind) if kind == "file" || kind == "card" => 2,
        Some(kind) if kind == "feff7" => 7,
        _ => 1,
    })
}

fn parse_config_records(input: &FeffInput) -> Result<Vec<String>> {
    let mut records = Vec::new();
    let mut index = 0_usize;
    while let Some(line) = input.lines.get(index) {
        if let LineKind::Card { keyword, args, .. } = &line.kind
            && keyword == "CONFIG"
            && args
                .first()
                .is_some_and(|arg| arg.eq_ignore_ascii_case("card"))
        {
            let Some(count_token) = args.get(1) else {
                return Err(parse_error(
                    line,
                    "CONFIG card requires a payload line count",
                ));
            };
            let count = parse_i32(line, count_token)?;
            if count < 0 {
                return Err(parse_error(
                    line,
                    "CONFIG card line count must be non-negative",
                ));
            }
            let count = usize::try_from(count)
                .map_err(|_| parse_error(line, "CONFIG card line count is out of range"))?;
            for offset in 1..=count {
                let Some(payload) = input.lines.get(index + offset) else {
                    return Err(parse_error(
                        line,
                        "CONFIG card payload is shorter than declared",
                    ));
                };
                match &payload.kind {
                    LineKind::SectionData { section, .. } if section == "CONFIG" => {
                        records.push(payload.raw.clone());
                    }
                    LineKind::SectionData { .. } | LineKind::Card { .. } => {
                        return Err(parse_error(
                            payload,
                            "CONFIG card payload ended before declared line count",
                        ));
                    }
                }
            }
            index += count;
        }
        index += 1;
    }
    Ok(records)
}

fn parse_egrid_records(input: &FeffInput) -> Result<Vec<String>> {
    let mut records = Vec::new();
    let mut index = 0_usize;
    while let Some(line) = input.lines.get(index) {
        if let LineKind::Card { keyword, args, .. } = &line.kind
            && keyword == "EGRID"
            && args.is_empty()
        {
            let mut block = Vec::new();
            let mut offset = 1_usize;
            while let Some(payload) = input.lines.get(index + offset) {
                match &payload.kind {
                    LineKind::SectionData { section, fields } if section == "EGRID" => {
                        block.push(fields.join(" "));
                    }
                    LineKind::SectionData { .. } | LineKind::Card { .. } => break,
                }
                offset += 1;
            }

            let text = block
                .iter()
                .map(|record| format!(" {record} \n"))
                .collect::<String>();
            parse_grid_inp(&text)?;
            records.extend(block);
            index += offset.saturating_sub(1);
        }
        index += 1;
    }
    Ok(records)
}

fn parse_density_records(input: &FeffInput) -> Result<Vec<String>> {
    let records = input
        .section_rows("DENSITY")
        .map(|line| line.raw.clone())
        .collect::<Vec<_>>();
    if records.is_empty() {
        return Ok(records);
    }

    let text = records
        .iter()
        .map(|record| format!("{record}\n"))
        .collect::<String>();
    let density_path = input
        .source
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("density.inp");
    DensityInput::parse_str(density_path, &text)?;
    Ok(records)
}

fn parse_reciprocal_space(input: &FeffInput) -> bool {
    input.cards().fold(false, |reciprocal, line| {
        let LineKind::Card { keyword, .. } = &line.kind else {
            return reciprocal;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("REAL") => false,
            Some("RECIPROCAL") => true,
            _ => reciprocal,
        }
    })
}

fn parse_cif_equivalence(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "EQUIVALENCE") else {
        return Ok(1);
    };
    let args = card_args(line)?;
    let selector = parse_optional_i32(line, args.first())?.unwrap_or(1);
    match selector {
        1 | 2 | 4 => Ok(selector),
        3 => Err(parse_error(
            line,
            "EQUIVALENCE 3 is not implemented by FEFF10",
        )),
        _ => Err(parse_error(line, "EQUIVALENCE must be 1, 2, 3, or 4")),
    }
}

fn cif_equivalence_mode(selector: i32) -> CifEquivalence {
    match selector {
        2 => CifEquivalence::AtomicNumber,
        4 => CifEquivalence::AutomaticLimit,
        _ => CifEquivalence::Crystallographic,
    }
}

fn parse_coordinate_mode(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "COORDINATES") else {
        return Ok(3);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "COORDINATES requires a selector"));
    };
    let mode = parse_i32(line, value)?;
    if (1..=6).contains(&mode) {
        Ok(mode)
    } else {
        Err(parse_error(line, "COORDINATES must be between 1 and 6"))
    }
}

fn parse_reciprocal_input(
    input: &FeffInput,
    nohole: i32,
    atoms: &[Atom],
    reciprocal: bool,
    cif_equivalence: i32,
) -> Result<Option<ReciprocalInput>> {
    if !reciprocal {
        return Ok(None);
    };
    let Some(reciprocal_line) = card_by_feff_name(input, "RECIPROCAL") else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "RECIPROCAL mode requires a RECIPROCAL card".to_string(),
        });
    };
    let k_mesh = parse_k_mesh(input)?;
    let absorber = parse_required_i32_card(input, "TARGET")?;
    let stretch = parse_strfac(input)?;

    let Some(lattice) = parse_lattice_block(input)? else {
        if let Some(cif_line) = card_by_feff_name(input, "CIF") {
            let cif_path = parse_cif_path(input, cif_line)?;
            let cif = read_cif(&cif_path)?;
            if absorber <= 0 {
                return Err(parse_error(
                    cif_line,
                    "TARGET must be positive for CIF input",
                ));
            }
            let target = usize::try_from(absorber)
                .map_err(|_| parse_error(cif_line, "TARGET is out of range for CIF input"))?;
            let structure = expand_cif_structure_with_equivalence(
                &cif,
                target,
                cif_equivalence_mode(cif_equivalence),
            )?;
            return Ok(Some(ReciprocalInput {
                ispace: 0,
                cell: Some(ReciprocalCell {
                    lattice_vectors: structure.lattice_vectors,
                    volume_scale: -1.0,
                    imaginary_energy: 0.0,
                    core_hole_strength: 1.0,
                    lattice_name: structure.lattice_name,
                    space_group_hm: structure.space_group_hm,
                    space_group: structure.space_group,
                    atom_count: structure.positions.len(),
                    absorber: i32::try_from(structure.absorber).map_err(|_| {
                        parse_error(cif_line, "expanded CIF absorber index is out of range")
                    })?,
                    core_hole: i32::from(nohole != 0),
                    k_mesh,
                    positions: structure.positions,
                    potentials: structure.potentials,
                    labels: structure.labels,
                    stretch,
                }),
            }));
        }
        return Err(parse_error(
            reciprocal_line,
            "RECIPROCAL requires LATTICE or CIF",
        ));
    };
    if atoms.is_empty() {
        return Err(parse_error(
            reciprocal_line,
            "RECIPROCAL with LATTICE requires ATOMS rows",
        ));
    }

    let space_group = parse_sgroup(input)?;
    let coordinate_mode = parse_coordinate_mode(input)?;
    let atoms = convert_lattice_atoms(input, &lattice, atoms, coordinate_mode)?;
    let positions = atoms.iter().map(|atom| [atom.x, atom.y, atom.z]).collect();
    let potentials = atoms.iter().map(|atom| atom.ipot).collect();

    Ok(Some(ReciprocalInput {
        ispace: 0,
        cell: Some(ReciprocalCell {
            lattice_vectors: lattice.vectors,
            volume_scale: -1.0,
            imaginary_energy: 0.0,
            core_hole_strength: 1.0,
            lattice_name: lattice.name,
            space_group_hm: "\0".repeat(8),
            space_group,
            atom_count: atoms.len(),
            absorber,
            core_hole: i32::from(nohole != 0),
            k_mesh,
            positions,
            potentials,
            labels: Vec::new(),
            stretch,
        }),
    }))
}

fn parse_band_input(input: &FeffInput) -> Result<BandInput> {
    let Some(line) = card_by_feff_name(input, "BAND") else {
        return Ok(default_band_input());
    };
    let args = card_args(line)?;
    if args.len() < 4 {
        return Err(parse_error(
            line,
            "BANDSTRUCTURE requires emin emax estep ikpath",
        ));
    }
    Ok(BandInput {
        mband: 1,
        energy_mesh: BandEnergyMesh {
            emin: parse_f64(line, &args[0])?,
            emax: parse_f64(line, &args[1])?,
            estep: parse_f64(line, &args[2])?,
        },
        nkp: parse_optional_i32(line, args.get(4))?.unwrap_or(0),
        ikpath: parse_i32(line, &args[3])?,
        freeprop: match args.get(5) {
            Some(value) => parse_logical(line, value)?,
            None => false,
        },
    })
}

fn default_band_input() -> BandInput {
    BandInput {
        mband: 0,
        energy_mesh: BandEnergyMesh {
            emin: 0.0,
            emax: 0.0,
            estep: 0.0,
        },
        nkp: 0,
        ikpath: -1,
        freeprop: false,
    }
}

fn parse_full_spectrum_input(active_cards: &[String]) -> FullSpectrumInput {
    FullSpectrumInput {
        m_full_spectrum: i32::from(active_cards.iter().any(|card| card == "FULLSPECTRUM")),
    }
}

fn parse_screen_input(input: &FeffInput) -> Result<ScreenInput> {
    let mut screen = ScreenInput::default();

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        if feff_card_token(keyword).map(|(_, display)| display) != Some("SCREEN") {
            continue;
        }

        let args = card_args(line)?;
        if args.len() < 2 {
            return Err(parse_error(line, "SCREEN requires keyword and value"));
        }

        let key = screen_key_prefix(&args[0]);
        let value = parse_f64(line, &args[1])?;
        if !value.is_finite() {
            return Err(parse_error(line, "SCREEN value must be finite"));
        }

        match key.as_str() {
            "ner" => screen.ner = parse_screen_i32(line, "ner", value)?,
            "nei" => screen.nei = parse_screen_i32(line, "nei", value)?,
            "max" => screen.maxl = parse_screen_i32(line, "maxl", value)?,
            "irr" => screen.irrh = parse_screen_i32(line, "irrh", value)?,
            "ien" => screen.iend = parse_screen_i32(line, "iend", value)?,
            "lfx" => screen.lfxc = parse_screen_i32(line, "lfxc", value)?,
            "emi" => screen.emin = value,
            "ema" => screen.emax = value,
            "eim" => screen.eimax = value,
            "erm" => screen.ermin = value,
            "rfm" => screen.rfms = value,
            "nrp" => screen.nrptx0 = parse_screen_i32(line, "nrptx0", value)?,
            "ico" => screen.icore = parse_screen_i32(line, "icore", value)?,
            _ => {
                return Err(parse_error(
                    line,
                    format!("unrecognized SCREEN keyword {:?}", args[0]),
                ));
            }
        }
    }

    Ok(screen)
}

fn screen_key_prefix(key: &str) -> String {
    key.chars().take(3).flat_map(char::to_lowercase).collect()
}

fn parse_screen_i32(line: &FeffLine, field: &str, value: f64) -> Result<i32> {
    let rounded = value.round();
    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        return Err(parse_error(
            line,
            format!("SCREEN {field} value is out of range"),
        ));
    }
    Ok(rounded as i32)
}

fn parse_cif_path(input: &FeffInput, line: &FeffLine) -> Result<PathBuf> {
    let args = card_args(line)?;
    let Some(path) = args.first() else {
        return Err(parse_error(line, "CIF requires a file path"));
    };
    let path = strip_card_delimiters(path);
    let path = PathBuf::from(path);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(input
            .source
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(path))
    }
}

fn strip_card_delimiters(value: &str) -> &str {
    let pairs = [
        ('"', '"'),
        ('\'', '\''),
        ('{', '}'),
        ('(', ')'),
        ('<', '>'),
        ('[', ']'),
    ];
    pairs
        .iter()
        .find_map(|(open, close)| {
            (value.starts_with(*open) && value.ends_with(*close) && value.len() >= 2)
                .then_some(&value[1..value.len() - 1])
        })
        .unwrap_or(value)
}

fn parse_cif_cluster(
    input: &FeffInput,
    radius: f64,
    needed: bool,
    cif_equivalence: i32,
) -> Result<Option<CifCluster>> {
    if !needed {
        return Ok(None);
    }
    let Some(cif_line) = card_by_feff_name(input, "CIF") else {
        return Ok(None);
    };
    let cif_path = parse_cif_path(input, cif_line)?;
    let cif = read_cif(&cif_path)?;
    let target = parse_cif_target(input, cif_line)?;
    expand_cif_cluster_with_equivalence(&cif, target, radius, cif_equivalence_mode(cif_equivalence))
        .map(Some)
}

fn cif_cluster_radius(scf: Option<&Scf>, fms: Option<&Fms>, rpath: Option<f64>) -> f64 {
    [scf.map(|scf| scf.radius), fms.map(|fms| fms.radius), rpath]
        .into_iter()
        .flatten()
        .fold(0.0, f64::max)
}

fn cif_cluster_potentials(cluster: &CifCluster) -> Result<Vec<Potential>> {
    cluster
        .potentials
        .iter()
        .map(|potential| {
            let xnatph = if potential.absorber {
                Some(0.01)
            } else {
                Some(potential.multiplicity as f64)
            };
            Ok(Potential {
                ipot: potential.ipot,
                z: Some(potential.atomic_number),
                z_token: potential.atomic_number.to_string(),
                tag: Some(potential.label.clone()),
                lmax1: None,
                lmax2: None,
                xnatph,
                spinph: None,
            })
        })
        .collect()
}

fn cif_cluster_atoms(cluster: &CifCluster) -> Vec<Atom> {
    cluster
        .atoms
        .iter()
        .map(|atom| Atom {
            x: atom.x,
            y: atom.y,
            z: atom.z,
            ipot: atom.potential,
            tag: None,
            distance: None,
            index: None,
        })
        .collect()
}

fn parse_cif_target(input: &FeffInput, cif_line: &FeffLine) -> Result<usize> {
    let Some(target_line) = card_by_feff_name(input, "TARGET") else {
        return Ok(1);
    };
    let args = card_args(target_line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(target_line, "TARGET requires a value"));
    };
    let target = parse_i32(target_line, value)?;
    if target <= 0 {
        return Err(parse_error(
            cif_line,
            "TARGET must be positive for CIF input",
        ));
    }
    usize::try_from(target)
        .map_err(|_| parse_error(cif_line, "TARGET is out of range for CIF input"))
}

struct LatticeBlock {
    name: String,
    vectors: [[f64; 3]; 3],
}

#[derive(Debug, Clone, Copy)]
struct PeriodicAtom {
    x: f64,
    y: f64,
    z: f64,
    ipot: i32,
    distance: f64,
}

fn parse_lattice_cluster_atoms(
    input: &FeffInput,
    atoms: &[Atom],
    radius: f64,
    reciprocal: bool,
) -> Result<Option<Vec<Atom>>> {
    if !reciprocal || card_by_feff_name(input, "CIF").is_some() {
        return Ok(None);
    }
    let Some(lattice) = parse_lattice_block(input)? else {
        return Ok(None);
    };
    if atoms.is_empty() {
        return Ok(None);
    }
    let target = parse_required_i32_card(input, "TARGET")?;
    if target <= 0 {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "TARGET must be positive for LATTICE input".to_string(),
        });
    }
    let target = usize::try_from(target - 1).map_err(|_| IoError::Parse {
        path: input.source.clone(),
        line: 0,
        message: "TARGET is out of range for LATTICE input".to_string(),
    })?;
    if target >= atoms.len() {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: format!(
                "TARGET {} is outside the ATOMS row range 1..={}",
                target + 1,
                atoms.len()
            ),
        });
    }
    let coordinate_mode = parse_coordinate_mode(input)?;
    let atoms = convert_lattice_atoms(input, &lattice, atoms, coordinate_mode)?;
    Ok(Some(expand_lattice_cluster(
        &lattice, &atoms, target, radius,
    )))
}

fn convert_lattice_atoms(
    input: &FeffInput,
    lattice: &LatticeBlock,
    atoms: &[Atom],
    coordinate_mode: i32,
) -> Result<Vec<Atom>> {
    let lengths = lattice_vector_lengths(input, lattice)?;
    atoms
        .iter()
        .map(|atom| convert_lattice_atom(input, lattice, lengths, atom, coordinate_mode))
        .collect()
}

fn convert_lattice_atom(
    input: &FeffInput,
    lattice: &LatticeBlock,
    lengths: [f64; 3],
    atom: &Atom,
    coordinate_mode: i32,
) -> Result<Atom> {
    let [a1_len, a2_len, a3_len] = lengths;
    let position = match coordinate_mode {
        1 => [atom.x / a1_len, atom.y / a1_len, atom.z / a1_len],
        2 => [atom.x, atom.y * a2_len / a1_len, atom.z * a3_len / a1_len],
        3 => [atom.x, atom.y, atom.z],
        4 => scale_vector(
            fractional_to_cartesian([atom.x, atom.y, atom.z], lattice.vectors),
            1.0 / a1_len,
        ),
        5 => {
            let fractional = [atom.x, atom.y * a1_len / a2_len, atom.z * a1_len / a3_len];
            scale_vector(
                fractional_to_cartesian(fractional, lattice.vectors),
                1.0 / a1_len,
            )
        }
        6 => {
            let fractional = [atom.x / a1_len, atom.y / a2_len, atom.z / a3_len];
            scale_vector(
                fractional_to_cartesian(fractional, lattice.vectors),
                1.0 / a1_len,
            )
        }
        _ => {
            return Err(IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "COORDINATES must be between 1 and 6".to_string(),
            });
        }
    };
    Ok(Atom {
        x: position[0],
        y: position[1],
        z: position[2],
        ipot: atom.ipot,
        tag: atom.tag.clone(),
        distance: atom.distance,
        index: atom.index,
    })
}

fn lattice_vector_lengths(input: &FeffInput, lattice: &LatticeBlock) -> Result<[f64; 3]> {
    let lengths = lattice.vectors.map(lattice_vector_length);
    if lengths
        .iter()
        .all(|length| length.is_finite() && *length > 0.0)
    {
        Ok(lengths)
    } else {
        Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "LATTICE vectors must have positive finite lengths".to_string(),
        })
    }
}

fn expand_lattice_cluster(
    lattice: &LatticeBlock,
    atoms: &[Atom],
    target: usize,
    radius: f64,
) -> Vec<Atom> {
    let [a1, a2, a3] = lattice.vectors;
    let ratomslist = 8.0_f64.max(1.33 * radius.max(0.0));
    let i1 = lattice_repeat_count(ratomslist, a1);
    let i2 = lattice_repeat_count(ratomslist, a2);
    let i3 = lattice_repeat_count(ratomslist, a3);
    let shifts = lattice_centering_shifts(&lattice.name);
    let lattice_scale = lattice_vector_length(a1);
    let absorber = lattice_atom_position(&atoms[target], lattice_scale);

    let mut expanded = Vec::new();
    let mut absorber_index = 0_usize;
    for j1 in -i1..=i1 {
        for j2 in -i2..=i2 {
            for j3 in -i3..=i3 {
                let translation = lattice_translation(j1, j2, j3, a1, a2, a3);
                for (index, atom) in atoms.iter().enumerate() {
                    let position =
                        add_vectors(lattice_atom_position(atom, lattice_scale), translation);
                    let mut ipot = atom.ipot;
                    if j1 == 0 && j2 == 0 && j3 == 0 && index == target {
                        ipot = 0;
                        absorber_index = expanded.len();
                    }
                    expanded.push(periodic_atom(position, ipot, absorber));

                    for shift in &shifts {
                        let shifted =
                            add_vectors(position, fractional_to_cartesian(*shift, [a1, a2, a3]));
                        expanded.push(periodic_atom(shifted, atom.ipot, absorber));
                    }
                }
            }
        }
    }

    feff_sort_periodic_atoms(&mut expanded, absorber_index);
    let cutoff = (lattice_vector_length(a1) * f64::from(i1))
        .min(lattice_vector_length(a2) * f64::from(i1))
        .min(lattice_vector_length(a3) * f64::from(i1));
    let keep = expanded
        .iter()
        .position(|atom| atom.distance > cutoff)
        .unwrap_or(expanded.len());
    expanded.truncate(keep);

    expanded
        .into_iter()
        .map(|atom| Atom {
            x: atom.x,
            y: atom.y,
            z: atom.z,
            ipot: atom.ipot,
            tag: None,
            distance: None,
            index: None,
        })
        .collect()
}

fn periodic_atom(position: [f64; 3], ipot: i32, absorber: [f64; 3]) -> PeriodicAtom {
    PeriodicAtom {
        x: position[0],
        y: position[1],
        z: position[2],
        ipot,
        distance: lattice_distance(position, absorber),
    }
}

fn lattice_atom_position(atom: &Atom, scale: f64) -> [f64; 3] {
    [atom.x * scale, atom.y * scale, atom.z * scale]
}

fn feff_sort_periodic_atoms(atoms: &mut [PeriodicAtom], mut absorber_index: usize) {
    for i in 0..atoms.len() {
        let mut min_index = i;
        let mut min_distance = atoms[i].distance;
        for (j, atom) in atoms.iter().enumerate().skip(i) {
            if atom.distance < min_distance {
                min_index = j;
                min_distance = atom.distance;
            }
        }
        atoms.swap(i, min_index);
        if i == absorber_index {
            absorber_index = min_index;
        }
        if min_index == absorber_index {
            absorber_index = i;
        }
    }
}

fn lattice_repeat_count(radius: f64, vector: [f64; 3]) -> i32 {
    (radius / lattice_vector_length(vector)).trunc() as i32 + 1
}

fn lattice_centering_shifts(lattice_name: &str) -> Vec<[f64; 3]> {
    match lattice_name {
        "F" => vec![[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        "CXY" => vec![[0.5, 0.5, 0.0]],
        "CXZ" => vec![[0.5, 0.0, 0.5]],
        "CYZ" => vec![[0.0, 0.5, 0.5]],
        "B" | "I" => vec![[0.5, 0.5, 0.5]],
        _ => Vec::new(),
    }
}

fn fractional_to_cartesian(position: [f64; 3], lattice_vectors: [[f64; 3]; 3]) -> [f64; 3] {
    [
        position[0].mul_add(
            lattice_vectors[0][0],
            position[1].mul_add(lattice_vectors[1][0], position[2] * lattice_vectors[2][0]),
        ),
        position[0].mul_add(
            lattice_vectors[0][1],
            position[1].mul_add(lattice_vectors[1][1], position[2] * lattice_vectors[2][1]),
        ),
        position[0].mul_add(
            lattice_vectors[0][2],
            position[1].mul_add(lattice_vectors[1][2], position[2] * lattice_vectors[2][2]),
        ),
    ]
}

fn lattice_translation(
    j1: i32,
    j2: i32,
    j3: i32,
    a1: [f64; 3],
    a2: [f64; 3],
    a3: [f64; 3],
) -> [f64; 3] {
    add_vectors(
        add_vectors(
            scale_vector(a1, f64::from(j1)),
            scale_vector(a2, f64::from(j2)),
        ),
        scale_vector(a3, f64::from(j3)),
    )
}

fn lattice_distance(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    lattice_vector_length([lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]])
}

fn lattice_vector_length(vector: [f64; 3]) -> f64 {
    vector[0]
        .mul_add(
            vector[0],
            vector[1].mul_add(vector[1], vector[2] * vector[2]),
        )
        .sqrt()
}

fn add_vectors(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn scale_vector(vector: [f64; 3], scale: f64) -> [f64; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn parse_lattice_block(input: &FeffInput) -> Result<Option<LatticeBlock>> {
    let Some(line) = card_by_feff_name(input, "LATTICE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(name) = args.first() else {
        return Err(parse_error(line, "LATTICE requires a lattice type"));
    };
    let scale = parse_optional_f64(line, args.get(1))?.unwrap_or(1.0);
    let rows = input.section_rows("LATTICE").collect::<Vec<_>>();
    if rows.len() < 3 {
        return Err(parse_error(line, "LATTICE requires three vector rows"));
    }

    let mut vectors = [[0.0; 3]; 3];
    for (idx, row) in rows.iter().take(3).enumerate() {
        let fields = section_fields(row)?;
        if fields.len() < 3 {
            return Err(parse_error(row, "LATTICE vector rows require x y z"));
        }
        vectors[idx] = [
            parse_f64(row, &fields[0])? * scale,
            parse_f64(row, &fields[1])? * scale,
            parse_f64(row, &fields[2])? * scale,
        ];
    }

    Ok(Some(LatticeBlock {
        name: name.clone(),
        vectors,
    }))
}

fn parse_k_mesh(input: &FeffInput) -> Result<ReciprocalKMesh> {
    let Some(line) = card_by_feff_name(input, "KMESH") else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: "RECIPROCAL requires KMESH".to_string(),
        });
    };
    let args = card_args(line)?;
    let Some(x) = args.first() else {
        return Err(parse_error(line, "KMESH requires at least one value"));
    };
    let x = parse_i32(line, x)?;
    let y = parse_optional_i32(line, args.get(1))?.unwrap_or(0);
    let z = parse_optional_i32(line, args.get(2))?.unwrap_or(0);
    let product = x * y * z;
    Ok(ReciprocalKMesh {
        total: if product == 0 { x } else { product },
        x,
        y,
        z,
        kind: parse_optional_i32(line, args.get(3))?.unwrap_or(1),
        use_symmetry: parse_optional_i32(line, args.get(4))?.unwrap_or(0) != 0,
    })
}

fn parse_required_i32_card(input: &FeffInput, keyword: &str) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, keyword) else {
        return Err(IoError::Parse {
            path: input.source.clone(),
            line: 0,
            message: format!("RECIPROCAL requires {keyword}"),
        });
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, format!("{keyword} requires a value")));
    };
    parse_i32(line, value)
}

fn parse_strfac(input: &FeffInput) -> Result<[f64; 3]> {
    let Some(line) = card_by_feff_name(input, "STRFAC") else {
        return Ok([0.0; 3]);
    };
    let args = card_args(line)?;
    if args.len() < 3 {
        return Err(parse_error(line, "STRFAC requires three values"));
    }
    Ok([
        parse_f64(line, &args[0])?,
        parse_f64(line, &args[1])?,
        parse_f64(line, &args[2])?,
    ])
}

fn parse_sgroup(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "SGROUP") else {
        return Ok(1);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Ok(1);
    };
    parse_i32(line, value)
}

fn parse_i32_6(input: &FeffInput, keyword: &str) -> Result<Option<[i32; 6]>> {
    let Some(line) = card_by_feff_name(input, keyword) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    if args.len() == 4 {
        let shared = parse_i32(line, &args[0])?;
        return Ok(Some([
            shared,
            shared,
            shared,
            parse_i32(line, &args[1])?,
            parse_i32(line, &args[2])?,
            parse_i32(line, &args[3])?,
        ]));
    }
    if args.len() < 6 {
        return Err(parse_error(
            line,
            format!("{keyword} requires either 4 FEFF7 or 6 FEFF8 integer values"),
        ));
    }
    let mut values = [0_i32; 6];
    for (idx, slot) in values.iter_mut().enumerate() {
        *slot = parse_i32(line, &args[idx])?;
    }
    Ok(Some(values))
}

fn parse_scf(input: &FeffInput) -> Result<Option<Scf>> {
    let Some(line) = card_by_feff_name(input, "SCF") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(radius) = args.first() else {
        return Err(parse_error(line, "SCF requires a radius"));
    };
    let mut iterations = parse_optional_i32(line, args.get(2))?.unwrap_or(100);
    if iterations <= 0 || iterations > 100 {
        iterations = 100;
    }
    let mut ca = parse_optional_f64(line, args.get(3))?.unwrap_or(0.2);
    if ca < 0.0 {
        ca = 0.0;
    }
    let mut nmix = parse_optional_i32(line, args.get(4))?.unwrap_or(1);
    if nmix <= 0 {
        nmix = 1;
    } else if nmix > 30 {
        nmix = 30;
    }
    let mut ecv = parse_optional_f64(line, args.get(5))?.unwrap_or(-40.0);
    if ecv >= 0.0 {
        ecv = -40.0;
    }

    Ok(Some(Scf {
        radius: parse_f64(line, radius)?,
        lfms: parse_optional_i32(line, args.get(1))?.unwrap_or(0).min(1),
        iterations,
        ca,
        nmix,
        ecv,
        icoul: parse_optional_i32(line, args.get(6))?.unwrap_or(0),
    }))
}

fn parse_exchange(input: &FeffInput) -> Result<Option<Exchange>> {
    let Some(line) = card_by_feff_name(input, "EXCHANGE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    if args.len() < 3 {
        return Err(parse_error(line, "EXCHANGE requires ixc, vr0, and vi0"));
    }

    Ok(Some(Exchange {
        ixc: parse_i32(line, &args[0])?,
        vr0: parse_f64(line, &args[1])?,
        vi0: parse_f64(line, &args[2])?,
        ixc0: parse_optional_i32(line, args.get(3))?,
    }))
}

fn parse_exafs(input: &FeffInput) -> Result<Option<Exafs>> {
    let Some(line) = card_by_feff_name(input, "EXAFS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(xkmax) = args.first() else {
        return Err(parse_error(line, "EXAFS requires an xkmax value"));
    };

    Ok(Some(Exafs {
        xkmax: parse_f64(line, xkmax)?,
    }))
}

fn parse_spectrum_grid(
    input: &FeffInput,
    exchange: Option<&Exchange>,
    ispec: i32,
) -> Result<SpectrumGrid> {
    let mut grid = SpectrumGrid {
        ixc0: exchange
            .and_then(|exchange| exchange.ixc0)
            .filter(|ixc0| *ixc0 >= 0)
            .unwrap_or_else(|| if (1..=4).contains(&ispec) { 2 } else { 0 }),
        ..SpectrumGrid::default()
    };

    if let Some((name, line)) = ["XANES", "DANES", "ELNES"]
        .into_iter()
        .find_map(|name| card_by_feff_name(input, name).map(|line| (name, line)))
    {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            grid.xkstep = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(2) {
            grid.vixan = parse_f64(line, value)?;
        }
        if grid.xkstep < 0.01 {
            grid.xkstep = 0.01;
        }
        if matches!(name, "XANES" | "ELNES") {
            if grid.xkstep > 2.0 {
                grid.xkstep = 0.5;
            }
            if grid.xkmax.abs() < 2.0 {
                grid.xkmax = 2.0;
            }
            if grid.xkmax.abs() > 200.0 {
                grid.xkmax = 200.0;
            }
        } else if grid.xkmax < 2.0 {
            grid.xkmax = 2.0;
        }
    } else if let Some(line) = card_by_feff_name(input, "XES") {
        let args = card_args(line)?;
        grid.xkstep = 0.01;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            grid.xkstep = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(2) {
            grid.vixan = parse_f64(line, value)?;
        }
        if grid.xkstep <= grid.xkmax {
            grid.xkstep = 0.01;
        }
        if grid.xkmax >= 0.0 {
            grid.xkmax = -40.0;
        }
    } else if let Some(line) = card_by_feff_name(input, "FPRIME") {
        let args = card_args(line)?;
        if args.len() < 2 {
            return Err(parse_error(line, "FPRIME requires emin and emax"));
        }
        grid.xkmax = parse_f64(line, &args[0])?;
        grid.xkstep = parse_f64(line, &args[1])?;
        if let Some(value) = args.get(2) {
            grid.vixan = parse_f64(line, value)?;
        }
        if grid.xkstep < grid.xkmax {
            grid.xkstep = grid.xkmax;
        }
    } else if let Some(line) =
        card_by_feff_name(input, "EXAFS").or_else(|| card_by_feff_name(input, "EXELFS"))
    {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
    }

    Ok(grid)
}

fn parse_temp(input: &FeffInput) -> Result<(f64, i32)> {
    let mut temperature = 0.0;
    let mut iscfxc = 11;

    for line in input.cards() {
        let LineKind::Card { keyword, .. } = &line.kind else {
            continue;
        };
        match feff_card_token(keyword).map(|(_, display)| display) {
            Some("TEMP") => {
                let args = card_args(line)?;
                if let Some(value) = args.first() {
                    temperature = parse_f64(line, value)?;
                    if !temperature.is_finite() {
                        return Err(parse_error(line, "TEMP value must be finite"));
                    }
                }
                if let Some(value) = args.get(1) {
                    iscfxc = parse_i32(line, value)?;
                }
            }
            Some("SCXC") => {
                let args = card_args(line)?;
                let Some(value) = args.first() else {
                    return Err(parse_error(line, "SCXC requires iscfxc"));
                };
                iscfxc = parse_i32(line, value)?;
                validate_scxc(line, iscfxc)?;
            }
            _ => {}
        }
    }

    Ok((temperature, iscfxc))
}

fn validate_scxc(line: &FeffLine, iscfxc: i32) -> Result<()> {
    if matches!(iscfxc, 11 | 12 | 21 | 22) {
        Ok(())
    } else {
        Err(parse_error(
            line,
            "SCXC iscfxc must be one of 11, 12, 21, or 22",
        ))
    }
}

fn parse_criteria(input: &FeffInput) -> Result<(f64, f64)> {
    let Some(line) = card_by_feff_name(input, "CRITERIA") else {
        return Ok((4.0, 2.5));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_f64(line, args.first())?.unwrap_or(4.0),
        parse_optional_f64(line, args.get(1))?.unwrap_or(2.5),
    ))
}

fn parse_pcriteria(input: &FeffInput) -> Result<(f64, f64)> {
    let Some(line) = card_by_feff_name(input, "PCRITERIA") else {
        return Ok((0.0, 0.0));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_f64(line, args.first())?.unwrap_or(0.0),
        parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
    ))
}

fn parse_lreal(input: &FeffInput) -> i32 {
    if card_by_feff_name(input, "RPHASES").is_some() {
        2
    } else {
        i32::from(card_by_feff_name(input, "RSIGMA").is_some())
    }
}

fn parse_iorder(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "IORD") else {
        return Ok(2);
    };
    let args = card_args(line)?;
    let Some(iorder) = args.first() else {
        return Err(parse_error(line, "IORDER requires iorder"));
    };
    parse_i32(line, iorder)
}

fn parse_mpse(input: &FeffInput) -> Result<(i32, i32)> {
    let Some(line) = card_by_feff_name(input, "MPSE") else {
        return Ok((0, 100));
    };
    let args = card_args(line)?;
    let mut i_plsmn = parse_optional_i32(line, args.first())?.unwrap_or(1);
    if i_plsmn == 4 {
        i_plsmn = 1;
    }
    let n_poles = parse_optional_i32(line, args.get(1))?.unwrap_or(100);
    Ok((i_plsmn, n_poles))
}

fn parse_ispec(input: &FeffInput) -> i32 {
    if card_by_feff_name(input, "COMPTON").is_some() || card_by_feff_name(input, "DENS").is_some() {
        5
    } else if card_by_feff_name(input, "FPRIME").is_some() {
        4
    } else if card_by_feff_name(input, "DANES").is_some() {
        3
    } else if card_by_feff_name(input, "XES").is_some() {
        2
    } else if card_by_feff_name(input, "XANES").is_some()
        || card_by_feff_name(input, "ELNES").is_some()
        || card_by_feff_name(input, "NRIXS").is_some()
    {
        1
    } else {
        0
    }
}

fn parse_ipol(input: &FeffInput) -> i32 {
    if card_by_feff_name(input, "XMCD").is_some() {
        2
    } else if card_by_feff_name(input, "POLARIZATION").is_some() {
        1
    } else {
        0
    }
}

fn parse_multipole(input: &FeffInput) -> Result<(i32, i32)> {
    let Some(line) = card_by_feff_name(input, "MULT") else {
        return Ok((0, 0));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_i32(line, args.first())?.unwrap_or(0),
        parse_optional_i32(line, args.get(1))?.unwrap_or(0),
    ))
}

fn parse_polarization_vector(input: &FeffInput) -> Result<[f64; 3]> {
    let Some(line) = card_by_feff_name(input, "POLARIZATION") else {
        return Ok([0.0; 3]);
    };
    let args = card_args(line)?;
    if args.len() < 3 {
        return Err(parse_error(line, "POLARIZATION requires x, y, and z"));
    }
    Ok([
        parse_f64(line, &args[0])?,
        parse_f64(line, &args[1])?,
        parse_f64(line, &args[2])?,
    ])
}

fn parse_ellipticity(input: &FeffInput) -> Result<(f64, [f64; 3])> {
    let Some(line) = card_by_feff_name(input, "ELLIPTICITY") else {
        return Ok((0.0, [0.0; 3]));
    };
    let args = card_args(line)?;
    if args.len() < 4 {
        return Err(parse_error(
            line,
            "ELLIPTICITY requires ellipticity and incident direction",
        ));
    }
    Ok((
        parse_f64(line, &args[0])?,
        [
            parse_f64(line, &args[1])?,
            parse_f64(line, &args[2])?,
            parse_f64(line, &args[3])?,
        ],
    ))
}

fn parse_spin(input: &FeffInput) -> Result<(i32, [f64; 3])> {
    let Some(line) = card_by_feff_name(input, "SPIN") else {
        return Ok((0, [0.0; 3]));
    };
    let args = card_args(line)?;
    let Some(spin) = args.first() else {
        return Err(parse_error(line, "SPIN requires a selector"));
    };
    let spin = parse_i32(line, spin)?;
    let default_vector = if spin == 0 { [0.0; 3] } else { [0.0, 0.0, 1.0] };
    Ok((
        spin,
        [
            parse_optional_f64(line, args.get(1))?.unwrap_or(default_vector[0]),
            parse_optional_f64(line, args.get(2))?.unwrap_or(default_vector[1]),
            parse_optional_f64(line, args.get(3))?.unwrap_or(default_vector[2]),
        ],
    ))
}

fn parse_nohole(input: &FeffInput) -> Result<i32> {
    if let (Some(corehole), Some(_)) = (
        card_by_feff_name(input, "COREHOLE"),
        card_by_feff_name(input, "NOHOLE"),
    ) {
        return Err(parse_error(
            corehole,
            "NOHOLE and COREHOLE cards are mutually exclusive",
        ));
    }

    if let Some(line) = card_by_feff_name(input, "COREHOLE") {
        let args = card_args(line)?;
        let Some(mode) = args.first() else {
            return Ok(-1);
        };
        return match mode.to_ascii_uppercase().as_str() {
            "NONE" => Ok(0),
            "RPA" => Ok(2),
            "FSR" | "REGULAR" => Ok(-1),
            _ => Err(parse_error(
                line,
                "COREHOLE must be NONE, RPA, FSR, or REGULAR",
            )),
        };
    }

    if let Some(line) = card_by_feff_name(input, "NOHOLE") {
        let args = card_args(line)?;
        return parse_optional_i32(line, args.first()).map(|value| value.unwrap_or(0));
    }

    Ok(-1)
}

fn parse_fms(input: &FeffInput) -> Result<Option<Fms>> {
    let Some(line) = card_by_feff_name(input, "FMS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(radius) = args.first() else {
        return Err(parse_error(line, "FMS requires a radius"));
    };

    let radius = parse_f64(line, radius)?;
    let lfms = parse_optional_i32(line, args.get(1))?.unwrap_or(0).min(1);
    let rdirec = parse_optional_f64(line, args.get(5))?.unwrap_or(2.0 * radius);

    Ok(Some(Fms {
        radius,
        lfms,
        minv: parse_optional_i32(line, args.get(2))?.unwrap_or(0),
        toler1: parse_optional_f64(line, args.get(3))?.unwrap_or(0.001),
        toler2: parse_optional_f64(line, args.get(4))?.unwrap_or(0.001),
        rdirec: if rdirec < 0.0 || rdirec > 2.0 * radius {
            2.0 * radius
        } else {
            rdirec
        },
    }))
}

fn parse_crpa(input: &FeffInput) -> Result<Crpa> {
    let Some(line) = card_by_feff_name(input, "CRPA") else {
        return Ok(Crpa::default());
    };
    let args = card_args(line)?;
    if args.len() < 2 {
        return Err(parse_error(line, "CRPA requires l and rcut values"));
    }
    Ok(Crpa {
        enabled: true,
        l: parse_i32(line, &args[0])?,
        rcut: parse_f64(line, &args[1])?,
    })
}

fn parse_compton(input: &FeffInput) -> Result<Compton> {
    let mut compton = Compton::default();

    if let Some(line) = card_by_feff_name(input, "COMPTON") {
        let args = card_args(line)?;
        compton.do_compton = true;
        if let Some(value) = args.first() {
            compton.pqmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            compton.npq = parse_i32(line, value)?;
        }
        if parse_optional_i32(line, args.get(2))?.unwrap_or(0) > 0 {
            compton.force_jzzp = true;
        }
    }

    compton.do_rhozzp = card_by_feff_name(input, "RHOZZP").is_some();

    if let Some(line) = card_by_feff_name(input, "CGRID") {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            compton.zpmax = parse_f64(line, value)?;
        }
        if let Some(value) = args.get(1) {
            compton.ns = parse_i32(line, value)?;
        }
        if let Some(value) = args.get(2) {
            compton.nphi = parse_i32(line, value)?;
        }
        if let Some(value) = args.get(3) {
            compton.nz = parse_i32(line, value)?;
        }
        if let Some(value) = args.get(4) {
            compton.nzp = parse_i32(line, value)?;
        }
    }

    Ok(compton)
}

fn parse_hubbard(input: &FeffInput) -> Result<Hubbard> {
    let Some(line) = card_by_feff_name(input, "HUBBARD") else {
        return Ok(Hubbard::default());
    };
    let args = card_args(line)?;
    if args.len() < 4 {
        return Err(parse_error(
            line,
            "HUBBARD requires U, J, fermi_shift, and l values",
        ));
    }
    Ok(Hubbard {
        i_hubbard: 2,
        mldos_hubb: 2,
        u: parse_f64(line, &args[0])?,
        j: parse_f64(line, &args[1])?,
        fermi_shift: parse_f64(line, &args[2])?,
        l: parse_i32(line, &args[3])?,
    })
}

fn parse_eels(input: &FeffInput) -> Result<Eels> {
    let magic_energy = parse_magic_energy(input)?;
    let section = if card_by_feff_name(input, "ELNES").is_some() {
        "ELNES"
    } else if card_by_feff_name(input, "EXELFS").is_some() {
        "EXELFS"
    } else {
        return Ok(Eels::default());
    };

    let rows = input.section_rows(section).collect::<Vec<_>>();
    let mut eels = Eels {
        enabled: true,
        ..Eels::default()
    };

    if let Some(line) = rows.first() {
        let fields = section_fields_before_star(line)?;
        if let Some(value) = fields.first() {
            eels.beam_energy = parse_f64(line, value)? * 1000.0;
        }
        if let Some(value) = fields.get(1) {
            eels.average = parse_i32(line, value)?;
        }
        if let Some(value) = fields.get(2) {
            eels.cross_terms = parse_i32(line, value)?;
        }
        if let Some(value) = fields.get(3) {
            eels.relativistic = parse_i32(line, value)?;
        }
        if let Some(value) = fields.get(4) {
            eels.input = parse_i32(line, value)?;
        }
        if let Some(value) = fields.get(5) {
            eels.spectrum_column = parse_i32(line, value)?;
        }
    }

    let mut row_index = 1;
    if eels.average != 1 {
        if let Some(line) = rows.get(row_index) {
            let fields = section_fields_before_star(line)?;
            if fields.len() >= 3 {
                let mut vector = [
                    parse_f64(line, fields[0])?,
                    parse_f64(line, fields[1])?,
                    parse_f64(line, fields[2])?,
                ];
                let norm =
                    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
                if norm > 0.0 {
                    for value in &mut vector {
                        *value /= norm;
                    }
                }
                eels.beam_direction = vector;
            }
        }
        row_index += 1;
    }

    if let Some(line) = rows.get(row_index) {
        let fields = section_fields_before_star(line)?;
        if fields.len() >= 2 {
            eels.collection_angle = parse_f64(line, fields[0])? / 1000.0;
            eels.convergence_angle = parse_f64(line, fields[1])? / 1000.0;
        }
    }
    row_index += 1;

    if let Some(line) = rows.get(row_index) {
        let fields = section_fields_before_star(line)?;
        if fields.len() >= 2 {
            eels.qmesh_radial = parse_i32(line, fields[0])?;
            eels.qmesh_angular = parse_i32(line, fields[1])?;
        }
    }
    row_index += 1;

    if let Some(line) = rows.get(row_index) {
        let fields = section_fields_before_star(line)?;
        if fields.len() >= 2 {
            eels.detector = [
                parse_f64(line, fields[0])? / 1000.0,
                parse_f64(line, fields[1])? / 1000.0,
            ];
        }
    }

    if let Some(magic_energy) = magic_energy {
        eels.magic = 1;
        eels.magic_energy = magic_energy;
    }

    if eels.average == 1 {
        eels.polarization_min = 10;
        eels.polarization_step = 1;
        eels.polarization_max = 10;
    } else {
        eels.polarization_min = 1;
        eels.polarization_step = if eels.cross_terms == 1 { 1 } else { 4 };
        eels.polarization_max = 9;
    }

    Ok(eels)
}

fn parse_magic_energy(input: &FeffInput) -> Result<Option<f64>> {
    let Some(line) = card_by_feff_name(input, "MAGIC") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "MAGIC requires emagic"));
    };
    Ok(Some(parse_f64(line, value)?))
}

fn parse_nrixs(input: &FeffInput) -> Result<Option<Nrixs>> {
    let Some(line) = card_by_feff_name(input, "NRIXS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let raw_nq = parse_optional_i32(line, args.first())?.unwrap_or(1);
    if raw_nq == i32::MIN {
        return Err(parse_error(line, "NRIXS q-vector count is too negative"));
    }
    let qaverage = raw_nq < 0;
    let nq = raw_nq.abs().max(1);
    let q_count = nq as usize;
    let mut q_vectors = Vec::with_capacity(q_count);
    q_vectors.push(parse_nrixs_card_vector(line, args, qaverage, nq > 1)?);
    for row in input.section_rows("NRIXS").take(q_count.saturating_sub(1)) {
        q_vectors.push(parse_nrixs_section_vector(row, qaverage)?);
    }
    if q_vectors.len() != q_count {
        return Err(parse_error(
            line,
            format!(
                "NRIXS nq={nq} requires {} q-vector rows after the card",
                q_count.saturating_sub(1)
            ),
        ));
    }
    let first = q_vectors[0];
    let ldecmx = match parse_scalar_card(input, "LDEC")? {
        Some(value) => value,
        None => parse_scalar_card(input, "LDECMX")?.unwrap_or(-1.0),
    };
    Ok(Some(Nrixs {
        nq,
        qaverage,
        qvec: first.vector,
        qnorm: first.norm,
        q_vectors,
        ldecmx: ldecmx as i32,
        lj: parse_scalar_card(input, "LJMAX")?.unwrap_or(0.0) as i32,
    }))
}

fn parse_nrixs_card_vector(
    line: &FeffLine,
    args: &[String],
    qaverage: bool,
    require_weight: bool,
) -> Result<NrixsQVector> {
    if qaverage {
        let qz = parse_optional_f64(line, args.get(1))?.unwrap_or(0.0);
        let weight = parse_nrixs_weight(line, args.get(2), args.get(3), require_weight)?;
        return Ok(NrixsQVector {
            vector: [0.0, 0.0, qz],
            norm: qz,
            weight,
        });
    }

    let vector = [
        parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
        parse_optional_f64(line, args.get(2))?.unwrap_or(0.0),
        parse_optional_f64(line, args.get(3))?.unwrap_or(0.0),
    ];
    let weight = parse_nrixs_weight(line, args.get(4), args.get(5), require_weight)?;
    Ok(NrixsQVector {
        vector,
        norm: vector[0].hypot(vector[1]).hypot(vector[2]),
        weight,
    })
}

fn parse_nrixs_section_vector(line: &FeffLine, qaverage: bool) -> Result<NrixsQVector> {
    let fields = section_fields_before_star(line)?;
    if qaverage {
        if fields.len() < 2 {
            return Err(parse_error(
                line,
                "NRIXS q-average row requires q and weight",
            ));
        }
        let qz = parse_f64(line, fields[0])?;
        let weight =
            parse_nrixs_weight(line, fields.get(1).copied(), fields.get(2).copied(), true)?;
        return Ok(NrixsQVector {
            vector: [0.0, 0.0, qz],
            norm: qz,
            weight,
        });
    }

    if fields.len() < 4 {
        return Err(parse_error(
            line,
            "NRIXS q-vector row requires qx qy qz and weight",
        ));
    }
    let vector = [
        parse_f64(line, fields[0])?,
        parse_f64(line, fields[1])?,
        parse_f64(line, fields[2])?,
    ];
    let weight = parse_nrixs_weight(line, fields.get(3).copied(), fields.get(4).copied(), true)?;
    Ok(NrixsQVector {
        vector,
        norm: vector[0].hypot(vector[1]).hypot(vector[2]),
        weight,
    })
}

fn parse_nrixs_weight(
    line: &FeffLine,
    real: Option<&String>,
    imaginary: Option<&String>,
    require_real: bool,
) -> Result<[f64; 2]> {
    if require_real {
        let Some(real) = real else {
            return Err(parse_error(line, "NRIXS q weight is required when nq > 1"));
        };
        return Ok([
            parse_f64(line, real)?,
            parse_optional_f64(line, imaginary)?.unwrap_or(0.0),
        ]);
    }
    Ok([
        parse_optional_f64(line, real)?.unwrap_or(1.0),
        parse_optional_f64(line, imaginary)?.unwrap_or(0.0),
    ])
}

fn parse_mdff(input: &FeffInput, nrixs: &mut Option<Nrixs>, eels: &Eels) -> Result<Mdff> {
    let Some(line) = card_by_feff_name(input, "MDFF") else {
        return Ok(default_mdff());
    };
    let args = card_args(line)?;
    let imdff = parse_optional_i32(line, args.first())?.unwrap_or(1);
    let mut mdff = Mdff {
        imdff,
        qqmdff: -1.0,
        cosmdff_angle: 0.0,
    };

    if imdff == 2 {
        match args.len() {
            1 => {}
            3 => {
                mdff.qqmdff = parse_f64(line, &args[1])?;
                mdff.cosmdff_angle = parse_f64(line, &args[2])?;
            }
            _ => {
                return Err(parse_error(
                    line,
                    "MDFF 2 requires either no q-prime parameters or qqmdff cosmdff",
                ));
            }
        }
    }

    match imdff {
        value if value <= 0 => Ok(default_mdff()),
        1 => {
            if nrixs.is_some() {
                Ok(mdff)
            } else {
                Err(parse_error(line, "MDFF 1 requires NRIXS"))
            }
        }
        2 => {
            let Some(nrixs) = nrixs.as_mut() else {
                return Err(parse_error(line, "MDFF 2 requires NRIXS"));
            };
            if nrixs.nq != 2 {
                return Err(parse_error(line, "MDFF 2 requires NRIXS nq=2"));
            }
            if mdff.qqmdff >= 0.0 {
                apply_generated_mdff_qprime(line, &mdff, nrixs)?;
            }
            Ok(mdff)
        }
        3 => {
            if eels.enabled {
                Ok(mdff)
            } else {
                Err(parse_error(line, "MDFF 3 requires ELNES or EXELFS"))
            }
        }
        _ => Err(parse_error(line, "MDFF selector must be 0, 1, 2, or 3")),
    }
}

fn apply_generated_mdff_qprime(line: &FeffLine, mdff: &Mdff, nrixs: &mut Nrixs) -> Result<()> {
    if nrixs.q_vectors.len() < 2 {
        return Err(parse_error(line, "MDFF 2 requires two NRIXS q-vector rows"));
    }
    let first = nrixs.q_vectors[0];
    if first.norm == 0.0 {
        return Err(parse_error(
            line,
            "MDFF 2 generated q-prime requires a nonzero first q-vector",
        ));
    }
    let scale = mdff.qqmdff / first.norm;
    let angle = mdff.cosmdff_angle.to_radians();
    let (sin_angle, cos_angle) = angle.sin_cos();
    nrixs.q_vectors[1].vector = [
        first.vector[0] * scale,
        scale * (first.vector[1] * cos_angle + first.vector[2] * sin_angle),
        scale * (-first.vector[1] * sin_angle + first.vector[2] * cos_angle),
    ];
    Ok(())
}

fn default_mdff() -> Mdff {
    Mdff {
        imdff: 0,
        qqmdff: -1.0,
        cosmdff_angle: 0.0,
    }
}

fn parse_rixs(input: &FeffInput) -> Result<Rixs> {
    let mut rixs = Rixs::default();

    if let Some(line) = card_by_feff_name(input, "EDGE") {
        let args = card_args(line)?;
        if let Some(edge) = args.first() {
            rixs.edges.clear();
            rixs.edges.push(edge.to_ascii_uppercase());
            for edge in args.iter().skip(1) {
                let edge = edge.to_ascii_uppercase();
                let is_valence = edge == "VAL";
                rixs.edges.push(edge);
                if is_valence {
                    rixs.mbconv = true;
                    break;
                }
            }
        }
    }

    if let Some(line) = card_by_feff_name(input, "RIXS") {
        let args = card_args(line)?;
        rixs.run = true;
        rixs.gamma_exp[0] = parse_optional_f64(line, args.first())?;
        rixs.gamma_exp[1] = parse_optional_f64(line, args.get(1))?;
        rixs.xmu = parse_optional_f64(line, args.get(2))?;
    }

    Ok(rixs)
}

fn parse_debye(input: &FeffInput) -> Result<Option<Debye>> {
    let Some(line) = card_by_feff_name(input, "DEBYE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(temperature) = args.first() else {
        return Err(parse_error(line, "DEBYE requires a temperature"));
    };
    let Some(debye_temperature) = args.get(1) else {
        return Err(parse_error(line, "DEBYE requires a Debye temperature"));
    };

    let idwopt = parse_optional_i32(line, args.get(2))?.unwrap_or(0);
    let dym_file = (idwopt == 5).then(|| {
        args.get(3)
            .map(|value| strip_card_delimiters(value).to_string())
            .unwrap_or_else(|| "feff.dym".to_string())
    });

    Ok(Some(Debye {
        temperature: parse_f64(line, temperature)?,
        debye_temperature: parse_f64(line, debye_temperature)?,
        idwopt,
        dym_file,
        dmdw_order: parse_optional_i32(line, args.get(4))?.unwrap_or(2),
        dmdw_type: parse_optional_i32(line, args.get(5))?.unwrap_or(0),
        dmdw_route: parse_optional_i32(line, args.get(6))?.unwrap_or(0),
    }))
}

fn parse_spring_input_text(input: &FeffInput, debye: Option<&Debye>) -> Result<Option<String>> {
    if !debye.is_some_and(|debye| matches!(debye.idwopt, 1 | 2)) {
        return Ok(None);
    }

    let path = input
        .source
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("spring.inp");
    let text = std::fs::read_to_string(&path).map_err(|source| IoError::io(&path, source))?;
    parse_spring_inp(&text)?;
    Ok(Some(text))
}

fn parse_dym_input(input: &FeffInput, debye: Option<&Debye>) -> Result<Option<AuxiliaryTextFile>> {
    let Some(dym_file) = debye
        .filter(|debye| debye.idwopt == 5)
        .and_then(|debye| debye.dym_file.as_deref())
    else {
        return Ok(None);
    };

    let output_name = relative_auxiliary_output_name(dym_file)?;
    let path = resolve_auxiliary_path(input, dym_file);
    let text = std::fs::read_to_string(&path).map_err(|source| IoError::io(&path, source))?;
    parse_dym(&text)?;
    let Some(output_name) = output_name else {
        return Ok(None);
    };
    Ok(Some(AuxiliaryTextFile { output_name, text }))
}

fn resolve_auxiliary_path(input: &FeffInput, name: &str) -> PathBuf {
    let path = PathBuf::from(name);
    if path.is_absolute() {
        path
    } else {
        input
            .source
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}

fn relative_auxiliary_output_name(name: &str) -> Result<Option<String>> {
    let path = Path::new(name);
    if path.is_absolute() {
        return Ok(None);
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(IoError::Parse {
                    path: path.to_path_buf(),
                    line: 0,
                    message: "DMDW auxiliary output path must stay within the output directory"
                        .to_string(),
                });
            }
        }
    }

    if normalized.as_os_str().is_empty() {
        return Err(IoError::Parse {
            path: path.to_path_buf(),
            line: 0,
            message: "DMDW auxiliary output path is empty".to_string(),
        });
    }

    Ok(Some(normalized.to_string_lossy().into_owned()))
}

fn parse_rpath(input: &FeffInput) -> Result<Option<f64>> {
    let Some(line) = card_by_feff_name(input, "RPATH") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(radius) = args.first() else {
        return Ok(Some(0.0));
    };
    Ok(Some(parse_f64(line, radius)?))
}

fn parse_nleg(input: &FeffInput) -> Result<Option<i32>> {
    let Some(line) = card_by_feff_name(input, "NLEG") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Ok(Some(7));
    };
    Ok(Some(parse_i32(line, value)?))
}

fn parse_path_symmetry(input: &FeffInput) -> Result<i32> {
    let Some(line) = card_by_feff_name(input, "SYMMETRY") else {
        return Ok(-1);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(line, "SYMMETRY requires ica"));
    };
    let ica = parse_i32(line, value)?;
    Ok(if (1..=7).contains(&ica) { ica } else { -1 })
}

fn parse_dims(input: &FeffInput) -> Result<Option<DimensionLimits>> {
    let Some(line) = card_by_feff_name(input, "DIMS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(nclusx) = args.first() else {
        return Err(parse_error(line, "DIMS requires nclusx"));
    };
    let Some(lx) = args.get(1) else {
        return Err(parse_error(line, "DIMS requires lx"));
    };

    Ok(Some(DimensionLimits {
        nclusx: parse_i32(line, nclusx)?,
        lx: parse_i32(line, lx)?,
    }))
}

fn parse_ldos(input: &FeffInput) -> Result<Option<Ldos>> {
    let Some(line) = card_by_feff_name(input, "LDOS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(emin) = args.first() else {
        return Err(parse_error(line, "LDOS requires emin"));
    };
    let Some(emax) = args.get(1) else {
        return Err(parse_error(line, "LDOS requires emax"));
    };
    let Some(eimag) = args.get(2) else {
        return Err(parse_error(line, "LDOS requires eimag"));
    };

    Ok(Some(Ldos {
        emin: parse_f64(line, emin)?,
        emax: parse_f64(line, emax)?,
        eimag: parse_f64(line, eimag)?,
        neldos: parse_optional_i32(line, args.get(3))?.unwrap_or(101),
        ldostype: parse_optional_i32(line, args.get(4))?.unwrap_or(0),
    }))
}

fn parse_interstitial(input: &FeffInput) -> Result<Option<Interstitial>> {
    let Some(line) = card_by_feff_name(input, "INTERSTITIAL") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    Ok(Some(Interstitial {
        mode: parse_optional_i32(line, args.first())?.unwrap_or(0),
        volume_scale: parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
    }))
}

fn parse_afolp(input: &FeffInput) -> Result<f64> {
    let Some(line) = card_by_feff_name(input, "AFOLP") else {
        return Ok(1.15);
    };
    let args = card_args(line)?;
    parse_optional_f64(line, args.first()).map(|value| value.unwrap_or(1.15))
}

fn parse_overlap_factors(input: &FeffInput) -> Result<Vec<OverlapFactor>> {
    let mut factors = Vec::new();
    for line in input.cards() {
        if let LineKind::Card { keyword, .. } = &line.kind
            && feff_card_token(keyword).map(|(_, display)| display) == Some("FOLP")
        {
            let args = card_args(line)?;
            if args.len() < 2 {
                return Err(parse_error(line, "FOLP requires ipot and folp"));
            }
            let factor = parse_f64(line, &args[1])?;
            if !factor.is_finite() {
                return Err(parse_error(line, "FOLP factor must be finite"));
            }
            factors.push(OverlapFactor {
                potential_index: parse_i32(line, &args[0])?,
                factor,
            });
        }
    }
    Ok(factors)
}

fn parse_ionizations(input: &FeffInput) -> Result<Vec<Ionization>> {
    let mut ionizations = Vec::new();
    for line in input.cards() {
        if let LineKind::Card { keyword, .. } = &line.kind
            && feff_card_token(keyword).map(|(_, display)| display) == Some("ION")
        {
            let args = card_args(line)?;
            if args.len() < 2 {
                return Err(parse_error(line, "ION requires ipot and ionization"));
            }
            let value = parse_f64(line, &args[1])?;
            if !value.is_finite() {
                return Err(parse_error(line, "ION value must be finite"));
            }
            ionizations.push(Ionization {
                potential_index: parse_i32(line, &args[0])?,
                value,
            });
        }
    }
    Ok(ionizations)
}

fn parse_potentials(input: &FeffInput) -> Result<Vec<Potential>> {
    input
        .section_rows("POTENTIALS")
        .map(|line| {
            let fields = section_fields_before_star(line)?;
            if fields.len() < 2 {
                return Err(parse_error(line, "POTENTIALS rows require ipot and Z"));
            }
            let z = parse_i32(line, fields[1])?;

            Ok(Potential {
                ipot: parse_i32(line, fields[0])?,
                z: Some(z),
                z_token: fields[1].clone(),
                tag: fields.get(2).map(|value| (*value).clone()),
                lmax1: parse_optional_i32(line, fields.get(3).copied())?,
                lmax2: parse_optional_i32(line, fields.get(4).copied())?,
                xnatph: parse_optional_f64(line, fields.get(5).copied())?,
                spinph: parse_optional_f64(line, fields.get(6).copied())?,
            })
        })
        .collect()
}

fn parse_atoms(input: &FeffInput) -> Result<Vec<Atom>> {
    input
        .section_rows("ATOMS")
        .map(|line| {
            let fields = section_fields_before_star(line)?;
            if fields.len() < 4 {
                return Err(parse_error(line, "ATOMS rows require x y z ipot"));
            }

            Ok(Atom {
                x: parse_f64(line, fields[0])?,
                y: parse_f64(line, fields[1])?,
                z: parse_f64(line, fields[2])?,
                ipot: parse_i32(line, fields[3])?,
                tag: fields.get(4).map(|value| (*value).clone()),
                distance: fields
                    .iter()
                    .skip(5)
                    .find_map(|value| parse_f64(line, value).ok()),
                index: fields
                    .iter()
                    .skip(5)
                    .find_map(|value| value.parse::<usize>().ok()),
            })
        })
        .collect()
}

fn card_args(line: &FeffLine) -> Result<&[String]> {
    match &line.kind {
        LineKind::Card { args, .. } => Ok(args),
        LineKind::SectionData { .. } => Err(parse_error(line, "expected card line")),
    }
}

fn section_fields(line: &FeffLine) -> Result<&[String]> {
    match &line.kind {
        LineKind::SectionData { fields, .. } => Ok(fields),
        LineKind::Card { .. } => Err(parse_error(line, "expected section data line")),
    }
}

fn section_fields_before_star(line: &FeffLine) -> Result<Vec<&String>> {
    Ok(section_fields(line)?
        .iter()
        .take_while(|field| field.as_str() != "*")
        .collect())
}

fn parse_i32(line: &FeffLine, value: &str) -> Result<i32> {
    value
        .parse::<i32>()
        .map_err(|_| parse_error(line, format!("invalid integer {value:?}")))
}

fn parse_optional_i32(line: &FeffLine, value: Option<&String>) -> Result<Option<i32>> {
    value.map(|value| parse_i32(line, value)).transpose()
}

fn parse_logical(line: &FeffLine, value: &str) -> Result<bool> {
    let normalized = value.trim_matches('.').to_ascii_uppercase();
    match normalized.as_str() {
        "T" | "TRUE" | "1" => Ok(true),
        "F" | "FALSE" | "0" => Ok(false),
        _ => Err(parse_error(line, format!("invalid logical {value:?}"))),
    }
}

fn parse_f64(line: &FeffLine, value: &str) -> Result<f64> {
    value
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error(line, format!("invalid float {value:?}")))
}

fn parse_optional_f64(line: &FeffLine, value: Option<&String>) -> Result<Option<f64>> {
    value.map(|value| parse_f64(line, value)).transpose()
}

fn parse_error(line: &FeffLine, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: line.location.path.clone(),
        line: line.location.line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context as _, ensure};

    #[test]
    fn extracts_common_structure_cards() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
EDGE K
S02 1.0
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 0
SCF 5.0 0 40 0.3
EXCHANGE 0 1.0 2.0
EXAFS 20.0
FMS 4.0 1 0 0.002 0.003 20.0
COMPTON 7.0 300 1
RHOZZP
CGRID 12.0 20 21 22 23
DEBYE 190 315 0
RPATH 5.5
DIMS 100 4
LDOS -30 20 0.1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0 0.0 0
1.0 0.0 0.0 1 Cu1 1.0 1
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(
            doc.active_cards,
            [
                "ATOMS",
                "CONTROL",
                "EXCHANGE",
                "TITLE",
                "RPATH",
                "DEBYE",
                "PRINT",
                "POTENTIALS",
                "EXAFS",
                "EDGE",
                "SCF",
                "FMS",
                "LDOS",
                "S02",
                "DIMS",
                "COMPTON",
                "RHOZZP",
                "CGRID"
            ]
        );
        assert_eq!(doc.titles, ["Cu crystal"]);
        let edge = doc.edge.context("missing parsed edge")?;
        assert_eq!(edge.label, "K");
        assert_eq!(doc.s02, Some(1.0));
        assert_eq!(doc.control, Some([1, 1, 1, 1, 1, 1]));
        assert_eq!(doc.scf.as_ref().map(|scf| scf.iterations), Some(40));
        assert_eq!(
            doc.exchange.as_ref().map(|exchange| exchange.vr0),
            Some(1.0)
        );
        assert_eq!(doc.exafs.as_ref().map(|exafs| exafs.xkmax), Some(20.0));
        assert_eq!(doc.ispec, 5);
        assert_eq!(doc.fms.as_ref().map(|fms| fms.radius), Some(4.0));
        assert_eq!(doc.fms.as_ref().map(|fms| fms.lfms), Some(1));
        assert_eq!(doc.fms.as_ref().map(|fms| fms.rdirec), Some(8.0));
        assert!(doc.compton.do_compton);
        assert!(doc.compton.do_rhozzp);
        assert!(doc.compton.force_jzzp);
        assert_eq!(doc.compton.pqmax, 7.0);
        assert_eq!(doc.compton.npq, 300);
        assert_eq!(doc.compton.ns, 20);
        assert_eq!(doc.compton.nphi, 21);
        assert_eq!(doc.compton.nz, 22);
        assert_eq!(doc.compton.nzp, 23);
        assert_eq!(doc.compton.zpmax, 12.0);
        assert_eq!(
            doc.debye.as_ref().map(|debye| debye.temperature),
            Some(190.0)
        );
        assert_eq!(doc.rpath, Some(5.5));
        assert_eq!(doc.dims, Some(DimensionLimits { nclusx: 100, lx: 4 }));
        assert_eq!(doc.ldos.as_ref().map(|ldos| ldos.eimag), Some(0.1));
        assert_eq!(doc.potentials.len(), 2);
        assert_eq!(doc.atoms.len(), 2);
        assert_eq!(doc.atoms[1].tag.as_deref(), Some("Cu1"));
        Ok(())
    }

    #[test]
    fn expands_feff7_control_and_print_cards_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CONTROL 0 1 0 1
PRINT 5 2 1 4
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.control, Some([0, 0, 0, 1, 0, 1]));
        assert_eq!(doc.print, Some([5, 5, 5, 2, 1, 4]));
        Ok(())
    }

    #[test]
    fn active_cards_use_feff_token_order_and_alias_names() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Alias test
PLASMON 2
SFCONV
WARNION
CONFIG card 1
2 1 0
XNCD
RMAX 4.5
POTENTIAL
0 29 Cu
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(
            doc.active_cards,
            [
                "TITLE",
                "RPATH",
                "POTENTIALS",
                "XMCD",
                "MPSE",
                "SFCONV",
                "CONFIGURATION",
                "WARN"
            ]
        );
        assert_eq!(
            doc.input_cards,
            [
                "TITLE",
                "MPSE",
                "SFCONV",
                "WARN",
                "CONFIGURATION",
                "XMCD",
                "RPATH",
                "POTENTIALS"
            ]
        );
        Ok(())
    }

    #[test]
    fn extracts_xsph_advanced_tdl_and_pmbse_controls_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TDLDA 7
PMBSE 3 4 5 6
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(doc.xsph_advanced.izstd, 1);
        assert_eq!(doc.xsph_advanced.ifxc, 7);
        assert_eq!(doc.xsph_advanced.ipmbse, 3);
        assert_eq!(doc.xsph_advanced.itdlda, 2);
        assert_eq!(doc.xsph_advanced.nonlocal, 4);
        assert_eq!(doc.xsph_advanced.ibasis, 6);

        let pmbse_only = FeffInput::parse_str(
            "feff.inp",
            r#"
PMBSE 3 4 5 6
END
"#,
        )?;
        let doc = FeffDocument::from_input(&pmbse_only)?;
        assert_eq!(doc.xsph_advanced.izstd, 0);
        assert_eq!(doc.xsph_advanced.ifxc, 5);
        assert_eq!(doc.xsph_advanced.ipmbse, 3);
        assert_eq!(doc.xsph_advanced.itdlda, 2);
        assert_eq!(doc.xsph_advanced.nonlocal, 4);
        assert_eq!(doc.xsph_advanced.ibasis, 6);
        Ok(())
    }

    #[test]
    fn extracts_cfaverage_and_absorber_potential_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CFAVERAGE 1 0 0
POTENTIALS
1 29 Cu
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
2.0 0.0 0.0 1 Cu2
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(doc.cfaverage.nabs, 3);
        assert_eq!(doc.cfaverage.iphabs, 1);
        assert_eq!(doc.cfaverage.rclabs, 100000.0);
        assert_eq!(doc.potentials.len(), 2);
        assert_eq!(doc.potentials[0].ipot, 0);
        assert_eq!(doc.potentials[0].z, Some(29));
        assert_eq!(doc.potentials[0].tag.as_deref(), Some("Cu"));
        assert_eq!(doc.potentials[1].ipot, 1);

        let limited = FeffInput::parse_str(
            "feff.inp",
            r#"
CFAVERAGE 1 2 5.0
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
2.0 0.0 0.0 1 Cu2
END
"#,
        )?;
        let doc = FeffDocument::from_input(&limited)?;
        assert_eq!(doc.cfaverage.nabs, 2);
        assert_eq!(doc.cfaverage.iphabs, 1);
        assert_eq!(doc.cfaverage.rclabs, 5.0);
        Ok(())
    }

    #[test]
    fn extracts_common_control_aliases_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITL Alias controls
CONT 1 0 1 0 1 0
PRIN 3 4 5 6 7 8
EXCH 2 1.25 0.5 9
CORR -1.5 0.75
RGRI 0.03
CORE NONE
UNFR
ABSO
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.titles, ["Alias controls"]);
        assert_eq!(doc.control, Some([1, 0, 1, 0, 1, 0]));
        assert_eq!(doc.print, Some([3, 4, 5, 6, 7, 8]));
        let exchange = doc.exchange.context("missing EXCHANGE alias")?;
        assert_eq!(exchange.ixc, 2);
        assert_eq!(exchange.vr0, 1.25);
        assert_eq!(exchange.vi0, 0.5);
        assert_eq!(exchange.ixc0, Some(9));
        assert_eq!(doc.corrections, [-1.5, 0.75]);
        assert_eq!(doc.rgrid, 0.03);
        assert_eq!(doc.nohole, 0);
        assert!(doc.unfreezef);
        assert!(doc.absolute);
        assert_eq!(
            doc.active_cards,
            [
                "CONTROL",
                "EXCHANGE",
                "TITLE",
                "PRINT",
                "CORRECTIONS",
                "RGRID",
                "UNFREEZEF",
                "ABSOLUTE",
                "COREHOLE"
            ]
        );
        assert_eq!(
            doc.input_cards,
            [
                "TITLE",
                "CONTROL",
                "PRINT",
                "EXCHANGE",
                "CORRECTIONS",
                "RGRID",
                "COREHOLE",
                "UNFREEZEF",
                "ABSOLUTE"
            ]
        );
        Ok(())
    }

    #[test]
    fn rejects_incomplete_exchange_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EXCHANGE 5
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete EXCHANGE should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("EXCHANGE requires ixc, vr0, and vi0"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_incomplete_magic_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
ELNES
MAGIC
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete MAGIC should be rejected")?;
        ensure!(
            error.to_string().contains("MAGIC requires emagic"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_incomplete_corrections_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CORRECTIONS -1.0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete CORRECTIONS should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("CORRECTIONS requires real and imaginary shifts"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_incomplete_crpa_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CRPA 2
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete CRPA should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("CRPA requires l and rcut values"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_incomplete_hubbard_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
HUBBARD 1.0 0.5
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete HUBBARD should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("HUBBARD requires U, J, fermi_shift, and l values"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_redundant_nohole_and_corehole_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
COREHOLE FSR
NOHOLE
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("NOHOLE plus COREHOLE should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("NOHOLE and COREHOLE cards are mutually exclusive"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn extracts_four_character_control_aliases_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITL Prefix aliases
EDGE K
HOLE 1 0.77
SCF 5.0 1 40 0.3
FMS 4.0 1 0 0.002 0.003 7.0
DEBY 190 315 0
RPAT 4.5
CRIT 3.1 2.2
PCRI 0.7 0.8
RMUL 2.0
AFOL 1.25
FOLP 1 1.35
PLAS 3 50
ELNE
100.0 1 1 1 1 1
MAGI 7112.0
POTE
0 29 Cu0
1 29 Cu1
ATOM
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.titles, ["Prefix aliases"]);
        assert_eq!(doc.hole, Some(1));
        assert_eq!(doc.s02, Some(0.77));
        assert_eq!(doc.critcw, 3.1);
        assert_eq!(doc.critpw, 2.2);
        assert_eq!(doc.pcritk, 0.7);
        assert_eq!(doc.pcrith, 0.8);
        assert_eq!(doc.r_multiplier, 2.0);
        assert_eq!(doc.rpath, Some(9.0));
        assert_eq!(doc.scf.as_ref().map(|scf| scf.radius), Some(10.0));
        assert_eq!(doc.fms.as_ref().map(|fms| fms.radius), Some(8.0));
        assert_eq!(
            doc.debye.as_ref().map(|debye| debye.temperature),
            Some(190.0)
        );
        assert_eq!(doc.afolp, 1.25);
        assert_eq!(doc.overlap_factors.len(), 1);
        assert_eq!(doc.overlap_factors[0].factor, 1.35);
        assert_eq!(doc.i_plsmn, 3);
        assert_eq!(doc.n_poles, 50);
        assert!(doc.eels.enabled);
        assert_eq!(doc.eels.magic, 1);
        assert_eq!(doc.eels.magic_energy, 7112.0);
        assert_eq!(doc.atoms[1].x, 2.0);
        assert_eq!(
            doc.active_cards,
            [
                "ATOMS",
                "HOLE",
                "TITLE",
                "FOLP",
                "RPATH",
                "DEBYE",
                "RMULT",
                "POTENTIALS",
                "CRITERIA",
                "PCRITERIA",
                "AFOLP",
                "EDGE",
                "SCF",
                "FMS",
                "MPSE",
                "ELNES",
                "MAGIC"
            ]
        );
        Ok(())
    }

    #[test]
    fn extracts_block_alias_rows_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTE
0 29 Cu0
1 29 Cu1
ATOM
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.potentials.len(), 2);
        assert_eq!(doc.potentials[0].tag.as_deref(), Some("Cu0"));
        assert_eq!(doc.atoms.len(), 2);
        assert_eq!(doc.atoms[1].tag.as_deref(), Some("Cu1"));
        assert_eq!(doc.active_cards, ["ATOMS", "POTENTIALS"]);
        assert_eq!(doc.input_cards, ["POTENTIALS", "ATOMS"]);
        Ok(())
    }

    #[test]
    fn extracts_compton_aliases_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
COMP 7.0 300 1
RHOZ
CGRI 12.0 20 21 22 23
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.ispec, 5);
        assert_eq!(doc.nohole, 0);
        assert!(doc.compton.do_compton);
        assert!(doc.compton.do_rhozzp);
        assert!(doc.compton.force_jzzp);
        assert_eq!(doc.compton.pqmax, 7.0);
        assert_eq!(doc.compton.npq, 300);
        assert_eq!(doc.compton.zpmax, 12.0);
        assert_eq!(doc.compton.ns, 20);
        assert_eq!(doc.compton.nphi, 21);
        assert_eq!(doc.compton.nz, 22);
        assert_eq!(doc.compton.nzp, 23);
        assert_eq!(doc.active_cards, ["COMPTON", "RHOZZP", "CGRID"]);
        assert_eq!(doc.input_cards, ["COMPTON", "RHOZZP", "CGRID"]);
        Ok(())
    }

    #[test]
    fn extracts_spectroscopy_aliases_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANE 20.0 1.25 0.2
POLA 1.0 0.0 0.0
ELLI 0.25 0.0 1.0 0.0
MULT 2 1
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.ispec, 1);
        assert_eq!(doc.ipol, 1);
        assert_eq!(doc.polarization_vector, [1.0, 0.0, 0.0]);
        assert_eq!(doc.ellipticity, 0.25);
        assert_eq!(doc.incidence_vector, [0.0, 1.0, 0.0]);
        assert_eq!(doc.le2, 2);
        assert_eq!(doc.l2lp, 1);
        assert_eq!(doc.spectrum_grid.xkmax, 20.0);
        assert_eq!(doc.spectrum_grid.xkstep, 1.25);
        assert_eq!(doc.spectrum_grid.vixan, 0.2);
        assert_eq!(
            doc.active_cards,
            ["XANES", "POLARIZATION", "ELLIPTICITY", "MULT"]
        );
        assert_eq!(
            doc.input_cards,
            ["XANES", "POLARIZATION", "ELLIPTICITY", "MULT"]
        );

        let fprime = FeffDocument::from_input(&FeffInput::parse_str(
            "feff.inp",
            "FPRI -5.0 10.0 0.25\nEND\n",
        )?)?;
        assert_eq!(fprime.ispec, 4);
        assert_eq!(fprime.spectrum_grid.xkmax, -5.0);
        assert_eq!(fprime.spectrum_grid.xkstep, 10.0);
        assert_eq!(fprime.spectrum_grid.vixan, 0.25);
        assert_eq!(fprime.active_cards, ["FPRIME"]);

        let exafs =
            FeffDocument::from_input(&FeffInput::parse_str("feff.inp", "EXAF 15.0\nEND\n")?)?;
        assert_eq!(exafs.exafs.as_ref().map(|exafs| exafs.xkmax), Some(15.0));
        assert_eq!(exafs.spectrum_grid.xkmax, 15.0);
        assert_eq!(exafs.active_cards, ["EXAFS"]);
        Ok(())
    }

    #[test]
    fn rejects_incomplete_polarization_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
POLARIZATION 1.0 0.0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete POLARIZATION should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("POLARIZATION requires x, y, and z"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_incomplete_ellipticity_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
ELLIPTICITY 0.25 0.0 1.0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete ELLIPTICITY should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("ELLIPTICITY requires ellipticity and incident direction"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_incomplete_fprime_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
FPRIME -5.0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete FPRIME should be rejected")?;
        ensure!(
            error.to_string().contains("FPRIME requires emin and emax"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn listed_reciprocal_requires_handoff_even_when_real_follows_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RECIPROCAL
REAL
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("listed RECIPROCAL should require reciprocal handoff cards")?;
        ensure!(
            error
                .to_string()
                .contains("KMESH and TARGET are required for RECIPROCAL card"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn later_reciprocal_overrides_real_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
REAL
RECIPROCAL
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("RECIPROCAL should still require reciprocal handoff cards")?;
        ensure!(
            error
                .to_string()
                .contains("KMESH and TARGET are required for RECIPROCAL card"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_reciprocal_without_lattice_or_cif_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RECIPROCAL
KMESH 10 0
TARGET 1
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("RECIPROCAL should require one structure source")?;
        ensure!(
            error
                .to_string()
                .contains("use either LATTICE or CIF with RECIPROCAL card"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_reciprocal_with_lattice_and_cif_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 1.0
CIF dummy.cif
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("RECIPROCAL should reject simultaneous LATTICE and CIF")?;
        ensure!(
            error
                .to_string()
                .contains("use either LATTICE or CIF with RECIPROCAL card"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_incomplete_strfac_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 1.0
STRFAC 1.0 2.0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete STRFAC should be rejected")?;
        ensure!(
            error.to_string().contains("STRFAC requires three values"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_cgrid_without_compton_or_rhozzp_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CGRID 10.0 32 32 32 120
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("CGRID should require COMPTON or RHOZZP")?;
        ensure!(
            error
                .to_string()
                .contains("Cannot use CGRID without COMPTON or RHOZZP.  Exiting."),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_hubbard_with_reciprocal_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 1.0
HUBBARD 1.0 0.5 0.0 2
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("HUBBARD should be rejected with RECIPROCAL")?;
        ensure!(
            error
                .to_string()
                .contains("Cannot use RECIPROCAL with HUBBARD."),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn converts_reciprocal_lattice_coordinates_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
COORDINATES 1
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let reciprocal = doc
            .reciprocal_input
            .as_ref()
            .and_then(|input| input.cell.as_ref())
            .context("missing reciprocal cell")?;

        assert_eq!(doc.coordinate_mode, 1);
        assert_eq!(reciprocal.positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(reciprocal.positions[1], [0.5, 0.0, 0.0]);
        assert!(
            doc.atoms
                .iter()
                .any(|atom| atom.ipot == 1 && (atom.x.abs() - 1.0).abs() < 1.0e-12)
        );
        Ok(())
    }

    #[test]
    fn extracts_jump_removal_aliases() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
JUMP
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert!(doc.jump_removal);
        assert_eq!(doc.active_cards, ["JUMPRM"]);
        assert_eq!(doc.input_cards, ["JUMPRM"]);
        Ok(())
    }

    #[test]
    fn extracts_nogeom_output_switch() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
NOGEOM
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert!(doc.no_geom);
        assert_eq!(doc.active_cards, ["NOGEOM"]);
        assert_eq!(doc.input_cards, ["NOGEOM"]);
        Ok(())
    }

    #[test]
    fn extracts_interstitial_alias_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
INTE 1 1.25
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        let interstitial = doc.interstitial.context("missing INTERSTITIAL")?;
        assert_eq!(interstitial.mode, 1);
        assert_eq!(interstitial.volume_scale, 1.25);
        assert_eq!(doc.active_cards, ["INTERSTITIAL"]);
        assert_eq!(doc.input_cards, ["INTERSTITIAL"]);
        Ok(())
    }

    #[test]
    fn extracts_hubbard_alias_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
HUBB 3.0 0.5 -0.1 2
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.hubbard.i_hubbard, 2);
        assert_eq!(doc.hubbard.mldos_hubb, 2);
        assert_eq!(doc.hubbard.u, 3.0);
        assert_eq!(doc.hubbard.j, 0.5);
        assert_eq!(doc.hubbard.fermi_shift, -0.1);
        assert_eq!(doc.hubbard.l, 2);
        assert_eq!(doc.active_cards, ["HUBBARD"]);
        assert_eq!(doc.input_cards, ["HUBBARD"]);
        Ok(())
    }

    #[test]
    fn extracts_nrixs_alias_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIX 1 0.0 0.0 2.0
LDEC 4
LJMAX 2
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        let nrixs = doc.nrixs.as_ref().context("missing NRIXS")?;
        assert_eq!(doc.ispec, 1);
        assert_eq!(nrixs.nq, 1);
        assert!(!nrixs.qaverage);
        assert_eq!(nrixs.qvec, [0.0, 0.0, 2.0]);
        assert_eq!(nrixs.qnorm, 2.0);
        assert_eq!(
            nrixs.q_vectors.as_slice(),
            &[NrixsQVector {
                vector: [0.0, 0.0, 2.0],
                norm: 2.0,
                weight: [1.0, 0.0],
            }]
        );
        assert_eq!(nrixs.ldecmx, 4);
        assert_eq!(nrixs.lj, 2);
        assert_eq!(doc.active_cards, ["XANES", "NRIXS", "LJMAX", "LDECMX"]);
        assert_eq!(doc.input_cards, ["XANES", "NRIXS", "LDECMX", "LJMAX"]);
        Ok(())
    }

    #[test]
    fn rejects_nrixs_without_xanes_or_exafs_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
NRIXS 1 0.0 0.0 1.0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("NRIXS without XANES or EXAFS should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("NRIXS must be combined with XANES or EXAFS"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_multiple_spectroscopy_cards_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
EXAFS
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("multiple spectroscopy cards should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("ERROR more than one type of spectroscopy selected"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_ldec_without_nrixs_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
LDEC 4
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("LDEC without NRIXS should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("LDEC and LJMAX cards only allowed with NRIXS"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_nrixs_forbidden_cards_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIXS 1 0.0 0.0 1.0
POLARIZATION 1.0 0.0 0.0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("NRIXS with POLARIZATION should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("card is explicitly forbidden for NRIXS"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn extracts_nrixs_multi_q_rows_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIXS 2 0.0 0.0 2.0 0.25
1.0 0.0 0.0 0.75
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        let nrixs = doc.nrixs.as_ref().context("missing NRIXS")?;
        assert_eq!(nrixs.nq, 2);
        assert_eq!(nrixs.qvec, [0.0, 0.0, 2.0]);
        assert_eq!(
            nrixs.q_vectors.as_slice(),
            &[
                NrixsQVector {
                    vector: [0.0, 0.0, 2.0],
                    norm: 2.0,
                    weight: [0.25, 0.0],
                },
                NrixsQVector {
                    vector: [1.0, 0.0, 0.0],
                    norm: 1.0,
                    weight: [0.75, 0.0],
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn extracts_nrixs_qaverage_complex_weights_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIXS -2 2.0 0.25 0.10
3.0 0.75 0.20
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        let nrixs = doc.nrixs.as_ref().context("missing NRIXS")?;
        assert_eq!(nrixs.nq, 2);
        assert!(nrixs.qaverage);
        assert_eq!(nrixs.qvec, [0.0, 0.0, 2.0]);
        assert_eq!(nrixs.qnorm, 2.0);
        assert_eq!(
            nrixs.q_vectors.as_slice(),
            &[
                NrixsQVector {
                    vector: [0.0, 0.0, 2.0],
                    norm: 2.0,
                    weight: [0.25, 0.10],
                },
                NrixsQVector {
                    vector: [0.0, 0.0, 3.0],
                    norm: 3.0,
                    weight: [0.75, 0.20],
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn extracts_mdff_handoff_controls_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIXS 1 0.0 0.0 2.0
MDFF 1
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.mdff.imdff, 1);
        assert_eq!(doc.mdff.qqmdff, -1.0);
        assert_eq!(doc.mdff.cosmdff_angle, 0.0);

        let error = FeffDocument::from_input(&FeffInput::parse_str("feff.inp", "MDFF 1\nEND\n")?)
            .err()
            .context("MDFF 1 without NRIXS should be rejected")?;
        ensure!(
            error.to_string().contains("MDFF 1 requires NRIXS"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn extracts_mdff2_generated_qprime_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
NRIXS 2 0.0 0.0 2.0 1.0
1.0 0.0 0.0 1.0
MDFF 2 3.0 45.0
END
"#,
        )?;
        let doc = FeffDocument::from_input(&input)?;
        let nrixs = doc.nrixs.as_ref().context("missing NRIXS")?;
        assert_eq!(doc.mdff.imdff, 2);
        assert_eq!(doc.mdff.qqmdff, 3.0);
        assert_eq!(doc.mdff.cosmdff_angle, 45.0);
        assert!((nrixs.q_vectors[1].vector[0] - 0.0).abs() < 1.0e-12);
        assert!((nrixs.q_vectors[1].vector[1] - 2.121_320_343_559_643).abs() < 1.0e-12);
        assert!((nrixs.q_vectors[1].vector[2] - 2.121_320_343_559_643).abs() < 1.0e-12);
        assert_eq!(nrixs.q_vectors[1].norm, 1.0);

        let error = FeffDocument::from_input(&FeffInput::parse_str(
            "feff.inp",
            "XANES\nNRIXS 1 0.0 0.0 2.0\nMDFF 2\nEND\n",
        )?)
        .err()
        .context("MDFF 2 without nq=2 should be rejected")?;
        ensure!(
            error.to_string().contains("MDFF 2 requires NRIXS nq=2"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn extracts_density_alias_payload_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
DENS
line line.dat 0.0 0.0 0.0 core
1.0 0.0 0.0 101
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.ispec, 5);
        assert_eq!(doc.density_records.len(), 2);
        assert_eq!(doc.active_cards, ["DENS"]);
        assert_eq!(doc.input_cards, ["DENS"]);
        Ok(())
    }

    #[test]
    fn extracts_configuration_alias_payload_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CONF card 1
2 1 0
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.config_type, 2);
        assert_eq!(doc.config_records, ["2 1 0"]);
        assert_eq!(doc.active_cards, ["CONFIGURATION"]);
        assert_eq!(doc.input_cards, ["CONFIGURATION"]);
        Ok(())
    }

    #[test]
    fn extracts_external_potential_restart_switches() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EXTPOT
RESTART
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert!(doc.external_pot);
        assert!(doc.restart_from_pot_bin);
        assert_eq!(doc.active_cards, ["EXTPOT", "RESTART"]);
        Ok(())
    }

    #[test]
    fn extracts_chemical_shift_alias() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CHSH 3
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.chsh_type, 3);
        assert_eq!(doc.active_cards, ["CHSHIFT"]);
        Ok(())
    }

    #[test]
    fn extracts_corval_and_highz_handoff_controls() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CORV -120
HIGHZ
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.corval_emin, -120.0);
        assert!(doc.finite_nucleus);
        assert_eq!(doc.active_cards, ["CORVAL", "HIGHZ"]);
        Ok(())
    }

    #[test]
    fn extracts_warnion_alias_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
WARN
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert!(doc.warn_ion);
        assert_eq!(doc.active_cards, ["WARN"]);
        Ok(())
    }

    #[test]
    fn extracts_scf_tail_controls() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SCFTH 1 6.5 640 80 5e-5
SCFR 2.0 4
TOLS 0.1 0.002 0.0003
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.scf_thermal.iscfth, 1);
        assert_eq!(doc.scf_thermal.emaxscf, 6.5);
        assert_eq!(doc.scf_thermal.negrid, 640);
        assert_eq!(doc.scf_thermal.nmu, 80);
        assert_eq!(doc.scf_thermal.xntol, 5.0e-5);
        assert!(doc.scf_ramp.enabled);
        assert_eq!(doc.scf_ramp.rfms_start, 2.0);
        assert_eq!(doc.scf_ramp.nramp, 4);
        assert_eq!(doc.scf_tolerances.tolmu, 0.001);
        assert_eq!(doc.scf_tolerances.tolq, 0.002);
        assert_eq!(doc.scf_tolerances.tolqp, 0.0003);
        assert_eq!(doc.active_cards, ["SCFTH", "SCFR", "TOLS"]);
        Ok(())
    }

    #[test]
    fn scales_scf_tolerances_for_negative_tols_factor() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TOLS -2
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.scf_tolerances.tolmu, -0.002);
        assert_eq!(doc.scf_tolerances.tolq, -0.002);
        assert_eq!(doc.scf_tolerances.tolqp, -0.0004);
        assert_eq!(doc.active_cards, ["TOLS"]);
        Ok(())
    }

    #[test]
    fn extracts_ff2x_convolution_and_damping_controls() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
MBCONV
SIG2 0.012
SIG3 0.034 250
SIGGK 0.056
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert!(doc.many_body_convolution);
        assert_eq!(doc.fine_structure_damping.sig2g, 0.012);
        assert_eq!(doc.fine_structure_damping.alphat, 0.034);
        assert_eq!(doc.fine_structure_damping.thetae, 250.0);
        assert_eq!(doc.fine_structure_damping.sig_gk, 0.056);
        assert_eq!(doc.active_cards, ["SIG2", "SIG3", "MBCONV", "SIGGK"]);
        Ok(())
    }

    #[test]
    fn ignores_empty_siggk_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SIGGK
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.fine_structure_damping.sig_gk, 0.0);
        assert_eq!(doc.active_cards, ["SIGGK"]);
        Ok(())
    }

    #[test]
    fn extracts_sfconv_alias_and_controls_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
XANES
SO2C
SELF
SFSE 2.5
RCONV 10.25 longfilename.dat
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert!(doc.sfconv);
        assert_eq!(
            doc.sfconv_input.control,
            SfconvControl {
                msfconv: 1,
                ipse: 1,
                ipsk: 1,
            }
        );
        assert_eq!(doc.sfconv_input.window.wsigk, 2.5);
        assert_eq!(doc.sfconv_input.window.cen, 10.25);
        assert_eq!(doc.sfconv_input.spectrum.ispec, 1);
        assert_eq!(doc.sfconv_input.spectrum.ipr6, 0);
        assert_eq!(doc.sfconv_input.cfname, "longfilename");
        assert_eq!(
            doc.active_cards,
            ["XANES", "SFCONV", "SELF", "SFSE", "RCONV"]
        );
        assert_eq!(
            doc.input_cards,
            ["XANES", "SFCONV", "SELF", "SFSE", "RCONV"]
        );
        Ok(())
    }

    #[test]
    fn extracts_genfmt_and_real_phase_switches() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
IORDER 4
POLARIZATION 1 0 0
RPHASES
NSTAR
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.iorder, 4);
        assert_eq!(doc.lreal, 2);
        assert!(doc.nstar);
        assert_eq!(
            doc.active_cards,
            ["IORD", "POLARIZATION", "RPHASES", "NSTAR"]
        );
        Ok(())
    }

    #[test]
    fn disables_nstar_without_linear_polarization_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
NSTAR
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert!(!doc.nstar);
        assert_eq!(doc.active_cards, ["NSTAR"]);
        Ok(())
    }

    #[test]
    fn extracts_path_symmetry_controls() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SYMMETRY 3
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.path_symmetry, 3);
        assert_eq!(doc.active_cards, ["SYMMETRY"]);
        Ok(())
    }

    #[test]
    fn clamps_invalid_path_symmetry_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SYMMETRY 9
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.path_symmetry, -1);
        assert_eq!(doc.active_cards, ["SYMMETRY"]);
        Ok(())
    }

    #[test]
    fn extracts_bandstructure_handoff_controls() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
BANDSTRUCTURE -5.0 10.0 0.25 2 64 T
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.band_input.mband, 1);
        assert_eq!(doc.band_input.energy_mesh.emin, -5.0);
        assert_eq!(doc.band_input.energy_mesh.emax, 10.0);
        assert_eq!(doc.band_input.energy_mesh.estep, 0.25);
        assert_eq!(doc.band_input.ikpath, 2);
        assert_eq!(doc.band_input.nkp, 64);
        assert!(doc.band_input.freeprop);
        assert_eq!(doc.active_cards, ["BAND"]);
        Ok(())
    }

    #[test]
    fn rejects_incomplete_bandstructure_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
BANDSTRUCTURE -5.0 10.0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete BANDSTRUCTURE should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("BANDSTRUCTURE requires emin emax estep ikpath"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn extracts_fullspectrum_handoff_switch() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
FULLSPECTRUM
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.full_spectrum_input.m_full_spectrum, 1);
        assert_eq!(doc.active_cards, ["FULLSPECTRUM"]);
        Ok(())
    }

    #[test]
    fn extracts_scxc_handoff_controls_in_input_order() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SCXC 22
TEMP 0.25 12
SCXC 21
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.electronic_temperature, 0.25);
        assert_eq!(doc.iscfxc, 21);
        assert_eq!(doc.active_cards, ["TEMP", "SCXC"]);
        assert_eq!(doc.input_cards, ["SCXC", "TEMP", "SCXC"]);
        Ok(())
    }

    #[test]
    fn rejects_invalid_scxc_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SCXC 99
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("invalid SCXC should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("SCXC iscfxc must be one of 11, 12, 21, or 22"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn extracts_screen_handoff_controls() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SCREEN rfms 5.5
SCREEN ner 64.4
SCREEN lfxc 2.4
SCREEN ermin 2e-3
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.screen_input.rfms, 5.5);
        assert_eq!(doc.screen_input.ner, 64);
        assert_eq!(doc.screen_input.lfxc, 2);
        assert_eq!(doc.screen_input.ermin, 0.002);
        assert_eq!(doc.screen_input.nei, 20);
        assert_eq!(doc.active_cards, ["SCREEN"]);
        assert_eq!(doc.input_cards, ["SCREEN", "SCREEN", "SCREEN", "SCREEN"]);
        Ok(())
    }

    #[test]
    fn rejects_incomplete_screen_card_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SCREEN rfms
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("incomplete SCREEN card should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("SCREEN requires keyword and value"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_unknown_screen_keyword_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SCREEN unknown 1.0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("unknown SCREEN keyword should be rejected")?;
        ensure!(
            error
                .to_string()
                .contains("unrecognized SCREEN keyword \"unknown\""),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn extracts_xsph_handoff_controls() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CHBROADENING 1
SETEDGE
EPS0 -2.0
EGAP 1.25
CHWIDTH 0.75
RLPRINT
ICORE 3
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.xsph_handoff.core_hole_broadening, 1);
        assert_eq!(doc.xsph_handoff.core_state, 3);
        assert_eq!(doc.xsph_handoff.eps0, -2.0);
        assert_eq!(doc.xsph_handoff.egap, 1.25);
        assert_eq!(doc.xsph_handoff.core_hole_width, Some(0.75));
        assert!(doc.xsph_handoff.set_edge);
        assert!(doc.xsph_handoff.print_radial_wavefunctions);
        assert_eq!(
            doc.active_cards,
            [
                "CHBROADENING",
                "SETE",
                "EPS0",
                "EGAP",
                "CHWIDTH",
                "RLPR",
                "ICOR"
            ]
        );
        Ok(())
    }

    #[test]
    fn extracts_opcons_number_density_controls() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
OPCONS
NUMDENS 0 8.5
NUMDENS 2 4.25
PREPS
POTENTIALS
0 29 Cu0
1 29 Cu1
2 8 O
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert!(doc.opcons);
        assert!(doc.opcons_input.run_opcons);
        assert!(doc.opcons_input.print_eps);
        assert_eq!(doc.opcons_input.number_densities, vec![8.5, -1.0, 4.25]);
        assert_eq!(doc.active_cards, ["POTENTIALS", "OPCONS", "NUMD", "PREP"]);
        Ok(())
    }

    #[test]
    fn extracts_opcons_alias_controls_like_feff() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
OPCO
NUMD 0 8.5
PREP
POTENTIALS
0 29 Cu0
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert!(doc.opcons);
        assert!(doc.opcons_input.run_opcons);
        assert!(doc.opcons_input.print_eps);
        assert_eq!(doc.opcons_input.number_densities, vec![8.5, -1.0]);
        assert_eq!(doc.active_cards, ["POTENTIALS", "OPCONS", "NUMD", "PREP"]);
        assert_eq!(doc.input_cards, ["OPCONS", "NUMD", "PREP", "POTENTIALS"]);
        Ok(())
    }

    #[test]
    fn extracts_single_scattering_cards_and_scales_distance() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
RMULTIPLIER 2.0
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
OVERLAP 1
0 12 2.55266
SS 29 1 48 2.99
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.overlap_shells.len(), 2);
        assert_eq!(doc.overlap_shells[0].potential_index, 0);
        assert_eq!(doc.overlap_shells[0].neighbor_potential_index, 1);
        assert_eq!(doc.overlap_shells[0].count, 12);
        ensure!(
            (doc.overlap_shells[0].distance - 5.10532).abs() < 1.0e-12,
            "unexpected scaled OVERLAP distance: {}",
            doc.overlap_shells[0].distance
        );
        assert_eq!(doc.single_scattering_paths.len(), 1);
        let path = doc
            .single_scattering_paths
            .first()
            .context("missing SS path")?;
        assert_eq!(path.index, 29);
        assert_eq!(path.potential_index, 1);
        assert_eq!(path.degeneracy, 48.0);
        ensure!(
            (path.distance - 5.98).abs() < 1.0e-12,
            "unexpected scaled SS distance: {}",
            path.distance
        );
        Ok(())
    }

    #[test]
    fn rejects_single_scattering_cards_without_overlap() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTENTIALS
0 29 Cu0
1 29 Cu1
SS 29 1 48 2.99
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("SS without OVERLAP should be rejected")?;

        ensure!(
            error
                .to_string()
                .contains("SS cards require an OVERLAP card"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_single_scattering_cards_without_overlap_rows() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
SS 29 1 48 2.99
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("SS without OVERLAP rows should be rejected")?;

        ensure!(
            error
                .to_string()
                .contains("SS cards require OVERLAP shell rows"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_atoms_with_overlap_geometry() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
ATOMS
0 0 0 0 Cu0
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("ATOMS with OVERLAP should be rejected")?;

        ensure!(
            error.to_string().contains("cannot use ATOMS and OVERLAP"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn extracts_manual_overlap_factors() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
AFOLP 1.30
FOLP 1 1.2
FOLP 2 0.8
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.afolp, 1.30);
        assert_eq!(doc.overlap_factors.len(), 2);
        assert_eq!(doc.overlap_factors[0].potential_index, 1);
        assert_eq!(doc.overlap_factors[0].factor, 1.2);
        assert_eq!(doc.overlap_factors[1].potential_index, 2);
        assert_eq!(doc.overlap_factors[1].factor, 0.8);
        Ok(())
    }

    #[test]
    fn extracts_ionization_cards() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
ION 1 0.2
ION 2 -0.1
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        assert_eq!(doc.ionizations.len(), 2);
        assert_eq!(doc.ionizations[0].potential_index, 1);
        assert_eq!(doc.ionizations[0].value, 0.2);
        assert_eq!(doc.ionizations[1].potential_index, 2);
        assert_eq!(doc.ionizations[1].value, -0.1);
        Ok(())
    }

    #[test]
    fn extracts_debye_dynamical_matrix_options() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())?;
        let input = FeffInput::parse_str(
            &input_path,
            r#"
DEBYE 450 315 5 feff.dym 6 0 1
END
"#,
        )?;

        let doc = FeffDocument::from_input(&input)?;
        let debye = doc.debye.context("missing DEBYE options")?;
        ensure!(debye.idwopt == 5, "unexpected idwopt: {}", debye.idwopt);
        assert_eq!(debye.dym_file.as_deref(), Some("feff.dym"));
        assert_eq!(debye.dmdw_order, 6);
        assert_eq!(debye.dmdw_type, 0);
        assert_eq!(debye.dmdw_route, 1);
        let dym_input = doc.dym_input.context("missing DMDW auxiliary")?;
        assert_eq!(dym_input.output_name, "feff.dym");
        assert_eq!(dym_input.text, minimal_dym_text());
        Ok(())
    }

    #[test]
    fn rejects_dmdw_auxiliary_parent_output_paths() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let input_path = temp.path().join("feff.inp");
        let input = FeffInput::parse_str(&input_path, "DEBYE 450 315 5 ../force.dym\nEND\n")?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("DMDW parent path should be rejected")?;

        ensure!(
            error.to_string().contains("output directory"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_non_numeric_potential_atomic_numbers() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
POTENTIALS
0 XXX Te
END
"#,
        )?;

        let error = FeffDocument::from_input(&input)
            .err()
            .context("non-numeric POTENTIALS Z token should be rejected")?;

        ensure!(
            error.to_string().contains("XXX"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn generates_potentials_for_cif_without_potentials_card() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let cif_path = temp.path().join("two-site.cif");
        std::fs::write(
            &cif_path,
            r#"
data_two_site
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H 0.0 0.0 0.0
O 0.5 0.5 0.5
"#,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(
            &input_path,
            r#"
CIF two-site.cif
TARGET 2
EDGE K
XANES
END
"#,
        )?;

        let input = FeffInput::parse_file(&input_path)?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(doc.potentials.len(), 3);
        assert_eq!(doc.potentials[0].ipot, 0);
        assert_eq!(doc.potentials[0].z, Some(8));
        assert_eq!(doc.potentials[0].tag.as_deref(), Some("O"));
        assert_eq!(doc.potentials[0].xnatph, Some(0.01));
        assert_eq!(doc.potentials[1].ipot, 1);
        assert_eq!(doc.potentials[1].z, Some(1));
        assert_eq!(doc.potentials[1].tag.as_deref(), Some("H"));
        assert_eq!(doc.potentials[1].xnatph, Some(1.0));
        assert_eq!(doc.potentials[2].ipot, 2);
        assert_eq!(doc.potentials[2].z, Some(8));
        assert_eq!(doc.potentials[2].tag.as_deref(), Some("O"));
        assert_eq!(doc.potentials[2].xnatph, Some(1.0));
        Ok(())
    }

    #[test]
    fn cif_equivalence_two_generates_atomic_number_potentials() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let cif_path = temp.path().join("three-site.cif");
        std::fs::write(
            &cif_path,
            r#"
data_three_site
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H1 0.0 0.0 0.0
H2 0.25 0.0 0.0
O1 0.5 0.5 0.5
"#,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(
            &input_path,
            r#"
CIF three-site.cif
TARGET 3
EQUI 2
FMS 4.0
EDGE K
XANES
END
"#,
        )?;

        let input = FeffInput::parse_file(&input_path)?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(doc.cif_equivalence, 2);
        assert_eq!(doc.potentials.len(), 3);
        assert_eq!(doc.potentials[0].ipot, 0);
        assert_eq!(doc.potentials[0].z, Some(8));
        assert_eq!(doc.potentials[0].tag.as_deref(), Some("O"));
        assert_eq!(doc.potentials[1].ipot, 1);
        assert_eq!(doc.potentials[1].z, Some(1));
        assert_eq!(doc.potentials[1].xnatph, Some(2.0));
        assert_eq!(doc.potentials[2].ipot, 2);
        assert_eq!(doc.potentials[2].z, Some(8));
        assert!(doc.atoms.iter().any(|atom| atom.ipot == 1));
        assert!(!doc.atoms.iter().any(|atom| atom.ipot == 3));
        assert!(doc.active_cards.iter().any(|card| card == "EQUIVALENCE"));
        Ok(())
    }

    #[test]
    fn cif_equivalence_four_collapses_when_potential_limit_is_exceeded() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let cif_path = temp.path().join("many-site.cif");
        let mut cif = String::from(
            r#"
data_many_sites
_cell_length_a 8.0
_cell_length_b 8.0
_cell_length_c 8.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
"#,
        );
        for index in 0..32 {
            let symbol = if index % 2 == 0 { "H" } else { "O" };
            let x = index as f64 / 64.0;
            cif.push_str(&format!("{symbol}{index} {x:.6} 0.0 0.0\n"));
        }
        std::fs::write(&cif_path, cif)?;

        let input_path = temp.path().join("feff.inp");
        std::fs::write(
            &input_path,
            r#"
CIF many-site.cif
TARGET 2
EQUIVALENCE 4
FMS 4.0
EDGE K
XANES
END
"#,
        )?;

        let input = FeffInput::parse_file(&input_path)?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(doc.cif_equivalence, 4);
        assert_eq!(doc.potentials.len(), 3);
        assert_eq!(doc.potentials[0].z, Some(8));
        assert_eq!(doc.potentials[1].z, Some(1));
        assert_eq!(doc.potentials[1].xnatph, Some(16.0));
        assert_eq!(doc.potentials[2].z, Some(8));
        assert_eq!(doc.potentials[2].xnatph, Some(16.0));
        assert!(doc.atoms.iter().any(|atom| atom.ipot == 1));
        assert!(!doc.atoms.iter().any(|atom| atom.ipot == 3));
        Ok(())
    }

    #[test]
    fn generates_atoms_for_cif_without_atoms_card() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let cif_path = temp.path().join("two-site.cif");
        std::fs::write(
            &cif_path,
            r#"
data_two_site
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H 0.0 0.0 0.0
O 0.5 0.5 0.5
"#,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(
            &input_path,
            r#"
CIF two-site.cif
TARGET 2
FMS 4.0
RMULTIPLIER 2.0
EDGE K
XANES
END
"#,
        )?;

        let input = FeffInput::parse_file(&input_path)?;
        let doc = FeffDocument::from_input(&input)?;

        assert!(!doc.atoms.is_empty());
        assert_eq!(doc.atoms[0].ipot, 0);
        assert_eq!(
            (
                doc.atoms[0].x.round() as i32,
                doc.atoms[0].y.round() as i32,
                doc.atoms[0].z.round() as i32,
            ),
            (0, 0, 0)
        );
        assert!(
            doc.atoms
                .iter()
                .any(|atom| atom.ipot == 1 && (atom.x.abs() - 4.0).abs() < 1.0e-9)
        );
        Ok(())
    }

    fn minimal_dym_text() -> &'static str {
        concat!(
            "    1\n",
            "    1\n",
            "   29\n",
            "   63.546000\n",
            "    0.00000000    0.00000000    0.00000000\n",
            "    1    1\n",
            "  1.000000E+00  0.000000E+00  0.000000E+00\n",
            "  0.000000E+00  1.000000E+00  0.000000E+00\n",
            "  0.000000E+00  0.000000E+00  1.000000E+00\n",
        )
    }
}
