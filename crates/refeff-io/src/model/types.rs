use std::path::PathBuf;

use crate::control_input::{BandInput, FullSpectrumInput, OpconsInput, ReciprocalInput};
use crate::global_input::CfAverage;
use crate::screen_input::ScreenInput;
use crate::sfconv_input::SfconvInput;
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
    /// Whether `rixs.inp` should read cached pole data.
    pub read_poles: bool,
    /// Whether the RIXS stage should derive spectra from cached `rixsET.dat`.
    pub skip_calc: bool,
    /// Many-body convolution switch, enabled by a `VAL` edge.
    pub mbconv: bool,
    /// Whether `rixs.inp` should read cached self-energy data.
    pub read_sigma: bool,
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
            read_poles: true,
            skip_calc: false,
            mbconv: false,
            read_sigma: false,
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
    /// Debye-Waller calculation mode after FEFF-compatible normalization.
    pub idwopt: i32,
    /// Selector supplied on the `DEBYE` card before normalization.
    pub requested_idwopt: i32,
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
