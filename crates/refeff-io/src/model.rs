//! Typed extraction of common FEFF input cards.
//!
//! This layer intentionally starts with stable structural cards and grows as
//! each FEFF module is ported. Unknown or module-specific cards remain
//! available in [`crate::FeffInput`] so no information is lost.

use std::path::PathBuf;

use crate::error::{IoError, Result};
use crate::input::{FeffInput, FeffLine, LineKind};

/// FEFF input projected into typed structures used by the Rust modules.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffDocument {
    /// Root input file.
    pub source: PathBuf,
    /// Active FEFF card names in FEFF token order, using canonical output names
    /// from `itoken_reverse`.
    pub active_cards: Vec<String>,
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
    /// Explicit `EGRID` switch used by `xsph`.
    pub i_grid: i32,
    /// Electronic temperature from `TEMP`, in eV.
    pub electronic_temperature: f64,
    /// Self-energy exchange selector for finite-temperature calculations.
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
    /// Real-self-energy switch from `RSIGMA`.
    pub lreal: i32,
    /// Multiple-pole self-energy model selector from `MPSE`/`PLASMON`.
    pub i_plsmn: i32,
    /// Number of poles for MPSE.
    pub n_poles: i32,
    /// Whether `OPCONS` should build optical constants from the database.
    pub opcons: bool,
    /// Whether spectral-function convolution is requested.
    pub sfconv: bool,
    /// Unfreeze f-electrons in the potential stage.
    pub unfreezef: bool,
    /// Atomic-configuration source selector for `pot.inp`.
    pub config_type: i32,
    /// Raw `CONFIG card` payload rows copied into `config.inp`.
    pub config_records: Vec<String>,
    /// Whether ionicity warnings are requested.
    pub warn_ion: bool,
    /// FEFF core-hole treatment selector (`nohole`) from `NOHOLE`/`COREHOLE`.
    pub nohole: i32,
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
    /// Debye-Waller settings from `DEBYE`, when present.
    pub debye: Option<Debye>,
    /// Path expansion radius from `RPATH`/`RMAX`, when present.
    pub rpath: Option<f64>,
    /// Maximum path leg count from `NLEG`.
    pub nleg: Option<i32>,
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nrixs {
    /// Number of q vectors.
    pub nq: i32,
    /// Whether FEFF should average over q directions.
    pub qaverage: bool,
    /// Momentum-transfer vector.
    pub qvec: [f64; 3],
    /// q-vector norm.
    pub qnorm: f64,
    /// Decomposition limit from `LDEC`.
    pub ldecmx: i32,
    /// Angular momentum limit from `LJMAX`.
    pub lj: i32,
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

/// User-requested dynamic allocation caps from the `DIMS` card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionLimits {
    /// Maximum FMS cluster size requested by the input.
    pub nclusx: i32,
    /// Maximum angular momentum requested by the input.
    pub lx: i32,
}

/// One row of the FEFF `POTENTIALS` table.
#[derive(Debug, Clone, PartialEq)]
pub struct Potential {
    /// FEFF potential index.
    pub ipot: i32,
    /// Parsed atomic number when the field is numeric.
    pub z: Option<i32>,
    /// Original Z token, preserved for `HIGHZ` placeholders such as `XXX`.
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
        let titles = parse_titles(input)?;
        let edge = parse_edge(input)?;
        let (hole, hole_s02) = parse_hole(input)?;
        let s02 = parse_scalar_card(input, "S02")?.or(hole_s02);
        let corrections = parse_corrections(input)?;
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
        let reciprocal = input.card("RECIPROCAL").is_some();
        let i_grid = i32::from(input.card("EGRID").is_some());
        let (electronic_temperature, iscfxc) = parse_temp(input)?;
        let rgrid = parse_scalar_card(input, "RGRID")?.unwrap_or(0.05);
        let (critcw, critpw) = parse_criteria(input)?;
        let (pcritk, pcrith) = parse_pcriteria(input)?;
        let lreal = i32::from(input.card("RSIGMA").is_some());
        let (i_plsmn, n_poles) = parse_mpse(input)?;
        let opcons = input.card("OPCONS").is_some();
        let sfconv = input.card("SFCONV").is_some();
        let unfreezef = input.card("UNFREEZEF").is_some();
        let config_type = parse_config_type(input)?;
        let config_records = parse_config_records(input)?;
        let warn_ion = input.card("WARNION").is_some();
        let nohole = parse_nohole(input)?;
        let absolute = input.card("ABSOLUTE").is_some();
        let mut fms = parse_fms(input)?;
        let crpa = parse_crpa(input)?;
        let compton = parse_compton(input)?;
        let hubbard = parse_hubbard(input)?;
        let eels = parse_eels(input)?;
        let rixs = parse_rixs(input)?;
        let nrixs = parse_nrixs(input)?;
        let nohole = if compton.do_compton || compton.do_rhozzp {
            0
        } else {
            nohole
        };
        let debye = parse_debye(input)?;
        let mut rpath = parse_rpath(input)?;
        let nleg = parse_nleg(input)?;
        let r_multiplier = parse_scalar_card(input, "RMULTIPLIER")?.unwrap_or(1.0);
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
        }
        let dims = parse_dims(input)?;
        let ldos = parse_ldos(input)?;
        let interstitial = parse_interstitial(input)?;
        let afolp = parse_afolp(input)?;
        let potentials = parse_potentials(input)?;
        let atoms = parse_atoms(input)?
            .into_iter()
            .map(|mut atom| {
                atom.x *= r_multiplier;
                atom.y *= r_multiplier;
                atom.z *= r_multiplier;
                atom.distance = atom.distance.map(|distance| distance * r_multiplier);
                atom
            })
            .collect();

        Ok(Self {
            source: input.source.clone(),
            active_cards,
            titles,
            edge,
            hole,
            s02,
            corrections,
            control,
            print,
            scf,
            exchange,
            exafs,
            spectrum_grid,
            reciprocal,
            i_grid,
            electronic_temperature,
            iscfxc,
            rgrid,
            critcw,
            critpw,
            pcritk,
            pcrith,
            lreal,
            i_plsmn,
            n_poles,
            opcons,
            sfconv,
            unfreezef,
            config_type,
            config_records,
            warn_ion,
            nohole,
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
            debye,
            rpath,
            nleg,
            r_multiplier,
            dims,
            ldos,
            interstitial,
            afolp,
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
            && keyword == "TITLE"
        {
            titles.push(raw_args.clone());
        }
    }
    Ok(titles)
}

fn parse_edge(input: &FeffInput) -> Result<Option<Edge>> {
    let Some(line) = input.card("EDGE") else {
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
    let Some(line) = input.card("HOLE") else {
        return Ok((None, None));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_i32(line, args.first())?,
        parse_optional_f64(line, args.get(1))?,
    ))
}

fn parse_scalar_card(input: &FeffInput, keyword: &str) -> Result<Option<f64>> {
    let Some(line) = input.card(keyword) else {
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

fn parse_corrections(input: &FeffInput) -> Result<[f64; 2]> {
    let Some(line) = input.card("CORRECTIONS") else {
        return Ok([0.0, 0.0]);
    };
    let args = card_args(line)?;
    Ok([
        parse_optional_f64(line, args.first())?.unwrap_or(0.0),
        parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
    ])
}

fn parse_config_type(input: &FeffInput) -> Result<i32> {
    let Some(line) = input.card("CONFIG") else {
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

fn parse_i32_6(input: &FeffInput, keyword: &str) -> Result<Option<[i32; 6]>> {
    let Some(line) = input.card(keyword) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let mut values = [0_i32; 6];
    for (idx, slot) in values.iter_mut().enumerate() {
        let Some(value) = args.get(idx) else {
            return Err(parse_error(
                line,
                format!("{keyword} requires 6 integer values"),
            ));
        };
        *slot = parse_i32(line, value)?;
    }
    Ok(Some(values))
}

fn parse_scf(input: &FeffInput) -> Result<Option<Scf>> {
    let Some(line) = input.card("SCF") else {
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
    let Some(line) = input.card("EXCHANGE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(ixc) = args.first() else {
        return Err(parse_error(line, "EXCHANGE requires an ixc value"));
    };

    Ok(Some(Exchange {
        ixc: parse_i32(line, ixc)?,
        vr0: parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
        vi0: parse_optional_f64(line, args.get(2))?.unwrap_or(0.0),
        ixc0: parse_optional_i32(line, args.get(3))?,
    }))
}

fn parse_exafs(input: &FeffInput) -> Result<Option<Exafs>> {
    let Some(line) = input.card("EXAFS") else {
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

    if let Some(line) = input
        .card("XANES")
        .or_else(|| input.card("DANES"))
        .or_else(|| input.card("ELNES"))
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
        if input.card("XANES").is_some() || input.card("ELNES").is_some() {
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
    } else if let Some(line) = input.card("XES") {
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
    } else if let Some(line) = input.card("FPRIME") {
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
        if grid.xkstep < grid.xkmax {
            grid.xkstep = grid.xkmax;
        }
    } else if let Some(line) = input.card("EXAFS").or_else(|| input.card("EXELFS")) {
        let args = card_args(line)?;
        if let Some(value) = args.first() {
            grid.xkmax = parse_f64(line, value)?;
        }
    }

    Ok(grid)
}

fn parse_temp(input: &FeffInput) -> Result<(f64, i32)> {
    let Some(line) = input.card("TEMP") else {
        return Ok((0.0, 11));
    };
    let args = card_args(line)?;
    let temperature = parse_optional_f64(line, args.first())?.unwrap_or(0.0);
    let iscfxc = parse_optional_i32(line, args.get(1))?.unwrap_or(11);
    Ok((temperature, iscfxc))
}

fn parse_criteria(input: &FeffInput) -> Result<(f64, f64)> {
    let Some(line) = input.card("CRITERIA").or_else(|| input.card("CRIT")) else {
        return Ok((4.0, 2.5));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_f64(line, args.first())?.unwrap_or(4.0),
        parse_optional_f64(line, args.get(1))?.unwrap_or(2.5),
    ))
}

fn parse_pcriteria(input: &FeffInput) -> Result<(f64, f64)> {
    let Some(line) = input.card("PCRITERIA").or_else(|| input.card("PCRIT")) else {
        return Ok((0.0, 0.0));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_f64(line, args.first())?.unwrap_or(0.0),
        parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
    ))
}

fn parse_mpse(input: &FeffInput) -> Result<(i32, i32)> {
    let Some(line) = input
        .card("MPSE")
        .or_else(|| input.card("PLASMON"))
        .or_else(|| input.card("PLAS"))
    else {
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
    if input.card("COMPTON").is_some() || input.card("DENSITY").is_some() {
        5
    } else if input.card("FPRIME").is_some() {
        4
    } else if input.card("DANES").is_some() {
        3
    } else if input.card("XES").is_some() {
        2
    } else if input.card("XANES").is_some()
        || input.card("ELNES").is_some()
        || input.card("NRIXS").is_some()
    {
        1
    } else {
        0
    }
}

fn parse_ipol(input: &FeffInput) -> i32 {
    if input.card("XMCD").is_some() || input.card("XNCD").is_some() {
        2
    } else if input.card("POLARIZATION").is_some() {
        1
    } else {
        0
    }
}

fn parse_multipole(input: &FeffInput) -> Result<(i32, i32)> {
    let Some(line) = input.card("MULTIPOLE").or_else(|| input.card("MULTIPOLES")) else {
        return Ok((0, 0));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_i32(line, args.first())?.unwrap_or(0),
        parse_optional_i32(line, args.get(1))?.unwrap_or(0),
    ))
}

fn parse_polarization_vector(input: &FeffInput) -> Result<[f64; 3]> {
    let Some(line) = input.card("POLARIZATION") else {
        return Ok([0.0; 3]);
    };
    let args = card_args(line)?;
    Ok([
        parse_optional_f64(line, args.first())?.unwrap_or(0.0),
        parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
        parse_optional_f64(line, args.get(2))?.unwrap_or(0.0),
    ])
}

fn parse_ellipticity(input: &FeffInput) -> Result<(f64, [f64; 3])> {
    let Some(line) = input.card("ELLIPTICITY") else {
        return Ok((0.0, [0.0; 3]));
    };
    let args = card_args(line)?;
    Ok((
        parse_optional_f64(line, args.first())?.unwrap_or(0.0),
        [
            parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
            parse_optional_f64(line, args.get(2))?.unwrap_or(0.0),
            parse_optional_f64(line, args.get(3))?.unwrap_or(0.0),
        ],
    ))
}

fn parse_spin(input: &FeffInput) -> Result<(i32, [f64; 3])> {
    let Some(line) = input.card("SPIN") else {
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
    if let Some(line) = input.card("COREHOLE") {
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

    if let Some(line) = input.card("NOHOLE") {
        let args = card_args(line)?;
        return parse_optional_i32(line, args.first()).map(|value| value.unwrap_or(0));
    }

    Ok(-1)
}

fn parse_fms(input: &FeffInput) -> Result<Option<Fms>> {
    let Some(line) = input.card("FMS") else {
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
    let Some(line) = input.card("CRPA") else {
        return Ok(Crpa::default());
    };
    let args = card_args(line)?;
    Ok(Crpa {
        enabled: true,
        l: parse_optional_i32(line, args.first())?.unwrap_or(3),
        rcut: parse_optional_f64(line, args.get(1))?.unwrap_or(1.600_000_023_841_858),
    })
}

fn parse_compton(input: &FeffInput) -> Result<Compton> {
    let mut compton = Compton::default();

    if let Some(line) = input.card("COMPTON") {
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

    compton.do_rhozzp = input.card("RHOZZP").is_some();

    if let Some(line) = input.card("CGRID") {
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
    let Some(line) = input.card("HUBBARD") else {
        return Ok(Hubbard::default());
    };
    let args = card_args(line)?;
    Ok(Hubbard {
        i_hubbard: 2,
        mldos_hubb: 2,
        u: parse_optional_f64(line, args.first())?.unwrap_or(0.0),
        j: parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
        fermi_shift: parse_optional_f64(line, args.get(2))?.unwrap_or(0.0),
        l: parse_optional_i32(line, args.get(3))?.unwrap_or(0),
    })
}

fn parse_eels(input: &FeffInput) -> Result<Eels> {
    let section = if input.card("ELNES").is_some() {
        "ELNES"
    } else if input.card("EXELFS").is_some() {
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

    if let Some(line) = input.card("MAGIC") {
        let args = card_args(line)?;
        eels.magic = 1;
        eels.magic_energy = parse_optional_f64(line, args.first())?.unwrap_or(0.0);
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

fn parse_nrixs(input: &FeffInput) -> Result<Option<Nrixs>> {
    let Some(line) = input.card("NRIXS") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let raw_nq = parse_optional_i32(line, args.first())?.unwrap_or(1);
    let qaverage = raw_nq < 0;
    let nq = raw_nq.abs().max(1);
    let qvec = if qaverage {
        let qz = parse_optional_f64(line, args.get(1))?.unwrap_or(0.0);
        [0.0, 0.0, qz]
    } else {
        [
            parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
            parse_optional_f64(line, args.get(2))?.unwrap_or(0.0),
            parse_optional_f64(line, args.get(3))?.unwrap_or(0.0),
        ]
    };
    let qnorm = qvec[0].hypot(qvec[1]).hypot(qvec[2]);
    let ldecmx = match parse_scalar_card(input, "LDEC")? {
        Some(value) => value,
        None => parse_scalar_card(input, "LDECMX")?.unwrap_or(-1.0),
    };
    Ok(Some(Nrixs {
        nq,
        qaverage,
        qvec,
        qnorm,
        ldecmx: ldecmx as i32,
        lj: parse_scalar_card(input, "LJMAX")?.unwrap_or(0.0) as i32,
    }))
}

fn parse_rixs(input: &FeffInput) -> Result<Rixs> {
    let mut rixs = Rixs::default();

    if let Some(line) = input.card("EDGE") {
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

    if let Some(line) = input.card("RIXS") {
        let args = card_args(line)?;
        rixs.run = true;
        rixs.gamma_exp[0] = parse_optional_f64(line, args.first())?;
        rixs.gamma_exp[1] = parse_optional_f64(line, args.get(1))?;
        rixs.xmu = parse_optional_f64(line, args.get(2))?;
    }

    Ok(rixs)
}

fn parse_debye(input: &FeffInput) -> Result<Option<Debye>> {
    let Some(line) = input.card("DEBYE") else {
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
            .cloned()
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

fn parse_rpath(input: &FeffInput) -> Result<Option<f64>> {
    let Some(line) = input.card("RPATH").or_else(|| input.card("RMAX")) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(radius) = args.first() else {
        return Ok(Some(0.0));
    };
    Ok(Some(parse_f64(line, radius)?))
}

fn parse_nleg(input: &FeffInput) -> Result<Option<i32>> {
    let Some(line) = input.card("NLEG") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Ok(Some(7));
    };
    Ok(Some(parse_i32(line, value)?))
}

fn parse_dims(input: &FeffInput) -> Result<Option<DimensionLimits>> {
    let Some(line) = input.card("DIMS") else {
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
    let Some(line) = input.card("LDOS") else {
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
    let Some(line) = input.card("INTERSTITIAL") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    Ok(Some(Interstitial {
        mode: parse_optional_i32(line, args.first())?.unwrap_or(0),
        volume_scale: parse_optional_f64(line, args.get(1))?.unwrap_or(0.0),
    }))
}

fn parse_afolp(input: &FeffInput) -> Result<f64> {
    let Some(line) = input.card("AFOLP") else {
        return Ok(1.15);
    };
    let args = card_args(line)?;
    parse_optional_f64(line, args.first()).map(|value| value.unwrap_or(1.15))
}

fn parse_potentials(input: &FeffInput) -> Result<Vec<Potential>> {
    input
        .section_rows("POTENTIALS")
        .map(|line| {
            let fields = section_fields_before_star(line)?;
            if fields.len() < 2 {
                return Err(parse_error(line, "POTENTIALS rows require ipot and Z"));
            }

            Ok(Potential {
                ipot: parse_i32(line, fields[0])?,
                z: parse_i32(line, fields[1]).ok(),
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
        Ok(())
    }

    #[test]
    fn extracts_debye_dynamical_matrix_options() -> anyhow::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
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
        Ok(())
    }
}
