//! Public data structures for FEFF `dmdw.out` reports.

/// Parsed contents of FEFF `dmdw.out`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutData {
    /// File header, or `None` for FEFF runs that produced an empty report.
    pub header: Option<DmdwOutHeader>,
    /// Whether the FEFF type-2 self-energy banner was present after the header.
    ///
    /// FEFF spells the word as `Enchancement`; the parser and writer preserve
    /// that historical output text for compatibility.
    pub mass_enhancement_header: bool,
    /// DMDW report sections in file order.
    pub sections: Vec<DmdwOutSection>,
}

impl DmdwOutData {
    /// Number of report sections.
    #[must_use]
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }
}

/// Header values written before the DMDW report sections.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutHeader {
    /// Lanczos recursion order used to build the pole expansion.
    pub lanczos_recursion_order: usize,
    /// Temperature declaration from the file header.
    pub temperature: DmdwOutTemperature,
    /// Dynamical-matrix file named in the DMDW run.
    pub dynamical_matrix_file: String,
}

/// Temperature declaration from the `dmdw.out` header.
#[derive(Debug, Clone, PartialEq)]
pub enum DmdwOutTemperature {
    /// Single-temperature DMDW run.
    Single(f64),
    /// Multi-temperature run whose temperatures are printed in result tables.
    ListedBelow,
}

/// One path, atom, or total-PDOS block from `dmdw.out`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutSection {
    /// Subject of the report block.
    pub subject: DmdwOutSubject,
    /// Projected DOS pole positions and weights.
    pub pdos_poles: Vec<DmdwOutPole>,
    /// Single-pole Einstein summary.
    pub einstein: Option<DmdwOutEinstein>,
    /// Moment-derived Einstein summaries.
    pub moments: Vec<DmdwOutMoment>,
    /// Reduced mass for path Debye-Waller output.
    pub reduced_mass_amu: Option<f64>,
    /// Path length for path Debye-Waller output.
    pub path_length_angstrom: Option<f64>,
    /// Single-temperature sigma2 value in `1e-3 Ang^2`.
    pub sigma2_1e_minus_3_angstrom2: Option<f64>,
    /// Multi-temperature sigma2 values in `1e-3 Ang^2`.
    pub sigma2_by_temperature: Vec<DmdwOutTemperatureValue>,
    /// Single-temperature vibrational free energy in eV.
    pub vibrational_free_energy_ev: Option<f64>,
    /// Multi-temperature vibrational free energy values in eV.
    pub vibrational_free_energy_by_temperature: Vec<DmdwOutTemperatureValue>,
    /// Single-temperature mean-square displacement in `1e-3 Ang^2`.
    pub u2_1e_minus_3_angstrom2: Option<f64>,
    /// Multi-temperature mean-square displacement values in `1e-3 Ang^2`.
    pub u2_by_temperature: Vec<DmdwOutTemperatureValue>,
    /// Whether the projected-DOS component completion line was written.
    pub projected_dos_component_computed: bool,
}

impl DmdwOutSection {
    /// Build an empty report section for the given subject.
    #[must_use]
    pub fn new(subject: DmdwOutSubject) -> Self {
        Self {
            subject,
            pdos_poles: Vec::new(),
            einstein: None,
            moments: Vec::new(),
            reduced_mass_amu: None,
            path_length_angstrom: None,
            sigma2_1e_minus_3_angstrom2: None,
            sigma2_by_temperature: Vec::new(),
            vibrational_free_energy_ev: None,
            vibrational_free_energy_by_temperature: Vec::new(),
            u2_1e_minus_3_angstrom2: None,
            u2_by_temperature: Vec::new(),
            projected_dos_component_computed: false,
        }
    }
}

/// Subject of a DMDW output block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DmdwOutSubject {
    /// Run type 0 path output.
    PathIndices(Vec<usize>),
    /// Atom-index output for atom-local run types, with optional direction.
    AtomIndex {
        /// FEFF atom indices printed in the block header.
        indices: Vec<usize>,
        /// Optional perturbation direction label.
        direction: Option<String>,
    },
    /// Total projected-density-of-states output.
    TotalPdos,
    /// Total vibrational free-energy output for selected paths.
    TotalVfe,
}

/// One projected-density-of-states pole.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutPole {
    /// Pole frequency in THz.
    pub frequency_thz: f64,
    /// Pole weight.
    pub weight: f64,
}

/// Single-pole Einstein-frequency summary.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutEinstein {
    /// Einstein frequency in THz.
    pub frequency_thz: f64,
    /// Associated Einstein temperature in K.
    pub temperature_kelvin: f64,
    /// Effective force constant in N/m.
    pub effective_force_constant_n_per_m: f64,
}

/// Moment-derived Einstein-frequency summary row.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutMoment {
    /// Moment order `n`.
    pub order: i32,
    /// Moment value in `THz^n`.
    pub moment_thz_power_n: f64,
    /// Derived frequency in THz, absent for the `n = 0` placeholder row.
    pub frequency_thz: Option<f64>,
    /// Derived temperature in K, absent for the `n = 0` placeholder row.
    pub temperature_kelvin: Option<f64>,
    /// Derived effective force constant in N/m.
    pub effective_force_constant_n_per_m: Option<f64>,
}

/// A temperature-dependent scalar result row.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwOutTemperatureValue {
    /// Temperature in K.
    pub temperature_kelvin: f64,
    /// Scalar result value in the units of the table that owns the row.
    pub value: f64,
}
