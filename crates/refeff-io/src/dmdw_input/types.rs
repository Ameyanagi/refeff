/// Parsed contents of a FEFF `dmdw.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub enum DmdwInput {
    /// FEFF sentinel for no standalone DMDW calculation.
    Disabled,
    /// Enabled standalone DMDW calculation.
    Enabled(DmdwCalculation),
}

/// Enabled DMDW calculation settings.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwCalculation {
    /// Module run flag written by FEFF.
    pub run: i32,
    /// Lanczos recursion order.
    pub order: i32,
    /// Temperature selector from the third data line.
    pub temperature_flag: i32,
    /// Sample temperature, or the lower grid bound for multi-temperature runs.
    pub temperature: f64,
    /// Upper temperature grid bound for multi-temperature runs.
    pub temperature_max: Option<f64>,
    /// Dynamical matrix calculation type selector.
    pub calculation_type: i32,
    /// Self-energy options for calculation type 2.
    pub self_energy_options: Option<DmdwSelfEnergyOptions>,
    /// Projected-density-of-states options for calculation type 5.
    pub pdos_options: Option<DmdwPdosOptions>,
    /// Dynamical matrix filename.
    pub dym_file: String,
    /// Number of path rows.
    pub path_count: usize,
    /// Selected path rows.
    pub paths: Vec<DmdwPath>,
}

/// FEFF DMDW self-energy options for calculation type 2.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwSelfEnergyOptions {
    /// Displacement selector, FEFF `disp_opt`.
    pub displacement_option: i32,
    /// Electron-energy unit selector, FEFF `E_k_opt`.
    pub energy_option: i32,
    /// Electron energy value, FEFF `E_k`.
    pub electron_energy: f64,
    /// ABINIT PDS file name.
    pub pds_file: String,
    /// Eliashberg coupling `a2f` file name.
    pub a2f_file: String,
}

/// FEFF DMDW projected-density-of-states output options.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwPdosOptions {
    /// PDOS output format selector from the DMDW type line.
    pub format: i32,
    /// Whether FEFF should write per-component PDOS sidecar files.
    pub write_partial: bool,
    /// Whether rectangular PDOS output should drop each bin to zero.
    pub drop_left_edges: bool,
    /// Gaussian PDOS broadening in THz.
    pub gaussian_broadening_thz: f64,
    /// Gaussian PDOS frequency-grid resolution in THz.
    pub gaussian_resolution_thz: f64,
}

impl Default for DmdwPdosOptions {
    fn default() -> Self {
        Self {
            format: 0,
            write_partial: false,
            drop_left_edges: false,
            gaussian_broadening_thz: 0.500,
            gaussian_resolution_thz: 0.001,
        }
    }
}

/// One path row from `dmdw.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwPath {
    /// Number of legs in the path row.
    pub leg_count: i32,
    /// Absorber selector written by FEFF.
    pub absorber_selector: i32,
    /// Potential selectors before the distance field.
    pub potentials: Vec<i32>,
    /// Maximum path distance field as written in `dmdw.inp`.
    ///
    /// FEFF's DMDW reader multiplies this value by its Angstrom-to-Bohr
    /// conversion before descriptor expansion.
    pub max_distance: f64,
}
