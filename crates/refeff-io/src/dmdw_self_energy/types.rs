use ndarray::Array1;
use num_complex::Complex64;

/// Parsed FEFF `dmdw_a2f.info` pole-weight diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwA2fInfoData {
    /// FEFF DMDW run type, normally `2` for this sidecar.
    pub calculation_type: i32,
    /// FEFF run-type 2 displacement option.
    pub displacement_option: i32,
    /// Requested Lanczos recursion order.
    pub lanczos_order: usize,
    /// FEFF `w_pole / 6.28` diagnostic values in THz.
    pub lanczos_frequency_thz: Array1<f64>,
    /// FEFF `wil` projected-DOS weights.
    pub lanczos_weight: Array1<f64>,
    /// FEFF projected-DOS normalization, `norm`.
    pub normalization: f64,
    /// Pole-weight `a2f` energies in eV.
    pub pole_energy_ev: Array1<f64>,
    /// Pole-weight `a2f` values.
    pub pole_weight: Array1<f64>,
    /// FEFF `lambda` mass-enhancement diagnostic.
    pub mass_enhancement: f64,
    /// FEFF `w0` characteristic phonon energy in eV.
    pub characteristic_energy_ev: f64,
}

/// Parsed FEFF `dmdw_Egrid.info` energy-window metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmdwEnergyGridInfo {
    /// Lowest printed spectral energy in meV.
    pub low_energy_mev: f64,
    /// Highest printed spectral energy in meV.
    pub high_energy_mev: f64,
    /// Spectral energy step in meV.
    pub step_mev: f64,
    /// Characteristic phonon energy `w0` in meV.
    pub characteristic_energy_mev: f64,
    /// Requested electron energy `E_k` in meV.
    pub electron_energy_mev: f64,
    /// Nearest grid energy selected for `E_k`, in meV.
    pub selected_energy_mev: f64,
}

/// Parsed FEFF `dmdw_spectral.info` spectral-function diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmdwSpectralInfoData {
    /// FEFF `Gamma_k` broadening in units of `w0`.
    pub gamma: f64,
    /// FEFF `epk = E_k - ReSE(E_k)` diagnostic in units of `w0`.
    pub effective_electron_energy: f64,
    /// FEFF central-difference cumulant derivative, `atot`.
    pub total_cumulant_derivative: Complex64,
    /// FEFF quasiparticle renormalization, `Zk`.
    pub quasiparticle_weight: Complex64,
}

/// Parsed FEFF `dmdw_reSE_a2F.dat` or `dmdw_imSE_a2F.dat` table.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwSelfEnergyDatData {
    /// Header/comment lines preserved before and around numeric rows.
    pub header_lines: Vec<String>,
    /// Self-energy sample energy in eV.
    pub energy_ev: Array1<f64>,
    /// Real or imaginary self-energy value in eV.
    pub value_ev: Array1<f64>,
}

/// Parsed FEFF `dmdw_Akw.dat` spectral-function table.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwAkwDatData {
    /// FEFF spectral-function normalization from the `# norm =` header.
    pub normalization: Option<f64>,
    /// Spectral-function energy in meV.
    pub energy_mev: Array1<f64>,
    /// Spectral-function magnitude.
    pub magnitude: Array1<f64>,
    /// Spectral-function phase in radians.
    pub phase: Array1<f64>,
    /// Real spectral-function component.
    pub real: Array1<f64>,
    /// Imaginary spectral-function component.
    pub imaginary: Array1<f64>,
}

impl DmdwSelfEnergyDatData {
    /// Number of self-energy samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

impl DmdwA2fInfoData {
    /// Number of Lanczos poles represented in the diagnostic table.
    #[must_use]
    pub fn pole_count(&self) -> usize {
        self.lanczos_frequency_thz.len()
    }
}

impl DmdwAkwDatData {
    /// Number of spectral-function samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_mev.len()
    }
}
