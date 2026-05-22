use ndarray::{Array1, Array2};
use num_complex::Complex64;

pub const FEFF_BIN_BOHR: f64 = 0.529_177_249;
/// FEFF v03 `feff.bin` default PAD width.
pub const FEFF_BIN_DEFAULT_PAD_WIDTH: usize = 8;

/// Potential label and atomic number entry from the `#@` record.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffBinPotential {
    /// FEFF six-character potential label.
    pub label: String,
    /// Atomic number for this potential.
    pub atomic_number: usize,
}

/// One path block from a FEFF v03 `feff.bin` file.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffBinPath {
    /// FEFF path index, `ipath`.
    pub index: usize,
    /// Path degeneracy.
    pub degeneracy: f64,
    /// Effective half path length in bohr. The text file stores Angstrom.
    pub effective_half_path_length_bohr: f64,
    /// Path importance criterion.
    pub criterion: f64,
    /// Potential index for each leg.
    pub potential_indices: Array1<usize>,
    /// Cartesian leg positions as `(leg, xyz)` in bohr.
    pub positions: Array2<f64>,
    /// First Euler angle for each leg.
    pub beta: Array1<f64>,
    /// Second Euler angle for each leg.
    pub eta: Array1<f64>,
    /// Leg distances in bohr.
    pub leg_distances: Array1<f64>,
    /// FEFF amplitude array, `amff`.
    pub amplitude: Array1<f64>,
    /// FEFF phase array, `phff`.
    pub phase: Array1<f64>,
}

impl FeffBinPath {
    /// Number of legs in this path.
    #[must_use]
    pub fn leg_count(&self) -> usize {
        self.potential_indices.len()
    }
}

/// FEFF v03 `feff.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffBinData {
    /// Version suffix written after `#_feff.bin v03:`.
    pub version: String,
    /// PAD field width, `mpadx`.
    pub pad_width: usize,
    /// Core-hole index, `ihole`.
    pub ihole: i32,
    /// GENFMT matrix order, `iorder`.
    pub order: i32,
    /// Initial-state angular momentum, `ilinit`.
    pub initial_angular_momentum: i32,
    /// Average Norman radius, `rnrmav`.
    pub average_norman_radius: f64,
    /// Fermi level, `xmu`.
    pub fermi_level: f64,
    /// Edge energy.
    pub edge_energy: f64,
    /// Potential table for FEFF indices `0:npot`.
    pub potentials: Vec<FeffBinPotential>,
    /// Central atom phase shift, `phc`.
    pub central_phase_shift: Array1<Complex64>,
    /// Complex momentum, `ck`.
    pub complex_momentum: Array1<Complex64>,
    /// Real momentum, `xk`.
    pub real_momentum: Array1<f64>,
    /// Path records.
    pub paths: Vec<FeffBinPath>,
    /// Raw parsed `feff.bin` text for exact re-emission when the typed content
    /// is unchanged.
    pub raw_text: Option<String>,
}

impl FeffBinData {
    /// Number of energy points, `ne`.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.central_phase_shift.len()
    }

    /// Number of potential entries represented by FEFF indices `0:npot`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.potentials.len()
    }
}
