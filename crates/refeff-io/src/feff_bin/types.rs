use ndarray::{Array1, Array2};
use num_complex::Complex64;
use refeff_core::{
    GenfmtFeffBinHeader, GenfmtFeffBinPotential, GenfmtJasDriverOutput, GenfmtJasPathOutputs,
    GenfmtOrdinaryDriverOutput, GenfmtOrdinaryPathOutputs, GenfmtRetainedPathOutput,
};

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

impl From<GenfmtFeffBinPotential> for FeffBinPotential {
    fn from(potential: GenfmtFeffBinPotential) -> Self {
        Self {
            label: potential.label,
            atomic_number: potential.atomic_number,
        }
    }
}

impl From<&GenfmtFeffBinPotential> for FeffBinPotential {
    fn from(potential: &GenfmtFeffBinPotential) -> Self {
        potential.clone().into()
    }
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

impl From<GenfmtRetainedPathOutput> for FeffBinPath {
    fn from(output: GenfmtRetainedPathOutput) -> Self {
        Self {
            index: output.path_index,
            degeneracy: output.degeneracy,
            effective_half_path_length_bohr: output.effective_half_path_length_bohr,
            criterion: output.criterion_percent,
            potential_indices: output.potential_indices,
            positions: output.positions,
            beta: output.beta_angles,
            eta: output.eta_angles,
            leg_distances: output.leg_lengths,
            amplitude: output.amplitudes,
            phase: output.phases,
        }
    }
}

impl From<&GenfmtRetainedPathOutput> for FeffBinPath {
    fn from(output: &GenfmtRetainedPathOutput) -> Self {
        output.clone().into()
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

    /// Build complete FEFF `feff.bin` data from prepared GENFMT output.
    ///
    /// Retained paths are copied in caller-supplied order, matching the order
    /// FEFF writes paths as it walks `paths.dat`.
    #[must_use]
    pub fn from_genfmt_output(
        header: &GenfmtFeffBinHeader,
        retained_paths: &[GenfmtRetainedPathOutput],
    ) -> Self {
        let mut data = Self::from(header);
        data.paths = retained_paths.iter().map(FeffBinPath::from).collect();
        data
    }

    /// Build complete FEFF `feff.bin` data from ordinary GENFMT path outputs.
    #[must_use]
    pub fn from_genfmt_ordinary_outputs(
        header: &GenfmtFeffBinHeader,
        outputs: &GenfmtOrdinaryPathOutputs,
    ) -> Self {
        Self::from_genfmt_output(header, &outputs.retained_paths)
    }

    /// Build complete FEFF `feff.bin` data from ordinary GENFMT driver output.
    #[must_use]
    pub fn from_genfmt_ordinary_driver_output(output: &GenfmtOrdinaryDriverOutput) -> Self {
        Self::from_genfmt_ordinary_outputs(&output.header, &output.path_sequence.outputs)
    }

    /// Build complete FEFF `feff.bin` data from GENFMTJAS path outputs.
    #[must_use]
    pub fn from_genfmt_jas_outputs(
        header: &GenfmtFeffBinHeader,
        outputs: &GenfmtJasPathOutputs,
    ) -> Self {
        Self::from_genfmt_output(header, &outputs.retained_paths)
    }

    /// Build complete FEFF `feff.bin` data from GENFMTJAS driver output.
    #[must_use]
    pub fn from_genfmt_jas_driver_output(output: &GenfmtJasDriverOutput) -> Self {
        Self::from_genfmt_jas_outputs(&output.header, &output.path_sequence.outputs)
    }
}

impl From<GenfmtFeffBinHeader> for FeffBinData {
    fn from(header: GenfmtFeffBinHeader) -> Self {
        Self {
            version: header.version,
            pad_width: header.pad_width,
            ihole: header.core_hole,
            order: header.order,
            initial_angular_momentum: header.initial_angular_momentum,
            average_norman_radius: header.average_norman_radius,
            fermi_level: header.fermi_level,
            edge_energy: header.edge_energy,
            potentials: header.potentials.into_iter().map(Into::into).collect(),
            central_phase_shift: header.central_phase_shifts,
            complex_momentum: header.complex_momenta,
            real_momentum: header.wave_numbers,
            paths: Vec::new(),
            raw_text: None,
        }
    }
}

impl From<&GenfmtFeffBinHeader> for FeffBinData {
    fn from(header: &GenfmtFeffBinHeader) -> Self {
        header.clone().into()
    }
}
