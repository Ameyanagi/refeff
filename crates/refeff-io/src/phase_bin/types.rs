use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;

use crate::error::{IoError, Result};

/// FEFF10 default PAD width used by `wrxsph`.
pub const PHASE_BIN_DEFAULT_PAD_WIDTH: usize = 8;
/// Number of scalar values in the FEFF `dum(3)` phase header block.
pub const PHASE_BIN_SCALARS: usize = 3;
/// Historical non-NRIXS transition-moment count read by old `rdxsph`.
pub const PHASE_BIN_DEFAULT_TRANSITION_COUNT: usize = 8;

/// Scalar `dum(3)` block from FEFF `phase.bin`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseBinScalars {
    /// Average Norman radius, `rnrmav`.
    pub average_norman_radius: f64,
    /// Fermi level position, `xmu`.
    pub fermi_level: f64,
    /// Edge energy, `edge`.
    pub edge_energy: f64,
}

impl PhaseBinScalars {
    /// Return the FEFF `dum(3)` values in `wrxsph` order.
    #[must_use]
    pub fn as_array(self) -> [f64; PHASE_BIN_SCALARS] {
        [
            self.average_norman_radius,
            self.fermi_level,
            self.edge_energy,
        ]
    }

    pub(super) fn from_slice(values: &[f64]) -> Result<Self> {
        if values.len() != PHASE_BIN_SCALARS {
            return Err(IoError::PhaseBinShape {
                field: "dum",
                actual: vec![values.len()],
                expected: vec![PHASE_BIN_SCALARS],
            });
        }
        Ok(Self {
            average_norman_radius: values[0],
            fermi_level: values[1],
            edge_energy: values[2],
        })
    }
}

/// Per-potential phase-shift block from FEFF `phase.bin`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBinPotential {
    /// Maximum angular momentum written for this potential.
    pub lmax: usize,
    /// Atomic number, `iz(iph)`.
    pub atomic_number: usize,
    /// FEFF six-character potential label, `potlbl(iph)`.
    pub label: String,
    /// Phase shifts as `(energy, l_slot -lmax..lmax, spin)`.
    pub phase_shifts: Array3<Complex64>,
}

/// Raw PAD blocks captured from a parsed FEFF `phase.bin` file.
///
/// Rendering reuses an individual block only when it still decodes to the
/// matching typed values, which preserves FEFF byte output without hiding
/// caller edits to the typed arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseBinRawPads {
    /// Raw `dum(3)` real PAD block.
    pub scalars: Option<String>,
    /// Raw `em(1:ne)` complex PAD block.
    pub energy_grid: Option<String>,
    /// Raw `eref(1:ne,1:nsp)` complex PAD block.
    pub reference_energy: Option<String>,
    /// Raw phase-shift PAD blocks as `(potential, spin)`.
    pub phase_shifts: Vec<Vec<Option<String>>>,
    /// Raw transition-moment PAD blocks as `q` slices.
    pub transition_moments: Vec<Option<String>>,
}

/// FEFF `phase.bin` contents from `XSPH/wrxsph.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBinData {
    /// Spin channel count, `nsp`.
    pub spin_count: usize,
    /// Energy grid count, `ne`.
    pub energy_count: usize,
    /// Main horizontal-axis energy count, `ne1`.
    pub main_energy_count: usize,
    /// Auxiliary horizontal-axis energy count, `ne3`.
    pub auxiliary_energy_count: usize,
    /// Core-hole index, `ihole`.
    pub ihole: i32,
    /// Fermi-level grid index, `ik0`.
    pub fermi_index: i32,
    /// PAD field width, `npadx`.
    pub pad_width: usize,
    /// FEFFQ final-state channel count, `kfinmax`.
    pub final_state_count: usize,
    /// Number of transition-moment channels written, `indmax`.
    pub transition_count: usize,
    /// Momentum-transfer vector count, `nq`.
    pub q_count: usize,
    /// FEFF scalar `dum(3)` block.
    pub scalars: PhaseBinScalars,
    /// Complex energy mesh, `em(1:ne)`.
    pub energy_grid: Array1<Complex64>,
    /// Reference/self-energy mesh as `(energy, spin)`, `eref`.
    pub reference_energy: Array2<Complex64>,
    /// Per-potential phase-shift blocks for FEFF `iph=0:nph`.
    pub potentials: Vec<PhaseBinPotential>,
    /// Transition moments as `(energy, q, transition, spin)`, `rkk`.
    pub transition_moments: Array4<Complex64>,
    /// Raw PAD blocks from a parsed FEFF file, used for exact compatible
    /// re-emission when the corresponding typed values are unchanged.
    pub raw_pads: Option<PhaseBinRawPads>,
}

impl PhaseBinData {
    /// Number of FEFF potential types represented by `0:nph`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.potentials.len()
    }
}
