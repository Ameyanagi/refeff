use ndarray::{Array1, Array2, Array3, ArrayView1};

use crate::error::{IoError, Result};

/// FEFF radial mesh size used by `wrpot`/`rdpot`.
pub const POT_BIN_RADIAL_POINTS: usize = 251;
/// Number of stored FEFF orbital channels after the FEFF10 superheavy update.
pub const POT_BIN_ORBITALS: usize = 41;
/// Number of polynomial expansion coefficients stored per orbital.
pub const POT_BIN_COEFFICIENTS: usize = 10;
/// Number of `iorb(-5:4, iph)` slots stored for each potential.
pub const POT_BIN_IORB_SLOTS: usize = 10;
/// Number of scalar values in the FEFF `dum(13)` block.
pub const POT_BIN_MISC_SCALARS: usize = 13;
/// FEFF10 default PAD width in `wrpot`.
pub const POT_BIN_DEFAULT_PAD_WIDTH: usize = 8;

/// Scalar `dum(13)` block from FEFF `pot.bin`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotBinScalars {
    /// Average Norman radius, `rnrmav`.
    pub average_norman_radius: f64,
    /// Fermi level position, `xmu`.
    pub fermi_level: f64,
    /// Muffin-tin zero/interstitial potential, `vint`.
    pub interstitial_potential: f64,
    /// Interstitial density, `rhoint`.
    pub interstitial_density: f64,
    /// Edge position, `emu`.
    pub edge_position: f64,
    /// Many-body amplitude reduction factor, `s02`.
    pub amplitude_reduction: f64,
    /// Relaxation-energy estimate, `erelax`.
    pub relaxation_energy: f64,
    /// Plasmon-frequency estimate, `wp`.
    pub plasmon_frequency: f64,
    /// Core-valence separation energy, `ecv`.
    pub core_valence_energy: f64,
    /// Interstitial-density `r_s` estimate, `rs`.
    pub density_radius: f64,
    /// Fermi-momentum estimate, `xf`.
    pub fermi_momentum: f64,
    /// Total cluster charge, `qtotel`.
    pub total_charge: f64,
    /// Total volume, `totvol`.
    pub total_volume: f64,
}

impl PotBinScalars {
    /// Return the FEFF `dum(13)` values in `wrpot` order.
    #[must_use]
    pub fn as_array(self) -> [f64; POT_BIN_MISC_SCALARS] {
        [
            self.average_norman_radius,
            self.fermi_level,
            self.interstitial_potential,
            self.interstitial_density,
            self.edge_position,
            self.amplitude_reduction,
            self.relaxation_energy,
            self.plasmon_frequency,
            self.core_valence_energy,
            self.density_radius,
            self.fermi_momentum,
            self.total_charge,
            self.total_volume,
        ]
    }

    pub(super) fn from_slice(values: &[f64]) -> Result<Self> {
        if values.len() != POT_BIN_MISC_SCALARS {
            return Err(IoError::PotBinShape {
                field: "dum",
                actual: vec![values.len()],
                expected: vec![POT_BIN_MISC_SCALARS],
            });
        }
        Ok(Self {
            average_norman_radius: values[0],
            fermi_level: values[1],
            interstitial_potential: values[2],
            interstitial_density: values[3],
            edge_position: values[4],
            amplitude_reduction: values[5],
            relaxation_energy: values[6],
            plasmon_frequency: values[7],
            core_valence_energy: values[8],
            density_radius: values[9],
            fermi_momentum: values[10],
            total_charge: values[11],
            total_volume: values[12],
        })
    }
}

/// FEFF `pot.bin` contents from `POT/wrpot.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct PotBinData {
    /// Title records written after the header.
    pub titles: Vec<String>,
    /// PAD field width, `npadx`.
    pub pad_width: usize,
    /// Core-hole switch, `nohole`.
    pub nohole: i32,
    /// Core-hole index, `ihole`.
    pub ihole: i32,
    /// Interstitial selector, `inters`.
    pub interstitial_selector: i32,
    /// Automatic overlap factor switch, `iafolp`.
    pub automatic_folp: i32,
    /// Jump-removal switch, `jumprm`.
    pub jump_mode: i32,
    /// Frozen-orbital switch, `iunf`.
    pub unfreeze_f: i32,
    /// FEFF scalar `dum(13)` block.
    pub scalars: PotBinScalars,
    /// Muffin-tin radial indices, `imt(0:nph)`.
    pub muffin_tin_indices: Array1<usize>,
    /// Muffin-tin radii, `rmt(0:nph)`.
    pub muffin_tin_radii: Array1<f64>,
    /// Norman radial indices, `inrm(0:nph)`.
    pub norman_indices: Array1<usize>,
    /// Atomic numbers, `iz(0:nph)`.
    pub atomic_numbers: Array1<usize>,
    /// Orbital `kappa(1:41)` values.
    pub kappa: Array1<i32>,
    /// Norman radii, `rnrm(0:nph)`.
    pub norman_radii: Array1<f64>,
    /// Muffin-tin overlap factors, `folp(0:nph)`.
    pub overlap_factors: Array1<f64>,
    /// Maximum overlap factors, `folpx(0:nph)`.
    pub max_overlap_factors: Array1<f64>,
    /// Potential multiplicities, `xnatph(0:nph)`.
    pub potential_multiplicities: Array1<f64>,
    /// Ionization values, `xion(0:nph)`.
    pub ionization: Array1<f64>,
    /// Initial-orbital large component, `dgc0(1:251)`.
    pub initial_large_component: Array1<f64>,
    /// Initial-orbital small component, `dpc0(1:251)`.
    pub initial_small_component: Array1<f64>,
    /// Large Dirac components as `(radial, orbital, potential)`, `dgc`.
    pub large_components: Array3<f64>,
    /// Small Dirac components as `(radial, orbital, potential)`, `dpc`.
    pub small_components: Array3<f64>,
    /// Large-component expansion coefficients as `(coefficient, orbital, potential)`, `adgc`.
    pub large_coefficients: Array3<f64>,
    /// Small-component expansion coefficients as `(coefficient, orbital, potential)`, `adpc`.
    pub small_coefficients: Array3<f64>,
    /// Total electron density as `(radial, potential)`, `edens`.
    pub electron_density: Array2<f64>,
    /// Coulomb potential as `(radial, potential)`, `vclap`.
    pub coulomb_potential: Array2<f64>,
    /// Total potential as `(radial, potential)`, `vtot`.
    pub total_potential: Array2<f64>,
    /// Valence density as `(radial, potential)`, `edenvl`.
    pub valence_density: Array2<f64>,
    /// Valence-only ground-state potential as `(radial, potential)`, `vvalgs`.
    pub valence_potential: Array2<f64>,
    /// Spin-up minus spin-down density as `(radial, potential)`, `dmag`.
    pub magnetization_density: Array2<f64>,
    /// Valence occupancy as `(orbital, potential)`, `xnval`.
    pub orbital_occupancy: Array2<f64>,
    /// Absorber orbital energies, `eorb(1:41)`.
    pub orbital_energies: Array1<f64>,
    /// Last occupied orbital indices as `(kappa_slot -5..4, potential)`, `iorb`.
    pub occupied_orbital_indices: Array2<i32>,
    /// Norman charges, `qnrm(0:nph)`.
    pub norman_charges: Array1<f64>,
    /// SCF valence occupations as `(angular_channel 0:lx, potential)`, `xnmues`.
    pub valence_occupancy: Array2<f64>,
    /// Raw parsed `pot.bin` text for exact re-emission when the typed content
    /// is unchanged.
    pub raw_text: Option<String>,
}

impl PotBinData {
    /// Number of potential types represented by the FEFF `0:nph` arrays.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.muffin_tin_indices.len()
    }

    /// Number of angular occupation rows represented by FEFF `0:lx`.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.valence_occupancy.nrows()
    }
}

/// Borrowed FULLSPECTRUM view of the `pot.bin` fields read by `rdpotp_fs.f90`.
///
/// FEFF `FULLSPECTRUM/rdpotp_fs.f90` reads only title records, atomic numbers,
/// potential multiplicities, and Norman radii from the much larger `pot.bin`
/// state. This view keeps those arrays borrowed from [`PotBinData`] for later
/// FULLSPECTRUM orchestration without copying the potential-state payload.
#[derive(Debug, Clone)]
pub struct FullSpectrumPotentialState<'a> {
    /// Title records, FEFF `title(1:ntitle)`.
    pub titles: &'a [String],
    /// Atomic numbers, FEFF `iz(0:nph)`.
    pub atomic_numbers: ArrayView1<'a, usize>,
    /// Potential multiplicities, FEFF `xnatph(0:nph)`.
    pub potential_multiplicities: ArrayView1<'a, f64>,
    /// Norman radii in Bohr, FEFF `rnrm(0:nph)`.
    pub norman_radii: ArrayView1<'a, f64>,
}

impl FullSpectrumPotentialState<'_> {
    /// Number of title records, FEFF `ntitle`.
    #[must_use]
    pub fn title_count(&self) -> usize {
        self.titles.len()
    }

    /// Number of potential slots represented by FEFF `0:nph` arrays.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.atomic_numbers.len()
    }

    /// FEFF `nph`, the highest potential index in the `0:nph` arrays.
    #[must_use]
    pub fn nph(&self) -> usize {
        self.potential_count().saturating_sub(1)
    }
}
