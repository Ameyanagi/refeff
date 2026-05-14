//! FEFF `pot.bin` text/PAD potential-state codec.
//!
//! FEFF10 writes `pot.bin` from `POT/wrpot.f90` as a formatted text file with
//! fixed-width integer records and PAD-encoded real arrays. This module keeps
//! the same field order and Fortran column-major traversal while exposing the
//! data as typed `ndarray` arrays.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ShapeBuilder};
use refeff_core::{FullSpectrumNumberDensityInput, full_spectrum_number_density};

use crate::error::{IoError, Result};
use crate::format::repeated_ints;
use crate::pad::{decode_f64, encode_reals};

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

    fn from_slice(values: &[f64]) -> Result<Self> {
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

/// Render FEFF `pot.bin` text.
pub fn pot_bin_string(data: &PotBinData) -> Result<String> {
    validate_pot_bin(data)?;

    if let Some(raw_text) = &data.raw_text
        && raw_pot_bin_matches(data, raw_text)?
    {
        return Ok(raw_text.clone());
    }

    let potential_count = data.potential_count();
    let mut out = String::new();
    write_int_line(
        &mut out,
        &[
            i64_from_usize(data.titles.len(), "ntitle")?,
            i64_from_usize(potential_count - 1, "nph")?,
            i64_from_usize(data.pad_width, "npadx")?,
            i64::from(data.nohole),
            i64::from(data.ihole),
            i64::from(data.interstitial_selector),
            i64::from(data.automatic_folp),
            i64::from(data.jump_mode),
            i64::from(data.unfreeze_f),
        ],
        4,
    )?;

    for title in &data.titles {
        writeln!(out, "{title}")?;
    }

    write_pad_values(&mut out, &data.scalars.as_array(), data.pad_width)?;
    write_i4_chunks(
        &mut out,
        &usize_array_to_i64("imt", &data.muffin_tin_indices)?,
    )?;
    write_pad_values(
        &mut out,
        &data.muffin_tin_radii.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;
    write_i4_chunks(&mut out, &usize_array_to_i64("inrm", &data.norman_indices)?)?;
    write_i4_chunks(&mut out, &usize_array_to_i64("iz", &data.atomic_numbers)?)?;
    write_i4_chunks(&mut out, &i32_array_to_i64(&data.kappa))?;

    for (field, values) in [
        ("rnrm", data.norman_radii.view()),
        ("folp", data.overlap_factors.view()),
        ("folpx", data.max_overlap_factors.view()),
        ("xnatph", data.potential_multiplicities.view()),
        ("xion", data.ionization.view()),
        ("dgc0", data.initial_large_component.view()),
        ("dpc0", data.initial_small_component.view()),
    ] {
        let flat = values.iter().copied().collect::<Vec<_>>();
        validate_finite_values(field, flat.iter().copied())?;
        write_pad_values(&mut out, &flat, data.pad_width)?;
    }

    for values in [
        data.large_components.view(),
        data.small_components.view(),
        data.large_coefficients.view(),
        data.small_coefficients.view(),
    ] {
        write_pad_values(&mut out, &flatten3(values), data.pad_width)?;
    }

    for values in [
        data.electron_density.view(),
        data.coulomb_potential.view(),
        data.total_potential.view(),
        data.valence_density.view(),
        data.valence_potential.view(),
        data.magnetization_density.view(),
        data.orbital_occupancy.view(),
    ] {
        write_pad_values(&mut out, &flatten2(values), data.pad_width)?;
    }

    write_pad_values(
        &mut out,
        &data.orbital_energies.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;

    for potential in 0..potential_count {
        let mut values = Vec::with_capacity(POT_BIN_IORB_SLOTS);
        for slot in 0..POT_BIN_IORB_SLOTS {
            values.push(i64::from(data.occupied_orbital_indices[(slot, potential)]));
        }
        write_i2_chunks(&mut out, &values)?;
    }

    write_pad_values(
        &mut out,
        &data.norman_charges.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?;
    write_pad_values(
        &mut out,
        &flatten2(data.valence_occupancy.view()),
        data.pad_width,
    )?;
    Ok(out)
}

/// Parse FEFF `pot.bin` text.
pub fn parse_pot_bin(text: &str) -> Result<PotBinData> {
    let mut lines = PotBinLines::new(text);
    let header = lines.int_values("header", 9)?;
    let title_count = usize_from_i64(header[0], "ntitle")?;
    let potential_count = usize_from_i64(header[1], "nph")?
        .checked_add(1)
        .ok_or_else(|| invalid_pot_bin("nph", "potential count overflowed"))?;
    let pad_width = usize_from_i64(header[2], "npadx")?;
    let nohole = i32_from_i64(header[3], "nohole")?;
    let ihole = i32_from_i64(header[4], "ihole")?;
    let interstitial_selector = i32_from_i64(header[5], "inters")?;
    let automatic_folp = i32_from_i64(header[6], "iafolp")?;
    let jump_mode = i32_from_i64(header[7], "jumprm")?;
    let unfreeze_f = i32_from_i64(header[8], "iunf")?;

    let mut titles = Vec::with_capacity(title_count);
    for _ in 0..title_count {
        titles.push(lines.title()?);
    }

    let misc = lines.pad_reals("dum", pad_width, POT_BIN_MISC_SCALARS)?;
    let scalars = PotBinScalars::from_slice(&misc)?;
    let muffin_tin_indices = lines.usize_array("imt", potential_count)?;
    let muffin_tin_radii = lines.real_array("rmt", pad_width, potential_count)?;
    let norman_indices = lines.usize_array("inrm", potential_count)?;
    let atomic_numbers = lines.usize_array("iz", potential_count)?;
    let kappa = lines.i32_array("kappa", POT_BIN_ORBITALS)?;
    let norman_radii = lines.real_array("rnrm", pad_width, potential_count)?;
    let overlap_factors = lines.real_array("folp", pad_width, potential_count)?;
    let max_overlap_factors = lines.real_array("folpx", pad_width, potential_count)?;
    let potential_multiplicities = lines.real_array("xnatph", pad_width, potential_count)?;
    let ionization = lines.real_array("xion", pad_width, potential_count)?;
    let initial_large_component = lines.real_array("dgc0", pad_width, POT_BIN_RADIAL_POINTS)?;
    let initial_small_component = lines.real_array("dpc0", pad_width, POT_BIN_RADIAL_POINTS)?;

    let radial_orbital_potential = checked_count3(
        "dgc",
        POT_BIN_RADIAL_POINTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let coefficient_orbital_potential = checked_count3(
        "adgc",
        POT_BIN_COEFFICIENTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let radial_potential = checked_count2("edens", POT_BIN_RADIAL_POINTS, potential_count)?;
    let orbital_potential = checked_count2("xnval", POT_BIN_ORBITALS, potential_count)?;

    let large_components = array3_from_fortran(
        "dgc",
        lines.pad_reals("dgc", pad_width, radial_orbital_potential)?,
        POT_BIN_RADIAL_POINTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let small_components = array3_from_fortran(
        "dpc",
        lines.pad_reals("dpc", pad_width, radial_orbital_potential)?,
        POT_BIN_RADIAL_POINTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let large_coefficients = array3_from_fortran(
        "adgc",
        lines.pad_reals("adgc", pad_width, coefficient_orbital_potential)?,
        POT_BIN_COEFFICIENTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let small_coefficients = array3_from_fortran(
        "adpc",
        lines.pad_reals("adpc", pad_width, coefficient_orbital_potential)?,
        POT_BIN_COEFFICIENTS,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let electron_density = array2_from_fortran(
        "edens",
        lines.pad_reals("edens", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let coulomb_potential = array2_from_fortran(
        "vclap",
        lines.pad_reals("vclap", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let total_potential = array2_from_fortran(
        "vtot",
        lines.pad_reals("vtot", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let valence_density = array2_from_fortran(
        "edenvl",
        lines.pad_reals("edenvl", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let valence_potential = array2_from_fortran(
        "vvalgs",
        lines.pad_reals("vvalgs", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let magnetization_density = array2_from_fortran(
        "dmag",
        lines.pad_reals("dmag", pad_width, radial_potential)?,
        POT_BIN_RADIAL_POINTS,
        potential_count,
    )?;
    let orbital_occupancy = array2_from_fortran(
        "xnval",
        lines.pad_reals("xnval", pad_width, orbital_potential)?,
        POT_BIN_ORBITALS,
        potential_count,
    )?;
    let orbital_energies = lines.real_array("eorb", pad_width, POT_BIN_ORBITALS)?;

    let mut occupied_orbital_indices = Array2::<i32>::zeros((POT_BIN_IORB_SLOTS, potential_count));
    for potential in 0..potential_count {
        let values = lines.i32_values("iorb", POT_BIN_IORB_SLOTS)?;
        for slot in 0..POT_BIN_IORB_SLOTS {
            occupied_orbital_indices[(slot, potential)] = values[slot];
        }
    }

    let norman_charges = lines.real_array("qnrm", pad_width, potential_count)?;
    let xnmues = lines.pad_reals_to_eof("xnmues", pad_width)?;
    if xnmues.is_empty() || xnmues.len() % potential_count != 0 {
        return Err(IoError::PotBinShape {
            field: "xnmues",
            actual: vec![xnmues.len()],
            expected: vec![potential_count],
        });
    }
    let angular_count = xnmues.len() / potential_count;
    let valence_occupancy = array2_from_fortran("xnmues", xnmues, angular_count, potential_count)?;
    lines.finish()?;

    let data = PotBinData {
        titles,
        pad_width,
        nohole,
        ihole,
        interstitial_selector,
        automatic_folp,
        jump_mode,
        unfreeze_f,
        scalars,
        muffin_tin_indices,
        muffin_tin_radii,
        norman_indices,
        atomic_numbers,
        kappa,
        norman_radii,
        overlap_factors,
        max_overlap_factors,
        potential_multiplicities,
        ionization,
        initial_large_component,
        initial_small_component,
        large_components,
        small_components,
        large_coefficients,
        small_coefficients,
        electron_density,
        coulomb_potential,
        total_potential,
        valence_density,
        valence_potential,
        magnetization_density,
        orbital_occupancy,
        orbital_energies,
        occupied_orbital_indices,
        norman_charges,
        valence_occupancy,
        raw_text: Some(text.to_string()),
    };
    validate_pot_bin(&data)?;
    Ok(data)
}

/// Write FEFF `pot.bin` text to a file.
pub fn write_pot_bin(path: impl AsRef<Path>, data: &PotBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, pot_bin_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `pot.bin` text from a file.
pub fn read_pot_bin(path: impl AsRef<Path>) -> Result<PotBinData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_pot_bin(&text)
}

/// Estimate FEFF FULLSPECTRUM species number density from parsed `pot.bin`.
///
/// This is the typed `pot.bin` adapter for `FULLSPECTRUM/rddens.f90`, using
/// `iz(0:nph)`, `xnatph(0:nph)`, and `rnrm(0:nph)` from the potential state.
pub fn fullspectrum_number_density_from_pot_bin(
    target_atomic_number: usize,
    data: &PotBinData,
) -> Result<f64> {
    full_spectrum_number_density(FullSpectrumNumberDensityInput {
        target_atomic_number,
        atomic_numbers: data.atomic_numbers.view(),
        potential_multiplicities: data.potential_multiplicities.view(),
        norman_radii: data.norman_radii.view(),
    })
    .map_err(|source| invalid_pot_bin("fullspectrum_number_density", source.to_string()))
}

/// Borrow the `pot.bin` fields consumed by FEFF `FULLSPECTRUM/rdpotp_fs.f90`.
pub fn fullspectrum_potential_state_from_pot_bin(
    data: &PotBinData,
) -> Result<FullSpectrumPotentialState<'_>> {
    validate_fullspectrum_potential_state(data)?;
    Ok(FullSpectrumPotentialState {
        titles: &data.titles,
        atomic_numbers: data.atomic_numbers.view(),
        potential_multiplicities: data.potential_multiplicities.view(),
        norman_radii: data.norman_radii.view(),
    })
}

fn validate_fullspectrum_potential_state(data: &PotBinData) -> Result<()> {
    let potential_count = data.potential_count();
    if potential_count == 0 {
        return Err(invalid_pot_bin("nph", "at least one potential is required"));
    }
    check_i4(i64_from_usize(data.titles.len(), "ntitle")?, "ntitle")?;
    for title in &data.titles {
        if title.contains('\n') || title.contains('\r') {
            return Err(invalid_pot_bin(
                "title",
                "title records cannot contain line terminators",
            ));
        }
    }
    check_i4(i64_from_usize(potential_count - 1, "nph")?, "nph")?;
    validate_len("iz", data.atomic_numbers.len(), potential_count)?;
    validate_len(
        "xnatph",
        data.potential_multiplicities.len(),
        potential_count,
    )?;
    validate_len("rnrm", data.norman_radii.len(), potential_count)?;
    validate_finite_values("xnatph", data.potential_multiplicities.iter().copied())?;
    validate_finite_values("rnrm", data.norman_radii.iter().copied())?;
    for &value in &data.atomic_numbers {
        check_i4(i64_from_usize(value, "iz")?, "iz")?;
    }
    Ok(())
}

fn validate_pot_bin(data: &PotBinData) -> Result<()> {
    if data.pad_width <= 2 {
        return Err(IoError::InvalidPadWidth(data.pad_width));
    }
    let potential_count = data.potential_count();
    if potential_count == 0 {
        return Err(invalid_pot_bin("nph", "at least one potential is required"));
    }
    check_i4(i64_from_usize(data.titles.len(), "ntitle")?, "ntitle")?;
    for title in &data.titles {
        if title.contains('\n') || title.contains('\r') {
            return Err(invalid_pot_bin(
                "title",
                "title records cannot contain line terminators",
            ));
        }
    }
    check_i4(i64_from_usize(potential_count - 1, "nph")?, "nph")?;
    check_i4(i64_from_usize(data.pad_width, "npadx")?, "npadx")?;
    for (field, value) in [
        ("nohole", data.nohole),
        ("ihole", data.ihole),
        ("inters", data.interstitial_selector),
        ("iafolp", data.automatic_folp),
        ("jumprm", data.jump_mode),
        ("iunf", data.unfreeze_f),
    ] {
        check_i4(i64::from(value), field)?;
    }

    validate_len("imt", data.muffin_tin_indices.len(), potential_count)?;
    validate_len("rmt", data.muffin_tin_radii.len(), potential_count)?;
    validate_len("inrm", data.norman_indices.len(), potential_count)?;
    validate_len("iz", data.atomic_numbers.len(), potential_count)?;
    validate_len("kappa", data.kappa.len(), POT_BIN_ORBITALS)?;
    validate_len("rnrm", data.norman_radii.len(), potential_count)?;
    validate_len("folp", data.overlap_factors.len(), potential_count)?;
    validate_len("folpx", data.max_overlap_factors.len(), potential_count)?;
    validate_len(
        "xnatph",
        data.potential_multiplicities.len(),
        potential_count,
    )?;
    validate_len("xion", data.ionization.len(), potential_count)?;
    validate_len(
        "dgc0",
        data.initial_large_component.len(),
        POT_BIN_RADIAL_POINTS,
    )?;
    validate_len(
        "dpc0",
        data.initial_small_component.len(),
        POT_BIN_RADIAL_POINTS,
    )?;
    validate_len("eorb", data.orbital_energies.len(), POT_BIN_ORBITALS)?;
    validate_len("qnrm", data.norman_charges.len(), potential_count)?;
    validate_shape3(
        "dgc",
        data.large_components.dim(),
        (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potential_count),
    )?;
    validate_shape3(
        "dpc",
        data.small_components.dim(),
        (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potential_count),
    )?;
    validate_shape3(
        "adgc",
        data.large_coefficients.dim(),
        (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potential_count),
    )?;
    validate_shape3(
        "adpc",
        data.small_coefficients.dim(),
        (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potential_count),
    )?;
    for (field, actual) in [
        ("edens", data.electron_density.dim()),
        ("vclap", data.coulomb_potential.dim()),
        ("vtot", data.total_potential.dim()),
        ("edenvl", data.valence_density.dim()),
        ("vvalgs", data.valence_potential.dim()),
        ("dmag", data.magnetization_density.dim()),
    ] {
        validate_shape2(field, actual, (POT_BIN_RADIAL_POINTS, potential_count))?;
    }
    validate_shape2(
        "xnval",
        data.orbital_occupancy.dim(),
        (POT_BIN_ORBITALS, potential_count),
    )?;
    validate_shape2(
        "iorb",
        data.occupied_orbital_indices.dim(),
        (POT_BIN_IORB_SLOTS, potential_count),
    )?;
    validate_shape2(
        "xnmues",
        data.valence_occupancy.dim(),
        (data.valence_occupancy.nrows(), potential_count),
    )?;
    if data.valence_occupancy.nrows() == 0 {
        return Err(invalid_pot_bin(
            "xnmues",
            "at least one angular occupation channel is required",
        ));
    }

    validate_finite_values("dum", data.scalars.as_array())?;
    for (field, values) in [
        ("rmt", data.muffin_tin_radii.view()),
        ("rnrm", data.norman_radii.view()),
        ("folp", data.overlap_factors.view()),
        ("folpx", data.max_overlap_factors.view()),
        ("xnatph", data.potential_multiplicities.view()),
        ("xion", data.ionization.view()),
        ("dgc0", data.initial_large_component.view()),
        ("dpc0", data.initial_small_component.view()),
        ("eorb", data.orbital_energies.view()),
        ("qnrm", data.norman_charges.view()),
    ] {
        validate_finite_values(field, values.iter().copied())?;
    }
    for (field, values) in [
        ("dgc", data.large_components.view()),
        ("dpc", data.small_components.view()),
        ("adgc", data.large_coefficients.view()),
        ("adpc", data.small_coefficients.view()),
    ] {
        validate_finite_values(field, values.iter().copied())?;
    }
    for (field, values) in [
        ("edens", data.electron_density.view()),
        ("vclap", data.coulomb_potential.view()),
        ("vtot", data.total_potential.view()),
        ("edenvl", data.valence_density.view()),
        ("vvalgs", data.valence_potential.view()),
        ("dmag", data.magnetization_density.view()),
        ("xnval", data.orbital_occupancy.view()),
        ("xnmues", data.valence_occupancy.view()),
    ] {
        validate_finite_values(field, values.iter().copied())?;
    }

    for (field, values) in [
        ("imt", data.muffin_tin_indices.view()),
        ("inrm", data.norman_indices.view()),
        ("iz", data.atomic_numbers.view()),
    ] {
        for &value in values {
            check_i4(i64_from_usize(value, field)?, field)?;
        }
    }
    for &value in &data.kappa {
        check_i4(i64::from(value), "kappa")?;
    }
    for &value in &data.occupied_orbital_indices {
        check_i2(i64::from(value), "iorb")?;
    }
    Ok(())
}

fn raw_pot_bin_matches(data: &PotBinData, raw_text: &str) -> Result<bool> {
    let mut parsed = parse_pot_bin(raw_text)?;
    parsed.raw_text = None;
    let mut expected = data.clone();
    expected.raw_text = None;
    Ok(parsed == expected)
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::PotBinShape {
            field,
            actual: vec![actual],
            expected: vec![expected],
        })
    }
}

fn validate_shape2(
    field: &'static str,
    actual: (usize, usize),
    expected: (usize, usize),
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::PotBinShape {
            field,
            actual: vec![actual.0, actual.1],
            expected: vec![expected.0, expected.1],
        })
    }
}

fn validate_shape3(
    field: &'static str,
    actual: (usize, usize, usize),
    expected: (usize, usize, usize),
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::PotBinShape {
            field,
            actual: vec![actual.0, actual.1, actual.2],
            expected: vec![expected.0, expected.1, expected.2],
        })
    }
}

fn validate_finite_values(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<()> {
    for value in values {
        if !value.is_finite() {
            return Err(invalid_pot_bin(
                field,
                format!("value must be finite, got {value}"),
            ));
        }
    }
    Ok(())
}

fn write_i4_chunks(out: &mut String, values: &[i64]) -> Result<()> {
    for chunk in values.chunks(20) {
        write_int_line(out, chunk, 4)?;
    }
    Ok(())
}

fn write_i2_chunks(out: &mut String, values: &[i64]) -> Result<()> {
    for chunk in values.chunks(8) {
        write_int_line(out, chunk, 2)?;
    }
    Ok(())
}

fn write_int_line(out: &mut String, values: &[i64], width: usize) -> Result<()> {
    for &value in values {
        check_fixed_int(value, width, "integer")?;
    }
    writeln!(out, "{}", repeated_ints(values.iter().copied(), width))?;
    Ok(())
}

fn write_pad_values(out: &mut String, values: &[f64], pad_width: usize) -> Result<()> {
    out.push_str(&encode_reals(values, pad_width)?);
    Ok(())
}

fn flatten2(values: ArrayView2<'_, f64>) -> Vec<f64> {
    let (rows, cols) = values.dim();
    let mut flat = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            flat.push(values[(row, col)]);
        }
    }
    flat
}

fn flatten3(values: ArrayView3<'_, f64>) -> Vec<f64> {
    let (rows, cols, planes) = values.dim();
    let mut flat = Vec::with_capacity(rows * cols * planes);
    for plane in 0..planes {
        for col in 0..cols {
            for row in 0..rows {
                flat.push(values[(row, col, plane)]);
            }
        }
    }
    flat
}

fn array2_from_fortran(
    field: &'static str,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
) -> Result<Array2<f64>> {
    Array2::from_shape_vec((rows, cols).f(), values).map_err(|_| IoError::PotBinShape {
        field,
        actual: vec![rows, cols],
        expected: vec![rows, cols],
    })
}

fn array3_from_fortran(
    field: &'static str,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    planes: usize,
) -> Result<Array3<f64>> {
    Array3::from_shape_vec((rows, cols, planes).f(), values).map_err(|_| IoError::PotBinShape {
        field,
        actual: vec![rows, cols, planes],
        expected: vec![rows, cols, planes],
    })
}

struct PotBinLines<'a> {
    lines: Vec<&'a str>,
    position: usize,
}

impl<'a> PotBinLines<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.lines().collect(),
            position: 0,
        }
    }

    fn finish(self) -> Result<()> {
        let count = self.lines[self.position..]
            .iter()
            .filter(|line| !line.trim().is_empty())
            .count();
        if count == 0 {
            Ok(())
        } else {
            Err(IoError::PotBinTrailingLines { count })
        }
    }

    fn title(&mut self) -> Result<String> {
        let line = self.next_line("title")?;
        Ok(line.to_string())
    }

    fn int_values(&mut self, field: &'static str, expected: usize) -> Result<Vec<i64>> {
        let mut values = Vec::with_capacity(expected);
        while values.len() < expected {
            let line = self.next_line(field)?;
            for token in line.split_whitespace() {
                if values.len() == expected {
                    break;
                }
                values.push(token.parse::<i64>().map_err(|_| IoError::PotBinParse {
                    field,
                    token: token.to_string(),
                })?);
            }
        }
        Ok(values)
    }

    fn i32_values(&mut self, field: &'static str, expected: usize) -> Result<Vec<i32>> {
        self.int_values(field, expected)?
            .into_iter()
            .map(|value| i32_from_i64(value, field))
            .collect()
    }

    fn i32_array(&mut self, field: &'static str, len: usize) -> Result<Array1<i32>> {
        Ok(Array1::from_vec(self.i32_values(field, len)?))
    }

    fn usize_array(&mut self, field: &'static str, len: usize) -> Result<Array1<usize>> {
        let values = self
            .int_values(field, len)?
            .into_iter()
            .map(|value| usize_from_i64(value, field))
            .collect::<Result<Vec<_>>>()?;
        Ok(Array1::from_vec(values))
    }

    fn real_array(
        &mut self,
        field: &'static str,
        pad_width: usize,
        len: usize,
    ) -> Result<Array1<f64>> {
        Ok(Array1::from_vec(self.pad_reals(field, pad_width, len)?))
    }

    fn pad_reals(
        &mut self,
        field: &'static str,
        pad_width: usize,
        expected: usize,
    ) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(expected);
        while values.len() < expected {
            let line = self.next_line(field)?;
            let decoded = decode_pad_line(field, line, pad_width)?;
            if decoded.is_empty() {
                return Err(IoError::PadPayload {
                    payload_len: 0,
                    unit_len: pad_width,
                });
            }
            for value in decoded {
                if values.len() < expected {
                    values.push(value);
                }
            }
        }
        Ok(values)
    }

    fn pad_reals_to_eof(&mut self, field: &'static str, pad_width: usize) -> Result<Vec<f64>> {
        let mut values = Vec::new();
        while self.position < self.lines.len() {
            let line = self.next_line(field)?;
            if line.trim().is_empty() {
                continue;
            }
            values.extend(decode_pad_line(field, line, pad_width)?);
        }
        Ok(values)
    }

    fn next_line(&mut self, field: &'static str) -> Result<&'a str> {
        let line = self
            .lines
            .get(self.position)
            .copied()
            .ok_or(IoError::PotBinMissing { field })?;
        self.position += 1;
        Ok(line)
    }
}

fn decode_pad_line(field: &'static str, line: &str, pad_width: usize) -> Result<Vec<f64>> {
    if pad_width <= 2 {
        return Err(IoError::InvalidPadWidth(pad_width));
    }
    let trimmed = line.trim_start().trim_end();
    let Some(found) = trimmed.chars().next() else {
        return Err(IoError::PotBinMissing { field });
    };
    if found != '!' {
        return Err(IoError::PadMarker {
            expected: '!',
            found,
        });
    }
    let payload = &trimmed[found.len_utf8()..];
    if payload.is_empty() || !payload.len().is_multiple_of(pad_width) {
        return Err(IoError::PadPayload {
            payload_len: payload.len(),
            unit_len: pad_width,
        });
    }

    let mut values = Vec::with_capacity(payload.len() / pad_width);
    for chunk in payload.as_bytes().chunks(pad_width) {
        let chunk =
            std::str::from_utf8(chunk).map_err(|source| IoError::PadChunkUtf8 { source })?;
        values.push(decode_f64(chunk, pad_width)?);
    }
    Ok(values)
}

fn usize_array_to_i64(field: &'static str, values: &Array1<usize>) -> Result<Vec<i64>> {
    values
        .iter()
        .map(|&value| i64_from_usize(value, field))
        .collect()
}

fn i32_array_to_i64(values: &Array1<i32>) -> Vec<i64> {
    values.iter().map(|&value| i64::from(value)).collect()
}

fn checked_count2(field: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_pot_bin(field, "array element count overflowed"))
}

fn checked_count3(field: &'static str, rows: usize, cols: usize, planes: usize) -> Result<usize> {
    checked_count2(field, rows, cols)?
        .checked_mul(planes)
        .ok_or_else(|| invalid_pot_bin(field, "array element count overflowed"))
}

fn i64_from_usize(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| invalid_pot_bin(field, format!("value {value} does not fit in i64")))
}

fn usize_from_i64(value: i64, field: &'static str) -> Result<usize> {
    if value < 0 {
        return Err(invalid_pot_bin(
            field,
            format!("value {value} must be non-negative"),
        ));
    }
    usize::try_from(value)
        .map_err(|_| invalid_pot_bin(field, format!("value {value} does not fit in usize")))
}

fn i32_from_i64(value: i64, field: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| invalid_pot_bin(field, format!("value {value} does not fit in i32")))
}

fn check_i4(value: i64, field: &'static str) -> Result<()> {
    check_fixed_int(value, 4, field)
}

fn check_i2(value: i64, field: &'static str) -> Result<()> {
    check_fixed_int(value, 2, field)
}

fn check_fixed_int(value: i64, width: usize, field: &'static str) -> Result<()> {
    let limit = 10_i64.pow(u32::try_from(width).map_err(|_| {
        invalid_pot_bin(field, format!("integer field width {width} is too large"))
    })?);
    let negative_limit = -(limit / 10 - 1);
    if value < negative_limit || value >= limit {
        Err(invalid_pot_bin(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

fn invalid_pot_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidPotBin {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_header_and_integer_chunks_like_feff() -> Result<()> {
        let data = sample_pot_bin_data();
        let text = pot_bin_string(&data)?;
        assert_eq!(
            text.lines().next(),
            Some("    2    1    8   -1    2    3    4    5    6")
        );

        assert!(text.lines().any(|line| {
            line == "  -20  -19  -18  -17  -16  -15  -14  -13  -12  -11  -10   -9   -8   -7   -6   -5   -4   -3   -2   -1"
        }));
        assert!(text.lines().any(|line| {
            line == "    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19"
        }));
        assert!(text.lines().any(|line| line == "   20"));
        assert!(text.lines().any(|line| line == " -5 -4 -3 -2 -1  0  1  2"));
        assert!(text.lines().any(|line| line == "  3  4"));
        Ok(())
    }

    #[test]
    fn roundtrips_pot_bin_text_with_pad_tolerance() -> Result<()> {
        let data = sample_pot_bin_data();
        let parsed = parse_pot_bin(&pot_bin_string(&data)?)?;
        assert_eq!(parsed.titles, data.titles);
        assert_eq!(parsed.pad_width, data.pad_width);
        assert_eq!(parsed.nohole, data.nohole);
        assert_eq!(parsed.ihole, data.ihole);
        assert_eq!(parsed.interstitial_selector, data.interstitial_selector);
        assert_eq!(parsed.automatic_folp, data.automatic_folp);
        assert_eq!(parsed.jump_mode, data.jump_mode);
        assert_eq!(parsed.unfreeze_f, data.unfreeze_f);
        assert_eq!(parsed.muffin_tin_indices, data.muffin_tin_indices);
        assert_eq!(parsed.norman_indices, data.norman_indices);
        assert_eq!(parsed.atomic_numbers, data.atomic_numbers);
        assert_eq!(parsed.kappa, data.kappa);
        assert_eq!(
            parsed.occupied_orbital_indices,
            data.occupied_orbital_indices
        );
        assert_close_iter(parsed.scalars.as_array(), data.scalars.as_array());
        assert_close_iter(parsed.muffin_tin_radii, data.muffin_tin_radii);
        assert_close_iter(parsed.large_components, data.large_components);
        assert_close_iter(parsed.large_coefficients, data.large_coefficients);
        assert_close_iter(parsed.electron_density, data.electron_density);
        assert_close_iter(parsed.orbital_occupancy, data.orbital_occupancy);
        assert_close_iter(parsed.valence_occupancy, data.valence_occupancy);
        Ok(())
    }

    #[test]
    fn derives_fullspectrum_number_density_from_pot_bin() -> Result<()> {
        let data = sample_pot_bin_data();
        let copper_density = fullspectrum_number_density_from_pot_bin(29, &data)?;
        let oxygen_density = fullspectrum_number_density_from_pot_bin(8, &data)?;
        let missing_density = fullspectrum_number_density_from_pot_bin(26, &data)?;

        assert!((copper_density - 0.004_604_023_193_216_264).abs() < 1.0e-16);
        assert!((oxygen_density - 0.018_416_092_772_865_055).abs() < 1.0e-16);
        assert_eq!(missing_density, 0.0);
        Ok(())
    }

    #[test]
    fn exposes_fullspectrum_rdpotp_fields_from_pot_bin() -> Result<()> {
        let data = sample_pot_bin_data();
        let state = fullspectrum_potential_state_from_pot_bin(&data)?;

        assert_eq!(state.title_count(), data.titles.len());
        assert_eq!(state.nph(), data.potential_count() - 1);
        assert_eq!(state.titles[0], data.titles[0]);
        assert!(state.atomic_numbers.iter().eq(data.atomic_numbers.iter()));
        assert!(
            state
                .potential_multiplicities
                .iter()
                .eq(data.potential_multiplicities.iter())
        );
        assert!(state.norman_radii.iter().eq(data.norman_radii.iter()));
        Ok(())
    }

    #[test]
    fn rejects_bad_fullspectrum_rdpotp_view_inputs() {
        let mut data = sample_pot_bin_data();
        data.norman_radii = Array1::zeros(data.potential_count().saturating_sub(1));

        assert!(matches!(
            fullspectrum_potential_state_from_pot_bin(&data),
            Err(IoError::PotBinShape { field: "rnrm", .. })
        ));
    }

    #[test]
    fn preserves_feff_title_record_spacing() -> Result<()> {
        let mut data = sample_pot_bin_data();
        data.titles[0] =
            " POT  SCF 100  4.0000   0, screened core-hole, AFOLP (folp(0)= 1.150)".to_string();
        let text = pot_bin_string(&data)?;
        assert_eq!(text.lines().nth(1), Some(data.titles[0].as_str()));
        let parsed = parse_pot_bin(&text)?;
        assert_eq!(parsed.titles[0], data.titles[0]);
        Ok(())
    }

    #[test]
    fn preserves_matching_raw_text() -> Result<()> {
        let data = sample_pot_bin_data();
        let text = pot_bin_string(&data)?;
        let mut parsed = parse_pot_bin(&text)?;
        let raw_text = parsed
            .raw_text
            .as_mut()
            .ok_or(IoError::PotBinMissing { field: "raw_text" })?;
        raw_text.push('\n');

        let mut expected = text.clone();
        expected.push('\n');
        assert_eq!(pot_bin_string(&parsed)?, expected);

        parsed.scalars.fermi_level += 1.0;
        assert_ne!(pot_bin_string(&parsed)?, expected);
        Ok(())
    }

    #[test]
    fn rejects_invalid_shapes_and_bad_tokens() {
        let mut bad = sample_pot_bin_data();
        bad.kappa = Array1::from_vec(vec![1]);
        assert!(matches!(
            pot_bin_string(&bad),
            Err(IoError::PotBinShape {
                field: "kappa",
                actual,
                expected,
            }) if actual == vec![1] && expected == vec![POT_BIN_ORBITALS]
        ));

        assert!(matches!(
            parse_pot_bin("not-an-int"),
            Err(IoError::PotBinParse {
                field: "header",
                ..
            })
        ));
    }

    fn sample_pot_bin_data() -> PotBinData {
        let potentials = 2;
        let angular_count = 4;
        PotBinData {
            titles: vec!["Cu crystal".to_string(), "second title".to_string()],
            pad_width: POT_BIN_DEFAULT_PAD_WIDTH,
            nohole: -1,
            ihole: 2,
            interstitial_selector: 3,
            automatic_folp: 4,
            jump_mode: 5,
            unfreeze_f: 6,
            scalars: PotBinScalars {
                average_norman_radius: 1.25,
                fermi_level: -0.4,
                interstitial_potential: -1.2,
                interstitial_density: 0.03,
                edge_position: 9.1,
                amplitude_reduction: 0.85,
                relaxation_energy: 0.15,
                plasmon_frequency: 2.4,
                core_valence_energy: -3.0,
                density_radius: 1.7,
                fermi_momentum: 0.9,
                total_charge: 42.0,
                total_volume: 11.0,
            },
            muffin_tin_indices: Array1::from_vec(vec![12, 13]),
            muffin_tin_radii: Array1::from_vec(vec![1.1, 1.2]),
            norman_indices: Array1::from_vec(vec![20, 21]),
            atomic_numbers: Array1::from_vec(vec![29, 8]),
            kappa: Array1::from_iter(-20..=20),
            norman_radii: Array1::from_vec(vec![2.1, 2.2]),
            overlap_factors: Array1::from_vec(vec![0.9, 0.8]),
            max_overlap_factors: Array1::from_vec(vec![1.3, 1.4]),
            potential_multiplicities: Array1::from_vec(vec![1.0, 4.0]),
            ionization: Array1::from_vec(vec![0.0, 1.0]),
            initial_large_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
                0.001 * (row + 1) as f64
            }),
            initial_small_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
                -0.001 * (row + 1) as f64
            }),
            large_components: Array3::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
                |(row, orbital, potential)| {
                    0.0001 * (row + 1) as f64 + 0.01 * orbital as f64 + 0.1 * potential as f64
                },
            ),
            small_components: Array3::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
                |(row, orbital, potential)| {
                    -0.0001 * (row + 1) as f64 - 0.01 * orbital as f64 - 0.1 * potential as f64
                },
            ),
            large_coefficients: Array3::from_shape_fn(
                (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
                |(coef, orbital, potential)| {
                    0.01 * (coef + 1) as f64 + 0.001 * orbital as f64 + 0.1 * potential as f64
                },
            ),
            small_coefficients: Array3::from_shape_fn(
                (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
                |(coef, orbital, potential)| {
                    -0.01 * (coef + 1) as f64 - 0.001 * orbital as f64 - 0.1 * potential as f64
                },
            ),
            electron_density: radial_matrix(potentials, 0.01),
            coulomb_potential: radial_matrix(potentials, -0.02),
            total_potential: radial_matrix(potentials, -0.03),
            valence_density: radial_matrix(potentials, 0.004),
            valence_potential: radial_matrix(potentials, -0.005),
            magnetization_density: radial_matrix(potentials, 0.0002),
            orbital_occupancy: Array2::from_shape_fn(
                (POT_BIN_ORBITALS, potentials),
                |(orbital, potential)| 0.2 * orbital as f64 + potential as f64,
            ),
            orbital_energies: Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| {
                -10.0 + orbital as f64 * 0.25
            }),
            occupied_orbital_indices: Array2::from_shape_fn(
                (POT_BIN_IORB_SLOTS, potentials),
                |(slot, _)| slot as i32 - 5,
            ),
            norman_charges: Array1::from_vec(vec![28.5, 7.5]),
            valence_occupancy: Array2::from_shape_fn(
                (angular_count, potentials),
                |(angular, potential)| 0.5 * angular as f64 + potential as f64,
            ),
            raw_text: None,
        }
    }

    fn radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
        Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
            scale * (row + 1) as f64 + potential as f64 * 0.125
        })
    }

    fn assert_close_iter(
        actual: impl IntoIterator<Item = f64>,
        expected: impl IntoIterator<Item = f64>,
    ) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-6,
                "{actual} != {expected}"
            );
        }
    }
}
