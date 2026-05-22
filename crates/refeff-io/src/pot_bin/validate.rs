use crate::error::{IoError, Result};

use super::common::{
    check_i2, check_i4, i64_from_usize, invalid_pot_bin, validate_finite_values, validate_len,
    validate_shape2, validate_shape3,
};
use super::types::{
    POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS, PotBinData,
};

pub(super) fn validate_pot_bin(data: &PotBinData) -> Result<()> {
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
