use refeff_core::{FullSpectrumNumberDensityInput, full_spectrum_number_density};

use crate::Result;

use super::common::{
    check_i4, i64_from_usize, invalid_pot_bin, validate_finite_values, validate_len,
};
use super::types::{FullSpectrumPotentialState, PotBinData};

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
