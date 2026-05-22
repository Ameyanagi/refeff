use ndarray::Array1;
use num_complex::Complex64;

use crate::Result;

use super::common::{
    DMDW_A2F_INFO_PATH, DMDW_AKW_DAT_PATH, DMDW_EGRID_INFO_PATH, DMDW_SELF_ENERGY_DAT_PATH,
    DMDW_SPECTRAL_INFO_PATH, invalid_data,
};
use super::types::{
    DmdwA2fInfoData, DmdwAkwDatData, DmdwEnergyGridInfo, DmdwSelfEnergyDatData,
    DmdwSpectralInfoData,
};

pub(super) fn validate_dmdw_egrid_info(data: &DmdwEnergyGridInfo) -> Result<()> {
    validate_finite_field(DMDW_EGRID_INFO_PATH, "low_energy_mev", data.low_energy_mev)?;
    validate_finite_field(
        DMDW_EGRID_INFO_PATH,
        "high_energy_mev",
        data.high_energy_mev,
    )?;
    validate_finite_field(DMDW_EGRID_INFO_PATH, "step_mev", data.step_mev)?;
    validate_finite_field(
        DMDW_EGRID_INFO_PATH,
        "characteristic_energy_mev",
        data.characteristic_energy_mev,
    )?;
    validate_finite_field(
        DMDW_EGRID_INFO_PATH,
        "electron_energy_mev",
        data.electron_energy_mev,
    )?;
    validate_finite_field(
        DMDW_EGRID_INFO_PATH,
        "selected_energy_mev",
        data.selected_energy_mev,
    )?;
    if data.high_energy_mev < data.low_energy_mev {
        return invalid_data(
            DMDW_EGRID_INFO_PATH,
            "high_energy_mev",
            "high energy must be greater than or equal to low energy",
        );
    }
    if data.step_mev <= 0.0 {
        return invalid_data(DMDW_EGRID_INFO_PATH, "step_mev", "step must be positive");
    }
    if data.characteristic_energy_mev <= 0.0 {
        return invalid_data(
            DMDW_EGRID_INFO_PATH,
            "characteristic_energy_mev",
            "w0 must be positive",
        );
    }
    Ok(())
}

pub(super) fn validate_dmdw_spectral_info(data: &DmdwSpectralInfoData) -> Result<()> {
    validate_finite_field(DMDW_SPECTRAL_INFO_PATH, "gamma", data.gamma)?;
    validate_finite_field(
        DMDW_SPECTRAL_INFO_PATH,
        "effective_electron_energy",
        data.effective_electron_energy,
    )?;
    validate_complex_field(
        DMDW_SPECTRAL_INFO_PATH,
        "total_cumulant_derivative",
        data.total_cumulant_derivative,
    )?;
    validate_complex_field(
        DMDW_SPECTRAL_INFO_PATH,
        "quasiparticle_weight",
        data.quasiparticle_weight,
    )?;
    if data.gamma <= 0.0 {
        return invalid_data(DMDW_SPECTRAL_INFO_PATH, "gamma", "gamma must be positive");
    }
    Ok(())
}

pub(super) fn validate_dmdw_a2f_info(data: &DmdwA2fInfoData) -> Result<()> {
    if data.lanczos_order == 0 {
        return invalid_data(
            DMDW_A2F_INFO_PATH,
            "lanczos_order",
            "Lanczos order must be positive",
        );
    }
    if data.pole_count() == 0 {
        return invalid_data(
            DMDW_A2F_INFO_PATH,
            "rows",
            "at least one Lanczos pole row is required",
        );
    }
    validate_length(
        DMDW_A2F_INFO_PATH,
        "lanczos_weight",
        data.lanczos_weight.len(),
        data.pole_count(),
    )?;
    validate_length(
        DMDW_A2F_INFO_PATH,
        "pole_energy_ev",
        data.pole_energy_ev.len(),
        data.pole_count(),
    )?;
    validate_length(
        DMDW_A2F_INFO_PATH,
        "pole_weight",
        data.pole_weight.len(),
        data.pole_count(),
    )?;
    validate_array(
        DMDW_A2F_INFO_PATH,
        "lanczos_frequency_thz",
        &data.lanczos_frequency_thz,
    )?;
    validate_array(DMDW_A2F_INFO_PATH, "lanczos_weight", &data.lanczos_weight)?;
    validate_array(DMDW_A2F_INFO_PATH, "pole_energy_ev", &data.pole_energy_ev)?;
    validate_array(DMDW_A2F_INFO_PATH, "pole_weight", &data.pole_weight)?;
    validate_finite_field(DMDW_A2F_INFO_PATH, "normalization", data.normalization)?;
    validate_finite_field(
        DMDW_A2F_INFO_PATH,
        "mass_enhancement",
        data.mass_enhancement,
    )?;
    validate_finite_field(
        DMDW_A2F_INFO_PATH,
        "characteristic_energy_ev",
        data.characteristic_energy_ev,
    )?;
    if data.normalization <= 0.0 {
        return invalid_data(
            DMDW_A2F_INFO_PATH,
            "normalization",
            "normalization must be positive",
        );
    }
    if data.characteristic_energy_ev <= 0.0 {
        return invalid_data(
            DMDW_A2F_INFO_PATH,
            "characteristic_energy_ev",
            "w0 must be positive",
        );
    }
    Ok(())
}

pub(super) fn validate_dmdw_self_energy_dat(data: &DmdwSelfEnergyDatData) -> Result<()> {
    validate_header_lines(DMDW_SELF_ENERGY_DAT_PATH, &data.header_lines)?;
    if data.point_count() == 0 {
        return invalid_data(
            DMDW_SELF_ENERGY_DAT_PATH,
            "rows",
            "at least one self-energy row is required",
        );
    }
    if data.value_ev.len() != data.point_count() {
        return invalid_data(
            DMDW_SELF_ENERGY_DAT_PATH,
            "value_ev",
            format!(
                "got {} value(s), expected {}",
                data.value_ev.len(),
                data.point_count()
            ),
        );
    }
    validate_array(DMDW_SELF_ENERGY_DAT_PATH, "energy", &data.energy_ev)?;
    validate_array(
        DMDW_SELF_ENERGY_DAT_PATH,
        "self-energy value",
        &data.value_ev,
    )
}

pub(super) fn validate_dmdw_akw_dat(data: &DmdwAkwDatData) -> Result<()> {
    if let Some(normalization) = data.normalization {
        validate_finite_field(DMDW_AKW_DAT_PATH, "normalization", normalization)?;
        if normalization <= 0.0 {
            return invalid_data(
                DMDW_AKW_DAT_PATH,
                "normalization",
                "normalization must be positive",
            );
        }
    }
    if data.point_count() == 0 {
        return invalid_data(
            DMDW_AKW_DAT_PATH,
            "rows",
            "at least one spectral-function row is required",
        );
    }
    validate_length(
        DMDW_AKW_DAT_PATH,
        "magnitude",
        data.magnitude.len(),
        data.point_count(),
    )?;
    validate_length(
        DMDW_AKW_DAT_PATH,
        "phase",
        data.phase.len(),
        data.point_count(),
    )?;
    validate_length(
        DMDW_AKW_DAT_PATH,
        "real",
        data.real.len(),
        data.point_count(),
    )?;
    validate_length(
        DMDW_AKW_DAT_PATH,
        "imaginary",
        data.imaginary.len(),
        data.point_count(),
    )?;
    validate_array(DMDW_AKW_DAT_PATH, "energy", &data.energy_mev)?;
    validate_array(DMDW_AKW_DAT_PATH, "magnitude", &data.magnitude)?;
    validate_array(DMDW_AKW_DAT_PATH, "phase", &data.phase)?;
    validate_array(DMDW_AKW_DAT_PATH, "real", &data.real)?;
    validate_array(DMDW_AKW_DAT_PATH, "imaginary", &data.imaginary)
}

fn validate_header_lines(path: &'static str, lines: &[String]) -> Result<()> {
    for (index, line) in lines.iter().enumerate() {
        if line.contains(['\n', '\r']) {
            return invalid_data(
                path,
                "header_lines",
                format!("header line {} contains an embedded newline", index + 1),
            );
        }
    }
    Ok(())
}

fn validate_length(
    path: &'static str,
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_data(
            path,
            field,
            format!("got {actual} value(s), expected {expected}"),
        )
    }
}

fn validate_array(path: &'static str, field: &'static str, values: &Array1<f64>) -> Result<()> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return invalid_data(
                path,
                field,
                format!("row {} value must be finite", index + 1),
            );
        }
    }
    Ok(())
}

fn validate_finite_field(path: &'static str, field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_data(path, field, "value must be finite")
    }
}

fn validate_complex_field(path: &'static str, field: &'static str, value: Complex64) -> Result<()> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        invalid_data(path, field, "complex value must be finite")
    }
}
