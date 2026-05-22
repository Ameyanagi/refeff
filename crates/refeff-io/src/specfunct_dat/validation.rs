use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use refeff_core::Real;

use crate::error::Result;

use super::support::invalid_specfunct_dat;
use super::types::{
    SPECFUNCT_DAT_INFO_COLUMNS, SfconvSpecfunctCompatibilityInput, SfconvSpecfunctData,
    SfconvSpecfunctExafsRowsInput, SfconvSpecfunctSpectralRowsInput, SfconvSpecfunctXanesRowsInput,
};

pub(super) const SPECFUNCT_DAT_SPECTRAL_ROWS: usize = 8;

pub(super) fn validate_specfunct_dat(data: &SfconvSpecfunctData) -> Result<()> {
    validate_finite_scalar(data.wigner_seitz_radius, "rs")?;
    validate_finite_scalar(data.core_hole_lifetime, "gammach")?;
    if data.pole_capacity() == 0 {
        return invalid_specfunct_dat("pole capacity must be positive");
    }
    if data.pole_count > data.pole_capacity() {
        return invalid_specfunct_dat(format!(
            "npl is {}, but pole capacity is {}",
            data.pole_count,
            data.pole_capacity()
        ));
    }
    validate_vector_shape(&data.pole_broadening, data.pole_capacity(), "plbrd")?;
    validate_vector_shape(&data.pole_weight, data.pole_capacity(), "plwt")?;
    validate_finite_vector(&data.pole_energy, "plengy")?;
    validate_finite_vector(&data.pole_broadening, "plbrd")?;
    validate_finite_vector(&data.pole_weight, "plwt")?;

    validate_info_shape(&data.spectral_info, "sfinfo")?;
    let momentum_count = data.momentum_count();
    validate_matrix_shape(
        &data.weights,
        momentum_count,
        SPECFUNCT_DAT_INFO_COLUMNS,
        "wgts",
    )?;
    let spectral_point_count = data.spectral_point_count();
    if spectral_point_count == 0 {
        return invalid_specfunct_dat("spectral table column count must be positive");
    }
    validate_spectral_table(
        &data.extrinsic_quasiparticle,
        momentum_count,
        spectral_point_count,
        "emsf",
    )?;
    validate_spectral_table(
        &data.extrinsic_satellite,
        momentum_count,
        spectral_point_count,
        "essf",
    )?;
    validate_spectral_table(
        &data.interference_quasiparticle,
        momentum_count,
        spectral_point_count,
        "xmsf",
    )?;
    validate_spectral_table(
        &data.interference_satellite,
        momentum_count,
        spectral_point_count,
        "xssf",
    )?;
    validate_spectral_table(
        &data.intrinsic_satellite,
        momentum_count,
        spectral_point_count,
        "xissf",
    )?;
    validate_spectral_table(
        &data.clipped_extrinsic_satellite,
        momentum_count,
        spectral_point_count,
        "escsf",
    )?;
    validate_spectral_table(
        &data.energy_grid,
        momentum_count,
        spectral_point_count,
        "engrid",
    )?;
    validate_finite_matrix(&data.spectral_info, "sfinfo")?;
    validate_finite_matrix(&data.weights, "wgts")?;
    Ok(())
}

pub(super) fn validate_specfunct_spectral_rows_input(
    input: SfconvSpecfunctSpectralRowsInput<'_>,
) -> Result<()> {
    validate_finite_scalar(input.wigner_seitz_radius, "rs")?;
    validate_finite_scalar(input.core_hole_lifetime, "gammach")?;
    if input.pole_energy.is_empty() {
        return invalid_specfunct_dat("pole capacity must be positive");
    }
    if input.pole_count > input.pole_energy.len() {
        return invalid_specfunct_dat("npl exceeds plengy length");
    }
    if input.pole_broadening.len() != input.pole_energy.len() {
        return invalid_specfunct_dat(format!(
            "plbrd length is {}, expected {}",
            input.pole_broadening.len(),
            input.pole_energy.len()
        ));
    }
    if input.pole_weight.len() != input.pole_energy.len() {
        return invalid_specfunct_dat(format!(
            "plwt length is {}, expected {}",
            input.pole_weight.len(),
            input.pole_energy.len()
        ));
    }

    let (momentum_count, info_cols) = input.spectral_info.dim();
    if momentum_count == 0 || info_cols != SPECFUNCT_DAT_INFO_COLUMNS {
        return invalid_specfunct_dat(format!(
            "sfinfo shape is {momentum_count}x{info_cols}, expected nonzero rows x {SPECFUNCT_DAT_INFO_COLUMNS}"
        ));
    }
    let weights_shape = input.weights.dim();
    if weights_shape != (momentum_count, SPECFUNCT_DAT_INFO_COLUMNS) {
        return invalid_specfunct_dat(format!(
            "wgts shape is {}x{}, expected {momentum_count}x{SPECFUNCT_DAT_INFO_COLUMNS}",
            weights_shape.0, weights_shape.1
        ));
    }

    let (spectral_momentum_count, spectral_rows, spectral_point_count) =
        input.spectral_function.dim();
    if spectral_momentum_count != momentum_count
        || spectral_rows != SPECFUNCT_DAT_SPECTRAL_ROWS
        || spectral_point_count == 0
    {
        return invalid_specfunct_dat(format!(
            "spectf shape is {spectral_momentum_count}x{spectral_rows}x{spectral_point_count}, expected {momentum_count}x{SPECFUNCT_DAT_SPECTRAL_ROWS}x nonzero columns"
        ));
    }
    let energy_shape = input.energy_grid.dim();
    if energy_shape != (momentum_count, spectral_point_count) {
        return invalid_specfunct_dat(format!(
            "engrid shape is {}x{}, expected {momentum_count}x{spectral_point_count}",
            energy_shape.0, energy_shape.1
        ));
    }

    validate_finite_view(input.pole_energy, "plengy")?;
    validate_finite_view(input.pole_broadening, "plbrd")?;
    validate_finite_view(input.pole_weight, "plwt")?;
    validate_finite_matrix_view(input.spectral_info, "sfinfo")?;
    validate_finite_matrix_view(input.weights, "wgts")?;
    validate_finite_spectral_view(input.spectral_function, "spectf")?;
    validate_finite_matrix_view(input.energy_grid, "engrid")?;
    Ok(())
}

pub(super) fn validate_compatibility_input(
    input: SfconvSpecfunctCompatibilityInput<'_>,
) -> Result<()> {
    validate_finite_scalar(input.wigner_seitz_radius, "input rs")?;
    validate_finite_scalar(input.core_hole_lifetime, "input gammach")?;
    if input.pole_count > input.pole_energy.len() {
        return invalid_specfunct_dat("input npl exceeds plengy length");
    }
    if input.pole_count > input.pole_broadening.len() {
        return invalid_specfunct_dat("input npl exceeds plbrd length");
    }
    if input.pole_count > input.pole_weight.len() {
        return invalid_specfunct_dat("input npl exceeds plwt length");
    }
    if input.momentum_grid.is_empty() {
        return invalid_specfunct_dat("input pgrid must not be empty");
    }
    validate_finite_view(input.pole_energy, "input plengy")?;
    validate_finite_view(input.pole_broadening, "input plbrd")?;
    validate_finite_view(input.pole_weight, "input plwt")?;
    validate_finite_view(input.momentum_grid, "input pgrid")?;
    Ok(())
}
pub(super) fn validate_specfunct_exafs_rows_input(
    input: SfconvSpecfunctExafsRowsInput<'_>,
) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_finite_scalar(input.chemical_potential, "exafs chemical potential")?;
    validate_finite_scalar(input.plasma_frequency, "exafs plasma frequency")?;
    if input.cache.asymmetric_phase != 0 {
        return invalid_specfunct_dat("EXAFS convolution requires an iasym=0 specfunct.dat cache");
    }
    if input.active_len == 0 {
        return invalid_specfunct_dat("EXAFS convolution active_len must be positive");
    }

    let signal_len = input.signal_energy.len();
    if signal_len < 2 {
        return invalid_specfunct_dat("EXAFS convolution signal length must be at least 2");
    }
    if input.active_len > signal_len {
        return invalid_specfunct_dat(format!(
            "EXAFS convolution active_len {} exceeds signal length {signal_len}",
            input.active_len
        ));
    }
    if input.active_len > input.photoelectron_momentum.len() {
        return invalid_specfunct_dat(format!(
            "EXAFS convolution active_len {} exceeds photoelectron momentum length {}",
            input.active_len,
            input.photoelectron_momentum.len()
        ));
    }

    validate_exafs_view(input.real_signal, signal_len, "exafs real signal")?;
    validate_exafs_view(input.imaginary_signal, signal_len, "exafs imaginary signal")?;
    validate_exafs_view(
        input.original_magnitude,
        signal_len,
        "exafs original magnitude",
    )?;
    validate_exafs_view(input.original_phase, signal_len, "exafs original phase")?;
    validate_exafs_view(input.phase_minus_2kr, signal_len, "exafs phase minus 2kr")?;
    validate_finite_view(input.signal_energy, "exafs signal energy")?;
    validate_finite_view(input.photoelectron_momentum, "exafs photoelectron momentum")?;
    Ok(())
}

fn validate_exafs_view(
    values: ArrayView1<'_, Real>,
    expected_len: usize,
    field: &'static str,
) -> Result<()> {
    if values.len() != expected_len {
        return invalid_specfunct_dat(format!(
            "{field} length {} does not match signal length {expected_len}",
            values.len()
        ));
    }
    validate_finite_view(values, field)
}
pub(super) fn validate_specfunct_xanes_rows_input(
    input: SfconvSpecfunctXanesRowsInput<'_>,
) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_finite_scalar(input.chemical_potential, "xanes chemical potential")?;
    validate_finite_scalar(input.plasma_frequency, "xanes plasma frequency")?;
    if input.active_len == 0 {
        return invalid_specfunct_dat("XANES convolution active_len must be positive");
    }

    let signal_len = input.prepared.excitation_energy.len();
    if signal_len < 2 {
        return invalid_specfunct_dat("XANES convolution signal length must be at least 2");
    }
    if input.active_len > signal_len {
        return invalid_specfunct_dat(format!(
            "XANES convolution active_len {} exceeds prepared signal length {signal_len}",
            input.active_len
        ));
    }
    if input.active_len > input.photoelectron_momentum.len() {
        return invalid_specfunct_dat(format!(
            "XANES convolution active_len {} exceeds photoelectron momentum length {}",
            input.active_len,
            input.photoelectron_momentum.len()
        ));
    }

    validate_vector_shape(
        &input.prepared.incident_energy,
        signal_len,
        "xanes incident energy",
    )?;
    validate_vector_shape(&input.prepared.absorption, signal_len, "xanes absorption")?;
    validate_vector_shape(
        &input.prepared.embedded_background,
        signal_len,
        "xanes embedded background",
    )?;
    validate_vector_shape(
        &input.prepared.imaginary_fine_structure,
        signal_len,
        "xanes imaginary fine structure",
    )?;
    validate_vector_shape(
        &input.prepared.real_fine_structure,
        signal_len,
        "xanes real fine structure",
    )?;
    validate_finite_view(
        input.prepared.incident_energy.view(),
        "xanes incident energy",
    )?;
    validate_finite_view(
        input.prepared.excitation_energy.view(),
        "xanes excitation energy",
    )?;
    validate_finite_view(input.prepared.absorption.view(), "xanes absorption")?;
    validate_finite_view(
        input.prepared.embedded_background.view(),
        "xanes embedded background",
    )?;
    validate_finite_view(
        input.prepared.imaginary_fine_structure.view(),
        "xanes imaginary fine structure",
    )?;
    validate_finite_view(
        input.prepared.real_fine_structure.view(),
        "xanes real fine structure",
    )?;
    validate_finite_view(input.photoelectron_momentum, "xanes photoelectron momentum")?;
    Ok(())
}
fn validate_info_shape(values: &Array2<f64>, field: &'static str) -> Result<()> {
    let (rows, cols) = values.dim();
    if rows == 0 || cols != SPECFUNCT_DAT_INFO_COLUMNS {
        return invalid_specfunct_dat(format!(
            "{field} shape is {rows}x{cols}, expected nonzero rows x {SPECFUNCT_DAT_INFO_COLUMNS}"
        ));
    }
    Ok(())
}

fn validate_spectral_table(
    values: &Array2<f64>,
    rows: usize,
    cols: usize,
    field: &'static str,
) -> Result<()> {
    validate_matrix_shape(values, rows, cols, field)?;
    validate_finite_matrix(values, field)
}

fn validate_vector_shape(
    values: &Array1<f64>,
    expected_len: usize,
    field: &'static str,
) -> Result<()> {
    if values.len() != expected_len {
        return invalid_specfunct_dat(format!(
            "{field} length is {}, expected {expected_len}",
            values.len()
        ));
    }
    Ok(())
}

fn validate_matrix_shape(
    values: &Array2<f64>,
    rows: usize,
    cols: usize,
    field: &'static str,
) -> Result<()> {
    let actual = values.dim();
    if actual != (rows, cols) {
        return invalid_specfunct_dat(format!(
            "{field} shape is {}x{}, expected {rows}x{cols}",
            actual.0, actual.1
        ));
    }
    Ok(())
}

pub(super) fn validate_finite_scalar(value: f64, field: &'static str) -> Result<()> {
    if !value.is_finite() {
        return invalid_specfunct_dat(format!("{field} must be finite"));
    }
    Ok(())
}

fn validate_finite_vector(values: &Array1<f64>, field: &'static str) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return invalid_specfunct_dat(format!("{field} row {} must be finite", index + 1));
        }
    }
    Ok(())
}

pub(super) fn validate_finite_view(values: ArrayView1<'_, f64>, field: &'static str) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return invalid_specfunct_dat(format!("{field} row {} must be finite", index + 1));
        }
    }
    Ok(())
}

fn validate_finite_matrix_view(values: ArrayView2<'_, f64>, field: &'static str) -> Result<()> {
    for ((row, col), value) in values.indexed_iter() {
        if !value.is_finite() {
            return invalid_specfunct_dat(format!(
                "{field} row {} column {} must be finite",
                row + 1,
                col + 1
            ));
        }
    }
    Ok(())
}

fn validate_finite_spectral_view(values: ArrayView3<'_, f64>, field: &'static str) -> Result<()> {
    for ((momentum, row, point), value) in values.indexed_iter() {
        if !value.is_finite() {
            return invalid_specfunct_dat(format!(
                "{field} momentum {} row {} column {} must be finite",
                momentum + 1,
                row + 1,
                point + 1
            ));
        }
    }
    Ok(())
}

fn validate_finite_matrix(values: &Array2<f64>, field: &'static str) -> Result<()> {
    for ((row, col), value) in values.indexed_iter() {
        if !value.is_finite() {
            return invalid_specfunct_dat(format!(
                "{field} row {} column {} must be finite",
                row + 1,
                col + 1
            ));
        }
    }
    Ok(())
}
