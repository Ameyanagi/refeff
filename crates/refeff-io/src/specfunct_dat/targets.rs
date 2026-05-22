use ndarray::{Array1, ArrayView1};
use refeff_core::{
    Real, SFCONV_SO2CONV_BOHR_ANGSTROM, SFCONV_SO2CONV_HARTREE_EV, SfconvExafsConvolution,
    SfconvFeffPathInterpolationInput, SfconvFeffPathSignalInput, SfconvPathAverage,
    SfconvPathAverageInput, SfconvSo2convExafsPreparationInput, SfconvSo2convXanesPreparationInput,
    sfconv_feff_path_signal, sfconv_interpolate_feff_path, sfconv_path_average,
    sfconv_so2conv_material_parameters, sfconv_so2conv_prepare_exafs_signal,
    sfconv_so2conv_prepare_xanes_signal,
};

use crate::chi_dat::{ChiDatData, validate_chi_dat};
use crate::error::Result;
use crate::sfconv_input::{
    SfconvSo2convFeffPathData, SfconvSo2convHeader, SfconvSo2convTargetData,
    sfconv_so2conv_chi_data_from_convolution_rows, sfconv_so2conv_feff_path_data_from_averages,
    sfconv_so2conv_xmu_data_from_convolution_rows,
};
use crate::xmu_dat::{XmuDatData, validate_xmu_dat};

use super::rows::{
    sfconv_specfunct_exafs_convolution_rows, sfconv_specfunct_xanes_convolution_rows,
    specfunct_exafs_error, specfunct_xanes_error,
};
use super::support::invalid_specfunct_dat;
use super::types::{
    SfconvSpecfunctChiDataInput, SfconvSpecfunctExafsRowsInput, SfconvSpecfunctFeffPathDataInput,
    SfconvSpecfunctTargetDataInput, SfconvSpecfunctXanesRowsInput, SfconvSpecfunctXmuDataInput,
};
use super::validation::{validate_finite_scalar, validate_finite_view, validate_specfunct_dat};

pub fn sfconv_specfunct_chi_data_from_cache(
    input: SfconvSpecfunctChiDataInput<'_>,
) -> Result<ChiDatData> {
    validate_specfunct_chi_data_input(input)?;
    let material =
        sfconv_so2conv_material_parameters(input.material).map_err(specfunct_exafs_error)?;
    let momentum = input
        .source
        .wave_number
        .mapv(|value| value / SFCONV_SO2CONV_BOHR_ANGSTROM);
    let prepared = sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
        momentum: momentum.view(),
        magnitude: input.source.magnitude.view(),
        phase: input.source.phase.view(),
        phase_minus_2kr: input
            .source
            .phase_minus_2kr
            .as_ref()
            .map(|values| values.view()),
        chemical_potential: material.chemical_potential_offset,
        active_len: input.source.point_count(),
        output_len: input.work_len,
    })
    .map_err(specfunct_exafs_error)?;
    let rows = sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
        cache: input.cache,
        signal_energy: prepared.signal_energy.view(),
        real_signal: prepared.real_signal.view(),
        imaginary_signal: prepared.imaginary_signal.view(),
        original_magnitude: prepared.original_magnitude.view(),
        original_phase: prepared.original_phase.view(),
        phase_minus_2kr: prepared.phase_minus_2kr.view(),
        photoelectron_momentum: input.photoelectron_momentum,
        active_len: input.source.point_count(),
        chemical_potential: material.chemical_potential_offset,
        cutoff: true,
        plasma_frequency: material.plasma_frequency,
    })?;
    sfconv_so2conv_chi_data_from_convolution_rows(input.source, &rows)
}

/// Build a convolved `feffNNNN.dat` path table from a cached `specfunct.dat`.
///
/// FEFF `SO2CONV` first maps the coarse path table onto a dense 0.05
/// inverse-Angstrom grid, convolves that raw EXAFS path signal, then averages
/// the many-body amplitude and phase corrections back onto the original path
/// grid. This helper performs that cache-backed path assembly and preserves the
/// original path table columns except for FEFF's `redfac2` and `caph2`
/// corrections.
pub fn sfconv_specfunct_feff_path_data_from_cache(
    input: SfconvSpecfunctFeffPathDataInput<'_>,
) -> Result<SfconvSo2convFeffPathData> {
    validate_specfunct_feff_path_data_input(input)?;
    let material =
        sfconv_so2conv_material_parameters(input.material).map_err(specfunct_exafs_error)?;
    let source_momentum = sfconv_specfunct_uniform_path_momentum(input.work_len);
    let path_momentum = input
        .source
        .wave_number_inverse_angstrom
        .mapv(|value| value / SFCONV_SO2CONV_BOHR_ANGSTROM);
    let effective_amplitude = input
        .source
        .effective_amplitude
        .mapv(|value| value * SFCONV_SO2CONV_BOHR_ANGSTROM);
    let mean_free_path = input
        .source
        .mean_free_path_angstrom
        .mapv(|value| value * SFCONV_SO2CONV_BOHR_ANGSTROM);

    let interpolated = sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
        source_momentum: source_momentum.view(),
        path_momentum: path_momentum.view(),
        central_phase: input.source.central_phase.view(),
        effective_amplitude: effective_amplitude.view(),
        effective_phase: input.source.effective_phase.view(),
        reduction_factor: input.source.reduction_factor.view(),
        mean_free_path: mean_free_path.view(),
    })
    .map_err(specfunct_exafs_error)?;
    let signal = sfconv_feff_path_signal(SfconvFeffPathSignalInput {
        momentum: source_momentum.view(),
        central_phase: interpolated.central_phase.view(),
        effective_amplitude: interpolated.effective_amplitude.view(),
        effective_phase: interpolated.effective_phase.view(),
        reduction_factor: interpolated.reduction_factor.view(),
        mean_free_path: interpolated.mean_free_path.view(),
        degeneracy: input.source.degeneracy,
        half_path_length: input.source.effective_half_path_length_angstrom
            * SFCONV_SO2CONV_BOHR_ANGSTROM,
    })
    .map_err(specfunct_exafs_error)?;
    let signal_energy = source_momentum
        .mapv(|momentum| momentum.powi(2) / 2.0 + material.chemical_potential_offset);
    let rows = sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
        cache: input.cache,
        signal_energy: signal_energy.view(),
        real_signal: signal.real.view(),
        imaginary_signal: signal.imaginary.view(),
        original_magnitude: signal.magnitude.view(),
        original_phase: signal.phase.view(),
        phase_minus_2kr: signal.phase_minus_2kr.view(),
        photoelectron_momentum: input.photoelectron_momentum,
        active_len: input.work_len,
        chemical_potential: material.chemical_potential_offset,
        cutoff: true,
        plasma_frequency: material.plasma_frequency,
    })?;
    let averages =
        sfconv_specfunct_feff_path_averages(source_momentum.view(), path_momentum.view(), &rows)?;
    sfconv_so2conv_feff_path_data_from_averages(input.source, &averages)
}

/// Build a convolved SO2CONV target from a compatible cached `specfunct.dat`.
///
/// This dispatcher preserves the target variant and applies the matching
/// cache-backed assembly helper for `xmu.dat`, `chi.dat`/`chipNNNN.dat`, or
/// `feffNNNN.dat`. The returned header marks the target as already convoluted,
/// matching the marker that `write_sfconv_so2conv_convoluted_target_data`
/// writes to FEFF text files.
pub fn sfconv_specfunct_target_data_from_cache(
    input: SfconvSpecfunctTargetDataInput<'_>,
) -> Result<SfconvSo2convTargetData> {
    validate_specfunct_dat(input.cache)?;
    let output = match input.source {
        SfconvSo2convTargetData::Xmu { header, data } => SfconvSo2convTargetData::Xmu {
            header: specfunct_convoluted_header(*header),
            data: sfconv_specfunct_xmu_data_from_cache(SfconvSpecfunctXmuDataInput {
                cache: input.cache,
                source: data,
                material: header.material,
                photoelectron_momentum: input.photoelectron_momentum,
                work_len: input.work_len,
            })?,
        },
        SfconvSo2convTargetData::Chi { header, data } => SfconvSo2convTargetData::Chi {
            header: specfunct_convoluted_header(*header),
            data: sfconv_specfunct_chi_data_from_cache(SfconvSpecfunctChiDataInput {
                cache: input.cache,
                source: data,
                material: header.material,
                photoelectron_momentum: input.photoelectron_momentum,
                work_len: input.work_len,
            })?,
        },
        SfconvSo2convTargetData::FeffPath { header, data } => SfconvSo2convTargetData::FeffPath {
            header: specfunct_convoluted_header(*header),
            data: sfconv_specfunct_feff_path_data_from_cache(SfconvSpecfunctFeffPathDataInput {
                cache: input.cache,
                source: data,
                material: header.material,
                photoelectron_momentum: input.photoelectron_momentum,
                work_len: input.work_len,
            })?,
        },
    };
    Ok(output)
}

/// Build a convolved `xmu.dat` from a compatible cached `specfunct.dat`.
///
/// This helper performs the FEFF `SO2CONV` XANES unit handoff, pads the signal
/// arrays with the same endpoint rule as `so2conv.f90`, applies cached
/// spectral-function convolution rows, and returns an `xmu.dat` table with the
/// original energy and wave-number columns preserved.
pub fn sfconv_specfunct_xmu_data_from_cache(
    input: SfconvSpecfunctXmuDataInput<'_>,
) -> Result<XmuDatData> {
    validate_specfunct_xmu_data_input(input)?;
    let material =
        sfconv_so2conv_material_parameters(input.material).map_err(specfunct_xanes_error)?;
    let incident_energy = input
        .source
        .photon_energy_ev
        .mapv(|value| value / SFCONV_SO2CONV_HARTREE_EV);
    let excitation_energy = input
        .source
        .relative_energy_ev
        .mapv(|value| value / SFCONV_SO2CONV_HARTREE_EV);
    let prepared = sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
        incident_energy: incident_energy.view(),
        excitation_energy: excitation_energy.view(),
        absorption: input.source.mu.view(),
        embedded_background: input.source.mu0.view(),
        active_len: input.source.point_count(),
        output_len: input.work_len,
    })
    .map_err(specfunct_xanes_error)?;
    let rows = sfconv_specfunct_xanes_convolution_rows(SfconvSpecfunctXanesRowsInput {
        cache: input.cache,
        prepared: &prepared,
        photoelectron_momentum: input.photoelectron_momentum,
        active_len: input.source.point_count(),
        chemical_potential: material.chemical_potential_offset + material.interstitial_potential,
        cutoff: false,
        plasma_frequency: material.plasma_frequency,
    })?;
    sfconv_so2conv_xmu_data_from_convolution_rows(input.source, &rows)
}

/// Convolve EXAFS rows with cached SO2CONV spectral functions.
///
/// This is the row-level bridge from a reusable `specfunct.dat` cache to the
/// existing core EXAFS convolution kernels. The returned rows can be applied to
/// `chi.dat`/`chipNNNN.dat` with `sfconv_so2conv_chi_data_from_convolution_rows`
fn validate_specfunct_chi_data_input(input: SfconvSpecfunctChiDataInput<'_>) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_chi_dat(input.source)?;
    if input.source.point_count() < 2 {
        return invalid_specfunct_dat(
            "EXAFS chi.dat convolution requires at least two source rows",
        );
    }
    if input.work_len < input.source.point_count() {
        return invalid_specfunct_dat(format!(
            "EXAFS chi.dat convolution work_len {} is smaller than source row count {}",
            input.work_len,
            input.source.point_count()
        ));
    }
    if input.photoelectron_momentum.len() < input.source.point_count() {
        return invalid_specfunct_dat(format!(
            "EXAFS chi.dat convolution momentum count {} is smaller than source row count {}",
            input.photoelectron_momentum.len(),
            input.source.point_count()
        ));
    }
    validate_finite_view(
        input.photoelectron_momentum,
        "exafs chi.dat photoelectron momentum",
    )
}

fn validate_specfunct_feff_path_data_input(
    input: SfconvSpecfunctFeffPathDataInput<'_>,
) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_finite_scalar(input.source.degeneracy, "feff path degeneracy")?;
    validate_finite_scalar(
        input.source.effective_half_path_length_angstrom,
        "feff path half length",
    )?;
    if input.source.point_count() < 2 {
        return invalid_specfunct_dat(
            "EXAFS feffNNNN.dat convolution requires at least two source rows",
        );
    }
    if input.work_len < 3 {
        return invalid_specfunct_dat(format!(
            "EXAFS feffNNNN.dat convolution work_len {} is smaller than FEFF minimum 3",
            input.work_len
        ));
    }
    if input.photoelectron_momentum.len() < input.work_len {
        return invalid_specfunct_dat(format!(
            "EXAFS feffNNNN.dat convolution momentum count {} is smaller than work_len {}",
            input.photoelectron_momentum.len(),
            input.work_len
        ));
    }
    validate_finite_view(
        input.photoelectron_momentum,
        "exafs feffNNNN.dat photoelectron momentum",
    )?;
    validate_feff_path_view_lengths(input.source)?;
    validate_finite_view(
        input.source.wave_number_inverse_angstrom.view(),
        "feff path wave number",
    )?;
    validate_finite_view(input.source.central_phase.view(), "feff path central phase")?;
    validate_finite_view(
        input.source.effective_amplitude.view(),
        "feff path effective amplitude",
    )?;
    validate_finite_view(
        input.source.effective_phase.view(),
        "feff path effective phase",
    )?;
    validate_finite_view(
        input.source.reduction_factor.view(),
        "feff path reduction factor",
    )?;
    validate_finite_view(
        input.source.mean_free_path_angstrom.view(),
        "feff path mean free path",
    )?;
    validate_finite_view(
        input.source.real_momentum_inverse_angstrom.view(),
        "feff path real momentum",
    )?;
    validate_feff_path_uniform_grid_coverage(input)
}

fn validate_specfunct_xmu_data_input(input: SfconvSpecfunctXmuDataInput<'_>) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_xmu_dat(input.source)?;
    if input.source.point_count() < 2 {
        return invalid_specfunct_dat(
            "XANES xmu.dat convolution requires at least two source rows",
        );
    }
    if input.work_len < 21 {
        return invalid_specfunct_dat(format!(
            "XANES xmu.dat convolution work_len {} is smaller than FEFF minimum 21",
            input.work_len
        ));
    }
    if input.work_len < input.source.point_count() {
        return invalid_specfunct_dat(format!(
            "XANES xmu.dat convolution work_len {} is smaller than source row count {}",
            input.work_len,
            input.source.point_count()
        ));
    }
    if input.photoelectron_momentum.len() < input.source.point_count() {
        return invalid_specfunct_dat(format!(
            "XANES xmu.dat convolution momentum count {} is smaller than source row count {}",
            input.photoelectron_momentum.len(),
            input.source.point_count()
        ));
    }
    validate_finite_view(
        input.photoelectron_momentum,
        "xanes xmu.dat photoelectron momentum",
    )
}

fn specfunct_convoluted_header(mut header: SfconvSo2convHeader) -> SfconvSo2convHeader {
    header.already_convoluted = true;
    header
}

fn validate_feff_path_view_lengths(source: &SfconvSo2convFeffPathData) -> Result<()> {
    let point_count = source.point_count();
    validate_feff_path_view_len("caph2", source.central_phase.len(), point_count)?;
    validate_feff_path_view_len("xmfeff2", source.effective_amplitude.len(), point_count)?;
    validate_feff_path_view_len("phfeff2", source.effective_phase.len(), point_count)?;
    validate_feff_path_view_len("redfac2", source.reduction_factor.len(), point_count)?;
    validate_feff_path_view_len("xlam2", source.mean_free_path_angstrom.len(), point_count)?;
    validate_feff_path_view_len(
        "realck2",
        source.real_momentum_inverse_angstrom.len(),
        point_count,
    )
}

fn validate_feff_path_view_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    invalid_specfunct_dat(format!(
        "feff path {field} length {actual} does not match source row count {expected}"
    ))
}

fn validate_feff_path_uniform_grid_coverage(
    input: SfconvSpecfunctFeffPathDataInput<'_>,
) -> Result<()> {
    let first = input.source.wave_number_inverse_angstrom[0];
    if first > 0.0 {
        return invalid_specfunct_dat(format!(
            "feff path grid starts at {first}, but SO2CONV dense path grid starts at 0"
        ));
    }
    let last = input.source.wave_number_inverse_angstrom[input.source.point_count() - 1];
    let dense_max = 0.05 * (input.work_len - 1) as Real;
    if last < dense_max {
        return invalid_specfunct_dat(format!(
            "feff path grid ends at {last}, below SO2CONV dense grid maximum {dense_max}"
        ));
    }
    Ok(())
}

fn sfconv_specfunct_uniform_path_momentum(work_len: usize) -> Array1<Real> {
    Array1::from_shape_fn(work_len, |row| {
        0.05 * row as Real / SFCONV_SO2CONV_BOHR_ANGSTROM
    })
}

fn sfconv_specfunct_feff_path_averages(
    source_momentum: ArrayView1<'_, Real>,
    path_momentum: ArrayView1<'_, Real>,
    rows: &[SfconvExafsConvolution],
) -> Result<Vec<SfconvPathAverage>> {
    let amplitude_reduction = Array1::from_iter(rows.iter().map(|row| row.amplitude_reduction));
    let phase_shift = Array1::from_iter(rows.iter().map(|row| row.phase_shift));
    (0..path_momentum.len())
        .map(|row| {
            let previous = if row == 0 {
                path_momentum[row]
            } else {
                path_momentum[row - 1]
            };
            let next = if row + 1 == path_momentum.len() {
                path_momentum[row]
            } else {
                path_momentum[row + 1]
            };
            sfconv_path_average(SfconvPathAverageInput {
                source_momentum,
                amplitude_reduction: amplitude_reduction.view(),
                phase_shift: phase_shift.view(),
                previous_momentum: previous,
                center_momentum: path_momentum[row],
                next_momentum: next,
                momentum_step: 0.05,
            })
            .map_err(specfunct_exafs_error)
        })
        .collect()
}
