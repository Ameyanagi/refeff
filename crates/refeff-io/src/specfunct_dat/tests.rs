use super::*;
use ndarray::{Array1, Array2, Array3};
use refeff_core::{Real, SfconvSo2convXanesPreparation};

use super::validation::SPECFUNCT_DAT_SPECTRAL_ROWS;
use crate::chi_dat::ChiDatData;
use crate::error::Result;
use crate::sfconv_input::{
    SfconvSo2convFeffPathData, SfconvSo2convHeader, SfconvSo2convTargetData,
};
use crate::xmu_dat::XmuDatData;

#[test]
fn roundtrips_specfunct_dat_bytes() -> Result<()> {
    let data = sample_specfunct_data();
    let bytes = specfunct_dat_bytes(&data)?;
    let parsed = parse_specfunct_dat(&bytes)?;
    assert_eq!(parsed, data);
    Ok(())
}

#[test]
fn preserves_column_major_matrix_order() -> Result<()> {
    let data = sample_specfunct_data();
    let bytes = specfunct_dat_bytes(&data)?;
    let parsed = parse_specfunct_dat(&bytes)?;
    assert_eq!(parsed.spectral_info[[0, 1]], data.spectral_info[[0, 1]]);
    assert_eq!(
        parsed.extrinsic_quasiparticle[[2, 1]],
        data.extrinsic_quasiparticle[[2, 1]]
    );
    Ok(())
}

#[test]
fn builds_specfunct_data_from_spectral_rows() -> Result<()> {
    let momentum_count = 2;
    let spectral_count = 3;
    let pole_energy = Array1::from_vec(vec![0.25, 0.50, 0.75]);
    let pole_broadening = Array1::from_vec(vec![0.03, 0.04, 0.05]);
    let pole_weight = Array1::from_vec(vec![0.6, 0.3, 0.1]);
    let mut spectral_info = Array2::from_shape_fn(
        (momentum_count, SPECFUNCT_DAT_INFO_COLUMNS),
        |(row, col)| 0.125 * row as f64 + 0.01 * col as f64,
    );
    spectral_info[[0, 0]] = 0.45;
    spectral_info[[1, 0]] = 0.90;
    let weights = Array2::from_shape_fn(
        (momentum_count, SPECFUNCT_DAT_INFO_COLUMNS),
        |(row, col)| 0.05 + 0.02 * row as f64 + 0.01 * col as f64,
    );
    let spectral_function = Array3::from_shape_fn(
        (momentum_count, SPECFUNCT_DAT_SPECTRAL_ROWS, spectral_count),
        |(momentum, row, point)| 100.0 * momentum as f64 + 10.0 * row as f64 + point as f64,
    );
    let energy_grid = Array2::from_shape_fn((momentum_count, spectral_count), |(row, col)| {
        -2.0 + row as f64 + 0.25 * col as f64
    });

    let data = sfconv_specfunct_data_from_spectral_rows(SfconvSpecfunctSpectralRowsInput {
        wigner_seitz_radius: 2.15,
        core_hole_lifetime: 0.019,
        asymmetric_phase: 1,
        satellite_type: 2,
        low_q_mode: 0,
        pole_count: 2,
        pole_energy: pole_energy.view(),
        pole_broadening: pole_broadening.view(),
        pole_weight: pole_weight.view(),
        spectral_info: spectral_info.view(),
        weights: weights.view(),
        spectral_function: spectral_function.view(),
        energy_grid: energy_grid.view(),
    })?;

    assert_eq!(data.pole_energy, pole_energy);
    assert_eq!(data.spectral_info, spectral_info);
    assert_eq!(data.weights, weights);
    assert_eq!(data.energy_grid, energy_grid);
    assert_eq!(
        data.extrinsic_quasiparticle[[1, 2]],
        spectral_function[[1, 0, 2]]
    );
    assert_eq!(
        data.extrinsic_satellite[[0, 1]],
        spectral_function[[0, 1, 1]]
    );
    assert_eq!(
        data.interference_quasiparticle[[1, 0]],
        spectral_function[[1, 2, 0]]
    );
    assert_eq!(
        data.interference_satellite[[1, 1]],
        spectral_function[[1, 3, 1]]
    );
    assert_eq!(
        data.intrinsic_satellite[[0, 2]],
        spectral_function[[0, 4, 2]]
    );
    assert_eq!(
        data.clipped_extrinsic_satellite[[1, 2]],
        spectral_function[[1, 7, 2]]
    );

    let parsed = parse_specfunct_dat(&specfunct_dat_bytes(&data)?)?;
    assert_eq!(parsed, data);
    Ok(())
}

#[test]
fn rejects_invalid_specfunct_spectral_row_inputs() {
    let momentum_count = 2;
    let spectral_count = 3;
    let pole_energy = Array1::from_vec(vec![0.25, 0.50]);
    let pole_broadening = Array1::from_vec(vec![0.03, 0.04]);
    let pole_weight = Array1::from_vec(vec![0.6, 0.4]);
    let spectral_info = Array2::from_shape_fn(
        (momentum_count, SPECFUNCT_DAT_INFO_COLUMNS),
        |(row, col)| 0.1 + row as f64 + 0.01 * col as f64,
    );
    let weights = Array2::from_shape_fn(
        (momentum_count, SPECFUNCT_DAT_INFO_COLUMNS),
        |(row, col)| 0.05 + 0.02 * row as f64 + 0.01 * col as f64,
    );
    let spectral_function = Array3::from_shape_fn(
        (momentum_count, SPECFUNCT_DAT_SPECTRAL_ROWS, spectral_count),
        |(momentum, row, point)| 100.0 * momentum as f64 + 10.0 * row as f64 + point as f64,
    );
    let energy_grid = Array2::from_shape_fn((momentum_count, spectral_count), |(row, col)| {
        -2.0 + row as f64 + 0.25 * col as f64
    });
    let assemble = |spectral_info: &Array2<f64>,
                    weights: &Array2<f64>,
                    spectral_function: &Array3<f64>,
                    energy_grid: &Array2<f64>| {
        sfconv_specfunct_data_from_spectral_rows(SfconvSpecfunctSpectralRowsInput {
            wigner_seitz_radius: 2.15,
            core_hole_lifetime: 0.019,
            asymmetric_phase: 1,
            satellite_type: 2,
            low_q_mode: 0,
            pole_count: 2,
            pole_energy: pole_energy.view(),
            pole_broadening: pole_broadening.view(),
            pole_weight: pole_weight.view(),
            spectral_info: spectral_info.view(),
            weights: weights.view(),
            spectral_function: spectral_function.view(),
            energy_grid: energy_grid.view(),
        })
    };

    let short_weights = Array2::zeros((momentum_count, SPECFUNCT_DAT_INFO_COLUMNS - 1));
    assert!(
        assemble(
            &spectral_info,
            &short_weights,
            &spectral_function,
            &energy_grid
        )
        .is_err()
    );

    let short_spectral_function = Array3::zeros((
        momentum_count,
        SPECFUNCT_DAT_SPECTRAL_ROWS - 1,
        spectral_count,
    ));
    assert!(
        assemble(
            &spectral_info,
            &weights,
            &short_spectral_function,
            &energy_grid
        )
        .is_err()
    );

    let long_energy_grid = Array2::zeros((momentum_count + 1, spectral_count));
    assert!(
        assemble(
            &spectral_info,
            &weights,
            &spectral_function,
            &long_energy_grid
        )
        .is_err()
    );

    let mut nonfinite_spectral_function = spectral_function.clone();
    nonfinite_spectral_function[[1, 6, 2]] = f64::NAN;
    assert!(
        assemble(
            &spectral_info,
            &weights,
            &nonfinite_spectral_function,
            &energy_grid
        )
        .is_err()
    );
}

#[test]
fn rejects_invalid_specfunct_dat_bytes() -> Result<()> {
    assert!(parse_specfunct_dat(&[]).is_err());

    let mut bytes = specfunct_dat_bytes(&sample_specfunct_data())?;
    bytes[0] = 0;
    assert!(parse_specfunct_dat(&bytes).is_err());

    let truncated = &bytes[..bytes.len() - 1];
    assert!(parse_specfunct_dat(truncated).is_err());
    Ok(())
}

#[test]
fn rejects_invalid_specfunct_shapes() {
    let mut data = sample_specfunct_data();
    data.pole_broadening = Array1::from_vec(vec![0.1, 0.2]);
    assert!(specfunct_dat_bytes(&data).is_err());

    let mut data = sample_specfunct_data();
    data.weights = Array2::zeros((data.momentum_count(), SPECFUNCT_DAT_INFO_COLUMNS - 1));
    assert!(specfunct_dat_bytes(&data).is_err());
}

#[test]
fn checks_so2conv_cache_compatibility() -> Result<()> {
    let data = sample_specfunct_data();
    let input = SfconvSpecfunctCompatibilityInput {
        wigner_seitz_radius: data.wigner_seitz_radius,
        core_hole_lifetime: data.core_hole_lifetime,
        asymmetric_phase: data.asymmetric_phase,
        satellite_type: data.satellite_type,
        low_q_mode: data.low_q_mode,
        pole_count: data.pole_count,
        pole_energy: data.pole_energy.view(),
        pole_broadening: data.pole_broadening.view(),
        pole_weight: data.pole_weight.view(),
        momentum_grid: data.spectral_info.column(0),
    };
    assert!(sfconv_specfunct_matches_so2conv_inputs(&data, input)?);

    let rounded_weight = &data.pole_weight + 1.0e-14;
    let rounded_header = SfconvSpecfunctCompatibilityInput {
        core_hole_lifetime: data.core_hole_lifetime + 1.0e-14,
        pole_weight: rounded_weight.view(),
        ..input
    };
    assert!(sfconv_specfunct_matches_so2conv_inputs(
        &data,
        rounded_header
    )?);

    let changed = SfconvSpecfunctCompatibilityInput {
        core_hole_lifetime: data.core_hole_lifetime + 1.0e-3,
        ..input
    };
    assert!(!sfconv_specfunct_matches_so2conv_inputs(&data, changed)?);
    Ok(())
}

#[test]
fn builds_momentum_interpolation_input_from_cache() -> Result<()> {
    let data = sample_specfunct_data();
    let input = sfconv_specfunct_momentum_interpolation_input(&data, 0.75)?;
    assert_eq!(input.momentum_grid, data.spectral_info.column(0));
    assert_eq!(input.self_energy_real, data.spectral_info.column(3));
    assert_eq!(input.energy_grid, data.energy_grid.view());
    assert_eq!(input.weights, data.weights.view());
    Ok(())
}

#[test]
fn interpolates_cached_spectral_row_to_momentum() -> Result<()> {
    let data = sample_specfunct_data();
    let interpolated = sfconv_specfunct_interpolate_momentum(&data, 0.75)?;

    assert_eq!(interpolated.energy.len(), data.spectral_point_count());
    assert_eq!(
        interpolated.spectral_function.nrows(),
        SPECFUNCT_DAT_INFO_COLUMNS
    );
    assert!(interpolated.self_energy_real > data.spectral_info[[0, 3]]);
    assert!(interpolated.self_energy_real < data.spectral_info[[1, 3]]);
    assert!(interpolated.weights[0] > data.weights[[0, 0]]);
    assert!(interpolated.weights[0] < data.weights[[1, 0]]);
    Ok(())
}

#[test]
fn convolves_exafs_rows_from_cache() -> Result<()> {
    let mut data = sample_specfunct_data();
    data.asymmetric_phase = 0;
    let input = sample_exafs_input(24);
    let momentum = Array1::from_vec(vec![0.75, 1.25, 1.75]);

    let rows = sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
        cache: &data,
        signal_energy: input.signal_energy.view(),
        real_signal: input.real_signal.view(),
        imaginary_signal: input.imaginary_signal.view(),
        original_magnitude: input.original_magnitude.view(),
        original_phase: input.original_phase.view(),
        phase_minus_2kr: input.phase_minus_2kr.view(),
        photoelectron_momentum: momentum.view(),
        active_len: momentum.len(),
        chemical_potential: 0.0,
        cutoff: false,
        plasma_frequency: 1.0,
    })?;

    assert_eq!(rows.len(), momentum.len());
    for row in rows {
        assert!(row.real.is_finite());
        assert!(row.imaginary.is_finite());
        assert!(row.magnitude.is_finite());
        assert!(row.output_phase.is_finite());
    }
    Ok(())
}

#[test]
fn rejects_invalid_exafs_row_inputs() {
    let mut data = sample_specfunct_data();
    let input = sample_exafs_input(4);
    let momentum = Array1::from_vec(vec![0.75]);

    assert!(
        sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
            cache: &data,
            signal_energy: input.signal_energy.view(),
            real_signal: input.real_signal.view(),
            imaginary_signal: input.imaginary_signal.view(),
            original_magnitude: input.original_magnitude.view(),
            original_phase: input.original_phase.view(),
            phase_minus_2kr: input.phase_minus_2kr.view(),
            photoelectron_momentum: momentum.view(),
            active_len: 2,
            chemical_potential: 0.0,
            cutoff: false,
            plasma_frequency: 1.0,
        })
        .is_err()
    );

    data.asymmetric_phase = 1;
    assert!(
        sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
            cache: &data,
            signal_energy: input.signal_energy.view(),
            real_signal: input.real_signal.view(),
            imaginary_signal: input.imaginary_signal.view(),
            original_magnitude: input.original_magnitude.view(),
            original_phase: input.original_phase.view(),
            phase_minus_2kr: input.phase_minus_2kr.view(),
            photoelectron_momentum: momentum.view(),
            active_len: 1,
            chemical_potential: 0.0,
            cutoff: false,
            plasma_frequency: 1.0,
        })
        .is_err()
    );
}

#[test]
fn builds_convoluted_chi_data_from_cache() -> Result<()> {
    let mut data = sample_specfunct_data();
    data.asymmetric_phase = 0;
    let source = sample_chi_dat(24);
    let momentum = Array1::from_vec(
        (0..source.point_count())
            .map(|row| 0.75 + 0.01 * row as f64)
            .collect(),
    );

    let output = sfconv_specfunct_chi_data_from_cache(SfconvSpecfunctChiDataInput {
        cache: &data,
        source: &source,
        material: sample_so2conv_material(),
        photoelectron_momentum: momentum.view(),
        work_len: 28,
    })?;

    assert_eq!(output.point_count(), source.point_count());
    assert_eq!(output.header_lines, source.header_lines);
    assert_eq!(output.wave_number, source.wave_number);
    assert!(output.phase_minus_2kr.is_some());
    assert!(output.ckp_real.is_none());
    assert!(output.ckp_imag.is_none());
    assert!(output.chi.iter().all(|value| value.is_finite()));
    assert!(output.magnitude.iter().all(|value| value.is_finite()));
    assert!(output.phase.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn rejects_invalid_chi_cache_convolution_inputs() {
    let mut data = sample_specfunct_data();
    data.asymmetric_phase = 0;
    let source = sample_chi_dat(24);
    let short_momentum = Array1::from_vec(vec![0.75]);
    let momentum = Array1::from_vec(vec![0.75; source.point_count()]);

    assert!(
        sfconv_specfunct_chi_data_from_cache(SfconvSpecfunctChiDataInput {
            cache: &data,
            source: &source,
            material: sample_so2conv_material(),
            photoelectron_momentum: short_momentum.view(),
            work_len: 28,
        })
        .is_err()
    );
    assert!(
        sfconv_specfunct_chi_data_from_cache(SfconvSpecfunctChiDataInput {
            cache: &data,
            source: &source,
            material: sample_so2conv_material(),
            photoelectron_momentum: momentum.view(),
            work_len: 2,
        })
        .is_err()
    );
    data.asymmetric_phase = 1;
    assert!(
        sfconv_specfunct_chi_data_from_cache(SfconvSpecfunctChiDataInput {
            cache: &data,
            source: &source,
            material: sample_so2conv_material(),
            photoelectron_momentum: momentum.view(),
            work_len: 28,
        })
        .is_err()
    );
}

#[test]
fn builds_convoluted_feff_path_data_from_cache() -> Result<()> {
    let mut data = sample_specfunct_data();
    data.asymmetric_phase = 0;
    let source = sample_feff_path_data(24);
    let momentum = Array1::from_vec((0..24).map(|row| 0.75 + 0.01 * row as f64).collect());

    let output = sfconv_specfunct_feff_path_data_from_cache(SfconvSpecfunctFeffPathDataInput {
        cache: &data,
        source: &source,
        material: sample_so2conv_material(),
        photoelectron_momentum: momentum.view(),
        work_len: momentum.len(),
    })?;

    assert_eq!(output.point_count(), source.point_count());
    assert_eq!(output.header_lines, source.header_lines);
    assert_eq!(
        output.wave_number_inverse_angstrom,
        source.wave_number_inverse_angstrom
    );
    assert_eq!(output.effective_amplitude, source.effective_amplitude);
    assert_eq!(output.effective_phase, source.effective_phase);
    assert_eq!(
        output.mean_free_path_angstrom,
        source.mean_free_path_angstrom
    );
    assert_eq!(
        output.real_momentum_inverse_angstrom,
        source.real_momentum_inverse_angstrom
    );
    assert!(output.central_phase.iter().all(|value| value.is_finite()));
    assert!(
        output
            .reduction_factor
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn rejects_invalid_feff_path_cache_convolution_inputs() {
    let mut data = sample_specfunct_data();
    data.asymmetric_phase = 0;
    let source = sample_feff_path_data(6);
    let short_momentum = Array1::from_vec(vec![0.75]);
    let momentum = Array1::from_vec(vec![0.75; 6]);

    assert!(
        sfconv_specfunct_feff_path_data_from_cache(SfconvSpecfunctFeffPathDataInput {
            cache: &data,
            source: &source,
            material: sample_so2conv_material(),
            photoelectron_momentum: short_momentum.view(),
            work_len: 6,
        })
        .is_err()
    );

    let mut shifted_source = source.clone();
    shifted_source.wave_number_inverse_angstrom[0] = 0.05;
    assert!(
        sfconv_specfunct_feff_path_data_from_cache(SfconvSpecfunctFeffPathDataInput {
            cache: &data,
            source: &shifted_source,
            material: sample_so2conv_material(),
            photoelectron_momentum: momentum.view(),
            work_len: 6,
        })
        .is_err()
    );

    let mut short_grid = source.clone();
    short_grid.wave_number_inverse_angstrom[5] = 0.20;
    assert!(
        sfconv_specfunct_feff_path_data_from_cache(SfconvSpecfunctFeffPathDataInput {
            cache: &data,
            source: &short_grid,
            material: sample_so2conv_material(),
            photoelectron_momentum: momentum.view(),
            work_len: 6,
        })
        .is_err()
    );
}

#[test]
fn dispatches_convoluted_target_data_from_cache() -> Result<()> {
    let mut cache = sample_specfunct_data();
    cache.asymmetric_phase = 0;
    let header = sample_so2conv_header();
    let xmu = SfconvSo2convTargetData::Xmu {
        header,
        data: sample_xmu_dat(24),
    };
    let chi = SfconvSo2convTargetData::Chi {
        header,
        data: sample_chi_dat(24),
    };
    let path = SfconvSo2convTargetData::FeffPath {
        header,
        data: sample_feff_path_data(24),
    };
    let momentum = Array1::from_vec((0..24).map(|row| 0.75 + 0.01 * row as f64).collect());

    let xmu_output = sfconv_specfunct_target_data_from_cache(SfconvSpecfunctTargetDataInput {
        cache: &cache,
        source: &xmu,
        photoelectron_momentum: momentum.view(),
        work_len: 28,
    })?;
    let chi_output = sfconv_specfunct_target_data_from_cache(SfconvSpecfunctTargetDataInput {
        cache: &cache,
        source: &chi,
        photoelectron_momentum: momentum.view(),
        work_len: 28,
    })?;
    let path_output = sfconv_specfunct_target_data_from_cache(SfconvSpecfunctTargetDataInput {
        cache: &cache,
        source: &path,
        photoelectron_momentum: momentum.view(),
        work_len: momentum.len(),
    })?;

    assert!(matches!(
        xmu_output,
        SfconvSo2convTargetData::Xmu { header, .. } if header.already_convoluted
    ));
    assert!(matches!(
        chi_output,
        SfconvSo2convTargetData::Chi { header, .. } if header.already_convoluted
    ));
    assert!(matches!(
        path_output,
        SfconvSo2convTargetData::FeffPath { header, .. } if header.already_convoluted
    ));
    Ok(())
}

#[test]
fn convolves_xanes_rows_from_cache() -> Result<()> {
    let mut data = sample_specfunct_data();
    data.asymmetric_phase = 0;
    let prepared = sample_xanes_preparation(24);
    let momentum = Array1::from_vec(vec![0.75, 1.25, 1.75]);

    let rows = sfconv_specfunct_xanes_convolution_rows(SfconvSpecfunctXanesRowsInput {
        cache: &data,
        prepared: &prepared,
        photoelectron_momentum: momentum.view(),
        active_len: momentum.len(),
        chemical_potential: 0.0,
        cutoff: false,
        plasma_frequency: 1.0,
    })?;

    assert_eq!(rows.len(), momentum.len());
    for row in rows {
        assert!(row.absorption.is_finite());
        assert!(row.embedded_background.is_finite());
        assert!(row.fine_structure.is_finite());
    }
    Ok(())
}

#[test]
fn rejects_invalid_xanes_row_inputs() {
    let data = sample_specfunct_data();
    let prepared = sample_xanes_preparation(4);
    let momentum = Array1::from_vec(vec![0.75]);

    assert!(
        sfconv_specfunct_xanes_convolution_rows(SfconvSpecfunctXanesRowsInput {
            cache: &data,
            prepared: &prepared,
            photoelectron_momentum: momentum.view(),
            active_len: 2,
            chemical_potential: 0.0,
            cutoff: false,
            plasma_frequency: 1.0,
        })
        .is_err()
    );
    assert!(
        sfconv_specfunct_xanes_convolution_rows(SfconvSpecfunctXanesRowsInput {
            cache: &data,
            prepared: &prepared,
            photoelectron_momentum: momentum.view(),
            active_len: 0,
            chemical_potential: 0.0,
            cutoff: false,
            plasma_frequency: 1.0,
        })
        .is_err()
    );
}

#[test]
fn builds_convoluted_xmu_data_from_cache() -> Result<()> {
    let mut data = sample_specfunct_data();
    data.asymmetric_phase = 1;
    let source = sample_xmu_dat(24);
    let momentum = Array1::from_vec(
        (0..source.point_count())
            .map(|row| 0.75 + 0.01 * row as f64)
            .collect(),
    );

    let output = sfconv_specfunct_xmu_data_from_cache(SfconvSpecfunctXmuDataInput {
        cache: &data,
        source: &source,
        material: sample_so2conv_material(),
        photoelectron_momentum: momentum.view(),
        work_len: 28,
    })?;

    assert_eq!(output.point_count(), source.point_count());
    assert_eq!(output.header_lines, source.header_lines);
    assert_eq!(output.photon_energy_ev, source.photon_energy_ev);
    assert_eq!(output.relative_energy_ev, source.relative_energy_ev);
    assert_eq!(output.wave_number, source.wave_number);
    assert!(output.mu.iter().all(|value| value.is_finite()));
    assert!(output.mu0.iter().all(|value| value.is_finite()));
    assert!(output.chi.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn rejects_invalid_xmu_cache_convolution_inputs() {
    let data = sample_specfunct_data();
    let source = sample_xmu_dat(24);
    let short_momentum = Array1::from_vec(vec![0.75]);
    let momentum = Array1::from_vec(vec![0.75; source.point_count()]);

    assert!(
        sfconv_specfunct_xmu_data_from_cache(SfconvSpecfunctXmuDataInput {
            cache: &data,
            source: &source,
            material: sample_so2conv_material(),
            photoelectron_momentum: short_momentum.view(),
            work_len: 28,
        })
        .is_err()
    );
    assert!(
        sfconv_specfunct_xmu_data_from_cache(SfconvSpecfunctXmuDataInput {
            cache: &data,
            source: &source,
            material: sample_so2conv_material(),
            photoelectron_momentum: momentum.view(),
            work_len: 20,
        })
        .is_err()
    );
}

fn sample_specfunct_data() -> SfconvSpecfunctData {
    let momentum_count = 3;
    let spectral_count = 2;
    let pole_capacity = 4;
    let mut spectral_info = Array2::from_shape_fn(
        (momentum_count, SPECFUNCT_DAT_INFO_COLUMNS),
        |(row, col)| row as f64 + 0.125 * col as f64,
    );
    for row in 0..momentum_count {
        spectral_info[[row, 0]] = 0.25 + row as f64;
    }

    SfconvSpecfunctData {
        wigner_seitz_radius: 2.05,
        core_hole_lifetime: 0.031,
        asymmetric_phase: 1,
        satellite_type: 2,
        low_q_mode: 0,
        pole_count: 3,
        pole_energy: Array1::from_vec((0..pole_capacity).map(|index| 0.5 + index as f64).collect()),
        pole_broadening: Array1::from_vec(
            (0..pole_capacity)
                .map(|index| 0.05 + 0.01 * index as f64)
                .collect(),
        ),
        pole_weight: Array1::from_vec(
            (0..pole_capacity)
                .map(|index| 1.0 / (index + 1) as f64)
                .collect(),
        ),
        spectral_info,
        weights: Array2::from_shape_fn(
            (momentum_count, SPECFUNCT_DAT_INFO_COLUMNS),
            |(row, col)| 0.01 * row as f64 + 0.02 * col as f64,
        ),
        extrinsic_quasiparticle: spectral_table(momentum_count, spectral_count, 10.0),
        extrinsic_satellite: spectral_table(momentum_count, spectral_count, 20.0),
        interference_quasiparticle: spectral_table(momentum_count, spectral_count, 30.0),
        interference_satellite: spectral_table(momentum_count, spectral_count, 40.0),
        intrinsic_satellite: spectral_table(momentum_count, spectral_count, 50.0),
        clipped_extrinsic_satellite: spectral_table(momentum_count, spectral_count, 60.0),
        energy_grid: spectral_table(momentum_count, spectral_count, 70.0),
    }
}

fn sample_so2conv_header() -> SfconvSo2convHeader {
    SfconvSo2convHeader {
        material: sample_so2conv_material(),
        already_convoluted: false,
    }
}

fn sample_feff_path_data(len: usize) -> SfconvSo2convFeffPathData {
    SfconvSo2convFeffPathData {
        header_lines: vec![
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 Mu= 18.76000 kf= 1.230000"
                .to_string(),
            "# path 1 reff 4 2.0000 2.5000".to_string(),
            " ------------------------------------------------------------------------------"
                .to_string(),
        ],
        leg_count: 4,
        degeneracy: 2.0,
        effective_half_path_length_angstrom: 2.5,
        wave_number_inverse_angstrom: Array1::from_shape_fn(len, |row| 0.05 * row as f64),
        central_phase: Array1::from_shape_fn(len, |row| 0.1 + 0.01 * row as f64),
        effective_amplitude: Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64),
        effective_phase: Array1::from_shape_fn(len, |row| 0.2 + 0.01 * row as f64),
        reduction_factor: Array1::from_shape_fn(len, |row| 0.9 + 0.001 * row as f64),
        mean_free_path_angstrom: Array1::from_shape_fn(len, |row| 8.0 + 0.05 * row as f64),
        real_momentum_inverse_angstrom: Array1::from_shape_fn(len, |row| 0.05 * row as f64),
    }
}

fn sample_chi_dat(len: usize) -> ChiDatData {
    let wave_number = Array1::from_shape_fn(len, |row| 0.2 + 0.02 * row as f64);
    let magnitude = Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64);
    let phase = Array1::from_shape_fn(len, |row| 0.1 + 0.03 * row as f64);
    ChiDatData {
        header_lines: vec![
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 Mu= 18.76000 kf= 1.230000"
                .to_string(),
            " ------------------------------------------------------------------------------"
                .to_string(),
        ],
        wave_number,
        chi: Array1::from_shape_fn(len, |row| 0.01 * row as f64),
        magnitude,
        phase: phase.clone(),
        phase_minus_2kr: Some(Array1::from_shape_fn(len, |row| {
            phase[row] - 0.04 * row as f64
        })),
        ckp_real: None,
        ckp_imag: None,
    }
}

fn sample_xmu_dat(len: usize) -> XmuDatData {
    XmuDatData {
        header_lines: vec![
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 Mu= 18.76000 kf= 1.230000"
                .to_string(),
            " ------------------------------------------------------------------------------"
                .to_string(),
        ],
        normalization: Some(1.0),
        photon_energy_ev: Array1::from_shape_fn(len, |row| 100.0 + 2.0 * row as f64),
        relative_energy_ev: Array1::from_shape_fn(len, |row| 1.0 + 2.0 * row as f64),
        wave_number: Array1::from_shape_fn(len, |row| 0.2 + 0.01 * row as f64),
        mu: Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64),
        mu0: Array1::from_shape_fn(len, |row| 0.8 + 0.01 * row as f64),
        chi: Array1::from_shape_fn(len, |row| 0.2 + 0.01 * row as f64),
    }
}

fn sample_so2conv_material() -> refeff_core::SfconvSo2convMaterialInput {
    refeff_core::SfconvSo2convMaterialInput {
        core_hole_width_ev: 1.729,
        wigner_seitz_radius: 2.05,
        interstitial_potential_ev: 12.34,
        chemical_potential_ev: 18.76,
        fermi_wave_number_inv_angstrom: 1.23,
    }
}

fn spectral_table(momentum_count: usize, spectral_count: usize, base: f64) -> Array2<f64> {
    Array2::from_shape_fn((momentum_count, spectral_count), |(row, col)| {
        base + row as f64 + 0.1 * col as f64
    })
}

struct SampleExafsInput {
    signal_energy: Array1<Real>,
    real_signal: Array1<Real>,
    imaginary_signal: Array1<Real>,
    original_magnitude: Array1<Real>,
    original_phase: Array1<Real>,
    phase_minus_2kr: Array1<Real>,
}

fn sample_exafs_input(len: usize) -> SampleExafsInput {
    let signal_energy = Array1::from_shape_fn(len, |row| row as f64 * 0.05);
    let real_signal = Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64);
    let imaginary_signal = Array1::from_shape_fn(len, |row| 0.4 + 0.01 * row as f64);
    let original_magnitude = Array1::from_shape_fn(len, |row| {
        let real = real_signal[row];
        let imaginary = imaginary_signal[row];
        (real * real + imaginary * imaginary).sqrt()
    });
    let original_phase =
        Array1::from_shape_fn(len, |row| imaginary_signal[row].atan2(real_signal[row]));
    let phase_minus_2kr = Array1::from_shape_fn(len, |row| original_phase[row] - 0.02 * row as f64);

    SampleExafsInput {
        signal_energy,
        real_signal,
        imaginary_signal,
        original_magnitude,
        original_phase,
        phase_minus_2kr,
    }
}

fn sample_xanes_preparation(len: usize) -> SfconvSo2convXanesPreparation {
    let excitation_energy = Array1::from_shape_fn(len, |row| row as f64 * 5.0);
    let absorption = Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64);
    let embedded_background = Array1::from_shape_fn(len, |row| 0.8 + 0.01 * row as f64);
    let imaginary_fine_structure = &absorption - &embedded_background;

    SfconvSo2convXanesPreparation {
        incident_energy: Array1::from_shape_fn(len, |row| 100.0 + row as f64 * 5.0),
        excitation_energy,
        absorption,
        embedded_background,
        imaginary_fine_structure,
        real_fine_structure: Array1::from_shape_fn(len, |row| 0.1 + 0.005 * row as f64),
    }
}
