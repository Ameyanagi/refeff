use super::*;

pub(crate) fn exc_dat_bench_data() -> ExcDatData {
    let count = 128;
    ExcDatData {
        header_lines: vec![
            "#SN#   Section:    1".to_string(),
            "#DT#  Double Double Double Double".to_string(),
        ],
        energy_ev: Array1::from_shape_fn(count, |index| 5.0 + index as f64 * 0.25),
        broadening_ev: Array1::from_shape_fn(count, |index| 0.05 + index as f64 * 0.0005),
        oscillator_strength: Array1::from_shape_fn(count, |index| {
            0.1 + (index as f64 * 0.01).sin().abs()
        }),
        auxiliary_weight: Some(Array1::from_shape_fn(count, |index| {
            0.2 + index as f64 * 0.02
        })),
    }
}

pub(crate) fn so2conv_specfunct_bench_data() -> SfconvSpecfunctData {
    let momentum_count = SFCONV_SO2CONV_MOMENTUM_GRID_LEN;
    let spectral_count = SFCONV_MKSPECTF_GRID_LEN;
    let pole_capacity = 5_000;
    let mut spectral_info = Array2::from_shape_fn((momentum_count, 8), |(row, col)| {
        0.01 * row as f64 + 0.001 * col as f64
    });
    for row in 0..momentum_count {
        spectral_info[[row, 0]] = 0.05 + 0.02 * row as f64;
    }

    SfconvSpecfunctData {
        wigner_seitz_radius: 2.05,
        core_hole_lifetime: 0.03125,
        asymmetric_phase: 1,
        satellite_type: 0,
        low_q_mode: 0,
        pole_count: 8,
        pole_energy: Array1::from_shape_fn(pole_capacity, |index| 0.25 + 0.01 * index as f64),
        pole_broadening: Array1::from_shape_fn(pole_capacity, |index| 0.02 + 0.0001 * index as f64),
        pole_weight: Array1::from_shape_fn(pole_capacity, |index| 1.0 / (1.0 + index as f64)),
        spectral_info,
        weights: Array2::from_shape_fn((momentum_count, 8), |(row, col)| {
            0.1 + 0.001 * row as f64 + 0.01 * col as f64
        }),
        extrinsic_quasiparticle: so2conv_specfunct_table(momentum_count, spectral_count, 0.1),
        extrinsic_satellite: so2conv_specfunct_table(momentum_count, spectral_count, 0.2),
        interference_quasiparticle: so2conv_specfunct_table(momentum_count, spectral_count, 0.3),
        interference_satellite: so2conv_specfunct_table(momentum_count, spectral_count, 0.4),
        intrinsic_satellite: so2conv_specfunct_table(momentum_count, spectral_count, 0.5),
        clipped_extrinsic_satellite: so2conv_specfunct_table(momentum_count, spectral_count, 0.6),
        energy_grid: Array2::from_shape_fn((momentum_count, spectral_count), |(row, col)| {
            -2.0 + 0.05 * col as f64 + 0.001 * row as f64
        }),
    }
}

pub(crate) fn so2conv_specfunct_table(rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        scale + 0.0001 * row as f64 + 0.0002 * col as f64
    })
}

pub(crate) struct So2convExafsBenchData {
    pub(crate) signal_energy: Array1<f64>,
    pub(crate) real_signal: Array1<f64>,
    pub(crate) imaginary_signal: Array1<f64>,
    pub(crate) original_magnitude: Array1<f64>,
    pub(crate) original_phase: Array1<f64>,
    pub(crate) phase_minus_2kr: Array1<f64>,
}

pub(crate) fn so2conv_exafs_bench_data(len: usize) -> So2convExafsBenchData {
    let signal_energy = Array1::from_shape_fn(len, |row| row as f64 * 0.02);
    let real_signal = Array1::from_shape_fn(len, |row| 1.0 + 0.001 * row as f64);
    let imaginary_signal = Array1::from_shape_fn(len, |row| 0.35 + 0.0005 * row as f64);
    let original_magnitude = Array1::from_shape_fn(len, |row| {
        let real = real_signal[row];
        let imaginary = imaginary_signal[row];
        (real * real + imaginary * imaginary).sqrt()
    });
    let original_phase =
        Array1::from_shape_fn(len, |row| imaginary_signal[row].atan2(real_signal[row]));
    let phase_minus_2kr =
        Array1::from_shape_fn(len, |row| original_phase[row] - 0.005 * row as f64);

    So2convExafsBenchData {
        signal_energy,
        real_signal,
        imaginary_signal,
        original_magnitude,
        original_phase,
        phase_minus_2kr,
    }
}

pub(crate) fn so2conv_xanes_preparation_bench_data(len: usize) -> SfconvSo2convXanesPreparation {
    let excitation_energy = Array1::from_shape_fn(len, |row| row as f64 * 2.0);
    let absorption = Array1::from_shape_fn(len, |row| 1.0 + 0.001 * row as f64);
    let embedded_background = Array1::from_shape_fn(len, |row| 0.8 + 0.0005 * row as f64);
    let imaginary_fine_structure = &absorption - &embedded_background;

    SfconvSo2convXanesPreparation {
        incident_energy: Array1::from_shape_fn(len, |row| 100.0 + row as f64 * 2.0),
        excitation_energy,
        absorption,
        embedded_background,
        imaginary_fine_structure,
        real_fine_structure: Array1::from_shape_fn(len, |row| 0.1 + 0.0002 * row as f64),
    }
}
