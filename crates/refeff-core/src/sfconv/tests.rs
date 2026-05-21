use ndarray::{Array1, Array2, ShapeBuilder, array};

use crate::Real;

use super::{
    SFCONV_MKSPECTF_GRID_LEN, SFCONV_SO2CONV_MOMENTUM_GRID_LEN, SfconvAdaptiveIntegral,
    SfconvBroadenedSelfEnergyBranch, SfconvBroadenedSelfEnergyDerivativeIntegrands,
    SfconvBroadenedSelfEnergyIntegrandInput, SfconvBroadenedSelfEnergyIntegrands,
    SfconvConvolutionInput, SfconvError, SfconvExafsConvolutionInput,
    SfconvExponentialReductionInput, SfconvExtrinsicSatelliteInput, SfconvExtrinsicSatelliteMode,
    SfconvExtrinsicSatelliteSplitInput, SfconvFeffPathInterpolationInput,
    SfconvFeffPathSignalInput, SfconvKramersKronigInput, SfconvMomentumSpectralInterpolation,
    SfconvMomentumSpectralInterpolationInput, SfconvPathAverageInput,
    SfconvPhotoelectronMomentumInput, SfconvPole, SfconvQLimits,
    SfconvQuasiparticleInterferenceInput, SfconvQuasiparticlePeakInput,
    SfconvQuasiparticlePoleInput, SfconvQuasiparticleTableInput, SfconvRenormalization,
    SfconvSatelliteContext, SfconvSatelliteCorrectionInput, SfconvSatellitePoleContributionsInput,
    SfconvSatelliteSelfEnergy, SfconvSatelliteTableInput, SfconvSelfEnergyContext,
    SfconvSo2convExafsEnergyPaddingInput, SfconvSo2convExafsPreparationInput,
    SfconvSo2convMaterialInput, SfconvSo2convMaterialParameters, SfconvSo2convSelfEnergyGridInput,
    SfconvSo2convSelfEnergySampleInput, SfconvSo2convXanesPreparationInput,
    SfconvSpectralCellInput, SfconvSpectralEnergyGrid, SfconvSpectralFinalizationInput,
    SfconvSpectralInterpolationInput, SfconvSpectralTableInput, SfconvSpectralWeightsInput,
    SfconvXanesConvolutionInput, sfconv_broadened_self_energy,
    sfconv_broadened_self_energy_derivative, sfconv_broadened_self_energy_derivative_integrands,
    sfconv_broadened_self_energy_integrands, sfconv_convolve, sfconv_correct_satellite_weights,
    sfconv_coupling_potential_squared, sfconv_exafs_convolution, sfconv_exponential_reduction,
    sfconv_extrinsic_beta, sfconv_extrinsic_satellite, sfconv_extrinsic_satellite_broadened,
    sfconv_extrinsic_satellite_debroadened, sfconv_feff_path_signal,
    sfconv_finalize_spectral_table, sfconv_find_singularities, sfconv_free_electron_exchange,
    sfconv_grater_integrate, sfconv_imaginary_self_energy, sfconv_imaginary_self_energy_derivative,
    sfconv_interference_quasiparticle, sfconv_interference_quasiparticle_integrand,
    sfconv_interference_satellite, sfconv_interference_satellite_integrand,
    sfconv_interpolate_feff_path, sfconv_interpolate_momentum_spectral_function,
    sfconv_interpolate_spectral_function, sfconv_intrinsic_satellite,
    sfconv_intrinsic_satellite_integrand, sfconv_inverse_pole_dispersion,
    sfconv_kramers_kronig_real_part, sfconv_path_average, sfconv_plasma_parameters,
    sfconv_plasmon_threshold_momentum, sfconv_pole_dispersion, sfconv_pole_dispersion_derivative,
    sfconv_pole_dispersion_second_derivative, sfconv_q_limits,
    sfconv_quasiparticle_interference_amplitude, sfconv_quasiparticle_main_peak,
    sfconv_quasiparticle_pole, sfconv_quasiparticle_table, sfconv_real_self_energy,
    sfconv_real_self_energy_derivative, sfconv_real_self_energy_derivative_integrand_lower,
    sfconv_real_self_energy_derivative_integrand_middle,
    sfconv_real_self_energy_derivative_integrand_upper, sfconv_real_self_energy_integrand_lower,
    sfconv_real_self_energy_integrand_middle, sfconv_real_self_energy_integrand_upper,
    sfconv_satellite_pole_contributions, sfconv_satellite_table, sfconv_select_pole,
    sfconv_self_energy_renormalization, sfconv_so2conv_broadened_self_energy_grid,
    sfconv_so2conv_broadened_self_energy_sample, sfconv_so2conv_material_parameters,
    sfconv_so2conv_momentum_grid, sfconv_so2conv_pad_exafs_energy_grid,
    sfconv_so2conv_photoelectron_momentum, sfconv_so2conv_prepare_exafs_signal,
    sfconv_so2conv_prepare_xanes_signal, sfconv_so2conv_unbroadened_self_energy_grid,
    sfconv_so2conv_unbroadened_self_energy_sample, sfconv_spectral_cell,
    sfconv_spectral_energy_grid, sfconv_spectral_table, sfconv_spectral_weights,
    sfconv_split_extrinsic_satellite, sfconv_xanes_convolution,
};

mod broadened;
mod convolution;
mod plasma_so2conv;
mod self_energy;
mod signal;
mod spectral;

fn mkrmu_reference_inputs(count: usize) -> (Array1<Real>, Array1<Real>, Array1<Real>) {
    let indices = (1..=count).map(|index| index as Real);
    let imaginary = Array1::from_iter(
        indices
            .clone()
            .map(|index| (0.17 * index).sin() + 0.01 * index),
    );
    let reference_imaginary =
        Array1::from_iter(indices.clone().map(|index| 0.2 * (0.11 * index).cos()));
    let energy = Array1::from_iter((0..count).map(|index| {
        let index = index as Real;
        0.05 * index + 0.002 * index * index
    }));
    (imaginary, reference_imaginary, energy)
}

fn plset_reference_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
    let energy = Array1::from_shape_fn(5, |index| {
        let i = index as Real + 1.0;
        0.12 * i + 0.015 * i * i
    });
    let weight = Array1::from_shape_fn(5, |index| {
        let i = index as Real + 1.0;
        0.25 + 0.07 * i
    });
    let broadening = Array1::from_shape_fn(5, |index| {
        let i = index as Real + 1.0;
        0.01 * i + 0.002 * i * i
    });
    (energy, weight, broadening)
}

fn interpsf_reference_inputs() -> (Array1<Real>, Array2<Real>) {
    let count = 110usize;
    let energy = Array1::from_shape_fn(count, |index| {
        let i = index as Real;
        -2.0 + 0.018 * i + 0.000_11 * i * i
    });
    let spectral_function = Array2::from_shape_fn((8, count).f(), |(row, column)| {
        let fortran_row = row as Real + 1.0;
        let i = column as Real;
        0.03 * fortran_row + 0.002 * i + 0.000_4 * fortran_row * i + 0.000_01 * i * i
    });
    (energy, spectral_function)
}

struct SfconvSubReference {
    spectral_energy: Array1<Real>,
    spectral_function: Array1<Real>,
    signal_energy: Array1<Real>,
    signal: Array1<Real>,
    weights: Array1<Real>,
}

fn sfconvsub_reference_inputs() -> SfconvSubReference {
    SfconvSubReference {
        spectral_energy: array![-0.18, -0.04, 0.0, 0.12, 0.31, 0.55, 0.82],
        spectral_function: array![0.05, 0.18, 0.30, 0.23, 0.14, 0.07, 0.02],
        signal_energy: array![0.40, 0.72, 0.95, 1.22, 1.58, 1.95],
        signal: array![0.62, 0.82, 0.74, 0.48, 0.22, 0.12],
        weights: array![0.72, 0.18, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

fn sfconv_reference_input<'a>(
    signal_energy: ndarray::ArrayView1<'a, Real>,
    signal: ndarray::ArrayView1<'a, Real>,
    spectral_energy: ndarray::ArrayView1<'a, Real>,
    spectral_function: ndarray::ArrayView1<'a, Real>,
    weights: ndarray::ArrayView1<'a, Real>,
) -> SfconvConvolutionInput<'a> {
    SfconvConvolutionInput {
        photoelectron_energy: 1.35,
        chemical_potential: 0.15,
        core_hole_lifetime: 0.08,
        signal_energy,
        signal,
        spectral_energy,
        spectral_function,
        weights,
        asymmetric_phase: false,
        cutoff: true,
        plasma_frequency: 0.55,
    }
}

fn mkspectf_quasiparticle_peak_input(
    grid: &SfconvSpectralEnergyGrid,
    index_1based: usize,
) -> SfconvQuasiparticlePeakInput {
    let index = index_1based - 1;
    SfconvQuasiparticlePeakInput {
        center_energy: grid.energy[index],
        lower_boundary: grid.boundaries[index],
        upper_boundary: grid.boundaries[index + 1],
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.93 + 0.08 * 0.06,
        quasiparticle_width: 0.08 * 0.82,
        plasma_frequency: 0.62,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
    }
}

fn mkspectf_quasiparticle_table_grid() -> (Array1<Real>, Array1<Real>) {
    let energy = array![-0.40, -0.12, -0.01, 0.02, 0.20, 0.55];
    let boundaries = array![-0.55, -0.25, -0.05, 0.005, 0.10, 0.36, 0.80];
    (energy, boundaries)
}

struct MkspectfSatelliteTableInputs {
    main_peak: Array1<Real>,
    quasiparticle_interference: Array1<Real>,
    extrinsic: Array1<Real>,
    interference: Array1<Real>,
    intrinsic: Array1<Real>,
    boundaries: Array1<Real>,
}

fn mkspectf_satellite_table_inputs() -> MkspectfSatelliteTableInputs {
    let main_peak = array![
        0.144_118_631_068_914_32,
        0.796_854_020_052_775_2,
        3.306_037_878_829_96,
        2.944_827_731_705_054,
        0.351_606_691_790_681_77,
        0.027_414_131_538_569_52,
    ];
    let quasiparticle_interference = array![
        0.031_993_167_546_517_99,
        0.176_895_131_355_183_62,
        0.733_913_602_898_189_5,
        0.653_727_879_020_868,
        0.078_053_834_660_399_79,
        0.006_085_714_920_760_973,
    ];
    let extrinsic = array![0.04, 0.09, -0.02, 0.18, 0.13, 0.07];
    let interference = array![0.01, 0.025, 0.006, 0.055, 0.04, 0.015];
    let intrinsic = array![0.02, 0.035, 0.012, 0.08, 0.065, 0.025];
    let boundaries = array![-0.55, -0.25, -0.05, 0.005, 0.10, 0.36, 0.80];
    MkspectfSatelliteTableInputs {
        main_peak,
        quasiparticle_interference,
        extrinsic,
        interference,
        intrinsic,
        boundaries,
    }
}

fn mkspectf_extrinsic_split_inputs() -> (Array2<Real>, Array1<Real>, Array1<Real>) {
    let mut spectral_function = Array2::<Real>::zeros((8, 8).f());
    for (row, values) in [
        (1, [0.10, 0.18, 0.35, 0.30, 0.22, 0.15, 0.25, 0.20]),
        (4, [0.02, 0.05, 0.11, 0.16, 0.13, 0.09, 0.12, 0.07]),
        (6, [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
        (7, [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
    ] {
        for (column, value) in values.into_iter().enumerate() {
            spectral_function[(row, column)] = value;
        }
    }
    let energy = array![-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0];
    let boundaries = array![-0.75, -0.45, -0.20, -0.05, 0.05, 0.20, 0.45, 0.80, 1.20];
    (spectral_function, energy, boundaries)
}

fn mkspectf_satellite_correction_inputs() -> (Array2<Real>, Array1<Real>) {
    let mut spectral_function = Array2::<Real>::zeros((8, 6).f());
    for (row, values) in [
        (1, [0.40, 0.18, 0.06, 0.50, 0.28, 0.08]),
        (3, [0.10, 0.16, 0.08, 0.35, 0.05, 0.03]),
        (4, [0.05, 0.04, 0.20, 0.03, 0.30, 0.20]),
        (6, [0.08, 0.05, 0.03, 0.12, 0.07, 0.02]),
        (7, [0.04, 0.02, 0.01, 0.06, 0.09, 0.03]),
    ] {
        for (column, value) in values.into_iter().enumerate() {
            spectral_function[(row, column)] = value;
        }
    }
    let boundaries = array![-0.4, -0.2, 0.0, 0.15, 0.35, 0.7, 1.1];
    (spectral_function, boundaries)
}

struct So2convMomentumSpectralInputs {
    momentum_grid: Array1<Real>,
    energy_grid: Array2<Real>,
    extrinsic_quasiparticle: Array2<Real>,
    extrinsic_satellite: Array2<Real>,
    interference_quasiparticle: Array2<Real>,
    interference_satellite: Array2<Real>,
    intrinsic_satellite: Array2<Real>,
    clipped_extrinsic_satellite: Array2<Real>,
    weights: Array2<Real>,
    self_energy_real: Array1<Real>,
    energy_correction: Array1<Real>,
    width: Array1<Real>,
    renormalization_real: Array1<Real>,
    renormalization_imag: Array1<Real>,
}

fn so2conv_momentum_spectral_inputs() -> So2convMomentumSpectralInputs {
    So2convMomentumSpectralInputs {
        momentum_grid: array![0.50, 1.00, 2.00, 4.00],
        energy_grid: array![
            [0.11, 0.12, 0.13, 0.14],
            [0.21, 0.22, 0.23, 0.24],
            [0.31, 0.32, 0.33, 0.34],
            [0.41, 0.42, 0.43, 0.44],
        ],
        extrinsic_quasiparticle: array![
            [1.11, 1.12, 1.13, 1.14],
            [1.21, 1.22, 1.23, 1.24],
            [1.31, 1.32, 1.33, 1.34],
            [1.41, 1.42, 1.43, 1.44],
        ],
        extrinsic_satellite: array![
            [2.22, 2.24, 2.26, 2.28],
            [2.42, 2.44, 2.46, 2.48],
            [2.62, 2.64, 2.66, 2.68],
            [2.82, 2.84, 2.86, 2.88],
        ],
        interference_quasiparticle: array![
            [3.33, 3.36, 3.39, 3.42],
            [3.63, 3.66, 3.69, 3.72],
            [3.93, 3.96, 3.99, 4.02],
            [4.23, 4.26, 4.29, 4.32],
        ],
        interference_satellite: array![
            [0.444, 0.448, 0.452, 0.456],
            [0.484, 0.488, 0.492, 0.496],
            [0.524, 0.528, 0.532, 0.536],
            [0.564, 0.568, 0.572, 0.576],
        ],
        intrinsic_satellite: array![
            [0.555, 0.560, 0.565, 0.570],
            [0.605, 0.610, 0.615, 0.620],
            [0.655, 0.660, 0.665, 0.670],
            [0.705, 0.710, 0.715, 0.720],
        ],
        clipped_extrinsic_satellite: array![
            [0.666, 0.672, 0.678, 0.684],
            [0.726, 0.732, 0.738, 0.744],
            [0.786, 0.792, 0.798, 0.804],
            [0.846, 0.852, 0.858, 0.864],
        ],
        weights: array![
            [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
            [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
            [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38],
            [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48],
        ],
        self_energy_real: array![41.0, 42.0, 43.0, 44.0],
        energy_correction: array![51.0, 52.0, 53.0, 54.0],
        width: array![61.0, 62.0, 63.0, 64.0],
        renormalization_real: array![71.0, 72.0, 73.0, 74.0],
        renormalization_imag: array![81.0, 82.0, 83.0, 84.0],
    }
}

fn so2conv_momentum_spectral_input<'a>(
    inputs: &'a So2convMomentumSpectralInputs,
    photoelectron_momentum: Real,
) -> SfconvMomentumSpectralInterpolationInput<'a> {
    SfconvMomentumSpectralInterpolationInput {
        photoelectron_momentum,
        momentum_grid: inputs.momentum_grid.view(),
        energy_grid: inputs.energy_grid.view(),
        extrinsic_quasiparticle: inputs.extrinsic_quasiparticle.view(),
        extrinsic_satellite: inputs.extrinsic_satellite.view(),
        interference_quasiparticle: inputs.interference_quasiparticle.view(),
        interference_satellite: inputs.interference_satellite.view(),
        intrinsic_satellite: inputs.intrinsic_satellite.view(),
        clipped_extrinsic_satellite: inputs.clipped_extrinsic_satellite.view(),
        weights: inputs.weights.view(),
        self_energy_real: inputs.self_energy_real.view(),
        energy_correction: inputs.energy_correction.view(),
        width: inputs.width.view(),
        renormalization_real: inputs.renormalization_real.view(),
        renormalization_imag: inputs.renormalization_imag.view(),
    }
}

fn so2conv_photoelectron_momentum_inputs() -> (Array1<Real>, Array1<Real>) {
    let momentum = array![0.0, 0.35, -0.40, 0.82, 1.10, 1.45];
    let self_energy = array![0.090, 0.105, 0.120, 0.150, 0.190, 0.250];
    (momentum, self_energy)
}

fn so2conv_self_energy_material() -> SfconvSo2convMaterialParameters {
    SfconvSo2convMaterialParameters {
        core_hole_lifetime: 0.03,
        interstitial_potential: 0.0,
        chemical_potential_offset: 0.20,
        fermi_wave_number: 1.0,
        fermi_momentum: 1.0,
        fermi_energy: 0.50,
        electron_concentration: 0.08,
        plasma_frequency: 0.70,
        dispersion_parameter: 0.33,
        initial_photoelectron_energy: 0.50,
        initial_photoelectron_momentum: 1.0,
        accuracy: 1.0e-4,
    }
}

fn so2conv_xanes_preparation_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>, Array1<Real>) {
    let count = 22;
    let incident_energy = Array1::from_shape_fn(count, |index| {
        let i = index as Real + 1.0;
        0.2 + 0.13 * (i - 1.0) + 0.002 * ((i as usize) % 3) as Real
    });
    let excitation_energy = Array1::from_shape_fn(count, |index| {
        let i = index as Real + 1.0;
        -0.4 + 0.11 * (i - 1.0) + 0.001 * ((i as usize) % 4) as Real
    });
    let embedded_background = Array1::from_shape_fn(count, |index| {
        let i = index as Real + 1.0;
        1.0 + 0.015 * (i - 1.0) + 0.0008 * ((i as usize) % 2) as Real
    });
    let absorption = Array1::from_shape_fn(count, |index| {
        let i = index as Real + 1.0;
        embedded_background[index] + 0.04 * (0.31 * i).sin() + 0.002 * (i - 1.0)
    });
    (
        incident_energy,
        excitation_energy,
        absorption,
        embedded_background,
    )
}

struct So2convFeffPathInterpolationInputs {
    source_momentum: Array1<Real>,
    path_momentum: Array1<Real>,
    central_phase: Array1<Real>,
    effective_amplitude: Array1<Real>,
    effective_phase: Array1<Real>,
    reduction_factor: Array1<Real>,
    mean_free_path: Array1<Real>,
    interpolated_central_phase: Array1<Real>,
    interpolated_effective_amplitude: Array1<Real>,
    interpolated_effective_phase: Array1<Real>,
    interpolated_reduction_factor: Array1<Real>,
    interpolated_mean_free_path: Array1<Real>,
}

fn so2conv_feff_path_interpolation_inputs() -> So2convFeffPathInterpolationInputs {
    So2convFeffPathInterpolationInputs {
        source_momentum: array![0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
        path_momentum: array![0.25, 0.75, 1.25, 1.75],
        central_phase: array![0.10, 0.20, 0.10, 0.30],
        effective_amplitude: array![1.00, 1.40, 1.10, 1.80],
        effective_phase: array![0.50, 0.70, 0.60, 1.00],
        reduction_factor: array![0.80, 0.90, 0.85, 0.95],
        mean_free_path: array![6.00, 7.00, 8.00, 9.00],
        interpolated_central_phase: array![0.0, 0.10, 0.15, 0.20, 0.15, 0.10, 0.20, 0.30, 0.0],
        interpolated_effective_amplitude: array![
            0.0, 1.00, 1.20, 1.40, 1.25, 1.10, 1.45, 1.80, 0.0
        ],
        interpolated_effective_phase: array![0.0, 0.50, 0.60, 0.70, 0.65, 0.60, 0.80, 1.00, 0.0],
        interpolated_reduction_factor: array![0.0, 0.80, 0.85, 0.90, 0.875, 0.85, 0.90, 0.95, 0.0],
        interpolated_mean_free_path: array![0.0, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 0.0],
    }
}

fn so2conv_path_average_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
    let source_momentum = array![0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25];
    let amplitude_reduction = array![0.82, 0.84, 0.88, 0.91, 0.89, 0.86, 0.83];
    let phase_shift = array![0.05, 0.08, 0.13, 0.17, 0.14, 0.09, 0.02];
    (source_momentum, amplitude_reduction, phase_shift)
}

fn assert_close(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn assert_real_slice_close(actual: &Array1<Real>, expected: &[Real], tolerance: Real) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close(actual, expected, tolerance);
    }
}

fn assert_so2conv_material_close(
    actual: SfconvSo2convMaterialParameters,
    expected: SfconvSo2convMaterialParameters,
    tolerance: Real,
) {
    assert_close(
        actual.core_hole_lifetime,
        expected.core_hole_lifetime,
        tolerance,
    );
    assert_close(
        actual.interstitial_potential,
        expected.interstitial_potential,
        tolerance,
    );
    assert_close(
        actual.chemical_potential_offset,
        expected.chemical_potential_offset,
        tolerance,
    );
    assert_close(
        actual.fermi_wave_number,
        expected.fermi_wave_number,
        tolerance,
    );
    assert_close(actual.fermi_momentum, expected.fermi_momentum, tolerance);
    assert_close(actual.fermi_energy, expected.fermi_energy, tolerance);
    assert_close(
        actual.electron_concentration,
        expected.electron_concentration,
        tolerance,
    );
    assert_close(
        actual.plasma_frequency,
        expected.plasma_frequency,
        tolerance,
    );
    assert_close(
        actual.dispersion_parameter,
        expected.dispersion_parameter,
        tolerance,
    );
    assert_close(
        actual.initial_photoelectron_energy,
        expected.initial_photoelectron_energy,
        tolerance,
    );
    assert_close(
        actual.initial_photoelectron_momentum,
        expected.initial_photoelectron_momentum,
        tolerance,
    );
    assert_close(actual.accuracy, expected.accuracy, tolerance);
}

fn assert_momentum_spectral_close(
    actual: &SfconvMomentumSpectralInterpolation,
    expected_energy: &[Real; 4],
    expected_rows: &[[Real; 4]; 8],
    expected_weights: &[Real; 8],
    expected_self_energy: &[Real; 5],
) {
    assert_real_slice_close(&actual.energy, expected_energy, 1.0e-15);
    for (row, expected) in expected_rows.iter().enumerate() {
        assert_real_slice_close(
            &actual.spectral_function.row(row).to_owned(),
            expected,
            1.0e-15,
        );
    }
    assert_real_slice_close(&actual.weights, expected_weights, 1.0e-15);
    assert_close(actual.self_energy_real, expected_self_energy[0], 1.0e-15);
    assert_close(actual.energy_correction, expected_self_energy[1], 1.0e-15);
    assert_close(actual.width, expected_self_energy[2], 1.0e-15);
    assert_close(
        actual.renormalization_real,
        expected_self_energy[3],
        1.0e-15,
    );
    assert_close(
        actual.renormalization_imag,
        expected_self_energy[4],
        1.0e-15,
    );
}

fn assert_pole_close(actual: SfconvPole, expected: SfconvPole) {
    assert_close(actual.energy, expected.energy, 1.0e-15);
    assert_close(actual.weight, expected.weight, 1.0e-15);
    assert_close(actual.broadening, expected.broadening, 1.0e-15);
}

fn assert_q_limits_close(actual: SfconvQLimits, expected: SfconvQLimits, tolerance: Real) {
    assert_eq!(actual.count, expected.count);
    assert_close(actual.q1, expected.q1, tolerance);
    assert_close(actual.q2, expected.q2, tolerance);
    assert_close(actual.q3, expected.q3, tolerance);
}

fn assert_integral_close(
    actual: SfconvAdaptiveIntegral,
    expected: SfconvAdaptiveIntegral,
    tolerance: Real,
) {
    assert_close(actual.value, expected.value, tolerance);
    assert_close(
        actual.estimated_error,
        expected.estimated_error,
        tolerance.max(1.0e-12),
    );
    assert_eq!(actual.evaluations, expected.evaluations);
    assert_eq!(actual.max_regions, expected.max_regions);
}

fn mksat_reference_context() -> SfconvSatelliteContext {
    SfconvSatelliteContext {
        plasma_frequency: 0.62,
        pole_energy: 0.47,
        dispersion_parameter: 0.28,
        photoelectron_energy: 0.85,
        accuracy: 1.0e-4,
    }
}

fn mksat_reference_self_energy() -> SfconvSatelliteSelfEnergy {
    SfconvSatelliteSelfEnergy {
        on_shell_real: 0.12,
        width: 0.08,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
        off_shell_real: 0.03,
        off_shell_imag: 0.025,
    }
}

fn senergies_reference_context(include_below_fermi: bool) -> SfconvSelfEnergyContext {
    SfconvSelfEnergyContext {
        fermi_energy: 0.50,
        fermi_momentum: 1.00,
        plasma_frequency: 0.62,
        pole_energy: 0.47,
        quasiparticle_energy: 0.91,
        photoelectron_momentum: (2.0_f64 * 0.85).sqrt(),
        accuracy: 1.0e-4,
        pole_broadening: 0.035,
        dispersion_parameter: 0.28,
        include_below_fermi,
    }
}
