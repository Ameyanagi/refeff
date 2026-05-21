use ndarray::{Array1, Array2};

use super::support::*;
use super::*;

/// Port of FEFF `SFCONV/mkspectf.f90` renormalization from self-energy slopes.
///
/// FEFF forms `xrz = 1 - d(Re Sigma)/dE`, `xiz = -d(Im Sigma)/dE`, and
/// returns the reciprocal complex factor used for the quasiparticle peak and
/// satellite amplitudes.
pub fn sfconv_self_energy_renormalization(
    real_derivative: Real,
    imaginary_derivative: Real,
) -> Result<SfconvRenormalization, SfconvError> {
    validate_finite_scalar("self-energy real derivative", real_derivative)?;
    validate_finite_scalar("self-energy imaginary derivative", imaginary_derivative)?;

    let real_inverse = 1.0 - real_derivative;
    let imaginary_inverse = -imaginary_derivative;
    let denominator = real_inverse.powi(2) + imaginary_inverse.powi(2);
    validate_nonzero_denominator("self-energy renormalization", denominator)?;

    let real = finite_result("renormalization real", real_inverse / denominator)?;
    let imaginary = finite_result(
        "renormalization imaginary",
        -imaginary_inverse / denominator,
    )?;
    let magnitude = checked_hypot("renormalization magnitude", real, imaginary)?;
    Ok(SfconvRenormalization {
        real,
        imaginary,
        magnitude,
    })
}

/// Port of FEFF `SFCONV/mkspectf.f90` exponential pole-reduction factor.
///
/// FEFF accumulates `xa += 3*wt*(omp/ompl)**2/(8*sqrt(2*ompl))` over the
/// active epsilon-inverse poles and returns `exp(-xa)`.
pub fn sfconv_exponential_reduction(
    input: SfconvExponentialReductionInput<'_>,
) -> Result<Real, SfconvError> {
    validate_exponential_reduction_input(input)?;

    let exponent = (0..input.pole_count).try_fold(0.0, |total, index| {
        let pole_energy = input.pole_energy[index];
        let pole_weight = input.pole_weight[index];
        let denominator = 8.0 * checked_sqrt("exponential reduction pole", 2.0 * pole_energy)?;
        validate_nonzero_denominator("exponential reduction pole", denominator)?;
        finite_result(
            "exponential reduction exponent",
            total
                + 3.0 * pole_weight * (input.plasma_frequency / pole_energy).powi(2) / denominator,
        )
    })?;
    finite_result("exponential reduction", (-exponent).exp())
}

/// Port of FEFF `SFCONV/mkspectf.f90` quasiparticle pole refinement.
///
/// FEFF computes `qpengy = ekp + width*z1i` and `qpwidth = width*z1`
/// after the final on-shell self-energy derivative pass. The returned pole
/// feeds the finite-element quasiparticle peak rows.
pub fn sfconv_quasiparticle_pole(
    input: SfconvQuasiparticlePoleInput,
) -> Result<SfconvQuasiparticlePole, SfconvError> {
    validate_quasiparticle_pole_input(input)?;

    let energy = finite_result(
        "quasiparticle energy",
        input.photoelectron_energy + input.width * input.renormalization.imaginary,
    )?;
    let width = finite_result(
        "quasiparticle width",
        input.width * input.renormalization.real,
    )?;
    validate_positive_scalar("quasiparticle width", width)?;
    Ok(SfconvQuasiparticlePole { energy, width })
}

/// Port of the `SFCONV/mkspectf.f90` fixed spectral-function energy mesh.
///
/// FEFF uses 112 nonuniform offsets around the quasiparticle peak and a
/// companion `wlim(0:npts)` boundary array to integrate each cell. The mesh is
/// scaled by the plasma frequency, `omp`.
pub fn sfconv_spectral_energy_grid(
    plasma_frequency: Real,
) -> Result<SfconvSpectralEnergyGrid, SfconvError> {
    validate_positive_scalar("plasma_frequency", plasma_frequency)?;

    let mut energy = Array1::<Real>::zeros(SFCONV_MKSPECTF_GRID_LEN);
    let dw = plasma_frequency / 30.0;
    let iqph = 54;
    let iqpl = 53;

    energy[feff_index(iqph)] = dw * 1.0e-2;
    energy[feff_index(iqpl)] = -dw * 1.0e-2;
    energy[feff_index(iqph + 1)] = dw * 2.0e-2;
    energy[feff_index(iqpl - 1)] = -dw * 2.0e-2;
    for i in 1..=30 {
        let offset = i as Real;
        energy[feff_index(i + 1 + iqph)] = offset * dw;
        energy[feff_index(iqpl - 1 - i)] = -offset * dw;
    }
    for i in 1..=3 {
        let offset = i as Real;
        energy[feff_index(i + 31 + iqph)] = energy[feff_index(31 + iqph)] + offset * dw;
        energy[feff_index(iqpl - 31 - i)] = energy[feff_index(iqpl - 31)] - offset * dw;
    }
    for i in 1..=3 {
        let offset = i as Real;
        energy[feff_index(i + 34 + iqph)] = energy[feff_index(34 + iqph)] + 2.0 * offset * dw;
        energy[feff_index(iqpl - 34 - i)] = energy[feff_index(iqpl - 33)] - 2.0 * offset * dw;
    }
    for i in 1..=3 {
        let offset = i as Real;
        energy[feff_index(i + 37 + iqph)] = energy[feff_index(37 + iqph)] + 4.0 * offset * dw;
        energy[feff_index(iqpl - 37 - i)] = energy[feff_index(iqpl - 36)] - 4.0 * offset * dw;
    }
    for i in 1..=12 {
        let offset = i as Real;
        energy[feff_index(i + 40 + iqph)] = energy[feff_index(40 + iqph)] + 10.0 * offset * dw;
        energy[feff_index(iqpl - 40 - i)] = energy[feff_index(iqpl - 39)] - 10.0 * offset * dw;
    }
    for i in 1..=6 {
        let offset = i as Real;
        energy[feff_index(i + 52 + iqph)] = energy[feff_index(52 + iqph)] + 30.0 * offset * dw;
    }

    let mut boundaries = Array1::<Real>::zeros(SFCONV_MKSPECTF_GRID_LEN + 1);
    for index in 1..SFCONV_MKSPECTF_GRID_LEN {
        boundaries[index] = 0.5 * (energy[index - 1] + energy[index]);
    }
    boundaries[0] = 2.0 * energy[0] - energy[1];
    boundaries[SFCONV_MKSPECTF_GRID_LEN] =
        2.0 * energy[SFCONV_MKSPECTF_GRID_LEN - 1] - energy[SFCONV_MKSPECTF_GRID_LEN - 2];

    validate_finite_array("spectral energy grid", energy.view())?;
    validate_finite_array("spectral energy boundaries", boundaries.view())?;
    Ok(SfconvSpectralEnergyGrid { energy, boundaries })
}

/// Port of the `SFCONV/mkspectf.f90` extrinsic quasiparticle peak cell.
///
/// FEFF stores the quasiparticle peak as a finite-element average over
/// `wlim(i-1)..wlim(i)`. The real renormalization contributes the integrated
/// Lorentzian term, while the imaginary renormalization contributes FEFF's
/// logarithmic asymmetric term with the same Gaussian damping used in
/// `mkspectf`.
pub fn sfconv_quasiparticle_main_peak(
    input: SfconvQuasiparticlePeakInput,
) -> Result<Real, SfconvError> {
    validate_quasiparticle_peak_input(input)?;

    let bin_width = input.upper_boundary - input.lower_boundary;
    let upper_delta =
        input.upper_boundary - input.quasiparticle_energy + input.photoelectron_energy;
    let lower_delta =
        input.lower_boundary - input.quasiparticle_energy + input.photoelectron_energy;
    let pi = std::f64::consts::PI;
    let atan_term = input.renormalization_real
        * ((upper_delta / input.quasiparticle_width).atan()
            - (lower_delta / input.quasiparticle_width).atan())
        / (pi * bin_width);

    let upper_norm = input.quasiparticle_width.powi(2) + upper_delta.powi(2);
    let lower_norm = input.quasiparticle_width.powi(2) + lower_delta.powi(2);
    validate_positive_scalar("quasiparticle peak lower norm", lower_norm)?;
    let log_argument = upper_norm / lower_norm;
    validate_positive_scalar("quasiparticle peak logarithm", log_argument)?;

    let center_delta =
        input.center_energy + input.photoelectron_energy - input.quasiparticle_energy;
    let gaussian = (-(center_delta / (2.0 * input.plasma_frequency)).powi(2)).exp();
    let log_term =
        input.renormalization_imag * log_argument.ln() * gaussian / (2.0 * pi * bin_width);

    finite_result("quasiparticle main peak", atan_term - log_term)
}

/// Port of the `SFCONV/mkspectf.f90` quasiparticle row assembly.
///
/// FEFF fills `spectf(1,:)` with finite-element quasiparticle peak averages
/// and `spectf(3,:)` with the proportional interference term. It also carries
/// endpoint-corrected integrals for both rows; those accumulators are returned
/// for tests and future full-driver assembly.
pub fn sfconv_quasiparticle_table(
    input: SfconvQuasiparticleTableInput<'_>,
) -> Result<SfconvQuasiparticleTable, SfconvError> {
    validate_quasiparticle_table_input(input)?;

    let pi = std::f64::consts::PI;
    let endpoint_main = ((input.boundaries[0] / input.endpoint_width).atan() + pi / 2.0) / pi
        + (pi / 2.0 - (input.boundaries[input.boundaries.len() - 1] / input.endpoint_width).atan())
            / pi;
    let mut integrated_interference = 2.0
        * endpoint_main
        * input.renormalization_magnitude
        * input.renormalization_real
        * input.interference_amplitude;
    let mut integrated_main =
        endpoint_main * input.renormalization_real * input.exponential_reduction;

    let mut main_peak = Array1::<Real>::zeros(input.energy.len());
    let mut interference_peak = Array1::<Real>::zeros(input.energy.len());
    for column in 0..input.energy.len() {
        let peak = sfconv_quasiparticle_main_peak(SfconvQuasiparticlePeakInput {
            center_energy: input.energy[column],
            lower_boundary: input.boundaries[column],
            upper_boundary: input.boundaries[column + 1],
            photoelectron_energy: input.photoelectron_energy,
            quasiparticle_energy: input.quasiparticle_energy,
            quasiparticle_width: input.quasiparticle_width,
            plasma_frequency: input.plasma_frequency,
            renormalization_real: input.renormalization_real,
            renormalization_imag: input.renormalization_imag,
        })?;
        let interference =
            2.0 * input.renormalization_magnitude * input.interference_amplitude * peak;
        let width = input.boundaries[column + 1] - input.boundaries[column];

        main_peak[column] = peak;
        interference_peak[column] = interference;
        integrated_main += peak * input.exponential_reduction * width;
        integrated_interference += interference * input.exponential_reduction * width;
    }

    validate_finite_array("quasiparticle main row", main_peak.view())?;
    validate_finite_array("quasiparticle interference row", interference_peak.view())?;
    finite_result("quasiparticle integrated main weight", integrated_main)?;
    finite_result(
        "quasiparticle integrated interference weight",
        integrated_interference,
    )?;
    Ok(SfconvQuasiparticleTable {
        main_peak,
        interference_peak,
        integrated_main_weight: integrated_main,
        integrated_interference_weight: integrated_interference,
    })
}

/// Port of FEFF `SFCONV/mkspectf.f90` quasiparticle-interference `ak` loop.
///
/// FEFF calls `xmkak(ekp)` once per active pole, multiplies by the empirical
/// `xreduc` factor and the pole weight, and accumulates the result into `ak`.
/// This helper preserves that accumulation and returns the combined integration
/// diagnostics from the underlying `xmkak` integrations.
pub fn sfconv_quasiparticle_interference_amplitude(
    input: SfconvQuasiparticleInterferenceInput<'_>,
) -> Result<SfconvQuasiparticleInterference, SfconvError> {
    validate_quasiparticle_interference_input(input)?;

    let mut amplitude = 0.0;
    let mut estimated_error = 0.0;
    let mut evaluations = 0;
    let mut max_regions = 0;

    for pole_index in 0..input.pole_count {
        let pole_weight = input.pole_weight[pole_index];
        let context = SfconvSatelliteContext {
            plasma_frequency: input.plasma_frequency,
            pole_energy: input.pole_energy[pole_index],
            dispersion_parameter: input.dispersion_parameter,
            photoelectron_energy: input.bare_photoelectron_energy,
            accuracy: input.accuracy,
        };
        let integral = sfconv_interference_quasiparticle(
            input.quasiparticle_energy,
            input.upper_energy,
            context,
        )?;
        let scale = input.interference_reduction * pole_weight;
        amplitude = finite_result(
            "quasiparticle interference amplitude",
            amplitude + integral.value * scale,
        )?;
        estimated_error = finite_result(
            "quasiparticle interference error",
            estimated_error + integral.estimated_error * scale.abs(),
        )?;
        evaluations += integral.evaluations;
        max_regions = max_regions.max(integral.max_regions);
    }

    Ok(SfconvQuasiparticleInterference {
        amplitude,
        estimated_error,
        evaluations,
        max_regions,
    })
}

/// Port of FEFF `SFCONV/mkspectf.f90` satellite pole contribution loop.
///
/// FEFF chooses pole-local broadenings from `max(5*dw, brd)` for `xmkxsat` and
/// `max(2*dw, brd)` for `xmkisat`, optionally adds the quasiparticle width for
/// `isattype.eq.3`, then accumulates `xsat` and `xisat` using the active pole
/// weights. This helper preserves that loop around the already ported
/// `xmkxsat` and `xmkisat` integrators.
pub fn sfconv_satellite_pole_contributions(
    input: SfconvSatellitePoleContributionsInput<'_>,
) -> Result<SfconvSatellitePoleContributions, SfconvError> {
    validate_satellite_pole_contributions_input(input)?;

    let mut interference_satellite = 0.0;
    let mut intrinsic_satellite = 0.0;
    let mut interference_estimated_error = 0.0;
    let mut intrinsic_estimated_error = 0.0;
    let mut evaluations = 0;
    let mut max_regions = 0;

    for pole_index in 0..input.pole_count {
        let pole_weight = input.pole_weight[pole_index];
        let pole_broadening = input.pole_broadening[pole_index];
        let width_offset = if input.include_full_broadening {
            input.quasiparticle_width
        } else {
            0.0
        };
        let interference_width = finite_result(
            "interference satellite width",
            (5.0 * input.uniform_width).max(pole_broadening) + width_offset,
        )?;
        let intrinsic_width = finite_result(
            "intrinsic satellite width",
            (2.0 * input.uniform_width).max(pole_broadening) + width_offset,
        )?;
        validate_positive_scalar("interference satellite width", interference_width)?;
        validate_positive_scalar("intrinsic satellite width", intrinsic_width)?;

        let context = SfconvSatelliteContext {
            plasma_frequency: input.plasma_frequency,
            pole_energy: input.pole_energy[pole_index],
            dispersion_parameter: input.dispersion_parameter,
            photoelectron_energy: input.bare_photoelectron_energy,
            accuracy: input.accuracy,
        };
        let interference =
            sfconv_interference_satellite(input.energy, interference_width, context)?;
        let intrinsic = sfconv_intrinsic_satellite(input.energy, intrinsic_width, context)?;

        let interference_scale = input.interference_reduction * pole_weight;
        interference_satellite = finite_result(
            "interference satellite contribution",
            interference_satellite + interference.value * interference_scale,
        )?;
        intrinsic_satellite = finite_result(
            "intrinsic satellite contribution",
            intrinsic_satellite + intrinsic.value * pole_weight,
        )?;
        interference_estimated_error = finite_result(
            "interference satellite error",
            interference_estimated_error + interference.estimated_error * interference_scale.abs(),
        )?;
        intrinsic_estimated_error = finite_result(
            "intrinsic satellite error",
            intrinsic_estimated_error + intrinsic.estimated_error * pole_weight.abs(),
        )?;
        evaluations += interference.evaluations + intrinsic.evaluations;
        max_regions = max_regions
            .max(interference.max_regions)
            .max(intrinsic.max_regions);
    }

    Ok(SfconvSatellitePoleContributions {
        interference_satellite,
        intrinsic_satellite,
        interference_estimated_error,
        intrinsic_estimated_error,
        evaluations,
        max_regions,
    })
}

/// Port of FEFF `SFCONV/mkspectf.f90` extrinsic satellite `isattype` branch.
///
/// FEFF selects one of four approximations for `esat`: the full-broadening
/// branch with `emain` removed, a local derivative expansion, the full
/// broadening branch, or the default de-broadened `xmkesat` expression.
pub fn sfconv_extrinsic_satellite(
    input: SfconvExtrinsicSatelliteInput,
) -> Result<Real, SfconvError> {
    validate_extrinsic_satellite_input(input)?;

    match input.mode {
        SfconvExtrinsicSatelliteMode::BroadenedMinusMain => finite_result(
            "extrinsic satellite",
            sfconv_extrinsic_satellite_broadened(input.energy, input.self_energy)?
                - input.main_peak,
        ),
        SfconvExtrinsicSatelliteMode::DerivativeExpansion => {
            validate_nonzero_denominator("derivative extrinsic satellite energy", input.energy)?;
            finite_result(
                "derivative extrinsic satellite",
                (input.self_energy.off_shell_imag
                    - input.self_energy.width
                    - input.energy * input.imaginary_derivative)
                    / (std::f64::consts::PI * input.energy.powi(2)),
            )
        }
        SfconvExtrinsicSatelliteMode::FullBroadening => {
            sfconv_extrinsic_satellite_broadened(input.energy, input.self_energy)
        }
        SfconvExtrinsicSatelliteMode::Debroadened => {
            sfconv_extrinsic_satellite_debroadened(input.energy, input.context, input.self_energy)
        }
    }
}

/// Port of one iteration of FEFF `SFCONV/mkspectf.f90` spectral row assembly.
///
/// This helper computes `emain`, `xmain`, `esat`, `xsat`, `xisat`, and the
/// combined row for one energy cell. Later table-level helpers still handle the
/// endpoint average, satellite split, clipping, and final weights.
pub fn sfconv_spectral_cell(
    input: SfconvSpectralCellInput<'_>,
) -> Result<SfconvSpectralCell, SfconvError> {
    validate_spectral_cell_input(input)?;

    let main_peak = sfconv_quasiparticle_main_peak(SfconvQuasiparticlePeakInput {
        center_energy: input.center_energy,
        lower_boundary: input.lower_boundary,
        upper_boundary: input.upper_boundary,
        photoelectron_energy: input.photoelectron_energy,
        quasiparticle_energy: input.quasiparticle_energy,
        quasiparticle_width: input.quasiparticle_width,
        plasma_frequency: input.context.plasma_frequency,
        renormalization_real: input.self_energy.renormalization_real,
        renormalization_imag: input.self_energy.renormalization_imag,
    })?;
    let renormalization_magnitude = checked_hypot(
        "spectral cell renormalization",
        input.self_energy.renormalization_real,
        input.self_energy.renormalization_imag,
    )?;
    let quasiparticle_interference = finite_result(
        "spectral cell quasiparticle interference",
        2.0 * renormalization_magnitude * input.interference_amplitude * main_peak,
    )?;
    let extrinsic_satellite = sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
        energy: input.center_energy,
        main_peak,
        imaginary_derivative: input.imaginary_derivative,
        mode: input.extrinsic_mode,
        context: input.context,
        self_energy: input.self_energy,
    })?;
    let satellite = sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
        energy: input.center_energy,
        uniform_width: input.uniform_width,
        quasiparticle_width: input.self_energy.width,
        plasma_frequency: input.context.plasma_frequency,
        bare_photoelectron_energy: input.context.photoelectron_energy,
        dispersion_parameter: input.context.dispersion_parameter,
        accuracy: input.context.accuracy,
        interference_reduction: input.interference_reduction,
        include_full_broadening: matches!(
            input.extrinsic_mode,
            SfconvExtrinsicSatelliteMode::FullBroadening
        ),
        pole_count: input.pole_count,
        pole_energy: input.pole_energy,
        pole_weight: input.pole_weight,
        pole_broadening: input.pole_broadening,
    })?;
    let mut combined_satellite = finite_result(
        "spectral cell combined satellite",
        extrinsic_satellite + satellite.intrinsic_satellite
            - 2.0 * satellite.interference_satellite,
    )?;
    if matches!(
        input.extrinsic_mode,
        SfconvExtrinsicSatelliteMode::FullBroadening
    ) {
        combined_satellite = finite_result(
            "spectral cell combined satellite",
            combined_satellite + quasiparticle_interference,
        )?;
    }

    Ok(SfconvSpectralCell {
        main_peak,
        extrinsic_satellite,
        quasiparticle_interference,
        interference_satellite: satellite.interference_satellite,
        intrinsic_satellite: satellite.intrinsic_satellite,
        combined_satellite,
        interference_estimated_error: satellite.interference_estimated_error,
        intrinsic_estimated_error: satellite.intrinsic_estimated_error,
        evaluations: satellite.evaluations,
        max_regions: satellite.max_regions,
    })
}

/// Port of the FEFF `SFCONV/mkspectf.f90` spectral-function cell loop.
///
/// This assembles rows 1 through 6 from the per-cell helper, preserves FEFF's
/// endpoint-corrected quasiparticle accumulators, and applies the legacy
/// average of the two quasiparticle-adjacent extrinsic-satellite cells. Later
/// helpers still split the extrinsic satellite, clip negative satellite weight,
/// and write the final eight-slot weight vector.
pub fn sfconv_spectral_table(
    input: SfconvSpectralTableInput<'_>,
) -> Result<SfconvSpectralTable, SfconvError> {
    validate_spectral_table_input(input)?;

    let columns = input.energy.len();
    let renormalization_magnitude = checked_hypot(
        "spectral table renormalization",
        input.self_energy.renormalization_real,
        input.self_energy.renormalization_imag,
    )?;
    let pi = std::f64::consts::PI;
    let endpoint_main = ((input.boundaries[0] / input.self_energy.width).atan() + pi / 2.0) / pi
        + (pi / 2.0
            - (input.boundaries[input.boundaries.len() - 1] / input.self_energy.width).atan())
            / pi;
    let mut integrated_quasiparticle_interference = finite_result(
        "spectral table integrated quasiparticle interference weight",
        2.0 * endpoint_main
            * renormalization_magnitude
            * input.self_energy.renormalization_real
            * input.interference_amplitude,
    )?;
    let mut integrated_main = finite_result(
        "spectral table integrated main weight",
        endpoint_main * input.self_energy.renormalization_real * input.exponential_reduction,
    )?;
    let mut integrated_extrinsic = 0.0;
    let mut integrated_interference = 0.0;
    let mut integrated_intrinsic = 0.0;
    let mut interference_estimated_error = 0.0;
    let mut intrinsic_estimated_error = 0.0;
    let mut evaluations = 0;
    let mut max_regions = 0;
    let mut spectral_function = Array2::<Real>::zeros((8, columns));

    for column in 0..columns {
        let width = input.boundaries[column + 1] - input.boundaries[column];
        let self_energy = SfconvSatelliteSelfEnergy {
            off_shell_real: input.off_shell_real[column],
            off_shell_imag: input.off_shell_imag[column],
            ..input.self_energy
        };
        let cell = sfconv_spectral_cell(SfconvSpectralCellInput {
            center_energy: input.energy[column],
            lower_boundary: input.boundaries[column],
            upper_boundary: input.boundaries[column + 1],
            photoelectron_energy: input.photoelectron_energy,
            quasiparticle_energy: input.quasiparticle_energy,
            quasiparticle_width: input.quasiparticle_width,
            interference_amplitude: input.interference_amplitude,
            extrinsic_mode: input.extrinsic_mode,
            imaginary_derivative: input.imaginary_derivative,
            uniform_width: input.uniform_width,
            interference_reduction: input.interference_reduction,
            context: input.context,
            self_energy,
            pole_count: input.pole_count,
            pole_energy: input.pole_energy,
            pole_weight: input.pole_weight,
            pole_broadening: input.pole_broadening,
        })?;

        integrated_main = finite_result(
            "spectral table integrated main weight",
            integrated_main + cell.main_peak * input.exponential_reduction * width,
        )?;
        integrated_quasiparticle_interference = finite_result(
            "spectral table integrated quasiparticle interference weight",
            integrated_quasiparticle_interference
                + cell.quasiparticle_interference * input.exponential_reduction * width,
        )?;
        integrated_extrinsic = finite_result(
            "spectral table integrated extrinsic weight",
            integrated_extrinsic + cell.extrinsic_satellite * input.exponential_reduction * width,
        )?;
        integrated_interference = finite_result(
            "spectral table integrated interference weight",
            integrated_interference
                + cell.interference_satellite * input.exponential_reduction * width,
        )?;
        integrated_intrinsic = finite_result(
            "spectral table integrated intrinsic weight",
            integrated_intrinsic + cell.intrinsic_satellite * input.exponential_reduction * width,
        )?;
        interference_estimated_error = finite_result(
            "spectral table interference satellite error",
            interference_estimated_error + cell.interference_estimated_error,
        )?;
        intrinsic_estimated_error = finite_result(
            "spectral table intrinsic satellite error",
            intrinsic_estimated_error + cell.intrinsic_estimated_error,
        )?;
        evaluations += cell.evaluations;
        max_regions = max_regions.max(cell.max_regions);

        spectral_function[(0, column)] = cell.main_peak;
        spectral_function[(1, column)] = cell.extrinsic_satellite;
        spectral_function[(2, column)] = cell.quasiparticle_interference;
        spectral_function[(3, column)] = cell.interference_satellite;
        spectral_function[(4, column)] = cell.intrinsic_satellite;
        spectral_function[(5, column)] = cell.combined_satellite;
    }

    let lower_column = feff_index(input.quasiparticle_lower_column_1based);
    let upper_column = feff_index(input.quasiparticle_upper_column_1based);
    let averaged_extrinsic =
        0.5 * (spectral_function[(1, lower_column)] + spectral_function[(1, upper_column)]);
    spectral_function[(1, lower_column)] = averaged_extrinsic;
    spectral_function[(1, upper_column)] = averaged_extrinsic;

    validate_finite_spectral_rows(spectral_function.view())?;
    Ok(SfconvSpectralTable {
        spectral_function,
        integrated_main_weight: integrated_main,
        integrated_quasiparticle_interference_weight: integrated_quasiparticle_interference,
        integrated_extrinsic_weight: integrated_extrinsic,
        integrated_interference_weight: integrated_interference,
        integrated_intrinsic_weight: integrated_intrinsic,
        interference_estimated_error,
        intrinsic_estimated_error,
        evaluations,
        max_regions,
    })
}

/// Port of the `SFCONV/mkspectf.f90` satellite row assembly.
///
/// FEFF fills rows 2, 4, and 5 from the extrinsic, interference, and intrinsic
/// satellite estimates, forms row 6 as their combined satellite contribution,
/// and accumulates the raw satellite weights before later splitting and
/// clipping. The extrinsic satellite is then averaged across the two
/// quasiparticle-adjacent cells, preserving FEFF's order of operations.
pub fn sfconv_satellite_table(
    input: SfconvSatelliteTableInput<'_>,
) -> Result<SfconvSatelliteTable, SfconvError> {
    validate_satellite_table_input(input)?;

    let columns = input.extrinsic_satellite.len();
    let mut spectral_function = Array2::<Real>::zeros((8, columns));
    let mut integrated_extrinsic = 0.0;
    let mut integrated_interference = 0.0;
    let mut integrated_intrinsic = 0.0;

    for column in 0..columns {
        let width = input.boundaries[column + 1] - input.boundaries[column];
        let extrinsic = input.extrinsic_satellite[column];
        let interference = input.interference_satellite[column];
        let intrinsic = input.intrinsic_satellite[column];
        let quasiparticle_interference = input.quasiparticle_interference[column];
        let mut combined = extrinsic + intrinsic - 2.0 * interference;
        if input.include_full_broadening_quasiparticle {
            combined += quasiparticle_interference;
        }

        integrated_extrinsic += extrinsic * width * input.exponential_reduction;
        integrated_interference += interference * width * input.exponential_reduction;
        integrated_intrinsic += intrinsic * width * input.exponential_reduction;

        spectral_function[(0, column)] = input.main_peak[column];
        spectral_function[(1, column)] = extrinsic;
        spectral_function[(2, column)] = quasiparticle_interference;
        spectral_function[(3, column)] = interference;
        spectral_function[(4, column)] = intrinsic;
        spectral_function[(5, column)] = combined;
    }

    let lower_column = feff_index(input.quasiparticle_lower_column_1based);
    let upper_column = feff_index(input.quasiparticle_upper_column_1based);
    let averaged_extrinsic =
        0.5 * (spectral_function[(1, lower_column)] + spectral_function[(1, upper_column)]);
    spectral_function[(1, lower_column)] = averaged_extrinsic;
    spectral_function[(1, upper_column)] = averaged_extrinsic;

    validate_finite_array("satellite table main row", spectral_function.row(0))?;
    validate_finite_array("satellite table extrinsic row", spectral_function.row(1))?;
    validate_finite_array(
        "satellite table quasiparticle row",
        spectral_function.row(2),
    )?;
    validate_finite_array("satellite table interference row", spectral_function.row(3))?;
    validate_finite_array("satellite table intrinsic row", spectral_function.row(4))?;
    validate_finite_array("satellite table combined row", spectral_function.row(5))?;
    finite_result(
        "satellite integrated extrinsic weight",
        integrated_extrinsic,
    )?;
    finite_result(
        "satellite integrated interference weight",
        integrated_interference,
    )?;
    finite_result(
        "satellite integrated intrinsic weight",
        integrated_intrinsic,
    )?;
    Ok(SfconvSatelliteTable {
        spectral_function,
        integrated_extrinsic_weight: integrated_extrinsic,
        integrated_interference_weight: integrated_interference,
        integrated_intrinsic_weight: integrated_intrinsic,
    })
}

/// Port of the `SFCONV/mkspectf.f90` extrinsic-satellite split.
///
/// FEFF scans the extrinsic satellite row from high to low energy, finds the
/// first derivative or curvature trigger after the satellite begins rising,
/// then copies `spectf(2)` into row 7 below that switch and row 8 at and above
/// it. The legacy code currently sets the smoothing width to zero, so this
/// helper preserves the resulting sharp split.
pub fn sfconv_split_extrinsic_satellite(
    input: SfconvExtrinsicSatelliteSplitInput<'_>,
) -> Result<SfconvExtrinsicSatelliteSplit, SfconvError> {
    validate_extrinsic_satellite_split_input(input)?;

    let columns = input.spectral_function.ncols();
    let mut derivative_switch = None;
    let mut curvature_switch = None;
    let mut satellite_started = false;

    for ii_1based in 2..columns {
        let column = columns - ii_1based;
        let satellite = input.spectral_function[(1, column)];
        let slope = (satellite - input.spectral_function[(1, column - 1)])
            / (input.energy[column] - input.energy[column - 1]);
        let high_slope = (input.spectral_function[(1, column + 1)] - satellite)
            / (input.energy[column + 1] - input.energy[column]);
        let curvature =
            (high_slope - slope) / (input.boundaries[column + 1] - input.boundaries[column]);
        let absolute_energy = input.energy[column] + input.photoelectron_energy;

        if slope > 0.0 && satellite > 0.0 {
            satellite_started = true;
        }

        let derivative_allowed = input.beta_zero > 0.0 || absolute_energy > 0.0;
        if slope < 0.0 && satellite_started && derivative_allowed && derivative_switch.is_none() {
            derivative_switch = Some((column, absolute_energy));
        }
        if curvature > 0.0 && satellite_started && curvature_switch.is_none() {
            curvature_switch = Some((column, absolute_energy));
        }
    }

    let (switch_column, switch_energy, derivative_triggered) =
        if let Some((column, energy)) = derivative_switch {
            (column, energy, true)
        } else if let Some((column, energy)) = curvature_switch {
            (column, energy, false)
        } else {
            return Err(SfconvError::MissingTrigger {
                field: "extrinsic satellite split",
            });
        };

    let mut spectral_function = input.spectral_function.to_owned();
    for column in 0..columns {
        spectral_function[(6, column)] = 0.0;
        spectral_function[(7, column)] = 0.0;
        if column >= switch_column {
            spectral_function[(7, column)] = spectral_function[(1, column)];
        } else {
            spectral_function[(6, column)] = spectral_function[(1, column)];
        }
    }

    validate_finite_array("extrinsic split row 7", spectral_function.row(6))?;
    validate_finite_array("extrinsic split row 8", spectral_function.row(7))?;
    finite_result("extrinsic split switch energy", switch_energy)?;
    Ok(SfconvExtrinsicSatelliteSplit {
        spectral_function,
        switch_column,
        switch_energy,
        derivative_triggered,
    })
}

/// Port of the final `SFCONV/mkspectf.f90` satellite clipping correction.
///
/// FEFF first forms the combined satellite row
/// `spectf(6)=spectf(2)-2*spectf(4)+spectf(5)`. Negative combined satellite
/// cells are clipped to zero, the surviving positive part is renormalized to
/// preserve the original integral, and the interference row is recomputed so
/// downstream interpolation sees the corrected combined satellite. The returned
/// weights are FEFF `weights(4:8)`.
pub fn sfconv_correct_satellite_weights(
    input: SfconvSatelliteCorrectionInput<'_>,
) -> Result<SfconvSatelliteCorrection, SfconvError> {
    validate_satellite_correction_input(input)?;

    let columns = input.spectral_function.ncols();
    let mut corrected = input.spectral_function.to_owned();
    for column in 0..columns {
        corrected[(5, column)] =
            corrected[(1, column)] - 2.0 * corrected[(3, column)] + corrected[(4, column)];
    }

    let mut clipped_negative_weight = 0.0;
    let mut uncorrected_satellite_weight = 0.0;
    for column in 0..columns {
        let width = input.boundaries[column + 1] - input.boundaries[column];
        let combined = corrected[(5, column)];
        uncorrected_satellite_weight += combined * width;
        if combined < 0.0 {
            clipped_negative_weight += combined * width;
            corrected[(5, column)] = 0.0;
            corrected[(3, column)] = 0.5 * (corrected[(1, column)] + corrected[(4, column)]);
        }
    }

    let correction_denominator = uncorrected_satellite_weight - clipped_negative_weight;
    validate_nonzero_denominator("satellite correction", correction_denominator)?;
    let correction_factor = (uncorrected_satellite_weight / correction_denominator).max(0.0);

    let mut weights = Array1::<Real>::zeros(5);
    for column in 0..columns {
        let width = input.boundaries[column + 1] - input.boundaries[column];
        corrected[(3, column)] = 0.5
            * (corrected[(1, column)] + corrected[(4, column)]
                - corrected[(5, column)] * correction_factor);
        weights[0] += corrected[(1, column)] * width * input.exponential_reduction;
        weights[1] += corrected[(3, column)] * width * input.exponential_reduction;
        weights[2] += corrected[(4, column)] * width * input.exponential_reduction;
        weights[3] += corrected[(7, column)] * input.exponential_reduction * input.uniform_width;
        weights[4] += corrected[(6, column)] * input.exponential_reduction * input.uniform_width;
    }

    validate_finite_array("satellite correction weights", weights.view())?;
    validate_finite_spectral_rows(corrected.view())?;
    finite_result("uncorrected satellite weight", uncorrected_satellite_weight)?;
    finite_result("clipped satellite weight", clipped_negative_weight)?;
    finite_result("satellite correction factor", correction_factor)?;
    Ok(SfconvSatelliteCorrection {
        spectral_function: corrected,
        weights,
        uncorrected_satellite_weight,
        clipped_negative_weight,
        correction_factor,
    })
}

/// Port of the final FEFF `SFCONV/mkspectf.f90` postprocessing sequence.
///
/// FEFF first splits the extrinsic satellite into satellite-like and
/// quasiparticle-like rows, then clips negative combined satellite weight, and
/// finally writes the eight `weights` values. This helper chains the already
/// ported stages without changing their order or formulas.
pub fn sfconv_finalize_spectral_table(
    input: SfconvSpectralFinalizationInput<'_>,
) -> Result<SfconvSpectralFinalization, SfconvError> {
    validate_spectral_finalization_input(input)?;

    let split = sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
        spectral_function: input.spectral_function,
        energy: input.energy,
        boundaries: input.boundaries,
        photoelectron_energy: input.photoelectron_energy,
        beta_zero: input.beta_zero,
    })?;
    let correction = sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
        spectral_function: split.spectral_function.view(),
        boundaries: input.boundaries,
        uniform_width: input.uniform_width,
        exponential_reduction: input.exponential_reduction,
    })?;
    let weights = sfconv_spectral_weights(SfconvSpectralWeightsInput {
        renormalization_real: input.renormalization_real,
        renormalization_imag: input.renormalization_imag,
        renormalization_magnitude: input.renormalization_magnitude,
        interference_amplitude: input.interference_amplitude,
        interference_reduction: input.interference_reduction,
        exponential_reduction: input.exponential_reduction,
        satellite_weights: correction.weights.view(),
    })?;

    Ok(SfconvSpectralFinalization {
        spectral_function: correction.spectral_function,
        weights,
        switch_column: split.switch_column,
        switch_energy: split.switch_energy,
        derivative_triggered: split.derivative_triggered,
        uncorrected_satellite_weight: correction.uncorrected_satellite_weight,
        clipped_negative_weight: correction.clipped_negative_weight,
        correction_factor: correction.correction_factor,
    })
}

/// Port of the final FEFF `SFCONV/mkspectf.f90` `weights(1:8)` assignment.
///
/// FEFF does not use the endpoint-corrected quasiparticle accumulators for the
/// final array. It writes the first three slots directly from the
/// renormalization constants and interference amplitude, then copies the five
/// corrected satellite weights into slots 4 through 8.
pub fn sfconv_spectral_weights(
    input: SfconvSpectralWeightsInput<'_>,
) -> Result<RealVec, SfconvError> {
    validate_spectral_weights_input(input)?;

    let mut weights = Array1::<Real>::zeros(8);
    weights[0] = input.renormalization_real * input.exponential_reduction;
    weights[1] = input.renormalization_imag * input.exponential_reduction;
    weights[2] = 2.0
        * input.renormalization_real
        * input.renormalization_magnitude
        * input.interference_amplitude
        * input.interference_reduction
        * input.exponential_reduction;
    for (index, weight) in input.satellite_weights.iter().copied().enumerate() {
        weights[index + 3] = weight;
    }

    validate_finite_array("spectral weights", weights.view())?;
    Ok(weights)
}
