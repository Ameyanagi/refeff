use super::*;

/// Port FEFF DMDW run-type 2 `SelfEn` for one electron energy.
///
/// `temperature_kelvin` is FEFF `Lanc_In%T(1)`, `energy_ev` is the complex
/// photoelectron energy `z`, and the pole arrays are the two columns of FEFF
/// `a2fpole`: phonon energy in eV and pole-weight `a2f`. The returned complex
/// value is FEFF `SE_a2f`.
pub fn dmdw_self_energy_from_a2f_poles(
    temperature_kelvin: Real,
    energy_ev: Complex,
    pole_energy_ev: ArrayView1<'_, Real>,
    pole_weight: ArrayView1<'_, Real>,
) -> Result<Complex, DebyeError> {
    validate_dmdw_self_energy_inputs(temperature_kelvin, energy_ev, pole_energy_ev, pole_weight)?;

    let denominator =
        DMDW_SELF_ENERGY_TWO_PI * DMDW_SELF_ENERGY_BOLTZMANN_EV_PER_K * temperature_kelvin;
    let thermal_energy = DMDW_SELF_ENERGY_BOLTZMANN_EV_PER_K * temperature_kelvin;
    let i = Complex::new(0.0, 1.0);
    let half = Complex::new(0.5, 0.0);

    let mut self_energy = Complex::new(0.0, 0.0);
    for (&phonon_energy, &weight) in pole_energy_ev.iter().zip(pole_weight.iter()) {
        ensure_finite("DMDW self-energy pole energy", phonon_energy)?;
        ensure_finite("DMDW self-energy pole weight", weight)?;
        if weight == 0.0 {
            continue;
        }
        ensure_positive("DMDW self-energy pole energy", phonon_energy)?;

        let phonon = Complex::new(phonon_energy, 0.0);
        let occupation = 1.0 / ((phonon_energy / thermal_energy).exp() - 1.0);
        ensure_finite_output("DMDW self-energy Bose occupation", occupation)?;

        let argp = half + i * (phonon - energy_ev) / denominator;
        let argm = half - i * (phonon + energy_ev) / denominator;
        let mode_self_energy = complex_digamma(argp)?
            - complex_digamma(argm)?
            - i * DMDW_SELF_ENERGY_TWO_PI * (occupation + 0.5);
        self_energy += mode_self_energy * weight;
    }

    ensure_finite_complex_output("DMDW self-energy", self_energy)?;
    Ok(self_energy)
}

/// Evaluate FEFF DMDW run-type 2 `SelfEn` on a real electron-energy grid.
///
/// The pole representation is the output of [`dmdw_pole_weighted_a2f`]. This
/// helper keeps the grid as `ndarray` storage so callers can write FEFF's
/// `dmdw_reSE_a2F.dat` and `dmdw_imSE_a2F.dat` sidecars without reshaping.
pub fn dmdw_self_energy_grid_from_a2f_poles(
    temperature_kelvin: Real,
    energy_ev: ArrayView1<'_, Real>,
    diagnostic: &DmdwPoleWeightedA2f,
) -> Result<DmdwSelfEnergyGrid, DebyeError> {
    if energy_ev.is_empty() {
        return Err(DebyeError::EmptyDmdwSelfEnergyGrid);
    }

    let self_energy = energy_ev
        .iter()
        .copied()
        .map(|energy| {
            ensure_finite("DMDW self-energy grid energy", energy)?;
            dmdw_self_energy_from_a2f_poles(
                temperature_kelvin,
                Complex::new(energy, 0.0),
                diagnostic.pole_energy_ev.view(),
                diagnostic.pole_weight.view(),
            )
        })
        .collect::<Result<Array1<_>, DebyeError>>()?;

    Ok(DmdwSelfEnergyGrid {
        energy_ev: energy_ev.to_owned(),
        self_energy,
    })
}

/// Port FEFF DMDW run-type 2 `Akw` cumulant spectral function.
///
/// `energy_w0` is the same uniformly spaced `w` grid FEFF writes to
/// `dmdw_Egrid.info`, expressed in units of `diagnostic.characteristic_energy_ev`.
/// `electron_energy_w0` is FEFF `Ek` after any meV-to-`w0` conversion.
/// `time_half_width` and `time_points` are FEFF `t0` and `Nt`; production
/// callers should use `t0 = 2000` and odd `Nt = 40001`.
pub fn dmdw_spectral_function_from_a2f_poles(
    temperature_kelvin: Real,
    energy_w0: ArrayView1<'_, Real>,
    electron_energy_w0: Real,
    characteristic_energy_ev: Real,
    diagnostic: &DmdwPoleWeightedA2f,
    time_half_width: Real,
    time_points: usize,
) -> Result<DmdwSpectralFunctionGrid, DebyeError> {
    ensure_positive("DMDW spectral temperature", temperature_kelvin)?;
    ensure_finite("DMDW spectral electron energy", electron_energy_w0)?;
    ensure_positive(
        "DMDW spectral characteristic energy",
        characteristic_energy_ev,
    )?;
    ensure_positive("DMDW spectral time half-width", time_half_width)?;
    if time_points < 3 || time_points.is_multiple_of(2) {
        return Err(DebyeError::InvalidDmdwSpectralTimeGrid {
            points: time_points,
        });
    }
    let energy_step = dmdw_uniform_spectral_energy_step(energy_w0)?;
    let half_time_points = (time_points - 1) / 2;
    let time_step = 2.0 * time_half_width / (time_points as Real - 1.0);
    ensure_positive("DMDW spectral time step", time_step)?;

    let zero_self_energy = dmdw_self_energy_from_a2f_poles(
        temperature_kelvin,
        Complex::new(0.0, 0.0),
        diagnostic.pole_energy_ev.view(),
        diagnostic.pole_weight.view(),
    )?;
    let beta_shift = zero_self_energy.im.abs() / characteristic_energy_ev;
    let gamma_w0 = beta_shift.max(0.005);
    ensure_finite_output("DMDW spectral beta shift", beta_shift)?;
    ensure_finite_output("DMDW spectral gamma", gamma_w0)?;

    let beta = energy_w0
        .iter()
        .copied()
        .map(|energy| {
            let shifted_energy_ev = (electron_energy_w0 + energy) * characteristic_energy_ev;
            let self_energy = dmdw_self_energy_from_a2f_poles(
                temperature_kelvin,
                Complex::new(shifted_energy_ev, 0.0),
                diagnostic.pole_energy_ev.view(),
                diagnostic.pole_weight.view(),
            )?;
            let value = (self_energy.im.abs() / characteristic_energy_ev - beta_shift)
                / std::f64::consts::PI;
            ensure_finite_output("DMDW spectral beta", value)?;
            Ok(value)
        })
        .collect::<Result<Array1<_>, DebyeError>>()?;

    let mut time_amplitude = Array1::from_elem(time_points, Complex::new(0.0, 0.0));
    time_amplitude[half_time_points] = Complex::new(1.0, 0.0);
    for time_index in 1..=half_time_points {
        let time = time_index as Real * time_step;
        let cumulant = dmdw_spectral_cumulant_at_time(energy_w0, beta.view(), energy_step, time)?;
        let amplitude = cumulant.exp();
        ensure_finite_complex_output("DMDW spectral time amplitude", amplitude)?;
        time_amplitude[half_time_points + time_index] = amplitude;
        time_amplitude[half_time_points - time_index] = amplitude.conj();
    }

    let spectral = energy_w0
        .iter()
        .copied()
        .map(|energy| {
            let value = dmdw_spectral_fourier_at_energy(
                energy,
                electron_energy_w0,
                gamma_w0,
                time_step,
                half_time_points,
                time_amplitude.view(),
            );
            ensure_finite_complex_output("DMDW spectral function", value)?;
            Ok(value)
        })
        .collect::<Result<Array1<_>, DebyeError>>()?;

    let integral = dmdw_trapezoid_complex(energy_step, spectral.view());
    ensure_finite_complex_output("DMDW spectral normalization integral", integral)?;
    let normalization = integral.norm();
    ensure_positive("DMDW spectral normalization", normalization)?;
    let spectral_function = spectral.mapv(|value| value / normalization);

    Ok(DmdwSpectralFunctionGrid {
        energy_w0: energy_w0.to_owned(),
        spectral_function,
        normalization,
        gamma_w0,
    })
}

fn dmdw_uniform_spectral_energy_step(energy_w0: ArrayView1<'_, Real>) -> Result<Real, DebyeError> {
    if energy_w0.len() < 2 {
        return Err(DebyeError::InvalidDmdwSpectralEnergyGrid {
            points: energy_w0.len(),
        });
    }
    for value in energy_w0.iter().copied() {
        ensure_finite("DMDW spectral energy", value)?;
    }
    let step = energy_w0[1] - energy_w0[0];
    ensure_positive("DMDW spectral energy step", step)?;
    let tolerance = (step.abs() * 1.0e-9).max(1.0e-12);
    for index in 2..energy_w0.len() {
        let actual_step = energy_w0[index] - energy_w0[index - 1];
        if (actual_step - step).abs() > tolerance {
            return Err(DebyeError::NonUniformDmdwSpectralEnergyGrid {
                row: index,
                step: actual_step,
                expected_step: step,
            });
        }
    }
    Ok(step)
}

fn dmdw_spectral_cumulant_at_time(
    energy_w0: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
    energy_step: Real,
    time: Real,
) -> Result<Complex, DebyeError> {
    let shifted_i = Complex::new(0.0, DMDW_SPECTRAL_DENOMINATOR_SHIFT);
    let integral = energy_w0
        .iter()
        .zip(beta.iter())
        .enumerate()
        .map(|(index, (&energy, &beta_value))| {
            let phase = energy * time;
            let numerator = Complex::new(phase.cos() - 1.0, -phase.sin());
            let denominator_base = Complex::new(energy, 0.0) - shifted_i;
            let denominator = denominator_base * denominator_base;
            let endpoint_weight = if index == 0 || index + 1 == energy_w0.len() {
                0.5
            } else {
                1.0
            };
            numerator * beta_value * endpoint_weight / denominator
        })
        .sum::<Complex>()
        * energy_step;
    ensure_finite_complex_output("DMDW spectral cumulant", integral)?;
    Ok(integral)
}

fn dmdw_spectral_fourier_at_energy(
    energy_w0: Real,
    electron_energy_w0: Real,
    gamma_w0: Real,
    time_step: Real,
    half_time_points: usize,
    time_amplitude: ArrayView1<'_, Complex>,
) -> Complex {
    (0..time_amplitude.len().saturating_sub(1))
        .map(|index| {
            let time_left = (index as Real - half_time_points as Real) * time_step;
            let time_right = time_left + time_step;
            let height_left =
                dmdw_spectral_fourier_kernel(energy_w0, electron_energy_w0, gamma_w0, time_left);
            let height_right =
                dmdw_spectral_fourier_kernel(energy_w0, electron_energy_w0, gamma_w0, time_right);
            (time_amplitude[index] * height_left + time_amplitude[index + 1] * height_right)
                * (0.5 * time_step)
        })
        .sum()
}

fn dmdw_spectral_fourier_kernel(
    energy_w0: Real,
    electron_energy_w0: Real,
    gamma_w0: Real,
    time: Real,
) -> Complex {
    Complex::new(
        -gamma_w0 * time.abs(),
        (energy_w0 - electron_energy_w0) * time,
    )
    .exp()
        / DMDW_SELF_ENERGY_TWO_PI
}

fn dmdw_trapezoid_complex(step: Real, values: ArrayView1<'_, Complex>) -> Complex {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if index == 0 || index + 1 == values.len() {
                value * 0.5
            } else {
                value
            }
        })
        .sum::<Complex>()
        * step
}
