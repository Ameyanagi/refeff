use super::*;

/// Port of FEFF `jpq`: Fourier transform `J(z,z')` into `J(p_q)`.
///
/// For `pq = 0`, FEFF uses a trapezoid rule over both axes. For nonzero `pq`,
/// it applies a piecewise-linear Fourier transform in `z'` and then `z`.
/// The returned scalar is the real part of FEFF's complex accumulator.
pub fn compton_profile(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
    input: ComptonProfileInput,
) -> Result<Real, ComptonError> {
    validate_finite("pq", input.pq)?;
    validate_finite("window_cutoff", input.window_cutoff)?;
    validate_grid_for_profile(grid, jzzp)?;
    validate_matrix_finite("jzzp", jzzp)?;
    let cutoff = profile_cutoff(grid, input.window, input.window_cutoff)?;

    compton_profile_with_validated_grid(grid, jzzp, input.pq, input.window, cutoff)
}

/// Evaluate FEFF `jpq` for a full momentum grid.
///
/// This is equivalent to calling [`compton_profile`] for each momentum value,
/// but validates the grid and `J(z,z')` cache once. It is the preferred API
/// when writing `compton.dat`.
pub fn compton_profiles(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
    momentum: ArrayView1<'_, Real>,
    window: ComptonWindow,
    window_cutoff: Real,
) -> Result<RealVec, ComptonError> {
    validate_finite("window_cutoff", window_cutoff)?;
    validate_grid_for_profile(grid, jzzp)?;
    validate_matrix_finite("jzzp", jzzp)?;
    let cutoff = profile_cutoff(grid, window, window_cutoff)?;
    momentum
        .iter()
        .map(|&pq| {
            validate_finite("pq", pq)?;
            compton_profile_with_validated_grid(grid, jzzp, pq, window, cutoff)
        })
        .collect()
}

fn profile_cutoff(
    grid: &ComptonGrid,
    window: ComptonWindow,
    window_cutoff: Real,
) -> Result<Real, ComptonError> {
    match window {
        ComptonWindow::Rectangular | ComptonWindow::CosineSquared => {
            let cutoff = if window_cutoff == 0.0 {
                grid.zp[grid.nzp() - 1]
            } else {
                window_cutoff
            };
            if cutoff <= 0.0 || !cutoff.is_finite() {
                return Err(ComptonError::InvalidWindowCutoff { value: cutoff });
            }
            Ok(cutoff)
        }
        ComptonWindow::Unwindowed => Ok(window_cutoff),
    }
}

fn compton_profile_with_validated_grid(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
    pq: Real,
    window: ComptonWindow,
    cutoff: Real,
) -> Result<Real, ComptonError> {
    if pq == 0.0 {
        return compton_profile_zero_pq(grid, jzzp, window, cutoff);
    }
    compton_profile_finite_pq(grid, jzzp, pq, window, cutoff)
}

fn compton_profile_zero_pq(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
    window: ComptonWindow,
    cutoff: Real,
) -> Result<Real, ComptonError> {
    let mut profile = 0.0;
    let mut previous_z_integral = 0.0;
    let mut previous_z = 0.0;
    for iz in 0..grid.nz() {
        let z = grid.z[iz];
        let mut z_integral = 0.0;
        let mut previous_zp = grid.zp[0];

        for izp in 1..grid.nzp() {
            let zp = grid.zp[izp];
            let weight = compton_window_weight(window, zp, cutoff);
            z_integral +=
                (jzzp[(iz, izp)] + jzzp[(iz, izp - 1)]) * 0.5 * (zp - previous_zp) * weight;
            previous_zp = zp;
        }

        if iz > 0 {
            profile += (z_integral + previous_z_integral) * 0.5 * (z - previous_z);
        }
        previous_z_integral = z_integral;
        previous_z = z;
    }

    validate_profile_result(profile)
}

fn compton_profile_finite_pq(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
    pq: Real,
    window: ComptonWindow,
    cutoff: Real,
) -> Result<Real, ComptonError> {
    let i_over_pq = Complex64::new(0.0, 1.0 / pq);
    let i_pq = Complex64::new(0.0, pq);
    let minus_i_pq = -i_pq;
    let inv_i_pq = Complex64::new(0.0, -1.0 / pq);
    let inv_minus_i_pq = Complex64::new(0.0, 1.0 / pq);
    let zp_phases = grid
        .zp
        .iter()
        .map(|&zp| (i_pq * zp).exp())
        .collect::<Vec<_>>();
    let z_phases = grid
        .z
        .iter()
        .map(|&z| (minus_i_pq * z).exp())
        .collect::<Vec<_>>();
    let zp_weights = grid
        .zp
        .iter()
        .map(|&zp| compton_window_weight(window, zp, cutoff))
        .collect::<Vec<_>>();
    let mut profile = Complex64::new(0.0, 0.0);
    let mut previous_z_integral = Complex64::new(0.0, 0.0);
    let mut previous_z = 0.0;
    for iz in 0..grid.nz() {
        let z = grid.z[iz];
        let mut z_integral = Complex64::new(0.0, 0.0);
        let mut previous_zp = grid.zp[0];

        for izp in 1..grid.nzp() {
            let zp = grid.zp[izp];
            let dzp = zp - previous_zp;
            if dzp == 0.0 {
                return Err(ComptonError::ZeroFourierInterval {
                    axis: "z'",
                    index: izp,
                });
            }
            let slope = Complex64::new((jzzp[(iz, izp)] - jzzp[(iz, izp - 1)]) / dzp, 0.0);
            let previous_value = Complex64::new(jzzp[(iz, izp - 1)], 0.0);
            let a = previous_value + slope * (Complex64::new(dzp, 0.0) + i_over_pq);
            let b = previous_value + slope * i_over_pq;
            z_integral += (zp_phases[izp] * a * inv_i_pq - zp_phases[izp - 1] * b * inv_i_pq)
                * zp_weights[izp];
            previous_zp = zp;
        }

        if iz > 0 {
            let dz = z - previous_z;
            if dz == 0.0 {
                return Err(ComptonError::ZeroFourierInterval {
                    axis: "z",
                    index: iz,
                });
            }
            let slope = (z_integral - previous_z_integral) / dz;
            let a = previous_z_integral + slope * (Complex64::new(dz, 0.0) - i_over_pq);
            let b = previous_z_integral - slope * i_over_pq;
            profile += z_phases[iz] * a * inv_minus_i_pq - z_phases[iz - 1] * b * inv_minus_i_pq;
        }
        previous_z_integral = z_integral;
        previous_z = z;
    }

    validate_profile_result(profile.re)
}

fn validate_profile_result(profile: Real) -> Result<Real, ComptonError> {
    if !profile.is_finite() {
        return Err(ComptonError::NonFiniteResult {
            name: "profile",
            value: profile,
        });
    }
    Ok(profile)
}

fn compton_window_weight(window: ComptonWindow, zp: Real, cutoff: Real) -> Real {
    match window {
        ComptonWindow::Rectangular => {
            if zp.abs() > cutoff {
                0.0
            } else {
                1.0
            }
        }
        ComptonWindow::CosineSquared => {
            if zp.abs() > cutoff {
                0.0
            } else {
                (std::f64::consts::PI * zp.abs() / (2.0 * cutoff))
                    .cos()
                    .powi(2)
            }
        }
        ComptonWindow::Unwindowed => 1.0,
    }
}
