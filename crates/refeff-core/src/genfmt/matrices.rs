use super::lambda::checked_i32;
use super::polynomial::alternating_sign;
use super::validation::*;
use super::*;

/// Build FEFF `fmtrxi` scattering-amplitude F matrix for one energy and leg pair.
///
/// The output is equivalent to FEFF `fmati(1:lam1x,1:lam2x,ilegp)` and uses
/// Fortran-order ndarray storage. The implementation keeps FEFF's j-averaged
/// phase-shift branch,
/// `(exp(2i ph(-l))-1)/(2i) * (l+1) + (exp(2i ph(l))-1)/(2i) * l`,
/// while reporting invalid shapes and non-finite inputs as Rust errors instead
/// of relying on common-block dimensions.
pub fn scattering_amplitude_matrix(
    input: ScatteringAmplitudeMatrixInput<'_>,
) -> Result<Array2<Complex>, GenfmtError> {
    let phase_offset = validate_scattering_amplitude_input(input)?;
    let angular_count = checked_count("angular_limit", input.angular_limit)?;
    let max_lambda_count = input.left_lambda_count.max(input.right_lambda_count);
    let max_m = input.angular_limit;
    let max_n = lambda_n_limit(input.n_indices, max_lambda_count)?;
    let max_m_count = checked_count("angular_limit", max_m)?;
    let max_n_count = checked_count("nlam", max_n)?;
    let mut gam = Array3::<Complex>::zeros((angular_count, max_m_count, max_n_count).f());
    let mut gamtl = Array3::<Complex>::zeros((angular_count, max_m_count, max_n_count).f());

    for l in 0..=input.angular_limit {
        let t_matrix = averaged_t_matrix(input.phase_shifts, phase_offset, l)?;
        for lambda in 0..max_lambda_count {
            let magnetic = lambda_abs_magnetic(input.m_indices[lambda], lambda)?;
            if magnetic > l {
                continue;
            }
            let order = lambda_order(input.n_indices[lambda], lambda)?;
            if order > max_n {
                continue;
            }

            if lambda < input.left_lambda_count {
                let combined_mn =
                    magnetic
                        .checked_add(order)
                        .ok_or(GenfmtError::InvalidLambdaIndex {
                            index: lambda,
                            field: "nlam",
                            value: input.n_indices[lambda],
                        })?;
                let normalization = xnlm_entry(input.xnlm, magnetic, l)?;
                gam[(l, magnetic, order)] = if combined_mn <= l {
                    let sign = alternating_sign(magnetic);
                    normalization
                        * sign
                        * complex_entry(
                            input.first_leg_polynomials,
                            "first_leg_polynomials",
                            l,
                            combined_mn,
                        )?
                } else {
                    Complex::new(0.0, 0.0)
                };
            }

            if lambda < input.right_lambda_count {
                let normalization = xnlm_entry(input.xnlm, magnetic, l)?;
                gamtl[(l, magnetic, order)] = t_matrix / normalization
                    * complex_entry(
                        input.second_leg_polynomials,
                        "second_leg_polynomials",
                        l,
                        order,
                    )?;
            }
        }
    }

    let mut matrix =
        Array2::<Complex>::zeros((input.left_lambda_count, input.right_lambda_count).f());
    for left in 0..input.left_lambda_count {
        let m1 = input.m_indices[left];
        let n1 = lambda_order(input.n_indices[left], left)?;
        let abs_m1 = lambda_abs_magnetic(m1, left)?;
        for right in 0..input.right_lambda_count {
            let m2 = input.m_indices[right];
            let n2 = lambda_order(input.n_indices[right], right)?;
            let abs_m2 = lambda_abs_magnetic(m2, right)?;
            let combined_mn = abs_m1
                .checked_add(n1)
                .ok_or(GenfmtError::InvalidLambdaIndex {
                    index: left,
                    field: "nlam",
                    value: input.n_indices[left],
                })?;
            let l_min = abs_m1.max(abs_m2).max(combined_mn).max(n2);
            let mut value = Complex::new(0.0, 0.0);

            for l in l_min..=input.angular_limit {
                if abs_m1 > l || abs_m2 > l {
                    continue;
                }
                let rotation =
                    rotation_entry(input.rotation, input.rotation_magnetic_offset, l, m1, m2)?;
                value += gam[(l, abs_m1, n1)] * gamtl[(l, abs_m2, n2)] * rotation;
            }

            if input.eta != 0.0 {
                value *= (-Complex::new(0.0, 1.0) * input.eta * (m1 as Real)).exp();
            }
            matrix[(left, right)] = value;
        }
    }

    Ok(matrix)
}

/// Build FEFF `mmtrxi` polarized scattering-amplitude matrix.
///
/// This is the polarization branch that contracts FEFF's energy-independent
/// transition matrix `bmati`, radial transition factors `rkk`, curved-wave
/// polynomial tables, and lambda indices into `fmati(1:lam1x,1:lam1x,ilegp)`.
/// The output uses Fortran-order ndarray storage and preserves FEFF's
/// transition loop order.
pub fn polarized_scattering_amplitude_matrix(
    input: PolarizedScatteringAmplitudeInput<'_>,
) -> Result<Array2<Complex>, GenfmtError> {
    validate_polarized_scattering_amplitude_input(input)?;
    let mut matrix = Array2::<Complex>::zeros((input.lambda_count, input.lambda_count).f());
    if input.lambda_count == 0 {
        return Ok(matrix);
    }

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let Some((min_l, max_l)) = active_transition_limits(&transition_l) else {
        return Ok(matrix);
    };
    let angular_count = checked_count("lind", max_l)?;
    let max_n = lambda_n_limit(input.n_indices, input.lambda_count)?;
    let max_n_count = checked_count("nlam", max_n)?;
    let mut gam = Array3::<Complex>::zeros((angular_count, angular_count, max_n_count).f());
    let mut gamtl = Array3::<Complex>::zeros((angular_count, angular_count, max_n_count).f());

    for l in min_l..=max_l {
        let t_matrix = (2 * l + 1) as Real;
        for lambda in 0..input.lambda_count {
            let signed_magnetic = input.m_indices[lambda];
            if signed_magnetic < 0 {
                continue;
            }
            let magnetic = lambda_abs_magnetic(signed_magnetic, lambda)?;
            if magnetic > l {
                continue;
            }
            let order = lambda_order(input.n_indices[lambda], lambda)?;
            let combined_mn =
                magnetic
                    .checked_add(order)
                    .ok_or(GenfmtError::InvalidLambdaIndex {
                        index: lambda,
                        field: "nlam",
                        value: input.n_indices[lambda],
                    })?;
            let normalization = xnlm_entry(input.xnlm, magnetic, l)?;
            gam[(l, magnetic, order)] = if combined_mn <= l {
                let sign = alternating_sign(magnetic);
                normalization
                    * sign
                    * complex_entry(
                        input.first_leg_polynomials,
                        "first_leg_polynomials",
                        l,
                        combined_mn,
                    )?
            } else {
                Complex::new(0.0, 0.0)
            };
            gamtl[(l, magnetic, order)] = t_matrix / normalization
                * complex_entry(
                    input.second_leg_polynomials,
                    "second_leg_polynomials",
                    l,
                    order,
                )?;
        }
    }

    for left in 0..input.lambda_count {
        let m1 = input.m_indices[left];
        let n1 = lambda_order(input.n_indices[left], left)?;
        let abs_m1 = lambda_abs_magnetic(m1, left)?;
        for right in 0..input.lambda_count {
            let m2 = input.m_indices[right];
            let n2 = lambda_order(input.n_indices[right], right)?;
            let abs_m2 = lambda_abs_magnetic(m2, right)?;
            let mut value = Complex::new(0.0, 0.0);

            for (k1, &maybe_l1) in transition_l.iter().enumerate() {
                let Some(l1) = maybe_l1 else {
                    continue;
                };
                if abs_m1 > l1 {
                    continue;
                }
                for (k2, &maybe_l2) in transition_l.iter().enumerate() {
                    let Some(l2) = maybe_l2 else {
                        continue;
                    };
                    if abs_m2 > l2 {
                        continue;
                    }
                    value += transition_matrix_entry(
                        input.transition_matrix,
                        input.transition_magnetic_offset,
                        m1,
                        k1,
                        m2,
                        k2,
                    )? * complex_vector_entry(input.radial_factors, "radial_factors", k1)?
                        * complex_vector_entry(input.radial_factors, "radial_factors", k2)?
                        * gam[(l1, abs_m1, n1)]
                        * gamtl[(l2, abs_m2, n2)];
                }
            }

            matrix[(left, right)] =
                value * (-Complex::new(0.0, 1.0) * input.eta * (m1 as Real)).exp();
        }
    }

    Ok(matrix)
}

/// Build FEFF `mmtr` energy-independent transition matrix.
///
/// FEFF calls `bcoef` before this step; this helper starts from the resulting
/// `bmat` tensor and applies the `mmtr.f90` rotation and phase rules. The
/// returned ndarray has FEFF `bmati(mu1,k1,mu2,k2)` axis order with signed
/// magnetic indices shifted by `magnetic_limit`.
pub fn energy_independent_transition_matrix(
    input: EnergyIndependentMatrixInput<'_>,
) -> Result<Array4<Complex>, GenfmtError> {
    validate_energy_independent_matrix_input(input)?;
    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let transition_count = transition_l.len();
    let magnetic_dim = checked_double_plus_one("magnetic_limit", input.magnetic_limit)?;
    let mut matrix = Array4::<Complex>::zeros(
        (
            magnetic_dim,
            transition_count,
            magnetic_dim,
            transition_count,
        )
            .f(),
    );
    if transition_count == 0 {
        return Ok(matrix);
    }

    let active_limit = input.magnetic_limit.min(input.initial_l);
    let active_limit_i32 = checked_i32("initial_l", active_limit)?;
    for mu1 in -active_limit_i32..=active_limit_i32 {
        let mu1_index = signed_magnetic_index(
            mu1,
            input.magnetic_limit,
            "magnetic_limit",
            "bmati",
            "mu1",
            magnetic_dim,
        )?;
        for mu2 in -active_limit_i32..=active_limit_i32 {
            let mu2_index = signed_magnetic_index(
                mu2,
                input.magnetic_limit,
                "magnetic_limit",
                "bmati",
                "mu2",
                magnetic_dim,
            )?;

            match input.rotations {
                TransitionRotationInput::Polarized {
                    first_rotation,
                    last_rotation,
                    first_eta,
                    last_eta,
                } => {
                    for (k1, &maybe_l1) in transition_l.iter().enumerate() {
                        let Some(l1) = maybe_l1 else {
                            continue;
                        };
                        let l1_i32 = checked_i32("lind", l1)?;
                        for (k2, &maybe_l2) in transition_l.iter().enumerate() {
                            let Some(l2) = maybe_l2 else {
                                continue;
                            };
                            let l2_i32 = checked_i32("lind", l2)?;
                            for m1 in -l1_i32..=l1_i32 {
                                for m2 in -l2_i32..=l2_i32 {
                                    let phase = (-Complex::new(0.0, 1.0)
                                        * (last_eta * (m2 as Real) + first_eta * (m1 as Real)))
                                        .exp();
                                    let first = rotation_entry(
                                        first_rotation,
                                        input.rotation_magnetic_offset,
                                        l1,
                                        mu1,
                                        m1,
                                    )?;
                                    let last = rotation_entry(
                                        last_rotation,
                                        input.rotation_magnetic_offset,
                                        l2,
                                        m2,
                                        mu2,
                                    )?;
                                    matrix[(mu1_index, k1, mu2_index, k2)] +=
                                        transition_b_matrix_entry(
                                            input.transition_b_matrix,
                                            input.transition_magnetic_offset,
                                            m1,
                                            input.spin_index,
                                            k1,
                                            m2,
                                            k2,
                                        )? * phase
                                            * first
                                            * last;
                                }
                            }
                        }
                    }
                }
                TransitionRotationInput::Unpolarized { combined_rotation } => {
                    for (k1, &maybe_l1) in transition_l.iter().enumerate() {
                        let Some(l1) = maybe_l1 else {
                            continue;
                        };
                        matrix[(mu1_index, k1, mu2_index, k1)] += transition_b_matrix_entry(
                            input.transition_b_matrix,
                            input.transition_magnetic_offset,
                            0,
                            input.spin_index,
                            k1,
                            0,
                            k1,
                        )? * rotation_entry(
                            combined_rotation,
                            input.rotation_magnetic_offset,
                            l1,
                            mu1,
                            mu2,
                        )?;
                    }
                }
            }
        }
    }

    Ok(matrix)
}

fn validate_scattering_amplitude_input(
    input: ScatteringAmplitudeMatrixInput<'_>,
) -> Result<usize, GenfmtError> {
    let angular_count = checked_count("angular_limit", input.angular_limit)?;
    validate_positive_limit("angular_limit", angular_count)?;
    validate_finite_scalar("eta", input.eta)?;

    let lambda_len = input.m_indices.len().min(input.n_indices.len());
    validate_lambda_count("left_lambda_count", input.left_lambda_count, lambda_len)?;
    validate_lambda_count("right_lambda_count", input.right_lambda_count, lambda_len)?;

    let phase_len = input.phase_shifts.len();
    if phase_len == 0 || phase_len.is_multiple_of(2) {
        return Err(GenfmtError::InvalidSignedPhaseShape { length: phase_len });
    }
    let phase_offset = phase_len / 2;
    let phase_required = phase_offset
        .checked_add(input.angular_limit)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "angular_limit",
            value: input.angular_limit,
        })?;
    ensure_axis_len("phase_shifts", "signed_l", phase_len, phase_required)?;

    ensure_axis_len("xnlm", "m", input.xnlm.shape()[0], angular_count)?;
    ensure_axis_len("xnlm", "l", input.xnlm.shape()[1], angular_count)?;
    ensure_axis_len(
        "first_leg_polynomials",
        "l",
        input.first_leg_polynomials.shape()[0],
        angular_count,
    )?;
    ensure_axis_len(
        "second_leg_polynomials",
        "l",
        input.second_leg_polynomials.shape()[0],
        angular_count,
    )?;
    ensure_axis_len("rotation", "l", input.rotation.shape()[0], angular_count)?;

    let rotation_required = input
        .rotation_magnetic_offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "rotation_magnetic_offset",
            value: input.rotation_magnetic_offset,
        })?;
    ensure_axis_len(
        "rotation",
        "m1",
        input.rotation.shape()[1],
        rotation_required,
    )?;
    ensure_axis_len(
        "rotation",
        "m2",
        input.rotation.shape()[2],
        rotation_required,
    )?;

    Ok(phase_offset)
}

fn validate_polarized_scattering_amplitude_input(
    input: PolarizedScatteringAmplitudeInput<'_>,
) -> Result<(), GenfmtError> {
    validate_finite_scalar("eta", input.eta)?;
    let lambda_len = input.m_indices.len().min(input.n_indices.len());
    validate_lambda_count("lambda_count", input.lambda_count, lambda_len)?;

    let transition_count = input.transition_angular_momenta.len();
    ensure_axis_len(
        "radial_factors",
        "transition",
        input.radial_factors.len(),
        transition_count,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "transition1",
        input.transition_matrix.shape()[1],
        transition_count,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "transition2",
        input.transition_matrix.shape()[3],
        transition_count,
    )?;

    let magnetic_required = input
        .transition_magnetic_offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "transition_magnetic_offset",
            value: input.transition_magnetic_offset,
        })?;
    ensure_axis_len(
        "transition_matrix",
        "m1",
        input.transition_matrix.shape()[0],
        magnetic_required,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "m2",
        input.transition_matrix.shape()[2],
        magnetic_required,
    )?;

    if let Some((_, max_l)) = active_transition_limits(&transition_angular_momenta(
        input.transition_angular_momenta,
    )?) {
        let angular_count = checked_count("lind", max_l)?;
        ensure_axis_len("xnlm", "m", input.xnlm.shape()[0], angular_count)?;
        ensure_axis_len("xnlm", "l", input.xnlm.shape()[1], angular_count)?;
        ensure_axis_len(
            "first_leg_polynomials",
            "l",
            input.first_leg_polynomials.shape()[0],
            angular_count,
        )?;
        ensure_axis_len(
            "second_leg_polynomials",
            "l",
            input.second_leg_polynomials.shape()[0],
            angular_count,
        )?;
    }

    Ok(())
}

fn validate_energy_independent_matrix_input(
    input: EnergyIndependentMatrixInput<'_>,
) -> Result<(), GenfmtError> {
    validate_positive_limit(
        "magnetic_limit",
        checked_count("magnetic_limit", input.magnetic_limit)?,
    )?;
    validate_positive_limit(
        "rotation_magnetic_offset",
        checked_count("rotation_magnetic_offset", input.rotation_magnetic_offset)?,
    )?;

    let transition_count = input.transition_angular_momenta.len();
    ensure_axis_len(
        "transition_b_matrix",
        "transition1",
        input.transition_b_matrix.shape()[2],
        transition_count,
    )?;
    ensure_axis_len(
        "transition_b_matrix",
        "transition2",
        input.transition_b_matrix.shape()[5],
        transition_count,
    )?;
    ensure_axis_len(
        "transition_b_matrix",
        "spin1",
        input.transition_b_matrix.shape()[1],
        input.spin_index + 1,
    )?;
    ensure_axis_len(
        "transition_b_matrix",
        "spin2",
        input.transition_b_matrix.shape()[4],
        input.spin_index + 1,
    )?;
    let transition_magnetic_required = input
        .transition_magnetic_offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "transition_magnetic_offset",
            value: input.transition_magnetic_offset,
        })?;
    ensure_axis_len(
        "transition_b_matrix",
        "m1",
        input.transition_b_matrix.shape()[0],
        transition_magnetic_required,
    )?;
    ensure_axis_len(
        "transition_b_matrix",
        "m2",
        input.transition_b_matrix.shape()[3],
        transition_magnetic_required,
    )?;

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    if let Some((_, max_l)) = active_transition_limits(&transition_l) {
        let angular_count = checked_count("lind", max_l)?;
        match input.rotations {
            TransitionRotationInput::Polarized {
                first_rotation,
                last_rotation,
                first_eta,
                last_eta,
            } => {
                validate_finite_scalar("first_eta", first_eta)?;
                validate_finite_scalar("last_eta", last_eta)?;
                validate_rotation_table(
                    "first_rotation",
                    first_rotation,
                    input.rotation_magnetic_offset,
                    angular_count,
                )?;
                validate_rotation_table(
                    "last_rotation",
                    last_rotation,
                    input.rotation_magnetic_offset,
                    angular_count,
                )?;
            }
            TransitionRotationInput::Unpolarized { combined_rotation } => {
                validate_rotation_table(
                    "combined_rotation",
                    combined_rotation,
                    input.rotation_magnetic_offset,
                    angular_count,
                )?;
            }
        }
    }

    Ok(())
}

fn validate_rotation_table(
    name: &'static str,
    rotation: ArrayView3<'_, Real>,
    offset: usize,
    angular_count: usize,
) -> Result<(), GenfmtError> {
    ensure_axis_len(name, "l", rotation.shape()[0], angular_count)?;
    let magnetic_required = offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "rotation_magnetic_offset",
            value: offset,
        })?;
    ensure_axis_len(name, "m1", rotation.shape()[1], magnetic_required)?;
    ensure_axis_len(name, "m2", rotation.shape()[2], magnetic_required)?;
    Ok(())
}

fn validate_lambda_count(
    name: &'static str,
    requested: usize,
    available: usize,
) -> Result<(), GenfmtError> {
    if requested <= available {
        Ok(())
    } else {
        Err(GenfmtError::LambdaCountOutOfRange {
            name,
            requested,
            available,
        })
    }
}

fn ensure_axis_len(
    table: &'static str,
    axis: &'static str,
    length: usize,
    required: usize,
) -> Result<(), GenfmtError> {
    if length >= required {
        Ok(())
    } else {
        Err(GenfmtError::TableAxisTooShort {
            table,
            axis,
            length,
            required,
        })
    }
}

fn lambda_n_limit(n_indices: ArrayView1<'_, i32>, count: usize) -> Result<usize, GenfmtError> {
    let mut max_n = 0;
    for index in 0..count {
        max_n = max_n.max(lambda_order(n_indices[index], index)?);
    }
    Ok(max_n)
}

fn transition_angular_momenta(
    transition_l: ArrayView1<'_, i32>,
) -> Result<Vec<Option<usize>>, GenfmtError> {
    transition_l
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if value < 0 {
                Ok(None)
            } else {
                let angular_momentum =
                    usize::try_from(value).map_err(|_| GenfmtError::InvalidLambdaIndex {
                        index,
                        field: "lind",
                        value,
                    })?;
                Ok(Some(angular_momentum))
            }
        })
        .collect()
}

fn active_transition_limits(transition_l: &[Option<usize>]) -> Option<(usize, usize)> {
    let mut limits: Option<(usize, usize)> = None;
    for &angular_momentum in transition_l.iter().flatten() {
        limits = Some(match limits {
            Some((minimum, maximum)) => {
                (minimum.min(angular_momentum), maximum.max(angular_momentum))
            }
            None => (angular_momentum, angular_momentum),
        });
    }
    limits
}

fn lambda_order(value: i32, index: usize) -> Result<usize, GenfmtError> {
    usize::try_from(value).map_err(|_| GenfmtError::InvalidLambdaIndex {
        index,
        field: "nlam",
        value,
    })
}

fn lambda_abs_magnetic(value: i32, index: usize) -> Result<usize, GenfmtError> {
    usize::try_from(value.unsigned_abs()).map_err(|_| GenfmtError::InvalidLambdaIndex {
        index,
        field: "mlam",
        value,
    })
}

fn averaged_t_matrix(
    phase_shifts: ArrayView1<'_, Complex>,
    phase_offset: usize,
    angular_momentum: usize,
) -> Result<Complex, GenfmtError> {
    let negative = complex_vector_entry(
        phase_shifts,
        "phase_shifts",
        phase_offset - angular_momentum,
    )?;
    let positive = complex_vector_entry(
        phase_shifts,
        "phase_shifts",
        phase_offset + angular_momentum,
    )?;
    let imaginary = Complex::new(0.0, 1.0);
    let negative_t =
        ((2.0 * imaginary * negative).exp() - Complex::new(1.0, 0.0)) / (2.0 * imaginary);
    let positive_t =
        ((2.0 * imaginary * positive).exp() - Complex::new(1.0, 0.0)) / (2.0 * imaginary);
    Ok(negative_t * (angular_momentum as Real + 1.0) + positive_t * angular_momentum as Real)
}

fn xnlm_entry(
    xnlm: ArrayView2<'_, Real>,
    magnetic: usize,
    angular_momentum: usize,
) -> Result<Real, GenfmtError> {
    let value = real_entry(xnlm, "xnlm", magnetic, angular_momentum)?;
    if value == 0.0 {
        Err(GenfmtError::ZeroLegendreNormalization {
            angular_momentum,
            magnetic,
        })
    } else {
        Ok(value)
    }
}

fn rotation_entry(
    rotation: ArrayView3<'_, Real>,
    offset: usize,
    angular_momentum: usize,
    first_magnetic: i32,
    second_magnetic: i32,
) -> Result<Real, GenfmtError> {
    let first = signed_magnetic_index(
        first_magnetic,
        offset,
        "rotation_magnetic_offset",
        "rotation",
        "m1",
        rotation.shape()[1],
    )?;
    let second = signed_magnetic_index(
        second_magnetic,
        offset,
        "rotation_magnetic_offset",
        "rotation",
        "m2",
        rotation.shape()[2],
    )?;
    let value = rotation[(angular_momentum, first, second)];
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableScalar {
            table: "rotation",
            row: angular_momentum,
            column: first,
            value,
        })
    }
}

fn transition_matrix_entry(
    transition_matrix: ArrayView4<'_, Complex>,
    offset: usize,
    first_magnetic: i32,
    first_transition: usize,
    second_magnetic: i32,
    second_transition: usize,
) -> Result<Complex, GenfmtError> {
    let first = signed_magnetic_index(
        first_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_matrix",
        "m1",
        transition_matrix.shape()[0],
    )?;
    let second = signed_magnetic_index(
        second_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_matrix",
        "m2",
        transition_matrix.shape()[2],
    )?;
    complex4_entry(
        transition_matrix,
        "transition_matrix",
        first,
        first_transition,
        second,
        second_transition,
    )
}

fn transition_b_matrix_entry(
    transition_b_matrix: ArrayView6<'_, Complex>,
    offset: usize,
    first_magnetic: i32,
    spin_index: usize,
    first_transition: usize,
    second_magnetic: i32,
    second_transition: usize,
) -> Result<Complex, GenfmtError> {
    let first = signed_magnetic_index(
        first_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_b_matrix",
        "m1",
        transition_b_matrix.shape()[0],
    )?;
    let second = signed_magnetic_index(
        second_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_b_matrix",
        "m2",
        transition_b_matrix.shape()[3],
    )?;
    complex6_entry(
        transition_b_matrix,
        "transition_b_matrix",
        [
            first,
            spin_index,
            first_transition,
            second,
            spin_index,
            second_transition,
        ],
    )
}

fn signed_magnetic_index(
    value: i32,
    offset: usize,
    offset_name: &'static str,
    table: &'static str,
    axis: &'static str,
    length: usize,
) -> Result<usize, GenfmtError> {
    let magnitude =
        usize::try_from(value.unsigned_abs()).map_err(|_| GenfmtError::InvalidLambdaIndex {
            index: 0,
            field: "mlam",
            value,
        })?;
    let required = offset
        .checked_add(magnitude)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: offset_name,
            value: offset,
        })?;
    let index = if value < 0 {
        offset.checked_sub(magnitude)
    } else {
        offset.checked_add(magnitude)
    }
    .ok_or(GenfmtError::TableAxisTooShort {
        table,
        axis,
        length,
        required,
    })?;
    ensure_axis_len(table, axis, length, index + 1)?;
    Ok(index)
}

fn complex_vector_entry(
    vector: ArrayView1<'_, Complex>,
    table: &'static str,
    index: usize,
) -> Result<Complex, GenfmtError> {
    ensure_axis_len(table, "index", vector.len(), index + 1)?;
    let value = vector[index];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableComplex {
            table,
            row: index,
            column: 0,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn complex_entry(
    table: ArrayView2<'_, Complex>,
    name: &'static str,
    row: usize,
    column: usize,
) -> Result<Complex, GenfmtError> {
    ensure_axis_len(name, "row", table.shape()[0], row + 1)?;
    ensure_axis_len(name, "column", table.shape()[1], column + 1)?;
    let value = table[(row, column)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableComplex {
            table: name,
            row,
            column,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn complex4_entry(
    table: ArrayView4<'_, Complex>,
    name: &'static str,
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> Result<Complex, GenfmtError> {
    ensure_axis_len(name, "axis0", table.shape()[0], i0 + 1)?;
    ensure_axis_len(name, "axis1", table.shape()[1], i1 + 1)?;
    ensure_axis_len(name, "axis2", table.shape()[2], i2 + 1)?;
    ensure_axis_len(name, "axis3", table.shape()[3], i3 + 1)?;
    let value = table[(i0, i1, i2, i3)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTensorComplex {
            table: name,
            i0,
            i1,
            i2,
            i3,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn complex6_entry(
    table: ArrayView6<'_, Complex>,
    name: &'static str,
    index: [usize; 6],
) -> Result<Complex, GenfmtError> {
    let [i0, i1, i2, i3, i4, i5] = index;
    ensure_axis_len(name, "axis0", table.shape()[0], i0 + 1)?;
    ensure_axis_len(name, "axis1", table.shape()[1], i1 + 1)?;
    ensure_axis_len(name, "axis2", table.shape()[2], i2 + 1)?;
    ensure_axis_len(name, "axis3", table.shape()[3], i3 + 1)?;
    ensure_axis_len(name, "axis4", table.shape()[4], i4 + 1)?;
    ensure_axis_len(name, "axis5", table.shape()[5], i5 + 1)?;
    let value = table[(i0, i1, i2, i3, i4, i5)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTensor6Complex {
            table: name,
            i0,
            i1,
            i2,
            i3,
            i4,
            i5,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn real_entry(
    table: ArrayView2<'_, Real>,
    name: &'static str,
    row: usize,
    column: usize,
) -> Result<Real, GenfmtError> {
    ensure_axis_len(name, "row", table.shape()[0], row + 1)?;
    ensure_axis_len(name, "column", table.shape()[1], column + 1)?;
    let value = table[(row, column)];
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableScalar {
            table: name,
            row,
            column,
            value,
        })
    }
}
