use super::lambda::checked_i32;
use super::polynomial::alternating_sign;
use super::validation::*;
use super::*;
use crate::{wigner_3j, wigner_rotation};

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

/// Build FEFF `mmtrjas0` spherical JAS/NRIXS transition tensor.
///
/// FEFF's original routine generates the final-state doubled-`j` list from
/// `kinit`, `ljmax`, `jmax`, and `lx1`, then rotates the spin-resolved
/// spherical transition amplitudes into `hbmatrs`. This helper keeps that
/// generation step explicit in the returned [`JasSpinTransitionMatrix`] while
/// storing the `mj` axis compactly as `mj = -initial_j2 + 2 * row`.
pub fn jas_spin_transition_matrix(
    input: JasSpinTransitionInput<'_>,
) -> Result<JasSpinTransitionMatrix, GenfmtError> {
    let (mj_count, magnetic_dim, generated_final_j2) = validate_jas_spin_transition_input(input)?;
    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let mut matrix =
        Array5::<Complex>::zeros((mj_count, 2, magnetic_dim, magnetic_dim, transition_l.len()).f());
    if transition_l.is_empty() {
        return Ok(JasSpinTransitionMatrix {
            matrix,
            generated_final_j2,
        });
    }

    let magnetic_limit = checked_i32("rotation_magnetic_offset", input.rotation_magnetic_offset)?;
    for (transition, &maybe_l1) in transition_l.iter().enumerate() {
        let Some(l1) = maybe_l1 else {
            continue;
        };
        let l1_i32 = checked_i32("lind", l1)?;
        let final_j2 = generated_final_j2[transition];

        for mu1 in -magnetic_limit..=magnetic_limit {
            let mu1_index = signed_magnetic_index(
                mu1,
                input.rotation_magnetic_offset,
                "rotation_magnetic_offset",
                "hbmatrs",
                "mu1",
                magnetic_dim,
            )?;
            for mu2 in -magnetic_limit..=magnetic_limit {
                let mu2_index = signed_magnetic_index(
                    mu2,
                    input.rotation_magnetic_offset,
                    "rotation_magnetic_offset",
                    "hbmatrs",
                    "mu2",
                    magnetic_dim,
                )?;
                for m1 in -l1_i32..=l1_i32 {
                    let first_rotation = rotation_entry(
                        input.first_rotation,
                        input.rotation_magnetic_offset,
                        l1,
                        mu1,
                        m1,
                    )?;
                    if first_rotation == 0.0 {
                        continue;
                    }
                    let first_phase =
                        (-Complex::new(0.0, 1.0) * input.first_eta * (m1 as Real)).exp();

                    for spin1 in 0..input.spin_channels {
                        let spin1_i32 = spin1 as i32;
                        let spin1_mj2 = 2 * spin1_i32 - 1;
                        let mj = 2 * m1 + spin1_mj2;
                        if mj.abs() > final_j2 {
                            continue;
                        }
                        let Some(mj_row) = compact_doubled_j_row(input.initial_j2, mj) else {
                            continue;
                        };
                        let first_coupling = jas_spin_coupling(l1_i32, final_j2, spin1_i32, mj)?;

                        for spin2 in 0..input.spin_channels {
                            let spin2_i32 = spin2 as i32;
                            let spin2_mj2 = 2 * spin2_i32 - 1;
                            let m2 = (mj - spin2_mj2) / 2;
                            if m2 > l1_i32 {
                                continue;
                            }
                            let second_coupling =
                                jas_spin_coupling(l1_i32, final_j2, spin2_i32, mj)?;
                            let last_rotation = rotation_entry(
                                input.last_rotation,
                                input.rotation_magnetic_offset,
                                l1,
                                m2,
                                mu2,
                            )?;
                            if last_rotation == 0.0 {
                                continue;
                            }
                            let last_phase =
                                (-Complex::new(0.0, 1.0) * input.last_eta * (m2 as Real)).exp();
                            matrix[(mj_row, spin2, mu2_index, mu1_index, transition)] +=
                                first_coupling
                                    * second_coupling
                                    * first_phase
                                    * first_rotation
                                    * last_phase
                                    * last_rotation;
                        }
                    }
                }
            }
        }
    }

    Ok(JasSpinTransitionMatrix {
        matrix,
        generated_final_j2,
    })
}

/// Build FEFF `mmtrjas` one-sided JAS/NRIXS transition tensors.
///
/// This is the q-resolved companion to [`jas_spin_transition_matrix`]. It
/// computes FEFF's `bcoefjas` weights for each compact `mj` row, rotates them
/// through the q-vector basis and endpoint path rotations, then returns the
/// `hbmatl`/`hbmatr` tensors in the axis order expected by
/// [`jas_left_right_amplitude_matrices`].
pub fn jas_one_sided_transition_matrices(
    input: JasOneSidedTransitionInput<'_>,
) -> Result<JasOneSidedTransitionMatrices, GenfmtError> {
    let (mj_count, magnetic_dim, generated_final_j2) =
        validate_jas_one_sided_transition_input(input)?;
    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let transition_count = transition_l.len();
    let q_count = input.q_phases.len();
    let mut left_matrix =
        Array4::<Complex>::zeros((mj_count, magnetic_dim, q_count, transition_count).f());
    let mut right_matrix =
        Array4::<Complex>::zeros((mj_count, magnetic_dim, q_count, transition_count).f());
    if transition_count == 0 || q_count == 0 {
        return Ok(JasOneSidedTransitionMatrices {
            left_matrix,
            right_matrix,
            generated_final_j2,
        });
    }

    let weights = jas_one_sided_transition_weights(input, &generated_final_j2)?;
    let max_active_l = active_transition_limits(&transition_l)
        .map(|(_, max_l)| max_l)
        .unwrap_or(0);
    let output_magnetic_limit = input.rotation_magnetic_offset.min(max_active_l);
    let output_magnetic_limit_i32 = checked_i32("rotation_magnetic_offset", output_magnetic_limit)?;
    let first_endpoint_phase = (-Complex::new(0.0, 1.0) * input.first_eta).exp();
    let last_endpoint_phase = (-Complex::new(0.0, 1.0) * input.last_eta).exp();

    for (transition, &maybe_l1) in transition_l.iter().enumerate() {
        let Some(l1) = maybe_l1 else {
            continue;
        };
        let l1_i32 = checked_i32("lind", l1)?;
        let magnetic_limit = output_magnetic_limit_i32.min(l1_i32);
        let left_phase = imaginary_unit_power(-input.final_lj_momenta[transition] - l1_i32);
        let right_phase = imaginary_unit_power(input.final_lj_momenta[transition] + l1_i32);

        for mu in -output_magnetic_limit_i32..=output_magnetic_limit_i32 {
            let mu_index = signed_magnetic_index(
                mu,
                input.rotation_magnetic_offset,
                "rotation_magnetic_offset",
                "jas_one_sided_transition_matrix",
                "mu",
                magnetic_dim,
            )?;
            for q in 0..q_count {
                let q_phase = complex_vector_entry(input.q_phases, "q_phases", q)?;
                let q_beta = real_vector_entry(input.q_beta_angles, "q_beta_angles", q)?;
                for m1 in -magnetic_limit..=magnetic_limit {
                    let left_rotation = jas_q_rotated_left_entry(
                        input,
                        JasQRotationContext {
                            angular_momentum: l1,
                            magnetic_limit,
                            output_magnetic: mu,
                            internal_magnetic: m1,
                            q_phase,
                            q_beta,
                            endpoint_phase: first_endpoint_phase,
                        },
                    )?;
                    let right_rotation = jas_q_rotated_right_entry(
                        input,
                        JasQRotationContext {
                            angular_momentum: l1,
                            magnetic_limit,
                            output_magnetic: mu,
                            internal_magnetic: m1,
                            q_phase,
                            q_beta,
                            endpoint_phase: last_endpoint_phase,
                        },
                    )?;
                    if left_rotation == Complex::new(0.0, 0.0)
                        && right_rotation == Complex::new(0.0, 0.0)
                    {
                        continue;
                    }

                    for spin in 0..input.spin_channels {
                        let mj = 2 * m1 + (2 * spin as i32 - 1);
                        if mj.abs() > input.initial_j2 {
                            continue;
                        }
                        let Some(mj_row) = compact_doubled_j_row(input.initial_j2, mj) else {
                            continue;
                        };
                        let weight = weights[(mj_row, spin, transition)];
                        if weight == 0.0 {
                            continue;
                        }
                        left_matrix[(mj_row, mu_index, q, transition)] +=
                            weight * left_rotation * left_phase;
                        right_matrix[(mj_row, mu_index, q, transition)] +=
                            weight * right_rotation * right_phase;
                    }
                }
            }
        }
    }

    Ok(JasOneSidedTransitionMatrices {
        left_matrix,
        right_matrix,
        generated_final_j2,
    })
}

/// Select and build the FEFF GENFMTJAS transition matrices.
///
/// This ports the driver branch in `genfmtjas.f90`: nonnegative `elpty` calls
/// `mmtrjas` for q-resolved left/right matrices, while negative `elpty` calls
/// `mmtrjas0` for spherical averaging.
pub fn genfmt_jas_transition_matrices(
    input: GenfmtJasTransitionMatricesInput<'_>,
) -> Result<GenfmtJasTransitionMatrices, GenfmtError> {
    validate_finite_scalar("ellipticity", input.ellipticity)?;
    if input.ellipticity >= 0.0 {
        Ok(GenfmtJasTransitionMatrices::LeftRight(
            jas_one_sided_transition_matrices(input.left_right)?,
        ))
    } else {
        Ok(GenfmtJasTransitionMatrices::Spherical(
            jas_spin_transition_matrix(input.spherical)?,
        ))
    }
}

/// Build FEFF `mmtrxijas0` JAS/NRIXS scattering-amplitude matrices.
///
/// This folds q-weighted radial factors, the spin-resolved `hbmatrs` transition
/// tensor, lambda indices, and leg polynomial tables into FEFF's `fmats` and
/// optional `lgfmats` arrays for one energy and leg pair. The full FEFF routine
/// reads most inputs from COMMON/module state; the Rust API takes them
/// explicitly and stores doubled-`j` rows compactly.
pub fn jas_scattering_amplitude_matrices(
    input: JasScatteringAmplitudeInput<'_>,
) -> Result<JasScatteringAmplitudeMatrices, GenfmtError> {
    let mj_count = validate_jas_scattering_amplitude_input(input)?;
    let mut amplitudes =
        Array4::<Complex>::zeros((mj_count, 2, input.lambda_count, input.lambda_count).f());
    let mut decomposed_amplitudes = input.decomposition_l_max.map(|l_max| {
        Array5::<Complex>::zeros(
            (
                mj_count,
                2,
                l_max + 1,
                input.lambda_count,
                input.lambda_count,
            )
                .f(),
        )
    });
    if input.lambda_count == 0 {
        return Ok(JasScatteringAmplitudeMatrices {
            amplitudes,
            decomposed_amplitudes,
        });
    }

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let Some((min_l, raw_max_l)) = active_transition_limits(&transition_l) else {
        return Ok(JasScatteringAmplitudeMatrices {
            amplitudes,
            decomposed_amplitudes,
        });
    };
    let max_l = raw_max_l.min(input.max_angular_momentum);
    if min_l > max_l {
        return Ok(JasScatteringAmplitudeMatrices {
            amplitudes,
            decomposed_amplitudes,
        });
    }

    let angular_count = checked_count("lind", max_l)?;
    let max_n = lambda_n_limit(input.n_indices, input.lambda_count)?;
    let max_n_count = checked_count("nlam", max_n)?;
    let mut gam = Array3::<Complex>::zeros((angular_count, angular_count, max_n_count).f());
    let mut gamtl = Array3::<Complex>::zeros((angular_count, angular_count, max_n_count).f());

    for l in min_l..=max_l {
        let angular_factor = (2 * l + 1) as Real;
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
                normalization
                    * alternating_sign(magnetic)
                    * complex_entry(
                        input.first_leg_polynomials,
                        "first_leg_polynomials",
                        l,
                        combined_mn,
                    )?
            } else {
                Complex::new(0.0, 0.0)
            };
            gamtl[(l, magnetic, order)] = angular_factor / normalization
                * complex_entry(
                    input.second_leg_polynomials,
                    "second_leg_polynomials",
                    l,
                    order,
                )?;
        }
    }

    let q_sums = jas_q_sums(input.radial_factors, input.q_weights)?;
    for left in 0..input.lambda_count {
        let m1 = input.m_indices[left];
        let n1 = lambda_order(input.n_indices[left], left)?;
        let abs_m1 = lambda_abs_magnetic(m1, left)?;
        for right in 0..input.lambda_count {
            let m2 = input.m_indices[right];
            let n2 = lambda_order(input.n_indices[right], right)?;
            let abs_m2 = lambda_abs_magnetic(m2, right)?;

            for (transition, &maybe_l) in transition_l.iter().enumerate() {
                let Some(l) = maybe_l else {
                    continue;
                };
                if l > max_l || abs_m1 > l || abs_m2 > l {
                    continue;
                }
                let angular_average = (2 * l + 1) as Real;
                let radial_polynomial =
                    q_sums[transition] * gamtl[(l, abs_m2, n2)] * gam[(l, abs_m1, n1)]
                        / angular_average;
                if radial_polynomial == Complex::new(0.0, 0.0) {
                    continue;
                }

                for spin in 0..=1 {
                    for mj_row in 0..mj_count {
                        let contribution = radial_polynomial
                            * jas_transition_matrix_entry(
                                input.transition_matrix,
                                input.transition_magnetic_offset,
                                mj_row,
                                spin,
                                m2,
                                m1,
                                transition,
                            )?;
                        amplitudes[(mj_row, spin, right, left)] += contribution;
                        if let Some(ref mut decomposed) = decomposed_amplitudes
                            && l < decomposed.shape()[2]
                        {
                            decomposed[(mj_row, spin, l, right, left)] += contribution;
                        }
                    }
                }
            }
        }

        if let Some(ref mut decomposed) = decomposed_amplitudes {
            let phase = (-Complex::new(0.0, 1.0) * input.eta * (m1 as Real)).exp();
            for right in 0..input.lambda_count {
                for l in 0..decomposed.shape()[2] {
                    for spin in 0..=1 {
                        for mj_row in 0..mj_count {
                            decomposed[(mj_row, spin, l, right, left)] *= phase;
                        }
                    }
                }
            }
        }
    }

    Ok(JasScatteringAmplitudeMatrices {
        amplitudes,
        decomposed_amplitudes,
    })
}

/// Build FEFF `mmtrxijas` left/right JAS scattering-amplitude matrices.
///
/// This is the one-sided companion to [`jas_scattering_amplitude_matrices`]:
/// it folds FEFF's `hbmatl`/`hbmatr` transition tables with q-resolved radial
/// factors and lambda polynomial tables into `fmatl`/`fmatr`, plus optional
/// angular-decomposition tables. FEFF applies the q weights and azimuthal phase
/// only to the left-hand matrices; this helper preserves that asymmetry.
pub fn jas_left_right_amplitude_matrices(
    input: JasLeftRightAmplitudeInput<'_>,
) -> Result<JasLeftRightAmplitudeMatrices, GenfmtError> {
    let mj_count = validate_jas_left_right_amplitude_input(input)?;
    let q_count = input.q_weights.len();
    let mut left_amplitudes = Array3::<Complex>::zeros((mj_count, q_count, input.lambda_count).f());
    let mut right_amplitudes =
        Array3::<Complex>::zeros((mj_count, q_count, input.lambda_count).f());
    let mut decomposed_left_amplitudes = input.decomposition_l_max.map(|l_max| {
        Array4::<Complex>::zeros((mj_count, q_count, l_max + 1, input.lambda_count).f())
    });
    let mut decomposed_right_amplitudes = input.decomposition_l_max.map(|l_max| {
        Array4::<Complex>::zeros((mj_count, q_count, l_max + 1, input.lambda_count).f())
    });
    if input.lambda_count == 0 {
        return Ok(JasLeftRightAmplitudeMatrices {
            left_amplitudes,
            right_amplitudes,
            decomposed_left_amplitudes,
            decomposed_right_amplitudes,
        });
    }

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let Some((min_l, raw_max_l)) = active_transition_limits(&transition_l) else {
        return Ok(JasLeftRightAmplitudeMatrices {
            left_amplitudes,
            right_amplitudes,
            decomposed_left_amplitudes,
            decomposed_right_amplitudes,
        });
    };
    let max_l = raw_max_l.min(input.max_angular_momentum);
    if min_l > max_l {
        return Ok(JasLeftRightAmplitudeMatrices {
            left_amplitudes,
            right_amplitudes,
            decomposed_left_amplitudes,
            decomposed_right_amplitudes,
        });
    }

    let angular_count = checked_count("lind", max_l)?;
    let max_n = lambda_n_limit(input.n_indices, input.lambda_count)?;
    let max_n_count = checked_count("nlam", max_n)?;
    let mut gam = Array3::<Complex>::zeros((angular_count, angular_count, max_n_count).f());
    let mut gamtl = Array3::<Complex>::zeros((angular_count, angular_count, max_n_count).f());

    for l in min_l..=max_l {
        let angular_factor = (2 * l + 1) as Real;
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
                normalization
                    * alternating_sign(magnetic)
                    * complex_entry(
                        input.first_leg_polynomials,
                        "first_leg_polynomials",
                        l,
                        combined_mn,
                    )?
            } else {
                Complex::new(0.0, 0.0)
            };
            gamtl[(l, magnetic, order)] = angular_factor / normalization
                * complex_entry(
                    input.second_leg_polynomials,
                    "second_leg_polynomials",
                    l,
                    order,
                )?;
        }
    }

    for lambda in 0..input.lambda_count {
        let magnetic = input.m_indices[lambda];
        let abs_magnetic = lambda_abs_magnetic(magnetic, lambda)?;
        let order = lambda_order(input.n_indices[lambda], lambda)?;

        for (transition, &maybe_l) in transition_l.iter().enumerate() {
            let Some(l) = maybe_l else {
                continue;
            };
            if l > max_l || abs_magnetic > l {
                continue;
            }
            let left_polynomial = gam[(l, abs_magnetic, order)];
            let right_polynomial = gamtl[(l, abs_magnetic, order)];
            for q in 0..q_count {
                let radial = complex_entry(input.radial_factors, "radial_factors", q, transition)?;
                for mj_row in 0..mj_count {
                    let left_contribution = radial
                        * jas_one_sided_transition_matrix_entry(
                            input.left_transition_matrix,
                            "left_transition_matrix",
                            input.transition_magnetic_offset,
                            mj_row,
                            magnetic,
                            q,
                            transition,
                        )?
                        * left_polynomial;
                    let right_contribution = radial
                        * jas_one_sided_transition_matrix_entry(
                            input.right_transition_matrix,
                            "right_transition_matrix",
                            input.transition_magnetic_offset,
                            mj_row,
                            magnetic,
                            q,
                            transition,
                        )?
                        * right_polynomial;
                    left_amplitudes[(mj_row, q, lambda)] += left_contribution;
                    right_amplitudes[(mj_row, q, lambda)] += right_contribution;

                    if let Some(ref mut decomposed_left) = decomposed_left_amplitudes
                        && l < decomposed_left.shape()[2]
                    {
                        decomposed_left[(mj_row, q, l, lambda)] += left_contribution;
                    }
                    if let Some(ref mut decomposed_right) = decomposed_right_amplitudes
                        && l < decomposed_right.shape()[2]
                    {
                        decomposed_right[(mj_row, q, l, lambda)] += right_contribution;
                    }
                }
            }
        }

        let phase = (-Complex::new(0.0, 1.0) * input.eta * (magnetic as Real)).exp();
        for q in 0..q_count {
            let q_weight = complex_vector_entry(input.q_weights, "q_weights", q)?;
            let left_factor = phase * q_weight;
            for mj_row in 0..mj_count {
                left_amplitudes[(mj_row, q, lambda)] *= left_factor;
            }
            if let Some(ref mut decomposed_left) = decomposed_left_amplitudes {
                for l in 0..decomposed_left.shape()[2] {
                    for mj_row in 0..mj_count {
                        decomposed_left[(mj_row, q, l, lambda)] *= left_factor;
                    }
                }
            }
        }
    }

    Ok(JasLeftRightAmplitudeMatrices {
        left_amplitudes,
        right_amplitudes,
        decomposed_left_amplitudes,
        decomposed_right_amplitudes,
    })
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

/// Build ordinary FEFF GENFMT `bmati` matrices for the active spin loop.
///
/// FEFF calls `mmtr` once per active spin, and `mmtr` rebuilds `bmat` with that
/// loop's spin selector before choosing its working spin slot. The Rust handoff
/// builds `bmat` once with the driver selector. In the two-spin branch that
/// source matrix is the unfurled `ispin=1` tensor, so both FEFF calls are
/// equivalent to selecting its last spin slot: the second call's `ispin=2`
/// fold copies that slot into slot zero before `mmtr` reads it.
pub fn genfmt_ordinary_transition_matrices(
    input: GenfmtOrdinaryTransitionMatricesInput<'_>,
) -> Result<GenfmtOrdinaryTransitionMatrices, GenfmtError> {
    validate_ordinary_transition_spin_inputs(input)?;

    let mut per_spin = Vec::with_capacity(input.active_spin_channel_count);
    let mut b_matrix_spin_indices = Vec::with_capacity(input.active_spin_channel_count);
    for _active_spin in 0..input.active_spin_channel_count {
        let b_matrix_spin_index = if input.active_spin_channel_count == 2 {
            input.available_spin_channels - 1
        } else {
            let mmtr_spin_argument = input.spin_selector;
            mmtr_b_matrix_spin_index(mmtr_spin_argument, input.available_spin_channels)?
        };
        let matrix = energy_independent_transition_matrix(EnergyIndependentMatrixInput {
            transition_angular_momenta: input.transition_angular_momenta,
            transition_b_matrix: input.transition_b_matrix,
            transition_magnetic_offset: input.transition_magnetic_offset,
            spin_index: b_matrix_spin_index,
            initial_l: input.initial_l,
            magnetic_limit: input.magnetic_limit,
            rotation_magnetic_offset: input.rotation_magnetic_offset,
            rotations: input.rotations,
        })?;
        b_matrix_spin_indices.push(b_matrix_spin_index);
        per_spin.push(matrix);
    }

    let shape = per_spin[0].shape();
    let mut matrices = Array5::<Complex>::zeros(
        (
            input.active_spin_channel_count,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
            .f(),
    );
    for (spin, matrix) in per_spin.iter().enumerate() {
        for m1 in 0..shape[0] {
            for transition1 in 0..shape[1] {
                for m2 in 0..shape[2] {
                    for transition2 in 0..shape[3] {
                        matrices[(spin, m1, transition1, m2, transition2)] =
                            matrix[(m1, transition1, m2, transition2)];
                    }
                }
            }
        }
    }

    Ok(GenfmtOrdinaryTransitionMatrices {
        matrices,
        b_matrix_spin_indices,
    })
}

fn validate_ordinary_transition_spin_inputs(
    input: GenfmtOrdinaryTransitionMatricesInput<'_>,
) -> Result<(), GenfmtError> {
    if input.available_spin_channels == 0 || input.available_spin_channels > 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "available_spin_channels",
            value: input.available_spin_channels,
        });
    }
    if input.active_spin_channel_count == 0 || input.active_spin_channel_count > 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "active_spin_channel_count",
            value: input.active_spin_channel_count,
        });
    }
    let expected_spin_count = if input.spin_selector == 1 {
        input.available_spin_channels
    } else {
        1
    };
    if input.active_spin_channel_count != expected_spin_count {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "active_spin_channel_count",
            value: input.active_spin_channel_count,
        });
    }
    Ok(())
}

fn mmtr_b_matrix_spin_index(
    mmtr_spin_argument: i32,
    available_spin_channels: usize,
) -> Result<usize, GenfmtError> {
    if mmtr_spin_argument == 1 {
        available_spin_channels
            .checked_sub(1)
            .ok_or(GenfmtError::InvalidAngularLimit {
                name: "available_spin_channels",
                value: available_spin_channels,
            })
    } else {
        Ok(0)
    }
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

fn validate_jas_spin_transition_input(
    input: JasSpinTransitionInput<'_>,
) -> Result<(usize, usize, Vec<i32>), GenfmtError> {
    validate_finite_scalar("first_eta", input.first_eta)?;
    validate_finite_scalar("last_eta", input.last_eta)?;
    let initial_orbital_j2 = initial_kappa_j2(input.initial_kappa)?;
    let mj_count = compact_doubled_j_count("jinit", input.initial_j2)?;
    compact_doubled_j_count("jmax", input.final_j2_max)?;
    if input.spin_channels == 0 || input.spin_channels > 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "spin_channels",
            value: input.spin_channels,
        });
    }
    if input.initial_j2 % 2 != initial_orbital_j2 % 2 {
        return Err(GenfmtError::InvalidDoubledAngularMomentum {
            field: "jinit",
            value: input.initial_j2,
        });
    }

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let transition_count = transition_l.len();
    let mut generated_final_j2 = jas_generated_final_j2(
        input.initial_kappa,
        input.final_lj_max,
        input.final_j2_max,
        input.max_angular_momentum,
    )?;
    if generated_final_j2.len() < transition_count {
        return Err(GenfmtError::InsufficientGeneratedTransitions {
            required: transition_count,
            generated: generated_final_j2.len(),
        });
    }
    generated_final_j2.truncate(transition_count);

    let magnetic_dim = input
        .rotation_magnetic_offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "rotation_magnetic_offset",
            value: input.rotation_magnetic_offset,
        })?;

    if let Some((_, max_l)) = active_transition_limits(&transition_l) {
        if max_l > input.max_angular_momentum {
            let length = checked_count("max_angular_momentum", input.max_angular_momentum)?;
            return Err(GenfmtError::TableAxisTooShort {
                table: "generated_transitions",
                axis: "lind",
                length,
                required: max_l + 1,
            });
        }
        let angular_count = checked_count("lind", max_l)?;
        validate_rotation_table(
            "first_rotation",
            input.first_rotation,
            input.rotation_magnetic_offset,
            angular_count,
        )?;
        validate_rotation_table(
            "last_rotation",
            input.last_rotation,
            input.rotation_magnetic_offset,
            angular_count,
        )?;

        let required_j2 = transition_l
            .iter()
            .zip(&generated_final_j2)
            .filter_map(|(&maybe_l, &j2)| maybe_l.map(|_| j2))
            .max()
            .unwrap_or(0);
        if required_j2 > input.initial_j2 {
            let required = compact_doubled_j_count("jind", required_j2)?;
            return Err(GenfmtError::TableAxisTooShort {
                table: "hbmatrs",
                axis: "mj",
                length: mj_count,
                required,
            });
        }
    }

    Ok((mj_count, magnetic_dim, generated_final_j2))
}

fn validate_jas_one_sided_transition_input(
    input: JasOneSidedTransitionInput<'_>,
) -> Result<(usize, usize, Vec<i32>), GenfmtError> {
    validate_finite_scalar("first_eta", input.first_eta)?;
    validate_finite_scalar("last_eta", input.last_eta)?;
    let initial_orbital_j2 = initial_kappa_j2(input.initial_kappa)?;
    let mj_count = compact_doubled_j_count("jinit", input.initial_j2)?;
    compact_doubled_j_count("jmax", input.final_j2_max)?;
    if input.spin_channels == 0 || input.spin_channels > 2 {
        return Err(GenfmtError::InvalidAngularLimit {
            name: "spin_channels",
            value: input.spin_channels,
        });
    }
    if input.initial_j2 % 2 != initial_orbital_j2 % 2 {
        return Err(GenfmtError::InvalidDoubledAngularMomentum {
            field: "jinit",
            value: input.initial_j2,
        });
    }
    let q_count = input.q_phases.len();
    validate_positive_limit("q_count", q_count)?;
    ensure_axis_len("q_beta_angles", "q", input.q_beta_angles.len(), q_count)?;

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    let transition_count = transition_l.len();
    ensure_axis_len(
        "final_lg_momenta",
        "transition",
        input.final_lg_momenta.len(),
        transition_count,
    )?;
    ensure_axis_len(
        "final_lj_momenta",
        "transition",
        input.final_lj_momenta.len(),
        transition_count,
    )?;

    let mut generated_final_j2 = jas_generated_final_j2(
        input.initial_kappa,
        input.final_lj_max,
        input.final_j2_max,
        input.max_angular_momentum,
    )?;
    if generated_final_j2.len() < transition_count {
        return Err(GenfmtError::InsufficientGeneratedTransitions {
            required: transition_count,
            generated: generated_final_j2.len(),
        });
    }
    generated_final_j2.truncate(transition_count);

    let magnetic_dim = input
        .rotation_magnetic_offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "rotation_magnetic_offset",
            value: input.rotation_magnetic_offset,
        })?;

    if let Some((_, max_l)) = active_transition_limits(&transition_l) {
        let angular_count = checked_count("lind", max_l)?;
        validate_rotation_table(
            "first_rotation",
            input.first_rotation,
            input.rotation_magnetic_offset,
            angular_count,
        )?;
        validate_rotation_table(
            "last_rotation",
            input.last_rotation,
            input.rotation_magnetic_offset,
            angular_count,
        )?;
        if max_l > 0 {
            for q in 0..q_count {
                let q_phase = complex_vector_entry(input.q_phases, "q_phases", q)?;
                if q_phase == Complex::new(0.0, 0.0) {
                    return Err(GenfmtError::ZeroComplex { field: "q_phases" });
                }
            }
        }
    }

    for transition in 0..transition_count {
        indexed_nonnegative_i32(
            "final_lg_momenta",
            transition,
            input.final_lg_momenta[transition],
        )?;
        indexed_nonnegative_i32(
            "final_lj_momenta",
            transition,
            input.final_lj_momenta[transition],
        )?;
    }
    for q in 0..q_count {
        complex_vector_entry(input.q_phases, "q_phases", q)?;
        real_vector_entry(input.q_beta_angles, "q_beta_angles", q)?;
    }

    Ok((mj_count, magnetic_dim, generated_final_j2))
}

fn validate_jas_scattering_amplitude_input(
    input: JasScatteringAmplitudeInput<'_>,
) -> Result<usize, GenfmtError> {
    validate_finite_scalar("eta", input.eta)?;
    let lambda_len = input.m_indices.len().min(input.n_indices.len());
    validate_lambda_count("lambda_count", input.lambda_count, lambda_len)?;
    let mj_count = compact_doubled_j_count("jinit", input.initial_j2)?;
    let q_count = input.q_weights.len();
    validate_positive_limit("q_count", q_count)?;
    let transition_count = input.transition_angular_momenta.len();

    ensure_axis_len(
        "radial_factors",
        "q",
        input.radial_factors.shape()[0],
        q_count,
    )?;
    ensure_axis_len(
        "radial_factors",
        "transition",
        input.radial_factors.shape()[1],
        transition_count,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "mj",
        input.transition_matrix.shape()[0],
        mj_count,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "spin",
        input.transition_matrix.shape()[1],
        2,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "transition",
        input.transition_matrix.shape()[4],
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
        "mu2",
        input.transition_matrix.shape()[2],
        magnetic_required,
    )?;
    ensure_axis_len(
        "transition_matrix",
        "mu1",
        input.transition_matrix.shape()[3],
        magnetic_required,
    )?;

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    if let Some((min_l, raw_max_l)) = active_transition_limits(&transition_l) {
        let max_l = raw_max_l.min(input.max_angular_momentum);
        if min_l <= max_l {
            let angular_count = checked_count("lind", max_l)?;
            let magnetic_count = checked_count(
                "mlam",
                lambda_m_limit(input.m_indices, input.lambda_count)?.min(max_l),
            )?;
            ensure_axis_len("xnlm", "m", input.xnlm.shape()[0], magnetic_count)?;
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
    }

    if let Some(l_max) = input.decomposition_l_max {
        checked_count("decomposition_l_max", l_max)?;
    }
    Ok(mj_count)
}

fn validate_jas_left_right_amplitude_input(
    input: JasLeftRightAmplitudeInput<'_>,
) -> Result<usize, GenfmtError> {
    validate_finite_scalar("eta", input.eta)?;
    let lambda_len = input.m_indices.len().min(input.n_indices.len());
    validate_lambda_count("lambda_count", input.lambda_count, lambda_len)?;
    let mj_count = compact_doubled_j_count("jinit", input.initial_j2)?;
    let q_count = input.q_weights.len();
    validate_positive_limit("q_count", q_count)?;
    let transition_count = input.transition_angular_momenta.len();

    ensure_axis_len(
        "radial_factors",
        "q",
        input.radial_factors.shape()[0],
        q_count,
    )?;
    ensure_axis_len(
        "radial_factors",
        "transition",
        input.radial_factors.shape()[1],
        transition_count,
    )?;
    validate_jas_one_sided_transition_matrix(
        "left_transition_matrix",
        input.left_transition_matrix,
        input.transition_magnetic_offset,
        mj_count,
        q_count,
        transition_count,
    )?;
    validate_jas_one_sided_transition_matrix(
        "right_transition_matrix",
        input.right_transition_matrix,
        input.transition_magnetic_offset,
        mj_count,
        q_count,
        transition_count,
    )?;

    let transition_l = transition_angular_momenta(input.transition_angular_momenta)?;
    if let Some((min_l, raw_max_l)) = active_transition_limits(&transition_l) {
        let max_l = raw_max_l.min(input.max_angular_momentum);
        if min_l <= max_l {
            let angular_count = checked_count("lind", max_l)?;
            let magnetic_count = checked_count(
                "mlam",
                lambda_m_limit(input.m_indices, input.lambda_count)?.min(max_l),
            )?;
            ensure_axis_len("xnlm", "m", input.xnlm.shape()[0], magnetic_count)?;
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
    }

    if let Some(l_max) = input.decomposition_l_max {
        checked_count("decomposition_l_max", l_max)?;
    }
    Ok(mj_count)
}

fn validate_jas_one_sided_transition_matrix(
    name: &'static str,
    matrix: ArrayView4<'_, Complex>,
    magnetic_offset: usize,
    mj_count: usize,
    q_count: usize,
    transition_count: usize,
) -> Result<(), GenfmtError> {
    ensure_axis_len(name, "mj", matrix.shape()[0], mj_count)?;
    let magnetic_required = magnetic_offset
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "transition_magnetic_offset",
            value: magnetic_offset,
        })?;
    ensure_axis_len(name, "mu", matrix.shape()[1], magnetic_required)?;
    ensure_axis_len(name, "q", matrix.shape()[2], q_count)?;
    ensure_axis_len(name, "transition", matrix.shape()[3], transition_count)?;
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

fn lambda_m_limit(m_indices: ArrayView1<'_, i32>, count: usize) -> Result<usize, GenfmtError> {
    let mut max_m = 0;
    for index in 0..count {
        max_m = max_m.max(lambda_abs_magnetic(m_indices[index], index)?);
    }
    Ok(max_m)
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

fn compact_doubled_j_count(field: &'static str, value: i32) -> Result<usize, GenfmtError> {
    if value < 0 {
        return Err(GenfmtError::InvalidDoubledAngularMomentum { field, value });
    }
    usize::try_from(value)
        .map_err(|_| GenfmtError::InvalidDoubledAngularMomentum { field, value })?
        .checked_add(1)
        .ok_or(GenfmtError::InvalidDoubledAngularMomentum { field, value })
}

fn compact_doubled_j_row(initial_j2: i32, mj: i32) -> Option<usize> {
    let shifted = mj.checked_add(initial_j2)?;
    if shifted < 0 || shifted % 2 != 0 {
        return None;
    }
    let row = shifted / 2;
    if row <= initial_j2 {
        usize::try_from(row).ok()
    } else {
        None
    }
}

fn initial_kappa_j2(kappa: i32) -> Result<i32, GenfmtError> {
    let abs_kappa = kappa
        .checked_abs()
        .filter(|&value| value > 0)
        .ok_or(GenfmtError::InvalidInitialKappa { kappa })?;
    abs_kappa
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(GenfmtError::InvalidInitialKappa { kappa })
}

fn jas_generated_final_j2(
    initial_kappa: i32,
    final_lj_max: usize,
    final_j2_max: i32,
    max_angular_momentum: usize,
) -> Result<Vec<i32>, GenfmtError> {
    if final_j2_max < 0 {
        return Err(GenfmtError::InvalidDoubledAngularMomentum {
            field: "jmax",
            value: final_j2_max,
        });
    }
    let initial_j2 = i64::from(initial_kappa_j2(initial_kappa)?);
    let final_j2_max = i64::from(final_j2_max);
    let final_lj_max_i64 =
        i64::try_from(final_lj_max).map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "ljmax",
            value: final_lj_max,
        })?;
    let final_l_limit = i64::try_from(max_angular_momentum)
        .map_err(|_| GenfmtError::InvalidAngularLimit {
            name: "max_angular_momentum",
            value: max_angular_momentum,
        })?
        .checked_mul(2)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "max_angular_momentum",
            value: max_angular_momentum,
        })?;
    let initial_parity = if initial_kappa > 0 { -1 } else { 1 };
    let mut generated = Vec::new();

    for lj in 0..=final_lj_max_i64 {
        let transition_j2 = lj.checked_mul(2).ok_or(GenfmtError::InvalidAngularLimit {
            name: "ljmax",
            value: final_lj_max,
        })?;
        let final_max = (transition_j2 + initial_j2).min(final_j2_max);
        let final_min = (transition_j2 - initial_j2).abs().max(1);
        let mut final_j2 = final_min;
        while final_j2 <= final_max {
            let parity_test = initial_j2
                .checked_add(final_j2)
                .and_then(|value| value.checked_add(transition_j2))
                .ok_or(GenfmtError::InvalidDoubledAngularMomentum {
                    field: "jmax",
                    value: i32::MAX,
                })?;
            let final_parity = if parity_test % 4 == 0 {
                -initial_parity
            } else {
                initial_parity
            };
            let final_l2 = if final_parity > 0 {
                final_j2.checked_sub(1)
            } else {
                final_j2.checked_add(1)
            }
            .ok_or(GenfmtError::InvalidDoubledAngularMomentum {
                field: "jmax",
                value: i32::MAX,
            })?;
            if final_l2 <= final_l_limit {
                generated.push(i32::try_from(final_j2).map_err(|_| {
                    GenfmtError::InvalidDoubledAngularMomentum {
                        field: "jmax",
                        value: i32::MAX,
                    }
                })?);
            }
            final_j2 =
                final_j2
                    .checked_add(2)
                    .ok_or(GenfmtError::InvalidDoubledAngularMomentum {
                        field: "jmax",
                        value: i32::MAX,
                    })?;
        }
    }

    Ok(generated)
}

fn jas_spin_coupling(
    angular_momentum: i32,
    final_j2: i32,
    spin: i32,
    mj: i32,
) -> Result<Real, GenfmtError> {
    let spin_mj2 = 2 * spin - 1;
    let doubled_l = angular_momentum
        .checked_mul(2)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "lind",
            value: usize::try_from(angular_momentum).unwrap_or(usize::MAX),
        })?;
    let mut coefficient = wigner_3j(1, final_j2, doubled_l, spin_mj2, -mj, 2)?;
    let sign_index = (angular_momentum + mj - 1) / 2;
    if sign_index % 2 != 0 {
        coefficient = -coefficient;
    }
    Ok(coefficient)
}

fn jas_one_sided_transition_weights(
    input: JasOneSidedTransitionInput<'_>,
    generated_final_j2: &[i32],
) -> Result<Array3<Real>, GenfmtError> {
    let initial_orbital_j2 = initial_kappa_j2(input.initial_kappa)?;
    let mj_count = compact_doubled_j_count("jinit", input.initial_j2)?;
    let transition_count = input.transition_angular_momenta.len();
    let doubled_lmax = checked_i32("max_angular_momentum", input.max_angular_momentum)?
        .checked_mul(2)
        .ok_or(GenfmtError::InvalidAngularLimit {
            name: "max_angular_momentum",
            value: input.max_angular_momentum,
        })?;
    let mut weights = Array3::<Real>::zeros((mj_count, 2, transition_count).f());

    for transition in 0..transition_count {
        let final_j2 = generated_final_j2[transition];
        let lj = indexed_nonnegative_i32(
            "final_lj_momenta",
            transition,
            input.final_lj_momenta[transition],
        )?;
        let doubled_lj = lj.checked_mul(2).ok_or(GenfmtError::InvalidLambdaIndex {
            index: transition,
            field: "final_lj_momenta",
            value: input.final_lj_momenta[transition],
        })?;
        let doubled_lg = indexed_nonnegative_i32(
            "final_lg_momenta",
            transition,
            input.final_lg_momenta[transition],
        )?
        .checked_mul(2)
        .ok_or(GenfmtError::InvalidLambdaIndex {
            index: transition,
            field: "final_lg_momenta",
            value: input.final_lg_momenta[transition],
        })?;

        for mj_row in 0..mj_count {
            let mj = -input.initial_j2
                + checked_i32("jinit", mj_row)?.checked_mul(2).ok_or(
                    GenfmtError::InvalidDoubledAngularMomentum {
                        field: "jinit",
                        value: input.initial_j2,
                    },
                )?;
            let abs_mj = mj.abs();
            let mut simple_3j = if abs_mj <= final_j2 {
                wigner_3j(initial_orbital_j2, doubled_lj, final_j2, -mj, 0, 2)?
            } else {
                0.0
            };
            if (mj + 1).rem_euclid(4) != 0 {
                simple_3j = -simple_3j;
            }

            for spin in 0..input.spin_channels {
                let mut ls_to_j = 0.0;
                if abs_mj <= final_j2 && abs_mj - 1 <= doubled_lmax {
                    let spin_mj2 = 2 * spin as i32 - 1;
                    let magnetic_l2 = mj.checked_sub(spin_mj2).ok_or(
                        GenfmtError::InvalidDoubledAngularMomentum {
                            field: "jinit",
                            value: input.initial_j2,
                        },
                    )?;
                    ls_to_j = wigner_3j(doubled_lg, 1, final_j2, magnetic_l2, spin_mj2, 2)?;
                    if (doubled_lg - 1 + mj).rem_euclid(4) != 0 {
                        ls_to_j = -ls_to_j;
                    }
                    ls_to_j *= (final_j2 as Real + 1.0).sqrt();
                }
                weights[(mj_row, spin, transition)] = ls_to_j * simple_3j;
            }
        }
    }

    Ok(weights)
}

#[derive(Debug, Clone, Copy)]
struct JasQRotationContext {
    angular_momentum: usize,
    magnetic_limit: i32,
    output_magnetic: i32,
    internal_magnetic: i32,
    q_phase: Complex,
    q_beta: Real,
    endpoint_phase: Complex,
}

fn jas_q_rotated_left_entry(
    input: JasOneSidedTransitionInput<'_>,
    context: JasQRotationContext,
) -> Result<Complex, GenfmtError> {
    let angular_momentum_i32 = checked_i32("lind", context.angular_momentum)?;
    let mut total = Complex::new(0.0, 0.0);
    for m2 in -context.magnetic_limit..=context.magnetic_limit {
        let phase = complex_integer_power(context.endpoint_phase, m2, "first_eta")?
            * complex_integer_power(context.q_phase, m2, "q_phases")?;
        let q_rotation = wigner_rotation(
            context.q_beta,
            angular_momentum_i32,
            m2,
            context.internal_magnetic,
            1,
        )?;
        let path_rotation = rotation_entry(
            input.first_rotation,
            input.rotation_magnetic_offset,
            context.angular_momentum,
            context.output_magnetic,
            m2,
        )?;
        total += phase * path_rotation * q_rotation;
    }
    Ok(total)
}

fn jas_q_rotated_right_entry(
    input: JasOneSidedTransitionInput<'_>,
    context: JasQRotationContext,
) -> Result<Complex, GenfmtError> {
    let angular_momentum_i32 = checked_i32("lind", context.angular_momentum)?;
    let mut total = Complex::new(0.0, 0.0);
    for m2 in -context.magnetic_limit..=context.magnetic_limit {
        let phase = complex_integer_power(context.endpoint_phase, m2, "last_eta")?;
        let q_rotation = wigner_rotation(
            context.q_beta,
            angular_momentum_i32,
            m2,
            context.internal_magnetic,
            1,
        )?;
        let q_rotated =
            (q_rotation * complex_integer_power(context.q_phase, m2, "q_phases")?).conj();
        let path_rotation = rotation_entry(
            input.last_rotation,
            input.rotation_magnetic_offset,
            context.angular_momentum,
            m2,
            context.output_magnetic,
        )?;
        total += phase * path_rotation * q_rotated;
    }
    Ok(total)
}

fn imaginary_unit_power(exponent: i32) -> Complex {
    match exponent.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

fn complex_integer_power(
    value: Complex,
    exponent: i32,
    field: &'static str,
) -> Result<Complex, GenfmtError> {
    if exponent == 0 {
        return Ok(Complex::new(1.0, 0.0));
    }
    if value == Complex::new(0.0, 0.0) && exponent < 0 {
        return Err(GenfmtError::ZeroComplex { field });
    }

    let mut power = Complex::new(1.0, 0.0);
    for _ in 0..exponent.unsigned_abs() {
        power *= value;
    }
    let result = if exponent < 0 {
        Complex::new(1.0, 0.0) / power
    } else {
        power
    };
    if result.re.is_finite() && result.im.is_finite() {
        Ok(result)
    } else {
        Err(GenfmtError::NonFiniteComplex {
            field,
            real: result.re,
            imaginary: result.im,
        })
    }
}

fn indexed_nonnegative_i32(
    field: &'static str,
    index: usize,
    value: i32,
) -> Result<i32, GenfmtError> {
    if value >= 0 {
        Ok(value)
    } else {
        Err(GenfmtError::InvalidLambdaIndex {
            index,
            field,
            value,
        })
    }
}

fn jas_q_sums(
    radial_factors: ArrayView2<'_, Complex>,
    q_weights: ArrayView1<'_, Complex>,
) -> Result<Vec<Complex>, GenfmtError> {
    let q_count = q_weights.len();
    let transition_count = radial_factors.shape()[1];
    let mut sums = vec![Complex::new(0.0, 0.0); transition_count];
    for (transition, sum) in sums.iter_mut().enumerate() {
        for q in 0..q_count {
            let radial = complex_entry(radial_factors, "radial_factors", q, transition)?;
            let weight = complex_vector_entry(q_weights, "q_weights", q)?;
            *sum += radial * radial * weight;
        }
    }
    Ok(sums)
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

fn jas_transition_matrix_entry(
    transition_matrix: ArrayView5<'_, Complex>,
    offset: usize,
    mj_row: usize,
    spin: usize,
    second_magnetic: i32,
    first_magnetic: i32,
    transition: usize,
) -> Result<Complex, GenfmtError> {
    let second = signed_magnetic_index(
        second_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_matrix",
        "mu2",
        transition_matrix.shape()[2],
    )?;
    let first = signed_magnetic_index(
        first_magnetic,
        offset,
        "transition_magnetic_offset",
        "transition_matrix",
        "mu1",
        transition_matrix.shape()[3],
    )?;
    complex5_entry(
        transition_matrix,
        "transition_matrix",
        [mj_row, spin, second, first, transition],
    )
}

fn jas_one_sided_transition_matrix_entry(
    transition_matrix: ArrayView4<'_, Complex>,
    name: &'static str,
    offset: usize,
    mj_row: usize,
    magnetic: i32,
    q: usize,
    transition: usize,
) -> Result<Complex, GenfmtError> {
    let mu = signed_magnetic_index(
        magnetic,
        offset,
        "transition_magnetic_offset",
        name,
        "mu",
        transition_matrix.shape()[1],
    )?;
    complex4_entry(transition_matrix, name, mj_row, mu, q, transition)
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

fn real_vector_entry(
    vector: ArrayView1<'_, Real>,
    table: &'static str,
    index: usize,
) -> Result<Real, GenfmtError> {
    ensure_axis_len(table, "index", vector.len(), index + 1)?;
    let value = vector[index];
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTableScalar {
            table,
            row: index,
            column: 0,
            value,
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

fn complex5_entry(
    table: ArrayView5<'_, Complex>,
    name: &'static str,
    index: [usize; 5],
) -> Result<Complex, GenfmtError> {
    let [i0, i1, i2, i3, i4] = index;
    ensure_axis_len(name, "axis0", table.shape()[0], i0 + 1)?;
    ensure_axis_len(name, "axis1", table.shape()[1], i1 + 1)?;
    ensure_axis_len(name, "axis2", table.shape()[2], i2 + 1)?;
    ensure_axis_len(name, "axis3", table.shape()[3], i3 + 1)?;
    ensure_axis_len(name, "axis4", table.shape()[4], i4 + 1)?;
    let value = table[(i0, i1, i2, i3, i4)];
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(GenfmtError::NonFiniteTensor5Complex {
            table: name,
            i0,
            i1,
            i2,
            i3,
            i4,
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
