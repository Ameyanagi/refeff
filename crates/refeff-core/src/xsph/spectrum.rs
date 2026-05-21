//! FEFF XSPH NRIXS transition weights and spectrum updates.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ShapeBuilder};

use crate::{Complex, Real, legendre_polynomials_into, wigner_3j};

use super::{
    XsphError, XsphLgSpectrumUpdateInput, XsphLjSpectrumUpdateInput, XsphSpectrumUpdateMode,
    doubled_j_from_kappa, usize_to_i32, validate_active_len, validate_cwig3j_doubled_argument,
    validate_cwig3j_integer_argument, validate_finite_complex, validate_finite_real,
    validate_indexed_angular_momentum,
};

/// Port of FEFF `XSPH/bcoefjas.f90`.
///
/// Builds the two spin-component NRIXS transition weights `hbmat(0:1, 1:indmax)`
/// for a single doubled initial magnetic quantum number. The returned array is
/// Fortran-order with shape `(2, active_len)`, matching FEFF's spin-first
/// storage.
#[allow(clippy::too_many_arguments)]
pub fn xsph_nrixs_transition_weights(
    initial_kappa: i32,
    initial_mj2: i32,
    lmax: usize,
    jmax: i32,
    ljmax: i32,
    lgind: ArrayView1<'_, i32>,
    ljind: ArrayView1<'_, i32>,
    active_len: usize,
) -> Result<Array2<Real>, XsphError> {
    if initial_kappa == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_active_len("lgind", lgind.len(), active_len)?;
    validate_active_len("ljind", ljind.len(), active_len)?;
    if jmax < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "jmax",
            index: 0,
            value: jmax,
        });
    }
    validate_cwig3j_doubled_argument("jmax", jmax, jmax)?;

    let lmax_i32 = usize_to_i32("lmax", lmax)?;
    let doubled_lmax = lmax_i32.checked_mul(2).ok_or(XsphError::SizeOutOfRange {
        name: "lmax",
        value: lmax,
    })?;
    let abs_ljmax = ljmax.checked_abs().ok_or(XsphError::IntegerOutOfRange {
        name: "ljmax",
        value: ljmax,
    })?;
    validate_cwig3j_integer_argument("ljmax", abs_ljmax)?;
    let jinit = doubled_j_from_kappa("initial_kappa", initial_kappa)?;
    validate_cwig3j_doubled_argument("initial_kappa", initial_kappa, jinit)?;
    let abs_initial_mj2 = initial_mj2
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_mj2",
            value: initial_mj2,
        })?;
    let initial_parity = if initial_kappa > 0 { -1 } else { 1 };

    let mut final_j2 = Vec::new();
    for lj in 0..=abs_ljmax {
        let lower = (2 * lj - jinit).abs().max(1);
        let upper = (2 * lj + jinit).min(jmax);
        let mut jfin = lower;
        while jfin <= upper {
            let final_parity = if (jinit + jfin + 2 * lj).rem_euclid(4) == 0 {
                -initial_parity
            } else {
                initial_parity
            };
            let final_l2 = if final_parity > 0 { jfin - 1 } else { jfin + 1 };
            if final_l2 <= doubled_lmax {
                final_j2.push(jfin);
            }
            jfin += 2;
        }
    }
    if final_j2.len() < active_len {
        return Err(XsphError::InsufficientGeneratedStates {
            required: active_len,
            generated: final_j2.len(),
        });
    }

    let mut weights = Array2::<Real>::zeros((2, active_len).f());
    for index in 0..active_len {
        let jfin = final_j2[index];
        let lj = validate_indexed_angular_momentum("ljind", index, ljind[index])?;
        let lg = validate_indexed_angular_momentum("lgind", index, lgind[index])?
            .checked_mul(2)
            .ok_or(XsphError::IntegerOutOfRange {
                name: "lgind",
                value: lgind[index],
            })?;
        validate_cwig3j_doubled_argument("jfin", jfin, jfin)?;
        validate_cwig3j_integer_argument("ljind", lj)?;
        validate_cwig3j_doubled_argument("lgind", lgind[index], lg)?;

        let mut simple_3j = if abs_initial_mj2 <= jfin {
            wigner_3j(jinit, 2 * lj, jfin, -initial_mj2, 0, 2)?
        } else {
            0.0
        };
        if (i64::from(initial_mj2) + 1).rem_euclid(4) != 0 {
            simple_3j = -simple_3j;
        }

        for spin_index in 0..=1 {
            let mut ls_to_j = 0.0;
            if abs_initial_mj2 <= jfin && abs_initial_mj2 - 1 <= doubled_lmax {
                let spin_mj2 = 2 * usize_to_i32("spin_index", spin_index)? - 1;
                let magnetic_l2 =
                    initial_mj2
                        .checked_sub(spin_mj2)
                        .ok_or(XsphError::IntegerOutOfRange {
                            name: "initial_mj2",
                            value: initial_mj2,
                        })?;
                ls_to_j = wigner_3j(lg, 1, jfin, magnetic_l2, spin_mj2, 2)?;
                if (i64::from(lg) - 1 + i64::from(initial_mj2)).rem_euclid(4) != 0 {
                    ls_to_j = -ls_to_j;
                }
                ls_to_j *= (f64::from(jfin) + 1.0).sqrt();
            }
            weights[(spin_index, index)] = ls_to_j * simple_3j;
        }
    }

    Ok(weights)
}

/// Port of FEFF `XSPH/specupdlg.f90`.
///
/// Updates the NRIXS angular-decomposition spectrum buckets `xseclg(0:ljmax)`
/// in place for one shared calculation and spin component. The transition
/// weights use compact magnetic columns: `mjinit = -jinit, -jinit+2, ..., jinit`
/// maps to `(mjinit + jinit) / 2`.
pub fn xsph_update_nrixs_lg_spectrum(
    input: XsphLgSpectrumUpdateInput<'_>,
    mut spectrum: ArrayViewMut1<'_, Complex>,
) -> Result<(), XsphError> {
    if input.calculation_index <= 0 {
        return Err(XsphError::NonPositiveCalculationIndex {
            calculation_index: input.calculation_index,
        });
    }
    if input.spin_index > 1 {
        return Err(XsphError::InvalidSpinIndex {
            spin_index: input.spin_index,
        });
    }
    if input.initial_j2 < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "initial_j2",
            index: 0,
            value: input.initial_j2,
        });
    }
    validate_cwig3j_doubled_argument("initial_j2", input.initial_j2, input.initial_j2)?;
    validate_active_len("index_map", input.index_map.len(), input.active_len)?;
    validate_active_len("orbital_l", input.orbital_l.len(), input.active_len)?;
    validate_active_len("final_lj", input.final_lj.len(), input.active_len)?;
    let channel_count = input
        .ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: input.ljmax })?;
    validate_active_len(
        "radial_integrals",
        input.radial_integrals.len(),
        channel_count,
    )?;
    validate_active_len("spectrum", spectrum.len(), channel_count)?;

    let magnetic_count = usize::try_from(input.initial_j2)
        .map_err(|_| XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?
        .checked_add(1)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?;
    let required_weights = [2, input.active_len, magnetic_count];
    let weight_shape = input.transition_weights.shape();
    let actual_weights = [weight_shape[0], weight_shape[1], weight_shape[2]];
    if actual_weights
        .iter()
        .zip(required_weights.iter())
        .any(|(actual, required)| actual < required)
    {
        return Err(XsphError::ShapeTooSmall {
            name: "transition_weights",
            required: required_weights,
            actual: actual_weights,
        });
    }

    let q_count = input.q_weights.len();
    validate_q_inputs(input.q_weights, input.q_cosines, q_count)?;
    let q_weights = xsph_effective_q_weights(input.q_weights, input.mix_dff)?;
    let q_pairs = xsph_q_pairs(input.mix_dff, input.mdff_mode, q_count)?;
    let legendre_count = channel_count;
    let mut legendre_by_pair = vec![0.0; q_count * q_count * legendre_count];
    for iq in 0..q_count {
        for iqq in 0..q_count {
            let cosine = input.q_cosines[(iq, iqq)];
            validate_finite_real("q_cosines", cosine)?;
            let offset = (iq * q_count + iqq) * legendre_count;
            legendre_polynomials_into(
                cosine,
                &mut legendre_by_pair[offset..offset + legendre_count],
            );
        }
    }

    for index in 0..input.active_len {
        let mapped = input.index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: input.index_map[index],
            })?;
        if mapped != input.calculation_index {
            continue;
        }

        let final_lj = validate_indexed_angular_momentum("final_lj", index, input.final_lj[index])?;
        let final_lj = usize::try_from(final_lj).map_err(|_| XsphError::IntegerOutOfRange {
            name: "final_lj",
            value: input.final_lj[index],
        })?;
        if final_lj > input.ljmax {
            return Err(XsphError::AngularMomentumOutOfRange {
                angular_momentum: final_lj,
                ljmax: input.ljmax,
            });
        }
        let orbital_l =
            validate_indexed_angular_momentum("orbital_l", index, input.orbital_l[index])?;
        let orbital_l = usize::try_from(orbital_l).map_err(|_| XsphError::IntegerOutOfRange {
            name: "orbital_l",
            value: input.orbital_l[index],
        })?;
        if orbital_l > input.ljmax {
            continue;
        }

        let trace = xsph_transition_trace(
            input.transition_weights,
            input.spin_index,
            index,
            input.initial_j2,
        )?;
        for &(iq, iqq) in &q_pairs {
            let legendre = legendre_by_pair[(iq * q_count + iqq) * legendre_count + final_lj];
            let radial = input.radial_integrals[final_lj];
            let amplitude = match input.mode {
                XsphSpectrumUpdateMode::Regular => {
                    -Complex::new(0.0, 1.0) * radial * radial * legendre
                }
                XsphSpectrumUpdateMode::Irregular => radial * legendre,
            };
            spectrum[orbital_l] -= amplitude * trace * q_weights[iq] * q_weights[iqq];
        }
    }

    Ok(())
}

/// Port of FEFF `XSPH/specupd.f90`.
///
/// Updates the NRIXS spectrum buckets `xsec(0:ljmax)` in place for one shared
/// calculation and spin component, accumulating regular-branch normalization in
/// `spectrum_norm`. FEFF assigns the complex q-weight product into a real
/// accumulator; this port preserves that behavior by using the real part.
pub fn xsph_update_nrixs_lj_spectrum(
    input: XsphLjSpectrumUpdateInput<'_>,
    mut spectrum: ArrayViewMut1<'_, Complex>,
    spectrum_norm: &mut Real,
) -> Result<(), XsphError> {
    validate_finite_real("spectrum_norm", *spectrum_norm)?;
    let channel_count = validate_lj_spectrum_update_input(input)?;
    validate_active_len("spectrum", spectrum.len(), channel_count)?;
    let q_count = input.q_weights.len();
    let q_weights = xsph_effective_q_weights(input.q_weights, input.mix_dff)?;
    let q_pairs = xsph_q_pairs(input.mix_dff, input.mdff_mode, q_count)?;
    let legendre_by_pair = xsph_legendre_by_q_pair(input.q_cosines, q_count, channel_count)?;

    for index in 0..input.active_len {
        let mapped = input.index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: input.index_map[index],
            })?;
        if mapped != input.calculation_index {
            continue;
        }

        let final_lj = validate_lj_update_channel(input, index)?;
        let radial = input.radial_integrals[final_lj];
        validate_finite_complex("radial_integrals", final_lj, radial)?;
        let trace = xsph_transition_trace(
            input.transition_weights,
            input.spin_index,
            index,
            input.initial_j2,
        )?;

        for &(iq, iqq) in &q_pairs {
            let legendre = legendre_by_pair[(iq * q_count + iqq) * channel_count + final_lj];
            let amplitude = xsph_spectrum_amplitude(radial, legendre, input.mode);
            let q_product = q_weights[iq] * q_weights[iqq];
            spectrum[final_lj] -= amplitude * trace * q_product;
            if input.mode == XsphSpectrumUpdateMode::Regular {
                *spectrum_norm += xsph_regular_norm_increment(radial, final_lj, q_product);
            }
        }
    }

    Ok(())
}

/// Port of FEFF `XSPH/specupdatom.f90`.
///
/// Updates per-final-state NRIXS spectrum slots `xsec(1:kfinmax)` in place for
/// one shared calculation and spin component, accumulating the same
/// regular-branch normalization used by [`xsph_update_nrixs_lj_spectrum`].
pub fn xsph_update_nrixs_atom_spectrum(
    input: XsphLjSpectrumUpdateInput<'_>,
    mut spectrum: ArrayViewMut1<'_, Complex>,
    spectrum_norm: &mut Real,
) -> Result<(), XsphError> {
    validate_finite_real("spectrum_norm", *spectrum_norm)?;
    let channel_count = validate_lj_spectrum_update_input(input)?;
    validate_active_len("spectrum", spectrum.len(), input.active_len)?;
    let q_count = input.q_weights.len();
    let q_weights = xsph_effective_q_weights(input.q_weights, input.mix_dff)?;
    let q_pairs = xsph_q_pairs(input.mix_dff, input.mdff_mode, q_count)?;
    let legendre_by_pair = xsph_legendre_by_q_pair(input.q_cosines, q_count, channel_count)?;

    for index in 0..input.active_len {
        let mapped = input.index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: input.index_map[index],
            })?;
        if mapped != input.calculation_index {
            continue;
        }

        let final_lj = validate_lj_update_channel(input, index)?;
        let radial = input.radial_integrals[final_lj];
        validate_finite_complex("radial_integrals", final_lj, radial)?;
        let trace = xsph_transition_trace(
            input.transition_weights,
            input.spin_index,
            index,
            input.initial_j2,
        )?;

        for &(iq, iqq) in &q_pairs {
            let legendre = legendre_by_pair[(iq * q_count + iqq) * channel_count + final_lj];
            let amplitude = xsph_spectrum_amplitude(radial, legendre, input.mode);
            let q_product = q_weights[iq] * q_weights[iqq];
            spectrum[index] -= amplitude * trace * q_product;
            if input.mode == XsphSpectrumUpdateMode::Regular {
                *spectrum_norm += xsph_regular_norm_increment(radial, final_lj, q_product);
            }
        }
    }

    Ok(())
}

fn validate_q_inputs(
    q_weights: ArrayView1<'_, Complex>,
    q_cosines: ArrayView2<'_, Real>,
    q_count: usize,
) -> Result<(), XsphError> {
    let shape = q_cosines.shape();
    let actual = [shape[0], shape[1]];
    let required = [q_count, q_count];
    if actual[0] < required[0] || actual[1] < required[1] {
        return Err(XsphError::MatrixTooSmall {
            name: "q_cosines",
            required,
            actual,
        });
    }
    for (index, &weight) in q_weights.iter().enumerate() {
        validate_finite_complex("q_weights", index, weight)?;
    }
    Ok(())
}

fn validate_lj_spectrum_update_input(
    input: XsphLjSpectrumUpdateInput<'_>,
) -> Result<usize, XsphError> {
    if input.calculation_index <= 0 {
        return Err(XsphError::NonPositiveCalculationIndex {
            calculation_index: input.calculation_index,
        });
    }
    if input.spin_index > 1 {
        return Err(XsphError::InvalidSpinIndex {
            spin_index: input.spin_index,
        });
    }
    if input.initial_j2 < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "initial_j2",
            index: 0,
            value: input.initial_j2,
        });
    }
    validate_cwig3j_doubled_argument("initial_j2", input.initial_j2, input.initial_j2)?;
    validate_active_len("index_map", input.index_map.len(), input.active_len)?;
    validate_active_len("final_lj", input.final_lj.len(), input.active_len)?;

    let channel_count = input
        .ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: input.ljmax })?;
    validate_active_len(
        "radial_integrals",
        input.radial_integrals.len(),
        channel_count,
    )?;

    let magnetic_count = usize::try_from(input.initial_j2)
        .map_err(|_| XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?
        .checked_add(1)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "initial_j2",
            value: input.initial_j2,
        })?;
    let required_weights = [2, input.active_len, magnetic_count];
    let weight_shape = input.transition_weights.shape();
    let actual_weights = [weight_shape[0], weight_shape[1], weight_shape[2]];
    if actual_weights
        .iter()
        .zip(required_weights.iter())
        .any(|(actual, required)| actual < required)
    {
        return Err(XsphError::ShapeTooSmall {
            name: "transition_weights",
            required: required_weights,
            actual: actual_weights,
        });
    }

    validate_q_inputs(input.q_weights, input.q_cosines, input.q_weights.len())?;
    Ok(channel_count)
}

fn validate_lj_update_channel(
    input: XsphLjSpectrumUpdateInput<'_>,
    index: usize,
) -> Result<usize, XsphError> {
    let final_lj = validate_indexed_angular_momentum("final_lj", index, input.final_lj[index])?;
    let final_lj = usize::try_from(final_lj).map_err(|_| XsphError::IntegerOutOfRange {
        name: "final_lj",
        value: input.final_lj[index],
    })?;
    if final_lj > input.ljmax {
        return Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: final_lj,
            ljmax: input.ljmax,
        });
    }
    Ok(final_lj)
}

fn xsph_legendre_by_q_pair(
    q_cosines: ArrayView2<'_, Real>,
    q_count: usize,
    legendre_count: usize,
) -> Result<Vec<Real>, XsphError> {
    let mut legendre_by_pair = vec![0.0; q_count * q_count * legendre_count];
    for iq in 0..q_count {
        for iqq in 0..q_count {
            let cosine = q_cosines[(iq, iqq)];
            validate_finite_real("q_cosines", cosine)?;
            let offset = (iq * q_count + iqq) * legendre_count;
            legendre_polynomials_into(
                cosine,
                &mut legendre_by_pair[offset..offset + legendre_count],
            );
        }
    }
    Ok(legendre_by_pair)
}

fn xsph_spectrum_amplitude(
    radial: Complex,
    legendre: Real,
    mode: XsphSpectrumUpdateMode,
) -> Complex {
    match mode {
        XsphSpectrumUpdateMode::Regular => -Complex::new(0.0, 1.0) * radial * radial * legendre,
        XsphSpectrumUpdateMode::Irregular => radial * legendre,
    }
}

fn xsph_regular_norm_increment(radial: Complex, final_lj: usize, q_product: Complex) -> Real {
    let denominator = (2 * final_lj + 1) as Real;
    radial.norm_sqr() / denominator * q_product.re
}

fn xsph_effective_q_weights(
    q_weights: ArrayView1<'_, Complex>,
    mix_dff: bool,
) -> Result<Vec<Complex>, XsphError> {
    q_weights
        .iter()
        .enumerate()
        .map(|(index, &weight)| {
            let effective = if mix_dff { weight } else { weight.sqrt() };
            validate_finite_complex("effective_q_weight", index, effective)?;
            Ok(effective)
        })
        .collect()
}

fn xsph_q_pairs(
    mix_dff: bool,
    mdff_mode: i32,
    q_count: usize,
) -> Result<Vec<(usize, usize)>, XsphError> {
    if !mix_dff {
        return Ok((0..q_count).map(|index| (index, index)).collect());
    }
    match mdff_mode {
        1 => Ok((0..q_count)
            .flat_map(|iq| (0..q_count).map(move |iqq| (iq, iqq)))
            .collect()),
        2 if q_count >= 2 => Ok(vec![(0, 1)]),
        2 => Err(XsphError::MatrixTooSmall {
            name: "q_weights",
            required: [2, 1],
            actual: [q_count, 1],
        }),
        _ => Err(XsphError::InvalidMdffMode { mdff_mode }),
    }
}

fn xsph_transition_trace(
    transition_weights: ArrayView3<'_, Real>,
    spin_index: usize,
    state_index: usize,
    initial_j2: i32,
) -> Result<Real, XsphError> {
    let mut trace = 0.0;
    let mut magnetic_j2 = -initial_j2;
    while magnetic_j2 <= initial_j2 {
        let magnetic_index = usize::try_from((magnetic_j2 + initial_j2) / 2).map_err(|_| {
            XsphError::IntegerOutOfRange {
                name: "initial_j2",
                value: initial_j2,
            }
        })?;
        let value = transition_weights[(spin_index, state_index, magnetic_index)];
        validate_finite_real("transition_weights", value)?;
        trace += value * value;
        magnetic_j2 += 2;
    }
    Ok(trace)
}
