use crate::{Real, angular::wigner_3j};

use super::{AtomMathError, AtomicBreitAngularCoefficients, doubled_j_from_kappa};

/// Port of FEFF `ATOM/bkmrdf.f90`, the Breit angular coefficients.
///
/// `left_kappa` and `right_kappa` are the relativistic kappa values for the
/// two orbitals. `rank` is FEFF's integer `k` for the Breit radial integral.
pub fn atomic_breit_angular_coefficients(
    left_kappa: i32,
    right_kappa: i32,
    rank: usize,
) -> Result<AtomicBreitAngularCoefficients, AtomMathError> {
    let left_j2 = doubled_j_from_kappa(left_kappa)?;
    let right_j2 = doubled_j_from_kappa(right_kappa)?;
    let rank_i32 = i32::try_from(rank).map_err(|_| AtomMathError::BreitRankOutOfRange { rank })?;
    let kappa_difference =
        right_kappa
            .checked_sub(left_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa,
                right_kappa,
            })?;
    let kappa_sum =
        right_kappa
            .checked_add(left_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa,
                right_kappa,
            })?;

    let mut coefficients = AtomicBreitAngularCoefficients {
        magnetic: [0.0; 3],
        retarded: [0.0; 3],
    };
    let mut angular_l = rank_i32
        .checked_sub(1)
        .ok_or(AtomMathError::BreitRankOutOfRange { rank })?;
    for order in 0..3 {
        if angular_l >= 0 {
            accumulate_breit_order(
                BreitOrderContext {
                    left_j2,
                    right_j2,
                    kappa_difference,
                    kappa_sum,
                    rank: rank_i32,
                    rank_usize: rank,
                    angular_l,
                    order,
                },
                &mut coefficients,
            )?;
        }
        angular_l = angular_l
            .checked_add(1)
            .ok_or(AtomMathError::BreitRankOutOfRange { rank })?;
    }

    Ok(coefficients)
}

#[derive(Debug, Clone, Copy)]
struct BreitOrderContext {
    left_j2: i32,
    right_j2: i32,
    kappa_difference: i32,
    kappa_sum: i32,
    rank: i32,
    rank_usize: usize,
    angular_l: i32,
    order: usize,
}

#[derive(Debug, Clone, Copy)]
struct BreitOrderTerms {
    cm: Real,
    cz: Real,
    cp: Real,
    d: Real,
    retardation: Option<BreitRetardationTerms>,
}

#[derive(Debug, Clone, Copy)]
struct BreitRetardationTerms {
    am: Real,
    az: Real,
    ap: Real,
    scale: Real,
}

fn accumulate_breit_order(
    context: BreitOrderContext,
    coefficients: &mut AtomicBreitAngularCoefficients,
) -> Result<(), AtomMathError> {
    let wigner_j3 = context
        .angular_l
        .checked_mul(2)
        .ok_or(AtomMathError::BreitRankOutOfRange {
            rank: context.rank_usize,
        })?;
    let wigner = wigner_3j(context.left_j2, context.right_j2, wigner_j3, -1, 1, 2)?;
    if wigner == 0.0 {
        return Ok(());
    }

    let angular_denominator = context
        .angular_l
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(AtomMathError::BreitRankOutOfRange {
            rank: context.rank_usize,
        })?;
    let squared_wigner = wigner * wigner;
    let terms = breit_order_terms(context, angular_denominator);

    if let Some(retardation) = terms.retardation {
        let mut retardation_scale = retardation.scale;
        let denominator = Real::from(angular_denominator.abs()) * terms.d;
        if denominator != 0.0 {
            retardation_scale /= denominator;
        }
        coefficients.retarded[0] +=
            squared_wigner * (retardation.am - retardation_scale * terms.cm);
        coefficients.retarded[1] +=
            (squared_wigner + squared_wigner) * (retardation.az - retardation_scale * terms.cz);
        coefficients.retarded[2] +=
            squared_wigner * (retardation.ap - retardation_scale * terms.cp);
    }

    if terms.d != 0.0 {
        let magnetic_scale = squared_wigner / terms.d;
        coefficients.magnetic[0] += terms.cm * magnetic_scale;
        coefficients.magnetic[1] += terms.cz * (magnetic_scale + magnetic_scale);
        coefficients.magnetic[2] += terms.cp * magnetic_scale;
    }

    Ok(())
}

fn breit_order_terms(context: BreitOrderContext, angular_denominator: i32) -> BreitOrderTerms {
    match context.order {
        0 => {
            let cm = square(context.kappa_difference + context.rank);
            let cz = square(context.kappa_difference) - square(context.rank);
            let cp = square(context.rank - context.kappa_difference);
            let scale = Real::from(context.rank);
            let retardation = breit_retardation_shape(
                context.kappa_difference,
                context.angular_l,
                angular_denominator,
                scale,
            );
            BreitOrderTerms {
                cm,
                cz,
                cp,
                d: scale * Real::from(context.rank + context.rank + 1),
                retardation: Some(retardation),
            }
        }
        1 => {
            let cm = square(context.kappa_sum);
            BreitOrderTerms {
                cm,
                cz: cm,
                cp: cm,
                d: Real::from(context.rank) * Real::from(context.rank + 1),
                retardation: None,
            }
        }
        _ => {
            let cm = square(context.kappa_difference - context.angular_l);
            let cz = square(context.kappa_difference) - square(context.angular_l);
            let cp = square(context.kappa_difference + context.angular_l);
            let scale = Real::from(context.angular_l);
            let retardation = breit_retardation_shape(
                context.kappa_difference,
                context.angular_l,
                -angular_denominator,
                scale,
            );
            BreitOrderTerms {
                cm,
                cz,
                cp,
                d: scale * Real::from(context.rank + context.rank + 1),
                retardation: Some(retardation),
            }
        }
    }
}

fn breit_retardation_shape(
    kappa_difference: i32,
    angular_l: i32,
    denominator: i32,
    scale: Real,
) -> BreitRetardationTerms {
    let next_l = angular_l + 1;
    let denominator = Real::from(denominator);
    BreitRetardationTerms {
        am: Real::from((kappa_difference - angular_l) * (kappa_difference + next_l)) / denominator,
        az: Real::from(kappa_difference * kappa_difference + angular_l * next_l) / denominator,
        ap: Real::from((angular_l + kappa_difference) * (kappa_difference - next_l)) / denominator,
        scale,
    }
}

fn square(value: i32) -> Real {
    let value = Real::from(value);
    value * value
}
