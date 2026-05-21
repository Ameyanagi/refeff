use ndarray::{Array1, ArrayView1};

use super::support::*;
use super::*;

/// Port of `SFCONV/senergies.f90` `findsing`.
pub fn sfconv_find_singularities(
    lower: Real,
    upper: Real,
    candidates: ArrayView1<'_, Real>,
) -> Result<RealVec, SfconvError> {
    validate_finite_scalar("singularity lower bound", lower)?;
    validate_finite_scalar("singularity upper bound", upper)?;
    let mut singularities = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, &candidate)| {
            if !candidate.is_finite() {
                return Some(Err(SfconvError::NonFiniteValue {
                    field: "singularity candidate",
                    row: index,
                    value: candidate,
                }));
            }
            let in_forward_interval = candidate > lower && candidate < upper;
            let in_reverse_interval = candidate < lower && candidate > upper;
            (in_forward_interval || in_reverse_interval).then_some(Ok(candidate))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if singularities.len() > SFCONV_GRATER_MAX_SINGULARITIES {
        return Err(SfconvError::TooManySingularities {
            count: singularities.len(),
            max: SFCONV_GRATER_MAX_SINGULARITIES,
        });
    }
    singularities.sort_by(|left, right| left.total_cmp(right));
    Ok(Array1::from_vec(singularities))
}

/// Port of `SFCONV/grater.f90`: adaptive real quadrature with split points.
///
/// `singularities` are FEFF `xsing`: ordered real split points inserted
/// between `lower` and `upper` before the adaptive stack starts. The returned
/// diagnostics mirror FEFF `error`, `numcal`, and `maxns`.
pub fn sfconv_grater_integrate(
    mut integrand: impl FnMut(Real) -> Result<Real, SfconvError>,
    lower: Real,
    upper: Real,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    singularities: &[Real],
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    validate_grater_input(
        lower,
        upper,
        absolute_tolerance,
        relative_tolerance,
        singularities,
    )?;

    let mut xleft = vec![0.0; SFCONV_GRATER_MAX_REGIONS];
    let mut fval = vec![[0.0; 3]; SFCONV_GRATER_MAX_REGIONS];
    let mut nstack = singularities.len() + 1;
    let mut max_regions = nstack;
    let mut estimated_error = 0.0;
    let mut value_total = 0.0;

    xleft[0] = lower;
    xleft[singularities.len() + 1] = upper;
    for (index, &singularity) in singularities.iter().enumerate() {
        xleft[index + 1] = singularity;
    }

    for region in 0..nstack {
        let delta = xleft[region + 1] - xleft[region];
        for point in 0..3 {
            fval[region][point] = eval_grater_integrand(
                &mut integrand,
                xleft[region] + delta * SFCONV_GRATER_DX[point],
                region * 3 + point,
            )?;
        }
    }
    let mut evaluations = nstack * 3;
    let total_interval = upper - lower;

    loop {
        if nstack + 3 >= SFCONV_GRATER_MAX_REGIONS {
            return Err(SfconvError::TooManyIntegrationRegions {
                max_regions: SFCONV_GRATER_MAX_REGIONS,
            });
        }

        let region = nstack - 1;
        let delta = xleft[region + 1] - xleft[region];
        xleft[region + 3] = xleft[region + 1];
        xleft[region + 1] = xleft[region] + delta * SFCONV_GRATER_DX[0] * 2.0;
        xleft[region + 2] = xleft[region + 3] - delta * SFCONV_GRATER_DX[0] * 2.0;
        fval[region + 2][1] = fval[region][2];
        fval[region + 1][1] = fval[region][1];
        fval[region][1] = fval[region][0];

        let mut weight_index = 0;
        let mut high_order = 0.0;
        let mut low_order = 0.0;
        for current_region in region..=region + 2 {
            let sub_delta = xleft[current_region + 1] - xleft[current_region];
            fval[current_region][0] = eval_grater_integrand(
                &mut integrand,
                xleft[current_region] + SFCONV_GRATER_DX[0] * sub_delta,
                evaluations,
            )?;
            evaluations += 1;
            fval[current_region][2] = eval_grater_integrand(
                &mut integrand,
                xleft[current_region] + SFCONV_GRATER_DX[2] * sub_delta,
                evaluations,
            )?;
            evaluations += 1;
            for point in 0..3 {
                high_order += SFCONV_GRATER_WT9[weight_index] * fval[current_region][point] * delta;
                low_order += fval[current_region][point] * SFCONV_GRATER_WT[point] * sub_delta;
                weight_index += 1;
            }
        }

        let difference = (high_order - low_order).abs();
        let fraction = delta / total_interval;
        let at_singularity = fraction <= 1.0e-8;
        if difference <= absolute_tolerance * fraction
            || difference <= relative_tolerance * high_order.abs()
            || (at_singularity && (fraction <= 1.0e-15 || difference <= absolute_tolerance * 0.1))
        {
            value_total += high_order;
            estimated_error += difference.abs();
            nstack -= 1;
            if nstack == 0 {
                return Ok(SfconvAdaptiveIntegral {
                    value: value_total,
                    estimated_error,
                    evaluations,
                    max_regions,
                });
            }
        } else {
            nstack += 2;
            max_regions = max_regions.max(nstack);
        }
    }
}
