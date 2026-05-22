use super::*;

/// Port of FEFF `cgratr`: adaptive complex quadrature with optional split points.
///
/// `singularities` are FEFF `xsing`: real split points inserted between
/// `lower` and `upper` before the adaptive stack starts. The integrand closure
/// receives complex abscissae and may return [`SelfEnergyError`] for invalid
/// internal states.
pub fn cgratr(
    integrand: impl Fn(Complex) -> Result<Complex, SelfEnergyError>,
    lower: Complex,
    upper: Complex,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    singularities: &[Real],
) -> Result<CgratrIntegral, SelfEnergyError> {
    validate_cgratr_input(
        lower,
        upper,
        absolute_tolerance,
        relative_tolerance,
        singularities,
    )?;

    let mut xleft = vec![Complex::new(0.0, 0.0); CGRATR_MAX_REGIONS];
    let mut fval = vec![[Complex::new(0.0, 0.0); 3]; CGRATR_MAX_REGIONS];
    let mut nstack = singularities.len() + 1;
    let mut max_regions = nstack;
    let mut estimated_error = 0.0;
    let mut value_total = Complex::new(0.0, 0.0);

    xleft[0] = lower;
    xleft[singularities.len() + 1] = upper;
    for (index, &singularity) in singularities.iter().enumerate() {
        xleft[index + 1] = Complex::new(singularity, 0.0);
    }

    for region in 0..nstack {
        let delta = xleft[region + 1] - xleft[region];
        for point in 0..3 {
            fval[region][point] = integrand(xleft[region] + delta * CGRATR_DX[point])?;
        }
    }
    let mut evaluations = nstack * 3;
    let total_interval = upper - lower;

    loop {
        if nstack + 3 >= CGRATR_MAX_REGIONS {
            return Err(SelfEnergyError::TooManyIntegrationRegions {
                max_regions: CGRATR_MAX_REGIONS,
            });
        }

        let region = nstack - 1;
        let delta = xleft[region + 1] - xleft[region];
        xleft[region + 3] = xleft[region + 1];
        xleft[region + 1] = xleft[region] + delta * CGRATR_DX[0] * 2.0;
        xleft[region + 2] = xleft[region + 3] - delta * CGRATR_DX[0] * 2.0;
        fval[region + 2][1] = fval[region][2];
        fval[region + 1][1] = fval[region][1];
        fval[region][1] = fval[region][0];

        let mut weight_index = 0;
        let mut high_order = Complex::new(0.0, 0.0);
        let mut low_order = Complex::new(0.0, 0.0);
        for current_region in region..=region + 2 {
            let sub_delta = xleft[current_region + 1] - xleft[current_region];
            fval[current_region][0] = integrand(xleft[current_region] + sub_delta * CGRATR_DX[0])?;
            fval[current_region][2] = integrand(xleft[current_region] + sub_delta * CGRATR_DX[2])?;
            evaluations += 2;
            for point in 0..3 {
                high_order += CGRATR_WT9[weight_index] * fval[current_region][point] * delta;
                low_order += fval[current_region][point] * CGRATR_WT[point] * sub_delta;
                weight_index += 1;
            }
        }

        let difference = (high_order - low_order).norm();
        let fraction = (delta / total_interval).re;
        let at_singularity = fraction <= 1.0e-8;
        if difference <= absolute_tolerance * fraction
            || difference <= relative_tolerance * high_order.norm()
            || (at_singularity && (fraction <= 1.0e-15 || difference <= absolute_tolerance * 0.1))
        {
            value_total += high_order;
            estimated_error += difference;
            nstack -= 1;
            if nstack == 0 {
                return Ok(CgratrIntegral {
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
fn validate_cgratr_input(
    lower: Complex,
    upper: Complex,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    singularities: &[Real],
) -> Result<(), SelfEnergyError> {
    ensure_finite_complex("cgratr lower", lower)?;
    ensure_finite_complex("cgratr upper", upper)?;
    if lower == upper || upper.re <= lower.re {
        return Err(SelfEnergyError::InvalidIntegrationInterval { lower, upper });
    }
    ensure_positive_tolerance("abr", absolute_tolerance)?;
    ensure_positive_tolerance("rlr", relative_tolerance)?;
    if singularities.len() > CGRATR_MAX_SINGULARITIES {
        return Err(SelfEnergyError::TooManySingularities {
            count: singularities.len(),
            max: CGRATR_MAX_SINGULARITIES,
        });
    }

    let min_bound = lower.re;
    let max_bound = upper.re;
    let mut previous = lower.re;
    for (index, &singularity) in singularities.iter().enumerate() {
        if !singularity.is_finite()
            || singularity <= min_bound
            || singularity >= max_bound
            || singularity <= previous
        {
            return Err(SelfEnergyError::InvalidSingularity {
                index,
                value: singularity,
            });
        }
        previous = singularity;
    }

    Ok(())
}

fn ensure_positive_tolerance(name: &'static str, value: Real) -> Result<(), SelfEnergyError> {
    ensure_finite_real(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SelfEnergyError::NonPositiveTolerance { name, value })
    }
}
