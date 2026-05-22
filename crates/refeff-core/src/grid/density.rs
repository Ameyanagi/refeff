use super::radial::{feff_legacy_loucks_radius, fortran_truncated_index};
use super::validation::*;
use super::*;

/// Find FEFF's Norman radius from an overlapped density profile.
///
/// This ports `POT/frnrm.f90`. FEFF integrates `rho * r**2 dr`, with `rho`
/// already stored as `4*pi*density`, until the accumulated charge reaches the
/// atom's `Z`. The first pass follows FEFF's hand-coded Simpson recurrence, then
/// the returned radius is refined by the same `somm2` endpoint correction used
/// in the original routine. The radial grid intentionally preserves FEFF's
/// default-real `xx.f90` constants before widening to double precision.
pub fn norman_radius_from_density(input: NormanRadiusInput<'_>) -> Result<NormanRadius, GridError> {
    if input.atomic_number == 0 {
        return Err(GridError::InvalidAtomicNumber {
            atomic_number: input.atomic_number,
        });
    }
    ensure_source_length(
        "overlapped_density",
        FRNRM_DENSITY_POINTS,
        input.overlapped_density.len(),
    )?;
    let density = input
        .overlapped_density
        .iter()
        .take(FRNRM_DENSITY_POINTS)
        .copied()
        .collect::<Vec<_>>();
    validate_slice_values("overlapped_density", &density)?;
    let radii = (1..=FRNRM_DENSITY_POINTS)
        .map(feff_legacy_loucks_radius)
        .collect::<Vec<_>>();
    let density_moments = density
        .iter()
        .zip(radii.iter())
        .map(|(&rho, &radius)| rho * radius * radius * radius)
        .collect::<Vec<_>>();

    let target_charge = input.atomic_number as Real;
    let scan = frnrm_initial_scan(&density, &radii, &density_moments, target_charge)?;
    let (index, mut fraction) = scan.crossing.ok_or(GridError::InsufficientNormanCharge {
        atomic_number: input.atomic_number,
        charge_found: scan.charge,
        max_radius: radii[FRNRM_DENSITY_POINTS - 1],
    })?;

    let mut radius = radii[index - 1] * (1.0 + fraction * FRNRM_LITERAL_DELTA);
    let correction_len = frnrm_correction_len(radius)?;
    ensure_source_length("overlapped_density", correction_len, FRNRM_DENSITY_POINTS)?;
    ensure_source_length("norman_correction", index + 1, correction_len)?;

    let correction_radii = &radii[..correction_len];
    let correction_values = correction_radii
        .iter()
        .zip(density.iter())
        .map(|(&ri, &rho)| rho * ri * ri)
        .collect::<Vec<_>>();

    let first_charge = somm2(
        correction_radii,
        &correction_values,
        FRNRM_LITERAL_DELTA,
        2.0,
        radius,
        0,
    )?;
    let first_delta = first_charge - target_charge;
    let density_at_radius =
        (1.0 - fraction) * correction_values[index - 1] + fraction * correction_values[index];
    validate_nonzero_finite_scalar("norman_correction_density", density_at_radius)?;

    let second_fraction = fraction - first_delta / density_at_radius;
    if (second_fraction - fraction).abs() > FRNRM_CORRECTION_THRESHOLD {
        radius = radii[index - 1] * (1.0 + second_fraction * FRNRM_LITERAL_DELTA);
        let second_charge = somm2(
            correction_radii,
            &correction_values,
            FRNRM_LITERAL_DELTA,
            2.0,
            radius,
            0,
        )?;
        let second_delta = second_charge - target_charge;
        let delta_difference = second_delta - first_delta;
        validate_nonzero_finite_scalar("norman_correction_delta", delta_difference)?;
        fraction = second_fraction - second_delta * (second_fraction - fraction) / delta_difference;
    } else {
        fraction = second_fraction;
    }

    Ok(NormanRadius {
        radius: radii[index - 1] * (1.0 + fraction * FRNRM_LITERAL_DELTA),
        index,
        fraction,
    })
}

/// Calculate FEFF's interstitial Fermi level from density and potential.
///
/// This ports `POT/fermi.f90`. FEFF stores `rhoint` as `4*pi*density`, so the
/// density parameter is `rs = (3 / rhoint)^(1/3)`, the Fermi momentum is
/// `xf = fa / rs`, and the chemical potential is `xmu = vint + xf**2 / 2`.
pub fn interstitial_fermi_level(input: FermiLevelInput) -> Result<FermiLevel, GridError> {
    validate_positive_finite_scalar("interstitial_density", input.interstitial_density)?;
    validate_finite_scalar("interstitial_potential", input.interstitial_potential)?;

    let density_parameter = (3.0 / input.interstitial_density).powf(1.0 / 3.0);
    let fermi_momentum = FEFF_FERMI_MOMENTUM_FACTOR / density_parameter;
    let chemical_potential = input.interstitial_potential + fermi_momentum.powi(2) / 2.0;

    Ok(FermiLevel {
        chemical_potential,
        density_parameter,
        fermi_momentum,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FrnrmInitialScan {
    crossing: Option<(usize, Real)>,
    charge: Real,
}

fn frnrm_initial_scan(
    density: &[Real],
    radii: &[Real],
    density_moments: &[Real],
    target_charge: Real,
) -> Result<FrnrmInitialScan, GridError> {
    let mut charge =
        (9.0 * density_moments[0] + 28.0 * density_moments[1] + 23.0 * density_moments[2]) / 480.0;
    charge += frnrm_initial_origin_correction(density, radii)?;

    let mut left = density_moments[3];
    let mut center = density_moments[4];
    let mut right = density_moments[5];

    for index in 7..=FRNRM_NRPTX {
        let far_left = left;
        left = center;
        center = right;
        right = if index <= FRNRM_DENSITY_POINTS {
            density_moments[index - 1]
        } else {
            0.0
        };
        let previous_charge = charge;
        charge += (13.0 * (center + left) - far_left - right) / 480.0;
        if charge >= target_charge {
            let increment = charge - previous_charge;
            validate_nonzero_finite_scalar("norman_charge_increment", increment)?;
            return Ok(FrnrmInitialScan {
                crossing: Some((index - 2, (target_charge - previous_charge) / increment)),
                charge,
            });
        }
    }

    Ok(FrnrmInitialScan {
        crossing: None,
        charge,
    })
}

fn frnrm_initial_origin_correction(density: &[Real], radii: &[Real]) -> Result<Real, GridError> {
    let d1 = 3.0;
    let delta = FRNRM_LITERAL_DELTA.exp() - 1.0;
    let second_coefficient =
        radii[0] / (d1 * (d1 + 1.0) * delta * ((d1 - 1.0) * FRNRM_LITERAL_DELTA).exp());
    let first_coefficient = radii[0] * (1.0 + 1.0 / (delta * (d1 + 1.0))) / d1;
    let correction = first_coefficient * density[0] * radii[0] * radii[0]
        - second_coefficient * density[1] * radii[1] * radii[1];
    validate_finite_scalar("norman_origin_correction", correction)?;
    Ok(correction)
}

fn frnrm_correction_len(radius: Real) -> Result<usize, GridError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(GridError::InvalidRadius { radius });
    }
    let grid_index =
        fortran_truncated_index((radius.ln() + FRNRM_LITERAL_OFFSET) / FRNRM_LITERAL_DELTA + 2.0);
    grid_index
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow {
            name: "norman_correction",
        })
}
