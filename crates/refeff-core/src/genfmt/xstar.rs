use super::validation::validate_finite_scalar;
use super::*;

/// Compute FEFF `xstar`, the central-atom plane-wave polarization factor.
///
/// FEFF evaluates the orientationally averaged `ystar` expression for the
/// primary polarization and, when `ellipticity != 0`, adds the secondary
/// polarization weighted by `ellipticity^2`. The vector cosines match
/// `xxcos` from `xstar.f90`, but zero-length and non-finite inputs are reported
/// as errors instead of allowing division by zero.
pub fn xstar(input: XStarInput) -> Result<Real, GenfmtError> {
    if !(1..=4).contains(&input.initial_l) {
        return Err(GenfmtError::InvalidInitialAngularMomentum {
            initial_l: input.initial_l,
        });
    }
    validate_finite_scalar("degeneracy", input.degeneracy)?;
    validate_finite_scalar("ellipticity", input.ellipticity)?;

    let x = normalized_dot("first_leg", input.first_leg, "last_leg", input.last_leg)?;
    let primary_y = normalized_dot(
        "primary_polarization",
        input.primary_polarization,
        "first_leg",
        input.first_leg,
    )?;
    let primary_z = normalized_dot(
        "primary_polarization",
        input.primary_polarization,
        "last_leg",
        input.last_leg,
    )?;
    let mut value = ystar(input.initial_l, x, primary_y, primary_z);

    if input.ellipticity != 0.0 {
        let secondary_y = normalized_dot(
            "secondary_polarization",
            input.secondary_polarization,
            "first_leg",
            input.first_leg,
        )?;
        let secondary_z = normalized_dot(
            "secondary_polarization",
            input.secondary_polarization,
            "last_leg",
            input.last_leg,
        )?;
        value += input.ellipticity
            * input.ellipticity
            * ystar(input.initial_l, x, secondary_y, secondary_z);
    }

    Ok(input.degeneracy * value / (1.0 + input.ellipticity * input.ellipticity))
}

fn normalized_dot(
    left_field: &'static str,
    left: [Real; 3],
    right_field: &'static str,
    right: [Real; 3],
) -> Result<Real, GenfmtError> {
    validate_vector(left_field, left)?;
    validate_vector(right_field, right)?;

    let dot = left.iter().zip(right).map(|(&a, b)| a * b).sum::<Real>();
    let left_norm = left.iter().map(|value| value * value).sum::<Real>();
    let right_norm = right.iter().map(|value| value * value).sum::<Real>();

    if left_norm == 0.0 {
        return Err(GenfmtError::ZeroVector { field: left_field });
    }
    if right_norm == 0.0 {
        return Err(GenfmtError::ZeroVector { field: right_field });
    }

    Ok(dot / (left_norm * right_norm).sqrt())
}

fn validate_vector(field: &'static str, vector: [Real; 3]) -> Result<(), GenfmtError> {
    for (index, value) in vector.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(GenfmtError::NonFiniteVector {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn ystar(initial_l: usize, x: Real, y: Real, z: Real) -> Real {
    const LEGENDRE: [[Real; 5]; 5] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [-0.5, 0.0, 1.5, 0.0, 0.0],
        [0.0, -1.5, 0.0, 2.5, 0.0],
        [0.375, 0.0, -3.75, 0.0, 4.375],
    ];
    let coefficients = LEGENDRE[initial_l];
    let l = initial_l as Real;

    let pln0 = coefficients
        .iter()
        .enumerate()
        .take(initial_l + 1)
        .map(|(power, coefficient)| coefficient * x.powi(power as i32))
        .sum::<Real>();
    let pln1 = coefficients
        .iter()
        .enumerate()
        .take(initial_l + 1)
        .skip(1)
        .map(|(power, coefficient)| {
            let power_real = power as Real;
            coefficient * power_real * x.powi(power as i32 - 1)
        })
        .sum::<Real>();
    let pln2 = coefficients
        .iter()
        .enumerate()
        .take(initial_l + 1)
        .skip(2)
        .map(|(power, coefficient)| {
            let power_real = power as Real;
            coefficient * power_real * (power_real - 1.0) * x.powi(power as i32 - 2)
        })
        .sum::<Real>();

    let ytemp = -l * pln0 + pln1 * (x + y * z) - pln2 * (y * y + z * z - 2.0 * x * y * z);
    ytemp * 3.0 / l / (4.0 * l * l - 1.0)
}
