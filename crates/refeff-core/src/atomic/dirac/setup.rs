use super::*;

/// Port of FEFF `ATOM/soldir.f90` setup before the first `intdir` call.
///
/// This evaluates the deterministic scalar state used by the shooting loop:
/// method fallback, point-nucleus small-component coefficient adjustment,
/// angular term, target node count, and the lower apparent-potential energy
/// bound.
pub fn atomic_dirac_solver_setup(
    input: AtomicDiracSolverSetupInput<'_>,
) -> Result<AtomicDiracSolverSetup, AtomMathError> {
    validate_dirac_solver_setup_input(&input)?;
    calculate_atomic_dirac_solver_setup(input)
}
fn calculate_atomic_dirac_solver_setup(
    input: AtomicDiracSolverSetupInput<'_>,
) -> Result<AtomicDiracSolverSetup, AtomMathError> {
    let requested_method = input.method;
    let method = if input.method <= 0 { 1 } else { input.method };
    let kappa = input.kappa as Real;
    let doubled_speed_of_light = input.speed_of_light + input.speed_of_light;
    let potential_origin = input.potential_coefficients[0];
    let mut initial_small_coefficient = input.initial_small_coefficient;

    if potential_origin < 0.0 {
        if input.kappa > 0 {
            initial_small_coefficient =
                -input.initial_large_coefficient * (kappa + input.origin_power) / potential_origin;
        } else if input.kappa < 0 {
            let denominator = kappa - input.origin_power;
            if denominator == 0.0 {
                return Err(AtomMathError::ZeroDiracSolverInitialCoefficientDenominator);
            }
            initial_small_coefficient =
                -input.initial_large_coefficient * potential_origin / denominator;
        }
    }

    let angular_term = kappa * (kappa + 1.0) / doubled_speed_of_light;
    let kappa_abs = input.kappa.unsigned_abs();
    let principal = i32::try_from(input.principal_quantum_number).map_err(|_| {
        AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        }
    })?;
    let kappa_abs_i32 = i32::try_from(kappa_abs).map_err(|_| {
        AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        }
    })?;
    let mut target_nodes = principal - kappa_abs_i32;
    if input.kappa < 0 {
        target_nodes += 1;
    }

    let mut energy_floor = 0.0;
    for row in 0..input.active_len {
        let radius = input.radii[row];
        let apparent =
            (angular_term / (radius * radius) + input.potential[row]) * input.speed_of_light;
        if apparent < energy_floor {
            energy_floor = apparent;
        }
    }
    if energy_floor >= 0.0 {
        return Err(AtomMathError::DiracSolverPotentialNotAttractive { energy_floor });
    }

    let energy = if input.energy < energy_floor {
        energy_floor * 0.9
    } else {
        input.energy
    };

    validate_finite_scalar("soldir_setup_energy", energy)?;
    validate_finite_scalar("soldir_setup_energy_floor", energy_floor)?;
    validate_finite_scalar(
        "soldir_setup_initial_small_coefficient",
        initial_small_coefficient,
    )?;
    validate_finite_scalar("soldir_setup_angular_term", angular_term)?;

    Ok(AtomicDiracSolverSetup {
        requested_method,
        method,
        energy,
        energy_floor,
        initial_small_coefficient,
        angular_term,
        target_nodes,
        doubled_speed_of_light,
    })
}
