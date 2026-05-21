use super::*;

/// Port of FEFF `sigm3`: correlated Einstein-model Morse cumulants.
///
/// `mean_square_relative_displacement` is FEFF `sig2`, `temperature` is `tk`,
/// `thermal_expansion` is `alphat` in inverse angstrom, and
/// `einstein_temperature` is `thetae`. FEFF stores several intermediates as
/// single precision `real`; this port keeps those roundings to match the
/// reference values.
pub fn morse_einstein_cumulants(
    mean_square_relative_displacement: Real,
    temperature: Real,
    thermal_expansion: Real,
    einstein_temperature: Real,
) -> Result<MorseCumulants, DebyeError> {
    ensure_positive("sig2", mean_square_relative_displacement)?;
    ensure_positive("tk", temperature)?;
    ensure_finite("alphat", thermal_expansion)?;
    ensure_positive("thetae", einstein_temperature)?;

    let scaled_thermal_expansion = thermal_expansion * BOHR_ANGSTROM;
    let z = to_feff_real((-einstein_temperature / temperature).exp());
    let occupation_ratio = to_feff_real(((1.0_f32 - z as f32) / (1.0_f32 + z as f32)) as Real);
    let sig02 = to_feff_real(occupation_ratio * mean_square_relative_displacement);
    let sig01 = to_feff_real(scaled_thermal_expansion * sig02 * 0.75);
    let first = sig01 * mean_square_relative_displacement / sig02;
    let third = (2.0
        - ((4.0_f32 / 3.0_f32) as Real) * (sig02 / mean_square_relative_displacement).powi(2))
        * first
        * mean_square_relative_displacement;

    ensure_finite_output("sig1", first)?;
    ensure_finite_output("sig3", third)?;
    ensure_finite_output("alphat", scaled_thermal_expansion)?;

    Ok(MorseCumulants {
        first,
        third,
        scaled_thermal_expansion,
    })
}

/// Port of FEFF `sigte3`: thermal-expansion first and third cumulants.
///
/// `central_atomic_number` and `neighbor_atomic_number` are FEFF `iz1` and
/// `iz2`; `mean_square_relative_displacement` is `sig2`; `thermal_expansion`
/// is `alphat`; `debye_temperature` is `thetad`; and
/// `effective_distance_angstrom` is FEFF's single-precision `reff`.
pub fn thermal_expansion_cumulants(
    central_atomic_number: usize,
    neighbor_atomic_number: usize,
    mean_square_relative_displacement: Real,
    thermal_expansion: Real,
    debye_temperature: Real,
    effective_distance_angstrom: Real,
) -> Result<ThermalExpansionCumulants, DebyeError> {
    let central_mass = atomic_weight(central_atomic_number)? * ATOMIC_MASS_UNIT;
    let neighbor_mass = atomic_weight(neighbor_atomic_number)? * ATOMIC_MASS_UNIT;
    ensure_positive("sig2", mean_square_relative_displacement)?;
    ensure_finite("alphat", thermal_expansion)?;
    ensure_positive("thetad", debye_temperature)?;
    ensure_positive("reff", effective_distance_angstrom)?;

    let reff = to_feff_real(effective_distance_angstrom);
    let reduced_mass = 1.0 / (1.0 / central_mass + 1.0 / neighbor_mass);
    let omega = (2.0 * BOLTZMANN * debye_temperature) / (3.0 * HBAR);
    let spring_constant = reduced_mass * omega.powi(2);
    let cubic_force_constant =
        spring_constant.powi(2) * reff * thermal_expansion / (3.0 * BOLTZMANN);
    let sig02 = HBAR * omega / spring_constant;
    let first = -3.0 * (cubic_force_constant / spring_constant) * mean_square_relative_displacement;
    let third = (2.0
        - ((4.0_f32 / 3.0_f32) as Real) * (sig02 / mean_square_relative_displacement).powi(2))
        * first
        * mean_square_relative_displacement;

    ensure_finite_output("sig1", first)?;
    ensure_finite_output("sig3", third)?;

    Ok(ThermalExpansionCumulants { first, third })
}

/// Port of FEFF `corrfn`: quantum Debye displacement correlation.
///
/// `distance_angstrom` is FEFF `rij`, `debye_temperature` is `thetad`,
/// `temperature` is `tk`, the atomic numbers are `iz1`/`iz2`, and
/// `average_wigner_seitz_radius_bohr` is `rsavg`.
pub fn quantum_debye_correlation(
    distance_angstrom: Real,
    debye_temperature: Real,
    temperature: Real,
    first_atomic_number: usize,
    second_atomic_number: usize,
    average_wigner_seitz_radius_bohr: Real,
) -> Result<DebyeCorrelation, DebyeError> {
    debye_correlation(DebyeCorrelationInput {
        routine: "corrfn",
        distance_angstrom,
        debye_temperature,
        temperature,
        first_atomic_number,
        second_atomic_number,
        average_wigner_seitz_radius_bohr,
        integrand: quantum_debye_integrand,
    })
}

/// Port of FEFF `corrfn2`: classical Debye displacement correlation.
pub fn classical_debye_correlation(
    distance_angstrom: Real,
    debye_temperature: Real,
    temperature: Real,
    first_atomic_number: usize,
    second_atomic_number: usize,
    average_wigner_seitz_radius_bohr: Real,
) -> Result<DebyeCorrelation, DebyeError> {
    debye_correlation(DebyeCorrelationInput {
        routine: "corrfn2",
        distance_angstrom,
        debye_temperature,
        temperature,
        first_atomic_number,
        second_atomic_number,
        average_wigner_seitz_radius_bohr,
        integrand: classical_debye_integrand,
    })
}

/// Port of FEFF `sigms`: quantum Debye-Waller factor for a scattering path.
///
/// `positions_angstrom` is an `(nleg + 1) x 3` ndarray view. Row `0` and the
/// final row correspond to FEFF's central-atom endpoints for a closed path.
/// `atomic_numbers` must contain one atomic number for each row.
pub fn quantum_debye_waller_factor(
    temperature: Real,
    debye_temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    positions_angstrom: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
) -> Result<Real, DebyeError> {
    debye_waller_factor(DebyeWallerInput {
        temperature,
        debye_temperature,
        average_wigner_seitz_radius_bohr,
        positions_angstrom,
        atomic_numbers,
        correlation: quantum_debye_correlation,
    })
}

/// Port of FEFF `sigcl`: classical Debye-Waller factor for a scattering path.
pub fn classical_debye_waller_factor(
    temperature: Real,
    debye_temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    positions_angstrom: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
) -> Result<Real, DebyeError> {
    debye_waller_factor(DebyeWallerInput {
        temperature,
        debye_temperature,
        average_wigner_seitz_radius_bohr,
        positions_angstrom,
        atomic_numbers,
        correlation: classical_debye_correlation,
    })
}

type CorrelationFn =
    fn(Real, Real, Real, usize, usize, Real) -> Result<DebyeCorrelation, DebyeError>;

struct DebyeWallerInput<'a> {
    temperature: Real,
    debye_temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    positions_angstrom: ArrayView2<'a, Real>,
    atomic_numbers: &'a [usize],
    correlation: CorrelationFn,
}

fn debye_waller_factor(input: DebyeWallerInput<'_>) -> Result<Real, DebyeError> {
    let DebyeWallerInput {
        temperature,
        debye_temperature,
        average_wigner_seitz_radius_bohr,
        positions_angstrom,
        atomic_numbers,
        correlation,
    } = input;
    validate_path(positions_angstrom, atomic_numbers)?;
    ensure_nonnegative("tk", temperature)?;
    ensure_positive("thetad", debye_temperature)?;
    ensure_positive("rsavg", average_wigner_seitz_radius_bohr)?;

    let nleg = positions_angstrom.nrows() - 1;
    let mut total = 0.0;
    for il in 1..=nleg {
        for jl in il..=nleg {
            let cij = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il,
                jl,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let cimjm = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il - 1,
                jl - 1,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let cijm = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il,
                jl - 1,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let cimj = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il - 1,
                jl,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let first_leg = path_segment(positions_angstrom, il)?;
            let second_leg = path_segment(positions_angstrom, jl)?;
            let first_norm = vector_norm(first_leg);
            let second_norm = vector_norm(second_leg);
            let leg_projection = dot(first_leg, second_leg) / (first_norm * second_norm);
            let mut contribution = cij + cimjm - cijm - cimj;
            if jl != il {
                contribution *= 2.0;
            }
            total += contribution * leg_projection;
        }
    }

    let value = total / 4.0;
    ensure_finite_output("sig2", value)?;
    Ok(value)
}

#[derive(Clone, Copy)]
struct DebyePathCorrelation {
    debye_temperature: Real,
    temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    correlation: CorrelationFn,
}

fn validate_path(
    positions: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
) -> Result<(), DebyeError> {
    if positions.nrows() < 2 || positions.ncols() != 3 {
        return Err(DebyeError::InvalidPathShape {
            rows: positions.nrows(),
            columns: positions.ncols(),
        });
    }
    if positions.nrows() != atomic_numbers.len() {
        return Err(DebyeError::InvalidAtomicNumberCount {
            positions: positions.nrows(),
            atomic_numbers: atomic_numbers.len(),
        });
    }
    for value in positions.iter().copied() {
        ensure_finite("path coordinate", value)?;
    }
    for &atomic_number in atomic_numbers {
        atomic_weight(atomic_number)?;
    }
    Ok(())
}

fn correlation_between_path_atoms(
    positions: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
    first: usize,
    second: usize,
    path_correlation: DebyePathCorrelation,
) -> Result<Real, DebyeError> {
    let distance = path_distance(positions, first, second);
    Ok((path_correlation.correlation)(
        distance,
        path_correlation.debye_temperature,
        path_correlation.temperature,
        atomic_numbers[first],
        atomic_numbers[second],
        path_correlation.average_wigner_seitz_radius_bohr,
    )?
    .value)
}

fn path_distance(positions: ArrayView2<'_, Real>, first: usize, second: usize) -> Real {
    vector_norm([
        positions[(first, 0)] - positions[(second, 0)],
        positions[(first, 1)] - positions[(second, 1)],
        positions[(first, 2)] - positions[(second, 2)],
    ])
}

fn path_segment(positions: ArrayView2<'_, Real>, leg: usize) -> Result<[Real; 3], DebyeError> {
    let segment = [
        positions[(leg, 0)] - positions[(leg - 1, 0)],
        positions[(leg, 1)] - positions[(leg - 1, 1)],
        positions[(leg, 2)] - positions[(leg - 1, 2)],
    ];
    if vector_norm(segment) == 0.0 {
        Err(DebyeError::ZeroLengthPathLeg { leg })
    } else {
        Ok(segment)
    }
}

struct DebyeCorrelationInput {
    routine: &'static str,
    distance_angstrom: Real,
    debye_temperature: Real,
    temperature: Real,
    first_atomic_number: usize,
    second_atomic_number: usize,
    average_wigner_seitz_radius_bohr: Real,
    integrand: fn(Real, Real, Real) -> Real,
}

fn debye_correlation(input: DebyeCorrelationInput) -> Result<DebyeCorrelation, DebyeError> {
    let DebyeCorrelationInput {
        routine,
        distance_angstrom,
        debye_temperature,
        temperature,
        first_atomic_number,
        second_atomic_number,
        average_wigner_seitz_radius_bohr,
        integrand,
    } = input;

    ensure_nonnegative("rij", distance_angstrom)?;
    ensure_positive("thetad", debye_temperature)?;
    ensure_nonnegative("tk", temperature)?;
    ensure_positive("rsavg", average_wigner_seitz_radius_bohr)?;
    let first_mass = atomic_weight(first_atomic_number)?;
    let second_mass = atomic_weight(second_atomic_number)?;

    let y_inverse = temperature / debye_temperature;
    let debye_wave_number = (9.0 * std::f64::consts::PI / 2.0).powf(1.0 / 3.0)
        / (average_wigner_seitz_radius_bohr * BOHR_ANGSTROM);
    let x = debye_wave_number * distance_angstrom;
    let factor = ((3.0_f32 / 2.0_f32) as Real) * DEBYE_CORRELATION_FACTOR
        / (debye_temperature * (first_mass * second_mass).sqrt());
    let (integral, estimated_error, iterations) =
        integrate_debye_romberg(routine, |w| integrand(w, x, y_inverse))?;
    let value = factor * integral;
    ensure_finite_output(routine, value)?;
    Ok(DebyeCorrelation {
        value,
        estimated_error,
        iterations,
    })
}

fn quantum_debye_integrand(w: Real, x: Real, y_inverse: Real) -> Real {
    if w < 1.0e-20 {
        return 2.0 * y_inverse;
    }
    let factor = if x > 0.0 { (w * x).sin() / x } else { w };
    let exp_term = (-w / y_inverse).exp();
    factor * (1.0 + exp_term) / (1.0 - exp_term)
}

fn classical_debye_integrand(w: Real, x: Real, y_inverse: Real) -> Real {
    if w < 1.0e-20 {
        return 2.0 * y_inverse;
    }
    let factor = if x > 0.0 { (w * x).sin() / x } else { w };
    factor * 2.0 * y_inverse / w
}

fn integrate_debye_romberg(
    routine: &'static str,
    integrand: impl Fn(Real) -> Real,
) -> Result<(Real, Real, usize), DebyeError> {
    let mut intervals = 1;
    let mut delta = 1.0;
    let mut previous_trapezoid = (integrand(0.0) + integrand(1.0)) / 2.0;
    let mut previous_extrapolated = previous_trapezoid;
    let mut estimated_error = Real::INFINITY;

    for iteration in 1..=DEBYE_ROMBERG_MAX_ITERATIONS {
        delta /= 2.0;
        let midpoint_sum = (1..=intervals)
            .map(|index| {
                let z = (2 * index - 1) as Real * delta;
                integrand(z)
            })
            .sum::<Real>();
        let trapezoid = previous_trapezoid / 2.0 + delta * midpoint_sum;
        let extrapolated = (4.0 * trapezoid - previous_trapezoid) / 3.0;
        estimated_error = relative_error(extrapolated, previous_extrapolated);
        if estimated_error < DEBYE_ROMBERG_TOLERANCE {
            return Ok((extrapolated, estimated_error, iteration));
        }
        previous_trapezoid = trapezoid;
        previous_extrapolated = extrapolated;
        intervals *= 2;
    }

    Err(DebyeError::IntegrationDidNotConverge {
        routine,
        iterations: DEBYE_ROMBERG_MAX_ITERATIONS,
        estimated_error,
    })
}

fn relative_error(current: Real, previous: Real) -> Real {
    if current == 0.0 {
        if previous == 0.0 { 0.0 } else { Real::INFINITY }
    } else {
        ((current - previous) / current).abs()
    }
}
