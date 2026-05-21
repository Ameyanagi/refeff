//! FEFF FOVRG exchange, nuclear, and local potential helpers.

use ndarray::Array1;

use crate::{Complex, Real};

use super::{
    FovrgError, FovrgExchangePotential, FovrgExchangePotentialInput, FovrgNuclearPotential,
    FovrgNuclearPotentialInput, FovrgPotentialDevelopment, FovrgPotentialDevelopmentInput,
    FovrgYkZkExchangeInput, complex_real_product_coefficient, exchange_coefficient_start,
    fovrg_yk_zk_exchange, real_product_coefficient, target_j_value, validate_active_len,
    validate_complex_input, validate_complex_result, validate_count_at_least, validate_finite,
    validate_matrix_cols, validate_matrix_rows, validate_nonzero_finite, validate_nonzero_kappa,
    validate_positive_finite, validate_radius, validate_real_input, validate_real_result,
};

/// Port of `FOVRG/potex.f90`: exchange-potential accumulation.
///
/// FEFF loops over bound orbitals and allowed multipoles, obtains the `yk`
/// exchange kernel from `yzkrdc`, accumulates the radial exchange potentials
/// `eg/ep`, updates their origin development coefficients `ceg/cep`, and
/// finally divides retained rows and coefficients by `cl`.
pub fn fovrg_exchange_potential(
    input: FovrgExchangePotentialInput<'_>,
) -> Result<FovrgExchangePotential, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len(
        "target_large_component",
        input.active_len,
        input.target_large_component.len(),
    )?;
    validate_active_len(
        "target_small_component",
        input.active_len,
        input.target_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "target_large_coefficients",
        input.coefficient_count,
        input.target_large_coefficients.len(),
    )?;
    validate_active_len(
        "target_small_coefficients",
        input.coefficient_count,
        input.target_small_coefficients.len(),
    )?;
    validate_matrix_rows(
        "bound_large_components",
        input.active_len,
        input.bound_large_components.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_components",
        input.active_len,
        input.bound_small_components.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_components",
        input.bound_orbital_count,
        input.bound_large_components.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_components",
        input.bound_orbital_count,
        input.bound_small_components.shape()[1],
    )?;
    validate_matrix_rows(
        "bound_large_coefficients",
        input.coefficient_count,
        input.bound_large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_coefficients",
        input.coefficient_count,
        input.bound_small_coefficients.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_coefficients",
        input.bound_orbital_count,
        input.bound_large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_coefficients",
        input.bound_orbital_count,
        input.bound_small_coefficients.shape()[1],
    )?;
    validate_matrix_rows(
        "angular_coefficients",
        input.bound_orbital_count,
        input.angular_coefficients.shape()[0],
    )?;
    validate_active_len(
        "orbital_powers",
        input.bound_orbital_count,
        input.orbital_powers.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_active_len(
        "orbital_lengths",
        input.bound_orbital_count,
        input.orbital_lengths.len(),
    )?;
    validate_active_len(
        "normalization",
        input.bound_orbital_count,
        input.normalization.len(),
    )?;
    validate_active_len(
        "radial_output_count",
        input.radial_output_count,
        input.active_len,
    )?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;
    validate_finite("target_power", input.target_power)?;
    validate_nonzero_finite("target_normalization", input.target_normalization)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input(
            "target_large_component",
            row,
            input.target_large_component[row],
        )?;
        validate_complex_input(
            "target_small_component",
            row,
            input.target_small_component[row],
        )?;
        validate_radius(row, input.radii[row])?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_components",
                row,
                input.bound_large_components[(row, orbital)],
            )?;
            validate_real_input(
                "bound_small_components",
                row,
                input.bound_small_components[(row, orbital)],
            )?;
        }
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "target_large_coefficients",
            coefficient,
            input.target_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "target_small_coefficients",
            coefficient,
            input.target_small_coefficients[coefficient],
        )?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_coefficients",
                coefficient,
                input.bound_large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "bound_small_coefficients",
                coefficient,
                input.bound_small_coefficients[(coefficient, orbital)],
            )?;
        }
    }
    for orbital in 0..input.bound_orbital_count {
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        validate_real_input("orbital_powers", orbital, input.orbital_powers[orbital])?;
        validate_real_input("normalization", orbital, input.normalization[orbital])?;
        validate_count_at_least("orbital_length", input.orbital_lengths[orbital], 1)?;
        for index in 0..input.angular_coefficients.shape()[1] {
            validate_real_input(
                "angular_coefficients",
                orbital,
                input.angular_coefficients[(orbital, index)],
            )?;
        }
    }

    let target_j = target_j_value(input.target_kappa);
    let mut large_potential = Array1::<Complex>::zeros(input.active_len);
    let mut small_potential = Array1::<Complex>::zeros(input.active_len);
    let mut large_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    let mut small_coefficients = Array1::<Complex>::zeros(input.coefficient_count);

    for orbital in 0..input.bound_orbital_count {
        let bound_j = target_j_value(input.kappa[orbital]);
        let max_multipole = (bound_j + target_j) / 2;
        let mut multipole = bound_j.abs_diff(max_multipole);
        if (input.kappa[orbital] < 0) != (input.target_kappa < 0) {
            multipole += 1;
        }
        let min_multipole = multipole;

        while multipole <= max_multipole {
            let angular_index = (multipole - min_multipole) / 2;
            validate_matrix_cols(
                "angular_coefficients",
                angular_index + 1,
                input.angular_coefficients.shape()[1],
            )?;
            let angular_coefficient = input.angular_coefficients[(orbital, angular_index)];
            if angular_coefficient != 0.0 {
                let transform = fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
                    large_component: input.bound_large_components.column(orbital),
                    small_component: input.bound_small_components.column(orbital),
                    large_coefficients: input.bound_large_coefficients.column(orbital),
                    small_coefficients: input.bound_small_coefficients.column(orbital),
                    partner_large_component: input.target_large_component,
                    partner_small_component: input.target_small_component,
                    partner_large_coefficients: input.target_large_coefficients,
                    partner_small_coefficients: input.target_small_coefficients,
                    radii: input.radii,
                    orbital_power: input.orbital_powers[orbital],
                    partner_power: input.target_power,
                    step: input.step,
                    angular_momentum: multipole,
                    coefficient_count: input.coefficient_count,
                    orbital_len: input.orbital_lengths[orbital],
                    source_len: input.source_len,
                    active_len: input.active_len,
                })?;

                for row in 0..input.active_len {
                    large_potential[row] += angular_coefficient
                        * transform.yk[row]
                        * input.bound_large_components[(row, orbital)];
                    small_potential[row] += angular_coefficient
                        * transform.yk[row]
                        * input.bound_small_components[(row, orbital)];
                }

                if let Some(coefficient_start) = exchange_coefficient_start(
                    multipole,
                    input.kappa[orbital],
                    input.target_kappa,
                    input.target_power,
                )
                .filter(|&start| start <= input.coefficient_count)
                {
                    for coefficient in coefficient_start..=input.coefficient_count {
                        let target_row = coefficient - 1;
                        let bound_row = coefficient - coefficient_start;
                        let scale = angular_coefficient
                            * transform.origin_constant
                            * input.normalization[orbital]
                            / input.target_normalization;
                        large_coefficients[target_row] +=
                            input.bound_large_coefficients[(bound_row, orbital)] * scale;
                        small_coefficients[target_row] +=
                            input.bound_small_coefficients[(bound_row, orbital)] * scale;
                    }
                }

                let product_start = 2 * input.kappa[orbital].unsigned_abs() as usize + 1;
                if product_start <= input.coefficient_count {
                    let scale = angular_coefficient * input.normalization[orbital].powi(2);
                    for coefficient in product_start..=input.coefficient_count {
                        let product_count = coefficient + 1 - product_start;
                        large_coefficients[coefficient - 1] -= scale
                            * complex_real_product_coefficient(
                                transform.yk_coefficients.view(),
                                input.bound_large_coefficients.column(orbital),
                                product_count,
                            );
                        small_coefficients[coefficient - 1] -= scale
                            * complex_real_product_coefficient(
                                transform.yk_coefficients.view(),
                                input.bound_small_coefficients.column(orbital),
                                product_count,
                            );
                    }
                }
            }
            multipole += 2;
        }
    }

    for coefficient in 0..input.coefficient_count {
        large_coefficients[coefficient] /= input.speed_of_light;
        small_coefficients[coefficient] /= input.speed_of_light;
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }
    for row in 0..input.active_len {
        if row < input.radial_output_count {
            large_potential[row] /= input.speed_of_light;
            small_potential[row] /= input.speed_of_light;
        } else {
            large_potential[row] = Complex::new(0.0, 0.0);
            small_potential[row] = Complex::new(0.0, 0.0);
        }
        validate_complex_result("large_potential", row, large_potential[row])?;
        validate_complex_result("small_potential", row, small_potential[row])?;
    }

    Ok(FovrgExchangePotential {
        large_potential,
        small_potential,
        large_coefficients,
        small_coefficients,
    })
}

/// Port of `FOVRG/nucdec.f90`: point-nucleus radial grid and potential.
///
/// FEFF10 currently resets the nuclear mass to zero inside `nucdec`, so the
/// active branch is the point-nucleus Coulomb potential:
/// `dr(i) = dr1 / dz * exp(hx * (i - 1))`, `dv(i) = -dz / dr(i)`, and
/// `av(1) = -dz` with all remaining development coefficients zero.
pub fn fovrg_nuclear_potential(
    input: FovrgNuclearPotentialInput,
) -> Result<FovrgNuclearPotential, FovrgError> {
    validate_positive_finite("nuclear_charge", input.nuclear_charge)?;
    validate_positive_finite("step", input.step)?;
    validate_positive_finite("first_radius_times_charge", input.first_radius_times_charge)?;
    validate_count_at_least("radial_count", input.radial_count, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 5)?;

    let first_radius = input.first_radius_times_charge / input.nuclear_charge;
    let mut radii = Array1::<Real>::zeros(input.radial_count);
    let mut potential = Array1::<Real>::zeros(input.radial_count);
    for row in 0..input.radial_count {
        radii[row] = first_radius * (input.step * row as Real).exp();
        validate_radius(row, radii[row])?;

        potential[row] = -input.nuclear_charge / radii[row];
        validate_real_result("nuclear_potential", row, potential[row])?;
    }

    let mut development_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    development_coefficients[0] = -input.nuclear_charge;
    validate_real_result("development_coefficients", 0, development_coefficients[0])?;

    Ok(FovrgNuclearPotential {
        development_coefficients,
        radii,
        potential,
        nucleus_index: 1,
        first_radius_times_charge: input.first_radius_times_charge,
    })
}

/// Port of `FOVRG/potdvp.f90`: potential development coefficients.
///
/// FEFF accumulates bound-orbital density development coefficients from
/// occupied large/small radial polynomials, integrates those coefficients into
/// a local potential expansion, adds the nuclear development, and divides the
/// resulting `av` coefficients by `cl`.
pub fn fovrg_potential_development(
    input: FovrgPotentialDevelopmentInput<'_>,
) -> Result<FovrgPotentialDevelopment, FovrgError> {
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_count_at_least("orbital_count", input.orbital_count, 1)?;
    validate_count_at_least("nuclear_coefficients", input.nuclear_coefficients.len(), 2)?;
    validate_count_at_least("radii", input.radii.len(), 1)?;
    validate_active_len(
        "nuclear_coefficients",
        input.coefficient_count,
        input.nuclear_coefficients.len(),
    )?;
    validate_matrix_rows(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.shape()[0],
    )?;
    let bound_orbitals = input.orbital_count - 1;
    validate_matrix_cols(
        "large_coefficients",
        bound_orbitals,
        input.large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "small_coefficients",
        bound_orbitals,
        input.small_coefficients.shape()[1],
    )?;
    validate_active_len(
        "electron_counts",
        bound_orbitals,
        input.electron_counts.len(),
    )?;
    validate_active_len("kappa", bound_orbitals, input.kappa.len())?;
    validate_active_len("normalization", bound_orbitals, input.normalization.len())?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_radius(0, input.radii[0])?;
    if input.coefficient_count > i32::MAX as usize - 1 {
        return Err(FovrgError::CountTooLarge {
            name: "coefficient_count",
            actual: input.coefficient_count,
            maximum: i32::MAX as usize - 1,
        });
    }

    for coefficient in 0..input.nuclear_coefficients.len() {
        validate_real_input(
            "nuclear_coefficients",
            coefficient,
            input.nuclear_coefficients[coefficient],
        )?;
    }
    for orbital in 0..bound_orbitals {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_real_input("normalization", orbital, input.normalization[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        for coefficient in 0..input.coefficient_count {
            validate_real_input(
                "large_coefficients",
                coefficient,
                input.large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "small_coefficients",
                coefficient,
                input.small_coefficients[(coefficient, orbital)],
            )?;
        }
    }

    let mut density_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    for orbital in 0..bound_orbitals {
        let kappa_abs = input.kappa[orbital].unsigned_abs() as usize;
        let leading_power = kappa_abs.saturating_mul(2);
        let product_count = input.coefficient_count + 2;
        if leading_power >= product_count {
            continue;
        }
        let max_product_order = product_count - leading_power;
        for product_order in 1..=max_product_order {
            let density_row = leading_power - 2 + product_order;
            density_coefficients[density_row - 1] += input.electron_counts[orbital]
                * (real_product_coefficient(
                    input.large_coefficients.column(orbital),
                    input.large_coefficients.column(orbital),
                    product_order,
                ) + real_product_coefficient(
                    input.small_coefficients.column(orbital),
                    input.small_coefficients.column(orbital),
                    product_order,
                ))
                * input.normalization[orbital].powi(2);
        }
    }

    let mut origin_correction = 0.0;
    for coefficient in 1..=input.coefficient_count {
        let row = coefficient - 1;
        density_coefficients[row] /= (coefficient + 2) as Real * (coefficient + 1) as Real;
        origin_correction +=
            density_coefficients[row] * input.radii[0].powi(coefficient as i32 + 1);
    }

    let mut potential_coefficients = Array1::from_iter(
        input
            .nuclear_coefficients
            .iter()
            .copied()
            .map(|value| Complex::new(value, 0.0)),
    );
    for coefficient in 1..=input.coefficient_count {
        let potential_row = coefficient + 3;
        if potential_row <= input.coefficient_count {
            potential_coefficients[potential_row - 1] -= density_coefficients[coefficient - 1];
        }
    }
    potential_coefficients[1] += origin_correction;
    for row in 0..potential_coefficients.len() {
        potential_coefficients[row] /= input.speed_of_light;
        validate_complex_result("potential_coefficients", row, potential_coefficients[row])?;
    }
    for row in 0..density_coefficients.len() {
        validate_real_result("density_coefficients", row, density_coefficients[row])?;
    }
    validate_real_result("origin_correction", 0, origin_correction)?;

    Ok(FovrgPotentialDevelopment {
        potential_coefficients,
        density_coefficients,
        origin_correction,
    })
}
