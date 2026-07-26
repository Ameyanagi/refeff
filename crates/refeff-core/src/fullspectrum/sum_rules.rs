//! FULLSPECTRUM density, Drude, valence, and sum-rule kernels.

use ndarray::Array1;

use crate::interpolation::{LintCache, lint_with_cache};
use crate::{Complex, Real};

use super::constants::{FEFF_ALPHA_INV, FEFF_BOHR_ANGSTROM, FEFF_HARTREE_EV, FEFF_HBAR_EV_SECONDS};
use super::types::*;
use super::validation::{
    push_sum_rule_value, validate_active_len, validate_finite_value, validate_matching_len,
    validate_positive, validate_transform_len,
};

/// Port of `FULLSPECTRUM/qsum.f90`: compute the effective electron count.
///
/// FEFF applies a trapezoid integral to `omega * eps2`, then scales it by
/// `1 / (2*pi^2*numden)`. An active length of zero or one follows the Fortran
/// loop semantics and returns zero.
pub fn full_spectrum_effective_electron_count(
    input: FullSpectrumQSumInput<'_>,
) -> Result<Real, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    validate_active_len("epsilon2", input.active_len, input.epsilon2.len())?;
    validate_active_len("omega", input.active_len, input.omega.len())?;

    for row in 0..input.active_len {
        validate_finite_value("epsilon2", row, input.epsilon2[row])?;
        validate_finite_value("omega", row, input.omega[row])?;
        if row > 0 && input.omega[row] < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: input.omega[row],
            });
        }
    }

    let integral = (0..input.active_len.saturating_sub(1))
        .map(|row| {
            let left = input.omega[row] * input.epsilon2[row];
            let right = input.omega[row + 1] * input.epsilon2[row + 1];
            0.5 * (left + right) * (input.omega[row + 1] - input.omega[row])
        })
        .sum::<Real>();
    let result = integral / (2.0 * std::f64::consts::PI.powi(2) * input.number_density);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value: result })
    }
}

/// Port of `FULLSPECTRUM/drdtrm.f90`: Drude free-electron epsilon term.
///
/// FEFF converts `tau` to a Hartree-scale width, uses `wp2 = 4*pi*numden`,
/// and writes both the eV-scaled width/plasma frequency and the complex
/// dielectric response on the active energy grid.
pub fn full_spectrum_drude_term(
    input: FullSpectrumDrudeInput<'_>,
) -> Result<FullSpectrumDrudeTerm, FullSpectrumError> {
    validate_positive("lifetime_seconds", input.lifetime_seconds)?;
    validate_finite_value("number_density", 0, input.number_density)?;
    if input.number_density < 0.0 {
        return Err(FullSpectrumError::NegativeValue {
            field: "number_density",
            row: 0,
            value: input.number_density,
        });
    }
    if input.omega.is_empty() {
        return Err(FullSpectrumError::EmptyTable { name: "drude_term" });
    }

    let gamma_hartree = FEFF_HBAR_EV_SECONDS / input.lifetime_seconds / FEFF_HARTREE_EV;
    let plasma_squared = 4.0 * std::f64::consts::PI * input.number_density;
    let mut epsilon = Vec::with_capacity(input.omega.len());

    for (row, omega) in input.omega.iter().copied().enumerate() {
        validate_finite_value("omega", row, omega)?;
        if omega <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "omega",
                row,
                value: omega,
            });
        }
        let denominator = omega.powi(2) + gamma_hartree.powi(2);
        let epsilon2 = plasma_squared * gamma_hartree / omega / denominator;
        let epsilon1 = -plasma_squared / denominator;
        let value = Complex::new(epsilon1, epsilon2);
        validate_finite_value("epsilon real", row, value.re)?;
        validate_finite_value("epsilon imaginary", row, value.im)?;
        epsilon.push(value);
    }

    Ok(FullSpectrumDrudeTerm {
        gamma_ev: gamma_hartree * FEFF_HARTREE_EV,
        plasma_frequency_ev: plasma_squared.sqrt() * FEFF_HARTREE_EV,
        omega: input.omega.to_owned(),
        epsilon: Array1::from_vec(epsilon),
    })
}

/// Port of `FULLSPECTRUM/rdval.f90`: valence `xmu.dat` contribution to eps2.
///
/// FEFF reads a valence `xmu.dat`, converts the normalized absorption to an
/// absolute cross section before this step, then linearly interpolates
/// `mu * 4*pi*alpha_inv*bohr^2*numden` onto the FULLSPECTRUM grid and divides
/// by the target photon energy.
pub fn full_spectrum_valence_epsilon2(
    input: FullSpectrumValenceInput<'_>,
) -> Result<Array1<Real>, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    validate_matching_len(
        "source_absorption_angstrom2",
        input.source_absorption_angstrom2.len(),
        input.source_energy_ev.len(),
    )?;
    validate_transform_len("source_energy_ev", input.source_energy_ev.len())?;

    for (row, value) in input.omega.iter().copied().enumerate() {
        validate_finite_value("omega", row, value)?;
        if value <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "omega",
                row,
                value,
            });
        }
    }

    let mut source_energy = Vec::with_capacity(input.source_energy_ev.len());
    for (row, energy_ev) in input.source_energy_ev.iter().copied().enumerate() {
        validate_finite_value("source_energy_ev", row, energy_ev)?;
        if energy_ev <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "source_energy_ev",
                row,
                value: energy_ev,
            });
        }
        let energy_hartree = energy_ev / FEFF_HARTREE_EV;
        if row > 0 && energy_hartree <= source_energy[row - 1] {
            return Err(FullSpectrumError::NonIncreasingOmega {
                row,
                previous: source_energy[row - 1],
                current: energy_hartree,
            });
        }
        source_energy.push(energy_hartree);
    }

    let scale = 4.0
        * std::f64::consts::PI
        * FEFF_ALPHA_INV
        * FEFF_BOHR_ANGSTROM.powi(2)
        * input.number_density;
    let mut source_absorption = Vec::with_capacity(input.source_absorption_angstrom2.len());
    for (row, absorption) in input
        .source_absorption_angstrom2
        .iter()
        .copied()
        .enumerate()
    {
        validate_finite_value("source_absorption_angstrom2", row, absorption)?;
        source_absorption.push(absorption * scale);
    }

    let final_source_energy = source_energy[source_energy.len() - 1];
    let mut epsilon2 = vec![0.0; input.omega.len()];
    let mut cache = LintCache::new();
    for (row, omega) in input.omega.iter().copied().enumerate() {
        if omega < final_source_energy {
            let interpolated =
                lint_with_cache(&source_energy, &source_absorption, omega, &mut cache)
                    .map_err(|source| FullSpectrumError::Interpolation { source })?;
            let value = interpolated / omega;
            if !value.is_finite() {
                return Err(FullSpectrumError::NonFiniteResult { value });
            }
            epsilon2[row] = value;
        }
    }

    Ok(Array1::from_vec(epsilon2))
}

/// Port of `FULLSPECTRUM/rddens.f90`: estimate a species number density.
///
/// FEFF divides the total multiplicity of potentials with atomic number `iz`
/// by the sum of Norman-sphere volumes `xnatph * 4*pi*rnrm^3/3`.
pub fn full_spectrum_number_density(
    input: FullSpectrumNumberDensityInput<'_>,
) -> Result<Real, FullSpectrumError> {
    if input.target_atomic_number == 0 {
        return Err(FullSpectrumError::InvalidAtomicNumber {
            atomic_number: input.target_atomic_number,
        });
    }
    let potential_count = input.atomic_numbers.len();
    validate_matching_len(
        "potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    validate_matching_len("norman_radii", input.norman_radii.len(), potential_count)?;
    if potential_count == 0 {
        return Err(FullSpectrumError::EmptyTable {
            name: "number_density",
        });
    }

    let mut target_atoms = 0.0;
    let mut total_volume = 0.0;
    for row in 0..potential_count {
        let atomic_number = input.atomic_numbers[row];
        if atomic_number == 0 {
            return Err(FullSpectrumError::InvalidAtomicNumber { atomic_number });
        }

        let multiplicity = input.potential_multiplicities[row];
        validate_finite_value("potential_multiplicities", row, multiplicity)?;
        if multiplicity <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "potential_multiplicities",
                row,
                value: multiplicity,
            });
        }

        let radius = input.norman_radii[row];
        validate_finite_value("norman_radii", row, radius)?;
        if radius <= 0.0 {
            return Err(FullSpectrumError::NonPositiveValue {
                field: "norman_radii",
                row,
                value: radius,
            });
        }

        if atomic_number == input.target_atomic_number {
            target_atoms += multiplicity;
        }
        total_volume += multiplicity * radius.powi(3) * 4.0 * std::f64::consts::PI / 3.0;
    }

    let value = target_atoms / total_volume;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value })
    }
}

/// Port of `FULLSPECTRUM/sumrules.f90`: cumulative optical sum rules.
///
/// FEFF reads `opconsKK.dat`, recomputes the loss function from `epsilon - 1`,
/// converts energy and absorption units to atomic units, then writes a row after
/// every input point. This helper keeps the same integration rules, including
/// the right-endpoint rule for `mu` columns and trapezoids for energy-weighted
/// dielectric/loss columns.
pub fn full_spectrum_sum_rules(
    input: FullSpectrumSumRulesInput<'_>,
) -> Result<FullSpectrumSumRules, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    if input.energy_ev.is_empty() {
        return Err(FullSpectrumError::EmptyTable { name: "sum_rules" });
    }
    validate_matching_len(
        "epsilon_minus_one",
        input.epsilon_minus_one.len(),
        input.energy_ev.len(),
    )?;
    validate_matching_len(
        "refractive_index_minus_one",
        input.refractive_index_minus_one.len(),
        input.energy_ev.len(),
    )?;
    validate_matching_len(
        "absorption_coefficient",
        input.absorption_coefficient.len(),
        input.energy_ev.len(),
    )?;

    let mut epsilon2_effective_electrons = Vec::with_capacity(input.energy_ev.len());
    let mut absorption_effective_electrons = Vec::with_capacity(input.energy_ev.len());
    let mut loss_effective_electrons = Vec::with_capacity(input.energy_ev.len());
    let mut absorption_refractive_sum = Vec::with_capacity(input.energy_ev.len());
    let mut refractive_index_sum_ratio = Vec::with_capacity(input.energy_ev.len());
    let mut log_loss_moment_ratio = Vec::with_capacity(input.energy_ev.len());

    let scale = 1.0 / (2.0 * std::f64::consts::PI.powi(2) * input.number_density);
    let mut previous_energy_hartree = 0.0;
    let mut previous_epsilon2 = 0.0;
    let mut previous_loss = 0.0;
    let mut sum_epsilon2 = 0.0;
    let mut sum_absorption = 0.0;
    let mut sum_loss = 0.0;
    let mut sum_absorption_refractive = 0.0;
    let mut sum_refractive = 0.0;
    let mut sum_abs_refractive = 0.0;
    let mut sum_log_loss = 0.0;

    for row in 0..input.energy_ev.len() {
        let energy_ev = input.energy_ev[row];
        let epsilon = input.epsilon_minus_one[row];
        let refractive_index = input.refractive_index_minus_one[row];
        let absorption = input.absorption_coefficient[row];

        validate_finite_value("energy_ev", row, energy_ev)?;
        validate_finite_value("epsilon_minus_one real", row, epsilon.re)?;
        validate_finite_value("epsilon_minus_one imaginary", row, epsilon.im)?;
        validate_finite_value("refractive_index_minus_one real", row, refractive_index.re)?;
        validate_finite_value(
            "refractive_index_minus_one imaginary",
            row,
            refractive_index.im,
        )?;
        validate_finite_value("absorption_coefficient", row, absorption)?;
        if row > 0 && energy_ev < input.energy_ev[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.energy_ev[row - 1],
                current: energy_ev,
            });
        }

        let energy_hartree = energy_ev / FEFF_HARTREE_EV;
        let mu_atomic = absorption * FEFF_BOHR_ANGSTROM / 1000.0;
        let epsilon2 = epsilon.im;
        let refractive_real = refractive_index.re;
        let loss = epsilon2 / (epsilon2.powi(2) + (epsilon.re + 1.0).powi(2));
        let delta_energy = energy_hartree - previous_energy_hartree;

        sum_epsilon2 += 0.5
            * (energy_hartree * epsilon2 + previous_energy_hartree * previous_epsilon2)
            * delta_energy;
        sum_absorption += mu_atomic * delta_energy;
        sum_loss +=
            0.5 * (energy_hartree * loss + previous_energy_hartree * previous_loss) * delta_energy;
        sum_absorption_refractive += mu_atomic * refractive_real * delta_energy;
        sum_refractive += delta_energy * refractive_real;
        sum_abs_refractive += delta_energy * refractive_real.abs();
        if previous_energy_hartree > 0.0 {
            sum_log_loss += 0.5
                * (energy_hartree.ln() * energy_hartree * loss
                    + previous_energy_hartree.ln() * previous_energy_hartree * previous_loss)
                * delta_energy;
        } else {
            sum_log_loss += energy_hartree.ln() * energy_hartree * loss * energy_hartree;
        }

        push_sum_rule_value(
            &mut epsilon2_effective_electrons,
            "epsilon2_effective_electrons",
            row,
            scale * sum_epsilon2,
        )?;
        push_sum_rule_value(
            &mut absorption_effective_electrons,
            "absorption_effective_electrons",
            row,
            FEFF_ALPHA_INV * scale * sum_absorption,
        )?;
        push_sum_rule_value(
            &mut loss_effective_electrons,
            "loss_effective_electrons",
            row,
            scale * sum_loss,
        )?;
        push_sum_rule_value(
            &mut absorption_refractive_sum,
            "absorption_refractive_sum",
            row,
            FEFF_ALPHA_INV * scale * sum_absorption_refractive,
        )?;
        push_sum_rule_value(
            &mut refractive_index_sum_ratio,
            "refractive_index_sum_ratio",
            row,
            sum_refractive / sum_abs_refractive,
        )?;
        push_sum_rule_value(
            &mut log_loss_moment_ratio,
            "log_loss_moment_ratio",
            row,
            sum_log_loss / sum_loss,
        )?;

        previous_energy_hartree = energy_hartree;
        previous_epsilon2 = epsilon2;
        previous_loss = loss;
    }

    Ok(FullSpectrumSumRules {
        energy_ev: input.energy_ev.to_owned(),
        epsilon2_effective_electrons: Array1::from_vec(epsilon2_effective_electrons),
        absorption_effective_electrons: Array1::from_vec(absorption_effective_electrons),
        loss_effective_electrons: Array1::from_vec(loss_effective_electrons),
        absorption_refractive_sum: Array1::from_vec(absorption_refractive_sum),
        refractive_index_sum_ratio: Array1::from_vec(refractive_index_sum_ratio),
        log_loss_moment_ratio: Array1::from_vec(log_loss_moment_ratio),
    })
}
