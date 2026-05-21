use crate::{Real, quadrature::somm};
use ndarray::Array1;

use super::{
    ATOM_FPF0_BOHR_ANGSTROM, ATOM_FPF0_FINE_STRUCTURE, ATOM_FPF0_FORM_FACTOR_POINTS,
    ATOM_FPF0_MOMENTUM_STEP_INV_ANGSTROM, AtomMathError, AtomicFormFactor, AtomicFormFactorInput,
    AtomicFormFactorOscillator, abs_kappa_i32, validate_finite_matrix, validate_finite_scalar,
    validate_finite_slice, validate_finite_vector, validate_matrix_shape,
    validate_orbital_table_len, validate_radial_table_len,
};

/// Port of FEFF `ATOM/fpf0.f90`, excluding direct file IO.
///
/// The returned structure mirrors the contents of `fpf0.dat`: scalar f-prime
/// corrections, dipole oscillator rows, and the fixed 81-point `f0(Q)` table on
/// FEFF's `0.5 Angstrom^-1` grid.
pub fn atomic_form_factor(
    input: AtomicFormFactorInput<'_>,
) -> Result<AtomicFormFactor, AtomMathError> {
    validate_form_factor_input(&input)?;
    AtomicFormFactorContext { input }.calculate()
}

struct AtomicFormFactorContext<'a> {
    input: AtomicFormFactorInput<'a>,
}

impl AtomicFormFactorContext<'_> {
    fn calculate(&self) -> Result<AtomicFormFactor, AtomMathError> {
        let total_energy_fprime =
            self.input.total_energy * ATOM_FPF0_FINE_STRUCTURE.powi(2) * 5.0 / 3.0;
        let relativistic_correction = -((self.input.atomic_number as Real) / 82.5).powf(2.37);
        validate_finite_scalar("fpf0_total_energy_fprime", total_energy_fprime)?;
        validate_finite_scalar("fpf0_relativistic_correction", relativistic_correction)?;

        let radii = self.input.radii.iter().copied().collect::<Vec<_>>();
        let zeros = vec![0.0; radii.len()];
        let oscillators = self.oscillators(&radii, &zeros)?;
        let (form_factor_momentum, form_factor) = self.form_factor_table(&radii, &zeros)?;

        Ok(AtomicFormFactor {
            atomic_number: self.input.atomic_number,
            total_energy_fprime,
            relativistic_correction,
            oscillators,
            form_factor_momentum,
            form_factor,
        })
    }

    fn oscillators(
        &self,
        radii: &[Real],
        zeros: &[Real],
    ) -> Result<Vec<AtomicFormFactorOscillator>, AtomMathError> {
        let hole = self.input.hole_orbital_1based - 1;
        let initial_kappa = self.input.kappas[hole];
        let mut oscillators = vec![AtomicFormFactorOscillator {
            oscillator_strength: 2.0 * Real::from(abs_kappa_i32(initial_kappa)?),
            excitation_energy: self.input.orbital_energies[hole],
            orbital_index_1based: self.input.hole_orbital_1based,
        }];

        for orbital in 0..self.input.kappas.len() {
            if self.input.occupations[orbital] <= 0.0 {
                continue;
            }
            let Some((large_multiplier, small_multiplier)) =
                fpf0_dipole_multipliers(initial_kappa, self.input.kappas[orbital])?
            else {
                continue;
            };
            let wave_number =
                (self.input.orbital_energies[orbital] - self.input.orbital_energies[hole]).abs()
                    * ATOM_FPF0_FINE_STRUCTURE;
            let integrand = radii
                .iter()
                .enumerate()
                .map(|(radial, &radius)| {
                    let bessel = fpf0_spherical_bessel_j0(wave_number * radius);
                    (large_multiplier
                        * self.input.initial_large_component[radial]
                        * self.input.small_components[(radial, orbital)]
                        + small_multiplier
                            * self.input.initial_small_component[radial]
                            * self.input.large_components[(radial, orbital)])
                        * bessel
                })
                .collect::<Vec<_>>();
            validate_finite_slice("fpf0_oscillator_integrand", &integrand)?;
            let radial_integral = somm(radii, &integrand, zeros, self.input.radial_step, 2.0, 0)?;
            let oscillator_strength = radial_integral * radial_integral / 3.0;
            validate_finite_scalar("fpf0_oscillator_strength", oscillator_strength)?;
            oscillators.push(AtomicFormFactorOscillator {
                oscillator_strength,
                excitation_energy: self.input.orbital_energies[orbital],
                orbital_index_1based: orbital + 1,
            });
        }

        Ok(oscillators)
    }

    fn form_factor_table(
        &self,
        radii: &[Real],
        zeros: &[Real],
    ) -> Result<(Array1<Real>, Array1<Real>), AtomMathError> {
        let momentum = Array1::from_shape_fn(ATOM_FPF0_FORM_FACTOR_POINTS, |index| {
            ATOM_FPF0_MOMENTUM_STEP_INV_ANGSTROM * index as Real
        });
        let mut form_factor = Array1::<Real>::zeros(ATOM_FPF0_FORM_FACTOR_POINTS);

        for (index, value) in form_factor.iter_mut().enumerate() {
            let wave_number =
                ATOM_FPF0_MOMENTUM_STEP_INV_ANGSTROM * ATOM_FPF0_BOHR_ANGSTROM * index as Real;
            let integrand = radii
                .iter()
                .enumerate()
                .map(|(radial, &radius)| {
                    self.input.density_4pi[radial]
                        * radius
                        * radius
                        * fpf0_spherical_bessel_j0(wave_number * radius)
                })
                .collect::<Vec<_>>();
            validate_finite_slice("fpf0_form_factor_integrand", &integrand)?;
            *value = somm(radii, &integrand, zeros, self.input.radial_step, 2.0, 0)?;
            validate_finite_scalar("fpf0_form_factor", *value)?;
        }

        Ok((momentum, form_factor))
    }
}

fn fpf0_dipole_multipliers(
    initial_kappa: i32,
    final_kappa: i32,
) -> Result<Option<(Real, Real)>, AtomMathError> {
    abs_kappa_i32(initial_kappa)?;
    abs_kappa_i32(final_kappa)?;
    let kappa_sum =
        initial_kappa
            .checked_add(final_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa: initial_kappa,
                right_kappa: final_kappa,
            })?;
    let mut kappa_difference =
        final_kappa
            .checked_sub(initial_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa: initial_kappa,
                right_kappa: final_kappa,
            })?;
    let difference_abs =
        kappa_difference
            .checked_abs()
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa: initial_kappa,
                right_kappa: final_kappa,
            })?;
    if kappa_sum != 0 && difference_abs != 1 {
        return Ok(None);
    }
    if difference_abs > 1 {
        kappa_difference = 0;
    }

    let two_j = 2.0 * Real::from(abs_kappa_i32(initial_kappa)?) - 1.0;
    let multipliers = match (kappa_difference, initial_kappa.is_positive()) {
        (-1, true) => (0.0, (2.0 * (two_j + 1.0) * (two_j - 1.0) / two_j).sqrt()),
        (-1, false) => (
            0.0,
            -(2.0 * (two_j + 1.0) * (two_j + 3.0) / (two_j + 2.0)).sqrt(),
        ),
        (0, true) => (
            -((two_j + 1.0) * two_j / (two_j + 2.0)).sqrt(),
            -((two_j + 1.0) * (two_j + 2.0) / two_j).sqrt(),
        ),
        (0, false) => (
            ((two_j + 1.0) * (two_j + 2.0) / two_j).sqrt(),
            ((two_j + 1.0) * two_j / (two_j + 2.0)).sqrt(),
        ),
        (1, true) => (
            (2.0 * (two_j + 1.0) * (two_j + 3.0) / (two_j + 2.0)).sqrt(),
            0.0,
        ),
        (1, false) => (-(2.0 * (two_j + 1.0) * (two_j - 1.0) / two_j).sqrt(), 0.0),
        _ => return Ok(None),
    };
    validate_finite_scalar("fpf0_large_multiplier", multipliers.0)?;
    validate_finite_scalar("fpf0_small_multiplier", multipliers.1)?;
    Ok(Some(multipliers))
}

fn fpf0_spherical_bessel_j0(argument: Real) -> Real {
    if argument == 0.0 {
        1.0
    } else {
        argument.sin() / argument
    }
}

fn validate_form_factor_input(input: &AtomicFormFactorInput<'_>) -> Result<(), AtomMathError> {
    if input.atomic_number == 0 {
        return Err(AtomMathError::InvalidFormFactorAtomicNumber {
            atomic_number: input.atomic_number,
        });
    }
    validate_finite_scalar("radial_step", input.radial_step)?;
    validate_finite_scalar("total_energy", input.total_energy)?;

    let radial_count = input.radii.len();
    validate_radial_table_len("density_4pi", radial_count, input.density_4pi.len())?;
    validate_radial_table_len(
        "initial_large_component",
        radial_count,
        input.initial_large_component.len(),
    )?;
    validate_radial_table_len(
        "initial_small_component",
        radial_count,
        input.initial_small_component.len(),
    )?;

    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "orbital_energies",
        orbital_count,
        input.orbital_energies.len(),
    )?;
    if !(1..=orbital_count).contains(&input.hole_orbital_1based) {
        return Err(AtomMathError::HoleOrbitalOutOfRange {
            hole_orbital_1based: input.hole_orbital_1based,
            orbital_count,
        });
    }
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_count,
        orbital_count,
    )?;

    for &kappa in input.kappas {
        abs_kappa_i32(kappa)?;
    }
    validate_finite_vector("radius", input.radii)?;
    validate_finite_vector("density_4pi", input.density_4pi)?;
    validate_finite_vector("initial_large_component", input.initial_large_component)?;
    validate_finite_vector("initial_small_component", input.initial_small_component)?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("orbital_energy", input.orbital_energies)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    Ok(())
}
